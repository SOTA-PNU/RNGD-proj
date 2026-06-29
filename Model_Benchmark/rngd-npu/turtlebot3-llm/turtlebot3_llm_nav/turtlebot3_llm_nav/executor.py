"""LLM이 만든 파이썬 코드를 추출해 제한된 샌드박스에서 실행하고 `plan` 함수를 꺼냅니다.

robot-sim/executor.py 를 ROS2 사람찾기 노드용으로 그대로 옮긴 것입니다.
'로봇이 코드를 받아 자기 자신에게 적용한다'의 핵심으로, LLM 응답에서 코드 블록을
뽑아 exec 하고, 그 안에서 정의된 plan(state) 컨트롤러를 매 제어주기마다 호출합니다.

기본 차단(완전한 보안 경계는 아님)
  * builtins 화이트리스트 + import 는 math 만 허용.
  * AST 검사로 던더(__...) 속성/이름·위험 내장함수 접근을 거부 — `().__class__.__subclasses__()`
    류의 우회와 `__globals__`/`__builtins__` 오염을 막습니다.
  * exec(빌드)와 plan() 호출 모두에 시간제한 — 모듈레벨·함수레벨 무한루프 방어.
  * 빌드마다 builtins 사전을 새로 복사해 플랜 간 상태 오염 차단.
주의: in-process exec 는 본질적으로 강한 격리가 아닙니다. **신뢰 가능한 로컬 연구용**으로,
      우리 시스템이 생성한 컨트롤러를 돌리는 데만 쓰세요. 진짜 격리가 필요하면 별도
      프로세스(seccomp/nsjail)나 WASM 런타임으로 분리해야 합니다.
"""
from __future__ import annotations

import ast
import builtins as _builtins
import math
import re
import threading
from typing import Callable, Tuple

# plan() 안에서 쓸 수 있는 안전한 내장 함수만 노출합니다.
_ALLOWED = (
    "abs min max round range len sum float int bool enumerate zip map filter "
    "sorted list tuple dict set pow isinstance any all reversed divmod str repr "
    "True False None print"
).split()
_SAFE_BUILTINS = {k: getattr(_builtins, k) for k in _ALLOWED if hasattr(_builtins, k)}


def _safe_import(name, *a, **k):
    """math 만 import 허용(결정성 유지). 그 외 모듈은 샌드박스에서 막습니다."""
    if name == "math":
        return math
    raise ImportError(f"샌드박스에서는 '{name}' 모듈을 쓸 수 없습니다 (math 만 허용)")


_SAFE_BUILTINS["__import__"] = _safe_import

# 이름으로 부르면 위험한 내장(화이트리스트에 없어 런타임 NameError 지만, AST 단계에서 명확히 차단).
_BANNED_NAMES = frozenset({
    "eval", "exec", "compile", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "open", "input", "breakpoint", "memoryview", "__import__",
})


def _validate_ast(code: str) -> "ast.AST":
    """exec 전에 AST 를 훑어 던더(__...) 속성/이름과 위험 내장함수 접근을 거부합니다.
    `().__class__.__base__.__subclasses__()` 류 우회와 `__globals__`/`__builtins__` 오염을 막습니다.
    (완전한 보안 경계는 아니지만, 알려진 in-process 탈출 수법을 차단합니다.)"""
    tree = ast.parse(code, "<llm_plan>", "exec")   # SyntaxError 면 여기서 발생
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"샌드박스 금지: 던더 속성 접근 '{node.attr}'")
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise ValueError(f"샌드박스 금지: 던더 이름 '{node.id}'")
            if node.id in _BANNED_NAMES:
                raise ValueError(f"샌드박스 금지: 위험 함수 '{node.id}'")
    return tree


# ── 코드 추출 ─────────────────────────────────────────────────────
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_DEF_PLAN = re.compile(r"(?m)^[ \t]*def\s+plan\s*\(")   # 진짜 함수 정의 줄만 잡는 앵커


def extract_code(text: str) -> str:
    """LLM 응답에서 파이썬 코드만 골라냅니다.
    1) ```python ... ``` 펜스가 있으면 그 안을(여러 개면 plan 을 정의한 마지막 블록을) 사용.
    2) 펜스가 없으면 'def plan' 이 보이는 본문 전체를 코드로 간주."""
    blocks = _FENCE.findall(text or "")
    if blocks:
        for b in reversed(blocks):
            if _DEF_PLAN.search(b):
                return b.strip()
        return blocks[-1].strip()
    # 펜스 없는 경우: '진짜' def plan( 정의 줄부터 끝까지를 코드로(산문 속 'def plan' 오인 방지)
    m = _DEF_PLAN.search(text or "")
    if m:
        return text[m.start():].strip()
    return (text or "").strip()


# ── 샌드박스 컴파일 ───────────────────────────────────────────────
def build_plan(code: str, timeout: float = 1.0) -> Tuple[Callable[[dict], object], str]:
    """코드를 검사·exec 해 plan(state) 함수를 돌려줍니다.
    AST 던더/위험이름 차단 → 컴파일 → 시간제한 exec(모듈레벨 무한루프 방어) → plan 추출.
    문법오류·금지구문·정의없음·exec타임아웃이면 예외(호출부가 잡아 수리 루프로 보냅니다)."""
    tree = _validate_ast(code)   # SyntaxError / ValueError(던더·금지) 발생 가능

    def _compile_exec(_ignored):
        ns = {"__builtins__": dict(_SAFE_BUILTINS), "math": math}  # 빌드마다 새 사본(플랜 간 오염 차단)
        exec(compile(tree, "<llm_plan>", "exec"), ns)             # NameError 등 정의시 오류도 여기서
        return ns

    ns = call_with_timeout(_compile_exec, None, timeout)
    fn = ns.get("plan")
    if not callable(fn):
        raise ValueError("코드에 호출 가능한 plan(state) 함수가 없습니다")
    return fn, code


# ── 시간제한 호출 ─────────────────────────────────────────────────
class PlanTimeout(Exception):
    pass


def call_with_timeout(fn: Callable, arg, timeout: float = 0.5):
    """plan(state) 한 번 호출에 시간제한을 둡니다.
    메인스레드면 SIGALRM(가장 정확), 아니면 워치독 스레드로 join-timeout 합니다.
    (ROS2 노드의 타이머 콜백은 보통 메인스레드가 아니므로 스레드 워치독 경로를 탑니다.)"""
    if threading.current_thread() is threading.main_thread():
        import signal

        def _handler(signum, frame):
            raise PlanTimeout("plan() 실행 시간 초과")

        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            return fn(arg)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old)
    else:
        box = {}

        def _run():
            try:
                box["ok"] = fn(arg)
            except BaseException as e:  # noqa: BLE001  - 예외도 그대로 전달
                box["err"] = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise PlanTimeout("plan() 실행 시간 초과")
        if "err" in box:
            raise box["err"]
        return box.get("ok")


# ── 액션 정규화 ───────────────────────────────────────────────────
def normalize_action(action, v_max: float, w_max: float, heading: float, dt: float) -> Tuple[float, float]:
    """plan() 이 돌려준 값을 (v, w) 유니사이클 입력으로 변환합니다.
    허용 형태: {'v','w'} | {'vx','vy'}(홀로노믹) | (v, w) 튜플/리스트.
    TurtleBot3 은 차동구동이라 보통 {'v','w'} 를 씁니다."""
    v = w = 0.0
    if isinstance(action, dict):
        if "v" in action or "w" in action:
            v = float(action.get("v", 0.0) or 0.0)
            w = float(action.get("w", 0.0) or 0.0)
        elif "vx" in action or "vy" in action:
            vx = float(action.get("vx", 0.0) or 0.0)
            vy = float(action.get("vy", 0.0) or 0.0)
            target = math.atan2(vy, vx)
            aerr = math.atan2(math.sin(target - heading), math.cos(target - heading))
            speed = math.hypot(vx, vy)
            v = speed * max(0.0, math.cos(aerr))          # 정면 성분만 전진
            w = aerr / dt if dt > 0 else aerr              # 목표 방향으로 회전
        else:
            raise ValueError("plan() 반환 dict 에 'v'/'w' 또는 'vx'/'vy' 가 없습니다")
    elif isinstance(action, (tuple, list)) and len(action) >= 2:
        v, w = float(action[0]), float(action[1])
    else:
        raise ValueError(f"plan() 반환형을 해석할 수 없습니다: {type(action).__name__}")

    if not (math.isfinite(v) and math.isfinite(w)):
        raise ValueError("plan() 이 NaN/inf 를 반환했습니다")
    v = _clamp(v, -v_max, v_max)
    w = _clamp(w, -w_max, w_max)
    return v, w


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x
