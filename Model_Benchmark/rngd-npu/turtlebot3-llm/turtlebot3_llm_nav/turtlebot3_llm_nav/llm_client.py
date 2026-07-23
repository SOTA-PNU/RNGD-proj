"""LLM 클라이언트: 실제 RNGD NPU 서버(furiosa-llm serve, OpenAI 호환)에 붙거나,
서버가 없을 때 쓰는 mock 폴백을 제공합니다.

실서버 모드는 chat_app.py 와 똑같은 방식으로 붙습니다:
  base_url = http://127.0.0.1:<port>/v1,  api_key="dummy",
  model_id = client.models.list().data[0].id,
  chat.completions.create(stream=True, stream_options={"include_usage": True}).
스트리밍하며 TTFT(첫 토큰까지 시간)·TPS·정확한 completion_tokens(usage)를 측정합니다.
(참고: rngd-npu/chat/chat_app.py, rngd-npu/robot-sim/llm_client.py)

mock 모드는 openai 패키지 없이도 동작하며(시스템 python3 가능), 결정적인 사람찾기 컨트롤러
코드를 돌려줘 폐루프 자체를 서버 없이 검증할 수 있게 합니다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CallMetrics:
    """LLM 호출 한 번의 성능 지표(코드 생성/수리 1회)."""
    ttft_s: float = 0.0           # 첫 토큰까지 걸린 시간
    total_s: float = 0.0          # 전체 생성 시간
    completion_tokens: int = 0    # 생성 토큰 수(usage 정확값, 없으면 글자수 추정)
    tps: float = 0.0              # 초당 토큰
    ok: bool = True
    error: str = ""

    def finalize(self):
        gen = max(1e-6, self.total_s - self.ttft_s)
        if self.completion_tokens:
            self.tps = self.completion_tokens / gen
        return self


# ── 공통 인터페이스 ───────────────────────────────────────────────
class BaseClient:
    name = "base"

    def complete(self, system: str, user: str, history: Optional[List[dict]] = None,
                 temperature: float = 0.2, max_tokens: int = 1024):
        raise NotImplementedError


# ── 실제 NPU 서버 클라이언트 ──────────────────────────────────────
class NpuClient(BaseClient):
    """furiosa-llm serve(OpenAI 호환)에 붙는 클라이언트. 포트만 알면 됩니다."""

    def __init__(self, port: int, model_label: str = "", timeout: float = 600.0):
        self.port = int(port)
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self.name = model_label or f"npu:{self.port}"
        self.timeout = timeout
        self._client = None
        self._model_id = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI  # 지연 import — mock 모드는 openai 불필요
            self._client = OpenAI(base_url=self.base_url, api_key="dummy", timeout=self.timeout)
        if self._model_id is None:
            self._model_id = self._client.models.list().data[0].id
        return self._client, self._model_id

    def ping(self) -> bool:
        try:
            import httpx
            return httpx.get(self.base_url + "/models", timeout=3.0).status_code == 200
        except Exception:
            return False

    def complete(self, system, user, history=None, temperature=0.2, max_tokens=1024):
        client, model_id = self._ensure()
        msgs = [{"role": "system", "content": system}]
        msgs += list(history or [])
        msgs.append({"role": "user", "content": user})

        m = CallMetrics()
        t0 = time.time()
        first = None
        chars = 0
        body = ""
        try:
            stream = client.chat.completions.create(
                model=model_id, messages=msgs, temperature=temperature,
                max_tokens=int(max_tokens), stream=True,
                stream_options={"include_usage": True})
            for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if not chunk.choices:
                    if usage is not None:
                        m.completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                    continue
                delta = chunk.choices[0].delta
                piece = getattr(delta, "content", None) or ""
                # 추론(reasoning) 토큰은 코드가 아니므로 본문엔 안 넣되 시간엔 반영
                reason = getattr(delta, "reasoning", None) or ""
                if (piece or reason) and first is None:
                    first = time.time()
                if piece:
                    body += piece
                    chars += len(piece)
            m.total_s = time.time() - t0
            m.ttft_s = (first - t0) if first else m.total_s
            if not m.completion_tokens:
                m.completion_tokens = max(1, chars // 4)   # usage 없으면 글자수로 추정
            m.finalize()
            return body, m
        except Exception as e:
            m.ok = False
            m.error = str(e)
            m.total_s = time.time() - t0
            return "", m


# ── mock 클라이언트(서버 불필요) ──────────────────────────────────
class MockClient(BaseClient):
    """서버 없이 노드를 검증하기 위한 가짜 LLM.
      mode='good'  : 처음부터 동작하는 '사람 추종' 컨트롤러를 돌려줍니다.
      mode='buggy' : 첫 응답은 'bearing 부호 반대로 도는' 버그 코드를, 수리 요청부터는
                     정상 코드를 돌려줘 self-debug 루프(코딩 성능 검증)를 서버 없이 재현합니다.
    호출 지연/토큰수는 그럴듯하게 흉내 냅니다(결정적)."""

    def __init__(self, mode: str = "good"):
        self.mode = mode
        self.name = f"mock:{mode}"
        self._calls = 0

    def complete(self, system, user, history=None, temperature=0.2, max_tokens=1024):
        is_repair = "FAILURE" in user or "repair" in user.lower()
        is_house = "search the house" in (system or "")
        if is_house:
            code = _HOUSE_BUGGY if (self.mode == "buggy" and not is_repair) else _HOUSE_GOOD
        elif self.mode == "buggy" and not is_repair:
            code = _BUGGY_CONTROLLER
        else:
            code = _GOOD_CONTROLLER
        self._calls += 1
        text = "```python\n" + code + "\n```"
        m = CallMetrics(ttft_s=0.05, total_s=0.05 + len(text) / 4000.0,
                        completion_tokens=max(1, len(text) // 4))
        m.finalize()
        return text, m


# 동작하는 사람추종 컨트롤러: 목표특징과 맞는 검출을 골라 그쪽으로 회전+전진,
# 가까운 lidar 광선은 피함, 안 보이면 천천히 회전하며 탐색.
_GOOD_CONTROLLER = '''\
import math

def plan(state):
    v_max = state['v_max']; w_max = state['w_max']
    lidar = state['lidar']; angles = state['lidar_angles']
    tgt = state.get('target', {})
    mem = state['memory']
    # 목표특징과 맞는 사람 검출 중 가장 신뢰높은 것 선택
    best = None
    for d in state.get('camera', []):
        f = d.get('features', {})
        ok = all(str(f.get(k)) == str(v) for k, v in tgt.items())
        if ok and (best is None or d.get('conf', 0) > best.get('conf', 0)):
            best = d
    # 가까운 정면 장애물 회피용 최소거리
    front = min([r for r, a in zip(lidar, angles) if abs(a) < 0.5] or [10.0])
    if best is None:                       # 안 보이면 제자리 탐색 회전
        return {'v': 0.0, 'w': 0.5 * w_max}
    b = best['bearing']
    w = max(-w_max, min(w_max, 1.6 * b))   # 사람 쪽으로 정렬
    v = v_max * (0.4 + 0.6 * max(0.0, math.cos(b)))
    if front < 0.5:                        # 끼이면 정지 회전
        v = 0.0
    elif best.get('distance', 9.0) < 0.7:  # 도착 근처면 감속
        v = 0.1 * v_max
    return {'v': v, 'w': w}
'''

# 버그 컨트롤러: bearing 부호를 뒤집어 사람 '반대로' 돈다 → 절대 도달 못 함(수리 대상).
_BUGGY_CONTROLLER = '''\
import math

def plan(state):
    v_max = state['v_max']; w_max = state['w_max']
    tgt = state.get('target', {})
    best = None
    for d in state.get('camera', []):
        f = d.get('features', {})
        if all(str(f.get(k)) == str(v) for k, v in tgt.items()):
            best = d
    if best is None:
        return {'v': 0.0, 'w': 0.5 * w_max}
    w = max(-w_max, min(w_max, -1.6 * best['bearing']))   # BUG: 부호 반대
    return {'v': 0.5 * v_max, 'w': w}
'''


# ── 집 미션 mock: 정상(경로추종+스캔+복귀+판정) / 버그(검색없이 '없음' 단정) ──
from turtlebot3_llm_nav.house_prompts import HOUSE_SCAFFOLD as _HOUSE_GOOD   # noqa: E402

_HOUSE_BUGGY = '''import math

def plan(state):
    # BUG: reports the object ABSENT immediately, without searching the house at all.
    return {'v': 0.0, 'w': 0.0, 'done': True, 'present': False}'''


def make_client(mock: Optional[str] = None, port: Optional[int] = None,
                model_label: str = "") -> BaseClient:
    """파라미터에 따라 적절한 클라이언트를 만듭니다.
       mock 이 주어지면 MockClient, 아니면 NpuClient(port)."""
    if mock:
        return MockClient(mode=mock)
    if port is None:
        raise ValueError("실서버 모드에는 port 가 필요합니다 (llm_port 파라미터)")
    return NpuClient(port=port, model_label=model_label)
