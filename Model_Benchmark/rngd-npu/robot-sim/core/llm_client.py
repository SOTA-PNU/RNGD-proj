"""LLM 클라이언트: 실제 RNGD NPU 서버(furiosa-llm serve, OpenAI 호환)에 붙거나,
서버가 없을 때 쓰는 mock 폴백을 제공합니다.

실서버 모드는 chat_app.py 와 똑같은 방식으로 붙습니다:
  base_url = http://127.0.0.1:<port>/v1,  api_key="dummy",
  model_id = client.models.list().data[0].id,
  chat.completions.create(stream=True, stream_options={"include_usage": True}).
스트리밍하며 TTFT(첫 토큰까지 시간)·TPS·정확한 completion_tokens(usage)를 측정합니다.
(참고: chat_app.py:684-733 _client/_stream_reply, npu_metrics.py)

mock 모드는 openai 패키지 없이도 동작하며(시스템 python3 가능), 결정적인 컨트롤러 코드를
돌려줘 시뮬레이터 폐루프 자체를 서버 없이 검증할 수 있게 합니다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
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
    """furiosa-llm serve(OpenAI 호환)에 붙는 클라이언트."""

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
                # 추론(reasoning) 토큰은 코드가 아니므로 본문에는 넣지 않되 시간엔 반영
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
    """서버 없이 시뮬레이터를 검증하기 위한 가짜 LLM.
      mode='good'  : 처음부터 동작하는 반응형 컨트롤러를 돌려줍니다.
      mode='buggy' : 첫 응답은 '목표 반대로 도는' 부호버그 코드를, 수리 요청부터는 정상 코드를
                     돌려줘 self-debug 루프(코딩 성능 검증)를 서버 없이 재현합니다.
    호출 지연/토큰수는 그럴듯하게 흉내 냅니다(결정적)."""

    def __init__(self, mode: str = "good"):
        self.mode = mode
        self.name = f"mock:{mode}"
        self._calls = 0

    def complete(self, system, user, history=None, temperature=0.2, max_tokens=1024):
        # (2-LLM) 로봇 두뇌(플래너) 요청이면 코드가 아니라 '자연어 지시'를 돌려줍니다.
        if "on-board PLANNER" in (system or ""):
            self._calls += 1
            txt = _brain_repair_text() if "did NOT complete the task" in user else _brain_initial_text(user)
            m = CallMetrics(ttft_s=0.04, total_s=0.04 + len(txt) / 3000.0,
                            completion_tokens=max(1, len(txt) // 4))
            m.finalize()
            return txt, m
        is_repair = ("did NOT get the robot to the goal" in user
                     or "did NOT reach the correct target" in user
                     or "did NOT complete the house mission" in user
                     or "fix the controller" in user.lower())   # 두뇌가 보낸 자연어 수리지시도 '수리'로 인식
        is_house = "explore the house" in (system or "")
        is_vision = ("SPECIFIC person" in (system or "")) or ("TARGET person features" in user)
        if is_house:
            code = _HOUSE_BUGGY if (self.mode == "buggy" and not is_repair) else _HOUSE_GOOD
        elif is_vision:
            code = _VISION_BUGGY if (self.mode == "buggy" and not is_repair) else _VISION_GOOD
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


# 동작하는 반응형 컨트롤러: 목표추종 + LiDAR 반발 + 정체 시 벽타기 탈출.
_GOOD_CONTROLLER = '''\
import math

def plan(state):
    x, y = state['pos']
    gx, gy = state['goal']
    th = state['heading']
    lidar = state['lidar']
    angles = state['lidar_angles']
    R = state['max_range']
    v_max = state['v_max']
    w_max = state['w_max']
    mem = state['memory']

    # 목표 방향 오차(각도 wrap)
    desired = math.atan2(gy - y, gx - x)
    aerr = math.atan2(math.sin(desired - th), math.cos(desired - th))

    # 장애물 반발: 가까운 광선일수록 그 반대편으로 더 세게 조향
    push = 0.0
    min_front = R
    for d, a in zip(lidar, angles):
        if d < R * 0.95:
            weight = (R - d) / R
            push -= math.sin(a) * weight * 1.8
            if abs(a) < 0.6:
                min_front = min(min_front, d)

    # 정체 감지: 목표까지 거리가 한동안 줄지 않으면 벽타기 모드
    dist = math.hypot(gx - x, gy - y)
    best = mem.get('best', 1e9)
    since = mem.get('since', 0)
    if dist < best - 0.05:
        mem['best'] = dist
        mem['since'] = 0
    else:
        mem['since'] = since + 1

    w = 1.5 * aerr + push
    v = v_max * (0.30 + 0.70 * min(1.0, min_front / (R * 0.6)))
    if min_front < 0.7:                 # 거의 끼임 → v=0 제자리 회전(위치 불변이라 재충돌 없음)
        v = 0.0
    elif min_front < 1.2:               # 가까우면 살살
        v = min(v, 0.3 * v_max)

    # 정면이 막혔는데 좌우 반발이 비슷하면(대칭 지역최소: 정면 장애물) 더 트인 쪽으로 튼다
    left = sum(d for d, a in zip(lidar, angles) if 0.1 < a < 1.7)
    right = sum(d for d, a in zip(lidar, angles) if -1.7 < a < -0.1)
    if min_front < R * 0.5 and abs(push) < 0.4:
        w += (0.9 if left >= right else -0.9) * w_max

    if mem.get('since', 0) > 25 and min_front < R * 0.9:
        # 막혀서 못 가면: 한쪽으로 벽을 끼고(좌회전) 천천히 전진해 빠져나간다
        w = 0.9 * w_max
        v = 0.45 * v_max
        if mem['since'] > 130:          # 충분히 돌았으면 다시 목표추종 시도
            mem['since'] = 0
            mem['best'] = 1e9

    w = max(-w_max, min(w_max, w))
    v = max(0.0, min(v_max, v))
    return {'v': v, 'w': w}
'''

# 부호버그 컨트롤러: aerr 부호를 뒤집어 목표 '반대로' 돈다 → 절대 도달 못 함(수리 대상).
_BUGGY_CONTROLLER = '''\
import math

def plan(state):
    x, y = state['pos']
    gx, gy = state['goal']
    th = state['heading']
    v_max = state['v_max']
    w_max = state['w_max']
    desired = math.atan2(gy - y, gx - x)
    aerr = math.atan2(math.sin(desired - th), math.cos(desired - th))
    w = -1.5 * aerr            # BUG: 부호 반대 → 목표에서 멀어지는 쪽으로 회전
    w = max(-w_max, min(w_max, w))
    return {'v': 0.6 * v_max, 'w': w}
'''

# ── 카메라 사람찾기: 정상 컨트롤러(전체 특징 일치 식별 + 탐색 + 회피) ──
_VISION_GOOD = '''\
import math

def plan(state):
    cam = state['camera']
    tgt = state['target']
    mem = state['memory']
    v_max = state['v_max']
    w_max = state['w_max']
    R = state['max_range']
    min_front = R
    for d, a in zip(state['lidar'], state['lidar_angles']):
        if abs(a) < 0.6:
            min_front = min(min_front, d)
    # target 의 '모든' 특징이 일치하는 사람만 진짜 목표(옷색만 같은 decoy 는 무시)
    match = None
    for p in cam:
        f = p.get('features', {})
        if all(f.get(k) == v for k, v in tgt.items()):
            match = p
            break
    if match is not None:
        w = 2.0 * match['bearing']
        v = 0.6 * v_max if min_front > 1.2 else 0.15 * v_max
        if min_front < 1.0:                       # 정면 막힘 → 회피 회전
            w += (0.8 if match['bearing'] >= 0 else -0.8) * w_max
        return {'v': max(0.0, min(v_max, v)), 'w': max(-w_max, min(w_max, w))}
    # 못 찾음 → 한 방향으로 스윕하며 전진 탐색, 막히면 회전
    side = mem.get('side') or 1
    mem['side'] = side
    if min_front < 1.2:
        return {'v': 0.0, 'w': side * 0.8 * w_max}
    return {'v': 0.5 * v_max, 'w': side * 0.25 * w_max}
'''

# ── 카메라 사람찾기: 순진한 버그 컨트롤러(옷색만 보고 가까운 빨강 decoy 로 감 → wrong_person) ──
_VISION_BUGGY = '''\
import math

def plan(state):
    cam = state['camera']
    tgt = state['target']
    if not cam:
        return {'v': 0.0, 'w': 0.6 * state['w_max']}
    # BUG: 보조 특징(모자) 무시하고 옷색만 보고 가장 정면의 빨강에게 직진 → decoy 에게 도착
    reds = [p for p in cam if p.get('features', {}).get('shirt') == tgt.get('shirt')]
    p = (reds or cam)[0]
    return {'v': 0.6 * state['v_max'], 'w': 1.8 * p['bearing']}
'''


# ── 집 미션: 정상 컨트롤러(웨이포인트 추종 + 스캔 + 복귀 + 판정) ──
# prompts 의 HOUSE_SCAFFOLD 와 동일한 '바로 도는' 컨트롤러를 mock good 으로 재사용합니다.
from prompts import HOUSE_SCAFFOLD as _HOUSE_GOOD   # noqa: E402

# ── 집 미션: 버그 컨트롤러(검색 없이 '없음'이라 즉시 단정 → missed_object → 수리 대상) ──
_HOUSE_BUGGY = '''import math

def plan(state):
    # BUG: reports the object ABSENT immediately, without searching the house at all.
    return {'v': 0.0, 'w': 0.0, 'done': True, 'present': False}'''


# ── (2-LLM) 로봇 두뇌 mock 이 코더에게 보내는 '자연어 지시' 텍스트 ──
def _brain_initial_text(user: str) -> str:
    if "explore the house" in user:
        return ("Explore the whole house on your own using the LiDAR — follow the walls so you pass through "
                "doorways and cover every room. Watch the camera the whole time and remember if you ever see "
                "the target object (it must match BOTH its label AND colour — ignore look-alike decoys that "
                "share only one). When you have searched enough, drive back to home and report whether the "
                "object was present.")
    if "target person" in user:
        return ("Use the camera to find the person who matches ALL the target features; ignore look-alikes "
                "that match only one. Turn toward the matching person and drive to them, slowing near walls "
                "sensed by the LiDAR. If nobody matching is visible, rotate in place to search.")
    return ("Drive toward the goal point, but watch the LiDAR: when a wall is close ahead, slow down and "
            "steer toward whichever side has more open space. If you stop making progress (a U-shaped wall), "
            "follow the wall around it, then head for the goal again.")


def _brain_repair_text() -> str:
    return ("Fix the controller. It hit walls or failed to finish the task. Make it stop and turn toward the "
            "more open side whenever a wall is close ahead, keep exploring and scanning until it has covered "
            "the area or actually seen the target, and be sure it returns home and reports before stopping.")


def make_client(mock: Optional[str] = None, port: Optional[int] = None,
                model_label: str = "") -> BaseClient:
    """CLI 옵션에 따라 적절한 클라이언트를 만듭니다.
       mock 이 주어지면 MockClient, 아니면 NpuClient(port)."""
    if mock:
        return MockClient(mode=mock)
    if port is None:
        raise ValueError("실서버 모드에는 port 가 필요합니다 (--port 또는 --model)")
    return NpuClient(port=port, model_label=model_label)
