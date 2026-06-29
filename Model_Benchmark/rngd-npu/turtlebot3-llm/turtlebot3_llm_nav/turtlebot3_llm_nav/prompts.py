"""LLM 프롬프트 모음 — '특정 사람 찾아가기' task 의 plan(state) 계약.

프롬프트는 영어로 둡니다(코더 모델이 영어 명세에서 더 잘 동작). 작은 코더 모델은 긴
멀티라인 코드에서 출력이 망가지는 경향이 있으므로, **짧고 깨끗한 plan() 만** 요청합니다.
(메모리: short clean code only.)

  SYSTEM        : state 스키마 + 출력계약 + 안전규칙(짧게).
  SCAFFOLD      : 바로 돌아가는 짧은 예시 컨트롤러(첫 user 메시지에 함께 보냄).
  initial_user(): 첫 코드 요청.
  repair_user() : 실패 유형별로 메시지를 달리해 self-debug 를 유도.
"""
from __future__ import annotations

import json

# ── SYSTEM ────────────────────────────────────────────────────────
SYSTEM = """You are a controller-coding robot. You write a SHORT Python function
`plan(state)` that drives a TurtleBot3 (differential drive) toward ONE specific
PERSON seen by its camera. The robot runs your plan() every control tick (~10 Hz).

`state` is a dict:
  pos          : (x, y) robot world position [m]
  heading      : robot yaw [rad]
  v_max, w_max : speed limits (linear m/s, angular rad/s)
  dt           : control period [s]
  lidar        : list of distances [m] (downsampled, robot frame)
  lidar_angles : list of angles [rad] aligned with lidar (0 = front, + = left)
  camera       : list of PERSON detections in view, each:
                   {'bearing': angle[rad] (+left,-right), 'distance': m,
                    'features': {...}, 'conf': 0..1}
  target       : dict of features of the person to reach, e.g. {'shirt':'red'}
                 (match a detection whose 'features' contains all target items)
  memory       : a dict that PERSISTS across calls (use it for state/counters)

`plan(state)` MUST return {'v': linear, 'w': angular}.
  v>0 forward, w>0 turns left. The node clamps to [-v_max,v_max]/[-w_max,w_max].

GOAL: rotate/drive so the TARGET person is in front, then approach until close.
RULES:
  - Pick the camera detection whose features match `target`; ignore others (wrong people).
  - Turn toward its 'bearing' (steer w ~ +k*bearing). Slow down as 'distance' shrinks.
  - If no matching person is visible, rotate slowly to search (do NOT drive blind).
  - Avoid obstacles: if a lidar reading in front is small, slow or stop and turn away.
  - Keep the code SHORT (~15-25 lines). Only `import math`. No other imports, no I/O.
Reply with ONE ```python code block defining plan(state). No prose."""


# ── SCAFFOLD (짧은 예시) ─────────────────────────────────────────
SCAFFOLD = '''```python
import math

def plan(state):
    v_max = state['v_max']; w_max = state['w_max']
    lidar = state['lidar']; angles = state['lidar_angles']
    tgt = state.get('target', {}); mem = state['memory']
    # choose matching person with highest confidence
    best = None
    for d in state.get('camera', []):
        f = d.get('features', {})
        if all(str(f.get(k)) == str(v) for k, v in tgt.items()):
            if best is None or d['conf'] > best['conf']:
                best = d
    front = min([r for r, a in zip(lidar, angles) if abs(a) < 0.5] or [10.0])
    if best is None:                       # not seen -> search by turning
        return {'v': 0.0, 'w': 0.5 * w_max}
    b = best['bearing']
    w = max(-w_max, min(w_max, 1.6 * b))
    v = v_max * (0.4 + 0.6 * max(0.0, math.cos(b)))
    if front < 0.5:                        # obstacle close -> stop & turn
        v = 0.0
    elif best['distance'] < 0.7:           # arrived region -> creep
        v = 0.1 * v_max
    return {'v': v, 'w': w}
```'''


# ── 첫 요청 ──────────────────────────────────────────────────────
def initial_user(target: dict, extra: str = "") -> str:
    """첫 코드 생성 요청. 목표 사람 특징과 scaffold 를 함께 줍니다."""
    tj = json.dumps(target, ensure_ascii=False)
    msg = (
        f"TASK: drive the robot to the person matching target = {tj}.\n"
        f"Here is a working scaffold; you may return it or improve it, but keep it SHORT:\n\n"
        f"{SCAFFOLD}\n"
    )
    if extra:
        msg += "\n" + extra
    return msg


# ── 수리 요청(실패유형별) ────────────────────────────────────────
_REPAIR_HINTS = {
    "collision": (
        "FAILURE=collision. The robot hit/nearly hit an obstacle (a lidar reading "
        "in front got below the safety margin). Add stronger obstacle avoidance: when "
        "the minimum front lidar distance is small, set v=0 and turn toward the more "
        "open side before approaching the person."),
    "lost_target": (
        "FAILURE=lost_target. The matching person was NOT visible for many ticks, and "
        "your search did not re-acquire them. When 'camera' has no matching detection, "
        "rotate IN PLACE to search (v=0, |w| moderate); optionally remember the last "
        "seen bearing in memory and turn that way first."),
    "wrong_person": (
        "FAILURE=wrong_person. The robot approached a person whose features do NOT match "
        "target. Filter strictly: only pick a detection whose 'features' contains ALL "
        "target items; ignore all others, even if closer."),
    "stuck": (
        "FAILURE=stuck. The robot stopped making progress (position barely changed) "
        "without reaching the target. Break the deadlock: rotate to find a clear heading "
        "and use memory to detect being stuck and try a different turn direction."),
    "exception": (
        "FAILURE=exception while running plan(): {detail}. Fix the bug. Guard for empty "
        "lists (camera/lidar may be empty), missing dict keys, and division by zero."),
    "timeout": (
        "FAILURE=timeout. plan() took too long (no infinite loops / heavy work). Keep it "
        "a few simple lines, O(n) over lidar/camera, and return quickly."),
    "no_valid_code": (
        "FAILURE=no_valid_code. Your last reply did not contain a valid plan(state). "
        "Reply with EXACTLY ONE ```python code block defining `def plan(state):` that "
        "returns {'v':..,'w':..}. No prose, keep it SHORT."),
}


def repair_user(reason: str, target: dict, telemetry: dict | None = None,
                detail: str = "") -> str:
    """실패 유형(reason)에 맞춰 수리 프롬프트를 구성합니다.
    telemetry: 노드가 모은 현재 상황(가까운 lidar, 보이는 사람 특징 등)."""
    hint = _REPAIR_HINTS.get(reason, _REPAIR_HINTS["no_valid_code"])
    if "{detail}" in hint:
        hint = hint.format(detail=detail or "unknown")
    tj = json.dumps(target, ensure_ascii=False)
    tele = ""
    if telemetry:
        tele = "\nCONTEXT: " + json.dumps(telemetry, ensure_ascii=False)[:600]
    return (
        f"Your previous plan() did NOT complete the task (reach person target={tj}).\n"
        f"{hint}{tele}\n\n"
        f"Return a corrected, SHORT plan(state) in ONE ```python block. No prose."
    )
