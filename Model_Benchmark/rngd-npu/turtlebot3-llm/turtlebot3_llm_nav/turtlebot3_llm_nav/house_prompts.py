"""집 안 물건 확인 후 복귀 미션의 LLM 프롬프트 — plan(state) 계약(ROS2/Gazebo 판).

헤드리스 검증판(`rngd-npu/robot-sim/prompts.py` 의 HOUSE_* )과 **동일한 계약**을 씁니다.
전역 플래너(또는 파라미터로 준 경로 waypoints)가 방을 도는 경로를 주고, LLM 은 그 경로를
따라가며 카메라로 물건을 스캔→판정하고 현관(home)으로 복귀하는 로컬 컨트롤러를 코딩합니다.

  SYSTEM   : state 스키마 + 출력계약(매 틱 {'v','w'}, 끝나면 {'done':True,'present':bool}).
  SCAFFOLD : 바로 도는 짧은 예시 컨트롤러(경로 추종 + 스캔 + 복귀 + 판정).
  initial_user(): 첫 코드 요청.  repair_user(): 실패 유형별 self-debug 유도(FAILURE= 형식).
"""
from __future__ import annotations

import json

# ── 바로 도는 예시 컨트롤러(헤드리스 HOUSE_SCAFFOLD 와 동일) ───────
HOUSE_SCAFFOLD = '''import math

def plan(state):
    mem = state['memory']
    wps = state['waypoints']; home = state['home']; obj = state['objective']
    x, y = state['pos']; th = state['heading']
    v_max = state['v_max']; w_max = state['w_max']
    lidar = state['lidar']; angles = state['lidar_angles']; R = state['max_range']
    if 'i' not in mem:
        mem['i'] = 0; mem['found'] = False; mem['phase'] = 'search'
        mem['spin'] = 0; mem['scanned'] = []
    # 1) scan: a detection matching ALL objective items means the object is here
    for d in state['scan']:
        f = d.get('features', {})
        if obj and all(str(f.get(k)) == str(val) for k, val in obj.items()):
            mem['found'] = True
    # 2) pick the point to drive to: next route waypoint while searching, else home
    if mem['phase'] == 'search' and mem['i'] >= len(wps):
        mem['phase'] = 'home'
    tx, ty = home if mem['phase'] == 'home' else wps[mem['i']]
    dist = math.hypot(tx - x, ty - y)
    # 3) finished: declare the verdict only after returning home
    if mem['phase'] == 'home' and dist <= state['goal_tol']:
        return {'v': 0.0, 'w': 0.0, 'done': True, 'present': mem['found']}
    # 4) reached a search waypoint: rotate a full turn to scan a NEW area, then advance
    if mem['phase'] == 'search' and dist <= 0.25:
        new_area = all(math.hypot(x - sx, y - sy) > 1.0 for sx, sy in mem['scanned'])
        if new_area and mem['spin'] < 40:
            mem['spin'] += 1
            return {'v': 0.0, 'w': 0.8 * w_max}
        if new_area:
            mem['scanned'].append((x, y))
        mem['spin'] = 0; mem['i'] += 1
        return {'v': 0.0, 'w': 0.0}
    # 5) pure-pursuit: turn toward the target point; slow down when not aligned or wall ahead
    desired = math.atan2(ty - y, tx - x)
    aerr = math.atan2(math.sin(desired - th), math.cos(desired - th))
    front = min([d for d, a in zip(lidar, angles) if abs(a) < 0.4] or [R])
    w = 2.0 * aerr
    v = v_max * max(0.0, math.cos(aerr)) ** 2
    if front < 0.4:
        v *= 0.3
    return {'v': max(0.0, min(v_max, v)), 'w': max(-w_max, min(w_max, w))}'''


HOUSE_SYSTEM = f"""You are the onboard controller of a TurtleBot3 home robot with a LiDAR and a forward CAMERA.
MISSION: search the house for a specific OBJECT, decide whether it is present, and RETURN to home,
then report your verdict. A global planner gives you a route of waypoints through the rooms; you write
ONE Python function plan(state), run every control tick (~10 Hz), that FOLLOWS the route, scans with the
camera, avoids walls, comes back home, and declares the result.

state is a dict:
  pos=(x,y) m; heading=theta rad (0=+x, CCW+); v_max,w_max,dt; robot_radius,goal_tol; bounds=(W,H)
  lidar=list of distances (m); lidar_angles=robot-frame ray angles (0=front, CCW+); max_range
  objective = features the target object MUST match, e.g. {{'label':'cup','color':'red'}}
  scan = list of objects the camera sees NOW (forward cone only; occluded ones excluded), each:
         {{'bearing': rad (+left,-right), 'distance': m, 'features': {{'label':..,'color':..}}, 'conf':0..1}}
  waypoints = list of [x,y] route points to visit IN ORDER (your search path through the rooms)
  home = [x,y] entrance to return to at the end
  memory = a dict that PERSISTS across calls (keep your waypoint index, a 'found' flag, phase, etc.)

Return {{'v':linear,'w':angular}} each tick (v>0 forward, w>0 left; clamped to limits). When the mission
is COMPLETE (you are back home and have decided), return {{'done':True,'present':<bool>}}.
'present' must be True only if you actually SAW the target during the search.

RULES:
  - Walk the waypoints in order (track the index in memory); when all are visited, drive to home.
  - SCAN every tick: an object counts as the target only if its 'features' match ALL of objective.
    BEWARE DECOYS that share just one feature (same color, different label) — do NOT count them.
  - Only declare done after within goal_tol of home. Report present=True iff you saw the target.
  - Avoid walls: if a front lidar reading is small, slow/stop and steer toward open space.
Here is a complete working controller — return it as-is or improve it (keep it correct):
```python
{HOUSE_SCAFFOLD}
```
Reply with ONE ```python code block defining plan(state). No prose."""


def initial_user(objective: dict, n_waypoints: int, extra: str = "") -> str:
    oj = json.dumps(objective, ensure_ascii=False)
    msg = (f"TASK: search the house for the object matching objective = {oj}, then return home and "
           f"report present/absent. The route has {n_waypoints} waypoints.\n"
           f"Return the scaffold above or an improved version. Keep plan(state) correct.\n")
    if extra:
        msg += "\n" + extra
    return msg


_REPAIR_HINTS = {
    "missed_object": (
        "FAILURE=missed_object. You reported the object ABSENT, but it WAS in the house and reachable "
        "along the route. Visit EVERY waypoint and SCAN at each new area (rotate a full turn) before "
        "deciding; set found=True as soon as any detection matches ALL objective features."),
    "searched_too_little": (
        "FAILURE=searched_too_little. You declared a verdict WITHOUT actually searching the house. Do NOT "
        "guess: follow the waypoints to the end scanning each room; report present only after a detection "
        "matches ALL objective features, report absent only after visiting (nearly) all waypoints, then go home."),
    "false_report": (
        "FAILURE=false_report. You reported the object PRESENT, but it was NOT there — you were fooled by "
        "a DECOY matching only ONE feature (same color, different label). Count an object as the target ONLY "
        "if its 'features' match ALL of objective; ignore partial matches."),
    "not_home": (
        "FAILURE=not_home. You declared done but the robot was NOT back at home. Finish the search, THEN "
        "drive to state['home'] and only return {'done':True,...} once within goal_tol of home."),
    "no_report": (
        "FAILURE=no_report. The robot moved but NEVER declared the mission done. After visiting all "
        "waypoints and returning home you MUST return {'done':True,'present':<bool>}. Track your waypoint "
        "index and phase in memory so you make progress and eventually finish."),
    "stuck": (
        "FAILURE=stuck. The robot stopped advancing through the waypoints. Use memory to detect being stuck "
        "and steer toward open space (compare left vs right lidar) to get around the wall, then resume."),
    "collision": (
        "FAILURE=collision. The robot hit a wall. When the nearest FRONT lidar (|angle|<~0.5) is small, set "
        "v=0 and turn toward the side with more free space before continuing along the route."),
    "exception": (
        "FAILURE=exception while running plan(): {detail}. Make plan() always return a dict: guard empty "
        "scan/lidar lists and missing dict keys with .get, and avoid division by zero."),
    "timeout": (
        "FAILURE=timeout. plan() took too long. Keep it a few simple lines, O(n) over lidar/scan, return fast."),
    "no_valid_code": (
        "FAILURE=no_valid_code. Your last reply had no valid plan(state). Reply with EXACTLY ONE "
        "```python code block defining def plan(state): that returns a dict. No prose, keep it correct."),
}


def repair_user(reason: str, objective: dict, telemetry: dict | None = None,
                detail: str = "") -> str:
    hint = _REPAIR_HINTS.get(reason, _REPAIR_HINTS["no_valid_code"])
    if "{detail}" in hint:
        hint = hint.format(detail=detail or "unknown")
    oj = json.dumps(objective, ensure_ascii=False)
    tele = ""
    if telemetry:
        tele = "\nCONTEXT: " + json.dumps(telemetry, ensure_ascii=False)[:600]
    return (
        f"Your previous plan() did NOT complete the house mission "
        f"(search for objective={oj} and return home).\n{hint}{tele}\n\n"
        f"Return a corrected, COMPLETE plan(state) in ONE ```python block. No prose."
    )
