"""LLM에게 보낼 프롬프트 모음.

로봇은 '현재 위치 + 목표 위치(+ 움직이며 얻는 LiDAR)'만 알고, LLM에게 자기를 움직일
컨트롤러 코드 plan(state) 를 짜 달라고 부탁합니다. 실패하면 무슨 일이 있었는지 적어
'고쳐 달라'고 다시 요청합니다(자가수리 = 코딩 성능의 핵심 검증).

프롬프트 설계 메모(실측 반영, 2026-06-18)
  * 작은 coder 모델(예: 7B)은 'def plan(state): ...' 처럼 말줄임표로 끝나는 스텁을 보여주면
    그걸 흉내 내 빈 함수만 쓰고 멈춥니다. 그래서 스펙은 산문/불릿으로 주고, **빌드 가능한
    완전한 최소 예시(직진 컨트롤러)** 를 스캐폴드로 제공해 그걸 '개선'하게 합니다. 회피·지역최소
    탈출·디버깅은 여전히 모델 몫이라 코딩 성능 측정은 그대로 유효합니다.
  * 프롬프트 본문은 코드 생성 신뢰도를 위해 영어로 둡니다. 사용자 로그/문서는 한국어 존댓말.
"""
from __future__ import annotations

import math
from typing import List

# 빌드 가능한 최소 예시(스캐폴드): 목표로 직진하되 장애물은 무시 → 모델이 회피를 더해야 함.
SCAFFOLD = """import math

def plan(state):
    x, y = state['pos']
    gx, gy = state['goal']
    desired = math.atan2(gy - y, gx - x)
    err = math.atan2(math.sin(desired - state['heading']), math.cos(desired - state['heading']))
    return {'v': 0.6 * state['v_max'], 'w': 2.0 * err}"""

SYSTEM = f"""You are the motion-planning function of a mobile robot, like an Autoware planning node.
Write ONE SHORT Python function plan(state) that the robot runs every control cycle to steer toward a
goal while avoiding obstacles it senses with a LiDAR.

state is a dict:
  state['pos']=(x,y) meters; state['heading']=theta rad (0=+x, CCW+); state['goal']=(gx,gy) meters
  state['lidar']=list of distances (m) to the nearest obstacle on each ray, capped at state['max_range']
  state['lidar_angles']=list of each ray's angle in the ROBOT frame (rad), 0=straight ahead, CCW+ (same order)
  state['max_range'] (a reading == max_range means nothing detected on that ray)
  state['v_max'], state['w_max'], state['dt'], state['robot_radius'], state['goal_tol'], state['bounds']=(W,H)
  state['memory']=a dict that PERSISTS across calls (use it to keep a stuck counter etc.)

Return {{'v': linear_speed, 'w': angular_speed}}: v in [-v_max,v_max] (forward +), w in [-w_max,w_max] (CCW +).
The simulator clamps out-of-range values.

Starting point — this drives straight at the goal but IGNORES obstacles. Improve it:
```python
{SCAFFOLD}
```

Rules:
- Output ONLY one ```python code block, nothing else (no prose, no explanation).
- Keep the function SHORT and CORRECT (about 10-25 lines). Never leave it empty and never use "...".
- Use only the `math` module. Consistent 4-space indentation. No other imports, no __dunder__ access.
- For obstacles: read state['lidar'] / state['lidar_angles']; if a ray near the front (|angle| small) is
  short, slow down and steer toward whichever side (left vs right rays) has more free space.
"""


def _fmt_lidar(dists: List[float], angles: List[float]) -> str:
    """LiDAR 값을 '각도°:거리m' 짧은 목록으로(정면 0°부터). 프롬프트가 너무 길지 않게 요약."""
    return ", ".join(f"{math.degrees(a):+.0f}deg:{d:.2f}" for d, a in zip(dists, angles))


def _state_block(state: dict) -> str:
    x, y = state["pos"]
    gx, gy = state["goal"]
    dist = math.hypot(gx - x, gy - y)
    W, H = state["bounds"]
    return (
        f"World bounds: x in [0, {W:.1f}], y in [0, {H:.1f}] m. "
        f"robot_radius {state['robot_radius']:.2f} m, goal_tol {state['goal_tol']:.2f} m, "
        f"v_max {state['v_max']:.2f}, w_max {state['w_max']:.2f}, dt {state['dt']:.2f}.\n"
        f"Current position ({x:.2f}, {y:.2f}), heading {state['heading']:.3f} rad. "
        f"Goal ({gx:.2f}, {gy:.2f}), straight-line distance {dist:.2f} m.\n"
        f"LiDAR now ({len(state['lidar'])} rays, robot frame) [angle:dist] = "
        f"{_fmt_lidar(state['lidar'], state['lidar_angles'])}."
    )


def initial_user(state: dict) -> str:
    return (
        _state_block(state) + "\n\n"
        "Write plan() so the robot moves toward the goal and avoids obstacles in front of it. "
        "Keep it short. Output one ```python code block."
    )


# 실패 유형별로 '딱 필요한 만큼만' 보강 지시를 줍니다(한 번에 다 시키지 않아 짧고 깨끗한 코드 유도).
_HINT = {
    "collision": (
        "Add stronger obstacle avoidance: each cycle, look at the front lidar rays (|angle| < ~0.6 rad). "
        "If the smallest front distance is below ~1.5 m, reduce v and turn toward the side (compare the sum "
        "of left rays vs right rays) that has MORE free space. Keep it short."),
    "out_of_bounds": (
        "The robot left the arena. Treat the walls like obstacles using the lidar (a short reading in any "
        "direction is a wall), slow down and turn away before hitting them. Keep it short."),
    "stuck": (
        "The robot is trapped in a local minimum (a concave / U-shaped wall). Add a stuck-escape using "
        "state['memory']: keep a counter of cycles without getting closer to the goal; when it grows large, "
        "switch to WALL-FOLLOWING for a while (turn one consistent direction and creep forward, even if it "
        "temporarily increases the distance to the goal) to get around the obstacle, then resume. Keep it short."),
    "exception": (
        "Fix the crash so plan() always returns a dict {'v':..,'w':..} for every input (guard against division "
        "by zero and make sure every code path returns). Keep it short."),
}


# ════════════════════════════════════════════════════════════════════
#  카메라로 '특정 사람' 찾기 과제(TurtleBot3 waffle 카메라 시나리오의 헤드리스 판)
# ════════════════════════════════════════════════════════════════════
VISION_SCAFFOLD = """import math

def plan(state):
    cam = state['camera']
    if not cam:                                  # 아무도 안 보이면 제자리 회전(탐색)
        return {'v': 0.0, 'w': 0.6 * state['w_max']}
    p = cam[0]                                    # (주의) 그냥 첫 사람에게 감 — 맞는 사람인지 안 따짐
    return {'v': 0.5 * state['v_max'], 'w': 1.5 * p['bearing']}"""

VISION_SYSTEM = f"""You are the motion-planning function of a mobile robot (a TurtleBot3 with a forward CAMERA
and a LiDAR). Your task: drive the robot to a SPECIFIC person, described by `target` features and
identified from CAMERA detections, while avoiding obstacles. Write ONE short Python function plan(state).

state is a dict:
  state['pos']=(x,y), state['heading']=theta rad (0=+x axis, CCW positive)
  state['lidar'], state['lidar_angles'] — obstacle distances (m) + robot-frame ray angles (0 = straight ahead)
  state['v_max'], state['w_max'], state['dt']
  state['target'] — dict of features the goal person MUST match, e.g. {{'shirt':'red','cap':True}}
  state['camera'] — list of people the camera sees RIGHT NOW (forward cone only, ~59 deg; occluded/behind
      people are NOT listed). Each item:
        {{'bearing': robot-frame angle (rad, 0 = straight ahead, + = left),
         'distance': meters, 'features': dict e.g. {{'shirt':'red','cap':False}}, 'conf': 0..1 (lower when far)}}
  state['memory'] — a dict that PERSISTS across calls (use it for search direction, a stuck counter, etc.)
NOTE: you are NOT given the goal coordinates. You must FIND the target person using the camera.

Return {{'v': linear_speed, 'w': angular_speed}}: v in [-v_max,v_max] (forward +), w in [-w_max,w_max] (CCW +).

Starting point — drives toward the first detected person without checking if it's the right one (improve it):
```python
{VISION_SCAFFOLD}
```

Rules:
- Output ONLY one ```python code block. Keep it SHORT (about 10-25 lines), consistent 4-space indent,
  use only the `math` module, no __dunder__ access.
- IDENTIFY THE TARGET: only a detection whose features match ALL of state['target'] is the goal. Beware
  DECOYS that share just one feature (e.g. same shirt color but no cap) — do NOT stop at them.
- If NO matching person is currently visible, ROTATE to search (remember the search direction in memory)
  instead of standing still or chasing a wrong person.
- Avoid obstacles with the lidar: if a front ray (|angle| small) is short, slow down and steer away.
"""


def _fmt_camera(cam) -> str:
    if not cam:
        return "(no person in view)"
    return "; ".join(
        f"[bearing {math.degrees(d['bearing']):+.0f}deg, {d['distance']:.1f}m, {d['features']}, conf {d['conf']:.2f}]"
        for d in cam)


def vision_state_block(state: dict) -> str:
    x, y = state["pos"]
    return (
        f"Robot at ({x:.2f}, {y:.2f}), heading {state['heading']:.2f} rad. "
        f"v_max {state['v_max']:.2f}, w_max {state['w_max']:.2f}, dt {state['dt']:.2f}.\n"
        f"TARGET person features = {state.get('target')}.\n"
        f"Camera sees now: {_fmt_camera(state.get('camera'))}.\n"
        f"LiDAR (front rays) = "
        + ", ".join(f"{math.degrees(a):+.0f}:{d:.1f}" for d, a in
                    zip(state['lidar'], state['lidar_angles']) if abs(a) < 1.6))


def vision_initial_user(state: dict) -> str:
    return (
        vision_state_block(state) + "\n\n"
        "Write plan() so the robot reaches the TARGET person (matching ALL target features), "
        "searching by rotating when not visible and avoiding obstacles. Keep it short. "
        "Output one ```python code block."
    )


_VISION_HINT = {
    "wrong_person": (
        "You drove to the WRONG person — one who matches only SOME of the target features. Compare ALL "
        "keys in state['target'] against each detection's 'features'; only approach a FULL match. Treat "
        "look-alikes (same shirt, no cap) as obstacles/ignore. Keep it short."),
    "lost_target": (
        "The robot never reached the target — it likely never came into the camera's forward cone, or you "
        "stopped searching. When no FULL match is visible, keep rotating one consistent direction (store it "
        "in state['memory']) to sweep the room, and move forward to explore new areas. Keep it short."),
    "stuck": (
        "The robot stopped making progress. If the target is not visible, rotate to search and move to a "
        "new vantage point; if it is visible, head toward it while avoiding obstacles with the lidar. Keep it short."),
    "collision": (
        "The robot hit an obstacle. Check front lidar rays (|angle|<~0.6); if the nearest is below ~1.5 m, "
        "slow down and steer toward the freer side before continuing toward the person. Keep it short."),
    "exception": (
        "Fix the crash so plan() always returns {'v':..,'w':..} (guard empty state['camera'] and missing "
        "feature keys with .get). Keep it short."),
}


def vision_repair_user(reason: str, detail: str, state: dict, prev_code: str) -> str:
    head = "Your previous plan() did NOT reach the correct target person.\n" f"Outcome: {reason}.\n{detail}\n\n"
    if reason == "no_valid_code":
        ctx = "The code above failed to load (syntax/indentation error), so it never ran.\n\n"
        hint = "Rewrite cleanly with consistent 4-space indentation; make sure plan() returns a dict."
    else:
        ctx = vision_state_block(state) + "\n\n"
        hint = _VISION_HINT.get(reason, "Diagnose what went wrong and fix it.")
    return (
        head + ctx +
        "Here is your previous code:\n```python\n" + prev_code + "\n```\n\n" +
        hint + " Output the COMPLETE corrected function in exactly one ```python code block."
    )


def repair_user(reason: str, detail: str, state: dict, prev_code: str) -> str:
    """실패 상황을 설명하고 plan() 을 고쳐 달라는 프롬프트(self-debug 루프의 핵심).
    실패 유형에 맞는 보강 지시를 점진적으로 줍니다. 빌드 실패는 주행 상태와 무관하므로 상태를 뺍니다."""
    head = "Your previous plan() did NOT get the robot to the goal.\n" f"Outcome: {reason}.\n{detail}\n\n"
    if reason == "no_valid_code":
        ctx = "The code above failed to load (a syntax / indentation / structure error), so it never ran.\n\n"
        hint = "Rewrite it cleanly with consistent 4-space indentation and make sure plan() returns a dict."
    else:
        ctx = _state_block(state) + "\n\n"
        hint = _HINT.get(reason, "Diagnose what went wrong and fix it.")
    return (
        head + ctx +
        "Here is your previous code:\n```python\n" + prev_code + "\n```\n\n" +
        hint + " Output the COMPLETE corrected function in exactly one ```python code block."
    )


# ════════════════════════════════════════════════════════════════════
#  집 안 물건 확인 후 복귀 미션 (TurtleBot3 House — 헤드리스 판, **자율주행**)
#  미리 정한 경로는 없습니다. 로봇은 LiDAR·카메라만으로 스스로 집을 돌아다니며(자율 탐색)
#  물건을 스캔→판정하고 현관(home)으로 복귀하는 컨트롤러를 코딩합니다.
# ════════════════════════════════════════════════════════════════════
HOUSE_SCAFFOLD = '''import math

def plan(state):
    mem = state['memory']; obj = state['objective']; home = state['home']
    x, y = state['pos']; th = state['heading']
    v_max = state['v_max']; w_max = state['w_max']
    lidar = state['lidar']; ang = state['lidar_angles']; R = state['max_range']
    if 'phase' not in mem:
        mem.update(phase='seek', found=False, t=0, crumbs=[], ri=-1,
                   lastc=None, ax=x, ay=y, stuck=0)
    mem['t'] += 1
    # scan: an object matching ALL objective features means the target is here
    for d in state['scan']:
        f = d.get('features', {})
        if obj and all(f.get(k) == v for k, v in obj.items()):
            mem['found'] = True
    # drop a breadcrumb every ~0.45 m (used to retrace the way home)
    if mem['lastc'] is None or math.hypot(x - mem['lastc'][0], y - mem['lastc'][1]) > 0.45:
        mem['crumbs'].append((x, y)); mem['lastc'] = (x, y)
    def rng(lo, hi):
        xs = [d for d, a in zip(lidar, ang) if lo <= a < hi]
        return min(xs) if xs else R
    front = rng(-0.4, 0.4); left = rng(0.9, 1.8); fleft = rng(0.2, 0.9)
    right = rng(-1.8, -0.9); fright = rng(-0.9, -0.2)
    if math.hypot(x - mem['ax'], y - mem['ay']) > 0.4:
        mem['ax'], mem['ay'] = x, y; mem['stuck'] = 0
    else:
        mem['stuck'] += 1
    # after exploring long enough (or as soon as found), head home
    if mem['phase'] in ('seek', 'follow') and mem['t'] > 4200:
        mem['phase'] = 'return'; mem['ri'] = len(mem['crumbs']) - 1
    if mem['phase'] == 'return':
        if math.hypot(x - home[0], y - home[1]) <= state['goal_tol']:
            return {'v': 0.0, 'w': 0.0, 'done': True, 'present': mem['found']}
        while mem['ri'] > 0:                        # retrace breadcrumbs back home
            tx, ty = mem['crumbs'][mem['ri']]
            if math.hypot(tx - x, ty - y) < 0.4: mem['ri'] -= 1
            else: break
        tx, ty = mem['crumbs'][mem['ri']] if mem['ri'] >= 0 else home
        desired = math.atan2(ty - y, tx - x)
        aerr = math.atan2(math.sin(desired - th), math.cos(desired - th))
        if front < 0.55:
            return {'v': 0.0, 'w': 0.85 * w_max * (1 if aerr >= 0 else -1)}
        return {'v': v_max * max(0.18, math.cos(aerr)) * 0.8,
                'w': max(-w_max, min(w_max, 1.8 * aerr))}
    if mem['stuck'] > 30:                           # deadlock escape
        mem['stuck'] = 0
        return {'v': 0.0, 'w': (0.9 if left >= right else -0.9) * w_max}
    if mem['phase'] == 'seek':                      # drive until a wall is near, then follow it
        if min(front, left, right, fleft, fright) < 1.4: mem['phase'] = 'follow'
        if front < 0.8: return {'v': 0.0, 'w': -0.85 * w_max}
        return {'v': 0.55 * v_max, 'w': 0.0}
    # left-hand wall following: keep the left wall ~d0 away; turn right if blocked ahead,
    # turn left if the wall opens up. This autonomously threads doorways and tours the rooms.
    d0 = 1.0
    if front < 0.75:
        return {'v': 0.0, 'w': -0.9 * w_max}
    if left > d0 + 0.7 and fleft > d0 + 0.5:
        return {'v': 0.45 * v_max, 'w': 0.8 * w_max}
    w = 1.3 * (left - d0)
    if fright < 0.4 or right < 0.35: w -= 0.7
    return {'v': 0.5 * v_max, 'w': max(-w_max, min(w_max, w))}'''


HOUSE_SYSTEM = f"""You are the onboard controller of a TurtleBot3 home robot with a LiDAR and a forward CAMERA.
MISSION: AUTONOMOUSLY explore the house to decide whether a specific OBJECT is present, then RETURN to
home and report your verdict. There is NO pre-planned route — you must drive the robot around the house
yourself, using only the LiDAR and the camera. You write ONE Python function plan(state), run every
control tick, that explores, scans, avoids walls, comes back home, and declares the result.

state is a dict:
  pos=(x,y) m; heading=theta rad (0=+x, CCW+); v_max,w_max,dt; robot_radius,goal_tol; bounds=(W,H)
  lidar=list of distances (m); lidar_angles=robot-frame ray angles (0=front, CCW+); max_range
  objective = features the target object MUST match, e.g. {{'label':'cup','color':'red'}}
  scan = list of objects the camera sees NOW (forward cone only; occluded ones excluded), each:
         {{'bearing': rad (+left,-right), 'distance': m, 'features': {{'label':..,'color':..}}, 'conf':0..1}}
  home = [x,y] entrance to return to at the end (you know your own pos, so you can navigate back)
  memory = a dict that PERSISTS across calls (keep your phase, a 'found' flag, breadcrumbs, etc.)
NOTE: you are NOT given any waypoints or a map. Explore on your own (e.g. wall-following).

Return {{'v':linear,'w':angular}} each tick (v>0 forward, w>0 left; clamped to limits). When the mission
is COMPLETE (you are back home and have decided), return {{'done':True,'present':<bool>}}.
'present' must be True only if you actually SAW the target during exploration.

RULES:
  - EXPLORE autonomously to cover the rooms (a good simple strategy: follow walls with the LiDAR, which
    threads doorways and tours rooms; remember where you've been in memory).
  - SCAN every tick: an object counts as the target only if its 'features' match ALL of objective.
    BEWARE DECOYS that share just one feature (same color but different label, etc.) — do NOT count them.
  - When you've searched enough (or found it), RETURN to home (navigate toward state['home']; retracing
    your own breadcrumb trail is a robust way back). Only declare done within goal_tol of home.
  - Avoid walls: if a front lidar reading is small, set v=0 and turn toward the more open side.
Here is a complete working controller — you may return it as-is or improve it (keep it correct):
```python
{HOUSE_SCAFFOLD}
```
Reply with ONE ```python code block defining plan(state). No prose."""


def _house_state_block(state: dict) -> str:
    x, y = state["pos"]
    home = state.get("home", [x, y])
    return (
        f"Robot at ({x:.2f}, {y:.2f}), heading {state['heading']:.2f} rad. "
        f"v_max {state['v_max']:.2f}, w_max {state['w_max']:.2f}, goal_tol {state['goal_tol']:.2f}.\n"
        f"OBJECTIVE object features = {state.get('objective')}.\n"
        f"home = ({home[0]:.2f}, {home[1]:.2f}). No route is given — explore on your own.\n"
        f"Camera sees now: {_fmt_scan(state.get('scan'))}.\n"
        f"LiDAR (front rays) = "
        + ", ".join(f"{math.degrees(a):+.0f}:{d:.1f}" for d, a in
                    zip(state['lidar'], state['lidar_angles']) if abs(a) < 1.6))


def _fmt_scan(scan) -> str:
    if not scan:
        return "(no object in view)"
    return "; ".join(
        f"[bearing {math.degrees(d['bearing']):+.0f}deg, {d['distance']:.1f}m, {d['features']}]"
        for d in scan)


def house_initial_user(state: dict) -> str:
    return (
        _house_state_block(state) + "\n\n"
        "Write plan() so the robot AUTONOMOUSLY explores the house (no route is given), scans for the "
        "OBJECTIVE object (matching ALL its features, ignoring decoys), returns home, and then returns "
        "{'done':True,'present':<bool>}. Output one ```python code block."
    )


_HOUSE_HINT = {
    "missed_object": (
        "You reported the object as ABSENT, but it WAS in the house and reachable. You gave up too early: "
        "explore MORE of the house (cover the rooms, e.g. by wall-following) and keep scanning before "
        "deciding, and set found=True as soon as any detection matches ALL objective features."),
    "searched_too_little": (
        "You declared a verdict WITHOUT actually exploring the house (you barely moved, and for a 'present' "
        "verdict you must actually SEE the target first). Do NOT guess. Drive around and cover the rooms "
        "(wall-following works well); report present only after a detection matches ALL objective features, "
        "and report absent only after exploring most of the house. Then return home."),
    "false_report": (
        "You reported the object as PRESENT, but it was NOT there — you were fooled by a DECOY that "
        "matches only ONE feature (e.g. same color, different label). Count an object as the target ONLY "
        "if its 'features' match ALL of objective; ignore partial matches."),
    "not_home": (
        "You declared done but the robot was NOT back at home. First finish exploring, THEN navigate to "
        "state['home'] (retracing your breadcrumb trail is robust) and only return {'done':True,...} once "
        "within goal_tol of home."),
    "no_report": (
        "The robot moved but NEVER declared the mission done. After exploring enough and returning home, "
        "you MUST return {'done':True,'present':<bool>}. Track your phase in memory and switch to returning "
        "home after a while so you eventually finish."),
    "stuck": (
        "The robot stopped making progress (stuck against a wall or spinning). Use memory to detect being "
        "stuck (position barely changing) and turn toward the more open side (compare left vs right lidar) "
        "to escape, then keep exploring."),
    "collision": (
        "The robot kept hitting walls. Avoid them: when the nearest FRONT lidar (|angle|<~0.5) is small, "
        "set v=0 and rotate (do not creep forward) toward the side with more free space before moving on."),
    "out_of_bounds": (
        "The robot left the house bounds. Treat short lidar readings as walls and steer away; stay inside."),
    "exception": (
        "plan() crashed. Make it always return a dict for every input: guard empty scan/lidar lists and "
        "missing dict keys with .get, and avoid division by zero."),
}


def house_repair_user(reason: str, detail: str, state: dict, prev_code: str) -> str:
    head = ("Your previous plan() did NOT complete the house mission "
            "(search for the object and return home).\n" f"Outcome: {reason}.\n{detail}\n\n")
    if reason == "no_valid_code":
        ctx = "The code above failed to load (syntax/indentation/structure error), so it never ran.\n\n"
        hint = "Rewrite cleanly with consistent 4-space indentation; make sure plan() returns a dict."
    else:
        ctx = _house_state_block(state) + "\n\n"
        hint = _HOUSE_HINT.get(reason, "Diagnose what went wrong and fix it.")
    return (
        head + ctx +
        "Here is your previous code:\n```python\n" + prev_code + "\n```\n\n" +
        hint + " Output the COMPLETE corrected function in exactly one ```python code block."
    )
