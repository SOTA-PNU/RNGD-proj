"""에피소드를 한 번 돌리면서 시각화용 프레임(로봇 자세·LiDAR)과 월드 정보를 기록합니다.
브라우저 뷰어(web_sim.py)와 turtle 뷰어(turtle_sim.py)가 공유합니다.

반환: (meta, frames)
  meta   = 월드/결과 요약(장애물·목표·시작·성공여부 등)
  frames = 제어주기별 [{x, y, h, lidar:[...], ang:[...]}]  — 재생하면 로봇이 움직이는 애니메이션이 됨
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from agent import DirectPipeline, NavAgent
from llm_client import MockClient, NpuClient
import scenarios as SC

# chat_app.py CATALOG 와 일치(코딩계열). run_sim.py 와 동일 매핑.
MODELS = {
    "coder7": 8002, "coder14": 8003, "coder14-base": 8007, "coder32": 8001,
    "a3b-fp8": 8000, "a3b": 8006, "qwen3-32b": 8004,
    "exaone-32b": 8011, "llama-70b": 8012, "qwen3-32b-tp32": 8013,
}


def _make_client(mock: Optional[str], model: Optional[str], port: Optional[int]):
    if mock:
        return MockClient(mode=mock)
    p = port or (MODELS.get(model) if model else None)
    if p is None:
        raise SystemExit(f"✗ --model(키 {list(MODELS)}) 또는 --port 가 필요합니다. 또는 --mock good.")
    cli = NpuClient(port=p, model_label=model or f"npu:{p}")
    if not cli.ping():
        raise SystemExit(f"✗ http://127.0.0.1:{p}/v1 연결 불가. 먼저 모델을 띄우세요 "
                         f"(cd .. && ./chat/serve_models.sh {model or ''}) 또는 --mock good 사용.")
    return cli


def record_episode(scenario: str, *, mock: Optional[str] = None, model: Optional[str] = None,
                   port: Optional[int] = None, seed: int = 7, max_steps: int = 1500,
                   max_replans: int = 4, temperature: float = 0.2, max_tokens: int = 700,
                   stuck_window: int = 90, frame_stride: int = 1, frame_cap: int = 4000,
                   quiet: bool = True) -> Tuple[dict, List[dict]]:
    world, start, goal, name = SC.make(scenario, seed=seed)
    client = _make_client(mock, model, port)
    house = getattr(world, "house_task", False)
    if house:                                # 집 미션은 자율로 집을 돌고 복귀 → 스텝·정체창 넉넉히
        max_steps = max(max_steps, 10000)
        stuck_window = max(stuck_window, 250)

    frames: List[dict] = []
    n = [0]

    def cb(robot, state):
        n[0] += 1
        if (n[0] - 1) % frame_stride != 0:
            return
        if len(frames) >= frame_cap:
            return
        fr = {
            "x": round(robot.x, 3), "y": round(robot.y, 3), "h": round(robot.heading, 4),
            "lidar": [round(d, 2) for d in state["lidar"]],
            "ang": [round(a, 4) for a in state["lidar_angles"]],
        }
        if state.get("camera") is not None:   # 사람찾기: 그 순간 카메라에 잡힌 사람들
            fr["cam"] = [{"bearing": d["bearing"], "distance": d["distance"],
                          "features": d["features"], "conf": d["conf"]} for d in state["camera"]]
        if state.get("scan") is not None:     # 집 미션: 그 순간 카메라에 잡힌 물건들 + 진행상태
            fr["scan"] = [{"bearing": d["bearing"], "distance": d["distance"],
                           "features": d["features"], "conf": d["conf"]} for d in state["scan"]]
            mem = state.get("memory", {})
            fr["found"] = bool(mem.get("found"))
            fr["phase"] = mem.get("phase", "search")
        frames.append(fr)

    agent = NavAgent(client, max_steps=max_steps, max_replans=max_replans,
                     stuck_window=stuck_window, temperature=temperature, max_tokens=max_tokens,
                     pipeline=DirectPipeline(), verbose=not quiet, frame_cb=cb)
    res = agent.run_house_episode(world, start, name) if house \
        else agent.run_episode(world, start, goal, name)

    # 마지막(목표) 자세도 한 프레임 추가 — 도달 순간을 보여주려고
    if res.path:
        lx, ly = res.path[-1]
        last = frames[-1] if frames else {}
        fr = {"x": round(lx, 3), "y": round(ly, 3),
              "h": last.get("h", 0.0), "lidar": last.get("lidar", []), "ang": last.get("ang", [])}
        if house:               # 집 미션: 종료 프레임에 최종 진행상태(발견·복귀단계) 반영
            fr["scan"] = []
            fr["found"] = bool(res.success and world.objective_present())
            fr["phase"] = "home"
        frames.append(fr)

    meta = {
        "scenario": name, "model": getattr(client, "name", "llm"),
        "width": world.width, "height": world.height,
        "obstacles": [{"cx": o.cx, "cy": o.cy, "r": o.r} for o in world.obstacles],
        "goal": [goal[0], goal[1]], "start": [start[0], start[1]],
        "robot_radius": world.robot_radius, "goal_tol": world.goal_tol,
        "max_range": world.max_range, "v_max": world.v_max,
        "success": res.success, "reason": res.terminate_reason,
        "steps": res.steps, "replans": res.replans,
        "code_valid_first": res.code_valid_first, "n_frames": len(frames),
    }
    if world.vision_task:   # 카메라 사람찾기: 사람·target·FOV 정보를 뷰어에 전달
        meta["vision"] = True
        meta["cam_fov"] = world.cam_fov
        meta["cam_range"] = world.cam_range
        meta["target"] = dict(world.target or {})
        meta["people"] = [{"x": p.x, "y": p.y, "features": dict(p.features),
                           "is_target": world.matches_target(p)} for p in world.people]
    if house:               # 집 미션: 벽·물건·경로·현관·판정 정보를 뷰어에 전달
        meta["house"] = True
        meta["cam_fov"] = world.cam_fov
        meta["cam_range"] = world.cam_range
        meta["objective"] = dict(world.objective or {})
        meta["present"] = world.objective_present()
        meta["home"] = list(world.home) if world.home else list(start)
        meta["waypoints"] = [[round(x, 2), round(y, 2)] for x, y in world.waypoints]
        meta["walls"] = [{"cx": round(s.cx, 3), "cy": round(s.cy, 3),
                          "th": round(math.atan2(s.uy, s.ux), 4),
                          "L": round(s.hl * 2, 3), "T": round(s.ht * 2, 3)} for s in world.walls]
        meta["items"] = [{"x": round(it.x, 2), "y": round(it.y, 2), "features": dict(it.features),
                          "is_target": world.item_matches(it.features)} for it in world.items]
    return meta, frames
