#!/usr/bin/env python3
"""turtle(파이썬 표준 graphics)로 로봇 내비게이션을 '창'에 띄워 봅니다.

로봇이 LiDAR 로 장애물을 감지하고, NPU LLM 이 짠 컨트롤러 코드로 목표까지 가는 과정을
실시간 창으로 보여줍니다. turtle 은 디스플레이(X11/화면)가 필요하므로 **모니터가 있는 PC**
(예: 노트북/맥에 이 저장소를 받아서)에서 실행하세요. 서버처럼 디스플레이가 없으면 자동으로
브라우저 버전(web_sim.py)을 안내합니다.

예시 (디스플레이 있는 PC에서)
  python3 turtle_sim.py --mock good --scenario trap
  python3 turtle_sim.py --model coder7 --scenario slalom     # 실제 NPU (openai 있는 환경)
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))

from sim_record import MODELS, record_episode  # noqa: E402
import scenarios as SC                          # noqa: E402

BG, OBST, GOAL, ROBOT, LIDAR, TRAIL, TXT = (
    "#0a0a0a", "#333333", "#cdbbff", "#dc2626", "#76d6ff", "#ff5a5a", "#e8e8e8")


def _no_display(scenario):
    print("⚠️  이 환경에서는 turtle 창을 열 수 없습니다 (디스플레이/tkinter 없음).")
    print("    모니터가 있는 PC(예: 맥/노트북)에 이 저장소를 받아 실행하시거나,")
    print("    헤드리스/원격이면 브라우저 버전을 쓰세요(챗 UI처럼 터널로 보입니다):")
    print(f"      python3 web_sim.py --mock good --scenario {scenario} --serve 7900")
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser(description="turtle 창으로 보는 로봇 내비게이션 시뮬레이터")
    ap.add_argument("--model", help=f"모델 키 {list(MODELS)}")
    ap.add_argument("--port", type=int, help="furiosa-llm serve 포트 직접 지정")
    ap.add_argument("--mock", choices=["good", "buggy"], help="서버 없이 가짜 LLM")
    ap.add_argument("--scenario", default="trap", help=f"{', '.join(SC.list_scenarios())}")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-replans", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--speed", type=float, default=1.0, help="재생 속도(클수록 빠름)")
    ap.add_argument("--lidar", action="store_true", help="LiDAR 광선도 그리기")
    args = ap.parse_args()

    # 1) 디스플레이/turtle 사용 가능 여부 먼저 확인(헤드리스면 즉시 안내)
    try:
        import turtle  # noqa: F401  (tkinter 없으면 여기서 실패)
    except Exception:
        _no_display(args.scenario)

    # 2) 에피소드 기록(여긴 디스플레이 불필요)
    print(f"▶ {args.scenario} 기록 중…")
    meta, frames = record_episode(
        args.scenario, mock=args.mock, model=args.model, port=args.port, seed=args.seed,
        max_replans=args.max_replans, max_tokens=args.max_tokens, quiet=True)
    print(f"  {'✅ 도달' if meta['success'] else '❌ '+meta['reason']}  "
          f"프레임 {len(frames)}개 · 재작성 {meta['replans']}회 — 창에서 재생합니다.")
    if not frames:
        print("  (프레임이 없습니다. 모델이 코드를 못 만들었을 수 있어요. --mock good 으로 먼저 시험해 보세요.)")
        return

    _animate(meta, frames, args)


def _animate(meta, frames, args):
    import turtle
    W, H = meta["width"], meta["height"]
    try:
        screen = turtle.Screen()
    except Exception:
        _no_display(meta["scenario"])
    screen.title(f"Robot Nav Sim · {meta['scenario']} · {meta['model']}")
    screen.bgcolor(BG)
    px = 760
    screen.setup(px + 20, px + 60)
    screen.setworldcoordinates(0, 0, W, H)   # 월드 좌표(미터) 그대로, y-up(turtle 기본)
    screen.tracer(0, 0)

    def pen(color, width=1):
        t = turtle.Turtle(visible=False)
        t.speed(0); t.pencolor(color); t.fillcolor(color); t.width(width)
        return t

    # 정적 요소(한 번만)
    static = pen(OBST)
    for o in meta["obstacles"]:
        static.penup(); static.goto(o["cx"], o["cy"] - o["r"]); static.setheading(0)
        static.pendown(); static.fillcolor(OBST); static.begin_fill()
        static.circle(o["r"]); static.end_fill()
    g = pen(GOAL, 2)
    gx, gy = meta["goal"]
    g.penup(); g.goto(gx, gy - meta["goal_tol"]); g.pendown(); g.pencolor(GOAL); g.circle(meta["goal_tol"])
    g.penup(); g.goto(gx, gy); g.dot(8, GOAL); g.goto(gx + 0.4, gy + 0.4); g.pencolor(TXT); g.write("GOAL", font=("sans", 10, "normal"))
    s = pen(LIDAR, 2)
    sx, sy = meta["start"]; s.penup(); s.goto(sx, sy); s.dot(8, LIDAR)
    s.goto(sx + 0.4, sy + 0.4); s.pencolor(TXT); s.write("START", font=("sans", 10, "normal"))

    trail = pen(TRAIL, 2)
    trail.penup(); trail.goto(frames[0]["x"], frames[0]["y"]); trail.pendown()
    lidar_pen = pen(LIDAR, 1)
    hud = pen(TXT)

    robot = turtle.Turtle()
    robot.shape("triangle"); robot.shapesize(1.0, 1.6); robot.pencolor(ROBOT); robot.fillcolor(ROBOT)
    robot.penup(); robot.speed(0)

    rr = meta["robot_radius"]; max_range = meta["max_range"]

    state = {"i": 0}
    delay = max(5, int(30 / max(0.1, args.speed)))

    def draw_hud(f):
        hud.clear()
        d = math.hypot(meta["goal"][0] - f["x"], meta["goal"][1] - f["y"])
        res = "도달" if meta["success"] else meta["reason"]
        hud.penup(); hud.goto(0.3, H - 0.6); hud.pencolor(TXT)
        hud.write(f"{meta['model']} · {meta['scenario']} · step {state['i']}/{len(frames)-1} · "
                  f"목표까지 {d:.1f}m · {res}", font=("sans", 11, "normal"))

    def draw_lidar(f):
        lidar_pen.clear()
        if not args.lidar:
            return
        for d, a in zip(f.get("lidar", []), f.get("ang", [])):
            wa = f["h"] + a
            lidar_pen.penup(); lidar_pen.goto(f["x"], f["y"]); lidar_pen.pendown()
            lidar_pen.pencolor(LIDAR if d < max_range - 0.05 else "#1c2a30")
            lidar_pen.goto(f["x"] + d * math.cos(wa), f["y"] + d * math.sin(wa))

    def stepfn():
        i = state["i"]
        if i >= len(frames):
            screen.update()
            return
        f = frames[i]
        robot.goto(f["x"], f["y"])
        robot.setheading(math.degrees(f["h"]))
        trail.goto(f["x"], f["y"])
        draw_lidar(f)
        draw_hud(f)
        screen.update()
        state["i"] = i + 1
        screen.ontimer(stepfn, delay)

    stepfn()
    print("창을 닫으려면 클릭하세요.")
    screen.exitonclick()


if __name__ == "__main__":
    main()
