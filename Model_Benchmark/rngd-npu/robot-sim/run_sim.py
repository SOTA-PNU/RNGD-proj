#!/usr/bin/env python3
"""로봇 내비게이션 폐루프 시뮬레이터 — CLI 러너.

로봇이 '현재 위치 + 목표 위치'만 알고, RNGD NPU 에 떠 있는 코딩 LLM 에게 자기를 움직일
컨트롤러 코드를 물어보고, 받은 코드를 자기 자신에게 적용해 목표까지 찾아가는 과정을
시뮬레이션으로 검증합니다. 코딩 성능(한 번에 맞히는가·고쳐서 맞히는가)을 숫자로 봅니다.

예시
  # 서버 없이 동작 확인(가짜 LLM, 시스템 python 으로 가능)
  python3 run_sim.py --mock good --scenario all
  python3 run_sim.py --mock buggy --scenario trap         # 자가수리(self-debug) 루프 시연

  # 실제 NPU 서버(먼저 ./serve_models.sh coder7 등으로 모델을 띄워야 함)
  .venv/bin/python run_sim.py --model coder7 --scenario all --report out.json
  .venv/bin/python run_sim.py --model a3b-fp8 --scenario trap --middleware threaded

옵션은 --help 참고. (실서버 모드는 openai 패키지가 있는 venv 로 실행하세요: chat/.venv 또는 ~/furiosa.)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))

from agent import DirectPipeline, NavAgent          # noqa: E402
from llm_client import MockClient, NpuClient         # noqa: E402
from metrics import dump_json, render_table          # noqa: E402
import scenarios as SC                               # noqa: E402
from viz import render_ascii, save_png               # noqa: E402

# chat_app.py CATALOG 와 일치하는 코딩계열 모델 매핑(키 → (포트, 표시명)).
# 코딩 성능이 관건이라 coder 계열을 우선 등록합니다. (출처: chat/chat_app.py:73-110)
# 표시명은 chat_app.py CATALOG 와 글자까지 동일하게 둬, CATALOG 표시명을 그대로 --model 에 넣어도 매칭됩니다.
MODELS = {
    "coder7":       (8002, "Qwen2.5-Coder-7B-Inst"),
    "coder14":      (8003, "Qwen2.5-Coder-14B-Inst"),
    "coder14-base": (8007, "Qwen2.5-Coder-14B tp8"),
    "coder32":      (8001, "Qwen2.5-Coder-32B-Inst"),
    "a3b-fp8":      (8000, "Qwen3-Coder-30B-A3B-Inst-FP8 tp8"),
    "a3b":          (8006, "Qwen3-Coder-30B-A3B-Inst tp8"),
    "qwen3-32b":    (8004, "Qwen3-32B-FP8"),
    "exaone-32b":   (8011, "EXAONE-4.0-32B-FP8"),
    "llama-70b":    (8012, "Llama-3.3-70B"),
    "qwen3-32b-tp32": (8013, "Qwen3-32B-FP8-tp32"),
}


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def build_client(args):
    if args.mock:
        return MockClient(mode=args.mock)
    # 실서버 모드: 포트 결정(우선순위: --port > --model)
    port, label, key = None, "", ""
    if args.model:
        if args.model in MODELS:
            port, label, key = (*MODELS[args.model], args.model)
        else:
            # 표시명으로도 찾아본다(정규화: 대소문자·공백·기호 무시)
            q = _norm(args.model)
            for k, (p, n) in MODELS.items():
                if _norm(n) == q:
                    port, label, key = p, n, k
                    break
    if args.port:
        port = args.port
    if port is None:
        sys.exit("✗ 실서버 모드에는 --model(키) 또는 --port 가 필요합니다. "
                 f"등록된 키: {', '.join(MODELS)}  또는 --mock good 으로 서버 없이 시험하세요.")
    client = NpuClient(port=port, model_label=label or f"npu:{port}")
    if not client.ping():
        sys.exit(
            f"✗ http://127.0.0.1:{port}/v1 에 연결할 수 없습니다.\n"
            f"  먼저 모델을 띄우세요:  cd .. && ./chat/serve_models.sh {key or '<키>'}  (키는 --list 참고)\n"
            f"  또는 서버 없이 시험:  python3 run_sim.py --mock good --scenario {args.scenario}")
    return client


def make_pipeline(args):
    if args.middleware == "off":
        return DirectPipeline(plan_timeout=args.plan_timeout)
    from middleware import MiddlewarePipeline
    return MiddlewarePipeline(plan_timeout=args.plan_timeout, threaded=(args.middleware == "threaded"))


def build_brain(args):
    """(2-LLM) 로봇 두뇌 LLM 클라이언트. --brain 없으면 None(단일-LLM, 기존 동작)."""
    if not args.brain:
        return None
    from brain import RobotBrain
    spec = args.brain
    if spec.startswith("mock"):
        bc = MockClient(mode=spec.split(":")[-1] if ":" in spec else "good")
    else:
        port, label = None, ""
        if spec in MODELS:
            port, label = MODELS[spec]
        elif spec.isdigit():
            port = int(spec)
        if port is None:
            sys.exit(f"✗ --brain 은 mock 또는 모델키{list(MODELS)} 또는 포트번호여야 합니다.")
        bc = NpuClient(port=port, model_label=label or f"npu:{port}")
        if not bc.ping():
            sys.exit(f"✗ 로봇 두뇌 모델(127.0.0.1:{port}) 연결 불가. 먼저 serve 하세요.")
    return RobotBrain(bc)


def run(args):
    client = build_client(args)
    brain = build_brain(args)
    pipeline = make_pipeline(args)
    agent = NavAgent(
        client, brain=brain, max_steps=args.max_steps, max_replans=args.max_replans,
        stuck_window=args.stuck_window, plan_timeout=args.plan_timeout,
        temperature=args.temperature, max_tokens=args.max_tokens,
        pipeline=pipeline, verbose=not args.quiet)

    names = SC.DEFAULT_SUITE if args.scenario in ("all", "suite") else [args.scenario]
    brain_note = f"  ·  로봇두뇌: {brain.name}" if brain else ""
    print(f"== 코더모델: {client.name}{brain_note}  ·  시나리오: {', '.join(names)}  ·  "
          f"미들웨어: {args.middleware} ==\n")

    results = []
    for name in names:
        world, start, goal, sc_name = SC.make(name, seed=args.seed)
        t0 = time.time()
        if getattr(world, "house_task", False):
            # 집 미션은 자율로 집을 돌고 복귀하므로 스텝 예산·정체창을 넉넉히(낮으면 no_report/stuck 오판).
            if agent.max_steps < 10000:
                agent.max_steps = 10000
            if agent.stuck_window < 250:
                agent.stuck_window = 250
            res = agent.run_house_episode(world, start, sc_name)
        else:
            res = agent.run_episode(world, start, goal, sc_name)
        res.model = client.name
        wall = time.time() - t0
        results.append(res)
        flag = "✅ 도달" if res.success else "❌ 실패"
        print(f"[{sc_name}] {flag}  reason={res.terminate_reason}  steps={res.steps}  "
              f"replans={res.replans}  코드첫빌드={'성공' if res.code_valid_first else '실패'}  "
              f"({wall:.1f}s, LLM {res.llm_calls}회/{res.llm_total_tokens}토큰)")
        if not args.no_viz:
            print(render_ascii(world, start, goal, res.path))
            if args.png_dir:
                os.makedirs(args.png_dir, exist_ok=True)
                p = save_png(world, start, goal, res.path, os.path.join(args.png_dir, f"{sc_name}.png"))
                if p:
                    print(f"  경로 PNG 저장: {p}")
        if pipeline and getattr(pipeline, "summary", None):
            st = pipeline.summary()
            if st.get("hops"):
                hop = "  ".join(f"{k}={v:.3f}ms" for k, v in st["hops"].items())
                print(f"  파이프라인({st.get('mode','direct')}): {hop}  사이클={st.get('cycle_hz',0):.0f}Hz")
        print()

    print(render_table(results))
    if args.report:
        meta = {"model": client.name, "mock": args.mock, "middleware": args.middleware,
                "max_steps": args.max_steps, "max_replans": args.max_replans, "seed": args.seed}
        dump_json(results, args.report, meta)
        print(f"\n리포트 저장: {args.report}")

    if hasattr(pipeline, "close"):
        pipeline.close()
    # 종료코드: 하나라도 실패하면 1(스크립트 자동화용)
    return 0 if all(r.success for r in results) else 1


def main():
    ap = argparse.ArgumentParser(description="RNGD NPU 코딩 LLM 로봇 내비게이션 시뮬레이터")
    ap.add_argument("--model", help=f"모델 키 {list(MODELS)} 또는 표시명")
    ap.add_argument("--port", type=int, help="furiosa-llm serve 포트 직접 지정(--model 대신)")
    ap.add_argument("--mock", choices=["good", "buggy"], help="서버 없이 가짜 LLM 으로 시험(good/buggy)")
    ap.add_argument("--brain", help="(2-LLM) 로봇 두뇌 LLM: mock | mock:good | 모델키 | 포트. "
                                    "주면 두뇌가 코더에게 자연어 지시를 만들어 보냄(없으면 단일-LLM)")
    ap.add_argument("--scenario", default="all",
                    help=f"시나리오 이름 또는 all. 가능: {', '.join(SC.list_scenarios())}")
    ap.add_argument("--seed", type=int, default=7, help="random 시나리오 시드")
    ap.add_argument("--max-steps", type=int, default=1500, help="에피소드당 최대 시뮬 스텝")
    ap.add_argument("--max-replans", type=int, default=4, help="최대 코드 재작성 횟수(self-debug)")
    ap.add_argument("--stuck-window", type=int, default=90, help="정체 판정 스텝수")
    ap.add_argument("--plan-timeout", type=float, default=0.5, help="plan() 한 번 호출 시간제한(초)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=700,
                    help="LLM 생성 토큰 상한(짧고 깨끗한 코드 유도; 길면 일부 모델이 열화)")
    ap.add_argument("--middleware", choices=["off", "sync", "threaded"], default="off",
                    help="ROS2식 노드 분리로 제어루프를 라우팅하고 홉 지연 측정")
    ap.add_argument("--no-viz", action="store_true", help="ASCII 경로 출력 끄기")
    ap.add_argument("--png-dir", help="경로 PNG 저장 폴더(matplotlib 있을 때)")
    ap.add_argument("--report", help="결과 JSON 저장 경로")
    ap.add_argument("--quiet", action="store_true", help="롤아웃 진행 로그 끄기")
    ap.add_argument("--list", action="store_true", help="시나리오/모델 목록 출력 후 종료")
    args = ap.parse_args()

    if args.list:
        print("시나리오:", ", ".join(SC.list_scenarios()))
        print("모델 키 :", ", ".join(f"{k}(:{p})" for k, (p, _) in MODELS.items()))
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
