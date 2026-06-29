#!/usr/bin/env python3
"""라이브 인터랙티브 시뮬레이터 — '맵에 세워둔 로봇에게 사용자가 task 를 고르면, 그 순간부터
로봇이 NPU 서버의 LLM 에게 코드를 요청·수정하며 task 를 수행하는 과정'을 브라우저로 실시간 관람.

핵심(이 파일이 web_sim.py 와 다른 점):
  * web_sim 은 에피소드를 '미리 다 돌려' 녹화한 걸 재생합니다. 여기 live_sim 은 **사용자가 task 를
    고른 그 순간부터 실제로 NPU 서버에 요청을 보내며** 한 스텝씩 진행하고, 그 과정(요청·받은 코드·
    빌드 결과·주행·실패·수리요청·새 코드·완료)을 **SSE 로 실시간 스트리밍**합니다.
  * 로봇은 처음엔 '움직이는 기능'만 있고(컨트롤러 plan() 없음, 정지) 화면에 세워져 있습니다.
    사용자가 task 를 고르면 그때 비로소 LLM 에게 plan(state) 코드를 받아 적용하기 시작합니다.

실행(실제 NPU 모델이 필요하므로 openai/fastapi 가 있는 chat venv 로):
  먼저 모델 serve:   cd .. && ./chat/serve_models.sh coder7        # 포트 8002
  그다음:            ../chat/.venv/bin/python live_sim.py --port 7910
  맥북:              alpacon tunnel furiosa-npu-e6ec40 -l 7910 -r 7910  → http://127.0.0.1:7910
  (서버 없이 흐름만 보려면 UI 모델 선택에서 'mock(good/buggy)' 선택 — 배관 확인용.)

주의: 이 파일은 일부러 `from __future__ import annotations` 를 쓰지 않습니다 — FastAPI 가 라우트
      파라미터(Pydantic 모델 등)의 타입을 실제 객체로 보고 본문/쿼리를 올바로 구분해야 하기 때문입니다.
"""
import argparse
import datetime
import difflib
import json
import math
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))

import house_world as HW                      # noqa: E402
from agent import NavAgent                     # noqa: E402
from llm_client import MockClient, NpuClient   # noqa: E402

# ── 챗서비스의 모델 매니저(CATALOG + dp/pp 온디맨드 serve)를 그대로 재사용 ──
# chat_app 을 import 하면 CATALOG·MGR(ServeManager)·_par_flags 를 얻습니다(gradio UI 는 __main__
# 안에만 있어 import 만으로는 서버가 뜨지 않음 — 실측 확인). 같은 dp/pp·카드회계 로직을 공유해
# 드리프트가 없습니다. import 실패 시(예: 다른 환경) mock 만 가능한 축소 모드로 동작합니다.
_CHAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chat")
try:
    sys.path.insert(0, _CHAT_DIR)
    import chat_app as _CHAT                    # noqa: E402
    CATALOG = _CHAT.CATALOG
    MGR = _CHAT.MGR
    HAVE_MGR = True
    _CHAT_ERR = ""
except Exception as e:                          # noqa: BLE001
    CATALOG, MGR, HAVE_MGR, _CHAT_ERR = {}, None, False, str(e)


def model_rows():
    """UI 모델 목록: CATALOG(상태·dp/pp 포함) + mock. 챗서비스 모델선택 패널과 같은 정보."""
    rows = []
    if HAVE_MGR:
        try:
            st = MGR.states()
        except Exception:
            st = {}
        for k, m in CATALOG.items():
            dp, pp = (MGR.par(k) if MGR else (1, 1))
            rows.append({"key": k, "label": m["name"], "kind": m["kind"],
                         "pp_fixed": m.get("pp_fixed"), "ctx": m.get("ctx"),
                         "port": m["port"], "state": st.get(k, "down"), "dp": dp, "pp": pp})
    rows.append({"key": "mock:good", "label": "mock — 정상 흐름 확인(서버 불필요)",
                 "kind": "mock", "port": 0, "state": "-", "dp": 1, "pp": 1})
    rows.append({"key": "mock:buggy", "label": "mock — 자가수리 흐름 확인(서버 불필요)",
                 "kind": "mock", "port": 0, "state": "-", "dp": 1, "pp": 1})
    return rows

# ── 맵 위에서 고를 수 있는 task 들(House 맵). 미리 테스트하지 않은, '해 봐야 아는' 과제들. ──
TASKS = [
    {"id": "cup_red", "title": "빨간 컵이 집에 있는지 확인하고 현관으로 복귀",
     "desc": "집을 방마다 돌며 카메라로 빨간 컵을 찾습니다. 있으면 present, 없으면 absent 로 "
             "판정하고 현관으로 돌아옵니다. (정답: 빨간 컵은 집에 있습니다.)",
     "kind": "house", "objective": {"label": "cup", "color": "red"}, "present": True},
    {"id": "cup_blue", "title": "파란 컵이 집에 있는지 확인하고 복귀",
     "desc": "같은 집에서 이번엔 파란 컵을 찾습니다. 색만 같은 빨간 컵에 속지 말아야 합니다. "
             "(정답: 파란 컵도 집에 있습니다.)",
     "kind": "house", "objective": {"label": "cup", "color": "blue"}, "present": True},
    {"id": "umbrella", "title": "노란 우산이 집에 있는지 확인하고 복귀",
     "desc": "집에 노란 우산은 없습니다. 끝까지 돌아본 뒤 'absent' 로 판정해야 합니다. 헷갈리게 놓인 "
             "다른 물건(노란 책·파란 우산 등)에 속으면 false_report 입니다.",
     "kind": "house", "objective": {"label": "umbrella", "color": "yellow"}, "present": False},
    {"id": "goto_room", "title": "집 안쪽 방까지 이동(짧은 코드)",
     "desc": "물건 검색이 아니라, 현관에서 집 안쪽 방의 한 지점까지 LiDAR 로 벽을 피해 이동합니다. "
             "필요한 코드가 짧아, 서빙 모델로도 비교적 잘 됩니다.",
     "kind": "nav"},
]
_TASK_BY_ID = {t["id"]: t for t in TASKS}


class _Aborted(Exception):
    pass


def build_world_for_task(task: dict):
    """task → (world, start, goal, kind). house 면 물건검색 미션, nav 면 점-이동 미션."""
    if task["kind"] == "house":
        w, start, goal, _ = HW.build_house(task["objective"], task["present"], task["id"])
        return w, start, goal, "house"
    w, start, goal, _ = HW.build_house_nav(name=task["id"])
    return w, start, goal, "nav"


def _static_map(world, start, goal, kind) -> dict:
    """뷰어가 한 번 받는 정적 맵 정보(벽·현관·경로·목표). 물건 위치는 보내지 않습니다
    (로봇이 카메라로 '발견'할 때 비로소 화면에 나타나게 — 미리 정답을 보여주지 않음)."""
    walls = [{"cx": round(s.cx, 3), "cy": round(s.cy, 3),
              "th": round(math.atan2(s.uy, s.ux), 4),
              "L": round(s.hl * 2, 3), "T": round(s.ht * 2, 3)} for s in world.walls]
    m = {"width": world.width, "height": world.height, "walls": walls,
         "home": list(world.home) if world.home else list(start),
         "start": list(start), "robot_radius": world.robot_radius,
         "max_range": world.max_range, "cam_fov": world.cam_fov,
         "cam_range": world.cam_range, "kind": kind}
    if kind == "house":
        m["waypoints"] = [[round(x, 2), round(y, 2)] for x, y in world.waypoints]
        m["objective"] = dict(world.objective or {})
    else:
        m["goal"] = list(goal)
    return m


# ── 한 번의 라이브 실행(별 스레드) ────────────────────────────────
class Runner:
    def __init__(self):
        self.q: queue.Queue = queue.Queue()   # 영속 큐(스트림이 항상 같은 큐를 읽도록 재할당 안 함)
        self.stop = threading.Event()
        self.thread = None
        self.active = False

    def start(self, task: dict, model_key: str, dp: int = 1, pp: int = 1, brain_key: str = ""):
        # 이전 실행 중단 + 합류(끝나며 옛 이벤트를 다 흘려보낸 뒤)
        self.stop.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        # 큐에 남은 옛 이벤트 비우기(새 스트림이 옛 'end' 등을 보지 않게)
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        self.stop = threading.Event()
        self.active = True
        self.thread = threading.Thread(target=self._run, args=(task, model_key, dp, pp, brain_key), daemon=True)
        self.thread.start()

    def _client(self, model_key: str, dp: int, pp: int, emit, role: str = "coder"):
        """모델 선택 → 클라이언트. 실제 모델은 챗서비스 MGR 로 dp/pp 온디맨드 serve 후 붙습니다.
        role='coder'(서버 코더) | 'brain'(로봇 두뇌) — serve 상태 표시에만 씀."""
        rk = "로봇 두뇌" if role == "brain" else "서버 코더"
        if model_key.startswith("mock"):
            return MockClient(mode=model_key.split(":")[-1] or "good")
        if not HAVE_MGR or model_key not in CATALOG:
            raise ValueError(f"모델 매니저를 쓸 수 없습니다({_CHAT_ERR or '알 수 없는 모델'}). "
                             f"mock 으로 흐름을 보거나, chat venv 로 실행하세요.")
        m = CATALOG[model_key]
        if m.get("pp_fixed"):
            pp = int(m["pp_fixed"])               # pp 고정 모델(예: coder32=pp2)
        if m["kind"] == "tp32":
            dp, pp = 1, 1
        emit({"type": "serve", "state": "request", "model": m["name"], "dp": dp, "pp": pp, "role": role})
        emit({"type": "status",
              "text": f"[{rk}] '{m['name']}' 를 dp={dp}·pp={pp} ({dp*pp}장)로 준비(serve) 중… "
                      "(이미 떠 있으면 즉시 재사용)"})
        MGR.request(model_key, dp, pp)
        deadline = time.time() + 900.0           # serve 로딩 대기(빌드된 아티팩트 적재)
        last = None
        while time.time() < deadline:
            if self.stop.is_set():
                raise _Aborted()
            try:
                st = MGR.states().get(model_key, "down")
            except Exception:
                st = "down"
            if st != last:
                emit({"type": "serve", "state": st, "model": m["name"], "dp": dp, "pp": pp, "role": role})
                last = st
            if st == "up":
                break
            if st == "error":
                err = ""
                try:
                    err = MGR._err.get(model_key, "")
                except Exception:
                    pass
                raise RuntimeError(f"모델 serve 실패: {err or 'serve_logs 확인'}")
            time.sleep(1.5)
        else:
            raise TimeoutError("모델 serve 대기 시간 초과(아티팩트 적재가 오래 걸립니다).")
        return NpuClient(port=m["port"], model_label=m["name"])

    def _run(self, task: dict, model_key: str, dp: int = 1, pp: int = 1, brain_key: str = ""):
        my_stop = self.stop                       # 이 실행 전용 중단 플래그(다음 실행 시작 시 set 됨)
        rec = {"task": task["id"], "title": task.get("title", ""), "kind": task["kind"],
               "model": model_key, "dp": dp, "pp": pp, "brain": brain_key or None,
               "objective": task.get("objective"),
               "present_truth": task.get("present"), "seq": [], "codes": [], "result": None}

        def emit(ev):
            # 중단/새 실행 시작 후엔 이 실행의 이벤트를 흘리지 않음(스트림 오염·느린 중단 방지)
            if my_stop.is_set():
                return
            t = ev.get("type")
            if t == "brain":
                rec["seq"].append("BRAIN(%s): %s" % (ev.get("phase"), (ev.get("text") or "")[:120]))
            elif t == "request":
                rec["seq"].append(("repair#%d(%s)" % (ev.get("attempt"), ev.get("reason"))
                                   if ev.get("phase") == "repair" else "initial request"))
            elif t == "code":
                rec["codes"].append({"attempt": ev.get("attempt"), "code": ev.get("code", ""),
                                     "metrics": ev.get("metrics")})
            elif t == "failure":
                rec["seq"].append("FAIL: %s — %s" % (ev.get("reason"), ev.get("detail", "")))
            elif t == "result":
                rec["result"] = {k: ev.get(k) for k in
                                 ("success", "reason", "replans", "steps", "tokens", "llm_calls")}
            self.q.put(ev)

        try:
            world, start, goal, kind = build_world_for_task(task)
            emit({"type": "map", **_static_map(world, start, goal, kind)})
            emit({"type": "status", "text": f"모델({model_key}) 준비 중…"})
            client = self._client(model_key, dp, pp, emit, role="coder")
            # (2-LLM) 로봇 두뇌 모델이 선택됐으면 그것도 serve 해서 붙임. 없으면 단일-LLM.
            brain = None
            if brain_key:
                from brain import RobotBrain
                bclient = self._client(brain_key, 1, 1, emit, role="brain")
                brain = RobotBrain(bclient)
                emit({"type": "status", "text": "준비 완료 — 로봇 두뇌가 코더에게 보낼 지시를 작성합니다…"})
            else:
                emit({"type": "status", "text": "준비 완료 — NPU 에 첫 코드를 요청합니다…"})

            # 프레임 콜백: 매 제어주기마다 로봇 자세 + (집이면) 카메라 검출을 스트리밍(스로틀)
            tick = {"n": 0}

            def frame_cb(robot, state):
                if my_stop.is_set():
                    raise _Aborted()
                tick["n"] += 1
                if (tick["n"] - 1) % 3 != 0:    # 3틱마다 한 프레임(과부하 방지)
                    return
                fr = {"type": "frame", "x": round(robot.x, 3), "y": round(robot.y, 3),
                      "h": round(robot.heading, 4),
                      "lidar": [round(d, 2) for d in state["lidar"]],
                      "ang": [round(a, 4) for a in state["lidar_angles"]]}
                if state.get("scan") is not None:
                    fr["scan"] = [{"bearing": d["bearing"], "distance": d["distance"],
                                   "features": d["features"]} for d in state["scan"]]
                    mem = state.get("memory", {})
                    fr["phase"] = mem.get("phase", "search")
                    fr["found"] = bool(mem.get("found"))
                emit(fr)
                time.sleep(0.012)               # 너무 빠른 진행을 살짝 늦춰 눈으로 따라가게

            agent = NavAgent(client, brain=brain, max_steps=10000 if kind == "house" else 2500,
                             max_replans=4, stuck_window=250 if kind == "house" else 110,
                             temperature=0.2, max_tokens=900 if kind == "house" else 600,
                             verbose=False, frame_cb=frame_cb, on_event=emit)
            if kind == "house":
                res = agent.run_house_episode(world, start, task["id"])
            else:
                res = agent.run_episode(world, start, goal, task["id"])

            # 끝나면 정답(물건 실제 배치)도 공개 — 로봇 판정이 맞았는지 사용자가 확인하게
            reveal = None
            if kind == "house":
                reveal = [{"x": round(it.x, 2), "y": round(it.y, 2),
                           "features": dict(it.features),
                           "is_target": world.item_matches(it.features)} for it in world.items]
            emit({"type": "result", "success": res.success, "reason": res.terminate_reason,
                  "replans": res.replans, "steps": res.steps,
                  "tokens": res.llm_total_tokens, "llm_calls": res.llm_calls,
                  "reveal_items": reveal})
            # 결과 정리를 robot-sim/results/ 에 저장(JSON + 사람이 읽는 .md)
            if not my_stop.is_set():
                try:
                    paths = _save_results(rec, reveal)
                    emit({"type": "saved", "json": paths[0], "md": paths[1]})
                except Exception as e:  # noqa: BLE001
                    emit({"type": "status", "text": f"결과 저장 실패: {e}"})
        except _Aborted:
            emit({"type": "status", "text": "중단됨."})
        except Exception as e:  # noqa: BLE001
            emit({"type": "error", "text": f"{type(e).__name__}: {e}"})
        finally:
            emit({"type": "end"})
            self.active = False


# 결과 정리가 저장되는 폴더: robot-sim/results/
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _save_results(rec, reveal):
    """한 task 의 '결과 정리'(대화 흐름 + 시작↔최종 코드 + 결과)를 robot-sim/results/ 에
    JSON(기계용) + .md(사람이 읽는 용) 두 파일로 저장하고, (json상대경로, md상대경로)를 돌려줍니다."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now()
    base = f"{ts.strftime('%Y%m%d_%H%M%S')}_{rec['task']}_{str(rec['model']).replace(':', '-')}"
    jpath = os.path.join(RESULTS_DIR, base + ".json")
    mpath = os.path.join(RESULTS_DIR, base + ".md")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts.isoformat(timespec="seconds"), **rec, "reveal_items": reveal},
                  f, ensure_ascii=False, indent=2)

    codes = rec["codes"]
    res = rec["result"] or {}
    first = codes[0]["code"] if codes else ""
    final = codes[-1]["code"] if codes else ""
    ok = res.get("success")
    L = [f"# {rec['title'] or rec['task']}", "",
         f"- 시각: {ts.strftime('%Y-%m-%d %H:%M:%S')}",
         f"- 모델: {rec['model']} (dp={rec['dp']}, pp={rec['pp']})"]
    if rec.get("objective"):
        L.append(f"- 찾는 물건(objective): {rec['objective']}  ·  (관찰자만 아는) 실제 존재: {rec.get('present_truth')}")
    L.append(f"- 결과: {'✅ 성공' if ok else '❌ 실패(' + str(res.get('reason')) + ')'}"
             f"  ·  코드 재작성 {res.get('replans', 0)}회  ·  스텝 {res.get('steps', 0)}"
             f"  ·  LLM {res.get('llm_calls', 0)}회/{res.get('tokens', 0)}토큰")
    L += ["", "## NPU 와의 대화 흐름"]
    L += [f"- {s}" for s in rec["seq"]] or ["- (요청 없음)"]
    L.append(f"- {'✅ 성공' if ok else '종료(' + str(res.get('reason')) + ')'}")
    L.append("")
    if len(codes) <= 1:
        L += ["## 코드 변화", ("첫 코드가 그대로 성공(수정 없음)." if codes else "받은 코드 없음.")]
    else:
        L += ["## 시작 코드 → 최종 코드 (diff)", "```diff"]
        L += list(difflib.unified_diff(first.splitlines(), final.splitlines(),
                                       fromfile="start", tofile="final", lineterm=""))
        L.append("```")
    L += ["", "## 최종 코드", "```python", final, "```", ""]
    with open(mpath, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    parent = os.path.dirname(RESULTS_DIR)
    return os.path.relpath(jpath, parent), os.path.relpath(mpath, parent)


RUNNER = Runner()


# ── FastAPI 앱 ────────────────────────────────────────────────────
def make_app():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    app = FastAPI()
    # three.js 를 로컬 번들로 서빙(CDN/인터넷 의존 없이 터널에서도 3D 가 뜨도록).
    _vendor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
    if os.path.isdir(_vendor):
        app.mount("/vendor", StaticFiles(directory=_vendor), name="vendor")

    class RunReq(BaseModel):
        task: str
        model: str = "coder7"
        dp: int = 1
        pp: int = 1
        brain: str = ""          # (2-LLM) 로봇 두뇌 모델 키. 빈 값이면 단일-LLM(코더에 직접)

    @app.get("/")
    def index():
        return HTMLResponse(PAGE)

    @app.get("/api/map")
    def static_map(task: str):
        t = _TASK_BY_ID.get(task)
        if not t:
            return JSONResponse({"error": "unknown task"}, status_code=400)
        world, start, goal, kind = build_world_for_task(t)
        return JSONResponse({"type": "map", **_static_map(world, start, goal, kind)})

    @app.get("/api/tasks")
    def tasks():
        return JSONResponse({
            "tasks": [{"id": t["id"], "title": t["title"], "desc": t["desc"], "kind": t["kind"]}
                      for t in TASKS],
            "models": model_rows(),
            "have_mgr": HAVE_MGR, "mgr_error": _CHAT_ERR,
        })

    @app.get("/api/serve_status")
    def serve_status():
        return JSONResponse({"models": model_rows()})

    @app.post("/api/run")
    def run(body: RunReq):
        task = _TASK_BY_ID.get(body.task)
        if not task:
            return JSONResponse({"ok": False, "error": "알 수 없는 task"}, status_code=400)
        RUNNER.start(task, body.model, dp=body.dp, pp=body.pp, brain_key=body.brain)
        return JSONResponse({"ok": True})

    @app.post("/api/stop")
    def stop():
        RUNNER.stop.set()
        return JSONResponse({"ok": True})

    @app.get("/api/stream")
    def stream():
        q = RUNNER.q

        def gen():
            yield "retry: 2000\n\n"
            while True:
                try:
                    ev = q.get(timeout=1.0)
                except queue.Empty:
                    yield ": ping\n\n"
                    continue
                yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
                if ev.get("type") == "end":
                    break

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Furiosa RNGD · 라이브 3D 로봇 task</title>
<style>
 :root{--bg:#0a0a0a;--card:#151515;--red:#dc2626;--cyan:#76d6ff;--purple:#cdbbff;--mute:#8a8a8a;--bd:#262626;--txt:#e8e8e8;--ok:#34d399;--amber:#f59e0b}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--txt);
   font-family:ui-sans-serif,system-ui,'Apple SD Gothic Neo',sans-serif}
 header{display:flex;align-items:center;gap:12px;padding:13px 20px;border-bottom:1px solid var(--bd)}
 header .logo{width:14px;height:14px;border-radius:3px;background:var(--red)}
 header b{font-size:15px} header .badge{margin-left:6px;font-size:11px;color:var(--mute);border:1px solid var(--bd);border-radius:6px;padding:2px 7px}
 /* 넓은 화면: 레이아웃을 '뷰포트 높이'에 가둔다 → 창을 키워도 3D 화면이 창 밖으로 안 넘침.
    캔버스는 남는 높이를 채우고, 오른쪽 패널이 길면 그 칸 안에서만 스크롤된다. */
 .wrap{display:flex;gap:16px;padding:16px 18px;align-items:stretch;
   height:calc(100vh - 58px);box-sizing:border-box;overflow:hidden}
 .stage{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:10px;
   flex:1 1 560px;min-width:380px;min-height:0;display:flex;flex-direction:column;gap:6px;overflow:hidden}
 #view{width:100%;flex:1 1 auto;min-height:0;border-radius:8px;background:#060606;display:block}
 .sensors{display:flex;gap:8px;flex:0 0 auto;height:200px}
 .scopebox{flex:1;min-width:0;background:#0c0c0c;border:1px solid var(--bd);border-radius:8px;
   padding:6px 8px;display:flex;flex-direction:column;gap:4px}
 .scopebox h4{margin:0;font-size:11px;color:var(--cyan);font-weight:600;display:flex;justify-content:space-between}
 .scopebox h4 span{color:var(--mute);font-weight:400}
 canvas.scope{flex:1;width:100%;min-height:0;background:#060606;border-radius:6px;display:block}
 .vhint{color:var(--mute);font-size:11px;margin:0;padding:0 2px;flex:0 0 auto}
 .side{flex:1 1 380px;min-width:330px;max-width:560px;display:flex;flex-direction:column;gap:12px;
   overflow-y:auto;min-height:0}
 /* 좁은 화면: 위아래로 쌓고 페이지 스크롤 허용(높이 고정 해제) */
 @media(max-width:980px){
   .wrap{flex-wrap:wrap;height:auto;overflow:visible}
   .side{overflow-y:visible;max-width:none}
   #view{height:62vh;flex:0 0 auto}
 }
 .panel{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:12px}
 .panel h3{margin:0 0 8px;font-size:13px;color:var(--cyan);font-weight:600}
 .panel.grow{flex:1 1 auto;min-height:260px;display:flex;flex-direction:column}  /* 로그 칸이 남는 높이를 채움 */
 .task{border:1px solid var(--bd);border-radius:10px;padding:9px 11px;margin:7px 0;cursor:pointer;transition:.15s}
 .task:hover{border-color:var(--red)} .task.sel{border-color:var(--red);background:#1d1414}
 .task b{font-size:13px} .task p{margin:4px 0 0;font-size:11.5px;color:var(--mute);line-height:1.5}
 .row{display:flex;gap:8px;align-items:center;margin-top:10px;flex-wrap:wrap}
 .lbl{color:var(--mute);font-size:12px;min-width:38px}
 select,button{background:#1b1b1b;color:var(--txt);border:1px solid var(--bd);border-radius:9px;padding:8px 11px;font-size:13px}
 button{cursor:pointer} button.go{background:var(--red);border-color:var(--red);color:#fff;font-weight:600}
 button:disabled{opacity:.45;cursor:not-allowed}
 select:disabled{opacity:.5}
 .st{font-size:10.5px;padding:2px 7px;border-radius:6px;border:1px solid var(--bd);color:var(--mute)}
 .st.up{color:#062;background:#0c2a18;border-color:#1d6b3f} .st.up::before{content:"● "}
 .st.loading{color:#7a4a00;background:#2a1f0c;border-color:#7a5a1d}
 .st.down{color:var(--mute)} .st.error{color:#fff;background:#3a1414;border-color:#7a1d1d}
 #log{flex:1 1 auto;min-height:max(220px,min(620px,calc(100vh - 560px)));
   overflow-y:auto;font-size:12px;resize:vertical}  /* 창 높이에 비례해 길게 + 가장자리 드래그로 더 늘리기 */
 .ev{border-left:3px solid var(--bd);padding:5px 9px;margin:7px 0;background:#101010;border-radius:0 8px 8px 0}
 .ev .k{font-size:10.5px;color:var(--mute);text-transform:uppercase;letter-spacing:.4px}
 .ev pre{margin:5px 0 0;white-space:pre-wrap;word-break:break-word;font-size:11px;color:#cfe8ff;
   background:#0b0b0b;border:1px solid #1a1a1a;border-radius:6px;padding:7px;max-height:190px;overflow:auto}
 .ev.req{border-color:var(--cyan)} .ev.code{border-color:var(--purple)}
 .ev.ok{border-color:var(--ok)} .ev.fail{border-color:var(--red)} .ev.fail b{color:#ff7676}
 .ev.serve{border-color:var(--amber)} .ev.brain{border-color:var(--purple);background:#14111c}
 .ev.done{border-color:var(--ok)} .ev.done.fail{border-color:var(--red)}
 .kv{display:flex;justify-content:space-between;font-size:12px;padding:4px 0;border-bottom:1px solid #1a1a1a}
 .kv span:first-child{color:var(--mute)}
 .hint{color:var(--mute);font-size:11.5px;line-height:1.6;margin-top:6px}
 details summary{cursor:pointer;color:var(--mute);font-size:11px}
</style>
<script type="importmap">
{ "imports": {
  "three": "/vendor/three/three.module.js",
  "three/addons/": "/vendor/three/addons/"
}}
</script></head>
<body>
<header><div class="logo"></div><b>Furiosa RNGD · 라이브 3D 로봇 task</b>
 <span class="badge">맵에서 모델·dp/pp 고르고 task 선택 → NPU 가 코드를 짜 수행</span></header>
<div class="wrap">
 <div class="stage"><canvas id="view"></canvas>
   <div class="sensors">
     <div class="scopebox"><h4>📡 LiDAR 스캔 <span id="lidarInfo">로봇 기준 360°</span></h4>
       <canvas id="lidarView" class="scope"></canvas></div>
     <div class="scopebox"><h4>📷 카메라 시야 <span id="camInfo">전방 약 59° · 검출</span></h4>
       <canvas id="camView" class="scope"></canvas></div>
   </div>
   <div class="vhint">3D 맵은 마우스로 회전·확대·이동. 아래는 로봇이 실제로 받는 LiDAR·카메라 신호를 따로 본 것입니다.</div>
 </div>
 <div class="side">
  <div class="panel">
   <h3>1) 모델 고르기 — 로봇 두뇌 + 서버 코더 (2-LLM)</h3>
   <div class="row"><span class="lbl">로봇 두뇌</span>
     <select id="brain" style="flex:1"></select><span id="bstate" class="st down">-</span></div>
   <div class="row"><span class="lbl">서버 코더</span>
     <select id="model" style="flex:1"></select><span id="mstate" class="st down">-</span></div>
   <div class="row"><span class="lbl">코더 dp</span><select id="dp"></select>
     <span class="lbl">코더 pp</span><select id="pp"></select></div>
   <div class="hint" id="parhint">로봇 두뇌(작은 모델)가 task 를 받아 '무엇을 만들어/고쳐 줘'를 자연어로
     서버 코더(큰 모델)에게 보내고, 코더가 코드를 만들어 줍니다. 두뇌를 '(없음)'으로 두면 단일-LLM(코더에 직접)
     으로 동작합니다. dp/pp 는 서버 코더용(tp8 dp×pp≤4, tp32 4장고정, 일부 pp고정).</div>
  </div>
  <div class="panel">
   <h3>2) 할 일(task) 고르기</h3>
   <div id="tasks"></div>
   <div class="hint" style="margin:4px 0 2px">※ 괄호 속 ‘정답’은 우리(관찰자)만 봅니다 — 로봇/LLM 에는
     찾을 물건 종류(objective)만 알려주고, 있는지 여부·위치는 주지 않습니다. 로봇이 스스로 탐색해 알아냅니다.</div>
   <div class="row">
    <button class="go" id="go" disabled>▶ 시작</button>
    <button id="stop" disabled>■ 중단</button>
   </div>
  </div>
  <div class="panel grow">
   <h3>3) NPU LLM 과의 과정(실시간)</h3>
   <div id="status" class="hint">대기 중 — 모델과 task 를 고르세요.</div>
   <div id="log"></div>
  </div>
  <div class="panel" id="resultPanel" style="display:none">
   <h3>결과 · 자동 정리</h3><div id="result"></div>
   <div id="summary"></div>
  </div>
 </div>
</div>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let M=null, sel=null, es=null, running=false, MODELS=[];
let runlog={codes:[], seq:[]};   // 한 실행의 코드 이력 + 대화 흐름(종료 후 자동 정리용)
let renderer, scene, camera, controls, mapGroup, robot, robotFwd, fovMesh, trailLine, trailPts=[];
let discovered={}, reveal=null, lastFrame=null;
const WALL_H=2.4;
const ICOL={red:0xdc2626,blue:0x3b82f6,green:0x22c55e,yellow:0xeab308};
const ICSS={red:'#dc2626',blue:'#3b82f6',green:'#22c55e',yellow:'#eab308'};

// ── 별도 센서 모니터: LiDAR 스코프 + 카메라 시야 ──
let lidarCv, lctx, camCv, cctx, DPRS=Math.max(1,window.devicePixelRatio||1);
function setupScopes(){
 lidarCv=document.getElementById('lidarView'); camCv=document.getElementById('camView');
 lctx=lidarCv.getContext('2d'); cctx=camCv.getContext('2d');
 resizeScopes();
 new ResizeObserver(resizeScopes).observe(lidarCv.parentElement);
}
function resizeScopes(){
 for(const c of [lidarCv,camCv]){ if(!c)continue;
   const w=c.clientWidth||220,h=c.clientHeight||150; c.width=w*DPRS; c.height=h*DPRS;
   c.getContext('2d').setTransform(DPRS,0,0,DPRS,0,0); }
 drawScopes(lastFrame);
}
function drawScopes(fr){ drawLidarScope(fr); drawCamView(fr); }

function drawLidarScope(fr){
 if(!lctx)return; const W=lidarCv.width/DPRS, H=lidarCv.height/DPRS, cx=W/2, cy=H/2, R=Math.min(W,H)/2-14;
 lctx.fillStyle='#060606'; lctx.fillRect(0,0,W,H);
 const maxr=(M&&M.max_range)||6;
 // 거리 링(2m 간격) + 라벨
 lctx.strokeStyle='#1d1d22'; lctx.fillStyle='#444'; lctx.font='9px sans-serif';
 for(let r=2;r<=maxr+0.01;r+=2){ const rr=(r/maxr)*R;
   lctx.beginPath(); lctx.arc(cx,cy,rr,0,7); lctx.stroke();
   lctx.fillText(r+'m', cx+2, cy-rr+9); }
 // 십자선
 lctx.strokeStyle='#161616'; lctx.beginPath(); lctx.moveTo(cx-R,cy); lctx.lineTo(cx+R,cy);
 lctx.moveTo(cx,cy-R); lctx.lineTo(cx,cy+R); lctx.stroke();
 if(!fr||!fr.lidar){ lctx.fillStyle='#555'; lctx.font='11px sans-serif'; lctx.textAlign='center';
   lctx.fillText('대기 중', cx, cy); lctx.textAlign='left'; }
 else {
   const lid=fr.lidar, ang=fr.ang||[];
   // 빔: robot frame a(0=정면,+=좌). 화면 정면=위. sx=cx-sin(a)*len, sy=cy-cos(a)*len
   for(let i=0;i<lid.length;i++){ const a=ang[i], d=Math.min(lid[i],maxr), len=(d/maxr)*R;
     const sx=cx-Math.sin(a)*len, sy=cy-Math.cos(a)*len, hit=lid[i]<maxr-0.05;
     lctx.strokeStyle=hit?'rgba(118,214,255,0.55)':'rgba(118,214,255,0.10)'; lctx.lineWidth=1;
     lctx.beginPath(); lctx.moveTo(cx,cy); lctx.lineTo(sx,sy); lctx.stroke();
     if(hit){ lctx.fillStyle='#76d6ff'; lctx.beginPath(); lctx.arc(sx,sy,2,0,7); lctx.fill(); } }
   // 점들을 잇는 외곽선(스캔 형태)
   lctx.strokeStyle='rgba(118,214,255,0.85)'; lctx.lineWidth=1.3; lctx.beginPath();
   for(let i=0;i<lid.length;i++){ const a=ang[i], d=Math.min(lid[i],maxr), len=(d/maxr)*R;
     const sx=cx-Math.sin(a)*len, sy=cy-Math.cos(a)*len; i?lctx.lineTo(sx,sy):lctx.moveTo(sx,sy); }
   lctx.closePath(); lctx.stroke();
 }
 // 로봇 + 정면 화살표(위)
 lctx.fillStyle='#dc2626'; lctx.beginPath(); lctx.arc(cx,cy,4,0,7); lctx.fill();
 lctx.fillStyle='#ff7a7a'; lctx.beginPath(); lctx.moveTo(cx,cy-9); lctx.lineTo(cx-4,cy-3); lctx.lineTo(cx+4,cy-3); lctx.closePath(); lctx.fill();
 const c=document.getElementById('lidarInfo'); if(c&&fr&&fr.lidar){ const f=fr.lidar.length;
   c.textContent=f+'빔 · '+maxr+'m'; }
}

function drawCamView(fr){
 if(!cctx)return; const W=camCv.width/DPRS, H=camCv.height/DPRS;
 cctx.fillStyle='#0a0d10'; cctx.fillRect(0,0,W,H);
 // 카메라 프레임 + 중앙 십자
 cctx.strokeStyle='#1d2730'; cctx.lineWidth=1; cctx.strokeRect(2,2,W-4,H-4);
 cctx.strokeStyle='#141a1f'; cctx.beginPath(); cctx.moveTo(W/2,0); cctx.lineTo(W/2,H);
 cctx.moveTo(0,H/2); cctx.lineTo(W,H/2); cctx.stroke();
 const fov=(M&&M.cam_fov)||1.0297;
 const scan=fr&&fr.scan; const isHouse=M&&M.kind==='house';
 if(!isHouse){ cctx.fillStyle='#555'; cctx.font='11px sans-serif'; cctx.textAlign='center';
   cctx.fillText('이 task 는 카메라 검출 없음 (이동 전용)', W/2, H/2-6); cctx.textAlign='left';
   document.getElementById('camInfo').textContent='—'; return; }
 if(!scan||!scan.length){ cctx.fillStyle='#666'; cctx.font='11px sans-serif'; cctx.textAlign='center';
   cctx.fillText(fr?'시야에 물건 없음':'대기 중', W/2, H/2-6); cctx.textAlign='left';
   document.getElementById('camInfo').textContent='검출 0'; return; }
 // 검출을 카메라 프레임에 배치: bearing(+좌) → 화면 x(좌가 0), 거리→크기(가까울수록 큼)
 for(const d of scan){ const b=d.bearing; if(Math.abs(b)>fov/2+0.05) continue;
   const fx=W*(0.5 - b/fov); const dist=Math.max(0.3,d.distance);
   const bh=Math.max(14,Math.min(H*0.82, (H*0.9)*(0.9/dist))); const bw=bh*0.62;
   const col=ICSS[(d.features||{}).color]||'#9ca3af';
   cctx.fillStyle=col; cctx.globalAlpha=0.85; cctx.fillRect(fx-bw/2,H/2-bh/2,bw,bh); cctx.globalAlpha=1;
   cctx.strokeStyle='#000'; cctx.lineWidth=1; cctx.strokeRect(fx-bw/2,H/2-bh/2,bw,bh);
   cctx.fillStyle='#e8e8e8'; cctx.font='10px sans-serif'; cctx.textAlign='center';
   cctx.fillText(((d.features||{}).label||'?'), fx, H/2-bh/2-4);
   cctx.fillStyle='#9fb3c0'; cctx.fillText(dist.toFixed(1)+'m', fx, H/2+bh/2+11); cctx.textAlign='left'; }
 document.getElementById('camInfo').textContent='검출 '+scan.length;
}

function resize3d(){
 const cv=document.getElementById('view');
 const w=cv.clientWidth||720, h=cv.clientHeight||560;
 renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix();
}
function init3d(){
 const cv=document.getElementById('view');
 renderer=new THREE.WebGLRenderer({canvas:cv,antialias:true});
 renderer.setPixelRatio(Math.min(2,window.devicePixelRatio||1));
 scene=new THREE.Scene(); scene.background=new THREE.Color(0x060606);
 scene.fog=new THREE.Fog(0x060606, 22, 46);
 camera=new THREE.PerspectiveCamera(50, (cv.clientWidth||720)/(cv.clientHeight||560), 0.1, 200);
 camera.position.set(0,18,16);
 controls=new OrbitControls(camera,renderer.domElement);
 controls.enableDamping=true; controls.target.set(0,0,0);
 scene.add(new THREE.AmbientLight(0x8088a0,1.1));
 const dl=new THREE.DirectionalLight(0xffffff,1.0); dl.position.set(8,20,6); scene.add(dl);
 const dl2=new THREE.DirectionalLight(0x88aaff,0.35); dl2.position.set(-10,12,-8); scene.add(dl2);
 mapGroup=new THREE.Group(); scene.add(mapGroup);
 resize3d();
 new ResizeObserver(resize3d).observe(cv);          // 창/칸 크기 바뀌면 3D 뷰도 따라감
 window.addEventListener('resize', resize3d);
 setupScopes();                                     // 별도 LiDAR·카메라 모니터 캔버스
 animate();
}
function animate(){ controls.update(); renderer.render(scene,camera); requestAnimationFrame(animate); }

function tx(x){ return x - M.width/2; }
function tz(y){ return -(y - M.height/2); }

function clearMap(){ if(!mapGroup)return; while(mapGroup.children.length) mapGroup.remove(mapGroup.children[0]);
 robot=null; fovMesh=null; trailLine=null; trailPts=[]; discovered={}; reveal=null; }

function buildMap(m){
 M=m; clearMap();
 // floor
 const floor=new THREE.Mesh(new THREE.PlaneGeometry(M.width,M.height),
   new THREE.MeshStandardMaterial({color:0x111316,roughness:0.95,metalness:0}));
 floor.rotation.x=-Math.PI/2; floor.position.set(0,0,0); mapGroup.add(floor);
 const grid=new THREE.GridHelper(Math.max(M.width,M.height),Math.max(M.width,M.height),0x223,0x181a1e);
 grid.position.y=0.01; mapGroup.add(grid);
 // walls
 const wmat=new THREE.MeshStandardMaterial({color:0x3a3d44,roughness:0.8,metalness:0.05});
 for(const w of (M.walls||[])){
   const g=new THREE.BoxGeometry(w.L, WALL_H, Math.max(0.12,w.T));
   const mesh=new THREE.Mesh(g,wmat); mesh.position.set(tx(w.cx),WALL_H/2,tz(w.cy));
   mesh.rotation.y=w.th; mapGroup.add(mesh);
 }
 // home pad
 const home=m.home;
 const hp=new THREE.Mesh(new THREE.RingGeometry(0.35,0.55,24),
   new THREE.MeshBasicMaterial({color:0x76d6ff,side:THREE.DoubleSide}));
 hp.rotation.x=-Math.PI/2; hp.position.set(tx(home[0]),0.02,tz(home[1])); mapGroup.add(hp);
 mapGroup.add(makeLabel('HOME', tx(home[0]), 1.2, tz(home[1]), '#76d6ff'));
 // route (house)
 if(m.waypoints&&m.waypoints.length>1){
   const pts=m.waypoints.map(p=>new THREE.Vector3(tx(p[0]),0.03,tz(p[1])));
   const rl=new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
     new THREE.LineBasicMaterial({color:0xcdbbff,transparent:true,opacity:0.35}));
   mapGroup.add(rl);
 }
 // goal (nav)
 if(m.goal){
   const gp=new THREE.Mesh(new THREE.RingGeometry(0.3,0.5,24),
     new THREE.MeshBasicMaterial({color:0xcdbbff,side:THREE.DoubleSide}));
   gp.rotation.x=-Math.PI/2; gp.position.set(tx(m.goal[0]),0.02,tz(m.goal[1])); mapGroup.add(gp);
   mapGroup.add(makeLabel('GOAL', tx(m.goal[0]),1.0,tz(m.goal[1]),'#cdbbff'));
 }
 // robot
 robot=new THREE.Group();
 const rr=Math.max(0.16,m.robot_radius);
 const body=new THREE.Mesh(new THREE.CylinderGeometry(rr,rr,0.42,20),
   new THREE.MeshStandardMaterial({color:0xdc2626,roughness:0.5,metalness:0.2}));
 body.position.y=0.23; robot.add(body);
 const nose=new THREE.Mesh(new THREE.ConeGeometry(rr*0.55,rr*1.4,16),
   new THREE.MeshStandardMaterial({color:0xff7a7a}));
 nose.rotation.z=-Math.PI/2; nose.position.set(rr*1.05,0.30,0); robot.add(nose);
 // FOV wedge (house)
 if(m.kind==='house'){
   const fov=m.cam_fov||1.03, rng=m.cam_range||4;
   const cg=new THREE.CircleGeometry(rng, 28, -fov/2, fov);
   fovMesh=new THREE.Mesh(cg, new THREE.MeshBasicMaterial({color:0x76d6ff,transparent:true,opacity:0.10,side:THREE.DoubleSide}));
   fovMesh.rotation.x=-Math.PI/2; fovMesh.position.y=0.04; robot.add(fovMesh);
 }
 robot.position.set(tx(m.start[0]),0,tz(m.start[1])); robot.rotation.y=0;
 mapGroup.add(robot);
 trailPts=[new THREE.Vector3(tx(m.start[0]),0.05,tz(m.start[1]))];
 trailLine=new THREE.Line(new THREE.BufferGeometry().setFromPoints(trailPts),
   new THREE.LineBasicMaterial({color:0xff4444})); mapGroup.add(trailLine);
 controls.target.set(0,0,0); camera.position.set(M.width*0.05, Math.max(M.width,M.height)*1.15, M.height*0.95);
 lastFrame=null; drawScopes(null);                  // 맵 새로 로드 시 센서 모니터 초기화
}

function makeLabel(text,x,y,z,color){
 const c=document.createElement('canvas'); c.width=128; c.height=40; const g=c.getContext('2d');
 g.fillStyle=color||'#fff'; g.font='bold 26px sans-serif'; g.textAlign='center'; g.fillText(text,64,28);
 const tex=new THREE.CanvasTexture(c); const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:tex,transparent:true}));
 sp.position.set(x,y,z); sp.scale.set(1.6,0.5,1); return sp;
}
function itemColor(f){ return ICOL[(f||{}).color] || 0x9ca3af; }
function addItem(it){
 const x=tx(it.x), z=tz(it.y);
 const box=new THREE.Mesh(new THREE.BoxGeometry(0.28,0.34,0.28),
   new THREE.MeshStandardMaterial({color:itemColor(it.features),roughness:0.5,emissive:itemColor(it.features),emissiveIntensity:0.15}));
 box.position.set(x,0.17,z); box.userData.item=true; mapGroup.add(box);
 mapGroup.add(makeLabel((it.features||{}).label||'', x,0.7,z,'#dcdcdc'));
 if(it.is_target){ const ring=new THREE.Mesh(new THREE.TorusGeometry(0.42,0.04,8,24),
   new THREE.MeshBasicMaterial({color:0xcdbbff})); ring.rotation.x=-Math.PI/2; ring.position.set(x,0.05,z);
   ring.userData.item=true; mapGroup.add(ring); }
}
function updateRobot(fr){
 if(!robot)return;
 robot.position.set(tx(fr.x),0,tz(fr.y)); robot.rotation.y=fr.h;
 trailPts.push(new THREE.Vector3(tx(fr.x),0.05,tz(fr.y))); if(trailPts.length>5000)trailPts.shift();
 trailLine.geometry.setFromPoints(trailPts);
 if(fr.scan){ for(const d of fr.scan){ const wa=fr.h+d.bearing;
   const ix=fr.x+d.distance*Math.cos(wa), iy=fr.y+d.distance*Math.sin(wa);
   const key=(d.features.label||'')+'|'+(d.features.color||'');
   if(!discovered[key]){ discovered[key]=1;
     addItem({x:ix,y:iy,features:d.features,
       is_target:M.objective&&Object.keys(M.objective).every(k=>d.features[k]===M.objective[k])}); } } }
 lastFrame=fr; drawScopes(fr);                       // 별도 LiDAR·카메라 모니터 갱신
}

// ── UI: tasks/models ──
function badge(state){ const cls=({up:'up',loading:'loading',stopping:'loading',error:'error'})[state]||'down';
 const txt=({up:'서빙 중',loading:'로딩…',stopping:'정리…',error:'오류',down:'내려감','-':'mock'})[state]||state;
 return {cls,txt}; }
function setMState(state){ const e=document.getElementById('mstate'); const b=badge(state);
 e.className='st '+b.cls; e.textContent=b.txt; }
function setBState(state){ const e=document.getElementById('bstate'); if(!e)return;
 if(!state){ e.className='st down'; e.textContent='-'; return; } const b=badge(state);
 e.className='st '+b.cls; e.textContent=b.txt; }
function curModel(){ return MODELS.find(m=>m.key===document.getElementById('model').value); }
function curBrain(){ const v=document.getElementById('brain').value; return v?MODELS.find(m=>m.key===v):null; }
document.getElementById('brain').onchange=()=>{ const b=curBrain(); setBState(b?b.state:null); };
function fillPar(){
 const m=curModel(); const dpS=document.getElementById('dp'), ppS=document.getElementById('pp');
 const opts=(s,vals,cur,dis)=>{ s.innerHTML=''; vals.forEach(v=>{const o=document.createElement('option');
   o.value=v;o.textContent=v;if(v==cur)o.selected=true;s.appendChild(o);}); s.disabled=dis; };
 if(!m||m.kind==='mock'){ opts(dpS,[1],1,true); opts(ppS,[1],1,true);
   document.getElementById('parhint').textContent='mock 은 서버 없이 흐름만 봅니다(dp/pp 무관).'; setMState('-'); return; }
 if(m.kind==='tp32'){ opts(dpS,[1],1,true); opts(ppS,[1],1,true);
   document.getElementById('parhint').textContent='tp32 모델 — 카드 4장 고정(dp/pp 설정 없음).'; }
 else if(m.pp_fixed){ opts(dpS,[1],1,true); opts(ppS,[m.pp_fixed],m.pp_fixed,true);
   document.getElementById('parhint').textContent='이 모델은 pp 가 '+m.pp_fixed+' 으로 고정입니다(레이어분할 '+m.pp_fixed+'장).'; }
 else { opts(dpS,[1,2,4],m.dp||1,false); opts(ppS,[1,2,4],m.pp||1,false);
   document.getElementById('parhint').textContent='tp8 — dp(복제)×pp(레이어분할) ≤ 4장. 카드를 넘으면 서버가 자동 보정합니다.'; }
 setMState(m.state);
}
document.getElementById('model').onchange=fillPar;
document.getElementById('dp').onchange=()=>{ const dp=+document.getElementById('dp').value, pp=+document.getElementById('pp').value;
 if(dp*pp>4){ document.getElementById('pp').value=Math.max(1,Math.floor(4/dp)); } };
document.getElementById('pp').onchange=()=>{ const dp=+document.getElementById('dp').value, pp=+document.getElementById('pp').value;
 if(dp*pp>4){ document.getElementById('dp').value=Math.max(1,Math.floor(4/pp)); } };

fetch('/api/tasks').then(r=>r.json()).then(d=>{
 MODELS=d.models; const ms=document.getElementById('model'), bs=document.getElementById('brain');
 d.models.forEach(m=>{ const o=document.createElement('option'); o.value=m.key;
   o.textContent=m.label+(m.kind!=='mock'?('  ·  '+m.kind):''); ms.appendChild(o); });
 // 로봇 두뇌 드롭다운: '(없음)' + 같은 모델들(작은 모델 권장)
 const none=document.createElement('option'); none.value=''; none.textContent='(없음 — 코더에 직접, 단일 LLM)'; bs.appendChild(none);
 d.models.forEach(m=>{ const o=document.createElement('option'); o.value=m.key;
   o.textContent=m.label+(m.kind!=='mock'?('  ·  '+m.kind):''); bs.appendChild(o); });
 // 기본값: 떠 있는 모델이 있으면 그것, 없으면 첫 모델
 const upM=d.models.find(m=>m.state==='up'); if(upM) ms.value=upM.key;
 fillPar(); setBState(null);
 const tdiv=document.getElementById('tasks');
 d.tasks.forEach(t=>{ const el=document.createElement('div'); el.className='task';
   el.innerHTML='<b>'+t.title+'</b><p>'+t.desc+'</p>';
   el.onclick=()=>{ if(running)return; sel=t.id;
     [...tdiv.children].forEach(c=>c.classList.remove('sel')); el.classList.add('sel');
     document.getElementById('go').disabled=false; loadMap(t.id); };
   tdiv.appendChild(el); });
 if(!d.have_mgr){ document.getElementById('parhint').innerHTML=
   '<span style="color:#f59e0b">모델 매니저(chat_app) 로드 실패 — mock 만 가능합니다.</span> '+(d.mgr_error||''); }
});
// serve 상태 주기적 갱신(로딩/업 표시)
setInterval(()=>{ if(running)return; fetch('/api/serve_status').then(r=>r.json()).then(d=>{
  MODELS=d.models; const m=curModel(); if(m) setMState(m.state);
  const b=curBrain(); setBState(b?b.state:null); }); }, 4000);

function loadMap(taskId){ fetch('/api/map?task='+encodeURIComponent(taskId)).then(r=>r.json()).then(d=>{
  if(d.error)return; buildMap(d); }); }

function addEv(cls,ksub,html){ const log=document.getElementById('log');
 const el=document.createElement('div'); el.className='ev '+cls;
 el.innerHTML='<div class="k">'+ksub+'</div>'+html; log.appendChild(el); log.scrollTop=log.scrollHeight; }
function esc(s){ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

document.getElementById('go').onclick=()=>{
 if(!sel)return; const model=document.getElementById('model').value;
 const brain=document.getElementById('brain').value||'';
 const dp=+document.getElementById('dp').value||1, pp=+document.getElementById('pp').value||1;
 running=true; runlog={codes:[], seq:[]};
 document.getElementById('log').innerHTML=''; document.getElementById('resultPanel').style.display='none';
 document.getElementById('summary').innerHTML='';
 document.getElementById('go').disabled=true; document.getElementById('stop').disabled=false;
 [...document.getElementById('tasks').children].forEach(c=>c.style.pointerEvents='none');
 document.getElementById('status').innerHTML='시작 — '+(brain?('로봇두뇌('+esc(brain)+') + '):'')+'코더('+esc(model)+') 준비…';
 if(es) es.close(); es=new EventSource('/api/stream'); es.onmessage=(e)=>handle(JSON.parse(e.data));
 fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({task:sel,model:model,dp:dp,pp:pp,brain:brain})});
};
document.getElementById('stop').onclick=()=>{
 // 즉시 멈춤: 서버 정리를 기다리지 않고 스트림을 끊고 화면을 멈춘다(중단이 바로 체감되게).
 fetch('/api/stop',{method:'POST'});
 if(es){es.close();es=null;}
 document.getElementById('status').innerHTML='■ 중단했습니다.';
 endRun();
};
function endRun(){ running=false; document.getElementById('stop').disabled=true; document.getElementById('go').disabled=false;
 [...document.getElementById('tasks').children].forEach(c=>c.style.pointerEvents='auto'); if(es){es.close();es=null;} }

function handle(ev){
 switch(ev.type){
  case 'map': buildMap(ev); break;
  case 'status': document.getElementById('status').innerHTML=esc(ev.text); break;
  case 'serve': {
    const isBrain=ev.role==='brain'; const st=(ev.state==='request'?'loading':ev.state);
    if(isBrain) setBState(st); else setMState(st);
    addEv('serve',(isBrain?'🧠 로봇 두뇌':'🖥 서버 코더')+' serve · '+esc(ev.model)+'  (dp='+ev.dp+'·pp='+ev.pp+')',
      '<b>'+esc(badge(ev.state).txt)+'</b>'+(ev.state==='up'?' — 준비 완료':'')); break; }
  case 'brain':
    runlog.seq.push('🧠 두뇌('+(ev.phase==='initial'?'첫 지시':'수리 지시'+(ev.reason?(' · '+ev.reason):''))+')');
    document.getElementById('status').innerHTML='🧠 로봇 두뇌가 코더에게 보낼 지시를 자연어로 작성했습니다…';
    addEv('brain','🧠 로봇 두뇌 → 서버 코더  ('+(ev.phase==='initial'?'첫 지시':'수리 지시')+
      (ev.metrics?('  ·  '+ev.metrics.tokens+'토큰'):'')+')',
      '<div style="white-space:pre-wrap;color:#cdbbff;font-size:11.5px">'+esc(ev.text||'')+'</div>'); break;
  case 'request':
    runlog.seq.push(ev.phase==='initial'?'🟦 첫 코드 요청':('🛠 수리 요청 #'+ev.attempt+(ev.reason?(' — '+ev.reason):'')));
    document.getElementById('status').innerHTML=(ev.phase==='initial'?'첫 코드를 NPU 에 요청 중…':('수리 요청('+ev.attempt+'): '+esc(ev.reason||'')));
    addEv('req',(ev.phase==='initial'?'NPU 요청 · 첫 코드':('NPU 요청 · 수리 #'+ev.attempt+(ev.reason?(' ('+ev.reason+')'):''))),
      '<details><summary>보낸 프롬프트 보기</summary><pre>'+esc((ev.prompt||'').slice(0,1400))+'</pre></details>'); break;
  case 'code':
    runlog.codes.push({attempt:ev.attempt, code:ev.code||'', metrics:ev.metrics});
    addEv('code','NPU 가 보낸 코드 (시도 '+ev.attempt+'  ·  '+(ev.metrics?('TTFT '+ev.metrics.ttft_s+'s · '+ev.metrics.tps+' tok/s · '+ev.metrics.tokens+'토큰'):'')+')',
      '<pre>'+esc(ev.code||'')+'</pre>'); break;
  case 'build':
    if(ev.ok) addEv('ok','코드 빌드 OK (시도 '+ev.attempt+')','<b style="color:#34d399">샌드박스 적재 성공 — 이 코드로 움직입니다.</b>');
    else addEv('fail','코드 빌드 실패 (시도 '+ev.attempt+')','<b>'+esc(ev.error||'')+'</b><div class="hint">→ NPU 에 고쳐 달라고 다시 요청합니다.</div>'); break;
  case 'frame':
    updateRobot(ev);
    if(ev.phase) document.getElementById('status').innerHTML=(ev.phase==='home'?'복귀 중…':'집 안 탐색 중…')+(ev.found?'  (목표 발견!)':''); break;
  case 'failure':
    runlog.seq.push('❌ 주행 실패 — '+ev.reason);
    addEv('fail','주행 실패 → 자가수리 ('+ev.replan+'/'+ev.max_replans+')',
      '<b>'+esc(ev.reason)+'</b><div class="hint">'+esc(ev.detail||'')+'</div>'); break;
  case 'result': showResult(ev); break;
  case 'saved': addEv('ok','결과 정리 저장됨','<div class="hint">robot-sim/<b>'+esc(ev.md)+'</b> (사람용)<br>robot-sim/'+esc(ev.json)+' (JSON)</div>');
    window.__savedPaths={md:ev.md,json:ev.json}; break;
  case 'error': addEv('fail','오류','<b>'+esc(ev.text)+'</b>'); document.getElementById('status').innerHTML='오류: '+esc(ev.text); break;
  case 'end': endRun(); break;
 }
}
function lineDiff(a,b){            // 간단한 LCS 라인 diff
 const A=a.split('\n'), B=b.split('\n'), m=A.length, n=B.length;
 const dp=Array.from({length:m+1},()=>new Array(n+1).fill(0));
 for(let i=m-1;i>=0;i--)for(let j=n-1;j>=0;j--)
   dp[i][j]=A[i]===B[j]?dp[i+1][j+1]+1:Math.max(dp[i+1][j],dp[i][j+1]);
 let i=0,j=0,out=[];
 while(i<m&&j<n){ if(A[i]===B[j]){out.push(['ctx',A[i]]);i++;j++;}
   else if(dp[i+1][j]>=dp[i][j+1]){out.push(['del',A[i]]);i++;} else {out.push(['add',B[j]]);j++;} }
 while(i<m)out.push(['del',A[i++]]); while(j<n)out.push(['add',B[j++]]);
 return out;
}
function renderDiff(a,b){
 const d=lineDiff(a,b); const changed=d.filter(x=>x[0]!=='ctx').length;
 const body=d.map(([t,l])=>{ const c=t==='del'?'#ff7676':t==='add'?'#34d399':'#7a7a7a';
   const p=t==='del'?'- ':t==='add'?'+ ':'  '; return '<div style="color:'+c+'">'+esc(p+l)+'</div>'; }).join('');
 return {changed, html:body};
}
function buildSummary(ev){
 const codes=runlog.codes; const n=codes.length; const nrep=Math.max(0,n-1);
 let h='<div class="hint" style="margin:6px 0">이 task 동안 NPU 와 주고받은 과정을 자동 정리했습니다.</div>';
 // 대화 흐름
 h+='<h4 style="margin:8px 0 4px;font-size:12px;color:#cdbbff">① NPU 와의 대화 흐름</h4>';
 h+='<div style="font-size:11.5px;line-height:1.7">'+ (runlog.seq.length?runlog.seq.map(s=>'· '+esc(s)).join('<br>'):'· (요청 없음)')
   +'<br>· '+(ev.success?'✅ 성공':'❌ 종료('+esc(ev.reason)+')')+'</div>';
 // 코드 변화
 h+='<h4 style="margin:12px 0 4px;font-size:12px;color:#cdbbff">② 시작 코드 → 최종(달성) 코드 변화</h4>';
 if(n===0){ h+='<div class="hint">받은 코드가 없습니다.</div>'; }
 else if(nrep===0){ h+='<div class="hint">첫 코드가 그대로 성공했습니다 — 수정 없음. ('+codes[0].code.split("\n").length+'줄)</div>'; }
 else {
   const d=renderDiff(codes[0].code, codes[n-1].code);
   h+='<div class="hint">코드를 '+nrep+'번 고쳐 성공했습니다. 바뀐 줄: <b>'+d.changed+'</b>개 '
     +'(<span style="color:#ff7676">- 시작 코드</span> / <span style="color:#34d399">+ 최종 코드</span>)</div>';
   h+='<pre style="max-height:220px;overflow:auto;background:#0b0b0b;border:1px solid #1a1a1a;border-radius:6px;padding:7px;font-size:11px;white-space:pre">'+d.html+'</pre>';
   h+='<details style="margin-top:6px"><summary>최종 코드 전체 보기</summary><pre style="max-height:240px;overflow:auto;font-size:11px">'+esc(codes[n-1].code)+'</pre></details>';
 }
 document.getElementById('summary').innerHTML=h;
}
function showResult(ev){
 if(ev.reveal_items){ reveal=ev.reveal_items; for(const it of ev.reveal_items){
   const key=(it.features.label||'')+'|'+(it.features.color||''); if(!discovered[key]){ discovered[key]=1; addItem(it); } } }
 const ok=ev.success;
 addEv('done'+(ok?'':' fail'),'완료','<b style="color:'+(ok?'#34d399':'#ff7676')+'">'+(ok?'✅ task 성공':'❌ 실패: '+esc(ev.reason))+'</b>');
 document.getElementById('result').innerHTML=
   kv('결과', ok?'<span style="color:#34d399">✅ 성공</span>':'<span style="color:#ff7676">❌ '+esc(ev.reason)+'</span>')
   +kv('코드 재작성(자가수리)', ev.replans+'회')+kv('총 스텝', ev.steps)+kv('LLM 호출', ev.llm_calls+'회 / '+ev.tokens+'토큰');
 buildSummary(ev);
 document.getElementById('resultPanel').style.display='block';
 document.getElementById('status').innerHTML=ok?'완료 — task 성공.':'완료 — 실패('+esc(ev.reason)+').';
}
function kv(k,v){return '<div class="kv"><span>'+k+'</span><span>'+v+'</span></div>';}

try{ init3d(); }catch(e){ document.getElementById('status').innerHTML=
  '3D 초기화 실패(인터넷에서 three.js 로드 필요): '+e.message; }
window.__live={get M(){return M;}, get robot(){return robot;}};  // 디버그/검증용
window.__dbg={buildSummary, renderDiff, setRunlog:(v)=>{runlog=v;}};
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description="라이브 인터랙티브 로봇 task 시뮬레이터(NPU LLM 실시간)")
    ap.add_argument("--port", type=int, default=7910)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    import uvicorn
    print(f"🌐 브라우저에서 열기:  http://127.0.0.1:{args.port}/")
    print(f"   맥북: alpacon tunnel furiosa-npu-e6ec40 -l {args.port} -r {args.port}  → 위 주소")
    print("   (task 를 고르면 그때부터 NPU 서버 LLM 에 요청을 시작합니다. 모델을 먼저 serve 하세요.)")
    uvicorn.run(make_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
