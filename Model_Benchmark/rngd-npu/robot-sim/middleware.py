"""ROS2식 노드 분리 + 컨테이너 간 연동 지연 측정 (SOAFEE/Autoware 매핑).

자율주행 런타임은 보통 perception → planning → control 컴포넌트를 따로 두고 미들웨어(ROS2 DDS)로
메시지를 주고받습니다. 과제 'Autoware 컴포넌트 분리·컨테이너화', '컨테이너 간 연동 성능 분석'을
시뮬레이터 안에서 재현하려고, 제어 한 사이클을 세 노드로 나눠 라우팅하고 홉 지연을 측정합니다.

  PerceptionNode  : 센서(LiDAR+pose) 관측을 /sensors 토픽으로 발행
  PlanningNode    : /sensors 를 받아 LLM이 짠 plan(state) 컨트롤러를 실행 → /cmd_vel 발행  ← '우리 LLM 서비스'
  ControlNode     : /cmd_vel 을 받아 구동(월드 step)에 넘길 (v, w) 를 확정

두 가지 실행 모드
  * sync   : 한 프로세스에서 함수호출로 직렬 실행(전송지연 ~0). 파이프라인 구조만 보고 싶을 때.
  * threaded: 각 노드를 별도 스레드+큐로 분리해 진짜 큐잉/문맥전환 지연을 측정.
              스레드 ≈ 컨테이너/프로세스 격리의 1차 근사입니다.

실제 컨테이너(ROS2 rclpy + DDS, 또는 SOAFEE 분리 배포)로 바꿀 자리는 _planning_loop /
_control_loop 의 큐 입출력 지점입니다(여기를 rclpy publisher/subscriber 로 교체).
측정 지표는 홉별 지연(ms)과 사이클 주파수(Hz)로, '컨테이너 분리에 따른 성능' 분석에 씁니다.
"""
from __future__ import annotations

import queue
import threading
from time import perf_counter as _now
from typing import Callable, Optional, Tuple

from executor import call_with_timeout, normalize_action


class MiddlewarePipeline:
    """agent.NavAgent 가 쓰는 ControlPipeline 인터페이스 구현(노드 분리 버전).
    cycle(state, plan_fn) -> (v, w) 한 번이 perception→planning→control 한 사이클입니다."""

    HOPS = ("hop_sense_plan", "plan_compute", "hop_plan_ctrl", "hop_ctrl_done", "cycle")

    def __init__(self, plan_timeout: float = 0.5, threaded: bool = False):
        self.plan_timeout = plan_timeout
        self.threaded = threaded
        self._plan_fn: Optional[Callable] = None
        self._acc = {k: 0.0 for k in self.HOPS}
        self._n = 0
        if threaded:
            self._sensor_q: "queue.Queue" = queue.Queue()
            self._cmd_q: "queue.Queue" = queue.Queue()
            self._resp_q: "queue.Queue" = queue.Queue()
            self._stop = False
            self._tp = threading.Thread(target=self._planning_loop, daemon=True)
            self._tc = threading.Thread(target=self._control_loop, daemon=True)
            self._tp.start()
            self._tc.start()

    # ── 공통 ──────────────────────────────────────────────────────
    def reset(self):
        self._acc = {k: 0.0 for k in self.HOPS}
        self._n = 0

    def _run_plan(self, state: dict) -> Tuple[float, float]:
        action = call_with_timeout(self._plan_fn, state, self.plan_timeout)
        return normalize_action(action, state["v_max"], state["w_max"],
                                state["heading"], state["dt"])

    def _accum(self, t_sense, t_recv, t_cmd, t_ctrl, t0, t_end):
        self._acc["hop_sense_plan"] += (t_recv - t_sense)   # perception → planning
        self._acc["plan_compute"] += (t_cmd - t_recv)       # planning(컨트롤러 실행)
        self._acc["hop_plan_ctrl"] += (t_ctrl - t_cmd)      # planning → control
        self._acc["hop_ctrl_done"] += (t_end - t_ctrl)      # control → cycle 복귀(resp 큐 홉)
        self._acc["cycle"] += (t_end - t0)                  # 사이클 전체
        self._n += 1

    def cycle(self, state: dict, plan_fn: Callable) -> Tuple[float, float]:
        self._plan_fn = plan_fn
        return self._cycle_threaded(state) if self.threaded else self._cycle_sync(state)

    # ── sync: 직렬 함수호출 ───────────────────────────────────────
    def _cycle_sync(self, state: dict) -> Tuple[float, float]:
        t0 = _now()
        t_sense = _now()                 # PerceptionNode: /sensors 발행
        t_recv = _now()                  # PlanningNode 수신(인프로세스 ~0)
        v, w = self._run_plan(state)     # PlanningNode: 컨트롤러 실행
        t_cmd = _now()                   # PlanningNode: /cmd_vel 발행
        t_ctrl = _now()                  # ControlNode 수신
        self._accum(t_sense, t_recv, t_cmd, t_ctrl, t0, _now())
        return v, w

    # ── threaded: 노드별 스레드+큐(진짜 전송지연) ─────────────────
    def _planning_loop(self):
        while not self._stop:
            try:
                msg = self._sensor_q.get(timeout=0.2)
            except queue.Empty:
                continue
            t_recv = _now()
            try:
                v, w = self._run_plan(msg["state"])
                err = None
            except BaseException as e:   # noqa: BLE001 — 예외도 그대로 control 거쳐 cycle 로 전달
                v = w = None
                err = e
            self._cmd_q.put({"v": v, "w": w, "err": err,
                             "t_sense": msg["t_sense"], "t_recv": t_recv, "t_cmd": _now()})

    def _control_loop(self):
        while not self._stop:
            try:
                msg = self._cmd_q.get(timeout=0.2)
            except queue.Empty:
                continue
            msg["t_ctrl"] = _now()
            self._resp_q.put(msg)

    def _cycle_threaded(self, state: dict) -> Tuple[float, float]:
        t0 = _now()
        self._sensor_q.put({"state": state, "t_sense": _now()})
        msg = self._resp_q.get()
        self._accum(msg["t_sense"], msg["t_recv"], msg["t_cmd"], msg["t_ctrl"], t0, _now())
        if msg["err"] is not None:
            raise msg["err"]
        return msg["v"], msg["w"]

    # ── 통계/정리 ─────────────────────────────────────────────────
    def summary(self) -> dict:
        if not self._n:
            return {"hops": {}, "cycle_hz": 0.0}
        hops = {k: self._acc[k] / self._n * 1000.0
                for k in ("hop_sense_plan", "plan_compute", "hop_plan_ctrl", "hop_ctrl_done")}
        cycle_ms = self._acc["cycle"] / self._n * 1000.0
        return {"hops": hops, "cycle_hz": (1000.0 / cycle_ms if cycle_ms else 0.0),
                "cycle_ms": cycle_ms, "mode": "threaded" if self.threaded else "sync"}

    def close(self):
        if self.threaded:
            self._stop = True
