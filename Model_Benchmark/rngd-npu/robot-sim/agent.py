"""폐루프 내비게이션 에이전트 — '로봇이 LLM에 코드를 물어보고, 받은 코드를 자기 자신에게
적용해 목표까지 스스로 찾아가는' 과정을 한 에피소드로 실행합니다.

흐름(코딩 성능 검증의 핵심)
  1) 현재 위치·목표·LiDAR 를 적어 LLM에게 plan(state) 컨트롤러 코드를 요청한다.
  2) 받은 코드를 샌드박스에서 실행해 plan 함수를 얻는다(빌드 실패 = 코드 결함 → 수리).
  3) 그 컨트롤러로 월드를 굴린다(매 스텝 plan 호출 → v,w → world.step).
  4) 충돌/정체/예외/스텝초과로 실패하면, 무슨 일이 있었는지 적어 LLM에게 '고쳐 달라'고
     다시 요청한다(self-debug). 목표 도달 또는 재작성 한도까지 반복한다.

제어 한 사이클(observe→plan→control)은 ControlPipeline 으로 추상화돼, 기본은 인프로세스
직접 실행이고 middleware 모드에선 ROS2식 노드 분리로 라우팅돼 홉 지연을 측정합니다.
"""
from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from executor import build_plan, call_with_timeout, extract_code, normalize_action
from metrics import EpisodeResult
from prompts import (SYSTEM, initial_user, repair_user,
                     VISION_SYSTEM, vision_initial_user, vision_repair_user,
                     HOUSE_SYSTEM, house_initial_user, house_repair_user)
from world import Robot, World


# ── 제어 파이프라인(기본: 직접 실행) ─────────────────────────────
class DirectPipeline:
    """observe→plan→정규화를 인프로세스에서 바로 실행. 미들웨어 분석이 필요 없을 때 사용."""

    def __init__(self, plan_timeout: float = 0.5):
        self.plan_timeout = plan_timeout
        self._compute_s = 0.0
        self._n = 0

    def reset(self):
        self._compute_s = 0.0
        self._n = 0

    def cycle(self, state: dict, plan_fn: Callable) -> Tuple[float, float]:
        t0 = time.perf_counter()
        action = call_with_timeout(plan_fn, state, self.plan_timeout)
        v, w = normalize_action(action, state["v_max"], state["w_max"],
                                state["heading"], state["dt"])
        self._compute_s += time.perf_counter() - t0
        self._n += 1
        return v, w

    def summary(self) -> dict:
        avg_ms = (self._compute_s / self._n * 1000.0) if self._n else 0.0
        return {"hops": {"plan_compute": avg_ms}, "cycle_hz": (1000.0 / avg_ms if avg_ms else 0.0)}


@dataclass
class _Outcome:
    reason: str          # goal / collision / out_of_bounds / stuck / step_budget / no_valid_code / exception
    detail: str
    state: dict
    success: bool = False


# ── 에이전트 ──────────────────────────────────────────────────────
class NavAgent:
    def __init__(self, llm, max_steps: int = 1200, max_replans: int = 4,
                 stuck_window: int = 90, plan_timeout: float = 0.5,
                 temperature: float = 0.2, max_tokens: int = 1200,
                 pipeline=None, verbose: bool = True, log=print, frame_cb=None,
                 on_event=None):
        self.llm = llm
        self.max_steps = max_steps
        self.max_replans = max_replans
        self.stuck_window = stuck_window
        self.plan_timeout = plan_timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.pipeline = pipeline or DirectPipeline(plan_timeout)
        self.verbose = verbose
        self.log = log
        self.frame_cb = frame_cb   # 시각화용: 매 제어주기마다 (robot, state) 를 받아 프레임 기록
        self.on_event = on_event   # 라이브 스트리밍용: 요청/코드/빌드/실패/수리/완료 이벤트 콜백(없으면 무동작)

    def _event(self, kind: str, **data):
        """라이브 뷰어로 보낼 이벤트(없으면 no-op — 회귀 영향 0)."""
        if self.on_event:
            try:
                self.on_event({"type": kind, **data})
            except Exception:
                pass

    @staticmethod
    def _m(m):
        """LLM 호출 지표를 이벤트용 dict 로."""
        return {"ttft_s": round(getattr(m, "ttft_s", 0.0), 2),
                "tps": round(getattr(m, "tps", 0.0), 1),
                "tokens": int(getattr(m, "completion_tokens", 0)),
                "ok": bool(getattr(m, "ok", True)),
                "error": getattr(m, "error", "")}

    # ── 한 컨트롤러로 월드를 굴려 본다(한 번의 롤아웃) ────────────
    def _rollout(self, world: World, robot: Robot, goal, plan_fn, budget: int,
                 res: EpisodeResult, prog: dict) -> _Outcome:
        # prog = {"best": 목표까지 최단거리, "since": 무진전 스텝수} — 에피소드 전체에서 누적(롤아웃마다
        # 새로 초기화하지 않음). 그래야 같은 지역최소를 여러 컨트롤러로 맴돌아도 결국 'stuck' 으로 잡힌다.
        memory: dict = {}
        self.pipeline.reset()
        near_wrong = 0                      # 엉뚱한 사람 근처에 머문 연속 스텝(사람찾기 과제)
        for _ in range(budget):
            state = world.observe(robot, goal, memory)
            if self.frame_cb:
                self.frame_cb(robot, state)
            try:
                v, w = self.pipeline.cycle(state, plan_fn)
            except Exception as e:
                res.exceptions += 1
                tb = traceback.format_exc(limit=3).strip().splitlines()
                detail = ("plan() raised an exception while running: "
                          + (tb[-1] if tb else str(e)))
                return _Outcome("exception", detail, state)

            px, py = robot.x, robot.y
            collided = world.step(robot, v, w)
            res.steps += 1
            res.path_length += math.hypot(robot.x - px, robot.y - py)
            res.path.append((round(robot.x, 3), round(robot.y, 3)))
            res.min_clearance = min(res.min_clearance, world.clearance(robot.x, robot.y))

            if world.goal_reached(robot, goal):
                return _Outcome("goal", "reached the goal", state, success=True)

            if collided:
                res.collisions += 1
                in_b = world.in_bounds(robot.x, robot.y)
                reason = "collision" if in_b else "out_of_bounds"
                detail = (f"The robot {'hit an obstacle' if in_b else 'drove out of bounds'} "
                          f"at ({robot.x:.2f}, {robot.y:.2f}) after {res.steps} steps.")
                return _Outcome(reason, detail, state)

            # 사람찾기 과제: 목표를 본 적 있는지 기록 + 엉뚱한 사람에 '정착'하면 wrong_person
            if world.vision_task:
                if world.target_in_view(robot):
                    prog["target_seen"] = True
                near = any(not world.matches_target(p)
                           and math.hypot(robot.x - p.x, robot.y - p.y) <= world.goal_tol
                           for p in world.people)
                near_wrong = near_wrong + 1 if near else 0
                if near_wrong >= 20:           # 2초 이상 엉뚱한 사람 옆에 머묾 = 오인 도착
                    wp = min((p for p in world.people if not world.matches_target(p)),
                             key=lambda p: math.hypot(robot.x - p.x, robot.y - p.y))
                    detail = (f"The robot stopped at the WRONG person (features {wp.features}); "
                              f"the target must match ALL of {world.target}.")
                    return _Outcome("wrong_person", detail, state)

            d = math.hypot(robot.x - goal[0], robot.y - goal[1])
            if d < prog["best"] - 0.05:                 # 목표에 더 다가감 → 진전
                prog["best"] = d
                prog["since"] = 0
                prog["anchor"] = (robot.x, robot.y)
            else:
                prog["since"] += 1
            if prog["since"] >= self.stuck_window:
                # 한동안 목표에 더 못 다가갔다. 단, 앵커에서 멀리 이동했다면 '우회 중'(벽타기 등)이라
                # 갇힌 게 아니므로 유예를 준다. 앵커 근처에서 맴돌기만 할 때(이동량 작음)만 stuck.
                net = math.hypot(robot.x - prog["anchor"][0], robot.y - prog["anchor"][1])
                if net < 2.0:
                    if world.vision_task and not prog.get("target_seen"):
                        detail = ("The robot never found the target person — it likely never came into the "
                                  "camera's forward cone, and the robot stopped exploring/searching.")
                        return _Outcome("lost_target", detail, state)
                    detail = (f"The robot got stuck near ({robot.x:.2f}, {robot.y:.2f}); it stopped "
                              f"making progress toward the goal (best remaining distance {prog['best']:.2f} m). "
                              "It is likely caught in a local minimum (e.g. a concave/U-shaped wall).")
                    return _Outcome("stuck", detail, state)
                prog["since"] = 0                       # 이동은 하고 있으니 더 지켜본다(우회 유예)
                prog["anchor"] = (robot.x, robot.y)

        state = world.observe(robot, goal, memory)
        return _Outcome("step_budget", f"ran out of the {budget}-step budget without reaching the goal.", state)

    # ── 한 에피소드(초기 코드 → 굴리기 → 수리 반복) ──────────────
    def run_episode(self, world: World, start, goal, scenario_name: str) -> EpisodeResult:
        res = EpisodeResult(scenario=scenario_name, model=getattr(self.llm, "name", "llm"))
        res.straight_dist = math.hypot(goal[0] - start[0], goal[1] - start[1])
        # 일반 과제는 목표 방향을 바라보고 시작. 사람찾기(vision)는 목표 좌표를 모르므로 정면(+x) 고정.
        h0 = 0.0 if world.vision_task else math.atan2(goal[1] - start[1], goal[0] - start[0])
        robot = Robot(float(start[0]), float(start[1]), heading=h0)

        # 과제 종류에 따라 프롬프트 세트 선택(일반 내비 vs 카메라 사람찾기)
        vis = world.vision_task
        sys_p = VISION_SYSTEM if vis else SYSTEM
        init_fn = vision_initial_user if vis else initial_user
        repair_fn = vision_repair_user if vis else repair_user

        # 1) 초기 코드 요청
        state0 = world.observe(robot, goal, {})
        self._event("request", phase="initial", attempt=0, prompt=init_fn(state0))
        text, m = self.llm.complete(sys_p, init_fn(state0),
                                    temperature=self.temperature, max_tokens=self.max_tokens)
        self._tally_llm(res, m)
        if not m.ok:
            res.terminate_reason = "llm_error"
            self._event("done", success=False, reason="llm_error", error=m.error)
            if self.verbose:
                self.log(f"  [{scenario_name}] LLM 호출 실패: {m.error}")
            return res

        prev_code = extract_code(text)
        self._event("code", attempt=0, code=prev_code, metrics=self._m(m))
        plan_fn = None
        outcome: Optional[_Outcome] = None
        try:
            plan_fn, prev_code = build_plan(prev_code)
            res.code_valid_first = True
            self._event("build", ok=True, attempt=0)
        except Exception as e:
            res.code_valid_first = False
            outcome = _Outcome("no_valid_code",
                               f"The previous code failed to load: {type(e).__name__}: {e}", state0)
            self._event("build", ok=False, attempt=0, error=f"{type(e).__name__}: {e}")
            if self.verbose:
                self.log(f"  [{scenario_name}] 첫 코드 빌드 실패 → 수리 요청")

        # 2) 굴리기 + 수리 반복
        prog = {"best": res.straight_dist, "since": 0,   # 에피소드 단위 진전 추적(롤아웃 사이 유지)
                "anchor": (robot.x, robot.y)}
        last_rollout: Optional[_Outcome] = None           # 실제로 굴린 결과만 기억(빌드실패와 구분)
        budget_left = self.max_steps
        replan = 0
        while True:
            if plan_fn is not None:
                outcome = self._rollout(world, robot, goal, plan_fn, budget_left, res, prog)
                last_rollout = outcome
                budget_left = self.max_steps - res.steps
                if outcome.success:
                    res.success = True
                    res.terminate_reason = "goal"
                    self._event("done", success=True, reason="goal", detail=outcome.detail)
                    break
                self._event("failure", reason=outcome.reason, detail=outcome.detail,
                            steps=res.steps, replan=replan + 1, max_replans=self.max_replans)
                if self.verbose:
                    self.log(f"  [{scenario_name}] 롤아웃 실패: {outcome.reason} "
                             f"(steps={res.steps}) → 수리 {replan+1}/{self.max_replans}")

            if replan >= self.max_replans or budget_left <= 0:
                # 마지막 '실제 롤아웃' 사유를 우선(막판 수리코드 빌드실패가 진짜 사유를 덮어쓰지 않게).
                # 컨트롤러가 한 번도 안 굴렀으면 no_valid_code/llm_error 로 둔다.
                res.terminate_reason = (last_rollout.reason if last_rollout
                                        else (outcome.reason if outcome else "no_valid_code"))
                self._event("done", success=False, reason=res.terminate_reason)
                break

            # 수리 요청(self-debug)
            replan += 1
            res.replans = replan
            repair_prompt = repair_fn(outcome.reason, outcome.detail, outcome.state, prev_code)
            self._event("request", phase="repair", attempt=replan,
                        reason=outcome.reason, prompt=repair_prompt)
            text, m = self.llm.complete(sys_p, repair_prompt,
                                        temperature=self.temperature, max_tokens=self.max_tokens)
            self._tally_llm(res, m)
            if not m.ok:
                res.terminate_reason = "llm_error"
                self._event("done", success=False, reason="llm_error", error=m.error)
                break
            new_code = extract_code(text)
            self._event("code", attempt=replan, code=new_code, metrics=self._m(m))
            try:
                plan_fn, prev_code = build_plan(new_code)
                self._event("build", ok=True, attempt=replan)
            except Exception as e:
                plan_fn = None
                outcome = _Outcome("no_valid_code",
                                   f"The rewritten code failed to load: {type(e).__name__}: {e}",
                                   outcome.state)
                prev_code = new_code
                self._event("build", ok=False, attempt=replan, error=f"{type(e).__name__}: {e}")
                if self.verbose:
                    self.log(f"  [{scenario_name}] 수리 코드 빌드 실패 → 재요청")

        # 3) 미들웨어/사이클 통계 반영
        st = self.pipeline.summary()
        res.cycle_hops = st.get("hops")
        res.cycle_hz = st.get("cycle_hz", 0.0)
        return res

    # ══════════════════════════════════════════════════════════════
    #  집 안 물건 확인 후 복귀 미션 (house_task)
    # ══════════════════════════════════════════════════════════════
    def _house_rollout(self, world: World, robot: Robot, home, plan_fn, budget: int,
                       res: EpisodeResult, memory: dict, prog: dict) -> _Outcome:
        """한 컨트롤러로 미션을 굴립니다. plan() 이 {'done':True,'present':bool} 을 돌려주면
        '현관 복귀 + 판정 정확' 을 검사해 성공/실패(missed_object/false_report/not_home)를 가립니다."""
        truth = world.objective_present()
        for _ in range(budget):
            state = world.observe_house(robot, memory)
            if self.frame_cb:
                self.frame_cb(robot, state)
            try:
                action = call_with_timeout(plan_fn, state, self.plan_timeout)
            except Exception as e:
                res.exceptions += 1
                tb = traceback.format_exc(limit=3).strip().splitlines()
                return _Outcome("exception",
                                "plan() raised while running: " + (tb[-1] if tb else str(e)), state)

            # 미션 종료 선언?
            if isinstance(action, dict) and action.get("done"):
                pres = action.get("present", None)
                claim = bool(action.get("found", False)) if pres is None else bool(pres)
                at_home = math.hypot(robot.x - home[0], robot.y - home[1]) <= world.goal_tol
                if not at_home:
                    return _Outcome("not_home",
                                    f"The robot declared the mission done at ({robot.x:.2f}, {robot.y:.2f}) "
                                    f"but home is ({home[0]:.2f}, {home[1]:.2f}) — it must return home first.",
                                    state)
                if claim != truth:
                    if truth and not claim:
                        return _Outcome("missed_object",
                                        "The robot reported the object ABSENT, but it IS in the house.", state)
                    return _Outcome("false_report",
                                    "The robot reported the object PRESENT, but it is NOT in the house "
                                    "(it was likely fooled by a decoy matching only one feature).", state)
                # 판정이 정답과 같아도 '실제로 자율탐색했다는 증거'를 요구(추측 통과 방지):
                #  present 주장 → 진짜로 목표를 카메라로 봤어야(target_seen),
                #  absent  주장 → 집의 닿는 영역을 충분히(>=30%) 실제로 돌아다녔어야 함(coverage).
                cover = getattr(world, "cover_cells", None) or set()
                ncov = max(1, len(cover))
                coverage = len(prog["visited"] & cover) / ncov
                searched = prog.get("target_seen") if claim else (coverage >= 0.30)
                if not searched:
                    pct = int(coverage * 100)
                    return _Outcome("searched_too_little",
                                    (f"The robot declared the object present but never actually saw it — "
                                     "you must SEE the target before reporting present."
                                     if claim else
                                     f"The robot declared the result after exploring only {pct}% of the house "
                                     "— it did not really search before deciding."), state)
                verdict = "present" if claim else "absent"
                return _Outcome("done", f"Mission complete: correctly reported the object {verdict} "
                                        "and returned home after searching.", state, success=True)

            # 이동
            try:
                v, w = normalize_action(action, state["v_max"], state["w_max"],
                                        state["heading"], state["dt"])
            except Exception as e:
                res.exceptions += 1
                return _Outcome("exception", f"plan() returned an unusable action: {e}", state)
            px, py = robot.x, robot.y
            collided = world.step(robot, v, w)
            res.steps += 1
            res.path_length += math.hypot(robot.x - px, robot.y - py)
            res.path.append((round(robot.x, 3), round(robot.y, 3)))
            res.min_clearance = min(res.min_clearance, world.clearance(robot.x, robot.y))
            if any(world.item_matches(d["features"]) for d in state["scan"]):
                prog["target_seen"] = True
            # 실제로 돌아다닌 영역 추적(거친 1m 셀; 에이전트가 독립 계산 — 자기보고 의존 안 함)
            prog["visited"].add((int(robot.x), int(robot.y)))
            if collided:
                res.collisions += 1
                # 자율 벽추종은 가끔 벽을 스칠 수 있음 → 단발 접촉은 실패로 보지 않고 계속(world.step 이
                # 벽 앞에 세워 줌). 너무 자주 들이받으면(상한 초과) 그때 '충돌' 실패로 수리 요청.
                if res.collisions > 150:
                    return _Outcome("collision",
                                    f"The robot kept bumping into walls ({res.collisions} times) — it is "
                                    "not avoiding obstacles well.", state)
            # 정체 감지: 일정 시간 위치가 거의 안 변하면 stuck
            net = math.hypot(robot.x - prog["anchor"][0], robot.y - prog["anchor"][1])
            if net > 1.0:
                prog["anchor"] = (robot.x, robot.y)
                prog["since"] = 0
            else:
                prog["since"] += 1
            if prog["since"] >= self.stuck_window:
                return _Outcome("stuck",
                                f"The robot stopped making progress near ({robot.x:.2f}, {robot.y:.2f}); "
                                "it is not advancing through the waypoints (likely caught on a wall).", state)

        state = world.observe_house(robot, memory)
        return _Outcome("no_report",
                        f"ran out of the {budget}-step budget without ever declaring the mission done.", state)

    def run_house_episode(self, world: World, start, name: str) -> EpisodeResult:
        """집 미션 한 에피소드: 초기 컨트롤러 요청 → 굴리기 → 실패 시 자가수리 반복."""
        res = EpisodeResult(scenario=name, model=getattr(self.llm, "name", "llm"))
        home = world.home or start
        res.straight_dist = 0.0
        robot = Robot(float(start[0]), float(start[1]), heading=0.0)
        memory: dict = {}                      # 미션 진행상태(롤아웃 사이 유지 — 수리해도 이어감)
        prog = {"anchor": (robot.x, robot.y), "since": 0, "target_seen": False, "visited": set()}

        state0 = world.observe_house(robot, memory)
        self._event("request", phase="initial", attempt=0,
                    prompt=house_initial_user(state0), mission=world.mission)
        text, m = self.llm.complete(HOUSE_SYSTEM, house_initial_user(state0),
                                    temperature=self.temperature, max_tokens=self.max_tokens)
        self._tally_llm(res, m)
        if not m.ok:
            res.terminate_reason = "llm_error"
            self._event("done", success=False, reason="llm_error", error=m.error)
            if self.verbose:
                self.log(f"  [{name}] LLM 호출 실패: {m.error}")
            return res

        prev_code = extract_code(text)
        self._event("code", attempt=0, code=prev_code, metrics=self._m(m))
        plan_fn = None
        outcome: Optional[_Outcome] = None
        try:
            plan_fn, prev_code = build_plan(prev_code)
            res.code_valid_first = True
            self._event("build", ok=True, attempt=0)
        except Exception as e:
            res.code_valid_first = False
            outcome = _Outcome("no_valid_code",
                               f"The previous code failed to load: {type(e).__name__}: {e}", state0)
            self._event("build", ok=False, attempt=0, error=f"{type(e).__name__}: {e}")
            if self.verbose:
                self.log(f"  [{name}] 첫 코드 빌드 실패 → 수리 요청")

        last_rollout: Optional[_Outcome] = None
        budget_left = self.max_steps
        replan = 0
        while True:
            if plan_fn is not None:
                outcome = self._house_rollout(world, robot, home, plan_fn, budget_left, res, memory, prog)
                last_rollout = outcome
                budget_left = self.max_steps - res.steps
                if outcome.success:
                    res.success = True
                    res.terminate_reason = "done"
                    self._event("done", success=True, reason="done", detail=outcome.detail)
                    break
                self._event("failure", reason=outcome.reason, detail=outcome.detail,
                            steps=res.steps, replan=replan + 1, max_replans=self.max_replans)
                if self.verbose:
                    self.log(f"  [{name}] 미션 실패: {outcome.reason} (steps={res.steps}) "
                             f"→ 수리 {replan+1}/{self.max_replans}")

            if replan >= self.max_replans or budget_left <= 0:
                res.terminate_reason = (last_rollout.reason if last_rollout
                                        else (outcome.reason if outcome else "no_valid_code"))
                self._event("done", success=False, reason=res.terminate_reason)
                break

            replan += 1
            res.replans = replan
            repair_prompt = house_repair_user(outcome.reason, outcome.detail, outcome.state, prev_code)
            self._event("request", phase="repair", attempt=replan,
                        reason=outcome.reason, prompt=repair_prompt)
            text, m = self.llm.complete(HOUSE_SYSTEM, repair_prompt,
                                        temperature=self.temperature, max_tokens=self.max_tokens)
            self._tally_llm(res, m)
            if not m.ok:
                res.terminate_reason = "llm_error"
                self._event("done", success=False, reason="llm_error", error=m.error)
                break
            new_code = extract_code(text)
            self._event("code", attempt=replan, code=new_code, metrics=self._m(m))
            try:
                plan_fn, prev_code = build_plan(new_code)
                self._event("build", ok=True, attempt=replan)
            except Exception as e:
                plan_fn = None
                outcome = _Outcome("no_valid_code",
                                   f"The rewritten code failed to load: {type(e).__name__}: {e}",
                                   outcome.state)
                prev_code = new_code
                self._event("build", ok=False, attempt=replan, error=f"{type(e).__name__}: {e}")
                if self.verbose:
                    self.log(f"  [{name}] 수리 코드 빌드 실패 → 재요청")

        res.cycle_hops = None
        res.cycle_hz = 0.0
        return res

    def _tally_llm(self, res: EpisodeResult, m):
        res.llm_calls += 1
        res.llm_total_s += m.total_s
        res.llm_total_tokens += m.completion_tokens
        if res.llm_calls == 1:
            res.ttft_s = m.ttft_s
        # 토큰 가중 평균 TPS: tps>0 인 호출의 토큰만 분모로(실패/0tps 호출이 분모를 부풀리지 않게)
        if m.tps > 0:
            num = getattr(res, "_tps_num", 0.0) + m.tps * m.completion_tokens
            den = getattr(res, "_tps_den", 0) + m.completion_tokens
            res._tps_num, res._tps_den = num, den
            res.tps = num / max(1, den)
