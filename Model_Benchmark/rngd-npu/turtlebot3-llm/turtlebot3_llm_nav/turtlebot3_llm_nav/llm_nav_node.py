#!/usr/bin/env python3
"""llm_nav_node — TurtleBot3(waffle)을 LLM 이 짠 plan(state) 컨트롤러로 폐루프 구동해
카메라로 본 '특정 사람'에게 다가가게 하는 rclpy 노드.

폐루프(robot-sim 하니스를 ROS2 로 옮긴 것):
  1) /odom /scan /camera/image_raw /camera/camera_info (+GT 면 /people_ground_truth) 구독.
  2) 제어주기(~10 Hz)마다 토픽들로 state dict 조립
     (pos/heading/lidar/lidar_angles/camera/target/memory/v_max/w_max/dt).
  3) LLM 에게 plan(state) 코드 요청 → 샌드박스 build → 매 주기 호출 → (v,w) clamp → cmd_vel 발행.
  4) 실패 감지하면(충돌/길잃음/엉뚱한사람/정체/예외/타임아웃/코드없음) 수리 프롬프트로 고친
     코드를 받아 계속(replan 상한). 목표 사람에 충분히 가까워지면 성공.

토픽(이 저장소 waffle 브리지 실측):
  sub  /scan(LaserScan) /camera/image_raw(Image) /camera/camera_info(CameraInfo) /odom(Odometry)
  sub  /people_ground_truth(std_msgs/String, JSON)  ← detector=ground_truth 일 때만
  pub  /cmd_vel  ← 이 ros_gz 브리지는 geometry_msgs/TwistStamped (waffle_bridge.yaml 확인).
       파라미터 cmd_vel_stamped 로 평범한 Twist 도 선택 가능.
"""
from __future__ import annotations

import json
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan, Image, CameraInfo
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import Twist, TwistStamped

from turtlebot3_llm_nav import perception, prompts
from turtlebot3_llm_nav.executor import (
    build_plan, extract_code, normalize_action, call_with_timeout, PlanTimeout,
)
from turtlebot3_llm_nav.llm_client import make_client


def _yaw_from_quat(q) -> float:
    """쿼터니언(z,w 위주)에서 yaw(rad)."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class LlmNavNode(Node):

    def __init__(self):
        super().__init__("llm_nav_node")

        # ── 파라미터 ──────────────────────────────────────────────
        self.declare_parameter("llm_port", 8002)                 # coder7 (chat CATALOG)
        self.declare_parameter("llm_mock", "")                   # "good"/"buggy" 면 서버 없이
        self.declare_parameter("target", '{"shirt": "red"}')     # 찾을 사람 특징(JSON)
        self.declare_parameter("detector", "ground_truth")       # ground_truth | yolo
        self.declare_parameter("max_replans", 5)
        self.declare_parameter("goal_tol", 0.6)                  # 이 거리[m] 안이면 도착
        self.declare_parameter("v_max", 0.22)                   # waffle 최대 선속도
        self.declare_parameter("w_max", 1.8)                    # 최대 각속도
        self.declare_parameter("control_hz", 10.0)
        self.declare_parameter("collision_dist", 0.22)          # 최소 scan 이 이 아래면 충돌
        self.declare_parameter("lost_ticks", 40)                # 목표 미검출 연속 N틱이면 lost
        self.declare_parameter("stuck_ticks", 80)               # 진전없음 N틱이면 stuck
        self.declare_parameter("stuck_eps", 0.05)               # 이동 임계[m]
        self.declare_parameter("lidar_beams", 24)               # state 에 넣을 다운샘플 광선 수
        self.declare_parameter("plan_timeout", 0.08)            # plan() 1회 시간제한[s]
        self.declare_parameter("cmd_vel_stamped", True)         # TwistStamped(이 브리지 기본) vs Twist
        self.declare_parameter("max_steps", 4000)               # 전체 안전 상한(틱)

        gp = self.get_parameter
        self.port = int(gp("llm_port").value)
        self.mock = str(gp("llm_mock").value) or None
        self.detector_kind = str(gp("detector").value)
        self.max_replans = int(gp("max_replans").value)
        self.goal_tol = float(gp("goal_tol").value)
        self.v_max = float(gp("v_max").value)
        self.w_max = float(gp("w_max").value)
        self.hz = float(gp("control_hz").value)
        self.dt = 1.0 / max(1.0, self.hz)
        self.collision_dist = float(gp("collision_dist").value)
        self.lost_ticks = int(gp("lost_ticks").value)
        self.stuck_ticks = int(gp("stuck_ticks").value)
        self.stuck_eps = float(gp("stuck_eps").value)
        self.lidar_beams = int(gp("lidar_beams").value)
        self.plan_timeout = float(gp("plan_timeout").value)
        self.stamped = bool(gp("cmd_vel_stamped").value)
        self.max_steps = int(gp("max_steps").value)
        try:
            self.target = json.loads(str(gp("target").value))
            if not isinstance(self.target, dict):
                raise ValueError
        except Exception:
            self.get_logger().warn("target 파라미터 파싱 실패 → {} 사용")
            self.target = {}

        # ── 인지/LLM/샌드박스 ─────────────────────────────────────
        self.detector = perception.make_detector(self.detector_kind)
        self.camera = perception.CameraModel()
        self.client = make_client(mock=self.mock, port=self.port, model_label="llm_nav")
        self._plan_fn = None                 # 현재 컨트롤러
        self._plan_code = ""
        self._memory: dict = {}              # plan 간 지속 메모리
        self._history: list = []             # LLM 대화 이력(수리 누적)
        self._replans = 0
        self._building = False               # LLM 호출 중(블로킹 방지)
        self._done = False

        # ── 최신 센서 캐시 ────────────────────────────────────────
        self._lock = threading.Lock()
        self._scan = None                    # (ranges, angle_min, angle_inc, range_max)
        self._pose = perception.RobotPose()
        self._have_odom = False
        self._image = None
        self._cam_seen = 0

        # 진전/길잃음 추적
        self._tick = 0
        self._last_progress_xy = (0.0, 0.0)
        self._best_dist = float("inf")
        self._stuck_count = 0
        self._lost_count = 0

        # ── QoS: 센서는 best-effort 가 흔함 ──────────────────────
        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(LaserScan, "/scan", self._on_scan, sensor_qos)
        self.create_subscription(Image, "/camera/image_raw", self._on_image, sensor_qos)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_caminfo, sensor_qos)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        if self.detector_kind in ("ground_truth", "gt", "oracle"):
            self.create_subscription(String, "/people_ground_truth", self._on_people, 10)

        cmd_type = TwistStamped if self.stamped else Twist
        self.cmd_pub = self.create_publisher(cmd_type, "/cmd_vel", 10)

        self.get_logger().info(
            f"llm_nav_node 시작 | client={self.client.name} port={self.port} "
            f"detector={self.detector_kind} target={self.target} "
            f"cmd_vel={'TwistStamped' if self.stamped else 'Twist'}")

        # 제어 타이머
        self.timer = self.create_timer(self.dt, self._control_tick)

    # ── 콜백 ──────────────────────────────────────────────────────
    def _on_scan(self, msg: LaserScan):
        with self._lock:
            self._scan = (list(msg.ranges), msg.angle_min, msg.angle_increment,
                          msg.range_max if msg.range_max > 0 else 3.5)

    def _on_image(self, msg: Image):
        with self._lock:
            self._image = msg

    def _on_caminfo(self, msg: CameraInfo):
        fx = msg.k[0] if getattr(msg, "k", None) and len(msg.k) > 0 else None
        self.camera.update_from_info(msg.width, msg.height, fx)

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose
        with self._lock:
            self._pose = perception.RobotPose(
                x=p.position.x, y=p.position.y, yaw=_yaw_from_quat(p.orientation))
            self._have_odom = True

    def _on_people(self, msg: String):
        # GT detector 에만 의미. 실검출기면 무시됨.
        if isinstance(self.detector, perception.GroundTruthDetector):
            self.detector.set_people(msg.data)

    # ── state 조립 ────────────────────────────────────────────────
    def _downsample_lidar(self, ranges, angle_min, angle_inc, range_max):
        """360 빔을 lidar_beams 개로 줄이고, inf/nan 은 range_max 로. 각도(rad, 0=front,+=left).
        scan 의 0~2π 를 -π~π 로 wrap 해서 로봇프레임으로 맞춥니다."""
        n = len(ranges)
        if n == 0:
            return [], []
        step = max(1, n // max(1, self.lidar_beams))
        out_r, out_a = [], []
        for i in range(0, n, step):
            r = ranges[i]
            if r is None or not math.isfinite(r) or r <= 0.0:
                r = range_max
            a = angle_min + i * angle_inc
            a = math.atan2(math.sin(a), math.cos(a))   # wrap to [-π,π]
            out_r.append(float(r))
            out_a.append(float(a))
        return out_r, out_a

    def _build_state(self):
        with self._lock:
            scan = self._scan
            pose = perception.RobotPose(self._pose.x, self._pose.y, self._pose.yaw)
            image = self._image
        lidar, angles = ([], [])
        rng_max = 3.5
        if scan is not None:
            ranges, amin, ainc, rmax = scan
            rng_max = rmax
            lidar, angles = self._downsample_lidar(ranges, amin, ainc, rmax)
        # 사람 검출
        try:
            cam = self.detector.detect(image, self.camera, pose)
        except NotImplementedError:
            self.get_logger().error("실검출기(yolo)는 STUB 입니다. detector:=ground_truth 로 실행하세요.")
            cam = []
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"detect 예외(무시): {e}")
            cam = []
        state = {
            "pos": (pose.x, pose.y),
            "heading": pose.yaw,
            "lidar": lidar,
            "lidar_angles": angles,
            "max_range": rng_max,
            "v_max": self.v_max,
            "w_max": self.w_max,
            "dt": self.dt,
            "camera": cam,
            "target": dict(self.target),
            "memory": self._memory,
        }
        return state, cam, lidar

    # ── 매칭/실패 판정 헬퍼 ──────────────────────────────────────
    def _match(self, det) -> bool:
        f = det.get("features", {})
        return all(str(f.get(k)) == str(v) for k, v in self.target.items())

    def _matched(self, cam):
        ms = [d for d in cam if self._match(d)]
        return min(ms, key=lambda d: d["distance"]) if ms else None

    # ── 제어 틱 ──────────────────────────────────────────────────
    def _control_tick(self):
        if self._done:
            return
        if not self._have_odom or self._scan is None:
            return  # 센서 준비 전엔 가만히
        if self._building:
            self._publish(0.0, 0.0)   # LLM 생성 중엔 정지(블로킹 방지)
            return
        self._tick += 1
        if self._tick > self.max_steps:
            self.get_logger().warn("max_steps 도달 — 종료")
            return self._finish(False, "max_steps")

        state, cam, lidar = self._build_state()

        # 컨트롤러가 없으면(최초) 생성
        if self._plan_fn is None:
            self._request_plan(initial=True)
            self._publish(0.0, 0.0)
            return

        # plan() 실행(시간제한)
        try:
            action = call_with_timeout(self._plan_fn, state, self.plan_timeout)
            v, w = normalize_action(action, self.v_max, self.w_max, state["heading"], self.dt)
        except PlanTimeout:
            return self._on_failure("timeout", state, cam, lidar)
        except Exception as e:  # noqa: BLE001
            return self._on_failure("exception", state, cam, lidar, detail=str(e))

        # 발행 전 안전 게이트: 충돌 임박 시 강제 정지(plan 무시 못 하게)
        min_front = self._min_front(lidar, state["lidar_angles"])
        if min_front < self.collision_dist:
            self._publish(0.0, 0.0)
            return self._on_failure("collision", state, cam, lidar)
        self._publish(v, w)

        # ── 성공/실패 모니터 ──────────────────────────────────────
        matched = self._matched(cam)
        # 1) 목표 도착?
        if matched is not None and matched["distance"] <= self.goal_tol:
            return self._finish(True, "reached")
        # 2) 엉뚱한 사람에 도착(목표 아닌 사람에 매우 근접 + 목표 안 보임)
        wrong = [d for d in cam if not self._match(d) and d["distance"] <= self.goal_tol]
        if wrong and matched is None:
            return self._on_failure("wrong_person", state, cam, lidar)
        # 3) 길잃음: 목표가 연속 N틱 안 보임
        if matched is None:
            self._lost_count += 1
            if self._lost_count >= self.lost_ticks:
                return self._on_failure("lost_target", state, cam, lidar)
        else:
            self._lost_count = 0
            # 진전 추적은 목표가 보일 때 거리로
            if matched["distance"] < self._best_dist - self.stuck_eps:
                self._best_dist = matched["distance"]
                self._stuck_count = 0
            else:
                self._stuck_count += 1
        # 4) 정체: 위치도 안 변하고 목표 거리도 안 줄면
        px, py = state["pos"]
        moved = math.hypot(px - self._last_progress_xy[0], py - self._last_progress_xy[1])
        if moved > self.stuck_eps:
            self._last_progress_xy = (px, py)
            self._stuck_count = max(0, self._stuck_count - 1)
        if self._stuck_count >= self.stuck_ticks:
            return self._on_failure("stuck", state, cam, lidar)

    def _min_front(self, lidar, angles):
        fr = [r for r, a in zip(lidar, angles) if abs(a) < 0.5]
        return min(fr) if fr else float("inf")

    # ── 실패 처리(수리 요청) ─────────────────────────────────────
    def _on_failure(self, reason, state, cam, lidar, detail=""):
        self._publish(0.0, 0.0)   # 안전: 수리 동안 정지
        self.get_logger().warn(f"실패 감지: {reason} (replan {self._replans}/{self.max_replans}) {detail}")
        if self._replans >= self.max_replans:
            return self._finish(False, f"max_replans:{reason}")
        # 카운터 리셋(같은 실패 즉시 재트리거 방지)
        self._lost_count = 0
        self._stuck_count = 0
        self._best_dist = float("inf")
        telemetry = {
            "min_front_lidar": round(self._min_front(lidar, state["lidar_angles"]), 3),
            "seen_people_features": [d.get("features", {}) for d in cam][:5],
            "matched_visible": self._matched(cam) is not None,
        }
        self._request_plan(initial=False, reason=reason, telemetry=telemetry, detail=detail)

    # ── LLM 호출(별도 스레드, 타이머 비블로킹) ──────────────────
    def _request_plan(self, initial: bool, reason: str = "", telemetry=None, detail=""):
        if self._building:
            return
        self._building = True

        def _work():
            try:
                if initial:
                    user = prompts.initial_user(self.target)
                    self._history = []
                else:
                    self._replans += 1
                    user = prompts.repair_user(reason, self.target, telemetry, detail)
                t0 = time.time()
                text, metrics = self.client.complete(
                    prompts.SYSTEM, user, history=self._history,
                    temperature=0.2, max_tokens=1024)
                if not metrics.ok:
                    self.get_logger().error(f"LLM 호출 실패: {metrics.error}")
                    return
                code = extract_code(text)
                try:
                    fn, code = build_plan(code, timeout=1.0)
                except Exception as e:  # noqa: BLE001 — 코드 불량 → 한 번 더 수리
                    self.get_logger().warn(f"plan 빌드 실패: {e}")
                    if self._replans < self.max_replans:
                        self._replans += 1
                        u2 = prompts.repair_user("no_valid_code", self.target, telemetry, str(e))
                        text2, _ = self.client.complete(
                            prompts.SYSTEM, u2,
                            history=self._history + [{"role": "user", "content": user},
                                                     {"role": "assistant", "content": text}],
                            temperature=0.2, max_tokens=1024)
                        fn, code = build_plan(extract_code(text2), timeout=1.0)
                        text = text2
                    else:
                        return
                self._plan_fn = fn
                self._plan_code = code
                # 대화 이력 누적(다음 수리에 맥락 제공) — 너무 길어지지 않게 최근 위주
                self._history = (self._history + [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": "```python\n" + code + "\n```"},
                ])[-6:]
                self.get_logger().info(
                    f"새 plan() 적용 ({'init' if initial else reason}) "
                    f"| {time.time()-t0:.1f}s TTFT={metrics.ttft_s:.2f}s "
                    f"TPS={metrics.tps:.1f} | {len(code)} chars")
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"plan 요청 스레드 예외: {e}")
            finally:
                self._building = False

        threading.Thread(target=_work, daemon=True).start()

    # ── 발행/종료 ────────────────────────────────────────────────
    def _publish(self, v: float, w: float):
        if self.stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_footprint"
            msg.twist.linear.x = float(v)
            msg.twist.angular.z = float(w)
        else:
            msg = Twist()
            msg.linear.x = float(v)
            msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def _finish(self, success: bool, why: str):
        self._publish(0.0, 0.0)
        self._done = True
        if success:
            self.get_logger().info(f"✅ 성공: 목표 사람 도달 ({why}) — 정지")
        else:
            self.get_logger().warn(f"❌ 종료(실패): {why} — 정지")


def main(args=None):
    rclpy.init(args=args)
    node = LlmNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._publish(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
