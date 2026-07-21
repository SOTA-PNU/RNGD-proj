#!/usr/bin/env python3
"""object_search_node — TurtleBot3(waffle)을 LLM 이 짠 plan(state) 컨트롤러로 폐루프 구동해
'집 안을 돌며 특정 물건이 있는지 확인하고 현관으로 복귀'하는 미션을 수행하는 rclpy 노드.

폐루프(robot-sim 의 house_search 하니스를 ROS2/Gazebo turtlebot3_house 로 옮긴 것):
  1) /odom /scan /camera/image_raw /camera/camera_info (+GT 면 /objects_ground_truth) 구독.
  2) 제어주기(~10 Hz)마다 토픽들로 state dict 조립 — 헤드리스와 같은 키:
     pos/heading/lidar/lidar_angles/scan/objective/waypoints/home/memory/v_max/w_max/dt...
  3) LLM 에게 plan(state) 코드 요청 → 샌드박스 build → 매 주기 호출.
     plan 이 {'v','w'} 면 cmd_vel 로 발행(경로 추종). {'done':True,'present':bool} 이면 미션 종료
     선언으로 보고, '현관 복귀 + 판정 정확(/objects_ground_truth 대조)'을 검사.
  4) 실패(충돌/정체/예외/타임아웃/missed_object/false_report/not_home/no_report/코드없음)면
     수리 프롬프트로 고친 코드를 받아 계속(replan 상한).

전제: 전역 경로(waypoints) 는 파라미터로 줍니다(실제 시스템에선 Nav2 등 전역 플래너가 제공).
      detector:=ground_truth 면 /objects_ground_truth(std_msgs/String JSON)로 물건 pose 를 받습니다.

토픽(이 저장소 waffle 브리지 실측, llm_nav_node 와 동일):
  sub  /scan(LaserScan) /camera/image_raw(Image) /camera/camera_info(CameraInfo) /odom(Odometry)
  sub  /objects_ground_truth(std_msgs/String, JSON)  ← detector=ground_truth 일 때만
  pub  /cmd_vel  ← 이 ros_gz 브리지는 geometry_msgs/TwistStamped (파라미터 cmd_vel_stamped).
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

from turtlebot3_llm_nav import perception, object_perception, house_prompts
from turtlebot3_llm_nav.executor import (
    build_plan, extract_code, normalize_action, call_with_timeout, PlanTimeout,
)
from turtlebot3_llm_nav.llm_client import make_client


def _yaw_from_quat(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


# turtlebot3_house 기본 경로(네이티브 좌표, home=(-2.0,-0.5)). 실제 월드/방배치에 맞춰 launch 의
# waypoints 파라미터로 바꿔 쓰세요(또는 Nav2 전역 플래너 경로로 대체). 방들을 한 바퀴 돌고 복귀.
_DEFAULT_WAYPOINTS = [
    [-0.5, 0.0], [2.0, 0.5], [4.0, -1.5], [1.0, -3.5], [-2.0, -3.5],
    [-5.0, -1.0], [-5.5, 2.5], [-2.0, 3.5], [1.0, 3.0], [-2.0, -0.5],
]
_DEFAULT_HOME = [-2.0, -0.5]


class ObjectSearchNode(Node):

    def __init__(self):
        super().__init__("object_search_node")

        # ── 파라미터 ──────────────────────────────────────────────
        self.declare_parameter("llm_port", 8002)                 # coder7 (chat CATALOG)
        self.declare_parameter("llm_mock", "")                   # "good"/"buggy" 면 서버 없이
        self.declare_parameter("objective", '{"label": "cup", "color": "red"}')
        self.declare_parameter("detector", "ground_truth")       # ground_truth | yolo
        self.declare_parameter("waypoints", json.dumps(_DEFAULT_WAYPOINTS))
        self.declare_parameter("home", json.dumps(_DEFAULT_HOME))
        self.declare_parameter("max_replans", 5)
        self.declare_parameter("goal_tol", 0.4)                  # 현관 복귀 판정 거리[m]
        self.declare_parameter("v_max", 0.22)                   # waffle 최대 선속도
        self.declare_parameter("w_max", 1.8)
        self.declare_parameter("control_hz", 10.0)
        self.declare_parameter("collision_dist", 0.22)          # 최소 scan 이 이 아래면 충돌
        self.declare_parameter("stuck_ticks", 200)              # 진전없음 N틱이면 stuck
        self.declare_parameter("stuck_eps", 0.05)
        self.declare_parameter("lidar_beams", 24)
        self.declare_parameter("plan_timeout", 0.08)
        self.declare_parameter("cmd_vel_stamped", True)
        self.declare_parameter("max_steps", 8000)
        self.declare_parameter("cam_range", 4.0)                 # 물건 인지 최대 거리[m]

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
        self.stuck_ticks = int(gp("stuck_ticks").value)
        self.stuck_eps = float(gp("stuck_eps").value)
        self.lidar_beams = int(gp("lidar_beams").value)
        self.plan_timeout = float(gp("plan_timeout").value)
        self.stamped = bool(gp("cmd_vel_stamped").value)
        self.max_steps = int(gp("max_steps").value)
        self.cam_range = float(gp("cam_range").value)
        self.objective = self._parse_json(gp("objective").value, {}, "objective")
        self.waypoints = self._parse_json(gp("waypoints").value, _DEFAULT_WAYPOINTS, "waypoints")
        self.home = self._parse_json(gp("home").value, _DEFAULT_HOME, "home")

        # ── 인지/LLM/샌드박스 ─────────────────────────────────────
        self.detector = object_perception.make_object_detector(
            self.detector_kind, max_range=self.cam_range)
        self.camera = perception.CameraModel()
        self.client = make_client(mock=self.mock, port=self.port, model_label="object_search")
        self._gt_objects: list = []          # /objects_ground_truth 전체(정답 판정용)
        self._plan_fn = None
        self._plan_code = ""
        self._memory: dict = {}
        self._history: list = []
        self._replans = 0
        self._building = False
        self._done = False

        # ── 최신 센서 캐시 ────────────────────────────────────────
        self._lock = threading.Lock()
        self._scan = None
        self._pose = perception.RobotPose()
        self._have_odom = False
        self._image = None

        # 진전/정체/탐색 추적
        self._tick = 0
        self._anchor = (0.0, 0.0)
        self._stuck_count = 0
        self._visited = set()        # 실제 방문한 경로 웨이포인트 index(독립 계산)
        self._target_seen = False    # 진짜 목표를 카메라로 실제 본 적 있는가

        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(LaserScan, "/scan", self._on_scan, sensor_qos)
        self.create_subscription(Image, "/camera/image_raw", self._on_image, sensor_qos)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_caminfo, sensor_qos)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        if self.detector_kind in ("ground_truth", "gt", "oracle"):
            self.create_subscription(String, "/objects_ground_truth", self._on_objects, 10)

        cmd_type = TwistStamped if self.stamped else Twist
        self.cmd_pub = self.create_publisher(cmd_type, "/cmd_vel", 10)

        self.get_logger().info(
            f"object_search_node 시작 | client={self.client.name} port={self.port} "
            f"detector={self.detector_kind} objective={self.objective} "
            f"waypoints={len(self.waypoints)} home={self.home} "
            f"cmd_vel={'TwistStamped' if self.stamped else 'Twist'}")

        self.timer = self.create_timer(self.dt, self._control_tick)

    def _parse_json(self, raw, default, what):
        try:
            v = json.loads(str(raw))
            return v
        except Exception:
            self.get_logger().warn(f"{what} 파라미터 파싱 실패 → 기본값 사용")
            return default

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

    def _on_objects(self, msg: String):
        if isinstance(self.detector, object_perception.ObjectGroundTruthDetector):
            self.detector.set_objects(msg.data)
        # 정답 판정용 전체 목록도 보관
        try:
            data = json.loads(msg.data)
            self._gt_objects = data.get("objects", data) if isinstance(data, (dict, list)) else []
            if isinstance(self._gt_objects, dict):
                self._gt_objects = self._gt_objects.get("objects", [])
        except Exception:
            pass

    # ── state 조립 ────────────────────────────────────────────────
    def _downsample_lidar(self, ranges, angle_min, angle_inc, range_max):
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
            a = math.atan2(math.sin(a), math.cos(a))
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
        try:
            seen = self.detector.detect(image, self.camera, pose)
        except NotImplementedError:
            self.get_logger().error("실검출기(yolo)는 STUB 입니다. detector:=ground_truth 로 실행하세요.")
            seen = []
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"detect 예외(무시): {e}")
            seen = []
        state = {
            "pos": (pose.x, pose.y),
            "heading": pose.yaw,
            "lidar": lidar,
            "lidar_angles": angles,
            "max_range": rng_max,
            "v_max": self.v_max,
            "w_max": self.w_max,
            "dt": self.dt,
            "robot_radius": 0.18,
            "goal_tol": self.goal_tol,
            "bounds": (100.0, 100.0),
            "memory": self._memory,
            "mission": "Search the house for the target object, then return home.",
            "objective": dict(self.objective),
            "scan": seen,
            "waypoints": [list(p) for p in self.waypoints],
            "home": list(self.home),
        }
        return state, seen, lidar, angles

    def _match_obj(self, feat: dict) -> bool:
        """검출 특징이 objective 의 모든 항목과 일치하면 True(한 특징만 같은 decoy 는 False)."""
        return bool(self.objective) and all(str(feat.get(k)) == str(v)
                                            for k, v in self.objective.items())

    # ── 정답(물건 존재) 판정: /objects_ground_truth 의 전체 목록과 objective 대조 ──
    def _objective_present(self) -> bool:
        for o in self._gt_objects:
            f = o.get("features", {}) if isinstance(o, dict) else {}
            if self.objective and all(str(f.get(k)) == str(v) for k, v in self.objective.items()):
                return True
        return False

    def _min_front(self, lidar, angles):
        fr = [r for r, a in zip(lidar, angles) if abs(a) < 0.5]
        return min(fr) if fr else float("inf")

    # ── 제어 틱 ──────────────────────────────────────────────────
    def _control_tick(self):
        if self._done:
            return
        if not self._have_odom or self._scan is None:
            return
        if self._building:
            self._publish(0.0, 0.0)
            return
        self._tick += 1
        state, seen, lidar, angles = self._build_state()
        # 탐색 증거 추적(컨트롤러 자기보고가 아니라 노드가 독립 계산)
        if any(self._match_obj(d.get("features", {})) for d in seen):
            self._target_seen = True
        px0, py0 = state["pos"]
        for wi, wp in enumerate(self.waypoints):
            if wi not in self._visited and math.hypot(px0 - wp[0], py0 - wp[1]) <= 0.6:
                self._visited.add(wi)
        if self._tick > self.max_steps:
            return self._on_failure("no_report", state, seen, lidar, angles,
                                    detail="max_steps reached without declaring done")

        if self._plan_fn is None:
            self._request_plan(initial=True)
            self._publish(0.0, 0.0)
            return

        # plan() 실행
        try:
            action = call_with_timeout(self._plan_fn, state, self.plan_timeout)
        except PlanTimeout:
            return self._on_failure("timeout", state, seen, lidar, angles)
        except Exception as e:  # noqa: BLE001
            return self._on_failure("exception", state, seen, lidar, angles, detail=str(e))

        # 미션 종료 선언?
        if isinstance(action, dict) and action.get("done"):
            self._publish(0.0, 0.0)
            pres = action.get("present", None)
            claim = bool(action.get("found", False)) if pres is None else bool(pres)
            px, py = state["pos"]
            at_home = math.hypot(px - self.home[0], py - self.home[1]) <= self.goal_tol
            truth = self._objective_present()
            if not at_home:
                return self._on_failure("not_home", state, seen, lidar, angles)
            if claim != truth:
                reason = "missed_object" if (truth and not claim) else "false_report"
                return self._on_failure(reason, state, seen, lidar, angles)
            # 추측 통과 방지: present 주장→목표를 실제로 봤어야, absent 주장→경로 60%+ 실제 방문
            n_wp = max(1, len(self.waypoints))
            searched = self._target_seen if claim else (len(self._visited) / n_wp >= 0.6)
            if not searched:
                return self._on_failure("searched_too_little", state, seen, lidar, angles,
                                        detail=f"visited {len(self._visited)}/{n_wp} waypoints")
            return self._finish(True, f"reported {'present' if claim else 'absent'} (correct) & home")

        # 이동
        try:
            v, w = normalize_action(action, self.v_max, self.w_max, state["heading"], self.dt)
        except Exception as e:  # noqa: BLE001
            return self._on_failure("exception", state, seen, lidar, angles, detail=str(e))

        # 발행 전 안전 게이트: 충돌 임박 시 강제 정지
        if self._min_front(lidar, angles) < self.collision_dist:
            self._publish(0.0, 0.0)
            return self._on_failure("collision", state, seen, lidar, angles)
        self._publish(v, w)

        # 정체 감지
        px, py = state["pos"]
        if math.hypot(px - self._anchor[0], py - self._anchor[1]) > 1.0:
            self._anchor = (px, py)
            self._stuck_count = 0
        else:
            self._stuck_count += 1
        if self._stuck_count >= self.stuck_ticks:
            return self._on_failure("stuck", state, seen, lidar, angles)

    # ── 실패 처리(수리 요청) ─────────────────────────────────────
    def _on_failure(self, reason, state, seen, lidar, angles, detail=""):
        self._publish(0.0, 0.0)
        self.get_logger().warn(f"실패 감지: {reason} (replan {self._replans}/{self.max_replans}) {detail}")
        if self._replans >= self.max_replans:
            return self._finish(False, f"max_replans:{reason}")
        self._stuck_count = 0
        telemetry = {
            "min_front_lidar": round(self._min_front(lidar or [], angles or []), 3),
            "seen_object_features": [d.get("features", {}) for d in (seen or [])][:5],
            "found_in_memory": bool(self._memory.get("found")),
            "phase": self._memory.get("phase"),
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
                    user = house_prompts.initial_user(self.objective, len(self.waypoints))
                    self._history = []
                else:
                    self._replans += 1
                    user = house_prompts.repair_user(reason, self.objective, telemetry, detail)
                t0 = time.time()
                text, metrics = self.client.complete(
                    house_prompts.HOUSE_SYSTEM, user, history=self._history,
                    temperature=0.2, max_tokens=1024)
                if not metrics.ok:
                    self.get_logger().error(f"LLM 호출 실패: {metrics.error}")
                    return
                code = extract_code(text)
                try:
                    fn, code = build_plan(code, timeout=1.0)
                except Exception as e:  # noqa: BLE001
                    self.get_logger().warn(f"plan 빌드 실패: {e}")
                    if self._replans < self.max_replans:
                        self._replans += 1
                        u2 = house_prompts.repair_user("no_valid_code", self.objective, telemetry, str(e))
                        text2, _ = self.client.complete(
                            house_prompts.HOUSE_SYSTEM, u2,
                            history=self._history + [{"role": "user", "content": user},
                                                     {"role": "assistant", "content": text}],
                            temperature=0.2, max_tokens=1024)
                        fn, code = build_plan(extract_code(text2), timeout=1.0)
                        text = text2
                    else:
                        return
                self._plan_fn = fn
                self._plan_code = code
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
            self.get_logger().info(f"✅ 미션 성공: 물건 확인 후 복귀 ({why}) — 정지")
        else:
            self.get_logger().warn(f"❌ 종료(실패): {why} — 정지")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectSearchNode()
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
