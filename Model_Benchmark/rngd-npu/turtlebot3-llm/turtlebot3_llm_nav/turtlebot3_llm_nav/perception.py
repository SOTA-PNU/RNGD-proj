"""PersonDetector — 카메라 Image(+보조정보)를 '사람 검출 리스트'로 바꾸는 추상화.

검출 하나의 형식(노드가 plan(state)['camera'] 로 넘기는 단위):
    {
      'bearing':  float,   # 로봇 정면 기준 좌우 각도(rad). +면 왼쪽, -면 오른쪽
      'distance': float,   # 추정 거리(m)
      'features': dict,    # 외형/속성 (예: {'shirt': 'red', 'height': 1.7})
      'conf':     float,   # 검출 신뢰도 0~1
    }

백엔드 2개를 같은 인터페이스(detect 메서드) 뒤에 둡니다.
  (a) GroundTruthDetector  — 실제 검출기 없이 Gazebo 에서 파이프라인을 돌리기 위한 것.
        시뮬레이터가 발행하는 '사람 ground-truth pose'(std_msgs/String JSON 토픽
        /people_ground_truth)를 받아 로봇 좌표계로 변환→카메라 FOV 안에 드는 사람만
        검출로 만듭니다. **이것은 '진짜 인식'이 아니라 시뮬레이션용 오라클입니다.**
  (b) YoloDetector          — 실제 검출기(YOLO/얼굴인식) 자리. 인터페이스만 채운 STUB 이며
        구현은 TODO 입니다. 카메라 Image 픽셀 bbox → bearing/distance 투영 헬퍼는 공유.

bearing/distance 투영(공유 헬퍼 pixel_to_bearing, estimate_distance)은
카메라 HFOV(waffle RealSense R200 = 1.02974 rad)와 이미지 폭으로 계산합니다.
세로 bbox 높이를 사람 키 가정(1.7m)에 맞춰 핀홀로 거리 추정.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# waffle 카메라(Intel RealSense R200) 기본값 — model.sdf 에서 검증.
DEFAULT_HFOV = 1.02974        # horizontal_fov [rad]
DEFAULT_IMG_W = 1920          # image width  [px]
DEFAULT_IMG_H = 1080          # image height [px]
ASSUMED_PERSON_HEIGHT = 1.7   # 거리추정용 사람 키 가정 [m]


# ── 공유 투영 헬퍼 ────────────────────────────────────────────────
def pixel_to_bearing(cx: float, img_w: int = DEFAULT_IMG_W, hfov: float = DEFAULT_HFOV) -> float:
    """이미지 가로 픽셀 x(중심 cx)를 로봇/카메라 정면기준 각도(rad)로.
    핀홀: 정규화 위치 u=(cx - W/2)/(W/2) ∈ [-1,1], bearing = atan(u * tan(hfov/2)).
    이미지 +x(오른쪽) → 로봇 -bearing(오른쪽). 부호 변환 포함."""
    half = img_w / 2.0
    u = (cx - half) / half                       # 오른쪽이 +
    bearing = math.atan(u * math.tan(hfov / 2.0))
    return -bearing                               # 오른쪽 픽셀 = 음의 bearing(로봇 우측)


def estimate_distance(bbox_h_px: float, img_h: int = DEFAULT_IMG_H,
                      hfov: float = DEFAULT_HFOV, img_w: int = DEFAULT_IMG_W,
                      person_h_m: float = ASSUMED_PERSON_HEIGHT) -> float:
    """사람 bbox 세로 픽셀높이로 거리(m)를 핀홀 추정.
    수직 FOV vfov = 2*atan(tan(hfov/2)*H/W). focal_y(px)=H/(2*tan(vfov/2)).
    distance = person_h_m * focal_y / bbox_h_px. (depth 센서 있으면 그걸로 대체 권장.)"""
    if bbox_h_px <= 1.0:
        return 99.0
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * (img_h / float(img_w)))
    focal_y = img_h / (2.0 * math.tan(vfov / 2.0))
    return max(0.1, person_h_m * focal_y / float(bbox_h_px))


# ── 공통 인터페이스 ───────────────────────────────────────────────
@dataclass
class CameraModel:
    """plan(state) 에 들어가는 카메라 내부정보(camera_info 로 갱신 가능)."""
    img_w: int = DEFAULT_IMG_W
    img_h: int = DEFAULT_IMG_H
    hfov: float = DEFAULT_HFOV

    def update_from_info(self, width: int, height: int, fx: Optional[float] = None):
        if width and height:
            self.img_w, self.img_h = int(width), int(height)
        if fx and fx > 0:                         # fx 가 있으면 그걸로 HFOV 재계산(더 정확)
            self.hfov = 2.0 * math.atan((self.img_w / 2.0) / fx)


class PersonDetector:
    """모든 백엔드가 따라야 하는 인터페이스.
    detect(image_msg, camera, robot_pose) -> List[detection dict]."""

    name = "base"

    def detect(self, image_msg, camera: CameraModel, robot_pose) -> List[dict]:
        raise NotImplementedError


# ── (a) Ground-truth 백엔드 (시뮬레이션용 오라클) ─────────────────
@dataclass
class RobotPose:
    """로봇 월드좌표 (x, y, yaw[rad])."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


class GroundTruthDetector(PersonDetector):
    """**실제 인식이 아님.** Gazebo 가 발행하는 사람 ground-truth pose 를 받아
    카메라 FOV 안의 사람만 '검출'로 투영합니다. 실검출기 없이 폐루프를 돌리기 위한 것.

    입력 토픽 /people_ground_truth (std_msgs/String) 의 JSON 형식(노드가 set_people 로 주입):
        [{"id":"p1","x":2.0,"y":1.0,"features":{"shirt":"red","height":1.7}}, ...]
    또는 {"people":[...]} 래핑도 허용.

    bearing/distance 는 픽셀이 아니라 실제 상대좌표로 정확히 계산하고(오라클이므로),
    FOV(±hfov/2) 밖이거나 max_range 초과면 보이지 않는 것으로 제외합니다.
    conf 는 정면에 가깝고 가까울수록 1 에 가깝게 둡니다(현실적 노이즈 흉내, 선택)."""

    name = "ground_truth"

    def __init__(self, max_range: float = 8.0, fov_margin: float = 0.0):
        self.max_range = float(max_range)
        self.fov_margin = float(fov_margin)       # FOV 가장자리 여유(rad), 0 이면 정확히 ±hfov/2
        self._people: List[dict] = []

    def set_people(self, raw) -> None:
        """노드의 /people_ground_truth 콜백에서 호출. raw 는 JSON 문자열 또는 파싱된 list/dict."""
        try:
            data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        except Exception:
            self._people = []
            return
        if isinstance(data, dict):
            data = data.get("people", [])
        self._people = list(data) if isinstance(data, list) else []

    def detect(self, image_msg, camera: CameraModel, robot_pose: RobotPose) -> List[dict]:
        # image_msg 는 쓰지 않습니다(오라클). 인터페이스 통일을 위해 받기만 함.
        out: List[dict] = []
        if robot_pose is None:
            return out
        half_fov = camera.hfov / 2.0 + self.fov_margin
        for p in self._people:
            try:
                px, py = float(p["x"]), float(p["y"])
            except (KeyError, TypeError, ValueError):
                continue
            dx, dy = px - robot_pose.x, py - robot_pose.y
            dist = math.hypot(dx, dy)
            if dist > self.max_range or dist < 1e-3:
                continue
            world_ang = math.atan2(dy, dx)
            bearing = math.atan2(math.sin(world_ang - robot_pose.yaw),
                                 math.cos(world_ang - robot_pose.yaw))
            if abs(bearing) > half_fov:           # 카메라 화각 밖 → 안 보임
                continue
            # 신뢰도: 정면에 가깝고 가까울수록 높게(0.5~1.0)
            conf = 0.5 + 0.5 * max(0.0, math.cos(bearing)) * max(0.0, 1.0 - dist / self.max_range)
            out.append({
                "bearing": bearing,
                "distance": dist,
                "features": dict(p.get("features", {})),
                "conf": round(conf, 3),
                "id": p.get("id"),
            })
        out.sort(key=lambda d: d["distance"])
        return out


# ── (b) 실제 검출기 백엔드 (STUB — 구현 TODO) ─────────────────────
class YoloDetector(PersonDetector):
    """실제 카메라 영상 기반 사람 검출기 자리(YOLO 사람검출 + 얼굴/옷색 특징추출).

    **STUB 입니다. _run_model 만 구현하면 됩니다.** 만족해야 할 계약:
      입력  : image_msg(sensor_msgs/Image), camera(CameraModel), robot_pose(RobotPose; 무시 가능)
      출력  : detect() 가 PersonDetector 표준 dict 리스트를 돌려준다.
              {'bearing','distance','features','conf'} (위 ground-truth 와 동일 스키마).

    구현 가이드:
      1) cv_bridge 로 image_msg → numpy BGR (또는 Image.data 직접 디코드).
      2) YOLO(person 클래스)로 bbox 들 검출 → 각 bbox 중심 cx, 높이 bbox_h.
      3) bearing = pixel_to_bearing(cx, camera.img_w, camera.hfov)
         distance = estimate_distance(bbox_h, camera.img_h, camera.hfov, camera.img_w)
         (depth 토픽이 있으면 distance 는 그걸로 대체 — 더 정확.)
      4) features: bbox crop 에서 셔츠색(HSV 평균)·얼굴인식 id·키 추정 등 추출.
      5) conf: 모델 score.
    의존성(런타임에만): ultralytics(YOLO) 또는 opencv-dnn, cv_bridge, numpy.
    """

    name = "yolo"

    def __init__(self, model_path: str = "", conf_thr: float = 0.4):
        self.model_path = model_path
        self.conf_thr = float(conf_thr)
        self._model = None
        self._bridge = None

    def _ensure(self):
        if self._model is None:
            # TODO: 실제 모델 로드. 예)
            #   from ultralytics import YOLO
            #   self._model = YOLO(self.model_path or "yolov8n.pt")
            # from cv_bridge import CvBridge
            # self._bridge = CvBridge()
            raise NotImplementedError(
                "YoloDetector 는 STUB 입니다. _ensure/_run_model 을 구현하세요 "
                "(ultralytics + cv_bridge). 시뮬레이션은 detector:=ground_truth 를 쓰세요.")

    def _run_model(self, bgr) -> List[Dict]:
        """numpy BGR → [{'cx','bbox_h','features','conf'}, ...] 반환하도록 구현.
        (bearing/distance 투영은 detect() 가 공유 헬퍼로 처리.)"""
        raise NotImplementedError("YOLO 추론을 여기 구현하세요.")

    def detect(self, image_msg, camera: CameraModel, robot_pose) -> List[dict]:
        self._ensure()                                       # STUB → NotImplementedError
        bgr = self._bridge.imgmsg_to_cv2(image_msg, "bgr8")  # type: ignore[union-attr]
        out: List[dict] = []
        for det in self._run_model(bgr):
            if det.get("conf", 0.0) < self.conf_thr:
                continue
            out.append({
                "bearing": pixel_to_bearing(det["cx"], camera.img_w, camera.hfov),
                "distance": estimate_distance(det["bbox_h"], camera.img_h,
                                              camera.hfov, camera.img_w),
                "features": dict(det.get("features", {})),
                "conf": float(det["conf"]),
            })
        return out


# ── 팩토리 ────────────────────────────────────────────────────────
def make_detector(kind: str = "ground_truth", **kw) -> PersonDetector:
    """파라미터 detector('ground_truth'|'yolo')로 백엔드 선택."""
    kind = (kind or "ground_truth").lower()
    if kind in ("ground_truth", "gt", "oracle"):
        return GroundTruthDetector(max_range=kw.get("max_range", 8.0))
    if kind in ("yolo", "real", "detector"):
        return YoloDetector(model_path=kw.get("model_path", ""),
                            conf_thr=kw.get("conf_thr", 0.4))
    raise ValueError(f"알 수 없는 detector 종류: {kind!r} (ground_truth|yolo)")
