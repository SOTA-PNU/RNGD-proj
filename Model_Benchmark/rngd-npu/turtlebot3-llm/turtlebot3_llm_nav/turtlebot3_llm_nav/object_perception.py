"""ObjectDetector — 카메라 Image(+보조정보)를 '물건 검출 리스트'로 바꾸는 추상화(집 미션용).

검출 하나의 형식(노드가 plan(state)['scan'] 으로 넘기는 단위):
    {
      'bearing':  float,   # 로봇 정면 기준 좌우 각도(rad). +면 왼쪽, -면 오른쪽
      'distance': float,   # 추정 거리(m)
      'features': dict,    # 물건 속성 (예: {'label': 'cup', 'color': 'red'})
      'conf':     float,   # 검출 신뢰도 0~1
    }

사람찾기(perception.py)와 같은 구조를, 사람 대신 '물건'에 적용한 것입니다. 투영 헬퍼
(pixel_to_bearing, estimate_distance)와 CameraModel 은 perception.py 를 그대로 재사용합니다.

  (a) ObjectGroundTruthDetector — 실제 검출기 없이 Gazebo 에서 폐루프를 돌리기 위한 오라클.
        토픽 /objects_ground_truth (std_msgs/String, JSON) 의 물건 pose 를 받아 카메라 FOV 안의
        것만 검출로 투영합니다. **'진짜 인식'이 아니라 시뮬레이션용 오라클입니다.**
  (b) YoloObjectDetector — 실제 영상기반 물체검출기 자리(STUB). cv_bridge + YOLO/색추출.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List

from turtlebot3_llm_nav.perception import (
    CameraModel, RobotPose, pixel_to_bearing, estimate_distance,
)


class ObjectDetector:
    """모든 백엔드가 따라야 하는 인터페이스. detect(image, camera, pose) -> List[detection dict]."""

    name = "base"

    def detect(self, image_msg, camera: CameraModel, robot_pose: RobotPose) -> List[dict]:
        raise NotImplementedError


# ── (a) Ground-truth 백엔드 (시뮬레이션용 오라클) ─────────────────
class ObjectGroundTruthDetector(ObjectDetector):
    """**실제 인식이 아님.** Gazebo 가 발행하는 물건 ground-truth pose 를 받아 카메라 FOV 안의
    물건만 '검출'로 투영합니다. 실검출기 없이 폐루프를 돌리기 위한 것.

    입력 토픽 /objects_ground_truth (std_msgs/String) 의 JSON(노드가 set_objects 로 주입):
        [{"id":"cup1","x":1.2,"y":0.9,"features":{"label":"cup","color":"red"}}, ...]
    또는 {"objects":[...]} 래핑도 허용. bearing/distance 는 실제 상대좌표로 정확히 계산하고,
    FOV(±hfov/2) 밖이거나 max_range 초과면 안 보이는 것으로 제외합니다."""

    name = "ground_truth"

    def __init__(self, max_range: float = 4.0, fov_margin: float = 0.0):
        self.max_range = float(max_range)
        self.fov_margin = float(fov_margin)
        self._objects: List[dict] = []

    def set_objects(self, raw) -> None:
        """노드의 /objects_ground_truth 콜백에서 호출. raw 는 JSON 문자열 또는 list/dict."""
        try:
            data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        except Exception:
            self._objects = []
            return
        if isinstance(data, dict):
            data = data.get("objects", [])
        self._objects = list(data) if isinstance(data, list) else []

    def detect(self, image_msg, camera: CameraModel, robot_pose: RobotPose) -> List[dict]:
        out: List[dict] = []
        if robot_pose is None:
            return out
        half_fov = camera.hfov / 2.0 + self.fov_margin
        for o in self._objects:
            try:
                ox, oy = float(o["x"]), float(o["y"])
            except (KeyError, TypeError, ValueError):
                continue
            dx, dy = ox - robot_pose.x, oy - robot_pose.y
            dist = math.hypot(dx, dy)
            if dist > self.max_range or dist < 1e-3:
                continue
            world_ang = math.atan2(dy, dx)
            bearing = math.atan2(math.sin(world_ang - robot_pose.yaw),
                                 math.cos(world_ang - robot_pose.yaw))
            if abs(bearing) > half_fov:
                continue
            conf = 0.5 + 0.5 * max(0.0, math.cos(bearing)) * max(0.0, 1.0 - dist / self.max_range)
            out.append({
                "bearing": bearing,
                "distance": dist,
                "features": dict(o.get("features", {})),
                "conf": round(conf, 3),
                "id": o.get("id"),
            })
        out.sort(key=lambda d: d["distance"])
        return out


# ── (b) 실제 검출기 백엔드 (STUB — 구현 TODO) ─────────────────────
class YoloObjectDetector(ObjectDetector):
    """실제 카메라 영상 기반 물체 검출기 자리(YOLO 물체검출 + 색/라벨 추출).

    **STUB 입니다.** 만족해야 할 계약:
      입력 : image_msg(sensor_msgs/Image), camera(CameraModel), robot_pose(무시 가능)
      출력 : detect() 가 {'bearing','distance','features','conf'} dict 리스트를 돌려준다.
    구현 가이드:
      1) cv_bridge 로 image_msg → numpy BGR.
      2) YOLO 로 물체 bbox 검출 → 각 bbox 중심 cx, 높이 bbox_h, 클래스(label).
      3) bearing = pixel_to_bearing(cx, camera.img_w, camera.hfov)
         distance = estimate_distance(bbox_h, camera.img_h, camera.hfov, camera.img_w, object_h)
         (depth 토픽이 있으면 distance 는 그걸로 대체 — 더 정확.)
      4) features: {'label': 클래스명, 'color': bbox crop HSV 평균색}.
      5) conf: 모델 score.
    의존성(런타임에만): ultralytics(YOLO) 또는 opencv-dnn, cv_bridge, numpy."""

    name = "yolo"

    def __init__(self, model_path: str = "", conf_thr: float = 0.4, object_h_m: float = 0.15):
        self.model_path = model_path
        self.conf_thr = float(conf_thr)
        self.object_h_m = float(object_h_m)
        self._model = None
        self._bridge = None

    def _ensure(self):
        if self._model is None:
            raise NotImplementedError(
                "YoloObjectDetector 는 STUB 입니다. _ensure/_run_model 을 구현하세요 "
                "(ultralytics + cv_bridge). 시뮬레이션은 detector:=ground_truth 를 쓰세요.")

    def _run_model(self, bgr) -> List[Dict]:
        """numpy BGR → [{'cx','bbox_h','features','conf'}, ...] 반환하도록 구현."""
        raise NotImplementedError("YOLO 추론을 여기 구현하세요.")

    def detect(self, image_msg, camera: CameraModel, robot_pose) -> List[dict]:
        self._ensure()
        bgr = self._bridge.imgmsg_to_cv2(image_msg, "bgr8")  # type: ignore[union-attr]
        out: List[dict] = []
        for det in self._run_model(bgr):
            if det.get("conf", 0.0) < self.conf_thr:
                continue
            out.append({
                "bearing": pixel_to_bearing(det["cx"], camera.img_w, camera.hfov),
                "distance": estimate_distance(det["bbox_h"], camera.img_h, camera.hfov,
                                              camera.img_w, self.object_h_m),
                "features": dict(det.get("features", {})),
                "conf": float(det["conf"]),
            })
        return out


def make_object_detector(kind: str = "ground_truth", **kw) -> ObjectDetector:
    """파라미터 detector('ground_truth'|'yolo')로 백엔드 선택."""
    kind = (kind or "ground_truth").lower()
    if kind in ("ground_truth", "gt", "oracle"):
        return ObjectGroundTruthDetector(max_range=kw.get("max_range", 4.0))
    if kind in ("yolo", "real", "detector"):
        return YoloObjectDetector(model_path=kw.get("model_path", ""),
                                  conf_thr=kw.get("conf_thr", 0.4))
    raise ValueError(f"알 수 없는 detector 종류: {kind!r} (ground_truth|yolo)")
