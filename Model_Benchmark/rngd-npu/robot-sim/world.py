"""2D 연속 평면 월드 · 유니사이클(차동구동) 로봇 · LiDAR 광선추적.

로봇은 자기 위치(pos)·방향(heading)·목표(goal)와 LiDAR 거리값만 알고,
그 정보로 LLM이 짜준 `plan(state)` 컨트롤러를 매 제어주기마다 실행해 스스로 움직입니다.

설계 원칙
  * 순수 표준 라이브러리(math)만 사용합니다. numpy 없이도 시스템 python3 로 그대로 돕니다.
  * 거리/충돌/광선추적은 모두 해석적으로(분석식) 계산해 빠르고 결정적입니다.
  * 단위는 미터/라디안/초. 좌표계는 +x 가 0 rad, 반시계(CCW)가 양(+) 각도입니다.

근거: LiDAR 기반 지역 회피 + 목표 추종은 자율주행(Autoware)의 perception→planning→control
파이프라인을 단순화한 것입니다. 광선-원 교차/광선-사각경계 교차는 표준 계산기하 공식입니다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


# ── 기하 요소 ─────────────────────────────────────────────────────
class Circle:
    """원형 장애물. 충돌·광선추적의 기본 단위입니다(사각형도 원 몇 개로 근사 가능)."""
    __slots__ = ("cx", "cy", "r")

    def __init__(self, cx: float, cy: float, r: float):
        self.cx, self.cy, self.r = float(cx), float(cy), float(r)

    def signed_dist(self, x: float, y: float) -> float:
        """점에서 원 표면까지의 부호거리(안쪽이면 음수). 충돌 판정에 씁니다."""
        return math.hypot(x - self.cx, y - self.cy) - self.r


class Segment:
    """두께가 있는 직사각형 벽(집 도면의 벽 한 장). 중심(cx,cy)·길이방향 각도(theta)·길이(L)·두께(T)로
    정의합니다. 충돌·광선추적을 벽 로컬좌표(길이축 a, 수직축 b)에서 해석적으로 계산해 빠르고 정확합니다.
    (집 도면은 실제 turtlebot3_house/model.sdf 의 Wall_* 박스를 그대로 옮긴 것 — house_world.py)"""
    __slots__ = ("cx", "cy", "ux", "uy", "hl", "ht")

    def __init__(self, cx: float, cy: float, theta: float, length: float, thickness: float):
        self.cx, self.cy = float(cx), float(cy)
        self.ux, self.uy = math.cos(theta), math.sin(theta)   # 길이방향 단위벡터
        self.hl = float(length) / 2.0
        self.ht = float(thickness) / 2.0

    def dist_to(self, x: float, y: float) -> float:
        """점 (x,y)에서 이 벽(사각형) 표면까지 거리(안이면 0). 충돌·여유거리 판정에 사용."""
        dx, dy = x - self.cx, y - self.cy
        a = dx * self.ux + dy * self.uy              # 길이방향 성분
        b = -dx * self.uy + dy * self.ux             # 수직방향 성분
        da = max(abs(a) - self.hl, 0.0)
        db = max(abs(b) - self.ht, 0.0)
        return math.hypot(da, db)

    def raycast_t(self, x: float, y: float, dx: float, dy: float) -> float:
        """광선 (x,y)+t·(dx,dy) 가 이 벽에 처음 닿는 t>0 (없으면 inf). 회전 슬랩(slab) 교차."""
        rx, ry = x - self.cx, y - self.cy
        ox = rx * self.ux + ry * self.uy
        oy = -rx * self.uy + ry * self.ux
        ddx = dx * self.ux + dy * self.uy
        ddy = -dx * self.uy + dy * self.ux
        tmin, tmax = -1e18, 1e18
        for o, d, h in ((ox, ddx, self.hl), (oy, ddy, self.ht)):
            if abs(d) < 1e-12:
                if o < -h or o > h:
                    return float("inf")              # 슬랩 밖에서 평행 → 안 만남
            else:
                t1, t2 = (-h - o) / d, (h - o) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    return float("inf")
        if tmax < 0:
            return float("inf")
        return tmin if tmin > 1e-9 else tmax


@dataclass
class Robot:
    """로봇 상태. 위치(x, y)·방향(heading, rad). 경로는 월드가 기록합니다."""
    x: float
    y: float
    heading: float = 0.0


@dataclass
class Person:
    """카메라로 찾을 사람. 위치(x, y) + 특징(features: 색·키 등 dict).
    '특정 사람에게 도달'은 features 가 target 과 맞는 사람을 카메라로 식별해 다가가는 과제입니다."""
    x: float
    y: float
    features: dict = field(default_factory=dict)


@dataclass
class Item:
    """집 안의 물건(집기). 위치(x, y) + 특징(features: label·color 등 dict).
    '특정 물건이 있는지 확인'은 features 가 objective 의 모든 항목과 일치하는 물건을 카메라로 보는 과제입니다."""
    x: float
    y: float
    features: dict = field(default_factory=dict)


# ── 월드 ──────────────────────────────────────────────────────────
class World:
    """직사각형 경계 + 원형 장애물 + 유니사이클 로봇이 들어 있는 시뮬레이션 월드."""

    def __init__(
        self,
        width: float = 24.0,
        height: float = 24.0,
        obstacles: Optional[Sequence[Circle]] = None,
        robot_radius: float = 0.35,
        goal_tol: float = 0.6,
        v_max: float = 1.6,
        w_max: float = 2.2,
        dt: float = 0.1,
        n_rays: int = 16,
        max_range: float = 6.0,
        people: Optional[Sequence["Person"]] = None,
        target: Optional[dict] = None,
        cam_fov: float = 1.0297,      # 카메라 수평 화각(rad) — waffle RealSense R200 과 동일(~59°)
        cam_range: float = 9.0,       # 카메라로 사람을 인지하는 최대 거리(m)
        vision_task: bool = False,    # True 면 컨트롤러에 목표 '좌표'를 숨기고 카메라로만 찾게 함
        walls: Optional[Sequence[dict]] = None,   # 집 도면 벽(Segment 파라미터 dict 목록)
        items: Optional[Sequence["Item"]] = None,  # 집 안 물건들
        house_task: bool = False,     # True 면 '집 안 물건 확인 후 복귀' 미션
        objective: Optional[dict] = None,          # 찾는 물건 특징(예 {'label':'cup','color':'red'})
        waypoints: Optional[Sequence[Tuple[float, float]]] = None,  # 전역 플래너가 준 방문 경로
        home: Optional[Tuple[float, float]] = None,                 # 복귀할 현관 좌표
        present: bool = False,        # (정답) objective 물건이 실제로 집에 있는가
        mission: str = "",            # 미션 설명 문구(프롬프트용)
    ):
        self.width = float(width)
        self.height = float(height)
        self.obstacles: List[Circle] = list(obstacles or [])
        self.robot_radius = float(robot_radius)
        self.goal_tol = float(goal_tol)
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.dt = float(dt)
        self.n_rays = int(n_rays)
        self.max_range = float(max_range)
        self.people: List["Person"] = list(people or [])
        self.target: Optional[dict] = dict(target) if target else None
        self.cam_fov = float(cam_fov)
        self.cam_range = float(cam_range)
        self.vision_task = bool(vision_task)
        # ── 집 미션(house_task) ────────────────────────────────────
        self.walls: List[Segment] = [
            w if isinstance(w, Segment) else Segment(w["cx"], w["cy"], w["theta"], w["L"], w["T"])
            for w in (walls or [])]
        self.items: List["Item"] = list(items or [])
        self.house_task = bool(house_task)
        self.objective: Optional[dict] = dict(objective) if objective else None
        self.waypoints: List[Tuple[float, float]] = [tuple(p) for p in (waypoints or [])]
        self.home: Optional[Tuple[float, float]] = tuple(home) if home else None
        self.present = bool(present)
        self.mission = str(mission)

    # ── 충돌 ──────────────────────────────────────────────────────
    def in_bounds(self, x: float, y: float) -> bool:
        r = self.robot_radius
        return (r <= x <= self.width - r) and (r <= y <= self.height - r)

    def collides(self, x: float, y: float) -> bool:
        """로봇 중심이 (x, y)일 때 경계 밖이거나 장애물에 닿으면 True."""
        if not self.in_bounds(x, y):
            return True
        rr = self.robot_radius
        for ob in self.obstacles:
            if math.hypot(x - ob.cx, y - ob.cy) < ob.r + rr:
                return True
        for wseg in self.walls:
            if wseg.dist_to(x, y) < rr:
                return True
        return False

    # ── 광선추적(LiDAR 한 줄기) ───────────────────────────────────
    def raycast(self, x: float, y: float, ang: float, max_dist: float = None) -> float:
        """(x, y)에서 월드각 ang 방향으로 쏜 광선이 가장 먼저 만나는 면까지의 거리.
        장애물(원)과 사방 벽 중 가장 가까운 교차점을 반환하며 max_dist(없으면 LiDAR max_range)로 자릅니다.
        카메라 가림 검사처럼 LiDAR 사거리보다 먼 거리까지 봐야 할 땐 max_dist 를 넘깁니다."""
        dx, dy = math.cos(ang), math.sin(ang)
        best = self.max_range if max_dist is None else float(max_dist)

        # 벽(직사각 경계): 로봇은 항상 경계 안에 있으므로 광선은 어딘가에서 벽을 만난다.
        if dx > 1e-9:
            t = (self.width - x) / dx
            if 0 <= t < best:
                best = t
        elif dx < -1e-9:
            t = (0.0 - x) / dx
            if 0 <= t < best:
                best = t
        if dy > 1e-9:
            t = (self.height - y) / dy
            if 0 <= t < best:
                best = t
        elif dy < -1e-9:
            t = (0.0 - y) / dy
            if 0 <= t < best:
                best = t

        # 장애물(원): |O + t·D − C|² = R² 를 풀어 가장 가까운 양의 t 를 찾는다.
        for ob in self.obstacles:
            fx, fy = x - ob.cx, y - ob.cy
            b = fx * dx + fy * dy
            c = fx * fx + fy * fy - ob.r * ob.r
            disc = b * b - c
            if disc < 0:
                continue
            sq = math.sqrt(disc)
            t = -b - sq
            if t < 0:
                t = -b + sq
            if 0 <= t < best:
                best = t

        # 벽(직사각 벽): 회전 슬랩 교차로 가장 가까운 양의 t.
        for wseg in self.walls:
            t = wseg.raycast_t(x, y, dx, dy)
            if 0 <= t < best:
                best = t
        return best

    def lidar(self, robot: Robot) -> Tuple[List[float], List[float]]:
        """로봇 둘레 n_rays 개 광선의 거리값과 (로봇 기준) 각도 목록.
        index 0 = 정면(heading 방향), 이후 반시계로 360°를 균등 분할합니다.
        거리값이 max_range 면 '그 방향엔 아무것도 없음'을 뜻합니다."""
        dists: List[float] = []
        angles: List[float] = []
        for i in range(self.n_rays):
            a_robot = 2.0 * math.pi * i / self.n_rays
            a_robot = math.atan2(math.sin(a_robot), math.cos(a_robot))  # [-pi, pi)
            d = self.raycast(robot.x, robot.y, robot.heading + a_robot)
            dists.append(d)
            angles.append(a_robot)
        return dists, angles

    # ── 한 스텝 전진(유니사이클 적분) ─────────────────────────────
    def step(self, robot: Robot, v: float, w: float, substeps: int = 6) -> bool:
        """제어입력 (v 선속도, w 각속도)를 dt 동안 적분해 로봇을 옮깁니다.
        장애물을 뚫지 않도록 dt 를 잘게 나눠(substeps) 충돌을 확인합니다.
        충돌하면 직전 위치에 멈추고 True 를 반환합니다."""
        v = _clamp(v, -self.v_max, self.v_max)
        w = _clamp(w, -self.w_max, self.w_max)
        ddt = self.dt / substeps
        for _ in range(substeps):
            nx = robot.x + v * math.cos(robot.heading) * ddt
            ny = robot.y + v * math.sin(robot.heading) * ddt
            nh = robot.heading + w * ddt
            if self.collides(nx, ny):
                robot.heading = _wrap(nh)   # 회전은 허용(제자리 회전으로 탈출 가능)
                return True
            robot.x, robot.y, robot.heading = nx, ny, _wrap(nh)
        return False

    # ── 카메라(사람 검출) ─────────────────────────────────────────
    def camera_view(self, robot: Robot) -> List[dict]:
        """로봇 카메라(전방 cam_fov 화각)에 들어온 사람들의 검출 목록.
        각 검출 = {bearing(로봇기준 각, rad), distance(m), features(특징 dict), conf(신뢰도 0~1)}.
        FOV 밖·cam_range 초과·장애물에 가려진 사람은 안 보입니다. 멀수록 conf 가 낮아
        특징 식별이 불확실해집니다(현실의 '특정 사람 식별' 어려움 반영)."""
        dets = []
        for p in self.people:
            dx, dy = p.x - robot.x, p.y - robot.y
            dist = math.hypot(dx, dy)
            if dist > self.cam_range or dist < 1e-3:
                continue
            world_ang = math.atan2(dy, dx)
            bearing = _wrap(world_ang - robot.heading)
            if abs(bearing) > self.cam_fov / 2.0:           # 화각 밖
                continue
            # 가림 검사는 사람 거리까지 봐야 하므로 max_dist=dist (LiDAR 6m 제한에 걸리면 안 됨)
            if self.raycast(robot.x, robot.y, world_ang, max_dist=dist) < dist - 0.3:
                continue
            conf = max(0.25, 1.0 - dist / self.cam_range)   # 멀수록 식별 신뢰도↓
            dets.append({"bearing": round(bearing, 4), "distance": round(dist, 3),
                         "features": dict(p.features), "conf": round(conf, 3)})
        dets.sort(key=lambda d: abs(d["bearing"]))          # 정면에 가까운 순
        return dets

    # ── 관측(컨트롤러에 넘길 state) ───────────────────────────────
    def observe(self, robot: Robot, goal: Tuple[float, float], memory: dict) -> dict:
        """LLM이 짠 plan(state) 가 받는 관측 딕셔너리를 만듭니다.
        일반 과제: 목표 좌표(goal)를 줍니다. vision_task(사람찾기): 좌표를 숨기고(None) 카메라 검출과
        target 특징만 줘, 카메라로 '특정 사람'을 식별해 다가가게 합니다.
        memory 는 호출 사이에 유지되는 내부 상태 저장소입니다(검색 방향, 정체 카운터 등)."""
        dists, angles = self.lidar(robot)
        state = {
            "pos": (robot.x, robot.y),
            "heading": robot.heading,
            "goal": None if self.vision_task else (float(goal[0]), float(goal[1])),
            "lidar": dists,
            "lidar_angles": angles,
            "max_range": self.max_range,
            "v_max": self.v_max,
            "w_max": self.w_max,
            "dt": self.dt,
            "robot_radius": self.robot_radius,
            "goal_tol": self.goal_tol,
            "bounds": (self.width, self.height),
            "memory": memory,
        }
        if self.vision_task:
            state["camera"] = self.camera_view(robot)   # 전방 사람 검출
            state["target"] = dict(self.target) if self.target else {}
            state["cam_fov"] = self.cam_fov
        return state

    def goal_reached(self, robot: Robot, goal: Tuple[float, float]) -> bool:
        return math.hypot(robot.x - goal[0], robot.y - goal[1]) <= self.goal_tol

    def matches_target(self, person: "Person") -> bool:
        """사람이 target 특징을 '전부' 만족하면 True(진짜 목표). 일부만 같은 decoy 는 False."""
        if not self.target:
            return False
        return all(person.features.get(k) == v for k, v in self.target.items())

    def target_in_view(self, robot: Robot) -> bool:
        """현재 카메라에 target(전체 특징 일치)인 사람이 한 명이라도 잡히면 True."""
        return any(all(det["features"].get(k) == v for k, v in (self.target or {}).items())
                   for det in self.camera_view(robot))

    # ── 물체 인식 카메라(집 미션) ─────────────────────────────────
    def scan_view(self, robot: Robot) -> List[dict]:
        """로봇 카메라(전방 cam_fov)에 들어온 물건들의 검출 목록.
        각 검출 = {bearing, distance, features(label·color 등), conf}. FOV 밖·cam_range 초과·
        벽이나 장애물에 가려진 물건은 안 보입니다(멀수록 conf 낮음 — 식별 불확실)."""
        dets = []
        for it in self.items:
            dx, dy = it.x - robot.x, it.y - robot.y
            dist = math.hypot(dx, dy)
            if dist > self.cam_range or dist < 1e-3:
                continue
            world_ang = math.atan2(dy, dx)
            bearing = _wrap(world_ang - robot.heading)
            if abs(bearing) > self.cam_fov / 2.0:
                continue
            if self.raycast(robot.x, robot.y, world_ang, max_dist=dist) < dist - 0.3:
                continue                                  # 벽/장애물에 가림
            conf = max(0.25, 1.0 - dist / self.cam_range)
            dets.append({"bearing": round(bearing, 4), "distance": round(dist, 3),
                         "features": dict(it.features), "conf": round(conf, 3)})
        dets.sort(key=lambda d: abs(d["bearing"]))
        return dets

    def item_matches(self, feat: dict) -> bool:
        """물건 특징이 objective 의 모든 항목과 일치하면 True(한 특징만 같은 decoy 는 False)."""
        if not self.objective:
            return False
        return all(feat.get(k) == v for k, v in self.objective.items())

    def objective_present(self) -> bool:
        """(정답) objective 와 일치하는 물건이 집 안에 실제로 있는가."""
        return any(self.item_matches(it.features) for it in self.items)

    def observe_house(self, robot: Robot, memory: dict) -> dict:
        """집 미션 컨트롤러 plan(state) 가 받는 관측(자율탐색). **미리 정한 경로(waypoints)는 주지
        않습니다** — 로봇은 LiDAR(lidar)·카메라(scan)·현관 좌표(home)·찾는 물건(objective)만 알고,
        스스로 집을 돌아다니며 스캔→판정→복귀하는 컨트롤러를 코딩해야 합니다."""
        dists, angles = self.lidar(robot)
        return {
            "pos": (robot.x, robot.y),
            "heading": robot.heading,
            "lidar": dists,
            "lidar_angles": angles,
            "max_range": self.max_range,
            "v_max": self.v_max,
            "w_max": self.w_max,
            "dt": self.dt,
            "robot_radius": self.robot_radius,
            "goal_tol": self.goal_tol,
            "bounds": (self.width, self.height),
            "memory": memory,
            "mission": self.mission,
            "objective": dict(self.objective or {}),
            "scan": self.scan_view(robot),
            "home": list(self.home) if self.home else [robot.x, robot.y],
        }

    def clearance(self, x: float, y: float) -> float:
        """현재 위치에서 가장 가까운 장애물/벽 표면까지의 여유 거리(안전마진 분석용)."""
        m = min(x, self.width - x, y, self.height - y)
        for ob in self.obstacles:
            m = min(m, math.hypot(x - ob.cx, y - ob.cy) - ob.r)
        for wseg in self.walls:
            m = min(m, wseg.dist_to(x, y))
        return m


# ── 작은 도우미 ───────────────────────────────────────────────────
def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _wrap(a: float) -> float:
    """각도를 [-pi, pi) 로 정규화."""
    return math.atan2(math.sin(a), math.cos(a))
