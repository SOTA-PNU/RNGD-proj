"""TurtleBot3 House 도면을 헤드리스 2D 월드로 옮기고, '집 안 물건 확인 후 복귀' 미션을 구성합니다.

이 파일이 하는 일
  1) 실제 Gazebo house 모델(`turtlebot3_house/model.sdf`)의 벽(Wall_* 박스)을 그대로 파싱해
     2D 벽(Segment) 도면을 만듭니다. → 헤드리스 도면이 실제 시뮬과 같은 방·문 배치가 됩니다.
     (출처: turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_house/model.sdf)
  2) 도면을 격자로 만들어(로봇 반지름만큼 부풀림) **전역 플래너(BFS)** 로 집 안을 도는 경로를
     계산하고, 직선가시(LOS)로 듬성하게 줄여 **웨이포인트 경로**를 만듭니다.
     이 경로는 로봇에게 '주어지는' 것입니다(Autoware 의 planning 노드 역할). 로봇(LLM)은 이
     경로를 **따라가며 장애물 회피·물체 스캔·복귀·판정하는 로컬 컨트롤러를 직접 코딩**합니다.
  3) 집 안에 물건(Item)들을 둡니다. 찾는 물건(target)과, 한 가지 특징만 같은 헷갈리는 물건
     (decoy)을 섞어, '대충 색만 보고' 판정하면 틀리도록(false_report) 만듭니다.

좌표계: 실제 도면(원점 집 중앙, x∈[-7.6,7.6], y∈[-5.4,5.4])을 (+8,+6) 평행이동해 16×12 m
        양수 평면으로 옮깁니다. home(현관)은 실제 spawn 기본값 (-2.0,-0.5) → (6.0,5.5).
        (출처: turtlebot3_gazebo/launch/turtlebot3_house.launch.py 의 x_pose/y_pose 기본값)
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from collections import deque
from typing import Dict, List, Optional, Tuple

from world import Item, Segment, World

# 도면 평행이동량과 헤드리스 월드 크기(실제 도면 extent 가 [0.4,15.6]×[0.6,11.4] 안에 들어옴).
SHIFT_X, SHIFT_Y = 8.0, 6.0
WORLD_W, WORLD_H = 16.0, 12.0
HOME = (-2.0 + SHIFT_X, -0.5 + SHIFT_Y)     # 현관(스폰 기본값) → (6.0, 5.5)

_SDF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "turtlebot3-llm", "turtlebot3_simulations",
                    "turtlebot3_gazebo", "models", "turtlebot3_house", "model.sdf")


# ── 1) 실제 house 벽 파싱 ─────────────────────────────────────────
def load_walls(sdf_path: str = _SDF) -> List[Dict]:
    """house model.sdf 의 Wall_* 링크를 (중심·각도·길이·두께) 벽 목록으로 파싱(평행이동 적용).

    주의: 각 벽 링크는 link 의 <pose> 와, 그 안의 <collision>/<visual> '하위 <pose>'(길이축 평면
    오프셋)를 함께 가집니다(21개 중 11개가 0이 아닌 길이축 오프셋, 최대 3.34m). 둘을 **합성**해야
    실제 도면과 일치합니다: world중심 = link_xy + R(link_yaw)·(sub_x, sub_y), 각도 = link_yaw+sub_yaw.
    (출처: model.sdf 의 Wall_104/108/106… 하위 pose 실측)"""
    with open(sdf_path, encoding="utf-8") as f:
        root = ET.fromstring(f.read())
    model = root.find("model")
    walls: List[Dict] = []
    for link in model.findall("link"):
        pose = link.find("pose")
        if pose is None:
            continue
        lv = [float(v) for v in pose.text.split()]
        if len(lv) < 6:
            continue
        lx, ly, lyaw = lv[0], lv[1], lv[5]
        size = None
        sub = (0.0, 0.0, 0.0)
        for tag in ("collision", "visual"):
            el = link.find(tag)
            if el is not None and el.find("geometry") is not None:
                box = el.find("geometry").find("box")
                if box is not None:
                    size = [float(v) for v in box.find("size").text.split()]
                    sp = el.find("pose")
                    if sp is not None:
                        sv = [float(v) for v in sp.text.split()]
                        if len(sv) >= 6:
                            sub = (sv[0], sv[1], sv[5])
                    break
        if size is None:
            continue
        # 링크 pose ⊕ 하위 pose 합성(평면 회전 적용)
        cx = lx + math.cos(lyaw) * sub[0] - math.sin(lyaw) * sub[1]
        cy = ly + math.sin(lyaw) * sub[0] + math.cos(lyaw) * sub[1]
        walls.append({"cx": cx + SHIFT_X, "cy": cy + SHIFT_Y, "theta": lyaw + sub[2],
                      "L": size[0], "T": size[1]})
    return walls


# ── 2) 격자 + 전역 플래너(BFS) ────────────────────────────────────
class _Grid:
    """벽을 로봇 반지름만큼 부풀린 점유격자. 자유공간 판정·BFS 경로·직선가시(LOS) 검사를 제공."""

    def __init__(self, walls: List[Dict], inflate: float, res: float = 0.1):
        self.res = res
        self.nx = int(WORLD_W / res)
        self.ny = int(WORLD_H / res)
        self.inflate = inflate
        self._segs = [Segment(w["cx"], w["cy"], w["theta"], w["L"], w["T"]) for w in walls]
        # 점유격자: 셀 중심이 벽(부풀림)에 닿거나 경계 밖이면 막힘
        self.occ = [[self._blocked(c, r) for c in range(self.nx)] for r in range(self.ny)]

    def _xy(self, c: int, r: int) -> Tuple[float, float]:
        return (c + 0.5) * self.res, (r + 0.5) * self.res

    def _blocked(self, c: int, r: int) -> bool:
        x, y = self._xy(c, r)
        if x < self.inflate or x > WORLD_W - self.inflate or \
           y < self.inflate or y > WORLD_H - self.inflate:
            return True
        return any(s.dist_to(x, y) < self.inflate for s in self._segs)

    def cell(self, x: float, y: float) -> Tuple[int, int]:
        return (min(self.nx - 1, max(0, int(x / self.res))),
                min(self.ny - 1, max(0, int(y / self.res))))

    def free(self, x: float, y: float) -> bool:
        c, r = self.cell(x, y)
        return not self.occ[r][c]

    def nearest_free(self, x: float, y: float) -> Tuple[float, float]:
        """(x,y)가 막혔으면 가장 가까운 자유셀 중심으로 당겨 줍니다(현관/물체 배치 보정)."""
        c0, r0 = self.cell(x, y)
        if not self.occ[r0][c0]:
            return x, y
        for rad in range(1, 40):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    c, r = c0 + dc, r0 + dr
                    if 0 <= c < self.nx and 0 <= r < self.ny and not self.occ[r][c]:
                        return self._xy(c, r)
        return x, y

    def reachable(self, start: Tuple[float, float]) -> set:
        """start 에서 4방 BFS 로 닿는 자유셀 집합."""
        sc = self.cell(*start)
        seen = {sc}
        q = deque([sc])
        while q:
            c, r = q.popleft()
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nc, nr = c + dc, r + dr
                if 0 <= nc < self.nx and 0 <= nr < self.ny and (nc, nr) not in seen \
                        and not self.occ[nr][nc]:
                    seen.add((nc, nr))
                    q.append((nc, nr))
        return seen

    def bfs_path(self, a: Tuple[float, float], b: Tuple[float, float]) -> List[Tuple[float, float]]:
        """a→b 4방 BFS 최단경로(셀 중심 좌표열). 닿지 못하면 빈 리스트."""
        sc, gc = self.cell(*a), self.cell(*b)
        if self.occ[sc[1]][sc[0]] or self.occ[gc[1]][gc[0]]:
            return []
        prev = {sc: None}
        q = deque([sc])
        while q:
            cur = q.popleft()
            if cur == gc:
                break
            c, r = cur
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nc, nr = c + dc, r + dr
                if 0 <= nc < self.nx and 0 <= nr < self.ny and (nc, nr) not in prev \
                        and not self.occ[nr][nc]:
                    prev[(nc, nr)] = cur
                    q.append((nc, nr))
        if gc not in prev:
            return []
        out = []
        cur = gc
        while cur is not None:
            out.append(self._xy(*cur))
            cur = prev[cur]
        out.reverse()
        return out

    def los(self, a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        """a→b 직선이 부풀린 벽을 지나지 않으면 True(직선가시)."""
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(d / (self.res * 0.7)))
        for i in range(n + 1):
            t = i / n
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            if not self.free(x, y):
                return False
        return True

    def resample(self, path: List[Tuple[float, float]], step: float = 0.4) -> List[Tuple[float, float]]:
        """촘촘한 BFS 경로(센터라인, 벽에서 inflate 만큼 떨어져 있음)를 일정 간격(step)으로 다시
        샘플링한 웨이포인트로 줄입니다. LOS 로 코너를 가로지르지 않으므로(센터라인을 그대로 따라감),
        로봇이 이 점들을 차례로 좇으면 항상 여유거리를 두고 벽을 피해 갑니다."""
        if not path:
            return []
        out = [path[0]]
        acc = 0.0
        prev = path[0]
        for p in path[1:]:
            acc += math.hypot(p[0] - prev[0], p[1] - prev[1])
            if acc >= step:
                out.append(p)
                acc = 0.0
            prev = p
        if out[-1] != path[-1]:
            out.append(path[-1])
        return out


def _farthest_points(grid: _Grid, home: Tuple[float, float], k: int) -> List[Tuple[float, float]]:
    """home 에서 닿는 자유공간에서 서로 멀리 떨어진 방문지(방 vantage) k개를 결정적으로 고릅니다.
    (farthest-point sampling: 먼저 home 에서 가장 먼 곳, 그다음 고른 점들에서 가장 먼 곳…)"""
    reach = [grid._xy(c, r) for (c, r) in grid.reachable(home)]
    chosen = [home]
    for _ in range(k):
        best, bestd = None, -1.0
        for p in reach:
            d = min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in chosen)
            if d > bestd:
                bestd, best = d, p
        if best is None:
            break
        chosen.append(best)
    return chosen[1:]


# ── 3) 미션 월드 빌드 ─────────────────────────────────────────────
def build_house(objective: Dict, present: bool, name: str, *,
                robot_radius: float = 0.15, cam_range: float = 4.0,
                n_rooms: int = 4) -> Tuple[World, Tuple[float, float], Tuple[float, float], str]:
    """집 안에서 objective 물건이 있는지 **자율 탐색**으로 확인하고 현관(home)으로 돌아오는 미션 월드.
    present=True 면 objective 와 일치하는 진짜 물건을 두고, False 면 두지 않습니다(decoy 만).
    **미리 정해둔 이동 경로(waypoints)는 주지 않습니다** — 로봇은 LiDAR·카메라만으로 스스로 집을
    돌아다녀야 합니다(자율주행). agent 의 anti-guess 게이트는 '실제로 얼마나 돌아다녔는지'(reachable
    cell coverage)를 독립 측정해, 대충 보고만으로 통과하지 못하게 합니다.
    반환: (World, start=home, goal=home, name) — agent 가 world.house_task 로 미션 루프를 탑니다."""
    walls = load_walls()
    grid = _Grid(walls, inflate=robot_radius + 0.18)
    home = grid.nearest_free(*HOME)

    # 방 vantage(서로 멀리 떨어진 지점)는 '물건 배치'에만 쓰고, 경로로는 주지 않습니다(자율탐색).
    vantages = _farthest_points(grid, home, n_rooms)

    # 물건 배치: 진짜 물건은 한 vantage 옆(보이게), decoy 는 다른 vantage 옆(한 특징만 일치).
    items: List[Item] = []
    label = objective.get("label", "object")
    color = objective.get("color", "red")
    alt_color = "red" if color != "red" else "green"          # objective 와 다른 색(decoy 가 정답과 같아지지 않게)
    decoy_specs = [{"label": "book", "color": color},          # 색만 같음(라벨 다름)
                   {"label": label, "color": alt_color}]       # 라벨만 같음(색 다름)
    for i, v in enumerate(vantages):
        spec = decoy_specs[i % len(decoy_specs)] if i < len(vantages) - 1 else None
        if i == len(vantages) - 1 and present:                 # 마지막 vantage 옆에 진짜 물건
            spec = dict(objective)
        if spec is None:
            continue
        px, py = grid.nearest_free(v[0] + 0.9, v[1] + 0.6)
        items.append(Item(px, py, dict(spec)))

    world = World(
        WORLD_W, WORLD_H, obstacles=[], walls=walls, items=items,
        robot_radius=robot_radius, goal_tol=0.6, v_max=1.2, w_max=2.0, dt=0.1,
        n_rays=16, max_range=5.0, cam_fov=1.0297, cam_range=cam_range,
        house_task=True, objective=dict(objective), waypoints=[], home=home,
        present=present,
        mission=("Autonomously explore the house with LiDAR and camera to check whether the target "
                 "object is present, then return to home."),
    )
    # agent anti-guess 용: home 에서 닿는 거친(1m) 셀 집합(자율탐색이 얼마나 돌았는지 독립 측정)
    world.cover_cells = set((int(grid._xy(c, r)[0]), int(grid._xy(c, r)[1]))
                            for (c, r) in grid.reachable(home))
    return world, home, home, name


def build_house_nav(goal: Optional[Tuple[float, float]] = None, name: str = "house_nav",
                    robot_radius: float = 0.15) -> Tuple[World, Tuple[float, float], Tuple[float, float], str]:
    """집 도면(House 벽) 위에서 현관(home)→목표 좌표까지 가는 '점-이동' 미션(일반 nav 계약).
    물건 검색이 아니라, LiDAR 로 벽을 피해 목표 좌표까지 가는 짧은 컨트롤러를 LLM 이 코딩합니다.
    goal 이 없으면 home 에서 가장 먼 자유공간(거실 안쪽)을 목표로 잡습니다."""
    walls = load_walls()
    grid = _Grid(walls, inflate=robot_radius + 0.18)
    home = grid.nearest_free(*HOME)
    if goal is None:
        far = _farthest_points(grid, home, 1)
        goal = far[0] if far else (home[0] + 3.0, home[1])
    goal = grid.nearest_free(*goal)
    world = World(
        WORLD_W, WORLD_H, obstacles=[], walls=walls,
        robot_radius=robot_radius, goal_tol=0.6, v_max=1.2, w_max=2.0, dt=0.1,
        n_rays=16, max_range=5.0,
    )
    return world, home, goal, name
