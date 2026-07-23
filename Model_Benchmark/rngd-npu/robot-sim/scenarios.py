"""난이도별 시나리오(맵 + 시작/목표). 각 시나리오는 '코딩 성능'의 다른 면을 시험합니다.

  open   : 장애물 없음 — 목표추종·각도제어가 맞는지(기본기).
  single : 정중앙 큰 장애물 하나 — 회피의 기본.
  slalom : 지그재그 장애물 — 연속 회피 + 목표복귀.
  trap   : 오목한 U자벽(지역최소) — 'naive 목표추종'은 갇힘. 벽타기/버그알고리즘으로 탈출해야 통과.
  random : 시드 고정 무작위 장애물밭 — 일반화.

trap 은 LLM이 state['memory'] 로 정체를 감지하고 탈출 로직을 '코딩'해야 풀리므로,
코딩 성능 차이를 가장 잘 드러냅니다.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Tuple

from world import Circle, Person, World

Scenario = Tuple[World, Tuple[float, float], Tuple[float, float], str]


def _open() -> Scenario:
    w = World(24, 24, obstacles=[])
    return w, (3.0, 3.0), (21.0, 21.0), "open"


def _single() -> Scenario:
    w = World(24, 24, obstacles=[Circle(12, 12, 3.0)])
    return w, (3.0, 12.0), (21.0, 12.0), "single"


def _slalom() -> Scenario:
    obs = [Circle(8, 9, 1.7), Circle(13, 15, 1.9), Circle(18, 9, 1.7)]
    w = World(24, 24, obstacles=obs)
    return w, (3.0, 12.0), (21.0, 12.0), "slalom"


def _trap() -> Scenario:
    """왼쪽으로 열린 U자(컵). 로봇은 목표(오른쪽)로 가다 컵 안 막힌 벽에 갇힌다.
    탈출하려면 일시적으로 목표 반대(왼쪽)로 빠져나와 위/아래로 우회해야 한다(지역최소)."""
    obs: List[Circle] = []
    for y in [v * 1.4 for v in range(6, 12)]:           # 막힌 벽(오른쪽), x=13, y≈8.4..15.4
        obs.append(Circle(13.0, y, 0.95))
    for x in [7 + i * 1.2 for i in range(6)]:             # 위 뚜껑 y=15.4
        obs.append(Circle(x, 15.4, 0.95))
    for x in [7 + i * 1.2 for i in range(6)]:             # 아래 뚜껑 y=8.4
        obs.append(Circle(x, 8.4, 0.95))
    w = World(26, 24, obstacles=obs)
    return w, (4.0, 12.0), (22.0, 12.0), "trap"


def _random(seed: int = 7) -> Scenario:
    rng = random.Random(seed)
    start, goal = (3.0, 3.0), (21.0, 21.0)
    obs: List[Circle] = []
    tries = 0
    while len(obs) < 8 and tries < 400:
        tries += 1
        cx = rng.uniform(4, 20)
        cy = rng.uniform(4, 20)
        r = rng.uniform(1.0, 2.2)
        # 시작/목표 근처는 비워 둔다(시작 가능·도달 가능 보장)
        if math.hypot(cx - start[0], cy - start[1]) < r + 2.0:
            continue
        if math.hypot(cx - goal[0], cy - goal[1]) < r + 2.0:
            continue
        if any(math.hypot(cx - o.cx, cy - o.cy) < r + o.r + 0.5 for o in obs):
            continue
        obs.append(Circle(cx, cy, r))
    w = World(24, 24, obstacles=obs)
    return w, start, goal, "random"


def _find_person() -> Scenario:
    """카메라로 '특정 사람'(빨간 옷 + 모자)에게 도달. TurtleBot3 waffle 카메라 과제의 헤드리스 판.
    난이도(=실패 유발 요소): 같은 빨강 옷의 decoy 가 '정면 더 가까이' 서 있어, 옷 색만 보고 가면
    엉뚱한 사람에게 도착(wrong_person). 진짜 target 은 모자(cap)까지 맞는 유일한 사람이라, 보조
    특징으로 식별하도록 코드를 고쳐야 풉니다. (vision 과제는 시작 방향을 정면(+x)으로 고정.)"""
    obs = [Circle(7, 7, 1.3), Circle(16, 9, 1.3)]     # 경로(좌하→우상) 밖 소품 장애물
    people = [
        Person(17.0, 17.0, {"shirt": "red", "cap": True}),    # ← target: 유일한 red+cap, 우상단
        Person(10.0, 12.0, {"shirt": "red", "cap": False}),   # decoy: 정면 더 가까이, 빨강·모자X
        Person(15.0, 7.0, {"shirt": "blue", "cap": False}),
        Person(6.0, 18.0, {"shirt": "green", "cap": True}),
    ]
    target = {"shirt": "red", "cap": True}
    w = World(22, 22, obstacles=obs, people=people, target=target, vision_task=True,
              goal_tol=1.0, cam_range=22.0)
    return w, (3.0, 12.0), (people[0].x, people[0].y), "find_person"


def _house_search() -> Scenario:
    """집(TurtleBot3 House) 안을 돌며 '빨간 컵'이 있는지 확인하고 현관으로 복귀. 빨간 컵은 실제로 있음
    (정답=present). 같은 빨강 책·파란 컵 decoy 가 섞여 있어, 색만 보고 판정하면 false_report."""
    import house_world as HW
    return HW.build_house({"label": "cup", "color": "red"}, present=True, name="house_search")


def _house_search_absent() -> Scenario:
    """같은 집이지만 '빨간 컵'이 없음(정답=absent). decoy(빨강 책·파란 컵)만 있어, 끝까지 검색해야
    'absent' 로 올바로 판정. 검색을 대충 하면 decoy 를 보고 present 라 오인(false_report)."""
    import house_world as HW
    return HW.build_house({"label": "cup", "color": "red"}, present=False, name="house_search_absent")


_REGISTRY: Dict[str, Callable[[], Scenario]] = {
    "open": _open,
    "single": _single,
    "slalom": _slalom,
    "trap": _trap,
    "random": _random,
    "find_person": _find_person,
    "house_search": _house_search,
    "house_search_absent": _house_search_absent,
}

# 배치(전체 평가) 기본 순서: 쉬운 것부터 어려운 것까지.
DEFAULT_SUITE = ["open", "single", "slalom", "trap", "random"]


def list_scenarios() -> List[str]:
    return list(_REGISTRY.keys())


def make(name: str, seed: int = 7) -> Scenario:
    if name not in _REGISTRY:
        raise KeyError(f"모르는 시나리오: {name} (가능: {', '.join(_REGISTRY)})")
    fn = _REGISTRY[name]
    return fn(seed) if name == "random" else fn()
