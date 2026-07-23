"""주행 경로를 콘솔에 ASCII 로 그립니다(의존성 없이 항상 동작).
matplotlib 이 있으면 PNG 도 저장합니다(선택). 기호: # 장애물 · S 시작 · G 목표 · R 도착 · · 경로."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from world import World


def render_ascii(world: World, start, goal, path: Optional[List[Tuple[float, float]]] = None,
                 cols: int = 62) -> str:
    rows = max(12, int(cols * world.height / world.width * 0.5))  # 문자 가로세로비 보정(*0.5)
    grid = [[" "] * cols for _ in range(rows)]

    def to_cell(x, y):
        cx = min(cols - 1, max(0, int(x / world.width * cols)))
        cy = min(rows - 1, max(0, int((world.height - y) / world.height * rows)))  # y 위로
        return cy, cx

    # 장애물: 셀 중심이 어떤 원 안이면 '#'
    for r in range(rows):
        wy = (1.0 - (r + 0.5) / rows) * world.height
        for c in range(cols):
            wx = (c + 0.5) / cols * world.width
            for ob in world.obstacles:
                if math.hypot(wx - ob.cx, wy - ob.cy) <= ob.r:
                    grid[r][c] = "#"
                    break

    # 경로
    if path:
        for (x, y) in path:
            cy, cx = to_cell(x, y)
            if grid[cy][cx] == " ":
                grid[cy][cx] = "·"

    sy, sx = to_cell(*start)
    gy, gx = to_cell(*goal)
    grid[sy][sx] = "S"
    grid[gy][gx] = "G"
    if path:
        ry, rx = to_cell(*path[-1])
        if grid[ry][rx] not in ("S", "G"):
            grid[ry][rx] = "R"

    top = "+" + "-" * cols + "+"
    body = "\n".join("|" + "".join(row) + "|" for row in grid)
    return f"{top}\n{body}\n{top}"


def save_png(world: World, start, goal, path, out_path: str) -> Optional[str]:
    """matplotlib 이 있으면 경로 PNG 저장, 없으면 None(조용히 건너뜀)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(5, 5))
    for ob in world.obstacles:
        ax.add_patch(plt.Circle((ob.cx, ob.cy), ob.r, color="#444"))
    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, "-", color="#dc2626", lw=1.5)
    ax.plot([start[0]], [start[1]], "o", color="#76d6ff", ms=8)
    ax.plot([goal[0]], [goal[1]], "*", color="#cdbbff", ms=14)
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal")
    ax.set_title("robot path")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
