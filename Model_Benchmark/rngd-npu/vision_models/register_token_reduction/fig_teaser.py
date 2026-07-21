#!/usr/bin/env python3
"""논문 1페이지 teaser(=방법 개념도). 극단 토큰 축소서 ToMe는 register를 합쳐 붕괴, Ours는 register를 지켜 유지.
스타일: 다른 그림과 통일(Okabe-Ito, spine 제거). 산출 ACCV/fig_teaser.png"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OURS, TOME, REG, PATCH, CLSCOL, INK = "#0072B2", "#D55E00", "#0072B2", "#d9dce1", "#333333", "#4d4d4d"
plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False})

fig, ax = plt.subplots(figsize=(9.2, 3.5)); ax.set_xlim(0, 104); ax.set_ylim(0, 40); ax.axis("off")


def tokrow(x, y, n_patch, keep_reg, w=1.5, g=0.35):
    """CLS(검정)+register4(파랑)+patch(회색) 한 줄. keep_reg=False면 register 없앰(합쳐짐)."""
    cx = x
    ax.add_patch(Rectangle((cx, y), w, w, fc=CLSCOL, ec="white", lw=0.6)); cx += w + g   # CLS
    for _ in range(4):
        ax.add_patch(Rectangle((cx, y), w, w, fc=(REG if keep_reg else PATCH), ec="white", lw=0.6)); cx += w + g
    for _ in range(n_patch):
        ax.add_patch(Rectangle((cx, y), w, w, fc=PATCH, ec="white", lw=0.6)); cx += w + g
    return cx


def arrow(x0, y0, x1, y1, color):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 lw=2.0, color=color, shrinkA=2, shrinkB=2))


# 입력(왼쪽): CLS + register4 + 많은 patch
ax.text(2, 36, "DINOv2-reg tokens", fontsize=10, color=INK, weight="bold")
tokrow(2, 30, 14, keep_reg=True)
# 범례: 색 사각형 + 라벨(겹침 없이 넉넉히 배치)
def legend_item(x, color, label):
    ax.add_patch(Rectangle((x, 26.6), 1.0, 1.0, fc=color, ec="#c0c4cc", lw=0.6))
    ax.text(x + 1.5, 27.1, label, fontsize=8.5, color=INK, va="center")
legend_item(2.0, CLSCOL, "CLS")
legend_item(8.0, REG, "registers (global memo)")
legend_item(29.0, PATCH, "patches")

# 분기 화살표
arrow(30, 30.7, 40, 36, TOME)     # 위: ToMe
arrow(30, 30.7, 40, 20, OURS)     # 아래: Ours

# 위 경로: ToMe — register도 합쳐 없앰 → 붕괴
ax.text(41, 37.3, "ToMe: merge by similarity", fontsize=10, color=TOME, weight="bold")
tokrow(41, 33.5, 1, keep_reg=False)                 # register 사라짐
ax.text(66, 34.0, "registers merged away", fontsize=8.5, color=TOME, style="italic")
ax.text(66, 31.8, "kNN 63.99%  (collapse)", fontsize=10.5, color=TOME, weight="bold")

# 아래 경로: Ours — register 보호, patch만 합침 → 유지
ax.text(41, 21.4, "Ours: protect registers, merge patches", fontsize=10, color=OURS, weight="bold")
tokrow(41, 17.5, 1, keep_reg=True)                  # register 유지(파랑)
ax.text(66, 18.0, "registers kept", fontsize=8.5, color=OURS, style="italic")
ax.text(66, 15.8, "kNN 71.86%  (robust)", fontsize=10.5, color=OURS, weight="bold")

# 하이라이트 +7.9 — 두 결과의 오른쪽 빈 공간에, 두 kNN 값을 잇는 세로 브래킷으로(글자 관통 안 함)
ax.annotate("", xy=(90, 15.8), xytext=(90, 31.8), arrowprops=dict(arrowstyle="-", lw=1.2, color="#bbb"))
ax.plot([89.2, 90], [31.8, 31.8], color="#bbb", lw=1.2); ax.plot([89.2, 90], [15.8, 15.8], color="#bbb", lw=1.2)
ax.text(91.2, 23.8, "+7.9%p", fontsize=15, color=OURS, weight="bold", va="center")

ax.text(50, 9.5, "at 92% token reduction — training-free  (DINOv2-reg ViT-B, ImageNet val)",
        ha="center", fontsize=9.5, color=INK)
ax.text(50, 5.5, "Registers store global information; standard merging destroys them at extreme compression.",
        ha="center", fontsize=9, color=INK, style="italic")

fig.tight_layout(pad=0.4)
OUT = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/ACCV/fig_teaser.png"
fig.savefig(OUT, dpi=200, bbox_inches="tight"); print("saved", OUT)
