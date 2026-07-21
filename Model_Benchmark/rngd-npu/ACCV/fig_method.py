#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig_method: 레지스터 인지 토큰 축소 method 다이어그램(직관 버전).
규칙: 그림 안에는 '라벨용 단어'만. 설명 문장은 캡션/본문에 둔다.
구성 = (좌) 한 블록의 동작 Attn->Merge->MLP + 병합 메커니즘(보호 우회 + 패치 병합),
       (우) 깊이(블록)에 따른 시퀀스 변화 대비: ToMe는 레지스터(파랑)를 잃고 Ours는 금색 링으로 끝까지 유지.
핵심 직관 = 파랑(레지스터)이 위(ToMe)에선 사라지고 아래(Ours)에선 남는 색 대비.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 색 (Okabe-Ito, 논문 identity: Ours=파랑 #0072B2)
C_CLS="#2b3a55"; C_REG="#0072B2"; C_PATCH="#dcdcda"; C_GONE="#eef0f2"
C_PROT="#E69F00"; INK="#222222"; MUT="#8a8a8a"; ARROW="#5b6b7a"
plt.rcParams.update({"font.size":12,"font.family":"DejaVu Sans"})

fig, ax = plt.subplots(figsize=(12.4, 6.0))
ax.set_xlim(0, 130); ax.set_ylim(0, 60); ax.axis("off")

def tok(x, y, w=2.3, h=2.7, fc=C_PATCH, ec="none", label="", tc="white",
        fs=7.5, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.6",
                 fc=fc, ec=ec, lw=1.0, zorder=3, alpha=alpha))
    if label:
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", color=tc,
                fontsize=fs, fontweight="bold", zorder=5)

def group_ring(x0, y, n, w=2.3, gap=0.55, h=2.7):
    """앞 n개 토큰(보호 집합)을 하나의 금색 링으로 감싼다."""
    span = n*(w+gap) - gap
    ax.add_patch(FancyBboxPatch((x0-0.7, y-0.7), span+1.4, h+1.4,
                 boxstyle="round,pad=0.05,rounding_size=1.0",
                 fc="none", ec=C_PROT, lw=2.1, zorder=4))

def arrow(x1, y1, x2, y2, lw=1.6, color=ARROW, style="-|>", mut=13, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=mut, lw=lw, color=color, zorder=2, linestyle=ls))

# spec: list of 'cls'|'reg'|'patch'  -> 한 줄 토큰 시퀀스. prot_n=앞에서 보호(링)할 개수.
def row(x0, y, specs, w=2.3, gap=0.55, prot_n=0):
    x = x0
    for kind in specs:
        if kind == "cls":   tok(x, y, w=w, fc=C_CLS, label="CLS", fs=6.2)
        elif kind == "reg": tok(x, y, w=w, fc=C_REG, label="R", fs=8)
        else:               tok(x, y, w=w, fc=C_PATCH)
        x += w + gap
    if prot_n > 0:
        group_ring(x0, y, prot_n, w=w, gap=gap)
    return x  # 끝 x

# ================= 상단 범례 =================
ly = 55.5
def legend_item(x, kind, text):
    if kind == "cls":   tok(x, ly, w=2.3, fc=C_CLS, label="CLS", fs=6.2)
    elif kind == "reg": tok(x, ly, w=2.3, fc=C_REG, label="R", fs=8)
    else:               tok(x, ly, w=2.3, fc=C_PATCH)
    ax.text(x+3.1, ly+1.35, text, ha="left", va="center", fontsize=10.5, color=INK)
legend_item(38, "cls", "class token")
legend_item(60, "reg", "register")
legend_item(82, "patch", "patch")
tok(101, ly, w=2.3, fc=C_REG, label="R", fs=8); group_ring(101, ly, 1)
ax.text(104.6, ly+1.35, "protected", ha="left", va="center", fontsize=10.5, color=C_PROT)

# ================= 좌측: 한 블록 =================
ax.text(2, 50.5, "One block", fontsize=13.5, fontweight="bold", color=INK)
by = 44
for i, name in enumerate(["Attention", "Merge", "MLP"]):
    bx = 2 + i*11.5
    fc = "#e7eff7" if name == "Merge" else "#f3f3f3"
    ec = C_REG if name == "Merge" else "#bcbcbc"
    ax.add_patch(FancyBboxPatch((bx, by), 10, 4.6, boxstyle="round,pad=0.15,rounding_size=1.0",
                 fc=fc, ec=ec, lw=1.3, zorder=2))
    ax.text(bx+5, by+2.3, name, ha="center", va="center", fontsize=10.5,
            color=INK, fontweight="bold")
    if i < 2: arrow(bx+10, by+2.3, bx+13, by+2.3)
ax.text(36.5, by+2.3, r"$\times L$", fontsize=13, color=INK, fontweight="bold", va="center")
# Merge -> 메커니즘 콜아웃 점선
ax.plot([13, 13], [by, 37.6], ls=(0,(1,2)), color="#9db4c9", lw=1.2, zorder=1)

# 병합 메커니즘 미니 인셋
ax.text(2, 39, "protect: pass through", fontsize=10.5, fontweight="bold", color=C_PROT)
row(3, 33.5, ["cls", "reg", "reg"], prot_n=3)
arrow(12.6, 34.8, 16.6, 34.8)
row(18.0, 33.5, ["cls", "reg", "reg"], prot_n=3)

ax.text(2, 30, r"merge $r$ similar patches", fontsize=10.5, fontweight="bold", color=INK)
# 두 유사 패치 -> 하나(큰) 패치
tok(4, 24.3, w=2.3, fc=C_PATCH); tok(8.4, 24.3, w=2.3, fc=C_PATCH)
ax.text(7.25, 25.65, r"$\approx$", fontsize=12, color=MUT, ha="center", va="center")
arrow(11.5, 25.65, 16.5, 25.65)
tok(17.3, 23.8, w=3.7, h=3.7, fc=C_PATCH, ec=MUT)  # 병합된(큰) 패치 = size 커짐
ax.text(19.15, 22.0, r"size $\uparrow$", fontsize=8.5, color=MUT, ha="center")

# 좌/우 구분선
ax.plot([44, 44], [3, 52], ls=(0,(2,3)), color="#d2d2d2", lw=1.1)

# ================= 우측: 깊이에 따른 시퀀스 =================
ax.text(47, 50.5, "Across blocks", fontsize=13.5, fontweight="bold", color=INK)

X1, X2, X3 = 54, 84, 110      # 세 스냅샷 시작 x
def lane(y, name, color, snaps):
    ax.text(46.5, y+1.3, name, fontsize=12, fontweight="bold", color=color, ha="left")
    ends = []
    for xs, (spec, pn) in zip([X1, X2, X3], snaps):
        e = row(xs, y, spec, prot_n=pn)
        ends.append(e)
    # 스냅샷 사이 화살표(=블록 진행)
    arrow(ends[0]+1.5, y+1.35, X2-1.5, y+1.35)
    arrow(ends[1]+1.5, y+1.35, X3-1.5, y+1.35)
    return ends

# ToMe: CLS만 보호 -> 레지스터 4 -> 2 -> 0 (파랑이 사라짐), 시퀀스도 짧아짐
tome = [(["cls"]+["reg"]*4+["patch"]*3, 1),
        (["cls"]+["reg"]*2+["patch"]*2, 1),
        (["cls"]+["patch"]*3,           1)]
# Ours: CLS+레지스터 보호(하나의 금색 링) -> 레지스터 유지, 패치만 줄어듦
ours = [(["cls"]+["reg"]*4+["patch"]*3, 5),
        (["cls"]+["reg"]*4+["patch"]*2, 5),
        (["cls"]+["reg"]*4+["patch"]*1, 5)]

lane(37, "ToMe", "#444444", tome)
lane(16, "Ours", C_REG,     ours)

# 첫 스냅샷 위 토큰 종류 라벨(직관용, 라벨만)
def brace_label(x0, x1, y, text, color=INK):
    ax.plot([x0, x0, x1, x1], [y, y+0.9, y+0.9, y], color=color, lw=1.0)
    ax.text((x0+x1)/2, y+1.6, text, ha="center", va="bottom", fontsize=9, color=color)
brace_label(X1, X1+2.3, 40.5, "CLS", C_CLS)
brace_label(X1+2.85, X1+2.85+4*2.85-0.55, 40.5, "registers", C_REG)
brace_label(X1+2.85+4*2.85, X1+2.85+7*2.85-0.55, 40.5, "patches", MUT)

# 깊이 라벨
for xx, lab in [(X1+3, "block 1"), (X2+2, "block $L/2$"), (X3+2, "block $L$")]:
    ax.text(xx, 11.5, lab, fontsize=9.5, color="#666", ha="center")

plt.tight_layout(pad=0.4)
plt.savefig("fig_method.pdf", bbox_inches="tight")
plt.savefig("fig_method.png", dpi=150, bbox_inches="tight")
print("saved fig_method.pdf/png")
