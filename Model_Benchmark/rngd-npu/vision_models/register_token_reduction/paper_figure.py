#!/usr/bin/env python3
"""논문 메인 그림 fig_result.png: 2-패널.
좌: token reduction% vs kNN top-1 (DINOv2-reg: Ours vs ToMe, full 기준선).
우: Δ(Ours-ToMe) vs reduction% (DINOv2-reg 상승 vs register-없는 DINOv2 ≈0 = 인과 대조군).
DINOv2-reg 곡선은 tab:main과 일치시키기 위해 pitome_compare 런(Run 2, val LOO)에서 읽는다.
대조군(register-없는 별도 모델)은 results/dinov2_noreg_control.txt 에서 읽는다.
사용: python paper_figure.py [out.png]"""
import sys, re, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
PIT = os.path.join(HERE, "pitome_compare", "results_acc.txt")
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/ACCV/fig_result.png"


def parse_pitome(path):
    """Run 2(pitome_compare): 'r comp tome pitome ours delta' → reg 곡선 (tome=col3, ours=col5)."""
    full = None; rows = []
    for ln in open(path):
        mb = re.match(r"\s*0\s+0\.0\s+([\d.]+)\s*\(", ln)      # r=0 baseline 줄
        if mb: full = float(mb.group(1))
        m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[+-]", ln)
        if m and int(m.group(1)) > 0:
            rows.append((float(m.group(2)), float(m.group(3)), float(m.group(5))))  # comp, tome, ours
    return full, rows


def parse_control(path):
    """Run 1 형식: 'r comp% tome ours (Δ=...)' → 대조군."""
    full = None; rows = []
    for ln in open(path):
        mf = re.search(r"full=([\d.]+)", ln)
        if mf: full = float(mf.group(1))
        m = re.match(r"\s*\d+\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)", ln)
        if m:
            rows.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
    return full, rows


def main():
    full_r, reg = parse_pitome(PIT)
    full_n, noreg = parse_control(f"{RES}/dinov2_noreg_control.txt")
    cr = [x[0] for x in reg]; t = [x[1] for x in reg]; o = [x[2] for x in reg]; dr = [x[2]-x[1] for x in reg]
    cn = [x[0] for x in noreg]; dn = [x[2]-x[1] for x in noreg]

    # 색맹-안전 팔레트(Okabe-Ito): Ours=블루(주역), ToMe=버밀리언, 대조군=중립 회색
    OURS, TOME, CTRL, INK = "#0072B2", "#D55E00", "#9AA0A6", "#4d4d4d"
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.linewidth": 0.8})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    # 좌: 정확도 vs 축소율
    if full_r:
        ax1.axhline(full_r, color=INK, ls=(0, (1, 1.6)), lw=1.0, alpha=.6)
        ax1.text(cr[-1], full_r + 0.12, f"Full model  {full_r:.1f}", ha="right", va="bottom",
                 fontsize=8.5, color=INK, alpha=.8)
    ax1.plot(cr, o, "-o", color=OURS, lw=2.0, ms=6, mec="white", mew=0.8, label="Ours", zorder=3)
    ax1.plot(cr, t, "--s", color=TOME, lw=2.0, ms=6, mec="white", mew=0.8, label="ToMe", zorder=2)
    ax1.set_xlabel("Token reduction (%)"); ax1.set_ylabel("ImageNet kNN top-1 (%)")
    ax1.set_title("DINOv2-reg, ImageNet val 50k")
    ax1.grid(axis="y", alpha=.25); ax1.legend(frameon=False, fontsize=10, loc="lower left")

    # 우: Δ(Ours−ToMe) vs 축소율 — register 유무 대조 (점별 라벨 없음: 축이 Δ를 보여줌)
    ax2.axhline(0, color="#c9ccd1", lw=1.0, zorder=1)
    ax2.plot(cr, dr, "-o", color=OURS, lw=2.0, ms=6, mec="white", mew=0.8, label="with registers", zorder=3)
    ax2.plot(cn, dn, "--^", color=CTRL, lw=1.8, ms=6, mec="white", mew=0.8, label="without registers", zorder=2)
    ax2.set_xlabel("Token reduction (%)"); ax2.set_ylabel(r"$\Delta$ kNN top-1  (Ours $-$ ToMe)")
    ax2.set_title("Register-specific gap")
    ax2.grid(axis="y", alpha=.25); ax2.legend(frameon=False, fontsize=10, loc="upper left")
    ax2.set_ylim(-1, max(dr) + 1.2)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(length=3, color="#c9ccd1")
    fig.tight_layout(w_pad=2.2); plt.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"saved {OUT}")
    print(f"reg full={full_r} extreme Δ={dr[-1]:+.2f} @ {cr[-1]}%  | noreg extreme Δ={dn[-1]:+.2f} @ {cn[-1]}%")


if __name__ == "__main__":
    main()
