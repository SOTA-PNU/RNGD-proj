#!/usr/bin/env python3
"""압축률 sweep 그림 생성: ablation_*.json (eval_ablation.py 출력) 또는 eval_imagenet 표를
읽어 '토큰 압축률 vs kNN 정확도' 곡선(전략별)을 그린다. main.md Figure용.
사용: python make_sweep_figure.py results/ablation_xxx.json [out.png]"""
import sys, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 색맹-안전 팔레트(Okabe-Ito). Ours=블루(주역, 굵게)·ToMe=버밀리언. 나머지 프록시는
# 얇게(선굵기·zorder로 강조 차등; 색은 마커+범례로 구분). "energy"는 평균-유사도 프록시(공식 PiToMe 아님).
# (label, color, marker, linestyle, lw, zorder)
STRAT_STYLE = {
    "ours":     ("Ours",      "#0072B2", "o", "-",   2.6, 5),
    "tome":     ("ToMe",      "#D55E00", "s", "--",  1.9, 4),
    "random":   ("Random",    "#CC79A7", "^", ":",   1.5, 3),
    "energy":   ("Energy",    "#E69F00", "D", "-.",  1.5, 3),
    "highnorm": ("High-norm", "#56B4E9", "v", ":",   1.5, 3),
}


def clean_model_name(m):
    m = (m or "").split(".")[0]
    if "reg4" in m: return "DINOv2-reg"
    if "dinov2" in m: return "DINOv2"
    return m


def main():
    if len(sys.argv) < 2:
        print("usage: make_sweep_figure.py <ablation.json> [out.png]"); return
    res = json.load(open(sys.argv[1]))
    out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace(".json", ".png")
    rows = sorted(res["rows"], key=lambda r: r["comp"])
    comps = [r["comp"] for r in rows]
    strats = [s for s in STRAT_STYLE if any(s in r for r in rows)]

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.linewidth": 0.8,
                         "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    full = res.get("full_knn")
    if full is not None:
        ax.axhline(full, color="#4d4d4d", lw=1.0, ls=(0, (1, 1.6)), alpha=0.6)
        ax.text(comps[0], full + 0.12, f"Full model  {full:.1f}", ha="left", va="bottom",
                fontsize=8.5, color="#4d4d4d", alpha=.8)
    for s in strats:
        ys = [r.get(s) for r in rows]
        label, color, mk, ls, lw, z = STRAT_STYLE[s]
        ax.plot(comps, ys, marker=mk, color=color, ls=ls, lw=lw, ms=6.5 if s == "ours" else 5.5,
                mec="white", mew=0.7, label=label, zorder=z, alpha=1.0 if s in ("ours", "tome") else 0.9)
    ax.set_xlabel("Token reduction (%)"); ax.set_ylabel("ImageNet kNN top-1 (%)")
    ax.set_title(f"{clean_model_name(res.get('model'))}, ImageNet val 50k")
    ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False, fontsize=9.5, loc="lower left", ncol=2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, color="#c9ccd1")
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")
    # 극단압축 행 요약(논문 문장용)
    ext = max(rows, key=lambda r: r["comp"])
    if "ours" in ext and "tome" in ext:
        print(f"extreme {ext['comp']}%: ours={ext['ours']} tome={ext['tome']} "
              f"Δ={ext['ours']-ext['tome']:+.2f}"
              + (f" energy={ext['energy']} random={ext['random']}" if 'energy' in ext else ""))


if __name__ == "__main__":
    main()
