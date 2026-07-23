#!/usr/bin/env python3
"""압축률 sweep 그림 생성: ablation_*.json (eval_ablation.py 출력) 또는 eval_imagenet 표를
읽어 '토큰 압축률 vs kNN 정확도' 곡선(전략별)을 그린다. main.md Figure용.
사용: python make_sweep_figure.py results/ablation_xxx.json [out.png]"""
import sys, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRAT_STYLE = {
    "ours": ("Ours (protect registers)", "#d62728", "o", "-", 2.4),
    "tome": ("ToMe (protect CLS only)", "#1f77b4", "s", "--", 1.8),
    "random": ("Random protect", "#7f7f7f", "^", ":", 1.5),
    "energy": ("Energy keep-prior (PiToMe-style)", "#2ca02c", "D", "-.", 1.8),
    "highnorm": ("High-norm protect", "#9467bd", "v", ":", 1.5),
}


def main():
    if len(sys.argv) < 2:
        print("usage: make_sweep_figure.py <ablation.json> [out.png]"); return
    res = json.load(open(sys.argv[1]))
    out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace(".json", ".png")
    rows = sorted(res["rows"], key=lambda r: r["comp"])
    comps = [r["comp"] for r in rows]
    strats = [s for s in STRAT_STYLE if any(s in r for r in rows)]

    plt.figure(figsize=(6.2, 4.4))
    full = res.get("full_knn")
    if full is not None:
        plt.axhline(full, color="black", lw=1, ls=(0, (1, 1)), alpha=0.6,
                    label=f"Full model ({full:.1f})")
    for s in strats:
        ys = [r.get(s) for r in rows]
        label, color, mk, ls, lw = STRAT_STYLE[s]
        plt.plot(comps, ys, marker=mk, color=color, ls=ls, lw=lw, ms=6, label=label)
    plt.xlabel("Token reduction (%)"); plt.ylabel("ImageNet kNN top-1 (%)")
    plt.title(f"{res.get('model','').split('.')[0]}  (N={res.get('N','?')})")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=8, loc="lower left")
    plt.tight_layout(); plt.savefig(out, dpi=180)
    print(f"saved {out}")
    # 극단압축 행 요약(논문 문장용)
    ext = max(rows, key=lambda r: r["comp"])
    if "ours" in ext and "tome" in ext:
        print(f"extreme {ext['comp']}%: ours={ext['ours']} tome={ext['tome']} "
              f"Δ={ext['ours']-ext['tome']:+.2f}"
              + (f" energy={ext['energy']} random={ext['random']}" if 'energy' in ext else ""))


if __name__ == "__main__":
    main()
