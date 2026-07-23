#!/usr/bin/env python3
"""sweep 결과(results/sweep_*.json) 집계 → 표 출력 + 차트 생성.
차트: (a) 모델 비교(정확도 vs throughput, 점크기=params), (b) vit_base 배치 스케일링, (c) 택틱 default vs optimized.
실행: python tools/make_sweep_report.py"""
import json, glob, os, warnings
warnings.filterwarnings("ignore")

VM = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models"
RES = f"{VM}/results"
FIG = f"{VM}/results/figures"
os.makedirs(FIG, exist_ok=True)

# 알려진 파라미터 수(M) — timm 로드 없이(컴파일/다운로드 회피). 필요시 동적계산 fallback.
PARAMS_M = {
    "vit_tiny_patch16_224": 5.7, "vit_small_patch16_224": 22.0, "vit_base_patch16_224": 86.6,
    "deit_tiny_patch16_224": 5.7, "deit_small_patch16_224": 22.1, "deit_base_patch16_224": 86.6,
}
def pm(name):
    for k, v in PARAMS_M.items():
        if k in name: return v
    return None
def short(name): return name.split(".")[0].replace("_patch16_224", "")


def load_all():
    rows = []
    for f in sorted(glob.glob(f"{RES}/sweep_*.json")):
        try:
            for r in json.load(open(f)):
                rows.append(r)
        except Exception as e:
            print("skip", f, e)
    return rows


def main():
    rows = load_all()
    if not rows:
        print("결과 없음 (results/sweep_*.json)"); return
    # 표
    print(f"\n{'model':26s} {'batch':>5s} {'opt':>4s} {'params(M)':>9s} {'NPU top1':>8s} {'CPU ref':>8s} "
          f"{'NPU ms/b':>9s} {'img/s':>8s} {'vs CPU':>7s} {'compile':>8s} {'uniq':>5s}")
    print("-" * 120)
    for r in sorted(rows, key=lambda x: (x.get("model",""), x.get("batch",0), x.get("optimize",False))):
        if r.get("compile") != "OK":
            print(f"{short(r.get('model','?')):26s} {r.get('batch','?'):>5} {'opt' if r.get('optimize') else '-':>4s}  COMPILE FAIL: {r.get('err','')[:60]}")
            continue
        print(f"{short(r['model']):26s} {r['batch']:>5} {'Y' if r.get('optimize') else '-':>4s} "
              f"{str(pm(r['model']) or '?'):>9s} {str(r.get('npu_top1_acc','-')):>8s} {str(r.get('cpu_top1_acc_ref','-')):>8s} "
              f"{str(r.get('npu_ms_per_batch','-')):>9s} {str(r.get('npu_img_per_s','-')):>8s} "
              f"{str(r.get('speedup_vs_cpu','-')):>7s} {str(r.get('compile_s','-')):>8s} {str(r.get('npu_unique_preds','-')):>5s}")
    json.dump(rows, open(f"{RES}/sweep_combined.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n[combined] {RES}/sweep_combined.json  (n={len(rows)})")

    # ---- 차트 ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib 없음 — 차트 생략"); return
    NAVY, RED, GREEN, AMBER = "#1F3864", "#C00000", "#2E7D32", "#E9A23B"

    # (a) 모델 비교: 정확도 vs throughput (batch1, optimize=False)
    m1 = [r for r in rows if r.get("compile") == "OK" and r.get("batch") == 1 and not r.get("optimize")
          and r.get("npu_top1_acc") is not None]
    if m1:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for r in m1:
            x, y = r["npu_img_per_s"], r["npu_top1_acc"]
            sz = (pm(r["model"]) or 20) * 8
            col = NAVY if "vit" in r["model"] else AMBER
            ax.scatter(x, y, s=sz, color=col, alpha=0.7, edgecolors="k", linewidths=0.5)
            ax.annotate(short(r["model"]), (x, y), fontsize=9, xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("NPU throughput (images/s, batch=1)", fontsize=11)
        ax.set_ylabel("NPU ImageNet top-1 (%)", fontsize=11)
        ax.set_title("Accuracy vs Speed on RNGD NPU (point size = params)\nblue=ViT, amber=DeiT", fontsize=12, color=NAVY, weight="bold")
        ax.grid(True, alpha=0.3); fig.tight_layout()
        fig.savefig(f"{FIG}/sweep_acc_vs_speed.png", dpi=150, bbox_inches="tight")
        print(f"[chart] {FIG}/sweep_acc_vs_speed.png")

    # (b) vit_base 배치 스케일링 (throughput vs batch)
    vb = sorted([r for r in rows if r.get("compile") == "OK" and "vit_base" in r.get("model", "")
                 and not r.get("optimize")], key=lambda x: x["batch"])
    if len(vb) >= 2:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        bs = [r["batch"] for r in vb]; tp = [r["npu_img_per_s"] for r in vb]; ms = [r["npu_ms_per_batch"] for r in vb]
        ax.plot(bs, tp, "o-", color=NAVY, lw=2.4, ms=9, label="throughput (img/s)")
        ax.set_xlabel("batch size", fontsize=11); ax.set_ylabel("throughput (images/s)", fontsize=11, color=NAVY)
        ax.set_xscale("log", base=2); ax.set_xticks(bs); ax.set_xticklabels(bs)
        ax2 = ax.twinx()
        ax2.plot(bs, ms, "s--", color=RED, lw=2, ms=8, label="latency (ms/batch)")
        ax2.set_ylabel("latency (ms/batch)", fontsize=11, color=RED)
        ax.set_title("vit_base/16 batch scaling on RNGD NPU", fontsize=12, color=NAVY, weight="bold")
        ax.grid(True, alpha=0.3); fig.tight_layout()
        fig.savefig(f"{FIG}/sweep_batch_scaling.png", dpi=150, bbox_inches="tight")
        print(f"[chart] {FIG}/sweep_batch_scaling.png")

    # (c) 택틱 default vs optimized (vit_base@1)
    d = next((r for r in rows if r.get("compile") == "OK" and "vit_base" in r.get("model","") and r.get("batch") == 1 and not r.get("optimize")), None)
    o = next((r for r in rows if r.get("compile") == "OK" and "vit_base" in r.get("model","") and r.get("batch") == 1 and r.get("optimize")), None)
    if d and o:
        fig, ax = plt.subplots(figsize=(6, 5))
        labels = ["default", "ForVisionModel\n+ pruning"]
        ms = [d["npu_ms_per_batch"], o["npu_ms_per_batch"]]
        bars = ax.bar(labels, ms, color=[NAVY, GREEN], width=0.55)
        for b, v in zip(bars, ms): ax.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}ms", ha="center", va="bottom", fontsize=11)
        imp = (1 - o["npu_ms_per_batch"]/d["npu_ms_per_batch"]) * 100
        ax.set_ylabel("NPU latency (ms/image, batch=1)", fontsize=11)
        ax.set_title(f"Tactic optimization on vit_base/16\nlatency {imp:+.1f}%  (acc {d.get('npu_top1_acc')}→{o.get('npu_top1_acc')})", fontsize=12, color=NAVY, weight="bold")
        ax.grid(True, axis="y", alpha=0.3); fig.tight_layout()
        fig.savefig(f"{FIG}/sweep_tactic_opt.png", dpi=150, bbox_inches="tight")
        print(f"[chart] {FIG}/sweep_tactic_opt.png")


if __name__ == "__main__":
    main()
