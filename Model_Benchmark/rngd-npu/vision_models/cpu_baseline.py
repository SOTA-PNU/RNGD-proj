#!/usr/bin/env python3
"""FP32(CPU) 기준 정확도를 NPU sweep과 '같은 ImageNet val 10000장'에서 측정.
NPU top-1과 apples-to-apples 비교용(저정밀 손실 정량화). CPU 전용 → NPU sweep과 병렬 가능.
사용: python cpu_baseline.py [--n_val 10000] [--batch 64]"""
import argparse, json, os, csv, time, warnings
warnings.filterwarnings("ignore")
import torch, timm
from PIL import Image

VM = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models"
VAL_DIR = f"{VM}/imagenet_val"
OUT = f"{VM}/results/cpu_baseline.json"
MODELS = [
    "vit_tiny_patch16_224.augreg_in21k_ft_in1k", "vit_small_patch16_224.augreg_in1k",
    "vit_base_patch16_224.augreg_in1k", "deit_tiny_patch16_224.fb_in1k",
    "deit_small_patch16_224.fb_in1k", "deit_base_patch16_224.fb_in1k",
]


def load_val(model, n):
    cfg = timm.data.resolve_model_data_config(model)
    tf = timm.data.create_transform(**cfg, is_training=False)
    rows = list(csv.DictReader(open(f"{VAL_DIR}/labels.csv")))[:n]
    xs, ys = [], []
    for r in rows:
        p = f"{VAL_DIR}/images/{r['filename']}"
        if os.path.exists(p):
            xs.append(tf(Image.open(p).convert("RGB"))); ys.append(int(r["label_idx"]))
    return torch.stack(xs), torch.tensor(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_val", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    res = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {r["model"] for r in res}
    for mn in MODELS:
        if mn in done:
            print(f"skip {mn} (done)", flush=True); continue
        m = timm.create_model(mn, pretrained=True).eval()
        for p in m.parameters(): p.requires_grad_(False)
        X, Y = load_val(m, args.n_val)
        correct = 0; t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(X), args.batch):
                out = m(X[i:i + args.batch])
                correct += (out.argmax(-1) == Y[i:i + args.batch]).sum().item()
        acc = round(100 * correct / len(X), 2)
        rec = {"model": mn, "fp32_cpu_top1": acc, "n_eval": len(X), "eval_s": round(time.time() - t0, 1)}
        print(f"{mn:42s} fp32 top1={acc}%  ({rec['eval_s']}s)", flush=True)
        res.append(rec); json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
