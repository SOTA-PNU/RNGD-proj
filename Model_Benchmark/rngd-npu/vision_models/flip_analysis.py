#!/usr/bin/env python3
"""실리콘 유발 예측 드리프트 검증: 같은 이미지에서 NPU(저정밀) vs FP32(CPU) 개별 예측이 뒤집히는가?
집계 정확도가 같아도 per-sample flip이 있으면 robustness/재현성 연구 주제가 성립.
측정: 일치율, flip 수, flip 중 (NPU맞고FP32틀림 / FP32맞고NPU틀림), flip 샘플의 top1-top2 margin 분포.
사용: python flip_analysis.py --model vit_base_patch16_224.augreg_in1k --npu 8 --n 10000"""
import argparse, json, os, csv, warnings
warnings.filterwarnings("ignore")
import torch, furiosa.torch
from furiosa.torch import CompileModule
from torch._decomp import core_aten_decompositions, get_decompositions
import timm
from PIL import Image

VM = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models"
VAL = f"{VM}/imagenet_val"
D = dict(core_aten_decompositions())
D.update(get_decompositions([torch.ops.aten._native_batch_norm_legit_no_training,
    torch.ops.aten._native_batch_norm_legit, torch.ops.aten.batch_norm, torch.ops.aten.native_batch_norm]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch16_224.augreg_in1k")
    ap.add_argument("--npu", type=int, default=8)
    ap.add_argument("--n", type=int, default=10000)
    args = ap.parse_args()
    m = timm.create_model(args.model, pretrained=True).eval()
    for p in m.parameters(): p.requires_grad_(False)
    cfg = timm.data.resolve_model_data_config(m); tf = timm.data.create_transform(**cfg, is_training=False)
    rows = list(csv.DictReader(open(f"{VAL}/labels.csv")))[:args.n]
    X, Y = [], []
    for r in rows:
        p = f"{VAL}/images/{r['filename']}"
        if os.path.exists(p): X.append(tf(Image.open(p).convert("RGB"))); Y.append(int(r["label_idx"]))
    X = torch.stack(X).contiguous(); Y = torch.tensor(Y)
    dev = torch.device("rngd", args.npu)
    with torch.no_grad():
        ep = torch.export.export(m, (torch.randn(1, 3, 224, 224),)).run_decompositions(D)
        cm = CompileModule.from_exported(ep); cm.to(dev)
    print(f"[compiled] {args.model} on rngd:{args.npu}, {len(X)} imgs", flush=True)

    npu_pred, cpu_pred, cpu_margin = [], [], []
    with torch.no_grad():
        for i in range(len(X)):
            x = X[i:i+1]
            co = m(x).float()[0]
            no = cm(x.to(dev), device=dev).to("cpu").float()[0]
            top2 = torch.topk(co, 2).values
            cpu_pred.append(int(co.argmax())); npu_pred.append(int(no.argmax()))
            cpu_margin.append(float(top2[0] - top2[1]))
    Y = Y.tolist()
    n = len(Y)
    agree = sum(1 for a, b in zip(npu_pred, cpu_pred) if a == b)
    flips = [i for i in range(n) if npu_pred[i] != cpu_pred[i]]
    npu_acc = 100*sum(1 for i in range(n) if npu_pred[i]==Y[i])/n
    cpu_acc = 100*sum(1 for i in range(n) if cpu_pred[i]==Y[i])/n
    npu_right_cpu_wrong = sum(1 for i in flips if npu_pred[i]==Y[i] and cpu_pred[i]!=Y[i])
    cpu_right_npu_wrong = sum(1 for i in flips if cpu_pred[i]==Y[i] and npu_pred[i]!=Y[i])
    import statistics as st
    flip_margins = [cpu_margin[i] for i in flips]
    noflip_margins = [cpu_margin[i] for i in range(n) if i not in set(flips)]
    res = {
        "model": args.model, "n": n,
        "npu_top1": round(npu_acc,2), "cpu_fp32_top1": round(cpu_acc,2),
        "agreement_pct": round(100*agree/n,3), "flip_count": len(flips), "flip_pct": round(100*len(flips)/n,3),
        "flip_npu_right_cpu_wrong": npu_right_cpu_wrong, "flip_cpu_right_npu_wrong": cpu_right_npu_wrong,
        "median_margin_flipped": round(st.median(flip_margins),4) if flip_margins else None,
        "median_margin_not_flipped": round(st.median(noflip_margins),4) if noflip_margins else None,
    }
    print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)
    json.dump(res, open(f"{VM}/results/flip_analysis_{args.model.split('.')[0]}.json","w"), indent=1, ensure_ascii=False)
    print("\n해석: flip_pct>0 이면 '집계정확도 동일해도 개별예측 드리프트 존재' → 드리프트 주제 성립.", flush=True)
    print("flip 샘플의 margin이 non-flip보다 작으면 '경계근처 샘플이 hw정밀에 민감' = 예측가능/수정가능.", flush=True)


if __name__ == "__main__":
    main()
