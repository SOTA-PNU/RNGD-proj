#!/usr/bin/env python3
"""ACCV 실험 sweep: NPU 이미지분류 정확도 + 속도 측정 (timm).
축1: 같은 모델(vit_base) 배치 sweep.  축2: 같은 배치, 모델 sweep(vit/deit tiny/small/base).
각 (model,batch): from_exported 정식 컴파일 -> EDF 저장 -> NPU 지연/throughput + CPU 지연 + top-1.
ImageNet val 폴더 있으면 실정확도, 없으면 6장 sanity로 붕괴 프록시.

사용: python vision_sweep.py --models vit_base_patch16_224.augreg_in1k --batches 1 2 4 8 --npu 9
      python vision_sweep.py --models <6모델...> --batches 1 --npu 9
출력: tmp/sweep_<tag>.json (append) + stdout 표
"""
import argparse, time, json, os, glob, warnings
warnings.filterwarnings("ignore")
import torch
import furiosa.torch
from furiosa.torch import CompileModule
from torch._decomp import core_aten_decompositions, get_decompositions
import timm
from PIL import Image
import csv

IMG_DIR = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/test_images"
VAL_DIR = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/imagenet_val"
SANITY = ["brambling.jpg", "tabby_cat.jpg", "convertible.jpg", "orange.jpg", "dog1.jpg", "astronaut.jpg"]
EDF_DIR = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/edf"
OUT = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/results/sweep_results.json"
os.makedirs(EDF_DIR, exist_ok=True)

DECOMP = dict(core_aten_decompositions())
DECOMP.update(get_decompositions([
    torch.ops.aten._native_batch_norm_legit_no_training, torch.ops.aten._native_batch_norm_legit,
    torch.ops.aten.batch_norm, torch.ops.aten.native_batch_norm]))


def load_eval(model, n_val):
    """ImageNet val 있으면 (imgs, labels) 실데이터, 없으면 sanity 6장(label=-1)."""
    cfg = timm.data.resolve_model_data_config(model)
    tf = timm.data.create_transform(**cfg, is_training=False)
    if os.path.isdir(VAL_DIR) and os.path.exists(f"{VAL_DIR}/labels.csv"):
        rows = list(csv.DictReader(open(f"{VAL_DIR}/labels.csv")))[:n_val]
        xs, ys = [], []
        for r in rows:
            p = f"{VAL_DIR}/images/{r['filename']}"
            if os.path.exists(p):
                xs.append(tf(Image.open(p).convert("RGB"))); ys.append(int(r["label_idx"]))
        return torch.stack(xs), torch.tensor(ys), "imagenet_val"
    xs = [tf(Image.open(f"{IMG_DIR}/{f}").convert("RGB")) for f in SANITY]
    return torch.stack(xs), torch.tensor([-1] * len(xs)), "sanity6"


def run_one(model_name, batch, npu, n_val, iters, optimize=False):
    res = {"model": model_name, "batch": batch, "optimize": optimize}
    m = timm.create_model(model_name, pretrained=True).eval()
    for p in m.parameters(): p.requires_grad_(False)
    X, Y, src = load_eval(m, n_val); res["eval_src"] = src
    X = X.contiguous()   # 컴파일러/런타임은 연속 NCHW만 (channels_last 거부)
    dev = torch.device("rngd", npu)
    edf_path = f"{EDF_DIR}/{model_name.replace('.','_')}_b{batch}{'_opt' if optimize else ''}.edf"

    ex = (X[:batch] if X.shape[0] >= batch else X[(torch.arange(batch) % X.shape[0])]).contiguous()
    t = time.time()
    try:
        with torch.no_grad():
            ep = torch.export.export(m, (torch.randn(batch, *X.shape[1:]),)).run_decompositions(DECOMP)
            if optimize:   # 택틱 최적화: 비전 전용 힌트 + 택틱 가지치기
                from furiosa.native_torch import compiler
                cfg = compiler.Config(tactic_hint=compiler.TacticHintConfig.ForVisionModel, enable_tactic_pruning=True)
                cm = CompileModule.from_exported(ep, compiler_config=cfg)
            else:
                cm = CompileModule.from_exported(ep)
        res["compile_s"] = round(time.time() - t, 1); res["compile"] = "OK"
        try: open(edf_path, "wb").write(cm.edf.serialize())
        except Exception: pass
    except Exception as e:
        c = e
        while c.__cause__ is not None: c = c.__cause__
        res["compile"] = "FAIL"; res["err"] = str(c).splitlines()[0][:120]
        return res
    cm.to(dev)

    # ---- 속도: NPU 지연/throughput ----
    with torch.no_grad():
        cm(ex.to(dev), device=dev)  # warmup
        xd = ex.to(dev)
        t0 = time.time()
        for _ in range(iters):
            o = cm(xd, device=dev)
        torch.cuda.synchronize() if False else None
        npu_ms = (time.time() - t0) / iters * 1000
        # CPU 참고
        t0 = time.time(); m(ex); cpu_ms = (time.time() - t0) * 1000
    res["npu_ms_per_batch"] = round(npu_ms, 2)
    res["npu_img_per_s"] = round(batch / (npu_ms / 1000), 1)
    res["cpu_ms_per_batch"] = round(cpu_ms, 1)
    res["speedup_vs_cpu"] = round(cpu_ms / npu_ms, 2)

    # ---- 정확도: 전체 eval set ----
    cats = None
    try:
        import json as _j
        from torchvision.models import ViT_B_16_Weights
        cats = ViT_B_16_Weights.IMAGENET1K_V1.meta["categories"]
    except Exception: pass
    # NPU는 전체 eval, CPU는 처음 cpu_cap장만(참고 정확도). CPU per-image가 대용량서 병목이라.
    cpu_cap = 256
    npu_correct = cpu_correct = cpu_total = total = ws = 0
    npu_preds = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = X[i:i+batch]
            if xb.shape[0] < batch:  # pad last
                xb = torch.cat([xb, xb[(torch.arange(batch-xb.shape[0]) % xb.shape[0])]])
                valid = X.shape[0] - i
            else:
                valid = batch
            no = cm(xb.to(dev), device=dev).to("cpu").float()
            npu_p = no.argmax(-1)[:valid]
            npu_preds += npu_p.tolist()
            yb = Y[i:i+valid]
            if src == "imagenet_val":
                npu_correct += (npu_p == yb).sum().item()
                if cpu_total < cpu_cap:                     # CPU 참고 정확도(소수만)
                    cpu_p = m(xb).float().argmax(-1)[:valid]
                    cpu_correct += (cpu_p == yb).sum().item(); cpu_total += valid
            total += valid
            if cats is not None:
                ws += sum(1 for pp in npu_p.tolist() if "window screen" in cats[pp])
    res["n_eval"] = total
    res["npu_unique_preds"] = len(set(npu_preds))   # 붕괴 지표(입력민감도)
    res["npu_window_screen"] = ws
    if src == "imagenet_val":
        res["npu_top1_acc"] = round(100 * npu_correct / total, 2)
        res["cpu_top1_acc_ref"] = round(100 * cpu_correct / max(cpu_total, 1), 2)
        res["cpu_ref_n"] = cpu_total
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--batches", nargs="+", type=int, default=[1])
    ap.add_argument("--npu", type=int, default=9)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default=OUT, help="결과 JSON 경로(병렬 실행 시 충돌 방지로 invocation마다 고유하게)")
    ap.add_argument("--optimize", action="store_true", help="택틱 최적화 컴파일(ForVisionModel+pruning)")
    args = ap.parse_args()
    print(f"[sweep] models={args.models} batches={args.batches} npu={args.npu} n_val={args.n_val} optimize={args.optimize} out={args.out}", flush=True)
    allres = json.load(open(args.out)) if os.path.exists(args.out) else []
    for mn in args.models:
        for b in args.batches:
            print(f"\n>>> {mn}  batch={b} optimize={args.optimize} ...", flush=True)
            r = run_one(mn, b, args.npu, args.n_val, args.iters, args.optimize)
            print("   ", {k: r[k] for k in r if k not in ("err",)}, flush=True)
            if "err" in r: print("    ERR:", r["err"], flush=True)
            allres.append(r); json.dump(allres, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
