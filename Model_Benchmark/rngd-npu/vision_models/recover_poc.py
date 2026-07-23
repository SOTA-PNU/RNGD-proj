#!/usr/bin/env python3
"""#1/#11 복구 PoC: 학습 ViT-B/16의 붕괴를 per-channel weight clip(레버2)으로 되살리나?

설계: EDF는 가중치 독립 → 저장된 EDF(랜덤/학습 무관, 같은 vit_b_16 그래프)를 재사용하고
가중치만 '클립한 학습본'으로 바꿔 NPU 재측정(재컴파일 0초). pct sweep으로 어느 강도가 듣는지 확인.
- clip: 각 weight 텐서를 출력채널별 |w| 백분위 pct에서 대칭 클램프(heavy tail 절단).
- class_token(k=393 outlier)도 클립.
baseline(pct=100, 무클립)은 붕괴 재현 + reuse-edf 경로 검증을 겸함.

사용: python recover_poc.py [--npu 0] [--pcts 100 99.9 99.5 99 98]
"""
import argparse, time, warnings
warnings.filterwarnings("ignore")
import torch
import furiosa.torch
from furiosa.torch import CompileModule
from furiosa.torch.custom_ops.edf import EdfModule
from furiosa.torch.export import ExportedProgramWeight, PASSES
from furiosa.native_torch import ir
from torch._decomp import core_aten_decompositions, get_decompositions
import torchvision.models as M
from torchvision.models import ViT_B_16_Weights
from PIL import Image

EDF_PATH = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/edf/vit_b_16_trained.edf"
IMG_DIR = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/test_images"
IMAGES = ["brambling.jpg", "tabby_cat.jpg", "convertible.jpg", "orange.jpg", "dog1.jpg", "astronaut.jpg"]
TRUTH = ["brambling", "Egyptian cat", "convertible", "orange", "Pembroke", "bobsled"]  # CPU FP32 top-1

DECOMP = dict(core_aten_decompositions())
DECOMP.update(get_decompositions([
    torch.ops.aten._native_batch_norm_legit_no_training,
    torch.ops.aten._native_batch_norm_legit,
    torch.ops.aten.batch_norm, torch.ops.aten.native_batch_norm,
]))


def clip_model(m, pct):
    """pct<100이면 출력채널별 |w| 백분위에서 대칭 클램프. pct=100이면 무변경."""
    if pct >= 100:
        return 0
    n = 0
    q = pct / 100.0
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() >= 2:
                w = p.data
                flat = w.reshape(w.shape[0], -1).abs()
                thr = torch.quantile(flat, q, dim=1)  # [out_ch]
                thr = thr.reshape([w.shape[0]] + [1] * (w.dim() - 1)).clamp_min(1e-8)
                p.data = torch.clamp(w, -thr, thr)
                n += 1
            elif name.endswith("class_token") or name == "class_token":
                t = p.data
                thr = torch.quantile(t.abs(), q).clamp_min(1e-8)
                p.data = torch.clamp(t, -thr, thr)
                n += 1
    return n


def build_cm(m, edf, dev):
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ep = torch.export.export(m, (x,)).run_decompositions(DECOMP)
        for fx in PASSES:
            ep = fx(ep)
        cm = CompileModule(EdfModule(edf), ExportedProgramWeight(ep))
    cm.to(dev)
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npu", type=int, default=0)
    ap.add_argument("--pcts", type=float, nargs="+", default=[100, 99.9, 99.5, 99, 98])
    args = ap.parse_args()

    w = ViT_B_16_Weights.IMAGENET1K_V1
    cats = w.meta["categories"]
    tf = w.transforms()
    imgs = [tf(Image.open(f"{IMG_DIR}/{f}").convert("RGB")).unsqueeze(0) for f in IMAGES]
    edf = ir.Edf.deserialize(open(EDF_PATH, "rb").read())
    dev = torch.device("rngd", args.npu)
    print(f"[setup] EDF 재사용 {EDF_PATH}  images={len(imgs)}  device=rngd:{args.npu}", flush=True)

    for pct in args.pcts:
        m = M.vit_b_16(weights=w).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        nclip = clip_model(m, pct)
        t = time.time()
        try:
            cm = build_cm(m, edf, dev)
            how = f"reuse-edf ({time.time()-t:.1f}s)"
        except Exception as e:
            print(f"[pct={pct}] reuse-edf 실패→fallback recompile: {type(e).__name__}: {str(e)[:90]}", flush=True)
            with torch.no_grad():
                ep = torch.export.export(m, (torch.randn(1, 3, 224, 224),)).run_decompositions(DECOMP)
                cm = CompileModule.from_exported(ep); cm.to(dev)
            how = f"recompiled ({time.time()-t:.1f}s)"

        cpu_ok = npu_ok = npu_ws = 0
        rows = []
        for f, x, truth in zip(IMAGES, imgs, TRUTH):
            with torch.no_grad():
                cpu = m(x); npu = cm(x.to(dev), device=dev).to("cpu").float()
            ct = cats[int(cpu.argmax(-1))]; nt = cats[int(npu.argmax(-1))]
            cpu_ok += (ct == truth); npu_ok += (nt == truth); npu_ws += ("window screen" in nt)
            rows.append(f"{f.split('.')[0]:12s} CPU={ct:16s} NPU={nt}")
        print(f"\n[pct={pct}  clipped_tensors={nclip}  {how}] "
              f"CPU정답 {cpu_ok}/6 · NPU정답 {npu_ok}/6 · NPU=window_screen {npu_ws}/6", flush=True)
        for r in rows:
            print("    " + r, flush=True)


if __name__ == "__main__":
    main()
