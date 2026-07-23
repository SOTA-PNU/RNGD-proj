#!/usr/bin/env python3
"""M2 복구: register 없이 NaN 고치기. M1이 'block9서 한 토큰이 mean+많은σ로 폭발→저정밀 오버플로→NaN'을 보였으니,
각 블록 출력에서 토큰 노름을 mean+kσ로 adaptive 클립(아웃라이어만 억제, 방향 보존) → from_exported로 재컴파일·NPU 실행 →
NaN→유한 + CPU(동일 클립) 대비 cosine 회복을 측정. 재학습·정밀도손잡이 없음.
사용: python token_clip_repair.py --npu 8 --k 6"""
import argparse, time, warnings
warnings.filterwarnings("ignore")
import torch, furiosa.torch
from furiosa.torch import CompileModule
from torch._decomp import core_aten_decompositions, get_decompositions
import timm
from PIL import Image
import torch.nn.functional as Fn

IMG_DIR = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/test_images"
IMGS = ["brambling.jpg", "tabby_cat.jpg", "convertible.jpg", "orange.jpg", "dog1.jpg", "astronaut.jpg"]
D = dict(core_aten_decompositions())
D.update(get_decompositions([torch.ops.aten._native_batch_norm_legit_no_training,
    torch.ops.aten._native_batch_norm_legit, torch.ops.aten.batch_norm, torch.ops.aten.native_batch_norm]))


def wrap_clip(blk, cap):
    # 블록 *입력*에 element-wise clamp: 극단 아티팩트 element(480 등)만 억제, 정상(99.9%<8)은 보존.
    # 입력에 걸어야 LN(scale-invariant)을 통과해 어텐션 오버플로를 막는다. clamp=지원 op(리덕션 없음).
    orig = blk.forward
    def f(x, *a, **kw):
        x = torch.clamp(x, -cap, cap)
        return orig(x, *a, **kw)
    blk.forward = f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npu", type=int, default=8)
    ap.add_argument("--cap", type=float, default=150.0)
    args = ap.parse_args()
    name = "vit_base_patch14_dinov2.lvd142m"
    m = timm.create_model(name, pretrained=True, num_classes=0, img_size=224).eval()
    for p in m.parameters(): p.requires_grad_(False)
    for blk in m.blocks:
        wrap_clip(blk, args.cap)
    cfg = timm.data.resolve_model_data_config(m); cfg["input_size"] = (3, 224, 224)
    tf = timm.data.create_transform(**cfg, is_training=False)
    imgs = [tf(Image.open(f"{IMG_DIR}/{f}").convert("RGB")).unsqueeze(0).contiguous() for f in IMGS]

    t = time.time()
    with torch.no_grad():
        ep = torch.export.export(m, (torch.randn(1, 3, 224, 224),)).run_decompositions(D)
        cm = CompileModule.from_exported(ep)
    print(f"[M2 clip cap={args.cap}] COMPILE_OK {time.time()-t:.0f}s", flush=True)
    try:
        open(f"/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/edf/dinov2_clip_cap{int(args.cap)}.edf", "wb").write(cm.edf.serialize())
    except Exception: pass
    dev = torch.device("rngd", args.npu); cm.to(dev)

    coss, nan = [], 0
    for f, x in zip(IMGS, imgs):
        with torch.no_grad():
            c = m(x).float()
            n = cm(x.to(dev), device=dev).to("cpu").float()
        cos = Fn.cosine_similarity(c, n, dim=-1).mean().item()
        coss.append(cos); nan += int(torch.isnan(n).any())
        print(f"  {f.split('.')[0]:12s} NPU~CPU(클립) cosine = {cos:.4f}  NaN={'Y' if torch.isnan(n).any() else 'N'}", flush=True)
    avg = sum(coss)/len(coss)
    print(f"\n[M2 결과 cap={args.cap}] 평균 cosine={avg:.4f}  NaN이미지={nan}/{len(imgs)}", flush=True)
    if nan == 0 and avg > 0.9:
        print(">>> 복구 성공: register 없이 토큰노름 클립만으로 NaN 제거 + NPU=CPU 충실도 회복 (재학습 불요) <<<", flush=True)
    elif nan == 0:
        print(">>> NaN은 제거됐으나 충실도 낮음 — k 조정 필요 <<<", flush=True)
    else:
        print(">>> 여전히 NaN — 클립 위치/강도 조정 필요 <<<", flush=True)


if __name__ == "__main__":
    main()
