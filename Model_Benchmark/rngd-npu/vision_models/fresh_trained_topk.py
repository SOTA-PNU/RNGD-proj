#!/usr/bin/env python3
"""결정적 검증: 학습된 vit_b_16을 '정식 from_exported'로 새로 컴파일 → NPU top-1이 맞나?
reuse-edf 가중치 바인딩 의심을 완전히 배제. + 지연시간(속도 베이스라인)도 측정.
'NPU 이미지 분류 성공 베이스'가 실재하는지 판가름."""
import argparse, time, warnings
warnings.filterwarnings("ignore")
import torch
import furiosa.torch
from furiosa.torch import CompileModule
from torch._decomp import core_aten_decompositions, get_decompositions
import torchvision.models as M
from torchvision.models import ViT_B_16_Weights
from PIL import Image

IMG_DIR = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/test_images"
IMAGES = ["brambling.jpg", "tabby_cat.jpg", "convertible.jpg", "orange.jpg", "dog1.jpg", "astronaut.jpg"]
TRUTH = ["brambling", "Egyptian cat", "convertible", "orange", "Pembroke", "bobsled"]
DECOMP = dict(core_aten_decompositions())
DECOMP.update(get_decompositions([
    torch.ops.aten._native_batch_norm_legit_no_training, torch.ops.aten._native_batch_norm_legit,
    torch.ops.aten.batch_norm, torch.ops.aten.native_batch_norm]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npu", type=int, default=16)
    args = ap.parse_args()
    w = ViT_B_16_Weights.IMAGENET1K_V1
    cats = w.meta["categories"]; tf = w.transforms()
    imgs = [tf(Image.open(f"{IMG_DIR}/{f}").convert("RGB")).unsqueeze(0) for f in IMAGES]

    m = M.vit_b_16(weights=w).eval()
    for p in m.parameters(): p.requires_grad_(False)
    imgs = [x.contiguous() for x in imgs]   # 컴파일러는 연속 NCHW만 — channels_last 입력 거부
    t = time.time()
    with torch.no_grad():
        ep = torch.export.export(m, (torch.randn(1, 3, 224, 224),)).run_decompositions(DECOMP)
        cm = CompileModule.from_exported(ep)
    print(f"[from_exported 학습가중치 컴파일 OK] {time.time()-t:.0f}s", flush=True)
    try:
        open("/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/edf/vit_b_16_fromexported.edf", "wb").write(cm.edf.serialize())
        print("  [save] vit_b_16_fromexported.edf", flush=True)
    except Exception as e:
        print(f"  [save] 실패 {type(e).__name__}", flush=True)
    dev = torch.device("rngd", args.npu)
    cm.to(dev)

    cpu_ok = npu_ok = npu_ws = 0
    # warmup
    with torch.no_grad():
        cm(imgs[0].to(dev), device=dev)
    for f, x, truth in zip(IMAGES, imgs, TRUTH):
        with torch.no_grad():
            cpu = m(x)
            xd = x.to(dev)
            t0 = time.time()
            npu = cm(xd, device=dev)
            dt = (time.time() - t0) * 1000
            npu = npu.to("cpu").float()
        ct = cats[int(cpu.argmax(-1))]; nt = cats[int(npu.argmax(-1))]
        cpu_ok += (ct == truth); npu_ok += (nt == truth); npu_ws += ("window screen" in nt)
        print(f"  {f.split('.')[0]:12s} CPU={ct:16s} NPU={nt:18s} ({dt:.1f}ms)", flush=True)
    print(f"\n[판정] from_exported 학습가중치:  CPU정답 {cpu_ok}/6 · NPU정답 {npu_ok}/6 · NPU=window {npu_ws}/6", flush=True)
    if npu_ok >= 4:
        print("  >>> NPU 분류 성공: 베이스 실재. reuse-edf 의 붕괴는 바인딩 아티팩트였음 <<<", flush=True)
    elif npu_ws >= 4:
        print("  >>> NPU 붕괴 확인(window screen): from_exported로도 무너짐 = 진짜 HW 저정밀 붕괴 <<<", flush=True)
    else:
        print("  >>> 부분/혼합 결과 — 추가분석 필요 <<<", flush=True)


if __name__ == "__main__":
    main()
