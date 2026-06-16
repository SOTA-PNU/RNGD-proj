#!/usr/bin/env python3
"""학습된 가중치 + 실제 이미지로 RNGD NPU에서 진짜 ImageNet 분류를 돌린다.

run_edf.py 가 "랜덤 입력으로 NPU==CPU 동치성"을 봤다면, 이건 **학습 가중치 + 진짜 사진**으로
"이 이미지가 무엇인가(top-5 클래스 + 확률)"를 NPU 에서 뽑고 CPU 와 대조한다.

핵심 설계 — EDF 는 "연산 프로그램", 가중치는 "런타임 입력"(furiosa/torch/custom_ops/edf.py:534
의 forward 가 module_weight.flatten_inputs 로 공급). 그래서 두 경로를 지원:
  - compile : 학습 가중치 모델을 그 자리에서 컴파일해 분류 (정석)
  - --reuse-edf <file> : 이미 있는 EDF(랜덤 가중치로 컴파일된 것)를 그대로 쓰고
    가중치만 학습본으로 갈아끼워 분류 → 가중치가 정말 런타임 입력인지 검증(재컴파일 0초)

사용:
  source ~/furiosa/bin/activate            # torchvision 0.25 필요
  python classify.py mobilenet_v2 --npu 0 --images /tmp/dog.jpg /tmp/cat.jpg
  python classify.py mobilenet_v2 --npu 0 --reuse-edf mobilenet_v2.edf --images /tmp/dog.jpg
"""
import argparse, time, warnings
warnings.filterwarnings("ignore")

import torch
import furiosa.torch                        # rngd 백엔드 등록 (torch 다음)
from furiosa.torch import CompileModule
from furiosa.torch.custom_ops.edf import EdfModule
from furiosa.torch.export import ExportedProgramWeight, PASSES
from furiosa.native_torch import ir
from torch._decomp import core_aten_decompositions, get_decompositions
from PIL import Image
import torchvision.models as M
from torchvision.models import MobileNet_V2_Weights, EfficientNet_B0_Weights, ResNet50_Weights

DECOMP = dict(core_aten_decompositions())
DECOMP.update(get_decompositions([
    torch.ops.aten._native_batch_norm_legit_no_training,
    torch.ops.aten._native_batch_norm_legit,
    torch.ops.aten.batch_norm,
    torch.ops.aten.native_batch_norm,
]))

# name -> (학습 가중치로 생성하는 함수, Weights enum: 전처리/라벨 제공)
REG = {
    "mobilenet_v2":    (lambda w: M.mobilenet_v2(weights=w),    MobileNet_V2_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (lambda w: M.efficientnet_b0(weights=w), EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def make_model(name):
    ctor, weights = REG[name]
    m = ctor(weights).eval()                 # ★ 학습된 ImageNet 가중치 로드
    for p in m.parameters():
        p.requires_grad_(False)
    return m, weights


def export_decompose(m, with_passes):
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ep = torch.export.export(m, (x,)).run_decompositions(DECOMP)
    if with_passes:                          # 재사용 경로는 from_exported 를 안 거치므로 PASSES 직접
        for fx_pass in PASSES:
            ep = fx_pass(ep)
    return ep


def preprocess(path, weights):
    """Weights enum 내장 전처리(resize/center-crop/normalize)를 그대로 사용."""
    tf = weights.transforms()
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0)              # 1x3x224x224


def topk(logits, cats, k=5):
    probs = torch.softmax(logits.float(), dim=-1)[0]
    p, idx = probs.topk(k)
    return [(cats[i], float(pi)) for pi, i in zip(p, idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(REG))
    ap.add_argument("--npu", type=int, default=0)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--reuse-edf", default=None,
                    help="이미 있는 EDF 파일(랜덤가중치 컴파일분)을 재사용; 가중치만 학습본으로")
    args = ap.parse_args()

    m, weights = make_model(args.model)
    cats = weights.meta["categories"]

    t = time.time()
    if args.reuse_edf:
        edf = ir.Edf.deserialize(open(args.reuse_edf, "rb").read())
        ep = export_decompose(m, with_passes=True)
        cm = CompileModule(EdfModule(edf), ExportedProgramWeight(ep))   # 재컴파일 없음
        how = f"REUSED_EDF({args.reuse_edf}, {time.time()-t:.1f}s, 컴파일 0)"
    else:
        ep = export_decompose(m, with_passes=False)
        cm = CompileModule.from_exported(ep)                            # 학습 가중치로 컴파일
        how = f"COMPILED({time.time()-t:.1f}s)"

    dev = torch.device("rngd", args.npu)
    cm.to(dev)
    print(f"[{args.model}] {how}  device=rngd:{args.npu}")

    for path in args.images:
        x = preprocess(path, weights)
        with torch.no_grad():
            cpu_logits = m(x)
            t = time.time()
            npu_logits = cm(x.to(dev), device=dev).to("cpu").float()
            ms = (time.time() - t) * 1000
        npu5, cpu5 = topk(npu_logits, cats), topk(cpu_logits, cats)
        name = path.split("/")[-1]
        match = "✓" if npu5[0][0] == cpu5[0][0] else "✗"
        print(f"\n  {name}  ({ms:.1f}ms)  NPU==CPU top-1 {match}")
        print(f"    NPU top-1: {npu5[0][0]} ({npu5[0][1]*100:.1f}%)   | CPU top-1: {cpu5[0][0]} ({cpu5[0][1]*100:.1f}%)")
        print(f"    NPU top-5: " + ", ".join(f"{c}={p*100:.1f}%" for c, p in npu5))


if __name__ == "__main__":
    main()
