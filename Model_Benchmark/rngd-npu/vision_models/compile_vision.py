#!/usr/bin/env python3
"""RNGD(furiosa.torch)로 비전/CNN 모델을 컴파일·실행해보는 검증 스크립트.

결과 해설은 info/README_vision_compile.md 참고.

핵심:
  - 경로는 furiosa.torch (furiosa-llm 내부 컴파일러). furiosa-compiler CLI 의 ONNX 경로,
    furiosa.models 의 비전 모델은 현재 SDK(2026.2.0)에서 못 씁니다.
  - batch_norm 은 furiosa 기본 분해 테이블에 없어서 직접 분해해야 importer 가 받습니다.
  - 중간 풀링 레이어(MaxPool/AvgPool)가 있으면 "multiple internal subgraphs" 로 막힙니다
    (ResNet stem maxpool, YOLO SPPF 등). strided-conv 로 다운샘플하는 MobileNet/EfficientNet 은 OK.

사용:
  source ~/furiosa/bin/activate
  # 모델 정의용(임시): pip install --no-deps torchvision timm
  python compile_vision.py mobilenet_v2            # 컴파일만
  python compile_vision.py mobilenet_v2 --run 3    # rngd:3 에서 실행 + CPU 대조
  python compile_vision.py resnet50                # 실패(unsupported EDF node: Cpu)
  python compile_vision.py resnet50 --no-maxpool   # stem 풀링 제거해도 ResNet50 은 여전히 실패(실측)
"""
import argparse, time, warnings
warnings.filterwarnings("ignore")

import torch
import furiosa.torch                       # PrivateUse1("rngd") 백엔드 등록 (torch 다음에 import)
from furiosa.torch import CompileModule
from torch._decomp import core_aten_decompositions, get_decompositions

# core-aten + batch_norm (batch_norm 은 furiosa STD_DECOMPOSITIONS 에 없음 → 직접 추가)
DECOMP = dict(core_aten_decompositions())
DECOMP.update(get_decompositions([
    torch.ops.aten._native_batch_norm_legit_no_training,
    torch.ops.aten._native_batch_norm_legit,
    torch.ops.aten.batch_norm,
    torch.ops.aten.native_batch_norm,
]))


def build(key, no_maxpool=False):
    """모델과 입력 shape 반환. 가중치는 그래프 검증 목적이라 랜덤(weights=None)."""
    if key == "mobilenetv1":
        import timm
        return timm.create_model("mobilenetv1_100", pretrained=False), (1, 3, 224, 224)
    if key == "yolov8m":
        from ultralytics import YOLO
        return YOLO("/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/weights/yolov8m.pt").model, (1, 3, 640, 640)
    import torchvision.models as M
    m = getattr(M, key)(weights=None)
    if no_maxpool and hasattr(m, "maxpool"):
        m.maxpool = torch.nn.Identity()    # ResNet stem 풀링 제거 → 컴파일 가능(해상도는 달라짐)
    return m, (1, 3, 224, 224)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--run", type=int, metavar="NPU_IDX", default=None,
                    help="컴파일 후 rngd:NPU_IDX 에서 실행 + CPU 대조 (이 머신은 npu3 이 빔)")
    ap.add_argument("--no-maxpool", action="store_true", help="ResNet stem MaxPool 제거")
    args = ap.parse_args()

    torch.manual_seed(0)
    m, shape = build(args.model, no_maxpool=args.no_maxpool)
    m = m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    x = torch.randn(*shape)

    with torch.no_grad():
        ref = m(x) if args.run is not None else None
        ep = torch.export.export(m, (x,)).run_decompositions(DECOMP)
    nodes = sum(1 for n in ep.graph.nodes if n.op == "call_function")

    t = time.time()
    try:
        with torch.no_grad():
            cm = CompileModule.from_exported(ep)
    except Exception as e:
        cause = e
        while cause.__cause__ is not None:
            cause = cause.__cause__
        print(f"[{args.model}] COMPILE_FAIL  nodes={nodes}  after={time.time()-t:.1f}s")
        print(f"  원인: {str(cause).splitlines()[0][:160]}")
        return
    print(f"[{args.model}] COMPILE_OK  nodes={nodes}  compile={time.time()-t:.1f}s  edf={type(cm.edf).__name__}")

    if args.run is not None:
        dev = torch.device("rngd", args.run)
        cm.to(dev)
        t = time.time()
        with torch.no_grad():
            out = cm(x.to(dev), device=dev)
        dt = (time.time() - t) * 1000
        out = out.to("cpu").float()
        diff = (out - ref.float()).abs()
        print(f"  RAN_ON_NPU rngd:{args.run}  latency={dt:.1f}ms  "
              f"max_abs_err={diff.max():.3g}  top1_match={int(out.argmax(-1))==int(ref.float().argmax(-1))}")


if __name__ == "__main__":
    main()
