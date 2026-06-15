#!/usr/bin/env python3
"""비전 모델을 표준 ONNX로 내보낸다 (furiosa-compiler 재현용).

소스 3종을 자동 분기 (compile_vision.py 의 build() 와 동일 규칙):
  - mobilenet_v1 / mobilenetv1   -> timm  mobilenetv1_100   (224)
  - yolov8m                      -> ultralytics YOLO         (640)
  - 그 외(resnet50, mobilenet_v2…) -> torchvision.models     (224)

사용:
  source ~/furiosa/bin/activate
  pip install --no-deps torchvision timm ultralytics   # 정의 라이브러리(없으면)
  python export_onnx.py                                  # resnet50, mobilenet_v2
  python export_onnx.py mobilenet_v1 yolov8m             # 원하는 것만
  python export_onnx.py mobilenet_v1 mobilenet_v2 yolov8m resnet50

이후:
  furiosa-compiler <name>.onnx --target-npu renegade -o <name>.edf
  → README_vision_compile.md 2절대로 표준 ONNX는 파싱 단계에서 거부될 것.
"""
import sys
import torch


def finalize(out):
    """torch 2.10 익스포터는 큰 가중치를 <out>.data 사이드카로 분리 저장한다.
    furiosa-compiler 재현(자족적 단일 파일)을 위해 가중치를 .onnx 안으로 합치고 검증한다."""
    try:
        import onnx, os
        model = onnx.load(out, load_external_data=True)          # 사이드카에서 가중치 로드
        onnx.save_model(model, out, save_as_external_data=False)  # 단일 파일로 인라인 저장
        side = out + ".data"
        if os.path.exists(side):
            os.remove(side)
        onnx.checker.check_model(onnx.load(out))
        print(f"OK exported {out}  ({os.path.getsize(out)/1e6:.1f}MB 단일파일, checker 통과)")
    except ImportError:
        print(f"OK exported {out}  (onnx 패키지 없음 → .onnx + .onnx.data 2파일 그대로)")


def export_one(key):
    out = f"{key}.onnx"

    if key in ("mobilenet_v1", "mobilenetv1"):
        import timm
        m = timm.create_model("mobilenetv1_100", pretrained=False).eval()
        x = torch.randn(1, 3, 224, 224)
        out_names = ["output"]

    elif key == "yolov8m":
        from ultralytics import YOLO
        m = YOLO("yolov8m.pt").model.eval()      # 가중치 자동 다운로드(~50MB)
        x = torch.randn(1, 3, 640, 640)
        out_names = None                          # 탐지 헤드는 다중 출력이라 자동 네이밍

    else:                                         # torchvision
        import torchvision.models as M
        m = getattr(M, key)(weights=None).eval()
        x = torch.randn(1, 3, 224, 224)
        out_names = ["output"]

    for p in m.parameters():
        p.requires_grad_(False)

    kw = dict(opset_version=18, input_names=["input"])  # torch 2.10 익스포터는 opset>=18만 구현
    if out_names is not None:
        kw["output_names"] = out_names
    with torch.no_grad():
        torch.onnx.export(m, x, out, **kw)
    finalize(out)


def main():
    names = sys.argv[1:] or ["resnet50", "mobilenet_v2"]
    for n in names:
        try:
            export_one(n)
        except Exception as e:
            print(f"FAIL {n}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")


if __name__ == "__main__":
    main()
