#!/usr/bin/env python3
"""M4(해자): NPU의 DINOv2 NaN을 '단순 저정밀 시뮬'로 재현할 수 있나?
plain DINOv2를 CPU에서 fp32/bf16/fp16으로 돌려 NaN 여부 확인. 시뮬이 NaN 안 내면 → NPU NaN은
실리콘 고유(TCP cast 경로) → 시뮬 기반 평가로는 못 잡음 = 실측 필수(moat). (CPU, 컴파일 불필요)"""
import warnings; warnings.filterwarnings("ignore")
import torch, timm
from PIL import Image
import torch.nn.functional as Fn

IMG_DIR = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/test_images"
IMGS = ["brambling.jpg", "tabby_cat.jpg", "convertible.jpg", "orange.jpg", "dog1.jpg", "astronaut.jpg"]
name = "vit_base_patch14_dinov2.lvd142m"

base = timm.create_model(name, pretrained=True, num_classes=0, img_size=224).eval()
cfg = timm.data.resolve_model_data_config(base); cfg["input_size"] = (3, 224, 224)
tf = timm.data.create_transform(**cfg, is_training=False)
xs = torch.stack([tf(Image.open(f"{IMG_DIR}/{f}").convert("RGB")) for f in IMGS])

print("M4: plain DINOv2를 저정밀로 시뮬하면 NaN 나는가? (NPU는 6/6 NaN)")
with torch.no_grad():
    ref = base(xs.float()).float()
for dt in [torch.float32, torch.bfloat16, torch.float16]:
    m = timm.create_model(name, pretrained=True, num_classes=0, img_size=224).eval().to(dt)
    with torch.no_grad():
        try:
            out = m(xs.to(dt)).float()
            nan = int(torch.isnan(out).any(dim=-1).sum())
            cos = Fn.cosine_similarity(out, ref, dim=-1).mean().item() if nan == 0 else float('nan')
            print(f"  {str(dt):16s} NaN이미지={nan}/{len(IMGS)}  cosine(vs fp32)={cos:.4f}", flush=True)
        except Exception as e:
            print(f"  {str(dt):16s} 예외: {type(e).__name__} {str(e)[:50]}", flush=True)
print("\n해석: bf16/fp16 시뮬이 NaN 0이면 → NPU의 NaN은 단순 bf16 아님 = TCP 실리콘 고유 경로 → 시뮬 못잡음 = 실측 필수(moat).")
