#!/usr/bin/env python3
"""[새 서버] DINOv2-reg S/B/L 사전학습 가중치를 미리 받아 HF 캐시에 저장합니다(첫 실험 전 1회, 온라인).
새 서버엔 가중치 캐시가 없어서, 이걸 안 하면 tome_core.py 가 오프라인 모드로 timm 가중치를 못 받을 수 있습니다.
한 번 받아두면 이후 실험은 캐시된 가중치로 돕니다. GPU 없어도 CPU 로 로드만 하면 다운로드됩니다.
사용: python warmup_models.py"""
import os
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
import timm

MODELS = [
    "vit_small_patch14_reg4_dinov2.lvd142m",
    "vit_base_patch14_reg4_dinov2.lvd142m",
    "vit_large_patch14_reg4_dinov2.lvd142m",
]

for name in MODELS:
    print(f"[warmup] {name} 다운로드/로드 중 ...", flush=True)
    timm.create_model(name, pretrained=True, num_classes=0, img_size=224)
    print(f"[ok] {name} 캐시 완료", flush=True)
print("완료: 세 모델 가중치 캐시됨. 이제 오프라인에서도 실행 가능합니다.")
