# vision_models/edf — 컴파일된 NPU 아티팩트(EDF) 모음

이 폴더는 vision 모델을 RNGD NPU용으로 컴파일한 **EDF**(furiosa.torch가 만든 연산 그래프 아티팩트)를 모아 둔 곳입니다.

## EDF가 무엇인가요
- EDF는 모델의 **아키텍처(연산 그래프)** 를 NPU 실행 형태로 컴파일한 파일입니다.
- **가중치는 EDF 안에 구워지지 않고 런타임 입력으로 들어갑니다.** 그래서 같은 EDF에 다른 가중치(학습/랜덤/수정)를 바꿔 끼워 재컴파일 없이(`reuse-edf`) 돌릴 수 있습니다.
- 컴파일은 한 번에 ~15–20분 걸리므로, 저장해 두면 같은 구조 실험을 0초 재컴파일로 반복할 수 있습니다.

## 파일 목록

| 파일 | 모델 | 만든 스크립트 | 비고 |
|---|---|---|---|
| `vit_b_16_trained.edf` | torchvision `vit_b_16` (학습 가중치, batch 1) | `classify.py` / 잔류성 검증 | 붕괴·복구 실험(recover_poc/recover_fold)의 `reuse-edf` 베이스 |
| `dinov2_b14.edf` | timm `vit_base_patch14_dinov2` (img_size 224, batch 1) | `dino_collapse.py` | foundation 인코더, NPU 출력 NaN 관측 |
| `vit_b_16_fromexported.edf` | torchvision `vit_b_16` (학습, from_exported 정식 컴파일) | `fresh_trained_topk.py` | reuse-edf 의심 배제용 정식 경로 검증 |
| `<model>_b<batch>.edf` | timm 모델 sweep (vit/deit tiny·small·base) | `vision_sweep.py` | ACCV 실험 sweep 산출물(모델×배치) |

## 만드는 방법
- 정식 컴파일은 `CompileModule.from_exported(ep)` 후 `cm.edf.serialize()` 로 저장합니다.
- **입력은 반드시 연속(NCHW contiguous)** 이어야 합니다. torchvision의 ViT 변환은 channels_last(비연속) 텐서를 만들 수 있어, 그대로 export하면 컴파일러가 `apply_dram_shape_guide` 에서 패닉합니다 → export는 `torch.randn(...)`(연속)으로, 실행 입력은 `.contiguous()` 로 넣습니다.

> 출처: 본 폴더 스크립트(`fresh_trained_topk.py`, `vision_sweep.py`, `dino_collapse.py`, `recover_fold.py`)에서 실제로 저장·사용하는 경로 기준으로 작성했습니다.
