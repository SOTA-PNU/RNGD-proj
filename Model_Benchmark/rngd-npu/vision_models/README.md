# vision_models — RNGD NPU 비전 모델 실험 모음

이 폴더는 RNGD NPU에서 비전 모델(CNN·ViT·foundation 인코더)을 **컴파일·실행·측정**하는 실험 코드와 산출물을 한곳에 모은 곳입니다. 여기저기 흩어져 있던 스크립트·EDF·로그·데이터를 정리했습니다.

## 폴더 구조

```
vision_models/
├── README.md                  # (이 문서) 전체 인덱스
├── classify.py                # 학습 가중치로 NPU vs CPU top-1 분류 (--reuse-edf 지원)
├── compile_vision.py          # 여러 비전 모델의 컴파일 가능 여부 점검 (랜덤 가중치)
├── fresh_trained_topk.py      # 정식 from_exported 컴파일로 "분류 성공 베이스" 검증 + 지연 측정
├── recover_poc.py             # 복구 시험 1: per-channel outlier 클립 sweep
├── recover_fold.py            # 복구 시험 2: per-channel 스케일 접기(SmoothQuant 등화)
├── dino_collapse.py           # foundation 인코더(DINOv2) 임베딩 충실도/붕괴 측정
├── vision_sweep.py            # ACCV 실험 sweep: 모델×배치 정확도+속도 측정
├── edf/                       # 컴파일된 NPU 아티팩트(EDF) + 설명 README
├── weights/                   # .pt 체크포인트 (mobilenet/efficientnet/yolov8 등)
├── test_images/               # sanity용 라벨 이미지 9장 (새·고양이·차·과일 등)
├── imagenet_val/              # HF ImageNet-1k val 부분집합 + labels.csv + META
├── results/                   # 실험 로그·JSON 결과
│   ├── exp_plan.json          #   ACCV 실험설계(모델/데이터셋/베이스라인) distill
│   ├── sweep_results.json     #   vision_sweep.py 산출(모델×배치 정확도·속도)
│   └── logs/                  #   각 실험 실행 로그(*.log)
└── tools/                     # 진단·생성 도구
    ├── probe_devices.py, probe2_devices.py   # 어느 NPU PE가 free인지 빠른 프로브
    ├── vit_residency_check.py                # NPU 실거주(잔류) 검증
    ├── make_diagrams.py, make_recovery_chart.py  # 논문용 개념도·결과 차트 생성
    ├── extract.py, prep_recovery_targets.py  # 보조 스크립트
    └── ppt/                   # 발표 PPT 빌더(산출물은 ../../ACCV/)
```

## 핵심 개념: EDF와 가중치 분리
- 컴파일 산출물 **EDF는 모델 구조(연산 그래프)만** 담고, **가중치는 런타임 입력**입니다.
- 그래서 같은 EDF에 학습/랜덤/수정 가중치를 바꿔 끼워 **재컴파일 없이(`reuse-edf`)** 돌릴 수 있습니다(컴파일 1회 ~15–20분 → 이후 ~18초).
- 자세한 EDF 목록·생성법은 [`edf/README.md`](edf/README.md) 참고.

## 디바이스 인덱싱 (실측)
- `rngd:N` 은 **전역 PE 인덱스**입니다: `rngd:0–7`=npu0, `8–15`=npu1, `16–23`=npu2, `24–31`=npu3.
- free PE 확인은 `tools/probe2_devices.py` 로(저장된 EDF로 0초 프로브).

## 자주 쓰는 실행 예시
```bash
source ~/furiosa/bin/activate

# 1) 분류 성공 베이스 검증 (정식 from_exported, 학습 가중치)
python fresh_trained_topk.py --npu 16

# 2) ACCV sweep: 6모델 batch=1 + vit-base 배치 sweep
python vision_sweep.py --models vit_base_patch16_224.augreg_in1k --batches 1 2 4 8 --npu 9
python vision_sweep.py --models vit_tiny_patch16_224.augreg_in21k_ft_in1k \
  vit_small_patch16_224.augreg_in1k vit_base_patch16_224.augreg_in1k \
  deit_tiny_patch16_224.fb_in1k deit_small_patch16_224.fb_in1k deit_base_patch16_224.fb_in1k \
  --batches 1 --npu 9

# 3) 복구 시험 (reuse-edf, 빠름)
python recover_poc.py  --npu 8 --pcts 100 99.5 99 98
python recover_fold.py --npu 8 --alphas 0.25 0.5 0.75
```

## 입력 텐서 주의 (실측 함정)
- 컴파일러는 **연속(NCHW contiguous) 입력만** 받습니다. torchvision ViT 변환은 channels_last(비연속) 텐서를 만들 수 있어, 그대로 export하면 `apply_dram_shape_guide` 에서 컴파일러가 패닉합니다.
- 해결: export는 `torch.randn(...)`(연속)으로, 실행 입력은 `.contiguous()` 로 넣습니다.

> 이 문서는 본 폴더의 실제 파일·스크립트 경로를 기준으로 작성했습니다. ACCV 발표/논문 산출물(PPT·논문 md)은 `../../ACCV/` 에 있습니다.
