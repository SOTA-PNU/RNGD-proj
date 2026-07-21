# gpu_quant — GPU 서버용 ViT/DeiT 양자화 평가

이 폴더는 **GPU 서버에 복사해서 그대로 실행**하면 되는 self-contained 양자화 실험입니다. ViT/DeiT 6모델을 여러 정밀도(FP32·FP16·BF16·INT8·FP8)로 양자화하고, ImageNet val에서 **top-1 정확도**와 **GPU 지연/throughput**을 측정합니다.

## 왜 GPU에서 하나요
RNGD NPU(furiosa.torch)는 **비전 모델용 양자화 공개 API가 없습니다**(FP8은 LLM 전용). NPU는 컴파일러가 내부적으로 저정밀 실행을 자동 적용할 뿐, 사용자가 INT8/FP8 PTQ를 거는 손잡이가 없습니다. 그래서 "양자화 기법 적용 → 정확도/속도 변화" 비교는 **GPU 서버에서** 수행합니다. (NPU 쪽 효율 측정은 상위 폴더 `../vision_sweep.py` 참고.)

## 빠른 실행
```bash
# (권장) 가상환경 활성화 후
bash run.sh
```
`run.sh` 는 ①의존성 설치 → ②ImageNet val 다운로드(HF, 토큰 불필요, 10장/클래스=10000장) → ③양자화 평가를 순서대로 합니다.

## 수동 실행
```bash
pip install -r requirements.txt
python prepare_imagenet.py --per_class 10          # ./imagenet_val/ 생성
python quantize_eval.py                            # 전체 측정
# 일부만:
python quantize_eval.py --models vit_base_patch16_224.augreg_in1k \
       --modes fp32 fp16 int8_weight int8_dynamic fp8 --batches 1 32 128
```

## 측정 대상
- **모델(6):** `vit_tiny/small/base_patch16_224`, `deit_tiny/small/base_patch16_224` (timm, ImageNet-1k 사전학습, HF 자동 다운로드)
- **정밀도(modes):**
  - `fp32` / `fp16` / `bf16` — torch만으로 동작
  - `int8_weight` — torchao weight-only INT8 (bf16 활성)
  - `int8_dynamic` — torchao dynamic activation + weight INT8
  - `fp8` — torchao FP8 weight-only (sm89+ GPU, 예: H100/Ada 필요)
- **배치:** 1 / 32 / 128 (지연·throughput 스케일)
- **지표:** ImageNet top-1(%), ms/batch, img/s

## 요구사항
- CUDA GPU (FP8은 H100/Ada 등 sm89+). CPU에서도 정확도는 측정되나 속도는 무의미.
- `torch>=2.4` + `torchao>=0.7` (INT8/FP8용). torchao 없으면 INT8/FP8은 자동 skip되고 FP32/FP16/BF16만 측정됩니다.

## 출력
- `results_gpu_quant.json` — 각 (모델, 정밀도, 배치)별 `top1`, `ms_per_batch`, `img_per_s` 레코드. 지원 안 되는 조합은 `ok:false`로 사유 기록.
- 표준출력에 표 형태로도 출력.

## NPU 결과와 비교하는 법
- 상위 `../results/sweep_combined.json` = **NPU(RNGD)** 의 같은 모델 정확도·속도.
- 본 폴더 `results_gpu_quant.json` = **GPU** 의 정밀도별 정확도·속도.
- 두 결과를 합치면 "NPU 자동 저정밀 vs GPU 명시적 PTQ"를 비교할 수 있습니다.

> 라벨 순서: ImageNet val의 `label_idx`는 표준 ILSVRC2012 / torchvision 순서라 timm 모델 출력과 바로 비교됩니다.
