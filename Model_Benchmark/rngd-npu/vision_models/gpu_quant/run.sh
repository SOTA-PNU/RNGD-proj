#!/usr/bin/env bash
# GPU 서버에서 한 번에 실행: 의존성 설치 → ImageNet val 다운로드 → 양자화 평가.
# 사용: bash run.sh   (가상환경 활성화 상태 권장)
set -e
cd "$(dirname "$0")"

echo "==> [1/3] 의존성 설치"
pip install -r requirements.txt

echo "==> [2/3] ImageNet val 준비 (10장/클래스 = 10000장, HF non-gated)"
if [ ! -f imagenet_val/DONE ]; then
  python prepare_imagenet.py --per_class 10
else
  echo "    imagenet_val/DONE 이미 존재 — 건너뜀"
fi

echo "==> [3/3] 양자화 평가 (6모델 × FP32/FP16/BF16/INT8/FP8 × batch 1/32/128)"
python quantize_eval.py

echo "==> 완료. 결과: results_gpu_quant.json"
