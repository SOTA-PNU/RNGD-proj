#!/usr/bin/env bash
# GPU 서버에서 한 번에: 의존성 설치 → ImageNet val 다운로드 → register-aware 토큰압축 평가.
# 사용: bash run.sh   (CUDA GPU 권장. CPU도 동작하나 느림)
set -e
cd "$(dirname "$0")"

echo "==> [1/3] 의존성 설치"
pip install -r requirements.txt

echo "==> [2/3] ImageNet val 준비 (기본 50장/클래스=풀 50k; 빠르게 보려면 prepare_data.py --per_class 10)"
[ -f imagenet_val/DONE ] || python prepare_data.py --per_class 50

echo "==> [3/3] 평가: DINOv2-reg, 압축률별 ToMe(prot=1) vs ours(prot=5) kNN 정확도"
python eval_imagenet.py --n 50000 --batch 128 --r_list 0 8 12 16 18 20

echo "==> 완료. (다른 모델: --model vit_base_patch14_dinov2.lvd142m / vit_base_patch16_clip_224.openai)"
