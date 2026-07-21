#!/usr/bin/env bash
# DINOv2 no-reg를 tab:extra와 '같은 프로토콜(정통 train-갤러리 1.28M)'로 측정.
#
# ⚠️ 프로토콜 정정(중요):
#   tab:extra 캡션은 "표준 train-갤러리 kNN"이고 DINOv3/ViT-5 no-reg도 gallery=train(1.28M)으로 쟀다.
#   그런데 기존 DINOv2 no-reg(results/dinov2_noreg_control.txt = 75.85…)는 eval_imagenet.py의
#   50k val self-kNN(leave-one-out)이라 캡션·다른 행과 프로토콜이 어긋난다.
#   → 여기서는 DINOv2 no-reg를 DINOv3/ViT-5와 동일하게 '정통 train-갤러리(1.28M)'로 다시 잰다.
#
# 방법: 검증된 canonical/run_canonical_faithful.sh(= compare_faithful.py --gallery train)를
#       레지스터 '없는' DINOv2 체크포인트로 그대로 호출만 한다(원본 코드 수정 없음).
#       no-reg 모델은 prefix=1(레지스터 0)이라 'ours'='tome'(보호할 레지스터가 없음) → no-reg 값 = tome 열.
#
# 대상:
#   DINOv2-S no-reg = vit_small_patch14_dinov2.lvd142m  (표에 빠져 있던 행)
#   DINOv2-B no-reg = vit_base_patch14_dinov2.lvd142m   (기존 50k-val 값을 train-갤러리로 교체)
#
# 실행(GPU 서버, train 1.28M 준비 상태):
#   CUDA_VISIBLE_DEVICES=0 bash extra_dinov2s_noreg/run_dinov2s_noreg.sh small   # DINOv2-S no-reg
#   CUDA_VISIBLE_DEVICES=1 bash extra_dinov2s_noreg/run_dinov2s_noreg.sh base    # DINOv2-B no-reg (권장: 재측정)
#   (인자 없으면 small→base 순차)
# ⚠️ 매우 무거움: train 1.28M 특징추출. small ~8-10h, base ~20-25h(canonical faithful과 동일 규모).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
RUNNER="$ROOT/canonical/run_canonical_faithful.sh"

run_one () {
  local MODEL="$1" TAG="$2"
  echo "=== [no-reg train-갤러리] $MODEL → tag=$TAG ==="
  RLIST="8 12 16 20" bash "$RUNNER" "$MODEL" "$TAG"
  echo "    결과: $ROOT/canonical/faithful_results/canonical_faithful_${TAG}.txt"
  echo "    읽는 법: no-reg 모델은 prefix=1이라 'tome' 열(=ours)이 no-reg 값. r=0/12/16/20 → 무압축/~55/74/92%."
}

case "${1:-both}" in
  small) run_one vit_small_patch14_dinov2.lvd142m small_noreg ;;
  base)  run_one vit_base_patch14_dinov2.lvd142m  base_noreg  ;;
  both)  run_one vit_small_patch14_dinov2.lvd142m small_noreg
         run_one vit_base_patch14_dinov2.lvd142m  base_noreg ;;
  *) echo "usage: $0 [small|base|both]"; exit 1 ;;
esac

echo ""
echo "==> 완료. 두 로그의 'tome' 열(무압축 r=0 / 55% r=12 / 74% r=16 / 92% r=20)을 보내주면"
echo "    tab:extra의 DINOv2-S no-reg 행 추가 + DINOv2-B no-reg 행 교체(50k-val→train-갤러리)를 반영한다."
