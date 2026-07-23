#!/usr/bin/env bash
# [A100×2] ViT-Base faithful + train 갤러리(1.28M) 실험 1·2·3·5 를 한 방에.
#   1+2 = canonical(tome/pitome/ours 한 표) → GPU0
#   3   = ablation(register vs 대안)        → GPU1
#   5   = reg-count 스윕 + 95%CI            → GPU0 (canonical 끝난 뒤)
# 필요한 새 파일 2개는 pitome_compare/ 에 있어야 함(ablation_train_faithful.py·regcount_train_faithful.py).
# train 데이터는 pitome_compare/imagenet_train, 파이썬 env 는 canonical faithful 돌리던 그거.
# 사용:  bash run_faithful_train_base.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
M=vit_base_patch14_reg4_dinov2.lvd142m
R="8 12 16 18 20"
mkdir -p "$HERE/canonical/faithful_results" "$HERE/ablation/results" "$HERE/robustness_50k"
ABL_OUT="$HERE/ablation/results/ablation_train_faithful_base.txt"
REG_OUT="$HERE/robustness_50k/reg_count_train_faithful_base.txt"

echo "== 프리플라이트: val 로 스크립트 정상동작 확인(수 분) =="
CUDA_VISIBLE_DEVICES=0 python "$HERE/pitome_compare/ablation_train_faithful.py" --model $M --gallery val --r_list 20
CUDA_VISIBLE_DEVICES=0 python "$HERE/pitome_compare/regcount_train_faithful.py" --model $M --gallery val --r_list 20
echo "== 프리플라이트 OK → 본실행(train 1.28M) 시작 =="

# GPU1: ablation(train) 백그라운드
CUDA_VISIBLE_DEVICES=1 nohup python "$HERE/pitome_compare/ablation_train_faithful.py" \
    --model $M --gallery train --r_list $R > "$ABL_OUT" 2>&1 &
ABL=$!
echo "  [GPU1] ablation → $ABL_OUT (PID $ABL)"

# GPU0: canonical(1+2) 포그라운드 (~12h)
echo "  [GPU0] canonical(head-to-head) 시작 ~12h"
CUDA_VISIBLE_DEVICES=0 bash "$HERE/canonical/run_canonical_faithful.sh" $M base

# GPU0: reg-count(train) (~18h)
echo "  [GPU0] reg-count 시작 ~18h → $REG_OUT"
CUDA_VISIBLE_DEVICES=0 python "$HERE/pitome_compare/regcount_train_faithful.py" \
    --model $M --gallery train --r_list $R > "$REG_OUT" 2>&1

echo "  [GPU1] ablation 끝날 때까지 대기 ..."
wait $ABL

echo "== 전부 완료 =="
echo "  1+2 canonical : $HERE/canonical/faithful_results/canonical_faithful_base.txt"
echo "  3   ablation   : $ABL_OUT"
echo "  5   reg-count  : $REG_OUT"
