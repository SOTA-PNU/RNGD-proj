#!/usr/bin/env bash
# 전체(S/B/L) 정통 kNN. GPU 서버(A100 2장)에서 실행.
# ⚠️ full 1.28M×3모델×r5 = ~80h(L 혼자 58h)라 비현실적 → 기본 gallery=260k(≈260k, DINOv2 kNN은 갤러리 크기에 강건, ~82 근접).
#   측정 처리량 기준 260k·r5 예상: S~1.3h, B~4.1h, L~13.7h (단일 A100 합계 ~19h). 2장이면 아래 [2-GPU]로 ~병목모델(L) 시간.
# 헤드라인 ViT-B의 '정확한 82'는 run_base_canonical.sh(full 1.28M)로 따로 확보하고, 여기선 S/B/L 추세 일관성 확인이 목적.
set -e
cd "$(dirname "$0")"

GALLERY_PC=${GALLERY_PC:-260}               # 260=≈260k(권장). 0=full 1.28M(정확82, 단 L 58h)
RLIST=${RLIST:-8 12 16 18 20}
GC=${GC:-1}
MODELS=${MODELS:-"vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m vit_large_patch14_reg4_dinov2.lvd142m"}

echo "=== [1] 데이터: val + train(per_class=$GALLERY_PC) ==="
[ -f imagenet_val/DONE ]   || python prepare_data.py --split val
[ -f imagenet_train/DONE ] || { echo "[train 다운로드 $(date +%T)]"; python prepare_data.py --split train --per_class "$GALLERY_PC"; }

echo "=== [2] 정통 kNN sweep: $MODELS ==="
echo "[START $(date +%T)]"; SECONDS=0
for M in $MODELS; do
    tag=$(echo "$M" | sed -E 's/vit_([a-z]+)_.*/\1/')
    echo "--- $M ($tag) $(date +%T) ---"
    python compare.py --mode acc --gallery train --model "$M" --r_list $RLIST --gallery_cache $GC \
        | tee "results_full_${tag}.txt"
done
echo "[END $(date +%T)]  총 = ${SECONDS}s ($((SECONDS/60))분)"

# [2-GPU 배분 예시] — 병목 L을 전용 GPU에, S+B를 다른 GPU에 동시:
#   CUDA_VISIBLE_DEVICES=0 MODELS="vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m" bash run_full_canonical.sh > full_g0.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=1 MODELS="vit_large_patch14_reg4_dinov2.lvd142m"                                      bash run_full_canonical.sh > full_g1.log 2>&1 &
