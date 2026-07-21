#!/usr/bin/env bash
# 전체(S/B/L) 정통 kNN. GPU 서버(A100 2장). 엔진·데이터는 ../pitome_compare 공유.
# 범위: ToMe·PiToMe·Ours 정확도를 S/B/L 세 크기에서 정통 프로토콜로 재측정(추세 일관성). ablation/dense/eval_v2/robustness 미포함.
# 기본 gallery = full 1.28M(GALLERY_PC=0, 정확한82·헤드라인과 동일 프로토콜). ⚠️ full×3모델×r5 = ~80h(L 혼자 58h).
#    비용 절감 원하면 GALLERY_PC=260(≈260k, ~82근접, 합 ~19h). 2-GPU 배분은 하단 예시.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ENGINE="$(cd "$HERE/../pitome_compare" && pwd)"

GALLERY_PC=${GALLERY_PC:-0}               # 0=full 1.28M(기본·정확82). 260=≈260k(빠름, ~82근접)
# ⚠️ r은 '블록당 제거수'라 블록 수가 다르면 같은 r=다른 압축률. S/B=12블록, L=24블록.
#   같은 압축률(37/55/74/83/92%)로 맞추려면 L은 절반 r을 써야 함(안 그러면 r≥11서 토큰 6개로 붕괴).
RLIST=${RLIST:-8 12 16 18 20}             # S·B(12블록): 37/55/74/83/92%
RLIST_L=${RLIST_L:-4 6 8 9 10}            # L(24블록): 같은 37/55/74/83/92% 에 대응
GC=${GC:-1}
MODELS=${MODELS:-"vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m vit_large_patch14_reg4_dinov2.lvd142m"}

rlist_for() { case "$1" in *large*) echo "$RLIST_L";; *) echo "$RLIST";; esac; }

echo "=== [1] 처리량/지연 (합성 배치·데이터 무관·수초) — 모델별 im/s + batch1 latency ==="
for M in $MODELS; do
    tag=$(echo "$M" | sed -E 's/vit_([a-z]+)_.*/\1/'); RM=$(rlist_for "$M")
    python "$ENGINE/compare.py" --mode tput --model "$M" --batch 128 --r_list 0 $RM | tee "$HERE/results_full_${tag}_tput.txt"
    python "$ENGINE/compare.py" --mode tput --model "$M" --batch 1   --r_list 0 $RM | tee "$HERE/results_full_${tag}_latency.txt"
done

echo "=== [2] 데이터(../pitome_compare 공유): val + train(per_class=$GALLERY_PC) ==="
[ -f "$ENGINE/imagenet_val/DONE" ]   || python "$ENGINE/prepare_data.py" --split val
[ -f "$ENGINE/imagenet_train/DONE" ] || { echo "[train 다운로드 $(date +%T)]"; python "$ENGINE/prepare_data.py" --split train --per_class "$GALLERY_PC"; }

echo "=== [3] 정통 kNN sweep: $MODELS ==="
echo "[START $(date +%T)]"; SECONDS=0
for M in $MODELS; do
    tag=$(echo "$M" | sed -E 's/vit_([a-z]+)_.*/\1/'); RM=$(rlist_for "$M")
    echo "--- $M ($tag, r=$RM) $(date +%T) ---"
    python "$ENGINE/compare.py" --mode acc --gallery train --model "$M" --r_list $RM --gallery_cache $GC \
        | tee "$HERE/results_full_${tag}.txt"
done
echo "[END $(date +%T)]  총 = ${SECONDS}s ($((SECONDS/60))분)"

# [2-GPU 배분] 병목 L을 전용 GPU에, S+B를 다른 GPU에 동시:
#   CUDA_VISIBLE_DEVICES=0 MODELS="vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m" bash run_full_canonical.sh > full_g0.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=1 MODELS="vit_large_patch14_reg4_dinov2.lvd142m"                                      bash run_full_canonical.sh > full_g1.log 2>&1 &
