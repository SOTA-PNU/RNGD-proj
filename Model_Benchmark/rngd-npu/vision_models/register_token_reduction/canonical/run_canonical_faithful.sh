#!/usr/bin/env bash
# [faithful] canonical (정통 train-갤러리 1.28M) kNN — 정식(faithful) harness.
# compare_faithful.py 사용(compare.py 엔진에 in-block+key-metric+prop-attn forward 주입, PiToMe=공식 pitome_bsm/pitome).
# ⚠️ 매우 무거움: train 1.28M 특징추출 × (tome/pitome/ours) × r. 정식이라 통제보다 느림.
#     대략: base(ViT-B) ~20~25h, small(ViT-S) ~8~10h. 2 GPU면 base=GPU0, small=GPU1 병렬 권장.
# 사용:  bash run_canonical_faithful.sh <MODEL> <TAG>
#   base : CUDA_VISIBLE_DEVICES=0 bash run_canonical_faithful.sh vit_base_patch14_reg4_dinov2.lvd142m base
#   small: CUDA_VISIBLE_DEVICES=1 bash run_canonical_faithful.sh vit_small_patch14_reg4_dinov2.lvd142m small
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ENGINE="$(cd "$HERE/../pitome_compare" && pwd)"
MODEL="${1:-vit_base_patch14_reg4_dinov2.lvd142m}"
TAG="${2:-base}"
RLIST="${RLIST:-8 12 16 18 20}"
GC="${GC:-1}"                 # 1=train 갤러리 특징 디스크캐시(재개 가능, 설정당 수GB)
mkdir -p "$HERE/faithful_results"
echo "=== [faithful canonical] $MODEL (gallery=ImageNet train 1.28M, 정식 harness) ==="
echo "    결과 → $HERE/faithful_results/canonical_faithful_${TAG}.txt (캐시 pitome_compare/feat_cache_faithful)"
python "$ENGINE/compare_faithful.py" --mode acc --gallery train --model "$MODEL" \
       --r_list $RLIST --gallery_cache $GC | tee "$HERE/faithful_results/canonical_faithful_${TAG}.txt"
echo "=== 완료: faithful_results/canonical_faithful_${TAG}.txt ==="
echo "    판정: 무압축 r=0 는 통제와 동일해야(병합無, harness독립). 압축구간 순서·교차점(~74%)이 통제 canonical과 같은지 대조."
