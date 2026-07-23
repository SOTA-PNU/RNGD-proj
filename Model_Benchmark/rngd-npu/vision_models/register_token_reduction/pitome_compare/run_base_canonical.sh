#!/usr/bin/env bash
# 베이스(헤드라인) ViT-B 정통 kNN 재현. GPU 서버(A100)에서 실행.
# 기본 = 공식 그대로: gallery=ImageNet train 전체 1.28M, r 5개(8·12·16·18·20). 무압축 baseline≈82 재현 + 정통 head-to-head.
# 측정 처리량(ViT-B ~350 img/s) 기준 단일 A100 예상 ~17시간(하룻밤). 특징 캐시로 중단 후 재개 가능.
#   2장 A100로 반토막: 아래 [2-GPU] 주석 참고.
set -e
cd "$(dirname "$0")"

GALLERY_PC=${GALLERY_PC:-0}                 # 0=train 전체 1.28M(공식 82 정확 재현). 빠르게=260(≈260k, ~82근접, ~4h)
RLIST=${RLIST:-8 12 16 18 20}               # 전체 곡선. 축소 원하면 "12 16 20"
MODEL=vit_base_patch14_reg4_dinov2.lvd142m
GC=${GC:-1}                                 # 1=갤러리 특징 캐시(재개 가능·설정당 ~2GB) / 0=디스크 절약

echo "=== [1/2] 데이터: val(query) + train(gallery, per_class=$GALLERY_PC) ==="
[ -f imagenet_val/DONE ]   || python prepare_data.py --split val
[ -f imagenet_train/DONE ] || { echo "[train 다운로드 $(date +%T)]"; python prepare_data.py --split train --per_class "$GALLERY_PC"; }

echo "=== [2/2] 정통 kNN (ViT-B, gallery=train, r=$RLIST) — 벽시계 측정 ==="
echo "[START $(date +%T)]"; SECONDS=0
python compare.py --mode acc --gallery train --model "$MODEL" --r_list $RLIST --gallery_cache $GC \
    | tee results_base_canonical.txt
echo "[END $(date +%T)]  총 = ${SECONDS}s ($((SECONDS/60))분)"
echo " r=0 이 ~82 근처면 정통 재현 성공. Ours>PiToMe(극단)·Ours>ToMe 확인."

# [2-GPU 반토막 예시] — 두 프로세스로 r을 나눠 동시 실행 후 로그 합치기:
#   CUDA_VISIBLE_DEVICES=0 RLIST="8 12 16"  bash run_base_canonical.sh > base_g0.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=1 RLIST="18 20"    bash run_base_canonical.sh > base_g1.log 2>&1 &
#   (각자 r=0 baseline도 재계산 → 상호 교차검증됨. feat_cache 공유로 중복 추출은 자동 회피)
