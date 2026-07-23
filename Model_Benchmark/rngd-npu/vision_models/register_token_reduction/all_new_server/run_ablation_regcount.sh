#!/usr/bin/env bash
# [선택] train-갤러리 faithful 진단: ablation(register vs random/energy/highnorm) + reg-count 스윕(+부트스트랩 CI).
#   = "val 로 돌렸던 ablation·reg-count 를 train 전체로" 그대로. 헤드라인(run_train_gallery.sh)과 별개 GPU 에 붙이면 됩니다.
# 사용:
#   bash run_ablation_regcount.sh          # 기본 = b, GPU3
#   bash run_ablation_regcount.sh b 3      # 모델 태그 b, GPU3
set -e
# shellcheck disable=SC1091
source "$(dirname "$0")/config.sh"
cd "$ENGINE"
TAG="${1:-b}"; GPU="${2:-3}"
declare -A MODEL=( [s]="$MODEL_S" [b]="$MODEL_B" [l]="$MODEL_L" )
mdl="${MODEL[$TAG]}"
if [ -z "$mdl" ]; then echo "알 수 없는 태그: $TAG (s|b|l)"; exit 1; fi
export CUDA_VISIBLE_DEVICES="$GPU"

echo "[train-gallery faithful] ablation $TAG=$mdl on GPU$GPU"
python ablation_train_faithful.py --model "$mdl" --gallery train \
    --r_list $RLIST --workers "$WORKERS" --cache_dir "$CACHE/abl_$TAG" --gallery_cache "$GALLERY_CACHE" \
    | tee "$RESULTS/ablation_train_faithful_$TAG.txt"

echo "[train-gallery faithful] reg-count+CI $TAG=$mdl on GPU$GPU"
python regcount_train_faithful.py --model "$mdl" --gallery train \
    --r_list $RLIST --workers "$WORKERS" --cache_dir "$CACHE/reg_$TAG" --gallery_cache "$GALLERY_CACHE" \
    | tee "$RESULTS/regcount_train_faithful_$TAG.txt"

echo "=== ablation·reg-count(train, faithful) 완료. 결과: $RESULTS/*_train_faithful_$TAG.txt ==="
