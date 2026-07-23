#!/usr/bin/env bash
# DINOv3-S+ / DINOv3-B (timm 공식 미러 가중치) 위 faithful register-보호 토큰축소.
# train 갤러리(1.28M) + val 50k 쿼리, 표준 kNN. 실행 전 selfcheck 통과 필수.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/config.sh" 2>/dev/null || true
cd "$HERE"
: "${GALLERY:=train}"; : "${RLIST:=8 12 16 18 20}"

echo "=== [0] selfcheck (어댑터 정확성 게이트) ==="
python selfcheck.py --model dinov3_base   || { echo "selfcheck FAIL: models_extra.py 수정 필요"; exit 1; }
python selfcheck.py --model dinov3_splus  || { echo "selfcheck FAIL"; exit 1; }

echo "=== [1] DINOv3-B  gallery=$GALLERY ==="
python run_extra.py --model dinov3_base  --gallery "$GALLERY" --r_list $RLIST \
  | tee "$RESULTS/extra_dinov3_base_${GALLERY}_faithful.txt"

echo "=== [2] DINOv3-S+ gallery=$GALLERY ==="
python run_extra.py --model dinov3_splus --gallery "$GALLERY" --r_list $RLIST \
  | tee "$RESULTS/extra_dinov3_splus_${GALLERY}_faithful.txt"

echo "=== [3] DINOv3-B reg-count 스윕(k=0..4) ==="
python run_extra.py --model dinov3_base  --gallery "$GALLERY" --r_list $RLIST --regsweep \
  | tee "$RESULTS/extra_dinov3_base_${GALLERY}_regsweep.txt"

echo "완료. 결과: $RESULTS/extra_dinov3_*"
