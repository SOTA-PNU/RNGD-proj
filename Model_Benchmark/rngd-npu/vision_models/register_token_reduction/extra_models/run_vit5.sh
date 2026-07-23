#!/usr/bin/env bash
# ViT-5(공식 wangf3014/ViT-5) 위 faithful register-보호 토큰축소.
# 사전: 공식 repo clone + 공식 체크포인트(.pth) 준비 후 아래 두 경로 지정(또는 config.sh 에 export).
#   git clone https://github.com/wangf3014/ViT-5   $VIT5_REPO
#   huggingface-cli download FengWang3211/ViT-5 vit5_base_patch16_224.pth --local-dir <dir>
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/config.sh" 2>/dev/null || true
cd "$HERE"
: "${GALLERY:=train}"; : "${RLIST:=8 12 16 18 20}"
: "${VIT5_REPO:?공식 repo clone 경로를 VIT5_REPO 로 지정하세요}"
: "${VIT5_CKPT:?공식 vit5_base .pth 경로를 VIT5_CKPT 로 지정하세요}"

echo "=== [0] selfcheck (어댑터 정확성 게이트) ==="
python selfcheck.py --model vit5_base --vit5_repo "$VIT5_REPO" --vit5_ckpt "$VIT5_CKPT" \
  || { echo "selfcheck FAIL: models_extra.py 의 ViT-5 rope 접근 수정 필요"; exit 1; }

echo "=== [1] ViT-5-B  gallery=$GALLERY ==="
python run_extra.py --model vit5_base --vit5_repo "$VIT5_REPO" --vit5_ckpt "$VIT5_CKPT" \
  --gallery "$GALLERY" --r_list $RLIST \
  | tee "$RESULTS/extra_vit5_base_${GALLERY}_faithful.txt"

echo "=== [2] ViT-5-B reg-count 스윕(k=0..4) ==="
python run_extra.py --model vit5_base --vit5_repo "$VIT5_REPO" --vit5_ckpt "$VIT5_CKPT" \
  --gallery "$GALLERY" --r_list $RLIST --regsweep \
  | tee "$RESULTS/extra_vit5_base_${GALLERY}_regsweep.txt"

echo "완료. 결과: $RESULTS/extra_vit5_*"
