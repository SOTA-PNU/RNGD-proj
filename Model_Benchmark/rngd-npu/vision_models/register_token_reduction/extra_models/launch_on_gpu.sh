#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────────
# 사용자가 로컬에서 한 번 실행 → GPU 서버로 번들 전송 + 전체 실험 자동 실행.
# 서버 비밀번호는 scp/ssh 가 각각 물어봅니다(사용자만 알고 있으므로 여기 저장 안 함).
#   실행:  bash launch_on_gpu.sh
#   (또는 채팅창에서:  ! bash Model_Benchmark/.../extra_models/launch_on_gpu.sh )
# 어댑터는 로컬 CPU 에서 selfcheck cosine=1.0 로 이미 검증됨. 서버에선 selfcheck 재확인 후 실험.
# ───────────────────────────────────────────────────────────────────────────────
set -e
SRV="${SRV:-jun@164.125.249.13}"; PORT="${PORT:-10022}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "[1/2] 번들 전송 → $SRV:~/extra_models  (비번 입력)"
scp -P "$PORT" -r "$HERE" "$SRV:~/extra_models"

echo "[2/2] 원격 GPU 서버에서 실험 실행  (비번 입력)"
ssh -p "$PORT" "$SRV" 'bash -s' <<'REMOTE'
set -e
cd ~/extra_models
[ -d .venv ] || bash setup_env.sh
source .venv/bin/activate
# 빠른 실패: 데이터(수 시간 다운로드) 전에 어댑터 정확성 게이트부터(랜덤 텐서, 데이터 불필요)
echo "===== selfcheck (데이터 다운로드 전 어댑터 검증) ====="
python selfcheck.py --model dinov3_base   || { echo "selfcheck FAIL"; exit 1; }
python selfcheck.py --model dinov3_splus  || { echo "selfcheck FAIL"; exit 1; }
# 데이터(용량 큼): val 50k + train 1.28M. 큰 디스크가 따로면 먼저  export DATA_ROOT=/big/disk
python prepare_data.py --split val
python prepare_data.py --split train --per_class 1300
echo "===== DINOv3-S+/B ====="
bash run_dinov3.sh
echo "===== ViT-5-B (공식 repo·ckpt 준비 후 실행) ====="
[ -d ~/ViT-5 ] || git clone https://github.com/wangf3014/ViT-5 ~/ViT-5
[ -f ~/vit5_ckpt/vit5_base_patch16_224.pth ] || huggingface-cli download FengWang3211/ViT-5 vit5_base_patch16_224.pth --local-dir ~/vit5_ckpt
export VIT5_REPO=~/ViT-5 VIT5_CKPT=~/vit5_ckpt/vit5_base_patch16_224.pth
bash run_vit5.sh
echo "완료. 결과: ~/extra_models/results/"
REMOTE

echo ""
echo "결과 회수:  scp -P $PORT -r $SRV:~/extra_models/results ./extra_models_results"
