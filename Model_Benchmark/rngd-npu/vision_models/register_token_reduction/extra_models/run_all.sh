#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# GPU 서버 올인원 실행기 — 이 폴더에서 `bash run_all.sh` 하나면 전부 실행됩니다.
#   환경설치 → (어댑터 selfcheck) → 데이터 준비(val 50k + train 1.28M) → DINOv3-S+/B → ViT-5-B
# 전제: 이 폴더가 이미 GPU 서버에 있고(리눅스 + NVIDIA GPU), 인터넷 접속 가능.
# 결과: results/extra_{dinov3_base,dinov3_splus,vit5_base}_train_faithful.txt (+ *_regsweep.txt)
# 개별 단계만 돌리려면 README '수동 실행' 참고(run_dinov3.sh / run_vit5.sh).
# ═══════════════════════════════════════════════════════════════════════════════
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"

echo "═══ [0/5] 환경 점검 ═══"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "⚠️  nvidia-smi 없음 — 이 스크립트는 NVIDIA GPU 서버용입니다. 중단."; exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head

echo "═══ [1/5] 파이썬 환경 (.venv) ═══"
if [ ! -d .venv ]; then
  bash setup_env.sh
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "═══ [2/5] 어댑터 정확성 게이트 (데이터 다운로드 前, 랜덤텐서·수초) ═══"
python selfcheck.py --model dinov3_base   || { echo "❌ selfcheck FAIL (dinov3_base)"; exit 1; }
python selfcheck.py --model dinov3_splus  || { echo "❌ selfcheck FAIL (dinov3_splus)"; exit 1; }
echo "✅ DINOv3 어댑터 r=0 == 공식 forward (cosine≈1.0)"

echo "═══ [3/5] 데이터 준비 (val 50k + train 1.28M, 수십 GB·수 시간) ═══"
echo "    (큰 디스크를 쓰려면 이 스크립트 중단하고  export DATA_ROOT=/큰디스크/경로  후 다시 실행)"
python prepare_data.py --split val
python prepare_data.py --split train --per_class 1300

echo "═══ [4/5] DINOv3-S+/B 실험 (train 갤러리 kNN) ═══"
bash run_dinov3.sh

echo "═══ [5/5] ViT-5-B 실험 (공식 repo·체크포인트 자동 준비) ═══"
: "${VIT5_REPO:=$HOME/ViT-5}"
: "${VIT5_CKPT:=$HOME/vit5_ckpt/vit5_base_patch16_224.pth}"
export VIT5_REPO VIT5_CKPT
[ -d "$VIT5_REPO" ]  || git clone https://github.com/wangf3014/ViT-5 "$VIT5_REPO"
[ -f "$VIT5_CKPT" ]  || huggingface-cli download FengWang3211/ViT-5 vit5_base_patch16_224.pth --local-dir "$(dirname "$VIT5_CKPT")"
bash run_vit5.sh

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ 전부 완료. 결과 파일:"
ls -1 results/extra_* 2>/dev/null
echo "이 results/ 파일들을 회수해서 Claude 에게 주면 논문 §일반성에 반영합니다."
echo "═══════════════════════════════════════════════════════════════"
