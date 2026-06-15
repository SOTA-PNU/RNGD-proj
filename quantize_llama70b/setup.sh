#!/usr/bin/env bash
# setup.sh — venv 만들고 torch(CUDA) + transformers 등 설치.
set -euo pipefail
cd "$(dirname "$0")"

VENV="${VENV:-./venv}"
PYTHON="${PYTHON:-python3}"

echo "── 1) 시스템 확인 ──"
$PYTHON --version
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "❌ nvidia-smi 없음 — GPU 머신에서 실행하세요."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "── 2) venv 생성 ($VENV) ──"
[ -d "$VENV" ] || $PYTHON -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install --upgrade pip

echo
echo "── 3) CUDA 버전 감지 후 torch 설치 ──"
# nvidia-smi 의 CUDA 버전 (driver) → PyTorch wheel index 매칭
CUDA_VER=$(nvidia-smi --query | awk -F ': *' '/CUDA Version/{print $2; exit}')
echo "   driver CUDA 버전: $CUDA_VER"
# 12.x → cu121, 11.8 → cu118 (PyTorch index 표기)
case "$CUDA_VER" in
    12.*) PT_INDEX="https://download.pytorch.org/whl/cu121" ;;
    11.8) PT_INDEX="https://download.pytorch.org/whl/cu118" ;;
    *)    PT_INDEX="https://download.pytorch.org/whl/cu121"
          echo "   ⚠️  알 수 없는 CUDA 버전 — cu121 로 fallback" ;;
esac
pip install torch --index-url "$PT_INDEX"

echo
echo "── 4) 나머지 패키지 ──"
pip install -r requirements.txt

echo
echo "── 5) 검증 ──"
python - <<'EOF'
import torch
from transformers import FineGrainedFP8Config
print(f"  torch              : {torch.__version__}")
print(f"  CUDA available     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}              : {p.name} ({p.total_memory/1e9:.0f} GB)")
print("  FineGrainedFP8     : import OK")
EOF

echo
echo "✅ setup 완료 — 이제 ./run.sh 또는 './download.sh && python quantize.py meta-llama/Llama-3.3-70B-Instruct'"
