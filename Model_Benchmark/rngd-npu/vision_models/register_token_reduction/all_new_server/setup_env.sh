#!/usr/bin/env bash
# [새 서버] Python 가상환경 + 의존성 설치 + 설치 검증. 서버당 한 번만 실행하면 됩니다.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PYBIN="${PYBIN:-python3}"
VENV="${VENV:-$HERE/.venv}"

echo "=== 1) 가상환경 만들기: $VENV ==="
"$PYBIN" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip

echo "=== 2) 의존성 설치 ==="
pip install -r "$HERE/requirements.txt"

echo "=== 3) 설치 검증 ==="
python - <<'PY'
import torch, timm, datasets, PIL
print("torch", torch.__version__, "| cuda avail", torch.cuda.is_available(), "| #gpu", torch.cuda.device_count())
print("timm", timm.__version__, "| datasets", datasets.__version__)
assert timm.__version__ >= "1.0", "timm>=1.0 필요(DINOv2-reg 로드)"
if not torch.cuda.is_available():
    print("⚠️ CUDA 미인식: 서버 CUDA 에 맞는 torch 휠을 pytorch.org 에서 재설치하세요 "
          "(예: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121)")
PY

echo ""
echo "=== 설치 완료 ==="
echo "다음: source $VENV/bin/activate  후  bash run_all.sh (또는 README 순서대로)"
