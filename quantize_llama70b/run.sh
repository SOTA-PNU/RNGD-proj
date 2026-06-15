#!/usr/bin/env bash
# run.sh — 환경 구축 → 원본 다운로드 → FP8 양자화 한 번에.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:-meta-llama/Llama-3.3-70B-Instruct}"

echo "════════════════════════════════════════════════════════════════"
echo "  Llama-3.3-70B FP8 양자화 파이프라인"
echo "  모델: $MODEL"
echo "════════════════════════════════════════════════════════════════"
echo

bash setup.sh
echo
bash download.sh "$MODEL"
echo
source ./venv/bin/activate
python quantize.py "$MODEL"

echo
echo "✅ 전체 파이프라인 완료. 결과: ./out/"
