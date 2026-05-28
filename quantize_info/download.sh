#!/usr/bin/env bash
# download.sh — 원본 BF16 가중치 다운로드 (safetensors 만, original/*.pth 제외).
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:-meta-llama/Llama-3.3-70B-Instruct}"

# venv 활성화 (있을 때만)
[ -f ./venv/bin/activate ] && source ./venv/bin/activate

# meta-llama 는 gated — 토큰 필요
if ! hf auth whoami >/dev/null 2>&1; then
    echo "⚠️  HF 로그인 필요 — meta-llama/* 는 gated 모델."
    echo "    'hf auth login' 으로 토큰 등록 후 다시 시도."
    echo "    토큰 발급: https://huggingface.co/settings/tokens"
    exit 1
fi

echo "── 다운로드: $MODEL (safetensors 만, ~141 GB) ──"
hf download "$MODEL" --exclude "original/*"
echo
echo "✅ 다운로드 완료. 캐시: ~/.cache/huggingface/hub/"
