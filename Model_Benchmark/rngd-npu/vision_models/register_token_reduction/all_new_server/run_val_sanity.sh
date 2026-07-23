#!/usr/bin/env bash
# [권장·빠름] val-LOO faithful 재현: 새 서버 환경이 A100 서버의 수치를 그대로 내는지 대조합니다(수십 분).
#   train 대장정(수 시간) 전에 이걸 먼저 돌려 "환경이 똑같다"를 확인하는 용도.
#   기대치(ViT-B, 50k, val-LOO): 무압축 76.33 / 정식 ToMe Δ+10.29@91% / PiToMe 대비 Ours 전구간 우세.
# 사용: bash run_val_sanity.sh        # 기본 GPU0
#      bash run_val_sanity.sh 1       # GPU1
set -e
# shellcheck disable=SC1091
source "$(dirname "$0")/config.sh"
cd "$ENGINE"
export CUDA_VISIBLE_DEVICES="${1:-0}"

echo "[val sanity] 정식 ToMe vs Ours"
python faithful_tome_h2h.py 50000        | tee "$RESULTS/val_faithful_tome_50k.txt"
echo "[val sanity] 정식 PiToMe vs ToMe vs Ours"
python faithful_pitome_h2h.py 50000      | tee "$RESULTS/val_faithful_pitome_50k.txt"
echo "[val sanity] ablation(register vs 대안)"
python eval_ablation_faithful.py --n 50000 --r_list $RLIST | tee "$RESULTS/val_ablation_faithful.txt"
echo "[val sanity] reg-count 스윕 + 부트스트랩 CI"
python reg_count_sweep_faithful.py 50000 | tee "$RESULTS/val_regcount_faithful.txt"
echo "[val sanity] 검색 mAP"
python retrieval_map_faithful.py 50000   | tee "$RESULTS/val_retrieval_faithful.txt"

echo "=== val-LOO faithful 재현 완료. 결과: $RESULTS/val_*.txt ==="
echo "    이 수치가 A100 결과(SESSION_STATUS.md '관문 통과')와 일치하면 새 서버 환경 OK → train-갤러리로 진행."
