#!/usr/bin/env bash
# 정통 kNN을 '온도가중 투표(DINOv2 공식식)'로 재채점 — GPU 서버에서 실행.
# ★기존 코드(compare.py·run_base_canonical.sh)는 전혀 안 건드림. 이 폴더만 독립.
# ★캐시(../../pitome_compare/feat_cache/*.pt)가 남아 있으면 재추출 없이 ~수초. 없으면 새로 추출(느림).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

MODEL=${MODEL:-vit_base_patch14_reg4_dinov2.lvd142m}
RLIST=${RLIST:-8 12 16 18 20}         # baseline(r=0)만 빠르게: RLIST="" (아래서 r=0 은 항상 포함)
K=${K:-20}
TEMP=${TEMP:-0.07}

echo "[START $(date +%T)]  weighted-kNN 재채점 (majority vs weighted 동시 출력)"
python "$HERE/weighted_knn.py" --model "$MODEL" --k "$K" --temp "$TEMP" --r_list $RLIST \
    | tee "$HERE/results_weighted_knn.txt"
echo "[END $(date +%T)]"
echo " r=0 의 weighted 열이 ~82 면 '격차=투표방식' 확인. 순서 유지되면 결론 불변 → tab:canonical weighted 로 갱신 검토."
