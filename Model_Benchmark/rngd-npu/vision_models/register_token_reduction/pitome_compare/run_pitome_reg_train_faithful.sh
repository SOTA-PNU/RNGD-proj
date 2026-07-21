#!/usr/bin/env bash
# [일반성 검증 · train 1.28M] 레지스터 보호가 PiToMe 병합 위에서도 이득인가 = 병합기 무관?
# GPU 서버(A100)에서 실행. 데이터 = pitome_compare/imagenet_train(1.28M)·imagenet_val(5만).
# ★원본 val 스크립트(robustness_50k/faithful_pitome_reg_h2h.py)는 안 건드림 — forward 만 import 재사용.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

MODEL=${MODEL:-vit_base_patch14_reg4_dinov2.lvd142m}
RLIST=${RLIST:-8 12 16 18 20}
STRATS=${STRATS:-tome pitome pitome_reg ours}   # 4-arm. tome/pitome/ours 는 canonical 캐시 히트라 무료 → 사실상 pitome_reg만 추출
CACHE=${CACHE:-$HERE/feat_cache_faithful}       # ★canonical faithful 캐시 재사용(신규 추출=pitome_reg뿐, ~20h→~5h)

echo "[START $(date +%T)] pitome_reg 일반성 (train 1.28M, faithful) · 캐시=$CACHE"
python "$HERE/pitome_reg_train_faithful.py" --gallery train --model "$MODEL" --r_list $RLIST --strats $STRATS --cache_dir "$CACHE" \
    | tee "$HERE/results_pitome_reg_train_faithful.txt"
echo "[END $(date +%T)]"
echo " reg@PiTo 열이 전부 >0 이면 = 레지스터 보호가 PiToMe 병합에서도 이득(병합기 무관) 확정."
echo " tome/ours 는 canonical/faithful_results/canonical_faithful_base.txt(train) 와 일치해야 함(엔진 정합)."
