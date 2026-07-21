#!/usr/bin/env bash
# [새 서버] 전체 파이프라인을 순서대로 안내·실행합니다. 환경설치(setup_env.sh)만 먼저 끝내 두세요.
#   0) setup_env.sh 로 .venv 만들고 activate 되어 있어야 함(torch/timm 설치).
#   1) 데이터 준비(val 5만 + train 1.28M). train 은 수십 GB·수 시간.
#   2) 모델 가중치 워밍업(온라인 1회).
#   3) (권장) val-LOO faithful 재현 — 환경 대조(수십 분).
#   4) ★ train-갤러리 faithful 헤드라인 S/B/L 병렬(GPU0/1/2).
#   5) (선택) train-갤러리 ablation+reg-count (GPU3).
# 사용: bash run_all.sh
set -e
# shellcheck disable=SC1091
source "$(dirname "$0")/config.sh"

echo "### 1) 데이터 준비 (없으면 다운로드; train 은 수십 GB·수 시간) ###"
python "$BUNDLE/prepare_data.py" --split val
python "$BUNDLE/prepare_data.py" --split train --per_class 1300

echo "### 2) 모델 가중치 워밍업 (온라인 1회) ###"
python "$BUNDLE/warmup_models.py"

echo "### 3) val-LOO faithful 재현 (환경 대조, 수십 분) ###"
bash "$BUNDLE/run_val_sanity.sh" 0

echo "### 4) train-갤러리 faithful 헤드라인 (S/B/L 병렬, GPU0/1/2) ###"
bash "$BUNDLE/run_train_gallery.sh" s b l

echo "### 5) (선택) train-갤러리 ablation + reg-count (GPU3, B) ###"
bash "$BUNDLE/run_ablation_regcount.sh" b 3

echo "### 전부 완료. 결과: $RESULTS/ ###"
