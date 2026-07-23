#!/usr/bin/env bash
# [새 서버 공통 설정] 모든 실행 스크립트가 맨 처음 이 파일을 source 합니다.
# 데이터 위치·모델 목록·환경변수를 여기 한 곳에서 관리합니다. 바꿀 건 DATA_ROOT 정도면 됩니다.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BUNDLE="$HERE"
export ENGINE="$HERE/engine"

# ── 데이터 루트(용량 큼: val ~7GB, train 1.28M ~수십GB). 큰 디스크가 따로면 그 경로로 바꾸세요. ──
export DATA_ROOT="${DATA_ROOT:-$HERE/data}"
export IMAGENET_VAL="$DATA_ROOT/imagenet_val"       # tome_core.py·eval_ablation_faithful.py 가 이 env 를 읽음
export IMAGENET_TRAIN="$DATA_ROOT/imagenet_train"
export RESULTS="${RESULTS:-$HERE/results}"
export CACHE="${CACHE:-$DATA_ROOT/feat_cache}"       # 특징 캐시(설정당 수 GB, 재개 가능) → 큰 디스크에

# ── 새 서버엔 가중치 캐시가 없으므로 오프라인 강제를 해제(첫 실행에 timm 가중치 다운로드 허용). ──
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# ── 모델(DINOv2-reg S/B/L). 헤드라인 = B. ──
export MODEL_S="vit_small_patch14_reg4_dinov2.lvd142m"
export MODEL_B="vit_base_patch14_reg4_dinov2.lvd142m"
export MODEL_L="vit_large_patch14_reg4_dinov2.lvd142m"

export RLIST="${RLIST:-8 12 16 18 20}"               # 블록당 제거 토큰 수(압축률). 논문 표와 동일.
export WORKERS="${WORKERS:-8}"                        # DataLoader 워커(NVMe면 8~16)
# train 갤러리 특징 캐시: 1=디스크에 저장(중단 후 재개 가능, 단 설정당 수 GB → 다 켜면 수백 GB).
#                         0=저장 안 함(디스크 크게 절약, 재개 시 재추출). 디스크 빠듯하면 0.
export GALLERY_CACHE="${GALLERY_CACHE:-1}"

mkdir -p "$DATA_ROOT" "$RESULTS" "$CACHE"
