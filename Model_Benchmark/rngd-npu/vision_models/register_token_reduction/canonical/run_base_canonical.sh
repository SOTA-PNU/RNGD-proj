#!/usr/bin/env bash
# 베이스(헤드라인) ViT-B 정통 kNN 재현. GPU 서버(A100)에서 실행.
# 이 폴더(canonical/)는 "정통 train-갤러리 kNN(baseline≈82 재현)" 전용. 엔진·데이터는 ../pitome_compare 를 공유한다.
# 범위: ToMe·PiToMe·Ours 정확도(=tab:main+tab:pitome)를 정통 프로토콜로 재측정. ablation/dense/eval_v2/robustness는 포함 안 함(README 참고).
# 기본 = 공식 그대로: gallery=ImageNet train 전체 1.28M, r 5개. 측정 처리량(ViT-B ~350 img/s) 기준 단일 A100 ~17h. 캐시로 재개 가능.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ENGINE="$(cd "$HERE/../pitome_compare" && pwd)"   # 공유 엔진(compare.py)·데이터(imagenet_*, feat_cache)

GALLERY_PC=${GALLERY_PC:-0}                 # 0=train 전체 1.28M(정확한82). 빠르게=260(≈260k, ~82근접, ~4h)
RLIST=${RLIST:-8 12 16 18 20}               # 전체 곡선. 축소=  "12 16 20"
MODEL=vit_base_patch14_reg4_dinov2.lvd142m
GC=${GC:-1}                                 # 1=갤러리 특징 캐시(재개·설정당~2GB) / 0=디스크 절약
SUF=${SUF:-}                                # 결과파일 접미사(2-GPU 분할시 _g0/_g1 로 충돌 방지)
TPUT=${TPUT:-1}                             # 1=처리량/지연 측정 / 0=생략(분할시 한쪽만 1)

if [ "$TPUT" = 1 ]; then
  echo "=== [1/3] 처리량/지연 (합성 배치·데이터 무관·수초) ==="
  # throughput=im/s(배치128), latency=배치1 ms/img. 프로토콜 무관(모델·압축률에만 의존).
  python "$ENGINE/compare.py" --mode tput --model "$MODEL" --batch 128 --r_list 0 8 12 16 18 20 | tee "$HERE/results_base_tput.txt"
  python "$ENGINE/compare.py" --mode tput --model "$MODEL" --batch 1   --r_list 0 8 12 16 18 20 | tee "$HERE/results_base_latency.txt"
fi

echo "=== [2/3] 데이터(../pitome_compare 공유): val + train(per_class=$GALLERY_PC) ==="
[ -f "$ENGINE/imagenet_val/DONE" ]   || python "$ENGINE/prepare_data.py" --split val
[ -f "$ENGINE/imagenet_train/DONE" ] || { echo "[train 다운로드 $(date +%T)]"; python "$ENGINE/prepare_data.py" --split train --per_class "$GALLERY_PC"; }

echo "=== [3/3] 정통 kNN (ViT-B, gallery=train, r=$RLIST) — 결과 canonical/results_base_canonical${SUF}.txt ==="
echo "[START $(date +%T)]"; SECONDS=0
python "$ENGINE/compare.py" --mode acc --gallery train --model "$MODEL" --r_list $RLIST --gallery_cache $GC \
    | tee "$HERE/results_base_canonical${SUF}.txt"
echo "[END $(date +%T)]  총 = ${SECONDS}s ($((SECONDS/60))분)"
echo " r=0 이 ~82 근처면 정통 재현 성공. Ours>PiToMe(극단)·Ours>ToMe 확인. (latency ms/img = 1000/im/s@batch1)"

# [2-GPU 반토막 ~10h] r 나눠 동시 실행(결과는 _g0/_g1, tput은 한쪽만). 끝나면 두 txt 합쳐 보면 됨:
#   CUDA_VISIBLE_DEVICES=0 RLIST="8 12 16" SUF=_g0        bash run_base_canonical.sh > base_g0.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=1 RLIST="18 20"   SUF=_g1 TPUT=0 bash run_base_canonical.sh > base_g1.log 2>&1 &
#   (양쪽 다 r=0 baseline 포함 → 82 재현을 상호 교차검증. feat_cache 공유로 중복 추출 자동 회피.)
