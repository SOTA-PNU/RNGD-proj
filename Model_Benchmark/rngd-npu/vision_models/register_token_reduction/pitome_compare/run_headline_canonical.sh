#!/usr/bin/env bash
# 헤드라인(ViT-B) 정통 kNN(train 갤러리) 타이밍 프로브. GPU 서버에서 실행.
# 목적: (1) 무압축 baseline ≈ 82.0 재현(프로토콜 정당성 앵커) + (2) 극단압축(92%) 정통 head-to-head
#       를 최소 비용으로 먼저 확인하고, 실측 시간을 보고 전체(S/L·전 r) 확대를 결정한다.
# 비용 급소: train 갤러리 특징추출은 (전략×r)마다 1회. 아래 프로브는 r=0(1회)+r=20(3전략)=train 4회로 최소화.
set -e
cd "$(dirname "$0")"

# 갤러리 크기: 기본 260/class(≈260k, ~82 근접·빠른 1차 타이밍용). 공식 82.0 정확 재현은 GALLERY_PC=0 (train 전체 1.28M).
GALLERY_PC=${GALLERY_PC:-260}
MODEL=${MODEL:-vit_base_patch14_reg4_dinov2.lvd142m}
RLIST=${RLIST:-20}                       # 극단점만. 전체 곡선은 "8 12 16 18 20"

echo "=== [0/3] 의존성 ==="
pip install -q -r requirements.txt

echo "=== [1/3] 데이터: val(query) + train(gallery, per_class=$GALLERY_PC) ==="
[ -f imagenet_val/DONE ]   || python prepare_data.py --split val
[ -f imagenet_train/DONE ] || { echo "[train 다운로드 시작 $(date +%T)]"; \
    /usr/bin/time -v python prepare_data.py --split train --per_class "$GALLERY_PC" 2>&1 | tail -3; }

echo "=== [2/3] 정통 kNN (gallery=train, ViT-B, r=$RLIST + r=0 baseline) — 전체 벽시계 측정 ==="
echo "[START $(date +%T)]"
SECONDS=0
python compare.py --mode acc --gallery train --model "$MODEL" --r_list $RLIST | tee results_acc_canonical.txt
echo "[END $(date +%T)]  총 소요 = ${SECONDS}s (= $((SECONDS/60))분)"

echo "=== [3/3] 판단 근거 ==="
echo " - 로그의 [timing] 줄에서 train 추출 img/s 와 kNN 초를 확인."
echo " - 전체 확대 비용 외삽:"
echo "     · r 5개(8 12 16 18 20)로 늘리면 train 추출 = 1 + 3×5 = 16회 (이 프로브는 4회)."
echo "     · ViT-S 는 더 빠르고, ViT-L(24블록)은 더 느림(대략 2~3×)."
echo " - r=0 이 ~82 근처면 정통 프로토콜 재현 성공 = baseline 논란 종결."
echo " - 전체 실행: RLIST=\"8 12 16 18 20\" bash run_headline_canonical.sh (필요시 MODEL 로 S/L 교체)"
