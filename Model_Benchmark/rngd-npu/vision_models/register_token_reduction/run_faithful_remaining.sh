#!/usr/bin/env bash
# 지금 돌리는 faithful_pitome·eval_ablation_faithful 가 끝나 GPU 가 비면 자동으로 이어서
# 남은 faithful 실험(retrieval mAP · dense)을 순차 실행한다.
# 감지 방식: nvidia-smi 로 memory.used < 임계값인 GPU 가 생기면 그 GPU 에서 실행(로그 의존 X).
# 사용:  nohup bash run_faithful_remaining.sh > faithful_remaining.log 2>&1 &
set -e
cd "$(dirname "$0")"
THRESH_MB=${THRESH_MB:-3000}     # 이 이하면 '빈 GPU'로 간주
echo "[$(date '+%H:%M:%S')] 빈 GPU 대기 중(현재 faithful_pitome·ablation 종료 대기)... 임계 ${THRESH_MB}MB"

wait_free_gpu() {
  while true; do
    FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
           | awk -F', ' -v t="$THRESH_MB" '$2+0 < t {print $1; exit}')
    [ -n "$FREE" ] && { echo "$FREE"; return; }
    sleep 30
  done
}

G=$(wait_free_gpu)
echo "[$(date '+%H:%M:%S')] 빈 GPU=$G → 남은 faithful 실행 시작"
export CUDA_VISIBLE_DEVICES=$G

echo "==> [1/2] faithful 검색 mAP (robustness_50k, 50k)"
( cd robustness_50k && python retrieval_map_faithful.py 50000 | tee retrieval_map_faithful_50k.log )

echo "==> [2/2] faithful dense (ADE20k, 전량 train)"
( cd dense && N_TRAIN=20210 bash run_faithful.sh )

echo "[$(date '+%H:%M:%S')] ✅ 남은 faithful 완료. 이제 faithful 핵심표 5개 전부 준비됨:"
echo "   main(faithful_tome_50k.log)·pitome(faithful_pitome_50k.log)·ablation(ablation/results/ablation_FAITHFUL_*.json)"
echo "   ·retrieval(robustness_50k/retrieval_map_faithful_50k.log)·dense(dense/results/dense_miou_FAITHFUL_*.json)"
