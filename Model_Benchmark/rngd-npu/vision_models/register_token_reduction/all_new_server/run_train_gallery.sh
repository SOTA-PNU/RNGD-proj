#!/usr/bin/env bash
# ★ [메인] train-갤러리 faithful 헤드라인: ToMe vs PiToMe vs Ours 를 정통 kNN(gallery=train 1.28M, query=val)로.
#   = "val 로 돌렸던 head-to-head 를 train 전체로" 그대로. compare_faithful.py(정식 forward 주입) 사용.
#   모델 태그를 여러 개 주면 각 모델을 GPU 0,1,2,... 에 하나씩 붙여 병렬로 돌립니다(A6000×4 활용).
# 사용:
#   bash run_train_gallery.sh            # 기본 = b (헤드라인만, GPU0)
#   bash run_train_gallery.sh s b l      # S=GPU0, B=GPU1, L=GPU2 병렬
#   SEQUENTIAL=1 bash run_train_gallery.sh s b l   # 디스크 I/O 약하면 순차 실행
set -e
# shellcheck disable=SC1091
source "$(dirname "$0")/config.sh"
cd "$ENGINE"

TAGS="${*:-b}"
declare -A MODEL=( [s]="$MODEL_S" [b]="$MODEL_B" [l]="$MODEL_L" )

run_one() {  # $1=tag $2=gpu
  local tag="$1" gpu="$2" mdl="${MODEL[$1]}" out="$RESULTS/canonical_faithful_$1.txt"
  echo "[train-gallery faithful] $tag=$mdl → GPU$gpu → $out"
  CUDA_VISIBLE_DEVICES="$gpu" python compare_faithful.py --mode acc --gallery train \
      --model "$mdl" --r_list $RLIST --workers "$WORKERS" \
      --cache_dir "$CACHE/main_$tag" --gallery_cache "$GALLERY_CACHE" > "$out" 2>&1
}

gpu=0; pids=()
for tag in $TAGS; do
  if [ -z "${MODEL[$tag]}" ]; then echo "알 수 없는 태그: $tag (s|b|l)"; exit 1; fi
  if [ "${SEQUENTIAL:-0}" = "1" ]; then
    run_one "$tag" "$gpu"
  else
    run_one "$tag" "$gpu" & pids+=($!)
  fi
  gpu=$((gpu+1))
done
if [ "${SEQUENTIAL:-0}" != "1" ]; then
  echo "병렬 실행 중(PID: ${pids[*]}). 진행: tail -f $RESULTS/canonical_faithful_*.txt"
  wait
fi
echo "=== train-gallery faithful 완료. 결과: $RESULTS/canonical_faithful_*.txt ==="
echo "    판정: 무압축(r=0)은 통제판과 같아야(병합 無). 압축구간 순서·교차점(~74%)이 val-LOO faithful 과 같은 그림이면 승격 확정."
