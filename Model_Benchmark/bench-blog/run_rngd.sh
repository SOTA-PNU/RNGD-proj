#!/usr/bin/env bash
# RNGD 측 블로그 재현 — qwen3-32b-fp8-tp8(1장=tp8)을 빈 카드에 serve하고 loadgen으로 측정.
# 결과: results/rngd.json  (compare.py 에서 pro6000.json 과 합쳐 리포트)
#
# 사용:
#   ./run_rngd.sh                       # 기본: 빈 카드 자동, ISL1024/OSL256, b1..256
#   CARD=3 ./run_rngd.sh                # npu3 에 serve
#   ISL=2048 OSL=512 BATCHES=1,8,32,128 ./run_rngd.sh
#
# 전제: source ~/furiosa/bin/activate 로 furiosa-llm 사용 가능, 아티팩트 존재.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ART="${ART:-$HOME/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen3-32b-fp8-tp8}"
PORT="${PORT:-8104}"
ISL="${ISL:-1024}"; OSL="${OSL:-256}"
BATCHES="${BATCHES:-1,8,16,32,64,256}"
WINDOW="${WINDOW:-30}"; WARMUP="${WARMUP:-8}"
MAXCONC="${MAXCONC:-256}"          # furiosa-llm --max-concurrency (동시 decode 요청 상한)
OUT="${OUT:-$HERE/results/rngd.json}"
LOG="${LOG:-$HERE/results/rngd_serve.log}"
mkdir -p "$HERE/results"

source ~/furiosa/bin/activate 2>/dev/null || { echo "furiosa venv 활성화 실패"; exit 1; }
command -v furiosa-llm >/dev/null || { echo "furiosa-llm 없음"; exit 1; }
[ -f "$ART/artifact.json" ] || { echo "아티팩트 없음: $ART"; exit 1; }

# 빈 카드 고르기 (메모리 0 인 카드). CARD 지정 시 그걸 사용. (mawk 호환)
pick_free_card() {
  furiosa-smi status 2>/dev/null | awk -F'|' '
    /npu[0-9]/ {
      dev=$3; gsub(/[^0-9]/,"",dev);
      split($5, a, "/"); used=a[1]+0;
      if (used < 1.0) { print dev; exit }
    }'
}
CARD="${CARD:-$(pick_free_card)}"
[ -z "${CARD:-}" ] && { echo "빈 카드 없음. 다른 serve를 내리거나 CARD=N 지정."; exit 1; }
echo "▶ RNGD serve: $(basename "$ART") on npu:$CARD  port=$PORT  (max-concurrency=$MAXCONC)"

# serve 기동 (단일 카드 = tp8). prefix-caching 은 켜되 loadgen 이 요청마다 고유 prefix 라 캐시히트 안 남.
nohup furiosa-llm serve "$ART" \
  --devices "npu:$CARD" --host 0.0.0.0 --port "$PORT" \
  --max-concurrency "$MAXCONC" --reasoning-parser qwen3 \
  > "$LOG" 2>&1 &
SERVE_PID=$!
cleanup() { echo "▷ serve 종료(pid $SERVE_PID)"; kill "$SERVE_PID" 2>/dev/null; wait "$SERVE_PID" 2>/dev/null; }
trap cleanup EXIT

# health 대기 (최대 240초)
echo -n "  서버 준비 대기"
for i in $(seq 1 240); do
  code=$(curl -s -m3 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
  [ "$code" = "200" ] && { echo " OK"; break; }
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then echo; echo "!! serve 프로세스 종료됨 — 로그:"; tail -20 "$LOG"; exit 1; fi
  echo -n "."; sleep 1
done
[ "$code" = "200" ] || { echo; echo "!! 240초 내 준비 실패 — 로그:"; tail -20 "$LOG"; exit 1; }

MODEL=$(curl -s -m5 "http://127.0.0.1:$PORT/v1/models" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
echo "  모델 id: $MODEL"

# loadgen 실행 (전력 = 이 카드만 샘플링)
python3 "$HERE/loadgen.py" \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
  --platform rngd --label "qwen3-32b-fp8 (1card/tp8, weight-fp8)" \
  --isl "$ISL" --osl "$OSL" --batches "$BATCHES" \
  --window "$WINDOW" --warmup "$WARMUP" --endpoint completions \
  --tokenizer "$ART" \
  --power rngd --power-devices "$CARD" \
  --out "$OUT"

echo "✅ RNGD 결과: $OUT"
