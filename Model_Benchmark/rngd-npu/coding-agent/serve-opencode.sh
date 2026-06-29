#!/usr/bin/env bash
# OpenCode 용 tool-capable furiosa-llm serve 기동 스크립트
# ---------------------------------------------------------------------------
# OpenCode 는 함수호출(tool calling) 기반 에이전트라, furiosa-llm serve 가
# --enable-auto-tool-choice --tool-call-parser 로 떠 있어야 합니다.
# 이 플래그 없이 뜬 serve 는 tool 요청을 HTTP 400 으로 거부합니다.
#
# 기본(벤더 README 권장 구성):
#   - 모델 : Qwen3-32B-FP8 (로컬 prebuilt 아티팩트 qwen3-32b-fp8-tp8, tp8 = 카드 1장)
#   - id   : --served-model-name furiosa-ai/Qwen3-32B-FP8
#            → 모델 id 가 벤더 기본값과 같아져서 opencode.sh 가 무수정 동작
#   - 파서 : --tool-call-parser hermes  --reasoning-parser qwen3   (Qwen3 정답)
#   - 카드 : npu:0  (1장)  ·  포트 :8000
#
# 다른 모델로 띄우려면 환경변수로 덮어쓰기. 예) 코딩특화 Qwen3-Coder 코더:
#   FURIOSA_ART=~/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen3-coder-30b-a3b-inst-fp8-tp8-65k-tc \
#   FURIOSA_NAME=furiosa-ai/Qwen3-Coder-30B-A3B-FP8 FURIOSA_PP=2 FURIOSA_DEVICES=npu:0,npu:1 \
#   bash serve-opencode.sh
#   (이때 opencode.json 의 모델 id 도 FURIOSA_NAME 과 맞춰야 함)
# ---------------------------------------------------------------------------
set -euo pipefail
source ~/furiosa/bin/activate 2>/dev/null || true

ART="${FURIOSA_ART:-/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen3-32b-fp8-tp8}"
NAME="${FURIOSA_NAME:-furiosa-ai/Qwen3-32B-FP8}"
DEVICES="${FURIOSA_DEVICES:-npu:0}"
PORT="${FURIOSA_PORT:-8000}"
PP="${FURIOSA_PP:-1}"
# 파서: 모델 계열에 맞게. tool 파서는 {hermes,llama4_json,llama3_json,openai} 중 하나.
TOOLPARSER="${FURIOSA_TOOLPARSER:-hermes}"          # Qwen 계열=hermes, Llama=llama3_json
REASONING="${FURIOSA_REASONING-qwen3}"             # thinking 모델만. FURIOSA_REASONING= 로 비우면 미사용(예: Qwen2.5-Coder)
LOG=~/RNGD-proj/Model_Benchmark/rngd-npu/chat/serve_logs/${PORT}.log

echo "[..] :$PORT 기존 serve 종료(있으면)"
pkill -f "furiosa-llm serve .*--port $PORT" 2>/dev/null || true
sleep 3

PP_ARG=(); [ "$PP" -gt 1 ] && PP_ARG=(-pp "$PP")
REASON_ARG=(); [ -n "$REASONING" ] && REASON_ARG=(--reasoning-parser "$REASONING")
echo "[..] tool-capable serve 기동: id=$NAME  카드=$DEVICES  pp=$PP  tool=$TOOLPARSER  reasoning=${REASONING:-none}"
echo "     로그: $LOG  (준비되면 'Uvicorn running on http://0.0.0.0:$PORT' 출력)"
: > "$LOG"
nohup furiosa-llm serve "$ART" \
  --served-model-name "$NAME" \
  --devices "$DEVICES" --host 0.0.0.0 --port "$PORT" \
  --enable-prefix-caching "${PP_ARG[@]}" \
  --enable-auto-tool-choice --tool-call-parser "$TOOLPARSER" "${REASON_ARG[@]}" \
  >> "$LOG" 2>&1 &
echo "[ok] pid=$!  ·  tail -f $LOG  로 준비 상태 확인"
