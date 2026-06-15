#!/usr/bin/env bash
# run_model.sh — furiosa-llm serve에 단일 요청 1개 보내고 응답·시간 측정
#
# 사용:
#   ./run_model.sh                         # 기본값으로
#   PORT=8000 ./run_model.sh               # 다른 포트
#   PROMPT="Hello" ./run_model.sh          # 다른 prompt
#   MAXT=512 ./run_model.sh                # 더 긴 응답
#
# 사전 조건: 다른 터미널에서 'furiosa-llm serve ... --port $PORT'가 돌고 있어야 함.
# 동시 요청 N개로 throughput 측정 → run_model_p.sh 사용.

set -uo pipefail

# ── 설정 (env var로 override 가능; MODEL은 서버에서 자동 감지) ──
PORT=${PORT:-8001}
MAXT=${MAXT:-256}
PROMPT=${PROMPT:-"Write a Python function to reverse a string."}
OUT=${OUT:-/tmp/run_model.json}

# ── 서버 헬스 체크 ──
if ! curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    echo "❌ 127.0.0.1:$PORT 서버에 못 붙음."
    echo "   먼저 다른 터미널에서 'furiosa-llm serve <artifact> --devices npu:0 --port $PORT' 띄우세요."
    exit 1
fi

# ── MODEL 자동 감지 (env var 우선) ──
if [ -z "${MODEL:-}" ]; then
    MODEL=$(curl -sf "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import json, sys
try: print(json.load(sys.stdin)['data'][0]['id'])
except Exception: pass
")
fi
if [ -z "$MODEL" ]; then
    echo "❌ MODEL 자동 감지 실패. MODEL=... 로 직접 지정하세요."
    exit 1
fi

echo "── 설정 ──"
printf "  %-18s : %s\n" "N (요청 수)"   "1"
printf "  %-18s : %s\n" "PORT"          "$PORT"
printf "  %-18s : %s\n" "MODEL"         "$MODEL"
printf "  %-18s : %s\n" "MAXT"          "$MAXT"
printf "  %-18s : %s\n" "PROMPT"        "$PROMPT"
printf "  %-18s : %s\n" "결과 저장"     "$OUT"
echo

# ── 요청 1개 발사 ──
echo "── 요청 시작 (N=1) ──"
START_NS=$(date +%s%N)
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAXT}" \
    -o "$OUT"
END_NS=$(date +%s%N)
ELAPSED=$(awk "BEGIN {printf \"%.2f\", ($END_NS - $START_NS) / 1e9}")

# ── 결과 분석 ──
TOK=$(python3 -c "
import json
try:
    a = json.load(open('$OUT'))
    print(a['usage']['completion_tokens'])
except Exception:
    print(0)
")
TPS=$(awk "BEGIN { if ($ELAPSED > 0) printf \"%.1f\", $TOK / $ELAPSED; else printf \"-\" }")
FINISH=$(python3 -c "
import json
try:
    print(json.load(open('$OUT'))['choices'][0].get('finish_reason','?'))
except Exception:
    print('?')
")

SUCCESS=$(awk "BEGIN { print ($TOK > 0) ? 1 : 0 }")

echo
echo "── 결과 ──"
printf "  %-18s : %s / %s\n"  "성공 / 요청"       "$SUCCESS" "1"
printf "  %-18s : %s s\n"     "총 소요 시간"      "$ELAPSED"
printf "  %-18s : %s\n"       "총 생성 토큰"      "$TOK"
printf "  %-18s : %s tok/s\n" "시스템 throughput" "$TPS"
printf "  %-18s : %s\n"       "finish_reason"     "$FINISH"
echo

# ── 응답 내용 출력 ──
echo "── 응답 내용 ──"
python3 -c "
import json
try:
    a = json.load(open('$OUT'))
    msg = a['choices'][0]['message']
    if msg.get('reasoning'):
        print('[reasoning]')
        print(msg['reasoning'])
        print()
    print('[content]')
    print(msg.get('content',''))
except Exception as e:
    print(f'(파싱 실패: {e})')
    print()
    print(open('$OUT').read())
"
echo
echo "  전체 JSON 저장       : $OUT"
