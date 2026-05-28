#!/usr/bin/env bash
# run_model_p.sh — furiosa-llm serve에 동시 요청 N개 쏘고 처리량 측정
#
# 사용:
#   ./run_model_p.sh                       # 기본값 (N=8, PORT=8000 등)
#   N=16 ./run_model_p.sh                  # 동시 요청 16개로
#   PORT=8001 ./run_model_p.sh             # 다른 포트
#   N=32 MAXT=256 ./run_model_p.sh         # 여러 개 동시 override
#
# 사전 조건: 다른 터미널에서 'furiosa-llm serve ... --port $PORT'가 돌고 있어야 함.
# dp 비교: 같은 스크립트를 (--devices npu:0)와 (--devices npu:0,npu:1) 서버에 각각 돌려 시간 비교.

set -uo pipefail

# ── 설정 (env var로 override 가능; MODEL은 서버에서 자동 감지) ──
N=${N:-8}                                                  # 동시 요청 수
PORT=${PORT:-8001}                                         # 서버 포트
MAXT=${MAXT:-256}                                          # max_tokens
PROMPT=${PROMPT:-"Write a Python function to reverse a string."}
OUT=${OUT:-/tmp/dptest}                                    # 응답 저장 폴더

# ── 준비 ──
mkdir -p "$OUT" && rm -f "$OUT"/resp_*.json

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
printf "  %-18s : %s\n" "N (요청 수)"   "$N"
printf "  %-18s : %s\n" "PORT"          "$PORT"
printf "  %-18s : %s\n" "MODEL"         "$MODEL"
printf "  %-18s : %s\n" "MAXT"          "$MAXT"
printf "  %-18s : %s\n" "PROMPT"        "$PROMPT"
printf "  %-18s : %s\n" "결과 저장"     "$OUT/"
echo

# ── 동시 요청 발사 ──
echo "── 요청 시작 (N=$N 동시) ──"
START_NS=$(date +%s%N)
for i in $(seq 1 "$N"); do
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAXT}" \
        -o "$OUT/resp_$i.json" &
done
wait
END_NS=$(date +%s%N)

ELAPSED=$(awk "BEGIN {printf \"%.2f\", ($END_NS - $START_NS) / 1e9}")

# ── 결과 분석 ──
SUCCESS=$(grep -l '"completion_tokens"' "$OUT"/resp_*.json 2>/dev/null | wc -l)
TOTAL_TOK=$(python3 -c "
import json, glob
total = 0
for f in sorted(glob.glob('$OUT/resp_*.json')):
    try:
        total += json.load(open(f))['usage']['completion_tokens']
    except Exception:
        pass
print(total)
")
TPS=$(awk "BEGIN {printf \"%.1f\", $TOTAL_TOK / $ELAPSED}")

AVG_S=$(awk "BEGIN {printf \"%.2f\", $ELAPSED / $N}")

echo
echo "── 결과 ──"
printf "  %-18s : %s / %s\n"  "성공 / 요청"       "$SUCCESS" "$N"
printf "  %-18s : %s s\n"     "총 소요 시간"      "$ELAPSED"
printf "  %-18s : %s s/req\n" "요청당 평균"       "$AVG_S"
printf "  %-18s : %s\n"       "총 생성 토큰"      "$TOTAL_TOK"
printf "  %-18s : %s tok/s\n" "시스템 throughput" "$TPS"
echo
echo "  전체 JSON 저장       : $OUT/resp_1.json ~ resp_$N.json"
echo "  (실패 응답엔 JSON에 'error' 필드 — 확인 권장)"
