#!/usr/bin/env bash
# RNGD NPU 채팅용 모델 서버 — 벤치한 9개 모델을 카드 예산(npu:0~3, 4장) 안에서 serve.
# chat_app.py 의 MODELS 포트와 일치해야 한다.
#
# 카드 예산: 모델 1개 = tp8이면 카드 1장, tp32면 카드 4장 전부. 그래서 동시에는
#   - tp8 모델 최대 4개 (npu:0~3에 하나씩)
#   - 또는 tp32 모델 1개 (4장 독점 — tp8과 같이 못 띄움)
#
# 사용:
#   ./serve_models.sh                    # 기본: 3종(coder7·coder14·qwen3-32b, tp8)을 빈 카드에 동시 serve
#   ./serve_models.sh 2                  # 기본 세트에서 가벼운 N개만
#   ./serve_models.sh coder7 coder14     # 고른 tp8 모델만 (빈 카드에 자동 배정)
#   ./serve_models.sh exaone-32b         # tp32 대형 모델 1개 (4장 전부)
#   ./serve_models.sh list               # 등록된 모델 키 보기
#   ./serve_models.sh stop               # 전부 종료
#
# 로그: chat/serve_logs/<port>.log  ·  서버 준비되면 "Uvicorn running" 출력됨.
set -u
A=~/RNGD-proj/Model_Benchmark/rngd-npu/artifacts
LOGDIR=~/RNGD-proj/Model_Benchmark/rngd-npu/chat/serve_logs
mkdir -p "$LOGDIR"
source ~/furiosa/bin/activate 2>/dev/null

# 모델 카탈로그:  키 = "포트|tp|아티팩트경로|추가 serve 인자"
#   tp = 8(카드 1장) 또는 32(카드 4장 전부). 포트는 chat_app.py 의 MODELS 와 일치해야 함.
# coder1.5(Qwen2.5-Coder-1.5B)는 furiosa-llm 2026.2.0이 출력 깨지게 컴파일해 제외(info/README_build.md 8.2).
declare -A CAT=(
  [coder7]="8002|8|$A/qwen2.5-coder-7b-inst-tp8|"
  [coder14]="8003|8|$A/qwen2.5-coder-14b-inst-tp8|"
  [qwen3-32b]="8004|8|$A/qwen3-32b-fp8-tp8|--reasoning-parser qwen3"
  [qwen3-32b-16k]="8005|8|$A/qwen3-32b-fp8-tp8-16k|--reasoning-parser qwen3"
  [exaone-32b]="8011|32|$A/exaone-4.0-32b-fp8-tp32/snapshots/8c42cdea3e7339fe3e3aefc5c7cff1f66b320f31|--reasoning-parser exaone4"
  [llama-70b]="8012|32|$A/llama-3.3-70b-inst-tp32/snapshots/2cbb7a6286be88e25072e56d3a64943e56408440|--tool-call-parser llama3_json"
  [qwen3-32b-tp32]="8013|32|$A/qwen3-32b-fp8-tp32/snapshots/1f5cf9426425998140e2dde6357ba0ee4f6820b2|--reasoning-parser qwen3"
  # 9번째 모델 Qwen3-Coder-30B-A3B-FP8 (qwen3-coder-30b-a3b-inst-fp8-tp8-65k) 은
  # 2026.2.0 런타임이 FP8 MoE serve 를 지원 안 해(엔진 init 시 패닉) 등록하지 않습니다.
)

DEFAULT_SET=(coder7 coder14 qwen3-32b)   # 기본 3종(tp8, 가벼운 것부터)

case "${1:-}" in
  stop) pkill -f "furiosa-llm serve" && echo "모든 serve 종료" || echo "실행 중인 serve 없음"; exit 0 ;;
  list)
    echo "등록된 모델 키 (포트 / tp / 아티팩트):"
    for k in "${!CAT[@]}"; do
      IFS='|' read -r P T ART _ <<< "${CAT[$k]}"
      printf "  %-16s :%s  tp%-2s  %s\n" "$k" "$P" "$T" "$(basename "${ART%%/snapshots*}")"
    done | sort
    echo "기본 세트: ${DEFAULT_SET[*]}"
    exit 0 ;;
esac

# 띄울 모델 목록 결정: 인자 없으면 기본 세트, 숫자면 기본 세트의 앞 N개, 그 외엔 키 목록.
if [ "$#" -eq 0 ]; then
  SEL=("${DEFAULT_SET[@]}")
elif [[ "$1" =~ ^[0-9]+$ ]]; then
  SEL=("${DEFAULT_SET[@]:0:$1}")
else
  SEL=("$@")
fi

# 키 유효성 + tp32 단독 검사
NEED32=0
for k in "${SEL[@]}"; do
  [ -n "${CAT[$k]:-}" ] || { echo "✗ 모르는 모델 키: $k   (./serve_models.sh list 로 확인)"; exit 1; }
  IFS='|' read -r _ T _ _ <<< "${CAT[$k]}"
  [ "$T" = "32" ] && NEED32=1
done
if [ "$NEED32" = "1" ] && [ "${#SEL[@]}" -gt 1 ]; then
  echo "✗ tp32 모델은 카드 4장을 모두 써서 단독으로만 띄울 수 있습니다. 하나만 지정하세요."
  exit 1
fi
if [ "$NEED32" = "0" ] && [ "${#SEL[@]}" -gt 4 ]; then
  echo "✗ tp8 모델은 카드가 4장뿐이라 동시에 최대 4개입니다(요청 ${#SEL[@]}개). 줄여서 지정하세요."
  exit 1
fi

# 빈 카드 풀에서 tp8은 1장씩 배정, tp32는 4장 전부.
FREE=(0 1 2 3)
for k in "${SEL[@]}"; do
  IFS='|' read -r PORT T ART EXTRA <<< "${CAT[$k]}"
  if [ ! -f "$ART/artifact.json" ]; then echo "⏭  skip $k — artifact 없음: $ART"; continue; fi
  if pgrep -f "furiosa-llm serve.*--port $PORT" >/dev/null 2>&1; then echo "✔  $k 포트 $PORT 이미 실행 중"; continue; fi
  if [ "$T" = "32" ]; then
    DEV="npu:0,npu:1,npu:2,npu:3"
  else
    if [ "${#FREE[@]}" -eq 0 ]; then echo "✗ 빈 카드 없음 — tp8 모델은 동시 최대 4개입니다."; exit 1; fi
    DEV="npu:${FREE[0]}"; FREE=("${FREE[@]:1}")
  fi
  echo "▶  $k ($DEV) → :$PORT   $(basename "${ART%%/snapshots*}")"
  nohup furiosa-llm serve "$ART" --devices "$DEV" --host 0.0.0.0 --port "$PORT" \
        --enable-prefix-caching $EXTRA > "$LOGDIR/$PORT.log" 2>&1 &
done

echo
echo "준비 확인:  tail -f $LOGDIR/<port>.log  →  'Uvicorn running' 뜨면 OK"
echo "채팅 UI:    cd ~/RNGD-proj/Model_Benchmark/rngd-npu/chat && .venv/bin/python chat_app.py"
