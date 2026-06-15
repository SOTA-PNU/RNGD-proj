#!/usr/bin/env bash
# ntop.sh — Furiosa RNGD NPU 실시간 모니터링 (watch 기반)
#
# 사용:
#   ./ntop.sh                # 1초마다 갱신
#   INTERVAL=0.5 ./ntop.sh   # 0.5초마다
#   ./ntop.sh --no-ps        # 프로세스 패널 끄기
#   ./ntop.sh --simple       # furiosa-smi 자체 --watch 만 (가장 가벼움)
#
# 종료: Ctrl+C
#
# 의존: furiosa-smi (Furiosa SDK), watch (procps)

set -u
set -o pipefail 2>/dev/null || true   # sh로 실행될 때 무시

INTERVAL=${INTERVAL:-1}
SHOW_PS=1
SIMPLE=0
for arg in "$@"; do
    case "$arg" in
        --no-ps)  SHOW_PS=0 ;;
        --simple) SIMPLE=1 ;;
        --help|-h)
            sed -n '2,11p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
    esac
done

if ! command -v furiosa-smi >/dev/null 2>&1; then
    echo "❌ furiosa-smi 가 PATH 에 없습니다. Furiosa SDK 설치를 확인해주세요." >&2
    exit 1
fi

if [ "$SIMPLE" = "1" ]; then
    exec furiosa-smi status --watch "$INTERVAL"
fi

# 본문을 임시파일로 빼서 watch 의 /bin/sh 가 bash-quoted 문자열을 다시 파싱하지
# 않도록 한다 (이전 버전에서 sh: Syntax error '(' unexpected 의 원인이었음).
TMP="$(mktemp --tmpdir ntop-XXXXXX.sh)"
trap 'rm -f "$TMP"' EXIT INT TERM

cat > "$TMP" <<'BODY'
#!/usr/bin/env bash
CY=$'\e[1;36m'; YL=$'\e[1;33m'; GR=$'\e[1;32m'; RD=$'\e[0;31m'
MG=$'\e[0;35m'; DM=$'\e[2m'; RS=$'\e[0m'

printf '%s=== Furiosa NPU  |  %s  |  Ctrl+C to quit ===%s\n\n' \
    "$CY" "$(date '+%Y-%m-%d %H:%M:%S')" "$RS"

printf '%s── info (firmware / temp / power) ──%s\n' "$CY" "$RS"
furiosa-smi info 2>&1 | sed -E "
    s/([0-9]+(\.[0-9]+)?°C)/${RD}\\1${RS}/g
    s/([0-9]+(\.[0-9]+)? W\b)/${MG}\\1${RS}/g
    s/(\| *npu[0-9]+ *\|)/${YL}\\1${RS}/g
"
echo

printf '%s── status (memory / per-core utilization) ──%s\n' "$CY" "$RS"
furiosa-smi status 2>&1 | sed -E "
    s/(alive)/${GR}\\1${RS}/g
    s/(dead|error|unknown)/${RD}\\1${RS}/g
    s/(\| *npu[0-9]+ *\|)/${YL}\\1${RS}/g
    s/(\([0-9]+\.[0-9]+%\))/${CY}\\1${RS}/g
"
echo

if [ "${SHOW_PS:-1}" = "1" ]; then
    printf '%s── furiosa-smi ps ──%s\n' "$CY" "$RS"
    PS_OUT="$(furiosa-smi ps 2>&1)"
    if echo "$PS_OUT" | grep -q "^| *[0-9]"; then
        echo "$PS_OUT"
    else
        printf '%s(NPU 점유 프로세스 없음)%s\n' "$DM" "$RS"
    fi
    echo

    printf '%s── host: memory + load ──%s\n' "$CY" "$RS"
    free -h | head -3
    LOAD="$(cut -d' ' -f1-3 /proc/loadavg)"
    printf 'loadavg: %s%s%s\n\n' "$GR" "$LOAD" "$RS"

    printf '%s── build / serve processes ──%s\n' "$CY" "$RS"
    printf '%s%-7s %-8s %5s %5s %10s  %s%s\n' "$DM" "PID" "USER" "%CPU" "%MEM" "ETIME" "CMD" "$RS"
    ps -eo pid,user,pcpu,pmem,etime,cmd --sort=-pcpu 2>/dev/null \
        | grep -E "furiosa-llm|python[0-9.]*  *.*(build|serve|orchestrator)" \
        | grep -vE "grep|ntop|npu_top|npu-top" \
        | head -8 \
        | awk -v y="$YL" -v r="$RD" -v g="$GR" -v c="$CY" -v rs="$RS" '
            {
              cmd = ""
              for (i = 6; i <= NF; i++) cmd = cmd $i " "
              if (length(cmd) > 110) cmd = substr(cmd, 1, 108) "…"
              printf "%s%6s%s %-8s %s%5s%%%s %s%5s%%%s %s%10s%s  %s\n",
                     y, $1, rs, $2, r, $3, rs, g, $4, rs, c, $5, rs, cmd
            }'
fi
BODY

export SHOW_PS
exec watch -n "$INTERVAL" -c -t -d "SHOW_PS=$SHOW_PS bash $TMP"
