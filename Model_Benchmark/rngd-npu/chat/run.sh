#!/usr/bin/env bash
# RNGD NPU Chat (gradio) 서버측 기동 헬퍼.
#   ./run.sh start    # 7860에 detached 기동(세션 종료해도 유지) → 맥북에서 alpacon tunnel 로 접속
#   ./run.sh stop     # 종료
#   ./run.sh status   # 상태
#   ./run.sh restart  # 재기동
# 접속(개인 맥북 터미널):
#   alpacon tunnel furiosa-npu-e6ec40 -l 7860 -r 7860   → 브라우저 http://127.0.0.1:7860
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${CHAT_PORT:-7860}"
PY="$HERE/.venv/bin/python"
APP="$HERE/chat_app.py"
LOG="$HERE/gradio.log"
PIDF="$HERE/.gradio.pid"

port_pid() { ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1; }

start() {
  local p; p="$(port_pid)"
  if [ -n "$p" ]; then echo "이미 실행 중 (PID $p, 포트 $PORT). 먼저 ./run.sh stop"; return 1; fi
  CHAT_PORT="$PORT" setsid "$PY" "$APP" > "$LOG" 2>&1 &
  # setsid 래퍼가 아닌 실제 리스너 PID 를 잡는다
  for _ in $(seq 1 40); do
    sleep 1; p="$(port_pid)"; [ -n "$p" ] && break
  done
  if [ -n "$p" ]; then echo "$p" > "$PIDF"; echo "기동됨 (PID $p, 포트 $PORT). 로그: $LOG";
    echo "맥북에서: alpacon tunnel furiosa-npu-e6ec40 -l $PORT -r $PORT  → http://127.0.0.1:$PORT";
  else echo "기동 실패 — 로그 확인: $LOG"; tail -n 20 "$LOG"; return 1; fi
}

stop() {
  local p; p="$(port_pid)"; [ -z "$p" ] && p="$(cat "$PIDF" 2>/dev/null)"
  if [ -z "$p" ]; then echo "실행 중인 인스턴스 없음 (포트 $PORT)"; rm -f "$PIDF"; return 0; fi
  kill "$p" 2>/dev/null; sleep 2
  if kill -0 "$p" 2>/dev/null; then kill -9 "$p" 2>/dev/null; sleep 1; fi
  rm -f "$PIDF"; echo "종료됨 (PID $p)"
}

status() {
  local p; p="$(port_pid)"
  if [ -n "$p" ]; then echo "실행 중: PID $p, 포트 $PORT  (http://127.0.0.1:$PORT)";
  else echo "꺼짐 (포트 $PORT)"; fi
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  restart) stop; sleep 1; start ;;
  status) status ;;
  *) echo "사용법: $0 {start|stop|restart|status}"; exit 1 ;;
esac
