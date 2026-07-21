#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# sdi connect — 서버 NPU 라우터로 가는 보안 SSH 터널 (부산대 서버: SSH 포트 10022)
#
# 이 서버는 외부 입구가 SSH(10022)뿐이라, 라우터(:8400)에 직접 HTTP로 못 붙습니다.
# 대신 SSH 터널로 내 PC의 localhost:8400 을 서버의 localhost:8400 에 연결합니다.
# (코딩 에이전트는 여전히 내 PC에서 로컬 실행 — SSH는 '암호화된 파이프'로만 씀)
#
# 사용:
#   SDI_SSH_USER=jun bash sdi-connect.sh        # 이 창은 켜 둔 채로(비밀번호 입력), 다른 창에서 sdi 사용
#   그 다음 다른 터미널에서:
#     SDI_SERVER=http://127.0.0.1:8400 bash install.sh   # (최초 1회 설치)
#     sdi                                                # 사용
# ───────────────────────────────────────────────────────────────────────────
set -euo pipefail
: "${SDI_SSH_USER:?서버 계정을 지정하세요 (예: SDI_SSH_USER=jun)}"
HOST="${SDI_SSH_HOST:-164.125.19.138}"     # 공인 IP
SSH_PORT="${SDI_SSH_PORT:-10022}"          # 서버 SSH 포트
LOCAL_PORT="${SDI_LOCAL_PORT:-8400}"       # 내 PC에서 쓸 포트(설치 시 SDI_SERVER=http://127.0.0.1:이포트)

# 이미 이 포트로 라우터가 닿으면(=터널이 이미 떠 있음) 새로 안 띄움
if curl -fsS --max-time 3 "http://127.0.0.1:$LOCAL_PORT/v1/models" >/dev/null 2>&1; then
  echo "✅ 이미 localhost:$LOCAL_PORT 로 서버에 연결돼 있습니다(터널 가동 중). 새 터널 불필요 — 바로 'sdi' 쓰세요."
  echo "   (굳이 새로 열려면 다른 포트로:  SDI_LOCAL_PORT=8401 bash sdi-connect.sh)"
  exit 0
fi

echo "🔌 터널 연결: localhost:$LOCAL_PORT  →  (ssh -p $SSH_PORT $SDI_SSH_USER@$HOST)  →  서버 localhost:8400"
echo "   비밀번호를 입력하세요. 이 창은 켜 둔 채로 다른 터미널에서 sdi 를 쓰세요. (종료: Ctrl-C)"
echo "   설치/사용 시 서버주소:  SDI_SERVER=http://127.0.0.1:$LOCAL_PORT"
# -N: 원격 셸 안 염(포트포워딩만) / ExitOnForwardFailure: 포트바인드 실패면 즉시 실패(조용히 넘어가지 않음) / ServerAlive: 끊김 감지
ssh -p "$SSH_PORT" -N -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L "$LOCAL_PORT:localhost:8400" "$SDI_SSH_USER@$HOST" || {
  echo ""
  echo "[!] 터널 실패. 'Address already in use' 였다면 맥의 localhost:$LOCAL_PORT 가 이미 사용 중입니다:"
  echo "    ① 이미 터널이 떠 있을 수 있음:  curl http://127.0.0.1:$LOCAL_PORT/v1/models   (모델 나오면 그냥 'sdi' 쓰면 됩니다)"
  echo "    ② 끊고 다시:  lsof -nP -iTCP:$LOCAL_PORT -sTCP:LISTEN  로 PID 확인 후 kill,  또는  pkill -f '$LOCAL_PORT:localhost:8400'"
  echo "    ③ 다른 포트로:  SDI_LOCAL_PORT=8401 bash sdi-connect.sh   (설치/사용도 SDI_SERVER=http://127.0.0.1:8401)"
  exit 1
}
