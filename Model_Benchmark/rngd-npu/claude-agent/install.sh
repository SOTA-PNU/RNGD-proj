#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# furio — Claude Code 같은 코딩 에이전트(openclaude)를 서버 NPU 에 붙여 쓰는 CLI 설치기 (macOS / Linux)
#
# openclaude(github.com/Gitlawb/openclaude)는 Claude Code 에서 파생된 오픈소스 CLI 라
# "정말 Claude 같은" 경험(권한 프롬프트·CLAUDE.md·서브에이전트·슬래시 등)을 그대로 주되,
# 추론은 서버 NPU(OpenAI 호환 라우터 :8400)에서 돈다. 도구 실행·파일 수정은 내 PC 로컬.
#
# 사용 (SDI_SERVER = 라우터 주소):
#   원격(집/외부): 먼저 SSH 터널 → SDI_SERVER=http://127.0.0.1:8400  (furio-connect.sh / README 2-A)
#   사내 LAN:     SDI_SERVER=http://10.125.19.138:8400
#   예) SDI_SERVER=http://127.0.0.1:8400 bash install.sh
#       (서버 인증 ON 이면 SDI_API_KEY=<키> 추가)
#
# 설치 후:  furio            # Claude 같은 코딩 에이전트 TUI (추론=서버 NPU)
#           furio -p "..."   # 비대화형 한 줄(print 모드)
# ───────────────────────────────────────────────────────────────────────────
set -euo pipefail
: "${SDI_SERVER:?SDI_SERVER 를 지정하세요 (예: http://127.0.0.1:8400  ← SSH 터널 권장)}"
SDI_SERVER="${SDI_SERVER%/}"
SDI_API_KEY="${SDI_API_KEY:-}"                       # 키는 선택 — 서버 인증 OFF 면 비워도 됨
if [ -n "$SDI_API_KEY" ]; then
  case "$SDI_API_KEY" in *[!A-Za-z0-9._-]*) echo "[fail] SDI_API_KEY 에 허용 안 되는 문자(영숫자와 . _ - 만)"; exit 1 ;; esac
fi
CMD="${FURIO_CMD:-furio}"                               # 명령 이름(리브랜딩: FURIO_CMD)
BRAND="${FURIO_BRAND:-Furio (Furiosa NPU)}"
MODEL="${FURIO_MODEL:-gpt-oss-120b}"                   # 기본 모델(agent-ready). /v1/models 의 id
MAXOUT="${FURIO_MAX_OUTPUT:-8192}"                     # ⚠️ openclaude 큰 시스템프롬프트(~17.6k)+출력이 ctx 초과 방지
AUTODEF="${FURIO_AUTO:-}"                              # 완전 자동(권한 미확인) 기본값. 빈값=확인모드(안전). 1|edits 등(아래 표)
ORIG_PATH="$PATH"
HOME_DIR="${FURIO_HOME:-$HOME/.$CMD}"                  # openclaude 격리 설치/설정 위치
BIN_DIR="${FURIO_BIN_DIR:-$HOME/.local/bin}"

echo "[1/4] Node ≥22 확인 (openclaude 요구)"
command -v node >/dev/null 2>&1 || { echo "[fail] node 없음 — Node ≥22 설치 후 재실행 (nvm: 'nvm install 22', 또는 brew install node)"; exit 1; }
NODEMAJ=$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)
[ "${NODEMAJ:-0}" -ge 22 ] || { echo "[fail] $(node -v) — openclaude 는 Node ≥22 필요. 'nvm install 22'(또는 brew install node) 후 재실행."; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "[fail] npm 없음(Node 설치 확인)"; exit 1; }

echo "[2/4] openclaude 설치 (격리 prefix: $HOME_DIR — 전역 npm 안 건드림)"
mkdir -p "$HOME_DIR" "$BIN_DIR"
npm install -g @gitlawb/openclaude@latest --prefix "$HOME_DIR" >/dev/null 2>&1 || { echo "[fail] openclaude 설치 실패 (npm 로그 확인)"; exit 1; }
OC_BIN="$HOME_DIR/bin/openclaude"
[ -x "$OC_BIN" ] || { echo "[fail] openclaude 바이너리 없음: $OC_BIN"; exit 1; }

echo "[3/4] 서버 도달 확인: $SDI_SERVER"
if command -v curl >/dev/null 2>&1; then
  AUTH=(); [ -n "$SDI_API_KEY" ] && AUTH=(-H "Authorization: Bearer $SDI_API_KEY")
  # macOS 기본 bash 3.2 + set -u 에선 빈 배열 "${AUTH[@]}" 가 unbound 오류 → ${arr[@]+...} 가드로 안전 확장
  if ! curl -fsS --max-time 10 ${AUTH[@]+"${AUTH[@]}"} "$SDI_SERVER/v1/models" >/dev/null 2>&1; then
    echo "[fail] 서버 도달 실패 $SDI_SERVER/v1/models"
    echo "       └ 원격이면 SSH 터널이 떠 있나요?  bash furio-connect.sh  (그 후 SDI_SERVER=http://127.0.0.1:8400)"
    echo "       └ 서버 인증 ON 이면 SDI_API_KEY=<키> 를 같이 주세요."
    exit 1
  fi
  echo "      [ok] /v1/models 응답"
fi

# 키는 0600 파일에만(있을 때). 래퍼는 런타임에 읽어 env 로 — 래퍼 텍스트엔 비밀 없음.
if [ -n "$SDI_API_KEY" ]; then (umask 177; printf '%s' "$SDI_API_KEY" > "$HOME_DIR/key"); else rm -f "$HOME_DIR/key"; fi

echo "[4/4] '$CMD' 명령 설치: $BIN_DIR/$CMD"
cat > "$BIN_DIR/$CMD" <<EOF
#!/usr/bin/env bash
# $BRAND — openclaude(Claude Code 계열) + 서버 NPU(OpenAI 호환 라우터). 키는 래퍼에 없음(0600 파일).
set -euo pipefail
export CLAUDE_CODE_USE_OPENAI=1
export OPENAI_BASE_URL="$SDI_SERVER/v1"
export OPENAI_MODEL="\${OPENAI_MODEL:-$MODEL}"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="\${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-$MAXOUT}"
export OPENCLAUDE_CONFIG_DIR="\${OPENCLAUDE_CONFIG_DIR:-$HOME_DIR/config}"   # 유저 기존 openclaude 와 격리
export OPENAI_API_KEY="\$( [ -f "$HOME_DIR/key" ] && cat "$HOME_DIR/key" || echo dummy )"
# 완전 자동 실행 모드 토글(FURIO_AUTO, 설치시 기본 '$AUTODEF'). 런타임 override 가능. ⚠️신뢰 폴더에서만.
FURIO_AUTO="\${FURIO_AUTO:-$AUTODEF}"
AUTO_ARGS=()
case "\$FURIO_AUTO" in
  1|yes|on|bypass|full) AUTO_ARGS=(--dangerously-skip-permissions) ;;  # 모든 권한 프롬프트 생략(완전자동)
  edits|accept)         AUTO_ARGS=(--permission-mode acceptEdits) ;;   # 파일편집만 자동(Bash 등은 확인)
esac
exec "$OC_BIN" \${AUTO_ARGS[@]+"\${AUTO_ARGS[@]}"} "\$@"   # macOS bash3.2 + set -u 빈배열 가드
EOF
chmod 755 "$BIN_DIR/$CMD"

echo
echo "✅ 설치 완료. (openclaude: $HOME_DIR, 명령: $BIN_DIR/$CMD)"
case ":$ORIG_PATH:" in
  *":$BIN_DIR:"*) : ;;
  *) echo "   ⚠️  로그인 셸 PATH 에 $BIN_DIR 가 없습니다. 추가하세요:"
     echo "       echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.zshrc   # (또는 ~/.bashrc) 후 새 터미널" ;;
esac
echo "   실행:  $CMD              # Claude 같은 코딩 에이전트 TUI"
echo "          $CMD -p \"...\"     # 비대화형 한 줄(print)"
echo "          $CMD --model gpt-oss-120b   # 또는 OPENAI_MODEL 로 모델 변경"
echo "   ⚠️ 작업은 일반 프로젝트 폴더에서(.claude 등 민감 경로엔 쓰기 차단). 터널 방식이면 furio 쓰는 동안 터널 유지."
echo "   제거:  rm -rf \"$HOME_DIR\" \"$BIN_DIR/$CMD\""
