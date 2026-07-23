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
AUTODEF="${FURIO_AUTO:-}"                              # 자동모드 기본값. 빈값=매번확인(가장 안전) / safe=규칙기반 / edits=편집만 / 1=완전자동
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
AUTH=(); [ -n "$SDI_API_KEY" ] && AUTH=(-H "Authorization: Bearer $SDI_API_KEY")

# 서버가 NPU 기능(모델별 LED·dp/pp)이 들어간 openclaude 포크를 빌드해 두었는지 확인.
# 있으면 npm 으로 '포크와 같은 버전'을 고정 설치한 뒤 dist 만 덮어쓴다 —
# 개인 PC 에 bun/빌드 툴체인을 깔 필요가 없고, install.sh 한 줄이 그대로 유지된다.
FORK_VER=$(curl -fsS --max-time 10 ${AUTH[@]+"${AUTH[@]}"} "$SDI_SERVER/router/client/manifest.json" 2>/dev/null | node -e '
let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{
  try{const j=JSON.parse(s);process.stdout.write(j.ok&&j.version?String(j.version):"")}catch(e){process.stdout.write("")}});' 2>/dev/null || echo "")

if [ -n "$FORK_VER" ]; then
  echo "      서버 포크 감지 — @gitlawb/openclaude@$FORK_VER 고정 설치 후 NPU 기능 dist 적용"
  NPM_SPEC="@gitlawb/openclaude@$FORK_VER"
else
  echo "      서버 포크 없음 — 업스트림 최신으로 설치(NPU LED/위젯 없이 동작)"
  NPM_SPEC="@gitlawb/openclaude@latest"
fi
npm install -g "$NPM_SPEC" --prefix "$HOME_DIR" >/dev/null 2>&1 || { echo "[fail] openclaude 설치 실패 ($NPM_SPEC — npm 로그 확인)"; exit 1; }
OC_BIN="$HOME_DIR/bin/openclaude"
[ -x "$OC_BIN" ] || { echo "[fail] openclaude 바이너리 없음: $OC_BIN"; exit 1; }

# 포크 dist 덮어쓰기. 실패하면 업스트림 dist 가 그대로 남아 furio 는 계속 동작한다
# (NPU LED·dp/pp 위젯만 빠짐) — 설치를 통째로 실패시키지 않는다.
if [ -n "$FORK_VER" ]; then
  DIST_DIR="$HOME_DIR/lib/node_modules/@gitlawb/openclaude/dist"
  [ -d "$DIST_DIR" ] || DIST_DIR=$(dirname "$(readlink -f "$OC_BIN" 2>/dev/null || echo "$OC_BIN")")/../dist
  OK=1
  for f in cli.mjs sdk.mjs; do
    if ! curl -fsS --max-time 300 ${AUTH[@]+"${AUTH[@]}"} -o "$DIST_DIR/$f.new" "$SDI_SERVER/router/client/$f" 2>/dev/null; then
      OK=0; break
    fi
    mv -f "$DIST_DIR/$f.new" "$DIST_DIR/$f" || { OK=0; break; }
  done
  rm -f "$DIST_DIR"/*.new 2>/dev/null
  if [ "$OK" = 1 ]; then
    echo "      [ok] NPU 기능 적용 (모델별 LED·dp/pp 표시)"
  else
    echo "      [warn] 포크 dist 내려받기 실패 — 업스트림 그대로 사용(NPU LED 없음)"
  fi
fi

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

# 모델별 '진짜' 컨텍스트 창을 라우터에서 받아 ctx.json 에 저장(단일 출처 = 서버 REGISTRY).
# ⚠️ 이게 없으면 openclaude 는 처음 보는 우리 모델 id 를 전부 128000 토큰으로 가정한다.
#    실제론 40960~262144 로 제각각이라, 작은 모델에선 컨텍스트 초과(400)가 나고 큰 모델은 손해다.
N=$(curl -fsS --max-time 10 ${AUTH[@]+"${AUTH[@]}"} "$SDI_SERVER/router/models" 2>/dev/null | node -e '
const fs=require("fs"); let s=""; process.stdin.on("data",d=>s+=d).on("end",()=>{
  try{ const j=JSON.parse(s), m={};
    for (const x of (j.data||[])) if (x && x.id && x.context) m[x.id]=x.context;
    const n=Object.keys(m).length;
    if (n) fs.writeFileSync(process.argv[1], JSON.stringify(m));
    process.stdout.write(String(n));
  }catch(e){ process.stdout.write("0") }});' "$HOME_DIR/ctx.json" 2>/dev/null || echo 0)
case "${N:-0}" in ''|*[!0-9]*) N=0 ;; esac
if [ "$N" -gt 0 ]; then
  echo "      [ok] 모델별 컨텍스트 ${N}개 기록 (ctx.json — openclaude 가 우리 ctx 를 정확히 알게 됨)"
else
  rm -f "$HOME_DIR/ctx.json"
  echo "      [warn] /router/models 응답 없음 — 모델별 ctx 미설정(openclaude 가 128000 으로 가정할 수 있음)"
fi

# 모델 선택 목록(/model)에 뜨는 설명을 라우터에서 받아 desc.json 에 저장.
# openclaude 는 원래 "Detected from Local OpenAI-compatible" 을 하드코딩하는데,
# 그건 이 모델이 NPU 몇 장을 어떤 병렬 구성으로 쓰는지 전혀 알려주지 않는다.
# 라우터가 주는 "tp8·dp2·pp1 · 2장 · ctx 40k · fxb" 로 바꿔 고를 때 판단이 되게 한다.
# (OpenAI /v1/models 규격엔 description 필드가 없어서 설치 때 받아 두는 것이 확실하다.)
D=$(curl -fsS --max-time 10 ${AUTH[@]+"${AUTH[@]}"} "$SDI_SERVER/router/models" 2>/dev/null | node -e '
const fs=require("fs"); let s=""; process.stdin.on("data",d=>s+=d).on("end",()=>{
  try{ const j=JSON.parse(s), m={};
    for (const x of (j.data||[])) if (x && x.id && x.description) m[x.id]=x.description;
    const n=Object.keys(m).length;
    if (n) fs.writeFileSync(process.argv[1], JSON.stringify(m));
    process.stdout.write(String(n));
  }catch(e){ process.stdout.write("0") }});' "$HOME_DIR/desc.json" 2>/dev/null || echo 0)
case "${D:-0}" in ''|*[!0-9]*) D=0 ;; esac
if [ "$D" -gt 0 ]; then
  echo "      [ok] 모델 설명 ${D}개 기록 (desc.json — /model 목록에 tp·dp·pp·카드 수 표시)"
else
  rm -f "$HOME_DIR/desc.json"
fi

# 안전 자동모드(FURIO_AUTO=safe)용 규칙 파일. 사용자가 편집할 수 있게 파일로 두고, 이미 있으면 건드리지 않는다.
#   auto-allow.txt : 물어보지 않고 바로 실행할 것(읽기·조회·테스트 등)
#   auto-deny.txt  : 아예 막을 것(파괴적·되돌릴 수 없는·외부로 나가는 명령)
#   둘 다 아닌 명령은 그대로 사람에게 물어본다 ← 이게 'safe' 의 핵심
if [ ! -f "$HOME_DIR/auto-allow.txt" ]; then
  cat > "$HOME_DIR/auto-allow.txt" <<'ALLOWEOF'
# 자동 승인(묻지 않음) — 읽기/조회/테스트처럼 되돌릴 수 있는 것만.
# 문법: 도구이름 또는 도구이름(명령 접두사:*)   예) Bash(git status:*)
# 한 줄에 하나. '#' 로 시작하면 주석. 편집 후 바로 반영됩니다(재설치 불필요).
Read
Glob
Grep
TodoWrite
Bash(ls:*)
Bash(pwd:*)
Bash(cat:*)
Bash(head:*)
Bash(tail:*)
Bash(wc:*)
Bash(file:*)
Bash(stat:*)
Bash(du:*)
Bash(df:*)
Bash(tree:*)
Bash(date:*)
Bash(echo:*)
Bash(which:*)
Bash(find:*)
Bash(grep:*)
Bash(rg:*)
Bash(diff:*)
Bash(git status:*)
Bash(git diff:*)
Bash(git log:*)
Bash(git show:*)
Bash(git branch:*)
Bash(git remote:*)
Bash(npm test:*)
Bash(npm run:*)
Bash(pytest:*)
Bash(make:*)
Bash(cargo test:*)
Bash(cargo build:*)
Bash(go test:*)
Bash(go build:*)
ALLOWEOF
fi
if [ ! -f "$HOME_DIR/auto-deny.txt" ]; then
  cat > "$HOME_DIR/auto-deny.txt" <<'DENYEOF'
# 차단(묻지도 않고 거부) — 파괴적이거나 되돌릴 수 없거나 밖으로 나가는 것.
# kill/pkill 은 이 서버의 라우터까지 죽일 수 있어 막아 둡니다.
Bash(rm -rf:*)
Bash(rm -fr:*)
Bash(sudo:*)
Bash(su:*)
Bash(dd:*)
Bash(mkfs:*)
Bash(fdisk:*)
Bash(parted:*)
Bash(shutdown:*)
Bash(reboot:*)
Bash(halt:*)
Bash(poweroff:*)
Bash(chown:*)
Bash(chmod 777:*)
Bash(kill:*)
Bash(pkill:*)
Bash(killall:*)
Bash(curl:*)
Bash(wget:*)
Bash(git push:*)
Bash(git reset --hard:*)
Bash(git clean:*)
Bash(crontab:*)
Bash(npm publish:*)
Bash(ssh:*)
Bash(scp:*)
DENYEOF
fi

# Shift+Tab 으로 자동모드(bypassPermissions)에 들어갈 수 있게 한다.
#
# ⚠️ 이 설정은 자동모드를 **켜지 않는다**. 모드 순환 목록에 '나타나게만' 한다.
#    openclaude 는 기본적으로 bypassPermissions 를 목록에서 숨기고, 켜려면 실행할 때마다
#    --dangerously-skip-permissions 를 붙이게 한다. 그러면 '세션 도중 잠깐만 자동으로'가
#    불가능해서, 결국 늘 위험한 플래그로 켜 두는 쪽으로 사람을 밀어붙인다.
#    이 설정을 주면 평소엔 안전한 default 로 쓰다가 필요할 때만 Shift+Tab 으로 올라갔다
#    내려올 수 있다. 스키마 설명도 정확히 그 용도다:
#      "Allow bypass permissions mode to appear in the mode list without requiring the CLI flag"
#
#    Shift+Tab 순환:  default → acceptEdits → plan → bypassPermissions → fullAccess → default
#
#    잠그고 싶으면 FURIO_ALLOW_BYPASS=0 으로 설치하면 된다(그러면 CLI 플래그로만 가능).
CFG_DIR="$HOME_DIR/config"
if [ "${FURIO_ALLOW_BYPASS:-1}" = "0" ]; then
  echo "      [skip] Shift+Tab 자동모드 비활성(FURIO_ALLOW_BYPASS=0)"
else
  mkdir -p "$CFG_DIR"
  # 기존 settings.json 이 있으면 다른 키는 보존하고 이 항목만 병합한다.
  node -e '
const fs=require("fs"), p=process.argv[1];
let j={};
try{ j=JSON.parse(fs.readFileSync(p,"utf8"))||{} }catch(e){ j={} }
if (typeof j!=="object"||Array.isArray(j)) j={};
j.permissions = (j.permissions && typeof j.permissions==="object" && !Array.isArray(j.permissions)) ? j.permissions : {};
j.permissions.allowBypassPermissionsMode = true;
fs.writeFileSync(p, JSON.stringify(j,null,2)+"\n");
' "$CFG_DIR/settings.json" 2>/dev/null \
    && { echo "      [ok] 자동모드 사용 가능 — 실행 후 Shift+Tab 을 눌러 모드를 바꾼다:"
         echo "           default(매번 확인) → Accept edits → Plan → Bypass Permissions → Full Access"
         echo "           처음 Bypass 로 올라갈 때만 확인창이 한 번 뜨고, 이후엔 저장돼 안 뜬다."; } \
    || echo "      [warn] settings.json 기록 실패 — 자동모드는 FURIO_AUTO=1 로만 가능"
fi

echo "[4/4] '$CMD' 명령 설치: $BIN_DIR/$CMD"
cat > "$BIN_DIR/$CMD" <<EOF
#!/usr/bin/env bash
# $BRAND — openclaude(Claude Code 계열) + 서버 NPU(OpenAI 호환 라우터). 키는 래퍼에 없음(0600 파일).
set -euo pipefail
export CLAUDE_CODE_USE_OPENAI=1
export OPENAI_BASE_URL="$SDI_SERVER/v1"
export OPENCLAUDE_CONFIG_DIR="\${OPENCLAUDE_CONFIG_DIR:-$HOME_DIR/config}"   # 유저 기존 openclaude 와 격리
# 기본 모델 = 직전에 쓰던 모델. openclaude 는 /model 로 바꿀 때마다 그 값을 config 의
# settings.json 에 저장한다(동기). 그걸 읽어 OPENAI_MODEL 기본값으로 쓰면, furio 를 다시
# 켜도 마지막 모델이 그대로 뜬다. 우선순위: 명시적 OPENAI_MODEL > 직전 모델 > 설치 기본값($MODEL).
# (furio --model 은 이와 무관하게 최우선으로 그 실행만 덮어쓴다.)
_LAST_MODEL="\$(node -e 'try{var fs=require("fs");var s=JSON.parse(fs.readFileSync(process.env.OPENCLAUDE_CONFIG_DIR+"/settings.json","utf8"));if(s&&typeof s.model==="string"&&s.model)process.stdout.write(s.model)}catch(e){}' 2>/dev/null || true)"
export OPENAI_MODEL="\${OPENAI_MODEL:-\${_LAST_MODEL:-$MODEL}}"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="\${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-$MAXOUT}"
export OPENAI_API_KEY="\$( [ -f "$HOME_DIR/key" ] && cat "$HOME_DIR/key" || echo dummy )"
# 모델별 실제 컨텍스트 창(설치 때 라우터에서 받아둔 값). 없으면 openclaude 는 128000 으로 잘못 가정한다.
if [ -z "\${CLAUDE_CODE_OPENAI_CONTEXT_WINDOWS:-}" ] && [ -s "$HOME_DIR/ctx.json" ]; then
  export CLAUDE_CODE_OPENAI_CONTEXT_WINDOWS="\$(cat "$HOME_DIR/ctx.json")"
fi
# 모델 선택 목록의 설명(설치 때 라우터에서 받아둔 값). 없으면 openclaude 기본 문구로 표시된다.
if [ -z "\${FURIO_MODEL_DESCRIPTIONS:-}" ] && [ -s "$HOME_DIR/desc.json" ]; then
  export FURIO_MODEL_DESCRIPTIONS="\$(cat "$HOME_DIR/desc.json")"
fi
# 타임아웃: NPU 는 모델을 늦게 올려서(라우터가 최대 480초 대기) openclaude 0.25.0 기본값이 빠듯하다.
export API_TIMEOUT_MS="\${API_TIMEOUT_MS:-900000}"                                # 응답헤더 마감(0.25.0 기본 600000)
export CLAUDE_STREAM_IDLE_TIMEOUT_MS="\${CLAUDE_STREAM_IDLE_TIMEOUT_MS:-600000}"  # SSE 유휴(0.25.0서 120s→90s 축소됨)
# 완전 자동 실행 모드 토글(FURIO_AUTO, 설치시 기본 '$AUTODEF'). 런타임 override 가능. ⚠️신뢰 폴더에서만.
FURIO_AUTO="\${FURIO_AUTO:-$AUTODEF}"
AUTO_ARGS=()
case "\$FURIO_AUTO" in
  1|yes|on|bypass|full) AUTO_ARGS=(--dangerously-skip-permissions) ;;  # 모든 권한 프롬프트 생략(완전자동)
  edits|accept)         AUTO_ARGS=(--permission-mode acceptEdits) ;;   # 파일편집만 자동(Bash 등은 확인)
  safe|rules)
    # 규칙 기반 '안전 자동모드': 안전한 건 자동 승인, 위험한 건 차단, 나머지는 사람에게 질문.
    # openclaude 네이티브 'auto' 모드는 쓰지 않는다 — 그건 Anthropic 서버측 분류기
    # (feature('TRANSCRIPT_CLASSIFIER'))로 매 행동의 안전성을 판정하는 방식이라, NPU 라우터를
    # 백엔드로 쓰는 우리 환경에선 분류기가 아예 없다. 게이트만 풀면 있지도 않은 안전판정을
    # 있다고 속이는 셈이 된다. 그래서 정식 기능인 --allowed-tools/--disallowed-tools 로
    # 동등한 동작을 만든다(규칙은 auto-allow.txt/auto-deny.txt — 사용자가 직접 편집 가능).
    # 완전 자동이 필요하면 실행 중 Shift+Tab 으로 bypassPermissions 에 올라가면 된다.
    AUTO_ARGS=(--permission-mode acceptEdits)
    _ALLOW=(); _DENY=()
    if [ -f "$HOME_DIR/auto-allow.txt" ]; then
      while IFS= read -r _r || [ -n "\$_r" ]; do
        case "\$_r" in ''|'#'*) ;; *) _ALLOW+=("\$_r") ;; esac
      done < "$HOME_DIR/auto-allow.txt"
    fi
    if [ -f "$HOME_DIR/auto-deny.txt" ]; then
      while IFS= read -r _r || [ -n "\$_r" ]; do
        case "\$_r" in ''|'#'*) ;; *) _DENY+=("\$_r") ;; esac
      done < "$HOME_DIR/auto-deny.txt"
    fi
    [ \${#_ALLOW[@]} -gt 0 ] && AUTO_ARGS+=(--allowed-tools "\${_ALLOW[@]}")
    [ \${#_DENY[@]}  -gt 0 ] && AUTO_ARGS+=(--disallowed-tools "\${_DENY[@]}")
    ;;
esac
# 도구 축소(선택): FURIO_TOOLS="Bash,Edit,Read,Write,Glob,Grep" 처럼 주면 도구정의 토큰(~8.8k)이 줄어
# 작은 ctx 모델에서 여유가 생긴다. 미지정이면 전체 도구(기본).
TOOL_ARGS=()
[ -n "\${FURIO_TOOLS:-}" ] && TOOL_ARGS=(--tools "\$FURIO_TOOLS")
exec "$OC_BIN" \${AUTO_ARGS[@]+"\${AUTO_ARGS[@]}"} \${TOOL_ARGS[@]+"\${TOOL_ARGS[@]}"} "\$@"   # macOS bash3.2 + set -u 빈배열 가드
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
