#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# sdi code 설치기 (macOS / Linux)
# 서버 리눅스에 SSH 하지 않고, 내 Mac/Linux 에서 서버 NPU LLM 에 붙는
# 코딩 에이전트 CLI("sdi")를 설치한다. (OpenCode 를 백엔드로 사용)
#
# 사용 (SDI_SERVER = 라우터 주소):
#   원격(집/외부): 먼저 SSH 터널 → SDI_SERVER=http://127.0.0.1:8400   (sdi-connect.sh / README 2-A)
#                  외부 입구가 SSH(10022)뿐이라 8400 직접 HTTP 는 안 됨.
#   사내 같은 LAN: SDI_SERVER=http://10.125.19.138:8400   (내부 사설 IP, 터널 불필요)
#   예) SDI_SERVER=http://127.0.0.1:8400 SDI_API_KEY=<발급받은키> bash install.sh
#
# 설치 후:  sdi            # 코딩 에이전트 TUI (추론은 서버 NPU)
#           sdi run "..."  # 비대화형
# ───────────────────────────────────────────────────────────────────────────
set -euo pipefail
: "${SDI_SERVER:?SDI_SERVER 를 지정하세요 (예: http://10.125.19.138:8400)}"
SDI_SERVER="${SDI_SERVER%/}"
SDI_API_KEY="${SDI_API_KEY:-}"   # 키는 선택 — 서버 인증이 OFF 면 비워도 접속 가능
# 키가 있으면 영숫자/.-_ 만 허용(설정 파일에 안전하게 들어가도록)
if [ -n "$SDI_API_KEY" ]; then
  case "$SDI_API_KEY" in
    *[!A-Za-z0-9._-]*) echo "[fail] SDI_API_KEY 에 허용되지 않는 문자가 있습니다(영숫자와 . _ - 만 허용)"; exit 1 ;;
  esac
fi

ORIG_PATH="$PATH"                                   # 유저 원본 PATH(경고 판정용; 아래에서 PATH 변형 전 캡처)
CMD="${SDI_CMD:-sdi}"                                # 명령(=provider) 이름 — 리브랜딩 시 SDI_CMD 로 변경(예: acme)
BRAND="${SDI_BRAND:-SDI Code (Furiosa NPU)}"         # picker 표시 이름
SDI_HOME="${SDI_HOME:-$HOME/.config/$CMD}"
BIN_DIR="${SDI_BIN_DIR:-$HOME/.local/bin}"
CFG="$SDI_HOME/opencode.json"
export PATH="$HOME/.opencode/bin:$PATH"             # opencode 실행용(.local/bin 은 안 넣음 — 경고 판정 오염 방지)

echo "[1/3] opencode(백엔드 런타임) 설치 확인"
if ! command -v opencode >/dev/null 2>&1; then
  echo "      설치 중…"
  curl -fsSL https://opencode.ai/install | bash -s -- --no-modify-path
  export PATH="$HOME/.opencode/bin:$PATH"
fi
command -v opencode >/dev/null 2>&1 || { echo "[fail] opencode 설치 실패"; exit 1; }

echo "[2/3] 서버에서 모델 목록 받아 sdi 설정 생성(키 포함, 0600): $SDI_SERVER"
mkdir -p "$SDI_HOME" "$BIN_DIR"
SDI_SERVER="$SDI_SERVER" SDI_API_KEY="$SDI_API_KEY" SDI_PROVIDER="$CMD" SDI_BRAND="$BRAND" python3 - "$CFG" <<'PYEOF'
import json, os, sys, urllib.request, urllib.error, urllib.parse
SERVER=os.environ["SDI_SERVER"].rstrip("/"); KEY=os.environ.get("SDI_API_KEY",""); OUT=sys.argv[1]; BASE=SERVER+"/v1"
PROV=os.environ.get("SDI_PROVIDER","sdi"); BRAND=os.environ.get("SDI_BRAND","SDI Code (Furiosa NPU)")
req=urllib.request.Request(BASE+"/models")
if KEY: req.add_header("Authorization","Bearer "+KEY)
try:
    with urllib.request.urlopen(req,timeout=10) as r: ids=[m["id"] for m in json.load(r).get("data",[])]
except urllib.error.HTTPError as e:                                  # 서버엔 닿았으나 응답이 오류
    print(f"[fail] 서버 응답 오류 {BASE}/models : HTTP {e.code}",file=sys.stderr)
    if e.code==401: print("       └ 401 = 서버 인증 ON. 발급받은 키로 재실행:  SDI_API_KEY=<키> bash install.sh",file=sys.stderr)
    sys.exit(1)
except Exception as e:                                               # 아예 도달 실패(네트워크 경로/방화벽/주소)
    u=urllib.parse.urlparse(SERVER); host=u.hostname or "?"; port=u.port or 80
    priv = host.startswith(("10.","192.168.")) or host.startswith("172.")
    print(f"[fail] 서버 도달 실패 {BASE}/models : {e}",file=sys.stderr)
    print("       └ 이건 IP·포트를 안 넣어서가 아니라(이미 SDI_SERVER 로 지정됨) '네트워크 경로'가 없을 때 납니다.",file=sys.stderr)
    print(f"       └ 점검: 이 PC가 서버와 같은 망인가요? {host} 가 사내 사설 IP({'그렇습니다' if priv else '확인필요'})면 외부에선 VPN/사내망 연결이 필요합니다.",file=sys.stderr)
    print(f"       └ 빠른 확인:  nc -vz {host} {port}   ( succeeded=정상→재시도 / 'No route to host'=망문제→VPN / refused=서버측 점검 )",file=sys.stderr)
    sys.exit(1)
if not ids: print("[fail] 서버에 모델 없음",file=sys.stderr); sys.exit(1)
def ctx(m):
    m=m.lower(); return 16384 if "16k" in m else (65536 if "a3b" in m else 32768)
# 서버 라우터가 /router/models 로 표시명·컨텍스트를 주면 그대로(서버와 동일·단일 출처). 없으면 이름 휴리스틱.
rich=None
try:
    rq=urllib.request.Request(SERVER+"/router/models")
    if KEY: rq.add_header("Authorization","Bearer "+KEY)
    with urllib.request.urlopen(rq,timeout=10) as r: rich=json.load(r).get("data") or None
except Exception: rich=None
if rich:
    ids=[mm["id"] for mm in rich]
    models={mm["id"]:{"name":mm.get("name") or mm["id"],
                      "limit":{"context":mm.get("context") or ctx(mm["id"]),"output":8192}} for mm in rich}
else:
    models={m:{"name":m,"limit":{"context":ctx(m),"output":8192}} for m in ids}
default="Qwen3-32B-FP8" if "Qwen3-32B-FP8" in ids else ids[0]
opts={"baseURL":BASE}
if KEY: opts["apiKey"]=KEY                                        # 키 없으면 apiKey 생략(무인증 접속)
cfg={"$schema":"https://opencode.ai/config.json","provider":{PROV:{"npm":"@ai-sdk/openai-compatible",
     "name":BRAND,"options":opts,"models":models}},
     "model":f"{PROV}/{default}","small_model":f"{PROV}/{default}"}
fd=os.open(OUT, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600)         # 키 포함 → 소유자 전용
with os.fdopen(fd,"w") as f: json.dump(cfg,f,indent=2,ensure_ascii=False); f.write("\n")
print(f"[ok] 모델 {len(ids)}개 등록 (기본 {default})")
PYEOF

echo "[3/3] '$CMD' 명령 설치: $BIN_DIR/$CMD"
# 래퍼는 OPENCODE_CONFIG(문서화된 크로스플랫폼 설정 경로)로 sdi 설정만 가리킨다.
# 키는 래퍼가 아니라 위 설정파일(0600)에만 있다 — 래퍼/환경에 비밀 없음.
cat > "$BIN_DIR/$CMD" <<EOF
#!/usr/bin/env bash
set -euo pipefail
[ -f "$CFG" ] || { echo "$CMD: 설정 없음($CFG) — install.sh 를 다시 실행하세요" >&2; exit 1; }
export OPENCODE_CONFIG="$CFG"
export PATH="\$HOME/.opencode/bin:\$PATH"
exec opencode "\$@"
EOF
chmod 755 "$BIN_DIR/$CMD"

echo
echo "✅ 설치 완료. (설정·키: $CFG [0600])"
case ":$ORIG_PATH:" in
  *":$BIN_DIR:"*) : ;;
  *) echo "   ⚠️  로그인 셸 PATH 에 $BIN_DIR 가 없습니다. 추가하세요:"
     echo "       echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.zshrc   # (또는 ~/.bashrc) 후 새 터미널" ;;
esac
echo "   실행:  $CMD              # 코딩 에이전트 TUI"
echo "          $CMD run \"...\"   # 비대화형 한 줄"
echo "          $CMD models       # 사용 가능한 서버 모델"
echo "          $CMD agent list   # 에이전트(빌트인+커스텀)"
echo "   제거:  rm -rf \"$SDI_HOME\" \"$BIN_DIR/$CMD\"   (키 회전 시 재실행)"
