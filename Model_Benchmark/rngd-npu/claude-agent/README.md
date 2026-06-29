# furio — Claude Code 같은 코딩 에이전트(openclaude) + 서버 NPU

[openclaude](https://github.com/Gitlawb/openclaude)

```
[내 Mac/Win] furio (openclaude, Node≥22) ──► localhost:8400 ──[SSH 터널 :10022]──► [서버] furiosa_router → furiosa-llm serve(NPU)
```

---

## 1. NPU 서버 — 라우터 기동

```bash
cd ~/RNGD-proj/Model_Benchmark/rngd-npu/coding-agent
bash serve-router.sh start                               # 이미 sdi 용으로 떠 있으면 그대로 사용
#curl -s localhost:8400/v1/models | python3 -m json.tool  # 확인
```

## 2. 사용자 — 각자 PC

> **사전 요구: Node ≥ 22** (openclaude 요구). 없으면 `nvm install 22` 또는 `brew install node`(Mac) / nodejs.org(Win).

### 2-A. 원격(집/외부) — SSH 터널 (권장)
**터미널 ①** — 터널(유지):
```bash
SDI_SSH_USER=jun bash furio-connect.sh
#   (= ssh -p 10022 -N -L 8400:localhost:8400 jun@164.125.19.138)  ← 비밀번호
```
>  다른 서버 계정이면 `jun`(또는 `SDI_SSH_USER`)만 바꾸면 됩니다(install/furio/설정 그대로).


**터미널 ②** — 설치(최초 1회) + 사용:
```bash
SDI_SERVER=http://127.0.0.1:8400 bash install.sh   # 인증 ON 이면 SDI_API_KEY=<키> 추가
furio                                               # Claude 같은 TUI
```

### 2-B. 사내 같은 LAN — 직접 (터널 불필요)
```bash
SDI_SERVER=http://10.125.19.138:8400 bash install.sh
furio
```

**Windows (PowerShell)** — 터미널① 터널 후:
```powershell
$env:SDI_SERVER="http://127.0.0.1:8400"
powershell -ExecutionPolicy Bypass -File install.ps1
furio
```

## 3. 사용

```bash
furio                       # Claude 같은 코딩 에이전트 TUI (추론=서버 NPU, 코딩=로컬)
furio -p "버그 고쳐줘"        # 비대화형 한 줄(print 모드)
furio --model Qwen3-32B-FP8 # 모델 변경(또는 OPENAI_MODEL 환경변수)
furio --continue            # 직전 대화 이어가기
```
TUI 안: `/` 슬래시 명령, 권한 프롬프트(Claude 처럼 도구 실행 전 확인), `@파일` 등. ⚠️ **작업은 일반 프로젝트 폴더에서** — `.claude` 등 민감 경로엔 쓰기가 차단됩니다. 터널 방식이면 furio 쓰는 동안 터널 유지.

### 완전 자동 실행 모드 (권한 확인 없이 끝까지)
기본은 Claude 처럼 도구 실행 전 **확인**합니다. 확인 없이 알아서 진행시키려면 두 가지 방법:

**① 한 번만(그 실행에만)** — 플래그로:
```bash
furio --dangerously-skip-permissions          # TUI, 모든 권한 프롬프트 생략(완전자동)
furio -p "리팩터링하고 테스트 돌려줘" --dangerously-skip-permissions   # 비대화형 1회 완전자동
furio --permission-mode acceptEdits           # 파일 편집만 자동(Bash 등 위험작업은 확인)
```

**② 늘 자동으로(영구)** — `FURIO_AUTO` 환경변수(설치기가 래퍼에 심어둠):
```bash
FURIO_AUTO=1 furio                # 이번 셸에서만 완전자동
echo 'export FURIO_AUTO=1' >> ~/.zshrc   # 항상 완전자동(새 터미널부터)
# 또는 설치 때 기본값으로 굽기:  FURIO_AUTO=1 SDI_SERVER=... bash install.sh
```
`FURIO_AUTO` 값: `1`/`yes`/`on`/`full`/`bypass` = 완전자동(`--dangerously-skip-permissions`), `edits`/`accept` = 편집만 자동(`--permission-mode acceptEdits`), 빈값 = 확인모드(기본).

> **껐다 켜기/정리**: `furio`·openclaude 는 맥에 설치된 로컬 명령이라 서버 stop 해도 안 지워집니다(정리 불필요). 다음엔 서버 `serve-router.sh start` + 맥 터널만. 완전 삭제는 `rm -rf ~/.furio ~/.local/bin/furio`.

---

## 4. 문제 해결

### ❗ `400 ... exceeds model maximum context length`
openclaude 의 시스템 프롬프트가 큽니다(~17.6k 토큰). 기본 출력토큰이 너무 크면 모델 ctx 를 넘습니다.
설치기가 **`CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192`** 를 자동 적용하지만, 직접 키웠다면 줄이세요:
```bash
CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192 furio
```
> ⚠️ **컨텍스트 16k 모델(`Qwen3-32B-FP8-16k`)은 openclaude 에 못 씁니다**(시스템 프롬프트만으로 16k 초과). ctx ≥ ~26k 모델만: Qwen3-32B(40960)·Coder-32B(32768)·a3b(65536)·Llama-70B(131072).

### `node: ... required` / Node 버전 오류
openclaude 는 Node ≥22 필요:
```bash
nvm install 22 && nvm use 22      # 또는 brew install node (Mac)
```
설치 후 `furio` 가 그 node 를 쓰도록 PATH 에 node≥22 가 잡혀 있어야 합니다.

### 모델이 도구 대신 권한/위치 얘기만 하고 파일을 안 만듦
`.claude` 같은 **민감 경로**에서 실행했을 때 납니다 — **일반 프로젝트 폴더**에서 실행하세요.

### `furio: command not found`
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc   # 후 새 터미널
```

### `서버 도달 실패` (설치/사용)
SSH 터널이 떠 있는지: `curl http://127.0.0.1:8400/v1/models` (모델 나오면 OK). 안 되면 `bash furio-connect.sh`. (자세히는 sdi-code/README.md 의 터널/`No route to host` 항목과 동일.)
