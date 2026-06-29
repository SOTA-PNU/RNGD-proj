# sdi code — 실행 명령 모음

서버 리눅스에서 **셸 작업을 하지 않고**, 각자 Mac/Windows에서 `sdi` 만 치면 **Claude Code 같은 터미널 코딩 에이전트**.
추론만 **서버 NPU**(OpenCode 백엔드 + 우리 라우터 `:8400`)에서 돕니다. 코딩(파일·도구 실행)은 **내 PC 로컬**. 배경·한계는 맨 아래.

```
[내 Mac/Win]  sdi (로컬 실행) ──► localhost:8400 ──[SSH 터널 :10022]──► [서버] furiosa_router → furiosa-llm serve(NPU 4장)
```

> ## 🌐 원격 접속 = SSH 터널 (부산대 서버: 외부 입구는 SSH 10022 뿐)
> 이 서버는 **외부에서 들어오는 입구가 SSH(`164.125.19.138` : `10022`, 계정+비밀번호)뿐**이고,
> 라우터 `:8400` 은 인터넷에 직접 안 열려 있습니다(실측: SSH `0.0.0.0:10022` 리슨, 공인 IP=164.125.19.138, 내부 IP=10.125.19.138).
> 그래서 원격에선 **SSH 터널**로 내 PC의 `localhost:8400` 을 서버 라우터에 연결해서 씁니다.
> - **원격(집/외부) — 권장**: SSH 터널 → `SDI_SERVER=http://127.0.0.1:8400`   (아래 **2-A**)
> - **사내 같은 LAN/서브넷**: `http://10.125.19.138:8400` 직접 (터널 불필요, **2-B**)
> - (관리자가 8400 을 공인에 포워딩+TLS 했다면 `http://164.125.19.138:8400` 직접도 가능하나, 기본 구성은 아님)
>
> 💡 SSH 터널은 **암호화 + SSH 로그인 인증**이라, 라우터를 인터넷에 평문으로 여는 것보다 안전합니다(권장 이유).
> "서버에 SSH로 들어가 작업"하는 게 아니라 **포트포워딩(`-N`)만** 켜 두고, 코딩은 **내 PC에서 로컬**로 합니다.

---

## 1. 서버 관리자 — 라우터 기동 (1회)

```bash
cd ~/RNGD-proj/Model_Benchmark/rngd-npu/coding-agent
bash serve-router.sh start                               # 라우터 기동(인증 OFF — 터널 뒤라면 OK)
SDI_API_KEY="$(openssl rand -hex 24)" bash serve-router.sh start   # 인증 ON(8400을 공인 노출할 때 필수)
bash serve-router.sh stop                                # 라우터 + 백엔드 serve 종료
curl -s localhost:8400/v1/models | python3 -m json.tool  # 동작 확인
```
> 터널 방식에선 8400을 인터넷에 안 여니 인증 OFF여도 무방(SSH 로그인이 이미 관문). 8400을 공인 직노출하면 **인증 ON 필수**.

## 2. 사용자 — 각자 PC

> **순서**: `install.sh`·`sdi` 는 서버 라우터가 **떠 있어야** 동작합니다(서버에서 1번 먼저). **SSH 터널은 순서 무관**(라우터 꺼져 있어도 터널은 열림 — 라우터 뜨면 그때부터 전달). 라우터는 상주 서비스라 한 번 켜두면 다음엔 맥에서 터널+sdi 만 하면 됩니다.

### 2-A. 원격(집/외부) — SSH 터널 (권장)
**터미널 ①** — 터널을 켜 두기(이 창은 유지). 헬퍼 또는 raw ssh 중 택1:
```bash
SDI_SSH_USER=jun bash sdi-connect.sh
#   (= ssh -p 10022 -N -L 8400:localhost:8400 jun@164.125.19.138)  ← 비밀번호 입력
```
> 💡 **다른 서버 계정**으로 쓰려면 `jun`(또는 `SDI_SSH_USER`)**만** 그 계정으로 바꾸면 됩니다 — 계정은 SSH 로그인에만 쓰여서 install.sh·`sdi`·설정·라우터는 그대로입니다. 그 계정이 10022로 SSH 접속만 되면 됩니다.

**터미널 ②** — 터널(localhost:8400)로 설치/사용:
```bash
SDI_SERVER=http://127.0.0.1:8400 bash install.sh   # 최초 1회 설치(서버 인증 ON이면 SDI_API_KEY=<키> 추가)
sdi                                                # 사용
```
> Windows 도 동일(OpenSSH 기본 내장). PowerShell 터미널 ①: `ssh -p 10022 -N -L 8400:localhost:8400 jun@164.125.19.138`
> 터미널 ②: `$env:SDI_SERVER="http://127.0.0.1:8400"; powershell -ExecutionPolicy Bypass -File install.ps1`
> 💡 비밀번호 반복이 싫으면 키 등록: `ssh-copy-id -p 10022 jun@164.125.19.138` (이후 무입력). 백그라운드로 두려면 `ssh -f -N ...`.

### 2-B. 사내 같은 LAN — 직접 (터널 불필요)
```bash
SDI_SERVER=http://10.125.19.138:8400 bash install.sh        # 사내 사설 IP 직접
SDI_SERVER=http://10.125.19.138:8400 sdi                    # (설치 후) 그냥 sdi
```

> 배포 파일: `install.sh`(+원격이면 `sdi-connect.sh`) / Windows `install.ps1`. 설치 후 PATH 경고가 뜨면 `~/.local/bin` 을 PATH에 추가.

## 3. 사용

```bash
sdi                       # 코딩 에이전트 TUI (추론=서버 NPU, 코딩=로컬)
sdi run "버그 고쳐줘"        # 비대화형 한 줄
sdi models                # 서버 모델 목록
sdi agent list            # 에이전트(빌트인+커스텀)
sdi run --agent reviewer "이 변경 리뷰해줘"
```
TUI 안: `/models`(모델 전환) · `@reviewer …`(서브에이전트) · `Tab`(primary 전환). ※ sdi 쓰는 동안 터미널①의 터널은 켜져 있어야 함.

> **껐다 켜기 / 정리**: `sdi` 는 맥에 설치된 **로컬 명령**(래퍼 + opencode 바이너리 + 설정파일)이라, 서버에서 `serve-router.sh stop` 해도
> 명령 자체는 그대로 남습니다 — **맥에서 따로 정리할 필요 없습니다.** 단 서버(또는 터널)가 꺼져 있으면 `sdi` 창은 열려도 **답변은 안 됩니다**(추론할 NPU에 못 닿음).
> 다음에 다시 쓰려면 🖥️ 서버 `serve-router.sh start` + 💻 맥 **터널만** 켜면 끝(install 은 최초 1회뿐). 완전히 지우려면 5번 제거 명령.

---

## 4. 문제 해결

### ❗ `No route to host` / `서버 도달 실패` (설치 [2/3])
원격에서 `http://164.125.19.138:8400` 또는 `http://10.125.19.138:8400` 에 **직접** 붙으면 납니다 — 8400은 인터넷에 안 열려 있고,
10.x는 사내 전용이라서요. **해결 = SSH 터널 쓰기(2-A)**:
```bash
# 터미널①
SDI_SSH_USER=jun bash sdi-connect.sh        # 또는: ssh -p 10022 -N -L 8400:localhost:8400 jun@164.125.19.138
# 터미널②
SDI_SERVER=http://127.0.0.1:8400 bash install.sh
```
- 터널 ssh 가 안 붙으면(타임아웃/refused): SSH 포트/계정 확인 — `ssh -p 10022 jun@164.125.19.138` 로 로그인부터 되는지 점검.
- 사내 LAN이면 터널 없이 `http://10.125.19.138:8400` 직접(2-B).

### ❗ `bind [127.0.0.1]:8400: Address already in use` (터널 켤 때)
맥의 로컬 8400 포트가 **이미 사용 중**입니다 — 보통 **이전에 띄운 터널이 아직 살아 있어서**입니다(즉 이미 연결돼 있을 수 있음).
```bash
# ① 이미 터널이 떠 있는지 확인 — 되면 새 터널 없이 그냥 sdi
curl http://127.0.0.1:8400/v1/models       # 모델 목록 나오면 OK → 바로  sdi

# ② 끊고 다시 열기 (맥에서)
lsof -nP -iTCP:8400 -sTCP:LISTEN            # 8400 잡은 PID 확인
kill <PID>                                 # 또는:  pkill -f '8400:localhost:8400'
ssh -p 10022 -N -L 8400:localhost:8400 jun@164.125.19.138

# ③ 충돌 피해 다른 포트로
SDI_LOCAL_PORT=8401 bash sdi-connect.sh             # 헬퍼(다른 포트)
SDI_SERVER=http://127.0.0.1:8401 bash install.sh    # 설치/사용도 8401
```
> `sdi-connect.sh` 는 이제 이미 연결돼 있으면 "이미 연결됨"으로 알리고 새 터널을 안 띄웁니다.

### `HTTP 401` (설치 [2/3])
서버 인증 ON. 키 추가:
```bash
SDI_SERVER=http://127.0.0.1:8400 SDI_API_KEY=<받은키> bash install.sh
```

### `sdi: command not found`
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc   # (또는 ~/.bashrc) 후 새 터미널
```

### `sdi` 실행 시 연결 안 됨
터미널①의 **터널이 꺼졌을 때** 입니다 — `sdi-connect.sh` 를 다시 켜세요(터널은 sdi 쓰는 내내 유지돼야 함).

### 새 서버 모델이 `sdi models` 에 안 보임 / 첫 응답이 한참 멈춤
- 모델 목록은 설치 시점에 굳음 → 터널 켠 뒤 `SDI_SERVER=http://127.0.0.1:8400 bash install.sh` 재실행.
- 큰 모델 첫 요청은 NPU 콜드스타트(최대 ~8분, `READY_TIMEOUT=480s`) 동안 무응답일 수 있음. warm 모델은 빠름.

---

## 5. 옵션 명령

```bash
# 리브랜딩(명령·표시이름 변경)
SDI_CMD=acme SDI_BRAND="Acme Code" SDI_SERVER=http://127.0.0.1:8400 bash install.sh   # 명령 acme, id acme/...

# 터널 옵션(계정/호스트/포트/로컬포트 바꾸기)
SDI_SSH_USER=<계정> SDI_SSH_HOST=164.125.19.138 SDI_SSH_PORT=10022 SDI_LOCAL_PORT=8400 bash sdi-connect.sh

# 키 회전 / 제거
SDI_SERVER=http://127.0.0.1:8400 SDI_API_KEY=<새키> bash install.sh
rm -rf ~/.config/sdi ~/.local/bin/sdi

# 멀티에이전트 프리셋(역할별 에이전트) — 자세히는 agents-preset/README.md
cp agents-preset/*.md <프로젝트>/.opencode/agents/      # 또는 ~/.config/opencode/agents/
```

## 6. 네트워킹(보안)

| 방법 | SDI_SERVER | 보안 | 비고 |
|---|---|---|---|
| **SSH 터널(10022)** — 권장 | `http://127.0.0.1:8400` | ✅ 암호화 + SSH 인증 | 외부 입구가 SSH뿐인 현 구성에 맞음 |
| VPN(Tailscale 등) | `http://100.x.x.x:8400` | ✅ 암호화·비공개 | 관리자가 VPN 구축 시 |
| 사내 LAN 직접 | `http://10.125.19.138:8400` | ⚠️ HTTP 평문 | 신뢰망 전용 |
| 공인 직노출 | `http://164.125.19.138:8400` | ❌ 평문+현재 인증OFF | **권장 안 함**. 하려면 TLS 리버스프록시+인증 ON |

⚠️ 라우터엔 TLS가 없습니다. **원격은 SSH 터널(또는 VPN)** 을 쓰세요. 8400을 공인 평문으로 직접 열지 마세요.

---

## 7. 배경 · 한계 · 검증 (요약)

OpenCode(MIT)는 **OpenAI 호환 엔드포인트 클라이언트**. `sdi` = opencode 바이너리 + `OPENCODE_CONFIG`(서버주소·키·모델 격리) 래퍼.
원격 추론만 서버 NPU로 가고, 도구 실행·파일은 내 PC 로컬(= Claude Code 와 동일한 구조).

- **서버 접속(실측)**: 외부 입구 = SSH `0.0.0.0:10022`(`sshd_config Port 10022`). 공인 IP `164.125.19.138` ↔ 내부 `10.125.19.138`(NAT). 라우터 `0.0.0.0:8400`.
- **인증 on/off**: 서버 `SDI_API_KEY` 설정 시 사용자도 키 필요(401/200). 현재 OFF — 터널 뒤라 SSH 로그인이 관문 역할.
- **switch model**: 클라가 모델 바꾸면 라우터가 해당 모델을 NPU에 실제로 띄움(모델별 파서) — 실측.
- **한계**: ① 콜드스타트 직렬화(큰 모델 첫 요청 ≤480s) ② 공유 NPU 4장 — 서로 다른 모델 동시 사용 시 LRU 교체로 진행중 응답 끊길 수 있음(소규모·공통모델 권장) ③ 단일 공유 키 ④ 클라 ctx 휴리스틱 추정 ⑤ TLS 없음(원격은 터널/VPN) ⑥ sdi 쓰는 동안 터널 유지 필요.
- 8-에이전트 적대적 리뷰 22지적 반영 + 인증 매트릭스·리브랜딩·agents 실측.
