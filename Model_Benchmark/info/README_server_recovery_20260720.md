# 서버 복구(2026-07-20) 점검 결과

이 문서는 2026년 7월 20일 서버 복구 직후에 "문제가 없는지" 전체 점검한 결과를 정리한 것입니다. 무엇이 왜 깨졌는지, 지금 무엇이 정상이고 무엇이 위험한지, 어떤 순서로 손봐야 하는지를 담았습니다.

점검 방식은 읽기 전용이었습니다. 설정을 고치거나 파일을 지우거나 서비스를 재시작한 것은 없습니다. 다만 NPU가 실제로 계산을 하는지 확인하려고 별도 venv(`/home/jun/furiosa-3.0-test`)를 새로 만들어 추론을 돌려봤고, 기존 검증된 venv(`/home/jun/furiosa`)는 건드리지 않았습니다.

---

## 1. 한 줄 요약

NPU 하드웨어 4장은 **정상**이고 실제로 토큰 생성까지 확인했습니다. 대신 **재부팅하면 서버가 정상적으로 안 올라옵니다**(fstab 누락). 그리고 5~6월에 빌드해 둔 **아티팩트 전부가 지금 SDK에서는 실행이 안 됩니다**.

---

## 2. 무슨 일이 있었는지 (시간순)

로그로 재구성한 내용입니다.

| 시각 | 사건 | 근거 |
|---|---|---|
| 7/13 10:53 | 루트 파일시스템 재설치 | `/` birth 타임스탬프, `/var/log/installer` |
| 7/13 12:27~12:48 | NPU 4장 펌웨어를 2026.3.0으로 플래시 | `/var/log/furiosa_rngd_updater.log` |
| 7/7 18:00~18:13 | **`/dev/nvme1n1p2`에 e2fsck 실행**, 고아 inode 약 46개를 lost+found로 회수 | `dumpe2fs` Last checked = 7/7 18:13, lost+found mtime 7/7 18:00 |
| 7/17 03:00~03:08 | 정상 종료 기록 없이 다운, 약 8분 뒤 재부팅 | journal 로그가 03:00:19에 끊기고 03:08:50에 새 부팅 |
| 7/20 14:17~15:36 | 부팅 5회 반복 시도 | `last` |
| 7/20 20:30 | `mv .../lost+found/#103022593/jun → /mnt/nvme1n1p2/home_jun` | `auth.log` sudo 기록 |
| 7/20 20:50 | 기존 `/home/jun` → `/home/jun.pre-restore-20260720`으로 이름 변경, bind 마운트 연결 | 디렉터리 mtime |

즉 지금 쓰고 있는 `/home/jun`은 **fsck가 lost+found에서 건져낸 트리 그 자체**입니다. 복사가 아니라 같은 inode를 그대로 옮긴 것이라 복구 과정에서 유실된 것은 없습니다.

두 가지 정정 사항이 있습니다(2026-07-21 root 권한으로 재확인).

- **파일시스템이 깨진 시점은 7/20이 아니라 7/7입니다.** `dumpe2fs`의 Last checked가 7/7 18:13이고 lost+found 항목들의 mtime도 7/7 18:00입니다. 즉 7/7에 이미 깨져 fsck가 돌았고, 7/20에는 그걸 **발견해서 복구**한 것입니다.
- **7/17 다운의 원인은 확정하지 못했습니다.** 정상 종료 기록 없이 끊긴 것은 맞지만(journal이 03:00:19에 끊기고 03:08:50에 새 부팅), 그 시점 커널 로그가 이미 로테이션으로 사라져 원인을 특정할 근거가 남아 있지 않습니다. 초기 점검에서 "하드웨어 MCE"로 보고했으나 **재확인에 실패했으므로 그 판정은 취소합니다.**

---

## 3. 지금 정상인 것

- **NPU 4장 전부 정상.** 인식·liveness·47.50 GiB·38~39°C·39W 모두 정상이고, PCIe는 4장 모두 Gen5 x16 최대 속도로 붙었으며 AER 오류 0입니다.
- **NPU가 실제로 계산합니다.** 벤더 최신 모델 `furiosa-ai/Qwen3-4B-FP8`을 npu0에 올려 32토큰을 0.43초에 정상 생성했고, 문장도 멀쩡했습니다. 요청 중 `furiosa-smi`로 npu0가 46.03/47.50 GiB를 쓰고 PID가 코어를 잡은 것까지 확인했습니다.
- **git 저장소 손상 없음.** 홈 아래 저장소 12개 전부 `git fsck` 통과, 손상 객체 0개.
- **소유권 문제 없음.** `/home/jun` 아래 96만여 개 항목 중 jun이 아닌 소유자는 하나도 없습니다.
- **아티팩트 파일 자체는 멀쩡합니다.** 100MB 넘는 파일 327개(총 1.42TB)를 처음~끝까지 읽어봤고 safetensors 221개가 선언한 텐서 범위를 모두 만족했습니다. 파일이 깨진 게 아닙니다.
- 실패한 systemd 유닛 없음, alpamon(Alpacon) 정상 동작, SSH 호스트 키 그대로, 네트워크·DNS 정상.

---

## 4. 반드시 손봐야 하는 것

### 4-1. ✅ 조치 완료 (2026-07-21) — 재부팅하면 홈이 사라지고 부팅이 멈추던 문제

`/etc/fstab`에 bind 줄은 있는데, **그 bind가 가리키는 원본 디스크를 마운트하는 줄이 없습니다.**

```
UUID=6bb00774-...  /mnt/nvme2n1p1  ext4  defaults  0 2
/mnt/nvme1n1p2/home_jun  /home/jun  none  bind  0 0      ← 원본이 없음
```

지금 `/mnt/nvme1n1p2`는 손으로 마운트한 상태라서 재부팅하면 사라집니다. systemd도 이걸 fstab이 아니라 현재 마운트 표에서 주워온 것으로 인식합니다(`FragmentPath=` 비어 있음, `SourcePath=/proc/self/mountinfo`). 그러면 bind 원본 경로가 없으니 `home-jun.mount`가 실패하고, `nofail`이 없어서 부팅이 emergency 모드로 떨어집니다. 시뮬레이션으로 확정한 사항입니다.

**고치는 법 — 재부팅 전에 하세요.** bind 줄 위에 원본 마운트를 추가합니다.

```bash
# /etc/fstab 에 bind 줄 '위'에 추가
UUID=0942cf73-afd9-4e61-a567-e4fb0725e0b1  /mnt/nvme1n1p2  ext4  defaults,nofail  0 2

sudo systemctl daemon-reload
sudo findmnt --verify --verbose
sudo mount -a          # 오류 없이 통과하는지 확인
```

**적용 결과 (2026-07-21 02:15~02:18, 스크립트 `~/fix_fstab_home_mount.sh`·`~/fix_fstab_nvme2_nofail.sh`, 백업 `/etc/fstab.bak.20260721-*`)**

최종 fstab은 이렇게 됐습니다.

```
UUID=6bb00774-...  /mnt/nvme2n1p1  ext4  defaults,nofail  0 2
UUID=0942cf73-...  /mnt/nvme1n1p2  ext4  defaults,nofail  0 2
/mnt/nvme1n1p2/home_jun  /home/jun  none  bind,nofail  0 0
```

재부팅 없이 확인한 증거입니다.

- `mnt-nvme1n1p2.mount`의 출처가 `SourcePath=/proc/self/mountinfo`(손으로 마운트) → **`SourcePath=/etc/fstab` + `FragmentPath=/run/systemd/generator/mnt-nvme1n1p2.mount`** 로 바뀜. 이게 "재부팅해도 살아난다"의 직접 증거입니다.
- `home-jun.mount`가 `Requires=mnt-nvme1n1p2.mount`, `After=mnt-nvme1n1p2.mount`, `RequiresMountsFor=/mnt/nvme1n1p2/home_jun`을 갖게 됨 → 원본이 먼저 마운트된 뒤 bind가 걸립니다. 순서 보장됩니다.
- `findmnt --verify` 오류 0, `mount -a` 오류 0, 홈 내용(56개 항목, RNGD-proj·.ssh) 그대로.

**덤으로 같이 고친 것**: `nvme2n1p1`이 `defaults`라서 `local-fs.target`의 **Requires**에 걸려 있었습니다. 즉 그 빈 디스크가 빠지면 부팅이 emergency로 떨어지는 상태였습니다. `nofail`을 줘서 지금은 세 마운트 모두 **Wants**로만 걸립니다 — 디스크가 하나 죽어도 부팅은 되고 SSH로 들어와 고칠 수 있습니다.

`nofail`을 넣은 이유는, 없으면 문제가 생겼을 때 콘솔 앞에 사람이 가야 하기 때문입니다. 대신 디스크가 안 붙으면 `/home/jun`이 빈 디렉터리로 뜨니, 부팅 후 홈이 비어 보이면 `findmnt /home/jun`부터 확인하세요.

### 4-2. 옛 아티팩트가 전부 실행 불가입니다 (NPU 고장이 아닙니다)

5~6월에 빌드한 아티팩트는 **로드는 되는데 실행에서 멈춥니다.** 4장 전부, 아티팩트 2종 전부에서 동일하게 `PXI-601 (Execution: Timeout)`이 나고 엔진이 죽으며 생성 토큰이 0입니다.

원인을 통제 비교로 확인했습니다. 같은 스택·같은 카드·같은 요청에서

| 대상 | 빌드한 llm rev | 결과 |
|---|---|---|
| `qwen2.5-coder-1.5b-inst-tp8` (5~6월) | `9f92da0` / 컴파일러 `5c885c73ee` | 0토큰, PXI-601 |
| `furiosa-ai/Qwen3-4B-FP8` (벤더 최신 fxb) | `2026.3.0-rc.5` / IR `2026.3.0` | **32토큰, 0.43초 정상** |

즉 로드 호환성과 실행 호환성은 별개이고, 옛 아티팩트는 2026.3.0 런타임에서 실행 단계가 안 맞습니다.

**고치는 법:** `Model_Benchmark/rngd-npu/artifacts/` 아래 아티팩트를 furiosa-llm 2026.3.0으로 **다시 빌드**해야 합니다. 위장(masquerade)한 30B 아티팩트와 qwen3-next host-loop 쪽도 같은 옛 컴파일러 스탬프를 달고 있어서 함께 영향을 받습니다. 당장 서비스가 필요하면 `/home/jun/.cache/furiosa/llm/fxb/`의 벤더 번들은 지금 바로 돕니다.

곁들여 알아둘 점 두 가지입니다.

- **venv 버전은 건드리지 마세요.** `furiosa-models 2026.2.0`, `furiosa-torch 2026.2.0`은 잘못된 게 아니라 2026.3.0이 요구하는 정상 핀입니다. 깨끗한 venv를 새로 만들어도 같은 버전이 나옵니다. `pip check`가 투덜대는 `furiosa-native-runtime 2026.2.0`은 옛 설치의 잔재이고 serve 경로에서 쓰이지 않습니다(제거해도 동작은 그대로).
- **엔진이 죽어도 HTTP는 200을 돌려줍니다.** `finish_reason: "stop"`에 `content: ""`, `completion_tokens: 0`이라 클라이언트 입장에선 "성공했는데 답이 비었다"로 보입니다. chat_app·coding-agent·robot-sim 같은 도구가 NPU 전면 장애를 "모델이 답을 잘 못한다"로 오해할 수 있으니 주의하세요.

### 4-3. ARM64 크로스 컴파일러가 없어 새 컴파일이 아예 안 됩니다

NPU 컴파일러는 PE 코어용 C 코드를 만든 뒤 `aarch64-linux-gnu-gcc`를 호출하는데, 이 바이너리가 머신에 없습니다(`gcc-aarch64-linux-gnu` 미설치, dpkg에 설치 기록도 없음). 7/13 루트 재설치 때 빠진 것으로 보입니다.

증상이 `furiosa.UnsupportedOpError: failed to compile the graph`로 나와서 연산 미지원처럼 보이지만, 실제로는 `No such file or directory (os error 2)`입니다. 4-2의 아티팩트 재빌드도 이것부터 깔아야 가능합니다.

```bash
sudo apt install gcc-aarch64-linux-gnu
```

단, 루트 디스크가 꽉 차 있어 설치가 중간에 실패할 수 있으니 4-4를 먼저 처리하는 편이 낫습니다.

### 4-4. 루트 디스크가 100%이고, 그 원인 디렉터리를 그냥 지우면 안 됩니다

`/dev/nvme0n1p3`(323G)가 100%이고 `/home/jun.pre-restore-20260720`이 280G를 차지합니다. ext4 예약 블록 덕에 root 권한으로는 아직 18G 정도 여유가 있어 당장 멈추지는 않지만, 일반 사용자 작업·패키지 설치·로그 기록은 실패합니다.

**그냥 지우면 안 되는 이유**가 있습니다. 이 디렉터리에는 라이브 홈에 **없는** 데이터가 약 260G 있습니다.

- HuggingFace 모델 4종 약 97G — `gpt-oss-120b`(62G), `Qwen3-Coder-30B-A3B-Instruct-FP8`(30G), `Qwen3-4B-FP8`(5.4G), `Llama-3.1-8B-Instruct`
- `gh` CLI 인증 정보 (`.config/gh/hosts.yml`) — 지우면 GitHub 로그인이 풀립니다
- 7/14~7/20 사이 Claude 작업 기록 159개 (`.claude/projects`, `jobs`, `sessions`)
- RNGD-proj의 커밋 2개와 미커밋 수정 3개 (아래 4-5)

반대로 라이브 홈에만 있는 것도 많습니다(ACCV용 timm/DINOv2·DINOv3 계열, SWE-bench 데이터셋, Qwen2.5-Coder 계열 등). 두 트리는 포함관계가 아니라 **서로 보완 관계**입니다.

**권장 순서:** 지우지 말고 옮기세요. 비어 있는 `/mnt/nvme2n1p1`(1.9T, 이미 fstab 등록됨)로 통째로 옮기면 루트가 즉시 풀리고 한 바이트도 안 잃습니다.

```bash
rsync -aHAX --info=progress2 /home/jun.pre-restore-20260720/ /mnt/nvme2n1p1/pre-restore-20260720/
# 검증 후에만 원본 삭제
```

옮기기 전에 **code-server / alpamon 에디터 세션을 먼저 껐다 켜세요.** 지금 프로세스 9개가 이 디렉터리를 작업 디렉터리로 잡고 열린 파일을 쓰고 있어서, 그대로 옮기면 세션이 깨지고 쓰던 내용이 조용히 사라집니다. `lsof +D /home/jun.pre-restore-20260720`가 비면 안전합니다.

### 4-5. RNGD-proj 히스토리가 갈라져 있습니다

- 라이브: `add-extra-models` 브랜치, `3137024`(7/4)
- pre-restore 사본: 7/14에 새로 clone한 것, `main`에 `b4c7d84`("furiosa/router 복구")와 `c52adbd`(7/16) 포함

커밋 2개는 GitHub에 푸시돼 있어 `git fetch`로 되찾을 수 있습니다. 하지만 **pre-restore 사본에만 있는 미커밋 수정 3건**(`furiosa_router.py` 등)은 다른 곳에 없습니다. 지우기 전에 반드시 뽑아 두세요.

```bash
git -C /home/jun.pre-restore-20260720/RNGD-proj diff > /home/jun/pre-restore-uncommitted.patch
git -C /home/jun/RNGD-proj fetch origin
```

덧붙여, 현재 작업물이 **한 번 깨진 적 있는 디스크에만** 있고 백업이 없습니다. `add-extra-models` 브랜치를 지금 푸시해 두시길 권합니다.

### 4-6. 개인키와 토큰이 다른 사용자에게 그대로 보입니다

복구된 트리는 권한이 전부 뭉개져서 파일 약 98.6%가 0755/0775가 됐습니다. 원래 실행 파일이 아닌 것에도 +x가 붙고, 700이어야 할 디렉터리가 755가 됐습니다.

문제는 이 머신에 실계정이 넷(`pusan`, `username`, `jun`, `chacha`) 있다는 점입니다.

| 대상 | 현재 권한 | 정상 |
|---|---|---|
| `/home/jun` | 755 | 750 |
| `~/.ssh` | 755 | 700 |
| `~/.ssh/id_ed25519` (**암호화 안 된 개인키**) | 755 | 600 |
| `~/.bash_history` | 755 | 600 |
| `~/.cache/huggingface/token` | 775 | 600 |

즉 **다른 로컬 계정이 jun의 SSH 개인키와 HuggingFace 토큰을 읽을 수 있습니다.** 게다가 `.bash_history`에는 `hf auth login --token hf_...` 명령이 평문으로 남아 있습니다.

한 가지 정정할 점은, SSH 접속 자체가 끊긴 것은 아니라는 겁니다. 실제로 확인해 보니 ssh 클라이언트는 키를 정상적으로 제시합니다(`Offering public key ... explicit`). 경고를 내는 쪽은 `ssh-keygen`입니다. OpenSSH 9.6은 "다른 사용자가 **쓸 수** 있는" 키를 거부하는데 755는 읽기만 되기 때문입니다. 그러니 **장애는 아니고 노출**입니다 — 그래서 오히려 조용히 지나가기 쉽습니다.

```bash
chmod 750 /home/jun
chmod 700 ~/.ssh && chmod 600 ~/.ssh/id_ed25519 ~/.ssh/config ~/.ssh/authorized_keys
chmod 644 ~/.ssh/*.pub ~/.ssh/known_hosts
chmod 600 ~/.bash_history ~/.claude.json ~/.cache/huggingface/token
chmod 700 ~/.cache ~/.config ~/.local
```

노출된 시간이 있으므로 **SSH 키와 HuggingFace 토큰은 새로 발급**하시길 권합니다. 키는 GitHub과 GPU 서버(`jun@164.125.249.13`)에 등록돼 있습니다.

---

## 5. 알아두면 좋은 것 (급하지 않음)

- **시계가 8분 빠릅니다.** 외부 UDP/123이 막혀 있어 NTP가 스스로 못 맞춥니다. TLS·git·패키지 설치에서 문제가 될 수 있으니 사내 NTP 서버를 지정하거나 수동으로 맞추세요.
- **lost+found에 133개 항목(약 286G)이 이름을 잃은 채 남아 있습니다.** 확인해 보니 예전 우분투 루트 파일시스템의 잔해입니다(`/usr` 6.6G, `/var` 15G, `/etc`, `/boot`, `/snap`, `/tmp`). 그중 `#17`(200G)과 `#18`(64G)은 **옛 swap 파일**이라 지워도 됩니다 — 이 둘만으로 264G가 풀립니다. jun의 데이터는 아니지만 pusan 소유 항목이 3개 있으니 머신 주인과 상의 후 정리하세요.
- **디스크 감시가 없습니다.** smartmontools가 설치돼 있지 않아 마모나 미디어 오류를 아무도 경고해 주지 않습니다. 1.36TB 작업 세트가 이중화 없는 디스크 한 장에 있으니 설치를 권합니다.
- **nvme2n1(RevuAhn NX2200)은 백업용으로만 쓰세요.** 컨슈머 등급 PCIe 3.0 x4라서, 데이터센터급 PCIe 5.0 x4인 Solidigm(nvme1n1)의 대체가 아니라 백업 대상으로 적합합니다.
- **일부 심볼릭 링크가 끊겨 있지만 복구 탓이 아닙니다.** RNGD-proj의 15개는 `Qwen3-VL-30B-A3B-Instruct-FP8` HF 캐시가 없어서이고, 나머지는 snap 관련으로 원래 그렇습니다.
- **`import furiosa.torch`가 실패합니다.** 복구 손상이 아니라 원래 있던 문제이고, `import torch`를 앞줄에 두면 됩니다.

---

## 6. 권장 처리 순서

1. `/etc/fstab`에 `nvme1n1p2` 마운트 추가 → `daemon-reload` → `findmnt --verify` (**재부팅 전 필수**)
2. 에디터 세션 종료 → pre-restore 미커밋 패치 추출 → `git fetch origin`
3. `/home/jun.pre-restore-20260720`을 `/mnt/nvme2n1p1`로 이동 → 루트 공간 확보
4. 권한 원복 → SSH 키·HF 토큰 재발급
5. `gcc-aarch64-linux-gnu` 설치 → 아티팩트 2026.3.0으로 재빌드
6. smartmontools 설치, NTP 정리, lost+found 정리

---

## 7. 디스크 최종 판정 (2026-07-21 root 권한으로 확인 완료)

초기 점검 때 sudo가 막혀 못 봤던 결정적 수치를 모두 확인했습니다. **세 디스크 모두 건강합니다.**

| | 모델 | 마모 | 미디어 오류 | 예비 블록 | 가동시간 | 비정상 종료 | 오류로그 |
|---|---|---|---|---|---|---|---|
| nvme0 | Samsung 970 PRO 512G (OS) | 0% | 0 | 100% | 380h | 30 | 104 |
| nvme1 | Solidigm 1.92T (홈) | **0%** | **0** | 100% | 2007h | 38 | **0** |
| nvme2 | RevuAhn 2T (신규) | 0% | 0 | 100% | 613h | 31 | 946 |

**디렉터리 트리가 통째로 날아갔던 nvme1(Solidigm)이 마모 0%, 미디어 오류 0건, 오류 로그 0건입니다.** 즉 이번 사고는 **디스크 고장이 아닙니다.** 계속 주 데이터 디스크로 쓰셔도 됩니다. ext4 상태도 세 개 모두 `clean`이고 오류 카운터가 없습니다.

nvme2의 오류 로그 946건은 미디어 오류가 아니라 명령 중단류 카운트이고 critical_warning은 0이지만, 컨슈머 등급이라는 점과 함께 **백업 전용으로 쓰라**는 판단을 뒷받침합니다.

root 예약 작업(`crontab -l -u root`)은 없었습니다.

## 8. 새로 발견한 것 — PCIe 오류가 계속 나고 있습니다

현재 커널 로그에 **`Hardware Error` 39,000건 이상**이 쌓여 있고 지금도 계속 늘고 있습니다.

- 발생 장치: `0000:ea:00.1` = **Intel Ethernet Controller X550** (온보드 10GbE 랜카드)
- 종류: `aer_cor_status: 0x00001000` — **정정 가능(correctable)** 오류라 통신은 되고 크래시도 안 납니다
- **NPU 4장(BDF `0b/59/b4/c6:00`)에서는 0건** — NPU와 무관합니다

당장 장애는 아니지만 두 가지가 걸립니다. 첫째, 이 랜카드의 PCIe 링크가 정상은 아니라는 신호입니다(재장착이나 슬롯 교체로 대개 잡힙니다). 둘째, `kern.log`와 `syslog`가 각각 **시간당 약 1MB씩** 이 오류로 불어나는데 그 로그가 **꽉 찬 루트 디스크**에 쌓입니다. 지금 여유가 1.4G라 당장은 아니어도 계속 갉아먹습니다.

급하게 로그만 막으려면 아래로 해당 장치의 AER 보고를 끌 수 있습니다(원인 자체를 고치는 건 아닙니다).

```bash
sudo setpci -s ea:00.1 ECAP_AER+0x08.L=0xffffffff   # 정정가능 오류 보고 마스크
```
