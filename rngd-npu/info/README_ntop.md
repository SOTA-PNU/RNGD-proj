# ntop / nctop — NPU·CPU 실시간 모니터

`furiosa-smi` 를 매번 직접 치는 대신 htop처럼 한 화면에서 보여줍니다.

```bash
ntop      # NPU 모니터 (1초마다 갱신)
nctop     # ntop + CPU 코어별 사용률 + 호스트 RAM
```

| 옵션 | 의미 |
|---|---|
| `-i 0.5` | 갱신 간격 0.5초 (기본 1.0) |
| `--no-ps` | 프로세스 패널 숨김 |
| `--cpu` / `--no-cpu` | CPU 패널 켜기/끄기 (`nctop` 은 켜진 채로 시작) |
| `--raw` | furiosa-smi 원본 JSON 디버그 |

종료: **q** 또는 **Ctrl+C**.

---

## 갱신 주기

- 디폴트 **1초**.
- `-i` 옵션으로 변경. 빠르게: `ntop -i 0.5` / 느리게: `ntop -i 2`.

---

## `ntop` 한 단어로 호출되는 원리

`ntop` 만 쳤을 때 어디서든 화면이 뜨도록 만들려면 **세 가지** 가 맞아야 합니다.

### 1. 파일 첫 줄에 "이 파일은 무슨 인터프리터로 실행할까?" 를 적는다 — **shebang**

`~/RNGD-proj/ntop.py` 의 첫 줄은 다음과 같습니다.

```python
#!/usr/bin/env python3
```

이 줄을 **shebang**(셔뱅) 이라고 부릅니다. 리눅스가 파일을 실행하라고 했을 때 첫 줄이 `#!` 로 시작하면, **그 뒤에 적힌 프로그램으로 이 파일을 열어준다는 약속**입니다.

- `/usr/bin/env python3` 이라고 적으면 → "PATH 에서 `python3` 찾아서 그걸로 이 파일을 실행해라"
- 만약 shebang 이 없으면 → 그냥 텍스트 파일로 취급되어 `./ntop.py` 가 동작하지 않음

bash 스크립트라면 `#!/usr/bin/env bash`, sh 라면 `#!/bin/sh` 식으로 적습니다.

### 2. 파일이 "실행 가능" 으로 표시돼 있어야 한다 — **실행 권한**

리눅스 파일에는 읽기(r) / 쓰기(w) / 실행(x) 세 가지 권한이 있습니다. `.py` 파일은 기본적으로 읽기·쓰기만 켜져 있어서 (`-rw-r--r--`), 그대로는 **명령처럼 실행되지 않습니다**. 다음 한 번으로 실행 권한을 켜줍니다.

```bash
chmod +x ~/RNGD-proj/ntop.py
```

이러면 권한이 `-rwxr-xr-x` 로 바뀌고, `./ntop.py` 처럼 직접 호출했을 때 shebang 덕에 python3 로 실행됩니다.

확인:

```bash
ls -la ~/RNGD-proj/ntop.py
# -rwxr-xr-x  ...  ntop.py     ← x 가 켜져있어야 OK
```

### 3. PATH 에 있는 곳에 심볼릭 링크를 둔다 — **어디서든 호출**

여기까지만 하면 `~/RNGD-proj/ntop.py` 라고 풀 경로를 쳐야 실행됩니다. 그냥 `ntop` 만으로 동작하게 하려면, **PATH 환경변수에 들어있는 디렉토리** 중 하나에 짧은 이름으로 링크를 만들어 둡니다.

`~/.local/bin` 이 PATH 에 들어있어서 (Ubuntu 디폴트, `~/.profile` 에서 처리됨) 거기에 심볼릭 링크 하나 만들면 끝.

```bash
ln -sfn ~/RNGD-proj/ntop.py  ~/.local/bin/ntop
```

확인:

```bash
which ntop
# /home/jun/.local/bin/ntop

readlink -f $(which ntop)
# /home/jun/RNGD-proj/ntop.py
```

### `nctop` 은 어떻게 다른 동작을 하나? — **argv[0] 트릭**

`~/.local/bin/nctop` 도 같은 `ntop.py` 를 가리키는 또 다른 심볼릭 링크입니다.

```bash
ln -sfn ~/RNGD-proj/ntop.py  ~/.local/bin/nctop
```

스크립트 안에서 "내가 어떤 이름으로 호출됐는지" 를 확인해서 동작을 바꿉니다.

```python
# ntop.py 안
import os, sys
invoked_as_nctop = os.path.basename(sys.argv[0]).lower().startswith("nctop")

# argparse 의 --cpu 옵션 default 를 위 변수로 줌
ap.add_argument("--cpu", action="store_true", default=invoked_as_nctop, ...)
```

즉 **한 파일에 두 명령**이 사는 셈입니다. `ntop` 으로 부르면 NPU 만, `nctop` 으로 부르면 CPU 패널까지 자동으로 켜짐.

---

## 만약 `~/.local/bin` 이 PATH 에 없다면

대부분의 Ubuntu/Debian 은 자동으로 들어있지만, 아니면 한 줄 추가:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

또는 `~/bin/` 디렉토리를 만들고 거기에 심볼릭 링크를 둬도 됩니다 (Ubuntu 가 다음 로그인부터 자동으로 PATH 에 추가).

---

## 데이터 출처

ntop 가 어디서 정보를 가져오는지:

- `furiosa-smi status --format json` — `pe_utilizations[]`, `memory.DRAM`
- `furiosa-smi info --format json` — `temperature`, `power`, `firmware`
- `furiosa-smi ps` — NPU 점유 프로세스
- `/proc/stat` (delta) — CPU 코어별 사용률
- `/proc/meminfo` — 호스트 RAM
- `/proc/loadavg`, `ps -eo`

JSON 파싱이 깨지면 `ntop --raw` 로 원본 확인.

---

## 파일 위치 정리

| 경로 | 무엇 |
|---|---|
| `~/RNGD-proj/ntop.py` | 본 코드 (Python + rich) |
| `~/RNGD-proj/ntop.sh` | bash + watch 버전 (rich 없이 원본 표 색칠) |
| `~/.local/bin/ntop` | `ntop.py` 심볼릭 링크 |
| `~/.local/bin/nctop` | `ntop.py` 심볼릭 링크 (argv[0] 으로 CPU 모드) |

의존성: `rich` (PyPI). 시스템 `/usr/bin/python3` 와 venv `~/furiosa/bin/python3` 양쪽에 이미 설치돼 있음.
