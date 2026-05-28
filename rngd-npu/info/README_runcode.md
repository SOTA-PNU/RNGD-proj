# 서버 띄우고 호출하기 — `furiosa-llm serve` + curl/SDK

빌드된 artifact를 `furiosa-llm serve`로 띄운 뒤, curl이나 OpenAI SDK 같은 HTTP 클라이언트로
호출해 측정·검증하는 방법을 정리한 참고서입니다. 앞으로 서버 띄우기·호출·테스트 관련 정보는
이 문서에 누적합니다.

소스 인용 위치는 모두 `~/furiosa/lib/python3.12/site-packages/furiosa_llm/`을 기준,
SDK 2026.2.0.

---

## 한눈에 보는 흐름

```
[ 빌드 끝난 artifact ]
        │
        │  furiosa-llm serve <path> --devices npu:0
        ▼
[ HTTP 서버 :8000 (OpenAI 호환 API) ]
        │
        │  POST /v1/chat/completions
        ▼
[ curl  /  OpenAI Python SDK  /  HTTP 라이브러리 ]
```

엔드포인트:
- `GET  /v1/models` — 등록된 모델 목록
- `POST /v1/chat/completions` — chat 메시지 → 응답 (대부분 쓰는 것)
- `POST /v1/completions` — 단순 텍스트 completion (legacy)
- `POST /v1/responses` — OpenAI Response API (Phase 1, 일부 지원)

---

## 1. 서버 띄우기

### 기본형

```bash
source ~/furiosa/bin/activate
furiosa-llm serve <artifact-경로> --devices npu:0 --port 8000
```

`Uvicorn running on http://0.0.0.0:8000` 떠야 준비 완료.

### 자주 쓰는 옵션

| 옵션 | 의미 |
|---|---|
| `--devices npu:0` | RNGD 1장 |
| `--devices npu:0,npu:1` | 2장. tp×pp가 1장 PE 안에 들어가면 자동 dp=2 |
| `--port 8000` | 포트. 다른 모델 동시 띄우려면 8001 등 |
| `--host 0.0.0.0` | 외부 접속 허용 (기본 localhost만 receive) |
| `--reasoning-parser {qwen3,exaone4,deepseek_r1}` | `<think>...</think>` 같은 reasoning tag를 별도 `reasoning` 필드로 분리 |
| `--tool-call-parser {hermes,llama4_json,llama3_json,openai}` | function calling 응답 파싱 |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | prefix 캐시 (기본 ON) |
| `--max-batch-size N` | 서버 한 번에 처리할 최대 배치 |
| `--max-model-len N` | artifact가 지원하는 범위 안에서 더 작게 자르기 |
| `--max-num-batched-tokens N` | continuous batching의 한 step당 토큰 한도 |
| `--chat-template <jinja>` | chat 템플릿 override (artifact에 없으면 필요) |

서버 종료: 그 터미널에서 `Ctrl-C` (한 번에 안 죽으면 한 번 더).

### HF id vs 로컬 경로 — prebuilt 서빙은 **항상 snapshot 경로**

prebuilt(`furiosa-ai/*`) 서빙 시 입력 형태별 결과:

| 입력 | 동작 | 결과 |
|---|---|---|
| HF id (`furiosa-ai/...`) | HF Hub `revision=v2026.2` 자동 검색 | **404 RevisionNotFoundError** (대부분 repo가 그 태그 없음) |
| 로컬 model 디렉터리 (`models--...--X/`) | 그 안엔 `blobs/`·`refs/`·`snapshots/`만, artifact 파일 없음 | tokenizer ValueError |
| **로컬 snapshot 디렉터리** (`snapshots/<hash>/`) | 정상 로드 | ✅ |

snapshot 자동 찾기 한 줄:
```bash
SNAP=$(find ~/.cache/huggingface/hub/models--furiosa-ai--Qwen2.5-Coder-7B-Instruct/snapshots -maxdepth 1 -mindepth 1 -type d | head -1)
furiosa-llm serve "$SNAP" --devices npu:0 --port 8001
```

`~/.bashrc`에 alias 만들어 두면 편함:
```bash
furserve() {
    local model=$1; shift
    local snap=$(find "$HOME/.cache/huggingface/hub/models--${model//\//--}/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1)
    [ -z "$snap" ] && { echo "캐시에 없음: $model"; return 1; }
    furiosa-llm serve "$snap" "$@"
}
# 사용: furserve furiosa-ai/Qwen2.5-Coder-7B-Instruct --devices npu:0 --port 8001
```

우리가 직접 빌드한 artifact는 항상 절대경로:
```bash
furiosa-llm serve ~/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen3-32b-fp8-tp8 --devices npu:0
```

---

## 2. curl로 빠른 테스트

### 등록된 모델 확인

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

결과의 `data[0].id`가 chat completion 요청 시 `"model"` 필드에 넣을 값.

### Chat completion 한 번

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
    "messages": [{"role":"user","content":"Write a Python function to reverse a string."}],
    "max_tokens": 256
  }' | python3 -m json.tool
```

응답의 `choices[0].message.content`가 모델 출력.

### 손쉽게 — 쉘 스크립트 두 개

`rngd-npu/`에 미리 만들어 둠. 단일 요청은 `run_model.sh`, 동시 N개 부하는 `run_model_p.sh`.

```bash
cd ~/RNGD-proj/Model_Benchmark/rngd-npu

# 단일 요청 — 응답 내용까지 출력 (verify·smoke test용)
./run_model.sh
PROMPT="Hello" MAXT=64 ./run_model.sh           # 옵션 override

# 동시 N개 부하 — 시스템 throughput 측정 (dp 비교용)
./run_model_p.sh
N=32 ./run_model_p.sh                            # 32개 동시
```

기본값(둘 다 동일): `PORT=8001, MODEL=Qwen/Qwen3-32B-FP8, MAXT=256, PROMPT="Write a Python function to reverse a string."`. 사전에 `furiosa-llm serve ... --port 8001`이 돌고 있어야 함.

---

## 3. curl 명령 부분별 의미

위 명령을 한 줄씩 풀어 보면:

| 부분 | 역할 |
|---|---|
| `curl` | HTTP 요청 보내는 명령 |
| `-s` | **silent.** 진행률·에러 출력 끔, 응답 body만 깔끔하게 |
| `http://127.0.0.1:8000/v1/chat/completions` | 보낼 URL. `127.0.0.1`=localhost, `8000`=서버 포트, `/v1/chat/completions`=OpenAI 호환 chat endpoint |
| `-H 'Content-Type: application/json'` | HTTP 헤더 — "이 요청 body는 JSON이야" 명시 |
| `-d '...'` | POST body (data). `-d`가 있으면 자동으로 POST 메소드. 작은따옴표 안의 JSON 문자열이 그대로 전송 |
| `\|` | pipe — curl 출력을 다음 명령의 입력으로 |
| `python3 -m json.tool` | Python `json.tool` 모듈. stdin의 JSON을 들여쓰기 정렬해서 출력 (안 거치면 한 줄로 빽빽해서 읽기 어려움) |

`-s` 없으면 다운로드 진행 표시줄이 응답 본문 위에 섞여 나옴. 늘 붙이는 게 깔끔.

---

## 4. JSON body 자세히

```json
{
  "model": "<model id>",          // /v1/models 결과의 id
  "messages": [                   // 대화 메시지 배열 (OpenAI 포맷)
    {"role": "system",    "content": "..."},   // 모델 가이드 (선택)
    {"role": "user",      "content": "..."},   // 사용자 입력
    {"role": "assistant", "content": "..."}    // 모델 이전 응답 (multi-turn 시)
  ],
  "max_tokens": 256,              // 응답 토큰 한도 (넘으면 잘림 → finish_reason="length")
  "temperature": 0.7,             // 0~2. 0=결정적, 높을수록 다양 (기본 1)
  "top_p": 0.9,                   // nucleus sampling (기본 1)
  "stream": false,                // true면 토큰 단위 streaming
  "tools": [...]                  // function calling용 (선택)
}
```

`messages`의 `role` 종류:
- `system` — 모델 행동 가이드. 한 대화에 보통 1개. 맨 앞 권장
- `user` — 사용자 입력
- `assistant` — 모델 응답. multi-turn 대화 history로 다시 넣을 때 사용
- `tool` — function 호출 결과

---

## 5. 자주 쓰는 변형

### 5.1 System prompt 추가

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"...",
    "messages":[
      {"role":"system","content":"You are a senior Python engineer. Reply only with code, no explanation."},
      {"role":"user","content":"Reverse a string."}
    ],
    "max_tokens":256
  }' | python3 -m json.tool
```

### 5.2 Streaming 응답 (토큰 단위 실시간)

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"...",
    "messages":[{"role":"user","content":"hi"}],
    "max_tokens":128,
    "stream": true
  }'
```

- `-N` (no-buffer): curl 출력 버퍼링 끔 — 토큰 도착하는 대로 화면에 표시
- `"stream": true`: 서버가 SSE(Server-Sent Events) 형태로 chunk별 응답

### 5.3 Multi-turn 대화

```json
"messages":[
  {"role":"system",   "content":"You are a helpful coder."},
  {"role":"user",     "content":"Write hello world in Python."},
  {"role":"assistant","content":"print('hello world')"},
  {"role":"user",     "content":"Now translate that to Rust."}
]
```
모델은 history 전체를 보고 다음 응답 생성. 매 턴 history를 누적해서 다시 보내야 함 (서버는 stateless).

### 5.4 Sampling 파라미터 조절

| 파라미터 | 효과 |
|---|---|
| `"temperature": 0` | 항상 같은 답 (결정적) |
| `"temperature": 1` | 기본 (적당히 다양) |
| `"temperature": 1.5+` | 더 창의적/난해 |
| `"top_p": 0.9` | 상위 90% 확률 토큰만 후보 |
| `"top_k": 50` | 상위 50개 토큰만 후보 (지원 시) |

---

## 5.5 동시 요청 부하 테스트 (dp=1 vs dp=2 직접 비교)

### 목적 — 한 줄

**dp=2가 정말 ~2배 빠른지 직접 측정해서 확인**하는 절차입니다. 정식 벤치마크는
`orchestrator.py` sweep이 자동으로 더 정교하게 하지만, 이건 dp 효과를 손으로 빠르게
검증·이해하기 위한 mini 실험.

### 실행 흐름 (총 ~15분)

```
[1] 터미널 A — dp=1로 서버 띄움 (1장)
     furiosa-llm serve <artifact> --devices npu:0 --port 8000

[2] 터미널 B — 부하 스크립트 실행, time 출력 기록  (예: real 30s)

[3] 터미널 A — 서버 종료 (Ctrl-C)

[4] 터미널 A — dp=2로 서버 다시 띄움 (2장, --devices만 바뀜)
     furiosa-llm serve <artifact> --devices npu:0,npu:1 --port 8000

[5] 터미널 B — 동일 부하 스크립트 다시 실행  (예: real 16s)

[6] 비교 → dp=1 30s vs dp=2 16s ≈ 1.9배 빠름 → dp=2 효과 정량 확인 ✓
```

### 알게 되는 것

1. **dp=2 실제 속도 향상 정량값** (이론 2배 vs 실측 1.7~1.9배)
2. **단일 요청은 dp 차이 없음** — N=1로 하면 dp=1 ≈ dp=2
3. **벤치마크 sweep의 동작 원리** — `batch_sizes` 축이 곧 N을 1·2·4·...·128로 변화시키는 것

이하 구체 스크립트와 메커니즘:

### 공통 — 부하 스크립트

**손쉽게:** `./run_model_p.sh` (`rngd-npu/run_model_p.sh`에 미리 만들어 둠) 한 줄이면 끝.
환경변수로 조절:
```bash
./run_model_p.sh                       # 기본 (N=8, PORT=8000, MAXT=128, MODEL=Qwen/Qwen3-32B-FP8)
N=16 ./run_model_p.sh                  # 동시 요청 16개
N=32 MAXT=256 PORT=8001 ./run_model_p.sh   # 여러 개 override
```
출력 예시:
```
── 결과 ──
  성공 응답          : 8 / 8
  총 소요 시간       : 17.34 s
  총 생성 토큰       : 1024
  시스템 throughput  : 59.1 tok/s
```

스크립트가 하는 일을 한 줄 식으로 풀면(직접 실행하고 싶다면):
```bash
N=8 PORT=8000 PROMPT="Write a Python function to reverse a string." MAXT=128
mkdir -p /tmp/dptest && rm -f /tmp/dptest/*

time (
  for i in $(seq 1 $N); do
    curl -s http://127.0.0.1:$PORT/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"Qwen/Qwen3-32B-FP8\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAXT}" \
      -o /tmp/dptest/resp_$i.json &
  done
  wait
)
```

`time (...)` → 모든 요청이 끝나는 데 걸린 총 시간 출력.

### 스크립트 동작 원리 — `&` · `wait` · `time`의 역할

```bash
time (                                    # ① 괄호 안 블록 전체의 wall-clock 시간 측정
  for i in $(seq 1 $N); do                # ② N번 반복 (i=1,2,...,N)
    curl -s ... -o /tmp/dptest/resp_$i.json &   # ③ 각 curl을 백그라운드로 던지고 즉시 다음으로
  done
  wait                                    # ④ 모든 백그라운드 작업 완료까지 블로킹
)
```

| 부분 | 의미 |
|---|---|
| **① `time (...)`** | bash 내장 명령 + 서브셸. 괄호 안 블록의 `real`(wall-clock)·`user`·`sys` 시간을 stderr로 출력. dp=1 vs dp=2 비교의 핵심 숫자 |
| **② `for i in $(seq 1 $N)`** | `$(seq 1 8)` → `1 2 3 4 5 6 7 8` 생성. `for`가 그걸 하나씩 `$i`에 담아 본문 실행 |
| **③ `&`** (가장 중요) | bash 백그라운드 실행 연산자. 명령 끝에 `&` 붙이면 그 명령을 백그라운드로 던지고 **즉시 다음 줄로 진행**. for 루프가 응답 안 기다리고 다음 i로 넘어감 → 1초 안에 N개 curl이 거의 동시에 떠 있음 |
| **④ `wait`** | 현재 셸의 모든 백그라운드 작업이 끝날 때까지 블로킹. 이게 없으면 `time` 괄호가 즉시 끝나서 측정 의미 없음 |

`&` 유무 차이:
```bash
for i in $(seq 1 8); do curl ... ; done    # & 없음 → 순차 (req1 끝→req2→...) → 약 72초
for i in $(seq 1 8); do curl ... & done    # & 있음 → 동시 (8개 거의 같은 순간 시작) → ~9초 (dp=1) / ~9~18초 (dp=2)
```

dp 비교가 의미 있으려면 **반드시 `&`로 동시 발사** 해야 합니다. 순차 실행은 dp 효과 안 보임.

### A. dp=1 (1장 서빙)

서버 (터미널 A):
```bash
furiosa-llm serve ~/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen3-32b-fp8-tp8 \
    --devices npu:0 --port 8000
```
다른 터미널에서 위 공통 부하 스크립트 실행. 서버 로그에 `[Engine 0]`만 보이고 `Running: 1 reqs`,
`Waiting: N-1 reqs`로 큐가 쌓이는 게 관찰됨.

### B. dp=2 (2장 서빙)

서버:
```bash
furiosa-llm serve ~/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen3-32b-fp8-tp8 \
    --devices npu:0,npu:1 --port 8000
```
같은 부하 스크립트 실행. 서버 로그에 `[Engine 0]`·`[Engine 1]` 둘 다 활성, 각각
`Running: ≥1` 동시 표시. round-robin이라 요청 절반씩 분배.

### C. 결과 분석 (tok/s 합산)

응답 JSON들에서 `usage.completion_tokens` 합을 총 시간으로 나누면 시스템 throughput:

```bash
TOTAL_TOK=$(python3 -c "
import json, glob
total = sum(json.load(open(f))['usage']['completion_tokens'] for f in glob.glob('/tmp/dptest/resp_*.json'))
print(total)
")
echo "Total completion tokens: $TOTAL_TOK"
# 위 time 출력의 real 시간으로 나누면 system tok/s
# 예: 1024 tokens / 16.0s = 64 tok/s
```

같은 N으로 dp=1 vs dp=2:

| N (동시 요청) | dp=1 (1장) | dp=2 (2장) | dp=2 이득 |
|---:|---|---|---|
| 1 | 단일 ~14 tok/s | 단일 ~14 tok/s | 없음 (한 엔진만 사용) |
| 2 | ~14 tok/s 시스템 (큐 대기) | ~28 tok/s 시스템 (둘 다 가동) | 약 2배 |
| 8 | ~14 tok/s 시스템 (큐 7개) | ~28 tok/s 시스템 (4·4 분배) | 약 2배 |
| 32 | 큐 31개, throughput 동일 | 큐 15·15, throughput ~2배 | 약 2배 |

### D. 부하 테스트 변형

**더 많은 동시 요청** — `N=32` 또는 `N=64`로 키워 saturation 확인.

**다양한 prompt 길이** — KV 캐시 압박 시험:
```bash
LONG_PROMPT=$(python3 -c "print('def solve(): ' * 200)")   # 약 2000 토큰
```

**`xargs -P`로 더 효율적인 병렬** (`&`·`wait` 대신):
```bash
seq 1 $N | xargs -I{} -P $N curl -s http://127.0.0.1:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"...\",\"messages\":[...],\"max_tokens\":128}" \
    -o /tmp/dptest/resp_{}.json
```

벤치마크의 `sweep` 태스크(`configs/models.yaml`의 `batch_sizes`)가 이 패턴을 자동화한 것이고, dp=1·dp=2의 throughput curve가 거기서 정량적으로 갈립니다.

---

## 6. Python OpenAI SDK로 같은 호출

OpenAI 공식 SDK가 `base_url` 바꿔 끼우면 furiosa-llm 서버에 그대로 붙어요.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",   # furiosa-llm 기본은 인증 없음 — 아무 값
)

resp = client.chat.completions.create(
    model="Qwen/Qwen3-32B-FP8",
    messages=[{"role":"user","content":"Reverse a string in Python."}],
    max_tokens=256,
)
print(resp.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="...",
    messages=[{"role":"user","content":"hi"}],
    max_tokens=64,
    stream=True,
):
    delta = chunk.choices[0].delta.content
    if delta: print(delta, end="", flush=True)
```

`bench-gpu/runners/tps.py` · `rngd-npu/runners/tps.py` 같은 측정 코드도 이 SDK로 작성돼 있어서, `base_url`만 바꾸면 GPU/NPU 측정 코드 동일하게 재사용됩니다.

---

## 7. 응답 JSON 읽는 법

```json
{
  "id": "chat-86be...",
  "object": "chat.completion",
  "created": 1779631724,
  "model": "/home/jun/RNGD-proj/.../artifacts/qwen3-32b-fp8-tp8",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "...",            ← ★ 모델 본문 응답
        "reasoning": null,            ← reasoning-parser 켜면 분리됨
        "tool_calls": []
      },
      "finish_reason": "stop",        ← stop / length / tool_calls 중 하나
      "logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 17,              ← 입력 토큰 수
    "completion_tokens": 256,         ← 생성 토큰 수
    "total_tokens": 273
  }
}
```

자주 보는 필드:
- **`choices[0].message.content`** — 모델 답변 본문
- **`finish_reason`**:
  - `stop` = EOS 토큰 만나서 자연 종료
  - `length` = max_tokens에 막혀서 잘림 → 보통 max_tokens 키워야 함
  - `tool_calls` = function call 시작
- **`usage.completion_tokens`** — 생성된 토큰 수. 처리량(tok/s) 계산 시 사용
- **`model`** — 서버가 인식한 모델 id. 우리 케이스는 artifact 절대경로로 등록됨

서버 stderr 로그(터미널 A)에도 측정값 보임:
```
Avg prompt throughput: 1.7 tokens/s,  Avg generation throughput: 14.6 tokens/s,
Running: 1 reqs,  RNGD KV cache usage: 0.3%
```

---

## 8. 자주 보는 에러 → 조치

| 에러 | 의미 | 조치 |
|---|---|---|
| `Invalid rev id: v2026.2` | HF repo에 v2026.2 revision 없음 | snapshot 경로 직접 지정 또는 `--revision <태그>` |
| `FileNotFoundError: Path does not exist: ...` | 손으로 친 캐시 경로의 모델명 오타 (예: `Llama3.1` vs `Llama-3.1`) | 정확한 이름 확인 — `ls ~/.cache/huggingface/hub/models--<org>--*` 로 실제 폴더명 보고 복사. 또는 `find ... -name "<patt>*"`로 snapshot 자동 찾기 |
| `ValueError: Couldn't instantiate the backend tokenizer ... need sentencepiece or tiktoken` | **메시지가 헷갈리는데 실제 원인 두 가지:** (a) snapshot 경로가 아니라 **model 컨테이너 디렉터리**(`blobs/`·`refs/`·`snapshots/`만 든)를 줬을 때 → tokenizer.json 못 찾아 fallback 실패. (b) 정말로 sentencepiece/tiktoken 미설치 (드물게, transformers 4.x 환경) | (a) **snapshot 경로까지 내려가서** 입력: `find <model_dir>/snapshots -maxdepth 1 -mindepth 1 -type d` 결과 사용. (b) `pip install sentencepiece tiktoken`. (a)가 압도적으로 자주 만나는 케이스 |
| `Required PEs: N, Actual: M` | artifact 빌드 tp ≠ 가용 PE | `--devices`로 PE 더 주거나, 작은 tp로 재빌드 |
| HBM OOM | weight + KV가 카드 HBM 초과 | `--max-model-len` 줄여 재빌드, `--max-batch-size` 축소 |
| `RuntimeError: expected value at line ...` | artifact.json 깨짐 (다운로드 미완성 등) | 캐시 폴더 삭제 후 재다운로드 |
| `RuntimeError: unknown variant 'text-generation', expected one of 'generate', 'embed', 'score'` | **옛 SDK로 빌드된 prebuilt + 새 SDK의 schema 충돌** (`"task_type"` enum 값 변경). `serve_model.sh`가 자동 패치하지만, 이건 schema 충돌의 첫 layer일 뿐 — 같은 prebuilt가 더 깊은 곳에서도 충돌함 (예: `missing field 'inputs'`) | 패치 후에도 깊은 schema 충돌 더 나오면 그만 patching하고 HF 원본부터 재빌드 또는 다른 prebuilt 사용. 실측: `furiosa-ai/Qwen2.5-Coder-7B-Instruct`는 다층 schema 불일치라 사실상 폐기 — Llama-3.1-8B-Instruct·Qwen3-32B-FP8 등 동작 확인된 prebuilt 권장 |
| `RuntimeError: missing field 'inputs' at line N column N` | 위와 같은 다층 schema 충돌의 deeper layer. 첫 패치(`task_type`) 이후 노출됨 | field-by-field 패치 무의미. 다른 prebuilt 사용 또는 HF 원본 재빌드 |
| `Connection refused` (curl) | 서버 안 떴거나 포트 다름 | 서버 로그 확인, 포트 일치 확인 |
| `400 Bad Request` model not found | `"model"` 필드 값이 등록된 id와 다름 | `/v1/models`로 정확한 id 확인 후 그대로 사용 |
| 응답 잘림 (`finish_reason: length`) | max_tokens 부족 | 숫자 키우기 |

---

## 9. 한 모델 두 카드 — 동시 서빙 vs dp=2

같은 머신에서 두 가지 방식:

### A. 두 모델 동시 (서로 다른 카드·포트)

```bash
# 터미널 A — Qwen2.5-Coder-7B를 npu:0에
furiosa-llm serve <coder-7b-path> --devices npu:0 --port 8000

# 터미널 B — Qwen3-32B-FP8을 npu:1에
furiosa-llm serve <qwen3-32b-path> --devices npu:1 --port 8001
```
→ 두 서버 독립. 클라이언트는 포트로 구분.

### B. 한 모델 두 카드 dp=2 (처리량 2배)

```bash
furiosa-llm serve <tp=8 artifact> --devices npu:0,npu:1 --port 8000
```
→ 엔진이 자동으로 dp=2 인식해 복제본 2개 실행. 서버 로그에 `[Engine 0]`과 `[Engine 1]`이 따로 찍힘.

#### Routing 동작 (실측)

라우터가 요청을 **두 엔진에 round-robin으로 분배**합니다. 같은 prompt를 4번 연달아 보내면:
```
요청 1 → Engine 0 (npu:0)
요청 2 → Engine 1 (npu:1)
요청 3 → Engine 0 (npu:0)   ← prefix cache hit 48.5% (요청 1과 같은 엔진)
요청 4 → Engine 1 (npu:1)   ← prefix cache hit 48.5% (요청 2와 같은 엔진)
```
`furiosa-smi`로 보면 npu:0과 npu:1이 번갈아 활성화돼 보임.

#### dp=2가 빨라지는 시점 ⚠️

**단일 요청의 throughput은 1장 dp=1과 똑같습니다** (~15~20 tok/s). 한 요청은 복제본 *하나*에서 처리되니까. dp=2의 이점은 **동시 요청 처리량 ~2배**:

| 시나리오 | 1장 dp=1 | 2장 dp=2 |
|---|---|---|
| 요청 1개 | ~15 tok/s, 17초 | ~15 tok/s, 17초 (변화 X) |
| 요청 2개 동시 | 1개는 큐 대기 → ~34초 | 평행 처리 → ~17초 (≈ 2배 빠름) |
| 요청 8개 동시 | 8개가 한 엔진 큐 | 4개씩 두 엔진 분배 |

동시 요청 효과 직접 확인:
```bash
for i in 1 2 3 4 5 6 7 8; do
    curl -s http://127.0.0.1:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"...","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}' \
      -o /tmp/resp_$i.json &
done
wait
```
서버 로그에서 두 엔진 다 `Running: ≥1 reqs` 동시 표시 → dp 활성.

#### Routing 정책 변경 (`--data-parallel-routing-policy`)

| 정책 | 동작 | 언제 유리 |
|---|---|---|
| `round_robin` (기본) | 요청을 번갈아 분배 | 요청들이 서로 다른 prompt — 균등 분산 |
| `prefix_aware` | 같은 prefix를 공유하는 요청을 같은 엔진으로 유도 | 긴 system prompt 공유 (코드 에이전트·RAG) — prefix cache 적중률↑ |

```bash
furiosa-llm serve <artifact> --devices npu:0,npu:1 --port 8000 \
    --data-parallel-routing-policy prefix_aware
```

(단 tp=8 artifact 한정. tp×pp가 카드 PE를 정확히 나누지 못하면 거부됨.)

---

## 10. 코드 참조

| 항목 | 파일:라인 |
|---|---|
| serve CLI 진입 | `cli/serve.py:385` `serve()` → `app.py:536` `run_server` |
| LLM 로드 (artifact resolve) | `api.py:216` `LLM.__init__` → `api.py:346` `_init_from_artifact` |
| HF id → 로컬 경로 다운로드 | `utils.py:314` `get_path_or_hf_download` (기본 revision = `FURIOSA_LLM_VERSION`) |
| OpenAI 호환 endpoint 라우터 | `server/app.py` 안의 `/v1/chat/completions`, `/v1/models`, `/v1/responses` |
| chat template 처리 | `server/chat_utils.py` |

---

## 11. 관련 문서

- [`README.md`](README.md) — 측정 파이프라인·orchestrator 사용법 (서버 자동 띄우기·sweep 등)
- [`README_build.md`](README_build.md) — 빌드 옵션·OOM 트러블슈팅
- [`README_preset.md`](README_preset.md) — `presets.py` 버킷 4종, fmt 지시문
- [`README_config.md`](README_config.md) — HF `config.json` 필드, FP8 fmt
- [`BUILD_COMPIL.md`](BUILD_COMPIL.md) — Pipeline build vs Compile 두 단계 차이
- OpenAI ChatCompletion API spec: <https://platform.openai.com/docs/api-reference/chat>
