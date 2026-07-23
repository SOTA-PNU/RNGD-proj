# furio NPU 상태 표시·병렬화 선택 설계

2026-07-22 · 대상: `claude-agent/`(furio 클라이언트) + `coding-agent/furiosa_router.py`(서버 라우터)

## 목표

1. `.openclaude.json` 의 `description` 을 `"Detected from Local OpenAI-compatible"` → tp/pp 서빙 정보로
2. `/model` 에서 pp·dp 를 선택 가능하게
3. NPU 에 올라간 모델 상태를 LED 로 표시 (초록=로드완료 / 노랑깜빡=전환중 / 빨강=미로드)
4. NPU 4장에 서로 다른 모델을 동시 서빙, 올라간 모델 전부를 상태줄에 표시
5. 개인 PC 에서 `install.sh` 만으로 전부 동작

## 실측으로 확정된 제약 (설계의 전제)

전부 2026-07-22 이 서버에서 직접 측정. 추측 아님.

| 제약 | 근거 |
|---|---|
| **`-tp` 는 아티팩트 로드 시 무시된다** | serve 로그: `When loading LLM from artifact, given -tp value will be ignored.` |
| **pp 는 FXB 아티팩트에서 미지원** | `pyo3_runtime.PanicException: FXB-based artifacts currently does not support pipeline parallelism.` |
| **dp 는 `--devices` 카드 수로 자동 결정** | mesh probe — 1장→`1 DP group`, 2장→`2 DP group`, 4장→`4 DP group` (전부 TP=8) |
| `-dp N` 을 명시해도 카드 수 추론과 동일 | `-dp 2 --devices npu:2,npu:3` → `2 DP group(s)` (플래그 없을 때와 같음) |
| tp32 모델은 4장 전부 점유 | `fxb show` / `artifact.json` 전수 조사 |
| `description` 은 openclaude 에 하드코딩 | `dist/cli.mjs` 에 `` description:`Detected from ${routeLabel}` `` 1곳 |
| statusLine 확장점 존재 (5초 타임아웃) | `executeStatusLineCommand(input, signal, timeoutMs=5000)` |

**따라서 요구사항 2의 실현 범위는 다음이 전부다:**

| 모델 | 아티팩트 | tp | dp 선택 | pp 선택 |
|---|---|---|---|---|
| Qwen3-8B-FP8 | fxb | 8 | ✅ 1·2·4 | ❌ (FXB) |
| Qwen3-4B-FP8 | fxb | 8 | ✅ 1·2·4 | ❌ (FXB) |
| Llama-3.1-8B-Instruct | v2 | 8 | ✅ 1·2·4 | ✅ 1·2·4 |
| Qwen2.5-0.5B-Instruct | v2 | 4 | ✅ | ✅ |
| 나머지 11종 (gpt-oss·Solar·K-EXAONE·Qwen3-Coder·Qwen3-30B×3·Qwen3-VL·Qwen3-32B·Llama-70B·EXAONE) | — | 32 | ❌ 4장 고정 | ❌ |

제약: `dp × pp ≤ 4` (카드 4장). tp32 는 `dp=pp=1, cards=4` 로 못박는다.

## 아키텍처

```
[개인 PC]  furio = openclaude 포크                [서버]  furiosa_router.py :8400
  ├ BuiltinStatusLine.tsx  ← NPU 세그먼트 추가      ├ ServeManager (chat_app.py 이식)
  │   └ useNpuStatus() 폴링 훅 ──── HTTP ─────────→ ├ GET /router/status  모델별 상태+카드+dp/pp
  ├ ModelPicker.tsx        ← dp/pp 위젯 추가        ├ GET /router/models  tp·dp/pp 선택지·desc
  ├ 모델 목록 description  ← /router/models 에서    └ POST /v1/chat/completions  (model+dp/pp)
  └ install.sh 가 포크 빌드본 설치
```

**단일 진실 원천은 서버 REGISTRY.** 클라이언트는 표시만 한다.

### 서버: ServeManager 이식

현재 라우터의 `Router` 는 상태가 `running` dict 뿐이라 "로딩 중"을 노출하지 못한다.
`chat/chat_app.py` 의 `ServeManager`(161–463행)가 이미 검증된 상태머신을 갖고 있으므로 이식한다:

- 상태: `down` / `loading` / `up` / `stopping` / `error`  ← LED 3색이 여기서 나옴
- `_discover()` — 살아있는 `furiosa-llm serve` 의 `--devices/-pp/-dp` 를 역파싱해 실제와 동기화
- `_pending` — 전환 중 dp/pp 변경 요청을 큐잉했다가 전환 종료 즉시 재적용
- `_par_flags(kind, dp, pp)` — 검증된 플래그 규칙 (pp=1 이면 플래그 없음, dp 는 카드 수로)
- `start_new_session=True` — **라우터/백엔드가 셸 종료에 같이 죽는 현행 버그의 해법**

### 서버: dp/pp 를 받는 경로

`/v1/chat/completions` 의 `model` 필드에 변형 접미사를 허용한다: `Qwen3-4B-FP8@dp2`.
포크한 클라이언트는 위젯 값을 이 형식으로 인코딩해 보낸다. 라우터가 파싱해 `_par_flags` 로 서빙.
(포크를 하더라도 OpenAI 호환 API 를 벗어나지 않기 위해 별도 필드가 아닌 model 문자열에 싣는다.)

### 클라이언트: LED 세그먼트

`BuiltinStatusLine.tsx` 는 이미 세그먼트 구조를 갖고 있다:

```ts
type StatusSegment = { key, priority, text, shortText?, color?: keyof Theme }
```

NPU 모델마다 세그먼트를 추가한다. priority 를 낮게(=오래 살아남게) 주고,
`shortText` 에 축약형을 넣어 좁은 터미널에서 자동 degrade 되게 한다.

```
▶ ● gpt-oss-120b 4장  ● Qwen3-4B-FP8@dp2 npu2,3  ● Llama-3.1-8B
  ↑현재선택  ↑초록          ↑노랑(전환중)              ↑빨강(미로드)
```

- 색: `up`→green, `loading`/`stopping`→yellow, `error`→red, `down`→dim
- **깜빡임**: 터미널엔 CSS 가 없으므로 `useNpuStatus()` 폴링 tick 마다 위상을 뒤집어
  `●`/`○` 를 교대(chat 의 `led-pulse` 를 문자로 재현). 전환 중인 모델에만 적용.
- 갱신: `useClaudeAiLimits()` 와 동일 패턴의 폴링 훅. **전환 중일 때만 짧은 주기(1s),
  유휴 시엔 길게(10s)** — chat 의 "유휴엔 Timer 를 끈다" 전략과 동일한 취지.

## 배포 (요구사항 5)

포크를 택했으므로 `install.sh` 는 npm 대신 우리 빌드본을 설치한다:

1. 서버에서 `bun run build` → `dist/cli.mjs`
2. 빌드 산출물을 tarball 로 서버에 게시 (라우터가 `GET /router/client.tgz` 로 서빙)
3. `install.sh` 가 `$SDI_SERVER/router/client.tgz` 를 받아 `~/.furio/lib` 에 풀고 래퍼 생성

→ 개인 PC 에는 여전히 `SDI_SERVER=... bash install.sh` 한 줄. bun/node22 빌드 툴체인 불필요.

## 리스크

| 리스크 | 대응 |
|---|---|
| 업스트림 openclaude 업데이트 추종 비용 | 수정은 3파일로 국소화(BuiltinStatusLine·ModelPicker·설치경로). 나머지는 upstream 그대로 |
| statusLine 폴링이 5초 타임아웃 초과 | 클라이언트 훅은 statusLine 이 아니라 in-process 폴링 → 타임아웃 무관 |
| 라우터 폴링 부하 | `/router/status` 는 furiosa-smi 호출 포함 → 서버측 1초 캐시 |
| tp32 모델 전환 시 2분 정지 | LED 노랑으로 진행 중임을 표시(이 기능의 존재 이유) |

## 구현 순서

1. 서버: `ServeManager` 이식 + `/router/status` 확장 + `@dp/@pp` 파싱 + `start_new_session`
2. 클라이언트 포크: `useNpuStatus()` 훅 + `BuiltinStatusLine` 세그먼트
3. 클라이언트 포크: `ModelPicker` dp/pp 위젯
4. `install.sh`: tarball 배포 경로 + description 주입
5. 소형 모델(Qwen3-4B)로 전 구간 실측 검증
