# `furiosa-llm build` 내부 호출 흐름

`furiosa-llm build <model> <out> -tp 8` 한 줄을 실행하면 그 안에서 네 개의 모듈이 순서대로 협업합니다.

| 파일 | 역할 한 줄 |
|---|---|
| `builder.py` | **오케스트레이터.** 전체 흐름 지휘 |
| `validator.py` | **문지기.** 각 설정·HF config·버킷이 규칙에 맞나 검증 |
| `resolver.py` | **해결사.** "사용자가 안 준 값"을 HF config 와 preset 에서 채움 |
| `presets.py` | **레시피 책.** 모델 종류별로 미리 정해둔 버킷 모음 |

> 모든 경로는 `~/furiosa/lib/python3.12/site-packages/furiosa_llm/artifact/` 기준.

---

## 큰 그림: 3단계

```
furiosa-llm build (CLI)
    │
    ▼
 ┌──────────────────────────────────────────┐
 │ ArtifactBuilder.__init__                 │   ← 본 문서의 핵심
 │   ① Validate  (validator.py)             │
 │   ② Resolve   (resolver.py + presets.py) │
 │ (builder.py:127~170)                     │
 └────────────┬─────────────────────────────┘
              │
              ▼
 ┌──────────────────────────────────────────┐
 │ ArtifactBuilder._build_model_artifact    │   ← 실제 빌드 (트레이싱 + 컴파일)
 │  (builder.py:172~)                       │
 └──────────────────────────────────────────┘
```

①·② 가 짧고 가볍습니다 (몇 초). 실제 시간은 ③ 트레이싱·컴파일 단계에서 다 잡아먹습니다 (`BUILD_COMPIL.md` 참고).

---

## ① Validate 단계 (`validator.py`)

`ArtifactBuilder.__init__` 의 첫 번째 블록 (`builder.py:153~158`):

```python
# ── Validate ──────────────────────────────────────────────────
validate_artifact_config(artifact_config, model_id_or_path=...)
validate_parallel_config(parallel_config)
if bucket_config.has_explicit_buckets() and not bucket_config.skip_validation:
    validate_bucket_config(bucket_config)
validate_hf_config(hf_config)
```

| 호출 | 무엇을 확인하나 (`validator.py`) |
|---|---|
| `validate_artifact_config` (202) | `copies_from_local` 의 파일이 실제로 있는지, `copies_from_model` 의 모델 파일이 HF Hub 에서 받아지는지 |
| `validate_parallel_config` (234) | `tensor_parallel_size ∈ {4, 8, 32}` / `pp ≥ 1` / 필요 디바이스 수 ≤ 8 (`ceil(tp/8) * pp ≤ 8`) |
| `validate_bucket_config` (73) | 사용자가 명시한 버킷이 있을 때만 — `(a, b)` 모양, 중복 없음, append bucket 의 attention > input_ids 등 |
| `validate_hf_config` (25) | HF config 에 `max_position_embeddings`, `num_hidden_layers`, `hidden_size`, `intermediate_size` 가 존재하는지. 없으면 즉시 에러 |

**의도:** 비싼 단계로 가기 전에 빨리 실패시키기. `tp=7` 같이 잘못된 값을 줘도 빌드를 한참 돌린 뒤 깨지지 않고 즉시 거부됩니다.

---

## ② Resolve 단계 (`resolver.py` + `presets.py`)

`ArtifactBuilder.__init__` 의 두 번째 블록 (`builder.py:160~170`):

```python
# ── Resolve ───────────────────────────────────────────────────
self._model_metadata = resolve_model_metadata(...)
self._max_model_len  = resolve_max_model_len(hf_config, model_config.max_model_len)
self._device_mesh    = resolve_device_mesh(parallel_config)
self._buckets        = ResolvedBuckets.resolve(
    self._model_metadata, bucket_config, self._max_model_len
)
```

### 2-1. `resolve_max_model_len` (`resolver.py:125~159`)

- 사용자가 `--max-model-len` 을 안 줬으면 → HF config 의 `max_position_embeddings` 그대로 사용
- 줬는데 `max_position_embeddings` 보다 크면 → `ValueError`
- 줬고 작거나 같으면 → 그대로

```python
def resolve_max_model_len(hf_config, max_model_len):
    mpe = hf_config.max_position_embeddings
    if max_model_len is None:
        return mpe
    if max_model_len > mpe:
        raise ValueError(...)
    return max_model_len
```

**이 단계가 "context window 의 천장"을 정해줍니다.** 이후 버킷 검증의 기준이 됩니다.

### 2-2. `ResolvedBuckets.resolve` (`resolver.py:34~122`)

버킷이 비어있으면 `presets.py` 에서 가져오고, 채워져 있으면 그대로 변환합니다. **`presets.py` 가 호출되는 유일한 지점**입니다.

```
 사용자가 -pb/-db/-ab 인자로 버킷을 줬나?
        │
   ┌────┴────┐
   YES       NO (전부 비어있음)
   │         │
   │         ▼
   │   ┌────────────────────────────────────┐
   │   │ presets.find_preset(model_metadata,│
   │   │                     max_model_len) │
   │   │  ↳ PRESET_REFS 에서 로그-거리로     │
   │   │    (model_type, hidden, inter) 매칭 │
   │   │  ↳ filter_preset_by_max_model_len 으로│
   │   │    max_model_len 보다 큰 버킷 제거 │
   │   └────────────────────────────────────┘
   │         │
   ▼         ▼
   "사용자 값 그대로"  /  "preset 값으로 채움"
        │
        ▼
   ┌─────────────────────────────────────┐
   │ 튜플 → AttentionBucket 객체로 변환 │
   └─────────────────────────────────────┘
        │
        ▼
   validate_resolved_buckets()         ← validator.py 재방문
```

**왜 두 갈래?** 사용자가 직접 버킷을 주면 그걸 따르고, 안 주면 SDK 가 준비해둔 prest 에서 모델별 추천값을 가져옴. 이게 `presets.py` 의 존재 이유입니다 — `README_preset.md` 에서 자세히 다룹니다.

### 2-3. `validate_resolved_buckets` (`validator.py:129~199`)

Resolve 가 끝난 뒤 다시 한 번 validator 가 호출됩니다. 이번엔 "max_model_len" 같이 resolve 된 값이 있어야 검증할 수 있는 규칙들:

1. **생성형 모델은 decode bucket 필수**. 없으면 에러.
2. **임베딩/리랭커는 decode bucket 있으면 경고** (어차피 무시됨).
3. **개별 버킷이 max_model_len 을 안 넘어야 함** — prefill / decode / append 각각 검사.
4. **합쳐서도 안 넘어야 함** — `compute_limits()` 가 tokenwise + attention 버킷을 종합해 `max_executable_len` 을 계산. 이게 `max_model_len` 보다 크면 에러.

이 규칙 4번이 우리 `presets.py` 작업할 때 자주 걸렸던 부분입니다 — 32K decode bucket 을 넣으면서 `max_model_len < 32K` 인 모델을 빌드하면 여기서 깨집니다.

---

## ③ 빌드 (`builder.py:172~`)

이건 본 문서 범위 밖. 요약만:
- 모델 모듈 로드 → `Pipeline` 객체 생성 → tracing (`Model Tracing Progress 0~89/89`)
- Ray worker 로 GraphModule export → 컴파일러로 12-단계 LIR → EDF 출력
- 결과를 `artifact.json` + `params/` 로 저장

자세한 phase 설명은 `BUILD_COMPIL.md` 참고.

---

## 코드 위치 빠른 참조

| 항목 | 위치 |
|---|---|
| `ArtifactBuilder.__init__` (전체) | `builder.py:116~170` |
| Validate 블록 | `builder.py:153~158` |
| Resolve 블록 | `builder.py:160~170` |
| `validate_hf_config` | `validator.py:25` |
| `validate_bucket_config` | `validator.py:73` |
| `validate_resolved_buckets` | `validator.py:129` |
| `validate_artifact_config` | `validator.py:202` |
| `validate_parallel_config` | `validator.py:234` |
| `resolve_max_model_len` | `resolver.py:125` |
| `ResolvedBuckets.resolve` | `resolver.py:34` |
| `find_preset` | `presets.py:268` |
| `filter_preset_by_max_model_len` | `presets.py:298` |

---

## 한 줄 요약

> **`builder.py` 가 지휘하고, `validator.py` 가 짧고 확실한 규칙으로 입구를 막고, `resolver.py` 가 사용자가 안 준 값을 채우는데, 그중 버킷만은 `presets.py` 라는 별도 레시피 책에서 꺼내온다.** 그리고 resolve 가 끝난 결과를 다시 한 번 validator 가 확인하고 통과시키면, 그 다음에야 진짜 빌드가 시작된다.
