# `presets.py` 자세히 — 버킷 프리셋 참고서

`furiosa_llm/artifact/presets.py`에 들어 있는 버킷 프리셋의 구조를 정리한 문서입니다.
앞으로 `presets.py` 관련 새 발견(버킷 종류, 매칭 규칙, fmt 지시문 등)은 모두 이 문서에
누적합니다.

소스 인용은 모두 `~/furiosa/lib/python3.12/site-packages/furiosa_llm/`을 기준,
SDK 2026.2.0.

---

## 1. 한눈에

`presets.py`는 두 가지를 정의합니다:

| 자료구조 | 역할 |
|---|---|
| `PresetRef` (`presets.py:30`) | "이 (model_type, hidden_size, intermediate_size) 조합에 어떤 BucketConfig를 쓸지" 등록표 한 줄 |
| `BucketConfig` (`types/config.py:58`) | 버킷 4종을 담는 컨테이너 |

빌드 시 `ResolvedBuckets.resolve()`(`resolver.py:34-122`)가 model metadata와 `--max-model-len`을
조합해서 `find_preset()`(`presets.py:268-295`)을 호출하고, 매칭된 `BucketConfig`를 가져와
`filter_preset_by_max_model_len()`으로 한 번 자르고, 그걸 `AttentionBucket` 객체들로 변환합니다.

---

## 2. 버킷 4종 — CLI/preset 입력 형식과 의미

### 2.1 `prefill_buckets` — `(batch_size, context_length)` 2-tuple

KV 캐시가 비어 있는 상태에서 **context_length 만큼의 토큰을 한 번에** 처리하는 모양.

| 항목 | 의미 |
|---|---|
| `batch_size` | 동시에 처리할 시퀀스 수 |
| `context_length` | 이 prefill 단계에서 처리할 토큰 수 (= 시퀀스 길이) |

**언제 쓰이나:** 첫 토큰을 만들기 직전 단계. 사용자 prompt 전체를 모델에 한 번에 흘려서
KV 캐시를 채우는 작업. 보통 짧은 길이(128~1024)만 들어 있습니다.

**내부 변환** (`resolver.py:101`):
```python
AttentionBucket.prefill(*bucket)  # = AttentionBucket(batch_size, context_length, kv_cache_size=0)
```
- `kv_cache_size = 0` → 캐시 없음, 모든 토큰이 새 입력
- `input_ids_size = attention_size = context_length`

### 2.2 `decode_buckets` — `(batch_size, context_length)` 2-tuple

KV 캐시가 거의 다 채워진 상태에서 **새 토큰 1개씩** 생성하는 모양.

| 항목 | 의미 |
|---|---|
| `batch_size` | 동시 진행 중인 시퀀스 수 |
| `context_length` | 이 시퀀스가 도달할 최대 길이 (KV + 새 토큰 1) |

**언제 쓰이나:** 생성 루프. 매 iteration에 토큰 하나만 새로 만들고 그 직전까지는 캐시 활용.

**내부 변환** (`resolver.py:104`):
```python
AttentionBucket.decode(*bucket)  # = AttentionBucket(batch_size, context_length, kv_cache_size=context_length-1)
```
- `input_ids_size = 1` (새 토큰 하나)
- `kv_cache_size = context_length - 1` (이전 토큰들은 다 캐시)

### 버킷 형식 결정 — 왜 어떤 값이 들어가는가?

**`prefill_buckets`** — 보통 `(1, 128), (1, 256), ..., (1, 1024)` 식.
- batch=1 위주 — 실제 서빙에서 prefill은 요청 1개씩 들어옴
- context 128~1024 step 128 — 짧은 prompt 빠르게, 어텐션 블록 정렬 단위

**`decode_buckets`** — `(N, K)` 다양한 조합.
- batch N — 동시 다수 요청 처리량 측정·운영용
- context K — 짧은 대화 ~ 긴 컨텍스트
- 큰 batch × 큰 context는 점진적으로 제거 (KV가 HBM 초과)
- 예: Llama 8B에 `(1, 128K)` ✓ 있지만 `(128, 128K)` ✗ 없음

**`append_buckets`** — prefix cache 확장·speculative decoding 시나리오. attn × input 카르테시안 곱.

### 2.3 `append_buckets` — `(batch_size, attention_size, input_ids_size)` 3-tuple

prefix 일부가 이미 캐시된 상태에서 **새 토큰 묶음을 한 번에** 추가하는 모양.

| 항목 | 의미 |
|---|---|
| `batch_size` | 보통 1 (preset에서 거의 (1, …, …)) |
| `attention_size` | 처리 후 전체 시퀀스 길이 (캐시 + 새 토큰들) |
| `input_ids_size` | 이번에 새로 넣는 토큰 수 |

**언제 쓰이나:** prefix caching 확장 시나리오. 시스템 prompt를 한 번 캐시해 두고, 사용자 요청별로
일부 토큰만 추가해 전체 그래프를 다시 안 만들어도 되게 함. multi-token decoding(예: speculative decoding 후
검증 결과로 N개 토큰 한 번에 commit)에도 활용.

**내부 변환** (`resolver.py:108`):
```python
AttentionBucket(b[0], b[1], b[1] - b[2])
# = AttentionBucket(batch_size, attention_size, kv_cache_size=attention_size-input_ids_size)
```

**검증 규칙** (`validator.py:122`): `attention_size > input_ids_size` 필수.
(같거나 작으면 prefill과 똑같아져서 의미 없음.)

### 2.4 `tokenwise_seq_lens` — `Sequence[int]` 단일 정수 리스트

attention 버킷과는 별개의, **단일 정수만 모인 list/tuple**. composable kernel 빌드 시 사용.

| 항목 | 의미 |
|---|---|
| `int` 각각 | `TokenwiseBucket(input_size=N)` 으로 변환됨 (`metadata/config_types.py`) |

**언제 쓰이나 (확실치 않음):** SDK 코드·docstring 모두 "tokenwise sequence lengths for composable
kernel"이라고만 표기. 추정 — attention과는 별개로 token 단위로 처리되는 non-attention path
(embedding, layernorm 등)의 그래프 컴파일에 쓰이는 듯합니다.

**보통 들어가는 값:** 작은 정수들 위주.
- EXAONE 4 32B: `(2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024)` (`presets.py:98`)
- Llama 3.1 8B: `(1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024)` (`presets.py:120`)
- Qwen2.5-0.5B: `(128, 1024, 2048, 3072, 4096)` (`presets.py:73`) — 예외적으로 큰 값들

검증 (`validator.py:188-199`): `max_executable_len`을 산출할 때 tokenwise도 attention과 함께 고려됨.

---

## 3. 내부 통합 표현 — `AttentionBucket`

`metadata/config_types.py:141` 정의:
```python
class AttentionBucket(BaseModel):
    batch_size: int
    attention_size: int
    kv_cache_size: int

    @property
    def input_ids_size(self) -> int:
        return self.attention_size - self.kv_cache_size
```

분류 property:
- `is_prefill`: `kv_cache_size == 0`
- `is_decode`: `kv_cache_size > 0 and input_ids_size == 1`
- `is_extend`: `kv_cache_size > 0 and input_ids_size > 1`

### 입력 형식 → 내부 표현 변환표

| CLI/preset 입력 | (batch_size, attention_size, kv_cache_size) | input_ids_size |
|---|---|---:|
| prefill `(b, c)` | `(b, c, 0)` | `c` |
| decode `(b, c)` | `(b, c, c-1)` | `1` |
| append `(b, a, i)` | `(b, a, a-i)` | `i` |

---

## 3.X `append_buckets` 형식 — 헬퍼 vs 직접 튜플, 그리고 exclude

같은 `append_buckets`인데 모델마다 표현이 다른 이유.

### `_build_append_buckets()` 헬퍼 (`presets.py:51-66`)

```python
def _build_append_buckets(attn_sizes, input_sizes, exclude=None):
    excluded = set(exclude) if exclude else set()
    return tuple(
        (1, a, i) for a in attn_sizes for i in input_sizes
        if a > i and (a, i) not in excluded
    )

_COMMON_APPEND_ATTN_SIZES = [256, 384, 512, 768, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
_COMMON_APPEND_INPUT_SIZES = list(range(128, 1024+1, 128))  # [128, 256, ..., 1024]
```

attn × input의 **카르테시안 곱**을 자동 생성, `attn > input` 필터 + exclude 적용.

### 형식 결정 트리

```
모델의 max_position이 2의 거듭제곱(1K, 2K, ..., 128K) 안에 들어맞나?
├─ YES → _build_append_buckets() 헬퍼 사용
│        ├─ 특정 조합에 known runtime bug 있나?
│        │   ├─ YES → exclude=[(a, i), ...] 로 그것만 제외
│        │   │       (예: EXAONE 4 32B의 exclude=[(1024, 256)] —
│        │   │        runtime timeout 버그, furiosa-runtime PR #3221 참조)
│        │   └─ NO  → exclude 없이 통째로 (Llama 3.1 8B / 70B)
└─ NO (40K·기타 비표준) → 직접 튜플로 박음
        (예: Qwen3-32B-FP8 — max_position 40960, (1, 40*1024, 128) 추가)

특수 케이스:
- MoE / sparse 모델 → 가볍게 직접 (Qwen3-30B-A3B: (1, 256, 128), (1, 8*1024, 128))
- Pooling 모델 (Embedding/Reranker) → append_buckets 자체 없음 (디코딩 없음)
```

### 모델별 실제

| 모델 | append 표현 | 이유 |
|---|---|---|
| EXAONE 4 32B | `_build_append_buckets(..., exclude=[(1024, 256)])` | helper + 알려진 1024×256 timeout 버그 회피 |
| Llama 3.1 8B | `_build_append_buckets(...)` | helper 그대로, 깔끔 |
| Llama 3.3 70B | `_build_append_buckets(...)` | helper 그대로 |
| Qwen3-32B-FP8 | 직접 튜플 (40K까지) | max_position=40960 (비표준), 40K 명시 박음 |
| Qwen3-30B-A3B MoE | 직접 튜플 (작게) | MoE 특성 + 가벼운 워크로드 |
| Qwen3 8B Embedding/Reranker | (없음) | pooling 모델 — 디코딩·append 없음 |

## 4. `# fmt: off` / `# fmt: on` 지시문

`presets.py`의 버킷 리스트 위·아래에 자주 보이는:

```python
# fmt: off
EXAONE_4_32B_PRESET = BucketConfig(
    prefill_buckets=tuple((1, x) for x in range(128, 1024 + 1, 128)),
    decode_buckets=(
        (1, 512), (1, 1024), (1, 2 * 1024), (1, 4 * 1024), ...
        ...
    ),
    ...
)
# fmt: on
```

이건 **Python formatter(black, ruff format 등)에게 "이 사이 줄은 자동 정렬 건드리지 마"라고
알려주는 지시 주석**입니다. presets.py 코드 자체 동작과는 아무 상관 없어요.

**왜 필요한가:** 버킷 리스트가 한 줄에 여러 튜플(`(1, 1024), (1, 2*1024), ...`)이 가지런히
정렬돼 있어서 한눈에 batch_size별·context별로 비교하기 쉽습니다. 만약 fmt: off가 없으면
black이 각 튜플을 자기만의 줄로 풀어버려서 ~100줄짜리 표가 되고 가독성이 망가져요.

**관련 코드:** `presets.py:79-100`(EXAONE), `:105-122`(Llama 3.1 8B), `:127-144`(Llama 3.3 70B),
`:149-178`(Qwen3 32B FP8) 등.

> 헷갈리지 마세요 — `fmt`라는 단어가 모델 `config.json`의 `quantization_config.fmt`에도
> 나오는데, 거기서의 `fmt`는 **FP8 부동소수점 포맷**(`e4m3` 등) 의미입니다. 완전히 다른 개념.
> 자세한 설명은 [`README_config.md`](README_config.md) 참고.

---

## 5. `PresetRef` 등록 — 모델 인식 방식

`presets.py:30` 정의:
```python
@dataclass(frozen=True)
class PresetRef:
    model_type: str
    hidden_size: int
    intermediate_size: int
    preset: BucketConfig
```

`presets.py:210` `PRESET_REFS` 튜플에 이 PresetRef들이 나열됨. 빌드 시 `find_preset()`이 이 목록을
훑어서 매칭.

### 네 인자 — 값은 누가 결정?

| 필드 | 누가 결정 | 비고 |
|---|---|---|
| `model_type` | **모델 제작자가 HF `config.json`에 박음** | `"qwen2"`, `"llama"`, `"qwen3_moe"` 등. 우리가 못 바꿈, 그대로 사용 |
| `hidden_size` | **모델 architecture 고정값** (학습 시점) | HF `config.json`의 `hidden_size` 그대로 |
| `intermediate_size` | **모델 architecture 고정값** | HF `config.json`의 `intermediate_size` 그대로 |
| `preset` | **furiosa 팀이 그 모델 사이즈에 맞춰 튜닝** | 우리도 추가·수정 가능 |

→ PresetRef 새로 등록할 때 앞 세 값은 *모델의 실제 config.json 값 그대로* 가져와야 함. 임의 못 만듦.
→ `preset`만 우리가 결정. `find_preset()` (아래)이 model_type 일치 후 (h, i) log-distance로 best match 선택.

### `find_preset` 매칭 규칙 (`presets.py:268-295`)

```python
def find_preset(model_type, hidden_size, intermediate_size):
    candidates = [r for r in PRESET_REFS if r.model_type == model_type]
    if not candidates:
        return None
    best = min(candidates, key=lambda r: abs(
        math.log(_approx_per_layer_params_b(r.hidden_size, r.intermediate_size))
        - math.log(_approx_per_layer_params_b(hidden_size, intermediate_size))
    ))
    return best.preset
```

1. `model_type` **정확 일치**인 후보만 추림
2. 그 후보들 중 `(hidden_size, intermediate_size)`로 계산한 layer당 파라미터 수의 **log-distance
   가 가장 가까운** 항목을 best match로 선택
3. → fine-tune 모델 등 사이즈가 미세하게 다른 경우도 가까운 preset 자동 매칭 가능

### 등록된 preset 7종 (`PRESET_REFS`)

| # | model_type | hidden_size | intermediate_size | 대표 모델 | preset 변수명 |
|---:|---|---:|---:|---|---|
| 1 | `qwen2` | 896 | 4864 | Qwen2.5-0.5B | `QWEN_2_5_0D5B_PRESET` |
| 2 | `exaone4` | 5120 | 27392 | EXAONE 4.0 32B | `EXAONE_4_32B_PRESET` |
| 3 | `llama` | 4096 | 14336 | Llama 3.1 8B | `LLAMA_3_1_8B_PRESET` |
| 4 | `llama` | 8192 | 28672 | Llama 3.3 70B | `LLAMA_3_3_70B_PRESET` |
| 5 | `qwen3` | 5120 | 25600 | Qwen3-32B-FP8 | `QWEN_3_32B_FP8_PRESET` |
| 6 | `qwen3` | 4096 | 12288 | Qwen3 Embedding/Reranker 8B | `QWEN_3_8B_POOLING_PRESET` |
| 7 | `qwen3_moe` | 2048 | 6144 | Qwen3 30B-A3B MoE | `QWEN_3_30B_A3B_PRESET` |

---

## 6. `filter_preset_by_max_model_len`

`presets.py:298-313`:
```python
def filter_preset_by_max_model_len(preset, max_model_len):
    return BucketConfig(
        prefill_buckets=tuple(b for b in preset.prefill_buckets if b[1] <= max_model_len),
        decode_buckets=tuple(b for b in preset.decode_buckets if b[1] <= max_model_len),
        append_buckets=tuple(b for b in preset.append_buckets if b[1] <= max_model_len),
        tokenwise_seq_lens=tuple(s for s in preset.tokenwise_seq_lens if s <= max_model_len),
    )
```

규칙:
- prefill/decode/append: 튜플의 **두 번째 원소**(`b[1]` = attention_size/context_length)가
  `max_model_len` 이하인 버킷만 유지
- tokenwise: 그 정수 자체가 `max_model_len` 이하인 것만 유지

→ `--max-model-len`을 작게 잡으면 큰 버킷이 *자동으로 빠집니다.* 빌드 시간·메모리를
줄이는 가장 쉬운 lever.

### Qwen3-32B-FP8 preset 필터링 예시

preset(`presets.py:147-178`)의 decode 버킷 중 가장 큰 attention_size가 **40960** (= 40×1024,
모델 native max를 그대로 반영). `--max-model-len` 값별 살아남는 버킷:

| `--max-model-len` | 살아남는 가장 큰 decode 버킷 |
|---:|---|
| 40960 (default) | `(N, 40960)` for N∈{1,2,4} ← 가장 무거움 |
| 32768 | `(N, 32768)` |
| 16384 | `(N, 16384)` |
| 8192 | `(N, 8192)` |

> ⚠️ **preset 값이 항상 2의 거듭제곱은 아닙니다.** Qwen3-32B-FP8처럼 모델 native에 맞춰
> 40960 같은 비-정수승 값이 들어 있을 수 있어요. preset 내용을 직접 확인하는 게 안전.

---

## 7. 새 모델용 preset 추가하기 (참고)

만약 SDK에 직접 preset을 추가하고 싶다면 (보통은 `-pb`/`-db`로 수동 지정이 더 쉽지만, repeat
대량 빌드면 코드 수정이 합리적):

1. 새 `BucketConfig` 상수 정의 (예: `MY_MODEL_PRESET = BucketConfig(prefill_buckets=..., ...)`)
2. `PRESET_REFS` 튜플에 `PresetRef(model_type=..., hidden_size=..., intermediate_size=..., preset=...)` 추가
3. furiosa-llm 재설치 또는 pip editable install

수동 버킷 지정 방식(`-pb`/`-db`)은 [`README_build.md`](README_build.md) 6절 참고.

---

## 8. 코드 빠른 참조

| 항목 | 파일:라인 |
|---|---|
| `PresetRef` 정의 | `artifact/presets.py:30` |
| 헬퍼 — append bucket 자동 생성 | `artifact/presets.py:51-66` |
| 개별 preset 상수들 (Qwen2.5, EXAONE, Llama, Qwen3 등) | `artifact/presets.py:70-203` |
| `PRESET_REFS` 등록 목록 | `artifact/presets.py:210-260` |
| `find_preset` 매칭 | `artifact/presets.py:268-295` |
| `filter_preset_by_max_model_len` | `artifact/presets.py:298-313` |
| `BucketConfig` 스키마 | `artifact/types/config.py:58-91` |
| `AttentionBucket` 정의 | `metadata/config_types.py:141` |
| 빌드 시 호출 (resolve 흐름) | `artifact/resolver.py:34-122` |
| 버킷 검증 규칙 | `artifact/validator.py:73-199` |

---

## 9. 관련 문서

- [`README_build.md`](README_build.md) — 빌드 옵션 전반, 트러블슈팅, 수동 버킷 지정 방법
- [`README_config.md`](README_config.md) — HF model config.json 필드 자세히 (이 문서의 *모델 쪽* counterpart)
- [`BUILD_COMPIL.md`](BUILD_COMPIL.md) — Pipeline build vs Compile 두 단계 차이
