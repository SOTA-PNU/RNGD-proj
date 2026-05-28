# RNGD 빌드 가이드 (`furiosa-llm build`)

`furiosa-llm build`로 HuggingFace 모델을 RNGD 아티팩트로 컴파일할 때 알아두면 좋은
내부 동작과 옵션을 정리한 참고서입니다. **`presets.py`에 등록 안 된 모델을 직접 빌드할 때**가
주 용도이고, 일반 옵션 의미와 자주 보는 에러도 함께 다룹니다.

빠르게 단계별로 따라하시려면 [`docs/COMPILING_MODELS.md`](docs/COMPILING_MODELS.md)
(다운로드 → 양자화 → 빌드 → 서빙)를 먼저 보시고, 여기서는 그 안쪽 메커니즘을
자세히 설명합니다.

소스 인용은 모두 `~/furiosa/lib/python3.12/site-packages/furiosa_llm/` 아래
경로를 기준으로 표기했습니다 (SDK 2026.2.0).

---

## 한눈에 보는 빌드 흐름

```
HF 모델 (id 또는 로컬 경로)
   │
   ▼
[1] HF config 로드
[2] 입력 검증 — config / parallel / bucket / artifact
[3] resolve — model_metadata, max_model_len, device mesh, buckets
   │
   ▼
[4] pipeline 빌드 (graph 생성, 워커 병렬 가능)  ← `Model Tracing Progress` ★ 메모리 위험 구간
[5] 컴파일 (EDF 바이너리, 워커 병렬 가능)       ← `Compilation Progress` 메모리 안정
   │
   ▼
[6] 저장 — artifact.json + binary_bundle.zip + tokenizer/config
```

코드 흐름: `builder.py:116` `__init__` (1~3단계) → `builder.py:315` `build` (4~6단계).

4~5단계의 두 phase 가 시간/메모리 거의 전부를 차지하는데 성격이 완전히 다릅니다. **OOM 은 거의 100% 4단계 (tracing) 에서 발생**, `Compilation Progress` 로그가 한 번이라도 뜨면 위험 구간은 통과한 상태입니다 (32B 빌드 4회 시도 중 OOM 4회 모두 tracing). 두 단계의 워커 옵션·메모리 특성·코드 위치는 [`BUILD_COMPIL.md`](BUILD_COMPIL.md) 에 별도 정리.

---

## 1. 사전 조건

### 1.1 지원 아키텍처 (`model_type`)

`furiosa-llm build`는 `furiosa.models.language.architecture/`에 모듈이 있는
`model_type`만 받습니다. SDK 2026.2.0 기준 목록:

| `model_type` | 아키텍처 파일 |
|---|---|
| `llama` | `llama.py` |
| `qwen2` | `qwen2.py` |
| `qwen3` | `qwen3/` |
| `qwen3_moe` | `qwen3_moe.py` |
| `qwen3_vl` | `qwen3_vl/` (멀티모달) |
| `exaone` | `exaone.py` |
| `exaone4` | `exaone4.py` |
| `exaone_moe` | `exaone_moe/` |
| `gpt_oss` | `gpt_oss.py` |
| `mistral` | `mistral.py` |
| `mllama4` | `mllama4.py` (Llama 4 멀티모달) |
| `phi3` | `phi3.py` |

빌드 전 확인:
```bash
python3 -c "from huggingface_hub import hf_hub_download; import json; \
print(json.load(open(hf_hub_download('Qwen/Qwen3-32B-FP8','config.json')))['model_type'])"
```
출력이 위 표에 없으면 빌드 불가입니다.

### 1.2 HF config 필수 필드

`validator.py:25-70` `validate_hf_config()`가 빌드 시작 시 확인:

| 필드 | 용도 |
|---|---|
| `max_position_embeddings` | `max_model_len` 기본값, 버킷 검증 |
| `num_hidden_layers` (또는 `num_layers`, `n_layer`) | pipeline 분할, compiler 설정 |
| `hidden_size` | preset 매칭, compiler 설정 |
| `intermediate_size` | preset 매칭, compiler 설정 |

하나라도 빠지면 빌드가 아래 에러로 즉시 멈춥니다:
```
The HuggingFace model config is missing required fields: [...].
Please check the model config or set 'hf_overrides' in ModelConfig.
```
→ `--additional-model-config max_position_embeddings=4096` 같은 식으로 덮어쓸 수 있습니다.

### 1.3 양자화

furiosa가 지원하는 FP8 = **fine-grained FP8, block_size 128, activation_scheme=dynamic**.
`Qwen/Qwen3-32B-FP8`이 정확히 이 방식이라 그대로 빌드 입력으로 쓰면 됩니다
(config.json의 `quantization_config` 확인). HF에 올라온 다른 FP8 변종은 양자화 방식이
다를 수 있어서, 안 맞으면 `transformers.FineGrainedFP8Config`로 직접 양자화해서
로컬 경로로 넣어주는 게 안전합니다 (절차는 `docs/COMPILING_MODELS.md` 2절).

bf16 모델은 별도 처리 없이 그대로 빌드 가능합니다.

---

## 2. 명령어 옵션 한눈에

`furiosa-llm build [options] MODEL OUTPUT_PATH`

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `MODEL` | 필수 | HF 모델 id 또는 로컬 경로(`.` 또는 `/`로 시작) |
| `OUTPUT_PATH` | 필수 | 아티팩트 저장 경로 |
| `--name NAME` | model id | 아티팩트 이름 |
| `-tp N` | **8** | tensor parallel size. **`{4, 8, 32}`만 허용** (`validator.py:246`) |
| `-pp N` | **1** | pipeline parallel size. 양의 정수 |
| `-pb b,c` | (preset) | prefill 버킷 수동 지정. 여러 번 반복 가능 |
| `-db b,c` | (preset) | decode 버킷 수동 지정. 여러 번 반복 가능 |
| `--max-model-len N` | `max_position_embeddings` | 최대 컨텍스트. 모델 값을 초과 못 함 |
| `--additional-model-config k=v` | — | HF config 필드 덮어쓰기 (여러 번 가능) |
| `--num-pipeline-builder-workers N` | **1** | pipeline 빌드 병렬도. 늘리면 빠르지만 RAM↑ |
| `--num-compile-workers N` | **1** | 컴파일 병렬도. 위와 동일 trade-off |
| `--trust-remote-code` | False | HF 커스텀 코드 허용 |
| `--bundle-binaries` / `--no-bundle-binaries` | bundle | EDF 바이너리를 `binary_bundle.zip` 한 파일로 묶음 |
| `--cache-dir DIR` | `$HOME/.cache/furiosa/llm` | 빌드 캐시 (재실행 시 재사용) |

출처: `furiosa-llm build --help`, `builder.py:319-320` (워커 기본값),
`validator.py:246` (tp 허용 값).

---

## 3. 병렬화 (`-tp` / `-pp`)

`validator.py:234-267` `validate_parallel_config()`의 규칙입니다.

- `SUPPORTED_TP_SIZES = {4, 8, 32}` — 그 외 값은 거부됩니다.
- `pp ≥ 1`.
- 필요한 디바이스 수 = `ceil(tp / 8) × pp ≤ 8` (= `MAX_DEVICES`).
- `NUM_PES_PER_NPU = 8` (RNGD 1장 = 8 PE, `device.py:6`).

가능한 조합 (디바이스 합계 ≤ 8 안에서):

| `tp` | `pp` | 카드 수 | 비고 |
|---:|---:|---:|---|
| 4 | 1~8 | 1~8 | `tp=4`은 1장 안에서 PE 4개만 사용 |
| 8 | 1~8 | 1~8 | `tp=8`은 1장 풀 PE |
| 32 | 1~2 | 4~8 | `tp=32`은 4장에 걸침 (furiosa prebuilt 32B/70B가 이 구성) |

> ⚠️ **pp>1은 architecture별로 미구현일 수 있습니다.** `parallelize/block_slicer.py:677`의
> `MODEL_ARCH_TO_BLOCK_SPLITTER_AND_WEIGHT_NODE_PATTERN`에 등록된 class만 pipeline
> parallel 빌드가 가능해요. 2026.2.0 기준:
> - ✅ `LlamaForCausalLM`, `GPTJForCausalLM`, `BertForQuestionAnswering`, `RobertaForQuestionAnswering`
> - ❌ Qwen3·Qwen2·Qwen3_MoE·EXAONE·Mistral·Phi 등 다수
>
> 미지원 모델에 `-pp 2` 주면 빌드 즉시 `NotImplementedError: Block slicing for {Class} is not supported.` 로 종료 (block_slicer.py:727).
> **회피:** dp(data parallel)를 쓰면 됩니다 — dp는 빌드 시 설정 아니고 **serve 시 `--devices npu:0,npu:1`로
> 카드를 더 주면 엔진이 자동 인식**해서 모델 복제본을 여러 카드에 띄웁니다. 모델이 카드 1장에 들어가는 경우(예: 32B FP8 tp=8)
> dp=2가 통신 오버헤드 없어 pp=2보다 효율적이기도 합니다.

**dp 활용 예시 (Qwen3-32B-FP8 1장 → 2장 서빙으로 처리량 2배):**
```bash
# tp=8 아티팩트 하나로 양쪽 다 가능
furiosa-llm serve .../qwen3-32b-fp8-tp8 --devices npu:0                # 1장
furiosa-llm serve .../qwen3-32b-fp8-tp8 --devices npu:0,npu:1          # 2장 dp=2 (engine 자동)
```

> 💡 **dp는 PE를 곱하는 방향만 가능 — 나누지 못합니다.** 엔진 규칙
> `dp × tp × pp = 가용 PE` (`resolver.py:170-202` 참고)에서 dp는 양의 정수.
> 즉 **tp=32로 박힌 artifact를 2장(16 PE)에 띄우려고 dp를 조절하는 건 불가능**
> (`dp × 32 = 16 → dp=0.5` 안 됨). tp=32 artifact는 dp 어떻게 잡든 항상 32×N PE
> (= 4·8·12·...장)가 필요. 작은 머신에 띄우려면 작은 tp로 재빌드가 유일한 길.
> (실측 에러: 2장에 prebuilt tp=32 띄우려 했을 때 `Required PEs: 32`로 거부됨.)

핵심 개념:
- **tp** = 하나의 데이터 병렬 그룹 안에서 PE를 몇 갈래로 쪼개는지
- **pp** = 그 위에 layer를 몇 단계로 쌓는지
- **dp**(data parallel) = 빌드 시에는 1로 고정(`resolver.py:181`), 서빙 시 디바이스 늘리면 자동 인식

대략적인 선택 가이드:

| 모델 / dtype | 권장 tp/pp | 카드 |
|---|---|--:|
| ~1.5B / bf16 | `tp=4~8` | 1 |
| ~8B / bf16 | `tp=8` | 1 |
| ~32B / FP8 | `tp=8` | 1 |
| ~32B / bf16 | `tp=8 pp=2` | 2 |
| ~70B / FP8 | `tp=8 pp=2` | 2 |
| ~70B / bf16 | `tp=8 pp=4` | 4 |

---

## 4. `max_model_len` 결정 규칙

`resolver.py:125-159` `resolve_max_model_len()`:

- `--max-model-len` 안 주면 → `hf_config.max_position_embeddings` 그대로 사용
- 줬는데 `max_position_embeddings`보다 크면 → `ValueError`로 종료
- 그 외 → 준 값 사용

즉 `--max-model-len`은 **모델 한도 안에서 더 작게 자르는 용도**입니다. 모델보다 크게는
못 키웁니다. 정 크게 쓰고 싶으면 `--additional-model-config max_position_embeddings=N`으로
모델 한도 자체를 덮어써야 합니다.

이 값은 **버킷 필터에도 영향**을 줍니다 — 이 값보다 큰 `attention_size`를 가진 버킷은
preset에서 제외되고, 수동 지정도 거부됩니다 (`validator.py:167-185`).

---

## 5. 버킷 시스템 (핵심)

### 5.1 버킷이 뭐고 왜 있나

RNGD는 AOT 컴파일이라 *(batch_size, context_length)* 조합 하나하나를 미리 그래프로
컴파일해 둡니다. 그 미리 빌드된 한 단위를 **버킷(bucket)** 이라 하고, 서빙 시 들어오는
요청을 가장 잘 맞는 버킷에 라우팅해서 처리합니다.

→ **모든 요청 모양을 다 빌드해 둘 필요는 없습니다.** 대표적인 모양만 골라 빌드합니다.
어떤 모양을 고르느냐가 곧 "버킷 설계"입니다.

### 5.2 버킷 4종

`metadata/config_types.py:141` `AttentionBucket(batch_size, attention_size, kv_cache_size)`:

- `input_ids_size = attention_size - kv_cache_size`
- 분류 (`is_prefill`/`is_decode`/`is_extend` property):

| 종류 | 정의 | 언제 |
|---|---|---|
| **prefill** | `kv_cache_size = 0` | 첫 토큰 단계, KV 캐시 없이 시작 |
| **decode** | `input_ids_size = 1` | 한 토큰씩 생성 중 |
| **extend** (= append) | `1 < input_ids_size < attention_size` | prefix-cache 일부 재사용 + 새 토큰 추가 |
| **tokenwise** | int 1개 (`TokenwiseBucket.input_size`) | composable kernel용 |

### 5.3 CLI 인자 형식

- `--prefill-buckets b,c` (`-pb`) → `(batch_size, context_length)` 튜플
- `--decode-buckets b,c` (`-db`) → `(batch_size, context_length)` 튜플
- **append**, **tokenwise**는 CLI 인자가 없음 — 필요하면 Python API(`BucketConfig`)로 줘야 합니다.

### 5.4 자동(preset) vs 수동

`resolver.py:34-122` `ResolvedBuckets.resolve()` 흐름:

1. 사용자가 bucket 필드 중 **하나라도 주면** 나머지도 다 줘야 합니다 (partial 금지).
   - 생성 모델: `prefill_buckets + decode_buckets + append_buckets + tokenwise_seq_lens` 다 필요
   - 비생성(임베딩/리랭커): `prefill_buckets + tokenwise_seq_lens` 만 필요
2. **다 비우면** → preset 찾기:
   - `find_preset(model_type, hidden_size, intermediate_size)` 호출
   - max_model_len으로 필터
   - preset 못 찾으면 → 다음 에러로 빌드 실패:
     ```
     No bucket configuration provided and no matching bucket preset found
     for model_type=X. Please provide explicit bucket configuration.
     ```

### 5.5 매칭 규칙 디테일

`presets.py:268-295` `find_preset()`:

1. `model_type`이 정확히 같은 항목만 후보 (예: `qwen3`은 `qwen3`만, `qwen3_moe`는 다른 그룹).
2. 그 후보들 안에서 `(hidden_size, intermediate_size)` 로 layer당 파라미터 수를 계산해서,
   **log-distance가 가장 가까운 항목을 best match**로 고릅니다.
3. → **사이즈가 정확히 같지 않아도 됩니다**. 같은 architecture의 fine-tune 모델은
   대체로 자동 매칭됩니다 (사이즈 차이가 너무 크면 버킷이 잘 안 맞을 수 있어서 그때는
   수동 지정이 안전합니다).

### 5.6 등록된 preset 목록 (`presets.py:210` `PRESET_REFS`)

| # | model_type | hidden_size | intermediate_size | 매칭되는 대표 모델 |
|---:|---|---:|---:|---|
| 1 | `qwen2` | 896 | 4864 | Qwen2.5-0.5B |
| 2 | `exaone4` | 5120 | 27392 | EXAONE 4.0 32B |
| 3 | `llama` | 4096 | 14336 | Llama 3.1 8B |
| 4 | `llama` | 8192 | 28672 | Llama 3.3 70B |
| 5 | `qwen3` | 5120 | 25600 | Qwen3 32B FP8 |
| 6 | `qwen3` | 4096 | 12288 | Qwen3 8B Embedding/Reranker |
| 7 | `qwen3_moe` | 2048 | 6144 | Qwen3 30B-A3B MoE |

이 표에 model_type 자체가 없는 `mistral`, `phi3`, `gpt_oss`, `mllama4`, `qwen3_vl`,
`exaone`, `exaone_moe`는 **무조건 수동 버킷 필요**입니다.

> 💡 **preset의 버킷 값이 항상 2의 거듭제곱은 아닙니다.** 일부 모델은 `max_position_embeddings`에
> 맞춰 모델 전용 값이 추가돼 있어요. 예: `QWEN_3_32B_FP8_PRESET`은 decode 버킷에 1k/2k/…/16k/32k와
> 함께 **(N, 40*1024) = (N, 40960)** 이 명시적으로 들어 있습니다 (Qwen3-32B의 native 한도 40960).
> 이 때문에 default `--max-model-len=40960`으로 빌드하면 32k·40k 둘 다 빌드 대상이고, 40k 버킷이
> 32k보다 활성화 메모리가 (40/32)²≈1.56× 무거워 OOM의 직접적인 원인이 될 수 있습니다.
> `--max-model-len`을 줄여 32k 이하로 잡으면 40k 버킷이 필터에서 제외돼 부담이 크게 줄어요.

---

## 6. `presets.py`에 없는 모델 빌드하기

이 섹션이 이 문서의 메인입니다.

### 6.1 언제 수동으로 줘야 하나

- `model_type` 자체가 preset에 등록 안 됨 (위 6종)
- 등록은 됐지만 사이즈가 너무 달라 매칭된 preset 버킷이 적당치 않을 때
- 버킷 종류·범위를 직접 조절하고 싶을 때 (예: 작은 컨텍스트만 쓰는데 큰 버킷 빌드 시간이 아까울 때)

### 6.2 검증 규칙 (`validator.py:73-199`)

수동으로 줄 때 통과해야 하는 규칙:

**입력 검증 (`validate_bucket_config`)**:
- prefill 최소 1개
- 모든 차원 양수
- 중복 금지
- append bucket: `attention_size > input_ids_size`

**해석 후 검증 (`validate_resolved_buckets`)**:
- 생성 모델은 decode 최소 1개 (없으면 거부)
- 비생성 모델에 decode 주면 무시되고 경고
- 각 버킷의 `attention_size ≤ max_model_len`
- combined `max_executable_len ≤ max_model_len`

### 6.3 어떤 버킷을 줄지 — 가이드라인

기본 전략은 가장 비슷한 preset 모양을 참고하는 겁니다. Llama 3.1 8B preset
(`presets.py:106-122`) 형태를 보면 감이 잡힙니다.

**prefill_buckets** — `(batch_size, context_length)`
- 보통 `batch_size=1`, context는 짧은 값들: `(1, 128), (1, 256), (1, 512), (1, 1024)`
- 더 긴 프리필이 자주 들어오면 (예: 4K 프롬프트가 흔함) 그 길이도 추가
- 무작정 늘리면 빌드 시간↑

**decode_buckets** — `(batch_size, context_length)` (생성 모델 필수)
- 서빙에서 기대하는 (동시 처리량 × 그 시퀀스의 최대 컨텍스트) 조합을 적으세요
- 예시 조합:
  - 단일 사용자 긴 컨텍스트만: `(1, 1024), (1, 4096), (1, 16384)`
  - 다수 사용자 짧은 컨텍스트: `(32, 1024), (32, 2048)`
  - 둘 다 가능성: 두 패턴 + 사이 값
- 모든 (batch, ctx)를 다 빌드할 필요 없습니다 — **서빙이 가장 가까운 버킷으로 라우팅**

**append_buckets** — `(batch_size, attention_size, input_ids_size)` (선택)
- prefix-cache 확장 — 캐시된 prefix에 새 토큰 묶음을 append하는 시나리오
- 제약: `attention_size > input_ids_size`
- CLI로 못 줌 → Python API 사용

**tokenwise_seq_lens** — int 튜플 (선택)
- composable kernel 빌드에 사용
- 보통 작은 정수들 `(1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024)`
- CLI로 못 줌 → Python API 사용

> 참고: append/tokenwise를 빼면 위 검증의 "combined max_executable_len" 계산 결과에
> 따라 빌드가 거부될 수 있습니다. 거부되면 Python API로 4종 다 채워 줘야 합니다.

### 6.4 CLI 빠른 예시 — `Mistral-7B-v0.3` (가상)

`mistralai/Mistral-7B-v0.3`은 `model_type=mistral` → preset 없음.

```bash
furiosa-llm build mistralai/Mistral-7B-v0.3 \
    ~/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/mistral-7b-tp8 \
    -tp 8 \
    --max-model-len 8192 \
    -pb 1,128 -pb 1,256 -pb 1,512 -pb 1,1024 \
    -db 1,1024  -db 1,2048  -db 1,4096  -db 1,8192 \
    -db 4,1024  -db 4,2048  -db 4,4096 \
    -db 16,1024 -db 16,2048 \
    -db 32,1024
```

이 명령으로 prefill 4개 + decode 9개가 빌드됩니다. tokenwise/append은 안 줬으니,
검증을 통과 못 하면 다음 절(Python API)로 가셔야 합니다.

### 6.5 Python API로 더 세밀하게

```python
from furiosa_llm.artifact import ArtifactBuilder
from furiosa_llm.artifact.types.config import (
    ModelConfig, ParallelConfig, BucketConfig,
)

builder = ArtifactBuilder(
    model_id_or_path="mistralai/Mistral-7B-v0.3",
    model_config=ModelConfig(max_model_len=8192),
    parallel_config=ParallelConfig(tensor_parallel_size=8, pipeline_parallel_size=1),
    bucket_config=BucketConfig(
        prefill_buckets=[(1, c) for c in (128, 256, 512, 1024)],
        decode_buckets=[
            (1, 1024), (1, 2048), (1, 4096), (1, 8192),
            (4, 1024), (4, 2048), (4, 4096),
            (16, 1024), (16, 2048),
            (32, 1024),
        ],
        append_buckets=[
            # (batch, attention_size, input_ids_size). attention_size > input_ids_size.
            (1, 256, 128), (1, 512, 128), (1, 512, 256),
            (1, 1024, 128), (1, 1024, 256), (1, 1024, 512),
            (1, 2048, 128), (1, 2048, 256), (1, 2048, 512), (1, 2048, 1024),
            (1, 4096, 128), (1, 4096, 1024),
            (1, 8192, 128), (1, 8192, 1024),
        ],
        tokenwise_seq_lens=(1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024),
    ),
)
builder.build("/path/to/artifacts/mistral-7b-tp8")
```

`builder.py:84-170` `ArtifactBuilder` 시그니처와 일치하는 인자입니다.

---

## 7. 메모리 관련 주의

### 7.1 빌드 시 (host RAM)

- 큰 모델 + 큰 버킷은 single worker가 수십~100GB+ 소비할 수 있습니다
  (Qwen3-32B-FP8 default preset 빌드 시 실측 ~107GB).
- `--num-pipeline-builder-workers`, `--num-compile-workers` 기본값이 **모두 1**이라
  이미 직렬입니다. 늘리면 빠르지만 워커당 메모리가 곱으로 늘어요.
- 빌드 OOM 회피책 1순위는 swap 확보 **+ Ray 메모리 감시 끄기**입니다. 단순 swap만으로는
  부족합니다 (아래 ⚠️ 참고):
  ```bash
  # 1) swap 확보 (재부팅 전까지 유효; /etc/fstab에 등록하면 영구)
  sudo fallocate -l 64G /swapfile
  sudo chmod 0600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile

  # 2) Ray 자체 OOM-kill 끄고 빌드 실행 (둘 중 하나)
  RAY_memory_monitor_refresh_ms=0 furiosa-llm build ...        # 감시 자체 끔
  # 또는
  RAY_memory_usage_threshold=0.99 furiosa-llm build ...        # 임계만 올림
  ```

  > ⚠️ **swap 함정 1 (Ray 자체 kill)** — Ray는 기본 `memory_usage_threshold=0.95`로
  > RAM 사용량만 보고 worker를 proactive하게 죽입니다. **swap을 거의 안 봅니다.**
  > 그래서 swap을 추가해도 RAM이 95%를 치면 OS가 swap을 쓰기 전에 Ray가 worker를
  > 먼저 죽여 OOM이 그대로 납니다.
  > 실측: 32B FP8 빌드에 swap 64GB 추가했지만 Ray 단독 kill 때문에 swap이 단 8MB만 쓰임.
  > 환경변수로 Ray 감시를 꺼야 swap이 실제로 동원됩니다.

  > ⚠️ **swap 함정 2 (OS 커널 OOM killer)** — Ray 감시를 꺼도 **Linux 커널의 OOM
  > killer**는 여전히 살아 있습니다. RAM 사용량이 임계 근처에 가면 커널이 가장 무거운
  > 프로세스를 죽이는데, 이때 systemd가 그 프로세스가 속한 스코프(`tmux-spawn-*.scope`)
  > 전체를 정리해버려서 **tmux 세션 자체가 통째로 사라집니다.**
  > 실측 (`journalctl`): `tmux-spawn-...scope: A process of this unit has been killed
  > by the OOM killer. ... Failed with result 'oom-kill'.` — tracing 6% 부근에서
  > `build_for_bucket` actor가 100GB+에 도달, swap 1.8GB 정도 동원된 시점에 커널이 정리.

  > ⚠️ **swap 함정 3 (systemd-oomd, PSI 기반)** — 위 둘을 다 막아도 **`systemd-oomd`**
  > 라는 또 다른 userspace OOM daemon이 따로 동작합니다. RAM·I/O **압력(PSI)** 이
  > 임계를 치면 OS가 swap을 본격 쓰기 *전에* cgroup 단위로 통째로 죽여요. 그래서
  > swap 추가가 **무의미**할 수 있습니다.
  > 실측 (`journalctl`): `systemd-oomd killed 291 process(es) in this unit. Failed with
  > result 'oom-kill'.` — `--max-model-len 16384`로 줄여 빌드해도 ~3시간 50분 후
  > swap 1.5GB만 쓴 상태에서 systemd-oomd가 tmux 스코프 통째로 정리.
  > → 회피: `sudo systemctl stop systemd-oomd` (재부팅 시까지) 또는 영구 disable.
  > 또는 `--max-model-len`을 더 줄여 actor 메모리 자체를 임계 밑으로 떨어뜨림.

  > **종합 — OOM-kill 3단 방어 체크리스트** (32B급 빌드 기준):
  > 1. Ray monitor 끄기: `RAY_memory_monitor_refresh_ms=0` env var
  > 2. swap 확보: `sudo swapon /swapfile` (최소 64GB+)
  > 3. systemd-oomd 중지: `sudo systemctl stop systemd-oomd`
  > 4. 그래도 안 되면 `--max-model-len`을 단계적으로 축소

- `--max-model-len` 축소 전략: 모델 native보다 작게 설정 → `presets.py:309-311`의 필터로
  큰 버킷이 빌드 대상에서 빠짐. 단 sweep/서빙에서 그 컨텍스트 길이를 더 이상 처리 못 하므로
  벤치마크 요구(가장 긴 프롬프트 + max_tokens)를 만족하는 선에서 가장 작게 잡는 게 안전.

### 7.2 서빙 시 (NPU HBM)

- 1장 ≈ 48GB / 2장 ≈ 96GB / 4장 ≈ 192GB
- 한 카드 안에 들어가야 할 것 = 모델 weight (tp/pp로 쪼개진 몫) + KV cache + 컴파일 메타
- weight 사이즈 ≈ 파라미터 수 × 1 byte(FP8) 또는 2 byte(bf16)
- KV cache per token (Qwen3-32B 예): `2 × num_layers × num_kv_heads × head_dim × 2 byte(bf16)`
- 서빙 HBM OOM이 뜨면 → `--max-model-len`을 줄여 재빌드하거나 카드 수를 늘립니다.

---

## 8. 자주 보는 에러와 조치

| 메시지 (요약) | 위치 | 조치 |
|---|---|---|
| `tensor_parallel_size=X is not supported. Supported values are [4, 8, 32].` | `validator.py:252` | tp를 {4,8,32} 중 하나로 |
| `The parallel configuration requires N RNGD device(s) ... but at most 8 devices are supported.` | `validator.py:262` | tp 또는 pp 줄이기 |
| `The HuggingFace model config is missing required fields: [...]` | `validator.py:65` | HF config 점검 또는 `--additional-model-config`로 보충 |
| `max_model_len=X exceeds max_position_embeddings=Y` | `resolver.py:152` | `--max-model-len`을 모델 한도 이하로 |
| `No bucket configuration provided and no matching bucket preset found for model_type=X` | `resolver.py:88-94` | `-pb`/`-db` 수동 지정 (이 문서 6절) |
| `Partial bucket configuration is not allowed.` | `resolver.py:79` | 모든 bucket 필드를 다 채우기 (또는 다 비우기) |
| `Generative models require at least one decode bucket.` | `validator.py:154` | `-db` 추가 |
| `Duplicate {prefill,decode,append} buckets found: ...` | `validator.py:273` | 중복 제거 |
| `prefill_buckets[i] context_length=X exceeds max_model_len=Y` | `validator.py:170` | 버킷 줄이거나 `--max-model-len` 키우기 |
| `The maximum executable length ... exceeds the model's maximum position embeddings` | `validator.py:194` | 버킷 줄이기 |
| `Ray killed N worker(s)` (OOM) | runtime | swap 추가 + `RAY_memory_monitor_refresh_ms=0` (또는 `RAY_memory_usage_threshold=0.99`) 환경변수 — swap만으로는 부족합니다 (7.1 ⚠️ 함정 1) |
| tmux 세션이 통째로 사라짐 + `journalctl`에 `tmux-spawn-*.scope: ... killed by the OOM killer` | runtime / systemd | Ray 감시까지 끈 상태에서도 OS 커널이 직접 정리. `--max-model-len`을 줄여 actor 메모리 자체를 축소해야 함 (7.1 ⚠️ 함정 2) |
| tmux 세션 사라짐 + `journalctl`에 `systemd-oomd killed N process(es)` | runtime / systemd | PSI 기반 userspace OOM daemon이 swap 쓰기 전에 정리. swap 추가만으론 안 됨. `sudo systemctl stop systemd-oomd` + `--max-model-len` 추가 축소 (7.1 ⚠️ 함정 3) |
| `NotImplementedError: Block slicing for {ModelClass} is not supported.` | `parallelize/block_slicer.py:727` | `-pp >1`을 줬는데 그 architecture가 pp 미지원. 2026.2.0 지원: Llama·GPTJ·Bert·Roberta만. 회피: `-pp 1` 빌드 + serve 시 `--devices npu:0,npu:1`로 dp 활용 (3절 ⚠️ 참고) |
| serving: `Required PEs: N, Actual: M` | runtime | 빌드 tp ≠ 가용 PE → `--devices` 늘리거나 작은 tp로 재빌드 |
| serving: HBM OOM | runtime | `--max-model-len` 줄여 재빌드 또는 카드 추가 |

---

## 9. 산출물

### 9.1 빌드 후 폴더 구조 (`builder.py:481-529`)

```
output_path/
├── artifact.json              ← 메타데이터, parallel_config, ...
├── binary_bundle.zip          ← 컴파일된 EDF (또는 풀어진 .edf, --no-bundle-binaries)
├── config.json                ← HF config 전체 저장
├── tokenizer.json / vocab / merges / ...
├── generation_config.json     ← 있을 때만
└── (선택) README, LICENSE     ← ArtifactConfig.copies_from_model로 지정 시
```

### 9.2 검증 명령

```bash
# parallel_config — tp/pp 값이 의도와 같은지
python3 -c "import json; \
a = json.load(open('PATH/artifact.json')); \
print(a.get('model', a).get('parallel_config'))"

# 스모크 서빙
furiosa-llm serve PATH --devices npu:0 --host 0.0.0.0 --port 8000
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"...","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
```

---

## 10. 캐시 위치

| 용도 | 경로 (기본) | 비고 |
|---|---|---|
| 빌드 캐시 (그래프/컴파일 단위) | `~/.cache/furiosa/llm` | `--cache-dir`로 변경. 재실행 시 재사용 |
| HF 모델 캐시 | `~/.cache/huggingface/hub` | `HF_HOME` 환경변수로 변경 |
| furiosa SDK venv | `~/furiosa/` | `lib/python3.12/site-packages/furiosa_llm/` 에 모든 furiosa 패키지가 있음. `~/furiosa/lib64` 는 `lib` 를 가리키는 심볼릭 링크라 어느 쪽 경로를 써도 동일 (multilib 호환용 보험) |

빌드 캐시는 `(model, parallel_config, bucket_config, ...)` 조합을 키로 쓰는 것 같습니다 —
같은 명령 재실행 시 이미 컴파일된 graph module은 그대로 재사용됩니다 (실측: 중단 후
재시작 시 `Try to load cached GraphModule at ~/.cache/furiosa/llm/graphmodules/...` 로그 나옴).

---

## 11. 코드 위치 빠른 참조

| 항목 | 파일:라인 |
|---|---|
| `ArtifactBuilder.__init__` | `artifact/builder.py:116` |
| `ArtifactBuilder.build` | `artifact/builder.py:315` |
| 워커 기본값 (모두 1) | `artifact/builder.py:319-320` |
| `resolve_max_model_len` | `artifact/resolver.py:125-159` |
| `ResolvedBuckets.resolve` | `artifact/resolver.py:34-122` |
| `find_preset` | `artifact/presets.py:268-295` |
| `filter_preset_by_max_model_len` | `artifact/presets.py:298-313` |
| `PRESET_REFS` (등록 목록) | `artifact/presets.py:210` |
| HF config 필수 필드 검증 | `artifact/validator.py:25-70` |
| Bucket 입력 검증 | `artifact/validator.py:73-126` |
| Resolved bucket 검증 | `artifact/validator.py:129-199` |
| ParallelConfig 검증 (tp/pp 한도) | `artifact/validator.py:234-267` |
| `BucketConfig` 스키마 | `artifact/types/config.py:58-91` |
| `AttentionBucket` 정의 | `metadata/config_types.py:141` |
| `NUM_PES_PER_NPU = 8` | `device.py:6` |

---

## 12. 빌드 가능 모델 인벤토리 (현재 머신 2장 + 125GB RAM + 200GB swap 기준)

종합 판정 = `model_type` 지원 × architecture pp 지원 × 1·2장 적재 가능 weight 크기 ×
빌드 host RAM 가용성.

### ✅ 확정 — 실제 빌드 완료

- `Qwen/Qwen3-32B-FP8` — qwen3, preset 정확 매칭, FP8 32GB, 1장 적재.
  `-tp 8 --max-model-len 16384` + 3중 방어(Ray off + oomd off + swap 200G) 검증됨.

### ✅ 가능 — preset 매칭 + 1장 적재 OK

| 모델 | model_type | weight | 권장 명령 요지 |
|---|---|---:|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | qwen2 | 1GB | `-tp 8` 매우 가벼움 |
| `meta-llama/Llama-3.1-8B-Instruct` | llama | 16GB bf16 | `-tp 8` (llama이라 `-pp 2`도 OK) |
| `LGAI-EXAONE/EXAONE-4.0-32B` (FP8) | exaone4 | 32GB FP8 | `-tp 8 --max-model-len 16384` + 3중 방어 |
| `Qwen3-Embedding-8B` / `Qwen3-Reranker-8B` (HF 원본) | qwen3 | 16GB bf16 | `-tp 8` 가벼움 |

### ✅ 가능 — HF에 FP8 변형 이미 있음 (양자화 단계 생략 가능)

Qwen이 공식 출시한 FP8 변형들 — 다운로드 즉시 빌드 가능. fine-grained FP8 dynamic, block_size 128로 furiosa 호환:

| 모델 | size | 비고 |
|---|---:|---|
| `Qwen/Qwen3-8B-FP8` | 8GB FP8 | 매우 가벼움 |
| `Qwen/Qwen3-30B-A3B-FP8` | 30GB FP8 MoE | preset 정확 매칭 |
| `Qwen/Qwen3-32B-FP8` | 32GB FP8 | (검증됨) |
| `BCCard/Qwen2.5-Coder-32B-Instruct-FP8-Dynamic` | 32GB FP8 | **3rd-party** Coder-32B FP8 — qwen2 preset 부적합이라 `-pb`/`-db` 수동 권장 |

### ✅ 가능 — 까다로움 (양자화 또는 큰 host RAM 필요)

| 모델 | model_type | 필요 조건 |
|---|---|---|
| `meta-llama/Llama-3.3-70B-Instruct` | llama | **FP8 양자화 필수** (bf16 140GB→2장 못 들어감). FP8 70GB → `-tp 8 -pp 2` (Llama이라 pp OK). 빌드 host RAM 빠듯할 수 있음 |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | qwen3_moe | preset 정확 매칭. bf16 1장 불가, **FP8 양자화 후 1장 가능 추정** |

### ⚠️ preset 부적합 — 수동 버킷 (`-pb`/`-db`) 필요

`qwen2` model_type은 preset이 0.5B용 한 개뿐이라 큰 Qwen2.5 변형들은 매칭은 되지만 버킷 부적절.
`Qwen2ForCausalLM`은 pp 미지원이라 무조건 1장 fit 해야 함.

| 모델 | weight (bf16) | 1장 fit? | 결론 |
|---|---:|---|---|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 3GB | ✅ | bf16 그대로 빌드+서빙 OK |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 14GB | ✅ | bf16 그대로 OK |
| `Qwen/Qwen2.5-Coder-14B-Instruct` | 28GB | ⚠️ tight (KV 여유 적음) | bf16 가능, `--max-model-len` 짧게(4K~8K) |
| `Qwen/Qwen2.5-Coder-32B-Instruct` | 64GB | ❌ | bf16 빌드 자체는 되지만 **서빙 불가** — 직접 FP8 양자화(→32GB) 후 재빌드해야 1장 서빙 |

> ⚠️ **"빌드 가능 ≠ 우리 머신서 서빙 가능"** — 빌드는 host CPU/RAM에서 AOT 컴파일이라
> HBM 체크 없이 통과합니다. 서빙 시점에 `Required PEs` 또는 HBM OOM이 뜰 수 있어요.
> bf16 32B+ 모델은 빌드는 되지만 1장 서빙 불가이고, qwen2/qwen3 등 pp 미지원
> architecture는 2장으로 분산도 못 하니, **FP8 양자화 단계가 추가로 필요**합니다.
> Qwen2.5-Coder/Qwen3-Coder는 Qwen이 FP8 변형을 공식 출시하지 않아서 직접 양자화 필요
> (`transformers.FineGrainedFP8Config`, `docs/COMPILING_MODELS.md` 2절).

### ⚠️ preset 없는 model_type — 항상 수동 버킷

`mistral`, `phi3`, `gpt_oss`, `qwen3_vl`, `exaone`, `exaone_moe`, `mllama4` — model_type 지원은 되지만
preset 등록 없음 → 무조건 `-pb`/`-db` 수동. 6절 가이드 참고.

### ❌ 빌드 불가

| 모델 | 이유 |
|---|---|
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | `model_type=qwen3_next` — SDK 2026.2.0 미지원 |
| 70B+ bf16 그대로 (Llama 외) | 1장 못 들어가고 pp 미지원이면 답 없음 |
| 80B+ dense | 2장(96GB) 초과, 4장 이상 머신 필요 |
| `furiosa-ai/*` prebuilt 재빌드 | 이미 컴파일된 binary + 원본 weight 미포함 — HF 원본부터 다시 |

### 새 모델 빠른 판정 순서

```
HF config.json → model_type 확인
  ├─ SDK 미지원 → ❌
  └─ 지원 →
      ├─ preset 정확 매칭? → default 빌드 시도
      └─ 미매칭 → -pb/-db 수동
              ↓
        weight 크기 (params × 1B FP8 / 2B bf16)
          ├─ ≤32GB → 1장 OK
          ├─ 32~64GB → Llama면 pp=2, 아니면 FP8 양자화
          └─ >96GB → 우리 머신엔 불가
              ↓
        빌드 host RAM 검토 (--max-model-len으로 조절)
```

---

## 13. 관련 문서

- [`README.md`](README.md) — 측정 파이프라인·orchestrator 사용법
- [`BUILD_FLOW.md`](BUILD_FLOW.md) — `builder.py / validator.py / resolver.py / presets.py` 호출 순서·역할 한 흐름으로 정리
- [`BUILD_COMPIL.md`](BUILD_COMPIL.md) — Pipeline build vs Compile 두 단계 차이 자세히
- [`README_preset.md`](README_preset.md) — `presets.py`의 버킷 4종, fmt 지시문, find_preset 매칭 등
- [`README_config.md`](README_config.md) — HF model `config.json` 필드, `max_position_embeddings` vs `--max-model-len`, FP8 `quantization_config.fmt` 등
- [`README_runcode.md`](README_runcode.md) — `furiosa-llm serve` 옵션·curl·OpenAI SDK 호출 자세히, serve-time 에러
- [`docs/COMPILING_MODELS.md`](docs/COMPILING_MODELS.md) — 다운로드부터 등록까지 단계별 튜토리얼
- [`docs/RUNNING_BENCHMARKS.md`](docs/RUNNING_BENCHMARKS.md) — 벤치마크 실행
- Furiosa 공식 문서:
  - 모델 준비: https://developer.furiosa.ai/latest/en/furiosa_llm/model-preparation.html
  - 병렬화: https://developer.furiosa.ai/latest/en/furiosa_llm/model-parallelism.html
  - 지원 모델: https://developer.furiosa.ai/latest/en/overview/supported_models.html
