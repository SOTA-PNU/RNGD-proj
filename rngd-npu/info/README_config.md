# HF model `config.json` — furiosa-llm 빌드 관점

HuggingFace 모델의 `config.json` 안에 들어 있는 필드들을 furiosa-llm 빌드 시 어떻게 쓰는지
정리한 문서입니다. 앞으로 `config.json` 관련 새 발견(모델별 차이, 필수 필드, 변환 규칙 등)은
모두 이 문서에 누적합니다.

소스 인용은 모두 `~/furiosa/lib/python3.12/site-packages/furiosa_llm/`을 기준,
SDK 2026.2.0.

---

## 1. furiosa-llm이 검증하는 필수 필드

`validator.py:25-70` `validate_hf_config()`가 빌드 시작 시 검사 — 하나라도 빠지면 즉시 에러:

| 필드 | 별명 (있는 경우) | 용도 |
|---|---|---|
| `max_position_embeddings` | — | `max_model_len` 기본값, 버킷 검증 |
| `num_hidden_layers` | `num_layers`, `n_layer` | pipeline parallel 분할, compiler 설정 |
| `hidden_size` | — | preset 매칭, compiler 설정 |
| `intermediate_size` | — | preset 매칭, compiler 설정 |

에러 메시지:
```
The HuggingFace model config is missing required fields: [...].
Please check the model config or set 'hf_overrides' in ModelConfig.
```
→ `--additional-model-config <key>=<value>`로 보충 가능.

---

## 2. `max_position_embeddings` ≠ `--max-model-len` (자주 헷갈리는 부분)

**다릅니다.** 관계는 "default fallback":

| 항목 | 의미 |
|---|---|
| `max_position_embeddings` | **모델 자체의 하드 천장.** config.json에 박혀 있고, 학습 시 정해진 positional encoding 인덱스 범위 |
| `--max-model-len` | **빌드 시 사용자가 선택하는 artifact의 컨텍스트 상한.** `max_position_embeddings`를 넘을 수 없음 |

### resolve 흐름 (`resolver.py:125-159`)

```python
def resolve_max_model_len(hf_config, max_model_len):
    max_pe = hf_config.max_position_embeddings
    if max_model_len is None:
        return max_pe                                      # ← 미지정 시 모델값 그대로
    if max_model_len > max_pe:
        raise ValueError("exceeds max_position_embeddings") # ← 모델 초과 금지
    return max_model_len
```

즉:
- `--max-model-len`을 안 주면 `max_position_embeddings` 값이 그대로 적용
- 주더라도 모델값 이하만 허용
- 정 더 크게 쓰고 싶으면 `--additional-model-config max_position_embeddings=N`으로 모델값 자체를
  덮어써야 함

### 효과 (`presets.py:298-313`)

resolve된 `max_model_len`은 **preset 버킷 필터링에도 영향**을 줍니다 — 이 값보다 큰
attention_size 가진 버킷은 preset에서 제거되고, 수동 지정도 거부됩니다(`validator.py:167-185`).
자세한 동작은 [`README_preset.md`](README_preset.md) 6절 참고.

---

## 3. `rope_scaling` — 컨텍스트 확장

RoPE(Rotary Position Embedding)에 스케일링 기법을 적용해 native 한도 이상의 컨텍스트를
처리할 수 있게 하는 설정.

- `null`(또는 누락) → native 한도(`max_position_embeddings`)만 사용
- 값이 있으면 → 그 기법(YaRN, linear, dynamic 등) 적용된 확장 컨텍스트 사용 가능

실측 데이터:

| 모델 | `max_position_embeddings` | `rope_scaling` | 비고 |
|---|---:|---|---|
| Qwen/Qwen3-32B-FP8 | 40960 | `null` | native 40k만 (HF 카드는 YaRN으로 131k 가능하다고 함 — 별도 적용 필요) |
| furiosa-ai/Llama-3.1-8B-Instruct | 131072 | (확장 내장) | RoPE 확장이 base config에 박혀 max=131k |
| furiosa-ai/Llama-3.3-70B-Instruct | 131072 | (확장 내장) | 동일 |
| furiosa-ai/EXAONE-4.0-32B-FP8 | 131072 | (확장 내장) | sliding_window=4096 별도 |
| furiosa-ai/Qwen2.5-0.5B-Instruct | 32768 | `null` | native 32k |

---

## 4. `model_type` — 아키텍처 식별자

빌드가 받는 architecture를 결정. `furiosa.models.language.architecture/`에 그 model_type 모듈이
있어야 빌드 가능. SDK 2026.2.0 지원 목록:

```
llama  qwen2  qwen3  qwen3_moe  qwen3_vl
exaone  exaone4  exaone_moe
gpt_oss  mistral  mllama4  phi3
```

목록 외 model_type이면 빌드 자체가 시작되지 않습니다.

자세한 architecture별 차이·preset 매핑은 [`README_preset.md`](README_preset.md) 5절,
빌드 단계별 설명은 [`BUILD_COMPIL.md`](BUILD_COMPIL.md) 참고.

---

## 5. `architectures` — PP 가능 여부 결정

`config.json`의 `architectures` 필드는 HF Transformers의 ModelForCausalLM class 이름 리스트.
furiosa의 pipeline parallel(`-pp >1`) 가능 여부는 **`parallelize/block_slicer.py:677`의 dict 키와
이 class 이름이 일치하는지로 결정**됩니다.

지원 class (2026.2.0):
- `LlamaForCausalLM`
- `GPTJForCausalLM`
- `BertForQuestionAnswering`
- `RobertaForQuestionAnswering`

미지원(예: `Qwen3ForCausalLM`, `Exaone4ForCausalLM`, `Qwen2ForCausalLM`)에서 `-pp 2` 주면:
```
NotImplementedError: Block slicing for {Class} is not supported.
```
회피: pp 대신 dp(serve 시 `--devices npu:0,npu:1`)로 카드 추가.
자세한 건 [`README_build.md`](README_build.md) 3절.

---

## 6. `quantization_config` — FP8 양자화 정보

FP8 모델일 때만 존재. furiosa 호환 조건:
- `quant_method = "fp8"`
- `activation_scheme = "dynamic"`
- `weight_block_size = [128, 128]` (fine-grained block-wise)

### `fmt` 필드 — FP8 부동소수점 포맷

`quantization_config` 안의 `fmt`는 **FP8의 비트 배치 방식**을 지정합니다 (presets.py의
`# fmt: off`와는 완전히 다른 개념 — 그건 Python formatter 지시문).

| `fmt` 값 | 의미 | 어디 쓰이나 |
|---|---|---|
| `"e4m3"` | 4-bit exponent + 3-bit mantissa (+ sign bit) | 정밀도 우선, 가중치에 보편적 |
| `"e5m2"` | 5-bit exponent + 2-bit mantissa | 동적 범위 우선, 활성화 등 |
| `null` | 명시 안 됨 (= 기본 e4m3 추정) | 일부 모델 |

실측:

| 모델 | `quant_method` | `activation_scheme` | `weight_block_size` | `fmt` |
|---|---|---|---|---|
| Qwen/Qwen3-32B-FP8 | fp8 | dynamic | [128, 128] | **e4m3** |
| furiosa-ai/EXAONE-4.0-32B-FP8 | fp8 | dynamic | [128, 128] | null |

→ 두 모델 다 furiosa-llm이 그대로 빌드 입력으로 받을 수 있는 방식. fmt 명시 여부는
furiosa 빌드 동작과 직접 관련 없는 듯합니다(주로 transformers·SGLang·vLLM 등이 참고).

bf16 모델은 `quantization_config` 자체가 없습니다.

---

## 7. 기타 자주 보는 필드

| 필드 | 의미 | 비고 |
|---|---|---|
| `num_attention_heads` | attention head 수 | |
| `num_key_value_heads` | KV head 수 (GQA의 경우 attention head보다 작음) | KV cache 메모리 산출에 사용 |
| `head_dim` | head 하나의 차원 | 없으면 `hidden_size / num_attention_heads`로 추정 |
| `vocab_size` | tokenizer vocab 크기 | embedding·LM head 메모리에 영향 |
| `torch_dtype` | HF 저장 시 dtype (`bfloat16`/`float16` 등) | FP8 모델은 별개 — `quantization_config` 따로 |
| `tie_word_embeddings` | input embedding과 output projection 공유 여부 | true면 LM head 메모리 절약 |
| `sliding_window` | attention 윈도우 크기 (전체가 아닌 윈도우 내만 attend) | EXAONE 4 등 |
| `rope_theta` | RoPE base θ | |
| `architectures` | HF model class 이름 리스트 | pp 가능 여부 결정 (5절) |

---

## 8. 모델별 비교 (실측, HF 캐시 직접 추출)

### 8.1 `Qwen/Qwen3-32B-FP8` (현재 빌드 중인 모델)

```yaml
model_type: qwen3
architectures: ["Qwen3ForCausalLM"]
max_position_embeddings: 40960          # ← native 40k. rope_scaling 없음
rope_scaling: null
rope_theta: 1000000
hidden_size: 5120
intermediate_size: 25600
num_hidden_layers: 64
num_attention_heads: 64
num_key_value_heads: 8                  # GQA: 8 head 하나에 attention head 8개 묶임
head_dim: 128
vocab_size: 151936
torch_dtype: bfloat16                   # weight 저장 dtype (FP8은 quantization_config)
tie_word_embeddings: false
quantization_config:
  quant_method: fp8
  activation_scheme: dynamic
  weight_block_size: [128, 128]
  fmt: e4m3
```

특이점: native max가 **40960** (= 40×1024, 일반적인 2의 거듭제곱 아님). furiosa preset도
이 값에 맞춰 40960 attention 버킷을 따로 추가해 둠 — [`README_preset.md`](README_preset.md) 6절.

### 8.2 `furiosa-ai/Llama-3.1-8B-Instruct`

```yaml
model_type: llama
architectures: ["LlamaForCausalLM"]     # ← pp 지원
max_position_embeddings: 131072         # 128k (RoPE 확장 내장)
hidden_size: 4096
intermediate_size: 14336
num_hidden_layers: 32
num_attention_heads: 32
num_key_value_heads: 8                  # GQA
head_dim: 128
vocab_size: 128256
tie_word_embeddings: false
```

### 8.3 `furiosa-ai/Llama-3.3-70B-Instruct`

```yaml
model_type: llama
architectures: ["LlamaForCausalLM"]     # ← pp 지원
max_position_embeddings: 131072
hidden_size: 8192
intermediate_size: 28672
num_hidden_layers: 80
num_attention_heads: 64
num_key_value_heads: 8                  # 8B와 동일 GQA 8:1
head_dim: 128
vocab_size: 128256
```

(bf16 → 양자화 정보 없음. 빌드 시 FP8로 변환 필요.)

### 8.4 `furiosa-ai/EXAONE-4.0-32B-FP8`

```yaml
model_type: exaone4
architectures: ["Exaone4ForCausalLM"]   # ← pp 미지원
max_position_embeddings: 131072
hidden_size: 5120
intermediate_size: 27392
num_hidden_layers: 64
num_attention_heads: 40                 # Qwen3-32B(64)와 달리 40개
num_key_value_heads: 8
head_dim: 128
vocab_size: 102400
sliding_window: 4096                    # ← EXAONE 특유: 윈도우 attention
tie_word_embeddings: false
quantization_config:
  quant_method: fp8
  activation_scheme: dynamic
  weight_block_size: [128, 128]
  fmt: null                             # Qwen3은 "e4m3", EXAONE은 null
```

특이점:
- `sliding_window` 있음 → EXAONE은 전체 시퀀스가 아닌 4096 윈도우만 attend
- `quantization_config.fmt`가 `null` (Qwen은 `"e4m3"`) — 표기 누락이거나 다른 기본값일 수 있음

### 8.5 `furiosa-ai/Qwen2.5-0.5B-Instruct` (smoke 테스트용)

```yaml
model_type: qwen2
architectures: ["Qwen2ForCausalLM"]     # ← pp 미지원 (Qwen2도 미지원)
max_position_embeddings: 32768          # 32k native
hidden_size: 896                        # 매우 작음
intermediate_size: 4864
num_hidden_layers: 24
num_attention_heads: 14                 # 작은 모델은 head 수도 작음
num_key_value_heads: 2                  # GQA 14:2
vocab_size: 151936
tie_word_embeddings: true               # ← true: input/output embedding 공유 (메모리 절약)
```

특이점: `tie_word_embeddings = true`. 작은 모델에서 흔함. LM head 가중치를 input embedding과
공유해 메모리·파라미터 수 절감.

### 8.6 `furiosa-ai/Qwen3-Embedding-8B` / `furiosa-ai/Qwen3-Reranker-8B`

```yaml
model_type: qwen3                       # 텍스트 생성용 Qwen3와 같은 model_type
architectures: ["Qwen3ForCausalLM"]
max_position_embeddings: 40960
hidden_size: 4096                       # 8B 변형: 5120(32B)보다 작음
intermediate_size: 12288                # 25600(32B)의 절반 이하
num_hidden_layers: 36                   # 32B(64)보다 적음
num_attention_heads: 32
num_key_value_heads: 8
head_dim: 128
vocab_size: 151665 (Embedding) / 151669 (Reranker)
tie_word_embeddings: false
```

특이점: model_type은 `qwen3` 그대로지만 task가 다름 (causal LM 아니라 pooling).
furiosa preset에서 `(qwen3, hidden=4096, intermediate=12288)` 항목이 이 모델용
(`QWEN_3_8B_POOLING_PRESET`, prefill만 가지고 decode 없음 — pooling 모델은 생성이 아니므로).

---

## 9. KV cache 메모리 계산에 필요한 필드 (참고)

서빙 시 카드별 HBM 여유 계산할 때 자주 쓰입니다.

**KV cache per token (bf16 KV 기준):**
```
2 (K+V) × num_hidden_layers × num_key_value_heads × head_dim × 2 bytes
```

| 모델 | per token KV (bf16) | 16k context per request | 32k context per request |
|---|---:|---:|---:|
| Qwen3-32B-FP8 | 2 × 64 × 8 × 128 × 2 = 262144 B (256 KB) | 4 GB | 8 GB |
| Llama 3.1 8B | 2 × 32 × 8 × 128 × 2 = 131072 B (128 KB) | 2 GB | 4 GB |
| Llama 3.3 70B | 2 × 80 × 8 × 128 × 2 = 327680 B (320 KB) | 5 GB | 10 GB |
| EXAONE 4 32B | 2 × 64 × 8 × 128 × 2 = 262144 B (256 KB) | 4 GB | 8 GB |
| Qwen2.5-0.5B | 2 × 24 × 2 × (896/14) × 2 = 12288 B (12 KB) | 192 MB | 384 MB |

(sliding_window 있는 모델은 윈도우 내만 가지므로 실제 KV는 더 적을 수 있음.)

---

## 10. 코드 빠른 참조

| 항목 | 파일:라인 |
|---|---|
| HF config 필수 필드 검증 | `artifact/validator.py:25-70` |
| `resolve_max_model_len` | `artifact/resolver.py:125-159` |
| `--additional-model-config`로 config 덮어쓰기 처리 | `artifact/types/config.py:50` `hf_overrides` |
| `resolve_model_metadata` (config 로딩) | `artifact/resolver.py:246-321` |
| architecture별 pp 지원 판정 | `parallelize/block_slicer.py:677` `MODEL_ARCH_TO_BLOCK_SPLITTER_AND_WEIGHT_NODE_PATTERN` |

---

## 11. 관련 문서

- [`README_build.md`](README_build.md) — 빌드 옵션 전반, OOM 트러블슈팅
- [`README_preset.md`](README_preset.md) — `presets.py` 버킷 구조 (이 문서의 *프리셋 쪽* counterpart)
- [`BUILD_COMPIL.md`](BUILD_COMPIL.md) — Pipeline build vs Compile 단계 차이
- HF Transformers 공식 — config 필드 일반 문서: <https://huggingface.co/docs/transformers/main_classes/configuration>
