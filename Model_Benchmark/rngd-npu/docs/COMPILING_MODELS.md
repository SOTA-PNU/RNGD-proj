# HF 모델 컴파일해서 RNGD에서 실행 — 자가 수행 가이드

출처:
- 모델 준비: https://developer.furiosa.ai/latest/en/furiosa_llm/model-preparation.html
- 병렬화: https://developer.furiosa.ai/latest/en/furiosa_llm/model-parallelism.html
- 지원 모델: https://developer.furiosa.ai/latest/en/overview/supported_models.html

워크플로: `HF 모델` → `[선택] FP8 양자화` → `furiosa-llm build` → `아티팩트` → `furiosa-llm serve`

각 단계에 **확인** = 성공 판정, **실패 시** = 증상별 조치.

---

## 지원 아키텍처

`furiosa-llm build`가 받는 model_type (SDK 2026.2.0 코드 기준):
`llama` `qwen2` `qwen3` `qwen3_moe` `exaone4` `gpt2` `gpt_oss`

| 용도 | model_type / 아키텍처 | 공식 상태 |
|---|---|---|
| text-gen (decoder) | `llama` `qwen2` `qwen3` `exaone4` | 검증·prebuilt 제공 |
| 임베딩 | `qwen3` / `Qwen3Model` | 검증·prebuilt 제공 |
| 리랭킹 | `qwen3` / `Qwen3ForSequenceClassification` | 검증·prebuilt 제공 |
| 미검증 | `qwen3_moe` `gpt2` `gpt_oss` | 공식 "planned". SDK 코드엔 존재(`qwen3_moe`는 버킷 프리셋도) — 빌드 시도는 되나 성공·정확도 미보장 |

- `qwen3_moe`는 빌드(컴파일)는 통과하지만 serve가 안 됩니다(Qwen3-Coder-30B-A3B-FP8 실측, 2026-06-04). 네이티브 런타임에 FP8 MoE 커널이 없어 `NativeLLMEngine` 생성 시 `Unsupported model metadata { Qwen3Moe, FP8 }`로 패닉합니다(`furiosa-generator/src/next_gen/hf_compat_next_gen.rs:367`). 즉 "컴파일 통과 ≠ serve 가능"입니다. 상세는 `info/README_build.md`(qwen3_moe 항목)를 참고하십시오.

- 임베딩/리랭킹은 pooling task — 빌드·서빙 옵션이 아래 decoder 흐름(1~6절)과 다름.
- 자동 버킷 프리셋 보유 model_type (`furiosa_llm/artifact/presets.py`): `qwen2` `exaone4` `llama` `qwen3` `qwen3_moe`. 프리셋과 `(model_type, hidden_size, intermediate_size)`가 일치하면 버킷 자동, 아니면 `-pb`/`-db` 수동.

```bash
python3 -c "
from huggingface_hub import hf_hub_download; import json
print(json.load(open(hf_hub_download('Qwen/Qwen2.5-1.5B-Instruct','config.json')))['model_type'])"
```
**확인**: 출력이 위 목록에 있어야 함. 없으면 빌드 불가.

## PE / 메모리 예산

- RNGD 1장 = 8 PE. 이 머신 = 2장 = **16 PE**. 빌드 시 `tp×pp ≤ 16`.
- 1장 HBM ≈ 48GB (관측: 8B bf16 weight+KV로 1장 ~43GB 사용).
- weight 메모리 ≈ 파라미터수 × (bf16: 2 byte / FP8: 1 byte).
- 서빙 시 HBM = weight + KV cache. 둘 합이 `tp` PE 분량 HBM 안에 들어가야 함.

| 모델 | dtype | weight | 권장 tp | 카드 |
|---|---|--:|--:|--:|
| ~1.5B | bf16 | ~3GB | 4–8 | ≤1 |
| ~8B | bf16 | ~16GB | 8 | 1 |
| ~32B | bf16 | ~64GB | 16 | 2 |
| ~32B | FP8 | ~32GB | 8–16 | 1–2 |

---

## 0. 환경

```bash
source ~/furiosa/bin/activate
```

## 1. 모델 다운로드

```bash
pip install --upgrade huggingface_hub
hf auth login --token $HF_TOKEN          # gated 모델만 필요
hf download "Qwen/Qwen2.5-1.5B-Instruct"
```
**확인**: `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/` 에 `*.safetensors` 존재.
**실패 시**: gated 모델 403 → HF 사이트에서 라이선스 동의 + 토큰 재확인.

## 2. (선택) FP8 양자화

bf16으로 쓸 거면 건너뜀 → 3절로. FP8 = HBM 절반·속도↑ (prebuilt 방식).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

model_id  = "Qwen/Qwen2.5-1.5B-Instruct"
save_path = "./qwen2.5-1.5b-fp8"

quantization_config = FineGrainedFP8Config(
    activation_scheme="dynamic",
    weight_block_size=(128, 128),
)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto",
    quantization_config=quantization_config, torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
```
- furiosa 지원 = fine-grained FP8 **dynamic**. HF의 임의 FP8 repo는 양자화 방식이 다를 수 있음 → 위 방식으로 직접 양자화 권장.
- `device_map="auto"`: GPU 없으면 CPU(느림). host RAM에 모델 bf16 전체가 올라가야 함.

**확인**: `save_path`에 `*.safetensors` + `config.json` 생성, `config.json`에 `quantization_config` 포함.
**실패 시**: host RAM 부족 → bf16으로 진행(2절 생략) 또는 더 작은 모델.

## 3. 아티팩트 빌드

입력 = HF id (bf16) 또는 2절 양자화 결과 로컬 경로.

### 3a. CLI

```bash
furiosa-llm build \
    Qwen/Qwen2.5-1.5B-Instruct \          # 또는 ./qwen2.5-1.5b-fp8
    ~/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen2.5-1.5b \      # 출력 경로
    -tp 8 \
    --max-model-len 4096 \
    --num-compile-workers 4
```

### 3b. Python API

```python
from furiosa_llm.artifact import ArtifactBuilder, ModelConfig, ParallelConfig

builder = ArtifactBuilder(
    model_id_or_path="Qwen/Qwen2.5-1.5B-Instruct",
    model_config=ModelConfig(max_model_len=4096),
    parallel_config=ParallelConfig(tensor_parallel_size=8, pipeline_parallel_size=1),
)
builder.build("/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen2.5-1.5b")
```

| CLI | Python | 의미 |
|---|---|---|
| `-tp N` | `ParallelConfig(tensor_parallel_size=N)` | PE 수 (기본 8) |
| `-pp N` | `ParallelConfig(pipeline_parallel_size=N)` | pipeline 단수 (기본 1) |
| `--max-model-len N` | `ModelConfig(max_model_len=N)` | 최대 context |
| `-pb b,c` / `-db b,c` | `BucketConfig(prefill_buckets=[(b,c)], decode_buckets=[(b,c)])` | 버킷 (미지정 시 2026.2+ 자동 preset) |
| `--num-compile-workers N` | — | 컴파일 병렬도 |
| `--trust-remote-code` | `ModelConfig(trust_remote_code=True)` | HF 커스텀 코드 |
| `--cache-dir DIR` | — | 빌드 캐시 (기본 `~/.cache/furiosa/llm`) |

**확인**:
- 출력 경로에 `artifact.json` + `binary_bundle.zip` + `config.json` + 토크나이저 생성.
- tp 값 확인:
  ```bash
  python3 -c "import json;print(json.load(open('$HOME/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen2.5-1.5b/artifact.json'))['model']['parallel_config'])"
  ```
- build는 host(CPU/RAM) AOT 컴파일 — NPU 불필요. 빌드 중 다른 터미널에서 `furiosa-smi info` 시 Power/Temp 변화 없어야 함.

**실패 시**:
- host RAM OOM → `--num-compile-workers`, `--num-pipeline-builder-workers` 를 1로.
- 컴파일 에러 → `tp` 또는 `--max-model-len` 조정 후 재시도.

### 3c. 같은 모델을 컨텍스트 길이만 다르게 빌드하면 artifact.json이 무엇이 달라지나

같은 Qwen3-32B-FP8을 전체 컨텍스트(40960)와 16384로 각각 빌드한 두 아티팩트(`artifacts/qwen3-32b-fp8-tp8`, `artifacts/qwen3-32b-fp8-tp8-16k`)의 `artifact.json`을 비교해 보면, 실제로 달라지는 곳은 버킷 목록 하나뿐입니다.

| artifact.json 항목 | 두 빌드 비교 |
|---|---|
| `version`, `generator_config` | 동일 |
| `metadata` | `artifact_id`·`timestamp`만 다릅니다(빌드할 때마다 새로 생성). `furiosa_llm_version`·`furiosa_compiler_version`은 동일 |
| `model.model_metadata` (hf_configs·양자화 설정 등) | 동일. `max_position_embeddings`도 둘 다 40960 그대로입니다. 모델이 아는 최대 길이 자체는 안 바뀝니다 |
| `model.parallel_config` | 동일 (`tensor_parallel_size=8`) |
| `model.pipeline_metadata_list[].attention_buckets` | 여기만 다릅니다. 16k 빌드는 `attention_size`가 16384를 넘는 버킷(32768·40960)을 전부 뺍니다 |
| `model.pipeline_metadata_list[].tokenwise_buckets` | 동일 (둘 다 input_size 1~1024 12종). tokenwise 값은 전부 16384 이하라 잘릴 게 없습니다 |
| 같은 폴더의 `config.json`·`generation_config.json`·`tokenizer*`·`chat_template.jinja`·weight safetensors | 모두 동일(weight는 폴더명 해시까지 같음) |

버킷 수는 전체(40960) 빌드가 116개, 16k 빌드가 98개로 18개가 빠졌습니다. 빠진 18개는 모두 `attention_size`가 32768 또는 40960인 긴 컨텍스트용 버킷입니다.

| batch | 40960 빌드 attention_size | 16k 빌드 attention_size |
|---|---|---|
| 1 | …, 8192, 16384, 32768, 40960 | …, 8192, 16384 |
| 2 | …, 8192, 16384, 32768, 40960 | …, 8192, 16384 |
| 4 | …, 8192, 16384, 32768, 40960 | …, 8192, 16384 |
| 8 / 16 / 32 | …, 8192, 16384, 32768 | …, 8192, 16384 |
| 64 | …, 8192, 16384 | …, 8192, 16384 (동일) |
| 128 / 256 | 1024, 2048, 4096 | 1024, 2048, 4096 (동일) |

`--max-model-len`을 줄이면 SDK가 그 길이를 넘는 버킷을 잘라내기 때문입니다. `furiosa_llm/artifact/presets.py`의 `filter_preset_by_max_model_len()`이 버킷의 `attention_size`가 `max_model_len`보다 큰 것을 전부 제거하고(presets.py:404), 이 함수는 빌드 중 `resolver.py:233`에서 불립니다. 그래서 16k 빌드에서는 32768·40960 버킷만 사라지고 나머지는 그대로 남습니다.

버킷이 줄면 컴파일되는 그래프 수도 줄어서 `binary_bundle.zip`도 작아집니다(관측: 40960 빌드 약 127MB, 16k 빌드 약 91MB). `artifact.json` 크기도 약 24.5MB에서 약 21.8MB로 줄었습니다.

정리하면, 같은 모델을 컨텍스트 길이만 다르게 빌드한 두 아티팩트의 차이는 지원하는 최대 입력 길이(버킷)가 전부이고, 모델 가중치·구조·토크나이저는 똑같습니다.

#### 버킷이 줄면 서빙할 때 생기는 제약

서빙 시 받을 수 있는 입력 길이가 그만큼 줄어듭니다. furiosa-llm은 아티팩트를 로드할 때 컴파일된 버킷 중 가장 큰 `attention_size`를 그대로 입력 한계로 잡습니다(`utils.py:458`의 `compute_bucket_lengths`로 계산해서 `api.py:374`의 `prompt_max_seq_len`에 넣음). HF config의 `max_position_embeddings`(두 빌드 다 40960)가 아니라 버킷이 한계를 정합니다.

- 전체(40960) 빌드: 입력 프롬프트 최대 40960 토큰
- 16k 빌드: 입력 프롬프트 최대 16384 토큰

이 한계를 넘는 요청이 오면 잘리지 않고 에러로 거부됩니다. chat과 completions 양쪽 다 똑같이 막습니다(`server/serving_chat.py:217`, `server/serving_completions.py:105`, 엔진 쪽 `llm_engine.py:388`·`api.py:460`). 에러 메시지는 다음과 같습니다.

```
This model's maximum input context length is 16384 tokens.
However, your messages resulted in N tokens. Please reduce the length of the messages.
```

생성(decode) 쪽도 가장 큰 decode 버킷이 16384라, 프롬프트와 생성 토큰을 합친 길이가 16384를 넘는 건 16k 빌드에서 불가능합니다. 그래서 프롬프트가 16384에 가까우면 생성할 여유 토큰이 거의 안 남습니다. 현재 SDK는 입력 길이만 미리 검사하고 "입력 + max_tokens" 합계는 미리 검사하지 않습니다(`api.py:598`에 FIXME로 남아 있음). 경계 근처 동작은 네이티브 런타임이 처리합니다.

반대로 16384 이하 길이의 요청에 대해서는 두 빌드가 완전히 같습니다. 16384까지의 prefill·decode·extend 버킷이 동일하고 batch 버킷(1~256)도 같아서, 속도나 패딩, 동시 처리에 차이가 없습니다. 즉 버킷을 줄인 대가는 "긴 컨텍스트를 못 쓴다" 하나뿐이고, 16k 안에서만 쓰면 손해가 없습니다. 덤으로 컴파일된 그래프가 적어서 `binary_bundle.zip`과 로드가 가벼워집니다.

## 4. 서빙

```bash
furiosa-llm serve ~/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen2.5-1.5b \
    --devices npu:0 --host 0.0.0.0 --port 8000
```
**확인**: 로그에 `Uvicorn running on http://0.0.0.0:8000`.
**실패 시**:
- `Required PEs: N, Actual: M` → 빌드 tp ≠ `--devices` PE 수. `--devices`를 늘리거나 작은 tp로 재빌드.
- HBM OOM → `--max-model-len` 축소, 또는 FP8로 재빌드.

## 5. 테스트

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-1.5b","messages":[{"role":"user","content":"Write a Python function to reverse a string."}],"max_tokens":128}' \
  | python3 -m json.tool
```
**확인**: `/v1/models`에 모델 표시, `/v1/chat/completions`가 코드 포함 응답 반환.

## 6. 벤치마크 프레임워크에 등록

`configs/models.yaml`의 `models:`에 추가 (`id` = 로컬 아티팩트 절대경로):

```yaml
  - id: /home/jun/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen2.5-1.5b
    revision: null
    role: main
    gen: true
    enabled: true
    serve_args: []
```
```bash
python orchestrator.py configs/models.yaml --tasks tps,sweep --models qwen2.5-1.5b
```

---

## 7. prebuilt 32B/70B를 더 적은 카드로 재빌드 시도

prebuilt 아티팩트(`furiosa-ai/Qwen3-32B-FP8` 등)는 `binary_bundle.zip`이 **tp=32(4장)로
컴파일**돼 있음. `artifact.json`의 숫자만 바꾸는 건 불가 — 메타데이터일 뿐 binary는
그대로 32 PE용. 또한 prebuilt repo엔 `binary_bundle.zip`만 있고 재빌드용 safetensors
weight가 **없음** → 원본 HF weight에서 다시 시작해야 함.

**실측 (2026-05-18 · `artifacts/qwen3-32b-fp8-tphack/`):** prebuilt `furiosa-ai/Qwen3-32B-FP8`의
`binary_bundle.zip`을 symlink하고 `artifact.json` 메타만 2장용으로 고친 artifact를 2장에 serve →
패닉 (`tphack_serve.log`):

```
panicked at itertools .../zip_eq_impl.rs: .zip_eq() reached end of one iterator before the other
```

→ 메타데이터 해킹은 불가로 **확인됨**. 아래 풀 재빌드만 유효.

### 절차 (Qwen3-32B를 2장=tp16으로)

```bash
source ~/furiosa/bin/activate

# 1) 원본 weight 다운로드 (bf16)
hf download Qwen/Qwen3-32B

# 2) 빌드 — 경로 A(bf16, 간단) 또는 B(FP8, HBM 절약) 중 택1

# A) bf16 그대로: 32B bf16 weight ~64GB → tp16(2장 HBM 합산 ~96GB)에 빠듯
furiosa-llm build Qwen/Qwen3-32B ~/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen3-32b-tp16 \
    -tp 16 --max-model-len 4096 --num-compile-workers 4

# B) FP8 후 빌드: 2절 FineGrainedFP8Config로 양자화(host RAM ~64GB+ 필요) → save_path
#    그 결과로 빌드
furiosa-llm build ./qwen3-32b-fp8 ~/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen3-32b-tp16 \
    -tp 16 --max-model-len 4096 --num-compile-workers 4

# 3) 서빙 (2장)
furiosa-llm serve ~/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen3-32b-tp16 --devices npu:0,npu:1
```

`-tp 8`(1장)로도 시도 가능하나 32B는 1장 HBM에 안 들어갈 가능성 높음 → tp16 우선.

### 확인 포인트 / 분기

| 단계 | 성공 | 실패 시그니처 → 조치 |
|---|---|---|
| build | `artifact.json`+`binary_bundle.zip` 생성, `parallel_config.tensor_parallel_size=16` | host RAM OOM → `--num-compile-workers 1 --num-pipeline-builder-workers 1` |
| build | 〃 | 컴파일 에러(tp16 버킷 미지원 등) → `--max-model-len` 축소, `-pb`/`-db` 수동 지정 |
| serve | `Uvicorn running` | `Required PEs: 16, Actual: N` → `--devices`를 npu:0,npu:1로 |
| serve | 〃 | HBM OOM → 경로 B(FP8)로 재빌드, `--max-model-len` 축소. 그래도 안 되면 **2장으론 불가** |
| 추론 | 5절 curl 정상 응답 | 응답 깨짐 → tp 변경으로 정확도 손상 가능, 다른 tp/버킷 재시도 |

### 주의

- furiosa가 prebuilt를 tp=32로 낸 데엔 이유(컴파일 버킷 제약·성능)가 있을 수 있음 → tp16 빌드/서빙 성공은 보장 안 됨. **해봐야 앎.**
- `-tp 16`을 SDK가 거부하면 `-tp 8 -pp 2`로 분해(8×2 = 16 PE = 2장) 후 재시도.
- 직접 빌드는 bf16/자가 FP8 → prebuilt FP8보다 성능 낮을 수 있음.
- 32B bf16 빌드는 host RAM을 크게 씀. RAM 부족 시 경로 B(FP8) 또는 swap 확보.

---

## 배포 (다른 머신으로)

```bash
tar czf qwen2.5-1.5b.tar.gz -C ~/RNGD-proj/Model_Benchmark/rngd-npu/artifacts qwen2.5-1.5b
# 대상 호스트에서:
tar xzf qwen2.5-1.5b.tar.gz
furiosa-llm serve ./qwen2.5-1.5b --devices npu:0
```

## 제약 / 트러블슈팅

| 증상 / 한계 | 조치 |
|---|---|
| `model_type` 미지원 | 지원 아키텍처 외는 빌드 불가 (SDK 2026.2.0) |
| FP8 외 양자화 | fine-grained FP8 dynamic만 지원 |
| gated 모델 다운로드 실패 | `hf auth login --token $HF_TOKEN` + HF 라이선스 동의 |
| build host OOM | `--num-compile-workers` / `--num-pipeline-builder-workers` 축소 |
| serve `Required PEs: N` | 빌드 tp ≠ 가용 PE → 작은 tp 재빌드 또는 `--devices` 조정 |
| serve HBM OOM | `--max-model-len` 축소, FP8 재빌드 |
| 커스텀 코드 모델 | `--trust-remote-code` |

## 자가 검증 체크리스트

```
[ ] 0  source ~/furiosa/bin/activate
[ ] 1  모델 다운로드 → *.safetensors 확인
[ ] 2  (선택) FP8 양자화 → save_path에 *.safetensors+config.json
[ ] 3  furiosa-llm build → artifact.json+binary_bundle.zip, parallel_config 값 확인
[ ] 3  빌드 중 furiosa-smi → NPU 점유 없음(host 컴파일) 확인
[ ] 4  furiosa-llm serve → "Uvicorn running" 확인
[ ] 5  curl /v1/chat/completions → 정상 응답 확인
[ ] 6  models.yaml 등록 → orchestrator.py 측정 확인
[ ] 7  (선택) 32B tp16 재빌드 → build/serve 성공 여부로 2장 가능성 판정
```
