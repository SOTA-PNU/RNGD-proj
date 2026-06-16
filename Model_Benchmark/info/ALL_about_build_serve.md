# furiosa-llm build & serve 의 모든 것 — RNGD SDK 2026.2.0 정독

> 이 문서는 `furiosa-llm build` 와 `furiosa-llm serve` 가 **한 줄 명령 안에서 실제로
> 무슨 일을 하는지**를, 파일 경로·줄 번호·코드 역할·원리까지 빠짐없이 정리한 SDK 학습
> 노트입니다. 모든 서술은 **직접 실행한 로그와 코드 정독으로 추출한 사실**만 담았습니다.
>
> - SDK 루트: `/home/jun/furiosa/lib/python3.12/site-packages/` (이하 **SDK** 로 줄임)
> - 예시 모델(빌드 결과물): `Qwen3-Coder-30B-A3B-Instruct-FP8`
>   - 아티팩트 위치: `Model_Benchmark/rngd-npu/artifacts/qwen3-coder-30b-a3b-inst-fp8-tp8-65k-tc/`
>   - (이 아티팩트는 serve 게이트 통과를 위해 `model_type` 을 `qwen3` 로 위장한 사본입니다.
>     위장 배경은 [README_qwen3_coder_next.md](README_qwen3_coder_next.md) 참고)
> - 빌드 과정 실측 로그: `qwen3-next-proj/logs/build_trace_full.log`
>   (미니 qwen3_moe 모델 — **qwen3-coder-30b 과 동일한 코드 경로**를 분 단위로 재현한 것)
> - serve 부팅 실측 로그: `qwen3-next-proj/logs/serve_30b_tc_newpath.log` (실물 30B 부팅)
>
> 작성일 2026-06-10.

---

## 0. 큰 그림 — 등장인물과 비유

`furiosa-llm` 은 **HuggingFace 의 PyTorch 모델을 RNGD NPU 가 먹을 수 있는 형태로
"번역·인쇄"한 뒤(build), 그 인쇄물을 "낭독"하는(serve)** 두 단계로 나뉩니다.

| 비유 | build | serve |
|---|---|---|
| 큰 그림 | 악보(PyTorch)를 **연주 가능한 MIDI 시퀀스(EDF)로 편곡·인쇄** | 인쇄된 MIDI 를 **악기(NPU)로 실시간 연주** |
| 결과물 | `artifact.json` + `binary_bundle.zip`(EDF) + 가중치 | 토큰 스트림(HTTP 응답) |
| 무거운 곳 | 트레이싱(메모리) + 컴파일(시간) | 가중치 적재 + KV 캐시 할당 |
| 누가 일하나 | Python(furiosa_llm) + Ray + 네이티브 컴파일러 | Python(얇은 HTTP) + 네이티브 런타임 |

### 0-1. 핵심 파일 지도 (헷갈리는 이름 정리)

**Python (우리가 읽을 수 있음, `SDK/furiosa_llm/`):**

| 경로 | 한 줄 역할 |
|---|---|
| `cli/main.py` | `furiosa-llm` 의 서브커맨드(build/serve/...) 등록·분기 |
| `cli/convert.py` | **build** 커맨드: 인자 → 설정객체 → `ArtifactBuilder` |
| `cli/serve.py` | **serve** 커맨드: 인자 → 서버 기동 |
| `artifact/builder.py` | **build 오케스트레이터** (`ArtifactBuilder`) |
| `artifact/validator.py` | 입력 규칙 검증 (빠른 실패) |
| `artifact/resolver.py` | 사용자가 안 준 값 채움(preset·device mesh·bucket) |
| `artifact/presets.py` | 모델별 추천 버킷 레시피북(`PRESET_REFS`) |
| `metadata/metadata.py` | `ModelMetadata` — 모델 정체성(타입·양자화·HF config) |
| `metadata/hf_utils.py` | `validate_model_support` — **빌드측 게이트** |
| `parallelize/trace.py` | 가중치 → param 캐시, torch→ATen 트레이싱 |
| `parallelize/new_pipeline_builder.py` | 버킷별 파이프라인 빌드(트레이싱+분할+컴파일 지휘) |
| `parallelize/block_slicer.py` | module_marker 삽입 + 커널 단위 그래프 컬러링 |
| `parallelize/graph_partitioner.py` | 컬러 → 파티션(스테이지) 매핑 |
| `parallelize/pipeline/builder/converter.py` | supertask → 컴파일러 호출(EDF) |
| `parallelize/compiler_config.py` | 스테이지별 컴파일러 설정 생성 |
| `api.py` | **serve 진입**(`LLM`/`_init_from_artifact`) |
| `llm_engine.py` | `stream_generate` 등 네이티브 엔진 래퍼 |
| `server/app.py`,`server/serving_chat.py` | OpenAI 호환 HTTP 라우팅·채팅 처리 |
| `sampling_params.py` | 샘플링 파라미터(온도·top_p·max_tokens) |
| `optimum/types.py` | `QuantizationConfig`(W..A..KV.. 표기), `QDtype` |

**모델 아키텍처 (역시 Python, `SDK/furiosa/models/`):**

| 경로 | 역할 |
|---|---|
| `language/architecture/qwen3_moe.py` | **qwen3_moe**(=30B-A3B) 아키텍처 |
| `language/__init__.py` | 아키텍처 클래스들을 `furiosa.models.*` 로 노출 |
| `common/export/serve/causal.py` | `make_example_inputs`(트레이싱용 더미 입력) |
| `core/quantization/` | FP8/INT8 양자화 연산 |
| `core/layers/moe/` | MoE 레이어(라우팅·expert) |

**네이티브 (우리가 못 읽음, `SDK/furiosa/*.so`) — 4총사 중 3개가 빌드/서빙 담당:**

| .so (정확한 경로) | pip 패키지 | 안에 든 Rust 크레이트 | 언제 쓰나 |
|---|---|---|---|
| `furiosa/native_llm_common.cpython-312-x86_64-linux-gnu.so` | furiosa-native-llm-common | `furiosa-llm-common`(컴파일러 API·아티팩트 타입) + `npu-compiler*` | **build**(컴파일) + serve(메타데이터 로드) |
| `furiosa/native_torch.cpython-312-x86_64-linux-gnu.so` | furiosa-torch | `furiosa-torch`(torch→EDF lowering) + `npu-executor` + `furiosa-hal2` | **build**(torch 그래프 → EDF IR) |
| `furiosa/native_runtime.cpython-312-x86_64-linux-gnu.so` | furiosa-native-runtime | `furiosa-generator`(serve 엔진·스케줄러·KV·샘플링) + `npu-compiler*` | **serve**(실행) |

> 4총사 중 `native_torch.so` 는 PyTorch C++(`libc10.so`)에 링크돼 있어 **빌드 환경에서만**
> 임포트됩니다(serve 전용 환경에선 `import furiosa.native_torch` → `ImportError: libc10.so`).
> 자세한 .so 역할은 [Part 4](#part-4)에서 다룹니다.

---

# Part 1. `furiosa-llm build` — 알고리즘 전체

예시 명령 (qwen3-coder-30b FP8, 1장 tp8, 64K 컨텍스트):

```bash
furiosa-llm build Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 ./out \
  -tp 8 --max-model-len 65536
```

이 한 줄이 끝나면 `./out/` 안에 `artifact.json`, `binary_bundle.zip`,
`params-...-W8fA16KV16-....safetensors`, `config.json`, `tokenizer.json`,
`chat_template.jinja` 가 생깁니다. 그 사이 벌어지는 일을 7단계로 봅니다.

```
furiosa-llm build (한 줄)
 │
 ├─[1.1] CLI 진입 ─ cli/main.py → cli/convert.py(인자→설정객체→ArtifactBuilder)
 ├─[1.2] Validate ─ builder.py:__init__ → validator.py (규칙 빠른 검증)
 ├─[1.3] Resolve  ─ builder.py:__init__ → resolver.py + presets.py (빈 값 채움)
 ├─[1.4] 가중치+양자화 ─ build()→_build_model_artifact()→get_param_file_with_cache()
 ├─[1.5] Pipeline build(트레이싱) ─ next_gen.build_pipeline → Ray actor → trace.py
 ├─[1.6] Compile(EDF)  ─ converter.py → native_llm_common.so(npu-compiler 12 pass)
 └─[1.7] 저장 ─ builder.py:__save_artifacts (zip + json + params + tokenizer)
```

## 1.1 CLI 진입 — 명령이 코드가 되기까지

**시작점:** `furiosa-llm` 실행파일은 `/home/jun/furiosa/bin/furiosa-llm` — 단 4줄짜리
파이썬 셸로, `from furiosa_llm.cli.main import main; sys.exit(main())` 만 합니다.

**서브커맨드 등록:** `SDK/furiosa_llm/cli/main.py:13-14`
```python
convert_parser = subparsers.add_parser("build", help="build model for RNGD")
add_convert_args(convert_parser)
```
- 비유: `main.py` 는 **접수창구**. "build" 라고 말하면 `convert.py` 담당자에게 넘깁니다.
- `main.py:27` `args.dispatch_function(args)` 가 실제 핸들러를 부르는데, 그 핸들러는
  `cli/convert.py` 의 `convert()` 함수입니다(이름이 build 가 아니라 **convert** 인 점 주의).

**인자 정의:** `SDK/furiosa_llm/cli/convert.py:15-121` (`add_convert_args`)
- `model`(위치), `output_path`(위치), `-tp`(기본 8), `-pp`(기본 1), `-pb`/`-db`(반복 가능,
  기본 []), `--max-model-len`(기본 None), `--additional-model-config`(반복), `--trust-remote-code`,
  `--bundle-binaries`(기본 True), `--cache-dir`, `--num-pipeline-builder-workers`,
  `--num-compile-workers`.
- 원리: `-tp 8` 기본값은 **RNGD 1장 = 8 PE** 라서. `-pp 1` 도 1장 기준.

**인자 → 설정객체:** `SDK/furiosa_llm/cli/convert.py:124-205` (`convert`)
- `-pb "1,128"` 같은 문자열을 `(1,128)` 정수 튜플로 파싱(:127, :135).
- `--additional-model-config key=value` 를 `key.partition('=')` 로 분해해
  `hf_overrides`/`seed`/컴파일 토글로 분배(:151-166). 모르는 키면 즉시 `ValueError`.
- **6개 설정객체** 생성(:169-195):
  - `ModelConfig`(trust_remote_code, hf_overrides, **seed_for_random_weight**, max_model_len) — :169
  - `ParallelConfig`(tp, pp) — :176
  - `BucketConfig`(prefill_buckets, decode_buckets) — :181
  - `CompilerConfig`(decomposition·blockwise·supertask·constant 토글) — :186
  - `ArtifactConfig`(bundle_binaries) — :193
- 그 다음 **오케스트레이터 생성·기동**(:197-219):
  ```python
  builder = ArtifactBuilder(args.model, args.name, model_config=..., parallel_config=...,
                            bucket_config=..., compiler_config=..., artifact_config=...)
  builder.build(args.output_path, num_pipeline_builder_workers=..., num_compile_workers=...,
                cache_dir=args.cache_dir)
  ```
- 비유: convert() 는 **주문서를 정식 양식(설정객체)으로 옮겨 적고**, 주방장(`ArtifactBuilder`)에게
  넘기는 웨이터입니다.

## 1.2 Validate — 비싼 일 하기 전 빠른 검문

`ArtifactBuilder.__init__` (`SDK/furiosa_llm/artifact/builder.py:116-170`)는 두 블록으로
나뉩니다. 첫 블록이 **Validate** (`builder.py:153-158`):

| 호출 (validator.py) | 무엇을 검사 |
|---|---|
| `validate_artifact_config` (`artifact/validator.py:202`) | 복사 대상 파일이 실제 존재하는지 |
| `validate_parallel_config` (`validator.py:234`) | `tp ∈ {4,8,32}`, `ceil(tp/8)*pp ≤ 8` |
| `validate_bucket_config` (`validator.py:73`) | 사용자가 버킷을 줬을 때만 모양 검증 |
| `validate_hf_config` (`validator.py:25`) | HF config 에 `max_position_embeddings`·`num_hidden_layers`·`hidden_size`·`intermediate_size` 존재 여부 |

- 원리: `tp=7` 같은 잘못된 값을 줘도 **몇 시간짜리 빌드를 돌린 뒤가 아니라 즉시** 거부.
- 비유: 공연장 입구의 **검표** — 표가 잘못되면 안에 들이지 않고 바로 돌려보냄.

## 1.3 Resolve — 사용자가 안 준 값을 채움

`__init__` 두 번째 블록 (`builder.py:160-170`):
```python
self._model_metadata = resolve_model_metadata(...)         # 정체성 확정
self._max_model_len  = resolve_max_model_len(hf_config, ...) # 컨텍스트 천장
self._device_mesh    = resolve_device_mesh(parallel_config)  # PE 배치
self._buckets        = ResolvedBuckets.resolve(...)          # 버킷 확정
```

**(a) `resolve_model_metadata`(정의 `artifact/resolver.py:246`) → `ModelMetadata` 생성(`resolver.py:311`)**
- 이때 `ModelMetadata.__init__`(`metadata/metadata.py:196`)이 **빌드측 게이트**를 호출:
  `validate_model_support`(`metadata/hf_utils.py:192`).
  - (`validate_model_support` 정의는 `hf_utils.py:197`, `ModelMetadata.__init__` 안
    `metadata/metadata.py:196` 에서 호출됨.)
  - 안에서 `get_optimized_cls`(`hf_utils.py:212`) → `get_models_lang_class`(`optimum/modeling.py:173`):
    `getattr(furiosa.models, model_cls.__name__)` — HF 클래스명(`Qwen3MoeForCausalLM`)과
    **똑같은 이름**의 클래스를 `furiosa.models` 에서 찾음. 이게 아키텍처 구현 연결 고리.
  - 이어서 `find_compiler_config(model_type, task, ~params)`(`hf_utils.py:217`)가
    네이티브 테이블(native_llm_common.so)에 `qwen3_moe` 가 있는지 확인. 없으면 `ValueError`.
    - (qwen3_next 처럼 없는 타입은 우리가 추가한 `_EXPERIMENTAL_MODEL_TYPES` 우회로 통과 —
      자세히는 feasibility 문서. qwen3_moe 는 정식 등록돼 있어 그냥 통과.)
- 비유: **모델의 신원조회** — "이 모델은 qwen3_moe 형이고, FP8 양자화고, 이 아키텍처
  클래스로 만든다" 를 확정.

**(b) `resolve_max_model_len` (`resolver.py:125`)**: `--max-model-len` 미지정시 HF의
`max_position_embeddings`(262144) 사용, 줬으면 그 값(단 천장 초과 불가). 30B 예시는 65536.

**(c) `ResolvedBuckets.resolve` (`resolver.py:34`)**: **버킷(=NPU가 컴파일할 고정 shape 목록)**
확정. 사용자가 `-pb/-db` 를 안 줬으면 `presets.find_preset`(`presets.py:395`)로 모델별
추천 버킷을 가져옴.
- `find_preset` 매칭 규칙(`presets.py:407-422`): `model_type` 정확 일치로 후보를 거른 뒤
  **per-layer 파라미터 수의 로그-거리 최근접**(`min(candidates, key=...abs(log(...) - log_input))`).
  30B-A3B 는 `(qwen3_moe, h=2048, i=6144)` → `QWEN_3_CODER_30B_A3B_PRESET`.
- **실측 로그**(`build_trace_full.log:5-7`, 미니 모델이지만 같은 경로):
  ```
  Found bucket preset for model_type=qwen3_moe, hidden_size=512, intermediate_size=1536
  Filtered bucket preset by max_model_len=2048
  The computed bucket limits are 1024
  ```
- 버킷 한 개 = `(batch, attention_size, kv_cache_size)`. prefill/decode/append/tokenwise
  4종이 있고, 미니의 경우 3개로 압축됨(`build_trace_full.log:11-13`):
  ```
  Attention buckets: [(1,128,kv0), (1,1024,kv1023), (1,256,kv128)]
  ```
- 비유: 버킷은 **옷 치수표**. "이 모델은 128/256/1024 토큰 길이용으로 재단해 둬라" 는 주문.
  NPU 는 동적 shape 가 약해서 **미리 정한 치수마다 따로 컴파일**합니다.

## 1.4 가중치 적재 + 양자화 — 무엇이 FP8 가 되나

`builder.build()` (`builder.py:315`)가 본격 시작. 핵심은
`_build_model_artifact`(`builder.py:172`) 안의 **param 파일 생성**:

```python
# builder.py:231-242
param_file_metadata = get_param_file_with_cache(model_creation_info,
                                                 param_file_cache_dir, max_shard_size=...)
```

**`get_param_file_with_cache` (`parallelize/trace.py:527`)**
- 모델+설정+양자화+가중치해시로 **캐시 키** 계산(`trace.py:539 hash_model`). 캐시에 있으면
  재사용(컴파일된 적 있으면 가중치 변환 생략).
- 없으면 `save_model(model.instantiate_model(), cache_path, "safetensors", ...)`(`trace.py:587`):
  - `instantiate_model()` 이 **torch 모델을 실제로 만들고 가중치를 적재하며 양자화를 접음**.
    아키텍처 클래스(`Qwen3MoeForCausalLM`)가 `LinearLayer`/`MoELayer` 에 `quant_config` 를
    넘겨 FP8 연산으로 구성(`qwen3_moe.py` 의 `Linear(..., quant_config=quant_config)`).
  - per-expert 로 흩어진 MoE 가중치는 `convert_from_mixtral_format`
    (`core/layers/moe/load_utils.py`)로 **fused 레이아웃**으로 합침
    (`qwen3_moe.py:473 _Qwen3MoeBase.transform_weights`).
- **결과 파일 이름이 곧 명세서**(`trace.py:577-579`):
  ```
  params-{모델식별자}-{아키텍처모듈}-{N}L-{양자화}-shard_size={크기}-{해시}.safetensors
  ```
  실물 30B: `params-Qwen3-Coder-30B-A3B-Instruct-FP8-qwen3_moe-48L-W8fA16KV16-shard_size=5000000000-d320...safetensors`
  - `48L` = 48 레이어, `W8fA16KV16` = **weight FP8 / activation BF16 / kv_cache BF16**.

**양자화 표기 W8fA16KV16 의 출처 (`optimum/types.py:178-183`)**
```python
def __str__(self) -> str:
    return "W{}A{}{}".format(self.weight.suffix(), self.activation.suffix(),
                             f"KV{self.kv_cache.suffix()}" if self.kv_cache else "")
```
- `QDtype.suffix()`(`optimum/types.py:93-105`): `"8f"`=FP8, `"16"`=BF16, `"8"`=INT8, `"4"`=INT4.
- 따라서 **W8fA16KV16 = 가중치만 FP8(e4m3), 활성·KV 는 BF16.** (mini 는 W16A16KV16 = 전부 BF16.)
- 실물 30B 의 `config.json` `quantization_config`(실측): `activation_scheme: dynamic`,
  `fmt: e4m3`, `modules_to_not_convert` 에 `lm_head`·각 레이어 `input_layernorm`·`mlp.gate`·
  `post_attention_layernorm` — 즉 **라우터 게이트·노름·임베딩은 FP8 미적용**(정밀도 보존).
- 비유: FP8 는 **그림을 색연필 256색→16색으로 줄여 인쇄**하는 것. 대부분 영역은 16색으로 충분히
  싸게 인쇄하되(weight FP8), 글자(라우터·노름)는 풀컬러로 남겨 또렷하게(BF16).
- **실측 캐시 히트 로그**(`build_trace_full.log:9`): `[CACHE] Accessing found parameter file
  from cache for model ... params-129a11f8830c-qwen3_moe-2L-W16A16KV16-...safetensors`.

## 1.5 Pipeline build (트레이싱) — 악보를 계산그래프로

`_build_model_artifact` 가 `next_gen.build_pipeline(...)`(`builder.py:268`)를 호출. 인자에
아키텍처 클래스 풀네임, 모델생성정보, device_mesh, 출력형식 `"edf"`, mppp(병렬화 정책),
버킷 설정 생성기, param 파일, paged_attention block_size 등이 들어감(`builder.py:268-295`).

**워커 구조** (`build_trace_full.log:14-17`):
```
Building local pipelines for 3 buckets: [...]
Number of pipeline builder workers: 1, ... compile workers: 1
Started a local Ray instance. dashboard http://127.0.0.1:8265
```
- **Ray** 로컬 인스턴스가 뜨고, `LocalPipelineGenerationActor`(`new_pipeline_builder.py`)가
  버킷마다 트레이싱을 수행. 워커 1개가 기본(메모리 안전). 비유: **Ray 는 작업반장**, 액터는 인부.

**버킷마다 트레이싱** (`build_trace_full.log:20-22`):
```
Model Tracing Progress: 0/3 → 1/3 → 2/3 → 3/3
```
- 핵심 호출: `make_example_inputs`(`SDK/furiosa/models/common/export/serve/causal.py:267`)가
  버킷 모양에 맞는 **더미 입력**(input_ids, position_ids, **kv_caches**(레이어당 K,V 튜플),
  attention_metadata, attention_masks)을 만들고(호출 `causal.py:355`, 정의 `_make_example_inputs` `causal.py:393`),
  `torch._dynamo.export`(`common/export/serve/base.py:56`)가 모델 forward 를 따라가며
  **ATen 그래프(IR)** 를 뜸.
  - 실측: `Generating ATen graph from torch ir graph` / `ATen graph generation took 0.18s`
    (`build_trace_full.log:27-28`).
  - kv_caches 의 shape 는 `get_model_dims`(`common/export/serve/utils.py:470`)가 HF config 의
    `num_key_value_heads`·`head_dim`·`num_hidden_layers` 를 **스칼라로** 읽어 **전 레이어 동일**
    하게 만듦. (이게 qwen3_next 가 막힌 구조적 이유 — feasibility 문서 참고.)

**module_marker 삽입 + 그래프 컬러링**:
- 트레이싱 중 `add_marker_op_hooks`(`parallelize/block_slicer.py:1107`)가 forward 훅으로
  **모듈 경계에 marker 연산**(`furiosa::module_marker`)을 꽂음. 대상은
  `KernelwisePartitioner.get_module_mark_config()` 정규식
  `(.*self_attn\.attn)|(model\.embed_tokens)|(embed_tokens)`(`graph_partitioner.py:50-58`)에
  맞는 모듈 + 모든 디코더 레이어.
- `get_kernelwise_sliced_color_bitmap_with_marker`(`block_slicer.py:968`)가 레이어 i 마다
  **색 3개**(2i=attn앞 tokenwise, 2i+1=attention, 2i+2=attn뒤 tokenwise)를 칠함.
  attention 색은 `*.self_attn.attn` 경로 모듈에만 시딩(`block_slicer.py:1013-1016`).
- `PartitionComposer.partition_gm`(`graph_partitioner.py:90`)이 색 → 파티션 ID 로 변환.
- 실측: `Detected dead nodes: set()`(`build_trace_full.log:31`) — 정상 모델은 죽은 노드 없음.
- 비유: marker 는 **악보에 그은 마디줄**. 컬러링은 "이 마디는 현악(attention), 저 마디는
  관악(tokenwise)" 식으로 **악기별로 묶는 것**. NPU 컴파일러는 이 묶음(=supertask/stage)
  단위로 따로 번역합니다.

**트레이싱 산출물**: 버킷별 FX/ATen 그래프 + 메타데이터. 가중치는 그래프에 넣지 않고
별도 param 파일로 분리(`build_trace_full.log:30` `28 weights will be saved in separate
param files`). 비유: **악보(그래프)와 가사집(가중치)을 따로 인쇄**.

## 1.6 Compile — 계산그래프를 NPU 명령어(EDF)로

트레이싱이 끝나면 `converter.py`(`parallelize/pipeline/builder/converter.py`)가 그래프를
**supertask** 로 쪼개고 컴파일러를 호출. 실측 로그(`build_trace_full.log:63-184`):

```
Compilation Progress: 0/6 → 6/6
```

**스테이지(=LayerRange) 분해** — 미니(2레이어) 실측으로 본 6개 컴파일 유닛:

| stage | LayerRange (실측 로그) | 종류 |
|---|---|---|
| stage_0 | `Embedding() → TransformerBlock(0, QkvProjection)` | tokenwise (임베딩+L0 qkv앞) |
| stage_1 | `TransformerBlock(0, Attention) → None` × **3 버킷**(128/1024/256) | **attention** |
| stage_2 | `TransformerBlock(0, OutputProjection) → TransformerBlock(1, QkvProjection)` | tokenwise |
| stage_4 | `TransformerBlock(1, OutputProjection) → OutputHeadAndPostProcess()` | tokenwise(+lm_head) |

- 즉 **레이어 하나가 [tokenwise앞 | attention | tokenwise뒤] 3조각으로 잘려** 컴파일됨.
  attention 조각만 버킷(길이)별로 따로 컴파일되고(stage_1 이 3번), tokenwise 조각은 길이
  무관이라 1번. (stage_3 = L1 attention 은 stage_1 과 모양이 같아 **컴파일 재사용**되어 로그에
  안 보임 → 6 유닛 = stage_0 + stage_1×3 + stage_2 + stage_4.)
- 30B(48레이어)면 이 패턴이 48겹으로 늘어 supertask 수가 훨씬 많아짐(과거 32B 실측 134개).

**스테이지별 컴파일러 설정** (`build_trace_full.log:66-70`):
```
Creating compiler config
Model type: qwen3_moe, task: generate, ~0.0026B per-layer params
Layer range details: LayerRange(start=Embedding(), end=TransformerBlock(idx=0, sub_layer=QkvProjection))
Bucket configuration: batch_size=1 attention_size=128 kv_cache_size=0
Using activation dequantization: False
```
- `create_llm_compiler_config_with_layer_range`(`parallelize/compiler_config.py:122`)가
  `(model_type, task, per_layer_params, num_chip, num_pe, bucket, layer_range)` 로 네이티브
  컴파일러 설정(YAML)을 받음. 알 수 없는 타입이면 `create_default_compiler_config()` 폴백
  (`compiler_config.py:138-142`).

**실제 NPU 컴파일 (12 패스)** — `converter.py` 가
`from furiosa.native_common.compiler import CompiledGraph`(빌더에서 import, `builder.py:408`)로
**native_llm_common.so** 의 컴파일러를 호출. 12단계 lowering(이전 빌드 로그에서 관측):
```
[1/12] dfg → primitive   [2/12] primitive → kernelized   [3/12] kernelized → prelower
[4/12] prelower → postlower   ... [12/12] resourcelir → edf
```
- 이 패스들은 Rust `npu-compiler` 크레이트(git `5c885c7`, 아티팩트
  `furiosa_compiler_version=5c885c73ee` 와 일치)가 런타임에 생성하는 단계명.
- 산출물: **EDF(Executable DataFlow)** 바이너리 — NPU 가 직접 실행하는 포맷(메모리 레이아웃·
  dataflow 스케줄·연산 분배가 박힘). 비유: EDF = **NPU 전용 기계어 + 배선도**.
- 그래프마다 해시·dump_tag 부여(`build_trace_full.log:86-87` `hash for the graph: 59cd...`),
  `~/.cache/furiosa/llm/graphmodules` 에 캐시. ⚠️ **이 캐시는 SDK 파이썬 코드 수정을 키에
  반영 안 함** — 아키텍처를 고치면 해당 `*.fx` 캐시를 지워야 재컴파일됨.

## 1.7 저장 — 아티팩트 패키징

`builder.build()` 끝에서 `__save_artifacts`(`builder.py:481`, 전처리 `:402`)가:
1. EDF 블롭들을 `binary_bundle.zip` 으로 묶음(`builder.py:437-444`, `--no-bundle-binaries`면
   개별 `.edf`). 비유: **여러 악기 파트보(EDF)를 한 폴더로 압축**.
2. `artifact.json` 작성 — `NextGenArtifact`(`builder.py:382`) 구조:
   - `metadata`(artifact_id, timestamp, furiosa_llm_version, furiosa_compiler_version)
   - `model.model_metadata`(model_type, task, **hf_configs**, llm_config=양자화·attention)
   - `model.parallel_config`(tp, pp)
   - `model.pipelines`(텐서·스테이지·블롭 참조) / `model.pipeline_metadata_list`(버킷 목록)
3. `params-*.safetensors`(가중치), `config.json`, `tokenizer.json`,
   `chat_template.jinja`, `generation_config.json` 복사.
4. 마지막 로그: `Artifact Build Completed`(`build_trace_full.log:185`).

실물 30B 아티팩트 디렉터리(실측):
```
artifact.json (18MB) · binary_bundle.zip (89MB) · config.json · generation_config.json
chat_template.jinja · tokenizer.json · tokenizer_config.json
params-...-qwen3_moe-48L-W8fA16KV16-...safetensors/ (가중치)
```

## 1.8 빌드 전체 타임라인 (미니 실측, 같은 코드 경로)

| 시각(로그) | 단계 |
|---|---|
| `resolver: Found bucket preset` | Resolve(preset 매칭) |
| `Calculated hashsum ... 322MB` | 가중치 해시 |
| `[CACHE] Accessing found parameter file` | param 캐시 적재 |
| `Attention buckets: [...]` | 버킷 확정 |
| `Started a local Ray instance` | 트레이싱 워커 기동 |
| `Model Tracing Progress 0→3/3` | 버킷별 트레이싱(~14s) |
| `Add tensors and supertasks` | supertask 구성 |
| `Compilation Progress 0→6/6` | EDF 컴파일(~7s) |
| `Artifact Build Completed` | 저장 완료 |

---

# Part 1B. qwen3-coder-30b vs qwen3-coder-next — 빌드 차이

| 항목 | **Qwen3-Coder-30B-A3B** (`qwen3_moe`) | **Qwen3-Coder-Next** (`qwen3_next`) |
|---|---|---|
| 레이어 구조 | **48겹 전부 동일** = (full attention + MoE) | **48겹 = 36 Gated DeltaNet(선형어텐션) + 12 full attention**, 모두 MoE |
| attention 단위 | 표준 multi-head + 페이지드 KV 캐시 | full 레이어만 KV; DeltaNet 레이어는 **순환 상태**(conv+recurrent), KV 없음 |
| KV 캐시 | 레이어당 (K,V) 1쌍, **전 레이어 균일** | 12 레이어만 KV; 36 레이어는 KV 없음 → **이종 구조** |
| 빌드측 게이트 | `find_compiler_config(qwen3_moe)` 정식 등록 → 통과 | 미등록 → `validate_model_support` 즉사(우회 필요) |
| 트레이싱 | 정상 | **순환 루프가 정적 언롤되어 그래프화 됨**(연산 자체는 가능) |
| **TP 분할** | 전 레이어가 `self_attn.attn` 보유 → 컬러 시딩 정상 | DeltaNet 레이어에 `self_attn.attn` 없음 → attn 색 미시딩 → 파티션 ID 희소 → `graph_partitioner.py` IndexError |

### 왜 같은 `furiosa-llm build` 인데 결과가 갈리나

빌드 파이프라인의 **모든 단계가 "dense 트랜스포머 + 페이지드 어텐션"을 암묵적 전제**로 합니다:

1. **`make_example_inputs`**(`causal.py:267`)는 `num_hidden_layers` 개의 균일한 (K,V) 캐시를
   만듭니다. 30B 는 48겹 다 KV 를 쓰므로 딱 맞지만, Next 는 36겹이 KV 를 안 써서
   "쓰지 않는 캐시 입력"이 dead node 가 됩니다.
2. **커널 컬러링**(`block_slicer.py:1013`)은 attention 파티션 색을 `*.self_attn.attn` 모듈에만
   칠합니다. 30B 는 48겹 모두 그 모듈이 있어 색이 연속(0,1,2,...)이지만, Next 의 DeltaNet
   레이어는 그 경로가 없어 **색이 비어(2i+1 누락)** 파티션 ID 가 `[0,2,4,6,7,8]` 처럼 끊깁니다.
3. **`PartitionComposer`**(`graph_partitioner.py:119-131`)는 "관측 색 개수" 길이 리스트를
   "원시 색 값"으로 인덱싱해서, 끊긴 ID 에서 `IndexError`.

→ 즉 **MoE·FP8·연산 종류가 문제가 아니라, "레이어마다 같은 모양의 KV 어텐션이 있다"는
구조 가정**이 30B 엔 맞고 Next 엔 안 맞아서 빌드 결과가 갈립니다. (우리는 Next 의 DeltaNet
순환부를 `self_attn.attn` 경로로 재배치 + KV 미선언으로 **분할까지는 통과**시켰고, 그 다음
`transform.py` 노드복제 단계에서 막혔습니다 — feasibility 문서 3절.)

비유: `furiosa-llm build` 는 **"모든 층이 똑같이 생긴 아파트"를 짓는 설비**입니다. 30B 는
실제로 모든 층이 똑같아(48겹 동일) 설비에 딱 맞지만, Next 는 4층마다 1층만 구조가 다른
(복층형) 건물이라 같은 설비로는 골조(파티션) 단계에서 어긋납니다.

---

# Part 2. `furiosa-llm serve` — 알고리즘 전체

예시 명령 (위장된 30B 아티팩트, 1장):
```bash
furiosa-llm serve \
  ./rngd-npu/artifacts/qwen3-coder-30b-a3b-inst-fp8-tp8-65k-tc \
  --devices npu:0 --host 0.0.0.0 --port 8000
```

```
furiosa-llm serve (한 줄)
 │
 ├─[2.1] CLI 진입 ─ cli/serve.py → server/app.py(run_server) → server/models.py
 ├─[2.2] LLM 생성 ─ api.py: LLM.__init__ → _init_from_artifact
 │        ├ 아티팩트 메타 로드(native_llm_common.so: NextGenArtifact.load_without_blob)
 │        ├ 디바이스/dp/pp resolve, 버킷 길이 계산
 │        └ NativeLLMEngine 생성(native_runtime.so) ← **여기서 게이트·가중치·KV·스케줄러**
 ├─[2.3] FastAPI/uvicorn 기동 ─ /v1/models, /v1/chat/completions 라우트
 └─[2.4] 요청 처리 ─ chat template → SamplingParams → engine.stream_generate → 토큰 스트림
```

> (이 문서 Part 번호: **Part 1** build · **Part 1B** 두 모델 빌드 차이 · **Part 2** serve ·
> **Part 3** .so 3총사 · **Part 4** 부록. Part 3 은 .so 설명이라 serve 뒤에 옵니다.)

## 2.1 CLI 진입 — 서버 기동까지

- `cli/main.py:16-17` 가 `serve` 서브커맨드를 `add_serve_args`(`cli/serve.py`)로 등록.
- serve 핸들러 → `server/app.py:run_server`(`app.py:533`, 실측 트레이스: `serve.py:385 → app.py:533 → uvicorn.run @ app.py:549`).
- `app.py:init_app` → `server/models.py:load_llm_from_args`(def `models.py:12`, `return LLM(...)` 는 `models.py:42`)가 `LLM` 을 생성.
  - ⚠️ `-tp` 는 serve 시 **무시**(아티팩트에 tp=8 이 이미 박힘) — `models.py:37-38` 경고.
    serve 가 존중하는 건 `-dp`/`-pp`/스케줄러 옵션뿐.

## 2.2 LLM 생성 — 아티팩트를 엔진에 싣기

`api.py:LLM.__init__`(def 는 `api.py:115`; 그 안에서 `_init_from_artifact` 호출이 `api.py:216`) → `_init_from_artifact`(`api.py:321`). 핵심 순서:

1. **아티팩트 메타 로드**(`api.py:343,349`):
   ```python
   from furiosa.native_llm_common import NextGenArtifact
   artifact = NextGenArtifact.load_without_blob(artifact_path)   # 가중치 빼고 메타만
   model_metadata = artifact.model.model_metadata
   ```
   - `load_without_blob` 은 **native_llm_common.so** 가 제공. EDF 블롭(무거움)은 빼고
     model_metadata·pipelines·parallel_config 만 읽음. 비유: **목차만 먼저 펼침**.
   - 메타 로드 직후 `api.py:354 artifact.override_with(...)` 가 실행되어 serve 시 옵션
     (`num_blocks_per_pp_stage`·`cache_dir` 등)을 아티팩트 메타에 덮어씀 — 엔진 생성(아래 4) 전 단계.
2. **디바이스 resolve**(`api.py:352`) `resolve_devices` — `"npu:0"` → 내부 Device 객체.
   `-dp`/`-pp` 검증(PE 수 = dp×pp×tp ≤ 보유 PE)은 실제로 **Rust 쪽**에서 수행.
3. **버킷 길이 계산**(`api.py:360`) `compute_bucket_lengths(pipeline_metadata_list)` →
   `max_prefill/decode_bucket_len` → `max_seq_len`(=65536) 산출. 이게 `/v1/models` 의
   `max_context_len` 으로 노출.
4. **네이티브 엔진 생성**(`api.py:381-400`) — **여기가 serve 의 심장**:
   ```python
   from furiosa.native_runtime.llm import NativeLLMEngine
   self.engine = NativeLLMEngine(
       artifact_path, None, devices, data_parallel_size, pipeline_parallel_size,
       max_io_memory_mb, self._serialize_obj(scheduler_config or SchedulerConfig()),
       structured_outputs_backend, tokenizer.backend_tokenizer.to_str(), None,
       enable_jit_compilation, jit_threshold, jit_max_workers, jit_unit_size,
       cache_dir, served_model_name)
   ```
   - 인자에 **`num_blocks`·`block_size` 가 없음** — KV 캐시 크기는 Python 이 안 정하고
     **런타임이 아티팩트의 KV 텐서에서 도출**(아래 2.2.x).
   - `scheduler_config`·`tokenizer` 는 **JSON 문자열**로 직렬화해 넘김 → Rust 가 자립적으로
     스케줄·디토큰.
   - 이 생성자 안에서 **모델타입 게이트(2겹 중 2번째)**가 발동: `furiosa-generator/src/next_gen/
     hf_compat_next_gen.rs:367`(패닉 Location, 실측 출력으로 확인). 위장 아티팩트는 `qwen3`
     이라 통과. 비유: **공연장 입구 신분증 검사 — 이름표만 봄.**
   - ⚠️ 사실 **1번째 게이트는 더 앞**입니다: `api.py:349 NextGenArtifact.load_without_blob`
     (native_llm_common.so)이 `model_type` 을 **serde enum 으로 역직렬화**하면서 미등록 값을
     먼저 거부합니다(`furiosa-llm-common/src/artifact/types/next_gen.rs:238`,
     `unknown variant 'qwen3_next', expected one of llama/exaone4/qwen2/qwen3/qwen3_moe/gpt_oss`).
     이 게이트의 **바이너리 레벨 해부**(radare2 디스어셈블: 첫 바이트 점프 테이블 매처,
     변형별 단말 블록 주소, 통과 방법)는 [README_qwen3_coder_next.md](README_qwen3_coder_next.md)
     에 별도 정리.

### 2.2.x 네이티브 엔진이 부팅하며 하는 일 (실측 로그)

`serve_30b_tc_newpath.log`(실물 30B 부팅) 실측:

| 시각 | 로그 (요약) | 의미 |
|---|---|---|
| 14:32:54 | `furiosa::llm::engine: Parallelism Config: tp=8, pp=1, dp=1` | 병렬화 확정(아티팩트 tp8, 1장이라 dp1·pp1) |
| 14:32:55 | `pipeline::resolve: PP device#0 allocation plan: Binary=83.2 MiB, Model weights=29.2 GiB, Reserved IO memory=4.0 GiB` | HBM 예산 계획 |
| 14:32:58 | `backing_file: Total size of parameters loaded: 29.2 GiB in 2.98 s` | 가중치 mmap 적재 |
| 14:32:59 | `pipeline::resolve: PP device#0 KV cache=14.3 GiB` | KV 캐시용 HBM |
| 14:33:01 | `memory_manager: Configured KV cache blocks, global_num_blocks: 155796` | **KV 블록 수 = 아티팩트 KV 텐서에서 도출** |
| 14:33:01 | `generator: Eager scheduler has started with: SchedulerConfig { ... prefix_cache_config { enabled:true } ... }` | 스케줄러 시작 |
| — | `Uvicorn running on http://127.0.0.1:8765` | HTTP 수신 시작 |

- **KV 캐시 도출 원리(실측 확정):** 런타임은 아티팩트 파이프라인 텐서 중 origin 이름이
  정규식 `(past_key_values_\d+_\d+|kv_caches.*)` 에 맞는 텐서를 찾아 그 DRAM shape 로
  블록 수를 계산. `num_hidden_layers` 같은 hf_config 값으로 계산하지 **않음**(실험으로
  hf_configs 의 num_hidden_layers 를 1/48 로 바꿔도 블록 수·출력 동일함을 확인).
  비유: **좌석 수는 "건물 설계도(아티팩트)"가 정하지, "안내문(hf_config)"이 정하지 않음.**
- **EBUSY 주의:** NPU 점유는 `NativeLLMEngine` 생성자(`api.py:383`)에서 **동기적으로**
  일어남. 다른 serve 가 npu:0 을 잡고 있으면 `ValueError: NPU error: Npu npu0pe0-3: EBUSY`
  로 즉시 실패하고 uvicorn 까지 못 감(실측).

## 2.3 / 2.4 요청 처리 — HTTP 한 방의 여정

`curl /v1/chat/completions` 가 들어오면:

1. **라우팅**(`server/app.py:186-191`): `@router.post("/v1/chat/completions")` →
   `openai_serving_chat.create_chat_completion(request, raw_request)`. HTTP 층은 NPU 를
   직접 건드리지 않음(얇은 껍데기).
2. **채팅 템플릿 + 토크나이즈**(`server/serving_chat.py:191-205`): `preprocess_chat(...)`이
   HF Jinja 채팅 템플릿(`chat_template.jinja`)을 적용하고 토큰화. NPU 는 **토큰 ID 만** 봄.
3. **샘플링 파라미터 변환**(`serving_chat.py:211` → `server/protocol.py:429` →
   `sampling_params.py:107`): 요청의 temperature/top_p/max_tokens 를 모델 기본값
   (`generation_config.json`)과 병합해 `SamplingParams` 생성. 스트리밍이면 `output_kind=DELTA`.
4. **네이티브로 넘김**(`llm_engine.py:578-627`): `AsyncLLMEngine.generate` 가 프롬프트를
   다시 토크나이즈(지연 최적화, :600)한 뒤
   ```python
   native_output_generator = self.native_engine.stream_generate(batch_encoding, sampling_params, request_id)  # llm_engine.py:611
   ```
   - **`stream_generate` 가 Python ↔ 네이티브 경계.** 이 너머의 **연속 배칭·스케줄링·KV
     블록 할당·prefix 캐시·샘플링·EDF 실행**은 전부 native_runtime.so 안에서 일어남.
   - ⚠️ Python 의 `LLMEngine.step()`(`llm_engine.py:483`)은 vLLM 호환용 껍데기 —
     **연속 배칭을 돌리지 않음.** 진짜 배치 루프는 Rust `furiosa-generator` 가 소유.
5. **응답 변환**: `NativeOutputConverter` 가 네이티브 출력 스트림을 OpenAI delta 로 변환해
   `async for` 로 흘려보냄(`llm_engine.py:616,626`).

비유: Python 은 **주문 받고 통역하는 홀**, native_runtime.so 는 **실제 요리하는 주방**.
홀은 주문서(SamplingParams)와 손님 말(토큰)만 주방에 넘기고, 접시(토큰)를 받아 나릅니다.

### 스케줄러·KV·prefix 캐시 (native_runtime.so 내부, strings 로 확인)

- `SchedulerConfig` — Python dataclass 는 **12필드**(`metadata/config_types.py:84-96`:
  `scheduler_kind`, `npu_queue_limit`, `max_processing_samples`, `spare_blocks_ratio`,
  `is_offline`, `estimation_time_limit_ms`, `prefix_cache_config`,
  `experimental_scheduling_loop_type`, `experimental_aggressive_batching`,
  `max_concurrency`, `max_num_batched_tokens`, `data_parallel_routing_policy`). 부팅 로그엔
  **11필드**로 직렬화(`is_offline` 는 `__post_init__` 에서 `scheduler_kind` 로 접힘). 거의 1:1.
- **prefix 캐시**: 기본 ON(`is_prefix_cache_enabled: true`). 같은 프롬프트 접두를 공유하면
  KV 블록 재사용. ⚠️ DeltaNet 같은 **비-KV 순환 상태**가 있는 하이브리드 모델에선 접두
  캐시가 페이지드 KV 만 복원하므로 **조용한 오답** 위험(30B 는 순수 KV 라 안전).
- 샘플링: `furiosa-generator/src/frontend/v2/sampling/{greedy,random}.rs` (greedy=탐욕,
  random=온도/top_p 샘플).

---

# Part 3. .so 3총사 — 못 읽지만 핵심인 네이티브

> 4개 .so 중 `furiosa_smi_py` 계열(장치 모니터링)을 빼면, build/serve 의 본체는 아래 3개.
> 모두 Rust 로 짜여 컴파일된 PyO3 확장이라 소스는 못 보지만, `strings` 로 내부 크레이트
> 경로와 역할을 추출했습니다(git `5c885c7`, npu-tools).

### (1) `furiosa/native_llm_common.cpython-312-x86_64-linux-gnu.so` — **빌드측 컴파일러 + 아티팩트**
- pip: `furiosa-native-llm-common 2026.2.0`
- 내부 크레이트: `furiosa-llm-common/src/compiler/*`, `.../artifact/*`, `.../pipeline/next_gen.rs`,
  `.../hf_config.rs` + `npu-compiler*`, `npu-alu`, `furiosa-mapping`
- 파이썬 노출: `furiosa.native_common.compiler`(부트스트랩 shim `furiosa/native_common/compiler/__init__.py`
  가 이 .so 를 임포트해 PyO3 가 등록) → `find_compiler_config`(빌드 게이트), `CompiledGraph`
  (실제 12-pass 컴파일), `create_llm_compiler_config_with_layer_range`, `ArtifactMetadata`,
  `NextGenArtifact`, `compute_limits`(버킷 한도), `wire_tasks`.
- **언제:** build 에서 EDF 컴파일·아티팩트 타입 전부 + serve 에서 메타데이터 로드
  (`NextGenArtifact.load_without_blob`, `api.py:343`) + 선택적 JIT.
- 비유: **편곡·인쇄 공장의 마스터 기계** — 악보를 NPU 기계어(EDF)로 찍어냄.

### (2) `furiosa/native_torch.cpython-312-x86_64-linux-gnu.so` — **빌드측 torch→EDF lowering**
- pip: `furiosa-torch 2026.2.0`
- 내부 크레이트: `furiosa-torch/src/compiler/*`, `.../ir/{dfg.rs,edf.rs}`,
  `furiosa-libtorch-bindings/*`, `npu-executor`, `furiosa-hal2`(하드웨어 추상화)
- PyTorch C++(`libc10.so`/`libtorch`)에 링크 → **빌드 환경에서만** 임포트 가능
  (serve 전용 환경: `import furiosa.native_torch` → `ImportError: libc10.so`).
- 안에 지원 ATen op 목록(`SUPPORTED_ATEN_OPS`)이 박혀 있어, 트레이싱된 torch FX 그래프를
  NPU EDF IR 로 lowering 할 때 어떤 연산이 가능한지 판정.
- **언제:** build 의 트레이싱·lowering. **serve 에선 안 쓰임.**
- 비유: **악보를 기계가 읽을 수 있게 음표 하나하나 검수·변환하는 번역기.**

### (3) `furiosa/native_runtime.cpython-312-x86_64-linux-gnu.so` — **서빙 엔진**
- pip: `furiosa-native-runtime 2026.2.0`
- 내부 크레이트: `furiosa-generator/src/{frontend,generator,model_poc,next_gen/*}` +
  `npu-compiler*`(선택적 JIT 용) + `npu-executor-common`
- 파이썬 노출: `furiosa.native_runtime.llm.NativeLLMEngine`(메서드 `generate`,
  `stream_generate`, `encode`, `abort_request`, `is_alive`, `shutdown`).
- 담당: 모델타입 게이트(`hf_compat_next_gen.rs:367`), 가중치 적재(`backing_file`),
  KV 캐시 풀(`scheduler/host_kv_cache_pool.rs`), 연속 배칭·스케줄러
  (`next_gen/generator.rs`, `scheduler/*`), prefix 캐시, 샘플링(`frontend/v2/sampling/*`),
  EDF 실행(`next_gen/pipeline/edf_wrapper.rs`).
- **언제:** serve 전부. (build 엔 안 쓰임.)
- 비유: **공연장 + 오케스트라 + 지휘자** 한 몸 — 인쇄물(EDF)을 받아 실시간 연주(생성).

### 한눈 요약

| .so | build | serve | 핵심 역할 |
|---|:---:|:---:|---|
| native_llm_common | ✅(컴파일) | ◐(메타 로드) | EDF 컴파일러 + 빌드 게이트 + 아티팩트 |
| native_torch | ✅(lowering) | ✗ | torch→EDF IR, op 지원 판정 |
| native_runtime | ✗ | ✅(전부) | serve 엔진·스케줄러·KV·샘플링·게이트 |

---

# Part 4. 부록 — 캐시·경로·함정

### 캐시 위치
| 용도 | 경로 |
|---|---|
| 컴파일된 그래프(.fx) | `~/.cache/furiosa/llm/graphmodules/` |
| 변환된 가중치(param) | `~/.cache/furiosa/llm/param_files/` |
| 컴파일된 EDF | `~/.cache/furiosa/llm/compiled_graphs/` |
| HF 원본 모델 | `~/.cache/huggingface/hub/` |

### 함정 모음 (실측)
1. **그래프 캐시는 SDK 코드 수정을 모름** — 아키텍처 `.py` 를 고치면
   `~/.cache/furiosa/llm/graphmodules/*<ClassName>*` 를 지워야 재컴파일됨.
2. **serve `-tp` 는 무시됨** — tp 는 아티팩트에 박힘. 바꾸려면 다시 build.
3. **KV 캐시 크기는 hf_config 가 아니라 아티팩트 KV 텐서가 결정** — hf_config 의
   num_hidden_layers 를 바꿔도 KV 블록 수 불변.
4. **serve 게이트는 `model_type` 문자열만 검사** — 연산은 이미 EDF 에 있으므로 위장 가능
   (단 `layer_types` 에 `linear_attention` 값이 있으면 Rust 파서 패닉, KV 차원은 불변 필수).
5. **NPU 점유는 엔진 생성자에서 동기적** — 다른 serve 가 잡고 있으면 EBUSY 즉시 실패.
6. **prefix 캐시 기본 ON** — 순환 상태 있는 하이브리드 모델에선 조용한 오답 위험.

### 빌드 vs 서빙 한 줄 대조
> **build = native_llm_common.so + native_torch.so 가 악보(torch)를 EDF 로 인쇄.
> serve = native_runtime.so 가 그 EDF 를 NPU 로 연주.** Python 은 양쪽 모두 **지시·통역**만
> 하고, 무거운 일(컴파일·실행)은 .so 안의 Rust 가 합니다.
