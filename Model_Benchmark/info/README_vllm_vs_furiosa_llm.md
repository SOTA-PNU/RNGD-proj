# vLLM vs Furiosa LLMEngine — 객관적 비교

두 LLM 서빙 라이브러리를 **실제 설치된 소스 코드**를 읽고 비교한 문서입니다. 모든 주장에는 `파일:줄` 근거를 답니다. 한 번 읽은 뒤 적대적으로 다시 검증한 결과를 담았습니다.

- **furiosa-llm 2026.2.0** — `~/furiosa/lib/python3.12/site-packages/furiosa_llm`
- **vLLM 0.10.0+cu126** — `Model_Benchmark/bench-gpu/.venv/lib/python3.12/site-packages/vllm` (V1 엔진 기준. 0.10.0은 구형 V0(`engine/`)와 신형 V1(`v1/`)을 함께 담고 있고, V1이 기본값이라 V1을 기준으로 봅니다.)

---

## 한 줄 결론

> **Furiosa = "미리 컴파일해 얼린 아티팩트 + 네이티브 블랙박스 런타임"**
> **vLLM = "실행 시점에 직접 쪼개고·스케줄하고·샘플링하는 열려 있는 Python 컨트롤 플레인"**

같은 문제를 정반대 철학으로 풉니다. 가장 먼저 짚어야 할 비대칭은 — **Furiosa의 실제 동작 로직 대부분이 Python에서 보이지 않는다**는 점입니다.

---

## 1. 가장 큰 차이 — 엔진 코어가 어디에 있나

`furiosa_llm.LLMEngine`은 `furiosa.native_runtime.llm.NativeLLMEngine`을 감싼 **얇은 Python 래퍼**입니다 (`furiosa_llm/llm_engine.py:23`). 소스 주석이 직접 말합니다:

> "While vLLM provides fine-grained control over decoding via the `step` method ... The Furiosa native engine handles scheduling and batching internally." (`llm_engine.py:256-263`)

- 요청 추가(`add_request`) 시 **즉시 생성 시작** (`llm_engine.py:362-363`), 실제 디코딩은 `native_engine.stream_generate(...)`로 네이티브 위임 (`:453-455`).
- 공개 `step()`은 디코딩을 돌리는 게 아니라, 백그라운드가 채운 결과 큐를 **꺼내기만** 함 (`llm_engine.py:483-489`).

vLLM V1은 **읽히는 Python 컨트롤 플레인 + (기본) 별도 EngineCore 프로세스**. `EngineCore.step()`이 진짜 디코딩 루프이고 `scheduler.schedule()` → `execute_model()` → `update_from_output()`이 코드에 그대로 보입니다 (`vllm/v1/engine/core.py:266-271`). 오프라인 기본은 ZMQ 멀티프로세스 (`vllm/v1/engine/llm_engine.py:131`, `vllm/envs.py:98`).

---

## 2. 차원별 비교표

| 항목 | Furiosa LLMEngine (2026.2.0) | vLLM (0.10.0, V1) |
|---|---|---|
| **엔진 코어** | 네이티브 `NativeLLMEngine` 래퍼. `step()`은 큐 비우기만. 동시성은 네이티브 소유 (`llm_engine.py:256-263, 483-489`) | `EngineCore.step()`이 실제 디코딩 루프, 전부 Python. 기본 멀티프로세스(ZMQ) (`v1/engine/core.py:266-271`) |
| **스케줄링·연속 배칭** | 알고리즘은 네이티브(.so) 안. Python엔 노브만(`max_concurrency`, `max_num_batched_tokens`, `npu_queue_limit`, `spare_blocks_ratio`). 청크 프리필은 **컴파일 타임 버킷** 개념 | `Scheduler.schedule()` 전부 Python. 토큰 예산 기반, 명시적 선점, FCFS/PRIORITY 정책, 청크 프리필 런타임 결정 (`v1/core/sched/scheduler.py:166-204, 255-271`) |
| **KV 캐시·페이징** | 페이지드 어텐션 컴파일 타임 선택, **블록 크기=1토큰 고정** 후 아티팩트에 구움 (`artifact/builder.py:193-200`). 런타임 메모리 노브 `spare_blocks_ratio`뿐. `gpu_memory_utilization`·블록 프로파일링·스왑 없음 | 블록 관리 전부 Python(`BlockPool`/`KVCacheManager`). 블록 크기 런타임 선택(1/8/16/32/64/128), 메모리 프로파일링으로 블록 수 산정, `gpu_memory_utilization=0.9` (`config.py:1608-1626`, `v1/core/kv_cache_utils.py:761-781`) |
| **병렬화 TP/PP/DP** | **빌드 타임 고정.** 자체 MPPP가 FX 그래프를 다시 써 통신 연산(SEND/RECV/ALL_REDUCE) 삽입, PP는 블록 슬라이싱 (`parallelize/mppp/api.py:134-137`). **TP는 빌드 시 고정**, 런타임엔 `tensor_parallel_size` 안 넘어감 | **전부 런타임 선택.** Executor가 `world_size=tp*pp` 워커 프로세스 생성, 로드 시 셀프 샤딩 (`v1/executor/multiproc_executor.py:53-90`). `tensor_parallel_size`가 1급 런타임 노브 |
| **모델 준비** | **엄격한 AOT.** HF→optimum/FX→그래프 분할→Furiosa 컴파일러→**디스크 아티팩트(EDF+safetensors+토크나이저)** 저장. 서빙 시 미리 빌드된 아티팩트 **로드만**(기본 재컴파일 안 함), **shape 정적·버킷 고정** (`artifact/builder.py:248-249, 452-455`, `presets.py:70-74`) | **런타임 로드.** safetensors/bin/gguf/bitsandbytes 등으로 시작 시 가중치 로드(HF 허브 다운로드 포함), `torch.compile`+CUDA 그래프 시작 시 캡처. **shape 동적**, 영속 컴파일 바이너리 없음 (`model_loader/default_loader.py:130-137`) |
| **양자화** | HF `quantization_config`에서 **오프라인 확정해 아티팩트에 구움**. 서빙 중 변경 불가(재빌드 필요). 커널은 네이티브 컴파일러 안 | **런타임 선택.** 기본 None→시작 시 자동 감지, awq/gptq/fp8/marlin/compressed-tensors 등 **약 30종** 플래그 선택 (`config.py:302-306`, `quantization/__init__.py:9-39`) |
| **샘플링** | vLLM 파생 `SamplingParams`지만 **핵심 부분집합만**. **n>1 거부**, presence/frequency_penalty·seed·stop 문자열·bad_words·logit_bias·사용자 logits_processor **없음**. 수식은 네이티브 위임 (`sampling_params.py:168-190, 295-296`) | **OpenAI 풀세트+α**, 전부 읽히는 Python/Triton 적용. 투기적 디코딩(ngram/eagle/medusa), 빔서치, 사용자 logits_processor, 구조화 출력 4종 (`sampling_params.py:183-227`, `v1/sample/sampler.py:48-58, 202-217`) |
| **서빙(OpenAI 호환)** | OpenAI 호환 서버 제공 — **소스가 "vLLM에서 가져옴" 명시** (`server/protocol.py:1-2`, `server/chat_utils.py:2-3`). 단 `to_sampling_params()`가 presence/frequency_penalty·seed·logit_bias·stop **문자열 조용히 버림** | 성숙한 OpenAI 서버가 핵심. 받은 샘플링 필드 실제 적용. LoRA 추가/제거, sleep/wake, `reset_prefix_cache`, Prometheus 등 운영 API 풍부 (`entrypoints/openai/api_server.py:979-1014, 1112-1193`) |
| **하드웨어** | **Furiosa RNGD NPU.** TP는 칩당 1/2/4/8 **PE를 컴파일 타임 융합(fusion)**해 구현, "TP 차수는 1·2·4 또는 8의 배수" 제약 (`device.py:6, 43-49, 92`). 실행물은 EDF 그래프 | **GPU(CUDA 1순위).** 일반 GPU 랭크 그리드, CUDA 그래프 + NCCL, FlashAttention/FlashInfer 커널 (`platforms/cuda.py:55-58, 285-292`) |

---

## 3. 각 진영의 진짜 강점 (소스로 확인)

### Furiosa만의 강점
- **한 번 빌드하면 시작이 빠르고 결정적** — 컴파일/양자화/분할 비용을 오프라인 한 번에 치르고 서빙은 로드만.
- **그래프 전체 사전 병렬화** — MPPP가 통신 토폴로지까지 정적 확정해 아티팩트에 담음.
- **하드웨어 밀착 TP** — NPU PE 융합 기반, `npu_queue_limit` 같은 vLLM에 없는 NPU 전용 노브.
- **정적 shape 버킷 + 모델별 프리셋 자동 매칭**으로 실행 형상 사전 검증.
- 스키마 수준에서 vLLM에 없는 일부: `reasoning_grammar`(추론 문법), `prompt_logprobs == -1`(전체 어휘 로그확률).

### vLLM만의 강점
- **전부 들여다보고 고칠 수 있음** — 엔진 루프·스케줄러·KV·병렬화·샘플링 모두 읽히는 Python.
- **재빌드 없는 런타임 유연성** — 임의 HF 체크포인트 즉시 실행, TP/PP/DP 런치마다 지정, 양자화 30종 플래그 선택.
- **동적 shape** — 버킷 사전 정의 불필요, 임의 입력 길이 적응.
- **완전한 샘플링 + 투기적 디코딩** — Furiosa Python API엔 전부 없음.
- **튜닝 가능한 메모리 관리**와 **풍부한 운영 API**.

### 공통점
- 둘 다 vLLM식 공개 API(`SamplingParams`, `LLMEngine.add_request/step`, `LLM.generate`, async 엔진, OpenAI 서버) — 단 `step()` 의미 다름.
- 둘 다 페이지드 어텐션·연속 배칭 개념, **prefix caching 기본 ON**.
- 둘 다 스트리밍, 빔서치(length penalty), logprobs/prompt_logprobs, 구조화/가이드 디코딩, PP 단계별 불균등 분할, DP 지원.

---

## 4. ⚠️ 공정하게 봐야 할 핵심 주의

이 비교에서 **"Furiosa엔 X가 없다"는 거의 모두 "Furiosa Python API 층에 없다"는 뜻**입니다. 스케줄링·연속 배칭, KV 블록 테이블·페이징·축출, 디바이스 간 통신 실행, 샘플링 수식, 양자화 커널이 모두 컴파일된 네이티브 런타임(`NativeLLMEngine`)·네이티브 컴파일러(`CompiledGraph`) 안에 있습니다. Python 소스로는 **설정 스키마·AOT 빌드 파이프라인·호출 지점**까지만 검증되고 **런타임 동작·정확성·성능은 확인 불가**입니다. 네이티브가 동등하거나 더 나은 걸 해도 여기선 안 보일 뿐입니다.

→ **어느 쪽이 절대적으로 낫다고 말할 수 없습니다.** Furiosa NPU에 고정·사전 최적화 배포가 목적이면 Furiosa, GPU에서 유연하고 투명하며 호환성 넓은 스택이 목적이면 vLLM입니다.

---

## 5. 검증 중 바로잡은 주장 (투명성)

- vLLM PRIORITY 큐는 2-튜플이 아니라 **3-튜플** `(priority, arrival_time, request)`로 정렬.
- Furiosa는 "서빙 중 절대 재컴파일 안 함"이 아니라 **기본 OFF의 opt-in JIT 경로(`enable_jit_compilation`) 존재** — 재양자화는 서빙 중 미호출.
- EDF는 네이티브 `CompiledGraph`에서 직렬화됨은 맞으나 바인딩이 PyO3/Rust인지는 미확인.
- `spare_blocks_ratio`는 "유일한" 게 아니라 **가장 직접적인** 블록 예약 노브.

---

---

## 6. 실행·운영 관점 차이 (명령어 기준)

> 슬라이드: `Model_Benchmark/ppt/RNGD_vLLM_vs_FCLM.pdf` / `.pptx` (빌드: `ppt/build_vllm_vs_fclm.js`, 11장)

엔진 내부 구조(위 1~5절)와 별개로, **실제로 쓰는 명령어**에서도 차이가 큽니다. 핵심은 furiosa가 **2단계(build→serve)**, vLLM이 **1단계(serve가 곧 로드)** 라는 점입니다.

| 단계 | furiosa-llm | vLLM |
|---|---|---|
| 설치 | `pip install furiosa-llm --extra-index-url <furiosa-pypi>` + NPU 드라이버/펌웨어/컴파일러 | `pip install vllm` (+ CUDA 런타임) |
| 모델 준비 | `furiosa-llm build <model> <out> -tp 8 --prefill-buckets ...` → 아티팩트(EDF) 생성 (`cli/convert.py`) | **없음** (HF id를 serve에 바로 전달) |
| 서버 실행 | `furiosa-llm serve ./artifacts/... --devices npu:0 --port 8000 --max-concurrency 32 --reasoning-parser qwen3` (`bench-blog/run_rngd.sh:41`) | `CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-32B --port 8000 --max-num-seqs 32 --tensor-parallel-size 2` (`bench-gpu/runners/server.py:59`) |
| 오프라인 API | `LLM("./artifacts/...", devices="npu:0")` — 인자가 **아티팩트 경로** (`api.py:115`) | `LLM(model="Qwen/Qwen3-32B", tensor_parallel_size=2)` — 인자가 **HF id** |
| 클라이언트 | OpenAI 클라이언트 동일. 단 presence/frequency_penalty·seed·logit_bias·stop(문자열) **무시** | OpenAI 클라이언트 동일. 받은 필드 실제 적용 |

### 튜닝 노브 대응 (벤치 비교용 정렬, `configs/models.yaml`)
- 동시 시퀀스: furiosa `--max-concurrency`/`--max-batch-size` ↔ vLLM `--max-num-seqs`
- 배치 토큰: `--max-num-batched-tokens` (**키 이름 동일**)
- 프리픽스 캐싱: `--enable-prefix-caching` (둘 다 기본 ON)
- 메모리: vLLM `--gpu-memory-utilization` ↔ furiosa는 직접 등가 없음(`--spare-blocks-ratio`가 근접)
- furiosa serve는 vLLM 플래그를 다수 미러링 + NPU 전용(`--devices npu:N`, `--prefill/decode-buckets`, `--npu-queue-limit`, `--scheduler-kind`, `--enable-jit-compilation`) 추가 (`cli/serve.py`)

### 실전 운영 함정 5가지
1. **tp는 serve에서 못 바꿈** — prebuilt 아티팩트는 tp가 바이너리에 박혀 `--tensor-parallel-size`가 무시됨(WARNING). 바꾸려면 build 재실행 (`README_runcode.md:499`, `server/models.py:38`, `api.py:383`). vLLM은 기동 플래그로 즉시 변경. (pp·dp는 furiosa serve에서도 변경 가능)
2. **프롬프트 길이 = 버킷 상한** — 정적 shape이라 최대 prefill 버킷 초과 프롬프트 거부. vLLM은 동적 shape.
3. **`--gpu-memory-utilization` 등가 없음** — NPU는 메모리 자동 프로파일링 노브가 없음.
4. **첫 준비는 무겁고 기동은 가볍다** — furiosa: build 1회(수십 분)→serve 로드만. vLLM: build 없음→매 기동 컴파일/캡처.
5. **아티팩트 이동성** — furiosa 아티팩트는 특정 모델·tp·SDK 버전에 묶임. vLLM은 HF 체크포인트만 있으면 어느 GPU에서나 동일 기동.

---

*근거: furiosa-llm 2026.2.0, vLLM 0.10.0 설치본 소스 직접 분석(8개 차원, 17개 에이전트 분석+적대적 검증) + repo 실제 스크립트(run_rngd.sh, runners/server.py, models.yaml, README_runcode.md). 작성일 2026-06-09.*
