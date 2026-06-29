# Qwen2.5-72B serve 실패 진단 + 빌드 트레이싱·컴파일 상세

두 가지를 다룹니다. (1) bf16 72B를 serve 못 하는 진짜 이유와 max_model_len·다른 인자로
풀 수 있는지, (2) 빌드 중 진행되는 "트레이싱"과 "그래프 컴파일"이 각각 무엇인지.

---

## 1. Qwen2.5-72B-Instruct serve가 안 되는 이유

### 1-1. 빌드된 max_model_len은 32768 입니다 (65536 아님)

아티팩트 `qwen2.5-72b-inst-tp8` 의 attention 버킷은 `[128, 256, 384, 512, 640, 768, 896,
1024, 2048, 4096, 8192, 16384, 32768]` 로 **최대 32768** 입니다(= 모델 `max_position_embeddings`
32768). 따라서 `--max-model-len 65536` 은 빌드값을 넘는 값이라 의미가 없습니다.

### 1-2. max_model_len 을 줄여도 serve 안 됩니다 (직접 3번 재현)

| 시도 | per-stage 가중치 할당 | 결과 |
|------|----------------------|------|
| 원본(serve_logs/8008.log) | 33.6 / 32.7 / 32.7 / 36.5 GiB | `-1803550720` |
| max-model-len **4096** | 33.6 / 32.7 / 32.7 / 36.5 GiB (동일) | `-1803550720` (동일) |
| max-model-len **65536** | 33.6 / 32.7 / 32.7 / 36.5 GiB (동일) | `-1803550720` (동일) |

- 실패 지점은 `furiosa_llm/api.py` 의 `NativeLLMEngine(...)` 초기화, 즉 **가중치를 NPU 에
  바인딩하는 단계**입니다. KV 캐시 할당에 도달하기도 전입니다.
- `max_model_len` 은 KV 캐시(attention 버킷)에만 영향을 줍니다. 가중치(stage 당 33~36 GiB)는
  모델 구조 + pp 분할로 정해지며 max_model_len 과 무관합니다. 그래서 **빌드 때 model_len 을
  줄여 빌드해도 per-stage 가중치 크기는 그대로라 serve 는 똑같이 실패**합니다.
- 세 번 모두 PP 할당 계획(가중치 GiB)이 글자 그대로 동일했습니다 = max_model_len 무관의 증거.

### 1-3. 다른 인자로도 bf16 72B 는 안 됩니다

- `-pp` 는 이미 최대 4(칩 4장)입니다. 더 쪼갤 수 없습니다.
- `-dp` 는 복제라 한 칩에 안 들어가는 큰 모델에는 도움이 안 됩니다.
- `--spare-blocks-ratio` 등은 스케줄러/KV 용이라 가중치 바인딩과 무관합니다.
- **진짜 벽**: pp4 일 때 칩당 인터칩 가중치 바인딩 한계가 약 32 GiB 부근입니다(실측: pp2 는
  29.7~31.4 GiB OK, **pp4 는 32.7 GiB 도 실패**). bf16 72B 는 stage 당 33~36 GiB 라 초과합니다.
  카드 용량(47.5 GiB)은 남는데도 드라이버가 DRAM 주소 배치에서 거부합니다(`-1803550720` =
  0x94800000, OOM 아님). 네이티브 메시지로는 `cannot find DRAM address for tensor` /
  `dram shape should have exactly one inter-chip axis` 계열입니다.

### 1-4. 그럼 무엇이 되나 (해법)

1. **FP8 양자화 72B** → stage 당 약 18 GiB(한계 아래) → `-pp 4` serve 가능. 현재 FP8 72B
   아티팩트가 없으니, 사전 양자화된 FP8 72B 를 받아 빌드(빌드는 GPU 불필요, NPU 점유 안 함)한 뒤
   pp4 로 띄우는 것이 정공법입니다.
2. **host-loop**(`qcn/serve_q25.py`, `qwen25_model.py`) — bf16 정확하지만 토큰당 수백 초로
   매우 느립니다. 이미 만들어져 있습니다.
3. qwen2.5-72b 의 벤더 prebuilt tp32 는 없습니다(레포에는 Llama-3.3-70B-tp32 만 있음).

---

## 2. 빌드의 두 단계: 트레이싱과 그래프 컴파일

`furiosa-llm build` 중에 두 진행바가 순서대로 나옵니다.

- `Model Tracing Progress: N/M`  (트레이싱)
- `Compilation Progress: N/M`    (그래프 컴파일)

한 줄 요약: **트레이싱은 PyTorch 모델을 "계산 그래프(설계도)"로 그려내는 단계**, **컴파일은 그
설계도 조각을 "NPU 가 실제 실행하는 기계어(EDF)"로 번역하는 단계**입니다.

진입점은 `furiosa_llm/artifact/builder.py` 의 `ArtifactBuilder.build()` 이고, 두 진행바 모두
`furiosa_llm/parallelize/new_pipeline_builder.py` 의 `build_pipeline()` 안에 있습니다
(트레이싱 1563-1571줄, 컴파일 1299-1307줄).

### 2-1. 트레이싱 (Model Tracing)

**무엇을 세나 (M):** M 은 **버킷(bucket) 개수**입니다. 버킷 = 입력 텐서의 모양 조합(prefill 용
attention_size 128/256/.../32768, decode 용 1토큰 등). 진행바 한 칸 = 그 모양으로 **모델 전체를
한 번 추적**한 것입니다. 레이어별이 아니라 "입력 모양별"입니다(`new_pipeline_builder.py:1563`,
버킷은 `pipeline_build_configurer.py:65-95`).

**무엇을 하나:**
1. HuggingFace 모델을 로드합니다. 이때 **양자화가 이미 적용된 상태**로 올라옵니다(bf16 면 그냥
   bfloat16, FP8 면 FP8). 즉 양자화는 트레이싱 전입니다(`metadata/metadata.py:338-374`).
   표기 `W16A16KV16` = 가중치/활성/KV 모두 bf16, `W8fA8fKV8f` = FP8(`optimum/types.py`).
2. `torch._dynamo.export` + `make_fx` 로 모델을 **ATen 수준 FX 그래프**로 추적합니다
   (`parallelize/trace.py:855-861`, `pipeline/builder/api.py`). 이때 실제 숫자 계산은 안 하고
   **FakeTensor**(모양만 있는 가짜 텐서)로 흘려 "어떤 연산이 어떤 순서로"만 기록합니다.
3. 그래프를 **커널 단위로 잘라(파티션)** 스테이지로 나눕니다(`graph_partitioner.py:44-73`).

**결과물:** 버킷별 ATen FX GraphModule(양자화 op 포함)을 커널 단위로 쪼갠 서브그래프들. 아직
NPU 기계어가 아니라 계산 설계도입니다.

**캐시(재빌드가 빨라지는 이유):** 추적 결과는 `~/.cache/furiosa/llm/graphmodules/*.fx` 에
저장됩니다(파일명 `Quantized_<클래스>-<해시>.fx`, `trace.py:1081-1084`). 캐시 키 = 모델/가중치/
양자화설정/torch 버전/입력 모양(`trace.py:1046-1063`). 이게 같으면 **다시 추적하지 않고 캐시
재사용** → 같은 모델 재빌드는 트레이싱을 건너뛰어 빠릅니다.

**스테이지 분할 규칙:** "어텐션 1개 = 1조각, 연속된 일반 연산(tokenwise) = 1조각"입니다
(`graph_partitioner.py:44-45`). 스테이지 종류는 2가지뿐
(`pipeline_build_configurer.py:120-125`): **ATTENTION**(배치·질의길이·KV길이로 변함),
**TOKENWISE**(임베딩·QKV/FFN/출력투영·출력헤드, 질의길이로만 변함). 개수는 대략
`2 × 레이어수 + 약간`.

### 2-2. 그래프 컴파일 (Compilation)

**무엇을 세나 (M):** 컴파일할 **태스크 수**입니다 = (스테이지 × 적용 버킷) 중 중복(data_blob)
제거(`new_pipeline_builder.py:1289-1300`). ATTENTION 스테이지는 attention 버킷마다 따로
컴파일되고(많아짐), TOKENWISE 스테이지는 질의길이가 같으면 한 번만 컴파일됩니다. 작은 모델은
~10여 개, 실제 프리셋은 수백 개.

**무엇을 하나 (닫힌 Rust 컴파일러 `npu-compiler`):** 각 스테이지의 FX 서브그래프를 NPU
기계어(EDF)로 낮춥니다(`converter.py:862-932` 가 네이티브 `compile()` 호출). 내부 단계(.so
문자열·소스 경로로 확인):
1. **전처리/분해**: aten → primitive 로 낮추고 트랜스포머 패턴(어텐션·MoE·softmax) 재작성.
2. **커널화(kernelize)**: primitive 들을 NPU 커널 단위로 묶음.
3. **택틱 탐색(tactic search)**: 커널마다 PE/메모리에 어떻게 펼칠지 최적 전략을 비용표로 탐색.
   **CPU 를 가장 많이 먹는 단계**(컴파일 Ray 액터가 `num_cpus=32`, `new_pipeline_builder.py:1076`).
4. **스케줄링**: 빔서치 스케줄러 + SRAM/레지스터 할당(부족하면 spill).
5. **클러스터 + DMA**: host 전송·load/store 삽입, DRAM 주소 충돌 해결.
6. **명령 생성**: LIR → 실제 명령/DMA 커맨드.
7. **EDF 백엔드**: 최종적으로 **C 소스를 생성 → 내장 C 컴파일러(`-O3 -nostdlib -static`)로
   PE 프로그램("renegade", RNGD 코드명)으로 컴파일**.

**결과물:** EDF blob = 컴파일·스케줄·명령까지 끝난 **실행 가능한 PE 프로그램**(단순 그래프가
아님). `target_npu`(예: `renegade-8pe`), graph_metadata, compiler_config 가 이 과정에 들어갑니다.

**왜 버킷마다 미리 컴파일하나:** NPU 컴파일러는 **정적 shape** 프로그램을 만듭니다(컴파일된
모양에서만 실행, `trace.py:858`). 그래서 prefill 길이 브래킷·decode(1토큰) 등을 미리 여러 개
컴파일해 두고, 런타임에 **실제 입력보다 크거나 같은 가장 작은 버킷을 골라 패딩**해서 씁니다.
버킷이 많으면 패딩 낭비는 줄지만 빌드가 길고 번들이 커집니다.

**번들링:** 컴파일된 EDF 들은 `binary_bundle.zip` 에, 어떤 버킷이 있는지는
`pipeline_metadata_list`(attention/tokenwise 버킷)에, 가중치는 옆에 safetensors 로 저장됩니다
(`builder.py:402-503`). serve 때 런타임이 입력 모양에 맞는 EDF 를 골라 거기에 **가중치를
바인딩**합니다.

### 2-3. 1번과의 연결

1번에서 72B 가 죽는 "가중치 바인딩"이 바로 **이 EDF 에 safetensors 가중치를 NPU DRAM 으로
올리는 serve 단계**입니다. 트레이싱·컴파일 자체는 통과해서 아티팩트가 만들어졌지만, serve 때
pp4 로 stage 당 33~36 GiB 를 칩에 바인딩하다가 인터칩 DRAM 배치 한계(~32 GiB)를 넘어 실패합니다.
그래서 빌드(트레이싱/컴파일) 인자가 아니라 **가중치 크기 자체(FP8)**를 줄여야 풀립니다.

---

## 출처

- 빌드 설정·버킷: `artifacts/qwen2.5-72b-inst-tp8/artifact.json`
- serve 실패 로그: `chat/serve_logs/8008.log` + 본 분석의 4096/65536 재현(동일 결과)
- 빌드 파이프라인: `furiosa_llm/{artifact/builder.py, parallelize/new_pipeline_builder.py,
  parallelize/trace.py, parallelize/graph_partitioner.py, parallelize/pipeline_build_configurer.py,
  parallelize/pipeline/builder/converter.py}`
- 더 깊은 빌드/서빙 정독: `info/ALL_about_build_serve.md`
