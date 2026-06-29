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

> 📞 **런타임 구조 — 실측 콜그래프(2026-06-25).** `furiosa-llm build` 를 실제로 돌리며 gdb·py-spy·bpftrace 로 잡은 전체 콜그래프는 [`../callgraph-analysis/03-synthesis/BUILD-CALLGRAPH-WALKTHROUGH.md`](../callgraph-analysis/03-synthesis/BUILD-CALLGRAPH-WALKTHROUGH.md)(정적 file:line 은 [`build-static-callgraph.md`](../callgraph-analysis/01-static/build-static-callgraph.md)). 핵심 실측 3가지:
> - **빌드는 Ray 멀티프로세스.** `--num-*-workers 1`(기본)이라도 **로컬 Ray 클러스터**를 띄우고, 트레이싱은 `ray::LocalPipelineGenerationActor`(`@ray.remote num_cpus=24`), 컴파일은 `ray::TaskCompileActor`(`@ray.remote num_cpus=32`) **별도 프로세스**에서 돕니다. 드라이버(`furiosa-llm build`)는 `build_pipeline → ray.get`(`new_pipeline_builder.py:1586`)에서 **파킹**만 합니다 (gdb: 드라이버 ~120 스레드 대부분 Ray RPC 대기). 워커 수를 늘리면 in-process 병렬이 아니라 **액터(프로세스)가 늘어나는 것**.
> - **빌드는 NPU 를 안 씁니다.** 드라이버·워커 모두 `/dev/rngd/*` 미오픈, 커널 트레이스에 빌드발 doorbell/DMA 0 (그 트래픽은 상주 furiosa-smi `tokio-runtime-w` 노이즈). 컴파일은 **순수 호스트 CPU**. 컴파일러 본체는 `furiosa.native_common.compiler` = `native_llm_common.cpython-312*.so`(143 MB, **스트립**)이고, `compile()`(`converter.py:913`)이 PyO3 경계.
> - **`failed to lower the operator O…(no tactic)` 의 위치.** 이 에러는 `TaskCompileActor.compile_task → compile()` 안 native lowering 에서 납니다. 실측에서 `Qwen2.5-Coder-1.5B-Instruct -tp 4`(default preset)는 컴파일 `stage_0`(Embedding→첫 QkvProjection)에서 `O1089 (no tactic)` 로 **실패**했습니다 (트레이싱 49태스크 통과 후, 컴파일 78태스크 중 stage_0; 계측과 무관한 컴파일러 한계). 네이티브 컴파일러는 자체 병렬 lowering 스레드풀(파킹 ~62 + 활성 다수)로 동작 — 스트립 `??` 프레임의 region(=컴파일 패스) 간이 명명은 [`gdb_build.native_names.md`](../callgraph-analysis/03-synthesis/full-callgraphs/gdb_build.native_names.md).

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

- `SUPPORTED_TP_SIZES = {4, 8, 32}` — 그 외 값은 거부됩니다. **⚠️ 단 이 로컬 validator 값은 실제 컴파일러 능력보다 앞서 있습니다.** Furiosa 공식 2026.2.0 문서는 빌드 시 `tensor_parallel_size`를 **4 또는 8만 지원**한다고 명시("can only be 4 or 8 in the 2026.2.0 release; future releases will lift this limitation", [model-parallelism](https://developer.furiosa.ai/latest/en/furiosa_llm/model-parallelism.html)). **tp=32는 검증은 통과하지만 네이티브 컴파일에서 실패합니다**(8절 `embedding_table` 에러).
- `pp ≥ 1`.
- 필요한 디바이스 수 = `ceil(tp / 8) × pp ≤ 8` (= `MAX_DEVICES`).
- `NUM_PES_PER_NPU = 8` (RNGD 1장 = 8 PE, `device.py:6`).

가능한 조합 (디바이스 합계 ≤ 8 안에서):

| `tp` | `pp` | 카드 수 | 비고 |
|---:|---:|---:|---|
| 4 | 1~8 | 1~8 | `tp=4`은 1장 안에서 PE 4개만 사용 |
| 8 | 1~8 | 1~8 | `tp=8`은 1장 풀 PE |
| ~~32~~ | — | — | **직접 빌드 불가(2026.2.0): tp build는 4/8만 공식 지원.** tp=32는 4칩 inter-chip 구성을 강제하는데 네이티브 컴파일러가 임베딩 입력에 inter-chip shape guide를 못 붙여 stage_0에서 실패(8절). furiosa **prebuilt** tp=32 아티팩트는 serve만 가능(내부/구버전 툴 산출 추정) |

> 📌 **4장(32 PE) 활용 — 문서 공식 권장 vs 2026.2.0 실제:** 문서는 `tp8×pp4×dp1`/`tp4×pp2×dp2`(곱=32)를 권장하지만, ⚠️ **이 SDK에선 `pp>1` 빌드가 전 모델 실패**(3.1절 🔴) + `tp=32` 직접 빌드도 실패(8절). 그래서 실제로 쓸 수 있는 빌드는 **`tp8`뿐**: 1장(47.5GB)에 들어가는 모델은 `-tp 8` 로 빌드한 뒤, **serve 때 `--devices`+`-pp`/`-dp` 로 4장을 활용** — dp=4(통째 복제, 처리량 4배) 또는 pp(레이어를 장마다 분할 — 자리·KV 확장용, ⚠️ 단일 요청 가속 아님: 실측 1장 50.3 vs pp2 48.3 tok/s) 또는 둘 조합(`dp×pp≤4`). ⚠️ **serve-time `-pp`는 빌드 pp(위 🔴)와 달리 정상 동작**(3.1절 ✅, 2026-06-09 실측). **1장 초과 모델(예: Llama-3.3-70B fp8 67.7GB, bf16 32B)** 은 로컬 빌드 경로가 없고 → **furiosa-ai prebuilt tp32 다운로드 serve**만 가능(8절 우회).

### 3.1 pp(`-pp >1`) 빌드 — 2026.2.0에선 **어떤 모델도 안 됨** (Llama 포함, 2026-06-02 정정)

> 🔴 **2026-06-02 중대 정정 (4-에이전트 적대검증).** 이전 이 절의 *"pp는 Llama·GPTJ·Bert·Roberta만 됨"* 은 **틀렸습니다.** 그건 `block_slicer.py:679-697` dict **등록 여부**에서 *추론*한 것일 뿐 — **pp>1 빌드가 성공한 적은 한 번도 없습니다**(이 머신 아티팩트 12개+ 전부 `pipeline_parallel_size=1`, 빌드 캐시에도 pp 조각 0개). 2026-06-02 `Llama-3.3-70B-fp8 -tp8 -pp4` 실측 → dict 등록 모델 Llama인데도 `ValueError: Unexpected node type ... got furiosa.module_marker`(`block_slicer.py:322`)로 사망.
> - **원인:** 차세대 빌더 `new_pipeline_builder.py:475`가 `gen_config`를 `other_configs` 없이 호출 → `use_marker_based_block_slicer=False` 강제(`mppp/api.py:214`) → **dict 가위** 사용. 그런데 **같은 빌더가 trace 때 `furiosa.module_marker`를 항상 삽입**(`new_pipeline_builder.py:435-463`, composable kernel이 partitioning_config 항상 공급) → dict 가위는 마커를 못 치워(remove_marker_nodes는 marker 경로에서만 호출) 충돌.
> - **pp=1만 되는 이유:** 단일 device면 슬라이싱을 아예 안 거침(`mppp/api.py:108` early-return) → 마커 무해. 그래서 모든 성공 아티팩트가 pp=1.
> - **사용자 수정 불가:** marker 가위 knob 없음(CLI·env·`--additional-model-config` whitelist·build kwarg 전부 0건; grep 확인). 소스 패치(`new_pipeline_builder.py:475`에 `other_configs={"use_marker_based_block_slicer":True}` 주입)만 가능하나 미지원·미검증이고, pp4는 4칩이라 패치해도 8절 inter-chip DramShapeGuide 벽에 또 막힐 수 있음.
> - **결론: 2026.2.0 공개 빌드 경로로 pp>1은 전 모델 불가.** dict 미등록(qwen·exaone·moe)은 더 일찍 `NotImplementedError`, dict 등록(Llama 등)은 더 가서 marker 충돌 `ValueError` — **둘 다 산출물 0.** 1장 초과 모델은 pp 대신 **prebuilt tp32 다운로드 serve**(8절 우회) 뿐.
>
> 아래 옛 설명은 marker↔dict 메커니즘 이해용으로 남기되, **"Llama는 pp가 된다"는 결론만 폐기**합니다.

> ✅ **단, "빌드 pp"와 "serve pp"는 다릅니다 — serve-time `-pp`는 됩니다 (2026-06-09 실측).** 위 🔴 는 전부 **`furiosa-llm build -pp >1`**(컴파일 때 pp 그래프를 새로 짜는 것) 얘기입니다. 반면 **이미 만든 `tp8`(pp=1) 아티팩트를 `furiosa-llm serve ... -pp 2`로 띄우는 것**은 정상 동작합니다 — serve 런타임이 **이미 컴파일된 블록(레이어)들을 카드별 파이프라인 스테이지로 나눠 배치**하기 때문(블록 단위 재컴파일이 아니라 배치라 marker/dict 슬라이서 벽을 안 탐).
> - **실측(coder7, qwen2.5-coder-7b-inst-tp8, pp=1로 빌드된 아티팩트):**
>   - `serve ... --devices npu:0,npu:1 -pp 2` → 로그 `Resolve 1 pipeline for 1 DP groups (DP=1, PP=2)`, `PP device#0 ... Model weights=6.7 GiB` / `PP device#1 ... 7.5 GiB`(가중치 14.2GiB가 2장에 쪼개짐) → Uvicorn 정상, 생성 정상.
>   - `-pp 4`(4장) → `DP=1, PP=4`, 가중치 3.7+3.1+3.1+4.5 GiB 4분할 → 정상.
>   - `-dp 2 -pp 2`(4장) → `DP=2, PP=2` → 정상. 즉 **카드 수 = tp_chip(1) × pp × dp**, dp는 `-dp` 생략 시 카드수/(tp×pp)로 자동 추론.
> - **즉 serve의 병렬화 3축**: `-tp`(아티팩트가 고정), `-pp`(레이어를 장마다 분할 — 한 요청을 파이프라인으로), `-dp`(통째 복제 — 동시요청 throughput). 채팅 UI(`chat/chat_app.py`)가 tp8 모델에 dp·pp 선택을 노출(`dp×pp≤4`, tp32는 비활성).
> - ⚠️ **serve pp는 "자리"용이지 단일 요청 가속이 아님 (2026-06-10 실측 정정):** coder7 단일 요청 1장 50.3 tok/s vs pp2 48.3 tok/s — 한 토큰이 스테이지를 순차 통과(파이프라인 버블)+카드간 전송으로 오히려 ~4% 느림. pp 가치 = ① 1장 초과 모델 분할 적재 ② KV 풀 확장(pp2면 38.8+38.0 GiB — 1장 ~40GiB의 2배) ③ 동시부하 파이프라이닝. **단일 대화 가속은 tp(=tp32 prebuilt)만 가능**(한 연산을 여러 장이 동시 분담). 적용 검증법(로그 `DP=`/`PP=` 줄, `PP device#N` 가중치 분할, smi 사용률 패턴)은 `chat/README.md` "dp·pp 가 진짜 적용됐는지" 절 참고.
> - 📌 **Q&A (2026-06-10): "pp 빌드는 안 되지만 serve pp는 되니까, tp8로 빌드하고 serve에서 `-pp`를 주면 되지 않나?"** — 메커니즘은 맞고 이미 그게 표준 경로입니다(위 ✅). 다만 모델 상황에 따라 의미가 갈립니다.
>   - **1장에 들어가는 모델(FP8 30B 코더 등):** pp를 줄 실익이 거의 없습니다. 단일 요청은 위 실측처럼 오히려 ~4% 느려지고, 처리량은 dp=4(통째 복제)가 정답입니다(위장 FP8 30B 실측 c32 1036 tok/s, `README_all_change.md`). pp는 KV 풀을 2배로 키워 장문맥·고동시성을 노릴 때만 고려.
>   - **1장 초과 bf16 모델(Qwen3-Coder-30B-A3B-Instruct 원본 등):** serve pp 발견(06-09)으로 **이론상 새 경로**가 열렸습니다 — 47.5GB 제한은 serve(적재) 쪽에만 걸리고 빌드는 호스트 컴파일이므로, bf16 `-tp 8` 빌드 후 serve `-pp 2`(가중치 ~61GB 개산 → ~30GB/장)로 쪼개 올리는 그림. 기존 "bf16 30B 불가" 결론(8절·인벤토리 표)은 serve pp 발견 *이전*의 판정이라 이 경로는 빠져 있었습니다. **단 미검증 3가지:** ① bf16 30B tp8 빌드 자체가 실측 0 — 트레이싱 피크 RAM 주의(FP8 30B도 default len에서 100GB+였음, `BUILD_COMPIL.md`; 호스트 RAM 125GB), `--max-model-len ≤65536` 캡은 양자화 무관하게 동일 필수(kv4<tp8). ② serve 게이트는 **BF16 MoE도 거부**(2026-06-10 mini-qwen3-moe 실측 — "BF16-weight MoE면 serve될 여지" 가설 반증) → masquerade 필요(BF16 MoE는 mini로 부팅+NPU 생성까지 실증, `README_all_change.md`). ③ 위장 아티팩트에 `-pp`를 건 실측은 아직 없음(pp 해석은 model_type 무관 기하 분할이라 동작할 것으로 예상). 비용 대비: pp2면 4장에서 dp≤2라 처리량은 FP8 dp4의 절반 이하 + 단일 속도 이득 없음 — **bf16 충실도(무양자화 가중치)가 꼭 필요할 때만 시도할 가치.**

### (구) pp 슬라이서 구조 — Llama·GPTJ·Bert·Roberta dict 등록 (메커니즘 참고용, 결론은 위 🔴 정정)

먼저 이 절에 나오는 파일들이 뭘 하는 애들인지 보고 가면 쉽습니다. 빌드를 **공장 조립 라인**으로 보면:

**A. 빌드를 진행하는 쪽** (`furiosa_llm/` = 공장) — pp를 정하는 순서대로

| 파일 | 하는 일 (쉽게) |
|---|---|
| `device.py` | 카드·PE를 어떻게 묶을지 정함 (`tp`→칩 안에서 쪼갬, `pp`→카드를 단계로 쌓음) |
| `pipeline/builder/api.py` | 모델을 추적(trace)하고 **marker(표식)를 심어** 파이프라인 조립 |
| `mppp/api.py` | **"어느 layer를 어느 카드(스테이지)에 둘지" 계획**을 세움 |
| `block_slicer.py` | 모델을 **블록 단위로 자르는 가위** — marker(arch 무관) ↔ **dict(Llama 등 4종만)** 두 종류 |
| `new_pipeline_builder.py` | 버킷마다 진짜로 자르는 단계 — **항상 dict 가위를 씀**(`:475`, `other_configs` 없이 호출). 동시에 trace 때 marker를 심어놓고(`:435-463`) dict 가위가 그걸 못 치워 **pp>1은 전부 실패**(위 🔴) |

**B. 모델이 어떻게 생겼는지 정의하는 쪽** (`furiosa/models/` = 설계도) — 참고용 (모델 layer 구조)

| 파일 | 하는 일 (쉽게) |
|---|---|
| `…/architecture/<model_type>.py` | 각 모델(`exaone4`,`qwen3_moe`…)의 layer 구조 정의 |
| `make_layers` (`common/utils/blocks.py`) | 디코더 블록들을 `nn.ModuleList`로 묶음 (이게 `model.layers`) |
| `MoELayer` (`core/layers/moe/moe.py`) | MoE expert를 **융합 가중치**로 구현 (개별 `nn.ModuleList`가 아님) |

**한 줄로:** `block_slicer`엔 가위가 둘(marker·dict)인데, 빌드의 진짜 자르는 단계가 **"옛 dict 가위"를 쓰면서 마커를 못 치워** — dict 미등록(qwen·exaone·moe)은 `NotImplementedError`, dict 등록(Llama 등)은 marker 충돌 `ValueError`로 **둘 다 pp 빌드 실패**(위 🔴 정정, 2026-06-02). 즉 "dict 등록 = pp 됨"이 아니라 **등록돼도 marker 때문에 안 됩니다.** (tp32 임베딩 실패는 또 다른 native 단계라 별개.)

> ❌ **실패 모드 ① dict 미등록 (qwen·exaone·moe) → `NotImplementedError`** (2026-06-01 실측). 사용자가 `qwen3_moe`를 직접 빌드:
> ```
> furiosa-llm build Qwen/Qwen3-Coder-30B-A3B-Instruct ... -tp 8 -pp 4 --max-model-len 65536
> → NotImplementedError: Block slicing for Qwen3MoeForCausalLM is not supported.   (block_slicer.py:727)
> ```

**왜 marker 경로가 안 구해주나 (이게 핵심):** 슬라이서는 둘 — ① marker(arch 무관) ② dict(`get_block_slicing_edges`, 위 4종만).
초기 그래프 추적 한 곳은 marker를 켜서(`pipeline/builder/api.py:662`) **코드만 보면 "arch 무관"처럼 보입니다.**
하지만 **실제 아티팩트를 만드는 "버킷별 파이프라인 생성"(`new_pipeline_builder.py:475`)은 marker 플래그를 안 넘깁니다**
→ `gen_pp_mpc`가 기본값(False)으로 **dict 가위**(`mppp/api.py:117`)를 타고 → dict에 없는 모델은 `NotImplementedError`.
이 버킷 단계가 빌드를 실제로 결정하므로, marker 경로가 있어도 무력합니다.

| `model_type` | pp(`-pp N`) | 근거 |
|---|---|---|
| `llama` / `gptj` / `bert` / `roberta` | ✅ 됨 | dict에 등록됨 (`block_slicer.py:679-701`) |
| `qwen2` / `qwen3` / `qwen3_moe` / `exaone4` | ❌ **안 됨** | dict 미등록 → `NotImplementedError` (qwen3_moe 2026-06-01 실측 실패) |
| 멀티모달(`mllama4` 등) | ❌/별도 | 위와 같음 |

> **⚠️ 제가 두 번 틀렸던 기록 (남겨 둡니다 — 같은 실수 방지용):**
> 1. **처음:** dict만 보고 "Llama 외 pp 불가" → ✅ **맞았음.**
> 2. **중간:** marker 경로(`api.py:662`)를 보고 "marker라 arch 무관, qwen·exaone·moe 다 됨"으로 **뒤집음** → ❌ 틀림.
> 3. 근거로 든 "EXAONE 빌드가 pp 통과"는 사실 **같은 단계(Model Tracing)에서 OOM으로 먼저 죽은 것** — 슬라이싱 통과가 아니었음(착각).
> 4. 사용자가 `qwen3_moe`를 `-tp8 -pp4`로 직접 빌드 → `NotImplementedError` → ②③이 틀렸음이 **실측으로 확정.**
>
> **교훈:** marker 경로는 존재하지만 **버킷 생성 단계(`new_pipeline_builder.py:475`)가 그걸 안 써서** 무력하다. 코드 한 줄(`api.py:662`)만 보고 단정하지 말고 **빌드를 돌려봐야** 진실이 나온다.

#### 🔍 더 깊이 — 왜 대부분 모델은 pp가 안 되나 (쉽게)

`block_slicer.py`에는 모델을 자르는 **가위가 두 개**예요:
- **새 가위(marker):** 모델에 스티커를 붙여 자릅니다 → 어떤 모델이든 OK.
- **옛 가위(dict):** "내가 아는 모델 목록"(`MODEL_ARCH_…` = Llama·GPTJ·Bert·Roberta)에 있을 때만 자릅니다 → 그 외엔 `NotImplementedError`.

빌드는 자르기를 **두 군데**서 합니다. 처음 한 번은 **새 가위**로 슬쩍 보고(`pipeline/builder/api.py:662`),
그 다음 **버킷마다 진짜로 자를 때**(`new_pipeline_builder.py:475` → `gen_pp_mpc`)는 **가위 종류를 안 알려줘서
옛 가위(dict)로 자릅니다**(`mppp/api.py:117`). 그래서 목록에 없는 qwen·exaone·moe는 이 단계에서
`Block slicing for ...is not supported` 로 멈춥니다.

> **한 문장:** 새 가위가 있어도 **빌드의 "진짜 자르는 단계"가 옛 가위를 쓰기 때문에**, 목록(Llama 등)에 없는 모델은 pp가 안 됩니다.
> (그래서 위 표가 ✅ 4종 / ❌ 나머지 인 거예요. — 이게 코드만 보고 "marker라 다 된다"고 했다가 실측에서 깨진 지점입니다.)

### 3.2 pp가 필요 없을 때 — dp(data parallel)

모델이 카드 1장에 들어가면(예: 32B FP8 `tp=8`) pp 대신 **dp가 낫습니다.** dp는 빌드 옵션이 아니라
**serve 할 때 `--devices`로 카드를 더 주면 엔진이 자동 인식**(`device.py:101` dp = 총PE / (pp×tp))해서
모델 복제본을 여러 카드에 띄웁니다. 통신 오버헤드가 없어 pp보다 효율적이지만, 처리량이 N배가 될 뿐 단일 요청이
빨라지는 건 아닙니다.

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

#### 5.2.1 prefill 과 encoder 는 같은 건가요?

자주 헷갈리는 부분이라 짚고 갑니다. **결론부터: 다릅니다.** 비슷해 보이는 건
둘 다 "입력 전체를 한 번에(병렬로) 통과시키는, 연산량이 큰 단계"라는 점 하나뿐입니다.
그래서 prefill 을 가끔 "프롬프트를 인코딩한다"고 느슨하게 부르기도 하지만, 셋이
완전히 다릅니다.

- **무엇이냐(범주)가 다릅니다.**
  - encoder 는 *모델 구조의 한 덩어리*입니다. 원조 트랜스포머의 encoder 스택,
    BERT 같은 encoder-only 모델이 여기 해당합니다
    (출처: Vaswani et al. 2017, *Attention Is All You Need*; Devlin et al. 2018, *BERT*).
  - prefill 은 *디코더를 돌리는 한 단계(phase)*입니다. furiosa-llm 에서는 위 표처럼
    `kv_cache_size = 0` 인 **버킷 분류**로 정의됩니다
    (`metadata/config_types.py` `AttentionBucket.is_prefill` = `kv_cache_size == 0`).
- **어텐션 방식이 다릅니다.** encoder 는 *양방향*입니다 — 한 토큰이 앞뒤 모든 토큰을
  봅니다. prefill 은 프롬프트를 한 번에 처리하긴 해도 디코더라서 *인과(causal) 마스크*가
  걸립니다 — i번째 토큰은 자기 이전 토큰까지만 봅니다. (예외: prefix-LM/T5 류는 입력을
  양방향으로 보기도 하지만, 표준 causal LM 은 prefill 도 인과 마스크입니다.)
- **가중치(정체)가 다릅니다.** encoder 는 *자기 전용 가중치를 가진 별도 부품*입니다.
  prefill 은 *디코더와 똑같은 가중치를 모드만 바꿔서 돌리는 것*입니다. 실제로 이 빌드의
  아티팩트도 prefill/decode/extend/tokenwise 가 **params 파일 하나를 공유**합니다
  (가중치는 한 벌, 버킷별로는 컴파일된 EDF 그래프만 다름 — `info/ALL_about_build_serve.md`).
- **결과물(목적)이 다릅니다.** encoder 는 *최종 임베딩(z)* 을 내놓아 디코더의
  cross-attention 이나 분류 헤드가 씁니다. prefill 은 *KV 캐시 + 첫 출력 토큰*을 내놓아
  뒤이은 decode 루프(한 토큰씩 생성)를 시작시킵니다
  (출처: NVIDIA, *Mastering LLM Techniques: Inference Optimization*).

**이 프로젝트에 대입하면:**

| | encoder 계열 | prefill (decoder 계열) |
|---|---|---|
| 이 레포의 예 | 임베딩/리랭커 모델 (`role: embedding`/`reranker`, `gen: false`) — Qwen3-Embedding-8B, Qwen3-Reranker-8B | 생성 모델 (`ForCausalLM`) — Llama, Qwen2/3, EXAONE |
| 버킷 | pooling 프리셋, **decode·append 없음** (`README_preset.md`) | prefill + decode + append + tokenwise |
| 출력 | 풀링된 임베딩 (lm_head 없음) | KV 캐시 + 다음 토큰 (lm_head 있음) |
| 엔드포인트 | `/v1/embeddings` | chat/completion |

> ⚠️ 한 가지 주의: 요즘은 "임베딩 = encoder"라는 등식도 깔끔하지 않습니다. 이 레포의
> Qwen3-Embedding 처럼 **decoder 기반 임베딩 모델**(E5-Mistral, NV-Embed, gte-Qwen2,
> LLM2Vec 등)이 오히려 MTEB 상위권입니다. 즉 "임베딩이라서 encoder"가 아니라,
> *어텐션이 양방향이고 별도 부품이며 최종 임베딩을 내놓느냐*로 구분하는 게 정확합니다.

#### 5.2.2 presets.py 의 prefill / decode / append 괄호 안 숫자

`artifact/presets.py` 의 `BucketConfig` 는 버킷 종류마다 튜플 모양이 다릅니다. 이유는
`artifact/resolver.py:101-108` 에서 각 튜플을 내부 `AttentionBucket(batch_size,
attention_size, kv_cache_size)` 로 바꾸는 방식이 다르기 때문입니다. 기억할 공식 하나:
**`input_ids_size = attention_size − kv_cache_size`** (이번에 새로 처리하는 토큰 수).

**prefill — `(batch_size, context_length)` 2개**
```python
AttentionBucket.prefill(b, c)  # → (b, c, kv_cache_size=0)
```
- `b` = 한 번에 처리하는 요청 수(배치). 프리셋들은 거의 다 **1** (프롬프트는 한 요청씩).
- `c` = 프롬프트 길이. `kv=0` 이라 `input_ids_size = c` → **c개 새 토큰을 한 번에** 처리.
- 즉 두 번째 숫자 = "한 방에 밀어 넣는 프롬프트 토큰 수".
- 예) `QWEN_3_32B_FP8` 의 `(1,128)…(1,1024)` = 배치1, 프롬프트를 128 간격으로 1024까지 커버.

**decode — `(batch_size, context_length)` 2개**
```python
AttentionBucket.decode(b, c)  # → (b, c, kv_cache_size=c-1)
```
- `b` = 배치. decode는 여러 시퀀스를 동시에 굴리므로 **1~256** 까지 큼(continuous batching).
- `c` = ⚠️ **총 컨텍스트 길이**(이미 캐시된 토큰 + 새 토큰 1개). `kv=c-1` 이라
  `input_ids_size = 1` → **새 토큰은 항상 1개**.
- 헷갈림 주의: decode의 두 번째 숫자는 "새로 만드는 토큰 수"가 **아니라**, 그 1개의 새
  토큰이 **얼마나 긴 문맥을 바라보느냐**(총 길이)입니다. 그래서 `(1,1024)…(1,128*1024)` 처럼
  1K~128K로 퍼져 있는 건 "대화가 그만큼 길어졌을 때 다음 한 토큰 생성"을 각각 커버하는 것.

**append (= extend) — `(batch_size, attention_size, input_ids_size)` 3개**
```python
AttentionBucket(b, a, a - i)  # kv_cache_size = a - i
```
- `b` = 배치(프리셋은 `_build_append_buckets` 로 전부 1 고정).
- `a` = 총 어텐션 윈도(캐시 + 새 토큰).
- `i` = 이번에 한꺼번에 넣는 **새 토큰 수**. 캐시된 건 `a − i` 개.
- 예) `(1, 512, 128)` = 배치1, 총 512 윈도 중 **128개가 새 토큰**, 384개는 이미 캐시.
- prefix-cache 일부 재사용 + 새 청크 추가(chunked prefill)에 쓰입니다. 제약: `a > i`.

**왜 prefill·decode는 2개인데 append만 3개인가** — `(attention_size, kv_cache_size)` 평면에서
prefill은 `kv=0` 끝, decode는 `input_ids_size=1` 끝, 즉 **양쪽 극단**이라 숫자 2개로 한 점이
정해집니다. append는 그 사이 **일반 케이스**라 `kv`·`input` 둘 다 자유 → 3개가 필요합니다.
(`is_prefill`=`kv==0`, `is_decode`=`kv>0 & input==1`, `is_extend`=`kv>0 & input>1`,
`metadata/config_types.py:164-173`.)

> 임베딩/리랭커(`QWEN_3_8B_POOLING_PRESET`)는 `prefill_buckets=((1, 8192),)` 만 있고
> decode·append 가 없습니다 — 생성을 안 하니 디코딩 버킷이 필요 없기 때문입니다.

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

### 5.6 등록된 preset 목록 (`artifact/presets.py:277` `PRESET_REFS`)

> 갱신(2026-06-15): 아래 표는 초기 7종 기준. 현재 `PRESET_REFS` 는 15개 항목 / 11종 preset 으로
> 확장됨(Qwen2.5-Coder·Qwen3-8B-pooling·Qwen3-30B-A3B·Qwen3-Coder-30B/480B·mini-smoke 추가).
> 최신은 SDK `artifact/presets.py:70-265`(정의) / `:277-384`(refs) 참고.

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
| 컴파일 중 `Compilation failed for stage id: stage_N ... Graph input#K must have Broadcast or Fixed DramShapeGuide (Name: embedding_table 또는 attn_O_proj_weight)` | 번들 네이티브 컴파일러 **2026.2.0** (rev 9f92da069; `furiosa.native_common.compiler.full_version()`로 확인 — PATH의 standalone `furiosa-compiler` 2025.3.0가 **아님**) | **근본원인 = tp=32가 2026.2.0에서 공식 미지원** (빌드 tp는 4/8만 — [model-parallelism](https://developer.furiosa.ai/latest/en/furiosa_llm/model-parallelism.html): *"tensor_parallel_size can only be 4 or 8 in the 2026.2.0 release; future releases will lift this limitation"*). 로컬 `validator.py:246`이 {4,8,32}로 32를 허용해 **검증은 통과하나 네이티브 컴파일에서 실패**.<br>**메커니즘 (2026-06-02 6-에이전트 적대검증·반례 0):** tp=32 → `device.py:124-140`에서 fusion_granularity=min(8,32)=8, across_fusioned=32//8=4 → **4칩 inter-chip** 구성. inter-chip 가중치는 Broadcast/Fixed DramShapeGuide(칩 간/칩 내 축 배치)가 필수인데 **공개 빌드 경로가 이걸 안 채움** — `converter.py:1461-1467` `generate_graph_metadata`는 `valid_length`만 set하고 `input_dram_shape_guide` 미설정, 네이티브 `GraphMetadataBuilder`도 노출 메서드가 `build/from_yaml/set_valid_length` 3개뿐(가이드 주입 setter 없음). 그래서 inter-chip 가중치가 `Free`로 남아 prelower 거부.<br>**⚠️ 어느 가중치가 먼저 터지는지는 컴파일 순서·모델에 따라 다름** (둘 다 inter-chip 가중치, root cause 동일): ▸ 2026-05-30 Qwen3-Coder-30B-A3B = `embedding_table` @ **stage_0**(Embedding→QKV). ▸ 2026-06-02 Llama-3.3-70B-fp8 = `attn_O_proj_weight` @ **stage_2**(block0 OutputProjection→block1 QkvProjection, symbol=128). stage 분해는 `block_slicer.py:1057-1102` per-layer 루프(모델 분기 없음)라 **모델 무관하게 동일 실패**(universality high). **tp≤8은 단일 칩이라 무관**(성공 아티팩트 전부 tp8). **MoE·`--max-model-len` 무관.**<br>**우회 (검증됨):** ✅ FP8 `-tp 8` 빌드 후 serve 시 `--devices npu:0~3` → dp=4 자동(처리량 4배). ❌ env/config/CLI 어디에도 가이드 공급 knob **없음**(grep 0건, 적대검증 확정) — `embedding_as_single_block` 류로도 안 됨(shape guide는 네이티브 책임). ✅ 진짜 tp32가 필요하면 **furiosa-ai prebuilt**(내부 toolchain `furiosa_llm_version=b62dbc1`/`compiler=d19a92a2f2`로 만든 tp32 아티팩트, HF 캐시 다운로드)만 길. 근본해결은 차기 SDK(2026.3+). (2026-05-30 Qwen3-Coder-A3B·2026-06-02 Llama-70B-fp8 실측, 6-에이전트 적대검증) |
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
| `NotImplementedError: Block slicing for {ModelClass} is not supported.` | `parallelize/block_slicer.py:727` | `-pp >1`인데 그 architecture가 dict **미등록**(qwen·exaone·qwen3_moe 등). dict 등록은 Llama·GPTJ·Bert·Roberta뿐이지만 ⚠️ **등록돼도 pp는 안 됨**(아래 marker 행). 회피: `-pp 1` 빌드 + serve 시 `--devices`로 dp 활용 (2026-06-01 qwen3_moe 실측, 3.1절) |
| `ValueError: Unexpected node type. expected: {aten.embedding, aten.add}, got: {furiosa.module_marker}` | `parallelize/block_slicer.py:322` (`get_first_rms_norm_edge_names` ← `get_block_slicing_edges` ← `gen_pp_mpc` ← `new_pipeline_builder.py:475`) | `-pp >1`인데 architecture가 dict **등록**(Llama 등)일 때 나는 에러. **pp>1 빌드가 전 모델 불가인 진짜 증거** (2026-06-02 `Llama-3.3-70B-fp8 -tp8 -pp4` 실측, 71분 trace 후 사망). 차세대 빌더가 `gen_config`를 `other_configs` 없이 호출→`use_marker_based_block_slicer=False`→dict 가위인데, 같은 빌더가 trace 때 `module_marker`를 항상 삽입(`new_pipeline_builder.py:435-463`)→dict 가위가 마커를 못 치워 충돌. pp=1은 슬라이싱 자체를 안 거쳐 무해. **사용자 knob 없음(소스 패치만, 미지원).** 회피: `-pp 1`+dp, 1장 초과 모델은 prebuilt tp32 serve. 상세 3.1절 🔴 (4-에이전트 적대검증) |
| serving: `Required PEs: N, Actual: M` | runtime | 빌드 tp ≠ 가용 PE → `--devices` 늘리거나 작은 tp로 재빌드 |
| serving: HBM OOM | runtime | `--max-model-len` 줄여 재빌드 또는 카드 추가 |
| 컴파일 중 `failed to lower the operator O861 (no tactic): AttentionKernel(... mask_tagged_shape: [0_1=128, 4_1=131072])` (stage_1, attention) | 번들 컴파일러 2026.2.0 (attention kernel tactic 부재) | ⚠️ **2026-06-02 재검증으로 메커니즘 정정** (이전 "64K 보편 상한" 서술은 틀림). 원인은 attention_size 절대값이 아니라 **`num_key_value_heads < tp`**. furiosa attention은 **head 단위 분할**(GQA/Megatron-TP 표준 동작, seq/context 분할 없음)이라 **131072 KV 길이는 tp를 키워도 PE당 그대로 131072**(tp32라서 되는 게 아님). ⚠️ **출처 정정(2026-06-11 워크플로 검증):** 'head 분할·seq 미분할'의 1차 근거는 ① 위 실패 텐서 shape `[131072,4,128]`(head축만 4로 줄고 seq는 131072 그대로) + ② GQA/Megatron-TP 표준 동작이다. 이전에 인용했던 `cc_calculator.py:274-291`은 head/seq/attention을 한 글자도 언급 않는 **범용 집합통신(Shard/Partial/Replicate placement 변환) 계산기**라 이 주장의 1차 출처가 아니다(grep `head`/`seq`/`attention` 0건; AllReduce는 L274, AllGather는 L291). KV head를 tp로 못 나누면(`kv_heads < tp`) KV가 **복제**돼 한 PE가 전체 head × 큰 seq attention을 떠안고 → 네이티브 tactic 없음. 실패 텐서 `[131072, 4, 128]` = qwen3_moe kv_head 4개가 한 PE에 통째로(4<8). **법칙: `kv_heads ≥ tp`(나눠떨어져 kv/PE≥1)면 131072도 tp8에서 컴파일됨** — 실측 근거: `llama-3.1-8b-tp8`(kv8=tp8, 131072 ✅), `exaone-4.0-32b-FP8`(131072 전버킷 ✅, 같은 W8fA16KV16), `qwen3-32b-fp8-tp8`(✅). 실패는 전부 kv<tp: qwen3-coder(4<8)·qwen2.5-coder-7b(4<8)·1.5b(2<8). FP8 무관(attention은 A16/KV16이라 bf16 계산, 실패 텐서 전부 bf16), embedding/tp32 무관. **모델별 해결:** kv<tp 모델(qwen3-coder 등)은 `--max-model-len ≤65536`으로 131072 버킷 제거(또는 kv가 나눠떨어지는 tp로 — qwen3-coder는 kv=4라 tp4면 131072 가능성, 미검증). kv≥tp 모델(Llama-3.3-70B kv8=tp8)은 캡 없이도 131072 빌드 가능. 단 네이티브 tactic 판정은 블랙박스라 "8 q-head/PE×131072×fp8×tp8 단일칩" 조합은 직접 관측 전(70B tp8pp4 빌드 진행 중). (2026-05-31 qwen 실패 + 2026-06-02 4-에이전트 메커니즘 재검증) |
| 컴파일 중 `failed to lower the operatorO#### (no tactic): TacticKernel kind: EinsumByDpe name: einsum_rope_k` (stage_0 QkvProjection, **symbol=2**) + 빌드가 **종료 안 되고 무한 hang**(progress 5/N 고정·CPU 발산·Ray `GetRequest::Wait()` 영구대기, SIGTERM으로만 끝남) | 번들 컴파일러 2026.2.0이 exaone4의 **seq_len=2 RoPE-K einsum**을 lower할 tactic 없음 | **EXAONE-4.0-32B-FP8 `-tp 8`(및 `-tp 4`) 빌드 시 발생 (2026-06-02 실측, 4-에이전트 적대검증).** 원인: exaone4는 **하이브리드 attention**(sliding_window=4096, layer_types LLLG)+global층 **NoPE**라 `rope_k` einsum이 sliding-mask와 fused되는데, **seq<4 강제 tokenwise 버킷**(`compiler_config.py:73-83`이 seq<4→AttentionBucket(seq,128,127) 리맵)에서 bmm이 degenerate→no tactic. qwen3·llama는 sliding fusion 없어 동일 버킷도 컴파일됨(einsum 식은 셋 다 동일, `rotary_embedding.py:196`). 128~1024 버킷은 성공, **딱 symbol=2만 실패**. no-tactic이 예외 안 던지고 다음 단계 재진입→무한 hang. **max-model-len 무관**(seq=2는 모든 캡에 존재, uncapped·65536 동일 wedge). **tp4도 ❌(~12%)** — degeneracy가 seq=2 자유차원+`Exaone4Prefill` 스케줄(model-type 키, tp 무관). **CLI 우회 없음** — seq=2는 `EXAONE_4_32B_PRESET.tokenwise_seq_lens=(2,4,…)`(`presets.py:110`) 강제, tactic-timeout/einsum-disable env·flag 0건(`--additional-model-config` whitelist 6키 거부). **prebuilt tp32가 되는 건 다른 toolchain**(artifact `furiosa_compiler_version=d19a92a2f2`≠로컬 `compiler_git_short_hash()=5c885c73ee`)이라 그런 것 — 로컬 2026.2.0은 **tp 무관 불가**. → **해결: prebuilt tp32 serve.** (Python ArtifactBuilder로 `BucketConfig(tokenwise_seq_lens` seq=2 제외`, skip_validation=True)` 또는 `CompilerConfig.compiler_config_overrides`로 tactic search 완화 가능하나 미지원·버킷셋 변경·위험.) |
| **저장(1.7) 단계** `shutil.Error: [Errno 28] No space left on device` (`builder.py:474 __preprocess_for_pipeline_save → :428 copy_param_file → shutil.copytree`) — **트레이싱·컴파일은 다 성공한 뒤** 마지막 param 복사에서만 죽음 | param 파일을 **하드링크가 아니라 `shutil.copytree`로 통째 복사**해 아티팩트 디렉터리에 또 한 벌 만듦 | **대형 bf16 모델 저장 시 디스크 함정 (2026-06-16 Qwen2.5-72B-Instruct `-tp 8` 실측).** bf16 72B param ≈ **136 GiB**. 저장 시 캐시(`~/.cache/furiosa/llm/param_files`, 136G) → 아티팩트(136G)로 **풀 복사**라 **순간 272G+ 필요**. tp8 72B는 트레이싱(89버킷)·컴파일(118유닛) 다 통과하고 **마지막 복사에서 `/home` 6G까지 차 ENOSPC**로 실패함(shard 9/31). **컴파일 실패 아님 — 순수 디스크.** **회피:** ① 저장 전 디스크 ≥ param크기+여유 확보(HF 가중치 캐시는 param 만든 뒤엔 불필요 → 삭제로 136G 회수 가능). ② **이미 같은 param 해시의 완성 아티팩트가 있으면 재빌드 말고 `cp -al`(하드링크 클론)** — 재빌드는 결정적이라 **바이트 동일**(param 해시·binary_bundle 크기 일치 실측)이므로 40분 재빌드+136G 낭비 대신 즉시 동일본 생성(디스크 0). ③ 근본적으론 `copy_param_file`을 `shutil.copytree(..., copy_function=os.link)` 로 패치하면 중복 0 (미적용, 요청 시). ⚠️ `--max-model-len`·컴파일과 무관, **모델 클수록·디스크 적을수록** 발생. |

### 8.1 9개 모델 실행에서 본 "빌드 OK · serve/요청 단계" 요약 (2026-06-05)

이번 실행은 enabled 9개 모델 벤치였습니다(단일 카드 tp8 6종 + 멀티 카드 tp32 3종). 위 8절 에러표가 **빌드/컴파일 단계** 실패라면, 아래는 **빌드는 통과했는데 serve·요청 단계에서 갈린 결과**입니다. (수치 출처: `rngd-npu/REPORT.md` 표1·표7, 2026-06-05 실행)

- **정상 serve + 측정 OK (6종).** 빌드·serve·요청 모두 정상. (⚠️ 단 `Coder-1.5B`는 serve·속도만 정상이고 **출력은 깨짐(gibberish)** — 아래 8.2. TPS는 무의미한 토큰을 빠르게 쏟아낸 값입니다.)
  - 단일 카드(tp8, 1장): `Qwen2.5-Coder-1.5B-tp8`(단일 95.5 TPS, peak 3443@c256, 가장 빠름 — ⚠️ **출력 깨짐, 8.2 참조**)·`Qwen2.5-Coder-7B-tp8`(50.3, 2225@c256)·`Qwen2.5-Coder-14B-tp8`(30.7, 1074@c256).
  - 멀티 카드(tp32, 4장): `EXAONE-4.0-32B-FP8-tp32`(30.4, 809@c256)·`Llama-3.3-70B-Instruct-tp32`(24.5, 383@c128). 이 둘은 로컬빌드 불가(8절 hang·OOM)라 **furiosa-ai prebuilt tp32 serve**로 측정한 것입니다(12절).

- **serve는 되나 극저속 — Qwen3-32B-FP8 트리오(`-tp32`/`-tp8`/`-tp8-16k`).** 셋 다 빌드 통과, serve 부팅·응답까지는 됨. 하지만 단일 TPS 측정(`tps`, c1·prompt256)이 **3종 전부 0/30 실패**라 REPORT 표2/단일TPS가 공란이고, 동시성 peak도 tp32 25@c1·tp8/16k 5@c1로 사실상 측정 불가 수준입니다(reasoning 모델 특성상 TTFT가 길어 극저속). 특히 **`Qwen3-32B-FP8-tp8`은 serve 도중 NPU `Unknown error -5`로 엔진이 종료**되고 그 뒤 요청이 400으로 떨어진 사례가 있었습니다. (참고: 빌드 자체는 12절에서 `Qwen3-32B-FP8 -tp8 --max-model-len 16384` "확정 — 실제 빌드 완료"로 기록됨 — 즉 **빌드 성공과 별개로 serve 측 가용성·성능은 보장되지 않음**.)

- **serve 실패 — `Qwen3-Coder-30B-A3B-Instruct-FP8-tp8-65k`.** 빌드는 성공하나 serve 부팅 시 네이티브 런타임에 FP8 MoE 커널이 없어 패닉(코드 1 종료). 메커니즘·근거는 12절 마지막 행에 상세 기록되어 있어 여기서는 생략합니다. (`Qwen3Moe`+`FP8` 게이트 거부, `hf_compat_next_gen.rs:367`, 2026-06-04 실측)

> 📌 위 9종은 **SWE-bench 추론까지는 9개 다 수행**됐으나, 채점이 Docker 소켓 권한거부(`PermissionError 13`)로 전건 실패해 resolved가 미채점입니다(`eval_result.json` 전부 returncode=1). 해결은 사용자를 `docker` 그룹에 추가하거나 rootless docker 사용입니다(serve/빌드와 무관한 호스트 권한 문제).

> ⚠️ **"빌드 성공 ≠ serve 성공 ≠ 빠름 ≠ 출력 정상"** — 이번 9종이 그 네 축이 다 다름을 보여줍니다. 빌드 통과(Coder 트리오·Qwen3-32B 트리오·Coder-A3B FP8) → serve 부팅 성공/실패(Coder-A3B만 패닉) → 측정 성능(Qwen3-32B 트리오만 극저속) → **출력 품질(Coder-1.5B만 깨짐, 8.2)**으로 갈립니다.

### 8.2 빌드·serve·속도 다 정상인데 출력만 깨짐 — tied word embedding (2026-06-08)

**증상.** `Qwen2.5-Coder-1.5B-tp8`이 serve·부팅·TPS는 멀쩡한데 응답이 **토큰 수프(gibberish)**입니다 (예: `0 以 Python`, `for for for`, 숫자/한자 반복). 처음 몇 토큰은 그럴듯하다가 무너집니다.

**한글 문제·채팅앱·스트리밍·샘플링 전부 아님 (재현으로 배제).** `/v1/chat/completions`에 **temperature 0(greedy)·stream=false**로 직접 curl 해도 깨집니다(영어 프롬프트도 동일). 즉 furiosa-llm 서버가 내놓는 출력 자체가 깨진 것.

**원인 = furiosa-llm 2026.2.0 컴파일 버그 (정확한 트리거 미특정).** ⚠️ 처음엔 tied word embedding을 의심했으나 **untie 재빌드로도 동일하게 깨져 기각됐습니다**(아래 "untie 시도→실패"). 아래 표는 상관일 뿐 인과가 아닙니다:

| 모델 | `tie_word_embeddings` | 출력 |
|---|---|---|
| Qwen2.5-Coder-1.5B | **True** (입력 임베딩=출력 lm_head 공유) | ❌ 깨짐 |
| Qwen2.5-0.5B | **True** | ❌ 깨짐 (SWE-bench 50건 중 47건 empty_patch, resolved 0) |
| Qwen2.5-Coder-7B / 14B | False (별도 lm_head) | ✅ 정상 |

- tie 가설이 매력적이었던 이유: 깨지는 1.5B/0.5B만 tied(True), 정상인 7B/14B는 untied(False). 원본 tied 아티팩트도 weight safetensors가 물리 텐서 `lm_head.weight` 하나에 `__metadata__: {'model.embed_tokens.weight': 'lm_head.weight'}` 별칭으로 tie를 표현 → "런타임이 이 공유 lm_head를 잘못 쓴다"는 그림이 그럴듯했습니다.
- **그러나 기각됨 (2026-06-08 실측).** lm_head를 embed 복사본으로 분리(`tie_word_embeddings=False`, 별칭 없는 진짜 별도 텐서 — 아티팩트 safetensors 3.1GB→3.57GB, `__metadata__: {}` 확인)해 재빌드·serve·greedy curl 했더니 **원본과 글자 그대로 동일하게 깨짐**(`0 以 Python`·`三四五六七`·`SSSSSSs`). tie가 원인이면 고쳐졌어야 합니다 → tie는 원인이 아니고, 2/2 vs 2/2 상관은 모델 크기와 엮인 우연.
- **남은 후보(미검증).** tie를 빼면 1.5B만의 이상치는 **`kv_heads=2`**(tp8에서 KV ×4 복제, 7B ×2·14B ×1)와 **`hidden_size=1536`**(셋 중 유일하게 작음). furiosa 2026.2.0이 이 작은 shape를 tp8로 쪼갤 때 어텐션을 잘못 컴파일하는 것으로 의심(이 문서 8절 `kv_heads<tp` 어텐션 행과 정합). 검증 실험은 **tp4 재빌드**(KV ×4→×2, hidden 1536/4=384)이나 미수행 — 사용자 결정으로 1.5B를 chat에서 제거(아래).

**시도했으나 실패한 우회 = untie 후 재빌드 (기록용).** lm_head를 embed_tokens 복사본으로 분리해 7B/14B와 같은 untied 경로를 태웠지만 위처럼 출력은 그대로 깨졌습니다. 재현 절차:

```python
# furiosa venv. lm_head를 embed 복사본으로 분리하고 tie 플래그를 끔
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", dtype=torch.bfloat16)
m.get_output_embeddings().weight = torch.nn.Parameter(m.get_input_embeddings().weight.detach().clone())
m.config.tie_word_embeddings = False
m.save_pretrained(OUT); AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct").save_pretrained(OUT)
# 이후 평소대로: furiosa-llm build OUT artifacts/qwen2.5-coder-1.5b-tp8-untied -tp 8 --max-model-len 32768
```

- untie 하면 weight가 분리돼 safetensors가 **+약 0.45GB**(lm_head 151936×1536×2byte≈467MB) 커집니다(3.1GB→3.55GB). config는 `tie_word_embeddings=False`가 되고 safetensors에 `lm_head.weight`·`model.embed_tokens.weight`가 각각 실재(별칭 없음).
- 빌드는 host AOT라 NPU 불필요(serve 중에도 가능). 자동 버킷 프리셋(`qwen2 hidden=1536 inter=8960`) 그대로 잡힙니다.
- **결과: 빌드·serve는 정상이나 출력 여전히 깨짐 → untie 무효.** 따라서 1.5B는 chat `CATALOG`/`serve_models.sh`에서 **제거**했습니다(2026-06-08, 사용자 결정). 로컬 furiosa 2026.2.0으로는 Qwen2.5-Coder-1.5B(및 같은 증상 0.5B)를 못 쓰니 코딩용은 **7B/14B**를 쓰십시오. 진짜 1.5B가 필요하면 tp4 재빌드 실험 또는 차기 SDK(2026.3+) 대기.

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
| 네이티브 compile 진입점 (파이썬→native) | `parallelize/pipeline/builder/converter.py:876`(import)·`:913`(`compile(...)`)·`:932`(except→RuntimeError) |
| 스테이지 컴파일 에러 래퍼 | `parallelize/pipeline/builder/new_pipeline_builder.py:1179`(`'Compilation failed for stage id: ...'`)·`:1184`(raise) |
| `compile()` = native PyO3 바인딩 | `furiosa/native_common/compiler/__init__.py:8`(`import_module("furiosa.native_llm_common")`)·`.pyi:36-59`(시그니처) |
| **tactic 통과/실패 판정 = native `.so`** (파이썬 아님) | `furiosa/native_runtime.cpython-…so`·`native_llm_common.cpython-…so` 내 Rust 크레이트 `tactic-solver`/`tactic-populator`/`npu-compiler-kernelize(tactic_context)`/`npu-compiler-decompose(tactic_kernel)`. 메시지 `(no tactic):`·`failed to lower the operator`·`lowering target should be AttentionKernel` (strings 확인, 소스 비공개). 파이썬은 호출+에러 재포장만 |
| tactic 튜닝 노브 (판정 아님, 설정) | `furiosa/native_torch/compiler.pyi`: `enable_tactic_pruning`·`tactic_hint`·`tactic_sorting_policy`·`TacticHintConfig`·`TacticSortingPolicy` |

---

## 12. 빌드 가능 모델 인벤토리 (현재 머신 **4장(32 PE)** + 125GB RAM + 200GB swap 기준)

> 📌 2026-05-30 RNGD 2장→4장(32 PE) 업그레이드. 이전 "2장" 기준으로 쓰인 일부 서술은 stale일 수 있음.
> 4장이면 tp=32(4장 분할) 또는 dp(1장 tp=8 아티팩트를 4장에 복제)가 가능. 단 **tp=32는 qwen3_moe 등 MoE에서 임베딩 컴파일 버그로 실패**(8절) → 1장에 들어가는 모델은 dp가 더 안전·효율적.

종합 판정 = `model_type` 지원 × architecture pp 지원 × 1·2장 적재 가능 weight 크기 ×
빌드 host RAM 가용성.

### ✅ 확정 — 실제 빌드 완료

- `Qwen/Qwen3-32B-FP8` — qwen3, preset 정확 매칭, FP8 32GB, 1장 적재.
  `-tp 8 --max-model-len 16384` + 3중 방어(Ray off + oomd off + swap 200G) 검증됨.

### ✅ 가능 — preset 매칭 + 1장 적재 OK

| 모델 | model_type | weight | 권장 명령 요지 |
|---|---|---:|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | qwen2 | 1GB | `-tp 8` 매우 가벼움 |
| `meta-llama/Llama-3.1-8B-Instruct` | llama | 16GB bf16 | `-tp 8` (1장 적재, dp로 복제). ⚠️ ~~`-pp 2`~~ 는 안 됨 — pp>1 전 모델 불가(3.1절 🔴). 완성 아티팩트 `llama-3.1-8b-inst-tp8`은 pp=1 |
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
| `meta-llama/Llama-3.3-70B-Instruct` | llama | ⚠️ **로컬 빌드 경로 없음 (2026-06-02 확정).** FP8 가중치 **67.7 GiB > 1장 47.5 GiB → `-tp 8` 불가**(serve OOM). tp16/24 미지원, **tp32 로컬빌드 미지원(4/8만)**, **`-pp` 전 모델 불가(3.1절 🔴)** — `-tp8 -pp4` 실측 시 marker `ValueError`로 사망. → **유일한 4장 경로 = furiosa-ai prebuilt `models--furiosa-ai--Llama-3.3-70B-Instruct`(bf16 tp32, len 131072) 다운로드 후 `serve`(8절 우회)**. fp8 tp32 prebuilt는 미배포 |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | qwen3_moe | preset 정확 매칭. bf16 1장 불가. **⚠️ `-tp 32` bf16 빌드는 stage_0 임베딩 컴파일에서 실패** (`embedding_table ... DramShapeGuide`, 8절 참고, 2026-05-30 실측, len 무관). **4장 단일 인스턴스 불가** — qwen3_moe는 `-pp` 미지원(`tp8×pp4` 빌드 시 `NotImplementedError: Block slicing for Qwen3MoeForCausalLM`, **2026-06-01 실측**, 3.1절)이고 `-tp 32` bf16은 임베딩 컴파일 실패(8절). 그래서 아래처럼 FP8를 1장에 올려 dp로 복제하는 게 **유일한 4장 경로**입니다. **권장: FP8 변형(`-FP8`)을 `-tp 8`로 빌드(30GB→1장 적재) 후 serve 시 `--devices`로 dp 복제 → 4장 처리량 확보, tp=32 회피.** ⚠️ **FP8 tp8도 `--max-model-len`을 반드시 ≤65536으로 줄 것** — 미지정 시 기본 262144로 잡혀 attention_size=131072 버킷에서 `no tactic`으로 실패함(2026-05-31 실측, 8절). 32768 권장(weight 29.3+KV 3.0=32.3GiB, 1칩 여유). FP8 tp8 + len≤65536은 stage_0 임베딩·stage_1 attention 모두 통과 확인(embedding 문제는 tp=8이라 발생 안 함). 🔴 **단, 빌드된 FP8 tp8 아티팩트는 serve 단계에서 실패**(2026-06-04 실측·6에이전트 적대검증 + strings 직접확인, len 65536): 아티팩트는 정상 로드되나(schema 3.0, `tp=8 pp=1 dp=1` 파싱 성공) `NativeLLMEngine` 생성(api.py:383)에서 `pyo3_runtime.PanicException: Unsupported model metadata { model_type: Qwen3Moe, task: Generate, weight: FP8, act/kv: BF16 }`로 패닉(next_gen 게이트 `furiosa-generator/src/next_gen/hf_compat_next_gen.rs:367`) → 코드 1 종료. **메커니즘 = MoE 자체나 FP8 자체가 아니라 "MoE×FP8" 조합이 런타임에 미빌드.** 네이티브 바이너리(`native_runtime.so`/`native_llm_common.so`)에는 `qwen3_moe` enum·A3B 스케줄 preset(`QWEN3_A3B_PREFILL`/`_DECODE_*`)·MoE 머신러리(`moe_router_*`, `*_tokenwise_moe_*`, `sparsify_moe`)가 모두 존재하지만, **MoE MLP 커널은 BF16짜리 둘(`blockwise_moe_w16a16`, `default_blockwise_moe_qwen3_w16a16`)뿐이고 FP8(`w8`) MoE 커널 문자열은 0건**(strings 직접 확인). dense는 동일 quant(weight=FP8/act=BF16/kv=BF16)로 serve 부팅 성공 → **FP8 자체는 지원**. 즉 FP8-weight MoE를 실행할 커널이 없어 serve 게이트가 (Qwen3Moe+FP8)을 거부. 빌드가 통과한 이유는 build측 `find_compiler_config('qwen3_moe','generate')`가 양자화 무관하게 `Qwen3_30b_a3b`를 반환하기 때문(quant-agnostic). 버전 skew 아님(artifact `furiosa_llm_version=9f92da0`=바이너리 해시 일치). **함의: 이론상 BF16-weight MoE면 serve될 여지 있으나, BF16 30B MoE는 1장 OOM·tp32 빌드버그·pp 미지원으로 빌드 자체가 막혀(8·3.1절) 2026.2.0엔 실측 경로 없음.** FP8 MoE prebuilt도 같은 커널 부재로 거부될 것. **"컴파일 통과 ≠ serve 가능."** 근본해결 = FP8 MoE 커널이 들어가는 차기 SDK(2026.3+). (`docs/COMPILING_MODELS.md` qwen3_moe=미검증/planned과 일치) |

### ⚠️ preset 부적합 — 수동 버킷 (`-pb`/`-db`) 필요

`qwen2` model_type은 preset이 0.5B용 한 개뿐이라 큰 Qwen2.5 변형들은 매칭은 되지만 버킷 부적절.
`Qwen2ForCausalLM`은 **pp 미지원**(3.1절 — dict 미등록)이라 무조건 1장 fit 해야 합니다.

| 모델 | weight (bf16) | 1장 fit? | 결론 |
|---|---:|---|---|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 3GB | ✅ | bf16 그대로 빌드+서빙 OK |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 14GB | ✅ | bf16 그대로 OK |
| `Qwen/Qwen2.5-Coder-14B-Instruct` | 28GB | ⚠️ tight (KV 여유 적음) | bf16 가능, `--max-model-len` 짧게(4K~8K) |
| `Qwen/Qwen2.5-Coder-32B-Instruct` | 64GB | ❌ | bf16 빌드 자체는 되지만 **서빙 불가** — 직접 FP8 양자화(→32GB) 후 재빌드해야 1장 서빙 |

> ⚠️ **"빌드 가능 ≠ 우리 머신서 서빙 가능"** — 빌드는 host CPU/RAM에서 AOT 컴파일이라
> HBM 체크 없이 통과합니다. 서빙 시점에 `Required PEs` 또는 HBM OOM이 뜰 수 있어요.
> bf16 32B+ 모델은 1장 서빙 불가이고, qwen2/qwen3·exaone은 **pp 미지원**(3.1절·2026-06-01 실측)이라
> 2장으로 분산도 못 합니다. → **FP8 양자화가 사실상 필수**입니다(1장 적재). (Llama 계열만 pp로 분산 가능)
> Qwen2.5-Coder/Qwen3-Coder는 Qwen이 FP8 변형을 공식 출시하지 않아서 직접 양자화 필요
> (`transformers.FineGrainedFP8Config`, `docs/COMPILING_MODELS.md` 2절).

### ⚠️ preset 없는 model_type — 항상 수동 버킷

`mistral`, `phi3`, `gpt_oss`, `qwen3_vl`, `exaone`, `exaone_moe`, `mllama4` — model_type 지원은 되지만
preset 등록 없음 → 무조건 `-pb`/`-db` 수동. 6절 가이드 참고.

### ❌ 빌드 불가

| 모델 | 이유 |
|---|---|
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | `model_type=qwen3_next` — SDK 2026.2.0 미지원 |
| 70B+ bf16 그대로 | bf16 weight(~140GB)가 host RAM(125GB)에 안 들어가 빌드 자체가 어려움 → FP8 양자화 필수 (pp 문제 아님) |
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
          ├─ 32~64GB → Llama면 pp=2(2장), 아니면 FP8 양자화로 1장
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
