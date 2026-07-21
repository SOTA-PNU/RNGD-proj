# Qwen3-Coder-Next-FP8 RNGD — 빌드·컴파일·서빙 실전 가이드

작성 2026-06-15. 이 문서는 **실제로 따라 할 수 있는 명령 중심**의 가이드입니다.
원리·증명·SDK 해부는 자매 문서를 참고하세요:
- 기술 전말(빌드벽·DPE·NPU 증명·DeltaNet 분해): `README_qwen3_next_TECH.md`
- 아티팩트 구조·binary_bundle: `README_qwen3_next_ARTIFACT.md`
- 이론(gated DeltaNet/attention): `README_gated_deltanet_STUDY.md`
- 변경 이력: `README_all_change.md`

---

## 0. 한눈에 — 무엇이 되고, 무엇이 벤더 몫인가

| 경로 | 상태 | 비고 |
|---|---|---|
| 컴퓨트 조각 a6 EDF 컴파일(분해형) | ✅ 됨 | `compile_edf_blobs.py`+`emit_dn_split_blobs.py`, 25/25 a6 |
| `binary_bundle.zip` 패키징 | ✅ 됨 | `pack_edf_bundle.py`, 30B와 동일 형식 |
| self-contained **host-loop 아티팩트** 빌드/실행 | ✅ 됨 | `build_artifact.py`/`run_artifact.py` |
| **host-loop OpenAI 서버**(단일/멀티카드) | ✅ 됨 | `qcn/serve.py`·`qcn/serve_mc.py` |
| **furiosa-llm serve 서버코드 + 우리 엔진** | ✅ 됨(실증) | `qcn/furiosa_serve_adapter.py` |
| `furiosa-llm build` 로 **컴퓨트 컴파일** | ⚠️ 가능(Python partitioner 추가 필요) | 막는 건 컴파일러가 아니라 partitioner coloring |
| `furiosa-llm build` 로 **정식 serve-able 단일 아티팩트** | ❌ 벤더 | KV 계약 + host 사전계산 + serde 게이트 |
| 정식 `furiosa-llm serve <artifact>` CLI 그대로 | ❌ 벤더 | serde `ModelType` 게이트 + native 순환상태 풀 없음 |

핵심 요약: **DeltaNet 컴퓨트는 "연산 1개 = 그래프 1개"로 쪼개면 전부 NPU 컴파일된다.**
막히는 건 컴파일이 아니라 **배포(autoregressive serve)의 cross-step 순환상태 관리**이고,
그건 host 추론 루프(우리가 보유)로 풀거나 벤더 런타임(2026.3+)을 기다린다.

---

## 1. 사전 준비

```bash
# 환경
PY=/home/jun/furiosa/bin/python                     # furiosa SDK 파이썬
PROJ=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj
ART=/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen3-coder-next-fp8-rngd

# NPU 상태 확인 (4장, 유휴면 0.00 GiB / ps 비어있음)
furiosa-smi status        # per-core util + HBM
furiosa-smi ps            # 어떤 PID가 어떤 core를 점유 중인지

# 모델 가중치 (이미 받아져 있음, 75GB / 40 shard)
ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next-FP8/snapshots/*/
```

**중요한 두 환경변수**
- `RNGD_DEV=rngd:N` — **글로벌 PE 인덱스**(카드 번호 아님). 이 머신은 4카드×8PE=**32 PE**: `0-7=npu0, 8-15=npu1, 16-23=npu2, 24-31=npu3` (실측: `rngd:9`→`furiosa-smi ps`에 `npu1:1`). npu0 의 0-3 은 자주 바쁘니 **빈 PE**(예: npu0 의 4-7, 또는 다른 카드 8/16/24)를 `furiosa-smi ps` 로 확인해 쓰세요.
- `QCN_DPE=1` — DPE(systolic matmul) 사용. prefill 4.69×·decode 1.59× 빠름(bf16 ~0.23% rel). 정밀 f32 검증이 필요하면 `QCN_DPE=0`. **모델 import 전에 설정해야** attn/moe/deltanet 분기에 반영됨(serve 파일들은 자동 처리).

---

## 2. 빌드 & 컴파일

### 2.1 컴퓨트 조각을 a6 EDF 블롭으로 컴파일

모델의 각 연산을 furiosa 정식 컴파일러(`furiosa.native_common.compiler.compile(..., target_ir="edf")`)로
a6 EDF 블롭으로 낮춥니다. **원리: 연산 1개 = 그래프 1개**. 한 그래프에 여러 contraction
패턴이나 복수 출력이 섞이면 a6 가 거부(`conflict between concrete labels` / `multiple
internal subgraphs`)하므로, DeltaNet 순환/conv/gate 도 단일 연산으로 쪼개 컴파일합니다.

```bash
# (1) 기본 컴퓨트 조각 (Linear 투영, SDPA, MoE SwiGLU, RMSNorm, lm_head/embedding 대표차원)
PYTHONPATH=$PROJ RNGD_DEV=rngd:4 $PY $PROJ/tk_kernels/compile_edf_blobs.py

# (2) DeltaNet 분해 조각 추가 (recurrent step 5조각 + conv-shift + gate beta/g)
PYTHONPATH=$PROJ RNGD_DEV=rngd:5 $PY $PROJ/tk_kernels/emit_dn_split_blobs.py
```

산출: `$ART/_edf_blobs/<md5>.edf` (a6 헤더 `a163456466 a6 656e6f646573`) + `_MASTER_summary.json`.

**DeltaNet 순환 step 분해(수치적으로 원본과 fp64 정확):**
```
state = state * α            # dn_recur_decay   (per-head scalar 감쇠)
kv    = k @ state            # dn_recur_contract(bmm 축소)
delta = (v - kv) * β         # dn_recur_delta   (elementwise)
state = state + k ⊗ delta    # dn_recur_outer(bmm 외적) + dn_recur_add
out   = q @ state            # dn_recur_contract(재사용)
```
- conv1d 단축 conv → `dn_conv1d_shift`: host 가 좌측 K-1 zero-pad → NPU 는 4회 shift-mul-add + SiLU (미지원 Conv1d `O136`·`constant_pad_nd` 회피)
- gate → `dn_gate_beta`=sigmoid(b) + `dn_gate_g`=-exp(A_log)·log(1+exp(a+dt)) (softplus 의 `log1p` 회피)

### 2.2 binary_bundle.zip 패키징 + 검증

`_edf_blobs/*.edf` 전체를 30B 모델과 동일한 flat ZIP_STORED 묶음으로 만들고,
각 블롭을 `CompiledGraph.deserialize` 로 라운드트립 검증합니다.

```bash
PYTHONPATH=$PROJ RNGD_DEV=rngd:5 $PY $PROJ/tk_kernels/pack_edf_bundle.py
# -> $ART/binary_bundle.zip (현재 25블롭, 515MB) + binary_bundle_manifest.json
#    "25/25 blobs deserialize as valid a6 CompiledGraph" 출력 확인
```

### 2.3 self-contained host-loop 아티팩트 빌드 & 실행

가중치(hf 캐시 포인터)·커널·config·tokenizer·entry_point 를 묶은 자체 완결형 아티팩트를 만듭니다.

```bash
# 빌드 (--emit-edf 면 compile_edf_blobs + emit_dn_split_blobs + pack 까지 실행해
#  25블롭 compute-complete binary_bundle.zip 재생성; RNGD_DEV 기본 rngd:4)
PYTHONPATH=$PROJ $PY $PROJ/qcn/build_artifact.py --out $ART --emit-edf
#   옵션: --link(커널 심볼릭) / --copy-weights(가중치 복사, 기본은 hf 캐시 포인터)

# 실행 (아티팩트에서 커널 로드 → NPU 디스패치)
PYTHONPATH=$PROJ RNGD_DEV=rngd:4 QCN_DPE=1 $PY $PROJ/qcn/run_artifact.py \
    --artifact $ART --prompt "def quicksort(arr):" --max-new 3
#   --chat 면 chat 템플릿 적용
```
아티팩트 구조: `artifact.json`(runtime=host-loop, entry_point `qcn.model:QCNModel`),
`kernels/`, `config.json`, `tokenizer*`, `binary_bundle.zip`(+manifest). 상세 ARTIFACT 문서.

### 2.4 furiosa-llm build 정식 경로 — 현재 막히는 지점과 뚫는 법

`furiosa-llm build <model> <out>` 는 내부적으로 **색칠된 서브모듈마다 `compile()` 1회**
(converter.py:913-928 의 `compile()` 호출; 산출 그래프가 단일임을 확인하는
`assert len(compiled.graphs)==1` 은 converter.py:1052)를 호출합니다 — 우리가 위에서 쓴
그 컴파일러와 동일합니다. 그런데 qwen3_next 에서 막히는 진짜 이유는 컴파일러가 아니라
**partitioner 의 거친 coloring** 입니다:

- `KernelwisePartitioner` 는 레이어를 3색(before_attn/attn/after_attn)으로만 나누고,
  `attn` 색을 `*.self_attn.attn` 경로에만 시딩합니다(block_slicer.py:1013-1016).
- DeltaNet 본체를 `self_attn.attn` 으로 재배치해 두면 **순환 전체가 한 색 = 한 서브모듈 =
  한 compile() 호출**로 뭉쳐, 그 멀티패턴 그래프를 a6 가 거부합니다.

**뚫는 법(Python, 벤더 불필요):** 순환을 §2.1 처럼 op 별 leaf 모듈로 쪼개 **각 op 이 별도
색/서브모듈**이 되게 하면(커스텀 partitioner 또는 아키텍처 재구성), 빌드의 per-submodule
compile() 도 통과합니다(§2.1 의 25/25 가 같은 compile() 산출물이라는 게 증거).

**그래도 정식 serve-able 단일 아티팩트는 안 됩니다(2개는 깊은 Python, 2개는 벤더):**
- KV 계약이 append-only (K,V) 동형쌍만 — 순환/conv 상태 슬롯 없음 (specs/inputs.py:61-74)
- 청크 tri-inverse(생략 시 발산)·cumsum·softplus 를 스텝마다 host 사전계산 (알고리즘적 → host-loop 강제)
- serde `ModelType` enum 에 `qwen3_next` 없음 + hf_compat 게이트 + 런타임 순환상태 풀 없음 (**벤더 .so**)

⚠️ 빌드 실험 시 주의(이전 세션 실측): 우리가 가역 수정한 SDK 파일(`transform.py`,
`hf_utils.py`, `qwen3_next.py` 등)을 쓰면 **그래프 캐시 `~/.cache/furiosa/llm/graphmodules/*Qwen3Next*`
를 수정 후 반드시 삭제**해야 합니다. 또 `hf_configs` 의 `layer_types` 에 `linear_attention`
이 들어가면 Rust 패닉이 납니다.

---

## 3. 서빙 (4가지 경로)

### 3.1 우리 단일카드 OpenAI 서버 — `qcn/serve.py`

가장 단순·안정. 요청 직렬화(단일 NPU). greedy(temperature=0)만.

```bash
PYTHONPATH=$PROJ RNGD_DEV=rngd:2 $PY $PROJ/qcn/serve.py            # 포트 8900
# 또는: PYTHONPATH=$PROJ RNGD_DEV=rngd:2 $PY -m uvicorn qcn.serve:app --host 0.0.0.0 --port 8900
```
엔드포인트: `GET /health`, `GET /v1/models`, `POST /v1/completions`, `POST /v1/chat/completions`.

```bash
# 호출 예시
curl -s localhost:8900/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"qwen3-coder-next-fp8-rngd",
  "messages":[{"role":"user","content":"Write a Python factorial function."}],
  "max_tokens":32, "temperature":0}'
```

### 3.2 우리 멀티카드 스트리밍 서버 — `qcn/serve_mc.py`

카드별 워커(spawn) + 요청 큐 + SSE 스트리밍 + 샘플링(top_p/temperature).

```bash
PYTHONPATH=$PROJ $PY $PROJ/qcn/serve_mc.py
#   환경변수:
#     QCN_CARDS  사용할 카드 수 (기본: furiosa-smi 감지, 최대 4)
#     QCN_DEVS   사용할 PE 인덱스 직접 지정 (있으면 QCN_CARDS 무시).
#                ⚠️ 값은 글로벌 PE 인덱스(rngd:N) → "0,1,2,3" 은 모두 npu0 의 4 PE(한 카드)에
#                몰립니다. 물리 4카드에 1 PE씩 분산하려면 "0,8,16,24" 처럼 카드당 1개를 지정.
#     QCN_DPE    1(기본 빠름)/0(f32 정확)
#     PORT       8900
#     QCN_STALL_S 토큰 무응답 watchdog 초 (기본 600)
```
메커니즘: serve_mc 는 워커별 CPU 스레드를 `QCN_THREADS`(=코어/N)로 캡해 워커 간 경합을
줄이도록 설계돼 있습니다(serve_mc.py 주석). 의도는 ~N× 처리량입니다.

⚠️ **처리량 주의(실측):** 그러나 80B host-loop 에서는 **동시 처리량 이득이 제한적**이었습니다
(2동시 ≈ 2× 시간). 요청당 host 작업(레이어별 FP8 역양자화 + torch.compile 글루, ~40코어)이
**host CPU/메모리대역을 포화**시켜, 스레드 캡으로도 2워커×캡 ≈ 전체 코어가 되기 때문입니다
(병목이 NPU 가 아니라 host). 진짜 선형 스케일은 pipeline-parallel(카드별 레이어 1/N 분산)이
필요하며 아직 미구현입니다. — 즉 serve_mc 는 기능(다카드·스트리밍)은 되지만 throughput
스케일은 이 host-loop 구조의 한계입니다.

### 3.3 furiosa-llm serve 서버코드를 우리 엔진으로 구동 — `qcn/furiosa_serve_adapter.py` ✅실증

furiosa-llm 의 Python serve 층(`AsyncLLMEngine`)은 `llm.engine` 을
`Union[NativeLLMEngine, FakeNativeLLMEngine]` 로 **덕타이핑**만 합니다(api.py:94). 그래서
네이티브 엔진과 같은 인터페이스를 가진 Python 엔진 `HostLoopEngine` 을 끼우면, furiosa-llm
자신의 `AsyncLLMEngine.generate` → `NativeOutputConverter` 가 우리 host 루프를 구동합니다.
**cross-step 순환상태 풀 = 요청별 host `state_cache`, sub-op 체이닝 = host 루프** — 이걸
네이티브 .so 가 아니라 Python serve 층에 구현한 셈입니다.

```python
import asyncio
from qcn.model import QCNModel
from qcn.furiosa_serve_adapter import build_async_engine
from furiosa_llm import SamplingParams

async def main():
    m = QCNModel()
    engine = build_async_engine(m)        # furiosa_llm.llm_engine.AsyncLLMEngine (진짜)
    async for out in engine.generate("def add(a, b):",
                                     SamplingParams(max_tokens=8, temperature=0.0), "req-1"):
        print(out.outputs[0].text)        # furiosa_llm.outputs.RequestOutput 스트리밍

asyncio.run(main())
```
실증(2026-06-15): `serve engine = furiosa_llm.llm_engine.AsyncLLMEngine`, `injected =
qcn.furiosa_serve_adapter.HostLoopEngine`, `RequestOutput` 스트리밍 성공, CPU 폴백 0,
all_on_npu=True.

**완전한 OpenAI HTTP 경로(`/v1/chat/completions`)도 가능:** `OpenAIServingChat` 이
`AsyncLLMEngine.from_llm(llm)` 을 호출(serving_chat.py:94)하므로, 아래 속성을 가진 **LLM 스텁**을
만들어 `OpenAIServingChat(stub, ...)` 에 넘기면 furiosa-llm 의 OpenAI 서버 핸들러가 우리 모델로
응답합니다:
- `.engine`(=HostLoopEngine), `.tokenizer`, `.prompt_max_seq_len`, `.max_seq_len_to_capture`
- `.model_metadata.task`(="generate"), `.model_metadata.trust_remote_code` (serving_chat.py:103)
- `.model_config` (serving_chat.py:97,102 에서 읽음 — 빠뜨리면 생성자에서 AttributeError)

### 3.4 정식 `furiosa-llm serve <artifact>` CLI — 왜 안 되나 + 인접 대안

literal CLI 는 `LLM._init_from_artifact`(api.py:383)에서 아티팩트로 `NativeLLMEngine`(컴파일된
Rust .so)을 생성하고, serde `ModelType` enum 이 `qwen3_next` 를 거부합니다(.so 에 문자열 0건).
→ **그대로는 불가(벤더 몫).**

인접 대안: 이미 빌드되는 MoE 코더(예: Qwen3-Coder-30B-A3B)를 **마스커레이드**(artifact.json
model_type → `qwen3`)로 정식 serve 하면 4장 dp ~1036 tok/s 가 납니다(DeltaNet 무관 경로).
도구 `qwen3-next-proj/masquerade_artifact.py`. 상세 TECH §2.

---

## 4. 성능 & 옵션

| 항목 | 값(실측) | 메모 |
|---|---|---|
| prefill (DPE, 20토큰) | ~77~143s | 프롬프트 길이·PE 점유 상태에 따라 |
| decode | ~35~44 s/tok | host-bound (FP8 역양자화+torch 글루) |
| DPE vs VE | prefill 4.69× / decode 1.59× | `QCN_DPE=1` 기본, bf16 ~0.23% rel |
| 정밀 검증 | `QCN_DPE=0` | f32, torch와 ~1e-7 |

- **속도 우선** → `QCN_DPE=1`(serve 기본). **정확도 검증** → `QCN_DPE=0`.
- decode 가 host-bound 이라 멀티카드 data-parallel 로는 안 빨라집니다(§3.2). 더 빠르게 = host
  사전계산 최적화 / pipeline-parallel / 벤더 serve.

---

## 5. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `furiosa::dfg only runs on CPU device` | dynamo 재컴파일 한도 초과 → CPU 폴백. model.py 가 `recompile_limit` 등을 크게 올림(이미 적용). 커스텀 코드면 동일 설정 필요. |
| `apply_chat_template` AttributeError | 일부 transformers 가 BatchEncoding 반환 → `ids["input_ids"]` 추출(이미 처리됨). |
| DPE 그래프가 garbage | DPE per-graph **2개 cap**(3개+면 조용히 오컴파일). DeltaNet 스캔은 `dn_chunk_full_dpe2.yaml`로 2개만 DPE. 출력축 O=1 거부 → 32배수 pad. |
| 서버가 포트 점유/응답 없음 | 옛 serve 프로세스 종료 + 포트 해제 후 재시작. `furiosa-smi ps` 로 PE 점유 확인. |
| 빌드 후 결과 안 바뀜 | 그래프 캐시 `~/.cache/furiosa/llm/graphmodules/*Qwen3Next*` 삭제. |
| PE 점유 충돌 | `RNGD_DEV` 를 빈 PE(4~7)로. `furiosa-smi ps` 확인. |

---

## 6. 무엇이 벤더(2026.3+) 몫인가 (정직한 경계)

우리가 **Python 으로** 할 수 있는 것: 컴퓨트 a6 컴파일(분해), binary_bundle, host-loop
아티팩트/서버, furiosa-llm serve 서버코드에 엔진 주입, (원하면) 빌드용 커스텀 partitioner.

**오직 벤더만** 할 수 있는 것(컴파일된 .so, 소스 없음):
1. serde `ModelType::Qwen3Next` enum 변형 + hf_compat 프리셋 (native_runtime.so)
2. 네이티브 linear-attention/Gated-DeltaNet 커널 또는 carried-state Loop 노드 (host 사전계산 제거)
3. 런타임 **cross-step 순환상태 풀**(read-modify-write) — paged-KV 는 append-only 라, 이게 없으면
   정식 `furiosa-llm serve` 로 autoregressive DeltaNet 디코드를 못 굴립니다.

즉 **컴파일은 우리 손으로 끝낼 수 있고, 정식 native serve 만 벤더 게이트**입니다. 그동안은
host 추론 루프(§3.1~3.3)가 완전한 대안입니다.

---

## 7. 파일 맵

```
qwen3-next-proj/
  qcn/
    model.py                  # 48레이어 host 추론 루프 (prefill/decode/generate/generate_stream)
    loader.py                 # safetensors mmap + FP8 blockwise dequant
    deltanet_layer*.py / attn_layer.py / moe.py   # 토큰 믹서 (NPU 커널 디스패치)
    generate.py               # CLI 생성 (--prompt --max-new --chat --out)
    serve.py                  # 단일카드 OpenAI 서버
    serve_mc.py               # 멀티카드 스트리밍 서버
    furiosa_serve_adapter.py  # ★ furiosa-llm AsyncLLMEngine 에 끼우는 HostLoopEngine
    build_artifact.py         # self-contained 아티팩트 빌드 (--emit-edf)
    run_artifact.py           # 아티팩트 적재·실행 (--artifact)
  tk_kernels/
    compile_edf_blobs.py      # 기본 컴퓨트 → a6 EDF
    emit_dn_split_blobs.py    # ★ DeltaNet 분해 → a6 EDF
    pack_edf_bundle.py        # binary_bundle.zip 패키징 + 검증
    dn_*.yaml                 # 손수 작성 NPU 커널 (TacticKernelModule)
rngd-npu/artifacts/qwen3-coder-next-fp8-rngd/   # 빌드 산출 아티팩트
```
