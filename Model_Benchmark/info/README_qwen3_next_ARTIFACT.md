# Qwen3-Coder-Next를 "아티팩트로 빌드"한다는 것 — 정직한 정리

작성 2026-06-12 · RNGD SDK 2026.2.0 · 4×RNGD(각 47.5GiB HBM) · host 125GB RAM / 253GB 디스크
관련: [README_qwen3_next_RUN.md](README_qwen3_next_RUN.md)(실행·서빙),
[README_qwen3_next_TECH.md](README_qwen3_next_TECH.md)(커널·DPE·serve 게이트 원리)

> 이 문서는 두 가지 질문에 정직하게 답합니다.
> ① **왜 Qwen3-Coder-Next는 진짜 furiosa-llm EDF 아티팩트로 빌드되지 않는가** (정확한 차단 지점),
> ② 그래도 우리가 **만들 수 있는 self-contained host-loop 아티팩트**는 무엇이고, 어떻게 빌드·적재·실행하며,
>    왜 그것이 NPU에서 도는가.
>
> 결론을 먼저 말씀드리면: **진짜 컴파일된 EDF 아티팩트는 2026.2.0에서 불가능**하지만,
> **self-contained host-loop 아티팩트는 빌드·적재·실행이 모두 검증되어 가능**합니다.
> 즉 이 host-loop 패키징의 의미에서 Qwen3-Coder-Next-FP8은 **"아티팩트로 빌드 가능"** 합니다.

---

## 1. 용어 정리 — "아티팩트 빌드"가 이 모델에서 뜻하는 두 가지

furiosa-llm에서 보통 **아티팩트(artifact)**란 `furiosa-llm build`가 만든 디렉터리를 말합니다.
실제 아티팩트(예: `rngd-npu/artifacts/qwen3-coder-30b-a3b-inst-tp8-65k-tc/`)는 다음을 담습니다.

| 파일 | 내용 |
|---|---|
| `artifact.json` | 메타데이터(model_type, hf_configs, llm_config 등) |
| `binary_bundle.zip` | **컴파일된 EDF 파이프라인**(NPU에서 실행되는 진짜 바이너리) |
| `config.json` / `generation_config.json` | 모델·생성 설정 |
| `params-*.safetensors` | 가중치 |
| `tokenizer*` | 토크나이저 |

여기서 핵심은 **`binary_bundle.zip` 안의 컴파일된 EDF**입니다. 이게 있어야 `furiosa-llm serve`가
그래프를 통째로 NPU에 올려 돌립니다. **Qwen3-Coder-Next는 이 EDF를 만들 수 없습니다**(2장에서 정확한 이유).

그래서 우리는 두 번째 의미의 아티팩트를 만듭니다 — **host-loop 아티팩트**입니다.
컴파일된 전체 그래프 EDF 대신, **손수 작성한 NPU 커널(YAML)들 + 매니페스트 + 토크나이저/설정**을
하나의 self-contained 디렉터리로 묶고, host 추론 루프(`qcn/`)가 그것을 적재해 NPU에서 돌립니다.
**연산은 진짜 NPU에서** 돌고, **오케스트레이션 루프만 host**에 있습니다.

---

## 2. 왜 진짜 EDF 아티팩트는 불가능한가 (real-EDF 시도의 정확한 차단 지점)

`furiosa-llm build`로 `model_type=qwen3_next`의 충실한 EDF를 만들려는 시도를 끝까지 밀어붙여
**실측으로** 어디까지 되고 어디서 막히는지 확인했습니다. 추정이 아니라 라이브 컴파일 프로브로 검증한 결과입니다.

### 2-1. 되는 부분 (진짜 직렬화 가능한 EDF로 컴파일됨)

- **full-attention** 레이어 (표준 SDPA),
- **MoE**(SwiGLU 전문가),
- **embedding**,
- 그리고 **새로 증명한 사실**: DeltaNet의 **게이트 부분**(sigmoid / exp / log 같은 standalone
  elementwise 연산)도 진짜 직렬화 가능한 `CompiledGraph` EDF로 컴파일됩니다.

즉 DeltaNet 레이어의 **상당 부분**은 EDF로 떨어집니다.

### 2-2. 막는 부분 (어떤 EDF로도 컴파일 불가 — 하드 블로커)

진짜 벽은 **DeltaNet의 순환(recurrent) 외적(outer-product) 상태 업데이트**입니다.

```
state += k ⊗ delta          # 외적으로 상태 누적
out    = einsum(state, q)    # 누적 상태와 q의 수축
```

이 연산은 **어떤 EDF로도 컴파일되지 않습니다.** 컴파일러 내부에서 Rust 패닉이 납니다:

- 패닉 위치: `global-compiler/src/lib.rs:100` 의 **`UnsupportedOpError`**.
- **`allow_external_operators` 플래그는 이 연산을 우회시키지 못합니다**(no-op).
- 함께 시도한 `unlowered`/external 설정 플래그들도 이 지점에서는 효과가 없었습니다(실측).

### 2-3. 우회로도 없다 (구조적 벽 2겹)

1. **bring-your-own-EDF API가 없음.** furiosa-llm 네임스페이스 전체를 `ExternalOperator`로
   grep해도 **0건**입니다. 즉 외부에서 만든 EDF 블롭을 36개 DeltaNet 레이어 자리에 끼워 넣을
   **주입 API 자체가 없습니다.**
2. **포맷도 호환 안 됨.** 설령 블롭이 있어도, torch가 만드는 `ir.Edf`(a5 CBOR 헤더)와
   아티팩트가 요구하는 `CompiledGraph`(a6 CBOR 헤더, `'binaries'` 필드 1개 추가)는 **서로 다른
   포맷**이라 torch-컴파일 EDF는 아티팩트 로더가 거부합니다.
3. **하이브리드 번들도 미완성.** a6 `CompiledGraph` 생산자(`native_common.compiler.compile`)가
   파이썬에서 호출 가능하므로 하이브리드 번들을 **기계적으로는** 만들 수 있지만, **DeltaNet에
   넣을 유효한 블롭이 없으므로** 항상 불완전합니다. 게다가 **두 번째 런타임 벽**이 있습니다 —
   paged-KV forward 계약에 **순환 상태(recurrent-state) 슬롯이 없습니다.** append-only paged-KV는
   매 디코드 스텝 새 블록만 추가할 뿐, 상태 S의 read-modify-write를 들 수 없습니다.

### 2-4. 진짜 EDF 아티팩트가 가능하려면 (벤더 몫)

다음 세 가지가 벤더(2026.3+)에서 와야 합니다.

1. DeltaNet 순환 외적에 대한 **컴파일러 지원 / 패닉 제거**,
2. 진짜 **external-op / bring-your-own-EDF 주입 API**,
3. 런타임의 **순환 상태 슬롯**.

그 전까지는 `qcn/`의 **host 추론 루프가 올바른 접근**입니다.

> **real-EDF 한 줄 정직 결론:** **PARTIAL, 사실상 NO.** full-attn·MoE·embedding·DeltaNet 게이트
> elementwise까지는 진짜 EDF로 컴파일되지만, **DeltaNet 순환 외적 상태 업데이트가 컴파일러
> 패닉(`lib.rs:100` `UnsupportedOpError`)을 내고, 그것을 끼워 넣을 external-op API도, 들어줄 런타임
> 상태 슬롯도 없어** compiler-honest한 self-contained EDF 아티팩트는 2026.2.0에서 만들 수 없습니다.

---

## 3. 우리가 만들 수 있는 것 — self-contained host-loop 아티팩트 (✅ 가능)

진짜 EDF는 못 만들지만, **host-loop 시스템을 self-contained 아티팩트로 패키징**하는 것은
**빌드·적재·실행이 모두 검증**되었습니다.

### 3-1. `build_artifact.py` — 어떻게 빌드되는가

빌더: `qwen3-next-proj/qcn/build_artifact.py`

```bash
PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj \
/home/jun/furiosa/bin/python qcn/build_artifact.py
    [--out  <아티팩트 디렉터리>]   # 기본: rngd-npu/artifacts/qwen3-coder-next-fp8-rngd
    [--copy-weights]              # 75GB FP8 safetensors도 번들(기본은 포인터만)
    [--link]                      # 복사 대신 HF 스냅샷 심볼릭 링크
```

빌더가 하는 일:

1. **NPU 커널 수집** — host 루프가 디스패치하는 손수 작성 TacticKernel YAML 8개를
   `tk_kernels/`에서 아티팩트 `kernels/`로 복사하고, 각 커널의 바이트 수·sha256·엔진
   (EinsumByVe / EinsumByDpe)을 매니페스트에 기록합니다.
2. **설정·토크나이저 수집** — HF 스냅샷에서 `config.json`·`generation_config.json`·
   `tokenizer.json`·`tokenizer_config.json`·`vocab.json`·`merges.txt`·`chat_template.jinja`를
   아티팩트로 복사합니다.
3. **가중치 포인터** — 75GB FP8 가중치는 **기본적으로 번들하지 않고** HF 캐시 스냅샷을 가리키는
   포인터만 기록합니다(`qcn.loader.QCNWeights`가 적재 시 mmap+dequant). `--copy-weights`로 번들 가능.
4. **`artifact.json` 매니페스트 작성** — 진짜 furiosa-llm 아티팩트 모양을 **합리적인 선에서만**
   흉내 내되, **정직하게 host-loop임을 명시**합니다:
   - `runtime == "host-loop"`, `model.model_metadata.runtime_kind == "host-loop"` (진짜 아티팩트면 `"edf"`),
   - `metadata.build_blocked_reason` 에 **2장의 EDF 차단 이유**를 그대로 기록,
   - `model.kernels` 에 디스패치 순서의 NPU 커널 목록(EDF 파이프라인 대신),
   - `model.runtime_module.entry_point == "qcn.model:QCNModel"` 등 진입점·소스·env.
5. **README.md 작성** — 아티팩트 자체 설명서.

종료 코드 0, 모든 산출물 존재, 매니페스트 well-formed 함을 검증했습니다.

### 3-2. 아티팩트 디렉터리 내용물

경로: `rngd-npu/artifacts/qwen3-coder-next-fp8-rngd/`

```
artifact.json          # 매니페스트 (runtime="host-loop", model_type, hf_configs,
                       #             커널 목록, 진입점, 가중치 포인터, EDF 차단 이유)
README.md              # 아티팩트 설명서
config.json            # HF 모델 설정
generation_config.json # 생성 설정
tokenizer.json / tokenizer_config.json / vocab.json / merges.txt / chat_template.jinja
kernels/               # host 루프가 디스패치하는 NPU TacticKernel YAML 8개:
    dn_linear.yaml          (EinsumByVe matmul, baseline)
    dn_linear_dpe.yaml      (EinsumByDpe matmul, QCN_DPE=1 빠른 경로)
    dn_chunk_full.yaml      (DeltaNet chunk scan, baseline)
    dn_chunk_full_dpe2.yaml (DeltaNet chunk scan on DPE)
    dn_conv1d.yaml          (짧은 causal conv)
    dn_l2norm.yaml          (head별 q/k L2 정규화)
    dn_gnorm.yaml           (DeltaNet gated RMSNorm)
    dn_gate.yaml            (sigmoid/silu 게이팅)
run_artifact_result.json   # run_artifact.py가 실행 후 기록(생성 텍스트 + NPU 증명)
```

`--copy-weights`를 쓰지 않으면 75GB 가중치는 **번들되지 않고** HF 캐시 포인터로만 들어갑니다.

### 3-3. `run_artifact.py` — 어떻게 적재·실행하는가

러너: `qwen3-next-proj/qcn/run_artifact.py`

```bash
PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj \
RNGD_DEV=rngd:4 QCN_DPE=1 \
/home/jun/furiosa/bin/python qcn/run_artifact.py \
    --artifact /home/jun/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen3-coder-next-fp8-rngd \
    --prompt "def quicksort(arr):" --max-new 2
```

"아티팩트를 적재한다"의 정확한 의미:

1. `<artifact>/artifact.json` 매니페스트를 읽습니다.
2. host-loop 모듈들의 **커널 베이스 경로를 아티팩트의 `kernels/` 로 재지정**합니다 — 즉 NPU
   YAML을 **소스 트리가 아니라 아티팩트에서** 적재합니다(`qcn.deltanet_layer._BASE`,
   `qcn.attn_layer.TK`, `qcn.moe.BASE`를 덮어씀). 이것이 아티팩트를 compute 측면에서
   self-contained하게 만드는 핵심입니다.
3. 매니페스트의 가중치 포인터 → HF 스냅샷, 토크나이저·설정은 아티팩트 디렉터리에서 해석합니다.
4. `qcn.model.QCNModel` 을 인스턴스화하고 greedy 생성을 돌립니다.
5. **NPU 실행 증명** — DeltaNet/attn/MoE의 **CPU-fallback 카운터가 0임을 단언**합니다
   (0이면 커널이 진짜 RNGD NPU에서 돌았다는 증거; host 루프는 오케스트레이션만).

결과는 `<artifact>/run_artifact_result.json` 에 기록됩니다.

### 3-4. 진짜 NPU에서 도는가 — 검증됨 (✅)

`run_artifact.py` 실행으로 다음 체인이 **모두 검증**되었습니다:

- 매니페스트 읽기 → host-loop 모듈 3개의 커널 경로를 **아티팩트 `kernels/` 로 재지정** →
  가중치/토크나이저 해석 → `QCNModel`(48 레이어, hidden=2048, top_k=10) 인스턴스화 →
  DeltaNet 커널을 **RNGD NPU로 디스패치**(실제 hidden-state 값, **CPU-fallback 0**).
- 첫 실행은 `RNGD_DEV=rngd:2`(=npu0pe2)에서 **일시적 EBUSY**를 만났는데(serve 프로세스가
  npu0의 PE0~PE3을 점유 중이었음), **빈 PE(`rngd:4`=npu0pe4)** 로 재실행하니 해소되어 host 루프가
  on-NPU로 실행됩니다.
- 48레이어 prefill + 디코드는 host 루프 특성상 토큰당 수십 초로 느리지만(전체 수 분),
  **적재 → 커널 재지정 → 인스턴스화 → NPU 디스패치 → 텍스트 생성** 체인은 완전히 증명됩니다.

> **참고(디바이스 매핑):** 이 host 루프에서 `RNGD_DEV=rngd:N` 의 `N` 은 **npu0의 PE 인덱스**로
> 해석됩니다(예: `rngd:2` → `npu0pe2`). 다른 카드가 serve 등으로 npu0의 일부 PE를 점유 중이면
> **비어 있는 PE 번호**(예: `rngd:4`~`rngd:7`)를 골라야 EBUSY를 피합니다. 점유 상태는
> `ls -l /proc/<pid>/fd | grep npu0pe` 로 확인할 수 있습니다.

---

## 3.5. binary_bundle.zip 추출 — 진짜 a6 EDF 블롭 묶음 (✅ 분해로 컴퓨트 완성)

검증 2026-06-12(17블롭) · **갱신 2026-06-15: DeltaNet 분해로 17 → 25블롭, 컴퓨트 완성**

> **2026-06-15 업데이트 — 막혔던 3조각을 분해해 넣었습니다.** 아래 본문은 17블롭
> `partial-edf` 시점의 기록입니다. 이후 a6 불가였던 3조각(`deltanet_recurrent_step`,
> `dn_conv1d_silu`, `dn_gate`)을 **"연산 1개 = 그래프 1개"로 쪼개** 8개 a6 블롭으로
> 추가했습니다(`tk_kernels/emit_dn_split_blobs.py`). 근본 원인은 연산 자체가 아니라
> 그래프 구성이었습니다 — a6 컴파일러가 한 그래프 내 복수 contraction 패턴('conflict
> between concrete labels')이나 복수 독립 출력 서브그래프('multiple internal subgraphs')를
> 거부합니다. 통과한 Linear/SDPA/SwiGLU 처럼 단일패턴으로 쪼개면 통과합니다.
> - recurrent step → `dn_recur_decay`/`dn_recur_contract`/`dn_recur_delta`/`dn_recur_outer`/
>   `dn_recur_add` (원본과 fp64 정확·fp32 rel 2.9e-7)
> - conv1d → `dn_conv1d_shift` (host-pad + 4×shift-mul-add + SiLU, O136·constant_pad_nd 회피)
> - gate → `dn_gate_beta`=sigmoid(b) + `dn_gate_g`=-exp(A_log)·log(1+exp(a+dt)) (softplus via log(1+exp))
>
> 결과: **n_blobs 17 → 25**, kind `partial-edf` → **`edf-split (compute-complete)`**,
> `pieces_without_edf` **3 → 0**, **25/25 `CompiledGraph.deserialize` a6 검증 통과**
> (515,281,551 B ≈ 515.3 MB, sha256 `d740ea47be510459e5591d9076357c55f696aae01951aaca060b56615417a996`).
> **남은 한계는 컴파일이 아니라 deploy**입니다 — serve 런타임에 cross-step 순환상태 풀이
> 없어(paged-KV는 append-only) 이 블롭들을 체인으로 굴리려면 벤더 런타임이 필요합니다.
> 상세: `README_qwen3_next_TECH.md` §7-2, 이론은 `README_gated_deltanet_STUDY.md` §4.

(이하 17블롭 시점의 원 기록)

위 host-loop 아티팩트에는 원래 컴파일된 `binary_bundle.zip`이 없었습니다(전체 그래프 EDF가
불가능하기 때문, 2장). 하지만 **모델 안에서 컴파일이 되는 연산 조각들만큼은** 다른 furiosa-llm
모델과 **똑같은 형식의 진짜 a6 EDF 블롭으로 추출**할 수 있습니다. 그래서 이 아티팩트에는 이제
`rngd-npu/artifacts/qwen3-coder-next-fp8-rngd/binary_bundle.zip`이 실제로 들어 있습니다.

### 무엇이 들어 있나 (측정값)

| 항목 | 값 |
|---|---|
| 파일 | `binary_bundle.zip` (514,038,667 B ≈ 514.0 MB) |
| sha256 | `9c1b6743b02b79e5c4d50c72d13dddab6919633a59ade088612256470714a1dc` |
| 블롭 수 | **17개** `.edf` |
| 압축 | **ZIP_STORED**(무압축) — 30B 실제 아티팩트와 동일 |
| 이름 규칙 | `<32-hex md5>.edf`, 루트에 평면 배치, 하위 폴더 없음 — 30B와 동일 |
| 헤더 | 각 `.edf` = `furiosa.native_common.compiler` `CompiledGraph.serialize()` a6 바이트<br>(`<8B 길이> a163456466 **a6** 656e6f646573`) — 30B 실제 블롭과 바이트 단위로 동일 형식 |

> a6의 `a6`은 "top-level `binaries` 필드를 포함한 6-키 맵" = furiosa-llm 아티팩트의
> CompiledGraph 형식입니다(torch.compile이 만드는 a5 `ir.Edf`가 아님).

**들어 있는 17개 조각**(2026.2.0 컴파일러가 a6로 lower 가능한 *모든* 연산):

- Linear 프로젝션 전부: `q_proj`, `kv_proj`, `o_proj`, `in_proj_qkvz`, `in_proj_ba`,
  `dn_out_proj`, MoE `moe_gate`/`moe_up`/`moe_down`, `router_gate`, `shared_gate`,
  `lm_head_repr`, `embedding_repr` — **DPE**(matmul) 13개
- `full_attn_sdpa`(GQA 16/2, hd256), `moe_expert_swiglu`(down(silu(gate)·up)) — **DPE** 2개
- `rmsnorm`, `gated_rmsnorm`(rms·silu(gate)) — **VE** 2개

즉 모델의 **matmul / 어텐션 / MoE / norm** 연산은 전부 진짜 NPU EDF 블롭으로 추출되어 있습니다.
(lm_head/embedding은 vocab 151936이 그대로면 너무 커서, 같은 연산을 vocab 16384로 줄인 대표
블롭으로 담았습니다 — note에 `repr`로 표기.)

### 무엇이 *안* 들어갔나 — 그리고 왜 (정직하게)

3개 조각은 2026.2.0에서 a6로 컴파일이 **안 되어** EDF 블롭이 없습니다.

| 조각 | 못 만든 이유 |
|---|---|
| `dn_conv1d_silu` | DeltaNet causal conv1d. 내부 op **O136이 컴파일러 미지원** |
| `dn_gate` | `aten::log1p`를 a6 프로듀서가 **import 못 함** |
| `deltanet_recurrent_step` | **선형 어텐션 순환 외적**(`state += k⊗delta; out=einsum(state,q)`). 컴파일러가 `'conflict between concrete labels'`로 패닉(`global-compiler/src/lib.rs:100`) |

이 **DeltaNet 순환 스텝 + 레이어 간 오케스트레이션**은 그대로 **host 루프**로 남습니다.
이유는 두 겹입니다 — ① 순환 자체가 컴파일이 안 되고, ② EDF 런타임에 토큰 사이로 DeltaNet
running state를 이어줄 **recurrent-state 슬롯이 없습니다**. 그래서 `artifact.json`의 `runtime`은
여전히 `host-loop`입니다.

### 정직한 위치 설정

- 이것은 **다른 모델들과 똑같은 진짜 binary_bundle.zip**입니다 — 같은 a6 형식, 같은 ZIP 레이아웃,
  matmul/어텐션/MoE/norm/gate 연산을 진짜 NPU EDF로 담은 묶음.
- 하지만 이것은 **`furiosa-llm serve`가 통째로 적재해 돌리는 완결 파이프라인은 아닙니다.**
  DeltaNet 순환 EDF + 런타임 상태 슬롯이 빠져 있고, 그 둘은 벤더(2026.3+) 영역입니다.
- `artifact.json` → `metadata.binary_bundle_kind = "partial-edf"`로 이 점을 명시합니다.

### 검증 방법 (재현 가능)

각 블롭을 zip에서 꺼내 **실제 SDK 로더**로 역직렬화했습니다.

```python
from furiosa.native_common.compiler import CompiledGraph
cg = CompiledGraph.deserialize(blob, tag)   # tag = 블롭 해시(로더가 쓰는 값), 실패 시 "" 폴백
assert cg.is_edf() is True
```

결과(2026-06-12 독립 재검증, zip을 새로 열어서):

- 17/17 역직렬화 성공, **전부 `is_edf()==True`**
- 17/17 파일명이 블롭 내용의 md5와 일치(content-addressed, 30B와 동일)
- 헤더 17/17이 30B 실제 블롭과 동일한 `a163456466 a6 656e6f646573` 마커
- 30B 실제 블롭(샘플 6개)도 같은 로더 경로로 역직렬화 성공 → 같은 형식 교차 확인
- `binary_bundle_manifest.json`의 17개 항목 ↔ zip 17개 엔트리 해시·바이트 완전 일치
- `artifact.json`의 `n_blobs`(17)·`zip_bytes`·`zip_sha256`·`pieces_with_edf`(17) 모두 zip과 일치

### 다시 만들려면

```bash
PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj RNGD_DEV=rngd:4 \
  /home/jun/furiosa/bin/python qwen3-next-proj/qcn/build_artifact.py --emit-edf
```

`--emit-edf`는 ① `tk_kernels/compile_edf_blobs.py`로 a6 EDF 블롭을 컴파일하고(NPU 필요, 느림),
② `tk_kernels/pack_edf_bundle.py`로 ZIP_STORED 묶음 + 라운드트립 검증 + 매니페스트를 만든 뒤,
③ `artifact.json`의 `binary_bundle` 섹션을 매니페스트로 채웁니다. `--emit-edf` 없이 빌드하면
이미 있는 `binary_bundle.zip`을 그대로 참조합니다.

---

## 4. 최종 정리

| 질문 | 답 |
|---|---|
| 진짜 furiosa-llm **EDF 아티팩트** 빌드 | **불가(deploy)** — 컴퓨트는 a6로 다 컴파일되지만(아래), serve 런타임에 cross-step 순환상태 풀·sub-op 네이티브 체이닝이 없음 (벤더 2026.3+ 필요) |
| self-contained **host-loop 아티팩트** 빌드 | **가능·검증됨** — 종료 0, 산출물 완비, 매니페스트 정직 |
| **binary_bundle.zip** 추출 | **가능·검증됨 (컴퓨트 완성)** — 2026-06-15 DeltaNet 순환/conv/gate를 "연산 1개=그래프 1개"로 분해해 추가 → **25/25 a6 `is_edf` 검증, `pieces_without_edf` 0**, kind `edf-split (compute-complete)`. 30B와 동일 형식 |
| 아티팩트 **적재·실행** | **검증됨** — 커널을 아티팩트에서 적재, NPU 디스패치, CPU-fallback 0 |
| 연산이 **NPU에서** 도는가 | **예** — host는 오케스트레이션만, token-mixer matmul은 RNGD NPU |
| 결론 | host-loop 패키징의 의미에서 **Qwen3-Coder-Next-FP8은 "아티팩트로 빌드 가능"** |

핵심 산출물:

- 빌더: `qwen3-next-proj/qcn/build_artifact.py`
- 러너: `qwen3-next-proj/qcn/run_artifact.py`
- 아티팩트: `rngd-npu/artifacts/qwen3-coder-next-fp8-rngd/`
- 깊은 원리(커널 DSL·DPE·serve 게이트): [README_qwen3_next_TECH.md](README_qwen3_next_TECH.md)
- 실행·서빙 실전: [README_qwen3_next_RUN.md](README_qwen3_next_RUN.md)
