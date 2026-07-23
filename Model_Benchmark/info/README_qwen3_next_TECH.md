# qwen3_next(Gated DeltaNet)를 RNGD에서 돌리기 — 통합 기술 보고서

작성 2026-06-11 · RNGD SDK furiosa-llm 2026.2.0 · npu-compiler(npu-tools git `3f23a71`)
· RNGD 4장(각 47.5GiB HBM, 펌웨어 2026.2.1) · host 125GB RAM / 253GB 디스크
변경 이력 전체: [README_all_change.md](README_all_change.md)

> 이 문서는 세 갈래의 연구를 하나의 이야기로 합칩니다:
> ① **왜 furiosa-llm 의 표준 빌드 경로로는 qwen3_next 를 못 굽는가**(빌드 계단 ①~⑥ +
> serve 게이트 radare2 분석), ② **그럼에도 `furiosa.torch.TacticKernelModule`로 NPU 커널을
> 손수 작성해 Gated DeltaNet 컴퓨트를 전부 NPU에서 실행**(HF 레퍼런스와 ~1e-6~1e-7 일치),
> ③ **DPE(systolic matmul) 엔진을 역설계해 가속**하고 **실제 80B 모델을 host 추론 루프로
> end-to-end 코드 생성 + OpenAI 호환 서빙**까지 성공한 과정입니다.
> 관련 세부: [README_qwen3_next_RUN.md](README_qwen3_next_RUN.md)(full-model 청사진),
> [ALL_about_build_serve.md](ALL_about_build_serve.md)(build/serve 일반).

---

## 0. 한눈 결론

| 경로 | 가능? | 핵심 이유 (전부 실측) |
|---|---|---|
| furiosa-llm 표준 **빌드**(qwen3_next 충실) | ❌ | 빌드측 열린 Python 단계(게이트 우회·TP 분할·노드복제)는 자력 통과하나, ⑥ `primitive→kernelized` 가 DeltaNet 의 standalone elementwise/스캔 tactic 을 못 붙임. **단 이 진단은 "FX→kernelize 경로 한정"** — TK-graph 직접작성은 통과(아래) |
| furiosa-llm 표준 **serve**(qwen3_next 라벨) | ❌ | serde enum `ModelType` 에 `qwen3_next` 없음(2겹 게이트) → "unknown variant" |
| **위장(masquerade)** 으로 표준 MoE serve | ✅ | model_type 문자열만 바꾸면 EDF 그대로 실행. Qwen3-Coder-30B-A3B-Inst-FP8 부활, 4장 dp **1036 tok/s** |
| radare2 로 컴파일러/게이트에 커널·코드 주입 | ❌ | 닫힌 정적링크 Rust, phf 정적테이블, 동적 재배치 fat-pointer. 길이-우선 serde 매처라 한 군데 패치론 안 뚫림 |
| **`TacticKernelModule`(DFG DSL) 로 NPU 커널 손수 작성** | ✅ (컴퓨트) | ⭐ Rust 소스 불필요. DeltaNet 전 컴퓨트(게이트·contraction·rank-1·conv1d·l2norm·gnorm·청크스캔) NPU 실행, HF ~1e-6~1e-7 |
| 순환(recurrence) 처리 | ✅ (우회) | host-loop(상태 스레딩) 또는 그래프 내 언롤(단일 EDF). 네이티브 `Loop` 노드만 벤더-lock |
| **EinsumByDpe**(systolic matmul) 가속 | ✅ | 역설계 성공, prefill **4.69×**·decode **1.59×**. per-graph 2-DPE cap 등 제약 있음 |
| **실제 80B 모델 host 추론 루프 + OpenAI 서빙** | ✅ | `qcn/` 으로 end-to-end 코드 생성 성공(quicksort 정확). 컴퓨트 CPU폴백 0 |
| furiosa-llm **serve 안** 네이티브 통합(decode cross-step 상태풀) | ❌ | append-only paged-KV 가 DeltaNet read-modify-write 상태 못 들음. **벤더(2026.3+) 전용** |

**한 문장 요약:** "컴퓨트 벽은 완전히 뚫렸다(손수 NPU 커널 + DPE 가속으로 80B 실모델
end-to-end). 남은 단 하나의 벤더-lock 은 furiosa-llm serve 런타임의 디코드-스텝 간 순환상태
풀뿐이고, 그건 host 추론 루프로 우회해 이미 서빙 중이다."

---

## 1. qwen3_next 구조 + 왜 표준 빌드가 막히나

### 1-1. Qwen3-Coder-Next 모델 특성 (HF 실측)

| 항목 | 값 (feasibility 문서 80B) | 값 (FP8 실모델, blueprint) |
|---|---|---|
| 파라미터 / 크기 | 79.7B / BF16 159.4GB (40 safetensors) | 80.4GB FP8 (40 shard), 다운로드 실측 75GB |
| model_type / arch | `qwen3_next` / `Qwen3NextForCausalLM` | 동일 |
| 레이어 | 48 = **36 Gated DeltaNet + 12 gated full attention**(`full_attention_interval=4`), 매 레이어 MoE | 동일 |
| hidden / vocab | — | 2048 / 151936 |
| DeltaNet | head_dim 256, q·k norm | key heads **16**, value heads **32**(n_rep=2), key/value head_dim **128**, conv kernel **4** |
| full attention | 16 heads / 2 KV heads / head_dim 256 / partial rotary **0.25** / q·k norm | 16 heads, 2 kv heads(GQA), head_dim 256 |
| MoE | 512 experts, top-10, moe_inter 512, shared expert 512, 전 레이어 | 동일 |
| 양자화 | BF16 | **FP8 blockwise**(weight_block_size **[128,128]**, dynamic act); **lm_head·embed_tokens 비양자화** |
| 컨텍스트 | 262144 | — |

DeltaNet 레이어는 토큰마다 갱신되는 **고정 크기 순환 상태**(레이어당 recurrent
`(num_v_heads, head_k, head_v)` + causal-conv 상태 `(conv_dim, kernel-1)`)를 유지해야
자기회귀 디코딩이 성립합니다. 이는 트랜스포머의 append-only KV 캐시와 **접근 패턴이
근본적으로 다릅니다**(read-modify-write vs append).

### 1-2. Gated DeltaNet 구조 공부 (qwen3_next / qwen3.5 / qwen3.6 공통)

**DeltaNet = softmax 어텐션을 "gated delta rule" 선형 순환으로 대체한 토큰 믹서.**
레이어당 고정 크기 행렬 상태 `S ∈ R^(d_k×d_v)` 를 토큰마다 갱신합니다.
(출처: `transformers/models/qwen3_next/modeling_qwen3_next.py`)

두 가지 수학적으로 동등한 실행 형태:

- **순환(recurrent) 형태**(`torch_recurrent_gated_delta_rule`, L547-586) — 디코드용, 토큰당 1스텝:
  ```
  state = state * g_t                       # 게이트 감쇠 (elementwise)
  kv_mem = (state * k_t).sum(-2)            # = Sᵀk  (matvec)
  delta  = (v_t - kv_mem) * beta_t          # elementwise
  state  = state + k_t ⊗ delta              # rank-1 외적 갱신
  out_t  = (state * q_t).sum(-2)            # = Sᵀq  (matvec)
  ```
- **청크(chunked) 형태**(`torch_chunk_gated_delta_rule`, L467-544) — 프리필용, 대부분 bmm/matmul.

**연산을 종류별로 분해**(컴파일 가능성의 핵심):

| 종류 | 연산 | 표준 컴파일러 처리 |
|---|---|---|
| (a) matmul/contraction | in_proj·out_proj, Sᵀk·Sᵀq, 청크 어텐션곱 | ✅ 커널화 가능(qwen3_moe 가 되는 이유) |
| (b) standalone elementwise | `beta=sigmoid(b)`, `g=-exp(A_log)*softplus(a+dt_bias)`, l2norm `rsqrt`, gated RMSNorm, 게이트 곱 | ⚠️ op 은 지원되나 DeltaNet 그래프 위치에선 tactic 미할당 |
| (c) depthwise conv1d | causal conv(q,k,v) | ✅ conv 지원(단 padding→cat, 융합split→별도conv) |
| (d) 제어흐름/상태 | 순환 스캔 루프, RMW 상태 S, 청크 삼각역행렬 루프 | ❌ tactic 없음 + 런타임 상태버퍼 없음 |

qwen3.5/3.6 도 같은 gated DeltaNet+gated attention 이라 **같은 (b)(d) 벽**을 공유합니다.
gated full-attention 레이어(12/48)는 표준이라 컴파일됩니다 — 문제는 36개 DeltaNet 뿐.

### 1-3. furiosa-llm build / serve 유기적 관계 (실측 규명)

```
furiosa-llm build <model> <out> -tp 8
  └ ArtifactBuilder
      ① validate_model_support (metadata/hf_utils.py)         ← model_type 게이트(빌드측)
         └ native find_compiler_config(model_type, task) 가 None 이면 즉시 ValueError
      ② resolve (presets.find_preset 로 버킷 채움)
      ③ Pipeline build = FX 트레이싱 (parallelize/new_pipeline_builder.py)
         └ 각 버킷마다 모델 forward 를 그래프(IR)로
      ④ TP 그래프 분할 (parallelize/graph_partitioner.py)      ← qwen3_next 가 깨진 곳
         └ 레이어별 (K,V) 입력을 파티션 단위로 매핑
      ⑤ Compile = supertask 단위로 NPU 컴파일러 → EDF 바이너리
      ⑥ artifact.json + binary_bundle.zip + params/ 저장

furiosa-llm serve <artifact> --devices ... [-dp/-pp]
  └ api.py:_init_from_artifact
      ⓐ NativeLLMEngine() 생성 시 model_type 게이트(serve측)          ← 위장으로 통과
         └ artifact.json 의 model_metadata.model_type 만 화이트리스트 검사
            허용: {llama, exaone4, qwen2, qwen3, qwen3_moe, gpt_oss, embed, score}
            (furiosa-generator/.../hf_compat_next_gen.rs:367)
      ⓑ KV 캐시 할당: pipeline_metadata 버킷 kv_cache_size 기준, hf_configs
         (num_hidden_layers/num_key_value_heads/head_dim) 로 shape 검증
      ⓒ -dp = 1장용 tp8 아티팩트를 여러 장에 복제(처리량↑),
         -pp = 스테이지를 장에 분할(큰 모델용). 둘 다 model_type 무관(기하학적).
```

게이트가 둘(④ 빌드 / ⓐ serve)이라는 점이 핵심입니다. **연산은 ⑤에서 EDF 로 다
구워지므로, 게이트만 통과하면 런타임은 그래프를 그대로 실행**합니다 — 이것이 위장이
동작하는 이유이자, qwen3_next 가 ④에서 막히는 이유입니다.

### 1-4. qwen3_next 빌드 실패 계단 (미니 모델 실측)

미니 합성 qwen3_next(4레이어=3 DeltaNet+1 full, 8 experts, ~175M)로 한 단계씩 통과:

| 단계 | 결과 | 비고 |
|---|---|---|
| ① 클래스 resolve | ✅ | `furiosa/models/.../qwen3_next.py` 작성 + `__init__` 등록 |
| ② 빌드 게이트 | ❌→✅ | `find_compiler_config(qwen3_next)=None` → 즉사. `_EXPERIMENTAL_MODEL_TYPES` 로 우회(per-kernel 컴파일은 default config 폴백) |
| ③ **FX 트레이싱** | ✅ | **DeltaNet 순환규칙·depthwise conv1d·gated norm·512→8 MoE·gated attention 전부 그래프화 성공.** = 연산 자체는 트레이싱 가능 |
| ④ **TP 분할** | ❌→✅ (자력 통과) | 근본 원인 = attention 색이 `*.self_attn.attn` 경로 모듈에만 시딩되는 하드코딩(`block_slicer.py:936-937`) → DeltaNet 레이어 attn 색 미시딩 → 파티션 ID 희소(`[0,2,4,6,7,8]` 실측) → PartitionComposer 인덱싱 IndexError. **해법(모델 코드만 수정):** DeltaNet 순환 본체를 `self_attn.attn` 경로 서브모듈로 재배치 + 가중치 이름 리매핑 + `make_example_inputs` 오버라이드로 미사용 KV 미선언 → **ID 연속 [0..8], 분할기 통과** ✅ |
| ⑤ 파티션 경계 노드 복제 | ❌→✅ (자력 통과) | `transform.py:116 replicate_nodes_with_multiple_colors` KeyError — 코드 가정 "다색 노드의 부모는 전부 같은 색"(`transform.py:113`)을 DeltaNet 그래프가 위반. 노드 복제로 통과(자력) |
| ⑥ **primitive→kernelized** | ❌ (FX 경로 한정 벽) | `[2/10] primitive→kernelized` 에서 standalone elementwise(Sigmoid, 심지어 평범한 Mul `[128,512]`)를 커널로 못 냄 — shape·2D/3D·mid-size 무관(`O945→O957→O982→O1057→O1565→O1288` 순으로 끝없이 막힘, IR 덤프로 실측) |

⑤→⑥ 사이의 **op-import** 단계에서 프론트엔드가 못 받는 op 들(`as_strided`/`log1p`/
`constant_pad_nd`)도 모델 코드 재작성으로 제거해 통과시켰음 — 즉 빌드측 열린 Python 계단은
게이트 우회·TP 분할·노드복제·op-import 제거까지 전부 자력으로 올랐고, 막은 벽은 오직 ⑥의
폐쇄 컴파일러 tactic 뿐입니다.

**FX 트레이싱 구현 메모(Q&A):** 구현체는 `furiosa/models/language/architecture/qwen3_next.py`
하나. SDK 가 버킷마다 `torch._dynamo.export`(`export/serve/base.py:56`)로 forward 를 따라가며
정적 그래프를 뜸. DeltaNet 순환은 `for i in range(seq_len)` 루프 → 버킷별 고정 길이로
**정적 언롤**(프리필 128토큰 버킷이면 128회 펼쳐짐), conv1d 는 `aten.conv1d(groups=conv_dim)`
로 기록. SDK 는 hidden_states 를 2D `(tokens, hidden)` 로 넘기므로 DeltaNet 진입 시 3D 로
reshape 하는 어댑터가 필요했음(실측 그래프 브레이크 후 수정).

**캐시 함정:** `~/.cache/furiosa/llm/graphmodules` 캐시는 SDK Python 코드 수정을 키에
반영하지 않음 — 아키텍처 수정 후 `rm .../graphmodules/*Qwen3Next*` 필수.

### 1-5. ⑥의 근본 원인 — "elementwise 미지원"이 아니라 "융합 tactic 부재"

`furiosa/native_torch/compiler.pyi` 의 권위 있는 지원 매트릭스를 실측:

- `is_importable()`(프론트엔드가 받는 ~160 op): `sigmoid`, `mul.Tensor`, `exp`, `log`,
  `rsqrt`, `sum.dim_IntList`, `cumsum`, `constant_pad_nd`, `mm`, `bmm` … **다 포함**.
- `is_supported_aten()`(**커널화 가능** ~100 op): `sigmoid`(L277), `mul.Tensor`(L342),
  `exp`(L275), `log`(L274), `rsqrt`(L285), `sum.dim_IntList`(L273), `cumsum`(L326),
  `constant_pad_nd`(L360), `mm`(L338)/`bmm`(L339) … **다 포함**.

즉 sigmoid·mul·exp·cumsum 은 공식적으로 커널화 지원 op 입니다. 그런데도 빌드는
`O957(sigmoid)`·`O1288(mul)` 에서 "is not an operator that is yet supported" 로 실패.

**정정된 진단:** 실패는 "op 미지원"이 아니라, 그 op 의 **primitive 인스턴스가 DeltaNet
그래프의 특정 위치/문맥(IR상 `context: Sub`)에서 tactic(하드웨어 실행계획)을 못 받아서**.
컴파일러는 elementwise 를 인접 matmul/conv/attention "앵커" 커널에 **융합(fuse)**하는 식으로
tactic 을 붙이는데, DeltaNet 의 게이트·순환 스캔 elementwise 는 융합할 앵커가 없거나
스테이지 출력으로 고립돼 tactic 이 없습니다. qwen3_moe 에선 같은 sigmoid/mul 이
MLP/attention matmul 에 융합돼 통과합니다. → 진짜 부재한 것은 **op 커널이 아니라
"gated-delta 스캔을 하나로 융합하는 tactic"**(FLA GPU 의 `use_qk_l2norm_in_kernel`·
`chunk_gated_delta_rule` 커널이 하는 일)입니다.

### 1-6. 컴파일러 커널이 "어디" 있고 왜 못 넣나 (radare2 검증)

**위치(strings/radare2 실측):** npu-tools(git `3f23a71`) Rust 워크스페이스가
`native_torch.so`(105MB)·`native_llm_common.so`(143MB)에 **정적 링크**.
- 커널라이저: `npu-compiler/crates/npu-compiler-kernelize/src/kernelize.rs` (미지원 op 거부:
  `" is not an operator that is yet supported by the compiler"` @native_torch 0x008798c4).
- op 의미/ALU primitive: `npu-executor-common/src/operator/calculate/{aten_ops/aten_impl.rs,
  primitive_ops.rs}` (`calculate/add`, `calculate/attention_kernel`, `calculate/conv` …).
- op-name→handler: **phf(perfect-hash) 정적 테이블**(런타임 레지스트리 아님; miss 시
  "no entry found for key" 패닉). `ldd`상 외부 libnpu/libfuriosa 의존 없음(완전 static),
  `dlopen`은 CUDA 드라이버 로드용(`libloading`)뿐 — **op 플러그인 메커니즘 없음**.

**왜 radare2 로 못 넣나:** 새 커널 = 새 lowering 패스 + tactic 선택 + 스케줄러 엔트리 +
TU/VE/DPE 명령 코드젠 = **레지스터 할당·재배치된 새 기계어 수천 바이트** → 스트립 105MB
바이너리에 손으로 끼워넣을 수 없음(빈 stub·점프슬롯·리다이렉트 심볼 없음). 우회 config 도 전부 막힘:

| 우회 시도 | 결과 | 이유 |
|---|---|---|
| `allow_unlowered_operators=true` | ❌ | 실측 빌드 **hang**. unlowered op 은 DRAM IR 레벨에 남아 실행 불가, 다운스트림 멈춤 |
| `allow_external_operators=true` | ❌ | **이미 컴파일된 EDF/DMA 블롭**만 주입(ExternalOperator). 임의 op 의 CPU 폴백 아님 |
| `furiosa::` 커스텀 네임스페이스 | ❌ | 전부 기존 aten op 분해이거나 사전컴파일 EDF 래퍼 — 새 코드젠 없음 |
| 순수 Python matmul 재정식화 | ❌ | chunked 로 cumsum→삼각행렬 matmul, pad→cat, mask→상수까지는 되나 **데이터 의존적 exp() decay 게이트·l2norm rsqrt 는 상수로 못 접어** standalone 으로 남음 |

**유일한 정공법(표준 경로):** Furiosa 의 npu-tools 소스로 (a) gated-delta 스캔 tactic 커널 추가
+ (b) 런타임에 순환상태 버퍼 추가 → 재컴파일 = **벤더 2026.3+**. (이는 serve 가 순환상태를
관리 못 하는 것과 **같은 뿌리** — NPU 스택 전체가 트랜스포머 matmul/conv 중심.)

**표준 빌드를 끝까지 충실히 만들려면 필요한 작업(차기 SDK / 벤더 몫):**
1. `parallelize/` 분할기 · `specs/inputs.py`(`CausalModelForwardInputs.kv_caches` **2-튜플
   강제**) · `create_kv_caches`(**균일 shape** 가정)를 확장해 **레이어별 이종 상태 텐서**
   (conv/recurrent)를 그래프 입출력으로 표현 — 우리 쪽에서 가능.
2. **DeltaNet prefill 커널의 RNGD-안전 재작성** — chunked 경로의 fp32 `cumsum`,
   `F.pad`(constant_pad_nd), 부분 `slice_scatter`, 큰 groups depthwise conv1d 회피
   (recurrent 경로는 ③에서 트레이싱됨) — 우리 쪽에서 가능.
3. 런타임(Rust)에 순환 상태 풀 + qwen3_next enum + 스케줄 preset 추가 — 폐쇄 바이너리라
   **벤더 전용**(게이트의 미지원 메시지도 "차기 버전에서 지원" 명시). 1·2는 우리 쪽에서
   가능하나 3 없이는 serve 불가 → **2026.3+ 또는 벤더 지원 필요**.

> ⚠️ 이 절(1-5/1-6)의 "벤더 전용" 결론은 **FX→kernelize 경로에 한정해 여전히 유효**하지만,
> TK-graph 직접 작성으로는 통과합니다(3절). 즉 "op 미지원"이 아니라 "그래프 위치/tactic"
> 문제였다는 1-5 진단이 3절에서 실증됩니다.

---

## 2. serve 게이트 radare2 분석 + 위장(masquerade)

> furiosa-llm 의 `.so` 는 Rust PyO3 확장이라 소스를 못 봅니다. **radare2 6.1.7**(소스 빌드)로
> `native_llm_common.so`·`native_runtime.so` 를 직접 디스어셈블해, "왜 qwen3_next 는 serve 가
> 안 되고 어떻게 통과시키는가"를 바이트 단위로 규명했습니다. **모든 주소·바이트는 실측(2026-06-10).**
> - 분석 도구: `Model_Benchmark/qwen3-next-proj/radare2/`(소스 빌드, `binr/radare2/radare2`)
>   실행: `LD_LIBRARY_PATH=<libr 경로들> ./binr/radare2/radare2 -2 <so>`(`-2` = stderr 억제)
> - 대상 .so(정확 경로):
>   - `/home/jun/furiosa/lib/python3.12/site-packages/furiosa/native_llm_common.cpython-312-x86_64-linux-gnu.so`(143MB)
>   - `/home/jun/furiosa/lib/python3.12/site-packages/furiosa/native_runtime.cpython-312-x86_64-linux-gnu.so`(163MB)
> - 두 .so 는 분석 전 `.orig` 로 백업, radare2 는 **읽기 전용**(`-w` 미사용)으로 돌려 원본 무변경(분석 후 `cmp` 로 PRISTINE 확인).

### 2-1. 게이트는 2겹

1. **load 게이트** — `native_llm_common.so` 가 아티팩트 로드 시 `model_type` 문자열을 serde
   enum `ModelType` 으로 역직렬화. 미등록 값이면 즉시 에러.
   (`api.py:349 NextGenArtifact.load_without_blob` → `furiosa-llm-common/src/artifact/types/next_gen.rs:238`)
2. **engine 게이트** — `native_runtime.so` 가 엔진 생성 시 model metadata 재검증.
   (`api.py:383 NativeLLMEngine(...)` → `furiosa-generator/src/next_gen/hf_compat_next_gen.rs:367`)

**허용 변형(generate)**: `llama, exaone4, qwen2, qwen3, qwen3_moe, gpt_oss`(6개).
(`embed`, `score` 는 pooling 용 별도.) `qwen3_next` 는 **없음** → serde 가 거부.
게다가 `model_type` 은 enum 변형뿐 아니라 **구조 로더**도 고름
(`qwen3`→`qwen3_32b`(dense), `qwen3_moe`→`qwen3_30b_a3b`(MoE)). 차원 기반.

### 2-2. 거부 현장 실측 — "unknown variant"

`mini-qwen3` dense 아티팩트의 `model_type` 을 `qwen3_next` 로 바꿔 serve 시도:

```
RuntimeError: unknown variant `qwen3_next`,
  expected one of `llama`, `exaone4`, `qwen2`, `qwen3`, `qwen3_moe`, `gpt_oss`
  at line 1 column 281
Location: furiosa-llm-common/src/artifact/types/next_gen.rs:238:34
```

가중치 로드 전, 아티팩트 메타(JSON)를 파싱하는 단계에서 serde 가 거부. "expected one of"
6개 목록이 곧 enum `ModelType` 의 generate 변형들.

### 2-3. 변형 테이블·VARIANTS 배열·매처 디스어셈블

**변형 테이블(`izz`/`/`):** `native_runtime.so` 의
`0x00d7ebfc ... ModelTypellamaexaone4qwen2qwen3qwen3_moegpt_ossembedscore ...`
`native_llm_common.so` 의 각 변형 정확 주소:

| 변형 | 주소(native_llm_common) | 길이 |
|---|---|---|
| `llama`(블롭 시작) | 0x00aca217 | 5 |
| `exaone4` | 0x00aca21c | 7 |
| `qwen2` | 0x00aca223 | 5 |
| `qwen3` | 0x00aca228 | 5 |
| `qwen3_moe` | 0x00aca22d | 9 |
| `gpt_oss` | 0x00aca236 | 7 |
| `embed`/`score` | 0x00aca23d / 0x00aca242 | 5 / 5 |

변형들은 **널 구분자 없이 한 덩어리**로 packed. serde 는 (포인터, 길이) 쌍으로 가리킴.

**VARIANTS 배열(`pxq`):** 에러 메시지 "expected one of"는 serde `unknown_variant(value, VARIANTS)`
출력. llama 포인터(0x00aca217)로 검색:
```
[r2]> /x 17a2ac0000000000      ; llama 포인터(LE) → 0x00019b00 hit
[r2]> pxq 96 @ 0x00019b00
0x00019b00  0x0000000000aca217  0x0000000008608808   ; (name_ptr=llama, payload_ptr)
0x00019b10  0x0000000000000008  0x0000000000aca21c   ; 0x8, (name_ptr=exaone4)
...
```
엔트리 = 24바이트 `(name_ptr, payload_ptr, 0x08)`. payload(0x086088xx)는 `{ptr=0(재배치), len}`
fat-pointer 풀이고 `ptr` 은 **로드 시 동적 재배치**(파일에는 0, 길이만 정적 7,5,5,9,7…).
⇒ 파일에서 포인터를 정적 패치해도 **로더가 재배치로 덮어씀** → 데이터 패치가 어려운 이유.

**진짜 매처 디스어셈블(`aar`→`pd`) — 첫 바이트 점프 테이블:**
`qwen3`(0x00aca228) 를 `lea` 하는 함수(0x1fcd350~0x1fcd402)가 serde `deserialize_identifier`
의 첫 바이트 점프 테이블 디스패치:
```asm
0x1fcd380  mov    rdx, rsi                 ; rdx = 입력 길이
0x1fcd383  movzx  eax, byte [rdi]          ; eax = 입력[0] (첫 바이트)
0x1fcd386  lea    rcx, [0x00aca0d4]        ; 점프 테이블 베이스(256 × i32)
0x1fcd38d  movsxd rax, dword [rcx+rax*4]   ; off = jumptable[첫바이트]
0x1fcd391  add    rax, rcx                 ; target = base + off
0x1fcd394  jmp    rax                      ; → 해당 변형 핸들러로
; --- 변형별 "단말 블록": (변형문자열, 길이) 싣고 compare 로 테일콜 ---
0x1fcd396  lea rdi,[0x00aca217] ; mov esi,5  ; jmp [0x0888c620]   ; llama
0x1fcd3a8  lea rdi,[0x00aca22d] ; mov esi,9  ; jmp [0x0888c620]   ; qwen3_moe
0x1fcd3ba  lea rdi,[0x00aca223] ; mov esi,5  ; jmp [0x0888c620]   ; qwen2
0x1fcd3cc  lea rdi,[0x00aca228] ; mov esi,5  ; jmp [0x0888c620]   ; qwen3  ★
0x1fcd3de  lea rdi,[0x00aca21c] ; mov esi,7  ; jmp [0x0888c620]   ; exaone4
0x1fcd3f0  lea rdi,[0x00aca236] ; mov esi,7  ; jmp [0x0888c620]   ; gpt_oss
0x1fcd402  int3
```
해석(우편 분류기 비유): ① **첫 글자로 1차 분류**(점프 테이블 0x00aca0d4) — `q` 로 시작하는
qwen2/qwen3/qwen3_moe 는 추가로 길이·뒷바이트로 2차 분류. ② **단말 블록** — 각 변형마다 "기대
문자열 + 길이"를 레지스터에 싣고 공용 비교함수 `[0x0888c620]` 으로 **테일콜**(jmp). 매치면
discriminant 를, 아니면 에러를 호출자에게 바로 반환. ③ `qwen3_next`(길이 10)는 어느 단말에도
안 맞아 에러 단말로 떨어짐. **핵심: 매칭은 packed 블롭이 아니라 이 단말 블록들의 (lea 주소, mov 길이)** 로 이뤄짐.

### 2-4. 리터럴 `qwen3_next` 바이너리 패치 실험 (사본에 직접 시도, 실측)

원본을 복사해 사본을 radare2(`-w`)로 패치하고 gate-1(load) 통과 검증(원본은 `.orig` 복원점):

**가설**: qwen3 단말(0x1fcd3cc)이 비교하는 문자열 `"qwen3"`(5) → `"qwen3_next"`(10) 으로 바꾸면 매처가 통과시킬 것.
```
wx 7177656e335f6e657874 @ 0x1fcd402   # 코드 케이브(int3 14B)에 "qwen3_next" 기록
wx 2f000000             @ 0x1fcd3cf   # qwen3 단말 lea disp32 → 케이브(0x1fcd402)
wx 0a                   @ 0x1fcd3d4   # mov esi,5 → mov esi,10
```
검증: `pd 2 @ 0x1fcd3cc` → `lea rdi,[0x1fcd402]; mov esi,0xa`, `ps @ 0x1fcd402` → `qwen3_next`. ✅

**결과**: 사본 import 후 `NextGenArtifact.load_without_blob(<qwen3_next>)` → **여전히
`unknown variant 'qwen3_next'` 거부** ❌. **단말 패치는 효과 없음.**

**배운 것**: 매처는 입력이 단말에 닿기 **전에 길이/구조로 먼저 분기**한다(serde 가 흔히 생성하는
`match len { 5 => …, 7 => …, 9 => … }` 길이-우선 디스패치). `qwen3_next`(길이 10)는 길이 버킷이
없어 **단말 도달 전 에러 경로**로 빠짐. 따라서 리터럴 통과에는 **길이 디스패치 + 라우팅 + 단말**을
좌표 맞춰 고치고 `native_runtime.so`(gate-2)에도 복제해야 하며, 스트립 143MB/163MB 에서 함수
경계를 추정해야 하므로 **고위험·고비용**(다른 model_type 로딩까지 깨질 수 있음). 설령 두 게이트를
다 뚫어도 우리 EDF 는 dense/MoE 라 **리터럴 qwen3_next 의 실제 DeltaNet 연산이 없음** — 라벨만 바뀜.

**안전성**: 패치는 전부 사본, 테스트 후 즉시 `.orig` 복원 → 두 `.so` `cmp` 로 PRISTINE 확인.

### 2-5. 안전·실측된 통과법 — 마스커레이드(위장)

게이트가 `model_type` **문자열만** 보고 연산은 EDF 에 이미 구워져 있으니, 라벨을 허용 변형으로
바꾸면 통과(바이너리 무패치, 가역).

| 아티팩트 | model_type | serve |
|---|---|---|
| `mini-qwen3-as-next` | `qwen3_next` | ❌ serde "unknown variant" |
| `mini-qwen3-next-served` | `qwen3`(위장) | ✅ 게이트 통과 + 토큰 생성 |

```bash
# qwen3_next 라벨 아티팩트를 통과시키는 변환 (KV 차원은 절대 불변)
python - <<'PY'
import json; p='artifact.json'; d=json.load(open(p)); md=d['model']['model_metadata']
md['model_type']='qwen3'                          # ← 게이트 통과(허용 변형 + dense 구조로더)
md['hf_configs']['model_type']='qwen3'
md['hf_configs']['architectures']=['Qwen3ForCausalLM']
json.dump(d,open(p,'w'))
PY
furiosa-llm serve <artifact> --devices npu:0 --port 8000   # → 부팅·생성 정상
```

- MoE 코더(예: Qwen3-Coder-30B-A3B)면 `qwen3_moe` 로 위장 → `qwen3_30b_a3b` 구조로 라우팅(더 충실).
  도구: `qwen3-next-proj/masquerade_artifact.py`(`--copy`로 원본 보존, 사본은 하드링크라 디스크 추가소모 없음).

**위장 함정(serve 측 KV 바인딩 정밀 메커니즘, 실측 정정):**
- KV 캐시 버퍼의 수·모양·메모리 예산은 전적으로 아티팩트 `pipelines[].tensors` 중
  `origin.name=="kv_caches"` 텐서 목록에서 파생(KVCachePlan·블록 수·출력이 nhl=1/2/48 모두
  바이트 동일). 초판의 "런타임이 hf_configs 의 num_hidden_layers 기준으로 캐시를 할당·검증"은
  **반증됨** — num_hidden_layers 교차검증은 없음. hf_configs 는 ①필수 필드 파싱(model_type,
  num_hidden_layers, num_attention_heads, max_position_embeddings, hidden_size, vocab_size,
  eos_token_id, architectures), ②model_type 화이트리스트 게이트, ③`layer_types` 값 검증, ④(JIT 시) 컴파일러 설정에 쓰임.
- ⚠️ **`layer_types` 함정**: Rust hf_config 파서는 `sliding_attention`/`full_attention` 값만 허용
  → **`linear_attention` 이 남아 있으면 "unexpected layer_types" 패닉**. 실제 Qwen3-Coder-Next
  config 의 layer_types 에는 linear_attention 이 **36개** 들어있으므로 위장 시 반드시 제거하거나
  전부 full_attention 으로 재작성. KV 차원(`num_hidden_layers/num_key_value_heads/head_dim`)은 불변 필수.
- ⚠️ **prefix cache 함정**: serve 기본값이 prefix cache ON. 페이지드 KV 만 복원하므로 비-KV
  순환 상태를 가진 하이브리드 모델에서는 캐시 히트 시 **조용히 틀린 출력**. 하이브리드 serve 실험 시 비활성화 필요.
- 함의: 레이어 일부만 KV 캐시를 갖는 하이브리드 아티팩트(12/48만 선언)도 serve 측 바인딩
  관점에선 **수용 가능** — 막는 것은 바인딩이 아니라 순환 상태의 프레임 간 유지(6절).

### 2-6. 달성한 대안 — Qwen3-Coder-30B-A3B-Inst-FP8 부활 (위장)

이전 세션에 "serve 패닉(qwen3_moe×FP8 커널 부재; api.py:383 → hf_compat_next_gen.rs:367 게이트가
qwen3_moe FP8·BF16 모두 거부, 2026.2.0 MoE serve 미개방)"으로 사장된 모델을, 빌드된 아티팩트의
`model_type` 을 `qwen3`(dense, 허용)로 위장해 부활. 도구 `qwen3-next-proj/masquerade_artifact.py`.

**4장 dp 서빙 실측(2026-06-10, max_ctx 65536):**

| 동시성 | 합산 처리량 | 스트림당 | 비고 |
|---:|---:|---:|---|
| 1 | 63.2 tok/s | 63.2 | 정상 코드 출력(prime/fib/quicksort 정확) |
| 8 | 429.7 tok/s | 53.7 | |
| 32 | 1036.2 tok/s | 32.4 | |

dense qwen3 스케줄러 preset 으로 MoE 그래프를 구동. **"serve 게이트만 통과시키면 컴파일된
그래프는 실행된다"를 실증**(기존 dense Qwen2.5-Coder-14B 단일 30.7 tok/s 대비 큰 향상).
직접 serve(위장 사본):
```bash
~/furiosa/bin/furiosa-llm serve \
  /home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/artifacts/qwen3-coder-30b-masq \
  --devices npu:0 --host 0.0.0.0 --port 8000          # 1장 (단일 63 tok/s)
# 4장 처리량(합산 ~1036 tok/s): --devices "npu:0,npu:1,npu:2,npu:3" -dp 4
```
(BF16 Qwen3-Coder-30B-A3B-Inst 도 `qwen3` 위장으로 2장(pp2)에서 정상 코드 생성:
`qwen3-coder-30b-a3b-inst-tp8-65k-tc`.)

### 2-7. 우리가 SDK 에 가한 변경 (최소·문서화)

| 파일 | 변경 | 되돌리기 |
|---|---|---|
| `furiosa_llm/artifact/presets.py` | MINI_SMOKE_PRESET + ref 3개(미니 빌드용) | 해당 블록 삭제 |
| `furiosa/models/language/architecture/qwen3_next.py` | 신규 아키텍처(빌드 ⑥에서 막힘, probe) | 파일 삭제 |
| `furiosa/models/language/__init__.py` | qwen3_next import/`__all__` 2줄 | 해당 줄 삭제 |
| `furiosa_llm/metadata/hf_utils.py` | `_EXPERIMENTAL_MODEL_TYPES` 빌드 게이트 우회 | 블록 삭제 |

원본 모델·아티팩트 가중치는 무변경.

---

## 3. 돌파 — TacticKernelModule 로 NPU 커널 손수 작성

> 질문: "컴파일러 커널 위치를 찾아 거기에 커널/연산 패턴을 넣고 radare2 로 코드를 주입하면
> 되지 않나?" 답: **radare2 커널 주입은 불가능**(2절·1-6). **그러나 Rust 소스 없이도
> `furiosa.torch.TacticKernelModule`로 DFG 커널을 손수 작성해 EDF 컴파일·NPU 실행이 가능**합니다.

### 3-1. TacticKernelModule — Rust 소스 없이 커널 작성

`furiosa/native_torch/compiler.pyi` 의 지원 매트릭스(1-5)가 "op 미지원이 아니다"를 보여줬고,
SDK 에는 **Python+YAML 로 DFG 커널을 손수 작성하는 공식 API** 가 들어 있습니다.

**경로:** `furiosa.torch.TacticKernelModule(dsl_yaml)` → `Dfg.parse` → `torch.ops.furiosa.dfg`
커스텀 op → `torch.compile(module, backend=furiosa.torch.backend)` → EDF 컴파일 + NPU 로드.
(Furiosa 자신이 `models/core/operators/tk_graphs/moe_blockwise_compute_wg_idx.yaml` MoE
work-group-index 커널을 이 방식으로 작성·사용 중.)

**실측(이 머신, npu-compiler 3f23a71):**
- ✅ 손수 작성 DSL 파싱(`TacticKernelModule`), CPU 실행(`DfgExecutor`) OK.
- ✅ **커스텀 elementwise add 커널 EDF 컴파일 → rngd:0 실행 → 정답**(CPU 폴백 0회 = NPU 실행 검증).
- ⚠️ 저수준 `compiler.compile(ExportedProgram)` 은 `furiosa::dfg` 재import 실패 → **고수준
  dynamo 백엔드 경로만 동작**. 입력 없는 SymArange-only 그래프는 NPU 컴파일서 segfault →
  **커스텀 커널은 실입력 텐서를 최소 1개 받아야 안전**.

**실행 레시피(확인됨):**
```python
import torch                                  # 먼저
import furiosa.torch as ft
m  = ft.TacticKernelModule(open(yaml).read())
cm = torch.compile(m, backend=ft.backend)
cm(*[t.to('rngd:0') for t in inputs])
```

**DSL 표현력(`SymTacticKernel`):** `EinsumByVe`/`EinsumByDpe` contraction + 1급 `Einsum`
(input/output equation), SSA vector ALU(`MulFxp`·`AddFxp`·`SubFxp`), unary(`Sigmoid`·`Exp`·
`NegExp`·`Erf`), reduction(`LocalReduceAdd`/`GlobalReduceAddFxp`), `Cumsum`·`Gather`·`Where`.
**carried-state `Loop` 노드**(initial_states·captured_inputs·final_states·inner_operators)도
손수 작성 가능(컴파일러 생성 전용 아님). 단 DSL 은 미문서화·미검증(shipped 예제는 loop-free MoE 하나뿐).

### 3-2. 7차 실증 — DeltaNet 핵심 연산을 직접 작성해 NPU 실행

| 커널 (DeltaNet 대응) | 파일 | NPU 실행 | 검증 |
|---|---|---|---|
| baseline elementwise add | `custom_add.yaml` | ✅ rngd:0 | int32, torch 일치, CPU폴백 0회 |
| **게이트** `v*sigmoid(b)`(Sigmoid+MulF 융합) | `dn_gate.yaml` | ✅ rngd:0 | **fp32**, max err 1.19e-7, CPU폴백 0 |
| **kv-read contraction** `out[v]=Σ_k S[k,v]·k[k]`(einsum kv,k→v) | `dn_einsum.yaml` | ✅ rngd:0 | torch.einsum 정확 일치, CPU폴백 0 |
| **순환 Loop**(carried-state, 누산 스켈레톤) | `dn_loop.yaml` | ❌ frontier | parse OK, 실행 불가 |

**DSL 실전 교훈(7차):** ① VE ALU 의 `*Fxp` op 은 **고정소수/정수 전용** — f32 엔 `MulF`/`AddF`
(`MulFxp` 는 `type_checker.rs:92` 서 거부). ② contraction 은 `kind: EinsumByVe` + broadcast
read(`tiles`)를 첫 read 로 + 둘째 vector_op 로 `LocalReduceAddFxpSat`(공유 라벨이 출력에 없으면
그 축이 축약). `EinsumByDpe`(진짜 MAC)는 추가 struct 채워야 동작(5절). ③ 커널은 실입력 텐서 ≥1개 필요.

**순환 Loop = 정확한 벽(두 겹, 둘 다 벤더 전용):**
- `option: Loop` 노드는 **parse 통과**(LoopInterface: loop_index·limit(정적 int)·initial_states·
  captured_inputs·final_states·local_tensors). 그러나 CPU DfgExecutor 에서 `loop_impl.rs:126`
  "loop index tensor must be SPM scalar or unlabeled [1]" 로 실패 — naive_yaml DSL 엔 **상수 차원
  표현이 없고**(Var(symbol)/BinOp 만; `Var:"1"` 은 자유 심볼이 됨), local_tensors 심볼이
  `symbolic_params` 로 해소 안 됨.
- 게다가 raw Dfg **Loop 는 NPU AOT 경로가 없음**(`furiosa::dfg only runs on CPU device`).

### 3-3. 8차 돌파 — 완전한 Gated DeltaNet 레이어를 NPU 에서 계산 (~1e-7)

손수 작성한 TK-graph 커널로 **완전한 Gated DeltaNet 레이어를 RNGD NPU 에서 계산**, HF
`torch_recurrent_gated_delta_rule` 과 **~1e-7 일치**. 순환은 (a) host-loop(상태 스레딩) 또는
(b) 그래프 내 언롤(단일 EDF) 로 처리 — 막혔던 `Loop` 노드 없이 우회 성공.

| 산출물 | 내용 | NPU 결과 | 검증 |
|---|---|---|---|
| `dn_einsum_f32.yaml` | fp32 contraction Σₖ S[k,v]·k[k] | ✅ rngd:0 | torch.einsum, err 2.4e-7 |
| `dn_rank1.yaml` | rank-1 외적 갱신 S+=k⊗δ | ✅ rngd:0 | 정확 일치 |
| `dn_decay.yaml`·`dn_delta.yaml` | S·decay, (v−kv)·β | ✅ rngd:0 | 정확 일치 |
| **`dn_step.yaml`** | delta-rule 한 스텝 전체(7-op 융합) | ✅ rngd:0 | Sout·out 둘 다, err <1.2e-7, CPU폴백 0 |
| **host-loop (T=8)** | 스텝 커널을 8토큰에 상태 스레딩 | ✅ rngd:0 | **HF ref 일치**, err 6e-8, CPU폴백 0/8 |
| **`dn_prefill_unroll4.yaml`** | T=4 언롤 → 단일 EDF, 한 번의 forward | ✅ rngd:0 | **HF ref 일치**, err 1.2e-7 |

**핵심 DSL 돌파 교훈(8차):**
- 외적(broadcast 2개)은 단일 Elementwise 면 EDF 가 Cpu 노드로 떨궈 실패 → **`EinsumByVe`(Reduce
  inst 없이)** 가 외적, Reduce inst 있으면 contraction. 둘 다 broadcast read 를 read0 로.
- per-head 스칼라(decay/β)는 caller 가 `torch.full` 로 동형 텐서로 materialize(진짜 [1]
  브로드캐스트 미지원), 또는 `{ConstFloat}` 직접 사용.
- 텐서 id 는 그래프 전역 flat, op-local `Tensor:0/1` 참조와 분리. 다출력 OK. fp32 는
  `MulF/AddF/SubF`+`LocalReduceAddF`.
- q 는 1/√d_k 스케일을 **커널 전에** 먹여야 HF 와 맞음(HF 는 루프 전 query*scale).
- 언롤: 스텝 7-op 바디를 T번 복제(매 step 새 intermediate id, S 를 다음 step S_in 으로 연결)
  → 단일 Dfg/EDF. `gen_unroll.py` 가 T 파라미터화(T=8,16 도 생성 가능).

**남은 단 하나의 진짜 vendor-lock = 그래프 내 네이티브 `Loop` 노드**(`dn_loop2.yaml`): 정적
`UnlabeledShape{[1]}` 로 첫 벽(SymExpr 상수)은 넘었으나, `loop_impl.rs:126` 이 loop_index 를
**스케줄러가 만드는 SpmShape 스칼라**로 요구 → naive_yaml 로 못 만듦. **그러나 언롤이 이를
불필요하게 만듦**(고정 seq 면 언롤, 가변이면 host-loop).

산출물: `qwen3-next-proj/tk_kernels/`(커널 YAML + `run_dn_step.py`·`host_loop_test.py`·
`unroll4_test.py`·`gen_unroll.py` 드라이버).

---

## 4. 컴퓨트 전부 NPU 검증 — 실차원·멀티헤드·청크·완전 레이어

### 4-1. 9차 스케일 — 실차원·멀티헤드·청크 형태까지 NPU 검증

| 항목 | 결과 | 검증 |
|---|---|---|
| **실차원 스케일** d_k=d_v=128 | ✅ `scale_test_d128.py` | dn_step.yaml 무변경(symbolic Var:K,V), torch 일치 오차 8.6e-6, CPU폴백 0 |
| **멀티헤드** H=4, d=128(`dn_step_mh.yaml`) | ✅ rngd:0 | 4헤드 전부 일치, 오차 1.5e-5, CPU폴백 0. head 는 batch축(특별처리 불필요) |
| **언롤 한계**(`dn_prefill_unroll{8..128}.yaml`) | ⚠️ 소프트 한계 | 컴파일은 T=128(896op)까지 OK 이나 시간 초선형(T64≈110s, T128≈449s); 정확도는 fp32 누적으로 **T~8 초과 시 drift** |
| **청크 형태 한 청크**(`dn_chunk.yaml`) | ✅✅ rngd:1 | chunk-parallel gated delta rule, **3개 matmul 코어 전부 NPU**, 실config C=64/d=128 오차 7.6e-6, CPU폴백 0 |

**핵심 스케일 발견(9차):**
- **head 축 = 그냥 batch 축**: 모든 텐서 shape 에 최외곽 `h` 라벨 추가, reduce 축엔 절대 안 넣음
  (contraction 은 head별 유지), tile 안 함. 한 번에 성공. symbolic 차원이라 dn_step.yaml 은
  **차원 불가지론** — 같은 YAML 이 d=4·d=128·H=4 다 동작.
- **행렬-행렬 matmul einsum `ck,dk→cd` 가 NPU 내려감**(8차 vector-broadcast einsum 의 일반화):
  read0=A[c,k] tiled d, read1=B[d,k] tiled c(둘 다 broadcast) + Reduce over 공유축 k → 출력 [c,d]
  (두 생존축이 서로 다른 피연산자에서). bmm/matmul 전부 표현 가능.
- **청크 형태가 긴 seq 의 정답**: 언롤은 O(T) 그래프 + T~8 정확도 한계. 청크는 matmul 위주라
  O(T/C) + 정확. cumsum 만 host(순차), exp/마스크는 NPU Unary 로.
- **새 frontier+해결**: no-reduce EinsumByVe 에서 read0=broadcast-1D, read1=full-2D 면 EDF 가
  Cpu노드로 떨궈 그래프 분할 실패(`clusterer/cluster.rs:32 "multiple internal subgraphs"`).
  no-reduce 외적은 **양쪽 다 1D broadcast** 일 때만 NPU. 해결: 한쪽을 2D 로 materialize(또는 2D Unary Exp) 후 matching-shape Elementwise MulF.

산출물 추가: `dn_step_mh.yaml`·`dn_chunk.yaml`·`dn_prefill_unroll{8..128}.yaml`·`scale_test_d128.py`·`gen_chunk.py`·`mh_test.py`·`unroll_limit_test.py`.

### 4-2. 10차 — 레이어 구성 컴퓨트 조각 전부 NPU 검증

| 조각 | 파일 | NPU 결과 | 검증 |
|---|---|---|---|
| **멀티청크 스캔**(inter-chunk state carry) | `dn_chunk_full.yaml`(12-op: 5 matmul+7 elementwise) | ✅ rngd:1/2 | NC=3 청크 host-loop, HF `torch_chunk_gated_delta_rule` 일치 out 1.5e-8·state 3e-8, **carry 실검증**(S_prev≠0), CPU폴백 0 |
| causal conv1d + SiLU | `dn_conv1d.yaml`(K=4 depthwise, 8-op) | ✅ rngd:0 | F.silu(conv1d) 일치 1.4e-6 |
| l2norm(q/k 정규화) | `dn_l2norm.yaml`(6-op) | ✅ rngd:0 | x·rsqrt(Σx²+eps) 일치 6e-8 |
| gated RMSNorm(출력) | `dn_gnorm.yaml`(10-op) | ✅ rngd:0 | Qwen3NextRMSNormGated 일치 2.9e-6 |

**멀티청크 핵심 설계:**
- **삼각역행렬 정련(HF L511-515)은 생략 불가**(T=I 면 1e30 발산). 단 T·value·k_cumdecay·decay_mask 는
  **S_prev 무의존**이라 host 사전계산 후 입력 주입, **S_prev 닿는 recurrence(4 matmul)만 NPU** —
  정련 honor + inter-chunk 상태 carry 온디바이스 정확.
- **q,k L2정규화 필수**(실모델 `use_qk_l2norm_in_kernel=True`): 안 하면 state 가 ~1e14 로 폭발해
  NPU fp32 matmul 상대오차 1e-6 이 절대오차 1e8 됨. 커널 수학은 어느 레짐서나 정확(CPU maxerr
  0.0). β 는 host 의 k_beta/v_beta 로만 진입(온칩 커널 β-불변).

**주변 op DSL 실측(`probe_unary/binary.py`):**
- 유효 Unary: `Exp`·`Sigmoid`·`Sqrt` 만(수치 정확). `Tanh/Sin/Cos/Log` 는 parse 되나 **값 틀림**
  (table_lookup 필요). `Rsqrt/Reciprocal/Silu/Gelu` 등은 enum 없음 → **native rsqrt 없음**:
  `1/√s = √s/s`(Sqrt 후 DivF(rt,se))로 구현(maxerr 6e-8).
- 유효 Binary: `MulF/AddF/SubF/DivF`. DivF 는 함정 다수(numerator==1 상수면 collapse). reduction `LocalReduceAddF` 만.
- **reduction 은 reduce축=외곽 + 생존축=내곽이고 생존축 ≥128 일 때만 NPU**(작으면 Cpu노드).
  l2norm 은 [d,m] 로 transpose 해 sumsq.
- **융합 inst ≤2개**(긴 chain 은 `preferred_ve_lhs is not an operand` 실패) → op 쪼개기.

산출물 추가: `dn_chunk_full.yaml`·`gen_chunk_full.py`·`run_dn_chunk_full.py`·`dn_conv1d/l2norm/gnorm.yaml`·`gen_dn_layer.py`·`test_dn_layer.py`·`probe_unary/binary.py`.

### 4-3. 11차 캡스톤 — 완전한 DeltaNet 레이어가 HF 와 ~1e-7 (적대적 검증 통과)

조각들을 **완전한 단일헤드 Gated DeltaNet 레이어 forward**(`full_layer.py`, host-오케스트레이션)로
조립 → HF `Qwen3NextGatedDeltaNet`(진짜 torch 경로) 전체와 대조. 별도 검증 에이전트가 적대적 재검증.

- **결과:** allclose(atol 1e-2)=True, **maxerr 3.9e-7**(hidden=256, K=V=32, T=32/2청크).
  T=48/3청크 2.5e-7, T=64/4청크 2.7e-7. **총 `_dfg_inner=0`**(모든 DeltaNet 고유 스테이지 NPU 실행).
- **NPU 실행 스테이지(스테이지별 dfg_delta=0):** conv1d+SiLU · l2norm(q)·l2norm(k) ·
  beta=sigmoid · 멀티청크 스캔(inter-chunk 상태 carry, 청크당 dn_chunk_full 1회, 4청크까지 검증) ·
  gated RMSNorm(core,z, z-게이팅 실재).
- **적대적 검증 4종 통과:** ①`_dfg_inner` spy 가 진짜 CPU경로라 count=0=NPU확정 ②HF 는 진짜 torch
  경로(`F.silu(conv1d)`+`torch_chunk_gated_delta_rule`, fla/fast-path 비활성) ③오차 1e-7 로 tol
  1e-2 보다 5자릿수 아래 + 오염주입 시 오차 1.2(민감) ④l2norm 실변환(norm 20→1.0)·gnorm z-게이팅
  (zero gate→zero out)·미달크기 reduction 은 조용한 폴백 아닌 **에러**라 가짜 통과 불가.
- **정직한 caveat(검증됨):** 이 조립에선 **host 실행** = in_proj/out_proj matmul(NPU-compilable,
  dn_einsum_f32 로 별도확인) + g 의 softplus 스칼라(native DSL op 없음; log 은 값오류라 못 씀) +
  청크의 S_prev-무의존 사전계산(삼각역행렬·cumsum·decay_mask). 단일헤드·소차원(hidden=256).
  l2norm/gnorm 은 행<128 이면 Cpu노드라 128 로 zero-pad(정확).

산출물: `full_layer.py`.

### 4-4. 12차 — 투영까지 NPU(96.74%)

**(A) 최대한-NPU 레이어**(`full_layer_npu.py` + `dn_linear.yaml`): nn.Linear `y=xWᵀ` 를
EinsumByVe matmul(`'ti,oi→to'`) 커널로 만들어 **in_proj_qkvz/in_proj_ba/out_proj 까지 NPU**.
완전 레이어 재검증: HF 와 **maxerr 1.6e-6**, 총 `_dfg_inner=0`, **matmul FLOP 의 96.74%가 NPU**
(1,458,176 MAC). host 잔여 3.26%(49,152 MAC)는 오직 **순차 삼각역행렬 T 사전계산**
(k_beta@kᵀ·T@v_beta·T@k_cumdecay) + softplus 스칼라뿐 — 둘 다 본질적 한계(softplus 는 native
DSL op 없고 log 은 값오류; tri-inverse 는 data-dependent 순차).

산출물: `dn_linear.yaml`·`full_layer_npu.py`·`test_dn_linear.py`.

---

## 5. DPE(EinsumByDpe) 돌파 — systolic matmul, 정확한 변환 레시피

> 8차~12차의 모든 matmul 은 `EinsumByVe`(벡터 엔진 = broadcast-multiply-reduce)로 돌았습니다.
> 그러나 VE matmul 은 `[.,o,i]` outer product 전체를 materialize 한 뒤 i 로 reduce 하므로
> N개 헤드/expert 를 배치하면 한 op 이 N배 데이터를 materialize — 처리량 이득 없이 일감만 N배.
> 진짜 속도 레버는 **`EinsumByDpe`(실제 MAC/DPE = systolic matmul 엔진)** 였습니다.

### 5-1. 배치-벡터엔진 퇴행 (왜 DPE 가 필요한가, 정직한 음성 결과)

baseline ~55.8s/tok 의 병목은 **NPU 디스패치 횟수**: 토큰당 NPU stage 가 deltanet 36000·
moe 59336(prefill+24토큰 누적) — DeltaNet 은 32 value-head 를 루프(헤드당 dn_chunk_full),
MoE 는 top-10 active expert 를 하나씩 호출.

**시도: 배치(batch-axis)로 디스패치 줄이기**(검증된 batch-axis 규칙: 헤드/expert 를 최외곽 축, tile/reduce 안 함):
- DeltaNet head-batched `dn_chunk_full_mh.yaml`: 32 dispatch → 1(32×), looped 와 **bit-identical**(maxerr 0.0), HF 8.9e-8 유지.
- MoE expert-batched `dn_linear_be.yaml`: 298 → 10 dispatch(29.8×), maxerr 1.19e-7, 전부 NPU.

**그러나 wall-clock 이 오히려 느려짐(REGRESSION).** 전체 모델 prefill 이 레이어당 ~7.5s →
~2분으로 **~16× 느림**. 원인: VE matmul 은 N개 배치 시 한 op 이 N배 데이터 materialize — 처리량
이득 없이(systolic 아님) 일감만 N배 + SRAM 압박. MoE 에이전트 실측: 배치 gate matmul 1회(E=10)
~89s = baseline 토큰 전체(55.8s)보다 큼. → **검증된 per-head/per-expert 경로로 복원**
(`model.py` 가 `DeltaNetLayerLooped` + `moe_forward_npu_unbatched` 사용; 55.8s/tok). 배치 버전은
참고용 보존(`deltanet_layer.py`·`moe.moe_forward_npu`·`dn_chunk_full_mh.yaml`·`dn_linear_be.yaml`).

### 5-2. EinsumByDpe 돌파 — 1.96~3.8× 빠름

"미해결 frontier"였던 `EinsumByDpe` 를 깸. `dn_linear_dpe.yaml` 이 NPU 실행(`_dfg_inner=0`),
torch F.linear 와 1e-2 일치, **EinsumByVe 대비 compute 3.8×·end-to-end 1.96× 빠름**.
3각도 정찰(컴파일러 덤프 CBOR 디코드·serde 역설계·점진적 에러추적 11회)로 정확한 레시피 확보.

**정확한 변환 레시피(EinsumByVe matmul → EinsumByDpe):**
- **kind**: `EinsumByVe` → `EinsumByDpe`.
- **contraction 을 `ein_ops.reduce` 로**(VE 는 `ein_ops:~` 였음):
  `{mode: Add, input: <pre-reduce 곱 [t,o,i] TensorLike>, axes: [<contract 축 LabelStride>], source: ""}` + `mul_source: ""`(string).
- **vector_ops 는 단일입력 identity passthrough**:
  `{inputs:[0], insts:[{def:1, Reduce LocalReduceAddF operand Tensor:0 axes Tag:[]}]}`
  (빈-axes Reduce = DSL 에 identity Unary 없어서 이게 유일한 통과 관용구).
- **reads/write 그대로**. → 모든 `'ti,oi→to'` matmul(dn_linear·dn_linear_be·dn_chunk_full QK/KV·dn_einsum)에 일반화.

**정밀도:** DPE 는 **bf16 systolic** 이라 ~0.23% rel(atol 1e-2; 1e-3 불가). 단 실모델이
FP8/bf16 이라 오히려 더 충실. **f32 정확 reduce 필요한 곳만 VE 유지.** 컴파일러는 8×8 matmul 도
항상 DPE 로 낮춤 → DPE 가 네이티브 정답, VE 는 우리가 손으로 쓴 것뿐.

**역설계 방법(재사용):**
- serde FIELDS 는 `.data.rel.ro` 의 (ptr,len) reloc 배열로 복원(strings 는 dedup 돼 순서 신뢰불가).
- DFG `serialize_to_str` = base64([8B len][CBOR]) → `cbor2.loads(b64decode(s)[8:])`.
- DPE op 은 `reduce_mode`+`acc_major_mode` 키 가진 dict.

산출물: `dn_linear_dpe.yaml`·`dpe_serde_fields.md`·`dpe_incremental_log.md`·`dpe_struct_from_dump.md`·`bench_dpe_vs_ve.py`·`dpe_result.md`.

### 5-3. DPE 를 모델 전체에 적용 — 실측 가속

`QCN_DPE=1` 플래그 하나로 attn·moe·deltanet 의 모든 matmul(proj·out_proj·QK/AV·MoE
gate/up/down·DeltaNet 스캔)을 DPE 로 전환. **실측 end-to-end:**

| 지표 | VE baseline | DPE | 가속 |
|---|---|---|---|
| Prefill(24토큰 프롬프트) | 360.8s | **76.9s** | **4.69×** |
| Decode | 55.8 s/tok | **35.15 s/tok** | **1.59×** |
| 투영 matmul 512→2048 | 13.09ms | 1.44ms | **9.06×** |
| expert gate 2048→512 | 5.48ms | 2.47ms | 2.21× |

- **정확성 보존**: 생성 코드 여전히 올바른 quicksort(첫 4토큰 byte-identical), 4레이어 HF 대조
  atol 1e-2 통과(per-layer maxerr ≤2.5e-2), attn layer-3 maxerr 5.5e-3·MoE layer-0 7e-4 vs HF.
  전부 NPU(`_dfg_inner=0`).
- **decode 가 1.59×로 prefill(4.69×)보다 modest 한 이유**: decode 는 일부 **host-bound**
  (DeltaNet 32헤드 cumsum·tri-inverse T-행렬, MoE 라우팅) — matmul 만 빨라지고 host 부분은 그대로.

**발견 제약(중요, 반드시 보존):**
- **제약 ① DPE per-graph 2개 cap**: 한 TacticKernel 그래프에 `EinsumByDpe` 3개+면
  `fuse_mamma_to_single_einsum_by_dpe` 융합 패스가 systolic array 를 오스케줄해 **조용히 오컴파일**
  (dfg=0 이나 garbage, maxerr 0.5). DeltaNet 스캔(5 matmul)은 2개만 DPE(`dn_chunk_full_dpe2.yaml`,
  maxerr 2.5e-4 통과), 나머지 VE. 전체 DPE화하려면 그래프를 ≤2-DPE 단위로 분할 필요(YAML 필드 아닌 파티셔닝).
- **제약 ② DPE 출력축 O=1 거부**(shared_expert_gate 등): O 를 32배수로 pad 후 slice(정확).
- **정밀도**: DPE 는 bf16 systolic(~0.23% rel, atol 1e-2). `QCN_DPE` 미설정 = 기본 VE(HF 와 1e-7).

산출물: `dn_chunk_full_dpe2.yaml`·`validate_moe_dpe.py`·`validate_deltanet_dpe.py`·`perf_dpe.json`·`generation_sample_dpe.json`.

→ **DPE = 검증된 진짜 가속 레버(배치-벡터엔진 퇴행과 대조). 서버는 `QCN_DPE=1` 로 4.69×/1.59× 빠르게.**

---

## 6. 실제 80B 모델 end-to-end + 남은 한계

> 손수 작성한 NPU DeltaNet 커널을 실제 **Qwen3-Coder-Next-FP8** 가중치로 엮어 host 추론 루프로
> 텍스트 생성 → OpenAI 호환 서빙까지. 상세 청사진: [README_qwen3_next_RUN.md](README_qwen3_next_RUN.md).

### 6-1. 용량 가능성 (✅ 253GB 로 확정)

| 자원 | 필요 | 보유 | 판정 |
|---|---|---|---|
| 디스크 | FP8 80.4GB 다운로드 | 253GB 여유 | ✅(여유 ~170GB) |
| 호스트 RAM | 80GB(FP8) — mmap 이면 touched layer 만 | 125GB(115 여유) | ✅(safetensors mmap, 레이어별 dequant) |
| NPU HBM | per-layer 컴퓨트만 상주(가중치는 host→NPU 스트리밍) | 47.5GB×4 | ✅(한 레이어 분량 ≪ 47.5GB) |

**핵심:** host 추론 루프는 전체 가중치를 NPU 에 다 올리지 않음. host(mmap)가 가중치를 보유,
**레이어별로** 필요한 가중치를 dequant→NPU 로 보내 컴퓨트 후 결과만 회수 → 80GB 모델도 47.5GB 카드로 동작.

### 6-2. Host 추론 루프 아키텍처

```
입력 토큰들 → embed_tokens(host, 비양자화 가중치)
 → for layer i in 0..47:
      ┌─ 입력 RMSNorm
      ├─ 토큰 믹서:
      │    i가 DeltaNet(36개) → in_proj(dn_linear) → split q,k,v,z,b,a → conv1d+SiLU(dn_conv1d)
      │        → l2norm q,k(dn_l2norm) → beta=sigmoid(dn_gate), g(host softplus)
      │        → [prefill] 청크 스캔(dn_chunk_full, 청크간 S carry)
      │          [decode] 순환 스텝(dn_step_mh) — host가 S·conv_state 보유
      │        → gated RMSNorm(dn_gnorm) → out_proj(dn_linear)
      │    i가 full-attn(12개) → q/k/v proj(dn_linear) → q/k RMSNorm → RoPE(host)
      │        → SDPA(matmul 커널) → o_proj(dn_linear). KV는 host append.
      ├─ post RMSNorm
      └─ MoE FFN: router(dn_linear)→top-10 게이팅(host)→선택 expert gate/up/down proj
            (dn_linear, FP8 dequant) + shared expert. SwiGLU.
 → final RMSNorm → lm_head(host 비양자화) → logits → 샘플링 → 다음 토큰 → 반복
```

**상태 관리(DeltaNet 핵심):** prefill 은 프롬프트를 청크(예: 64토큰)로 나눠 `dn_chunk_full` 처리,
청크간 S 를 host carry. decode 는 토큰마다 `dn_step_mh`, **host 가 S[32,128,128]·conv_state[conv_dim,3]
를 torch 텐서로 보유**(진짜 read-modify-write). full-attn 레이어 KV 도 host append → serve 런타임
append-only paged-KV 한계 우회.

**FP8 처리:** 가중치는 FP8 blockwise(128×128 스케일). 레이어 로드 시 host dequant→bf16/fp32
후 `dn_linear` 입력(우리 커널은 fp32 검증). 비양자화(lm_head·embed)는 그대로.

### 6-3. 실행 진행 (qcn/, 2026-06-11) — end-to-end 성공

- ✅ **다운로드**: Qwen3-Coder-Next-FP8 75GB, 40/40 shard(HF 캐시).
- ✅ **가중치 로더** `qcn/loader.py`(safetensors mmap + FP8 blockwise dequant) — 검증.
- ✅ **컴포넌트(실가중치 HF 대조, 전부 NPU `_dfg_inner=0`):**
  - DeltaNet 레이어 `qcn/deltanet_layer.py`(16/32헤드·d128): vs HF **maxerr 8.9e-8**, out 32×2048·state 32×128×128.
  - Full-attn 레이어 `qcn/attn_layer.py`(GQA 16/2·head_dim 256·partial RoPE·게이트): vs HF **5.96e-7**,
    **matmul 100% NPU**(q/k/v/o proj+16×q@kᵀ+16×attn@v).
  - MoE `qcn/moe.py`(512 experts top-10+shared): vs HF **1.79e-7**, 97% NPU, 73/512 expert 활성.
  - `dn_linear` HW 한계(처리): weight read O·I ≤ ~2²⁰(출력축 타일링), **출력축 32배수**(zero-pad), token축 ≥128(pad).
- ✅ **전체 forward** `qcn/model.py`(48레이어 가중치 스트리밍): 첫 4레이어 실 HF Qwen3NextModel 대조
  **maxerr ~1e-6**(layer0~3 post-residual 2.4e-7~8.9e-7, after-norm 8.6e-6), 전 mixer NPU
  `_dfg_inner=0`. **RMSNorm 규약 정정: 디코더/q·k norm 은 `(1+weight)`, DeltaNet gated norm 만
  plain weight.** dynamo recompile_limit 상향 필수(멀티-shape).
- ✅ **decode 루프 + 생성** `qcn/generate.py`: 프롬프트 `def quicksort(arr):` → **생성**
  `\n    if len(arr) <= 1:\n        return arr\n    else:\n        pivot = arr[0]\n`(문법적으로
  올바른 quicksort). prefill 360s, decode 24토큰 avg 55.8s/tok, NPU stages deltanet=36000·attn=10824·
  moe=59336, **CPU폴백 0**. prefill↔decode 일관성 1e-7(상태 S·conv tail·KV 캐시 스레딩 정확).
  샘플 `qcn/generation_sample.json`.
- ✅ **serve 래퍼(A안)** `qcn/serve.py`(FastAPI OpenAI 호환): 모델 1회 로드 + 요청 lock 직렬화.
  - `/v1/completions`: `"def add(a, b):"` → `"\n    return a"`(OpenAI 형식, usage).
  - `/v1/chat/completions`: Qwen chat 템플릿, `"Write a Python one-liner to sum a list."` → `"```python\ntotal"`.
  - 실행: `PYTHONPATH=<proj> RNGD_DEV=rngd:2 ~/furiosa/bin/python qcn/serve.py`(포트 8900).
    성능: 첫 요청 prefill 컴파일 ~360s, 정상 ~55s/tok(`QCN_DPE=1` 이면 4.69×/1.59× 빠름, 5-3절).

→ **🏆 실제 Qwen3-Coder-Next-FP8(80B)가 RNGD NPU 에서 우리 손수 커널로 end-to-end 코드 생성 + OpenAI 호환 API 서빙 성공.**

### 6-4. Serve 통합 3안 + 남은 단 하나의 한계

prefill 단일-forward + decode 상태를 host 가 들면 **생성은 host 루프로 가능**. 프로덕션 서빙은:

- **A안 (host 루프 + 경량 API 래퍼)**: host 루프 위에 FastAPI `/v1/chat/completions` 래퍼. 단일/소수
  요청 OK. 연속배칭·paged-KV 재사용은 우리가 구현(중간 난이도). **벤더 불필요.** ← 채택, 실동작(6-3).
- **B안 (furiosa-llm serve 안 통합)**: decode cross-step 순환상태 풀이 닫힌 Rust 런타임 소유 →
  **벤더(2026.3+) 전용.** 코드 정독 후 확정한 실측 근거:
  - ❌ **DeltaNet 상태를 KV-캐시로 위장**: 정적 게이트(`specs/inputs.py:69` k.shape==v.shape,
    `utils.py:227` is_kvcache 이름regex)는 통과하나 **런타임이 슬롯 인덱스를 append 식으로 소유**
    (`paged_attention.py:126` cache[idx]=val, idx 는 runtime scheduler 가 토큰마다 새 블록,
    block_size=1 고정) → 상태가 매 스텝 다른 슬롯에 흩어져 **read-modify-write 불가**.
    output→next-input aliasing 은 닫힌 Rust KVCachePlan 소유(Python 훅 없음). SSM/mamba/conv_state
    기구 furiosa_llm/런타임에 **전무**. DeltaNet 파티셔너 지원도 없음(`graph_partitioner.py:130` IndexError).
- **C안 (하이브리드)**: full-attn+MoE 는 furiosa-llm serve(qwen3_moe 위장), DeltaNet 만 우리 커널 —
  단 serve 가 레이어별 커스텀 커널 주입을 안 받음 → 사실상 A안으로 수렴.

**남은 큰 일:** ① 투영/softplus/사전계산까지 NPU(거의 matmul, easy; softplus·tri-inverse·cumsum 은
순차/log 한계) ② 처리량 최적화(연속배칭·상주·FP8 온칩 — 정확성 우선, 후순위). ③ **furiosa-llm
serve 안 네이티브 통합(B안)만 벤더 몫.**

### 6-5. 벤더 요청 대상 네이티브 바이너리

| pip 패키지(2026.2.0) | .so 파일(site-packages/furiosa/) | 역할 | qwen3_next 에 필요한 작업 |
|---|---|---|---|
| `furiosa-native-runtime` | `native_runtime.cpython-312-x86_64-linux-gnu.so`(163MB) | serve 엔진 전체: model_type 게이트(`furiosa-generator/src/next_gen/hf_compat_next_gen.rs:367`), 스케줄러·preset, KV 캐시 풀(`host_kv_cache_pool.rs`), prefix cache | ① 게이트 enum 에 qwen3_next 추가 ② **선형어텐션 순환상태(conv+recurrent) 요청별 풀 관리** ← 최대 작업 ③ 스케줄 preset ④ prefix cache 상태 인지 |
| `furiosa-native-llm-common` | `native_llm_common.cpython-312-x86_64-linux-gnu.so`(143MB) | 빌드측 게이트 `find_compiler_config(model_type, task)` + 컴파일러 설정 테이블(`hf_config.rs` 파서 — layer_types 의 `linear_attention` 미지원) | ① 컴파일러 설정 테이블에 qwen3_next 추가 ② hf_config 파서 layer_types 에 linear_attention 허용 |
| `furiosa-torch` | `native_torch.cpython-312-x86_64-linux-gnu.so`(105MB) | NPU 컴파일러(op lowering) | (참고) DeltaNet chunked prefill 용 fp32 cumsum·constant_pad_nd·부분 slice_scatter 지원 시 prefill 성능 개선 — 순환식 prefill 은 현 op 셋으로도 가능 |

요청 문구 예: "2026.2.0 의 furiosa-native-runtime / furiosa-native-llm-common 에 model_type
`qwen3_next`(Qwen3-Next/Qwen3-Coder-Next, hybrid Gated DeltaNet + gated attention + MoE) 지원
추가 계획이 있는지, 특히 **디코드 스텝 간 순환 상태(레이어당 conv state + recurrent state) 풀
관리**가 로드맵에 있는지 문의. 빌드 측 Python 아키텍처는 자체 구현 보유. **컴퓨트 NPU 커널은
TacticKernelModule 로 자체 작성·검증 완료**(HF ~1e-6)."

### 6-6. 리스크 / 미지수

- FP8 dequant 정확도(blockwise 128×128) — dequant 후 우리 fp32 커널은 검증됨.
- 성능: host↔NPU 레이어별 왕복은 느림(decode 토큰당 48레이어 × 커널 수). 정확성 우선, 성능은
  DPE(5절)·배칭·상주·FP8 온칩으로 최적화.
- full-attn(head_dim 256)·MoE(512 experts) 커널 스케일 — 원리는 matmul, 스케일만.
- 768토큰 이상 긴 prefill 의 청크 수 증가 — 청크 스캔 O(T/C) 라 OK.
- lm_head(vocab 151936)·embed 큰 matmul — host 또는 NPU 분할.

---

## 부록 A. 재현용 radare2 명령 모음

```bash
# 환경 (소스 빌드 radare2)
cd Model_Benchmark/qwen3-next-proj/radare2
LIBS=$(find $PWD/libr -name '*.so' | xargs -n1 dirname | sort -u | tr '\n' ':')
R2="LD_LIBRARY_PATH=$LIBS $PWD/binr/radare2/radare2 -2"
SO=~/furiosa/lib/python3.12/site-packages/furiosa/native_llm_common.cpython-312-x86_64-linux-gnu.so

# 변형 테이블·VARIANTS·매처
eval $R2 -q -c 'izz~ModelType' $SO                       # 변형 블롭
eval $R2 -q -c 'e search.in=io.maps; / qwen3_moe' $SO    # 변형 주소
eval $R2 -q -c 'e search.in=io.maps; /x 17a2ac0000000000; pxq 96 @ hit0_0' $SO  # VARIANTS 배열
eval $R2 -q -c 'e anal.in=io.maps; aar; axt 0x00aca228' $SO          # 매처 단말 찾기
eval $R2 -q -c 'e anal.in=io.maps; s 0x1fcd350; pd 60' $SO           # 매처 디스어셈블
```

## 부록 B. 산출물·아티팩트 경로 모음

- 손수 커널·드라이버: `qwen3-next-proj/tk_kernels/`
  - 스텝/순환: `dn_step.yaml`·`dn_step_mh.yaml`·`dn_rank1.yaml`·`dn_decay.yaml`·`dn_delta.yaml`·`dn_gate.yaml`·`dn_einsum.yaml`·`dn_einsum_f32.yaml`
  - 언롤: `dn_prefill_unroll4.yaml`·`dn_prefill_unroll{8..128}.yaml`(`gen_unroll.py`)
  - 청크: `dn_chunk.yaml`·`dn_chunk_full.yaml`·`dn_chunk_full_mh.yaml`(`gen_chunk.py`·`gen_chunk_full.py`)
  - 레이어 조각: `dn_conv1d.yaml`·`dn_l2norm.yaml`·`dn_gnorm.yaml`·`dn_linear.yaml`·`dn_linear_be.yaml`
  - DPE: `dn_linear_dpe.yaml`·`dn_chunk_full_dpe2.yaml`(+ `dpe_serde_fields.md`·`dpe_incremental_log.md`·`dpe_struct_from_dump.md`·`dpe_result.md`·`bench_dpe_vs_ve.py`·`perf_dpe.json`)
  - Loop frontier: `dn_loop.yaml`·`dn_loop2.yaml`
  - probe/test 드라이버: `probe_unary.py`·`probe_binary.py`·`run_dn_step.py`·`host_loop_test.py`·`unroll4_test.py`·`mh_test.py`·`unroll_limit_test.py`·`scale_test_d128.py`·`run_dn_chunk_full.py`·`gen_dn_layer.py`·`test_dn_layer.py`·`test_dn_linear.py`·`full_layer.py`·`full_layer_npu.py`
- 실모델 host 추론 루프: `qwen3-next-proj/qcn/`
  - `loader.py`·`deltanet_layer.py`·`attn_layer.py`·`moe.py`·`model.py`·`generate.py`·`serve.py`
  - 검증 샘플: `generation_sample.json`·`generation_sample_dpe.json`·`validate_moe_dpe.py`·`validate_deltanet_dpe.py`
- 위장 도구·아티팩트: `qwen3-next-proj/masquerade_artifact.py`,
  `qwen3-next-proj/artifacts/qwen3-coder-30b-masq`,
  `rngd-npu/artifacts/qwen3-coder-30b-a3b-inst-fp8-tp8-65k-tc`,
  `qwen3-coder-30b-a3b-inst-tp8-65k-tc`
- radare2: `qwen3-next-proj/radare2/`(소스 빌드)

---

## 7. (2026-06-15) NPU 실행 재증명 + DeltaNet 분해 컴파일

이 절은 세 가지 확인 작업의 결과입니다: (7-1) 토큰 생성이 정말 NPU에서 일어나는지
CPU util까지 동원해 다시 증명, (7-2) DeltaNet 컴파일을 직접 쪼개 a6 번들에 넣기,
그리고 (배경) gated DeltaNet/attention 이론은 별도 문서 `README_gated_deltanet_STUDY.md`
로 정리했습니다.

### 7-1. "정말 NPU가 토큰을 만드는가" — 4중 증명 (실측)

실험: `Qwen3-Coder-Next-FP8` 80B 를 host 추론 루프로 `rngd:4`(npu0 core4)에서 3토큰
생성하면서, 별도 프로세스가 `furiosa-smi` 와 host CPU 를 동시에 샘플링했습니다.

생성 결과는 정상이었습니다 — prompt 20토큰 prefill 142.9s, 디코드 44.1/42.8/43.0 s/tok,
출력 `"Here's a"`(코드 작성 프롬프트의 자연스러운 시작), prefill top-5 `Here/```/...`.

증명은 네 갈래로 교차합니다.

1. **소프트웨어 폴백 카운터 = 0.** `furiosa.torch.custom_ops.dfg._dfg_inner` 는 NPU 가
   아닌 **CPU 전용 폴백 경로**입니다(`deltanet_layer.py:45-52` 가 이 함수를 spy 로 감싸
   호출 횟수를 셈). NPU 실행 시 이 경로는 우회되므로 카운터가 0 이면 순수 NPU 입니다.
   3토큰 생성 동안 DeltaNet/attn/MoE 합쳐 **5760회 NPU 커널 디스패치, 폴백 카운터 0**
   (deltanet=0, attn=0, moe=0 → `ALL_ON_NPU=True`).
2. **디바이스 런타임이 PID 를 NPU core 에 바인딩.** `furiosa-smi ps` 는 생성 구간
   (t≈13~283s) 내내 `<드라이버 PID>@npu0:4` 를 보여 주다가, 드라이버 종료 직후 빈칸이
   됩니다(유휴 시엔 ps 가 비어 있음). 런타임이 디바이스를 **점유**했다는 것 자체가 NPU
   실행 신호입니다 — CPU 폴백만 했다면 디바이스를 안 잡아 ps 에 안 뜹니다.
3. **CPU 가 바쁜 것은 폴백이 아니라 host 오케스트레이션이다.** 생성 중 host CPU 25~44%,
   드라이버 프로세스 %CPU 가 최대 ~3415%(≈34코어)로 매우 바빴지만, 이는 **weight 레이어별
   스트리밍 + FP8 blockwise 역양자화 + torch.compile 글루** 때문입니다(설계상 host 작업).
   1번(폴백=0)이 이 CPU 사용이 폴백이 아님을 증명합니다. 즉 "CPU 가 바쁘다 ≠ CPU 가
   연산한다".
4. **NPU 하드웨어 수치 지문(0.23%).** `bench_dpe_vs_ve.py`(독립 코드 경로)로 같은 PE 에서
   DPE(systolic) matmul 을 검증하면 torch `F.linear` 대비 rel **0.23%** 오차가 나옵니다 —
   이는 NPU 의 bf16 systolic MAC 엔진(EinsumByDpe) 고유 지문입니다. CPU fp32 였다면 VE
   경로처럼 ~3e-7 이어야 합니다(실측 VE 경로 maxabs 2.98e-7). 우리 프로덕션 생성은
   `QCN_DPE=1` 이라 이 DPE 커널을 쓰므로 출력이 같은 하드웨어 지문을 가집니다. bench 도
   폴백=0 으로 순수 NPU 재확인.

**왜 power/util 은 평탄(39W, core 0~1.7%)이었나 (정직한 해석).** furiosa-smi 의 power·
core-util 은 이 커널 입도에서는 **둔감한 지표**입니다. 대조 실험: 4000회 타이트 matmul
루프(`bench_dpe_vs_ve.py`)에서도 power 38~39W·core-util 0.0% 로 평탄했습니다 — per-call
커널이 작고 Python/전송 오버헤드가 duty-cycle 을 지배하기 때문입니다. 즉 39W 평탄은
"NPU 미사용"이 아니라 "지표가 짧은 버스트를 평균내며 못 잡음 + host-bound"의 결과이고,
신뢰할 신호는 위 1·2·4(폴백=0, PID 바인딩, 하드웨어 지문)입니다. 이 host-bound 특성은
이전의 "멀티카드 data-parallel 처리량 무효" 결론과도 일치합니다.

### 7-2. DeltaNet 컴파일을 직접 쪼개 a6 번들 완성 (돌파)

이전까지 binary_bundle 은 17블롭 `partial-edf` 였고, **3조각이 a6 불가**로 빠져 있었습니다:
`deltanet_recurrent_step`(라벨 충돌), `dn_conv1d_silu`(O136), `dn_gate`(log1p). 이번에
"통과한 다른 것들은 어떻게 쪼개졌나"를 분석해 직접 분해를 시도했습니다.

**핵심 관찰(실측 `tk_kernels/` 의 분해 프로브).** 통과한 `full_attn_sdpa` 는 3D
`torch.matmul`(=bmm) 단일 패턴, `moe_expert_swiglu`/`Linear` 도 단일 깔끔한 연산이었습니다.
recurrent step 은 외적·축소·감쇠가 한 그래프에 섞여 있었던 게 문제였습니다.

| 조각 | a6 결과 |
|---|---|
| recurrent step 원본(broadcast) | FAIL `conflict between concrete labels: Concrete(3)/(1)` |
| recurrent step bmm 재작성(통째로) | **FAIL** (같은 라벨 충돌) |
| `sub_outer`(외적 k⊗δ, bmm) | **OK a6** |
| `sub_contract`(축소 q@state, bmm) | **OK a6** |
| `sub_decay`(state×α) | **OK a6** |
| `nn.Conv1d` depthwise | FAIL `O136 not supported` |
| `conv_shift`(host-pad+shift-mul-add+SiLU) | **OK a6** |
| `softplus` | FAIL `log1p not importable` |
| `log(1+exp)` 통째(+sigmoid 2출력) | FAIL `multiple internal subgraphs` |
| `exp`/`log(1+x)`/`sigmoid` 개별 | **OK a6** |

**결론: 막힌 3조각의 진짜 원인은 연산 자체가 아니라 "그래프 구성"이었다.** a6 컴파일러는
(a) 한 그래프 안에 **복수의 contraction 패턴**이 섞이면 라벨 통일에 실패하고, (b) 한
모듈이 **복수의 독립 출력 서브그래프**를 가지면 거부하며, (c) 일부 op(Conv1d→O136,
softplus→log1p)는 미지원입니다. **"연산 1개 = 그래프 1개"로 쪼개면** 통과한 Linear/SDPA/
SwiGLU 와 같은 단일패턴 그래프가 되어 전부 a6 통과합니다.

그래서 recurrent step 을 5개 a6 블롭으로 분해했습니다(별도 프로브로 수치 검증):

```
state = state * α            # dn_recur_decay   (per-head scalar 감쇠)
kv    = k @ state            # dn_recur_contract(bmm 축소)
delta = (v - kv) * β         # dn_recur_delta   (elementwise)
state = state + k ⊗ delta    # dn_recur_outer(bmm 외적) + dn_recur_add
out   = q @ state            # dn_recur_contract(재사용)
```

쪼갠 시퀀스는 원본 broadcast recurrent step 과 **fp64 에서 1.4e-12(실수 연산상 정확),
fp32 에서 rel 2.9e-7**(reduction 순서 차이일 뿐). conv1d 는 `dn_conv1d_shift`(host 가
좌측 K-1 zero-pad → NPU 는 4회 shift-mul-add + SiLU, constant_pad_nd 회피), gate 는
`dn_gate_beta`=sigmoid(b) + `dn_gate_g`=-exp(A_log)·log(1+exp(a+dt)) 로 분해했습니다.

**통합 결과.** 이 8개 분해 블롭을 실제 config 차원으로 컴파일해 아티팩트에 추가했습니다
(`tk_kernels/emit_dn_split_blobs.py` → `pack_edf_bundle.py` 재패킹). binary_bundle 은
**17블롭 → 25블롭**, kind `partial-edf` → **`edf-split (compute-complete)`**,
`pieces_without_edf` **3 → 0**, 25/25 전부 `CompiledGraph.deserialize` a6 검증 통과
(515.3MB, sha256 `d740ea47…`).

**남은 한계는 컴파일이 아니라 deploy 입니다.** 이제 모든 컴퓨트가 a6 EDF 로 번들됐지만,
furiosa-llm `serve` 런타임은 여전히 (1) cross-step 순환상태 풀이 없고(paged-KV 는
append-only K/V, 순환은 read-modify-write), (2) 이 분해 sub-op 들을 네이티브로 체이닝할
방법이 없습니다. 그래서 host 추론 루프 아티팩트로 남으며, 분해 블롭들을 체인 serve
파이프라인으로 굴리려면 벤더 런타임(2026.3+)이 필요합니다. (이론·이유 상세:
`README_gated_deltanet_STUDY.md` §4.)
