# Qwen3-Coder-Next를 RNGD NPU에서 돌리고 쓰는 법 (실행·사용 가이드)

작성 2026-06-11 · RNGD SDK 2026.2.0 · 4×RNGD(각 47.5GiB HBM, 펌웨어 2026.2.1) · host 125GB RAM / 253GB 디스크
관련: [README_qwen3_next_TECH.md](README_qwen3_next_TECH.md)(커널·DPE·serve 게이트의 깊은 원리),
[README_all_change.md](README_all_change.md)(변경 이력)

> 이 문서는 **실제 Qwen3-Coder-Next-FP8(80B) 전체 모델을 RNGD NPU에서 돌리고 쓰는 실전 방법**을
> 정리합니다. 손수 작성한 NPU 커널을 실제 가중치와 엮어 ① host 추론 루프로 텍스트를 생성하고
> ② OpenAI 호환 API로 서빙하는 전 과정입니다. 커널 내부 DSL, DPE 엔진 역설계, serve 게이트
> 바이너리 분석 같은 깊은 원리는 동반 문서 [README_qwen3_next_TECH.md](README_qwen3_next_TECH.md)를
> 참고하세요.

---

## 1. 무엇인가 — host 추론 루프 + 손수 작성한 NPU 커널

이 모델은 **furiosa-llm build로 만든 아티팩트가 아닙니다.** 대신 host(파이썬)가 추론 루프를
들고, 레이어마다 필요한 컴퓨트를 **손수 작성한 TacticKernelModule 커널**로 NPU에 보내 실행한 뒤
결과만 회수하는 구조입니다.

**왜 furiosa-llm build가 아닌가:**

- `Qwen/Qwen3-Coder-Next`(model_type `qwen3_next`)의 충실한 빌드·서빙은 **2026.2.0에서 불가능**
  합니다. 막는 것은 "연산 미지원"이 아니라 **구조**입니다.
  - (a) **빌드 측 TP 그래프 분할기**가 모든 레이어가 페이지드 (K,V) 캐시를 소비하도록 가정 →
    KV를 안 쓰는 DeltaNet 레이어(48개 중 36개)가 dead node를 만들어 분할기가 깨짐. 단 이 벽은
    모델 코드 수정만으로 자력 통과 가능함을 실증(④ TP 분할까지 통과, 자세히는 TECH 문서).
  - (b) **serve 런타임(폐쇄 Rust 바이너리)이 DeltaNet 순환 상태를 디코드 스텝 간 유지·관리할 수
    없음.** append-only paged-KV(매 스텝 새 블록, block_size=1 고정)라 read-modify-write 상태 S를
    들 수 없음. 이것만이 진짜 벤더 몫(2026.3+). 둘 다 미니 모델로 실측 확인.
- 그래서 **컴퓨트 벽은 우리가 뚫었습니다.** `furiosa.torch.TacticKernelModule`로 DFG 커널을
  손수 작성해(Rust 소스 불필요) Gated DeltaNet 레이어를 RNGD NPU에서 실제 계산, HuggingFace
  레퍼런스와 ~1e-7 일치시켰습니다(적대적 검증 통과). DeltaNet 레이어 컴퓨트의 **96.74%가 NPU**
  (matmul FLOP 1,458,176 MAC 중)에서 돕니다.
- 순환(decode 스텝 간 상태)은 serve 런타임이 못 들지만, **furiosa-llm serve "밖"의 커스텀 host
  추론 루프**라면 host가 상태 S·conv_state·KV를 torch 텐서로 보유(진짜 RMW)해 우회할 수 있습니다.
  이 문서가 다루는 게 바로 그 host 추론 루프입니다.

요약하면: **빌드/서빙 게이트는 닫혀 있지만, 컴퓨트 커널은 우리가 작성 가능 → host가 루프를
들고 NPU를 가속기로 쓰는 방식으로 전체 모델이 동작.** 표준 트랜스포머/MoE 부분(full-attn·MoE)도
같은 host 루프에서 손수 커널로 처리합니다(별도 위장 불필요).

> 참고 대안(코더가 목적이고 DeltaNet이 꼭 아니어도 될 때): 사장됐던 **Qwen3-Coder-30B-A3B-Instruct-FP8**
> (Qwen3세대 MoE 코더)을 메타데이터 위장으로 serve 부활시켜 4장 dp에서 단일 63.2 tok/s·동시 32명
> 합산 1036.2 tok/s를 실측했습니다. 위장 방법·실측 표·운영안은 [README_qwen3_next_TECH.md](README_qwen3_next_TECH.md)
> 참고. 이 RUN 문서는 **DeltaNet을 그대로 살린 Qwen3-Coder-Next 전체 모델**에 집중합니다.

---

## 2. 모델과 용량 (Qwen/Qwen3-Coder-Next-FP8)

### 2-1. 실측 모델 스펙 (HF config)

| 항목 | 값 |
|---|---|
| model_type / arch | `qwen3_next` / `Qwen3NextForCausalLM` |
| 파라미터 / 크기 | 79.7B, **FP8 80.4GB** (40 safetensors shard); BF16이면 159.4GB |
| 레이어 | 48 (`full_attention_interval=4` → 12 full-attn + 36 DeltaNet), **매 레이어 MoE** |
| hidden / vocab | 2048 / 151936 |
| DeltaNet | key heads 16, value heads 32(n_rep=2), key/value head_dim 128, conv kernel 4 |
| full attention | 16 heads, 2 kv heads(GQA), head_dim 256, partial rotary 0.25, q·k norm |
| MoE | 512 experts, top-10 + shared expert, moe_intermediate 512, 전 레이어 |
| 컨텍스트 | 262144 |
| 양자화 | FP8 blockwise(weight_block_size [128,128], dynamic act); **lm_head·embed_tokens는 비양자화** |

DeltaNet 레이어는 토큰마다 갱신되는 **고정 크기 순환 상태**(레이어당 recurrent
`(num_v_heads, head_k, head_v)` + causal-conv 상태 `(conv_dim, kernel-1)`)를 유지해야 자기회귀
디코딩이 성립합니다. 트랜스포머의 append-only KV 캐시와는 **접근 패턴이 근본적으로 다릅니다**
(read-modify-write vs append) — 이것이 serve 런타임이 못 드는 이유.

### 2-2. 용량 가능성 (✅ 253GB로 가능 확정)

| 자원 | 필요 | 보유 | 판정 |
|---|---|---|---|
| 디스크 | FP8 80.4GB 다운로드 | 253GB 여유 | ✅ (여유 ~170GB) |
| 호스트 RAM | 80GB(FP8) — mmap이면 touched layer만 | 125GB(115 여유) | ✅ (safetensors mmap, 레이어별 dequant) |
| NPU HBM | per-layer 컴퓨트만 상주(가중치는 host→NPU 스트리밍) | 47.5GB×4 | ✅ (한 레이어 분량 ≪ 47.5GB) |

**핵심:** host 추론 루프는 전체 가중치를 NPU에 다 올리지 않습니다. host(mmap)가 가중치를 보유하고,
**레이어별로** 필요한 가중치를 dequant→NPU로 보내 컴퓨트 후 결과만 회수합니다. 따라서 80GB
모델도 47.5GB 카드 한 장으로 동작합니다(느리지만 정확). 전체 상주·연속배칭이 필요한 프로덕션
serve는 별도이며 벤더 몫(1절 (b)).

---

## 3. 아키텍처 — 48레이어 host 파이프라인

```
입력 토큰들 → embed_tokens(host, 비양자화 가중치)
 → for layer i in 0..47:
      ┌─ 입력 RMSNorm
      ├─ 토큰 믹서:
      │    i가 DeltaNet 레이어(36개) → 우리 커널 파이프라인:
      │        in_proj(dn_linear) → split q,k,v,z,b,a → conv1d+SiLU(dn_conv1d)
      │        → l2norm q,k(dn_l2norm) → beta=sigmoid(dn_gate), g(host softplus)
      │        → [prefill] 청크 스캔(dn_chunk_full, 청크간 S carry)
      │          [decode] 순환 스텝(dn_step_mh) — host가 S·conv_state 보유
      │        → gated RMSNorm(dn_gnorm) → out_proj(dn_linear)
      │    i가 full-attn 레이어(12개) → 표준 어텐션:
      │        q/k/v proj(dn_linear) → q/k RMSNorm → RoPE(host) → SDPA(matmul 커널)
      │        → o_proj(dn_linear). KV는 host가 보유(append).
      ├─ post RMSNorm
      └─ MoE FFN: router(dn_linear)→top-10 게이팅(host)→선택 expert
            gate/up/down proj(dn_linear, FP8 dequant) + shared expert. SwiGLU.
 → final RMSNorm → lm_head(host 비양자화) → logits → 샘플링 → 다음 토큰 → 반복
```

### 3-1. 각 컴포넌트가 NPU에서 어떻게 도는가 (qcn/ 파일 매핑)

전체 모델 코드는 `qwen3-next-proj/qcn/`에 있고, 커널 YAML은 `qwen3-next-proj/tk_kernels/`에
있습니다. 실가중치로 HF와 대조 검증한 컴포넌트:

| 컴포넌트 | qcn 파일 | 사용 커널(tk_kernels) | HF 대조 maxerr | NPU 비중 |
|---|---|---|---|---|
| 가중치 로더(FP8 dequant) | `loader.py` | — | — | host |
| DeltaNet 레이어(16/32헤드·d128) | `deltanet_layer.py` / `deltanet_layer_looped.py` | `dn_linear`·`dn_conv1d`·`dn_l2norm`·`dn_gate`·`dn_chunk_full`·`dn_step_mh`·`dn_gnorm` | **8.9e-8** | matmul 96.74% NPU |
| Full-attn 레이어(GQA 16/2·head_dim 256·partial RoPE·게이트) | `attn_layer.py` | `dn_linear`(q/k/v/o proj) + 16×q@kᵀ + 16×attn@v | **5.96e-7** | **matmul 100% NPU** |
| MoE(512 experts top-10+shared) | `moe.py` | `dn_linear`(gate/up/down) | **1.79e-7** | 97% NPU(73/512 expert 활성) |
| 전체 모델(48레이어 스트리밍) | `model.py` | 위 전부 | 첫 4레이어 vs HF **~1e-6** | 전 mixer NPU `_dfg_inner=0` |
| 생성 루프(prefill+decode) | `generate.py` | 위 전부 | — | CPU폴백 0 |
| serve 래퍼(OpenAI API) | `serve.py` | 위 전부 | — | — |

**상태 관리(DeltaNet의 핵심):**

- **prefill:** 프롬프트를 청크(예: 64토큰)로 나눠 `dn_chunk_full`로 처리, 청크간 상태 S를 host가
  carry(NC=3 청크 검증, inter-chunk carry 실검증 S_prev≠0).
- **decode:** 토큰마다 `dn_step_mh` 순환 스텝, **host가 S[32,128,128]·conv_state[conv_dim,3]를
  torch 텐서로 보유**(진짜 read-modify-write). full-attn 레이어 KV도 host가 append. 이로써 serve
  런타임의 append-only paged-KV 한계를 우회.
- prefill↔decode 일관성 1e-7(상태 S·conv tail·KV 캐시 스레딩 정확).

**중요 규약·제약(반드시 지킬 것):**

- **RMSNorm 규약:** 디코더 RMSNorm과 q·k norm은 `(1+weight)`, DeltaNet gated norm만 plain weight.
  (full-model에서 정정한 핵심 버그.)
- **dn_linear HW 한계(처리됨):** weight read O·I ≤ ~2²⁰(출력축 타일링), 출력축 32배수(zero-pad),
  token축 ≥128(pad).
- **dynamo recompile_limit 상향 필수**(멀티-shape). full-model은 레이어마다 shape이 달라짐.
- **FP8 처리:** 가중치는 FP8 blockwise(128×128 스케일). 레이어 로드 시 host에서 dequant→bf16/fp32
  후 `dn_linear` 등에 입력(우리 커널은 fp32 검증). 비양자화(lm_head·embed)는 그대로.

> 커널 내부 DSL(외적은 EinsumByVe, contraction은 Reduce inst, head는 batch축, 유효 Unary는
> Exp/Sigmoid/Sqrt만, rsqrt는 `√s/s`로 구현, 융합 inst ≤2개, l2norm/gnorm은 행<128이면 128로
> zero-pad 등)과 멀티청크 스캔의 삼각역행렬 사전계산 honor 설계는 [README_qwen3_next_TECH.md](README_qwen3_next_TECH.md)
> 참고.

---

## 4. 가중치 위치와 로더 (FP8 dequant)

- **다운로드:** Qwen3-Coder-Next-FP8 ~75GB, 40/40 shard. HF 캐시에 저장(safetensors).
- **로더:** `qcn/loader.py` — safetensors **mmap** + **FP8 blockwise dequant**(128×128 스케일) +
  레이어별 가중치 추출. touched layer만 RAM에 올라오므로 80GB 모델도 125GB host에서 동작.
- dequant 후 우리 fp32 커널은 검증됨(FP8 blockwise 128×128 정확도 확인). lm_head(vocab 151936)·
  embed_tokens는 비양자화라 그대로 사용.

---

## 5. 실행법 — generate.py / serve.py

프로젝트 루트: `/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj` (이하 `<proj>`).
파이썬: `~/furiosa/bin/python`. NPU 디바이스는 `RNGD_DEV` 환경변수로 지정.

### 5-1. 텍스트 생성 (generate.py)

```bash
PYTHONPATH=<proj> RNGD_DEV=rngd:2 ~/furiosa/bin/python qcn/generate.py
```

실측 동작: 프롬프트 `def quicksort(arr):` →
```
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
```
(문법적으로 올바른 quicksort). 샘플 출력: `qcn/generation_sample.json`.

### 5-2. OpenAI 호환 API 서빙 (serve.py)

`qcn/serve.py`는 FastAPI OpenAI 호환 서버입니다. 모델을 1회 로드하고 요청을 lock으로 직렬화합니다.

```bash
PYTHONPATH=<proj> RNGD_DEV=rngd:2 ~/furiosa/bin/python qcn/serve.py   # 포트 8900
```

엔드포인트(실측):

- `/v1/completions`: `"def add(a, b):"` → `"\n    return a"` (OpenAI 형식 + usage 필드).
- `/v1/chat/completions`: Qwen chat 템플릿 적용. `"Write a Python one-liner to sum a list."`
  → `"```python\ntotal"`.

### 5-3. ⚡ DPE 가속 켜기 (`QCN_DPE=1`) — 권장

`QCN_DPE=1` 플래그 하나로 attn·moe·deltanet의 모든 matmul(proj·out_proj·QK/AV·MoE
gate/up/down·DeltaNet 스캔)을 systolic MAC 엔진(EinsumByDpe)으로 전환합니다. **정확성을 보존하면서
prefill 4.69×·decode 1.59× 빨라집니다**(6절 표). 서버·생성 둘 다에 적용:

```bash
PYTHONPATH=<proj> RNGD_DEV=rngd:2 QCN_DPE=1 ~/furiosa/bin/python qcn/serve.py
PYTHONPATH=<proj> RNGD_DEV=rngd:2 QCN_DPE=1 ~/furiosa/bin/python qcn/generate.py
```

- `QCN_DPE` 미설정 = 기본 VE(벡터 엔진), HF와 1e-7로 정확. `QCN_DPE=1` = DPE(bf16 systolic,
  atol 1e-2, 빠름). 실모델이 FP8/bf16이라 DPE 정밀도가 오히려 더 충실합니다.
- DPE 엔진 역설계 레시피·제약(per-graph 2개 cap, 출력축 O=1 거부)은 7절·TECH 문서 참고.

> **주의:** 첫 요청은 prefill 컴파일에 시간이 걸립니다(VE 기준 ~360s, DPE 기준 ~76.9s). 이후 정상
> 속도로 동작합니다. 정확성 우선 설계이며 처리량 최적화(연속배칭 등)는 후순위입니다.

---

## 6. 성능과 정확성

### 6-1. 성능 (실측, end-to-end)

| 지표 | VE baseline | DPE (`QCN_DPE=1`) | 가속 |
|---|---|---|---|
| Prefill (24토큰 프롬프트) | 360.8s | **76.9s** | **4.69×** |
| Decode | 55.8 s/tok | **35.15 s/tok** | **1.59×** |
| 투영 matmul 512→2048 | 13.09ms | 1.44ms | **9.06×** |
| expert gate 2048→512 | 5.48ms | 2.47ms | 2.21× |

(VE baseline 별도 실측: prefill 360s, decode 24토큰 avg 55.8s/tok; NPU stages
deltanet=36000·attn=10824·moe=59336; CPU폴백 0.)

**왜 baseline이 느린가(병목 = NPU 디스패치 횟수):** 토큰당 NPU stage가 deltanet 36000·moe
59336(prefill+24토큰 누적)에 달합니다. DeltaNet은 32 value-head를 헤드당 `dn_chunk_full`로 루프
호출하고, MoE는 top-10 active expert를 하나씩 호출하기 때문입니다.

**decode가 1.59×로 prefill(4.69×)보다 modest한 이유:** decode는 일부 **host-bound**입니다
(DeltaNet 32헤드 cumsum·삼각역행렬 T-행렬, MoE 라우팅) — matmul만 빨라지고 host 부분은 그대로.

#### 음성 결과 vs 양성 결과 — 속도 레버는 배치가 아니라 DPE

- **❌ 배치(batch-axis) 시도는 퇴행(REGRESSION):** 헤드/expert를 최외곽 batch축으로 묶어 디스패치
  수를 줄이는 시도. 디스패치는 줄였으나(DeltaNet 32→1=32×, MoE 298→10=29.8×, 둘 다 정확성 유지)
  **wall-clock은 오히려 ~16× 느려짐**(레이어당 ~7.5s → ~2분). 원인(아키텍처): RNGD의 VE matmul은
  벡터 엔진(EinsumByVe = broadcast-multiply-reduce)이라 `[.,o,i]` outer product 전체를 materialize
  한 뒤 i로 reduce함 → N개 헤드/expert를 배치하면 한 op이 N배 데이터를 materialize(systolic이
  아니라 처리량 이득 없이 일감만 N배 + SRAM 압박). MoE 실측: 배치된 gate matmul 1회(E=10) ~89s =
  baseline 토큰 전체(55.8s)보다 큼. → 검증된 per-head/per-expert 경로로 복원(`model.py`가
  `DeltaNetLayerLooped` + `moe_forward_npu_unbatched` 사용; 55.8s/tok, end-to-end 검증된 유일
  config). 배치 버전은 참고용 보존(`deltanet_layer.py`·`moe.moe_forward_npu`·`dn_chunk_full_mh.yaml`·`dn_linear_be.yaml`).
- **✅ DPE는 양성(진짜 가속):** EinsumByDpe(실제 MAC/systolic matmul 엔진)는 VE 대비 compute 3.8×·
  end-to-end 1.96×(단일 matmul 기준), 모델 전체로는 위 표의 4.69×/1.59×. 배치-벡터엔진 퇴행과
  대조되는, 검증된 진짜 속도 레버입니다.

**진짜 속도 레버 정리:** ① `QCN_DPE=1`(systolic matmul) ② 4장 멀티카드 ③ 벤더 serve. 배치-벡터엔진은 답이 아님.

### 6-2. 정확성

- **VE 기본 경로:** 전체 forward 첫 4레이어 실HF Qwen3NextModel 대조 **maxerr ~1e-6**(layer0~3
  post-residual 2.4e-7~8.9e-7, after-norm 8.6e-6), 전 mixer(DeltaNet·full-attn·MoE) NPU
  `_dfg_inner=0`. 컴포넌트별 maxerr는 3-1절 표(DeltaNet 8.9e-8, attn 5.96e-7, MoE 1.79e-7).
- **DPE 경로:** 생성 코드 여전히 올바른 quicksort(첫 4토큰 byte-identical), 4레이어 HF 대조 atol
  1e-2 통과(per-layer maxerr ≤2.5e-2), attn layer-3 maxerr 5.5e-3·MoE layer-0 7e-4 vs HF. 전부 NPU
  `_dfg_inner=0`. DeltaNet 스캔은 DPE 2개 cap 때문에 `dn_chunk_full_dpe2.yaml`로 maxerr 2.5e-4 통과.
- **생성 결과 일관성:** `def quicksort(arr):` 프롬프트가 VE/DPE 둘 다 문법적으로 올바른 quicksort를
  생성, 첫 4토큰 byte-identical. DPE 샘플: `qcn/generation_sample_dpe.json`.

> DPE 정밀도(~0.23% rel, bf16 systolic)·2개 cap·출력축 패딩 등 제약과 그 근거는 7절·TECH 문서.

---

## 7. 한계와 다음 단계

### 7-1. DPE 적용 제약 (실측)

- **① DPE per-graph 2개 cap:** 한 TacticKernel 그래프에 `EinsumByDpe` 3개+면
  `fuse_mamma_to_single_einsum_by_dpe` 융합 패스가 systolic array를 오스케줄해 **조용히 오컴파일**
  (dfg=0이나 garbage, maxerr 0.5). DeltaNet 스캔(5 matmul)은 2개만 DPE(`dn_chunk_full_dpe2.yaml`,
  maxerr 2.5e-4 통과), 나머지 VE. 전체 DPE화하려면 그래프를 ≤2-DPE 단위로 분할 필요(YAML 필드가
  아니라 파티셔닝 문제).
- **② DPE 출력축 O=1 거부**(shared_expert_gate 등): O를 32배수로 pad 후 slice(정확).
- **정밀도:** DPE는 bf16 systolic(~0.23% rel, atol 1e-2; 1e-3 불가). f32 정확 reduce가 필요한
  곳만 VE 유지.

### 7-2. 본질적 한계 / 미지수

- **성능:** host↔NPU 레이어별 왕복은 느림(decode 토큰당 48레이어 × 커널 수). 정확성 우선 설계.
  추가 최적화 레버는 배칭(현재 퇴행)이 아니라 DPE 확대·상주·FP8 온칩·멀티카드.
- **host 잔여 컴퓨트(본질적):** g의 **softplus 스칼라**(native DSL op 없고 log은 값 오류라 못 씀),
  청크의 **순차 삼각역행렬(tri-inverse)·cumsum·decay_mask** 사전계산(data-dependent 순차). DeltaNet
  matmul FLOP의 3.26%(49,152 MAC). 이 부분이 decode를 host-bound로 만드는 주범.
- **FP8 dequant 정확도**(blockwise 128×128) — dequant 후 fp32 커널은 검증됨.
- **긴 prefill:** 768토큰 이상은 청크 수가 늘지만 청크 스캔이 O(T/C)라 OK. 언롤 방식은 O(T)
  그래프 + T~8 초과 시 fp32 누적 drift라 prefill은 청크 경로를 씀.
- **lm_head(vocab 151936)·embed 큰 matmul** — host 또는 NPU 분할.

### 7-3. 프로덕션 serve 통합 (3안)

prefill 단일-forward + decode 상태를 host가 들면 **생성은 host 루프로 가능**합니다. 고처리량
프로덕션 서빙(연속배칭·paged-KV 재사용)은:

- **A안 (현재 채택, host 루프 + 경량 API 래퍼):** `qcn/serve.py`. 단일/소수 요청 동작 확인.
  연속배칭·paged-KV 재사용은 우리가 직접 구현(중간 난이도). **벤더 불필요.** 현실적 프로덕션 경로.
- **B안 (furiosa-llm serve 네이티브 통합):** decode cross-step 순환상태 풀이 닫힌 Rust 런타임
  소유(append-only paged-KV, RMW 불가, Python 훅 없음 — 실측 확정). **벤더(2026.3+) 전용.**
- **C안 (하이브리드):** full-attn+MoE는 serve(qwen3_moe 위장), DeltaNet만 우리 커널 — 단 serve가
  레이어별 커스텀 커널 주입을 안 받음 → 사실상 A안으로 수렴.

→ **계획: A안**으로 프로덕션-사용 가능 수준까지. B안은 벤더 의존으로 명시. 벤더에 요청할 구체
대상(native_runtime.so의 순환상태 풀, native_llm_common.so의 qwen3_next enum/layer_types 허용)은
[README_qwen3_next_TECH.md](README_qwen3_next_TECH.md) 참고.

---

## 7-4. 아티팩트로 빌드하기 (host-loop 아티팩트)

이 host 루프 시스템은 **self-contained 아티팩트 디렉터리로 패키징**할 수 있습니다.
진짜 furiosa-llm EDF 아티팩트(`binary_bundle.zip`)는 DeltaNet 순환 외적이 컴파일러 패닉을 내
**2026.2.0에서 불가능**하지만, **host-loop 아티팩트는 빌드·적재·실행이 모두 검증**되었습니다.

```bash
# 빌드: 커널 YAML + 매니페스트 + 토크나이저/설정을 self-contained 디렉터리로 패키징
PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj \
/home/jun/furiosa/bin/python qcn/build_artifact.py
#   -> rngd-npu/artifacts/qwen3-coder-next-fp8-rngd/  (artifact.json, kernels/, tokenizer*, ...)

# 적재+실행: 매니페스트를 읽고 NPU 커널을 아티팩트에서 적재해 RNGD NPU에서 생성
PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj \
RNGD_DEV=rngd:4 QCN_DPE=1 \
/home/jun/furiosa/bin/python qcn/run_artifact.py \
    --artifact /home/jun/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen3-coder-next-fp8-rngd \
    --prompt "def quicksort(arr):" --max-new 2
```

자세한 의미(왜 EDF가 안 되는지의 정확한 차단 지점, 아티팩트 내용물, NPU 실행 증명)는
**[README_qwen3_next_ARTIFACT.md](README_qwen3_next_ARTIFACT.md)** 를 참고하세요.

---

## 부록 A. 핵심 산출물 경로

프로젝트 루트: `/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/`

**full-model 호스트 루프 (`qcn/`):**
`loader.py`·`deltanet_layer.py`·`deltanet_layer_looped.py`·`attn_layer.py`·`moe.py`·`model.py`·
`generate.py`·`serve.py`. **아티팩트 빌드/실행:** `build_artifact.py`·`run_artifact.py`
(→ `rngd-npu/artifacts/qwen3-coder-next-fp8-rngd/`, 자세히는
[README_qwen3_next_ARTIFACT.md](README_qwen3_next_ARTIFACT.md)). 검증: `validate_deltanet_layer.py`·`validate_deltanet_dpe.py`·
`validate_moe_dpe.py`·`validate_chunk_scan_batched.py`·`bench_layer.py`. 샘플/성능:
`generation_sample.json`·`generation_sample_dpe.json`·`perf_dpe.json`.

**손수 작성 커널 (`tk_kernels/`):**
`dn_linear.yaml`·`dn_conv1d.yaml`·`dn_l2norm.yaml`·`dn_gate.yaml`·`dn_chunk_full.yaml`·
`dn_step_mh.yaml`·`dn_gnorm.yaml`(검증된 per-head/per-expert 경로). DPE: `dn_linear_dpe.yaml`·
`dn_chunk_full_dpe2.yaml`. 배치(참고 보존): `dn_chunk_full_mh.yaml`·`dn_linear_be.yaml`.
DPE 역설계 기록: `dpe_serde_fields.md`·`dpe_incremental_log.md`·`dpe_struct_from_dump.md`·
`dpe_result.md`·`bench_dpe_vs_ve.py`. (커널 연구 단계별 산출물 전체는 TECH 문서.)
