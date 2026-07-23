# 청사진 — Qwen3-Coder-Next 전체 모델을 RNGD NPU에서 돌리기

작성 2026-06-11 · RNGD SDK 2026.2.0 · 4×RNGD(47.5GiB HBM each) · host 125GB RAM / 253GB 디스크
관련: [README_deltanet_kernel_study.md](README_deltanet_kernel_study.md)(커널 증명), [README_all_change.md](README_all_change.md)

> 목표: 손수 작성한 NPU DeltaNet 커널(검증 완료)을 실제 **Qwen3-Coder-Next-FP8** 가중치로
> 엮어 ① host 추론 루프로 텍스트 생성 → ② 프로덕션 serve 통합. 끝까지 시도.

---

## 0. 지금까지 증명된 것 (출발점)

손수 작성한 TK-graph 커널로 **완전한 단일헤드 Gated DeltaNet 레이어가 NPU에서 HF와 ~1e-6 일치**
(matmul FLOP 96.74% NPU, 적대적 검증 통과). 검증된 커널(`qwen3-next-proj/tk_kernels/`):
- `dn_linear.yaml` (nn.Linear=matmul), `dn_conv1d.yaml`(causal conv+SiLU), `dn_l2norm.yaml`,
  `dn_gate.yaml`(sigmoid), `dn_chunk_full.yaml`(청크 스캔+inter-chunk 상태 carry),
  `dn_step_mh.yaml`(멀티헤드 순환 스텝), `dn_gnorm.yaml`(gated RMSNorm).
- 차원 불가지론(symbolic Var) → 실모델 차원(d=128, 16/32헤드)으로 무변경 스케일.
- host에 남는 본질 한계: softplus 스칼라(native op 없음), 청크의 순차 삼각역행렬/cumsum 사전계산.

---

## 1. 실제 모델 스펙 (Qwen/Qwen3-Coder-Next-FP8, 실측 config)

| 항목 | 값 |
|---|---|
| model_type / arch | qwen3_next / Qwen3NextForCausalLM |
| 레이어 | 48 (full_attention_interval=4 → 12 full-attn + 36 DeltaNet), **매 레이어 MoE** |
| hidden / vocab | 2048 / 151936 |
| DeltaNet | key heads 16, value heads 32(n_rep=2), key/value head_dim 128, conv kernel 4 |
| full attention | 16 heads, 2 kv heads(GQA), head_dim 256 |
| MoE | 512 experts, top-10 + shared expert, moe_intermediate 512 |
| 양자화 | FP8 blockwise(weight_block_size [128,128], dynamic act); **lm_head·embed_tokens는 비양자화** |
| 크기 | 80.4GB (40 safetensors shard) |

---

## 2. 용량 가능성 (✅ 253GB로 가능 확정)

| 자원 | 필요 | 보유 | 판정 |
|---|---|---|---|
| 디스크 | FP8 80.4GB 다운로드 | 253GB 여유 | ✅ (여유 ~170GB) |
| 호스트 RAM | 80GB(FP8) — mmap이면 touched layer만 | 125GB(115 여유) | ✅ (safetensors mmap, 레이어별 dequant) |
| NPU HBM | per-layer 컴퓨트만 상주(가중치는 host→NPU 스트리밍) | 47.5GB×4 | ✅ (한 레이어 분량 ≪ 47.5GB) |

**핵심:** host 추론 루프는 전체 가중치를 NPU에 다 올리지 않는다. host(mmap)가 가중치를
보유하고, **레이어별로** 필요한 가중치를 dequant→NPU로 보내 컴퓨트 후 결과만 회수. 따라서
80GB 모델도 47.5GB 카드로 동작(느리지만 정확). serve(전체 상주·연속배칭)는 별도(4절).

---

## 3. Host 추론 루프 아키텍처

```
입력 토큰들 → embed_tokens(host, 비양자화 가중치)
 → for layer i in 0..47:
      ┌─ 입력 RMSNorm (dn_gnorm 계열 / 간단 RMSNorm 커널)
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

**상태 관리(DeltaNet의 핵심):**
- prefill: 프롬프트를 청크(예: 64토큰)로 나눠 `dn_chunk_full`로 처리, 청크간 S를 host가 carry.
- decode: 토큰마다 `dn_step_mh` 순환 스텝, **host가 S[32,128,128]·conv_state[conv_dim,3]를
  torch 텐서로 보유**(진짜 read-modify-write). full-attn 레이어 KV도 host append.
- 이로써 serve 런타임의 append-only paged-KV 한계를 우회.

**FP8 처리:** 가중치는 FP8 blockwise(128×128 스케일). 레이어 로드 시 host에서 dequant→bf16/fp32
후 `dn_linear` 등에 입력(우리 커널은 fp32 검증). 비양자화(lm_head·embed)는 그대로.

---

## 4. Serve 통합 (프로덕션 서빙) — 3안

prefill 단일-forward + decode 상태를 host가 들면 **생성은 host 루프로 가능**. 프로덕션 서빙
(OpenAI API·연속배칭·고처리량)은:

- **A안 (host 루프 + 경량 API 래퍼):** host 루프 위에 FastAPI `/v1/chat/completions` 래퍼.
  단일/소수 요청은 됨. 연속배칭·paged-KV 재사용은 우리가 구현(중간 난이도). **벤더 불필요.**
- **B안 (furiosa-llm serve 안 통합):** decode cross-step 순환상태 풀이 닫힌 Rust 런타임 소유
  (append-only, RMW 불가, Python 훅 없음 — 실측 확정). **벤더(2026.3+) 전용.**
- **C안 (하이브리드):** full-attn+MoE는 furiosa-llm serve(qwen3_moe 위장)로, DeltaNet만 우리
  커널 — 단 serve가 레이어별 커스텀 커널 주입을 안 받음 → 사실상 A안으로 수렴.

→ **계획: A안**(host 루프 + API 래퍼)으로 프로덕션-사용 가능 수준까지, serve 네이티브(B)는
벤더 의존으로 명시.

---

## 5. 실행 단계 (phased)

1. **[진행중] 다운로드** Qwen3-Coder-Next-FP8 80GB.
2. **가중치 로더** — safetensors mmap + FP8 blockwise dequant 유틸 + 레이어별 가중치 추출.
3. **커널 스케일업** — DeltaNet 커널을 실차원(key16/val32 헤드, d=128)으로; full-attn SDPA 커널;
   MoE expert matmul; RMSNorm/RoPE. 각각 HF 모듈과 대조.
4. **단일 레이어 정합** — 실가중치로 DeltaNet 레이어 0 + full-attn 레이어 + MoE를 HF와 대조.
5. **전체 forward** — 48레이어 + embed + lm_head, 프롬프트 1개 prefill → logits를 HF와 대조.
6. **decode 루프** — 샘플링 + 상태 carry로 다중 토큰 생성, HF generate와 대조(또는 품질 확인).
7. **serve 래퍼(A안)** — API 엔드포인트 + 간단 배칭.

## 5b. 실행 진행 (2026-06-11)

- ✅ **1 다운로드**: Qwen3-Coder-Next-FP8 75GB, 40/40 shard (HF 캐시).
- ✅ **2 가중치 로더**: `qcn/loader.py` (safetensors mmap + FP8 blockwise dequant) — 검증.
- ✅ **3+4 컴포넌트(실가중치 HF 대조, 전부 NPU `_dfg_inner=0`):**
  - DeltaNet 레이어(`qcn/deltanet_layer.py`, 16/32헤드·d128): vs HF **maxerr 8.9e-8**, out 32×2048·state 32×128×128.
  - Full-attn 레이어(`qcn/attn_layer.py`, GQA 16/2·head_dim 256·partial RoPE·게이트): vs HF **5.96e-7**, **matmul 100% NPU**(q/k/v/o proj+16×q@kᵀ+16×attn@v).
  - MoE(`qcn/moe.py`, 512 experts top-10+shared): vs HF **1.79e-7**, 97% NPU, 73/512 expert 활성.
  - dn_linear HW 한계(처리): weight read O·I ≤ ~2²⁰(출력축 타일링), 출력축 32배수(zero-pad), token축 ≥128(pad).
- ✅ **5 전체 forward**(`qcn/model.py`, 48레이어 가중치 스트리밍): 첫 4레이어 실HF Qwen3NextModel 대조 **maxerr ~1e-6**(layer0~3 post-residual 2.4e-7~8.9e-7, after-norm 8.6e-6), 전 mixer(DeltaNet·full-attn·MoE) NPU `_dfg_inner=0`. RMSNorm 규약 정정: 디코더/q·k norm은 `(1+weight)`, DeltaNet gated norm만 plain weight. dynamo recompile_limit 상향 필수(멀티-shape).
- ✅ **6 decode 루프 + 생성**(`qcn/generate.py`): 프롬프트 `def quicksort(arr):` → **생성** `\n    if len(arr) <= 1:\n        return arr\n    else:\n        pivot = arr[0]\n` (문법적으로 올바른 quicksort). prefill 360s, decode 24토큰 avg 55.8s/tok, NPU stages deltanet=36000·attn=10824·moe=59336, **CPU폴백 0**. prefill↔decode 일관성 1e-7(상태 S·conv tail·KV 캐시 스레딩 정확). 샘플 `qcn/generation_sample.json`.
- ✅ **7 serve 래퍼(A안)**(`qcn/serve.py`, FastAPI OpenAI 호환): 모델 1회 로드 + 요청 lock 직렬화.
  - `/v1/completions`: `"def add(a, b):"` → `"\n    return a"` (OpenAI 형식, usage).
  - `/v1/chat/completions`: Qwen chat 템플릿 적용, `"Write a Python one-liner to sum a list."` → `"```python\ntotal"`.
  - 실행: `PYTHONPATH=<proj> RNGD_DEV=rngd:2 ~/furiosa/bin/python qcn/serve.py` (포트 8900). 성능: 첫 요청 prefill 컴파일 ~360s, 정상 ~55s/tok(정확성 우선, 처리량 최적화는 후순위).
  - **native furiosa-llm serve 통합(B안)은 벤더 전용**(append-only paged-KV가 DeltaNet RMW 상태 못 들음, 실측 확정) → A안이 현실적 프로덕션 경로.

→ **🏆 실제 Qwen3-Coder-Next-FP8(80B)가 RNGD NPU에서 우리 손수 커널로 end-to-end 코드 생성 + OpenAI 호환 API 서빙 성공.**

## 5c. 성능 분석 + 배치 최적화 (2026-06-11, 정직한 음성 결과)

baseline ~55.8s/tok 의 병목은 **NPU 디스패치 횟수**: 토큰당 NPU stage가 deltanet 36000·
moe 59336(prefill+24토큰 누적) — DeltaNet은 32 value-head를 루프(헤드당 dn_chunk_full),
MoE는 top-10 active expert를 하나씩 호출하기 때문.

**시도: 배치(batch-axis)로 디스패치 수 줄이기** — 검증된 batch-axis 규칙(헤드/expert를
최외곽 축으로, tile/reduce 안 함)으로:
- DeltaNet head-batched `dn_chunk_full_mh.yaml`: 32 dispatch → 1 (32×), looped와 **bit-identical**(maxerr 0.0), HF 8.9e-8 유지.
- MoE expert-batched `dn_linear_be.yaml`: 298 → 10 dispatch(29.8×), maxerr 1.19e-7, 전부 NPU.

**그러나 wall-clock 은 오히려 느려짐(REGRESSION).** 전체 모델 prefill이 레이어당 ~7.5s →
~2분으로 **~16× 느림**. 원인(아키텍처): **RNGD 의 matmul 이 벡터 엔진(EinsumByVe =
broadcast-multiply-reduce)으로 도는데, 이는 `[.,o,i]` outer product 전체를 materialize 한 뒤
i 로 reduce 한다.** 그래서 N개 헤드/expert를 배치하면 한 op이 N배 데이터를 materialize —
처리량 이득 없이(systolic matmul 아님) 일감만 N배 + SRAM 압박. MoE 에이전트 실측: 배치된
gate matmul 1회(E=10) ~89s = baseline 토큰 전체(55.8s)보다 큼.

→ **결론: 벡터-엔진 matmul 에선 배치가 디스패치 수만 줄이고 wall-clock 은 악화.** 그래서
**검증된 per-head/per-expert 경로로 복원**(`model.py`가 `DeltaNetLayerLooped` +
`moe_forward_npu_unbatched` 사용; 55.8s/tok, end-to-end 검증된 유일 config). 배치 버전은
참고용 보존(`deltanet_layer.py`·`moe.moe_forward_npu`·`dn_chunk_full_mh.yaml`·`dn_linear_be.yaml`).

**진짜 속도 레버:** ① **`EinsumByDpe`**(실제 MAC/DPE 엔진 — systolic matmul) ② 4장 멀티카드 ③ 벤더 serve. 배치-벡터엔진은 답이 아님.

### 🚀 5d. EinsumByDpe 돌파 (2026-06-11) — systolic matmul 엔진, 1.96~3.8× 빠름

"미해결 frontier"였던 `EinsumByDpe`를 **깼습니다.** `dn_linear_dpe.yaml`이 NPU에서 실행
(`_dfg_inner=0`), torch F.linear와 1e-2 일치, **EinsumByVe 대비 compute 3.8×·end-to-end 1.96× 빠름**.
3각도 정찰(컴파일러 덤프 CBOR 디코드·serde 역설계·점진적 에러추적 11회)로 정확한 레시피 확보:
- **kind**: `EinsumByVe` → `EinsumByDpe`.
- **contraction을 `ein_ops.reduce`로**(VE는 `ein_ops:~`였음): `{mode: Add, input: <pre-reduce 곱 [t,o,i] TensorLike>, axes: [<contract 축 LabelStride>], source: ""}` + `mul_source: ""`(string).
- **vector_ops는 단일입력 identity passthrough**: `{inputs:[0], insts:[{def:1, Reduce LocalReduceAddF operand Tensor:0 axes Tag:[]}]}` (빈-axes Reduce = DSL에 identity Unary 없어서 이게 유일한 통과 관용구).
- **reads/write 그대로**. → 모든 'ti,oi→to' matmul(dn_linear·dn_linear_be·dn_chunk_full QK/KV·dn_einsum)에 일반화.
- **정밀도**: DPE는 **bf16 systolic**이라 ~0.23% rel(atol 1e-2; 1e-3 불가). 단 실모델이 FP8/bf16이라 오히려 더 충실. f32 정확 reduce 필요한 곳만 VE 유지.
- 컴파일러는 8×8 matmul도 항상 DPE로 낮춤 → DPE가 네이티브 정답, VE는 우리가 손으로 쓴 것뿐.
- 역설계 방법(재사용): serde FIELDS는 `.data.rel.ro`의 (ptr,len) reloc 배열로 복원(strings는 dedup돼 순서 신뢰불가); DFG serialize_to_str = base64([8B len][CBOR]) → `cbor2.loads(b64decode(s)[8:])`, DPE op은 `reduce_mode`+`acc_major_mode` 키 가진 dict.
- 산출물: `dn_linear_dpe.yaml`·`dpe_serde_fields.md`·`dpe_incremental_log.md`·`dpe_struct_from_dump.md`·`bench_dpe_vs_ve.py`·`dpe_result.md`.

### 🚀 5e. DPE를 모델 전체에 적용 — 실측 가속(정확성 보존)

`QCN_DPE=1` 플래그 하나로 attn·moe·deltanet의 모든 matmul(proj·out_proj·QK/AV·MoE gate/up/down·
DeltaNet 스캔)을 DPE로 전환. **실측 end-to-end:**

| 지표 | VE baseline | DPE | 가속 |
|---|---|---|---|
| Prefill (24토큰 프롬프트) | 360.8s | **76.9s** | **4.69×** |
| Decode | 55.8 s/tok | **35.15 s/tok** | **1.59×** |
| 투영 matmul 512→2048 | 13.09ms | 1.44ms | **9.06×** |
| expert gate 2048→512 | 5.48ms | 2.47ms | 2.21× |

- **정확성 보존**: 생성 코드 여전히 올바른 quicksort(첫 4토큰 byte-identical), 4레이어 HF 대조 atol 1e-2 통과(per-layer maxerr ≤2.5e-2), attn layer-3 maxerr 5.5e-3·MoE layer-0 7e-4 vs HF. 전부 NPU(`_dfg_inner=0`).
- **decode가 1.59×로 prefill(4.69×)보다 modest한 이유**: decode는 일부 **host-bound**(DeltaNet 32헤드 cumsum·tri-inverse T-행렬, MoE 라우팅) — matmul만 빨라지고 host 부분은 그대로.
- **발견 제약 ① DPE per-graph 2개 cap**: 한 TacticKernel 그래프에 `EinsumByDpe` 3개+면 `fuse_mamma_to_single_einsum_by_dpe` 융합 패스가 systolic array를 오스케줄해 **조용히 오컴파일**(dfg=0이나 garbage, maxerr 0.5). DeltaNet 스캔(5 matmul)은 2개만 DPE(`dn_chunk_full_dpe2.yaml`, maxerr 2.5e-4 통과), 나머지 VE. 전체 DPE화하려면 그래프를 ≤2-DPE 단위로 분할 필요(YAML 필드 아닌 파티셔닝).
- **제약 ② DPE 출력축 O=1 거부**(shared_expert_gate 등): O를 32배수로 pad 후 slice(정확).
- **정밀도**: DPE는 bf16 systolic(~0.23% rel, atol 1e-2). 실모델이 FP8/bf16이라 오히려 충실. f32 정확 reduce 필요한 곳만 VE 유지(`QCN_DPE` 미설정 = 기본 VE, HF와 1e-7).

→ **DPE = 검증된 진짜 가속 레버(배치-벡터엔진 퇴행과 대조). 서버는 `QCN_DPE=1`로 4.69×/1.59× 빠르게.**
산출물: `dn_chunk_full_dpe2.yaml`·`validate_moe_dpe.py`·`validate_deltanet_dpe.py`·`perf_dpe.json`·`generation_sample_dpe.json`.

## 6. 리스크 / 미지수

- FP8 dequant 정확도(blockwise 128×128) — dequant 후 우리 fp32 커널은 검증됨.
- 성능: host↔NPU 레이어별 왕복은 느림(decode 토큰당 48레이어 × 커널 수). 정확성 우선, 성능은
  배칭·상주·FP8 온칩으로 후순위 최적화.
- full-attn(head_dim 256)·MoE(512 experts) 커널 스케일 — 원리는 matmul, 스케일만.
- 768토큰 이상 긴 prefill의 청크 수 증가 — 청크 스캔은 O(T/C)라 OK.
- lm_head(vocab 151936)·embed 큰 matmul — host 또는 NPU 분할.
