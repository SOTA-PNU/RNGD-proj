# DeltaNet 커널 연구 — "컴파일러에 커널을 넣으면 되지 않나?" 정직한 답 (공부용)

작성 2026-06-10 · RNGD SDK 2026.2.0 · npu-compiler(npu-tools git `3f23a71`)

> 질문: Qwen3-Coder-Next(및 후속 qwen3.5/3.6)가 쓰는 **Gated DeltaNet**가 컴파일이 안 되니,
> "컴파일러 커널 위치를 찾아 거기에 커널/연산 패턴을 넣고 radare2로 코드를 주입하면 되지
> 않나?" 이 문서는 구조를 공부하고 그 가능성을 **사실 기반**으로 검증한 결과입니다.
> 결론부터: **radare2 커널 주입은 불가능**. 그러나 **중대한 정정(2026-06-10 7차)** —
> Rust 소스 없이도 **`furiosa.torch.TacticKernelModule`로 DFG 커널을 손수 작성해 EDF로
> 컴파일·NPU 실행이 가능**합니다(실측 확인). 즉 "벤더 전용"은 절반만 맞습니다: **컴퓨트
> 커널은 우리가 작성 가능**, 다만 **serve 런타임의 cross-step 순환상태 풀은 벤더 전용**.
> (지난 "standalone elementwise는 근본 불가" 결론도 부정확 — 2절 정정.)

---

## 🚀🚀 8차 돌파 (2026-06-11) — Gated DeltaNet 레이어를 실제로 NPU에서 계산 성공

**"한계를 뚫었다."** 손수 작성한 TK-graph 커널로 **완전한 Gated DeltaNet 레이어를 RNGD NPU에서
계산**해 HuggingFace `torch_recurrent_gated_delta_rule`과 **~1e-7로 일치**시켰습니다. 두 방식 모두 성공:

| 산출물 | 내용 | NPU 결과 | 검증 |
|---|---|---|---|
| `dn_einsum_f32.yaml` | fp32 contraction Σₖ S[k,v]·k[k] | ✅ rngd:0 | torch.einsum, err 2.4e-7 |
| `dn_rank1.yaml` | rank-1 외적 갱신 S+=k⊗δ | ✅ rngd:0 | 정확 일치 |
| `dn_decay.yaml`·`dn_delta.yaml` | S·decay, (v−kv)·β | ✅ rngd:0 | 정확 일치 |
| **`dn_step.yaml`** | **delta-rule 한 스텝 전체(7-op 융합)** | ✅ rngd:0 | Sout·out 둘 다, err <1.2e-7, CPU폴백 0 |
| **host-loop (T=8)** | 스텝 커널을 8토큰에 상태 스레딩 | ✅ rngd:0 | **HF ref 일치**, err 6e-8, CPU폴백 0/8 |
| **`dn_prefill_unroll4.yaml`** | **T=4 언롤 → 단일 EDF, 한 번의 forward** | ✅ rngd:0 | **HF ref 일치**, err 1.2e-7 |

→ 즉 지난 세션에 kernelize가 거부했던 게이트·contraction·rank-1·순환을 **TK-graph로 직접
작성하니 NPU에서 정확히 실행**됩니다. **순환은 (a) host-loop(상태 스레딩) 또는 (b) 그래프
내 언롤(단일 EDF) 로 처리** — 막혔던 `Loop` 노드 없이 우회 성공.

**핵심 DSL 돌파 교훈:**
- 외적(broadcast 2개)은 단일 Elementwise면 EDF가 Cpu 노드로 떨궈 실패 → **`EinsumByVe` (Reduce inst 없이)** 가 외적, Reduce inst 있으면 contraction. 둘 다 broadcast read를 read0로.
- per-head 스칼라(decay/β)는 caller가 `torch.full`로 동형 텐서로 materialize(진짜 [1] 브로드캐스트는 미지원), 또는 `{ConstFloat}` 직접 사용.
- 텐서 id는 그래프 전역 flat, op-local `Tensor:0/1` 참조와 분리. 다출력 OK. fp32는 `MulF/AddF/SubF`+`LocalReduceAddF`.
- q는 1/√d_k 스케일을 **커널 전에** 먹여야 HF와 맞음(HF는 루프 전 query*scale).
- 언롤: 스텝 7-op 바디를 T번 복제(매 step 새 intermediate id, S를 다음 step S_in으로 연결) → 단일 Dfg/EDF. `gen_unroll.py`가 T 파라미터화(T=8,16도 생성 가능).

**남은 단 하나의 진짜 vendor-lock = 그래프 내 네이티브 `Loop` 노드** (`dn_loop2.yaml`):
정적 `UnlabeledShape{[1]}`로 첫 벽(SymExpr 상수)은 넘었으나, `loop_impl.rs:126`이 loop_index를
**스케줄러가 만드는 SpmShape 스칼라**로 요구 → naive_yaml로 못 만듦. **그러나 언롤이 이를
불필요하게 만듦**(고정 seq면 언롤, 가변이면 host-loop).

**아직 남은 일(엔지니어링, 원리는 증명됨):** ① 실제 차원으로 스케일(d_k=d_v=128, 32헤드,
36레이어 — 검증은 d=4 미니). ② 긴 seq는 청크 형태 필요(언롤은 그래프가 T에 선형). ③
furiosa-llm **serve** 통합 — prefill은 단일 forward로 가능해졌으나 autoregressive decode의
cross-step 상태는 여전히 serve 런타임 paged-KV 한계 → 커스텀 host 추론 루프(우리 방식)나
벤더 상태풀 필요. **즉 컴퓨트 벽은 완전히 뚫렸고, 남은 건 스케일·serve 통합.**

산출물: `qwen3-next-proj/tk_kernels/` (커널 YAML + `run_dn_step.py`·`host_loop_test.py`·
`unroll4_test.py`·`gen_unroll.py` 드라이버).

### 🚀 9차 스케일 (2026-06-11) — 실차원·멀티헤드·청크 형태까지 NPU 검증

8차 돌파(미니 d=4)를 실모델 규모로 밀어붙임. **남은 일 ①②③(스케일·멀티헤드·긴 seq) 모두 해결:**

| 항목 | 결과 | 검증 |
|---|---|---|
| **실차원 스케일** d_k=d_v=128 | ✅ `scale_test_d128.py` | dn_step.yaml 무변경(symbolic Var:K,V), torch 일치 오차 8.6e-6, CPU폴백 0 |
| **멀티헤드** H=4, d=128 (`dn_step_mh.yaml`) | ✅ rngd:0 | 4헤드 전부 일치, 오차 1.5e-5, CPU폴백 0. head는 batch축(특별처리 불필요) |
| **언롤 한계** (`dn_prefill_unroll{8..128}.yaml`) | ⚠️ 소프트 한계 | 컴파일은 T=128(896op)까지 OK이나 시간 초선형(T64≈110s, T128≈449s); 정확도는 fp32 누적으로 **T~8 초과 시 drift** |
| **청크 형태 한 청크** (`dn_chunk.yaml`) | ✅✅ rngd:1 | chunk-parallel gated delta rule, **3개 matmul 코어 전부 NPU**, 실config C=64/d=128 오차 7.6e-6, CPU폴백 0 |

**핵심 스케일 발견:**
- **head 축 = 그냥 batch 축**: 모든 텐서 shape에 최외곽 `h` 라벨 추가, reduce 축엔 절대 안 넣음(contraction은 head별 유지), tile 안 함. 한 번에 성공. symbolic 차원이라 dn_step.yaml은 **차원 불가지론** — 같은 YAML이 d=4·d=128·H=4 다 동작.
- **행렬-행렬 matmul einsum `ck,dk→cd`가 NPU 내려감**(8차 vector-broadcast einsum의 일반화): read0=A[c,k] tiled d, read1=B[d,k] tiled c (둘 다 broadcast) + Reduce over 공유축 k → 출력 [c,d](두 생존축이 서로 다른 피연산자에서). bmm/matmul 전부 표현 가능.
- **청크 형태가 긴 seq의 정답**: 언롤은 O(T) 그래프 + T~8 정확도 한계. 청크는 matmul 위주라 O(T/C) + 정확. cumsum만 host(순차), exp/마스크는 NPU Unary로.
- **새 frontier+해결**: no-reduce EinsumByVe에서 read0=broadcast-1D, read1=full-2D면 EDF가 Cpu노드로 떨궈 그래프 분할 실패(`clusterer/cluster.rs:32 "multiple internal subgraphs"`). no-reduce 외적은 **양쪽 다 1D broadcast**일 때만 NPU. 해결: 한쪽을 2D로 materialize(또는 2D Unary Exp) 후 matching-shape Elementwise MulF.

**남은 일(갱신):** ④ 멀티-청크 스캔(청크 간 상태 carry) + 완전한 DeltaNet 레이어 조립(입력투영·conv1d·l2norm·gated RMSNorm + 스캔) ⑤ furiosa-llm serve 통합(decode cross-step 상태 = paged-KV 한계, 커스텀 host루프 or 벤더). 산출물 추가: `dn_step_mh.yaml`·`dn_chunk.yaml`·`dn_prefill_unroll{8..128}.yaml`·`scale_test_d128.py`·`gen_chunk.py`·`mh_test.py`·`unroll_limit_test.py`.

### 🚀 10차 — 완전한 DeltaNet 레이어 컴퓨트 조각 전부 NPU 검증 (2026-06-11)

레이어를 이루는 **모든 컴퓨트 조각**을 NPU에서 만들고 HF와 대조 완료. ④의 핵심(멀티청크 스캔)과 주변 op 전부 통과:

| 조각 | 파일 | NPU 결과 | 검증 |
|---|---|---|---|
| **멀티청크 스캔**(inter-chunk state carry) | `dn_chunk_full.yaml` (12-op: 5 matmul+7 elementwise) | ✅ rngd:1/2 | NC=3 청크 host-loop, HF `torch_chunk_gated_delta_rule` 일치 out 1.5e-8·state 3e-8, **carry 실검증**(S_prev≠0), CPU폴백 0 |
| causal conv1d + SiLU | `dn_conv1d.yaml` (K=4 depthwise, 8-op) | ✅ rngd:0 | F.silu(conv1d) 일치 1.4e-6 |
| l2norm (q/k 정규화) | `dn_l2norm.yaml` (6-op) | ✅ rngd:0 | x·rsqrt(Σx²+eps) 일치 6e-8 |
| gated RMSNorm (출력) | `dn_gnorm.yaml` (10-op) | ✅ rngd:0 | Qwen3NextRMSNormGated 일치 2.9e-6 |

**멀티청크 핵심 설계:**
- **삼각역행렬 정련(HF L511-515)은 생략 불가**(T=I면 1e30 발산). 단 T·value·k_cumdecay·decay_mask는 **S_prev 무의존**이라 host 사전계산 후 입력 주입, **S_prev 닿는 recurrence(4 matmul)만 NPU** — 정련 honor + inter-chunk 상태 carry 온디바이스 정확.
- **q,k L2정규화 필수**(실모델 `use_qk_l2norm_in_kernel=True`): 안 하면 state가 ~1e14로 폭발해 NPU fp32 matmul 상대오차 1e-6이 절대오차 1e8 됨. 커널 수학은 어느 레짐서나 정확(CPU maxerr 0.0). β는 host의 k_beta/v_beta로만 진입(온칩 커널 β-불변).

**주변 op DSL 실측(probe_unary/binary.py):**
- 유효 Unary: `Exp`·`Sigmoid`·`Sqrt`만(수치 정확). `Tanh/Sin/Cos/Log`는 parse되나 **값 틀림**(table_lookup 필요). `Rsqrt/Reciprocal/Silu/Gelu` 등은 enum 없음 → **native rsqrt 없음**: `1/√s = √s/s`(Sqrt 후 DivF(rt,se))로 구현(maxerr 6e-8).
- 유효 Binary: `MulF/AddF/SubF/DivF`. DivF는 함정 다수(numerator==1 상수면 collapse). reduction `LocalReduceAddF`만.
- **reduction은 reduce축=외곽 + 생존축=내곽이고 생존축 ≥128일 때만 NPU**(작으면 Cpu노드). l2norm은 [d,m]로 transpose해 sumsq.
- **융합 inst ≤2개**(긴 chain은 `preferred_ve_lhs is not an operand` 실패) → op 쪼개기.

→ **이제 레이어의 모든 컴퓨트 조각이 NPU 검증됨**: 투영(matmul)·conv1d·l2norm·청크스캔(상태carry)·gated RMSNorm. **남은 일: 이 조각들을 완전한 레이어 forward로 조립(HF Qwen3NextGatedDeltaNet 대조) + serve 통합(decode cross-step 상태).** 산출물 추가: `dn_chunk_full.yaml`·`gen_chunk_full.py`·`run_dn_chunk_full.py`·`dn_conv1d/l2norm/gnorm.yaml`·`gen_dn_layer.py`·`test_dn_layer.py`·`probe_unary/binary.py`.

### 🏆 11차 캡스톤 (2026-06-11) — 완전한 DeltaNet 레이어가 NPU에서 HF와 ~1e-7 일치 (적대적 검증 통과)

조각들을 **완전한 단일헤드 Gated DeltaNet 레이어 forward**로 조립(`full_layer.py`, host-오케스트레이션) →
**HF `Qwen3NextGatedDeltaNet`(진짜 torch 경로) 전체와 대조 성공.** 별도 검증 에이전트가 적대적 재검증.

- **결과:** allclose(atol 1e-2)=True, **maxerr 3.9e-7** (hidden=256, K=V=32, T=32/2청크). T=48/3청크 2.5e-7, T=64/4청크 2.7e-7. **총 `_dfg_inner=0`**(모든 DeltaNet 고유 스테이지 NPU 실행).
- **NPU 실행 스테이지(스테이지별 dfg_delta=0):** conv1d+SiLU · l2norm(q)·l2norm(k) · beta=sigmoid · **멀티청크 스캔(inter-chunk 상태 carry, 청크당 dn_chunk_full 1회, 4청크까지 검증)** · gated RMSNorm(core,z, z-게이팅 실재).
- **적대적 검증 4종 통과:** ①`_dfg_inner` spy가 진짜 CPU경로라 count=0=NPU확정 ②HF는 진짜 torch 경로(`F.silu(conv1d)`+`torch_chunk_gated_delta_rule`, fla/fast-path 비활성) ③오차 1e-7로 tol 1e-2보다 5자릿수 아래 + 오염주입 시 오차 1.2(민감) ④l2norm 실변환(norm 20→1.0)·gnorm z-게이팅(zero gate→zero out)·미달크기 reduction은 조용한 폴백 아닌 **에러**라 가짜 통과 불가.
- **정직한 caveat(검증됨):** 이 조립에선 **host 실행** = in_proj/out_proj matmul(NPU-compilable, dn_einsum_f32로 별도확인) + g의 softplus 스칼라(native DSL op 없음; log은 값오류라 못 씀) + 청크의 S_prev-무의존 사전계산(삼각역행렬·cumsum·decay_mask). 단일헤드·소차원(hidden=256). l2norm/gnorm은 행<128이면 Cpu노드라 128로 zero-pad(정확).

→ **DeltaNet '고유' 컴퓨트(선형어텐션 순환의 본질)는 완전히 NPU에서 돈다 — HF와 ~1e-7.** 남은 일:
①투영/softplus/사전계산까지 NPU로(거의 matmul, easy; softplus·tri-inverse·cumsum은 순차/log한계) ②멀티헤드·실차원·긴seq 스케일(원리 증명됨) ③**full-model 통합**(36 DeltaNet + 12 full-attn + MoE; 후자는 표준 NPU/위장으로 됨) ④**serve**(decode cross-step 상태=paged-KV 한계 → 커스텀 host 추론루프 or 벤더 상태풀). 산출물: `full_layer.py`.

### 🏁 12차 — 투영까지 NPU(96.74%) + serve 경로 확정 (2026-06-11)

**(A) 최대한-NPU 레이어** (`full_layer_npu.py` + `dn_linear.yaml`): nn.Linear `y=xWᵀ`를 EinsumByVe matmul('ti,oi→to')
커널로 만들어 **in_proj_qkvz/in_proj_ba/out_proj까지 NPU**로. 완전 레이어 재검증: HF와 **maxerr 1.6e-6**,
총 `_dfg_inner=0`, **matmul FLOP의 96.74%가 NPU**(1,458,176 MAC). host 잔여 3.26%(49,152 MAC)는
오직 **순차 삼각역행렬 T 사전계산**(k_beta@kᵀ·T@v_beta·T@k_cumdecay) + softplus 스칼라뿐 —
둘 다 본질적 한계(softplus는 native DSL op 없고 log은 값오류; tri-inverse는 data-dependent 순차).

**(B) serve(autoregressive decode) 경로 — 코드 정독 후 확정:**
- ❌ **DeltaNet 상태를 KV-캐시로 위장**: 정적 게이트(`specs/inputs.py:69` k.shape==v.shape, `utils.py:227` is_kvcache 이름regex)는 통과하나 **런타임이 슬롯 인덱스를 append식으로 소유**(`paged_attention.py:126` cache[idx]=val, idx는 runtime scheduler가 토큰마다 새 블록, block_size=1 고정) → 상태가 매 스텝 다른 슬롯에 흩어져 **read-modify-write 불가**. output→next-input aliasing은 닫힌 Rust KVCachePlan 소유(Python 훅 없음). SSM/mamba/conv_state 기구 furiosa_llm/런타임에 **전무**.
- ✅ **furiosa-llm serve '밖' 커스텀 host 추론 루프 = 확실히 가능(권장)**: S[h,k,v]·conv_state를 host torch 텐서로 보유(진짜 RMW), prefill은 청크 경로(dn_chunk_full), decode는 per-step 커널(dn_step_mh·dn_einsum_f32·dn_conv1d/l2norm/gate/gnorm)을 매 스텝 rngd 호출. 런타임 append 스케줄러 우회. 비용: 샘플링/배칭/연속배칭을 직접 구현.
- ⚠️ **furiosa-llm serve '안'**: 벤더 전용(cross-step 순환상태 풀 + qwen3_next enum/preset + DeltaNet 파티셔너 지원, `graph_partitioner.py:130` IndexError) → 2026.3+.

→ **정리: DeltaNet 레이어 컴퓨트는 96.74% NPU·HF 일치로 사실상 완성. 배포는 커스텀 host 추론
루프로 가능(furiosa-llm serve 통합만 벤더 몫).** 남은 큰 빌드 = full-model(48레이어+MoE+임베딩+샘플링)
host 추론 루프 — 컴퓨트 조각은 전부 증명됨, 실가중치로 엮는 엔지니어링. 산출물: `dn_linear.yaml`·`full_layer_npu.py`·`test_dn_linear.py`.

---

## ⭐ 7차 정정 (2026-06-10) — TacticKernelModule: Rust 소스 없이 커널 작성 가능

지난 "벤더 컴파일러 소스 없으면 불가" 결론은 **불완전**했습니다. Furiosa SDK에는
**Python+YAML로 DFG 커널을 손수 작성하는 공식 API**가 들어있고, 실측으로 NPU 실행까지
확인했습니다.

**경로:** `furiosa.torch.TacticKernelModule(dsl_yaml)` → `Dfg.parse` → `torch.ops.furiosa.dfg`
커스텀 op → `torch.compile(module, backend=furiosa.torch.backend)` → EDF 컴파일 + NPU 로드.
(Furiosa 자신이 `models/core/operators/tk_graphs/moe_blockwise_compute_wg_idx.yaml` MoE
work-group-index 커널을 이 방식으로 작성·사용 중.)

**실측 (이 머신, npu-compiler 3f23a71):**
- ✅ 손수 작성한 DSL 파싱(`TacticKernelModule`), CPU 실행(`DfgExecutor`) OK.
- ✅ **커스텀 elementwise add 커널을 EDF로 컴파일 → rngd:0에서 실행 → 정답 반환**(CPU 폴백
  0회 호출로 NPU 실행 검증). `torch.compile(..., backend=furiosa.torch.backend)` 경로.
- ⚠️ 저수준 `compiler.compile(ExportedProgram)` 는 `furiosa::dfg` 재import 실패(고수준
  dynamo 백엔드 경로만 동작). 입력 없는 SymArange-only 그래프는 NPU 컴파일서 segfault →
  커스텀 커널은 실입력 텐서를 최소 1개 받아야 안전.

**DSL 표현력 (gated-delta에 충분):** `SymTacticKernel`(실연산) — `EinsumByVe`/`EinsumByDpe`
contraction + 1급 `Einsum`(input/output equation), SSA vector ALU(`MulFxp`·`AddFxp`·`SubFxp`),
unary(`Sigmoid`·`Exp`·`NegExp`·`Erf`), reduction(`LocalReduceAdd`/`GlobalReduceAddFxp`),
`Cumsum`·`Gather`·`Where`. **carried-state `Loop` 노드**(initial_states·captured_inputs·
final_states·inner_operators)도 **손수 작성 가능**(컴파일러 생성 전용 아님) → 순환 스캔을
언롤 없이 표현 가능. 단 DSL은 미문서화·미검증(셔ipped 예제는 loop-free MoE 하나뿐),
컴파일러가 일부 loop 설정 거부(outer loop의 TP축·padding·split-tensor, dynamic_limit은
concrete index 텐서 필요).

**그래서 정정된 가능/불가:**
| 항목 | 가능? | 비고 |
|---|---|---|
| DeltaNet **컴퓨트 커널**(gating·contraction·rank-1·l2norm·**carried-state Loop**) 작성 | ✅(원리상) | TacticKernelModule DSL. 매우 어렵고 미문서화이나 **벤더 소스 불필요** |
| 그 커널을 EDF로 컴파일·NPU 실행 | ✅(실측) | `torch.compile`+furiosa backend |
| **serve 런타임 cross-step 순환상태 풀** | ❌ | native_runtime.so는 paged-KV만. artifact 계약이 (K,V) 동형 튜플 강제, state/conv_state origin 없음. Python API 없음 → **벤더 전용** |
| npu-tools **소스** 편집/재컴파일 | ❌ | 비공개(github엔 furiosa-sdk 프론트엔드만). 머신에 소스 없음 |

→ **컴퓨트는 우리가 만들 수 있으나, 토큰별 디코드 사이 상태 유지(serve)는 벤더 몫.**
prefill(단일 forward, Loop가 그래프 내부서 상태 운반) 또는 furiosa-llm serve 밖의 커스텀
host 추론 루프(state-in/state-out를 호스트가 스레딩, PRNG 커널 패턴)로는 컴퓨트 실행 가능.

### ⭐⭐ 7차 실증 — DeltaNet 핵심 연산을 실제로 손수 작성해 NPU에서 돌림

말로만 "가능"이 아니라, TK-graph 커널을 **직접 작성·컴파일·NPU 실행**까지 했습니다
(`qwen3-next-proj/tk_kernels/`, npu-compiler 3f23a71, rngd:0). 이게 핵심 증거입니다 —
**지난 세션에 kernelize가 거부했던 바로 그 standalone 연산(sigmoid·mul·contraction)을,
TK-graph로 직접 작성하니 NPU에서 정확히 실행됩니다** (즉 op 미지원이 아니라 그래프 위치/
tactic 문제였다는 2절 진단을 실증 확인).

| 커널 (DeltaNet 대응) | 파일 | NPU 컴파일·실행 | 검증 |
|---|---|---|---|
| baseline elementwise add | `custom_add.yaml` | ✅ rngd:0 | int32, torch 일치, CPU폴백 0회 |
| **게이트** `v*sigmoid(b)` (Sigmoid+MulF 융합) | `dn_gate.yaml` | ✅ rngd:0 | **fp32**, max err 1.19e-7, CPU폴백 0회 |
| **kv-read contraction** `out[v]=Σ_k S[k,v]·k[k]` (einsum kv,k→v) | `dn_einsum.yaml` | ✅ rngd:0 | torch.einsum 정확 일치, CPU폴백 0회 |
| **순환 Loop**(carried-state, 누산 스켈레톤) | `dn_loop.yaml` | ❌ **frontier** | parse는 OK, 실행 불가 |

**실행 레시피(확인됨):** `import torch` 먼저 → `import furiosa.torch as ft` →
`m=TacticKernelModule(open(yaml).read())` → `cm=torch.compile(m, backend=ft.backend)` →
`cm(*[t.to('rngd:0') for t in inputs])`. (저수준 `compiler.compile(ExportedProgram)`은
`furiosa::dfg` 재import 실패 — 고수준 dynamo backend 경로만 AOT-EDF 컴파일.)

**DSL 실전 교훈:** ① VE ALU의 `*Fxp` op은 **고정소수/정수 전용** — f32엔 `MulF`/`AddF`
써야 함(`MulFxp`는 type_checker.rs:92서 거부). ② contraction은 `kind: EinsumByVe` +
broadcast read(`tiles`)를 첫 read로 + 둘째 vector_op로 `LocalReduceAddFxpSat`(공유 라벨이
출력에 없으면 그 축이 축약됨). `EinsumByDpe`(진짜 MAC)는 추가 struct 채워야 동작. ③ 커널은
실입력 텐서 ≥1개 필요(입력 없으면 segfault).

**순환 Loop = 정확한 벽 (두 겹, 둘 다 벤더 전용):**
- `option: Loop` 노드는 **parse는 통과**(LoopInterface: loop_index·limit(정적 int)·
  initial_states·captured_inputs·final_states·local_tensors). 하지만 CPU DfgExecutor에서
  `loop_impl.rs:126` "loop index tensor must be SPM scalar or unlabeled [1]"로 실패 —
  naive_yaml DSL엔 **상수 차원 표현이 없고**(Var(symbol)/BinOp만; `Var:"1"`은 자유 심볼이
  됨), local_tensors 심볼이 `symbolic_params`로 해소 안 됨. 즉 **loop_index의 상수 [1]
  차원을 DSL로 못 적음**.
- 게다가 raw Dfg **Loop는 NPU AOT 경로가 없음**(`furiosa::dfg only runs on CPU device`).
  einsum·gate는 AOT-EDF로 NPU 실행됐지만 Loop는 그 경로를 안 탐.

→ **결론(실증): DeltaNet의 per-step/per-chunk 컴퓨트(게이트·contraction·rank-1·norm)는
우리가 손수 작성해 NPU에서 돌릴 수 있다. 그러나 이들을 순환으로 엮는 (a) 그래프 내 Loop
(상수-차원 loop_index DSL 미지원 + Loop AOT 경로 없음)와 (b) serve 런타임 cross-step 상태풀
— 둘 다 벤더(npu-tools 소스/2026.3+) 전용이다.** 손수 만든 커널들은 `tk_kernels/`에 보존.

관련: [README_qwen3_next_feasibility.md](README_qwen3_next_feasibility.md),
[README_radare2_gate_analysis.md](README_radare2_gate_analysis.md), [README_all_change.md](README_all_change.md).

---

## 0. 한눈 결론

| 경로 | 가능? | 이유 (실측) |
|---|---|---|
| radare2로 새 커널/코드 주입 | ❌ | 컴파일러는 닫힌 컴파일된 Rust(.so에 정적링크). op 지원=컴파일타임 enum+match+phf 정적테이블, 런타임 플러그인/레지스트리 없음. 새 커널=새 lowering 패스+tactic+TU/VE/DPE 코드젠(수천 바이트 재배치 기계어) → 스트립 105MB 바이너리에 손패치 불가(stub/jump-slot/심볼 없음) |
| `allow_unlowered_operators=true` config | ❌ | 실측: 빌드가 **hang**. unlowered op은 DRAM IR 레벨에 남아 실행 불가, 다운스트림이 멈춤 |
| `allow_external_operators=true` config | ❌ | **이미 컴파일된 EDF/DMA 블롭**만 주입(ExternalOperator 노드). 임의 op의 CPU 폴백 아님 |
| 순수 Python으로 matmul 재정식화 | ❌ | chunked 형태로 cumsum→삼각행렬 matmul, pad→cat, mask→상수 까지는 되나 **데이터 의존적 exp() decay 게이트·l2norm rsqrt는 상수로 못 접어** standalone으로 남음 |
| **TacticKernelModule(DFG DSL) 로 커널 손수 작성** | ✅ 컴퓨트만 | ⭐ Rust 소스 불필요. 커스텀 커널 EDF 컴파일·NPU 실행 실측 확인(아래 7차). serve 상태풀은 별개(불가) |
| **벤더 융합 스캔 커널 + 런타임 상태풀 (2026.3+)** | ✅ serve 완전 | 컴퓨트는 위로 가능하나 **토큰별 디코드 상태 유지(serve)는 벤더 전용** |

---

## 1. Gated DeltaNet 구조 공부 (qwen3_next / qwen3.5 / qwen3.6 공통)

**DeltaNet = softmax 어텐션을 "gated delta rule" 선형 순환으로 대체한 토큰 믹서.**
레이어당 고정 크기 행렬 상태 `S ∈ R^(d_k×d_v)`를 토큰마다 갱신합니다. (출처:
`transformers/models/qwen3_next/modeling_qwen3_next.py`)

두 가지 수학적으로 동등한 실행 형태:
- **순환(recurrent) 형태** (`torch_recurrent_gated_delta_rule`, L547-586) — 디코드용, 토큰당 1스텝:
  ```
  state = state * g_t                       # 게이트 감쇠 (elementwise)
  kv_mem = (state * k_t).sum(-2)            # = Sᵀk  (matvec)
  delta  = (v_t - kv_mem) * beta_t          # elementwise
  state  = state + k_t ⊗ delta              # rank-1 외적 갱신
  out_t  = (state * q_t).sum(-2)            # = Sᵀq  (matvec)
  ```
- **청크(chunked) 형태** (`torch_chunk_gated_delta_rule`, L467-544) — 프리필용, 대부분 bmm/matmul.

레이어 구성: 48레이어 중 `full_attention_interval`(=4)마다 1개는 **gated full attention**
(12개), 나머지 **36개가 Gated DeltaNet**. 모든 레이어 MLP는 MoE.

**연산을 종류별로 분해** (이게 핵심 공부 포인트):
| 종류 | 연산 | 컴파일러 처리 |
|---|---|---|
| (a) matmul/contraction | in_proj·out_proj, Sᵀk·Sᵀq, 청크 어텐션곱 | ✅ 커널화 가능(qwen3_moe가 되는 이유) |
| (b) standalone elementwise | `beta=sigmoid(b)`, `g=-exp(A_log)*softplus(a+dt_bias)`, l2norm `rsqrt`, gated RMSNorm, 게이트 곱 | ⚠️ op은 지원되나(아래 2절) DeltaNet 그래프 위치에선 tactic 미할당 |
| (c) depthwise conv1d | causal conv (q,k,v) | ✅ conv 지원(단 padding→cat, 융합split→별도conv 필요) |
| (d) 제어흐름/상태 | 순환 스캔 루프, read-modify-write 상태 S, 청크 삼각역행렬 루프 | ❌ tactic 없음 + 런타임 상태버퍼 없음 |

qwen3.5/3.6도 같은 gated DeltaNet+gated attention이라 **같은 (b)(d) 벽**을 공유합니다.
gated full-attention 레이어(12/48)는 표준이라 컴파일됩니다 — 문제는 36개 DeltaNet뿐.

---

## 2. ⚠️ 지난 결론 정정 — "standalone elementwise 불가"는 부정확했다

`furiosa/native_torch/compiler.pyi`의 **권위 있는 지원 매트릭스**를 실측한 결과:

- `is_importable()` (프론트엔드가 받는 ~160 op): `sigmoid`, `mul.Tensor`, `exp`, `log`,
  `rsqrt`, `sum.dim_IntList`, `cumsum`, `constant_pad_nd`, `mm`, `bmm` … **다 포함**.
- `is_supported_aten()` (**커널화 가능** ~100 op): `sigmoid`(L277), `mul.Tensor`(L342),
  `exp`(L275), `log`(L274), `rsqrt`(L285), `sum.dim_IntList`(L273), `cumsum`(L326),
  `constant_pad_nd`(L360), `mm`(L338)/`bmm`(L339) … **다 포함**.

즉 **sigmoid·mul·exp·cumsum은 공식적으로 커널화 지원 op**입니다. 그런데도 빌드는
`O957(sigmoid)`·`O1288(mul)`에서 "is not an operator that is yet supported"로 실패했습니다.

**정정된 진단:** 실패는 "op이 미지원"이어서가 아니라, 그 op의 **primitive 인스턴스가
DeltaNet 그래프의 특정 위치/문맥(IR상 `context: Sub`)에서 tactic(하드웨어 실행계획)을
할당받지 못해서**입니다. 컴파일러는 elementwise를 인접 matmul/conv/attention "앵커" 커널에
**융합(fuse)**하는 식으로 tactic을 붙이는데, DeltaNet의 게이트·순환 스캔 elementwise는
융합할 앵커가 없거나(또는 스테이지 출력으로 고립돼) tactic이 없습니다. qwen3_moe에선 같은
sigmoid/mul이 MLP/attention matmul에 융합돼 통과합니다.

→ 따라서 진짜 부재한 것은 **op 커널이 아니라 "gated-delta 스캔을 하나로 융합하는 tactic"**
입니다. FLA(GPU)의 `use_qk_l2norm_in_kernel`·`chunk_gated_delta_rule` 커널이 하는 일 —
게이트·l2norm·contraction·상태갱신을 한 커널로 융합 — 을 NPU 컴파일러가 못 합니다.

---

## 3. 컴파일러 커널이 "어디" 있고 왜 못 넣나 (radare2 검증)

**위치 (strings/radare2 실측):** npu-tools(git `3f23a71`) Rust 워크스페이스가
`native_torch.so`(105MB)·`native_llm_common.so`(143MB)에 **정적 링크**.
- 커널라이저: `npu-compiler/crates/npu-compiler-kernelize/src/kernelize.rs` (미지원 op 거부:
  `" is not an operator that is yet supported by the compiler"` @native_torch 0x008798c4).
- op 의미/ALU primitive: `npu-executor-common/src/operator/calculate/{aten_ops/aten_impl.rs,
  primitive_ops.rs}` (`calculate/add`, `calculate/attention_kernel`, `calculate/conv` …).
- op-name→handler: **phf(perfect-hash) 정적 테이블** (런타임 레지스트리 아님; miss 시
  "no entry found for key" 패닉). `ldd`상 외부 libnpu/libfuriosa 의존 없음(완전 static),
  `dlopen`은 CUDA 드라이버 로드용(`libloading`)뿐 — **op 플러그인 메커니즘 없음**.

**왜 radare2로 못 넣나:**
- 새 커널 = 새 lowering 패스 + tactic 선택 + 스케줄러 엔트리 + TU/VE/DPE 명령 코드젠.
  이는 **레지스터 할당·재배치된 새 기계어 수천 바이트**라, 스트립된 105MB 바이너리에
  손으로 어셈블해 끼워넣을 수 없음(빈 stub·점프슬롯·리다이렉트할 심볼이 없음).
- `allow_external_operators`는 **이미 컴파일된 EDF 블롭**만 받음(코드젠이 이미 있어야 함).
  `allow_unlowered_operators`는 op을 DRAM IR에 남길 뿐 실행 불가 — **실측: 빌드 hang**.
- `furiosa::` 커스텀 op 네임스페이스도 전부 **기존 aten op으로의 분해**거나 사전컴파일 EDF
  래퍼 — 새 코드젠 없음.

**유일한 정공법:** Furiosa의 npu-tools 소스로 (a) gated-delta 스캔 tactic 커널을 추가하고
(b) 런타임에 순환상태 버퍼를 넣어 재컴파일 → 즉 **벤더 2026.3+ 릴리스**. (이는 serve
런타임이 순환상태를 관리 못 하는 것과 **같은 뿌리** — NPU 스택 전체가 트랜스포머
matmul/conv 중심.)

---

## 4. 그래서 지금 할 수 있는 것 / 할 수 없는 것

- ✅ **표준 트랜스포머/MoE(qwen3_moe 등)**: 위장(masquerade)으로 serve. 실측: BF16
  Qwen3-Coder-30B-A3B-Instruct를 `model_type=qwen3`로 위장해 2장(pp2)에서 정상 코드 생성
  (`qwen3-coder-30b-a3b-inst-tp8-65k-tc`).
- ❌ **Gated DeltaNet(qwen3_next/3.5/3.6)**: 컴파일 자체가 막힘. radare2·config·Python
  재정식화 모두 불가. 벤더 융합 스캔 커널 필요.
- 빌드 파이프라인의 **열린 Python 단계는 우리가 전부 뚫었음**(게이트 우회, TP 분할,
  transform.py 노드복제, as_strided/log1p/pad 제거) — 벽은 오직 폐쇄 컴파일러의 tactic.

**벤더에 요청할 것 (구체):** npu-compiler에 **gated linear-attention(gated delta rule)
스캔 tactic 커널** 추가 — 게이트(sigmoid/softplus/exp)·q/k l2norm·청크 contraction·rank-1
상태갱신을 하나로 융합, + npu-runtime에 **레이어별 고정크기 순환상태 버퍼**. 이 둘이면
qwen3_next뿐 아니라 같은 구조의 qwen3.5/3.6도 커버됩니다.
