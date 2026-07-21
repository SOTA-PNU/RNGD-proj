# Gated DeltaNet와 Gated Attention 정리 — Qwen3-Next(qwen3_next) 이해를 위한 스터디 노트

작성일: 2026-06-15. 모든 식과 수치는 아래 "참고자료"의 논문 본문, 그리고 로컬에 설치된 Hugging Face transformers 구현(`modeling_qwen3_next.py`)과 실제 모델 config로 검증했습니다.

---

## 0. 이 문서를 왜 쓰는지

Qwen3-Next(model_type `qwen3_next`)는 기존 트랜스포머와 구조가 꽤 다릅니다. 레이어의 4분의 3을 "Gated DeltaNet"이라는 선형 어텐션으로 바꾸고, 나머지 풀 어텐션 레이어에도 "gated attention"을 붙였습니다. 이 두 가지를 제대로 이해하려면 (1) 선형 어텐션과 delta rule이 무엇인지, (2) 거기에 "gating"이 뭘 더하는지, (3) Qwen3-Next가 이걸 어떻게 조립했는지, (4) 왜 이 구조가 NPU의 paged-KV 런타임과 안 맞는지를 순서대로 봐야 합니다. 이 문서는 식을 직접 유도하면서 그 네 가지를 정리합니다.

---

## 1. 선형 어텐션과 delta rule

### 1.1 softmax 어텐션이 비싼 이유, 선형 어텐션의 아이디어

보통의 softmax 어텐션은 토큰 t의 출력을 이렇게 계산합니다.

```
o_t = Σ_{i≤t}  [ exp(q_t·k_i) / Σ_{j≤t} exp(q_t·k_j) ] · v_i
```

분자에 `exp(q_t·k_i)`가 들어 있어서, 모든 토큰 쌍(t, i)에 대해 점수를 따로 계산해야 합니다. 길이 N 시퀀스면 N×N개의 점수가 나오므로 학습 시 계산량이 O(N²), 그리고 추론 시에는 과거의 K/V를 전부 들고 있어야 합니다(이게 KV 캐시입니다).

선형 어텐션의 핵심은 "`exp(q·k)`라는 커널을 두 개의 특징사상(feature map) 내적 `φ(q)·φ(k)`로 근사하면, 합의 순서를 바꿔서 N×N 점수를 안 만들어도 된다"는 것입니다. 분모를 떼고(최근 구현들은 수치 안정성 때문에 분모를 제거합니다) `φ`를 항등사상으로 두면 매우 단순해집니다 (DeltaNet 논문 §2.1, [Yang et al. 2024]):

```
S_t = S_{t-1} + v_t k_tᵀ          (상태 갱신, S는 d×d 행렬)
o_t = S_t q_t                      (읽기)
```

여기서 `S_t = Σ_{i≤t} v_i k_iᵀ` 는 "지금까지 본 key→value 연관"을 한 개의 행렬에 누적한 메모리입니다. 토큰 하나당 행렬 갱신과 행렬-벡터 곱만 하면 되므로 시퀀스 길이에 대해 O(N) 입니다. 더 중요한 점은, 더 이상 과거 토큰들의 K/V를 전부 저장할 필요가 없다는 것입니다 — 고정 크기(d×d)의 상태 행렬 `S` 하나만 들고 다니면 됩니다. (DeltaNet 논문 §2.1)

### 1.2 단순 누적의 한계 → delta rule

위 단순 선형 어텐션의 문제는 "갱신이 덧셈뿐"이라는 점입니다. 새 연관을 계속 더하기만 하므로 옛 연관을 지울 방법이 없습니다. 시퀀스 길이 L이 차원 d를 넘으면 key들이 "충돌(collision)"하면서 메모리가 뭉개집니다 (DeltaNet 논문 §2.2, Schlag et al. 2021 인용). 좋은 모델이라면 새 정보를 넣을 자리를 만들기 위해 덜 중요한 연관을 지울 줄 알아야 합니다.

DeltaNet은 여기에 **delta rule**(= Widrow-Hoff 학습 규칙, 1960)을 도입합니다. 갱신 전에 현재 key로 옛 값을 한 번 "읽어 보고", 예측과 실제 target의 차이(=델타)만큼만 고칩니다 (DeltaNet 논문 §2.2):

```
S_t = S_{t-1} − β_t (S_{t-1} k_t − v_t) k_tᵀ
```

`S_{t-1} k_t`는 현재 메모리가 key `k_t`에 대해 내놓는 예측값(옛 value), `v_t`는 정답 target, `β_t ∈ (0,1)`는 학습률(쓰기 강도)입니다. 이 식은 두 가지로 해석됩니다.

**(a) 온라인 경사하강(SGD) 한 스텝.** 회귀 손실 `L_t(S) = ½‖S k_t − v_t‖²`를 정의하면 (DeltaNet 논문 §2.2):

```
S_t = S_{t-1} − β_t ∇L_t(S_{t-1}) = S_{t-1} − β_t (S_{t-1} k_t − v_t) k_tᵀ
```

즉 메모리 행렬 S는 "key→value 회귀"를 매 토큰마다 한 스텝씩 SGD로 학습하는 빠른 가중치(fast weight)입니다. 이 관점이 뒤(2장)의 "gating = weight decay"라는 통찰로 이어집니다.

**(b) 읽고-지우고-쓰기.** 같은 식을 풀어 쓰면 (DeltaNet 논문 §2.2):

```
v_old = S_{t-1} k_t            (옛 값 읽기)
v_new = β_t v_t + (1−β_t) v_old   (옛 값과 새 값을 보간)
S_t = S_{t-1} − v_old k_tᵀ + v_new k_tᵀ
                 └─ 지우기 ─┘   └─ 쓰기 ─┘
```

`β_t = 1`이면 옛 값을 완전히 지우고 새 값으로 덮어쓰고, `β_t = 0`이면 메모리를 건드리지 않습니다. 이게 단순 덧셈 대비 DeltaNet의 핵심 차이입니다.

위 갱신식을 다시 묶으면 **일반화된 Householder 변환** 형태가 됩니다 (Gated DeltaNet 논문, Table 1 / DeltaNet §3):

```
S_t = S_{t-1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
```

괄호 안 `(I − β_t k_t k_tᵀ)`가 전이행렬입니다. 즉 DeltaNet은 "전이행렬이 항등행렬이 아니라 Householder인 행렬값 RNN"입니다. 이 재해석이 학습 병렬화의 열쇠입니다.

### 1.3 순환형(recurrent) vs 청크 병렬형(chunkwise)

위 식들은 토큰을 하나씩 순서대로 처리하는 **순환형**입니다. 추론(특히 1토큰씩 만드는 decode)에는 좋지만, 학습 때는 시퀀스 방향으로 병렬화가 안 되고, 또 행렬-벡터 곱이 대부분이라 GPU/NPU의 행렬곱 가속기(tensor core 등)를 못 씁니다 (DeltaNet 논문 §2.1).

반대편 극단인 **완전 병렬형**은 `O = (QKᵀ ⊙ M) V` 처럼 한 번에 다 곱하지만, 길이 L에 대해 O(L²d) FLOPs라 길어지면 비쌉니다.

**청크 병렬형(chunkwise)**은 그 중간입니다. 시퀀스를 길이 C짜리 청크로 쪼개고(보통 C=64 또는 128), "청크 사이는 순환(상태 S를 청크→청크로 넘김), 청크 안은 병렬(행렬곱)"로 처리합니다 (DeltaNet 논문 §2.1). 단순 선형 어텐션이면 이건 쉽습니다 (DeltaNet 식 1, 2):

```
S_{[t+1]} = S_{[t]} + V_{[t]}ᵀ K_{[t]}                          (청크 간: 상태 carry)
O_{[t]}   = Q_{[t]} S_{[t]}ᵀ + (Q_{[t]} K_{[t]}ᵀ ⊙ M_C) V_{[t]}   (청크 내: 병렬)
```

`Q_{[t]} S_{[t]}ᵀ`는 직전 청크까지의 상태에서 읽은 **inter-chunk(청크 간)** 기여이고, `(QKᵀ⊙M)V`는 현재 청크 안의 **intra-chunk(청크 내)** 기여입니다. C=L이면 완전 병렬형, C=1이면 순환형으로 자연스럽게 환원됩니다.

DeltaNet에서 청크 병렬화가 어려운 이유는 전이행렬이 항등행렬이 아니라 Householder `(I − β_t k_t k_tᵀ)`라서, 청크 내에서 이 행렬들의 누적곱 `∏(I − β k kᵀ)`이 등장하기 때문입니다. 이걸 매 스텝 행렬로 펼치면 메모리가 터집니다. DeltaNet 논문의 기여가 바로 이 누적곱을 **WY 표현(WY representation)** 으로 압축하는 것입니다 (DeltaNet §3.1, 식 4–7; Householder 곱에 대한 고전 결과 Bischof & Loan 1985):

```
P_{[t]} = I − W_{[t]}ᵀ K_{[t]}              (Householder 누적곱을 W,K로 압축)
H_{[t]} = U_{[t]}ᵀ K_{[t]}
```

그리고 W, U는 **UT 변환(UT transform)** 으로 한 번에 구합니다 (Gated DeltaNet 논문 식 6–7, Joffrain et al. 2006 인용):

```
T_{[t]} = (I + strictLower(diag(β_{[t]}) K_{[t]} K_{[t]}ᵀ))⁻¹ diag(β_{[t]})
W_{[t]} = T_{[t]} K_{[t]},   U_{[t]} = T_{[t]} V_{[t]}
```

여기서 등장하는 하삼각 행렬의 역행렬(`T`)이, 로컬 구현에서 forward-substitution 루프로 풀리는 부분입니다. transformers의 `torch_chunk_gated_delta_rule`에서 그대로 확인됩니다 — 청크 크기만큼 도는 삼각역행렬 정련 루프입니다 (`modeling_qwen3_next.py:511-515`):

```python
for i in range(1, chunk_size):
    row = attn[..., i, :i].clone()
    sub = attn[..., :i, :i].clone()
    attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
attn = attn + torch.eye(chunk_size, ...)   # = T (= (I - strictLower)^{-1})
```

이렇게 W, U를 구해 두면 청크 갱신·출력이 전부 행렬곱(matmul)으로 떨어져서 tensor core를 쓸 수 있습니다 (DeltaNet 식 8, 9):

```
S_{[t+1]} = S_{[t]} P_{[t]} + H_{[t]} = S_{[t]} + (U_{[t]} − W_{[t]} S_{[t]}ᵀ)ᵀ K_{[t]}
O_{[t]}   = Q_{[t]} S_{[t]}ᵀ + (Q_{[t]} K_{[t]}ᵀ ⊙ M)(U_{[t]} − W_{[t]} S_{[t]}ᵀ)
```

요약하면: **순환형은 추론에 좋고, 청크 병렬형은 학습에 좋다. 둘은 수학적으로 같은 결과를 내며, 청크 크기 C로 그 사이를 조절한다.** 로컬 구현이 두 함수를 따로 가지고 있는 것도 이 때문입니다 — prefill/학습용 `torch_chunk_gated_delta_rule`(`:467`)와 1토큰 decode용 `torch_recurrent_gated_delta_rule`(`:547`).

---

## 2. "gated"가 더하는 것

### 2.1 Gated DeltaNet: 데이터 의존 감쇠(망각) 게이트

플레인 DeltaNet에는 시간에 따른 감쇠가 없습니다. delta rule이 "정밀한 부분 수정"은 잘 하지만, 갑자기 화제가 바뀌어 메모리 전체를 빠르게 비워야 할 때는 느립니다. 반대로 Mamba2/GLA류는 스칼라/벡터 감쇠 게이트로 메모리 전체를 빠르게 잊을 수 있지만, 특정 연관만 콕 집어 고치는 건 못 합니다. 이 둘은 상보적입니다 (Gated DeltaNet 논문 초록·§3.1: "gating enables rapid memory erasure while the delta rule facilitates targeted updates").

비교를 위해 Mamba2의 갱신부터 보면 (Gated DeltaNet 논문 §2.1):

```
S_t = α_t S_{t-1} + v_t k_tᵀ,     α_t ∈ (0,1)   (데이터 의존 스칼라 감쇠)
```

`α_t`가 매 스텝 모든 key-value 연관을 균일하게 감쇠시킵니다. 단순 망각은 되지만 delta rule의 정밀 수정은 없습니다.

**Gated DeltaNet은 두 메커니즘을 한 식으로 합칩니다** (Gated DeltaNet 논문 식 10):

```
S_t = S_{t-1} ( α_t (I − β_t k_t k_tᵀ) ) + β_t v_t k_tᵀ
```

- `α_t`(감쇠/망각 게이트): 데이터 의존 스칼라, 상태 전체를 적응적으로 감쇠 → 빠른 망각
- `β_t (I − β_t k_t k_tᵀ)`(delta rule): 특정 key→value 연관을 정밀 수정
- `α_t → 1`이면 순수 delta rule, `β_t → 0`이면 순수 Mamba2 감쇠로 환원됩니다 (Gated DeltaNet §1)

이 통합이 깔끔한 이유는 **온라인 학습 관점**에서 드러납니다. Gated DeltaNet 논문 Table 1은 여러 선형 RNN을 "온라인 학습 목적함수"로 통일합니다:

| 방법 | 온라인 목적함수 | 갱신식 |
|---|---|---|
| Linear Attn | `‖S_t − S_{t-1}‖²_F − 2⟨S_t k_t, v_t⟩` | `S_t = S_{t-1} + v_t k_tᵀ` |
| Mamba2 | `‖S_t − α_t S_{t-1}‖²_F − 2⟨S_t k_t, v_t⟩` | `S_t = α_t S_{t-1} + v_t k_tᵀ` |
| DeltaNet | `‖S_t − S_{t-1}‖²_F − 2⟨S_t k_t, β_t(v_t − S_{t-1}k_t)⟩` | `S_t = S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ` |
| **Gated DeltaNet** | `‖S_t − α_t S_{t-1}‖²_F − 2⟨S_t k_t, β_t(v_t − α_t S_{t-1}k_t)⟩` | `S_t = S_{t-1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ` |

`‖S_t − S_{t-1}‖²_F` 항은 "이전 상태에서 너무 멀어지지 마라"는 정규화(메모리 유지)입니다. 그런데 메모리가 정보로 포화되면 이 유지 항이 오히려 독이 됩니다(여러 정보가 겹쳐 저장돼 정확히 못 꺼냄). `α_t`를 곱해 정규화를 `‖S_t − α_t S_{t-1}‖²_F`로 느슨하게 풀어 주면, 필요할 때 옛 상태에서 의도적으로 벗어나 선택적으로 잊을 수 있습니다 (Gated DeltaNet §3.1).

같은 통찰을 fast-weight/SGD 관점으로 보면, delta rule은 회귀 손실 `L(S) = ½‖S k_t − v_t‖²`에 대한 test-time SGD 한 스텝이고, **gating(α_t)은 거기에 적응적 weight decay를 넣은 것**입니다 (Gated DeltaNet §3.1: "the gated delta rule can be viewed as incorporating an adaptive weight decay term α_t into the SGD update"). 딥러닝에서 흔히 쓰는 weight decay와 같은 역할 — 망각, 안정화, 발산 방지 — 을 메모리 행렬에 적용하는 셈입니다.

실제 효과는 논문 Table 2(S-NIAH "건초더미 속 바늘찾기")에서 분명합니다. 8K 길이 number-in-haystack에서 DeltaNet 14.4, Mamba2 17.0인데 Gated DeltaNet 29.6 — 둘의 장점을 합쳐 둘 다를 능가합니다 (Gated DeltaNet Table 2).

**로컬 구현에서의 게이트.** transformers의 Qwen3-Next는 `α_t`를 직접 저장하지 않고, 로그-감쇠 `g_t`를 통해 만듭니다. `β_t`는 시그모이드입니다 (`modeling_qwen3_next.py:756-758`):

```python
beta = b.sigmoid()
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)   # g = log α_t ≤ 0
```

즉 `α_t = exp(g_t) ∈ (0,1)` (g가 항상 0 이하이므로). 순환 구현을 보면 식 10과 정확히 대응됩니다 (`:577-580`):

```python
last_recurrent_state = last_recurrent_state * g_t          # × α_t  (감쇠)
kv_mem = (last_recurrent_state * k_t...).sum(dim=-2)        # α_t S_{t-1} k_t  (옛 값 읽기)
delta  = (v_t - kv_mem) * beta_t                            # β_t (v_t − α_t S_{t-1} k_t)
last_recurrent_state = last_recurrent_state + k_t... * delta...   # + delta·kᵀ
```

이것이 식 10 `S_t = S_{t-1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ`의 한 토큰 전개입니다. 청크 구현(`:507-537`)도 같은 수학을 `g.cumsum`(누적 감쇠) + decay mask + WY/삼각역행렬로 병렬화한 것뿐입니다.

### 2.2 Gated attention: 어텐션 "출력"에 거는 게이트

위 1·2.1절의 gating은 *상태 메모리*에 거는 게이트(감쇠)였습니다. **"gated attention"은 이와 다른, 어텐션 출력에 거는 게이트**입니다. softmax 어텐션(SDPA)의 결과 벡터에 시그모이드/SiLU 게이트를 원소별로 곱합니다.

핵심 논문은 [Qiu et al. 2025, "Gated Attention for Large Language Models"] (arXiv 2505.06708, NeurIPS 2025 Best Paper)인데, 결론이 단순합니다 — **헤드별 시그모이드 게이트를 SDPA 출력 뒤에 붙이는 작은 수정이 일관되게 성능을 올린다.** 왜 좋은지를 두 가지로 설명합니다 (논문 초록):

1. **비선형성 추가.** softmax 어텐션은 본질적으로 V에 대한 가중평균(저랭크 선형사상)이라 표현력이 제한됩니다. 출력에 query 의존 게이트를 곱하면 이 저랭크 사상에 비선형성이 들어갑니다.
2. **query 의존 희소 게이팅.** 게이트 점수가 query에 따라 희소해지면서 SDPA 출력을 선택적으로 통과시킵니다. 이게 "attention sink"(첫 토큰 등에 어텐션이 쏠리는 현상)를 완화하고, 학습 안정성·더 큰 학습률 허용·긴 문맥 외삽을 개선합니다 (논문 초록: "mitigates 'attention sink' and enhances long-context extrapolation").

**Qwen3-Next의 풀 어텐션이 바로 이 gated attention입니다.** 로컬 구현에서 q_proj가 보통의 두 배 크기를 내놓고, 절반을 query로, 절반을 게이트로 씁니다 (`modeling_qwen3_next.py:358-359, 387-390, 419-420`):

```python
self.q_proj = nn.Linear(hidden, num_heads * head_dim * 2, ...)   # 2배 크기
...
query_states, gate = torch.chunk(self.q_proj(hidden).view(..., -1, head_dim*2), 2, dim=-1)
...
attn_output = attn_output.reshape(*input_shape, -1).contiguous()
attn_output = attn_output * torch.sigmoid(gate)     # ← 출력 게이트(시그모이드)
attn_output = self.o_proj(attn_output)
```

즉 `output = o_proj( SDPA(q,k,v) ⊙ σ(gate) )`. 게이트가 query에서 나오므로 query 의존이고, 시그모이드라 (0,1) 범위입니다 — 논문이 권장한 형태 그대로입니다.

참고로 Gated DeltaNet 쪽에도 비슷한 "출력 게이트"가 따로 있습니다. core 출력에 z(게이트)로 SiLU 게이팅된 RMSNorm을 겁니다 (`Qwen3NextRMSNormGated`, `:66-81`):

```python
hidden_states = hidden_states * F.silu(gate.to(torch.float32))   # SiLU 출력 게이트 + norm
```

정리하면 Qwen3-Next에는 게이트가 세 종류 있습니다: (a) DeltaNet 상태 감쇠 게이트 `α_t`, (b) DeltaNet 출력의 SiLU gated-RMSNorm, (c) 풀 어텐션 출력의 시그모이드 gated-attention. 이름이 다 "gated"라 헷갈리기 쉬운데 거는 위치가 각각 다릅니다.

---

## 3. Qwen3-Next의 구체 구조

대상 모델: `Qwen/Qwen3-Coder-Next-FP8`. 아래 수치는 로컬 config(`~/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next-FP8/snapshots/da6e2ed27304dd39abadd9c82ef50e8de67bdd4c/config.json`)와 transformers의 config 클래스로 직접 계산해 확인했습니다.

### 3.1 레이어 인터리빙 (선형 : 풀 = 3 : 1)

- 총 48레이어 (`num_hidden_layers: 48`), `full_attention_interval: 4`
- 실제 `layer_types`(config 클래스가 계산)를 펼쳐 보면: `linear, linear, linear, full` 패턴이 12번 반복 → **36개 linear_attention + 12개 full_attention**. 4의 배수 인덱스(3,7,11,…,47)마다 풀 어텐션이 들어갑니다.
- 즉 레이어의 75%가 Gated DeltaNet, 25%가 (gated) 풀 어텐션입니다. Qwen 측 설명도 "Replaces standard attention with the combination of Gated DeltaNet and Gated Attention" 라는 3:1 하이브리드입니다 (Qwen3-Next 모델 카드).
- 왜 섞나: 순수 선형 어텐션은 빠르고 메모리는 좋지만 정밀 검색(in-context retrieval)이 약하고, 풀 어텐션은 정확하지만 비쌉니다. 주기적으로 끼운 풀 어텐션 레이어가 "정확한 전역 참조"를 담당하고, 다수의 Gated DeltaNet이 비용을 낮춥니다 (DeltaNet 논문 §1: 글로벌 어텐션 2층을 섞은 하이브리드가 강한 트랜스포머 baseline을 능가).

레이어 디스패치는 `modeling_qwen3_next.py:908-912`에서 `layer_types[layer_idx]`로 갈립니다(`linear_attention` → `Qwen3NextGatedDeltaNet`, `full_attention` → `Qwen3NextAttention`).

### 3.2 선형(Gated DeltaNet) 레이어 헤드 구성

config에서:
- `linear_num_key_heads: 16`, `linear_num_value_heads: 32` — **key 헤드 16개, value 헤드 32개** (value 헤드가 2배)
- `linear_key_head_dim: 128`, `linear_value_head_dim: 128` — head_dim 128
- `linear_conv_kernel_dim: 4` — depthwise short-conv 커널 크기 4

value 헤드가 key 헤드의 2배라서, 구현은 query/key를 `repeat_interleave(2)`로 늘려 32개로 맞춥니다 (`:759-761`). short-conv는 q/k/v를 합친 채널에 거는 depthwise causal Conv1d로, 토큰 직전 몇 개를 섞어 지역 패턴을 잡습니다(Mamba 계열의 공통 트릭) (`:608-615`, `conv_dim = key_dim*2 + value_dim`). 그래서 DeltaNet 레이어의 추론 캐시에는 상태 `S` 외에 conv 상태도 함께 들어갑니다(4.절).

### 3.3 풀 어텐션 레이어 구성 (gated, partial RoPE)

config에서:
- `num_attention_heads: 16`, `num_key_value_heads: 2` — **q 헤드 16, kv 헤드 2** (GQA, 그룹 8)
- `head_dim: 256` — 풀 어텐션 head_dim은 256 (선형 레이어의 128보다 큼)
- `partial_rotary_factor: 0.25` — **부분 RoPE**: head_dim의 25%(256×0.25=64)에만 회전 위치인코딩을 적용. 나머지는 위치 정보 없이 통과.
- 출력 게이트(2.2절): q_proj가 `16 × 256 × 2` 크기를 내고 절반을 시그모이드 게이트로 사용.
- q/k에 head_dim 단위 RMSNorm(`q_norm`, `k_norm`, `:370-373`) — QK-Norm.

### 3.4 MoE

config에서:
- `num_experts: 512`, `num_experts_per_tok: 10` — **512개 전문가 중 토큰당 10개(top-k=10) 활성**
- `shared_expert_intermediate_size: 512`, `moe_intermediate_size: 512` — 공유 전문가 1개 + 작은 전문가 intermediate 512
- `decoder_sparse_step: 1` — 모든 디코더 레이어가 MoE (mlp_only_layers 비어 있음)
- `norm_topk_prob: true` — top-k 라우팅 가중치 정규화

MoE 블록은 `Qwen3NextSparseMoeBlock`(`:880`)이고, `(layer_idx+1) % decoder_sparse_step == 0`이면 MLP 대신 MoE를 씁니다(`:915-917`). 80B급 모델인데 토큰당 활성 파라미터가 3B 수준인 건 이 극단적 희소 MoE(512중 10) 때문입니다 (Qwen3-Next 모델 카드: total 80B / activated 3B).

### 3.5 기타

- `hidden_size: 2048`, `vocab_size: 151936`, `max_position_embeddings: 262144`(약 256K 문맥), `rope_theta: 5000000`
- 양자화: FP8 blockwise(`weight_block_size: [128,128]`, dynamic). 단 라우터 게이트·conv1d·in_proj_ba 등 민감 모듈은 FP8 변환 제외(`modules_to_not_convert`).
- 안정화: Qwen 측은 zero-centered & weight-decayed layernorm 등을 추가했다고 밝힙니다 (모델 카드).

근거 코드/설정 위치:
- 구현: `/home/jun/furiosa/lib/python3.12/site-packages/transformers/models/qwen3_next/modeling_qwen3_next.py`
  - delta rule(청크): `:467-544`, delta rule(순환): `:547-586`, GatedDeltaNet forward: `:685-800`, gated attention: `:387-420`, gated RMSNorm: `:66-81`, 캐시: `:84-171`, 레이어 디스패치: `:908-917`
- 설정: 위 config.json

---

## 4. 왜 순환형은 NPU의 paged-KV 런타임에서 어려운가

이 부분은 우리 프로젝트에서 직접 막혀 본 지점이라, 로컬 메모리(`qwen3-next-blocker.md`)와 코드로 근거를 댑니다.

### 4.1 두 종류의 캐시가 근본적으로 다르다

transformers의 `Qwen3NextDynamicCache` 주석과 코드가 차이를 명확히 보여 줍니다 (`modeling_qwen3_next.py:84-95`).

- **풀 어텐션 레이어**의 캐시는 `key_cache`/`value_cache`이고, 모양이 `(batch, num_heads, seq_len, head_dim)` — **seq_len 축이 있어 토큰마다 늘어납니다.** 갱신은 단순 append입니다 (`:128-129`):
  ```python
  self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
  ```
  이게 바로 paged-KV가 다루도록 설계된 대상입니다. 새 토큰의 K/V를 빈 슬롯에 **덧붙이기만(append-only)** 하면 됩니다.

- **선형(Gated DeltaNet) 레이어**의 캐시는 `recurrent_states`와 `conv_states`이고, 모양이 `(batch, d_inner, d_state)` — **seq_len 축이 없는 고정 크기 행렬**입니다(주석: "constant shape regardless of seq_len"). 갱신은 append가 아니라 **읽고-수정하고-다시 쓰는(read-modify-write)** 방식입니다 (`:577-580`, `:789`):
  ```python
  recurrent_state = recurrent_state * α_t            # 읽고
  recurrent_state = recurrent_state + k_t ⊗ delta    # 수정해서
  cache_params.recurrent_states[layer_idx] = last_recurrent_state   # 같은 자리에 덮어씀
  ```

### 4.2 paged-KV가 이걸 못 담는 이유

paged-KV 캐시는 softmax 어텐션의 K/V를 위해 설계된 자료구조입니다. 토큰별 K/V를 고정 크기 "페이지(블록)"에 채워 넣고, 시퀀스가 길어지면 페이지를 더 할당하는 식입니다. 동작의 전제가 두 가지입니다: (1) 저장 대상이 토큰마다 하나씩 생기는 (K, V) 쌍이고, (2) 갱신이 **append-only** 라는 것.

Gated DeltaNet의 순환 상태 `S_t`는 이 전제를 둘 다 깹니다:

1. **append가 아니라 RMW다.** `S_t`는 토큰을 추가하는 게 아니라, 같은 (key_dim × value_dim) 행렬을 매 스텝 제자리에서 감쇠시키고(×α_t) 갱신합니다. paged-KV에는 "기존 슬롯을 읽어서 곱하고 더해 되쓰는" 연산 모델 자체가 없습니다. 우리 로컬 분석으로도, furiosa-llm 런타임이 KV 슬롯을 append식으로 소유하고 별칭(aliasing)으로 제자리 수정할 Python 훅이 없어서, KV로 위장해도 RMW가 불가능했습니다 (`qwen3-next-blocker.md`).

2. **저장 대상이 (K,V) 쌍이 아니라 상태 행렬 + conv 상태다.** 아티팩트/런타임 계약이 (K, V) 동형 텐서를 강제하고, "상태 원본(state origin)"이라는 개념이 없습니다. SSM/선형어텐션용 상태 풀이 런타임에 아예 존재하지 않습니다.

즉 풀 어텐션 레이어와 선형 레이어는 디코드 시 **본질적으로 다른 메모리 모델**을 요구합니다. 같은 모델 안에서 한쪽은 append-only KV(풀 어텐션 12층), 다른 쪽은 RMW 순환 상태(선형 36층)가 동시에 굴러가야 하는데, 기존 NPU serve 스택은 전자만 압니다.

### 4.3 그래서 무엇이 막히고, 무엇은 되는가

우리 실측 결론을 정확히 옮기면 (`qwen3-next-blocker.md`):

- **컴퓨트(연산) 자체는 NPU에서 됩니다.** delta rule 한 스텝, 청크 풀바디, inter-chunk 상태 carry, 멀티헤드, 실제 차원(d=128), 투영까지 손수 작성한 TK-graph 커널로 NPU에서 HF 참조와 ~1e-7 일치시켰고, 실제 Qwen3-Coder-Next-FP8(80B) end-to-end 코드 생성·OpenAI 호환 서빙까지 host 추론 루프로 성공했습니다. 즉 "연산이 NPU에서 안 된다"는 게 아닙니다.
- **막히는 건 deploy 경로, 그중에서도 autoregressive serve의 cross-step 상태 관리입니다.** host가 직접 추론 루프를 돌리면(상태 S와 conv 상태를 host가 RMW로 소유, per-step 커널만 NPU) 됩니다. 하지만 표준 furiosa-llm `serve`에 태우려면, 런타임이 순환 상태 풀을 가져야 하고 native Loop 노드와 EDF 아티팩트 계약(append-only KV 전제)을 바꿔야 합니다. 이건 벤더 컴파일러/런타임 변경(2026.3+ 기대) 없이는 불가능합니다.

**한 줄 요약:** Gated DeltaNet의 순환 상태는 "토큰마다 늘어나는 append-only KV"가 아니라 "매 스텝 제자리에서 읽고-쓰는 고정 크기 행렬"이라, softmax 어텐션 전용으로 설계된 paged-KV 캐시에 담거나 갱신할 수 없습니다. 이것이 표준 NPU serve를 위해 벤더 런타임 변경이 필요한 핵심 이유입니다.

---

## 참고자료 (References)

논문
- Yang, Wang, Shen, Panda, Kim. "Parallelizing Linear Transformers with the Delta Rule over Sequence Length." NeurIPS 2024. arXiv:2406.06484. https://arxiv.org/abs/2406.06484 — 선형 어텐션/recurrent·parallel·chunkwise 식 1–2, delta rule·SGD 해석 §2.2, WY 재파라미터화 §3.1
- Yang, Kautz, Hatamizadeh. "Gated Delta Networks: Improving Mamba2 with Delta Rule." ICLR 2025. arXiv:2412.06464. https://arxiv.org/abs/2412.06464 — Mamba2/DeltaNet 비교 §2, WY/UT 변환 식 3–9, 게이트 delta rule 식 10, 온라인 학습 Table 1, S-NIAH Table 2. 공식 구현: https://github.com/NVlabs/GatedDeltaNet
- Qiu, Wang, Zheng, et al. "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free." NeurIPS 2025 (Best Paper). arXiv:2505.06708. https://arxiv.org/abs/2505.06708 — SDPA 출력 시그모이드 게이트, 비선형성·희소성·attention sink 완화. 구현: https://github.com/qiuzh20/gated_attention
- 배경: Schlag et al. 2021 (Delta rule fast weights); Widrow & Hoff 1960 (LMS/Widrow-Hoff); Dao & Gu 2024 (Mamba2/SSD); Liu et al. 2024 (Longhorn, online-learning framework) — 위 두 DeltaNet 논문에서 인용.

Qwen3-Next 아키텍처
- Qwen 공식 모델 카드: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking — 3:1 하이브리드, Gated DeltaNet+Gated Attention, 512 expert/10 active/1 shared, total 80B / activated 3B
- Hugging Face transformers 구현(로컬): `/home/jun/furiosa/lib/python3.12/site-packages/transformers/models/qwen3_next/modeling_qwen3_next.py`
- 모델 설정(로컬): `~/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next-FP8/snapshots/da6e2ed27304dd39abadd9c82ef50e8de67bdd4c/config.json`

NPU/serve 제약(프로젝트 실측)
- `/home/jun/.claude/projects/-home-jun-RNGD-proj/memory/qwen3-next-blocker.md`
- 관련 프로젝트 문서: `Model_Benchmark/info/README_qwen3_next_TECH.md`, `README_qwen3_next_RUN.md`, `README_qwen3_next_ARTIFACT.md`
