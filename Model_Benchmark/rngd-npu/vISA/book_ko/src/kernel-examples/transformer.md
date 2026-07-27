# Transformer 구조



이 장은 transformer 모델의 각 연산을 구체적인 TCP 하드웨어 구성 요소에 대응시킨다.
Llama 3 70B 를 예제로 삼는다.
디코더 전용 모델이므로 두 단계(입력 인코딩을 위한 prefill 과 토큰 생성을 위한 decode)를 거치며, 각 단계의 TCP 하드웨어 특성이 다르다.

## 모델 매개변수

다음 매개변수들이 Llama 3 70B 구조를 정의하며, 이 장 전체의 모든 shape 표기에서 참조된다:

**시퀀스 차원** (입출력 길이를 제어):
- `B`: 배치 크기
- `s_in`: 입력 시퀀스 길이
- `s_max`: 최대 시퀀스 길이/컨텍스트 길이
- `s`: 지금까지 처리한 전체 시퀀스 길이 (prefill + decode)

**모델 크기** (어휘 수와 레이어 수):
- `V = 128256`: 어휘 크기
- `D = 8192`: 은닉 차원/임베딩 크기
- `F = 28672`: FFN up projection 의 중간 차원
- `L = 80`: 레이어 수

**어텐션 헤드 차원** (어텐션을 어떻게 분할하는지):
- `h_q = 64`: 쿼리 헤드 개수
- `h_kv = 8`: 키/값 헤드 개수
- `G = 8`: 어텐션 그룹 개수 (`= h_q / h_kv`)
- `d_k = 128`: 헤드 차원 (`D / h_q` 와 같음)
- `d_k_prime = 64`: RoPE 계산을 위해 분할한 헤드 차원
- `f = 2`: 인접 헤드의 주파수 차원 (`d_k = d_k_prime * f`)

## Prefill 단계


prefill 단계는 입력 시퀀스 전체를 병렬로 처리해 첫 토큰을 출력하면서, 계산한 Key/Value 쌍을 KV cache 로 저장한다.
아래 하위 절들이 각 단계를 순서대로 설명한다.

### 1. 임베딩 조회

임베딩 조회는 입력 토큰을 벡터 공간 표현으로 변환한다.

- **입력**
  - `input: shape![B, s_in]`
  - 입력 텍스트의 토큰 인덱스 (각 토큰이 어떤 어휘 항목에 대응하는지)
- **가중치**
  - `w_emb: shape![V, D]`
  - 각 어휘 항목에 대한 사전 학습된 임베딩 값 테이블
- **출력**
  - `x_0: shape![B, s_in, D]`
- **연산**
  - `x_0 = gather(index: input, table: w_emb)`
  - gather: index 텐서에 지정된 인덱스 값으로 테이블에서 값을 읽는 연산.
    - TensorDMA 가 처리한다.

### 2. Transformer 레이어 (L 회 반복)

각 transformer 레이어는 어텐션과 피드포워드 연산을 순차로 적용한다.
각 레이어 `l = 1, ..., L` 에 대해 다음을 수행한다:

#### 2.1. 입력 레이어 정규화


입력 레이어 정규화는 어텐션 이전에 활성값을 정규화하며, Vector Engine 이 처리한다.

- **입력**
  - `x_prev: shape![B, s_in, D]` (이전 레이어에서 온 레이어 입력)
- **출력**
  - `x_norm: shape![B, s_in, D]`
- **연산**
  - RMSNorm 을 적용한다
  - `x_norm = RMSNorm(x_prev)`
  - RMSNorm: Root Mean Square Layer Normalization
    - Vector Engine 이 처리한다.


#### 2.2. Multi-Head Grouped Query Attention (GQA)

Grouped Query Attention (GQA) 은 여러 쿼리 헤드가 key/value 헤드를 공유하게 해 KV cache 크기를 줄이고 메모리 효율을 높인다.

##### 2.2.1. QKV 프로젝션

QKV 프로젝션은 Contraction Engine 에서 별개의 einsum 연산 3 개를 실행하며, 각각 `D` 축을 리듀스해 Query, Key, Value 텐서를 만든다.
- **입력**
  - `x_norm: shape![B, s_in, D]`
- **가중치**
  - `w_q: shape![D, h_q, d_k]`
  - `w_k: shape![D, h_kv, d_k]`
  - `w_v: shape![D, h_kv, d_k]`
- **출력**
  - `Q: shape![B, s_in, h_q, d_k]`
  - `K: shape![B, s_in, h_kv, d_k]`
  - `V: shape![B, s_in, h_kv, d_k]`
- **연산**
  - `Q = einsum(x_norm, w_q)`
  - `K = einsum(x_norm, w_k)`
  - `V = einsum(x_norm, w_v)`
  - matmul 은 einsum 에 대응한다: 브로드캐스트한 뒤 원소별 곱, 그 다음 reduce-add.
  - 원소별 곱: Contraction Engine
  - reduce-add 는 범위별로 분해된다:
    - packet 리듀스: Packet Reducer
    - time 리듀스: Time Reducer
    - slice 리듀스: 전역 덧셈 트리
    - split 리듀스: 인터리브 fetch + Vector Engine 이항 연산
    - cluster/chip 리듀스: DMA + 인터리브 fetch + Vector Engine 이항 연산

##### 2.2.2. Rotary Position Embedding (RoPE)

Rotary Position Embedding (RoPE) 은 회전 변환을 통해 Query 와 Key 텐서에 위치 정보를 적용한다.
- **입력**
  - `Q: shape![B, s_in, h_q, d_k]`
  - `K: shape![B, s_in, h_kv, d_k]`
  - `d_k = d_k_prime * f`
    - `d_k` 축을 분할해 TCP 친화적인 방식으로 RoPE 회전을 적용한다.
- **RoPE 테이블**
  - `w_rope: shape![s_max, d_k_prime, 2, 2]`
  - 시퀀스 위치와 헤드 위치를 기준으로 미리 계산한 cos/sin 값 테이블.
  - RoPE 연산은 `d_k` 값들 중 연속한 쌍을 묶어 cos/sin 으로 회전 변환을 적용한다.
  - TCP 친화적인 실행을 위해 cos/sin 회전 변환을 나타내는 2 × 2 행렬을 저장한다.
- **위치**
  - `position: shape![s_in]`
  - `position(i) = i`
- **출력**
  - `Q_rope: shape![B, h_q, s_in, d_k]`
  - `K_rope: shape![B, h_kv, s_in, d_k]`
- **연산**
  - **RoPE 테이블 조회**
    - `t_rope: shape![s_in, d_k_prime, 2, 2] = gather(index: position, table: w_rope)`
  - **RoPE 적용**
    - 회전 행렬 값이 준비되면 RoPE 계산은 einsum 연산으로 귀결된다.
    - **Reshape (noop)**
      - `Q: shape![B, s_in, h_q, d_k] == shape![B, s_in, h_q, d_k_prime, f]`
      - `K: shape![B, s_in, h_kv, d_k] == shape![B, s_in, h_kv, d_k_prime, f]`
      - `t_rope: shape![s_in, d_k_prime, 2, 2] == shape![s_in, d_k_prime, f, 2]`
    - **einsum**
      - `Q_rope = einsum(Q, t_rope)`
        - `(shape![B, s_in, h_q, d_k_prime, f], shape![s_in, d_k_prime, f, 2]) -> shape![B, h_q, s_in, d_k_prime, 2] == shape![B, h_q, s_in, d_k]`
      - `K_rope = einsum(K, t_rope)`
        - `(shape![B, s_in, h_kv, d_k_prime, f], shape![s_in, d_k_prime, f, 2]) -> shape![B, h_kv, s_in, d_k_prime, 2] == shape![B, h_kv, s_in, d_k]`

##### 2.2.3. KV Cache 에 저장

KV cache 는 decode 단계에서 재사용하려고 현재 레이어의 Key 와 Value 를 저장해 중복 계산을 피한다.

- **입력**
  - `K_rope: shape![B, h_kv, s_in, d_k]`
  - `V: shape![B, s_in, h_kv, d_k]`
- **KV Cache** (레이어 `l` 용)
  - `kv_cache_l_K: shape![B, h_kv, s_in, d_k]`
  - `kv_cache_l_V: shape![B, h_kv, s_in, d_k]`
- **연산**
  - `kv_cache_l_K = K_rope`
  - `kv_cache_l_V = V`
  - 캐시 저장: einsum 계산 결과를 DM 에서 HBM 으로 저장하며, TensorDMA 가 처리한다.

##### 2.2.4. Grouped Query Attention 계산

Grouped Query Attention 은 각 key/value 헤드를 여러 쿼리 헤드가 공유하게 하며, 위 모델 매개변수에 정의한 대로 쿼리 헤드 `G = 8` 개가 KV 헤드 하나를 공유한다.

**2.2.4.1. 어텐션 스코어 계산**

어텐션 스코어는 내적 유사도로 쿼리 위치와 키 위치 사이의 관련성을 측정한다.

- **입력**
  - `Q_rope: shape![B, h_q, s_in, d_k]`
  - `K_rope: shape![B, h_kv, s_in, d_k]`
- **출력**
  - `scores: shape![B, h_q, s_in, s_in]`
- **연산**
  - `scores = (Q_rope @ K_rope.T) / sqrt(d_k)`
  - **Reshape (noop)**
    - 내적 연산은 einsum 으로 표현할 수 있다.
      einsum 연산의 의미를 정확히 나타내려면 출력 shape 관점에서 각 텐서의 shape 축을 정밀하게 구분해야 한다.
    - `Q_rope: shape![B, h_q, s_in, d_k] == shape![B, G, h_kv, s_in_q, d_k]`
    - `K_rope: shape![B, h_kv, s_in, d_k] == shape![B, h_kv, s_in_k, d_k]`
  - **einsum**
    - `scores_before_normalize = einsum(Q_rope, K_rope)`
    - `(shape![B, G, h_kv, s_in_q, d_k], shape![B, h_kv, s_in_k, d_k]) -> shape![B, G, h_kv, s_in_q, s_in_k] == shape![B, h_q, s_in, s_in]`
    - 이 einsum 표현식은 `G` 가 `K_rope` 로부터 브로드캐스트되고 `d_k` 가 리듀스되었음을 보여준다.
  - **정규화**
    - `scores = scores_before_normalize / sqrt(d_k)`
    - `sqrt(d_k)` 로 나누는 것은 `1/sqrt(d_k)` 를 곱하는 것으로 계산할 수 있다. `1/sqrt(d_k)` 값은 미리 계산해 두고, Vector Engine 이 단순한 상수 곱을 수행한다.

**2.2.4.2. 코잘 마스크 적용**

코잘 마스킹은 토큰이 미래 위치를 참조하지 못하게 막는다.
prefill 단계에서는 `s_in` 개 토큰을 병렬로 처리하지만, `i` 번째 토큰은 위치 `i` 이후의 토큰을 참조해서는 안 된다.

- **입력**
  - `scores: shape![B, h_q, s_in, s_in]`
  - `attention_mask: shape![s_in, s_in]`
  - `attention_mask(i, j) = true if j <= i, false if j > i`
- **출력**
  - `scores_masked: shape![B, h_q, s_in, s_in]`
- **연산**
  - `scores_masked(b, h, i, j) = scores(b, h, i, j) if j <= i, -inf if j > i`
  - Vector Engine 에서는 `attention_mask` 텐서를 branch log 에 쓴 뒤 분기 연산으로 처리한다.

**2.2.4.3. Softmax 적용**

Softmax 는 어텐션 스코어를 키 위치에 대한 확률 분포로 정규화한다.

- **입력**
  - `scores_masked: shape![B, h_q, s_in, s_in]`
- **출력**
  - `attn_weights: shape![B, h_q, s_in, s_in]`
- **연산**
  - `attn_weights = softmax(scores_masked)`
  - Softmax 는 각 쿼리가 값을 결합할 때 각 토큰을 참조할 비율을 계산한다.
  - 두 `s_in` 차원 중 키에 대응하는 축을 리듀스한다.
  - `softmax(x)_i = exp(x_i) / sum_j(exp(x_j))`
  - Vector Engine 이 처리한다

**2.2.4.4. 가중합 (어텐션 출력)**

가중합은 어텐션 가중치에 따라 Value 벡터를 결합해 어텐션 출력을 계산한다.

- **입력**
  - `attn_weights: shape![B, h_q, s_in, s_in]`
  - `V: shape![B, s_in, h_kv, d_k]`
- **출력**
  - `attn_output: shape![B, h_q, s_in, d_k]`
- **연산**
  - **Reshape (noop)**
    - `attn_weights: shape![B, h_q, s_in, s_in] == shape![B, G, h_kv, s_in_q, s_in_kv]`
    - `V: shape![B, s_in, h_kv, d_k] == shape![B, h_kv, s_in_kv, d_k]`
  - **einsum**
    - `attn_output = einsum(attn_weights, V)`
    - `(shape![B, G, h_kv, s_in_q, s_in_kv], shape![B, h_kv, s_in_kv, d_k]) -> shape![B, G, h_kv, s_in_q, d_k] == shape![B, h_q, s_in, d_k]`
    - 이 einsum 표현식은 `G` 가 `V` 로부터 브로드캐스트되고 `s_in_kv` 가 리듀스되었음을 보여준다.

##### 2.2.5. 출력 프로젝션

출력 프로젝션은 멀티헤드 어텐션 결과를 하나의 은닉 상태 벡터로 결합한다.

- **입력**
  - `attn_output: shape![B, h_q, s_in, d_k]`
- **가중치**
  - `w_o: shape![h_q, d_k, D]`
- **출력**
  - `attn_out: shape![B, s_in, D]`
- **연산**
  - `attn_out = einsum(attn_output, w_o)`
  - `(shape![B, h_q, s_in, d_k], shape![h_q, d_k, D]) -> shape![B, s_in, D]`

##### 2.2.6. 잔차 연결

잔차 연결은 어텐션 출력을 레이어 입력에 더해 학습 중 그래디언트 흐름을 개선한다.

- **입력**
  - `x_prev: shape![B, s_in, D]` (레이어 입력)
  - `attn_out: shape![B, s_in, D]` (어텐션 출력)
- **출력**
  - `x_attn: shape![B, s_in, D]`
- **연산**
  - `x_attn = x_prev + attn_out`
  - 원소별 덧셈: Vector Engine 이 처리한다

#### 2.3. Feed-Forward Network (FFN)

Feed-Forward Network 는 어텐션 이후 각 토큰에 독립적으로 비선형 변환을 적용한다.

##### 2.3.1. 어텐션 후 레이어 정규화

어텐션 후 정규화는 FFN 계산 이전에 활성값을 안정화한다.

- **입력**
  - `x_attn: shape![B, s_in, D]`
- **출력**
  - `x_ffn_norm: shape![B, s_in, D]`
- **연산**
  - `x_ffn_norm = RMSNorm(x_attn)`
  - RMSNorm: Vector Engine 이 처리한다

##### 2.3.2. SwiGLU FFN

SwiGLU (Swish-Gated Linear Unit) 는 Llama 3 의 활성 함수로, 게이팅과 Swish 비선형성을 결합한다.

- **입력**
  - `x_ffn_norm: shape![B, s_in, D]`
- **가중치**
  - `w_gate: shape![D, F]` (게이트 프로젝션)
  - `w_up: shape![D, F]` (업 프로젝션)
  - `w_down: shape![F, D]` (다운 프로젝션)
- **출력**
  - `ffn_out: shape![B, s_in, D]`
- **연산**
  - **게이트 프로젝션**:
    - `gate = einsum(x_ffn_norm, w_gate)`
    - `(shape![B, s_in, D], shape![D, F]) -> shape![B, s_in, F]`
  - **업 프로젝션**:
    - `up = einsum(x_ffn_norm, w_up)`
    - `(shape![B, s_in, D], shape![D, F]) -> shape![B, s_in, F]`
  - **SwiGLU 활성화**:
    - `activated = SiLU(gate) * up`
    - SiLU (Swish): `SiLU(x) = x * sigmoid(x)`
    - `*`: 원소별 곱
    - Vector Engine 이 처리한다
  - **다운 프로젝션**:
    - `ffn_out = einsum(activated, w_down)`
    - `(shape![B, s_in, F], shape![F, D]) -> shape![B, s_in, D]`

##### 2.3.3. 잔차 연결

FFN 잔차 연결은 FFN 출력을 어텐션 후 출력에 더한다.

- **입력**
  - `x_attn: shape![B, s_in, D]` (어텐션 후 출력)
  - `ffn_out: shape![B, s_in, D]` (FFN 출력)
- **출력**
  - `x_l: shape![B, s_in, D]` (레이어 `l` 의 최종 출력)
- **연산**
  - `x_l = x_attn + ffn_out`
  - 원소별 덧셈: Vector Engine 이 처리한다

### 3. 최종 레이어 정규화

최종 레이어 정규화는 transformer 레이어 80 개를 모두 통과한 뒤에 적용한다.

- **입력**
  - `x_L: shape![B, s_in, D]` (마지막 레이어 출력)
- **출력**
  - `x_final: shape![B, s_in, D]`
- **연산**
  - `x_final = RMSNorm(x_L)`
  - RMSNorm: Vector Engine 이 처리한다

### 4. Language Model Head (출력 레이어)

language model head 는 마지막 토큰 위치의 은닉 상태를 다음 토큰 예측을 위한 어휘 로짓으로 변환한다.

- **입력**
  - `x_final: shape![B, s_in, D]`
- **가중치**
  - `w_lm_head: shape![D, V]`
  - 보통 `w_lm_head = w_emb.T` 이다 (weight tying)
- **출력**
  - `logits: shape![B, V]`
- **연산**
  - **Slice**: prefill 단계에서는 마지막 토큰만 사용한다
    - `x_last: shape![B, D] = x_final[:, -1, :]`
    - 다음 토큰을 예측하려고 마지막 토큰의 은닉 상태만 추출한다
    - 슬라이스 연산은 shape 에 따라 단순한 view 연산으로 처리하거나, parallel copy 로 데이터 일부를 직접 읽어 옮긴다.
  - **einsum**: 어휘에 대한 로짓 계산
    - `logits = einsum(x_last, w_lm_head)`
    - `(shape![B, D], shape![D, V]) -> shape![B, V]`

<a id="5-sampling"></a>
### 5. 샘플링

샘플링은 로짓 값을 확률 분포로 변환하고 다음 토큰을 선택한다.
이 과정은 TCP 가 아니라 Host 에서 일어난다.

- **입력**
  - `logits: shape![B, V]`
  - `temperature: scalar` (샘플링 온도 매개변수, 보통 0.7~1.0)
- **출력**
  - `next_token: shape![B]` (배치별 다음 토큰 인덱스)
- **연산**
  - **온도 스케일링**:
    - `logits_scaled = logits / temperature`
    - 온도가 높을수록 토큰 선택이 다양해지고, 낮을수록 결정론적으로 선택된다
    - `1/temperature` 값은 미리 계산해 두고 Vector Engine 에서 상수 곱으로 처리한다
  - **Softmax**:
    - `probs: shape![B, V] = softmax(logits_scaled)`
    - `softmax(x)_i = exp(x_i) / sum_j(exp(x_j))`
    - 어휘 축(`V`)에 대해 softmax 를 적용한다
  - **토큰 샘플링**:
    - 확률 분포 `probs` 에서 다음 토큰 인덱스를 샘플링한다
    - 샘플링 전략:
      - Greedy: `next_token = argmax_i(probs_i)`
      - Top-k 샘플링: 확률 상위 k 개 토큰에서만 샘플링한다
      - Top-p (nucleus) 샘플링: 누적 확률이 p 를 넘는 가장 작은 토큰 집합에서 샘플링한다


## Decode 단계

decode 단계는 prefill 과 같은 연산 순서(임베딩, transformer 레이어, LM head, 샘플링)를 재사용하지만, 한 번에 토큰 하나만 처리하고 KV 쌍을 다시 계산하는 대신 캐시된 것을 재사용한다.
EOS 토큰이 나오거나 최대 길이에 도달할 때까지 자기회귀적으로 이어진다.

decode 를 prefill 과 구분 짓는 특징은 세 가지다:

- **단일 토큰 입력**: `s_in = 1` (가장 최근 출력 토큰만 쿼리로 사용한다)
- **KV cache 재사용**: 이전에 계산한 Key 와 Value 텐서를 다시 계산하지 않고 재사용한다
- **자기회귀 생성**: 각 토큰 예측은 캐시를 통해 이전 토큰 전부를 참조한다

각 디코딩 스텝 `s = s_prefill + 1, ..., s_max` 에 대해:

### 1. 임베딩 조회

임베딩 조회는 앞서 생성한 토큰을 벡터 표현으로 변환한다.

- **입력**
  - `input: shape![B, 1]`
  - 이전 스텝에서 샘플링한 토큰 인덱스
- **가중치**
  - `w_emb: shape![V, D]`
- **출력**
  - `x_0: shape![B, 1, D]`
- **연산**
  - `x_0 = gather(index: input, table: w_emb)`
  - TensorDMA 가 처리한다

### 2. Transformer 레이어 (L 회 반복)

각 transformer 레이어는 캐시된 KV 쌍을 재사용하면서 토큰 하나를 어텐션과 FFN 으로 처리한다.
각 레이어 `l = 1, ..., L` 에 대해 다음을 수행한다:

#### 2.1. 입력 레이어 정규화

입력 레이어 정규화는 어텐션 계산을 위해 토큰을 준비한다.

- **입력**
  - `x_prev: shape![B, 1, D]` (이전 레이어에서 온 레이어 입력)
- **출력**
  - `x_norm: shape![B, 1, D]`
- **연산**
  - `x_norm = RMSNorm(x_prev)`
  - Vector Engine 이 처리한다

#### 2.2. Multi-Head Grouped Query Attention (GQA)

decode 단계의 어텐션은 현재 토큰(쿼리)과 캐시된 모든 토큰(키/값) 사이의 어텐션을 계산한다.

##### 2.2.1. QKV 프로젝션

QKV 프로젝션은 현재 토큰에 대해서만 Query, Key, Value 를 계산한다.

- **입력**
  - `x_norm: shape![B, 1, D]`
- **가중치**
  - `w_q: shape![D, h_q, d_k]`
  - `w_k: shape![D, h_kv, d_k]`
  - `w_v: shape![D, h_kv, d_k]`
- **출력**
  - `Q: shape![B, 1, h_q, d_k]`
  - `K_new: shape![B, 1, h_kv, d_k]`
  - `V_new: shape![B, 1, h_kv, d_k]`
- **연산**
  - `Q = einsum(x_norm, w_q)`
  - `K_new = einsum(x_norm, w_k)`
  - `V_new = einsum(x_norm, w_v)`
  - `(shape![B, 1, D], shape![D, h_q/kv, d_k]) -> shape![B, 1, h_q/kv, d_k]`

##### 2.2.2. Rotary Position Embedding (RoPE)

RoPE 는 현재 시퀀스 위치에 대응하는 위치 인코딩을 적용한다.

- **입력**
  - `Q: shape![B, 1, h_q, d_k]`
  - `K_new: shape![B, 1, h_kv, d_k]`
- **RoPE 테이블**
  - `w_rope: shape![s_max, d_k_prime, 2, 2]`
- **위치**
  - `position: shape![1]`
  - `position(0) = s` (지금까지 처리한 전체 시퀀스 길이)
- **출력**
  - `Q_rope: shape![B, h_q, 1, d_k]`
  - `K_rope: shape![B, h_kv, 1, d_k]`
- **연산**
  - **RoPE 테이블 조회**
    - `t_rope: shape![1, d_k_prime, 2, 2] = gather(index: position, table: w_rope)`
  - **RoPE 적용**
    - **Reshape (noop)**
      - `Q: shape![B, 1, h_q, d_k] == shape![B, 1, h_q, d_k_prime, f]`
      - `K_new: shape![B, 1, h_kv, d_k] == shape![B, 1, h_kv, d_k_prime, f]`
      - `t_rope: shape![1, d_k_prime, 2, 2] == shape![1, d_k_prime, f, 2]`
    - **einsum**
      - `Q_rope = einsum(Q, t_rope)`
        - `(shape![B, 1, h_q, d_k_prime, f], shape![1, d_k_prime, f, 2]) -> shape![B, h_q, 1, d_k_prime, 2] == shape![B, h_q, 1, d_k]`
      - `K_rope = einsum(K_new, t_rope)`
        - `(shape![B, 1, h_kv, d_k_prime, f], shape![1, d_k_prime, f, 2]) -> shape![B, h_kv, 1, d_k_prime, 2] == shape![B, h_kv, 1, d_k]`

##### 2.2.3. KV Cache 갱신

KV cache 갱신은 이후 토큰 생성을 위해 새 Key 와 Value 를 기존 캐시에 덧붙인다.

- **입력**
  - `kv_cache_l_K: shape![B, h_kv, s-1, d_k]` (기존 캐시)
  - `kv_cache_l_V: shape![B, h_kv, s-1, d_k]` (기존 캐시)
  - `K_rope: shape![B, h_kv, 1, d_k]` (새 Key)
  - `V_new: shape![B, 1, h_kv, d_k]` (새 Value)


- **출력**
  - `kv_cache_l_K: shape![B, h_kv, s, d_k]` (갱신된 캐시)
  - `kv_cache_l_V: shape![B, h_kv, s, d_k]` (갱신된 캐시)
- **연산**
  - **Concatenate**: 기존 캐시에 새 K, V 를 추가한다
    - `kv_cache_l_K[s-1] = K_rope`
    - `kv_cache_l_V[s-1] = V_new`
    - concat 축을 어떻게 배치하느냐에 따라 처리가 달라진다.
      슬라이스 사이의 데이터 이동은 RoutingEngine/parallel copy 를 쓰고, 원소 사이의 데이터 이동은 parallel copy 를 쓴다.
    - DMA 로 HBM 에서 concat 하는 것도 가능하다.

##### 2.2.4. Grouped Query Attention 계산

어텐션 계산은 현재 Query 를 KV cache 전체에 대해 사용해, 과거 토큰 중 어느 것이 현재 출력에 기여하는지를 결정한다.

**2.2.4.1. 어텐션 스코어 계산**

어텐션 스코어는 현재 Query 와 캐시된 모든 Key 사이의 유사도를 측정한다.

- **입력**
  - `Q_rope: shape![B, h_q, 1, d_k]`
  - `kv_cache_l_K: shape![B, h_kv, s, d_k]`
- **출력**
  - `scores: shape![B, h_q, 1, s]`
- **연산**
  - `scores = (Q_rope @ kv_cache_l_K.T) / sqrt(d_k)`
  - **Reshape (noop)**
    - `Q_rope: shape![B, h_q, 1, d_k] == shape![B, G, h_kv, 1, d_k]`
    - `kv_cache_l_K: shape![B, h_kv, s, d_k] == shape![B, h_kv, s, d_k]`
  - **einsum**
    - `scores_before_normalize = einsum(Q_rope, kv_cache_l_K)`
    - `(shape![B, G, h_kv, 1, d_k], shape![B, h_kv, s, d_k]) -> shape![B, G, h_kv, 1, s] == shape![B, h_q, 1, s]`
    - 이 einsum 표현식은 `G` 가 `kv_cache_l_K` 로부터 브로드캐스트되고 `d_k` 가 리듀스되었음을 보여준다.
  - **정규화**
    - `scores = scores_before_normalize / sqrt(d_k)`
    - Vector Engine 에서 상수 곱으로 처리한다

**2.2.4.2. Softmax 적용**

Softmax 는 스코어를 어텐션 가중치로 변환한다.
decode 에서는 현재 토큰이 과거 토큰만 참조하므로 코잘 마스크가 필요 없다.

- **입력**
  - `scores: shape![B, h_q, 1, s]`
- **출력**
  - `attn_weights: shape![B, h_q, 1, s]`
- **연산**
  - `attn_weights = softmax(scores)`
  - softmax 는 마지막 축(`s`, 즉 과거 토큰 전부)에 대해 적용한다
  - `softmax(x)_i = exp(x_i) / sum_j(exp(x_j))`
  - Vector Engine 이 처리한다

**2.2.4.3. 가중합 (어텐션 출력)**

가중합은 어텐션 가중치에 따라 캐시된 Value 를 결합해 어텐션 출력을 만든다.

- **입력**
  - `attn_weights: shape![B, h_q, 1, s]`
  - `kv_cache_l_V: shape![B, h_kv, s, d_k]`
- **출력**
  - `attn_output: shape![B, h_q, 1, d_k]`
- **연산**
  - **Reshape (noop)**
    - `attn_weights: shape![B, h_q, 1, s] == shape![B, G, h_kv, 1, s]`
    - `kv_cache_l_V: shape![B, h_kv, s, d_k] == shape![B, h_kv, s, d_k]`
  - **einsum**
    - `attn_output = einsum(attn_weights, kv_cache_l_V)`
    - `(shape![B, G, h_kv, 1, s], shape![B, h_kv, s, d_k]) -> shape![B, G, h_kv, 1, d_k] == shape![B, h_q, 1, d_k]`
    - 이 einsum 표현식은 `G` 가 `kv_cache_l_V` 로부터 브로드캐스트되고 `s` 가 리듀스되었음을 보여준다.

##### 2.2.5. 출력 프로젝션

출력 프로젝션은 어텐션 결과를 다시 은닉 차원으로 변환한다.

- **입력**
  - `attn_output: shape![B, h_q, 1, d_k]`
- **가중치**
  - `w_o: shape![h_q, d_k, D]`
- **출력**
  - `attn_out: shape![B, 1, D]`
- **연산**
  - `attn_out = einsum(attn_output, w_o)`
  - `(shape![B, h_q, 1, d_k], shape![h_q, d_k, D]) -> shape![B, 1, D]`

##### 2.2.6. 잔차 연결

잔차 연결은 어텐션 출력과 레이어 입력을 결합한다.

- **입력**
  - `x_prev: shape![B, 1, D]` (레이어 입력)
  - `attn_out: shape![B, 1, D]` (어텐션 출력)
- **출력**
  - `x_attn: shape![B, 1, D]`
- **연산**
  - `x_attn = x_prev + attn_out`
  - 원소별 덧셈: Vector Engine 이 처리한다

#### 2.3. Feed-Forward Network (FFN)

decode 단계의 FFN 은 prefill 과 동일하지만 토큰 하나만 처리한다 (시퀀스 길이 = 1).

##### 2.3.1. 어텐션 후 레이어 정규화

어텐션 후 정규화는 FFN 처리를 위해 토큰을 준비한다.

- **입력**
  - `x_attn: shape![B, 1, D]`
- **출력**
  - `x_ffn_norm: shape![B, 1, D]`
- **연산**
  - `x_ffn_norm = RMSNorm(x_attn)`
  - Vector Engine 이 처리한다

##### 2.3.2. SwiGLU FFN

SwiGLU 는 프로젝션 3 개로 게이트 활성 함수를 적용한다.

- **입력**
  - `x_ffn_norm: shape![B, 1, D]`
- **가중치**
  - `w_gate: shape![D, F]`
  - `w_up: shape![D, F]`
  - `w_down: shape![F, D]`
- **출력**
  - `ffn_out: shape![B, 1, D]`
- **연산**
  - **게이트 프로젝션**:
    - `gate = einsum(x_ffn_norm, w_gate)`
    - `(shape![B, 1, D], shape![D, F]) -> shape![B, 1, F]`
  - **업 프로젝션**:
    - `up = einsum(x_ffn_norm, w_up)`
    - `(shape![B, 1, D], shape![D, F]) -> shape![B, 1, F]`
  - **SwiGLU 활성화**:
    - `activated = SiLU(gate) * up`
    - Vector Engine 이 처리한다
  - **다운 프로젝션**:
    - `ffn_out = einsum(activated, w_down)`
    - `(shape![B, 1, F], shape![F, D]) -> shape![B, 1, D]`

##### 2.3.3. 잔차 연결

FFN 잔차 연결은 레이어의 최종 출력을 만든다.

- **입력**
  - `x_attn: shape![B, 1, D]`
  - `ffn_out: shape![B, 1, D]`
- **출력**
  - `x_l: shape![B, 1, D]`
- **연산**
  - `x_l = x_attn + ffn_out`
  - 원소별 덧셈: Vector Engine 이 처리한다

### 3. 최종 레이어 정규화

최종 레이어 정규화는 language model head 를 위해 출력을 준비한다.

- **입력**
  - `x_L: shape![B, 1, D]`
- **출력**
  - `x_final: shape![B, 1, D]`
- **연산**
  - `x_final = RMSNorm(x_L)`
  - Vector Engine 이 처리한다

### 4. Language Model Head

language model head 는 은닉 상태를 어휘 로짓으로 프로젝션한다.
prefill 과 달리 토큰이 하나뿐이므로 슬라이스 연산이 필요 없다.

- **입력**
  - `x_final: shape![B, 1, D]`
- **가중치**
  - `w_lm_head: shape![D, V]`
- **출력**
  - `logits: shape![B, V]`
- **연산**
  - **Reshape/Squeeze**: 시퀀스 차원을 제거한다
    - `x_squeezed: shape![B, D] = squeeze(x_final)`
  - **einsum**: 어휘에 대한 로짓 계산
    - `logits = einsum(x_squeezed, w_lm_head)`
    - `(shape![B, D], shape![D, V]) -> shape![B, V]`

### 5. 샘플링

샘플링은 [Prefill 샘플링](#5-sampling) 과 동일하다: 온도 스케일링, softmax, 토큰 선택을 Host 에서 수행한다.

### 6. 종료 조건

다음 세 조건 중 하나라도 충족되면 생성이 종료된다:

- **EOS 토큰 생성**: 샘플링된 토큰이 End-of-Sequence 토큰인 경우
- **최대 길이 도달**: `s >= s_max`
- **사용자 정의 종료 조건**: 특정 패턴이나 조건이 충족된 경우

생성을 계속한다면 `s <- s + 1` 로 갱신하고 다음 디코딩 스텝으로 돌아간다.

## Prefill 단계와 Decode 단계 비교

prefill 은 compute-bound (입력 토큰 전체에 대한 대규모 병렬 연산)이고 decode 는 memory-bound (스텝마다 토큰 하나에 대한 KV cache 접근)이다.
아래 표가 주요 차이를 정리한다:

| 특성 | Prefill 단계 | Decode 단계 |
|------|---------------|--------------|
| 입력 시퀀스 길이 | `s_in` (가변) | 1 (고정) |
| 병렬 처리 | `s_in` 개 토큰을 병렬 처리 | 토큰 1 개만 처리 |
| KV Cache | 생성 및 저장 | 읽기 및 갱신 |
| 어텐션 계산 | 코잘 마스크 필요 | 코잘 마스크 불필요 |
| 어텐션 shape | `shape![B, h_q, s_in, s_in]` | `shape![B, h_q, 1, s]` |
| 연산 특성 | Compute-bound (대규모 연산) | Memory-bound (KV cache 접근) |
| 처리량 | 높음 (병렬 처리) | 낮음 (순차 처리) |
| 지연 | 상대적으로 높음 | 낮음 (토큰당) |
