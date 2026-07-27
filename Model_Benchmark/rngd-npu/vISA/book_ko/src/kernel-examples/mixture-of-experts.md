# Mixture of Experts


Mixture of Experts(MoE)는 각 토큰을 전체 `E` 개 Expert 모두가 아니라 그중 `K` 개에만 라우팅하여 모델 용량을 키운다(`K` 와 `E` 는 아래에서 정의하는 모델 매개변수다).
이 희소 활성화는 추론 비용을 감당할 수준으로 유지하면서 많은 매개변수를 쓸 수 있게 한다.
이 예제는 TCP 하드웨어에서 MoE 를 구현하는 방법을 보이며, 두 가지 핵심 과제에 집중한다. 제어 흐름 기반 TopK 라우팅을 브랜치 없는 행렬 연산으로 대체하는 것, 그리고 희소한 Expert 연산을 블록 단위로 실행하는 것이다.

## 배경: 기본 FFN

FFN 과 MoE 에 익숙한 독자는 [TCP 에서의 MoE 구현](#moe-implementation-on-tcp) 으로 건너뛰어도 된다.

기본 FFN 은 두 개의 선형 투영(up 과 down)으로 이루어지며, 둘을 합치면 `T × D → T × F → T × D` 로 매핑한다.
다음은 MoE 가 대체하는 기준선으로서 이 단순화된 FFN(gate projection 없이 up/down projection 만)을 설명한다:

- **입력**
  - `x_ffn_norm: T x D`
- **가중치**
  - `W_up: D x F` (up projection)
  - `W_down: F x D` (down projection)
- **출력**
  - `ffn_out: T x D`
- **연산**
  - **Up projection**:
    - `up = einsum(x_ffn_norm, W_up)`
    - `(T x D), (D x F) -> T x F`
  - **Down projection**:
    - `ffn_out = einsum(up, W_down)`
    - `(T x F), (F x D) -> T x D`

## MoE 구조

MoE 는 단일 FFN 을 Expert 라 부르는 `E` 개의 독립적인 FFN 으로 대체한다.
각 Expert 는 자신만의 가중치를 가진다:
- `W_up[0], W_up[1], ..., W_up[E-1]`
- `W_down[0], W_down[1], ..., W_down[E-1]`

모든 Expert 를 계산하면 연산량이 `E` 배로 늘어난다.
이를 피하기 위해 MoE 는 라우터로 토큰마다 가장 적합한 `Top-K` 개 Expert 만 선택하여 희소 연산을 가능하게 한다.

## 모델 매개변수

다음 인자들이 MoE 레이어를 정의한다:
- `T`: 토큰 수
  - prefill: `T = B * S_in`
  - decode: `T = B`
- `D`: 히든 차원
- `F`: ffn up projection 결과의 중간 차원
- `E`: 전체 Expert 수 (보통 128)
- `K`: ffn 을 적용할 Expert 수
  - `llama4`: 1, `gpt-oss`: 4, `qwen3`: 8

## MoE 처리 단계

MoE 처리는 네 단계로 이루어진다. gating(Expert 별 점수 계산), Top-K 선택(토큰마다 가장 좋은 `K` 개 Expert 고르기), 희소 Expert 연산(선택된 Expert 적용), 그리고 combining(Expert 출력을 라우팅 가중치로 병합)이다.

### 1. Gating (라우터)

라우터는 모든 토큰에 대해 Expert 마다 점수를 계산하여, 각 토큰을 어느 Expert 가 처리할지 정한다:
- **입력**
  - `x_norm: T x D`
- **가중치**
  - `W_router: D x E` (Gating 네트워크 가중치)
- **출력**
  - `scores: T x E`
- **연산**
  - `scores = einsum(x_norm, W_router)`
  - `(T x D), (D x E) -> T x E`
  - 토큰마다 `E` 개 Expert 에 대한 점수(Logit)를 계산한다

### 2. `Top-K` 선택

이 단계는 1단계의 라우터 점수를 바탕으로 `Top-K` Expert 를 선택하고, 선택된 Expert 마다 가중치를 계산한다:

- **입력**
  - `scores: T x E`
- **출력**
  - `topk_indices: T x K` (토큰별 선택된 Expert ID)
  - `routing_weights: T x K` (토큰별 선택된 Expert 의 가중치)
- **연산**
  - **`Top-K` 선택**:
    - `raw_weights, topk_indices = topk(scores, K)`
    - 토큰마다 점수가 가장 높은 `K` 개 Expert 인덱스와 점수를 뽑는다
  - **Softmax 정규화**:
    - `routing_weights = softmax(raw_weights)`
    - 선택된 `K` 개 점수를 확률 값으로 변환한다 (토큰마다 합이 1)
    - `softmax(x)[i] = exp(x[i]) / sum(exp(x[j]) for j in 0..K)`

각 토큰 `t` 의 출력은 다음으로 구성된다:
- `topk_indices[t, :]`: `K` 개 Expert ID (`0 <= e < E`)
- `routing_weights[t, :]`: 해당 Expert 들의 가중치 (합은 1)

### 3. 희소 Expert 연산

선택된 Expert 만 연산을 수행하므로 이 단계는 희소하다.
총 `T * K` 번의 Expert 호출이 일어나지만, 각 Expert 는 자신을 선택한 토큰에 대해서만 계산한다.

각 토큰 `t in [0, T-1]` 과 선택된 Expert `k in [0, K-1]` 에 대해:

- **선택된 Expert ID**: `e = topk_indices[t, k]`
- **입력**
  - `x_norm[t]: D` (토큰 `t` 의 입력)
- **가중치** (Expert `e` 의 가중치)
  - `W_up[e]: D x F`
  - `W_down[e]: F x D`
- **출력**
  - `y[t, k]: D` (토큰 `t` 의 `k` 번째 Expert 출력)
- **연산**
  - **Up projection**:
    - `up = einsum(x_norm[t], W_up[e])`
    - `D, (D x F) -> F`
  - **Down projection**:
    - `y[t, k] = einsum(up, W_down[e])`
    - `F, (F x D) -> D`

모든 `(t, k)` 쌍의 결과는 `y_experts: T x K x D` 로 모인다.

### 4. 가중 합 (Combine)

마지막 단계는 앞서 계산한 라우팅 가중치로 `K` 개 Expert 출력을 결합한다:

- **입력**
  - `y_experts: T x K x D`
  - `routing_weights: T x K` (각 Expert 의 가중치)
- **출력**
  - `ffn_out: T x D`
- **연산**
  - `ffn_out = einsum(y_experts, routing_weights)`

그 결과 각 토큰은 자신이 선택한 `K` 개 Expert 출력의 가중 평균을 받는다.

<a id="moe-implementation-on-tcp"></a>
## TCP 에서의 MoE 구현


MoE 의 TCP 구현은 하드웨어에 특화된 두 기법을 쓴다. Vector Engine 의 비트 조작과 filter 연산을 이용한 브랜치 없는 TopK, 그리고 정적 모양 scatter/gather 패턴을 이용한 블록 단위 실행이다.

### 1. 개요와 설계 철학

#### 1.1. 논리적 실행과 물리적 실행 잇기

TCP 에서 MoE 를 구현할 때의 근본적인 과제는 두 가지다:

- **과제 1**: 제어 흐름과 병렬 구조의 충돌
  - 문제: 일반적인 `Top-K` 알고리즘은 데이터 값에 따라 실행 경로가 달라지는 분기문을 쓴다.
    이런 분기문은 명령 하나로 수천 개 원소를 처리하는 SIMT 기반 가속기에서 성능 저하를 일으킨다.
  - 해결: 제어 흐름을 완전히 제거하고 행렬 연산과 비트 조작으로 Branchless `Top-K` 기법을 쓰는 것이 필수다.
- **과제 2**: 논리적 Routing 과 물리적 실행의 간극
  - 문제: 논리적으로 MoE 는 각 토큰이 자신에게 맞는 Expert 를 찾는 과정이다(Token-centric).
    그러나 그대로 구현하면 메모리 접근이 불규칙해지고 Expert 당 처리할 토큰 수가 동적으로 바뀌어 TCP 컴파일러 효율이 떨어진다.
  - 해결: Expert 가 주체가 되어 토큰을 모으는 방식(Expert-centric)으로 관점을 바꿔야 한다.

#### 1.2. TCP 구현의 핵심 기법

이 과제들을 다루는 핵심 기법은 두 가지다:

- **Branchless `TopK`**: 행렬 연산만으로 라우팅을 수행하여 모든 제어 흐름을 없앤다
- **블록 단위 실행**: 고정 크기 `Block` 단위로 묶인 데이터로 선택된 Expert 만 처리한다

다음 절에서 각 기법을 자세히 설명한다.

### 2. Branchless `TopK`

Branchless `TopK` 는 제어 흐름 기반 정렬을 순수한 행렬 연산으로 대체한다.
이 방식은 세 단계로 이루어진다. 점수와 인덱스를 합치는 비트 패킹, 순서를 정하는 병렬 랭킹, 그리고 상위 `K` 개 결과를 뽑는 필터링이다.

#### 2.1. 비트 패킹 (점수와 인덱스 합치기)

비트 패킹은 점수와 인덱스를 하나의 값으로 묶어, 정렬 중에 점수 순서가 바뀌어도 Expert ID 가 보존되게 한다.
TCP Vector Engine 은 256 개 Slice 전체에서 고정된 명령 시퀀스를 동시에 실행하므로, 주소나 제어 경로가 런타임 데이터 값에 의존하는 연산은 모두 고정된 행렬 연산 시퀀스로 다시 써야 한다:

- **입력**
  - `scores: T x E`
  - `Index_expert: E`
    - `Index_expert(e) = e where e = 0, 1, 2, ..., E - 1`
- **출력**
  - `Packed_Value: T x E`
    - (score, index) 가 패킹된 텐서.
  - `Packed_Value_cmp: T x E`
    - (score, index) 가 패킹된 텐서로, 정수 비교로 score 크기를 비교할 수 있도록 전처리된 것.
- **연산**
  - **패킹**
    - Expert Score(예: `bf16`)를 상위 비트에, Expert Index(예: `int16`)를 하위 비트에 두어 하나의 32비트 정수(또는 부동소수점)를 만든다.
    - `Packed_Value_unprocessed = (Score << 16) | Index`
    - Vector Engine 에서 처리된다.
  - **비교 트릭**
    - 이 전처리 덕분에 단순한 정수 비교로 score 값의 크기를 비교할 수 있다.
    - Bit Flipping 전처리는 부동소수점 값을 정수 값으로 비교할 때 음수의 크기 관계가 뒤집히는 문제를 해결한다.
      이로써 정수 비교기만으로 정확한 Top-K 선택이 가능하다.
    - ```rust,ignore
      Packed_Value_cmp = if Packed_Value >= 0 {
          Packed_Value
      } else {
          Packed_Value ^ 0x7fff0000
      }
      ```

#### 2.2. 병렬 랭킹 (All-to-All 비교)

병렬 랭킹은 순차 정렬 대신 모든 Expert 의 순서를 동시에 결정한다.
이 방식은 `E x E` 번의 비교가 필요하지만, 제어 흐름 없이 행렬 연산만 쓰므로 TCP 효율은 높게 유지된다:

- **입력**
  - `Packed_Value_cmp: T x E`
    - 비교 트릭이 적용된 32비트 Packed Tensor.
- **출력**
  - `Rank: T x E`
    - 각 Expert 의 순위(0 기반 순위). 점수가 높을수록 0 에 가깝다.
- **연산**
  - **브로드캐스트 & 비교**
    - `Packed_Value_cmp` 를 `E` 축을 따라 복제(Tile)하여 `T x E x E` 모양으로 확장한다.
      모든 Expert 쌍 `(i, j)` 에 대해 크기 관계를 비교한다.
    - `Compare[t, i, j] = 1 if Packed_Value_cmp[t, j] > Packed_Value_cmp[t, i] else 0`
    - 의미: "Expert `j` 의 점수가 Expert `i` 의 점수보다 높은가?"
  - **순위 계산 (ReduceSum)**
    - `E`(비교 대상) 축을 따라 합하여 순위를 계산한다.
    - `Rank[t, i] = sum(Compare[t, i, j] for j in 0..E)`
    - 의미: "나보다 점수가 높은 Expert 의 총 개수"가 내 순위가 된다.

#### 2.3. 필터링 & 언패킹

필터링은 순위를 기준으로 상위 `K` 개 항목을 뽑고, 이어서 언패킹이 패킹된 점수와 인덱스를 분리한다:

- **입력**
  - `Rank: T x E`
  - `Packed_Value: T x E`
    - 참고: 나중에 정확한 Score/Index 를 복원하려면 비교 트릭을 적용하기 전의 원래 Packed Value 를 써야 한다.
- **출력**
  - `TopK_Indices: T x K`
  - `TopK_Scores: T x K`
  - `routing_weights: T x K` (Token 당 선택된 K 개 Expert 의 가중치)
- **연산**
  - **필터링 (`FilterCompaction`)**
    - `Top-K` 조건(`Rank < K`)을 만족하는 원소만 남긴다.
    - `Mask[t, i] = 1 if Rank[t, i] < K else 0`
    - Mask 가 True 인 위치의 `Packed_Value` 만 모아 `T x K` 크기로 압축한다.
    - 결과: `Selected_Packed: T x K`
    - Vector Engine 의 filter 기능을 쓴다.
  - **언패킹**
    - 선택된 32비트 값에서 비트 연산으로 점수와 인덱스를 복원한다.
    - Score 추출: `TopK_Scores = Selected_Packed >> 16` (이후 `bf16` 타입으로 재해석)
    - Index 추출: `TopK_Indices = Selected_Packed & 0xffff`
  - **Softmax 정규화**
    - 뽑아낸 `Top-K` Scores 에 Softmax 를 적용하여 최종 가중치를 계산한다.
      이는 나중에 Combine 단계에서 쓰인다.
    - `routing_weights[t, k] = exp(TopK_Scores[t, k]) / sum(exp(TopK_Scores[t, j]) for j in 0..K)`

### 3. 블록 단위 실행

블록 단위 실행은 TCP 의 정적 모양 제약을 지키면서 `Top-K` 라우팅 결정에 따라 데이터를 물리적으로 재배치한다.

#### 3.1. 문제: 동적 모양과 메모리 폭발

핵심 과제는 Expert 마다 할당되는 토큰 수 `L_e` 가 입력에 따라 동적으로 달라진다는 점이다.
최악의 경우 모든 토큰이 특정 Expert 에 몰리면 `L_e ~ T` 가 된다.

이 과제를 다루는 접근은 두 가지다:

- **단순한 해법**: 모든 Expert 에 최대 크기 `T` 의 버퍼를 할당하면 `E x T x D` 크기의 메모리가 필요하고, 그 대부분은 패딩으로 낭비된다.
- **블록 단위 해법**: 가변 길이 `L_e` 대신 고정 크기 `Block`(`B`) 단위로 데이터를 관리하여 메모리 사용량을 대략 `T x K` 수준으로 최적화한다.

#### 3.2. Grid 크기 계산

같은 Expert 로 가는 토큰들은 `B` 개 토큰짜리 블록으로 묶이며, 덕분에 Expert 하나만 로드한 채 블록 단위 연산을 할 수 있다.
Grid 크기(모든 Expert 에 걸친 전체 블록 수)는 모든 토큰을 처리하는 데 블록이 몇 개 필요한지를 결정한다.

필요한 전체 블록 수(`Grid Size`, `G`)는 Expert 별로 필요한 블록 수의 합으로 계산된다:
- Expert `e` 에 할당된 블록 수
  - `e` 에 할당된 토큰 수: `Count_e`
  - 블록 수: `ceil(Count_e / B)`
- `G = sum(ceil(Count_e / B) for e in 0..E)`

컴파일러는 최악의 경우 `G` 값을 계산하여 메모리 공간을 할당한다.
런타임에는 희소 연산이 비어 있는 Grid 에 대한 실행을 건너뛴다.

모든 Expert 가 토큰 하나만 든 grid 를 포함하는 최악의 경우에는 `(T*K - E) / B + E` 개의 Grid 가 필요하다.

#### 3.3. 인덱스와 Expert ID 생성 (`Cumsum` 기반 주소 계산)

각 토큰의 목적지 블록 주소(`Scatter_Idx`)와 블록별 Expert 할당(`Expert_IDs`)은 토큰-Expert 할당 마스크에 대한 cumsum 으로 병렬 계산된다.
(`Cumsum` 은 branch logging 을 이용해 Vector Engine 에서 구현된다. 하드웨어 구현은 [4절](#4-cumsum-implementation-on-npu)을 보라.)
이 방식은 루프를 피하고 효율적인 병렬 실행을 가능하게 한다:
- **입력**
  - `TopK_Indices: T x K`
  - `Expert_Indices: E = [0, 1, ..., E-1]`
  - `Block_Range: G = [0, 1, ..., G-1]` (최대 블록 수만큼의 수열, 예: 32)
- **출력**
  - `Scatter_Idx: T x K` (각 토큰이 이동할 최종 1D 주소)
  - `Expert_IDs: G` (각 Block 이 담당하는 Expert 번호)
- **연산**
  - **마스크 생성 (One-Hot)**
    - 인덱스를 계산 가능한 마스크 형태로 변환한다.
    - `Expert_Mask: T x K x E = one_hot(TopK_Indices, depth=E)`
  - **히스토그램**
    - 마스크를 합하여 Expert 별로 할당된 토큰 수를 센다.
    - `Count: E = reduce_sum(Expert_Mask, axis: (T, K))`
  - **Block 계산**
    - 각 Expert 에 필요한 Block 수를 계산한다.
      - `Num_Blocks: E = ceil(Count / B)`
  - **전역 오프셋 계산**
    - Cumsum 을 통해 각 Expert 가 전체 Grid(`G`) 에서 시작하는 Block Start Index 를 얻는다.
    - `Global_Offset: E = cumsum(Num_Blocks) - Num_Blocks`
  - **지역 오프셋 계산**
    - Mask 와 Cumsum 으로 각 토큰이 해당 Expert 의 대기열에서 몇 번째인지 계산한다.
    - `Cumsum_Mask: T x K x E = cumsum(Expert_Mask, axis: (T, K))`
    - `Token_Rank: T x K = gather(Cumsum_Mask, index: TopK_Indices)`
    - `Local_Offset: T x K = Token_Rank - 1`
  - **Expert ID 확장**
    - `Diff: E x G = Num_Blocks - Block_Range`
    - `Grid: E x G`
      - ```rust,ignore
        Grid(e, i) = if Diff(e, i) > 0 {
            Expert_Indices(e)
        } else {
            -1
        }
        ```
    - `Expert_IDs: G = filter_compaction(Grid, condition=(Grid >= 0))`
    - 예)
      - expert 0: 2 blocks, expert 1: 3 blocks, expert 3: 3 blocks
      - Diff[0] = [2, 1, 0, -1, -2, ...], Diff[1] = [3, 2, 1, 0, -1, ...]: expert 별 할당 블록 수만큼 양수 항을 가진다.
      - Grid[0] = [0, 0, -1,-1, ...], Grid[1] = [1, 1, 1, -1, -1, ...]: expert 별 할당 블록 수만큼 expert id 를 가진다.
      - Expert_IDs = [0, 0, 1, 1, 1, 3, 3, 3]: Grid 에서 0 이상인 값(expert id)만 필터링한다.
  - **주소 합성**
    - `Scatter_Idx = (Global_Offset * B) + Local_Offset`
    - `T` 개 토큰 각각이 어느 블록의 블록 내 어느 위치에 해당하는지 계산한다. `Scatter_Idx in [0, G * B)`

#### 3.4. Dispatch (블록 단위 Scatter)

Dispatch 는 계산된 주소로 토큰을 물리적으로 재배치하여, 각 토큰을 지정된 블록 위치에 놓는다:

- **입력**
  - `x_norm: T x D` (Attention 과 norm 이후의 입력)
  - `Scatter_Idx: T x K` (각 토큰이 이동할 최종 1D 주소)
- **출력**
  - `x_blocked: G x B x D` (재배치된 Blocked Tensor)
- **연산**
  - **Scatter**
    - `Scatter_Idx` 위치에 토큰 `x_norm` 을 놓는다.

#### 3.5. 희소 연산 (Weight Gather)

희소 연산은 정렬된 Block 들에 Expert 가중치를 적용한다.
핵심은 할당된 토큰이 있는 Expert 의 가중치만 모은다는 점이다:

- **입력**
  - `x_blocked: G x B x D`
  - `Expert_IDs: G` (각 Block 이 담당하는 Expert 번호)
- **출력**
  - `y_blocked: G x B x D`
- **연산**
  - **Weight Gather**
    - `Expert_IDs` 를 인덱스로 삼아 필요한 가중치만 가져온다.
    - `W_gathered_up: G x D x F = gather(W_up, index: Expert_IDs)`
    - `W_gathered_down: G x F x D = gather(W_down, index: Expert_IDs)`
  - **Sparse MLP**
    - 유효한 Block(`G`)에 대해서만 연산이 수행된다.
    - `up: G x B x F = einsum(x_blocked, W_gathered_up)`
    - `y_blocked: G x B x D = einsum(up, W_gathered_down)`

#### 3.6. Combine (가중 합)

Combine 은 결과를 원래 토큰 순서로 되돌리고 Routing 확률을 적용한다.
이것이 MoE 레이어 출력을 만드는 마지막 단계다:

- **입력**
  - `y_blocked: G x B x D`
  - `Scatter_Idx: T x K`
  - `routing_weights: T x K`
- **출력**
  - `moe_out: T x D` (최종 MoE 레이어 출력)
- **연산**
  - **Gather**
    - `Scatter_Idx` 를 역으로 사용하여 `y_blocked` 에서 원래 토큰 순서로 결과를 가져온다.
    - `y_restored: T x K x D = gather(y_blocked, index: Scatter_Idx)`
  - **가중 합**
    - Top-K 과정에서 얻은 `routing_weights` 를 곱하여 최종 출력을 합산한다.
    - `y_weighted: T x K x D = einsum(y_restored, routing_weights)`
    - `moe_out: T x D = reduce_sum(y_weighted, axis: K)`

### 4. TCP 에서의 `Cumsum` 구현

TCP 에서 cumsum 은 branch logging 을 이용해 Vector Engine 에서 구현된다:

1. 정적 branch logger 를 만든다: 합을 계산할 축(크기 n)에 대해,

   ```rust,ignore
   branch(i) = if i == 0 {
       0
   } else if i < n - 1 {
       1
   } else {
       2  // i == n - 1
   }
   ```

2. Vector Engine 을 다음과 같이 설정한다:

   ```rust,ignore
   add %mainstream, OperandRead(branch = 1, 2)
   WriteOperand(branch = 0, 1)
   ```
