# Split 리듀스


split 리듀스는 논리적 리듀스 축을 연속된 단일 하드웨어 차원에 매핑할 수 없을 때의 축약(리듀스)을 다룬다.
축은 여러 개의 별도 텐서 인스턴스로 쪼개지고, 이들은 각각 독립적으로 fetch 한 뒤 합쳐야 한다.
fetch 는 인터리브드 fetch 를 쓰고, 결합은 Vector Engine 이항 연산을 쓴다.

## Split 리듀스를 쓸 때

split 리듀스는 다음 경우에 해당한다:

- **쪼개야 할 때**: 리듀스 축이 너무 커서 단일 텐서로는 VRF(슬라이스당 8KB)에 들어가지 않아, 논리 축을 여러 물리 텐서 인스턴스로 쪼개야 한다.
- **데이터가 이미 쪼개져 있을 때**: 여러 텐서 인스턴스가 같은 논리 리듀스 축의 서로 다른 부분을 독립적으로 갖고 있다(예: 서로 다른 모델 레이어, expert, 시간 구간에서 나온 것).
- **칩 간 통신을 피할 때**: 데이터가 같은 chip/cluster 에 있으나 별도의 메모리 할당에 있어, DMA 기반 방식보다 인터리브드 fetch 가 더 효율적이다.

다중 인스턴스 fetch 후 결합 패턴으로서, split 리듀스는 TCP 의 리듀스 계층에서 슬라이스 수준과 칩 수준 리듀스 사이에 놓인다:

- **Packet 리듀스**: 단일 packet 안에서(Packet Reducer)
- **Time 리듀스**: 시간 차원에 걸쳐(Time Reducer)
- **Slice 리듀스**: 클러스터 안 슬라이스에 걸쳐(Inter-Slice Reducer)
- **Split 리듀스**: 여러 독립 텐서 인스턴스에 걸쳐, *인터리브드 fetch*(별도 텐서 인스턴스에서 번갈아 로드)와 Vector Engine 이항 연산을 조합해 수행
- **Chip/Cluster 리듀스**: 칩 또는 클러스터에 걸쳐(DMA + 인터리브드 fetch + Vector Engine 이항 연산)


## 구현: 인터리브드 Fetch

이 fetch 패턴은 별도의 텐서 인스턴스를 인덱싱하는 인터리브 차원 `I` 를 도입해, Vector Engine 이 리듀스하는 시간 인터리브 스트림을 만든다:

```rust,ignore
// Two tensor instances to be reduced together
let tensor_0: DmTensor<bf16, m![1], m![1], m![1], m![A, B]> = ...;
let tensor_1: DmTensor<bf16, m![1], m![1], m![1], m![A, B]> = ...;

// Interleaved fetch creates alternating time stream: I=2 dimension
let interleaved: TuTensor<bf16, m![1], m![1], m![1],
    m![I: 2, A], m![B]
> = ctx.main.begin_interleaved().fetch(&tensor_0, &tensor_1);

// Vector Engine reduction combines the I dimension
let reduced: TuTensor<bf16, m![1], m![1], m![1],
    m![A], m![B]
> = interleaved.reduce_add(axis: I);
```

인터리브드 fetch 는 시간 차원에서 텐서 인스턴스를 번갈아 오간다. `time[0]` 은 `tensor_0` 의 데이터를, `time[1]` 은 `tensor_1` 의 데이터를, `time[2]` 는 다시 `tensor_0` 의 데이터를 갖는 식이다. Vector Engine 은 인터리브 차원에 걸쳐 이항 연산(add, max, min)을 수행해 리듀스를 완료한다.

## 예제 1: Layer Normalization Split 리듀스

Layer normalization 은 Hidden 차원이 VRF 용량을 넘을 때 split 리듀스를 부른다.
특징 차원을 여러 청크로 쪼개 따로 처리한 뒤 합쳐야 한다.
Layer normalization 은 특징 차원 전체에 대해 통계(평균, 분산)를 계산하므로 Hidden 축 전체를 리듀스해야 한다.

**문제:**
Layer normalization 은 각 토큰에 대해 모든 특징의 평균과 분산을 계산해야 한다.
수식은 다음과 같다:
```text
output = (input - mean) / sqrt(variance + epsilon)
```
여기서 평균과 분산은 `Hidden` 차원 전체에 대해 계산된다.

`Hidden` 이 매우 크면(원소 8,192 개처럼) 텐서가 8KB VRF 에 들어가지 않으므로 한 번의 연산으로 리듀스할 수 없다.

**입력:**
트랜스포머 활성값을 나타내는 3D 텐서:
- **모양**: `[Batch=32, SeqLen=128, Hidden=8192]`
- **데이터 타입**: `bf16` (원소당 2 바이트)
- **총 크기**: 32 × 128 × 8192 × 2 바이트 = 64 MB
- **토큰별 슬라이스**: 4,096 개 토큰(32 × 128) 각각에 대해 8,192 개 특징 = 토큰당 16 KB
- **VRF 제약**: 슬라이스당 8KB 뿐 ≈ `bf16` 원소 4,096 개
- **문제**: 한 토큰의 8,192 개 특징을 동시에 로드할 수 없다

**해결 전략:**
`Hidden` 차원을 4,096 원소짜리 청크 두 개로 쪼갠다:
- **청크 0**: `[Batch=32, SeqLen=128, Hidden_0=4096]` - 특징의 앞 절반
- **청크 1**: `[Batch=32, SeqLen=128, Hidden_1=4096]` - 특징의 뒤 절반
- 청크 하나 = 원소 4,096 개 × 2 바이트 = 8KB, VRF 에 들어간다

### 단계별 실행

#### 1 단계: 부분 통계 계산

먼저 각 청크의 통계를 독립적으로 계산한다:

```rust,ignore
// Chunk 0: Hidden dimensions 0..4096
let chunk_0: DmTensor<bf16, m![1], m![1], m![1], m![Batch, SeqLen, Hidden_0: 4096]> = ...;

// Chunk 1: Hidden dimensions 4096..8192
let chunk_1: DmTensor<bf16, m![1], m![1], m![1], m![Batch, SeqLen, Hidden_1: 4096]> = ...;

// Compute sum for each chunk (using Packet Reducer + Inter-Slice Reducer)
let sum_0: DmTensor<f32, m![1], m![1], m![1], m![Batch, SeqLen]> = chunk_0.reduce_sum(axis: Hidden_0);
let sum_1: DmTensor<f32, m![1], m![1], m![1], m![Batch, SeqLen]> = chunk_1.reduce_sum(axis: Hidden_1);
```

#### 2 단계: 인터리브드 Fetch 와 결합

split 리듀스로 부분합을 합친다:

```rust,ignore
// Fetch both chunks in interleaved pattern
let interleaved_sums: TuTensor<f32, m![1], m![1], m![1],
    m![I: 2, Batch, SeqLen], m![1]
> = ctx.main.begin_interleaved().fetch(&sum_0, &sum_1);

// Vector Engine adds across I dimension to get total sum
let total_sum: TuTensor<f32, m![1], m![1], m![1],
    m![Batch, SeqLen], m![1]
> = interleaved_sums.reduce_add(axis: I);

// Compute mean: total_sum / Hidden
let mean = total_sum * (1.0 / 8192.0);  // Vector Engine scalar multiply
```

#### 3 단계: 분산 계산

2 단계에서 계산한 `mean` 을 써서 부분 분산을 계산하고 합친다:

```rust,ignore
// Compute squared differences for each chunk
let sq_diff_0 = (chunk_0 - mean).square().reduce_sum(axis: Hidden_0);
let sq_diff_1 = (chunk_1 - mean).square().reduce_sum(axis: Hidden_1);

// Split reduce to combine variance contributions
let interleaved_vars: TuTensor<f32, m![1], m![1], m![1],
    m![I: 2, Batch, SeqLen], m![1]
> = ctx.main.begin_interleaved().fetch(&sq_diff_0, &sq_diff_1);

let total_variance = interleaved_vars.reduce_add(axis: I);
let std = total_variance.sqrt();
```

**출력:**

세 단계는 layer normalization 에 필요한 통계를 만든다:
- **평균**: `[Batch=32, SeqLen=128]` - 토큰당 평균값 하나로, 8,192 개 특징 전체의 평균을 나타낸다
- **표준편차**: `[Batch=32, SeqLen=128]` - 토큰당 표준편차 값 하나
- **결과**: 이 통계로 각 토큰의 8,192 개 특징을 정규화한다:
  ```text
  normalized_chunk_0 = (chunk_0 - mean) / std
  normalized_chunk_1 = (chunk_1 - mean) / std
  ```

통계를 두 청크로 나눠 계산해도 8,192 개 특징 전체를 한 번에 계산한 것과 수학적으로 같은 결과가 나온다:
- **수학적으로**: `mean([a,b,c,d,e,f]) = (sum(a,b,c) + sum(d,e,f)) / 6`
- **실제로는**: `mean([Hidden_0, Hidden_1]) = (sum(Hidden_0) + sum(Hidden_1)) / 8192`

split 리듀스는 VRF 용량 한계에도 불구하고 전역 통계를 계산한다.

### 하드웨어 매핑

split 리듀스 연산은 다음과 같이 하드웨어에 매핑된다.

| 연산 | 하드웨어 구성요소 | 사이클 |
|-----------|-------------------|--------|
| chunk_0 fetch | Fetch Engine | 32 바이트 flit 당 ~1 사이클 |
| chunk_1 fetch | Fetch Engine (인터리브) | 32 바이트 flit 당 ~1 사이클 |
| 인터리브 차원 생성 | Fetch Sequencer | 0 (구조적 변환) |
| I 에 걸친 이항 add | Vector Engine | packet 당 1 사이클 |


### 성능 분석

**split 리듀스의 총 사이클**:
- 두 텐서 fetch: `2 * (Batch * SeqLen * ceil(Hidden / flit_elements))` 사이클
- Vector Engine 리듀스: `(Batch * SeqLen)` 사이클
- 합계: fetch 시간이 지배적이며, 이 예제에서는 ~8K 사이클

**병목**: 두 텐서 인스턴스를 순차적으로 fetch 하는 메모리 대역폭.

**최적화**: 가능하면 리듀스 축을 쪼개지 않도록 계산을 재구성한다. 축을 쪼개야 한다면 분할 인스턴스 개수를 최소화한다.

## 예제 2: 쪼개진 배치에 걸친 Batch Normalization

배치 차원이 독립된 두 할당에 쪼개져 있을 때, split 리듀스는 할당별 통계를 합쳐 전역 batch normalization 결과를 만든다.
Batch normalization 은 배치 차원 전체에 걸쳐 통계를 계산하므로 모든 할당을 함께 리듀스해야 한다.

### 문제 설정

- **입력**: `[Batch_0 = 256, ...], [Batch_1 = 256, ...]` (별개의 배치 텐서 두 개)
- **리듀스 목표**: 512 개 예제 전체에 걸친 평균과 분산 계산
- **제약**: 메모리 한계 때문에 512 배치용 단일 텐서를 할당할 수 없다

### 실행 패턴

```rust,ignore
// Two batch allocations
let batch_0: DmTensor<bf16, m![1], m![1], m![1], m![Batch_0: 256, C, H, W]> = ...;
let batch_1: DmTensor<bf16, m![1], m![1], m![1], m![Batch_1: 256, C, H, W]> = ...;

// Compute per-batch statistics (reduce over H, W)
let batch_stats_0 = batch_0.reduce_mean(axis: [H, W]);  // [Batch_0=256, C]
let batch_stats_1 = batch_1.reduce_mean(axis: [H, W]);  // [Batch_1=256, C]

// Split reduce to combine batch statistics
let interleaved: TuTensor<f32, m![1], m![1], m![1],
    m![I: 2, Batch: 256, C], m![1]
> = ctx.main.begin_interleaved().fetch(&batch_stats_0, &batch_stats_1);

// Compute global statistics across all batches
let global_mean = interleaved.reduce_mean(axis: I);  // Average the two batch means
```

이 패턴은 인터리브 차원을 늘려 두 개보다 많은 분할로 자연스럽게 확장된다. 네 개로 쪼개면 `I: 4` 식이다.

## 예제 3: Mixture of Experts 부분 리듀스


### 문제 설정

- **Expert 출력**: 서로 다른 expert 평가에서 나온 여러 텐서
- **라우팅 가중치**: 각 expert 가 얼마나 기여하는지 정하는 가중치
- **목표**: expert 출력에 걸친 가중합

### 실행 패턴

```rust,ignore
// Expert outputs from separate evaluations (simplified: 2 experts)
let expert_0_output: DmTensor<bf16, m![1], m![1], m![1], m![Tokens, Hidden]> = ...;
let expert_1_output: DmTensor<bf16, m![1], m![1], m![1], m![Tokens, Hidden]> = ...;

let routing_weights: [f32; 2] = [0.7, 0.3];  // Per-expert weights

// Apply routing weights during fetch using zero-point arithmetic or scaling
let weighted_0 = expert_0_output * routing_weights[0];
let weighted_1 = expert_1_output * routing_weights[1];

// Split reduce to combine weighted expert contributions
let interleaved: TuTensor<bf16, m![1], m![1], m![1],
    m![I: 2, Tokens], m![Hidden]
> = ctx.main.begin_interleaved().fetch(&weighted_0, &weighted_1);

let combined_output = interleaved.reduce_add(axis: I);
```


## 예제 4: 윈도우에 걸친 시간 리듀스


### 문제 설정

- **입력**: 시간 청크로 쪼개진 비디오 프레임 또는 시퀀스 토큰
- **목표**: 모든 청크에 걸친 전역 통계 계산
- **제약**: 메모리 한계 때문에 모든 청크를 동시에 로드할 수 없다

### 실행 패턴

```rust,ignore
// Temporal chunks
let chunk_t0: DmTensor<bf16, m![1], m![1], m![1], m![Time_0: 128, Features]> = ...;
let chunk_t1: DmTensor<bf16, m![1], m![1], m![1], m![Time_1: 128, Features]> = ...;
let chunk_t2: DmTensor<bf16, m![1], m![1], m![1], m![Time_2: 128, Features]> = ...;
let chunk_t3: DmTensor<bf16, m![1], m![1], m![1], m![Time_3: 128, Features]> = ...;

// Compute per-chunk max (e.g., for max pooling over time)
let max_t0 = chunk_t0.reduce_max(axis: Time_0);  // [Features]
let max_t1 = chunk_t1.reduce_max(axis: Time_1);  // [Features]
let max_t2 = chunk_t2.reduce_max(axis: Time_2);  // [Features]
let max_t3 = chunk_t3.reduce_max(axis: Time_3);  // [Features]

// Split reduce with I=4 to find global maximum
let interleaved: TuTensor<bf16, m![1], m![1], m![1],
    m![I: 4], m![Features]
> = ctx.main.begin_interleaved().fetch(&max_t0, &max_t1, &max_t2, &max_t3);

let global_max = interleaved.reduce_max(axis: I);
```

## 다른 리듀스 방법과의 비교

split 리듀스에는 주요 대안이 두 가지 있다. 같은 텐서 분산에는 slice 리듀스/Inter-Slice Reducer, 칩 간 데이터에는 chip/cluster 리듀스다.
이 중 무엇을 고를지는 데이터 위치, 텐서 모양, 그리고 데이터를 단일 할당으로 합칠 수 있는지에 달렸다.

### Split 리듀스 vs. Slice 리듀스 (Inter-Slice Reducer)

| 항목 | Split 리듀스 | Slice 리듀스 (Inter-Slice Reducer) |
|--------|--------------|-------------------|
| 데이터 배치 | 독립된 여러 텐서 | 슬라이스에 걸친 단일 텐서 |
| fetch 패턴 | 여러 소스에서 인터리브드 fetch | 단일 연속 fetch |
| 리듀스 하드웨어 | Vector Engine 이항 연산 | Inter-Slice Reducer |
| 일반적인 사이클 | fetch 시간의 ~2배 | ~256 사이클 (슬라이스 리듀스) |
| 사용 사례 | 데이터가 단일 텐서에 들어가지 않음 | 데이터가 하드웨어에 분산됨 |

**split 리듀스가 나은 경우**: 메모리 할당 제약 때문에 단일 텐서로 합칠 수 없는 여러 텐서 인스턴스가 모두 같은 chip/cluster 에 있을 때.

**slice 리듀스가 나은 경우**: 슬라이스에 걸친 단일 텐서를 할당해 하드웨어가 분산을 자동으로 처리하게 할 때.

### Split 리듀스 vs. Chip/Cluster 리듀스

| 항목 | Split 리듀스 | Chip/Cluster 리듀스 |
|--------|--------------|---------------------|
| 데이터 위치 | 같은 chip/cluster | 칩/클러스터에 걸침 |
| 통신 | 로컬 메모리 fetch | 칩 인터커넥트를 통한 DMA |
| 오버헤드 | 최소 (인터리브드 fetch) | 큼 (DMA + 동기화) |
| 대역폭 | SRAM 대역폭 | 칩 인터커넥트 대역폭 |

**split 리듀스가 나은 경우**: 별도 할당에 있더라도 모든 데이터가 같은 칩에 있을 때.

**chip/cluster 리듀스가 나은 경우**: 데이터가 물리적으로 분리된 처리 유닛에 분산되어 칩 간 통신이 필요할 때.

## 구현 방법

split 리듀스 연산은 다음 하드웨어 프리미티브로 매핑된다:

- **인터리브드 fetch**: `begin_interleaved()` 모드의 Fetch Engine 이 `I` 인터리브 차원을 만든다
- **I 에 걸친 리듀스**: 인터리브 축을 리듀스하도록 설정된 Vector Engine 이항 연산(add, max, min)
- **2 분할일 때의 대안**: 명시적 인터리브 차원 없이 이항 연산을 직접 쓸 수 있다

### 두 인스턴스 최적화

정확히 두 인스턴스로 쪼개는 흔한 경우에는, Vector Engine 이 명시적 인터리브 차원을 만들지 않고 리듀스를 수행할 수 있다:

```rust,ignore
// Direct binary operation for 2-way split
let sum_0: TuTensor<f32, m![1], m![1], m![1], m![A], m![B]> = ...;
let sum_1: TuTensor<f32, m![1], m![1], m![1], m![A], m![B]> = ...;

// Fetch both and add in one operation
let total = sum_0.binary_add(sum_1);  // No interleave dimension needed
```

이 최적화는 fetch 와 리듀스를 단일 파이프라인 연산으로 결합해 오버헤드를 줄인다.

## 성능 고려사항

### 사이클 분석

split 리듀스의 사이클 수는 fetch 시간이 지배하며, Vector Engine 사이클과 파이프라인 중첩이 부차적 요인이다:

- **Fetch 사이클**: `N_splits * fetch_cycles_per_tensor`
- **Vector Engine 사이클**: `Time_dim_size * cycles_per_packet` (보통 packet 당 1 사이클)
- **파이프라인 중첩**: 가능한 경우 fetch 와 VE 연산이 겹칠 수 있다

**총 사이클** ≈ `N_splits * fetch_cycles + max(0, VE_cycles - pipeline_overlap)`

### 메모리 대역폭

split 리듀스는 분할 개수에 비례해 메모리 대역폭을 소비한다:

- **2 분할**: 단일 텐서 대비 메모리 대역폭 2배
- **4 분할**: 단일 텐서 대비 메모리 대역폭 4배

**최적화**: VRF 용량 안에서 개별 텐서 크기를 최대화해 분할 개수를 최소화한다.

### 대안과의 비교

N 개 텐서 인스턴스를 합쳐야 하는 리듀스의 경우:

| 방법 | 사이클 | 메모리 BW | 복잡도 |
|--------|--------|-----------|------------|
| Split 리듀스 (인터리브드) | ~N * fetch + VE | N * tensor_size | 낮음 |
| 순차 fetch + 누적 | ~N * (fetch + VE) | N * tensor_size | 중간 |
| 단일 버퍼로 DMA + 리듀스 | DMA + single_reduce | N * tensor_size | 높음 |

인터리브드 fetch 를 쓰는 split 리듀스는 같은 칩 안의 리듀스에서 성능과 구현 단순성의 균형이 가장 좋다.

## 제약과 한계

### 하드웨어 제약

- **인터리브 차원 크기**: Fetch Engine 능력에 의해 제한된다
- **텐서 정렬**: 인터리빙하려면 모든 텐서 인스턴스의 모양이 호환되어야 한다
- **VRF 용량**: 인터리빙 후 합쳐진 텐서가 VRF(슬라이스당 8KB)에 들어가야 한다


### Split 리듀스가 최적이 아닐 때

- **단일 텐서가 가능할 때**: 데이터가 텐서 할당 하나에 들어가면 대신 slice 리듀스(Inter-Slice Reducer)를 쓴다
- **칩 간 리듀스가 필요할 때**: 데이터가 칩에 걸치면 DMA 를 쓰는 chip/cluster 리듀스를 쓴다
- **분할 개수가 아주 많을 때**: ~8 분할을 넘으면 다른 메모리 관리 전략을 고려한다

### 모범 사례

- **분할 최소화**: 필요한 분할 개수가 최소가 되도록 텐서 할당을 설계한다
- **2 의 거듭제곱 분할**: 하드웨어 활용을 최적화하려면 가능한 경우 2, 4, 8 분할을 쓴다
- **리듀스 결과 재사용**: 같은 조합이 여러 번 필요하면 split 리듀스 결과를 캐시한다
- **메모리 레이아웃 고려**: 효율적인 인터리브드 fetch 패턴이 가능하도록 텐서 할당을 구성한다
