# Intra-Slice Reduce

[Intra-Slice Chain](./intra-slice-chain.md) 의 `IntraSliceReduce` 단계는 각 슬라이스의 `Time` 과 `Packet` 에 놓인 차원을 리듀스한다(`Chip`, `Cluster`, `Slice` 는 그대로 통과한다).
[Inter-Slice Reducer](./inter-slice-reducer.md) 는 클러스터의 256개 슬라이스에 걸쳐 리듀스하는 상보적인 경우를 다룬다.

<a id="interface"></a>

## 예제

리듀스 호출의 핵심 매개변수는 다음과 같다.

- **`REDUCE_LABEL`**: 리듀스할 축.
  리듀스는 이 축을 실은 `Time` 과 `Packet` 의 모든 인수를 없애므로, 그것들이 출력 모양(`OutTime`, `OutPacket`)에 나타나서는 안 된다.
  예를 들어 `R` 이 `R / 4` 는 `Time` 에, `R % 4` 는 `Packet` 에 놓이도록 쪼개져 있으면, `REDUCE_LABEL = R` 로 지정하여 둘 다 없앤다.
- **`op`**: 리듀스 연산. `IntraSliceReduceOpI32` 는 `AddSat`, `Max`, `Min` 을 제공하고, `IntraSliceReduceOpF32` 는 `Add`, `Max`, `Min` 을 제공한다.
- **`OutTime`, `OutPacket`**: 리듀스 후의 출력 `Time` 과 `Packet` 모양.
  입력 `Time` 과 `Packet` 에서 모든 `REDUCE_LABEL` 인수를 제거한 것과 일치한다.

아래 예제들은 각 매개변수 조합을 하나씩 보여 준다.

### `Time` 에서의 리듀스

`R` 이 `Time` 에만 있으므로, 이 단계는 시간 단계에 걸쳐 누적한다.
포화 덧셈으로 \\(output[a] = \sum_{r \in R} input[a, r]\\) 을 계산한다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, R = 16];

// R in Time → temporal accumulation. Packet is non-reduce.
fn reduce_time<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1 # 2], m![A / 2], m![R], m![A % 2 # 8], i32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1 # 2], m![A / 2], m![1], m![A % 2 # 4], i32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_trim::<m![A % 2 # 4]>()       // 8-way → 4-way
        // R eliminated from Time
        .vector_intra_slice_reduce::<R, m![1], m![A % 2 # 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
}
# 
# let mut ctx = Context::acquire();
# 
# let i: VectorBranchTensor<'_, _, i32, m![1], m![1 # 2], m![A / 2], m![R], m![A % 2 # 8], i32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let _o = reduce_time(i);
```

### `Packet` 에서의 리듀스

`R` 이 `Packet` 에만 있으므로, 하드웨어는 각 flit 안에서 4-way 트리 리듀스를 돌리고 시간 누적은 건너뛴다.
\\(output[a] = \sum_{r \in R} input[a, r]\\) 을 계산한다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, R = 4];

// R in Packet → tree reduce within flit.
fn reduce_packet<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1 # 2], m![A / 2], m![A % 2], m![R # 8], f32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1 # 2], m![A / 2], m![A % 2], m![1 # 4], f32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_trim::<m![R]>()             // 8-way → 4-way
        // R eliminated from Packet
        .vector_intra_slice_reduce::<R, m![A % 2], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
}
# 
# let mut ctx = Context::acquire();
# 
# let i: VectorBranchTensor<'_, _, f32, m![1], m![1 # 2], m![A / 2], m![A % 2], m![R # 8], f32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let _o = reduce_packet(i);
```

### 둘 다에서의 리듀스

`R` 이 `Packet` 과 `Time` 에 걸쳐 쪼개진다.
`R % 4` 는 `Packet` 에서 각 flit 안의 트리 리듀스가 되고, `R / 4` 는 `Time` 에서 시간 단계에 걸쳐 누적된다.
\\(output[a] = \max_{r \in R} input[a, r]\\) 을 계산한다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 256, R = 16];

// R % 4 in Packet → spatial tree reduce
// R / 4 in Time → temporal accumulation
fn reduce_time_packet<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1 # 2], m![A], m![R / 4], m![R % 4 # 8], f32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1 # 2], m![A], m![1], m![1 # 4], f32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_trim::<m![R % 4]>()            // 8-way → 4-way
        // R eliminated from both Time and Packet
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpF32::Max,
        )
}
# 
# let mut ctx = Context::acquire();
# 
# let i: VectorBranchTensor<'_, _, f32, m![1], m![1 # 2], m![A], m![R / 4], m![R % 4 # 8], f32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let _o = reduce_time_packet(i);
```

### 슬라이스별 리듀스

`R` 은 `Slice`, `Time`, `Packet` 에 걸쳐 부분을 갖지만, intra-slice 리듀서는 `Time` 과 `Packet` 부분만 접으므로 `Slice` 에 놓인 `R` 의 부분은 출력에 남는다.
여기서 `R = 13` 은 레이아웃에 맞추려고 32로 패딩되고(`R # 32`), 이어서 `R` 은 4개 슬라이스에 걸쳐 쪼개진다(슬라이스당 `R` 위치 8개).
위치 0-12 만 실제 원소를 담으므로, 그 경계에 걸친 슬라이스(위치 8-15: 실제 5개 뒤에 패딩 3개)가 경계 슬라이스이고, 그 뒤의 슬라이스들은 전부 패딩이다.
[VCG](#padding-strategy) 가 슬라이스별 리듀스 개수를 몰아 주어 각 슬라이스가 자신의 실제 원소만 리듀스하게 한다(정확한 매핑은 [Valid Count Generator](./vcg.md) 참고).

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![R = 13];

// R split across all three: Slice (groups of 8), Time (pairs within group), Packet (4 elements).
fn reduce_slice_time_packet<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1 # 2], m![R # 32 / 8 # 256], m![R # 32 / 4 % 2], m![R # 32 % 4 # 8], i32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1 # 2], m![R # 32 / 8 # 256], m![1], m![1 # 4], i32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_trim::<m![R # 32 % 4]>()       // 8-way → 4-way
        // R eliminated from Time and Packet (accumulated within each slice)
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::Min,
        )
}
# 
# let mut ctx = Context::acquire();
# 
# let i: VectorBranchTensor<'_, _, i32, m![1], m![1 # 2], m![R # 32 / 8 # 256], m![R # 32 / 4 % 2], m![R # 32 % 4 # 8], i32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let _o = reduce_slice_time_packet(i);
```

## 구조

이 단계는 `Time` 축과 `Packet` 축에 대해 서로 다른 기구를 돌린다.

### `Time` 에서의 리듀스

이 단계는 슬롯 용량 8 인 [시간 누산기](../contraction-engine/time-reducer.md#constraints) 모델을 적용한다(따라서 `InnerTime::SIZE ≤ 8`).

위의 `reduce_time` 에서는 `Time = m![R]` 이고 `OutTime = m![1]` 이므로, `R` 이 가장 바깥쪽 리듀스 차원이고 `InnerTime = m![1]` 이다(`InnerTime::SIZE = 1`). 슬롯 하나가 모든 `R` 값을 출력으로 누적한다.

`InnerTime::SIZE` 가 8을 넘으면 API 가 호출을 거부한다. 예를 들면 다음과 같다:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# axes![A = 6, B = 16, R = 16];
fn invalid_too_many_slots<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1 # 2], m![A / 3 # 256], m![R, A % 3, B % 4], m![B / 4 # 8], i32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1 # 2], m![A / 3 # 256], m![A % 3, B % 4], m![B / 4], i32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_trim::<m![B / 4]>()
        // Time      = m![R, A % 3, B % 4]
        // OutTime   = m![A % 3, B % 4]
        // InnerTime = m![A % 3, B % 4], InnerTime::SIZE = 3 × 4 = 12 > 8
        .vector_intra_slice_reduce::<R, m![A % 3, B % 4], m![B / 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
    // Rejected: 12 accumulator slots required, but only 8 are available.
}
# 
# let mut ctx = Context::acquire();
# 
# let i: VectorBranchTensor<'_, _, i32, m![1], m![1 # 2], m![A / 3 # 256], m![R, A % 3, B % 4], m![B / 4 # 8], i32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let _o = invalid_too_many_slots(i);
```

### `Packet` 에서의 리듀스

flit 당 4개의 Packet 원소는 두 경로 중 하나를 택한다.

- `OutPacket = Packet`: 원소들이 리듀스되지 않고 그대로 통과한다. 사이클당 출력 4개이며, 각각 `Time` 에 걸쳐 독립적으로 누적된다.
- `OutPacket = m![1 # 4]`: 원소들이 2단 트리 `op(op(a, b), op(c, d))` 를 거쳐 값 하나로 접힌다. 사이클당 출력 1개와 패딩 위치 3개다.

<a id="padding-strategy"></a>

리듀스 축이 하드웨어 차원에 맞추려고 패딩되면, 패딩된 위치에는 리듀스가 배제해야 할 임의의 데이터가 들어 있다.
패딩 배제는 두 가지 전략으로 처리한다.

- **VCG (Valid Count Generator)**: 축 배치가 지원되는 경우 이쪽이 낫다.
  컴파일러가 매핑으로부터 VCG 를 자동으로 설정하고, VCG 는 각 flit 에 `valid_count` 를 태깅하여 패딩 원소를 자동으로 배제한다.
  Slice, Time, Packet 에 걸친 모든 축 배치가 지원되지는 않는다.
  자세한 내용은 [Valid Count Generator](./vcg.md) 를 참고한다.

- **항등원 패딩**: 데이터가 [Intra-Slice Chain](./intra-slice-chain.md) 에 닿기 전에 패딩 위치를 리듀스 연산의 항등원으로 채운다.
  [Fetch Engine 의 마스킹](../../moving-tensors/fetch-engine.md#masking)이 fetch 중에 패딩 위치에 항등값을 쓴다:

  | 연산 | 항등원 |
  |-----------|-----------------|
  | `AddSat` / `Add` | `0` / `0.0` |
  | `Max` | `i32::MIN` / `f32::NEG_INFINITY` |
  | `Min` | `i32::MAX` / `f32::INFINITY` |

  이 전략은 리듀스 연산 앞에 역함수가 없는 변환이 오지 않을 때만 쓸 수 있다.
  예를 들어 `exp(x) + exp(y) + ...`(지수 합)에서는 어떤 값 `p` 도 `exp(p) = 0`(덧셈의 항등원)을 만족하지 않으므로 항등원 패딩을 적용할 수 없다.


## 성능

트리 리듀스가 Intra-Slice Chain 안에서 완전히 파이프라인되어 flit 당 추가 비용이 없으므로, 처리량은 사이클당 flit 하나를 유지한다.

지연에는 첫 출력까지 `n` flit 사이클의 지연이 더해지는데, 여기서 `n` 은 리듀스 축의 시간 단계 수다. 이 단계가 결과를 내보내기 전에 한 리듀스 그룹의 입력 flit 을 전부 누적해야 하기 때문이다.
여러 엔진으로 이루어진 파이프라인에서는 이 누적 지연이 첫 flit 을 기다리는 하류 엔진들을 멈추게 한다.

