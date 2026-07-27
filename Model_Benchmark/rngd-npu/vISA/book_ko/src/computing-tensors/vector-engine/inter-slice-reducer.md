# Inter-Slice Reducer

Inter-Slice Reducer 는 클러스터 안 256개 슬라이스에 걸쳐 텐서를 리듀스한다.
`Chip`, `Cluster`, `Packet` 은 보존하고, `Slice` 와 `Time` 은 `OutSlice` 와 `OutTime` 으로 다시 쓴다.
출력 텐서는 입력 모드와 무관하게 항상 `Way8` 이다.

## 인터페이스

리듀서에는 `vector_init()` 직후에 진입하거나(아래에 보인 `InterFirst` 경로), [호환되는 intra-slice 단계](./intra-slice-chain.md#transitioning-to-the-inter-slice-reducer)에서 진입할 수 있다(`IntraFirst` 경로로, intra-slice 텐서에 대해 같은 `vector_inter_slice_reduce()` 메서드를 호출한다).
아래에 보인 시그니처는 `VectorInitTensor` 쪽 변형이다. 전이를 지원하는 단계의 intra-slice 텐서에도 같은 메서드가 존재하므로 호출부 모습은 동일하다.

inter-slice 리듀서는 `i32` 와 `f32` 에 대해 별도의 API 를 제공한다.

### `i32` 연산

```rust,ignore
impl<'l, const T: Tu, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorInitTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet>
{
    /// Performs inter-slice reduce for i32 as the first VE operation.
    #[primitive(VectorInitTensor::vector_inter_slice_reduce)]
    pub fn vector_inter_slice_reduce<OutSlice: M, OutTime: M>(
        self,
        op: InterSliceReduceOpI32,
    ) -> VectorInterSliceReduceTensor<'l, T, i32, Chip, Cluster, OutSlice, OutTime, Packet, { VeOrder::InterFirst }>
    {
        let reduced = self.inner.reduce(op.reduce_fn(), op.identity(), true);
        create_inter_slice_reduce_tensor(self.ctx, reduced)
    }
}
```

`InterSliceReduceOpI32` 연산:

| 연산 | 설명 |
|-----------|-------------|
| `Add` | 랩어라운드 덧셈 |
| `AddSat` | 포화 덧셈 |
| `Max` | 최댓값 |
| `Min` | 최솟값 |

### `f32` 연산

```rust,ignore
impl<'l, const T: Tu, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorInitTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet>
{
    /// Performs inter-slice reduce for f32 as the first VE operation.
    #[primitive(VectorInitTensor::vector_inter_slice_reduce)]
    pub fn vector_inter_slice_reduce<OutSlice: M, OutTime: M>(
        self,
        op: InterSliceReduceOpF32,
    ) -> VectorInterSliceReduceTensor<'l, T, f32, Chip, Cluster, OutSlice, OutTime, Packet, { VeOrder::InterFirst }>
    {
        let reduced = self.inner.reduce(op.reduce_fn(), op.identity(), true);
        create_inter_slice_reduce_tensor(self.ctx, reduced)
    }
}
```

`InterSliceReduceOpF32` 연산:

| 연산 | 설명 |
|-----------|-------------|
| `Add` | 부동소수점 덧셈 |
| `Max` | 최댓값 |
| `Min` | 최솟값 |
| `Mul` | 부동소수점 곱셈 |

## 제약

지원하는 `Slice → OutSlice` 및 `Time → OutTime` 모양은 네 가지 규칙을 따른다.

1. **가장 안쪽부터 리듀스한다.** 리듀스되는 `Slice` 부분은 가장 안쪽 인수여야 하고, 연속적이며, 리듀스 비율 `r` 까지 스트라이드 1 이어야 한다.
2. **리듀스된 축의 자리 채우기.** 리듀스된 각 인수가 `OutSlice` 에서 차지하던 자리는 더미(`1 # n`), 새 차원에 대한 브로드캐스트, `Time` 으로부터의 승격 중 하나로 채워진다.
3. **자리 채우기 종류는 자유롭게 섞인다.** 더미·브로드캐스트·승격 자리는 `OutSlice` 안에서 순서에 상관없이 함께 나타날 수 있다.
4. **`Time` 에서 `OutSlice` 로의 승격은 순서를 바꾼다.** `Time → OutTime` 부분은 살아남은 인수들의 상대 순서를 보존하지만, `Time → OutSlice` 승격 경로는 순서를 보존하지 않는다. 승격된 인수가 `OutSlice` 에서 갖는 위치는 `Time` 에서의 위치와 무관하다.

## 예제

아래 예제의 수식은 einsum 표기를 쓴다.
입력 쪽에는 나타나지만 출력 쪽에는 없는 차원은 리듀스(합산)되고, 출력 쪽에는 나타나지만 입력 쪽에는 없는 차원은 브로드캐스트된다.

### 더미로 채우기

이 패스는 `R` 에 걸쳐 입력을 합하고 결과를 더미 자리에 넣는다.
einsum 형태는 `AR -> A` 이다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2, R = 4];

// When R is reduced and no other dimension fills its slot, the output keeps the slot as a 1 # n dummy.
// One position holds the reduced value, and the remaining n - 1 are padding.
// `# n` denotes dimension multiplicity (see the Mapping Expressions doc).
fn inter_slice_add<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8, R], m![1], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8, 1 # 4], m![1], m![A % 8]> {
    input
        .vector_init()
        // sum across R, the freed R-slot becomes the 1 # 4 dummy in OutSlice
        .vector_inter_slice_reduce::<m![A / 8, 1 # 4], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, i32, m![1], m![B], m![A / 8, R], m![1], m![A % 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = inter_slice_add(c);
```

```text
Slice = [A / 8, R]  ->  [A / 8, 1 # 4]
Time  = [1]         ->  [1]
```

### 새 Slice 차원으로 브로드캐스트

이 패스는 `R` 을 리듀스하고 그 결과를 새 차원 `X` 에 걸쳐 브로드캐스트한다.
einsum 형태는 `PRW -> PWX` 이며, 출력 쪽의 새 `X` 가 브로드캐스트된다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![B = 2, P = 8, R = 4, W = 64, X = 4];

// A fresh non-reduce dimension X takes the slot that R leaves behind.
// The reduced value broadcasts across every position of X.
fn broadcast_into_x<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![B], m![W, R], m![1], m![P]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![B], m![W, X], m![1], m![P]> {
    input
        .vector_init()
        // sum across R, broadcast result over X (fresh OutSlice dimension)
        .vector_inter_slice_reduce::<m![W, X], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, f32, m![1], m![B], m![W, R], m![1], m![P]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = broadcast_into_x(c);
```

```text
Slice = [W, R]  ->  [W, X]
Time  = [1]     ->  [1]
```

### `Time` 에서 `OutSlice` 로 승격

이 패스는 `R` 을 리듀스하고 `Time` 차원 `V` 를 OutSlice 로 승격한다.
einsum 형태는 `PRSUVW -> PSUVW` 이다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![B = 2, P = 8, R = 4, S = 2, U = 2, V = 4, W = 64];

// A dimension from Time (here V) is promoted into OutSlice to fill R's slot.
// The promoted dimension does not need to be outermost in Time.
fn axis_promotion<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![B], m![W, R], m![S, V, U], m![P]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![B], m![W, V], m![S, U], m![P]> {
    input
        .vector_init()
        // sum across R, V moves from Time to OutSlice
        .vector_inter_slice_reduce::<m![W, V], m![S, U]>(InterSliceReduceOpF32::Add)
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, f32, m![1], m![B], m![W, R], m![S, V, U], m![P]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = axis_promotion(c);
```

```text
Slice = [W, R]     ->  [W, V]
Time  = [S, V, U]  ->  [S, U]
```

리듀서와 intra-slice 체인을 어느 순서로든 결합한 예제는 [Vector Engine 예제](./index.md#examples)를 참고한다.

## 성능

리듀스 비율 `r`(리듀스 그룹 하나에 속한 슬라이스 개수)이 주된 조정 손잡이다.
inter-slice 리듀스의 지연은 `O(r)` 사이클로, 리듀스 그룹을 링으로 한 바퀴 도는 정도다.
전체 시간은 입력 스트리밍 시간에 그 링 크기만큼의 꼬리를 더한 값이다.

실제로는 상류 작업(부분합을 만드는 축약이나 `vector_inter_slice_reduce()` 이전의 intra-slice 작업)이 흔히 지배적이어서 이 링 꼬리를 가리므로, 리듀서는 병목이 아니다.
리듀서는 `r` 이 클 때(꼬리가 길어질 때), 그리고 고정된 꼬리를 여러 packet 에 걸쳐 상각하지 못하는 작은 텐서에서 드러난다.
