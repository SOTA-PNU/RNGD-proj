# Vector Engine

Vector Engine 은 원소 단위 연산과 축약(리듀스)을 수행한다.
활성함수(GELU, SiLU), 정규화(softmax, layer norm), 이항 연산, intra-slice 및 inter-slice 리듀스가 그 예다.

이 엔진은 32비트 타입 `i32` 와 `f32` 만 받는다.
상류의 [Contraction Engine](../contraction-engine/index.md) 이 타입을 자동으로 넓힌다(`bf16` 곱은 `f32` 로, `i8` 곱은 `i32` 로 누적된다).
그 엔진을 우회하는 경우, [Fetch Engine](../../moving-tensors/fetch-engine.md#type-casting) 이 타입 캐스팅 어댑터로 입력을 넓혀야 한다.

<a id="interface"></a>
## 인터페이스

Tensor Unit 한 번의 호출에서 Vector Engine 부분은 `vector_init()` 부터 `vector_final()` 까지 이어지는 메서드 체인이다.
이 엔진은 두 하위 조각으로 이루어진다. [Intra-Slice Chain](./intra-slice-chain.md)(원소 단위 / 이항 / 슬라이스별 리듀스 단계)과 [Inter-Slice Reducer](./inter-slice-reducer.md)(클러스터 안 256개 슬라이스에 걸쳐 리듀스)다.
`vector_init()` 과 `vector_final()` 사이에서는 체인만, 리듀서만, 또는 둘 다 실행할 수 있다.
둘 다 실행할 때 순서는 `IntraFirst`(체인 다음 리듀서) 또는 `InterFirst`(리듀서 다음 체인)이다.

intra-slice 체인에는 `vector_intra_slice_tag()` 로 진입하고, 입력이 두 병렬 스트림으로 나눌 2-way 그룹핑 축을 실어 오는 경우에는 `vector_intra_slice_unzip()` 으로 진입한다([Pair Mode](./intra-slice-chain.md#pair-mode) 참고).
두 진입점 모두 `vector_init()` 직후에, 또는 inter-slice 리듀서의 출력에서 발동한다.
inter-slice 리듀서에는 `vector_inter_slice_reduce()` 로 진입하며, `vector_init()` 직후이거나 [호환되는 intra-slice 단계](./intra-slice-chain.md#transitioning-to-the-inter-slice-reducer)에서 들어간다.
단계별 API 는 [Intra-Slice Chain](./intra-slice-chain.md) 과 [Inter-Slice Reducer](./inter-slice-reducer.md) 를 참고한다.

아래 시그니처는 `vector_init()` 쪽 진입 메서드만 다룬다.
같은 이름의 메서드(`vector_intra_slice_tag`, `vector_inter_slice_reduce`)가 체인↔리듀서 전이를 위해 체인 텐서와 리듀서 텐서에도 존재하며, 그것들은 하위 페이지에서 다룬다.

```rust,ignore
impl<'l, const T: Tu, P: CanApplyVectorInit, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet>
{
    /// Initializes Vector Engine processing for this tensor.
    #[primitive(TuTensor::vector_init)]
    pub fn vector_init(self) -> VectorInitTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet> {
        VectorInitTensor::new(self.ctx, self.inner)
    }
}

impl<'l, const T: Tu, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorInitTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet>
{
    /// Enters VE intra-slice pipeline (single stream).
    #[primitive(VectorInitTensor::vector_intra_slice_tag)]
    pub fn vector_intra_slice_tag(
        self,
        branch: TagMode,
    ) -> VectorBranchTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, D, Fresh, { VeOrder::IntraFirst }> {
        VectorBranchTensor::new(self.ctx, self.inner, branch)
    }

    /// Enters VE intra-slice pipeline (two-group / unzip).
    #[primitive(VectorInitTensor::vector_intra_slice_unzip)]
    pub fn vector_intra_slice_unzip<I: AxisName, TileTime: M, SplitTime: M>(
        self,
    ) -> VectorTensorPair<'l, T, D, stage::Tag, Chip, Cluster, Slice, SplitTime, Packet> {
        VectorTensorPair::new::<I, Time, TileTime>(self.ctx, self.inner)
    }
}

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

<a id="examples"></a>
## 예제

Vector Engine 호출이 어떤 모습인지 감을 잡도록 대표 예제 몇 개를 싣고, 전체 API 안내는 하위 페이지 [Intra-Slice Chain](./intra-slice-chain.md) 과 [Inter-Slice Reducer](./inter-slice-reducer.md) 로 미룬다.

### ReLU 활성함수

이 패스는 ReLU 를 원소 단위로 적용하여 \\(output[b, k, m, n] = \max(input[b, k, m, n], 0)\\) 을 계산한다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![B = 2, K = 256, M = 16, N = 16];

// ReLU activation after batched matrix multiplication.
// Chain-only pass, so the reducer is skipped and the path trivially resolves to IntraFirst.
// Both clusters (B = 2) and all 256 slices (K) carry real data, no padding.
fn relu<'l, const T: Tu>(
    input: ContractTensor<'l, T, f32, m![1], m![B], m![K], m![M, N / 8], m![N % 8]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![B], m![K], m![M, N / 8], m![N % 8]> {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // max(x, 0), the ReLU itself
        .vector_clip(ClipBinaryOpF32::Max, 0.0f32)
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: ContractTensor<'_, _, f32, m![1], m![B], m![K], m![M, N / 8], m![N % 8]> = ContractTensor::new(&mut ctx.main, Tensor::zero());
# let _o = relu(c);
```

### ReLU 후 리듀스

이 패스는 슬라이스마다 ReLU 를 적용한 뒤 `R` 에 걸쳐 리듀스하여 \\(output[a, b] = \sum_{r \in R} \max(input[a, b, r], 0)\\) 을 얻는다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2, R = 4];

// Chain applies ReLU, then reducer sums across slices.
// IntraFirst shape (chain runs first, then reducer).
// Both clusters (B = 2), all 256 slices (A / 8 * R), and full Way8 packet (A % 8) carry real data.
fn relu_then_reduce<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8, R], m![1], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8, 1 # 4], m![1], m![A % 8]> {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // max(x, 0), the ReLU
        .vector_clip(ClipBinaryOpI32::Max, 0)
        // sum across R slices
        .vector_inter_slice_reduce::<m![A / 8, 1 # 4], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, i32, m![1], m![B], m![A / 8, R], m![1], m![A % 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = relu_then_reduce(c);
```

### 리듀스 후 바이어스

이 패스는 `R` 에 걸쳐 리듀스한 뒤 상수 바이어스를 더하여 \\(output[a, b] = \left(\sum_{r \in R} input[a, b, r]\right) + 100\\) 을 얻는다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2, R = 4];

// Reducer sums across slices, then chain adds a bias to the reduced result.
// InterFirst shape (reducer runs first, then chain).
// Both clusters (B = 2), all 256 slices, and full Way8 packet carry real data.
fn reduce_then_add<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8, R], m![1], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8, 1 # 4], m![1], m![A % 8]> {
    input
        .vector_init()
        // sum across R slices
        .vector_inter_slice_reduce::<m![A / 8, 1 # 4], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_intra_slice_tag(TagMode::Zero)
        // add bias 100
        .vector_fxp(FxpBinaryOp::AddFxp, 100)
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, i32, m![1], m![B], m![A / 8, R], m![1], m![A % 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = reduce_then_add(c);
```

### Intra-Slice 및 Inter-Slice 리듀스

이 패스는 intra-slice 리듀서(`R` 의 Time·Packet 부분)와 inter-slice 리듀서(`R` 의 Slice 부분)를 결합하여 `R` 을 완전히 리듀스한다.
einsum 형태는 `BR -> B` 이고, 포화 덧셈을 쓴다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![B = 2, R = 8192];

// R splits across Slice (R / 32 = 256), Time (R % 32 / 4 = 8), and Packet (R % 4, padded to 8 in Way8).
// Chain runs intra-slice reduce over R's Time and Packet portions, then the reducer collapses the Slice portion.
// IntraFirst shape (chain runs first, then reducer).
fn full_sum<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![R / 32], m![R % 32 / 4], m![R % 4 # 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![1 # 256], m![1], m![1 # 8]> {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // Way8 → Way4 (back 4 packet positions were padding)
        .vector_narrow_trim::<m![R % 4]>()
        // sum over R's Time and Packet portions
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(IntraSliceReduceOpI32::AddSat)
        // Way4 → Way8
        .vector_widen_pad::<m![1 # 8]>()
        // sum over R's Slice portion across all 256 slices
        .vector_inter_slice_reduce::<m![1 # 256], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, i32, m![1], m![B], m![R / 32], m![R % 32 / 4], m![R % 4 # 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = full_sum(c);
```

### Pair Add

이 패스는 `I` 를 따라 교차된 두 그룹을 언집(unzip)하여 쌍 단위로 더한다.
einsum 형태는 `ABI -> AB` 이다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 2048, B = 2, I = 2];

// Pair-mode entry via unzip, then a zip op fuses the two streams with an add.
// Both clusters (B = 2), all 256 slices (A / 8), and full Way8 packet (A % 8) carry real data.
fn pair_add<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8], m![I], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]> {
    input
        .vector_init()
        // split into group 0 and group 1 along I
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        // group0 + group1
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, i32, m![1], m![B], m![A / 8], m![I], m![A % 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = pair_add(c);
```
