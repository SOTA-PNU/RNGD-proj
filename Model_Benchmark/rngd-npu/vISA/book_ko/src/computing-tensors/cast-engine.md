# Cast Engine

Cast Engine 은 [Commit Engine](../moving-tensors/commit-engine.md) 이 DM 에 쓰기 전에 파이프라인의 `f32`/`i32` 결과를 더 낮은 정밀도 타입(예: `bf16`)으로 좁혀 저장 비용을 줄인다.


## 인터페이스

`CollectTensor`, `ContractTensor`, `VectorFinalTensor` 는 모두 같은 의미의 `.cast()` 를 제공한다.

```rust,ignore
//
// The Cast Engine accepts only `VeScalar` inputs (hardware constraint), so the
// bound lives on the impl rather than on a wider trait.
impl<'l, const T: Tu, P: CanApplyCast, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Casts each element to type `OutD` and pads the output packet back to one
    /// 32-byte flit.
    #[primitive(TuTensor::cast)]
    pub fn cast<OutD: Scalar, OutPacket: M>(self) -> CastTensor<'l, T, OutD, Chip, Cluster, Slice, Time, OutPacket, B>
    where
        D: Cast<OutD>,
    {
        verify_cast::<D, OutD, Packet, OutPacket>();
        CastTensor::new(self.ctx, self.inner.map(|v| v.cast()).transpose(false))
    }
}
```

`.cast::<OutD, OutPacket>()` 는 각 원소를 `OutD` 타입으로 변환하고 출력을 다시 32바이트 flit 하나로 패딩한다.
커널 작성자가 `OutD`(목표 타입)와 `OutPacket`(출력 원소 레이아웃)을 고른다.
나머지는 컴파일러가 유도한다.

Cast Engine 은 [수학적 텐서 이동](../mapping-tensors/tensor-semantics.md#mathematical-tensor-move)이 아니지만, 텐서의 모양을 보존하고 원소 타입만 바꾼다.
모든 차원이 그대로 통과하며, 예외는 `Packet` 레이아웃으로 출력이 여전히 32바이트 flit 하나에 들어가도록 다시 패딩한다.

아래 예제는 8원소 `i32` packet(8 × 4 = 32바이트)을 `i8` 로 캐스팅한다.
캐스팅 후 8개 원소가 8바이트를 차지하므로, `A # 32` 가 출력을 다시 32바이트로 패딩한다:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![B = 4, A = 8];

fn cast_i32_to_i8<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![1 # 2], m![1 # 256], m![B], m![A]>,
) -> CastTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![B], m![A # 32]> {
    input.cast()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, i32, m![1], m![1 # 2], m![1 # 256], m![B], m![A]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = cast_i32_to_i8(c);
```

입력 데이터가 32바이트를 채우지 않을 수도 있다.
아래 예제는 데이터 원소 4개가 8로 패딩된 `i32` 입력(`A # 8`, 32바이트)을, 같은 원소 4개가 32로 패딩된 `i8` 출력(`A # 32`, 역시 32바이트)으로 캐스팅한다:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 4];

fn cast_padded<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![1 # 2], m![1 # 256], m![1], m![A # 8]>,
) -> CastTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![1], m![A # 32]> {
    input.cast()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, i32, m![1], m![1 # 2], m![1 # 256], m![1], m![A # 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = cast_padded(c);
```

## 지원하는 캐스팅

입력은 각각 32바이트 flit 이다.
지원하는 원본 타입은 `f32` 와 `i32` 이며, 각각 정해진 목표 타입이 있다:

| 입력 타입 (`D`) | 지원 출력 타입 (`OutD`) |
| --- | --- |
| `i32` | `i4`, `i8`, `i16` |
| `f32` | `f8e5m2`, `f8e4m3`, `f16`, `bf16` |


## 성능

Cast Engine 은 결코 파이프라인의 병목이 아니다. flit 에 유효 데이터가 얼마나 실려 있든 사이클당 flit 하나를 처리한다.
하류의 [Commit Engine](../moving-tensors/commit-engine.md) 이 활용도가 낮은 flit 들을 모아 조밀한 DM 쓰기로 만들므로 DM 대역폭이 낭비되지 않는다.
