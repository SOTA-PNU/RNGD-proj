# Commit Adapter

Commit Adapter 는 [Commit Engine](../moving-tensors/commit-engine.md) 이 패킷 스트림을 DM 에 쓰기 전에 그 스트림에 원소 단위 변환을 적용한다.
Tensor Unit 의 출력 쪽에서 [Fetch Adapter](./fetch-adapter.md) 를 그대로 반영한다.

어댑터의 각 단계는 상류 텐서의 전용 `.commit_xxx(...)` 메서드로 이어 붙으며, 체인은 언제나 실제 DM 쓰기를 하는 `.commit(...)` 으로 끝난다. [절단(트리밍)](#trimming) 은 필수 첫 단계다: `.commit()` / `.commit_view()` 는 `.commit_trim(...)` 뒤에만 도달할 수 있으므로 모든 커밋은 먼저 절단된다(flit 패딩을 떨어뜨리는 방법이 이것이다). 나머지 단계는 드물게 쓰이며 절단 뒤에 이어진다. 그 다음 main 컨텍스트와 sub 컨텍스트가 갈라진다. 별도 연산인 [Generate Mode](#generate-mode) 는 개념상 Commit Adapter 의 일부이지만 홀로 선다(`TuTensor` 에서 체인으로 이어지지 않는다).

- Main 파이프라인: [절단(트리밍)](#trimming) → [타입 캐스팅](#type-casting) (선택적으로 ReLU 융합) → `.commit()`.
- Sub 파이프라인: [절단(트리밍)](#trimming) → [valid count 패킹](#valid-count-packing) → `.commit()`.
- Sub 우회: [Generate Mode](#generate-mode) 는 독립 API 로 32비트 상수 하나를 DM 에 직접 쓰며, Tensor Unit 파이프라인을 통째로 건너뛴다.

| 연산 | Main | Sub |
| --- | --- | --- |
| [절단(트리밍)](#trimming) | ✅ | ✅ |
| [타입 캐스팅](#type-casting) (선택적 ReLU 융합) | ✅ | ❌ |
| [valid count 패킹](#valid-count-packing) | ❌ | ✅ |
| [Generate Mode](#generate-mode) | ❌ | ✅ (UC, [§Generate Mode](#generate-mode) 참고) |

<a id="trimming"></a>
## 절단(트리밍)

Tensor Unit 파이프라인의 스트림 패킷은 언제나 32바이트 *flit* 이지만([Collect Engine](./collect-engine.md) 참고), flit 은 용량보다 적은 수의 유효 원소를 담을 수 있고 뒤쪽 원소는 패딩으로 채워진다.
flit 을 그대로 통째로 쓰면 유효 영역 너머의 DM 바이트가 flit 의 패딩 값으로 덮어써진다.

트리밍은 각 flit 의 앞쪽 `valid_size` 개 원소만 DM 에 쓰고 뒤쪽 패딩을 버려서 이 문제를 푼다.
컴파일러가 출력 텐서 매핑에서 `valid_size` 를 유도한다.
사용자가 직접 설정하지 않는다.
`D[valid_size]` 는 8, 16, 24, 32 바이트 중 하나여야 한다(32 는 절단 없음을 뜻한다).
트리밍은 지연을 거의 더하지 않는다.

모든 커밋이 떨어뜨릴 패딩을 가진 것은 아니지만 트리밍은 Commit Adapter 의 필수 첫 단계다: `valid_size` 가 이미 32바이트이면 flit 은 전부 유효하고 절단은 무연산이 된다.
`.commit()` 이 `.commit_trim(...)` 뒤에만 도달할 수 있기에 필수이며, 그래서 체인을 고정하고 [타입 캐스팅](#type-casting) (main) 과 [valid count 패킹](#valid-count-packing) (sub) 보다 앞서 실행된다.

```rust,ignore
// `D: MaterializableScalar` here (trim is the commit path's mandatory first stage) keeps i5/i9 uncommittable.
impl<
    'l,
    const T: Tu,
    P: CanApplyCommitTrim,
    D: MaterializableScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Commit Adapter's trimming stage.
    ///
    /// Drops the trailing padding from each flit so DM stores only valid
    /// elements. `OutPacket` is the post-trim layout the kernel
    /// promises; the compiler derives the trim count from the input and
    /// output mappings.
    #[primitive(TuTensor::commit_trim)]
    pub fn commit_trim<OutPacket: M>(self) -> CommitTrimTensor<'l, T, D, Chip, Cluster, Slice, Time, OutPacket, B> {
        verify_commit_trim::<D, Packet, OutPacket>();
        // `transpose(false)` is type-system filler; real trim lowering lands with the backend wiring.
        CommitTrimTensor::new(self.ctx, self.inner.transpose(false))
    }
}
```

`.commit_trim::<OutPacket>()` 은 절단 후의 패킷을 선언하고, 이어지는 `.commit(...)` 이 절단된 스트림에 대해 DM 쓰기를 수행한다. 둘은 완전히 분리되어 있다.

```rust,ignore
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![M = 4, K = 2, W = 8, N = 16, J = 64];

fn commit_trim_i8_padding<'l, const T: Tu>(
    input: CastTensor<'l, T, i8, m![1], m![1], m![1], m![M, K], m![W # 32]>,
) -> CommitTrimTensor<'l, T, i8, m![1], m![1], m![1], m![M, K], m![W]> {
    // 8 valid i8 out of 32 padded; OutPacket drops the `# 32` padding.
    input.commit_trim::<m![W]>()
}

fn commit_trim_f32_non_padding<'l, const T: Tu>(
    input: ContractTensor<'l, T, f32, m![1], m![1], m![1], m![M, K], m![W]>,
) -> CommitTrimTensor<'l, T, f32, m![1], m![1], m![1], m![M, K], m![W = 4]> {
    // 4 valid f32 out of 8; OutPacket resizes `W` to 4.
    input.commit_trim::<m![W = 4]>()
}

fn commit_trim_bf16_with_transpose<'l, const T: Tu>(
    input: CastTensor<'l, T, bf16, m![1], m![1], m![1], m![M, K], m![N]>,
) -> CommitTrimTensor<'l, T, bf16, m![1], m![1], m![1], m![M, K], m![N = 8]> {
    // 8 valid bf16 out of 16; OutPacket resizes `N` to 8.
    input.commit_trim::<m![N = 8]>()
}

fn commit_trim_i4_no_trim<'l, const T: Tu>(
    input: CastTensor<'l, T, i4, m![1], m![1], m![1], m![M, K], m![J]>,
) -> CommitTrimTensor<'l, T, i4, m![1], m![1], m![1], m![M, K], m![J]> {
    // No trimming; `OutPacket == Packet`.
    input.commit_trim::<m![J]>()
}
#
# let mut ctx = Context::acquire();
# let a: CastTensor<'_, _, i8, m![1], m![1], m![1], m![M, K], m![W # 32]> = CastTensor::new(&mut ctx.main, Tensor::zero());
# let _o = commit_trim_i8_padding(a);
# let b: ContractTensor<'_, _, f32, m![1], m![1], m![1], m![M, K], m![W]> = ContractTensor::new(&mut ctx.main, Tensor::zero());
# let _o = commit_trim_f32_non_padding(b);
# let c: CastTensor<'_, _, bf16, m![1], m![1], m![1], m![M, K], m![N]> = CastTensor::new(&mut ctx.main, Tensor::zero());
# let _o = commit_trim_bf16_with_transpose(c);
# let d: CastTensor<'_, _, i4, m![1], m![1], m![1], m![M, K], m![J]> = CastTensor::new(&mut ctx.main, Tensor::zero());
# let _o = commit_trim_i4_no_trim(d);
```

<a id="type-casting"></a>
## 타입 캐스팅

타입 캐스팅은 커밋 경로에서 `f32` 데이터를 `bf16` 형식으로 변환하며, 선택적으로 ReLU 활성함수를 같은 패스에 융합한다.
Tensor Unit 파이프라인의 타입 변환 대부분은 [Cast Engine](./cast-engine.md) 이 처리한다.
Commit Adapter 의 타입 캐스팅은 한 가지 특정 경우, 즉 main 컨텍스트의 축약을 sub 컨텍스트의 Vector Engine 작업과 병렬로 돌리기 위해 존재한다.
Cast Engine 은 Vector Engine 위에 얹혀 있어서 변환하는 동안 Vector Engine 을 점유한다.
main 컨텍스트가 `f32` → `bf16` 변환을 Cast Engine 으로 수행하면 Vector Engine 이 바빠져 sub 컨텍스트가 병렬로 돌 수 없다.
대신 변환을 Commit Adapter 로 보내면 Vector Engine 이 sub 컨텍스트를 위해 비어 있게 된다.
sub 컨텍스트 자체는 타입 캐스팅을 지원하지 않는다(위 지원 표와 일치한다).

`commit_cast` 는 `Activation` 을 받는다. `Activation::None` 은 단순 캐스트다. `Activation::Relu` 는 같은 캐스트의 일부로 음수 값을 0 으로 클램프한다. ReLU 는 독립된 하드웨어 단계가 없고, 좁히는 캐스트(`f32` → `bf16` + ReLU)에 융합된 형태로만 존재한다.


```rust,ignore
impl<
    'l,
    const T: Tu,
    P: CanApplyCommitCast,
    D: MaterializableScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Commit Adapter's type-casting stage, optionally fusing a
    /// ReLU.
    ///
    /// Folds an `f32` → `bf16` (or other narrowing) cast into the commit
    /// path, leaving the [Cast Engine](crate::engine::cast) free for
    /// sub-context Vector Engine work. `activation` selects the optional
    /// fused ReLU; ReLU has no standalone hardware stage.
    #[primitive(TuTensor::commit_cast)]
    pub fn commit_cast<OutD: Scalar>(
        self,
        _activation: Activation,
    ) -> CommitCastTensor<'l, T, OutD, Chip, Cluster, Slice, Time, Packet, B>
    where
        D: Cast<OutD>,
    {
        verify_commit_cast::<D, OutD>();
        CommitCastTensor::new(self.ctx, self.inner.map(|v| v.cast()))
    }
}
```

```rust,ignore
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![N = 4, C = 3, H = 4, W = 8];

fn commit_cast_example<'l, const T: Tu>(
    input: ContractTensor<'l, T, f32, m![1], m![1], m![1], m![N, C, H], m![W]>,
) -> CommitCastTensor<'l, T, bf16, m![1], m![1], m![1], m![N, C, H], m![W]> {
    // Cast f32 to bf16 (values preserved), no activation. A real main
    // commit runs `.commit_trim()` first, then `.commit(...)` after.
    // W = 8 f32 elements (32 bytes) stays 8 bf16 elements (16 bytes).
    input.commit_cast::<bf16>(Activation::None)
}

fn commit_cast_relu_example<'l, const T: Tu>(
    input: ContractTensor<'l, T, f32, m![1], m![1], m![1], m![N, C, H], m![W]>,
) -> CommitCastTensor<'l, T, bf16, m![1], m![1], m![1], m![N, C, H], m![W]> {
    // f32 -> bf16 with a fused ReLU: negative values clamped to zero.
    // e.g. [-5.0, -0.1, 0.0, 3.7] -> [0.0, 0.0, 0.0, 3.7]
    input.commit_cast::<bf16>(Activation::Relu)
}
```

<a id="valid-count-packing"></a>
## valid count 패킹

valid count 패킹은 sub 컨텍스트 전용 단계로, 패킷마다 가변 개수의 유효 원소를 커밋하고 출력에서 패딩을 제외한다.


```rust,ignore
impl<
    'l,
    const T: Tu,
    P: CanApplyCommitValidCountPack,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Commit Adapter's valid-count-packing stage (sub-context
    /// only). The count comes from a per-call argument; the trailing
    /// elements are discarded. The packed stream keeps the input
    /// `Time` / `Packet` shape at this skeleton stage.
    // TODO: `_valid_count` is currently discarded. The backend
    // `TuOperationCommitValidCountPack` record does not store it yet.
    #[primitive(TuTensor::commit_valid_count_pack)]
    pub fn commit_valid_count_pack(
        self,
        _valid_count: usize,
    ) -> CommitValidCountPackTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B> {
        verify_commit_valid_count_pack::<D, Time, Packet>();
        CommitValidCountPackTensor::new(self.ctx, self.inner.transpose(false))
    }
}
```

<a id="generate-mode"></a>
## Generate Mode

Generate Mode 는 sub 컨텍스트 전용 ITOS(immediate-to-SRAM) 쓰기에 쓰인다. Sequencer 가 하드웨어에 상수 `u32` 값과 sub 컨텍스트에서 유도한 DM 주소를 넘기고, 런타임이 그 주소에 상수를 직접 쓴다.

DM 에서 페치하지 않으며 상류 Tensor Unit 스트림도 소비하지 않는다.
상수 `value` 와 목적지가 유일한 입력이고, 나머지 Tensor Unit 파이프라인(Fetch / Switch / Collect / Contraction / Vector / Cast / Transpose / Commit Adapter 단계)은 통째로 우회된다.

