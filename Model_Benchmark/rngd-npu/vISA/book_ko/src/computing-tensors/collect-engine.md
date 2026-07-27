# Collect Engine

하위 엔진 전부([Contraction Engine](./contraction-engine/index.md), [Vector Engine](./vector-engine/index.md), [Cast Engine](./cast-engine.md), [Transpose Engine](./transpose-engine.md), [Commit Engine](../moving-tensors/commit-engine.md))는 정확히 32 바이트짜리 *flit* 만 소비한다.
Collect Engine 은 임의 크기의 패킷을 두 단계로 flit 하나에 맞춰 정규화한다:
1. 입력 패킷을 다음 32 바이트 경계까지 **패딩**한다.
   패킷이 이미 32 바이트로 정렬되어 있으면 건너뛴다.
2. flit 경계에서 **분할**한다. 안쪽 32 바이트는 `Packet2` 가 되고, 바깥쪽 flit 개수는 `Time2` 로 흡수된다.
   패킷이 이미 32 바이트면 건너뛴다.

그 결과로 나온 `CollectTensor` 는 파이프라인을 따라 하위 엔진으로 흘러가거나 [Register Files](./register-files.md) 에 저장된다.

## 인터페이스

`SwitchTensor` 와 `FetchTensor` 는 둘 다 같은 의미의 `.collect()` 를 제공한다.
`FetchTensor` 진입점은 슬라이스 분배가 필요 없을 때 Switch Engine 을 건너뛴다.

```rust,ignore
impl<'l, const T: Tu, P: CanApplyCollect, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Normalizes packet to exactly 32 bytes (one flit).
    ///
    /// Pads to flit-aligned boundary, then splits: inner 32 bytes become
    /// `Packet2`, outer flit portion is absorbed into `Time2`. For packets
    /// already ≤ 32 bytes, only padding is added.
    #[primitive(TuTensor::collect)]
    pub fn collect<Time2: M, Packet2: M>(self) -> CollectTensor<'l, T, D, Chip, Cluster, Slice, Time2, Packet2, B> {
        verify_collect::<D, Time, Packet, Time2, Packet2>();
        CollectTensor::new(self.ctx, self.inner.transpose(false))
    }
}
```

## 예제

### 단일 flit 패킷

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, B = 32];

fn collect_identity<'l, const T: Tu>(
    input: SwitchTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![A], m![B]>,
) -> CollectTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![A], m![B # 32]> {
    // B=32 elements × 1 byte (i8) = 32 bytes = one flit.
    // Time and Packet pass through unchanged.
    input.collect()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: SwitchTensor<'_, _, i8, m![1], m![1 # 2], m![1 # 256], m![A], m![B]> = SwitchTensor::new(&mut ctx.main, Tensor::zero());
# let _o = collect_identity(c);
```

입력 패킷이 이미 정확히 32 바이트이면 `collect` 는 그대로 통과시킨다(`B = 32` 원소 × `i8` 1 바이트 = 32 바이트).

```text
Before:   Time = m![A]
          Packet = m![B]
          ┌──────────────────────────┐
          │            B             │  32 bytes
          └──────────────────────────┘

After:    Time = m![A]
          Packet = m![B # 32]
          ┌──────────────────────────┐
          │          B # 32          │  32 bytes
          └──────────────────────────┘
```

### flit 미만 패킷

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, B = 16];

fn collect_padding<'l, const T: Tu>(
    input: SwitchTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![A], m![B]>,
) -> CollectTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![A], m![B # 32]> {
    // B=16 elements × 1 byte = 16 bytes < 32 bytes.
    // Padded to 32 bytes: Packet2 = m![B # 32].
    // Time unchanged since it fits in one flit.
    input.collect()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: SwitchTensor<'_, _, i8, m![1], m![1 # 2], m![1 # 256], m![A], m![B]> = SwitchTensor::new(&mut ctx.main, Tensor::zero());
# let _o = collect_padding(c);
```

입력 패킷이 32 바이트보다 작으면 `collect` 는 32 바이트로 패딩한다(`B = 16` 원소 × `i8` 1 바이트 = 16 바이트).

```text
Before:   Time = m![A]
          Packet = m![B]
          ┌────────────┐
          │     B      │  16 bytes
          └────────────┘

After:    Time = m![A]
          Packet = m![B # 32]
          ┌────────────┬─────────────┐
          │     B      │     pad     │  32 bytes
          └────────────┴─────────────┘
```

### 다중 flit 패킷

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, B = 32];

fn collect_multi_flit<'l, const T: Tu>(
    input: SwitchTensor<'l, T, bf16, m![1], m![1 # 2], m![1 # 256], m![A], m![B]>,
) -> CollectTensor<'l, T, bf16, m![1], m![1 # 2], m![1 # 256], m![A, B / 16], m![B % 16]> {
    // B=32 elements × 2 bytes (bf16) = 64 bytes = 2 flits.
    // Inner 16 elements = 32 bytes → Packet2 = m![B % 16].
    // Outer 2 flits → absorbed into Time2 = m![A, B / 16].
    input.collect()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: SwitchTensor<'_, _, bf16, m![1], m![1 # 2], m![1 # 256], m![A], m![B]> = SwitchTensor::new(&mut ctx.main, Tensor::zero());
# let _o = collect_multi_flit(c);
```

입력 패킷이 32 바이트를 넘으면 `collect` 는 flit 단위로 쪼개고 바깥쪽 flit 개수를 Time 으로 흡수한다(`B = 32` 원소 × `bf16` 2 바이트 = 64 바이트이므로 `B / 16 = 2` flit).

```text
Before:   Time = m![A]
          Packet = m![B]
          ┌──────────────────────────┬──────────────────────────┐
          │       B / 16 == 0        │       B / 16 == 1        │  64 bytes
          └──────────────────────────┴──────────────────────────┘
                    32 bytes                   32 bytes

After:    Time = m![A, B / 16]
          Packet = m![B % 16]
          ┌──────────────────────────┐
          │          B % 16          │  32 bytes  × B/16 time steps
          └──────────────────────────┘
```

### 패딩이 있는 다중 flit 패킷

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, B = 56];

fn collect_multi_flit_padded<'l, const T: Tu>(
    input: SwitchTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![A], m![B]>,
) -> CollectTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![A, B # 64 / 32], m![B # 64 % 32]> {
    // B is not 32-byte aligned; first pad B to a multiple of 32 bytes.
    // B # 64=64 elements × 1 byte (i8) = 64 bytes = 2 flits.
    // Inner 32 elements = 32 bytes → Packet2 = m![B # 64 % 32].
    // Outer 2 flits → absorbed into Time2 = m![A, B # 64 / 32].
    input.collect()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: SwitchTensor<'_, _, i8, m![1], m![1 # 2], m![1 # 256], m![A], m![B]> = SwitchTensor::new(&mut ctx.main, Tensor::zero());
# let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| { collect_multi_flit_padded(c) }));
```

입력 패킷이 32 바이트에 정렬되어 있지 않으면 먼저 패딩된다(`B = 51` 원소 × `i8` 1 바이트 = 51 바이트, 64 로 패딩).
그런 다음 `collect` 는 flit 단위로 쪼개고 바깥쪽 flit 개수(`B # 64 / 32 = 2`)를 Time 으로 흡수한다.

```text
Before:   Time = m![A]
          Packet = m![B]
          ┌──────────────────────────┬───────────────┐
          │       B / 32 == 0        │  B / 32 == 1  │  51 bytes
          └──────────────────────────┴───────────────┘
                    32 bytes             19 bytes

Padded:   Time = m![A]
          Packet = m![B # 64]
          ┌──────────────────────────┬───────────────┬──────────┐
          │       B / 32 == 0        │  B / 32 == 1  │   pad    │  64 bytes
          └──────────────────────────┴───────────────┴──────────┘
                    32 bytes                   32 bytes

After:    Time = m![A, B # 64 / 32]
          Packet = m![B # 64 % 32]
          ┌──────────────────────────┐
          │       B # 64 % 32        │  32 bytes  × B # 64 / 32 time steps
          └──────────────────────────┘
```

## 레지스터 파일 적재

정규화가 끝나면 `CollectTensor` 를 [`.to_trf()`](#to-trf) 로 Tensor Register File 에, 또는 [`.to_vrf()`](#to-vrf) 로 Vector Register File 에 저장한다.
아래 "To TRF" / "To VRF" 절이 저장 방식(`time_inner` 도출, `[time_inner, Packet]` 을 `Element` 로 시퀀싱)을 설명한다.

<a id="to-trf"></a>
### To TRF

`.to_trf::<Row, Element>()` 는 TRF 를 행 차원을 따라 분할한다.
커널 작성자는 `Row` (TRF 안의 행 레이아웃으로, `Row::SIZE` 는 {1, 2, 4, 8} 중 하나)와 `Element` (행마다의 원소 레이아웃)를 고른다.
그러면 컴파일러는 `Time` 이 `[Row, time_inner]` 로 분해되고 `[time_inner, Packet]` 이 `Element` 로 시퀀싱되도록 하는 `time_inner` 을 찾아내며, 그 결과 TRF 의 각 행은 연속된 `time_inner` 개의 flit 로 채워진다.

`.to_trf()` 는 TRF 전체(`TrfAddress::Full`)를 쓴다. 두 텐서가 TRF 를 서로 독립적으로 차지하게 하려면, 영역을 고르는 `TrfAddress` 와 함께 `.to_trf_at::<Row, Element>(address)` 를 쓴다:
- `Full`: TRF 전체.
- `FirstHalf` / `SecondHalf`: TRF 를 반씩 둘로 나눈 것.

컴파일러는 결과 텐서의 전체 바이트 크기를 선택된 영역의 용량으로 제한한다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![B = 32];

fn load_trf<'l, const T: Tu>(
    input: CollectTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![1], m![B]>,
) -> TrfTensor<i8, m![1], m![1 # 2], m![1 # 256], m![1], m![B]> {
    input.to_trf()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, i8, m![1], m![1 # 2], m![1 # 256], m![1], m![B]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = load_trf(c);
```

<a id="to-vrf"></a>
### To VRF

`.to_vrf::<Element>()` 는 flit 들을 VRF 에 저장한다. `.to_vrf_at::<Element>(address)` 는 원시 `Address` 에 저장한다(영역을 한정해 고르는 기능은 없다).
커널 작성자는 VRF 안의 목적지 원소 레이아웃인 `Element` 를 고른다.
어떤 `Scalar` 원소 타입이든 받는 `.to_trf` 와 달리, `.to_vrf` 는 `VeScalar` 원소 타입(즉 `i32` 또는 `f32`)을 요구한다. 하위의 Vector Engine 이 이 타입들만 소비하기 때문이다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![B = 64];

fn load_vrf<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![1 # 2], m![1 # 256], m![B / 8], m![B % 8]>,
) -> VrfTensor<i32, m![1], m![1 # 2], m![1 # 256], m![B]> {
    input.to_vrf()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, i32, m![1], m![1 # 2], m![1 # 256], m![B / 8], m![B % 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = load_vrf(c);
```

