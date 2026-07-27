# Time Reducer

Time Reducer 는 [Packet Reducer](./packet-reducer.md) 의 `[Lane, Packet]` 출력을 *시간 누산기*로 `Time` 에 걸쳐 누적해 `OutTime` 으로 만든다.

## 인터페이스

`.contract_time::<OutTime>()` 이 Time Reducer 를 호출한다.
`OutTime` 은 살아남는 `Time` 차원을 지정한다(나머지는 합해져 사라진다).

```rust,ignore
impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Time: M, Packet: M, B: Backend>
    ContractPacketTensor<'l, T, D, Chip, Cluster, Slice, Lane, Time, Packet, B>
{
    /// Accumulates per-cycle contractions over the `Time` dimension via the shared
    /// accumulator buffer, shrinking input `Time` to `OutTime`. The axes present in
    /// `Time` but absent from `OutTime` are reduce-added.
    #[primitive(ContractPacketTensor::contract_time)]
    pub fn contract_time<OutTime: M>(
        self,
    ) -> ContractTimeTensor<'l, T, D, Chip, Cluster, Slice, Lane, OutTime, Packet, B> {
        verify_contract_time(Time::to_value(), OutTime::to_value());
        // Carry the deferred operands forward unreduced: the fused contraction at `contract_lane`
        // performs this Time reduction too. This stage only re-types the carrier to `OutTime`.
        ContractTimeTensor::new(self.ctx, self.inner, Time::to_value())
    }
}
```

예를 들어 아래 커널은 2D 텐서를 `B` 를 따라 축약(리듀스)한다(`A` 만 살아남는다).

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 2048, B = 32];

/// Reduces along B; A survives.
fn reduce_b<'l, const T: Tu>(
    // Streaming operand: Slice = m![A / 8] (256 outer A chunks across slices).
    // Time = m![B / 16, A % 8]; Packet = m![B % 16].
    // B splits across Packet (B % 16) and Time (B / 16): each cycle produces a partial sum.
    input: CollectTensor<'l, T, bf16, m![1], m![1 # 2], m![A / 8], m![B / 16, A % 8], m![B % 16]>,
    // TRF operand: single-lane weight per slice.
    trf: &TrfTensor<bf16, m![1], m![1 # 2], m![A / 8], m![1], m![B]>,
    // Output: one f32 per (slice, A % 8) cell.
) -> ContractTensor<'l, T, f32, m![1], m![1 # 2], m![A / 8], m![A % 8], m![1 # 8]> {
    input
         // Outer: Lane = m![1], OutTime = m![B / 16, A % 8], OutPacket = m![B % 16].
         .contract_outer::<m![B / 16, A % 8], m![B % 16], _, _, _>(trf)
         // Packet Reducer: OutPacket = m![1]. Collapses B % 16 spatially.
         .contract_packet::<m![1]>()
         // Time Reducer: OutTime = m![A % 8]. Accumulator receives
         // Time::SIZE = (B / 16) × (A % 8) = 2 × 8 = 16 flits; B / 16 outer
         // chunks accumulate into 8 slots indexed by A % 8.
         .contract_time::<m![A % 8]>()
         // Lane Folder: Lane folds into OutPacket. Sequential mode (Lane = m![1]).
         .contract_lane::<m![A % 8], m![1 # 8]>(LaneMode::Sequential)
}
# 
# let mut ctx = Context::acquire();
# 
# let a: CollectTensor<'_, _, bf16, m![1], m![1 # 2], m![A / 8], m![B / 16, A % 8], m![B % 16]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let b: TrfTensor<bf16, m![1], m![1 # 2], m![A / 8], m![1], m![B]> = unsafe { TrfTensor::from_addr(TrfAddress::Full) };
# let _o = reduce_b(a, &b);
```

## 구조

Time Reducer 는 Packet Reducer 의 사이클별 `[Lane, Packet]` 출력을 받는다.
하드웨어는 상류에서 `Lane::SIZE ≤ 8`(공간적으로 병렬인 레인)과 `Packet::SIZE ≤ 32` 로 상한을 둔다.

매 사이클 Time Reducer 는 `[Lane, Packet]` 공간 격자를 `Time` 에 걸쳐 접어 `OutTime` 으로 만든다.
`OutTime` 은 `Time` 의 부분집합이어야 하며 살아남는 차원들의 상대 순서가 보존되어야 한다(`verify_contract_time` 이 강제한다).
`Time` 에 있으나 `OutTime` 에 없는 차원은 합해져 사라지고, 그중 가장 바깥 차원이 flit 을 따라 반복한다.

`InnerTime` 을 `Time` 의 안쪽 비-리듀스 차원(가장 바깥 리듀스 차원보다 안쪽에 있으면서 `OutTime` 에 살아남는 차원)이라 하자.
위 `reduce_b` 에서는 `Time = m![B / 4, A % 8]` 이고 `OutTime = m![A % 8]` 이므로 `B / 4` 가 가장 바깥 리듀스 차원(`Time::SIZE = 2 × 8 = 16` 개 flit 을 따라 반복)이고 `InnerTime = m![A % 8]` 이다.

누적에는 `InnerTime` 튜플 값마다 하나씩, `[Lane, Packet]` 슬롯이 `InnerTime::SIZE` 개 필요하다. 같은 튜플을 가진 flit 은 같은 슬롯에 누적된다.
`reduce_b` 에서는 `B / 4 = 2` 번의 반복에 걸쳐 8개 슬롯이 누적되고, flit 15 이후 버퍼는 최종 리듀스 결과를 담아 [Lane Folder](./lane-folder.md) 로 넘긴다:

```text
Time = m![B / 4, A % 8]
          ~~~~~  ~~~~~
        outer R  inner non-R (A % 8)

Flit sequence (B / 4 has values 0,1; A % 8 has values 0..7):

  flit #0:  B/4=0, A%8=0  ──→ ┌─────────────────┐
  flit #8:  B/4=1, A%8=0  ──→ │ slot 0 (A%8=0)  │  accumulates B for A%8=0
                              └─────────────────┘

  flit #1:  B/4=0, A%8=1  ──→ ┌─────────────────┐
  flit #9:  B/4=1, A%8=1  ──→ │ slot 1 (A%8=1)  │  accumulates B for A%8=1
                              └─────────────────┘
   ⋮

  flit #7:  B/4=0, A%8=7  ──→ ┌─────────────────┐
  flit #15: B/4=1, A%8=7  ──→ │ slot 7 (A%8=7)  │  accumulates B for A%8=7
                              └─────────────────┘

  8 non-reduce positions → 8 slots used
```

<a id="constraints"></a>
## 제약

`InnerTime::SIZE` 가 슬롯 용량(버퍼가 담는 슬롯 개수)을 넘지 않으면 매핑이 버퍼에 들어맞는다.

슬롯 용량은 버퍼의 셀 1,024개와 하류 [Lane Folder](./lane-folder.md) 의 `LaneMode` 에서 따라 나온다.
각 슬롯은 `LaneMode` 가 모양을 정하는 `[Lane, Packet]` 청크이므로, 슬롯 개수는 1,024 를 청크당 셀 수로 나눈 값이다:

| `LaneMode` | 청크 모양 | 청크당 셀 수 | 슬롯 용량 |
|------------|----------------------|--------------------|----------------------|
| `Interleaved` | `[Lane # 8, Packet]` | `8 × Packet::SIZE` | `128 / Packet::SIZE` |
| `Sequential` | `[Lane, Packet # 32]` | `Lane::SIZE × 32` | `32 / Lane::SIZE` |

`reduce_b` 는 하류에서 `Lane::SIZE = 1` 로 `.contract_lane(LaneMode::Sequential)` 을 쓰므로 슬롯 용량이 32 이고 `InnerTime::SIZE = 8` 은 여유 있게 들어간다.
`InnerTime::SIZE` 가 슬롯 용량을 넘는다면 `Time` 을 재구성하거나(예: `B` 를 더 쪼갠다) `LaneMode` 를 바꿔 처리량과 슬롯 여유를 맞바꾼다.

## 성능

처리량은 입력 쪽에서 사이클당 packet 하나다.
입력 `N` 개를 출력 하나로 리듀스하고 나면 실효 출력률은 입력의 `1 / N` 이다.

크기 `N` 인 `Time` 차원 리듀스의 지연은 대략 `N` 사이클이다.
