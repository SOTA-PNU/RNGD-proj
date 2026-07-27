# Lane Folder

Lane Folder 는 Contraction Engine 의 마지막 단계다.
`Lane` 의 값 8개를 `OutPacket`(Interleaved) 또는 `OutTime`(Sequential) 으로 옮겨 `Lane` 차원을 없앤다.
값을 합하지는 않는다: 이 단계는 `Lane` 을 리듀스하는 것이 아니라 다른 축으로 접는다.

## 인터페이스

`.contract_lane(mode)` 가 Lane Folder 를 호출한다.
이 단계는 상류 Time Reducer 의 버퍼를 8원소 폭 출력 버스로 한 사이클에 하나씩 비워내며, `LaneMode` 가 각 사이클의 flit 이 무엇을 싣는지 고른다.

```rust,ignore
impl<
    'l,
    const T: Tu,
    D: ContractionCast + MaterializableScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Lane: M,
    Time: M,
    Packet: M,
    B: Backend,
> ContractTimeTensor<'l, T, D, Chip, Cluster, Slice, Lane, Time, Packet, B>
{
    /// Folds the `Lane` dimension into the output stream.
    /// `LaneMode::Interleaved` relocates `Lane` into `OutPacket`;
    /// `LaneMode::Sequential` relocates `Lane` into `OutTime`.
    #[primitive(ContractTimeTensor::contract_lane)]
    pub fn contract_lane<OutTime: M, OutPacket: M>(
        self,
        mode: LaneMode,
    ) -> ContractTensor<'l, T, D, Chip, Cluster, Slice, OutTime, OutPacket, B> {
        verify_contract_lane(
            Lane::to_value(),
            Time::to_value(),
            Packet::to_value(),
            OutTime::to_value(),
            OutPacket::to_value(),
            self.pre_reduce_time,
            mode,
        );
        // Finalize the carried operands with ONE fused contraction onto this stage's input mapping
        // `[Chip, Cluster, Slice, Lane, Time, Packet]` (Packet/Time were never actually reduced by the
        // earlier stages, only relabeled to their post-stage extents). The Lane fold relayout
        // (`transpose(false)`) then runs on the result, exactly as it always has.
        //
        // `out` is rebuilt here from this stage's own type params, independently of the `pre_reduce`
        // stashed by `contract_outer`; `Backend::contraction` reduces `pre_reduce` onto `out` via
        // `pre_reduce.carve(out)`, the same mapping-algebra carve `reduce` uses elsewhere, which is the
        // authority on whether `out` is a valid restriction of `pre_reduce` -- NOT a manual re-check here.
        // A naive per-symbol `.axes()` comparison is unsound for that: a contracted symbol can split
        // across a spatial slot this fold never touches (e.g. `Cluster`, carrying a `K`-fragment that
        // survives to `out` unreduced) and the slots that actually get contracted (`Time`/`Packet`,
        // carrying the rest of `K`); `pre_reduce`'s canonical `K` term then legitimately has a different
        // shape (wider modulo) than `out`'s, even though `out` is a correct restriction of `pre_reduce`.
        let contraction = self.inner;
        let out = <m![{ Chip }, { Cluster }, { Slice }, { Lane }, { Time }, { Packet }]>::to_value();
        let reduced: Tensor<D, m![{ Chip }, { Cluster }, { Slice }, { Lane }, { Time }, { Packet }], B> =
            Tensor::from_inner(B::contraction(
                &contraction.lhs,
                &contraction.rhs,
                &contraction.lhs_map,
                &contraction.rhs_map,
                &contraction.pre_reduce,
                &out,
            ));
        ContractTensor::new(self.ctx, reduced.transpose(false))
    }
}
```

아래 최소 예제들은 `ContractTimeTensor`(상류 Time Reducer 의 출력)를 받아 `.contract_lane(...)` 만 호출하므로, 각 예제는 Lane Folder 를 따로 떼어 보여준다.
입력 `Packet` 은 [Packet Reducer](./packet-reducer.md) 를 거쳐 살아남은 크기를 지니며, 레인당 `{1, 2, 4, 8, 16, 32}` 원소 중 하나다.

### Interleaved

`Lane` 차원이 `OutPacket` 으로 접힌다: 매 사이클 8개 레인 전체에서 열 위치 하나를 읽고(레인당 값 하나, flit 당 값 8개), `Lane` 은 가장 안쪽 `OutPacket` 으로 구체화된다.

```text
OutTime   = [Time, Packet]
OutPacket = [Lane # 8]
```

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![N = 8, M = 4, P = 16];

/// Lane folds into OutPacket.
fn lane_interleaved<'l, const T: Tu>(
    // Input from upstream Time Reducer: Lane = m![N], Time = m![M], Packet = m![P].
    input: ContractTimeTensor<'l, T, f32, m![1], m![1 # 2], m![1 # 256], m![N], m![M], m![P]>,
    // Output: OutTime = m![M, P] = [Time, Packet], OutPacket = m![N] = [Lane].
) -> ContractTensor<'l, T, f32, m![1], m![1 # 2], m![1 # 256], m![M, P], m![N]> {
    input.contract_lane::<m![M, P], m![N]>(LaneMode::Interleaved)
}
# 
# let mut ctx = Context::acquire();
# 
# let a: CollectTensor<'_, _, bf16, m![1], m![1 # 2], m![1 # 256], m![M], m![P]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let b: TrfTensor<bf16, m![1], m![1 # 2], m![1 # 256], m![N], m![P]> = unsafe { TrfTensor::from_addr(TrfAddress::Full) };
# 
# let i: ContractTimeTensor<'_, _, f32, m![1], m![1 # 2], m![1 # 256], m![N], m![M], m![P]> = a 
#     .contract_outer::<m![M], m![P], m![N], m![P], _>(&b)
#     .contract_packet::<m![P]>()
#     .contract_time::<m![M]>();
# 
# let _o = lane_interleaved(i);
```

<a id="sequential"></a>
### Sequential

`Lane` 차원이 `OutTime` 으로 접힌다: 매 사이클 한 레인의 `Packet` 에서 열 위치 8개를 읽고(flit 당 값 8개), `Lane` 은 연속된 사이클을 따라 반복한다.
각 사이클이 8원소 폭이므로 `Packet` 은 먼저 8(버스 폭)의 배수까지 패딩된 뒤, 레인당 `PadPacket / 8` 사이클과 사이클당 `PadPacket % 8` 원소로 쪼개진다.

```text
PadPacket = Packet # align_up(Packet::SIZE, 8)   (pad Packet up to the next multiple of 8)
OutTime   = [Time, Lane, PadPacket / 8]
OutPacket = [PadPacket % 8]
```

`Packet::SIZE < 32` 일 때 `[PadPacket / 8]::SIZE = ceil(Packet::SIZE / 8)` 은 packet 당 사이클 수다(예: `Packet::SIZE = 4` 면 1 사이클, `Packet::SIZE = 16` 이면 2 사이클).

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![N = 8, M = 4, P = 16];

/// Lane folds into OutTime.
fn lane_sequential<'l, const T: Tu>(
    // Input from upstream Time Reducer: Lane = m![N], Time = m![M], Packet = m![P].
    input: ContractTimeTensor<'l, T, f32, m![1], m![1 # 2], m![1 # 256], m![N], m![M], m![P]>,
    // Output: OutTime = m![M, N, P / 8] = [Time, Lane, Packet / 8], OutPacket = m![P % 8] = [Packet % 8].
) -> ContractTensor<'l, T, f32, m![1], m![1 # 2], m![1 # 256], m![M, N, P / 8], m![P % 8]> {
    input.contract_lane::<m![M, N, P / 8], m![P % 8]>(LaneMode::Sequential)
}
# 
# let mut ctx = Context::acquire();
# 
# let a: CollectTensor<'_, _, bf16, m![1], m![1 # 2], m![1 # 256], m![M], m![P]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let b: TrfTensor<bf16, m![1], m![1 # 2], m![1 # 256], m![N], m![P]> = unsafe { TrfTensor::from_addr(TrfAddress::Full) };
# 
# let i: ContractTimeTensor<'_, _, f32, m![1], m![1 # 2], m![1 # 256], m![N], m![M], m![P]> = a 
#     .contract_outer::<m![M], m![P], m![N], m![P], _>(&b)
#     .contract_packet::<m![P]>()
#     .contract_time::<m![M]>();
# 
# let _o = lane_sequential(i);
```

## 제약

Lane Folder 자체에는 제약이 없다.
여기서 고른 `LaneMode` 가 상류 Time Reducer 가 강제하는 슬롯 용량 한계를 결정한다([Time Reducer 제약](./time-reducer.md#constraints) 참고).


## 성능

Interleaved 모드에서는 `Lane < 8` 일 때 처리량이 `Lane::SIZE / 8` 만큼 떨어진다(비활성 레인이 버스 자리를 비운다).

Sequential 모드에서 `Packet::SIZE < 8` 이면(예: Packet Reducer 가 `bf16` packet 의 절반을 접은 뒤의 `Packet::SIZE = 4`), 각 사이클은 8원소 버스 폭 전부가 아니라 정확히 `Packet::SIZE` 개 원소를 싣는다: 사이클당 출력은 더 좁지만 패딩도 없고 낭비되는 버스 자리도 없다.

지연은 무시할 만하다: Lane Folder 는 레인별 출력을 재배치할 뿐이고 버퍼를 비우는 시간 외에 사이클을 더하지 않는다.
