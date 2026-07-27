# Packet Reducer

Packet Reducer 는 단일 Packet 안에서 가장 안쪽 축약 축들을 더해 리듀스하며, 리듀스 트리는 레인마다 하나씩 있다.

## 인터페이스

`.contract_packet()` 이 Packet Reducer 를 호출한다.
각 레인은 `i4`, `i8`, `f8`, `bf16` 원소로 이루어진 32 B 또는 64 B Packet 을 받으며, 이는 [Outer](./outer.md) 단계의 `OutPacket` 에서 물려받은 것이다.
형식적으로는 \\(\text{output}[i] = \sum_{j} \text{input}[i, j]\\) 를 계산하며, `i` 는 살아남는(출력) 축을 훑고 `j` 는 `Packet` 안의 축약 축을 훑는다.

```rust,ignore
impl<
    'l,
    const T: Tu,
    D: Scalar,
    Storage: ContractionCast<Output = D>,
    Chip: M,
    Cluster: M,
    Slice: M,
    Lane: M,
    Time: M,
    Packet: M,
    B: Backend,
> ContractOuterTensor<'l, T, D, Storage, Chip, Cluster, Slice, Lane, Time, Packet, B>
{
    /// Spatial reduction within `Packet`: validates the reduce-add along the contracted axes inside
    /// `Packet` that the fused fold at `contract_lane` will perform. `D` is the widened accumulator the
    /// deferred carrier stays keyed on; the DPE input packet is still sized in `Storage` bytes.
    #[primitive(ContractOuterTensor::contract_packet)]
    pub fn contract_packet<OutPacket: M>(
        self,
    ) -> ContractPacketTensor<'l, T, D, Chip, Cluster, Slice, Lane, Time, OutPacket, B> {
        verify_contract_packet::<Storage, Packet, OutPacket>();
        // Carry the deferred operands forward unreduced: the fused contraction at `contract_lane`
        // performs this Packet reduction too. This stage only re-types the carrier to `OutPacket`.
        ContractPacketTensor::new(self.ctx, self.inner)
    }
}
```

아래 커널은 8 개 레인을 모두 병렬로 쓴다. 트리 깊이 5 가 `bf16` 원소 32 개짜리 `B` 를 따라 합산하여, `f32` 값 하나를 `A` 위치마다 내놓는다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 32, B = 32, C = 8];

fn matmul<'l, const T: Tu>(
    input: CollectTensor<'l, T, bf16, m![1], m![1 # 2], m![1 # 256], m![A, B / 16], m![B % 16]>,
    trf: &TrfTensor<bf16, m![1], m![1 # 2], m![1 # 256], m![C], m![B]>,
) -> ContractTensor<'l, T, f32, m![1], m![1 # 2], m![1 # 256], m![A], m![C]> {
    //
    // Spatial reduction: tree depth 5 reduces 32 bf16 elements along B → f32
    // Output (Interleaved): Time = [A], Packet = [C]
    input.contract_outer::<m![A], m![B], _, _, _>(&trf)
         .contract_packet::<m![1]>()
         .contract_time::<m![A]>()
         .contract_lane::<m![A], m![C]>(LaneMode::Interleaved)
}
# 
# let mut ctx = Context::acquire();
# 
# let a: CollectTensor<'_, _, bf16, m![1], m![1 # 2], m![1 # 256], m![A, B / 16], m![B % 16]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let b: TrfTensor<bf16, m![1], m![1 # 2], m![1 # 256], m![C], m![B]> = unsafe { TrfTensor::from_addr(TrfAddress::Full) };
# let _o = matmul(a, &b);
```

## 구조

```text
ReducePacket = Packet / 2^d        for 0 ≤ d ≤ log2(Packet::SIZE)
OutPacket    = ReducePacket        if ReducePacket::SIZE ≤ 32
               ReducePacket = 32   otherwise
```

Packet Reducer 는 먼저 입력 Packet 에 대해 레인마다 독립적인 [리듀스 트리](https://en.wikipedia.org/wiki/Reduction_operator) 를 돌린다.
깊이 0 에서 트리의 잎은 입력 Packet 의 원소를 담고, 이후 깊이마다 쌍을 더해 원소 개수를 절반으로 줄인다.
트리의 최대 깊이는 `log2(Packet::SIZE)` 이므로 `i4` 는 7 (`Packet::SIZE = 128`), `i8` / `f8` 은 6 (64), `bf16` 은 5 다 (32).
사용자가 준 `OutPacket` 으로부터 컴파일러가 트리 깊이 `d` 를 도출하고, 트리는 가장 안쪽 `2^d` 개 원소를 소비해 `ReducePacket` 을 만든다.

그다음 Packet Reducer 는 `ReducePacket` 을 `OutPacket` 으로 절단(트리밍)하며, 그 크기는 32 원소로 제한된다. 하위의 Time Reducer 가 가진 레인당 누산기의 열이 32 개뿐이기 때문이다.
`ReducePacket::SIZE > 32` 이면 바깥쪽 더미가 잘려 나가고 가장 안쪽 32 개 원소만 살아남는다.
예를 들어 `i4` 는 128 원소 Packet 으로 도착하므로 `d ∈ {0, 1}` 은 128 원소 또는 64 원소 `ReducePacket` 을 만들고, 둘 다 `OutPacket::SIZE = 32` 로 절단된다.



## 성능

지연은 트리 깊이에 따라 달라진다. `i4` 는 7 사이클, `i8`/`f8` 은 6, `bf16` 은 5 다.
원소 타입이 넓을수록 Packet 하나에 들어가는 원소가 적어서 깊이가 줄어든다.
덧셈 트리는 완전히 파이프라인화되어 있어, 깊이는 첫 출력까지의 지연만 늘릴 뿐 정상 상태 처리량을 떨어뜨리지 않는다. 파이프라인이 한번 채워지면 매 사이클 Packet 하나가 들어가고 리듀스된 출력 하나가 나온다.

`Lane < 8` 이면 비활성 레인의 리듀스 트리가 놀기 때문에, 사이클당 처리량이 `Lane::SIZE / 8` 에 비례해 떨어진다.
