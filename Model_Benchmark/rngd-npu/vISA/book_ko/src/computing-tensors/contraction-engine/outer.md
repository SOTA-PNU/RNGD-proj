# Outer

Outer 단계는 두 피연산자를 서로 맞는 모양으로 브로드캐스트한 뒤 원소 단위로 곱한다.

"Outer" 라는 이름은 선형대수의 **외적**에서 왔다.
벡터 `u`(길이 `n`)와 `v`(길이 `m`)에 대해 `u v^T` 는 `(u v^T)[i, j] = u[i] × v[j]` 인 `n × m` 행렬이다.
그 행렬은 `u` 를 열 축(길이 `m`)을 따라 브로드캐스트하고, `v` 를 행 축(길이 `n`)을 따라 브로드캐스트한 뒤 원소 단위로 곱해서 만들어진다.
Outer 단계의 세 하위 단계는 바로 이 의미론을 그대로 하드웨어로 구현한 것이며, 직렬로 실행된다:

- [Stream Adapter](#stream-adapter) 는 [Collect Engine](../collect-engine.md) 에서 오는 스트리밍 피연산자를 처리(및 브로드캐스트)한다.
- [TRF Sequencer](#trf-sequencer) 는 TRF SRAM 에서 오는 TRF 피연산자를 처리(및 브로드캐스트)한다.
- [Multiplier](#multiplier) 는 피연산자 타입을 넓히고(`i4` / `i8` 은 `i32` 로, `f8` / `bf16` 은 `f32` 로) 정렬된 두 피연산자를 원소 단위로 곱한다.

출력은 결합 매핑 `[Chip, Cluster, Slice, Lane, Time, Packet]` 의 곱해진 텐서 하나이며, [Packet Reducer](./packet-reducer.md) 가 리듀스-덧셈할 준비가 된 상태다.

## 인터페이스

`CollectTensor` 의 `.contract_outer(&trf)` 가 Outer 단계를 호출한다.

```rust,ignore
impl<
    'l,
    const T: Tu,
    P: CanApplyContractOuter,
    D: Scalar + ContractionCast,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Outer stage: stashes the two un-broadcast operands (widened to the accumulator type)
    /// and the layouts [`super::lane::contract_lane`] needs to fuse them into a [`LazyContraction`]. No
    /// materializing alternative, no per-backend branch -- every backend fuses the same way.
    #[primitive(TuTensor::contract_outer)]
    pub fn contract_outer<OutTime: M, OutPacket: M, Lane: M, TrfElement: M, TrfD>(
        self,
        trf_tensor: &TrfTensor<TrfD, Chip, Cluster, Slice, Lane, TrfElement, B>,
    ) -> ContractOuterTensor<'l, T, <D as ContractionCast>::Output, D, Chip, Cluster, Slice, Lane, OutTime, OutPacket, B>
    where
        D: Cast<<D as ContractionCast>::Output>,
        // The weight (TRF) type must form a valid contraction-engine operand pair with the
        // stream type `D`: same type, or a mixed integer precision within a
        // family (i4/i5 x i4/i5, i8/i9 x i8/i9). Both operands widen to the
        // stream's accumulator for the multiply.
        TrfD: Scalar + ContractionWeight<D> + Cast<<D as ContractionCast>::Output>,
    {
        type Out<D> = <D as ContractionCast>::Output;

        // Skipping the broadcast transpose does not skip its validity contract -- a malformed
        // contraction would otherwise slip past these asserts and panic far downstream instead.
        stream_adapter::verify_stream_adapter::<D, Lane, Time, Packet, OutTime, OutPacket>();
        trf_sequencer::verify_trf_sequencer::<TrfD, Lane, TrfElement, OutTime, OutPacket>();

        // The operands keep their own compact layouts: lhs is `[Chip, Cluster, Slice, Time, Packet]`
        // (`self.inner`), rhs is `[Chip, Cluster, Slice, Lane, TrfElement]` (`trf_tensor`). A
        // bare-buffer backend reads its strides from these; `MathStorage` ignores them (its axes live
        // in the storage).
        let lhs_map = <m![{ Chip }, { Cluster }, { Slice }, { Time }, { Packet }]>::to_value();
        let rhs_map = <m![{ Chip }, { Cluster }, { Slice }, { Lane }, { TrfElement }]>::to_value();
        let pre_reduce = <m![{ Chip }, { Cluster }, { Slice }, { Lane }, { OutTime }, { OutPacket }]>::to_value();

        // Widen each operand to the `Out<D>` accumulator up front, at the operand's own (compact) size,
        // not pre_reduce's -- the fold at `contract_lane` then runs entirely in `Out<D>`. Parity-identical
        // to the old per-cell widen (same `Cast::cast` per element), it just never allocates the
        // pre_reduce-shaped broadcast this stage used to build.
        //
        // `map_bounded`, NOT the plain `map`: `D` (the stream side) and `TrfD` (the TRF side) may each
        // legitimately be a non-`MaterializableScalar` staging type (`i5`/`i9`, produced by
        // `fetch_zero_point_sub` -- "an i5/i9 may still be a contraction weight resident in the TRF" per
        // its own doc). Such a type's storage-native length recovery over-reports (its `BITS` names a
        // real-hardware wire width, disconnected from its host in-memory size), so a plain `map`'s
        // internal whole-buffer walk reads/writes past the buffer once the count crosses the true
        // element count -- this is the ONE place either operand is read back as a whole tensor before
        // `contract_lane`'s fused `Backend::contraction` (its only other lifetime is written-once by
        // `fetch_zero_point_sub`, never read back that way), so it is also the one place this matters.
        let contraction = LazyContraction {
            lhs: self.inner.map_bounded(|v| -> Out<D> { v.cast() }).inner,
            rhs: trf_tensor.inner.map_bounded(|v| -> Out<D> { v.cast() }).inner,
            lhs_map,
            rhs_map,
            pre_reduce,
        };
        ContractOuterTensor::new(self.ctx, contraction)
    }
}
```

각자의 어댑터(스트리밍 경로는 Stream Adapter, TRF 경로는 TRF Sequencer)를 거치고 나면 두 경로 모두 서로 맞는 모양의 `Lane` / `Time` / `Packet` 을 공급하고, Multiplier 가 정렬된 위치끼리 원소 단위로 곱한다.

스트리밍 피연산자의 `Time` / `Packet` 은 출력의 `OutTime` / `OutPacket` 으로 매핑된다:
`OutPacket` 은 [Packing](#packing) 을 통해 `Time` 의 가장 안쪽 크기 1 또는 2 인자를 흡수한다.
`OutTime` 은 `Time` 의 남은 인자들을 유지하며, [Broadcast](#broadcast) 를 통해 가장 안쪽 위치에 브로드캐스트 인자가 더해진다.
브로드캐스트 인자는 TRF 피연산자의 `Lane` / `Element`(스트리밍 피연산자가 TRF 매핑에 맞춰 복제되는 곳)에서 오고, `OutTime` / `OutPacket` 에는 나타나지만 입력에도 TRF 에도 없는 순수 출력 축에서도 온다(예: `D` 가 브로드캐스트되는 einsum `AB, BC -> ABCD`).

`TrfTensor` 의 모양은 `[Chip, Cluster, Slice, Lane, Element]` 이고 `Chip` / `Cluster` / `Slice` / `Lane` 은 공간적으로 병렬이다: `Chip` / `Cluster` / `Slice` 는 출력으로 그대로 지나가고, `Lane` 은 레인별 데이터를 하드웨어 레인 1–8개에 나눈다. `Element`([`.to_trf()`](../register-files.md#tensor-register-file) 가 정하는 레인별 배치)는 TRF Sequencer 가 재배치해 `OutTime` / `OutPacket` 을 채운다.

<a id="stream-adapter"></a>
## Stream Adapter

Stream Adapter 는 스트리밍 `Time` / `Packet` 을 [Packing](#packing) 과 [Broadcast](#broadcast) 두 연산을 통해 연산 모양(`Lane` / `OutTime` / `OutPacket`)으로 변환한다.
컴파일러는 사용자가 준 `OutTime` / `OutPacket`, TRF 피연산자, 그리고 순수 출력 브로드캐스트 축들에서 세 자유 변수(`PackSize`, `LaneBroadcast`, `TimeBroadcast`)를 도출해 다음 매핑을 만든다:

```text
Lane      = LaneBroadcast
OutTime   = [Time / PackSize, TimeBroadcast]
OutPacket = [Time % PackSize, Packet] # (64 / D::SIZE)
```

<a id="packing"></a>
### Packing

Collect Engine 은 32 B flit 을 만들고, Outer 단계는 `PackSize × 32` B 크기의 packet 을 내보낸다(RNGD 에서는 32 또는 64).
Packing 은 연속된 flit `PackSize ∈ {1, 2}` 개를 packet 하나로 합친다:

```text
PackTime   = [Time / PackSize]
PackPacket = [Time % PackSize, Packet] # (PackSize × 32 / D::SIZE)
```

`PackSize` 는 `OutPacket` 을 입력 `Packet` 과 맞춰 보면서 정해진다: `OutPacket` 이 `Time` 의 가장 안쪽 크기 2 인자를 흡수하면 `PackSize = 2`, 그렇지 않으면 `PackSize = 1` 이다.
동등하게 `PackSize = OutPacket::SIZE * D::SIZE / 32` 이므로, 사용자가 `OutPacket`(32 B 또는 64 B)을 고르면 Packing 의 collect flit 개수는 따라온다.

하드웨어는 내부적으로 항상 64 B packet 으로 동작한다; `PackSize = 1` 일 때 쓰이지 않는 32 B 절반은 0 을 담으며, 이 0 은 논리적 `OutPacket` 타입으로 전파되지 않는다. 따라서 하류 단계(Packet Reducer, Lane Folder)는 `PackSize × 32` B 페이로드만 보게 되어 더미 사이클을 피한다. [Lane Folder 의 Sequential 설명](./lane-folder.md#sequential)을 참고한다.

<a id="broadcast"></a>
### Broadcast

패킹 후 Stream Adapter 는 `LaneBroadcast`(TRF 의 `Lane` 매핑, ∈ {1, 2, 4, 8})로 데이터를 공간적으로, `TimeBroadcast` 로 시간적으로 브로드캐스트한다.
`TimeBroadcast` 는 입력 `Time` 에 없는 TRF `Element` 의 인자들을 담당하고, 입력에도 TRF 에도 없이 `OutTime` 에만 있는 순수 출력 축도 담당한다: 같은 브로드캐스트 장치가 두 경우 모두에 걸쳐 packet 을 복제한다.
각 목적지는 동일한 `OutPacket` 을 받는다:

```text
Lane      = LaneBroadcast
OutTime   = [PackTime, TimeBroadcast]
OutPacket = PackPacket
```

`TimeBroadcast` 인자는 `OutTime` 의 *가장 안쪽* 위치를 차지한다: 바깥의 `PackTime` 인자를 반복하기 전에 같은 `OutPacket` 이 그 인자들에 걸쳐 다시 전송된다.

### 예제

아래 예제는 두 연산을 모두 사용한다: Packing 은 `Time` 의 가장 안쪽 크기 2 인자 `L` 을 `Packet` 으로 흡수하고(`PackSize = 2`), Lane Broadcast 는 그 결과 packet 을 `N = 8` 개 레인으로 분배하며, Time Broadcast 는 TRF 에만 있는 `B = 5` 축에 걸쳐 스트리밍 데이터를 타일링한다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![M = 32, N = 8, K = 16, L = 2, B = 5];

fn stream_adapter_example<'l, const T: Tu>(
    input: CollectTensor<'l, { T }, bf16, m![1], m![1 # 2], m![1 # 256], m![M, L], m![K]>,
    trf: &TrfTensor<bf16, m![1], m![1 # 2], m![1 # 256], m![N], m![B, L, K]>,
) -> ContractOuterTensor<'l, { T }, f32, bf16, m![1], m![1 # 2], m![1 # 256], m![N], m![M, B], m![L, K]> {
    // Packing (PackSize = 2):
    //   L = 2 (innermost Time) absorbed into Packet.
    //   PackTime = [M = 32], PackPacket = [L = 2, K = 16] = 32 bf16 = 64B.
    // Lane Broadcast: same packet to all N = 8 lanes.
    // Time Broadcast: B = 5 (TRF-only) added at innermost OutTime.
    //   OutTime = [M, B = 5], OutPacket = [L = 2, K = 16].
    input.contract_outer::<m![M, B], m![L, K], _, _, _>(trf)
}
# 
# let mut ctx = Context::acquire();
# 
# let a: CollectTensor<'_, _, bf16, m![1], m![1 # 2], m![1 # 256], m![M, L], m![K]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let b: TrfTensor<bf16, m![1], m![1 # 2], m![1 # 256], m![N], m![B, L, K]> = unsafe { TrfTensor::from_addr(TrfAddress::Full) };
# let _o = stream_adapter_example(a, &b);
```

### 제약

- `OutPacket::SIZE * Storage::SIZE ∈ {32, 64}` 바이트(RNGD 기준). 여기서 `Storage` 는 넓히기 전의 피연산자 dtype 이다(예: `bf16` = 2 B 이며, 결과 텐서가 지니는 넓혀진 `f32` 누산기가 *아니다*): `PackSize = 1` 이면 32, `PackSize = 2` 면 64. 사용자가 이 크기를 고르면 Packing 의 collect flit 개수는 따라온다.
- `PackSize ∈ {1, 2}` ([Packing](#packing) 참고).
- `Lane::SIZE ∈ {1, 2, 4, 8}`.

### 성능

`PackSize` 가 MAC 활용률을 정한다.
`PackSize = 2` 는 64 B 를 가득 채우고 모든 MAC 을 쓴다.
`PackSize = 1` 은 32 B 만 채우므로 0 으로 패딩된 절반은 항상 0 을 곱하게 되고 실효 처리량은 절반이 된다.

`PackSize = 2` 는 Packet 하나에 2 사이클이 걸리지만(32 B flit 두 개가 64 B Packet 하나로 합쳐진다) 이것이 파이프라인 병목은 아니다: 상류가 완전한 fetch 속도로 매 사이클 32 B flit 하나를 공급하므로, Stream Adapter 는 flit 이 도착하는 속도만큼 소비하고 하류의 소비에 맞춰 2 사이클마다 Packet 하나를 내보낸다.

Time Broadcasting 은 fetch 비용을 분산시킨다.
브로드캐스트 인자는 다시 fetch 하지 않고 같은 스트리밍 packet 을 여러 사이클에 걸쳐 재사용하므로, 그 인자들에 대한 대역폭 비용이 사라진다.

전체적으로 Stream Adapter 는 fetch 대역폭(fetch 당 최대 32 B/사이클)에 묶인다.
활용률을 최대로 하려면 슬라이스에 걸쳐 fetch 패턴을 인터리브한다.

<a id="trf-sequencer"></a>
## TRF Sequencer

TRF Sequencer 는 `TrfTensor` 를 읽어 그 `Element` 를 [Packet Reducer](./packet-reducer.md) 를 위한 `OutTime` / `OutPacket` 으로 재배치한다.
TRF 저장 배치(레인, 뱅크, 행, 더블 버퍼링, 캐시)는 [Register Files](../register-files.md) 를 참고한다.

매핑:

```text
OutTime   = (sequencing over [Element / ReadSize] with broadcasts)
OutPacket = [PacketBroadcast, Element % ReadSize]
```

`OutPacket` 은 매 사이클 TRF 읽기 한 번으로 채워진다: 매 사이클 완전한 `OutPacket` 하나(두 뱅크에 걸쳐 레인당 640 비트, 한 뱅크만 읽을 때는 320 비트)가 생성된다.
레인 8개에 걸친 슬라이스별 합계(바이트 단위)는 [To Contraction Engine](../register-files.md#to-contraction-engine) 을 참고한다.
읽기는 `Element` 의 가장 안쪽 연속 구간을 끌어와 복제해서 64 B `OutPacket` 을 채운다.
컴파일러는 `Element % ReadSize == OutPacket % ReadSize` 이면서 `ReadSize * D::SIZE ≤ 64` 바이트인 가장 큰 `ReadSize` 를 고른다: `ReadSize` 가 넓으면 레인당 TRF 두 뱅크에 걸치고, 좁으면 한 뱅크만 쓴다.

`OutTime` 은 `Element / ReadSize` 를 시퀀싱하며(선택적으로 브로드캐스트를 더해) 여러 사이클에 걸쳐 채워진다.
TRF Sequencer 는 다른 모든 [sequencer](../../moving-tensors/sequencer.md) 와 같은 중첩 루프 설정을 쓴다.


### 예제

이 예제에서는 `ReadSize` 가 64 B 읽기 한 번으로 `Element` 전체를 덮으므로 `Element / ReadSize` 는 자명하고 sequencer 는 브로드캐스트만 반복한다:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![M = 32, N = 8, K = 32];

fn trf_sequencer_full_read<'l, const T: Tu>(
    input: CollectTensor<'l, T, bf16, m![1], m![1 # 2], m![1 # 256], m![M, K / 16], m![K % 16]>,
    trf: &TrfTensor<bf16, m![1], m![1 # 2], m![1 # 256], m![N], m![K]>,
) -> ContractOuterTensor<'l, T, f32, bf16, m![1], m![1 # 2], m![1 # 256], m![N], m![M], m![K]> {
    // Element         = K
    // ReadSize        = 32
    // PacketBroadcast = 1
    // OutTime         = M      (sequencing over [K / 32] (= 1), broadcast M)
    // OutPacket       = K      (= [1, K % 32])
    input.contract_outer::<m![M], m![K], _, _, _>(trf)
}
# 
# let mut ctx = Context::acquire();
# 
# let a: CollectTensor<'_, _, bf16, m![1], m![1 # 2], m![1 # 256], m![M, K / 16], m![K % 16]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let b: TrfTensor<bf16, m![1], m![1 # 2], m![1 # 256], m![N], m![K]> = unsafe { TrfTensor::from_addr(TrfAddress::Full) };
# let _o = trf_sequencer_full_read(a, &b);
```

이 예제에서는 `ReadSize` 가 `Element` 의 일부만 덮으므로 `Element / ReadSize` 가 자명하지 않고 sequencer 는 브로드캐스트와 함께 바깥 `Element` 인자를 반복한다:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![M = 32, N = 8, K = 16, L = 2, O = 2];

fn trf_sequencer_partial_read<'l, const T: Tu>(
    input: CollectTensor<'l, T, bf16, m![1], m![1 # 2], m![1 # 256], m![O, M, L], m![K]>,
    trf: &TrfTensor<bf16, m![1], m![1 # 2], m![1 # 256], m![N], m![O, K]>,
) -> ContractOuterTensor<'l, T, f32, bf16, m![1], m![1 # 2], m![1 # 256], m![N], m![O, M], m![L, K]> {
    // Element         = [O, K]
    // ReadSize        = 16
    // PacketBroadcast = L
    // OutTime         = [O, M]    (sequencing over [O, K] / 16 (= O), broadcast M)
    // OutPacket       = [L, K]    (= [L, [O, K] % 16])
    input.contract_outer::<m![O, M], m![L, K], _, _, _>(trf)
}
# 
# let mut ctx = Context::acquire();
# 
# let a: CollectTensor<'_, _, bf16, m![1], m![1 # 2], m![1 # 256], m![O, M, L], m![K]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let b: TrfTensor<bf16, m![1], m![1 # 2], m![1 # 256], m![N], m![O, K]> = unsafe { TrfTensor::from_addr(TrfAddress::Full) };
# let _o = trf_sequencer_partial_read(a, &b);
```

### 제약

- **하드웨어 차원**: `Chip::SIZE`, `Cluster::SIZE`, `Slice::SIZE` 는 하드웨어 구성과 일치해야 한다([Sequencer](../../moving-tensors/sequencer.md#configuration) 참고).
- **주소 정렬**: `Element % ReadSize` 가 64 B 전체를 덮으면 읽기가 레인당 TRF 두 뱅크에 걸치므로, sequencer 의 기준 주소와 모든 스트라이드가 64 B 에 정렬되어야 한다.

### 구조

여러 사이클에 걸쳐 sequencer 는 다른 모든 [sequencer](../../moving-tensors/sequencer.md) 와 같은 중첩 루프 설정으로 `Element` 의 바깥 인자들(즉 `Element / ReadSize`)을 반복하므로, `TrfTensor` 하나는 내용을 다 소진하기까지 `Element / ReadSize` 사이클을 걷는다.
`PacketBroadcast` 인자는 한 사이클 안에서 같은 행을 복제해, TRF 읽기 대역폭을 추가로 쓰지 않고 자연스러운 `ReadSize` 를 넘겨 64 B `OutPacket` 을 채운다.

### 성능

처리량은 사이클마다 레인당 완전한 `OutPacket` 하나다: 두 뱅크를 읽으면 레인당 640 비트, 한 뱅크만 읽으면 레인당 320 비트다.
슬라이스별 바이트 합계는 [Register Files: To Contraction Engine](../register-files.md#to-contraction-engine) 을 참고한다.
TRF 읽기 캐시와 뱅크 교대([Register Files: Double Buffering](../register-files.md#double-buffering) 참고)는 브로드캐스트 재사용과 좁은 읽기에 걸쳐 동시에 진행되는 sub 컨텍스트 저장이 막히지 않게 한다.

<a id="multiplier"></a>
## Multiplier

Multiplier 는 Stream Adapter 와 TRF Sequencer 에서 온 정렬된 두 피연산자를 소비하고, 하류 누산기가 넘치지 않도록 각 입력 원소를 축약 출력 타입으로 넓힌 뒤(`i4`/`i8` -> `i32`, `f8`/`bf16` -> `f32`) 원소 단위로 곱한다.
그 출력은 결합 매핑 `[Chip, Cluster, Slice, Lane, Time, Packet]` 의 텐서 하나이며, [Packet Reducer](./packet-reducer.md) 의 입력이 된다.
매 `Time` 사이클마다 모든 `Lane` 이 병렬로 완전한 `packet` 분량의 곱을 만들어낸다.
