# 레지스터 파일

[Collect Engine](./collect-engine.md) 은 [Contraction Engine](./contraction-engine/index.md) 과 [Vector Engine](./vector-engine/index.md) 으로 스트리밍한다.
두 엔진은 슬라이스마다 하나씩 있는 레지스터 파일에서도 입력을 받는다: Tensor Register File(TRF) 은 Contraction Engine 에, Vector Register File(VRF) 은 Vector Engine 에 공급한다.
이 레지스터 파일들은 소비하는 엔진이 돌기 전에 채워져 있어야 한다.

<a id="tensor-register-file"></a>
## Tensor Register File

### 인터페이스

`TrfTensor` 는 TRF 에 저장된 텐서다:

```rust,ignore
/// Tensor stored in the tensor register file.
#[primitive(TrfTensor)]
#[derive(Debug)]
pub struct TrfTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Element: M, B: Backend = CurrentBackend> {
    pub(crate) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Lane, Element>>>>, B>,
    #[expect(dead_code)]
    address: Option<TrfAddress>,
    _marker: PhantomData<(D, Chip, Cluster, Slice, Lane, Element)>,
}
```

`Chip` / `Cluster` / `Slice` 는 소스에서 그대로 통과한다. `Lane` 은 공간 병렬성을 인덱싱한다(활성 레인 1, 2, 4, 8 개). `Element` 는 레인별 레이아웃을 담는다.

#### Collect Engine 에서

`.to_trf::<Lane, Element>()` 는 `CollectTensor` 에서 TRF 전체에 `TrfTensor` 를 만들고, `.to_trf_at::<Lane, Element>(address)` 는 `TrfAddress` 영역을 대상으로 한다:

```rust,ignore
impl<'l, const T: Tu, P: CanApplyToTrf, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Stores to the tensor register file.
    #[primitive(TuTensor::to_trf)]
    pub fn to_trf<Lane: M, Element: M>(self) -> TrfTensor<D, Chip, Cluster, Slice, Lane, Element, B> {
        verify_to_trf::<D, Lane, Time, Packet, Element>(&TrfAddress::Full);
        TrfTensor::new(self.inner.transpose(false), None)
    }

    /// Stores to the tensor register file at `address`.
    #[primitive(TuTensor::to_trf_at)]
    pub fn to_trf_at<Lane: M, Element: M>(
        self,
        address: TrfAddress,
    ) -> TrfTensor<D, Chip, Cluster, Slice, Lane, Element, B> {
        verify_to_trf::<D, Lane, Time, Packet, Element>(&address);
        TrfTensor::new(self.inner.transpose(false), Some(address))
    }
}
```

`.to_trf` 는 스트리밍 `Time` / `Packet` 을 `Lane` / `Element` 로 재구성한다:

```text
Lane    = Time / FlitsPerLane
Element = [Time % FlitsPerLane, Packet]
```

여기서 `FlitsPerLane` 은 컴파일러가 `Lane` 과 `Time` 에서 유도하며, 그래서 각 레인은 연속된 `FlitsPerLane` 개의 flit 으로 채워진다.

예를 들어 matmul 커널에서 `Lane` 은 출력 채널을 담고 `Element` 는 축약되는 축을 담는다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![V = 32, M = 32, N = 8, K = 32];

type Chip    = m![1];
type Cluster = m![V / 16];
type Slice   = m![V % 16 # 256];
type Lane    = m![N];

/// Stores matmul weights into TRF for consumption by `bmatmul` in
/// [Contraction Engine: Example: Batched MatMul](./contraction-engine/index.md#example-batched-matmul).
fn store_bmatmul_trf<'l, const T: Tu>(
    input: CollectTensor<'l, T, bf16, Chip, Cluster, Slice, m![N, K / 16], m![K % 16]>,
) -> TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![K]> {
    input.to_trf_at(TrfAddress::FirstHalf)
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, bf16, Chip, Cluster, Slice, m![N, K / 16], m![K % 16]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = store_bmatmul_trf(c);
```

#### Data Memory 에서

완전히 연속적인 입력 접근(빈틈이나 재정렬이 없는 경우)에 대해 TRF 는 *short command*(StoTRF) 를 지원하는데, 이는 Data Memory 에서 TRF 로 데이터를 곧바로 적재해 Fetch → Switch → Collect → `to_trf()` 파이프라인 전체를 우회하는 간결한 하드웨어 명령어다.
이 지름길은 임의 레이아웃 지원을 내주고 더 낮은 설정 오버헤드를 얻는다.


<a id="to-contraction-engine"></a>
#### Contraction Engine 으로

읽기 한 번은 8 레인 × (1 또는 2) 뱅크 × 1 행 × 뱅크당 320비트를 훑으며, 좁은 읽기(뱅크 하나)는 사이클당 레인당 320비트를, 넓은 읽기(뱅크 둘)는 사이클당 레인당 640비트를 낸다: 활성 레인 전부가 같은 행의 뱅크 하나 또는 둘에 병렬로 접근한다(좁은 읽기는 뱅크 하나, 넓은 읽기는 둘).
슬라이스당으로는 8 레인 전체에서 320 바이트/사이클(좁은 읽기) 또는 640 바이트/사이클(넓은 읽기)이 된다.
시퀀서가 이 읽기들을 행에 걸쳐 반복하고 브로드캐스트하는 방식은 [TRF Sequencer](./contraction-engine/outer.md#trf-sequencer) 를 참고한다.

### 구조

TRF 는 8 레인 × 2 뱅크 × 128 행 × 320비트 = 슬라이스당 80 KB 구조를 갖는 뱅크형 SRAM 이다.
8 개 레인은 병렬로 동작하며, 접근마다 1, 2, 4, 8 개가 활성이 된다.

320비트 행 하나에 원소가 몇 개나 들어가는지는 데이터 타입에 달려 있다:

| 타입 | 저장되는 원소 크기 | 행당 원소 수 |
|------|---------------------|------------------|
| `i4` → `i5` | 5 비트 | 64 |
| `i4` → `i9` | 9 비트 (근사치, 320비트 행에 맞게 올림) | 32 |
| `i8` / `f8` | 8 비트 | 32 (행당 40 바이트) |
| `bf16` | 16 비트 | 16 (행당 32 바이트) |

저장 시 승격되는 것은 `i4` 원소뿐이다: `i4 → i5`(5비트) 와 `i4 → i9`(9비트) 는 fetch adapter 의 선택적 제로포인트 감산을 위한 여유를 남기며, 이 감산은 `i4` 중간값을 니블당 1비트 넓힐 수 있다.
`i8` / `f8` 과 `bf16` 은 고유 폭(각각 8비트와 16비트)을 유지한다. 320비트 행은 평평한 8비트 또는 16비트 패킹에 비해 여유를 더 갖고 있어서 같은 물리적 행 폭이 모든 타입을 감당한다.


활성 레인이 8 개보다 적으면 각 활성 레인이 더 많은 행을 본다(행 수가 늘어난 것처럼). 활성 개수를 절반으로 줄이면 활성 레인당 행 수가 두 배가 된다(예: 4 개 활성 → 뱅크당 256 행, 1 개 활성 → 1024 행).

<a id="double-buffering"></a>
### 더블 버퍼링

TRF 는 각 뱅크를 두 반쪽으로 나눠 더블 버퍼링을 가능하게 한다: [TRF Sequencer](./contraction-engine/outer.md#trf-sequencer) 가 한쪽 반에서 적재하는 동안 저장이 다른 반쪽을 채우고, 반복 회차 사이에 둘을 뒤집을 수 있다.
세 가지 주소 모드가 영역을 고르며 저장 시점에 고정된다: `Full` 은 뱅크당 128 행 전부를, `FirstHalf` 는 0–63 행을, `SecondHalf` 는 64–127 행을 쓴다.
반쪽 모드는 슬라이스당 용량을 40 KB 로 제한한다.

main 과 sub 컨텍스트에 걸쳐 이 반쪽들을 쓰는 커널 패턴은 [스케줄링: 더블 버퍼링 패턴](../scheduling.md#double-buffering-pattern) 을 참고한다.

두 반쪽은 같은 뱅크를 공유하므로, 서로 다른 행을 대상으로 하더라도 읽기와 쓰기가 뱅크 수준에서 경합한다.
한 사이클에 둘이 같은 뱅크를 대상으로 하면 읽기가 우선하는데, 축약 파이프라인은 이번 사이클에 데이터가 필요한 반면 저장은 기다릴 수 있기 때문이다.

TRF 는 읽기 캐시와 뱅크 교대로 이 경합을 완화한다.

TRF 읽기는 재사용이 많다: 같은 데이터가 보통 여러 사이클에 걸쳐 브로드캐스트되므로, 직접 사상 읽기 캐시(8 레인 × 2 뱅크 × 4 행 × 320비트 = 2.5 KB)가 뱅크 앞에 놓여 반복되는 읽기를 흡수한다.
이 캐시는 동시에 일어나는 저장과의 경합도 덜어 준다.
적중하면 읽기가 뱅크를 건너뛰므로 그 사이클에 저장이 뱅크를 쓸 수 있다.
실패하면 캐시가 뱅크에서 다시 채우며 그 사이클 동안 뱅크를 점유한다.

좁은 읽기(≤ 32 바이트)에는 뱅크 교대가 두 번째 완화책을 더한다.
읽기는 뱅크 하나만 쓰므로 두 뱅크를 32바이트 단위로 번갈아 쓸 수 있다.
그러면 읽기와 쓰기가 연이은 사이클에서 서로 다른 뱅크에 놓여, 캐시 실패에도 경합을 피한다.
넓은 읽기(64 바이트)는 매 사이클 두 뱅크를 모두 점유하므로 캐시가 실패할 때마다 동시 저장이 막힌다. 좁은 읽기는 캐시가 실패해도 절반 대역폭의 교대를 유지한다.


## Vector Register File

VRF 는 [Collect Engine](./collect-engine.md) 에서 또는 Data Memory 에서 직접 기록되고, [Vector Engine](./vector-engine/index.md) 이 읽는다.

### 인터페이스

`VrfTensor` 는 VRF 에 저장된 텐서다:

```rust,ignore
/// Tensor stored in the vector register file (VRF).
#[primitive(VrfTensor)]
#[derive(Debug, Clone)]
pub struct VrfTensor<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend = CurrentBackend> {
    pub(crate) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Element>>>, B>,
    #[expect(dead_code)]
    address: Option<Address>,
    _marker: PhantomData<(D, Chip, Cluster, Slice, Element)>,
}
```

`Chip` / `Cluster` / `Slice` 는 소스에서 그대로 통과한다. `Element` 는 (슬라이스)별 레이아웃을 담는다.

#### Collect Engine 에서

`.to_vrf::<Element2>()` 는 `CollectTensor` 에서 flit 들을 VRF 에 저장해 `VrfTensor` 를 만들고, `.to_vrf_at::<Element2>(address)` 는 원시 `Address` 에 저장한다:

```rust,ignore
impl<'l, const T: Tu, P: CanApplyToVrf, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Stores to the vector register file.
    #[primitive(TuTensor::to_vrf)]
    pub fn to_vrf<Element: M>(self) -> VrfTensor<D, Chip, Cluster, Slice, Element, B> {
        VrfTensor::new(self.inner.transpose(false), None)
    }

    /// Stores to the vector register file at `address`.
    #[primitive(TuTensor::to_vrf_at)]
    pub fn to_vrf_at<Element: M>(self, address: Address) -> VrfTensor<D, Chip, Cluster, Slice, Element, B> {
        VrfTensor::new(self.inner.transpose(false), Some(address))
    }
}
```

`.to_vrf` 는 스트리밍 `Time` / `Packet` 을 `Element2` 로 평탄화한다:

```text
Element2 = [Time, Packet]
```

`Element2` 는 사용자가 고른다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![B = 64];

fn store_vrf<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![1 # 2], m![1 # 256], m![B / 8], m![B % 8]>,
) -> VrfTensor<i32, m![1], m![1 # 2], m![1 # 256], m![B]> {
    input.to_vrf()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, i32, m![1], m![1 # 2], m![1 # 256], m![B / 8], m![B % 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = store_vrf(c);
```

#### Data Memory 에서

완전히 연속적인 입력 접근(빈틈이나 재정렬이 없는 경우)에 대해 VRF 는 *short command*(StoVRF) 를 지원하는데, 이는 Data Memory 에서 VRF 로 데이터를 곧바로 적재해 Fetch → Switch → Collect → `to_vrf()` 파이프라인 전체를 우회하는 간결한 하드웨어 명령어다.
이 지름길은 임의 레이아웃 지원을 내주고 더 낮은 설정 오버헤드를 얻는다.


### 구조

