# 공간 차원과 시간 차원

`HostTensor<D, E>` 는 단일 매핑으로 자신의 레이아웃을 온전히 담는다.
디바이스 텐서는 레이아웃을 여러 전용 차원으로 나눠 담는다.

- **공간 차원**: `Chip`, `Cluster`, `Slice` 가 데이터를 하드웨어 계층에 분산한다. 스트림 텐서에서는 `Packet` 이 추가로 각 시간 반복 안의 병렬 전달량을 정한다.
- **시간 차원**: `Time` 이 스트림 텐서에서 전달 반복의 순서를 매긴다.

## 공간 차원

하드웨어 계층의 각 공간 수준은 텐서 타입에서 자신만의 타입 매개변수를 가지며, 이로써 공간 병렬성이 가능해진다.
각 수준의 모든 유닛은 같은 매핑을 공유한다고 가정한다.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use std::marker::PhantomData;
// Assumed throughout this page.
axes![A = 8, B = 512];

// HBM tensors
struct HbmTensor<D: Scalar, Chip: M, Element: M> {
    /* ... */
    # _marker: PhantomData<(D, Chip, Element)>,
}

// SRAM tensors
// DM (Data Memory), TRF (Tensor Register File), and VRF (Vector Register File)
struct DmTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M> {
    /* ... */
    # _marker: PhantomData<(D, Chip, Cluster, Slice, Element)>,
}
struct TrfTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Element: M> {
    /* ... */
    # _marker: PhantomData<(D, Chip, Cluster, Slice, Lane, Element)>,
}
struct VrfTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M> {
    /* ... */
    # _marker: PhantomData<(D, Chip, Cluster, Slice, Element)>,
}
```

HBM 텐서는 공간 병렬성을 위해 데이터를 칩들에 분산한다. 각 칩이 자기 몫의 데이터를 동시에 처리한다.
예를 들어 `HbmTensor<bf16, m![A], m![B]>` 는 `8 × 512 = 4096` 개 원소를 8개 칩에 칩당 512개씩 분산한다.
`i` 번째 칩의 `j` 번째 원소는 텐서 인덱스 `i![A: i, B: j]` 를 담는다.

SRAM 텐서 타입은 더 세밀한 병렬성을 위해 `Cluster` 와 `Slice` 차원을 더한다.
`TrfTensor` 는 추가로 `Lane` 차원을 가져 TRF 데이터를 슬라이스당 8개 레인에 분산한다.
자세한 내용은 [Contraction Engine](../computing-tensors/contraction-engine/index.md) 을 보라.

모든 저장 텐서(`HostTensor`, `HbmTensor`, 그리고 SRAM 타입들)는 원소 데이터를 시작 주소에 배치한다.
예를 들어 주소 `addr` 에 있는 `DmTensor<D, ..., Element>` 는 `addr..(addr + Element::SIZE * size_of::<D>())` 바이트를 차지하며, `TrfTensor` 와 `VrfTensor` 도 같은 방식을 따른다.

### 제약

- **`Chip`, `Cluster`, `Slice` 크기**: 하드웨어 개수와 정확히 일치해야 한다.

  | 단위      | 개수            | 제약                  | 패딩 예시        |
  |-----------|------------------|-----------------------------|------------------------|
  | `Chip`    | 시스템에 따라 다름 | `Chip::SIZE == NUM_CHIPS`   | `m![1 # NUM_CHIPS]`    |
  | `Cluster` | Chip 당 2         | `Cluster::SIZE == 2`        | `m![1 # 2]`            |
  | `Slice`   | Cluster 당 256    | `Slice::SIZE == 256`        | `m![X / N # 256]`      |

  커널이 하드웨어가 제공하는 것보다 적은 유닛을 쓸 때는 어떤 차원이든 `#` 로 패딩할 수 있다.
  예를 들어 `type Cluster = m![1 # 2]` 는 활성 클러스터 1개와 패딩 전용 클러스터 1개를 써서 칩당 2 클러스터라는 하드웨어 요구를 만족한다.

  > [!NOTE]
  > 런타임은 칩 단위로 동작하므로(`#[device(chip = N)]`), 칩이나 클러스터의 부분 사용은 아직 지원하지 않는다.
  > 이는 향후 릴리스에서 완화될 수 있다.

- **`Element` 크기**: `Element::SIZE * size_of::<D>()` 는 유닛당 SRAM 용량을 넘어서는 안 되며, 이 용량은 텐서 타입에 따라 다르다.

  | 타입        | 단위          | 제약                                    |
  |-------------|---------------|-----------------------------------------------|
  | `DmTensor`  | Slice 당 512KB | `Element::SIZE * size_of::<D>() <= 512KB`     |
  | `TrfTensor` | Lane 당 8KB     | `Lane::SIZE <= 8`, `Element::SIZE * size_of::<D>() <= 8KB` |
  | `VrfTensor` | Slice 당 8KB   | `Element::SIZE * size_of::<D>() <= 8KB`      |

- **`Element` 정렬**: 시작 주소는 `size_of::<D>()` 의 배수여야 한다. 비정렬 쓰기는 read-modify-write 사이클을 필요로 해 DM 접근을 대략 50× 느리게 만들 수 있기 때문이다.

## 시간 차원

`TuTensor` 는 [Tensor Unit](../computing-tensors/index.md) 을 스트림으로 흐르는 텐서 데이터를 나타낸다.
SRAM 타입과 같은 `Chip`, `Cluster`, `Slice` 차원을 유지하면서 스트리밍을 위해 `Time` 과 `Packet` 을 더한다.
`Time` 이 시간 차원이다. 전달 반복의 순서를 매긴다.
공간 차원과 달리 `Time` 에는 하드웨어가 부과하는 크기 한도가 없으며 처리할 데이터 양에 따라 커진다.
`Packet` 은 각 슬라이스가 시간 반복마다 몇 개의 원소를 받는지를 결정하는 추가 공간 차원이다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use std::marker::ConstParamTy;
# use std::marker::PhantomData;
axes![N = 4, C = 64, H = 32, W = 32];

/// Pipeline stage.
/// `Vector` is intentionally absent: the Vector Engine uses a separate typestate
/// (`VectorBranchTensor` and friends) that tracks branch, ALU, and other Vector-specific state.
/// `Commit` is intentionally absent: once the Commit Engine writes results back to DM,
/// the data is at rest and the type becomes `DmTensor`, not `TuTensor`.
# #[derive(PartialEq, Eq, ConstParamTy)]
enum Position {
    Begin,       // After the start of the pipeline
    Fetch,       // After the Fetch Engine
    Switch,      // After the Switch Engine
    Collect,     // After the Collect Engine
    Contraction, // After the Contraction Engine
    Reduce,      // After the Reduce Engine
    Cast,        // After the Cast Engine
    Transpose,   // After the Transpose Engine
}

struct TuTensor<
    'l,                // Lifetime tied to the Tensor Unit context
    const P: Position,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
> {
    /* ... */
    #  _marker: PhantomData<&'l (D, Chip, Cluster, Slice, Time, Packet)>,
}

type T<'l> = TuTensor<
    'l,
    { Position::Fetch }, // Fetch Engine's output
    bf16,
    m![1],           // Chip: single chip
    m![1],           // Cluster: single cluster
    m![C / 2],       // Slice: distribute 64 channels across 32 slices
    m![N, H, W],     // Time: iterate over batch (N) and spatial (H, W) dimensions
    m![C % 2],       // Packet: 2 channels per cycle
>;
```

타입 `T` 는 총 형상이 \\(\\{N=4, C=64, H=32, W=32\\}\\) 인 텐서를 32개 슬라이스에 걸쳐 스트리밍한다(`Slice::SIZE = m![C / 2]::SIZE = 32`).
`Time` 차원(`m![N, H, W]`)의 크기는 `4 * 32 * 32 = 4096` 이며, 이는 시간 반복이 4,096회라는 뜻이다.
각 시간 반복마다 `Packet` 차원 `m![C % 2]` 는 각 슬라이스에 2개 채널을 전달한다.
32개 슬라이스가 병렬로 동작하므로, 각 시간 반복은 총 `32 * 2 = 64` 개 채널을 처리한다.
