# DMA Engine

DMA Engine 은 Tensor Unit 파이프라인을 거치지 않고 메모리 계층 사이에서 텐서를 직접 옮긴다.
각 전송은 서로 맞물린 두 단계를 짝지어 수행한다:
- **[Read Sequencer](#architecture)**: 소스 계층에서 읽는다.
- **[Write Sequencer](#architecture)**: 대상 계층에 쓰며, 레이아웃 변환을 동반할 수 있다.

DMA 전송은 [수학적 텐서 이동](../mapping-tensors/tensor-semantics.md#mathematical-tensor-move)이다: 레이아웃이 다르더라도 출력은 입력과 같은 수학적 텐서를 담는다.
Tensor DMA 는 DMN 간, 클러스터 간, 칩 간 전송을 아우르며, 칩 ID 는 시스템 전체에서 전역으로 합의된다.


전송 처리량과 관련된 고려사항은 [최적화](#optimizations)를 참고한다.

## 인터페이스

DMA 전송은 한 메모리 계층의 텐서를 받아 다른(또는 같은) 계층의 텐서를 만든다.
커널 작성자는 소스 텐서에 `.to_dm()`, `.to_hbm()` 또는 관련 메서드를 호출하면서 `DmaContext` 를 넘긴다:
- `Context::tdma`: 온칩 전송(HBM ↔ HBM, HBM ↔ DM, DM ↔ DM)을 위한 Tensor DMA 컨텍스트.
- `Context::pdma`: 호스트 ↔ HBM 전송을 위한 PCIe DMA 컨텍스트([PCIe DMA](#pcie-dma) 참고).

```rust,ignore
impl<D: Scalar, Chip: M, Element: M, B: Backend> HbmTensor<D, Chip, Element, B> {
    /// Converts to data memory tensor.
    #[primitive(HbmTensor::to_dm)]
    pub fn to_dm<Cluster: M, Slice: M, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        assert_dma_layout::<
            D,
            m![{ Chip }, { Element }],
            Element,
            m![{ Chip }, { Cluster }, { Slice }, { Element2 }],
            Element2,
        >(DMA_SRAM_WRITE_WIDTH);
        DmTensor::new(self.inner.transpose(true), None)
    }

    /// Converts to data memory tensor at `address`.
    #[primitive(HbmTensor::to_dm_at)]
    pub fn to_dm_at<Cluster: M, Slice: M, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
        address: Address,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        assert_dma_layout::<
            D,
            m![{ Chip }, { Element }],
            Element,
            m![{ Chip }, { Cluster }, { Slice }, { Element2 }],
            Element2,
        >(DMA_SRAM_WRITE_WIDTH);
        DmTensor::new(self.inner.transpose(true), Some(address))
    }

    /// Reshapes the tensor to a different mapping at the same HBM address, consuming `self`.
    /// The HBM analogue of [`DmTensor::reshape`]; both delegate to [`Tensor::reshape`].
    ///
    /// # Safety
    ///
    /// The per-level sizes (`Chip::SIZE == Chip2::SIZE`, `Element`) are asserted at compile time below
    /// (see [`constraints::assert_hbm_reshape_dimension_preserved`]); the genuine precondition is
    /// [`Tensor::reshape`]'s: the old and new mappings must lay the elements out in the SAME physical
    /// (wire) order, so the relabel moves no data. Axis regrouping (merge/split) preserves wire order
    /// and is valid; a permutation is not (use a transpose). Equal sizes do not guarantee this.
    /// Consuming `self` is the safety contract made explicit: the old-shaped handle cannot survive to
    /// alias the same HBM bytes under a conflicting mapping.
    #[primitive(HbmTensor::reshape)]
    pub unsafe fn reshape<Chip2: M, Element2: M>(self) -> HbmTensor<D, Chip2, Element2, B> {
        constraints::assert_hbm_reshape_dimension_preserved::<Chip, Chip2, Element, Element2>();
        let reshaped = unsafe { self.inner.reshape::<m![{ Chip2 }, { Element2 }]>() };
        HbmTensor::new(reshaped, self.address)
    }
}
```

컴파일러는 소스와 대상 텐서 타입에서 read 시퀀서와 write 시퀀서의 설정을 유도한다.
커널 작성자는 대상 타입의 `Cluster`, `Slice`, `Element`(DM 텐서의 경우) 또는 `Element`(HBM 텐서의 경우)를 지정하며, 여기에 레이아웃 변환이 인코딩된다.

아래 예제는 HBM-to-HBM 전송 두 번으로 텐서를 `[A, B, C]` 에서 `[C, A, B]` 로 전치한다:

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, B = 16, C = 32];

fn transpose_simple(
    ctx: &mut Context,
    input: &HbmTensor<f32, m![1], m![A, B, C]>,
) -> HbmTensor<f32, m![1], m![C, A, B]> {
    // Step 1: [A, B, C] → [A, C, B]
    let intermediate: HbmTensor<f32, m![1], m![A, C, B]> = input.to_hbm(&mut ctx.tdma);

    // Step 2: [A, C, B] → [C, A, B]
    intermediate.to_hbm(&mut ctx.tdma)
}
#
# let mut ctx = Context::acquire();
# let in_hbm = unsafe { HbmTensor::<f32, m![1], m![A, B, C]>::from_addr(0) };
# let _out_hbm = transpose_simple(&mut ctx, &in_hbm);
```

계층을 가로지르는 전송도 대상 타입의 매핑을 통해 레이아웃 변환을 받는다.
HBM-to-DM 전송에서 대상 DM 텐서는 `Cluster` 와 `Slice` 축을 추가하며, 이 축들이 텐서를 하드웨어 파티션에 분산하고 브로드캐스트한다.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 2048];

fn hbm_to_dm(
    ctx: &mut Context,
    input: &HbmTensor<i8, m![1], m![A]>,
) -> DmTensor<i8, m![1], m![1 # 2], m![A / 8], m![A % 8]> {
    input.to_dm::<m![1 # 2], m![A / 8], m![A % 8]>(&mut ctx.tdma)
}
#
# let mut ctx = Context::acquire();
# let in_hbm = unsafe { HbmTensor::<i8, m![1], m![A]>::from_addr(0) };
# let _out_dm = hbm_to_dm(&mut ctx, &in_hbm);
```

여기서 2,048 원소 벡터는 슬라이스당 256 원소(`Slice = m![A / 8]`)로 분산되고 슬라이스당 8 원소(`Element = m![A % 8]`)를 가지며, 2개 클러스터에 걸쳐 펼쳐진다.

<a id="architecture"></a>
## 구조

각 RNGD 칩은 DMN 쌍마다 하나씩 8개의 DMA Engine 을 가지며, 최대 8개의 독립적인 전송을 병렬로 수행한다.
단일 DMA Engine 은 짝을 이루는 read 시퀀서와 write 시퀀서를 구동하고, 텐서 이동은 aggregate 를 통해 여러 엔진에 걸쳐 퍼진다.
아래 소절들은 이 엔진의 정적 구조, 시퀀서 표현, 동적 동작, 컴파일러 유도, aggregate 연산을 설명한다.

> [!NOTE]
> SRAM-to-SRAM 전송에서는 DMA 가 SRAM 슬라이스 대역폭을 다 쓰지 못할 수 있으므로, Tensor Unit([Fetch](./fetch-engine.md) 및 [Commit](./commit-engine.md) Engine 경유)이 DMA 보다 효율적인 경우가 많다.
> 다만 실제로는 HBM 대역폭이 병목인 것이 보통이어서, HBM ↔ DM 전송에서는 이 격차가 덜 중요하다.

### 정적 구조

`Chip`, `Cluster`, `Slice` 는 하드웨어의 공간 병렬성 차원이다.
칩당 8개의 DMA Engine 은 서로 다른 메모리 구성요소 사이의 전송을 병렬로 처리한다(예: 엔진 #0 이 HBM ↔ DM 을 맡는 동안 엔진 #1 이 DM ↔ DM 을 맡는다).

각 DMA Engine 은 짝을 이루는 read 시퀀서와 write 시퀀서를 한 박자로 함께 구동한다.
read 시퀀서는 소스 주소를 훑고, write 시퀀서는 대상 주소를 훑는다.
레이아웃 변환이 같은 논리 원소가 소스 메모리와 대상 메모리에 나타나는 순서를 바꾸므로, 둘은 같은 루프 횟수를 공유하되 서로 다른 스트라이드와 베이스 주소를 쓴다.
컴파일러는 read 횟수와 write 횟수가 일치한다는 점을 이용해, 이 쌍을 루프 항목마다 스트라이드 쌍을 갖는 단일 시퀀서로 간결하게 표현한다.

컴파일러는 단일 텐서 이동을 사용 가능한 DMA Engine 들에 분배하는데, 작업을 칩·클러스터·슬라이스 차원을 따라 분할하고 각 분할을 하나의 DMA Engine 에 할당한다.
어떤 DMA Engine 이든 어떤 전송이든 처리한다.
로컬 DMN 접근이 DMN 간 접근보다 빠르므로, 기본적으로 컴파일러는 소스 DM 의 로컬 DMA Engine 을 고른다.
커널 작성자가 엔진을 명시적으로 지정할 수도 있다.

### 시퀀서 표현


컴파일러는 각 DMA Engine 의 작업을 소스·대상 주소 지정과 짝지어진 `DmaSequencer` 로 표현한다:

```rust,ignore
struct DmaSequencer {
    entries: Vec<DmaEntry>,
    stride0: u16,        // 1..=4096, per-iteration packet size in bytes
    source_base: usize,
    dest_base: usize,
}

struct DmaEntry {
    axis: AxisName,
    size: usize,
    source_stride: isize,
    dest_stride: isize,
}
```

레이아웃 변환이 같은 논리 원소를 소스 메모리와 대상 메모리 사이에서 재배열하므로, 각 항목은 `size` 는 공유하되 `source_stride` 와 `dest_stride` 는 따로 갖는 루프를 지정한다.
가장 안쪽 루프의 스트라이드 `stride0`(1 ~ 4,096 바이트)이 반복당 패킷 크기를 정한다.
완전한 DMA 명령은 시퀀서를 엔진의 위치 및 매체와 함께 묶는다:

```rust,ignore
struct DmaDescriptor {
    sequencer: DmaSequencer,
    source_media: Media,
    dest_media: Media,
}

struct DmnIndex {
    chip: ChipIndex,
    cluster_in_chip: ClusterInChipIndex,
    slice_in_cluster: SliceInClusterIndex,
}

enum Media {
    Hbm(ChipIndex),
    Dm(DmnIndex),
    Spm(DmnIndex),
}

enum Dtype {
    I4, I8, F8E4M3, F8E5M2, I16, Bf16, F16, I32, F32,
}
```

동종 aggregate 는 참여하는 모든 DMA Engine 에 걸쳐 매개변수화된 하나의 디스크립터 템플릿을 쓴다.
이종 aggregate 는 각 DMN 을 그 DMN 전용 디스크립터와 짝짓는 `HashMap<DmnIndex, DmaDescriptor>` 를 쓴다.
DM 텐서 명세는 정확한 메모리 위치를 식별하기 위해 매핑 표현식에 칩·클러스터·슬라이스를 포함해야 한다.

### 동적 동작

`DmaSequencer` 의 각 루프는 행 우선 순서로 카운터를 증가시키며, 스트라이드 쌍에서 read 주소와 write 주소를 유도한다.
`base = (0, 256³)` 인 아래 시퀀서의 경우:

```text
[
  A -> 256 : (65,536, 256),
  B -> 256 : (256, 65,536),
  C -> 256 : (1, 1),
] : 256
```

| 반복 `i`                   | 카운터        | read 주소 | write 주소                            |
|----------------------------|---------------|-----------|---------------------------------------|
| 0                          | `(0, 0, 0)`   | 0         | `write_base`                          |
| 1                          | `(0, 0, 1)`   | 1         | `1 + write_base`                      |
| ...                        | ...           | ...       | ...                                   |
| 255                        | `(0, 0, 255)` | 255       | `255 + write_base`                    |
| 256                        | `(0, 1, 0)`   | 256       | `65,536 + write_base`                 |
| `i = a·256² + b·256 + c`   | `(a, b, c)`   | `i`       | `256·a + 256²·b + c + write_base`     |

`stride0 = 256` 이면 하드웨어는 반복마다 256 바이트를 읽고 쓰므로, 반복 0 은 `(A, B, C) = (0, 0, 0..255)` 에 해당하는 값 전부를 하나의 패킷으로 처리한다.
전송은 대략 시작 지연 500 사이클에 데이터 전송 256 × 256 사이클을 더한 시간에 끝난다.

### 컴파일러 유도

소스와 대상 텐서 매핑(`In`, `Out`)과 스트림 모양(`Stream`)이 주어지면, 컴파일러는 read 시퀀서와 write 시퀀서를 유도한다:
- **Read 시퀀서**: `In` 을 스트림 모양에 투영해 소스 계층으로 향하는 루프별 스트라이드를 만든다.
- **Write 시퀀서**: `Out` 을 스트림 모양에 투영해 대상 계층으로 향하는 루프별 스트라이드를 만든다.
- **통합 시퀀서**: 둘을 병합해 각 항목이 read 스트라이드와 write 스트라이드를 짝으로 갖게 한다.
- **패킷 크기**: 연속 read/write 볼륨에서 `stride0` 를 추론한다.
  read 와 write 가 모두 연속된 256 바이트에 접근하면 최적 `stride0` 는 256 이다.

`axes![A=256, B=256, C=256]` 위에서 `Stream = m![A, B, C]` 로 레이아웃 변환 `m![A, B, C] → m![B, A, C]` 을 수행하는 경우, 컴파일러는 인덱스 관계 `m![A, B, C]::map(i) = i![A: i / 65,536, B: (i % 65,536) / 256, C: i % 256]`(표기법은 [매핑 표현식](../mapping-tensors/mapping-expressions.md) 참고)를 써서 다음을 유도한다:

```text
read_sequencer  = [
  A -> 256 : 65,536,
  B -> 256 : 256,
  C -> 256 : 1,
] : 256, HBM @ 0
write_sequencer = [
  A -> 256 : 256,
  B -> 256 : 65,536,
  C -> 256 : 1,
] : 256, HBM @ 256³
```

이 둘은 항목마다 스트라이드 쌍을 갖는 단일 `DmaSequencer` 로 합쳐진다.
통합된 `DmaSequencer` 의 각 방향(read 또는 write)은 명료함을 위해 스트라이드 쌍 괄호를 생략하고 그 방향의 단일 스트라이드 시퀀서로 표시할 수 있다.

### Aggregate 연산

aggregate 는 텐서 모양이 DMN 들에 고르게 나누어떨어지는지에 따라 두 형태 중 하나를 취한다.

모양이 고르게 나누어떨어지면 참여하는 모든 엔진이 *동종(homogeneous)* aggregate 로 동작한다.
모든 DMA Engine 이 같은 매개변수화된 스트림 환경(`Stream = { chip, cluster, slice, time, packet }`)을 쓰고 베이스 주소만 다르다.

모양이 고르게 나누어떨어지지 않으면 컴파일러는 *이종(heterogeneous)* aggregate 로 물러난다.
각 DMN 은 `StreamFn(chip, cluster, slice)` 를 통해 자기 스트림 환경을 받고, 경계 DMN 은 유효 영역을 넘어 쓰는 것을 피하려고 작업을 여러 DMA 명령으로 쪼갠다.
입력과 출력 매핑 환경(`In` 과 `Out`)은 동종 경우와 구조적으로 동일하게 유지되므로, 전체 논리 텐서 이동은 잘 정의된다.

두 형태 모두에 적용되는 정확성 불변식이 둘 있다:
- **동일 매체 타입**: 참여하는 모든 DMA Engine 은 같은 소스 매체와 대상 매체를 써야 한다.
- **단일 통합 매핑**: 입력 텐서 매핑 하나와 출력 텐서 매핑 하나가 전체 전송을 지배한다.

명령마다 각자의 시작 지연이 붙으므로, aggregate 를 동종으로 유지하도록 DMN 들에 고르게 나누어떨어지는 텐서 모양을 택하는 편이 좋다.

## 제약

DMA Engine 은 하드웨어 수준의 정렬 규칙과 패킷 크기 규칙을 강제한다.
이를 어기면 단순한 성능 저하가 아니라 정확성 오류나 하드웨어 예외가 발생한다.

- **주소 정렬**:

  | 계층 | Read | Write |
  |------|------|-------|
  | HBM | 1 바이트 | 1 바이트 |
  | DM (SRAM) | 1 바이트 | 8 바이트 |

  HBM ↔ DM 전송은 위 표와 무관하게 read 주소, write 주소, 패킷 크기에 대해 추가로 8 바이트 정렬을 요구한다.
  비대칭적인 DM 규칙은 비대칭적인 SRAM 하드웨어를 반영한다.
  read 포트는 바이트 선택 로직으로 임의의 바이트 범위를 뽑아내지만, write 포트는 8 바이트 뱅크 폭 단위 전체로 동작한다.
  따라서 정렬되지 않은 DM write 는 Read-Modify-Write 동작을 유발해 write 시간을 세 배로 늘리고 해당 뱅크의 다른 동작을 막는다.

  컴파일러는 이 제약들을 하드웨어 불변식으로 강제한다.

- **패킷 크기**: 최대 패킷 크기는 4,096 바이트이며, 트랜잭션이 256 beat × 16 바이트 데이터 폭을 넘을 수 없다는 AXI 프로토콜 제약으로 정해진다.

<a id="optimizations"></a>
## 최적화

DMA 처리량은 세 가지 요인이 결정한다: 메모리 대역폭, 채널 및 DMN 인터리빙, 그리고 패킷 분할을 동반한 시작 지연이다.

### 메모리 대역폭

각 계층에는 달성 가능한 처리량의 한계를 정하는 최대 대역폭이 있고, 실제 속도는 스트리밍 경로에서 가장 느린 구성요소로 제한된다.

| 계층 | 최대 대역폭 |
|------|----------------|
| HBM | 칩당 1.5 TB/s (0.75 GHz 에서 채널당 48 GB/s × 32 채널) |
| DM | 클러스터당 256 B/cycle (DMN 인터리빙 시 DMN 당 128 B/cycle) |
| SPM | 클러스터당 128 B/cycle (같은 칩 안에서만, 아직 API 에 노출되지 않음) |
| PCIe | read 와 write 모두 30 B/cycle ([PCIe DMA](#pcie-dma) 참고) |

DMA Engine 하나는 자체적으로 최대 256 B/cycle 을 옮긴다.
HBM 대역폭은 HBM 데이터를 전송하는 모든 엔진이 공유하므로, HBM 을 포화시키는 aggregate 는 엔진별 합이 아니라 1.5 TB/s 라는 HBM 최대치에 묶인다.

같은 클러스터 안의 DM-to-DM 전송은 read 와 write 두 단계가 같은 DM 뱅크 접근을 두고 경합하므로 직렬화된다.
HBM ↔ DM 같은 계층 간 전송은 read 단계와 write 단계를 파이프라인으로 겹친다.

> [!NOTE]
> SRAM-to-SRAM 전송에서는 DMA 가 SRAM 슬라이스 대역폭을 다 쓰지 못할 수 있으므로, Tensor Unit([Fetch](./fetch-engine.md) 및 [Commit](./commit-engine.md) Engine 경유)이 DMA 보다 효율적인 경우가 많다.
> 다만 실제로는 HBM 대역폭이 병목인 것이 보통이어서, HBM ↔ DM 전송에서는 이 격차가 덜 중요하다.

### 채널 및 DMN 인터리빙

최대 대역폭을 유지하려면 접근 패턴이 하부 메모리 파티션들에 걸쳐 인터리빙되어야 한다.

HBM 채널 선택은 주소 비트 9 ~ 28 을 쓰고, 주소 비트 8 은 스택 비트다.
요청을 32개 채널 전부로 퍼뜨리려면 접근 패턴이 이 비트들을 모두 토글해야 한다.
스택 비트(주소 비트 8) 하나만 빠져도 모든 요청이 32개 채널 중 16개로만 몰려 유효 대역폭이 절반이 된다.
같은 HBM 뱅크를 반복해서 때리는 접근 패턴(연속 접근에서 행 주소 비트 21 이상을 토글)은 접근당 약 40 사이클의 행 충돌 페널티를 유발해 대역폭을 한 자릿수 배로 떨어뜨린다.
FR-FCFS 메모리 스케줄링이 처리량을 일부 회복하지만, 근본 비용은 여전히 심각하다.

DM 대역폭은 두 DMN(각각 128 B/cycle)을 번갈아 써야 나오므로, 단일 DMN 접근 패턴은 DM 대역폭을 절반으로 만든다.

### 시작 지연과 패킷 분할

각 DMA 명령은 데이터 전송이 시작되기 전에 약 500 사이클의 고정 시작 지연을 치른다.
여러 전송을 더 적은 수의 명령으로 묶으면 이 비용이 분산되는 반면, 이종 aggregate 는 DMN 별 명령으로 쪼개져 명령마다 지연을 지불한다.

하나의 명령 안에서 하드웨어는 각 패킷을 256 바이트 단위로 쪼개므로, n 바이트 패킷은 `ceil(n / 256)` 개의 AXI 요청이 된다.
따라서 4,095 바이트 패킷은 16개의 요청이 들고, 4,099 바이트 패킷(소수 길이인 데다 4,096 바이트 한계를 어중간하게 넘는다)은 여러 명령으로 쪼개야 한다.
가장 안쪽 루프의 스트라이드(`stride0`)가 패킷 정렬을 결정한다: `stride0` 가 256 바이트 정렬이면 사이클 수는 `ceil(stride0 / 256)` 이다.
`stride0` 가 256 바이트 정렬이 아니면, HBM write 는 부분 256 바이트 블록에 대해 Read-Modify-Write 페널티를 추가로 치른다.
HBM read 는 `ceil` 오버헤드만 치르고, DM 동작은 이런 종류의 정렬 어긋남에 거의 영향을 받지 않는다.

DMA 는 DM 뱅크 접근 우선순위가 가장 낮으므로, Fetch 나 Commit Engine 이 같은 뱅크에 64회 이상 연속 접근하면 DMA 를 굶겨 NoC 타임아웃을 일으킬 수 있다.
자세한 내용은 [DM 뱅크 기아](./memory-performance.md#bank-starvation)를 참고한다.

## 상세 예제

아래 예제들은 대표적인 전송 패턴에 대해 구체적인 시퀀서 설정과 사이클 추정치를 제시한다.
예제 1 ~ 3 은 각 계층 쌍에 대해 잘 튜닝된 단일 엔진 사례를 다룬다.
예제 4 와 5 는 10배 이상 손해를 보는 병리적 접근 패턴을 대비시킨다.
예제 6 은 모양이 DMN 들에 고르게 나누어떨어지지 않을 때의 이종 분할을 보여준다.

시퀀서 설정에는 두 개의 주소 스트라이드 심볼을 쓴다:
- `slice_stride`: 슬라이스 내부 DM 파티션 하나의 가상 주소 범위(4 MB).
- `DMN_stride`: 같은 클러스터 안의 두 DMN 사이 주소 범위.

### 예제 1: HBM ↔ HBM 레이아웃 변환

인자:
- `axes![A = 8, B = 8, C = 256]`
- `dtype = i8`
- 소스: 오프셋 `0` 의 HBM, 매핑 `m![A, B, C]`
- 대상: 오프셋 `16,384` 의 HBM, 매핑 `m![B, A, C]`
- 스트림: time `m![A, B]`, packet `m![C]`

생성된 시퀀서:

```text
read = [
  A -> 8   : 2,048,
  B -> 8   : 256,
  C -> 256 : 1,
] : 256, HBM @ 0

write = [
  A -> 8   : 256,
  B -> 8   : 2,048,
  C -> 256 : 1,
] : 256, HBM @ 16,384
```

가장 안쪽이 아닌 스트라이드들(256 과 2,048)이 HBM 주소 비트 8(스택)과 11(채널)을 토글해, 모든 요청을 서로 다른 HBM 채널로 퍼뜨려 병렬로 실행되게 한다.
256 바이트 전송 하나는 0.75 GHz 에서 채널당 4 사이클이 걸리지만, 채널 병렬 분산이 최대치에 가까운 대역폭을 유지한다.
총 시간: 1 GHz 기준 read 요청 64회 + write 요청 64회에 시작 500 사이클을 더해 대략 628 사이클.

4개의 DMA Engine 이 HBM ↔ HBM 트래픽을 나눠 쓰면, 각 엔진은 0.75 TB/s 의 read 대역폭 중 약 0.1875 TB/s 를 받는다.
`stride0 = 256` 이더라도 그 몫으로는 어떤 엔진도 사이클당 요청 하나를 완료하지 못한다.

### 예제 2: 전대역폭 HBM → DM

이 계층 간 전송은 HBM 채널들과 두 DMN 에 걸쳐 인터리빙해 read 와 write 를 파이프라인으로 겹친다.

인자:
- `axes![A = 256, B = 256, C = 256]`
- `dtype = i8`
- 소스: 칩 0 의 HBM, 매핑 `m![B, A, C]`
- 대상: 칩 0, 클러스터 0, 슬라이스 0 의 DM. 슬라이스 매핑 `m![A / 4]`, 원소 매핑 `m![A % 4, B, C]`
- 스트림: time `m![B, A % 4, A / 4 % 32, A / 128]`, packet `m![C]`

생성된 시퀀서:

```text
read = [
  B      -> 256 : 65,536,
  A%4    -> 4   : 256,
  A/4%32 -> 32  : 1,024,
  A/128  -> 2   : 32,768,
  C      -> 256 : 1,
] : 256, HBM @ 0

write = [
  B      -> 256 : 256,
  A%4    -> 4   : 65,536,
  A/4%32 -> 32  : slice_stride,
  A/128  -> 2   : DMN_stride,
  C      -> 256 : 1,
] : 256, DM @ 0
```

HBM 쪽에서는 `A/128=2` 에 걸린 32,768 스트라이드가 채널 간 접근을 인터리빙하고, 하드웨어 명령 큐가 65,536개(256 × 4 × 32 × 2)의 요청을 모두 흐르게 유지한다.
DM 쪽에서는 `slice_stride` 와 `DMN_stride` 가 연속된 256 바이트 write 를 두 DMN 에 걸쳐 인터리빙해, 둘 다 사이클당 요청 하나를 유지한다.
read 와 write 가 계층을 가로질러 파이프라인되므로, 총 시간은 대략 max(read 65,536 사이클, write 65,536 사이클) + 시작 500 ≈ 66,036 사이클이다.

### 예제 3: 한 클러스터 안에서의 DM → DM

같은 클러스터 안의 DM-to-DM 전송은 read 와 write 가 같은 DM 뱅크 접근을 두고 경합하므로 직렬화된다.

인자:
- `axes![A = 256, B = 256, C = 256]`
- `dtype = i8`
- 소스: 칩 0, 클러스터 0, 슬라이스 0, 원소 오프셋 `0` 의 DM. 슬라이스 매핑 `m![A / 4]`, 원소 매핑 `m![A % 4, B, C]`
- 대상: 칩 0, 클러스터 0, 슬라이스 0, 원소 오프셋 `4·256·256` 의 DM. 슬라이스 매핑 `m![A / 4]`, 원소 매핑 `m![B, A % 4, C]`
- 스트림: time `m![B, A % 4, A / 4 % 32, A / 128]`, packet `m![C]`

생성된 시퀀서:

```text
read = [
  B      -> 256 : 1,
  A%4    -> 4   : 65,536,
  A/4%32 -> 32  : slice_stride,
  A/128  -> 2   : DMN_stride,
  C      -> 256 : 1,
] : 256, DM @ 0

write = [
  B      -> 256 : 1,024,
  A%4    -> 4   : 256,
  A/4%32 -> 32  : slice_stride,
  A/128  -> 2   : DMN_stride,
  C      -> 256 : 1,
] : 256, DM @ (4·256·256)
```

DMN 인터리빙과 슬라이스 인터리빙이 각 단계에 256 B/cycle 전부를 주지만, 두 단계는 직렬화된다.
총 시간: 대략 131,072 사이클(read 65,536 + write 65,536) + 시작 500.

> [!NOTE]
> 가능하면 `C` 를 256 의 배수로 고른다.
> `0 < r < 256` 인 `C = 256n + r` 의 경우, 전체 데이터 양은 조금밖에 달라지지 않는데도 각 접근이 더 많은 요청으로 쪼개지기 때문에 사이클 수가 `n+1` 배로 늘어난다.

### 예제 4: HBM 뱅크 충돌 병리 사례

여기서 DM 인터리빙은 건강하지만, 병리적인 HBM 접근 패턴이 잘 튜닝된 사이클 수의 약 10배를 치르게 만든다.

인자:
- 칩 1개(DMN 8개)
- `axes![A = 64, B = 2,048, C = 1,024]`
- `dtype = i8`
- 소스: HBM, 클러스터 매핑 `m![B / 1024]`, 슬라이스 매핑 `m![B / 256 % 4, A]`, 원소 매핑 `m![B % 256, C]`
- 대상: DM, 슬라이스 매핑 `m![A / 4]`, 원소 매핑 `m![B, A % 4, C]`
- 스트림: cluster `m![B / 1024]`, slice `m![B / 256 % 4]`, time `m![B % 256, C / 256, A % 32, A / 32]`, packet `m![C % 256]`

`(cluster_i, dmn_j)` 별로 생성된 시퀀서:

```text
read = [
  B%256 -> 256 : 1,024,
  C/256 -> 4   : 256,
  A%32  -> 32  : 2²¹,
  A/32  -> 2   : 2²⁶,
  C%256 -> 256 : 1,
] : 256, HBM @ (i·2²⁰ + j·2¹⁸)

write = [
  B%256 -> 256 : 1,024,
  C/256 -> 4   : 256,
  A%32  -> 32  : slice_stride,
  A/32  -> 2   : DMN_stride,
  C%256 -> 256 : 1,
] : 256, DM @ (cluster_i, dmn_j, 0)
```

`A%32` 와 `A/32` 에 걸린 스트라이드는 HBM 뱅크 안의 행을 선택하는 HBM 주소 비트 21 과 26 을 토글한다.
그래서 각 채널 안의 연속 접근이 거의 매 요청마다 한 행을 닫고 다음 행을 열어, 접근당 약 40 사이클을 치른다.
`C / 256 = 4`(스트라이드 256)를 통한 채널 인터리빙이 요청을 32개 채널 전부로 퍼뜨리기는 하지만, 각 채널 안의 행 충돌 비용은 감출 수 없다.

성능 분석:
- HBM read: 32개 채널에 걸쳐 총 524,288 요청 = 채널당 16,384 × 약 40 사이클 ≈ 655,360 사이클.
- DM write: DMN 당 65,536 요청을 사이클당 하나씩 처리하며, read 지연 아래에 감춰진다.

총 시간: 대략 655,360 사이클 + 시작 500 ≈ 655,860 사이클.
FR-FCFS 스케줄링이 처리량을 일부 회복하지만, 한 자릿수 배 규모의 페널티는 남는다.

### 예제 5: 스택 비트 누락 병리 사례

이 접근 패턴은 HBM 의 스택 차원(주소 비트 8)에 걸친 인터리빙에 실패해, 모든 트래픽을 채널의 절반으로 보내고 유효 대역폭을 절반으로 만든다.

인자:
- 칩 1개(DMN 8개)
- `axes![A = 8, B = 64, C = 8, D = 512]`
- `dtype = i8`
- 소스: HBM, 매핑 `m![A, B, C, D]`
- 대상: DM, 클러스터 매핑 `m![A / 4]`, 슬라이스 매핑 `m![A % 4, B]`, 원소 매핑 `m![C, D % 256]`
- 스트림: cluster `m![A / 4]`, slice `m![A % 4]`, time `m![C, B % 32, B / 32]`, packet `m![D % 256]`

`(cluster_i, dmn_j)` 별로 생성된 시퀀서:

```text
read = [
  C     -> 8   : 512,
  B%32  -> 32  : 4,096,
  B/32  -> 2   : 131,072,
  D%256 -> 256 : 1,
] : 256, HBM @ (i·2²⁰ + j·2¹⁸)

write = [
  C     -> 8   : 256,
  B%32  -> 32  : slice_stride,
  B/32  -> 2   : DMN_stride,
  D%256 -> 256 : 1,
] : 256, DM @ (cluster_i, dmn_j, 0)
```

`C` 의 스트라이드 512 는 HBM 주소 비트 8(스택 비트)을 전혀 토글하지 않으므로, 8개의 DMN 이 32개 HBM 채널 중 16개에 몰린다.

성능 분석:
- HBM read(병목): 16개 채널에 걸쳐 총 4,096 요청 = 채널당 256 × 1 GHz 기준 요청당 약 5.3 사이클 ≈ 1,357 사이클.
- DM write: DMN 당 512 요청이며, read 아래로 파이프라인된다.

총 시간: 대략 1,357 사이클 + 시작 500 ≈ 1,857 사이클.
32개 채널 전부에 스택 비트 인터리빙을 되살리면 HBM 사이클 수가 절반이 된다.

### 예제 6: 이종 DMN 분할

텐서 모양이 DMN 들에 고르게 나누어떨어지지 않으면, 컴파일러는 경계 DMN 의 작업을 여러 명령으로 쪼개고 각 명령은 자기 몫의 시작 지연을 치른다.

인자:
- 칩 4개
- `axes![A = 15, B = 32, C = 256, D = 8]`
- `dtype = i8`
- 소스: DM, (`A' = A + 1#` 로 표기하여) 칩 매핑 `m![D / 2]`, 클러스터 매핑 `m![D % 2]`, 슬라이스 매핑 `m![A' / 4, A' / 2 % 2, B]`, 원소 매핑 `m![A' % 2, C]`
- 대상: HBM, 칩 매핑 `m![D / 2]`, 원소 매핑 `m![D % 2, B, A, C]`
- 스트림(DMN 별, `StreamFn(chip_i, cluster_j, slice_k)` 로 표현):

```text
StreamFn(chip_i, cluster_j, slice_k) = let A' = A + 1# in
  { chip: m![(D / 2) @ i = 1], cluster: m![(D % 2) @ j = 1],
    slice: m![(A' / 4) @ k = 1],
    time: (k == 0,1,2): m![A' % 2, B, A' / 2 % 2, C]
          (k == 3, exec #0): m![A' % 2, B, A' / 2 = 1, C]
          (k == 3, exec #1): m![A' = 1, B, A' / 2 % 2 @ 1, C],
    packet: m![C] }
```

차원 `A = 15` 는 4개의 DMN 으로 나누어떨어지지 않으므로(15 = 3·4 + 3), DMN 0 ~ 2 는 각각 4개 원소를 처리하고 DMN 3 은 3개만 처리한다.
DMN 3 에 디스크립터를 하나만 두면 네 번째 원소를 유효 영역 너머로 쓰게 되므로, 컴파일러는 DMN 3 의 작업을 합쳐서 정확히 3개 원소를 덮는 두 개의 명령으로 쪼갠다.

성능 분석:
- DMN 0 ~ 2(각각 명령 1개): 약 256 사이클 + 시작 500 ≈ 756 사이클.
- DMN 3(명령 2개): 데이터 약 192 사이클 + 시작 1,000(각 500) ≈ 1,192 사이클.

총 시간: DMN 3 에 발목이 잡혀 대략 1,192 사이클.
이 분할 비용을 피하려면 DMN 들에 고르게 나누어떨어지는 텐서 모양을 고른다.

## Shuffle 연산

Shuffle 연산은 파티션별 소스 패턴에 따라 텐서를 클러스터들 또는 칩들에 재분배한다.
메서드는 `to_dm` / `to_hbm` 관례에 맞춰 소스 텐서에서 이어 붙인다: `dm_cluster_shuffle` 과 `dm_chip_shuffle` 은 `DmTensorView` 에, `hbm_cluster_shuffle` 과 `hbm_chip_shuffle` 은 `HbmTensor` 에 있다.
shuffle 패턴은 각 대상 클러스터 또는 칩에 대해 어느 소스 클러스터 또는 칩이 그 데이터를 제공하는지 지정한다.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 256, B = 4096];

fn cluster_shuffle(
    ctx: &mut Context,
    input: &DmTensor<i32, m![A / 4 % 4], m![A / 2 % 2], m![B % 16, B / 16 % 16], m![B / 256, A % 2, A / 16]>,
) -> DmTensor<i32, m![A / 4 % 4], m![A / 2 % 2], m![B % 16, B / 16 % 16], m![B / 256, A % 2, A / 16]> {
    // Shuffle pattern [1, 0]: cluster 0 ↔ cluster 1
    input.view().dm_cluster_shuffle::<2>(&mut ctx.tdma, &[1, 0])
}
# let mut ctx = Context::acquire();
# let input_dm = unsafe { DmTensor::<i32, m![A / 4 % 4], m![A / 2 % 2], m![B % 16, B / 16 % 16], m![B / 256, A % 2, A / 16]>::from_addr(0) };
# let _output_dm = cluster_shuffle(&mut ctx, &input_dm);
```

칩 간 shuffle 은 시스템 전역의 칩 ID 를 쓴다.
`hbm_chip_shuffle` 은 DMA 컨텍스트(`tdma` 또는 `pdma`)에 대해 제네릭인데, 칩 간 연산이 HBM ↔ HBM 이고 HBM ↔ HBM 이 Tensor DMA 와 PCIe DMA 가 모두 지원하는 유일한 DMA 쌍이기 때문이다.
나머지 shuffle 메서드들, 그리고 일반적으로 HBM ↔ DM 이나 DM ↔ DM 같은 DMA 쌍은 컨텍스트 제네릭이 아니다.

## Scatter 와 Gather

Scatter 와 gather 는 고정된 스트라이드가 아니라 인덱스 텐서에서 계산한 주소로 텐서 원소를 옮긴다.

`DmTensor::dma_scatter` 는 DM 값을 인덱스 텐서가 고른 HBM 행에 쓴다.
`HbmTensor::dma_gather_scaled` 와 `HbmTensor::dma_gather_unscaled` 는 HBM 행을 읽어 인덱스 텐서가 고른 행 위치로 DM 에 넣는다.
두 gather 변형은 인덱스가 어디에 있는지와 그 값을 어떻게 읽는지만 다르며, 아래에서 설명한다.

```rust
# extern crate furiosa_opt_std;
# extern crate tokio;
# use furiosa_opt_std::prelude::*;
axes![K = 512, D = 128, C = 612, G = 512, CL = 2];

fn scatter_minimal(
    ctx: &mut Context,
    data: &HbmTensor<bf16, m![1], m![K, D]>,
    index: &HbmTensor<i32, m![1], m![K]>,
    output: &mut HbmTensor<bf16, m![1], m![C, D]>,
) {
    let data_dm: DmTensor<bf16, m![1], m![1 # 2], m![K / 2], m![K % 2, D]> =
        data.to_dm(&mut ctx.tdma);

    data_dm.dma_scatter::<m![K], _, _>(index, output);
}

fn gather_minimal(
    table: &HbmTensor<bf16, m![1], m![K, D]>,
    index: &HbmTensor<i32, m![1], m![G]>,
    // The gather axis itself partitions into Slice x Element (`G / 2 = 256`, a valid slice count)
    // with `D` folded into the Element side alongside the `G % 2` remainder; `C = 612` (used above
    // for the scatter cache) has no divisor landing on a valid 64 | 128 | 256 slice count, so the
    // gather count here is the separate, slice-friendly `G` instead.
) -> DmTensor<bf16, m![1], m![1 # 2], m![G / 2], m![G % 2, D]> {
    table.dma_gather_scaled(index)
}

fn gather_unscaled(
    ctx: &mut Context,
    table: &HbmTensor<bf16, m![1], m![K, D]>,
    // Raw row positions per cluster. The gather reads the index off SPM, so the kernel first
    // stages it on-chip with `to_dm`; a real per-cluster (`CL`) partition avoids broadcast padding.
    index: &HbmTensor<i32, m![1], m![CL, G]>,
) -> DmTensor<bf16, m![1], m![CL], m![G / 2], m![G % 2, D]> {
    let index_dm: DmTensor<i32, m![1], m![CL], m![G / 2], m![G % 2]> =
        index.to_dm(&mut ctx.tdma);
    table.dma_gather_unscaled(&index_dm)
}
#
# #[tokio::main]
# async fn main() {
#     let mut ctx = Context::acquire();
# 
#     let index = &(HostTensor::<i32, m![K]>::zero().to_hbm(&mut ctx.pdma).await);
#     let data = unsafe { HbmTensor::<bf16, m![1], m![K, D]>::from_addr(0) };
#     let mut output_hbm = unsafe { HbmTensor::<bf16, m![1], m![C, D]>::from_addr(0) };
# 
#     scatter_minimal(&mut ctx, &data, &index, &mut output_hbm);
#     gather_minimal(&data, &(HostTensor::<i32, m![G]>::zero().to_hbm(&mut ctx.pdma).await));
#     let placed_index = &(HostTensor::<i32, m![CL, G]>::zero().to_hbm(&mut ctx.pdma).await);
#     gather_unscaled(&mut ctx, &data, placed_index);
# }
```

스케일 적용 변형(`dma_gather_scaled`, `dma_scatter`)은 인덱스를 DRAM 의 `HbmTensor` 에서 받아 그 값을 gather/scatter 축을 따르는 바이트 오프셋으로 읽는다: 행 `r` 을 지정하려면 `r` 에 한 행의 바이트 크기(그 행의 원소 개수 곱하기 원소의 바이트 크기, 예를 들어 폭 128 의 `bf16` 행이면 `128 * 2 = 256`)를 곱해 넘긴다.
반면 `dma_gather_unscaled` 는 `to_dm` 으로 DRAM 에서 온칩에 올려둔 `DmTensor` 인덱스를 받아 그 값을 가공되지 않은 행 위치로 읽으며, 페이지드 어텐션 블록 테이블처럼 온칩에서 계산된 인덱스에 쓴다.
부수 효과에 유의한다: seed 가 인덱스를 SPM 에서 읽으므로, 컴파일러는 인덱스를 올려두기 위해 DM 에서 SPM 으로 가는 DMA 를 추가로 내보낸다(SPM 은 앞서 언급한 온칩 계층이며, 아직 사용자에게 노출되는 타입은 아니다).
그 scatter 대응물인 `dma_scatter_unscaled` 는 아직 구현되지 않았다.

<a id="pcie-dma"></a>
## PCIe DMA

PCIe DMA(`Context::pdma`)는 호스트 시스템 메모리와 디바이스 HBM 사이에서 텐서를 옮긴다.
이는 온칩 Tensor DMA 와는 별개의 물리 엔진이다.
PCIe DMA 는 호스트 ↔ HBM 만 처리하고, Tensor DMA 는 모든 온칩 전송을 처리한다.

커널 작성자는 `HostTensor` 에 `.to_hbm()`(호스트 → 디바이스)을, `HbmTensor` 에 `.to_host()`(디바이스 → 호스트)를 호출한다.
둘 다 비동기 연산이다.

```rust,ignore
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use rand::{rngs::SmallRng, SeedableRng};
axes![A = 8, B = 512];

async fn upload_and_download(ctx: &mut Context) {
    let mut rng = SmallRng::seed_from_u64(0);
    let host: HostTensor<i8, m![A, B]> = HostTensor::rand(&mut rng);

    // Host → HBM (allocator-assigned address)
    let hbm: HbmTensor<i8, m![A], m![B]> = host.to_hbm(&mut ctx.pdma).await;

    // HBM → host (back to system memory)
    let _round_tripped: HostTensor<i8, m![A, B]> = hbm.to_host(&mut ctx.pdma).await;
}
```

`HostTensor` 는 `Element` 매핑만 지니지만(호스트 메모리에는 칩/클러스터/슬라이스 분할이 없다), 대상 `HbmTensor` 는 칩들에 분산하기 위해 `Chip` 축을 추가한다.
`to_hbm` 이 새로운 `Chip` 및 `Element` 타입 매개변수를 받으므로, HBM 의 대상 원소 레이아웃은 호스트 레이아웃과 다를 수 있다.

PCIe DMA 대역폭은 30 B/cycle 로, 온칩 Tensor DMA(256 B/cycle)보다 한 자릿수 배 느리다.
알고리즘은 호스트 ↔ 디바이스 트래픽을 최소화하고, 데이터를 한 번 업로드해 여러 온칩 연산에서 재사용해야 한다.
