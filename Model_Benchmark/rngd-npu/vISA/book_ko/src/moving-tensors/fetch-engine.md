# Fetch Engine

Fetch Engine 은 DM 텐서를 읽어 Tensor Unit 을 위한 패킷 스트림을 만드는데, 이는 슬라이스별 시퀀서로 DM 을 읽고 `FetchTensor` 를 내보내는 [수학적 텐서 이동](../mapping-tensors/tensor-semantics.md#mathematical-tensor-move) 이다.

## 인터페이스

`BeginTensor` 는 Tensor Unit 파이프라인 입구에서 DM 에 상주하는 텐서를 나타낸다.
그 `Time` 은 `m![1]` 이고(파이프라인이 시작하기 전에는 시간 반복이 없다) `Packet` 은 DM 안의 원소 레이아웃이다.

`BeginTensor::fetch()` 는 시퀀서를 돌려 [Fetch Adapter](../computing-tensors/fetch-adapter.md), [Switch Engine](../computing-tensors/switch-engine.md), [Collect Engine](../computing-tensors/collect-engine.md) 으로 공급되는 `FetchTensor` 패킷 스트림을 만든다.
`assert_eq!` 호출이 `Cluster::SIZE`, `Slice::SIZE`, 패킷 정렬에 대한 하드웨어 제약을 강제한다([제약](#constraints) 참고).

```rust,ignore
impl<'l, const T: Tu, P: CanApplyFetch, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Fetch Sequencer.
    ///
    /// Configures per-slice DM reads and produces a `FetchTensor` with the
    /// chosen `OutTime` / `OutPacket`. The element type is unchanged. Type
    /// casts and other adapter transforms are applied by the per-stage
    /// `fetch_mask` / `fetch_table_lookup` / `fetch_cast` methods.
    #[primitive(TuTensor::fetch)]
    pub fn fetch<OutTime: M, OutPacket: M>(self) -> FetchTensor<'l, T, D, Chip, Cluster, Slice, OutTime, OutPacket, B> {
        verify_fetch::<Cluster, Slice, Time, Packet, OutTime, OutPacket>();
        FetchTensor::new(self.ctx, self.inner.transpose(true))
    }
}
```

[텐서 매핑](../mapping-tensors/index.md) 에서 소개했듯이, `Chip`, `Cluster`, `Slice`, `Time`, `Packet` 매핑은 데이터를 공간과 시간에 걸쳐 분배한다.
`.fetch()` 는 입력의 `Chip`, `Cluster`, `Slice` 차원을 그대로 보존하는데, 각 슬라이스가 자기 DM 파티션을 독립적으로 읽기 때문이다.
이후 [Switch Engine](../computing-tensors/switch-engine.md) 이 슬라이스 사이로 데이터를 옮기며 `Slice` 매핑을 바꾼다.

`fetch()` 는 Fetch Sequencer 를 설정하는 `OutTime` 과 `OutPacket` 타입 매개변수를 받는다.
`OutTime` 은 출력 스트림의 시간 스텝 수를 정하고, `OutPacket` 은 각 패킷 안의 원소 레이아웃을 정한다.
`OutPacket` 선택이 성능에 미치는 영향은 [최적화](#optimizations) 를 참고한다.

다음 예제는 DM 에서 `i8` 행렬을 `i8` 패킷 스트림으로 페치한다.
출력 `FetchTensor` 는 512 개의 시간 스텝을 스트리밍하며, 각각은 32 원소 `i8` 패킷(32 바이트)이다.
여기서 `OutTime = m![A]`, `OutPacket = m![B]` 다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![CH = 4, CL = 2, S = 256, A = 512, B = 32];

fn fetch_matrix_example<'l, const T: Tu>(
    input: BeginTensor<'l, T, i8, m![CH], m![CL], m![S], m![1], m![A, B]>,
) -> FetchTensor<'l, T, i8, m![CH], m![CL], m![S], m![A], m![B]> {
    input.fetch::<m![A], m![B]>()
}
```

`Chip`, `Cluster`, `Slice` 는 하드웨어 공간 병렬성 차원이다.
Fetch Sequencer 는 모든 슬라이스에서 독립적으로 돌며, 각각 자기 로컬 DM 파티션을 다룬다.
위 예제에서 `Chip = m![CH]`, `Cluster = m![CL]`, `Slice = m![S]` (`CH = 4`, `CL = 2`, `S = 256`) 는 칩당 클러스터 2 개, 클러스터당 슬라이스 256 개를 갖는 4 칩 RNGD 시스템(총 2,048 슬라이스)을 반영하며, 각 슬라이스는 자기 `A×B` 부분 텐서에 같은 시퀀서 패턴을 돌린다.


<a id="constraints"></a>
## 제약

- **하드웨어 차원**: `Chip::SIZE`, `Cluster::SIZE`, `Slice::SIZE` 는 하드웨어 구성과 일치해야 한다([Sequencer](./sequencer.md#architecture) 참고).

## 다중 읽기 패킷

패킷 축이 DM 에서 연속이 아닐 수 있고 하드웨어가 한 번에 최대 32 바이트만 읽으므로, 패킷 하나를 준비하는 데 여러 번의 하드웨어 읽기가 필요할 수 있다.
[main 컨텍스트](../computing-tensors/index.md#execution-context) 에서 `read_size` 는 시퀀서 `max_access_size` 의 최대 약수이며(`max_access_size` 는 [Sequencer 구조](./sequencer.md#access-size) 참고), `D[read_size]` 가 1, 2, 4, 8, 16, 32 바이트가 되도록 한다.
[sub 컨텍스트](../computing-tensors/index.md#execution-context) 에서 `read_size` 는 8 바이트로 고정된다.
컴파일러가 `read_size` 를 `fetch()` 의 입력 원소 타입(그리고 하류의 [Fetch Adapter](../computing-tensors/fetch-adapter.md) 캐스트)에서 유도하며, 사용자가 직접 설정하지 않는다.
`Packet::SIZE > read_size` 일 때마다 다중 읽기가 일어난다.
예를 들어 main 컨텍스트의 24 바이트 패킷은 `read_size = 8` 과 패킷당 3 번의 읽기를 강제한다.
총 사이클 수는 `Time::SIZE * (Packet::SIZE / read_size)` 다.

다음 예제들은 같은 `i4` 텐서, 곧 모양이 `m![N, C, H, W]` 인 텐서(`N=4, C=3, H=4, W=16`)를 서로 다른 네 가지 `OutPacket` 선택으로 페치한다.
```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![N = 4, C = 3, H = 4, W = 16];

/// Sequencer config: [N = 4 : 192, C = 3 : 64, H = 4 : 16, W = 16 : 1].
/// max_access_size = 16; read_size = 16 (8 bytes); reads per packet = 1; cycles = 48
fn fetch_batch_1<'l, const T: Tu>(
    input: BeginTensor<'l, T, i4, m![1], m![1 # 2], m![1 # 256], m![1], m![N, C, H, W]>,
) -> FetchTensor<'l, T, i4, m![1], m![1 # 2], m![1 # 256], m![N, C, H], m![W]> {
    input.fetch()
}

/// Sequencer config: [N = 4 : 192, C = 3 : 64, H / 2 = 2 : 32, H % 2 = 2 : 16, W = 16 : 1].
/// max_access_size = 32; read_size = 32 (16 bytes); reads per packet = 1; cycles = 24
fn fetch_batch_2<'l, const T: Tu>(
    input: BeginTensor<'l, T, i4, m![1], m![1 # 2], m![1 # 256], m![1], m![N, C, H, W]>,
) -> FetchTensor<'l, T, i4, m![1], m![1 # 2], m![1 # 256], m![N, C, H / 2], m![H % 2, W]> {
    input.fetch()
}

/// Sequencer config: [N = 4 : 192, C = 3 : 64, H = 4 : 16, W = 16 : 1].
/// max_access_size = 64; read_size = 64 (32 bytes); reads per packet = 1; cycles = 12
fn fetch_batch_3<'l, const T: Tu>(
    input: BeginTensor<'l, T, i4, m![1], m![1 # 2], m![1 # 256], m![1], m![N, C, H, W]>,
) -> FetchTensor<'l, T, i4, m![1], m![1 # 2], m![1 # 256], m![N, C], m![H, W]> {
    input.fetch()
}

/// Sequencer config: [N = 4 : 192, C = 3 : 64, H = 4 : 16, W = 16 : 1].
/// max_access_size = 192; read_size = 64 (32 bytes); reads per packet = 3; cycles = 12
fn fetch_batch_4<'l, const T: Tu>(
    input: BeginTensor<'l, T, i4, m![1], m![1 # 2], m![1 # 256], m![1], m![N, C, H, W]>,
) -> FetchTensor<'l, T, i4, m![1], m![1 # 2], m![1 # 256], m![N], m![C, H, W]> {
    input.fetch()
}
#
# let mut ctx = Context::acquire();
#
# let b: BeginTensor<'_, _, i4, m![1], m![1 # 2], m![1 # 256], m![1], m![N, C, H, W]> = BeginTensor::new(&mut ctx.main, Tensor::zero());
# let _o = fetch_batch_1(b);
#
# let b: BeginTensor<'_, _, i4, m![1], m![1 # 2], m![1 # 256], m![1], m![N, C, H, W]> = BeginTensor::new(&mut ctx.main, Tensor::zero());
# let _o = fetch_batch_2(b);
#
# let b: BeginTensor<'_, _, i4, m![1], m![1 # 2], m![1 # 256], m![1], m![N, C, H, W]> = BeginTensor::new(&mut ctx.main, Tensor::zero());
# let _o = fetch_batch_3(b);
#
# let b: BeginTensor<'_, _, i4, m![1], m![1 # 2], m![1 # 256], m![1], m![N, C, H, W]> = BeginTensor::new(&mut ctx.main, Tensor::zero());
# let _o = fetch_batch_4(b);
```

## 인터리빙

인터리빙은 매핑이 동일한 두 텐서를 하나의 시퀀서 연산으로 합쳐, 같은 계산에 두 텐서가 모두 필요할 때 오버헤드를 줄인다.
명시적인 `Time` 축이 두 텐서 사이의 교대를 인코딩한다.

다음 예제에서 main 컨텍스트는 `begin_interleaved()` 로 인터리브된 텐서를 만든다.
첫 번째 시간 반복은 `lhs` 에서, 두 번째는 `rhs` 에서, 세 번째는 다시 `lhs` 에서 페치하는 식이다.
한 번의 페치 연산에서 인터리브할 수 있는 텐서는 최대 두 개다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 16, B = 32, I = 2];

/// Interleaves two input tensors into a single packet stream.
/// Useful for operations like 'input1 + input2' in the Vector Engine.
/// The interleaved BeginTensor is created via Tu.begin_interleaved().
/// The `I = 2` axis in Time encodes alternation between the two tensors.
fn fetch_interleaved<'l>(
    ctx: &'l mut Context,
    lhs: &'l DmTensor<i8, m![1], m![1 # 2], m![1 # 256], m![A, B]>,
    rhs: &'l DmTensor<i8, m![1], m![1 # 2], m![1 # 256], m![A, B]>,
) -> FetchTensor<'l, { Tu::Main }, i8, m![1], m![1 # 2], m![1 # 256], m![A, I], m![B]> {
    ctx.main.begin_interleaved::<I, _, _, _, _, _>(lhs.view(), rhs.view()).fetch()
}
#
# let mut ctx = Context::acquire();
#
# let lhs = unsafe { DmTensor::from_addr(0) };
# let rhs = unsafe { DmTensor::from_addr(0) };
# let _o = fetch_interleaved(&mut ctx, &lhs, &rhs);
```

<a id="optimizations"></a>
## 최적화

세 가지 요인이 Fetch Sequencer 의 처리량을 결정한다.

- **입력 대역폭**: `read_size` 는 DM 안의 축 연속성과 패킷 크기에 제한된다.
  인접하지 않은 축은 `max_access_size` 를, 따라서 `read_size` 를 줄인다([비연속 패킷](./sequencer.md#non-contiguous-packets) 참고).
  연속 구간보다 작은 패킷도 `read_size` 를 제한한다.
  더 큰 2 의 거듭제곱으로 패딩하면 값이 올라간다([패킷 패딩](#example-packet-padding) 참고).

  나아가 같은 뱅크를 연속으로 64 번 이상 치는 접근 패턴은 우선순위가 낮은 [Commit Engine](./commit-engine.md) 과 [DMA Engine](./dma-engine.md) 을 굶기고 치명적인 NoC 타임아웃을 일으킬 수 있다.

  자세한 내용은 [메모리 성능](./memory-performance.md) 을 참고한다.
- **출력 대역폭**: 하류의 [Collect Engine](../computing-tensors/collect-engine.md) 이 Fetch 의 패킷을 32바이트 *flit* 으로 변환하므로, 32 바이트에 정렬되지 않는 패킷 크기는 대역폭을 낭비한다.
  20 바이트 패킷은 flit 하나를 12 바이트의 0 패딩으로 채워 `12 / 32 = 37.5%` 를 낭비한다.
  40 바이트 패킷은 flit 두 개(총 64 바이트)에 걸치며 두 번째 flit 의 마지막 24 바이트를 0 으로 패딩해 `24 / 64 = 37.5%` 를 낭비한다.
- **공간 병렬성**: 페치를 슬라이스에 걸쳐 분산하면 처리량이 최대가 된다.

<a id="example-packet-padding"></a>
### 예제: 패킷 패딩

`OutPacket` 을 더 큰 2 의 거듭제곱 원소 개수로 패딩하면 `read_size` 를 키울 수 있다.
아래 세 예제는 패킷을 2 바이트에서 16 바이트, 32 바이트로 키워 같은 30 바이트 텐서를 각각 15, 3, 1 사이클에 페치한다:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 3, B = 5, C = 2];

/// Smallest packet: only C dimension padded to 8bytes. Takes 15 cycles.
fn fetch_packet_C<'l, const T: Tu>(
    input: BeginTensor<'l, T, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![1], m![A, B, C]>,
) -> FetchTensor<'l, T, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![A, B], m![C # 8]> {
    input.fetch()
}

/// Medium packet: B and C dimensions padded to 16 bytes. Takes 3 cycles.
fn fetch_packet_BC<'l, const T: Tu>(
    input: BeginTensor<'l, T, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![1], m![A, B, C]>,
) -> FetchTensor<'l, T, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![A], m![[B, C] # 16]> {
    input.fetch()
}

/// Largest packet: all dimensions padded to 32 bytes. Takes 1 cycle.
fn fetch_packet_ABC<'l, const T: Tu>(
    input: BeginTensor<'l, T, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![1], m![A, B, C]>,
) -> FetchTensor<'l, T, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![1], m![[A, B, C] # 32]> {
    input.fetch()
}

#
# let mut ctx = Context::acquire();
# let x: BeginTensor<'_, _, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![1], m![A, B, C]> = BeginTensor::new(&mut ctx.main, Tensor::zero());
# let _o = fetch_packet_C(x);
# let y: BeginTensor<'_, _, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![1], m![A, B, C]> = BeginTensor::new(&mut ctx.main, Tensor::zero());
# let _o = fetch_packet_BC(y);
# let z: BeginTensor<'_, _, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![1], m![A, B, C]> = BeginTensor::new(&mut ctx.main, Tensor::zero());
# let _o = fetch_packet_ABC(z);
```

이 예제들에서 패딩은 실제 데이터 너머까지 읽지만, 패딩 값은 계산에 영향을 주지 않으므로 안전하다.
패딩 전략이 다르면 `FetchTensor` 매핑도 달라지며, 이는 하류 구성 요소에 영향을 줄 수 있다.
