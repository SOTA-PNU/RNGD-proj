# Sequencer

sequencer 는 메모리 버퍼를 패킷 스트림으로 읽고, 패킷 스트림을 메모리로 되쓴다.
Fetch Engine 과 Commit Engine 은 각각 sequencer 하나를 써서 DM 을 주소 지정한다.
DMA Engine 은 읽기 sequencer 와 쓰기 sequencer 를 사슬로 이어, 중간 버퍼 없이 DM, SPM, HBM 사이로 데이터를 옮긴다.

## 인터페이스

`MemTensor` 와 `StreamTensor` 는 버퍼-스트림 패턴만 따로 떼어 담은 교육용 의사 타입이며, 덕분에 이 페이지는 각 엔진의 전체 타입 기계를 끌어들이지 않고 sequencer 동작 원리를 설명할 수 있다.
실제 엔진 API 는 다른 타입(`DmTensor`, `HbmTensor`, `TuTensor`, …)을 쓰지만, 모든 구체적인 쌍은 아래에 그려진 동일한 `MemTensor` → `StreamTensor` 형태로 대응된다.

`MemTensor` 는 어떤 메모리 매핑 `Buf` 에 데이터를 담으며, 구체적인 버퍼 텐서(DM, SPM, HBM)는 무엇이든 이 역할을 한다.

```rust,ignore
/// A generic buffer-backed tensor.
/// Anything that holds data in a memory `Buf` and can be streamed in or out.
#[derive(Debug)]
pub struct MemTensor<D: Scalar, Buf: M> {
    inner: Tensor<D, Buf>,
}
```

`StreamTensor` 는 전송 중인 텐서다.
수명 `'l` 은 스트림을 원본 버퍼에 묶어, 스트림이 자기 데이터보다 오래 살아남지 못하게 한다.
`Time` 은 시간 매핑(시간에 걸친 반복)이고 `Packet` 은 공간 매핑(단일 패킷의 내용)이다.

```rust,ignore
/// A streaming view of a tensor in flight.
/// `Packet` is the per-cycle shape and `Time` is the multi-cycle shape.
#[derive(Debug)]
pub struct StreamTensor<'l, D: Scalar, Time: M, Packet: M> {
    inner: Tensor<D, Pair<Time, Packet>>,
    _marker: PhantomData<&'l ()>,
}
```

`read` 는 `MemTensor` 를 `StreamTensor` 로 변환하고 `write` 는 그 반대이며, 둘 다 값을 보존한다.
각 엔진의 전체 API 는 그 위에 공간 차원(`Chip`, `Cluster`, `Slice`)을 더하며, 이는 엔진별 페이지에서 다룬다.

```rust,ignore
impl<D: Scalar, Buf: M> MemTensor<D, Buf> {
    /// Reads a stream from this buffer with the supplied `Time` and `Packet` mapping.
    /// `(Time, Packet)` may be a broadcast of `Buf` (matches Fetch / Switch / DMA-read behavior).
    pub fn read<'l, Time: M, Packet: M>(&'l self) -> StreamTensor<'l, D, Time, Packet> {
        StreamTensor {
            inner: self.inner.transpose(true),
            _marker: PhantomData,
        }
    }

    /// Writes a stream back into this buffer.
    /// Broadcast is rejected: each `Buf` slot must have exactly one source position in `(Time, Packet)` (matches Commit behavior).
    pub fn write<'l, Time: M, Packet: M>(&mut self, stream: StreamTensor<'l, D, Time, Packet>) {
        self.inner = stream.inner.transpose(false);
    }
}
```

어떤 `MemTensor` 에 대해서도 유효한 `Time` 과 `Packet` 조합은 여럿 존재하며, 각각 서로 다른 `StreamTensor` 를 만든다.
유효한 선택지 중에서는 `Packet` 크기가 클수록 대역폭 활용이 좋아지며, [메모리 성능](./memory-performance.md)이 그 절충을 자세히 다룬다.


## 예제

다음 예제들은 위의 핵심 API 를 사용한 흔한 읽기·쓰기 패턴을 보여 준다.
아래 [구조](#architecture)는 컴파일러가 각 패턴의 하드웨어 설정을 어떻게 도출하는지 설명한다.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{MemTensor, StreamTensor};
axes![A = 8, B = 512, N = 4, C = 3, H = 8, W = 8, T = 4, P = 4];

/// Strided access: read 8×512 tensor as 128 packets of 32 elements.
/// Time = m![A, B / 32] produces 8 * 16 = 128 time steps.
/// Packet = m![B % 32] delivers 32 consecutive elements per packet.
fn strided_read<'l>(
    buf: &'l MemTensor<bf16, m![A, B]>,
) -> StreamTensor<'l, bf16, m![A, B / 32], m![B % 32]> {
    buf.read()  // Automatic type inference
}

/// Strided write: write 128 packets of 32 elements back to 8×512 tensor.
fn strided_write(
    buf: &mut MemTensor<bf16, m![A, B]>,
    stream: StreamTensor<bf16, m![A, B / 32], m![B % 32]>,
) {
    buf.write(stream)
}

/// Axis reordering read: change traversal from [N, C, H, W] to [W, H, C, N].
/// Time = m![W, H, C, N] iterates in reversed axis order.
/// Packet = m![1] delivers single-element packets.
fn axis_reordering_read<'l>(
    buf: &'l MemTensor<bf16, m![N, C, H, W]>,
) -> StreamTensor<'l, bf16, m![W, H, C, N], m![1]> {
    buf.read()
}

/// Axis reordering write: write [W, H, C, N] stream back to [N, C, H, W] buffer.
fn axis_reordering_write(
    buf: &mut MemTensor<bf16, m![N, C, H, W]>,
    stream: StreamTensor<bf16, m![W, H, C, N], m![1]>,
) {
    buf.write(stream)
}

/// Tiling read: break axes into sub-blocks for cache efficiency.
/// Time = m![A % 2, B % 4, A / 2, B / 4] tiles A into 2 × 4, B into 4 × 128 blocks.
/// Packet = m![C # 32] pads C to 32 elements per packet.
fn tiling_read<'l>(
    buf: &'l MemTensor<i8, m![A, B, C # 8]>,
) -> StreamTensor<'l, i8, m![A % 2, B % 4, A / 2, B / 4], m![C # 32]> {
    buf.read()
}

/// Tiling write: write tiled stream back to buffer.
fn tiling_write(
    buf: &mut MemTensor<i8, m![A, B, C # 8]>,
    stream: StreamTensor<i8, m![A % 2, B % 4, A / 2, B / 4], m![C # 32]>,
) {
    buf.write(stream)
}

/// Broadcasting read: replicate elements absent from `Buf`.
/// Time = m![T, A] broadcasts T temporally (same data repeated T times).
/// Packet = m![P] broadcasts P spatially (same element fills packet).
fn broadcasting_read<'l>(
    buf: &'l MemTensor<i8, m![A]>,
) -> StreamTensor<'l, i8, m![T, A], m![P]> {
    buf.read()
}

/// Broadcasting write: write broadcast stream back to buffer.
/// This is rejected as each `Buf` slot must have exactly one source position in `(Time, Packet)`
/// This code will panic when run
fn broadcasting_write(
    buf: &mut MemTensor<i8, m![A]>,
    stream: StreamTensor<i8, m![T, A], m![P]>,
) {
    buf.write(stream)
}
# 
# let buf_read = MemTensor::<bf16, m![A, B]>::from_vec(vec![bf16::from_f32(1f32); 8 * 512]);
# let mut buf_write = MemTensor::<bf16, m![A, B]>::from_vec(vec![bf16::from_f32(1f32); 8 * 512]);
# 
# let stream = strided_read(&buf_read);
# strided_write(&mut buf_write, stream);
# 
# // -----------------------------------------------------------------------------------
# 
# let buf_read = MemTensor::<bf16, m![N, C, H, W]>::from_vec(vec![bf16::from_f32(1f32); 4 * 3 * 8 * 8]);
# let mut buf_write = MemTensor::<bf16, m![N, C, H, W]>::from_vec(vec![bf16::from_f32(0f32); 4 * 3 * 8 * 8]);
# 
# let stream = axis_reordering_read(&buf_read);
# axis_reordering_write(&mut buf_write, stream);
# 
# // -----------------------------------------------------------------------------------
# 
# let buf_read = MemTensor::<i8, m![A, B, C # 8]>::from_vec(vec![1i8; 8 * 512 * 8]);
# let mut buf_write = MemTensor::<i8, m![A, B, C # 8]>::from_vec(vec![0i8; 8 * 512 * 8]);
# 
# let stream = tiling_read(&buf_read);
# tiling_write(&mut buf_write, stream);
# 
# // -----------------------------------------------------------------------------------
# 
# let buf_read = MemTensor::<i8, m![A]>::from_vec(vec![1i8; 8 ]);
# let mut buf_write = MemTensor::<i8, m![A]>::from_vec(vec![0i8; 8 ]);
#
# let stream = broadcasting_read(&buf_read);
# let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
#     broadcasting_write(&mut buf_write, stream);
# }));
# assert!(result.is_err()); 
# 
```

<a id="architecture"></a>
## 구조

각 sequencer 호출은 입력·출력 텐서 매핑으로부터 sequencer 하드웨어가 실행할 중첩 루프 설정으로 컴파일된다.
각 설정은 `[size_0 : stride_0, size_1 : stride_1, ...] : packet_size` 형태이며, 아래첨자 0 이 가장 바깥 루프다:

```rust
struct Config {
    /// Each entry defines a nested loop level.
    entries: Vec<Entry>,
    /// Number of elements per packet.
    packet_size: usize,
}

struct Entry {
    /// Number of iterations for this loop level.
    size: usize,
    /// Memory address distance (in elements) to skip after each iteration.
    stride: isize,
}
```

각 엔트리는 텐서 순회의 한 차원을 인코딩한다.
`size` 필드는 이 루프가 몇 번 반복하는지를 정하고, `stride` 필드는 연속한 반복 사이의 메모리 오프셋을 정한다.

<a id="access-size"></a>
### 접근 크기

`max_access_size = gcd(Packet::SIZE, contiguous_run)` 는 하드웨어 접근 한 번당 원소 개수다.

여기서 `contiguous_run` 은 `Config` 엔트리 중 가장 안쪽에 있는, 물리적으로 연속인 구간의 원소 개수다.
`max_access_size` 가 클수록 패킷당 접근 횟수가 줄어든다.
다음은 `Config` 로부터 `max_access_size` 를 계산하는 방법을 보여 준다:

```rust
# struct Config { entries: Vec<Entry>, packet_size: usize }
# struct Entry { size: usize, stride: isize }
# fn gcd(mut a: usize, mut b: usize) -> usize { while b != 0 { (a, b) = (b, a % b); } a }
impl Config {
    fn contiguous_run(&self) -> usize {
        // Walk pairs from innermost outward; stop at the first non-contiguous pair.
        // Two adjacent entries (n_outer : s_outer) and (n_inner : s_inner)
        // are contiguous when s_outer == n_inner * s_inner.
        let mut contiguous_run = self.entries.last().map_or(1, |e| e.size);
        for w in self.entries.windows(2).rev() {
            if w[0].stride == w[1].size as isize * w[1].stride {
                contiguous_run *= w[0].size;
            } else {
                break;
            }
        }
        contiguous_run
    }

    fn max_access_size(&self) -> usize {
        gcd(self.packet_size, self.contiguous_run())
    }
}
# 
# let config = Config {
#     entries: vec![
#         Entry { size: 4, stride: 192 },
#         Entry { size: 3, stride: 64 },
#         Entry { size: 8, stride: 8 },
#         Entry { size: 8, stride: 1 },
#     ],
#     packet_size: 8,
# };
# 
# assert_eq!(config.contiguous_run(), 768);
# assert_eq!(config.max_access_size(), 8);
```

대부분의 경우 패킷 레이아웃은 DM 에서 완전히 연속이며 `max_access_size == Packet::SIZE` 다.
`max_access_size < Packet::SIZE` 인 경우는 [비연속 패킷](#non-contiguous-packets)을 보라.

### 동작 방식

`m![N, C, H, W]` → `m![W, H, C, N]` 에 대한 `Config` 는 스트림의 축마다 엔트리를 하나씩 가지며, 각 스트라이드는 원본 버퍼에서 그 축이 차지하는 폭과 같다.
`Packet = m![1]` 이므로 `Packet::SIZE = max_access_size = 1` 이고 sequencer 는 루프 반복마다 DM 접근을 한 번씩 낸다.

```rust,ignore
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{MemTensor, StreamTensor};
# struct Config {
#     entries: Vec<Entry>,
#     packet_size: usize,
# }
# struct Entry {
#     size: usize,
#     stride: isize,
# }
axes![N = 4, C = 3, H = 8, W = 8];

fn read_nchw_whcn(buf: &MemTensor<bf16, m![N, C, H, W]>) ->
                     StreamTensor<bf16, m![W, H, C, N], m![1]> {
    // Compiler-generated configuration: [8 : 1, 8 : 8, 3 : 64, 4 : 192] : 1
    let config = Config {
        entries: vec![
            Entry { size: 8, stride: 1 },   // W
            Entry { size: 8, stride: 8 },   // H
            Entry { size: 3, stride: 64 },  // C
            Entry { size: 4, stride: 192 }, // N
        ],
        packet_size: 1,
    };

    // The hardware executes the configuration as nested loops:
    for w in 0..8 {
        for h in 0..8 {
            for c in 0..3 {
                for n in 0..4 {
                    // Read each address
                    let addr = 1 * w + 8 * h + 64 * c + 192 * n;
                    // yield buf[addr];
                }
            }
        }
    }

    buf.read()
}

fn write_whcn_nchw(buf: &mut MemTensor<bf16, m![N, C, H, W]>,
                  stream: StreamTensor<bf16, m![W, H, C, N], m![1]>) {
    // The compiler generates an identical config for writing
    // The hardware executes the configuration as nested loops:
    for w in 0..8 {
        for h in 0..8 {
            for c in 0..3 {
                for n in 0..4 {
                    // Write to each address
                    let addr = 1 * w + 8 * h + 64 * c + 192 * n;
                    // buf[addr] = stream.next();
                }
            }
        }
    }
}
```




## 설정

다음 패턴들이 커널 작성자가 마주칠 만한 설정의 대부분을 다룬다.

### 축 전치

스트림이 버퍼와 다른 순서로 축을 방문하도록 축을 전치할 수 있으며, 컴파일러는 그 순서로 메모리를 순회하는 데 필요한 스트라이드를 계산한다.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{MemTensor, StreamTensor};
axes![A = 8, B = 8, C = 8];

fn read_rearranging<'l>(
    buf: &'l MemTensor<i8, m![A, B, C # 32]>,  // Buf
) -> StreamTensor<'l, i8, m![B, A], m![C # 16]> {  // Time, Packet
    buf.read()
}
#
# let buf_read = MemTensor::<i8, m![A, B, C # 32]>::from_vec(vec![1i8; 8 * 8 * 32]);
# let _stream = read_rearranging(&buf_read);
```

컴파일러는 결합된 매핑 `m![B, A, C # 16]` 을 항 단위로 처리하면서 그 과정에서 `Buf` 를 변형해 설정 엔트리를 생성한다.
각 항에 대해 엔트리 크기는 항 크기와 같고, 스트라이드는 그 항이 현재 `Buf` 안에서 차지하는 부피와 같다.
항을 처리한 뒤에는 그 축이 소비되었음을 반영하도록 `Buf` 가 갱신된다:

| 항 | 엔트리 | 스트라이드 출처 | 처리 후 `Buf` |
|------|-------|---------------|---------------|
| `B` | `8 : 32` | `m![C # 32]::SIZE` | `m![A, 1 # 8, C # 32]` |
| `A` | `8 : 256` | `m![1 # 8, C # 32]::SIZE` | `m![1 # 64, C # 32]` |
| `C # 16` | `16 : 1` | 연속 (`Packet` 차원) | `1 # 2048` |

`Packet::SIZE = max_access_size = 16` 이다.
가장 안쪽 엔트리 `16 : 1` 은 연속이므로 하드웨어는 패킷 전체를 한 번의 접근으로 전송한다.


### 축 분할

타일링은 캐시 효율을 위해서나 tensor unit 버퍼 크기에 맞추기 위해 논리 축을 하위 블록으로 쪼개며, 컴파일러는 그 축을 여러 엔트리로 분할해 이를 구현한다.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{MemTensor, StreamTensor};
axes![A = 8, B = 8, C = 4];

fn read_splitting<'l>(
    buf: &'l MemTensor<i8, m![A, B, C # 8]>,  // Buf
) -> StreamTensor<'l, i8, m![A % 2, B % 4, A / 2, B / 4], m![C # 32]> {  // Time, Packet
    buf.read()
}
#
# let buf_read = MemTensor::<i8, m![A, B, C # 8]>::from_vec(vec![1i8; 8 * 8 * 8]);
# let _stream = read_splitting(&buf_read);
```

`A % 2` 와 `A / 2` 같은 표현식은 축 `A` 를 별개의 엔트리로 분할한다.
컴파일러는 `m![A % 2, B % 4, A / 2, B / 4, C # 32]` 를 항 단위로 처리한다:

| 항 | 엔트리 | 스트라이드 출처 | 처리 후 `Buf` |
|------|-------|---------------|---------------|
| `A % 2` | `2 : 64` | `m![B, C # 8]::SIZE` | `m![A / 2, 1 # 2, B, C # 8]` |
| `B % 4` | `4 : 8` | `m![C # 8]::SIZE` | `m![A / 2, 1 # 2, B / 4, 1 # 4, C # 8]` |
| `A / 2` | `4 : 128` | `m![1 # 2, B / 4, 1 # 4, C # 8]::SIZE` | `m![1 # 8, B / 4, 1 # 4, C # 8]` |
| `B / 4` | `2 : 32` | `m![1 # 4, C # 8]::SIZE` | `m![1 # 64, C # 8]` |
| `C # 32` | `32 : 1` | 연속 (`Packet` 차원) | `1 # 512` |

`Packet::SIZE = max_access_size = 32` 이다.


### 축 슬라이싱

슬라이싱은 메모리 레이아웃에서 인덱스의 일부 범위만 읽으며, 이는 인덱싱된 뷰가 원본 텐서의 부분집합을 선택할 때 생기는 상황이다.

```rust,ignore
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{MemTensor, StreamTensor};
axes![A = 16, B = 8, C = 8];

fn read_slicing<'l>(
    buf: &'l MemTensor<i8, m![A, B, C]>,  // Buf
) -> StreamTensor<'l, i8, m![A / 4, A % 4 = 3, B / 4, B % 4 = 2], m![C]> {  // Time, Packet
    buf.read()
}
#
# let buf_read = MemTensor::<i8, m![A, B, C]>::from_vec(vec![1i8; 16 * 8 * 8]);
# let _stream = read_slicing(&buf_read);
```

`= 3` 표기는 `A % 4` 를 4 회가 아니라 3 회 반복으로 제한해, 하드웨어를 텐서의 부분 영역으로 한정한다.
컴파일러는 `m![A / 4, A % 4 = 3, B / 4, B % 4 = 2, C]` 를 항 단위로 처리한다:

| 항 | 엔트리 | 스트라이드 출처 | 처리 후 `Buf` |
|------|-------|---------------|---------------|
| `A / 4` | `4 : 256` | `m![A % 4, B, C]::SIZE` | `m![1 # 4, A % 4, B, C]` |
| `A % 4 = 3` | `3 : 64` | `m![B, C]::SIZE` (3 으로 슬라이싱) | `m![1 # 16, B, C]` |
| `B / 4` | `2 : 32` | `m![B % 4, C]::SIZE` | `m![1 # 32, B % 4, C]` |
| `B % 4 = 2` | `2 : 8` | `m![C]::SIZE` (2 로 슬라이싱) | `m![1 # 128, C]` |
| `C` | `8 : 1` | 연속 (`Packet` 차원) | `1 # 1024` |

`Packet::SIZE = max_access_size = 8` 이다.


### 축 브로드캐스트

브로드캐스트는 스트림이 `Buf` 에 없는 축을 방문할 때 원소를 여러 패킷이나 시간 단계에 걸쳐 복제한다.
`Time` 이나 `Packet` 에는 있지만 `Buf` 에는 없는 축(또는 `N / 512` 같은 부분 축 조각)은 모두 브로드캐스트 엔트리가 되며, 스트라이드 표에 `: 0` 으로 표시된다(하드웨어가 반복마다 같은 주소를 다시 방문한다).

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{MemTensor, StreamTensor};
axes![A = 16, T = 4, P = 4];

fn read_broadcasting<'l>(
    buf: &'l MemTensor<i8, m![A]>,  // Buf
) -> StreamTensor<'l, i8, m![T, A], m![P]> {  // Time, Packet
    buf.read()
}
#
# let buf_read = MemTensor::<i8, m![A]>::from_vec(vec![1i8; 16]);
# let _stream = read_broadcasting(&buf_read);
```

컴파일러는 `m![T, A, P]` 를 항 단위로 처리한다:

| 항 | 엔트리 | 스트라이드 출처 | 처리 후 `Buf` |
|------|-------|---------------|---------------|
| `T` | `4 : 0` | `Buf` 에 없음 (브로드캐스트) | `m![A]` |
| `A` | `16 : 1` | `A` 가 `m![A]` 안에 있음 | `1 # 16` |
| `P` | `4 : 0` | `Buf` 에 없음 (브로드캐스트) | `1 # 16` |

`Packet::SIZE = max_access_size = 4` 이다.
`P` 는 브로드캐스트되므로 같은 원소가 패킷 전체에 복제된다(공간 브로드캐스트).
`T` 는 브로드캐스트되므로 같은 데이터가 시간 단계마다 반복된다(시간 브로드캐스트).

같은 규칙은 `Time` 이나 `Packet` 이 `Buf` 에 없는 축의 조각을 참조할 때도 적용된다.
예를 들어 `m![N % 512]` 버퍼를 `StreamTensor<m![N / 512], m![N % 512]>` 로 읽으면 `N / 512` 시간 엔트리에서 브로드캐스트된다. 버퍼의 512 개 원소가 `N / 512` 번의 바깥 반복마다 재사용된다.


<a id="merging-entries"></a>
### 엔트리 병합

하드웨어는 설정당 최대 8 개 엔트리를 지원하므로, 변환이 그보다 많이 만들어 내면 컴파일러가 인접 엔트리를 병합해 그 한도를 맞춘다.
인접한 엔트리 `(n1 : s1)` 과 `(n2 : s2)` 는 `(n1 * n2 : s2)` 로 병합되며, 이는 물리적으로 연속일 때, 즉 `s1 == n2 * s2` 일 때다.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{MemTensor, StreamTensor};
axes![N = 8, C = 8, H = 8, W = 32];

fn read_merging<'l>(
    buf: &'l MemTensor<i8, m![N, C, H, W]>,  // Buf
) -> StreamTensor<'l, i8, m![W / 16, H % 2, H / 2, C / 2, C % 2, N / 2, N % 2, W / 8 % 2], m![W % 8]> {  // Time, Packet
    buf.read()
}
#
# let buf_read = MemTensor::<i8, m![N, C, H, W]>::from_vec(vec![1i8; 8 * 8 * 8 * 32]);
# let _stream = read_merging(&buf_read);
```

컴파일러는 `m![W / 16, H % 2, H / 2, C / 2, C % 2, N / 2, N % 2, W / 8 % 2, W % 8]` 을 항 단위로 처리해 초기 엔트리 9 개를 만든다:

| 항 | 엔트리 | 스트라이드 출처 |
|------|-------|---------------|
| `W / 16` | `2 : 16` | `m![W % 16]::SIZE` |
| `H % 2` | `2 : 32` | `m![W]::SIZE` |
| `H / 2` | `4 : 64` | `m![H % 2, W]::SIZE` |
| `C / 2` | `4 : 512` | `m![C % 2, H, W]::SIZE` |
| `C % 2` | `2 : 256` | `m![H, W]::SIZE` |
| `N / 2` | `4 : 4096` | `m![N % 2, C, H, W]::SIZE` |
| `N % 2` | `2 : 2048` | `m![C, H, W]::SIZE` |
| `W / 8 % 2` | `2 : 8` | `m![W % 8]::SIZE` |
| `W % 8` | `8 : 1` | 연속 (패킷 차원) |

엔트리 9 개는 하드웨어 한도인 8 개를 넘으므로, 컴파일러는 `s1 == n2 * s2` 인 연속 쌍을 병합한다.
`H % 2 -> (2 : 32)` 와 `H / 2 -> (4 : 64)` 의 엔트리는 물리적으로 연속이 아니어서 병합되지 않는다(\\(s_1 \neq n_2 \times s_2 \iff 32 \neq 4 \times 64\\)).
최종 설정은 엔트리 6 개다.
마지막 병합은 시간/패킷 경계를 가로지른다. `W/8%2 (2:8)` 와 `W%8 (8:1)` 이 `W%16 (16:1)` 로 병합된다.

| 항 | 엔트리 | 병합된 엔트리 |
|------|-------|----------------|
| `W / 16` | `2 : 16` |  |
| `H % 2` | `2 : 32` |  |
| `H / 2` | `4 : 64` |  |
| `C` | `8 : 256` | `C / 2 (4 : 512)`,<br>`C % 2 (2 : 256)` |
| `N` | `8 : 2048` | `N / 2 (4 : 4096)`,<br>`N % 2 (2 : 2048)` |
| `W % 16` | `16 : 1` | `W / 8 % 2 (2 : 8)`,<br>`W % 8 (8 : 1)` |

`Packet::SIZE = max_access_size = 8` 이다.


<a id="non-contiguous-packets"></a>
### 비연속 패킷

DM 레이아웃이 패킷 구간 안에서 스트라이드 불연속을 가지면 `max_access_size < Packet::SIZE` 이고, 하드웨어는 패킷마다 한 번이 아니라 연속 하위 블록마다 한 번씩 접근을 낸다.
아래 예제는 원소 32 개짜리 패킷(`m![A, B]`)을 각 B 행이 DM 에서 16 칸으로 패딩된 버퍼에 쓴다.
A 의 스트라이드가 8 이 아니라 16 이므로 패킷 구간이 연속이 아니고, 하드웨어는 1 번이 아니라 4 번의 접근을 낸다:

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{MemTensor, StreamTensor};
axes![A = 4, B = 8];

fn write_padded(
    buf: &mut MemTensor<i8, m![A, B # 16]>,
    stream: StreamTensor<i8, m![1], m![A, B]>,
) {
    // Compiler-generated configuration: [
    //   A -> 4 : 16,   (16 != 8 * 1, NOT contiguous — padding gap after each B row)
    //   B -> 8 : 1,    (packet sub-block, contiguous)
    // ] : 32
    buf.write(stream)
}
#
# let buf_read = MemTensor::<i8, m![A, B]>::from_vec(vec![1i8; 4 * 8]);
# let mut buf_write = MemTensor::<i8, m![A, B # 16]>::from_vec(vec![0i8; 4 * 16]);
# let stream = buf_read.read();
# write_padded(&mut buf_write, stream);
```

`Packet::SIZE = 32`, `contiguous_run = 8`, `max_access_size = 8` 이다.

비연속 스트라이드는 `Packet` 이 원본 레이아웃에서 서로 인접하지 않은 축을 담을 때도 생긴다.
아래 예제는 같은 `m![N, C, H, W]` 버퍼를 서로 다른 두 가지 `Packet` 선택으로 읽는다.
가장 안쪽 축 `W` 만 `Packet` 에 두면 `max_access_size = Packet::SIZE = 8` 이 되어 패킷당 접근이 한 번이다.
`m![N, H, W]` 를 `Packet` 에 두면 `C` 를 건너뛰므로 원본에서 N 의 스트라이드(96)가 H×W(32)와 같지 않다. 그래서 `contiguous_run = 32`, `max_access_size = 32` 이고, 하드웨어는 패킷당 1 번이 아니라 4 번의 접근을 낸다.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{MemTensor, StreamTensor};
axes![N = 4, C = 3, H = 4, W = 8];

// Compiler-generated configuration: [
//   N -> 4 : 96,   (96 == 3 × 32, contiguous)
//   C -> 3 : 32,   (32 == 4 × 8,  contiguous)
//   H -> 4 : 8,    (8  == 8 × 1,  contiguous)
//   W -> 8 : 1,    (packet dimension)
// ] : 8
// contiguous_run = 8 (W); ×4 (H): 8==8×1 ✓; ×3 (C): 32==4×8 ✓; ×4 (N): 96==3×32 ✓; all axes contiguous
// max_access_size = gcd(packet_size, contiguous_run) = packet_size = 8
fn read_contiguous<'l>(
    buf: &'l MemTensor<i8, m![N, C, H, W]>,
) -> StreamTensor<'l, i8, m![N, C, H], m![W]> {
    buf.read()
}
#
# let buf_read = MemTensor::<i8, m![N, C, H, W]>::from_vec(vec![1i8; 4 * 3 * 4 * 8]);
# let _stream = read_contiguous(&buf_read);

// Compiler-generated configuration: [
//   C -> 3 : 32,   (time dimension)
//   N -> 4 : 96,   (96 != 4 × 8 = 32, NOT contiguous — C axis interspersed)
//   H -> 4 : 8,    (8  == 8 × 1, contiguous)
//   W -> 8 : 1,    (packet dimension)
// ] : 128
// contiguous_run = 8 (W); ×4 (H): 8==8×1 ✓ → 32; ×4 (N): 96!=4×8 ✗ stop → 32
// max_access_size = gcd(128, 32) = 32; hardware issues 128/32 = 4 accesses per packet
fn read_non_contiguous<'l>(
    buf: &'l MemTensor<i8, m![N, C, H, W]>,
) -> StreamTensor<'l, i8, m![C], m![N, H, W]> {
    buf.read()
}
#
# let buf_read = MemTensor::<i8, m![N, C, H, W]>::from_vec(vec![1i8; 4 * 3 * 4 * 8]);
# let _stream = read_non_contiguous(&buf_read);
```

## 제약

RNGD 에서는 다음 하드웨어 한도 중 하나라도 넘으면 컴파일 오류가 난다:

- **엔트리 한도**: 최대 8 개 엔트리이므로, 컴파일러는 가능한 경우 인접 엔트리를 병합한다(설정 절의 [엔트리 병합](#merging-entries) 참고).
- **반복 한도**: 엔트리당 `size <= 65,536`.
- **패킷 크기**: 1, 2, 4, 8, 16, 32 바이트 중 하나여야 한다.
- **패킷 페치**: 가장 안쪽 엔트리 `n : s` 는 다음 중 하나를 만족해야 한다:
  - 연속 접근(인접 원소): `(s == 0 || s == 1) && n % packet_size == 0`
  - 이산 접근(단일 원소 패킷): `packet_size == 1`

병합이 실패하거나 한도를 넘으면 텐서 매핑을 다시 설계하거나 연산을 여러 sequencer 호출로 나눈다.


### 호환되는 축 분해

`Buf` 와 스트림 양쪽에 이름이 나오는 축은 같은 분해를 써야 한다.
컴파일러는 스트림을 항 단위로 훑으면서 `Buf` 에서 축을 소비한다([구조](#architecture) 참고).
`Buf` 가 축을 한 방식으로 쪼개고 스트림이 다른 방식으로 쪼개어 공통 세분이 없으면, 어떤 순회 순서도 통하지 않아 설정이 거부된다. 양쪽의 전체 원소 개수가 같더라도 그렇다.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{MemTensor, StreamTensor};
axes![A = 15];

fn read_incompatible<'l>(
    buf: &'l MemTensor<i8, m![A % 5, A / 5]>,  // Buf
) -> StreamTensor<'l, i8, m![1], m![A % 3, A / 3]> {  // Time, Packet
    buf.read() // Compilation error: incompatible decomposition
}
#
# let buf_read = MemTensor::<i8, m![A % 5, A / 5]>::from_vec(vec![1i8; 15]);
# let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| { read_incompatible(&buf_read) }));
# assert!(result.is_err());
```

`Buf` 는 `A` 를 `5 × 3` 으로 분해하지만 스트림은 `3 × 5` 로 분해한다.
`gcd(5, 3) = 1` 이므로 어느 분해도 다른 쪽을 세분하지 못한다. 컴파일러는 이미 5 블록 분할로 확정된 `Buf` 에서 `A % 3` 을 소비할 수 없다.


## 간접 접근


위의 모든 엔트리는 고정 스트라이드를 쓴다. 반복 사이의 메모리 오프셋이 일정하다.
`IndirectLoop` 은 반복마다 가변 오프셋을 허용해 이를 확장하며, 데이터 의존 접근 패턴을 가진 게더 연산을 가능하게 한다.

표준 패턴 `(limit, stride)` 는 `(limit, [offset0, offset1, ...])` 가 되며, 각 반복은 주어진 수열에서 서로 다른 오프셋을 사용한다.
이는 인덱스가 런타임에 정해지는 임베딩 조회 같은 연산을 지원한다.
