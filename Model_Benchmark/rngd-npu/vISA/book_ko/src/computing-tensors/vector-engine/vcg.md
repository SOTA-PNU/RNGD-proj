# Valid Count Generator

[Intra-Slice Reduce](./intra-slice-reduce.md) 단계는 `REDUCE_LABEL`(예: `R`)로 지정된 축을 그 `Time` 및 `Packet` 인수에 걸쳐 축약(리듀스)하고, `Slice` 인수는 출력에 남긴다.
그 축은 하드웨어 차원에 맞추기 위해 패딩이 필요한 경우가 많으며, 추가된 패딩 위치에는 임의의 데이터가 들어 있어 축약에서 제외해야 한다.

Valid Count Generator(VCG)가 이 문제를 해결한다.
사용자는 매핑에서 `R` 을 `Slice`, `Time`, `Packet` 에 걸친 하위 표현식으로 배치한다.
그러면 컴파일러가 VCG 를 구성해, 8원소 flit 마다 몇 개의 원소가 실제 데이터인지를 나타내는 `valid_size` 개수를 태그로 붙인다.
`Time` 또는 `Slice` 의 각 하위 표현식은 time filter 에 할당된 sequencer 카운터에 대응한다.
`Packet` 의 각 하위 표현식은 packet clipper 를 구동한다.

이 페이지 전반에서 대문자로 시작하는 `Slice`, `Time`, `Packet` 은 매핑 차원을 가리키고, 소문자 `slice` 와 `time step` 은 런타임 인스턴스를 가리킨다.

VCG 가 동작하려면 매핑이 `R` 을 특정한 형태로 표현해야 한다.
`R` 은 하드웨어 정렬 크기로 패딩되며, 일반적으로 논할 때는 `R # PADDED_SIZE` 로 쓴다.
구체적인 예제에서는 실제 패딩된 값(예: `R # 16`, `R # 48`)을 쓴다.
그러면 각 하위 표현식은 `R # PADDED_SIZE / n % m`(스트라이드 `n`, 모듈로 `m`) 형태인 `R # PADDED_SIZE` 의 인수가 된다.
`/ n` 과 `% m` 의 의미는 [스트라이드와 모듈로](../../mapping-tensors/mapping-expressions.md#stride-and-modulo)를 참고한다.
각 하위 표현식은 하나의 하드웨어 차원에 할당된다.
한 가지 가능한 분배는 `R = 43` 을 `R # 48` 로 패딩해 세 차원 전체에 나누는 것이다:

```text
Slice:  R # 48 / 8       (stride 8, 6 positions)
Time:   R # 48 / 2 % 4   (stride 2, 4 positions)
Packet: R # 48 % 2       (stride 1, 2 positions)
```

## 구조

VCG 는 각 flit 에 `valid_size(s, t) ∈ {0, 1, ..., 8}` 을 할당한다. 여기서 `s` 는 slice id(모든 `Slice` 하위 표현식에 걸친 flit 의 위치를 인코딩한 정수)이고 `t` 는 time step 이다.
flit 의 처음 `valid_size` 개 원소가 실제 데이터이고 나머지는 패딩이다.

```rust,ignore
struct VcgConfig {
    time_filters:   [TimeFilterConfig; 3], // for R's sub-expressions in Time and/or Slice
    packet_clipper: PacketClipperConfig,   // for R's sub-expressions in Packet
}

impl VcgConfig {
    fn valid_size(&self, s: u64, t: u64) -> u32 {
        if self.time_filters.iter().all(|tf| tf.valid(s, t)) {
            self.packet_clipper.valid_size(t)
        } else {
            0
        }
    }
}
```

flit `(s, t)` 에 대해 모든 타이머가 유효하다고 보고하면, packet clipper 가 그 flit 안의 몇 개 원소가 실제 데이터인지 결정한다.
어느 한 타이머라도 무효라고 보고하면, packet clipper 가 무엇이라 하든 그 flit 의 모든 원소가 제외된다.
두 구성 요소인 `TimeFilterConfig` 와 `PacketClipperConfig` 는 아래 절에서 설명한다.

### Time Filter

각 slice `s` 에 대해 time filter 는 각 time step `t` 가 유효한 `R` 데이터를 담는지 판정한다.
아래 소절들은 가장 단순한 경우에서 시작해 복잡도를 더해 가며 `fn valid()` 를 단계적으로 쌓아 올린다.
`R` 이 `Time` 이나 `Slice` 에 하위 표현식을 갖지 않으면, `slice_mask = 0` 과 `slice_thres = 1` 로 설정해 time filter 를 비활성화한다. 그러면 모든 `s` 에 대해 `s & 0 = 0 < 1` 이 되어 `Less` 갈래를 타므로 `fn valid()` 는 항상 `true` 를 반환한다. `slice_thres = 1` 이라는 선택은 관례이며, `0` 은 항상 그보다 작으므로 어떤 양수 값이든 동작한다.

```rust,ignore
struct TimeFilterConfig {
    // How R's index is reconstructed from t.
    sequencer: Sequencer,

    // Slice classification (see `R in Slice and Time`, `R in Time and Slice`).
    slice_mask:  u32,
    slice_thres: u32,
    time_thres:  u32,
    mode:        TimeFilterMode, // SliceMajor | TimeMajor
}

impl TimeFilterConfig {
    /// Returns true if flit (s, t) carries valid R data.
    fn valid(&self, s: u64, t: u64) -> bool {
        let idx = self.sequencer.index(t);
        match ((s & self.slice_mask).cmp(&self.slice_thres), self.mode) {
            (Less,    _)          => true,
            (Greater, SliceMajor) => false,
            _                     => idx < self.time_thres as u64,
        }
    }
}
```

#### `R` 을 `Time` 으로

가장 단순한 경우, `R` 은 다른 축 없이 `Time` 전체를 차지한다.
각 time step `t` 는 하나의 `R` 인덱스에 직접 대응하며, `t < R::SIZE` 일 때 유효하다.

```rust,ignore
// The compiler emits roughly:
TimeFilterConfig {
    sequencer:   [R # PADDED_SIZE -> size PADDED_SIZE : stride 1],  // idx = t
    slice_mask:  0,           // no slice partitioning
    slice_thres: 0,           // 0 cmp 0 = Equal -> falls to `idx < time_thres`
    time_thres:  R::SIZE,     // valid when idx < R::SIZE
    mode:        SliceMajor,  // arbitrary; only the Equal arm is hit
}
```

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, R = 12, X = 128];

fn reduce_time_only<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1 # 2], m![X, A / 4], m![R # 16], m![A % 4 # 8], i32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1 # 2], m![X, A / 4], m![1], m![A % 4], i32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_trim::<m![A % 4]>()
        //   Slice     = m![X, A / 4]
        //   Time      = m![R # 16]     (steps < R::SIZE valid)
        //   Packet    = m![A % 4]
        //   OutTime   = m![1]          (R eliminated from Time)
        //   OutPacket = m![A % 4]
        .vector_intra_slice_reduce::<R, m![1], m![A % 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
}
# 
# let mut ctx = Context::acquire();
# 
# let i: VectorBranchTensor<'_, _, i32, m![1], m![1 # 2], m![X, A / 4], m![R # 16], m![A % 4 # 8], i32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let _o = reduce_time_only(i);
```

위 예제에서 sequencer 는 `R # 16` 을 `size 16 : stride 1` 로 한 번 순회하므로 모든 time step 에 대해 `idx = t` 이다.
`time_thres = R::SIZE = 12` 이므로 처음 12 개 time step(`t = 0..11`)은 유효하고 나머지 4 개(`t = 12..15`)는 걸러진다.
그러면 intra-slice reduce 는 slice 마다 실제 `R` 원소 12 개만 정확히 접는다.

<a id="r-in-time"></a>
#### `Time` 안의 `R`

VCG 는 `R` 이 다른 축과 공간을 공유하며 여러 하위 표현식으로 임의의 순서로 나타나는 `Time` 매핑을 지원한다.
time filter 는 [Sequencer](../../moving-tensors/sequencer.md)를 사용해 `t` 를 하위 표현식별 카운터로 분해한다. `R` 에 할당된 카운터에 대해 `value × stride` 를 합하면 `idx` 가 되며, 이는 그 time step 의 `R` 인덱스를 인코딩한다.

다음 예제는 `R = 10` 을 `R # 12` 로 패딩해 사용하며, `Time` 안에서 `R` 의 두 하위 표현식 사이에 `A` 가 놓인다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 3, B = 4, R = 10, X = 64];

// R = 10, padded to R # 12, split as (size 3, stride 4) × (size 4, stride 1).
// time filter sums (R # 12 / 4 value) * 4 + (R # 12 % 4 value) * 1 to recover R index regardless of A.
fn reduce_time_reordered<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1 # 2], m![X # 256], m![R # 12 / 4, A, R # 12 % 4], m![B # 8], i32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1 # 2], m![X # 256], m![A], m![B], i32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_trim::<m![B]>()
        //   Slice     = m![X # 256]
        //   Time      = m![R # 12 / 4, A, R # 12 % 4]
        //   Packet    = m![B]
        //   OutTime   = m![A]    (R eliminated; A survives)
        //   OutPacket = m![B]
        .vector_intra_slice_reduce::<R, m![A], m![B]>(
            IntraSliceReduceOpI32::AddSat,
        )
}
# 
# let mut ctx = Context::acquire();
# 
# let i: VectorBranchTensor<'_, _, i32, m![1], m![1 # 2], m![X # 256], m![R # 12 / 4, A, R # 12 % 4], m![B # 8], i32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let _o = reduce_time_reordered(i);
```

컴파일러는 이 배치에 대해 time filter 를 다음과 같이 구성한다:

```rust,ignore
TimeFilterConfig {
    sequencer:   [R # 12 / 4 -> size 3 : stride 4,   // assigned to time filter
                  A          -> size 3 : stride 0,   // not assigned (A's OutTime)
                  R # 12 % 4 -> size 4 : stride 1],  // assigned to time filter
    slice_mask:  0,           // no slice partitioning (R is only in Time)
    slice_thres: 0,           // every slice falls to the `idx < time_thres` arm
    time_thres:  R::SIZE,     // 10
    mode:        SliceMajor,  // arbitrary; only the Equal arm is hit
}
```

`A` 항목은 stride 가 0 이므로 `idx` 에 전혀 기여하지 않으며, 이는 `Time` 안에서 `A` 의 위치가 유효성에 영향을 주지 않는다는 뜻이다.
나머지 두 항목이 `R` 을 `(R # 12 / 4 value) × 4 + (R # 12 % 4 value)` 로 복원한다.

예를 들면:
- `t = 13` 에서 sequencer 상태는 `(R # 12 / 4, A, R # 12 % 4) = (1, 0, 1)` 이고, `idx = 1 × 4 + 0 + 1 × 1 = 5` 가 된다. `idx < 10` 이므로 유효하다.
- `t = 27` 에서 sequencer 상태는 `(2, 0, 3)` 이고, `idx = 2 × 4 + 0 + 3 × 1 = 11` 이 된다. `idx ≥ 10` 이므로 무효다(패딩된 두 `R` 위치 중 하나다).

<a id="r-in-slice-and-time"></a>
#### `Slice` 와 `Time` 안의 `R`

`SliceMajor` 모드는 `Slice` 와 `Time` 양쪽에 여러 `R` 하위 표현식을 허용하되, `Slice` 하위 표현식이 `Time` 하위 표현식보다 더 상위(더 큰 스트라이드)여야 한다.
`Slice` 안에서 하위 표현식은 스트라이드 내림차순(상위가 하위보다 먼저)으로 나타나야 하며, 각각은 2 의 거듭제곱 크기와 2 의 거듭제곱 스트라이드를 가져야 그 비트들이 `slice_mask` 의 연속 구간을 차지한다.
`Time` 안에서는 하위 표현식이 임의의 순서로 나타나도 된다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![R = 11, X = 64];

fn reduce_slice_time_slicemajor<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1 # 2], m![R # 16 / 8, X, R # 16 / 4 % 2], m![R # 16 % 2, R # 16 / 2 % 2], m![1 # 8], i32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1 # 2], m![R # 16 / 8, X, R # 16 / 4 % 2], m![1], m![1 # 4], i32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_trim::<m![1 # 4]>()
        //   Slice     = m![R # 16 / 8, X, R # 16 / 4 % 2]   (major R sub-exprs, descending R-stride order required)
        //   Time      = m![R # 16 % 2, R # 16 / 2 % 2]      (minor R sub-exprs, any order OK)
        //   Packet    = m![1 # 4]
        //   OutTime   = m![1]          (R eliminated)
        //   OutPacket = m![1 # 4]
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::Min,
        )
}
# 
# let mut ctx = Context::acquire();
# 
# let i: VectorBranchTensor<'_, _, i32, m![1], m![1 # 2], m![R # 16 / 8, X, R # 16 / 4 % 2], m![R # 16 % 2, R # 16 / 2 % 2], m![1 # 8], i32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let _o = reduce_slice_time_slicemajor(i);
```

이 예제는 `R = 11`(`R # 16` 으로 패딩)을 다음 하위 표현식으로 `Slice` 와 `Time` 에 배치한다.

| 차원 | 하위 표현식 | 스트라이드 |
|-----------|----------------|--------|
| `Slice` | `R # 16 / 8` | 8 |
| `Slice` | `R # 16 / 4 % 2` | 4 |
| `Time` | `R # 16 % 2` | 1 |
| `Time` | `R # 16 / 2 % 2` | 2 |

이 예제는 4 개의 slice 그룹(`Slice` 로부터의 `R` 기여값 `0`, `4`, `8`, `12` 마다 하나)과 slice 당 4 번의 time 반복으로 나뉜다.
이 배치에는 세 가지 영역이 있다. 2 개 slice 는 완전히 유효하고, 1 개 slice 는 부분적으로 유효하며, 1 개 slice 는 완전히 무효다.

| `Slice` 로부터의 `R` 기여값 | 반복에 걸친 `R` 값 | 유효 time step |
|-------------------------------|------------------------------|------------------|
| `0` | `0, 2, 1, 3` | 4 (모두 `< R::SIZE`) |
| `4` | `4, 6, 5, 7` | 4 (모두 `< R::SIZE`) |
| `8` | `8, 10, 9, 11` | 3 (`R = 11` 무효) |
| `12` | `12, 14, 13, 15` | 0 (모두 `≥ R::SIZE`) |

컴파일러는 아래의 time filter 설정을 내보낸다.

```rust,ignore
TimeFilterConfig {
    sequencer:   [R # 16 % 2     -> size 2 : stride 1,
                  R # 16 / 2 % 2 -> size 2 : stride 2],
    slice_mask:  0b1000001,  // bits 0 (R # 16 / 4 % 2) and 6 (R # 16 / 8) carry the R contribution
    slice_thres: 64,         // masked id encoding the boundary R contribution (= 8)
    time_thres:  3,          // = R::SIZE - boundary = 11 - 8
    mode:        SliceMajor,
}
```

각 필드는 유효성 판정의 한 부분을 인코딩한다:

- `sequencer` 는 각 `Time` 하위 표현식을 그 `R` 스트라이드와 함께 기록하므로, 복원된 `idx` 는 각 time step `t` 에서 `Time` 으로부터의 `R` 기여값과 같아진다. 이 예제에서 `t` 가 `[0, 4)` 를 훑는 동안 `idx` 는 `{0, 1, 2, 3}` 의 값을 갖는다.
- `slice_mask` 는 slice id 에서 `Slice` 로부터의 `R` 기여값을 담는 비트를 추출한다. 이 예제에서 비트 `0` 은 `R # 16 / 4 % 2` 를 담고 비트 `6` 은 `R # 16 / 8` 을 담으므로 `slice_mask = 0b1000001` 이다.
- `slice_thres` 는 `slice_mask` 안에서 부분 유효 slice 의 `R` 기여값을 인코딩하는 비트 패턴이다. 위 표에서 부분 유효 slice 의 `R` 기여값은 `8` 이며, 비트 `6`(`R # 16 / 8 = 1`)을 세우고 비트 `0`(`R # 16 / 4 % 2 = 0`)을 지워 인코딩한다. 따라서 `slice_thres = 64` 다.
- `time_thres` 는 `R::SIZE` 에서 부분 유효 slice 의 `R` 기여값을 뺀 값이다. 이 예제에서는 `time_thres = 11 - 8 = 3` 이다.

`fn valid` 는 모든 flit 에 대해 올바른 유효성을 반환한다. 이를 확인하려면 `R` 인덱스를 `r = r_slice + idx` 로 분해하고 — 여기서 `r_slice` 는 `Slice` 로부터의 `R` 기여값(`s & slice_mask` 로 인코딩됨)이고 `idx` 는 `Time` 으로부터의 기여값이다 — slice 비교의 세 가지 경우를 살펴본다.

- `(s & slice_mask) < slice_thres` 일 때는 모든 time step 이 유효하다. `r_slice` 가 부분 유효 slice 의 기여값보다 최소한 slice 간격 하나만큼 작으므로, 모든 `idx` 에 대해 `r = r_slice + idx < R::SIZE` 다.
- `(s & slice_mask) = slice_thres` 일 때는 `idx < time_thres` 인 경우에만 time step 이 유효하다. 이것이 부분 유효 slice 이며, `time_thres` 의 정의에 따라 그 `r_slice = R::SIZE - time_thres` 다. 따라서 `r = r_slice + idx < R::SIZE` 는 정확히 `idx < time_thres` 일 때 성립한다.
- `(s & slice_mask) > slice_thres` 일 때는 어떤 time step 도 유효하지 않다. `r_slice` 가 부분 유효 slice 의 기여값보다 최소한 slice 간격 하나만큼 크므로 `r_slice ≥ R::SIZE` 다.

#### `Time` 과 `Slice` 안의 `R`

`TimeMajor` 모드는 [`SliceMajor`](#r-in-slice-and-time)의 쌍대다. 상위/하위 역할이 뒤집혀 `Time` 하위 표현식이 더 큰 스트라이드를, `Slice` 하위 표현식이 더 작은 스트라이드를 갖는다.
`Slice` 내부와 `Time` 내부의 순서 규칙, 그리고 `Slice` 하위 표현식에 대한 2 의 거듭제곱 크기/스트라이드 요구 사항은 `SliceMajor` 에서 그대로 이어진다.

`TimeMajor` 는 이렇게 물려받은 규칙 위에 제약을 하나 더한다.
`PADDED_SIZE` 는 `slice_span × time_span` 으로 분해되며, `slice_span` 과 `time_span` 은 각각 `Slice` 와 `Time` 안에 있는 `R` 하위 표현식들의 크기 곱이라는 점을 상기한다.
`TimeMajor` 는 `PADDED_SIZE - R::SIZE ≤ slice_span` 을 요구하며, 이는 최대 `slice_span` 개의 `R` 위치만 과도 패딩될 수 있다는 뜻이다.
이 제약은 필수적이며, 이를 위반하는 배치는 VCG 가 지원하지 않는다([`Time` 과 `Slice` 안의 `R`, 과도 패딩](#r-in-time-and-slice-over-padded) 참고).

```rust,should_panic
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![R = 13, X = 64];

fn reduce_time_slice_timemajor<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1 # 2], m![R # 16 / 2 % 2, X, R # 16 % 2], m![R # 16 / 4 % 2, R # 16 / 8], m![1 # 8], i32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1 # 2], m![R # 16 / 2 % 2, X, R # 16 % 2], m![1], m![1 # 4], i32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_trim::<m![1 # 4]>()
        //   Slice     = m![R # 16 / 2 % 2, X, R # 16 % 2]
        //   Time      = m![R # 16 / 4 % 2, R # 16 / 8]
        //   Packet    = m![1 # 4]
        //   OutTime   = m![1]          (R eliminated)
        //   OutPacket = m![1 # 4]
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
}
# 
# let mut ctx = Context::acquire();
# 
# let i: VectorBranchTensor<'_, _, i32, m![1], m![1 # 2], m![R # 16 / 2 % 2, X, R # 16 % 2], m![R # 16 / 4 % 2, R # 16 / 8], m![1 # 8], i32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let _o = reduce_time_slice_timemajor(i);
```

이 예제는 `R = 13`(`R # 16` 으로 패딩)을 다음 하위 표현식으로 `Slice` 와 `Time` 에 배치한다.

| 차원 | 하위 표현식 | 스트라이드 |
|-----------|----------------|--------|
| `Time` | `R # 16 / 4 % 2` | 4 |
| `Time` | `R # 16 / 8` | 8 |
| `Slice` | `R # 16 / 2 % 2` | 2 |
| `Slice` | `R # 16 % 2` | 1 |

이 하위 표현식들은 `slice_span = 4`, `time_span = 4`, 그리고 과도 패딩 `PADDED_SIZE - R::SIZE = 3` 을 주며, 이는 제약 `3 ≤ 4` 를 만족한다.
`Slice` 로부터의 `R` 기여값(`0`, `1`, `2`, `3`)으로 구분되는 4 개 slice 각각은 4 번의 time 반복에 걸쳐 4 개의 `R` 값을 훑는다.
이 배치에는 두 가지 영역이 있다. 1 개 slice 는 완전히 유효하고, 3 개는 부분적으로 유효하다(각각 반복 하나씩을 잃는다).

| `Slice` 로부터의 `R` 기여값 | 반복에 걸친 `R` 값 | 유효 time step |
|-------------------------------|------------------------------|------------------|
| `0` | `0, 8, 4, 12` | 4 (모두 `< R::SIZE`) |
| `1` | `1, 9, 5, 13` | 3 (`R = 13` 무효) |
| `2` | `2, 10, 6, 14` | 3 (`R = 14` 무효) |
| `3` | `3, 11, 7, 15` | 3 (`R = 15` 무효) |

컴파일러는 아래의 time filter 설정을 내보낸다.

```rust,ignore
TimeFilterConfig {
    sequencer:   [R # 16 / 4 % 2 -> size 2 : stride 1,    // 1 = 4 / slice_span
                  R # 16 / 8     -> size 2 : stride 2],   // 2 = 8 / slice_span
    slice_mask:  0b1000001,  // bits 0 (R # 16 % 2) and 6 (R # 16 / 2 % 2) carry the R contribution
    slice_thres: 1,          // masked id encoding r_slice = 1
    time_thres:  3,          // = time_span - 1 (partial slices drop the over-padded last iteration)
    mode:        TimeMajor,
}
```

이 설정은 `SliceMajor` 와 세 필드에서 다르다(`slice_mask` 는 같은 패턴을 따른다):

- `sequencer`: 모든 `Time` 하위 표현식의 스트라이드를 먼저 `slice_span` 으로 나눈다. 이 예제에서는 스트라이드 `4` 와 `8` 이 `1` 과 `2` 가 된다.
- `slice_thres`: 부분 유효 slice 의 `r_slice = slice_span - (PADDED_SIZE - R::SIZE)` 를 인코딩한다. 이 예제에서 목표 값은 `4 - 3 = 1` 이고, 비트 `0` 을 세우고(`R # 16 % 2 = 1`) 비트 `6` 을 지워(`R # 16 / 2 % 2 = 0`) 인코딩하면 `slice_thres = 1` 이 된다.
- `time_thres`: 항상 `time_span - 1` 이다. 이 예제에서는 `time_thres = 3` 이다.

`fn valid` 는 모든 flit 에 대해 올바른 유효성을 반환한다. 이를 확인하려면 `R` 인덱스를 `r = idx * slice_span + r_slice` 로 분해하고 — 여기서 `r_slice` 는 `s & slice_mask` 에서 디코딩한 그 slice 의 `R` 기여값이다 — slice 비교의 두 가지 경우를 살펴본다.

- `(s & slice_mask) < slice_thres` 일 때는 모든 time step 이 유효하다. `r_slice < slice_span - (PADDED_SIZE - R::SIZE)` 이므로 모든 `idx` 에 대해 `r ≤ (time_span - 1) * slice_span + r_slice < R::SIZE` 다.
- `(s & slice_mask) ≥ slice_thres` 일 때는 `idx < time_thres = time_span - 1` 인 경우에만 time step 이 유효하다. 그 slice 의 `r_slice ≥ slice_span - (PADDED_SIZE - R::SIZE)` 다. `idx ≤ time_span - 2` 에 대해서는 `r ≤ (time_span - 2) * slice_span + (slice_span - 1) < (time_span - 1) * slice_span ≤ R::SIZE` 다. `idx = time_span - 1` 에 대해서는 `r ≥ (time_span - 1) * slice_span + slice_span - (PADDED_SIZE - R::SIZE) = R::SIZE` 다.

`SliceMajor` 와 달리 `TimeMajor` 에는 "전부 무효" 영역이 없다. 패딩 제약 `PADDED_SIZE - R::SIZE ≤ slice_span` 이 과도 패딩을 충분히 빡빡하게 제한해 어떤 slice 도 완전히 무효가 되지 않는다.

### Packet Clipper

각 flit 에 대해 packet clipper 는 `valid_size(t)` 를 계산하는데, 이는 `t` 에만 의존하고(slice `s` 에는 의존하지 않고) 따라서 같은 time step 에서는 모든 slice 가 같은 개수를 받는다.
이 slice 독립성이 VCG 가 표현할 수 있는 배치를 제한한다.
아래 소절들은 `fn valid_size()` 를 단계적으로 쌓아 올린다.
`R` 이 `Packet` 에 하위 표현식을 갖지 않으면, 빈 sequencer 와 함께 `axis_size = packet_span = 8` 로 설정해 packet clipper 를 비활성화하며, 그러면 `fn valid_size()` 는 항상 `8`(flit 전체)을 반환한다.

```rust,ignore
struct PacketClipperConfig {
    sequencer:   Sequencer,
    axis_size:   u32, // R::SIZE
    packet_span: u32, // R positions per flit
}

impl PacketClipperConfig {
    /// Returns the valid element count for the flit at time step t.
    fn valid_size(&self, t: u64) -> u32 {
        let idx = self.sequencer.index(t);
        (self.axis_size - idx).clamp(0, self.packet_span)
    }
}
```

packet clipper 는 `Packet = m![R # PADDED_SIZE % packet_span # 8]` 을 요구한다.
다른 축이 `R` 과 `Packet` 을 공유하거나 `R` 이 `Packet` 안에서 여러 하위 표현식으로 쪼개지면, `fn valid_size()` 가 의존하는 연속 접두 성질이 깨진다([표현할 수 없는 패턴](#inexpressible-patterns) 참고).


#### `R` 을 `Packet` 으로

가장 단순한 경우, `R` 이 flit 하나에 들어가므로(`R::SIZE ≤ 8`) time step 이나 slice 와 무관하게 모든 flit 이 동일한 `valid_size = R::SIZE` 를 갖는다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, R = 3, X = 64];

fn reduce_packet_only<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1 # 2], m![X, A / 2], m![1], m![R # 8], f32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1 # 2], m![X, A / 2], m![1], m![1 # 4], f32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_trim::<m![R # 4]>()
        //   Slice     = m![X, A / 2]
        //   Time      = m![1]
        //   Packet    = m![R # 4]
        //   OutTime   = m![1]
        //   OutPacket = m![1 # 4]  (R eliminated from Packet)
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
}
# 
# let mut ctx = Context::acquire();
# 
# let i: VectorBranchTensor<'_, _, f32, m![1], m![1 # 2], m![X, A / 2], m![1], m![R # 8], f32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let _o = reduce_packet_only(i);
```

위 예제는 `R = 3`(`R # 8` 로 패딩)을 단일 하위 표현식으로 `Packet` 에만 배치한다.
컴파일러는 packet clipper 를 다음과 같이 구성한다:

```rust,ignore
PacketClipperConfig {
    sequencer:   [],   // empty: a single flit holds all of R
    axis_size:   3,    // R::SIZE
    packet_span: 8,    // R fits in 8 flit positions
}
```

모든 flit 이 `valid_size = clamp(3 - 0, 0, 8) = 3` 을 갖는다(time step 과 slice 에 걸쳐 일정하다).

#### `Time` 과 `Packet` 안의 `R`

VCG 는 `Time` 과 `Packet` 양쪽에 하위 표현식을 갖는 `R` 을 지원한다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, R = 10, X = 64];

fn reduce_time_packet<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1 # 2], m![X, A / 2], m![R # 16 / 4], m![R # 16 % 4 # 8], f32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1 # 2], m![X, A / 2], m![1], m![1 # 4], f32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_trim::<m![R # 16 % 4]>()
        //   Slice     = m![X, A / 2]
        //   Time      = m![R # 16 / 4]
        //   Packet    = m![R # 16 % 4 # 8]
        //   OutTime   = m![1]          (R eliminated)
        //   OutPacket = m![1 # 4]
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
}
# 
# let mut ctx = Context::acquire();
# 
# let i: VectorBranchTensor<'_, _, f32, m![1], m![1 # 2], m![X, A / 2], m![R # 16 / 4], m![R # 16 % 4 # 8], f32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let _o = reduce_time_packet(i);
```

위 예제는 `R = 10`(`R # 16` 으로 패딩)을 `Time` 과 `Packet` 에 걸쳐 배치한다:

| 차원 | 하위 표현식 | 스트라이드 |
|-----------|----------------|--------|
| `Time` | `R # 16 / 4` | 4 |
| `Packet` | `R # 16 % 4 # 8` | 1 |

컴파일러는 packet clipper 를 다음과 같이 구성한다:

```rust,ignore
PacketClipperConfig {
    sequencer:   [R # 16 / 4 -> size 4 : stride 4],
    axis_size:   10,   // R::SIZE
    packet_span: 4,    // 4 R positions per flit (the trailing 4 flit positions are always padding)
}
```

이 설정으로 sequencer 는 `R` 인덱스에 대한 `Time` 기여값인 `idx(t)` 를 복원한다([`Time` 안의 `R`](#r-in-time) 참고).
`Time` 이 `axis_size = 10` 개 원소 중 `idx(t)` 개를 덮고 나면 `axis_size - idx(t)` 개가 남는다.
packet clipper 는 이 가운데 가능한 만큼을 flit 에 채우되 `packet_span = 4` 로 상한을 둔다.
따라서 `fn valid_size(t) = clamp(10 - idx(t), 0, 4)` 이며, 뒤쪽 4 개 flit 위치는 항상 패딩이고 `R::SIZE` 가 `packet_span` 의 배수가 아니면 마지막 flit 이 부분 개수를 담는다:

```text
  flit 0: idx =  0  →  clamp(10 -  0, 0, 4) = 4  [████    ]
  flit 1: idx =  4  →  clamp(10 -  4, 0, 4) = 4  [████    ]
  flit 2: idx =  8  →  clamp(10 -  8, 0, 4) = 2  [██      ]
  flit 3: idx = 12  →  clamp(10 - 12, 0, 4) = 0  [        ]
```


## 전부 합치기

앞 절들은 하나 또는 두 차원에 배치된 단일 `R` 하위 표현식을 다뤘다.
실제로 Intra-Slice Reduce 는 단일 `REDUCE_LABEL`(`R`)에 더해 패딩된 비축약 축들을 받으며, VCG 는 그 전부를 추적한다. 패딩된 각 축(`R` 이든 다른 축이든)은 time filter 슬롯 하나를 차지하고, `R` 의 `Packet` 부분은 packet clipper 를 차지한다.
이 예제는 축 하나에서 셋까지 쌓아 올려 각 차원의 기여를 분명히 보여 준다.

원래 모양은 `[H, C, W] = [5, 5, 19]` 다.
각 축은 배치에 따라 slice/time/packet 부분으로 쪼개진다:

| 축 | 패딩 | Slice          | Time            | Packet         |
|------|--------|----------------|-----------------|----------------|
| `H`  | `# 8`  | `H # 8 / 2` (크기 4) | `H # 8 % 2` (크기 2)  | -                    |
| `C`  | `# 8`  | `C # 8 / 2` (크기 4) | `C # 8 % 2` (크기 2)  | -                    |
| `W`  | `# 24` | -                    | `W # 24 / 8` (크기 3) | `W # 24 % 8` (크기 8) |

이어지는 단계에서 간결함을 위해 `Ho`/`Co`/`Wo` 와 `Hi`/`Ci`/`Wi` 를 약칭으로 쓴다. `*o` 는 표의 해당 행에서 가장 왼쪽 인수(`H`/`C` 는 slice, `W` 는 time)이고, `*i` 는 그 오른쪽의 다음 인수다.

### 1 단계: `W=19` 만 (packet clipper, time filter 없음)

지금은 H 와 C 를 무시한다.
time filter 0 과 1 을 비활성화한다(`slice_mask=0, slice_thres=1`).
모든 slice 가 3 개 flit(`Wo` 크기 3)을 처리하고, packet clipper 는 톱니 모양을 만들어 낸다:

```text
valid_size: 8, 8, 3
              ^     ^
            full   19 - 16 = 3 (partial)
```

time filter 가 없으므로 모든 slice 가 정확히 같은 패턴을 받는다:

```text
All slices, all flits:
flit 0: ████████  (valid_size=8)
flit 1: ████████  (valid_size=8)
flit 2: ███       (valid_size=3)
```

### 2 단계: `C=5` 추가 (packet clipper + time filter 0)

이제 `C` 축 타이머(time filter 0)를 활성화한다.
`C=5` 는 `Co`(slice, 크기 4) × `Ci`(time, 크기 2)로 쪼개진다.
`C` time filter 설정은 `slice_mask=0b0011`(`slice_id` 에서 `Co` 를 추출), `slice_thres=2`, `time_thres=1`, `SliceMajor` 다.

이제 각 slice 는 6 개 flit 을 돌린다: `Ci` 크기 2 × `Wo` 크기 3.
`C` time filter 는 slice 를 그 `Co` 값으로 분류한다:

| `Co` | 그룹 | 효과 |
|------|-------|--------|
| 0 | 아래 (`< 2`) | 6 개 flit 전부가 packet clipper 의 패턴을 통과한다 |
| 1 | 아래 (`< 2`) | 동일 |
| 2 | 경계 (`= 2`) | `Ci=0` 에서 유효, `Ci=1` 에서 무효 |
| 3 | 위 (`> 2`) | 6 개 flit 전부 `valid_size = 0` |

slice 당 결과(6 개 flit = `Ci` 크기 2 × `Wo` 크기 3):

```text
Co=0:  [8,8,3, 8,8,3]   (both Ci steps valid)
Co=1:  [8,8,3, 8,8,3]   (same)
Co=2:  [8,8,3, 0,0,0]   (Ci=0 valid, Ci=1 invalid)
Co=3:  [0,0,0, 0,0,0]   (all invalid)
```

타이머의 효과에 주목한다.
어떤 slice 는 통째로 0 이 되고, 경계 slice 는 뒤쪽 절반을 잃는다.
하지만 유효한 flit 안에서는 packet clipper 가 만든 `[8,8,3]` 패턴이 그대로다.

### 3 단계: `H=5` 추가 (packet clipper + time filter 0 + time filter 1)

이제 `H` 축 타이머(time filter 1)를 활성화한다.
`H=5` 는 `Ho`(slice, 크기 4) × `Hi`(time, 크기 2)로 쪼개진다.
`H` time filter 설정은 `slice_mask=0b1100`(`slice_id` 에서 `Ho` 를 추출), `slice_thres=0b1000`, `time_thres=1`, `SliceMajor` 다.

slice id 는 두 slice 인수를 `slice_id = Ho * 4 + Co` 로 인코딩하며, 16 개 slice 가 된다.
이제 각 slice 는 12 개 flit 을 돌린다: `Hi` 크기 2 × `Ci` 크기 2 × `Wo` 크기 3.

| 구성 요소 | 축 | 추적 대상 | 설정 |
|-----|------|----------------|------------|
| packet clipper | `W=19` | flit 당 `R` 개수 | `axis_size=19`, `packet_span=8`, sequencer = `[Wo -> size 3 : stride 8]` |
| time filter 0 | `C=5` | slice 당 `Co` 유효성 | `slice_mask=0b0011`, `slice_thres=2`, `time_thres=1`, `SliceMajor` |
| time filter 1 | `H=5` | slice 당 `Ho` 유효성 | `slice_mask=0b1100`, `slice_thres=0b1000`, `time_thres=1`, `SliceMajor` |

`H` time filter 는 slice 를 `Ho` 로 분류하며, `C` time filter 가 `Co` 로 하는 것과 같은 논리다:

| `Ho` | 그룹 | 효과 |
|------|-------|--------|
| 0 | 아래 | 열림 |
| 1 | 아래 | 열림 |
| 2 | 경계 | `Hi=0` 에서 열림, `Hi=1` 에서 닫힘 |
| 3 | 위 | 닫힘 |

최종 `valid_size` 는 두 time filter 가 모두 `true` 를 반환하면 packet clipper 의 개수이고, 둘 중 하나라도 `false` 를 반환하면 `0` 이다.

아래의 전체 히트맵은 16 개 slice(열, `Ho` 로 묶음) × 12 개 flit(행, `(Hi, Ci)` 로 묶음)이다.
오른쪽 주석(`H:`, `C:`)은 각 행에서 어느 타이머가 작동하는지를 표시한다. `v` = 유효, `>` = 경계, `x` = 무효.
먼저 이 주석을 훑어 어느 행 × 열 블록이 전부 0 이어야 하는지(타이머 중 하나라도 `x`)와 데이터를 담는지를 예측한 다음, 셀을 읽어 `[8, 8, 3]` packet 차원 톱니를 확인한다.


```text
                     Ho=0       |Ho=1       |Ho=2       |Ho=3
                Co:  0  1  2  3 | 0  1  2  3| 0  1  2  3| 0  1  2  3
     H time filter: v  v  v  v  | v  v  v  v| >  >  >  >| x  x  x  x
     C time filter: v  v  >  x  | v  v  >  x| v  v  >  x| v  v  >  x
--------------------------------------------------------------------------------
 t= 0  Hi=0,Ci=0  W  8  8  8  0 | 8  8  8  0| 8  8  8  0| 0  0  0  0  H:v C:v
 t= 1             |  8  8  8  0 | 8  8  8  0| 8  8  8  0| 0  0  0  0
 t= 2             |  3  3  3  0 | 3  3  3  0| 3  3  3  0| 0  0  0  0
                                |           |           |
 t= 3  Hi=0,Ci=1  W  8  8  0  0 | 8  8  0  0| 8  8  0  0| 0  0  0  0  H:v C:>
 t= 4             |  8  8  0  0 | 8  8  0  0| 8  8  0  0| 0  0  0  0
 t= 5             |  3  3  0  0 | 3  3  0  0| 3  3  0  0| 0  0  0  0
                                |           |           |
 t= 6  Hi=1,Ci=0  W  8  8  8  0 | 8  8  8  0| 0  0  0  0| 0  0  0  0  H:> C:v
 t= 7             |  8  8  8  0 | 8  8  8  0| 0  0  0  0| 0  0  0  0
 t= 8             |  3  3  3  0 | 3  3  3  0| 0  0  0  0| 0  0  0  0
                                |           |           |
 t= 9  Hi=1,Ci=1  W  8  8  0  0 | 8  8  0  0| 0  0  0  0| 0  0  0  0  H:> C:>
 t=10             |  8  8  0  0 | 8  8  0  0| 0  0  0  0| 0  0  0  0
 t=11             |  3  3  0  0 | 3  3  0  0| 0  0  0  0| 0  0  0  0

Legend: `v` = below (all valid), `>` = boundary (partial), `x` = above (all invalid)
```

- `Ho=3` 열(가장 오른쪽 4 개): 전부 0(`H` time filter `x`, 항상 닫힘).
- `Co=3` 열(4 번째마다): 전부 0(`C` time filter `x`).
- `Co=2` 열(`H:v C:>`): `C` time filter 가 경계이므로 `Ci=0` 인 행만 통과한다. `Co=1` 과 `Co=2` 를 비교한다.
- `Ho=2` 열(`H:> C:v`): `H` time filter 가 경계이므로 `Hi=0` 인 행만 통과한다. `Ho=1` 과 `Ho=2` 를 비교한다.
- `Ho=2 × Co=2`(둘 다 `>`): 두 경계의 교집합인 `(Hi=0, Ci=0)` 행만 통과한다.
- 유효한 셀 안에서는 packet clipper 가 만든 `[8, 8, 3]` 톱니가 slice 와 무관하게 늘 동일하게 나타난다.


<a id="inexpressible-patterns"></a>
## 표현할 수 없는 패턴

다음 배치들은 VCG 로 표현할 수 없다.
각 소절은 해당 배치를 보이고 그 이유를 설명한다.

### `Slice` 안의 `R`, 순서 어긋남

`R` 이 `Slice` 안에 여러 하위 표현식을 가질 때, 바깥쪽 하위 표현식은 안쪽 것들보다 큰 스트라이드를 가져야 한다.
이 순서를 뒤집으면 slice 별 `R` 인덱스 범위가 단조롭지 않게 되어 단일 `slice_thres` 로는 포착할 수 없다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![R = 13, X = 32];

// NOT supported: inner sub-expression (/ 2 % 4, stride 2) placed outside major (/ 8, stride 8) in Slice.
// Produces non-monotonic slice validity (S6 valid after S5 partial); VCG cannot express this.
fn reduce_wrong_ordering<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1 # 2], m![X, R # 16 / 2 % 4, R # 16 / 8], m![R # 16 % 2], m![1 # 8], i32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1 # 2], m![X, R # 16 / 2 % 4, R # 16 / 8], m![1], m![1 # 4], i32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_trim::<m![1 # 4]>()
        //   Slice     = m![X, R # 16 / 2 % 4, R # 16 / 8]
        //   Time      = m![R # 16 % 2]
        //   Packet    = m![1 # 8]
        //   OutTime   = m![1]          (R eliminated)
        //   OutPacket = m![1 # 4]
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::Min,
        )
}
# 
# let mut ctx = Context::acquire();
# 
# let i: VectorBranchTensor<'_, _, i32, m![1], m![1 # 2], m![X, R # 16 / 2 % 4, R # 16 / 8], m![R # 16 % 2], m![1 # 8], i32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let _o = reduce_wrong_ordering(i);
```

### `Slice` 와 `Time` 안의 `R`, 뒤섞임

`R` 의 하위 표현식이 Slice 인수가 두 Time 인수 사이에 놓이도록 분배되면, slice 마다 필요한 유효 time step 개수가 달라진다.
단일 `slice_thres` 로는 이런 slice 별 편차를 표현할 수 없다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![R = 13, X = 64];

// NOT supported: Time-Slice-Time interleave.
// Different slices need different valid step counts (e.g., S2: 3/4, S3: 2/4); single threshold cannot express this.
fn reduce_wrong_interleave<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1 # 2], m![X, R # 16 / 2 % 4], m![R # 16 / 8, R # 16 % 2], m![1 # 8], i32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1 # 2], m![X, R # 16 / 2 % 4], m![1], m![1 # 4], i32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_trim::<m![1 # 4]>()
        //   Slice     = m![X, R # 16 / 2 % 4]
        //   Time      = m![R # 16 / 8, R # 16 % 2]
        //   Packet    = m![1 # 8]
        //   OutTime   = m![1]          (R eliminated)
        //   OutPacket = m![1 # 4]
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::Min,
        )
}
# 
# let mut ctx = Context::acquire();
# 
# let i: VectorBranchTensor<'_, _, i32, m![1], m![1 # 2], m![X, R # 16 / 2 % 4], m![R # 16 / 8, R # 16 % 2], m![1 # 8], i32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let _o = reduce_wrong_interleave(i);
```

### `Slice` 와 `Time` 안의 `R`, 과도 패딩

`TimeMajor` 모드는 `PADDED_SIZE - R::SIZE ≤ slice_span` 을 요구한다. 최대 `slice_span` 개의 `R` 위치만 과도 패딩된다.

아래 예제에서 `R = 14`, `Slice = m![X, R # 20 % 4]`(`slice_span = 4`), 그리고 `A = 3` 인 `Time = m![A, R # 20 / 4]` 다(따라서 `R # 20 / 4` 로부터 `time_span = 5` 이고, `Time::SIZE = A × time_span = 15` 다).
제약은 `Time::SIZE` 가 아니라 `time_span` 에 걸린다. `A` 같은 `Time` 안의 비-`R` 축은 `R` 의 인덱스를 바꾸지 않고 순환하므로 제약에 영향을 주지 않는다.
올바른 패딩은 `R # 16` 이며(`16 - 14 = 2 ≤ slice_span = 4` 이므로) `time_span = 4` 를 준다.
`R # 20` 을 쓰면 `R` 이 과도 패딩되어 `time_span = 5` 가 되고, 실제 데이터가 없는 `Time` 반복이 하나 더 붙는다.
`slice_thres` 아래 slice(slice 기여값 = 0)에서는 `R # 20 / 4 = 4` 일 때 sequencer 로 복원한 `idx` 가 `4 × 4 = 16` 에 도달해 `R` 인덱스 16 ≥ `R::SIZE` 가 된다. 도달해서는 안 될 패딩이다.
`Emulation` 이 정확히 이것을 잡아낸다. 과도 패딩된 배치 위에서 축약하면 낡고 도달 불가능한 `R` 인덱스로 실행되는 대신 패닉(`out x residue must factor the operand`)이 난다.

```rust,should_panic
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 3, R = 14, X = 64];

fn reduce_time_major_wrong<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1 # 2], m![X, R # 20 % 4], m![A, R # 20 / 4], m![1 # 8], i32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1 # 2], m![X, R # 20 % 4], m![A], m![1 # 4], i32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_trim::<m![1 # 4]>()
        //   Slice     = m![X, R # 20 % 4]
        //   Time      = m![A, R # 20 / 4]   (A is non-R; time_span = 5 from R # 20 / 4)
        //   Packet    = m![1 # 8]
        //   OutTime   = m![A]              (R eliminated; A survives)
        //   OutPacket = m![1 # 4]
        // NOT supported: R # 20 over-pads (20 - 14 = 6 > slice_span = 4).
        // time_span = 5 > 4. Below-group slices include time steps where R # 20 / 4 = 4 (R = 16 padding).
        .vector_intra_slice_reduce::<R, m![A], m![1 # 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
}
# 
# let mut ctx = Context::acquire();
# 
# let i: VectorBranchTensor<'_, _, i32, m![1], m![1 # 2], m![X, R # 20 % 4], m![A, R # 20 / 4], m![1 # 8], i32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let _o = reduce_time_major_wrong(i);
```

### `Packet` 안의 `R`, 복잡한 형태

packet clipper 는 `Packet = m![R # PADDED_SIZE % packet_span # 8]` 을 요구한다.
다른 형태는 `fn valid_size()` 가 의존하는 연속 접두 성질을 깨뜨린다.

첫 번째 예제는 `R` 의 상위 부분을 `Packet` 에 배치하므로(`R # 24 % 8` 대신 `R # 24 / 8` 형태), 접두부가 `R` 의 다음 연속 구간을 담는 대신 서로 다른 `R` 스트라이드의 위치들을 섞는다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, R = 19, X = 64];

fn reduce_wrong_packet_outer<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1 # 2], m![X, A / 2], m![R # 24 % 8], m![R # 24 / 8 # 8], f32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1 # 2], m![X, A / 2], m![1], m![1 # 4], f32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_trim::<m![R # 24 / 8 # 4]>()
        //   Slice     = m![X, A / 2]
        //   Time      = m![R # 24 % 8]
        //   Packet    = m![R # 24 / 8 # 8]   (NOT supported: major R in Packet)
        //   OutTime   = m![1]          (R eliminated)
        //   OutPacket = m![1 # 4]
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
}
# 
# let mut ctx = Context::acquire();
# 
# let i: VectorBranchTensor<'_, _, f32, m![1], m![1 # 2], m![X, A / 2], m![R # 24 % 8], m![R # 24 / 8 # 8], f32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let _o = reduce_wrong_packet_outer(i);
```

두 번째 예제는 `R` 이 다른 축 `A` 와 `Packet` 을 공유하므로, `A` 의 원소가 접두 기반 개수 계산에서 패딩으로 취급되는 위치를 차지한다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 4, R = 19, X = 256];

fn reduce_wrong_mixed_packet<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1 # 2], m![X], m![R # 24 / 2], m![R # 24 % 2, A], f32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1 # 2], m![X], m![1], m![A], f32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_split::<m![R # 24], m![A]>()
        //   at Vector Init:
        //   Slice     = m![X]
        //   Time      = m![R # 24 / 2]
        //   Packet    = m![R # 24 % 2, A]   (NOT supported: A shares Packet with R)
        //   OutTime   = m![1]          (R eliminated; A silently excluded by prefix valid_size)
        //   OutPacket = m![A]
        .vector_intra_slice_reduce::<R, m![1], m![A]>(
            IntraSliceReduceOpF32::Add,
        )
}
# 
# let mut ctx = Context::acquire();
# let i: VectorBranchTensor<'_, _, f32, m![1], m![1 # 2], m![X], m![R # 24 / 2], m![R # 24 % 2, A], f32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let _o = reduce_wrong_mixed_packet(i);
```

### `Slice` 와 `Packet` 안의 `R`

`R` 이 `Slice` 와 `Packet` 으로 나뉘는 경우다.
왜 표현할 수 없는지 보려면 `R = 2045`(`R # 2048` 로 패딩)를 `Slice = m![R # 2048 / 8]`(256 개 slice)와 `Packet = m![R # 2048 % 8]`(8 원소 flit)으로 쪼갠 경우를 생각한다.
유일한 time step `t = 0` 에서 `fn valid_size` 는 모든 slice 에 대해 `clamp(2045 - 0, 0, 8) = 8` 을 계산한다.
그러나 slice 255 의 flit 은 `R` 인덱스 2040–2047 을 담고 있고, 그중 다섯 개만 실제 데이터다:

`t = 0`, `slice = 255` 에서:

| flit 위치 | 0    | 1    | 2    | 3    | 4    | 5     | 6     | 7     |
|---------------|------|------|------|------|------|-------|-------|-------|
| `R` 인덱스     | 2040 | 2041 | 2042 | 2043 | 2044 | 2045  | 2046  | 2047  |
| 유효?        | 예  | 예  | 예  | 예  | 예  | [pad] | [pad] | [pad] |

slice 0–254 는 정당하게 `valid_size = 8` 이 필요하지만, slice 255 는 `valid_size = 5` 가 필요하다.
`fn valid_size(t)` 는 주어진 `t` 에 대해 값 하나만 반환할 수 있으므로 어떤 단일 설정으로도 되지 않는다.

`R::SIZE % packet_span = 0`(모든 packet 이 꽉 차거나 비어 있음)인 퇴화된 하위 경우는 Slice 만 있는 경우로 환원되며 지원된다.

```rust,should_panic
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![R = 2045];

// NOT supported: R = 2045 split across Slice (/ 8, 256 slices) and Packet (% 8).
// Slices 0-254 need valid_size = 8; slice 255 needs valid_size = 5. fn valid_size(t) cannot vary by slice.
fn reduce_wrong_slice_packet<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1 # 2], m![R # 2048 / 8], m![1], m![R # 2048 % 8], i32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1 # 2], m![R # 2048 / 8], m![1], m![1 # 4], i32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_trim::<m![R # 2048 % 4]>()
        //   Slice     = m![R # 2048 / 8]
        //   Time      = m![1]
        //   Packet    = m![R # 2048 % 8]
        //   OutTime   = m![1]          (R eliminated)
        //   OutPacket = m![1 # 4]
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
}
# 
# let mut ctx = Context::acquire();
# let i: VectorBranchTensor<'_, _, i32, m![1], m![1 # 2], m![R # 2048 / 8], m![1], m![R # 2048 % 8], i32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let _o = reduce_wrong_slice_packet(i);
```

## 제약

| 구성 요소 | 용량 |
|-----------|----------|
| Packet clipper | 1 개 |
| Time filter | 3 개 |
| time filter / packet clipper 당 sequencer 항목 | 8 ([Sequencer](../../moving-tensors/sequencer.md) 참고) |

유효성 추적이 필요한 패딩된 축은 각각 time filter 하나 또는 packet clipper 를 차지한다.
한 번의 호출에서 최대 4 개 축을 추적할 수 있다(packet clipper 1 개 + time filter 3 개).
패딩되지 않은 축은 슬롯이 필요 없다(`slice_mask=0, slice_thres=1` 이 time filter 를 비활성화해 항상 `true` 를 반환하게 한다).

intra-slice reduce 는 단일 `REDUCE_LABEL` 을 받으므로, "다축"이란 동시에 여러 축약을 한다는 뜻이 아니라 축약 축 `R` 하나에 패딩된 비축약 축들이 더해진다는 뜻이다.


## 하류의 4-Way 연산


VCG 는 어떤 narrowing 도 하기 전에 8-way flit 마다 `valid_size` 를 만들어 낸다.
하류의 `Narrow` 단계는 각 8-way flit 을 4-way 절반들로 쪼개며, narrow 를 적용하는 방식이 각 `valid_size` 가 절반들 사이에 어떻게 나뉘는지를 결정한다.

| 연산 | 입력 | 출력 | Valid Count 변환 |
|-----------|-------|--------|---------------------------|
| `vector_narrow_split` | 8-way flit (`valid_size = v`) | 4-way flit 두 개 | 하위: `min(v, 4)`, 상위: `max(v - 4, 0)` |
| `vector_narrow_trim` | 8-way flit (`valid_size = v`) | 4-way flit 한 개 | `min(v, 4)` |
| `vector_widen_concat` | 4-way flit 두 개 (`v_low`, `v_high`) | 8-way flit | `v_low + v_high` |
| `vector_widen_pad` | 4-way flit | 8-way flit | 변화 없음 |

`vector_narrow_split` 과 `vector_widen_concat` 은 접두 성질을 보존한다.
`vector_narrow_trim` 의 경우 매핑이 `v <= 4` 를 정적으로 보장해야 한다.
상위 4 개 원소가 유효할 수 있다면 그것들을 잘라 내면서 데이터를 잃게 된다.
