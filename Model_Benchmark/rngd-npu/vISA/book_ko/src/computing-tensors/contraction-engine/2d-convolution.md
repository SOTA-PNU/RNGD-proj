# 2D Convolution


2D convolution 은 einsum `$(H + Fh)$(W + Fw)K, FhFwKC -> HWC` 이며, 공간 출력 축은 `H`, `W`, 출력 채널은 `C`, 축약 축은 `Fh`, `Fw`, `K` 다.
`$(W + Fw)` 슬라이딩은 [Stream Adapter 의 shift-reuse](#shift-stream-shift-unit) 로 매핑된다 (아래에 설명한다).

[변형](#variants) 의 네 가지 변형은 Stream Adapter 가 슬라이딩 윈도우를 시프트하는 방식(conv 전용 기구)에서 서로 다르다.
어떤 축을 `Time` 에 둘지 고르는 문제는 [matmul 매핑 논의](./index.md#example-batched-matmul) 와 같은 트레이드오프를 따르므로 반복하지 않는다.

<a id="variants"></a>
## 변형

### 필터 스트라이드 1

스트라이드 1 컨볼루션에서는 데이터가 Stream Adapter 에 닿기 전에 Fetch Engine 이 `$(H+Fh)` 슬라이딩을 처리한다.
그다음 Stream Adapter 가 shift-reuse 로 `$(W+Fw)` 차원을 처리하여 계산에서 `Fw, W` 슬라이딩을 만든다.
아래 예제는 shift-stride 1 로 시프트를 두 번 한다.

```text
// Configuration: input_type = bf16, trf_type = bf16
//        Input mapping: [ H: [H=30, Fh=3, K=32, $(W=30 + Fw=3)=32] ] (1)
//          TRF mapping: [ Lane: [C=8] | H: [K=32, C=24, Fh=3, Fw=3] ] (1)
//  Contraction mapping: [ H: [H=30, C/8=3, Fh=3, K=32, Fw=3] | Lane: [C=8] | T: [W=30+2#] ] (1)
// Accumulation mapping: [ H: [H=30, C=32] | T: [W=30+2#] ] (1)
```

### 필터 스트라이드 2

스트라이드 2 컨볼루션에서는 shift-stride 2 (출력 위치마다 시프트 한 번) 가 스트라이드된 윈도우를 뽑아낸다.
입력 표현식 `$(W:2=15 + Fw=4)=32` 는 `Fw/2=2, (W=15, Fw=2)` 로 인수분해되는데, 스트라이드 `:2` 인 크기 2 축을 외적으로 뽑아낸 것이다. 즉 `$(W:2=15 + (Fw/2:2=2, Fw=2))=32` 가 `Fw/2=2, $(W:2=15, Fw=2)` 가 된다.

```text
// Configuration: input_type = bf16, trf_type = bf16
//        Input mapping: [ H: [H=15, Fh=4, K=32, $(W:2=15 + Fw=4)=32] ] (1)
//          TRF mapping: [ Lane: [C=8] | H: [K=32, C=24, Fh=4, Fw=4] ] (1)
//  Contraction mapping: [ H: [H=15, C/8=3, Fh=4, K=32, Fw/2=2] | Lane: [C=8] | T: [W=15+1#, Fw=2] ] (1)
// Accumulation mapping: [ H: [H=15, C=32] | T: [W=15+1#] ] (1)
```

앞의 예제는 시프트 버퍼가 가득 차지 않아 MAC 을 충분히 쓰지 못한다.
`feed_flits` 를 (기본값 2 에서) 3 으로 설정하면 시프트 버퍼에 flit 을 더 채워 MAC 활용률을 100 % 로 만든다.
그러면 변환 `$(W:2=16 + Fw=4)=34` 가 `Fw/2=2, (W=16, Fw=2)` 를 만든다.

```text
// Configuration: feed_flits = 3, input_type = bf16, trf_type = bf16
//        Input mapping: [ H: [H=16, Fh=4, K=32, $(W:2=16 + Fw=4)=34] ] (1)
//          TRF mapping: [ Lane: [C=8] | H: [K=32, C=24, Fh=4, Fw=4] ] (1)
//  Contraction mapping: [ H: [H=16, C/8=3, Fh=4, K=32, Fw/2=2] | Lane: [C=8] | T: [W=16, Fw=2] ] (1)
// Accumulation mapping: [ H: [H=16, C=32] | T: [W=16] ] (1)
```

### 딜레이션 2

딜레이션 2 컨볼루션에서는 필터가 스트라이드 2 만큼 떨어진 입력 위치를 샘플링한다.
shift-stride 2 로 시프트를 2 번 하면 이 확장된 필터 위치들을 뽑아낸다.

변환 `$(W=27 + Fw:2=3)=32` 는 `Fw=3, W=27` 을 만드는데, 선형결합에서 스트라이드 `:2` 인 크기 3 축을 외적으로 뽑아낸 것이다. 즉 `$(W=27 + Fw:2=3)=32` 가 `Fw=3, $(W=27)` 이 된다.

```text
// Configuration: input_type = bf16, trf_type = bf16
//        Input mapping: [ H: [H=27, Fh=3, K=32, $(W=27 + Fw:2=3)=32] ] (1)
//          TRF mapping: [ Lane: [C=8] | H: [K=32, C=24, Fh=3, Fw=3] ] (1)
//  Contraction mapping: [ H: [H=27, C/8=3, Fh=3, K=32, Fw=3] | Lane: [C=8] | T: [W=27+5#] ] (1)
// Accumulation mapping: [ H: [H=27, C=32] | T: [W=27+5#] ] (1)
```

### 필터 스트라이드 2, 딜레이션 2

스트라이드 2 와 딜레이션 2 를 결합하면 딜레이션 2 단독일 때와 비슷한 시프트 연산이 필요하다.

변환 `$(W:2=14 + Fw:2=3)=31 + 1#` 은 `Fw=3, W=14, 1+1#` 을 만드는데, 선형결합에서 스트라이드 `:2` 인 크기 3 축을 외적으로 뽑아낸 것이다. 즉 `$(W:2=14 + Fw:2=3)=31` 이 `Fw=3, $(W:2=14)` 가 된다.

TRF 는 더미 슬롯에 0 을 담아야 한다. 그래야 `1+1#` 을 `1+1z` 와 축약할 때 1 이 나온다. 그러지 않으면 임의값 `1#` 이 나오는데, 이는 `1+1#` 을 `1+1#` 과 축약한 결과다.
표기 `1z` 는 `1#` (더미 패딩) 과 같지만 임의값 대신 0 으로 채운다.

```text
// Configuration: input_type = bf16, trf_type = bf16
//        Input mapping: [ H: [H=14, Fh=3, K=32, $(W:2=14 + Fw:2=3)=31+1#] ] (1)
//          TRF mapping: [ Lane: [C=8] | H: [K=32, C=24, Fh=3, Fw=3, 1+1z] ] (1)
//  Contraction mapping: [ H: [H=14, C/8=3, Fh=3, K=32, Fw=3] | Lane: [C=8] | T: [W=14+2#, 1+1z] ] (1)
// Accumulation mapping: [ H: [H=14, C=32] | T: [W=14+2#] ] (1)
```

## 컨볼루션을 위한 Stream Adapter 기구

컨볼루션 워크로드는 같은 입력 원소를 다시 가져오지 않도록 슬라이딩 윈도우 데이터 재사용이 필요하다.
[Stream Adapter](./outer.md#stream-adapter) 는 세 가지 확장으로 이 재사용을 제공한다. 3-flit 수집, transpose, shift-and-reuse 다.

einsum (행렬곱) 워크로드만 다룬다면 이 절은 건너뛴다.

### Flit Buffer: feed_flits 3

`feed_flits: 3` 은 Flit Buffer 를 기본 2 flit 용량 너머로 확장하여, 연속된 32 B flit 세 개로 96 바이트를 전부 채운다.
세 번째 flit 은 [Stream Shift Unit](#shift-stream-shift-unit) 에 다시 가져오지 않고도 윈도우를 시프트할 만큼의 버퍼 데이터를 주며, 그 대가로 Packet 마다 flit 하나 분량의 버퍼링이 더 든다.

| 매개변수 | 값 | 설명 |
|-----------|-------|-------------|
| `feed_flits: 3` | 96 바이트 | shift-reuse 를 위해 flit 3 개를 모두 예약 |

Stream Shift Unit 이 그 추가 flit 을 어떻게 쓰는지는 [Shift](#shift-stream-shift-unit) 를 참고한다.

### Transpose

Packet Reducer 는 가장 안쪽 축부터 인접한 쌍을 리듀스하므로, 축약 축이 가장 안쪽에 있어야 한다.
들어오는 데이터의 축 순서가 다르면 Transpose 가 32 B flit 안에서 축 순서를 재배열한다.

#### 지원하는 Transpose

지원하는 transpose 는 데이터 타입에 따라 다르다 (총량은 항상 32 B 다).

| 데이터 타입 | 지원하는 Transpose |
|-----------|---------------------|
| int4      | `[4][16] → [16][4]` |
| i8/fp8    | `[2][16] → [16][2]`, `[4][8] → [8][4]` |
| bf16      | `[2][8] → [8][2]` |

> 이 타입들이 Contraction Engine 이 계산할 수 있는 타입이다.
> `i32` 나 `f32` 같은 타입은 Packet Reducer 의 리듀스 트리를 쓸 수 없다.

#### 예제 1: flit 안에서의 Transpose

```text
axes![A = 3, B = 2, C = 2, D = 8];

// Input: time -> [1], num_flits -> [3_a], flit -> [2_b × 2_c × 8_d], i8
//
// Possible transpose outputs:
//   1. [2][16] → [16][2]: flit -> [2_c × 8_d × 2_b]
//   2. [4][8]  → [8][4]:  flit -> [8_d × 2_b × 2_c]
//
// From the Stream Shift Unit onward, num_flits × flit = flits:
//   1. flits -> [3_a × 2_c × 8_d × 2_b]
//   2. flits -> [3_a × 8_d × 2_b × 2_c]
```

#### 예제 2: 축약 매핑에서의 Transpose

```text
axes![P = 64, A = 2, B = 2, C = 16];

// Configuration: feed_flits = 2, datatype = bf16, transpose_flit(32B) = true
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, c_1=16, b_1=2] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2] | Lane: [] | T: [b_1=2, c_1=16] ] (16)
```

- `c_1=16, b_1=2` 가 `b_1=2, c_1=16` 이 되어, Packet Reducer 리듀스를 위해 `b` 를 가장 안쪽 위치로 옮긴다.
- 축 순서가 제대로 잡히면 Stream Adapter 가 슬라이딩 윈도우 연산에 shift-and-reuse 를 적용한다.

<a id="shift-stream-shift-unit"></a>
### Shift (Stream Shift Unit)

Stream Shift Unit 은 컨볼루션 같은 슬라이딩 윈도우 연산을 위해 shift-and-reuse 를 수행한다.
겹치는 윈도우를 여러 번 가져오는 대신, 데이터를 한 번 가져와 시프트해서 여러 윈도우를 만든다.


시프트는 세 개의 매개변수가 제어한다.

- **`initial_shift`**: 데이터를 처음 적재할 때의 시작 오프셋.
- **`shift_stride`**: 시프트 차원을 따라 반복마다 시프트할 양.
- **`pop_dim`**: 새 데이터를 가져오도록 유발하는 차원.

#### initial_shift

`initial_shift` 매개변수는 데이터가 시프트 버퍼에 들어올 때의 시작 오프셋을 정한다.
버퍼는 낮은 주소에서 높은 주소 순으로 원소를 담는다.
`initial_shift` 가 음수면 데이터를 뒤쪽 위치(높은 주소, 상위 비트)로 밀고, 앞쪽 위치는 0 으로 패딩한다.
`initial_shift` 가 양수면 앞쪽 위치(낮은 주소, 하위 비트)로 밀고, 뒤쪽 위치를 패딩한다.

##### 유효 범위

| 데이터 타입 | 범위 | 서로 다른 값의 개수 |
|-----------|-------|-----------------|
| i4  | -15 ~ 16 | 32 |
| i8  | -7 ~ 8   | 16 |
| bf16 | -3 ~ 4   | 8  |

서로 다른 값의 개수(32, 16, 8)는 32 B flit 안 원소 개수의 절반에 해당하며, Stream Shift Unit 의 버퍼 용량과 맞아떨어진다.
범위가 비대칭인(음수보다 양수가 하나 더 많은) 이유는, 서로 다른 위치의 개수를 2 의 거듭제곱으로 인코딩하면서 0 이 정확한 중간점에 오지 않기 때문이다.

`initial_shift` 를 적용한 뒤 출력은 Packet Reducer 를 위해 64 B 로 잘리고, 패딩은 0 으로 채워진다.

##### 예제: 음수 initial_shift

```text
axes![A = 96, B = 3];

// Input: time -> [3_b], flits -> [a], i8
// initial_shift = -7
//
// After initial shift: time -> [3_b], Lane -> [7_pad + a.slice(57)]
//   - Left portion is zero-padded
//   - After shifting by 7, only the lower 64 B are sliced and output
//   - For feed_flits = 1 or 2, portions beyond the actual fetched region are zero-padded
```

##### 예제: 양수 initial_shift

```text
axes![A = 96, B = 3];

// Input: time -> [3_b], flit -> [a], i8
// initial_shift = 8
//
// After initial shift: time -> [3_b], Lane -> [a.offset(8).slice(64)]
//   - 8-element shift + 64 B slicing
//   - For feed_flits ≠ 3, the upper 8 bytes are zero-padded
```

##### 인덱스별 initial_shift

`initial_shift_dim` 매개변수는 인덱스마다 다른 시프트를 선택한다.

| `initial_shift_dim` | 동작 |
|---------------------|----------|
| 8 | 모든 인덱스에 대해 단일 시프트 값 |
| 0..7 | 인덱스 값에 따라 `initial_shift_elements[i]` 사용 |

```text
axes![A = 96, B = 3];

// Input: time -> [3_b], flits -> [a], i8
// seq_limits: [3, 1, 1, 1, 1, 1, 1, 1] (flits not shown in index)
// initial_shift_dim = 0 (b-axis)
// initial_shift_elements = [-7, 8, 0]
//
// at b = 0: Lane -> [7_pad + a.slice(57)]
// at b = 1: Lane -> [a.offset(8).slice(64)]
// at b = 2: Lane -> [a.slice(64)]
```

`initial_shift` 는 flit 이 pop 될 때 한 번 적용되고, 재사용/시프트 중에는 적용되지 않는다.

##### 예제: 음수 initial_shift (축약 매핑)

```text
axes![P = 64, A = 2, C = 32];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift = -1, init_shift_range = (-3, 4)
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, [[c_1=(1,31)]+1]=32] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2] | Lane: [] | T: [c_1=32] ] (16)
```

- `init_shift: -1` 이면 데이터가 왼쪽으로 원소 1 개만큼 밀리고, 시작 부분에 0 패딩 원소 1 개가 추가된다.
- `[[c_1=(1,31)]+1]`: 첫 원소는 패딩이고, 31 개 원소가 원본 데이터다.

##### 예제: 양수 initial_shift (축약 매핑)

```text
axes![P = 64, A = 2, C = 32];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift = 1, init_shift_range = (-3, 4)
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, [1+[c_1=31]]=32] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2] | Lane: [] | T: [[[c_1=31]+1]=32] ] (16)
```

- `init_shift: 1` 이면 데이터가 오른쪽으로 원소 1 개만큼 밀려, 첫 원소가 버려지고 끝에 0 패딩 1 개가 추가된다.
- Switch Engine 매핑 `[1+[c_1=31]]`: 첫 원소가 포함된다.
- 축약 매핑 `[[c_1=31]+1]`: 데이터 원소 31 개 뒤에 패딩 원소 1 개가 온다.

##### 예제: 간접 벡터를 사용한 인덱스별 initial_shift

```text
axes![P = 64, A = 2, C = 32];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift_tag = a_1, init_shifts = [-1, 1], init_shift_range = (-3, 4)
//   indirect_vecs: [I0 = (c_1=32)[1, -1]]
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2 @ I0_1, c_1=32] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2] | Lane: [] | T: [[[c_1=31]+1]=32] ] (16)
```

- `init_shift_tag: a_1` 과 `init_shifts: [-1, 1]` 이면 인덱스마다 시프트가 달라진다.
- `a_1=0` 에서는 -1 만큼(왼쪽) 시프트한다. `a_1=1` 에서는 1 만큼(오른쪽) 시프트한다.
- 간접 벡터 `[1, -1]` 은 `c_1=32` 에서 각 `a_1` 인덱스마다 어떤 원소를 고를지 제어한다.

##### 예제: 인터리브된 슬라이딩 윈도우와 initial_shift

```text
axes![P = 64, A = 2, B = 31, C = 3];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift = -1, init_shift_range = (-3, 4), shift_stride = 1
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, $[(b_1=(1,30):1)+(c_1=3:1)]=32] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2, c_1=3] | Lane: [] | T: [[[b_1=31]+1]=32] ] (16)
```

- `$[(b_1=(1,30):1)+(c_1=3:1)]` 은 인터리브된 패턴이다. `b_1` 은 오프셋 1 에서 스트라이드 1 로 시작하고, `c_1` 은 스트라이드 1 로 3 회 반복한다.
- 초기 시프트 -1 이 시작 부분에 패딩 원소 1 개를 추가하여 `[[b_1=31]+1]` 을 만든다.

#### shift_stride 와 pop_dim

`pop_dim` 매개변수는 새 데이터를 언제 가져올지 표시한다. `pop_dim` 의 인덱스가 증가하면 새 데이터가 적재되고 `initial_shift` 가 다시 적용된다.
`shift_dim` 은 재사용이 일어나는 차원이다. `shift_dim` 을 따라 반복할 때마다 `shift_stride` 개 원소만큼 시프트가 추가로 적용된다.

`pop_dim` 아래에 있으면서 `shift_dim` 이 아닌 인덱스는 타일링된(브로드캐스트된) 출력을 만들며, 그래서 차원 순서는 `tile → shift_dim → pop_dim` (안쪽에서 바깥쪽) 이 된다.

##### 유효한 `shift_stride` 범위

| 데이터 타입 | 범위 |
|-----------|-------|
| i4  | 0 ~ 31 |
| i8  | 0 ~ 15 |
| bf16 | 0 ~ 7  |

##### 예제: pop_dim 과 함께 쓰는 shift_stride

```text
axes![A = 96, B = 3];

// initial_shift = -1, shift_dim = 1, shift_stride = 3, pop_dim = 2
// seq_limits: [2, 3, 3, 1, 1, 1, 1, 1]
// Input: time -> [b = 3], flits -> [a], i8
//
// flits #0 (indexer: [0, 0, 0]): Apply initial shift.
//   [1_pad + a], slice to 64 → Lane -> [(1_pad + a) % 63]
// flits #1 (indexer: [1, 0, 0]): Same as #0 (dim0 is not shift_dim → tiling)
// flits #2 (indexer: [0, 1, 0]): shift_dim=1, apply shift_stride=3 from #0 state.
//   [a @ 2], slice to 64 → Lane -> [(a @ 2) % 64]
// flits #3 (indexer: [1, 1, 0]): Same as #2 (dim0 is not shift_dim → tiling)
// flits #4 (indexer: [0, 2, 0]): shift_dim=1, apply shift_stride=3 from #2 state.
//   [a @ 5], slice to 64 → Lane -> [(a @ 5) % 64]
// flits #5 (indexer: [1, 2, 0]): Same as #4 (dim0 is not shift_dim → tiling)
// flits #6 (indexer: [0, 0, 1]): pop_dim=2, fetch new flits and apply initial shift.
//   [1_pad + a], slice to 64 → Lane -> [(1_pad + a) % 63]
//
// Output mapping: time -> [b × (1_pad + a) / f=3:3 × Broadcast=2], Lane -> [(1_pad + a) / w=64:1]
```

#### Shift 예제

##### 스트라이드 1 Shift

```text
axes![P = 64, A = 2, B = 31, C = 3];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift = -1, init_shift_range = (-3, 4), shift_stride = 1
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, $[(b_1=(1,30):1)+(c_1=3:1)]=32] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2, c_1=3] | Lane: [] | T: [[[b_1=31]+1]=32] ] (16)
```

##### 스트라이드 2 Shift

```text
axes![P = 64, A = 2, B = 16, C = 4];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift = -2, init_shift_range = (-3, 4), shift_stride = 2
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, $[(b_1=(1,15):2)+(c_1=4:1)]=32] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2, c_2=2] | Lane: [] | T: [b_1=16, c_1=2] ] (16)
```

##### shift 없는 데이터 재사용 (타일링)

```text
axes![P = 64, A = 2, C = 32];

// Configuration: feed_flits = 2, datatype = bf16
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, c_1=32] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2, #t_1=5] | Lane: [] | T: [c_1=32] ] (16)
```

- `#t_1=5` 는 시프트 없이 같은 데이터를 5 번 브로드캐스트한다.

##### shift 차원과 함께 쓰는 pop_dim

```text
axes![P = 64, A = 2, C = 32];

// Configuration: feed_flits = 2, datatype = bf16
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, c_1=32] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2, #s_1=3, #t_1=5] | Lane: [] | T: [c_1=32] ] (16)
```

- `#s_1=3`: 시프트 차원으로, 데이터가 3 번 시프트된다 (shift-and-reuse).
- `#t_1=5`: pop_dim 으로, 결과를 5 번 브로드캐스트(타일링)한다.
- 새 데이터는 가장 바깥 차원을 넘어갈 때만 가져온다.

##### 슬라이딩 윈도우·타일링과 함께 쓰는 pop_dim

```text
axes![P = 64, A = 2, B = 31, C = 3];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift = -1, init_shift_range = (-3, 4), shift_stride = 1
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, $[(b_1=(1,30):1)+(c_1=3:1)]=32] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2, c_1=3, #t_1=5] | Lane: [] | T: [[[b_1=31]+1]=32] ] (16)
```

- `c_1=3` 이 스트라이드 1 인 시프트 차원 역할을 하여, 데이터 버퍼마다 시프트된 윈도우 3 개를 생성한다.
- `#t_1=5` 는 그 윈도우 3 개 각각을 5 번 타일링(브로드캐스트)한다.
- 새 데이터는 `c_1` 과 `#t_1` 반복이 모두 끝난 뒤에만 가져온다.

### 제약

하드웨어는 각 고급 연산을 제한하며, 아래 표는 그 한계와 물리적 원인을 정리한다.

| 제약 | 한계 | 원인 |
|------------|-------|-------|
| Flit Buffer 용량 | `feed_flits` ∈ {1, 2, 3} | 96 바이트 물리 버퍼 (레지스터 파일 저장) |
| Transpose 범위 | 단일 32 B flit 안 | 고정 기능 순열 네트워크 |
| 시프트 버퍼 범위 | i4: [-15, 16], i8: [-7, 8], bf16: [-3, 4] | Stream Shift Unit 안의 제한된 레지스터 체인 |
| 매핑 정렬 | Stream Adapter 출력이 TRF Sequencer 축약 매핑과 일치해야 함 | Packet Reducer 에 버퍼링이나 재정렬 기능이 없음 |


#### 설계 근거

- **96 바이트 Flit Buffer**: 하위의 Packet Reducer 를 위한 단일 사이클 접근에는 일반 SRAM 이 아니라 레지스터 파일 저장이 필요하다.
  레지스터 파일은 비트당 면적을 훨씬 많이 차지하므로, 큰 버퍼는 감당하기 어려울 만큼 비싸진다.
  96 바이트(flit 3 개)는 쓸모 있는 컨볼루션 패턴과 실리콘 비용을 절충한 값이다.
- **단일 flit Transpose**: transpose 를 단일 flit 으로 제한하면 흔한 경우(축약 축을 가장 안쪽에 두는 것)에 연산이 빠르게 유지된다.
  여러 flit 에 걸쳐 확장하려면 훨씬 큰 순열 네트워크나 다중 사이클 버퍼링이 필요하다.
- **시프트 버퍼 한계**: Stream Shift Unit 은 레지스터 체인으로 데이터를 물리적으로 밀어서 슬라이딩 윈도우를 구현한다.
  현재 한계(`i8` 은 15, `bf16` 은 7)는 하드웨어 비용을 합리적으로 유지하면서 흔한 컨볼루션 필터 크기(3×3, 5×5, 7×7)를 지원한다.


- **TRF Sequencer 정렬**: Outer 단계의 원소 단위 곱셈과 Packet Reducer 의 트리는 정확히 정렬된 입력 스트림을 기대하는 고정 기능 곱셈-누산 배열을 이룬다.
  이 배열은 버퍼링하거나 재정렬하거나 어긋난 매핑에 맞춰줄 수 없으므로, 정렬이 어긋난 매핑은 완만한 성능 저하가 아니라 잘못된 계산을 낳는다.

### 성능

- **Transpose 지연**: flit 안 transpose 는 1-2 사이클의 지연을 더한다.
  축약 축이 가장 안쪽이 아닐 때 이 오버헤드는 피할 수 없다.
- **시프트 설정**: 시프트 설정이 잘못되면 성능만 떨어지는 것이 아니라 결과가 틀린다.
  그 밖의 고려 사항은 다음과 같다.
  - 초기 시프트 설정 오버헤드(셋업 사이클).
  - 스트라이드와 `pop_dim` 매개변수는 버퍼 관리에 영향을 준다.
- **데이터 재사용의 이점**: 시프트 연산을 제대로 설정하면 여러 출력 위치에 걸쳐 입력 데이터를 재사용하여 컨볼루션의 메모리 대역폭을 크게 줄인다.
  시프트가 없으면 출력마다 별도의 입력 페치가 필요하다.
