# Transpose Engine

Transpose Engine 은 `Time` 과 `Packet` 차원을 맞바꾸고, `Chip`, `Cluster`, `Slice` 차원은 그대로 둔다.

<a id="interface"></a>
## 인터페이스

`CollectTensor` 와 `VectorFinalTensor` 는 둘 다 `.transpose()` 를 제공한다.
`VectorFinalTensor` 진입점은 Vector Engine 출력에서 곧바로 Transpose Engine 으로 데이터를 넣는다.

```rust,ignore
// `D: MaterializableScalar` (see its doc) excludes i5/i9 stagings from transpose.
impl<
    'l,
    const T: Tu,
    P: CanApplyTranspose,
    D: MaterializableScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Performs the transpose operation.
    #[primitive(TuTensor::transpose)]
    pub fn transpose<OutTime: M, OutPacket: M>(
        self,
    ) -> TransposeTensor<'l, T, D, Chip, Cluster, Slice, OutTime, OutPacket, B> {
        verify_transpose::<D, Time, Packet, OutTime, OutPacket>();
        TransposeTensor::new(self.ctx, self.inner.transpose(false))
    }
}
```

커널 작성자가 `OutTime` 과 `OutPacket`(출력 차원 레이아웃)을 고르고, 컴파일러는 그 결과를 [매개변수](#parameters) 에 나열된 하드웨어 제약과 대조해 검증한다.

아래 예제는 8×16 `i8` 행렬을 전치하며, 이 행렬의 16 폭 행은 각각 두 개의 입력 packet(`D = 2`)에서 모은 것이다.
이 예제는 이 페이지의 나머지 부분에서 계속 기준 예제로 쓰인다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![P = 256, B = 2, C = 8, D = 2, E = 8];

fn basic_transpose<'l, const T: Tu>(
    input: CollectTensor<'l, T, i8, m![1], m![1 # 2], m![P], m![B, C, D], m![E # 32]>,
) -> TransposeTensor<'l, T, i8, m![1], m![1 # 2], m![P], m![B, D, E], m![C # 32]> {
    input.transpose()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, i8, m![1], m![1 # 2], m![P], m![B, C, D], m![E # 32]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = basic_transpose(c);
```

## 구조

아래 다섯 개의 전치 단계는 [인터페이스](#interface) 의 기준 예제로 설명한다.

<a id="parameters"></a>
### 매개변수

`valid_size` 는 Transpose Engine 이 32바이트 입력 버스에서 한 사이클에 읽는 유효 원소 개수이며, 모든 입력 flit 은 `bit-width × valid_size` 형태로 도착하고, 32바이트 flit 의 남는 바이트는 패딩으로 취급되어 Unpack 단계에서 버려진다.
데이터는 `CollectTensor::transpose()`(`Fetch → [Switch →] Collect → [Cast →] Transpose` 이후) 또는 `VectorFinalTensor::transpose()`([Vector Engine](./vector-engine/index.md) 에서 직접)를 거쳐 Transpose Engine 에 도달한다.
[Contraction Engine](./contraction-engine/index.md) 은 `32b × 8` 만 내보내고, [Vector Engine](./vector-engine/index.md) 과 [Fetch Engine](../moving-tensors/fetch-engine.md) 은 아래 표의 어떤 조합이든 내보낸다.

`in_cols`, `in_rows`, `out_rows` 는 커널 작성자의 `OutTime` 과 `OutPacket` 선택으로 정해진다.
넷 모두 원소 크기의 제약을 받는다:

| 원소 크기 | `valid_size` | 최대 `in_rows` | 유효 `in_cols` |
|--------------|--------------|---------------|-----------------|
| 4비트        | 16           | 16            | 16, 32          |
| 8비트        | 8            | 8             | 8, 16, 32       |
| 16비트       | 8            | 4             | 8, 16, 32       |
| 32비트       | 8            | 2             | 8, 16, 32       |

기준 예제(`i8` 이므로 `valid_size = 8`)에서 컴파일러는 다음을 유도한다:

| 매개변수   | 값 | 비고      |
|-------------|-------|------------|
| `in_cols`   | 16    | `D = 2` 개 packet 을 모음 × `valid_size = 8` |
| `in_rows`   | 8     | `C::SIZE`  |
| `out_rows`  | 16    | `D·E` (= `in_cols`, 완전히 활용) |

```text
                 in_cols                  in_rows # F
           ┌─────────────────┐         ┌──────────────────┐
           │ 12 13 14 15 ... │         │ 3  7  11 15  ... │
 in_rows   │ 8  9  10 11 ... │  ────►  │ 2  6  10 14  ... │  out_rows
           │ 4  5  6  7  ... │         │ 1  5  9  13  ... │
           │ 0  1  2  3  ... │         │ 0  4  8  12  ... │
           └─────────────────┘         └──────────────────┘
                data_in                      data_out
```

### Unpack

32바이트 입력 packet 은 각각 `valid_size` 개의 유효 원소를 실어 나르며, Unpack 단계는 flit 의 나머지를 패딩으로 버린다.

기준 예제에서: `[C, D, E # 32]` → `[C, D, E]`.

### Gather

입력 행렬의 한 행은 `in_cols = packets_per_col × valid_size` 개 원소 폭이며, 연속된 `packets_per_col` 개 packet — 가장 안쪽 시간 단계 — 으로 조립된다.
Gather 단계는 그 packet 들을 하나의 행으로 이어 붙이고, 그 위로 `in_rows` 개의 시간 단계가 쌓여 `[in_rows × in_cols]` 입력 행렬이 된다.
(Unpack 과 Gather 는 둘 다 엔진이 입력을 읽는 동안 일어난다. 별도의 버퍼링된 패스가 아니다.)

기준 예제에서는 가장 안쪽 `D = 2` 개 packet 이 각각 `valid_size = 8` 을 기여하여 `in_cols = 16` 폭 행을 이루고, 그 위의 `C = 8` 개 시간 단계가 쌓여 `[8 × 16]` 입력 행렬이 된다.

### Transpose

행렬이 전치된다: `[in_rows × in_cols]` → `[in_cols × in_rows]`.

기준 예제에서: `[C, D, E]` → `[D, E, C]`.

### Trim

일부 입력 packet 이 `valid_size` 보다 적은 수의 유효 원소를 실으면, 전치된 행렬에는 패딩된 행이 생긴다.
Trim 단계는 그 행들을 버려 `[out_rows × in_rows]` 를 만들며, 이때 `out_rows ≤ in_cols` 다.

기준 예제에서: `[D, E, C]` → `[D, E, C]` (입력이 완전히 활용되므로 절단되는 행이 없다).
Trim 이 실제로 행을 버리는 경우는 [작은 행렬](#small-matrix) 예제를 보라.

### Align

전치된 행은 `in_rows` 개 원소 폭이지만, DM packet 은 32바이트여야 한다.
Align 단계는 각 행을 32바이트 flit 으로 패딩하여 `[out_rows × (in_rows # F)]` 모양을 만들며, `F` 는 `D[F]` 가 32바이트가 되도록 고른다.

기준 예제에서: `[D, E, C]` → `[D, E, C # 32]`.

### 지연

> [!NOTE]
> 공식은 [성능](#performance) 을 먼저 읽어라.

기준 예제에서는 `in_cols = 16 ≤ 16` 이므로 더블 버퍼링이 선택된다.
`in_flits = 16`, `out_rows = 16`, `n = 2` 이면 전체 지연은 `16 + 1 × max(16, 16) + 16 = 48` 사이클이다.

## 예제

<a id="small-matrix"></a>
### 작은 행렬

이 예제는 `out_rows < in_cols` 일 때 Trim 단계가 패딩된 행을 버리는 모습을 보인다:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![P = 256, A = 4, B = 2];

fn small_transpose<'l, const T: Tu>(
    input: CollectTensor<'l, T, i8, m![1], m![1 # 2], m![P], m![A], m![B # 32]>,
) -> TransposeTensor<'l, T, i8, m![1], m![1 # 2], m![P], m![B], m![A # 32]> {
    input.transpose()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, i8, m![1], m![1 # 2], m![P], m![A], m![B # 32]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = small_transpose(c);
```

매개변수:

| 매개변수   | 값 | 비고                       |
|-------------|-------|-----------------------------|
| `in_cols`   | 8     | `B::SIZE = 2`, 8 로 패딩  |
| `in_rows`   | 4     | `A::SIZE`                   |
| `out_rows`  | 2     | `B::SIZE`                   |

단계:
- **Unpack**: `[A, B # 32]` → `[A, B # 8]`.
- **Gather**: `[A, B # 8]` → `[A, B # 8]` (`packets_per_col = 1` 이므로 각 packet 이 이미 완전한 `in_cols = 8` 행이다).
- **Transpose**: `[A, B # 8]` → `[B # 8, A]`.
- **Trim**: `[B # 8, A]` → `[B, A]` (패딩된 행 6 개가 절단된다).
- **Align**: `[B, A]` → `[B, A # 32]`.

지연: `in_cols = 8 ≤ 16` 이므로 더블 버퍼링이 선택된다.
`in_flits = 4`, `out_rows = 2`, `n = 1` 이면 전체는 `4 + 0 × max(4, 2) + 2 = 6` 사이클이다.

### 큰 열

이 예제는 `in_cols > 16` 으로 만들어 싱글 버퍼링을 강제한다. 그러면 입력과 출력이 겹치지 못해 전체 사이클이 늘어난다:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![P = 256, B = 2, C = 8, D = 4, E = 8];

fn large_col_transpose<'l, const T: Tu>(
    input: CollectTensor<'l, T, i8, m![1], m![1 # 2], m![P], m![B, C, D], m![E # 32]>,
) -> TransposeTensor<'l, T, i8, m![1], m![1 # 2], m![P], m![B, D, E], m![C # 32]> {
    input.transpose()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, i8, m![1], m![1 # 2], m![P], m![B, C, D], m![E # 32]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = large_col_transpose(c);
```

매개변수:

| 매개변수   | 값 | 비고                |
|-------------|-------|----------------------|
| `in_cols`   | 32    | `D::SIZE × E::SIZE`  |
| `in_rows`   | 8     | `C::SIZE`            |
| `out_rows`  | 32    | `D::SIZE × E::SIZE`  |

단계:
- **Unpack**: `[C, D, E # 32]` → `[C, D, E]`.
- **Gather**: 가장 안쪽 `D = 4` 개 packet 이 각 `in_cols = 32` 행을 이루고, `C = 8` 개 시간 단계가 쌓여 `[8 × 32]` 입력 행렬이 된다.
- **Transpose**: `[C, D, E]` → `[D, E, C]`.
- **Trim**: `[D, E, C]` → `[D, E, C]` (절단되는 행 없음).
- **Align**: `[D, E, C]` → `[D, E, C # 32]`.

지연: `in_cols = 32 > 16` 이므로 싱글 버퍼링이 선택된다.
`in_flits = 32`, `out_rows = 32`, `n = 2` (B) 이면 전체는 `2 × (32 + 32) = 128` 사이클이다.

### 16비트 데이터 타입

이 예제는 `bf16` 을 쓴다. 원소가 넓어져 최대 `in_rows` 가 절반(8 이 아니라 4)이 되고, 32바이트 출력 flit 은 16 개 원소로 줄어든다(`i8` 이라면 32 개):

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![P = 256, C = 8, D = 4, E = 8];

fn bf16_transpose<'l, const T: Tu>(
    input: CollectTensor<'l, T, bf16, m![1], m![1 # 2], m![P], m![C, D], m![E # 16]>,
) -> TransposeTensor<'l, T, bf16, m![1], m![1 # 2], m![P], m![C, E], m![D # 16]> {
    input.transpose()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, bf16, m![1], m![1 # 2], m![P], m![C, D], m![E # 16]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = bf16_transpose(c);
```

매개변수:

| 매개변수   | 값 | 비고      |
|-------------|-------|------------|
| `in_cols`   | 8     | `E::SIZE`  |
| `in_rows`   | 4     | `D::SIZE`  |
| `out_rows`  | 8     | `E::SIZE`  |

단계:
- **Unpack**: `[D, E # 16]` → `[D, E]`.
- **Gather**: `[D, E]` → `[D, E]` (`packets_per_col = 1` 이며, 각 packet 이 이미 완전한 `in_cols = 8` 행이다).
- **Transpose**: `[D, E]` → `[E, D]`.
- **Trim**: `[E, D]` → `[E, D]` (절단되는 행 없음).
- **Align**: `[E, D]` → `[E, D # 16]`.

지연: `in_cols = 8 ≤ 16` 이므로 더블 버퍼링이 선택된다.
`in_flits = 4`, `out_rows = 8`, `n = 8` (C) 이면 전체는 `4 + 7 × max(4, 8) + 8 = 68` 사이클이다.

### 4비트 데이터 타입

이 예제는 `i4` 를 쓴다. `valid_size = 16` 이 사이클당 원소 수를 두 배로 만들고 최대 `in_rows` 는 16 으로 올라가며(16 × 4 비트 = 8 바이트), 32바이트 flit 은 64 개 원소로 커진다:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![P = 256, B = 4, C = 16, E = 16];

fn i4_transpose<'l, const T: Tu>(
    input: CollectTensor<'l, T, i4, m![1], m![1 # 2], m![P], m![B, C], m![E # 64]>,
) -> TransposeTensor<'l, T, i4, m![1], m![1 # 2], m![P], m![B, E], m![C # 64]> {
    input.transpose()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, i4, m![1], m![1 # 2], m![P], m![B, C], m![E # 64]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = i4_transpose(c);
```

매개변수:

| 매개변수   | 값 | 비고      |
|-------------|-------|------------|
| `in_cols`   | 16    | `E::SIZE`  |
| `in_rows`   | 16    | `C::SIZE`  |
| `out_rows`  | 16    | `E::SIZE`  |

단계:
- **Unpack**: `[C, E # 64]` → `[C, E]`.
- **Gather**: `[C, E]` → `[C, E]` (`packets_per_col = 1` 이며, 각 packet 이 이미 완전한 `in_cols = 16` 행이다).
- **Transpose**: `[C, E]` → `[E, C]`.
- **Trim**: `[E, C]` → `[E, C]` (절단되는 행 없음).
- **Align**: `[E, C]` → `[E, C # 64]`.

지연: `in_cols = 16 ≤ 16` 이므로 더블 버퍼링이 선택된다.
`in_flits = 16`, `out_rows = 16`, `n = 4` (B) 이면 전체는 `16 + 3 × max(16, 16) + 16 = 80` 사이클이다.

### 32비트 데이터 타입

이 예제는 `f32` 를 쓴다. [Contraction Engine](./contraction-engine/index.md) 이 내보내는 두 가지 32비트 포맷(`32b × 8`, `f32` 또는 `i32`) 중 하나다.
원소가 넓어져 최대 `in_rows` 는 2 로 떨어지고(2 × 4 바이트 = 8 바이트), 32바이트 flit 은 8 개 원소로 줄어든다:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![P = 256, B = 4, D = 2, E = 8];

fn f32_transpose<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![1 # 2], m![P], m![B, D], m![E # 8]>,
) -> TransposeTensor<'l, T, f32, m![1], m![1 # 2], m![P], m![B, E], m![D # 8]> {
    input.transpose()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, f32, m![1], m![1 # 2], m![P], m![B, D], m![E # 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = f32_transpose(c);
```

매개변수:

| 매개변수   | 값 | 비고                       |
|-------------|-------|-----------------------------|
| `in_cols`   | 8     | `E::SIZE`                   |
| `in_rows`   | 2     | `D::SIZE`                   |
| `out_rows`  | 8     | `E::SIZE`                   |

단계:
- **Unpack**: `[D, E # 8]` → `[D, E]`.
- **Gather**: `[D, E]` → `[D, E]` (`packets_per_col = 1` 이며, 각 packet 이 이미 완전한 `in_cols = 8` 행이다).
- **Transpose**: `[D, E]` → `[E, D]`.
- **Trim**: `[E, D]` → `[E, D]` (절단되는 행 없음).
- **Align**: `[E, D]` → `[E, D # 8]`.

지연: `in_cols = 8 ≤ 16` 이므로 더블 버퍼링이 선택된다.
`in_flits = 2`, `out_rows = 8`, `n = 4` (B) 이면 전체는 `2 + 3 × max(2, 8) + 8 = 34` 사이클이다.

<a id="performance"></a>
## 성능

한 버스트는 `n = OutTime::SIZE / out_rows` 번의 전치 반복을 수행한다.
각 반복은 `in_flits = in_rows × (in_cols / valid_size)` 개의 입력 flit 과 `out_rows` 개의 출력 flit 을 옮긴다.

### 버퍼링 모드

Transpose Engine 에는 내부 버퍼가 두 개 있고, 각각 16 개 열을 담는다.
컴파일러는 `in_cols ≤ 16` 이면 더블 버퍼링을, 그렇지 않으면 싱글 버퍼링을 고른다. 더블 버퍼링은 입력과 출력을 겹쳐 전체 사이클을 줄이고, 싱글 버퍼링은 두 국면을 직렬화한다.

### 싱글 버퍼링 지연

전체 버스트 지연은 `n × (in_flits + out_rows)` 다.
두 버퍼를 함께 쓰므로 반복마다 입력과 출력이 더해진다.

### 더블 버퍼링 지연

전체 버스트 지연은 `in_flits + (n - 1) × max(in_flits, out_rows) + out_rows` 이며, 세 국면으로 나뉜다:

- **입력 전용 국면** (`in_flits` 사이클): 첫 번째 버퍼가 채워진다.
- **겹침 국면** (`(n - 1) × max(in_flits, out_rows)` 사이클): 한 버퍼가 입력을 받는 동안 다른 버퍼가 동시에 출력을 만들어내므로, 느린 쪽이 각 반복의 속도를 정한다.
- **출력 전용 국면** (`out_rows` 사이클): 마지막 버퍼가 비워진다.
