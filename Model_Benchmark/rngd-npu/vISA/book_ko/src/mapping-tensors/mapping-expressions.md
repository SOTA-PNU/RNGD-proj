# 매핑 표현식

매핑 표현식은 매핑을 인코딩하는 Rust 타입이다.
이 페이지는 그 생성자와 동치 규칙을 정의한다.


## 축 크기

`axes!` 매크로는 축 식별자와 그 크기를 선언한다.
다음 선언은 이 절 전체에 적용된다:

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, B = 512];
```

## 매핑 인터페이스

`m![H, W]` 같은 매핑 표현식은 각 텐서 인덱스를 버퍼 위치에 할당하는 Rust 타입이다.
모든 매핑 표현식은 `M` 트레잇을 구현하며, 이 트레잇은 버퍼 크기와 버퍼 위치에서 텐서 인덱스로 가는 함수를 제공한다:

```rust
// Inside `furiosa_opt_std::prelude`...
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use std::fmt::Debug;
pub trait M: Debug + Clone {
    /// The computed size for the given shape.
    const SIZE: usize;

    /// Converts the mapping expression type into a value.
    fn to_value() -> Mapping;

    /// Converts a buffer index to a tensor index, returning `None` if out-of-bounds.
    fn map(i: usize) -> Option<Index>;
}

/// Tensor index: a map from axis identifiers to coordinate values.
pub struct Index { /* ... */ }

/// Constructs tensor indices.
/// `i![A: 2, B: 3]` creates an `Index` with A = 2 and B = 3.
macro_rules! i {
    # () => {};
    /* ... */
}
```

### 사용 예제: Host Tensor

`M` 트레잇 위에 세워진 가장 단순한 구체 타입은 `HostTensor<D, E>` 다: 원소 타입이 `D` 이고 레이아웃이 매핑 `E` 로 완전히 결정되는 호스트 메모리 버퍼다.
`E` 는 버퍼 크기(`E::SIZE`)와 버퍼 위치에서 텐서 인덱스로의 대응(`E::map`)을 모두 결정한다.
`HostTensor<bf16, m![A, B]>` 는 `bf16` 데이터 원소 4,096 개를 담는다.
`HostTensor<D, E>` 가 텐서 \\(T\\) 를 *담는다*는 것은 다음과 같은 경우다:
- 모든 버퍼 인덱스 `i` 와 텐서 인덱스 `ti` 중 `E::map(i) = Some(ti)` 인 것에 대해,
- `i` 번째 원소가 인덱스 `ti` 에서의 텐서 \\(T\\) 값을 저장한다.

`HbmTensor` 와 `DmTensor` 같은 디바이스 텐서는 여러 매핑 표현식에 걸친 더 복잡한 레이아웃을 가진다; 자세한 내용은 [공간 차원과 시간 차원](./spatial-temporal-dimensions.md) 을 보라.

## 생성자

매핑 표현식은, 레이아웃 `E`(`HostTensor<D, E>` 의 레이아웃)를 비롯해, 작은 생성자들을 합성해서 만들며, 각 생성자는 더 단순한 매핑을 변환하거나 결합한다.
이 표현식들은 산술과 비슷한 연산자(`/`, `%`, 그리고 패딩을 위한 `#`)를 써서 텐서 인덱스와 선형 버퍼 인덱스 사이의 매핑을 간결하게 정의한다.

### Symbol

심볼은 크기가 모양 선언에서 오는 대문자 한 글자다.
매핑 `m![A]` 는 8 개의 버퍼 인덱스를 축을 따라 텐서 인덱스로 선형으로 매핑한다:

```rust
# extern crate furiosa_opt_std;
# extern crate furiosa_mapping;
# use furiosa_opt_std::prelude::*;
#
# axes![A = 8];
#
type E = m![A]; // Symbol<Ident::A, 8>

fn test_symbol() {
    assert_eq!(E::map(0), Some(i![A: 0]));
    assert_eq!(E::map(1), Some(i![A: 1]));
    assert_eq!(E::map(2), Some(i![A: 2]));
    for i in 0..E::SIZE {
        assert_eq!(E::map(i), Some(i![A: i]));
    }
    assert_eq!(E::map(E::SIZE), None);
}
#
# test_symbol();
```

```rust,ignore
impl<S: AxisName> M for Symbol<S> {
    const SIZE: usize = S::SIZE;

    fn to_value() -> Mapping {
        Mapping::Symbol {
            symbol: S::NAME,
            size: S::SIZE,
        }
    }

    fn map(i: usize) -> Option<Index> {
        (i < S::SIZE).then(|| {
            let mut index = Index::new();
            index.add_term(
                Term {
                    inner: Atom::Symbol {
                        symbol: S::NAME,
                        size: S::SIZE,
                    },
                    stride: 1,
                    modulo: S::SIZE,
                },
                i,
            );
            index
        })
    }
}
```

> [!NOTE]
> 모든 심볼 `A` 에 대해, 0 번째 인덱스 `i![A: 0]` 은 빈 텐서 인덱스 `i![]` 와 동치다.

### Pair

페어 매핑 `m![A, B]` 는 모양이 \\(\\{A=8, B=512\\}\\) 인 2D 텐서를 4,096 개 원소의 버퍼로 저장한다.
매핑 `Pair<L, R>` 은 두 공간의 데카르트 곱을 선형 버퍼로 매핑하며, 이때 `L` 이 메이저 차원이고 `R` 이 마이너 차원이다.
크기는 `L::SIZE * R::SIZE` 이고, 매핑은 인덱스를 분해하는 데 내림 나눗셈과 모듈로를 쓴다.
`m![A, B, C, D]` 는 `Pair<A, Pair<B, Pair<C, D>>>` 로 확장되며 우결합이다.

```rust
# extern crate furiosa_opt_std;
# extern crate furiosa_mapping;
# use furiosa_opt_std::prelude::*;
#
# axes![A = 8, B = 512];
#
type E = m![A, B]; // Pair<m![A], m![B]>

fn test_pair() {
    // First 512 elements hold A=0, next 512 hold A=1
    assert_eq!(E::map(0),   Some(i![A: 0, B: 0]));
    assert_eq!(E::map(511), Some(i![A: 0, B: 511]));
    assert_eq!(E::map(512), Some(i![A: 1, B: 0]));
    assert_eq!(E::map(519), Some(i![A: 1, B: 7])); // 519 == 512 * 1 + 7
    for i in 0..E::SIZE {
        assert_eq!(E::map(i), Some(i![A: i / <m![B]>::SIZE, B: i % <m![B]>::SIZE]));
    }
    assert_eq!(E::map(E::SIZE), None);
}
#
# test_pair();
```

```rust,ignore
impl<L, R> M for Pair<L, R>
where
    L: M,
    R: M,
{
    const SIZE: usize = L::SIZE * R::SIZE;

    fn to_value() -> Mapping {
        Mapping::Pair {
            left: RBox::new(L::to_value()),
            right: RBox::new(R::to_value()),
        }
    }

    fn map(i: usize) -> Option<Index> {
        let mut l = L::map(i / R::SIZE)?;
        let r = R::map(i % R::SIZE)?;
        l.add(r);
        Some(l)
    }
}
```

### Identity

항등 매핑 `m![1]` 은 버퍼 인덱스 `0` 을 빈 텐서 인덱스 `i![]` 로 매핑하는 단일 원소 버퍼를 만든다.
이는 `Pair` 의 항등원 역할을 한다: `m![1, A]` 와 `m![A, 1]` 은 둘 다 `m![A]` 와 동치다.

```rust
# extern crate furiosa_opt_std;
# extern crate furiosa_mapping;
# use furiosa_opt_std::prelude::*;
#
type E = m![1]; // Identity

fn test_identity() {
    assert_eq!(E::map(0), Some(i![]));
    assert_eq!(E::map(1), None);
}
#
# test_identity();
```

```rust,ignore
/// The identity mapping (size-1 broadcast), the unit written `m![1]`.
pub type Identity = Broadcast<1>;
```

### Padding

패딩은 쓰이지 않는 버퍼 공간을 덧붙여 데이터를 하드웨어 요구에 맞춰 정렬한다.
예를 들어 DMA 엔진은 행이 64 바이트 경계에서 시작할 것을 요구한다.
`axes![C = 13, D = 61]` 에서 `m![C, D]` 는 `61` 이 `64` 로 나누어떨어지지 않으므로 정렬되지 않은 행을 만든다.
`m![C, D # 64]` 는 행마다 여분의 원소 3 개를 써서 각 행을 64 바이트 경계에 정렬함으로써 이를 해결한다.

```rust
# extern crate furiosa_opt_std;
# extern crate furiosa_mapping;
# use furiosa_opt_std::prelude::*;
#
axes![C = 13, D = 61];

type E = m![C, D # 64]; // Pair<m![C], Padding<m![D], 64>>

fn test_padding() {
    assert_eq!(E::map(0),  Some(i![C: 0, D: 0]));
    assert_eq!(E::map(60), Some(i![C: 0, D: 60]));
    assert_eq!(E::map(61), None); // padding
    assert_eq!(E::map(62), None); // padding
    assert_eq!(E::map(63), None); // padding
    assert_eq!(E::map(64), Some(i![C: 1, D: 0]));
}
#
# test_padding();
```

```rust,ignore
impl<L, const SIZE: usize, const KIND: PaddingKind> M for Padding<L, SIZE, KIND>
where
    L: M,
{
    const SIZE: usize = SIZE;

    fn to_value() -> Mapping {
        Mapping::Padding {
            inner: RBox::new(L::to_value()),
            padding: SIZE,
            kind: KIND,
        }
    }

    fn map(i: usize) -> Option<Index> {
        L::map(i)
    }
}
```

패딩된 슬롯은 개수뿐 아니라 그 내용도 타입의 일부다.
세 가지 종류가 추적된다.

- `m![A # m]` (또는 `m![A #{*} m]`) 은 크기 `m` 까지의 top 패딩이다.
  슬롯은 접근 가능하지만 임의의 값을 담는다.
  raw DM 텐서가 이것을 가진다. `#` 는 축약형이고, `#{*}` 는 그 종류를 명시적으로 적은 것이다.
- `m![A #{0} m]` 은 크기 `m` 까지의 0 으로 채운 패딩이다.
  슬롯은 접근 가능하며 0 을 담는 것으로 알려져 있다.
  Fetch Adapter 의 [마스킹](../computing-tensors/fetch-adapter.md#masking) 단계가 `# m` 으로부터 이것을 만든다.
- `m![A #{!} m]` 은 크기 `m` 까지의 bottom 패딩이다.
  슬롯은 접근 불가능하며 읽기/쓰기는 정의되지 않은 동작이다.
  이는 컴파일러가 피해야 하는 주소를 모델링한다.

`#` 는 기본적으로 top 종류다.
Rust 타입 수준에서는 `PaddingKind` 의 const 제네릭을 `Padding<L, SIZE, KIND>` 에 붙여 이를 그대로 반영한다.
`Padding<L, N>` 은 `KIND = PaddingKind::Top` 이고, `Padding<L, N, { PaddingKind::Zero }>` 는 0 으로 채우는 변형이며, `Padding<L, N, { PaddingKind::Bottom }>` 은 접근 불가능하다.

### Resize

리사이즈는 새 크기를 넘어서는 인덱스를 잘라내어 매핑을 더 작은 논리 크기로 제한하며, 그 범위 밖의 원소는 버린다.
버퍼를 늘리는 패딩과 달리 리사이즈는 논리적 뷰를 줄인다.
매핑 `m![D = 2]` 는 축 `D` 의 앞 2 개 원소만 취해 인덱스 `D = 0` 과 `D = 1` 을 만든다.

```rust
# extern crate furiosa_opt_std;
# extern crate furiosa_mapping;
# use furiosa_opt_std::prelude::*;
#
axes![C = 2, D = 3];
type E = m![C, D = 2]; // Pair<m![C], Resize<m![D], 2>>

fn test_resize() {
    assert_eq!(E::map(0), Some(i![C: 0, D: 0]));
    assert_eq!(E::map(1), Some(i![C: 0, D: 1]));
    assert_eq!(E::map(2), Some(i![C: 1, D: 0]));
    assert_eq!(E::map(3), Some(i![C: 1, D: 1]));
    assert_eq!(E::map(4), None);
}
#
# test_resize();
```

```rust,ignore
impl<L, const SIZE: usize> M for Resize<L, SIZE>
where
    L: M,
{
    const SIZE: usize = SIZE;

    fn to_value() -> Mapping {
        Mapping::Resize {
            inner: RBox::new(L::to_value()),
            resize: SIZE,
        }
    }

    fn map(i: usize) -> Option<Index> {
        if i < SIZE { L::map(i) } else { None }
    }
}
```

### 타일링

타일링은 데이터 복사 없이 순수하게 메타데이터만 변환하는 *인덱스 뷰*로 구현된다.
`.tile()` 메서드는 한 차원을 타일 크기로 리사이즈하고 버퍼 안으로 오프셋을 잡아 타일을 추출한다.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
#
# axes![A = 8, B = 512];
#
let tensor = unsafe { HbmTensor::<bf16, m![1], m![A, B]>::from_addr(0) };
let view = tensor.view(); // HbmTensorView::<'_, bf16, m![1], m![A, B]>
let tile01 = view.tile::<m![B], 2, m![A, B = 2 # 512]>(0); // HbmTensorView::<'_, bf16, m![1], m![A, B = 2 # 512]>
let tile23 = view.tile::<m![B], 2, m![A, B = 2 # 512]>(2); // HbmTensorView::<'_, bf16, m![1], m![A, B = 2 # 512]>
```

`.tile()` 메서드는 세 개의 타입 매개변수와 한 개의 값 매개변수를 받는다.
- *타일 차원* `m![B]` 는 어느 차원을 따라 나눌지 지정한다.
- *타일 크기* `2` 는 타일당 원소 개수를 지정한다.
- *타일 매핑* `m![A, B = 2 # 512]` 는 결과 뷰의 매핑을 정의한다.
  매핑 `B = 2 # 512` 는 차원 `B` 가 뷰 안에서는 논리 크기 `2` 를 가지지만 `512` 의 물리적 공간 안에 존재함을 뜻한다.
  `# 512` 가 없으면 타일 사이의 스트라이드가 512 가 아니라 2 가 되어, 뷰가 잘못된 버퍼 위치에서 읽게 된다.
- *시작 인덱스* 는 어느 타일을 추출할지 지정한다.
  `0` 을 넘기면 범위 `0..2` 가 `tile01` 로 잡히고, `2` 를 넘기면 범위 `2..4` 가 `tile23` 으로 잡힌다.

<a id="stride-and-modulo"></a>
### 스트라이드와 모듈로

스트라이드(`/`)와 모듈로(`%`)는 하나의 차원을 둘로 분해한다: 바깥(블록 인덱스)과 안쪽(블록 안의 위치).
512 개 원소를 가진 축 `B` 를 각각 64 개 원소인 8 개 블록으로 나눈다고 하자.
매핑 `m![B / 64, B % 64]` 는 8 × 64 격자를 만들며, 첫 번째 차원은 어느 블록인지를 고르고 두 번째 차원은 그 블록 안의 위치를 고른다:

```rust
# extern crate furiosa_opt_std;
# extern crate furiosa_mapping;
# use furiosa_opt_std::prelude::*;
# axes![A = 8, B = 512];
type D1 = m![B / 64]; // stride with size 8
type D2 = m![B % 64]; // modulo with size 64

type E = m![B / 64, B % 64]; // equivalent to `m![B]`

fn test_stride_modulo() {
    assert_eq!(E::map(130), Some(i![B / 64: 2, B % 64: 2])); // block 2, position 2: B = 64*2 + 2 = 130
    assert_eq!(E::map(130), <m![B]>::map(130));               // same result as flat m![B]

    for i in 0..8 {
        assert_eq!(D1::map(i), Some(i![B / 64: i]));
    }
    assert_eq!(D1::map(8), None);

    for j in 0..64 {
        assert_eq!(D2::map(j), Some(i![B % 64: j]));
    }
    assert_eq!(D2::map(64), None);

    for i in 0..8 {
        for j in 0..64 {
            assert_eq!(
                E::map(64 * i + j),
                <m![B]>::map(64 * i + j),
            );
        }
    }
    assert_eq!(E::map(512), None);
}
#
# test_stride_modulo();
```

```rust,ignore
impl<L, const SIZE: usize> M for Stride<L, SIZE>
where
    L: M,
{
    const SIZE: usize = {
        assert!(L::SIZE % SIZE == 0, "Stride size must divide the original size");
        L::SIZE / SIZE
    };

    fn to_value() -> Mapping {
        Mapping::Stride {
            inner: RBox::new(L::to_value()),
            stride: SIZE,
        }
    }

    fn map(i: usize) -> Option<Index> {
        if i < Self::SIZE { L::map(i * SIZE) } else { None }
    }
}

impl<L, const SIZE: usize> M for Modulo<L, SIZE>
where
    L: M,
{
    const SIZE: usize = {
        assert!(L::SIZE % SIZE == 0, "Modulo size must divide the original size");
        SIZE
    };

    fn to_value() -> Mapping {
        Mapping::Modulo {
            inner: RBox::new(L::to_value()),
            modulo: SIZE,
        }
    }

    fn map(i: usize) -> Option<Index> {
        if i < Self::SIZE { L::map(i % L::SIZE) } else { None }
    }
}
```

스트라이드와 모듈로 매핑은 표 형태로 시각화할 수 있다.
매핑 `m![B / 4, B % 4]` 를 `B::SIZE = 16` 인 경우로 생각해 보자.
다음 표는 버퍼 인덱스가 어떻게 배열되는지 보인다: 각 행은 `B / 4`(스트라이드 축)의 특정 인덱스에 대응하고, 각 열은 `B % 4`(모듈로 축)의 인덱스에 대응한다:

|                 | `i![B % 4: 0]` | `i![B % 4: 1]` | `i![B % 4: 2]` | `i![B % 4: 3]` |
| --------------- | -------------- | -------------- | -------------- | -------------- |
| `i![B / 4: 0]` | `i![B: 0]`     | `i![B: 1]`     | `i![B: 2]`     | `i![B: 3]`     |
| `i![B / 4: 1]` | `i![B: 4]`     | `i![B: 5]`     | `i![B: 6]`     | `i![B: 7]`     |
| `i![B / 4: 2]` | `i![B: 8]`     | `i![B: 9]`     | `i![B: 10]`    | `i![B: 11]`    |
| `i![B / 4: 3]` | `i![B: 12]`    | `i![B: 13]`    | `i![B: 14]`    | `i![B: 15]`    |

모듈로는 버퍼 크기를 다루는 방식에서 리사이즈와 다르다:
- 리사이즈는 새 크기를 넘어서는 인덱스를 잘라내어 버퍼를 줄인다.
- 모듈로는 원래 버퍼 크기를 유지하면서 그것을 같은 크기의 블록들로 나눈다.

이 연산들은 복잡한 분해를 위해 중첩할 수 있다.
다음 예제는 `B` 를 세 개의 차원으로 쪼개며, 이때 버퍼의 비트 레이아웃은 텐서 인덱스의 비트 레이아웃과 다르다.

```rust
# extern crate furiosa_opt_std;
# extern crate furiosa_mapping;
# use furiosa_opt_std::prelude::*;
# axes![A = 8, B = 512];
// B's bits: 6 - 8,  0 - 4,          5
// Values:   0 - 7, 0 - 31,      0 - 1
type E = m![B / 64, B % 32, B / 32 % 2];

fn test_nested_stride() {
    assert_eq!(E::map(67), Some(i![B: 97])); // 67 = 64*1 + 2*1 + 1 (i=1,j=1,k=1) → B = 64*1 + 1 + 32*1 = 97
    // Verify B=97 round-trips: 97/64=1, 97%32=1, (97/32)%2=1
    assert_eq!(97 / 64, 1);
    assert_eq!(97 % 32, 1);
    assert_eq!((97 / 32) % 2, 1);

    // buffer index: 64 * i + 2 * j + k (i = block, j = position within block, k = sub-block)
    // tensor index B: 64 * i + j + 32 * k (rearranges bit positions)
    for i in 0..8 {
        for j in 0..32 {
            for k in 0..2 {
                assert_eq!(
                    E::map(64 * i + 2 * j + k),
                    Some(i![B: 64 * i + j + 32 * k]),
                );
            }
        }
    }
    assert_eq!(E::map(512), None);
}
#
# test_nested_stride();
```

이런 식의 비트 재배열은 뱅크 인터리빙이나 캐시 효율을 위해 주소 비트를 재정렬하는 하드웨어 메모리 레이아웃에 자연스럽게 대응한다.
2 진수로 보면 이는 비트 위치를 재배열한다: 버퍼 `001_00001_1` 이 `B = 001_1_00001` 이 된다.
버퍼는 비트를 `[8:6]_[5:1]_[0]` 으로 묶는 반면 `B` 는 `[8:6]_[5]_[4:0]` 으로 묶는다.

타일링은 개별 원소가 아니라 블록 단위로 동작할 수도 있다.
다음 예제는 `m![B / 32]` 를 써서 블록 단위로 타일링하며 겹치는 타일을 만든다:

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# axes![A = 8, B = 512];
let tensor = unsafe { HbmTensor::<bf16, m![1], m![A, B]>::from_addr(0) };
for i in 0..15 {
    let tile = tensor.view().tile::<m![B / 32], 2, m![A, B / 32 = 2 # 16, B % 32]>(i);
}
```

`B = 512` 일 때 차원 `B / 32` 는 0-15 로 번호가 매겨진 16 개의 블록을 가진다.
각 타일은 인덱스 `i` 에서 시작하는 연속된 2 개의 블록을 취한다.
타일 0 은 블록 `{0, 1}` 을 덮고, 타일 1 은 블록 `{1, 2}` 를 덮으며, 이런 식으로 타일 14 가 블록 `{14, 15}` 를 덮는 데까지 이어진다.
연속한 타일이 블록 하나를 공유하므로 이 타일들은 서로 겹친다.

각 타일이 정확히 2 개의 블록을 담으므로 타일 매핑 `B / 32 = 2` 는 블록 차원을 2 로 리사이즈한다.
블록 하나로 타일링할 때는 그 차원이 값을 하나만 가지므로 `B / 32 = 1` 이 항등 `m![1]` 로 단순해진다.

### Escape

복잡한 매핑에는 타입 별칭을 정의하고 `{ ... }` 로 참조한다.
별도의 매핑 `L = m![A]` 와 `R = m![B]` 가 있을 때, 이를 `m![{ L }, { R }]` 로 결합하면 `m![A, B]` 와 같은 결과를 낸다:

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# axes![A = 8, B = 512];
type L = m![A];
type R = m![B];
type E = m![{ L }, { R }]; // equivalent to `m![A, B]`

fn test_escape() {
    for i in 0..E::SIZE {
        assert_eq!(E::map(i), <m![A, B]>::map(i));
    }
}
#
# test_escape();
```

이 이스케이프 문법은 복잡한 매핑을 이름 붙은 재사용 가능한 구성 요소로 분해한다.


### 고급 생성자


#### 스큐 축

스큐 축은 두 차원에 걸친 대각선 접근 패턴을 만든다.
스큐 축은 기존 축들 사이의 산술적 차이로 정의되는 파생 축 이름을 도입한다; 예를 들어 `B' = B - A` 는 새 축 `B'` 를 정의하며, 이 축의 좌표는 어느 지점에서든 `B` 에서 `A` 를 뺀 값이다.
특정 wavefront 계산처럼 대각선을 따라 데이터를 처리하는 알고리즘이 이 패턴을 쓴다.

표현식 `m![A, B' = 4]` 는 `B' = B - A` 일 때 각 행이 이전 행에 대해 이동된 매핑을 만든다.
`=` 연산자는 스큐 이후의 논리 크기를 지정한다.
결과는 모듈러 산술로 순환한다.

예를 들어 `axes![A = 4, B = 4]` 와 `B' = B - A` 인 경우는 다음과 같다:

| (A, B') | (A, B) |
|---------|--------|
| (0, 0)  | (0, 0) |
| (0, 1)  | (0, 1) |
| (0, 2)  | (0, 2) |
| (0, 3)  | (0, 3) |
| (1, 0)  | (1, 1) |
| (1, 1)  | (1, 2) |
| (1, 2)  | (1, 3) |
| (1, 3)  | (1, 0) |

`A = 1` 이고 `B' = 3` 일 때, 원래의 `B` 좌표는 모듈러 산술을 통해 `0` 으로 순환하는데, `B = (B' + A) % 4 = (3 + 1) % 4 = 0` 이기 때문이다.

#### 간접 시퀀싱

#### 슬라이딩 (선형 결합)

> [!NOTE]
> 선형 결합 표현식 `$(e1:n1, ..., ed:nd)` 는 여러 차원을 지정된 스트라이드로 결합한다.
> 형식적 정의: `size_S($(e1:n1, ..., ed:nd)) = 1 + sum_k((size_S(ek) - 1) * nk)`.
> 매핑 `S, $(e1:n1, ..., ed:nd) |- si ~ ti` 는, `si1...sid, ti1...tid` 가 존재하여 모든 `k` 에 대해 `S, ek |- sik ~ tik`, `si = sum_k(sik * nk)`, `ti = sum_k(tik * nk)` 를 만족하면 성립한다.
>
> 선형 결합은 외적 합을 인코딩할 수 있다: `e1 * e2` 는 `$(e1 : size_S(e2), e2 : 1)` 와 동치다.
> 다만 외적 합이 축 재정렬에 더 강하기 때문에 더 선호된다.
> `e1 * e2` 를 `e2 * e1` 로 바꿔도 스트라이드를 수동으로 고칠 필요가 없다.

슬라이딩 연산은 겹치는 데이터 블록에 접근하며, 합성곱 신경망에 필수적이다.
각 행이 한 번에 한 원소씩 미끄러지는 3 원소 슬라이스인, 모양 \\(\\{N=5, F=3\\}\\) 의 텐서를 나타내는 9 개 원소 버퍼를 생각해 보자.
\\((N, F)\\) 에 있는 텐서 원소는 버퍼 인덱스 \\(N + 2F\\) 로 매핑된다:

$$
\begin{array}{c|ccc}
  & F=0 & F=1 & F=2 \\\\
\hline
N=0 & 0 & 2 & 4 \\\\
N=1 & 1 & 3 & 5 \\\\
N=2 & 2 & 4 & 6 \\\\
N=3 & 3 & 5 & 7 \\\\
N=4 & 4 & 6 & 8 \\\\
\end{array}
$$

> [!NOTE]
> 이 슬라이딩 패턴에서는 하나의 공간 인덱스가 여러 텐서 인덱스로 매핑될 수 있다.
> 예를 들어 공간 인덱스 `4` 는 `{4_N}`, `{2_N, 1_F}`, `{2_F}` 로 동시에 매핑된다.
> 이는 `(S, e).maps(si, ti)` 가 일대일이 아니라는 성질을 보여 준다.

이는 `N` 축이 스트라이드 `1` 을 갖고 `F` 축이 스트라이드 `2` 를 갖는 선형 결합 표현식으로 나타낼 수 있으며, 전체 크기는 `1 + (5-1)*1 + (3-1)*2 = 9` 가 된다.



## 동치 매핑

서로 다른 생성자 조합이 같은 매핑을 만들어 낼 수 있다.
구체적으로, 매핑 `E1` 과 `E2` 는 다음을 만족할 때 *동치*다:
- `E1::SIZE == E2::SIZE` 이고,
- 모든 `i` 에 대해 `E1::map(i) == E2::map(i)`.

이 동치 관계는 반사적이고 대칭적이며 추이적이다.
다음 항등식들이 흔한 동치 관계를 담고 있다:

- **페어의 항등원**: 모든 `E` 에 대해 `E` 는 `m![{ E }, 1]` 과 `m![1, { E }]` 둘 다와 동치다.
- **스트라이드-모듈로 분해**: 모든 `E` 에 대해, 그 크기 `E::SIZE` 가 `n` 으로 나누어떨어지면 `E` 와 `m![{ E } / n, { E } % n]` 은 동치다.
- **페어 사영**: 모든 `A` 와 `B` 에 대해 `m![[{ A }, { B }] / B::SIZE]` 는 `m![A]` 와 동치이고 `m![[{ A }, { B }] % B::SIZE]` 는 `m![B]` 와 동치다.
- **페어의 결합법칙**: 모든 `E1`, `E2`, `E3` 에 대해 `m![{ E1 }, { E2 }, { E3 }]`, `m![[{ E1 }, { E2 }], { E3 }]`, `m![{ E1 }, [{ E2 }, { E3 }]]` 은 동치다.
- **멱등 연산**: 모든 `E` 에 대해 `E` 는 `m![{ E } / 1]`, `m![{ E } # E::SIZE]`, `m![{ E } = E::SIZE]` 와 동치다.
- **1 로 나눈 모듈로**: 모든 `E` 에 대해 `m![E % 1]` 은 항등 매핑 `m![1]` 과 동치다.
