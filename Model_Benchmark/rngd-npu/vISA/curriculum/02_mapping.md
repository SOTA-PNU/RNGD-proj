# 02 · 매핑과 텐서 (가장 중요)

이 문서는 vISA 커리큘럼 모듈 02입니다. vISA에서 제일 어렵고 제일 중요한 개념입니다. `axes![]`와 `m![]` 매핑 언어(`/ % # = ,`)로 논리 축을 하드웨어 계층에 어떻게 펼치는지, 공간/시간 차원이 무엇인지 익힙니다.
*선행: 01 멘탈 모델 · 예상 시간: 반나절*

## 학습 목표

- [ ] `m![A / 8 # 256]` 같은 매핑을 보고 '어느 슬라이스에 몇 개씩'인지 읽는다
- [ ] `/`(Stride)·`%`(Modulo)·`#`(Padding)·`=`(Resize)·`,`(Pair)의 의미를 구분한다
- [ ] 공간 차원(Chip/Cluster/Slice/Lane)과 시간 차원(Time/Packet)을 구분한다
- [ ] `--backend typecheck`로 합법·불법 매핑을 직접 확인한다

## 1. 개념

## 0. 왜 "매핑"부터 배워야 하나

vISA에서 가장 먼저, 그리고 가장 정확하게 잡고 가야 하는 개념이 매핑(mapping)입니다. 데이터 이동·연산·스케줄 같은 나머지가 전부 이 위에 얹히기 때문입니다.

출발점은 한 문장입니다. "텐서는 원소들의 고유한 순서를 갖지 않는다"(docs/src/mapping-tensors/index.md:3-4). NumPy 배열을 떠올리면 보통 행 우선(row-major) 메모리 모습까지 같이 떠올리지만, vISA는 그 둘을 일부러 분리합니다. 텐서는 "축 이름 → 값"이라는 수학적 대응일 뿐이고(quick-start.md:13-18), 그 값들을 실제 버퍼의 어느 칸에 놓을지를 정하는 별도의 규칙이 바로 매핑입니다. 즉 매핑은 "텐서 인덱스 ↔ 버퍼 위치"의 사전입니다.

이렇게 분리하는 이유는 성능입니다. 하드웨어는 메모리를 연속 블록 단위로 읽기 때문에, 멀리 떨어져 저장된 원소들을 모으려면 전송이 더 많이 필요합니다(index.md:10). 그래서 어떤 축을 major(가장 바깥, 가장 느리게 변함), 어떤 축을 minor(가장 안쪽, 가장 빠르게 변하며 연속 저장됨)로 둘지가 곧 성능을 좌우합니다. 그리고 한 번 할당하고 나면 레이아웃은 복사·전치 없이는 못 바꾸므로(index.md:12), 할당 시점의 선택이 이후 모든 연산을 제약합니다. 이 무거운 선택을 사람이 stride/offset 숫자로 직접 만지는 대신, vISA는 "논리 축"으로 선언만 하면 컴파일러가 물리 배치·정렬·스케줄을 유도하도록 합니다(index.md:82-95). 이게 "선언적(declarative) 접근"입니다.

## 1. H×W 예제로 보는 레이아웃 3종

6행×8열 텐서(H=6, W=8)를 두고 세 가지 매핑을 비교합니다(index.md:14-91).

- H-major: `m![H, W]` — W를 따라 스캔하면 연속, H를 따라 스캔하면 캐시라인당 한 원소씩만 읽음.
- W-major: `m![W, H]` — 반대.
- 2×2 타일: `m![H / 2, W / 2, H % 2, W % 2]` — 앞 두 차원이 타일 인덱스, 뒤 두 차원이 타일 내부 위치. 두 축 모두 적당한 지역성을 얻는 대신 주소 계산식이 복잡해짐.

핵심 직관 하나. 매핑식에서 왼쪽이 major, 오른쪽이 minor입니다(index.md:88). 그리고 `H / 2`, `W % 2` 같은 표현이 한 축을 "바깥 블록 인덱스"와 "블록 내부 위치"로 쪼개는 도구입니다. 이 분해는 단순히 메모리 모양만 정하는 게 아니라 하드웨어 실행 구조까지 결정합니다. 바깥 차원은 하드웨어 시간 루프(`Time`)가 되고, 안쪽 차원은 병렬 레인 폭(`Packet`)이 됩니다(index.md:76-79). 이 두 이름은 책 전체에서 쓰이는 하드웨어 차원 이름입니다.

## 2. axes! — 축 이름과 크기 선언

매핑을 쓰려면 먼저 축을 선언합니다.

```rust
axes![A = 8, B = 512];
```

이 매크로는 각 `이름 = 크기` 쌍을 유닛 구조체 하나와 `AxisName` 구현으로 펼칩니다(furiosa-mapping-macro/src/lib.rs:54-112). 펼쳐진 코드는 `pub struct A;`와 `impl AxisName for A { const NAME = Ident::new("A"); const SIZE = 8; }` 형태입니다(lib.rs:98-110, AxisName 정의는 furiosa-mapping/src/mapping.rs:12-17). 즉 축은 "타입"이고, 크기는 그 타입의 연관 상수 `SIZE`입니다. 그래서 나중에 `<m![A]>::SIZE`처럼 컴파일타임에 크기를 읽을 수 있습니다. 같은 invocation 안에서 이름을 중복 선언하면(`axes![A=8, A=16]`) 컴파일 에러를 냅니다(lib.rs:75-96).

## 3. M 트레잇 — 모든 매핑이 지켜야 할 계약

`m![...]`이 만들어 내는 건 전부 Rust 타입이고, 그 타입들은 `M` 트레잇을 구현합니다(mapping.rs:22-31).

```rust
pub trait M: Debug + Clone {
    const SIZE: usize;                  // 버퍼 크기(원소 개수)
    fn to_value() -> Mapping;           // 런타임 값 표현으로 변환
    fn map(i: usize) -> Option<Index>;  // 버퍼 위치 i -> 텐서 인덱스
}
```

여기서 가장 중요한 두 가지가 `SIZE`와 `map`입니다. `SIZE`는 이 매핑이 차지하는 버퍼 칸 수, `map(i)`는 "i번째 버퍼 칸이 텐서의 어느 인덱스를 담는가"를 돌려줍니다. 범위를 벗어나면 `None`입니다. 이 둘이 매핑의 의미 전부입니다. 그래서 두 매핑이 "동등(equivalent)"하다는 건 정확히 `SIZE`가 같고 모든 `i`에 대해 `map(i)`가 같다는 뜻입니다(mapping-expressions.md:442-458).

텐서 인덱스는 `Index` 타입으로 표현되고, `i![A: 2, B: 3]`처럼 `i!` 매크로로 만듭니다(mapping-expressions.md:30-39, i! 구현은 lib.rs:207-231). 한 가지 알아둘 점: 모든 심볼 `A`에 대해 0번 인덱스 `i![A: 0]`은 빈 인덱스 `i![]`과 같습니다(mapping-expressions.md:86-87).

가장 단순한 구체 타입은 `HostTensor<D, E>`입니다. 원소 타입 `D`, 레이아웃 `E`로, 버퍼 크기와 위치→인덱스 대응이 `E` 하나로 완전히 결정됩니다. 예: `HostTensor<bf16, m![A, B]>`는 bf16 원소 4,096개(8×512)를 담습니다(mapping-expressions.md:41-50).

## 4. m![] 미니언어 — 연산자별 정확한 의미

`m![]`은 작은 생성자들을 합성해 매핑을 만듭니다. 각 연산자가 `M`을 구현하는 별도 타입으로 펼쳐집니다(펼침 규칙: furiosa-mapping-macro/src/parser/mod.rs:73-122). 하나씩, 실제 구현과 함께 봅니다.

### 4-1. Symbol(축 하나)

`m![A]`는 `Symbol<A>`이고 `SIZE = A::SIZE`, `map(i) = i![A: i]`입니다(mapping.rs:61-89). 즉 버퍼 위치 i가 그대로 축 좌표 i가 됩니다.

```rust
type E = m![A]; // axes![A = 8]
E::map(0) == Some(i![A: 0]);
E::map(7) == Some(i![A: 7]);
E::map(8) == None; // SIZE=8 이라 범위 밖
```

### 4-2. `,` Pair(곱공간을 선형 버퍼로)

`m![A, B]`는 `Pair<m![A], m![B]>`입니다. 두 공간의 데카르트 곱을 선형 버퍼로 펴되, 왼쪽 L이 major, 오른쪽 R이 minor입니다. 크기는 `L::SIZE * R::SIZE`이고, 핵심은 map 구현입니다(mapping.rs:210-230).

```rust
fn map(i) {
    let l = L::map(i / R::SIZE)?;  // 몫 = 바깥(major) 좌표
    let r = R::map(i % R::SIZE)?;  // 나머지 = 안쪽(minor) 좌표
    Index::add(&mut l, r); Some(l)
}
```

floor 나눗셈과 나머지로 인덱스를 분해한다는 점을 꼭 기억하세요. `m![A, B]`(A=8, B=512)에서 처음 512칸은 A=0, 다음 512칸은 A=1입니다. `map(519) = i![A:1, B:7]` (519 = 512*1 + 7)(mapping-expressions.md:104-115). 그리고 `m![A, B, C, D]`는 오른쪽 결합으로 `Pair<A, Pair<B, Pair<C, D>>>`로 펼쳐집니다(mapping-expressions.md:94).

### 4-3. Identity(`m![1]`)

`m![1]`은 단일 원소 버퍼로, 0번 위치를 빈 인덱스 `i![]`에 대응시킵니다(mapping.rs:40-50). Pair의 항등원이라 `m![1, A]`와 `m![A, 1]` 모두 `m![A]`과 동등합니다(mapping-expressions.md:122-142). 파서에서 정수 리터럴 중 `1`만 Identity로 허용되고, 그 외 숫자(`m![64]`)는 컴파일 에러입니다(parser.lalrpop:59-65). 크기가 필요하면 `axes!`로 이름을 붙이라는 뜻입니다.

### 4-4. `#` Padding(버퍼를 늘려 정렬)

`m![C, D # 64]`는 `Pair<m![C], Padding<m![D], 64>>`입니다. Padding은 버퍼 크기를 SIZE로 키우지만, `map`은 원래 inner 그대로라 늘어난 칸은 어떤 텐서 인덱스에도 대응하지 않습니다(mapping.rs:183-200). 그래서 `map`이 그 영역에서 `None`을 돌려줍니다.

```rust
axes![C = 13, D = 61];
type E = m![C, D # 64]; // 각 행을 61->64로 패딩
E::map(60) == Some(i![C:0, D:60]);
E::map(61) == None; // 패딩
E::map(63) == None; // 패딩
E::map(64) == Some(i![C:1, D:0]); // 다음 행 시작
```

왜 필요하냐면, 예컨대 DMA 엔진이 각 행을 64바이트 경계에서 시작하라고 요구하는데 61은 64의 배수가 아니라서 그렇습니다(mapping-expressions.md:144-149). 패딩 칸은 "쓰레기 값을 담아도 되는 여분 슬롯"으로 이해하면 됩니다(quick-start.md:88).

### 4-5. `=` Resize(논리 크기를 줄임)

`m![D = 2]`는 `Resize<m![D], 2>`로, 앞 2개만 남기고 그 뒤 인덱스를 잘라 버립니다(mapping.rs:157-173). 패딩이 버퍼를 늘리는 것과 반대로, Resize는 논리 뷰를 줄입니다.

```rust
axes![C = 2, D = 3];
type E = m![C, D = 2]; // 각 C에서 D는 0,1 만
E::map(2) == Some(i![C:1, D:0]);
E::map(4) == None;
```

### 4-6. `/` Stride 와 `%` Modulo(한 차원을 둘로 쪼갬)

이 둘이 함께 한 축을 "바깥 블록 인덱스"와 "블록 내부 위치"로 분해합니다. 512짜리 축 B를 64씩 8블록으로:

```rust
type E = m![B / 64, B % 64]; // m![B] 와 동등
```

- `B / 64` = `Stride<m![B], 64>`: 64개마다 하나씩 골라 크기 8(=512/64). `map(i) = inner.map(i * 64)`(mapping.rs:99-118).
- `B % 64` = `Modulo<m![B], 64>`: 크기 64, `map(i) = inner.map(i % inner.SIZE)`(mapping.rs:128-147).

표로 보면(B=16, `m![B/4, B%4]`) 행이 `B/4`(블록), 열이 `B%4`(블록 내 위치)이고 칸 값이 원래 B입니다(mapping-expressions.md:277-286). `E::map(130) = i![B/64:2, B%64:2]` (130 = 64*2 + 2)(mapping-expressions.md:246).

Modulo와 Resize의 차이가 헷갈리기 쉬운데(mapping-expressions.md:288-290): Resize는 버퍼 자체를 잘라 줄이고, Modulo는 원래 버퍼 크기를 유지하면서 같은 크기 블록으로 나눌 뿐입니다.

이걸 중첩하면 비트 재배열도 됩니다. `m![B / 64, B % 32, B / 32 % 2]`는 버퍼 비트 묶음과 텐서 인덱스 비트 묶음이 달라지는 예로, 뱅크 인터리빙·캐시 효율을 위한 주소 비트 재배치에 그대로 대응합니다(mapping-expressions.md:292-329).

### 4-7. Escape(`{ ... }`)와 대괄호

복잡한 매핑은 타입 별칭으로 빼고 `{ ... }`로 참조합니다. `m![{ L }, { R }]`(L=m![A], R=m![B])는 `m![A, B]`과 동등합니다(mapping-expressions.md:352-373). 대괄호 `[ ... ]`는 그룹핑(결합 순서 지정)에 쓰입니다. 파서에서 `(...)`와 `[...]`는 둘 다 묶음 역할입니다(parser.lalrpop:67-68).

### 4-8. .tile() — 데이터 복사 없는 부분 뷰

`.tile()`은 한 차원을 타일 크기로 resize하고 버퍼 안으로 오프셋만 주는 순수 메타데이터 변환입니다(mapping-expressions.md:201-227). 타입 파라미터 3개와 값 1개를 받습니다: 타일 차원(`m![B]`), 타일 크기(`2`), 결과 뷰 매핑(`m![A, B = 2 # 4]`), 그리고 시작 인덱스. 여기서 `B = 2 # 4`가 중요합니다. 뷰 안에서 B의 논리 크기는 2지만 물리 보폭은 4라는 뜻으로, `# 4`가 없으면 타일 간 stride가 2가 되어 잘못된 위치를 읽습니다. 실제 예제 tile.rs에서 `input.tile::<m![B], 1, m![A, 1 # 32]>(b)`로 한 열씩 떼어 전치 복사합니다(examples/.../tile.rs:11-13). 블록 단위 타일이나 겹치는 타일(`m![B / 32], 2`로 연속 2블록씩)도 만들 수 있습니다(mapping-expressions.md:331-350).

### 4-9. 고급: Skewed 축과 Sliding(선형결합)

- Skewed: `B' = B - A` 같은 파생 축으로 대각 접근 패턴을 만듭니다. `m![A, B' = 4]`는 행마다 한 칸씩 밀리며 모듈러로 감깁니다. A=1, B'=3이면 `B = (B'+A)%4 = 0`(mapping-expressions.md:379-402). 웨이브프론트 계산에 씁니다.
- Sliding(선형결합 `$(e1:n1, ...)`): 겹치는 블록 접근으로 합성곱(CNN)에 필수입니다. 9칸 버퍼를 shape {N=5, F=3}로 보며 각 행이 3칸 슬라이스를 한 칸씩 미는 패턴은 `(N,F) -> N + 2F`로, 크기는 `1 + (5-1)*1 + (3-1)*2 = 9`. 이때 한 버퍼 위치가 여러 텐서 인덱스에 동시에 대응할 수 있어 1:1이 아닙니다(mapping-expressions.md:404-438).

## 5. 연산자 우선순위 — 파서가 알려주는 진실

문법을 보면 한눈에 정리됩니다(parser.lalrpop:39-69).

- 가장 느슨한 건 `,`(Pair). `Mapping0` 수준에서 오른쪽 결합입니다(`A, B, C` = `A,(B,(C))`). 그래서 `m![A,B,C,D]`가 오른쪽 결합 Pair가 됩니다.
- 더 강하게 묶는 건 `/`(Stride), `%`(Modulo), `=`(Resize), `#`(Padding). 모두 `Mapping1` 수준에서 같은 우선순위, 왼쪽 결합입니다. 따라서 `A / 4 % 2`는 `(A / 4) % 2`로, 왼쪽에서 오른쪽으로 차례대로 적용됩니다.
- 가장 강한 건 심볼/`1`/escape/괄호(`Mapping2`).

즉 `m![A, B / 64]`는 `Pair<m![A], Stride<m![B],64>>`이지 `Stride<Pair<...>,64>`가 아닙니다. `,`가 항상 가장 바깥입니다. 이걸 알면 복잡한 디바이스 텐서 타입을 읽을 때 헷갈리지 않습니다.

## 6. 동등 매핑 규칙(정규화)

매핑은 정규형으로 정규화되어 기호적으로 검증되므로(index.md:95), 다음 항등식들이 컴파일타임 불변식이 됩니다(mapping-expressions.md:442-458):

- 항등원: 임의 E에 대해 `E ≡ m![{E}, 1] ≡ m![1, {E}]`.
- stride-modulo 분해: SIZE가 n으로 나눠지면 `E ≡ m![{E}/n, {E}%n]`.
- 사영: `m![[{A},{B}] / B::SIZE] ≡ m![A]`, `m![[{A},{B}] % B::SIZE] ≡ m![B]`.
- Pair 결합법칙: `m![{E1},{E2},{E3}] ≡ m![[{E1},{E2}],{E3}] ≡ m![{E1},[{E2},{E3}]]`.
- 무연산: `E ≡ m![{E}/1] ≡ m![{E} # E::SIZE] ≡ m![{E} = E::SIZE]`.
- `m![E % 1] ≡ m![1]`.

## 7. 컴파일타임 나눗셈 제약 — 가장 자주 만나는 에러

`/`와 `%`는 아무 숫자나 못 씁니다. Stride와 Modulo의 `SIZE`는 const 평가 중 `assert!(L::SIZE % SIZE == 0)`을 합니다(mapping.rs:104, mapping.rs:133). 메시지는 각각 "Stride size must divide the original size", "Modulo size must divide the original size"입니다. 예를 들어 B=512에서 `m![B / 7]`은 512 % 7 != 0 이라 컴파일이 실패합니다. 이건 런타임 검사가 아니라 타입 수준 불변식이라, 잘못된 레이아웃은 애초에 빌드되지 않습니다. NPU 없이 `--backend typecheck`만으로도 잡힙니다.

## 8. 논리 축이 하드웨어 계층에 분배되는 방식

`HostTensor`는 매핑 하나로 끝이지만, 디바이스 텐서는 레이아웃을 여러 전용 차원으로 쪼갭니다(spatial-temporal-dimensions.md:1-8). 하드웨어 계층은 Chip > Cluster > Slice > Lane이고(quick-start.md:43-53), 각 계층마다 텐서 타입에 별도의 `M` 타입 파라미터가 붙습니다.

```rust
struct HbmTensor<D, Chip, Element> { ... }
struct DmTensor<D, Chip, Cluster, Slice, Element> { ... }
struct TrfTensor<D, Chip, Cluster, Slice, Lane, Element> { ... }
struct VrfTensor<D, Chip, Cluster, Slice, Element> { ... }
```

(spatial-temporal-dimensions.md:22-41). 의미는 간단합니다. 같은 계층의 모든 유닛은 같은 매핑을 공유하고, 각 계층 파라미터가 "이 축들을 이 하드웨어 차원으로 나눠 담는다"를 뜻합니다. 예: `HbmTensor<bf16, m![A], m![B]>`(A=8, B=512)는 4096 원소를 8칩에 칩당 512개씩 나눠 담고, i번째 칩의 j번째 원소가 `i![A:i, B:j]`입니다(spatial-temporal-dimensions.md:43-46).

실전 분배 예(quick-start.md:82): `DmTensor<bf16, m![1], m![1 # 2], m![A / 8 # 256], m![A % 8]>` (A=2048)는 — 칩 1개(`m![1]`), 클러스터 2개 중 1개(`m![1 # 2]`), 256슬라이스에 분산(`m![A / 8 # 256]`, 2048/8=256), 슬라이스당 8원소(`m![A % 8]`)입니다. 즉 A의 각 원소가 정확히 한 슬라이스 안의 한 위치로 갑니다.

### 8-1. Chip/Cluster/Slice 크기 제약

이 세 차원 크기는 하드웨어 개수와 정확히 같아야 합니다(spatial-temporal-dimensions.md:54-69).

| 유닛 | 개수 | 제약 | 패딩 예 |
|---|---|---|---|
| Chip | 시스템마다 | `Chip::SIZE == NUM_CHIPS` | `m![1 # NUM_CHIPS]` |
| Cluster | 칩당 2 | `Cluster::SIZE == 2` | `m![1 # 2]` |
| Slice | 클러스터당 256 | `Slice::SIZE == 256` | `m![X / N # 256]` |

커널이 하드웨어보다 적은 유닛만 쓰면 `#`로 패딩합니다. `type Cluster = m![1 # 2]`는 활성 클러스터 1개 + 패딩 1개로 "칩당 2클러스터" 요구를 만족시킵니다(spatial-temporal-dimensions.md:64-66). 단 현재 런타임은 칩 단위(`#[device(chip = N)]`)로 동작해서 칩·클러스터의 부분 사용은 아직 지원되지 않습니다(spatial-temporal-dimensions.md:67-69).

### 8-2. Element 크기·정렬 제약

`Element::SIZE * size_of::<D>()`가 유닛별 SRAM 용량을 넘으면 안 됩니다(spatial-temporal-dimensions.md:71-77): DmTensor는 슬라이스당 512KB, VrfTensor는 슬라이스당 8KB, TrfTensor는 레인당 8KB이며 `Lane::SIZE <= 8`. 또 시작 주소는 `size_of::<D>()`의 배수여야 합니다. 정렬이 어긋나면 read-modify-write가 생겨 DM 접근이 약 50배 느려지기 때문입니다(spatial-temporal-dimensions.md:79).

## 9. 공간 vs 시간 차원: Time과 Packet

파이프라인을 흐르는 텐서는 `TuTensor`로, SRAM 타입의 Chip/Cluster/Slice에 더해 `Time`과 `Packet`을 갖습니다(spatial-temporal-dimensions.md:81-126).

- `Time`(시간 차원): 전달 반복(iteration)을 순서대로 셉니다. 하드웨어가 강제하는 크기 상한이 없고, 처리할 데이터 양에 따라 커집니다.
- `Packet`(추가 공간 차원): 한 번의 시간 반복마다 각 슬라이스가 받는 원소 수를 정합니다.

`TuTensor`는 `const P: Position`으로 파이프라인 어느 단계의 출력인지도 타입에 새깁니다(Position enum: Begin/Fetch/Switch/Collect/Contraction/Reduce/Cast/Transpose, spatial-temporal-dimensions.md:103-112). Vector와 Commit이 빠진 이유까지 주석에 적혀 있습니다(전자는 별도 typestate, 후자는 끝나면 DmTensor가 됨).

worked example(spatial-temporal-dimensions.md:128-143): shape {N=4, C=64, H=32, W=32}를 Fetch 출력으로 스트리밍할 때

```rust
Slice  = m![C / 2]   // 64채널을 32슬라이스에 분산
Time   = m![N, H, W] // 4*32*32 = 4096 번 반복
Packet = m![C % 2]   // 매 반복 슬라이스당 2채널
```

32슬라이스가 병렬이고 슬라이스당 2채널이니, 한 시간 반복마다 32*2 = 64채널이 동시에 처리됩니다. "C를 / 와 % 로 쪼개 바깥(/2)은 슬라이스 공간에, 안쪽(%2)은 패킷에 둔다"가 정확히 8-1에서 본 `A / 8 # 256` / `A % 8` 패턴과 같은 사고방식입니다.

## 10. 텐서 시맨틱: "담는다(hold)"와 "명세(specify)"

마지막으로 이 모든 매핑이 수학적으로 무엇을 보장하는지입니다(tensor-semantics.md).

"담는다(hold)": 텐서 변수가 수학적 텐서 T를 담는다는 건, 각 원소가 "그 자리에 관여하는 모든 매핑 차원들이 만든 부분 인덱스의 합" 위치의 T 값을 저장한다는 뜻입니다(tensor-semantics.md:7-23). HostTensor는 매핑 하나라 단순합니다: `E::map(i) = Some(ti)`이면 i번째 원소가 T의 ti 값. HbmTensor는 매핑을 둘(Chip, Element)로 쪼개고, 각자 서로 겹치지 않는 축을 담당해서, i번째 칩의 j번째 원소가 `Chip::map(i) + Element::map(j)` 위치의 T 값을 저장합니다. 다른 모든 텐서 타입도 같은 규칙을 더 많은 차원으로 확장할 뿐입니다(부분 인덱스들의 합).

"명세(specify)": 함수가 무엇을 하는지는, 입력이 담는 텐서로 출력이 담는 텐서를 기술하는 것입니다(tensor-semantics.md:24-44). 예로 `elementwise_add`는 "lhs가 T1, rhs가 T2를 담으면 반환값이 T1+T2를 담는다". 표현(레이아웃·메모리 계층)이 뭐든 출력이 올바른 수학 텐서를 담으면 그 함수는 옳습니다.

여기서 "수학적 텐서 이동(mathematical tensor move)"이라는 특수한 경우가 중요합니다(tensor-semantics.md:46-64). `f(T) = T`, 즉 표현만 바뀌고 수학적 값은 그대로인 연산입니다. `.to_dm()`이 대표적입니다: hbm이 T를 담으면 반환 DmTensor도 같은 T를 담습니다. 매핑은 칩/클러스터/슬라이스/엘리먼트로 다르게 펼쳐지지만 담는 텐서는 동일합니다. 이 관점 덕분에 데이터 이동과 연산을 같은 파이프라인에서 합성 가능한 것으로 추론할 수 있습니다(index.md:97-99). 이것이 vISA가 레이아웃을 타입으로 끌어올린 진짜 이유입니다 — 매핑이 달라도 "담는 값"이 같음을 컴파일타임에 보장합니다.

## 2. 핵심 API · 패턴

| 이름 | 쓰는 법 | 설명 | 출처 |
|---|---|---|---|
| `axes!` | `axes![A = 8, B = 512];` | 각 이름=크기를 유닛 구조체 + AxisName impl(NAME, SIZE 상수)로 펼침. 같은 호출 내 이름 중복은 컴파일 에러. | `furiosa-mapping-macro/src/lib.rs:54-112, furiosa-mapping/src/mapping.rs:12-17` |
| `M 트레잇` | `trait M { const SIZE: usize; fn to_value() -> Mapping; fn map(i: usize) -> Option<Index>; }` | 모든 m![] 타입이 구현. SIZE=버퍼 칸 수, map(i)=버퍼 위치->텐서 인덱스(범위 밖이면 None). 두 매핑 동등 = SIZE 같고 모든 i에 map(i) 같음. | `furiosa-mapping/src/mapping.rs:22-31` |
| `m![] (Pair, ',')` | `m![A, B] => Pair<m![A], m![B]>; map(i): l=L::map(i / R::SIZE), r=R::map(i % R::SIZE)` | 왼쪽 major, 오른쪽 minor. 우측 결합(m![A,B,C,D]=Pair<A,Pair<B,Pair<C,D>>>). floor 나눗셈/나머지로 분해. | `furiosa-mapping/src/mapping.rs:204-231, mapping-expressions.md:89-120` |
| `m![] (Stride '/')` | `m![B / 64] => Stride<m![B], 64>; SIZE = L::SIZE/64; map(i)=L::map(i*64)` | 한 축의 바깥 블록 인덱스. SIZE 계산 중 assert!(L::SIZE % 64 == 0) 컴파일타임 검사. | `furiosa-mapping/src/mapping.rs:99-118` |
| `m![] (Modulo '%')` | `m![B % 64] => Modulo<m![B], 64>; SIZE=64; map(i)=L::map(i % L::SIZE)` | 블록 내부 위치. 버퍼 크기 유지하며 분할. assert!(L::SIZE % 64 == 0) 컴파일타임 검사. m![E % 1] ≡ m![1]. | `furiosa-mapping/src/mapping.rs:121-148` |
| `m![] (Resize '=')` | `m![D = 2] => Resize<m![D], 2>; SIZE=2; map(i)= i<2 ? L::map(i) : None` | 논리 뷰를 잘라 줄임(버퍼 늘리는 Padding과 반대). 잘린 인덱스는 버려짐. | `furiosa-mapping/src/mapping.rs:151-174` |
| `m![] (Padding '#')` | `m![D # 64] => Padding<m![D], 64>; SIZE=64; map(i)=L::map(i) (늘어난 칸은 None)` | 버퍼를 SIZE로 키워 정렬/하드웨어 유닛 수 맞춤. 여분 슬롯은 임의 값 가능. m![{E} # E::SIZE] ≡ E. | `furiosa-mapping/src/mapping.rs:177-201` |
| `m![1] (Identity)` | `type E = m![1]; E::map(0)=Some(i![]); E::map(1)=None` | Pair 항등원. m![1,A] ≡ m![A,1] ≡ m![A]. 1 외의 bare 정수(m![64])는 컴파일 에러. | `furiosa-mapping/src/mapping.rs:34-51, parser.lalrpop:59-65` |
| `i!` | `i![A: 2, B: 3] -> Index` | 텐서 인덱스 생성. i![A: 0]은 빈 인덱스 i![]과 동등. | `mapping-expressions.md:30-39, furiosa-mapping-macro/src/lib.rs:207-231` |
| `.tile()` | `view.tile::<m![B], 2, m![A, B = 2 # 4]>(start)` | 복사 없는 부분 뷰(indexed view). 타입파라미터=타일차원,타일크기,결과매핑 + 값=시작인덱스. 결과매핑 '# 4'로 물리 보폭 보존 안 하면 잘못된 위치 읽음. | `mapping-expressions.md:201-227, examples/.../tile.rs:11-13` |
| `.to_dm()` | `hbm.to_dm(&mut ctx.tdma, addr) -> DmTensor<...>` | 수학적 텐서 이동(f(T)=T): 매핑은 Chip/Cluster/Slice/Element로 다르게 펼쳐지지만 담는 텐서는 동일. addr는 슬라이스 내 DM 오프셋. | `tensor-semantics.md:46-64` |

## 3. 실험 (직접 돌리기)

> 실험은 NPU 없이 `simulation`·`typecheck`로 돌아갑니다. 실행법은 [`../experiments/README.md`](../experiments/README.md), MNIST는 `cargo furiosa-opt test`(npu 전용).

### 실험 02.1 — Pair 매핑을 손으로 예측한 뒤 map()으로 검증
*난이도 1/5 · 기반: `new`*

**목표** — m![A,B]가 floor 나눗셈/나머지로 버퍼 위치를 텐서 인덱스로 바꾸는 규칙(왼쪽 major)을 몸으로 익힌다.

```bash
# base-template 파생 프로젝트의 src/gemv.rs 맨 아래에 아래 테스트 모듈을 추가:
#   #[cfg(test)] mod m_play {
#       use furiosa_opt_std::prelude::*;
#       axes![A = 8, B = 512];
#       #[test] fn pair() {
#           assert_eq!(<m![A, B]>::SIZE, 4096);
#           assert_eq!(<m![A, B]>::map(0),   Some(i![A:0, B:0]));
#           assert_eq!(<m![A, B]>::map(519), Some(i![A:1, B:7]));
#           assert_eq!(<m![A, B]>::map(4096), None);
#       }
#   }
cargo test --bin gemv pair -- --nocapture
```
**관찰** — 테스트 통과. map(519)이 i![A:1,B:7] (519=512*1+7)임을 확인. map(4096)은 None. 매핑은 순수 CPU 코드라 NPU 없이 plain cargo test로 실행됨.

**심화** — m![B, A](W-major)로 바꿔 map(7)이 무엇이 되는지 먼저 예측한 뒤 확인하라. major/minor가 뒤집힌다.

### 실험 02.2 — Stride/Modulo가 m![B]와 동등함을 확인
*난이도 2/5 · 기반: `new`*

**목표** — B / 64 와 B % 64 가 한 축을 바깥 블록/내부 위치로 쪼개고 합치면 원래와 같아짐을 본다.

```bash
# 위 m_play 모듈에 추가:
#   #[test] fn stride_modulo() {
#       type E = m![B / 64, B % 64];
#       assert_eq!(E::map(130), Some(i![B/64:2, B%64:2]));
#       for i in 0..512 { assert_eq!(E::map(i), <m![B]>::map(i)); }
#   }
cargo test --bin gemv stride_modulo
```
**관찰** — map(130)=i![B/64:2, B%64:2] (130=64*2+2)이고 모든 i에서 m![B]와 동일. 동등 매핑 정의(SIZE 같고 모든 map 같음)를 직접 검증한 셈.

**심화** — m![B/4, B%4](B=16로 axes 변경)로 mapping-expressions.md:281-286의 4x4 표를 손으로 재현해 보라.

### 실험 02.3 — 나눗셈 제약 컴파일 에러 만들기 (find-the-type-error)
*난이도 2/5 · 기반: `new`*

**목표** — / 와 % 의 컴파일타임 divisibility 불변식을 직접 깨뜨려 본다. NPU 불필요.

```bash
# m_play 모듈에 잘못된 매핑 하나 추가 (B=512):
#   type Bad = m![B / 7];  // 512 % 7 != 0
#   #[test] fn bad() { let _ = <Bad>::SIZE; }
cargo build --bin gemv   # 또는: cargo furiosa-opt run --release --bin gemv --backend typecheck
```
**관찰** — 빌드 실패. const 평가에서 'Stride size must divide the original size' assert가 터진다(mapping.rs:104). %로 바꿔 m![B % 7]로 하면 'Modulo size must ...'(mapping.rs:133). 잘못된 레이아웃은 애초에 컴파일되지 않음을 확인.

**심화** — m![B / 8] (512%8==0)로 고치면 통과. 어떤 나눗수가 허용되는지 512의 약수로 실험.

### 실험 02.4 — GEMV 커널을 시뮬레이션에서 실행: 논리 축의 슬라이스 분배 관찰
*난이도 3/5 · 기반: `base-template/src/gemv.rs`*

**목표** — m![1 # 256](256슬라이스), Time/Packet 분배가 실제 커널에서 어떻게 쓰이는지 본다.

```bash
cargo furiosa-opt run --release --bin gemv
cargo furiosa-opt test --release --bin gemv
```
**관찰** — 시뮬레이션 실행 성공 후 호스트 레퍼런스 대비 검증 통과. quick-start.md:221-245의 switch_gemv 타입에서 입력이 m![1 # 256] 슬라이스로 패딩 분배되고 J가 Time/Packet으로 타일링됨을 코드와 대조.

**심화** — gemm을 같은 방식으로 실행하고 type Slice = m![I / 32, J / 32](2D 슬라이스 매핑, quick-start.md:264)이 어떻게 출력 16x16 타일을 슬라이스에 배치하는지 추적.

### 실험 02.5 — Padding과 Resize의 차이를 map()으로 대조
*난이도 2/5 · 기반: `new`*

**목표** — # 는 버퍼를 늘려 여분 슬롯(None)을 만들고, = 는 논리 뷰를 잘라 줄인다는 정반대 동작을 직접 본다.

```bash
# axes![C = 13, D = 61]; 로 m_play 변경 후:
#   #[test] fn pad_vs_resize() {
#       type P = m![C, D # 64];
#       assert_eq!(P::map(61), None); assert_eq!(P::map(64), Some(i![C:1, D:0]));
#       axes![X = 2, Y = 3];
#       type R = m![X, Y = 2];
#       assert_eq!(R::SIZE, 4); assert_eq!(R::map(2), Some(i![X:1, Y:0]));
#   }
cargo test --bin gemv pad_vs_resize
```
**관찰** — Padding: 행이 61->64로 늘고 61~63은 None(여분), 64에서 다음 행 시작. Resize: Y가 3->2로 줄어 전체 SIZE가 4. 버퍼 확장 vs 논리 축소 차이 확인.

**심화** — tile.rs의 m![A, 1 # 32]에서 '# 32'를 빼고(m![A, 1]) 빌드/실행이 어떻게 달라지는지 예측 후 확인 — 타일 보폭이 틀어져 잘못된 위치를 읽게 됨(mapping-expressions.md:225).

## 4. 연습문제 (손으로, 컴파일 없이)

**Q1.** axes![A = 8, B = 512]에서 type E = m![A, B]일 때 E::map(1025)가 돌려주는 텐서 인덱스를 구하라. (힌트: 1025를 R::SIZE=512로 나눈 몫과 나머지)

<details><summary>정답/힌트</summary>

1025 = 512*2 + 1 이므로 Some(i![A: 2, B: 1]).

</details>

**Q2.** axes![B = 16]에서 type E = m![B / 4, B % 4]일 때 E::map(11)은? 그리고 같은 입력에 대해 <m![B]>::map(11)과 같은가?

<details><summary>정답/힌트</summary>

11 = 4*2 + 3 -> i![B/4: 2, B%4: 3], 이는 B=11과 동일하므로 m![B]::map(11)=i![B:11]과 같다. 동등 매핑.

</details>

**Q3.** 다음 중 컴파일 에러가 나는 것을 모두 고르고 이유를 적어라: (a) axes![B=512]; m![B / 8]  (b) m![B / 7]  (c) m![64]  (d) m![1]  (e) m![B % 3]

<details><summary>정답/힌트</summary>

(b) 512%7!=0 Stride assert, (c) bare 64는 Identity(1)만 허용이라 에러, (e) 512%3!=0 Modulo assert. (a)(d)는 정상.

</details>

**Q4.** axes![C = 13, D = 61]에서 m![C, D # 64]의 E::SIZE는 얼마이며, E::map(62)는 무엇인가? 또 같은 칸이 유효 데이터인가?

<details><summary>정답/힌트</summary>

SIZE = 13 * 64 = 832. map(62)=None(패딩 영역, 61~63은 어떤 인덱스에도 대응 안 함). 유효 데이터 아님(여분 슬롯).

</details>

**Q5.** shape {N=4, C=64, H=32, W=32}를 Slice=m![C / 2], Time=m![N, H, W], Packet=m![C % 2]로 스트리밍할 때 (1) 슬라이스 수 (2) 총 시간 반복 수 (3) 한 시간 반복에 처리되는 채널 수를 각각 구하라.

<details><summary>정답/힌트</summary>

(1) 64/2 = 32슬라이스 (2) 4*32*32 = 4096 (3) 32슬라이스 * 2채널 = 64채널.

</details>

**Q6.** m![A, B / 64, B % 64] 와 m![A, B]는 동등한가? '동등'의 정의를 들어 한 줄로 근거를 대라. (A=8, B=512)

<details><summary>정답/힌트</summary>

동등(SIZE 같고 모든 i에 map 같음). B/64,B%64는 B를 그대로 복원(stride-modulo 분해 항등식)하므로 두 SIZE 모두 4096, 모든 i에서 map 동일 -> 동등.

</details>

**Q7.** type Cluster = m![1 # 2]는 무엇을 의미하며 왜 이렇게 쓰는가? Cluster::SIZE는?

<details><summary>정답/힌트</summary>

활성 클러스터 1개 + 패딩 1개. Cluster::SIZE는 2(하드웨어 칩당 2클러스터 요건 충족). 1클러스터만 쓰지만 타입 제약을 패딩으로 만족.

</details>

## 5. 흔한 함정

- ',' (Pair)가 항상 가장 바깥(가장 느슨한) 연산자다. m![A, B / 64]는 Pair<m![A], Stride<m![B],64>>이지 Stride<Pair<...>,64>가 아니다. / % = # 는 ','보다 강하게, 서로 같은 우선순위로 왼쪽 결합한다(A / 4 % 2 = (A/4)%2).  
  ↳ 출처 `furiosa-mapping-macro/src/parser/parser.lalrpop:39-69`
- '/'와 '%'의 나눗수는 원래 축 크기를 정확히 나눠야 한다. 안 그러면 런타임이 아니라 컴파일 시점에 const assert가 터진다(예: B=512에서 m![B/7]). 메시지는 'Stride/Modulo size must divide the original size'.  
  ↳ 출처 `furiosa-mapping/src/mapping.rs:104, furiosa-mapping/src/mapping.rs:133`
- Resize(=)와 Modulo(%)를 혼동하기 쉽다. Resize는 버퍼 자체를 잘라 SIZE를 줄이고, Modulo는 원래 버퍼 크기를 유지하며 같은 크기 블록으로 partition한다. 둘은 SIZE 처리가 정반대다.  
  ↳ 출처 `docs/src/mapping-tensors/mapping-expressions.md:288-290`
- .tile()의 결과 매핑에서 '# 물리크기'를 빼면 타일 간 stride가 논리 크기로 줄어 잘못된 버퍼 위치를 읽는다. m![A, B = 2 # 4]에서 '# 4'가 빠지면 보폭이 2가 되어 깨진다.  
  ↳ 출처 `mapping-expressions.md:223-227`
- bare 정수 리터럴은 1만 허용된다(Identity). m![64] 같은 표현은 컴파일 에러이며, 크기가 필요하면 axes!로 이름 붙인 축을 써야 한다.  
  ↳ 출처 `furiosa-mapping-macro/src/parser/parser.lalrpop:59-65, furiosa-mapping-macro/src/lib.rs:128-136`
- Chip/Cluster/Slice 크기는 하드웨어 개수와 정확히 일치해야 한다(Cluster::SIZE==2, Slice::SIZE==256). 적게 쓰면 '#'로 패딩해 채워야 하고, 현재 런타임은 칩 단위라 칩/클러스터 부분 사용은 아직 안 된다.  
  ↳ 출처 `docs/src/mapping-tensors/spatial-temporal-dimensions.md:56-69`
- Padding으로 늘어난 칸은 map()이 None을 돌려주는 '여분 슬롯'이라 임의 값을 담을 수 있다. 이 칸을 유효 데이터로 착각하면 안 된다.  
  ↳ 출처 `docs/src/mapping-tensors/mapping-expressions.md:159-167, quick-start.md:88`
- DM 텐서 시작 주소가 size_of::<D>() 배수로 정렬되지 않으면 read-modify-write로 약 50배 느려진다. main/sub 컨텍스트가 같은 평면 SRAM을 공유하므로 주소 충돌도 직접 피해야 한다.  
  ↳ 출처 `spatial-temporal-dimensions.md:79, quick-start.md:100-102`

## 6. 핵심 정리 & 다음

기억할 사실:
- RNGD 하드웨어 계층은 Chip > Cluster(칩당 2) > Slice(클러스터당 256) > Lane(슬라이스당 8, Contraction Engine MAC 배열의 한 행). 따라서 디바이스 텐서 타입의 Cluster::SIZE는 정확히 2, Slice::SIZE는 정확히 256이어야 한다. (`quick-start.md:43-53, docs/src/mapping-tensors/spatial-temporal-dimensions.md:56-62`)
- 메모리 계층 용량: HBM 48GB/1.5TB/s, DM(온칩 SRAM) 총 256MB이며 슬라이스당 512KB, TRF 레인당 8KB(레인 8개/슬라이스), VRF 슬라이스당 8KB. Element 크기는 이 한계를 넘으면 안 된다(DmTensor<=512KB/slice, TrfTensor/VrfTensor<=8KB). (`quick-start.md:63-71, spatial-temporal-dimensions.md:71-77`)
- DM 접근 시 시작 주소가 size_of::<D>() 배수로 정렬되지 않으면 read-modify-write 사이클이 생겨 약 50배 느려진다. 그래서 Element 정렬이 하드웨어 제약으로 강제된다. (`spatial-temporal-dimensions.md:79`)
- DMA 엔진은 각 행을 64바이트 경계에서 시작하도록 요구한다. 행 길이가 64의 배수가 아니면 m![C, D # 64]처럼 패딩으로 정렬해야 한다. (`docs/src/mapping-tensors/mapping-expressions.md:144-149`)
- Time 차원은 하드웨어가 강제하는 크기 상한이 없고 처리할 데이터 양에 따라 커지는 순차 루프 카운터이며, Packet은 한 시간 반복마다 각 슬라이스가 받는 원소 수를 정하는 추가 공간 차원이다. 예: Slice=m![C/2](32슬라이스), Packet=m![C%2](2채널)면 반복당 32*2=64채널 처리. (`spatial-temporal-dimensions.md:81-143`)
- 현재 런타임은 칩 단위(#[device(chip = N)])로만 동작하여 칩·클러스터의 부분 사용은 아직 지원되지 않는다(향후 완화 가능). 따라서 클러스터를 덜 쓰면 m![1 # 2]처럼 패딩으로 2클러스터 요건을 채운다. (`spatial-temporal-dimensions.md:64-69`)
- stride/modulo 분해는 단순 메모리 레이아웃을 넘어 하드웨어 주소 비트 재배열(뱅크 인터리빙, 캐시 효율)에 직접 대응한다. m![B/64, B%32, B/32%2]는 버퍼 비트 묶음 [8:6]_[5:1]_[0]을 텐서 인덱스 [8:6]_[5]_[4:0]으로 재배치한다. (`mapping-expressions.md:292-329`)

➡️ 다음: [03_elementwise.md](./03_elementwise.md)
