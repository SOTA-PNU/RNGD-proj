# Intra-Slice Chain

Intra-Slice Chain 은 각 슬라이스의 데이터에 대해 원소 단위 연산, 이항 연산, 슬라이스 내 축약(리듀스) 연산을 독립적으로 수행한다.
활성함수나 정규화 같은 축약 이후 처리를 담당한다.
예를 들어 \\(\operatorname{sigmoid}(XW + b)\\) 를 계산할 때 \\(XW\\) 는 [Contraction Engine](../contraction-engine/index.md) 에서 실행하고, 덧셈과 sigmoid 활성함수는 Intra-Slice Chain 에서 실행한다.

## 인터페이스

아래 예제는 고정소수점 bias 를 적용하고, float 경로에서 sigmoid 를 실행하며(4-way 로 narrow 했다가 다시 widen), ReLU clip 으로 마무리한다.
\\(output[a, b] = \max(\operatorname{sigmoid}(input[a, b] + 100), 0)\\) 를 계산한다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2];

fn staged_pipeline<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]> {
    input
    .vector_init()
    .vector_intra_slice_tag(TagMode::Zero)
    // input + 100
    .vector_fxp(FxpBinaryOp::AddFxp, 100)
    // i32 → f32 (fixed-point, int_width = 31)
    .vector_fxp_to_fp(31)
    // Way8 → Way4 for the float path
    .vector_narrow_trim::<m![A % 2 # 4]>()
    // sigmoid(input + 100)
    .vector_fp_unary(FpUnaryOp::Sigmoid)
    // Way4 → Way8
    .vector_widen_pad::<m![A % 2 # 8]>()
    // f32 → i32
    .vector_fp_to_fxp(31)
    // max(sigmoid(input + 100), 0)
    .vector_clip(ClipBinaryOpI32::Max, 0)
    .vector_final()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, i32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = staged_pipeline(c);
```

### 파이프라인

체인은 (위에서 보인 것처럼) `vector_intra_slice_tag()` 로 시작하거나([`vector_init()`](./index.md#interface) 바로 뒤 또는 inter-slice reducer 의 출력에서), [Pair Mode](#pair-mode) 의 경우 `vector_intra_slice_unzip()` 으로 `vector_init()` 바로 뒤에서 시작한다(아래 참조).

```rust,ignore
#[primitive(VectorInitTensor::vector_intra_slice_tag)]
pub fn vector_intra_slice_tag(
    self,
    branch: TagMode,
) -> VectorBranchTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, D, Fresh, { VeOrder::IntraFirst }> {

#[primitive(VectorInitTensor::vector_intra_slice_unzip)]
pub fn vector_intra_slice_unzip<I: AxisName, TileTime: M, SplitTime: M>(
    self,
) -> VectorTensorPair<'l, T, D, stage::Tag, Chip, Cluster, Slice, SplitTime, Packet> {
```

진입 이후 체인은 아래 파이프라인 단계들을 고정된 순서로 거친다. 소프트웨어는 필요한 단계만 이어 붙이고 나머지는 건너뛴다. 예제가 `Logic`, `FpDiv`, `Filter` 를 건너뛴 것과 같다.
각 행은 체인에서 그 단계의 위치(`#`), 이름(`Stage`), 그 단계를 유발하는 API 메서드(`Method`), 실행되는 way(`Way`, 사이클당 8 원소 또는 4 원소), 피연산자를 받는지 여부(`Operand`)를 적는다.
타입 시스템이 모든 단계 전이를 컴파일 타임에 강제하므로, 메서드는 앞선 체인이 호환되는 상태에 도달한 뒤에만 호출할 수 있게 된다.
단계별 상세는 아래 [단계](#stages) 에 있다.

| # | Stage | Method | Way | Operand | → Inter-Slice Reducer |
|---|-------|--------|-----|-------------|------------------------|
| 1 | Entry | `vector_intra_slice_tag()` | Way8 | – | – |
| 2 | Logic | `vector_logic()` | Way8 | 예 | 예 |
| 3 | Fxp | `vector_fxp()` | Way8 | 예 | 예 |
| 4 | FxpToFp | `vector_fxp_to_fp()` | Way8 | – | 예 |
| 5 | Narrow | `vector_narrow_split()` / `vector_narrow_trim()` | Way8 → Way4 | – | – |
| 6 | Float | `vector_fp_unary/binary/ternary()` | Way4 | 예 | – |
| 7 | IntraSliceReduce | `vector_intra_slice_reduce()` | Way4 | – | – |
| 8 | FpDiv | `vector_fp_div()` | Way4 | 예 | – |
| 9 | Widen | `vector_widen_concat()` / `vector_widen_pad()` | Way4 → Way8 | – | 예 |
| 10 | FpToFxp | `vector_fp_to_fxp()` | Way8 | – | 예 |
| 11 | Clip | `vector_clip()` | Way8 | 예 | 예 |
| 12 | Filter | `vector_filter()` | Way8 | – | – |

각 단계는 8-way(사이클당 8 원소) 또는 4-way(사이클당 4 원소)로 실행된다.
부동소수점 클러스터는 처리량이 절반인 ALU 를 체인의 나머지에 맞춰 상쇄하려고 4-way 로 실행된다.
따라서 float 경로를 쓰는 체인은 8-way 로 진입하고, float 단계 앞에서 `Narrow`(`vector_narrow_split` 또는 `vector_narrow_trim`)를 호출하며, 그 뒤에 `Widen`(`vector_widen_concat` 또는 `vector_widen_pad`)을 호출해 8-way 로 돌아온다.
예제가 정확히 이렇게 한다. `vector_narrow_trim` 다음 `vector_fp_unary(Sigmoid)` 다음 `vector_widen_pad`.

<a id="transitioning-to-the-inter-slice-reducer"></a>

체인은 `vector_final()`(예제처럼) 또는 `vector_inter_slice_reduce()`([Inter-Slice Reducer](./inter-slice-reducer.md) 로 넘김)로 빠져나간다.
두 출구 모두 8-way 를 요구하므로, 활성화된 4-way 단계는 먼저 `Widen` 을 거쳐야 한다.

### 피연산자

이항 연산과 삼항 연산에는 두 종류의 슬롯이 있다. **stream**(진행 중인 텐서, 즉 메서드 체인의 `self` 이며 체인이 고정한다)과 하나 또는 두 개의 **operand**(추가 입력)다.
각 피연산자는 세 가지 소스 중 하나에서 온다.

| 소스 | 예 | 설명 |
|--------|---------|-------------|
| Constant | `100`, `2.5f32` | 모든 원소로 브로드캐스트되는 스칼라 |
| VRF 텐서 | `VeRhs::vrf(&vrf_tensor)` | Vector Engine 진입 전에 `.to_vrf()` 로 미리 로드 |
| Stash | `Stash` | 앞선 체인 단계의 스냅샷, 아래에서 설명 |

삼항 연산(`FmaF`)은 쌍 `(operand0, operand1)` 을 받는다.

같은 연산 메서드가 인자 타입에 따라 서로 다른 소스를 받아들인다:

```rust,ignore
.vector_fxp(FxpBinaryOp::AddFxp, 100)                // operand from constant
.vector_fxp(FxpBinaryOp::MulInt, VeRhs::vrf(&vrf))   // operand from VRF tensor
.vector_clip(ClipBinaryOpI32::Max, Stash)            // operand from stash (set earlier)
```

`Stash` 소스는 `vector_stash()` 에서 나온다. 이 메서드는 진행 중인 텐서를 스냅샷으로 남겨서, 이후의 이항 또는 삼항 연산이 `Stash` 피연산자로 다시 읽을 수 있게 한다.
전형적인 용도는 `max(f(x), x)` 같은 residual 이나 skip-connection 이며, 원래의 `x` 가 중간 단계들을 건너 살아남아야 하는 경우다.
`vector_stash()` 는 `Stashable` 단계(`Branch`, `Logic`, `Fxp`, `Narrow`, `Fp`, `FpDiv`, `Clip`) 어디서든 호출한다. 스냅샷은 Tensor Unit 호출이 끝날 때까지 살아 있고, `Stash` 를 받는 이후의 모든 이항·삼항 호출에 공급된다.
슬롯은 일회용이고(두 번째 `vector_stash()` 는 컴파일 타임 오류다) 타입이 있어서, `f32` stash 는 `f32` 연산에만 공급된다.
stash 는 읽기도 한 번뿐이다. 정확히 하나의 이후 연산에만 공급된다(읽으면 슬롯이 `Occupied` 를 지나므로, 두 번째 `Stash` 읽기는 컴파일 타임 오류다). 두 번 넘게 읽어야 하는 값은 stash 가 아니다 - 여러 번 읽을 수 있는 VRF(`VeRhs::vrf`)에 넣는다.
매핑은 진행 중인 텐서를 따라가므로, `Narrow` 전에 잡은 stash 는 `Widen` 뒤에도 그대로 쓸 수 있다.
Stash 는 [Pair Mode](#pair-mode) 에서 쓸 수 없고, [inter-slice reducer](./inter-slice-reducer.md) 로 가는 `IntraFirst` 전이가 stash 를 버리므로, `vector_inter_slice_reduce()` 전에 stash 한 것은 그 뒤에 사라진다.

그다음 **argument mode** 가 어느 슬롯이 stream 이고 어느 슬롯이 operand 인지 고르므로, 같은 연산으로 예컨대 `stream + operand` 나 `operand - stream` 을 계산할 수 있다.
예를 들어 `BinaryArgMode::Mode10` 은 슬롯을 뒤바꿔서 `SubFxp` 가 `operand - stream` 을 계산하게 한다:

```rust,ignore
.vector_fxp_with_mode(FxpBinaryOp::SubFxp, BinaryArgMode::Mode10, 7)  // computes 7 - stream
```

`BinaryArgMode` 는 이항 연산의 슬롯 중 어느 둘이 stream 이고 어느 것이 operand 인지 고른다(단항 연산은 모드가 없고 항상 `op(stream)` 으로 실행된다):

<a id="argument-modes"></a>

| BinaryArgMode | 슬롯 | 계산 |
|---------------|-------|-------------|
| `Mode00` | stream / stream | `op(stream, stream)` |
| `Mode01` | stream / operand | `op(stream, operand)` (기본값) |
| `Mode10` | operand / stream | `op(operand, stream)` |
| `Mode11` | operand / operand | `op(operand, operand)` |

`TernaryArgMode` 는 삼항 연산에 대해 같은 일을 한다:

| TernaryArgMode | 슬롯 | 계산 |
|----------------|-------|-------------|
| `Mode012` | stream / operand0 / operand1 | `op(stream, operand0, operand1)` (기본값) |
| `Mode002` | stream / stream / operand1 | `op(stream, stream, operand1)` |
| `Mode102` | operand0 / stream / operand1 | `op(operand0, stream, operand1)` |
| `Mode112` | operand0 / operand0 / operand1 | `op(operand0, operand0, operand1)` |
| `Mode020` | stream / operand1 / stream | `op(stream, operand1, stream)` |
| `Mode021` | stream / operand1 / operand0 | `op(stream, operand1, operand0)` |
| `Mode120` | operand0 / operand1 / stream | `op(operand0, operand1, stream)` |

<a id="pair-mode"></a>
### Pair Mode

Pair mode 는 원소가 교차 배치된 두 그룹으로 나뉘는 텐서에 대해 체인을 실행하므로, 연산이 두 그룹을 서로 관계지을 수 있다(예: 쌍 단위 덧셈, 비대칭 스케일).
진입은 `vector_intra_slice_unzip()` 이며, `vector_init()` 바로 뒤에서 2-way 그룹화 축을 지닌 collected 텐서에 적용한다.
체인을 `vector_intra_slice_unzip()` 으로 시작하면 이후의 [Filter](#filter) 단계를 쓸 수 없다.
내부적으로 `vector_intra_slice_unzip()` 은 `TagMode::AxisToggle` 을 사용해 2-way 그룹화 축에서 각 원소의 `GroupId` 를 유도한다.

흐름은 네 단계다:
1. `vector_intra_slice_unzip()` 이 입력을 두 개의 병렬 스트림(group 0 과 group 1)으로 나눈다.
2. 체인이 두 그룹을 보조를 맞춰 함께 단계들에 통과시킨다(**paired** 구간).
3. `_zip` 연산이 두 스트림을 다시 하나로 합친다(**merged** 구간).
4. 합쳐진 스트림은 일반 체인처럼 `vector_final()` 로 이어진다.

paired 구간의 단계는 두 갈래로 나뉜다:
- **공통 단계**(`vector_fxp_to_fp`, `vector_narrow_split`, `vector_widen_concat`, `vector_fp_to_fxp`)는 두 그룹에 동일하게 작용한다.
  `vector_narrow_trim` 과 `vector_widen_pad` 는 쌍에서 쓸 수 없다. 대신 `_split` / `_concat` 변형을 쓴다.
- **그룹별 연산**은 그룹마다 인자를 하나씩 받는다:
  - 이항과 삼항(`vector_fxp`, `vector_fp_binary`, `vector_fp_ternary`, `vector_clip` 등)은 한쪽에 `()` 를 주면 그쪽을 건너뛰고, 양쪽에 서로 다른 피연산자를 줄 수도 있다.
  - 단항(`vector_fp_unary`)은 예외다. 플래그 `(op, group0_apply, group1_apply)` 를 받으며 `false` 는 그 그룹을 건너뛴다.

Pair mode 는 연산에 따라 [`BinaryArgMode`](#argument-modes) 를 다르게 해석한다. 그룹별 연산(`vector_fxp_with_mode`, `vector_fp_binary_with_mode` 등)은 각 그룹 안에서 독립적으로 모드를 적용하고(`0` 은 그 그룹의 stream, `1` 은 그 그룹의 operand), `_zip` 연산(`vector_fxp_zip_with_mode` 등)은 두 슬롯을 두 그룹 스트림으로 받는다(`0` 은 Group 0 의 stream, `1` 은 Group 1 의 stream):

| `_zip` `BinaryArgMode` | 슬롯 | 계산 |
|------------------------|-------|-------------|
| `Mode00` | group0 / group0 | `op(group0, group0)` |
| `Mode01` | group0 / group1 | `op(group0, group1)` (기본값) |
| `Mode10` | group1 / group0 | `op(group1, group0)` |
| `Mode11` | group1 / group1 | `op(group1, group1)` |

Pair mode 의 제약:
- `stash()` 와 `filter()` 는 pair mode 전체(paired 구간과 merged 구간 모두)에서 쓸 수 없다.
- `_zip` 이전(paired 구간)에는 `vector_inter_slice_reduce()` 가 그룹별 텐서에서 제공되지 않으므로 체인이 [inter-slice reducer](./inter-slice-reducer.md) 로 전이할 수 없다. `_zip` 이후(merged 구간)에는 결과가 다시 `Commitable` 이 되며, 현재 단계가 그 전이를 지원하면 `vector_inter_slice_reduce()` 를 호출할 수 있다.
- ALU 사용은 두 그룹이 공유한다. 어느 한쪽 그룹에서 쓴 ALU 는 양쪽 모두에서 소비된 것으로 친다.

<a id="stages"></a>
## 단계

한 단계 안에서 각 ALU 는 Tensor Unit 호출당 최대 한 번 실행된다.
이는 여러 연산자가 단계 안의 ALU 풀을 공유하는 `Logic`, `Fxp`, `Fp`, `Clip` 에서 주로 문제가 된다.
예를 들어 `tanh(sqrt(x))` 는 `tanh` 와 `sqrt` 가 둘 다 `FpFpu` ALU 를 소비하므로 한 번의 Tensor Unit 호출에 담을 수 없다.

### Tag

Tag 단계는 체인의 진입점이며, flit 안의 각 32비트 원소에 4비트 `Tag`(0-15)를 부여한다. 이후 단계들은 이 태그로 조건부 연산을 적용한다.
비트 3(MSB)은 `GroupId` 이고, Filter 와 pair mode 가 원소를 Group 0 / Group 1 로 나누는 데 쓴다.
비트 0..2 는 비교 결과로 채워지는 범용 플래그 비트다.

`TagMode` 는 각 원소에 대해 4비트를 어떻게 계산할지 고른다:

| `TagMode` | 각 태그 비트가 채워지는 방식 |
|-----------|----------------------------|
| `Zero` | 네 비트 모두 0. 모든 원소의 태그가 0 이다. |
| `AxisToggle { axis }` | 비트 3(`GroupId`) = `axis_index % 2`(`axis` 축 기준). 비트 0..2 는 0 으로 남는다. |
| `Comparison([cmp0, cmp1, cmp2, cmp3])` | 각 원소 `x` 에 대해, 비트 `i` = `1` 인 것은 `cmp_i(x)` 가 성립할 때에 한한다. 네 비교는 같은 `x` 를 보며 그 dtype 과 일치해야 한다(전부 `InputCmpI32` 이거나 전부 `InputCmpF32`). 각 `cmp_i` 는 `InputCmp`(`Less`, `Greater`, `Equal`, `LessUnsigned`, `GreaterUnsigned`, `True`, `False`)에서 (op, boundary)를 독립적으로 고른다. |
| `ValidCount` | [Valid Count Generator](./vcg.md) 출력에서 유도한 비트. |
| `Vrf` | VRF 에서 로드한 비트. 앞선 TuExec 이 미리 써 둔 것이다(호출 간 태그 재사용을 가능하게 한다). |

예를 들어 `i32` 데이터와 `Comparison([Less{0}, Equal{5}, Greater{100}, True])` 에서, 원소 `x = 7` 은 비트 `0/0/0/1`(LSB 먼저)을 내므로 태그는 `0b1000 = 8` 이다.


태그가 부여되고 나면, 이후의 이항·삼항 연산은 `GroupId` MSB 를 기준으로 피연산자에 조건을 걸어 태그 그룹마다 다른 값을 보게 할 수 있다.
`BinaryOperandTag::always(operand)` 는 모든 그룹에 적용되고, `BinaryOperandTag::group(operand, GroupId::Zero)` 는 group 0 에만 적용된다.
`TernaryOperandTag` 는 그 삼항 형태다.

### Logic Cluster

Logic Cluster 는 `i32` 또는 `f32`(비트 수준)에 대해 비트 단위 연산을 수행한다.
8-way 로 실행된다.

이 단계는 ALU 클래스 다섯 개(`LogicAnd`, `LogicOr`, `LogicXor`, `LogicLshift`, `LogicRshift`)를 노출하며, 각각 Tensor Unit 호출당 한 번 실행할 수 있다.
같은 클래스를 공유하는 연산자들은 한 호출로 합칠 수 없다.

`i32` 연산:

| Op | ALU | 설명 |
|----|-----|------|
| `BitAnd` | `LogicAnd` | 비트 and |
| `BitOr` | `LogicOr` | 비트 or |
| `BitXor` | `LogicXor` | 비트 xor |
| `LeftShift` | `LogicLshift` | 논리 왼쪽 시프트 |
| `LogicRightShift` | `LogicRshift` | 논리 오른쪽 시프트 |
| `ArithRightShift` | `LogicRshift` | 산술 오른쪽 시프트 |

`f32` 연산:

| Op | ALU | 설명 |
|----|-----|------|
| `BitAnd` | `LogicAnd` | fp 비트 패턴에 대한 비트 and |
| `BitOr` | `LogicOr` | fp 비트 패턴에 대한 비트 or |
| `BitXor` | `LogicXor` | fp 비트 패턴에 대한 비트 xor |

### Fxp Cluster

Fxp Cluster 는 `i32` 에 대해 정수 연산과 고정소수점 연산을 수행한다.
8-way 로 실행된다.

이 단계는 ALU 클래스 네 개(`FxpAdd`, `FxpLshift`, `FxpMul`, `FxpRshift`)를 노출하며, 각각 Tensor Unit 호출당 한 번 실행할 수 있다.
같은 클래스를 공유하는 연산자들은 한 호출로 합칠 수 없다.

| Op | ALU | 설명 |
|----|-----|------|
| `AddFxp` | `FxpAdd` | 랩어라운드 덧셈 |
| `AddFxpSat` | `FxpAdd` | 포화 덧셈 |
| `SubFxp` | `FxpAdd` | 랩어라운드 뺄셈 |
| `SubFxpSat` | `FxpAdd` | 포화 뺄셈 |
| `LeftShift` | `FxpLshift` | 논리 왼쪽 시프트 |
| `LeftShiftSat` | `FxpLshift` | 포화 왼쪽 시프트 |
| `MulFxp` | `FxpMul` | 고정소수점 곱셈 |
| `MulInt` | `FxpMul` | 정수 곱셈 |
| `LogicRightShift` | `FxpRshift` | 논리 오른쪽 시프트 |
| `ArithRightShift` | `FxpRshift` | 산술 오른쪽 시프트 |
| `ArithRightShiftRound` | `FxpRshift` | 반올림이 있는 산술 오른쪽 시프트 |

단일 ALU 규칙은 예를 들어 둘 다 `FxpAdd` 를 대상으로 하는 두 연산을 거부한다:

```rust,ignore
// PANICS: "FxpAdd is already in use"
input
    .vector_init()
    .vector_intra_slice_tag(TagMode::Zero)
    .vector_fxp(FxpBinaryOp::AddFxp, 10)    // uses FxpAdd
    .vector_fxp(FxpBinaryOp::MulInt, 2)     // uses FxpMul ✓
    .vector_fxp(FxpBinaryOp::SubFxp, 5)     // uses FxpAdd again ✗
    .vector_final()
```

### FxpToFp 변환

FxpToFp 변환 단계는 `i32` 를 `f32` 로 변환한다.
`int_width` 매개변수는 변환에 쓸 정수 비트 폭을 지정한다. `int_width = 31` 이 표준 `i32` ↔ `f32` 변환이다.

| 메서드 | 효과 |
|--------|--------|
| `vector_fxp_to_fp(int_width)` | `i32` 스트림을 `f32` 로 변환 |

### Narrow

`Narrow` 단계는 8-way 를 4-way 로 전환한다.
8-way packet 은 활성 원소 8개를 담고(`Packet = m![... # 8]`), 4-way packet 은 4개를 담는다(`Packet = m![... # 4]`).
narrow 는 float 및 리듀스 경로의 처리량을 절반으로 줄이므로, 같은 논리 텐서 모양이 두 배의 packet 또는 Tensor Unit 호출을 쓴다.

| 메서드 | 사용 시점 | 효과 |
|--------|----------|--------|
| `vector_narrow_split()` | 양쪽 절반 모두 실제 데이터를 담을 때 | 8-way flit 하나를 앞 4개 packet 과 뒤 4개 packet 으로 나누고 `Time` 과 `Packet` 을 갱신 |
| `vector_narrow_trim()` | 뒤쪽 4개 원소가 이미 패딩이거나 무관할 때 | 앞쪽 4개 원소만 남김 |

모양 의미:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2, S = 64];

fn vector_narrow_split_semantics<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![B], m![S / 4 # 256], m![S % 4], m![A % 8], i32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorNarrowTensor<'l, T, i32, m![1], m![B], m![S / 4 # 256], m![S % 4, A / 4 % 2], m![A % 4], i32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input.vector_narrow_split::<m![S % 4, A / 4 % 2], m![A % 4]>()
    // shape semantics: [T], [P] -> [T, P / 2], [P % 4]
}

fn vector_narrow_trim_semantics<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8], f32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorNarrowTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 4], f32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input.vector_narrow_trim::<m![A % 2 # 4]>()
    // shape semantics: [T], [P] -> [T], [P = 4]
}
# 
# let mut ctx = Context::acquire();
# 
# let i: VectorBranchTensor<'_, _, i32, m![1], m![B], m![S / 4 # 256], m![S % 4], m![A % 8], i32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let _o = vector_narrow_split_semantics(i);
# 
# let i: VectorBranchTensor<'_, _, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8], f32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let _o = vector_narrow_trim_semantics(i);
```

### Float Cluster

Float Cluster 는 `f32` 에 대해 단항·이항·삼항 부동소수점 연산을 제공한다.
4-way 로 실행되므로 입력은 이미 `Narrow` 를 거쳤어야 한다.

독립적인 ALU 다섯 개(`FpFma`, `FpFpu`, `FpExp`, `FpMul0`, `FpMul1`)를 노출하며, 각각 Tensor Unit 호출당 한 번 실행할 수 있다.
ALU 계획이 가장 중요한 단계가 여기다.

단항 연산:

| Op | ALU | 설명 |
|----|-----|------|
| `Exp` | `FpExp` | 지수 |
| `NegExp` | `FpExp` | 음의 지수 |
| `Sqrt` | `FpFpu` | 제곱근 |
| `Tanh` | `FpFpu` | 쌍곡탄젠트 |
| `Sigmoid` | `FpFpu` | 시그모이드 |
| `Erf` | `FpFpu` | 오차 함수 |
| `Log` | `FpFpu` | 자연로그 |
| `Sin` | `FpFpu` | 사인 |
| `Cos` | `FpFpu` | 코사인 |

이항 연산:

| Op | ALU | 설명 |
|----|-----|------|
| `AddF` | `FpFma` | 부동소수점 덧셈 |
| `SubF` | `FpFma` | 부동소수점 뺄셈 |
| `MulF(FpMulAlu::Mul0)` | `FpMul0` | 곱셈 |
| `MulF(FpMulAlu::Mul1)` | `FpMul1` | 곱셈 |
| `MulF(FpMulAlu::Fma)` | `FpFma` | 곱셈 |
| `DivF` | `FpFpu` | `Fp` 단계 안에서의 나눗셈 |

삼항 연산:

| Op | ALU | 설명 |
|----|-----|------|
| `FmaF` | `FpFma` | 융합 곱셈-덧셈 |

예를 들어 `exp(sqrt(((x + 1) * 2) * 3))` 을 계산하려면:
- `x1 = x + 1` 은 FpFma 로(`FpBinaryOp::AddF`)
- `x2 = x1 * 2` 는 FpMul0 로(`FpBinaryOp::MulF(FpMulAlu::Mul0)`)
- `x3 = x2 * 3` 은 FpMul1 로(`FpBinaryOp::MulF(FpMulAlu::Mul1)`)
- `x4 = sqrt(x3)` 은 FpFpu 로(`FpUnaryOp::Sqrt`)
- `x5 = exp(x4)` 는 FpExp 로(`FpUnaryOp::Exp`)

### IntraSliceReduce

IntraSliceReduce 단계는 슬라이스 하나 안에서 축을 리듀스한다.
4-way 로 실행된다.
이 단계는 전용 누산기 트리 ALU 를 사용하므로 사용자가 고를 수 있는 ALU 를 노출하지 않는다.

| 데이터 타입 | 지원 연산 |
|-----------|---------------|
| `i32` | `AddSat`, `Max`, `Min` |
| `f32` | `Add`, `Max`, `Min` |

자세한 내용은 [Intra-Slice Reduce](./intra-slice-reduce.md) 를 참조한다.

### FpDiv

FpDiv 단계는 부동소수점 나눗셈을 수행한다.
4-way 로 실행된다.
이 단계는 전용 부동소수점 제산기를 사용하므로 사용자가 고를 수 있는 ALU 를 노출하지 않는다.

| Op | 설명 |
|----|------|
| `FpDivBinaryOp::DivF` | 전용 부동소수점 나눗셈 |

### Widen

`Widen` 단계는 4-way 에서 다시 8-way 로 전이한다.
그러면 이후 단계들(`FpToFxp`, `Clip`, `Filter`, `Output`)이 다시 8원소 packet 을 본다.

| 메서드 | 사용 시점 | 효과 |
|--------|----------|--------|
| `vector_widen_concat()` | 앞선 `vector_narrow_split()` 을 되돌릴 때 | 4-way packet 두 개를 다시 8-way flit 하나로 합침 |
| `vector_widen_pad()` | 앞선 `vector_narrow_trim()` 을 되돌릴 때 | 4-way packet 을 무효 채움값으로 8원소까지 패딩 |

모양 의미:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2, S = 64, R = 8];

fn vector_widen_concat_semantics<'l, const T: Tu>(
    input: VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![B], m![S / 4 # 256], m![A / 4 % 2], m![A % 4], i32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorWidenTensor<'l, T, i32, m![1], m![B], m![S / 4 # 256], m![1], m![A % 8], i32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input.vector_widen_concat::<m![1], m![A % 8]>()
    // shape semantics: [T, P / 2], [P % 4] -> [T], [P]
}

fn vector_widen_pad_semantics<'l, const T: Tu>(
    input: VectorFpTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 4], f32, Fresh, { stage::VeOrder::IntraFirst }>,
) -> VectorWidenTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8], f32, Fresh, { stage::VeOrder::IntraFirst }>
{
    input.vector_widen_pad::<m![A % 2 # 8]>()
    // shape semantics: [T], [P] -> [T], [P # 8]
}
# 
# let mut ctx = Context::acquire();
# 
# let i: VectorBranchTensor<'_, _, i32, m![1], m![B], m![S / 4 # 256], m![R, A / 4 % 2], m![A % 4 # 8], i32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let i = i
#     .vector_narrow_trim::<m![A % 4]>()
#     .vector_intra_slice_reduce::<R, m![A / 4 % 2], m![A % 4]>(IntraSliceReduceOpI32::AddSat);
# 
# let _o = vector_widen_concat_semantics(i);
# 
# let i: VectorBranchTensor<'_, _, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8], f32, Fresh, { stage::VeOrder::IntraFirst }> = VectorBranchTensor::new(&mut ctx.main, Tensor::zero(), TagMode::Zero);
# let i = i.vector_narrow_trim::<m![A % 2 # 4]>().vector_fp_unary(FpUnaryOp::Exp);
# let _o = vector_widen_pad_semantics(i);
```

### FpToFxp 변환

FpToFxp 변환 단계는 `f32` 를 다시 `i32` 로 변환한다.
`int_width` 매개변수는 정수 비트 폭을 지정한다.

| 메서드 | 효과 |
|--------|--------|
| `vector_fp_to_fxp(int_width)` | `f32` 스트림을 다시 `i32` 로 변환 |

### Clip Cluster

Clip Cluster 는 클램핑 연산과 비교 연산을 수행한다.
8-way 로 실행된다.

이 단계는 ALU 클래스 세 개(`ClipAdd`, `ClipMax`, `ClipMin`)를 노출하며, 각각 Tensor Unit 호출당 한 번 실행할 수 있다.

`i32` 연산:

| Op | ALU | 설명 |
|----|-----|------|
| `Min` | `ClipMin` | 최솟값 |
| `Max` | `ClipMax` | 최댓값 |
| `AbsMin` | `ClipMin` | 절대 최솟값 |
| `AbsMax` | `ClipMax` | 절대 최댓값 |
| `AddFxp` | `ClipAdd` | 랩어라운드 덧셈 |
| `AddFxpSat` | `ClipAdd` | 포화 덧셈 |

`f32` 연산:

| Op | ALU | 설명 |
|----|-----|------|
| `Min` | `ClipMin` | 최솟값 |
| `Max` | `ClipMax` | 최댓값 |
| `AbsMin` | `ClipMin` | 절대 최솟값 |
| `AbsMax` | `ClipMax` | 절대 최댓값 |
| `Add` | `ClipAdd` | 부동소수점 덧셈 |

<a id="filter"></a>
### Filter

Filter 단계는 `TagFilter` 에서 유도한 실행 마스크(각 원소 `Tag` 의 `GroupId` MSB 로 매칭)를 적용해 출력 flit 을 걸러낸다.
8-way 이면서 `Standalone` 컨텍스트에서만 쓸 수 있다.
소스 `impl` 은 `VectorTensor` 위에 `CanTransitionTo<Filter>` 를 가진 모든 단계에 대해 존재하며, 이는 모든 intra-slice 단계와 `InterSliceReduce` 를 포함한다.


### Output

Output 단계는 Vector Engine 파이프라인에서 빠져나간다.
결과는 [Cast Engine](../cast-engine.md), [Transpose Engine](../transpose-engine.md), [Commit Engine](../../moving-tensors/commit-engine.md) 으로 이어질 수 있다.

## 예제

### `i32` 파이프라인

분기 뒤에 상수를 더하는 최소한의 `i32` 체인이다.
Fxp 단계는 8-way 로 실행되므로 narrow 나 widen 이 필요 없다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 2048, B = 2];

fn add_constant<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]> {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, 100)
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
#
# let i: CollectTensor<'_, _, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = add_constant(i);
```

### `f32` 파이프라인

`vector_narrow_trim()` 은 float 연산 전에 텐서를 8-way 에서 4-way 로 바꾸는 `Narrow` 단계다.
`vector_widen_pad()` 는 그 뒤에 다시 8-way 로 바꾸는 `Widen` 단계다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2];

fn sigmoid<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]> {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![A % 2 # 4]>() // Narrow: Way8 -> Way4
        .vector_fp_unary(FpUnaryOp::Sigmoid)
        .vector_widen_pad::<m![A % 2 # 8]>() // Widen: Way4 -> Way8
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
#
# let i: CollectTensor<'_, _, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = sigmoid(i);
```

### 단일 스트림 Argument Mode

`BinaryArgMode::Mode10` 은 stream 과 operand 위치를 뒤바꾸므로, `SubFxp` 는 `operand - stream`(여기서는 `7 - x`)을 계산하며, 기본값인 `stream - operand` 가 아니다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 2048, B = 2];

fn bias_minus_x<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]> {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_with_mode(FxpBinaryOp::SubFxp, BinaryArgMode::Mode10, 7) // compute 7 - x
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
#
# let i: CollectTensor<'_, _, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = bias_minus_x(i);
```

### VRF 피연산자

미리 로드해 둔 VRF 데이터를 피연산자로 쓴다:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 2048, B = 2, N = 256];

fn vrf_add<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8], m![N], m![A % 8]>,
    vrf: &VrfTensor<i32, m![1], m![B], m![A / 8], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8], m![N], m![A % 8]> {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, vrf)
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
#
# let i: CollectTensor<'_, _, i32, m![1], m![B], m![A / 8], m![N], m![A % 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let v: VrfTensor<i32, m![1], m![B], m![A / 8], m![A % 8]> = unsafe { VrfTensor::from_addr(0) };
# let _o = vrf_add(i, &v);
```

### Fp 전용 경로에서의 Stash

이른 단계에서 stash 한 다음, 나중에 Clip 연산에서 사용한다.
`max(2 * x, x)` 를 구현한다:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2];

fn residual_max<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]> {
    input
        .vector_init()                                        // enter VE
        .vector_intra_slice_tag(TagMode::Zero) // start the intra-slice path
        .vector_stash()                                       // save original x
        .vector_narrow_trim::<m![A % 2 # 4]>()                  // narrow to Way4
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 2.0f32) // compute 2 * x
        .vector_widen_pad::<m![A % 2 # 8]>()                   // widen back to Way8
        .vector_clip(ClipBinaryOpF32::Max, Stash)             // max(2 * x, x)
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
#
# let i: CollectTensor<'_, _, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = residual_max(i);
```

### Fxp 전용 경로에서의 Stash

이른 단계에서 stash 한 다음, 나중에 Clip 연산에서 사용한다.
`max(x + bias, x)` 를 구현한다:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 2048, B = 2];

fn stash_at_fxp<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]> {
    input
        .vector_init()                                      // enter VE
        .vector_intra_slice_tag(TagMode::Zero) // start the intra-slice path
        .vector_stash()                                     // save original x
        .vector_fxp(FxpBinaryOp::AddFxp, 100)               // compute x + bias
        .vector_clip(ClipBinaryOpI32::Max, Stash)           // compute max(x + bias, x)
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
#
# let i: CollectTensor<'_, _, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = stash_at_fxp(i);
```

### Narrow 와 Widen 을 건너는 Stash

narrow 전에 stash 하고, widen 후에 소비한다.
`max(sigmoid(x), x)` 를 계산한다:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2];

fn stash_across_narrow_widen<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]> {
    input
        .vector_init()                                     // enter VE
        .vector_intra_slice_tag(TagMode::Zero) // start the intra-slice path
        .vector_stash()                                    // save x (Way8)
        .vector_narrow_trim::<m![A % 2 # 4]>()               // narrow to Way4
        .vector_fp_unary(FpUnaryOp::Sigmoid)               // compute sigmoid(x) in Way4
        .vector_widen_pad::<m![A % 2 # 8]>()                // widen back to Way8
        .vector_clip(ClipBinaryOpF32::Max, Stash)          // compute max(sigmoid(x), x)
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
#
# let i: CollectTensor<'_, _, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = stash_across_narrow_widen(i);
```

### Pair 덧셈

교차 배치된 두 그룹을 정수 덧셈으로 zip 한다:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 2048, B = 2, I = 2];

fn pair_add<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8], m![I], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]> {
    input
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
#
# let i: CollectTensor<'_, _, i32, m![1], m![B], m![A / 8], m![I], m![A % 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = pair_add(i);
```

### Pair 한쪽 전처리

비대칭 전처리로 zip 전에 group 0 만 스케일한다:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 2048, B = 2, I = 2];

fn pair_preprocess_one_side<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8], m![I], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]> {
    input
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_fxp(FxpBinaryOp::MulInt, 10, ())   // group 0 only
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
#
# let i: CollectTensor<'_, _, i32, m![1], m![B], m![A / 8], m![I], m![A % 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = pair_preprocess_one_side(i);
```

### Zip 을 쓰는 Pair Float 파이프라인

두 그룹 모두 float 경로를 지난다(narrow -> fp -> zip -> widen):

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2, I = 2];

fn pair_fp_mul_zip<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![B], m![A / 2], m![I], m![A % 2 # 8]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]> {
    input
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_narrow_split::<m![1 # 2], m![A % 2 # 4]>()        // both groups: Way8 -> Way4
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))   // group0 * group1 (Way4)
        .vector_widen_concat::<m![1], m![A % 2 # 8]>()           // Way4 -> Way8
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
#
# let i: CollectTensor<'_, _, f32, m![1], m![B], m![A / 2], m![I], m![A % 2 # 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = pair_fp_mul_zip(i);
```

### Pair 그룹별 전처리

zip 하기 전에 각 그룹에 서로 다른 연산을 적용한다:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2, I = 2];

fn pair_asymmetric_preprocess<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![B], m![A / 2], m![I], m![A % 2 # 8]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]> {
    input
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_narrow_split::<m![1 # 2], m![A % 2 # 4]>()
        .vector_fp_unary(FpUnaryOp::Exp, true, false)         // group 0: exp(x), group 1: skip
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))   // exp(group0) * group1
        .vector_widen_concat::<m![1], m![A % 2 # 8]>()
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
#
# let i: CollectTensor<'_, _, f32, m![1], m![B], m![A / 2], m![I], m![A % 2 # 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = pair_asymmetric_preprocess(i);
```

### Pair Zip 의 Argument Mode

`BinaryArgMode::Mode10` 은 zip 할 때 두 그룹 스트림을 뒤바꾼다:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2, I = 2];

fn pair_sub_reverse<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![B], m![A / 2], m![I], m![A % 2 # 8]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]> {
    input
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_narrow_split::<m![1 # 2], m![A % 2 # 4]>()
        .vector_fp_zip_with_mode(FpBinaryOp::SubF, BinaryArgMode::Mode10) // compute group1 - group0
        .vector_widen_concat::<m![1], m![A % 2 # 8]>()
        .vector_final()
}
# 
# let mut ctx = Context::acquire();
#
# let i: CollectTensor<'_, _, f32, m![1], m![B], m![A / 2], m![I], m![A % 2 # 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = pair_sub_reverse(i);
```

## 성능

처리량은 Logic, Fxp, Clip 클러스터에서 온전한 8-way(사이클당 8 원소)다.
Float 클러스터는 4-way 로 실행되며, float 경로를 감싸는 Narrow/Widen 이 실제로는 실효 처리량을 절반으로 만든다.

지연은 사용한 ALU 하나당 한 사이클씩 늘어난다.
여러 ALU 에 걸치는 연산은 지연이 누적된다.
예를 들어 `exp(sqrt(x))` 는 2 사이클을 더한다(sqrt 의 FpFpu 와 exp 의 FpExp).
