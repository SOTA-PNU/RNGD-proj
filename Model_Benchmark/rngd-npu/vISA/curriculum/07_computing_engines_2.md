# 07 · 연산 엔진 II — Vector/Cast/Transpose

이 문서는 vISA 커리큘럼 모듈 07입니다. Vector 엔진의 서브 파이프라인과 풍부한 op 셋(Exp/Sqrt/Tanh/Sigmoid, reduce, clip 등)으로 softmax·layernorm을 조립하는 법, 그리고 Cast·Transpose의 제약을 배웁니다.
*선행: 06 연산 엔진 I · 예상 시간: 하루*

## 학습 목표

- [ ] vector_init→intra-slice chain→vector_final 구조를 안다
- [ ] intra/inter-slice reduce와 VCG(Valid Count Generator)를 안다
- [ ] Vector op로 max-subtract→exp→sum→div(softmax)를 따라간다
- [ ] Cast(dtype 변환)와 Transpose(within-flit) 제약을 안다

## 1. 개념

## 0. 큰 그림: Vector Engine은 무엇을 하나요

RNGD의 한 Tensor Unit 안에는 여러 엔진이 파이프라인으로 줄지어 있습니다. 그중 Vector Engine(VE)은 "원소 단위(elementwise) 계산"과 "리덕션(reduction)"을 담당합니다. 활성화 함수(GELU, SiLU), 정규화(softmax, layer norm), 이항/삼항 연산, 그리고 축 합/최대/최소가 전부 여기서 나옵니다 (docs/src/computing-tensors/vector-engine/index.md:1-4).

가장 먼저 기억할 사실 하나: **VE는 32비트 타입(i32, f32)만 받습니다** (index.md:6). 왜냐면 VE는 정밀도 손실 없이 누적·계산하는 단계라서, 입력을 미리 32비트로 넓혀(widen) 두어야 하기 때문입니다. 보통 앞단의 Contraction Engine이 자동으로 넓혀줍니다(bf16 곱은 f32로, i8 곱은 i32로 누적). Contraction을 건너뛰면 Fetch Engine의 타입 캐스트 어댑터가 대신 넓혀 줍니다 (index.md:7-8). 좁은 타입(bf16, i8 등)으로 되돌리는 일은 VE가 끝난 뒤 Cast Engine이 합니다(아래 8장).

VE 호출은 메서드 체인으로 표현합니다. 시작은 항상 `vector_init()`, 끝은 항상 `vector_final()`입니다 (index.md:12). 이 둘 사이에 두 개의 하위 부품이 들어갑니다.

- **Intra-Slice Chain(슬라이스 내부 체인)**: 각 슬라이스의 데이터를 독립적으로 원소 단위/이항/슬라이스 내부 리덕션 처리 (intra-slice-chain.md).
- **Inter-Slice Reducer(슬라이스 간 리듀서)**: 한 클러스터의 256개 슬라이스를 가로질러 리덕션 (inter-slice-reducer.md).

둘 다 쓸 수도, 하나만 쓸 수도 있습니다. 둘 다 쓸 때 순서는 두 가지뿐입니다: `IntraFirst`(체인 먼저, 그다음 리듀서) 또는 `InterFirst`(리듀서 먼저, 그다음 체인) (index.md:15). 이 순서는 타입에 `VeOrder` 상수로 박혀서 컴파일 타임에 강제됩니다.

진입 메서드는 세 가지입니다 (index.md:17-19):
- `vector_intra_slice_tag(...)` — 체인을 단일 스트림으로 시작.
- `vector_intra_slice_unzip::<...>()` — 입력이 2-way 그룹 축을 들고 있을 때 두 병렬 스트림으로 쪼개 시작(Pair Mode).
- `vector_inter_slice_reduce::<...>(...)` — 리듀서부터 시작(InterFirst).

가장 단순한 ReLU 예시를 보면 감이 옵니다 (index.md:50-59):
```rust
input
    .vector_init()
    .vector_intra_slice_tag(TagMode::Zero)
    .vector_clip(ClipBinaryOpF32::Max, 0.0f32)  // max(x, 0)
    .vector_final()
```
리듀서를 안 쓰니 경로는 자동으로 IntraFirst로 풀립니다.

---

## 1. Intra-Slice Chain — 고정된 12단 파이프라인

체인의 핵심 개념은 **단계(stage)가 고정된 순서로 줄지어 있고, 필요한 것만 호출하고 나머지는 건너뛴다**는 것입니다 (intra-slice-chain.md:52). 그리고 **모든 단계 전이를 타입 시스템이 컴파일 타임에 검사**합니다. 즉 앞 단계가 적절한 상태에 도달해야만 다음 메서드가 호출 가능해집니다 (intra-slice-chain.md:54). 잘못된 순서로 부르면 컴파일이 안 됩니다.

순서표 (intra-slice-chain.md:57-70):

| # | 단계 | 메서드 | Way | 피연산자 |
|---|------|--------|-----|---------|
| 1 | Entry(Tag) | `vector_intra_slice_tag()` | 8 | – |
| 2 | Logic | `vector_logic()` | 8 | 있음 |
| 3 | Fxp | `vector_fxp()` | 8 | 있음 |
| 4 | FxpToFp | `vector_fxp_to_fp()` | 8 | – |
| 5 | Narrow | `vector_narrow_split()` / `vector_narrow_clip()` | 8→4 | – |
| 6 | Float | `vector_fp_unary/binary/ternary()` | 4 | 있음 |
| 7 | IntraSliceReduce | `vector_intra_slice_reduce()` | 4 | – |
| 8 | FpDiv | `vector_fp_div()` | 4 | 있음 |
| 9 | Widen | `vector_widen_concat()` / `vector_widen_pad()` | 4→8 | – |
| 10 | FpToFxp | `vector_fp_to_fxp()` | 8 | – |
| 11 | Clip | `vector_clip()` | 8 | 있음 |
| 12 | Filter | `vector_filter()` | 8 | – |

### Way8 vs Way4 (왜 narrow/widen이 있나)

"Way"는 한 사이클에 처리하는 원소 수입니다. 대부분 단계는 8-way(8개/사이클)로 돕니다. 그런데 **부동소수 클러스터(Float, IntraSliceReduce, FpDiv)는 ALU가 절반 처리량이라 4-way로 돕니다** (intra-slice-chain.md:72-73). 그래서 float 경로를 쓰려면 그 앞에서 `Narrow`로 8→4로 줄이고, 끝나면 `Widen`으로 4→8로 되돌려야 합니다 (intra-slice-chain.md:74). 8-way 패킷은 `m![... # 8]`, 4-way는 `m![... # 4]`로 표기합니다.

- `vector_narrow_split()` — 앞4·뒤4 둘 다 진짜 데이터일 때 한 8-way flit을 앞4/뒤4 두 패킷으로 쪼갬.
- `vector_narrow_clip()` — 뒤 4개가 이미 패딩/불필요일 때 앞 4개만 남김.
- 되돌릴 때는 대칭으로 `vector_widen_concat()`(split의 역) / `vector_widen_pad()`(clip의 역) (intra-slice-chain.md:280-388).

체인의 출구(`vector_final()` 또는 `vector_inter_slice_reduce()`)는 둘 다 8-way를 요구하므로, 4-way 단계가 살아 있으면 반드시 먼저 Widen을 거쳐야 합니다 (intra-slice-chain.md:80).

실제 f32 시그모이드 한 번 거는 최소 파이프라인 (intra-slice-chain.md:498-508 / 예제 코드 furiosa-opt-examples/src/vector_engine/normal.rs):
```rust
.vector_intra_slice_tag(TagMode::Zero)
.vector_narrow_clip::<m![A % 2 # 4]>()   // Way8 → Way4
.vector_fp_unary(FpUnaryOp::Sigmoid)
.vector_widen_pad::<m![A % 2 # 8]>()     // Way4 → Way8
```

### 단계별 상세

**Tag(진입)** — 각 32비트 원소에 4비트 `Tag`(0~15)를 붙입니다. 비트3(MSB)은 `GroupId`로 Filter와 Pair Mode가 쓰고, 비트0~2는 비교 결과로 채우는 범용 플래그입니다 (intra-slice-chain.md:182-184). `TagMode`로 비트 채우는 방식을 고릅니다 (intra-slice-chain.md:188-194):
- `Zero` — 전부 0(가장 흔함).
- `AxisToggle { axis }` — 비트3 = `axis_index % 2` (unzip이 내부적으로 사용).
- `Comparison([cmp0..cmp3])` — 원소 x에 대해 비트i = cmp_i(x). 같은 x를 보고 같은 dtype이어야 함. 예: i32 데이터에 `Comparison([Less{0}, Equal{5}, Greater{100}, True])`이고 x=7이면 비트 0/0/0/1 → tag=0b1000=8 (intra-slice-chain.md:196).
- `ValidCount` — VCG 출력에서 유도(아래 5장).
- `Vrf` — 이전 TuExec가 VRF에 써둔 태그 재사용.
태그가 붙으면 이후 이항/삼항 연산이 GroupId에 따라 피연산자를 다르게 줄 수 있습니다(`BinaryOperandTag::always` / `::group`).

**Logic 클러스터** — i32 또는 f32(비트 패턴)에 대한 비트 연산. 8-way (intra-slice-chain.md:203-228). ALU 클래스 5개: `LogicAnd/Or/Xor/Lshift/Rshift`. i32는 BitAnd/Or/Xor/LeftShift/LogicRightShift/ArithRightShift, f32는 BitAnd/Or/Xor. (ArithRightShift와 LogicRightShift는 같은 `LogicRshift` ALU를 공유함에 주의.)

**Fxp 클러스터** — i32 정수·고정소수 산술. 8-way (intra-slice-chain.md:230-251). ALU 4개: `FxpAdd, FxpLshift, FxpMul, FxpRshift`. 대표 연산: AddFxp(wrapping)/AddFxpSat/SubFxp/SubFxpSat(전부 FxpAdd 공유), LeftShift/LeftShiftSat(FxpLshift), MulFxp/MulInt(FxpMul), LogicRightShift/ArithRightShift/ArithRightShiftRound(FxpRshift).

**FxpToFp 변환** — i32→f32. `int_width` 인자로 정수 비트폭 지정, `int_width=31`이 표준 i32↔f32 변환 (intra-slice-chain.md:265-272).

**Float 클러스터** — f32 unary/binary/ternary. 4-way. **여기가 ALU 계획이 제일 중요한 곳** (intra-slice-chain.md:310-347). 독립 ALU 5개: `FpFma, FpFpu, FpExp, FpMul0, FpMul1`.
- Unary: Exp/NegExp(FpExp), Sqrt/Tanh/Sigmoid/Erf/Log/Sin/Cos(전부 FpFpu).
- Binary: AddF/SubF(FpFma), MulF(Mul0)→FpMul0, MulF(Mul1)→FpMul1, MulF(Fma)→FpFma, DivF(FpFpu).
- Ternary: FmaF(FpFma).
곱을 세 개나 한 invocation에 넣을 수 있는 비결이 `MulF`의 ALU 선택(Mul0/Mul1/Fma)입니다. 예: `exp(sqrt(((x+1)*2)*3))`은 AddF(FpFma)→MulF(Mul0)→MulF(Mul1)→Sqrt(FpFpu)→Exp(FpExp)로 ALU가 안 겹치게 배치합니다 (intra-slice-chain.md:349-354).

**IntraSliceReduce** — 슬라이스 내부(Time, Packet)의 축을 접습니다. 4-way. 전용 누적 트리 ALU(`ReduceAccTree`)라 사용자가 ALU를 고르지 않습니다 (intra-slice-chain.md:356-367). i32: AddSat/Max/Min, f32: Add/Max/Min. (자세히 3장.)

**FpDiv** — 전용 부동소수 나눗셈기(`ReduceFpDiv`). 4-way. `FpDivBinaryOp::DivF` (intra-slice-chain.md:369-377). 주의: float 나눗셈은 두 길이 있습니다 — Fp 단계 안의 `vector_fp_binary(DivF)`(FpFpu ALU 사용)과, 별도 FpDiv 단계의 `vector_fp_div()`(전용 divider). 전자는 다른 FpFpu 연산(sqrt 등)과 경합하고, 후자는 독립 자원이라 sqrt와 같이 쓸 수 있습니다.

**FpToFxp 변환** — f32→i32. `int_width` 지정 (intra-slice-chain.md:414-421).

**Clip 클러스터** — 클램핑/비교. 8-way. ALU 3개: `ClipAdd, ClipMax, ClipMin` (intra-slice-chain.md:423-449). i32/f32 공통 Min/Max/AbsMin/AbsMax, 그리고 더하기(i32 AddFxp/AddFxpSat, f32 Add). ReLU의 `max(x,0)`이 여기서 ClipMax로 구현됩니다.

**Filter** — `TagFilter`(GroupId MSB 매칭)로 만든 실행 마스크로 출력 flit을 거릅니다. 8-way + `Standalone` 컨텍스트에서만 가능 (intra-slice-chain.md:451-455).

**Output** — VE 파이프라인 출구. 결과는 Cast Engine, Transpose Engine, 또는 Commit Engine으로 갑니다 (intra-slice-chain.md:458-461).

### 단계 안의 단일-ALU 규칙 (꼭 외우세요)

**한 단계 안에서 각 ALU는 한 Tensor Unit invocation당 최대 한 번**만 돕니다 (intra-slice-chain.md:176). Logic/Fxp/Fp/Clip처럼 ALU 풀을 공유하는 단계에서 중요합니다. 예를 들어 `tanh(sqrt(x))`는 tanh도 sqrt도 둘 다 FpFpu를 쓰므로 한 invocation에 못 들어갑니다 (intra-slice-chain.md:178). 같은 이유로 AddFxp 두 번 연속은 런타임에 "FxpAdd is already in use" 패닉을 냅니다 (intra-slice-chain.md:252-262, 실제 예제 normal.rs의 `ve_elementwise_fxp_chain`).

### 피연산자(Operand)의 세 출처와 인자 모드

이항/삼항 연산에는 **스트림**(체인을 흘러가는 self 텐서)과 **피연산자**(추가 입력)가 있습니다. 피연산자는 세 곳에서 옵니다 (intra-slice-chain.md:82-91):
- **상수**: `100`, `2.5f32` — 모든 원소에 브로드캐스트.
- **VRF 텐서**: `VeRhs::vrf(&vrf)` — VE 진입 전 `.to_vrf()`로 미리 적재. (예제 normal.rs `ve_elementwise_vrf` / `ve_elementwise_multi_vrf`.)
- **Stash**: 체인 앞 단계의 스냅샷.

같은 메서드가 인자 타입으로 출처를 구분합니다 (intra-slice-chain.md:97-101):
```rust
.vector_fxp(FxpBinaryOp::AddFxp, 100)               // 상수
.vector_fxp(FxpBinaryOp::MulInt, VeRhs::vrf(&vrf))  // VRF
.vector_clip(ClipBinaryOpI32::Max, Stash)           // 스태시
```

**Stash**(`vector_stash()`)는 잔차/스킵 연결(`max(f(x), x)`처럼 원본 x를 나중에 다시 쓰기)에 씁니다 (intra-slice-chain.md:103-108). 규칙:
- 호출 가능한 단계는 Stashable한 곳: Branch, Logic, Fxp, Narrow, Fp, FpDiv, Clip.
- **단일 사용**(두 번째 `vector_stash()`는 컴파일 오류), **타입 고정**(i32 스태시는 i32 연산에만). 교차 타입은 런타임 패닉(예제 `ve_stash_fxp_fp`, `ve_stash_fp_fxp`).
- 매핑이 스트림을 따라가므로 Narrow 전에 스태시해도 Widen 후에 쓸 수 있습니다(예제 `ve_stash_across_narrow_widen`).
- **Pair Mode에서는 사용 불가**, 그리고 **IntraFirst로 inter-slice reducer로 넘어가면 스태시가 사라집니다**.

**인자 모드**는 어느 슬롯이 스트림이고 어느 게 피연산자인지 바꿉니다(unary는 모드 없음, 항상 op(stream)). `BinaryArgMode` (intra-slice-chain.md:121-126): Mode00=op(s,s), Mode01=op(s,operand)(기본), Mode10=op(operand,s), Mode11=op(operand,operand). 예: `vector_fxp_with_mode(SubFxp, Mode10, 7)`은 `7 - x`를 계산 (intra-slice-chain.md:113-114). `TernaryArgMode`도 7가지 슬롯 배치가 있습니다(Mode012가 기본, intra-slice-chain.md:130-138).

### Pair Mode (두 그룹 동시 처리)

입력 원소가 두 그룹으로 교차(interleave)되어 있을 때, 두 그룹을 관계 짓는 연산(쌍별 덧셈, 비대칭 스케일 등)을 할 수 있습니다 (intra-slice-chain.md:140-145). 흐름은 4단계 (intra-slice-chain.md:147-151):
1. `vector_intra_slice_unzip::<I, ...>()` — 2-way 그룹 축을 기준으로 group0/group1 두 스트림으로 분리(내부적으로 `TagMode::AxisToggle` 사용).
2. 두 그룹을 lock-step로 진행(paired phase).
3. `_zip` 연산으로 두 스트림을 하나로 융합(merged phase).
4. 이후는 보통 체인처럼 `vector_final()`까지.

paired phase의 연산 두 종류 (intra-slice-chain.md:153-158):
- **공통 단계**(`vector_fxp_to_fp`, `vector_narrow_split`, `vector_widen_concat`, `vector_fp_to_fxp`)는 두 그룹에 똑같이 적용. 단 `vector_narrow_clip`/`vector_widen_pad`는 pair에서 못 쓰고 `_split`/`_concat`를 써야 함.
- **그룹별 연산**: 이항/삼항은 그룹당 인자 하나씩 받고, 한쪽에 `()`를 주면 그쪽은 스킵(예제 `ve_group_pair_preprocess_g0`의 `vector_fxp(MulInt, 10, ())`). unary는 예외로 `(op, group0_apply, group1_apply)` 플래그(예: `vector_fp_unary(Exp, true, false)`).

`_zip` 연산의 BinaryArgMode는 의미가 달라집니다: 두 슬롯이 "두 그룹의 스트림"이 됩니다(0=group0, 1=group1). 예: `vector_fp_zip_with_mode(SubF, Mode10)`은 `group1 - group0` (intra-slice-chain.md:160-167). Pair Mode 제약 (intra-slice-chain.md:169-172): stash·filter 사용 불가, `_zip` 전(paired phase)에는 inter-slice reducer로 못 넘어감, ALU는 두 그룹이 공유(한쪽이 쓴 ALU는 양쪽 모두 소모로 침). 실제 예제는 furiosa-opt-examples/src/vector_engine/zip.rs 전체.

---

## 2. 연산 집합 한눈 정리 (op 모듈)

실제 op enum은 furiosa-opt-std/src/engine/vector/op/mod.rs에 정의돼 있습니다.
- `LogicBinaryOpI32`(BitAnd/BitOr/BitXor/LeftShift/LogicRightShift/ArithRightShift), `LogicBinaryOpF32`(BitAnd/BitOr/BitXor) (op/mod.rs:56, 133).
- `FxpBinaryOp`(AddFxp/AddFxpSat/SubFxp/SubFxpSat/LeftShift/LeftShiftSat/MulFxp/MulInt/LogicRightShift/ArithRightShift/ArithRightShiftRound) (op/mod.rs:203).
- `FpUnaryOp`(Exp/NegExp/Sqrt/Tanh/Sigmoid/Erf/Log/Sin/Cos) (op/mod.rs:323).
- `FpBinaryOp`(AddF/SubF/MulF(alu)/MaskMulF(alu)/DivF), `FpTernaryOp`(FmaF/MaskFmaF) (op/mod.rs:378, 415).
- `IntraSliceReduceOpI32`(AddSat/Max/Min), `IntraSliceReduceOpF32`(Add/Max/Min) (op/mod.rs:455, 474).
- `InterSliceReduceOpI32`(Add/AddSat/Max/Min), `InterSliceReduceOpF32`(Add/Max/Min/Mul) (op/mod.rs:497, 511).
- `FpDivBinaryOp`(DivF) (op/mod.rs:550).
- `ClipBinaryOpI32`(Min/Max/AbsMin/AbsMax/AddFxp/AddFxpSat), `ClipBinaryOpF32`(Min/Max/AbsMin/AbsMax/Add) (op/mod.rs:603, 675).
ALU 매핑은 각 enum의 `alu()` 메서드에 코드로 박혀 있으니, "어떤 연산이 어떤 ALU를 먹나"가 궁금하면 거기를 보면 정확합니다(예: op/mod.rs:261-268의 `FxpBinaryOp::alu`).

---

## 3. Intra-Slice Reduce — Time/Packet 축 접기

이 단계는 각 슬라이스 안의 `Time`과 `Packet`에 들어있는 차원을 줄입니다. `Chip/Cluster/Slice`는 그대로 통과합니다 (intra-slice-reduce.md:3). 256 슬라이스를 가로지르는 것은 다음 4장의 inter-slice reducer 몫입니다.

호출의 핵심 파라미터 (intra-slice-reduce.md:12-18):
- `REDUCE_LABEL`(예 `R`): 줄일 축. Time/Packet에서 이 축을 품은 모든 인수가 사라지므로 출력 shape(OutTime, OutPacket)에 나타나면 안 됩니다.
- `op`: `IntraSliceReduceOpI32`(AddSat/Max/Min) 또는 F32(Add/Max/Min).
- `OutTime, OutPacket`: 리덕션 후 Time/Packet shape(입력에서 REDUCE_LABEL 인수만 뺀 것).

시그니처는 `vector_intra_slice_reduce::<Reduce, OutTime, OutPacket>(op)`이고, Way4 상태에서만 호출됩니다(타입 바운드 `{ Way4 }`) (furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:1440). 그래서 예제들은 항상 reduce 전에 `vector_narrow_clip`/`split`을 거칩니다 (furiosa-opt-examples/src/vector_engine/reduce.rs:18-19).

R이 어디에 놓이느냐에 따라 동작이 달라집니다 (intra-slice-reduce.md:21-96):
- **Time에만**: 시간 단계을 가로질러 누적(temporal accumulation).
- **Packet에만**: 한 flit 안에서 4-way 트리 리듀스 `op(op(a,b), op(c,d))`.
- **둘 다**: Packet 부분은 flit 내 트리, Time 부분은 시간 누적.
- **Slice 부분이 남으면**: intra-slice는 Time/Packet만 접고 Slice 부분은 출력에 남습니다(per-slice reduction). 이때 패딩 제외는 VCG가 처리(아래).

### 아키텍처 제약 (왜 reduce가 거부될 수 있나)

Time 리덕션은 contraction의 temporal accumulator 모델을 쓰며 **슬롯 용량이 8**입니다(`InnerTime::SIZE ≤ 8`) (intra-slice-reduce.md:131-133). `InnerTime`(= OutTime, 즉 리듀스 후 살아남는 안쪽 Time 인수의 곱)이 8을 넘으면 API가 거부합니다. 예: OutTime이 `m![A%3, B%4]`라 3×4=12개 슬롯이 필요하면 "8개뿐"이라 거부 (intra-slice-reduce.md:135-155).

Packet 리덕션은 4개 원소가 두 길 중 하나 (intra-slice-reduce.md:160-164):
- `OutPacket = Packet`: 안 줄임. 사이클당 4출력, 각자 Time 누적.
- `OutPacket = m![1 # 4]`: 2단 트리로 한 값으로 접음. 사이클당 1출력 + 패딩 3.

### 패딩 처리 두 전략 (정확도의 핵심)

리듀스 축을 하드웨어 크기에 맞추려 패딩하면, 패딩 칸엔 쓰레기 값이 들어가서 리덕션에서 빼야 합니다 (intra-slice-reduce.md:166-186):
1. **VCG(Valid Count Generator)** — 축 배치가 지원되면 우선. 컴파일러가 매핑에서 VCG를 자동 설정, 각 flit에 `valid_count`를 태깅해 패딩을 자동 제외(5장).
2. **Identity-element 패딩** — 패딩 칸을 리듀스 연산의 항등원으로 미리 채움(Fetch Engine의 masking이 적재 시 기록). 항등원: AddSat/Add→0, Max→i32::MIN/f32::NEG_INFINITY, Min→i32::MAX/f32::INFINITY (intra-slice-reduce.md:179-183). **단, 리듀스 앞에 비가역 변환이 없어야 함**. softmax의 `exp(x)+exp(y)+...`처럼 합의 항등원 0을 만드는 p(즉 exp(p)=0)가 없으면 이 전략은 못 씁니다 → VCG가 필요해집니다 (intra-slice-reduce.md:184-185). 이게 실무에서 자주 물리는 함정입니다.

성능 (intra-slice-reduce.md:188-193): 처리량은 사이클당 1 flit 유지(트리 리듀스가 완전 파이프라인). 다만 첫 출력까지 리듀스 축의 시간 단계 수 n만큼 지연(누적 그룹의 모든 flit을 모아야 결과가 나오므로). 멀티 엔진 파이프라인에선 이 지연이 하류 엔진을 멈출 수 있습니다.

---

## 4. Inter-Slice Reducer — 256 슬라이스 가로지르기

한 클러스터의 256개 슬라이스를 가로질러 리덕션합니다 (inter-slice-reducer.md:3-6). `Chip/Cluster/Packet`은 보존하고 `Slice/Time`을 `OutSlice/OutTime`으로 다시 씁니다. **출력은 입력 모드와 무관하게 항상 Way8**입니다.

진입 두 경로 (inter-slice-reducer.md:9): `vector_init()` 직후(InterFirst), 또는 호환되는 intra-slice 단계에서(IntraFirst). 같은 `vector_inter_slice_reduce()` 메서드라 호출 모양은 동일합니다. i32/f32 API가 따로 있습니다(시그니처: vector_tensor.rs:640, 661). 연산: i32 Add/AddSat/Max/Min, f32 Add/Max/Min/Mul (inter-slice-reducer.md:20-42).

### 4가지 shape 규칙 (Slice→OutSlice, Time→OutTime)

(inter-slice-reducer.md:46-51)
1. **안쪽부터 리듀스**: Slice의 리듀스 부분은 가장 안쪽 인수들이어야 하고, 연속적이며 stride 1로 리덕션 비율 r까지.
2. **리듀스된 축 슬롯의 대체**: 비워진 슬롯은 셋 중 하나로 채움 — 더미(`1 # n`), 새 차원으로 브로드캐스트, 또는 Time에서 끌어올림(promotion).
3. **대체 종류 자유 혼합**: 더미·브로드캐스트·승격 슬롯이 OutSlice 안에서 임의 순서로 섞여도 됨.
4. **Time→OutSlice 승격은 순서를 깸**: Time→OutTime 경로는 살아남는 인수의 상대 순서를 보존하지만, Time→OutSlice 승격은 위치가 독립적(승격된 인수의 OutSlice 위치는 Time에서의 위치와 무관).

예시 셋 (einsum 표기):
- **더미 대체** `AR -> A`: R 합치고 빈 슬롯을 `1 # 4` 더미로 (inter-slice-reducer.md:72-80, 예제 reduce.rs `ve_inter_slice_reduce_add_sat_i32`).
- **새 차원 브로드캐스트** `PRW -> PWX`: R 합치고 결과를 새 X에 브로드캐스트 (inter-slice-reducer.md:101-108).
- **Time 승격** `PRSUVW -> PSUVW`: R 합치며 Time의 V를 OutSlice로 끌어올림 (inter-slice-reducer.md:130-137, 예제 `ve_inter_slice_reduce_promote_f32`).

성능 (inter-slice-reducer.md:148-155): 핵심 손잡이는 리덕션 비율 r(한 리덕션 그룹의 슬라이스 수). 지연은 O(r) 사이클(리덕션 그룹 링을 한 바퀴). 보통 상류 작업(contraction의 부분합 생성, reducer 앞의 intra-slice 작업)이 이 링 꼬리를 가려서 병목이 아닙니다. r이 크거나 작은 텐서일 때 꼬리가 드러납니다.

### IntraFirst/InterFirst 실제 예시
- IntraFirst(체인→리듀서): ReLU 후 슬라이스 간 합 (index.md:75-86).
- InterFirst(리듀서→체인): 슬라이스 간 합 후 바이어스 더하기 (index.md:102-113, 예제 reduce.rs `ve_vru_then_vau_i32`).
- 둘 다 써서 R 전체 접기: intra가 R의 Time/Packet, inter가 R의 Slice를 접음 (index.md:130-145).

---

## 5. VCG (Valid Count Generator) — 패딩 자동 제외 장치

intra-slice reduce가 R을 Time/Packet에서 접을 때, R을 하드웨어 정렬 크기로 패딩하면 패딩 칸의 쓰레기를 빼야 합니다. VCG가 이걸 자동화합니다 (vcg.md:1-11). 사용자는 R을 `Slice/Time/Packet`에 부분식(sub-expression)으로 배치하고, 컴파일러가 VCG를 구성해 각 8-원소 flit에 "진짜 데이터가 몇 개인지"(`valid_size`)를 태깅합니다. Time/Slice의 부분식은 sequencer 카운터→time filter로, Packet의 부분식은 packet clipper로 갑니다 (vcg.md:8-11).

R은 `R # PADDED_SIZE` 형태로 패딩되고, 각 부분식은 `R # PADDED_SIZE / n % m`(stride n, modulo m) 꼴입니다 (vcg.md:14-19). 예: R=43을 R#48로 패딩해 세 차원에 분배 (vcg.md:20-26).

### 구조

`valid_size(s, t) ∈ {0..8}`을 각 flit에 부여(s=슬라이스 id, t=시간 단계). 처음 valid_size개가 진짜, 나머지는 패딩 (vcg.md:30). 의사코드 (vcg.md:33-48): time filter 3개가 전부 valid라고 하면 packet clipper가 개수를 정하고, 하나라도 invalid면 그 flit 전체를 제외.

**Time Filter** (vcg.md:54-83) — 각 슬라이스 s에서 각 시간 단계 t가 valid R 데이터를 들었는지 판정. 필드: sequencer(t→R index 복원), slice_mask/slice_thres/time_thres, mode(SliceMajor|TimeMajor). `valid()` 로직: `(s & slice_mask)`를 slice_thres와 비교해 Less면 무조건 true, (Greater & SliceMajor)면 false, 그 외엔 `idx < time_thres`. R이 Time/Slice에 부분식이 없으면 slice_mask=0, slice_thres=1로 비활성(항상 true) (vcg.md:58).

배치하는 경우들:
- **R as Time**: t가 곧 R index. `t < R::SIZE`면 valid (vcg.md:85-127). 예제처럼 R#16에서 R::SIZE=12면 t=0..11 valid, 12..15 제외.
- **R in Time(다른 축과 공유, 순서 무관)**: sequencer가 t를 부분식별 카운터로 분해, R 할당 카운터의 `value×stride` 합이 idx (vcg.md:128-180). 비-R 축(A)은 stride 0이라 idx에 기여 안 함 → 위치 무관.
- **R in Slice and Time (SliceMajor)** (vcg.md:181-253): Slice 부분식이 더 major(큰 stride). Slice 안에서는 stride 내림차순, 각 부분식은 2의 거듭제곱 크기·stride여야 slice_mask 비트가 연속. Time 안에서는 순서 자유. 슬라이스를 "전부 valid / 경계 / 전부 invalid" 세 영역으로 나눔. slice_thres가 경계 슬라이스의 R 기여 인코딩, time_thres = R::SIZE - 경계기여.
- **R in Time and Slice (TimeMajor)** (vcg.md:255-332): SliceMajor의 쌍대(major/minor 역전). 추가 제약 **`PADDED_SIZE - R::SIZE ≤ slice_span`**(slice_span = Slice 부분식 크기 곱). 즉 과패딩 칸이 slice_span개 이하여야 함. TimeMajor엔 "전부 invalid" 영역이 없음(패딩이 빡빡해서).

**Packet Clipper** (vcg.md:334-355) — 각 flit의 `valid_size(t)`를 t만으로 계산(슬라이스 s 무관, 모든 슬라이스가 같은 t에 같은 개수). `valid_size(t) = clamp(axis_size - idx(t), 0, packet_span)`. 요구 형태: **`Packet = m![R # PADDED_SIZE % packet_span # 8]`** (vcg.md:357). 다른 축이 Packet을 R과 공유하거나 R이 Packet에서 여러 부분식으로 쪼개지면 contiguous-prefix 성질이 깨져 안 됨.
- **R as Packet**(R::SIZE ≤ 8): 모든 flit이 valid_size=R::SIZE (vcg.md:361-399).
- **R in Time and Packet**: sequencer가 idx(t) 복원, 톱니 모양 valid_size(예: 4,4,2,0) (vcg.md:401-455).

여러 축 동시 추적 (vcg.md:458-587): intra-slice reduce는 REDUCE_LABEL 하나 + 추가 패딩 비-R 축들을 받고, VCG가 모두 추적. 각 패딩 축은 time filter 슬롯 1개 또는 packet clipper를 차지. `[H,C,W]=[5,5,19]`를 단계적으로 쌓아 16슬라이스×12flit 히트맵으로 검증하는 예시가 자세합니다.

**표현 불가 패턴들** (vcg.md:589-788) — 외워두면 디버깅에 도움:
- R이 Slice에서 stride 역순(major가 안쪽) → 비단조 → 단일 slice_thres로 못 잡음.
- R의 Slice 인수가 두 Time 인수 사이에 끼임(interleave) → 슬라이스마다 valid 단계 수가 달라짐.
- TimeMajor 과패딩(`PADDED_SIZE - R::SIZE > slice_span`) → below 그룹이 패딩을 몰래 포함.
- Packet에 R의 major부(`/8` 형태)나 R과 다른 축 공유 → prefix 성질 깨짐.
- R이 Slice와 Packet에 걸침 → 슬라이스마다 valid_size가 달라야 하는데 packet clipper는 t로만 정함(예: R=2045, 슬라이스 0~254는 8, 255는 5 필요). 단 `R::SIZE % packet_span = 0`인 퇴화 경우는 Slice-only로 환원돼 지원.

**용량 제약** (vcg.md:790-802): packet clipper 1개, time filter 3개, 각 sequencer 엔트리 8개. 한 invocation에 최대 4축(1 packet clipper + 3 time filter) 추적. 패딩 없는 축은 슬롯 불필요.

**하류 4-way 변환** (vcg.md:805-820): VCG는 narrow 전 8-way flit에 valid_size를 만들고, 이후 Narrow가 어떻게 쪼개냐에 따라 valid가 나뉨. split_way4: low=min(v,4), high=max(v-4,0). trim_way4: min(v,4)(단 매핑이 v≤4를 정적 보장해야 함, 아니면 데이터 손실). concat_way8: v_low+v_high. pad_way8: 그대로.

---

## 6. 이것들로 softmax/layernorm/활성화 만들기

- **ReLU**: `vector_clip(Max, 0)` 한 방 (index.md:50-59).
- **Sigmoid/Tanh/Erf/...**: Narrow → `vector_fp_unary(...)` → Widen (normal.rs `ve_elementwise_fp_unary`, intra-slice-chain.md:498-508).
- **SiLU/Swish (x·sigmoid(x))**: `vector_stash()`로 x 저장 → narrow → sigmoid → widen → `vector_fp_binary(MulF, Stash)`. 잔차 패턴(`max(sigmoid(x), x)`) 예제가 그대로 본보기 (intra-slice-chain.md:609-628).
- **GELU**: erf 기반이면 `FpUnaryOp::Erf` + Fma/Mul 조합, tanh 근사면 `Tanh` + 다항(Mul0/Mul1/Fma로 ALU 분산).
- **softmax** (한 축 R에 대해): ① max-reduce로 안정화 상수 m 구함 → ② x-m → ③ exp → ④ sum-reduce → ⑤ 나눗셈. 핵심 주의: exp 뒤의 sum은 비가역이라 **identity 패딩이 안 통하고 VCG가 필요**합니다 (intra-slice-reduce.md:184-185). 축이 슬라이스에 걸리면 inter-slice reducer로 합을 모으고, 슬라이스 내부면 intra-slice reduce + VCG.
- **layer norm**: ① mean = sum-reduce / N → ② x-mean → ③ 제곱(MulF) → ④ var = sum-reduce / N → ⑤ `sqrt(var+eps)` (FpFpu) → ⑥ 나눗셈(`vector_fp_div`로 전용 divider 쓰면 sqrt와 ALU 안 겹침) → ⑦ 스케일·시프트(Fma). reduce 예제들(reduce.rs)이 합/최대/최소의 실동작을 보여줍니다.

---

## 7. Cast Engine — 정밀도 좁히기

VE 결과(f32/i32)를 Commit이 DM에 쓰기 전에 더 좁은 타입(bf16 등)으로 변환해 저장 비용을 줄입니다 (docs/src/computing-tensors/cast-engine.md:1-3). `CollectTensor`, `ContractTensor`, `VectorFinalTensor`가 모두 같은 의미의 `.cast()`를 노출합니다 (cast-engine.md:7).

시그니처: `.cast::<OutD, OutPacket>()` (furiosa-opt-std/src/engine/cast.rs:38). 입력은 반드시 **정확히 한 32바이트 flit**이어야 하고(`VeScalar` 즉 i32/f32만 입력 가능, cast.rs:32), 출력도 다시 32바이트 flit로 패딩됩니다. shape는 그대로 통과하고 Packet 레이아웃만 다시 패딩됩니다 (cast-engine.md:16-18). 예: i32 8원소(32B)를 i8로 캐스트하면 8B가 되므로 `A # 32`로 다시 32B 패딩 (cast-engine.md:23-33).

합법 캐스트 (cast-engine.md:54-61, 구현은 cast.rs):
- i32 → i4, i8, i16
- f32 → f8e5m2, f8e4m3, f16, bf16

성능: 절대 병목이 아님. flit당 1사이클(유효 데이터 양 무관). 하류 Commit이 덜 찬 flit을 모아 dense하게 써서 DM 대역폭 낭비 없음 (cast-engine.md:64-66).

검증 로직(`verify_cast`, cast.rs:54-83)은 입력 flit=32B, 출력 flit=32B, 데이터 항(패딩 제외)이 일치하는지를 컴파일/런타임에 확인합니다.

---

## 8. Transpose Engine — Time↔Packet 맞바꾸기

`Time`과 `Packet` 차원을 맞바꾸고 `Chip/Cluster/Slice`는 그대로 둡니다 (docs/src/computing-tensors/transpose-engine.md:3). `CollectTensor`와 `VectorFinalTensor`가 `.transpose::<OutTime, OutPacket>()`를 노출(후자는 VE 출력에서 바로) (transpose-engine.md:7-8, furiosa-opt-std/src/engine/transpose.rs:50).

### 파라미터와 "flit 내부" 제약

- `valid_size`: 32바이트 입력 버스에서 사이클당 읽는 유효 원소 수. 각 입력 flit은 `bit-width × valid_size` 형태로 오고 나머지 바이트는 Unpack이 버림 (transpose-engine.md:38).
- `in_cols`, `in_rows`, `out_rows`는 작성자의 OutTime/OutPacket 선택으로 정해집니다.

원소 크기별 제약표 (transpose-engine.md:45-50):

| 원소 크기 | valid_size | Max in_rows | 허용 in_cols |
|----------|-----------|-------------|--------------|
| 4-bit | 16 | 16 | 16, 32 |
| 8-bit | 8 | 8 | 8, 16, 32 |
| 16-bit | 8 | 4 | 8, 16, 32 |
| 32-bit | 8 | 2 | 8, 16, 32 |

여기서 **flit 내부 제약**의 본질은 코드의 `in_rows × sizeof(D) ≤ 8 바이트`입니다 (transpose.rs:85, 상수 `TRANSPOSE_MAX_IN_ROWS_BYTES = 8`). 즉 전치된 한 열(=출력 패킷의 in_rows)은 8바이트(=하나의 8-lane 묶음) 안에 들어와야 합니다. 그래서 원소가 넓어질수록 max in_rows가 8→4→2로 줄고, 4-bit는 16까지 늘어납니다. 추가로 Packet·OutPacket은 둘 다 32바이트여야 하고(transpose.rs:69-79), `out_rows ≤ in_cols`여야 합니다(transpose.rs:139). Contraction Engine은 `32b × 8`만 내보내고, VE와 Fetch Engine은 위 표의 임의 조합을 내보냅니다 (transpose-engine.md:40).

### 4단계 (running 예: i8 8×8)

(transpose-engine.md:71-98)
1. **Unpack**: 각 32B 패킷에서 valid_size개만 남기고 패딩 버림. 시간 단계들이 in_cols 폭의 한 행을 이루고, in_rows 행이 쌓여 `[in_rows × in_cols]` 행렬. `[D, E#32] → [D, E]`.
2. **Transpose**: `[in_rows × in_cols] → [in_cols × in_rows]`. `[D, E] → [E, D]`.
3. **Trim**: 일부 입력 패킷이 valid_size보다 적게 들었으면 전치 후 패딩 행이 생기는데 이를 버려 `[out_rows × in_rows]`(out_rows ≤ in_cols). 완전 활용이면 안 버림. (Small Matrix 예제 transpose-engine.md:110-142에서 6행 trim.)
4. **Align**: 전치된 행은 in_rows 폭이라 32B로 패딩 → `[out_rows × (in_rows # F)]`, F는 D[F]=32B가 되게.

### 버퍼링과 지연

내부 버퍼 2개, 각 16열 보유 (transpose-engine.md:288). `in_cols ≤ 16`이면 이중 버퍼링(입출력 겹침으로 사이클 단축), 초과면 싱글 버퍼링(직렬화) (transpose-engine.md:289).
- 버스트는 `n = OutTime::SIZE / out_rows`번 반복, 각 반복은 `in_flits = in_rows × (in_cols / valid_size)` 입력 flit과 out_rows 출력 flit을 옮김 (transpose-engine.md:283-284).
- 싱글: `n × (in_flits + out_rows)` (transpose-engine.md:291-293).
- 더블: `in_flits + (n-1) × max(in_flits, out_rows) + out_rows`(입력채움 + 겹침 + 마지막드레인) (transpose-engine.md:296-303).
running 예(i8 8×8)는 `8 + 7×max(8,8) + 8 = 72` 사이클 (transpose-engine.md:105-106). bf16(transpose-engine.md:178-210), i4(212-244), f32(246-279) 예제가 원소 크기 효과를 보여줍니다.

검증 로직 `verify_transpose`(transpose.rs:66-156)와 그 안의 단위 테스트(transpose.rs:158-218 `mod tests::valid` — basic/small/small_no_slicing/large_col/bf16)가 NPU 없이 shape 제약을 그대로 검사합니다.

## 2. 핵심 API · 패턴

| 이름 | 쓰는 법 | 설명 | 출처 |
|---|---|---|---|
| `vector_init` | `.vector_init() -> VectorInitTensor<...>` | VE 진입점. CollectTensor/ContractTensor 등에서 호출. 반드시 vector_final()로 끝난다. | `furiosa-opt-std/src/engine/vector/tensor/mod.rs:89` |
| `vector_intra_slice_tag` | `.vector_intra_slice_tag(branch: TagMode) -> VectorBranchTensor<...,{IntraFirst}>` | 체인을 단일 스트림으로 시작. TagMode: Zero/AxisToggle/Comparison/ValidCount/Vrf. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:613` |
| `vector_intra_slice_unzip` | `.vector_intra_slice_unzip::<I: AxisName, TileTime, SplitTime>() -> VectorTensorPair<...>` | Pair Mode 진입. 2-way 그룹 축 I로 group0/group1 분리(내부 TagMode::AxisToggle). | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:624` |
| `vector_inter_slice_reduce` | `.vector_inter_slice_reduce::<OutSlice, OutTime>(op: InterSliceReduceOpI32\|F32) -> VectorInterSliceReduceTensor<...>` | 256 슬라이스 리듀스. vector_init 직후(InterFirst) 또는 호환 intra 단계(IntraFirst)에서 동일 메서드명. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:640,661` |
| `vector_intra_slice_reduce` | `.vector_intra_slice_reduce::<Reduce: AxisName, OutTime, OutPacket>(op: IntraSliceReduceOpI32\|F32)` | Time/Packet 축 접기. Way4 상태에서만 호출 가능(앞에 narrow 필요). i32: AddSat/Max/Min, f32: Add/Max/Min. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:1440` |
| `vector_narrow_clip / vector_narrow_split` | `.vector_narrow_clip::<Packet2>() \| .vector_narrow_split::<Time2, Packet2>()` | Way8→Way4. clip은 뒤4가 패딩일 때 앞4만, split은 앞4/뒤4 둘 다 데이터. Pair에선 _clip/_pad 금지, _split/_concat 사용. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:1215,1258` |
| `vector_widen_pad / vector_widen_concat` | `.vector_widen_pad::<Packet2>() \| .vector_widen_concat::<Time2, Packet2>()` | Way4→Way8. narrow_clip/narrow_split의 각 역연산. final/inter_slice_reduce 전엔 반드시 8-way로 복귀. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:1590,1633` |
| `vector_fxp / vector_fxp_with_mode` | `.vector_fxp(FxpBinaryOp, operand) \| .vector_fxp_with_mode(op, BinaryArgMode, operand)` | i32 정수/고정소수. operand는 상수/VeRhs::vrf/Stash. Mode10이면 operand-stream. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:1130,1140` |
| `vector_fp_unary/binary/ternary` | `.vector_fp_unary(FpUnaryOp) / .vector_fp_binary(FpBinaryOp, op) / .vector_fp_ternary(FpTernaryOp, (op0,op1))` | Way4 float 단계. MulF는 FpMulAlu(Mul0/Mul1/Fma)로 ALU 선택. Pair에선 unary가 (op, g0, g1) 플래그. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:1295,1307,1340` |
| `vector_fxp_to_fp / vector_fp_to_fxp` | `.vector_fxp_to_fp(int_width) / .vector_fp_to_fxp(int_width)` | i32↔f32 변환. int_width=31이 표준 i32↔f32. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:1173,1670` |
| `vector_clip / vector_clip_with_mode` | `.vector_clip(ClipBinaryOpI32\|F32, operand)` | 8-way 클램핑/비교. ReLU=Max(x,0). ALU: ClipAdd/ClipMax/ClipMin. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:1706` |
| `vector_fp_div / vector_fp_div_with_mode` | `.vector_fp_div(operand) \| .vector_fp_div_with_mode(BinaryArgMode, operand)` | 전용 부동소수 divider(ReduceFpDiv). Fp단계의 DivF(FpFpu)와 달리 sqrt 등과 ALU 안 겹침. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:1544,1553` |
| `vector_stash` | `.vector_stash() -> (same stage)` | 러닝 텐서 스냅샷. 단일 사용·타입 고정. Stashable 단계(Branch/Logic/Fxp/Narrow/Fp/FpDiv/Clip)에서만. Pair 불가, IntraFirst inter 전이 시 소멸. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:415, docs/src/computing-tensors/vector-engine/intra-slice-chain.md:103-108` |
| `vector_filter` | `.vector_filter::<Time2>() ` | TagFilter(GroupId MSB)로 출력 flit 마스킹. 8-way + Standalone 컨텍스트에서만, Pair Mode 불가. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:1792` |
| `_zip 연산군` | `.vector_clip_zip(op) / .vector_fxp_zip(op) / .vector_fp_zip(op) / .vector_logic_zip(op) / *_zip_with_mode(op, BinaryArgMode)` | Pair Mode의 두 그룹 스트림 융합. _zip의 ArgMode는 두 슬롯이 group0/group1 스트림. | `furiosa-opt-examples/src/vector_engine/zip.rs:19,138,161,186` |
| `cast` | `.cast::<OutD: Scalar, OutPacket: M>() -> CastTensor<...>` | 입력 1 flit(32B)→OutD로 변환 후 32B 재패딩. 입력은 VeScalar(i32/f32)만. i32→i4/i8/i16, f32→f8e5m2/f8e4m3/f16/bf16. | `furiosa-opt-std/src/engine/cast.rs:38` |
| `transpose` | `.transpose::<OutTime: M, OutPacket: M>() -> TransposeTensor<...>` | Time↔Packet 교환. Packet/OutPacket=32B, in_rows×sizeof(D)≤8B, in_cols∈{8,16,32}, out_rows≤in_cols. | `furiosa-opt-std/src/engine/transpose.rs:50` |

## 3. 실험 (직접 돌리기)

> 실험은 NPU 없이 `simulation`·`typecheck`로 돌아갑니다. 실행법은 [`../experiments/README.md`](../experiments/README.md), MNIST는 `cargo furiosa-opt test`(npu 전용).

### 실험 07.1 — VE 풀 파이프라인을 시뮬레이션으로 돌려보기 (fxp→fp→clip)
*난이도 1/5 · 기반: `furiosa-opt-examples/tests/vector_engine/normal.rs`*

**목표** — vector_init→tag→fxp(AddFxp)→fxp_to_fp→narrow→fp(MulF)→widen→fp_to_fxp→clip(Max,Min)→final 한 사이클의 결과가 호스트 레퍼런스와 일치하는지 확인하며 각 단계가 어떻게 이어지는지 본다.

```bash
cargo furiosa-opt test -p furiosa-opt-examples --test vector_engine_tests test_ve_elementwise_full_pipeline -- --nocapture
```
**관찰** — 테스트 통과(green). 레퍼런스는 clamp(((x+100) as f32 * 2.5).round() as i32, 0, 1000). 커널 정의는 normal.rs:48의 ve_elementwise_full_pipeline. narrow_clip→fp→widen_pad가 float 경로를 감싸는 모양을 코드에서 확인.

**심화** — normal.rs의 ve_elementwise_full_pipeline에서 MulF 상수 2.5f32를 3.0f32로 바꾸고 test의 expected도 3.0으로 고쳐 다시 통과시켜보기.

### 실험 07.2 — 단일-ALU 규칙 위반을 '예측-후-실행'으로 확인
*난이도 2/5 · 기반: `furiosa-opt-examples/src/vector_engine/normal.rs`*

**목표** — 같은 FxpAdd ALU를 쓰는 AddFxp와 SubFxp를 한 invocation에 넣으면 왜 패닉하는지 체득.

```bash
cargo furiosa-opt test -p furiosa-opt-examples --test vector_engine_tests test_ve_elementwise_fxp_chain -- --nocapture
```
**관찰** — 테스트는 catch_unwind로 패닉을 기대하므로 '통과'한다. 커널(normal.rs:28 ve_elementwise_fxp_chain)은 AddFxp→MulInt→SubFxp 순인데 AddFxp와 SubFxp가 둘 다 FxpAdd라 'FxpAdd is already in use'로 패닉. 패닉 메시지를 로그에서 확인.

**심화** — 중간 SubFxp를 LeftShift(FxpLshift ALU)로 바꾸면 충돌이 사라져 패닉이 안 나는지 추론해보기(테스트는 패닉을 기대하므로 그땐 실패하게 됨).

### 실험 07.3 — intra-slice reduce vs inter-slice reduce 비교
*난이도 2/5 · 기반: `furiosa-opt-examples/tests/vector_engine/reduce.rs`*

**목표** — 같은 합(saturating add)이라도 축이 Time/Packet에 있을 때(intra)와 Slice에 걸칠 때(inter) API와 매핑이 어떻게 달라지는지 본다.

```bash
cargo furiosa-opt test -p furiosa-opt-examples --test vector_engine_tests test_ve_intra_slice_reduce_add_fxp_sat test_ve_inter_slice_reduce_add_sat_i32 -- --nocapture
```
**관찰** — 둘 다 통과, 결과는 동일한 R축 saturating add. 그러나 intra(reduce.rs:4)는 narrow_clip 후 vector_intra_slice_reduce::<R, m![1], m![A%2#4]>, inter(reduce.rs:220)는 vector_inter_slice_reduce::<m![A/8, 1#4], m![1]>로 R이 Slice에 놓여 더미 1#4로 대체됨을 코드에서 대조.

**심화** — AddSat을 Max로 바꾼 두 변형(test_ve_intra_slice_reduce_max_i32, test_ve_inter_slice_reduce_max_i32)도 돌려 항등원이 i32::MIN으로 바뀌는 레퍼런스를 확인.

### 실험 07.4 — InterFirst 순서(리듀서→체인) 동작 확인
*난이도 2/5 · 기반: `furiosa-opt-examples/src/vector_engine/reduce.rs`*

**목표** — vector_init 직후 inter_slice_reduce를 부르고 그 출력에서 intra_slice_tag로 체인을 이어 +100 하는 InterFirst 경로를 실행.

```bash
cargo furiosa-opt test -p furiosa-opt-examples --test vector_engine_tests test_ve_vru_then_vau_i32 -- --nocapture
```
**관찰** — 통과. 레퍼런스는 R축 saturating_add 후 +100. 커널(reduce.rs:288)에서 vector_inter_slice_reduce 다음에 vector_intra_slice_tag→vector_fxp(AddFxp,100)이 오는 것이 InterFirst.

**심화** — vector_fxp(AddFxp,100)을 vector_clip(Max,0)으로 바꾸면 어떤 결과가 나올지 손으로 예측(음수 합이 0으로 클램프).

### 실험 07.5 — Pair Mode zip 실행 (두 입력을 교차해 쌍별 연산)
*난이도 3/5 · 기반: `furiosa-opt-examples/tests/vector_engine/zip.rs`*

**목표** — begin_interleaved + vector_intra_slice_unzip + _zip 으로 두 텐서를 합치는 Pair Mode 전체 흐름을 본다.

```bash
cargo furiosa-opt test -p furiosa-opt-examples --test vector_engine_tests test_ve_group_pair_add test_ve_group_pair_preprocess_g0 -- --nocapture
```
**관찰** — 통과. add(zip.rs:4)는 unzip→clip_zip(AddFxp)으로 lhs+rhs. preprocess_g0(zip.rs:51)는 vector_fxp(MulInt, 10, ())로 group0만 ×10 후 zip → (lhs*10)+rhs. ()가 한쪽 스킵임을 코드에서 확인.

**심화** — src/vector_engine/zip.rs의 ve_group_pair_preprocess_g0에서 (10, ())를 ((), 10)으로 바꾸면 결과가 lhs+(rhs*10)이 되는지 예측하고, 그건 preprocess_g1 테스트와 같아짐을 확인.

### 실험 07.6 — Transpose 'flit 내부' 제약을 typecheck 단위테스트로 깨보기
*난이도 3/5 · 기반: `furiosa-opt-std/src/engine/transpose.rs`*

**목표** — in_rows × sizeof(D) ≤ 8바이트 제약이 컴파일/검증 단계에서 어떻게 걸리는지, NPU 없이 verify_transpose로 확인.

```bash
cargo furiosa-opt test -p furiosa-opt-std transpose::tests::valid -- --nocapture
```
**관찰** — basic/small/large_col/bf16 단위테스트 통과. 각 경우의 in_cols/in_rows/out_rows가 표(transpose-engine.md:45-50)와 일치. verify_transpose(transpose.rs:66)가 Packet=32B, in_rows≤8B, in_cols∈{8,16,32}, out_rows≤in_cols를 assert로 검사.

**심화** — transpose.rs의 valid::basic을 복사해 i8 대신 f32로 같은 in_rows=8을 주면(8×4=32B>8B) assert가 터지는지 손으로 예측(원소가 넓어지면 max in_rows가 2로 줄기 때문).

### 실험 07.7 — 활성화 직접 만들기: 풀 파이프라인에 sigmoid 끼우기
*난이도 4/5 · 기반: `furiosa-opt-examples/src/vector_engine/normal.rs`*

**목표** — float 경로에 fp_unary(Sigmoid)를 추가하고 ALU 경합(FpFpu 단일 사용) 규칙을 지켜 새 커널을 작성·검증.

```bash
# normal.rs에 새 커널 ve_sigmoid_then_relu 추가 후
cargo furiosa-opt test -p furiosa-opt-examples --test vector_engine_tests -- --nocapture
```
**관찰** — tag→narrow_clip→fp_unary(Sigmoid)→widen_pad→clip(Max,0) 체인이 컴파일되고 호스트 레퍼런스 max(sigmoid(x),0)와 일치. Sigmoid는 FpFpu라 sqrt/tanh를 같은 invocation에 또 넣으면 안 됨을 확인.

**심화** — 같은 커널에서 Sigmoid 뒤에 Tanh도 추가해보고 둘 다 FpFpu라 단일-ALU 규칙으로 막히는지(컴파일/런타임) 확인.

## 4. 연습문제 (손으로, 컴파일 없이)

**Q1.** 다음 체인이 컴파일될까? `.vector_intra_slice_tag(Zero).vector_fp_unary(FpUnaryOp::Sigmoid).vector_final()` (narrow/widen 없음). 안 되면 무엇을 추가해야 하나?

<details><summary>정답/힌트</summary>

안 됨. Float 단계는 Way4 상태에서만 호출되므로 fp_unary 앞에 vector_narrow_clip/split, 뒤에 vector_widen_pad/concat이 필요하다(final은 8-way 요구).

</details>

**Q2.** i32 데이터에 TagMode::Comparison([Greater{10}, Less{0}, Equal{3}, False])를 적용했을 때 x=12의 4비트 태그(LSB first)와 정수 tag 값은?

<details><summary>정답/힌트</summary>

비트i = cmp_i(12): Greater{10}=1, Less{0}=0, Equal{3}=0, False=0 → 1/0/0/0 → tag = 0b0001 = 1.

</details>

**Q3.** `exp(x) + exp(y) + ...` 형태의 sum-reduce에서 패딩 칸을 identity-element(0)로 채우는 전략이 왜 실패하나? 대안은?

<details><summary>정답/힌트</summary>

덧셈 항등원은 0인데 exp(p)=0을 만드는 입력 p가 없어 패딩 자리에 0을 만들 수 없다. 비가역 변환(exp)이 reduce 앞에 있어 불가 → VCG로 valid_size 태깅해 제외해야 한다.

</details>

**Q4.** intra-slice reduce에서 Time=m![R, A%3, B%4], OutTime=m![A%3, B%4]일 때 필요한 누적 슬롯 수와 허용 여부는?

<details><summary>정답/힌트</summary>

InnerTime=OutTime의 곱 = 3×4 = 12 > 8 → 거부됨(슬롯 8개 한도 초과).

</details>

**Q5.** inter-slice reduce로 einsum `AR -> A`를 할 때, 리듀스된 R 슬롯은 OutSlice에서 어떻게 채워지나? 세 가지 대체 방식 중 무엇인가?

<details><summary>정답/힌트</summary>

다른 차원이 안 채우면 더미 1#n으로 채움(dummy replacement). 한 위치만 리듀스 값, 나머지 n-1은 패딩. 예: m![A/8, 1 # 4].

</details>

**Q6.** f32 8×8 행렬을 transpose할 때 max in_rows는? i8 대비 왜 달라지나?

<details><summary>정답/힌트</summary>

f32는 32비트라 max in_rows=2(2×4=8B). i8은 max 8. in_rows×sizeof(D)≤8B 제약 때문에 원소가 넓을수록 한 열에 담을 수 있는 행 수가 준다.

</details>

**Q7.** `vector_fp_zip_with_mode(FpBinaryOp::SubF, BinaryArgMode::Mode10)`은 Pair Mode에서 무엇을 계산하나?

<details><summary>정답/힌트</summary>

_zip의 Mode10은 슬롯을 group1/group0으로 잡으므로 group1 - group0.

</details>

**Q8.** VCG의 packet clipper만으로는 R이 Slice와 Packet에 동시에 걸친 경우(R=2045, Slice=/8, Packet=%8)를 왜 표현 못 하나?

<details><summary>정답/힌트</summary>

valid_size(t)는 t에만 의존해 모든 슬라이스에 같은 값을 준다. 그런데 슬라이스 0~254는 8, 슬라이스 255는 5가 필요해 t 하나로 두 값을 못 낸다. (R::SIZE%packet_span=0이면 Slice-only로 환원돼 예외적으로 가능.)

</details>

**Q9.** layer norm에서 sqrt(var+eps)와 나눗셈을 같은 invocation에 넣고 싶다. 나눗셈을 vector_fp_binary(DivF)로 하면 무슨 문제가 생기고 어떻게 피하나?

<details><summary>정답/힌트</summary>

DivF는 FpFpu를 쓰는데 sqrt도 FpFpu라 단일-ALU 규칙으로 충돌. 전용 divider를 쓰는 vector_fp_div(FpDiv 단계, ReduceFpDiv)로 바꾸면 sqrt와 ALU가 겹치지 않는다.

</details>

**Q10.** i32 8원소(32B) 패킷을 i8로 cast하면 OutPacket을 왜 A#32로 줘야 하나?

<details><summary>정답/힌트</summary>

i8 8원소는 8바이트뿐이라 한 flit(32B)이 안 된다. cast 출력은 다시 32바이트로 패딩되므로 8원소를 32칸으로 패딩한 A#32가 정확한 출력 레이아웃이다.

</details>

## 5. 흔한 함정

- float 연산을 부르기 전 narrow(8→4), 부른 뒤 widen(4→8)을 안 하면 컴파일이 안 된다. final/inter_slice_reduce 출구는 둘 다 8-way를 요구한다.  
  ↳ 출처 `docs/src/computing-tensors/vector-engine/intra-slice-chain.md:74,80`
- 같은 ALU를 한 invocation에 두 번 쓰면 거부/패닉. AddFxp 두 번→'FxpAdd is already in use', tanh(sqrt(x))는 둘 다 FpFpu라 불가. 곱은 MulF(Mul0/Mul1/Fma)로 ALU를 분산해야 여러 개 가능.  
  ↳ 출처 `docs/src/computing-tensors/vector-engine/intra-slice-chain.md:176-178,252-262`
- softmax처럼 exp 뒤에 sum-reduce가 오면 identity 패딩(0)이 불가능(exp(p)=0인 p가 없음)하다. 반드시 VCG로 패딩을 제외해야 정확하다.  
  ↳ 출처 `docs/src/computing-tensors/vector-engine/intra-slice-reduce.md:184-185`
- Stash는 단일 사용·타입 고정이다. 두 번째 vector_stash()는 컴파일 오류, i32 스태시를 f32로 읽으면 런타임 패닉. Pair Mode에선 아예 못 쓰고, IntraFirst로 inter-slice reducer로 넘어가면 스태시가 사라진다.  
  ↳ 출처 `docs/src/computing-tensors/vector-engine/intra-slice-chain.md:103-108`
- intra-slice reduce의 Time 누적 슬롯은 8개뿐. OutTime(InnerTime)의 곱이 8을 넘으면 reduce 호출이 거부된다(예: 3×4=12).  
  ↳ 출처 `docs/src/computing-tensors/vector-engine/intra-slice-reduce.md:135-155`
- VCG는 모든 축 배치를 표현하지 못한다. Slice 부분식 stride 역순, Slice-Time interleave, TimeMajor 과패딩(PADDED_SIZE-R::SIZE > slice_span), Packet에 R major부/타축 공유, R이 Slice+Packet 동시(R::SIZE%packet_span≠0)는 전부 불가.  
  ↳ 출처 `docs/src/computing-tensors/vector-engine/vcg.md:589-788`
- packet clipper의 Packet 형태는 정확히 m![R # PADDED_SIZE % packet_span # 8]이어야 한다. 다른 축이 Packet을 R과 공유하면 그 축 원소가 prefix valid_size 때문에 조용히 패딩으로 취급돼 잘못 제외된다.  
  ↳ 출처 `docs/src/computing-tensors/vector-engine/vcg.md:357-358,721-743`
- VCG의 trim_way4는 v≤4를 정적으로 보장하는 매핑에서만 안전하다. 상위 4원소가 valid일 수 있으면 trim이 데이터를 잃는다.  
  ↳ 출처 `docs/src/computing-tensors/vector-engine/vcg.md:819-820`
- Pair Mode에서는 vector_narrow_clip/vector_widen_pad와 stash·filter를 못 쓴다. narrow/widen은 _split/_concat 변형을 써야 하고, unzip으로 시작하면 Filter 단계가 아예 막힌다.  
  ↳ 출처 `docs/src/computing-tensors/vector-engine/intra-slice-chain.md:144,154-155,170`
- Cast 입력은 정확히 한 32바이트 flit이어야 한다. 입력 패킷이 32바이트가 아니면 verify_cast가 assert로 막는다. 또 입력은 VeScalar(i32/f32)만 가능.  
  ↳ 출처 `furiosa-opt-std/src/engine/cast.rs:54-63`
- Transpose에서 원소가 넓어지면 max in_rows가 줄어든다(8bit=8, 16bit=4, 32bit=2). in_rows×sizeof(D)>8바이트면 verify_transpose가 막는다. out_rows>in_cols도 불가.  
  ↳ 출처 `furiosa-opt-std/src/engine/transpose.rs:83-88,139-144`
- float 나눗셈은 두 경로다: Fp 단계의 vector_fp_binary(DivF)는 FpFpu를 먹어 sqrt/tanh 등과 경합하고, 별도 FpDiv 단계의 vector_fp_div는 전용 divider(ReduceFpDiv)라 경합이 없다. layer norm처럼 sqrt와 나눗셈을 같이 쓰면 후자가 유리.  
  ↳ 출처 `docs/src/computing-tensors/vector-engine/intra-slice-chain.md:340-341,369-377`

## 6. 핵심 정리 & 다음

기억할 사실:
- Vector Engine은 32비트 타입(i32, f32)만 받는다. 좁은 타입은 상류 Contraction Engine(bf16곱→f32, i8곱→i32 누적) 또는 Fetch Engine의 타입캐스트 어댑터가 미리 넓혀줘야 한다. (`docs/src/computing-tensors/vector-engine/index.md:6-8`)
- 체인 단계는 12개로 고정 순서이며(Tag→Logic→Fxp→FxpToFp→Narrow→Float→IntraSliceReduce→FpDiv→Widen→FpToFxp→Clip→Filter), Logic/Fxp/Clip은 8-way, Float/IntraSliceReduce/FpDiv는 4-way로 돈다. 부동소수 ALU가 절반 처리량이라 float 경로는 Narrow(8→4)·Widen(4→8)로 감싼다. (`docs/src/computing-tensors/vector-engine/intra-slice-chain.md:57-74`)
- 한 단계 안에서 각 ALU는 한 Tensor Unit invocation당 최대 한 번만 동작한다. 같은 ALU를 두 번 쓰면 거부/패닉(예: AddFxp 두 번 → 'FxpAdd is already in use'; tanh와 sqrt는 둘 다 FpFpu라 동거 불가). (`docs/src/computing-tensors/vector-engine/intra-slice-chain.md:176-178,252-262`)
- Float 클러스터는 독립 ALU 5개(FpFma, FpFpu, FpExp, FpMul0, FpMul1). 곱을 Mul0/Mul1/Fma로 분산하면 한 invocation에 여러 곱을 넣을 수 있다. Exp/NegExp만 FpExp, sqrt/tanh/sigmoid/erf/log/sin/cos는 전부 FpFpu. (`furiosa-opt-std/src/engine/vector/op/mod.rs:367-373, docs/src/computing-tensors/vector-engine/intra-slice-chain.md:315,349-354`)
- Inter-Slice Reducer는 한 클러스터의 256 슬라이스를 가로질러 리듀스하고, 출력은 입력 모드와 무관하게 항상 Way8이다. 지연은 리덕션 비율 r에 대해 O(r)(링 한 바퀴). (`docs/src/computing-tensors/vector-engine/inter-slice-reducer.md:3-6,148-151`)
- Intra-Slice Reduce의 Time 누적 슬롯 용량은 8(InnerTime::SIZE ≤ 8). 초과하면 API가 reduce 호출을 거부한다. (`docs/src/computing-tensors/vector-engine/intra-slice-reduce.md:131-155`)
- VCG 자원: packet clipper 1개 + time filter 3개, 각 sequencer 엔트리 8개. 한 invocation에서 최대 4개 패딩 축까지 유효성 추적 가능. (`docs/src/computing-tensors/vector-engine/vcg.md:790-802`)
- Packet clipper는 valid_size를 시간 t로만 계산(슬라이스 무관)하므로 R이 Slice와 Packet에 동시에 걸치면 슬라이스마다 다른 valid_size가 필요해 표현 불가(R::SIZE % packet_span = 0인 경우만 예외). (`docs/src/computing-tensors/vector-engine/vcg.md:336-338,746-763`)

➡️ 다음: [08_scheduler.md](./08_scheduler.md)
