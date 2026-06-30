# 03 · 원소 단위 연산과 파이프라인 기초

이 문서는 vISA 커리큘럼 모듈 03입니다. 가장 단순한 커널들로 파이프라인의 뼈대(begin→fetch→collect→vector→commit)와 `to_dm`/`to_hbm`, 그리고 `sub` 컨텍스트로 VRF에 미리 싣는 패턴을 배웁니다.
*선행: 02 매핑 · 예상 시간: 반나절*

## 학습 목표

- [ ] begin→fetch→collect→vector_init→vector_*→vector_final→commit 흐름을 안다
- [ ] constant_add·elementwise_mul·binary_add·vrf_add를 직접 돌린다
- [ ] `sub` 컨텍스트가 VRF에 피연산자를 미리 싣는 이유를 안다
- [ ] `vector_fxp`로 간단한 원소 연산을 한다

## 1. 개념

## 0. 큰 그림: 데이터는 어디서 어디로 흐르나

vISA(Furiosa TCP Virtual ISA) 커널을 처음 보면 `begin → fetch → collect → vector_init → ... → commit` 같은 긴 체인이 나옵니다. 이게 무서워 보이지만, 사실은 "데이터가 메모리에서 연산기로, 다시 메모리로 흘러가는 길"을 한 줄씩 적어둔 것뿐입니다. 먼저 하드웨어 지형부터 머릿속에 그려둡시다.

TCP 장치는 4단계로 중첩돼 있습니다(`docs/src/quick-start.md:45-53`):
- **Chip**: 최상위 단위, HBM(대용량 메모리)을 들고 있음
- **Cluster**: 칩당 2개, 각자 256개의 slice를 묶음
- **Slice**: 클러스터당 256개, 각자 하나의 Tensor Unit(연산 파이프라인)을 굴림
- **Lane**: slice당 8개, Contraction Engine의 MAC 배열 한 줄

메모리 계층은 이렇습니다(`docs/src/quick-start.md:65-71`):
- **HbmTensor**(HBM): 패키지 위 48GB, 1.5TB/s. 가중치·활성값 장기 보관소
- **DmTensor**(DM, 온칩 SRAM): 총 256MB, slice당 512KB. 연산의 주 작업 메모리
- **VrfTensor**(VRF): slice당 8KB. Vector Engine이 매 사이클 읽는 피연산자 레지스터
- **TrfTensor**(TRF): Contraction Engine용 레지스터

핵심 흐름: 호스트(CPU)가 PCIe DMA로 데이터를 **HBM**에 올리고 → Tensor DMA로 **DM**에 내리고 → Tensor Unit 파이프라인(Fetch→…→Vector→…→Commit)이 DM에서 스트림을 빨아들여 계산하고 결과를 다시 DM에 쓰고 → Tensor DMA로 **HBM**에 올리고 → 호스트가 PCIe DMA로 가져갑니다(`docs/src/moving-tensors/index.md:1-30`). 즉 NPU 연산기는 HBM을 직접 만지지 않습니다. 항상 DM을 거칩니다.

Tensor Unit은 고정된 파이프라인입니다: **Fetch → Switch → Collect → Contraction → Vector → Cast → Transpose → Commit**(`docs/src/quick-start.md:57`). 대부분 단계는 slice 안에서 독립적으로 돕니다. 원소 단위 커널에서는 Contraction(행렬곱)을 건너뛰고 Fetch → Collect → Vector → Commit만 씁니다.

## 1. 텐서 매핑: m![] 의 / % # 이 뭔가

vISA는 "이 논리축이 하드웨어의 어디에 흩어지는가"를 **타입**으로 적습니다. 그래서 타입이 좀 길어 보입니다. 예를 들어 `axes![A = 2048]`로 축 A를 선언하면(`base-template/src/kernel/constant_add_kernel.rs:3`), 이렇게 나눕니다:

```rust
pub type Chip    = m![1];
pub type Cluster = m![1 # 2];
pub type Slice   = m![A / 8 # 256];
```

`m![]` 안에서 쓰는 연산자 3개의 의미입니다(`docs/src/quick-start.md:85-88`):
- `/` 는 **stride로 쪼개기**: `A / 8` = 2048/8 = 256, 즉 slice 인덱스 256개
- `%` 는 **안쪽 개수**: `A % 8` = slice 안의 8개 원소
- `#` 는 **하드웨어 단위 수로 패딩**: `# 256` = slice 256개에 맞춰 채움(남는 칸은 임의값)

그래서 `DmTensor<i32, m![1], m![1#2], m![A/8 # 256], m![A%8]>`는 "i32 벡터를, 칩 1개·클러스터 1개(2개 중)·256 slice에 8개씩"이라는 뜻입니다. A의 각 원소는 정확히 한 slice의 한 자리로 매핑됩니다.

여기에 파이프라인을 흐를 때만 등장하는 두 축이 더 있습니다(`docs/src/quick-start.md:90`): **Time**(파이프라인 반복 회차)과 **Packet**(한 회차에 처리하는 원소들). DmTensor 타입은 `<dtype, Chip, Cluster, Slice, Element>` 5칸이지만, 스트리밍 텐서(FetchTensor 등)는 Element 자리가 `Time, Packet`으로 바뀝니다.

## 2. 첫 커널: Constant Add 한 줄씩

`base-template/src/kernel/constant_add_kernel.rs`는 i32 벡터 2048개에 상수 1을 더합니다. 전체를 보며 각 줄이 왜 있는지 봅시다.

```rust
let dm = input.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma, 0);
```
HBM에 있던 입력을 DM으로 내립니다. `to_dm`은 평평한 2048개를 256 slice에 8개씩 흩뿌립니다. 두 번째 인자 `0`은 **DM 주소**입니다. DM은 main/sub가 공유하는 평평한 SRAM이라, 프로그래머가 주소를 직접 정해 겹치지 않게 해야 합니다(`docs/src/quick-start.md:101`). `ctx.tdma`는 Tensor DMA 컨텍스트(HBM↔DM 전용).

```rust
.begin(dm.view())
```
Tensor Unit 연산을 시작합니다. DM 텐서의 view를 파이프라인 입구에 물립니다(`furiosa-opt-std/src/context.rs:88`).

```rust
.fetch::<i32, m![1], m![A % 8]>()
```
Fetch Engine이 DM에서 8개짜리 패킷을 파이프라인으로 빨아들입니다. 제네릭 3개는 `<출력 dtype, Time, Packet>`입니다(`furiosa-opt-std/src/engine/fetch.rs:35`). 여기선 Time=1(한 회차), Packet=8. Fetch는 제약을 검사합니다: **Cluster는 반드시 2, Slice는 반드시 256, 출력 패킷은 32바이트 정렬**(`furiosa-opt-std/src/engine/fetch.rs:48-58`). 또 Fetch는 타입 캐스팅도 합니다 — 예를 들어 i8을 읽어 i32로 넓혀줄 수 있습니다(`docs/src/moving-tensors/fetch-engine.md:311-314`). Vector Engine은 32비트만 받으므로(아래 3장), Contraction을 건너뛰면 Fetch에서 미리 넓혀야 합니다.

```rust
.collect::<m![1], m![A % 8]>()
```
Collect Engine이 스트림을 **정확히 한 flit(32바이트 = i32 8개)**로 정규화합니다(`furiosa-opt-std/src/engine/collect.rs:43`, 검증은 같은 파일 `verify_collect`). flit는 이 하드웨어가 한 번에 다루는 데이터 한 덩어리라고 생각하면 됩니다. i32 8개가 정확히 32바이트라 `A % 8`=8이 딱 한 flit입니다.

```rust
.vector_init()
.vector_intra_slice_tag(TagMode::Zero)
.vector_fxp(FxpBinaryOp::AddFxp, 1)
.vector_final()
```
여기가 Vector Engine 구간입니다. `vector_init()`로 진입(`furiosa-opt-std/src/engine/vector/tensor/mod.rs:89`), `vector_intra_slice_tag(TagMode::Zero)`로 intra-slice 체인 시작. `TagMode::Zero`는 "모든 원소를 조건 없이 매 사이클 실행"이라는 뜻입니다(`docs/src/quick-start.md:132`, `furiosa-opt-std/src/engine/vector/branch.rs:17-21`). `vector_fxp(FxpBinaryOp::AddFxp, 1)`가 8개 원소 각각에 1을 더합니다. `vector_final()`로 VE를 빠져나옵니다(`furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:470`).

```rust
.commit::<m![A % 8]>(1 << 12);
```
결과를 DM 주소 `1<<12`(4096)에 씁니다(`furiosa-opt-std/src/engine/commit.rs:27`). 입력 dm은 주소 0에 있으니 출력은 다른 주소를 골라 겹치지 않게 한 겁니다. commit 입력은 한 flit(32B)여야 하고, 출력 패킷은 8/16/24/32 바이트만 허용됩니다(`verify_commit`).

```rust
result.to_hbm(&mut ctx.tdma, 1 << 28)
```
DM 결과를 다시 HBM으로 올립니다(`furiosa-opt-std/src/tensor/memory.rs:387`).

호스트 쪽(`base-template/src/constant_add.rs`)은 단순합니다. `Context::acquire()`로 컨텍스트를 얻고, `HostTensor::<i32, m![A]>::rand`로 난수 입력을 만들고, `to_hbm(&mut ctx.pdma, 0)`로 HBM에 올린 뒤(`ctx.pdma`는 PCIe DMA, 호스트↔HBM), `launch(constant_add_kernel, (&mut ctx, &in_hbm)).await`로 커널을 실행합니다. 테스트는 `expected[i] = in[i].wrapping_add(1)`로 참조값을 만들고 결과와 비교합니다(`base-template/src/constant_add.rs:28-39`). 한 가지 알아둘 점: **typecheck 백엔드에서는 실제 텐서가 비어 있어(phantom) 비교 루프가 0번 돌고 단언이 그냥 통과**합니다(`base-template/src/constant_add.rs:34-36`). 진짜 값 검증은 simulation 백엔드에서 일어납니다.

## 3. Vector Engine 입문: 32비트 전용, 두 엔진, 고정 순서, ALU 한 번 규칙

Vector Engine(VE)은 원소 단위 계산과 reduction을 담당합니다 — 활성함수(GELU, SiLU), 정규화(softmax, layer norm), 이항 연산, slice 내/간 reduction 등(`docs/src/computing-tensors/vector-engine/index.md:1-4`).

**중요한 제약 하나: VE는 32비트 타입(i32, f32)만 받습니다**(`docs/src/computing-tensors/vector-engine/index.md:6-8`). bf16·i8 같은 좁은 타입은 Contraction Engine이 곱하면서 자동으로 넓혀주지만(bf16→f32, i8→i32), Contraction을 건너뛰면 Fetch의 타입캐스트 어댑터가 넓혀야 합니다.

VE는 두 조각으로 나뉩니다(`docs/src/computing-tensors/vector-engine/index.md:11-20`):
- **Intra-Slice Chain**: slice 안에서 원소 단위/이항/slice-내 reduce. `vector_intra_slice_tag()`(단일 스트림) 또는 `vector_intra_slice_unzip()`(2그룹 짝 모드)로 진입.
- **Inter-Slice Reducer**: 클러스터의 256 slice를 가로질러 reduce. `vector_inter_slice_reduce()`로 진입.

`vector_init()`과 `vector_final()` 사이에서 체인만, reducer만, 또는 둘 다 돌릴 수 있습니다. 둘 다면 순서는 `IntraFirst`(체인 먼저) 또는 `InterFirst`(reducer 먼저)입니다.

### 3.1 Intra-Slice Chain의 고정 파이프라인

체인은 **정해진 순서**로 단계를 지나가고, 소프트웨어는 필요한 것만 부르고 나머지는 건너뜁니다(`docs/src/computing-tensors/vector-engine/intra-slice-chain.md:57-70`). 순서는:

1 Entry(tag) → 2 Logic → 3 Fxp → 4 FxpToFp → 5 Narrow → 6 Float → 7 IntraSliceReduce → 8 FpDiv → 9 Widen → 10 FpToFxp → 11 Clip → 12 Filter

타입 시스템이 단계 전이를 컴파일 타임에 강제합니다. 즉 앞 단계를 적절히 지나야 다음 메서드를 부를 수 있게 됩니다. 예를 들어 `vector_fp_*`(Float)는 Fxp/FxpToFp 다음에만 호출 가능합니다.

**Way(8/4) 개념**: Logic·Fxp·Clip은 8-way(사이클당 8원소), Float 클러스터는 4-way(절반)로 돕니다. 그래서 float 연산을 쓰려면 `vector_narrow_clip`/`vector_narrow_split`으로 8→4로 좁혔다가, 끝나면 `vector_widen_pad`/`vector_widen_concat`으로 4→8로 되돌려야 합니다(`docs/src/computing-tensors/vector-engine/intra-slice-chain.md:72-80`). `normal.rs`의 `ve_elementwise_fp_unary`가 정확히 narrow → `vector_fp_unary(FpUnaryOp::Exp)` → widen을 합니다(`furiosa-opt-examples/src/vector_engine/normal.rs:118-136`).

### 3.2 Tag 단계: 조건부 실행의 씨앗

체인 입구 Tag 단계는 각 32비트 원소에 4비트 `Tag`(0~15)를 붙입니다(`docs/src/computing-tensors/vector-engine/intra-slice-chain.md:180-196`). 비트3(MSB)은 `GroupId`(Filter·짝 모드가 사용), 비트0~2는 비교 결과 플래그입니다. `TagMode`(`furiosa-opt-std/src/engine/vector/branch.rs:17`)는:
- `Zero`: 모든 비트 0, 모두 무조건 실행(원소 단위 커널의 기본)
- `AxisToggle { axis }`: GroupId = 축 인덱스 % 2 (짝 모드의 내부 구현)
- `Comparison([cmp0..cmp3])`: 원소 단위 비교 4개 결과로 비트를 채움
- `ValidCount`, `Vrf`: VCG 출력 / VRF에서 태그 재사용

### 3.3 단계별 연산과 "ALU 한 번" 규칙 (가장 자주 걸리는 함정)

각 단계 안에는 ALU 풀이 있고, **한 ALU는 Tensor Unit 한 번 호출당 최대 한 번만** 쓸 수 있습니다(`docs/src/computing-tensors/vector-engine/intra-slice-chain.md:176-178`). 같은 ALU를 쓰는 연산 두 개를 한 패스에 넣으면 패닉입니다.

Fxp 클러스터(i32 정수/고정소수, 8-way)의 ALU 배치(`docs/src/computing-tensors/vector-engine/intra-slice-chain.md:230-251`, 열거형 `furiosa-opt-std/src/engine/vector/op/mod.rs:203`):
- `AddFxp`, `AddFxpSat`, `SubFxp`, `SubFxpSat` → 전부 **FxpAdd** ALU 공유
- `MulFxp`, `MulInt` → **FxpMul**
- 시프트류 → FxpLshift / FxpRshift

그래서 `normal.rs:27-45`의 `ve_elementwise_fxp_chain`은 일부러 실패하도록 만든 예제입니다: `AddFxp`(FxpAdd) → `MulInt`(FxpMul, OK) → `SubFxp`(FxpAdd 또 사용!) → **"FxpAdd is already in use" 패닉**. 테스트 `test_ve_elementwise_fxp_chain`은 이 패닉을 catch해서 "정말 패닉하는지"를 검증합니다(`furiosa-opt-examples/tests/vector_engine_tests.rs:70-80`).

Float 클러스터(f32, 4-way)는 5개 ALU(FpFma, FpFpu, FpExp, FpMul0, FpMul1)를 노출합니다(`docs/src/computing-tensors/vector-engine/intra-slice-chain.md:310-347`). `MulF`는 `FpMulAlu::Mul0/Mul1/Fma`로 어느 곱셈기를 쓸지 직접 고를 수 있습니다 — 곱셈 두 번을 한 패스에 넣고 싶으면 Mul0·Mul1로 분산하면 됩니다. `ve_elementwise_full_pipeline`이 fxp→fp 변환, narrow, fp 곱, widen, fp→fxp, clip을 한 줄로 엮은 종합 예제입니다(`furiosa-opt-examples/src/vector_engine/normal.rs:47-73`).

Clip 클러스터(8-way)에는 `Min/Max/AbsMin/AbsMax`(클램핑)과 `AddFxp`(또 다른 덧셈 경로!)가 있습니다(`furiosa-opt-std/src/engine/vector/op/mod.rs:603`). 즉 Fxp의 FxpAdd가 이미 찼어도 Clip의 ClipAdd로 한 번 더 더할 수 있습니다 — ALU가 다른 단계라 별개입니다.

### 3.4 피연산자 소스와 인자 모드, Stash

이항/삼항 연산의 추가 입력(operand)은 세 곳에서 옵니다(`docs/src/computing-tensors/vector-engine/intra-slice-chain.md:82-101`):
- **상수**: `vector_fxp(FxpBinaryOp::AddFxp, 100)`
- **VRF 텐서**: `vector_fxp(FxpBinaryOp::MulInt, &vrf)` 또는 `VeRhs::vrf(&vrf)`
- **Stash**: `vector_stash()`로 찍어둔 이전 단계 스냅샷을 나중에 `Stash`로 다시 읽음. 잔차 연결 `max(f(x), x)` 같은 데 씀(`furiosa-opt-examples/src/vector_engine/normal.rs:97-116`의 `ve_elementwise_stash_i32`).

**인자 모드(BinaryArgMode)**는 stream과 operand의 자리를 바꿔 같은 연산으로 다른 식을 만듭니다(`docs/src/computing-tensors/vector-engine/intra-slice-chain.md:110-127`): `Mode01`(기본, stream-operand), `Mode10`(operand-stream). 그래서 `vector_fxp_with_mode(FxpBinaryOp::SubFxp, BinaryArgMode::Mode10, 7)`는 `7 - x`를 계산합니다(`furiosa-opt-examples/src/vector_engine/normal.rs:205-224`).

## 4. VRF와 sub 컨텍스트: 두 번째 피연산자를 미리 채워두기

원소 단위 곱처럼 입력이 둘이면, 하나는 파이프라인으로 흘리고 다른 하나는 **VRF**(slice당 레지스터)에 미리 채워둡니다. VE가 매 사이클 VRF를 읽어 stream과 결합합니다(`docs/src/computing-tensors/register-files.md:114-116`).

### 4.1 두 컨텍스트: main과 sub

모든 커널은 두 실행 컨텍스트를 **동시에** 굴립니다(`docs/src/quick-start.md:94-104`): `ctx.main`(주 계산)과 `ctx.sub`(보조 파이프라인, 보통 operand를 TRF/VRF에 미리 채움). main이 아직 sub가 채우는 중인 operand를 필요로 하면 **자동으로 sub를 기다립니다**(동기화는 알아서). 단 둘은 같은 평평한 SRAM을 공유하므로 주소가 겹치면 안 됩니다.

### 4.2 Elementwise Mul 한 줄씩

`base-template/src/kernel/elementwise_mul_kernel.rs`:

```rust
let lhs_dm = lhs.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma, 0);
let rhs_dm = rhs.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma, 1 << 12);
```
두 입력을 DM의 **다른 주소**(0, 4096)에 내려 겹침을 피합니다(`elementwise_mul_kernel.rs:16-17`).

```rust
let rhs_vrf: VrfTensor<i32, Chip, Cluster, Slice, m![A % 8]> = ctx
    .sub
    .begin(rhs_dm.view())
    .fetch::<i32, m![1], m![A % 8]>()
    .collect::<m![A % 8 / 8], m![A % 8 % 8]>()
    .to_vrf(0);
```
**sub 컨텍스트**가 rhs를 VRF에 채웁니다. main과 똑같이 begin→fetch→collect를 하지만 마지막이 `.to_vrf(0)`입니다(`furiosa-opt-std/src/engine/collect.rs:72`). `to_vrf`는 스트리밍의 Time/Packet을 Element2로 평탄화하고 주어진 raw 주소에 저장합니다(`docs/src/computing-tensors/register-files.md:128-143`).

```rust
let result = ctx
    .main
    .begin(lhs_dm.view())
    .fetch::<i32, m![1], m![A % 8]>()
    .collect::<m![1], m![A % 8]>()
    .vector_init()
    .vector_intra_slice_tag(TagMode::Zero)
    .vector_fxp(FxpBinaryOp::MulInt, &rhs_vrf)
    .vector_final()
    .commit::<m![A % 8]>(1 << 13);
```
**main 컨텍스트**가 lhs를 흘리며 `vector_fxp(FxpBinaryOp::MulInt, &rhs_vrf)`로 slice마다 자기 8개 lhs와 VRF의 8개 rhs를 곱합니다(`elementwise_mul_kernel.rs:29-39`). operand 자리에 상수 대신 `&rhs_vrf`(VRF 텐서 참조)를 넣은 게 핵심입니다 — `vector_fxp`의 operand는 `impl IntoOperands`라 상수든 VRF든 받습니다(`furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:1130`). 테스트 참조는 `lhs[i].wrapping_mul(rhs[i])`입니다(`base-template/src/elementwise_mul.rs:36`).

### 4.3 vrf_add: 브로드캐스트되는 VRF operand

`furiosa-opt-examples/src/vrf_add.rs`는 `[A,B]` 행렬에 `[B]` 벡터를 더합니다(B축 브로드캐스트). sub가 rhs(`[B]`)를 VRF에 채우고(`vrf_add.rs:25-30`), main이 lhs(`[A,B]`)를 흘리며 `vector_fxp(FxpBinaryOp::AddFxp, &rhs_vrf)`로 더합니다(`vrf_add.rs:33-41`). 여러 VRF operand를 한 패스에서 쓰는 예도 있습니다 — `ve_elementwise_multi_vrf`는 vrf1으로 AddFxp, vrf2로 MulInt, vrf1으로 다시 ClipAdd를 합니다(서로 다른 ALU라 가능, `furiosa-opt-examples/src/vector_engine/normal.rs:440-480`). 참고로 통합 테스트 `test_vrf_add`는 `#[ignore = "Failing on cpu"]`라 기본 실행에서 빠집니다(`furiosa-opt-examples/tests/vector_engine_tests.rs:18-19`).

## 5. 두 입력을 다르게 합치기: begin_interleaved와 짝(Pair) 모드

VRF 말고 두 입력을 합치는 다른 길이 있습니다: 두 스트림을 한 텐서로 **교차(interleave)**해 넣고, VE 안에서 다시 두 그룹으로 풀어 합치는 방식입니다.

`furiosa-opt-examples/src/binary_add.rs`(i8 두 텐서를 i32로 더함):
```rust
ctx.main
    .begin_interleaved::<I, _, _, _, _, _>(lhs.view(), rhs.view())
    .fetch::<i32, m![I], m![A % 8]>()
    .collect::<m![I], m![A % 8]>()
    .vector_init()
    .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
    .vector_clip_zip(ClipBinaryOpI32::AddFxp)
    .vector_final()
    .commit_view(out)
```
- `begin_interleaved::<I, ...>(lhs, rhs)`가 두 입력을 2칸짜리 축 `I`로 끼워넣습니다(lhs→그룹0, rhs→그룹1). 내부적으로 두 입력을 tile해서 한 텐서로 만듭니다(`furiosa-opt-std/src/context.rs:98-114`). 여기서 `i8`을 fetch에서 `i32`로 넓힙니다(FetchCast).
- `vector_intra_slice_unzip::<I, ...>()`가 짝 모드 진입입니다. 입력의 2-way 축 I로 GroupId를 만들어(`TagMode::AxisToggle` 사용) 두 병렬 스트림으로 분리합니다(`docs/src/computing-tensors/vector-engine/intra-slice-chain.md:140-160`, `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:624`).
- `vector_clip_zip(ClipBinaryOpI32::AddFxp)`가 두 그룹을 다시 하나로 합치며 더합니다(group0+group1). zip 후 결과는 그룹1로 암묵 필터됩니다(`binary_add.rs:40-44`).

짝 모드의 4단계 흐름(`docs/src/computing-tensors/vector-engine/intra-slice-chain.md:147-151`): ①unzip으로 두 스트림 분리 → ②두 그룹이 lock-step으로 단계 통과(paired 단계) → ③`_zip` 연산으로 둘을 하나로 융합(merged 단계) → ④이후 보통 체인처럼 final로. `zip.rs`에 변형이 많습니다: `ve_group_pair_add`(단순 덧셈), `ve_group_pair_preprocess_g0`(그룹0만 `vector_fxp(MulInt, 10, ())` — `()`는 그 쪽 건너뛰기), `ve_group_pair_fxp`(`vector_fxp_zip`), `ve_group_pair_fp`(narrow_split→fp_zip→widen_concat) 등(`furiosa-opt-examples/src/vector_engine/zip.rs`).

짝 모드 제약(`docs/src/computing-tensors/vector-engine/intra-slice-chain.md:169-172`): `stash()`·`filter()` 사용 불가, `vector_narrow_clip`/`vector_widen_pad` 대신 `_split`/`_concat` 사용, `_zip` 전에는 inter-slice reducer로 못 넘어감, ALU 사용량은 두 그룹이 공유(한 그룹에서 쓴 ALU는 양쪽 다 쓴 걸로 침).

## 6. reduce 계열도 같은 골격

reduce 예제들(`furiosa-opt-examples/src/vector_engine/reduce.rs`)도 같은 begin→fetch→collect→vector_init→…→commit 골격입니다. 차이는 VE 구간 안에서:
- **Intra-slice reduce**: `vector_intra_slice_reduce::<축, Time, Packet>(IntraSliceReduceOpI32::AddSat/Max/Min)`로 slice 내부 reduce. 4-way라 앞뒤로 narrow_clip/widen_pad를 끼웁니다(`reduce.rs:3-25`).
- **Inter-slice reduce**: `vector_inter_slice_reduce::<OutSlice, OutTime>(InterSliceReduceOpI32::AddSat/Max)`로 256 slice 가로질러 reduce(`reduce.rs:219-237`). i32는 `AddSat/Max/Min`, f32는 `Add/Max/Min`.
- 둘 다 쓰면 `vru_then_vau`처럼 inter-slice reduce 후 intra 체인으로 bias를 더할 수 있습니다(`reduce.rs:287-304`).

큰 축을 Slice·Time·Packet으로 쪼개 부분은 intra로, 나머지는 inter로 합치는 패턴이 `docs/src/computing-tensors/vector-engine/index.md:116-145`(full_sum)에 잘 나와 있습니다.

## 7. 실행과 검증 정리

- **시뮬레이션 실행**(base-template bin): `cargo furiosa-opt run --release --bin constant_add` → "Constant Add: kernel ran" 출력. 실제 값 계산.
- **참조 대조 테스트**: `cargo furiosa-opt test --release --bin constant_add` → 호스트 참조(out=in+1)와 비교.
- **typecheck 백엔드**: `--backend typecheck`를 붙이면 텐서가 phantom이라 값 비교 루프가 생략되고, 타입·매핑 정합성만 검사합니다(`base-template/src/constant_add.rs:34-36`). 컴파일/타입 오류 실험에 좋습니다.
- **examples 크레이트**(라이브러리 + 통합 테스트): `cargo furiosa-opt test --release --test binary_add_tests`, `--test vector_engine_tests`, `--test fetch_commit_tests` 형태로 돕니다(`furiosa-opt-examples/tests/`).

요약하면, 원소 단위 커널은 전부 같은 골격입니다: HBM에서 `to_dm`으로 내리고 → `begin/fetch/collect`로 파이프라인에 흘리고 → `vector_init`부터 `vector_final`까지 VE로 계산하고 → `commit`으로 DM에 쓰고 → `to_hbm`으로 올린다. 달라지는 건 VE 안에서 어떤 연산을 부르느냐, operand를 상수·VRF·짝 그룹 중 무엇으로 주느냐뿐입니다.

## 2. 핵심 API · 패턴

| 이름 | 쓰는 법 | 설명 | 출처 |
|---|---|---|---|
| `HbmTensor::to_dm` | `input.to_dm::<Cluster, Slice, Element2>(&mut ctx.tdma, address) -> DmTensor` | HBM 텐서를 DM(온칩 SRAM)으로 내림. address는 겹치지 않게 직접 지정. ctx.tdma = Tensor DMA 컨텍스트. | `furiosa-opt-std/src/tensor/memory.rs:423` |
| `DmTensor::to_hbm` | `result.to_hbm(&mut ctx.tdma, address) -> HbmTensor` | DM 결과를 HBM으로 올림. 호스트↔HBM은 ctx.pdma(PCIe DMA)를 씀. | `furiosa-opt-std/src/tensor/memory.rs:387,712` |
| `TuContext::begin` | `ctx.main.begin(dm.view()) -> BeginTensor` | Tensor Unit 파이프라인 시작. ctx.main 또는 ctx.sub에서 호출. | `furiosa-opt-std/src/context.rs:88` |
| `TuContext::begin_interleaved` | `ctx.main.begin_interleaved::<I, _, _, _, _, _>(lhs.view(), rhs.view()) -> BeginTensor` | 두 입력을 2칸 축 I로 교차해 한 텐서로. lhs=그룹0, rhs=그룹1. 짝 모드의 입구. | `furiosa-opt-std/src/context.rs:98` |
| `TuTensor::fetch` | `.fetch::<D2, Time2, Packet2>() -> FetchTensor` | DM→파이프라인 스트림. D2로 타입 캐스팅(i8→i32 widen) 가능. Cluster=2·Slice=256·32B정렬 검사. | `furiosa-opt-std/src/engine/fetch.rs:35` |
| `TuTensor::collect` | `.collect::<Time2, Packet2>() -> CollectTensor` | 스트림을 정확히 한 flit(32B)로 정규화. | `furiosa-opt-std/src/engine/collect.rs:43` |
| `TuTensor::to_vrf` | `.to_vrf::<Element>(address) -> VrfTensor` | Collect 결과를 VRF에 저장(보통 ctx.sub에서). Time/Packet을 Element로 평탄화. | `furiosa-opt-std/src/engine/collect.rs:72` |
| `TuTensor::vector_init` | `.vector_init() -> VectorInitTensor` | Vector Engine 구간 진입. 입력은 VeScalar(i32/f32)만. | `furiosa-opt-std/src/engine/vector/tensor/mod.rs:89` |
| `VectorInitTensor::vector_intra_slice_tag` | `.vector_intra_slice_tag(TagMode::Zero) -> VectorBranchTensor` | 단일 스트림 intra-slice 체인 시작. TagMode::Zero = 모든 원소 무조건 실행. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:613` |
| `VectorInitTensor::vector_intra_slice_unzip` | `.vector_intra_slice_unzip::<I, TileTime, SplitTime>() -> VectorTensorPair` | 짝 모드 진입. 2-way 축 I로 두 그룹 분리(내부적으로 TagMode::AxisToggle). | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:624` |
| `VectorTensor::vector_fxp` | `.vector_fxp(FxpBinaryOp::AddFxp, operand) — operand는 상수/&VrfTensor 둘 다 가능` | i32 고정소수 이항 연산(Way8). AddFxp/SubFxp는 FxpAdd, MulInt/MulFxp는 FxpMul ALU. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:1130` |
| `VectorTensor::vector_fxp_with_mode` | `.vector_fxp_with_mode(FxpBinaryOp::SubFxp, BinaryArgMode::Mode10, 7) // 7 - x` | 인자 모드로 stream/operand 자리 교체. Mode10 = operand - stream. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:1140` |
| `VectorTensorPair::vector_clip_zip` | `.vector_clip_zip(ClipBinaryOpI32::AddFxp) // group0 + group1` | 짝 모드에서 두 그룹을 Clip 단계 연산으로 융합. ClipBinaryOpI32: Min/Max/AbsMin/AbsMax/AddFxp/AddFxpSat. | `furiosa-opt-examples/src/vector_engine/zip.rs:19, furiosa-opt-std/src/engine/vector/op/mod.rs:603` |
| `VectorTensor::vector_final` | `.vector_final() -> VectorFinalTensor` | Vector Engine 구간 종료. 이후 commit/cast/transpose 가능. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:470` |
| `TuTensor::commit / commit_view` | `.commit::<Element>(address) -> DmTensor  /  .commit_view(out)` | 결과를 DM 주소에 쓰거나(commit), 미리 만든 DmTensorViewMut에 씀(commit_view). 입력 한 flit, 출력 8/16/24/32B. | `furiosa-opt-std/src/engine/commit.rs:27,34` |
| `launch` | `launch(kernel_fn, (&mut ctx, &in_hbm)).await -> HbmTensor` | #[device(chip=1)] 커널을 컨텍스트와 인자로 실행. 호스트 프로그램에서 호출. | `base-template/src/constant_add.rs:12` |

## 3. 실험 (직접 돌리기)

> 실험은 NPU 없이 `simulation`·`typecheck`로 돌아갑니다. 실행법은 [`../experiments/README.md`](../experiments/README.md), MNIST는 `cargo furiosa-opt test`(npu 전용).

### 실험 03.1 — Constant Add를 시뮬레이션으로 돌리고 참조와 대조하기
*난이도 1/5 · 기반: `base-template/src/constant_add.rs`*

**목표** — begin→fetch→collect→vector_init→vector_fxp→vector_final→commit 전체 체인이 실제로 out=in+1을 만드는지 눈으로 확인한다.

```bash
cargo furiosa-opt run --release --bin constant_add
cargo furiosa-opt test --release --bin constant_add
```
**관찰** — run은 'Constant Add: kernel ran' 출력. test는 expected[i]=in[i].wrapping_add(1)과 비교해 통과(녹색). 실패 시 'constant_add mismatch at i=...' 메시지가 뜬다.

**심화** — 끝에 --backend typecheck를 붙여 같은 test를 다시 실행해보고, 값 비교 루프가 생략되어도(phantom 텐서) 통과하는 것을 확인한다(constant_add.rs:34-36).

### 실험 03.2 — 상수 바꾸기: +1 을 +5 로 (predict-then-test)
*난이도 2/5 · 기반: `base-template/src/kernel/constant_add_kernel.rs`*

**목표** — vector_fxp의 operand가 결과에 어떻게 직결되는지, 그리고 호스트 참조도 같이 바꿔야 함을 익힌다.

```bash
# 커널: .vector_fxp(FxpBinaryOp::AddFxp, 1) -> .vector_fxp(FxpBinaryOp::AddFxp, 5)
# 호스트 테스트 참조: x.wrapping_add(1) -> x.wrapping_add(5) (base-template/src/constant_add.rs:30)
cargo furiosa-opt test --release --bin constant_add
```
**관찰** — 두 곳을 모두 바꾸면 통과. 커널만 바꾸고 참조를 안 바꾸면 'constant_add mismatch'로 실패 — 이로써 시뮬레이션이 실제로 +5를 계산했음이 증명된다.

**심화** — AddFxp를 FxpBinaryOp::MulInt로 바꾸고 operand를 3으로, 참조를 wrapping_mul(3)으로 바꿔 곱셈도 같은 골격으로 됨을 확인.

### 실험 03.3 — ALU 충돌 패닉 재현: FxpAdd 두 번
*난이도 2/5 · 기반: `base-template/src/kernel/constant_add_kernel.rs`*

**목표** — '한 ALU는 호출당 한 번' 규칙을 직접 깨뜨려 'FxpAdd is already in use' 패닉을 본다.

```bash
# 커널의 vector_fxp 줄을 아래 3줄로 교체(normal.rs:38-40의 실패 예제와 동일 구조):
#   .vector_fxp(FxpBinaryOp::AddFxp, 10)
#   .vector_fxp(FxpBinaryOp::MulInt, 2)
#   .vector_fxp(FxpBinaryOp::SubFxp, 5)
cargo furiosa-opt run --release --bin constant_add
```
**관찰** — AddFxp(FxpAdd)와 SubFxp(FxpAdd)가 충돌해 'FxpAdd is already in use' 패닉. MulInt(FxpMul)는 통과한다. 실험 후 원복.

**심화** — SubFxp를 Clip 단계 연산 .vector_clip(ClipBinaryOpI32::AddFxp, 5)로 옮기면(다른 단계의 ClipAdd ALU) 패닉이 사라지는 것을 확인. 참조도 (x+10)*2 후 max/add에 맞춰 조정.

### 실험 03.4 — Elementwise Mul: sub 컨텍스트 + VRF operand
*난이도 3/5 · 기반: `base-template/src/elementwise_mul.rs`*

**목표** — sub가 rhs를 VRF에 채우고 main이 vector_fxp(MulInt, &rhs_vrf)로 곱하는 흐름을 실행으로 확인한다.

```bash
cargo furiosa-opt run --release --bin elementwise_mul
cargo furiosa-opt test --release --bin elementwise_mul
```
**관찰** — test는 lhs[i].wrapping_mul(rhs[i])와 비교해 통과. lhs_dm(주소 0)과 rhs_dm(주소 1<<12)이 다른 DM 주소를 쓰는 것에 주목(겹치면 안 됨).

**심화** — MulInt를 AddFxp로 바꾸고(여전히 operand는 &rhs_vrf), 참조를 wrapping_add로 바꿔 VRF operand가 곱셈뿐 아니라 덧셈에도 그대로 쓰임을 확인.

### 실험 03.5 — Binary Add: begin_interleaved + 짝 모드 unzip/zip
*난이도 2/5 · 기반: `furiosa-opt-examples/tests/binary_add_tests.rs`*

**목표** — VRF 대신 두 입력을 교차해 넣고 unzip→clip_zip으로 합치는 경로를 검증한다. i8→i32 fetch widen도 함께 본다.

```bash
cargo furiosa-opt test --release --test binary_add_tests
```
**관찰** — test_binary_add_2048이 (lhs as i32)+(rhs as i32)와 비교해 통과. lhs는 그룹0, rhs는 그룹1로 들어가 vector_clip_zip(AddFxp)로 더해진다(binary_add.rs:33-44).

**심화** — zip 전에 그룹0만 전처리하도록 .vector_fxp(FxpBinaryOp::MulInt, 2, ())를 삽입(zip.rs의 ve_group_pair_preprocess_g0 참고)하고, 참조를 2*lhs+rhs로 바꿔 통과시켜본다. ()가 '그 그룹 건너뛰기'임을 확인.

### 실험 03.6 — Vector Engine 모음 테스트 + fp 경로 narrow/widen
*난이도 3/5 · 기반: `furiosa-opt-examples/tests/vector_engine_tests.rs`*

**목표** — fxp/fp/logic/stash/VRF 변형들이 한 번에 검증되는 것을 보고, f32 경로가 narrow→fp→widen으로 감싸짐을 코드로 확인한다.

```bash
cargo furiosa-opt test --release --test vector_engine_tests
```
**관찰** — test_ve_elementwise_fxp_const(+100), test_ve_elementwise_fxp_chain(패닉 검증), stash/ternary 테스트들이 통과. ve_elementwise_fp_unary는 narrow_clip→fp_unary(Exp)→widen_pad 구조(normal.rs:118-136).

**심화** — ve_elementwise_full_pipeline(normal.rs:47-73)의 .vector_fp_binary(MulF(Mul0), 2.5)를 2.0으로 바꾼 뒤 해당 테스트를 다시 돌려 결과 변화를 추적(참조 갱신 필요). Float ALU 선택(Mul0/Mul1)의 의미도 음미.

### 실험 03.7 — Vector 없는 파이프라인 골격: fetch→collect→commit
*난이도 2/5 · 기반: `furiosa-opt-examples/src/fetch_commit.rs`*

**목표** — Vector Engine을 완전히 빼고도 fetch→collect→commit만으로 데이터가 통과함을 보고, 파이프라인 백본을 분리해 이해한다.

```bash
cargo furiosa-opt test --release --test fetch_commit_tests
```
**관찰** — fetch_commit_simple은 vector_init 없이 begin→fetch::<i32,...>→collect→commit만으로 i8 입력을 i32로 옮기고 축을 [A,B]→[B,A]로 재배치한다(fetch.rs의 i8→i32 widen이 핵심). 테스트 통과.

**심화** — fetch의 출력 dtype을 i32 대신 i8로 두려고 시도하고 --backend typecheck로 검사해, collect/commit의 flit·정렬 제약(32B)으로 타입이 막히는지 관찰.

### 실험 03.8 — 타입 검사로 commit 제약 깨보기 (find-the-type-error)
*난이도 3/5 · 기반: `base-template/src/kernel/constant_add_kernel.rs`*

**목표** — commit 출력 패킷이 8/16/24/32B만 허용된다는 제약을 typecheck 백엔드에서 직접 부딪힌다.

```bash
# .commit::<m![A % 8]>(...) 의 Element를 잘못된 크기로 바꿔본다 (예: m![A % 8 # 3] 처럼 32B가 아닌 패킷)
cargo furiosa-opt test --release --bin constant_add --backend typecheck
```
**관찰** — commit/collect의 verify가 'Commit input packet must be exactly 32 bytes' 또는 출력 8/16/24/32B 위반을 보고하며 막힌다(commit.rs verify_commit). 올바른 m![A % 8](=8개 i32=32B)로 되돌리면 통과.

**심화** — fetch의 Packet을 32B 정렬이 안 되는 크기로 바꿔 fetch.rs verify_fetch의 'Fetch output packet must be 32-byte aligned' 어서션도 유발해본다.

## 4. 연습문제 (손으로, 컴파일 없이)

**Q1.** axes![A = 2048]일 때 constant_add_kernel의 입력을 256 slice에 8개씩 흩뿌리는 Slice·Element 매핑 타입을 m![] 표기로 쓰시오. 그리고 왜 한 slice의 Element가 정확히 한 flit인지 설명하시오.

<details><summary>정답/힌트</summary>

Slice = m![A / 8 # 256](2048/8=256 slice를 256으로 패딩), Element = m![A % 8](slice당 8개). i32 8개 = 32바이트 = 1 flit이라 collect가 딱 한 flit로 정규화된다.

</details>

**Q2.** 다음 체인이 패닉하는 이유와 패닉 메시지를 말하시오: vector_intra_slice_tag(Zero) → vector_fxp(AddFxp,10) → vector_fxp(MulInt,2) → vector_fxp(SubFxp,5).

<details><summary>정답/힌트</summary>

AddFxp와 SubFxp가 둘 다 FxpAdd ALU를 쓰는데 한 패스에서 같은 ALU 재사용 금지. MulInt는 FxpMul이라 OK. 메시지: 'FxpAdd is already in use'.

</details>

**Q3.** vector_fxp_with_mode(FxpBinaryOp::SubFxp, BinaryArgMode::Mode10, 7)을 입력 x=3에 적용하면 결과는? Mode01이었다면?

<details><summary>정답/힌트</summary>

Mode10은 op(operand, stream)=7-3=4. Mode01(기본)은 op(stream, operand)=3-7=-4.

</details>

**Q4.** elementwise_mul_kernel에서 rhs를 ctx.sub로 VRF에 채우고 lhs를 ctx.main으로 흘린다. main이 곱셈을 시작할 때 sub가 아직 VRF를 다 못 채웠다면 어떻게 되는가? 그리고 lhs_dm과 rhs_dm 주소가 같으면?

<details><summary>정답/힌트</summary>

main은 자동으로 sub를 기다려 동기화된다(quick-start.md:99). 주소가 같으면 둘이 같은 DM 영역을 덮어써 결과가 망가진다 — 그래서 0과 1<<12로 분리한다.

</details>

**Q5.** binary_add는 begin_interleaved로 lhs·rhs를 축 I에 끼우고 vector_intra_slice_unzip 후 vector_clip_zip(AddFxp)로 합친다. lhs는 어느 그룹이고, 만약 ClipBinaryOpI32::AddFxp 대신 (가상의) 비대칭 빼기를 group1-group0 모드(Mode10)로 한다면 결과 식은?

<details><summary>정답/힌트</summary>

lhs=그룹0, rhs=그룹1(begin_interleaved 순서). _zip의 Mode10은 op(group1, group0)이므로 rhs - lhs.

</details>

**Q6.** f32 sigmoid를 한 패스로 계산하려면 vector_intra_slice_tag(Zero) 다음과 vector_final() 이전에 어떤 두 메서드를 반드시 끼워야 하며, 그 이유는?

<details><summary>정답/힌트</summary>

vector_narrow_clip(Way8→Way4)을 fp_unary(Sigmoid) 앞에, vector_widen_pad(Way4→Way8)을 뒤에. Float 클러스터가 4-way라 좁혔다가 8-way로 되돌려야 final로 나갈 수 있기 때문.

</details>

## 5. 흔한 함정

- main과 sub(그리고 모든 to_dm/commit)는 같은 평평한 DM 주소 공간을 공유한다. 주소를 안 겹치게 직접 정하지 않으면 텐서가 서로 덮어쓴다. elementwise_mul은 lhs_dm=0, rhs_dm=1<<12로 분리한다.  
  ↳ 출처 `docs/src/quick-start.md:101-102, base-template/src/kernel/elementwise_mul_kernel.rs:16-17`
- 한 Vector Engine 패스에서 같은 ALU를 두 번 쓰면 패닉한다. AddFxp/SubFxp는 둘 다 FxpAdd ALU라 함께 쓰면 'FxpAdd is already in use'. 덧셈을 두 번 하려면 하나를 Clip 단계의 AddFxp(ClipAdd ALU)로 옮긴다.  
  ↳ 출처 `furiosa-opt-examples/src/vector_engine/normal.rs:21-45, docs/src/computing-tensors/vector-engine/intra-slice-chain.md:252-263`
- 커널의 연산을 바꿨다면 호스트 테스트의 참조(reference) 계산도 함께 바꿔야 한다. 안 그러면 시뮬레이션은 맞게 계산했는데 mismatch로 실패한다(오히려 시뮬레이션이 진짜 계산했다는 증거).  
  ↳ 출처 `base-template/src/constant_add.rs:28-39`
- typecheck 백엔드에서는 출력 텐서가 phantom이라 값 비교 루프가 0번 돌고 단언이 그냥 통과한다. 실제 수치 검증은 simulation 백엔드에서 해야 한다.  
  ↳ 출처 `base-template/src/constant_add.rs:34-36`
- Vector Engine은 32비트(i32/f32)만 받는다. Contraction을 건너뛰는 원소 단위 커널에서 i8 같은 좁은 입력은 fetch에서 i32로 widen해야 한다(binary_add가 i8 입력을 .fetch::<i32,...>로 넓힌다).  
  ↳ 출처 `docs/src/computing-tensors/vector-engine/index.md:6-8, furiosa-opt-examples/src/binary_add.rs:35`
- float 연산 앞뒤로 narrow/widen을 빼먹으면 타입이 안 맞아 컴파일이 막힌다. Float 단계는 Way4라 vector_narrow_clip/split(8→4) 이후에만 호출되고, vector_widen_pad/concat(4→8)으로 되돌려야 final/clip로 갈 수 있다.  
  ↳ 출처 `docs/src/computing-tensors/vector-engine/intra-slice-chain.md:72-80`
- 짝 모드에서는 stash()와 filter()가 막히고, vector_narrow_clip/vector_widen_pad 대신 반드시 _split/_concat 변형을 써야 한다. _zip 이전(paired 단계)에는 inter-slice reducer로도 못 넘어간다.  
  ↳ 출처 `docs/src/computing-tensors/vector-engine/intra-slice-chain.md:155-172`
- vrf_add의 통합 테스트 test_vrf_add는 #[ignore = "Failing on cpu"]로 표시돼 있어 기본 test 실행에서 빠진다. 굳이 돌리려면 -- --ignored가 필요하고 cpu 백엔드에서는 실패할 수 있다. VRF add 자체는 vector_engine_tests의 다른 테스트들로 검증된다.  
  ↳ 출처 `furiosa-opt-examples/tests/vector_engine_tests.rs:18-19`
- fetch는 Cluster=2·Slice=256를 강제하고 출력 패킷은 32B 정렬, collect 출력은 정확히 한 flit(32B), commit 출력은 8/16/24/32B만 허용한다. Element 매핑 크기를 잘못 잡으면 이 verify들에서 막힌다.  
  ↳ 출처 `furiosa-opt-std/src/engine/fetch.rs:48-58, furiosa-opt-std/src/engine/commit.rs:46-55`

## 6. 핵심 정리 & 다음

기억할 사실:
- TCP 하드웨어는 4단계 중첩: Chip(시스템 의존) > Cluster(칩당 2개, 256 slice 묶음) > Slice(클러스터당 256개, 각자 Tensor Unit 1개) > Lane(slice당 8개, MAC 배열 한 줄). (`docs/src/quick-start.md:45-53`)
- 메모리 계층 용량: HBM 48GB·1.5TB/s, DM 총 256MB(slice당 512KB), VRF slice당 8KB(Vector Engine operand 레지스터), TRF slice당 8KB/lane. (`docs/src/quick-start.md:65-71`)
- Tensor Unit은 고정 파이프라인: Fetch → Switch → Collect → Contraction → Vector → Cast → Transpose → Commit. 대부분 단계는 slice 내부에서 독립 실행. (`docs/src/quick-start.md:57-59`)
- flit는 32바이트 = i32 8개. Collect Engine은 스트림을 정확히 한 flit(32B)로 정규화한다(그래서 m![A%8]=8이 한 flit). (`furiosa-opt-std/src/engine/collect.rs:43, base-template/src/kernel/constant_add_kernel.rs:20`)
- Vector Engine은 32비트 타입(i32, f32)만 입력으로 받는다. 좁은 타입은 Contraction Engine이 곱하며 자동 widen(bf16→f32, i8→i32)하거나, Contraction을 건너뛰면 Fetch의 타입캐스트 어댑터가 widen해야 한다. (`docs/src/computing-tensors/vector-engine/index.md:6-8`)
- Fetch Engine 제약: Cluster 크기는 반드시 2, Slice 크기는 반드시 256, 출력 패킷은 32바이트(FETCH_ALIGN_BYTES) 정렬이어야 한다. (`furiosa-opt-std/src/engine/fetch.rs:48-58`)
- Commit Engine 제약: 입력 패킷은 정확히 한 flit(32B), 출력 패킷은 8/16/24/32 바이트만 허용, 잘라내기는 Packet에서만 가능. (`furiosa-opt-std/src/engine/commit.rs:46-55`)
- Vector Engine 단계 안에서 각 ALU는 Tensor Unit 한 번 호출당 최대 한 번만 사용 가능. Fxp 단계의 AddFxp/SubFxp는 모두 FxpAdd ALU를 공유하므로 한 패스에 둘을 넣으면 'FxpAdd is already in use' 패닉. (`docs/src/computing-tensors/vector-engine/intra-slice-chain.md:176-178,230-263`)

➡️ 다음: [04_contraction.md](./04_contraction.md)
