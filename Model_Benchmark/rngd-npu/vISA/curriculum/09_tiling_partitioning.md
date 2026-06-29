# 09 · 타일링과 분할 전략

이 문서는 vISA 커리큘럼 모듈 09입니다. DM 용량(512KB/slice)을 넘는 워크로드를 시간·공간으로 쪼개는 법: 타일링, split-K, 그리고 chip/cluster reduce(손으로 짜는 ReduceScatter)를 matmul 예제로 봅니다.
*선행: 04 텐서 축약, 06 연산 엔진 I · 예상 시간: 하루*

## 학습 목표

- [ ] 시간 분할 vs 공간 분할을 구분한다
- [ ] split-K 누적이 무엇이고 언제 쓰는지 안다
- [ ] chip/cluster reduce를 dm_shuffle+parallel_copy+add로 손코딩하는 패턴을 안다
- [ ] matmul reduce 변형들을 읽고 트레이스한다

## 1. 개념

## 0. 이 장이 다루는 것 — "커널을 조합한다"는 의미

지금까지의 장들은 부품을 따로 설명했습니다. 매핑 식(`m![...]`)으로 텐서를 하드웨어 계층(Chip/Cluster/Slice/Lane/Time/Packet)에 어떻게 뿌리는지, 데이터를 어떻게 옮기는지(Fetch/Switch/Collect/Commit/DMA), 어떻게 계산하는지(Contraction/Vector Engine), 어떻게 스케줄하는지였죠. 이 장은 그 부품들을 합쳐서 "실제로 돌아가는 커널 하나"를 만드는 방법입니다. 출처: `docs/src/kernel-examples/index.md:3`.

큰 그림부터 잡읍시다. RNGD(TCP) 하드웨어에는 두 가지 근본 한계가 있어서, 큰 텐서를 그냥 한 번에 처리할 수 없습니다.

- DM(데이터 메모리, on-chip SRAM): 전체 256MB이지만 **slice당 512KB**가 한도입니다 (`docs/src/quick-start.md:68`).
- VRF(Vector Engine 피연산자 레지스터): **slice당 8KB**뿐입니다 (`docs/src/quick-start.md:71`, `docs/src/kernel-examples/tiling.md:7`).
- TRF(Contraction Engine 입력): lane당 8KB, slice당 8 lane이라 full 모드 64KB/slice, half 모드 32KB/slice (`docs/src/quick-start.md:70`, `furiosa-opt-examples/src/matmul/matmul_16384.rs:52`).

이 한계 때문에 "텐서를 쪼개서, 순서대로 또는 병렬로 처리하고, 부분 결과를 다시 합치는" 패턴이 거의 모든 커널의 뼈대가 됩니다. 이 장의 모든 주제(타일링, split-reduce, chip/cluster reduce)는 결국 같은 질문에 대한 답입니다: **"한 번에 안 들어가는 걸 어떻게 쪼개고, 쪼갠 부분 합(partial sum)을 어떻게 다시 더하느냐."**

## 1. 시간 분할 vs 공간 분할 (Temporal vs Spatial Partitioning)

가장 먼저 머리에 박아둘 개념입니다. 일을 나누는 축에는 두 종류가 있습니다.

- **공간 분할(Spatial)**: 서로 다른 하드웨어 유닛이 동시에 다른 조각을 처리합니다. 칩이 여러 개, 클러스터가 여러 개, slice가 256개, lane이 여러 개 — 이들에 일을 뿌리면 "동시에" 끝납니다. 매핑 식에서 `Chip`, `Cluster`, `Slice`, `Lane`, `Packet` 차원이 여기 해당합니다.
- **시간 분할(Temporal)**: 같은 하드웨어가 시간을 나눠 여러 번(loop) 돌면서 조각을 처리합니다. 매핑의 `Time` 차원이 여기 해당합니다.

`docs/src/kernel-examples/index.md:8-15`의 표가 이걸 정리합니다. 각 차원이 "어디서 정의되고(어느 메모리에 뿌려지고)" "어디서 reduce되는지(누가 합치는지)"가 핵심입니다.

| 차원 | 종류 | 합치는 하드웨어 |
|---|---|---|
| `Chip` | 공간 | DMA + Vector Engine |
| `Cluster` | 공간 | DMA + Vector Engine |
| `Slice` | 공간 | Vector Engine (Inter-Slice Reducer, 일명 GAT 글로벌 가산기 트리) |
| `Lane` | 공간 | Contraction Engine |
| `Time` | 시간 | Contraction Engine (Time Reducer 누산기) |
| `Packet` | 공간 | Contraction Engine (Packet Reducer 트리) |

여기서 중요한 직관 하나: **reduce(합산) 축을 어느 차원에 배치하느냐가 곧 성능을 결정**합니다. 같은 행렬곱이라도 reduce할 축(K축)을 Packet/Time/Slice에 두면 칩 안에서 싸게 끝나지만, Chip/Cluster에 두면 칩 사이 통신(DMA)이 필요해서 비쌉니다. 그래서 문서는 "가능하면 reduce 축은 Slice/Element에 두라"고 권합니다 (`docs/src/kernel-examples/chip-cluster-reduce.md:314`).

`quick-start.md:284`도 같은 말을 합니다: 512KB/slice를 넘는 워크로드에는 두 전략이 있다 — temporal은 타일을 시간순으로, spatial은 타일을 병렬 유닛으로.

## 2. 타일링 (Tiling)

### 2.1 왜 필요한가

타일링은 큰 텐서를 on-chip 메모리에 들어가는 작은 타일로 쪼개는 것입니다 (`docs/src/kernel-examples/tiling.md:6`). 세 가지 상황에서 씁니다 (`tiling.md:12-18`):

1. 한 차원이 한 번의 하드웨어 패스에 안 들어갈 때 (DM/VRF 용량 초과).
2. 이미 불러온 데이터를 재사용해 메모리 대역폭을 아끼고 싶을 때.
3. 공간 차원은 이미 다 썼는데, 시간(loop)으로 더 나눠야 할 때.

### 2.2 타일 크기를 무엇으로 정하나

타일 크기가 모든 걸 결정합니다 (`tiling.md:23`). 조건은:
- VRF(8KB) / DM 용량에 맞을 것,
- 정렬 제약(32바이트 flit)을 만족할 것,
- fetch와 compute를 겹치려면(더블 버퍼링) 여유 공간을 남길 것.

타일 크기가 정해지면 실행은 4단계입니다 (`tiling.md:24`): (1) 바깥 차원에서 타일 루프, (2) HBM→DM로 타일 fetch, (3) 계산, (4) 부분 결과 누적 후 write-back.

### 2.3 실제 코드로 보는 타일링 — `matmul_16384`

16384×16384 i8 행렬곱입니다. lhs/rhs가 각각 256MB라 1칩 DM(256MB)에 동시에 안 들어갑니다. 그래서 두 겹으로 타일링합니다 (`furiosa-opt-examples/src/matmul/matmul_16384.rs:165-208`):

```rust
for i in 0..32 {                 // C축을 512씩 32타일 (결과를 바로 HBM에 쓸 수 있어 C부터 자름)
    let rhs_tile = rhs.view().tile::<m![C / 512], 1, m![B, 1 # 32, C % 512]>(i);
    ...
    for j in 0..4 {              // B(reduce축)를 4096씩 4타일 — TRF(작은 메모리)에 맞추려고
        let lhs_tile = lhs.view().tile::<m![B / 4096], 1, m![A, 1 # 4, B % 4096]>(j);
        let rhs_subtile = rhs_tile.tile::<m![B / 4096], 1, ...>(j);
        if j == 0 { contraction_over_b_2048(ctx, lhs_tile, rhs_subtile, result0...); }
        else { contraction_over_b_2048(..., temp); /* 누적 */ }
    }
}
```

여기서 두 가지 타일링 의도를 구분하세요. C를 자르는 건 "출력 결과를 바로 HBM에 쓸 수 있으니 먼저"라는 출력 타일링이고, B를 자르는 건 "reduce 축인데 작은 TRF(64KB/slice)에 안 들어가니 잘라서 누적"이라는 reduce 타일링입니다. 주석이 그대로 설명합니다 (`matmul_16384.rs:177-179`). B를 자른 뒤 부분곱들을 다시 더하는 것이 바로 다음 절의 split-reduce입니다.

`tile`의 시그니처는 `furiosa-opt-std/src/tensor/view.rs:99` (`fn tile<I, E2, const LEN>(&self, start) -> TensorView`)이고, 내부에서 `assert_div`로 매핑이 나눠떨어지는지 컴파일 타임에 검사합니다. 즉 타일 분할이 산술적으로 안 맞으면 빌드가 실패합니다 — 이게 vISA가 "타입으로 매핑을 검증한다"는 핵심입니다.

### 2.4 한 패스에 안 들어갈 때 컴파일러가 알아서: Tensor Segmentation

타일링을 손으로 하지 않아도, VRF 8KB/slice를 넘는 텐서는 컴파일러가 자동으로 여러 execution으로 쪼갭니다 (`docs/src/kernel-examples/fetch-commit-engine.md:398-510`). 예: `[A=2048, B=32]` f8 = 64KB는 8KB 한도를 넘으므로 A를 1024씩 두 segment로 자릅니다.

```rust
let seg_0 = ctx.main.begin(input.view().slice(A, 0..1024)).fetch...().commit(256*1024);
let seg_1 = ctx.main.begin(input.view().slice(A, 1024..2048)).fetch...().commit(256*1024 + 32*1024);
```

여기서 꼭 기억할 사실(`fetch-commit-engine.md:429-435`): **8KB는 slice당 한도이지 전체 한도가 아닙니다.** 클러스터에 slice가 256개라 총 2MB지만, slice 하나는 8KB뿐입니다. 그래서 `m![1,1,1,2048,32]`처럼 slice 차원에 아무것도 안 뿌리면 64KB가 전부 slice 0에 몰려 터집니다. reduce처럼 "한 slice가 행/열 전체를 들고 있어야 하는" 연산에서 이 한도에 자주 걸립니다.

## 3. Split Reduce (= split-K 누적)

### 3.1 언제 쓰나

reduce(합산)하려는 논리 축이 **하나의 연속된 하드웨어 차원에 매핑되지 못할 때** 씁니다 (`docs/src/kernel-examples/split-reduce.md:4`). 세 경우 (`split-reduce.md:10-14`):
- 축이 VRF(8KB)에 한 번에 안 들어가서 여러 물리 텐서로 쪼개야 할 때 (나눠야 함),
- 이미 데이터가 여러 텐서에 나뉘어 있을 때 (레이어/expert/시간 조각별로),
- 같은 칩 안에 있지만 메모리 할당이 따로라 DMA보다 interleaved fetch가 더 쌀 때 (칩 간 통신 회피).

reduce 계층에서의 위치 (`split-reduce.md:16-22`): Packet reduce(패킷 안) → Time reduce(시간축) → Slice reduce(클러스터 내 slice간, Inter-Slice Reducer) → **Split reduce(독립 텐서 인스턴스들 사이, interleaved fetch + VE binary op)** → Chip/Cluster reduce(칩/클러스터 간, DMA 추가).

### 3.2 핵심 기법: Interleaved Fetch

여러 텐서를 "번갈아" 시간축으로 읽어들여, 시간축에 `I`라는 인터리브 차원을 만든 다음 Vector Engine이 그 `I` 축을 binary op로 합칩니다 (`split-reduce.md:27-43`).

```rust
let interleaved = ctx.main.begin_interleaved().fetch(&tensor_0, &tensor_1); // I=2 차원 생성
let reduced = interleaved.reduce_add(axis: I);                              // VE가 I축 합산
```

`time[0]`=tensor_0, `time[1]`=tensor_1, `time[2]`=tensor_0 ... 이렇게 번갈아 흐르고, VE가 add/max/min을 인터리브 축으로 수행합니다 (`split-reduce.md:45`). 2개만 합칠 땐 인터리브 차원 없이 `sum_0.binary_add(sum_1)`로도 됩니다 (`split-reduce.md:309-320`).

대표 예: LayerNorm. Hidden=8192 bf16 = 16KB라 VRF 8KB에 안 들어가니 4096씩 두 청크로 나눠 각각 부분합을 내고, split-reduce로 합쳐 전체 평균/분산을 구합니다 (`split-reduce.md:47-148`). 수학적으로 `mean([H0,H1]) = (sum(H0)+sum(H1))/8192`로 동일합니다.

성능 직관 (`split-reduce.md:324-343`): 비용은 fetch가 지배합니다. 2-way 분할이면 메모리 대역폭 2배, 4-way면 4배. 그래서 **분할 수를 최소화**하라(각 텐서를 VRF 한도까지 키워라)가 제1원칙입니다. 분할은 2/4/8 같은 2의 거듭제곱이 하드웨어에 잘 맞습니다.

### 3.3 실제 코드 — `matmul_split_reduce.rs`의 VE 누적 패턴

문서의 의사코드(`reduce_add`)는 실제 std API에서 더 저수준입니다. `[1024,2048]×[2048] -> [1024]`에서 B(reduce축)를 512씩 4타일로 자르고, 부분곱을 VE binary add로 누적합니다 (`furiosa-opt-examples/src/matmul/matmul_split_reduce.rs:12-46`):

```rust
ctx.main
    .begin_interleaved::<I, _, _, _, _, _>(rhs.view(), lhs.view()) // I=2 인터리브 (rhs→Group0, lhs→Group1)
    .fetch::<i32, m![A / 8 % 8, I], m![A % 8]>()
    .collect::<m![A / 8 % 8, I], m![A % 8]>()
    .vector_init()
    .vector_intra_slice_unzip::<I, m![A / 8 % 8, 1 # 2], m![A / 8 % 8]>() // I=2를 두 그룹으로 풀기
    .vector_clip_zip(ClipBinaryOpI32::AddFxp)                            // Group0 + Group1 = 합
    .vector_final()
    .commit_view(out)
```

여기 "stash/zip" 메커니즘을 이해하는 게 중요합니다. `begin_interleaved`가 두 텐서를 I=2 시간 인터리브 스트림으로 만들고(`furiosa-opt-std/src/context.rs:98`), `vector_intra_slice_unzip`가 그걸 Group0/Group1 두 그룹으로 갈라(`furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:624`), `vector_clip_zip(Add)`가 둘을 더해 **결과를 Group1 자리에 놓습니다**(`furiosa-opt-std/src/engine/vector/tensor/vector_tensor_pair.rs:1385` 도크: "Result = op(group0, group1), result is placed in Group 1 positions"). 그래서 코드 주석 "result is implicitly filtered to Group 1 only"가 이 뜻입니다.

누적은 running accumulator 레지스터가 아니라 **result0/result1/temp 3개 버퍼 핑퐁**으로 합니다 (`matmul_split_reduce.rs:82-134`): j=0은 result0에 직접, 그 뒤로는 temp에 부분곱을 내고 add_split_contractions로 이전 누적과 더해 result0↔result1을 번갈아 채웁니다. 이게 split-K를 VE 이진 덧셈으로 구현한 모습입니다.

참고로 contraction 본체는 항상 같은 체인입니다 (`matmul_split_reduce.rs:84-95`): `begin → fetch → collect → contract_outer(&rhs_trf) → contract_packet → contract_time → contract_lane(LaneMode) → vector_init → vector_inter_slice_reduce(Add) → vector_final → commit`. 즉 einsum = (broadcast) + elementwise multiply(Contraction) + reduce(Packet/Time/Lane는 Contraction, Slice는 `vector_inter_slice_reduce`=GAT). matmul에서 reduce용 B/64를 Slice에 뿌렸기 때문에 GAT가 그걸 접고 결과를 broadcast하며, 그 broadcast를 표현하려고 `X=32` 더미 축을 둡니다 (`matmul_16384.rs:170-172`).

## 4. Chip / Cluster Reduce — 손으로 짠 ReduceScatter / AllReduce

### 4.1 먼저: 클러스터 안에서 끝낼 수 있으면 끝내라

칩/클러스터 경계를 넘기 전에, 한 클러스터 안에서 Contraction Engine + Inter-Slice Reducer만으로 큰 텐서를 스칼라까지 접을 수 있습니다. `m![A]`, A=65536을 스칼라로 reduce하는 데 약 **296 cycle**입니다 (`docs/src/kernel-examples/chip-cluster-reduce.md:13-34`):
- Packet `A % 32` → Packet Reducer 트리(깊이 5) = 5 cycle,
- Time `A / 32 % 8` → Time Reducer 누산 = 8 cycle,
- Slice `A / 256` → Inter-Slice Reducer = 256 cycle.

이 세 단계를 합쳐 65536→1. 칩 간 통신은 비싸니, 가능하면 이렇게 칩 안에서 처리하라는 게 출발점입니다.

### 4.2 ReduceScatter — 대각 슬라이스 + ChipShuffle

reduce 축이 `Chip`/`Cluster`에 걸쳐 있으면 칩 간에 부분 결과를 합쳐야 합니다. ReduceScatter는 reduce하면서 결과를 각 칩에 한 조각씩 나눠 갖게 합니다(메모리 절약). 4칩 예 (`chip-cluster-reduce.md:42-185`): 입력은 `[A=4,B=4]` i8, chip이 A를 들고 있고(chip k가 A=k의 모든 B), 목표는 A로 합산하되 결과를 B로 흩뿌리기.

알고리즘은 6단계입니다. **대각 패턴**으로 T0..T3 네 개의 중간 텐서를 만든 뒤 더합니다:
- T0 = Slice(0,1,2,3): 각 칩이 대각 원소 선택,
- T1 = Slice(3,0,1,2) + ChipShuffle(1,2,3,0): 한 칸 회전한 대각,
- T2 = Slice(2,3,0,1) + ChipShuffle(2,3,0,1),
- T3 = Slice(1,2,3,0) + ChipShuffle(3,0,1,2),
- Step5: VE add로 T0+T1+T2+T3 → A축 소거,
- Step6: AllGather로 필요한 분배.

대각 슬라이스가 핵심인 이유: "각 출력 원소에 필요한 데이터를 모든 칩에서 미리 모아둬서 통신 라운드를 최소화"하기 위함입니다 (`chip-cluster-reduce.md:185`).

### 4.3 AllReduce — 균일 회전

AllReduce는 ReduceScatter와 달리 **모든 칩이 동일한 전체 결과**를 갖습니다(데이터 병렬 학습의 gradient 평균 등). 대각이 아니라 균일 ChipShuffle 회전을 씁니다 (`chip-cluster-reduce.md:188-312`): 원본 T0에 회전 3번으로 T1,T2,T3을 만들고 넷을 더하면 모든 칩이 같은 합을 얻습니다(덧셈 교환법칙). n칩이면 회전 n-1번 (8칩=7번, 16칩=15번).

### 4.4 구현 프리미티브 3종 (`chip-cluster-reduce.md:316-339`)

- **Asymmetric Slice(비대칭 슬라이스)**: 서브컨텍스트에서 `ParallelCopy`(`stos` = Store to SRAM)로 구현. 특정 칩 위치에서 일부 원소만 골라 복사. main 계산과 겹쳐 실행 가능.
- **Shuffle**: 칩 내부는 `DmaCommand`(HBM↔HBM), 칩 간은 `PCIeDmaCommand`. 칩 간 데이터 이동이라 chip/cluster reduce의 주요 비용. 수백~수천 cycle.
- **Tensor Addition**: main 컨텍스트에서 interleaved fetch + VE binary add. 별도 누산 버퍼 없이 번갈아 들어온 피연산자를 더함.

### 4.5 실제 코드 — `matmul_cluster_reduce.rs` (2클러스터) / `matmul_chip_reduce.rs` (4칩)

2클러스터 ReduceScatter (`furiosa-opt-examples/src/matmul/matmul_cluster_reduce.rs:67-121`):
```rust
let sliced0 = ctx.sub.parallel_copy_cluster_slice::<2, ...>(tensor.view(), &[0, 1]); // 대각
let sliced1 = ctx.sub.parallel_copy_cluster_slice::<2, ...>(tensor.view(), &[1, 0]); // 반대 대각
let shuffled1 = sliced1.view().dm_cluster_shuffle::<2>(&mut ctx.tdma, &[1, 0]);       // 클러스터 0↔1 교환
ctx.main.begin_interleaved::<I,...>(sliced0.view(), shuffled1.view())                 // T0+T1
    .fetch...().vector_init().vector_intra_slice_unzip::<I,...>().vector_clip_zip(ClipBinaryOpI32::AddFxp)...;
```
`parallel_copy_cluster_slice`의 인덱스 배열 `&[0,1]`/`&[1,0]`이 문서의 Slice(0,1)/Slice(1,0)에 정확히 대응합니다 (`furiosa-opt-std/src/context.rs:132`). `dm_cluster_shuffle`의 패턴이 ChipShuffle 회전입니다 (`furiosa-opt-std/src/tensor/memory.rs:907`).

4칩 버전은 같은 패턴을 4개로 확장합니다 (`furiosa-opt-examples/src/matmul/matmul_chip_reduce.rs:64-171`): `parallel_copy_chip_slice`로 T0(=[0,1,2,3]), T1(slice=[3,0,1,2]+shuffle=[1,2,3,0]), T2([2,3,0,1]+[2,3,0,1]), T3([1,2,3,0]+[3,0,1,2])를 만들고, 2단계 이진 덧셈으로 sum01=T0+T1, sum012=sum01+T2, reduced=sum012+T3. 문서 §ReduceScatter의 대각+회전 패턴 그대로입니다. 마지막에 `reshape`로 reduce된 chip 차원 자리에 C/2%4를 승격시켜 출력 분배를 표현합니다.

## 5. Fetch / Commit Engine — 레이아웃 변환과 쓰기 단위

데이터가 흐르는 전체 경로는: `input → fetch sequencer → Switch Engine → Collect Engine → commit unit → output` (`docs/src/kernel-examples/fetch-commit-engine.md:3`). 이 경로에서 레이아웃을 바꾸고 쓰기 단위를 정합니다. 네 가지 예제가 각 측면을 보여줍니다.

### 5.1 Axis Permutation (축 순서 바꾸기, 계산 없이)

`[A,B,C] → [B,A,C]`를 Switch Engine으로 재배치합니다 (`fetch-commit-engine.md:8-131`). fetch가 시간순 `[A,B]`로 읽어도, write sequencer 설정이 주소를 `[B,A]` 순서로 만들어 줍니다. 시퀀서 표기 `[axis=count:stride, ...] @ base / commit_size`의 뜻: 각 축마다 count번 반복하며 stride 바이트씩 전진, `@`는 기준 주소, `/`는 commit당 쓰는 바이트 (`fetch-commit-engine.md:78`). 예: `[A=3:8, B=5:24, C=8:1] @ 1024 / 8`. **계산 자원 0**, 메모리 read/write 주소 패턴만으로 transpose/NCHW↔NHWC를 합니다.

제약 3가지 (`fetch-commit-engine.md:118-122`): (1) commit은 항상 8바이트 단위(8바이트 정렬), (2) 시퀀서 엔트리는 최대 8개, (3) write sequencer가 주소를 정하므로 flit 시간순과 다른 비연속 쓰기가 가능(그래서 AB→BA 가능).

### 5.2 Full-flit Commit (32바이트 통째 쓰기)

차원이 자연스럽게 32바이트(=flit) 경계에 맞으면 commit slicer 없이 32바이트 flit을 통째로 씁니다 (`fetch-commit-engine.md:133-219`). Example 1이 15 cycle 걸리던 걸 B,C를 합쳐 32바이트로 패딩하면 3 cycle로 5배 빨라집니다. 교훈: "압축 저장보다, 하드웨어 단위(32B)에 맞춰 패딩하는 게 더 빠를 때가 있다." 단, write sequencer는 stride가 0이면 안 되므로 commit 시 broadcast(재사용)나 선택적 버리기는 불가합니다 (`fetch-commit-engine.md:221`).

### 5.3 Tail Padding과 Fetch Size — 패딩 선택이 성능을 좌우

`[A=65, B=2]`를 commit할 때, tail 패딩(=`dummy`) 양에 따라 가능한 fetch_size가 달라집니다 (`fetch-commit-engine.md:224-365`). 하드웨어 fetch_size는 8/16/24/32바이트만 됩니다.

| dummy | fetch_size | cycle | 메모리 오버헤드 | 평가 |
|---|---|---|---|---|
| 7 | 24B | 6 | 9.7% | Good |
| 15 | 16B | 10 | 18.8% | Moderate |
| 23 | 8B | 22 | 26.1% | Poor |
| 31 | 32B | 6 | 32.3% | Best |

dummy=23(8B fetch)은 22 cycle, dummy=31(32B fetch)은 6 cycle — 거의 4배 차. 매핑 식 `m![A # 72]`의 `# 72`가 A를 72까지 패딩한다는 뜻이고, 이 패딩 값이 곧 dummy입니다. **데이터+패딩이 32로 나눠떨어지게 패딩하라**가 원칙. 메모리 몇 바이트 더 쓰는 비용은 보통 무시할 만합니다.

그리고 비대칭성 하나 (`fetch-commit-engine.md:374-395`): **read sequencer는 텐서 범위 밖을 overfetch해도 OK(읽기만 하니까)지만, write sequencer는 절대 텐서 경계를 넘어 쓰면 안 됩니다**(남의 영역 침범). 그래서 dummy=23은 32바이트 commit을 쓸 수 없습니다.

### 5.4 Tensor Segmentation — 2.4에서 설명함 (VRF 8KB 초과 자동 분할).

## 6. Transformer 커널 (Llama 3 70B) 전체 워크스루

`docs/src/kernel-examples/transformer.md`는 transformer의 모든 연산을 TCP 하드웨어 부품에 매핑합니다. 디코더-only 모델이라 **prefill(입력 인코딩)**과 **decode(토큰 생성)** 두 단계로 돕니다.

파라미터 (`transformer.md:19-31`): V=128256(vocab), D=8192(hidden), F=28672(FFN 중간), L=80(레이어), h_q=64(쿼리 헤드), h_kv=8(KV 헤드), G=8(=h_q/h_kv 그룹), d_k=128(헤드 차원), RoPE용 d_k=d_k_prime(64)*f(2).

### 6.1 Prefill 단계 (입력 전체를 병렬 처리)

1. **Embedding Lookup**: `x_0 = gather(input, w_emb)`. gather는 TensorDMA가 처리 (`transmer.md:39-54`).
2. **레이어 L번 반복**:
   - **Input RMSNorm** → Vector Engine (`transformer.md:61-74`).
   - **GQA(Grouped Query Attention)**:
     - **QKV Projection**: Q/K/V 각각 `einsum(x_norm, w_*)`로 D축을 reduce. einsum = broadcast → elementwise multiply(Contraction Engine) → reduce-add(scope별: packet=Packet Reducer, time=Time Reducer, slice=global adder tree, split=interleaved fetch+VE, cluster/chip=DMA+VE) (`transformer.md:81-105`). 이 reduce 분해가 1~5장과 이 장 전체를 잇는 고리입니다.
     - **RoPE**: d_k를 d_k_prime×f로 쪼개 cos/sin 2×2 회전 행렬을 `gather`로 lookup한 뒤 einsum으로 적용. "회전 행렬만 준비되면 RoPE는 einsum 하나"라는 게 TCP-friendly 트릭 (`transformer.md:107-139`).
     - **KV Cache 저장**: einsum 결과를 DM→HBM, TensorDMA (`transformer.md:141-154`).
     - **Attention Scores**: `scores = (Q @ K.T)/sqrt(d_k)`. einsum으로 d_k reduce, G는 K에서 broadcast (`transformer.md:160-182`). `/sqrt(d_k)`는 상수곱으로 VE.
     - **Causal Mask**: `j<=i`만 통과, 나머지 -inf. attention_mask를 VE의 **branch log**에 써서 분기 연산으로 처리 (`transformer.md:184-197`).
     - **Softmax**: key 축 reduce, VE (`transformer.md:199-212`).
     - **Weighted Sum**: `einsum(attn_weights, V)`로 s_in_kv reduce, G는 V에서 broadcast (`transformer.md:214-230`).
     - **Output Projection**: `einsum(attn_output, w_o)` → `[B,s_in,D]` (`transformer.md:232-244`).
     - **Residual**: `x_attn = x_prev + attn_out`, VE elementwise add (`transformer.md:246-257`).
   - **FFN(SwiGLU)**: Post-Attn RMSNorm → gate/up projection(einsum) → `activated = SiLU(gate)*up` (SiLU(x)=x*sigmoid(x), VE) → down projection(einsum) → Residual (`transformer.md:259-314`).
3. **Final RMSNorm** (L층 후, `transformer.md:316-326`).
4. **LM Head**: prefill에선 마지막 토큰만 slice해서 `einsum(x_last, w_lm_head)` → logits `[B,V]`. 보통 `w_lm_head = w_emb.T`(weight tying) (`transformer.md:328-346`).
5. **Sampling**: temperature scaling → softmax → 토큰 선택(greedy/top-k/top-p). **Host에서** 수행, TCP 아님 (`transformer.md:348-372`).

### 6.2 Decode 단계 (한 토큰씩, KV 캐시 재사용)

연산 순서는 prefill과 같지만 세 가지가 다릅니다 (`transformer.md:375-384`):
- **단일 토큰 입력**: s_in=1 (가장 최근 토큰만 query),
- **KV 캐시 재사용**: K/V를 새로 계산하지 않고 캐시에서 읽음 → KV Cache **Update**(concat: slice간 이동은 RoutingEngine/parallel copy, element간은 parallel copy, HBM concat은 DMA) (`transformer.md:473-493`),
- **autoregressive**: 매 토큰이 캐시의 모든 과거 토큰을 참조. **causal mask 불필요**(현재 토큰은 과거만 봄) (`transformer.md:521-524`).

Attention 모양도 다릅니다: prefill은 `[B,h_q,s_in,s_in]`, decode는 `[B,h_q,1,s]`.

### 6.3 Prefill vs Decode 성격 (`transformer.md:680-694`)

| | Prefill | Decode |
|---|---|---|
| 입력 길이 | s_in(가변) | 1(고정) |
| 병렬성 | s_in 토큰 동시 | 1 토큰 |
| KV Cache | 생성·저장 | 읽기·갱신 |
| Causal mask | 필요 | 불필요 |
| 성격 | **compute-bound** | **memory-bound** |
| 처리량/지연 | 높음/상대적 높음 | 낮음/토큰당 낮음 |

이 차이가 서빙 최적화의 출발점입니다: prefill은 연산을 키워 처리량을, decode는 KV 캐시 접근(메모리 대역폭)을 줄여 지연을 잡습니다.

## 7. Mixture of Experts (MoE) — branchless 라우팅과 blockwise 희소 계산

MoE는 토큰마다 E개 expert 중 K개만 골라 처리해 파라미터는 많이 쓰되 계산은 아끼는 구조입니다 (`docs/src/kernel-examples/mixture-of-experts.md:4`). 파라미터: E(보통 128), K(llama4=1, gpt-oss=4, qwen3=8) (`mixture-of-experts.md:48-50`).

논리 흐름 4단계 (`mixture-of-experts.md:53-129`): Gating(라우터가 `scores=einsum(x,W_router)`로 expert별 점수) → Top-K 선택(+softmax로 routing_weights) → Sparse Expert 계산(선택된 expert만 up/down projection) → Combine(K개 출력을 routing_weights로 가중합).

TCP에서의 두 난제와 해법 (`mixture-of-experts.md:138-158`):
- **난제1**: Top-K의 분기문(데이터 값에 따라 실행 경로가 달라짐)이 SIMT 가속기에서 치명적 → **branchless Top-K**(행렬연산+비트조작만).
- **난제2**: 토큰 중심 라우팅은 메모리 접근이 불규칙하고 expert별 토큰 수가 동적 → **expert 중심 + blockwise**(고정 크기 블록).

### 7.1 Branchless TopK (제어 흐름 0)

VE는 256 slice 전체에 동일 명령을 동시에 실행하므로, 주소나 제어가 런타임 값에 의존하면 안 됩니다. 그래서 정렬을 행렬연산으로 바꿉니다 (`mixture-of-experts.md:160-244`):

1. **Bit Packing**: score(상위 16비트)와 index(하위 16비트)를 32비트 하나로 묶음. `Packed = (Score << 16) | Index`. 정렬로 점수 순서가 바뀌어도 expert ID가 따라다니게 하기 위함.
   - **Comparison Trick**: float을 정수로 비교하면 음수 대소가 뒤집히는 문제를, `Packed >= 0이면 그대로, 아니면 Packed ^ 0x7fff0000`으로 비트 플립해서 정수 비교만으로 정확한 Top-K가 되게 함.
2. **Parallel Ranking (All-to-All 비교)**: `Packed_cmp`를 E축으로 타일링해 `T×E×E`로 만들고 모든 쌍 비교: `Compare[t,i,j] = 1 if val[j] > val[i] else 0`. 그 다음 `Rank[t,i] = sum_j Compare[t,i,j]` (나보다 점수 높은 expert 수 = 내 순위). E×E 비교지만 제어흐름이 없어 TCP에 효율적.
3. **Filtering & Unpacking**: `Mask = (Rank < K)`인 것만 FilterCompaction으로 `T×K`로 압축. 그 다음 unpack: `Scores = Packed >> 16`(bf16 재해석), `Indices = Packed & 0xffff`. 마지막에 softmax로 routing_weights.

### 7.2 Blockwise 실행 (정적 shape 유지)

문제: expert별 토큰 수 `L_e`가 동적(최악엔 한 expert에 다 몰려 L_e~T) (`mixture-of-experts.md:250-258`). naive하게 expert마다 T 크기 버퍼를 잡으면 `E×T×D`로 대부분 패딩 낭비. 해법: 고정 크기 블록 B로 관리해 메모리를 `T×K` 수준으로.

- **Grid Size G**: `G = sum_e ceil(Count_e / B)`. 컴파일러가 최악의 G를 계산해 공간을 잡고, 런타임엔 빈 grid는 실행 스킵. 최악은 `(T*K - E)/B + E` (`mixture-of-experts.md:262-274`).
- **주소 계산(cumsum 기반)** (`mixture-of-experts.md:276-324`): 루프 없이 병렬로 각 토큰의 목적지 블록 주소와 블록별 expert ID를 계산.
  - **One-Hot**: `Expert_Mask = one_hot(TopK_Indices, depth=E)` (`T×K×E`).
  - **Histogram**: `Count = reduce_sum(Expert_Mask, axis:(T,K))` — expert별 토큰 수.
  - **Block 수**: `Num_Blocks = ceil(Count / B)`.
  - **Global offset**: `cumsum(Num_Blocks) - Num_Blocks` — 각 expert의 전체 grid 내 시작 블록.
  - **Local offset**: `Cumsum_Mask = cumsum(Expert_Mask)`, `Token_Rank = gather(Cumsum_Mask, TopK_Indices)`, `Local_Offset = Token_Rank - 1` — expert 큐 안에서 내 위치.
  - **Expert ID 확장**: `Diff = Num_Blocks - Block_Range`, `Grid(e,i) = Expert_Indices(e) if Diff>0 else -1`, `Expert_IDs = filter_compaction(Grid, Grid>=0)`.
  - **주소 합성**: `Scatter_Idx = Global_Offset*B + Local_Offset` (∈ [0, G*B)).
- **Dispatch(Scatter)**: `x_norm`을 `Scatter_Idx` 위치로 흩뿌려 `x_blocked: G×B×D` (`mixture-of-experts.md:326-337`).
- **Sparse 계산(Weight Gather)**: `Expert_IDs`로 필요한 expert 가중치만 gather한 뒤 valid 블록(G)만 up/down einsum (`mixture-of-experts.md:339-358`).
- **Combine**: `Scatter_Idx`를 역으로 gather해 원래 토큰 순서로 복원, routing_weights 곱하고 K축 reduce_sum → `moe_out: T×D` (`mixture-of-experts.md:359-378`).

### 7.3 Cumsum on TCP (branch logging)

cumsum도 분기가 없어야 해서 VE의 branch logger로 구현합니다 (`mixture-of-experts.md:379-401`): 축 길이 n에 대해 `branch(i)= 0 if i==0, 1 if i<n-1, 2 else`로 정적 logger를 만들고, `add %mainstream, OperandRead(branch=1,2)` / `WriteOperand(branch=0,1)`로 설정. causal mask, cumsum처럼 "값에 따라 다르게 처리해야 하는" 것들이 전부 branch log로 정적화된다는 게 TCP 프로그래밍의 큰 패턴입니다.

## 8. 한 장으로 묶는 핵심 직관

모든 예제가 같은 질문의 변주입니다. "한 번에 안 들어가니 쪼갠다(타일링/segmentation) → 쪼갠 reduce를 다시 합친다." 합치는 위치가 칩 안이면 split-reduce(interleaved fetch + VE add), 칩/클러스터 밖이면 chip/cluster reduce(거기에 ParallelCopy slice + DMA shuffle 추가). 그리고 데이터 레이아웃은 fetch-commit 경로에서 계산 없이 바꾸되 32바이트 정렬을 노린다. Transformer/MoE는 이 도구들의 조합일 뿐이고, 특히 "값에 의존하는 제어흐름은 전부 정적 행렬연산/branch log로 바꾼다"가 TCP에서 큰 모델을 돌리는 비결입니다.

## 2. 핵심 API · 패턴

| 이름 | 쓰는 법 | 설명 | 출처 |
|---|---|---|---|
| `TuContext::begin` | `ctx.main.begin(dm_tensor_view) -> BeginTensor — fetch/switch/collect/commit 또는 contract 체인의 시작점` | 단일 텐서 텐서유닛 연산 시작. main(스트리밍)과 sub(TRF/VRF 프리페치) 두 컨텍스트가 있다. | `furiosa-opt-std/src/context.rs:88` |
| `TuContext::begin_interleaved` | `ctx.main.begin_interleaved::<I, ...>(lhs_view, rhs_view) -> BeginTensor (I=2 인터리브 시간 차원 생성; lhs->Group0, rhs->Group1)` | split-reduce와 chip/cluster reduce의 덧셈 단계 핵심. 두 텐서를 번갈아 시간축으로 합쳐 VE가 I축을 reduce하게 한다. | `furiosa-opt-std/src/context.rs:98` |
| `vector_intra_slice_unzip` | `.vector_intra_slice_unzip::<I, TileTime, SplitTime>() -> VectorTensorPair — 인터리브 I=2를 Group0/Group1 두 그룹으로 분리` | vector_init() 다음에 호출. 이후 vector_clip_zip으로 두 그룹을 합친다. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:624` |
| `vector_clip_zip` | `.vector_clip_zip(ClipBinaryOpI32::AddFxp) / ClipBinaryOpF32::Add — Group0과 Group1을 op로 합치고 결과를 Group1 자리에 둔다(Way8 필요)` | 그래서 코드 주석 'result is implicitly filtered to Group 1 only'. split/chip/cluster reduce의 실제 덧셈. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor_pair.rs:1385` |
| `vector_inter_slice_reduce` | `.vector_inter_slice_reduce::<OutSlice, OutTime>(InterSliceReduceOpI32::Add / OpF32::Add) — slice 차원을 GAT(global adder tree)로 접고 broadcast` | matmul에서 reduce축(B/64)을 Slice에 뿌린 뒤 이걸로 접는다. broadcast 결과 표현용으로 X=32 더미 축을 둔다. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:512` |
| `parallel_copy_cluster_slice / parallel_copy_chip_slice` | `ctx.sub.parallel_copy_cluster_slice::<CLUSTER_DIM, AxisToSlice, ...>(view, &[slice_idx_per_cluster]) — 비대칭 슬라이스(ParallelCopy/stos)` | ReduceScatter/AllReduce의 대각 슬라이스. &[0,1]=대각, &[1,0]=반대 대각. AxisToSlice는 Element의 가장 바깥 축이어야 한다. | `furiosa-opt-std/src/context.rs:132, furiosa-opt-std/src/context.rs:179` |
| `dm_cluster_shuffle / dm_chip_shuffle` | `view.dm_chip_shuffle::<CHIP_DIM>(&mut ctx.tdma, &[src_per_dst]) -> DmTensor — DMA로 칩/클러스터 간 재분배(ChipShuffle 회전)` | shuffle_pattern[target]=source. 칩 간 이동이라 chip/cluster reduce의 주 비용(수백~수천 cycle). | `furiosa-opt-std/src/tensor/memory.rs:907, furiosa-opt-std/src/tensor/memory.rs:941` |
| `TensorView::tile` | `view.tile::<m![Idx], LEN, m![Element2]>(start) -> TensorView — 매핑식을 따라 텐서를 타일로 분할` | 내부에서 assert_div로 나눠떨어짐을 컴파일타임 검증. 안 맞으면 빌드 실패(매핑 타입 안전성). | `furiosa-opt-std/src/tensor/view.rs:99` |
| `contract 체인` | `.contract_outer::<...>(&trf) .contract_packet::<...>() .contract_time::<...>() .contract_lane::<...>(LaneMode::Sequential/Interleaved)` | einsum의 elementwise multiply + Packet/Time/Lane reduce. 이어서 vector_inter_slice_reduce로 Slice를 접는다. | `furiosa-opt-examples/src/matmul/matmul_split_reduce.rs:88-91` |
| `fetch/switch/collect/commit` | `.fetch::<D, Time, Packet>() .switch::<...>(SwitchConfig::...) .collect::<...>() .commit(addr)` | 데이터 이동·레이아웃 변환 경로. switch는 transpose/broadcast/permute(계산 0), collect는 32바이트 flit 정규화, commit은 write sequencer 설정으로 쓰기. | `furiosa-opt-examples/src/matmul/matmul_16384.rs:21-29` |
| `to_dm / to_hbm / to_trf / to_vrf` | `hbm.to_dm::<Cluster, Slice, Element>(&mut ctx.tdma, addr); ctx.sub.begin(view)...to_trf(TrfAddress::Full); ...to_vrf(0)` | 메모리 계층 간 적재. sub 컨텍스트로 TRF/VRF에 피연산자를 프리페치하면서 main이 스트리밍한다. | `furiosa-opt-examples/src/matmul/matmul_cluster_reduce.rs:20-42` |

## 3. 실험 (직접 돌리기)

> 실험은 NPU 없이 `simulation`·`typecheck`로 돌아갑니다. 실행법은 [`../experiments/README.md`](../experiments/README.md), MNIST는 `cargo furiosa-opt test`(npu 전용).

### 실험 09.1 — split-reduce2 행렬곱을 시뮬레이션으로 실행·검증(유일하게 NPU 없이 통과하는 matmul 테스트)
*난이도 1/5 · 기반: `furiosa-opt-examples/src/matmul/matmul_split_reduce2.rs`*

**목표** — split-K 누적 + interleaved-fetch VE 덧셈 패턴이 호스트 레퍼런스와 일치하는지 확인하고, bf16 허용 오차로 통과하는 모습을 본다.

```bash
cd <furiosa-opt 워크스페이스 루트> && cargo furiosa-opt test --release --test matmul_tests test_matmul_with_split_reduce2 -- --nocapture
```
**관찰** — 테스트가 PASS. 검증 루프가 (diff <= 1.5.max(norm*0.2)) bf16 허용 오차로 비교한다(matmul_tests.rs:146). 다른 matmul 테스트들은 #[ignore="Failing on cpu"]라 이 변형만 시뮬레이션에서 통과한다.

**심화** — matmul_split_reduce2.rs의 for i in 0..4 루프를 0..2로 바꾸면(=K의 절반만 누적) 결과가 레퍼런스와 어긋나 테스트가 실패하는지 예측하고 확인하라. split-K 타일을 빠뜨리면 부분합만 남는다는 걸 체감.

### 실험 09.2 — matmul reduce 4종을 typecheck 백엔드로 컴파일(매핑 대수 검증, NPU/시뮬레이션 불필요)
*난이도 2/5 · 기반: `furiosa-opt-examples/src/matmul/matmul_chip_reduce.rs`*

**목표** — split/cluster/chip reduce 커널이 매핑 타입 검사를 통과하는지 본다. NPU 없이도 mapping 식 정합성을 전부 검증하는 게 vISA의 핵심.

```bash
cd <워크스페이스 루트> && cargo furiosa-opt check --release -p furiosa-opt-examples
```
**관찰** — 에러 없이 통과. tile/fetch/collect/commit/parallel_copy/shuffle의 모든 m![...] 매핑이 나눠떨어짐(assert_div)과 모양 일치를 만족한다는 뜻.

**심화** — matmul_chip_reduce.rs:80의 parallel_copy_chip_slice 인덱스 &[0,1,2,3]를 &[0,1,2]로 줄여보라(CHIP_DIM=4와 배열 길이 불일치). 컴파일 에러가 어디서 나는지 확인.

### 실험 09.3 — chip-reduce의 ChipShuffle 회전 패턴을 깨고 결과 오류 예측
*난이도 3/5 · 기반: `furiosa-opt-examples/src/matmul/matmul_chip_reduce.rs`*

**목표** — ReduceScatter의 대각 슬라이스+회전이 '왜 그 순열이어야 하는지'를 깨뜨려서 이해한다.

```bash
matmul_chip_reduce.rs:94의 dm_chip_shuffle::<4>(&mut ctx.tdma, &[1,2,3,0]) 를 &[0,1,2,3](항등)으로 바꾼 뒤: cargo furiosa-opt check --release -p furiosa-opt-examples (타입은 통과) — 그리고 종이에서 T1이 더 이상 회전 대각이 아니게 되어 어느 칩의 합이 틀어지는지 추적
```
**관찰** — 타입은 통과하지만(셔플은 데이터 이동일 뿐 모양 불변) 논리적으로 chip 0..3의 부분합 조합이 chip-cluster-reduce.md:96-110의 의도와 달라진다. 즉 '컴파일 OK ≠ 수치 정확'.

**심화** — 원복 후 sliced1=&[3,0,1,2]와 shuffle=&[1,2,3,0]이 어떻게 짝을 이뤄 chip 0이 (3,0)을 받게 되는지 chip-cluster-reduce.md:96-110과 1:1로 대응시켜 표로 정리.

### 실험 09.4 — 타일링 수학 직접 계산 — matmul_16384의 타일 수/메모리
*난이도 2/5 · 기반: `furiosa-opt-examples/src/matmul/matmul_16384.rs`*

**목표** — 왜 C는 32타일, B는 4타일인지 용량 제약으로 역산한다(타일 크기가 모든 걸 결정한다는 원리 체득).

```bash
종이/계산기. 입력: A=B=C=16384, i8(1바이트). DM 256MB, TRF 64KB/slice. (1) lhs=A*B, rhs=B*C 각 바이트 수 계산 → 둘 다 256MB라 동시에 DM에 못 올림 확인. (2) C/512=32, B/4096=4 타일 수 확인. (3) rhs 서브타일 (B%4096)*(C%512) 바이트가 왜 TRF에 맞아야 하는지.
```
**관찰** — lhs=rhs=16384*16384=256MB. C를 512씩 자르면 32타일, B를 4096씩 자르면 4타일(matmul_16384.rs:167,180). C부터 자르는 이유는 결과를 바로 HBM에 쓸 수 있어서(matmul_16384.rs:166).

**심화** — B 타일을 2048(=8타일)로 바꾸면 누적 횟수와 부분합 버퍼 핑퐁이 어떻게 늘어나는지, split-reduce.md:338의 'N-way 분할 = N배 대역폭'과 연결해 설명.

### 실험 09.5 — base-template GEMM을 시뮬레이션 실행 + 레퍼런스 테스트
*난이도 1/5 · 기반: `base-template/src/gemm.rs`*

**목표** — 타일링/contraction의 가장 단순한 형태를 NPU 없이 돌려보고 'run'과 'test'의 차이를 익힌다.

```bash
base-template로 프로젝트 생성 후(cargo generate ... base-template) 루트에서: cargo furiosa-opt run --release --bin gemm  그리고  cargo furiosa-opt test --release --bin gemm
```
**관찰** — run은 'GEMM: kernel ran' 출력(gemm.rs:16), test는 내장 #[tokio::test] matches_reference가 bf16 합을 f32로 계산한 레퍼런스와 비교(gemm.rs:20~). 둘 다 시뮬레이션 백엔드로 NPU 없이 동작.

**심화** — gemm_kernel의 K(reduce축) 크기를 키워 VRF 8KB를 넘기면 segmentation/타일링이 필요해지는 지점을 찾아라(tiling.md:24의 4단계와 연결).

### 실험 09.6 — split-reduce의 덧셈 연산 바꿔보고 의미 변화 예측
*난이도 3/5 · 기반: `furiosa-opt-examples/src/matmul/matmul_split_reduce.rs`*

**목표** — vector_clip_zip의 op가 reduce 종류(add/max/min)를 결정함을 확인 — split reduce는 합산만이 아니라 max-pooling 등도 된다(split-reduce.md:261-266).

```bash
matmul_split_reduce.rs:24의 ClipBinaryOpI32::AddFxp 를 다른 enum 변형으로 바꾸려면 먼저: grep -n 'enum ClipBinaryOpI32' -A40 furiosa-opt-std/src/engine/vector/op/mod.rs 로 가용 op 확인 → 적용 후 cargo furiosa-opt check --release -p furiosa-opt-examples
```
**관찰** — AddFxp 대신 다른 op로 바꾸면 누적이 더 이상 행렬곱 합산이 아니게 되어 test_matmul_with_split_reduce(있다면)가 레퍼런스와 불일치. enum은 op/mod.rs:603에 정의.

**심화** — split-reduce.md Example 4(temporal max)처럼 max 계열 op로 split reduce를 구성하면 어떤 reduce 의미가 되는지 적어보라.

## 4. 연습문제 (손으로, 컴파일 없이)

**Q1.** split-reduce.md Example 1에서 Hidden=8192를 4096짜리 두 청크로 나눠 LayerNorm 통계를 낸다. 청크별 평균이 sum_0/4096, sum_1/4096일 때 전체 평균을 청크 평균만으로 어떻게 구하는가? (interleaved reduce_add 후 어떤 상수를 곱하는지 포함)

<details><summary>정답/힌트</summary>

total_sum = reduce_add(I=2)로 sum_0+sum_1을 구한 뒤 mean = total_sum * (1.0/8192). 청크 평균을 단순 평균내는 게 아니라 전체 sum을 8192로 나눈다(split-reduce.md:107-112).

</details>

**Q2.** matmul_chip_reduce.rs에서 4칩 ReduceScatter의 T1을 만들 때 parallel_copy_chip_slice(&[3,0,1,2]) 다음 dm_chip_shuffle::<4>(&[1,2,3,0])을 한다. 셔플 후 chip 0이 받는 (A,B) 원소는 무엇인가?

<details><summary>정답/힌트</summary>

slice [3,0,1,2]로 chip0=(0,3),chip1=(1,0),chip2=(2,1),chip3=(3,2) 선택 → shuffle[target]=source, [1,2,3,0]이므로 chip0은 source chip1의 (1,0)을 받는다. chip-cluster-reduce.md:96-110과 일치.

</details>

**Q3.** fetch-commit-engine.md Example 3에서 데이터가 A=65바이트일 때 패딩 dummy=23과 dummy=31 중 무엇이 빠르고 왜인가? cycle 수로 답하라.

<details><summary>정답/힌트</summary>

dummy=31이 6 cycle(32바이트 fetch), dummy=23은 22 cycle(8바이트 fetch). 65+31=96은 32로 나눠떨어져 32바이트 fetch 가능하지만 65+23=88은 32 배수가 아니라 write가 경계를 넘으면 안 돼서 8바이트로 떨어진다(fetch-commit-engine.md:353-395).

</details>

**Q4.** Transformer prefill에서 scores=einsum(Q_rope,K_rope)는 어떤 축을 reduce하고 어떤 축을 broadcast하는가? GQA 관점에서 답하라.

<details><summary>정답/힌트</summary>

d_k 축을 reduce, G(=8) 축을 K_rope에서 broadcast한다. shape![B,G,h_kv,s_in_q,d_k] x shape![B,h_kv,s_in_k,d_k] -> shape![B,G,h_kv,s_in_q,s_in_k](transformer.md:177-179).

</details>

**Q5.** MoE branchless TopK에서 Packed_Value를 Rank로 바꾼 뒤, Top-K를 어떻게 고르고, 왜 unpack에는 comparison-trick 적용 전 원본 Packed_Value를 써야 하는가?

<details><summary>정답/힌트</summary>

Mask=(Rank<K)로 FilterCompaction해 T×K로 압축. comparison trick(^0x7fff0000)은 비교용으로 비트를 바꿨으므로 score/index 복원에는 trick 적용 전 원본 Packed_Value를 써야 정확하다(mixture-of-experts.md:222-240).

</details>

**Q6.** matmul_split_reduce.rs는 reduce를 result0/result1/temp 세 버퍼로 핑퐁한다. 왜 단일 누산 레지스터를 쓰지 않는가? VE zip 메커니즘으로 설명하라.

<details><summary>정답/힌트</summary>

VE 덧셈은 두 입력(Group0/Group1)을 interleaved fetch로 받아 결과를 Group1 자리에 내는 구조라, 새 부분곱(temp)을 이전 누적과 더하려면 두 텐서가 동시에 존재해야 한다. 그래서 in-place 누산 대신 이전 결과↔새 결과를 번갈아 쓰는 더블버퍼가 필요(vector_tensor_pair.rs:1385, matmul_split_reduce.rs:113-134).

</details>

## 5. 흔한 함정

- 8KB는 slice당 VRF 한도이지 전체가 아니다. 텐서를 slice 차원에 분산하지 않으면(m![1,1,1,2048,32]처럼) 모든 데이터가 slice 0에 몰려 8KB를 초과해 segmentation이 강제된다. slice 분산을 줘도 reduce처럼 한 slice가 행 전체를 들어야 하면 여전히 터질 수 있다.  
  ↳ 출처 `docs/src/kernel-examples/fetch-commit-engine.md:429-435`
- 패딩을 아무 값(예: 23바이트)으로 주면 8바이트 fetch로 떨어져 22 cycle, 32 정렬(31바이트)이면 6 cycle. 메모리 몇 바이트 아끼려다 4배 느려진다. write sequencer는 경계를 못 넘으므로 데이터+패딩이 32 배수가 아니면 32바이트 commit 자체가 불가하다.  
  ↳ 출처 `docs/src/kernel-examples/fetch-commit-engine.md:353-395`
- ChipShuffle/cluster shuffle 패턴은 데이터 이동일 뿐 텐서 모양을 바꾸지 않아 잘못된 순열을 넣어도 cargo furiosa-opt check(타입검사)는 통과한다. 즉 '컴파일 OK'가 '수치 정확'을 보장하지 않는다 — ReduceScatter 대각/회전 순열은 손으로 검증해야 한다.  
  ↳ 출처 `docs/src/kernel-examples/chip-cluster-reduce.md:96-135`
- split reduce의 비용은 fetch가 지배해 N-way 분할이면 메모리 대역폭이 N배다. 분할 수를 늘려 잘게 쪼개는 건 비싸다 — 각 텐서를 VRF 한도까지 키워 분할 수를 최소화하고 2/4/8 같은 2의 거듭제곱을 써라.  
  ↳ 출처 `docs/src/kernel-examples/split-reduce.md:338-376`
- vector_clip_zip은 결과를 Group1 자리에만 둔다('implicitly filtered to Group 1 only'). 그리고 Way8 모드가 필요하다. 이 동작을 모르면 인터리브 덧셈 결과를 어디서 읽어야 할지 헷갈린다.  
  ↳ 출처 `furiosa-opt-std/src/engine/vector/tensor/vector_tensor_pair.rs:1385`
- tile은 컴파일타임 assert_div로 매핑이 나눠떨어지는지 검사한다. 타일 분할이 산술적으로 안 맞거나 parallel_copy의 const 차원과 인덱스 배열 길이가 다르면 런타임이 아니라 빌드에서 실패한다.  
  ↳ 출처 `furiosa-opt-std/src/tensor/view.rs:99`
- examples 크레이트의 matmul 테스트 대부분은 #[ignore="Failing on cpu"] 또는 "takes too much time"이라 시뮬레이션에서 안 돈다. NPU 없이 실제로 실행·검증되는 matmul 테스트는 test_matmul_with_split_reduce2 하나뿐이다. 나머지는 cargo furiosa-opt check(typecheck)로만 검증하라.  
  ↳ 출처 `furiosa-opt-examples/tests/matmul_tests.rs:14,37,87,158`
- reduce 축을 Chip/Cluster에 두면 DMA 셔플(수백~수천 cycle)이 끼어 매우 비싸다. 가능하면 reduce 축은 Slice/Element에 배치해 GAT/VE로 칩 안에서 끝내라. matmul_16384가 B/64를 Slice에 둔 이유.  
  ↳ 출처 `docs/src/kernel-examples/chip-cluster-reduce.md:314`

## 6. 핵심 정리 & 다음

기억할 사실:
- DM(데이터 메모리, on-chip SRAM)은 전체 256MB이지만 slice당 한도는 512KB다. 큰 텐서 타일링의 1차 제약. (`docs/src/quick-start.md:68`)
- VRF(Vector Engine 피연산자 레지스터 파일)는 slice당 8KB. 이 한도를 넘으면 split-reduce/segmentation이 필요하다(bf16 기준 약 4096개). (`docs/src/quick-start.md:71, docs/src/kernel-examples/tiling.md:7`)
- TRF(Contraction Engine 입력)는 lane당 8KB, slice당 8 lane이라 full 모드 64KB/slice, half 모드 32KB/slice. matmul에서 rhs를 작은 조각으로 자르는 이유가 이 TRF 용량 때문. (`docs/src/quick-start.md:70, furiosa-opt-examples/src/matmul/matmul_16384.rs:52`)
- 클러스터당 slice는 256개라 VRF 총합 2MB지만, 8KB 한도는 slice당이지 전체가 아니다. slice 차원에 분산을 안 주면 모든 데이터가 slice 0에 몰려 8KB를 넘긴다. (`docs/src/kernel-examples/fetch-commit-engine.md:429-435`)
- flit은 32바이트. fetch/packet 정렬은 8바이트 단위이고, 가능한 fetch_size는 8/16/24/32바이트뿐이다. (`docs/src/kernel-examples/fetch-commit-engine.md:64, fetch-commit-engine.md:120, fetch-commit-engine.md:255`)
- 한 클러스터 안에서 65,536개 원소를 스칼라로 reduce하는 데 약 296 cycle(Packet Reducer 트리 깊이5=5, Time Reducer 8회=8, Inter-Slice Reducer 256). 칩 간 통신 없이 끝낼 수 있다. (`docs/src/kernel-examples/chip-cluster-reduce.md:13-34`)
- AllReduce는 n칩에서 회전 ChipShuffle n-1번이면 된다(4칩=3, 8칩=7, 16칩=15). reduce 축은 가능하면 Slice/Element에 두는 게 좋다(칩 간 통신 회피). (`docs/src/kernel-examples/chip-cluster-reduce.md:312-314`)
- write sequencer는 텐서 경계를 넘어 쓰면 안 되지만(남의 메모리 침범), read sequencer는 dummy 영역까지 overfetch해도 안전하다. 이 비대칭 때문에 패딩이 32로 안 나눠떨어지면 32바이트 commit이 불가하다. (`docs/src/kernel-examples/fetch-commit-engine.md:394`)

➡️ 다음: [10_real_workloads.md](./10_real_workloads.md)
