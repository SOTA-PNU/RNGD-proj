# 05 · 축약 엔진 (Contraction Engine / DPE, systolic)

> **1줄 요약**: 축약 엔진은 두 텐서를 공유 축으로 축약(matmul·conv)하는 하드웨어로, `contract_outer → contract_packet → contract_time → contract_lane` 4단계가 각각 **Outer(브로드캐스트·곱)**, **Packet Reducer(트리 축약)**, **Time Reducer(누산 슬롯)**, **Lane Folder(Lane 접기)** 라는 물리 파이프라인 단계 하나씩에 정확히 대응한다. bf16 을 먹고 f32 로 확장 누산하며, 이 bf16 입력 양자화가 `dpe_result.md` 의 ~0.23% 오차의 근원이다.

**대응 책 섹션** (모두 `/home/jun/.claude/jobs/46bc5c7e/tmp/docs/` 아래):
- `computing-tensors__contraction-engine__index.md` — 4단계 개요·매핑 전략(K/M/V in Time)
- `computing-tensors__contraction-engine__outer.md` — Stream Adapter / TRF Sequencer / Multiplier
- `computing-tensors__contraction-engine__packet-reducer.md` — lane당 reduction tree
- `computing-tensors__contraction-engine__time-reducer.md` — temporal accumulator 슬롯
- `computing-tensors__contraction-engine__lane-folder.md` — Interleaved / Sequential
- `computing-tensors__contraction-engine__2d-convolution.md` — conv = matmul + Stream Adapter 확장

**뒷받침 소스**: `~/.cargo/registry/src/index.crates.io-*/furiosa-opt-std-0.4.0/src/engine/contraction/{mod,packet,time,lane}.rs`, `.../contraction/outer/{mod,stream_adapter,trf_sequencer}.rs` (이하 `contraction/*.rs`).
**우리 코드**: `/home/jun/yik/src/kernel/gemm_kernel.rs`.
**대조 커리큘럼**: `.../vISA/curriculum/06_computing_engines_1.md` (v0.2.0 기준).
**실기(real NPU) 실측**: 7-3 절 · 2절 `lane_size` 실기 함정 · 6절 TRF 용량 → 전수는 [13-NPU-실기-매트릭스](./13-NPU-실기-매트릭스.md), 게이팅 절차는 [12-예제-전수실행](./12-예제-전수실행.md).

---

## 0. 용어 정리 — "DPE / systolic" 는 책 용어가 아니다

| 부르는 곳 | 이름 | 근거 |
|---|---|---|
| furiosa-opt 책 | **Contraction Engine** | `contraction-engine__index.md:1` |
| 사용자 런타임·벤치 | **DPE**, 구어로 "systolic array" | `dpe_result.md:1,2,20` ("systolic/DPE-MAC engine") |
| GROUND_TRUTH | **Contraction(DPE)** | `_GROUND_TRUTH.md:50` |

- 책은 이 하드웨어를 **트리 축약 + 시간 누산기** 파이프라인으로 기술한다(고정 기능 MAC 배열, `2d-convolution.md:417`). 고전적 2D systolic wavefront 로 명시하지 않는다. "systolic" 은 사용자·벤더의 구어이므로 이 문서는 **책 표현(Contraction Engine)** 을 기준으로 쓰고, 필요할 때만 DPE 라 부른다. (엄밀히 systolic array 인지 여부는 **미확인** — 책의 서술은 tree+accumulator.)
- DPE 약자 확장(예: Dot Product Engine)은 GROUND_TRUTH·책 어디에도 정의가 없어 쓰지 않는다. **미확인**.

---

## 1. 4단계 파이프라인 — API ↔ 하드웨어 ↔ 차원

책이 뭐라 하나(`index.md:15-62`): 작업을 4단계로 인수분해하되 **Broadcast·Multiply 가 1단계, Reduce 가 3단계**. 각 단계는 겹치지 않는 자기 차원만 만진다.

```
Collect ─┐
         ├─►[ Outer: Stream Adapter + TRF Sequencer → Multiply ]─► Packet Reducer ─► Time Reducer ─► Lane Folder ─► Vector
TRF ─────┘
```
(`index.md:18-52`)

| API (호출 순서) | 하드웨어 단계 | 하는 일 | 만지는 차원 | 소스 |
|---|---|---|---|---|
| `.contract_outer(&trf)` | **Outer** | 두 피연산자 브로드캐스트 후 원소곱(+타입 확장) | `Time`/`Packet` → `Lane`/`OutTime`/`OutPacket` | `contraction/outer/mod.rs:120` |
| `.contract_packet::<OutPacket>()` | **Packet Reducer** | Packet 안 축약 축을 lane당 트리로 합 | `Packet` → `OutPacket` | `contraction/packet.rs:35` |
| `.contract_time::<OutTime>()` | **Time Reducer** | Time 걸쳐 누산기로 누적 | `Time` → `OutTime` | `contraction/time.rs:20` |
| `.contract_lane::<OutTime,OutPacket>(mode)` | **Lane Folder** | Lane 을 Packet/Time 으로 접기(합산 아님) | `Lane` 제거 | `contraction/lane.rs:51` |

- 공간 차원 `Chip`/`Cluster`/`Slice` 는 4단계 모두 **그대로 통과**. slice·chip 간 축약이 더 필요하면 다운스트림 **Vector 엔진**이 처리(`index.md:60`).
- **핵심 하드웨어 상한**: Outer 가 `Lane ≤ 8`, `Packet ≤ 64 B` 로 제한(RNGD) — `index.md:62`, `_GROUND_TRUTH.md:49`.
- 지연 예산: 4단계 + Inter-Slice Reducer 로 65,536 → 1 스칼라가 ~296 cycle(`index.md:64`).

### 초심자용: 왜 4개로 쪼갰나
matmul `C[i,j] = Σ_k A[i,k]·B[k,j]` 를 (1) **곱**을 만들고 (2) **k 를 합**하는 두 일로 보면, 축약 축 k 가 하드웨어 좌표계 `[Lane, Time, Packet]` 어디에 놓이든 합할 수 있어야 한다. 그래서 합을 세 곳에서 나눠 한다: **Packet 안(공간, 트리)**, **Time 걸쳐(시간, 누산기)**, 그리고 마지막에 **Lane 은 합이 아니라 재배치(fold)**. 곱은 맨 앞 Outer 가 담당. 이 분업이 정확히 4개의 `.contract_*` 호출이다.

### 함정
- 4단계는 **지연(deferred)** 실행이다. `contract_outer` 는 곱을 실제로 만들지 않고 두 피연산자를 `LazyContraction` 에 넣어 두고, Packet/Time reducer 는 그것을 **그대로 통과**시키며, `contract_lane` 에서 **단 한 번** `Backend::contraction` 으로 융합한다(`contraction/mod.rs:10-13`, `lane.rs:78-89`). 그래서 넓은 출력 GEMM 도 `[.., Lane, ..]` 모양 중간 외적을 메모리에 절대 물질화하지 않는다(`outer/mod.rs:33-41`). 즉 `contract_packet`/`contract_time` 은 "검증 + 타입 재라벨"만 한다(`packet.rs:39-41`, `time.rs:24-26`).

---

## 2. Outer — 외적의 하드웨어 구현 (`.contract_outer`)

책(`outer.md:1-14`): 이름은 선형대수 **외적** `u vᵀ` 에서 왔다. u 를 열축으로, v 를 행축으로 브로드캐스트해 원소곱하면 외적 행렬이 나온다. Outer 의 세 하위 단계가 이 의미를 직렬로 구현한다.

시그니처(**0.4**, 제네릭 **5개**):
```rust
// contraction/outer/mod.rs:120-123
pub fn contract_outer<OutTime: M, OutPacket: M, Lane: M, TrfElement: M, TrfD>(
    self,
    trf_tensor: &TrfTensor<TrfD, Chip, Cluster, Slice, Lane, TrfElement, B>,
) -> ContractOuterTensor<'l, T, <D as ContractionCast>::Output, D, Chip, Cluster, Slice, Lane, OutTime, OutPacket, B>
```
- 스트리밍 피연산자(Collect 에서)는 `self`, TRF 피연산자(가중치)는 `&trf_tensor`. `Lane`/`TrfElement`/`TrfD` 는 보통 `trf` 타입에서 추론 → 턴피시로 `OutTime`, `OutPacket` 만 주면 된다.
- **0.3 은 제네릭 4개**(`OutTime, OutPacket, Lane, TrfElement`), 0.4 는 `TrfD` 가 추가된 5개. `_GROUND_TRUTH.md:22`.

### 2-1. Stream Adapter — Packing + Broadcast
스트리밍 `Time`/`Packet` 을 연산 모양으로 바꾼다(`outer.md:33-42`). 컴파일러가 세 자유변수 `PackSize`, `LaneBroadcast`, `TimeBroadcast` 를 유도:
```text
Lane      = LaneBroadcast
OutTime   = [Time / PackSize, TimeBroadcast]
OutPacket = [Time % PackSize, Packet] # (64 / D::SIZE)
```
- **Packing**: Collect 는 32 B flit 을 내고, Outer 는 `PackSize×32` B 패킷을 낸다. `PackSize ∈ {1,2}` 개 flit 을 한 패킷으로 합침. **`PackSize = OutPacket::SIZE × D::SIZE / 32`** (`outer.md:55`). 즉 사용자가 OutPacket 을 64 B 로 고르면 PackSize=2, 32 B 면 PackSize=1. 하드웨어는 항상 내부적으로 64 B 패킷; PackSize=1 이면 남는 32 B 절반은 0 곱(논리 OutPacket 엔 전파 안 됨) → `outer.md:57`.
- **Broadcast**: packing 뒤 공간적으로 `LaneBroadcast`(TRF 의 Lane, ∈{1,2,4,8}), 시간적으로 `TimeBroadcast`(TRF `Element` 중 입력 Time 에 없는 인자 + 순수 출력 축)로 복제(`outer.md:59-71`). TimeBroadcast 인자는 **OutTime 의 가장 안쪽**에 놓인다.
- 검증만 하고 실제 확장은 안 함(`stream_adapter.rs:21` → `config_stream_adapter`). "확장을 물질화하진 않지만 검증 계약은 지킨다"(`stream_adapter.rs:4-7`).

### 2-2. TRF Sequencer — ReadSize 로 Element 펼치기
`TrfTensor` 를 읽어 `Element` 를 `OutTime`/`OutPacket` 으로 reshape(`outer.md:123-141`):
```text
OutTime   = ([Element / ReadSize] 시퀀싱 + 브로드캐스트)
OutPacket = [PacketBroadcast, Element % ReadSize]
```
- 매 cycle **한 번의 TRF 읽기가 OutPacket 하나**를 채움: 양 bank = lane당 640 bit, 한 bank = 320 bit(`outer.md:135`, `_GROUND_TRUTH.md:51`).
- 컴파일러는 `Element % ReadSize == OutPacket % ReadSize` 이고 `ReadSize × D::SIZE ≤ 64` B 인 **가장 큰 ReadSize** 를 고름(`outer.md:138`). 넓으면 양 bank, 좁으면 한 bank.
- **주의**: TRF 쪽 검증(`verify_trf_sequencer`)은 **0.4 에서 아직 TODO 스텁**이다 — 스트림 쪽만 검증하고 TRF 쪽 매핑 확장은 미검증(`trf_sequencer.rs:15-19`).
- **실측 보정(2026-07-24, 실기 컴파일)**: eDSL 스텁과 별개로 **`--backend npu` 컴파일의 mir 단계가 TRF 매핑 오류를 잡는다.** `contract_outer_assertions` 의 오류 표본을 실제 컴파일해 받은 에러 원문:
  - `trf_mapping::invalid_lane_mapping` → ``mir: `to_trf` lane mismatch: time_outer != Lane: A != E / 4``
  - `trf_mapping::invalid_mapping` → ``mir: `to_trf` element mismatch: [time_inner, Packet] != Element: B != (A, C)``
  - `trf_mapping::invalid_lane_not_divisible_by_time` / `trf_lane_time::invalid_lane_exceeds_time` → `mir: Lane::SIZE (4) does not divide Time::SIZE (6)` / `(2)`
  - `cpacket_mapping::invalid_mapping` → ``mir: `contract_outer`: inner flit of OutPacket must equal the input Packet``
- **단, 걸리는 시점이 `npu` 컴파일뿐이다.** `typecheck`/`emulation` 빌드는 이 표본들을 그대로 통과시킨다(두 백엔드는 **호출되는 커널만** 처리하므로 아무도 부르지 않는 `invalid_*` 표본은 그냥 지나간다 — `12-예제-전수실행.md` §5.3. 반면 `--backend npu` 는 크레이트의 모든 `#[device]` 를 빌드 시점에 낮춰서 죽는다). **타입체크·에뮬레이션 통과 ≠ 실기 로우어링 성공.** → [12-예제-전수실행](./12-예제-전수실행.md), [13-NPU-실기-매트릭스](./13-NPU-실기-매트릭스.md)

### 2-3. Multiplier — 타입 확장 후 곱 (ContractionCast)
정렬된 두 피연산자를 받아 **각 원소를 축약 출력 타입으로 확장** 후 원소곱(`outer.md:216-220`):
```text
i4 / i8  → i32
f8 / bf16 → f32
```
- 확장은 **Outer 진입 시점**에 일어난다: `contract_outer` 가 storage 타입 `D` 피연산자를 `<D as ContractionCast>::Output` 으로 캐스트해 `LazyContraction` 에 저장(`outer/mod.rs:8-12,132,161-167`). 다운스트림 전 단계가 누산기 폭(f32)을 그대로 들고 가고, 좁히기는 소비자가 다음에 붙이는 `cast` 에서 일어남.
- 출력은 `[Chip,Cluster,Slice,Lane,Time,Packet]` 단일 텐서. 매 Time cycle 마다 모든 Lane 이 패킷 하나 분량 곱을 병렬 산출(`outer.md:220`).
- **실기 확인(i8→i32)**: `contract_element_types_answer::answer_i8_contract` 가 실기 매트릭스에서 **PASS**. 이 테스트는 `out[a,r] = Σ_k input[a,k]·trf[r,k]` 를 i32 로 손계산한 오라클과 **완전 일치**를 요구한다(`tests/contract_element_types_answer_tests.rs:28-38`, `assert_eq!` 로 Vec 비교). 상류는 이걸 시뮬레이션 정답지로 썼지만 **실기 빌드에서도 그대로 통과**한다 → i8 확장 축약은 값까지 맞다. (bf16 경로의 오차는 9절.)

### 함정
- **`OutPacket::SIZE × Storage::SIZE ∈ {32,64}` B** — 여기서 Storage 는 **확장 전** dtype(bf16=2 B), f32 누산기가 아님(`outer.md:105`, `outer/mod.rs:86-87`). 코드가 `assert_packet_one_or_two_flit::<Storage, Packet>()` 로 검사.
- `Lane::SIZE ∈ {1,2,4,8}` (`outer.md:107`). 컴파일러도 같은 값을 강제한다 — 실측 에러 원문 `mir: Lane::SIZE must be 1, 2, 4, or 8, got 3` / `got 16`(`lane_size::invalid_size_{3,16}`).
- **성능**: PackSize=2 는 64 B 를 다 채워 MAC 전부 사용, PackSize=1 은 32 B 만 채워 절반 낭비(`outer.md:111-113`).

### 함정(실기) — `lane_size` 4형제 중 **8만** 로우어링된다 (2026-07-24 실측)

`contract_outer_assertions::lane_size` 는 Lane=1/2/4/8 을 각각 "valid" 표본으로 갖는다. **`--backend npu` 컴파일에서는 8만 성공하고 1·2·4 는 전부 실패한다.** 이름이 `valid_*` 라고 믿으면 틀린다.

| 커널 | Lane | 실기 컴파일 | 에러(원문) |
|---|--:|---|---|
| `lane_size::valid_size_1` | 1 | **FAIL** | `lir: incorrect buffer size at T7: DramShape { inner: []\|[A_1=8:8] }: buffer.size() (256) != num_chips * intra_chip_size (228)` |
| `lane_size::valid_size_2` | 2 | **FAIL** | 같은 `lir` 에러, `(232)` |
| `lane_size::valid_size_4` | 4 | **FAIL** | 같은 `lir` 에러, `(240)` |
| `lane_size::valid_size_8` | 8 | **OK** | (실기 테스트 `test_valid_size_8` 도 PASS) |

- 네 커널은 Lane 크기와 거기 딸린 출력 타입만 다르다 — 출력 `HbmTensor<i32, Chip, ..>` 의 매핑이 각각 `m![A, 1 # 8]` / `m![A, R / 4 # 8]` / `m![A, R / 2 # 8]` / `m![A, R # 8]`(A=8, R=8, `type Lane = m![R]`), 물리 크기는 어느 쪽이든 8행 × 8슬롯 × 4 B = **256 B**(`src/contract_outer_assertions.rs:29,63,97,131`).
- 컴파일러가 센 크기 228/232/240 은 **7×32 + (4/8/16)** — **마지막 레인 그룹의 꼬리 패딩만 빠졌다**(산술 정황). 부분 충전 `# 8` 그룹의 DRAM 크기 계산 결함이며, 같은 서명이 `switch_assertions::alignment::aligned_fetch_packet_i4`(i4, 256 vs 240)에도 나온다.
- **Lane<8 자체가 금지된 건 아니다**: `mnist` 의 두 FC 층은 TRF `Lane = m![1]` + `contract_lane::<m![1], m![1 # 8]>(Interleaved)` 로 **실기에서 돈다**(아래 7절 실기 현황). 막히는 건 **부분 충전 레인 그룹을 그대로 HBM 출력 타입에 노출하는 패턴**이다.
  (`mnist::forward` 의 HBM 출력은 `m![C]`=16 으로 패딩 인자가 아예 없다 — `src/mnist/mod.rs:189`.)
- 정리: **부분 충전 `# 8` 레인 그룹이 HBM 출력 타입에 남아 있으면 현재 실기로 못 간다.** 꽉 채우면 통과하는지는 표본이 `valid_size_8` 하나뿐이라 **미확인**. 전수 표: [13-NPU-실기-매트릭스](./13-NPU-실기-매트릭스.md)

---

## 3. Packet Reducer — lane당 reduction tree (`.contract_packet`)

책(`packet-reducer.md:1-9`): 각 lane 이 32/64 B 패킷을 받아 **Packet 안**의 축약 축을 더한다. lane당 독립 트리 하나.
```text
ReducePacket = Packet / 2^d       (0 ≤ d ≤ log2(Packet::SIZE))
OutPacket    = ReducePacket        (ReducePacket::SIZE ≤ 32 이면)
             = 32 로 클립           (아니면)
```
(`packet-reducer.md:45-49`)
- depth 0 = 잎이 패킷 원소, 매 depth 마다 쌍을 더해 원소 절반. **최대 깊이 = log2(Packet::SIZE)**: i4 는 7(128원소), i8/f8 은 6(64), **bf16 은 5(32)** (`packet-reducer.md:53`).
- 컴파일러가 사용자의 `OutPacket` 에서 깊이 d 를 유도, 안쪽 `2^d` 원소를 소비.
- **왜 OutPacket ≤ 32 원소?** 다운스트림 Time Reducer 의 lane당 누산기 열이 **32개**뿐(`packet-reducer.md:56`). 코드로도 `assert_packet_pow2_within_accumulator_cols` — "2의 거듭제곱이고 ≤ 32"(`contraction/mod.rs:60-68`, `TEMPORAL_ACCUMULATOR_COLS`).

### 함정
- 검증은 **Storage(확장 전) 바이트**로 패킷 크기를 잰다(`packet.rs:49-51`) — DPE 는 storage 폭 flit 을 읽고 나서 확장하니까. 누산기 폭(f32)으로 재면 틀린다.
- **성능**: 지연 = 트리 깊이(bf16 5 cycle)지만 완전 파이프라인이라 정상 처리량은 매 cycle 1패킷 in/out. `Lane<8` 이면 노는 lane 트리만큼 처리량 `Lane::SIZE/8` 로 감소(`packet-reducer.md:64-68`).

---

## 4. Time Reducer — 누산기 슬롯 (`.contract_time`)

책(`time-reducer.md:1-9`): Packet Reducer 의 `[Lane, Packet]` 출력을 `Time` 에 걸쳐 `OutTime` 으로 누적(temporal accumulator). `OutTime` 이 살아남는 Time 차원을 지명, 나머지는 합산.
- 제약: **`OutTime` 은 `Time` 의 부분집합, 살아남는 차원의 상대 순서 보존**(`verify_contract_time`, `time-reducer.md:58`, `time.rs:33-34`). 소스 테스트가 **안쪽 축 축소(`valid_reduce_inner`)와 바깥 축 축소(`valid_reduce_outer`) 둘 다 valid** 임을 보인다(`time.rs:52-62`) — 즉 reduce 축이 Time 의 안/밖 어디든 가능.
- 핵심 개념 **InnerTime** = 가장 바깥 reduce 차원보다 **안쪽**이면서 OutTime 에 살아남는 차원(`time-reducer.md:61`). 누적은 **InnerTime::SIZE 개 슬롯**을 쓴다(같은 InnerTime 튜플의 flit 이 같은 슬롯에 누적).

**슬롯 용량**(버퍼 1,024 cell, LaneMode 가 청크 모양 결정) — `time-reducer.md:92-100`:

| LaneMode | 청크 모양 | 청크당 cell | 슬롯 용량 |
|---|---|---|---|
| Interleaved | `[Lane # 8, Packet]` | `8 × Packet::SIZE` | `128 / Packet::SIZE` |
| Sequential | `[Lane, Packet # 32]` | `Lane::SIZE × 32` | `32 / Lane::SIZE` |

- `InnerTime::SIZE` 가 슬롯 용량을 넘으면 Time 을 더 쪼개거나 LaneMode 를 바꿔 처리량↔슬롯을 맞바꾼다.

### 함정
- Time Reducer 는 **contract_lane 의 LaneMode 를 미리 알아야** 슬롯 용량을 안다. 그래서 `contract_time` 은 pre-reduce Time 매핑을 저장해 두고(`time.rs:26`, `contraction/mod.rs:116-118`), `contract_lane` 이 그것으로 슬롯 경계를 검사한다(`lane.rs:60-61`). 즉 두 단계가 교차 검증된다.

---

## 5. Lane Folder — Lane 접기 (`.contract_lane`)

책(`lane-folder.md:1-11`): Contraction 의 **마지막** 단계. `Lane` 의 8개 값을 **합산이 아니라 재배치**로 없앤다. Time Reducer 버퍼를 8원소 폭 버스로 매 cycle 하나씩 비우며, `LaneMode` 가 각 flit 이 뭘 담을지 정함.
```rust
// contraction/lane.rs:17-22
pub enum LaneMode { Interleaved, Sequential }
```

| 모드 | Lane 이 접히는 곳 | 매핑 | 언제 |
|---|---|---|---|
| **Interleaved** | `OutPacket` | `OutTime=[Time,Packet]`, `OutPacket=[Lane # 8]` | 매 cycle 8 lane 의 한 열(값 8개) — 출력 채널을 lane 에 편 matmul |
| **Sequential** | `OutTime` | `PadPacket=Packet#align_up(SIZE,8)`, `OutTime=[Time,Lane,PadPacket/8]`, `OutPacket=[PadPacket%8]` | 매 cycle 한 lane 의 8열 — Lane=1 인 reduce 류 |

(`lane-folder.md:19-26, 57-65`)

- **Lane Folder 자체엔 제약 없음.** 여기서 고른 LaneMode 가 **업스트림 Time Reducer 의 슬롯 용량 한계를 결정**한다(`lane-folder.md:98-100`).
- 여기서 비로소 융합 축약이 실행됨: `Backend::contraction(lhs, rhs, ..., pre_reduce, out)` 한 번(`lane.rs:78-89`).

### 함정
- Interleaved 는 `Lane<8` 이면 빈 버스 칸만큼 처리량 `Lane::SIZE/8` 감소. Sequential 은 `Packet::SIZE<8` 이면 매 cycle 딱 그만큼만 실어 패딩·낭비 없음(`lane-folder.md:105-107`).

---

## 6. 제약 한눈에

| 제약 | 값 | 출처 |
|---|---|---|
| `Lane::SIZE` | ∈ {1,2,4,8}, ≤ 8 | `outer.md:107`, `index.md:62` |
| `Packet` | ≤ 64 B (bf16 32원소) | `index.md:62`, `_GROUND_TRUTH.md:49` |
| `OutPacket::SIZE × Storage::SIZE` | ∈ {32,64} B | `outer.md:105`, `outer/mod.rs:86` |
| `PackSize` | ∈ {1,2} = OutPacket·D/32 | `outer.md:55` |
| Packet Reducer 최대 트리 깊이 | log2(Packet::SIZE): i4=7, i8/f8=6, bf16=5 | `packet-reducer.md:53` |
| contract_packet OutPacket | 2의 거듭제곱, ≤ 32 (누산기 32열) | `contraction/mod.rs:60-68`, `packet-reducer.md:56` |
| Time Reducer 누산 버퍼 | 1,024 cell | `time-reducer.md:94` |
| TRF 읽기 대역폭 | narrow 320 / wide 640 bit·lane·cycle | `outer.md:135` |
| **TRF 용량 `TrfAddress::Full`** | **65,536 B = 8 lane × 8 KB** | **실기 컴파일 에러 실측**: `mir: TRF data (524288 bytes = 8 lanes x 65536 bytes) exceeds register file capacity (65536 bytes)` (`trf_size::invalid_to_trf_full`) |
| **TRF 용량 `TrfAddress::FirstHalf`** | **32,768 B = 4 KB/lane** | **실기 컴파일 에러 실측**: `mir: TRF data (65536 bytes = 8 lanes x 8192 bytes) exceeds register file capacity (32768 bytes)` (`trf_size::invalid_to_trf_half`) |

> 위 두 줄이 이 문서의 옛 미확인("TRF 8 KB/lane vs 10 KB/lane")을 닫는다. 컴파일러가 강제하는 값은 **전체 65,536 B / 8 lane = 8 KB/lane** 이고, `FirstHalf` 는 그 절반(4 KB/lane)이다. 커리큘럼 인용의 80 KB/slice(=10 KB/lane)는 이 값과 맞지 않는다.

---

## 7. 우리 `gemm_kernel.rs` 완전 해부

파일: `/home/jun/yik/src/kernel/gemm_kernel.rs`. `axes![I=512, J=512, K=64]` (`gemm_kernel.rs:3`).

> **m! 표기 주의**: `J / 8 % 4` 등은 512 를 나눈 **정수 산술이 아니라** J-인덱스의 **혼합기수(mixed-radix) 인자 크기**다. 즉 `J % 8`=크기 8 인자, `(J/8) % 4`=크기 4 인자, `J / 32`=크기 16 인자. 곱은 8·4·16 = 512.

### 7-0. 축 분해 (검증됨)

| 논리 축 | 슬라이스로 | Lane 으로 | Time 으로 | Packet 으로 | 검산 |
|---|---|---|---|---|---|
| I=512 | `I/32`=16 | — | `I%32`=32 | — | 16·32=512 |
| J=512 | `J/32`=16 | `J%8`=8 | `J/8%4`=4 | — | 16·4·8=512 |
| K=64 | — | — | `K/32`=2 | `K%32`=32 | 2·32=64 |

- `Slice = m![I/32, J/32]` = 16×16 = **256 슬라이스**, 각 슬라이스가 담당하는 출력 타일 = `m![I%32, J%32]` = **32×32** (`gemm_kernel.rs:8`).
- 검산: 256 슬라이스 × 1024(32×32) = 262,144 = I·J. ✅
- **함정(문서 오류)**: `gemm_kernel.rs:8` 주석 "Each slice handles a 16 × 16 output tile" 은 **틀렸다**. 16×16 은 슬라이스 **격자**(256개)이고, 타일은 32×32 다. `_GROUND_TRUTH.md:32-34`.
- **함정**: `gemm_kernel.rs:23-24` 주석 "Switch Engine distributes..." 도 오해다. 이 커널엔 `.switch()` 호출이 없고, 분배는 `to_dm` 목표 타입의 `Slice` 파라미터가 한다. `_GROUND_TRUTH.md:35`.

### 7-1. TRF 적재 (가중치 B, sub 컨텍스트) — `gemm_kernel.rs:25-30`
```rust
let b_trf: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![J / 8 % 4, K]> = ctx.sub
    .begin(b.view())
    .fetch::<m![J % 8, J / 8 % 4], m![K]>()
    .collect::<m![J % 8, J / 8 % 4, K / 16], m![K % 16]>()
    .to_trf(TrfAddress::Full);
```
- TRF: `Lane = m![J%8]` = 8 (출력 열/채널), `Element = m![J/8%4, K]` = 4×64 = **256 bf16 = 512 B/lane**. TRF 예산(8 KB/lane) 안에 여유롭게 들어감 — 이 8 KB/lane 은 컴파일러가 실제로 강제하는 값으로 실측 확인됐다(6절 표).
- bf16 64원소 = 128 B = 4 flit → collect 가 `K/16`=4 를 Time 으로 흡수, `K%16`=16(32 B)을 Packet 으로 (Multi-Flit 케이스, `06_computing_engines_1.md:161`).

### 7-2. 그 4줄 — 실제 매핑 인자로 (`gemm_kernel.rs:40-43`)

```rust
.contract_outer::<m![I % 32, J / 8 % 4, K / 32], m![K % 32], _, _>(&b_trf)  // (1)
.contract_packet::<m![1]>()                                                 // (2)
.contract_time::<m![I % 32, J / 8 % 4]>()                                   // (3)
.contract_lane::<m![I % 32, J / 8 % 4], m![J % 8]>(LaneMode::Interleaved)    // (4)
```
입력(main, 행렬 A): `.collect::<m![I%32, J/8%4, K/16], m![K%16]>()` → `Time=[32,4,4]`, `Packet=[16]`.

**(1) `contract_outer` — Outer**

| 산출 | 값 | 의미 |
|---|---|---|
| `OutTime` | `[I%32, J/8%4, K/32]` = [32,4,2] | 출력행 32 × 출력열묶음 4 × K시간 2 |
| `OutPacket` | `[K%32]` = 32 (bf16 = **64 B**) | K 32개를 한 패킷에 (공간 축약 예정) |
| `Lane` | `[J%8]` = 8 | 출력 채널 8개를 8 lane 에 (b_trf 에서 추론) |
| `PackSize` | `32×2/32` = **2** | 64 B 꽉 채움 → **MAC 100% 활용** |

- Stream Adapter: A 의 collect `Packet=[K%16=16]` 에 Time 안쪽 인자 하나(크기 2)를 Packing 으로 흡수 → 32원소 64 B 패킷. A 는 `Lane=J%8` 과 TRF-only 축 `J/8%4` 로 **broadcast**(A 는 J·채널에 무관하므로 복제).
- TRF Sequencer: Element `[J/8%4, K]` 에서 ReadSize=32(K 32개, 양 bank) 읽어 OutPacket 채움, 스트림 전용 축 `I%32` 는 broadcast.
- Multiplier: **bf16 × bf16 → f32** 확장(`outer/mod.rs:120-131`). ← 9절의 0.23% 오차 지점.

**(2) `contract_packet::<m![1]>` — Packet Reducer**
- Packet=`[K%32]`=32 bf16 → 트리 **깊이 log2(32)=5** → 1. K 의 32개를 **공간 트리**로 합. OutPacket=1. (`packet-reducer.md:53`)

**(3) `contract_time::<m![I%32, J/8%4]>` — Time Reducer**
- `Time=[I%32, J/8%4, K/32]`=[32,4,2] → `OutTime=[I%32, J/8%4]`=[32,4]. **남은 K 인자 `K/32`=2 를 시간 누적**.
- 그래서 전체 K 축약 = **Packet 트리(32) × Time 누적(2) = 64 = K**. ✅
- reduce 축 `K/32` 가 Time 의 **가장 안쪽** → InnerTime = 없음 → 슬롯 1개만 필요(Interleaved·Packet=1 이면 용량 128, 여유). 안쪽 축소는 `time.rs:52-56` 의 `valid_reduce_inner` 가 valid 로 보증.

**(4) `contract_lane::<m![I%32, J/8%4], m![J%8]>(Interleaved)` — Lane Folder**
- `Lane=J%8`=8 → **OutPacket 으로 접힘**(Interleaved). `OutTime=[I%32, J/8%4]`=[32,4], `OutPacket=[J%8]`=8.
- 슬라이스당 셀 = 32×4×8 = **1024 = 32×32 타일**. ✅ (Time flit 총 256개, lane당 출력 128개.)

이후 `.cast::<bf16, m![J%8#16]>().commit_trim::<m![J%8]>().commit(0)` → `DmTensor<bf16, .., m![I%32, J%32]>` = **32×32 타일/슬라이스** (`gemm_kernel.rs:44-46`). 타일 크기가 코드 자체로 32×32 임이 재확인된다(주석 16×16 반박).

> **주의(미확인)**: `Cluster = m![1 # 2]`(`gemm_kernel.rs:6`)는 논리 1, 하드웨어 2 의 **dummy 패딩**이다. 출력 전체가 256 슬라이스(한 cluster 분)에 이미 다 들어가므로, 2번째 cluster 는 구별되는 데이터를 안 가진다(중복/유휴). chip 은 2 cluster×256 = 512 슬라이스이므로 이 커널은 공간 셀의 절반만 구별 사용하는 셈으로 보인다 — 정확한 활용률은 컴파일러 덤프로 확인 필요. **미확인**. (패딩이라는 관용구 자체는 벤더 예제 주석 12곳으로 확인됨 — 미확인 2번. 단 그 주석은 벤더 예제 것이고 이 커널의 의도는 별개다.)

### 커리큘럼 대조 (`06_computing_engines_1.md:349-356`)
- 커리큘럼도 같은 결론: **축약 축 K 는 Packet 에, 출력 축은 spatial(Slice/Lane)+Time 에**. 우리 gemm 이 정확히 그 권장 매핑(K 를 packet 에, 출력 채널을 Lane=J%8 에)이다. "K in Time" 는 32 곱셈기 중 1개만 일하는 **1/32 MAC 퇴화 예제**(`index.md:88`)이며, 우리 커널은 그 반대로 K 를 Packet(트리)에 둬 PackSize=2 로 MAC 을 꽉 채운다.

### 7-3. 실기 현황 — 축약 커널이 실제로 NPU 에서 도는가 (2026-07-24 실측)

> **읽는 법**: PASS/FAIL 은 **실기 실행 결과**(실제 NPU 에서 커널을 돌린 것), "로우어링 실패"는 `--backend npu` **컴파일** 결과, 사이클 수치는 **컴파일러 스케줄 모델 예측**이다. 섞어 읽지 말 것. 전수 표는 [13-NPU-실기-매트릭스](./13-NPU-실기-매트릭스.md).

- **`contract_outer_assertions` 실기 13/13 PASS.** 실기 빌드에 올라간 valid 테스트 13개가 전부 통과(cpacket_mapping 2, cpacket_size 2, lane_size 1, time_broadcast 4, trf_mapping 2, trf_size 2). 테스트 파일의 valid 표본은 **16개**이고 `lane_size::valid_size_{1,2,4}` 는 로우어링 실패로 게이팅되어 실행조차 되지 않았다(2절). **실기로 확인된 것은 Outer 단계의 매핑 계약**이다 — 13개 표본 모두 뒤 3단계는 `contract_packet::<m![1]>` / `contract_time::<m![A]>` / `contract_lane(Interleaved)` 라는 자명한 인자로만 지나가므로, 3~5절의 트리 깊이·1,024 cell·Sequential 슬롯 용량 값은 여전히 **책 기반**이다.
- **end-to-end 축약이 실기에서 검증된 예제는 `mnist` 하나뿐.** `mnist_tests::test_mnist` PASS(이미지 10장 전부 정답). 매핑은 `contract_outer::<m![X/32], m![X%32], _,_,_>(&input_trf)` → `contract_packet::<m![1]>` → `contract_time::<m![1]>` → `contract_lane::<m![1], m![1 # 8]>(Interleaved)` (`src/mnist/mod.rs:27-30, 110-113`). 즉 **가중치가 아니라 활성값을 TRF 에 올리고 Lane 은 1 만 쓴다** — 우리 gemm(가중치 TRF, Lane=8, K 를 Packet)과 정반대 배치다.
- **벤더 `matmul` 예제 7개는 전부 실기 로우어링 실패**: `matmul_16384`, `matmul_4096`, `matmul_chip_reduce`, `matmul_cluster_reduce`, `matmul_with_split_reduce`, `matmul_with_split_reduce2`, `matmul_wo_broadcast`. 축약 전후 매핑이 어긋나 mir 에서 죽는 사례:
  - `matmul_wo_broadcast` → `mir: Collect time mismatch. Expected: A / 4 % 4, got: A / 4`
  - `matmul_with_split_reduce` → `mir: commit_trim packet mismatch. Expected A % 4 # 8 or a trimming of it, got A % 8`
  - 나머지 5개는 실패 지점이 서로 다르다(한 계열로 묶으면 트리아지가 틀린다): `matmul_16384` → `visa: Branch conversion is not yet implemented`(호스트 `for`/`if` 타일링), `matmul_chip_reduce` → `strides([8,128,4,...]) is not aligned by 8`, `matmul_cluster_reduce` → `tail_size % min_align (1) != 0`, `matmul_4096` → `visa: verification of operator failed: IndexAccess ... must have axis with given tag`(`slice_tile` 에필로그), `matmul_with_split_reduce2` → **컴파일러 ICE**(`mir: internal compiler error: ... is used before definition` — 매핑 오류가 아니라 컴파일러 자체 버그) → [12-예제-전수실행](./12-예제-전수실행.md)
  → **실기에서 돌아가는 참조 matmul 예제는 없다.** 새 축약 매핑의 기준으로 쓸 수 있는 것은 `mnist` 와 `contract_outer_assertions` 의 valid 표본뿐이다.
- **스케줄 모델 예측**(`compile --dump-schedule`, NPU 점유 없음 → 실기 테스트와 동시 실행 가능) — `lane_size::valid_size_8`: span **4,119** cycle / 7 inst, DmaEngine **1,234** · PeCore **600** · SubContext 327 · MainContext 282. PeCore 는 덤프한 130 커널 중 **102개에서 600 cycle 로 동일**하고 전체 사이클의 **96.5% 가 DmaEngine** 이다 → 축약 매핑을 잘 잡아도 **비용은 DMA 쪽에 있다**([13-NPU-실기-매트릭스](./13-NPU-실기-매트릭스.md)).

---

## 8. conv = matmul (2D Convolution)

책(`2d-convolution.md:4-8`): 2D conv 는 einsum `$(H+Fh)$(W+Fw)K, FhFwKC → HWC`. 축약 축 = `Fh, Fw, K`, 출력 = `H, W, C`. **축약 코어(Outer→Packet→Time→Lane)는 matmul 과 동일**; 다른 건 **Stream Adapter 가 sliding-window 재사용 확장 3종을 더 갖는다**는 점뿐이다(einsum 만 쓰면 이 절은 건너뛰어도 됨, `2d-convolution.md:88`).

> **실기 커버리지 0 (실측)**: 이 절은 전부 책 기반이다. 벤더 예제 크레이트에는 **convolution 커널이 한 개도 없고**(`src/` 전체에 `convolution`/`conv2d` 없음), 실기 커널 매트릭스 200개에도 conv 는 0건이다. 즉 여기 적힌 한계값은 **실기로 확인되지 않았다** — [13-NPU-실기-매트릭스](./13-NPU-실기-매트릭스.md).

Stream Adapter 확장(`2d-convolution.md:83-99, 148-165`):

| 확장 | 파라미터 | 한계 | 물리 원인 |
|---|---|---|---|
| Flit Buffer | `feed_flits ∈ {1,2,3}` | 96 B 물리 버퍼(레지스터 파일) | `2d-convolution.md:400,408` |
| Transpose | 32 B flit 내, bf16 `[2][8]→[8][2]` 등 | 축약 축을 안쪽으로 | `2d-convolution.md:110-116` |
| Shift(Stream Shift Unit) | `initial_shift`/`shift_stride`/`pop_dim` | bf16 shift [-3,4], stride 0-7 | 레지스터 체인 길이 |

- 왜 conv 가 matmul 로 환원되나: 슬라이딩 윈도우가 만드는 겹치는 입력을 **매번 re-fetch 하지 않고 shift 로 재생성**하면, Packet Reducer 트리 입장에선 그냥 "축약 축이 안쪽에 정렬된 패킷"이 될 뿐이다. 그래서 conv 4변형(Filter-Stride 1/2, Dilation 2, Stride2+Dilation2)은 **Stream Adapter 의 shift 설정 차이**로만 갈리고, 어느 축을 Time 에 둘지는 matmul 과 **같은 트레이드오프**(`2d-convolution.md:8`).
- **함정**: Outer 곱 + Packet Reducer 트리는 **버퍼링·재정렬이 안 되는 고정 기능 MAC 배열**이라, 매핑이 어긋나면 성능 저하가 아니라 **틀린 계산**이 나온다. Stream Adapter 출력은 TRF Sequencer contraction 매핑과 반드시 일치해야 함(`2d-convolution.md:403,417-418`).

---

## 9. bf16 → f32 확장 누산과 ~0.23% 오차 (dpe_result 연결)

**사실 연쇄**:
1. 축약 엔진(DPE)이 **먹을 수 있는 입력 타입은 i4/i8/f8/bf16 뿐** — f32 는 못 먹는다. f32 는 Vector 엔진(VE) 몫(`_GROUND_TRUTH.md:50`).
2. Multiplier 는 곱 **전에** bf16 을 f32 로 확장하고, 축약(Packet 트리·Time 누산)을 **f32 로** 수행한다(ContractionCast, `outer.md:216-218`, `outer/mod.rs:120-131`). → **누산 자체의 손실은 막힌다**.
3. 그런데 **입력 피연산자 x·W 는 이미 bf16 으로 반올림**되어 배열에 들어간다(`dpe_result.md:34-38`: "the compiler feeds the systolic array in bf16 (`dpe_element_type = trf_element_type = Bfloat16`)"). 우리 커널도 `a,b: HbmTensor<bf16, ..>`(`gemm_kernel.rs:14-15`)로 bf16 입력.
4. 그래서 남는 오차는 **누산이 아니라 bf16 입력 양자화**다. bf16 은 가수 7비트(~2^-8) → 원소당 ~0.4%, 축약 평균 후 **relmean ~0.23%**.

`dpe_result.md` 실측(rngd:2, torch F.linear 대조):

| 커널 | shape (T,I,O) | maxabs | relmean | allclose@1e-2 | @1e-3 |
|---|---|---|---|---|---|
| **DPE** | 128,512,2048 | 1.30e-3 | **0.23%** | True | False |
| **DPE** | 256,2048,512 | 2.35e-3 | **0.23%** | True | False |
| **VE** | 128,512,2048 | 2.98e-7 | 0.00% | True | True |

(`dpe_result.md:25-32`)

- DPE 는 **1e-2 에서 일치, 1e-3 에선 불일치** — "빠른 경로의 대가"(`dpe_result.md:34-40`). VE 는 f32 유지라 ~1e-7 정확하지만 **DPE 가 end-to-end ~1.96x, compute-only ~3.8x 빠름**(`dpe_result.md:48-61`).
- **결론(사용자 실무)**: attention QK·projection 처럼 ~0.23% 를 허용하는 matmul 은 DPE 로, f32-정확 축약이 필요하면 VE 로(`dpe_result.md:65-70`).

> 정리: **"bf16→f32 확장"은 누산 손실을 막는 장치이지, 오차의 원인이 아니다.** 오차는 그 앞단, 피연산자를 bf16 으로 표현한 데서 온다. 이 구분을 뭉개면 안 된다.

---

## 10. 0.3 vs 0.4 API·커리큘럼 대조 (함정 모음)

우리 `yik` 프로젝트는 **`furiosa-opt-std = "0.3"` 핀**이고 `src/furiosa-opt.tag` 마커를 쓴다(`/home/jun/yik/Cargo.toml`). 반면 설치된 래퍼는 **0.4 세대**(`_GROUND_TRUTH.md:4-8`). 그래서 `gemm_kernel.rs` 는 **현재 래퍼로 커널 패키지로 인식되지 않는다**("no kernel packages found", `_GROUND_TRUTH.md:10,14-16`). 아래는 이 문서의 4단계와 직접 관련된 0.3↔0.4 차이:

| 항목 | 0.3 (yik 코드) | 0.4 (설치 크레이트) | 근거 |
|---|---|---|---|
| `contract_outer` 제네릭 | 4개 `<OutTime,OutPacket,_,_>` | **5개** `<..,_,_,_>`(TrfD 추가) | `_GROUND_TRUTH.md:22`, `outer/mod.rs:120` |
| `to_dm`/`to_trf`/`commit` 주소 | 명시(`to_dm(&mut ctx.tdma, 0)`) | 주소 인자 **제거** | `_GROUND_TRUTH.md:23`, `gemm_kernel.rs:18-19,46` |
| 백엔드 이름 | `simulation` (Cargo lints) | **없음**; typecheck/emulation/npu | `_GROUND_TRUTH.md:8`, `yik/Cargo.toml` |
| 커널 마커 | `furiosa-opt.tag` 0바이트 | `[package.metadata.furiosa-opt]` | `_GROUND_TRUTH.md:9-10` |

- 커리큘럼(`06_computing_engines_1.md`)은 **v0.2.0** 기준이라 `--backend simulation` 언급(실험 06.1 의 `cargo furiosa-opt run --bin gemm`)이 현재 래퍼에서 **안 통한다**(`_GROUND_TRUTH.md:28-29`). 4단계 API 설명 자체(`06_computing_engines_1.md:241-347`)는 0.4 소스와 **일치**하므로 개념 참고용으로 유효하다.
- **실기 미실행은 이제 "이 커널 한정"이다**(구 "NPU 금지" 항목 정정): 실기(real NPU)는 이 서버에서 열려 있고, 벤더 예제 크레이트로 **실기 테스트 89개 중 80개 통과**를 실제로 찍었다([13-NPU-실기-매트릭스](./13-NPU-실기-매트릭스.md)). 다만 `gemm_kernel.rs` 는 위 표의 0.3 핀·tag 마커 때문에 **현재 래퍼로 컴파일 자체가 안 되므로**, 7절 매핑 수치는 여전히 책 공식 + 소스 **해석**이고 컴파일러 덤프로 교차검증되지 않았다(아래 미확인 1).

---

## 미확인으로 남긴 것

1. **`gemm_kernel.rs` 덤프 미실시**: 7절의 매핑 인자·PackSize·타일 크기는 책 공식(`outer.md:55` 등) + 소스로부터의 해석이다. 덤프 자체는 이 세션에 **벤더 예제 130개 커널로 실시했지만**(7-3), yik 은 0.3 API·tag 마커라 현재 0.4 래퍼로 컴파일 불가 → **이 커널**의 스케줄 대조는 미확인(포팅 필요).
2. **Cluster=`m![1#2]` 활용률**: 패딩이라는 **의도**는 확인됐다 — 예제 소스 12곳이 `type Cluster = m![1 # 2]; // 1 logical cluster, padded to 2 (hardware has 2 clusters/chip)` 로 못박아 둔다(예: `src/transformer/embedding/rms_norm.rs:15`). 2번째 cluster 가 유휴인지 중복인지, 공간 셀 절반만 쓰는지는 여전히 미확인(7-2 주의 참조).
3. **"systolic" 여부**: 책은 tree+accumulator 로 기술; 고전 systolic array 인지 명시 없음. DPE 약자 확장도 미정의.
4. **TRF Sequencer 검증 범위**: 0.4 의 `verify_trf_sequencer` 는 여전히 TODO 스텁(`trf_sequencer.rs:15-19`). **mir 단계가 TRF 매핑 오류를 잡는다는 것은 실측 확인**(2-2)했으나, mir 가 실제로 커버하는 범위와 스텁이 놓치는 케이스는 `contract_outer_assertions` 오류 표본 밖에서 검증하지 않았다.
5. **부분 충전 레인 그룹**: `lane_size::valid_size_{1,2,4}` 를 막는 `lir` 크기 오산(2절 실기 함정)의 회피 조건 — HBM 출력을 꽉 찬 레인 그룹으로 잡으면 통과하는지는 표본이 없어 미확인.

> 해결된 항목: 구 4번 **"TRF 용량 8 KB/lane vs 10 KB/lane"** → 컴파일러가 강제하는 값을 실측해 6절 표로 이동(Full 65,536 B = 8 KB/lane, FirstHalf 32,768 B = 4 KB/lane).
