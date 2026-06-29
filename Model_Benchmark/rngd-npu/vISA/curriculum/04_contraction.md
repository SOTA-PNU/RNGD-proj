# 04 · 텐서 축약(Contraction)

이 문서는 vISA 커리큘럼 모듈 04입니다. dot product·GEMV·GEMM으로 Contraction 엔진(`contract_outer/packet/time/lane`)과 TRF, Switch 브로드캐스트, 출력 타일을 슬라이스에 분산하는 법을 배웁니다.
*선행: 03 원소별 연산 · 예상 시간: 반나절*

## 학습 목표

- [ ] contract_outer→packet→time→lane 각 단계가 무엇을 줄이는지 안다
- [ ] TRF에 정지 피연산자를 싣는 이유(.to_trf)를 안다
- [ ] GEMV/GEMM에서 Switch 브로드캐스트가 왜 필요한지 안다
- [ ] dot_product·gemv·gemm을 돌리고 수치 검증한다

## 1. 개념

## 큰 그림: "축약(contraction)"이 곧 RNGD의 본업입니다

RNGD 칩의 정식 이름이 TCP, 즉 Tensor **Contraction** Processor입니다. 칩 설계 자체가 "두 텐서를 곱해서 공유 축(공통 축)을 따라 합치는" 연산을 빠르게 하려고 만들어졌어요. 우리가 매일 쓰는 행렬곱(matmul), 행렬·벡터 곱(GEMV), 내적(dot product)이 전부 이 한 가지 패턴의 변종입니다. 책에서는 이걸 세 단계로 쪼갭니다(docs/src/quick-start.md:30): **Broadcast(맞춰 늘리기) → Multiply(원소별 곱) → Reduce(합치기)**.

einsum 표기로 보면 한눈에 들어옵니다(quick-start.md:35-39):
- 내적: `I, I → 1` (브로드캐스트 없음, 두 축이 이미 같음)
- GEMV: `IJ, J → I` (벡터 x 를 I 방향으로 늘려 곱하고 J를 합침)
- GEMM: `IK, JK → IJ` (A는 J 방향, B는 I 방향으로 늘려 곱하고 K를 합침)

여기서 **"합쳐서 사라지는 축"이 contraction 축**입니다. 내적은 I, GEMV는 J, GEMM은 K가 사라집니다. vISA 커널을 짤 때 가장 먼저 정해야 하는 게 "어떤 축을 어디서(공간으로? 시간으로?) 줄일 것인가"예요. 이게 성능을 좌우합니다.

## 하드웨어 계층과 매핑 — 커널이 "어디서" 도는지

축약을 이해하려면 칩의 4단 계층을 먼저 머리에 넣어야 합니다(quick-start.md:47-52):

- **Chip**: 최상위. HBM(48GB, 1.5TB/s)을 가집니다.
- **Cluster**: 칩당 2개. 각 클러스터가 256개 슬라이스를 묶습니다.
- **Slice**: 클러스터당 256개. 슬라이스 하나가 Tensor Unit 하나를 굴립니다. 슬라이스들은 기본적으로 **각자 자기 DM 조각에서 독립적으로** 돕니다.
- **Lane**: 슬라이스당 8개. Contraction Engine MAC 배열의 한 행입니다.

vISA의 핵심 아이디어는 이 계층을 **Rust 타입으로 노출**한다는 점입니다. 예를 들어 `DmTensor<bf16, m![1], m![1 # 2], m![A / 8 # 256], m![A % 8]>` 는 "bf16 벡터를 칩 1개, 클러스터 2개 중 1개, 256슬라이스에 분산(슬라이스당 A/8개 인덱스), 슬라이스 안에 8개씩"이라는 뜻입니다(quick-start.md:82). `m![]` 안의 세 연산자만 알면 됩니다(quick-start.md:85-88):
- `/` 는 stride로 쪼갬: `A / 8` → 2048/8 = 256개 (보통 슬라이스 개수)
- `%` 는 안쪽 개수: `A % 8` → 슬라이스 안 8개
- `#` 는 하드웨어 단위로 패딩: `# 256` → 256슬라이스에 맞춰 채움(남는 칸은 쓰레기값)

그리고 파이프라인을 흐르는 텐서에는 두 가지 특수 축이 더 붙습니다(quick-start.md:90): **`Time`(파이프라인 반복 횟수, 시간축)** 과 **`Packet`(한 번에 처리하는 원소 묶음, 공간축)**. 축약에서 "K를 Packet에 두면 공간 병렬로 한 번에 줄이고, Time에 두면 사이클마다 누적"하는 식으로 갈립니다. 이게 04 모듈에서 제일 중요한 직관입니다.

## Tensor Unit 파이프라인과 두 개의 컨텍스트

Tensor Unit은 고정 파이프라인입니다(quick-start.md:57): **Fetch → Switch → Collect → Contraction → Vector → Cast → Transpose → Commit**. 축약 커널은 이 중 Collect까지 데이터를 흘려보낸 뒤 Contraction Engine을 두드리고, 마지막에 Cast로 타입을 되돌려 Commit합니다.

커널은 동시에 도는 두 컨텍스트를 씁니다(quick-start.md:96-99):
- `ctx.main`: 실제 축약 계산이 흐르는 곳.
- `ctx.sub`: 한 피연산자를 미리 TRF(또는 VRF)에 적재해 두는 프리페치 라인. main이 필요로 할 때 sub가 아직 못 끝냈으면 **자동으로 기다려** 동기화합니다.

이게 축약의 비대칭성과 직결됩니다. 축약은 피연산자 둘 중 **하나는 파이프라인으로 흘려보내고(streaming), 다른 하나는 TRF에 가만히 세워둡니다(stationary)**. 흐르는 쪽은 Collect Engine에서 오고, 세워둔 쪽은 sub 컨텍스트의 `.to_trf()`가 미리 채워 놓습니다(contraction-engine/index.md:9-11).

## Contraction Engine 4단 — 축약이 실제로 줄어드는 곳

Contraction Engine은 네 단으로 나뉩니다(contraction-engine/index.md:54-60). 하나는 곱하기, 셋은 줄이기입니다. 커널 체인의 `.contract_outer → .contract_packet → .contract_time → .contract_lane` 네 호출이 정확히 이 네 단입니다.

**1) Outer (`.contract_outer(&trf)`) — Broadcast + Multiply**
두 피연산자를 공통 모양 `[Chip, Cluster, Slice, Lane, Time, Packet]`으로 맞춰 늘린 뒤 원소별로 곱합니다(outer.md:1-14). 내부적으로 세 부품:
- **Stream Adapter**: 흐르는 피연산자를 처리. Collect가 32바이트 flit를 주면, `PackSize ∈ {1,2}`개를 묶어 32B 또는 64B 패킷을 만듭니다(이게 "Packing", outer.md:44-57). PackSize=2면 64B를 꽉 채워 MAC 100% 활용, PackSize=1이면 절반(32B)이 0으로 패딩돼 throughput 절반(outer.md:105-107). dot_product 커널 주석 "Pair consecutive 32-byte flits into 64-byte packets, halving time steps (A/16 → A/32)"가 바로 PackSize=2입니다.
- **TRF Sequencer**: 세워둔(TRF) 피연산자를 매 사이클 64B씩 읽어 같은 모양으로 늘립니다(outer.md:117-135).
- **Multiplier**: 두 피연산자를 곱하면서 **출력 타입을 넓힙니다**: `i4/i8 → i32`, `f8/bf16 → f32`(outer.md:12,200). 누적 중 오버플로를 막으려는 거예요. 그래서 bf16 행렬곱의 누산기는 f32입니다.

타입 시그니처는 `.contract_outer::<OutTime, OutPacket, _, _>(&trf)` 형태입니다(furiosa-opt-std/src/engine/contraction/outer/mod.rs:74). 뒤 두 제네릭(`Lane`, `TrfElement`)은 TRF 텐서에서 추론되므로 `_, _`로 둡니다. 앞 두 개 `OutTime`/`OutPacket`이 "곱한 결과를 어떤 Time/Packet 모양으로 내보낼지"를 정합니다.

**2) Packet Reducer (`.contract_packet::<OutPacket>()`) — Packet 안에서 공간 합산**
한 패킷 안의 contraction 축을 **레인마다 독립적인 덧셈 트리**로 한 방에 줄입니다(packet-reducer.md:1-9). 수식으로 `output[i] = Σ_j input[i, j]` — 여기서 `i`는 살아남을 출력 축, `j`는 패킷 안에서 줄일 축입니다. 그래서 타입 제네릭 `OutPacket`은 **살아남는 부분**을 적습니다. 전부 다 줄이면 `m![1]`(dot_product, gemm). 일부만 남기면 그걸 적습니다(matmul_4096는 `.contract_packet::<m![A % 2]>()` — B%32은 줄이고 A%2는 패킷에 남김). 트리 깊이는 타입별로 다릅니다: bf16 5단(32원소), i8/f8 6단(64), i4 7단(128)(packet-reducer.md:47,58). 깊이는 첫 출력 지연만 늘릴 뿐, 파이프라인이 차면 매 사이클 1패킷 throughput은 유지됩니다.

**3) Time Reducer (`.contract_time::<OutTime>()`) — Time축 누적**
Packet Reducer가 매 사이클 내놓는 `[Lane, Packet]`을 시간축(Time)을 따라 누산기에 더합니다(time-reducer.md:1-7). `OutTime`에 **살아남을 Time 차원**을 적고, 나머지는 합쳐서 사라집니다. dot_product/gemv는 `m![1]`(전부 누적해 스칼라/행 결과), gemm은 `m![I # ...]`처럼 출력 축을 남깁니다. 누산기는 1,024셀이고 `LaneMode`에 따라 슬롯 용량이 정해집니다(time-reducer.md:88-94) — Interleaved면 `128/Packet::SIZE`, Sequential이면 `32/Lane::SIZE` 슬롯. `InnerTime::SIZE`가 이 용량을 넘으면 빌드가 막혀서 Time을 더 쪼개거나 LaneMode를 바꿔야 합니다.

**4) Lane Folder (`.contract_lane::<OutTime, OutPacket>(mode)`) — Lane을 접기**
마지막 단. 8개 레인을 **합치는 게 아니라 다른 축으로 옮겨 접습니다**(lane-folder.md:1-6). 두 모드(furiosa-opt-std/src/engine/contraction/lane.rs:21-26):
- `LaneMode::Interleaved`: Lane을 `OutPacket`의 가장 안쪽으로 넣음 → 매 사이클 8레인의 같은 열 위치 1개씩, 8값/flit. 출력에 여러 출력채널을 나란히 둘 때 씁니다(gemm, gemv, dot_product 전부 Interleaved).
- `LaneMode::Sequential`: Lane을 `OutTime`으로 넣음 → 한 레인의 패킷을 8원소씩 여러 사이클에 걸쳐 뽑음. matmul 예제들(matmul_4096, split_reduce)이 Sequential을 씁니다.

**중요 제약**: Outer 단에서 `Lane ≤ 8`, `Packet ≤ 64B`로 막혀 있습니다(contraction-engine/index.md:62). 그래서 한 슬라이스가 한 번에 다룰 수 있는 공간 병렬은 8레인 × 64B로 한정됩니다. 그보다 큰 K는 Packet(공간) + Time(시간)으로 쪼개야 합니다.

## Cast로 되돌리기 — f32 누산기 → bf16

Multiplier가 bf16을 f32로 넓혀 누적했으니, HBM에 다시 쓰기 전에 `.cast::<bf16, OutPacket>()`로 줄여야 합니다(furiosa-opt-std/src/engine/cast.rs:38). Cast Engine은 **출력 패킷을 반드시 32바이트(1 flit)로 패딩**합니다(cast.rs:48-52). bf16은 16비트라 32바이트 = 16개 → `m![1 # 16]`. 그래서 dot_product가 `.cast::<bf16, m![1 # 16]>()`를 쓰는 거예요. 여기서 `m![1 # 8]`(8개=16바이트)로 잘못 쓰면 "output packet must be exactly 32 bytes" 단언에 걸립니다. 이건 좋은 실험 소재입니다(아래 실험 참고).

## TRF와 `.to_trf()` / `TrfAddress::Full`

TRF(Tensor Register File)는 Contraction Engine 전용 슬라이스별 SRAM입니다. 구조는 **8레인 × 2뱅크 × 128행 × 320비트 = 슬라이스당 80KB**(register-files.md:72). `.to_trf::<Lane, Element>(address)`가 Collect 스트림을 TRF에 적재합니다(furiosa-opt-std/src/engine/collect.rs:56). 흐르는 Time/Packet을 `Lane`(공간 병렬, 1/2/4/8)과 `Element`(레인별 레이아웃)로 재배치합니다(register-files.md:27-34). matmul에서는 보통 `Lane`이 출력채널, `Element`가 contraction 축을 담습니다.

주소 인자는 `TrfAddress` enum이고 세 가지입니다(furiosa-opt-std/src/tensor/memory.rs:101-108):
- `Full`: 128행 전부, 용량 65,536바이트(memory.rs:116). dot_product/gemv/gemm이 모두 `Full`을 써서 TRF 전체를 한 텐서에 줍니다.
- `FirstHalf`(0–63행) / `SecondHalf`(64–127행): 각 32,768바이트. **더블 버퍼링**용입니다 — 한 반쪽을 읽는 동안 다른 반쪽을 채워 main/sub를 겹쳐 돌립니다(register-files.md:90-96). matmul_split_reduce2가 루프 안에서 `FirstHalf`/`SecondHalf`를 번갈아 쓰는 게 그 예입니다.

## Switch 브로드캐스트 — GEMV·GEMM이 슬라이스를 가로질러야 하는 이유

내적은 모든 슬라이스가 **같은 축**을 줄이므로 슬라이스 간 데이터 이동이 필요 없습니다(dot_product가 Switch를 안 씀). 그런데 GEMV `IJ, J → I`는 출력 I를 슬라이스에 분산(`Slice = m![I]`)하므로, **각 슬라이스가 자기 행과 곱할 벡터 전체를 받아야** 합니다. 그래서 벡터를 모든 I 슬라이스로 뿌리는 브로드캐스트가 필요하고, 그걸 **Switch Engine**이 합니다(quick-start.md:223-226). Switch Engine은 256슬라이스 링 네트워크로 슬라이스 간에 데이터를 옮기는 유일한 엔진입니다(switch-engine.md:3).

브로드캐스트 설정은 `SwitchConfig::Broadcast01`입니다(furiosa-opt-std/src/engine/switch.rs:35-42, 필드 `slice1, slice0, time0`). 이름의 숫자는 "Slice에서 빠져나가 Time으로 가는 하위차원"을 뜻합니다 — `Broadcast01`은 slice0과 slice1 둘 다 Time으로 보내 서브링 전체에 뿌립니다(switch-engine.md:16-17,64). quick-start의 GEMV 의사코드(quick-start.md:236-245)가 정확히 이걸 씁니다:
```rust
input.switch(SwitchConfig::Broadcast01 { slice1: 256, slice0: 1, time0: 1 })
```
즉 "벡터(256슬라이스에 한 칸씩 흩어진 것)를 256슬라이스 전체로 복제"하는 겁니다. GEMM도 같은 원리로 B의 각 타일을 맞는 슬라이스로 옮겨, 슬라이스마다 자기 J 조각만 보게 합니다(quick-start.md:265).

흥미로운 점: base-template의 gemv_kernel.rs/gemm_kernel.rs 본문에는 명시적 `.switch()` 호출이 **안 보입니다**. 책 본문(quick-start.md)이 개념을 의사코드로 분리해 보여주고, 실제 커널은 TRF 적재 + contraction 매핑이 같은 분배를 끌어냅니다(주석 "The Switch Engine automatically broadcasts the vector to all I slices", gemv_kernel.rs:23). 명시적 `.switch(SwitchConfig::Broadcast01 ...)`이 손으로 쓰인 실제 코드는 matmul_split_reduce2의 더 복잡한 변종(`InterTranspose`, `TransposedBroadcast1`)에서 볼 수 있습니다.

## `Slice = m![I/32, J/32]` — GEMM의 출력 타일링

GEMM의 새 개념은 출력 두 축을 **함께** 슬라이스에 매핑하는 것입니다(gemm_kernel.rs:8, quick-start.md:264):
```rust
pub type Slice = m![I / 32, J / 32]; // 슬라이스 하나가 16 × 16 출력 타일 담당
```
`I/32`와 `J/32`를 곱하면 슬라이스 좌표가 되고, 슬라이스 안에는 `I%32`(=32... 실은 16, 아래 주의)·`J%32` 조각이 들어갑니다. 책 주석은 "each slice handles a 16 × 16 output tile"이라 적습니다 — 256슬라이스 = 16×16 격자라서, 출력 행렬을 16×16 슬라이스 격자에 타일로 깐 셈입니다. `Lane = m![J % 8]`로 J의 안쪽 8개를 8레인에 얹고, 마지막에 `.contract_lane::<m![I % 32, J / 8 % 4], m![J % 8]>(LaneMode::Interleaved)`로 레인을 출력 패킷에 접어 I·J를 모두 출력에 보존합니다(gemm_kernel.rs:43, quick-start.md:267).

## K를 따라 줄이는 흐름을 추적하기 (제일 중요한 직관)

dot_product(A=2048)로 추적해 봅시다(dot_product_kernel.rs:36-39):
```
contract_outer::<m![A / 32], m![A % 32], _, _>(&rhs)  // OutTime=A/32(=64), OutPacket=A%32(=32원소=64B, PackSize=2)
contract_packet::<m![1]>()                            // 패킷 안 32원소를 트리로 → 스칼라
contract_time::<m![1]>()                              // 64개 Time스텝을 누적 → 스칼라
contract_lane::<m![1], m![1 # 8]>(Interleaved)        // Lane=1(8로 패딩) 접기
```
즉 2048개 곱을 **"32(Packet, 공간) × 64(Time, 시간)"** 으로 쪼개 줄입니다. 공간 트리로 32개를 한 방에, 그걸 64사이클 누적. 이게 RNGD 축약의 표준 분해입니다.

gemm(K=64)이면 `contract_outer::<..., m![K % 32], _, _>` 로 K%32(=32)를 Packet에, K/32(=2)를 Time(`.contract_time::<m![I % 32, J / 8 % 4]>` 가 줄이는 시간 안에 포함)에 둡니다 → K=64 = 32(공간) × 2(시간). 핵심 원칙(contraction-engine/index.md:123): **K는 가능하면 Packet에 둬서 트리로 한 번에 줄이고, 남는 출력 축(V,M,N)을 Cluster/Slice/Lane에 흩뿌려 병렬을 극대화**하세요. K를 Time에만 두면(K-in-Time) bf16 기준 32개 곱셈기 중 1개만 일해서 MAC 활용률 1/32로 떨어집니다 — 책이 "교육용 나쁜 예"로만 보여주는 경우입니다(index.md:88).

## 슬라이스/칩을 가로지르는 합은 Contraction이 못 합니다 — Vector Engine으로

Contraction Engine은 한 슬라이스 **안에서** Packet/Time/Lane만 줄입니다. contraction 축이 슬라이스나 클러스터·칩에 걸쳐 분산되면, 그 마지막 합은 **Vector Engine의 inter-slice reduce**가 받아야 합니다(contraction-engine/index.md:60). 그래서 큰 matmul들이 contraction 뒤에 `.vector_init().vector_inter_slice_reduce::<...>(InterSliceReduceOpI32::Add).vector_final()`을 답니다(matmul_4096.rs:35-37, matmul_wo_broadcast.rs:33-35). 이걸 모르면 "왜 결과가 슬라이스별 부분합으로만 나오지?"에서 막힙니다.

## 큰 예제들이 보여주는 패턴 (matmul 폴더)

- **matmul_4096** (i8, `[4096,4096]×[4096]→[4096]`): contraction으로 슬라이스별 부분합을 낸 뒤 `vector_inter_slice_reduce`로 합칩니다. 출력은 i32 누산 후 `.cast::<i8, ...>()`로 되돌리고, `unsafe HbmTensor::from_addr` + `slice_tile` + `to_hbm_view`로 HBM에 직접 씁니다(matmul_4096.rs:42-46). Cluster를 `m![A / 2048 % 2]`로 써 A를 두 클러스터에 나눕니다.
- **matmul_wo_broadcast** (i8, 4칩, `[32768]·[32768]→[1]`): 이름 그대로 **Switch 브로드캐스트가 필요 없는** 큰 내적입니다. 두 피연산자를 동일하게 분산하고 칩 4개에 걸쳐 곱한 뒤 inter-slice reduce로 합칩니다(matmul_wo_broadcast.rs). 내적은 브로드캐스트가 없다는 걸 큰 규모로 보여줍니다.
- **matmul_split_reduce** (i8, `[1024,2048]×[2048]→[1024]`): contraction 축 B(=2048)를 512씩 4타일로 쪼개 **부분합을 Vector Engine에 쌓아 누적**합니다(split-K). `begin_interleaved` + `vector_intra_slice_unzip` + `vector_clip_zip(ClipBinaryOpI32::AddFxp)`로 이전 타일과 새 타일을 합칩니다(matmul_split_reduce.rs:18-46, 64-135). DM 용량(512KB/슬라이스)을 못 넘기게 K를 시간 분할하는 실전 패턴입니다.
- **matmul_split_reduce2** (bf16, `[64,1024]×[1024,128]→[64,128]`): 가장 복잡. 루프 안에서 `.switch(SwitchConfig::InterTranspose{...})`, `.switch(SwitchConfig::TransposedBroadcast1{...})`로 데이터를 재배치하고, TRF를 `FirstHalf`/`SecondHalf`로 더블버퍼링하며, 누산기를 DM에 두고 매 반복 `vector_clip_zip(ClipBinaryOpF32::Add)`로 누적합니다. 명시적 Switch + 더블버퍼 + 루프 누적의 종합 예제입니다.

## 호스트 쪽 검증 흐름 (test가 어떻게 도는가)

각 커널엔 짝이 되는 호스트 프로그램이 있습니다(experiments/src/dot_product.rs 등). `Context::acquire()`로 컨텍스트를 잡고, `HostTensor::<bf16, m![A]>::rand(&mut rng)`로 입력을 만들고, `.to_hbm(&mut ctx.pdma, addr)`로 HBM에 올린 뒤 `launch(kernel, (&mut ctx, &lhs_hbm, &rhs_hbm))`로 커널을 띄웁니다(dot_product.rs:8-14). 테스트는 같은 입력을 **f32로 직접 계산한 레퍼런스**와 비교하는데, 허용오차를 `max(0.02 * |expected|, 0.5)`처럼 둡니다(dot_product.rs:48) — bf16 반올림 때문이에요. matmul 테스트는 `Tensor::contraction::<m![A, B, C], _, _>(...)`라는 호스트 레퍼런스 축약과 비교합니다(matmul_tests.rs:31-33). 단, 여러 matmul 테스트는 `#[ignore = "Failing on cpu"]`라 시뮬레이션에선 typecheck로만 검증하고, 값 검증은 split_reduce2가 시뮬레이션에서 통과합니다(matmul_tests.rs:110-115).

## 실행 백엔드 세 가지 (NPU 없이)

experiments 폴더에서 `cargo furiosa-opt` 플러그인이 `--cfg backend="..."`만 끼우고 나머지 cargo 플래그는 그대로 넘깁니다(experiments/README.md):
- `cargo furiosa-opt run --release --bin gemm` → **simulation**(기본). 호스트 CPU에서 실제 값 계산.
- `cargo furiosa-opt --backend typecheck run --release --bin gemm` → **매핑이 합법인지**만 빠르게. 커널 본문을 빈 텐서로 실행해 런타임 매핑 단언까지 확인.
- `cargo furiosa-opt test --release --bin gemm` → 호스트 레퍼런스와 수치 비교.

주의: `cargo check`는 타입만 보고 **커널 본문을 실행하지 않아** "Collect output packet must be exactly 32 bytes" 같은 런타임 매핑 단언을 못 잡습니다. 그래서 매핑 검증은 `--backend typecheck run`을 써야 합니다(experiments/README.md, 출처 introduction.md:133).

## 2. 핵심 API · 패턴

| 이름 | 쓰는 법 | 설명 | 출처 |
|---|---|---|---|
| `contract_outer` | `.contract_outer::<OutTime, OutPacket, _, _>(&trf)  // 뒤 두 제네릭(Lane, TrfElement)은 TRF에서 추론` | Outer 단(Broadcast+Multiply). 흐르는 피연산자와 TRF 피연산자를 공통 모양으로 늘려 원소별 곱. OutTime/OutPacket이 곱 결과의 시간/패킷 모양을 정함. 곱하며 i8→i32, bf16→f32로 widening. | `furiosa-opt-std/src/engine/contraction/outer/mod.rs:74` |
| `contract_packet` | `.contract_packet::<OutPacket>()  // OutPacket = 패킷 안에서 살아남는(줄이지 않는) 부분` | Packet Reducer. 패킷 안 contraction 축을 레인별 덧셈 트리로 공간 합산. 전부 줄이면 m![1]. | `furiosa-opt-std/src/engine/contraction/packet.rs:34` |
| `contract_time` | `.contract_time::<OutTime>()  // OutTime = 살아남는 Time 차원, 나머지는 누적되어 사라짐` | Time Reducer. Packet Reducer의 매 사이클 출력을 시간축 누산기에 더함. OutTime ⊆ Time(순서 보존) 검사. | `furiosa-opt-std/src/engine/contraction/time.rs:22` |
| `contract_lane` | `.contract_lane::<OutTime, OutPacket>(LaneMode::Interleaved)  // 또는 LaneMode::Sequential` | Lane Folder. Lane을 합치지 않고 OutPacket(Interleaved) 또는 OutTime(Sequential)으로 접음. 8원소 버스. | `furiosa-opt-std/src/engine/contraction/lane.rs:45` |
| `LaneMode` | `enum LaneMode { Interleaved, Sequential }` | Interleaved=Lane을 OutPacket으로(출력채널 나란히), Sequential=Lane을 OutTime으로(레인 순차). dot/gemv/gemm은 Interleaved, matmul 큰 예제들은 Sequential. | `furiosa-opt-std/src/engine/contraction/lane.rs:19-26` |
| `to_trf` | `.to_trf::<Lane, Element>(address)  // 보통 .to_trf(TrfAddress::Full)` | Collect 스트림을 TRF에 적재. 흐르는 Time/Packet을 Lane(공간 1/2/4/8)·Element(레인별 레이아웃)로 재배치. 보통 sub 컨텍스트에서 호출해 미리 채워둠. | `furiosa-opt-std/src/engine/collect.rs:56` |
| `TrfAddress` | `enum TrfAddress { FirstHalf, SecondHalf, Full }  // Full.capacity()=65536, Half=32768` | TRF 적재 영역 선택. Full=128행 전부, FirstHalf=0–63행, SecondHalf=64–127행(더블버퍼링). dot/gemv/gemm은 Full, split_reduce2는 FirstHalf/SecondHalf 번갈아. | `furiosa-opt-std/src/tensor/memory.rs:101-119` |
| `cast` | `.cast::<bf16, m![1 # 16]>()  // bf16 16원소=32바이트(1 flit)` | f32 누산 결과를 출력 타입으로 되돌림. 출력 패킷을 정확히 32바이트로 패딩(bf16=16원소). 크기 틀리면 'output packet must be exactly 32 bytes' 단언. | `furiosa-opt-std/src/engine/cast.rs:38` |
| `switch / SwitchConfig::Broadcast01` | `input.switch(SwitchConfig::Broadcast01 { slice1: 256, slice0: 1, time0: 1 })` | GEMV/GEMM에서 한 피연산자를 모든 출력 슬라이스로 브로드캐스트. slice0·slice1을 Slice에서 Time으로 옮겨 서브링 전체에 복제. 명시적 사용 예는 matmul_split_reduce2(InterTranspose/TransposedBroadcast1). | `furiosa-opt-std/src/engine/switch.rs:35-42 + docs/src/quick-start.md:236-245` |
| `vector_inter_slice_reduce` | `.vector_init().vector_inter_slice_reduce::<OutSlice, ReduceDim>(InterSliceReduceOpI32::Add).vector_final()` | contraction 축이 슬라이스에 걸칠 때 Contraction이 못 하는 슬라이스 간 합을 Vector Engine이 마무리. 큰 matmul 필수 패턴. | `furiosa-opt-examples/src/matmul/matmul_4096.rs:35-37` |
| `launch` | `let out_hbm = launch(gemm_kernel, (&mut ctx, &a_hbm, &b_hbm)).await;` | 호스트에서 #[device] 커널을 띄움. Context::acquire()로 ctx 잡고 HostTensor::rand→to_hbm→launch→to_host 순서. | `base-template/src/gemm.rs:14` |

## 3. 실험 (직접 돌리기)

> 실험은 NPU 없이 `simulation`·`typecheck`로 돌아갑니다. 실행법은 [`../experiments/README.md`](../experiments/README.md), MNIST는 `cargo furiosa-opt test`(npu 전용).

### 실험 04.1 — dot_product 시뮬레이션 실행 + 값 검증
*난이도 1/5 · 기반: `Model_Benchmark/rngd-npu/vISA/experiments/src/dot_product.rs`*

**목표** — 축약 4단(outer→packet→time→lane) 전체가 도는 가장 단순한 커널을 실제로 돌려 결과가 호스트 레퍼런스(f32 내적)와 맞는지 확인한다.

```bash
cd /home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vISA/experiments && cargo furiosa-opt run --release --bin dot_product && cargo furiosa-opt test --release --bin dot_product
```
**관찰** — run은 'Dot Product: kernel ran' 출력, test는 통과(허용오차 max(0.02*|expected|,0.5) 안). A=2048이 Packet 32 × Time 64로 줄어든다는 걸 dot_product_kernel.rs:36-39와 대조.

**심화** — test가 비교하는 레퍼런스(dot_product.rs:36-41)가 왜 f32로 합산 후 bf16으로 반올림하는지, 허용오차가 왜 필요한지 설명해 보기.

### 실험 04.2 — gemv / gemm 실행과 검증 (브로드캐스트와 출력 타일링)
*난이도 1/5 · 기반: `Model_Benchmark/rngd-npu/vISA/experiments/src/kernel/gemm_kernel.rs`*

**목표** — GEMV의 Switch 브로드캐스트(벡터를 모든 I 슬라이스로)와 GEMM의 Slice=m![I/32,J/32] 출력 타일링이 결과를 바꾸지 않고 올바르게 도는지 본다.

```bash
cd /home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vISA/experiments && cargo furiosa-opt test --release --bin gemv && cargo furiosa-opt test --release --bin gemm
```
**관찰** — 둘 다 통과. gemm은 I=512,J=512,K=64라 256슬라이스 16×16 격자에 출력 타일이 깔린다. gemm_kernel.rs:8의 Slice 타입과 quick-start.md:264 설명을 대조.

**심화** — gemm 허용오차가 dot_product보다 큰 이유(K 누적이 더 많아 bf16 오차 누적)를 gemm.rs:55의 tol과 연결지어 설명.

### 실험 04.3 — cast 패킷 크기 일부러 깨뜨리기 (typecheck로 매핑 단언 보기)
*난이도 2/5 · 기반: `Model_Benchmark/rngd-npu/vISA/experiments/src/kernel/dot_product_kernel.rs`*

**목표** — Cast Engine이 출력 패킷을 32바이트로 강제한다는 사실을 실패로 확인한다. cargo check로는 안 잡히고 typecheck 백엔드여야 잡힌다는 것도 체감.

```bash
cd /home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vISA/experiments && sed -i 's/.cast::<bf16, m!\[1 # 16\]>()/.cast::<bf16, m![1 # 8]>()/' src/kernel/dot_product_kernel.rs && cargo furiosa-opt --backend typecheck run --release --bin dot_product ; git checkout src/kernel/dot_product_kernel.rs 2>/dev/null || sed -i 's/.cast::<bf16, m!\[1 # 8\]>()/.cast::<bf16, m![1 # 16]>()/' src/kernel/dot_product_kernel.rs
```
**관찰** — typecheck 실행이 'output packet must be exactly 32 bytes' 류 단언으로 실패해야 한다(bf16 8개=16바이트). 마지막 명령이 원복. 비교로 'cargo check'만 하면 통과(본문 미실행)임을 확인.

**심화** — 왜 16원소(m![1 # 16])가 정답인지 cast.rs:35-44와 cast-engine.md로 근거 찾기.

### 실험 04.4 — 행렬 크기(axes!) 바꿔 K 축약 분해 관찰
*난이도 3/5 · 기반: `Model_Benchmark/rngd-npu/vISA/experiments/src/kernel/dot_product_kernel.rs`*

**목표** — contraction 축 크기를 바꿔도 결과가 맞는지, 그리고 Packet×Time 분해가 어떻게 달라지는지 본다. 'change matrix sizes(axes!)' 실습.

```bash
cd /home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vISA/experiments && sed -i 's/axes!\[A = 2048\];/axes![A = 4096];/' src/kernel/dot_product_kernel.rs && cargo furiosa-opt test --release --bin dot_product ; sed -i 's/axes!\[A = 4096\];/axes![A = 2048];/' src/kernel/dot_product_kernel.rs
```
**관찰** — A=4096이어도 test 통과(호스트가 같은 axes를 import하므로 자동 일치). A/32=128 Time스텝 × 32 Packet으로 줄어든다 — A=2048(64×32) 대비 Time만 2배. 마지막 명령이 원복.

**심화** — gemv에서 J를 2048→1024로 바꿔(반드시 32의 배수 유지) 같은 실험. Time=J/32가 64→32로 줄어드는 걸 확인. gemv_kernel.rs:8-9의 Time/Packet 정의와 대조.

### 실험 04.5 — matmul split-K 누적을 시뮬레이션에서 값 검증
*난이도 4/5 · 기반: `furiosa-opt-examples/src/matmul/matmul_split_reduce2.rs`*

**목표** — contraction 축을 타일로 쪼개 Vector Engine으로 부분합을 누적하는 실전 split-reduce 패턴이 정답을 내는지 본다(시뮬레이션에서 값까지 통과하는 유일한 matmul 변종).

```bash
cd ~/furiosa-opt && cargo furiosa-opt test --release -p furiosa-opt-examples test_matmul_with_split_reduce2
```
**관찰** — test_matmul_with_split_reduce2 통과(허용오차 1.5 또는 norm*0.2, matmul_tests.rs:115-155). bf16 [64,1024]×[1024,128]를 K 128타일×루프로 누적. 코드에서 TrfAddress::FirstHalf/SecondHalf 더블버퍼와 SwitchConfig::InterTranspose/TransposedBroadcast1를 찾아본다.

**심화** — test_matmul_4096는 #[ignore="Failing on cpu"]라 값은 시뮬에서 틀린다. 대신 매핑 합법성만 보려면 typecheck로: cargo furiosa-opt --backend typecheck test --release -p furiosa-opt-examples test_matmul_4096 -- --ignored

### 실험 04.6 — matmul 변종들의 매핑 합법성 typecheck 일괄 확인
*난이도 2/5 · 기반: `furiosa-opt-examples/tests/matmul_tests.rs`*

**목표** — 값 계산(시뮬) 없이 매핑 타입이 합법인지만 빠르게 확인하는 typecheck 워크플로를 익힌다. cpu에서 실패 표시된 변종도 typecheck는 통과해야 한다.

```bash
cd ~/furiosa-opt && cargo furiosa-opt --backend typecheck test --release -p furiosa-opt-examples matmul -- --ignored
```
**관찰** — typecheck는 커널 본문을 빈 텐서로 실행해 contract_*·switch·to_trf의 매핑 단언만 검사한다. 매핑이 합법인 변종은 통과. 값 불일치(cpu 실패)는 typecheck 관심사가 아님을 체감.

**심화** — matmul_wo_broadcast가 왜 Switch 브로드캐스트 없이도 합법인지(내적은 브로드캐스트 불필요) 소스(matmul_wo_broadcast.rs)와 einsum I,I→1로 설명.

## 4. 연습문제 (손으로, 컴파일 없이)

**Q1.** dot_product_kernel(A=2048)에서 .contract_outer::<m![A / 32], m![A % 32], _, _>(&rhs) 다음에 .contract_packet::<m![1]>(), .contract_time::<m![1]>()가 옵니다. (a) Packet Reducer가 한 번에 줄이는 원소 수, (b) Time Reducer가 누적하는 사이클 수, (c) 둘을 곱한 총 축약 길이를 구하세요.

<details><summary>정답/힌트</summary>

A%32 = 32원소(=64B, bf16, PackSize=2) → 트리로 32개 한 방에. A/32 = 2048/32 = 64 사이클 누적. 32 × 64 = 2048 = A. 즉 contraction 축 A를 '공간 32 × 시간 64'로 분해해 줄인다.

</details>

**Q2.** 다음 코드의 오류를 찾으세요: bf16 누산 결과를 .cast::<bf16, m![1 # 8]>() 로 캐스트했다. 왜 빌드(정확히는 typecheck 실행)가 막히나요? 올바른 형태는?

<details><summary>정답/힌트</summary>

Cast Engine은 출력 패킷을 정확히 32바이트로 맞춰야 한다. bf16은 16비트라 8원소=16바이트뿐 → 단언 위반. 32바이트는 16원소이므로 m![1 # 16]이 정답. (cast-engine.md, cast.rs:48-52)

</details>

**Q3.** GEMV가 IJ, J → I 이고 axes![I = 256, J = 2048]일 때 gemv_kernel은 Slice = m![I], Time = m![J / 32], Packet = m![J % 32]로 둡니다. (a) Time 사이클 수, (b) 각 사이클 Packet 원소 수, (c) 슬라이스 하나가 무슨 일을 하는지 한 줄로.

<details><summary>정답/힌트</summary>

Time = J/32 = 2048/32 = 64 사이클. Packet = J%32 = 32원소(64B). 슬라이스 i 하나가 행 A[i,:]와 (브로드캐스트로 받은) 벡터 x를 곱해 y_i = Σ_j A[i,j]x[j]를 계산. I=256이 256슬라이스에 1행씩 분산.

</details>

**Q4.** GEMM에서 K를 Time에만 두는 'K-in-Time' 배치는 bf16에서 왜 나쁜가요? 활용률 숫자와 함께 답하고, 대신 어떻게 두어야 하는지 쓰세요.

<details><summary>정답/힌트</summary>

bf16 패킷은 32원소(곱셈기 32개)인데 K를 Time에 두면 Packet에 1개만 실려(m![1 # 32] 패딩) 매 사이클 곱셈기 32개 중 1개만 일함 → MAC 활용률 1/32. K는 Packet에 둬 덧셈 트리로 한 번에 줄이고, 남는 V·M·N을 Cluster/Slice/Lane에 분산해야 함. (index.md:88,123)

</details>

**Q5.** matmul_4096(i8)에서 contraction 뒤에 .vector_init().vector_inter_slice_reduce(...).vector_final()이 붙고, 출력은 i32였다가 .cast::<i8, ...>()로 돌아갑니다. 두 가지 '왜'를 답하세요: (1) 왜 inter_slice_reduce가 필요한가, (2) 왜 누산이 i32인가.

<details><summary>정답/힌트</summary>

(1) Contraction Engine은 한 슬라이스 안에서만 Packet/Time/Lane을 줄인다. contraction 축이 여러 슬라이스에 분산되면 슬라이스 간 최종 합은 Contraction이 못 하고 Vector Engine의 inter_slice_reduce가 받아야 함. (2) Multiplier가 i8→i32로 widening해 누적 오버플로를 막기 때문. 마지막에 i8로 cast해 되돌림. (index.md:60, outer.md:200)

</details>

**Q6.** LaneMode::Interleaved와 Sequential의 차이를 .contract_lane::<OutTime, OutPacket>() 관점에서 설명하고, dot/gemv/gemm이 왜 Interleaved를 쓰는지 한 줄로.

<details><summary>정답/힌트</summary>

Interleaved는 Lane을 OutPacket의 가장 안쪽으로 접어 매 사이클 8레인의 같은 열을 8값/flit로 내보냄(출력채널 나란히). Sequential은 Lane을 OutTime으로 접어 한 레인의 패킷을 순차로 뽑음. dot/gemv/gemm은 출력 원소들을 레인에 나란히 펴서 한 번에 내보내는 게 자연스러워 Interleaved를 씀(matmul 큰 예제는 슬라이스 간 reduce 전 단계라 Sequential). (lane-folder.md, lane.rs:21-26)

</details>

## 5. 흔한 함정

- Contraction Engine만으로 슬라이스 간 합을 끝낼 수 있다고 착각. contract_*는 한 슬라이스 안의 Packet/Time/Lane만 줄인다. contraction 축이 여러 슬라이스/클러스터/칩에 걸치면 결과가 슬라이스별 부분합으로만 남고, 반드시 Vector Engine의 vector_inter_slice_reduce로 마무리해야 한다.  
  ↳ 출처 `docs/src/computing-tensors/contraction-engine/index.md:60 + furiosa-opt-examples/src/matmul/matmul_4096.rs:35-37`
- cast 출력 패킷 크기를 아무렇게나 둠. Cast Engine은 출력을 정확히 32바이트(1 flit)로 강제한다. bf16은 16원소(m![1 # 16])여야 하고 m![1 # 8](16바이트)은 단언 위반. 게다가 cargo check로는 안 잡히고 --backend typecheck run이어야 잡힌다.  
  ↳ 출처 `furiosa-opt-std/src/engine/cast.rs:48-52 + Model_Benchmark/rngd-npu/vISA/experiments/README.md`
- contraction 축 K를 Time에만 둠. bf16에서 패킷 곱셈기 32개 중 1개만 일해 MAC 활용률 1/32로 떨어진다. K는 Packet에 둬 덧셈 트리로 한 번에 줄이고, 남는 출력 축을 Cluster/Slice/Lane에 분산하는 게 정석.  
  ↳ 출처 `docs/src/computing-tensors/contraction-engine/index.md:88,123`
- contract_packet/_time의 제네릭에 '줄일 축'을 적는다고 오해. 반대다 — 제네릭(OutPacket/OutTime)에는 '살아남는 축'을 적고, 적지 않은 contraction 축이 줄어 사라진다. 전부 줄이려면 m![1].  
  ↳ 출처 `furiosa-opt-std/src/engine/contraction/packet.rs:34 + docs/src/computing-tensors/contraction-engine/packet-reducer.md:9`
- Packet을 64바이트보다 크게, Lane을 8보다 크게 잡으려 함. Outer 단 하드웨어 상한 Lane≤8, Packet≤64B를 넘으면 안 된다. 큰 K는 Packet(공간)+Time(시간)으로 쪼개야 한다.  
  ↳ 출처 `docs/src/computing-tensors/contraction-engine/index.md:62`
- Time Reducer 누산기 용량 초과. InnerTime::SIZE가 슬롯 용량(Interleaved=128/Packet::SIZE, Sequential=32/Lane::SIZE)을 넘으면 컴파일에서 막힌다. Time을 더 쪼개거나 LaneMode를 바꿔 슬롯 여유를 확보해야 한다.  
  ↳ 출처 `docs/src/computing-tensors/contraction-engine/time-reducer.md:88-97`
- main/sub가 같은 SRAM을 공유하는데 주소를 겹치게 둠. to_dm/to_trf/commit의 주소 인자를 프로그래머가 직접 안 겹치게 줘야 한다(예: rhs를 1<<12 같은 다른 base에). 겹치면 데이터가 깨진다.  
  ↳ 출처 `docs/src/quick-start.md:101-102 + base-template/src/kernel/dot_product_kernel.rs:18-19`
- 시뮬레이션에서 모든 matmul 테스트가 값까지 통과할 거라 기대. 여러 변종은 #[ignore="Failing on cpu"]라 시뮬에선 값이 안 맞는다(NPU 전용). 시뮬에서 값까지 통과하는 건 test_matmul_with_split_reduce2뿐이고, 나머지는 typecheck로 매핑 합법성만 보거나 -- --ignored로 NPU에서 돌려야 한다.  
  ↳ 출처 `furiosa-opt-examples/tests/matmul_tests.rs:36-41,110-115`
- 새 커널을 src/bin/이나 examples/에 두고 cargo furiosa-opt가 인식하길 기대. 플러그인은 src/에 뿌리내린 cargo 타겟만 스캔하고 src/bin·examples·tests는 조용히 건너뛴다. 반드시 Cargo.toml에 [[bin]] name=... path="src/..."를 명시해야 한다.  
  ↳ 출처 `Model_Benchmark/rngd-npu/vISA/experiments/README.md (출처 introduction.md:63,79)`

## 6. 핵심 정리 & 다음

기억할 사실:
- RNGD 계층: 칩당 클러스터 2개, 클러스터당 슬라이스 256개, 슬라이스당 레인 8개. 레인 하나가 Contraction Engine MAC 배열의 한 행이다. (`docs/src/quick-start.md:47-52`)
- Contraction Engine Outer 단의 하드웨어 상한: Lane ≤ 8, Packet ≤ 64바이트(RNGD). 그래서 한 슬라이스가 한 번에 다루는 공간 병렬은 8레인 × 64B로 제한된다. (`docs/src/computing-tensors/contraction-engine/index.md:62`)
- Multiplier는 누적 오버플로 방지를 위해 곱하면서 타입을 넓힌다: i4/i8 → i32, f8/bf16 → f32. 그래서 bf16 행렬곱 누산기는 f32다. (`docs/src/computing-tensors/contraction-engine/outer.md:12,200`)
- Packet Reducer 덧셈 트리 깊이는 타입별로 다르다: bf16 5단(패킷 32원소), i8/f8 6단(64), i4 7단(128). 깊이는 첫 출력 지연만 늘리고 정상 throughput(매 사이클 1패킷)은 유지된다. (`docs/src/computing-tensors/contraction-engine/packet-reducer.md:47,58-60`)
- Packing의 PackSize가 MAC 활용률을 정한다: PackSize=2면 64B를 꽉 채워 모든 MAC 사용, PackSize=1이면 32B만 채우고 나머지 절반이 0 곱이라 throughput 절반. (`docs/src/computing-tensors/contraction-engine/outer.md:105-107`)
- K를 Time에만 두는 배치(K-in-Time)는 bf16 기준 32개 곱셈기 중 1개만 일해 MAC 활용률이 1/32로 떨어지는 퇴화 커널이다. K는 Packet에 둬 트리로 줄이는 게 정석. (`docs/src/computing-tensors/contraction-engine/index.md:88,123`)
- TRF는 슬라이스당 8레인 × 2뱅크 × 128행 × 320비트 = 80KB. Full 주소모드 용량은 65,536바이트, FirstHalf/SecondHalf는 각 32,768바이트(더블버퍼링용). (`docs/src/computing-tensors/register-files.md:72,92-94 + furiosa-opt-std/src/tensor/memory.rs:111-119`)
- Time Reducer 누산기는 1,024셀. 슬롯 용량은 LaneMode에 의존: Interleaved면 128/Packet::SIZE, Sequential이면 32/Lane::SIZE. InnerTime::SIZE가 이를 넘으면 컴파일 단계에서 막힌다. (`docs/src/computing-tensors/contraction-engine/time-reducer.md:88-94`)

➡️ 다음: [05_moving_tensors.md](./05_moving_tensors.md)
