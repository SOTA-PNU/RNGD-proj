# 06 · 연산 엔진 I — 분배와 축약

이 문서는 vISA 커리큘럼 모듈 06입니다. Switch 토폴로지, Collect(32바이트 flit 정규화), TRF/VRF 레지스터 뱅킹, 그리고 Contraction 엔진 내부(Outer/Packet/Time/Lane, 2D conv)를 깊게 봅니다.
*선행: 04 텐서 축약, 05 텐서 옮기기 · 예상 시간: 하루*

## 학습 목표

- [ ] Switch 토폴로지(Broadcast01/Broadcast1/Transpose/InterTranspose/CustomBroadcast)를 구분한다
- [ ] Collect가 만드는 32바이트 flit과 64바이트 packet 관계를 안다
- [ ] TRF/VRF 뱅킹과 이중 버퍼링 구조를 안다
- [ ] Contraction 하위 단계(Outer→Packet→Time→Lane)와 2D conv 매핑을 안다

## 1. 개념

## 0. 큰 그림: Tensor Unit 파이프라인이 무엇을 하는가

RNGD 칩 안에는 "Tensor Unit(TU)"라는 온칩 연산 파이프라인이 있습니다. 하는 일은 한 문장으로 끝납니다. **DM(Data Memory)에서 텐서를 읽어, 여덟 개의 엔진을 차례로 통과시키며 변형하고, 결과를 다시 DM에 씁니다** (docs/src/computing-tensors/index.md:4-6).

순서는 이렇습니다: `Fetch → Switching → Collect → Contraction → Vector → Cast → Transpose → Commit` (index.md:21). 데이터는 한 덩어리로 흐르지 않고 **"패킷(packet) 스트림"** 으로 흐릅니다. 매 사이클(cycle)마다 패킷 하나가 흘러 들어가고, 각 엔진은 이 스트림을 소비하면서 매 사이클의 레이아웃과 반복 순서를 바꿔 다음 엔진으로 내보냅니다 (index.md:8-9). 이걸 GPU에 비유하면, GPU는 "커널 하나 = 메모리 읽고 → 계산하고 → 쓰기"를 한 덩어리로 보지만, 여기서는 그 흐름이 하드웨어 단계(엔진)별로 물리적으로 쪼개져 있고, 각 단계 사이를 32바이트짜리 작은 봉투(flit)에 담아 컨베이어 벨트로 넘기는 구조라고 보면 됩니다.

### flit(플릿)이라는 공용 규격

이 컨베이어 벨트의 표준 봉투 크기가 **32바이트 = 1 flit** 입니다. Fetch가 가져온 패킷은 크기가 제각각이지만, **Collect 엔진이 모든 패킷을 정확히 32바이트 flit 단위로 정규화**합니다 (index.md:10). 그 뒤로 오는 Contraction, Vector, Cast, Transpose, Commit 엔진은 전부 이 flit만 먹습니다 (index.md:11, collect-engine.md:3). 왜 굳이 규격을 통일할까요? 다운스트림 엔진들이 "매 사이클 정확히 32바이트"라는 고정 폭을 전제로 만들어진 고정 기능(fixed-function) 회로이기 때문입니다. 폭이 들쭉날쭉하면 버퍼링과 재정렬이 필요해지고, 그건 실리콘 면적과 지연을 잡아먹습니다.

### 5차원 좌표계: [Chip, Cluster, Slice, Time, Packet]

TU 안을 흐르는 모든 텐서 스트림은 다섯 개의 차원을 답니다: `[Chip, Cluster, Slice, Time, Packet]` (index.md:50). 이게 vISA를 이해하는 핵심 좌표계라서 꼭 둘로 나눠서 외워야 합니다.

- **공간 차원(spatial): Chip, Cluster, Slice.** 물리적으로 "어느 하드웨어 칸에서 도느냐"입니다. slice 하나하나가 자기만의 파이프라인 인스턴스를 돌립니다. slice들이 모여 cluster, cluster들이 모여 chip이 됩니다 (index.md:51). 한 칩에는 **slice가 256개**, cluster당 256 slice 구조입니다 (switch-engine.md:3, index.md:54).
- **시간 차원(temporal): Time, Packet.** slice 하나가 처리하는 "스트림"의 모양입니다. Packet은 한 사이클에 함께 들어오는 데이터(공간적 병렬), Time은 사이클을 거듭하며 반복되는 축(시간적 직렬)입니다.

엔진들은 주로 `Time`/`Packet`을 따라 모양을 바꿉니다. 공간 차원(Chip/Cluster/Slice)은 **거의 모든 엔진이 그대로 통과**시킵니다. 예외는 딱 둘입니다: **Switch 엔진은 `Slice`를 바꿉니다**(slice 사이로 데이터를 옮기니까), 그리고 Vector 엔진의 inter-slice reducer는 cluster 안 256개 slice를 합치면서 `Slice`를 접습니다 (index.md:54).

### 두 개의 레지스터 파일: TRF와 VRF

Contraction 엔진과 Vector 엔진은 **피연산자 두 개 중 하나는 파이프라인 스트림에서, 다른 하나는 전용 레지스터 파일에서** 가져옵니다 (index.md:56). Contraction에는 **TRF(Tensor Register File)**, Vector에는 **VRF(Vector Register File)**가 붙습니다. Collect 엔진이 `.to_trf()`로 TRF에, `.to_vrf()`로 VRF에 채워 넣습니다 (index.md:58). 행렬곱으로 치면, 스트리밍으로 흐르는 입력(A)과 미리 TRF에 올려둔 가중치(B)를 곱하는 식입니다.

### 실행 컨텍스트(execution context): Main / Sub / DMA

스케줄러는 세 개의 독립 실행 스트림을 노출합니다 (index.md:64-71).
- **Main**: 커널의 주 연산을 위해 TU 파이프라인 전체를 구동합니다.
- **Sub**: 같은 파이프라인의 부분집합을 구동합니다. 보통 **main이 계산하는 동안 다음 피연산자를 TRF/VRF에 미리 적재(prefetch)** 하는 용도입니다. **Sub는 Contraction 엔진을 못 씁니다**(몇 가지 기능도 빠짐) (index.md:74).
- **DMA**: TU와 별개로 DMA 엔진만 구동합니다(HBM↔DM, HBM↔SPM, DM↔SPM).

한 컨텍스트 안에서는 연산이 직렬화되지만, 컨텍스트가 다르면 병렬로 돕니다. 그래서 흔한 패턴이 **이중 버퍼링**입니다: sub가 다음 배치를 TRF에 채우는 동안 main이 현재 배치를 계산합니다 (index.md:78). 실제 커널 코드에서 `ctx.sub.begin(...).to_trf(...)`로 가중치를 올리고, `ctx.main.begin(...).contract_outer(&trf)...`로 계산하는 모습이 그대로 보입니다 (base-template/src/kernel/gemm_kernel.rs:30-46). 주의: Vector 엔진과 Cast 엔진은 "한 번에 한 컨텍스트만" 쓰는 하나의 스케줄링 단위라서, sub가 Vector를 쓰는 동안 main은 Cast 대신 Commit 엔진의 타입 캐스팅을 써서 직렬화를 피합니다 (index.md:80-82).

---

## 1. Switch Engine — slice 사이로 데이터를 나르는 링 네트워크

다른 모든 엔진은 각자 자기 slice의 DM 파티션 안에서만 놉니다. **Switch 엔진만 slice 경계를 넘어 데이터를 옮깁니다** (switch-engine.md:3). 한 칩의 256개 slice를 하나의 물리적 **링(ring) 네트워크**로 엮어서, ① 한 slice 값을 그룹 전체에 **브로드캐스트**하거나, ② slice끼리 값을 **맞바꾸거나(transpose)**, ③ 어느 slice가 어느 값을 갖는지 **순열(permute)** 합니다.

### 인터페이스

`FetchTensor::switch()`가 `SwitchTensor`를 만듭니다. `Chip`, `Cluster`, `Packet`과 실제 데이터 값은 보존하고, **오직 `Slice`와 `Time` 매핑만** 설정에 맞게 바뀝니다 (switch-engine.md:7-8). 시그니처는:

```rust
pub fn switch<OutSlice: M, OutTime: M>(self, config: SwitchConfig)
    -> SwitchTensor<'l, T, D, Chip, Cluster, OutSlice, OutTime, Packet, B>
```
(furiosa-opt-std/src/engine/switch.rs:678-684)

커널 작성자가 `OutSlice`, `OutTime`, 그리고 `SwitchConfig`를 고릅니다. 컴파일러는 `OutSlice`/`OutTime`이 설정이 요구하는 차원 구조와 맞는지 검증하고, 안 맞으면 컴파일을 실패시킵니다. **모든 설정은 `InSlice::SIZE == OutSlice::SIZE`를 강제**합니다 — switch는 총 slice 수를 절대 바꾸지 않습니다 (switch-engine.md:19-20, switch.rs:692-700).

설정 이름의 숫자 접미사는 "`Slice`에서 빠져나가는 slice 하위 차원"을 나타냅니다. `Broadcast01`은 `slice1`과 `slice0` 둘 다, `Broadcast1`은 `slice1`만 빠집니다 (switch-engine.md:16-17).

### sub-ring(서브링)과 ring_size

각 정규 설정은 256개 slice를 `ring_size`개씩 묶은 **병렬 sub-ring**들로 나눕니다 (switch-engine.md:24). 정규 설정에서 `ring_size`는 보통 `slice1 × slice0`으로 컴파일러가 유도합니다 (switch-engine.md:25). `ring_size`는 두 가지를 동시에 정합니다: ① 한 ring이 몇 개 slice를 덮는가(분할 단위), ② 사이클 비용. ring_size가 크면 한 ring이 더 많은 slice를 덮지만 ring당 비용이 비싸고, 작으면 더 많은 병렬 ring으로 쪼개져 ring당 비용이 쌉니다 (switch-engine.md:393-395).

### 여섯 가지 정규 설정 (switch-engine.md:33-40)

| 설정 | 하는 일 | ring_size |
|---|---|---|
| `Forwarding` | 교환 없이 그대로 통과 | 1 |
| `Broadcast01` | 안쪽 두 slice 차원(slice1, slice0)을 sub-ring 전체에 브로드캐스트 | slice1×slice0 |
| `Broadcast1` | slice1만 브로드캐스트, slice0은 Slice에 유지 | slice1×slice0 |
| `Transpose` | Slice 안에서 slice1↔slice0 맞바꿈 | slice1×slice0 |
| `InterTranspose` | Slice의 slice1과 Time의 time1을 맞바꿈 | slice1×slice0 |
| `TransposedBroadcast1` | slice0을 Time으로 브로드캐스트하면서 slice1을 가장 안쪽 Slice로(=Transpose 후 Broadcast1) | slice1×slice0 |

**Forwarding은 SwitchConfig 변형이 없습니다.** slice 간 교환이 필요 없으면 `.switch()`를 건너뛰고 `FetchTensor`에 바로 `.collect()`를 호출하면 됩니다 (switch-engine.md:46-47).

**Broadcast01**: 입력 `[slice2|slice1|slice0][time1|time0]`이 출력 `[slice2|X|Y][time1|slice1|time0|slice0]`이 됩니다 (switch-engine.md:68-85). slice1, slice0이 `Slice`를 떠나 `Time`으로 들어가고, 비어버린 `Slice` 자리에 새 브로드캐스트 축 `X`, `Y`가 채워집니다. 모든 slice가 자기 패킷을 ring을 돌려 보내고, 모든 slice가 ring_size개 패킷을 다 받아서 결국 각 출력 slice가 그룹 전체 데이터를 갖게 됩니다 (switch-engine.md:87-88).

**Broadcast1**: 입력 `[slice2|slice1|slice0][time0]` → 출력 `[slice2|X|slice0][time0|slice1]` (switch-engine.md:121-137). 물리적 ring 크기는 Broadcast01과 같지만(`slice1×slice0`), 브로드캐스트는 slice1을 따라서만 일어나고 slice0은 제자리에 보존됩니다.

**Transpose**: `[slice2|slice1|slice0]` → `[slice2|slice0|slice1]` (switch-engine.md:176-181). 각 sub-ring이 데이터를 순환시켜 모든 slice가 자기 swap 파트너의 값을 갖게 합니다. **입력 Time과 출력 Time이 (정규화 후) 일치해야** 합니다 (switch-engine.md:186).

**InterTranspose**: `Slice`와 `Time` 사이를 가로지르는 교환입니다. slice1이 Time으로, time1이 Slice로 들어갑니다 (switch-engine.md:214, 226-233). 세 가지 사이즈 제약이 있습니다 (switch-engine.md:238-242): ① `slice2×slice1×slice0 == 256` (256 slice 전부 커버), ② `time1.SIZE == slice1` (맞바꾸는 두 차원 크기 일치), ③ `InTime::SIZE`가 `slice1×time0`으로 나눠떨어져야 함(time2 분해가 정수가 되도록).

**TransposedBroadcast1**: `[slice2|slice1|slice0][time0]` → `[slice2|Y|slice1][time0|slice0]` (switch-engine.md:275-291). 이름 그대로 Transpose 다음 Broadcast1을 한 번에 적용한 것과 같습니다.

> 공통 규칙: **브로드캐스트 축(X, Y)은 새 축이어야 합니다.** 입력 `Slice`나 입력 `Time`에 이미 등장하는 축을 브로드캐스트 축으로 쓰면 안 됩니다 (switch-engine.md:31). 예제 파일 `switch_assertions.rs`에 이 규칙을 어긴 경우(`invalid_broadcast_axes_not_new`, switch_assertions.rs:393-409)가 일부러 들어 있습니다.

### 아키텍처: 링 + 라우터 + snoop bitmap

256 slice가 slice당 라우터 하나씩 달린 하나의 물리적 링으로 배열되고, `256 / ring_size`개의 병렬 sub-ring으로 쪼개집니다 (switch-engine.md:323). 링크는 **양방향**이고 끝은 wrap-around로 이어집니다 (switch-engine.md:327-333).

Switch 엔진은 **snoop bitmap**으로 설정됩니다: 256개 엔트리(slice당 하나), 각 엔트리는 "이 출력 slice에 어느 소스 slice들의 데이터가 도착해야 하는가"를 적습니다 (switch-engine.md:335). 정규 설정은 내장 bitmap 생성기를 갖고, `CustomBroadcast`는 컴파일러가 사용자의 입출력 매핑으로부터 임의의 bitmap을 합성합니다 (switch-engine.md:336-337).

각 라우터는 bitmap 엔트리에 따라 들어오는 패킷마다 세 동작의 조합을 결정합니다 (switch-engine.md:340-344): **Output**(로컬 slice의 다운스트림으로 내보내기), **Forward right**(오른쪽 이웃에게), **Forward left**(왼쪽 이웃에게). 각 sub-ring에서 맨 왼쪽 라우터는 자기 데이터를 오른쪽으로, 맨 오른쪽은 왼쪽으로 보냅니다. ring_size>2면 중간 라우터들은 왼쪽 이웃 데이터를 output하면서 오른쪽으로 forward합니다 (switch-engine.md:346-348). 링크 1개 통과에 1사이클이 걸려서, 2-slice sub-ring 트레이스를 보면 맨 왼쪽은 cycle 0에, 맨 오른쪽은 cycle 1에 시작해 cycle 4면 두 slice 모두 4개 패킷을 다 갖습니다(switch-engine.md:356-364). bitmap이 변형을 인코딩하는 방식은 두 가지입니다 (switch-engine.md:366-369): **브로드캐스트 모양**(여러 출력 slice가 동일한 엔트리), **Slice→Time 모양**(한 출력 slice가 여러 소스 slice를 나열 = 여러 시간 단계에 걸쳐 모음).

### 성능 공식 (이거 하나는 꼭 외우세요)

> **cycles ≈ ring_size × Time::SIZE × flits_per_packet** (switch-engine.md:389)
>
> 여기서 **flits_per_packet = sizeof(D) × Packet::SIZE / 32** (switch-engine.md:397).

세 인자의 의미 (switch-engine.md:391-398): `ring_size`는 flit 하나가 sub-ring 한 바퀴 도는 사이클, `Time::SIZE`는 시간 단계 수(매 단계마다 순회 반복), `flits_per_packet`는 패킷이 몇 개 flit인지(flit마다 순회 반복). 모든 sub-ring이 병렬로 돌므로 이 ring당 사이클이 곧 칩 전체 지연입니다.

예를 들어 Broadcast01 예제는 `4 × 64 × 8 = 2048` 사이클입니다 (ring_size=4, Time::SIZE=64, f32라 flits_per_packet = 4×64/32 = 8) (switch-engine.md:115). i8 Broadcast1 예제는 `32 × 64 × 2 = 4096` (flits_per_packet = 1×64/32 = 2) (switch-engine.md:168).

### CustomBroadcast — 정규 설정으로 표현 못 하는 경우

```rust
CustomBroadcast { ring_size: usize }
```
(switch.rs:74-77, switch-engine.md:406-415)

정규 설정이 못 하는 두 패턴을 커버합니다 (switch-engine.md:422-425): **자유 transpose+broadcast**(임의 순열/브로드캐스트), **부분 차원 추출**(한 차원의 일부 값만 Time으로 이동 — 정규 설정은 항상 차원 전체를 옮김).

세 가지 예제로 이해하면 됩니다.
- **Example 1 (임의 순열)**: 안쪽 slice 4축을 `[3,2,1,0]`으로 뒤집기. ring_size=256이 필수인데, 순열이 모든 slice에 의존성을 만들고 반복 구조가 없어서 입출력 slice가 ring 인덱스상 임의로 멀 수 있기 때문입니다 (switch-engine.md:434-472).
- **Example 2 (다축 브로드캐스트)**: 비연속 두 축(`A%2`, `B%2`)을 Slice→Time으로 이동. `Broadcast01`은 브로드캐스트 축이 연속이어야 해서 못 하지만 커스텀 bitmap은 가능합니다 (switch-engine.md:474-510).
- **Example 3 (부분 축 추출 = slicing)**: `B%4`의 4개 값 중 3개만 Time으로 이동(`B % 4 = 3`은 버림). `Broadcast1`은 항상 전체를 옮기므로 못 합니다 (switch-engine.md:512-551). bitmap이 직접 보여줍니다: `bitmap[0] = {0,1,2}`면 3개 소스만 받음.

**CustomBroadcast의 여섯 제약** (switch-engine.md:553-602):
1. **브로드캐스트 축은 새 축** (입력 Slice/Time에 없어야 함).
2. **각 브로드캐스트 축은 출력 Slice에 정확히 한 번** (`m![A/4, X, X]`처럼 두 번 쓰면 무효).
3. **브로드캐스트 축에 패딩 금지** (`X # 4` 형태 불가 — 패딩 위치의 라우팅 목적지가 정의 안 됨).
4. **순서 보존**: Slice→Time으로 옮기는 축들은 입력 slice에서의 상대 순서를 지켜야 함. 라우터 버퍼가 패킷 하나뿐이라 재정렬이 불가능하기 때문입니다.
5. **가장 안쪽 Time 위치**: Slice→Time 축은 출력 Time의 가장 안쪽에 놓여야 함. 다른 slice 데이터가 패킷 안에서 마지막에 도착하기 때문. (`Broadcast01`은 `time0` 파라미터로 이 제약을 우회하지만 커스텀은 못 함.)
6. **ring_size는 2의 거듭제곱**이어야 하고, 컴파일러가 입출력 매핑에서 유도한 값과 일치해야 함.

### 설정 오버헤드

커스텀 snoop bitmap을 쓰면 설정 데이터를 Switch 엔진의 SFR(Special Function Register)로 스트리밍하는데, 이 SFR 쓰기가 **DMA 엔진과 sub 컨텍스트를 점유**합니다 (switch-engine.md:431-432). 즉 고정 사이클 stall이 아니라 "스케줄링 병렬성 감소"로 비용이 나타납니다.

---

## 2. Collect Engine — 32바이트 flit로 정규화

Collect는 임의 크기 패킷을 **딱 1 flit(32바이트)** 로 두 단계로 정규화합니다 (collect-engine.md:3-8):
1. **Pad**: 패킷을 32바이트 경계까지 채움 (이미 32B 정렬이면 생략).
2. **Split**: flit 경계에서 쪼갬 — 안쪽 32바이트가 `Packet2`가 되고, 바깥쪽 flit 개수는 `Time2`로 흡수됨 (이미 32B면 생략).

시그니처:
```rust
pub fn collect<Time2: M, Packet2: M>(self)
    -> CollectTensor<'l, T, D, Chip, Cluster, Slice, Time2, Packet2, B>
```
(collect.rs:43). `SwitchTensor`와 `FetchTensor` 둘 다 `.collect()`를 노출하며, `FetchTensor` 진입점은 slice 분배가 필요 없을 때 Switch 엔진을 건너뜁니다 (collect-engine.md:14-15).

### 네 가지 경우 (collect-engine.md:23-167)

dtype의 바이트 크기 × Packet 원소 수가 32바이트와 어떻게 비교되는지가 전부입니다.

- **Single-Flit (정확히 32B)**: 그대로 통과. 예: i8 32원소 = 32B → `Packet = m![B # 32]`, Time 불변 (collect-engine.md:25-54).
- **Sub-Flit (32B 미만)**: 32B로 패딩. 예: i8 16원소 = 16B → 16B 패딩 추가 (collect-engine.md:58-88).
- **Multi-Flit (32B 초과)**: flit로 쪼개고 바깥 flit 개수를 Time으로 흡수. 예: bf16 32원소 = 64B = 2 flit → 안쪽 16원소가 `Packet2 = m![B % 16]`, 바깥 2 flit이 `Time2 = m![A, B / 16]` (collect-engine.md:92-123).
- **Multi-Flit + Padding (32B 비정렬)**: 먼저 32B 배수로 패딩 후 쪼갬. 예: i8 51원소 = 51B → 64B 패딩 → 2 flit → `Time2 = m![A, B # 64 / 32]`, `Packet2 = m![B # 64 % 32]` (collect-engine.md:127-167).

이 네 경우는 `switch_assertions.rs`의 `alignment`, `packet` 모듈에 실제 커널로 다 들어 있습니다(예: `aligned_fetch_packet_bf16`가 bf16을 `collect::<m![A, B/16], m![B%16]>()`로 multi-flit 처리, switch_assertions.rs:114). `collect_time_mismatch`(switch_assertions.rs:233-251)는 bf16 64B를 잘못된 `Time2 = m![A]`로 받아서 "Collect time mismatch"를 내는 의도된 오답 예제입니다.

### TRF/VRF로 적재

정규화된 `CollectTensor`는 다운스트림 엔진으로 흘러가거나, `.to_trf()`/`.to_vrf()`로 레지스터 파일에 저장됩니다 (collect-engine.md:171). 자세한 reshape는 다음 섹션에서.

---

## 3. Register Files — TRF와 VRF의 내부 구조

Collect 엔진은 Contraction과 Vector로 스트리밍하고, 이 두 엔진은 각각 **slice당 하나씩** 있는 레지스터 파일에서 두 번째 피연산자를 가져옵니다 (register-files.md:3-4). 이 파일들은 소비 엔진이 돌기 전에 미리 채워져 있어야 합니다.

### TRF (Tensor Register File)

`.to_trf::<Lane, Element>(address)`가 `CollectTensor`를 `TrfTensor`로 만듭니다 (register-files.md:21, collect.rs:56-60):
```rust
pub fn to_trf<Lane: M, Element: M>(self, address: TrfAddress)
    -> TrfTensor<D, Chip, Cluster, Slice, Lane, Element, B>
```
`Chip`/`Cluster`/`Slice`는 그대로 통과, **`Lane`은 공간적 병렬성을 인덱싱(활성 lane 1/2/4/8개)**, `Element`는 lane당 레이아웃입니다 (register-files.md:17). 스트리밍 `Time`/`Packet`을 `Lane`/`Element`로 reshape하는 규칙 (register-files.md:29-34):
```text
Lane    = Time / FlitsPerLane
Element = [Time % FlitsPerLane, Packet]
```
`FlitsPerLane`은 컴파일러가 `Lane`과 `Time`에서 유도하며, 각 lane이 `FlitsPerLane`개의 연속 flit으로 채워집니다. 행렬곱에서는 보통 `Lane`이 출력 채널을, `Element`가 축약(contraction) 축을 담습니다 (register-files.md:37).

**검증 규칙** (collect.rs:121-139, verify_to_trf): ① `Lane::SIZE ∈ {1,2,4,8}`, ② 총 바이트 `Lane::SIZE × Element::SIZE × sizeof(D)`가 선택한 TRF 영역 용량에 맞아야 함, ③ `Lane::SIZE`가 `Time::SIZE`를 나눠야 하고 Time의 바깥 인자들이 Lane과 같아야 함. `contract_outer_assertions.rs`의 `lane_size` 모듈이 1/2/4/8은 valid, 3/16은 invalid임을 직접 보여줍니다 (contract_outer_assertions.rs:21-208).

**TrfAddress** (register-files.md:180-183, memory.rs:101): `Full`(TRF 전체), `FirstHalf`/`SecondHalf`(반씩 나눠 두 텐서가 독립 점유). 컴파일러가 결과 텐서의 총 바이트를 선택 영역 용량으로 한정합니다.

### TRF 아키텍처 (외울 숫자들)

> **TRF = 8 lanes × 2 banks × 128 rows × 320 bits = slice당 80 KB** (register-files.md:72)

8개 lane이 병렬로 동작하며 접근당 1/2/4/8개가 활성됩니다 (register-files.md:73). 320비트 한 행에 dtype별로 몇 원소가 들어가는지 (register-files.md:75-85):

| 타입 | 저장 원소 크기 | 행당 원소 |
|---|---|---|
| i4→i5 | 5비트 | 64 |
| i4→i9 | 9비트 | 32 |
| i8/f8 | 8비트 | 32 (행당 40바이트) |
| bf16 | 16비트 | 16 (행당 32바이트) |

**i4만 저장 시 승격(promote)** 됩니다: `i4→i5`, `i4→i9`. fetch 어댑터의 선택적 zero-point 빼기 여유를 두기 위함입니다(니블당 1비트 확장 가능). i8/f8, bf16은 네이티브 폭(8/16비트) 유지 (register-files.md:84-85).

**활성 lane이 8개 미만이면 행이 늘어난 것처럼 동작**합니다. 활성 수를 절반으로 줄이면 활성 lane당 행이 두 배가 됩니다 (4활성 → bank당 256행, 1활성 → 1024행) (register-files.md:88).

### Contraction 엔진으로의 읽기 (대역폭)

각 읽기는 8 lanes × (1 또는 2) banks × 1 row × bank당 320비트를 커버합니다 (register-files.md:66-67): **narrow read(1 bank) = lane당 320비트, wide read(2 banks) = lane당 640비트**. slice당으로는 narrow 320 B/cycle, wide 640 B/cycle (8 lane 합산). 즉 lane당 320비트 = 40바이트, 8 lane이면 320바이트. wide면 640바이트.

### 이중 버퍼링 + 캐시 + bank 교번

TRF는 각 bank를 두 반으로 나눠 이중 버퍼링을 합니다: 한 반을 sequencer가 읽는 동안 store가 다른 반을 채우고, 반복 사이에 뒤집습니다 (register-files.md:92). 세 주소 모드는 store 시점에 고정됩니다: `Full`(bank당 128행), `FirstHalf`(0-63행), `SecondHalf`(64-127행). 반 모드는 slice당 용량을 **40 KB로 상한** (register-files.md:93-94).

두 반이 같은 bank를 공유하므로 행이 달라도 bank 수준에서 읽기/쓰기가 경합합니다. 같은 cycle에 같은 bank를 노리면 **읽기가 우선**입니다(contraction 파이프라인이 이번 cycle에 데이터가 필요하고, store는 기다릴 수 있으니까) (register-files.md:98-99). 이 경합을 두 장치로 완화합니다 (register-files.md:101-111):
- **read cache**: 8 lanes × 2 banks × 4 rows × 320비트 = **2.5 KB** 직접 사상 캐시. TRF 읽기는 같은 데이터를 여러 cycle 브로드캐스트하는 재사용이 많아서 효과적. hit면 bank를 건너뛰어 store가 그 cycle에 bank를 씀.
- **bank 교번(narrow read 한정, ≤32B)**: 읽기가 1 bank만 쓰므로 32바이트 단위로 두 bank를 번갈아 사용. 그러면 읽기/쓰기가 연속 cycle에 다른 bank에 놓여 캐시 miss에도 경합 회피. **wide read(64B)는 매 cycle 두 bank를 다 쓰므로 miss 때마다 동시 store를 막습니다.**

### 데이터 메모리에서 직접 (short command)

완전히 연속 접근(틈/재정렬 없음)이면 TRF는 **StoTRF**(VRF는 **StoVRF**)라는 짧은 명령으로 DM에서 직접 적재해 Fetch→Switch→Collect→to_trf() 전체 파이프라인을 우회합니다 (register-files.md:60-62, 157-160). 임의 레이아웃 지원을 포기하는 대신 셋업 오버헤드가 낮습니다.

### VRF (Vector Register File)

`.to_vrf::<Element>(address)`는 flit을 raw `Address`에 저장합니다(영역 선택 없음) (collect.rs:72, register-files.md:130-136):
```text
Element2 = [Time, Packet]   // 스트리밍 Time/Packet을 그냥 평탄화
```
**중요한 제약**: `.to_trf`는 아무 `Scalar` 타입이나 받지만, **`.to_vrf`는 `VeScalar`(i32 또는 f32)만** 받습니다 — 다운스트림 Vector 엔진이 그 타입만 먹기 때문입니다 (register-files.md:203, collect.rs:72, scalar.rs:18-27). (참고: book의 VRF Architecture 절은 본문이 비어 있어 banking 세부는 문서화되어 있지 않습니다 — register-files.md:163-165.)

---

## 4. Contraction Engine — matmul/conv/attention의 심장

Contraction은 두 텐서를 받아 공유 축을 따라 축약하는 이항 연산(matmul, convolution)을 합니다 (contraction-engine/index.md:3). Quick Start의 핵심을 다시 보면: 축약은 **Broadcast → Multiply → Reduce** 세 단계로 분해되고, 한 피연산자는 Collect에서 스트리밍, 다른 하나는 TRF에 상주합니다 (index.md:6-9).

### 4단계 파이프라인 (index.md:15-62)

작업를 네 단계로 나누되, **Broadcast/Multiply가 1단계, Reduce가 3단계**입니다. 각 단계는 자기만의 겹치지 않는 차원을 다룹니다.

```
Collect ─┐
         ├─► [Outer: Stream Adapter + TRF Sequencer → Multiply] ─► Packet Reducer ─► Time Reducer ─► Lane Folder ─► Vector
TRF ─────┘
```
(index.md:18-52)

- **Outer (Broadcast & Multiply)**: 두 피연산자를 `[Chip,Cluster,Slice,Lane,Time,Packet]` 공통 모양으로 브로드캐스트하고 원소 단위 곱. 세 하위 단계: Stream Adapter(스트리밍 피연산자 브로드캐스트), TRF Sequencer(TRF 피연산자 브로드캐스트), Multiplier(타입 확장 후 곱). (index.md:54-56)
- **Packet Reducer (Packet 안 축약)**: Packet에 사상된 축약 축을 lane당 병렬 트리로 reduce-add. (index.md:57)
- **Time Reducer (Time 가로질러 축약)**: 매 cycle 결과를 공유 누산기에 누적. (index.md:58)
- **Lane Folder (Lane 접기)**: `Lane`을 `OutPacket` 또는 `OutTime`으로 흡수해 출력 스트림으로 내보냄(합산 아님). slice/chip 간 축약은 다운스트림 Vector 엔진이 처리. (index.md:59-60)

> **핵심 하드웨어 상한**: Outer 단계가 `Lane ≤ 8`, `Packet ≤ 64 B`로 제한합니다(RNGD) (index.md:62).

### 4-1. Outer — 외적(outer product)의 하드웨어 구현

이름이 outer product에서 왔습니다: 벡터 u(길이 n), v(길이 m)에 대해 `u v^T`는 u를 열축으로, v를 행축으로 브로드캐스트하고 원소 단위로 곱한 n×m 행렬입니다 (outer.md:5-7). Outer의 세 하위 단계가 정확히 이 의미를 직렬로 구현합니다.

`.contract_outer(&trf)` 시그니처 (outer.md:18-22, contraction/outer/mod.rs:74-76):
```rust
pub fn contract_outer<OutTime: M, OutPacket: M, Lane: M, TrfElement: M>(
    self, trf_tensor: &TrfTensor<D, Chip, Cluster, Slice, Lane, TrfElement, B>) -> ...
```

#### Stream Adapter: Packing + Broadcast

스트리밍 `Time`/`Packet`을 연산 모양(`Lane`/`OutTime`/`OutPacket`)으로 바꿉니다. 컴파일러가 세 자유 변수 `PackSize`, `LaneBroadcast`, `TimeBroadcast`를 유도합니다 (outer.md:35-42):
```text
Lane      = LaneBroadcast
OutTime   = [Time / PackSize, TimeBroadcast]
OutPacket = [Time % PackSize, Packet] # (64 / D::SIZE)
```

**Packing**: Collect는 32B flit을 내지만 Outer는 `PackSize × 32`B 패킷(RNGD에서 32 또는 64)을 냅니다. `PackSize ∈ {1,2}`개의 연속 flit을 한 패킷으로 합칩니다 (outer.md:46-47). 공식: **`PackSize = OutPacket::SIZE × D::SIZE / 32`** (outer.md:55). 즉 사용자가 OutPacket을 32B로 고르면 PackSize=1, 64B로 고르면 PackSize=2. 하드웨어는 내부적으로 항상 64B 패킷으로 동작하고, PackSize=1이면 남는 32B 절반은 0으로 채워지지만 논리 OutPacket에는 전파되지 않습니다 (outer.md:57).

**Broadcast**: packing 후 공간적으로 `LaneBroadcast`(TRF의 Lane 매핑, ∈{1,2,4,8}), 시간적으로 `TimeBroadcast`로 복제합니다 (outer.md:61). `TimeBroadcast`는 TRF `Element` 중 입력 `Time`에 없는 인자들과, 입력에도 TRF에도 없는 순수 출력 축(예: einsum `AB,BC->ABCD`의 D)을 커버합니다 (outer.md:62, 29). **TimeBroadcast 인자는 `OutTime`의 가장 안쪽**에 놓입니다 — 같은 OutPacket을 이 인자들에 걸쳐 다시 보낸 뒤에야 바깥 PackTime 인자로 넘어갑니다 (outer.md:71).

**제약** (outer.md:99-101): `OutPacket::SIZE × D::SIZE ∈ {32,64}`바이트, `PackSize ∈ {1,2}`, `Lane::SIZE ∈ {1,2,4,8}`.

**성능** (outer.md:105-114): **PackSize가 MAC 활용률을 정합니다.** PackSize=2는 64B를 다 채워 모든 MAC 사용, PackSize=1은 32B만 채워서 0 곱하기 절반이 낭비 → 유효 처리량 절반. PackSize=2는 패킷당 2 cycle이지만, 업스트림이 매 cycle 32B flit 1개를 공급하므로 병목이 아닙니다. Time Broadcasting은 같은 패킷을 re-fetch 없이 재사용해 대역폭 비용을 없앱니다.

#### TRF Sequencer: ReadSize로 Element를 펼치기

`TrfTensor`를 읽어 `Element`를 `OutTime`/`OutPacket`으로 reshape합니다 (outer.md:117-127):
```text
OutTime   = ([Element / ReadSize] 시퀀싱 + 브로드캐스트)
OutPacket = [PacketBroadcast, Element % ReadSize]
```
매 cycle 한 번의 TRF 읽기가 OutPacket 하나를 채웁니다 (양 bank 640비트/lane, 한 bank 320비트/lane) (outer.md:129). 읽기는 `Element`의 가장 안쪽 연속 부분을 끌어와 64B OutPacket을 채우도록 복제합니다. 컴파일러는 **`Element % ReadSize == OutPacket % ReadSize`이고 `ReadSize × D::SIZE ≤ 64`바이트**인 가장 큰 `ReadSize`를 고릅니다 — 넓으면 두 bank, 좁으면 한 bank (outer.md:131-132). 두 예제: full-read는 ReadSize가 Element 전체를 한 번에(outer.md:142-158), partial-read는 Element/ReadSize가 비자명해서 sequencer가 바깥 인자를 반복(outer.md:163-179). **주소 정렬 제약**: Element%ReadSize가 64B를 다 덮으면 양 bank를 걸쳐 읽으므로 base 주소와 모든 stride가 64B 정렬이어야 합니다 (outer.md:185).

#### Multiplier: 타입 확장 후 곱

Stream Adapter와 TRF Sequencer에서 정렬된 두 피연산자를 받아, 누산기 오버플로 방지를 위해 각 원소를 축약 출력 타입으로 확장합니다: **`i4`/`i8` → `i32`, `f8`/`bf16` → `f32`** (outer.md:200, index.md:56). 그리고 원소 단위로 곱합니다. 출력은 `[Chip,Cluster,Slice,Lane,Time,Packet]` 단일 텐서이고, 매 Time cycle마다 모든 Lane이 패킷 하나 분량의 곱을 병렬로 냅니다 (outer.md:201-202).

### 4-2. Packet Reducer — lane당 reduction tree

`.contract_packet()`이 호출합니다. 각 lane이 32B/64B 패킷(i4/i8/f8/bf16)을 받아 Packet 안의 축약 축을 더합니다 (packet-reducer.md:7-9):
```text
ReducePacket = Packet / 2^d      (0 ≤ d ≤ log2(Packet::SIZE))
OutPacket = ReducePacket         (ReducePacket::SIZE ≤ 32이면)
            아니면 32로 클립
```
(packet-reducer.md:39-43). lane당 독립 **reduction tree**를 돌립니다: depth 0에서 잎이 패킷 원소를 갖고, 매 depth마다 쌍을 더해 원소 수를 절반으로 (packet-reducer.md:45-48). **최대 트리 깊이 = log2(Packet::SIZE)**: i4는 7(128원소), i8/f8는 6(64), bf16은 5(32) (packet-reducer.md:47).

> **왜 OutPacket이 32원소로 상한인가?** 다운스트림 Time Reducer의 lane당 누산기 열이 **32개뿐**이라서입니다 (packet-reducer.md:50). `ReducePacket::SIZE > 32`면 바깥 dummy를 슬라이스하고 안쪽 32원소만 남깁니다. 예: i4는 128원소로 들어와서 d∈{0,1}이 128/64 ReducePacket을 내도 둘 다 32로 클립 (packet-reducer.md:51-52).

**성능** (packet-reducer.md:56-62): 지연 = 트리 깊이(i4 7, i8/f8 6, bf16 5 cycle). 더 넓은 타입은 깊이가 얕음(패킷에 원소가 덜 들어가니까). adder tree는 완전 파이프라인이라 깊이는 first-output 지연만 늘리고 정상 상태 처리량(매 cycle 1패킷 in/out)은 안 줄임. **Lane<8이면 비활성 lane의 트리가 놀아서 처리량이 `Lane::SIZE/8`로 비례 감소.**

### 4-3. Time Reducer — 누산기 슬롯

`.contract_time::<OutTime>()`. Packet Reducer의 `[Lane, Packet]` 출력을 `Time`에 걸쳐 `OutTime`으로 누적합니다(temporal accumulator) (time-reducer.md:3-7). `OutTime`은 살아남는 Time 차원을 지명하고 나머지는 합산됩니다 (contraction/time.rs:22-25). 제약: **`OutTime`은 `Time`의 부분집합이고 살아남는 차원의 상대 순서가 보존**되어야 함(verify_contract_time) (time-reducer.md:52).

핵심 개념 **InnerTime**: 가장 바깥 reduce 차원보다 안쪽이면서 OutTime에 살아남는 차원들. 예제 `reduce_b`에서 `Time = m![B/4, A%8]`, `OutTime = m![A%8]`이면 `B/4`가 가장 바깥 reduce 차원(Time::SIZE = 2×8 = 16 flit을 반복), `InnerTime = m![A%8]` (time-reducer.md:55-56). 누적은 **InnerTime::SIZE개 슬롯**이 필요합니다(InnerTime 튜플 값마다 하나). 같은 튜플의 flit이 같은 슬롯에 누적됩니다 (time-reducer.md:58). reduce_b는 8슬롯이 B/4=2회에 걸쳐 누적되어 flit 15 이후 최종 결과를 Lane Folder로 넘깁니다 (time-reducer.md:59-82).

**슬롯 용량** (time-reducer.md:84-97): 버퍼의 **1,024 cells**와 다운스트림 Lane Folder의 LaneMode가 정합니다. 각 슬롯은 `[Lane, Packet]` 청크이고 모양은 LaneMode가 결정:

| LaneMode | 청크 모양 | 청크당 cell | 슬롯 용량 |
|---|---|---|---|
| Interleaved | `[Lane # 8, Packet]` | 8 × Packet::SIZE | 128 / Packet::SIZE |
| Sequential | `[Lane, Packet # 32]` | Lane::SIZE × 32 | 32 / Lane::SIZE |

`InnerTime::SIZE`가 슬롯 용량을 넘으면 Time을 더 쪼개거나 LaneMode를 바꿔 처리량과 슬롯 여유를 맞바꿉니다 (time-reducer.md:96-97).

**성능** (time-reducer.md:99-104): 입력측 처리량은 매 cycle 1패킷, 출력 유효율은 `1/N`(N개를 1개로 축약), 지연은 약 N cycle.

### 4-4. Lane Folder — Lane 차원 제거

Contraction의 마지막 단계. `Lane`의 8개 값을 `OutPacket`(Interleaved) 또는 `OutTime`(Sequential)으로 재배치합니다. **합산이 아니라 접기(fold)** 입니다 (lane-folder.md:3-5). 업스트림 Time Reducer 버퍼를 8원소 폭 출력 버스로 매 cycle 하나씩 비우며, LaneMode가 각 cycle flit이 뭘 담을지 정합니다 (lane-folder.md:10-11). 입력 Packet은 Packet Reducer에서 살아남은 `{1,2,4,8,16,32}` 중 하나입니다 (lane-folder.md:17).

**Interleaved** (lane-folder.md:19-26): Lane이 OutPacket으로 접힘. 매 cycle 8개 lane의 한 열 위치를 읽음(lane당 1값, flit당 8값). Lane이 가장 안쪽 OutPacket으로 materialize.
```text
OutTime   = [Time, Packet]
OutPacket = [Lane # 8]
```

**Sequential** (lane-folder.md:44-55): Lane이 OutTime으로 접힘. 매 cycle 한 lane의 Packet에서 8개 열을 읽음(flit당 8값), Lane은 cycle을 거듭하며 반복. 각 cycle이 8원소 폭이라 Packet을 먼저 8의 배수로 패딩 후 쪼갬:
```text
PadPacket = Packet # align_up(Packet::SIZE, 8)
OutTime   = [Time, Lane, PadPacket / 8]
OutPacket = [PadPacket % 8]
```

**Lane Folder 자체엔 제약이 없습니다.** 여기서 고른 LaneMode가 업스트림 Time Reducer의 슬롯 용량 한계를 결정합니다 (lane-folder.md:73-76). 성능: Interleaved는 Lane<8이면 처리량 `Lane::SIZE/8`로 감소(빈 버스 칸), Sequential은 Packet::SIZE<8이면 매 cycle 딱 그만큼만 실어 나름(패딩/낭비 없음) (lane-folder.md:79-83).

### 4-5. 매핑 전략: 어느 축을 Time에 둘 것인가 (Batched MatMul VMK,KN→VMN)

같은 batched matmul도 어느 축을 `Time`에 두느냐로 성능이 천양지차입니다 (index.md:66-201).

- **K in Time (퇴화 예제)**: 축약 축 K를 Time에 두면 Packet이 `1 # 32`로 패딩되어 축약이 Packet Reducer 트리(공간) 대신 cycle별 순차로 진행됩니다. 그러면 **32개 곱셈기 중 1개만 일함 = bf16에서 1/32 MAC 활용률**. 교육용 baseline일 뿐입니다 (index.md:88).
- **M in Time / V in Time (권장)**: K를 `Packet`에 두어 Packet Reducer 트리로 병렬 축약하고, 살아남는 축(V,M,N)을 Cluster/Slice/Lane에 펼쳐 공간 병렬성을 최대화합니다 (index.md:123-128, 166-168). K가 Packet보다 크면 K를 Packet(공간)과 Time(시간)으로 쪼갭니다.

교훈: **축약 축은 Packet에, 출력 축은 spatial(Slice/Lane)+Time에.** base-template의 gemm_kernel이 정확히 이 패턴입니다 — K를 collect로 packet에 넣고(`m![K%16]`), contract_outer→packet→time→lane으로 흐릅니다 (gemm_kernel.rs:38-45).

---

## 5. 2D Convolution — Stream Adapter의 sliding-window 확장

2D conv는 einsum `$(H+Fh)$(W+Fw)K, FhFwKC -> HWC`입니다 (2d-convolution.md:4). einsum(행렬곱)만 쓸 거면 이 섹션은 건너뛰어도 됩니다 (2d-convolution.md:88). conv가 특별한 이유는 **sliding window 데이터 재사용**이 필요하기 때문입니다 — 겹치는 윈도우를 매번 re-fetch하면 낭비라서, Stream Adapter가 세 가지 확장으로 한 번 fetch한 데이터를 shift해 여러 윈도우를 만듭니다 (2d-convolution.md:85-86).

### 세 가지 확장

**① Flit Buffer (feed_flits)** (2d-convolution.md:90-99): 기본 2-flit 버퍼를 `feed_flits: 3`으로 늘려 96바이트(3 flit)를 다 채웁니다. 세 번째 flit이 shift 유닛에 re-fetch 없이 윈도우를 옮길 여유 데이터를 줍니다. `feed_flits ∈ {1,2,3}` — 96바이트 물리 버퍼 한계 때문입니다 (2d-convolution.md:400).

**② Transpose** (2d-convolution.md:101-146): Packet Reducer는 가장 안쪽 축부터 인접 쌍을 reduce하므로 **축약 축이 반드시 가장 안쪽**이어야 합니다. 들어온 데이터의 축 순서가 다르면 32B flit 안에서 축을 재정렬합니다. 지원 transpose는 dtype별로 (총 부피는 항상 32B):

| 타입 | 지원 transpose |
|---|---|
| i4 | `[4][16]→[16][4]` |
| i8/f8 | `[2][16]→[16][2]`, `[4][8]→[8][4]` |
| bf16 | `[2][8]→[8][2]` |

i32/f32는 Packet Reducer 트리를 못 쓰므로 여기 없습니다 (2d-convolution.md:116-117). transpose는 단일 flit 범위로만(고정 기능 순열망), 1-2 cycle 지연 추가 (2d-convolution.md:411, 422).

**③ Shift (Stream Shift Unit)** (2d-convolution.md:148-392): re-fetch 대신 데이터를 한 번 가져와 shift해 여러 윈도우를 만듭니다. 세 파라미터:
- `initial_shift`: 데이터가 처음 버퍼에 들어올 때 시작 오프셋. 음수면 늦은 위치로(높은 주소) 밀고 앞을 0 패딩, 양수면 이른 위치로 밀고 뒤를 0 패딩 (2d-convolution.md:160-165).
- `shift_stride`: shift_dim을 따라 매 반복 shift할 양.
- `pop_dim`: 새 데이터를 fetch할 시점을 표시(이 인덱스가 증가하면 새 데이터 로드 + initial_shift 재적용) (2d-convolution.md:292).

**shift 범위(레지스터 체인 한계)** (2d-convolution.md:169-176, 402): i4 [-15,16](32값), i8 [-7,8](16값), bf16 [-3,4](8값). 이 distinct 값 수(32/16/8)는 32B flit 원소 수의 절반과 일치합니다. **shift_stride 범위**: i4 0-31, i8 0-15, bf16 0-7 (2d-convolution.md:299-303). 차원 순서는 `tile → shift_dim → pop_dim`(안→밖). shift_dim이 아닌 pop_dim 아래 인덱스는 tiled(broadcast) 출력을 냅니다 (2d-convolution.md:295).

### 4가지 conv 변형 (2d-convolution.md:11-81)

Stream Adapter가 sliding window를 어떻게 shift하느냐로 갈립니다: **Filter-Stride 1**(Fetch가 H 슬라이딩, Stream Adapter가 W 슬라이딩 shift-reuse), **Filter-Stride 2**(shift-stride 2, 출력 위치당 1 shift; feed_flits=3으로 MAC 채움), **Dilation 2**(shift-stride 2 + 2 shifts로 dilated 위치 추출), **Stride2+Dilation2**(조합). Stride2+Dilation2에서는 TRF가 dummy 슬롯에 0을 담아야 `1+1#`과 `1+1z`의 축약이 1이 되도록 합니다(`1z`는 0으로 채운 dummy 패딩) (2d-convolution.md:72-73).

### 설계 근거 (왜 이런 한계인가) (2d-convolution.md:406-418)

- **96바이트 Flit Buffer**: 다운스트림 Packet Reducer가 단일 cycle 접근하려면 SRAM이 아니라 레지스터 파일 저장이 필요한데, 레지스터 파일은 비트당 면적이 훨씬 비싸서 96바이트가 유용성과 실리콘 비용의 절충점.
- **단일 flit Transpose**: 흔한 경우(축약 축 안쪽)를 빠르게. 다중 flit이면 훨씬 큰 순열망이나 다중 cycle 버퍼 필요.
- **shift 한계**: 흔한 필터(3×3,5×5,7×7)를 지원하면서 비용 합리화.
- **TRF Sequencer 정렬**: Outer의 곱셈과 Packet Reducer 트리는 **버퍼링·재정렬이 안 되는 고정 기능 MAC 배열**이라, 매핑이 어긋나면 우아한 성능 저하가 아니라 **틀린 계산**이 나옵니다. Stream Adapter 출력은 TRF Sequencer contraction 매핑과 반드시 일치해야 합니다 (2d-convolution.md:403, 417-418).

## 2. 핵심 API · 패턴

| 이름 | 쓰는 법 | 설명 | 출처 |
|---|---|---|---|
| `FetchTensor::switch / SwitchTensor::switch` | `pub fn switch<OutSlice: M, OutTime: M>(self, config: SwitchConfig) -> SwitchTensor<...OutSlice, OutTime, Packet...>` | Slice/Time 매핑만 바꾸고 Chip/Cluster/Packet/값은 보존. InSlice::SIZE == OutSlice::SIZE 강제(verify_switch). | `furiosa-opt-std/src/engine/switch.rs:678-700` |
| `SwitchConfig` | `enum { Broadcast01{slice1,slice0,time0}, Broadcast1{slice1,slice0}, Transpose{slice1,slice0}, InterTranspose{slice1,slice0,time0}, CustomBroadcast{ring_size}, TransposedBroadcast1{slice1,slice0} }` | Forwarding 변형은 없음 — 교환이 필요 없으면 switch를 건너뛰고 collect를 직접 호출. | `furiosa-opt-std/src/engine/switch.rs:31-89, switch-engine.md:46-47` |
| `collect` | `pub fn collect<Time2: M, Packet2: M>(self) -> CollectTensor<...Time2, Packet2...>` | 패킷을 32B flit로 정규화(pad + split). SwitchTensor/FetchTensor 둘 다 노출. verify_collect가 Time2/Packet2 정합성 검증. | `furiosa-opt-std/src/engine/collect.rs:43-47` |
| `to_trf` | `pub fn to_trf<Lane: M, Element: M>(self, address: TrfAddress) -> TrfTensor<...Lane, Element...>` | Lane::SIZE ∈ {1,2,4,8}, Lane::SIZE가 Time::SIZE를 나눠야 함, 총 바이트가 영역 용량에 맞아야 함(verify_to_trf). | `furiosa-opt-std/src/engine/collect.rs:56-60, 121-139` |
| `to_vrf` | `pub fn to_vrf<Element: M>(self, address: Address) -> VrfTensor<...Element...>  // D: VeScalar` | Element2 = [Time, Packet] 평탄화. i32/f32만 허용(VeScalar bound). | `furiosa-opt-std/src/engine/collect.rs:66-76` |
| `TrfAddress` | `enum TrfAddress { Full, FirstHalf, SecondHalf }` | Full=128행/bank, 반 모드는 40KB/slice 상한. 이중 버퍼링용. | `furiosa-opt-std/src/tensor/memory.rs:101-127, register-files.md:93-94` |
| `contract_outer` | `pub fn contract_outer<OutTime: M, OutPacket: M, Lane: M, TrfElement: M>(self, trf_tensor: &TrfTensor<...Lane, TrfElement...>) -> ContractOuterTensor` | Stream Adapter + TRF Sequencer 브로드캐스트 후 Multiplier가 타입 확장(i4/i8→i32, f8/bf16→f32) 곱. OutPacket::SIZE×D::SIZE ∈ {32,64}. | `furiosa-opt-std/src/engine/contraction/outer/mod.rs:74-99, outer.md:99` |
| `contract_packet` | `pub fn contract_packet<OutPacket: M>(self) -> ContractPacketTensor` | lane당 reduction tree로 Packet 내 축약. OutPacket 32원소 상한(Time Reducer 누산기 32열). | `furiosa-opt-std/src/engine/contraction/packet.rs:16-45, packet-reducer.md:50` |
| `contract_time` | `pub fn contract_time<OutTime: M>(self) -> ContractTimeTensor` | OutTime은 Time의 부분집합(상대 순서 보존, verify_contract_time). 없는 차원은 reduce-add. | `furiosa-opt-std/src/engine/contraction/time.rs:22-25, 35-38` |
| `contract_lane` | `pub fn contract_lane<OutTime: M, OutPacket: M>(self, mode: LaneMode) -> ContractTensor` | Lane을 OutPacket(Interleaved) 또는 OutTime(Sequential)으로 접음(합산 아님). 고른 mode가 Time Reducer 슬롯 용량을 결정. | `furiosa-opt-std/src/engine/contraction/lane.rs:45-51, lane-folder.md:73-76` |
| `LaneMode` | `enum LaneMode { Interleaved, Sequential }` | Interleaved=lane을 패킷에 끼워 넣어 매 cycle 8 lane 한 열, Sequential=lane을 시간으로 순차. | `furiosa-opt-std/src/engine/contraction/lane.rs:21-26` |

## 3. 실험 (직접 돌리기)

> 실험은 NPU 없이 `simulation`·`typecheck`로 돌아갑니다. 실행법은 [`../experiments/README.md`](../experiments/README.md), MNIST는 `cargo furiosa-opt test`(npu 전용).

### 실험 06.1 — GEMM 커널을 시뮬레이션으로 돌려 4단계 파이프라인 관찰
*난이도 1/5 · 기반: `base-template/src/kernel/gemm_kernel.rs`*

**목표** — contract_outer→contract_packet→contract_time→contract_lane의 전체 체인이 실제로 도는 것을 보고, 호스트 레퍼런스와 일치하는지 확인한다.

```bash
cargo furiosa-opt run --release --bin gemm
cargo furiosa-opt test --release --bin gemm
```
**관찰** — run은 "GEMM: kernel ran" 출력. test는 matches_reference가 통과(C[i,j]=sum_k A·B를 f32로 계산 후 bf16 비교, 허용오차 5% 또는 1.0). gemm_kernel이 K를 packet(K%16)에 두고 Lane=J%8에 출력 채널을 펼친 권장 매핑임을 코드에서 확인.

**심화** — gemv, dot_product 바이너리도 같은 방식으로 돌려 Time/Packet/Lane 배치 차이를 비교(gemv는 J를 Time+Packet으로 쪼개 reduce, dot_product는 Packet에만).

### 실험 06.2 — Switch 정규 설정 5종의 valid 경우 타입체크/시뮬레이션
*난이도 2/5 · 기반: `furiosa-opt-examples/src/switch_assertions.rs`*

**목표** — Broadcast1/Broadcast01/Transpose/InterTranspose 매핑이 컴파일러 검증을 통과하는지 확인하고, 각 OutSlice/OutTime 구조를 코드와 대조한다.

```bash
cargo furiosa-opt test --test switch_assertions_tests
```
**관찰** — test_valid_basic, test_valid_only_slice1, test_valid_single_axis, test_valid(inter_transpose) 등 valid_* 테스트가 모두 통과. invalid_* 함수들은 테스트에서 호출되지 않음(컴파일 타임 panic 예제라 의도적으로 제외).

**심화** — switch_assertions.rs의 broadcast1::valid_basic에서 SwitchConfig::Broadcast1{slice1:4, slice0:4}를 slice1:0으로 바꾸면 invalid_slice1_zero와 같은 검증 실패가 나는지 예측한 뒤, 임시로 valid_basic을 복제해 확인.

### 실험 06.3 — Collect 네 가지 정규화 경우를 dtype 바꿔가며 예측-후-확인
*난이도 2/5 · 기반: `furiosa-opt-examples/src/switch_assertions.rs`*

**목표** — 같은 원소 수라도 dtype(i8 vs bf16)에 따라 패킷 바이트가 달라져 single/multi-flit 분기가 바뀜을 체득한다.

```bash
cargo furiosa-opt test --test switch_assertions_tests
```
**관찰** — test_aligned_fetch_packet_i8(B=32×1B=32B=single-flit)와 test_aligned_fetch_packet_bf16(B=32×2B=64B=2 flit, collect를 m![A,B/16],m![B%16]로)이 둘 다 통과. 같은 axes B=32인데 collect 매핑이 dtype 때문에 다른 점을 코드(switch_assertions.rs:103-118)에서 확인.

**심화** — collect-engine.md의 collect_multi_flit_padded(B=51 i8 → 64B 패딩 → 2 flit) 매핑을 종이에 그린 뒤, 새 함수로 옮겨 타입체크가 통과하는지 본다.

### 실험 06.4 — contract_outer Lane 크기 1/2/4/8 valid, 3/16 invalid 검증
*난이도 3/5 · 기반: `furiosa-opt-examples/src/contract_outer_assertions.rs`*

**목표** — Lane::SIZE ∈ {1,2,4,8} 제약을 손으로 확인하고, 비허용 크기가 어디서 막히는지 본다.

```bash
cargo furiosa-opt test --test contract_outer_assertions_tests
```
**관찰** — lane_size 모듈의 test_valid_size_1/2/4/8이 통과. invalid_size_3, invalid_size_16은 테스트에서 호출되지 않음(컴파일 타임 검증 실패 예제). 코드(contract_outer_assertions.rs:148-208)에서 trf의 Lane 매핑이 1#3 / 1#16으로 잘못된 점 확인.

**심화** — valid_size_8을 복제해 contract_lane의 OutPacket을 m![R # 8]에서 m![R # 16]로 바꾸면 Lane Folder/누산기 관련 제약이 어떻게 반응하는지 예측한 뒤 확인.

### 실험 06.5 — find-the-error: collect Time2 일부러 깨뜨려 'Collect time mismatch' 재현
*난이도 3/5 · 기반: `furiosa-opt-examples/src/switch_assertions.rs`*

**목표** — Collect의 Time/Packet 정합성 검증이 어떤 매핑에서 어떻게 실패하는지 직접 본다.

```bash
cargo furiosa-opt test --test switch_assertions_tests   # 기준 통과 확인 후, 아래 수정 적용
# switch_assertions.rs packet::valid_padding 의 collect::<m![A], m![B]>() 를
# collect::<m![A], m![B % 16]>() 로 바꾸고(bf16 가정 필요시 dtype도 bf16로) 다시 test
```
**관찰** — collect_time_mismatch(switch_assertions.rs:233-251)가 보여주듯, bf16 64B(2 flit)를 Time2=m![A]로 받으면 바깥 flit이 흡수되지 않아 verify_collect가 'Collect time mismatch'로 panic. 올바른 Time2는 m![A, B/16].

**심화** — i8(1B)로 같은 매핑을 시도하면 32B라 single-flit이므로 통과함을 예측-확인해, '바이트가 분기를 결정한다'는 점을 재확인.

### 실험 06.6 — switch 사이클 공식을 코드 파라미터로 직접 계산-검증
*난이도 2/5 · 기반: `furiosa-opt-examples/src/switch_assertions.rs`*

**목표** — ring_size × Time::SIZE × flits_per_packet 공식을 실제 예제 파라미터에 적용해 문서의 사이클 수와 맞춰본다(NPU 불필요, 종이+코드).

```bash
cargo furiosa-opt test --test switch_assertions_tests   # valid_only_slice1 등이 컴파일/시뮬 통과하는지만 확인
```
**관찰** — broadcast01::valid_only_slice1(slice1:256, slice0:1 → ring_size=256)과 broadcast1::valid_basic(slice1:4, slice0:4 → ring_size=16) 같은 경우에서, switch-engine.md:115·168의 공식을 적용하면 사이클이 나온다. flits_per_packet = sizeof(i8)×Packet::SIZE/32 임을 dtype/Packet에서 계산.

**심화** — 같은 매핑을 bf16으로 바꾸면 flits_per_packet이 2배가 되어 사이클이 2배가 됨을 손으로 계산해 확인.

## 4. 연습문제 (손으로, 컴파일 없이)

**Q1.** axes![A=256, B=64, C=63]에서 입력 Slice=m![A], Packet=m![C # 64], dtype=f32. SwitchConfig::Broadcast01{slice1:2, slice0:2, time0:4}를 적용한다. 사이클 수를 구하라(Time::SIZE=64). flits_per_packet과 ring_size도 명시할 것.

<details><summary>정답/힌트</summary>

ring_size = slice1×slice0 = 2×2 = 4. flits_per_packet = sizeof(f32)×Packet::SIZE/32 = 4×64/32 = 8. cycles = 4×64×8 = 2048. (switch-engine.md:115와 일치)

</details>

**Q2.** 같은 switch 작업을 dtype만 i8로 바꾸고 Packet=m![C # 64]로 두면 사이클이 어떻게 변하나? 왜?

<details><summary>정답/힌트</summary>

flits_per_packet = 1×64/32 = 2로 줄어든다. cycles = 4×64×2 = 512. dtype이 작아 패킷의 flit 수가 줄어 순회 반복이 줄기 때문.

</details>

**Q3.** Packet Reducer에 들어온 패킷이 bf16 32원소다. 모두 합해 1원소로 만들려면 reduction tree 깊이는? i4 128원소면?

<details><summary>정답/힌트</summary>

bf16: log2(32)=5. i4: log2(128)=7. 단 i4는 OutPacket이 32원소 상한이라 d∈{0,1}로 128/64를 낸 뒤 32로 클립될 수 있음(packet-reducer.md:47-52).

</details>

**Q4.** collect_multi_flit_padded: i8, Packet=m![B](B=51 원소)를 collect한다. 패딩 후 바이트, flit 수, 결과 Time2/Packet2 매핑을 써라.

<details><summary>정답/힌트</summary>

51B → 32배수로 패딩 → 64B = 2 flit. Time2 = m![A, B # 64 / 32](바깥 2 flit 흡수), Packet2 = m![B # 64 % 32](안쪽 32B). (collect-engine.md:133-167)

</details>

**Q5.** Time Reducer를 LaneMode::Sequential, Lane::SIZE=1로 쓴다. 슬롯 용량은? InnerTime::SIZE=8이면 맞는가?

<details><summary>정답/힌트</summary>

Sequential 슬롯 용량 = 32 / Lane::SIZE = 32/1 = 32. InnerTime::SIZE=8 ≤ 32이라 여유 있게 맞음(time-reducer.md:93-96).

</details>

**Q6.** spot-the-error: 출력 Slice = m![A / 4, X, X] (X는 새 브로드캐스트 축)로 CustomBroadcast를 구성했다. 무엇이 틀렸나?

<details><summary>정답/힌트</summary>

'각 브로드캐스트 축은 출력 Slice에 정확히 한 번' 제약 위반. X가 두 번 등장해 라우팅 bitmap에 정의된 의미가 없음(switch-engine.md:564-568).

</details>

**Q7.** contract_outer에서 OutPacket을 64B로 두면(bf16, OutPacket 32원소) PackSize와 MAC 활용률은? 32B(16원소)로 두면?

<details><summary>정답/힌트</summary>

64B: PackSize = 32×2/32 = 2 → 모든 MAC 사용(100%). 32B: PackSize = 16×2/32 = 1 → 0 곱셈 절반 낭비로 유효 처리량 절반(outer.md:55, 105-107).

</details>

**Q8.** K(축약 축)를 Packet 대신 Time에 둔 batched matmul(bf16)의 MAC 활용률은? 왜 권장되지 않나?

<details><summary>정답/힌트</summary>

1/32. Packet이 1#32로 패딩되어 축약이 Packet Reducer 트리(공간) 대신 cycle별 순차로 진행되어 32 곱셈기 중 1개만 일함. 권장은 K를 Packet에 두기(index.md:88, 123).

</details>

**Q9.** InterTranspose{slice1:2, slice0:16, time0:2}, InTime::SIZE는 무엇으로 나눠떨어져야 하나? slice2는?

<details><summary>정답/힌트</summary>

InTime::SIZE는 slice1×time0 = 2×2 = 4로 나눠떨어져야 함. slice2×slice1×slice0=256 → slice2 = 256/(2×16) = 8(switch-engine.md:240-242, 264).

</details>

**Q10.** to_vrf에 bf16 CollectTensor를 저장하려 한다. 컴파일되나?

<details><summary>정답/힌트</summary>

안 됨. to_vrf는 VeScalar(i32/f32)만 받는다. bf16은 to_trf로는 되지만 VRF는 불가(register-files.md:203, collect.rs:72).

</details>

## 5. 흔한 함정

- Switch는 slice 수를 절대 못 바꾼다. OutSlice::SIZE != InSlice::SIZE면 verify_switch가 'Switch input and output slice sizes must match'로 panic한다. 브로드캐스트는 새 축(X,Y)으로 자리를 채워 SIZE를 유지하는 것이지 늘리는 게 아니다.  
  ↳ 출처 `furiosa-opt-std/src/engine/switch.rs:692-700, switch-engine.md:20`
- SwitchConfig에 Forwarding 변형이 없다. slice 간 교환이 필요 없는데 굳이 switch를 끼우려 하지 말고 FetchTensor에 바로 collect를 호출해야 한다.  
  ↳ 출처 `docs/src/computing-tensors/switch-engine.md:46-47`
- Collect의 single/multi-flit 분기는 원소 수가 아니라 '원소 수 × dtype 바이트'로 결정된다. B=32가 i8이면 32B(single-flit)지만 bf16이면 64B(2 flit)라 collect 매핑이 달라진다. 같은 매핑을 dtype만 바꿔 재사용하면 'Collect time mismatch'.  
  ↳ 출처 `docs/src/computing-tensors/collect-engine.md:40, 108, switch_assertions.rs:233-251`
- Packet Reducer의 OutPacket은 32원소를 넘을 수 없다. Time Reducer 누산기 열이 32개뿐이기 때문. i4(128원소)처럼 패킷이 크면 트리로 줄여도 32 초과분은 dummy 슬라이스로 버려진다 — 이걸 모르고 64원소 결과를 기대하면 안 됨.  
  ↳ 출처 `docs/src/computing-tensors/contraction-engine/packet-reducer.md:50-52`
- PackSize=1(OutPacket 32B)은 컴파일은 되지만 곱셈기 절반이 0과 곱해져 유효 처리량이 절반이 된다. 가능하면 OutPacket을 64B(PackSize=2)로 맞춰 MAC을 다 쓰는 게 핵심 튜닝 포인트.  
  ↳ 출처 `docs/src/computing-tensors/contraction-engine/outer.md:105-107`
- 축약 축(K)을 Time에 두면 bf16에서 MAC 활용률이 1/32로 떨어진다. matmul은 반드시 축약 축을 Packet에 두고 Packet Reducer 트리로 병렬 축약해야 한다. K in Time은 교육용 baseline일 뿐.  
  ↳ 출처 `docs/src/computing-tensors/contraction-engine/index.md:88, 123`
- to_trf의 Lane::SIZE는 {1,2,4,8}만 허용된다. 3이나 16을 쓰면 verify_to_trf가 panic. 또 Lane::SIZE는 Time::SIZE를 나눠야 하고 Time의 바깥 인자가 Lane과 같아야 한다.  
  ↳ 출처 `furiosa-opt-std/src/engine/collect.rs:121-139, contract_outer_assertions.rs:148-208`
- CustomBroadcast에서 Slice→Time으로 옮기는 축은 입력 slice에서의 상대 순서를 지켜야 하고(라우터 버퍼가 1패킷뿐이라 재정렬 불가), 출력 Time의 가장 안쪽에 놓여야 한다. Broadcast01의 time0 같은 우회 수단이 없다.  
  ↳ 출처 `docs/src/computing-tensors/switch-engine.md:578-596`
- 2D conv에서 Stream Adapter 출력 매핑이 TRF Sequencer contraction 매핑과 어긋나면, 하드웨어가 고정 기능 MAC 배열이라 우아한 성능 저하가 아니라 '틀린 결과'가 나온다. shift 설정 실수도 마찬가지로 오답을 낸다.  
  ↳ 출처 `docs/src/computing-tensors/contraction-engine/2d-convolution.md:403, 417-418, 424`
- wide TRF read(64B)는 매 cycle 양 bank를 점유해서, cache miss 때마다 동시 sub-context store를 막는다(이중 버퍼링 효율 저하). narrow read(≤32B)만 bank 교번으로 경합을 회피한다. 큰 ReadSize가 항상 좋은 건 아니다.  
  ↳ 출처 `docs/src/computing-tensors/register-files.md:108-111`
- to_vrf는 i32/f32(VeScalar)만 받는다. bf16/i8을 VRF에 넣으려 하면 컴파일 실패. 그런 타입은 TRF(Contraction)용이다.  
  ↳ 출처 `docs/src/computing-tensors/register-files.md:203, furiosa-opt-std/src/engine/vector/scalar.rs:18-27`
- switch_assertions.rs와 contract_outer_assertions.rs의 invalid_* 함수는 테스트에서 호출되지 않는다(컴파일 타임 검증 panic을 문서화한 의도적 오답 예제). valid_* 만 실제로 실행/검증된다 — invalid 경우를 돌려보려면 직접 복제해 트리거해야 한다.  
  ↳ 출처 `furiosa-opt-examples/tests/switch_assertions_tests.rs (valid_*만 호출), furiosa-opt-examples/src/switch_assertions.rs:336-447`

## 6. 핵심 정리 & 다음

기억할 사실:
- 한 RNGD 칩에는 256개의 slice가 있고, slice들이 cluster로, cluster들이 chip으로 묶인다. 각 slice는 자기만의 파이프라인 인스턴스를 돌리는 공간 차원이다. (`docs/src/computing-tensors/index.md:51-54, switch-engine.md:3`)
- flit(플릿)은 32바이트 고정 크기다. Collect 엔진이 모든 패킷을 32바이트 flit으로 정규화하며, 다운스트림 엔진(Contraction/Vector/Cast/Transpose/Commit)은 전부 flit만 소비한다. (`docs/src/computing-tensors/index.md:10, collect-engine.md:3`)
- TU 스트림은 5차원 [Chip, Cluster, Slice, Time, Packet]을 가진다. Chip/Cluster/Slice는 공간(spatial), Time/Packet은 시간(temporal). 공간 차원을 바꾸는 엔진은 Switch(Slice 이동)와 Vector의 inter-slice reducer(Slice 접기)뿐이다. (`docs/src/computing-tensors/index.md:50-54`)
- Switch 엔진은 256 slice를 양방향 링 네트워크(slice당 라우터 1개, wrap-around)로 엮고, 256/ring_size개의 병렬 sub-ring으로 분할한다. 각 ring 링크 통과는 1 cycle. (`docs/src/computing-tensors/switch-engine.md:323-333, 352`)
- Switch 사이클 추정 = ring_size × Time::SIZE × flits_per_packet, 여기서 flits_per_packet = sizeof(D) × Packet::SIZE / 32. 모든 sub-ring이 병렬이라 ring당 사이클이 곧 칩 전체 지연. (`docs/src/computing-tensors/switch-engine.md:389-397`)
- 정규 Switch 설정의 ring_size = slice1 × slice0. CustomBroadcast는 ring_size를 직접 지정하되 2의 거듭제곱이어야 하고 컴파일러가 유도한 값과 일치해야 한다. (`docs/src/computing-tensors/switch-engine.md:36-40, 599-601`)
- InterTranspose는 slice2×slice1×slice0==256, time1.SIZE==slice1, InTime::SIZE가 slice1×time0으로 나눠떨어질 것을 강제한다. (`docs/src/computing-tensors/switch-engine.md:238-242`)
- TRF는 slice당 8 lanes × 2 banks × 128 rows × 320 bits = 80 KB. 접근당 1/2/4/8 lane이 활성화된다. (`docs/src/computing-tensors/register-files.md:72-73`)

➡️ 다음: [07_computing_engines_2.md](./07_computing_engines_2.md)
