# 05 · 텐서 옮기기 (DMA·Fetch·Commit·메모리 성능)

이 문서는 vISA 커리큘럼 모듈 05입니다. 데이터를 메모리 계층 사이로 옮기는 엔진들(Sequencer, Fetch, Commit, DMA)과 성능의 핵심인 뱅크 충돌(치명적일 수 있음)을 익힙니다.
*선행: 04 텐서 축약 · 예상 시간: 반나절*

## 학습 목표

- [ ] 메모리 계층별 용량·대역폭과 타일링이 필요한 이유를 안다
- [ ] Sequencer 한도(8 entries / 65536 iters)를 안다
- [ ] Fetch 패딩, Commit 옵션(절단/ReLU/valid-count)을 안다
- [ ] 같은 뱅크 64회 충돌이 클러스터 리셋을 부른다는 걸 안다

## 1. 개념

## 0. 이 장의 큰 그림

vISA에서 "계산"은 Tensor Unit이 하지만, 그 전에 데이터를 올바른 자리로 옮겨야 합니다. 이 장은 데이터가 메모리 계층 사이를 어떻게 흐르는지를 다룹니다. 핵심은 세 개의 전용 이동 엔진입니다 (docs/src/moving-tensors/index.md:4):

- Fetch 엔진: DM → Tensor Unit 스트림 (읽어서 파이프라인에 흘려보냄)
- Commit 엔진: Tensor Unit 스트림 → DM (계산 결과를 다시 써넣음)
- DMA 엔진: DM·SPM·HBM 중 임의의 두 계층 사이를 직접 이동 (Tensor Unit 파이프라인을 건드리지 않음)

데이터 흐름을 한 줄로 그리면 이렇습니다 (docs/src/moving-tensors/memory-performance.md:163): `DMA가 HBM에서 DM을 채움 → Fetch가 DM에서 TU로 스트리밍 → Collect가 TRF/VRF에 적재 → Contraction/Vector가 레지스터파일을 읽어 계산 → Commit이 결과를 DM에 기록`.

설계 철학을 먼저 잡으면 이해가 빠릅니다. API는 "프로그래머가 통제하는 것"만 노출합니다. 즉 어느 엔진이 어떤 텐서를 옮기고, 논리 축(axis)이 하드웨어 차원에 어떻게 매핑되는지만 선언하면, 뱅크 스케줄링·스트라이드 계산·정렬 같은 저수준은 컴파일러가 처리합니다 (docs/src/moving-tensors/index.md:33).

그리고 세 엔진은 모두 같은 기반 부품을 씁니다. 바로 Sequencer입니다. Sequencer를 먼저 이해하면 Fetch/Commit/DMA가 전부 같은 원리의 변주임을 알게 됩니다.

---

## 1. 메모리 계층과 대역폭

칩당 피크 대역폭은 다음과 같습니다 (docs/src/moving-tensors/memory-performance.md:9): DM 2 TB/s, SPM 2 TB/s, HBM 1.5 TB/s. DM은 칩당 256MB SRAM(작업 메모리), HBM은 칩당 48GB DRAM(대용량 저장)입니다.

여기에 더해 계산 엔진에 직접 붙는 레지스터파일이 두 개 있습니다 (docs/src/moving-tensors/memory-performance.md:158): TRF는 `.to_trf()`로 채우고 Contraction 엔진이 매 사이클 읽으며, VRF는 `.to_vrf()`로 채우고 Vector 엔진이 매 사이클 읽습니다. 이 둘은 전용 이동 엔진이 아니라 Tensor Unit 프리미티브가 채우므로 "Computing Tensors" 장에서 다룹니다 (docs/src/moving-tensors/index.md:9).

DM과 SPM은 둘 다 온칩 SRAM이고 문서화된 지연/용량 차이는 없습니다. 차이는 용도(그리고 그에 따른 컴파일러 할당 정책)뿐입니다 (docs/src/moving-tensors/memory-performance.md:142). DM은 파이프라인을 흐르는 텐서의 일반 작업 메모리이고, SPM은 컴파일러가 명시적으로 고른 "작고 자주 재사용되는 값"(스칼라 상수, 활성화함수 룩업테이블, DMN별 소규모 작업셋)을 두는 곳입니다. DM에서 매번 다시 읽어오기 싫은 데이터를 옆에 staging해 두는 용도라고 보면 됩니다. 각 DMN이 전용 SPM을 가져서 DMN 간 경합이 없고 128 B/cycle을 냅니다.

---

## 2. Sequencer — 세 엔진의 공통 심장

Sequencer는 "메모리 버퍼를 패킷 스트림으로 읽고, 패킷 스트림을 다시 메모리에 쓰는" 부품입니다 (docs/src/moving-tensors/sequencer.md:3). Fetch와 Commit은 각각 시퀀서 하나로 DM을 주소 지정하고, DMA는 읽기 시퀀서와 쓰기 시퀀서를 사슬처럼 묶어 중간 버퍼 없이 옮깁니다.

### 2.1 BufTensor ↔ StreamTensor (교육용 모형)

문서는 설명을 위해 `BufTensor`(메모리에 있는 버퍼)와 `StreamTensor`(흐르는 중인 텐서)라는 가상 타입을 씁니다 (docs/src/moving-tensors/sequencer.md:9, furiosa-opt-std/src/tensor/pseudo.rs:23). 실제 엔진은 `DmTensor`, `HbmTensor`, `TuTensor` 등을 쓰지만 모두 같은 buffer→stream 모양으로 환원됩니다.

StreamTensor는 두 가지 매핑으로 나뉩니다 (docs/src/moving-tensors/sequencer.md:18): `Time`은 시간 매핑(시간에 따른 반복), `Packet`은 공간 매핑(한 패킷 안의 내용). 라이프타임 `'l`이 스트림을 원본 버퍼에 묶어, 데이터보다 스트림이 더 오래 살 수 없게 합니다.

`read`가 BufTensor를 StreamTensor로, `write`가 그 반대로 변환하며 값은 보존합니다 (furiosa-opt-std/src/tensor/pseudo.rs:42). 중요한 비대칭이 하나 있습니다. read는 브로드캐스트를 허용(Fetch/Switch/DMA-read와 일치)하지만 write는 브로드캐스트를 거부(Commit과 일치)합니다 — 각 버퍼 슬롯은 정확히 하나의 소스 위치를 가져야 하기 때문입니다 (furiosa-opt-std/src/tensor/pseudo.rs:51).

같은 BufTensor라도 유효한 (Time, Packet) 조합은 많고, 각각 다른 StreamTensor를 만듭니다. 그중 Packet이 클수록 대역폭 효율이 좋습니다 (docs/src/moving-tensors/sequencer.md:33).

### 2.2 Config — 시퀀서가 실제로 실행하는 것

각 시퀀서 호출은 입력/출력 매핑으로부터 중첩 루프 설정으로 컴파일됩니다 (docs/src/moving-tensors/sequencer.md:119). 형식은 `[size_0 : stride_0, size_1 : stride_1, ...] : packet_size`이고 subscript 0이 가장 바깥 루프입니다. 각 `Entry`는 `size`(반복 횟수)와 `stride`(반복마다 건너뛸 메모리 거리, 원소 단위)를 가집니다.

예를 들어 `m![N, C, H, W]` → `m![W, H, C, N]` (N=4, C=3, H=8, W=8)은 다음으로 컴파일됩니다 (docs/src/moving-tensors/sequencer.md:199): `[8:1, 8:8, 3:64, 4:192] : 1`. 즉 W는 stride 1로 8번, H는 stride 8로 8번, 식으로 도는 4중 루프이며 주소는 `addr = 1*w + 8*h + 64*c + 192*n`입니다. Packet=m![1]이라 매 반복 1번 접근합니다.

### 2.3 access_size — 패킷당 몇 번 접근하나

`access_size = gcd(Packet::SIZE, contiguous_run)` 입니다 (docs/src/moving-tensors/sequencer.md:143). 여기서 `contiguous_run`은 Config 엔트리들 중 물리적으로 연속한 가장 안쪽 구간의 원소 수입니다. 안쪽에서 바깥쪽으로 인접한 두 엔트리 `(n_outer:s_outer)`, `(n_inner:s_inner)`가 `s_outer == n_inner * s_inner`이면 연속으로 판정해 곱해 나가고, 처음으로 불연속인 쌍에서 멈춥니다 (docs/src/moving-tensors/sequencer.md:154).

access_size가 클수록 패킷당 접근 횟수가 적어 좋습니다. 대부분의 경우 DM에서 패킷 레이아웃이 완전 연속이라 `access_size == Packet::SIZE`이지만, 그렇지 않은 경우(Non-Contiguous Packets)에는 `access_size < Packet::SIZE`가 되어 하드웨어가 연속 부분블록마다 한 번씩 접근합니다 (docs/src/moving-tensors/sequencer.md:429).

대표 예: `m![A, B # 16]` 버퍼에 32원소 패킷 `m![A, B]`를 쓰면 B행마다 16칸으로 패딩되어 A의 stride가 8이 아닌 16이 됩니다. 패킷 span이 불연속이라 1번이 아니라 4번 접근합니다 (`Packet::SIZE=32, contiguous_run=8, access_size=8`). 또 다른 예: `m![N, C, H, W]`를 Packet=m![N, H, W]로 읽으면 C를 건너뛰어 N의 소스 stride(96)가 H×W(32)와 다르므로 `contiguous_run=32, access_size=32`, 패킷당 4번 접근합니다 (docs/src/moving-tensors/sequencer.md:486).

### 2.4 변환 패턴 (term-by-term 규칙)

컴파일러는 합쳐진 매핑(Time과 Packet을 이어붙인 것)을 항(term)별로 처리하며, 각 항마다 entry size = 항의 크기, stride = 그 항이 현재 Buf에서 차지하는 부피로 정합니다. 처리한 항은 Buf에서 소거됩니다 (docs/src/moving-tensors/sequencer.md:268). 패턴은 다섯 가지입니다.

- 축 전치(Transpose): 스트림이 버퍼와 다른 순서로 축을 방문. 컴파일러가 필요한 stride를 계산 (docs/src/moving-tensors/sequencer.md:253).
- 축 분할(Split/Tiling): `A % 2`, `A / 2`처럼 한 축을 여러 엔트리로 쪼갬. 캐시 효율이나 TU 버퍼 크기 맞춤용 (docs/src/moving-tensors/sequencer.md:284).
- 축 슬라이스(Slice): `A % 4 = 3`처럼 4가 아니라 3번만 반복해 부분 영역만 읽음. 인덱싱 뷰가 부분집합을 고를 때 발생 (docs/src/moving-tensors/sequencer.md:314).
- 브로드캐스트(Broadcast): Time/Packet에는 있는데 Buf에는 없는 축은 stride `: 0` 엔트리가 되어 같은 주소를 반복 방문. P가 브로드캐스트면 공간 브로드캐스트(패킷 안에서 같은 원소 복제), T면 시간 브로드캐스트(시간축으로 같은 데이터 반복) (docs/src/moving-tensors/sequencer.md:344). 부분 축 fragment(`N / 512` 등)가 Buf에 없을 때도 똑같이 적용됩니다.
- 엔트리 병합(Merging): 하드웨어는 엔트리 최대 8개만 허용하므로, 변환이 9개 이상을 만들면 컴파일러가 인접한 연속 엔트리를 병합합니다. `(n1:s1)`과 `(n2:s2)`는 `s1 == n2 * s2`일 때 `(n1*n2 : s2)`로 합쳐집니다 (docs/src/moving-tensors/sequencer.md:380). 병합은 time/packet 경계도 넘을 수 있습니다(예: `W/8%2 (2:8)`와 `W%8 (8:1)`이 `W%16 (16:1)`로).

### 2.5 제약 (어기면 컴파일 에러)

RNGD에서 다음 하드웨어 한계를 넘으면 컴파일 에러입니다 (docs/src/moving-tensors/sequencer.md:493):
- 엔트리 한계: 최대 8개 (그래서 병합)
- 반복 한계: 엔트리당 `size <= 65,536`
- 패킷 크기: 1, 2, 4, 8, 16, 32 바이트 중 하나
- 패킷 페치: 가장 안쪽 엔트리 `n:s`는 둘 중 하나를 만족해야 함 — 연속 접근(`(s==0 || s==1) && n % packet_size == 0`) 또는 이산 접근(`packet_size == 1`)

병합 실패나 한계 초과 시에는 매핑을 다시 설계하거나 여러 시퀀서 호출로 쪼개야 합니다.

또 하나 미묘한 제약: 호환 가능한 축 분해(Compatible Axis Decomposition)입니다. Buf와 스트림 양쪽에 등장하는 축은 같은 분해를 써야 합니다. 예를 들어 Buf가 `m![A % 5, A / 5]`(A=15를 5×3)인데 스트림이 `m![A % 3, A / 3]`(3×5)이면, `gcd(5,3)=1`이라 어느 쪽도 다른 쪽을 세분하지 못해 거부됩니다. 총 원소 수가 같아도 안 됩니다 (docs/src/moving-tensors/sequencer.md:507).

### 2.6 간접 접근 (Indirect Access)

위의 모든 엔트리는 고정 stride를 씁니다(반복 간 오프셋 일정). `IndirectLoop`은 반복마다 가변 오프셋을 허용해, 데이터 의존적 접근 패턴(gather)을 가능케 합니다. 표준 `(limit, stride)`가 `(limit, [offset0, offset1, ...])`이 되어 임베딩 룩업처럼 런타임에 인덱스가 정해지는 연산을 지원합니다 (docs/src/moving-tensors/sequencer.md:530).

---

## 3. Fetch 엔진 — DM에서 파이프라인으로

Fetch 엔진은 두 단계입니다 (docs/src/moving-tensors/fetch-engine.md:3): Fetch Sequencer(슬라이스별 시퀀서로 DM을 읽어 패킷 스트림 생성)와 Fetch Adapter(선택적 원소별 변환).

### 3.1 인터페이스와 핵심 제약

`BeginTensor`는 DM에 있는, TU 파이프라인 입구의 텐서입니다. Time은 m![1](파이프라인 시작 전 시간 반복 없음), Packet은 DM에서의 원소 레이아웃입니다. `.fetch()`가 이를 `FetchTensor` 패킷 스트림으로 바꿉니다.

실제 시그니처는 다음과 같습니다 (furiosa-opt-std/src/engine/fetch.rs:35):
```
pub fn fetch<D2: Scalar, Time2: M, Packet2: M>(self) -> FetchTensor<...>
    where D: FetchCast<D2>
```
출력 타입 파라미터로 시퀀서와 어댑터를 동시에 설정합니다: `D2`는 타입 캐스팅(i8→i32 등), `Time2`는 출력 스트림의 시간 스텝 수, `Packet2`는 패킷 안 원소 레이아웃입니다. 컴파일러가 나머지 하드웨어 설정을 출력 타입에서 유도합니다 (docs/src/moving-tensors/fetch-engine.md:23).

`.fetch()`는 Chip, Cluster, Slice 차원을 입력에서 그대로 보존합니다. 각 슬라이스가 자기 DM 파티션을 독립적으로 읽기 때문입니다 (docs/src/moving-tensors/fetch-engine.md:20).

여기서 절대 잊으면 안 되는 하드웨어 검증이 있습니다 (furiosa-opt-std/src/engine/fetch.rs:49, `verify_fetch`): Cluster::SIZE는 반드시 2, Slice::SIZE는 반드시 256, 출력 패킷은 8바이트(`FETCH_ALIGN_BYTES`) 정렬이어야 합니다. 이 셋 중 하나라도 어기면 런타임 assert로 panic합니다. 4칩 RNGD는 4 chip × 2 cluster × 256 slice = 2,048 슬라이스가 각자 같은 시퀀서 패턴을 자기 A×B 부분텐서에 돌립니다 (docs/src/moving-tensors/fetch-engine.md:51).

### 3.2 Multi-Read와 read_size

패킷을 준비하려면 여러 하드웨어 read가 필요할 수 있습니다. 패킷 축이 DM에서 연속이 아닐 수 있고, 하드웨어는 한 번에 최대 32바이트만 읽기 때문입니다 (docs/src/moving-tensors/fetch-engine.md:60). main-context에서 `read_size`는 access_size의 약수 중 `D[read_size]`가 1/2/4/8/16/32 바이트가 되는 가장 큰 값입니다. sub-context에서는 8바이트 고정입니다. `Packet::SIZE > read_size`이면 multi-read가 일어나고, 총 사이클 수는 `Time::SIZE * (Packet::SIZE / read_size)`입니다.

같은 i4 `m![N,C,H,W]`(N=4,C=3,H=4,W=8)를 Packet2만 바꿔 페치하면 사이클이 크게 달라집니다 (docs/src/moving-tensors/fetch-engine.md:75): Packet=m![W]→access_size 8, 48사이클; Packet=m![H%2,W]→access_size 16, 24사이클; Packet=m![H,W]→access_size 32, 12사이클; Packet=m![C,H,W]→access_size 96이지만 read_size는 32(16바이트) 상한이라 패킷당 3 read, 그래도 12사이클. 패킷을 키우면 보통 빨라진다는 직관을 기억하세요.

### 3.3 Interleaving

매핑이 같은 두 텐서를 한 시퀀서 연산으로 합치는 기법입니다. 명시적 Time 축이 두 텐서 간 교대를 인코딩해, 첫 시간 반복은 lhs, 둘째는 rhs, 식으로 번갈아 페치합니다. 한 페치에 최대 두 텐서까지입니다 (docs/src/moving-tensors/fetch-engine.md:108). main-context에서 `ctx.main.begin_interleaved::<I,...>(lhs.view(), rhs.view()).fetch()`로 만듭니다. `input1 + input2` 같은 Vector 연산에 유용합니다.

### 3.4 최적화 3요소

Fetch 처리량은 세 가지로 결정됩니다 (docs/src/moving-tensors/fetch-engine.md:138):
- 입력 대역폭: read_size는 DM 축 연속성과 패킷 크기에 의해 제한됩니다. 비인접 축은 access_size를, 그래서 read_size를 떨어뜨립니다. 패킷을 더 큰 2의 거듭제곱으로 패딩하면 올라갑니다.
- 출력 대역폭: 다운스트림 Collect 엔진이 패킷을 32바이트 flit으로 변환하므로, 32바이트에 안 맞는 패킷은 대역폭을 낭비합니다. 20바이트 패킷은 flit 하나에 12바이트 제로패딩→37.5% 낭비, 40바이트 패킷은 flit 두 개(64바이트)에 걸쳐 24바이트 제로패딩→역시 37.5% 낭비.
- 공간 병렬성: 페치를 슬라이스 전반에 분산하면 처리량 최대화.
- 그리고 경고: 같은 뱅크를 연속 64회 이상 때리는 접근은 우선순위 낮은 Commit/DMA 엔진을 굶겨 치명적 NoC 타임아웃을 부릅니다 (docs/src/moving-tensors/fetch-engine.md:145). 7절에서 자세히.

패킷 패딩 예시는 인상적입니다 (docs/src/moving-tensors/fetch-engine.md:153): 같은 30바이트 텐서를 패킷을 2→16→32바이트로 키우면 15→3→1 사이클로 줄어듭니다. 패딩이 실제 데이터 너머를 읽지만, 패딩 값은 계산에 안 쓰이므로 안전합니다.

### 3.5 Fetch Adapter — 네 단계 변환

어댑터는 시퀀서 패킷에 원소별 변환을 가합니다. main-context는 네 단계 전부, sub-context는 zero-point만 지원합니다 (docs/src/moving-tensors/fetch-engine.md:191).

(1) 마스킹: 패딩된 원소를 중립값으로 덮어 다운스트림에 영향 없게 합니다. 패딩이 필요한 이유는 TU 내부 데이터 경로가 고정폭 단위(32바이트 flit = 8원소 × 32비트)로 동작하므로, 크기가 flit 폭의 배수가 아닌 축은 올림(예: 63→64)됩니다. 마스킹 없으면 패딩 슬롯이 임의값이라 sum/max 같은 리덕션이 망가집니다 (docs/src/moving-tensors/fetch-engine.md:196). 설정은 세 파라미터로 합니다: `last_dim`(마스킹할 차원 인덱스, 0이 가장 안쪽), `left_pad`(왼쪽에서 0으로 만들 개수), `last_dim_rightmost_valid_count[0]`(오른쪽에서 0으로 만들 개수; 4비트 타입 0-255, f32 0-31로 제한 — 최종 패킷이 256바이트 안에 있어야 하므로). 패딩 모양에 따라 세 케이스가 있습니다 (그림 fetch-engine-padding-1/2/3.png):
  - 케이스 1: 가장 안쪽 축에 연속된 왼쪽 패딩 하나 + 오른쪽 패딩 하나. 예: A=32,B=90, Element=m![A,B#96], lpad=2, rightmost_valid=4 → `(#2 + B + #4)`의 앞 2·뒤 4 값이 0 (docs/src/moving-tensors/fetch-engine.md:229).
  - 케이스 2: 케이스 1과 같은 마스킹이지만 패딩 영역이 연속이 아니라 쪼개진 경우. Time 순서만 `m![B'/32, A]`로 바뀜 (docs/src/moving-tensors/fetch-engine.md:248).
  - 케이스 3: 케이스 1·2의 오른쪽 패딩 한계(255×4비트)를 풀어, 엔트리 인덱스마다 자기 `last_dim_rightmost_valid_count[i]`를 줌. 축 크기가 8 이하일 때 `[0..8]` 8개까지 지원. 예: A=32,B=97,f32, valid_count=[16,16,16,16,16,16,1,0] → 97개 유효, 31개 무효 마스킹 (docs/src/moving-tensors/fetch-engine.md:264).

(2) 테이블 인덱싱: 각 값을 미리 구성한 룩업테이블의 인덱스로 보고 테이블 엔트리를 대신 출력합니다. Sigmoid·GeLU 같은 비선형 활성화나 MXFP4 같은 커스텀 인코딩 변환에 씁니다. `input.fetch_with_table(table)`로 호출 (docs/src/moving-tensors/fetch-engine.md:281).

(3) 타입 캐스팅: DM에서 스트리밍하며 타입을 변환합니다(저정밀 저장 + 고정밀 계산). 1~2 사이클 지연. RNGD 지원: i4→i5/i32, i8→i9/i32, i16→i32, f8e4m3/f8e5m2/bf16/f16→f32, f32→bf16. RNGD-S 추가: i4→i9, i16→i9, f8e4m3/f8e5m2→bf16 (docs/src/moving-tensors/fetch-engine.md:316). 중요한 추가 제약: 캐스트 출력이 페치당 단일 32바이트 flit에 들어가야 합니다. 예를 들어 i8→i32에서 read_size=8(8바이트)은 8×4=32B로 유효하지만, read_size=16(16바이트)은 16×4=64B라 무효입니다 (docs/src/moving-tensors/fetch-engine.md:354).

(4) Zero-point 빼기: 양자화 정수를 계산 타입으로 바꿀 때 zero-point 오프셋을 동시에 뺍니다(비대칭 양자화). zero-point는 입력 타입 범위 안이어야 합니다(i8이면 -128..=127). `fetch_with_zero_point(zp)` 또는 interleaving과 함께 `fetch_with_zero_points([zp1, zp2])`로 텐서마다 다른 zero-point를 뺄 수 있습니다 (docs/src/moving-tensors/fetch-engine.md:363).

---

## 4. Commit 엔진 — 파이프라인에서 DM으로

Commit은 Fetch의 거울상이지만 역방향입니다 (docs/src/moving-tensors/commit-engine.md:15). 두 단계: Commit Adapter(원소별 연산)와 Commit Sequencer(슬라이스별 DM 쓰기).

### 4.1 인터페이스와 검증

`TuTensor`는 파이프라인 끝의 Chip/Cluster/Slice/Time/Packet 텐서입니다. `.commit(address)`가 이를 DM의 `DmTensor`로 바꿉니다. 실제 시그니처 (furiosa-opt-std/src/engine/commit.rs:27):
```
pub fn commit<Element: M>(self, address: Address) -> DmTensor<...>
```
출력 `Element` 매핑이 Time과 Packet을 대체하며 스트림이 DM에 어떻게 놓일지 정의합니다. 핵심 기능: Element는 입력 스트림 대비 Time 축을 재배열할 수 있어, 커밋하면서 전치(transpose)를 수행합니다 (docs/src/moving-tensors/commit-engine.md:24).

검증 규칙은 `verify_commit`에 있습니다 (furiosa-opt-std/src/engine/commit.rs:47): (1) 입력 패킷은 정확히 한 flit(32바이트)여야 함, (2) 출력 패킷은 8/16/24/32 바이트 중 하나(`COMMIT_OUT_PACKET_SIZES`), (3) 출력 Time은 입력 Time의 유효한 전치여야 함, (4) 출력 패킷은 입력 Packet의 truncation(슬라이스)일 수 있음. `commit_view(dst)`는 새 DmTensor 대신 기존 가변 뷰에 씁니다.

### 4.2 Commit Adapter — main/sub로 갈리는 기능

| 연산 | Main | Sub |
|------|------|-----|
| Truncating | 가능 | 가능 |
| Type Casting | 가능 | 불가 |
| Valid Count Packing | 불가 | 가능 |
| Generate Mode | 불가 | 가능 |
(docs/src/moving-tensors/commit-engine.md:53)

(1) Truncating: TU 파이프라인 패킷은 항상 32바이트 flit이지만, flit이 용량보다 적은 유효 원소를 담고 뒤를 패딩으로 채울 수 있습니다. flit을 통째로 쓰면 유효 영역 너머 DM 바이트를 패딩으로 덮어버립니다. Truncating은 각 flit의 앞쪽 `valid_size` 원소만 쓰고 뒤 패딩을 버립니다. `valid_size`는 컴파일러가 출력 매핑에서 유도하며, `D[valid_size]`는 8/16/24/32 바이트(32는 truncation 없음)여야 합니다. 거의 0 지연입니다 (docs/src/moving-tensors/commit-engine.md:61).

(2) Type Casting: f32를 DM에 쓰기 직전 bf16으로 변환합니다(선택적으로 ReLU 적용). 보통 타입 변환은 Cast 엔진이 하지만, Cast 엔진은 Vector 엔진 위에 있어 변환 중 Vector 엔진을 점유합니다. main-context contraction을 sub-context Vector 작업과 병렬로 돌리고 싶을 때, 변환을 Commit 엔진으로 우회시키면 Vector 엔진이 비어 sub-context가 병렬 실행됩니다 (docs/src/moving-tensors/commit-engine.md:113). 표준 변환은 `commit_cast(0)`, ReLU 동반은 `commit_cast_relu(0)`(음수를 0으로 클램프, 예: [-5.0,-0.1,0.0,3.7]→[0.0,0.0,0.0,3.7]) (docs/src/moving-tensors/commit-engine.md:140).

(3) Valid Count Packing: 런타임 카운트에 따라 유효 원소만 골라 커밋해, 패딩/무효 데이터를 출력 버퍼에서 제외합니다. 가변 길이 결과(필터링, 동적 시퀀스 길이)에서 의미 있는 원소만 DM에 써서 메모리 낭비를 막습니다 (docs/src/moving-tensors/commit-engine.md:160).

(4) Generate Mode: `ITOS`(immediate-to-SRAM) 명령으로 단일 32비트 값을 지정 주소에 써, TU 실행 파이프라인을 우회합니다 (docs/src/moving-tensors/commit-engine.md:167).

### 4.3 Commit Sequencer — Multi-Write와 정렬

쓰기도 패킷 축이 DM에서 비연속이면 여러 하드웨어 write가 필요합니다. 쓰기당 원소 수는 `write_size = gcd(valid_size, access_size)`입니다 (docs/src/moving-tensors/commit-engine.md:183). sub-context에서는 `D[write_size]`가 8바이트 고정입니다. 총 사이클은 `Time::SIZE * (valid_size / write_size)`이고, 이 나눗셈은 항상 정확합니다. main-context에서는 `valid_size == write_size`라 패킷당 1 사이클; sub-context에서는 write_size=8 고정에 valid_size가 8/16/24/32라 나누면 1/2/3/4입니다.

대표 패턴 (docs/src/moving-tensors/commit-engine.md:191): 전치 없는 경우 access_size=64, valid_size=8, write_size=gcd(64,8)=8, 패킷당 1 write. 패딩 청킹 `m![K], m![M, W]` → `m![K, M, W # 16]`은 access_size=8, valid_size=32, write_size=gcd(8,32)=8이라 32바이트 패킷이 M축 따라 4×8바이트 write로 쪼개집니다(offset 0/16/32/48) (docs/src/moving-tensors/commit-engine.md:245).

제약 (docs/src/moving-tensors/commit-engine.md:176): Chip/Cluster/Slice가 하드웨어와 일치해야 하고, 모든 시퀀서 stride는 8바이트의 배수, `D[valid_size]`는 8/16/24/32 바이트여야 합니다.

Slice Bitmap: 한 클러스터(슬라이스 256개) 전체를 덮는 256비트 마스크로, 어느 슬라이스가 커밋 데이터를 받을지 게이팅합니다. `...01`은 슬라이스 0만, `...10`은 0 빼고 전부 (docs/src/moving-tensors/commit-engine.md:252).

최적화 (docs/src/moving-tensors/commit-engine.md:258): 순차 주소로 쓰면 병렬 뱅크 접근 가능(DMN당 128 B/cycle, DMN 인터리빙 시 256 B/cycle); 같은 뱅크 64회+ 연속은 뱅크 기아 유발; 정렬 쓰기는 불변식(주소와 쓰기 단위 둘 다 항상 8바이트 정렬)이라 부분 뱅크 쓰기는 절대 일어나지 않습니다.

---

## 5. DMA 엔진 — 계층 간 직접 이동

DMA는 Tensor Unit을 거치지 않고 계층 사이를 직접 옮깁니다. 매 전송은 읽기 시퀀서(소스에서 읽기)와 쓰기 시퀀서(목적지에 쓰기, 레이아웃 변환 가능)를 짝지웁니다 (docs/src/moving-tensors/dma-engine.md:3). DMA 전송은 mathematical tensor move입니다 — 레이아웃이 달라도 출력은 입력과 같은 수학적 텐서를 담습니다.

### 5.1 두 종류의 컨텍스트와 호출법

- `Context::tdma`: Tensor DMA, 온칩 전송용 (HBM↔HBM, HBM↔DM, DM↔DM)
- `Context::pdma`: PCIe DMA, host↔HBM용 (docs/src/moving-tensors/dma-engine.md:18)

소스 텐서에 `.to_dm()`, `.to_hbm()`, `.to_host()` 등을 호출하며 DmaContext를 넘깁니다. 실제 시그니처 (furiosa-opt-std/src/tensor/memory.rs:423):
```
pub fn to_dm<Cluster: M, Slice: M, Element2: M>(&self, dma: &mut DmaContext<{Dma::Tensor}>, address: Address) -> DmTensor<...>
```
HBM→HBM 전치 예시 (docs/src/moving-tensors/dma-engine.md:28): `[A,B,C]`→`[C,A,B]`를 두 번의 HBM-to-HBM 전송으로 합니다. 첫 단계 주소 0, 둘째 단계 주소 0x1000. HBM→DM은 목적지 DM 타입에 Cluster와 Slice 축을 더해 하드웨어 파티션에 분산합니다. 예: 2,048원소를 슬라이스당 8원소(Element=m![A%8]), 슬라이스 256개(Slice=m![A/8]), 2 클러스터로 (docs/src/moving-tensors/dma-engine.md:50). 주소는 두 번째 인자(예: `0x3000`, `0x2000`)로 명시합니다.

### 5.2 정적 구조

칩당 DMA 엔진은 8개(DMN 쌍마다 하나)로, 최대 8개 독립 전송을 병렬 실행합니다 (docs/src/moving-tensors/dma-engine.md:67). 각 엔진은 읽기/쓰기 시퀀서를 lockstep으로 돌립니다. 둘은 같은 루프 카운트를 공유하지만 stride와 base 주소가 달라, 같은 논리 원소가 소스와 목적지 메모리에 다르게 나타나도록 합니다 (docs/src/moving-tensors/dma-engine.md:80). 기본적으로 컴파일러는 소스 DM의 로컬 DMA 엔진을 고릅니다(로컬 DMN 접근이 더 빠름).

시퀀서 표현 (docs/src/moving-tensors/dma-engine.md:93): `DmaSequencer`는 엔트리들, `stride0`(1..=4096, 가장 안쪽 루프 stride = 반복당 패킷 바이트 수), source_base/dest_base를 가집니다. 각 `DmaEntry`는 공유 `size` 하나에 별도의 `source_stride`/`dest_stride`를 가집니다. `Media`는 `Hbm(chip)`, `Dm(dmn)`, `Spm(dmn)`입니다.

### 5.3 컴파일러 유도와 동적 동작

소스/목적지 매핑과 스트림 모양으로부터 읽기 시퀀서(In을 스트림에 투영), 쓰기 시퀀서(Out을 투영), 통합 시퀀서(둘을 병합해 엔트리마다 읽기·쓰기 stride 짝)를 만들고, 연속 read/write 부피에서 stride0을 추론합니다 — 읽기·쓰기 둘 다 256 연속 바이트면 최적 stride0=256 (docs/src/moving-tensors/dma-engine.md:168). 각 루프는 row-major로 카운터를 올리며 짝 stride에서 read/write 주소를 유도합니다. stride0=256이면 반복당 256바이트를 읽고 씁니다 (docs/src/moving-tensors/dma-engine.md:143).

### 5.4 Aggregate (균등/비균등)

텐서 모양이 DMN에 고르게 나뉘면 모든 엔진이 homogeneous aggregate를 돕니다(같은 스트림 환경, base 주소만 다름). 안 나뉘면 heterogeneous aggregate로 폴백해 DMN마다 자기 스트림 환경을 갖고, 경계 DMN은 유효 영역 너머 쓰기를 피하려 작업을 여러 명령으로 쪼갭니다 (docs/src/moving-tensors/dma-engine.md:196). 각 명령은 자기 startup 지연을 내므로, 균등 분할되는 모양을 선호해야 합니다. 불변식 두 개: 참여 엔진은 같은 소스/목적지 media를 써야 하고, 전송 전체를 하나의 입력·출력 매핑이 지배해야 합니다.

### 5.5 제약 (정렬과 패킷 크기)

주소 정렬 (docs/src/moving-tensors/dma-engine.md:217):

| 계층 | Read | Write |
|------|------|-------|
| HBM | 1 byte | 1 byte |
| DM (SRAM) | 1 byte | 8 bytes |

HBM↔DM 전송은 위 표와 무관하게 read 주소·write 주소·패킷 크기 모두 8바이트 정렬이 추가로 필요합니다. DM의 비대칭 규칙은 SRAM 하드웨어 비대칭을 반영합니다 — read 포트는 byte-select로 임의 바이트 범위를 뽑지만, write 포트는 8바이트 뱅크폭 단위로만 동작합니다. 정렬 안 된 DM write는 Read-Modify-Write를 일으켜 쓰기 시간이 3배가 되고 해당 뱅크의 다른 연산을 막습니다. (그래서 fetch_commit 예제 주석이 "Element의 가장 안쪽 축이 소스의 가장 안쪽(B, stride 1)과 맞아야 DMA tail이 min_align=8을 만족"이라고 못 박습니다 — furiosa-opt-examples/src/fetch_commit.rs:15.) 이 검증은 `assert_dma_layout`이 합니다(furiosa-opt-std/src/tensor/memory.rs).

패킷 크기: 최대 4,096 바이트. AXI 프로토콜이 트랜잭션을 256 beats × 16바이트로 제한하기 때문입니다 (docs/src/moving-tensors/dma-engine.md:231).

### 5.6 최적화 3요소

(docs/src/moving-tensors/dma-engine.md:233) 메모리 대역폭, 채널·DMN 인터리빙, startup 지연 + 패킷 분할입니다. 계층별 피크는 HBM 1.5 TB/s(칩당, 32채널×48GB/s), DM 256 B/cycle(클러스터당, DMN 인터리빙; DMN당 128), SPM 128 B/cycle(클러스터당, 같은 칩만), PCIe 30 B/cycle. 각 DMA 엔진은 자체로 최대 256 B/cycle. HBM 대역폭은 모든 엔진이 공유하므로 aggregate가 HBM을 포화시키면 1.5 TB/s가 상한입니다 (docs/src/moving-tensors/dma-engine.md:248).

같은 클러스터 DM↔DM 전송은 read와 write가 같은 DM 뱅크 접근을 두고 경합해 직렬화됩니다. HBM↔DM 같은 교차 계층 전송은 read/write 단계가 파이프라인됩니다 (docs/src/moving-tensors/dma-engine.md:251). 참고로 SRAM↔SRAM 전송은 Fetch/Commit이 DMA보다 효율적일 때가 많습니다(DMA가 SRAM 슬라이스 대역폭을 덜 쓰기 때문). 다만 실제로는 HBM 대역폭이 병목이라 HBM↔DM에서는 이 차이가 덜 중요합니다 (docs/src/moving-tensors/dma-engine.md:71).

채널·DMN 인터리빙 (docs/src/moving-tensors/dma-engine.md:259): HBM 채널 선택은 주소 비트 9~28, 스택 비트는 비트 8. 모든 32채널에 퍼뜨리려면 이 비트들을 다 토글해야 합니다. 스택 비트(8)를 놓치면 요청이 32채널 중 16개로만 가서 유효 대역폭이 절반이 됩니다. 같은 HBM 뱅크를 반복 때리면(연속 접근마다 row 주소 비트 21+ 토글) 접근당 약 40사이클의 row-conflict 페널티로 대역폭이 한 자릿수 배 떨어집니다. DM 대역폭은 두 DMN(각 128 B/cycle)을 번갈아야 하며, 단일 DMN 패턴은 DM 대역폭을 절반으로 만듭니다.

startup과 패킷 분할 (docs/src/moving-tensors/dma-engine.md:270): 각 DMA 명령은 약 500사이클 고정 startup 지연을 냅니다. 여러 전송을 적은 명령으로 합치면 이 비용을 분할상환합니다. 한 명령 안에서 하드웨어는 패킷을 256바이트 단위로 쪼개므로 n바이트 패킷은 `ceil(n/256)` AXI 요청이 됩니다. stride0이 256정렬이면 사이클은 `ceil(stride0/256)`; 정렬 안 되면 HBM write는 부분 256바이트 블록에 RMW 페널티를 추가로 냅니다(HBM read는 ceil 오버헤드만, DM은 거의 영향 없음).

### 5.7 대표 예제로 보는 직관

- 예제 2 (HBM→DM 풀 대역폭): read/write가 채널과 두 DMN 전반에 인터리빙되어 파이프라인. 총 ≈ max(65,536 read, 65,536 write) + 500 ≈ 66,036 사이클 (docs/src/moving-tensors/dma-engine.md:327).
- 예제 4 (HBM 뱅크 충돌 병리): A%32, A/32 stride가 HBM 비트 21, 26(뱅크 내 row 선택)을 토글해 거의 매 요청마다 row를 닫고 엽니다. 잘 튜닝된 경우 대비 약 10배(≈655,860 사이클) (docs/src/moving-tensors/dma-engine.md:400).
- 예제 5 (스택 비트 누락): C stride 512가 비트 8을 절대 토글 안 해 8 DMN이 32채널 중 16개에 몰림 → 유효 대역폭 절반 (docs/src/moving-tensors/dma-engine.md:443).
- 예제 6 (비균등 분할): A=15가 4 DMN에 안 나뉘어(15=3·4+3) DMN 3만 작업을 두 명령으로 쪼개 각각 startup 500을 냄 → DMN 3이 병목(≈1,192 사이클) (docs/src/moving-tensors/dma-engine.md:482).
- 팁: C는 가능하면 256의 배수로. `C=256n+r`(0<r<256)이면 데이터 양은 거의 안 변해도 사이클이 n+1배로 늡니다 (docs/src/moving-tensors/dma-engine.md:396).

### 5.8 Shuffle / Scatter-Gather / PCIe

Shuffle (docs/src/moving-tensors/dma-engine.md:514): 텐서를 클러스터/칩 전반에 파티션별 소스 패턴대로 재분배. `DmTensorView`의 `dm_cluster_shuffle`/`dm_chip_shuffle`, `HbmTensor`의 `hbm_cluster_shuffle`/`hbm_chip_shuffle`. 칩 간은 시스템 전역 칩 ID를 씁니다. `hbm_chip_shuffle`만 DMA 컨텍스트 제네릭(tdma/pdma 둘 다)인데, HBM↔HBM이 두 DMA가 공통으로 지원하는 유일한 쌍이기 때문입니다.

Scatter/Gather (docs/src/moving-tensors/dma-engine.md:538): 고정 stride가 아니라 인덱스 텐서가 계산한 주소로 이동. `DmTensor::dma_scatter`는 DM 값을 인덱스 위치의 HBM에 쓰고, `HbmTensor::dma_gather`는 인덱스 위치의 HBM row를 DM으로 읽습니다. `scaled` 인자: true는 인덱스를 gather 축 byte-offset으로, false는 raw row 위치로 해석 (docs/src/moving-tensors/dma-engine.md:570).

PCIe DMA (docs/src/moving-tensors/dma-engine.md:574): host 시스템 메모리 ↔ device HBM. 온칩 Tensor DMA와 별개의 물리 엔진입니다. `HostTensor`에 `.to_hbm()`(host→device), `HbmTensor`에 `.to_host()`(device→host), 둘 다 async. `HostTensor`는 Element 매핑만 가지고(host 메모리엔 chip/cluster/slice 파티션 없음) 목적지 HbmTensor가 Chip 축을 더합니다. 대역폭 30 B/cycle로 온칩(256)보다 한 자릿수 느리니, host↔device 트래픽은 최소화하고 한 번 올린 데이터를 온칩 연산 여러 번에 재사용해야 합니다.

---

## 6. 메모리 성능 — 깊이 들어가기

위반 시 처리량이 깨지는 규칙 요약 (docs/src/moving-tensors/memory-performance.md:18): DM 뱅크 기아(연속 같은 뱅크 64회 미만 유지, 위반=NoC 타임아웃→하드웨어 리셋), DM DMN 인터리빙(클러스터당 2 DMN 번갈아, 위반=50% 손실), DM 슬라이스 인터리빙(DMN당 32 슬라이스 분산), HBM 정렬(256바이트, unaligned read 2×·write ~50× RMW), HBM 뱅크 충돌(같은 뱅크 row 전환 회피, 30~40× 저하), HBM 채널 인터리빙(32채널 분산).

### 6.1 DM 구조

기하 (docs/src/moving-tensors/memory-performance.md:30): 칩당 256MB, 클러스터 2/칩, DMN 8/클러스터, 슬라이스 32/DMN, 뱅크 16/슬라이스, row 4096/뱅크, 8바이트/row. 슬라이스 하나는 512KB SRAM에 16개 병렬 뱅크(각 8바이트)로 총 128 B/cycle. 개별 뱅크 접근은 직렬이지만 주소 공간이 연속 128바이트를 16뱅크에 8바이트씩 분산해 병렬 접근을 가능케 합니다. DM 주소 비트: 0~2 byte, 3~6 bank, 7~18 row (docs/src/moving-tensors/memory-performance.md:53). 그래서 연속 주소가 서로 다른 뱅크로 가 순차 스캔 시 병렬 접근이 됩니다.

DMN/슬라이스 인터리빙 (docs/src/moving-tensors/memory-performance.md:62): DMN 하나는 128 B/cycle뿐(32 슬라이스가 데이터 경로 공유)이라 표준 256바이트 전송 단위는 DMN당 2사이클이 듭니다. 두 DMN에 파이프라인하면 연속 처리량이 유지됩니다. 슬라이스는 DMA/Fetch/Commit이 공유하므로 32 슬라이스에 요청을 퍼뜨리면 경합이 줍니다. 각 슬라이스는 2-entry 명령 큐를 가지며, M개 슬라이스에 분산하면 슬라이스당 필요 처리량이 1/M로 떨어집니다.

### 6.2 뱅크 기아 — 가장 위험한 함정

뱅크 기아는 DMA 엔진이 높은 우선순위 엔진이 점유한 DM 뱅크를 무한정 기다리며 막히는 현상입니다. 64-접근 규칙이 이를 막습니다. 어기면 NoC 타임아웃과 전체 클러스터 리셋이 일어나 모든 계산 상태가 날아갑니다 (docs/src/moving-tensors/memory-performance.md:86).

DM 컨트롤러 우선순위 (docs/src/moving-tensors/memory-performance.md:93): main-context Fetch > main-context Commit > sub-context Fetch > sub-context Commit > DMA. DMA가 최하위인 이유는 계산 엔진이 정상 동작 중 데이터에 먼저 접근해야 하기 때문입니다. 그런데 높은 우선순위 엔진이 같은 뱅크를 계속 때리면 DMA 요청이 큐에서 진전을 못 합니다. Tensor DMA는 NoC 허브로 DRAM·DMN과 통신하는데, 각 포트(DMA, DRAM, DMN)는 4,096 사이클 안에 요청을 ack해야 합니다. 4,096 사이클 무응답이면 NoC가 트랜잭션을 죽은 것으로 선언하고 예외 상태로 진입합니다(교착·무한 hang 탐지용 안전장치). 타임아웃이 걸리면 우아한 복구가 없고, 유일한 복구는 전체 클러스터 도메인 리셋입니다.

왜 64인가? `(TDMA_IO_BYTE / DMN_IO_BYTE) * Max_Consecutive_Access * DMN_SIZE < 4096`이고 TDMA_IO_BYTE=256, DMN_IO_BYTE=128, DMN_SIZE=32을 넣으면 `2 * x * 32 < 4096 → x < 64`입니다 (docs/src/moving-tensors/memory-performance.md:110). 최악의 경우에도 DMA 요청이 NoC 타임아웃 전에 끝나도록 보장합니다.

이 한계는 누적입니다 (docs/src/moving-tensors/memory-performance.md:122): 같은 뱅크에 대한 모든 엔진의 총 접근이 64 미만이어야 합니다. 예) main 30 + sub 20 + DMA 1 = 51(안전), main 30 + sub 35 + DMA 1 = 66(기아 유발). 컴파일러는 개별 명령을 64 미만으로 유지하지만, 여러 명령이 동시에 돌면 합이 64에 닿는 것까지는 막지 못합니다. 그래서 TU 연산이 64-접근 한계를 위반할 것 같으면 컴파일러가 그 연산을 DMA를 점유하는 것처럼 스케줄해 동시 DMA를 막습니다. 이는 main/sub/DMA 병렬성을 희생하지만 치명적 리셋을 피합니다. 절대 64회+ 연속 같은 뱅크 패턴을 쓰지 마세요. main이 sub를 굶기는 건 덜 심각합니다(NoC 타임아웃·리셋 없이 처리 시간만 증가).

### 6.3 HBM 구조

기하 (docs/src/moving-tensors/memory-performance.md:173): 칩당 48GB, 스택 2/칩, 채널 16/스택, 슬라이스 3/채널, 뱅크그룹 4/슬라이스, 뱅크 4/뱅크그룹, row 16K/뱅크, 2K바이트/row. 32채널 × 48GB/s = 1.5 TB/s. 채널 컨트롤러는 0.75GHz에서 64B/cycle인데 사이클당 8 burst라 실효 6GHz. 기본 전송 단위 256바이트 = 채널당 4 클럭 사이클 (docs/src/moving-tensors/memory-performance.md:186).

HBM 주소 비트는 병렬 순차 접근에 최적화된 비선형 매핑입니다 (docs/src/moving-tensors/memory-performance.md:201): 0~7 byte, 8 stack, 9~12 channel, 13 bank group, 14~16 byte, 17~18 bank, 19 bank group, 20 slice, 21~33 row, 34 slice, 35 row. 특이점 셋: 슬라이스는 3개뿐인데 비트 2개(20, 34); 비트 34는 row의 영향을 받아 34·35가 동시에 1이 안 되게 해 연속 48GB 주소 공간을 보장; 채널은 비트 9~12와 13~28의 XOR. 이 순서 덕에 순차 접근이 스택·채널·뱅크그룹·뱅크에 동시에 퍼집니다.

### 6.4 HBM 성능 함정

오정렬 (docs/src/moving-tensors/memory-performance.md:232): 256바이트 경계를 넘는 read는 두 번 전송(2× 페널티), 정렬 안 된 write는 RMW(약 50× 페널티 — 256바이트 단위를 통째로 읽고 일부만 바꿔 다시 씀). 256바이트 최소 단위는 비트 0~7로 정의됩니다.

뱅크 충돌 (docs/src/moving-tensors/memory-performance.md:244): HBM 뱅크는 한 번에 한 row만 엽니다. 같은 뱅크 안 다른 row로 전환하면 현재 row를 닫고 새 row를 여는데 40~50ns(1.5GHz에서 60~75 사이클) 지연이 붙어, 이미 열린 row 접근보다 30~40× 느립니다. 모든 row는 닫힌 채 시작하므로 첫 접근은 항상 open 비용을 냅니다. 채널 인터리빙이 완화합니다 — 32채널에 분산하면 충돌이 줍니다. 비트 8~12(스택·채널)를 낮은 주소에 두는 게 핵심입니다. 특히 스택은 비트 8 하나에만 대응하니, 프로그래머가 두 스택을 번갈아 접근하도록 명시적으로 보장해야 풀 스택 인터리빙이 됩니다. 컨트롤러는 FR-FCFS(이미 열린 row 우선) 스케줄링과 뱅크 간 명령 인터리빙으로 row 전환 지연을 숨기지만, 같은 뱅크에서 계속 row를 바꾸는 패턴은 여전히 크게 저하됩니다.

tCCD (Column-to-Column Delay) (docs/src/moving-tensors/memory-performance.md:269): 같은 채널에서 연속 read/write 명령 사이 최소 시간으로 최대 발행률을 정합니다. 같은 슬라이스 다른 뱅크그룹 = 2사이클(이상적, 상대 성능 1); 다른 슬라이스 = 3사이클(2/3, 데이터 경로 전환); 같은 슬라이스 같은 뱅크그룹 = 4사이클(1/2, 네 뱅크가 I/O 버퍼 공유). 대개는 tCCD가 병목이 되기 전에 뱅크 충돌이나 채널 인터리빙이 먼저 지배합니다.

마지막으로, 어느 연산이 병렬로 도는지와 컨텍스트 할당을 검증하려면 Schedule Viewer(`--dump-schedule` → furiosa-schedule-viewer GUI)를 쓸 수 있습니다 (docs/src/moving-tensors/memory-performance.md:136).

## 2. 핵심 API · 패턴

| 이름 | 쓰는 법 | 설명 | 출처 |
|---|---|---|---|
| `TuTensor::fetch` | `fn fetch<D2: Scalar, Time2: M, Packet2: M>(self) -> FetchTensor<...> where D: FetchCast<D2>` | BeginTensor를 패킷 스트림으로. D2=타입캐스트, Time2=시간 스텝, Packet2=패킷 레이아웃. Cluster=2/Slice=256/8바이트정렬 검증. | `furiosa-opt-std/src/engine/fetch.rs:35` |
| `ctx.main.begin_interleaved` | `ctx.main.begin_interleaved::<I, _, _, _, _, _>(lhs.view(), rhs.view()).fetch()` | 매핑 같은 두 텐서를 한 페치로 교대. 최대 2개. I=2 축이 교대를 인코딩. | `docs/src/moving-tensors/fetch-engine.md:127` |
| `fetch_with_table / fetch_with_zero_point(s)` | `input.fetch_with_table(table); input.fetch_with_zero_point(zp); input.fetch_with_zero_points([zp1,zp2])` | Fetch Adapter: 룩업테이블(Sigmoid/GeLU/MXFP4), 비대칭 양자화 zero-point 빼기(텐서별 가능). | `docs/src/moving-tensors/fetch-engine.md:301` |
| `TuTensor::commit / commit_view` | `fn commit<Element: M>(self, address: Address) -> DmTensor<...>; fn commit_view<Element>(self, dst: DmTensorViewMut)` | 스트림을 DM에. Element가 Time/Packet 대체하며 Time 재배열로 전치 가능. 입력=32B flit, 출력=8/16/24/32B. | `furiosa-opt-std/src/engine/commit.rs:27` |
| `commit_cast / commit_cast_relu` | `input.commit_cast(0); input.commit_cast_relu(0)` | DM 쓰기 직전 f32→bf16(+선택적 ReLU). Vector 엔진을 비워 sub-context 병렬 실행 가능케 함. | `docs/src/moving-tensors/commit-engine.md:121` |
| `HbmTensor::to_dm` | `fn to_dm<Cluster: M, Slice: M, Element2: M>(&self, dma: &mut DmaContext<{Dma::Tensor}>, address: Address) -> DmTensor<...>` | HBM→DM. Cluster/Slice 축으로 파티션 분산. assert_dma_layout으로 8바이트 정렬 검증. | `furiosa-opt-std/src/tensor/memory.rs:423` |
| `to_hbm (DmTensor/HbmTensor)` | `fn to_hbm<const DMA: Dma, Element2: M>(&self, dma: &mut DmaContext<{DMA}>, address: Address) -> HbmTensor<...>` | DM→HBM 또는 HBM→HBM(전치/레이아웃 변환). 두 번째 인자가 목적지 HBM 주소. | `furiosa-opt-std/src/tensor/memory.rs:387` |
| `Context::tdma / Context::pdma` | `&mut ctx.tdma (온칩 HBM/DM/SPM), &mut ctx.pdma (host↔HBM)` | Tensor DMA는 온칩 전송, PCIe DMA는 host↔HBM 전용. PCIe는 async, 30 B/cycle. | `docs/src/moving-tensors/dma-engine.md:18` |
| `HostTensor::to_hbm / HbmTensor::to_host` | `host.to_hbm(&mut ctx.pdma, addr).await; hbm.to_host::<m![...]>(&mut ctx.pdma).await` | PCIe DMA. HostTensor는 Element만, 목적지 HbmTensor가 Chip 축 추가. 둘 다 async. | `docs/src/moving-tensors/dma-engine.md:580` |
| `dma_scatter / dma_gather` | `data_dm.dma_scatter::<m![K],_,_>(index, output, scaled); table.dma_gather::<...>(index, addr, scaled)` | 인덱스 텐서 기반 가변 주소 이동. scaled=true는 byte-offset, false는 raw row 위치. | `docs/src/moving-tensors/dma-engine.md:538` |
| `dm_cluster_shuffle / hbm_chip_shuffle` | `input.view().dm_cluster_shuffle::<2>(&mut ctx.tdma, &[1,0])` | 클러스터/칩 전반 재분배. hbm_chip_shuffle만 DMA 컨텍스트 제네릭(HBM↔HBM이 공통 지원 쌍이므로). | `docs/src/moving-tensors/dma-engine.md:514` |
| `tile / view / view_mut` | `input.tile::<m![B], 1, m![A, 1 # 32]>(b); output.view_mut().tile::<...>(b)` | 텐서의 부분 영역 뷰 생성(슬라이싱). DMA copy(to_hbm_view/to_dm_view)와 함께 부분 전송에 사용. | `furiosa-opt-examples/src/tile.rs:11` |
| `DmTensor::reshape (unsafe)` | `let reshaped: DmTensor<...> = unsafe { dm_tensor.reshape() };` | 메모리 레이아웃 그대로 두고 매핑 타입만 재해석. 안전성은 호출자 책임이라 unsafe. | `furiosa-opt-examples/src/reshape.rs:17` |

## 3. 실험 (직접 돌리기)

> 실험은 NPU 없이 `simulation`·`typecheck`로 돌아갑니다. 실행법은 [`../experiments/README.md`](../experiments/README.md), MNIST는 `cargo furiosa-opt test`(npu 전용).

### 실험 05.1 — fetch+collect+commit 파이프라인 시뮬레이션 실행
*난이도 1/5 · 기반: `furiosa-opt-examples/src/fetch_commit.rs`*

**목표** — DM→TU→DM의 전체 이동 경로(to_dm 후 fetch→collect→commit→to_hbm)가 시뮬레이션에서 수치적으로 맞는지 확인하고, i8→i32 캐스트와 [A,B]→[B,A] 전치가 함께 일어나는 걸 본다.

```bash
cd ~/furiosa-opt/furiosa-opt-examples && cargo furiosa-opt test --release --test fetch_commit_tests test_fetch_commit_simple_host
```
**관찰** — 테스트 통과(녹색). 호스트 레퍼런스가 out[b*4096+a] = (a*8+b) as i8 as i32로 전치+캐스트를 검증한다(tests/fetch_commit_tests.rs:18). 통과는 곧 fetch가 i8을 i32로 올리고 collect가 32B flit으로 정규화하고 commit이 [B,A]로 다시 쓴 결과가 모두 일치함을 의미.

**심화** — --backend typecheck 로 같은 명령을 돌려보면 값 비교 루프가 빈 텐서라 0회 반복으로 통과한다. simulation과 typecheck의 차이(값 검증 유무)를 체감하라.

### 실험 05.2 — Fetch의 Cluster=2 / Slice=256 제약을 일부러 깨보기
*난이도 2/5 · 기반: `furiosa-opt-examples/src/fetch_assertions.rs`*

**목표** — verify_fetch가 강제하는 Cluster::SIZE==2, Slice::SIZE==256 하드웨어 불변식이 실제로 panic을 내는지 확인한다(이미 invalid/valid 함수가 준비돼 있음).

```bash
cd ~/furiosa-opt/furiosa-opt-examples && cargo furiosa-opt --backend typecheck test --release --test fetch_assertions_tests
```
**관찰** — valid_cluster_size / valid_slice_size 테스트는 통과. invalid_* 함수(to_dm을 m![1 # 4] 클러스터나 m![1 # 512] 슬라이스로 호출)는 실행되면 furiosa-opt-std/src/engine/fetch.rs:49의 assert_eq!로 'Cluster size must be 2' 또는 'Slice size must be 256' panic을 낸다. invalid 함수를 호출하는 #[should_panic] 테스트를 직접 추가해 메시지를 확인하라.

**심화** — valid 함수의 to_dm Cluster를 m![1 # 2]에서 m![1 # 4]로 바꾸고, fetch의 결과 타입도 맞춰 컴파일한 뒤 simulation 실행 시 panic 메시지를 직접 읽어보라.

### 실험 05.3 — predict-then-run: 3D 전치 [A,B,C]→[C,A,B]
*난이도 2/5 · 기반: `furiosa-opt-examples/src/transpose.rs`*

**목표** — HBM↔HBM tdma 전치가 두 단계(0→0x1000)로 일어날 때 결과 인덱스를 손으로 예측한 뒤 시뮬레이션 결과와 맞춰본다.

```bash
cd ~/furiosa-opt/furiosa-opt-examples && cargo furiosa-opt test --release --test transpose_tests test_transpose_simple
```
**관찰** — 통과. 레퍼런스(tests/transpose_tests.rs:30)가 expected[c,a,b] = a*16*32 + b*32 + c로 [A=8,B=16,C=32]를 [C,A,B]로 재배열한다. 실행 전에 (a,b,c)=(1,2,3)의 입력 선형 인덱스와 출력 위치를 직접 계산해 맞춰보라.

**심화** — transpose_simple을 한 단계 to_hbm 한 번으로 줄여서([A,B,C]→[C,A,B] 직접) 결과가 같은지 확인. 중간 단계가 정말 필요 없는지 따져보라.

### 실험 05.4 — DMA tail 8바이트 정렬 위반 만들기 (find-the-error)
*난이도 3/5 · 기반: `furiosa-opt-examples/src/fetch_commit.rs`*

**목표** — to_dm의 Element 가장 안쪽 축을 소스의 연속 축과 어긋나게 만들어 assert_dma_layout의 'DMA tail alignment violation'을 유발한다. HBM→DM write 8바이트 정렬 규칙을 체감.

```bash
cd ~/furiosa-opt/furiosa-opt-examples && cp src/fetch_commit.rs /tmp/fc_backup.rs && cargo furiosa-opt --backend typecheck test --release --test fetch_commit_tests
```
**관찰** — 먼저 원본이 통과함을 확인(주석이 'Element 가장 안쪽 = B(8×i8=8바이트)라 min_align=8 만족'이라 설명, src/fetch_commit.rs:15). 이어 to_dm의 Element를 m![A / 8 % 2, A % 8, B]에서 B를 빼거나 B를 더 작은 조각으로 바꿔 tail이 8바이트 미만이 되게 하면 furiosa-opt-std/src/tensor/memory.rs의 assert_dma_layout이 panic. 실험 후 cp /tmp/fc_backup.rs src/fetch_commit.rs 로 복구.

**심화** — i8 대신 i32로 바꾸면 같은 B=8이 32바이트가 되어 정렬 여유가 어떻게 달라지는지 예측하고 확인하라.

### 실험 05.5 — view 기반 DM↔DM 부분 복사 + 패딩 마스킹 관찰
*난이도 3/5 · 기반: `furiosa-opt-examples/src/view.rs`*

**목표** — tile/view로 만든 부분 텐서끼리 DM-to-DM 복사가 어떻게 회전(rotate)되는지, 그리고 패딩 슬롯(index 63)이 Uninit으로 보존되는지 확인한다.

```bash
cd ~/furiosa-opt/furiosa-opt-examples && cargo furiosa-opt test --release --test view_tests
```
**관찰** — test_view_simpl 통과: 입력 [0,1,2,3]이 출력 [3,0,1,2]로 회전(tests/view_tests.rs:24). test_view_padding은 simulation/typecheck 전용(#[cfg(not(backend="npu"))])으로, 63번째 패딩 슬롯이 Opt::Uninit으로 남는다. view.rs의 두 tile 복사(input012→output123, input3→output0)가 회전을 만드는 원리를 코드와 대조하라.

**심화** — 두 복사의 순서를 바꾸면 결과가 어떻게 깨지는지 예측한 뒤(겹치는 슬롯 때문) 실행으로 확인.

### 실험 05.6 — base-template 풀 파이프라인 실행 + 뱅크/슬라이스 매핑 바꾸기
*난이도 2/5 · 기반: `furiosa-opt-examples/base-template/src/kernel/constant_add_kernel.rs`*

**목표** — constant_add 커널(to_dm→fetch→collect→vector→commit→to_hbm)을 돌려 전체 이동+계산 경로를 보고, Slice 매핑을 바꿔 같은 결과가 나오는지(레이아웃 무관 수학적 동등성) 확인한다.

```bash
# base-template로 생성한 프로젝트에서: cargo furiosa-opt run --release --bin constant_add ; cargo furiosa-opt test --release --bin constant_add
```
**관찰** — run은 'Constant Add: kernel ran' 출력, test는 out[i]=in[i]+1 검증 통과(constant_add.rs:28). 커널은 2048원소를 256슬라이스(슬라이스당 8원소)로 to_dm한 뒤 fetch/collect/vector_fxp(AddFxp,1)/commit 한다(constant_add_kernel.rs:12). Slice를 m![A / 8 # 256]로 두는 이유(Slice::SIZE=256 제약)를 fetch 제약과 연결해 이해하라.

**심화** — axes![A]를 4096으로 키우고 Slice 매핑을 m![A / 16 # 256]으로 바꿔 슬라이스당 16원소가 되게 한 뒤 여전히 통과하는지(그리고 Slice::SIZE는 여전히 256이어야 함) 확인.

## 4. 연습문제 (손으로, 컴파일 없이)

**Q1.** Config = [8:1, 8:8, 3:64, 4:192] : 1 (sequencer.md의 NCHW→WHCN 예). 이 시퀀서의 contiguous_run과 access_size는? 그리고 패킷당 몇 번 접근하는가?

<details><summary>정답/힌트</summary>

가장 안쪽 엔트리 W=8:1에서 시작. 다음 H=8:8은 s_outer=1 != n_inner*s_inner? 안쪽→바깥 판정: 8(W의 stride 윗단계)와 비교. 사실 Packet=m![1]이라 packet_size=1 → access_size=gcd(1, contiguous_run)=1. 패킷당 1번 접근(Packet::SIZE=1).

</details>

**Q2.** i8 텐서를 fetch하는데 access_size=64로 나왔다. main-context에서 read_size는 몇 바이트(몇 원소)인가? Packet::SIZE=64라면 패킷당 read 횟수와, Time::SIZE=10일 때 총 사이클은?

<details><summary>정답/힌트</summary>

read_size는 access_size의 약수 중 D[read_size]가 1/2/4/8/16/32바이트인 최대. i8은 1바이트/원소라 read_size=32원소(32바이트)가 최대. 패킷당 read=Packet::SIZE/read_size=64/32=2. 총 사이클=Time::SIZE*(Packet::SIZE/read_size)=10*2=20.

</details>

**Q3.** 어떤 커널이 main-context Fetch로 같은 DM 뱅크를 연속 30회, 동시에 sub-context Fetch가 같은 뱅크를 35회, DMA가 1회 요청한다. 안전한가? 위험하면 무슨 일이 일어나는가?

<details><summary>정답/힌트</summary>

64-접근 규칙은 누적. 30+35+1=66 >= 64라 뱅크 기아 발생. DMA 요청이 NoC 4096사이클 타임아웃에 걸려 전체 클러스터 도메인 리셋(모든 계산 상태 손실). 64 미만으로 유지해야 함(memory-performance.md:122).

</details>

**Q4.** spot-the-error: `to_dm::<m![1 # 4], m![1 # 256], m![A, B]>(...)` 후 그 DmTensor를 fetch한다. 컴파일/실행 시 무엇이 잘못되는가?

<details><summary>정답/힌트</summary>

Cluster가 m![1 # 4]라 Cluster::SIZE=4. verify_fetch가 Cluster::SIZE==2를 요구하므로 'Cluster size must be 2, got 4' panic(fetch.rs:49). Cluster는 m![1 # 2]여야 함.

</details>

**Q5.** HBM에서 4,099바이트 패킷으로 write를 하려 한다. 왜 문제이며 하드웨어가 어떻게 처리하는가? 추가로 stride0이 256의 배수가 아니면 HBM write에 무슨 페널티가 붙는가?

<details><summary>정답/힌트</summary>

DMA 최대 패킷은 4,096바이트(AXI 256 beats×16). 4,099는 한계를 넘어 여러 명령으로 쪼개야 함(명령마다 ~500 startup). stride0이 256 비정렬이면 부분 256바이트 블록에 RMW 페널티(약 50×)가 HBM write에 추가됨(dma-engine.md:276).

</details>

**Q6.** 버퍼가 m![A % 5, A / 5] (A=15)로 저장돼 있는데 스트림을 m![A % 3, A / 3]로 읽으려 한다. 왜 거부되는가?

<details><summary>정답/힌트</summary>

버퍼는 A를 5×3으로, 스트림은 3×5로 분해. gcd(5,3)=1이라 어느 분해도 다른 쪽을 세분하지 못함. 컴파일러가 5-block에 이미 commit된 Buf에서 A%3을 소비할 수 없어 incompatible decomposition으로 거부(총 원소 수가 같아도) — sequencer.md:507.

</details>

**Q7.** Fetch 출력 패킷을 20바이트로 잡았다. Collect의 32바이트 flit 관점에서 출력 대역폭 낭비율은? 40바이트라면?

<details><summary>정답/힌트</summary>

20바이트는 flit 하나(32B)에 12바이트 제로패딩 → 12/32 = 37.5% 낭비. 40바이트는 flit 두 개(64B)에 걸쳐 마지막 24바이트 제로패딩 → 24/64 = 37.5% 낭비(fetch-engine.md:148).

</details>

**Q8.** Commit padding_chunking 예: access_size=8, valid_size=32. write_size는? 32바이트 패킷은 몇 번의 write로 쪼개지며 DM offset은?

<details><summary>정답/힌트</summary>

write_size=gcd(valid_size, access_size)=gcd(32,8)=8. 패킷당 write=valid_size/write_size=32/8=4. M축 따라 offset 0,16,32,48에 8바이트씩(commit-engine.md:233).

</details>

## 5. 흔한 함정

- 같은 DM 뱅크를 연속 64회 이상 접근하면(누적 — main+sub+DMA 합산) NoC 타임아웃 후 전체 클러스터가 리셋되어 모든 계산 상태가 사라진다. 성능 저하가 아니라 하드웨어 리셋이라 가장 치명적. 절대 64+ 연속 같은 뱅크 패턴을 쓰지 말 것.  
  ↳ 출처 `docs/src/moving-tensors/memory-performance.md:108`
- Fetch는 Cluster::SIZE가 반드시 2, Slice::SIZE가 반드시 256이어야 한다(verify_fetch의 assert). to_dm에서 Cluster를 m![1 # 4]나 Slice를 m![1 # 512]로 잡으면 fetch에서 panic. 출력 패킷도 8바이트 정렬 필수.  
  ↳ 출처 `furiosa-opt-std/src/engine/fetch.rs:49`
- HBM↔DM DMA는 주소 정렬 표(DM write 8바이트)와 무관하게 read 주소·write 주소·packet 크기 모두 8바이트 정렬이 추가로 필요하다. 정렬 안 된 DM write는 RMW로 쓰기 시간이 3배가 되고 해당 뱅크를 막는다. to_dm의 Element 가장 안쪽 축을 소스 연속 축과 맞춰 DMA tail이 8바이트 배수가 되게 해야 함.  
  ↳ 출처 `docs/src/moving-tensors/dma-engine.md:224`
- HBM 스택 비트(주소 비트 8)를 토글하지 않는 접근 패턴은 32채널 중 16개로만 트래픽을 몰아 유효 대역폭을 절반으로 깎는다. 채널 비트(9-28)는 자연스레 인터리빙돼도 스택 비트는 프로그래머가 명시적으로 두 스택을 번갈아야 한다.  
  ↳ 출처 `docs/src/moving-tensors/memory-performance.md:257`
- DMA 패킷 stride0이 256바이트의 배수가 아니면 HBM write가 마지막 부분 블록에 RMW 페널티(약 50×)를 낸다. 또 C(가장 안쪽 패킷 축)가 256의 배수가 아니면(C=256n+r) 데이터 양은 거의 안 변해도 사이클이 n+1배로 폭증한다.  
  ↳ 출처 `docs/src/moving-tensors/dma-engine.md:396`
- 텐서 모양이 DMN에 고르게 나뉘지 않으면 컴파일러가 heterogeneous aggregate로 폴백해 경계 DMN을 여러 명령으로 쪼개고, 명령마다 ~500사이클 startup을 따로 낸다(예제 6: A=15가 4 DMN에 안 나뉘어 DMN 3이 병목). 가능하면 DMN 수로 나눠떨어지는 모양을 쓸 것.  
  ↳ 출처 `docs/src/moving-tensors/dma-engine.md:482`
- Fetch 타입 캐스팅은 캐스트 출력이 페치당 단일 32바이트 flit에 들어가야 한다. i8→i32에서 read_size를 16(16바이트)으로 키우면 16×4=64B가 되어 무효. 캐스트가 있으면 read_size 상한이 더 낮아진다는 점을 잊기 쉽다.  
  ↳ 출처 `docs/src/moving-tensors/fetch-engine.md:354`
- 마스킹을 빼먹으면 flit 폭 배수가 아닌 축(예: 63→64로 올림)의 패딩 슬롯이 임의값을 가져 sum/max 같은 리덕션 결과가 조용히 틀어진다. 패딩 슬롯은 반드시 중립값(sum이면 0)으로 마스킹해야 한다.  
  ↳ 출처 `docs/src/moving-tensors/fetch-engine.md:196`
- reshape는 unsafe다. 메모리 레이아웃을 그대로 두고 매핑 타입만 재해석하므로, 입력 DM 레이아웃과 목표 매핑이 실제로 일치하는지는 호출자가 보장해야 한다. 잘못 쓰면 조용히 잘못된 데이터를 읽는다.  
  ↳ 출처 `furiosa-opt-examples/src/reshape.rs:17`
- pseudo read는 브로드캐스트(Buf에 없는 축을 stride 0으로 복제)를 허용하지만 write/Commit은 거부한다. 각 버퍼 슬롯이 정확히 하나의 소스 위치를 가져야 하기 때문. Commit 경로에서 브로드캐스트 축을 기대하면 안 된다.  
  ↳ 출처 `furiosa-opt-std/src/tensor/pseudo.rs:51`

## 6. 핵심 정리 & 다음

기억할 사실:
- 세 이동 엔진의 역할: Fetch = DM→Tensor Unit 스트림, Commit = Tensor Unit 스트림→DM, DMA = DM·SPM·HBM 중 임의 두 계층 직접 이동. TRF/VRF는 이동 엔진이 아니라 TU 프리미티브가 채움. (`docs/src/moving-tensors/index.md:4`)
- 칩당 메모리 피크 대역폭: DM 2 TB/s, SPM 2 TB/s, HBM 1.5 TB/s. (`docs/src/moving-tensors/memory-performance.md:9`)
- Sequencer Config 한계: 엔트리 최대 8개, 엔트리당 size <= 65,536, 패킷 크기는 1/2/4/8/16/32 바이트 중 하나, 가장 안쪽 엔트리는 연속((s==0||s==1)&&n%packet_size==0) 또는 이산(packet_size==1) 접근. (`docs/src/moving-tensors/sequencer.md:493`)
- access_size = gcd(Packet::SIZE, contiguous_run); contiguous_run은 인접 엔트리가 s_outer == n_inner*s_inner일 때까지 곱해 나간 가장 안쪽 연속 구간 길이. (`docs/src/moving-tensors/sequencer.md:143`)
- Fetch 하드웨어 검증: Cluster::SIZE는 반드시 2, Slice::SIZE는 반드시 256, 출력 패킷은 8바이트(FETCH_ALIGN_BYTES) 정렬. 위반 시 런타임 panic. (`furiosa-opt-std/src/engine/fetch.rs:49`)
- Fetch read_size: main-context는 access_size의 약수 중 D[read_size]가 1/2/4/8/16/32 바이트인 최대값, sub-context는 8바이트 고정. Packet::SIZE>read_size면 multi-read, 사이클 = Time::SIZE*(Packet::SIZE/read_size). (`docs/src/moving-tensors/fetch-engine.md:60`)
- Fetch 타입 캐스팅은 캐스트 출력이 페치당 단일 32바이트 flit에 들어가야 함. i8→i32에서 read_size=8(8B)은 32B로 유효, read_size=16(16B)은 64B로 무효. (`docs/src/moving-tensors/fetch-engine.md:354`)
- Commit 입력 패킷은 정확히 한 flit(32바이트), 출력 패킷은 8/16/24/32 바이트 중 하나(COMMIT_OUT_PACKET_SIZES). 모든 시퀀서 stride는 8바이트 배수. (`furiosa-opt-std/src/engine/commit.rs:47`)

➡️ 다음: [06_computing_engines_1.md](./06_computing_engines_1.md)
