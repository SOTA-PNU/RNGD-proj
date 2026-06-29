# 01 · 멘탈 모델과 큰 그림

> **이 모듈에서 배우는 것**: vISA가 무엇이고 왜 존재하는지, RNGD 하드웨어 계층·Tensor Unit 파이프라인·메모리 계층·실행 컨텍스트·수학 배경(텐서 수축)·4개 백엔드, 그리고 5개 quick-start 커널의 큰 그림을 잡습니다.  
> **선행**: 없음 (여기서 시작) · **예상 시간**: 40분  
> **참고** — 용어는 [`../GLOSSARY.md`](../GLOSSARY.md), API 빠른참조는 [`../CHEATSHEET.md`](../CHEATSHEET.md), 실험 실행법은 [`../experiments/README.md`](../experiments/README.md)

## 학습 목표

- [ ] vISA가 PyTorch/XLA·저수준 커널 API와 어떤 점에서 다른지 한 문장으로 설명한다
- [ ] Chip/Cluster/Slice/Lane 계층과 RNGD 개수를 말한다
- [ ] Tensor Unit 8단계 파이프라인과 메모리 5계층의 역할을 안다
- [ ] 텐서 수축 = Broadcast→Multiply→Reduce 임을 안다
- [ ] 4개 백엔드(typecheck/simulation/emulation/npu)의 차이를 안다

## 1. 개념

## 1. vISA / TCP가 무엇이고 왜 존재하나요

FuriosaAI의 가속기는 이름이 TCP, 즉 Tensor Contraction Processor 입니다. "텐서 수축(tensor contraction)을 빠르게 돌리려고 만든 추론용 칩"이라는 뜻이 이름에 그대로 담겨 있습니다 (introduction.md:3). 그리고 vISA(Virtual Instruction Set Architecture, 가상 명령어 집합)는 그 칩을 직접 프로그래밍하는 인터페이스입니다 (introduction.md:6-7).

여기서 핵심은 "추상화의 높이"입니다. 같은 NPU 칩을 다루는 길이 세 가지가 있는데 높이가 다릅니다.

- PyTorch / XLA 같은 고수준 프레임워크: 메모리 레이아웃과 하드웨어 스케줄링을 전부 숨겨 줍니다. 편하지만 "어느 메모리에 올릴지, 어느 순서로 돌릴지"를 내가 정할 수 없습니다 (introduction.md:4).
- 저수준 커널 API: 바이트 단위로 주소를 계산해야 하는 세계입니다. 강력하지만 너무 번거롭습니다.
- vISA: 그 중간입니다. "텐서 단위로 생각하면서도, 메모리 할당과 Tensor Unit 스케줄링은 내가 직접 제어"합니다 (introduction.md:7). 즉 byte 단위 reasoning 없이도 직접 제어권을 줍니다 (introduction.md:4).

우리 프로젝트 맥락에서 한 가지를 분명히 해두면 좋습니다. furiosa-llm의 컴파일러는 닫힌(closed) 미리 빌드된 바이너리라 우리가 손댈 수 없고, 그 컴파일러가 만드는 EDF 포맷(CBOR 그래프)으로만 serve 게이트를 통과합니다. 반면 vISA는 furiosa-opt-std 라는 공개 Rust API로, RNGD의 저수준 커널을 직접 작성할 수 있게 열어 줍니다. 다만 vISA가 만드는 산출물 포맷(.bin, pert-ipc)은 furiosa-llm의 .edf와 서로 달라서, vISA로 짠 것을 정식 serve 스택에 주입하지는 못합니다. 같은 칩을 향하는 "다른 경로"라고 이해하면 됩니다. 이 강의는 그 다른 경로, 즉 vISA 자체를 처음부터 배우는 자리입니다.

대상 독자는 두 부류입니다 (introduction.md:8). vISA를 손으로 직접 쓰는 프로그래머, 그리고 vISA를 자동 생성하는 컴파일러를 만드는 개발자입니다. 둘 다 기본적인 Rust 지식을 전제합니다 (introduction.md:9). vISA 프로그램은 결국 "furiosa-opt-std API를 쓰는 평범한 Rust 프로그램"이기 때문입니다 (introduction.md:114).

한 가지 주의: 이 빌드는 알파 테스트, 실험적이고 미완성인 소프트웨어입니다 (introduction.md:13-15). 프로덕션에 쓰기 전에는 Furiosa 엔지니어와 상의해야 합니다.

## 2. 수학 배경 — 텐서, 모양(shape), 텐서 수축, einsum

TCP를 이해하려면 먼저 이 칩이 "텐서 네이티브" 프로세서라는 걸 받아들여야 합니다 (quick-start.md:9). 즉 텐서가 1급 시민입니다.

### 텐서와 모양(shape)

텐서는 "텐서 인덱스 → 값"으로 가는 함수이고, 그 텐서의 모양(shape)이 어떤 인덱스가 유효한지를 정합니다 (quick-start.md:13).

여기서 일반적인 NumPy 사고방식과 결정적으로 다른 점이 하나 있습니다. **모양은 "이름 붙은 축들의 순서 없는 집합(unordered set)"입니다** (quick-start.md:15). 무슨 말이냐면, `{N=4, C=3}` 과 `{C=3, N=4}` 는 완전히 같은 텐서입니다 (quick-start.md:16). 의미를 결정하는 것은 축의 위치가 아니라 축의 이름이라는 뜻입니다. NumPy에서는 `(4,3)` 과 `(3,4)` 가 다른 배열이지만, vISA에서는 축에 N, C 같은 이름을 붙이기 때문에 순서가 의미를 갖지 않습니다.

이게 왜 중요할까요. 나중에 보겠지만 vISA에서는 같은 데이터를 슬라이스에 펼치거나(Slice 축), 시간축(Time)으로 쪼개거나, 패킷(Packet)으로 나누는 식으로 "같은 논리 축을 하드웨어의 여러 차원에 재배치"합니다. 축에 이름이 있고 순서가 없기 때문에 이런 재배치를 타입으로 자유롭게 표현할 수 있습니다.

물론 한번 축 순서를 정하고 나면, 텐서는 우리가 익숙한 다차원 배열처럼 행동합니다 (quick-start.md:20). 0D는 스칼라(예: 5.2), 1D는 벡터, 2D는 행렬, 4D는 `{N=4, C=3, H=256, W=512}` 같은 RGB 이미지 배치입니다 (quick-start.md:21-24).

### 텐서 수축(tensor contraction) = 행렬곱의 일반화

텐서 수축은 행렬곱을 임의의 텐서로 일반화한 것입니다 (quick-start.md:29). 정의는 단순합니다. 두 입력 텐서를 원소별로 곱한 뒤, 둘이 공유하는(수축되는) 축을 따라 합칩니다.

그리고 **모든 수축은 정확히 세 단계로 분해됩니다: Broadcast → Multiply → Reduce** (quick-start.md:30). 이 세 단계가 나중에 하드웨어 엔진들과 거의 1:1로 대응되기 때문에 꼭 기억해야 합니다.

- Broadcast: 한쪽에만 있는 축을 다른 쪽으로 펼칩니다.
- Multiply: 원소별 곱셈.
- Reduce: 공유 축을 따라 더해서 그 축을 없앱니다.

이걸 짧게 적는 표기법이 einsum 입니다 (quick-start.md:31). 규칙은: 각 입력 텐서를 그 축 라벨로 나열하고, `→` 뒤에 출력 축을 적고, 입력에는 있는데 출력에 없는 축이 곧 수축되는 축입니다.

세 가지 예가 quick-start.md:35-39 표에 정리돼 있습니다.

- 내적(Dot product): `I, I → 1`. 두 입력 모두 축이 I로 같아서 Broadcast가 필요 없고, `x_i y_i` 를 곱한 뒤 `Σ_i x_i y_i` 로 줄입니다.
- GEMV(행렬×벡터): `IJ, J → I`. 벡터 x를 I 방향으로 Broadcast하고, `A_ij x_j` 를 곱한 뒤 J로 줄여 `y_i = Σ_j A_ij x_j`.
- GEMM(행렬×행렬): `IK, KJ → IJ`. A는 J 방향으로, B는 I 방향으로 Broadcast하고, `A_ik B_kj` 를 곱한 뒤 K로 줄여 `C_ij = Σ_k A_ik B_kj`.

(주의: 위 표는 교과서식 `KJ`로 적혀 있지만, 실제 quick-start의 GEMM 커널은 B를 전치해서 `JK`로 저장합니다. 뒤의 GEMM 절과 Gotcha에서 다시 짚습니다.)

## 3. 하드웨어 계층 — Chip / Cluster / Slice / Lane

TCP 디바이스는 네 단계로 중첩된 하드웨어로 이루어집니다 (quick-start.md:45-52). RNGD 기준 개수까지 외워두면 코드의 숫자들이 바로 이해됩니다.

- Chip: 최상위 단위, HBM을 가집니다. 개수는 시스템마다 다릅니다(system-dependent).
- Cluster: 칩당 2개. 256개의 슬라이스를 묶습니다.
- Slice: 클러스터당 256개. 각 슬라이스가 하나의 Tensor Unit을 돌립니다.
- Lane: 슬라이스당 8개. Contraction Engine의 MAC(곱셈-누산) 배열의 한 행(row)입니다.

계산해 보면 칩 하나에 클러스터 2개 × 슬라이스 256개 = 512 슬라이스, 그리고 슬라이스마다 레인 8개니까 512×8 = 4096 레인이 됩니다. 핵심 직관은 "슬라이스가 병렬 처리의 기본 단위"라는 점입니다. 슬라이스마다 독립된 Tensor Unit 파이프라인이 돌아갑니다 (computing-tensors/index.md:51). 그래서 뒤에 나오는 `Slice = m![A / 8 # 256]` 같은 타입은 "데이터를 256개 병렬 슬라이스에 어떻게 펼칠까"를 정하는 일입니다.

## 4. Tensor Unit — 8단계 고정 파이프라인

Tensor Unit은 온칩 연산 파이프라인입니다. DM에서 텐서를 읽어, 8개 엔진을 거쳐 변형한 뒤, 결과를 다시 DM에 씁니다 (computing-tensors/index.md:6-7).

순서는 고정입니다: **Fetch → Switch → Collect → Contraction → Vector → Cast → Transpose → Commit** (quick-start.md:57, computing-tensors/index.md:21).

데이터는 이 파이프라인을 "사이클당 패킷 하나"씩 스트림으로 흘러갑니다 (computing-tensors/index.md:8). 각 엔진은 이 스트림을 소비하고 생산하면서, 사이클당 레이아웃과 반복 순서를 조금씩 바꿔 나갑니다.

각 엔진의 역할과 핵심 제약(computing-tensors/index.md:39-48):

- Fetch: DM에서 데이터를 파이프라인으로 로드합니다. 패킷은 8바이트 정렬이어야 하고, Slice는 바뀌지 않습니다.
- Switch(Switching): 데이터를 슬라이스 사이로 이동시킵니다. 링(ring) 네트워크를 쓰며, 유일하게 Slice를 바꿀 수 있습니다 (quick-start.md:59). 대부분 엔진은 슬라이스 안에서 독립적으로 동작하는데, Switch만 슬라이스들을 연결해 데이터를 슬라이스 배열 전체에 퍼뜨립니다.
- Collect: 들어온 패킷들을 정확히 32바이트짜리 flit으로 정규화합니다. 출력은 정확히 flit 하나입니다 (computing-tensors/index.md:10, 43). 이 32바이트 flit이 그 아래 모든 엔진(Contraction, Vector, Cast, Transpose, Commit)이 다루는 단위입니다 (computing-tensors/index.md:11).
- Contraction: einsum, 즉 matmul·convolution·attention을 수행합니다. 한 피연산자는 TRF에 상주하고 다른 하나는 스트림으로 흘러 들어옵니다.
- Vector: 원소별·이항·리듀스 연산. 입력은 i32 또는 f32만 받습니다 (이건 중요한 제약입니다).
- Cast: 정밀도 낮추기(예: f32 → bf16)를 배치로 수행합니다. 출력은 정확히 flit 하나.
- Transpose: flit 안에서만 원소를 재배열합니다(within-flit only).
- Commit: 결과를 DM에 다시 씁니다. flit 정렬된 쓰기.

### 스트림이 가진 5차원과 두 부류

Tensor Unit 안을 흐르는 모든 텐서 스트림은 다섯 차원 `[Chip, Cluster, Slice, Time, Packet]` 을 가지며, 이게 두 부류로 나뉩니다 (computing-tensors/index.md:50-52).

- 공간(spatial) 차원: Chip, Cluster, Slice. 슬라이스마다 자기 파이프라인 인스턴스가 돌고, 슬라이스는 클러스터로, 클러스터는 칩으로 묶입니다.
- 슬라이스별 스트림: Time, Packet. Time은 파이프라인 반복(iteration)을 인덱싱하고, Packet은 각 반복 안의 원소를 인덱싱합니다 (quick-start.md:90).

엔진들은 Time/Packet을 파이프라인을 따라 재구성합니다. 공간 차원은 두 엔진을 빼고 모두 보존됩니다: Switch는 Slice를 바꾸고(슬라이스 간 데이터 이동), Vector의 inter-slice reducer는 클러스터 안 256개 슬라이스를 가로질러 합치며 Slice를 붕괴시킵니다 (computing-tensors/index.md:54).

또 하나 기억할 구조: Contraction과 Vector 엔진은 한쪽 피연산자를 파이프라인 스트림에서, 다른 한쪽을 전용 슬라이스별 레지스터 파일에서 받습니다 (computing-tensors/index.md:56). TRF는 Contraction에, VRF는 Vector에 먹입니다. Collect 엔진이 `.to_trf()`로 TRF에, `.to_vrf()`로 VRF에 써넣습니다 (computing-tensors/index.md:57-58).

## 5. 메모리 계층 — HBM / DM / SPM / TRF / VRF

데이터가 어디에 사느냐가 vISA 프로그래밍의 절반입니다 (quick-start.md:65-71).

- HbmTensor: 패키지 위(on-package). 48 GB, 1.5 TB/s. 장기 보관용 — 가중치와 활성값의 큰 저장소.
- DmTensor: 온칩 SRAM. 총 256 MB, 슬라이스당 512 KB. 연산의 주 작업 메모리(primary working memory)입니다.
- SpmTensor: 온칩 SRAM. 크기는 TBD, 칩당 2 TB/s. 시간적 지역성(temporal locality)이 높은 임시·중간 결과용이며, 컴파일러가 관리합니다.
- TrfTensor: 온칩 SRAM. 레인당 8 KB(슬라이스당 8레인). Contraction Engine용 레지스터 파일.
- VrfTensor: 온칩 SRAM. 슬라이스당 8 KB. Vector Engine용 피연산자 레지스터 파일.

머릿속 그림은 이렇습니다. 큰 데이터는 HBM에 있고 → DMA로 DM(빠른 온칩 작업 메모리)으로 옮긴 뒤 → 파이프라인이 DM에서 읽어 계산하고 → 다시 DM에 쓰고 → DMA로 HBM으로 보냅니다. TRF/VRF는 파이프라인 도중 "한쪽 피연산자를 슬라이스마다 상주시켜 매 사이클 읽는" 아주 작고 빠른 칸입니다.

용량 감각도 중요합니다. DM은 슬라이스당 512 KB뿐이라, 큰 워크로드는 DM 용량을 넘깁니다. 그래서 타일링이 필요합니다 (quick-start.md:282-284): 시간적 분할(temporal partitioning)은 타일을 시간 순서로 처리하고, 공간적 분할(spatial partitioning)은 타일을 병렬 하드웨어 유닛에 펼칩니다.

## 6. 텐서 매핑 — 타입으로 표현하는 하드웨어 분배

vISA의 진짜 매력이자 처음엔 가장 어려운 부분입니다. **타입 시스템이 하드웨어 계층을 그대로 드러냅니다** (quick-start.md:79). 각 텐서 타입은 원소 타입(예: bf16)과, 각 논리 축이 하드웨어 계층에 어떻게 분배되는지를 함께 인코딩합니다 (quick-start.md:80).

문서의 예시를 풀어 보면(quick-start.md:82-83): `axes![A = 2048]` 일 때

```
DmTensor<bf16, m![1], m![1 # 2], m![A / 8 # 256], m![A % 8]>
```

는 "A 축을 가진 bf16 벡터를, 칩 하나(`m![1]`)의, 두 클러스터 중 하나(`m![1 # 2]`)에, 256개 슬라이스에 분산(`m![A / 8 # 256]`)하고, 슬라이스당 8원소(`m![A % 8]`)"로 둔다는 뜻입니다. 결과적으로 A의 각 원소는 정확히 한 슬라이스 안의 잘 정의된 위치로 매핑됩니다.

`m![]` 매핑 표현식의 세 연산자(quick-start.md:85-88):

- `/` 는 stride로 나눕니다: `A / 8` 은 2048 / 8 = 256개의 슬라이스 인덱스를 만듭니다.
- `%` 는 안쪽 개수를 줍니다: `A % 8` 은 슬라이스 안 8개 인덱스.
- `#` 는 하드웨어 유닛 개수에 맞춰 패딩합니다: `# 256` 은 256 슬라이스로 채우고, 남는 슬롯에는 임의 값(arbitrary)이 들어갑니다.

그래서 `m![1 # 2]` 는 "값 1이지만 2개 클러스터 칸 중 하나"이고, `m![A / 8 # 256]` 은 "A를 8로 나눈 256개를 256 슬라이스에 정확히 채움"입니다.

여기에 파이프라인 전용 두 파라미터가 더 있습니다 (quick-start.md:90): Time은 파이프라인 반복을, Packet은 각 반복 안의 원소를 인덱싱합니다. 즉 `DmTensor<...>` 같은 메모리 텐서는 Chip/Cluster/Slice/Element로 표현되고, 파이프라인을 흐르는 텐서(FetchTensor, CollectTensor 등)는 Chip/Cluster/Slice/Time/Packet으로 표현됩니다.

## 7. 실행 컨텍스트 — main / sub (그리고 DMA)

모든 디바이스 커널에는 별도 하드웨어 자원에서 동시에 도는 두 실행 컨텍스트가 있습니다: `ctx.main` 과 `ctx.sub` (quick-start.md:96).

- main: 주 계산을 돌립니다. Tensor Unit의 모든 엔진을 구동할 수 있습니다 (computing-tensors/index.md:73).
- sub: 동시 파이프라인을 돌립니다. 보통 main이 계산하는 동안 TRF나 VRF로 피연산자를 미리 가져오는(prefetch) 용도입니다 (quick-start.md:98). sub는 Contraction Engine과 몇몇 기능을 못 쓰고, 나머지는 main과 같습니다 (computing-tensors/index.md:74).
- DMA: 사실 세 번째 컨텍스트입니다 (computing-tensors/index.md:71). Tensor Unit 바깥에서 DMA 엔진만 구동해 HBM↔DM, HBM↔SPM, DM↔SPM 사이 대량 데이터를 옮깁니다. 코드에서는 `ctx.tdma`(텐서 DMA)와 `ctx.pdma`(PCIe DMA)로 나타납니다 (context.rs:63-65).

동기화 규칙(quick-start.md:99): main이 필요한 피연산자를 sub가 아직 가져오는 중이면, main이 자동으로 sub의 완료를 기다립니다. 즉 정확성은 보장됩니다.

스케줄링 규칙(computing-tensors/index.md:77-82): 한 컨텍스트 안에서는 연산이 직렬화되고, 서로 다른 컨텍스트끼리는 병렬로 돕니다. 그래서 sub가 다음 배치를 TRF/VRF로 미리 채우는 동안 main이 현재 배치를 계산하는 더블 버퍼링(double-buffering), DMA가 독립적으로 대량 이동을 겹치는 오버랩(overlap)이 가능합니다. 단, 일부 엔진은 한 번에 한 컨텍스트만 구동할 수 있는 "하나의 스케줄링 단위"를 이룹니다 — 예를 들어 Vector Engine과 Cast Engine이 한 단위라서, sub가 Vector를 돌리는 동안 main은 Cast 대신 Commit Engine의 타입 캐스팅으로 우회합니다.

메모리 주의(quick-start.md:101-102): 두 컨텍스트는 같은 평평한(flat) 온칩 SRAM을 공유합니다. 그래서 프로그래머가 DM 주소를 명시적으로 배정해 텐서끼리 겹치지 않게 해야 합니다. 이게 `.to_dm()`, `.commit()` 의 `addr` 인자입니다. 주소는 충돌하면 안 되지만 비연속(non-contiguous)이어도 됩니다. 실제 커널에서 `0`, `1 << 12`, `1 << 13`, `1 << 28` 같은 서로 다른 주소가 보이는 이유가 이것입니다.

타입 시그니처에서는 const-generic `Tu` 가 어느 컨텍스트를 흐르는 텐서인지 식별합니다: `{ Tu::Main }` 또는 `{ Tu::Sub }` (quick-start.md:104, context.rs:20-27).

## 8. 4개 백엔드 — 무엇이 실제로 도는가

vISA 프로그램은 furiosa-opt-std API를 쓰는 Rust 프로그램이고, `cargo furiosa-opt` 가 `--cfg backend="..."` 를 세팅해 어떤 백엔드가 커널을 평가할지 고릅니다 (introduction.md:114). 백엔드는 네 개입니다 (Cargo.toml.liquid:10의 check-cfg가 `simulation`, `emulation`, `npu`, `typecheck` 넷을 선언; backend.rs:7-15가 넷을 모두 설명).

- typecheck: 커널 몸체가 phantom(빈) 텐서로 돕니다. 값 계산은 건너뛰고 모양/매핑 오류만 빠르게 잡습니다 (introduction.md:118, backend.rs:14-15). SDK·NPU 불필요.
- simulation: 기본값. 호스트 CPU에서 매핑 표현식으로 연산을 해석하는 완전한 호스트측 해석(full host-side interpretation)입니다. 수치 정확성을 검증합니다 (introduction.md:119, backend.rs:9). SDK·NPU 불필요.
- emulation: 미래의 Cpu+Buffer 인터프리터용 호스트측 `BufRawTensor` 저장소인데, 오늘은 값 생성 메서드가 전부 `todo!()` 플레이스홀더입니다 (backend.rs:10-11). 즉 실제로는 아직 동작하지 않습니다.
- npu: 컴파일된 EDF를 실제 하드웨어에서 돌립니다 (introduction.md:120). 호스트 코드가 네이티브 스테이징 버퍼를 소유하고 `to_hbm`/`from_hbm`에서 DMA를 수행하지만, 호스트에서 텐서 수학을 해석하지는 않습니다 (backend.rs:12-13). 실제 NPU와 SDK가 필요합니다 (introduction.md:40).

중요한 현실(introduction.md:42-44): simulation과 typecheck는 SDK가 전혀 필요 없고 호스트에서 NPU 의존성 없이 돕니다. 그리고 공개 SDK 배포판에는 오늘 호스트용 NPU 시뮬레이터가 없습니다. 물리 NPU가 없으면 simulation(호스트 해석) 또는 typecheck(매핑/모양 검증만)를 쓰면 됩니다. 우리 환경에서 NPU 없이 학습할 수 있는 이유가 바로 이것입니다.

명령 형태(introduction.md:122-140): `cargo furiosa-opt run --release` 가 기본 simulation. `--backend typecheck`, `--backend npu` 로 바꿉니다. `cargo furiosa-opt` 는 모든 cargo 플래그를 그대로 전달하므로 build/test/check/run 모두 대응됩니다. 한 가지 함정: 그냥 `cargo check` 는 타입 체커만 돌고 커널 함수 몸체를 실행하지 않아서 `Collect output packet must be exactly 32 bytes` 같은 매핑 단언에 도달하지 못합니다. 그런 검증은 `--backend typecheck run` 으로 해야 합니다 (introduction.md:133).

## 9. 프로젝트 레이아웃 (base-template)

`cargo generate furiosa-ai/furiosa-opt base-template` 로 스캐폴딩하면 다섯 개 워크드 예제가 함께 옵니다 (introduction.md:48-54). 구조의 핵심(introduction.md:58-79):

- `src/furiosa-opt.tag`: rustc 플러그인이 스캔하는 마커. 반드시 `src/` 바로 아래 있어야 합니다.
- `src/lib.rs`: `pub mod kernel;` (lib.rs:1-5).
- `src/kernel/`: 모든 `#[device]` 함수가 사는 곳. `mod.rs` 가 각 커널을 re-export.
- `src/<name>.rs`: 각 호스트 프로그램. `Cargo.toml` 에 `[[bin]] path = "src/<name>.rs"` 로 등록됩니다 (Cargo.toml.liquid:15-33).

왜 `[[bin]]` 의 명시적 경로가 load-bearing이냐면(introduction.md:79): rustc 플러그인은 `src/` 에 루팅된 cargo 타깃만 스캔하고 `src/bin/`, `examples/`, `tests/` 아래 것은 조용히 건너뛰기 때문입니다. 그래서 호스트 프로그램을 반드시 `src/` 바로 밑에 두고 명시적으로 등록해야 합니다.

각 호스트 프로그램의 `main()` 은 `launch(kernel, ...)` 만 합니다 (launch는 runtime/mod.rs:195). 호스트 레퍼런스와의 값 비교는 같은 파일의 `#[cfg(test)] mod tests` 블록에 있습니다. 예를 들어 constant_add.rs:28-39 는 `out[i] = in[i] + 1` 을 호스트에서 계산해 비교합니다. typecheck 백엔드에서는 `actual` 이 빈 Vec(phantom)이라 비교 루프가 0번 돌아 단언이 자연히 통과합니다 (constant_add.rs:34-39, introduction.md:100-102).

커널을 추가하려면(introduction.md:105): `src/kernel/` 에 파일 하나 드롭 + `src/kernel/mod.rs` 에 `pub mod ...;` 추가 + `src/` 에 호스트 프로그램 작성 + `Cargo.toml` 에 `[[bin]] path = "src/<name>.rs"` 선언. 네 군데를 맞춰야 합니다.

## 10. 다섯 개 quick-start 커널 — 개념 훑기

이제 다섯 예제를 위 개념으로 읽어 봅시다. 각각이 "새 하드웨어 개념 하나"씩을 도입합니다 (quick-start.md:3-5).

### (1) Constant Addition — Vector Engine + DM 분배 (constant_add_kernel.rs)

정수 벡터의 모든 원소에 상수 1을 더합니다. 칩 1개, 두 클러스터 중 1개, 그 안의 256 슬라이스 전부, 슬라이스당 8원소 그룹 하나를 씁니다 (quick-start.md:113). 타입으로는 `Slice = m![A / 8 # 256]`, 슬라이스 안은 `m![A % 8]` (constant_add_kernel.rs:7, 12).

흐름: `input.to_dm()` 이 평평한 2048 원소를 256 슬라이스에 8개씩 쪼개 HBM→DM으로 옮기고(constant_add_kernel.rs:12), `begin → fetch → collect → vector_init → vector_intra_slice_tag → vector_fxp → vector_final → commit` 체인이 각 슬라이스를 한 패스로 처리합니다 (quick-start.md:131). `vector_fxp(FxpBinaryOp::AddFxp, 1)` 가 상수 1 더하기입니다 (constant_add_kernel.rs:25). `TagMode::Zero` 는 파이프라인을 매 사이클 실행하도록 설정합니다 (quick-start.md:132). 입력 DM은 주소 0, 결과는 `1 << 12`(=4096)에 커밋해 겹치지 않게 하고, 마지막에 `1 << 28` 으로 HBM에 씁니다 (constant_add_kernel.rs:28, 31). 새 개념: Vector Engine과 to_dm 분배.

### (2) Elementwise Multiplication — sub 컨텍스트 + VRF (elementwise_mul_kernel.rs)

같은 모양 두 벡터를 원소별로 곱합니다 (quick-start.md:148). 한 피연산자는 파이프라인을 흐르고, 다른 하나는 VRF(슬라이스별 레지스터 파일, Vector Engine이 매 사이클 읽음)에 둡니다 (quick-start.md:150).

새 개념은 sub 컨텍스트입니다 (quick-start.md:175). sub가 `rhs_dm` 을 Fetch → Collect → `.to_vrf(0)` 로 VRF에 미리 싣고(elementwise_mul_kernel.rs:20-26), main은 `lhs_dm` 을 스트림하며 `vector_fxp(FxpBinaryOp::MulInt, &rhs_vrf)` 로 VRF 짝과 곱합니다 (elementwise_mul_kernel.rs:37). 하드웨어는 가능한 만큼 두 컨텍스트를 동시에 돌립니다. 겹침 방지를 위해 lhs는 0, rhs는 `1 << 12` 에 둡니다 (elementwise_mul_kernel.rs:16-17, quick-start.md:177).

### (3) Dot Product — Contraction Engine + TRF (dot_product_kernel.rs)

내적 `I, I → 1` 은 두 피연산자를 같은 축으로 줄이며 Broadcast가 없습니다 (quick-start.md:195). 한쪽은 파이프라인을 흐르고, 다른 하나는 TRF(Contraction Engine이 매 사이클 읽는 슬라이스별 레지스터 파일)에 고정됩니다 (quick-start.md:197). sub가 `rhs` 를 Fetch → Collect → `.to_trf(TrfAddress::Full)` 로 싣습니다(`Full` 은 TRF 전체를 이 텐서에 할당, dot_product_kernel.rs:27, quick-start.md:199).

main 쪽 Contraction 4단계가 핵심입니다 (quick-start.md:201-206, dot_product_kernel.rs:36-40):
- `.contract_outer::<m![A/32], m![A%32], _, _>(&rhs)`: Stream Adapter가 인접한 32바이트 flit 둘을 Outer 단계의 64바이트 패킷으로 짝짓고(시간 스텝 A/16→A/32로 절반), TRF Sequencer가 고정 RHS를 읽어, 레인별 원소 곱셈기에 먹입니다.
- `.contract_packet::<m![1]>()`: 그 곱들을 하드웨어 리덕션 트리로 공간적으로 더합니다.
- `.contract_time::<m![1]>()`: 시간적으로 누산해 슬라이스당 스칼라를 만듭니다.
- `.contract_lane::<m![1], m![1 # 8]>(LaneMode::Interleaved)`: 8개 레인을 출력으로 접습니다(여기선 Lane=m![1]이라 자명한 fold).
마지막에 `.cast::<bf16, ...>()` 가 f32 누산 결과를 bf16으로 되돌립니다 (dot_product_kernel.rs:40, quick-start.md:206). 여기선 `Slice = m![1 # 256]` 로 1개 슬라이스만 활성화합니다(dot_product_kernel.rs:7). 새 개념: Contraction Engine과 TRF.

### (4) GEMV — Switch Engine 브로드캐스트 (gemv_kernel.rs)

GEMV `IJ, J → I` 는 출력 차원 I를 슬라이스에 분배합니다: 각 슬라이스가 한 행 `y_i = Σ_j A_ij x_j` 를 계산합니다 (quick-start.md:223, gemv_kernel.rs:7의 `Slice = m![I]`). 내적과 달리 각 슬라이스가 자기 행과 곱하려면 전체 벡터가 필요하므로, 수축 전에 벡터를 모든 슬라이스로 브로드캐스트해야 합니다 (quick-start.md:224).

그 브로드캐스트를 Switch Engine이 합니다 (quick-start.md:226): `SwitchConfig::Broadcast01` 이 Fetch와 Collect 사이에서 벡터를 모든 I 슬라이스에 분배합니다 (quick-start.md:239-244). 벡터가 퍼지고 나면 J에 대한 수축이 타일 단위로 진행되는데, `Time = m![J / 32]` 가 타일 반복을, `Packet = m![J % 32]` 가 타일 안 원소를 인덱싱합니다 (gemv_kernel.rs:8-9, quick-start.md:227-228). main은 행렬을 흘리며 `contract_outer::<Time, Packet, _, _>(&vector_trf)` 부터 같은 Contraction 4단계를 거쳐 행별 결과를 냅니다 (gemv_kernel.rs:39-43). 새 개념: Switch Engine.

### (5) GEMM — 두 출력 차원을 슬라이스에 함께 매핑 (gemm_kernel.rs)

GEMM은 출력 차원이 둘(I와 J)입니다 (quick-start.md:261). 새 개념은 `type Slice = m![I / 32, J / 32]` 로, **두 출력 차원을 동시에 Slice에 매핑**해 각 슬라이스가 출력 행렬의 16×16 타일을 계산하게 하는 것입니다 (quick-start.md:264, gemm_kernel.rs:8). Switch Engine이 B의 각 타일을 맞는 슬라이스로 옮겨, 각 슬라이스가 자기 J 부분만 보게 합니다 (quick-start.md:265). 각 행렬은 자기에게 없는 출력 차원으로 브로드캐스트됩니다: A는 J로, B는 I로 (quick-start.md:262).

Contraction은 (gemm_kernel.rs:40-43): `contract_packet::<m![1]>()` 가 K를 공간적으로 줄이고, `contract_time::<m![I % 32, J / 8 % 4]>()` 가 시간 누산하며 I·J를 보존하고, `contract_lane::<..., m![J % 8]>(LaneMode::Interleaved)` 가 Lane을 출력 패킷에 접으며 I·J를 함께 보존합니다 (quick-start.md:266-267). 여기서 `Lane = m![J % 8]` 로 레인을 J에 씁니다(gemm_kernel.rs:9). 새 개념: 다차원 출력의 Slice 공동 매핑.

### 다섯 예제를 관통하는 한 장의 그림

constant_add는 Vector+DM, elementwise_mul은 sub+VRF, dot_product는 Contraction+TRF, gemv는 Switch 브로드캐스트, gemm은 2D 출력 타일링 — 이렇게 한 단계씩 쌓아 올립니다. 공통 골격은 늘 같습니다: HBM→DM(to_dm) → 파이프라인(begin/fetch/collect/...연산.../commit) → DM→HBM(to_hbm). 호스트 프로그램은 `Context::acquire()` 로 컨텍스트를 잡고, `HostTensor::rand` 로 입력을 만들고, `.to_hbm(&mut ctx.pdma, addr)` 로 PCIe DMA를 통해 HBM에 올린 뒤, `launch(kernel, (&mut ctx, ...))` 로 커널을 띄웁니다 (constant_add.rs:8-12). 이 패턴이 다섯 예제에서 모두 동일합니다.

## 2. 핵심 API · 패턴

| 이름 | 쓰는 법 | 설명 | 출처 |
|---|---|---|---|
| `Context::acquire` | `let mut ctx = Context::acquire(); // ctx.main, ctx.sub (TuContext), ctx.tdma, ctx.pdma (DmaContext)` | 디바이스 컨텍스트(싱글톤 Mutex)를 잡는다. main=주 계산, sub=동시 prefetch, tdma=텐서 DMA, pdma=PCIe DMA. | `furiosa-opt-std/src/context.rs:57-83` |
| `#[device(chip = N)]` | `#[device(chip = 1)] pub fn my_kernel(ctx: &mut Context, input: &HbmTensor<...>) -> HbmTensor<...> { ... }` | 함수를 launch()용 디바이스 진입점으로 표시. src/kernel/ 아래 두고 mod.rs에서 re-export 해야 한다. | `furiosa-opt-macro/src/lib.rs:95-101; base-template/src/kernel/constant_add_kernel.rs:9` |
| `launch` | `let out = launch(my_kernel, (&mut ctx, &in_hbm)).await;` | 디바이스 커널을 선택된 백엔드로 실행. 호스트 main()과 #[tokio::test] 양쪽에서 사용. | `furiosa-opt-std/src/runtime/mod.rs:195` |
| `HostTensor::rand / to_buf / to_hbm / to_host` | `let x = HostTensor::<i32, m![A]>::rand(&mut rng); let x_hbm = x.to_hbm(&mut ctx.pdma, 0).await; let v: Vec<i32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.to_buf();` | 호스트측 입력 생성/검증용. to_hbm은 PCIe DMA(pdma)로 HBM에 올림, to_host는 되읽음. typecheck에선 빈 Vec. | `base-template/src/constant_add.rs:10-12,36` |
| `HbmTensor::to_dm` | `let dm = input.to_dm::<Cluster, Slice, Element>(&mut ctx.tdma, addr);` | HBM→DM 텐서 DMA. 타입 인자로 Cluster/Slice/슬라이스내 Element 매핑을 지정해 데이터를 256 슬라이스에 분배. addr는 DM 주소. | `furiosa-opt-std/src/tensor/memory.rs:423; base-template/src/kernel/constant_add_kernel.rs:12` |
| `DmTensor::to_hbm` | `result.to_hbm(&mut ctx.tdma, 1 << 28)` | DM→HBM 텐서 DMA. 커널의 마지막에 결과를 HBM으로 되보냄. | `furiosa-opt-std/src/tensor/memory.rs:387; base-template/src/kernel/constant_add_kernel.rs:31` |
| `TuContext::begin → fetch → collect` | `ctx.main.begin(dm.view()).fetch::<D, Time, Packet>().collect::<Time2, Packet2>()` | 파이프라인 시작. begin은 DmTensorView를 받고, fetch는 DM에서 스트림으로 로드(패킷 8바이트 정렬), collect는 32바이트 flit으로 정규화. | `furiosa-opt-std/src/context.rs:84; furiosa-opt-std/src/engine/fetch.rs:35; furiosa-opt-std/src/engine/collect.rs:43` |
| `vector_init / vector_intra_slice_tag / vector_fxp / vector_final` | `.vector_init().vector_intra_slice_tag(TagMode::Zero).vector_fxp(FxpBinaryOp::AddFxp, 1).vector_final()` | Vector Engine 진입/태깅/연산/종료. vector_fxp는 i32 전용·Way8 요구. TagMode::Zero는 매 사이클 실행. 둘째 인자는 상수(1) 또는 &VrfTensor. | `furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:1130; base-template/src/kernel/constant_add_kernel.rs:22-27` |
| `FxpBinaryOp` | `FxpBinaryOp::AddFxp (wrapping_add), FxpBinaryOp::MulInt (wrapping_mul), AddFxpSat (saturating_add)` | Vector Engine 고정소수점 이항 연산 종류. constant_add는 AddFxp, elementwise_mul은 MulInt 사용. | `furiosa-opt-std/src/engine/vector/op/mod.rs:203-219; furiosa-opt-std/src/engine/vector/op/semantics.rs:64,82` |
| `to_vrf / to_trf` | `....collect(...).to_vrf(0)  /  ....collect(...).to_trf(TrfAddress::Full)` | Collect 출력을 슬라이스별 레지스터 파일에 적재. VRF는 Vector Engine, TRF는 Contraction Engine이 매 사이클 읽음. TrfAddress::Full은 TRF 전체 할당. | `base-template/src/kernel/elementwise_mul_kernel.rs:26; base-template/src/kernel/dot_product_kernel.rs:27` |
| `contract_outer / contract_packet / contract_time / contract_lane` | `.contract_outer::<TimeOut, PacketOut, _, _>(&rhs_trf).contract_packet::<m![1]>().contract_time::<m![1]>().contract_lane::<m![1], m![1 # 8]>(LaneMode::Interleaved)` | Contraction Engine 4단계. outer=64바이트 패킷 짝짓기+TRF 읽기+레인 곱, packet=공간 리덕션, time=시간 누산, lane=레인 folding. dot/gemv/gemm 공통 골격. | `base-template/src/kernel/dot_product_kernel.rs:36-39; docs/src/quick-start.md:201-206` |
| `switch / SwitchConfig::Broadcast01` | `input.switch(SwitchConfig::Broadcast01 { slice1: 256, slice0: 1, time0: 1 })` | Switch Engine으로 벡터를 모든 I 슬라이스에 브로드캐스트(Fetch와 Collect 사이). GEMV에서 각 슬라이스가 전체 벡터를 갖게 함. | `docs/src/quick-start.md:236-244` |
| `cast / commit` | `.cast::<bf16, m![1 # 16]>().commit::<m![A % 8]>(1 << 12)` | cast는 f32 누산 결과를 bf16 등으로 낮춤(출력 flit 하나), commit은 결과를 DM 주소에 flit 정렬로 씀. | `base-template/src/kernel/dot_product_kernel.rs:40-41` |
| `axes! / m! / type Chip,Cluster,Slice` | `axes![A = 2048]; pub type Slice = m![A / 8 # 256]; // / 는 stride 분할, % 는 안쪽 개수, # 는 HW 유닛 수 패딩` | axes!는 명명 축과 크기 선언, m!은 매핑 표현식(축→하드웨어 분배). DmTensor/HbmTensor 타입 인자로 사용. | `docs/src/quick-start.md:85-88; base-template/src/kernel/constant_add_kernel.rs:3-7` |

## 3. 실험 (직접 돌리기)

> NPU 없이 `simulation`(기본)·`typecheck`로 돌아갑니다. 큰 예제(matmul 변형·MNIST·트랜스포머)는 `00_SETUP.md`에서 클론한 상위 `furiosa-opt` 저장소나 [`../reference/examples/`](../reference/examples/)에서 보고, MNIST 테스트는 `cargo furiosa-opt test`(npu 백엔드 전용)임에 유의하세요.

### 실험 01.1 — 첫 시뮬레이션: GEMM 돌려보고 레퍼런스 검증까지
*난이도 1/5 · 기반: `base-template/src/gemm.rs`*

**목표** — NPU 없이 simulation 백엔드로 커널이 끝까지 돌고, 호스트 레퍼런스와 수치가 맞는지 본다. cargo furiosa-opt run/test의 차이를 체득한다.

```bash
cargo install cargo-generate cargo-binstall 2>/dev/null; cargo binstall -y cargo-furiosa-opt; cargo generate furiosa-ai/furiosa-opt base-template --name viatest && cd viatest && cargo furiosa-opt run --release --bin gemm && cargo furiosa-opt test --release --bin gemm
```
**관찰** — run은 'GEMM: kernel ran' 출력. test는 gemm.rs:53-57의 허용오차(tol = max(0.05*|e|, 1.0)) 안에서 C[i,j]=Σ_k A[i,k]*B[j,k]가 맞아 통과. bf16 누산이라 정확히 같지 않고 오차 범위로 비교한다는 점에 주목.

**심화** — --bin을 constant_add, elementwise_mul, dot_product, gemv로 바꿔 5개 모두 run+test 통과를 확인.

### 실험 01.2 — typecheck vs simulation: phantom 텐서의 의미 체감
*난이도 2/5 · 기반: `base-template/src/constant_add.rs`*

**목표** — typecheck 백엔드가 값 계산을 건너뛰고 모양/매핑만 본다는 것을, 같은 test가 어떻게 '자명하게' 통과하는지로 이해한다.

```bash
cd viatest && cargo furiosa-opt --backend typecheck test --release --bin constant_add && cargo furiosa-opt test --release --bin constant_add
```
**관찰** — 둘 다 통과하지만 이유가 다르다. typecheck에선 actual이 빈 Vec(phantom)이라 비교 루프가 0번 돌아 단언 자체가 실행되지 않는다(constant_add.rs:34-39). simulation에선 실제로 out[i]=in[i]+1을 계산해 비교한다. 즉 typecheck 통과 ≠ 수치 정확성 보장.

**심화** — cargo furiosa-opt --backend typecheck run --release --bin constant_add 도 실행해, run은 'kernel ran'만 찍고 값 검증이 없음을 확인.

### 실험 01.3 — 상수를 바꿔 레퍼런스를 깨뜨려 보기 (predict-then-run)
*난이도 2/5 · 기반: `base-template/src/kernel/constant_add_kernel.rs`*

**목표** — 커널만 고치고 호스트 레퍼런스는 그대로 두면 test가 실패함을 직접 확인해, 커널/레퍼런스가 짝이라는 걸 못박는다.

```bash
cd viatest && sed -i 's/FxpBinaryOp::AddFxp, 1)/FxpBinaryOp::AddFxp, 5)/' src/kernel/constant_add_kernel.rs && cargo furiosa-opt test --release --bin constant_add; echo EXIT=$?
```
**관찰** — 커널은 +5를 더하는데 레퍼런스(constant_add.rs:30)는 여전히 +1이라, simulation test가 'constant_add mismatch'로 실패(EXIT 비0)한다. 예측: 거의 모든 원소에서 4 차이.

**심화** — constant_add.rs의 reference도 wrapping_add(5)로 맞춰 다시 test → 통과. 그 뒤 git checkout으로 원복.

### 실험 01.4 — 출력 모양 예측: dot product는 왜 m![1]인가
*난이도 2/5 · 기반: `base-template/src/kernel/dot_product_kernel.rs`*

**목표** — 내적이 I,I→1이라 출력이 슬라이스당 스칼라가 되는 과정을 코드의 contract 단계와 연결해 예측하고 확인한다.

```bash
cd viatest && cargo furiosa-opt --backend typecheck run --release --bin dot_product && cargo furiosa-opt run --release --bin dot_product
```
**관찰** — typecheck가 통과하면 contract_outer→packet→time→lane→cast→commit의 타입 사슬이 일관적이라는 뜻. 반환 타입이 HbmTensor<bf16, Chip, m![1]>(dot_product_kernel.rs:16)인 이유: contract_packet/time/lane이 A를 전부 reduce해 슬라이스당 스칼라(m![1])만 남기고 cast가 f32→bf16로 되돌린다.

**심화** — dot_product_kernel.rs 주석대로 Slice를 m![1 # 256]에서 m![A / 8 # 256]로 바꾸면 타입 사슬이 깨지는지 typecheck로 확인(의도적 에러 관찰).

### 실험 01.5 — 매핑 에러 만들어 보기: Slice 개수를 일부러 틀리기
*난이도 3/5 · 기반: `base-template/src/kernel/elementwise_mul_kernel.rs`*

**목표** — 타입 시스템이 하드웨어 매핑 불일치를 컴파일/타입체크 단계에서 잡아준다는 vISA의 핵심을 에러 메시지로 체험한다.

```bash
cd viatest && sed -i 's/m!\[A \/ 8 # 256\]/m![A \/ 8 # 128]/' src/kernel/elementwise_mul_kernel.rs && cargo furiosa-opt --backend typecheck run --release --bin elementwise_mul; echo EXIT=$?
```
**관찰** — A=2048, A/8=256인데 #128로 패딩하라고 하면 256개를 128 슬라이스에 넣을 수 없어 모양/매핑 단언이 실패한다(EXIT 비0). 'cargo check'가 아니라 '--backend typecheck run'이어야 커널 몸체의 매핑 단언에 도달한다는 점(introduction.md:133)을 함께 확인.

**심화** — git checkout src/kernel/elementwise_mul_kernel.rs로 원복 후, 대신 rhs_dm 주소(1<<12)를 lhs_dm과 같은 0으로 바꾸면 주소 겹침이 어떻게 드러나는지 simulation run으로 관찰.

## 4. 연습문제 (손으로, 컴파일 없이)

**Q1.** axes![A = 2048]일 때 타입 m![A / 8 # 256]과 m![A % 8]은 각각 무엇을 의미하나? 두 값(슬라이스 개수와 슬라이스당 원소 수)을 숫자로 답하라.

<details><summary>정답/힌트</summary>

/ 는 stride 분할이라 2048/8=256(슬라이스 인덱스), %는 안쪽 개수라 8(슬라이스당 원소). #256은 256 슬라이스에 패딩. 즉 256 슬라이스 × 8원소 = 2048 전체가 정확히 채워진다. (quick-start.md:85-88, constant_add_kernel.rs:7,12)

</details>

**Q2.** 다음 einsum을 Broadcast/Multiply/Reduce로 분해하라: (a) I,I→1 (내적) (b) IJ,J→I (GEMV). 각각 무슨 축이 reduce되고 브로드캐스트가 필요한지 말하라.

<details><summary>정답/힌트</summary>

(a) 두 입력 모두 I라 Broadcast 없음, x_i y_i 곱, Σ_i로 I를 reduce해 스칼라. (b) 벡터 x를 I로 Broadcast, A_ij x_j 곱, Σ_j로 J를 reduce해 y_i. 출력에 없는 축이 reduce 대상. (quick-start.md:35-39)

</details>

**Q3.** constant_add 커널의 vector_fxp(FxpBinaryOp::AddFxp, 1)을 그대로 두고 호스트 레퍼런스만 in[i]+2로 바꾼 뒤 (1) simulation test, (2) typecheck test를 돌리면 각각 통과/실패는? 이유는?

<details><summary>정답/힌트</summary>

(1) simulation: 커널은 +1, 레퍼런스는 +2라 거의 모든 원소가 1 차이 → 실패. (2) typecheck: actual이 phantom 빈 Vec이라 비교 루프 0회 → 자명 통과. typecheck 통과가 수치 정확성을 보장하지 않음을 보여준다. (introduction.md:100-102, constant_add.rs:34-39)

</details>

**Q4.** 이 4개 연산을 main과 sub 중 어느 컨텍스트가 맡는 게 자연스러운가? (a) lhs 스트림 곱셈 (b) rhs를 VRF/TRF에 prefetch (c) Contraction Engine 호출. 그리고 (c)를 sub가 못 하는 이유는?

<details><summary>정답/힌트</summary>

(a) main, (b) sub(prefetch가 sub의 전형 용도), (c) main. sub는 Contraction Engine과 일부 기능을 못 쓰기 때문(computing-tensors/index.md:74). main이 계산하는 동안 sub가 다음 피연산자를 채우는 더블 버퍼링 구조.

</details>

**Q5.** 다음 중 파이프라인에서 Slice(공간) 차원을 바꿀 수 있는 엔진은? Fetch / Switch / Collect / Cast / Vector의 inter-slice reducer. 각각 보존/변경을 표시하라.

<details><summary>정답/힌트</summary>

Switch는 Slice 변경 가능(슬라이스 간 이동, 링 네트워크), Vector의 inter-slice reducer는 256 슬라이스를 가로질러 합쳐 Slice 붕괴. Fetch/Collect/Cast는 Slice 보존(대부분 엔진은 슬라이스 내부에서 독립 동작). (computing-tensors/index.md:41-54)

</details>

**Q6.** GEMM 커널의 type Slice = m![I / 32, J / 32]는 무엇을 뜻하나? 각 슬라이스가 출력 행렬에서 몇×몇 타일을 계산하며, 왜 두 출력 차원을 함께 Slice에 매핑하는가?

<details><summary>정답/힌트</summary>

I와 J 두 출력 차원을 동시에 Slice에 매핑해 각 슬라이스가 16×16 출력 타일을 맡는다. 내적/GEMV와 달리 출력이 2차원(I,J)이라, 두 차원을 슬라이스에 펼쳐야 전체 출력 행렬을 병렬로 채울 수 있다. Switch가 B 타일을 맞는 슬라이스로 옮긴다. (quick-start.md:264-265, gemm_kernel.rs:8)

</details>

**Q7.** spot-the-error: 어떤 학습자가 bf16 두 벡터를 원소별로 곱하려고 vector_fxp(FxpBinaryOp::MulInt, &rhs_vrf)를 bf16 텐서에 그대로 썼다. 무엇이 문제이고 어떻게 고치나?

<details><summary>정답/힌트</summary>

vector_fxp는 i32 전용(Vector Engine 입력은 i32/f32만, Way8 요구). elementwise_mul 예제가 i32를 쓰는 이유다. bf16 곱은 부동소수 경로(예: vector_fxp 대신 적절한 float 연산)나 Contraction 경로로 가야 한다. (computing-tensors/index.md:45, vector_tensor.rs:1128-1135)

</details>

## 5. 흔한 함정

- 모양이 '순서 없는 축 집합'이라는 점을 놓치면 NumPy식 (행,열) 직관으로 코드를 잘못 읽는다. {N=4,C=3}과 {C=3,N=4}는 같은 텐서이고, 의미는 위치가 아니라 축 이름이 결정한다. m![]의 / % # 도 위치가 아니라 이름 붙은 축에 작용한다.  
  ↳ 출처 `docs/src/quick-start.md:15-16`
- GEMM 수학 배경 표(quick-start.md:39)는 교과서식 IK,KJ→IJ (A_ik B_kj)로 적혀 있지만, 실제 quick-start GEMM 커널과 GEMM 절은 B를 전치해 IK,JK→IJ (C_ij=Σ_k A_ik B_jk)로 구현한다. B는 m![J,K]로 저장되고 호스트 레퍼런스도 B를 K 청크로 자른다. 표만 보고 B 레이아웃을 K-major로 가정하면 틀린다.  
  ↳ 출처 `docs/src/quick-start.md:39,261-262; base-template/src/gemm.rs:33-48`
- 그냥 cargo check는 타입 체커만 돌고 커널 함수 몸체를 실행하지 않아 'Collect output packet must be exactly 32 bytes' 같은 매핑 단언에 도달하지 못한다. 매핑/모양 단언까지 확인하려면 반드시 cargo furiosa-opt --backend typecheck run 을 써야 한다.  
  ↳ 출처 `docs/src/introduction.md:133`
- typecheck test가 통과해도 수치가 맞는다는 뜻이 아니다. typecheck에선 actual이 phantom 빈 Vec이라 비교 루프가 0번 돈다. 수치 정확성은 simulation test로만 확인된다.  
  ↳ 출처 `docs/src/introduction.md:100-102; base-template/src/constant_add.rs:34-39`
- main과 sub가 같은 평평한 SRAM을 공유하므로 to_dm/commit의 addr를 직접 안 겹치게 줘야 한다. 그래서 lhs는 0, rhs는 1<<12, 결과는 1<<13 식으로 분리한다. 주소를 겹치면 텐서가 서로를 덮어쓴다(충돌 금지, 비연속은 허용).  
  ↳ 출처 `docs/src/quick-start.md:101-102; base-template/src/kernel/elementwise_mul_kernel.rs:16-17`
- Vector Engine 입력은 i32 또는 f32만 받는다. vector_fxp는 i32 전용이고 Way8 모드를 요구한다. 그래서 constant_add/elementwise_mul은 i32를 쓴다. bf16 원소별 연산을 vector_fxp로 바로 하려 하면 막힌다.  
  ↳ 출처 `docs/src/computing-tensors/index.md:45; furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:1128-1135`
- host 프로그램(src/*.rs)을 src/bin/, examples/, tests/ 아래 두면 rustc 플러그인이 조용히 건너뛴다. 반드시 src/ 바로 아래 두고 Cargo.toml에 [[bin]] path="src/<name>.rs"로 명시 등록해야 한다(이 경로가 load-bearing).  
  ↳ 출처 `docs/src/introduction.md:79`
- emulation 백엔드는 이름만 있고 실제로는 값 생성 메서드가 전부 todo!() 플레이스홀더라 동작하지 않는다. 백엔드를 emulation으로 골라 값을 기대하면 안 된다. NPU 없이 값을 보려면 simulation을 써야 한다.  
  ↳ 출처 `furiosa-opt-std/src/runtime/backend.rs:10-11`
- 공개 SDK에는 호스트용 NPU 시뮬레이터가 없다. --backend npu는 실제 물리 NPU와 SDK가 있어야만 의미가 있고, 없으면 simulation/typecheck로만 학습·평가 가능하다.  
  ↳ 출처 `docs/src/introduction.md:42-44`
- Rust 툴체인이 특정 nightly(nightly-2026-05-01)에 ABI 고정돼 있다. furiosa-opt는 rustc 드라이버라 rust-toolchain.toml이 핀한 채널이 아니면 안 돈다. 프로젝트 디렉터리로 cd 하면 cargo가 자동 활성화한다.  
  ↳ 출처 `docs/src/introduction.md:25-31`

## 6. 핵심 정리 & 다음

기억할 사실:
- TCP(Tensor Contraction Processor)는 추론 워크로드를 겨냥한 대규모 병렬 AI 가속기다. PyTorch/XLA처럼 메모리 레이아웃과 스케줄링을 숨기지 않고 직접 제어를 노출하되, 저수준 커널 API의 바이트 단위 reasoning은 요구하지 않는다. (`docs/src/introduction.md:3-4`)
- 하드웨어 4계층(RNGD): Chip은 시스템마다 개수가 다르며 HBM을 가짐 / Cluster는 칩당 2개로 256 슬라이스를 묶음 / Slice는 클러스터당 256개로 각자 하나의 Tensor Unit을 돌림 / Lane은 슬라이스당 8개로 Contraction Engine MAC 배열의 한 행. (칩당 512 슬라이스, 4096 레인) (`docs/src/quick-start.md:47-52`)
- 메모리 계층(RNGD 용량): HBM 48 GB·1.5 TB/s(on-package, 장기 보관) / DM 총 256 MB·슬라이스당 512 KB(온칩 SRAM, 주 작업 메모리) / SPM 크기 TBD·칩당 2 TB/s(컴파일러 관리) / TRF 레인당 8 KB(Contraction용) / VRF 슬라이스당 8 KB(Vector용). (`docs/src/quick-start.md:65-71`)
- Tensor Unit은 고정된 8단계 파이프라인이다: Fetch → Switch → Collect → Contraction → Vector → Cast → Transpose → Commit. 데이터는 사이클당 패킷 하나씩 스트림으로 흐른다. 대부분 엔진은 슬라이스 내부에서 독립 동작하고, Switch만 슬라이스들을 연결한다. (`docs/src/quick-start.md:57-59`)
- Collect 엔진은 들어온 패킷을 정확히 32바이트 flit으로 정규화하며 출력은 flit 하나다. 이 32바이트 flit이 하위 모든 엔진(Contraction/Vector/Cast/Transpose/Commit)이 다루는 단위다. (`docs/src/computing-tensors/index.md:10-11,43`)
- 엔진별 핵심 제약: Fetch는 패킷 8바이트 정렬·Slice 불변 / Switch는 링 네트워크·Slice 변경 가능 / Vector는 입력이 i32 또는 f32만 / Cast는 출력 flit 하나 / Transpose는 flit 안에서만 / Commit은 flit 정렬 쓰기 / Contraction은 한 피연산자 TRF 상주·다른 하나 스트림. (`docs/src/computing-tensors/index.md:39-48`)
- 파이프라인 스트림은 5차원 [Chip, Cluster, Slice, Time, Packet]을 가진다. Chip/Cluster/Slice는 공간 차원(슬라이스마다 독립 파이프라인), Time/Packet은 슬라이스별 스트림. 공간 차원은 Switch(Slice 변경)와 Vector의 inter-slice reducer(클러스터 내 256 슬라이스를 가로질러 합쳐 Slice 붕괴)를 제외한 모든 엔진이 보존한다. (`docs/src/computing-tensors/index.md:50-54`)
- 실행 컨텍스트는 main(주 계산, 모든 TU 엔진 구동) / sub(동시 prefetch, Contraction 엔진과 일부 기능 제외) / DMA(TU 바깥, HBM↔DM·HBM↔SPM·DM↔SPM)의 셋이다. 한 컨텍스트 안은 직렬, 컨텍스트 간은 병렬(더블 버퍼링·오버랩). Vector와 Cast는 한 번에 한 컨텍스트만 쓰는 한 스케줄링 단위를 이룬다. (`docs/src/computing-tensors/index.md:69-82`)

➡️ 다음: [02_mapping.md](./02_mapping.md)
