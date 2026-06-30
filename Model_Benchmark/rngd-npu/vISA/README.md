# vISA 완전 학습 커리큘럼 — RNGD NPU를 직접 손으로 프로그래밍하기

이 문서는 vISA 커리큘럼의 시작점이자 전체 안내입니다. vISA를 전혀 모르는 상태에서 RNGD NPU 커널을 직접 짤 수 있는 수준까지 단계별로 데려갑니다.

> 자료: 퓨리오사 공식 책 <https://developer.furiosa.ai/furiosa-opt/book/> · 저장소 <https://github.com/furiosa-ai/furiosa-opt> (v0.2.0, 원본은 `reference/`). 관련 분석: `../../info/README_virtual_isa.md`

---

## 0. 60초 요약 — vISA가 뭔가요?

- **TCP(Tensor Contraction Processor) = RNGD 칩**. vISA는 그 칩의 **프로그래밍 인터페이스**입니다.
- PyTorch/XLA는 "메모리 배치·스케줄을 알아서 해주는" 높은 층, 어셈블리는 "바이트 다 직접 다루는" 낮은 층이라면, **vISA는 그 중간**입니다. **텐서 단위로 생각하되, 메모리 할당과 연산 엔진 스케줄은 내가 직접** 정합니다.
- 커널은 **그냥 Rust 함수**입니다. 타입시스템이 "이 텐서가 어느 메모리에, 어느 슬라이스에, 어떤 순서로 흐르는지"를 **컴파일 시점에 검사**해요. 잘못 짜면 빌드가 안 됩니다(틀린 결과물을 뱉는 게 아니라).
- 우리가 그동안 싸운 비공개 `furiosa-llm`(모델 컴파일러)보다 **한 계층 아래**입니다. op 지원 목록·model_type 게이트·serve 게이트를 **건너뛰고** 커널을 칩에 올릴 수 있어요.

> 한 문장: **"RNGD의 연산 엔진·메모리·스케줄을 Rust 타입으로 직접 조립하는 저수준 커널 언어."**

---

## 1. 왜 배우나 (우리 작업과의 연결)

- 고수준 컴파일러가 거부하던 연산(gather/scatter, 커스텀 attention, **recurrent/DeltaNet**류)을 **엔진을 직접 두드려** 표현할 길입니다.
- 하드웨어 매뉴얼 자체가 자산입니다 — "왜 컴파일러가 이 op를 조건부로만 지원했나(8-타일 제약 등)"를 1차 사료로 이해합니다.
- 단, 현실도 알고 시작합니다(자세히는 `../../info/README_virtual_isa.md`):
  - 컴파일러(`cargo-furiosa-opt` = `npu_opt`)는 **비공개 미리빌드 바이너리**입니다(소스 없음, binstall만).
  - vISA가 만드는 `.bin`은 furiosa-llm의 `.edf`(CBOR 그래프)와 **포맷이 다릅니다**(pert-ipc 명령어). 같은 칩에서 돌지만 **serve/masquerade에 끼워 넣지는 못합니다** — 별개 실행 경로.
  - **알파** 단계. 퓨리오사 공식 답변(2026-06): 온칩 순환상태(Persistent Kernel)·동적 shape는 2026~1년 로드맵 → 지금은 우리 host-loop·버킷 우회가 "다리".

---

## 2. 이 커리큘럼 사용법

1. **`00_SETUP.md`** 먼저: 툴체인 + 비공개 컴파일러 바이너리 설치, 첫 커널 실행까지(NPU 불필요).
2. **`curriculum/01` → `11`** 순서대로: 각 모듈은 ① 개념 설명 → ② 실제 코드 읽기 → ③ **직접 돌리는 실험** → ④ 손으로 푸는 연습문제 로 구성됩니다.
3. 실험은 **`experiments/`** 폴더(돌아가는 cargo 프로젝트)에서 합니다. 큰 예제는 `reference/examples/` 와 클론한 상위 저장소에서요.
4. **`PROGRESS.md`** 로 진도를 체크하세요. 막히면 **`GLOSSARY.md`**(용어)·**`CHEATSHEET.md`**(API 빠른참조)를 옆에 두고.

> 학습 원칙: **읽기만 하지 말고 무조건 돌려보기.** vISA는 매핑 타입이 머리로만 이해가 안 됩니다. `--backend typecheck` 로 "이 매핑이 합법인가"를 즉시 확인하고, `--backend simulation` 으로 "결과가 맞나"를 봅니다. 손이 먼저입니다.

---

## 3. 학습 경로 (11개 모듈)

| # | 모듈 | 한 줄 | 다루는 책 챕터 |
|---|---|---|---|
| 01 | **큰 그림과 전체 구조** | vISA가 왜/무엇, 하드웨어 계층, 파이프라인 한눈에 | introduction, quick-start(개념) |
| 02 | **매핑 & 텐서** (제일 중요) | `axes![]`·`m![]`(`/ % # , =`), 공간/시간 차원, 텐서 의미 | mapping-tensors/* |
| 03 | **원소 단위 연산 & 파이프라인 기초** | constant_add·elementwise_mul, Fetch/Collect/Vector/Commit, sub+VRF | quick-start, vector 기초 |
| 04 | **텐서 축약(Contraction)** | dot/gemv/gemm, Contraction 엔진, TRF, Switch 브로드캐스트 | quick-start, contraction-engine(intro) |
| 05 | **텐서 옮기기(Moving)** | 메모리 계층, Sequencer, Fetch/Commit/DMA 엔진, 뱅크 충돌 | moving-tensors/* |
| 06 | **연산 엔진 I — 분배 & 축약** | Switch 토폴로지, Collect, 레지스터파일, Contraction 내부(Outer/Packet/Time/Lane, 2D conv) | computing-tensors(분배+contraction) |
| 07 | **연산 엔진 II — Vector/Cast/Transpose** | Vector 엔진(reduce·VCG·op셋), softmax/layernorm 조립, Cast, Transpose | computing-tensors(vector+cast+transpose) |
| 08 | **스케줄러 & 동시성** | main/sub/tdma/pdma, 해저드(RAW/WAR/WAW), 이중 버퍼링, 주소 충돌 | scheduler |
| 09 | **타일링 & 분할 전략** | 시간/공간 분할, split-K, chip/cluster reduce(손코딩 ReduceScatter) | kernel-examples(tiling~reduce) |
| 10 | **실전 사례** | 트랜스포머 전 과정(attention/softmax/RoPE/KV/MLP/norm/lm_head), MoE, MNIST | kernel-examples(transformer,MoE) + 예제 |
| 11 | **마무리 실습과 숙달** | 직접 커널 짜기, 비공개 컴파일러/EDF 현실, Schedule Viewer, DeltaNet 연결, 다음 단계 | 종합 |

예상 소요: 빠르게 훑으면 01~04로 **하루 안에 "커널을 읽고 돌릴 수 있음"**, 05~08로 **엔진을 이해**, 09~11로 **직접 설계**. 깊게 가면 모듈당 반나절~하루.

---

## 4. 폴더 구조

```
vISA/
├── README.md              ← 지금 이 문서 (시작점)
├── 00_SETUP.md            ← 환경 구축 + 첫 커널
├── GLOSSARY.md            ← 용어집
├── CHEATSHEET.md          ← API/명령어 빠른참조
├── PROGRESS.md            ← 학습 체크리스트
├── curriculum/            ← 01~11 모듈 (개념+실험+연습)
├── experiments/           ← 돌아가는 cargo 프로젝트 (실험장)
└── reference/             ← 퓨리오사 원본 오프라인 복사본
    ├── book/              ← 공식 책 (docs/src 전체)
    └── examples/          ← base-template + furiosa-opt-examples 소스
```

---

## 5. 지금 당장 한 줄로 시작

```bash
# 1) 환경부터
open 00_SETUP.md     # (또는 그냥 열어서 따라하기)

# 2) 설치 끝났으면 첫 커널 (NPU 없이 시뮬레이션)
cd experiments && cargo furiosa-opt run --release --bin constant_add
```

준비되셨으면 **`00_SETUP.md`** 로 갑니다.
