# ACCV 2026 논문 — 착상 경위 + 필요 지식

이 문서는 ACCV 2026 논문([info/README_accv2026_paper_plan.md])의 배경 자료로, furiosa-opt
(<https://github.com/furiosa-ai/furiosa-opt> · <https://developer.furiosa.ai/furiosa-opt/book/>)에서
무엇을 보고 이 주제를 떠올려 실험으로 옮겼는지와, 논문을 쓰는 데 필요한 지식을 정리합니다.

---

## 1. 주소에서 무엇을 보고 → 어떻게 주제가 나왔나

### 1-1. 본 것 (사실, 책·repo·바이너리로 검증)

두 주소를 처음엔 "이걸로 뭘 할 수 있나"로 통째 분석했고([info/README_virtual_isa.md]), 그때 본
세 가지가 이번 주제의 씨앗이 됐습니다.

1. **vISA는 RNGD의 정밀도를 *명시적으로* 노출합니다.** 책의 Contraction 엔진은 저정밀 operand
   (i4/i8/f8/bf16)를 받아 **넓은 누적기로 누적**합니다(i4/i8→**i32**, f8/bf16→**f32**). 별도의
   **Cast 엔진**으로 정밀도 변환을 직접 하고, **Vector 엔진**에 `vector_fxp`(고정소수점)·
   `vector_clip`(클램프)이 있습니다(Vector는 i32/f32만 처리, widening은 Contraction이 담당).
   → 한마디로 **"하드웨어는 정밀도를 사람이 통제할 수 있게 설계돼 있다."**
2. **그런데 우리가 실제로 쓰는 고수준 경로(`furiosa.torch`)엔 그 손잡이가 0개입니다.** 직접
   실측해 보니([info/README_vision_compile.md] 한계 ③) 학습된 CNN이 감소정밀도 로워링으로
   **top-1 ~0%로 붕괴**(모든 사진을 "window screen"으로)하는데, `compiler_config`에 정밀도/양자화
   필드가 하나도 없어 끌 방법이 없었습니다.
3. **simulation 백엔드가 있습니다.** NPU 없이 호스트에서 커널을 수치로 돌려 검증할 수 있어,
   실험을 칩 점유 없이 빠르게 반복할 수 있습니다.

### 1-2. 그래서 떠올린 주제

위 1·2를 나란히 놓으면 **간극**이 보입니다 — *하드웨어는 정밀도를 제어할 수 있는데(주소에서 확인),
고수준 경로는 그 제어를 막아 학습 모델을 망가뜨린다(우리 실측).* 이 간극을 메우는 것이 주제입니다:

> "감소정밀도 NPU에서 학습 CNN이 왜 붕괴하는지 진단하고, **하드웨어 정밀도 모델 하에서 per-channel
> 고정소수점(int8) 스케일·클립을 최적화**해 ImageNet 정확도를 되살린다."

메커니즘 상세(누적은 넓고 손실은 operand cast → 단순 float 재스케일은 무효, 제어된 int8 +
clipping이 레버, 도구는 vISA Cast/`vector_fxp`)는 [paper_plan §2]에 있어 여기선 생략합니다.

### 1-3. 어떻게 활용했나 (주소의 각 요소 → 논문·실험)

| 주소에서 본 것 (출처) | 그것이 알려준 것 | 논문·실험에서의 활용 |
|---|---|---|
| Contraction: i4/i8/f8/bf16 → i32/f32 누적 (book contraction-engine) | 누적기는 넓고, 손실은 **operand cast**에 있다 | 붕괴 메커니즘을 operand-precision으로 특정 → **Q_npu surrogate**를 충실히 구성(M3), E5(cast vs 누적 분해)·E6(bf16 vs i8) 설계 |
| Cast 엔진 + `vector_fxp`/`vector_clip` (book cast/vector-engine) | int8 캐스트·per-channel 스케일·클립을 **명시적으로** 줄 수 있다 | 복구 방법의 실체(제어된 int8 + 클립) + **온칩 명시-정밀 복구(M5b/upside)**의 구현 도구 |
| 2D conv = einsum `$(H+Fh)$(W+Fw)K,FhFwKC->HWC`, Stream-Adapter shift-reuse (book 2d-convolution) | conv를 contraction으로 손수 짤 청사진 | upside에서 conv를 명시-정밀 GEMM/contraction으로 재구현하는 설계 근거 |
| simulation 백엔드(호스트 수치, NPU 불필요) (repo README, book intro) | NPU 점유 없이 수치 검증 가능 | **M4 복구 최적화를 시뮬레이션으로 입증** = 6일/날짜무관 실현성의 핵심 |
| op·model_type 게이트 우회, 닫힌 컴파일러(prebuilt) (repo, [virtual-isa] §2·§7) | 고수준이 막는 걸 저수준에서 표현 가능 / 단 컴파일러는 블랙박스 | "고수준 경로엔 정밀도 손잡이 0"이라는 문제의식의 근거 + vISA를 임계경로에서 빼는 이유(cold-start·블랙박스 리스크) |

> 정직성: **헤드라인(시뮬 복구·붕괴·진단)은 vISA 없이** 고수준 `furiosa.torch` 경로 + 호스트
> 코드로 냅니다. furiosa-opt는 ① 정밀도 모델을 세울 **1차 사료**이자 ② 온칩 명시-정밀 복구의
> **upside 도구**입니다. "vISA 손코딩이 코어"라고 과장하지 않습니다.

---

## 2. 필요한 지식 정리 (study guide)

이 논문을 직접 구현·집필하려면 아래 7개 영역이 필요합니다. 각 항목은 *무엇을 / 왜 / 핵심 레퍼런스*
순입니다.

### A. 저정밀 추론·양자화 기초
- **무엇:** PTQ(post-training quantization) vs QAT, fake-quant, per-tensor vs **per-channel** 스케일,
  대칭/비대칭, **outlier(이상치) 채널 문제**, calibration. 우리 방법의 직계 조상.
- **왜:** 우리 복구는 "양자화기를 못 학습시키는 닫힌 컴파일러"에서 per-channel 스케일·클립을
  *최적화*하는 것. 선행연구와의 차이(새로움)를 정확히 말하려면 필수.
- **레퍼런스:** SmoothQuant(<https://arxiv.org/abs/2211.10438>, 활성-가중치 difficulty migration) ·
  AWQ(<https://arxiv.org/abs/2306.00978>, salient-channel 보호) · DFQ/Data-Free Quant
  (<https://arxiv.org/abs/1906.04721>, weight equalization·우리 invertibility의 원조) ·
  LSQ(<https://arxiv.org/abs/1902.08153>) · **nuLSQ(ACCV'24, 우리 트랙·전례)** · Robust Quantization
  (NeurIPS'20, kurtosis 정규화).

### B. 부동소수·고정소수 산술
- **무엇:** fp32/bf16/tf32/fp8의 지수·만티사 구조, **상대정밀도가 스케일 불변**이라는 성질,
  round-to-nearest, **넓은 누적기(i32/f32) vs 좁은 operand**, 누적 시 상쇄(cancellation)·오차 전파.
- **왜:** "bf16 스케일은 왜 안 듣고 int8 스케일은 왜 듣는가"가 논문의 메커니즘 핵심. E5/E6의 해석이
  여기에 달림.
- **레퍼런스:** Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic*
  · Higham, *Accuracy and Stability of Numerical Algorithms*(상쇄·누적 오차) · bf16/tf32 사양(NVIDIA/
  Google 문서).

### C. RNGD/TCP 하드웨어 정밀도 모델
- **무엇:** Contraction(Broadcast·Multiply·Reduce, i4/i8→i32·f8/bf16→f32), Cast 엔진,
  Vector 엔진(`vector_fxp`·`vector_clip`·intra/inter-slice reduce, i32/f32 처리), SRAM 계층,
  8-lane Contraction cap·8-tile 정렬, ~0.23% matmul 감소정밀도.
- **왜:** Q_npu surrogate를 *추측이 아니라 사양에 맞춰* 세우고, 온칩 복구(M5b)를 설계하려면.
- **레퍼런스:** furiosa-opt book — contraction-engine / cast-engine / vector-engine /
  2d-convolution / scheduling(<https://developer.furiosa.ai/furiosa-opt/book/>) · 우리
  [info/README_op_support.md](§3 0.23%, §6 dtype, 8-tile) · [info/README_virtual_isa.md](§3 하드웨어 모델).

### D. furiosa-opt vISA 프로그래밍
- **무엇:** 커널=Rust 함수 + 타입스테이트 파이프라인, `axes![]`/`m![]` 매핑 대수, 수동 SRAM/축 배치,
  백엔드(typecheck/simulation/emulation/npu), 무엇이 공개/닫힘인지, `.bin`(pert-ipc)≠furiosa-llm `.edf`.
- **왜:** upside(온칩 명시-정밀 복구)를 구현하려면. cold-start(rustup nightly-2026-05-01 + cargo
  binstall + 닫힌 `.a`)·블랙박스 리스크를 알아야 임계경로에서 빼는 판단이 섬.
- **레퍼런스:** repo <https://github.com/furiosa-ai/furiosa-opt> · book quick-start/introduction ·
  [info/README_virtual_isa.md] 전체.

### E. furiosa.torch 컴파일·실행 경로
- **무엇:** `torch.export` → 분해(특히 **batch_norm 직접 분해**) → `CompileModule.from_exported` → EDF;
  **가중치는 EDF에 굽지 않고 fp32 런타임 입력**(그래서 EDF가 가중치 독립·재컴파일 없이 스왑 가능);
  `--reuse-edf`; 중간 풀링이 그래프를 쪼개는 한계.
- **왜:** M0–M5a 전부 이 경로 위에서 돈다. 가중치 스왑으로 Pareto를 싸게 도는 트릭의 근거.
- **레퍼런스:** [info/README_vision_compile.md](되는 방법·EDF 저장/재사용·한계) ·
  `Model_Benchmark/rngd-npu/vision_models/classify.py`·`rngd-npu/run_edf.py`(실제 코드).

### F. 최적화 기법
- **무엇:** closed-form 스케일(SmoothQuant식 difficulty migration), **coordinate descent**,
  KL/출력오차 surrogate 목적함수, 제약 최적화, (후속) **혼합정밀도 비트할당 = Multiple-Choice
  Knapsack(MCKP)**, 민감도(Hessian/kurtosis) 기반 랭킹.
- **왜:** "Optimization Methods" 트랙 논문이려면 목적·변수·제약·해법이 또렷해야 함. coordinate
  descent가 closed-form을 이긴다는 ablation이 새로움의 일부.
- **레퍼런스:** Boyd & Vandenberghe, *Convex Optimization*(좌표하강·제약) · HAWQ/HAWQ-V2(민감도 기반
  혼합정밀도) · Multiple-Choice Knapsack 표준 문헌.

### G. 평가·집필 실무
- **무엇:** ImageNet-1k val 프로토콜(synset→라벨 매핑, 전처리, top-1/top-5), 통계적 표본(5k vs 50k),
  공정 베이스라인·ablation, **이중맹검 익명화**("a closed-compiler reduced-precision NPU"로 서술),
  Springer LNCS 14쪽 포맷, OpenReview 제출.
- **왜:** "ImageNet top-1이 헤드라인"이라는 스코프 안전장치와 desk-reject(익명성·분량) 회피.
- **레퍼런스:** torchvision ImageNet 가중치/정확도(MobileNet_V2 71.9%) · ACCV 2026 author-guidelines
  (<https://accv2026.org/submissions/author-guidelines/>) · LNCS 템플릿 · [paper_plan §1].

---

## 출처
furiosa-opt book <https://developer.furiosa.ai/furiosa-opt/book/>(contraction-engine·cast-engine·
vector-engine·2d-convolution, 2026-06-29 WebFetch 실측) · repo
<https://github.com/furiosa-ai/furiosa-opt> · 내부: [info/README_virtual_isa.md] ·
[info/README_vision_compile.md] · [info/README_op_support.md] · [info/README_accv2026_paper_plan.md].
선행연구 URL은 위 본문에 인라인.
