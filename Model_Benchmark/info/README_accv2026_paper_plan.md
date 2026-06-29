# ACCV 2026 논문 전략 — 결정권자 분석 + furiosa-opt 기반 OPTIMIZATION 주제·7/5 전투계획

> 작성일: 2026-06-29. ACCV 2026(<https://accv2026.org/>)에 낼 논문의 (1) 심사·채택 결정권자
> 성향을 웹으로 조사하고, (2) 그 성향에 맞춰 RNGD NPU + furiosa-opt(공개 virtual ISA,
> <https://github.com/furiosa-ai/furiosa-opt> · 책 <https://developer.furiosa.ai/furiosa-opt/book/>)로
> 풀 수 있는 **최적화(Optimization) 주제와 실험**을 정한 기록입니다. 기술 근거는 이 레포의
> [info/README_virtual_isa.md], [info/README_vision_compile.md], [info/README_op_support.md]이고,
> 핵심 주장은 실제 코드·바이너리로 검증했습니다(아래 §검증).

---

## 0. 한 줄 결론

- **타깃 확정: ACCV 2026 본 트랙. 등록 7월 3일, 제출 7월 5일(오늘 6/29 기준 6일).** 워크숍 폴백은
  버립니다.
- **추천 주제(OPTIMIZATION): "Silent Precision Collapse → Per-Channel Scale Optimization".**
  학습된 CNN이 감소정밀도 NPU에서 top-1 ~0%로 붕괴하는 현상을 진단하고, **정확도 손실을
  최소화하는 per-channel 스케일·클립을 최적화**해서(라벨·재학습 없이) ImageNet 정확도를 되살리는
  논문입니다. "Optimization Methods"가 ACCV 정식 트랙이고, 바로 그 트랙의 nuLSQ(ACCV'24, 양자화
  step-size 최적화)가 채택 전례라 스코프가 안전합니다.
- **무엇이 보장되나(검증 반영):** 붕괴 특성화 + kurtosis 진단 + 충실한 호스트 정밀도 surrogate +
  per-channel int8 스케일·클립 **최적화의 복구를 *시뮬레이션*으로 입증** — 전부 기존 코드
  (`vision_models/classify.py`, `run_edf.py`) + 호스트 코드로 **vISA 없이** 됩니다. **실제 칩 위
  복구**는 ① furiosa.torch 가중치 접기(빠르나 bf16 스케일불변이라 불확실) ② vISA 명시 int8 Cast
  (정공법이나 cold-start) 중 하나로, **Day-1 게이트**가 판정합니다(§5). 어느 쪽이든 논문은 성립.
- 배경·지식: 주소 활용 경위와 필요 지식은 [info/README_accv2026_background.md] 참고.
- **누가 좋아하나:** Jiwen Lu(최적화+양자화+하드웨어 공동설계), Hyunjung Shim(진단 먼저→고치기)의
  취향 직격. 단 두 가지 정직성 보정이 필수입니다(§3 메커니즘, §4 효율 축).

---

## 1. 학회와 결정권자

### 1-1. 기본 정보·마감 (출처: <https://accv2026.org/>, /submissions/)

| 항목 | 내용 |
|---|---|
| 정식명·장소 | 18th ACCV, 일본 오사카, 2026-12-14~18 |
| 등급·채택률 | CORE **B**, **~32%**(2024 약 32%, 2022 33.4%) |
| 심사 | **이중맹검**, OpenReview(`afcv.org/ACCV/2026`), 논문당 ~3 리뷰어 + AC, 1쪽 익명 rebuttal |
| 형식 | 본문 **14쪽** + 참고문헌 무제한, **Springer LNCS 필수**, 부록 허용 |

| 마감(23:59 GMT) | 날짜 |
|---|---|
| **본 트랙 등록** | **2026-07-03** |
| **본 트랙 제출** | **2026-07-05** |
| 부록 제출 | 2026-07-08 (← 추가 실험은 여기로 넣을 수 있음) |
| 리뷰/Rebuttal/결과 | 08-26 / 09-02 / 09-20 |

### 1-2. accept 결정권자 — 프로그램 체어 4인 (출처: /organizers/ + 각자 Scholar·홈페이지, §출처)

| 체어 (소속) | 연구 색깔 | 우리 주제와의 궁합 |
|---|---|---|
| **Jiwen Lu** (Tsinghua, h-index 114) | 효율 딥러닝의 거물: 양자화·이진망·동적연산·**알고리즘-하드웨어 공동설계**·엣지배포 | ⭐⭐⭐ 챔피언감. 최적화+측정 기반 trade-off를 보상(FLOP-only 싫어함). "순수 커널/시스템"은 스코프 밖이라 명시 → CV 지표 전면 필수 |
| **Hyunjung Shim** (KAIST, h-index 29) | 생성모델 + **양자화(DGQ, ICLR'25)**, 데이터셋 distillation | ⭐⭐⭐ 가장 강한 우군. DGQ가 *outlier 먼저 진단→고치기* — 우리 구조와 DNA 동일 |
| **Norimichi Ukita** (TTI) | 초해상도/복원·포즈·diffusion, **알고리즘적 효율** | ⭐⭐ 중립~약우호. 잘 정의된 최적화 + ImageNet 지표를 앞세우면 OK. 얇은 CV 하드웨어 논문 싫어함 |
| **Miaomiao Liu** (ANU) | 3D·기하 우선(Hartley 계보), 엄밀성 | ⭐ 챔피언 기대 X, **반대만 안 하게**. 잘 정의된 문제·통제실험·재현성으로 엄밀하게 |

**일반 체어(톤):** Nagahara(deep-sensing 광학+센서+네트워크 공동설계), Nayar(계산카메라), Leal-Taixé
(NVIDIA, 추적·자율주행·실시간), Yao(인간/비디오). 센서·광학 공동설계와 실배포를 선호, 문제와
동떨어진 SW-only 효율 트릭엔 미지근.

### 1-3. ACCV에서 효율/최적화 논문이 통하는 조건

효율·양자화·최적화 논문은 ACCV에서 **통합니다 — "CV 기여 + 비전 과제 정확도 + 정확도-비용
trade-off"로 포장됐을 때만.** 전례: nuLSQ(양자화 step-size *최적화*, ACCV'24), ReLUify distillation
(ACCV'24), 경량 optical flow(ACCV'22). **스코프 밖:** 비전 기여 없는 순수 NPU 마이크로아키텍처·
컴파일러·커널 스케줄링·서빙 → MLSys/MICRO/ISCA. → **논문의 얼굴은 ImageNet 정확도, vISA/커널은
부록.**

---

## 2. 핵심 현상 — Silent Precision Collapse (검증된 사실)

[info/README_vision_compile.md] 한계 ③에서 이 머신으로 실측한 것:

> 학습된 ImageNet CNN(MobileNetV2)을 RNGD에 올리면 컴파일·실행은 되는데 **감소정밀도 로워링이
> 학습 가중치를 무너뜨려 모든 사진을 "window screen" 한 클래스로 오분류**합니다(CPU는 정답,
> NPU top-1 ≈ 0%). 원인은 가중치의 **heavy-tailed 분포**(depthwise conv kurtosis 11.6), 이를 끌
> **Python 옵션이 없습니다**(compiler_config에 정밀도/양자화 필드 0개).

**메커니즘 (furiosa-opt 책으로 검증·정정 — 중요).** Contraction 엔진은 **저정밀 operand를 받아
넓은 누적기로 누적**합니다(i4/i8→i32, f8/bf16→f32; Vector 엔진은 i32/f32만 처리 — [book
contraction-engine·vector-engine] 실측). 즉 **누적기는 좁지 않고, 줄어든 정밀도는 operand cast에
있습니다.** 따라서:
> 닫힌 furiosa.torch 경로의 붕괴 = bf16-operand 캐스트 오차가 *학습된(heavy-tailed·악조건)* 망의
> 깊은 누적에서 상쇄(cancellation)로 증폭된 것(랜덤·정조건 망은 1e-11로 멀쩡).

**핵심 함의(정직):** 부동소수 bf16 스케일은 상대정밀도가 **스케일 불변**이라, 단순 per-channel
float 재스케일만으론 못 고칩니다. 진짜 복구 레버는 **(a) 제어된 int8 표현(per-channel 스케일이
실제 의미를 갖는 고정소수점) + (b) outlier clipping**이고, 이를 가능케 하는 게 furiosa-opt의
**명시적 Cast 엔진 + vector_fxp/clip**입니다. 어느 성분이 지배적인지는 §4 **E5**(operand-cast vs
누적 오차 분해)·**E6**(bf16 vs i8 모드)으로 가립니다. 이 정정을 빼면 Shim이 메커니즘에서 바로 찌릅니다.

---

## 3. 추천 주제 (1순위, OPTIMIZATION)

**제목(가안):** *Silent Precision Collapse: Per-Channel Scale Optimization to Recover Trained
CNNs on a Closed-Compiler Reduced-Precision NPU*

**한 줄 주장:** 학습된 CNN이 감소정밀도 NPU에서 top-1 ~0%로 붕괴하는 원인은 heavy-tailed
가중치이며, 하드웨어 정밀도 모델 하에서 **per-channel 고정소수점(int8) 스케일·클립을 최적화**하면
라벨·재학습 없이 ImageNet 정확도를 거의 FP32까지 되살립니다.

**최적화 문제(이게 "Optimization Methods" 트랙인 이유):**
- 변수: 층별 per-channel 이전 스케일 d, 클립 임계 t (8-tile 규칙에 따라 8채널 묶음 단위 상수).
- 제약: (C1) d를 인접 affine(BN·bias)에 접어 넣어 **FP32에서 logit 불변**(DFQ/SmoothQuant식
  equalization — 새 정리 아님, 정직하게 인정), (C2) 하드웨어: 8-tile 정렬·dtype∈{bf16,i8}·
  bf16 표현 가능 범위.
- 목적: 보정 모델 f_NPU(누적기 명시) 하에서 ImageNet 손실(또는 FP32 logit과의 KL surrogate) 최소화.
- 해법: **kurtosis로 warm-start**(11.6 같은 heavy-tail 채널에 더 강한 이전)한 closed-form →
  호스트 surrogate 위 coordinate descent → 칩에서 검증.

**진짜 CV 기여 3가지:**
1. **특성화:** 일화적 5장 실패를 ImageNet **top-1/top-5 붕괴(실제 칩, 다중 아키텍처)** 로 정량화 +
   "랜덤가중치는 1e-11로 통과, 학습가중치만 붕괴"라는 *거짓 양성 점검 함정* 경고.
2. **진단(Shim식):** per-layer kurtosis·dynamic-range가 붕괴를 **예측**(학습 불필요·전이 가능한
   임계값, AUC). 왜 LLM의 FP8/MXFP4는 살아남고(per-block scale) 비전 conv는 안 되는지 설명.
3. **최적화 복구:** kurtosis-guided 스케일·클립 최적화로 near-FP32 복구. 새로움 = **양자화기/정밀도
   손잡이가 전혀 없는 닫힌 컴파일러 + 범위 미통제 *부동소수 누적* 이라는, 기존 PTQ가 안 다룬 영역**
   에서 FP32-foldable-only 제약으로 푸는 점 + kurtosis-α가 SmoothQuant 고정 α를 이긴다는 ablation.

**furiosa-opt(vISA)의 자리(정직).** *헤드라인 복구 결과는 vISA 없이* 고수준 `furiosa.torch` 경로 +
호스트 최적화로 냅니다. furiosa-opt는 ① **하드웨어 누적·정밀도 모델을 충실히 세울 1차 사료**(공개
ISA 매뉴얼)이고 ② **명시적 Cast/고정소수점으로 누적을 직접 통제하는 upside·후속**(§4 E5 GEMM-sim,
§6 온칩)입니다. "vISA 손코딩 커널이 코어"라고 과장하지 않습니다.

---

## 4. 실험 설계

데이터: ImageNet-1k val. **sweep은 5k stratified, 최종 운영점은 full 50k**(추론 ~6.7ms/img라 수 분).
백엔드: **칩**=실제 RNGD(npu3), **host**=numpy/torch 수치, **vISA-sim**=furiosa-opt simulation(upside).

| # | 실험 | 지표 | 백엔드 | 베이스라인 | 상태 |
|---|---|---|---|---|---|
| E1 | 붕괴 특성화(MobileNetV2·EfficientNet-B0) | top-1/top-5 + per-layer rel-L2 | 칩 vs FP32 | FP32(71.9%) vs NPU(~0%) | **보장** |
| E2 | 누적기 surrogate Q_npu 충실도 | sim top-1/argmax가 칩과 일치(불일치율) + **held-out·2nd-model 검증** | host vs 칩 | 칩(정답) | **보장** |
| E3 | kurtosis 진단·예측기 | kurtosis·range ↔ rel오차(Spearman, 붕괴층 AUC) | host | — | **보장** |
| E4 | **복구 + 비용 곡선 + ablation** | top-1 vs **calibration budget**; clip-only/scale-only/both; kurtosis-α vs SmoothQuant α=0.5 vs uniform; calib 1/8/32/128 | 칩(복구) + host(sweep) | 아래 | **보장** |
| E5 | operand-cast vs **accumulator** 오차 분해(메커니즘 증명) | 두 오차원 기여 분리 | host Q_npu (+ vISA-sim GEMM은 upside) | 무보정 | **보장(host)** |
| E6 | **정밀도 모드 축**(진짜 bits 축) | top-1 @ bf16-operand vs i8-operand | 칩 (+ vISA-sim i8/bf16) | — | 보장(칩) |
| E7 | 일반화: Q_npu의 만티사·누적폭 sweep | "한 칩 버그"가 아니라 FP-누적 일반현상 | host | — | 보장 |
| E8 | (upside) 온칩 vISA 명시-정밀 GEMM 복구 | 복구 top-1을 vISA로 | vISA-sim → 칩 | furiosa.torch 붕괴 | **후속/미래연구** |

**베이스라인:** FP32(상한) · 무보정 붕괴(하한) · per-tensor 단일 스케일 · **SmoothQuant α=0.5** ·
**AWQ식 salient-channel** · clip-only · nuLSQ(전례로 인용, 닫힌 컴파일러라 직접 적용 불가 = 우리
새로움의 근거).

**핵심 절약(검증됨):** EDF는 가중치 독립(랜덤·학습 가중치 EDF 바이트 동일, 가중치는 fp32 런타임
입력 — [README_vision_compile.md]). 따라서 **접어 넣은 가중치를 컴파일된 프로그램 하나에 끼워
넣어** 설정마다 수백 초 재컴파일 없이 Pareto를 돌립니다(`--reuse-edf`).

**정직성 보정 2개(반박 차단):**
- **"latency Pareto"라고 부르지 말 것.** 가중치 접기 복구는 연산량·지연이 **불변**입니다. 축 이름을
  **"정확도 복구 vs calibration 예산(방법의 데이터/연산 효율)"** 로. 진짜 bits/효율 축은 E6의
  **bf16 vs i8 정밀도 모드 sweep**으로 따로 만듭니다(Lu의 "측정 Pareto" 요구 충족).
- **circular surrogate 금지.** Q_npu를 한 split에 calibrate하고 **held-out split + 다른 모델**에서
  예측↔실측 순위상관을 검증(Shim의 식별성 지적 차단).

---

## 5. 진도 계획 — 날짜 무관 마일스톤 (M0→M6) + 7/5 매핑

> **핵심: 날짜가 아니라 마일스톤으로 굴립니다.** 각 마일스톤은 독립적인 "진도"이고 논문 버전이
> 단조 증가합니다 — 어느 시점에 마감이 오든 *그때까지 도달한 최고 마일스톤*을 제출하면 됩니다.
> **M0–M4는 vISA 없이 완결되는 본 트랙 논문**, M5부터가 칩 복구·강한 버전. 날짜에 안 묶이니
> 7/5을 못 맞춰도(또는 더 밀어붙여도) 진도는 끊기지 않습니다.

| 마일스톤 | 산출물 | 진입 조건 → 종료(다음으로 넘어가는) 조건 | vISA |
|---|---|---|---|
| **M0 평가 하네스** | ImageNet val 로더 + synset→라벨 매핑 | — → **FP32 host top-1 == torchvision 71.9%**(라벨순서 버그 차단) | X |
| **M1 붕괴 정량화** | 칩 top-1 붕괴 곡선(MobileNetV2·EfficientNet-B0) | M0 → 칩 top-1≈0% + per-layer rel-L2 곡선 확보 | X |
| **M2 진단** | kurtosis·range → 붕괴층 예측기(AUC) | M1 → 예측기가 붕괴층을 유의하게 분리 | X |
| **M3 충실 surrogate Q_npu** | bf16-operand 캐스트 + f32 누적 모델, 칩 logit에 calibrate | M1 → **held-out·2nd-model**에서 sim↔칩 top-1/argmax 낮은 불일치율 | X |
| **M4 최적화 복구(시뮬)** | per-channel int8 스케일·클립 최적화(kurtosis warm-start→coordinate descent) + ablation + calibration-budget 곡선 | M2·M3 → 시뮬 near-FP32 복구 + (descent가 closed-form 이김 or "closed-form 충분") | X |
| **M5 칩 복구** | (a) furiosa.torch 가중치 접기 재실행 **또는** (b) vISA 명시 int8 Cast로 온칩 | M4 → 실제 칩 top-1 복구 수치(부분이라도) | (b)면 ○ |
| **M6 일반화·확장** | Q_npu 만티사/모드 sweep(일반현상) · 혼합정밀도 비트할당 CAMP(MCKP) · 3rd 아키텍처 · vISA 온칩 e2e | M5 → 강한 버전(IJCV/CVPR) | ○ |

**= 어디서 멈춰도 제출 가능:** M0–M4 도달 = 완결된 본 트랙 논문(시뮬 복구). M5 도달 = 칩 복구로
Lu 풀 챔피언 전환. M6 = 강한 venue 확장.

### 5-1. 7/5을 노린다면 (위 마일스톤의 날짜 매핑)

> 원칙: **furiosa-opt(vISA)는 임계경로에서 뺀다.** 코어(M0–M4)는 전부 `vision_models/classify.py`
> (furiosa.torch) + 호스트 코드로 돈다. rustup/cargo cold-start는 스파인(≈M4) 100% 끝난 뒤 upside로만.

| 날짜 | 할 일 | 그날의 게이트 |
|---|---|---|
| **6/29(오늘)** | ImageNet val 다운로드 시작(유일한 긴 wall-clock). val 로더 + synset→라벨 매핑. **FP32 host top-1 == torchvision 71.9% 확인** | 라벨 매핑이 71.9% 안 나오면 다른 거 손대기 전에 고치기 |
| **6/30 (Day1, ★GO/NO-GO)** | `vision_models/classify.py`로 칩 붕괴 측정(E1). **가장 단순한 개입**(per-channel clip + closed-form SmoothQuant/AWQ scale)을 가중치에 접어 넣고 EDF 가중치 스왑으로 npu3에서 500장 복구 수치 읽기 | **복구가 유의미?** GO→A안(온칩 복구 논문) / ~0%면 즉시 B안(누적기-국소화 발견 + surrogate 복구)으로 서사 전환. **어느 쪽이든 논문 성립** |
| **7/1 (Day2)** | Q_npu 누적기 surrogate + 충실도(held-out·2nd-model, E2). kurtosis 진단·AUC(E3). EfficientNet-B0 칩 붕괴(다중 아키텍처) | EfficientNet-B0 컴파일 확인(안 되면 MobileNetV3-L/RegNetY로 교체, 약속 전 컴파일 먼저) |
| **7/2 (Day3)** | coordinate-descent 솔버. 복구 + **calibration-budget 곡선** + ablation(clip/scale/both, kurtosis-α vs 0.5 vs uniform). bf16 vs i8 정밀도 모드 sweep(E6) | **14쪽 commit 게이트**: 복구 크기 OK + (coordinate descent가 closed-form을 이기거나 "closed-form이면 충분"으로 정직 강등) |
| **7/3 (Day4, ★등록 마감)** | **오늘 등록.** operand-cast vs accumulator 분해(E5, 메커니즘 증명). 최종 운영점 full 50k. intro/method/그림 쓰기 | 등록 완료. 메커니즘 분해가 누적기 우세를 보이면 복구 상한 정직 보고 |
| **7/4 (Day5)** | results/related-work/ablation 작성. LNCS 포맷·익명화. **(upside)** rustup nightly-2026-05-01 + `cargo binstall cargo-furiosa-opt` + GEMM smoke → 되면 vISA-sim GEMM 분해 그림 추가 | GEMM smoke 1일 내 안 되면 vISA upside 포기(논문 안 막힘) |
| **7/5 (Day6, 제출)** | 마무리·교정·부록. **23:59 기다리지 말고 일찍 제출**(OpenReview 사고 버퍼). 남은 실험은 7/8 부록으로 | 제출 완료 |

---

## 6. 리스크 · 폴백 (red-team)

**노력으로 못 막는 블로커(밤새도 해결 안 됨):** ① 닫힌 누적기에서 복구 *크기*(물리라 강제 불가) ②
닫힌 컴파일러가 접어 넣은 스케일을 재정규화로 무효화할 가능성 ③ 새 vISA conv 커널이 닫힌
lowering을 통과할지 + TODO npu-dispatch.

**무조건 7/5에 나오는 최소 완성 논문(폴백, vISA 0):** "닫힌 컴파일러 감소정밀도 NPU 위 학습 CNN의
조용한 붕괴 — 실제 칩 위 ImageNet top-1 붕괴(다중 아키텍처) 특성화 + kurtosis 근본원인 진단 +
실측에 맞춘 누적 surrogate(Q_npu) + per-channel 스케일/클립 최적화로 **surrogate에서 복구 입증**
(Gate A 통과 시 실제 칩에서도)." → 이것만으로 완결된 CV 최적화 논문. 칩 복구·vISA는 전부 upside.

**descope:** MobileNetV1 삭제(torchvision에 없음). 온칩 vISA conv 커널 = 미래연구 한 문장.
furiosa-opt를 임계경로에서 제거. vISA-GEMM-sim은 스파인 끝난 뒤만.

**리뷰어 공격 → 방어:**
- "효율 Pareto가 공허" → 축을 calibration-budget로 정명 + E6 bf16/i8 bits sweep으로 진짜 효율 축.
- "SmoothQuant 증분" → 새로움 = 양자화기·손잡이 없는 *부동소수 누적* 영역 + FP32-foldable-only 제약 +
  kurtosis-α가 고정 α 이김(ablation) + 측정-fit Q_npu. equalization/invertibility는 DFQ 것이라 인정.
- "circular surrogate" → held-out + 2nd-model 검증.
- "한 칩 버그" → Q_npu 만티사·누적폭 sweep(E7) = 일반 현상.
- "얇은 CV/시스템" → ImageNet top-1 헤드라인, vISA 부록. 이중맹검 → "a closed-compiler
  reduced-precision NPU with no precision API"로 익명화.

**후속(강한 venue): 혼합정밀도 비트 할당(CAMP).** 층별 정밀도(i4/i8/f8/bf16)를 MCKP로 최적화해
정확도-비트 Pareto를 만드는 각도(차점안, Lu 적합도 최고). furiosa-opt의 정밀도 엔진을 정면으로
쓰는 furiosa-opt-native 확장 → ACCV2027/CVPR/IJCV.

---

## 검증 (load-bearing 주장, 2026-06-29 실측)

- ✅ `vision_models/classify.py` 존재 — IMAGENET1K_V1 학습가중치로 mobilenet_v2/efficientnet_b0를
  `rngd:N`(NPU)·CPU 양쪽 실행, top-1/top-5 비교, `--reuse-edf` 지원. (`run_edf.py`도 존재, "window
  screen" 붕괴 기록 보유.) **단 문서의 경로는 `rngd-npu/classify.py`가 아니라 `rngd-npu/vision_models/classify.py`.**
- ✅ furiosa-opt **cold** — `cargo-furiosa-opt` 없음, nightly-2026-05-01 미설치. vISA 작업은 진짜
  cold-start(그래서 임계경로에서 제외).
- ✅ EDF 가중치 독립(랜덤·학습 EDF 바이트 동일) → 재컴파일 없이 가중치 스왑 가능 — Pareto sweep
  실현성의 근거. (출처 [README_vision_compile.md])
- 수치 인용: top-1 71.9%(torchvision MobileNet_V2), matmul ~0.23%([README_op_support.md] §3),
  kurtosis 11.6·"window screen"·정밀도 손잡이 0개([README_vision_compile.md] 한계③), 6.7ms warm(동).

---

## 출처

**ACCV:** <https://accv2026.org/> · /organizers/ · /submissions/ · /submissions/author-guidelines/ ·
OpenReview <https://openreview.net/group?id=afcv.org/ACCV/2026/Conference> ·
CORE <https://portal.core.edu.au/conf-ranks/167/> · ACCV2024 <https://openaccess.thecvf.com/ACCV2024> ·
nuLSQ <https://openaccess.thecvf.com/content/ACCV2024/papers/Gongyo_Learning_Non-Uniform_Step_Sizes_for_Neural_Network_Quantization_ACCV_2024_paper.pdf>

**프로그램 체어:** Lu <https://scholar.google.com/citations?user=TN8uDQoAAAAJ> ·
<https://ivg.au.tsinghua.edu.cn/jiwen_lu/biography.html> | Shim
<https://scholar.google.com/citations?user=KB5XZGIAAAAJ> · <https://kaist-cvml.github.io/> ·
DGQ <https://arxiv.org/html/2501.04304> | Ukita <https://www.toyota-ti.ac.jp/Lab/Denshi/iim/ukita/> ·
<https://scholar.google.com/citations?user=Tgbsbs8AAAAJ> | Liu <https://users.cecs.anu.edu.au/~mliu/>

**선행연구(베이스라인·포지셔닝):** SmoothQuant(arXiv 2211.10438) · AWQ(2306.00978) · DFQ/Data-Free
Quant(ICCV'19) · OCS · Robust Quantization(NeurIPS'20).

**furiosa-opt:** repo <https://github.com/furiosa-ai/furiosa-opt> · 책
<https://developer.furiosa.ai/furiosa-opt/book/> · 내부분석 [info/README_virtual_isa.md] ·
비전컴파일 [info/README_vision_compile.md] · op지원 [info/README_op_support.md]
