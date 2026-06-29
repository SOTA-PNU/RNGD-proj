# ACCV 2026 논문 계획 — Silent Precision Collapse (OPTIMIZATION)

이 문서는 ACCV 2026에 낼 논문의 주제·전략·실험 계획 개요를 담은 계획서입니다. 자세한 실험 과정·논문 주제 설명·furiosa-opt
코드 해설은 `rngd-npu/ACCV/` 폴더에 옮겨 정리했습니다. 결정권자 분석 근거·필요 지식은
[info/README_accv2026_background.md]에 있습니다.

---

## 0. 결론

- **타깃:** ACCV 2026 본 트랙(등록 7/3, 제출 7/5). ACCV 정식 트랙 "Optimization Methods".
- **주제:** *Silent Precision Collapse → Per-Channel Scale Optimization.* 학습된 CNN이 감소정밀도
  NPU에서 top-1 ~0%로 붕괴하는 현상을 진단하고, **per-channel 고정소수점(int8) 스케일·클립을
  최적화**해(라벨·재학습 없이) ImageNet 정확도를 되살립니다. 전례 nuLSQ(ACCV'24, 양자화 step-size
  최적화)와 같은 장르라 스코프 안전.
- **무엇이 보장되나(vISA 없이):** 붕괴 특성화 + kurtosis 진단 + 호스트 정밀도 surrogate + 최적화
  복구를 **시뮬레이션으로 입증** — 전부 기존 코드(`vision_models/classify.py`, `run_edf.py`) + 호스트
  코드. **실제 칩 복구**는 ① furiosa.torch 가중치 접기(빠르나 불확실) ② vISA 명시 int8 Cast(정공법,
  cold-start) 중 하나로 **Day-1 게이트**가 판정(§5). 어느 쪽이든 논문 성립.
- **적합:** Jiwen Lu(최적화+양자화+하드웨어 공동설계), Hyunjung Shim(진단 먼저→고치기) 직격. 단
  정직성 보정 2개 필수(§4).

---

## 1. 결정권자·스코프 (왜 이 주제가 맞나)

논문 accept/reject는 **프로그램 체어 4인**이 결정합니다.

| 체어 | 색깔 | 우리 주제와의 궁합 |
|---|---|---|
| **Jiwen Lu** (Tsinghua) | 양자화·이진망·동적연산·**알고리즘-하드웨어 공동설계**·엣지배포 | ⭐⭐⭐ 챔피언감. 측정 기반 정확도-효율 trade-off 보상(FLOP-only 싫어함). "순수 커널/시스템"은 스코프 밖 → CV 지표 전면 |
| **Hyunjung Shim** (KAIST) | 생성모델 + **양자화(DGQ)**, distillation | ⭐⭐⭐ 가장 강한 우군. DGQ가 *outlier 먼저 진단→고치기* — 우리 구조와 동일 |
| **Norimichi Ukita** (TTI) | 초해상도·포즈·diffusion, 알고리즘적 효율 | ⭐⭐ 잘 정의된 최적화 + ImageNet 지표면 OK. 얇은 CV 하드웨어 논문 싫어함 |
| **Miaomiao Liu** (ANU) | 3D·기하 우선, 엄밀성 | ⭐ 반대만 안 하게 — 통제실험·재현성으로 엄밀하게 |

- **스코프 규칙:** 효율/최적화 논문은 **"비전 과제 정확도 + 정확도-비용 trade-off"로 포장돼야** 통함
  (전례 nuLSQ·ACCV'24). 순수 NPU/컴파일러/커널/서빙은 스코프 밖(MLSys/MICRO行). → **헤드라인은
  ImageNet 정확도, vISA/커널은 부록.**
- **마감:** 등록 7/3 · 제출 7/5 · 부록 7/8. **형식:** 14쪽 LNCS, 이중맹검, OpenReview.

---

## 2. 핵심 현상 + 메커니즘

[info/README_vision_compile.md] 한계 ③ 실측:

> 학습된 CNN(MobileNetV2)을 RNGD에 올리면 컴파일·실행은 되는데 **감소정밀도 로워링이 학습
> 가중치를 무너뜨려 모든 사진을 "window screen"으로 오분류**합니다(CPU 정답, NPU top-1 ≈ 0%).
> 원인은 가중치의 **heavy-tailed 분포**(depthwise conv kurtosis 11.6), 끌 **Python 옵션 없음**.

**메커니즘(furiosa-opt 책으로 검증).** Contraction 엔진은 저정밀 operand를 받아 **넓은 누적기로
누적**합니다(i4/i8→i32, f8/bf16→f32). 즉 **누적기는 좁지 않고, 손실은 operand cast에 있습니다.**
붕괴 = bf16-operand 캐스트 오차가 *학습된(악조건)* 망의 깊은 누적에서 상쇄로 증폭된 것(랜덤·정조건
망은 1e-11). **함의:** 부동소수 스케일은 상대정밀도가 **스케일 불변**이라 단순 float 재스케일론 못
고침 → 진짜 레버는 **(a) 제어된 int8 표현 + (b) outlier clipping**이고, 이를 가능케 하는 게
furiosa-opt의 **명시적 Cast 엔진 + vector_fxp/clip**. 어느 성분이 지배적인지는 §4 E5·E6으로 판정.

---

## 3. 주제·기여·최적화 문제

**제목(가안):** *Silent Precision Collapse: Per-Channel Scale Optimization to Recover Trained
CNNs on a Closed-Compiler Reduced-Precision NPU*

**한 줄 주장:** 학습된 CNN이 감소정밀도 NPU에서 top-1 ~0%로 붕괴하는 원인은 heavy-tailed
가중치이며, 하드웨어 정밀도 모델 하에서 **per-channel 고정소수점(int8) 스케일·클립을 최적화**하면
라벨·재학습 없이 ImageNet 정확도를 거의 FP32까지 되살립니다.

**최적화 문제(= "Optimization Methods" 트랙인 이유):**
- 변수: 층별 per-channel 이전 스케일 d, 클립 임계 t (8-tile 규칙 → 8채널 묶음 단위 상수).
- 제약: (C1) d를 인접 affine(BN·bias)에 접어 **FP32에서 logit 불변**(DFQ/SmoothQuant식 equalization
  — 새 정리 아님, 정직 인정), (C2) 하드웨어: 8-tile 정렬·dtype∈{bf16,i8}.
- 목적: 정밀도 모델 f_NPU 하에서 ImageNet 손실(또는 FP32 logit과의 KL surrogate) 최소화.
- 해법: **kurtosis warm-start**한 closed-form → 호스트 surrogate 위 coordinate descent → 칩 검증.

**진짜 CV 기여 3가지:**
1. **특성화:** 일화적 실패를 ImageNet top-1/top-5 붕괴(실제 칩, 다중 아키텍처)로 정량화 + "랜덤
   가중치는 1e-11로 통과, 학습가중치만 붕괴"라는 *거짓 양성 점검 함정* 경고.
2. **진단(Shim식):** per-layer kurtosis·dynamic-range가 붕괴를 **예측**(학습 불필요·전이 가능한
   임계값, AUC). 왜 LLM의 FP8/MXFP4는 살아남고(per-block scale) 비전 conv는 안 되는지 설명.
3. **최적화 복구:** kurtosis-guided 스케일·클립 최적화로 near-FP32 복구. 새로움 = **양자화기·정밀도
   손잡이가 없는 닫힌 컴파일러 + 범위 미통제 부동소수 누적**이라는 기존 PTQ가 안 다룬 영역 +
   kurtosis-α가 SmoothQuant 고정 α를 이긴다는 ablation.

> furiosa-opt(vISA)는 ① 정밀도 모델을 세울 1차 사료 ② 명시적 Cast로 온칩 복구하는 upside(§4 E8)일
> 뿐, **헤드라인 복구는 vISA 없이** 고수준 경로 + 호스트 최적화로 냅니다.

---

## 4. 실험 설계

데이터: ImageNet-1k val(sweep 5k stratified, 최종 운영점 full 50k). 백엔드: **칩**=RNGD(npu3),
**host**=numpy/torch, **vISA-sim**=furiosa-opt simulation(upside).

| # | 실험 | 지표 | 백엔드 | 상태 |
|---|---|---|---|---|
| E1 | 붕괴 특성화(MobileNetV2·EfficientNet-B0) | top-1/top-5 + per-layer rel-L2 | 칩 vs FP32 | 보장 |
| E2 | surrogate Q_npu 충실도(held-out·2nd-model 검증) | sim↔칩 top-1/argmax 불일치율 | host vs 칩 | 보장 |
| E3 | kurtosis 진단·예측기 | kurtosis·range ↔ rel오차(Spearman, 붕괴층 AUC) | host | 보장 |
| E4 | **복구 + 비용 곡선 + ablation** | top-1 vs calibration budget; clip/scale/both; kurtosis-α vs SmoothQuant α=0.5 | 칩 + host | 보장 |
| E5 | operand-cast vs accumulator 오차 분해(메커니즘 증명) | 두 오차원 기여 분리 | host Q_npu | 보장 |
| E6 | 정밀도 모드 축(진짜 bits 축) | top-1 @ bf16 vs i8 | 칩 | 보장 |
| E7 | 일반화: Q_npu 만티사·누적폭 sweep | "한 칩 버그" 아닌 FP-누적 일반현상 | host | 보장 |
| E8 | (upside) 온칩 vISA 명시-정밀 GEMM 복구 | 복구 top-1을 vISA로 | vISA-sim→칩 | 후속 |

**베이스라인:** FP32(상한) · 무보정 붕괴(하한) · per-tensor 단일 스케일 · SmoothQuant α=0.5 ·
AWQ salient-channel · clip-only · nuLSQ(전례 인용, 닫힌 컴파일러라 직접 적용 불가 = 우리 새로움 근거).

**실험 절약:** EDF는 가중치 독립(가중치=fp32 런타임 입력) → 접은 가중치를 컴파일된 프로그램 하나에
끼워(`--reuse-edf`) **재컴파일 없이** Pareto를 돌립니다.

**정직성 보정 2개(반박 차단):**
- **"latency Pareto" 금지** — 가중치 접기 복구는 지연 불변. 축을 **"복구 vs calibration 예산"**, 진짜
  bits 축은 E6의 **bf16 vs i8 sweep**으로 따로.
- **circular surrogate 금지** — Q_npu를 한 split에 calibrate, **held-out + 다른 모델**로 예측↔실측 검증.

---

## 5. 진도 계획 — 마일스톤 (M0→M6) + 7/5 매핑

> 날짜가 아니라 **마일스톤으로 굴립니다.** 논문 버전이 단조 증가 — 마감이 오면 *도달한 최고
> 마일스톤*을 제출. **M0–M4 = vISA 없이 완결되는 본 트랙 논문**, M5부터 칩 복구·강한 버전.

| 마일스톤 | 산출물 | 종료(다음으로) 조건 | vISA |
|---|---|---|---|
| **M0 평가 하네스** | ImageNet val 로더 + synset→라벨 매핑 | FP32 host top-1 == torchvision 71.9% | X |
| **M1 붕괴 정량화** | 칩 top-1 붕괴 곡선(2개 아키텍처) | 칩 top-1≈0% + per-layer rel-L2 곡선 | X |
| **M2 진단** | kurtosis·range → 붕괴층 예측기(AUC) | 예측기가 붕괴층 유의 분리 | X |
| **M3 충실 surrogate Q_npu** | bf16-operand 캐스트 + f32 누적 모델 | held-out·2nd-model에서 sim↔칩 낮은 불일치율 | X |
| **M4 최적화 복구(시뮬)** | int8 스케일·클립 최적화 + ablation + 비용 곡선 | 시뮬 near-FP32 복구 + (descent>closed-form or "closed-form 충분") | X |
| **M5 칩 복구** | (a) furiosa.torch 가중치 접기 **또는** (b) vISA 명시 int8 Cast | 실제 칩 top-1 복구 수치(부분이라도) | (b)면 ○ |
| **M6 일반화·확장** | Q_npu 모드 sweep · 혼합정밀도 CAMP(MCKP) · 3rd 아키텍처 · vISA 온칩 e2e | 강한 버전(IJCV/CVPR) | ○ |

### 5-1. 7/5 날짜 매핑 (vISA는 임계경로에서 제외 — 코어는 `classify.py`+호스트)

| 날짜 | 할 일 | 게이트 |
|---|---|---|
| **6/29** | ImageNet val 다운로드 시작. val 로더 + synset 매핑(M0) | FP32 host top-1 == 71.9% |
| **6/30 (★GO/NO-GO)** | 칩 붕괴 측정(E1). 단순 개입(clip+closed-form scale) 접어 npu3에서 500장 복구 읽기 | **복구 유의미?** GO→칩 복구안 / ~0%→operand-cast 국소화+surrogate 복구안. **어느 쪽이든 논문** |
| **7/1** | Q_npu surrogate + 충실도(E2). kurtosis 진단(E3). EfficientNet-B0 칩 붕괴 | EfficientNet-B0 컴파일 확인(안 되면 다른 망) |
| **7/2** | coordinate-descent 솔버. 복구+비용곡선+ablation(E4). bf16 vs i8 sweep(E6) | **14쪽 commit 게이트** |
| **7/3 (★등록)** | **등록.** operand-cast vs accumulator 분해(E5). full 50k. intro/method 작성 | 등록 완료 |
| **7/4** | results/related-work/ablation. LNCS·익명화. (upside) furiosa-opt cold-start→되면 vISA-sim 분해 그림 | cold-start 1일 내 안 되면 vISA upside 포기 |
| **7/5 (제출)** | 마무리·교정·부록. **일찍 제출**(OpenReview 버퍼). 남은 실험은 7/8 부록 | 제출 완료 |

---

## 6. 리스크 · 폴백

**노력으로 못 막는 블로커:** ① 닫힌 정밀도 경로의 복구 *크기*(물리) ② 닫힌 컴파일러가 접은 스케일을
재정규화로 무효화할 가능성 ③ 새 vISA conv 커널이 닫힌 lowering 통과 + TODO npu-dispatch.

**최소 완성 논문(폴백, vISA 0):** 칩 위 ImageNet 붕괴 특성화(다중 아키텍처) + kurtosis 진단 + 실측에
맞춘 surrogate + per-channel 스케일/클립 최적화로 **시뮬에서 복구 입증**(Gate A면 칩에서도). 이것만으로
완결된 CV 최적화 논문. 칩 복구·vISA는 전부 upside.

**descope:** MobileNetV1 제외(torchvision에 없음). 온칩 vISA conv = 미래연구. furiosa-opt 임계경로 제거.

**리뷰어 공격 → 방어:** "효율 Pareto 공허"→calibration-budget 정명 + bf16/i8 bits sweep · "SmoothQuant
증분"→부동소수 누적·FP32-foldable-only 영역 + kurtosis-α 우위 ablation(equalization은 DFQ 것 인정) ·
"circular surrogate"→held-out·2nd-model · "한 칩 버그"→Q_npu sweep · "얇은 CV"→ImageNet 헤드라인,
vISA 부록, "a closed-compiler reduced-precision NPU"로 익명화.

**후속(강한 venue): CAMP** — 층별 정밀도(i4/i8/f8/bf16)를 MCKP로 최적화하는 혼합정밀도 비트할당.
furiosa-opt 정밀도 엔진을 정면으로 쓰는 확장 → ACCV2027/CVPR/IJCV.

---

## 7. 실험 실행 방법

구체적 실행 절차(환경 준비, M0~M5 단계별 명령어·코드 골격)는 `rngd-npu/ACCV/02_실험계획.md`로
옮겼습니다. furiosa-opt의 어느 코드를 왜 쓰는지는 `rngd-npu/ACCV/03_furiosa-opt_코드해설.md` 참고.

---

## 출처

ACCV <https://accv2026.org/>(/submissions/, /organizers/) · nuLSQ(ACCV'24, 전례)
<https://openaccess.thecvf.com/content/ACCV2024/papers/Gongyo_Learning_Non-Uniform_Step_Sizes_for_Neural_Network_Quantization_ACCV_2024_paper.pdf> ·
furiosa-opt <https://github.com/furiosa-ai/furiosa-opt> · 책 <https://developer.furiosa.ai/furiosa-opt/book/> ·
내부 [info/README_virtual_isa.md] · [info/README_vision_compile.md] · [info/README_op_support.md].
**인물·선행연구 전체 출처와 주소 활용 경위·필요 지식은 [info/README_accv2026_background.md].**
