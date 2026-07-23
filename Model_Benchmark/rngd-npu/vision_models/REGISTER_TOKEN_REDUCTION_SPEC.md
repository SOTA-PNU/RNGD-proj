# Register-aware Token Reduction — 연구 스펙

이 문서는 ACCV 투고용 새 주제의 설계·신규성·실험계획을 정리한 것입니다. (효율·경량화, 양자화 아님. 알고리즘이 기여, GPU에서 개발, NPU는 보조 검증.)

## 논문 주제
재학습 없이, register/큰(고노름) 토큰을 보호하며 중복 패치 토큰을 병합해 foundation 비전 인코더(특히 register를 가진 DINOv2-reg 계열)를 고압축·가속하는 토큰 압축 방법.

## 기존 문제
- foundation 인코더는 토큰 수백 개를 전부 처리해 느림. 토큰 압축(병합/가지치기)으로 줄이지만, 기존 방법(ToMe, PITOME 등)은 토큰 중요도를 **유사도·에너지 같은 간접 신호**로 판단함.
- 이 모델들에는 **사진 전체 요약을 담은 소수의 register/고노름 토큰**이 있는데, 기존 방법은 이 구조를 **명시적으로 안 써서** 그 핵심 토큰을 합쳐 없애거나(특히 register는 의미상 "배경"처럼 보여 합쳐지기 쉬움) 압축률을 못 높임. 고압축·dense(분할/깊이) 작업에서 손실이 큼.

## 해결 방법
1. 보호 대상 식별: 명시적 register 토큰 + (없으면) 고노름 토큰 top-k.
2. 이들을 병합에서 제외(보호), 나머지 중복 패치만 bipartite soft matching으로 강하게 병합.
3. 재학습 없는 plug-in. dense 작업은 병합을 역추적(unmerge)해 해상도 복원.

## 해결 결과 *(Phase 0 = 검증됨 / Phase 1~2 = 측정 예정)*
- **Phase 0 (CPU, 같은 병합서 보호대상만 변경, full 대비 CLS cosine):** DINOv2-reg에서 register 보호가 표준 ToMe 대비 **고압축(92%)서 +0.19**(0.875 vs 0.687), 73.6%서 +0.09. plain DINOv2=극단압축선 이득, CLIP=미미.
- **Phase 0b (거친 병합, kNN, 1200장):** 92% 압축서 ours 64.92 vs tome 60.83 (+4.1%).
- **Phase 1 예비 (★정식 size-가중 ToMe, kNN, DINOv2-reg, 800장):** 92% 압축서 ours **58.75** vs tome **55.00 (+3.75%)**, 게다가 ours는 74%→92%서 거의 안 떨어짐(58.6→58.75)=극단압축에 견고, ToMe는 급락(60.6→55.0). 중간(74%)선 ToMe가 약간 우위 → **우리 영역=극단 압축(>90%)**. (strawman 아님: 공정한 size-가중 ToMe 기준)
- **목표:** 극단 압축에서 더 높은 정확도(분류·검색·dense). GPU 풀스케일·PITOME 비교·dense로 확정.
- **GPU 패키지:** `register_token_reduction/` (tome_reg.py·eval_imagenet.py 검증됨, prepare_data·run.sh·README, 복사·실행형).

## 일반화
- register를 가진 인코더군 전반(DINOv2-reg, DINOv3 등 — register가 표준이 되는 추세)에 적용.
- "register를 압축의 사전지식(prior)으로 쓴다"는 원리는 가지치기·적응계산 등 다른 효율 기법으로 확장 가능. 해석가능성용 register를 효율 도구로 재해석.

## 신규성 위치 (직접 확인)
- **PITOME** (NeurIPS'24, 2405.16148): 유사도/스펙트럴 기준, register 무관. artifact 토큰을 오히려 합쳐버림 → 정면 차별점.
- **FNA** (2507.16018): artifact를 landmark로 어텐션 *근사*(토큰 유지). 우리는 토큰 *축소* — 다른 축(상보적), 단 "artifact=효율유용" 선점 위험 → 기여를 "토큰 축소 + ToMe가 register 낭비함 + dense/극단압축"으로 날카롭게.
- 근거: 2505.05892(register=global정보), 2506.08010(train-free register). 빈자리 = register를 명시적 보호기준으로 한 토큰 축소.

## 실험 계획
- **Phase 1 (GPU, 본실험):** 정식 ToMe(key기반+size가중) 구현, ToMe/PITOME/ours/random 비교. 모델 DINOv2-reg(주)·DINOv2·CLIP·SigLIP. 지표: ImageNet linear-probe/kNN, 검색, **dense(분할 mIoU/깊이)**. 압축률 sweep(중간~극단 >90%). 결정적 ablation: register 보호 vs 무작위 vs 에너지(PITOME).
- **Phase 2 (NPU, 보너스):** 토큰 줄인 모델 실칩 지연/throughput(forward-only 컴파일됨).
- **데이터:** ImageNet val(보유 10000) + 분할/깊이 표준셋. GPU 서버로 파일 이동해 실행.

## 위험 & 대응
- 효과가 register 모델에 집중 → 주제를 "register-aware"로 좁혀 정직하게 프레이밍(비-register는 보조).
- FNA 동시기 선점 → 메커니즘(축소 vs 근사) 차이 + ToMe-register-낭비 발견으로 차별화.
- Phase 0 거친 prototype → Phase 1서 정식 ToMe·실 downstream·dense로 확정.

## 도구
- `phase0_register_tome.py` (CPU 가설검증), `phase0b_knn_accuracy.py` (실 정확도), `token_norms.py`(고노름 토큰 측정).
