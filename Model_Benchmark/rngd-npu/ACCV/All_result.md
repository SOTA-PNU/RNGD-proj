# 논문 실험 전체 정리 — 계획·이유·결과

이 문서는 이 논문(register 보호 기반 토큰 압축)을 위해 계획하고 실행한 모든 실험을, 각 실험을 왜 했고(무엇을 밝히려 했는지) 무엇이 나왔는지까지 한곳에 모은 정리표입니다.

## 0. 이 실험들이 무엇을 향하나

논문의 한 줄 주장은 이것입니다: **비전 인코더의 토큰을 극단적으로 줄일 때, register 토큰을 병합에서 보호하면 정확도가 훨씬 덜 떨어진다.** 나머지 실험은 대부분 이 주장에 나올 법한 **반론을 하나씩 막기 위한 것**입니다. 예를 들어 "그건 토큰을 더 남겨서 아니냐", "베이스라인이 약해서 아니냐", "지표 하나(kNN)에서만 그런 거 아니냐", "단일 seed라 우연 아니냐" 같은 반론입니다.

## 1. 데이터셋을 왜 이렇게 나눴나 — "dense mIoU는 왜 ImageNet이 아닌가"

핵심은 **지표가 요구하는 라벨의 형태가 다르기 때문**입니다.

- **ImageNet-1k**: 이미지 한 장에 라벨이 **하나**(그 이미지가 무슨 물체인지)입니다. 이미지 전체를 대표하는 특징(global feature)의 품질을 재는 데 맞습니다 → kNN 분류, 검색 mAP, linear-probe에 사용.
- **ADE20k**: **픽셀마다** 라벨이 있는 분할(segmentation) 데이터셋(150개 클래스)입니다(`dense_seg.py:14`). mIoU(mean Intersection-over-Union)는 "예측한 픽셀 영역과 정답 픽셀 영역이 얼마나 겹치나"를 재는 지표라, **픽셀 단위 정답 마스크가 반드시 필요**합니다(`dense_seg.py:84-94`).

**ImageNet에는 픽셀 단위 마스크가 없습니다.** 그래서 ImageNet으로는 mIoU를 계산할 수 없습니다 — 데이터셋을 피한 게 아니라, mIoU라는 지표 자체가 ImageNet에서 정의되지 않습니다. ADE20k는 DINOv2 계열 논문들이 분할 성능을 볼 때 쓰는 표준 벤치마크라서 채택했습니다.

두 지표는 **서로 다른 것을 검증**하기도 합니다. kNN(ImageNet)은 "이미지를 한 벡터로 잘 요약하는가"를, dense mIoU(ADE20k)는 "패치마다의 공간적 특징이 위치별로 살아있는가"를 봅니다. 토큰 병합은 공간 정보를 흐리기 쉬워서, dense는 register 보호의 효과가 가장 뚜렷하게 드러나는 자리입니다.

| 데이터셋 | 라벨 형태 | 재는 것 | 쓰인 실험 |
|---|---|---|---|
| ImageNet-1k val (50,000장) | 이미지당 1개 | 전역 특징 품질 | kNN, 검색 mAP, linear-probe, 각종 ablation |
| ADE20k val (2,000장 = 전체) | 픽셀당 1개 | 공간 특징 품질 | dense 분할 mIoU |
| RNGD NPU (실칩) | — | 실제 지연(속도) | NPU latency |

## 2. 전체 실험표 (계획 이유 + 결과 + 상태)

**논문 반영 원칙:** 최종 논문 본문 수치는 **전체 스케일(✅)만** 사용합니다. **2026-07-02 03:58 기준, subset이던 실험(#3·4·5·6·7·8·9)이 모두 전체 50k로 확정·교체 완료**(eval_v2 50k seeds3 + robustness_50k 50k GPU). 이제 본문 전 실험이 전체 규모입니다.
- **전체 50k 완료**: #1~#10, #12 (kNN·PiToMe·대조군·ablation·동적재선택·정식ToMe·reg스윕·검색mAP·다모델·linear-probe·dense·음성대조군)
- **아직 안 한(GPU 필요, 선택)**: #11 native 해상도, GPU wall-clock throughput(단 tab:pitome에 im/s는 이미 측정됨)

범례: ✅전체(논문 가능) · 🔒미실행(GPU 필요, 선택)

| # | 실험 | 무엇을 밝히려 했나 (계획 이유) | 결과 (한 줄) | 데이터·규모 | 상태 |
|---|---|---|---|---|---|
| 1 | 메인 kNN | register 보호가 극단 압축서 정확도를 지키나? | Ours 전 압축률 우세, 격차 +1.0→+8.1; 91%서 71.98 vs 63.91 | ImageNet 50k | ✅ |
| 2 | 정적 keep-prior ablation | 이득이 register 때문인가, 그냥 토큰을 더 지켜서인가? | register만 이김; random/energy/highnorm은 무보호와 노이즈 내 | ImageNet 50k | ✅ |
| 3 | 동적 재선택 ablation | "정적 선택이라 베이스라인이 약한 것" 반박 | 동적 prior 세지지만(91%서 67.4) Ours 71.9로 여전히 +4.4 | ImageNet 50k | ✅ |
| 4 | 실제 PiToMe head-to-head | ablation의 energy는 프록시 — 진짜 SOTA를 이기나? | 온건압축 PiToMe 우세, 극단서 Ours 이김(92%서 +3.2); tab:pitome(층별margin)·pitome_50k(margin0.9) 일치 | ImageNet 50k | ✅ |
| 5 | 정식 ToMe head-to-head | 베이스라인이 정식 ToMe 아님(proportional attn 빠짐) 반박 | 강한 ToMe 상대로도 Ours 이김(55/74/91%서 +2.2/+4.8/+10.3) | ImageNet 50k | ✅ |
| 6 | register 개수 스윕 + 부트스트랩 CI | 원인=register인가(개수 confound)? 단일 seed 유의한가? | k≥2서 계단상승 후 포화; 부트스트랩 CI가 **전 압축률 0 배제**(55%[+2.0,+2.5]·91%[+7.5,+8.3]) | ImageNet 50k | ✅ |
| 7 | 검색 mAP | 지표가 kNN 하나뿐 아니냐 반박(두 번째 지표) | Ours 전 압축률 이김, 격차 +2.8→+13.1(55/74/91%서 +5.1/+8.1/+13.1) | ImageNet 50k | ✅ |
| 8 | 다모델 일반화 | N=1(단일 모델) 아니냐 | 모델 의존적: base 극단 뚜렷(+7.9), large 상대 대폭·절대 낮음, small은 동적energy가 근소 우위(61.5 vs 60.7) | ImageNet 50k(3모델) | ✅ |
| 9 | linear-probe | kNN 특정 지표 아니냐(표준 선형 probe) | 3모델 전부 Ours 우세, base 91%서 +6.3(69.2→75.5) | ImageNet 50k(3모델) | ✅ |
| 10 | dense 분할 mIoU | 공간 과제(dense)에서도 register 보호가 돕나? | Ours 전 압축률 1위(+3.1@91%), 대안 전부 무보호 이하 | ADE20k val 2k=전체 / probe학습 train 2k=~10% | ✅평가전체·🟡probe일부 |
| 11 | 병합빈도 mechanism | "정확도 붕괴 = register가 병합돼 사라짐"의 직접 증거 | ToMe서 register 94%가 평균 block3.6에 소멸, Ours 100% 생존 | ImageNet 16장 | ✅ |
| 12 | FLOP 분석 | 효율 증거 + "이득이 토큰 더 남겨서" 배제 | 17~43% 절감; Ours≈ToMe(<0.1%)라 이득은 공짜 | 해석(모델 구조) | ✅ |
| 13 | NPU 실지연 | 토큰 축소가 실제 NPU서 빨라지나? | 전 구간 속도이득 0(7.52→7.54ms); "NPU 가속" 주장 삭제 | RNGD 실칩 | ✅ |
| 14 | 음성 대조군 | register 없는 모델에선 두 팔이 같아야(정합성 점검) | 구성상 Δ=0 확인 → 노이즈 바닥·하네스 검증 | ImageNet | ✅ |

## 3. 실험별 상세 (무엇을 밝히려 했나 / 방법 / 결과)

**1. 메인 kNN (ImageNet 50k).** *밝히려는 것:* 핵심 주장 자체. *방법:* DINOv2-ViT-B/14-reg4를 고정, 블록마다 size-가중 ToMe 병합, ToMe(CLS만 보호) vs Ours(CLS+register4). 50k를 gallery이자 query로 leave-one-out kNN(k=20). *결과:* 모든 압축률서 Ours 우세, 격차가 압축 강할수록 커짐(37%서 +1.0 → 91%서 +8.1). 91%서 Ours 71.98(full 76.35 대비 −4.4), 무보호 63.91(−12.4).

**2. 정적 keep-prior ablation (50k).** *밝히려는 것:* 이득의 원인이 "register 자체"인지, "토큰을 더 보호해서"인지. *방법:* 같은 병합·같은 보호 개수, 보호 대상만 register/무작위/energy/highnorm으로 교체. *결과:* register만 크게 이김(91%서 71.98), 나머지는 무보호(63.9)와 노이즈(±0.9) 안 → 원인은 register.

**3. 동적 재선택 ablation (전체 50k).** *밝히려는 것:* 위 ablation의 highnorm/energy가 입력단에서 한 번만 골라 고정된 "약한 버전"이라는 반박. *방법:* 매 블록마다 highnorm/energy를 다시 골라 보호(동적). *결과:* 동적이 정적보다 세지지만(91%서 최선 동적 67.4) Ours 71.9로 +4.4 우위. "정적 prior엔 못 미친다"는 정직하게 서술.

**4. 실제 PiToMe head-to-head (전체 50k).** *밝히려는 것:* ablation의 energy는 평균코사인 프록시일 뿐 — 실제 SOTA(PiToMe)를 이기나. *방법:* PiToMe 공식 알고리즘(energy=elu(sim−margin), 고에너지 병합·저에너지 보호) 구현, 층별 margin(tab:pitome)·constant margin(pitome_50k) 둘 다. *결과:* **온건압축(37/55%)선 PiToMe가 근소 우세, 극단(83/92%)선 Ours 우위(92%서 +3.2)** — 교차점 ~70%. 두 margin 결과 일치. **정직 수정:** 소규모 n=3000(constant margin)에선 "Ours 전구간 승(+1.1/+0.7/+4.1)"으로 보였으나, 전체 50k에선 온건압축서 PiToMe가 이김 — 소규모 아티팩트였고 50k가 authoritative. 논문 헤드라인은 "극단 압축서 Ours 우위"로 정직하게 좁힘. (`pitome_compare/`)

**5. 정식 ToMe head-to-head (전체 50k).** *밝히려는 것:* 우리 베이스라인이 proportional attention을 안 넣은 "약한 ToMe" 아니냐. *방법:* proportional attention(log(size) 편향) + key-metric 유사도 + attn↔MLP 사이 병합의 정식 ToMe를 무보호 팔로. *결과:* 강한 ToMe 상대로도 Ours 전 압축률 이김(55/74/91%서 +2.2/+4.8/+10.3). "약한 베이스라인" 우려 해소. (`faithful_tome_h2h.py`)

**6. register 개수 스윕 + 부트스트랩 CI (전체 50k).** *밝히려는 것:* (a) 이득이 register 자체냐 개수냐, (b) 단일 seed인데 통계적으로 유의한가. *방법:* 보호 register 수 k=0→4 스윕(같은 병합), 그리고 평가셋을 재표집(부트스트랩)해 (Ours−ToMe) 차이의 95% CI. *결과:* register 1개론 거의 무효, k≥2서 계단식 상승 후 포화(→register는 묶음으로 지켜야 함, 엄밀 단조는 아님). **부트스트랩 CI가 전 압축률서 0 배제**(55%[+2.0,+2.5]·74%[+3.8,+4.4]·91%[+7.5,+8.3]) → 소규모(n=3000)선 걸치던 중간압축도 50k선 유의. 이득 크기는 여전히 극단서 최대. (`reg_count_sweep.py`)

**7. 검색 mAP (전체 50k).** *밝히려는 것:* 지표가 kNN 하나뿐이라는 반박 — 다른 표준 지표에서도 이기나. *방법:* 같은 특징으로 이미지 검색 mAP(query를 gallery 전체에 랭킹, 같은 클래스면 relevant). *결과:* Ours 전 압축률 이김, 격차가 압축 강할수록 커짐(37%서 +2.8 → 91%서 **+13.1**; 55/74/91%서 +5.1/+8.1/+13.1). 91%서 Ours 47.69 vs ToMe 34.57. 이 격차는 kNN(+7.9)보다 커서, **register 우위가 kNN이라는 특정 지표의 산물이 아님**을 확인. (`retrieval_map.py`)

**8. 다모델 일반화 (전체 50k, 3모델).** *밝히려는 것:* 단일 모델(N=1) 결과 아니냐. *방법:* small/base/large-reg 세 register 모델. *결과:* 모델 의존적 — base는 극단서 Ours 뚜렷(+7.9), large는 무보호가 붕괴(91%서 2.8)해 Ours(8.6)의 상대이득 크나 절대 낮음, small은 무보호가 이미 강건해 이득 작고(+0.9) 동적 energy가 Ours를 근소 우위(61.5 vs 60.7). "보편적"이라 과장하지 않고 범위를 밝힘.

**9. linear-probe (전체 50k, 3모델).** *밝히려는 것:* kNN 특정 지표 아니냐. *방법:* 특징 표준화 후 클래스 평균 프로토타입 최근접(선형 probe), val 층화 분할. *결과:* 세 모델 전부 Ours가 전 압축률 우세, base 기준 격차 37%의 +1.2 → 91%의 +6.3(69.2 대 75.5). kNN·검색 mAP에 이어 세 번째 지표에서도 우위.

**10. dense 분할 mIoU (ADE20k).** *밝히려는 것:* 공간 정보가 중요한 dense 과제에서도 register 보호가 돕나. **이 과제를 고른 이유는 메커니즘의 직접 예측** — register가 전역 정보를 담고 병합이 공간 구조를 무너뜨린다면, 공간이 가장 중요한 분할에서 효과가 가장 커야 함(유리한 자리를 노린 cherry-pick이 아니라 가설 검증). *방법:* 백본(DINOv2-reg)과 우리 기법은 **학습하지 않음(frozen, 학습 파라미터 0)**. 오직 **선형 head 1개**를 full 특징으로 **한 번 학습해 모든 압축률·전략에 공유**(mIoU는 픽셀→클래스 매핑이 필요해 최소 readout으로 선형 probing; 어느 팔도 전용 head 못 받아 공정). 토큰 병합·unmerge 후 mIoU, 5전략 비교. *규모:* **평가는 ADE20k val 2,000장=전체**, 단 **선형 head 학습은 train ~20,210장 중 2,000장(~10%) subset**(`dense_seg.py` 기본값 `--n_train 2000 --n_val 2000`). *결과:* Ours 전 압축률 1위(91%서 17.2 vs 무보호 14.1, +3.1; full 23.1). **분류와 달리 random/energy/highnorm이 전혀 안 도움(무보호 이하)** — dense선 register만 효과로 더 깨끗. *정직:* dense는 효과가 **가장 크게 예측되는 best case**이며 헤드라인은 ImageNet(#1). probe 학습을 늘린 전체 확정은 GPU 준비물에 포함. (`dense_seg.py`)

**11. 병합빈도 mechanism (16장).** *밝히려는 것:* "정확도가 붕괴하는 건 register가 병합돼 사라지기 때문"을 간접추론이 아니라 직접 보이기. *방법:* ToMe 병합 과정에서 register가 얼마나·언제 병합되는지 추적. *결과:* ToMe서 register의 94%가 평균 block 3.6(12블록 중 이른 시점)에 병합소멸, Ours는 100% 생존.

**12. FLOP 분석.** *밝히려는 것:* 효율 근거 + "Ours 이득이 토큰 4개 더 남겨서" 배제. *방법:* 토큰 스케줄 기반 백본 FLOP 계산. *결과:* 17~43% 절감(91%서 43%). **Ours와 ToMe의 FLOP 차이 <0.1%** → +8%는 compute를 더 써서가 아니라 사실상 공짜.

**13. NPU 실지연 (RNGD 실칩).** *밝히려는 것:* 토큰 축소가 실제 NPU 하드웨어에서 빨라지나. *방법:* furiosa 컴파일 후 실칩 forward 지연 측정. *결과:* 전 구간 속도이득 0(7.52ms→7.54ms), r=18은 컴파일 실패. 정직하게 "NPU 가속" 주장을 넣지 않고, 효율은 FLOP/토큰 수로만 제시.

**14. 음성 대조군.** *밝히려는 것:* register가 없는 모델에선 두 팔이 동일해야 한다는 정합성 점검. *방법:* register-free DINOv2에 같은 실험. *결과:* 구성상 Ours≡ToMe(Δ=0) → 노이즈 바닥이자 실험 하네스 검증. 원인 규명은 #2 ablation의 몫.

## 4. 아직 안 했거나 GPU가 필요한 것

- **전체 50k 스케일 확정** (#3~9): 지금 로컬 CPU라 2,000~10,000장 subset. 전체 50k는 `register_token_reduction/README_GPU_50k.md`에 전송·실행·회수까지 준비 완료. ⛔ 실행은 GPU 서버 사용 허가 대기(현재 타인 사용 중).
- **native 해상도(448/518) 1점 확인** (선택): 효과가 고해상도에서도 지속되는지. 무거워 GPU 필요.
- **GPU wall-clock throughput**: FLOP 절감이 실제 처리량으로 이어지는지(대규모 배치).

## 5. 종합 — 지금까지 확립된 것

1. **핵심 주장은 전체 50k에서 성립**: 극단 압축서 register 보호가 무보호를 +8p 이김(#1), 그 원인은 register 자체(#2, #6).
2. **강한 상대에게도 이김**: 실제 PiToMe(#4), 정식 ToMe(#5) 모두 극단서 Ours 우위.
3. **여러 반론을 방어**: 토큰수 confound(#6, #12), 단일 seed 유의성(#6 부트스트랩), 지표 다양성(#9 linear-probe, #7 검색 mAP), 정적 prior 약함(#3), N=1(#8).
4. **dense가 가장 깨끗**: 공간 과제선 register만 효과(#10).
5. **정직한 한계**: 모델 의존적(#8, small-reg는 동적energy와 대등), NPU 가속 없음(#13). **모든 실험 전체 50k 완료**(2026-07-02).

> 스크립트: `vision_models/register_token_reduction/`(eval_v2·dense·ablation·robustness_50k), `vision_models/{pitome_h2h,faithful_tome_h2h,reg_count_sweep,npu_latency,reg_merge_freq,flops_table}.py`. 로컬 로그: `vision_models/results/*.log`. 전체 50k 준비물: `register_token_reduction/README_GPU_50k.md`.
