# 새 논문 주제 — 필요사항·실현가능성·실행계획

이 문서는 RNGD NPU로 끝까지 갈 수 있는 새 비전 논문 주제의 **필요사항·충족여부·실행계획**을 정리한 것입니다. (기존 "NPU에서 ViT 분류 성공"은 *환경 구축*에 해당하므로, 그 위에서 새 기여를 세웁니다.)

## 1. 주제 (한 줄)
**"Compiles, Then Dies — 닫힌 저정밀 NPU에서 비전 트랜스포머의 배포가능성 지도":**
파운데이션 인코더(DINOv2 등)가 컴파일은 되는데 실행하면 **침묵형 NaN으로 죽는** 현상을 진단·인과규명·복구하고, 살아남은 ViT의 **결정론적 예측 드리프트**까지 묶어 "어떤 비전망이 닫힌 칩에 배포 가능한가"의 지도를 그린다.

## 2. 왜 이 주제인가 (닫힌 SDK 제약을 *통과*하는 유일한 부류)
- RNGD 닫힌 SDK는 **양자화/정밀도 제어는 막지만**(앞 조사 확정), **forward-only 컴파일·실행·프로파일링·관찰은 허용**. → "내 양자화를 칩에 적용" 주제는 전부 사망, "**칩이 하는 일을 관찰·진단·복구**" 주제는 생존.
- **이미 실측 증거 보유:** DINOv2 = 컴파일 OK / 실행 6/6 NaN. ViT·DeiT = FP32 동등(72.41 vs 72.43). Swin = 컴파일 FAIL("multiple internal subgraphs"). → 세 가지 배포 결과(생존/침묵사/컴파일거부)가 손에 있음.
- **해자(moat):** 실제 닫힌 상용 실리콘에서만 관찰됨(시뮬 bit-op로 NaN 재현 불가). 체어 직격: Lu(효율 ViT)+Shim(진단→수정).

## 3. 필요사항 & 충족 여부

| 필요 | 용도 | 충족? |
|---|---|---|
| **RNGD 카드(forward-only)** | NaN/드리프트/컴파일프런티어 실측, 프로파일러(DMA·연산 분리 latency) | ✅ 4장 보유, furiosa.torch 검증됨 |
| **GPU 서버** | ① FP32 레퍼런스(임베딩/로짓) ② 시뮬 양자화(bf16/int8 fake-quant) = *sim≠silicon* 증명 ③ DINOv2 linear-probe 학습(복구 화질 측정) | ⏳ **사용자 보유 확인됨** — 아래 4 패키지 충족 필요 |
| **데이터** | ImageNet val(분류 ViT), DINOv2/SSL은 임베딩+linear-probe용 라벨 | ✅ 균형 ImageNet val 10000장 보유 |
| **모델** | DINOv2(±register), CLIP/SigLIP, MAE, ViT/DeiT, Swin | ✅ 전부 timm/HF 자동 다운(reg4 등 확인됨) |
| **복구 코드** | weight-only 가중치접기를 **from_exported**로(⛔reuse-edf 금지) | 🔧 recover_fold 로직 보유, DINOv2용 재적용 필요 |
| **프로파일러** | 어디서 NaN 나는지·DMA/연산 비용 | ✅ `RNGDProfiler` 확인(per-op latency, energy는 외부계측 필요) |

### GPU 서버에서 필요한 것(충족 조건)
- 패키지: `torch>=2.4, timm>=1.0, torchao>=0.7, datasets, pillow` (양자화 비교용 — 이미 `gpu_quant/requirements.txt`에 정리)
- 역할: (a) 각 인코더 FP32/fake-quant 레퍼런스 임베딩 생성, (b) NPU 결과와 대조(NaN·드리프트), (c) DINOv2 linear-probe로 복구 전후 다운스트림 정확도.
- **진행 가능 판정:** GPU에 위 패키지만 깔리면 즉시 가능. 본 머신(NPU 서버)엔 GPU 없으니 GPU 측 스크립트는 `gpu_quant/`처럼 **복사·실행형**으로 제공 → 사용자가 GPU 서버에서 실행.

## 4. Make-or-Break (지금 검증 중)
**[A] register 자연실험** — `vit_base_patch14_reg4_dinov2`(레지스터 토큰 보유판)가 NPU서 **유한**이고 일반 DINOv2가 NaN이면 → **"고노름 아티팩트 토큰이 하드웨어 수치붕괴의 원인"을 인과적으로 증명**(가장 싼 결정적 실험, rngd:8 진행 중).
**[B] 예측 드리프트** — ViT가 iso-accuracy인데도 NPU vs FP32 per-sample 예측이 수% 뒤집히고 그 샘플 margin이 작은가(flip_analysis.py).

## 5. 실행 계획 (단계)
- **M0 (now):** register 인과실험 + 드리프트율 측정 = 두 축의 생사 판정.
- **M1:** NaN 국소화 — 멀티아웃풋으로 어느 블록/토큰/sub-op(LN rsqrt·softmax·LayerScale)에서 NaN 발생하는지 칩 위에서.
- **M2:** 복구 — DINOv2를 from_exported로 weight-only fold 재적용 → NaN→유한, cosine·linear-probe 회복.
- **M3:** 배포가능성 taxonomy — DINOv2/CLIP/SigLIP/MAE/ViT/DeiT/Swin을 생존/침묵사/컴파일거부로 분류 + 원인.
- **M4:** GPU sim≠silicon — fake-quant로는 NaN·드리프트 재현 안 됨을 보여 실리콘 필수성 증명.
- **M5:** 집필. 마감 도달 최고 마일스톤 제출.

## 6. 마감/타깃
ACCV 2026(07-05)은 5일이라 비현실적 → **CVPR 2027(약 11월) 등 수개월 여유 타깃** 권장. 단 M0~M2(인과+복구)는 며칠 내 가능 → 조기에 핵심 결과 확보.
