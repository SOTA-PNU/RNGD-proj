# random 대조군 결과 — register-aware 토큰 압축 (2026-07-01)

이 문서는 "ours의 +3.75%가 정말 register 특수성인가, 아니면 보호 개수/ToMe 약점인가"를 가리는
**random 대조군**(아무 patch 4개 보호) 실측 결과입니다. 리뷰어가 반드시 요구할 베이스라인이라 선제 검증했습니다.

## 셋업
- 그분들 정식 size-가중 ToMe(`register_token_reduction/tome_reg.py`의 `merge_step`) **그대로** 사용.
- 같은 보호 개수(5=CLS+4)로 세 전략 비교: tome(CLS만) / ours(CLS+register4) / random(CLS+무작위 patch4, register는 병합 대상).
- DINOv2-reg, ImageNet-val 400장(40클래스), multi-layer, kNN top-1. CPU. 무압축 상한 79.25%.
- 도구 `h2h_random_control.py`, 로그 `results/logs/h2h_random_control.log`.

## 결과

| r | 압축% | tome | **ours** | random | Δ(ours−random) |
|---|---|---|---|---|---|
| 12 | 55.2 | 74.25 | 76.75 | 75.25 | +1.50 |
| 16 | 73.6 | 72.25 | 77.00 | 70.25 | +6.75 |
| 20 | **91.2** | 67.50 | **76.25** | 66.50 | **+9.75** |

## 판정 = GO
- **ours가 random을 극단압축(91%)서 +9.75%로 크게 이김.** random 4개 보호는 tome보다도 나쁨(66.5<67.5) → 이득은 "토큰 4개 더 보호"가 아니라 **register 자체의 특수성**.
- **ours는 압축에 견고**(55→91%서 76.75→76.25, 거의 불변), tome·random은 붕괴.
- 해석: register는 12개 층에 걸쳐 전역정보를 *누적*하는 토큰이라, 보호 시 multi-layer 이득이 큼. (단일레이어 post-hoc `h2h_global_token.py`에선 이 이득을 못 잡아 random이 맞먹어 보였음 — proxy 한계.)

## 남은 게이트 (정직)
1. ★**PiToMe 정면 head-to-head** — 진짜 강한 경쟁자(NeurIPS'24). tome·random은 이겼으나 PiToMe는 별도 must-win(GPU 본실험).
2. **일반성** — register 없는 모델(plain/CLIP)에서 고노름 토큰 보호로 같은 이득 나오나.
3. **규모·dense** — ImageNet 50k·여러 모델·ADE20k seg mIoU로 확정.
4. **NPU 실측 속도**(Lu의 측정 화폐, 보조).
