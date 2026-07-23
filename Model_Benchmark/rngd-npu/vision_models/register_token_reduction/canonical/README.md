# 정통(canonical) kNN 재현 — 이 폴더가 하는 일

이 문서는 "정통 프로토콜"(gallery = ImageNet **train** 전체, 무압축 baseline ≈ **82.0** 재현) 실험을 모아둔 폴더의 안내입니다. 우리 다른 실험들은 계산 절약을 위해 **val leave-one-out** kNN(gallery = query = val 5만)을 쓰는데, 그 절대값이 공인값(82.0)보다 낮아 리뷰어가 의심할 수 있어, **헤드라인 비교만 정통 프로토콜로 다시** 재는 것입니다.

## ⚠️ 이 두 스크립트가 커버하는 범위 (중요)
`compare.py`(ToMe · PiToMe · Ours 비교 엔진)를 **정통 train-갤러리**로 돌립니다. 즉 **논문의 두 핵심 표만** 재현합니다:
- **tab:main** (ToMe vs Ours) 의 정통판
- **tab:pitome** (공식 PiToMe head-to-head) 의 정통판
- + throughput

## ❌ 커버하지 **않는** 실험 (val leave-one-out 그대로 유지)
| 실험 | 위치 | 지표 | 정통판? |
|---|---|---|---|
| ablation (register/random/energy/highnorm) | `../ablation/` | kNN | ❌ 상대 비교라 val-LOO 유지 |
| eval_v2 (S/B/L 다전략 + linear-probe) | `../eval_v2/`(`results_50k/`) | kNN·선형프로브 | ❌ 이미 50k 완료(val-LOO), 3-seed |
| robustness (검색 mAP·부트스트랩·정식 ToMe) | `../robustness_50k/` | mAP·kNN·부트스트랩 | ❌ **이미 50k 완료**(val-LOO, `*_50k.log`, 논문 §aux 반영) |
| dense (분할) | `../dense/` | **mIoU** | ⛔ 해당 없음(ADE20k+선형 seg head, kNN 갤러리 개념 자체가 없음) |

> 참고: robustness_50k는 스크립트만 있던 상태에서 **50k 결과 로그 4개**(faithful_tome·pitome·reg_count_sweep·retrieval_map)로 채워졌고, 논문 §aux(정식 ToMe Δ·검색 mAP·부트스트랩 CI)가 이미 그 값으로 갱신됨. 이들은 **val-LOO** 기준이며 정통(train-갤러리) 재현 대상이 아님(정통은 헤드라인 ToMe/PiToMe/Ours만).

→ **정통 프로토콜을 이 4개까지 확장하려면** 각 스크립트(`eval_imagenet.py`·`eval_ablation.py`·`eval_v2.py`·robustness)에 `--gallery train`을 추가해야 하고, 비용이 크게 늘어납니다(특히 ViT-L). 현재는 **헤드라인(ToMe/PiToMe/Ours)만** 정통 재현하는 게 목적입니다.

## 실행
엔진·데이터·특징캐시는 `../pitome_compare/` 를 **공유**합니다(대용량 ImageNet을 중복 저장하지 않으려고). 결과 텍스트만 이 폴더에 저장됩니다.

```bash
# 베이스(헤드라인) — 정확한 82 재현. 단일 A100 ~17h(2장 ~9h). 스크립트 하단 2-GPU 예시.
bash run_base_canonical.sh
#   빠른 감: GALLERY_PC=260 RLIST="20" bash run_base_canonical.sh   (~30분~1h, ~82 근접)

# 전체(S/B/L)
bash run_full_canonical.sh
```

## 비용 요약 (측정 처리량: S 1200 / B 350 / L 102 img/s, A100)
| 설정(단일 A100) | S | B | L | 합계 |
|---|---|---|---|---|
| full 1.28M, r 5 | 5.1h | 17.1h | 58.2h | 80h |
| 260k, r 5 | 1.3h | 4.1h | 13.7h | 19h |

- **base = full 1.28M**(정확한 82, ~17h), **full(S/B/L) = 1.28M**.
- 결과가 나오면 `../pitome_compare/process_results.py` 로 표·그림 반영(정통판은 논문 tab에 별도 열/각주로 추가 예정).

## 관련
- 엔진·val-LOO 기본 실행: `../pitome_compare/` (compare.py, run.sh, process_results.py, `--gallery {val,train}`).
