# robustness_50k — 보조 실험의 전체 50k 확정용 묶음

이 문서는 GPU 서버에서 전체 ImageNet 검증셋(50,000장)으로 돌릴 보조 실험 3종을 정리한 안내서입니다.

## 왜 이 폴더가 있나요

논문의 **메인 결과(kNN 표·ablation 표)는 이미 전체 50,000장**으로 측정했습니다. 다만 아래 세 가지 보강 실험은 그동안 로컬 머신(GPU 없이 CPU)에서만 돌 수 있어 일부(2,000~3,000장)만 측정했습니다. kNN 정확도는 gallery에 같은 클래스 이웃이 얼마나 많은지에 크게 좌우돼서, 표본이 작으면 수치가 흔들립니다. 그래서 이 세 실험도 **전체 50k로 다시 확정**하려고 서버용으로 묶었습니다.

세 실험 모두 감사(보수화) 과정에서 나온 지적에 답하는 것들입니다.

## 무엇이 들어 있나요

| 파일 | 무엇을 보는가 | 어떤 지적에 답하는가 |
|---|---|---|
| `reg_count_sweep.py` | 보호하는 register 개수를 0→4로 늘리며 정확도 변화 + 부트스트랩 95% 신뢰구간 | "이득이 register 때문인지, 그냥 토큰을 더 남겨서인지" / "단일 seed라 유의성 불명" |
| `faithful_tome_h2h.py` | 정식 ToMe(proportional attention 포함) vs Ours | "베이스라인이 정식 ToMe가 아니다(약한 상대)" |
| `pitome_h2h.py` | 실제 PiToMe(공식 알고리즘) vs Ours vs ToMe | "energy는 프록시일 뿐, 실제 PiToMe와 직접 비교 필요" |
| `retrieval_map.py` | 이미지 검색 mAP(두 번째 표준 지표) ToMe vs Ours | "지표가 kNN 하나뿐 아니냐" |
| `tome_core.py` | 공용 함수(병합·kNN·부트스트랩·모델/데이터 로드) | (라이브러리) |

### 각 실험을 어떻게 읽나요

- **reg_count_sweep**: `k=0`이 ToMe(클래스 토큰만 보호), `k=4`가 Ours(클래스 토큰+register 4개 보호)입니다. **k가 커질수록 정확도가 단조로 오르면**, 이득의 원인이 "토큰을 더 남겨서"가 아니라 "register를 하나씩 더 지켜서"라는 뜻입니다. 끝의 `95%CI`는 평가셋을 다시 뽑아(부트스트랩) 구한 (Ours−ToMe) 차이의 신뢰구간입니다. **하한이 0보다 크면** 단일 seed여도 통계적으로 견고하다는 뜻입니다.
- **faithful_tome_h2h**: 베이스라인을 일부러 강하게(정식 ToMe의 proportional attention·key 유사도·attn↔MLP 사이 병합) 만든 뒤 Ours와 붙입니다. 그래도 극단 압축에서 Ours가 이기면, 앞선 우위가 "약한 상대 덕분"이 아님을 확정합니다.
- **pitome_h2h**: energy 기반 최신 기법(PiToMe)의 공식 알고리즘을 그대로 구현해 Ours와 붙입니다.

## 어떻게 돌리나요 (서버에서)

전제: 서버에 `~/register_token_reduction/imagenet_val`(50k 이미지 + `labels.csv`)이 있고, 파이썬 환경에 `torch`, `timm`, `pillow`가 설치돼 있어야 합니다(이미 eval_v2 돌린 환경 그대로면 됩니다).

```bash
cd ~/register_token_reduction/robustness_50k
# 전체 50,000장 (인자 생략하면 기본 50000)
python reg_count_sweep.py 50000    | tee reg_count_sweep_50k.log
python faithful_tome_h2h.py 50000  | tee faithful_tome_50k.log
python pitome_h2h.py 50000         | tee pitome_50k.log
# 한 번에:  bash run_all.sh
```

이미지 경로가 다르면 `IMAGENET_VAL` 환경변수로 지정하세요:

```bash
IMAGENET_VAL=/경로/imagenet_val python reg_count_sweep.py 50000
```

## 결과가 나오면

세 `.log` 파일을 로컬로 가져오면 논문 표/PPT에 전체 50k 수치로 갱신합니다. 로컬로 보내는 예시:

```bash
scp -P 10022 jun@164.125.249.13:~/register_token_reduction/robustness_50k/*_50k.log \
  ~/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/register_token_reduction/robustness_50k/
```

## 참고

- 세 스크립트 모두 GPU가 있으면 자동으로 사용하고, 없으면 CPU로 돕니다. 50k는 GPU에서 수십 분 안에 끝나지만 CPU에서는 매우 오래 걸립니다.
- 모델은 `vit_base_patch14_reg4_dinov2.lvd142m`(register 4개) 하나로 고정돼 있습니다. 메인 논문과 같은 조건입니다.
