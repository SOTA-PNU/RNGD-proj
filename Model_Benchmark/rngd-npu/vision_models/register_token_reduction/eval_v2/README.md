# eval_v2 — 강화 평가 (감사 지적 보완, GPU 패키지)

이 폴더는 보수화 감사가 지적한 약점을 실험으로 메우는 **강화 평가**입니다. 기존 본실험(tome vs ours, 단일 seed, kNN)에 다음을 추가합니다.

| 감사 지적 | eval_v2의 보완 |
|---|---|
| 단일 seed·오차막대 없음 | **다중 seed(기본 3)** → 셀마다 평균±표준편차 |
| 비표준 kNN(val 자기갤러리) | **linear-probe top-1**(클래스내 40/10 split) 병행 |
| 단일 모델(N=1) | **register 모델 3종**: DINOv2 small/base/large-reg4 |
| keep-prior가 입력단 고정(허수아비) | **동적 재선택**(`energy_dyn`/`highnorm_dyn`, 매 블록 현재 토큰서 재선택) |
| 효율 무증거·동일예산 아님 | **FLOP 비율**(토큰 스케줄 기반)·**팔별 최종 토큰수** 보고 |

## 왜 이걸로 방어되나
- ours가 **여러 register 모델**에서 **여러 seed 평균**으로, **동적으로 재선택한 highnorm/energy**까지 이기면 → "register가 특별하다"가 단일 우연/허수아비가 아님이 강해짐.
- linear-probe로 같은 결론이 나오면 지표 의존성 해소.
- FLOP/토큰수로 "효율"을 실측 속도 없이도 정량화(단, 실측 speedup은 별도 open).

## 실행
```bash
bash run.sh        # deps → ImageNet 50k → 3모델×3seed×7전략×(kNN+linear-probe)
```
가벼운 확인: `python eval_v2.py --models vit_base_patch14_reg4_dinov2.lvd142m --n 5000 --seeds 3 --r_list 12 20`

## 산출물
- `results/eval_v2_seeds3.json` — 모델별·압축률별·전략별 {kNN mean±std, linear-probe, 최종 토큰수, FLOP비율}.

## 회수
`scp -r -P <port> jun@<this-host>:~/register_token_reduction/eval_v2/results ./`

## 남은 것(이 패키지 밖)
- **실제 PiToMe head-to-head**: 공식 repo(github.com/hchautran/PiToMe)를 별도 클론해 동일 축소율서 비교 필요. eval_v2의 `energy`/`energy_dyn`은 PiToMe의 보호신호 취지를 흉내낸 keep-prior 프록시일 뿐, PiToMe 방법 자체가 아님.
- 동일 토큰예산 정밀정렬(현재는 팔별 최종 토큰수를 보고해 차이를 공개).

## 파일
`eval_v2.py`(본체) · `tome_reg.py`(병합) · `prepare_data.py`(데이터) · `run.sh` · `requirements.txt`.
