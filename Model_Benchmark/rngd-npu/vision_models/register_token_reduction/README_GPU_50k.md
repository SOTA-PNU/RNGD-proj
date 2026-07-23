# 전체 50k 실험 한 번에 돌리기 (GPU 서버)

이 문서는 논문 보조 실험들을 GPU 서버에서 전체 데이터셋으로 확정하기 위해, 무엇을 전송하고 무엇을 어떤 순서로 돌리는지 한곳에 모은 안내서입니다.

## 한눈에 — 어떤 결과가 어디서 나오나

| 논문에 넣을 결과 | 폴더 | 전체셋 의미 |
|---|---|---|
| 동적 재선택 비교 + 정적 ablation + linear-probe + 다seed | `eval_v2/` | ImageNet val 50,000 |
| 실제 PiToMe head-to-head | `robustness_50k/` | ImageNet val 50,000 |
| 정식 ToMe head-to-head | `robustness_50k/` | ImageNet val 50,000 |
| 보호 register 개수 스윕 + 부트스트랩 CI | `robustness_50k/` | ImageNet val 50,000 |
| dense(분할) mIoU | `dense/` | ADE20k val 2,000 = **이미 전체** |

즉 세 폴더(`eval_v2`, `robustness_50k`, `dense`)만 서버에 있으면 다섯 결과가 전부 전체셋으로 나옵니다.

## 1단계 — 전송 (로컬에서 실행, 비번 jun)

원격 폴더를 만들면서 보냅니다. dense는 이미 서버에서 한 번 돌리셨다면 생략 가능합니다.

```bash
BASE=~/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/register_token_reduction
DST=jun@164.125.249.13

# robustness_50k (신규 — 서버에 없음)
ssh -p 10022 $DST 'mkdir -p ~/register_token_reduction/robustness_50k'
scp -r -P 10022 $BASE/robustness_50k/* $DST:~/register_token_reduction/robustness_50k/

# eval_v2 (최신본으로 덮어쓰기 — 아래 주의 참고)
ssh -p 10022 $DST 'mkdir -p ~/register_token_reduction/eval_v2'
scp -r -P 10022 $BASE/eval_v2/* $DST:~/register_token_reduction/eval_v2/

# dense (이미 돌렸으면 생략)
ssh -p 10022 $DST 'mkdir -p ~/register_token_reduction/dense'
scp -r -P 10022 $BASE/dense/* $DST:~/register_token_reduction/dense/
```

> ⚠️ **eval_v2는 반드시 최신본으로 덮어쓰세요.** linear-probe 버그(클래스 분리 split·NCM) 수정이 로컬 최신본에만 있습니다. 서버가 옛 버전이면 linear-probe가 0으로 나옵니다.

## 2단계 — 실행 (서버에서)

```bash
cd ~/register_token_reduction

# (a) 동적 재선택 + 정적 ablation + linear-probe + 3seed  — base 모델 전체 50k
python eval_v2/eval_v2.py --models vit_base_patch14_reg4_dinov2.lvd142m \
  --n 50000 --seeds 3 --r_list 8 12 16 18 20 --linear_probe | tee eval_v2_base_50k.log

# (b) PiToMe·정식 ToMe·register 스윕  — 전체 50k, 한 번에
cd robustness_50k && bash run_all.sh && cd ..

# (c) dense mIoU (val은 이미 전체 2000, probe 학습만 강화)
python dense/dense_seg.py --n_train 8000 --n_val 2000 \
  --r_list 0 8 12 16 18 20 | tee dense_50k.log
```

세 모델(small/base/large-reg) 전부 보려면 (a)에서 `--models` 인자를 빼면 기본 3모델을 돕니다(더 오래 걸림).

## 3단계 — 결과 회수 (로컬에서)

```bash
DST=jun@164.125.249.13
LOCAL=~/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/register_token_reduction
scp -P 10022 $DST:'~/register_token_reduction/*_50k.log' $LOCAL/results/
scp -P 10022 $DST:'~/register_token_reduction/robustness_50k/*_50k.log' $LOCAL/robustness_50k/
scp -P 10022 $DST:'~/register_token_reduction/eval_v2/results/*.json' $LOCAL/eval_v2/results/
scp -P 10022 $DST:'~/register_token_reduction/dense/results/*.json' $LOCAL/dense/results/
```

가져온 `.log`/`.json`을 주시면 논문 표와 PPT를 전체 50k 수치로 갱신합니다.

## 참고

- 모든 스크립트는 GPU 있으면 자동 사용, 없으면 CPU. 50k는 GPU에서 실험당 수십 분입니다.
- dense의 ADE20k 검증셋은 원래 2,000장이라 `--n_val 2000`이 이미 전체 검증입니다.
