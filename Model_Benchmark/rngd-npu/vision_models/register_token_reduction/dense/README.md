# dense — register-aware 토큰압축의 분할(segmentation) 평가 (GPU 패키지)

이 폴더는 GPU 서버에 복사해 실행하는, "register-aware 토큰 압축" 논문의 **dense 실험**입니다. 분류(kNN)에 이어, **공간 구조가 중요한 분할(segmentation) mIoU**에서 register 보호의 이득이 더 큼을 보입니다.

## 왜 dense인가 (가장 강한 추가 증거)
분류는 이미지 1개당 벡터 1개라 토큰을 좀 잃어도 견딥니다. 분할은 **patch마다 예측**이 필요해 토큰 정체성이 중요합니다. register가 전역 정보를 들고 있으니, 이를 지키며 압축하면 dense에서 이득이 분류보다 큽니다. (로컬 예비: 91% 압축서 dense feature 충실도 ours 0.71 vs tome 0.49 vs energy 0.53 — 분류 격차보다 큼.)

## 방법
1. **선형 seg head 학습(전략 공통):** frozen DINOv2의 patch feature(무압축)에 선형 분류기 1개를 ADE20k train에 학습(150클래스, 0=미표기 무시). 모든 전략이 같은 head를 씀(공정).
2. **압축·복원·평가:** 각 압축률·보호전략에서 토큰을 size-가중 ToMe로 병합하되, **원본 patch→최종 토큰 매핑을 층마다 추적(unmerge)**해 patch 격자 feature를 복원 → head 적용 → 이미지 해상도로 upsample → mIoU.
3. 보호전략: `tome`(CLS만)·`ours`(CLS+register)·`random`·`energy`(PiToMe식)·`highnorm`.

`tome_reg_dense.py`의 `merge_step_track`은 상위 `tome_reg.py`의 병합과 **수치적으로 동일**함을 로컬 검증했고(diff 0), r=0에서 dense==full(항등)도 확인했습니다.

## 실행
```bash
bash run.sh      # 의존성 → ADE20k(scene_parse_150) 스트리밍 → head 학습 → mIoU → 그림
```
개별:
```bash
python dense_seg.py --model vit_base_patch14_reg4_dinov2.lvd142m --n_train 2000 --n_val 2000 --epochs 60 --r_list 0 8 12 16 18 20
python make_sweep_figure.py results/dense_miou_vit_base_patch14_reg4_dinov2.json
```

## 산출물
- `results/dense_miou_<model>.json` — 압축률별 전략별 mIoU.
- `results/dense_miou_<model>.png` — 압축률 vs mIoU sweep.

## 회수
```bash
scp -r -P <port> jun@<this-host>:~/register_token_reduction/dense/results ./
```

## 주의 / 설계 결정
- **patch 격자 16×16**(DINOv2 patch14 @ 224). 저해상도 probing이라 mIoU 절대값은 표준 세팅보다 낮을 수 있으나, **전략 간 상대 비교는 유효**합니다. 더 높이려면 `IMG`를 518 등으로(비용↑).
- ADE20k는 `datasets`로 **스트리밍**(별도 다운로드 스크립트 불필요). train/val 각 2000장 기본(늘리면 안정↑).
- head는 **무압축 feature에 1회 학습** 후 고정 — "압축된 feature가 분할 정보를 얼마나 보존하나"를 측정.
- register 없는 모델(plain DINOv2)은 `ours`가 degenerate(=tome)이라 `--strats tome random energy highnorm`로 실행, `highnorm`이 대리.
- 기대: **극단 압축(>90%)서 `ours` mIoU가 나머지를 크게 상회** → dense에서 register 우위 확정.

## 파일
- `tome_reg_dense.py` — 추적/unmerge 가능한 size-가중 병합.
- `dense_seg.py` — ADE20k 선형 probe mIoU 본체.
- `make_sweep_figure.py` — 그림.
- `run.sh` / `requirements.txt`.
