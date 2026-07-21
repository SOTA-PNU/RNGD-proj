#!/usr/bin/env bash
# GPU 서버용: register-aware 토큰압축의 dense(분할) 평가 — ADE20k 선형 probe mIoU.
# frozen DINOv2 patch feature에 선형 seg head 1개 학습(전략 공통) → 압축률·보호전략별
# 토큰 병합·unmerge해 dense feature 복원 → mIoU. dense는 patch 정체성 중요 → register 이득 큼.
# 사용: bash run.sh   (CUDA GPU 권장)
set -e
cd "$(dirname "$0")"

echo "==> [1/3] 의존성 설치"
pip install -r requirements.txt

# N_TRAIN: ADE20k train 전량(20,210) 사용 = 표준 선형-probe 프로토콜(절대 mIoU 정상치). val은 전량 2000.
#   ADE20k train은 ~2만장뿐이라 전량이어도 특징추출 ~1분(ViT-B) + head 학습 몇 분 — ImageNet과 달리 값쌈.
#   메모리: 전량 특징 CPU ~16GB·학습시 GPU ~10GB(A100 OK). 부족하면 N_TRAIN 낮추기(빠른 시험 N_TRAIN=2000).
N_TRAIN=${N_TRAIN:-20210}
echo "==> [2/3] dense seg mIoU — DINOv2-reg (ADE20k, n_train=$N_TRAIN, 5-way 보호전략)"
python dense_seg.py --model vit_base_patch14_reg4_dinov2.lvd142m \
       --n_train $N_TRAIN --n_val 2000 --epochs 60 --r_list 0 8 12 16 18 20

echo "==> (선택) register 없는 DINOv2 대조 (ours 제외, highnorm이 대리)"
python dense_seg.py --model vit_base_patch14_dinov2.lvd142m \
       --n_train $N_TRAIN --n_val 2000 --epochs 60 --r_list 0 8 12 16 18 20 \
       --strats tome random energy highnorm || true

echo "==> [3/3] 압축률 sweep 그림"
for j in results/dense_miou_*.json; do python make_sweep_figure.py "$j"; done

echo "==> 완료. results/ 의 *.json·*.png 를 로컬로 회수(scp)."
echo "    scp -r -P <port> jun@<this-host>:~/register_token_reduction/dense/results ./"
