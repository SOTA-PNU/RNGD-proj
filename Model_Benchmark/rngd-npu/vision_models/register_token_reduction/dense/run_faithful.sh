#!/usr/bin/env bash
# [faithful] dense(ADE20k 분할 mIoU) 재실행 — 정식 harness(prop-attn+key-metric+attn↔MLP병합).
# tome_reg_dense_faithful 사용. N_TRAIN 전량(20210), val 전량(2000). 결과 dense_miou_FAITHFUL_*.json.
set -e
cd "$(dirname "$0")"
pip install -q -r requirements.txt || true
N_TRAIN=${N_TRAIN:-20210}
echo "==> [faithful] DINOv2-reg dense (ADE20k, n_train=$N_TRAIN, 5전략)"
python dense_seg_faithful.py --model vit_base_patch14_reg4_dinov2.lvd142m \
       --n_train $N_TRAIN --n_val 2000 --epochs 60 --r_list 0 8 12 16 18 20
echo "==> [faithful] (대조) register 없는 DINOv2 (ours 제외)"
python dense_seg_faithful.py --model vit_base_patch14_dinov2.lvd142m \
       --n_train $N_TRAIN --n_val 2000 --epochs 60 --r_list 0 8 12 16 18 20 \
       --strats tome random energy highnorm || true
echo "==> faithful dense 완료. results/dense_miou_FAITHFUL_*.json"
