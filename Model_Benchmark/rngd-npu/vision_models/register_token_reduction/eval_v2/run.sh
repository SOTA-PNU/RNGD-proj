#!/usr/bin/env bash
# GPU 서버용: register-aware 토큰축소 '강화 평가'(감사 지적 보완).
# 추가: 다중 seed(오차막대)·linear-probe(표준지표)·register 모델 3종·동적 재선택 keep-prior·FLOP/토큰수.
# 사용: bash run.sh   (CUDA GPU 권장; 50k×3모델×3seed×7전략은 시간 걸림 — 필요시 --n 축소)
set -e
cd "$(dirname "$0")"
export IMAGENET_VAL="$(pwd)/imagenet_val"

echo "==> [1/3] deps"; pip install -r requirements.txt
echo "==> [2/3] ImageNet val 50k"; [ -f imagenet_val/DONE ] || python prepare_data.py --per_class 50
echo "==> [3/3] 강화 평가 (3 register 모델 × 3 seed × 정적+동적 keep-prior × kNN+linear-probe)"
python eval_v2.py \
  --models vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m vit_large_patch14_reg4_dinov2.lvd142m \
  --n 50000 --batch 128 --seeds 3 --r_list 8 12 16 18 20 \
  --strats tome ours random energy highnorm energy_dyn highnorm_dyn --linear_probe

echo "==> 완료. results/eval_v2_seeds3.json 를 로컬로 회수:"
echo "    scp -r -P <port> jun@<this-host>:~/register_token_reduction/eval_v2/results ./"
