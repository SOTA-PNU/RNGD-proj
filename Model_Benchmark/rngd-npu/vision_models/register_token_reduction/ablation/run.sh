#!/usr/bin/env bash
# GPU 서버용: register keep-prior "결정적 ablation" 풀스케일.
# 같은 size-가중 ToMe 병합에서 '무엇을 보호하느냐'만 바꿔 비교:
#   tome(CLS만) / ours(register) / random / energy(PiToMe식 keep-prior) / highnorm
# main.md의 [TODO: ablation register vs energy vs random] 과 SPEC '결정적 ablation'을 채운다.
# 사용: bash run.sh   (CUDA GPU 권장)
set -e
cd "$(dirname "$0")"
export IMAGENET_VAL="$(pwd)/imagenet_val"

echo "==> [1/4] 의존성 설치"
pip install -r requirements.txt

echo "==> [2/4] ImageNet val 50k 준비 (클래스당 50장)"
[ -f imagenet_val/DONE ] || python prepare_data.py --per_class 50

echo "==> [3/4] ablation 실행"
# 주력: DINOv2-reg (register 보유). ours=register 보호.
python eval_ablation.py --model vit_base_patch14_reg4_dinov2.lvd142m \
       --n 50000 --batch 128 --k 20 --r_list 8 12 16 18 20
# 일반성(register 없는 모델): ours 행은 tome와 동일(degenerate)이라 'highnorm'이 ours의 대리.
python eval_ablation.py --model vit_base_patch16_clip_224.openai \
       --n 50000 --batch 128 --k 20 --r_list 8 12 16 18 20 \
       --strats tome random energy highnorm || true
python eval_ablation.py --model vit_base_patch14_dinov2.lvd142m \
       --n 50000 --batch 128 --k 20 --r_list 8 12 16 18 20 \
       --strats tome random energy highnorm || true

echo "==> [4/4] 압축률 sweep 그림"
for j in results/ablation_*.json; do python make_sweep_figure.py "$j"; done

echo "==> 완료. results/ 의 *.json·*.png 를 로컬로 회수하세요(scp)."
echo "    예) scp -r -P <port> jun@<this-host>:~/register_token_reduction/ablation/results ./"
