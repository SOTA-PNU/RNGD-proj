#!/usr/bin/env bash
# ── [쉬운 경로] 우리 PiToMe/ToMe 포팅을 DeiT 에서 실행 → 압축별 top-1 곡선.
#    공식 논문 공개 수치(DeiT-S/T)와 대조하면 포팅 충실성 1차 검증. (공식 repo 불필요)
#    GPU 권장. 우리 최신 env(timm>=1.0)에서 그대로 돎.
set -e
cd "$(dirname "$0")"

echo "==> [1/3] 의존성"
pip install -r requirements.txt

echo "==> [2/3] ImageNet val 준비 (기존 것 있으면 재사용)"
if [ ! -f imagenet_val/DONE ]; then
  # pitome_compare 에서 이미 받아둔 val 이 있으면 심볼릭 링크로 재사용(재다운로드 회피)
  if [ -f ../pitome_compare/imagenet_val/DONE ]; then
    echo "    ../pitome_compare/imagenet_val 재사용(심볼릭 링크)"
    ln -sfn ../pitome_compare/imagenet_val imagenet_val
  else
    echo "    새로 다운로드(HF non-gated 미러)"
    python prepare_data.py --split val
  fi
fi

echo "==> [3/3] DeiT-S · DeiT-T 교차검증 (공식과 같은 ratio·같은 지표=분류 top-1)"
# DeiT-S: 공개 참조 있음(baseline 79.8, PiToMe−ToMe≈+1.4). 가장 중요한 대조.
python deit_compare.py --model deit_small_patch16_224 --n_val 50000 --ratio_list 0.975 0.95 0.925 0.9
echo ""
# DeiT-T: 두 번째 참조점(baseline 72.3, PiToMe−ToMe≈+1.9).
python deit_compare.py --model deit_tiny_patch16_224  --n_val 50000 --ratio_list 0.975 0.95 0.925 0.9

echo ""
echo "==> 대조 리포트(우리 곡선 + 공개 참조)"
python compare_report.py

echo ""
echo "==> 완료. r=0 정확도가 공식 baseline 과 맞고(파이프라인 OK),"
echo "    (pitome−tome) 격차 추세가 공개 참조와 같으면 포팅 충실성 1차 검증 완료."
echo "    공식 repo 실측 대조까지 = bash run_official_pitome.sh (또는 처음부터 bash run_all.sh)."
