#!/usr/bin/env bash
# (선택) 공식 PiToMe 리포 독립 교차검증. compare.py 의 pitome_step 이 공식 selection 을 옳게 옮겼는지,
# '그들 환경·그들 지원 모델(DeiT)'에서 그들 코드로 직접 재현해 대조한다.
# ── 왜 여기서 DINOv2-reg 를 안 돌리나: 공식 리포는 timm==0.4.12 핀(구버전)이라 DINOv2-reg(timm>=1.0 필요)를
#    로드 못 한다. 따라서 '같은-모델(DINOv2-reg) head-to-head' 는 compare.py 가 담당(공식 merge.py 를 소스대로 포팅),
#    이 스크립트는 '공식 코드가 자기 모델에서 내는 수치 재현' 이라는 독립 sanity check 역할.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HERE/_official_pitome"

echo "=== [1/4] 공식 리포 클론 (github.com/hchautran/PiToMe) ==="
[ -d "$REPO" ] || git clone --depth 1 https://github.com/hchautran/PiToMe "$REPO"
cd "$REPO"

echo "=== [2/4] 전용 conda env (그들 핀 그대로: timm==0.4.12 등) ==="
cat <<'EOF'
  # 공식 설치 절차 (README 그대로):
  conda create -n pitome python=3.10 -y && conda activate pitome
  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
  pip install -r requirements.txt      # timm==0.4.12, datasets, accelerate, ml_collections ...
  # (salesforce-lavis 는 ITR용 — IC만 볼거면 실패해도 무방)
EOF
read -p "위 env 를 이미 만들고 activate 했으면 Enter, 아니면 Ctrl-C: " _

echo "=== [3/4] ImageNet val 배치: data/ic/ 아래에 ILSVRC val + imagenet_class_index.json ==="
cat <<'EOF'
  기대 구조 (tasks/ic/utils.py: DATA_PATH={cwd}/data/ic, class ImageNetKaggle):
    PiToMe/data/ic/
      ILSVRC/Data/CLS-LOC/val/*.JPEG
      imagenet_class_index.json
      LOC_val_solution.csv         # (Kaggle ImageNet-Object-Localization 포맷)
  ※ 이미 표준 ImageNet val 이 있으면 심볼릭 링크로 맞춰도 됨.
EOF

echo "=== [4/4] pitome vs tome, 같은 ratio 로 평가 (DeiT-base/224) ==="
# eval_ic.sh ARCH SIZE INPUT RATIO ALGO  →  main_ic.py --model deit-base-224 --algo <> --ratio <> --eval
for RATIO in 0.95 0.925 0.9 0.875; do
  for ALGO in pitome tome; do
    echo "--- $ALGO ratio=$RATIO ---"
    bash scripts/eval_scripts/eval_ic.sh deit base 224 $RATIO $ALGO || echo "(위 조합 실패 — 로그 확인)"
  done
done
echo "=== 완료. 공식 pitome 곡선이 우리 compare.py 의 pitome 곡선과 같은 추세면 포팅 검증 OK ==="
echo "   (DeiT 는 register 없음 → 여기 pitome 절대수치는 그들 논문 재현용, register head-to-head 는 compare.py)"
