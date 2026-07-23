#!/usr/bin/env bash
# ── [②] 공식 PiToMe 알고리즘을 그들 env(timm==0.4.12)로 DeiT 에서 직접 실행 → 우리 포팅과 실측 대조.
#    비대화형: 공식 repo 클론 + 전용 conda env 자동 생성 + 우리 로컬 val 로 평가 + 리포트.
#    ★ gated imagenet-1k / HF 토큰 불필요 — 공식 '알고리즘 코드'만 불러 우리 val 로 돌린다.
#    필요: conda(스크립트가 env 자동 생성), CUDA GPU, 인터넷(클론+체크포인트).
set -e
cd "$(dirname "$0")"
HERE="$(pwd)"
REPO="$HERE/_official_pitome"
ENV="pitome_off"
RATIOS="0.975 0.95 0.925 0.9"

# (선택) HF rate-limit 완화용 토큰. ★ 여기 붙여넣으세요 ★ — 없어도 됩니다(gated 아님):
#   deit_compare/hf_token.txt 파일에 토큰 한 줄만 넣으면 자동 사용.
if [ -f "$HERE/hf_token.txt" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HERE/hf_token.txt")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  echo "==> HF_TOKEN 사용(rate-limit 완화). (필수 아님)"
fi

command -v conda >/dev/null 2>&1 || { echo "conda 가 필요합니다(공식 env 격리용). miniconda 설치 후 재실행."; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "==> [1/5] 공식 repo 클론"
[ -d "$REPO/algo" ] || git clone --depth 1 https://github.com/hchautran/PiToMe "$REPO"

echo "==> [2/5] 전용 conda env '$ENV' (timm==0.4.12 격리; 우리 최신 env 와 안 섞임)"
conda env list | awk '{print $1}' | grep -qx "$ENV" || conda create -y -n "$ENV" python=3.10
conda run -n "$ENV" python -c "import torch" 2>/dev/null \
  || conda install -y -n "$ENV" pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
# 공식 의존성에서 salesforce-lavis(ITR용, 무거움/자주깨짐)·wandb 제외하고 설치 → algo import 에 충분
grep -vE 'salesforce-lavis|wandb' "$REPO/requirements.txt" > "$HERE/_req_trim.txt"
conda run -n "$ENV" pip install -q -r "$HERE/_req_trim.txt"
# 안전망: lavis 를 뺐으므로 algo 가 흔히 쓰는 einops/scipy 를 보강(이미 있으면 무시)
conda run -n "$ENV" pip install -q einops scipy || true

echo "==> [3/5] ImageNet val 준비(재사용/비-gated 미러)"
DATA="$HERE/imagenet_val"
if [ ! -f "$DATA/DONE" ]; then
  if [ -f ../pitome_compare/imagenet_val/DONE ]; then
    DATA="$(cd ../pitome_compare/imagenet_val && pwd)"; echo "    ../pitome_compare/imagenet_val 재사용"
  else
    echo "    새로 다운로드"; conda run -n "$ENV" python prepare_data.py --split val
  fi
fi

echo "==> [4/5] 공식 알고리즘으로 DeiT-S · DeiT-T 실행 (우리 val, 공식 전처리)"
for M in deit_small_patch16_224 deit_tiny_patch16_224; do
  conda run -n "$ENV" python official_deit_driver.py \
    --repo "$REPO" --model "$M" --data_root "$DATA" --n_val 50000 --ratio_list $RATIOS
done

echo "==> [5/5] ①(우리 포팅) ↔ ②(공식) 대조 리포트"
python compare_report.py    # stdlib 만 씀(현재 env 로 OK)

echo ""
echo "==> 완료. results/official__*.json 저장, 위 대조표 판정 참고."
echo "    ①(우리) 결과가 없으면 먼저 bash run.sh 실행 후 이 스크립트 재실행(또는 bash run_all.sh)."
