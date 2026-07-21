#!/usr/bin/env bash
# ── 전부 자동(GPU 서버 한 방): ①우리 포팅 + ②공식 repo 실측 + 대조 리포트.
#    ① 은 현재 env(timm>=1.0), ② 는 스크립트가 만드는 전용 conda env(timm==0.4.12) 에서 격리 실행.
#    gated 데이터/HF 토큰 불필요. conda + CUDA GPU + 인터넷만 있으면 됨.
set -e
cd "$(dirname "$0")"

echo "############### ① 우리 포팅 (현재 env) ###############"
bash run.sh

echo ""
echo "############### ② 공식 repo 실측 (전용 conda env) ###############"
bash run_official_pitome.sh

echo ""
echo "############### 최종 대조 (①↔②) ###############"
python compare_report.py
echo ""
echo "끝. results/ 에 ours_port__*.json · official__*.json 저장. 위 표의 '판정' 참고."
