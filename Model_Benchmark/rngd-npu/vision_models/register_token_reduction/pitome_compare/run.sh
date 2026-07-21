#!/usr/bin/env bash
# PiToMe(공식 에너지) vs ToMe vs Ours — 같은-예산 정확도 + throughput. GPU 서버에서 실행.
# 기본 프로토콜 = val leave-one-out k-NN(갤러리=쿼리=val 5만, 자기 제외) — 논문 전 실험과 동일, train 다운로드 불필요.
# (정통 train-갤러리로 승급하려면: prepare_data.py --split train 후 compare.py --gallery train)
set -e
cd "$(dirname "$0")"

echo "=== [0/3] 의존성 ==="
pip install -q -r requirements.txt

echo "=== [1/3] 데이터 준비 (val 5만, 최초 1회) ==="
[ -f imagenet_val/DONE ] || python prepare_data.py --split val

echo "=== [2/3] 정확도 (val leave-one-out k-NN, val 5만, k=20; 세 방법 모두 우리가 직접 재측정) ==="
python compare.py --mode acc --gallery val --r_list 8 12 16 18 20 | tee results_acc.txt

echo "=== [3/3] Throughput (im/s, r=0=무압축 기준선) ==="
python compare.py --mode tput --batch 128 --iters 50 --r_list 0 8 12 16 18 20 | tee results_tput.txt

echo "=== 완료. results_acc.txt / results_tput.txt 확인 ==="
echo "   (선택) 공식 PiToMe 리포 교차검증: bash run_official_pitome.sh"
echo "   (선택) 정통 train-갤러리 승급: python prepare_data.py --split train --per_class 1300 && python compare.py --mode acc --gallery train"
