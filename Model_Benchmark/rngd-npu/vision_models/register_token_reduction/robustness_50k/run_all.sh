#!/usr/bin/env bash
# robustness_50k 전체 50k 실행. 서버에서: bash run_all.sh
set -e
cd "$(dirname "$0")"
N="${1:-50000}"
echo "== reg_count_sweep (k 스윕 + 부트스트랩 CI) =="
python reg_count_sweep.py "$N"   | tee reg_count_sweep_50k.log
echo "== faithful_tome_h2h (정식 ToMe vs Ours) =="
python faithful_tome_h2h.py "$N" | tee faithful_tome_50k.log
echo "== retrieval_map (검색 mAP, 두 번째 지표) =="
python retrieval_map.py "$N"     | tee retrieval_map_50k.log
# NOTE: PiToMe 비교는 pitome_compare/(층별 margin, 정식·throughput 포함)에서 이미 50k 완료.
#       여기 pitome_h2h.py(constant-margin)는 구버전이라 run_all에서 제외(모순 방지). 필요시 개별 실행.
echo "완료. *_50k.log 3개를 로컬로 scp 하세요 (README 참고)."
