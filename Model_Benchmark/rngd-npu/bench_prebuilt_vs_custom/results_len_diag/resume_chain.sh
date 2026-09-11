#!/usr/bin/env bash
# 세션이 끊겨도 이어지도록 setsid 로 떼어 돌린 재개 묶음 (2026-09-11 17:0x, 세션 재시작 뒤).
#   NPU 메모리가 풀릴 때까지 대기 → 라우터 → 카운터 기록기 → 남은 5종 → E4, E5 대조 → chain.done
set -u
B=/home/jun/RNGD-proj/.claude/worktrees/max-tokens-probe/Model_Benchmark/rngd-npu/bench_prebuilt_vs_custom
PY=/home/jun/furiosa/bin/python
cd "$B"
echo "[$(date '+%F %T')] chain start"
for i in $(seq 1 60); do
  used=$(furiosa-smi status 2>/dev/null | grep -E "npu[0-3]" | awk -F'|' '{print $5}' | awk -F/ '{print $1+0}' | sort -n | tail -1)
  echo "  npu 최대 사용 ${used} GiB"
  awk -v u="${used:-99}" 'BEGIN{exit !(u < 2.0)}' && break
  sleep 10
done
pgrep -f 'furiosa_router[.]py serve' >/dev/null || bash /home/jun/.claude/jobs/35467ad1/tmp/start_router.sh
pgrep -f 'diag_sampler[.]py' >/dev/null || nohup "$PY" diag_sampler.py results_len_diag/sampler.jsonl >/dev/null 2>&1 &
echo "[$(date '+%F %T')] 재개 2: 남은 5종 (세션 재시작 뒤)" >> results_len.log
"$PY" length_probe.py Solar-Open-100B-NVFP4A16 Qwen3-Coder-30B-A3B-Instruct-FP8 gpt-oss-120b \
    K-EXAONE-236B-A23B-NVFP4A16 Qwen3-30B-A3B-FP8 >> results_len.log 2>&1
echo "[$(date '+%F %T')] probe done"
"$PY" diag.py conc Qwen3-4B-FP8 --prompt long --caps 1024 16384 1024 16384 --tag E4 > results_len_diag/diag_E4_conc.log 2>&1
"$PY" diag.py conc Qwen3-30B-A3B-Instruct-2507-FP8 --prompt long --caps 1024 16384 1024 16384 --tag E5 > results_len_diag/diag_E5_conc.log 2>&1
echo "[$(date '+%F %T')] chain done"
touch results_len_diag/chain.done
