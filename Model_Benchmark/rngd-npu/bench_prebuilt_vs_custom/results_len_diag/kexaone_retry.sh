#!/usr/bin/env bash
# K-EXAONE 재시도 (2026-09-12). 09-11 시도는 적재 중 호스트 메모리가 바닥나 OOM 으로 serve 가 죽었다
# (가중치 136.6 GiB > 메모리 125 GiB, oom_kill 1). 그 사이 스왑이 8 GB 에서 207 GB 로 늘어 다시 해 볼 만하다.
#   1) 모델 파일 페이지 캐시를 내려 여유 확보
#   2) 라우터 기동, 커널 카운터 기록 시작
#   3) serve 가 뜨면 oom_score_adj 를 올린다 — 또 바닥나도 커널이 남의 프로세스 대신 이 serve 를 고르게
#   4) 다른 12종과 같은 조건(max_tokens 16384)으로 측정
set -u
B=/home/jun/RNGD-proj/.claude/worktrees/max-tokens-probe/Model_Benchmark/rngd-npu/bench_prebuilt_vs_custom
PY=/home/jun/furiosa/bin/python
cd "$B"
rm -f results_len_diag/kexaone.done
echo "[$(date '+%F %T')] 캐시 내리기"
"$PY" diag_evict.py > results_len_diag/kexaone_evict.log 2>&1
cat results_len_diag/kexaone_evict.log
pgrep -f 'furiosa_router[.]py serve' >/dev/null || bash /home/jun/.claude/jobs/35467ad1/tmp/start_router.sh > results_len_diag/kexaone_router.log 2>&1
pgrep -f 'diag_sampler[.]py' >/dev/null || nohup "$PY" diag_sampler.py results_len_diag/sampler.jsonl >/dev/null 2>&1 &
(
  for i in $(seq 1 900); do
    p=$(pgrep -f 'furiosa-llm serve furiosa-ai/K-EXAONE' | head -1)
    if [ -n "${p:-}" ]; then
      echo 500 > "/proc/$p/oom_score_adj" 2>/dev/null && echo "[$(date '+%F %T')] oom_score_adj=500 on $p"
      break
    fi
    sleep 2
  done
) &
echo "[$(date '+%F %T')] 재시도: K-EXAONE (스왑 207 GB)" >> results_len.log
"$PY" length_probe.py K-EXAONE-236B-A23B-NVFP4A16 >> results_len.log 2>&1
echo "[$(date '+%F %T')] 끝, oom_kill=$(awk '/^oom_kill/{print $2}' /proc/vmstat)"
touch results_len_diag/kexaone.done
