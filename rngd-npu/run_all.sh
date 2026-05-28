#!/usr/bin/env bash
# 전체 벤치마크 파이프라인 (Furiosa RNGD / furiosa-llm).
# 사용: STAGE=gen ./run_all.sh   (STAGE: preflight|smoke|gen|embed|swebench|report|all)
set -uo pipefail
cd "$(dirname "$0")"
[ -f ~/furiosa/bin/activate ] && source ~/furiosa/bin/activate

CONFIG=${CONFIG:-configs/models.yaml}
STAGE=${STAGE:-all}
LOG_DIR=results/_run_logs
PY=run_all                                # py 파일들 위치
mkdir -p "$LOG_DIR"
log() { echo "[$(date +'%F %T')] $*" | tee -a "$LOG_DIR/run_all.log"; }

step_preflight() { log "=== preflight ==="; bash preflight.sh 2>&1 | tee "$LOG_DIR/preflight.log"; }

step_smoke()  { log "=== smoke (Qwen2.5-0.5B) ===";  python -u "$PY/orchestrator.py" "$CONFIG" --tasks tps --models Qwen2.5-0.5B 2>&1 | tee "$LOG_DIR/smoke.log"; }
step_gen()    { log "=== gen (tps + sweep + memsweep) ==="; python -u "$PY/orchestrator.py" "$CONFIG" --tasks tps,sweep,memsweep 2>&1 | tee "$LOG_DIR/gen.log"; }
step_embed()  { log "=== embed + rerank ==="; python -u "$PY/orchestrator.py" "$CONFIG" --tasks embed,rerank 2>&1 | tee "$LOG_DIR/embed.log"; }
step_swebench() {
  log "=== SWE-bench 추론 + 채점 ==="
  command -v docker >/dev/null 2>&1 || { log "docker 없음 — 스킵"; return; }
  python -c "import swebench" 2>/dev/null || pip install swebench
  python -u "$PY/orchestrator.py" "$CONFIG" --tasks swebench 2>&1 | tee "$LOG_DIR/swebench.log"
  bash eval_swebench.sh 2>&1 | tee "$LOG_DIR/swebench_eval.log" || true
}
step_report() {
  log "=== 집계 + 리포트 ==="
  python "$PY/analyze.py" --csv "$LOG_DIR/summary.csv" 2>&1 | tee "$LOG_DIR/analyze.log"
  python "$PY/report.py" 2>&1 | tee "$LOG_DIR/report.log" || true
}

case "$STAGE" in
  preflight) step_preflight ;;
  smoke)     step_preflight; step_smoke ;;
  gen)       step_preflight; step_smoke; step_gen ;;
  embed)     step_embed ;;
  swebench)  step_swebench ;;
  report)    step_report ;;
  all)       step_preflight; step_smoke; step_gen; step_embed; step_swebench; step_report ;;
  *) echo "unknown STAGE=$STAGE (preflight|smoke|gen|embed|swebench|report|all)"; exit 1 ;;
esac
log "done."
