#!/bin/bash
# run_C.sh — corrected native+kernel sampling on a live serve during sustained
# generation. Fixes Run A: perf targets only the serve pid(+live children);
# py-spy --native runs WITHOUT --nonblocking. py-spy and perf are sequenced
# (each over its own generation) to avoid mutual interference.
set -u
OUT=/home/jun/chacha/callgraph-analysis/02-dynamic
LOG=$OUT/logs
SC=$OUT/scripts
MODEL=/home/jun/chacha/qwen2.5-coder-7b-inst-tp8
PORT=12346
PYSPY=/home/jun/furiosa/bin/py-spy
S(){ echo jun | sudo -S -p '' "$@"; }

long_gen(){ # $1=outfile $2=max_tokens
  curl -s -m 300 "http://127.0.0.1:$PORT/v1/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODELID\",\"prompt\":\"Implement a fully documented LRU cache class in Python with get and put methods, then write unit tests.\",\"max_tokens\":$2,\"temperature\":0,\"stream\":false}" \
    >"$1" 2>>"$LOG/longgen_curl.log"; }

echo "[C] $(date +%T) kernel bpftrace (inference window) ..."
S bpftrace "$SC/kernel_trace.bt" >"$LOG/kernel_trace_C.log" 2>"$LOG/kernel_trace_C.err" &
sleep 4

source /home/jun/furiosa/bin/activate
export RUST_LOG=info RUST_BACKTRACE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[C] $(date +%T) launching serve on :$PORT ..."
furiosa-llm serve "$MODEL" --port "$PORT" >"$LOG/serve_C.log" 2>&1 &
SERVE_PID=$!
echo "SERVE_PID_C=$SERVE_PID" >"$LOG/pids_C.txt"

READY=0
for i in $(seq 1 150); do
  if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" -o "$LOG/models_C.json" 2>/dev/null \
     && grep -qE '"(id|data|object)"' "$LOG/models_C.json" 2>/dev/null; then READY=1; echo "[C] READY ~$((i*2))s"; break; fi
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "[C] serve exited early"; break; }
  sleep 2
done
echo "READY_C=$READY" >>"$LOG/pids_C.txt"

if [ "$READY" = 1 ]; then
  MODELID=$(python -c "import json;print(json.load(open('$LOG/models_C.json'))['data'][0]['id'])" 2>/dev/null)
  [ -z "$MODELID" ] && MODELID="$MODEL"

  # ---- Round 1: py-spy --native (blocking) over a long generation ----
  echo "[C] $(date +%T) round1: py-spy --native flamegraph ..."
  long_gen "$LOG/long_gen1.json" 512 & GEN=$!
  sleep 2
  S "$PYSPY" record --native --subprocesses -p "$SERVE_PID" -o "$LOG/pyspy_infer.svg" -d 22 \
     >"$LOG/pyspy_infer.log" 2>&1
  # instantaneous native dump mid-generation
  S "$PYSPY" dump --native -p "$SERVE_PID" >"$LOG/pyspy_dump.txt" 2>&1
  wait $GEN 2>/dev/null

  # ---- Round 2: perf -g dwarf (native+kernel) over a long generation ----
  echo "[C] $(date +%T) round2: perf record dwarf ..."
  KIDS=$(pgrep -P "$SERVE_PID" 2>/dev/null | paste -sd, -)
  TARGET="$SERVE_PID${KIDS:+,$KIDS}"
  echo "perf target=$TARGET" >>"$LOG/pids_C.txt"
  long_gen "$LOG/long_gen2.json" 512 & GEN=$!
  sleep 2
  S perf record -g --call-graph dwarf -o "$LOG/perf_infer.data" -p "$TARGET" -- sleep 22 >"$LOG/perf_infer.log" 2>&1
  wait $GEN 2>/dev/null

  echo "[C] $(date +%T) perf post-processing ..."
  S perf report -i "$LOG/perf_infer.data" --stdio --no-children 2>/dev/null | head -600 >"$LOG/perf_infer_report.txt"
  S perf report -i "$LOG/perf_infer.data" --stdio 2>/dev/null | head -600 >"$LOG/perf_infer_report_children.txt"
  S perf script -i "$LOG/perf_infer.data" 2>/dev/null | head -8000 >"$LOG/perf_infer_script.txt"
fi

echo "[C] $(date +%T) teardown ..."
kill -INT "$SERVE_PID" 2>/dev/null
for i in $(seq 1 20); do kill -0 "$SERVE_PID" 2>/dev/null || break; sleep 1; done
kill -9 "$SERVE_PID" 2>/dev/null
S pkill -INT -f kernel_trace.bt 2>/dev/null; sleep 3; S pkill -TERM -f kernel_trace.bt 2>/dev/null
S chown -R jun:jun "$LOG" 2>/dev/null
ls -la "$LOG/pyspy_infer.svg" "$LOG/perf_infer.data" 2>&1
echo "[C] DONE"
