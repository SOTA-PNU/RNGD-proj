#!/bin/bash
# run_A.sh — live multi-layer dynamic capture of `furiosa-llm serve`
# Layers: eBPF kernel (bpftrace) | native+kernel sampling (perf) | py+native sampling (py-spy)
#         | native backtraces (gdb) | OpenAI request firing (curl)
# Self-sequences with internal sleeps; safe to run backgrounded as user `jun`.
set -u
OUT=/home/jun/chacha/callgraph-analysis/02-dynamic
LOG=$OUT/logs
SC=$OUT/scripts
MODEL=/home/jun/chacha/qwen2.5-coder-7b-inst-tp8
PORT=12345
PYSPY=/home/jun/furiosa/bin/py-spy
S(){ echo jun | sudo -S -p '' "$@"; }

echo "[A] $(date +%T) starting kernel bpftrace ..."
S bpftrace "$SC/kernel_trace.bt" >"$LOG/kernel_trace.log" 2>"$LOG/kernel_trace.err" &
sleep 4   # let probes attach

echo "[A] $(date +%T) launching serve ..."
source /home/jun/furiosa/bin/activate
export RUST_LOG=info RUST_BACKTRACE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
furiosa-llm serve "$MODEL" --port "$PORT" >"$LOG/serve.log" 2>&1 &
SERVE_PID=$!
echo "SERVE_PID=$SERVE_PID" | tee "$LOG/pids.txt"

echo "[A] $(date +%T) waiting for readiness ..."
READY=0; GOT_LOAD_SNAP=0
for i in $(seq 1 150); do
  if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" -o "$LOG/models.json" 2>/dev/null \
     && grep -qE '"(id|data|object)"' "$LOG/models.json" 2>/dev/null; then
    READY=1; echo "[A] $(date +%T) READY after ~$((i*2))s"; break
  fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then echo "[A] $(date +%T) serve EXITED EARLY (see serve.log)"; break; fi
  # mid-load native backtrace (captures NativeLLMEngine __init__ / weight load in the act)
  if [ "$i" = 5 ] && [ "$GOT_LOAD_SNAP" = 0 ]; then
    echo "[A] gdb load snapshot #1 ..."
    S gdb -p "$SERVE_PID" -batch -nx -ex "set pagination off" -ex "set confirm off" \
      -ex "info threads" -ex "thread apply all bt" -ex "detach" -ex "quit" \
      >"$LOG/gdb_load_10s.txt" 2>&1
    GOT_LOAD_SNAP=1
  fi
  if [ "$i" = 12 ]; then
    echo "[A] gdb load snapshot #2 ..."
    S gdb -p "$SERVE_PID" -batch -nx -ex "set pagination off" -ex "set confirm off" \
      -ex "thread apply all bt" -ex "detach" -ex "quit" >"$LOG/gdb_load_24s.txt" 2>&1
  fi
  sleep 2
done
echo "READY=$READY" | tee -a "$LOG/pids.txt"

# process tree + open NPU fds (definitive device mapping)
PGID=$(ps -o pgid= -p "$SERVE_PID" 2>/dev/null | tr -d ' ')
PIDS=$(pgrep -g "$PGID" 2>/dev/null | paste -sd, -)
echo "PGID=$PGID PIDS=$PIDS" | tee -a "$LOG/pids.txt"
{ echo "== ps -T (threads) =="; ps -L -o pid,tid,comm,psr -p "$SERVE_PID";
  echo "== pgrep -g $PGID =="; pgrep -ag "$PGID"; } >"$LOG/serve_pstree.txt" 2>&1
{ echo "== /proc/$SERVE_PID/fd -> rngd nodes =="; S ls -l /proc/$SERVE_PID/fd 2>/dev/null | grep -iE 'rngd|npu';
  for p in ${PIDS//,/ }; do echo "-- pid $p --"; S ls -l /proc/$p/fd 2>/dev/null | grep -iE 'rngd|npu'; done; } >"$LOG/npu_fds.txt" 2>&1

if [ "$READY" = 1 ]; then
  echo "[A] $(date +%T) gdb idle (steady-state thread structure) ..."
  S gdb -p "$SERVE_PID" -batch -nx -ex "set pagination off" -ex "set confirm off" \
    -ex "info threads" -ex "thread apply all bt" -ex "detach" -ex "quit" \
    >"$LOG/gdb_idle.txt" 2>&1

  MODELID=$(python -c "import json;print(json.load(open('$LOG/models.json'))['data'][0]['id'])" 2>/dev/null)
  [ -z "$MODELID" ] && MODELID="$MODEL"
  echo "MODELID=$MODELID" | tee -a "$LOG/pids.txt"

  echo "[A] $(date +%T) starting inference-window samplers (perf dwarf + py-spy native, 30s) ..."
  S perf record -g --call-graph dwarf -o "$LOG/perf_infer.data" -p "$PIDS" -- sleep 30 \
     >"$LOG/perf_infer.log" 2>&1 &
  PERF_W=$!
  S "$PYSPY" record --native --nonblocking --subprocesses -p "$SERVE_PID" \
     -o "$LOG/pyspy_infer.svg" -d 30 >"$LOG/pyspy_infer.log" 2>&1 &
  PYSPY_W=$!
  sleep 4   # let samplers attach

  echo "[A] $(date +%T) firing chat completion (non-stream) ..."
  curl -s -m 150 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODELID\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function fib(n) that returns the nth Fibonacci number, with a short docstring.\"}],\"max_tokens\":160,\"temperature\":0,\"stream\":false}" \
    >"$LOG/chat_response.json" 2>"$LOG/chat_curl.log"; echo "[A] chat rc=$?"

  echo "[A] $(date +%T) firing streaming completion ..."
  curl -s -N -m 150 "http://127.0.0.1:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODELID\",\"prompt\":\"def quicksort(arr):\\n\",\"max_tokens\":96,\"temperature\":0,\"stream\":true}" \
    >"$LOG/completion_stream.txt" 2>>"$LOG/chat_curl.log"; echo "[A] stream rc=$?"

  # gdb during a 3rd long generation to catch the native execute/stream stack mid-flight
  ( sleep 1; S gdb -p "$SERVE_PID" -batch -nx -ex "set pagination off" -ex "set confirm off" \
       -ex "thread apply all bt" -ex "detach" -ex "quit" >"$LOG/gdb_infer.txt" 2>&1 ) &
  curl -s -m 150 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODELID\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain and implement merge sort in Python with comments.\"}],\"max_tokens\":256,\"temperature\":0}" \
    >"$LOG/chat_response2.json" 2>>"$LOG/chat_curl.log"; echo "[A] chat2 rc=$?"

  wait "$PERF_W" 2>/dev/null
  wait "$PYSPY_W" 2>/dev/null

  echo "[A] $(date +%T) post-processing perf ..."
  S perf report -i "$LOG/perf_infer.data" --stdio --no-children 2>/dev/null | head -500 >"$LOG/perf_infer_report.txt"
  S perf report -i "$LOG/perf_infer.data" --stdio 2>/dev/null | head -500 >"$LOG/perf_infer_report_children.txt"
  S perf script -i "$LOG/perf_infer.data" 2>/dev/null | head -4000 >"$LOG/perf_infer_script.txt"
fi

echo "[A] $(date +%T) shutting down serve (SIGINT) ..."
kill -INT "$SERVE_PID" 2>/dev/null
for i in $(seq 1 20); do kill -0 "$SERVE_PID" 2>/dev/null || break; sleep 1; done
kill -9 "$SERVE_PID" 2>/dev/null

echo "[A] $(date +%T) stopping bpftrace ..."
S pkill -INT -f kernel_trace.bt 2>/dev/null
sleep 3
S pkill -TERM -f kernel_trace.bt 2>/dev/null

S chown -R jun:jun "$LOG" 2>/dev/null
echo "[A] $(date +%T) NPU state after teardown:"
furiosa-smi info 2>&1 | grep -E "npu[0-9]" | head
echo "[A] DONE"
