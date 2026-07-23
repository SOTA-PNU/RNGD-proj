#!/bin/bash
# build_run_A.sh — live multi-layer dynamic capture of `furiosa-llm build`
# Layers: eBPF kernel (bpftrace, proves build never touches the NPU)
#         | native backtraces (gdb on driver + Ray workers, tracing & compile phases)
#         | Python call graph (py-spy dump + record on the Ray workers)
#         | RUST_LOG native module flow (build.log)
# Build runs the FX trace + AOT compile inside Ray worker processes
# (ray::LocalPipelineGenerationActor for trace, ray::TaskCompileActor for compile),
# so the interesting Python+native frames live in those workers, not the driver.
# Self-sequences with phase detection on build.log. Run as user jun (uses sudo for gdb/bpftrace).
set -u
CA=/home/jun/RNGD-proj/Model_Benchmark/callgraph-analysis
LOG=$CA/02-dynamic/logs
SC=$CA/02-dynamic/scripts
MODEL=${BUILD_MODEL:?set BUILD_MODEL}
OUT=${BUILD_OUT:-/home/jun/.claude/jobs/b0976d8e/tmp/build_out_A}
PYSPY=/home/jun/furiosa/bin/py-spy
S(){ echo jun | sudo -S -p '' "$@"; }

gdb_snap(){ # $1=pid  $2=outfile
  S gdb -p "$1" -batch -nx -ex "set pagination off" -ex "set confirm off" \
    -ex "info threads" -ex "thread apply all bt" -ex "detach" -ex "quit" >"$2" 2>&1
}

echo "[bA] $(date +%T) starting kernel bpftrace ..."
S bpftrace "$SC/kernel_trace.bt" >"$LOG/kernel_trace_build.log" 2>"$LOG/kernel_trace_build.err" &
sleep 4

echo "[bA] $(date +%T) launching build ..."
source /home/jun/furiosa/bin/activate
export RUST_LOG=info RUST_BACKTRACE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
rm -rf "$OUT"
echo "BUILD_START $(date +%s) $(date +%T)" >"$LOG/build.log"
/usr/bin/time -v furiosa-llm build "$MODEL" "$OUT" -tp 4 --max-model-len 2048 \
  --name qwen25-coder-1p5b-tp4 >>"$LOG/build.log" 2>&1 &
BPID=$!
echo "BUILD_PID=$BPID" | tee "$LOG/build_pids.txt"

GOT_TRACE=0; GOT_COMPILE=0; PYSPY_W=""
for i in $(seq 1 600); do
  kill -0 "$BPID" 2>/dev/null || { echo "[bA] build exited"; break; }
  # ---- tracing phase: snapshot the LocalPipelineGenerationActor worker ----
  if [ "$GOT_TRACE" = 0 ] && grep -aq "Model Tracing Progress" "$LOG/build.log" 2>/dev/null; then
    sleep 25   # let a trace task get deep into dynamo/make_fx
    W=$(pgrep -f "ray::LocalPipelineGenerationActor" | head -1)
    echo "[bA] $(date +%T) TRACE snapshot driver=$BPID worker=$W"
    { echo "== pstree =="; S pstree -ap "$BPID" 2>/dev/null;
      echo "== ray procs =="; pgrep -af "ray::" | grep -v grep; } >"$LOG/build_pstree.txt" 2>&1
    { echo "== driver fds (rngd?) =="; S ls -l /proc/$BPID/fd 2>/dev/null | grep -iE "rngd|npu";
      echo "== worker $W fds (rngd?) =="; S ls -l /proc/$W/fd 2>/dev/null | grep -iE "rngd|npu";
      echo "(empty above => build never opened an NPU device node)"; } >"$LOG/build_fds.txt" 2>&1
    [ -n "$W" ] && S "$PYSPY" dump --pid "$W" >"$LOG/pyspy_build_trace.txt" 2>&1
    [ -n "$W" ] && gdb_snap "$W" "$LOG/gdb_build_trace_worker.txt"
    gdb_snap "$BPID" "$LOG/gdb_build_driver.txt"
    # start a py-spy record on the worker for a tracing-phase flamegraph
    [ -n "$W" ] && S "$PYSPY" record --nonblocking --pid "$W" -o "$LOG/pyspy_build_trace.svg" \
       -d 60 -f flamegraph >"$LOG/pyspy_build_trace_rec.log" 2>&1 &
    PYSPY_W=$!
    GOT_TRACE=1
  fi
  # ---- compile phase: snapshot the TaskCompileActor worker (native compiler .so) ----
  if [ "$GOT_COMPILE" = 0 ] && grep -aq "Compilation Progress" "$LOG/build.log" 2>/dev/null; then
    sleep 3
    W=$(pgrep -f "ray::TaskCompileActor" | head -1)
    [ -z "$W" ] && W=$(pgrep -f "ray::" | grep -v IDLE | head -1)
    echo "[bA] $(date +%T) COMPILE snapshot worker=$W"
    [ -n "$W" ] && S "$PYSPY" dump --pid "$W" >"$LOG/pyspy_build_compile.txt" 2>&1
    # take several gdb snapshots during compile to catch the native compiler mid-flight
    for k in 1 2 3; do
      [ -n "$W" ] && gdb_snap "$W" "$LOG/gdb_build_compile_${k}.txt"
      sleep 4
    done
    cp -f "$LOG/gdb_build_compile_1.txt" "$LOG/gdb_build_compile.txt" 2>/dev/null
    GOT_COMPILE=1
  fi
  [ "$GOT_TRACE" = 1 ] && [ "$GOT_COMPILE" = 1 ] && { wait "$BPID" 2>/dev/null; break; }
  sleep 3
done

wait "$BPID" 2>/dev/null
echo "BUILD_END rc=$? $(date +%s) $(date +%T)" >>"$LOG/build.log"
echo "[bA] $(date +%T) stopping bpftrace ..."
S pkill -INT -f kernel_trace.bt 2>/dev/null; sleep 3; S pkill -TERM -f kernel_trace.bt 2>/dev/null
S chown -R jun:jun "$LOG" 2>/dev/null
echo "[bA] artifact:"; ls -la "$OUT" 2>&1 | head
echo "[bA] DONE"
