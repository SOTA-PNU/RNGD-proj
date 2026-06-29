#!/bin/bash
# run_B.sh — deterministic Python call graph (viztracer) over the load+inference
# path, plus a clean load-focused kernel trace. Run after Run A frees the NPU.
set -u
OUT=/home/jun/chacha/callgraph-analysis/02-dynamic
LOG=$OUT/logs
SC=$OUT/scripts
S(){ echo jun | sudo -S -p '' "$@"; }

echo "[B] $(date +%T) starting load-focused kernel bpftrace ..."
S bpftrace "$SC/kernel_trace.bt" >"$LOG/kernel_trace_B.log" 2>"$LOG/kernel_trace_B.err" &
sleep 4

source /home/jun/furiosa/bin/activate
export RUST_LOG=info RUST_BACKTRACE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[B] $(date +%T) running viztracer over load_and_infer.py ..."
viztracer --tracer_entries 12000000 --output_file "$LOG/viz_load_infer.json" \
  -- "$SC/load_and_infer.py" >"$LOG/viz_run.log" 2>&1
echo "[B] viztracer rc=$?"
ls -la "$LOG/viz_load_infer.json" 2>&1

echo "[B] $(date +%T) stopping bpftrace ..."
S pkill -INT -f kernel_trace.bt 2>/dev/null; sleep 3; S pkill -TERM -f kernel_trace.bt 2>/dev/null
S chown -R jun:jun "$LOG" 2>/dev/null

echo "[B] $(date +%T) parsing viztracer json -> call graph ..."
python "$SC/parse_viz.py" "$LOG/viz_load_infer.json" >"$LOG/viz_callgraph_summary.txt" 2>&1
echo "[B] parse rc=$?"
tail -50 "$LOG/viz_callgraph_summary.txt"
echo "[B] $(date +%T) NPU after:"; furiosa-smi info 2>&1 | grep -E "npu[0-9]" | head -1
echo "[B] DONE"
