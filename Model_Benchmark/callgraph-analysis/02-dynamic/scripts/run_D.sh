#!/bin/bash
# run_D.sh — clean perf-only native+kernel capture (NO py-spy; bounded by timeout).
# Goal: capture userspace->syscall->furiosa_rngd driver CALL STACKS during inference
# (perf symbolizes kernel frames via kallsyms; native_runtime.so frames are hex/stripped).
set -u
OUT=/home/jun/chacha/callgraph-analysis/02-dynamic
LOG=$OUT/logs
MODEL=/home/jun/chacha/qwen2.5-coder-7b-inst-tp8
PORT=12347
S(){ echo jun | sudo -S -p '' "$@"; }

source /home/jun/furiosa/bin/activate
export RUST_LOG=info HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[D] $(date +%T) launching serve :$PORT ..."
furiosa-llm serve "$MODEL" --port "$PORT" >"$LOG/serve_D.log" 2>&1 &
SERVE=$!
READY=0
for i in $(seq 1 90); do
  curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" -o "$LOG/models_D.json" 2>/dev/null \
    && grep -qE '"(id|data|object)"' "$LOG/models_D.json" 2>/dev/null && { READY=1; echo "[D] READY ~$((i*2))s"; break; }
  kill -0 "$SERVE" 2>/dev/null || { echo "[D] serve died"; break; }
  sleep 2
done

if [ "$READY" = 1 ]; then
  MODELID=$(python -c "import json;print(json.load(open('$LOG/models_D.json'))['data'][0]['id'])" 2>/dev/null); [ -z "$MODELID" ] && MODELID="$MODEL"
  echo "[D] $(date +%T) firing long generation + perf (dwarf, 12s) ..."
  ( curl -s -m 200 "http://127.0.0.1:$PORT/v1/completions" -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODELID\",\"prompt\":\"Write a long, fully documented Python implementation of a B-tree with insert, delete, search and tests.\",\"max_tokens\":600,\"temperature\":0,\"stream\":false}" \
      >"$LOG/long_gen_D.json" 2>/dev/null ) & GEN=$!
  sleep 3
  S timeout 40 perf record -g --call-graph dwarf -o "$LOG/perf_infer.data" -p "$SERVE" -- sleep 12 >"$LOG/perf_infer.log" 2>&1
  echo "[D] perf rc=$?"
  wait $GEN 2>/dev/null
  echo "[D] $(date +%T) perf post-process ..."
  S perf report -i "$LOG/perf_infer.data" --stdio --no-children 2>/dev/null | head -400 >"$LOG/perf_infer_report.txt"
  S perf report -i "$LOG/perf_infer.data" --stdio 2>/dev/null | head -500 >"$LOG/perf_infer_report_children.txt"
  S perf script -i "$LOG/perf_infer.data" 2>/dev/null >"$LOG/perf_infer_script.txt"
  # native->kernel evidence: frames touching the driver / ioctl syscall path
  grep -iE "rngd|ncdev_ioctl|doorbell|dma_transfer|npu_|__x64_sys_ioctl|entry_SYSCALL_64|do_vfs_ioctl|__se_sys_ioctl" \
     "$LOG/perf_infer_script.txt" 2>/dev/null | sed -E 's/^[0-9a-f]+ //' | sort | uniq -c | sort -rn | head -50 >"$LOG/perf_kernel_frames.txt"
fi
echo "[D] $(date +%T) teardown ..."
kill -INT "$SERVE" 2>/dev/null
for i in $(seq 1 15); do kill -0 "$SERVE" 2>/dev/null || break; sleep 1; done
kill -9 "$SERVE" 2>/dev/null
S chown -R jun:jun "$LOG" 2>/dev/null
ls -la "$LOG/perf_infer.data" "$LOG/perf_kernel_frames.txt" 2>&1 | awk '{print $5,$NF}'
echo "[D] DONE"
