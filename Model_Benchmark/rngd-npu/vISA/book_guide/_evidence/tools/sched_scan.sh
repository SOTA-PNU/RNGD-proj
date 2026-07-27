#!/usr/bin/env bash
# Dump the hardware schedule for every kernel that compiles for the npu backend
# and reduce it to per-kernel cycle characteristics.
#
# `compile --dump-schedule` is an ahead-of-time compiler artifact: it needs no NPU
# device, so this can run while the real-hardware test matrix occupies the NPU.
set -u
cd /home/jun/.claude/jobs/46bc5c7e/tmp/visa_ex_gated || exit 1
. "$HOME/.cargo/env"
D=/home/jun/.claude/jobs/46bc5c7e/tmp/ex_logs/sched
mkdir -p "$D"

cat /home/jun/.claude/jobs/46bc5c7e/tmp/ok_kernels.txt | xargs -P 10 -I{} bash -c '
  n="{}"; safe=$(echo "$n" | tr ":" "_")
  CARGO_TARGET_DIR=target_sched timeout 300 cargo furiosa-opt compile \
      -p furiosa-opt-examples "$n" --dump-schedule '"$D"'/$safe.json >/dev/null 2>&1
'
echo "dumped: $(ls "$D"/*.json 2>/dev/null | wc -l)"
