#!/usr/bin/env bash
# Definitive real-hardware matrix: run every NPU-available test in its OWN process
# so a hanging kernel cannot poison the ones that follow it.
#
# Records per test: result, wall time, HAL error (-110 = ETIMEDOUT), value-mismatch
# assertion, panic site, and whether the process aborted.
set -u
cd /home/jun/.claude/jobs/46bc5c7e/tmp/visa_ex_gated || exit 1
. "$HOME/.cargo/env"

IN=/home/jun/.claude/jobs/46bc5c7e/tmp/npu_all_tests.txt
OUT=/home/jun/.claude/jobs/46bc5c7e/tmp/ex_logs/npu_matrix.tsv
DETAIL=/home/jun/.claude/jobs/46bc5c7e/tmp/ex_logs/npu_matrix_detail
mkdir -p "$DETAIL"
: > "$OUT"

while IFS=$'\t' read -r bin test; do
  [ -z "${bin:-}" ] && continue
  safe=$(echo "${bin}__${test}" | tr ':/' '__')
  start=$(date +%s%N)
  out=$(CARGO_TARGET_DIR=target_npu timeout 150 \
        cargo furiosa-opt --backend npu test -p furiosa-opt-examples --release \
        --test "$bin" -- --exact "$test" --nocapture 2>&1 | grep -vE "^\s+[0-9]+,")
  rc=$?
  end=$(date +%s%N)
  ms=$(( (end-start)/1000000 ))

  echo "$out" | tail -80 > "$DETAIL/$safe.txt"

  if   echo "$out" | grep -q "^test result: ok\. 1 passed"; then st=PASS
  elif [ $rc -eq 124 ];                                    then st=TIMEOUT
  elif echo "$out" | grep -q "test result: FAILED";        then st=FAIL
  elif echo "$out" | grep -q "cannot unwind";              then st=ABORT
  else                                                          st=OTHER; fi

  hal=$(echo "$out"  | grep -c "os error -110")
  mism=$(echo "$out" | grep -c "assertion \`left == right\` failed")
  load=$(echo "$out" | grep -c "furiosa_kernel_load")
  panic=$(echo "$out" | grep -oE "panicked at [^:]+:[0-9]+" | head -1 | sed 's/panicked at //')

  printf "%s\t%s\t%s\t%s\thal=%s\tmismatch=%s\tload=%s\t%s\n" \
         "$st" "$ms" "$bin" "$test" "$hal" "$mism" "$load" "${panic:-}" >> "$OUT"
  printf "%-8s %6sms  %-34s %s\n" "$st" "$ms" "$bin" "$test"
done < "$IN"

echo "=== SUMMARY ==="
awk -F'\t' '{c[$1]++} END{for(k in c) printf "%s=%d ",k,c[k]; print ""}' "$OUT"
echo ALLDONE >> "$OUT"
