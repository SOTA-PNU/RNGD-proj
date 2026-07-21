#!/bin/bash
# build_run_B.sh — deterministic Python call graph (viztracer) over the build DRIVER.
# viztracer traces only the driver process (cli.main -> ArtifactBuilder -> build_pipeline
# -> Ray submit -> __save_artifacts). The FX-trace / partition / compile bodies run in
# separate Ray worker processes and appear as driver->worker boundary leaves (the Ray
# `.remote()` call), analogous to the Python->native boundary leaves in serve.
# Worker-side Python is captured by py-spy in build_run_A.sh.
set -u
CA=/home/jun/RNGD-proj/Model_Benchmark/callgraph-analysis
LOG=$CA/02-dynamic/logs
SC=$CA/02-dynamic/scripts
export BUILD_MODEL=${BUILD_MODEL:?set BUILD_MODEL}
export BUILD_OUT=${BUILD_OUT:-/home/jun/.claude/jobs/b0976d8e/tmp/build_out_viz}
export BUILD_TP=4 BUILD_MAXLEN=2048

source /home/jun/furiosa/bin/activate
export RUST_LOG=warn HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
rm -rf "$BUILD_OUT"

echo "[bB] $(date +%T) running viztracer over build_driver.py ..."
viztracer --tracer_entries 20000000 --output_file "$LOG/viz_build.json" \
  -- "$SC/build_driver.py" >"$LOG/viz_build_run.log" 2>&1
echo "[bB] viztracer rc=$?"
ls -la "$LOG/viz_build.json" 2>&1

echo "[bB] $(date +%T) parsing viztracer json -> call graph ..."
python "$SC/parse_viz.py" "$LOG/viz_build.json" >"$LOG/viz_build_summary.txt" 2>&1
echo "[bB] parse rc=$?"
tail -40 "$LOG/viz_build_summary.txt"
echo "[bB] DONE"
