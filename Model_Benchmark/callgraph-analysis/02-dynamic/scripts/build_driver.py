"""Deterministic driver for `furiosa-llm build`, for viztracer.

Runs the build CLI *in-process* (no extra wrapper process) so viztracer traces
the whole driver: cli.main.main -> cli.convert (build handler) -> ArtifactBuilder
-> build_pipeline -> Ray submit -> __save_artifacts.

The heavy graph work (FX trace, partition, compile) runs inside Ray worker
processes (ray::LocalPipelineGenerationActor / ray::TaskCompileActor), which are
SEPARATE processes and therefore opaque to viztracer — the Ray `.remote()` call
is the driver->worker boundary, exactly as a native PyO3 call is the Python->Rust
boundary in serve. Capture the worker side with py-spy/gdb (build_run_A.sh).
"""
import os
import sys
import time

MODEL = os.environ.get("BUILD_MODEL")
OUT = os.environ.get("BUILD_OUT", "/home/jun/.claude/jobs/b0976d8e/tmp/build_out_viz")
TP = os.environ.get("BUILD_TP", "4")
MAXLEN = os.environ.get("BUILD_MAXLEN", "2048")


def main():
    from furiosa_llm.cli.main import main as cli_main

    sys.argv = [
        "furiosa-llm", "build", MODEL, OUT,
        "-tp", TP, "--max-model-len", MAXLEN,
        "--name", "qwen25-coder-1p5b-tp4-viz",
    ]
    t0 = time.time()
    print(f"[viz] build start argv={sys.argv}", flush=True)
    try:
        cli_main()
    except SystemExit as e:
        print(f"[viz] cli SystemExit={e.code}", flush=True)
    print(f"[viz] build done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
