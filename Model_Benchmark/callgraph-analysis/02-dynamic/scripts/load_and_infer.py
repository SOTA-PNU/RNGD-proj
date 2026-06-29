"""Deterministic load + inference driver for viztracer.

Replicates the serve-time engine path WITHOUT uvicorn so the trace is bounded
and flushes on exit:
    LLM(path)            -> __init__ -> _init_from_artifact -> NativeLLMEngine(...)   (load/"build")
    llm.generate(prompt) -> encode -> self.engine.generate(...)                       (inference)
The native (Rust) work is opaque to viztracer; the last Python frame before the
native call IS the Python->native boundary.
"""
import sys
import time

MODEL = "/home/jun/chacha/qwen2.5-coder-7b-inst-tp8"


def main():
    from furiosa_llm import LLM, SamplingParams

    t0 = time.time()
    print(f"[B] constructing LLM({MODEL}) ...", flush=True)
    llm = LLM(MODEL)
    t1 = time.time()
    print(f"[B] LLM constructed in {t1 - t0:.1f}s", flush=True)

    sp = SamplingParams(max_tokens=48, temperature=0.0)
    print("[B] generate() ...", flush=True)
    out = llm.generate("def quicksort(arr):\n", sp)
    t2 = time.time()
    print(f"[B] generate done in {t2 - t1:.1f}s", flush=True)
    try:
        print("[B] OUTPUT:", repr(out)[:600], flush=True)
    except Exception as e:
        print("[B] output repr failed:", e, flush=True)

    # second short generation to exercise the per-token loop again
    out2 = llm.generate("Write a haiku about NPUs.", SamplingParams(max_tokens=32, temperature=0.0))
    print(f"[B] generate2 done in {time.time() - t2:.1f}s", flush=True)

    try:
        llm.shutdown()
    except Exception as e:
        print("[B] shutdown:", e, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
