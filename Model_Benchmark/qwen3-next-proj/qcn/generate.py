#!/usr/bin/env python3
# =============================================================================
# End-to-end autoregressive GENERATION for Qwen3-Coder-Next-FP8 on the RNGD NPU.
#
# Uses QCNModel (qcn/model.py): prefill threads DeltaNet recurrent state + conv
# tail and full-attention KV through all 48 layers; decode_step feeds one new
# token at a time, continuing the DeltaNet recurrence (carried S + conv state)
# and appending to the attention KV cache.  Greedy/argmax sampling.
#
# Every token mixer matmul (DeltaNet chunk-scan, full-attn SDPA, MoE SwiGLU)
# runs on the NPU via TacticKernel YAMLs; we monkeypatch _dfg_inner and assert
# the CPU-fallback counters stay 0 (proof the kernels ran on-device).
#
# Run:
#   PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj \
#   RNGD_DEV=rngd:2 /home/jun/furiosa/bin/python qcn/generate.py \
#       --prompt "def quicksort(arr):" --max-new 24
# =============================================================================
import os
import sys
import json
import argparse
import torch

sys.path.insert(0, "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj")
from qcn.model import QCNModel
from qcn import deltanet_layer as _dn
from qcn import attn_layer as _attn
from qcn import moe as _moe


def npu_proof():
    return {
        "deltanet_cpu_fallbacks": _dn._CALLS["n"],
        "attn_cpu_fallbacks":     _attn.CALLS["n"],
        "moe_cpu_fallbacks":      _moe.CALLS["n"],
        "deltanet_npu_stages":    None,   # filled by caller from model.dn
        "attn_npu_stages":        len(_attn.NPU_STAGES),
        "moe_npu_stages":         len(_moe.NPU_STAGES),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="def quicksort(arr):")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--chat", action="store_true",
                    help="wrap prompt in the Qwen chat template")
    ap.add_argument("--out", default="/home/jun/RNGD-proj/Model_Benchmark/"
                                     "qwen3-next-proj/qcn/generation_sample.json")
    args = ap.parse_args()

    dev = os.environ.get("RNGD_DEV", "rngd:2")
    print("=" * 78)
    print(f"Qwen3-Coder-Next-FP8 GENERATION on RNGD NPU  dev={dev}")
    print(f"prompt={args.prompt!r}  chat={args.chat}  max_new={args.max_new}")
    print("=" * 78)

    model = QCNModel(dev=dev)
    res = model.generate(args.prompt, max_new_tokens=args.max_new, chat=args.chat,
                         greedy=True, verbose=True)

    print("-" * 78)
    print("PROMPT     :", repr(res["prompt"]))
    print("GENERATED  :", repr(res["generated_text"]))
    print("-" * 78)
    print("FULL TEXT:")
    print(res["full_text"])
    print("-" * 78)

    # ---- NPU-exec proof ----
    proof = {
        "deltanet_cpu_fallbacks": _dn._CALLS["n"],
        "attn_cpu_fallbacks":     _attn.CALLS["n"],
        "moe_cpu_fallbacks":      _moe.CALLS["n"],
        "deltanet_npu_stages":    len(model.dn.npu_stages),
        "attn_npu_stages":        len(_attn.NPU_STAGES),
        "moe_npu_stages":         len(_moe.NPU_STAGES),
    }
    all_npu = (proof["deltanet_cpu_fallbacks"] == 0 and
               proof["attn_cpu_fallbacks"] == 0 and
               proof["moe_cpu_fallbacks"] == 0)
    print("NPU-exec proof (_dfg_inner CPU-fallback counters, MUST be 0):")
    print(f"  deltanet={proof['deltanet_cpu_fallbacks']}  "
          f"attn={proof['attn_cpu_fallbacks']}  moe={proof['moe_cpu_fallbacks']}")
    print(f"  NPU stages run: deltanet={proof['deltanet_npu_stages']} "
          f"attn={proof['attn_npu_stages']} moe={proof['moe_npu_stages']}")
    print(f"  ALL_MIXERS_ON_NPU = {all_npu}")
    print("-" * 78)

    pt = res["per_token_s"]
    if pt:
        print(f"prefill: {res['prefill_s']:.1f}s   "
              f"decode: {len(pt)} tokens, "
              f"avg {sum(pt)/len(pt):.1f}s/tok  "
              f"(min {min(pt):.1f}s max {max(pt):.1f}s)")
    print("=" * 78)

    out = dict(res)
    out["npu_proof"] = proof
    out["all_mixers_on_npu"] = all_npu
    out["dev"] = dev
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
