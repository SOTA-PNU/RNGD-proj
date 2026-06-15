"""RECON-A v2: lower a plain matmul via compiler.lower() -> Dfg, serialize the
DFG, and search for EinsumByDpe / TuContraction. Try ByEstimation sorting (cost
model) so the compiler picks the FAST engine (DPE for big matmuls)."""
import os, sys, traceback
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
import torch
import torch.nn as nn
import furiosa.torch as ft
from furiosa.native_torch import compiler

OUT = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/_dpe_dfg"
os.makedirs(OUT, exist_ok=True)


class MM(nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)


def make_config(sorting, hint):
    c = compiler.Config()
    try:
        c.tactic_sorting_policy = sorting
    except Exception as e:
        print("  (could not set sorting:", e, ")")
    try:
        c.tactic_hint = hint
    except Exception as e:
        print("  (could not set hint:", e, ")")
    return c


def lower_and_search(label, a, b, cfg):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    try:
        ep = torch.export.export(MM(), (a, b))
    except Exception as e:
        print("  export failed:", type(e).__name__, e)
        return None
    try:
        dfg = compiler.lower(ep, compiler_config=cfg)
    except Exception as e:
        print("  lower failed:", type(e).__name__, e)
        traceback.print_exc()
        return None
    try:
        s = dfg.serialize_to_str()
    except Exception as e:
        print("  serialize failed:", e)
        s = None
    # also pprint form
    try:
        pp = dfg.pprint()
    except Exception:
        pp = None
    fn = os.path.join(OUT, label + ".dfg.txt")
    if s:
        open(fn, "w").write(s)
    if pp:
        open(os.path.join(OUT, label + ".pprint.txt"), "w").write(str(pp))
    has_dpe = bool(s and "EinsumByDpe" in s)
    has_ve = bool(s and "EinsumByVe" in s)
    has_tc = bool(s and "TuContraction" in s)
    # find any OperatorTactic* present
    import re
    tactics = sorted(set(re.findall(r"(?:Operator)?Tactic[A-Za-z]+|EinsumBy[A-Za-z]+", s or ""))) if s else []
    print(f"  serialized {len(s) if s else 0} bytes -> {fn}")
    print(f"  EinsumByDpe={has_dpe}  EinsumByVe={has_ve}  TuContraction={has_tc}")
    print(f"  tactic tokens found: {tactics}")
    if pp:
        # show kind lines
        for ln in str(pp).splitlines():
            if any(k in ln for k in ("Dpe", "Einsum", "Ve ", "kind", "Tactic", "Contraction")):
                print("   PP|", ln.strip()[:160])
    return {"dpe": has_dpe, "ve": has_ve, "tc": has_tc, "tactics": tactics, "s": s}


def main():
    torch.manual_seed(0)
    sizes = [
        ("mm8",      (8, 8),       (8, 8)),
        ("mm32",     (32, 32),     (32, 32)),
        ("mm128",    (128, 256),   (256, 128)),
        ("mm_big1",  (128, 2048),  (2048, 512)),
        ("mm_big2",  (512, 2048),  (2048, 2048)),
        ("mm_huge",  (256, 4096),  (4096, 4096)),
    ]
    policies = [
        ("ByEstimation_Default", compiler.TacticSortingPolicy.ByEstimation, compiler.TacticHintConfig.Default),
        ("ByEstimation_Misc",    compiler.TacticSortingPolicy.ByEstimation, compiler.TacticHintConfig.ForMiscModel),
        ("ByEstimation_Prefill", compiler.TacticSortingPolicy.ByEstimation, compiler.TacticHintConfig.ForLlmModelPrefill),
    ]
    found = {}
    for pol_name, sorting, hint in policies:
        for sz_name, ash, bsh in sizes:
            a = torch.randn(*ash, dtype=torch.float32)
            b = torch.randn(*bsh, dtype=torch.float32)
            cfg = make_config(sorting, hint)
            label = f"{pol_name}__{sz_name}"
            r = lower_and_search(label, a, b, cfg)
            if r:
                found[label] = (r["dpe"], r["ve"], r["tactics"])
    print(f"\n{'#'*70}\nSUMMARY\n{'#'*70}")
    any_dpe = False
    for k, (dpe, ve, tactics) in found.items():
        any_dpe = any_dpe or dpe
        print(f"  {k:40s} DPE={dpe} VE={ve} {tactics}")
    print("\nANY_DPE:", any_dpe)
    print("OUT:", OUT)


if __name__ == "__main__":
    main()
