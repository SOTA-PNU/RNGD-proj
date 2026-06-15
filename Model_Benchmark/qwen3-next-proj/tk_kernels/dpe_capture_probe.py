"""RECON-A v3: run a matmul through the REAL torch.compile+ft.backend path, but
monkeypatch native compiler.compile/lower to capture the (decomposed, importable)
ExportedProgram the backend feeds in. Then run compiler.lower() on that captured
EP to obtain the Dfg with selected tactics, serialize it, and grep for EinsumByDpe.
Sweep sizes + sorting policies."""
import os, re, traceback
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
import torch
import torch.nn as nn
import furiosa.torch as ft
from furiosa.native_torch import compiler

OUT = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/_dpe_dfg"
os.makedirs(OUT, exist_ok=True)
DEV = os.environ.get("RNGD_DEV", "rngd:2")

CAP = {"ep": None, "cfg": None}
_orig_compile = compiler.compile

def _spy_compile(ep, compiler_config=None, dram_shape_guides=...):
    CAP["ep"] = ep
    CAP["cfg"] = compiler_config
    return _orig_compile(ep, compiler_config=compiler_config, dram_shape_guides=dram_shape_guides)

compiler.compile = _spy_compile


class MM(nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)


def serialize_lowered(label, cfg_override):
    """Lower the captured EP with cfg_override and serialize the DFG."""
    ep = CAP["ep"]
    if ep is None:
        print("  no EP captured")
        return None
    try:
        dfg = compiler.lower(ep, compiler_config=cfg_override)
    except Exception as e:
        print("  lower(captured ep) failed:", type(e).__name__, str(e)[:200])
        return None
    s = dfg.serialize_to_str()
    open(os.path.join(OUT, label + ".dfg.txt"), "w").write(s)
    try:
        pp = str(dfg.pprint())
        open(os.path.join(OUT, label + ".pprint.txt"), "w").write(pp)
    except Exception:
        pp = ""
    tactics = sorted(set(re.findall(r"EinsumBy[A-Za-z]+|TuContraction[A-Za-z]*|OperatorTactic[A-Za-z]+", s)))
    print(f"  DFG {len(s)}B  DPE={'EinsumByDpe' in s}  VE={'EinsumByVe' in s}  TC={'TuContraction' in s}")
    print(f"  tokens: {tactics}")
    return {"dpe": "EinsumByDpe" in s, "ve": "EinsumByVe" in s, "tc": "TuContraction" in s, "s": s, "pp": pp}


def mk_cfg(sorting, hint):
    c = compiler.Config()
    try: c.tactic_sorting_policy = sorting
    except Exception: pass
    try: c.tactic_hint = hint
    except Exception: pass
    return c


def main():
    torch.manual_seed(0)
    sizes = [
        ("mm8",     (8, 8),      (8, 8)),
        ("mm32",    (32, 32),    (32, 32)),
        ("mm128",   (128, 256),  (256, 128)),
        ("mm_big1", (128, 2048), (2048, 512)),
        ("mm_big2", (512, 2048), (2048, 2048)),
        ("mm_huge", (256, 4096), (4096, 4096)),
    ]
    pols = [
        ("Est_Default", compiler.TacticSortingPolicy.ByEstimation, compiler.TacticHintConfig.Default),
        ("Est_Misc",    compiler.TacticSortingPolicy.ByEstimation, compiler.TacticHintConfig.ForMiscModel),
        ("Est_Prefill", compiler.TacticSortingPolicy.ByEstimation, compiler.TacticHintConfig.ForLlmModelPrefill),
        ("Est_NoConstr",compiler.TacticSortingPolicy.ByEstimation, compiler.TacticHintConfig.NoConstraint),
    ]
    found = {}
    for sz, ash, bsh in sizes:
        a = torch.randn(*ash, dtype=torch.float32)
        b = torch.randn(*bsh, dtype=torch.float32)
        CAP["ep"] = None
        print(f"\n{'='*70}\n{sz}: {ash} x {bsh}\n{'='*70}")
        # 1) trigger real backend to capture the decomposed EP
        try:
            cm = torch.compile(MM(), backend=ft.backend)
            out = cm(a.to(DEV), b.to(DEV))
            if isinstance(out, (tuple, list)): out = out[0]
            out.detach().to("cpu")
            print("  backend run OK; EP captured:", CAP["ep"] is not None)
        except Exception as e:
            print("  backend run failed:", type(e).__name__, str(e)[:160])
        # 2) lower captured EP under each policy
        for pname, sort, hint in pols:
            label = f"{sz}__{pname}"
            print(f" [{pname}]")
            r = serialize_lowered(label, mk_cfg(sort, hint))
            if r:
                found[label] = (r["dpe"], r["ve"], r["tc"])

    print(f"\n{'#'*70}\nSUMMARY\n{'#'*70}")
    any_dpe = False
    for k, (dpe, ve, tc) in sorted(found.items()):
        any_dpe = any_dpe or dpe
        print(f"  {k:30s} DPE={dpe} VE={ve} TuContraction={tc}")
    print("\nANY_DPE_FOUND:", any_dpe)
    print("OUT:", OUT)


if __name__ == "__main__":
    main()
