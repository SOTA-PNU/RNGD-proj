"""RECON-A: force the Furiosa compiler to lower a plain matmul and dump the IR,
then search for EinsumByDpe / TuContraction in the dumped tactic IR.

Run:
  PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj RNGD_DEV=rngd:2 \
  /home/jun/furiosa/bin/python tk_kernels/dpe_dump_probe.py
"""
import os, sys, json, glob, traceback, shutil

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
# turn on every dump env var we found in native_torch.so, in case summary= isn't enough
DUMP_ROOT = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/_dpe_dump"
if os.path.isdir(DUMP_ROOT):
    shutil.rmtree(DUMP_ROOT)
os.makedirs(DUMP_ROOT, exist_ok=True)
os.environ["DUMP_IR_JSON"] = "1"
os.environ["DUMP_DOT"] = "1"
os.environ["DUMP_TACTIC_GRAPH"] = "1"
os.environ["TACTIC_GRAPH_DUMP_PATH"] = DUMP_ROOT
os.environ["SCHEDULER_CDFG_DUMP"] = "1"

import torch
import torch.nn as nn
import furiosa.torch as ft
from furiosa.torch.config import with_context, cache_context

DEV = os.environ.get("RNGD_DEV", "rngd:2")


class MM(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return torch.matmul(a, b)


class Lin(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.l = nn.Linear(i, o, bias=False)
    def forward(self, x):
        return self.l(x)


def search_dir(d):
    """Walk a dump dir, parse every *_ir_viewer.json and grep for tactic kinds."""
    hits = {"EinsumByDpe": [], "EinsumByVe": [], "files": [], "other_kinds": {}}
    for root, _dirs, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            hits["files"].append(p)
            if not (f.endswith(".json")):
                continue
            try:
                txt = open(p, "r", errors="replace").read()
            except Exception:
                continue
            if "EinsumByDpe" in txt:
                hits["EinsumByDpe"].append(p)
            if "EinsumByVe" in txt:
                hits["EinsumByVe"].append(p)
            # also try structured viewer parse: operators[].kind
            if f.endswith("_ir_viewer.json"):
                try:
                    j = json.loads(txt)
                    ops = j.get("operators") if isinstance(j, dict) else None
                    if isinstance(ops, list):
                        for op in ops:
                            k = op.get("kind") if isinstance(op, dict) else None
                            if k:
                                hits["other_kinds"].setdefault(str(k), 0)
                                hits["other_kinds"][str(k)] += 1
                except Exception:
                    pass
    return hits


def run_case(label, module, inputs, dumpdir):
    os.makedirs(dumpdir, exist_ok=True)
    print(f"\n{'='*70}\nCASE {label}\n{'='*70}")
    try:
        with cache_context(root_dir=dumpdir, summary=True):
            cm = torch.compile(module, backend=ft.backend)
            dev_in = [t.to(DEV) for t in inputs]
            out = cm(*dev_in)
            if isinstance(out, (tuple, list)):
                out = out[0]
            outc = out.detach().to("cpu").float()
        print(f"  ran OK, out shape={tuple(outc.shape)}")
    except Exception as e:
        print(f"  EXC during compile/run: {type(e).__name__}: {e}")
        traceback.print_exc()
    hits = search_dir(dumpdir)
    print(f"  dump files written: {len(hits['files'])}")
    for ff in sorted(set(os.path.basename(x) for x in hits["files"])):
        print(f"     - {ff}")
    print(f"  EinsumByDpe in: {[os.path.basename(x) for x in hits['EinsumByDpe']]}")
    print(f"  EinsumByVe  in: {[os.path.basename(x) for x in hits['EinsumByVe']]}")
    print(f"  operators[].kind histogram: {hits['other_kinds']}")
    return hits


def main():
    torch.manual_seed(0)
    cases = []
    # tiny -> large matmul, to see if compiler switches VE->DPE by size
    sizes = [
        ("mm_8x8x8",     (8, 8),      (8, 8)),
        ("mm_32x32x32",  (32, 32),    (32, 32)),
        ("mm_128x256x128",(128, 256), (256, 128)),
        ("mm_128x2048x512",(128,2048),(2048,512)),
        ("mm_512x2048x2048",(512,2048),(2048,2048)),
    ]
    all_hits = {}
    for name, ash, bsh in sizes:
        a = torch.randn(*ash, dtype=torch.float32)
        b = torch.randn(*bsh, dtype=torch.float32)
        d = os.path.join(DUMP_ROOT, name)
        all_hits[name] = run_case(name, MM(), [a, b], d)

    # nn.Linear cases (the canonical case mentioned in the task)
    lin_cases = [
        ("lin_2048_512", 2048, 512, 128),
        ("lin_256_128",  256,  128, 32),
    ]
    for name, i, o, t in lin_cases:
        x = torch.randn(t, i, dtype=torch.float32)
        d = os.path.join(DUMP_ROOT, name)
        all_hits[name] = run_case(name, Lin(i, o), [x], d)

    print(f"\n{'#'*70}\nGLOBAL SUMMARY\n{'#'*70}")
    any_dpe = False
    for name, h in all_hits.items():
        dpe = bool(h["EinsumByDpe"])
        any_dpe = any_dpe or dpe
        print(f"  {name:24s} DPE={dpe}  VE={bool(h['EinsumByVe'])}  kinds={h['other_kinds']}")
    print(f"\nANY_DPE_FOUND: {any_dpe}")
    print(f"DUMP_ROOT: {DUMP_ROOT}")


if __name__ == "__main__":
    main()
