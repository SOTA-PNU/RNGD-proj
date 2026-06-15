"""Minimal DPE probe: compile+exec dn_linear_dpe.yaml on rngd:2, print FULL error.
Usage: PYTHONPATH=... RNGD_DEV=rngd:2 python run_dpe_probe.py [yaml_path]
"""
import os, sys, traceback, torch
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule
import torch.nn.functional as F

YAML = sys.argv[1] if len(sys.argv) > 1 else "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/dn_linear_dpe.yaml"
DEV  = os.environ.get("RNGD_DEV", "rngd:2")
torch.manual_seed(0)

def log(*a): print(*a, flush=True)

# spy on CPU-fallback path
import furiosa.torch.custom_ops.dfg as dfgmod
calls = {"n": 0}
_orig = dfgmod._dfg_inner
def spy(*a, **kw):
    calls["n"] += 1
    return _orig(*a, **kw)
dfgmod._dfg_inner = spy

log("=" * 72)
log(f"DPE PROBE  yaml={YAML}  dev={DEV}")
log("=" * 72)

# --- parse ---
try:
    m = TacticKernelModule(open(YAML).read())
    log("[parse] OK")
except Exception as ex:
    log("[parse] FAILED")
    traceback.print_exc()
    log("PARSE_ERROR:", repr(str(ex))[:2000])
    sys.exit(2)

cm = torch.compile(m, backend=ft.backend)

# shapes: T=128 (>=128 padded), I=512, O=2048  (the slow matmul shape from task ctx)
T, I, O = 128, 512, 2048
x = torch.randn(T, I, dtype=torch.float32) * 0.1
W = torch.randn(O, I, dtype=torch.float32) * 0.05
ref = F.linear(x, W)

# --- compile + exec ---
try:
    res = cm(x.contiguous().to(DEV), W.contiguous().to(DEV))
    y = (res[0] if isinstance(res, (list, tuple)) else res).detach().to("cpu").float()
    err = (y - ref).abs().max().item()
    ok = torch.allclose(y, ref, atol=1e-3)
    flag = "NPU" if calls["n"] == 0 else f"CPU-FALLBACK(+{calls['n']})"
    log(f"[exec] OK  err={err:.3e} allclose={ok} dfg={flag}")
    log("EXEC_SUCCESS")
except Exception as ex:
    log("[exec] FAILED")
    traceback.print_exc()
    log("EXEC_ERROR:", repr(str(ex))[:3000])
    sys.exit(3)
