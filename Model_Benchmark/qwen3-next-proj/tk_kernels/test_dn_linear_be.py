"""BATCHED-EXPERT dn_linear_be.yaml test: y[e,t,o]=sum_i x[e,t,i]*W[e,o,i].
Verifies a leading expert-batch axis 'e' (one dispatch over E experts) vs a torch
reference that LOOPS F.linear per expert. Token axis padded to >=128. Proves NPU exec
(monkeypatch _dfg_inner, must stay 0)."""
import os, sys, torch
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule
import torch.nn.functional as F
import furiosa.torch.custom_ops.dfg as dfgmod

YAML = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/dn_linear_be.yaml"
DEV  = os.environ.get("RNGD_DEV", "rngd:2")
PADT = 128
torch.manual_seed(0)
def log(*a): print(*a, flush=True)

calls = {"n": 0}
_orig = dfgmod._dfg_inner
def spy(*a, **kw):
    calls["n"] += 1
    return _orig(*a, **kw)
dfgmod._dfg_inner = spy

m = TacticKernelModule(open(YAML).read())
cm = torch.compile(m, backend=ft.backend)

def npu_linear_be(x, W):
    E, T, I = x.shape
    O = W.shape[1]
    before = calls["n"]
    if T < PADT:
        xp = torch.zeros(E, PADT, I, dtype=x.dtype); xp[:, :T] = x
        res = cm(xp.contiguous().to(DEV), W.contiguous().to(DEV))
        y = (res[0] if isinstance(res, (list, tuple)) else res).detach().to("cpu").float()[:, :T]
    else:
        res = cm(x.contiguous().to(DEV), W.contiguous().to(DEV))
        y = (res[0] if isinstance(res, (list, tuple)) else res).detach().to("cpu").float()
    return y, calls["n"] - before

def ref_be(x, W):
    return torch.stack([F.linear(x[e], W[e]) for e in range(x.shape[0])], 0)

log("=" * 72)
log("VALIDATE dn_linear_be.yaml  y[e,t,o]=sum_i x[e,t,i]*W[e,o,i]  vs per-expert F.linear")
log("=" * 72)

# Only the cases needed to prove correctness + the real MoE gate/up/down shapes.
cases = [
    ("E=2 sanity",      2, 8, 64,   32),
    ("gate_proj E=10", 10, 8, 2048, 512),    # E, T, I, O
    ("down_proj E=10", 10, 8, 512, 2048),
]
all_pass = True
for name, E, T, I, O in cases:
    x = torch.randn(E, T, I, dtype=torch.float32) * 0.1
    W = torch.randn(E, O, I, dtype=torch.float32) * 0.05
    ref = ref_be(x, W)
    try:
        y, d = npu_linear_be(x, W)
        err = (y - ref).abs().max().item()
        ok  = torch.allclose(y, ref, atol=1e-3)
        flag = "NPU" if d == 0 else f"CPU-FALLBACK(+{d})"
        log(f"[{name}] E={E} T={T} I={I} O={O}  err={err:.3e} allclose={ok} dfg={flag}")
        case_pass = ok and d == 0
    except Exception as ex:
        log(f"[{name}] E={E} T={T} I={I} O={O}  EXCEPTION: {type(ex).__name__}: {str(ex)[:200]}")
        case_pass = False
    all_pass = all_pass and case_pass

log("=" * 72)
log(f"total _dfg_inner calls: {calls['n']}")
log(f"ALL_CASES_PASS: {all_pass}")
log("=" * 72)
