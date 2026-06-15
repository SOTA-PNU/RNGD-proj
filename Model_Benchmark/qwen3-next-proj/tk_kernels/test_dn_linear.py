import os, torch  # FIRST per recipe
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule
import torch.nn.functional as F

YAML = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/dn_linear.yaml"
DEV  = os.environ.get("RNGD_DEV", "rngd:1")
PADT = 128

torch.manual_seed(0)

# --- spy on _dfg_inner (CPU fallback path) ---
import furiosa.torch.custom_ops.dfg as dfgmod
calls = {"n": 0}
_orig = dfgmod._dfg_inner
def spy(*a, **kw):
    calls["n"] += 1
    return _orig(*a, **kw)
dfgmod._dfg_inner = spy

m = TacticKernelModule(open(YAML).read())
cm = torch.compile(m, backend=ft.backend)

def npu_linear(x, W, pad_token=False):
    """y = x @ W.T  via NPU. x:[T,I] W:[O,I] -> [T,O]. Optionally pad token axis."""
    before = calls["n"]
    T, I = x.shape
    if pad_token and T < PADT:
        xp = torch.zeros(PADT, I, dtype=x.dtype)
        xp[:T] = x
        res = cm(xp.contiguous().to(DEV), W.contiguous().to(DEV))
        y = (res[0] if isinstance(res, (list, tuple)) else res).detach().to("cpu").float()[:T]
    else:
        res = cm(x.contiguous().to(DEV), W.contiguous().to(DEV))
        y = (res[0] if isinstance(res, (list, tuple)) else res).detach().to("cpu").float()
    return y, calls["n"] - before

print("=" * 70)
print("VALIDATE dn_linear.yaml  y[t,o]=sum_i x[t,i]*W[o,i]  vs torch F.linear")
print("=" * 70)

# Test the actual shapes the layer uses + a few sanity shapes.
cases = [
    ("in_proj_qkvz", 32, 256, 128),   # T, hidden=I, proj_qkvz=O (=2K+2V=128)
    ("in_proj_ba",   32, 256, 2),     # T, hidden, 2
    ("out_proj",     32, 32, 256),    # T, value_dim=I, hidden=O
    ("square_small", 32, 32, 32),
    ("tall_T128",    128, 256, 128),
]
all_pass = True
for name, T, I, O in cases:
    x = torch.randn(T, I, dtype=torch.float32)
    W = torch.randn(O, I, dtype=torch.float32)
    ref = F.linear(x, W)                       # [T,O]
    # try WITHOUT pad first
    y0, d0 = npu_linear(x, W, pad_token=False)
    err0 = (y0 - ref).abs().max().item()
    ok0  = torch.allclose(y0, ref, atol=1e-3)
    # WITH token pad to 128
    y1, d1 = npu_linear(x, W, pad_token=True)
    err1 = (y1 - ref).abs().max().item()
    ok1  = torch.allclose(y1, ref, atol=1e-3)
    flag0 = "NPU" if d0 == 0 else f"CPU-FALLBACK(+{d0})"
    flag1 = "NPU" if d1 == 0 else f"CPU-FALLBACK(+{d1})"
    print(f"\n[{name}] T={T} I={I} O={O}")
    print(f"   no-pad : err={err0:.3e} allclose={ok0} dfg={flag0}")
    print(f"   pad128 : err={err1:.3e} allclose={ok1} dfg={flag1}")
    # pass = at least one config is exact AND ran on NPU (dfg==0)
    case_pass = (ok0 and d0 == 0) or (ok1 and d1 == 0)
    all_pass = all_pass and case_pass

print("\n" + "=" * 70)
print("total _dfg_inner calls:", calls["n"])
print("ALL_CASES_PASS:", all_pass)
print("=" * 70)
