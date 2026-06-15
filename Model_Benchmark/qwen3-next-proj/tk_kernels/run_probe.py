import os, sys, torch
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule
import furiosa.torch.custom_ops.dfg as dfgmod

BASE = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/"
DEV = os.environ.get("RNGD_DEV", "rngd:1")
name = sys.argv[1]
C, K, V = 16, 32, 32
torch.manual_seed(0)

calls = {"n": 0}
_orig = dfgmod._dfg_inner
def spy(*a, **kw):
    calls["n"] += 1; return _orig(*a, **kw)
dfgmod._dfg_inner = spy

if name == "mm":
    q = torch.randn(C, K); k = torch.randn(C, K)
    inputs = [q, k]; ref = q @ k.transpose(-1,-2)
elif name == "exp":
    x = torch.randn(C, C); inputs = [x]; ref = torch.exp(x)
elif name == "outbranch":
    q = torch.randn(C, K); k = torch.randn(C, K); v = torch.randn(C, V)
    gd = torch.randn(C, C)
    inputs = [q, k, v, gd]
    dm = torch.exp(gd); ref = (q @ k.transpose(-1,-2) * dm) @ v
elif name == "sbranch":
    k = torch.randn(C, K); v = torch.randn(C, V); wlog = torch.randn(C)
    inputs = [k, v, wlog]
    ref = (torch.exp(wlog).unsqueeze(1) * k).transpose(-1,-2) @ v
elif name == "sbranch_noexp":
    wk = torch.rand(C); k = torch.randn(C, K); v = torch.randn(C, V)
    inputs = [wk, k, v]
    ref = (wk.unsqueeze(1) * k).transpose(-1,-2) @ v
elif name == "state":
    kd = torch.randn(C, K); v = torch.randn(C, V)
    inputs = [kd, v]
    ref = kd.transpose(-1,-2) @ v
elif name == "sbranch2":
    wlog2d = torch.randn(C, K); k = torch.randn(C, K); v = torch.randn(C, V)
    inputs = [wlog2d, k, v]
    ref = (torch.exp(wlog2d) * k).transpose(-1,-2) @ v
else:
    raise SystemExit("unknown probe " + name)

try:
    m = TacticKernelModule(open(BASE + f"probe_{name}.yaml").read())
    cm = torch.compile(m, backend=ft.backend)
    res = cm(*[t.to(DEV) for t in inputs])
    o = (res[0] if isinstance(res,(list,tuple)) else res).to('cpu')
    ok = torch.allclose(o, ref, atol=1e-2)
    print(f"PROBE {name}: shape={tuple(o.shape)} allclose={ok} maxerr={(o-ref).abs().max().item():.4g} dfg_inner={calls['n']} PASS={bool(ok and calls['n']==0)}")
except Exception as e:
    import traceback
    print(f"PROBE {name}: COMPILE/RUN ERROR")
    # only the key line
    s = traceback.format_exc()
    for line in s.splitlines():
        if "RuntimeError" in line or "Error" in line or "subgraph" in line or "Cpu" in line or "EinsumByVe" in line:
            print("  ", line.strip())
