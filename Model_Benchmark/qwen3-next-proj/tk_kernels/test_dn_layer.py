#!/usr/bin/env python3
# Validate the three SURROUNDING DeltaNet-layer kernels on NPU vs torch.
import os, traceback
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
import torch
import torch.nn.functional as F
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule
import furiosa.torch.custom_ops.dfg as dfgmod

BASE = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/"
DEV = os.environ.get("RNGD_DEV", "rngd:0")

_orig = dfgmod._dfg_inner
STATE = {"n": 0}
def _spy(*a, **k):
    STATE["n"] += 1
    return _orig(*a, **k)
dfgmod._dfg_inner = _spy

def run(yaml, inputs):
    STATE["n"] = 0
    dsl = open(BASE + yaml).read()
    m = TacticKernelModule(dsl)
    cm = torch.compile(m, backend=ft.backend)
    res = cm(*[t.to(DEV) for t in inputs])
    if not isinstance(res, (tuple, list)):
        res = [res]
    return [r.detach().to("cpu").float() for r in res], STATE["n"]

torch.manual_seed(0)
report = {}

# ===================== (1) CAUSAL DEPTHWISE CONV1D + SiLU =====================
print("\n========== (1) dn_conv1d : causal depthwise conv1d K=4 + SiLU ==========")
try:
    C, T, K = 128, 128, 4
    x = torch.randn(C, T, dtype=torch.float32)
    w = torch.randn(C, K, dtype=torch.float32)
    # torch reference: F.silu(F.conv1d(x, w, groups=C, padding=K-1)[..., :T])
    xb = x.unsqueeze(0)                       # [1,C,T]
    wt = w.unsqueeze(1)                       # [C,1,K]
    conv = F.conv1d(xb, wt, groups=C, padding=K - 1)[..., :T]  # [1,C,T]
    ref = F.silu(conv).squeeze(0)             # [C,T]

    # build pre-shifted causal taps for the kernel
    x_pad = torch.cat([torch.zeros(C, K - 1, dtype=torch.float32), x], dim=-1)  # [C, T+K-1]
    # F.conv1d cross-correlates filter w[c,j] with window; out[t] = sum_j x_pad[c, t+j]*w[c,j]
    xs = [x_pad[:, j:j + T].contiguous() for j in range(K)]                     # xs_j[c,t]
    wfull = [w[:, j:j + 1].expand(C, T).contiguous() for j in range(K)]         # wj[c,t]=w[c,j]
    inputs = xs + wfull
    outs, n = run("dn_conv1d.yaml", inputs)
    y = outs[0]
    ok = torch.allclose(y, ref, atol=1e-3)
    me = (y - ref).abs().max().item()
    print(f"  shape {tuple(y.shape)} allclose={ok} maxerr={me:.3e} dfg_inner={n}")
    report["conv1d"] = (True, ok, me, n)
except Exception as e:
    print("  FAIL:", type(e).__name__)
    print(traceback.format_exc())
    report["conv1d"] = (False, False, -1, -1)

# ===================== (2) L2NORM over last dim =====================
print("\n========== (2) dn_l2norm : x*rsqrt(sum_d x^2 + eps) ==========")
try:
    # real-ish DeltaNet dims: M rows (heads*positions, INNER survive >=128), D feature (reduce)
    M, D = 128, 128
    eps = 1e-6
    x = torch.randn(M, D, dtype=torch.float32)
    ref = x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)   # [M,D]
    # square/sumsq use x_dm=[d,m]; scale-back uses x_md=[m,d]; output is [m,d]
    x_dm = x.t().contiguous()                                    # [D,M]
    x_md = x.contiguous()                                        # [M,D]
    ones_d = torch.ones(D, dtype=torch.float32)
    eps_full = torch.full((M,), eps, dtype=torch.float32)
    inputs = [x_dm, ones_d, eps_full, x_md]
    outs, n = run("dn_l2norm.yaml", inputs)
    y = outs[0]                                                  # [M,D]
    ok = torch.allclose(y, ref, atol=1e-3)
    me = (y - ref).abs().max().item()
    print(f"  shape {tuple(y.shape)} allclose={ok} maxerr={me:.3e} dfg_inner={n}")
    report["l2norm"] = (True, ok, me, n)
except Exception as e:
    print("  FAIL:", type(e).__name__)
    print(traceback.format_exc())
    report["l2norm"] = (False, False, -1, -1)

# ===================== (3) GATED RMSNORM =====================
print("\n========== (3) dn_gnorm : Qwen3NextRMSNormGated ==========")
try:
    M, D = 128, 128
    eps = 1e-6
    x = torch.randn(M, D, dtype=torch.float32)
    gate = torch.randn(M, D, dtype=torch.float32)
    weight = torch.randn(D, dtype=torch.float32)
    # torch reference == Qwen3NextRMSNormGated.forward (L66-81)
    var = x.pow(2).mean(-1, keepdim=True)
    hn = x * torch.rsqrt(var + eps)
    hn = weight * hn
    ref = hn * F.silu(gate)                                       # [M,D]
    # square/sumsq use x_dm=[d,m]; scale-back+gate use [m,d]; output is [m,d]
    x_dm = x.t().contiguous()                                     # [D,M]
    x_md = x.contiguous()                                         # [M,D]
    gate_md = gate.contiguous()                                   # [M,D]
    weight_md = weight.unsqueeze(0).expand(M, D).contiguous()     # weight[d] -> [M,D]
    ones_d = torch.ones(D, dtype=torch.float32)
    invD_full = torch.full((M,), 1.0 / D, dtype=torch.float32)
    eps_full = torch.full((M,), eps, dtype=torch.float32)
    inputs = [x_dm, ones_d, invD_full, eps_full, weight_md, gate_md, x_md]
    outs, n = run("dn_gnorm.yaml", inputs)
    y = outs[0]                                                   # [M,D]
    ok = torch.allclose(y, ref, atol=1e-3)
    me = (y - ref).abs().max().item()
    print(f"  shape {tuple(y.shape)} allclose={ok} maxerr={me:.3e} dfg_inner={n}")
    report["gnorm"] = (True, ok, me, n)
except Exception as e:
    print("  FAIL:", type(e).__name__)
    print(traceback.format_exc())
    report["gnorm"] = (False, False, -1, -1)

print("\n==================== SUMMARY ====================")
for k, v in report.items():
    compiled, ok, me, n = v
    print(f"{k:10s}: compiled={compiled} allclose={ok} maxerr={me:.3e} dfg_inner={n} (NPU-exec={n==0})")
