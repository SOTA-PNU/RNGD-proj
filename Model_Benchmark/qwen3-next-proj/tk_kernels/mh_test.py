"""MULTI-HEAD DeltaNet step test: dn_step_mh.yaml at H=4, d_k=d_v=128, fp32, on rngd:0.
Verifies BOTH outputs (Sout[H,K,V], out[H,V]) vs a torch reference that LOOPS the
5 gated-delta equations PER HEAD (allclose atol 1e-3), and asserts _dfg_inner NOT
called (== ran on NPU, not the CPU fallback)."""
import sys, torch
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule

YAML = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/dn_step_mh.yaml"
H = int(sys.argv[1]) if len(sys.argv) > 1 else 4
K = V = int(sys.argv[2]) if len(sys.argv) > 2 else 128
torch.manual_seed(0)

# Per-head random inputs (distinct per head so a head-axis bug would show).
S       = torch.randn(H, K, V, dtype=torch.float32)
q       = torch.randn(H, K, dtype=torch.float32) * (1.0 / (K ** 0.5))  # HF: q pre-scaled by 1/sqrt(d_k)
k       = torch.randn(H, K, dtype=torch.float32)
v       = torch.randn(H, V, dtype=torch.float32)
beta_s  = torch.rand(H, dtype=torch.float32)                # per-head scalar
g_t     = (torch.rand(H, dtype=torch.float32) - 0.5)        # per-head gate
decay_s = torch.exp(g_t)                                    # per-head scalar
beta_full  = beta_s.unsqueeze(1).expand(H, V).contiguous()        # [H,V]
decay_full = decay_s.view(H, 1, 1).expand(H, K, V).contiguous()   # [H,K,V]
inputs = [S, q, k, v, beta_full, decay_full]

# torch reference: LOOP the 5 eqs per head independently.
Sout_ref = torch.empty(H, K, V, dtype=torch.float32)
out_ref  = torch.empty(H, V, dtype=torch.float32)
for h in range(H):
    S1   = S[h] * decay_s[h]
    kv   = (S1 * k[h].unsqueeze(1)).sum(0)
    delta = (v[h] - kv) * beta_s[h]
    South = S1 + torch.outer(k[h], delta)
    outh  = (South * q[h].unsqueeze(1)).sum(0)
    Sout_ref[h] = South
    out_ref[h]  = outh

# NPU-exec proof: _dfg_inner is the CPU-fallback interpreter; must NOT be called.
import furiosa.torch.custom_ops.dfg as dfgmod
calls = {"n": 0}
_orig = dfgmod._dfg_inner
def spy(*a, **kw):
    calls["n"] += 1
    return _orig(*a, **kw)
dfgmod._dfg_inner = spy

m = TacticKernelModule(open(YAML).read())
cm = torch.compile(m, backend=ft.backend)
res = cm(*[t.to('rngd:0') for t in inputs])
Sout = res[0].to('cpu'); out = res[1].to('cpu')

print("Sout shape:", tuple(Sout.shape), "expected", (H, K, V))
print("out  shape:", tuple(out.shape),  "expected", (H, V))
sout_ok = Sout.shape == Sout_ref.shape and torch.allclose(Sout, Sout_ref, atol=1e-3)
out_ok  = out.shape  == out_ref.shape  and torch.allclose(out,  out_ref,  atol=1e-3)
sout_maxerr = (Sout - Sout_ref).abs().max().item() if Sout.shape == Sout_ref.shape else float('nan')
out_maxerr  = (out  - out_ref ).abs().max().item() if out.shape  == out_ref.shape  else float('nan')
# per-head maxerr to prove no head got cross-contaminated
per_head_sout = [(Sout[h]-Sout_ref[h]).abs().max().item() for h in range(H)] if Sout.shape==Sout_ref.shape else []
per_head_out  = [(out[h]-out_ref[h]).abs().max().item() for h in range(H)] if out.shape==out_ref.shape else []

print(f"H={H} K=V={K}")
print("Sout allclose:", sout_ok, "maxerr:", sout_maxerr)
print("out  allclose:", out_ok,  "maxerr:", out_maxerr)
print("per-head Sout maxerr:", per_head_sout)
print("per-head out  maxerr:", per_head_out)
print("_dfg_inner calls:", calls["n"], "(0 == NPU)")
print("MH_PASS:", bool(sout_ok and out_ok and calls["n"] == 0))
