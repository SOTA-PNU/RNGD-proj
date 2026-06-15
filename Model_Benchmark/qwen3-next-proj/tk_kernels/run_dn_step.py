import torch
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule

YAML = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/dn_step.yaml"

torch.manual_seed(0)
K, V = 4, 4

# fp32 random inputs
S      = torch.randn(K, V, dtype=torch.float32)
q      = torch.randn(K,    dtype=torch.float32)
k      = torch.randn(K,    dtype=torch.float32)
v      = torch.randn(V,    dtype=torch.float32)
beta_s = float(torch.rand(1).item())          # scalar in (0,1)
g_t    = float((torch.rand(1) - 0.5).item())   # scalar log-decay
decay_s = float(torch.exp(torch.tensor(g_t)).item())

# materialized scalars (phase-1 verified same-shape broadcast)
beta_full  = torch.full((V,),   beta_s,  dtype=torch.float32)
decay_full = torch.full((K, V), decay_s, dtype=torch.float32)

# input order MUST match yaml inputs: 0=S 1=q 2=k 3=v 4=beta_full 5=decay_full
inputs = [S, q, k, v, beta_full, decay_full]

# ---- torch reference of the SAME 5 equations ----
S1_ref    = S * decay_s
kv_ref    = (S1_ref * k.unsqueeze(1)).sum(dim=0)        # sum_k S1[k,v]*k[k]
delta_ref = (v - kv_ref) * beta_s
Sout_ref  = S1_ref + torch.outer(k, delta_ref)          # k[k]*delta[v]
out_ref   = (Sout_ref * q.unsqueeze(1)).sum(dim=0)      # sum_k Sout[k,v]*q[k]

# ---- spy on _dfg_inner (CPU fallback path) ----
import furiosa.torch.custom_ops.dfg as dfgmod
calls = {"n": 0}
orig = dfgmod._dfg_inner
def spy(*a, **kw):
    calls["n"] += 1
    return orig(*a, **kw)
dfgmod._dfg_inner = spy

# ---- compile + run on NPU ----
m = TacticKernelModule(open(YAML).read())
cm = torch.compile(m, backend=ft.backend)
npu_in = [t.to('rngd:0') for t in inputs]
res = cm(*npu_in)

print("RESULT TYPE:", type(res))
# res is a LIST of outputs [Sout, out]
Sout_npu = res[0].to('cpu')
out_npu  = res[1].to('cpu')

print("=== INPUTS ===")
print("S =", S.tolist())
print("q =", q.tolist())
print("k =", k.tolist())
print("v =", v.tolist())
print("beta =", beta_s, " g_t =", g_t, " decay =", decay_s)
print("=== Sout NPU ===\n", Sout_npu.tolist())
print("=== Sout REF ===\n", Sout_ref.tolist())
print("=== out NPU ===\n", out_npu.tolist())
print("=== out REF ===\n", out_ref.tolist())

sout_ok = torch.allclose(Sout_npu, Sout_ref, atol=1e-3)
out_ok  = torch.allclose(out_npu,  out_ref,  atol=1e-3)
sout_maxerr = (Sout_npu - Sout_ref).abs().max().item()
out_maxerr  = (out_npu  - out_ref ).abs().max().item()

print("=== VERIFY ===")
print("Sout allclose(atol=1e-3):", sout_ok, " maxerr:", sout_maxerr)
print("out  allclose(atol=1e-3):", out_ok,  " maxerr:", out_maxerr)
print("_dfg_inner call count:", calls["n"], "(0 == ran on NPU)")
print("OVERALL_PASS:", bool(sout_ok and out_ok and calls["n"] == 0))
