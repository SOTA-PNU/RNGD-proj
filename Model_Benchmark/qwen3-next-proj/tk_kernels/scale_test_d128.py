"""남은 일 #1: 실차원 스케일. dn_step.yaml(symbolic K,V)을 실제 head 차원 d_k=d_v=128 에서
컴파일·NPU 실행하고 torch 레퍼런스와 일치하는지 검증. 미니(d=4)에서 실차원으로의 확장 리스크 확인."""
import sys, torch
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule

YAML = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/dn_step.yaml"
K = V = int(sys.argv[1]) if len(sys.argv) > 1 else 128
torch.manual_seed(0)

S = torch.randn(K, V, dtype=torch.float32)
q = torch.randn(K, dtype=torch.float32) * (1.0 / (K ** 0.5))   # HF: query pre-scaled by 1/sqrt(d_k)
k = torch.randn(K, dtype=torch.float32)
v = torch.randn(V, dtype=torch.float32)
beta_s = float(torch.rand(1).item())
g_t = float((torch.rand(1) - 0.5).item())
decay_s = float(torch.exp(torch.tensor(g_t)).item())
beta_full = torch.full((V,), beta_s, dtype=torch.float32)
decay_full = torch.full((K, V), decay_s, dtype=torch.float32)
inputs = [S, q, k, v, beta_full, decay_full]

# torch reference (same 5 eqs)
S1 = S * decay_s
kv = (S1 * k.unsqueeze(1)).sum(0)
delta = (v - kv) * beta_s
Sout_ref = S1 + torch.outer(k, delta)
out_ref = (Sout_ref * q.unsqueeze(1)).sum(0)

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

sout_ok = torch.allclose(Sout, Sout_ref, atol=1e-3)
out_ok = torch.allclose(out, out_ref, atol=1e-3)
print(f"K=V={K}")
print("Sout allclose:", sout_ok, "maxerr:", (Sout - Sout_ref).abs().max().item())
print("out  allclose:", out_ok, "maxerr:", (out - out_ref).abs().max().item())
print("_dfg_inner calls:", calls["n"], "(0 == NPU)")
print("SCALE_PASS:", bool(sout_ok and out_ok and calls["n"] == 0))
