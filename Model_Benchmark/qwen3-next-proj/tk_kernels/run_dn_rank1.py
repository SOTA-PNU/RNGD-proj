import torch
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule

YAML = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/dn_rank1.yaml"

d_k, d_v = 4, 4
torch.manual_seed(0)
S = torch.arange(1, d_k * d_v + 1, dtype=torch.float32).reshape(d_k, d_v)
k_vec = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
delta = torch.tensor([0.5, -1.0, 2.0, 0.25], dtype=torch.float32)

ref = S + torch.outer(k_vec, delta)
print("S =\n", S)
print("k_vec =", k_vec)
print("delta =", delta)
print("torch ref (S + outer) =\n", ref)

# monkeypatch _dfg_inner to detect NPU execution
import furiosa.torch.custom_ops.dfg as dfgmod
called = {"n": 0}
orig = dfgmod._dfg_inner
def spy(*a, **kw):
    called["n"] += 1
    return orig(*a, **kw)
dfgmod._dfg_inner = spy

m = TacticKernelModule(open(YAML).read())
cm = torch.compile(m, backend=ft.backend)
# YAML input order: 0=k_vec[k], 1=delta[v], 2=S[k,v]
inputs = [k_vec, delta, S]
out = cm(*[t.to("rngd:0") for t in inputs])
if isinstance(out, (list, tuple)):
    out = out[0]
out_cpu = out.to("cpu")
print("NPU out =\n", out_cpu)
print("_dfg_inner called count =", called["n"])

ok = torch.allclose(out_cpu, ref, rtol=1e-5, atol=1e-6)
print("allclose =", ok)
print("max abs diff =", (out_cpu - ref).abs().max().item())
assert ok, "MISMATCH vs torch reference"
assert called["n"] == 0, f"_dfg_inner WAS called ({called['n']}x) -> ran on CPU offline path, not NPU"
print("PASS: NPU exec correct, _dfg_inner NOT called")
