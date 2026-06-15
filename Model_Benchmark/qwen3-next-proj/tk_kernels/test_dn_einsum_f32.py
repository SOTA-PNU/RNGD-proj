import torch  # FIRST per recipe
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule

YAML = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/dn_einsum_f32.yaml"

torch.manual_seed(0)
d_k, d_v = 4, 4
k_vec = torch.randn(d_k, dtype=torch.float32)
S = torch.randn(d_k, d_v, dtype=torch.float32)

# torch reference: einsum('kv,k->v', S, k_vec)
ref = torch.einsum('kv,k->v', S, k_vec)

m = TacticKernelModule(open(YAML).read())

# --- monkeypatch to verify NPU exec path: _dfg_inner must NOT be called ---
import furiosa.torch.custom_ops.dfg as dfgmod
called = {"hit": False}
orig = dfgmod._dfg_inner
def spy(*a, **kw):
    called["hit"] = True
    return orig(*a, **kw)
dfgmod._dfg_inner = spy

cm = torch.compile(m, backend=ft.backend)
# inputs order == reads order == [k_vec(input0), S(input1)]
inputs = [k_vec, S]
out = cm(*[t.to('rngd:0') for t in inputs])
if isinstance(out, (list, tuple)):
    out = out[0]
out_cpu = out.detach().to('cpu')

print("d_k,d_v =", d_k, d_v)
print("k_vec   =", k_vec.tolist())
print("S       =", S.tolist())
print("ref     =", ref.tolist())
print("npu_out =", out_cpu.tolist())
ok = torch.allclose(out_cpu, ref, atol=1e-4)
print("ALLCLOSE_atol1e-4 =", ok)
print("max_abs_err =", (out_cpu - ref).abs().max().item())
print("_dfg_inner_CALLED =", called["hit"])
print("RESULT_PASS =", bool(ok and not called["hit"]))
