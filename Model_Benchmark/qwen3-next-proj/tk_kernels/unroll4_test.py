import torch  # FIRST
import sys
import numpy as np
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule

YAML = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/dn_prefill_unroll4.yaml"
DEV = "rngd:0"
T = 4
K = 4
V = 4
H = 1
B = 1

# ---- NPU exec guard ----
import furiosa.torch.custom_ops.dfg as dfgmod
_orig_inner = dfgmod._dfg_inner
_inner_calls = {"n": 0}
def _guarded_inner(*a, **kw):
    _inner_calls["n"] += 1
    return _orig_inner(*a, **kw)
dfgmod._dfg_inner = _guarded_inner

torch.manual_seed(0)
query = torch.randn(B, T, H, K, dtype=torch.float32)
key   = torch.randn(B, T, H, K, dtype=torch.float32)
value = torch.randn(B, T, H, V, dtype=torch.float32)
beta_param = torch.randn(B, T, H, dtype=torch.float32)
beta = beta_param.sigmoid()
g = -torch.nn.functional.softplus(torch.randn(B, T, H, dtype=torch.float32))

def torch_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=None):
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
    for i in range(sequence_length):
        q_t = query[:, :, i]; k_t = key[:, :, i]; v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state

ref_out, ref_final_S = torch_recurrent_gated_delta_rule(query, key, value, g, beta)
ref_out_TV = ref_out[0, :, 0, :].numpy()  # [T, V]
ref_S = ref_final_S[0, 0].numpy()

scale = 1.0 / (K ** 0.5)
q_T = (query.transpose(1, 2)[0, 0] * scale)  # [T,K] pre-scaled
k_T = key.transpose(1, 2)[0, 0]
v_T = value.transpose(1, 2)[0, 0]
beta_T = beta.transpose(1, 2)[0, 0]
g_T = g.transpose(1, 2)[0, 0]
decay_T = g_T.exp()

# Build the single flat input list matching the generator id order:
#   0 = S0[K,V]; then per t: q,k,v,beta,decay
S0 = torch.zeros(K, V, dtype=torch.float32)
inputs = [S0]
for t in range(T):
    inputs.append(q_T[t].contiguous())
    inputs.append(k_T[t].contiguous())
    inputs.append(v_T[t].contiguous())
    inputs.append(beta_T[t].expand(V).contiguous())
    inputs.append(decay_T[t].expand(K, V).contiguous())

m = TacticKernelModule(open(YAML).read())
cm = torch.compile(m, backend=ft.backend)
outs = cm(*[x.to(DEV) for x in inputs])

# outputs order: out_0, out_1, out_2, out_3, final_S
npu_out = np.stack([outs[t].detach().to("cpu").float().numpy() for t in range(T)], axis=0)  # [T,V]
npu_S = outs[T].detach().to("cpu").float().numpy()

np.set_printoptions(precision=7)
print("=== UNROLLED single-EDF prefill (T=%d) ===" % T)
print("out_NPU =\n", npu_out)
print("out_REF =\n", ref_out_TV)
maxerr = float(np.max(np.abs(npu_out - ref_out_TV)))
ok = np.allclose(npu_out, ref_out_TV, atol=1e-3)
print("out allclose(atol=1e-3) =", ok, " maxabserr = %.3e" % maxerr)

serr = float(np.max(np.abs(npu_S - ref_S)))
sok = np.allclose(npu_S, ref_S, atol=1e-3)
print("final-state allclose(atol=1e-3) =", sok, " maxabserr = %.3e" % serr)

print("_dfg_inner call count =", _inner_calls["n"], "(0 => ran on NPU)")
overall = ok and sok and (_inner_calls["n"] == 0)
print("OVERALL_PASS =", overall)
sys.exit(0 if overall else 1)
