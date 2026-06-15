import torch  # FIRST
import sys
import numpy as np
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule

YAML = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/dn_step.yaml"
DEV = "rngd:0"

T = 8
K = 4   # d_k
V = 4   # d_v
H = 1   # num_heads

# ---- NPU exec guard: monkeypatch _dfg_inner to assert it is NOT called ----
import furiosa.torch.custom_ops.dfg as dfgmod
_orig_inner = dfgmod._dfg_inner
_inner_calls = {"n": 0}
def _guarded_inner(*a, **kw):
    _inner_calls["n"] += 1
    return _orig_inner(*a, **kw)
dfgmod._dfg_inner = _guarded_inner

# -------------------- random inputs (HF-style) --------------------
torch.manual_seed(0)
# HF shapes: (batch, seq, num_heads, head_dim). Use batch=1, num_heads=1.
B = 1
query = torch.randn(B, T, H, K, dtype=torch.float32)
key   = torch.randn(B, T, H, K, dtype=torch.float32)
value = torch.randn(B, T, H, V, dtype=torch.float32)
# beta in (0,1) via sigmoid of a random param (HF uses b.sigmoid())
beta_param = torch.randn(B, T, H, dtype=torch.float32)
beta = beta_param.sigmoid()
# g is the log-decay (typically negative). HF does g.exp() inside.
g = -torch.nn.functional.softplus(torch.randn(B, T, H, dtype=torch.float32))

# -------------------- HF reference (exact copy of L547-586 body) --------------------
def torch_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=None):
    initial_dtype = query.dtype
    # transpose(1,2): (B,T,H,D) -> (B,H,T,D)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None else initial_state.to(value)
    )
    states = []
    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
        states.append(last_recurrent_state.clone())
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state, states

ref_out, ref_final_S, ref_states = torch_recurrent_gated_delta_rule(query, key, value, g, beta)
# ref_out shape: (B, T, H, V). squeeze to (T, V)
ref_out_TV = ref_out[0, :, 0, :].numpy()  # [T, V]

# -------------------- prepare per-step NPU inputs --------------------
# The step kernel implements (per its header), for a single head:
#   S1 = S * decay ; kv = sum_k S1*k ; delta=(v-kv)*beta ; Sout=S1 + k (x) delta ; out = sum_k Sout*q
# This matches HF EXACTLY IF we feed q already scaled (query*scale), and
# decay_full[K,V] = g_t.exp() broadcast, beta_full[V] = beta_t broadcast.
scale = 1.0 / (K ** 0.5)
# Transpose HF to (H,T,D) ordering, then squeeze head:
q_HTD = (query.transpose(1, 2)[0, 0] * scale)  # [T,K]   <-- pre-scaled q
k_HTD = key.transpose(1, 2)[0, 0]              # [T,K]
v_HTD = value.transpose(1, 2)[0, 0]            # [T,V]
beta_T = beta.transpose(1, 2)[0, 0]            # [T]
g_T = g.transpose(1, 2)[0, 0]                  # [T]
decay_T = g_T.exp()                            # [T]

# -------------------- compile step kernel --------------------
m = TacticKernelModule(open(YAML).read())
cm = torch.compile(m, backend=ft.backend)

# -------------------- host loop --------------------
S = torch.zeros(K, V, dtype=torch.float32)
npu_out = np.zeros((T, V), dtype=np.float32)
npu_states = []
for t in range(T):
    q_t = q_HTD[t].contiguous()
    k_t = k_HTD[t].contiguous()
    v_t = v_HTD[t].contiguous()
    beta_full = beta_T[t].expand(V).contiguous()        # [V]
    decay_full = decay_T[t].expand(K, V).contiguous()   # [K,V]
    inputs = [S, q_t, k_t, v_t, beta_full, decay_full]
    Sout, out_t = cm(*[x.to(DEV) for x in inputs])
    S = Sout.detach().to("cpu").float()
    npu_out[t] = out_t.detach().to("cpu").float().numpy()
    npu_states.append(S.clone())

# -------------------- compare --------------------
np.set_printoptions(precision=7, suppress=False)
print("=== HOST-LOOP DeltaNet (T=%d, K=%d, V=%d, H=%d) ===" % (T, K, V, H))
print("out_NPU =")
print(npu_out)
print("out_REF =")
print(ref_out_TV)
maxerr = float(np.max(np.abs(npu_out - ref_out_TV)))
ok = np.allclose(npu_out, ref_out_TV, atol=1e-3)
print("out allclose(atol=1e-3) =", ok, " maxabserr = %.3e" % maxerr)

# final-state check
ref_S = ref_final_S[0, 0].numpy()
npu_S = S.numpy()
serr = float(np.max(np.abs(npu_S - ref_S)))
sok = np.allclose(npu_S, ref_S, atol=1e-3)
print("final-state allclose(atol=1e-3) =", sok, " maxabserr = %.3e" % serr)

print("_dfg_inner call count =", _inner_calls["n"], "(0 => ran on NPU)")
overall = ok and sok and (_inner_calls["n"] == 0)
print("OVERALL_PASS =", overall)
sys.exit(0 if overall else 1)
