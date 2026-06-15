import os, sys, torch
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule

# HF reference
sys.path.insert(0, "/home/jun/furiosa/lib/python3.12/site-packages")
from transformers.models.qwen3_next.modeling_qwen3_next import torch_chunk_gated_delta_rule, l2norm

YAML = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/dn_chunk_full.yaml"
NC = int(os.environ.get("NC", 3))
C  = int(os.environ.get("C", 16))
K  = int(os.environ.get("K", 32))
V  = int(os.environ.get("V", 32))
DEV = os.environ.get("RNGD_DEV", "rngd:1")
N = NC * C
torch.manual_seed(0)

# ---- full-sequence inputs in HF layout [B,S,H,D] ----
B, H = 1, 1
q = torch.randn(B, N, H, K, dtype=torch.float32)
k = torch.randn(B, N, H, K, dtype=torch.float32)
v = torch.randn(B, N, H, V, dtype=torch.float32)
# Real-model operating regime: q,k are L2-normalized (use_qk_l2norm_in_kernel=True),
# beta is a learned gate in (0,1), and the log-decay g is strictly NEGATIVE
# (g = -softplus(...)*A). L2-norm bounds k_i@k^T ~ O(1) so the triangular-inverse
# T-matrix and the recurrent state stay well-conditioned; the gated delta rule
# WITHOUT this normalization explodes to ~1e14 magnitudes and no absolute tol is
# meetable. The kernel math is exact in any regime (verified on CPU maxerr 0.0);
# this matches what the deployed model actually feeds the kernel.
beta  = (torch.rand(B, N, H, dtype=torch.float32) * 0.5 + 0.25)   # beta in (0.25,0.75)
g_log = -torch.rand(B, N, H, dtype=torch.float32) * 0.5           # per-step log-decay < 0

# ---- HF multi-chunk reference (ground truth) ----
hf_out, hf_state = torch_chunk_gated_delta_rule(
    q.clone(), k.clone(), v.clone(), g_log.clone(), beta.clone(),
    chunk_size=C, initial_state=None, output_final_state=True,
    use_qk_l2norm_in_kernel=True)
hf_out = hf_out[0, :, 0, :]        # [N, V]
hf_state = hf_state[0, 0]          # [K, V]

# host-side per-head tensors (l2norm q,k FIRST as HF L480-481, then scale q)
scale = 1.0 / (K ** 0.5)
qn = l2norm(q, dim=-1)[0, :, 0, :]
kn = l2norm(k, dim=-1)[0, :, 0, :]
qf = qn * scale                    # [N,K] pre-scaled
kf = kn
vf = v[0, :, 0, :]
gf = g_log[0, :, 0]                # [N]
bf = beta[0, :, 0]                 # [N]

mask_incl   = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=0)
mask_strict = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1)

def build_T(attn_neg):
    # HF L511-515 triangular-inverse refinement, then + I.
    a = attn_neg.clone()
    for i in range(1, C):
        row = a[i, :i].clone()
        sub = a[:i, :i].clone()
        a[i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return a + torch.eye(C, dtype=a.dtype)

# ---- spy on _dfg_inner (CPU fallback path) ----
import furiosa.torch.custom_ops.dfg as dfgmod
calls = {"n": 0}
_orig = dfgmod._dfg_inner
def spy(*a, **kw):
    calls["n"] += 1
    return _orig(*a, **kw)
dfgmod._dfg_inner = spy

# ---- compile kernel once ----
m = TacticKernelModule(open(YAML).read())
cm = torch.compile(m, backend=ft.backend)

# ---- HOST LOOP over chunks, chaining state S ----
S = torch.zeros(K, V, dtype=torch.float32)
outs = []
chunk_errs = []
state_errs = []
for ci in range(NC):
    sl = slice(ci * C, (ci + 1) * C)
    q_i = qf[sl]; k_i = kf[sl]; v_i = vf[sl]; g_i = gf[sl]; b_i = bf[sl]
    g_cum = torch.cumsum(g_i, dim=0)                     # [C]
    decay = ((g_cum.unsqueeze(1) - g_cum.unsqueeze(0)).tril().exp()).tril()  # [C,C]
    # HF beta-weighting: k_beta = k_i*beta, v_beta = v_i*beta (HF L498-499)
    k_beta = k_i * b_i.unsqueeze(-1)                      # [C,K]
    v_beta = v_i * b_i.unsqueeze(-1)                      # [C,V]
    # host-only (no S_prev dep): T matrix (built from k_beta), value, k_cumdecay
    attn0 = -((k_beta @ k_i.transpose(-1, -2)) * decay).masked_fill(mask_incl, 0)
    T = build_T(attn0)
    value = T @ v_beta                                    # [C,V]  HF L516
    kcd   = T @ (k_beta * g_cum.exp().unsqueeze(-1))      # [C,K]  HF L517 k_cumdecay
    # host broadcasts for the gate terms
    decay_strict = decay.masked_fill(mask_strict, 0.0)    # [C,C]
    gexp_k  = g_cum.exp().unsqueeze(1).expand(C, K).contiguous()           # [C,K]
    wdecay  = (g_cum[-1] - g_cum).exp().unsqueeze(1).expand(C, K).contiguous()  # [C,K]
    sdecay  = torch.full((K, V), g_cum[-1].exp().item(), dtype=torch.float32)   # [K,V]

    kin = [q_i.contiguous(), k_i.contiguous(), value.contiguous(),
           decay_strict.contiguous(), kcd.contiguous(), gexp_k,
           wdecay, sdecay, S.contiguous()]
    res = cm(*[t.to(DEV) for t in kin])
    out_chunk = res[0].to('cpu')      # [C,V]
    S_next    = res[1].to('cpu')      # [K,V]
    S = S_next                        # carry state forward
    outs.append(out_chunk)

    err = (out_chunk - hf_out[sl]).abs().max().item()
    chunk_errs.append(err)
    print(f"chunk {ci}: out maxerr vs HF = {err:.3e}")

my_out = torch.cat(outs, dim=0)       # [N,V]
full_ok   = torch.allclose(my_out, hf_out, atol=1e-2)
full_err  = (my_out - hf_out).abs().max().item()
state_ok  = torch.allclose(S, hf_state, atol=1e-2)
state_err = (S - hf_state).abs().max().item()

print(f"NC={NC} C={C} K={K} V={V}  N={N}")
print("per-chunk maxerr:", [f"{e:.3e}" for e in chunk_errs])
print(f"FULL out [{N},{V}] allclose(1e-2): {full_ok}  maxerr: {full_err:.3e}")
print(f"FINAL state [{K},{V}] allclose(1e-2): {state_ok}  maxerr: {state_err:.3e}")
print("_dfg_inner call count:", calls["n"], "(0 == ran on NPU)")
print("INTER_CHUNK_STATE_CARRY_MATCHES:", bool(state_ok))
print("OVERALL_PASS:", bool(full_ok and state_ok and calls["n"] == 0))
