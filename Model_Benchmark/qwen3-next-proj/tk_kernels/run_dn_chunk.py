import os, sys, torch
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule

YAML = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/dn_chunk.yaml"

# chunk size C, head dims d_k=d_v
C = int(sys.argv[1]) if len(sys.argv) > 1 else 16
K = int(sys.argv[2]) if len(sys.argv) > 2 else 32
V = int(sys.argv[3]) if len(sys.argv) > 3 else 32
DEV = os.environ.get("RNGD_DEV", "rngd:0")
torch.manual_seed(0)

# ---- raw chunk inputs (mirror HF torch_chunk_gated_delta_rule, ONE chunk) ----
q_raw = torch.randn(C, K, dtype=torch.float32)
k_raw = torch.randn(C, K, dtype=torch.float32)
v_raw = torch.randn(C, V, dtype=torch.float32)
beta  = torch.rand(C, dtype=torch.float32)          # per-position gate in (0,1)
g_log = (torch.rand(C, dtype=torch.float32) - 0.5) * 0.5   # per-position log-decay

scale = 1.0 / (K ** 0.5)
q = q_raw * scale                                    # HF pre-scales query

# HF beta-weighting (simplified chunk uses k/v directly in attn @ v_new with
# initial_state=0; we fold beta into v to match attn@v_beta? -> we SIMPLIFY to
# the basic scan: attn @ v (no beta on v) so the matmul core is exercised cleanly).
# To keep an honest single-chunk match we set v_used = v_raw and k_used = k_raw.
k_used = k_raw
v_used = v_raw

# cumulative decay within chunk (HF: g = g.cumsum(-1))
g_cum = torch.cumsum(g_log, dim=0)                   # [C]

# decay_mask[c,d] = exp(g_cum[c]-g_cum[d]) for d<=c else 0   (HF tril().exp().tril())
gdiff_full = g_cum.unsqueeze(1) - g_cum.unsqueeze(0)        # [C,C]  (c rows, d cols)
BIG = 30.0
causal = torch.tril(torch.ones(C, C))                      # 1 where d<=c
gdiff = torch.where(causal.bool(), gdiff_full, torch.full_like(gdiff_full, -BIG))

# wlog[d] = g_cum[last]-g_cum[d]   (HF: (g[...,-1]-g).exp() weights for state)
# broadcast to 2D [C,K] for the NPU (the per-row state-decay log over all k).
wlog_vec = g_cum[-1] - g_cum                                # [C]
wlog2d = wlog_vec.unsqueeze(1).expand(C, K).contiguous()    # [C,K]

inputs = [q, k_used, v_used, gdiff, wlog2d]

# ---- torch reference of the SAME on-NPU equations ----
decay_mask_ref = torch.exp(gdiff)                          # ~0 in upper triangle
qk_ref   = q @ k_used.transpose(-1, -2)                    # [C,C]
attn_ref = qk_ref * decay_mask_ref                         # masked+decayed
out_ref  = attn_ref @ v_used                               # [C,V]
wk_ref   = torch.exp(wlog2d)                               # [C,K]
kdecay_ref = wk_ref * k_used                               # [C,K]
S_ref    = kdecay_ref.transpose(-1, -2) @ v_used           # [K,V]

# ---- cross-check vs an HF-faithful single-chunk reference that ALSO skips the
# triangular-inverse refinement (T=I) and uses beta=1, initial_state=0. This is
# exactly the "basic chunked scan" the task permits; the gap below should be ~0,
# confirming our simplified equations ARE the HF chunk math minus the T-refinement.
# HF chunk (L505-537) with T=I, beta=1, S_prev=0:
#   decay_mask[c,d] = exp(g_cum[c]-g_cum[d]) tril (incl diag)        [L509]
#   attn   = (q_i @ k_i^T * decay_mask) masked_fill(triu(diag=1),0)  [L529, q pre-scaled]
#   core_attn_out = attn @ v_i      (attn_inter=0, v_new=v_i)        [L531-533]
#   state  = (k_i * exp(g_last-g_cum)[...,None]).T @ v_i             [L536]
hf_decay = torch.tril(torch.exp(g_cum.unsqueeze(1) - g_cum.unsqueeze(0)))   # tril incl diag
hf_attn  = (q @ k_raw.transpose(-1,-2) * hf_decay)
hf_attn  = hf_attn.masked_fill(torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1), 0)
hf_out   = hf_attn @ v_raw                                                   # [C,V]
hf_kdec  = k_raw * torch.exp(g_cum[-1] - g_cum).unsqueeze(1)                 # [C,K]
hf_state = hf_kdec.transpose(-1,-2) @ v_raw                                  # [K,V]

# ---- spy on _dfg_inner (CPU fallback path) ----
import furiosa.torch.custom_ops.dfg as dfgmod
calls = {"n": 0}
_orig = dfgmod._dfg_inner
def spy(*a, **kw):
    calls["n"] += 1
    return _orig(*a, **kw)
dfgmod._dfg_inner = spy

# ---- compile + run on NPU ----
m = TacticKernelModule(open(YAML).read())
cm = torch.compile(m, backend=ft.backend)
res = cm(*[t.to(DEV) for t in inputs])
out_npu = res[0].to('cpu')
S_npu   = res[1].to('cpu')

print(f"C={C} K={K} V={V}")
print("out_npu shape", tuple(out_npu.shape), " S_npu shape", tuple(S_npu.shape))

out_ok = torch.allclose(out_npu, out_ref, atol=1e-2)
S_ok   = torch.allclose(S_npu,   S_ref,   atol=1e-2)
out_err = (out_npu - out_ref).abs().max().item()
S_err   = (S_npu   - S_ref  ).abs().max().item()
print("out  vs our-ref allclose(1e-2):", out_ok, " maxerr:", out_err)
print("S    vs our-ref allclose(1e-2):", S_ok,   " maxerr:", S_err)

# NPU output vs the HF-faithful (T=I) single-chunk reference -> should be ~0.
hf_out_ok = torch.allclose(out_npu, hf_out, atol=1e-2)
hf_S_ok   = torch.allclose(S_npu,   hf_state, atol=1e-2)
hf_out_err = (out_npu - hf_out).abs().max().item()
hf_S_err   = (S_npu   - hf_state).abs().max().item()
print("NPU out vs HF-faithful(T=I) out  allclose(1e-2):", hf_out_ok, " maxerr:", hf_out_err)
print("NPU S   vs HF-faithful(T=I) state allclose(1e-2):", hf_S_ok,  " maxerr:", hf_S_err)

print("_dfg_inner call count:", calls["n"], "(0 == ran on NPU)")
print("OVERALL_PASS:", bool(out_ok and S_ok and hf_out_ok and hf_S_ok and calls["n"] == 0))
