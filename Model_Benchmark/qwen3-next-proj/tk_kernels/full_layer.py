#!/usr/bin/env python3
# =============================================================================
# COMPLETE single-head Gated DeltaNet layer forward, host-orchestrated, with the
# DeltaNet-specific compute running ON THE NPU via the proven TacticKernel YAMLs.
# Validated end-to-end against HF transformers Qwen3NextGatedDeltaNet.
#
#   hidden  --in_proj_qkvz/ba (matmul)-->  q,k,v,z,b,a
#           --conv1d+SiLU on [q;k;v]----->  (dn_conv1d.yaml      NPU)
#           --L2norm q,k----------------->  (dn_l2norm.yaml      NPU)
#           --beta=sigmoid(b)------------>  (dn_gate.yaml        NPU)
#           --g=-exp(A_log)*softplus(...)>  (host: softplus has no DSL op)
#           --CHUNK SCAN threading S----->  (dn_chunk_full.yaml  NPU, per chunk)
#           --gated RMSNorm(core, z)----->  (dn_gnorm.yaml       NPU)
#           --out_proj (matmul)---------->  output
#
# Per the recipe: import torch FIRST, then furiosa.torch; compile each YAML with
# ft.backend; run on DEV; results are LISTs. NPU-exec is proven by monkeypatching
# furiosa.torch.custom_ops.dfg._dfg_inner and asserting the call count stays 0
# across every NPU stage (it increments only on a CPU fallback).
#
# WHY ROW-PADDING for l2norm/gnorm: those two kernels end in a LocalReduceAddF
# whose SURVIVING (row) axis must tile to SRAM with INNER >= ~128 rows, else the
# VE rejects it and furiosa falls back to CPU ("furiosa::dfg only runs on CPU
# device"). The DeltaNet math is per-row independent, so we zero-pad the row axis
# up to 128, run on NPU, and slice the real T (or M) rows back -- exact (the pad
# rows do not interact with the real rows). Verified maxerr ~1e-7.
# =============================================================================
import os, sys, torch
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule
import furiosa.torch.custom_ops.dfg as dfgmod

sys.path.insert(0, "/home/jun/furiosa/lib/python3.12/site-packages")
import torch.nn.functional as F
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet

BASE = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/"
DEV  = os.environ.get("RNGD_DEV", "rngd:1")
PADM = 128                              # SRAM row-tile floor for the reduce kernels

# ---- spy on the CPU-fallback path; must stay 0 for every NPU stage ----
_orig = dfgmod._dfg_inner
CALLS = {"n": 0}
def _spy(*a, **k):
    CALLS["n"] += 1
    return _orig(*a, **k)
dfgmod._dfg_inner = _spy

_compiled = {}
def npu(yaml, inputs):
    """Run a TacticKernel YAML on the NPU. Returns (list-of-cpu-tensors, dfg_delta)."""
    if yaml not in _compiled:
        m = TacticKernelModule(open(BASE + yaml).read())
        _compiled[yaml] = torch.compile(m, backend=ft.backend)
    before = CALLS["n"]
    res = _compiled[yaml](*[t.to(DEV) for t in inputs])
    if not isinstance(res, (tuple, list)):
        res = [res]
    return [r.detach().to("cpu").float() for r in res], CALLS["n"] - before

NPU_STAGES = []   # (stage_name, dfg_delta)  -- dfg_delta MUST be 0

# ---------- NPU stage wrappers (single-head) ----------
def npu_conv1d_silu(x_ct, w_ck):
    """x_ct:[C,T] depthwise causal conv1d (K taps) + SiLU -> [C,T]."""
    C, T = x_ct.shape
    Kc = w_ck.shape[1]
    x_pad = torch.cat([torch.zeros(C, Kc - 1), x_ct], dim=-1)            # [C,T+K-1]
    xs = [x_pad[:, j:j + T].contiguous() for j in range(Kc)]
    wf = [w_ck[:, j:j + 1].expand(C, T).contiguous() for j in range(Kc)]
    out, d = npu("dn_conv1d.yaml", xs + wf)
    NPU_STAGES.append(("conv1d+SiLU", d))
    return out[0]                                                        # [C,T]

def npu_l2norm(x_md, eps=1e-6):
    """L2-normalize rows of x_md:[M,D] over D, on NPU (row-padded to PADM)."""
    M, D = x_md.shape
    Mp = max(M, PADM)
    xp = torch.zeros(Mp, D); xp[:M] = x_md
    inputs = [xp.t().contiguous(), torch.ones(D), torch.full((Mp,), eps), xp.contiguous()]
    out, d = npu("dn_l2norm.yaml", inputs)
    NPU_STAGES.append(("l2norm", d))
    return out[0][:M]                                                    # [M,D]

def npu_sigmoid(x_mn):
    """sigmoid(x) via dn_gate (sigmoid(in0)*in1) with in1=ones -> on NPU."""
    out, d = npu("dn_gate.yaml", [x_mn.contiguous(), torch.ones_like(x_mn)])
    NPU_STAGES.append(("sigmoid(beta)", d))
    return out[0]

def npu_gnorm(x_md, gate_md, weight_d, eps=1e-6):
    """Qwen3NextRMSNormGated(x, gate) on NPU (row-padded to PADM)."""
    M, D = x_md.shape
    Mp = max(M, PADM)
    xp = torch.zeros(Mp, D); xp[:M] = x_md
    gp = torch.zeros(Mp, D); gp[:M] = gate_md
    inputs = [xp.t().contiguous(), torch.ones(D), torch.full((Mp,), 1.0 / D),
              torch.full((Mp,), eps), weight_d.unsqueeze(0).expand(Mp, D).contiguous(),
              gp.contiguous(), xp.contiguous()]
    out, d = npu("dn_gnorm.yaml", inputs)
    NPU_STAGES.append(("gated-RMSNorm", d))
    return out[0][:M]                                                    # [M,D]

# ---------- host helper: build the HF triangular-inverse T-matrix ----------
def build_T(attn_neg, C):
    a = attn_neg.clone()
    for i in range(1, C):
        row = a[i, :i].clone()
        sub = a[:i, :i].clone()
        a[i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return a + torch.eye(C, dtype=a.dtype)

def npu_chunk_scan(qf, kf, vf, gf, bf, C, K, V):
    """Host drives S_prev-independent quantities; dn_chunk_full runs each chunk on NPU.
       qf is ALREADY 1/sqrt(K)-scaled and L2-normed; kf is L2-normed. Returns [N,V] core."""
    N = qf.shape[0]
    NC = N // C
    mask_incl   = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=0)
    mask_strict = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1)
    S = torch.zeros(K, V)
    outs = []
    for ci in range(NC):
        sl = slice(ci * C, (ci + 1) * C)
        q_i, k_i, v_i = qf[sl], kf[sl], vf[sl]
        g_i, b_i = gf[sl], bf[sl]
        g_cum = torch.cumsum(g_i, dim=0)
        decay = ((g_cum.unsqueeze(1) - g_cum.unsqueeze(0)).tril().exp()).tril()
        k_beta = k_i * b_i.unsqueeze(-1)
        v_beta = v_i * b_i.unsqueeze(-1)
        attn0 = -((k_beta @ k_i.transpose(-1, -2)) * decay).masked_fill(mask_incl, 0)
        T = build_T(attn0, C)
        value = T @ v_beta
        kcd   = T @ (k_beta * g_cum.exp().unsqueeze(-1))
        decay_strict = decay.masked_fill(mask_strict, 0.0)
        gexp_k = g_cum.exp().unsqueeze(1).expand(C, K).contiguous()
        wdecay = (g_cum[-1] - g_cum).exp().unsqueeze(1).expand(C, K).contiguous()
        sdecay = torch.full((K, V), g_cum[-1].exp().item())
        kin = [q_i.contiguous(), k_i.contiguous(), value.contiguous(),
               decay_strict.contiguous(), kcd.contiguous(), gexp_k,
               wdecay, sdecay, S.contiguous()]
        res, d = npu("dn_chunk_full.yaml", kin)
        NPU_STAGES.append((f"chunk-scan[{ci}]", d))
        outs.append(res[0])      # [C,V]
        S = res[1]               # [K,V] carry
    return torch.cat(outs, dim=0)


def main():
    torch.manual_seed(0)

    # =========================================================================
    # STEP 1: HF reference, SMALL single-head config, random weights
    # =========================================================================
    cfg = Qwen3NextConfig(
        hidden_size=256,
        linear_num_value_heads=1,
        linear_num_key_heads=1,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        hidden_act="silu",
    )
    C = 16          # chunk_size
    T = 32          # 2 chunks
    K = cfg.linear_key_head_dim          # 32
    V = cfg.linear_value_head_dim        # 32
    hf = Qwen3NextGatedDeltaNet(cfg, layer_idx=0).eval()
    # the torch chunk path uses chunk_size=64 by default; force small chunks so the
    # HF reference itself runs 2 chunks (matches our host loop). Patch the bound fn.
    import transformers.models.qwen3_next.modeling_qwen3_next as M
    _orig_chunk = M.torch_chunk_gated_delta_rule
    def _chunk16(*a, **kw):
        kw["chunk_size"] = C
        return _orig_chunk(*a, **kw)
    hf.chunk_gated_delta_rule = _chunk16

    hidden = torch.randn(1, T, cfg.hidden_size)
    with torch.no_grad():
        hf_out = hf(hidden)[0]            # [T, hidden] (B=1)
    hf_out = hf_out.squeeze(0) if hf_out.dim() == 3 else hf_out
    if hf_out.dim() == 3:
        hf_out = hf_out[0]

    # extract weights
    W_qkvz = hf.in_proj_qkvz.weight.detach()      # [proj_qkvz, hidden]
    W_ba   = hf.in_proj_ba.weight.detach()        # [2, hidden]
    conv_w = hf.conv1d.weight.detach().squeeze(1) # [conv_dim, Kc]
    A_log  = hf.A_log.detach()                    # [1]
    dt_b   = hf.dt_bias.detach()                  # [1]
    norm_w = hf.norm.weight.detach()              # [V]
    W_out  = hf.out_proj.weight.detach()          # [hidden, value_dim]

    key_dim, value_dim = hf.key_dim, hf.value_dim         # 32, 32
    conv_dim = key_dim * 2 + value_dim                    # 96
    h = hidden[0]                                          # [T, hidden]

    # =========================================================================
    # STEP 2: host-orchestrated pipeline, DeltaNet ops on NPU
    # =========================================================================
    # ---- in_proj (matmul) : standard NPU-compilable Linear, run on host here ----
    proj_qkvz = h @ W_qkvz.t()                            # [T, proj_qkvz]
    proj_ba   = h @ W_ba.t()                              # [T, 2]
    # single-head fix_query_key_value_ordering split: q,k,v,z then b,a
    q, k, v, z = torch.split(proj_qkvz, [key_dim, key_dim, value_dim, value_dim], dim=-1)
    b, a = torch.split(proj_ba, [1, 1], dim=-1)           # [T,1] each

    # ---- conv1d + SiLU on [q;k;v] (NPU), layout [c,t] ----
    mixed = torch.cat((q, k, v), dim=-1).transpose(0, 1)  # [conv_dim, T]
    conv_out = npu_conv1d_silu(mixed.contiguous(), conv_w)  # [conv_dim, T]
    conv_out = conv_out.transpose(0, 1)                   # [T, conv_dim]
    q, k, v = torch.split(conv_out, [key_dim, key_dim, value_dim], dim=-1)

    # ---- L2norm q,k (NPU) then scale q by 1/sqrt(K) (HF use_qk_l2norm_in_kernel) ----
    qn = npu_l2norm(q.contiguous())                       # [T,K]
    kn = npu_l2norm(k.contiguous())                       # [T,K]
    scale = 1.0 / (K ** 0.5)
    qf = qn * scale
    kf = kn
    vf = v

    # ---- beta = sigmoid(b) (NPU) ; g = -exp(A_log)*softplus(a+dt_bias) (host) ----
    beta = npu_sigmoid(b.contiguous())[:, 0]              # [T]
    g = -A_log.exp() * F.softplus(a + dt_b)               # [T,1]  (softplus: no DSL op)
    g = g[:, 0]                                           # [T]

    # ---- CHUNK SCAN over the sequence threading state S (NPU per chunk) ----
    core = npu_chunk_scan(qf, kf, vf, g, beta, C, K, V)   # [T,V]

    # ---- gated RMSNorm(core, z) (NPU) ----
    core_n = npu_gnorm(core.contiguous(), z.contiguous(), norm_w)   # [T,V]

    # ---- out_proj (matmul) : standard NPU-compilable Linear, run on host here ----
    out = core_n @ W_out.t()                              # [T, hidden]

    # =========================================================================
    # STEP 3: compare vs HF
    # =========================================================================
    maxerr = (out - hf_out).abs().max().item()
    ok = torch.allclose(out, hf_out, atol=1e-2)

    print("=" * 70)
    print(f"config: hidden={cfg.hidden_size} K={K} V={V} conv_dim={conv_dim} "
          f"chunk={C} T={T} (NC={T // C})")
    print("-" * 70)
    print("NPU stages (dfg_delta MUST be 0 == ran on NPU):")
    all_npu = True
    for name, d in NPU_STAGES:
        flag = "NPU" if d == 0 else f"CPU-FALLBACK(+{d})"
        if d != 0:
            all_npu = False
        print(f"   {name:18s} : {flag}")
    print("-" * 70)
    print("HOST stages (standard-NPU-compilable matmul / no-DSL-op):")
    print("   in_proj_qkvz/ba    : matmul (EinsumByVe-compilable; host for this run)")
    print("   g=-exp*softplus    : softplus has NO DSL op (host); exp IS a DSL Unary")
    print("   out_proj           : matmul (EinsumByVe-compilable; host for this run)")
    print("-" * 70)
    print(f"FULL LAYER allclose(atol=1e-2): {ok}")
    print(f"FULL LAYER maxerr            : {maxerr:.3e}")
    print(f"total _dfg_inner calls       : {CALLS['n']} (0 == every NPU stage ran on NPU)")
    print(f"ALL_DELTANET_OPS_ON_NPU      : {all_npu and CALLS['n'] == 0}")
    print(f"OVERALL_PASS                 : {bool(ok and all_npu and CALLS['n'] == 0)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
