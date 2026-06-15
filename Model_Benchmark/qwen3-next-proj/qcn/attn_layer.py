#!/usr/bin/env python3
# =============================================================================
# Qwen3-Coder-Next FULL-ATTENTION layer (a "full_attention" layer, e.g. layer 3:
# (3+1)%4==0) assembled with the matmuls running ON THE NPU, validated against
# HF transformers Qwen3NextAttention with the REAL layer-3 weights.
#
# Model: Qwen/Qwen3-Coder-Next-FP8 (FP8 blockwise weights, dequant via QCNWeights).
# Layer-3 config (full-attn): 16 query heads, 2 kv heads (GQA n_rep=8),
#   head_dim=256, partial RoPE (partial_rotary_factor=0.25 -> rotary_dim=64,
#   rope_theta=5e6), q_norm/k_norm RMSNorm over head_dim, output sigmoid gate.
#
# HF forward (modeling_qwen3_next.py Qwen3NextAttention.forward), verified:
#   q_proj(x) -> view(...,-1,head_dim*2) -> chunk(2,dim=-1) => (query, gate)
#   gate = gate.reshape(...,-1)                       # [B,T, n_heads*head_dim]
#   q = q_norm(query.view(B,T,n_heads,head_dim)).transpose(1,2)   # RMSNorm on head_dim
#   k = k_norm(k_proj(x).view(B,T,n_kv,head_dim)).transpose(1,2)
#   v = v_proj(x).view(B,T,n_kv,head_dim).transpose(1,2)
#   q,k = apply_rotary_pos_emb(q,k,cos,sin)           # partial RoPE on first rotary_dim
#   attn = softmax(repeat_kv(k); q@k^T * scaling + causal_mask) @ repeat_kv(v)
#   attn_output = attn.transpose(1,2).reshape(B,T,-1)
#   attn_output = attn_output * sigmoid(gate)         # <-- the output gate
#   out = o_proj(attn_output)
#
# NPU split (proven dn_linear.yaml EinsumByVe matmul, _dfg_inner==0):
#   * q_proj / k_proj / v_proj / o_proj            -> NPU dn_linear  (y = x @ W^T)
#   * per-head SDPA q@k^T  (scores[i,j]=sum_d q[i,d]k[j,d])  -> NPU dn_linear(q,k)
#   * per-head SDPA attn@v (out[i,d]=sum_j attn[i,j]v[j,d]) -> NPU dn_linear(attn, v^T)
# HOST (no DSL op / not a matmul, exact in fp32):
#   * q_norm/k_norm RMSNorm over head_dim (rsqrt of a row-reduce)
#   * partial RoPE (cos/sin gather + rotate_half)
#   * row-softmax with causal mask  (no NPU softmax kernel; the two matmuls
#     around it run on NPU)
#   * sigmoid gate apply
#
# NPU exec is proven by monkeypatching furiosa.torch.custom_ops.dfg._dfg_inner
# and asserting the call-count delta stays 0 for every NPU matmul stage.
# =============================================================================
import os, sys, glob, json
import torch

# torch FIRST, then furiosa.torch (per the proven recipe).
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule
import furiosa.torch.custom_ops.dfg as dfgmod

sys.path.insert(0, "/home/jun/furiosa/lib/python3.12/site-packages")
import torch.nn.functional as F

REPO = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj"
TK   = os.path.join(REPO, "tk_kernels")
DEV  = os.environ.get("RNGD_DEV", "rngd:2")   # rngd:1/2/3 if 0 busy
PADT = 128                                     # token-axis SRAM tile floor for dn_linear
# QCN_DPE=1 routes npu_linear through the FAST systolic/DPE-MAC matmul kernel
# (dn_linear_dpe.yaml, kind: EinsumByDpe -- ~3.8x faster than the EinsumByVe twin).
# The DPE yaml has an IDENTICAL tensor signature + reads/write tiling to dn_linear.yaml
# (same [t,i]@[o,i]->[t,o], same O*I SRAM budget, same PADT/PADO padding), so it is a
# drop-in swap of the kernel filename. DPE accumulates in reduced (bf16) precision:
# expect ~0.23-1.6% relmean (validate at atol/rtol 1e-2, NEVER 1e-3). The real model
# is FP8/bf16 anyway. Default OFF -> exact EinsumByVe path is unchanged.
DPE  = os.environ.get("QCN_DPE", "0") == "1"
LINEAR_YAML = "dn_linear_dpe.yaml" if DPE else "dn_linear.yaml"

# ---- spy on the CPU-fallback path; the delta must stay 0 for every NPU stage ----
_orig_dfg = dfgmod._dfg_inner
CALLS = {"n": 0}
def _spy(*a, **k):
    CALLS["n"] += 1
    return _orig_dfg(*a, **k)
dfgmod._dfg_inner = _spy

_compiled = {}
def _npu(yaml, inputs):
    """Run a TacticKernel YAML on the NPU. Returns (list-of-cpu-f32-tensors, dfg_delta)."""
    if yaml not in _compiled:
        m = TacticKernelModule(open(os.path.join(TK, yaml)).read())
        _compiled[yaml] = torch.compile(m, backend=ft.backend)
    before = CALLS["n"]
    res = _compiled[yaml](*[t.to(DEV) for t in inputs])
    if not isinstance(res, (tuple, list)):
        res = [res]
    return [r.detach().to("cpu").float() for r in res], CALLS["n"] - before

NPU_STAGES = []                          # (name, dfg_delta) -- dfg_delta MUST be 0
FLOPS = {"npu": 0, "host": 0}            # matmul MAC counts for the NPU-vs-host split

# dn_linear EinsumByVe tiles the [o,i] weight read into SRAM; empirically the
# product O*I must stay <= ~2^20 (e.g. I=2048->O<=512, I=1024->O<=1024) or the
# VE rejects the tile and furiosa falls back to CPU. We split the OUTPUT axis O
# into chunks of size <= floor(2^20 / I) and concat the results (each chunk is an
# independent set of output features -> exact). Token axis is padded to PADT.
OI_BUDGET = 1 << 20


PADO = 32                                # the EinsumByVe output(o) axis tiles to 32


def _npu_linear_chunk(x_ti, W_oi):
    """One dn_linear call: y[t,o]=sum_i x[t,i]*W[o,i] (==F.linear). The token(t) axis
       is padded to PADT and the output(o) axis to a multiple of PADO (the VE tiles
       both; pad rows/cols are zero and do not interact -- sliced back off, exact)."""
    T, I = x_ti.shape
    O = W_oi.shape[0]
    Tp = max(T, PADT)
    Op = ((O + PADO - 1) // PADO) * PADO
    xp = x_ti
    if Tp != T:
        xp = torch.zeros(Tp, I, dtype=x_ti.dtype); xp[:T] = x_ti
    Wp = W_oi
    if Op != O:
        Wp = torch.zeros(Op, I, dtype=W_oi.dtype); Wp[:O] = W_oi
    out, d = _npu(LINEAR_YAML, [xp.contiguous(), Wp.contiguous()])
    return out[0][:T, :O], d


def npu_linear(x_ti, W_oi, name):
    """y[t,o] = sum_i x[t,i]*W[o,i] == F.linear(x, W), on NPU via LINEAR_YAML
       (dn_linear.yaml EinsumByVe, or dn_linear_dpe.yaml EinsumByDpe when QCN_DPE=1).
       Output axis O auto-tiled so each call keeps O*I within the SRAM budget."""
    T, I = x_ti.shape
    O = W_oi.shape[0]
    o_tile = max(1, OI_BUDGET // max(I, 1))
    parts, dtot = [], 0
    for o0 in range(0, O, o_tile):
        y_c, d = _npu_linear_chunk(x_ti, W_oi[o0:o0 + o_tile].contiguous())
        parts.append(y_c)
        dtot += d
    y = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
    NPU_STAGES.append((name, dtot))
    FLOPS["npu"] += T * I * O
    return y                                         # [T,O]


def npu_matmul_AB(A_tk, B_ok, name):
    """C[t,o] = sum_k A[t,k]*B[o,k]  == A @ B^T, on NPU via dn_linear.yaml.
       Used for SDPA q@k^T (B=k) and attn@v (A=attn, B=v^T). Row(t)-padded to PADT."""
    return npu_linear(A_tk, B_ok, name)


# ----------------------------- host helpers ---------------------------------
def rms_norm_headdim(x, weight, eps):
    """Qwen3NextRMSNorm over the last (head_dim) axis. weight stored as raw, the
       module multiplies by (1.0 + weight). x:[...,D]. fp32 throughout."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return out * (1.0 + weight.float())


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(q, k, cos, sin):
    """HF apply_rotary_pos_emb (unsqueeze_dim=1). cos/sin: [B,T,rotary_dim].
       q,k: [B, heads, T, head_dim]. Rotary applied to first rotary_dim, pass-through rest."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_embed, k_embed


def rope_cos_sin(position_ids, head_dim, partial_rotary_factor, rope_theta):
    """HF Qwen3NextRotaryEmbedding default rope (host). Returns cos,sin [B,T,rotary_dim]."""
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    inv_freq_e = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1).float()
    pos_e = position_ids[:, None, :].float()
    freqs = (inv_freq_e @ pos_e).transpose(1, 2)        # [B,T,dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)             # [B,T,dim]
    return emb.cos(), emb.sin()


# ============================================================================
# The NPU full-attention layer forward (B=1).
# ============================================================================
class QCNFullAttentionNPU:
    """Full-attention layer-3 forward for Qwen3-Coder-Next, matmuls on NPU."""

    def __init__(self, weights, cfg, layer_idx=3):
        self.layer_idx = layer_idx
        self.cfg = cfg
        self.n_heads = cfg.num_attention_heads          # 16
        self.n_kv = cfg.num_key_value_heads             # 2
        self.head_dim = cfg.head_dim                    # 256
        self.n_rep = self.n_heads // self.n_kv          # 8
        self.eps = cfg.rms_norm_eps
        self.scaling = self.head_dim ** -0.5
        rp = cfg.rope_parameters
        self.partial = rp.get("partial_rotary_factor", 1.0)
        self.rope_theta = rp["rope_theta"]
        p = f"model.layers.{layer_idx}.self_attn."
        self.Wq = weights.get(p + "q_proj.weight", torch.float32)   # [8192,2048]
        self.Wk = weights.get(p + "k_proj.weight", torch.float32)   # [512,2048]
        self.Wv = weights.get(p + "v_proj.weight", torch.float32)   # [512,2048]
        self.Wo = weights.get(p + "o_proj.weight", torch.float32)   # [2048,4096]
        self.q_norm_w = weights.get(p + "q_norm.weight", torch.float32)  # [256]
        self.k_norm_w = weights.get(p + "k_norm.weight", torch.float32)  # [256]

    def forward(self, hidden_states, position_ids):
        """hidden_states: [B=1, T, hidden]. position_ids: [B=1, T]. Returns [1,T,hidden]."""
        assert hidden_states.shape[0] == 1, "this assembly runs B=1"
        h = hidden_states[0].float()                          # [T, hidden]
        T = h.shape[0]
        H, Dk, nkv = self.n_heads, self.head_dim, self.n_kv

        # ---- q/k/v projections on NPU ----
        q_full = npu_linear(h, self.Wq, "q_proj")             # [T, 16*256*2=8192]
        k_lin  = npu_linear(h, self.Wk, "k_proj")             # [T, 2*256=512]
        v_lin  = npu_linear(h, self.Wv, "v_proj")             # [T, 2*256=512]

        # ---- split q_proj output into (query, gate): view(T,-1,head_dim*2),chunk(2,-1) ----
        q_full = q_full.view(T, H, Dk * 2)
        query, gate = torch.chunk(q_full, 2, dim=-1)          # each [T,H,Dk]
        gate = gate.reshape(T, H * Dk)                        # [T, 4096]

        # ---- q_norm/k_norm RMSNorm over head_dim (HOST) ----
        q = rms_norm_headdim(query, self.q_norm_w, self.eps)  # [T,H,Dk]
        k = rms_norm_headdim(k_lin.view(T, nkv, Dk), self.k_norm_w, self.eps)  # [T,nkv,Dk]
        v = v_lin.view(T, nkv, Dk)                            # [T,nkv,Dk]

        # to [B,heads,T,Dk]
        q = q.transpose(0, 1).unsqueeze(0)                    # [1,H,T,Dk]
        k = k.transpose(0, 1).unsqueeze(0)                    # [1,nkv,T,Dk]
        v = v.transpose(0, 1).unsqueeze(0)                    # [1,nkv,T,Dk]

        # ---- partial RoPE (HOST) ----
        cos, sin = rope_cos_sin(position_ids, Dk, self.partial, self.rope_theta)
        q, k = apply_partial_rope(q, k, cos, sin)             # [1,H,T,Dk], [1,nkv,T,Dk]

        # ---- GQA repeat_kv, then per-head SDPA (matmuls on NPU, softmax on host) ----
        q = q[0]                                              # [H,T,Dk]
        k = k[0]; v = v[0]                                    # [nkv,T,Dk]
        causal = torch.full((T, T), float("-inf")).triu(1)    # [T,T] additive mask
        out_heads = []
        for hd in range(H):
            kv = hd // self.n_rep                             # GQA group
            qh = q[hd]                                        # [T,Dk]
            kh = k[kv]                                        # [T,Dk]
            vh = v[kv]                                        # [T,Dk]
            # scores[i,j] = sum_d qh[i,d]*kh[j,d]  ==  qh @ kh^T   (NPU)
            scores = npu_matmul_AB(qh.contiguous(), kh.contiguous(), f"sdpa_qk[h{hd}]")  # [T,T]
            FLOPS["host"] += 0
            scores = scores * self.scaling + causal           # scale + causal (host)
            attn = torch.softmax(scores, dim=-1)              # row-softmax (HOST)
            # out[i,d] = sum_j attn[i,j]*vh[j,d]  ==  attn @ vh  ==  dn_linear(attn, vh^T) (NPU)
            oh = npu_matmul_AB(attn.contiguous(), vh.t().contiguous(), f"sdpa_av[h{hd}]")  # [T,Dk]
            out_heads.append(oh)
        attn_out = torch.stack(out_heads, dim=0)              # [H,T,Dk]
        attn_out = attn_out.transpose(0, 1).reshape(T, H * Dk)  # [T, 4096]

        # ---- output sigmoid gate (HOST) ----
        attn_out = attn_out * torch.sigmoid(gate)             # [T,4096]

        # ---- o_proj on NPU ----
        out = npu_linear(attn_out, self.Wo, "o_proj")         # [T, hidden]
        return out.unsqueeze(0)                               # [1,T,hidden]

    def forward_decode(self, hidden_states, position_ids, kv_cache):
        """Single-step (or chunk) decode against a carried KV cache.
        hidden_states:[1,Tnew,hidden]; position_ids:[1,Tnew] (absolute positions);
        kv_cache=(K_cached [nkv,Tc,Dk] post-rope, V_cached [nkv,Tc,Dk]).
        Returns ([1,Tnew,hidden] out, (K_full,V_full) updated cache).
        Mirrors forward() but K/V are concatenated cache+new and the causal mask
        spans the full [Tnew, Tc+Tnew] history (each new query attends to all past
        keys + causal among the new ones)."""
        assert hidden_states.shape[0] == 1, "B=1"
        h = hidden_states[0].float()                          # [Tnew, hidden]
        Tnew = h.shape[0]
        H, Dk, nkv = self.n_heads, self.head_dim, self.n_kv
        K_c, V_c = kv_cache
        Tc = K_c.shape[1]

        # q/k/v projections (NPU)
        q_full = npu_linear(h, self.Wq, "q_proj.dec")
        k_lin  = npu_linear(h, self.Wk, "k_proj.dec")
        v_lin  = npu_linear(h, self.Wv, "v_proj.dec")

        q_full = q_full.view(Tnew, H, Dk * 2)
        query, gate = torch.chunk(q_full, 2, dim=-1)
        gate = gate.reshape(Tnew, H * Dk)

        q = rms_norm_headdim(query, self.q_norm_w, self.eps)             # [Tnew,H,Dk]
        k_new = rms_norm_headdim(k_lin.view(Tnew, nkv, Dk), self.k_norm_w, self.eps)
        v_new = v_lin.view(Tnew, nkv, Dk)

        q = q.transpose(0, 1).unsqueeze(0)                              # [1,H,Tnew,Dk]
        k_new = k_new.transpose(0, 1).unsqueeze(0)                      # [1,nkv,Tnew,Dk]
        v_new = v_new.transpose(0, 1).unsqueeze(0)

        cos, sin = rope_cos_sin(position_ids, Dk, self.partial, self.rope_theta)
        q, k_new = apply_partial_rope(q, k_new, cos, sin)

        q = q[0]                                                        # [H,Tnew,Dk]
        k_new = k_new[0]; v_new = v_new[0]                              # [nkv,Tnew,Dk]
        # append new K/V to the cache
        K_full = torch.cat([K_c, k_new], dim=1)                         # [nkv,Tc+Tnew,Dk]
        V_full = torch.cat([V_c, v_new], dim=1)
        Ttot = K_full.shape[1]

        # causal mask over the FULL history: new query i (abs pos Tc+i) attends
        # to all keys j with j <= Tc+i.  Additive [Tnew, Ttot].
        rows = torch.arange(Tnew).unsqueeze(1) + Tc                     # abs query pos
        cols = torch.arange(Ttot).unsqueeze(0)
        causal = torch.where(cols <= rows, 0.0, float("-inf"))         # [Tnew,Ttot]

        out_heads = []
        for hd in range(H):
            kv = hd // self.n_rep
            qh = q[hd]                                                  # [Tnew,Dk]
            kh = K_full[kv]                                             # [Ttot,Dk]
            vh = V_full[kv]                                             # [Ttot,Dk]
            scores = npu_matmul_AB(qh.contiguous(), kh.contiguous(), f"sdpa_qk.dec[h{hd}]")  # [Tnew,Ttot]
            scores = scores * self.scaling + causal
            attn = torch.softmax(scores, dim=-1)
            oh = npu_matmul_AB(attn.contiguous(), vh.t().contiguous(), f"sdpa_av.dec[h{hd}]")  # [Tnew,Dk]
            out_heads.append(oh)
        attn_out = torch.stack(out_heads, dim=0).transpose(0, 1).reshape(Tnew, H * Dk)
        attn_out = attn_out * torch.sigmoid(gate)
        out = npu_linear(attn_out, self.Wo, "o_proj.dec")
        return out.unsqueeze(0), (K_full, V_full)


# ============================================================================
# Validation entry point: HF reference (real layer-3 weights) vs NPU assembly.
# ============================================================================
def _snap():
    d = sorted(glob.glob("/home/jun/.cache/huggingface/hub/"
                         "models--Qwen--Qwen3-Coder-Next-FP8/snapshots/*/"))
    assert d, "model snapshot not found"
    return d[-1]


def main():
    from qcn.loader import QCNWeights
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextAttention, Qwen3NextRotaryEmbedding,
    )

    torch.manual_seed(0)
    LAYER = 3
    assert (LAYER + 1) % 4 == 0, "layer must be a full_attention layer"

    snap = _snap()
    cfg = Qwen3NextConfig.from_pretrained(snap)
    W = QCNWeights(snap=snap)

    T = 24                                                    # short seq
    hidden = torch.randn(1, T, cfg.hidden_size, dtype=torch.float32) * 0.1
    position_ids = torch.arange(T).unsqueeze(0)               # [1,T]

    # ---------------- HF reference with REAL layer-3 weights ----------------
    hf = Qwen3NextAttention(cfg, layer_idx=LAYER).eval()
    p = f"model.layers.{LAYER}.self_attn."
    with torch.no_grad():
        hf.q_proj.weight.copy_(W.get(p + "q_proj.weight", torch.float32))
        hf.k_proj.weight.copy_(W.get(p + "k_proj.weight", torch.float32))
        hf.v_proj.weight.copy_(W.get(p + "v_proj.weight", torch.float32))
        hf.o_proj.weight.copy_(W.get(p + "o_proj.weight", torch.float32))
        hf.q_norm.weight.copy_(W.get(p + "q_norm.weight", torch.float32))
        hf.k_norm.weight.copy_(W.get(p + "k_norm.weight", torch.float32))
    hf = hf.float()

    rotary = Qwen3NextRotaryEmbedding(cfg).eval()
    cos, sin = rotary(hidden, position_ids)                   # [1,T,rotary_dim]
    # causal additive mask [B,1,T,T] for eager attention
    cmask = torch.full((T, T), float("-inf")).triu(1)[None, None]
    with torch.no_grad():
        hf_out, _ = hf(hidden, position_embeddings=(cos, sin), attention_mask=cmask)
    hf_out = hf_out.float()                                   # [1,T,hidden]

    # ---------------- NPU assembly (real layer-3 weights) ----------------
    npu_layer = QCNFullAttentionNPU(W, cfg, layer_idx=LAYER)
    out = npu_layer.forward(hidden, position_ids)             # [1,T,hidden]

    # ---------------- compare ----------------
    maxerr = (out - hf_out).abs().max().item()
    rel = maxerr / (hf_out.abs().max().item() + 1e-9)
    ok = torch.allclose(out, hf_out, atol=1e-2)

    print("=" * 74)
    print(f"Qwen3-Coder-Next FULL-ATTENTION layer {LAYER}  (real FP8-dequant weights)")
    print(f"config: hidden={cfg.hidden_size} n_heads={cfg.num_attention_heads} "
          f"n_kv={cfg.num_key_value_heads} head_dim={cfg.head_dim} "
          f"n_rep={cfg.num_attention_heads // cfg.num_key_value_heads}")
    print(f"        partial_rotary={cfg.rope_parameters['partial_rotary_factor']} "
          f"rotary_dim={int(cfg.head_dim * cfg.rope_parameters['partial_rotary_factor'])} "
          f"rope_theta={cfg.rope_parameters['rope_theta']} eps={cfg.rms_norm_eps} T={T}")
    print("-" * 74)
    print("NPU matmul stages (dfg_delta MUST be 0 == ran on NPU):")
    all_npu = True
    # collapse the 16x2 per-head SDPA stages for readability
    proj_stages = [(n, d) for n, d in NPU_STAGES if "sdpa" not in n]
    qk = [d for n, d in NPU_STAGES if n.startswith("sdpa_qk")]
    av = [d for n, d in NPU_STAGES if n.startswith("sdpa_av")]
    for n, d in proj_stages:
        flag = "NPU" if d == 0 else f"CPU-FALLBACK(+{d})"
        all_npu &= (d == 0)
        print(f"   {n:14s} : {flag}")
    print(f"   sdpa_qk x{len(qk):<3d}  : {'NPU' if sum(qk)==0 else 'CPU-FALLBACK'}  "
          f"(per-head q@k^T)")
    print(f"   sdpa_av x{len(av):<3d}  : {'NPU' if sum(av)==0 else 'CPU-FALLBACK'}  "
          f"(per-head attn@v)")
    all_npu &= (sum(qk) == 0 and sum(av) == 0)
    print("-" * 74)
    print("HOST stages (no DSL op / not a matmul):")
    print("   q_norm/k_norm RMSNorm over head_dim (rsqrt of a row-reduce)")
    print("   partial RoPE (cos/sin + rotate_half), rotary_dim=64")
    print("   row-softmax with causal mask (no NPU softmax kernel)")
    print("   sigmoid output gate apply")
    print("-" * 74)
    tot = FLOPS["npu"] + FLOPS["host"]
    fnpu = 100.0 * FLOPS["npu"] / tot if tot else 0.0
    print("MATMUL FLOP (MAC) BREAKDOWN -- NPU vs HOST:")
    print(f"   NPU  matmul MACs : {FLOPS['npu']:>14,d}  ({fnpu:6.2f}%)")
    print(f"     q/k/v/o_proj   + per-head q@k^T + per-head attn@v  (all NPU dn_linear)")
    print(f"   HOST matmul MACs : {FLOPS['host']:>14,d}  ({100.0-fnpu:6.2f}%)")
    print("-" * 74)
    print(f"allclose(atol=1e-2) : {ok}")
    print(f"maxerr              : {maxerr:.3e}   (rel {rel:.3e})")
    print(f"total _dfg_inner    : {CALLS['n']} (0 == every NPU matmul ran on NPU)")
    print(f"ALL_MATMULS_ON_NPU  : {all_npu and CALLS['n'] == 0}")
    print(f"OVERALL_PASS        : {bool(ok and all_npu and CALLS['n'] == 0)}")
    print("=" * 74)


if __name__ == "__main__":
    main()
