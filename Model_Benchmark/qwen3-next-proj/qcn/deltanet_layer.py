#!/usr/bin/env python3
# =============================================================================
# REAL-CONFIG multi-head Gated DeltaNet layer for Qwen3-Coder-Next-FP8.
#
# This is the REAL-dims generalization of tk_kernels/full_layer_npu.py (proven
# single-head, maxerr ~1e-6).  Qwen3-Coder-Next DeltaNet layer dims:
#     linear_num_key_heads   = 16   (q,k heads)
#     linear_num_value_heads = 32   (v,z heads)  -> n_rep = 32/16 = 2
#     head_k_dim = head_v_dim = 128
#     conv kernel 4,  conv_dim = key_dim*2 + value_dim = 2048*2 + 4096 = 8192
#     in_proj_qkvz : hidden(2048) -> 12288 = q(2048)+k(2048)+v(4096)+z(4096)
#     in_proj_ba   : hidden(2048) -> 64    = b(32)+a(32)
#
# HF (transformers Qwen3NextGatedDeltaNet) reshapes q,k to 16 heads then
# repeat_interleave(2) so the chunk scan runs on 32 heads (K=V=128) -- and the
# gated-delta recurrence is per-head INDEPENDENT.  So we LOOP the proven
# single-head dn_chunk_full.yaml across the 32 heads, threading a per-head
# state S[h] of shape [K,V].  Everything with a matmul / elementwise / reduce
# runs ON THE NPU; only the two inherently-host pieces remain on host:
#   (a) g = -exp(A_log)*softplus(a+dt_bias)  (softplus has no DSL op),
#   (b) the S_prev-independent per-chunk precompute (tri-inverse T-matrix via
#       the sequential refinement loop, cumsum, decay_mask).
#
# DeltaNet-specific NPU ops are proven on-device by monkeypatching
# furiosa.torch.custom_ops.dfg._dfg_inner and asserting the call count stays 0.
#
#   forward(hidden_states, weights, state=None) -> (out, new_state)
#     hidden_states : [B,T,H] or [T,H]   (H = hidden_size = 2048)
#     weights       : dict with keys in_proj_qkvz, in_proj_ba, conv1d_weight,
#                     A_log, dt_bias, norm_weight, out_proj   (all torch f32)
#     state         : optional [num_v_heads, K, V] incoming recurrent state
#     returns       : (out [B,T,H] (or [T,H]),  new_state [num_v_heads,K,V])
# =============================================================================
import os
import torch
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule
import furiosa.torch.custom_ops.dfg as dfgmod
import torch.nn.functional as F

_BASE = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/"
_PADM = 128                       # SRAM row-tile floor for the reduce kernels


# ---- spy on the CPU-fallback path; must stay 0 for every NPU stage ----
_orig_dfg = dfgmod._dfg_inner
_CALLS = {"n": 0}
def _spy(*a, **k):
    _CALLS["n"] += 1
    return _orig_dfg(*a, **k)
if dfgmod._dfg_inner is not _spy:
    dfgmod._dfg_inner = _spy


_compiled = {}
# Count of NPU kernel dispatches (one per compiled-module invocation). _CALLS above
# counts only CPU fallbacks (must stay 0); _NPU_DISPATCHES counts actual on-NPU runs
# so the looped-vs-batched chunk-scan dispatch reduction can be measured.
_NPU_DISPATCHES = {"n": 0, "by_yaml": {}}
def _npu(yaml, inputs, dev):
    """Run a TacticKernel YAML on the NPU. Returns (list-of-cpu-tensors, dfg_delta)."""
    if yaml not in _compiled:
        m = TacticKernelModule(open(_BASE + yaml).read())
        _compiled[yaml] = torch.compile(m, backend=ft.backend)
    _NPU_DISPATCHES["n"] += 1
    _NPU_DISPATCHES["by_yaml"][yaml] = _NPU_DISPATCHES["by_yaml"].get(yaml, 0) + 1
    before = _CALLS["n"]
    try:
        res = _compiled[yaml](*[t.to(dev) for t in inputs])
    except Exception as e:
        shapes = [tuple(t.shape) for t in inputs]
        raise RuntimeError(f"[NPU {yaml}] failed; input shapes={shapes}: {e}") from e
    if not isinstance(res, (tuple, list)):
        res = [res]
    return [r.detach().to("cpu").float() for r in res], _CALLS["n"] - before


def _build_T(attn_neg, C):
    """HF triangular-inverse refinement (modeling_qwen3_next L511-515), host."""
    a = attn_neg.clone()
    for i in range(1, C):
        row = a[i, :i].clone()
        sub = a[:i, :i].clone()
        a[i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return a + torch.eye(C, dtype=a.dtype)


class DeltaNetLayer:
    """Reusable REAL-config Gated DeltaNet layer; DeltaNet ops run on the NPU."""

    def __init__(self, config, dev=None, chunk_size=64):
        self.hidden_size = config["hidden_size"]
        self.num_k_heads = config["linear_num_key_heads"]      # 16
        self.num_v_heads = config["linear_num_value_heads"]    # 32
        self.head_k_dim  = config["linear_key_head_dim"]       # 128
        self.head_v_dim  = config["linear_value_head_dim"]     # 128
        self.conv_kernel = config["linear_conv_kernel_dim"]    # 4
        self.eps         = config.get("rms_norm_eps", 1e-6)
        self.key_dim   = self.head_k_dim * self.num_k_heads    # 2048
        self.value_dim = self.head_v_dim * self.num_v_heads    # 4096
        self.conv_dim  = self.key_dim * 2 + self.value_dim     # 8192
        self.n_rep     = self.num_v_heads // self.num_k_heads   # 2
        self.dev = dev or os.environ.get("RNGD_DEV", "rngd:0")
        self.chunk_size = chunk_size
        # tally of which stages ran on NPU (name -> dfg_delta; 0 == on NPU)
        self.npu_stages = []

    # ---------------- NPU stage wrappers ----------------
    _OTILE = 512    # EinsumByVe write tile: O up to 512 runs on NPU, >512 falls back
    _ITILE = 2048   # EinsumByVe reduce tile: I up to 2048 OK on NPU, 4096 falls back

    def _linear(self, x_ti, W_oi, name):
        """y[t,o] = sum_i x[t,i]*W[o,i] == F.linear(x,W), NPU dn_linear.yaml.
        Token axis padded to _PADM if below it (pad rows don't interact with
        real rows, sliced back off).  Two tilings keep every matmul on the NPU:
          - OUTPUT axis O tiled in <=_OTILE column-blocks of W, concatenated
            (the EinsumByVe write tile rejects O>~512 -> CPU fallback otherwise).
          - REDUCE axis I tiled in <=_ITILE blocks, partial matmuls SUMMED
            (the reduce tile rejects I>~2048; sum-of-partials is exact since
            sum_i = sum over I-blocks of the within-block sum).
        out_proj needs the I tiling (value_dim=4096); in_proj needs only O."""
        T, I = x_ti.shape
        O = W_oi.shape[0]
        keep = T
        if T < _PADM:
            xp = torch.zeros(_PADM, I, dtype=x_ti.dtype); xp[:T] = x_ti
            x_use = xp
        else:
            x_use = x_ti
        cols = []
        dtot = 0
        for o0 in range(0, O, self._OTILE):
            acc = None
            for i0 in range(0, I, self._ITILE):
                xb = x_use[:, i0:i0 + self._ITILE].contiguous()
                Wb = W_oi[o0:o0 + self._OTILE, i0:i0 + self._ITILE].contiguous()
                out, d = _npu("dn_linear.yaml", [xb, Wb], self.dev)
                dtot += d
                acc = out[0] if acc is None else acc + out[0]
            cols.append(acc[:keep])
        self.npu_stages.append((name, dtot))
        return torch.cat(cols, dim=-1)

    def _conv1d_silu(self, x_ct, w_ck, conv_prefix=None):
        """x_ct:[C,T] depthwise causal conv1d (K taps) + SiLU -> [C,T] on NPU.
        conv_prefix (optional) [C, Kc-1]: the previous Kc-1 conv inputs to prepend
        instead of zeros (the carried conv state during decode; HF
        torch_causal_conv1d_update L443-458).  Returns (out [C,T], new_prefix
        [C,Kc-1] == last Kc-1 columns of the full conv-input)."""
        C, T = x_ct.shape
        Kc = w_ck.shape[1]
        if conv_prefix is None:
            conv_prefix = torch.zeros(C, Kc - 1, dtype=x_ct.dtype)
        x_pad = torch.cat([conv_prefix, x_ct], dim=-1)                   # [C,T+K-1]
        xs = [x_pad[:, j:j + T].contiguous() for j in range(Kc)]
        wf = [w_ck[:, j:j + 1].expand(C, T).contiguous() for j in range(Kc)]
        out, d = _npu("dn_conv1d.yaml", xs + wf, self.dev)
        self.npu_stages.append(("conv1d+SiLU", d))
        new_prefix = x_pad[:, -(Kc - 1):].clone() if Kc > 1 else conv_prefix
        return out[0], new_prefix

    def _l2norm(self, x_md, name):
        """L2-normalize rows of x_md:[M,D] over D, on NPU (row-padded to _PADM)."""
        M, D = x_md.shape
        Mp = max(M, _PADM)
        xp = torch.zeros(Mp, D); xp[:M] = x_md
        inputs = [xp.t().contiguous(), torch.ones(D), torch.full((Mp,), self.eps), xp.contiguous()]
        out, d = _npu("dn_l2norm.yaml", inputs, self.dev)
        self.npu_stages.append((name, d))
        return out[0][:M]

    def _sigmoid(self, x_mn):
        """sigmoid(x) via dn_gate (sigmoid(in0)*in1) with in1=ones -> on NPU."""
        out, d = _npu("dn_gate.yaml", [x_mn.contiguous(), torch.ones_like(x_mn)], self.dev)
        self.npu_stages.append(("sigmoid(beta)", d))
        return out[0]

    def _gnorm(self, x_md, gate_md, weight_d):
        """Qwen3NextRMSNormGated(x, gate) over head_v_dim, NPU (row-padded to _PADM)."""
        M, D = x_md.shape
        Mp = max(M, _PADM)
        xp = torch.zeros(Mp, D); xp[:M] = x_md
        gp = torch.zeros(Mp, D); gp[:M] = gate_md
        inputs = [xp.t().contiguous(), torch.ones(D), torch.full((Mp,), 1.0 / D),
                  torch.full((Mp,), self.eps),
                  weight_d.unsqueeze(0).expand(Mp, D).contiguous(),
                  gp.contiguous(), xp.contiguous()]
        out, d = _npu("dn_gnorm.yaml", inputs, self.dev)
        self.npu_stages.append(("gated-RMSNorm", d))
        return out[0][:M]

    # ---------------- HEAD-BATCHED chunk scan (NPU dn_chunk_full_mh per chunk) -------
    def _build_T_batched(self, attn_neg, C):
        """HF triangular-inverse refinement, batched over a leading head axis.
        attn_neg:[H,C,C] -> T:[H,C,C].  Same recurrence as host _build_T but the
        per-head independent rows are refined for ALL heads at once (host work only;
        the inverse is small CxC and has no S_prev dependency)."""
        a = attn_neg.clone()
        for i in range(1, C):
            row = a[:, i, :i].clone()                             # [H,i]
            sub = a[:, :i, :i].clone()                            # [H,i,i]
            a[:, i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)  # [H,i]
        return a + torch.eye(C, dtype=a.dtype)

    def _chunk_scan(self, qf, kf, vf, gf, bf, S):
        """ALL heads at once.  qf/kf/vf:[H,N,*], gf/bf:[H,N], S:[H,K,V].
        Host drives the S_prev-independent per-chunk precompute BATCHED over H, then
        calls dn_chunk_full_mh.yaml ONCE per chunk (all H heads -> one NPU dispatch),
        threading the batched recurrent state S[H,K,V].  Returns (core [H,N,V],
        S_next [H,K,V]).  This replaces the old 32-head python loop (32 dispatches
        per chunk) with a single head-batched dispatch per chunk -- ~H x fewer."""
        H, N, K = qf.shape
        V = vf.shape[2]
        C = self.chunk_size
        # pad sequence up to a chunk multiple (HF pads with zeros), exactly as HF
        pad = (C - N % C) % C
        if pad:
            qf = F.pad(qf, (0, 0, 0, pad)); kf = F.pad(kf, (0, 0, 0, pad))
            vf = F.pad(vf, (0, 0, 0, pad)); gf = F.pad(gf, (0, pad)); bf = F.pad(bf, (0, pad))
        Ntot = qf.shape[1]
        NC = Ntot // C
        mask_incl   = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=0)
        mask_strict = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1)
        outs = []
        for ci in range(NC):
            sl = slice(ci * C, (ci + 1) * C)
            q_i, k_i, v_i = qf[:, sl], kf[:, sl], vf[:, sl]      # [H,C,K]/[H,C,V]
            g_i, b_i = gf[:, sl], bf[:, sl]                      # [H,C]
            g_cum = torch.cumsum(g_i, dim=1)                     # [H,C]
            # decay[H,C,C] = tril(exp(g_cum[:,c]-g_cum[:,d])) lower-tri
            diff = g_cum.unsqueeze(2) - g_cum.unsqueeze(1)       # [H,C,C]
            decay = diff.tril().exp().tril()
            k_beta = k_i * b_i.unsqueeze(-1)                     # [H,C,K]
            v_beta = v_i * b_i.unsqueeze(-1)                     # [H,C,V]
            # host: per-head tri-inverse T (no S_prev dep), batched over H
            kkT = torch.matmul(k_beta, k_i.transpose(-1, -2))    # [H,C,C]
            attn0 = -(kkT * decay).masked_fill(mask_incl, 0)
            T = self._build_T_batched(attn0, C)                  # [H,C,C]
            value = torch.matmul(T, v_beta)                      # [H,C,V]
            kcd   = torch.matmul(T, k_beta * g_cum.exp().unsqueeze(-1))   # [H,C,K]
            decay_strict = decay.masked_fill(mask_strict, 0.0)   # [H,C,C]
            gexp_k = g_cum.exp().unsqueeze(2).expand(H, C, K).contiguous()  # [H,C,K]
            wdecay = (g_cum[:, -1:] - g_cum).exp().unsqueeze(2).expand(H, C, K).contiguous()
            sdecay = g_cum[:, -1].exp().view(H, 1, 1).expand(H, K, V).contiguous()  # [H,K,V]
            kin = [q_i.contiguous(), k_i.contiguous(), value.contiguous(),
                   decay_strict.contiguous(), kcd.contiguous(), gexp_k,
                   wdecay, sdecay, S.contiguous()]
            res, d = _npu("dn_chunk_full_mh.yaml", kin, self.dev)
            self.npu_stages.append((f"chunk-scan[c{ci}]", d))
            outs.append(res[0])      # [H,C,V]
            S = res[1]               # [H,K,V] carry
        core = torch.cat(outs, dim=1)[:, :N]                    # [H,N,V] drop chunk pad
        return core, S

    # ---------------- full layer ----------------
    def forward(self, hidden_states, weights, state=None, conv_state=None,
                return_conv=False):
        squeeze_back = (hidden_states.dim() == 2)
        if squeeze_back:
            hidden_states = hidden_states.unsqueeze(0)
        B, T, H = hidden_states.shape
        assert B == 1, "this orchestrated path is written for B=1"
        assert H == self.hidden_size
        h = hidden_states[0].float()                            # [T,H]

        W_qkvz   = weights["in_proj_qkvz"]                      # [12288, H]
        W_ba     = weights["in_proj_ba"]                        # [64, H]
        conv_w   = weights["conv1d_weight"]                     # [conv_dim, 1, Kc] or [conv_dim, Kc]
        A_log    = weights["A_log"]                             # [num_v_heads]
        dt_bias  = weights["dt_bias"]                           # [num_v_heads]
        norm_w   = weights["norm_weight"]                       # [head_v_dim]
        W_out    = weights["out_proj"]                          # [H, value_dim]
        if conv_w.dim() == 3:
            conv_w = conv_w.squeeze(1)                          # [conv_dim, Kc]

        nk, nv = self.num_k_heads, self.num_v_heads
        hk, hv = self.head_k_dim, self.head_v_dim
        rep = self.n_rep

        # ---- in_proj on NPU ----
        proj_qkvz = self._linear(h, W_qkvz, "in_proj_qkvz")     # [T,12288]
        proj_ba   = self._linear(h, W_ba,   "in_proj_ba")       # [T,64]

        # ---- HF fix_query_key_value_ordering (interleaved per k-head layout) ----
        # qkvz reshape -> [T, nk, 2*hk + 2*hv*rep] then split q,k,v,z
        qkvz = proj_qkvz.view(T, nk, 2 * hk + 2 * hv * rep)
        split_qkvz = [hk, hk, rep * hv, rep * hv]
        query, key, value, z = torch.split(qkvz, split_qkvz, dim=-1)
        value = value.reshape(T, nv, hv)                        # [T,32,128]
        z     = z.reshape(T, nv, hv)                            # [T,32,128]
        ba = proj_ba.view(T, nk, 2 * rep)
        b, a = torch.split(ba, [rep, rep], dim=-1)
        b = b.reshape(T, nv)                                    # [T,32]
        a = a.reshape(T, nv)                                    # [T,32]
        # query,key currently [T, nk, hk]; flatten to channels for conv
        query = query.reshape(T, nk * hk)                       # [T,2048]
        key   = key.reshape(T, nk * hk)                         # [T,2048]
        value_flat = value.reshape(T, nv * hv)                  # [T,4096]

        # ---- conv1d + SiLU over [q;k;v] (conv_dim=8192) on NPU, layout [c,t] ----
        mixed = torch.cat((query, key, value_flat), dim=-1).transpose(0, 1)   # [conv_dim,T]
        conv_out, new_conv_state = self._conv1d_silu(
            mixed.contiguous(), conv_w, conv_prefix=conv_state)               # [conv_dim,T]
        conv_out = conv_out.transpose(0, 1)                                   # [T,conv_dim]
        query, key, value_flat = torch.split(
            conv_out, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        # ---- reshape to heads ----
        query = query.reshape(T, nk, hk)                        # [T,16,128]
        key   = key.reshape(T, nk, hk)                          # [T,16,128]
        value = value_flat.reshape(T, nv, hv)                   # [T,32,128]

        # ---- L2norm q,k per (position,head) on NPU, scale q by 1/sqrt(hk) ----
        # stack all (T*nk) rows -> one [T*nk, hk] L2norm call
        qn = self._l2norm(query.reshape(T * nk, hk).contiguous(), "l2norm(q)").reshape(T, nk, hk)
        kn = self._l2norm(key.reshape(T * nk, hk).contiguous(), "l2norm(k)").reshape(T, nk, hk)
        scale = 1.0 / (hk ** 0.5)
        qf = qn * scale
        kf = kn

        # ---- beta = sigmoid(b) on NPU ; g = -exp(A_log)*softplus(a+dt_bias) host ----
        beta = self._sigmoid(b.contiguous())                    # [T,32]
        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)   # [T,32] (softplus host)

        # ---- repeat_interleave q,k to nv heads (HF L759-761) ----
        qf = qf.repeat_interleave(rep, dim=1)                   # [T,32,128]
        kf = kf.repeat_interleave(rep, dim=1)                   # [T,32,128]

        # ---- HEAD-BATCHED chunk scan: ALL nv heads in ONE NPU dispatch/chunk ----
        # stack per-head q/k/v/g/beta into [H,...] and run dn_chunk_full_mh once per
        # chunk (was: 32-head python loop calling dn_chunk_full per head).
        if state is None:
            state = torch.zeros(nv, hk, hv)
        qf_h = qf.transpose(0, 1).contiguous()                  # [nv,T,hk]
        kf_h = kf.transpose(0, 1).contiguous()                  # [nv,T,hk]
        vf_h = value.transpose(0, 1).contiguous()               # [nv,T,hv]
        gf_h = g.transpose(0, 1).contiguous()                   # [nv,T]
        bf_h = beta.transpose(0, 1).contiguous()                # [nv,T]
        core_h, new_state = self._chunk_scan(qf_h, kf_h, vf_h, gf_h, bf_h, state)
        core = core_h.transpose(0, 1).contiguous()              # [T,nv,hv]

        # ---- gated RMSNorm(core, z) per (position,head) over head_v_dim on NPU ----
        core_2d = core.reshape(T * nv, hv)                      # [T*32,128]
        z_2d    = z.reshape(T * nv, hv)
        core_n = self._gnorm(core_2d.contiguous(), z_2d.contiguous(), norm_w)
        core_n = core_n.reshape(T, nv * hv)                     # [T, value_dim]

        # ---- out_proj on NPU ----
        out = self._linear(core_n.contiguous(), W_out, "out_proj")   # [T,H]

        out = out.unsqueeze(0)                                  # [1,T,H]
        if squeeze_back:
            out = out[0]
        if return_conv:
            return out, new_state, new_conv_state
        return out, new_state

    # ---------------- diagnostics ----------------
    @staticmethod
    def total_dfg_calls():
        return _CALLS["n"]

    def all_on_npu(self):
        return all(d == 0 for _, d in self.npu_stages)


def load_layer0_weights(W, dtype=torch.float32, layer=0):
    """Load real DeltaNet weights for model.layers.<layer>.linear_attn.* via QCNWeights."""
    p = f"model.layers.{layer}.linear_attn."
    return {
        "in_proj_qkvz":  W.get(p + "in_proj_qkvz.weight", dtype),
        "in_proj_ba":    W.get(p + "in_proj_ba.weight", dtype),
        "conv1d_weight": W.get(p + "conv1d.weight", dtype),
        "A_log":         W.get(p + "A_log", dtype),
        "dt_bias":       W.get(p + "dt_bias", dtype),
        "norm_weight":   W.get(p + "norm.weight", dtype),
        "out_proj":      W.get(p + "out_proj.weight", dtype),
    }
