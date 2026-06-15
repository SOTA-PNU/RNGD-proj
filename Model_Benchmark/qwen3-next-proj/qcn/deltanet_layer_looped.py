#!/usr/bin/env python3
# REFERENCE COPY: the ORIGINAL per-head-LOOPED DeltaNetLayer (calls dn_chunk_full
# once PER value-head -> 32 NPU dispatches/chunk).  Kept verbatim so the rewritten
# head-batched deltanet_layer.py can be validated to MATCH it (same math) and so the
# NPU-dispatch reduction (looped vs batched) can be counted side-by-side.
# This is identical to the pre-rewrite deltanet_layer.py except the class is renamed
# DeltaNetLayerLooped and it shares the SAME _CALLS / _dfg spy as deltanet_layer
# (imported below) so dispatch counts are comparable across both.
import os
import torch
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule
import furiosa.torch.custom_ops.dfg as dfgmod
import torch.nn.functional as F

# share the spy + _CALLS counter + compiled-kernel cache with deltanet_layer
from qcn.deltanet_layer import _npu, _CALLS, _build_T, _BASE, _PADM

# QCN_DPE=1 routes this layer's matmuls onto the FAST systolic/DPE-MAC engine:
#   * the proj / out_proj linears -> dn_linear_dpe.yaml  (EinsumByDpe, ~2-3.8x VE)
#   * the chunk scan              -> dn_chunk_full_dpe2.yaml  (2 chunk matmuls on
#     the DPE; the 5-DPE dn_chunk_full_dpe.yaml MISCOMPILES -- a per-graph DPE-fuse
#     cap of 2 corrupts 3+, see tk_kernels/dpe_incremental_log.md, so dpe2 is the
#     largest VALIDATING variant: maxerr_vs_hf 2.53e-04 @ atol 1e-2).
# Default (QCN_DPE unset/0) keeps the f32-exact EinsumByVe kernels (~1e-7 vs HF).
DPE        = os.environ.get("QCN_DPE", "0") == "1"
LINEAR_YAML = "dn_linear_dpe.yaml" if DPE else "dn_linear.yaml"
CHUNK_YAML  = "dn_chunk_full_dpe2.yaml" if DPE else "dn_chunk_full.yaml"


class DeltaNetLayerLooped:
    """ORIGINAL per-head-looped Gated DeltaNet layer (dn_chunk_full per head)."""

    def __init__(self, config, dev=None, chunk_size=64):
        self.hidden_size = config["hidden_size"]
        self.num_k_heads = config["linear_num_key_heads"]
        self.num_v_heads = config["linear_num_value_heads"]
        self.head_k_dim  = config["linear_key_head_dim"]
        self.head_v_dim  = config["linear_value_head_dim"]
        self.conv_kernel = config["linear_conv_kernel_dim"]
        self.eps         = config.get("rms_norm_eps", 1e-6)
        self.key_dim   = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim  = self.key_dim * 2 + self.value_dim
        self.n_rep     = self.num_v_heads // self.num_k_heads
        self.dev = dev or os.environ.get("RNGD_DEV", "rngd:0")
        self.chunk_size = chunk_size
        self.npu_stages = []

    _OTILE = 512
    _ITILE = 2048

    def _linear(self, x_ti, W_oi, name):
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
                out, d = _npu(LINEAR_YAML, [xb, Wb], self.dev)
                dtot += d
                acc = out[0] if acc is None else acc + out[0]
            cols.append(acc[:keep])
        self.npu_stages.append((name, dtot))
        return torch.cat(cols, dim=-1)

    def _conv1d_silu(self, x_ct, w_ck, conv_prefix=None):
        C, T = x_ct.shape
        Kc = w_ck.shape[1]
        if conv_prefix is None:
            conv_prefix = torch.zeros(C, Kc - 1, dtype=x_ct.dtype)
        x_pad = torch.cat([conv_prefix, x_ct], dim=-1)
        xs = [x_pad[:, j:j + T].contiguous() for j in range(Kc)]
        wf = [w_ck[:, j:j + 1].expand(C, T).contiguous() for j in range(Kc)]
        out, d = _npu("dn_conv1d.yaml", xs + wf, self.dev)
        self.npu_stages.append(("conv1d+SiLU", d))
        new_prefix = x_pad[:, -(Kc - 1):].clone() if Kc > 1 else conv_prefix
        return out[0], new_prefix

    def _l2norm(self, x_md, name):
        M, D = x_md.shape
        Mp = max(M, _PADM)
        xp = torch.zeros(Mp, D); xp[:M] = x_md
        inputs = [xp.t().contiguous(), torch.ones(D), torch.full((Mp,), self.eps), xp.contiguous()]
        out, d = _npu("dn_l2norm.yaml", inputs, self.dev)
        self.npu_stages.append((name, d))
        return out[0][:M]

    def _sigmoid(self, x_mn):
        out, d = _npu("dn_gate.yaml", [x_mn.contiguous(), torch.ones_like(x_mn)], self.dev)
        self.npu_stages.append(("sigmoid(beta)", d))
        return out[0]

    def _gnorm(self, x_md, gate_md, weight_d):
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

    def _chunk_scan_head(self, qf, kf, vf, gf, bf, S, hidx):
        N, K = qf.shape
        V = vf.shape[1]
        C = self.chunk_size
        pad = (C - N % C) % C
        if pad:
            qf = F.pad(qf, (0, 0, 0, pad)); kf = F.pad(kf, (0, 0, 0, pad))
            vf = F.pad(vf, (0, 0, 0, pad)); gf = F.pad(gf, (0, pad)); bf = F.pad(bf, (0, pad))
        Ntot = qf.shape[0]
        NC = Ntot // C
        mask_incl   = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=0)
        mask_strict = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1)
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
            T = _build_T(attn0, C)
            value = T @ v_beta
            kcd   = T @ (k_beta * g_cum.exp().unsqueeze(-1))
            decay_strict = decay.masked_fill(mask_strict, 0.0)
            gexp_k = g_cum.exp().unsqueeze(1).expand(C, K).contiguous()
            wdecay = (g_cum[-1] - g_cum).exp().unsqueeze(1).expand(C, K).contiguous()
            sdecay = torch.full((K, V), g_cum[-1].exp().item())
            kin = [q_i.contiguous(), k_i.contiguous(), value.contiguous(),
                   decay_strict.contiguous(), kcd.contiguous(), gexp_k,
                   wdecay, sdecay, S.contiguous()]
            res, d = _npu(CHUNK_YAML, kin, self.dev)
            self.npu_stages.append((f"chunk-scan[h{hidx}c{ci}]", d))
            outs.append(res[0])
            S = res[1]
        core = torch.cat(outs, dim=0)[:N]
        return core, S

    def forward(self, hidden_states, weights, state=None, conv_state=None,
                return_conv=False):
        squeeze_back = (hidden_states.dim() == 2)
        if squeeze_back:
            hidden_states = hidden_states.unsqueeze(0)
        B, T, H = hidden_states.shape
        assert B == 1
        assert H == self.hidden_size
        h = hidden_states[0].float()

        W_qkvz   = weights["in_proj_qkvz"]
        W_ba     = weights["in_proj_ba"]
        conv_w   = weights["conv1d_weight"]
        A_log    = weights["A_log"]
        dt_bias  = weights["dt_bias"]
        norm_w   = weights["norm_weight"]
        W_out    = weights["out_proj"]
        if conv_w.dim() == 3:
            conv_w = conv_w.squeeze(1)

        nk, nv = self.num_k_heads, self.num_v_heads
        hk, hv = self.head_k_dim, self.head_v_dim
        rep = self.n_rep

        proj_qkvz = self._linear(h, W_qkvz, "in_proj_qkvz")
        proj_ba   = self._linear(h, W_ba,   "in_proj_ba")

        qkvz = proj_qkvz.view(T, nk, 2 * hk + 2 * hv * rep)
        split_qkvz = [hk, hk, rep * hv, rep * hv]
        query, key, value, z = torch.split(qkvz, split_qkvz, dim=-1)
        value = value.reshape(T, nv, hv)
        z     = z.reshape(T, nv, hv)
        ba = proj_ba.view(T, nk, 2 * rep)
        b, a = torch.split(ba, [rep, rep], dim=-1)
        b = b.reshape(T, nv)
        a = a.reshape(T, nv)
        query = query.reshape(T, nk * hk)
        key   = key.reshape(T, nk * hk)
        value_flat = value.reshape(T, nv * hv)

        mixed = torch.cat((query, key, value_flat), dim=-1).transpose(0, 1)
        conv_out, new_conv_state = self._conv1d_silu(
            mixed.contiguous(), conv_w, conv_prefix=conv_state)
        conv_out = conv_out.transpose(0, 1)
        query, key, value_flat = torch.split(
            conv_out, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        query = query.reshape(T, nk, hk)
        key   = key.reshape(T, nk, hk)
        value = value_flat.reshape(T, nv, hv)

        qn = self._l2norm(query.reshape(T * nk, hk).contiguous(), "l2norm(q)").reshape(T, nk, hk)
        kn = self._l2norm(key.reshape(T * nk, hk).contiguous(), "l2norm(k)").reshape(T, nk, hk)
        scale = 1.0 / (hk ** 0.5)
        qf = qn * scale
        kf = kn

        beta = self._sigmoid(b.contiguous())
        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)

        qf = qf.repeat_interleave(rep, dim=1)
        kf = kf.repeat_interleave(rep, dim=1)

        if state is None:
            state = torch.zeros(nv, hk, hv)
        core = torch.empty(T, nv, hv)
        new_state = torch.empty(nv, hk, hv)
        for hh in range(nv):
            core_h, S_h = self._chunk_scan_head(
                qf[:, hh, :], kf[:, hh, :], value[:, hh, :],
                g[:, hh], beta[:, hh], state[hh], hh)
            core[:, hh, :] = core_h
            new_state[hh] = S_h

        core_2d = core.reshape(T * nv, hv)
        z_2d    = z.reshape(T * nv, hv)
        core_n = self._gnorm(core_2d.contiguous(), z_2d.contiguous(), norm_w)
        core_n = core_n.reshape(T, nv * hv)

        out = self._linear(core_n.contiguous(), W_out, "out_proj")

        out = out.unsqueeze(0)
        if squeeze_back:
            out = out[0]
        if return_conv:
            return out, new_state, new_conv_state
        return out, new_state
