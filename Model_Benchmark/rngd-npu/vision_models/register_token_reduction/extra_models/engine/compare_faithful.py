#!/usr/bin/env python3
"""[faithful] compare.py 를 정식(faithful) harness 로 실행하는 래퍼.
compare.py 의 데이터로딩·train갤러리·캐싱·kNN(val-LOO/train)·throughput 기계는 그대로 재사용하고,
병합 forward(reduced_forward)만 통제(post-block+feature) → 정식(in-block+key-metric+prop-attn)으로 교체.
PiToMe = 공식 pitome_bsm/pitome 이식(faithful_pitome_h2h 와 동일 로직). tome/ours = key-metric bipartite.
캐시는 feat_cache_faithful 로 분리(통제 캐시와 안 섞이게).

사용(canonical faithful 예):
  python compare_faithful.py --mode acc --gallery train --model <MODEL> --r_list 8 12 16 18 20 --gallery_cache 1
  python compare_faithful.py --mode acc --gallery val   --model <MODEL> --r_list 8 12 16 18 20     # val-LOO faithful
"""
import os, sys, math
import torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare   # 통제 엔진(데이터/캐시/kNN/train갤러리) 재사용


# ─────────── 정식 병합 (faithful_pitome_h2h.py 와 동일 로직) ───────────
@torch.no_grad()
def _tome_merge(x, size, metric, r, n_protect):
    B, T, C = x.shape
    r = min(r, max((T - n_protect) // 2, 0))
    if r <= 0:
        return x, size
    mr = metric[:, n_protect:]
    am, bm = mr[:, ::2], mr[:, 1::2]
    an = F.normalize(am, dim=-1); bn = F.normalize(bm, dim=-1)
    scores = an @ bn.transpose(-1, -2)
    node_max, node_idx = scores.max(-1)
    edge = node_max.argsort(-1, descending=True)[..., None]
    unm_idx, src_idx = edge[:, r:, :], edge[:, :r, :]
    dst_idx = node_idx[..., None].gather(1, src_idx)
    xp, xr = x[:, :n_protect], x[:, n_protect:]; sp, sr = size[:, :n_protect], size[:, n_protect:]
    xa, xb = xr[:, ::2], xr[:, 1::2]; sa, sb = sr[:, ::2], sr[:, 1::2]
    unm = xa.gather(1, unm_idx.expand(-1, -1, C)); unm_s = sa.gather(1, unm_idx.expand(-1, -1, 1))
    src = xa.gather(1, src_idx.expand(-1, -1, C)); src_s = sa.gather(1, src_idx.expand(-1, -1, 1))
    b_acc = (xb * sb).scatter_add(1, dst_idx.expand(-1, -1, C), src * src_s)
    s_acc = sb.scatter_add(1, dst_idx.expand(-1, -1, 1), src_s)
    return torch.cat([xp, unm, b_acc / s_acc], 1), torch.cat([sp, unm_s, s_acc], 1)


def _make_official_pitome_merge(metric_r, r, margin, use_bsm):
    B, P, C = metric_r.shape
    bat = torch.arange(B, device=metric_r.device).unsqueeze(1)
    mm = F.normalize(metric_r, p=2, dim=-1)
    sim = mm @ mm.transpose(-1, -2)
    energy = F.elu(sim - margin, alpha=1.0).mean(dim=-1)
    indices = energy.argsort(dim=-1, descending=True)
    if use_bsm:
        a_idx, b_idx = indices[..., ::2], indices[..., 1::2]
        La, Lb = a_idx.shape[-1], b_idx.shape[-1]
        s = sim.gather(-1, b_idx.unsqueeze(-2).expand(B, P, Lb))
        s = s.gather(-2, a_idx.unsqueeze(-1).expand(B, La, Lb))
        node_max, node_idx = s.max(dim=-1)
        edge = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx, src_idx = edge[..., r:, :], edge[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        def merge(y, mode):
            src, dst = y[bat, a_idx], y[bat, b_idx]
            n, t1, c = src.shape
            unm = src.gather(-2, unm_idx.expand(n, t1 - r, c))
            src = src.gather(-2, src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
            return torch.cat([unm, dst], dim=1)
        return merge
    else:
        merge_idx, prot_idx = indices[..., :2 * r], indices[..., 2 * r:]
        a_idx, b_idx = merge_idx[..., ::2], merge_idx[..., 1::2]
        s = sim.gather(-1, b_idx.unsqueeze(-2).expand(B, P, r))
        s = s.gather(-2, a_idx.unsqueeze(-1).expand(B, r, r))
        _, dst_idx = s.max(dim=-1)

        def merge(y, mode):
            protected = y[bat, prot_idx]
            src, dst = y[bat, a_idx], y[bat, b_idx]
            dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, r, y.shape[-1]), src, reduce=mode)
            return torch.cat([protected, dst], dim=1)
        return merge


@torch.no_grad()
def _pitome_merge(x, size, metric, r, margin, use_bsm):
    xc, xr = x[:, :1], x[:, 1:]; sc, sr = size[:, :1], size[:, 1:]; mr = metric[:, 1:]
    P = xr.shape[1]
    r = min(r, P // 2)
    if r <= 0:
        return x, size
    merge = _make_official_pitome_merge(mr, r, margin, use_bsm)
    xo = merge(xr * sr, mode="sum"); so = merge(sr, mode="sum")
    return torch.cat([xc, xo / so], 1), torch.cat([sc, so], 1)


@torch.no_grad()
def reduced_forward_faithful(m, x, r, strat, nprefix):
    """compare.reduced_forward 와 동일 시그니처. 반환=CLS feature. 정식 harness."""
    n_protect = nprefix if strat == "ours" else 1
    t = m._pos_embed(m.patch_embed(x)); B = t.shape[0]
    size = torch.ones(B, t.shape[1], 1, dtype=t.dtype, device=t.device)
    H = m.blocks[0].attn.num_heads; L = len(m.blocks); n_bsm = math.ceil(L * 0.5)
    for li, blk in enumerate(m.blocks):
        xn = blk.norm1(t); B, Nt, C = xn.shape; hd = C // H
        qkv = blk.attn.qkv(xn).reshape(B, Nt, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        bias = size.log().reshape(B, 1, 1, Nt)
        xa = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        xa = blk.attn.proj(xa.transpose(1, 2).reshape(B, Nt, C))
        metric = k.mean(1)
        t = t + blk.drop_path1(blk.ls1(xa))
        margin = 0.75 - 0.75 * (li / max(L, 1))
        if strat == "pitome":
            t, size = _pitome_merge(t, size, metric, r, margin, use_bsm=(li < n_bsm))
        else:
            t, size = _tome_merge(t, size, metric, r, n_protect)
        t = t + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(t))))
    return m.norm(t)[:, 0]


# ─────────── 통제 엔진에 정식 forward 주입 + 캐시 분리 ───────────
compare.reduced_forward = reduced_forward_faithful
if "--cache_dir" not in sys.argv:
    sys.argv += ["--cache_dir", os.path.join(HERE, "feat_cache_faithful")]

if __name__ == "__main__":
    print("[faithful] compare.py 엔진에 정식 forward 주입(in-block+key-metric+prop-attn; PiToMe=공식 pitome_bsm/pitome). "
          "캐시=feat_cache_faithful.", flush=True)
    compare.main()
