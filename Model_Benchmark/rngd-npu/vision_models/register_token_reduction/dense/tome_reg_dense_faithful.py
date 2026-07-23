#!/usr/bin/env python3
"""[faithful] dense용 정식(faithful) 병합 + un-merge 추적.
= tome_reg_dense.py 와 동일한 '원본 patch→최종 토큰' 추적(newpos 합성)에, 병합 forward만 통제→정식 교체.
정식 harness = proportional attention(size.log bias) + key-metric(k.mean) + attn↔MLP 사이 병합.
핵심: merge_metric_track()이 (x_out, size_out, newpos) 반환 — 선택은 key-metric, 추적 인덱스 로직은 동일."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def merge_metric_track(x, size, metric, r, n_protect):
    """정식 ToMe(key-metric bipartite) size-가중 병합 + old→new 위치맵 newpos[B,T].
    선택만 key-metric(metric)으로, 병합·추적은 tome_reg_dense.merge_step_track 과 동일 인덱스 로직."""
    B, T, C = x.shape
    dev = x.device
    r = min(r, max((T - n_protect) // 2, 0))
    if r <= 0:
        return x, size, torch.arange(T, device=dev).unsqueeze(0).expand(B, T).contiguous()
    mr = metric[:, n_protect:]
    am, bm = mr[:, ::2], mr[:, 1::2]
    an = F.normalize(am, dim=-1); bn = F.normalize(bm, dim=-1)
    scores = an @ bn.transpose(-1, -2)
    node_max, node_idx = scores.max(dim=-1)
    edge = node_max.argsort(dim=-1, descending=True)[..., None]
    unm_idx, src_idx = edge[:, r:, :], edge[:, :r, :]
    dst_idx = node_idx[..., None].gather(1, src_idx)
    xp, xr = x[:, :n_protect], x[:, n_protect:]
    sp, sr = size[:, :n_protect], size[:, n_protect:]
    a, b = xr[:, ::2, :], xr[:, 1::2, :]
    sa, sb = sr[:, ::2, :], sr[:, 1::2, :]
    La, Lb = a.shape[1], b.shape[1]
    unm = a.gather(1, unm_idx.expand(-1, -1, C)); unm_s = sa.gather(1, unm_idx.expand(-1, -1, 1))
    src = a.gather(1, src_idx.expand(-1, -1, C));  src_s = sa.gather(1, src_idx.expand(-1, -1, 1))
    b_acc = (b * sb).scatter_add(1, dst_idx.expand(-1, -1, C), src * src_s)
    s_acc = sb.scatter_add(1, dst_idx.expand(-1, -1, 1), src_s)
    x_out = torch.cat([xp, unm, b_acc / s_acc], dim=1)
    s_out = torch.cat([sp, unm_s, s_acc], dim=1)
    Lu = La - r
    newpos = torch.empty(B, T, dtype=torch.long, device=dev)
    newpos[:, :n_protect] = torch.arange(n_protect, device=dev).unsqueeze(0)
    bi = torch.arange(Lb, device=dev)
    newpos[:, n_protect + 2 * bi + 1] = (n_protect + Lu + bi).unsqueeze(0)
    urank = (n_protect + torch.arange(Lu, device=dev)).unsqueeze(0).expand(B, -1)
    newpos.scatter_(1, n_protect + 2 * unm_idx[..., 0], urank)
    new_src = n_protect + Lu + dst_idx[..., 0]
    newpos.scatter_(1, n_protect + 2 * src_idx[..., 0], new_src)
    return x_out, s_out, newpos


def _chosen_extra(t, strat, nprefix, nreg, gen):
    """tome_reg_dense.py 와 동일: 전략별 정적 보호 patch 인덱스."""
    B, T, C = t.shape
    if strat == "tome":
        return [[] for _ in range(B)]
    if strat == "ours":
        return [list(range(1, nprefix)) for _ in range(B)]
    p0 = nprefix; pt = t[:, p0:]; P = pt.shape[1]
    if strat == "random":
        return [(p0 + torch.randperm(P, generator=gen)[:nreg]).tolist() for _ in range(B)]
    if strat == "highnorm":
        return (pt.norm(dim=-1).topk(nreg, dim=1).indices + p0).tolist()
    if strat == "energy":
        pn = F.normalize(pt, dim=-1); sim = pn @ pn.transpose(-1, -2)
        energy = sim.clamp(min=0).mean(dim=-1)
        return (energy.topk(nreg, dim=1, largest=False).indices + p0).tolist()
    return [[] for _ in range(B)]


@torch.no_grad()
def reduced_forward_dense(model, x, r_pb, strat, gen=None, proportional=True):
    """정식 forward(prop-attn+key-metric+attn↔MLP병합)로 블록마다 병합·추적. tome_reg_dense.reduced_forward_dense 와
    인터페이스 동일: 반환 dense[B,Npatch,C], cls[B,C], final_T."""
    nprefix = getattr(model, "num_prefix_tokens", 1); nreg = max(nprefix - 1, 4)
    t = model._pos_embed(model.patch_embed(x))
    B, T, C = t.shape
    Npatch = model.patch_embed.num_patches
    extra = _chosen_extra(t, strat, nprefix, nreg, gen)
    perms = []
    for bidx in range(B):
        prot = [0] + sorted(extra[bidx]); rest = [i for i in range(T) if i not in set(prot)]
        perms.append(prot + rest)
    perm = torch.tensor(perms, device=t.device)
    n_protect = 1 + (nreg if strat == "ours" else (0 if strat == "tome" else nreg))
    t = t.gather(1, perm[..., None].expand(-1, -1, C))
    inv = perm.argsort(dim=1)
    orig2cur = inv.gather(1, (nprefix + torch.arange(Npatch, device=t.device)).unsqueeze(0).expand(B, -1))
    size = torch.ones(B, T, 1, device=t.device, dtype=t.dtype)
    H = model.blocks[0].attn.num_heads
    for blk in model.blocks:
        xn = blk.norm1(t); B, Nt, C = xn.shape; hd = C // H
        qkv = blk.attn.qkv(xn).reshape(B, Nt, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        bias = size.log().reshape(B, 1, 1, Nt) if proportional else None
        xa = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        xa = blk.attn.proj(xa.transpose(1, 2).reshape(B, Nt, C))
        metric = k.mean(1)
        t = t + blk.drop_path1(blk.ls1(xa))
        t, size, newpos = merge_metric_track(t, size, metric, r_pb, n_protect)
        orig2cur = newpos.gather(1, orig2cur)
        t = t + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(t))))
    t = model.norm(t)
    dense = t.gather(1, orig2cur[..., None].expand(-1, -1, C))
    return dense, t[:, 0], t.shape[1]
