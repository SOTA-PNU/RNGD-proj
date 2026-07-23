#!/usr/bin/env python3
"""Register-aware token merging (size-가중 bipartite soft matching) — GPU/CPU 공용.
같은 병합에서 n_protect만 바꿔 ablation:
  - ToMe baseline : n_protect=1 (CLS만 보호)
  - ours          : n_protect=1+num_register (CLS+register 보호)
register는 timm DINOv2-reg에서 prefix(cls 다음 4개)라 prefix 보호로 깔끔히 처리(배치 가능).
size-가중: 합쳐진 토큰이 대표하는 원본 토큰 수로 가중평균(ToMe의 핵심, Bolya ICLR'23 arXiv:2210.09461).
사용은 eval_imagenet.py 참고."""
import torch, torch.nn.functional as F


@torch.no_grad()
def merge_step(x, size, r, n_protect):
    """x:[B,T,C], size:[B,T,1]. 앞 n_protect개 보호. 비보호 중 r개 병합(size-가중). 반환 (x',size')."""
    B, T, C = x.shape
    r = min(r, max((T - n_protect) // 2, 0))
    if r <= 0:
        return x, size
    xp, xr = x[:, :n_protect], x[:, n_protect:]
    sp, sr = size[:, :n_protect], size[:, n_protect:]
    a, b = xr[:, ::2, :], xr[:, 1::2, :]            # bipartite split
    sa, sb = sr[:, ::2, :], sr[:, 1::2, :]
    an = F.normalize(a, dim=-1); bn = F.normalize(b, dim=-1)
    scores = an @ bn.transpose(-1, -2)              # [B,|a|,|b|]
    node_max, node_idx = scores.max(dim=-1)          # 각 a의 최적 b
    edge = node_max.argsort(dim=-1, descending=True)[..., None]   # [B,|a|,1]
    unm_idx, src_idx = edge[:, r:, :], edge[:, :r, :]
    dst_idx = node_idx[..., None].gather(1, src_idx)              # [B,r,1]
    unm = a.gather(1, unm_idx.expand(-1, -1, C)); unm_s = sa.gather(1, unm_idx.expand(-1, -1, 1))
    src = a.gather(1, src_idx.expand(-1, -1, C));  src_s = sa.gather(1, src_idx.expand(-1, -1, 1))
    # b에 size-가중 누적
    b_acc = (b * sb).scatter_add(1, dst_idx.expand(-1, -1, C), src * src_s)
    s_acc = sb.scatter_add(1, dst_idx.expand(-1, -1, 1), src_s)
    b_merged = b_acc / s_acc                          # 가중평균
    x_out = torch.cat([xp, unm, b_merged], dim=1)
    s_out = torch.cat([sp, unm_s, s_acc], dim=1)
    return x_out, s_out


@torch.no_grad()
def reduced_forward(model, x, r_per_block, n_protect):
    """timm ViT를 블록마다 r_per_block개 병합하며 forward. CLS 임베딩 반환. x:[B,3,H,W]."""
    t = model._pos_embed(model.patch_embed(x))        # [B,T,C] (cls+reg+pos)
    B, T, C = t.shape
    size = torch.ones(B, T, 1, device=t.device, dtype=t.dtype)
    for blk in model.blocks:
        t = blk(t)
        t, size = merge_step(t, size, r_per_block, n_protect)
    t = model.norm(t)
    return t[:, 0]                                    # CLS
