#!/usr/bin/env python3
"""Dense용 register-aware 토큰 병합: 상위 tome_reg.py와 동일한 size-가중 병합에
'원본 patch → 최종 살아남은 토큰' 추적을 추가해, 병합 후에도 patch 격자로 되돌려(unmerge)
dense feature map을 복원한다. 분할(segmentation) 평가에 필요.

핵심: merge_step_track()이 (x_out, size_out, newpos)를 반환 — newpos[b,i]=old token i가 간 새 위치.
reduced_forward_dense()가 블록마다 newpos를 합성해 orig2cur(원본 patch→현재 토큰)를 유지하고,
마지막에 각 patch를 자기 대표 토큰의 feature로 채운 [B, Npatch, C] dense map을 만든다.
보호 전략(tome/ours/random/energy/highnorm)은 입력에서 보호토큰을 앞으로 재배열해 처리."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def merge_step_track(x, size, r, n_protect):
    """상위 merge_step과 수치적으로 동일 + old→new 위치맵 newpos[B,T] 반환."""
    B, T, C = x.shape
    dev = x.device
    r = min(r, max((T - n_protect) // 2, 0))
    if r <= 0:
        return x, size, torch.arange(T, device=dev).unsqueeze(0).expand(B, T).contiguous()
    xp, xr = x[:, :n_protect], x[:, n_protect:]
    sp, sr = size[:, :n_protect], size[:, n_protect:]
    a, b = xr[:, ::2, :], xr[:, 1::2, :]
    sa, sb = sr[:, ::2, :], sr[:, 1::2, :]
    La, Lb = a.shape[1], b.shape[1]
    an = F.normalize(a, dim=-1); bn = F.normalize(b, dim=-1)
    scores = an @ bn.transpose(-1, -2)
    node_max, node_idx = scores.max(dim=-1)                 # [B,La]
    edge = node_max.argsort(dim=-1, descending=True)[..., None]   # [B,La,1]
    unm_idx, src_idx = edge[:, r:, :], edge[:, :r, :]       # [B,Lu,1],[B,r,1]
    dst_idx = node_idx[..., None].gather(1, src_idx)        # [B,r,1]
    unm = a.gather(1, unm_idx.expand(-1, -1, C)); unm_s = sa.gather(1, unm_idx.expand(-1, -1, 1))
    src = a.gather(1, src_idx.expand(-1, -1, C));  src_s = sa.gather(1, src_idx.expand(-1, -1, 1))
    b_acc = (b * sb).scatter_add(1, dst_idx.expand(-1, -1, C), src * src_s)
    s_acc = sb.scatter_add(1, dst_idx.expand(-1, -1, 1), src_s)
    b_merged = b_acc / s_acc
    x_out = torch.cat([xp, unm, b_merged], dim=1)
    s_out = torch.cat([sp, unm_s, s_acc], dim=1)
    Lu = La - r
    # --- old->new 위치맵 ---
    newpos = torch.empty(B, T, dtype=torch.long, device=dev)
    newpos[:, :n_protect] = torch.arange(n_protect, device=dev).unsqueeze(0)
    bi = torch.arange(Lb, device=dev)
    newpos[:, n_protect + 2 * bi + 1] = (n_protect + Lu + bi).unsqueeze(0)      # b 토큰
    urank = (n_protect + torch.arange(Lu, device=dev)).unsqueeze(0).expand(B, -1)  # a 미병합
    newpos.scatter_(1, n_protect + 2 * unm_idx[..., 0], urank)
    new_src = n_protect + Lu + dst_idx[..., 0]                                  # a 병합→b
    newpos.scatter_(1, n_protect + 2 * src_idx[..., 0], new_src)
    return x_out, s_out, newpos


def _chosen_extra(t, strat, nprefix, nreg, gen):
    """각 이미지가 CLS 외 추가로 보호할 patch 인덱스(B개 리스트). tome=없음, ours=register."""
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
def reduced_forward_dense(model, x, r_pb, strat, gen=None):
    """블록마다 병합하며 forward. 반환:
       dense[B,Npatch,C] = 각 원본 patch를 자기 대표 토큰 feature로 채운 격자,
       cls[B,C], final_T(최종 토큰수).  strat별 보호토큰은 입력에서 앞으로 재배열."""
    nprefix = getattr(model, "num_prefix_tokens", 1); nreg = max(nprefix - 1, 4)
    t = model._pos_embed(model.patch_embed(x))              # [B,T,C]
    B, T, C = t.shape
    Npatch = model.patch_embed.num_patches
    extra = _chosen_extra(t, strat, nprefix, nreg, gen)
    perms = []
    for bidx in range(B):
        prot = [0] + sorted(extra[bidx])
        rest = [i for i in range(T) if i not in set(prot)]
        perms.append(prot + rest)
    perm = torch.tensor(perms, device=t.device)             # [B,T] new->old
    n_protect = 1 + (nreg if strat == "ours" else (0 if strat == "tome" else nreg))
    t = t.gather(1, perm[..., None].expand(-1, -1, C))
    inv = perm.argsort(dim=1)                               # old->new (재배열 직후)
    # 원본 patch p (old pos nprefix+p) 의 현재 위치
    orig2cur = inv.gather(1, (nprefix + torch.arange(Npatch, device=t.device)).unsqueeze(0).expand(B, -1))
    size = torch.ones(B, T, 1, device=t.device, dtype=t.dtype)
    for blk in model.blocks:
        t = blk(t)
        t, size, newpos = merge_step_track(t, size, r_pb, n_protect)
        orig2cur = newpos.gather(1, orig2cur)               # 합성
    t = model.norm(t)
    dense = t.gather(1, orig2cur[..., None].expand(-1, -1, C))   # [B,Npatch,C]
    return dense, t[:, 0], t.shape[1]
