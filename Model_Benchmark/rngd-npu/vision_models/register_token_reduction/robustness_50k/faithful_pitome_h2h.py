#!/usr/bin/env python3
"""[50k] 정식(faithful) 3자 비교: ToMe vs PiToMe vs Ours — 전부 선행연구 공식 메커니즘.
공통 forward = proportional attention(size.log bias) + key-metric(k.mean) + attn↔MLP 사이 병합.
  - tome  : CLS만 보호, size-가중 bipartite soft matching (key metric)
  - pitome: 공식 PiToMe 그대로 — 앞 ceil(L/2)층 pitome_bsm, 뒤 pitome(에너지경로); margin=0.75-0.75·ℓ/L,
            에너지 E=elu(cos(key)-margin).mean, merge_wavg(size-가중). CLS만 보호(register 개념 없음).
  - ours  : CLS+register 보호, size-가중 bipartite (key metric)
※ pitome 는 공식 algo/pitome/merge.py 의 pitome()·pitome_bsm()·pitome_vision()·merge_wavg() 를 소스대로 이식.
   층별 use_bsm_pitome = [True]*ceil(L/2)+[False]*나머지 (공식 forward 스케줄).
사용(전체): python faithful_pitome_h2h.py 50000
"""
import sys, math, torch, torch.nn.functional as F
from tome_core import knn, load_model_and_data

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"
RLIST = [8, 12, 16, 18, 20]


# ─────────────────────────── ToMe / Ours (key-metric bipartite) ───────────────────────────
@torch.no_grad()
def tome_merge(x, size, metric, r, n_protect, margin=None, use_bsm=None):
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


# ─────────────────────────── PiToMe (공식 소스 이식) ───────────────────────────
def _make_official_pitome_merge(metric_r, r, margin, use_bsm):
    """공식 algo/pitome/merge.py 이식: metric_r=CLS 제외 key metric [B,P,C]. merge(y,mode) 클로저 반환."""
    B, P, C = metric_r.shape
    bat = torch.arange(B, device=metric_r.device).unsqueeze(1)
    mm = F.normalize(metric_r, p=2, dim=-1)
    sim = mm @ mm.transpose(-1, -2)                                   # [B,P,P]
    energy = F.elu(sim - margin, alpha=1.0).mean(dim=-1)              # 공식 Eq.4
    indices = energy.argsort(dim=-1, descending=True)                # 고에너지 우선
    if use_bsm:                                                       # 공식 pitome_bsm
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
    else:                                                            # 공식 pitome (에너지경로)
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
def pitome_merge(x, size, metric, r, n_protect, margin, use_bsm):
    """공식 merge_wavg 로 size-가중 적용. n_protect=1(CLS만) 가정."""
    xc, xr = x[:, :1], x[:, 1:]; sc, sr = size[:, :1], size[:, 1:]; mr = metric[:, 1:]
    P = xr.shape[1]
    r = min(r, P // 2)
    if r <= 0:
        return x, size
    merge = _make_official_pitome_merge(mr, r, margin, use_bsm)
    xo = merge(xr * sr, mode="sum")                                  # merge_wavg
    so = merge(sr, mode="sum")
    xo = xo / so
    return torch.cat([xc, xo], 1), torch.cat([sc, so], 1)


@torch.no_grad()
def forward_faithful(m, x, r, strat, nprefix, proportional=True):
    """공통 정식 forward. strat 로 병합규칙·보호개수 분기. ours 만 register 까지 보호."""
    n_protect = nprefix if strat == "ours" else 1
    t = m._pos_embed(m.patch_embed(x)); B = t.shape[0]
    size = torch.ones(B, t.shape[1], 1, dtype=t.dtype, device=t.device)
    H = m.blocks[0].attn.num_heads; L = len(m.blocks)
    n_bsm = math.ceil(L * 0.5)                                       # 공식: 앞 절반 pitome_bsm
    for li, blk in enumerate(m.blocks):
        xn = blk.norm1(t); B, Nt, C = xn.shape; hd = C // H
        qkv = blk.attn.qkv(xn).reshape(B, Nt, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        bias = size.log().reshape(B, 1, 1, Nt) if proportional else None
        xa = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        xa = blk.attn.proj(xa.transpose(1, 2).reshape(B, Nt, C))
        metric = k.mean(1)
        t = t + blk.drop_path1(blk.ls1(xa))
        margin = 0.75 - 0.75 * (li / max(L, 1))
        if strat == "pitome":
            t, size = pitome_merge(t, size, metric, r, n_protect, margin, use_bsm=(li < n_bsm))
        else:
            t, size = tome_merge(t, size, metric, r, n_protect)
        t = t + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(t))))
    return m.norm(t)[:, 0], t.shape[1]


def main():
    m, X, Y, dev = load_model_and_data(MODEL, N)
    nprefix = m.num_prefix_tokens
    print(f"{MODEL} · {len(X)}장 · dev={dev} · 정식(faithful) 3자: ToMe/PiToMe/Ours "
          f"(prop-attn+key-metric+attn↔MLP병합; PiToMe=공식 pitome_bsm/pitome 이식)", flush=True)
    print(f"{'r':>3} {'comp%':>6} {'ToMe':>7} {'PiToMe':>7} {'Ours':>7} {'O-PiToMe':>9} {'O-ToMe':>7}", flush=True)
    for r in RLIST:
        res = {}
        for s in ["tome", "pitome", "ours"]:
            fs = [forward_faithful(m, X[i:i+128].to(dev), r, s, nprefix)[0].float().cpu() for i in range(0, len(X), 128)]
            res[s] = knn(torch.cat(fs), Y)
        _, ft = forward_faithful(m, X[:128].to(dev), r, "ours", nprefix)
        comp = 100 * (1 - ft / (m.patch_embed.num_patches + nprefix))
        dop, dot = res["ours"] - res["pitome"], res["ours"] - res["tome"]
        print(f"{r:>3} {comp:>6.1f} {res['tome']:>7.2f} {res['pitome']:>7.2f} {res['ours']:>7.2f} {dop:>+9.2f} {dot:>+7.2f}", flush=True)
    print("\n해석: 정식 harness서 극단압축 Ours>PiToMe(O-PiToMe>0)이면 tab:pitome 을 통제→정식 승급.", flush=True)
    print("      tome/ours 는 faithful_tome_h2h.py 와 일치해야 함(일관성 점검).", flush=True)


if __name__ == "__main__":
    main()
