#!/usr/bin/env python3
"""[50k] 정식 ToMe(proportional attention + key-metric + attn↔MLP 병합) vs Ours(register 보호).
감사 지적 '베이스라인이 정식 ToMe 아님(proportional attention 미적용)'에 대응. 두 팔 동일 정식 메커니즘, 보호만 다름.
공식 ToMe(Bolya ICLR'23): proportional attn = 스케일된 attn logit에 log(size) 더함(SDPA attn_mask),
유사도 = attention key(head 평균), 병합 = attn 뒤·MLP 앞 size-가중.  kNN.
사용(전체): python faithful_tome_h2h.py 50000"""
import sys, torch, torch.nn.functional as F
from tome_core import knn, load_model_and_data

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"


@torch.no_grad()
def merge_metric(x, size, metric, r, n_protect):
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


@torch.no_grad()
def forward_faithful(m, x, r, n_protect, proportional=True):
    t = m._pos_embed(m.patch_embed(x)); B = t.shape[0]
    size = torch.ones(B, t.shape[1], 1, dtype=t.dtype, device=t.device)
    H = m.blocks[0].attn.num_heads
    for blk in m.blocks:
        xn = blk.norm1(t); B, Nt, C = xn.shape; hd = C // H
        qkv = blk.attn.qkv(xn).reshape(B, Nt, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        bias = size.log().reshape(B, 1, 1, Nt) if proportional else None
        xa = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        xa = blk.attn.proj(xa.transpose(1, 2).reshape(B, Nt, C))
        metric = k.mean(1)
        t = t + blk.drop_path1(blk.ls1(xa))
        t, size = merge_metric(t, size, metric, r, n_protect)
        t = t + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(t))))
    return m.norm(t)[:, 0], t.shape[1]


def main():
    m, X, Y, dev = load_model_and_data(MODEL, N)
    nprefix = m.num_prefix_tokens
    print(f"{MODEL} · {len(X)}장 · dev={dev} · 정식 ToMe(proportional attn + key-metric + attn↔MLP 병합)", flush=True)
    print(f"{'r':>3} {'comp%':>6} {'ToMe(정식)':>11} {'Ours(정식+reg)':>15}  판정", flush=True)
    for r in [8, 12, 16, 18, 20]:
        res = {}; ft = None
        for name, npr in [("tome", 1), ("ours", nprefix)]:
            fs = [forward_faithful(m, X[i:i+128].to(dev), r, npr)[0].float().cpu() for i in range(0, len(X), 128)]
            res[name] = knn(torch.cat(fs), Y)
        _, ft = forward_faithful(m, X[:128].to(dev), r, nprefix)
        comp = 100 * (1 - ft / (m.patch_embed.num_patches + nprefix))
        d = res["ours"] - res["tome"]
        v = "ours>ToMe✅" if d > 0.5 else ("≈" if abs(d) <= 0.5 else "ToMe우위❌")
        print(f"{r:>3} {comp:>6.1f} {res['tome']:>11.2f} {res['ours']:>15.2f}  Δ={d:+.2f} {v}", flush=True)
    print("\n해석: 정식 ToMe(proportional attn 포함)서도 register 보호가 이기면 = '약한 베이스라인' 아님 확정.", flush=True)


if __name__ == "__main__":
    main()
