#!/usr/bin/env python3
"""[50k] 실제 PiToMe(공식 repo algo/pitome/merge.py) vs Ours(register 보호) vs ToMe.
감사 지적 'energy는 프록시일 뿐, 실제 PiToMe와 head-to-head 필요'에 대응.
PiToMe: CLS 제외, energy=elu(sim-margin).mean, 고에너지(중복) 상위 2r 병합·저에너지 보호(재학습 없음).
Ours/ToMe: 같은 size-가중 병합, 보호대상만 CLS(=ToMe)↔CLS+register(=Ours).  kNN.
사용(전체): python pitome_h2h.py 50000 [margin]"""
import sys, torch, torch.nn.functional as F
from tome_core import forward_kprotect, knn, load_model_and_data

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
MARGIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"


@torch.no_grad()
def pitome_reduce(x, size, r, margin=MARGIN, alpha=1.0):
    B, T, C = x.shape
    cls, xr = x[:, :1], x[:, 1:]; cls_s, sr = size[:, :1], size[:, 1:]
    Tr = xr.shape[1]
    if r <= 0 or 2 * r > Tr:
        return x, size
    mm = F.normalize(xr, dim=-1); sim = mm @ mm.transpose(-1, -2)
    energy = F.elu(sim - margin, alpha=alpha).mean(-1)
    idx = energy.argsort(dim=-1, descending=True)
    merge_idx = idx[:, :2 * r]; prot_idx = idx[:, 2 * r:]
    a_idx, b_idx = merge_idx[:, ::2], merge_idx[:, 1::2]
    s = sim.gather(-1, b_idx.unsqueeze(1).expand(B, Tr, r)).gather(-2, a_idx.unsqueeze(-1).expand(B, r, r))
    dst = s.max(-1).indices
    bi = torch.arange(B).unsqueeze(1)
    prot, prot_s = xr[bi, prot_idx], sr[bi, prot_idx]
    a, a_s = xr[bi, a_idx], sr[bi, a_idx]; b, b_s = xr[bi, b_idx], sr[bi, b_idx]
    b_acc = (b * b_s).scatter_add(1, dst.unsqueeze(-1).expand(B, r, C), a * a_s)
    s_acc = b_s.scatter_add(1, dst.unsqueeze(-1).expand(B, r, 1), a_s)
    return torch.cat([cls, prot, b_acc / s_acc], 1), torch.cat([cls_s, prot_s, s_acc], 1)


@torch.no_grad()
def forward_pitome(m, x, r):
    t = m._pos_embed(m.patch_embed(x)); B, T, C = t.shape
    size = torch.ones(B, T, 1, dtype=t.dtype, device=t.device)
    for blk in m.blocks:
        t = blk(t); t, size = pitome_reduce(t, size, r)
    return m.norm(t)[:, 0], t.shape[1]


def main():
    m, X, Y, dev = load_model_and_data(MODEL, N)
    nprefix = m.num_prefix_tokens
    print(f"{MODEL} · {len(X)}장 · dev={dev} · PiToMe margin={MARGIN}", flush=True)
    print(f"{'r':>3} {'comp%':>6} {'tome':>7} {'ours':>7} {'PiToMe(real)':>13}  판정", flush=True)
    for r in [8, 12, 16, 18, 20]:
        feats = {}
        for s, npr in [("tome", 1), ("ours", nprefix)]:
            fs = [forward_kprotect(m, X[i:i+128].to(dev), r, npr)[0].float().cpu() for i in range(0, len(X), 128)]
            feats[s] = knn(torch.cat(fs), Y)
        pf, finalT = [], None
        for i in range(0, len(X), 128):
            e, finalT = forward_pitome(m, X[i:i+128].to(dev), r)
            pf.append(e.float().cpu())
        feats["pitome"] = knn(torch.cat(pf), Y)
        comp = 100 * (1 - finalT / (m.patch_embed.num_patches + nprefix))
        d = feats["ours"] - feats["pitome"]
        v = "ours>PiToMe✅" if d > 0.5 else ("≈" if abs(d) <= 0.5 else "PiToMe우위❌")
        print(f"{r:>3} {comp:>6.1f} {feats['tome']:>7.2f} {feats['ours']:>7.2f} {feats['pitome']:>13.2f}  Δ={d:+.2f} {v}", flush=True)
    print("\n해석: ours가 실제 PiToMe를 극단압축서 이기면 = 'register 보호 > energy 기반 SOTA' 직접 근거.", flush=True)


if __name__ == "__main__":
    main()
