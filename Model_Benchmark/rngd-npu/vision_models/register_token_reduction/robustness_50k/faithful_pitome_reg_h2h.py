#!/usr/bin/env python3
"""[일반성 검증] 레지스터 보호가 ToMe 전용 트릭이 아니라 병합기와 무관한 keep-rule임을 실증.
공식 PiToMe 병합 위에도 레지스터 보호를 얹어(PiToMe+reg) 비교한다. PiToMe+reg > PiToMe 이면
= 레지스터 보호가 PiToMe 병합에서도 이득 → "ToMe에 더한 것"이 아니라 병합기 무관 일반 규칙.
4-arm: tome(CLS만) / pitome(CLS만, 공식) / pitome_reg(CLS+register) / ours(=ToMe+register).
검증된 faithful_pitome_h2h.py 의 tome_merge·공식 pitome selection(_make_official_pitome_merge)을
그대로 import 재사용하고, pitome_merge 만 앞 n_protect개 보호로 일반화(n_protect=1이면 공식과 동일).
자체 정합: tome/ours/pitome(n_protect=1) 는 faithful_pitome_50k.log 와 일치해야 함.
사용(전체): python faithful_pitome_reg_h2h.py 50000
"""
import sys, math, torch, torch.nn.functional as F
from tome_core import knn, load_model_and_data
from faithful_pitome_h2h import tome_merge, _make_official_pitome_merge

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"
RLIST = [8, 12, 16, 18, 20]


@torch.no_grad()
def pitome_merge_np(x, size, metric, r, n_protect, margin, use_bsm):
    """공식 PiToMe 병합. 앞 n_protect개(CLS[+register])는 무조건 보호(병합 후보서 제외).
    n_protect=1 이면 공식 pitome_merge 와 완전히 동일. n_protect=nprefix 면 register까지 강제 보호."""
    xc, xr = x[:, :n_protect], x[:, n_protect:]
    sc, sr = size[:, :n_protect], size[:, n_protect:]
    mr = metric[:, n_protect:]
    P = xr.shape[1]
    r = min(r, P // 2)
    if r <= 0:
        return x, size
    merge = _make_official_pitome_merge(mr, r, margin, use_bsm)   # 나머지 패치에 공식 에너지 selection
    xo = merge(xr * sr, mode="sum"); so = merge(sr, mode="sum"); xo = xo / so   # merge_wavg
    return torch.cat([xc, xo], 1), torch.cat([sc, so], 1)


@torch.no_grad()
def forward_faithful(m, x, r, strat, nprefix, proportional=True):
    """strat: tome / pitome / pitome_reg / ours. reg 붙은 것(pitome_reg)과 ours만 register 보호."""
    n_protect = nprefix if strat in ("ours", "pitome_reg") else 1
    t = m._pos_embed(m.patch_embed(x)); B = t.shape[0]
    size = torch.ones(B, t.shape[1], 1, dtype=t.dtype, device=t.device)
    H = m.blocks[0].attn.num_heads; L = len(m.blocks); n_bsm = math.ceil(L * 0.5)
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
        if strat in ("pitome", "pitome_reg"):
            t, size = pitome_merge_np(t, size, metric, r, n_protect, margin, use_bsm=(li < n_bsm))
        else:
            t, size = tome_merge(t, size, metric, r, n_protect)
        t = t + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(t))))
    return m.norm(t)[:, 0], t.shape[1]


def main():
    m, X, Y, dev = load_model_and_data(MODEL, N)
    nprefix = m.num_prefix_tokens
    print(f"{MODEL} · {len(X)}장 · dev={dev} · [일반성] 레지스터 보호가 PiToMe 병합 위에서도 이득인가", flush=True)
    print(f"{'r':>3} {'comp%':>6} {'ToMe':>7} {'PiToMe':>7} {'PiTo+reg':>8} {'Ours':>7} "
          f"{'reg효과@PiTo':>11} {'reg효과@ToMe':>11}", flush=True)
    for r in RLIST:
        res = {}
        for s in ["tome", "pitome", "pitome_reg", "ours"]:
            fs = [forward_faithful(m, X[i:i+128].to(dev), r, s, nprefix)[0].float().cpu() for i in range(0, len(X), 128)]
            res[s] = knn(torch.cat(fs), Y)
        _, ft = forward_faithful(m, X[:128].to(dev), r, "ours", nprefix)
        comp = 100 * (1 - ft / (m.patch_embed.num_patches + nprefix))
        d_pito = res["pitome_reg"] - res["pitome"]   # PiToMe 병합에 register 얹은 이득
        d_tome = res["ours"] - res["tome"]            # ToMe 병합에 register 얹은 이득(기존 헤드라인)
        print(f"{r:>3} {comp:>6.1f} {res['tome']:>7.2f} {res['pitome']:>7.2f} {res['pitome_reg']:>8.2f} "
              f"{res['ours']:>7.2f} {d_pito:>+11.2f} {d_tome:>+11.2f}", flush=True)
    print("\n해석: 'reg효과@PiTo'(PiToMe+reg − PiToMe) > 0 이면 = 레지스터 보호가 PiToMe 병합 위에서도 이득 "
          "→ 병합기 무관 일반 규칙 실증(ToMe효과와 나란히). tome/ours/pitome 는 faithful_pitome_50k.log 와 일치해야 함(정합).", flush=True)


if __name__ == "__main__":
    main()
