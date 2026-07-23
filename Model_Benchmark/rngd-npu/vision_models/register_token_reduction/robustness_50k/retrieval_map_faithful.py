#!/usr/bin/env python3
"""[faithful] 검색 mAP — 정식(faithful) harness에서 ToMe vs Ours.
= retrieval_map.py 의 forward 만 통제(forward_kprotect)→정식(forward_faithful)으로 교체.
정식 forward = faithful_tome_h2h.forward_faithful (prop-attn + key-metric + attn↔MLP 병합) 재사용.
k=0=ToMe(CLS만), k=nprefix=Ours(CLS+register). 사용: python retrieval_map_faithful.py 50000"""
import sys, torch, torch.nn.functional as F
from tome_core import load_model_and_data
from faithful_tome_h2h import forward_faithful

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"


def retrieval_map(Fe, Y, chunk=256):
    """mean Average Precision(%). retrieval_map.py 와 동일."""
    Fn = F.normalize(Fe, dim=-1); n = Fn.shape[0]
    ranks = torch.arange(1, n + 1, dtype=torch.float)
    aps = []
    for i in range(0, n, chunk):
        s = Fn[i:i+chunk] @ Fn.T
        b = s.shape[0]
        for j in range(b): s[j, i + j] = -2
        order = s.argsort(dim=1, descending=True)
        rel = (Y[order] == Y[i:i+b].unsqueeze(1)).float()
        nrel = rel.sum(1)
        prec = rel.cumsum(1) / ranks
        ap = (prec * rel).sum(1) / nrel.clamp(min=1)
        aps.extend(ap[nrel > 0].tolist())
    return 100 * sum(aps) / len(aps)


def main():
    m, X, Y, dev = load_model_and_data(MODEL, N)
    nprefix = m.num_prefix_tokens
    print(f"{MODEL} · {len(X)}장 · dev={dev} · 검색 mAP [faithful: prop-attn+key-metric+attn↔MLP병합]", flush=True)
    print(f"{'r':>3} {'comp%':>6} {'ToMe mAP':>9} {'Ours mAP':>9}  판정", flush=True)
    for r in [8, 12, 16, 18, 20]:
        res = {}; ft = None
        for name, npr in [("tome", 1), ("ours", nprefix)]:
            fs = []
            for i in range(0, len(X), 128):
                e, ft = forward_faithful(m, X[i:i+128].to(dev), r, npr)
                fs.append(e.float().cpu())
            res[name] = retrieval_map(torch.cat(fs), Y)
        comp = 100 * (1 - ft / (m.patch_embed.num_patches + nprefix))
        d = res["ours"] - res["tome"]
        v = "ours>ToMe✅" if d > 0.3 else ("≈" if abs(d) <= 0.3 else "ToMe우위❌")
        print(f"{r:>3} {comp:>6.1f} {res['tome']:>9.2f} {res['ours']:>9.2f}  Δ={d:+.2f} {v}", flush=True)
    print("\n해석: 정식 harness서도 검색 mAP에서 register 보호가 이기면 = 특징 자체 우수(지표·harness 무관).", flush=True)


if __name__ == "__main__":
    main()
