#!/usr/bin/env python3
"""[50k] 이미지 검색(retrieval) mAP — ToMe vs Ours.  (감사 ③ 보강: 지표 다양화)
지금까지 지표가 kNN top-1 하나뿐 → '지표 하나 아니냐' 반론 대비. 같은 특징에 표준 검색지표 mAP 추가.
mAP: 각 query를 gallery(자기 제외) 전체에 코사인 유사도로 랭킹, 같은 클래스면 relevant.
     average precision을 query마다 구해 평균(mean AP). kNN(투표)과 다른 '랭킹 품질' 축.
k=0=ToMe(CLS만 보호), k=4=Ours(CLS+register4). 같은 size-가중 병합.
사용(전체): python retrieval_map.py 50000"""
import sys, torch, torch.nn.functional as F
from tome_core import forward_kprotect, load_model_and_data

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"


def retrieval_map(Fe, Y, chunk=256):
    """mean Average Precision(%). chunk 단위 벡터화."""
    Fn = F.normalize(Fe, dim=-1); n = Fn.shape[0]
    ranks = torch.arange(1, n + 1, dtype=torch.float)
    aps = []
    for i in range(0, n, chunk):
        s = Fn[i:i+chunk] @ Fn.T                       # [b,n]
        b = s.shape[0]
        for j in range(b): s[j, i + j] = -2            # 자기 제외
        order = s.argsort(dim=1, descending=True)      # [b,n]
        rel = (Y[order] == Y[i:i+b].unsqueeze(1)).float()
        nrel = rel.sum(1)                              # [b]
        prec = rel.cumsum(1) / ranks                   # precision@위치
        ap = (prec * rel).sum(1) / nrel.clamp(min=1)   # [b]
        aps.extend(ap[nrel > 0].tolist())
    return 100 * sum(aps) / len(aps)


def main():
    m, X, Y, dev = load_model_and_data(MODEL, N)
    nprefix = m.num_prefix_tokens
    print(f"{MODEL} · {len(X)}장 · dev={dev} · 검색 mAP", flush=True)
    print(f"{'r':>3} {'comp%':>6} {'ToMe mAP':>9} {'Ours mAP':>9}  판정", flush=True)
    for r in [8, 12, 16, 18, 20]:
        res = {}; ft = None
        for name, npr in [("tome", 1), ("ours", nprefix)]:
            fs = []
            for i in range(0, len(X), 128):
                e, ft = forward_kprotect(m, X[i:i+128].to(dev), r, npr)
                fs.append(e.float().cpu())
            res[name] = retrieval_map(torch.cat(fs), Y)
        comp = 100 * (1 - ft / (m.patch_embed.num_patches + nprefix))
        d = res["ours"] - res["tome"]
        v = "ours>ToMe✅" if d > 0.3 else ("≈" if abs(d) <= 0.3 else "ToMe우위❌")
        print(f"{r:>3} {comp:>6.1f} {res['tome']:>9.2f} {res['ours']:>9.2f}  Δ={d:+.2f} {v}", flush=True)
    print("\n해석: kNN 외 검색 mAP(랭킹 지표)에서도 register 보호가 이기면 = 특징 자체가 우수(지표 무관).", flush=True)


if __name__ == "__main__":
    main()
