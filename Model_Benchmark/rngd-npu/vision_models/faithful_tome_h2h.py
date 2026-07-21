#!/usr/bin/env python3
"""정식 ToMe(proportional attention + key-metric 유사도 + attn↔MLP 사이 병합) vs Ours(register 보호).
감사 지적 '베이스라인이 정식 ToMe 아님(proportional attention 미적용)'에 대응.
공식 ToMe(Bolya ICLR'23, PiToMe repo algo/tome/patch/timm.py) 그대로:
  - proportional attention: 스케일된 attn logit에 log(size) 더함(SDPA attn_mask).
  - 유사도: attention key(head 평균)로 bipartite soft matching.
  - 병합: 블록의 attn 뒤·MLP 앞에서 size-가중 평균.
tome=CLS만 보호, ours=CLS+register 보호. 둘 다 동일한 정식 메커니즘 → 공정. DINOv2-reg, kNN.
사용: python faithful_tome_h2h.py [n]"""
import os, sys, csv
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import warnings; warnings.filterwarnings("ignore")
import torch, timm, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__) + "/register_token_reduction/eval_v2")
from eval_v2 import knn

VAL = os.environ.get("IMAGENET_VAL", "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/imagenet_val")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"


@torch.no_grad()
def merge_metric(x, size, metric, r, n_protect):
    """정식 ToMe bipartite: 유사도는 metric(key), 병합은 x(size-가중). 앞 n_protect 보호."""
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
        xn = blk.norm1(t); B, N, C = xn.shape; hd = C // H
        qkv = blk.attn.qkv(xn).reshape(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        bias = size.log().reshape(B, 1, 1, N) if proportional else None   # 정식 proportional attention
        xa = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)       # 스케일 내부적용
        xa = blk.attn.proj(xa.transpose(1, 2).reshape(B, N, C))
        metric = k.mean(1)                                                # key-metric(head 평균)
        t = t + blk.drop_path1(blk.ls1(xa))
        t, size = merge_metric(t, size, metric, r, n_protect)             # attn↔MLP 사이 병합
        t = t + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(t))))
    return m.norm(t)[:, 0], t.shape[1]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=224).eval().to(dev)
    nprefix = m.num_prefix_tokens
    cfg = timm.data.resolve_model_data_config(m); cfg["input_size"] = (3, 224, 224)
    tf = timm.data.create_transform(**cfg, is_training=False)
    rows = list(csv.DictReader(open(f"{VAL}/labels.csv")))[:N]
    X = torch.stack([tf(Image.open(f"{VAL}/images/{r['filename']}").convert("RGB")) for r in rows])
    Y = torch.tensor([int(r["label_idx"]) for r in rows])
    print(f"{MODEL} · {N}장 · 정식 ToMe(proportional attn + key-metric + attn↔MLP 병합)", flush=True)
    print(f"{'r':>3} {'comp%':>6} {'ToMe(정식)':>11} {'Ours(정식+reg보호)':>17}  판정", flush=True)
    for r in [12, 16, 20]:
        res = {}; ft = None
        for name, npr in [("tome", 1), ("ours", nprefix)]:
            fs = [forward_faithful(m, X[i:i+40].to(dev), r, npr)[0].float().cpu() for i in range(0, N, 40)]
            res[name] = knn(torch.cat(fs), Y)
        _, ft = forward_faithful(m, X[:40].to(dev), r, nprefix)
        comp = 100 * (1 - ft / (m.patch_embed.num_patches + nprefix))
        d = res["ours"] - res["tome"]
        v = "✅ ours>ToMe" if d > 0.5 else ("≈ 비슷" if abs(d) <= 0.5 else "❌ ToMe 우위")
        print(f"{r:>3} {comp:>6.1f} {res['tome']:>11.2f} {res['ours']:>17.2f}  Δ={d:+.2f} {v}", flush=True)
    print("\n해석: 정식 ToMe(proportional attention 포함)에서도 register 보호가 이기면 = '베이스라인이 약해서'가 아님을 확정.", flush=True)


if __name__ == "__main__":
    main()
