#!/usr/bin/env python3
"""[faithful] keep-prior ablation을 정식(faithful) harness에서 재실행.
= eval_ablation.py 의 전략별 정적 보호선택(chosen_extra) 그대로 + 병합 forward만 통제→정식으로 교체.
정식 harness = proportional attention(size.log bias) + key-metric(k.mean) + attn↔MLP 사이 병합
(Bolya ToMe ICLR'23 공식). tab:ablation 을 정식 harness 로 승급(헤드라인과 'Ours' 정의 일치).
전략(모두 CLS 공통 보호, 추가로 #reg개): tome / ours(register) / random / energy / highnorm.
사용(전체): python eval_ablation_faithful.py --n 50000 --batch 128 --r_list 8 12 16 18 20
"""
import argparse, os, csv, json, warnings
warnings.filterwarnings("ignore")
import torch, timm
import torch.nn.functional as F
from PIL import Image

VAL = os.environ.get("IMAGENET_VAL", "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/imagenet_val")
STRATS = ["tome", "ours", "random", "energy", "highnorm"]


def chosen_extra(t, strat, nprefix, nreg, gen):
    """eval_ablation.py 와 동일: 입력층에서 전략별로 추가 보호할 patch 인덱스(정적 선택)."""
    B, T, C = t.shape
    if strat == "tome":
        return [[] for _ in range(B)]
    if strat == "ours":
        return [list(range(1, nprefix)) for _ in range(B)]
    patch0 = nprefix
    pt = t[:, patch0:]; P = pt.shape[1]
    if strat == "random":
        return [(patch0 + torch.randperm(P, generator=gen)[:nreg]).tolist() for _ in range(B)]
    if strat == "highnorm":
        return (pt.norm(dim=-1).topk(nreg, dim=1).indices + patch0).tolist()
    if strat == "energy":
        pn = F.normalize(pt, dim=-1); sim = pn @ pn.transpose(-1, -2)
        energy = sim.clamp(min=0).mean(dim=-1)
        return (energy.topk(nreg, dim=1, largest=False).indices + patch0).tolist()
    return [[] for _ in range(B)]


@torch.no_grad()
def merge_metric(x, size, metric, r, n_protect):
    """정식 ToMe: key-metric bipartite soft matching, size-가중 (faithful_tome_h2h 와 동일)."""
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
def reduced_forward_strat_faithful(m, x, r_pb, strat, nprefix, nreg, gen, proportional=True):
    """전략별 정적 보호선택 + 정식 forward(prop-attn + key-metric + attn↔MLP 병합)."""
    t = m._pos_embed(m.patch_embed(x)); B, T, C = t.shape
    extra = chosen_extra(t, strat, nprefix, nreg, gen)
    perms = []
    for b in range(B):
        prot = [0] + sorted(extra[b]); rest = [i for i in range(T) if i not in set(prot)]
        perms.append(prot + rest)
    t = t.gather(1, torch.tensor(perms, device=t.device)[..., None].expand(-1, -1, C))   # 보호를 앞으로
    n_protect = 1 + (nreg if strat != "tome" else 0)
    size = torch.ones(B, T, 1, device=t.device, dtype=t.dtype)
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
        t, size = merge_metric(t, size, metric, r_pb, n_protect)
        t = t + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(t))))
    return m.norm(t)[:, 0], t.shape[1]


def knn(Fe, Y, k=20, chunk=2048):
    Fn = F.normalize(Fe, dim=-1); n = Fn.shape[0]; correct = 0
    for i in range(0, n, chunk):
        s = Fn[i:i+chunk] @ Fn.T
        for j in range(s.shape[0]): s[j, i+j] = -2.0
        idx = s.topk(k, dim=1).indices
        correct += (torch.mode(Y[idx], dim=1).values == Y[i:i+s.shape[0]]).sum().item()
    return 100 * correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--r_list", type=int, nargs="+", default=[8, 12, 16, 18, 20])
    ap.add_argument("--strats", nargs="+", default=STRATS)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = timm.create_model(args.model, pretrained=True, num_classes=0, img_size=224).eval().to(dev)
    nprefix = getattr(m, "num_prefix_tokens", 1); nreg = max(nprefix - 1, 4)
    cfg = timm.data.resolve_model_data_config(m); cfg["input_size"] = (3, 224, 224)
    tf = timm.data.create_transform(**cfg, is_training=False)
    rows = list(csv.DictReader(open(f"{VAL}/labels.csv")))[:args.n]
    xs, ys = [], []
    for r in rows:
        p = f"{VAL}/images/{r['filename']}"
        if os.path.exists(p): xs.append(tf(Image.open(p).convert("RGB"))); ys.append(int(r["label_idx"]))
    X = torch.stack(xs); Y = torch.tensor(ys); N = len(ys)
    npatch = (224 // m.patch_embed.patch_size[0]) ** 2; T0 = nprefix + npatch
    gen = torch.Generator().manual_seed(20260705)
    print(f"[setup·faithful] {args.model} dev={dev} N={N} prefix={nprefix}(reg={nreg}) patches={npatch} (prop-attn+key+attn↔MLP병합)", flush=True)

    with torch.no_grad():
        full = torch.cat([m.norm(m.blocks(m._pos_embed(m.patch_embed(X[i:i+args.batch].to(dev)))))[:, 0].float().cpu()
                          for i in range(0, N, args.batch)])
    res = {"model": args.model, "N": N, "harness": "faithful", "full_knn": round(knn(full, Y, args.k), 2), "rows": []}
    print(f"full kNN={res['full_knn']:.2f}\n", flush=True)
    print(f"{'r':>3} {'comp%':>6} " + " ".join(f"{s:>9}" for s in args.strats), flush=True)
    for r_pb in args.r_list:
        accs = {}; finalT = None
        for s in args.strats:
            feats = []
            for i in range(0, N, args.batch):
                e, ft = reduced_forward_strat_faithful(m, X[i:i+args.batch].to(dev), r_pb, s, nprefix, nreg, gen)
                feats.append(e.float().cpu()); finalT = ft
            accs[s] = round(knn(torch.cat(feats), Y, args.k), 2)
        comp = round(100 * (1 - finalT / T0), 1)
        res["rows"].append({"r": r_pb, "comp": comp, **accs})
        print(f"{r_pb:>3} {comp:>6.1f} " + " ".join(f"{accs[s]:>9.2f}" for s in args.strats), flush=True)

    outdir = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, f"ablation_FAITHFUL_{args.model.split('.')[0]}_n{N}.json")
    json.dump(res, open(outp, "w"), indent=2)
    print(f"\n저장: {outp}", flush=True)
    print("판정: 정식 harness서도 극단압축서 ours가 random/energy/highnorm 이기면 = register keep-prior 특수성 유지.", flush=True)


if __name__ == "__main__":
    main()
