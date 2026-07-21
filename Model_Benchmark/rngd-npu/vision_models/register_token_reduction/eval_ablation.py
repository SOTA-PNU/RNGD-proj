#!/usr/bin/env python3
"""결정적 ablation: 같은 size-가중 ToMe 병합(tome_reg.merge_step)에서 '무엇을 보호하느냐'만 바꿔
register 보호가 무작위/에너지/고노름 보호보다 나은지 측정. main.md의 [TODO ablation: register vs
energy vs random]과 SPEC '결정적 ablation'을 채운다. GPU/CPU 공용, 결과를 JSON으로도 저장.

전략(모두 CLS는 공통 보호, 추가로 #reg개 보호 — register는 다른 전략에선 mergeable=공정):
  tome     : 추가 보호 0 (CLS만)                      ← 보호-없음 바닥
  ours     : register 토큰 #reg개 보호
  random   : 무작위 patch #reg개 보호
  energy   : 저에너지(평균 유사도 낮은=고유) patch #reg개 보호  ← PiToMe식 keep-prior
  highnorm : 고노름 patch #reg개 보호                 ← register 없는 모델용 대리

⚠️ 이는 '보호 기준(keep-prior)'의 통제 비교(병합 메커니즘=ToMe 고정). PiToMe '방법 전체'(층별 에너지
재채점 병합)와의 비교는 공식 PiToMe repo로 별도 수행 권장. 사용:
  python eval_ablation.py --n 50000 --batch 128 --r_list 8 12 16 18 20   (GPU면 풀)
"""
import argparse, os, csv, json, warnings
warnings.filterwarnings("ignore")
import torch, timm
import torch.nn.functional as F
from PIL import Image
from tome_reg import merge_step

VAL = os.environ.get("IMAGENET_VAL", "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/imagenet_val")
STRATS = ["tome", "ours", "random", "energy", "highnorm"]


def chosen_extra(t, strat, nprefix, nreg, gen):
    """t:[B,T,C] (pos-embed 후). 각 이미지가 추가 보호할 patch 인덱스 리스트(B개) 반환. CLS·register는 제외 후보."""
    B, T, C = t.shape
    if strat in ("tome",):
        return [[] for _ in range(B)]
    if strat == "ours":
        return [list(range(1, nprefix)) for _ in range(B)]      # register 슬롯
    patch0 = nprefix
    pt = t[:, patch0:]                                           # [B,P,C]
    P = pt.shape[1]
    if strat == "random":
        out = []
        for b in range(B):
            sel = (patch0 + torch.randperm(P, generator=gen)[:nreg]).tolist()
            out.append(sel)
        return out
    if strat == "highnorm":
        idx = pt.norm(dim=-1).topk(nreg, dim=1).indices + patch0   # [B,nreg]
        return idx.tolist()
    if strat == "energy":
        pn = F.normalize(pt, dim=-1)
        sim = pn @ pn.transpose(-1, -2)                          # [B,P,P]
        energy = sim.clamp(min=0).mean(dim=-1)                   # [B,P] 높을수록 중복(redundant)
        idx = energy.topk(nreg, dim=1, largest=False).indices + patch0   # 저에너지=고유 → 보호(PiToMe식)
        return idx.tolist()
    return [[] for _ in range(B)]


@torch.no_grad()
def reduced_forward_strat(model, x, r_pb, strat, nprefix, nreg, gen):
    t = model._pos_embed(model.patch_embed(x))                  # [B,T,C]
    B, T, C = t.shape
    extra = chosen_extra(t, strat, nprefix, nreg, gen)
    # 각 이미지: [CLS]+extra(보호) 를 앞으로 재배열
    perms = []
    for b in range(B):
        prot = [0] + sorted(extra[b])
        rest = [i for i in range(T) if i not in set(prot)]
        perms.append(prot + rest)
    perm = torch.tensor(perms, device=t.device)
    n_protect = 1 + (nreg if strat != "tome" else 0)
    t = t.gather(1, perm[..., None].expand(-1, -1, C))
    size = torch.ones(B, T, 1, device=t.device, dtype=t.dtype)
    for blk in model.blocks:
        t = blk(t)
        t, size = merge_step(t, size, r_pb, n_protect)
    return model.norm(t)[:, 0], t.shape[1]


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
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
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
    print(f"[setup] {args.model} dev={dev} N={N} prefix={nprefix}(reg={nreg}) patches={npatch}", flush=True)

    # 무압축 상한
    with torch.no_grad():
        full = torch.cat([m.norm(m.blocks(m._pos_embed(m.patch_embed(X[i:i+args.batch].to(dev)))))[:, 0].float().cpu()
                          for i in range(0, N, args.batch)])
    res = {"model": args.model, "N": N, "full_knn": round(knn(full, Y, args.k), 2), "rows": []}
    print(f"full kNN={res['full_knn']:.2f}\n", flush=True)
    hdr = f"{'r':>3} {'comp%':>6} " + " ".join(f"{s:>9}" for s in args.strats)
    print(hdr, flush=True)
    for r_pb in args.r_list:
        accs = {}; finalT = None
        for s in args.strats:
            feats = []
            for i in range(0, N, args.batch):
                e, ft = reduced_forward_strat(m, X[i:i+args.batch].to(dev), r_pb, s, nprefix, nreg, gen)
                feats.append(e.float().cpu()); finalT = ft
            accs[s] = round(knn(torch.cat(feats), Y, args.k), 2)
        comp = round(100 * (1 - finalT / T0), 1)
        res["rows"].append({"r": r_pb, "comp": comp, **accs})
        print(f"{r_pb:>3} {comp:>6.1f} " + " ".join(f"{accs[s]:>9.2f}" for s in args.strats), flush=True)

    outdir = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, f"ablation_{args.model.split('.')[0]}_n{N}.json")
    json.dump(res, open(outp, "w"), indent=2)
    print(f"\n저장: {outp}", flush=True)
    print("판정: 극단압축서 ours가 random/energy/highnorm을 이기면 = register keep-prior의 특수성 입증.", flush=True)


if __name__ == "__main__":
    main()
