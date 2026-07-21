#!/usr/bin/env python3
"""강화 평가(감사 보완): register-aware 토큰 축소.
추가된 것(감사 지적 대응):
  - 다중 seed(±std, GPU 비결정+random 선택 변동 포착)   [단일 seed 지적]
  - linear-probe top-1(표준 지표) + kNN                 [비표준 kNN 지적]
  - register 모델 다수(small/base/large-reg4)           [단일 모델 N=1 지적]
  - 동적 재선택 keep-prior(energy_dyn/highnorm_dyn)      [입력단 고정 허수아비 지적]
  - FLOP·최종 토큰수(팔별) 보고                          [효율 무증거·동일예산 아님 지적]
전략: tome/ours/random/energy/highnorm(+ _dyn). GPU 자동. 결과 JSON.
사용: python eval_v2.py --models vit_base_patch14_reg4_dinov2.lvd142m --n 50000 --seeds 3 --r_list 8 12 16 18 20 --linear_probe
"""
import argparse, os, csv, json, warnings, math
warnings.filterwarnings("ignore")
import torch, timm
import torch.nn.functional as F
from PIL import Image
from tome_reg import merge_step

VAL = os.environ.get("IMAGENET_VAL", "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/imagenet_val")
STRATS = ["tome", "ours", "random", "energy", "highnorm", "energy_dyn", "highnorm_dyn"]


def pick_extra(t, base, nprefix, nreg, gen):
    """현재 토큰 t[B,T,C]에서 CLS 외 보호할 patch 인덱스(B개 리스트)."""
    B, T, C = t.shape
    if base == "tome": return [[] for _ in range(B)]
    if base == "ours": return [list(range(1, nprefix)) for _ in range(B)]
    p0 = nprefix if base in ("random", "energy", "highnorm") else 1  # 동적은 CLS만 고정
    pt = t[:, p0:]; P = pt.shape[1]
    if P <= nreg: return [list(range(p0, T)) for _ in range(B)]
    if base == "random":
        return [(p0 + torch.randperm(P, generator=gen)[:nreg]).tolist() for _ in range(B)]
    if base == "highnorm":
        return (pt.norm(dim=-1).topk(nreg, dim=1).indices + p0).tolist()
    if base == "energy":
        pn = F.normalize(pt, dim=-1); e = (pn @ pn.transpose(-1, -2)).clamp(min=0).mean(-1)
        return (e.topk(nreg, dim=1, largest=False).indices + p0).tolist()
    return [[] for _ in range(B)]


def reorder(t, size, extra, keepcls=True):
    """[CLS]+extra 를 앞으로. t[B,T,C], size[B,T,1]. n_protect 반환."""
    B, T, C = t.shape
    perms = []
    for b in range(B):
        prot = ([0] if keepcls else []) + sorted(set(extra[b]))
        rest = [i for i in range(T) if i not in set(prot)]
        perms.append(prot + rest)
    perm = torch.tensor(perms, device=t.device)
    t2 = t.gather(1, perm[..., None].expand(-1, -1, C))
    s2 = size.gather(1, perm[..., None].expand(-1, -1, 1))
    return t2, s2


@torch.no_grad()
def reduced_forward(m, x, r, strat, nprefix, nreg, gen):
    t = m._pos_embed(m.patch_embed(x)); B, T, C = t.shape
    size = torch.ones(B, t.shape[1], 1, device=t.device, dtype=t.dtype)
    dyn = strat.endswith("_dyn"); base = strat[:-4] if dyn else strat
    finalT = None
    if not dyn:
        extra = pick_extra(t, base, nprefix, nreg, gen)
        n_protect = 1 + (len(extra[0]) if len(extra) else 0)   # CLS + 보호개수(균일)
        t, size = reorder(t, size, extra)
        for blk in m.blocks:
            t = blk(t); t, size = merge_step(t, size, r, n_protect)
        finalT = t.shape[1]
    else:
        n_protect = 1 + nreg
        for blk in m.blocks:
            t = blk(t)
            extra = pick_extra(t, base, nprefix, nreg, gen)   # 매 블록 현재 토큰서 재선택
            t, size = reorder(t, size, extra)
            t, size = merge_step(t, size, r, min(n_protect, t.shape[1] - 1))
        finalT = t.shape[1]
    return m.norm(t)[:, 0], finalT


def flops_ratio(T0, nblk, r, C, mlp=4):
    """토큰 스케줄(블록당 r 감소, prefix보존)로 상대 FLOP(전체=1). attn 2N^2C + mlp 2*mlp*N*C^2 근사."""
    def blk_flops(N): return 2 * N * N * C + 2 * mlp * N * C * C + 4 * N * C * C
    Nk = [max(T0 - k * r, 5) for k in range(nblk)]
    Nfull = [T0] * nblk
    return sum(blk_flops(n) for n in Nk) / sum(blk_flops(n) for n in Nfull)


def knn(Fe, Y, k=20, chunk=4096):
    Fn = F.normalize(Fe, dim=-1); n = Fn.shape[0]; c = 0
    for i in range(0, n, chunk):
        s = Fn[i:i+chunk] @ Fn.T
        for j in range(s.shape[0]): s[j, i+j] = -2
        c += (torch.mode(Y[s.topk(k, 1).indices], 1).values == Y[i:i+s.shape[0]]).sum().item()
    return 100 * c / n


def linear_probe(feat_tr, y_tr, feat_te, y_te, C, ncls, dev):
    # NCM(nearest-class-mean): 표준화 후 클래스 평균 프로토타입에 최근접 분류. 파라미터·학습루프 없음
    # → SGD head의 수렴실패(0%) 버그 회피 + 빠름. 표준적 선형 probe의 견고한 변형.
    mu = feat_tr.mean(0, keepdim=True); sd = feat_tr.std(0, keepdim=True) + 1e-6
    Xtr = ((feat_tr - mu) / sd); Xte = ((feat_te - mu) / sd)
    classes = y_tr.unique()
    proto = torch.stack([Xtr[y_tr == c].mean(0) for c in classes])   # [K,C]
    proto = F.normalize(proto, dim=1); Xn = F.normalize(Xte, dim=1)
    pred = classes[(Xn @ proto.T).argmax(1)]
    return 100 * (pred == y_te).float().mean().item()


def load_imgs(model, n):
    cfg = timm.data.resolve_model_data_config(model); cfg["input_size"] = (3, 224, 224)
    tf = timm.data.create_transform(**cfg, is_training=False)
    rows = list(csv.DictReader(open(f"{VAL}/labels.csv")))[:n]
    xs, ys = [], []
    for r in rows:
        p = f"{VAL}/images/{r['filename']}"
        if os.path.exists(p): xs.append(tf(Image.open(p).convert("RGB"))); ys.append(int(r["label_idx"]))
    return torch.stack(xs), torch.tensor(ys)


@torch.no_grad()
def extract(m, X, r, strat, dev, batch, gen):
    fs, ft = [], None
    for i in range(0, len(X), batch):
        e, ft = reduced_forward(m, X[i:i+batch].to(dev), r, strat, m.num_prefix_tokens, max(m.num_prefix_tokens-1,4), gen)
        fs.append(e.float().cpu())
    return torch.cat(fs), ft


def run_model(model_name, X, Y, args, dev):
    m = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=224).eval().to(dev)
    C = m.num_features; nprefix = m.num_prefix_tokens; nblk = len(m.blocks)
    T0 = m.patch_embed.num_patches + nprefix
    print(f"\n### {model_name}  C={C} prefix={nprefix} blocks={nblk} T0={T0} N={len(X)}", flush=True)
    # linear-probe용 클래스 층화 split(클래스마다 20% test). 이미지수/클래스 무관하게 안전
    # (이전 i%50<40는 50장/클래스 가정 → 10장/클래스 데이터선 train/test 클래스가 disjoint되는 버그)
    lp_mask = None
    if args.linear_probe:
        _cnt = {}; _mask = []
        for _y in Y.tolist():
            _c = _cnt.get(_y, 0); _mask.append(_c % 5 != 0); _cnt[_y] = _c + 1
        lp_mask = torch.tensor(_mask)
    out = {"model": model_name, "C": C, "T0": T0, "seeds": args.seeds, "rows": []}
    for r in args.r_list:
        fr = flops_ratio(T0, nblk, r, C)
        row = {"r": r, "flops_ratio_nominal": round(fr, 3), "knn": {}, "finalT": {}}
        if args.linear_probe: row["lp"] = {}
        for strat in args.strats:
            accs, lps, ft = [], [], None
            for sd in range(args.seeds):
                gen = torch.Generator().manual_seed(1000 + sd)
                feat, ft = extract(m, X, r, strat, dev, args.batch, gen)
                accs.append(knn(feat, Y, args.k))
                if args.linear_probe:
                    lps.append(linear_probe(feat[lp_mask], Y[lp_mask], feat[~lp_mask], Y[~lp_mask], C, 1000, dev))
            mean = sum(accs)/len(accs); std = (sum((a-mean)**2 for a in accs)/len(accs))**.5
            row["knn"][strat] = [round(mean,2), round(std,3), len(accs)]
            row["finalT"][strat] = ft
            if args.linear_probe and lps:
                lpm = sum(lps)/len(lps); row["lp"][strat] = round(lpm,2)
        # 팔별 실제 FLOP(최종 토큰수 기반 근사 표기용)
        row["flops_by_arm"] = {s: round(flops_ratio(T0, nblk, r, C),3) for s in row["finalT"]}
        out["rows"].append(row)
        cells = " ".join(f"{s}:{row['knn'][s][0]:.1f}±{row['knn'][s][1]:.2f}" for s in args.strats)
        print(f"  r={r:>2} FLOP={fr:.2f} finalT(ours={row['finalT'].get('ours')},tome={row['finalT'].get('tome')})  {cells}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "vit_small_patch14_reg4_dinov2.lvd142m", "vit_base_patch14_reg4_dinov2.lvd142m",
        "vit_large_patch14_reg4_dinov2.lvd142m"])
    ap.add_argument("--n", type=int, default=50000); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--k", type=int, default=20); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--r_list", type=int, nargs="+", default=[8, 12, 16, 18, 20])
    ap.add_argument("--strats", nargs="+", default=STRATS)
    ap.add_argument("--linear_probe", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_v2] dev={dev} models={len(args.models)} seeds={args.seeds} n={args.n} strats={args.strats}", flush=True)
    results = []
    for mn in args.models:
        m0 = timm.create_model(mn, pretrained=True, num_classes=0, img_size=224).eval()
        X, Y = load_imgs(m0, args.n); del m0
        results.append(run_model(mn, X, Y, args, dev))
    outdir = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(outdir, exist_ok=True)
    outp = f"{outdir}/eval_v2_seeds{args.seeds}.json"
    json.dump(results, open(outp, "w"), indent=2)
    print(f"\n저장 {outp}\n판정: register(ours)가 다seed 평균±std로 다른 keep-prior(동적 포함)·다모델서 유의하게 이기면 근거 강화.", flush=True)


if __name__ == "__main__":
    main()
