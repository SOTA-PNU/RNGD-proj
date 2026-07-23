#!/usr/bin/env python3
"""[faithful · train 갤러리] 보호 register 개수 스윕(k=0..#reg) + 부트스트랩 CI 를 정통 train-갤러리 kNN 에서.
val-LOO(reg_count_sweep_faithful.py)와 **같은 정식 forward**(faithful_tome_h2h.forward_faithful, n_protect 분기)를 그대로,
평가만 train 갤러리로. k=0=ToMe(CLS만) … k=#reg=Ours(CLS+register 전부).
부트스트랩 95%CI(ours-tome) 는 val 쿼리(50k) 재표집(train 갤러리 고정) — 단일 seed 견고성.
※ 두 레이아웃 모두 동작(pitome_compare/ 형제폴더 import, 또는 all_new_server/engine/ 같은폴더).
사용(A100):
  python regcount_train_faithful.py --model vit_base_patch14_reg4_dinov2.lvd142m --gallery train --r_list 8 12 16 18 20
"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
import torch, timm, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "..", "robustness_50k"), os.path.join(HERE, "..", "ablation")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import compare                                          # 데이터/캐시/특징추출 엔진 재사용
from faithful_tome_h2h import forward_faithful          # (m,x,r,n_protect)->(feat,T), 검증됨
from tome_core import bootstrap_ci                      # harness 무관

DATA_ROOT = getattr(compare, "DATA_ROOT", getattr(compare, "HERE", HERE))


@torch.no_grad()
def knn_gallery_correct(Gf, Gy, Qf, Qy, k, dev, chunk=256):
    """train 갤러리 kNN, 쿼리별 정답여부(bool) — 부트스트랩용(compare.knn_gallery 의 per-query 판)."""
    G = F.normalize(Gf.to(dev).float(), dim=-1).half(); Gy = Gy.to(dev)
    Qn = F.normalize(Qf.float(), dim=-1).half(); out = []
    for i in range(0, len(Qn), chunk):
        s = Qn[i:i+chunk].to(dev) @ G.T
        pred = torch.mode(Gy[s.topk(k, dim=1).indices], dim=1).values
        out.append((pred == Qy[i:i+s.shape[0]].to(dev)).cpu())
    return torch.cat(out)


@torch.no_grad()
def knn_loo_correct(Qf, Qy, k, dev, chunk=256):
    """val leave-one-out kNN, 쿼리별 정답여부(bool)."""
    Fn = F.normalize(Qf.to(dev).float(), dim=-1).half(); Y = Qy.to(dev); n = len(Fn); out = []
    for i in range(0, n, chunk):
        s = Fn[i:i+chunk] @ Fn.T
        for j in range(s.shape[0]): s[j, i+j] = -2.0
        out.append((torch.mode(Y[s.topk(k, dim=1).indices], dim=1).values == Y[i:i+s.shape[0]]).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--gallery", choices=["val", "train"], default="train")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--r_list", type=int, nargs="+", default=[8, 12, 16, 18, 20])
    ap.add_argument("--cache_dir", default=os.path.join(HERE, "feat_cache_regcount"))
    ap.add_argument("--gallery_cache", type=int, choices=[0, 1], default=1,
                    help="1=train 갤러리 특징 디스크 캐시(재개 가능, k×r 당 수 GB) / 0=캐시 안 함(디스크 절약, 재개 시 재추출)")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = timm.create_model(args.model, pretrained=True, num_classes=0, img_size=224).eval().to(dev)
    nprefix = getattr(m, "num_prefix_tokens", 1)
    npatch = m.patch_embed.num_patches; L = len(m.blocks); mtag = args.model.split(".")[0]
    tf = compare.make_tf(m)

    # n_protect 로 분기하는 정식 forward 를 엔진에 주입. 캐시키(strat 문자열)를 canonical(tome/ours)과 맞춰 재사용 유도:
    #   k=0 → "tome"(n_protect=1), k=nprefix-1 → "ours"(n_protect=nprefix), 그 외 → "k{n}".
    def label(k):
        if k == 0: return "tome"
        if k == nprefix - 1: return "ours"
        return f"k{1+k}"

    def np_forward(mm, x, r, strat, npref):
        # strat 문자열에서 n_protect 복원: "tome"→1, "ours"→nprefix, "k{n}"→n
        if strat == "tome": npr = 1
        elif strat == "ours": npr = nprefix
        else: npr = int(strat[1:])
        return forward_faithful(mm, x, r, npr)[0]
    compare.reduced_forward = np_forward

    assert os.path.exists(f"{DATA_ROOT}/imagenet_val/DONE"), f"val 미준비: {DATA_ROOT}/imagenet_val (prepare_data.py --split val)"
    if args.gallery == "train":
        assert os.path.exists(f"{DATA_ROOT}/imagenet_train/DONE"), f"train 미준비: {DATA_ROOT}/imagenet_train (prepare_data.py --split train --per_class 1300)"
    proto = "정통 kNN(gallery=train, query=val)" if args.gallery == "train" else "val leave-one-out kNN"
    print(f"[setup·faithful·{args.gallery}] {args.model} dev={dev} prefix={nprefix}(CLS+{nprefix-1}reg) k-sweep 0..{nprefix-1} data={DATA_ROOT}", flush=True)
    print(f"[proto] {proto}, k={args.k}  (캐시키: k0=tome, k{nprefix-1}=ours → canonical 캐시 재사용 가능)", flush=True)
    print(f"{'r':>3} {'comp%':>6} | " + " ".join(f"k={k}".rjust(7) for k in range(nprefix)) +
          "   단조?  95%CI(ours-tome)", flush=True)

    for r in args.r_list:
        corr = {}
        for k in range(nprefix):
            st = label(k)
            Qf, Qy = compare.extract_split(m, "val", r, st, nprefix, tf, args.batch, args.workers, dev, args.cache_dir, mtag)
            if args.gallery == "train":
                Gf, Gy = compare.extract_split(m, "train", r, st, nprefix, tf, args.batch, args.workers, dev, args.cache_dir, mtag, save=bool(args.gallery_cache))
                corr[k] = knn_gallery_correct(Gf, Gy, Qf, Qy, args.k, dev)
            else:
                corr[k] = knn_loo_correct(Qf, Qy, args.k, dev)
        accs = [100 * corr[k].float().mean().item() for k in range(nprefix)]
        mono = "↑" if all(accs[i] <= accs[i+1] + 0.05 for i in range(len(accs)-1)) else "비단조"
        final = nprefix + max(npatch - L * r, 1); comp = 100 * (1 - final / (nprefix + npatch))
        lo, hi, mean = bootstrap_ci(corr[nprefix-1], corr[0])
        sig = "유의✅" if lo > 0 else "≈"
        print(f"{r:>3} {comp:6.1f} | " + " ".join(f"{a:7.2f}" for a in accs) +
              f"   {mono}  Δ={mean:+.2f} [{lo:+.2f},{hi:+.2f}] {sig}", flush=True)
    print("\n해석: train 갤러리·정식 harness 서도 k 단조↑ = 원인=register. CI 하한>0 = 단일 seed 여도 통계적 견고.", flush=True)


if __name__ == "__main__":
    main()
