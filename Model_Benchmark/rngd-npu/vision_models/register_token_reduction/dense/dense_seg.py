#!/usr/bin/env python3
"""ADE20k 선형 probe 분할(segmentation) mIoU로 register-aware 토큰압축 평가.
frozen DINOv2 patch feature에 선형 seg head 1개를 학습(전략 무관 공통) → 압축률·보호전략별로
토큰을 병합·unmerge해 dense feature 복원 → head 적용 → mIoU. dense는 patch 정체성이 중요해
register 보호 이득이 분류보다 클 것으로 기대. GPU 자동. 결과 JSON 저장.
사용: python dense_seg.py --n_train 2000 --n_val 2000 --r_list 0 8 12 16 18 20"""
import argparse, os, json, warnings
warnings.filterwarnings("ignore")
import torch, timm
import torch.nn.functional as F
from tome_reg_dense import reduced_forward_dense

STRATS = ["tome", "ours", "random", "energy", "highnorm"]
NUM_CLASSES = 150          # ADE20k (1..150 유효, 0=미표기=ignore)
IMG = 224
EVAL_RES = 224             # mIoU 계산 해상도(예측 logit을 여기로 upsample)


def load_ade(split, n):
    from datasets import load_dataset
    try:                                                   # datasets>=5: parquet, no trust_remote_code
        ds = load_dataset("scene_parse_150", split=split, streaming=True)
    except TypeError:
        ds = load_dataset("scene_parse_150", split=split, streaming=True, trust_remote_code=True)
    except Exception:                                      # 스크립트형 폴백
        ds = load_dataset("scene_parse_150", split=split, streaming=True, trust_remote_code=True)
    out = []
    for ex in ds:
        out.append((ex["image"], ex["annotation"]))
        if len(out) >= n:
            break
    return out


def to_tensors(img, tf):
    x = tf(img.convert("RGB"))
    return x


def label_tensor(ann, res):
    import numpy as np
    from PIL import Image
    lab = ann.resize((res, res), Image.NEAREST)
    return torch.from_numpy(np.array(lab)).long()          # [res,res], 0..150


@torch.no_grad()
def extract_dense(model, X, r, strat, dev, gen, batch):
    feats, Ts = [], None
    for i in range(0, len(X), batch):
        d, _, ft = reduced_forward_dense(model, X[i:i+batch].to(dev), r, strat, gen)
        feats.append(d.cpu()); Ts = ft
    return torch.cat(feats), Ts                             # [N,Npatch,C]


def train_head(feat_full, labels_patch, C, dev, epochs=60, lr=0.01):
    """feat_full:[N,Npatch,C], labels_patch:[N,hp,wp] (patch격자 라벨). 선형 head 학습(0 무시)."""
    N, Np, _ = feat_full.shape
    hp = int(Np ** 0.5)
    Xtr = feat_full.reshape(-1, C)                          # [N*Np, C]
    ytr = labels_patch.reshape(-1)                          # [N*Np]
    keep = ytr > 0                                          # 0=ignore
    Xtr, ytr = Xtr[keep].to(dev), (ytr[keep] - 1).to(dev)   # 1..150 -> 0..149
    head = torch.nn.Linear(C, NUM_CLASSES).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    n = Xtr.shape[0]; bs = 16384
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            loss = F.cross_entropy(head(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"    head ep{ep} loss={tot/max(1,n//bs):.3f}", flush=True)
    return head.eval()


@torch.no_grad()
def miou(head, feat, labels_full, dev, hp):
    """feat:[N,Npatch,C] → logit [hp,wp,150] → upsample EVAL_RES → argmax → mIoU vs labels_full.
    per-class intersection/union을 bincount로 벡터화(0=ignore)."""
    N = feat.shape[0]
    inter = torch.zeros(NUM_CLASSES); union = torch.zeros(NUM_CLASSES)
    for i in range(N):
        lg = head(feat[i].to(dev)).reshape(hp, hp, NUM_CLASSES).permute(2, 0, 1)[None]  # [1,150,hp,hp]
        up = F.interpolate(lg, size=(EVAL_RES, EVAL_RES), mode="bilinear", align_corners=False)
        pred = up.argmax(1)[0].cpu().reshape(-1)            # 0..149
        gt = labels_full[i].reshape(-1)                     # 0..150
        valid = gt > 0
        p = pred[valid]; g = gt[valid] - 1                  # 0..149
        pc = torch.bincount(p, minlength=NUM_CLASSES).float()
        gc = torch.bincount(g, minlength=NUM_CLASSES).float()
        ic = torch.bincount(p[p == g], minlength=NUM_CLASSES).float()
        inter += ic; union += pc + gc - ic
    present = union > 0
    return 100 * (inter[present] / union[present].clamp(min=1)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--r_list", type=int, nargs="+", default=[0, 8, 12, 16, 18, 20])
    ap.add_argument("--strats", nargs="+", default=STRATS)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = timm.create_model(args.model, pretrained=True, num_classes=0, img_size=IMG).eval().to(dev)
    cfg = timm.data.resolve_model_data_config(m); cfg["input_size"] = (3, IMG, IMG)
    tf = timm.data.create_transform(**cfg, is_training=False)
    C = m.num_features; Np = m.patch_embed.num_patches; hp = int(Np ** 0.5)
    gen = torch.Generator().manual_seed(20260705)
    print(f"[setup] {args.model} dev={dev} C={C} patch grid={hp}x{hp}", flush=True)

    print("데이터 로드(ADE20k)...", flush=True)
    tr = load_ade("train", args.n_train); va = load_ade("validation", args.n_val)
    Xtr = torch.stack([to_tensors(im, tf) for im, _ in tr])
    ytr_patch = torch.stack([label_tensor(an, hp) for _, an in tr])          # [N,hp,hp]
    Xva = torch.stack([to_tensors(im, tf) for im, _ in va])
    yva_full = torch.stack([label_tensor(an, EVAL_RES) for _, an in va])     # [N,res,res]

    print("full feature 추출 + head 학습...", flush=True)
    feat_tr_full, _ = extract_dense(m, Xtr, 0, "tome", dev, gen, args.batch)  # r=0=full
    head = train_head(feat_tr_full, ytr_patch, C, dev, args.epochs)

    T0 = Np + getattr(m, "num_prefix_tokens", 1)
    res = {"model": args.model, "n_train": args.n_train, "n_val": args.n_val, "rows": []}
    print(f"\n{'r':>3} {'comp%':>6} " + " ".join(f"{s:>9}" for s in args.strats) + "   (mIoU)", flush=True)
    for r in args.r_list:
        mious = {}
        if r == 0:                                          # full: 전략 무관 → 한 번 계산 후 공유
            feat, ft = extract_dense(m, Xva, 0, "tome", dev, torch.Generator().manual_seed(1), args.batch)
            v = round(miou(head, feat, yva_full, dev, hp), 2)
            for s in args.strats: mious[s] = v
        else:
            ft = T0
            for s in args.strats:
                feat, ft = extract_dense(m, Xva, r, s, dev, torch.Generator().manual_seed(1), args.batch)
                mious[s] = round(miou(head, feat, yva_full, dev, hp), 2)
        comp = round(100 * (1 - ft / T0), 1)
        res["rows"].append({"r": r, "comp": comp, **{s: mious[s] for s in args.strats}})
        print(f"{r:>3} {comp:>6.1f} " + " ".join(f"{mious[s]:>9.2f}" for s in args.strats), flush=True)

    outdir = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, f"dense_miou_{args.model.split('.')[0]}.json")
    json.dump(res, open(outp, "w"), indent=2)
    print(f"\n저장: {outp}\n판정: 극단압축서 ours mIoU가 tome/random/energy/highnorm보다 크게 높으면 = dense서 register 우위 확정.", flush=True)


if __name__ == "__main__":
    main()
