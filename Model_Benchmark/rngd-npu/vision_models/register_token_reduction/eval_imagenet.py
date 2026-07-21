#!/usr/bin/env python3
"""Register-aware token merging 평가: DINOv2-reg CLS 특징을 압축률·보호전략별 추출 → ImageNet val kNN 정확도.
디바이스 자동(cuda 있으면 GPU). 같은 size-가중 ToMe에서 n_protect만 바꿔 ablation:
  tome(n_protect=1, CLS만)  vs  ours(n_protect=1+#reg, CLS+register).
사용: python eval_imagenet.py --n 2000 --batch 32     (GPU면 --n 50000 가능)"""
import argparse, os, csv, warnings, time
warnings.filterwarnings("ignore")
import torch, timm
import torch.nn.functional as F
from PIL import Image
from tome_reg import reduced_forward

VAL = os.environ.get("IMAGENET_VAL", "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/imagenet_val")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--r_list", type=int, nargs="+", default=[0, 16, 20])
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = timm.create_model(args.model, pretrained=True, num_classes=0, img_size=224).eval().to(dev)
    nprefix = getattr(m, "num_prefix_tokens", 1)
    nreg = nprefix - 1
    cfg = timm.data.resolve_model_data_config(m); cfg["input_size"] = (3, 224, 224)
    tf = timm.data.create_transform(**cfg, is_training=False)
    rows = list(csv.DictReader(open(f"{VAL}/labels.csv")))[:args.n]
    xs, ys = [], []
    for r in rows:
        p = f"{VAL}/images/{r['filename']}"
        if os.path.exists(p): xs.append(tf(Image.open(p).convert("RGB"))); ys.append(int(r["label_idx"]))
    X = torch.stack(xs); Y = torch.tensor(ys); N = len(ys)
    npatch = (224 // m.patch_embed.patch_size[0]) ** 2
    print(f"[setup] {args.model} dev={dev} N={N} prefix={nprefix}(reg={nreg}) patches={npatch} blocks={len(m.blocks)}", flush=True)

    cache_dir = os.path.join(os.path.dirname(__file__), "results", "feat_cache")
    os.makedirs(cache_dir, exist_ok=True)

    def extract(r_pb, n_protect):
        # 특징 캐싱: 느린 추출을 디스크에 저장해 재실행/kNN 재계산 시 재사용
        cp = f"{cache_dir}/{args.model.split('.')[0]}_r{r_pb}_p{n_protect}_n{N}.pt"
        if os.path.exists(cp):
            return torch.load(cp)
        feats = []
        with torch.no_grad():
            for i in range(0, N, args.batch):
                xb = X[i:i+args.batch].to(dev)
                feats.append(reduced_forward(m, xb, r_pb, n_protect).float().cpu())
        F_all = torch.cat(feats); torch.save(F_all, cp)
        return F_all

    def knn(Fe, chunk=2048):
        # 청크 kNN: 50000x50000(10GB) 행렬을 한 번에 안 만들고 쿼리 청크별로 처리(메모리 안전)
        Fn = F.normalize(Fe, dim=-1); n = Fn.shape[0]; correct = 0
        for i in range(0, n, chunk):
            s = Fn[i:i+chunk] @ Fn.T                       # [b, n]
            for j in range(s.shape[0]):
                s[j, i+j] = -2.0                           # self 제외
            idx = s.topk(args.k, dim=1).indices
            correct += (torch.mode(Y[idx], dim=1).values == Y[i:i+s.shape[0]]).sum().item()
        return 100 * correct / n

    ours_hdr = f"ours(prot={nprefix})"
    print(f"\n{'r/blk':>5} {'reduce%':>7} {'tome(prot=1)':>13} {ours_hdr:>14}", flush=True)
    for r_pb in args.r_list:
        final = nprefix + max(npatch - len(m.blocks) * r_pb, 1)
        red = 100 * (1 - final / (nprefix + npatch))
        if r_pb == 0:
            acc = knn(extract(0, nprefix)); print(f"{r_pb:>5} {red:6.1f}% {'full='+format(acc,'.2f'):>13}", flush=True)
        else:
            at = knn(extract(r_pb, 1)); ao = knn(extract(r_pb, nprefix))
            print(f"{r_pb:>5} {red:6.1f}% {at:>13.2f} {ao:>14.2f}  (Δ={ao-at:+.2f})", flush=True)
    print("\n해석: 고압축(r=20)서 ours>tome 이면 register 보호가 실제 정확도 살림(정식 size-가중 ToMe 기준).", flush=True)


if __name__ == "__main__":
    main()
