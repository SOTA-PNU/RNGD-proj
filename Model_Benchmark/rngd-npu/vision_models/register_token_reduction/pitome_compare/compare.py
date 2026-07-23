#!/usr/bin/env python3
"""PiToMe(공식 selection 그대로) vs ToMe vs Ours(register 보호) — 같은 예산 정확도 + throughput.
평가 지표=표준 kNN top-1. 기본 프로토콜(--gallery val)=val leave-one-out(갤러리=쿼리=val 5만, 자기 제외, k=20),
논문 전 실험과 동일. 승급 옵션(--gallery train)=정통 train 갤러리(DINOv2 공식 82.0 재현 잣대).
세 방법 모두 '블록 뒤·같은 metric(post-block x)·블록당 같은 제거량 r'로 통제 — 차이는 '무엇을 지키고 합칠지'뿐:
  - tome   : CLS 1개만 보호 + size-가중 bipartite soft matching(BSM)로 r쌍 병합
  - pitome : PiToMe 공식 selection 그대로 — 에너지 E=elu(cos−m).mean, m=0.75−0.75·l/L(공식 deit 패처값),
             고에너지 2r=병합·나머지=보호(저에너지), a→best-b 병합, size-가중. **CLS만 보호(register 개념 없음).**
  - ours   : CLS + register 전부 보호 + 동일 BSM
지표: kNN top-1(정확도, --mode acc) + throughput im/s(--mode tput). 핵심 = '같은 압축률서 ours가 더 정확'.
사용: python compare.py --mode acc  --r_list 8 12 16 18 20           # (train/val 준비 후)
      python compare.py --mode tput --batch 128 --r_list 0 8 12 16 18 20
"""
import argparse, os, csv, time, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn.functional as F, timm
from PIL import Image

HERE = os.path.dirname(__file__)


@torch.no_grad()
def merge_step(x, size, r, n_protect):
    """size-가중 bipartite soft matching (ToMe). 앞 n_protect개 보호, 비보호 중 r쌍 병합. (tome/ours 공용)"""
    B, T, C = x.shape
    r = min(r, max((T - n_protect) // 2, 0))
    if r <= 0: return x, size
    xp, xr = x[:, :n_protect], x[:, n_protect:]
    sp, sr = size[:, :n_protect], size[:, n_protect:]
    a, b = xr[:, ::2, :], xr[:, 1::2, :]; sa, sb = sr[:, ::2, :], sr[:, 1::2, :]
    an, bn = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    scores = an @ bn.transpose(-1, -2)
    node_max, node_idx = scores.max(dim=-1)
    edge = node_max.argsort(dim=-1, descending=True)[..., None]
    unm_idx, src_idx = edge[:, r:, :], edge[:, :r, :]
    dst_idx = node_idx[..., None].gather(1, src_idx)
    unm = a.gather(1, unm_idx.expand(-1, -1, C)); unm_s = sa.gather(1, unm_idx.expand(-1, -1, 1))
    src = a.gather(1, src_idx.expand(-1, -1, C)); src_s = sa.gather(1, src_idx.expand(-1, -1, 1))
    b_acc = (b * sb).scatter_add(1, dst_idx.expand(-1, -1, C), src * src_s)
    s_acc = sb.scatter_add(1, dst_idx.expand(-1, -1, 1), src_s)
    return torch.cat([xp, unm, b_acc / s_acc], dim=1), torch.cat([sp, unm_s, s_acc], dim=1)


@torch.no_grad()
def pitome_step(x, size, r, margin):
    """PiToMe 공식 selection 그대로(algo/pitome/merge.py `pitome_vision`+`pitome`, merge_wavg). CLS(0번)만 보호.
    에너지 E_i=mean_j elu(cos(x_i,x_j)−margin) → 고에너지 2r개 병합대상, 나머지 보호. a=병합[::2]→best-b[1::2]."""
    B, T, C = x.shape
    xc, xr = x[:, :1], x[:, 1:]; sc, sr = size[:, :1], size[:, 1:]
    P = xr.shape[1]
    r = min(r, P // 2)
    if r <= 0: return x, size
    m = F.normalize(xr, dim=-1)
    sim = m @ m.transpose(-1, -2)                              # cos
    energy = F.elu(sim - margin, alpha=1.0).mean(dim=-1)       # 공식 Eq.4(코드): elu(cos−m).mean
    idx = energy.argsort(dim=-1, descending=True)              # 고에너지 우선
    merge_idx, prot_idx = idx[:, :2 * r], idx[:, 2 * r:]
    a_idx, b_idx = merge_idx[:, ::2], merge_idx[:, 1::2]       # 공식: 정렬된 병합집합을 짝/홀로 쪼갬
    sab = sim.gather(1, a_idx[..., None].expand(-1, -1, P)).gather(2, b_idx[:, None, :].expand(-1, r, -1))
    dst_local = sab.max(dim=-1).indices                        # a마다 가장 닮은 b
    xrw = xr * sr                                              # merge_wavg: size-가중 합→나눔
    a_w = xrw.gather(1, a_idx[..., None].expand(-1, -1, C)); sa = sr.gather(1, a_idx[..., None])
    b_w = xrw.gather(1, b_idx[..., None].expand(-1, -1, C)); sb = sr.gather(1, b_idx[..., None])
    b_acc = b_w.scatter_add(1, dst_local[..., None].expand(-1, -1, C), a_w)
    s_acc = sb.scatter_add(1, dst_local[..., None], sa)
    b_out = b_acc / s_acc
    prot = xr.gather(1, prot_idx[..., None].expand(-1, -1, C)); sprot = sr.gather(1, prot_idx[..., None])
    return torch.cat([xc, prot, b_out], dim=1), torch.cat([sc, sprot, s_acc], dim=1)


@torch.no_grad()
def reduced_forward(m, x, r, strat, nprefix):
    t = m._pos_embed(m.patch_embed(x)); B, T, C = t.shape
    size = torch.ones(B, T, 1, device=t.device, dtype=t.dtype)
    L = len(m.blocks)
    for li, blk in enumerate(m.blocks):
        t = blk(t)
        if strat == "tome":
            t, size = merge_step(t, size, r, 1)
        elif strat == "ours":
            t, size = merge_step(t, size, r, nprefix)
        elif strat == "pitome":
            margin = 0.75 - 0.75 * (li / max(L, 1))            # 공식 deit 패처 스케줄
            t, size = pitome_step(t, size, r, margin)
    return m.norm(t)[:, 0]


class ImgFolder(torch.utils.data.Dataset):
    """지연 로딩(train 128만 장을 RAM에 안 올림). labels.csv(filename,label_idx,...) 기반."""
    def __init__(self, root, tf):
        self.root, self.tf = root, tf
        self.rows = list(csv.DictReader(open(f"{root}/labels.csv")))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(f"{self.root}/images/{r['filename']}").convert("RGB")
        return self.tf(img), int(r["label_idx"])


@torch.no_grad()
def extract_split(m, split, r, strat, nprefix, tf, batch, workers, dev, cache_dir, model_name, save=True):
    """압축설정(strat,r)로 split 전체의 CLS 특징 추출. 캐시 있으면 재사용(수시간 실행 재개용).
    save=False면 디스크에 안 씀(대용량 train 갤러리 캐시 32GB+ 회피용; 단 재개시 재추출)."""
    os.makedirs(cache_dir, exist_ok=True)
    ck = f"{cache_dir}/{model_name}__{strat}__r{r}__{split}.pt"
    if os.path.exists(ck):
        d = torch.load(ck); return d["feat"], d["label"]
    ds = ImgFolder(f"{HERE}/imagenet_{split}", tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True)
    feats, labs, done, N = [], [], 0, len(ds)
    t0 = time.time()
    for xb, yb in dl:
        feats.append(reduced_forward(m, xb.to(dev, non_blocking=True), r, strat, nprefix).half().cpu())
        labs.append(yb); done += len(yb)
        if done % 51200 < batch: print(f"    [{strat} r={r} {split}] {done}/{N}  ({done/max(time.time()-t0,1e-9):.0f} img/s)", flush=True)
    dt = time.time() - t0
    print(f"[timing] extract {split} strat={strat} r={r}: {dt:.1f}s for {N} imgs = {N/max(dt,1e-9):.0f} img/s", flush=True)
    feat = torch.cat(feats); label = torch.cat(labs)
    if save:
        torch.save({"feat": feat, "label": label}, ck)
    return feat, label


@torch.no_grad()
def knn_gallery(Gf, Gy, Qf, Qy, k, dev, chunk=256):
    """정통 kNN: gallery=train, query=val. 각 query를 train 최근접 k개 다수결로 분류."""
    G = F.normalize(Gf.to(dev).float(), dim=-1).half(); Gy = Gy.to(dev)
    Qn = F.normalize(Qf.float(), dim=-1).half(); correct = 0
    for i in range(0, len(Qn), chunk):
        s = Qn[i:i+chunk].to(dev) @ G.T                    # [chunk, Ntrain]
        idx = s.topk(k, dim=1).indices
        pred = torch.mode(Gy[idx], dim=1).values
        correct += (pred == Qy[i:i+s.shape[0]].to(dev)).sum().item()
    return 100 * correct / len(Qn)


@torch.no_grad()
def knn_loo(Qf, Qy, k, dev, chunk=256):
    """val leave-one-out k-NN: 갤러리=쿼리=val, 각 이미지는 자기 자신만 빼고 이웃을 찾음(라벨 누수 없음).
    표준 kNN top-1 지표이며, 논문 전 실험과 동일 프로토콜. 경쟁 방법도 여기서 우리가 직접 재측정(상대 Δ가 기여)."""
    Fn = F.normalize(Qf.to(dev).float(), dim=-1).half(); Y = Qy.to(dev); n = len(Fn); correct = 0
    for i in range(0, n, chunk):
        s = Fn[i:i+chunk] @ Fn.T                           # [chunk, N]
        for j in range(s.shape[0]): s[j, i+j] = -2.0       # self 제외
        idx = s.topk(k, dim=1).indices
        correct += (torch.mode(Y[idx], dim=1).values == Y[i:i+s.shape[0]]).sum().item()
    return 100 * correct / n


def make_tf(m):
    cfg = timm.data.resolve_model_data_config(m); cfg["input_size"] = (3, 224, 224)
    return timm.data.create_transform(**cfg, is_training=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["acc", "tput"], default="acc")
    ap.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--r_list", type=int, nargs="+", default=[8, 12, 16, 18, 20])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--gallery", choices=["val", "train"], default="val",
                    help="val=val leave-one-out(갤러리=쿼리=val, 논문 전 실험과 동일, 기본) / train=정통 train 갤러리(승급용)")
    ap.add_argument("--cache_dir", default=os.path.join(HERE, "feat_cache"))
    ap.add_argument("--gallery_cache", type=int, choices=[0, 1], default=1,
                    help="1=train 갤러리 특징을 디스크 캐시(재개 가능, 단 설정당 ~2GB) / 0=캐시 안 함(디스크 절약, 재개시 재추출)")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = timm.create_model(args.model, pretrained=True, num_classes=0, img_size=224).eval().to(dev)
    nprefix = getattr(m, "num_prefix_tokens", 1); npatch = m.patch_embed.num_patches; L = len(m.blocks)
    strategies = ["tome", "pitome", "ours"]
    mtag = args.model.split(".")[0]
    print(f"[setup] {args.model} dev={dev} prefix={nprefix} patches={npatch} blocks={L} mode={args.mode}", flush=True)

    if args.mode == "acc":
        tf = make_tf(m)
        assert os.path.exists(f"{HERE}/imagenet_val/DONE"), "val 미준비: python prepare_data.py --split val"
        if args.gallery == "train":
            assert os.path.exists(f"{HERE}/imagenet_train/DONE"), "train(gallery) 미준비: python prepare_data.py --split train --per_class 1300"
            print(f"[proto] 정통 kNN: gallery=ImageNet train, query=val, k={args.k}", flush=True)
        else:
            print(f"[proto] val leave-one-out k-NN: 갤러리=쿼리=val 5만(자기 제외), k={args.k} (논문 전 실험과 동일; 경쟁 방법도 여기서 우리가 직접 재측정)", flush=True)
        print(f"\n{'r':>3} {'comp%':>6} " + " ".join(f"{s:>8}" for s in strategies) + "   Δ(ours-pitome)", flush=True)
        rows = [0] + list(args.r_list) if 0 not in args.r_list else list(args.r_list)
        for r in rows:
            accs = {}
            for st in (["ours"] if r == 0 else strategies):   # r=0은 세 방법 동일 → 1회만
                Qf, Qy = extract_split(m, "val", r, st, nprefix, tf, args.batch, args.workers, dev, args.cache_dir, mtag)
                if args.gallery == "train":
                    Gf, Gy = extract_split(m, "train", r, st, nprefix, tf, args.batch, args.workers, dev, args.cache_dir, mtag, save=bool(args.gallery_cache))
                    tk = time.time(); accs[st] = knn_gallery(Gf, Gy, Qf, Qy, args.k, dev)
                    print(f"[timing] kNN(train gallery {Gf.shape[0]}) strat={st} r={r}: {time.time()-tk:.1f}s", flush=True)
                else:
                    accs[st] = knn_loo(Qf, Qy, args.k, dev)
            if r == 0:
                ref = "DINOv2 공식 kNN≈82.0 대조" if args.gallery == "train" else "이게 곧 'DINOv2 kNN을 50k val(leave-one-out)로 재측정한' baseline"
                print(f"{0:>3} {0.0:6.1f} " + f"{accs['ours']:8.2f} (무압축 baseline, {ref})", flush=True); continue
            final = nprefix + max(npatch - L * r, 1); comp = 100 * (1 - final / (nprefix + npatch))
            print(f"{r:>3} {comp:6.1f} " + " ".join(f"{accs[s]:8.2f}" for s in strategies) + f"   {accs['ours']-accs['pitome']:+.2f}", flush=True)
        print("\n해석: 같은 압축률(comp%)서 ours > pitome 이면 'register 보호 > PiToMe 에너지 selection' 입증(핵심 정확도 우위).", flush=True)
        print("      절대값이 아니라 세 방법의 Δ가 기여 — 모두 같은 프로토콜로 우리가 직접 측정(남의 논문 수치 미인용).", flush=True)

    else:  # throughput (합성 배치 — 데이터셋 무관, 순수 연산속도. 표준 방식)
        x = torch.randn(args.batch, 3, 224, 224, device=dev)
        print(f"\n{'r':>3} {'comp%':>6} " + " ".join(f"{s+'(im/s)':>13}" for s in strategies), flush=True)
        for r in args.r_list:
            row = {}
            for st in strategies:
                for _ in range(5): reduced_forward(m, x, r, st, nprefix)   # warmup
                if dev.type == "cuda": torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(args.iters): reduced_forward(m, x, r, st, nprefix)
                if dev.type == "cuda": torch.cuda.synchronize()
                row[st] = args.batch * args.iters / (time.time() - t0)
            final = nprefix + max(npatch - L * r, 1); comp = 100 * (1 - final / (nprefix + npatch))
            print(f"{r:>3} {comp:6.1f} " + " ".join(f"{row[s]:13.0f}" for s in strategies), flush=True)
        print("\n해석: 세 방법 im/s가 비슷하면 → 'ours의 정확도 이득이 속도를 안 깎는다(공짜)'. (r=0=무압축 기준선)", flush=True)


if __name__ == "__main__":
    main()
