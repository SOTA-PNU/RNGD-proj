#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_extra.py — DINOv3(-S+/-B)·ViT-5 에서 register-보호 faithful 토큰축소 평가.

DINOv2 실험(engine/compare.py)의 kNN·데이터 파이프라인을 **그대로 재사용**하되, forward 만
rope-aware 어댑터(models_extra.py)로 교체합니다. 프로토콜은 논문 헤드라인과 동일:
  - gallery=train(ImageNet 1.28M), query=val 50k, 표준 kNN top-1 (승급 잣대)
  - 또는 gallery=val (val leave-one-out, 일관성 확인)
faithful = proportional attention(log size 편향) + attention-key metric + attention↔MLP 사이 병합.

★ 전략(rope 모델에서 model-exact 한 것만 씀 — README '설계 결정' 참조):
  - ours  : CLS + register 전부 보호, patch 만 병합.  r=0 == 공식 forward (selfcheck 로 검증).
  - noreg : register 를 시퀀스에서 제거(같은 가중치의 '레지스터 없는 모델'), patch 만 병합.
            rope-안전한 무보호 baseline. Δ = ours − noreg = 압축 하에서 register 의 기여.
  - (옵션) --regsweep : k=0..4 register 보호 스윕(논문 reg-count 실험의 타 아키텍처 재현).

사용:
  python run_extra.py --model dinov3_base  --gallery train --r_list 8 12 16 18 20
  python run_extra.py --model dinov3_splus --gallery train --r_list 8 12 16 18 20
  python run_extra.py --model vit5_base --vit5_repo <clone> --vit5_ckpt <pth> --gallery train --r_list 8 12 16 18 20
"""
import argparse, os, sys, time, csv, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "engine"))
import compare as C                      # ImgFolder, knn_gallery, knn_loo, make_tf 재사용
import models_extra as MX

DATA_ROOT = os.environ.get("DATA_ROOT", HERE)
C.DATA_ROOT = DATA_ROOT                   # compare 의 데이터 루트도 맞춤


@torch.no_grad()
def extract(model, fwd, split, r, n_reg_keep, tf, batch, workers, dev, cache_dir, tag, save=True):
    os.makedirs(cache_dir, exist_ok=True)
    ck = f"{cache_dir}/{tag}__k{n_reg_keep}__r{r}__{split}.pt"
    if os.path.exists(ck):
        d = torch.load(ck); return d["feat"], d["label"]
    ds = C.ImgFolder(f"{DATA_ROOT}/imagenet_{split}", tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True)
    feats, labs, done, N, t0 = [], [], 0, len(ds), time.time()
    for xb, yb in dl:
        f = fwd(model, xb.to(dev, non_blocking=True), r, n_reg_keep=n_reg_keep)
        feats.append(f.half().cpu()); labs.append(yb); done += len(yb)
        if done % 51200 < batch:
            print(f"    [k{n_reg_keep} r{r} {split}] {done}/{N} ({done/max(time.time()-t0,1e-9):.0f} img/s)", flush=True)
    feat, label = torch.cat(feats), torch.cat(labs)
    if save: torch.save({"feat": feat, "label": label}, ck)
    return feat, label


def acc_at(model, fwd, r, n_reg_keep, tf, args, dev, tag):
    Qf, Qy = extract(model, fwd, "val", r, n_reg_keep, tf, args.batch, args.workers, dev, args.cache_dir, tag)
    if args.gallery == "train":
        Gf, Gy = extract(model, fwd, "train", r, n_reg_keep, tf, args.batch, args.workers, dev,
                         args.cache_dir, tag, save=bool(args.gallery_cache))
        return C.knn_gallery(Gf, Gy, Qf, Qy, args.k, dev)
    return C.knn_loo(Qf, Qy, args.k, dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["dinov3_splus", "dinov3_base", "vit5_base"])
    ap.add_argument("--gallery", choices=["val", "train"], default="train")
    ap.add_argument("--r_list", type=int, nargs="+", default=[8, 12, 16, 18, 20])
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--cache_dir", default=os.path.join(DATA_ROOT, "feat_cache_extra"))
    ap.add_argument("--gallery_cache", type=int, choices=[0, 1], default=1)
    ap.add_argument("--regsweep", action="store_true", help="k=0..4 register 보호 스윕(고정 r=max)")
    ap.add_argument("--vit5_repo", default=os.environ.get("VIT5_REPO"))
    ap.add_argument("--vit5_ckpt", default=os.environ.get("VIT5_CKPT"))
    ap.add_argument("--dinov3_hub", action="store_true", help="공식 facebookresearch/dinov3(게이트) 사용")
    ap.add_argument("--dinov3_weights", default=os.environ.get("DINOV3_WEIGHTS"))
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kw = {}
    if args.model.startswith("dinov3"):
        kw = dict(hub=args.dinov3_hub, weights=args.dinov3_weights) if args.dinov3_hub else {}
    if args.model == "vit5_base":
        assert args.vit5_repo and args.vit5_ckpt, "vit5 는 --vit5_repo(공식 clone) 와 --vit5_ckpt(공식 .pth) 필요"
        kw = dict(ckpt=args.vit5_ckpt, repo_dir=args.vit5_repo)
    model, nprefix, fwd = MX.get_model_and_forward(args.model, device=dev, img_size=args.img_size, **kw)
    nreg = nprefix - 1
    tf = C.make_tf(model)
    tag = args.model
    assert os.path.exists(f"{DATA_ROOT}/imagenet_val/DONE"), "val 미준비: python prepare_data.py --split val"
    if args.gallery == "train":
        assert os.path.exists(f"{DATA_ROOT}/imagenet_train/DONE"), "train 미준비: python prepare_data.py --split train --per_class 1300"
    print(f"[setup] {args.model} prefix={nprefix}(CLS+{nreg}reg) blocks={len(model.blocks)} gallery={args.gallery}", flush=True)

    if args.regsweep:
        r = max(args.r_list)
        print(f"\n[reg-count sweep] r={r} 고정, k=protect한 register 수\n{'k':>2} {'acc':>8}", flush=True)
        for kk in range(0, nreg + 1):
            a = acc_at(model, fwd, r, kk, tf, args, dev, tag)
            print(f"{kk:>2} {a:8.2f}", flush=True)
        print("해석: k 증가에 acc 증가 → register 보호가 압축 하에서 도움(타 아키텍처 재현).", flush=True)
        return

    print(f"\n{'r':>3} {'comp%':>6} {'ours':>8} {'noreg':>8}   Δ(ours-noreg)", flush=True)
    rows = [0] + list(args.r_list) if 0 not in args.r_list else list(args.r_list)
    npatch = None
    for r in rows:
        ours = acc_at(model, fwd, r, nreg, tf, args, dev, tag)     # k=nreg = 전 register 보호
        if r == 0:
            print(f"{0:>3} {0.0:6.1f} {ours:8.2f}  (무압축; ours r=0 == 공식 forward, selfcheck 로 검증)", flush=True)
            continue
        noreg = acc_at(model, fwd, r, 0, tf, args, dev, tag)       # k=0 = register 제거
        # comp%: 대략치(정확 토큰수는 로그의 img/s 와 별개). L*r 만큼 patch 감소.
        L = len(model.blocks)
        print(f"{r:>3} {'':6} {ours:8.2f} {noreg:8.2f}   {ours-noreg:+.2f}", flush=True)
    print("\n해석: 같은 r 에서 ours > noreg 이면 '레지스터 보호가 압축 하 정확도를 지킨다'가"
          " DINOv2 를 넘어 rope 기반 register 모델에서도 성립. noreg=같은 가중치의 레지스터 없는 모델(rope 안전).", flush=True)


if __name__ == "__main__":
    main()
