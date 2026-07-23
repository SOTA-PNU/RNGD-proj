#!/usr/bin/env python3
"""[faithful · train 갤러리] keep-prior ablation 을 정통 train-갤러리 kNN 에서 재실행.
val-LOO(eval_ablation_faithful.py)와 **같은 전략·같은 정식 forward** 를 그대로, 평가만 train 갤러리로.
  - 정식 forward = reduced_forward_strat_faithful (prop-attn + key-metric + attn↔MLP 병합), eval_ablation_faithful 에서 import(검증됨).
  - 데이터/캐시/kNN(train·val) = compare.py 엔진 재사용.
전략(모두 CLS 공통 보호 + 추가 #reg): tome / ours(register) / random / energy / highnorm.
※ 이 파일은 두 레이아웃 모두에서 동작: (a) register_token_reduction/pitome_compare/ (A100 서버, 형제폴더 import),
   (b) all_new_server/engine/ (같은 폴더에 의존파일 복사됨). compare.DATA_ROOT 없으면 compare.HERE 로 폴백.
사용(A100, train 데이터는 pitome_compare/imagenet_train):
  python ablation_train_faithful.py --model vit_base_patch14_reg4_dinov2.lvd142m --gallery train --r_list 8 12 16 18 20
"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
import torch, timm
HERE = os.path.dirname(os.path.abspath(__file__))
# 같은 폴더 + 형제 폴더(robustness_50k, ablation) 를 import 경로에 추가(양 레이아웃 호환)
for _p in (HERE, os.path.join(HERE, "..", "robustness_50k"), os.path.join(HERE, "..", "ablation")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import compare                                                     # 데이터/캐시/kNN(train·val) 엔진 재사용
from eval_ablation_faithful import reduced_forward_strat_faithful  # 검증된 정식 전략 forward 재사용

DATA_ROOT = getattr(compare, "DATA_ROOT", getattr(compare, "HERE", HERE))
STRATS = ["tome", "ours", "random", "energy", "highnorm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--gallery", choices=["val", "train"], default="train")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--r_list", type=int, nargs="+", default=[8, 12, 16, 18, 20])
    ap.add_argument("--strats", nargs="+", default=STRATS)
    ap.add_argument("--cache_dir", default=os.path.join(HERE, "feat_cache_ablation"))
    ap.add_argument("--gallery_cache", type=int, choices=[0, 1], default=1,
                    help="1=train 갤러리 특징 디스크 캐시(재개 가능, 전략×r 당 수 GB) / 0=캐시 안 함(디스크 절약, 재개 시 재추출)")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = timm.create_model(args.model, pretrained=True, num_classes=0, img_size=224).eval().to(dev)
    nprefix = getattr(m, "num_prefix_tokens", 1); nreg = max(nprefix - 1, 4)
    npatch = m.patch_embed.num_patches; L = len(m.blocks); mtag = args.model.split(".")[0]
    tf = compare.make_tf(m)
    gen = torch.Generator().manual_seed(20260705)

    # 검증된 정식 전략 forward 를 compare 엔진에 주입(전략 문자열로 분기, CLS 특징 반환)
    def strat_forward(mm, x, r, strat, npref):
        return reduced_forward_strat_faithful(mm, x, r, strat, npref, nreg, gen)[0]
    compare.reduced_forward = strat_forward

    assert os.path.exists(f"{DATA_ROOT}/imagenet_val/DONE"), f"val 미준비: {DATA_ROOT}/imagenet_val (prepare_data.py --split val)"
    if args.gallery == "train":
        assert os.path.exists(f"{DATA_ROOT}/imagenet_train/DONE"), f"train 미준비: {DATA_ROOT}/imagenet_train (prepare_data.py --split train --per_class 1300)"
    proto = "정통 kNN(gallery=train, query=val)" if args.gallery == "train" else "val leave-one-out kNN"
    print(f"[setup·faithful·{args.gallery}] {args.model} dev={dev} prefix={nprefix}(reg={nreg}) patches={npatch} blocks={L} data={DATA_ROOT}", flush=True)
    print(f"[proto] {proto}, k={args.k}", flush=True)
    print(f"{'r':>3} {'comp%':>6} " + " ".join(f"{s:>9}" for s in args.strats), flush=True)

    for r in args.r_list:
        accs = {}
        for st in args.strats:
            Qf, Qy = compare.extract_split(m, "val", r, st, nprefix, tf, args.batch, args.workers, dev, args.cache_dir, mtag)
            if args.gallery == "train":
                Gf, Gy = compare.extract_split(m, "train", r, st, nprefix, tf, args.batch, args.workers, dev, args.cache_dir, mtag, save=bool(args.gallery_cache))
                accs[st] = compare.knn_gallery(Gf, Gy, Qf, Qy, args.k, dev)
            else:
                accs[st] = compare.knn_loo(Qf, Qy, args.k, dev)
        final = nprefix + max(npatch - L * r, 1); comp = 100 * (1 - final / (nprefix + npatch))
        print(f"{r:>3} {comp:6.1f} " + " ".join(f"{accs[s]:9.2f}" for s in args.strats), flush=True)
    print("\n해석: train 갤러리·정식 harness 서도 ours 가 random/energy/highnorm 을 이기면 "
          "= register keep-prior 특수성이 정통 프로토콜에서도 유지됨.", flush=True)


if __name__ == "__main__":
    main()
