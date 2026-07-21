#!/usr/bin/env python3
"""[일반성 검증 · train 갤러리] 레지스터 보호가 '병합기 무관' keep-rule 임을 정통 train-갤러리 kNN 에서 재검증.
val-LOO 판(robustness_50k/faithful_pitome_reg_h2h.py)과 **같은 4-arm·같은 정식 forward** 를 그대로,
평가만 train 1.28M 갤러리로 바꾼 것. 원본 val 스크립트는 수정하지 않고 forward 만 import 재사용한다.

  - 4-arm: tome(CLS만) / pitome(CLS만·공식) / pitome_reg(CLS+register) / ours(=ToMe+register).
  - 검증된 정식 forward = faithful_pitome_reg_h2h.forward_faithful (prop-attn + key-metric + attn↔MLP 병합).
  - 데이터(지연로딩 1.28M)·캐시·kNN(train·val) = compare.py 엔진 재사용(ablation_train_faithful.py 와 동일 패턴).

핵심 지표: reg@PiTo = pitome_reg − pitome > 0 이면 = 레지스터 보호가 PiToMe 병합 위에서도 이득
          → 레지스터 보호는 ToMe 전용 트릭이 아니라 병합기 무관 일반 규칙(train·정식 harness 서도).
정합: tome/ours 는 canonical_faithful_base.txt(train)와, --gallery val 실행 시 faithful_pitome_50k.log 와 일치해야 함.

※ 양 레이아웃 호환: (a) pitome_compare/ (A100 서버, 형제 robustness_50k import — 기본), (b) all_new_server/engine/
   (그 경우 faithful_pitome_reg_h2h.py·faithful_pitome_h2h.py·tome_core.py 를 같은 폴더에 복사해야 함).

★비용 단축(캐시 재사용): 기본 --cache_dir = canonical faithful 캐시(feat_cache_faithful). tome/pitome/ours 는
canonical_faithful 실행이 이미 train 1.28M 으로 뽑아둔 특징을 **그대로 히트**한다(두 엔진의 forward·merge 가
AST 수준 동일함을 검증). ⇒ 실제 신규 추출은 **pitome_reg 한 팔뿐** → 4-arm 다 돌려도 ~20h 아니라 ~5h.

사용(A100, train 데이터=pitome_compare/imagenet_train, 캐시=pitome_compare/feat_cache_faithful):
  python pitome_reg_train_faithful.py --gallery train            # 4-arm, tome/pitome/ours 캐시히트·pitome_reg만 추출
  python pitome_reg_train_faithful.py --gallery val              # 환경 대조(원본 val 수치와 일치해야)
  # 캐시가 없거나 확신 없으면 전용 캐시로 4-arm 전부 새로:  --cache_dir feat_cache_pitome_reg
"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
import torch, timm

HERE = os.path.dirname(os.path.abspath(__file__))
# 같은 폴더(compare) + 형제 robustness_50k(forward·의존) 를 import 경로에 추가(양 레이아웃 호환)
for _p in (HERE, os.path.join(HERE, "..", "robustness_50k")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import compare  # 데이터/캐시/kNN(train·val) 엔진 재사용(reduced_forward 를 아래서 주입)

# ⚠️ faithful_pitome_reg_h2h 는 모듈 최상단에서 int(sys.argv[1]) 을 실행한다(원본 그대로).
#    import 하는 순간 우리 argv(예: --gallery)를 정수로 바꾸려다 크래시하므로, import 동안만 argv 를 중립화.
_saved_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    from faithful_pitome_reg_h2h import forward_faithful  # 검증된 4-arm 정식 forward(원본 미변경)
finally:
    sys.argv = _saved_argv

DATA_ROOT = getattr(compare, "DATA_ROOT", getattr(compare, "HERE", HERE))
STRATS = ["tome", "pitome", "pitome_reg", "ours"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--gallery", choices=["val", "train"], default="train")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--r_list", type=int, nargs="+", default=[8, 12, 16, 18, 20])
    ap.add_argument("--strats", nargs="+", default=STRATS,
                    help="기본 4-arm. 비용 절감 시 'pitome pitome_reg' 만(tome/ours 는 canonical_faithful_base.txt 참조).")
    ap.add_argument("--cache_dir", default=os.path.join(HERE, "feat_cache_faithful"),
                    help="★기본=canonical faithful 캐시. tome/pitome/ours 는 canonical_faithful 이 이미 뽑아둔 것을 "
                         "그대로 재사용(forward·merge 가 AST-동일함을 검증) → 실제 신규 추출은 pitome_reg 뿐. "
                         "캐시가 없으면(다른 머신) 자동 재추출.")
    ap.add_argument("--gallery_cache", type=int, choices=[0, 1], default=1)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = timm.create_model(args.model, pretrained=True, num_classes=0, img_size=224).eval().to(dev)
    nprefix = getattr(m, "num_prefix_tokens", 1)
    npatch = m.patch_embed.num_patches; L = len(m.blocks); mtag = args.model.split(".")[0]
    tf = compare.make_tf(m)

    # 검증된 4-arm 정식 forward 를 compare 엔진의 reduced_forward 자리에 주입(CLS 특징 반환).
    def strat_forward(mm, x, r, strat, npref):
        return forward_faithful(mm, x, r, strat, npref)[0]
    compare.reduced_forward = strat_forward

    assert os.path.exists(f"{DATA_ROOT}/imagenet_val/DONE"), \
        f"val 미준비: {DATA_ROOT}/imagenet_val (prepare_data.py --split val)"
    if args.gallery == "train":
        assert os.path.exists(f"{DATA_ROOT}/imagenet_train/DONE"), \
            f"train 미준비: {DATA_ROOT}/imagenet_train (prepare_data.py --split train --per_class 1300)"

    proto = "정통 kNN(gallery=train 1.28M, query=val)" if args.gallery == "train" else "val leave-one-out kNN"
    print(f"[setup·faithful·{args.gallery}·일반성] {args.model} dev={dev} prefix={nprefix} "
          f"patches={npatch} blocks={L} data={DATA_ROOT}", flush=True)
    print(f"[proto] {proto}, k={args.k} · arms={args.strats} · cache={args.cache_dir}", flush=True)

    hp = ("pitome" in args.strats) and ("pitome_reg" in args.strats)
    ht = ("tome" in args.strats) and ("ours" in args.strats)
    hdr = f"{'r':>3} {'comp%':>6} " + " ".join(f"{s:>10}" for s in args.strats)
    if hp: hdr += f" {'reg@PiTo':>9}"
    if ht: hdr += f" {'reg@ToMe':>9}"
    print(hdr, flush=True)

    for r in args.r_list:
        accs = {}
        for st in args.strats:
            Qf, Qy = compare.extract_split(m, "val", r, st, nprefix, tf, args.batch, args.workers, dev, args.cache_dir, mtag)
            if args.gallery == "train":
                Gf, Gy = compare.extract_split(m, "train", r, st, nprefix, tf, args.batch, args.workers, dev,
                                               args.cache_dir, mtag, save=bool(args.gallery_cache))
                accs[st] = compare.knn_gallery(Gf, Gy, Qf, Qy, args.k, dev)
            else:
                accs[st] = compare.knn_loo(Qf, Qy, args.k, dev)
        final = nprefix + max(npatch - L * r, 1); comp = 100 * (1 - final / (nprefix + npatch))
        row = f"{r:>3} {comp:6.1f} " + " ".join(f"{accs[s]:10.2f}" for s in args.strats)
        if hp: row += f" {accs['pitome_reg'] - accs['pitome']:+9.2f}"
        if ht: row += f" {accs['ours'] - accs['tome']:+9.2f}"
        print(row, flush=True)

    print("\n해석: 'reg@PiTo'(PiToMe+reg − PiToMe) > 0 이면 = 레지스터 보호가 PiToMe 병합 위에서도 이득 "
          "→ 병합기 무관 일반 keep-rule 을 train·정식 harness 서 재확인. "
          "tome/ours 는 canonical_faithful_base.txt(train) 와 일치해야 함(엔진 정합).", flush=True)


if __name__ == "__main__":
    main()
