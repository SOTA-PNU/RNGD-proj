#!/usr/bin/env python3
"""[②] 공식 PiToMe repo 의 '진짜' 알고리즘 코드를 그들 env(timm==0.4.12)에서 DeiT 로 직접 실행.

핵심: 공식 repo 의 데이터 파이프라인(gated imagenet-1k)을 우회하고, 그들의 알고리즘 코드
(algo/pitome/patch/deit.py 의 apply_patch)만 그대로 불러 우리 로컬 val 로 평가한다. 이렇게 하면
'공식 코드가 낸 수치'를 얻으면서도 gated 데이터·150GB 다운로드·HF 토큰이 필요 없다.

공식 API(소스 검증, 2026-07):
  from algo.pitome.patch.deit import apply_patch  # (tome 도 algo.tome.patch.deit)
  apply_patch(model)          # __class__ 스왑; model.ratio=1.0 하드코딩, 마진은 내부 [0.75→0] 스케줄
  model.ratio = <float>       # ★ 압축률(보존비율)은 패치 후 직접 설정
  logits, flop = model(x)     # ★ 패치된 forward 는 (logits, flop) 튜플 반환
전처리: 공식 main_ic build_transform 그대로 = Resize(256,bicubic)→CenterCrop(224)→ToTensor→Norm.

사용(공식 conda env 안에서, run_official_pitome.sh 가 호출):
  python official_deit_driver.py --repo <clone경로> --model deit_small_patch16_224 \
         --data_root <우리 imagenet_val> --n_val 50000 --ratio_list 0.975 0.95 0.925 0.9
"""
import argparse, os, csv, json, math, sys, warnings
warnings.filterwarnings("ignore")
import torch
from PIL import Image
from torchvision import transforms

_MEAN, _STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
OFFICIAL_TF = transforms.Compose([                              # = 공식 main_ic build_transform
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])
HERE = os.path.dirname(os.path.abspath(__file__))


class ImgFolder(torch.utils.data.Dataset):
    def __init__(self, root, tf, n):
        self.root, self.tf = root, tf
        self.rows = list(csv.DictReader(open(f"{root}/labels.csv")))[:n]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        return self.tf(Image.open(f"{self.root}/images/{r['filename']}").convert("RGB")), int(r["label_idx"])


def sim_comp(T0, L, ratio):
    """공식 스케줄(블록마다 r=floor(T*(1-ratio)))로 최종 토큰수 → 압축%. mild ratio 에선 캡 미발동=정확."""
    T = T0
    if ratio < 1.0:
        for _ in range(L):
            T -= int(math.floor(T * (1.0 - ratio)))
    return 100.0 * (1 - max(T, 1) / T0)


@torch.no_grad()
def eval_acc(m, dl, dev):
    correct = total = 0
    for xb, yb in dl:
        out = m(xb.to(dev, non_blocking=True))
        logits = out[0] if isinstance(out, (tuple, list)) else out     # 패치 모델은 (logits, flop)
        correct += (logits.argmax(1).cpu() == yb).sum().item(); total += len(yb)
    return 100.0 * correct / total


def build_patched(model_name, algo, dev, apply):
    import timm                                                        # 0.4.12
    m = timm.create_model(model_name, pretrained=True).eval()          # num_classes 기본=1000
    apply(m)                                                           # algo.<x>.patch.deit.apply_patch
    return m.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="클론된 공식 PiToMe repo 경로(algo/ 있는 루트)")
    ap.add_argument("--model", default="deit_small_patch16_224")
    ap.add_argument("--data_root", required=True, help="labels.csv+images/ 있는 우리 val 폴더")
    ap.add_argument("--n_val", type=int, default=50000)
    ap.add_argument("--ratio_list", type=float, nargs="+", default=[0.975, 0.95, 0.925, 0.9])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)                                      # 공식 algo/ import 가능하게
    try:
        from algo.pitome.patch.deit import apply_patch as pitome_patch
        from algo.tome.patch.deit import apply_patch as tome_patch
        import timm
    except Exception as e:
        print(f"[FATAL] 공식 algo import 실패({e}). timm==0.4.12 env·repo 경로 확인.", flush=True); sys.exit(2)
    assert timm.__version__.startswith("0.4"), f"공식은 timm==0.4.12 필요(현재 {timm.__version__})"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ImgFolder(args.data_root, OFFICIAL_TF, args.n_val)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    print(f"[official] {args.model} dev={dev} n_val={len(ds)} (timm {timm.__version__})", flush=True)

    # 기준 정보용 토큰수
    probe = timm.create_model(args.model, pretrained=False)
    T0 = probe.patch_embed.num_patches + 1; L = len(probe.blocks); del probe

    # baseline(ratio=1.0): pitome 패치 + ratio 1.0 = 병합 없음
    mp = build_patched(args.model, "pitome", dev, pitome_patch); mp.ratio = 1.0
    base = eval_acc(mp, dl, dev)
    print(f"\n[official] r=0 무압축 top-1 = {base:.2f}", flush=True)

    out = {"source": "official", "model": args.model, "n_val": len(ds), "baseline_r0": round(base, 2), "rows": []}
    # pitome: 같은 패치모델에서 ratio 만 바꿔 스윕
    acc = {"pitome": {}, "tome": {}}
    for ratio in sorted(args.ratio_list, reverse=True):
        mp.ratio = ratio
        acc["pitome"][ratio] = eval_acc(mp, dl, dev)
    del mp
    # tome: 새 모델 패치 후 스윕
    mt = build_patched(args.model, "tome", dev, tome_patch)
    for ratio in sorted(args.ratio_list, reverse=True):
        mt.ratio = ratio
        acc["tome"][ratio] = eval_acc(mt, dl, dev)
    del mt

    print(f"\n{'ratio':>6} {'comp%':>6} {'tome':>7} {'pitome':>7}  {'Δ(P-T)':>7}", flush=True)
    for ratio in sorted(args.ratio_list, reverse=True):
        a_t, a_p = acc["tome"][ratio], acc["pitome"][ratio]; comp = sim_comp(T0, L, ratio)
        out["rows"].append({"ratio": ratio, "comp": round(comp, 1),
                            "tome": round(a_t, 2), "pitome": round(a_p, 2), "delta_PT": round(a_p - a_t, 2)})
        print(f"{ratio:6.3f} {comp:6.1f} {a_t:7.2f} {a_p:7.2f}  {a_p-a_t:+7.2f}", flush=True)

    outdir = os.path.join(HERE, "results"); os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, f"official__{args.model}.json")
    json.dump(out, open(outp, "w"), indent=2); print(f"\n[저장] {outp}", flush=True)
    print("[다음] python compare_report.py 로 ①(우리 포팅)↔②(공식) 나란히 대조.", flush=True)


if __name__ == "__main__":
    main()
