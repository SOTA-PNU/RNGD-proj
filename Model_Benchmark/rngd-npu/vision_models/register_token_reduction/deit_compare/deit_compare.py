#!/usr/bin/env python3
"""DeiT 교차검증 — 우리 PiToMe/ToMe 포팅이 공식 PiToMe와 일치하는지 확인.

배경: 논문 헤드라인(DINOv2-reg) PiToMe 비교는 pitome_compare/compare.py 의 pitome_step(공식
algo/pitome/merge.py 소스대로 포팅)을 쓴다. 공식 repo 는 timm==0.4.12 핀이라 DINOv2-reg(timm>=1.0
필요)를 못 올리므로, "우리 포팅이 진짜 공식과 같은가"를 공통 모델 DeiT 에서 확인한다.

이 스크립트: compare.py 의 merge_step(ToMe)·pitome_step(PiToMe) 를 **그대로** 가져와, 공식과 동일한
'per-block 보존비율(ratio) 스케줄'로 DeiT 에 적용하고 **ImageNet val top-1 분류 정확도**(공식과 같은
지표)를 잰다. r=0(무압축) 정확도가 DeiT 공식치(S=79.8, B=81.98, T=72.2)와 맞으면 파이프라인·라벨순서
정상 확인. 그다음 압축별 PiToMe/ToMe 곡선을 공식 논문 공개 수치(아래 REF)와 대조한다.

공식 정합: --ratio = 보존비율(클수록 약한 압축). 각 블록에서 r=floor(T*(1-ratio)) 제거(공식 merge.py
`r=math.floor(T - T*ratio)`, 모든 블록 균등 스케줄). CLS 1개만 보호(DeiT 는 register 없음 → ours=tome).

사용:
  python deit_compare.py --model deit_small_patch16_224 --n_val 50000 --ratio_list 0.975 0.95 0.925 0.9
  (빠른 시험: --n_val 5000)
"""
import argparse, os, csv, math, json, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn.functional as F, timm
from PIL import Image
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))

# 공식 PiToMe main_ic 와 동일한 eval 전처리(Resize256 bicubic → CenterCrop224 → Norm).
# ①(우리 포팅)·②(공식 repo) 를 같은 잣대로 맞추려고 timm 기본(crop 0.9=248) 대신 공식(256) 을 씀.
_MEAN, _STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
OFFICIAL_TF = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# 공식 PiToMe 논문(arXiv:2405.16148, Table 5/6, off-the-shelf=재학습없음 / fine-tuned) 공개 수치.
# ※ 논문에 DeiT-B 는 없음 → B 는 참조 없음(우리 곡선 단독). S/T 만 대조 가능.
REF = {
    "deit_small_patch16_224": {"baseline": 79.8, "note": "@~37% FLOPs↓(4.6→2.9G): ToMe 77.7/79.4, PiToMe 79.1/79.8 (off-the-shelf/ft)"},
    "deit_tiny_patch16_224":  {"baseline": 72.3, "note": "@~34% FLOPs↓(1.2→0.79G): ToMe 68.9/70.0, PiToMe 70.8/71.6 (off-the-shelf/ft)"},
    "deit_base_patch16_224":  {"baseline": 81.98, "note": "논문에 DeiT-B 없음 → 공개 참조치 없음(우리 곡선 단독; r=0=81.98 로 파이프라인만 검증)"},
}


@torch.no_grad()
def merge_step(x, size, r, n_protect):
    """size-가중 bipartite soft matching (ToMe). 앞 n_protect개 보호, 비보호 중 r쌍 병합.
    ── compare.py 와 동일(검증 대상 포팅 그대로)."""
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
    """PiToMe 공식 selection 그대로(algo/pitome/merge.py `pitome_vision`+merge_wavg). CLS(0번)만 보호.
    ── compare.py 와 동일(검증 대상 포팅 그대로)."""
    B, T, C = x.shape
    xc, xr = x[:, :1], x[:, 1:]; sc, sr = size[:, :1], size[:, 1:]
    P = xr.shape[1]
    r = min(r, P // 2)
    if r <= 0: return x, size
    m = F.normalize(xr, dim=-1)
    sim = m @ m.transpose(-1, -2)
    energy = F.elu(sim - margin, alpha=1.0).mean(dim=-1)
    idx = energy.argsort(dim=-1, descending=True)
    merge_idx, prot_idx = idx[:, :2 * r], idx[:, 2 * r:]
    a_idx, b_idx = merge_idx[:, ::2], merge_idx[:, 1::2]
    sab = sim.gather(1, a_idx[..., None].expand(-1, -1, P)).gather(2, b_idx[:, None, :].expand(-1, r, -1))
    dst_local = sab.max(dim=-1).indices
    xrw = xr * sr
    a_w = xrw.gather(1, a_idx[..., None].expand(-1, -1, C)); sa = sr.gather(1, a_idx[..., None])
    b_w = xrw.gather(1, b_idx[..., None].expand(-1, -1, C)); sb = sr.gather(1, b_idx[..., None])
    b_acc = b_w.scatter_add(1, dst_local[..., None].expand(-1, -1, C), a_w)
    s_acc = sb.scatter_add(1, dst_local[..., None], sa)
    b_out = b_acc / s_acc
    prot = xr.gather(1, prot_idx[..., None].expand(-1, -1, C)); sprot = sr.gather(1, prot_idx[..., None])
    return torch.cat([xc, prot, b_out], dim=1), torch.cat([sc, sprot, s_acc], dim=1)


@torch.no_grad()
def reduced_forward_ratio(m, x, ratio, strat, nprefix):
    """공식 스케줄로 압축한 뒤 정규화한 토큰 반환. 각 블록에서 r=floor(T_cur*(1-ratio)) 제거(공식 균등 스케줄).
    strat: 'tome'(CLS만 보호 BSM) | 'pitome'(공식 에너지 selection) | 'none'(무압축)."""
    t = m._pos_embed(m.patch_embed(x))
    B, T, C = t.shape
    size = torch.ones(B, T, 1, device=t.device, dtype=t.dtype)
    L = len(m.blocks)
    for li, blk in enumerate(m.blocks):
        t = blk(t)
        r = 0 if ratio >= 1.0 else int(math.floor(t.shape[1] * (1.0 - ratio)))  # 공식: r=floor(T - T*ratio)
        if r > 0:
            if strat == "tome":
                t, size = merge_step(t, size, r, nprefix)
            elif strat == "pitome":
                margin = 0.75 - 0.75 * (li / max(L, 1))          # 공식 deit 패처 마진 스케줄(포팅값)
                t, size = pitome_step(t, size, r, margin)
    return m.norm(t)


class ImgFolder(torch.utils.data.Dataset):
    def __init__(self, root, tf, n):
        self.root, self.tf = root, tf
        self.rows = list(csv.DictReader(open(f"{root}/labels.csv")))[:n]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(f"{self.root}/images/{r['filename']}").convert("RGB")
        return self.tf(img), int(r["label_idx"])


@torch.no_grad()
def eval_acc(m, dl, ratio, strat, nprefix, dev):
    """ImageNet val top-1 분류 정확도(공식과 같은 지표). 압축 토큰 → norm → forward_head(=pool+head)."""
    correct = total = 0
    final_T = None
    for xb, yb in dl:
        t = reduced_forward_ratio(m, xb.to(dev, non_blocking=True), ratio, strat, nprefix)
        final_T = t.shape[1]
        logits = m.forward_head(t)                                # pool('token'→CLS)+fc_norm(Identity)+head
        correct += (logits.argmax(1).cpu() == yb).sum().item(); total += len(yb)
    return 100.0 * correct / total, final_T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deit_small_patch16_224",
                    help="deit_small_patch16_224(권장, 공개 참조있음) | deit_tiny_patch16_224 | deit_base_patch16_224")
    ap.add_argument("--n_val", type=int, default=50000, help="val 이미지 수(기본 전량 5만; 빠른 시험 5000)")
    ap.add_argument("--ratio_list", type=float, nargs="+", default=[0.975, 0.95, 0.925, 0.9],
                    help="공식과 동일한 보존비율(클수록 약한 압축). 공식 eval 예시값과 맞춤")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data_root", default=os.path.join(HERE, "imagenet_val"),
                    help="labels.csv+images/ 있는 폴더(기본 ./imagenet_val; pitome_compare/imagenet_val 심볼릭 링크 가능)")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ★ 분류 헤드를 살려야 함(num_classes 미지정=1000). compare.py 는 kNN이라 num_classes=0 이었음 — 여기선 다름.
    m = timm.create_model(args.model, pretrained=True).eval().to(dev)
    nprefix = getattr(m, "num_prefix_tokens", 1)
    assert nprefix == 1, f"이 스크립트는 prefix=1(CLS만) DeiT 전용. distilled(prefix=2)는 미지원. (got {nprefix})"
    npatch = m.patch_embed.num_patches; L = len(m.blocks)
    tf = OFFICIAL_TF                                        # 공식과 동일 전처리(①↔② 동일 잣대)
    ref = REF.get(args.model, {"baseline": None, "note": "(참조 없음)"})
    print(f"[setup] {args.model} dev={dev} prefix={nprefix} patches={npatch} blocks={L} n_val={args.n_val}", flush=True)
    print(f"[ref]  공식 baseline top-1 = {ref['baseline']} | {ref['note']}", flush=True)

    assert os.path.exists(f"{args.data_root}/labels.csv"), \
        f"val 미준비: python prepare_data.py --split val  (또는 --data_root 로 기존 imagenet_val 지정)"
    ds = ImgFolder(args.data_root, tf, args.n_val)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    print(f"[data] {len(ds)} val images from {args.data_root}", flush=True)

    T0 = nprefix + npatch
    # r=0 무압축 = 파이프라인/라벨 검증(공식 baseline 과 일치해야 함)
    base_acc, _ = eval_acc(m, dl, 1.0, "none", nprefix, dev)
    ok = "" if ref["baseline"] is None else ("  ✅일치" if abs(base_acc - ref["baseline"]) < 0.5 else "  ⚠️불일치(라벨순서/전처리 점검)")
    print(f"\n[검증] r=0 무압축 top-1 = {base_acc:.2f}  (공식 {ref['baseline']}){ok}", flush=True)

    out = {"source": "ours_port", "model": args.model, "n_val": len(ds),
           "baseline_r0": round(base_acc, 2), "ref_baseline": ref["baseline"], "rows": []}
    print(f"\n{'ratio':>6} {'comp%':>6} {'tome':>7} {'pitome':>7}  {'Δ(P-T)':>7}", flush=True)
    for ratio in sorted(args.ratio_list, reverse=True):                  # 약→강 압축
        a_t, ft = eval_acc(m, dl, ratio, "tome", nprefix, dev)
        a_p, _  = eval_acc(m, dl, ratio, "pitome", nprefix, dev)
        comp = 100.0 * (1 - ft / T0)
        out["rows"].append({"ratio": ratio, "comp": round(comp, 1),
                            "tome": round(a_t, 2), "pitome": round(a_p, 2), "delta_PT": round(a_p - a_t, 2)})
        print(f"{ratio:6.3f} {comp:6.1f} {a_t:7.2f} {a_p:7.2f}  {a_p-a_t:+7.2f}", flush=True)

    outdir = os.path.join(HERE, "results"); os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, f"ours_port__{args.model}.json")
    json.dump(out, open(outp, "w"), indent=2); print(f"\n[저장] {outp}", flush=True)
    print("\n[해석] 우리 포팅의 (pitome−tome) 격차·곡선이 공식 논문 값과 같은 추세면 포팅 정확.", flush=True)
    print("       공개 참조(같은 계열): DeiT-S off-the-shelf 에서 PiToMe−ToMe ≈ +1.4%p, DeiT-T ≈ +1.9%p.", flush=True)
    print("       (공식 절대 FLOPs 매핑이 달라 정확한 동일점 비교는 run_official_pitome.sh 로 실측 대조.)", flush=True)


if __name__ == "__main__":
    main()
