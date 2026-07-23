#!/usr/bin/env python3
"""실제 PiToMe(공식 repo algo/pitome/merge.py) vs Ours(register 보호) vs ToMe head-to-head.
감사 지적 '#1 must-win: energy는 프록시일 뿐, 실제 PiToMe와 head-to-head 필요'에 대응.
PiToMe: CLS 제외 후 energy=elu(sim-margin).mean, 고에너지(중복) 상위 2r 병합·저에너지 보호(재학습 없음).
Ours/ToMe: eval_v2.reduced_forward 재사용(size-가중 ToMe, 보호대상만 변경). DINOv2-reg, kNN. (로컬 CPU/GPU)
사용: python pitome_h2h.py [n] [margin]"""
import os, sys, csv
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import warnings; warnings.filterwarnings("ignore")
import torch, timm, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__) + "/register_token_reduction/eval_v2")
from eval_v2 import reduced_forward, knn

VAL = os.environ.get("IMAGENET_VAL", "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/imagenet_val")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
MARGIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"


@torch.no_grad()
def pitome_reduce(x, size, r, margin=MARGIN, alpha=1.0):
    """공식 PiToMe 병합(CLS 제외, energy 상위 2r 병합). size-가중(공정비교). x:[B,T,C]."""
    B, T, C = x.shape
    cls, xr = x[:, :1], x[:, 1:]; cls_s, sr = size[:, :1], size[:, 1:]
    Tr = xr.shape[1]
    if r <= 0 or 2 * r > Tr:
        return x, size
    m = F.normalize(xr, dim=-1); sim = m @ m.transpose(-1, -2)
    energy = F.elu(sim - margin, alpha=alpha).mean(-1)          # [B,Tr]
    idx = energy.argsort(dim=-1, descending=True)               # 고에너지 먼저
    merge_idx = idx[:, :2 * r]; prot_idx = idx[:, 2 * r:]       # 상위 2r 병합, 나머지 보호
    a_idx, b_idx = merge_idx[:, ::2], merge_idx[:, 1::2]        # [B,r]
    s = sim.gather(-1, b_idx.unsqueeze(1).expand(B, Tr, r)).gather(-2, a_idx.unsqueeze(-1).expand(B, r, r))
    dst = s.max(-1).indices                                     # [B,r] 각 a가 합쳐질 b
    bi = torch.arange(B).unsqueeze(1)
    prot, prot_s = xr[bi, prot_idx], sr[bi, prot_idx]
    a, a_s = xr[bi, a_idx], sr[bi, a_idx]; b, b_s = xr[bi, b_idx], sr[bi, b_idx]
    b_acc = (b * b_s).scatter_add(1, dst.unsqueeze(-1).expand(B, r, C), a * a_s)
    s_acc = b_s.scatter_add(1, dst.unsqueeze(-1).expand(B, r, 1), a_s)
    return torch.cat([cls, prot, b_acc / s_acc], 1), torch.cat([cls_s, prot_s, s_acc], 1)


@torch.no_grad()
def forward_pitome(m, x, r):
    t = m._pos_embed(m.patch_embed(x)); B, T, C = t.shape
    size = torch.ones(B, T, 1, dtype=t.dtype)
    for blk in m.blocks:
        t = blk(t); t, size = pitome_reduce(t, size, r)
    return m.norm(t)[:, 0], t.shape[1]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=224).eval().to(dev)
    nprefix = m.num_prefix_tokens; nreg = nprefix - 1
    cfg = timm.data.resolve_model_data_config(m); cfg["input_size"] = (3, 224, 224)
    tf = timm.data.create_transform(**cfg, is_training=False)
    rows = list(csv.DictReader(open(f"{VAL}/labels.csv")))[:N]
    X = torch.stack([tf(Image.open(f"{VAL}/images/{r['filename']}").convert("RGB")) for r in rows])
    Y = torch.tensor([int(r["label_idx"]) for r in rows])
    print(f"{MODEL} · {N}장 · dev={dev} · PiToMe margin={MARGIN}", flush=True)
    print(f"{'r':>3} {'comp%':>6} {'tome':>7} {'ours':>7} {'PiToMe(real)':>13}  판정", flush=True)
    for r in [12, 16, 20]:
        feats = {}
        for s in ["tome", "ours"]:
            fs = [reduced_forward(m, X[i:i+40].to(dev), r, s, nprefix, nreg, torch.Generator().manual_seed(0))[0].float().cpu()
                  for i in range(0, N, 40)]
            feats[s] = knn(torch.cat(fs), Y)
        pf, finalT = [], None
        for i in range(0, N, 40):
            e, finalT = forward_pitome(m, X[i:i+40].to(dev), r)
            pf.append(e.float().cpu())
        feats["pitome"] = knn(torch.cat(pf), Y)
        comp = 100 * (1 - finalT / (m.patch_embed.num_patches + nprefix))
        d = feats["ours"] - feats["pitome"]
        verdict = "✅ ours>PiToMe" if d > 0.5 else ("≈ 비슷" if abs(d) <= 0.5 else "❌ PiToMe 우위")
        print(f"{r:>3} {comp:>6.1f} {feats['tome']:>7.2f} {feats['ours']:>7.2f} {feats['pitome']:>13.2f}  Δ(ours-PiToMe)={d:+.2f} {verdict}", flush=True)
    print("\n해석: ours가 실제 PiToMe를 극단압축서 이기면 = 'register 보호가 energy 기반 SOTA보다 낫다' 직접 근거.", flush=True)


if __name__ == "__main__":
    main()
