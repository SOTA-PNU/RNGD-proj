#!/usr/bin/env python3
"""★결정적 대조군: 그분들 multi-layer size-가중 ToMe(tome_reg.merge_step) 그대로 쓰되,
'register 4개 보호(ours)' vs '아무 patch 4개 보호(random)' vs 'CLS만(tome)'을 **같은 보호 개수**로 비교.
핵심질문: ours의 +3.75%가 정말 'register라서'인가, 아니면 '보호 토큰을 4개 더 둬서/ToMe가 극단압축서 약해서'인가?
  - random이 ours에 맞먹으면 → +3.75%는 register 특수성이 아님(리뷰어가 죽일 지점).
  - ours가 random을 유의하게 이기면 → register 특수성 = 진짜 기여(GO).
지표: kNN top-1(그분들과 동일). CPU 배치. 별도 로그로 병렬 GPU 실험과 충돌 없음.
사용: <furiosa python> h2h_random_control.py [N] [batch]
"""
import os, sys, csv
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn.functional as F, timm
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__) + "/register_token_reduction")
from tome_reg import merge_step                      # 그분들 정식 size-가중 병합 그대로

VM = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models"
IMGDIR = f"{VM}/imagenet_val/images"; LABELS = f"{VM}/imagenet_val/labels.csv"
MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"; NREG = 4
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 25
R_SWEEP = [12, 16, 20]                               # r_per_block (극단압축 영역 포함)
STRATS = ["tome", "ours", "random"]


def perm_for(strat, B, T, dev):
    """각 이미지의 토큰 재배열 perm[B,T] (보호 토큰을 앞으로) 와 n_protect 반환."""
    base = torch.arange(T, device=dev)
    if strat == "tome":
        return base.expand(B, T).clone(), 1           # CLS만
    if strat == "ours":
        return base.expand(B, T).clone(), 1 + NREG     # CLS+register (이미 prefix라 그대로)
    # random: CLS + 비-register patch 4개 무작위 보호 (register는 mergeable=공정 대조)
    perms = []
    g = torch.Generator(device="cpu").manual_seed(20260705)
    cand = torch.arange(1 + NREG, T)                  # register 제외 patch 후보
    for b in range(B):
        sel = cand[torch.randperm(len(cand), generator=g)[:NREG]]
        prot = torch.cat([torch.tensor([0]), sel])
        rest = torch.tensor([i for i in range(T) if i not in set(prot.tolist())])
        perms.append(torch.cat([prot, rest]))
    return torch.stack(perms).to(dev), 1 + NREG


@torch.no_grad()
def reduced_cls(model, x, r, strat):
    t = model._pos_embed(model.patch_embed(x))        # [B,T,C]
    B, T, C = t.shape
    perm, n_protect = perm_for(strat, B, T, t.device)
    t = t.gather(1, perm[..., None].expand(-1, -1, C))
    size = torch.ones(B, t.shape[1], 1, dtype=t.dtype)
    for blk in model.blocks:
        t = blk(t)
        t, size = merge_step(t, size, r, n_protect)
    return model.norm(t)[:, 0], t.shape[1]            # CLS, 최종 토큰수


def load(n):
    rows = []
    with open(LABELS) as f:
        for r in csv.DictReader(f):
            rows.append((r["filename"], int(r["label_idx"])))
            if len(rows) >= n: break
    return rows


def knn_top1(emb, labels, k=10):
    e = F.normalize(emb, dim=1); sim = e @ e.T; sim.fill_diagonal_(-2)
    idx = sim.topk(k, dim=1).indices
    votes = labels[idx]
    pred = torch.mode(votes, dim=1).values
    return (pred == labels).float().mean().item() * 100


def main():
    rows = load(N); files = [r[0] for r in rows]; labels = torch.tensor([r[1] for r in rows])
    print(f"{MODEL} · {len(files)}장 · {labels.unique().numel()}클래스 · batch {BATCH}", flush=True)
    model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=224).eval()
    cfg = timm.data.resolve_model_data_config(model); cfg["input_size"] = (3, 224, 224)
    tf = timm.data.create_transform(**cfg, is_training=False)
    print("이미지 로드...", flush=True)
    imgs = torch.stack([tf(Image.open(f"{IMGDIR}/{fn}").convert("RGB")) for fn in files])

    # 무압축 상한
    with torch.no_grad():
        full = torch.cat([model.norm(model.blocks(model._pos_embed(model.patch_embed(
            imgs[i:i+BATCH]))))[:, 0] for i in range(0, len(imgs), BATCH)])
    print(f"무압축 kNN top-1 = {knn_top1(full, labels):.2f}%  (토큰 {model.patch_embed.num_patches+1+NREG})\n", flush=True)

    print(f"{'r':>3} {'comp%':>6} | " + " ".join(f"{s:>8}" for s in STRATS) + "   판정", flush=True)
    print("-" * 60, flush=True)
    T0 = model.patch_embed.num_patches + 1 + NREG
    for r in R_SWEEP:
        accs = {}; finalT = None
        for s in STRATS:
            embs = []
            for i in range(0, len(imgs), BATCH):
                e, ft = reduced_cls(model, imgs[i:i+BATCH], r, s)
                embs.append(e); finalT = ft
            accs[s] = knn_top1(torch.cat(embs), labels)
        comp = 100 * (1 - finalT / T0)
        d = accs["ours"] - accs["random"]
        verdict = "✅ ours>random(GO)" if d > 0.8 else ("≈ random(위험)" if abs(d) <= 0.8 else "❌ random우위")
        print(f"{r:>3} {comp:>6.1f} | " + " ".join(f"{accs[s]:>8.2f}" for s in STRATS)
              + f"   Δ(ours-rand)={d:+.2f} {verdict}", flush=True)
    print("\n핵심: 극단압축(comp>90%)서 ours가 random을 0.8%+ 이기면 = register 특수성 입증(GO).", flush=True)
    print("      ours≈random이면 = +3.75%는 'register라서'가 아니라 보호개수/ToMe약점 → 주제 재고.", flush=True)


if __name__ == "__main__":
    main()
