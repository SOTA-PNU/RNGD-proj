#!/usr/bin/env python3
"""GO/NO-GO head-to-head (CPU): '어떤 토큰을 지킬 것인가' 기준만 바꿔, 극단 압축에서
다운스트림 분류 정보가 가장 잘 보존되는 기준을 가린다. phase0의 통제 설계(보호집합만 변경)에
★빠져 있던 핵심 베이스라인 PiToMe(에너지)를 추가하고, 지표를 CLS-코사인 대리값에서
실제 top-1(NCM, nearest-class-mean)로 격상.

전략(구조적으로 CLS만 공통 보존, 나머지 keep 예산을 기준별로 채움 — reg 토큰도 평가대상):
  tome   : 가장 '독특한'(최대 유사도 낮은) 토큰 keep          ← 표준 ToMe류 (reg 안 지킴)
  pitome : 에너지(평균 유사도) 낮은 토큰 keep                 ← PiToMe류 (NeurIPS'24, reg 안 지킴)
  ours   : register 슬롯 + 고노름 토큰 강제 보존 후 나머지 tome로 채움  ← 우리 기준
  random : 무작위 keep (대조군)

비교는 '병합 메커니즘 고정 + 보호 기준만 변경'이라 기준 자체의 우열을 격리한다.
지표 ① fidelity: 축소 임베딩 vs 무압축 임베딩 cosine.  ② NCM top-1: 무압축 임베딩으로 만든
클래스 프로토타입에 축소 임베딩을 최근접 분류한 실제 정확도.  핵심질문: 극단 압축(keep 5~10%)서
ours 가 pitome 를 이기는가.  사용: <furiosa python> h2h_global_token.py [N] [model]
"""
import os, sys, csv
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import warnings; warnings.filterwarnings("ignore")
import torch, timm
import torch.nn.functional as F
from PIL import Image

VM = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models"
IMGDIR = f"{VM}/imagenet_val/images"
LABELS = f"{VM}/imagenet_val/labels.csv"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500          # 균형: 처음 N장 = N/10 클래스 × 10
MODEL = sys.argv[2] if len(sys.argv) > 2 else "vit_base_patch14_reg4_dinov2.lvd142m"
NREG = 4 if "reg4" in MODEL else 0
KEEP_RATIOS = [0.5, 0.25, 0.10, 0.05]
STRATS = ["tome", "pitome", "ours", "random"]


def load_labels(n):
    rows = []
    with open(LABELS) as f:
        for r in csv.DictReader(f):
            rows.append((r["filename"], int(r["label_idx"])))
            if len(rows) >= n: break
    return rows


@torch.no_grad()
def extract(model, files):
    cfg = timm.data.resolve_model_data_config(model); cfg["input_size"] = (3, 224, 224)
    tf = timm.data.create_transform(**cfg, is_training=False)
    toks = []   # 각 이미지의 최종블록 토큰 [T,D]
    for fn in files:
        x = tf(Image.open(f"{IMGDIR}/{fn}").convert("RGB")).unsqueeze(0)
        feats = model.forward_features(x)        # [1,T,D] (timm ViT: prefix 포함 토큰열)
        toks.append(feats.squeeze(0))
    return toks


def protect_indices(x, strat, budget, nprefix):
    """x:[T,D]. CLS(0) 외에 keep할 토큰 인덱스(비-CLS)를 budget개 고른다."""
    T = x.shape[0]
    idx_all = torch.arange(1, T)                 # CLS 제외 후보 (reg 포함)
    xn = F.normalize(x, dim=-1)
    if strat == "random":
        g = torch.Generator().manual_seed(1234 + budget)
        sel = idx_all[torch.randperm(len(idx_all), generator=g)[:budget]]
        return sel
    sim = xn[idx_all] @ xn[idx_all].T            # [C,C]
    sim.fill_diagonal_(-1)
    if strat == "tome":
        score = sim.max(dim=1).values            # 최대 유사도 (낮을수록 독특)
        sel = idx_all[score.argsort()[:budget]]
    elif strat == "pitome":
        energy = sim.clamp(min=0).mean(dim=1)    # 평균 양의 유사도 (낮을수록 고유)
        sel = idx_all[energy.argsort()[:budget]]
    elif strat == "ours":
        force = list(range(1, nprefix))          # register 슬롯 강제 보존
        norms = x[idx_all].norm(dim=-1)
        hi = idx_all[norms.argsort(descending=True)[:max(1, budget // 4)]].tolist()  # 고노름 강제
        forced = []
        for i in force + hi:
            if i not in forced and i < T: forced.append(i)
        forced = forced[:budget]
        remain = budget - len(forced)
        if remain > 0:
            rest = torch.tensor([i for i in idx_all.tolist() if i not in set(forced)])
            xn2 = xn[rest]; s2 = xn2 @ xn2.T; s2.fill_diagonal_(-1)
            fill = rest[s2.max(dim=1).values.argsort()[:remain]].tolist()
            forced += fill
        sel = torch.tensor(sorted(forced)) if forced else idx_all[:budget]
    return sel


def reduce_pool(x, keep_idx):
    """keep_idx(비-CLS) + CLS 를 survivor로, 나머지는 최근접 survivor에 접어 평균.
    반환: (pooled[D] mean-pool,  dense_fid 스칼라=각 원본 토큰을 자기 survivor merged값으로
    복원했을 때 원본과의 평균 cosine — 공간/dense 과제 보존도)."""
    T, D = x.shape
    keep = torch.cat([torch.tensor([0]), keep_idx.long()]).unique()
    K = len(keep)
    survivors = x[keep]                          # [K,D]
    mask = torch.ones(T, dtype=torch.bool); mask[keep] = False
    nonkeep = torch.arange(T)[mask]
    pooled_sum = survivors.clone(); cnt = torch.ones(K)
    assign = torch.full((T,), -1, dtype=torch.long)      # 각 토큰 → survivor의 행(0..K-1)
    assign[keep] = torch.arange(K)
    if len(nonkeep) > 0:
        xn = F.normalize(x, dim=-1)
        sim = xn[nonkeep] @ F.normalize(survivors, dim=-1).T   # [M,K]
        nearest = sim.argmax(dim=1)
        assign[nonkeep] = nearest
        pooled_sum.index_add_(0, nearest, x[nonkeep])
        cnt.index_add_(0, nearest, torch.ones(len(nonkeep)))
    merged = pooled_sum / cnt[:, None]           # [K,D]
    recon = merged[assign]                        # [T,D] 각 토큰의 복원값
    dense_fid = F.cosine_similarity(recon[1:], x[1:], dim=-1).mean().item()  # CLS 제외 dense 보존
    return merged.mean(dim=0), dense_fid


def main():
    rows = load_labels(N)
    files = [r[0] for r in rows]; labels = torch.tensor([r[1] for r in rows])
    print(f"모델 {MODEL} (nreg={NREG}) · 이미지 {len(files)}장 · 클래스 {labels.unique().numel()}개", flush=True)
    model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=224).eval()
    print("특징 추출 중...", flush=True)
    toks = extract(model, files)
    T = toks[0].shape[0]; nprefix = 1 + NREG
    full_emb = torch.stack([t.mean(dim=0) for t in toks])     # 무압축 mean-pool 임베딩
    # NCM 클래스 프로토타입(무압축 기준)
    classes = labels.unique()
    proto = torch.stack([F.normalize(full_emb[labels == c].mean(0), dim=0) for c in classes])

    def ncm_top1(emb):
        e = F.normalize(emb, dim=1)
        pred = classes[(e @ proto.T).argmax(dim=1)]
        return (pred == labels).float().mean().item() * 100

    print(f"\n무압축 상한 NCM top-1 = {ncm_top1(full_emb):.1f}%  (이 표본·NCM 기준)\n", flush=True)
    print("표기: [poolfid / NCMtop1 / DENSEfid]  (poolfid=풀링보존, top1=분류정확도, DENSEfid=공간/dense보존)", flush=True)
    print(f"{'keep':>6} {'kept':>5} | " + " ".join(f"{s:>21}" for s in STRATS), flush=True)
    print("-" * 96, flush=True)
    summary = {}
    for r in KEEP_RATIOS:
        budget = max(1, round(r * (T - 1)))
        cells = []
        for s in STRATS:
            embs = []; pfids = []; dfids = []
            for x in toks:
                ki = protect_indices(x, s, budget, nprefix)
                pe, dfid = reduce_pool(x, ki)
                embs.append(pe)
                pfids.append(F.cosine_similarity(pe.unsqueeze(0), x.mean(0).unsqueeze(0)).item())
                dfids.append(dfid)
            embs = torch.stack(embs)
            pf = sum(pfids) / len(pfids); top1 = ncm_top1(embs); df = sum(dfids) / len(dfids)
            summary[(r, s)] = (pf, top1, df)
            cells.append(f"{pf:.2f}/{top1:4.1f}/{df:.2f}")
        print(f"{r:>6.2f} {budget:>5} | " + " ".join(f"{c:>21}" for c in cells), flush=True)

    print("\n=== GO/NO-GO 판정 ===", flush=True)
    print("(A) 분류(NCM top-1): ours vs pitome / random", flush=True)
    for r in KEEP_RATIOS:
        o, p, t, rd = (summary[(r, s)] for s in ["ours", "pitome", "tome", "random"])
        print(f"  keep {r:.2f}:  ours={o[1]:.1f}  pitome={p[1]:.1f}  random={rd[1]:.1f}  "
              f"→ Δ(ours-random)={o[1]-rd[1]:+.1f}", flush=True)
    print("(B) DENSE 보존(공간 과제 대리): ours vs random/pitome — random이 무너지고 ours가 버티면 = 진짜 차별점", flush=True)
    for r in KEEP_RATIOS:
        o, p, t, rd = (summary[(r, s)] for s in ["ours", "pitome", "tome", "random"])
        d = o[2] - rd[2]
        verdict = "✅ ours≫random(dense서 차별)" if d > 0.02 else ("≈ 비슷(차별 약함)" if d > -0.02 else "❌ random 우위")
        print(f"  keep {r:.2f}:  ours_dense={o[2]:.3f}  random={rd[2]:.3f}  pitome={p[2]:.3f}  tome={t[2]:.3f}  "
              f"→ Δ(ours-random)={d:+.3f}  {verdict}", flush=True)
    print("\n해석: 분류(A)는 random도 잘해 변별력 낮음(=논문 헤드라인 금지). 진짜 승부는 DENSE(B):", flush=True)
    print("      극단압축서 ours의 dense보존이 random/pitome을 크게 이기면 → 논제 GO(단 ADE20k seg로 확증 필요).", flush=True)


if __name__ == "__main__":
    main()
