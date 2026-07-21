#!/usr/bin/env python3
"""보호 register 개수 스윕(k=0..4) + 데이터 부트스트랩 신뢰구간.  (감사 ③ 보강, GPU 불요)
목적 1 — 인과 격리: 이득이 '토큰을 더 보호해서'가 아니라 'register 그 자체'라면,
  보호 register 개수 k를 0→4로 늘릴수록 정확도가 단조 증가해야 한다.
  k=0 = ToMe(CLS만), k=4 = Ours(CLS+register4). 같은 size-가중 ToMe 병합, 보호개수만 변경.
목적 2 — 유의성(단일 seed 우려 대응): 병합/kNN은 CPU에서 결정적이라 seed 분산=0 → 다seed 오차막대가
  CPU에서 무의미. 대신 평가셋을 부트스트랩 재표집해 (ours−tome) gap의 95% CI를 구한다(0을 넘지 않으면 유의).
DINOv2-reg는 CLS(0)+register(1..4)가 prefix라 재정렬 없이 n_protect=1+k로 'CLS+register k개 보호'가 됨.
사용: python reg_count_sweep.py [n]"""
import os, sys, csv
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import warnings; warnings.filterwarnings("ignore")
import torch, timm, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__) + "/register_token_reduction/eval_v2")
from tome_reg import merge_step

VAL = os.environ.get("IMAGENET_VAL", "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/imagenet_val")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"


@torch.no_grad()
def forward_kprotect(m, x, r, nprot):
    t = m._pos_embed(m.patch_embed(x)); B, T, C = t.shape
    size = torch.ones(B, T, 1, dtype=t.dtype, device=t.device)
    for blk in m.blocks:
        t = blk(t); t, size = merge_step(t, size, r, nprot)
    return m.norm(t)[:, 0], t.shape[1]


def knn_correct(Fe, Y, k=20, chunk=2048):
    """항목별 leave-one-out kNN 정답여부(bool). 부트스트랩용으로 per-item 보존."""
    Fn = F.normalize(Fe, dim=-1); n = Fn.shape[0]
    correct = torch.zeros(n, dtype=torch.bool)
    for i in range(0, n, chunk):
        s = Fn[i:i+chunk] @ Fn.T
        for j in range(s.shape[0]): s[j, i+j] = -2
        pred = torch.mode(Y[s.topk(k, 1).indices], 1).values
        correct[i:i+s.shape[0]] = (pred == Y[i:i+s.shape[0]])
    return correct


def bootstrap_ci(c_ours, c_tome, B=2000, seed=0):
    """항목 재표집으로 (ours−tome) 정확도차의 95% CI(%). paired: 같은 인덱스로 두 팔 동시 재표집."""
    g = torch.Generator().manual_seed(seed); n = len(c_ours)
    do = c_ours.float(); dt = c_tome.float(); diffs = []
    for _ in range(B):
        idx = torch.randint(0, n, (n,), generator=g)
        diffs.append((do[idx].mean() - dt[idx].mean()).item() * 100)
    diffs.sort()
    return diffs[int(0.025 * B)], diffs[int(0.975 * B)], sum(diffs) / len(diffs)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=224).eval().to(dev)
    nprefix = m.num_prefix_tokens; T0 = m.patch_embed.num_patches + nprefix
    cfg = timm.data.resolve_model_data_config(m); cfg["input_size"] = (3, 224, 224)
    tf = timm.data.create_transform(**cfg, is_training=False)
    rows = list(csv.DictReader(open(f"{VAL}/labels.csv")))[:N]
    X = torch.stack([tf(Image.open(f"{VAL}/images/{r['filename']}").convert("RGB")) for r in rows])
    Y = torch.tensor([int(r["label_idx"]) for r in rows])
    print(f"{MODEL} · {N}장 · prefix={nprefix}(CLS+{nprefix-1}reg) · dev={dev}", flush=True)
    print("[실험1] 보호 register 개수 k 스윕 (k=0=ToMe … k=4=Ours), 같은 병합", flush=True)
    print(f"{'r':>3} {'comp%':>6} | " + " ".join(f"k={k}" .rjust(7) for k in range(nprefix)) + "   단조?  95%CI(ours-tome)", flush=True)
    for r in [12, 16, 20]:
        corr = {}; ft = None
        for k in range(nprefix):           # 0..4
            fs = []
            for i in range(0, N, 40):
                e, ft = forward_kprotect(m, X[i:i+40].to(dev), r, 1 + k)
                fs.append(e.float().cpu())
            corr[k] = knn_correct(torch.cat(fs), Y)
        accs = [100 * corr[k].float().mean().item() for k in range(nprefix)]
        comp = 100 * (1 - ft / T0)
        mono = all(accs[i] <= accs[i+1] + 0.05 for i in range(len(accs)-1))   # 단조 비감소(±노이즈)
        lo, hi, mean = bootstrap_ci(corr[nprefix-1], corr[0])
        sig = "유의(>0)" if lo > 0 else "불명"
        cells = " ".join(f"{a:7.2f}" for a in accs)
        print(f"{r:>3} {comp:>6.1f} | {cells}   {'✅단조' if mono else '❌비단조'}  Δ={mean:+.2f} [{lo:+.2f},{hi:+.2f}] {sig}", flush=True)
    print("\n해석:", flush=True)
    print(" · 단조 증가면 = 이득이 '보호 개수'가 아니라 register를 하나씩 더 지킬수록 쌓임 → 원인=register.", flush=True)
    print(" · 95%CI 하한>0이면 = +gap이 데이터 재표집에도 0을 넘지 않음 = 단일 seed여도 통계적으로 견고.", flush=True)


if __name__ == "__main__":
    main()
