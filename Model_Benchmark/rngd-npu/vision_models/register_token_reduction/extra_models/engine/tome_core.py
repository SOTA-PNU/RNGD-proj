#!/usr/bin/env python3
"""robustness_50k 공용 코어 — 다른 스크립트가 재사용(서버에서 self-contained 실행).
merge_step(size-가중 ToMe 병합) · forward_kprotect(앞 n개 보호 forward) · knn · knn_correct ·
bootstrap_ci · load_model_and_data. GPU 있으면 자동 cuda."""
import os, csv
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import warnings; warnings.filterwarnings("ignore")
import torch, timm, torch.nn.functional as F
from PIL import Image

VAL = os.environ.get("IMAGENET_VAL", os.path.expanduser("~/register_token_reduction/imagenet_val"))


@torch.no_grad()
def merge_step(x, size, r, n_protect):
    """앞 n_protect개 보호. 비보호 중 r개 size-가중 bipartite 병합(ToMe, Bolya ICLR'23)."""
    B, T, C = x.shape
    r = min(r, max((T - n_protect) // 2, 0))
    if r <= 0:
        return x, size
    xp, xr = x[:, :n_protect], x[:, n_protect:]
    sp, sr = size[:, :n_protect], size[:, n_protect:]
    a, b = xr[:, ::2, :], xr[:, 1::2, :]; sa, sb = sr[:, ::2, :], sr[:, 1::2, :]
    an = F.normalize(a, dim=-1); bn = F.normalize(b, dim=-1)
    scores = an @ bn.transpose(-1, -2)
    node_max, node_idx = scores.max(dim=-1)
    edge = node_max.argsort(dim=-1, descending=True)[..., None]
    unm_idx, src_idx = edge[:, r:, :], edge[:, :r, :]
    dst_idx = node_idx[..., None].gather(1, src_idx)
    unm = a.gather(1, unm_idx.expand(-1, -1, C)); unm_s = sa.gather(1, unm_idx.expand(-1, -1, 1))
    src = a.gather(1, src_idx.expand(-1, -1, C)); src_s = sa.gather(1, src_idx.expand(-1, -1, 1))
    b_acc = (b * sb).scatter_add(1, dst_idx.expand(-1, -1, C), src * src_s)
    s_acc = sb.scatter_add(1, dst_idx.expand(-1, -1, 1), src_s)
    return torch.cat([xp, unm, b_acc / s_acc], 1), torch.cat([sp, unm_s, s_acc], 1)


@torch.no_grad()
def forward_kprotect(m, x, r, nprot):
    """DINOv2-reg는 CLS(0)+register(1..4)가 prefix → nprot=1+k면 CLS+register k개 보호(재정렬 불요)."""
    t = m._pos_embed(m.patch_embed(x)); B, T, C = t.shape
    size = torch.ones(B, T, 1, dtype=t.dtype, device=t.device)
    for blk in m.blocks:
        t = blk(t); t, size = merge_step(t, size, r, nprot)
    return m.norm(t)[:, 0], t.shape[1]


def knn(Fe, Y, k=20, chunk=4096):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # kNN도 GPU서 실행(있으면)
    Fn = F.normalize(Fe.to(dev), dim=-1); Y = Y.to(dev); n = Fn.shape[0]; c = 0
    for i in range(0, n, chunk):
        s = Fn[i:i+chunk] @ Fn.T
        rows = torch.arange(s.shape[0], device=dev); s[rows, i + rows] = -2   # 자기 제외(벡터화)
        c += (torch.mode(Y[s.topk(k, 1).indices], 1).values == Y[i:i+s.shape[0]]).sum().item()
    return 100 * c / n


def knn_correct(Fe, Y, k=20, chunk=4096):
    """항목별 정답여부(bool) — 부트스트랩용. GPU서 계산 후 CPU로 반환(부트스트랩은 CPU)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Fn = F.normalize(Fe.to(dev), dim=-1); Y = Y.to(dev); n = Fn.shape[0]
    correct = torch.zeros(n, dtype=torch.bool, device=dev)
    for i in range(0, n, chunk):
        s = Fn[i:i+chunk] @ Fn.T
        rows = torch.arange(s.shape[0], device=dev); s[rows, i + rows] = -2   # 자기 제외(벡터화)
        pred = torch.mode(Y[s.topk(k, 1).indices], 1).values
        correct[i:i+s.shape[0]] = (pred == Y[i:i+s.shape[0]])
    return correct.cpu()


def bootstrap_ci(c_a, c_b, B=2000, seed=0):
    """(a−b) 정확도차(%)의 95% CI. paired 재표집."""
    g = torch.Generator().manual_seed(seed); n = len(c_a)
    da = c_a.float(); db = c_b.float(); diffs = []
    for _ in range(B):
        idx = torch.randint(0, n, (n,), generator=g)
        diffs.append((da[idx].mean() - db[idx].mean()).item() * 100)
    diffs.sort()
    return diffs[int(0.025 * B)], diffs[int(0.975 * B)], sum(diffs) / len(diffs)


def load_model_and_data(model_name, n):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=224).eval().to(dev)
    cfg = timm.data.resolve_model_data_config(m); cfg["input_size"] = (3, 224, 224)
    tf = timm.data.create_transform(**cfg, is_training=False)
    rows = list(csv.DictReader(open(f"{VAL}/labels.csv")))[:n]
    X = torch.stack([tf(Image.open(f"{VAL}/images/{r['filename']}").convert("RGB")) for r in rows])
    Y = torch.tensor([int(r["label_idx"]) for r in rows])
    return m, X, Y, dev
