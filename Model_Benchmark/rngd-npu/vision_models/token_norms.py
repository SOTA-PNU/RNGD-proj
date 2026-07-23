#!/usr/bin/env python3
"""M1: NaN의 메커니즘 증거 — plain DINOv2엔 극단적 고노름 '아티팩트 토큰'이 있고, register판은 그걸
register 슬롯에 가둔다(흡수)는 걸 per-token L2 norm으로 보인다. (CPU FP32, 컴파일 불필요)
사용: python token_norms.py"""
import warnings; warnings.filterwarnings("ignore")
import torch, timm
from PIL import Image
import numpy as np

IMG_DIR = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/test_images"
IMGS = ["brambling.jpg", "tabby_cat.jpg", "convertible.jpg", "orange.jpg"]


def analyze(model_name, n_reg):
    m = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=224).eval()
    cfg = timm.data.resolve_model_data_config(m); cfg["input_size"] = (3, 224, 224)
    tf = timm.data.create_transform(**cfg, is_training=False)
    xs = torch.stack([tf(Image.open(f"{IMG_DIR}/{f}").convert("RGB")) for f in IMGS])
    norms_per_block = []   # [block][img] = per-token L2 norm tensor
    blocks = m.blocks
    hooks = []
    store = {}
    def mk(i):
        def f(mod, inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            store[i] = t.detach()
        return f
    for i, blk in enumerate(blocks):
        hooks.append(blk.register_forward_hook(mk(i)))
    with torch.no_grad():
        m(xs)
    for h in hooks: h.remove()
    # token 구조: [cls] + [reg×n_reg] + [patch×256]  (timm DINOv2 prefix=1+n_reg)
    nblk = len(blocks)
    prefix = 1 + n_reg
    print(f"\n=== {model_name}  (blocks={nblk}, prefix tokens=cls+{n_reg}reg, patches=256) ===", flush=True)
    print(f"{'blk':>3} {'max|tok|':>9} {'med|tok|':>9} {'ratio':>6} {'argmax위치':>12} {'>20×med 토큰수':>12}", flush=True)
    for i in [0, nblk//4, nblk//2, 3*nblk//4, nblk-1]:
        t = store[i]                       # [B, T, D]
        tn = t.norm(dim=-1)                # [B, T] per-token norm
        b0 = tn[0]                         # 첫 이미지
        med = b0.median().item(); mx = b0.max().item(); arg = int(b0.argmax())
        where = "cls" if arg == 0 else (f"reg{arg-1}" if arg < prefix else f"patch{arg-prefix}")
        n_extreme = int((b0 > 20*med).sum())
        print(f"{i:>3} {mx:>9.1f} {med:>9.2f} {mx/med:>6.1f} {where:>12} {n_extreme:>12}", flush=True)
    # 마지막 블록: prefix(reg) vs patch 의 최대 노름 비교 (register가 고노름을 가져갔나)
    last = store[nblk-1][0]
    ln = last.norm(dim=-1)
    pref_max = ln[:prefix].max().item() if prefix > 0 else 0
    patch_max = ln[prefix:].max().item()
    print(f"  [마지막블록] prefix(cls+reg) max|tok|={pref_max:.1f}  vs  patch max|tok|={patch_max:.1f}", flush=True)
    return patch_max, pref_max


print("M1: 고노름 아티팩트 토큰이 NaN의 원인인가 — plain vs register 토큰 노름 비교")
p_plain, _ = analyze("vit_base_patch14_dinov2.lvd142m", 0)
p_reg, pref_reg = analyze("vit_base_patch14_reg4_dinov2.lvd142m", 4)
print("\n=== 결론 ===", flush=True)
print(f"plain DINOv2 patch 최대노름 = {p_plain:.1f}", flush=True)
print(f"reg  DINOv2 patch 최대노름 = {p_reg:.1f}  (register가 고노름 흡수하면 patch 노름↓, register 노름↑)", flush=True)
print(f"reg  DINOv2 prefix(reg포함) 최대노름 = {pref_reg:.1f}", flush=True)
if p_plain > 2 * p_reg:
    print(">>> plain의 극단적 고노름 패치토큰이 register판에선 사라짐 → register가 흡수 → NaN 원인=고노름 토큰 (인과 일치) <<<", flush=True)
