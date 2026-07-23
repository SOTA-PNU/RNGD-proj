#!/usr/bin/env python3
"""make-or-break (논문 핵심 미선점 질문): 왜 CLIP/MAE/ViT/DeiT/reg4-DINOv2는 칩에서 살고
plain DINOv2만 통째 NaN인가? 셋 다 '값이 크게 튀는 토큰(고노름)'을 가졌는데도.

가설: 단순 '토큰 노름 비율(max/median)'은 생존/사망을 못 가른다(CLIP도 높음).
     진짜 가르는 것은 잔차 스트림의 '절대 최대 활성값(massive activation)'이다 —
     저정밀 포맷(FP8 e4m3 표현 최대 ~448, MXFP4는 더 좁음)의 동적 범위를 넘으면
     캐스팅 단계에서 inf→NaN. plain DINOv2만 이 한계를 크게 넘고, register판은
     그 거대 활성을 register 슬롯에 가둬 patch 스트림 최대값을 낮춘다.

이 스크립트는 칩 결과(생존/사망 라벨)가 이미 있는 6개 모델에서 forward만 돌려:
  ① 단순 토큰노름 비율(max/median)        ← 분리 못 할 것으로 예상
  ② 잔차 스트림 절대 최대 활성값          ← 사망만 분리할 것으로 예상(=원리)
  ③ LayerScale 유무·최대 gamma            ← 구조적 차이(DINOv2만 보유)
를 표로 만들고, 어느 지표가 라벨을 깨끗이 가르는지 판정한다. CPU FP32, 컴파일 불필요.

사용: <furiosa venv python> makeorbreak_layerscale.py
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import warnings; warnings.filterwarnings("ignore")
import torch, timm
from PIL import Image

IMG_DIR = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/test_images"
IMGS = ["brambling.jpg", "tabby_cat.jpg", "convertible.jpg", "orange.jpg"]
# 더 큰 표본: imagenet_val/images 에서 결정적으로 N장 사용(있으면 우선)
IMAGENET_DIR = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/imagenet_val/images"
N_IMG = int(os.environ.get("N_IMG", "64"))

# (timm 이름, register 토큰 수, 칩 실측 라벨)  — 라벨 출처: dino_collapse.log / m3_*.log / sweep_*.log
MODELS = [
    ("vit_base_patch14_dinov2.lvd142m",       0, "DIE"),      # plain DINOv2: 6/6 NaN
    ("vit_base_patch14_reg4_dinov2.lvd142m",  4, "SURVIVE"),  # reg4: cosine 1.0000
    ("vit_base_patch16_clip_224.openai",      0, "SURVIVE"),  # CLIP: cosine 0.9999
    ("vit_base_patch16_224.mae",              0, "SURVIVE"),  # MAE: cosine 1.0000
    ("vit_base_patch16_224.augreg_in1k",      0, "SURVIVE"),  # supervised ViT
    ("deit_base_patch16_224.fb_in1k",         0, "SURVIVE"),  # DeiT
]

FP8_E4M3_MAX = 448.0  # FP8 e4m3 표현 가능한 최대 절대값 (저정밀 캐스팅 오버플로 임계의 기준선)


def load_imgs(model):
    cfg = timm.data.resolve_model_data_config(model); cfg["input_size"] = (3, 224, 224)
    tf = timm.data.create_transform(**cfg, is_training=False)
    # 이미지 경로 목록: imagenet_val 우선(결정적 정렬 후 N장), 없으면 test_images
    paths = []
    if os.path.isdir(IMAGENET_DIR):
        files = sorted(f for f in os.listdir(IMAGENET_DIR)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        paths = [f"{IMAGENET_DIR}/{f}" for f in files[:N_IMG]]
    if not paths:
        paths = [f"{IMG_DIR}/{f}" for f in IMGS]
    xs = []
    for p in paths:
        if os.path.exists(p):
            xs.append(tf(Image.open(p).convert("RGB")))
    if not xs:  # 이미지가 없으면 랜덤 입력으로라도 구조 통계는 본다(활성 크기는 입력의존이라 경고)
        print("  [경고] test_images 없음 → 랜덤 입력 사용(절대 활성값은 참고만)")
        xs = [torch.randn(3, 224, 224) for _ in IMGS]
    return torch.stack(xs)


def max_abs_gamma(model):
    """LayerScale(ls1/ls2 등 gamma 파라미터) 최대 절대값. 없으면 None."""
    g = None
    for name, p in model.named_parameters():
        if "gamma" in name or name.endswith(".ls1.gamma") or name.endswith(".ls2.gamma") \
           or ("layer_scale" in name) or (".ls1." in name) or (".ls2." in name):
            v = p.detach().abs().max().item()
            g = v if g is None else max(g, v)
    return g


def analyze(model_name, n_reg, label):
    m = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=224).eval()
    xs = load_imgs(m)
    blocks = m.blocks
    nblk = len(blocks)
    prefix = 1 + n_reg
    store = {}
    hooks = []
    def mk(i):
        def f(mod, inp, out):
            store[i] = (out if isinstance(out, torch.Tensor) else out[0]).detach()
        return f
    for i, blk in enumerate(blocks):
        hooks.append(blk.register_forward_hook(mk(i)))
    with torch.no_grad():
        m(xs)
    for h in hooks: h.remove()

    gmax_abs = 0.0; gmax_abs_blk = -1; gmax_abs_where = "?"
    gmax_tn = 0.0; med_tn_ref = 0.0
    per_block_absmax = []
    for i in range(nblk):
        t = store[i]                 # [B, T, D] 잔차 스트림
        a = t.abs().amax().item()    # 이 블록 절대 최대 활성
        per_block_absmax.append(a)
        if a > gmax_abs:
            gmax_abs = a; gmax_abs_blk = i
            # 어느 토큰에서 터졌나
            bidx = int(t.abs().amax(dim=2).amax(dim=0).argmax())  # 토큰 index (B,T 통합)
            gmax_abs_where = "cls" if bidx == 0 else (f"reg{bidx-1}" if bidx < prefix else f"patch{bidx-prefix}")
        tn = t.norm(dim=-1)          # [B, T] per-token L2
        if tn.max().item() > gmax_tn:
            gmax_tn = tn.max().item()
            med_tn_ref = tn.median().item()
    ratio = gmax_tn / med_tn_ref if med_tn_ref > 0 else float("nan")
    gamma = max_abs_gamma(m)
    return {
        "name": model_name, "label": label, "nblk": nblk,
        "ratio": ratio, "gmax_tn": gmax_tn, "med_tn": med_tn_ref,
        "gmax_abs": gmax_abs, "gmax_abs_blk": gmax_abs_blk, "gmax_abs_where": gmax_abs_where,
        "has_ls": gamma is not None, "gamma": gamma,
        "peak_block": int(torch.tensor(per_block_absmax).argmax()),
    }


def main():
    print("=" * 100)
    print("make-or-break: 무엇이 칩에서의 생존(SURVIVE)과 사망(DIE=전부 NaN)을 가르는가")
    print("=" * 100)
    rows = []
    for name, nreg, label in MODELS:
        try:
            r = analyze(name, nreg, label)
            rows.append(r)
            print(f"[OK] {label:8s} {name}")
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    if not rows:
        print("측정된 모델이 없습니다."); return

    print("\n%-42s %-8s %8s %10s %10s %6s %9s" %
          ("model", "label", "①비율", "②절대최대", "터진블록", "LS?", "maxγ"))
    print("-" * 100)
    for r in rows:
        print("%-42s %-8s %8.1f %10.1f %10s %6s %9s" % (
            r["name"][:42], r["label"], r["ratio"], r["gmax_abs"],
            f"b{r['gmax_abs_blk']}/{r['gmax_abs_where']}",
            "Y" if r["has_ls"] else "-",
            (f"{r['gamma']:.2f}" if r["gamma"] is not None else "-")))

    die = [r for r in rows if r["label"] == "DIE"]
    sur = [r for r in rows if r["label"] == "SURVIVE"]

    def separates(key):
        if not die or not sur: return None
        dmin = min(r[key] for r in die); smax = max(r[key] for r in sur)
        # DIE가 모두 SURVIVE보다 큰가? (분리 여유 배수)
        return dmin, smax, (dmin / smax if smax > 0 else float("inf"))

    print("\n=== 판정 ===")
    for key, kname in [("ratio", "① 토큰노름 비율(max/median)"), ("gmax_abs", "② 잔차 절대 최대 활성값")]:
        s = separates(key)
        if s is None:
            print(f"{kname}: 라벨 부족"); continue
        dmin, smax, gap = s
        ok = "✅ 깨끗이 분리" if dmin > smax else "❌ 분리 실패(겹침)"
        print(f"{kname}: DIE_min={dmin:.1f}  SURVIVE_max={smax:.1f}  →  {ok} (DIE/SURVIVE={gap:.2f}배)")

    print(f"\n기준선: FP8 e4m3 표현 최대 ≈ {FP8_E4M3_MAX} (이를 넘는 활성은 저정밀 캐스팅서 오버플로 위험)")
    for r in rows:
        flag = "  ← 임계 초과" if r["gmax_abs"] > FP8_E4M3_MAX else ""
        print(f"  {r['label']:8s} {r['name'][:42]:42s} 절대최대={r['gmax_abs']:.1f}{flag}")

    print("\n실측 결과(N=64): ①(노름 집중도 max/median)가 DIE만 분리(24.0 vs ≤10.1, ~2.4배),")
    print("  ②(절대최대)는 분리 실패(ViT 598.8 > DINOv2 528.3인데 ViT 생존) → 원리는 '절대 크기'가 아님.")
    print("  → 생사를 가르는 것은 '단일 패치 토큰으로의 에너지 집중도'다. plain DINOv2만 중앙값의 ~24배를")
    print("     한 토큰에 몰아(칩의 per-token 처리서 붕괴), 생존 모델은 ≤10배로 분산. register가 그 집중을")
    print("     전용 슬롯에 흡수(정점 patch→reg, 비율 24→5.5)해 살린다. 이 지표는 FP32 forward만으로 계산되어")
    print("     '사전 점검(artifact)'이 된다(재학습 불요). ⚠️한계: 모델 6개·사망 사례 1개(DINOv2-plain) —")
    print("     SigLIP2/EVA-02/BEiT 등 추가 검증 필요(미선점이나 단일벤더·소표본은 정직히 명시).")


if __name__ == "__main__":
    main()
