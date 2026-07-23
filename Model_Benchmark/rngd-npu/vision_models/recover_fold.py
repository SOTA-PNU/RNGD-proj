#!/usr/bin/env python3
"""#1/#11 복구 v2: per-channel 스케일 접기(SmoothQuant식 등화)로 ViT 붕괴 회복 시험.

클립(v1)은 실패(약하면 무효·세면 모델 파괴). 스케일 접기는 FP32 출력 불변이라 CPU를 안 망가뜨림.
접기 가능한 경계 = LayerNorm -> Linear (그 사이 비선형 없음, 잔차와 분리):
  - ln_1 -> self_attention.in_proj  (입력차원 768)
  - ln_2 -> mlp.0 (Linear 768->3072)
등화: s_j = max|X_j|^a / max|W_:,j|^(1-a)   (X=LN출력 활성 per-ch max, W=Linear 입력열)
  fold: LN.weight[j]/=s_j, LN.bias[j]/=s_j  ;  Linear.weight[:,j]*=s_j   => 출력 불변, 가중치 outlier 완화
calib = 테스트 6장(활성 통계). EDF 가중치독립 -> reuse-edf로 재컴파일 0초.

사용: python recover_fold.py --npu 8 --alphas 0.5 0.25 0.75
"""
import argparse, time, warnings
warnings.filterwarnings("ignore")
import torch
import furiosa.torch
from furiosa.torch import CompileModule
from furiosa.torch.custom_ops.edf import EdfModule
from furiosa.torch.export import ExportedProgramWeight, PASSES
from furiosa.native_torch import ir
from torch._decomp import core_aten_decompositions, get_decompositions
import torchvision.models as M
from torchvision.models import ViT_B_16_Weights
from PIL import Image

EDF_PATH = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/edf/vit_b_16_trained.edf"
IMG_DIR = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/test_images"
IMAGES = ["brambling.jpg", "tabby_cat.jpg", "convertible.jpg", "orange.jpg", "dog1.jpg", "astronaut.jpg"]
TRUTH = ["brambling", "Egyptian cat", "convertible", "orange", "Pembroke", "bobsled"]

DECOMP = dict(core_aten_decompositions())
DECOMP.update(get_decompositions([
    torch.ops.aten._native_batch_norm_legit_no_training, torch.ops.aten._native_batch_norm_legit,
    torch.ops.aten.batch_norm, torch.ops.aten.native_batch_norm]))


def calibrate(m, imgs):
    """LN_1, LN_2 출력 per-channel max|act| 수집 (calib 이미지)."""
    acts = {}
    hooks = []
    def mk(name):
        def f(mod, inp, out):
            a = out.detach().abs().amax(dim=(0, 1))  # [768]
            acts[name] = torch.maximum(acts[name], a) if name in acts else a
        return f
    enc = m.encoder.layers
    for i, blk in enumerate(enc):
        hooks.append(blk.ln_1.register_forward_hook(mk(f"{i}.ln1")))
        hooks.append(blk.ln_2.register_forward_hook(mk(f"{i}.ln2")))
    with torch.no_grad():
        for x in imgs:
            m(x)
    for h in hooks:
        h.remove()
    return acts


def fold(m, acts, alpha):
    """SmoothQuant 등화 접기. 반환=접은 텐서 수."""
    n = 0
    enc = m.encoder.layers
    with torch.no_grad():
        for i, blk in enumerate(enc):
            for lname, ln, lin, is_mha in [(f"{i}.ln1", blk.ln_1, blk.self_attention, True),
                                           (f"{i}.ln2", blk.ln_2, blk.mlp[0], False)]:
                X = acts[lname].clamp(min=1e-6)                    # [768]
                W = lin.in_proj_weight if is_mha else lin.weight   # [*,768]
                Wcol = W.abs().amax(dim=0).clamp(min=1e-6)         # [768]
                s = (X ** alpha) / (Wcol ** (1 - alpha))           # [768]
                s = s.clamp(min=1e-3, max=1e3)
                ln.weight.data /= s
                ln.bias.data /= s
                W.data *= s.unsqueeze(0)
                n += 1
    return n


def build_cm(m, edf, dev):
    with torch.no_grad():
        ep = torch.export.export(m, (torch.randn(1, 3, 224, 224),)).run_decompositions(DECOMP)
        for fx in PASSES:
            ep = fx(ep)
        cm = CompileModule(EdfModule(edf), ExportedProgramWeight(ep))
    cm.to(dev)
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npu", type=int, default=8)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.25, 0.75])
    args = ap.parse_args()

    w = ViT_B_16_Weights.IMAGENET1K_V1
    cats = w.meta["categories"]; tf = w.transforms()
    imgs = [tf(Image.open(f"{IMG_DIR}/{f}").convert("RGB")).unsqueeze(0) for f in IMAGES]
    edf = ir.Edf.deserialize(open(EDF_PATH, "rb").read())
    dev = torch.device("rngd", args.npu)
    print(f"[setup] device=rngd:{args.npu}  images={len(imgs)}  (스케일 접기 복구 v2)", flush=True)

    # 0) baseline (접기 없음)로 reuse-edf + 붕괴 재현
    for alpha in [None] + list(args.alphas):
        m = M.vit_b_16(weights=w).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        tag = "baseline(no-fold)"
        if alpha is not None:
            acts = calibrate(m, imgs)
            nf = fold(m, acts, alpha)
            tag = f"fold alpha={alpha} (folded={nf})"
        t = time.time()
        try:
            cm = build_cm(m, edf, dev)
            how = f"reuse-edf {time.time()-t:.1f}s"
        except Exception as e:
            print(f"[{tag}] reuse-edf 실패: {str(e)[:80]}", flush=True)
            continue
        cpu_ok = npu_ok = npu_ws = 0
        rows = []
        for f, x, truth in zip(IMAGES, imgs, TRUTH):
            with torch.no_grad():
                cpu = m(x); npu = cm(x.to(dev), device=dev).to("cpu").float()
            ct = cats[int(cpu.argmax(-1))]; nt = cats[int(npu.argmax(-1))]
            cpu_ok += (ct == truth); npu_ok += (nt == truth); npu_ws += ("window screen" in nt)
            rows.append(f"{f.split('.')[0]:12s} CPU={ct:16s} NPU={nt}")
        print(f"\n[{tag}  {how}]  CPU정답 {cpu_ok}/6 (FP32불변 확인) · NPU정답 {npu_ok}/6 · NPU=window {npu_ws}/6", flush=True)
        for r in rows:
            print("    " + r, flush=True)


if __name__ == "__main__":
    main()
