#!/usr/bin/env python3
"""#11: 현대 foundation 인코더(DINOv2)도 NPU 저정밀서 붕괴하나? — 임베딩 충실도로 측정.

DINOv2는 분류 헤드가 없는 임베딩 인코더 → top-1 대신 NPU vs CPU '임베딩 코사인 충실도' +
'서로 다른 이미지 임베딩이 NPU에서 한 점으로 퇴화(mode collapse)하는지'로 붕괴를 본다.
timm vit_base_patch14_dinov2: img_size=224면 timm이 pos_embed를 init때 resample(고정 버퍼) →
forward에 interpolate 없음. LayerScale=per-channel mul(안전). 중간 풀링 없음 → vit_b_16처럼 컴파일 기대.

사용: python dino_collapse.py --model vit_base_patch14_dinov2.lvd142m --npu 0
"""
import argparse, time, warnings
warnings.filterwarnings("ignore")
import torch
import furiosa.torch
from furiosa.torch import CompileModule
from torch._decomp import core_aten_decompositions, get_decompositions
import timm
from PIL import Image
import torch.nn.functional as Fn

IMG_DIR = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/test_images"
IMAGES = ["brambling.jpg", "tabby_cat.jpg", "convertible.jpg", "orange.jpg", "dog1.jpg", "astronaut.jpg"]

DECOMP = dict(core_aten_decompositions())
DECOMP.update(get_decompositions([
    torch.ops.aten._native_batch_norm_legit_no_training, torch.ops.aten._native_batch_norm_legit,
    torch.ops.aten.batch_norm, torch.ops.aten.native_batch_norm]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--npu", type=int, default=0)
    args = ap.parse_args()

    m = timm.create_model(args.model, pretrained=True, num_classes=0, img_size=224).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    cfg = timm.data.resolve_model_data_config(m)
    cfg["input_size"] = (3, 224, 224)   # 모델을 img_size=224로 만들었으니 전처리도 224로 강제
    tf = timm.data.create_transform(**cfg, is_training=False)
    imgs = [tf(Image.open(f"{IMG_DIR}/{f}").convert("RGB")).unsqueeze(0) for f in IMAGES]

    x0 = imgs[0]
    t = time.time()
    with torch.no_grad():
        ep = torch.export.export(m, (x0,)).run_decompositions(DECOMP)
    nodes = sum(1 for n in ep.graph.nodes if n.op == "call_function")
    try:
        with torch.no_grad():
            cm = CompileModule.from_exported(ep)
    except Exception as e:
        cause = e
        while cause.__cause__ is not None:
            cause = cause.__cause__
        print(f"[{args.model}] COMPILE_FAIL nodes={nodes} after {time.time()-t:.0f}s :: {str(cause).splitlines()[0][:140]}", flush=True)
        return
    print(f"[{args.model}] COMPILE_OK nodes={nodes} compile={time.time()-t:.0f}s", flush=True)
    try:
        _ep = f"/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/edf/{args.model.split('.')[0]}.edf"
        open(_ep, "wb").write(cm.edf.serialize())
        print(f"  [save] {_ep.split('/')[-1]} 저장", flush=True)
    except Exception as e:
        print(f"  [save] EDF 저장 실패: {type(e).__name__}", flush=True)
    dev = torch.device("rngd", args.npu)
    cm.to(dev)

    cpu_embs, npu_embs, coss = [], [], []
    for f, x in zip(IMAGES, imgs):
        with torch.no_grad():
            c = m(x).float()
            n = cm(x.to(dev), device=dev).to("cpu").float()
        cpu_embs.append(c); npu_embs.append(n)
        cos = Fn.cosine_similarity(c, n, dim=-1).mean().item()
        coss.append(cos)
        print(f"  {f.split('.')[0]:12s} NPU~CPU emb cosine = {cos:.4f}", flush=True)
    # mode collapse 점검: 서로 다른 이미지의 NPU 임베딩이 한 점으로?
    C = torch.cat(cpu_embs); N = torch.cat(npu_embs)
    def pairwise_mean_cos(M):
        Mn = Fn.normalize(M, dim=-1)
        S = Mn @ Mn.t()
        k = S.shape[0]
        return (S.sum() - k) / (k * (k - 1))  # 평균 off-diagonal
    print(f"\n  [요약] NPU~CPU 평균 임베딩 코사인 = {sum(coss)/len(coss):.4f}  (1=완벽, 낮을수록 붕괴)", flush=True)
    print(f"  서로 다른 이미지간 코사인: CPU={pairwise_mean_cos(C):.4f} vs NPU={pairwise_mean_cos(N):.4f}  "
          f"(NPU가 1에 가까우면 mode collapse=전부 같은 임베딩)", flush=True)


if __name__ == "__main__":
    main()
