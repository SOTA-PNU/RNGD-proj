#!/usr/bin/env python3
"""mechanism 직접 증거: 표준 ToMe(protect CLS only)가 register 토큰을 실제로 '합쳐 없애나'?
merge_step_track로 원본→현재 토큰을 층마다 추적해, 각 register(입력 pos 1..4)가 몇 번째 블록에서
다른 토큰과 병합되는지(=정체성 소멸) 측정. Ours는 register 보호라 병합 0(구성상). (로컬 CPU)
사용: python reg_merge_freq.py"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import warnings; warnings.filterwarnings("ignore")
import torch, timm
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__) + "/register_token_reduction/dense")
from tome_reg_dense import merge_step_track

IMGDIR = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/imagenet_val/images"
R = 16          # 블록당 병합 수(74% 축소 영역)
NIMG = 16


@torch.no_grad()
def probe(m, x, n_protect):
    """n_protect=1(ToMe, register 무보호) 또는 5(Ours). 각 register 최초 병합 블록 반환(None=끝까지 생존)."""
    t = m._pos_embed(m.patch_embed(x))          # [1,T,C]
    T0 = t.shape[1]; nprefix = 5; regs = [1, 2, 3, 4]
    dev = t.device
    orig2cur = torch.arange(T0, device=dev)     # 원본 i -> 현재 위치
    size = torch.ones(1, T0, 1, dtype=t.dtype, device=dev)
    first = {r: None for r in regs}
    for k, blk in enumerate(m.blocks):
        t = blk(t)
        t, size, newpos = merge_step_track(t, size, R, n_protect)
        orig2cur = newpos[0][orig2cur]          # 합성(배치1)
        for reg in regs:
            if first[reg] is None:
                cur = orig2cur[reg].item()
                mates = (orig2cur == cur).sum().item()   # 같은 최종토큰을 공유하는 원본 수
                if mates > 1:                            # 누군가와 병합됨
                    first[reg] = k
    return first


def main():
    m = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).eval()
    cfg = timm.data.resolve_model_data_config(m); cfg["input_size"] = (3, 224, 224)
    tf = timm.data.create_transform(**cfg, is_training=False)
    files = sorted(os.listdir(IMGDIR))[:NIMG]
    X = torch.stack([tf(Image.open(f"{IMGDIR}/{f}").convert("RGB")) for f in files])

    print(f"DINOv2-reg, r={R}(블록당), 이미지 {NIMG}장. 각 register가 몇 번째 블록서 병합되나(12블록).")
    for label, npr in [("ToMe (register 무보호)", 1), ("Ours (register 보호)", 5)]:
        merged_blocks = []; survived = 0; total = 0
        for i in range(len(X)):
            first = probe(m, X[i:i+1], npr)
            for reg, b in first.items():
                total += 1
                if b is None: survived += 1
                else: merged_blocks.append(b)
        avg = sum(merged_blocks)/len(merged_blocks) if merged_blocks else float("nan")
        print(f"\n[{label}]")
        print(f"  register-이미지쌍 {total}개 중 병합된 것: {len(merged_blocks)} ({100*len(merged_blocks)/total:.0f}%), "
              f"끝까지 생존: {survived} ({100*survived/total:.0f}%)")
        if merged_blocks:
            print(f"  최초 병합 블록 평균: {avg:.1f}/12 (낮을수록 일찍 사라짐), 최소~최대: {min(merged_blocks)}~{max(merged_blocks)}")
    print("\n해석: ToMe서 register가 이른 블록에 자주 병합돼 사라지면 = '표준 병합이 register를 합쳐 없앤다'는 직접 증거.")
    print("      Ours는 보호라 100% 생존(구성상). → 정확도 이득의 메커니즘을 간접추론이 아닌 직접측정으로 뒷받침.")


if __name__ == "__main__":
    main()
