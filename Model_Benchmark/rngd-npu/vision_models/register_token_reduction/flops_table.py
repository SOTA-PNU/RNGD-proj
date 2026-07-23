#!/usr/bin/env python3
"""FLOP 분석(순수 계산, GPU 불요): DINOv2-reg 백본에서 토큰 축소가 줄이는 FLOP.
ToMe(protect CLS)와 Ours(protect CLS+register4)를 각 r에서 비교 — 팔별 실제 토큰 스케줄로.
효율의 정직한 근거: 토큰/FLOP은 실제로 줄어듦(단, 실측 NPU wall-clock은 이득 0 = 별개).
사용: python flops_table.py"""
import json

T0, C, NBLK, MLP = 261, 768, 12, 4          # DINOv2-ViT-B/14-reg4


def blk_flops(N):                            # 블록 1개 FLOP(상대비교용, 2*MAC)
    return 2 * (12 * N * C * C + 2 * N * N * C)   # (qkv+proj)=4NC^2, mlp=8NC^2, attn=2N^2C


def schedule(r, n_protect):
    """merge_step의 r 캡((N-n_protect)//2) 반영. 블록 입력 토큰수 리스트와 최종 토큰수."""
    N = T0; Ns = []
    for _ in range(NBLK):
        Ns.append(N)
        reff = min(r, max((N - n_protect) // 2, 0))
        N -= reff
    return Ns, N


def total(r, n_protect):
    Ns, finalN = schedule(r, n_protect)
    return sum(blk_flops(n) for n in Ns), finalN


FULL = NBLK * blk_flops(T0)
print(f"DINOv2-ViT-B/14-reg4: T0={T0}, C={C}, blocks={NBLK}. 무압축 FLOP=1.000\n")
print(f"{'r':>3} {'ToMe finalT/FLOP':>18} {'Ours finalT/FLOP':>18} {'FLOP 절감(ours)':>14}")
rows = []
for r in [8, 12, 16, 18, 20]:
    ft, fN = total(r, 1); ot, oN = total(r, 5)
    fr_t, fr_o = ft / FULL, ot / FULL
    print(f"{r:>3} {fN:>7}/{fr_t:>8.3f}   {oN:>7}/{fr_o:>8.3f}   {(1-fr_o)*100:>10.1f}%  (ours가 tome보다 FLOP {(fr_o-fr_t)*100:+.1f}%p)")
    rows.append({"r": r, "tome_finalT": fN, "tome_flop_ratio": round(fr_t, 3),
                 "ours_finalT": oN, "ours_flop_ratio": round(fr_o, 3),
                 "ours_flop_reduction_pct": round((1 - fr_o) * 100, 1)})
json.dump({"model": "vit_base_patch14_reg4_dinov2", "T0": T0, "C": C, "blocks": NBLK, "rows": rows},
          open("/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/register_token_reduction/results/flops.json", "w"), indent=2)
print("\n해석: 토큰 축소는 FLOP을 실제로 줄임(극단서 ~40%+ 절감). Ours는 register 보호로 tome보다 FLOP이 근소히 높음(정직한 iso-budget 차이).")
print("      단, 이 FLOP 절감이 실측 지연으로 이어지진 않음(NPU wall-clock 이득 0) — 효율 주장은 FLOP에 한정.")
