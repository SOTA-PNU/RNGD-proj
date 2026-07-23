#!/usr/bin/env python3
"""[faithful] 보호 register 개수 스윕(k=0..4) + 부트스트랩 CI — 정식(faithful) harness.
= reg_count_sweep.py 의 forward 만 통제(forward_kprotect)→정식(forward_faithful)으로 교체.
정식 forward = faithful_tome_h2h.forward_faithful (prop-attn+key-metric+attn↔MLP병합). k=0=ToMe, k=4=Ours.
knn_correct·bootstrap_ci 는 harness 무관(특징/정답벡터 위에서 동작)이라 tome_core 그대로 재사용.
사용(전체): python reg_count_sweep_faithful.py 50000"""
import sys, torch
from tome_core import knn_correct, bootstrap_ci, load_model_and_data
from faithful_tome_h2h import forward_faithful

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"


def main():
    m, X, Y, dev = load_model_and_data(MODEL, N)
    nprefix = m.num_prefix_tokens
    print(f"{MODEL} · {len(X)}장 · prefix={nprefix}(CLS+{nprefix-1}reg) · dev={dev} · [faithful]", flush=True)
    print("[k 스윕] k=0=ToMe … k=4=Ours, 같은 정식 병합(prop-attn+key+attn↔MLP)", flush=True)
    print(f"{'r':>3} {'comp%':>6} | " + " ".join(f"k={k}".rjust(7) for k in range(nprefix)) +
          "   단조?  95%CI(ours-tome)", flush=True)
    for r in [8, 12, 16, 18, 20]:
        corr = {}; ft = None
        for k in range(nprefix):
            fs = []
            for i in range(0, len(X), 128):
                e, ft = forward_faithful(m, X[i:i+128].to(dev), r, 1 + k)   # n_protect=1+k (CLS+register k)
                fs.append(e.float().cpu())
            corr[k] = knn_correct(torch.cat(fs), Y)
        accs = [100 * corr[k].float().mean().item() for k in range(nprefix)]
        mono = "↑" if all(accs[i] <= accs[i+1] + 0.05 for i in range(len(accs)-1)) else "비단조"
        comp = 100 * (1 - ft / (m.patch_embed.num_patches + nprefix))
        lo, hi, mean = bootstrap_ci(corr[nprefix-1], corr[0])
        sig = "유의✅" if lo > 0 else "≈"
        print(f"{r:>3} {comp:>6.1f} | " + " ".join(f"{a:7.2f}" for a in accs) +
              f"   {mono}  Δ={mean:+.2f} [{lo:+.2f},{hi:+.2f}] {sig}", flush=True)
    print("\n해석: 정식 harness서도 k 단조↑ = 원인=register. CI하한>0 = 단일 seed여도 통계적 견고.", flush=True)
    print("      k=0(ToMe)·k=4(Ours) 는 faithful_tome_h2h.py 와 일치해야 함(일관성 점검).", flush=True)


if __name__ == "__main__":
    main()
