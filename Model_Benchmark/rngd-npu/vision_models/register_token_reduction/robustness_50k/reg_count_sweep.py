#!/usr/bin/env python3
"""[50k] 보호 register 개수 스윕(k=0..4) + 부트스트랩 신뢰구간.  (감사 ③ 보강)
목적1 인과격리: 이득이 '토큰을 더 보호'가 아니라 'register 자체'라면 k 늘릴수록 정확도 단조↑.
              k=0=ToMe(CLS만), k=4=Ours(CLS+register4). 같은 병합, 보호개수만 변경.
목적2 유의성: 병합/kNN은 결정적이라 seed분산=0 → 평가셋 부트스트랩으로 (ours−tome) gap의 95% CI.
사용(전체): python reg_count_sweep.py 50000"""
import sys, torch
from tome_core import forward_kprotect, knn_correct, bootstrap_ci, load_model_and_data

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"


def main():
    m, X, Y, dev = load_model_and_data(MODEL, N)
    nprefix = m.num_prefix_tokens; T0 = m.patch_embed.num_patches + nprefix
    print(f"{MODEL} · {len(X)}장 · prefix={nprefix}(CLS+{nprefix-1}reg) · dev={dev}", flush=True)
    print("[k 스윕] k=0=ToMe … k=4=Ours, 같은 size-가중 병합", flush=True)
    print(f"{'r':>3} {'comp%':>6} | " + " ".join(f"k={k}".rjust(7) for k in range(nprefix)) +
          "   단조?  95%CI(ours-tome)", flush=True)
    for r in [8, 12, 16, 18, 20]:
        corr = {}; ft = None
        for k in range(nprefix):
            fs = []
            for i in range(0, len(X), 128):
                e, ft = forward_kprotect(m, X[i:i+128].to(dev), r, 1 + k)
                fs.append(e.float().cpu())
            corr[k] = knn_correct(torch.cat(fs), Y)
        accs = [100 * corr[k].float().mean().item() for k in range(nprefix)]
        comp = 100 * (1 - ft / T0)
        mono = all(accs[i] <= accs[i+1] + 0.05 for i in range(len(accs)-1))
        lo, hi, mean = bootstrap_ci(corr[nprefix-1], corr[0])
        sig = "유의(>0)" if lo > 0 else "불명"
        cells = " ".join(f"{a:7.2f}" for a in accs)
        print(f"{r:>3} {comp:>6.1f} | {cells}   {'단조✅' if mono else '비단조❌'}  "
              f"Δ={mean:+.2f} [{lo:+.2f},{hi:+.2f}] {sig}", flush=True)
    print("\n해석: k 단조↑ = 원인=register(하나씩 지킬수록 이득 누적). CI하한>0 = 단일 seed여도 통계적 견고.", flush=True)


if __name__ == "__main__":
    main()
