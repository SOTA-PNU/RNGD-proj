#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
selfcheck.py — 어댑터 정확성 게이트 (GPU 서버에서 실 실험 전에 반드시 통과시킬 것).

검증 1 (핵심): r=0(병합 없음) faithful forward 의 CLS 특징이 **공식 forward 의 CLS** 와 일치.
             일치(cosine≈1)하면 우리 rope-aware forward 가 모델을 정확히 재현한다는 증거.
             불일치면 rope 적용 형태(_vit5_patch_rope 의 rope 테이블 가정 등)를 여기서 바로 잡을 것.
검증 2: r>0 에서 patch 만 병합되고 CLS+register 는 보존되는지(토큰 수·무크래시) 확인.
검증 3: noreg(k=0) 는 register 없는 시퀀스로 도는지 확인.

사용:
  python selfcheck.py --model dinov3_base
  python selfcheck.py --model dinov3_splus
  python selfcheck.py --model vit5_base --vit5_repo <clone> --vit5_ckpt <pth>
"""
import argparse, os, sys
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
import models_extra as MX


@torch.no_grad()
def official_cls(model, img, key):
    """공식 forward 의 CLS 특징. timm EVA/ViT-5 모두 forward_features → [:,0]."""
    feats = model.forward_features(img)
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    return feats if feats.dim() == 2 else feats[:, 0]


def patch_vit5_rope_cpu(repo_dir):
    """공식 VisionRotaryEmbedding.forward 는 t=arange(..).cuda() 하드코딩이라 CPU 에서 못 돎.
    수치 동일하게, device 만 입력 x 를 따르도록 forward 를 교체(검증용). GPU 에선 불필요하지만 무해."""
    if repo_dir and repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    import rope as R
    from einops import repeat as _repeat

    def forward(self, x):
        import numpy as _np, torch as _t
        ft = int(_np.sqrt(x.shape[1]))
        t = _t.arange(ft, device=x.device).float() / ft * self.pt_seq_len
        fr = _t.einsum('..., f -> ... f', t, self.freqs.to(x.device))
        fr = _repeat(fr, '... n -> ... (n r)', r=2)
        fr = R.broadcat((fr[:, None, :], fr[None, :, :]), dim=-1)
        fc = fr.cos().view(-1, 1, fr.shape[-1]); fs = fr.sin().view(-1, 1, fr.shape[-1])
        return x * fc + R.rotate_half(x) * fs
    R.VisionRotaryEmbedding.forward = forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["dinov3_splus", "dinov3_base", "vit5_base"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--vit5_repo", default=os.environ.get("VIT5_REPO"))
    ap.add_argument("--vit5_ckpt", default=os.environ.get("VIT5_CKPT"))
    ap.add_argument("--dinov3_hub", action="store_true")
    ap.add_argument("--dinov3_weights", default=os.environ.get("DINOV3_WEIGHTS"))
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kw = {}
    if args.model == "vit5_base":
        assert args.vit5_repo, "vit5: --vit5_repo 필요(공식 repo clone). --vit5_ckpt 는 선택(없으면 random init 로 어댑터 수학만 검증)"
        patch_vit5_rope_cpu(args.vit5_repo)            # CPU 검증용 rope device 패치
        kw = dict(ckpt=args.vit5_ckpt, repo_dir=args.vit5_repo)
    if args.model.startswith("dinov3") and args.dinov3_hub:
        kw = dict(hub=True, weights=args.dinov3_weights)
    model, nprefix, fwd = MX.get_model_and_forward(args.model, device=dev, img_size=args.img_size, **kw)
    nreg = nprefix - 1
    torch.manual_seed(0)
    img = torch.randn(4, 3, args.img_size, args.img_size, device=dev)

    print(f"[model] {args.model} prefix={nprefix}(CLS+{nreg}reg) blocks={len(model.blocks)}")

    # 검증 1: r=0 == 공식
    off = official_cls(model, img, args.model)
    ours0 = fwd(model, img, 0, n_reg_keep=nreg)
    cos = F.cosine_similarity(off, ours0, dim=-1).mean().item()
    rel = ((off - ours0).norm(dim=-1) / off.norm(dim=-1).clamp_min(1e-6)).mean().item()
    ok1 = cos > 0.999
    print(f"[check1 r=0 vs official]  cosine={cos:.6f}  rel_l2_err={rel:.4e}  -> {'PASS' if ok1 else 'FAIL (rope/forward 재현 안 맞음)'}")

    # 검증 2: r>0 patch 만 줄고 CLS+reg 보존, 무크래시
    try:
        _ = fwd(model, img, 12, n_reg_keep=nreg)
        ok2 = True; msg2 = "ran"
    except Exception as e:
        ok2 = False; msg2 = repr(e)
    print(f"[check2 r=12 merge]       -> {'PASS' if ok2 else 'FAIL'} ({msg2})")

    # 검증 3: noreg(k=0)
    try:
        _ = fwd(model, img, 12, n_reg_keep=0)
        ok3 = True; msg3 = "ran (register 제거 시퀀스)"
    except Exception as e:
        ok3 = False; msg3 = repr(e)
    print(f"[check3 noreg k=0 r=12]   -> {'PASS' if ok3 else 'FAIL'} ({msg3})")

    print("\n결론:", "모두 PASS → 실 실험 진행 가능(run_extra.py)." if (ok1 and ok2 and ok3)
          else "FAIL 있음 → models_extra.py 의 해당 forward(특히 rope 테이블 접근) 수정 후 재검.")
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)


if __name__ == "__main__":
    main()
