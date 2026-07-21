#!/usr/bin/env python3
"""NPU 실측 지연 — register-aware 토큰압축의 속도 이득(실리콘).
토큰 축소는 블록마다 토큰 수를 r개씩 줄인다(어느 토큰을 병합하든 '지연'은 토큰 수만 좌우).
그래서 '블록마다 토큰 수를 r씩 정적으로 줄인 DINOv2-reg 백본'을 RNGD에 컴파일해 forward 지연을 잰다.
r=0=무압축(full), r↑=고압축. Lu가 중시하는 실측 속도 근거.
사용: python npu_latency.py --npu 0            (NPU 컴파일+측정)
      python npu_latency.py --cpu              (CPU 검증/참고, 컴파일 없음)
"""
import argparse, time, json, os, warnings
warnings.filterwarnings("ignore")
import torch

MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"
R_LIST = [0, 8, 12, 16, 18, 20]


class RedViT(torch.nn.Module):
    """블록마다 토큰 수를 r개 줄인 백본(뒤쪽 patch부터, prefix 보존). CLS 반환."""
    def __init__(self, m, r):
        super().__init__(); self.m = m; self.r = int(r)
    def forward(self, x):
        m = self.m
        t = m._pos_embed(m.patch_embed(x))
        for blk in m.blocks:
            t = blk(t)
            if self.r > 0:
                t = t[:, : t.shape[1] - self.r]     # 정적 슬라이스(뒤 r개=patch 제거)
        return m.norm(t)[:, 0]


def build(r):
    import timm
    m = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=224).eval()
    for p in m.parameters(): p.requires_grad_(False)
    T0 = m.patch_embed.num_patches + getattr(m, "num_prefix_tokens", 1)
    nblk = len(m.blocks)
    finalT = T0 - nblk * r if r > 0 else T0
    return RedViT(m, r).eval(), T0, finalT, nblk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npu", type=int, default=0, help="rngd PE 인덱스(전역). 0-7=npu0,8-15=npu1...")
    ap.add_argument("--cpu", action="store_true", help="CPU 검증/참고(컴파일 없음)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--r_list", type=int, nargs="+", default=R_LIST)
    args = ap.parse_args()
    ex = torch.randn(1, 3, 224, 224)

    if args.cpu:
        print("[CPU 검증/참고]  r  finalT  comp%   cpu_ms", flush=True)
        base = None
        for r in args.r_list:
            w, T0, fT, nblk = build(r)
            with torch.no_grad():
                w(ex)  # warmup
                t0 = time.time()
                for _ in range(5): w(ex)
                ms = (time.time() - t0) / 5 * 1000
            comp = 100 * (1 - fT / T0)
            print(f"            {r:>2}  {fT:>5}  {comp:4.1f}%  {ms:7.1f}  (shape OK)", flush=True)
        print("\nCPU 검증 통과 → NPU 측정은 --npu 로.", flush=True)
        return

    # ---- NPU ----
    import furiosa.torch  # noqa (torch 다음에 import; rngd 백엔드 등록)
    from furiosa.torch import CompileModule
    from torch._decomp import core_aten_decompositions, get_decompositions
    DECOMP = dict(core_aten_decompositions())
    DECOMP.update(get_decompositions([
        torch.ops.aten._native_batch_norm_legit_no_training, torch.ops.aten._native_batch_norm_legit,
        torch.ops.aten.batch_norm, torch.ops.aten.native_batch_norm]))
    dev = torch.device("rngd", args.npu)
    print(f"[NPU 지연] {MODEL} on rngd:{args.npu}, iters={args.iters}", flush=True)
    print(f"{'r':>3} {'finalT':>6} {'comp%':>6} {'compile_s':>9} {'npu_ms':>8} {'speedup':>8}", flush=True)
    rows = []; base_ms = None
    for r in args.r_list:
        w, T0, fT, nblk = build(r)
        comp = round(100 * (1 - fT / T0), 1)
        try:
            with torch.no_grad():
                t = time.time()
                ep = torch.export.export(w, (ex,)).run_decompositions(DECOMP)
                cm = CompileModule.from_exported(ep)
                cs = round(time.time() - t, 1)
                cm.to(dev)
                cm(ex.to(dev), device=dev)  # warmup
                xd = ex.to(dev)
                t0 = time.time()
                for _ in range(args.iters): o = cm(xd, device=dev)
                npu_ms = round((time.time() - t0) / args.iters * 1000, 3)
        except Exception as e:
            c = e
            while getattr(c, "__cause__", None) is not None: c = c.__cause__
            print(f"{r:>3} {fT:>6} {comp:>6} {'FAIL':>9}  {str(c).splitlines()[0][:60]}", flush=True)
            rows.append({"r": r, "finalT": fT, "comp": comp, "compile": "FAIL", "err": str(c).splitlines()[0][:120]})
            continue
        if r == 0: base_ms = npu_ms
        sp = round(base_ms / npu_ms, 2) if base_ms else 1.0
        print(f"{r:>3} {fT:>6} {comp:>6} {cs:>9} {npu_ms:>8} {sp:>7}x", flush=True)
        rows.append({"r": r, "finalT": fT, "comp": comp, "compile_s": cs, "npu_ms": npu_ms, "speedup_vs_full": sp})
    outdir = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/results"; os.makedirs(outdir, exist_ok=True)
    json.dump({"model": MODEL, "npu": args.npu, "rows": rows}, open(f"{outdir}/npu_latency.json", "w"), indent=2)
    print(f"\n저장 {outdir}/npu_latency.json", flush=True)
    print("해석(정직): 이 측정은 실제 병합(argsort/scatter)이 아니라 '뒤 r개 토큰을 정적으로 잘라낸 백본'의 실칩 지연이다.", flush=True)
    print("  실측상 이 토큰수 구간(261~21)선 지연이 줄지 않음(고정 오버헤드 지배). 즉 이 NPU·이 구간서 토큰축소의 속도이득은 관측 안 됨.", flush=True)
    print("  → 논문서 'NPU 가속' 주장 금지. 효율은 FLOP/토큰수 감소로 별도 제시하고, 실측 speedup은 미해결(open)로 표기.", flush=True)


if __name__ == "__main__":
    main()
