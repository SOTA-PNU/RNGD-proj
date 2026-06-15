"""
bench_dpe_vs_ve.py — Validate + benchmark the hand-authored EinsumByDpe matmul
(dn_linear_dpe.yaml) against the EinsumByVe matmul (dn_linear.yaml) for the SAME
linear  y[t,o] = sum_i x[t,i]*W[o,i]  == F.linear(x, W).

WHAT IT DOES
  1. VALIDATE both kernels vs torch.F.linear: maxerr, relmean, allclose @ 1e-2/1e-3,
     and confirm _dfg_inner==0 (pure NPU, no CPU fallback).
  2. BENCHMARK steady-state wall-clock (2nd+ calls post-compile) for the realistic
     size x[128,2048] W[512,2048] -> [128,512], report DPE ms, VE ms, speedup.

RUN
  PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj RNGD_DEV=rngd:2 \
    /home/jun/furiosa/bin/python bench_dpe_vs_ve.py
"""
import os, sys, time, torch  # torch FIRST per recipe
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule
import torch.nn.functional as F

DPE_YAML = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/dn_linear_dpe.yaml"
VE_YAML  = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/dn_linear.yaml"
DEV      = os.environ.get("RNGD_DEV", "rngd:2")
ITERS    = int(os.environ.get("BENCH_ITERS", "20"))
WARMUP   = 3  # compile + warm steady-state before timing

torch.manual_seed(0)

def log(*a): print(*a, flush=True)

# --- spy on _dfg_inner (CPU fallback path); 0 calls => pure NPU ---
import furiosa.torch.custom_ops.dfg as dfgmod
_CALLS = {"n": 0}
_orig = dfgmod._dfg_inner
def _spy(*a, **kw):
    _CALLS["n"] += 1
    return _orig(*a, **kw)
dfgmod._dfg_inner = _spy

def build(yaml_path):
    m = TacticKernelModule(open(yaml_path).read())
    return torch.compile(m, backend=ft.backend)

def run_once(cm, x, W):
    """One forward; returns cpu float result + #dfg_inner calls for THIS call."""
    before = _CALLS["n"]
    res = cm(x.contiguous().to(DEV), W.contiguous().to(DEV))
    y = (res[0] if isinstance(res, (list, tuple)) else res).detach().to("cpu").float()
    return y, _CALLS["n"] - before

def validate(name, cm, T, I, O):
    x = torch.randn(T, I, dtype=torch.float32) * 0.1
    W = torch.randn(O, I, dtype=torch.float32) * 0.05
    ref = F.linear(x, W)
    y, dfg = run_once(cm, x, W)
    maxabs  = (y - ref).abs().max().item()
    denom   = ref.abs().mean().item() + 1e-12
    relmean = (y - ref).abs().mean().item() / denom
    ok_2 = torch.allclose(y, ref, atol=1e-2, rtol=1e-2)
    ok_3 = torch.allclose(y, ref, atol=1e-3, rtol=1e-3)
    flag = "NPU(dfg=0)" if dfg == 0 else f"CPU-FALLBACK(+{dfg})"
    log(f"  [{name:3s}] T={T} I={I} O={O}: maxabs={maxabs:.3e} relmean={relmean*100:.2f}% "
        f"allclose@1e-2={ok_2} @1e-3={ok_3} {flag}")
    return dict(maxabs=maxabs, relmean=relmean, ok_2=ok_2, ok_3=ok_3, dfg=dfg)

def bench(name, cm, T, I, O, iters=ITERS, warmup=WARMUP):
    """Steady-state wall-clock per forward (median + mean of `iters` post-warmup)."""
    x = torch.randn(T, I, dtype=torch.float32) * 0.1
    W = torch.randn(O, I, dtype=torch.float32) * 0.05
    # Pre-move weights/inputs once is NOT representative of the kernel call cost we
    # care about (host->device varies); we time the full cm(...) incl. .to(DEV) the
    # same way for both kernels so the comparison is apples-to-apples.
    xd = x.contiguous().to(DEV)
    Wd = W.contiguous().to(DEV)
    def one():
        r = cm(xd, Wd)
        out = r[0] if isinstance(r, (list, tuple)) else r
        # force materialization / device sync by copying to host
        return out.detach().to("cpu")
    for _ in range(warmup):
        one()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        one()
        ts.append((time.perf_counter() - t0) * 1e3)  # ms
    ts.sort()
    mean = sum(ts) / len(ts)
    med  = ts[len(ts) // 2]
    log(f"  [{name:3s}] T={T} I={I} O={O}: mean={mean:.3f} ms  median={med:.3f} ms  "
        f"min={ts[0]:.3f} ms  (n={iters})")
    return dict(mean=mean, median=med, min=ts[0])

def main():
    log("=" * 78)
    log(f"DPE vs VE matmul  dev={DEV}  iters={ITERS}")
    log("=" * 78)

    dpe = build(DPE_YAML)
    ve  = build(VE_YAML)

    # ---------------- VALIDATION ----------------
    log("\n--- VALIDATE vs torch F.linear ---")
    log(" DPE (EinsumByDpe, dn_linear_dpe.yaml):")
    val_shapes = [(128, 512, 2048), (128, 256, 128), (256, 2048, 512)]
    dpe_val = [validate("DPE", dpe, T, I, O) for (T, I, O) in val_shapes]
    log(" VE  (EinsumByVe,  dn_linear.yaml):")
    ve_val  = [validate("VE",  ve,  T, I, O) for (T, I, O) in val_shapes]

    # ---------------- BENCHMARK ----------------
    # Realistic size from the task: x[128,2048] @ W[512,2048] -> [128,512]
    T, I, O = 128, 2048, 512
    log(f"\n--- BENCHMARK  x[{T},{I}] @ W[{O},{I}] -> [{T},{O}]  (steady-state) ---")
    b_dpe = bench("DPE", dpe, T, I, O)
    b_ve  = bench("VE",  ve,  T, I, O)

    sp_mean = b_ve["mean"]   / b_dpe["mean"]
    sp_med  = b_ve["median"] / b_dpe["median"]
    log("\n" + "=" * 78)
    log("RESULT")
    log("=" * 78)
    log(f"  DPE : {b_dpe['mean']:.3f} ms (mean) / {b_dpe['median']:.3f} ms (median)")
    log(f"  VE  : {b_ve['mean']:.3f} ms (mean) / {b_ve['median']:.3f} ms (median)")
    log(f"  SPEEDUP (VE/DPE): {sp_mean:.2f}x (mean)  {sp_med:.2f}x (median)")

    dpe_pure_npu = all(v["dfg"] == 0 for v in dpe_val)
    dpe_ok_1e2   = all(v["ok_2"] for v in dpe_val)
    log(f"  DPE pure-NPU (dfg==0 all shapes): {dpe_pure_npu}")
    log(f"  DPE allclose@1e-2 (all shapes)  : {dpe_ok_1e2}")
    log(f"  total _dfg_inner calls          : {_CALLS['n']}")
    log("=" * 78)

if __name__ == "__main__":
    main()
