"""REAL EDF attempt v2 — use the PROPER furiosa.torch path (CompileModule.from_module,
which runs furiosa.torch.export.PASSES first, the marker/metadata passes the raw
compiler.compile lacks). This is the same path torch.compile(backend=ft.backend) uses.

Tests:
  - matmul control (must succeed -> proves harness correct)
  - GateOnly (standalone sigmoid/exp/log)
  - DeltaStep (recurrent body, one step)
For each: does it compile to an EdfModule? Is .edf serializable? Does running on NPU work?
"""
import os, sys, traceback, time
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
import torch
import torch.nn as nn
import furiosa.torch as ft
from furiosa.torch.custom_ops.edf import CompileModule
from furiosa.native_torch import compiler


def cfg(**kw):
    c = compiler.Config()
    for k, v in kw.items():
        try:
            setattr(c, k, v)
        except Exception as e:
            print(f"   (could not set {k}={v}: {e})")
    return c


def try_via_backend(label, mod, args, config):
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    t0 = time.time()
    try:
        cm = CompileModule.from_module(mod, args, compiler_config=config)
    except Exception as e:
        msg = str(e)
        print(f"  COMPILE FAILED after {time.time()-t0:.1f}s: {type(e).__name__}")
        for ln in msg.splitlines()[-10:]:
            print("   |", ln[:200])
        return ("compile_fail", type(e).__name__ + ": " + msg[:300])
    dt = time.time() - t0
    print(f"  COMPILE OK in {dt:.1f}s -> {type(cm).__name__}")
    try:
        edf = cm.edf
        blob = edf.serialize()
        print(f"  ir.Edf SERIALIZE OK: {len(blob)} bytes  npu_node={edf.npu_node is not None}")
    except Exception as e:
        print(f"  serialize/edf failed: {e}")
        blob = None
    # try to actually run on NPU
    try:
        dev = torch.device("rngd")
        cm = cm.to(dev)
        npu_args = tuple(a.to(dev) for a in args)
        out = cm(*npu_args)
        print(f"  NPU RUN OK -> out types: {[type(o).__name__ for o in (out if isinstance(out, (list,tuple)) else [out])][:4]}")
    except Exception as e:
        print(f"  NPU RUN FAILED: {type(e).__name__}: {str(e)[:200]}")
    return ("ok", len(blob) if blob else 0)


class GateOnly(nn.Module):
    def __init__(self, nv):
        super().__init__()
        self.A_log = nn.Parameter(torch.randn(nv))
        self.dt_bias = nn.Parameter(torch.randn(nv))

    def forward(self, a_):
        beta = a_.transpose(1, 2).sigmoid().float()
        _sp_in = a_.transpose(1, 2).float() + self.dt_bias.unsqueeze(-1)
        g = -torch.exp(self.A_log.float()).unsqueeze(-1) * torch.log(torch.exp(_sp_in) + 1.0)
        return beta, g


class DeltaStep(nn.Module):
    def forward(self, state, q_t, k_t, v_t, g_t, beta_t):
        g_t = g_t.exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta_t.unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out_t = (state * q_t.unsqueeze(-1)).sum(dim=-2)
        return state, out_t


class MM(nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)


def main():
    torch.manual_seed(0)
    b, nv, dk, dv, s = 1, 8, 64, 64, 16
    state = torch.randn(b, nv, dk, dv)
    q_t = torch.randn(b, nv, dk); k_t = torch.randn(b, nv, dk)
    v_t = torch.randn(b, nv, dv); g_t = torch.randn(b, nv); beta_t = torch.randn(b, nv)
    a_ = torch.randn(b, s, nv)
    ma = torch.randn(128, 256); mb = torch.randn(256, 128)

    configs = [("default", cfg()),
               ("allow_external+unlowered", cfg(allow_external_operators=True, allow_unlowered_operators=True))]
    results = {}
    for cname, c in configs:
        results[("MM(control)", cname)] = try_via_backend(f"MM control [{cname}]", MM(), (ma, mb), c)
        results[("GateOnly", cname)] = try_via_backend(f"GateOnly [{cname}]", GateOnly(nv), (a_,), c)
        results[("DeltaStep", cname)] = try_via_backend(f"DeltaStep [{cname}]", DeltaStep(), (state, q_t, k_t, v_t, g_t, beta_t), c)

    print(f"\n{'#'*72}\nSUMMARY\n{'#'*72}")
    for (mod, cname), (status, info) in results.items():
        print(f"  {mod:16s} | {cname:26s} -> {status:14s} | {str(info)[:80]}")


if __name__ == "__main__":
    main()
