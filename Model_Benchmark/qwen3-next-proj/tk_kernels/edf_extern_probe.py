"""REAL EDF artifact attempt — probe whether furiosa.native_torch.compiler.compile()
can lower DeltaNet-style standalone-elementwise graphs to a serializable EDF blob,
with allow_external_operators / allow_unlowered_operators toggled.

Question being answered:
  (A) Does compiler.compile(ep) -> ir.Edf succeed for the recurrent gated-delta body?
  (B) Does allow_external_operators=True change the kernelizer break?
  (C) Is the produced ir.Edf serializable (blob.serialize() bytes)? Is that the same
      thing the artifact binary_bundle stores (CompiledGraph) — or a different type?
"""
import os, sys, traceback, time
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
import torch
import torch.nn as nn
import furiosa.torch as ft  # noqa
from furiosa.native_torch import compiler


def cfg(**kw):
    c = compiler.Config()
    for k, v in kw.items():
        try:
            setattr(c, k, v)
        except Exception as e:
            print(f"   (could not set {k}={v}: {e})")
    return c


def try_compile(label, mod, args, config):
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    t0 = time.time()
    try:
        ep = torch.export.export(mod, args, strict=False)
    except Exception as e:
        print(f"  EXPORT FAILED: {type(e).__name__}: {str(e)[:300]}")
        return ("export_fail", str(e)[:300])
    try:
        edf = compiler.compile(ep, compiler_config=config)
    except Exception as e:
        msg = str(e)
        print(f"  COMPILE FAILED after {time.time()-t0:.1f}s: {type(e).__name__}")
        # print the most informative tail of the error
        for ln in msg.splitlines()[-12:]:
            print("   |", ln[:200])
        return ("compile_fail", type(e).__name__ + ": " + msg[:500])
    dt = time.time() - t0
    print(f"  COMPILE OK in {dt:.1f}s -> type={type(edf).__name__}")
    try:
        blob = edf.serialize()
        print(f"  SERIALIZE OK: {len(blob)} bytes, head={blob[:8]!r}")
    except Exception as e:
        print(f"  SERIALIZE FAILED: {e}")
        blob = None
    try:
        print(f"  npu_node present: {edf.npu_node is not None}")
    except Exception as e:
        print(f"  npu_node check failed: {e}")
    return ("ok", len(blob) if blob else 0)


# ---- Module 1: a single recurrence step (pure standalone elementwise + reduce) ----
class DeltaStep(nn.Module):
    def forward(self, state, q_t, k_t, v_t, g_t, beta_t):
        # mirrors _GatedDeltaNetCore inner loop (one step), all standalone elementwise
        g_t = g_t.exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta_t.unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out_t = (state * q_t.unsqueeze(-1)).sum(dim=-2)
        return state, out_t


# ---- Module 2: just the gate computation (sigmoid/exp/log standalone) ----
class GateOnly(nn.Module):
    def __init__(self, nv):
        super().__init__()
        self.A_log = nn.Parameter(torch.randn(nv))
        self.dt_bias = nn.Parameter(torch.randn(nv))

    def forward(self, a_):  # a_: (b, s, nv)
        beta = a_.transpose(1, 2).sigmoid().float()
        _sp_in = a_.transpose(1, 2).float() + self.dt_bias.unsqueeze(-1)
        g = -torch.exp(self.A_log.float()).unsqueeze(-1) * torch.log(torch.exp(_sp_in) + 1.0)
        return beta, g


# ---- Module 3: a plain matmul (control — must succeed) ----
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

    configs = [
        ("default", cfg()),
        ("allow_external+unlowered", cfg(allow_external_operators=True, allow_unlowered_operators=True)),
    ]

    results = {}
    for cname, c in configs:
        results[("MM(control)", cname)] = try_compile(f"MM control [{cname}]", MM(), (ma, mb), c)
        results[("GateOnly", cname)] = try_compile(f"GateOnly (sigmoid/exp/log) [{cname}]", GateOnly(nv), (a_,), c)
        results[("DeltaStep", cname)] = try_compile(f"DeltaStep (recurrent body) [{cname}]", DeltaStep(), (state, q_t, k_t, v_t, g_t, beta_t), c)

    print(f"\n{'#'*72}\nSUMMARY\n{'#'*72}")
    for (mod, cname), (status, info) in results.items():
        print(f"  {mod:16s} | {cname:26s} -> {status:14s} | {str(info)[:80]}")


if __name__ == "__main__":
    main()
