#!/usr/bin/env python3
# Probe which Unary operator NAMES the TK-graph DSL accepts and lower to NPU.
# Emits a 1-op Elementwise kernel y[m,n] = Unary(x[m,n]) for each candidate name,
# compiles+runs on rngd, reports compile/run/dfg_inner.
import os, traceback
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
import torch
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule
import furiosa.torch.custom_ops.dfg as dfgmod

DEV = os.environ.get("RNGD_DEV", "rngd:0")

_orig = dfgmod._dfg_inner
STATE = {"n": 0}
def _spy(*a, **k):
    STATE["n"] += 1
    return _orig(*a, **k)
dfgmod._dfg_inner = _spy

UNARY_TMPL = """#naive_yaml
---
tensors:
  inner:
    0:
      shape:
        DynamicUnlabeledShape:
          inner:
            sizes:
              - Var: M
              - Var: N
      element_type: Float32
      buffer: []
      name: ""
      source: Unknown
      buffer_type: Sram
    1:
      shape:
        DynamicUnlabeledShape:
          inner:
            sizes:
              - Var: M
              - Var: N
      element_type: Float32
      buffer: []
      name: ""
      source: Unknown
      buffer_type: Sram
inputs:
  - 0
outputs:
  - 1
operators:
  operators:
    0:
      name: probe_unary
      option:
        SymTacticKernel:
          inputs:
            - 0
          output: 1
          inner:
            inner:
              reads:
                - input:
                    shape:
                      inner:
                        - tag:
                            LabelStride:
                              label:
                                inner: "0"
                              stride: 1
                          size:
                            Var: M
                        - tag:
                            LabelStride:
                              label:
                                inner: "1"
                              stride: 1
                          size:
                            Var: N
                    element_type: Float32
                  table_lookup: None
                  has_transmutation: false
                  subtraction: None
                  paddings: []
                  slides: []
                  strides: []
                  tiles: []
              ein_ops: ~
              vector_ops:
                inputs:
                  - 0
                insts:
                  - def: 1
                    expr:
                      Unary:
                        operator: __OP__
                        operand:
                          Tensor: 0
                    source: ""
              write:
                input:
                  shape:
                    inner:
                      - tag:
                          LabelStride:
                            label:
                              inner: "0"
                            stride: 1
                        size:
                          Var: M
                      - tag:
                          LabelStride:
                            label:
                              inner: "1"
                            stride: 1
                        size:
                          Var: N
                  element_type: Float32
                output:
                  shape:
                    inner:
                      - tag:
                          LabelStride:
                            label:
                              inner: "0"
                            stride: 1
                        size:
                          Var: M
                      - tag:
                          LabelStride:
                            label:
                              inner: "1"
                            stride: 1
                        size:
                          Var: N
                  element_type: Float32
                has_transmutation: false
            kind: Elementwise
            sparsity: None
  next_operator_index: 1
hidden_outputs: []
"""

# candidate unary names to try (PascalCase serde variants)
CANDS = ["Exp", "Sigmoid", "Rsqrt", "RSqrt", "Sqrt", "Reciprocal", "Recip",
         "Silu", "SiLU", "Tanh", "Gelu", "Square", "Abs", "Neg", "Identity",
         "Sin", "Cos", "Log", "Ln"]

x = torch.rand(2, 3, dtype=torch.float32) + 0.5  # positive for sqrt/rsqrt/log

def torch_ref(name, x):
    import math
    if name == "Exp": return torch.exp(x)
    if name == "Sigmoid": return torch.sigmoid(x)
    if name in ("Rsqrt", "RSqrt"): return torch.rsqrt(x)
    if name == "Sqrt": return torch.sqrt(x)
    if name in ("Reciprocal", "Recip"): return torch.reciprocal(x)
    if name in ("Silu", "SiLU"): return torch.nn.functional.silu(x)
    if name == "Tanh": return torch.tanh(x)
    if name == "Gelu": return torch.nn.functional.gelu(x)
    if name == "Square": return x * x
    if name == "Abs": return x.abs()
    if name == "Neg": return -x
    if name == "Identity": return x
    if name == "Sin": return torch.sin(x)
    if name == "Cos": return torch.cos(x)
    if name in ("Log", "Ln"): return torch.log(x)
    return None

print("DEVICE", DEV)
results = {}
for name in CANDS:
    STATE["n"] = 0
    dsl = UNARY_TMPL.replace("__OP__", name)
    status = "?"
    try:
        m = TacticKernelModule(dsl)
        cm = torch.compile(m, backend=ft.backend)
        out = cm(x.to(DEV))
        if isinstance(out, (tuple, list)):
            out = out[0]
        oc = out.detach().to("cpu").float()
        ref = torch_ref(name, x)
        ok = torch.allclose(oc, ref, atol=1e-3) if ref is not None else None
        maxerr = (oc - ref).abs().max().item() if ref is not None else -1
        status = f"COMPILED+RAN ok={ok} maxerr={maxerr:.2e} dfg_inner={STATE['n']}"
    except Exception as e:
        msg = str(e).splitlines()
        # grab the most informative line
        info = " | ".join([l for l in msg if l.strip()][:3])
        status = f"FAIL {type(e).__name__}: {info[:240]}"
    results[name] = status
    print(f"  {name:12s} -> {status}")

print("\n==== UNARY PROBE SUMMARY ====")
for k, v in results.items():
    print(f"{k:12s}: {v}")
