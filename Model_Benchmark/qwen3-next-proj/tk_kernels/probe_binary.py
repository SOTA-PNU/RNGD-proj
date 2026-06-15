#!/usr/bin/env python3
# Probe which Binary operator NAMES the TK-graph DSL accepts + lower to NPU.
import os
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

BIN_TMPL = """#naive_yaml
---
tensors:
  inner:
    0:
      shape: {DynamicUnlabeledShape: {inner: {sizes: [{Var: M}, {Var: N}]}}}
      element_type: Float32
      buffer: []
      name: ""
      source: Unknown
      buffer_type: Sram
    1:
      shape: {DynamicUnlabeledShape: {inner: {sizes: [{Var: M}, {Var: N}]}}}
      element_type: Float32
      buffer: []
      name: ""
      source: Unknown
      buffer_type: Sram
    2:
      shape: {DynamicUnlabeledShape: {inner: {sizes: [{Var: M}, {Var: N}]}}}
      element_type: Float32
      buffer: []
      name: ""
      source: Unknown
      buffer_type: Sram
inputs: [0, 1]
outputs: [2]
operators:
  operators:
    0:
      name: probe_binary
      option:
        SymTacticKernel:
          inputs: [0, 1]
          output: 2
          inner:
            inner:
              reads:
                - input:
                    shape: {inner: [{tag: {LabelStride: {label: {inner: "0"}, stride: 1}}, size: {Var: M}}, {tag: {LabelStride: {label: {inner: "1"}, stride: 1}}, size: {Var: N}}]}
                    element_type: Float32
                  table_lookup: None
                  has_transmutation: false
                  subtraction: None
                  paddings: []
                  slides: []
                  strides: []
                  tiles: []
                - input:
                    shape: {inner: [{tag: {LabelStride: {label: {inner: "0"}, stride: 1}}, size: {Var: M}}, {tag: {LabelStride: {label: {inner: "1"}, stride: 1}}, size: {Var: N}}]}
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
                inputs: [0, 1]
                insts:
                  - def: 2
                    expr:
                      Binary:
                        operator: __OP__
                        lhs: {Tensor: 0}
                        rhs: {Tensor: 1}
                    source: ""
              write:
                input:
                  shape: {inner: [{tag: {LabelStride: {label: {inner: "0"}, stride: 1}}, size: {Var: M}}, {tag: {LabelStride: {label: {inner: "1"}, stride: 1}}, size: {Var: N}}]}
                  element_type: Float32
                output:
                  shape: {inner: [{tag: {LabelStride: {label: {inner: "0"}, stride: 1}}, size: {Var: M}}, {tag: {LabelStride: {label: {inner: "1"}, stride: 1}}, size: {Var: N}}]}
                  element_type: Float32
                has_transmutation: false
            kind: Elementwise
            sparsity: None
  next_operator_index: 1
hidden_outputs: []
"""

CANDS = ["MulF", "AddF", "SubF", "DivF", "MaxF", "MinF", "PowF", "Div", "RDivF", "DivideF"]
a = torch.rand(2, 3, dtype=torch.float32) + 1.0
b = torch.rand(2, 3, dtype=torch.float32) + 1.0

def ref(name, a, b):
    if name == "MulF": return a * b
    if name == "AddF": return a + b
    if name == "SubF": return a - b
    if name in ("DivF", "Div", "DivideF"): return a / b
    if name == "RDivF": return b / a
    if name == "MaxF": return torch.maximum(a, b)
    if name == "MinF": return torch.minimum(a, b)
    if name == "PowF": return a ** b
    return None

print("DEVICE", DEV)
for name in CANDS:
    STATE["n"] = 0
    dsl = BIN_TMPL.replace("__OP__", name)
    try:
        m = TacticKernelModule(dsl)
        cm = torch.compile(m, backend=ft.backend)
        out = cm(a.to(DEV), b.to(DEV))
        if isinstance(out, (tuple, list)): out = out[0]
        oc = out.detach().to("cpu").float()
        r = ref(name, a, b)
        ok = torch.allclose(oc, r, atol=1e-3) if r is not None else None
        me = (oc - r).abs().max().item() if r is not None else -1
        print(f"  {name:10s} -> COMPILED+RAN ok={ok} maxerr={me:.2e} dfg_inner={STATE['n']}")
    except Exception as e:
        info = " | ".join([l for l in str(e).splitlines() if l.strip()][:2])
        print(f"  {name:10s} -> FAIL {type(e).__name__}: {info[:160]}")
