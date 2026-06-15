#!/usr/bin/env python3
# Emit minimal single-op probe kernels to find which construct fragments the
# graph ("multiple internal subgraphs"). Reuses the same emitter as gen_chunk.
import io, sys

VAR = {"c": "C", "d": "C", "k": "K", "v": "V"}

def shape_block(dims, indent):
    pad = " " * indent
    out = [f"{pad}DynamicUnlabeledShape:", f"{pad}  inner:", f"{pad}    sizes:"]
    for d in dims:
        out.append(f"{pad}      - Var: {VAR[d]}")
    return "\n".join(out)

def axis_block(label, indent, is_tile):
    pad = " " * indent
    head = f"{pad}- axis:" if is_tile else f"{pad}- tag:"
    return "\n".join([head, f"{pad}    LabelStride:", f"{pad}      label:",
                      f"{pad}        inner: \"{label}\"", f"{pad}      stride: 1",
                      f"{pad}  size:", f"{pad}    Var: {VAR[label]}"])

def read_blocks(dims, base_indent, tile_indent):
    base = [d for d in dims if not d.startswith("tile:")]
    tiles = [d.split(":")[1] for d in dims if d.startswith("tile:")]
    inner = "\n".join(axis_block(d, base_indent, False) for d in base)
    tile = "\n".join(axis_block(t, tile_indent, True) for t in tiles)
    return inner, tile

def emit(tensors, inputs, outputs, ops, path):
    out = io.StringIO(); w = out.write
    w("#naive_yaml\n---\ntensors:\n  inner:\n")
    for tid in sorted(tensors):
        w(f"    {tid}:\n      shape:\n")
        w(shape_block(tensors[tid], 8) + "\n")
        w("      element_type: Float32\n      buffer: []\n      name: \"\"\n      source: Unknown\n      buffer_type: Sram\n")
    w("inputs:\n")
    for i in inputs: w(f"  - {i}\n")
    w("outputs:\n")
    for o in outputs: w(f"  - {o}\n")
    w("operators:\n  operators:\n")
    for idx, opd in enumerate(ops):
        w(f"    {idx}:\n      name: {opd['name']}\n      option:\n        SymTacticKernel:\n          inputs:\n")
        for i in opd["inputs"]: w(f"            - {i}\n")
        w(f"          output: {opd['output']}\n          inner:\n            inner:\n              reads:\n")
        for rdims in opd["reads"]:
            inner_str, tile_str = read_blocks(rdims, 24, 20)
            w("                - input:\n                    shape:\n                      inner:\n")
            w(inner_str + "\n")
            w("                    element_type: Float32\n                  table_lookup: None\n                  has_transmutation: false\n                  subtraction: None\n                  paddings: []\n                  slides: []\n                  strides: []\n")
            if tile_str:
                w("                  tiles:\n"); w(tile_str + "\n")
            else:
                w("                  tiles: []\n")
        w("              ein_ops: ~\n              vector_ops:\n                inputs:\n")
        if opd["unary"] is not None:
            w("                  - 0\n                insts:\n                  - def: 1\n                    expr:\n                      Unary:\n")
            w(f"                        operator: {opd['unary']}\n                        operand:\n                          Tensor: 0\n                    source: \"\"\n")
            last_def = 1
        else:
            w("                  - 0\n                  - 1\n                insts:\n                  - def: 2\n                    expr:\n                      Binary:\n")
            w(f"                        operator: {opd['op']}\n                        lhs:\n                          Tensor: 0\n                        rhs:\n                          Tensor: 1\n                    source: \"\"\n")
            last_def = 2
        if opd["reduce"] is not None:
            w(f"                  - def: {last_def+1}\n                    expr:\n                      Reduce:\n                        operator: LocalReduceAddF\n                        operand:\n")
            w(f"                          Tensor: {last_def}\n                        axes:\n                          Tag:\n                            - inner: \"{opd['reduce']}\"\n                    source: \"\"\n")
        inner_str, _ = read_blocks(opd["write"], 24, 24)
        w("              write:\n                input:\n                  shape:\n                    inner:\n")
        w(inner_str + "\n")
        w("                  element_type: Float32\n                output:\n                  shape:\n                    inner:\n")
        w(inner_str + "\n")
        w("                  element_type: Float32\n                has_transmutation: false\n")
        w(f"            kind: {opd['kind']}\n            sparsity: None\n")
    w(f"  next_operator_index: {len(ops)}\nhidden_outputs: []\n")
    open(path, "w").write(out.getvalue())
    print("wrote", path)

BASE = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/"

# probe A: matrix-matrix matmul qk[c,d] = sum_k q[c,k]*k[d,k]
emit({0: ("c","k"), 1: ("d","k"), 2: ("c","d")},
     [0,1], [2],
     [dict(name="mm", kind="EinsumByVe", unary=None, op="MulF", inputs=[0,1], output=2,
           reads=[("c","k","tile:d"),("d","k","tile:c")], write=("c","d"), reduce="k")],
     BASE + "probe_mm.yaml")

# probe B: Unary Exp on a [c,d] matrix
emit({0: ("c","d"), 1: ("c","d")},
     [0], [1],
     [dict(name="ex", kind="Elementwise", unary="Exp", op=None, inputs=[0], output=1,
           reads=[("c","d")], write=("c","d"), reduce=None)],
     BASE + "probe_exp.yaml")

# probe OUT-branch: decay-exp -> qk matmul -> mask-mul -> out matmul
emit({0:("c","k"),1:("d","k"),2:("d","v"),3:("c","d"),
      5:("c","d"),6:("c","d"),7:("c","d"),8:("c","v")},
     [0,1,2,3], [8],
     [dict(name="decay_exp", kind="Elementwise", unary="Exp", op=None, inputs=[3], output=5,
           reads=[("c","d")], write=("c","d"), reduce=None),
      dict(name="qk", kind="EinsumByVe", unary=None, op="MulF", inputs=[0,1], output=6,
           reads=[("c","k","tile:d"),("d","k","tile:c")], write=("c","d"), reduce="k"),
      dict(name="attn", kind="Elementwise", unary=None, op="MulF", inputs=[6,5], output=7,
           reads=[("c","d"),("c","d")], write=("c","d"), reduce=None),
      dict(name="out", kind="EinsumByVe", unary=None, op="MulF", inputs=[7,2], output=8,
           reads=[("c","d","tile:v"),("d","v","tile:c")], write=("c","v"), reduce="d")],
     BASE + "probe_outbranch.yaml")

# probe S-branch: wk-exp -> kdecay tiled-mul -> state matmul
emit({1:("d","k"),2:("d","v"),4:("d",),9:("d",),10:("d","k"),11:("k","v")},
     [1,2,4], [11],
     [dict(name="wk_exp", kind="Elementwise", unary="Exp", op=None, inputs=[4], output=9,
           reads=[("d",)], write=("d",), reduce=None),
      dict(name="kdecay", kind="EinsumByVe", unary=None, op="MulF", inputs=[9,1], output=10,
           reads=[("d","tile:k"),("d","k")], write=("d","k"), reduce=None),
      dict(name="state", kind="EinsumByVe", unary=None, op="MulF", inputs=[10,2], output=11,
           reads=[("d","k","tile:v"),("d","v","tile:k")], write=("k","v"), reduce="d")],
     BASE + "probe_sbranch.yaml")

# probe S-branch NO-EXP: wk given directly as input -> kdecay -> state matmul
emit({1:("d","k"),2:("d","v"),9:("d",),10:("d","k"),11:("k","v")},
     [9,1,2], [11],
     [dict(name="kdecay", kind="EinsumByVe", unary=None, op="MulF", inputs=[9,1], output=10,
           reads=[("d","tile:k"),("d","k")], write=("d","k"), reduce=None),
      dict(name="state", kind="EinsumByVe", unary=None, op="MulF", inputs=[10,2], output=11,
           reads=[("d","k","tile:v"),("d","v","tile:k")], write=("k","v"), reduce="d")],
     BASE + "probe_sbranch_noexp.yaml")

# probe S state-matmul ALONE: state[k,v] = sum_d kdecay[d,k]*v[d,v]
emit({10:("d","k"),2:("d","v"),11:("k","v")},
     [10,2], [11],
     [dict(name="state", kind="EinsumByVe", unary=None, op="MulF", inputs=[10,2], output=11,
           reads=[("d","k","tile:v"),("d","v","tile:k")], write=("k","v"), reduce="d")],
     BASE + "probe_state.yaml")

# probe S-branch v2: wlog2d[d,k] input -> Exp (2D) -> kdecay (matching-shape Elementwise) -> state matmul
emit({1:("d","k"),2:("d","v"),12:("d","k"),9:("d","k"),10:("d","k"),11:("k","v")},
     [12,1,2], [11],
     [dict(name="wk_exp2d", kind="Elementwise", unary="Exp", op=None, inputs=[12], output=9,
           reads=[("d","k")], write=("d","k"), reduce=None),
      dict(name="kdecay", kind="Elementwise", unary=None, op="MulF", inputs=[9,1], output=10,
           reads=[("d","k"),("d","k")], write=("d","k"), reduce=None),
      dict(name="state", kind="EinsumByVe", unary=None, op="MulF", inputs=[10,2], output=11,
           reads=[("d","k","tile:v"),("d","v","tile:k")], write=("k","v"), reduce="d")],
     BASE + "probe_sbranch2.yaml")
