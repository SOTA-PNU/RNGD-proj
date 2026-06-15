#!/usr/bin/env python3
# Generate dn_chunk.yaml : ONE CHUNK of the chunk-parallel gated delta rule,
# the MATMUL-heavy intra-chunk computation, as a single fp32 TK-graph DFG.
#
# Mirrors HF torch_chunk_gated_delta_rule (transformers qwen3_next L467-544),
# SIMPLIFIED to a single chunk with initial_state = 0 (so attn_inter = 0,
# v_prime = 0, v_new = v_i) and WITHOUT the triangular-inverse refinement loop
# (L511-515). This is the "basic chunked scan" the task explicitly permits.
#
# Axes:  c  = query position within chunk (size C)   [HF: 1st chunk axis]
#        c2 = key   position within chunk (size C)    [HF: 2nd chunk axis, "d"]
#        k  = key/query head dim (size K = d_k)
#        v  = value head dim (size V = d_v)
#
# On-NPU computation (initial_state=0):
#   op0 decay_mask[c,c2] = exp(gdiff[c,c2])             Unary Exp
#         gdiff = (g_cum[c]-g_cum[c2]) with upper(c2>c) preset to -BIG so exp->0.
#         (this single Exp authors BOTH the decay-mask exp() AND the causal mask)
#   op1 qk[c,c2]   = sum_k q[c,k] * k[c2,k]             EinsumByVe MATMUL (reduce k)
#   op2 attn[c,c2] = qk[c,c2] * decay_mask[c,c2]        Elementwise MulF
#   op3 out[c,v]   = sum_c2 attn[c,c2] * v[c2,v]        EinsumByVe MATMUL (reduce c2)
#   op4 wk[c2,k]   = exp(wlog2d[c2,k])                  Unary Exp (2D)
#         wlog2d = (g_cum[last]-g_cum[c2]) broadcast over k  (per-row state-decay log)
#   op5 kdecay[c2,k] = wk[c2,k] * k[c2,k]               Elementwise MulF (matching shapes)
#   op6 S[k,v]     = sum_c2 kdecay[c2,k] * v[c2,v]      EinsumByVe MATMUL (reduce c2)
#
# NOTE (frontier found on rngd): the EARLIER design used wk as a 1D [c2] vector and
# kdecay = EinsumByVe tiled-mul (read0 = wk[c2] broadcast over k, read1 = k[c2,k]
# full 2D, NO reduce). The EDF backend places THAT specific op on a Cpu node
# ("unsupported EDF node: Cpu(... kind: EinsumByVe ... no reduce, one full-2D +
# one broadcast-1D read")). A no-reduce EinsumByVe lowers to NPU only when BOTH
# reads are broadcast 1D (true outer product, dn_rank1) -- NOT when one read is
# already full-rank over a surviving axis. Fix: pass the state-decay log as a 2D
# [c2,k] tensor (caller broadcasts), Exp it on NPU (2D Unary, proven), then use a
# plain Elementwise MulF with MATCHING shapes (proven, dn_step op0). This keeps
# the exp() authored on the NPU and uses only EDF-lowerable op shapes.
#
# Outputs: out[c,v]  (chunk's core_attn_out)  and  S[k,v] (chunk's state contribution).
#
# The 3 contractions (op1, op3, op6) are the matmul CORE that MUST run on NPU.
# op1 is the matrix-MATRIX einsum 'ck,dk->cd' (both surviving axes from different
# operands, reduce over shared k); the others are 'cd,dv->cv' / 'dk,dv->kv'.
#
# fp32 ops: Binary MulF (Binary), Unary Exp, reduction LocalReduceAddF.
# EinsumByVe broadcast read MUST be read0. A matrix-matrix einsum needs BOTH
# reads broadcast/tiled along the OTHER operand's surviving axis.
import io, sys

# ---- tensor id allocation ----
# inputs
Q, KK, VV, GDIFF, WLOG = 0, 1, 2, 3, 4
N_IN = 5
# intermediates / outputs
DECAY  = 5   # [c,c2]
QK     = 6   # [c,c2]
ATTN   = 7   # [c,c2]
OUT    = 8   # [c,v]   graph output
WK     = 9   # [c2]
KDECAY = 10  # [c2,k]
S      = 11  # [k,v]   graph output

# shape tuples by var-label key. "c"->C, "d"->C(=c2 axis), "k"->K, "v"->V
tensors = {
    Q:      ("c", "k"),
    KK:     ("d", "k"),
    VV:     ("d", "v"),
    GDIFF:  ("c", "d"),
    WLOG:   ("d", "k"),   # state-decay log, broadcast to 2D by caller
    DECAY:  ("c", "d"),
    QK:     ("c", "d"),
    ATTN:   ("c", "d"),
    OUT:    ("c", "v"),
    WK:     ("d", "k"),   # exp(WLOG)
    KDECAY: ("d", "k"),
    S:      ("k", "v"),
}
inputs  = [Q, KK, VV, GDIFF, WLOG]
outputs = [OUT, S]

# Each op:
#   kind: Elementwise | EinsumByVe
#   unary: None | "Exp"      (Unary op, single input)
#   op:    binary op name (MulF/...) when not unary
#   inputs: tensor ids
#   output: tensor id
#   reads: list of dim tuples; a dim "tile:X" means broadcast/tiled along axis X
#   write: dim tuple
#   reduce: None | reduced axis label (single)
ops = []
# op0: decay_mask = Exp(gdiff)         Unary Exp, [c,d]
ops.append(dict(name="chunk_decay_exp", kind="Elementwise", unary="Exp", op=None,
                inputs=[GDIFF], output=DECAY,
                reads=[("c", "d")], write=("c", "d"), reduce=None))
# op1: qk[c,d] = sum_k q[c,k]*k[d,k]   EinsumByVe matmul, reduce k
#   read0 = q[c,k] tiled along new axis d  -> [c,k] real + tile d
#   read1 = k[d,k] tiled along new axis c  -> [d,k] real + tile c
ops.append(dict(name="chunk_qk_matmul", kind="EinsumByVe", unary=None, op="MulF",
                inputs=[Q, KK], output=QK,
                reads=[("c", "k", "tile:d"), ("d", "k", "tile:c")],
                write=("c", "d"), reduce="k"))
# op2: attn = qk * decay_mask          Elementwise MulF, [c,d]
ops.append(dict(name="chunk_attn_maskmul", kind="Elementwise", unary=None, op="MulF",
                inputs=[QK, DECAY], output=ATTN,
                reads=[("c", "d"), ("c", "d")], write=("c", "d"), reduce=None))
# op3: out[c,v] = sum_d attn[c,d]*v[d,v]   EinsumByVe matmul, reduce d
#   read0 = attn[c,d] tiled along new axis v -> [c,d] real + tile v
#   read1 = v[d,v] tiled along new axis c    -> [d,v] real + tile c
ops.append(dict(name="chunk_out_matmul", kind="EinsumByVe", unary=None, op="MulF",
                inputs=[ATTN, VV], output=OUT,
                reads=[("c", "d", "tile:v"), ("d", "v", "tile:c")],
                write=("c", "v"), reduce="d"))
# op4: wk[d,k] = Exp(wlog2d[d,k])      Unary Exp, [d,k]
ops.append(dict(name="chunk_wk_exp", kind="Elementwise", unary="Exp", op=None,
                inputs=[WLOG], output=WK,
                reads=[("d", "k")], write=("d", "k"), reduce=None))
# op5: kdecay[d,k] = wk[d,k]*k[d,k]    Elementwise MulF, MATCHING shapes
ops.append(dict(name="chunk_kdecay_mul", kind="Elementwise", unary=None, op="MulF",
                inputs=[WK, KK], output=KDECAY,
                reads=[("d", "k"), ("d", "k")], write=("d", "k"), reduce=None))
# op6: S[k,v] = sum_d kdecay[d,k]*v[d,v]   EinsumByVe matmul, reduce d
#   read0 = kdecay[d,k] tiled along new axis v -> [d,k] real + tile v
#   read1 = v[d,v] tiled along new axis k      -> [d,v] real + tile k
ops.append(dict(name="chunk_state_matmul", kind="EinsumByVe", unary=None, op="MulF",
                inputs=[KDECAY, VV], output=S,
                reads=[("d", "k", "tile:v"), ("d", "v", "tile:k")],
                write=("k", "v"), reduce="d"))

# ---- emit YAML ----
# axis label -> size Var. c and d are BOTH size C (chunk size); use Var C for both.
VAR = {"c": "C", "d": "C", "k": "K", "v": "V"}

def shape_block(dims, indent):
    pad = " " * indent
    lines = [f"{pad}DynamicUnlabeledShape:",
             f"{pad}  inner:",
             f"{pad}    sizes:"]
    for d in dims:
        lines.append(f"{pad}      - Var: {VAR[d]}")
    return "\n".join(lines)

def axis_block(label, indent, is_tile):
    pad = " " * indent
    if is_tile:
        head = f"{pad}- axis:"
    else:
        head = f"{pad}- tag:"
    return "\n".join([
        head,
        f"{pad}    LabelStride:",
        f"{pad}      label:",
        f"{pad}        inner: \"{label}\"",
        f"{pad}      stride: 1",
        f"{pad}  size:",
        f"{pad}    Var: {VAR[label]}",
    ])

def read_blocks(dims, base_indent, tile_indent):
    base = [d for d in dims if not d.startswith("tile:")]
    tiles = [d.split(":")[1] for d in dims if d.startswith("tile:")]
    inner = "\n".join(axis_block(d, base_indent, False) for d in base)
    tile = "\n".join(axis_block(t, tile_indent, True) for t in tiles)
    return inner, tile

out = io.StringIO()
w = out.write
w("#naive_yaml\n")
w("# AUTO-GENERATED by gen_chunk.py : ONE CHUNK of chunk-parallel gated delta rule.\n")
w("# Matmul-heavy intra-chunk core (initial_state=0, no tri-inverse refinement).\n")
w("# Inputs: 0=q[c,k] 1=k[d,k] 2=v[d,v] 3=gdiff[c,d] 4=wlog2d[d,k].\n")
w("# Outputs: 8=out[c,v]  11=S[k,v].\n")
w("---\n")
w("tensors:\n  inner:\n")
for tid in sorted(tensors):
    w(f"    {tid}:\n")
    w("      shape:\n")
    w(shape_block(tensors[tid], 8) + "\n")
    w("      element_type: Float32\n")
    w("      buffer: []\n")
    w("      name: \"\"\n")
    w("      source: Unknown\n")
    w("      buffer_type: Sram\n")
w("inputs:\n")
for i in inputs:
    w(f"  - {i}\n")
w("outputs:\n")
for o in outputs:
    w(f"  - {o}\n")
w("operators:\n  operators:\n")
for idx, opd in enumerate(ops):
    w(f"    {idx}:\n")
    w(f"      name: {opd['name']}\n")
    w("      option:\n")
    w("        SymTacticKernel:\n")
    w("          inputs:\n")
    for i in opd["inputs"]:
        w(f"            - {i}\n")
    w(f"          output: {opd['output']}\n")
    w("          inner:\n")
    w("            inner:\n")
    w("              reads:\n")
    for rdims in opd["reads"]:
        inner_str, tile_str = read_blocks(rdims, base_indent=24, tile_indent=20)
        w("                - input:\n")
        w("                    shape:\n")
        w("                      inner:\n")
        w(inner_str + "\n")
        w("                    element_type: Float32\n")
        w("                  table_lookup: None\n")
        w("                  has_transmutation: false\n")
        w("                  subtraction: None\n")
        w("                  paddings: []\n")
        w("                  slides: []\n")
        w("                  strides: []\n")
        if tile_str:
            w("                  tiles:\n")
            w(tile_str + "\n")
        else:
            w("                  tiles: []\n")
    w("              ein_ops: ~\n")
    w("              vector_ops:\n")
    w("                inputs:\n")
    if opd["unary"] is not None:
        # single-input unary
        w("                  - 0\n")
        w("                insts:\n")
        w("                  - def: 1\n")
        w("                    expr:\n")
        w("                      Unary:\n")
        w(f"                        operator: {opd['unary']}\n")
        w("                        operand:\n")
        w("                          Tensor: 0\n")
        w("                    source: \"\"\n")
        last_def = 1
    else:
        w("                  - 0\n")
        w("                  - 1\n")
        w("                insts:\n")
        w("                  - def: 2\n")
        w("                    expr:\n")
        w("                      Binary:\n")
        w(f"                        operator: {opd['op']}\n")
        w("                        lhs:\n")
        w("                          Tensor: 0\n")
        w("                        rhs:\n")
        w("                          Tensor: 1\n")
        w("                    source: \"\"\n")
        last_def = 2
    if opd["reduce"] is not None:
        w(f"                  - def: {last_def + 1}\n")
        w("                    expr:\n")
        w("                      Reduce:\n")
        w("                        operator: LocalReduceAddF\n")
        w("                        operand:\n")
        w(f"                          Tensor: {last_def}\n")
        w("                        axes:\n")
        w("                          Tag:\n")
        w(f"                            - inner: \"{opd['reduce']}\"\n")
        w("                    source: \"\"\n")
    inner_str, _ = read_blocks(opd["write"], base_indent=24, tile_indent=24)
    w("              write:\n")
    w("                input:\n")
    w("                  shape:\n")
    w("                    inner:\n")
    w(inner_str + "\n")
    w("                  element_type: Float32\n")
    w("                output:\n")
    w("                  shape:\n")
    w("                    inner:\n")
    w(inner_str + "\n")
    w("                  element_type: Float32\n")
    w("                has_transmutation: false\n")
    w(f"            kind: {opd['kind']}\n")
    w("            sparsity: None\n")
w(f"  next_operator_index: {len(ops)}\n")
w("hidden_outputs: []\n")

path = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/dn_chunk.yaml"
with open(path, "w") as f:
    f.write(out.getvalue())
print("wrote", path, " n_ops =", len(ops))
