#!/usr/bin/env python3
# Generate dn_prefill_unroll4.yaml : the DeltaNet recurrent step unrolled T=4
# times in ONE DFG (single EDF). Chains S0 (zero init given as input) through
# 4 copies of the 7-op step body. Per-timestep inputs:
#   q_t[K], k_t[K], v_t[V], beta_full_t[V], decay_full_t[K,V]
# Outputs: out_0[V], out_1[V], out_2[V], out_3[V]  (and final S).
#
# Body op pattern (per t), feeding S_in -> S_out:
#   op a Elementwise MulF : S1   = S_in * decay_t        (S_in, decay_t -> S1)
#   op b EinsumByVe ctr   : kv   = sum_k S1*k_t          (k_t[tiled v], S1 -> kv)
#   op c Elementwise SubF : tmp  = v_t - kv              (v_t, kv -> tmp)
#   op d Elementwise MulF : delta= tmp * beta_t          (tmp, beta_t -> delta)
#   op e EinsumByVe outer : outer= k_t (x) delta         (k_t[tiled v], delta[tiled k] -> outer)
#   op f Elementwise AddF : S_out= S1 + outer            (S1, outer -> S_out)
#   op g EinsumByVe ctr   : out_t= sum_k S_out*q_t       (q_t[tiled v], S_out -> out_t)
import io, sys

# T (unroll depth) and output path are overridable via argv:
#   python gen_unroll.py [T] [out_path]
T = int(sys.argv[1]) if len(sys.argv) > 1 else 4
_OUT_OVERRIDE = sys.argv[2] if len(sys.argv) > 2 else None
K = 4
V = 4

# ---------- tensor id allocation ----------
# inputs: S0=0, then per t (t=0..3): q,k,v,beta,decay  -> 5 ids each
#   t: q=1+5t, k=2+5t, v=3+5t, beta=4+5t, decay=5+5t
# So input ids 0..20 (1 + 5*4 = 21 inputs).
S0 = 0
def q_id(t):    return 1 + 5*t
def k_id(t):    return 2 + 5*t
def v_id(t):    return 3 + 5*t
def beta_id(t): return 4 + 5*t
def decay_id(t):return 5 + 5*t
N_IN = 1 + 5*T  # = 21

# intermediates + outputs: per t we create S1,kv,tmp,delta,outer,Sout,out_t (7 ids)
# start after inputs
_next = [N_IN]
def alloc():
    i = _next[0]; _next[0]+=1; return i

# we thread S: S_in for t=0 is S0; for t>0 it's Sout of previous step.
tensors = {}  # id -> ("K","V") or ("V",) or ("K",) shape tuple of var labels
# declare inputs
tensors[S0] = ("K","V")
for t in range(T):
    tensors[q_id(t)] = ("K",)
    tensors[k_id(t)] = ("K",)
    tensors[v_id(t)] = ("V",)
    tensors[beta_id(t)] = ("V",)
    tensors[decay_id(t)] = ("K","V")

ops = []   # list of dicts describing each operator
out_ids = []   # per-t out tensor id
S_in = S0
final_S = None
for t in range(T):
    S1   = alloc(); tensors[S1]   = ("K","V")
    kv   = alloc(); tensors[kv]   = ("V",)
    tmp  = alloc(); tensors[tmp]  = ("V",)
    delta= alloc(); tensors[delta]= ("V",)
    outer= alloc(); tensors[outer]= ("K","V")
    Sout = alloc(); tensors[Sout] = ("K","V")
    out_t= alloc(); tensors[out_t]= ("V",)
    out_ids.append(out_t)
    final_S = Sout

    qt, kt, vt, bt, dt = q_id(t), k_id(t), v_id(t), beta_id(t), decay_id(t)
    # op a: S1 = S_in * decay   (Elementwise MulF) 2D
    ops.append(dict(name=f"t{t}_decay", kind="Elementwise", op="MulF",
                    inputs=[S_in, dt], output=S1,
                    reads=[("K","V"),("K","V")], write=("K","V"), reduce=None))
    # op b: kv = sum_k S1*k  (EinsumByVe ctr): read0=k tiled v, read1=S1
    ops.append(dict(name=f"t{t}_kv", kind="EinsumByVe", op="MulF",
                    inputs=[kt, S1], output=kv,
                    reads=[("K","tile:v"),("K","V")], write=("V",), reduce="k"))
    # op c: tmp = v - kv (SubF) 1D
    ops.append(dict(name=f"t{t}_sub", kind="Elementwise", op="SubF",
                    inputs=[vt, kv], output=tmp,
                    reads=[("V",),("V",)], write=("V",), reduce=None))
    # op d: delta = tmp * beta (MulF) 1D
    ops.append(dict(name=f"t{t}_mulbeta", kind="Elementwise", op="MulF",
                    inputs=[tmp, bt], output=delta,
                    reads=[("V",),("V",)], write=("V",), reduce=None))
    # op e: outer = k (x) delta (EinsumByVe outer, no reduce): read0=k tiled v, read1=delta tiled k
    ops.append(dict(name=f"t{t}_outer", kind="EinsumByVe", op="MulF",
                    inputs=[kt, delta], output=outer,
                    reads=[("K","tile:v"),("V","tile:k")], write=("K","V"), reduce=None))
    # op f: Sout = S1 + outer (AddF) 2D
    ops.append(dict(name=f"t{t}_addstate", kind="Elementwise", op="AddF",
                    inputs=[S1, outer], output=Sout,
                    reads=[("K","V"),("K","V")], write=("K","V"), reduce=None))
    # op g: out_t = sum_k Sout*q (EinsumByVe ctr): read0=q tiled v, read1=Sout
    ops.append(dict(name=f"t{t}_out", kind="EinsumByVe", op="MulF",
                    inputs=[qt, Sout], output=out_t,
                    reads=[("K","tile:v"),("K","V")], write=("V",), reduce="k"))
    S_in = Sout  # thread state

# outputs: all out_t plus final S
outputs = out_ids + [final_S]

# ---------- emit YAML ----------
# Axis LABELS are lowercase "k"/"v" everywhere (matching dn_step.yaml, where
# the Reduce axes Tag references "k"); their SIZE Vars are uppercase K/V.
LABEL = {"K":"k", "V":"v", "k":"k", "v":"v"}   # dim-key -> axis label
VAR   = {"K":"K", "V":"V", "k":"K", "v":"V"}   # dim-key -> size Var
def shape_block(dims, indent):
    """dims: tuple of var-labels e.g. ('K','V'); emit DynamicUnlabeledShape sizes."""
    pad = " "*indent
    lines = [f"{pad}DynamicUnlabeledShape:",
             f"{pad}  inner:",
             f"{pad}    sizes:"]
    for d in dims:
        lines.append(f"{pad}      - Var: {d}")
    return "\n".join(lines)

def axis_block(label, size_var, indent, is_tile):
    pad = " "*indent
    if is_tile:
        return "\n".join([
            f"{pad}- axis:",
            f"{pad}    LabelStride:",
            f"{pad}      label:",
            f"{pad}        inner: \"{label}\"",
            f"{pad}      stride: 1",
            f"{pad}  size:",
            f"{pad}    Var: {size_var}",
        ])
    else:
        return "\n".join([
            f"{pad}- tag:",
            f"{pad}    LabelStride:",
            f"{pad}      label:",
            f"{pad}        inner: \"{label}\"",
            f"{pad}      stride: 1",
            f"{pad}  size:",
            f"{pad}    Var: {size_var}",
        ])

def read_shape_inner(dims, base_indent, tile_indent):
    """dims like ('K','V') or ('K','tile:v'); returns the non-tile axes as shape.inner tags
       (at base_indent) and the tile axes (at tile_indent). Returns (inner_str, tiles_str)."""
    base = [d for d in dims if not d.startswith("tile:")]
    tiles = [d.split(":")[1] for d in dims if d.startswith("tile:")]
    inner_lines = []
    for d in base:
        inner_lines.append(axis_block(LABEL[d], VAR[d], base_indent, is_tile=False))
    tile_lines = []
    for tl in tiles:
        tile_lines.append(axis_block(LABEL[tl], VAR[tl], tile_indent, is_tile=True))
    return "\n".join(inner_lines), "\n".join(tile_lines)

out = io.StringIO()
w = out.write
w("#naive_yaml\n")
w("# AUTO-GENERATED by gen_unroll.py : DeltaNet step unrolled T=%d in ONE DFG.\n" % T)
w("# Single-forward prefill kernel. Inputs: S0[K,V] + per-t (q,k,v,beta,decay).\n")
w("# Outputs: out_0..out_%d [V] and final S[K,V].\n" % (T-1))
w("---\n")
w("tensors:\n  inner:\n")
for tid in sorted(tensors):
    dims = tensors[tid]
    w(f"    {tid}:\n")
    w("      shape:\n")
    w(shape_block(dims, 8) + "\n")
    w("      element_type: Float32\n")
    w("      buffer: []\n")
    w("      name: \"\"\n")
    w("      source: Unknown\n")
    w("      buffer_type: Sram\n")
w("inputs:\n")
for i in range(N_IN):
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
        inner_str, tile_str = read_shape_inner(rdims, base_indent=24, tile_indent=20)
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
    w("                  - 0\n")
    w("                  - 1\n")
    w("                insts:\n")
    # inst producing the binary product into local 2
    w("                  - def: 2\n")
    w("                    expr:\n")
    w("                      Binary:\n")
    w(f"                        operator: {opd['op']}\n")
    w("                        lhs:\n")
    w("                          Tensor: 0\n")
    w("                        rhs:\n")
    w("                          Tensor: 1\n")
    w("                    source: \"\"\n")
    if opd["reduce"] is not None:
        w("                  - def: 3\n")
        w("                    expr:\n")
        w("                      Reduce:\n")
        w("                        operator: LocalReduceAddF\n")
        w("                        operand:\n")
        w("                          Tensor: 2\n")
        w("                        axes:\n")
        w("                          Tag:\n")
        w(f"                            - inner: \"{opd['reduce']}\"\n")
        w("                    source: \"\"\n")
    # write block
    inner_str, _ = read_shape_inner(opd["write"], base_indent=24, tile_indent=24)
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

path = _OUT_OVERRIDE or ("/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/dn_prefill_unroll%d.yaml" % T)
with open(path, "w") as f:
    f.write(out.getvalue())
print("wrote", path)
print("N_IN =", N_IN, " n_ops =", len(ops), " outputs =", outputs)
print("input id map per t: q,k,v,beta,decay")
for t in range(T):
    print(f"  t{t}: q={q_id(t)} k={k_id(t)} v={v_id(t)} beta={beta_id(t)} decay={decay_id(t)}")
