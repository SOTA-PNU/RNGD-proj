#!/usr/bin/env python3
# Generate dn_chunk_full.yaml  AND  dn_chunk_full_mh.yaml.
#
# dn_chunk_full.yaml      : ONE FULL CHUNK of the chunk-parallel gated delta rule
#                           WITH inter-chunk state carry, single head, fp32 DFG.
# dn_chunk_full_mh.yaml   : the SAME 12-op graph but HEAD-BATCHED -- a leading
#                           head-batch axis "h" is prepended (OUTERMOST) to EVERY
#                           tensor shape and EVERY read/write base shape, so all H
#                           value-heads run in ONE NPU dispatch.  Per the proven
#                           batch-axis rule (dn_step_mh.yaml at H=4): the new "h"
#                           label is NEVER tiled and NEVER added to a Reduce inst's
#                           axes; the per-chunk matmuls stay per-head because "h" is
#                           carried in both operands and the output, never reduced.
#                           The contraction reduce axes ("k","d") are unchanged.
#
# Mirrors HF torch_chunk_gated_delta_rule (transformers qwen3_next L467-544),
# the per-chunk body L527-537, for a GENERIC incoming state S_prev (chunk idx>0).
# Unlike dn_chunk.yaml (S_prev=0 => attn_inter=0, v_prime=0) this kernel adds the
# three inter-chunk terms and the across-chunk state update so that, host-looped,
# it reproduces the full multi-chunk scan EXACTLY (verified maxerr 0.0 on CPU).
#
# The triangular-inverse refinement (HF L511-515) and the per-chunk decay_mask /
# beta weighting depend ONLY on within-chunk q,k,v,g (NOT on S_prev), so they are
# precomputed on the HOST and fed as inputs (value=T@v_i, kcd=T@(k_i*exp(g_cum)),
# decay_strict, the gate broadcasts). EVERYTHING that touches S_prev -- i.e. the
# actual chunk-scan recurrence -- runs ON THE NPU here (4 of the matmuls involve
# S_prev or v_new). This keeps the recurrence on-device while staying inside the
# EDF-lowerable op set.
#
# Axes:  c = query position within chunk (size C)
#        d = key   position within chunk (size C)  [same size, distinct label]
#        k = key/query head dim (size K = d_k)
#        v = value head dim (size V = d_v)
#
# Kernel INPUTS:
#   0 q[c,k]         pre-scaled query of the chunk
#   1 kk[d,k]        key of the chunk (k_i)
#   2 value[d,v]     HF "value" = T @ v_i              (host, no S_prev dep)
#   3 decay_strict[c,d] = tril(exp(g_cum[c]-g_cum[d])) with strict-upper = 0  (host)
#   4 kcd[c,k]       HF "k_cumdecay" = T @ (k_i*exp(g_cum))  (host, no S_prev dep)
#   5 gexp_k[c,k]    exp(g_cum[c]) broadcast over k    (host)
#   6 wdecay[d,k]    exp(g_cum[-1]-g_cum[d]) broadcast over k  (host)
#   7 sdecay[k,v]    exp(g_cum[-1]) broadcast to [K,V] (host scalar bcast)
#   8 Sprev[k,v]     incoming recurrent state
#
# Kernel OUTPUTS:
#   17 out[c,v]      chunk core_attn_out = attn_inter + attn @ v_new
#   21 Snext[k,v]    updated state S_prev*exp(g_last) + (k_i*wdecay)^T @ v_new
#
# ON-NPU ops:
#   op0  qk[c,d]        = sum_k q[c,k]*kk[d,k]            EinsumByVe matmul (reduce k)
#   op1  attn[c,d]      = qk[c,d]*decay_strict[c,d]       Elementwise MulF
#   op2  qg[c,k]        = q[c,k]*gexp_k[c,k]              Elementwise MulF
#   op3  attn_inter[c,v]= sum_k qg[c,k]*Sprev[k,v]        EinsumByVe matmul (reduce k)
#   op4  v_prime[d,v]   = sum_k kcd[d,k]*Sprev[k,v]       EinsumByVe matmul (reduce k)
#   op5  v_new[d,v]     = value[d,v]-v_prime[d,v]         Elementwise SubF
#   op6  intra[c,v]     = sum_d attn[c,d]*v_new[d,v]      EinsumByVe matmul (reduce d)
#   op7  out[c,v]       = attn_inter[c,v]+intra[c,v]      Elementwise AddF
#   op8  kdec[d,k]      = kk[d,k]*wdecay[d,k]             Elementwise MulF
#   op9  kv[k,v]        = sum_d kdec[d,k]*v_new[d,v]      EinsumByVe matmul (reduce d)
#   op10 Sdec[k,v]      = Sprev[k,v]*sdecay[k,v]          Elementwise MulF
#   op11 Snext[k,v]     = Sdec[k,v]+kv[k,v]               Elementwise AddF
#
# NOTE on labels: kcd is fed as label [d,k] (NOT [c,k]) so that v_prime lands as
# [d,v] -- the SAME label as `value` -- making op5 a matching-shape Elementwise and
# letting op6/op9 reduce over `d` cleanly (dn_chunk.yaml proven pattern). q for op3
# is the SAME tensor as op0's q but read with the [c,k] label; gexp_k is [c,k].
#
# fp32 ops: Binary MulF/SubF/AddF, reduction LocalReduceAddF. Matrix-matrix einsum
# 'ik,jk->ij' as EinsumByVe: read0 tiled along new axis j, read1 tiled along new
# axis i, MulF then Reduce over shared inner axis. (Proven in dn_chunk.yaml.)
import io

# ---- tensor id allocation ----
Q, KK, VALUE, DECAY, KCD, GEXP, WDECAY, SDECAY, SPREV = range(9)
N_IN = 9
QK         = 9    # [c,d]
ATTN       = 10   # [c,d]
QG         = 11   # [c,k]
ATTN_INTER = 12   # [c,v]
V_PRIME    = 13   # [d,v]
V_NEW      = 14   # [d,v]
INTRA      = 15   # [c,v]
KDEC       = 16   # [d,k]
OUT        = 17   # [c,v]  graph output
KV         = 18   # [k,v]
SDEC       = 19   # [k,v]
SNEXT      = 20   # [k,v]  graph output

# axis-label tuples per tensor
tensors = {
    Q:          ("c", "k"),
    KK:         ("d", "k"),
    VALUE:      ("d", "v"),
    DECAY:      ("c", "d"),
    KCD:        ("d", "k"),
    GEXP:       ("c", "k"),
    WDECAY:     ("d", "k"),
    SDECAY:     ("k", "v"),
    SPREV:      ("k", "v"),
    QK:         ("c", "d"),
    ATTN:       ("c", "d"),
    QG:         ("c", "k"),
    ATTN_INTER: ("c", "v"),
    V_PRIME:    ("d", "v"),
    V_NEW:      ("d", "v"),
    INTRA:      ("c", "v"),
    KDEC:       ("d", "k"),
    OUT:        ("c", "v"),
    KV:         ("k", "v"),
    SDEC:       ("k", "v"),
    SNEXT:      ("k", "v"),
}
inputs  = [Q, KK, VALUE, DECAY, KCD, GEXP, WDECAY, SDECAY, SPREV]
outputs = [OUT, SNEXT]

ops = []
# op0: qk[c,d] = sum_k q[c,k]*kk[d,k]   matmul reduce k
ops.append(dict(name="qk_matmul", kind="EinsumByVe", unary=None, op="MulF",
                inputs=[Q, KK], output=QK,
                reads=[("c", "k", "tile:d"), ("d", "k", "tile:c")],
                write=("c", "d"), reduce="k"))
# op1: attn[c,d] = qk*decay_strict   elementwise
ops.append(dict(name="attn_maskmul", kind="Elementwise", unary=None, op="MulF",
                inputs=[QK, DECAY], output=ATTN,
                reads=[("c", "d"), ("c", "d")], write=("c", "d"), reduce=None))
# op2: qg[c,k] = q*gexp_k   elementwise
ops.append(dict(name="qg_mul", kind="Elementwise", unary=None, op="MulF",
                inputs=[Q, GEXP], output=QG,
                reads=[("c", "k"), ("c", "k")], write=("c", "k"), reduce=None))
# op3: attn_inter[c,v] = sum_k qg[c,k]*Sprev[k,v]   matmul reduce k
ops.append(dict(name="attn_inter_matmul", kind="EinsumByVe", unary=None, op="MulF",
                inputs=[QG, SPREV], output=ATTN_INTER,
                reads=[("c", "k", "tile:v"), ("k", "v", "tile:c")],
                write=("c", "v"), reduce="k"))
# op4: v_prime[d,v] = sum_k kcd[d,k]*Sprev[k,v]   matmul reduce k
ops.append(dict(name="v_prime_matmul", kind="EinsumByVe", unary=None, op="MulF",
                inputs=[KCD, SPREV], output=V_PRIME,
                reads=[("d", "k", "tile:v"), ("k", "v", "tile:d")],
                write=("d", "v"), reduce="k"))
# op5: v_new[d,v] = value[d,v]-v_prime[d,v]   elementwise sub
ops.append(dict(name="v_new_sub", kind="Elementwise", unary=None, op="SubF",
                inputs=[VALUE, V_PRIME], output=V_NEW,
                reads=[("d", "v"), ("d", "v")], write=("d", "v"), reduce=None))
# op6: intra[c,v] = sum_d attn[c,d]*v_new[d,v]   matmul reduce d
ops.append(dict(name="intra_matmul", kind="EinsumByVe", unary=None, op="MulF",
                inputs=[ATTN, V_NEW], output=INTRA,
                reads=[("c", "d", "tile:v"), ("d", "v", "tile:c")],
                write=("c", "v"), reduce="d"))
# op7: out[c,v] = attn_inter+intra   elementwise add
ops.append(dict(name="out_add", kind="Elementwise", unary=None, op="AddF",
                inputs=[ATTN_INTER, INTRA], output=OUT,
                reads=[("c", "v"), ("c", "v")], write=("c", "v"), reduce=None))
# op8: kdec[d,k] = kk*wdecay   elementwise
ops.append(dict(name="kdec_mul", kind="Elementwise", unary=None, op="MulF",
                inputs=[KK, WDECAY], output=KDEC,
                reads=[("d", "k"), ("d", "k")], write=("d", "k"), reduce=None))
# op9: kv[k,v] = sum_d kdec[d,k]*v_new[d,v]   matmul reduce d
ops.append(dict(name="kv_matmul", kind="EinsumByVe", unary=None, op="MulF",
                inputs=[KDEC, V_NEW], output=KV,
                reads=[("d", "k", "tile:v"), ("d", "v", "tile:k")],
                write=("k", "v"), reduce="d"))
# op10: Sdec[k,v] = Sprev*sdecay   elementwise
ops.append(dict(name="sdec_mul", kind="Elementwise", unary=None, op="MulF",
                inputs=[SPREV, SDECAY], output=SDEC,
                reads=[("k", "v"), ("k", "v")], write=("k", "v"), reduce=None))
# op11: Snext[k,v] = Sdec+kv   elementwise add
ops.append(dict(name="snext_add", kind="Elementwise", unary=None, op="AddF",
                inputs=[SDEC, KV], output=SNEXT,
                reads=[("k", "v"), ("k", "v")], write=("k", "v"), reduce=None))

# ---- emit YAML ----
VAR = {"h": "H", "c": "C", "d": "C", "k": "K", "v": "V"}

def shape_block(dims, indent, mh=False):
    pad = " " * indent
    lines = [f"{pad}DynamicUnlabeledShape:",
             f"{pad}  inner:",
             f"{pad}    sizes:"]
    dd = (("h",) + tuple(dims)) if mh else tuple(dims)
    for d in dd:
        lines.append(f"{pad}      - Var: {VAR[d]}")
    return "\n".join(lines)

def axis_block(label, indent, is_tile):
    pad = " " * indent
    head = f"{pad}- axis:" if is_tile else f"{pad}- tag:"
    return "\n".join([
        head,
        f"{pad}    LabelStride:",
        f"{pad}      label:",
        f"{pad}        inner: \"{label}\"",
        f"{pad}      stride: 1",
        f"{pad}  size:",
        f"{pad}    Var: {VAR[label]}",
    ])

def read_blocks(dims, base_indent, tile_indent, mh=False):
    # The head-batch axis "h" is prepended ONLY to the base (storage) dims; it is
    # NEVER added to the tile list (the broadcast axes), exactly as dn_step_mh.yaml.
    base = [d for d in dims if not d.startswith("tile:")]
    tiles = [d.split(":")[1] for d in dims if d.startswith("tile:")]
    if mh:
        base = ["h"] + base
    inner = "\n".join(axis_block(d, base_indent, False) for d in base)
    tile = "\n".join(axis_block(t, tile_indent, True) for t in tiles)
    return inner, tile

def prereduce_shape(write_dims, reduce_axis, base_indent, mh=False):
    # The DPE ein_ops.reduce.input is the PRE-reduction product tensor: the write
    # axes followed by the contracted axis (matches dn_linear_dpe.yaml's [t,o,i]).
    # 'h' (mh batch axis) is prepended OUTERMOST, never reduced -- same rule as reads.
    dims = list(write_dims) + [reduce_axis]
    if mh:
        dims = ["h"] + dims
    return "\n".join(axis_block(d, base_indent, False) for d in dims)


def emit(mh, dpe=False, dpe_max=None):
    # dpe=True  : convert the EinsumByVe matmuls to the fast DPE (systolic) engine.
    # dpe_max=N : convert only the FIRST N matmuls to DPE, leaving the rest on the
    #             VECTOR engine. This exists because the TacticKernel lowering only
    #             tolerates <=2 EinsumByDpe ops per graph: with 3+ DPE ops in ONE
    #             kernel the systolic scheduler silently produces GARBAGE (out maxabs
    #             ~0.6 vs ~1e-3 with <=2, dfg_inner still 0 i.e. on-NPU but wrong) --
    #             measured independent of WHICH ops / data deps (the mamba->single-
    #             EinsumByDpe fusion pass mis-fuses them). So dpe_max=2 is the largest
    #             value that still matches HF at atol 1e-2. See dpe_incremental_log.md.
    out = io.StringIO()
    w = out.write
    w("#naive_yaml\n")
    if dpe:
        cap = "ALL 5" if dpe_max is None else f"the first {dpe_max}"
        w(f"# AUTO-GENERATED by gen_chunk_full.py (dpe=True, dpe_max={dpe_max}): ONE FULL CHUNK\n")
        w(f"# with inter-chunk state carry, but {cap} internal matmul ops run on the FAST DPE\n")
        w("# (systolic MAC) engine (kind: EinsumByDpe) instead of the VECTOR engine\n")
        w("# (EinsumByVe). The pure-elementwise ops (decay mul, sub, add) stay Elementwise.\n")
        w("# Proven DPE recipe from dn_linear_dpe.yaml / dpe_incremental_log.md: the\n")
        w("# contraction moves into ein_ops.reduce (pre-reduce product = [write axes...,\n")
        w("# contracted axis]) and vector_ops becomes a single identity passthrough (empty-\n")
        w("# axes LocalReduceAddF). DPE is bf16 systolic so expect ~0.23% relmean (validate\n")
        w("# at atol 1e-2, NOT 1e-3).\n")
        if dpe_max is None:
            w("# WARNING: with all 5 matmuls on DPE this graph DOES NOT match HF -- the\n")
            w("# TacticKernel lowering corrupts graphs with >2 EinsumByDpe ops (out maxabs\n")
            w("# ~0.5, dfg_inner still 0). Use dn_chunk_full_dpe2.yaml (dpe_max=2) for a\n")
            w("# variant that actually validates at atol 1e-2.\n")
        w("# Inputs: 0=q[c,k] 1=kk[d,k] 2=value[d,v] 3=decay_strict[c,d] 4=kcd[d,k]\n")
        w("#         5=gexp_k[c,k] 6=wdecay[d,k] 7=sdecay[k,v] 8=Sprev[k,v].\n")
        w("# Outputs: 17=out[c,v]  20=Snext[k,v].\n")
    elif mh:
        w("# AUTO-GENERATED by gen_chunk_full.py : ONE FULL CHUNK, HEAD-BATCHED (all H heads).\n")
        w("# Leading head axis 'h' is prepended OUTERMOST to every tensor + every read/write\n")
        w("# base shape; 'h' is NEVER tiled and NEVER reduced (proven batch-axis rule,\n")
        w("# dn_step_mh.yaml). Per-chunk matmuls stay per-head: 'h' carried in both\n")
        w("# operands and the output, reduce axes stay 'k'/'d'. One dispatch for all heads.\n")
        w("# Inputs: 0=q[h,c,k] 1=kk[h,d,k] 2=value[h,d,v] 3=decay_strict[h,c,d] 4=kcd[h,d,k]\n")
        w("#         5=gexp_k[h,c,k] 6=wdecay[h,d,k] 7=sdecay[h,k,v] 8=Sprev[h,k,v].\n")
        w("# Outputs: 17=out[h,c,v]  20=Snext[h,k,v].\n")
    else:
        w("# AUTO-GENERATED by gen_chunk_full.py : ONE FULL CHUNK with inter-chunk state carry.\n")
        w("# Inputs: 0=q[c,k] 1=kk[d,k] 2=value[d,v] 3=decay_strict[c,d] 4=kcd[d,k]\n")
        w("#         5=gexp_k[c,k] 6=wdecay[d,k] 7=sdecay[k,v] 8=Sprev[k,v].\n")
        w("# Outputs: 17=out[c,v]  20=Snext[k,v].\n")
    w("---\n")
    w("tensors:\n  inner:\n")
    for tid in sorted(tensors):
        w(f"    {tid}:\n")
        w("      shape:\n")
        w(shape_block(tensors[tid], 8, mh) + "\n")
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
    _dpe_done = [0]   # how many matmuls have been DPE-converted so far (for dpe_max)
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
            inner_str, tile_str = read_blocks(rdims, base_indent=24, tile_indent=20, mh=mh)
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
        # --- DPE conversion: the EinsumByVe matmuls (those with a reduce axis) move
        # their contraction into ein_ops.reduce and use an identity vector_ops. The
        # pure-elementwise ops (no reduce) are left on the VECTOR engine unchanged. ---
        is_matmul = opd["kind"] == "EinsumByVe"
        use_dpe = dpe and is_matmul and (dpe_max is None or _dpe_done[0] < dpe_max)
        if use_dpe:
            _dpe_done[0] += 1
            assert opd["reduce"] is not None, f"DPE op {opd['name']} has no reduce axis"
            prod_str = prereduce_shape(opd["write"], opd["reduce"], base_indent=24, mh=mh)
            w("              ein_ops:\n")
            w("                reduce:\n")
            w("                  mode: Add\n")
            w("                  input:\n")
            w("                    shape:\n")
            w("                      inner:\n")
            w(prod_str + "\n")
            w("                    element_type: Float32\n")
            w("                  axes:\n")
            w("                    - LabelStride:\n")
            w("                        label:\n")
            w(f"                          inner: \"{opd['reduce']}\"\n")
            w("                        stride: 1\n")
            w("                  source: \"\"\n")
            w("                mul_source: \"\"\n")
            # identity passthrough of the single contracted EinOps result: a Reduce
            # over an EMPTY axes list (the only identity idiom; no DSL identity Unary).
            w("              vector_ops:\n")
            w("                inputs:\n")
            w("                  - 0\n")
            w("                insts:\n")
            w("                  - def: 1\n")
            w("                    expr:\n")
            w("                      Reduce:\n")
            w("                        operator: LocalReduceAddF\n")
            w("                        operand:\n")
            w("                          Tensor: 0\n")
            w("                        axes:\n")
            w("                          Tag: []\n")
            w("                    source: \"\"\n")
        else:
            w("              ein_ops: ~\n")
            w("              vector_ops:\n")
            w("                inputs:\n")
            if opd["unary"] is not None:
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
        inner_str, _ = read_blocks(opd["write"], base_indent=24, tile_indent=24, mh=mh)
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
        w(f"            kind: {'EinsumByDpe' if use_dpe else opd['kind']}\n")
        w("            sparsity: None\n")
    w(f"  next_operator_index: {len(ops)}\n")
    w("hidden_outputs: []\n")
    return out.getvalue()


_DIR = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/"
#   (fname, mh, dpe)
#   dn_chunk_full.yaml      : single-head VE-engine reference (matches HF ~1e-7).
#   dn_chunk_full_mh.yaml   : head-batched VE-engine variant.
#   dn_chunk_full_dpe.yaml  : single-head but the 5 internal matmuls on the DPE
#                             (systolic) engine (kind: EinsumByDpe). bf16 systolic
#                             => validate at atol 1e-2, NOT 1e-3. Elementwise ops
#                             (decay mul, sub, add) stay on the VECTOR engine.
#   dn_chunk_full_dpe.yaml  : all 5 matmuls on DPE (per the task ask). NOTE: does NOT
#                             validate vs HF -- the lowering corrupts >2 EinsumByDpe
#                             ops in one graph (out maxabs ~0.5). Kept as the literal
#                             "convert each" artifact + the blocker evidence.
#   dn_chunk_full_dpe2.yaml : the WORKING variant -- only the first 2 matmuls on DPE
#                             (the lowering cap), rest VE. Matches HF at atol 1e-2.
for fname, mh, dpe, dpe_max in [("dn_chunk_full.yaml", False, False, None),
                                ("dn_chunk_full_mh.yaml", True, False, None),
                                ("dn_chunk_full_dpe.yaml", False, True, None),
                                ("dn_chunk_full_dpe2.yaml", False, True, 2)]:
    path = _DIR + fname
    with open(path, "w") as f:
        f.write(emit(mh, dpe, dpe_max))
    print("wrote", path, " n_ops =", len(ops), " mh =", mh, " dpe =", dpe, " dpe_max =", dpe_max)
