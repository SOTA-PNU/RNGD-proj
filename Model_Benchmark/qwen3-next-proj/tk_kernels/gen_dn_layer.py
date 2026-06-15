#!/usr/bin/env python3
# Emit the three SURROUNDING DeltaNet-layer TK-graph kernels:
#   (1) dn_conv1d.yaml : causal depthwise conv1d (K=4) + SiLU
#   (2) dn_l2norm.yaml : L2 normalize over last dim
#   (3) dn_gnorm.yaml  : Qwen3NextRMSNormGated  y = x*rsqrt(mean(x^2)+eps)*w * silu(gate)
#
# Proven DSL facts (empirically probed on rngd 2026.2.1):
#   Unary valid+correct : Exp, Sigmoid, Sqrt        (Rsqrt/Reciprocal = unknown enum variant)
#   Binary valid        : MulF, AddF, SubF, DivF    (PowF = unknown variant)
#   Reduce              : LocalReduceAddF (over a Tag axis absent from the output)
#   EinsumByVe          : broadcast/tiled read MUST be read0
#   Scalars             : materialized to operand shape by the caller (torch.full)
#   rsqrt(s) = DivF(1, Sqrt(s))   (no native rsqrt/reciprocal unary)
import io

VARMAP = {"c": "C", "t": "T", "m": "M", "d": "D"}

def _sz(label):
    return VARMAP[label]

def shape_inline(dims, indent):
    pad = " " * indent
    lines = [f"{pad}DynamicUnlabeledShape:", f"{pad}  inner:", f"{pad}    sizes:"]
    for d in dims:
        lines.append(f"{pad}      - Var: {_sz(d)}")
    return "\n".join(lines)

def labelstride(label, indent, is_tile):
    pad = " " * indent
    head = f"{pad}- axis:" if is_tile else f"{pad}- tag:"
    return "\n".join([
        head,
        f"{pad}    LabelStride:",
        f"{pad}      label:",
        f"{pad}        inner: \"{label}\"",
        f"{pad}      stride: 1",
        f"{pad}  size:",
        f"{pad}    Var: {_sz(label)}",
    ])

def emit(tensors, inputs, outputs, ops, path, header=""):
    """tensors: {tid: (dim,...)}.  ops: list of dicts:
        name, kind ('Elementwise'|'EinsumByVe'), inputs:[tid...], output tid,
        reads: list of (base_dims, tile_dims) tuples,
        insts: list of inst dicts, write: base_dims tuple."""
    out = io.StringIO(); w = out.write
    w("#naive_yaml\n")
    if header:
        for ln in header.splitlines():
            w(f"# {ln}\n")
    w("---\ntensors:\n  inner:\n")
    for tid in sorted(tensors):
        w(f"    {tid}:\n      shape:\n")
        w(shape_inline(tensors[tid], 8) + "\n")
        w("      element_type: Float32\n      buffer: []\n      name: \"\"\n      source: Unknown\n      buffer_type: Sram\n")
    w("inputs:\n")
    for i in inputs: w(f"  - {i}\n")
    w("outputs:\n")
    for o in outputs: w(f"  - {o}\n")
    w("operators:\n  operators:\n")
    for idx, op in enumerate(ops):
        w(f"    {idx}:\n      name: {op['name']}\n      option:\n        SymTacticKernel:\n          inputs:\n")
        for i in op["inputs"]: w(f"            - {i}\n")
        w(f"          output: {op['output']}\n          inner:\n            inner:\n              reads:\n")
        for (base, tiles) in op["reads"]:
            w("                - input:\n                    shape:\n                      inner:\n")
            w("\n".join(labelstride(d, 24, False) for d in base) + "\n")
            w("                    element_type: Float32\n                  table_lookup: None\n                  has_transmutation: false\n                  subtraction: None\n                  paddings: []\n                  slides: []\n                  strides: []\n")
            if tiles:
                w("                  tiles:\n")
                w("\n".join(labelstride(d, 20, True) for d in tiles) + "\n")
            else:
                w("                  tiles: []\n")
        w("              ein_ops: ~\n              vector_ops:\n                inputs:\n")
        for vi in op["vinputs"]: w(f"                  - {vi}\n")
        w("                insts:\n")
        for inst in op["insts"]:
            w(f"                  - def: {inst['def']}\n                    expr:\n")
            if inst["kind"] == "unary":
                w("                      Unary:\n")
                w(f"                        operator: {inst['op']}\n                        operand:\n                          Tensor: {inst['a']}\n")
            elif inst["kind"] == "binary":
                w("                      Binary:\n")
                w(f"                        operator: {inst['op']}\n                        lhs:\n                          Tensor: {inst['a']}\n                        rhs:\n                          Tensor: {inst['b']}\n")
            elif inst["kind"] == "reduce":
                w("                      Reduce:\n")
                w(f"                        operator: LocalReduceAddF\n                        operand:\n                          Tensor: {inst['a']}\n                        axes:\n                          Tag:\n                            - inner: \"{inst['axis']}\"\n")
            w("                    source: \"\"\n")
        wb = op["write"]
        w("              write:\n                input:\n                  shape:\n                    inner:\n")
        w("\n".join(labelstride(d, 22, False) for d in wb) + "\n")
        w("                  element_type: Float32\n                output:\n                  shape:\n                    inner:\n")
        w("\n".join(labelstride(d, 22, False) for d in wb) + "\n")
        w("                  element_type: Float32\n                has_transmutation: false\n")
        w(f"            kind: {op['kind']}\n            sparsity: None\n")
    w(f"  next_operator_index: {len(ops)}\nhidden_outputs: []\n")
    open(path, "w").write(out.getvalue())
    print("wrote", path)

BASE = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/"

# ============================================================================
# (1) dn_conv1d.yaml : causal depthwise conv1d (K=4) + SiLU, layout [c, t]
# Caller pre-shifts the causal window into 4 tensors xs0..xs3 (each [C,T]):
#   x_pad = cat([zeros(C,3), x], dim=-1) ; xs_j[c,t] = x_pad[c, t+j]   j=0..3
#   so   acc[c,t] = sum_j xs_j[c,t] * w[c,j]
# Per-channel weights materialized to [C,T] by caller: wj[c,t] = w[c,j].
# Then SiLU(acc) = acc * sigmoid(acc).
# Split into small (<=2 inst) Elementwise ops, mirroring the proven dn_step style
# (a single 9-inst fused op fails: "preferred_ve_lhs is not an operand").
#   tids: 0..3 = xs0..3 ; 4..7 = w0..3 ; inter 8..14 ; output 15 = y[c,t]
# ============================================================================
def ct_op(name, op, a, b, out, unary=False):
    if unary:
        insts = [dict(kind="unary", op=op, a=0, **{"def": 1})]
        return dict(name=name, kind="Elementwise", inputs=[a], output=out,
                    vinputs=[0], reads=[(("c", "t"), ())],
                    insts=insts, write=("c", "t"))
    insts = [dict(kind="binary", op=op, a=0, b=1, **{"def": 2})]
    return dict(name=name, kind="Elementwise", inputs=[a, b], output=out,
                vinputs=[0, 1], reads=[(("c", "t"), ()), (("c", "t"), ())],
                insts=insts, write=("c", "t"))

conv_tensors = {i: ("c", "t") for i in range(16)}
conv_ops = [
    ct_op("conv_p0", "MulF", 0, 4, 8),    # p0 = xs0*w0
    ct_op("conv_p1", "MulF", 1, 5, 9),    # p1 = xs1*w1
    ct_op("conv_a1", "AddF", 8, 9, 10),   # acc1 = p0+p1
    ct_op("conv_p2", "MulF", 2, 6, 11),   # p2 = xs2*w2
    ct_op("conv_a2", "AddF", 10, 11, 12), # acc2 = acc1+p2
    ct_op("conv_p3", "MulF", 3, 7, 13),   # p3 = xs3*w3
    ct_op("conv_acc", "AddF", 12, 13, 14),# acc = acc2+p3
    # SiLU(acc) = acc*sigmoid(acc) fused in one op (proven dn_gate pattern)
    dict(name="conv_silu", kind="Elementwise", inputs=[14], output=15,
         vinputs=[0], reads=[(("c", "t"), ())],
         insts=[dict(kind="unary", op="Sigmoid", a=0, **{"def": 1}),
                dict(kind="binary", op="MulF", a=0, b=1, **{"def": 2})],
         write=("c", "t")),
]
emit(
    conv_tensors, [0, 1, 2, 3, 4, 5, 6, 7], [15], conv_ops,
    BASE + "dn_conv1d.yaml",
    header=("CAUSAL DEPTHWISE CONV1D (K=4) + SiLU, layout [c,t].\n"
            "Caller pre-shifts causal window: x_pad=cat([zeros(C,3),x],-1); xs_j[c,t]=x_pad[c,t+j].\n"
            "acc = sum_{j=0..3} xs_j * w_j ; w_j[c,t]=weight[c,j] materialized to [C,T].\n"
            "SiLU(acc)=acc*sigmoid(acc).  All fp32 Elementwise (MulF/AddF/Sigmoid).\n"
            "Split into <=2-inst ops; one 9-inst fused op fails 'preferred_ve_lhs'."))

# ============================================================================
# (2) dn_l2norm.yaml : y = x * rsqrt(sum_over_feature x^2 + eps).
# AXIS RULES (proven on rngd):
#  * a reduce-to-vector (sum over feature) lowers to the NPU only when the
#    REDUCED axis is the OUTER axis and the surviving (row) axis is the large
#    INNER axis -> the square/sumsq ops use a [d,m] copy of x (d=feature OUTER).
#  * a no-reduce broadcast EinsumByVe writes its output in (read0.base, read0.tile)
#    order; the per-row scalar inv[m] is read0 tiled along "d" -> output [m,d].
#    So the scale-back + final output are [m,d] (== torch layout, no transpose).
# Caller therefore passes BOTH x_dm[d,m] (for square/sumsq) and x_md[m,d] (scale).
#   op0 Elementwise : sq[d,m]  = x_dm*x_dm                     (MulF)
#   op1 EinsumByVe  : ss[m]    = sum_d ones[d]*sq[d,m]         (read0=ones tiled m, reduce OUTER d)
#   op2 Elementwise : se[m]    = ss + eps                      (AddF, eps_full[m])
#   op3 Elementwise : rt[m]    = Sqrt(se)                      (Unary Sqrt)
#   op4 Elementwise : inv[m]   = rt / se                       (DivF == 1/sqrt(se))
#   op5 EinsumByVe  : y[m,d]   = inv[m]*x_md[m,d]              (read0=inv tiled d, NO reduce)
#   inputs: 0=x_dm[d,m] 1=ones_d[d] 2=eps_full[m] 3=x_md[m,d]
#   inter : 4=sq[d,m] 5=ss[m] 6=se[m] 7=rt[m] 8=inv[m] ; out 9=y[m,d]
# RSQRT IDENTITY: 1/sqrt(se) = sqrt(se)/se. The NPU vector engine has Sqrt but NO
# rsqrt/reciprocal unary; and DivF(one, rt) (numerator==1) collapses to rt on the VE
# (empirically). DivF(rt, se) avoids the numerator==1 trap and is exact (maxerr 6e-8).
# ============================================================================
l2_tensors = {
    0: ("d", "m"), 1: ("d",), 2: ("m",), 3: ("m", "d"),
    4: ("d", "m"), 5: ("m",), 6: ("m",), 7: ("m",), 8: ("m",), 9: ("m", "d"),
}
l2_ops = [
    dict(name="l2_square", kind="Elementwise", inputs=[0, 0], output=4,
         vinputs=[0, 1],
         reads=[(("d", "m"), ()), (("d", "m"), ())],
         insts=[dict(kind="binary", op="MulF", a=0, b=1, **{"def": 2})],
         write=("d", "m")),
    dict(name="l2_sumsq", kind="EinsumByVe", inputs=[1, 4], output=5,
         vinputs=[0, 1],
         reads=[(("d",), ("m",)), (("d", "m"), ())],
         insts=[dict(kind="binary", op="MulF", a=0, b=1, **{"def": 2}),
                dict(kind="reduce", a=2, axis="d", **{"def": 3})],
         write=("m",)),
    dict(name="l2_addeps", kind="Elementwise", inputs=[5, 2], output=6,
         vinputs=[0, 1],
         reads=[(("m",), ()), (("m",), ())],
         insts=[dict(kind="binary", op="AddF", a=0, b=1, **{"def": 2})],
         write=("m",)),
    dict(name="l2_sqrt", kind="Elementwise", inputs=[6], output=7,
         vinputs=[0],
         reads=[(("m",), ())],
         insts=[dict(kind="unary", op="Sqrt", a=0, **{"def": 1})],
         write=("m",)),
    dict(name="l2_rsqrt", kind="Elementwise", inputs=[7, 6], output=8,
         vinputs=[0, 1],
         reads=[(("m",), ()), (("m",), ())],
         insts=[dict(kind="binary", op="DivF", a=0, b=1, **{"def": 2})],
         write=("m",)),
    dict(name="l2_scale", kind="EinsumByVe", inputs=[8, 3], output=9,
         vinputs=[0, 1],
         reads=[(("m",), ("d",)), (("m", "d"), ())],
         insts=[dict(kind="binary", op="MulF", a=0, b=1, **{"def": 2})],
         write=("m", "d")),
]
emit(l2_tensors, [0, 1, 2, 3], [9], l2_ops, BASE + "dn_l2norm.yaml",
     header=("L2NORM over feature dim: y = x*rsqrt(sum_d x^2 + eps).\n"
             "Reduce ops use a [d,m] copy of x (reduce OUTER d, survive INNER m) so the\n"
             "adder-tree lowers to NPU; scale-back uses [m,d] (read0=inv tiled d -> [m,d] out).\n"
             "rt=Sqrt(se) ; inv=rt/se == 1/sqrt(se) (no native rsqrt; DivF(one,rt) collapses to rt).\n"
             "Inputs: 0=x_dm[d,m] 1=ones_d[d] 2=eps_full[m] 3=x_md[m,d]. Output y[m,d]."))

# ============================================================================
# (3) dn_gnorm.yaml : Qwen3NextRMSNormGated. Same axis rules as dn_l2norm:
#   square/sumsq on a [d,m] copy of x (reduce OUTER d), everything from the
#   scale-back onward in [m,d] (read0=inv tiled d -> [m,d] output == torch layout).
#   y = (x * rsqrt(mean_d(x^2)+eps) * weight) * silu(gate)
#   mean_d(x^2) = (1/D)*sum_d x^2 ; caller passes invD_full[m]=1/D (multiply, no div).
#   op0 Elementwise : sq[d,m] = x_dm*x_dm
#   op1 EinsumByVe  : ss[m]   = sum_d ones[d]*sq[d,m]          (reduce OUTER d)
#   op2 Elementwise : var[m]  = ss * invD                      (MulF, invD_full[m])
#   op3 Elementwise : ve[m]   = var + eps                      (AddF, eps_full[m])
#   op4 Elementwise : rt[m]   = Sqrt(ve)
#   op5 Elementwise : inv[m]  = rt / ve                        (DivF == 1/sqrt(ve))
#   op6 EinsumByVe  : xn[m,d] = inv[m]*x_md[m,d]               (read0=inv tiled d)
#   op7 Elementwise : xw[m,d] = xn * weight                    (MulF, weight_md[m,d])
#   op8 Elementwise : sg[m,d] = gate*sigmoid(gate)            (silu(gate), 2 insts)
#   op9 Elementwise : y[m,d]  = xw * sg                        (MulF)
#   inputs: 0=x_dm[d,m] 1=ones_d[d] 2=invD[m] 3=eps[m] 4=weight_md[m,d]
#           5=gate_md[m,d] 6=x_md[m,d]
#   inter : 7=sq 8=ss 9=var 10=ve 11=rt 12=inv 13=xn 14=xw 15=sg ; out 16=y[m,d]
# RSQRT IDENTITY (same as dn_l2norm): inv = rt/ve == 1/sqrt(ve); DivF(one,rt) would
# collapse to rt on the VE, so divide sqrt by the variance instead.
# ============================================================================
g_tensors = {
    0: ("d", "m"), 1: ("d",), 2: ("m",), 3: ("m",),
    4: ("m", "d"), 5: ("m", "d"), 6: ("m", "d"),
    7: ("d", "m"), 8: ("m",), 9: ("m",), 10: ("m",), 11: ("m",), 12: ("m",),
    13: ("m", "d"), 14: ("m", "d"), 15: ("m", "d"), 16: ("m", "d"),
}
g_ops = [
    dict(name="gn_square", kind="Elementwise", inputs=[0, 0], output=7,
         vinputs=[0, 1], reads=[(("d", "m"), ()), (("d", "m"), ())],
         insts=[dict(kind="binary", op="MulF", a=0, b=1, **{"def": 2})], write=("d", "m")),
    dict(name="gn_sumsq", kind="EinsumByVe", inputs=[1, 7], output=8,
         vinputs=[0, 1], reads=[(("d",), ("m",)), (("d", "m"), ())],
         insts=[dict(kind="binary", op="MulF", a=0, b=1, **{"def": 2}),
                dict(kind="reduce", a=2, axis="d", **{"def": 3})], write=("m",)),
    dict(name="gn_mean", kind="Elementwise", inputs=[8, 2], output=9,
         vinputs=[0, 1], reads=[(("m",), ()), (("m",), ())],
         insts=[dict(kind="binary", op="MulF", a=0, b=1, **{"def": 2})], write=("m",)),
    dict(name="gn_addeps", kind="Elementwise", inputs=[9, 3], output=10,
         vinputs=[0, 1], reads=[(("m",), ()), (("m",), ())],
         insts=[dict(kind="binary", op="AddF", a=0, b=1, **{"def": 2})], write=("m",)),
    dict(name="gn_sqrt", kind="Elementwise", inputs=[10], output=11,
         vinputs=[0], reads=[(("m",), ())],
         insts=[dict(kind="unary", op="Sqrt", a=0, **{"def": 1})], write=("m",)),
    dict(name="gn_rsqrt", kind="Elementwise", inputs=[11, 10], output=12,
         vinputs=[0, 1], reads=[(("m",), ()), (("m",), ())],
         insts=[dict(kind="binary", op="DivF", a=0, b=1, **{"def": 2})], write=("m",)),
    dict(name="gn_scale", kind="EinsumByVe", inputs=[12, 6], output=13,
         vinputs=[0, 1], reads=[(("m",), ("d",)), (("m", "d"), ())],
         insts=[dict(kind="binary", op="MulF", a=0, b=1, **{"def": 2})], write=("m", "d")),
    dict(name="gn_weight", kind="Elementwise", inputs=[13, 4], output=14,
         vinputs=[0, 1], reads=[(("m", "d"), ()), (("m", "d"), ())],
         insts=[dict(kind="binary", op="MulF", a=0, b=1, **{"def": 2})], write=("m", "d")),
    dict(name="gn_silu_gate", kind="Elementwise", inputs=[5], output=15,
         vinputs=[0], reads=[(("m", "d"), ())],
         insts=[dict(kind="unary", op="Sigmoid", a=0, **{"def": 1}),
                dict(kind="binary", op="MulF", a=0, b=1, **{"def": 2})], write=("m", "d")),
    dict(name="gn_gatemul", kind="Elementwise", inputs=[14, 15], output=16,
         vinputs=[0, 1], reads=[(("m", "d"), ()), (("m", "d"), ())],
         insts=[dict(kind="binary", op="MulF", a=0, b=1, **{"def": 2})], write=("m", "d")),
]
emit(g_tensors, [0, 1, 2, 3, 4, 5, 6], [16], g_ops, BASE + "dn_gnorm.yaml",
     header=("GATED RMSNORM (Qwen3NextRMSNormGated, modeling_qwen3_next.py L66-81).\n"
             "y = (x*rsqrt(mean_d(x^2)+eps)*weight) * silu(gate). Output [m,d].\n"
             "mean=(1/D)*sum_d x^2 (reduce on [d,m] copy) ; inv=rt/ve==1/sqrt(ve) (Sqrt+DivF) ;\n"
             "silu(gate)=gate*sigmoid(gate).\n"
             "Inputs: 0=x_dm[d,m] 1=ones_d[d] 2=invD_full[m] 3=eps_full[m]\n"
             "        4=weight_md[m,d] 5=gate_md[m,d] 6=x_md[m,d]."))

print("DONE")
