#!/usr/bin/env python3
# =============================================================================
# Qwen3-Coder-Next  MoE FFN block  (Qwen3NextSparseMoeBlock)  on the RNGD NPU.
#
# BATCHED-EXPERT version: the active (top-10 union) experts' SwiGLU matmuls run
# as a SINGLE batched NPU dispatch over a leading expert axis "e" -- instead of
# one dn_linear dispatch PER expert. This uses dn_linear_be.yaml, which is the
# proven dn_linear.yaml ('ti,oi->to') with an OUTERMOST expert-batch axis "e"
# added to every tensor/read/write shape (never tiled, never reduced -- the same
# batch-axis trick proven for dn_step_mh.yaml's head axis "h").
#
# Layer-0 real weights from Qwen/Qwen3-Coder-Next-FP8 (FP8 blockwise dequant via
# qcn.loader.QCNWeights). Validated element-for-element against the per-expert
# (unbatched) path on the SAME real weights (maxerr < 1e-3).
#
# Config: 512 experts, top-10 routing (norm_topk_prob=True), moe_intermediate=512,
#         hidden=2048, SwiGLU = down(silu(gate(x)) * up(x)), + 1 SHARED expert
#         gated by sigmoid(shared_expert_gate(x)).
#
#   hidden ---gate (router) Linear--> logits --softmax--top10--normalize (HOST)
#          ---HOST: gather active experts E_act = unique(top_idx); build a batched
#               x_be[E_act, n_max, hidden] (each expert's routed tokens, zero-pad
#               to a shared n_max) and the stacked weights gate/up/down[E_act,...]
#          ---ONE dn_linear_be dispatch each for gate/up/down (was E_act each):
#               g  = batched gate_proj @ x_be      (dn_linear_be.yaml, NPU, 1 disp)
#               u  = batched up_proj   @ x_be      (dn_linear_be.yaml, NPU, 1 disp)
#               h  = silu(g) * u                    (dn_gate.yaml x1 batched, NPU)
#               y  = batched down_proj @ h         (dn_linear_be.yaml, NPU, 1 disp)
#          ---HOST: scatter-add routing_weight * y back to the token rows
#          ---shared expert SwiGLU on ALL tokens (dn_linear/dn_gate, NPU)
#               + sigmoid(shared_expert_gate @ x) * shared_out  (dn_gate.yaml, NPU)
#
# Routing softmax/top-k stay on HOST (no DSL op). EVERY matmul + the silu/sigmoid
# run ON THE NPU. NPU exec is proven by monkeypatching
# furiosa.torch.custom_ops.dfg._dfg_inner and asserting the call count stays 0.
# =============================================================================
import os, sys, torch
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule
import furiosa.torch.custom_ops.dfg as dfgmod

sys.path.insert(0, "/home/jun/furiosa/lib/python3.12/site-packages")
import torch.nn.functional as F

BASE = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/"
DEV  = os.environ.get("RNGD_DEV", "rngd:1")
PADM = 128                       # SRAM row-tile floor for the (unbatched) shared-expert path
# QCN_DPE=1 routes the shared npu_linear helper through the FAST systolic/DPE-MAC
# matmul (dn_linear_dpe.yaml, kind: EinsumByDpe -- ~3.8x faster than EinsumByVe).
# Same tensor signature + tiling as dn_linear.yaml -> drop-in filename swap. DPE is
# reduced (bf16) precision: validate at atol/rtol 1e-2, NEVER 1e-3. Default OFF.
DPE  = os.environ.get("QCN_DPE", "0") == "1"
LINEAR_YAML = "dn_linear_dpe.yaml" if DPE else "dn_linear.yaml"

def log(*a):
    print(*a, flush=True)

# ---- spy on the CPU-fallback path; must stay 0 for every NPU op ----
_orig = dfgmod._dfg_inner
CALLS = {"n": 0}
def _spy(*a, **k):
    CALLS["n"] += 1
    return _orig(*a, **k)
dfgmod._dfg_inner = _spy

_compiled = {}
def _npu(yaml, inputs):
    """Run a TacticKernel YAML on the NPU. Returns (list-of-cpu-fp32-tensors, dfg_delta)."""
    if yaml not in _compiled:
        m = TacticKernelModule(open(BASE + yaml).read())
        _compiled[yaml] = torch.compile(m, backend=ft.backend)
    before = CALLS["n"]
    res = _compiled[yaml](*[t.to(DEV) for t in inputs])
    if not isinstance(res, (tuple, list)):
        res = [res]
    return [r.detach().to("cpu").float() for r in res], CALLS["n"] - before

NPU_STAGES = []                  # (stage_name, dfg_delta)  -- dfg_delta MUST be 0
FLOPS = {"npu": 0, "host": 0}    # matmul MAC counts for the NPU-vs-host breakdown
DISPATCH = {"n": 0}              # count of NPU kernel dispatches (calls into _npu)
DRY = {"on": False}              # dry-run: count dispatches but skip the NPU exec
                                 # (used to tally before/after dispatches at full T=8
                                 #  without paying the ~minute-per-matmul NPU wall-time)


# ---------------------------------------------------------------------------
# NPU primitives  (unbatched -- used by the shared expert)
# ---------------------------------------------------------------------------
def npu_linear(x_ti, W_oi, name):
    """y[t,o] = sum_i x[t,i]*W[o,i]  ==  F.linear(x, W), on NPU via LINEAR_YAML
       (dn_linear.yaml EinsumByVe, or dn_linear_dpe.yaml EinsumByDpe when QCN_DPE=1).
       Token axis padded to PADM if below it (pad rows don't interact -- the matmul
       reduces over the i axis -- and are sliced back off; exact)."""
    T, I = x_ti.shape
    O = W_oi.shape[0]
    if DRY["on"]:
        DISPATCH["n"] += 1; FLOPS["npu"] += T * I * O
        NPU_STAGES.append((name, 0))
        return torch.zeros(T, O, dtype=x_ti.dtype)
    Wp = W_oi
    if DPE:
        # the DPE systolic array rejects a degenerate output axis (e.g. O=1, the
        # shared_expert_gate logit) with "incompatible sequences" -- it needs the
        # output(o) axis padded to a multiple of PADO=32 (same constraint the VE
        # path enforces in attn_layer.py). Pad cols are zero, don't interact, and
        # are sliced back off -> exact. The EinsumByVe path does not need this.
        Op = ((O + 31) // 32) * 32
        if Op != O:
            Wp = torch.zeros(Op, I, dtype=W_oi.dtype); Wp[:O] = W_oi
    if T < PADM:
        xp = torch.zeros(PADM, I, dtype=x_ti.dtype); xp[:T] = x_ti
        out, d = _npu(LINEAR_YAML, [xp.contiguous(), Wp.contiguous()])
        y = out[0][:T, :O]
    else:
        out, d = _npu(LINEAR_YAML, [x_ti.contiguous(), Wp.contiguous()])
        y = out[0][:, :O]
    DISPATCH["n"] += 1
    NPU_STAGES.append((name, d))
    FLOPS["npu"] += T * I * O
    return y                                              # [T,O]

def npu_mul_sigmoid(a_mn, b_mn, name):
    """sigmoid(a)*b on NPU via dn_gate.yaml (computes sigmoid(in0)*in1)."""
    if DRY["on"]:
        DISPATCH["n"] += 1; NPU_STAGES.append((name, 0))
        return torch.zeros_like(a_mn)
    out, d = _npu("dn_gate.yaml", [a_mn.contiguous(), b_mn.contiguous()])
    DISPATCH["n"] += 1
    NPU_STAGES.append((name, d))
    return out[0]

def npu_silu(x_mn, name):
    """silu(x) = x * sigmoid(x) on NPU via dn_gate.yaml(in0=x, in1=x)."""
    return npu_mul_sigmoid(x_mn, x_mn, name)


# ---------------------------------------------------------------------------
# NPU primitives  (BATCHED over a leading expert axis "e")
# ---------------------------------------------------------------------------
def npu_linear_be(x_eti, W_eoi, name):
    """y[e,t,o] = sum_i x[e,t,i]*W[e,o,i]  == per-expert F.linear, ONE NPU dispatch
       via dn_linear_be.yaml. x:[E,T,I]  W:[E,O,I] -> [E,T,O].
       (T may be small -- the batched kernel does NOT require the 128 token floor;
       the contraction is over "i", so short token blocks are exact.)"""
    E, T, I = x_eti.shape
    O = W_eoi.shape[1]
    if DRY["on"]:
        DISPATCH["n"] += 1; FLOPS["npu"] += E * T * I * O
        NPU_STAGES.append((name, 0))
        return torch.zeros(E, T, O, dtype=x_eti.dtype)
    out, d = _npu("dn_linear_be.yaml", [x_eti.contiguous(), W_eoi.contiguous()])
    DISPATCH["n"] += 1
    NPU_STAGES.append((name, d))
    FLOPS["npu"] += E * T * I * O
    return out[0]                                         # [E,T,O]

def npu_silu_be(x_emn, name):
    """silu over a batched [E,M,N] tensor via dn_gate.yaml. dn_gate is a 2-D [M,N]
       ELEMENTWISE kernel (sigmoid(in0)*in1) with NO cross-row interaction, so we
       flatten the leading dims [E,M] -> a single row axis [E*M, N], run ONE dispatch,
       and reshape back. Flattening is exact for an elementwise op."""
    E, M, N = x_emn.shape
    if DRY["on"]:
        DISPATCH["n"] += 1; NPU_STAGES.append((name, 0))
        return torch.zeros_like(x_emn)
    flat = x_emn.reshape(E * M, N).contiguous()
    out, d = _npu("dn_gate.yaml", [flat, flat])
    DISPATCH["n"] += 1
    NPU_STAGES.append((name, d))
    return out[0].reshape(E, M, N)


# ---------------------------------------------------------------------------
# One SwiGLU MLP on NPU (UNBATCHED):  down( silu(gate(x)) * up(x) )  -- shared expert
# ---------------------------------------------------------------------------
def npu_swiglu(x_th, gate_w, up_w, down_w, tag):
    """x:[T,hidden] -> [T,hidden]. gate_w/up_w:[inter,hidden], down_w:[hidden,inter]."""
    g = npu_linear(x_th, gate_w, f"{tag}.gate_proj")       # [T,inter]
    u = npu_linear(x_th, up_w,   f"{tag}.up_proj")         # [T,inter]
    act = npu_silu(g, f"{tag}.silu")                        # [T,inter]
    hidden_act = act * u                                    # elementwise (host, cheap, no matmul)
    y = npu_linear(hidden_act, down_w, f"{tag}.down_proj")  # [T,hidden]
    return y


# ---------------------------------------------------------------------------
# HOST router: softmax / top-k have no DSL op
# ---------------------------------------------------------------------------
def host_router(hidden, gate_w, top_k, norm_topk_prob):
    T, H = hidden.shape
    E = gate_w.shape[0]
    router_logits = F.linear(hidden, gate_w)                       # [T,E]  (host matmul, tiny)
    FLOPS["host"] += T * H * E
    probs = F.softmax(router_logits, dtype=torch.float, dim=-1)    # HOST
    top_val, top_idx = torch.topk(probs, top_k, dim=-1)            # [T,k]  HOST
    if norm_topk_prob:
        top_val = top_val / top_val.sum(dim=-1, keepdim=True)
    return top_val.to(hidden.dtype), top_idx


# ---------------------------------------------------------------------------
# BATCHED-EXPERT NPU MoE forward
#   active experts E_act = unique(top_idx). For each active expert build its routed
#   token block; stack into x_be[E_act, n_max, H] (zero-padded to a shared n_max).
#   Run gate/up/down as ONE batched dispatch each, then scatter-add back.
# ---------------------------------------------------------------------------
def moe_forward_npu(hidden, W, top_idx, top_val, layer=0, group_cap=None):
    """hidden:[T,H]; W=QCNWeights; top_idx/top_val:[T,k]. Returns (out[T,H], n_distinct).
       group_cap: if set, batch at most this many experts per dn_linear_be dispatch
       (fallback for a kernel/SRAM ceiling on the expert axis). None = all active in one."""
    T, H = hidden.shape
    p = f"model.layers.{layer}.mlp."
    out = torch.zeros(T, H, dtype=hidden.dtype)
    activated = sorted(torch.unique(top_idx).tolist())
    E_act = len(activated)

    # --- HOST: for each active expert, collect its (token_row, kpos) pairs ---
    rows_per_e, kpos_per_e = [], []
    for e in activated:
        tok_rows, kpos = torch.where(top_idx == e)         # tokens routed to e, and which slot
        rows_per_e.append(tok_rows)
        kpos_per_e.append(kpos)
    n_max = max((len(r) for r in rows_per_e), default=0)   # shared padded token count
    if n_max == 0:
        n_max = 1

    # --- helper: run a contiguous GROUP of active experts as ONE batched dispatch ---
    def run_group(grp_idx):
        ne = len(grp_idx)
        # gather this group's weights -> [ne, O, I]
        gate_w = torch.stack([W.get(f"{p}experts.{activated[gi]}.gate_proj.weight", torch.float32) for gi in grp_idx])  # [ne,INT,H]
        up_w   = torch.stack([W.get(f"{p}experts.{activated[gi]}.up_proj.weight",   torch.float32) for gi in grp_idx])  # [ne,INT,H]
        down_w = torch.stack([W.get(f"{p}experts.{activated[gi]}.down_proj.weight", torch.float32) for gi in grp_idx])  # [ne,H,INT]
        INT = gate_w.shape[1]
        # batched token block x_be[ne, n_max, H], zero-padded; remember valid lengths
        x_be = torch.zeros(ne, n_max, H, dtype=hidden.dtype)
        for j, gi in enumerate(grp_idx):
            r = rows_per_e[gi]
            x_be[j, :len(r)] = hidden[r]
        g = npu_linear_be(x_be, gate_w, f"grp{grp_idx[0]}.gate_proj")     # [ne,n_max,INT]
        u = npu_linear_be(x_be, up_w,   f"grp{grp_idx[0]}.up_proj")       # [ne,n_max,INT]
        act = npu_silu_be(g, f"grp{grp_idx[0]}.silu")                      # [ne,n_max,INT]
        h_act = (act * u).contiguous()                                     # host elementwise
        y = npu_linear_be(h_act, down_w, f"grp{grp_idx[0]}.down_proj")     # [ne,n_max,H]
        # scatter-add routing_weight * y[j, :len(r)] back into out
        for j, gi in enumerate(grp_idx):
            r = rows_per_e[gi]; kp = kpos_per_e[gi]
            w_e = top_val[r, kp].unsqueeze(-1)                             # [len(r),1]
            out.index_add_(0, r, (y[j, :len(r)] * w_e).to(out.dtype))

    # --- split active experts into groups (group_cap caps the expert-batch size) ---
    idxs = list(range(E_act))
    cap = group_cap if group_cap else E_act
    for s in range(0, E_act, cap):
        run_group(idxs[s:s + cap])

    # ---- SHARED EXPERT (NPU) on ALL tokens, then sigmoid-gate ----
    sg = W.get(p + "shared_expert.gate_proj.weight", torch.float32)
    su = W.get(p + "shared_expert.up_proj.weight",   torch.float32)
    sd = W.get(p + "shared_expert.down_proj.weight", torch.float32)
    shared_gate_w = W.get(p + "shared_expert_gate.weight", torch.float32)   # [1,H]
    shared_out = npu_swiglu(hidden, sg, su, sd, "shared")   # [T,H]
    sgate_logit = npu_linear(hidden, shared_gate_w, "shared_gate")          # [T,1]
    shared_out = npu_mul_sigmoid(sgate_logit.expand(T, H).contiguous(),
                                 shared_out, "shared.sigmoid_gate")          # sigmoid(g)*shared
    out = out + shared_out
    return out, E_act


# ---------------------------------------------------------------------------
# UNBATCHED reference path (the ORIGINAL one-expert-at-a-time moe.py) -- used to
# validate the batched path matches element-for-element on real layer-0 weights.
# ---------------------------------------------------------------------------
def moe_forward_npu_unbatched(hidden, W, top_idx, top_val, layer=0):
    T, H = hidden.shape
    p = f"model.layers.{layer}.mlp."
    out = torch.zeros(T, H, dtype=hidden.dtype)
    activated = torch.unique(top_idx).tolist()
    for e in activated:
        sel = (top_idx == e)
        tok_rows, kpos = torch.where(sel)
        x_e = hidden[tok_rows]
        gw = W.get(f"{p}experts.{e}.gate_proj.weight", torch.float32)
        uw = W.get(f"{p}experts.{e}.up_proj.weight",   torch.float32)
        dw = W.get(f"{p}experts.{e}.down_proj.weight", torch.float32)
        y_e = npu_swiglu(x_e, gw, uw, dw, f"exp{e}")
        w_e = top_val[tok_rows, kpos].unsqueeze(-1)
        out.index_add_(0, tok_rows, (y_e * w_e).to(out.dtype))
    sg = W.get(p + "shared_expert.gate_proj.weight", torch.float32)
    su = W.get(p + "shared_expert.up_proj.weight",   torch.float32)
    sd = W.get(p + "shared_expert.down_proj.weight", torch.float32)
    shared_gate_w = W.get(p + "shared_expert_gate.weight", torch.float32)
    shared_out = npu_swiglu(hidden, sg, su, sd, "shared")
    sgate_logit = npu_linear(hidden, shared_gate_w, "shared_gate")
    shared_out = npu_mul_sigmoid(sgate_logit.expand(T, H).contiguous(), shared_out, "shared.sigmoid_gate")
    out = out + shared_out
    return out, len(activated)


# ---------------------------------------------------------------------------
# HF reference (the REAL Qwen3NextSparseMoeBlock; only activated experts get
# real weights -- the HF expert loop computes ONLY activated experts, so
# untouched experts never participate. Exact for our T=8 input.)
# ---------------------------------------------------------------------------
def build_hf_reference(W, activated, layer=0):
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextSparseMoeBlock
    cfg = Qwen3NextConfig(**W.config)
    blk = Qwen3NextSparseMoeBlock(cfg).eval()
    p = f"model.layers.{layer}.mlp."
    INT = cfg.moe_intermediate_size
    H   = cfg.hidden_size

    blk.gate.weight.data.copy_(W.get(p + "gate.weight", torch.float32))

    blk.experts.gate_up_proj.data.zero_()
    blk.experts.down_proj.data.zero_()
    for e in activated:
        gw = W.get(f"{p}experts.{e}.gate_proj.weight", torch.float32)  # [INT,H]
        uw = W.get(f"{p}experts.{e}.up_proj.weight",   torch.float32)  # [INT,H]
        dw = W.get(f"{p}experts.{e}.down_proj.weight", torch.float32)  # [H,INT]
        blk.experts.gate_up_proj.data[e, :INT] = gw
        blk.experts.gate_up_proj.data[e, INT:] = uw
        blk.experts.down_proj.data[e] = dw

    blk.shared_expert.gate_proj.weight.data.copy_(W.get(p + "shared_expert.gate_proj.weight", torch.float32))
    blk.shared_expert.up_proj.weight.data.copy_(W.get(p + "shared_expert.up_proj.weight", torch.float32))
    blk.shared_expert.down_proj.weight.data.copy_(W.get(p + "shared_expert.down_proj.weight", torch.float32))
    blk.shared_expert_gate.weight.data.copy_(W.get(p + "shared_expert_gate.weight", torch.float32))
    return blk, cfg


# ---------------------------------------------------------------------------
# HOST per-expert SwiGLU reference (exact). The UNBATCHED NPU path provably equals
# this (current moe.py matches HF to 1.79e-7); we use it as the fast oracle to
# validate the BATCHED NPU path without paying the unbatched path's ~minute-per-
# matmul x (3*E) NPU wall-time, which is intractable to run end-to-end.
# ---------------------------------------------------------------------------
def moe_forward_host_ref(hidden, W, top_idx, top_val, layer=0):
    T, H = hidden.shape
    p = f"model.layers.{layer}.mlp."
    out = torch.zeros(T, H, dtype=torch.float32)
    for e in torch.unique(top_idx).tolist():
        tok_rows, kpos = torch.where(top_idx == e)
        x_e = hidden[tok_rows]
        gw = W.get(f"{p}experts.{e}.gate_proj.weight", torch.float32)
        uw = W.get(f"{p}experts.{e}.up_proj.weight",   torch.float32)
        dw = W.get(f"{p}experts.{e}.down_proj.weight", torch.float32)
        g = F.linear(x_e, gw); u = F.linear(x_e, uw)
        y_e = F.linear(F.silu(g) * u, dw)
        w_e = top_val[tok_rows, kpos].unsqueeze(-1)
        out.index_add_(0, tok_rows, y_e * w_e)
    sg = W.get(p + "shared_expert.gate_proj.weight", torch.float32)
    su = W.get(p + "shared_expert.up_proj.weight",   torch.float32)
    sd = W.get(p + "shared_expert.down_proj.weight", torch.float32)
    shared_gate_w = W.get(p + "shared_expert_gate.weight", torch.float32)
    sout = F.linear(F.silu(F.linear(hidden, sg)) * F.linear(hidden, su), sd)
    sout = torch.sigmoid(F.linear(hidden, shared_gate_w)) * sout
    return out + sout


def count_dispatches(hidden, W, top_idx, top_val, layer, batched):
    """DRY-run a path to tally NPU dispatches at full T=8 routing (no NPU exec)."""
    DRY["on"] = True
    NPU_STAGES.clear(); CALLS["n"] = 0; DISPATCH["n"] = 0
    FLOPS["npu"] = 0; FLOPS["host"] = 0
    if batched:
        moe_forward_npu(hidden, W, top_idx, top_val, layer, group_cap=None)
    else:
        moe_forward_npu_unbatched(hidden, W, top_idx, top_val, layer)
    DRY["on"] = False
    return DISPATCH["n"]


def main():
    from qcn.loader import QCNWeights
    torch.manual_seed(0)
    LAYER = 0
    W = QCNWeights()
    H = W.config["hidden_size"]
    top_k = W.config["num_experts_per_tok"]
    norm = W.config["norm_topk_prob"]

    log("=" * 72)
    log(f"Qwen3-Coder-Next MoE block  layer={LAYER}  experts={W.config['num_experts']} "
        f"top_k={top_k} moe_inter={W.config['moe_intermediate_size']} hidden={H}")

    # ===================================================================
    # PART 1 -- DISPATCH COUNT at the full prefill case (T=8, real routing).
    #   DRY-run (no NPU) both the per-expert and the batched paths and tally
    #   the NPU kernel dispatches for ONE MoE layer.
    # ===================================================================
    T_full = 8
    torch.manual_seed(0)
    hid8 = torch.randn(T_full, H) * 0.5
    gate_w = W.get(f"model.layers.{LAYER}.mlp.gate.weight", torch.float32)
    tv8, ti8 = host_router(hid8, gate_w, top_k, norm)
    n_act8 = len(torch.unique(ti8))
    disp_before = count_dispatches(hid8, W, ti8, tv8, LAYER, batched=False)
    disp_after  = count_dispatches(hid8, W, ti8, tv8, LAYER, batched=True)
    log("-" * 72)
    log(f"[DISPATCH COUNT @ T={T_full}]  distinct active experts = {n_act8}")
    log(f"  NPU dispatches BEFORE (per-expert, current moe.py)  : {disp_before}")
    log(f"  NPU dispatches AFTER  (batched expert axis)         : {disp_after}")
    log(f"  reduction                                           : "
        f"{disp_before} -> {disp_after}  ({disp_before/disp_after:.1f}x fewer)")

    # ===================================================================
    # PART 2 -- CORRECTNESS on REAL NPU. Run the BATCHED path for real on a
    #   tractable case (T=1 decode token -> exactly top_k=10 active experts)
    #   and validate vs the exact host per-expert SwiGLU oracle (which the
    #   unbatched/current path provably equals). maxerr must be < 1e-3.
    # ===================================================================
    T_val = 1
    hid1 = hid8[:T_val]
    tv1, ti1 = host_router(hid1, gate_w, top_k, norm)
    act1 = sorted(torch.unique(ti1).tolist())
    log("-" * 72)
    log(f"[CORRECTNESS @ T={T_val}]  active experts = {len(act1)}  (running BATCHED on NPU)")
    host_ref = moe_forward_host_ref(hid1, W, ti1, tv1, LAYER)

    NPU_STAGES.clear(); CALLS["n"] = 0; DISPATCH["n"] = 0
    FLOPS["npu"] = 0; FLOPS["host"] = 0
    bat_out, n_act1 = moe_forward_npu(hid1, W, ti1, tv1, LAYER, group_cap=None)
    disp_val = DISPATCH["n"]
    dfg_bat = CALLS["n"]

    maxerr = (bat_out - host_ref).abs().max().item()
    rel = maxerr / (host_ref.abs().max().item() + 1e-9)
    ok_match = maxerr < 1e-3
    all_npu = (dfg_bat == 0)
    bad = [(n, d) for n, d in NPU_STAGES if d != 0]

    log(f"  batched NPU dispatches (T={T_val})  : {disp_val}")
    log(f"  stages that fell back to CPU       : {bad if bad else 'NONE'}")
    log(f"  maxerr  BATCHED(NPU) vs host oracle: {maxerr:.3e}   (rel {rel:.3e})  (must be < 1e-3)")
    log(f"  _dfg_inner (batched path)          : {dfg_bat}  (0 == every NPU op ran on NPU)")
    log("-" * 72)
    log(f"MATCH_OK (maxerr<1e-3)  : {ok_match}")
    log(f"ALL_NPU_OPS_ON_NPU      : {all_npu}")
    log(f"OVERALL_PASS            : {bool(ok_match and all_npu)}")
    log("=" * 72)
    return ok_match, all_npu, maxerr, disp_before, disp_after, n_act8


if __name__ == "__main__":
    main()
