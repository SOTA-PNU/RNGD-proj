"""Compile Qwen3-Coder-Next COMPUTE pieces to REAL a6 CompiledGraph EDF blobs.

Uses the SAME producer the artifact build uses:
    furiosa.native_common.compiler.compile(mod, args, "renegade-8pe", target_ir="edf")
        -> CompileResult; r.graphs[0] is a CompiledGraph; cg.is_edf()==True;
           cg.serialize() -> a6 bytes (header: a1 63 45 64 66 'a6' 65 6e 6f 64 65 73
           = {cEdf: {6-key-map} enodes...}), exactly the format binary_bundle.zip
           <hash>.edf files use (builder.py:452-455 writes CompiledGraph.serialize()).

Each compiled subgraph -> serialize -> md5(bytes).edf saved into
    artifacts/qwen3-coder-next-fp8-rngd/_edf_blobs/
Records (piece, input-shapes, edf-hash, bytes, a6-header-confirmed, npu_node).

The DeltaNet recurrent step is attempted last and is EXPECTED to FAIL the a6
producer (no EDF) — documented as the one piece with no artifact blob.

Run:
  PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj RNGD_DEV=rngd:4 \
    /home/jun/furiosa/bin/python tk_kernels/compile_edf_blobs.py

Real Qwen3-Coder-Next-FP8 config (config.json):
  hidden 2048, 48 layers (full-attn every 4th: idx 3,7,..,47 -> 12; rest 36 DeltaNet),
  full-attn: 16 q heads / 2 kv heads, head_dim 256, partial-RoPE,
  DeltaNet: 16 key heads / 32 value heads, key/val head_dim 128, conv kernel 4,
  MoE: 512 experts, top-10, moe_intermediate 512, shared-expert 512, vocab 151936.

MEASURED RESULTS (2026-06-12, SDK 2026.2.0, renegade-8pe):
  a6 CompiledGraph EDF (compiler.compile, == binary_bundle.zip format) PRODUCED for:
    all Linear projections (q/k/v/o, in_proj_qkvz/ba, out_proj, MoE gate/up/down,
    router, shared, lm_head, embedding), full_attn_sdpa, moe_expert_swiglu,
    rmsnorm, gated_rmsnorm.  Header confirmed: ...a163456466 a6 656e6f646573.
  a6 producer FAILS (RuntimeError, no a6 EDF) for:
    - dn_conv1d_silu        : "O136 is not an operator that is yet supported"
    - dn_gate (sigmoid/exp/softplus standalone): marker_ops.rs:23 (dfg_import)
    - deltanet_recurrent_step: "conflict between concrete labels: Concrete(3)
                                and Concrete(1)"  <- the one piece with no a6 blob
  NOTE the a6 producer (compiler.compile) is STRICTER than the torch backend
  (furiosa.torch.custom_ops.edf.CompileModule.from_module). Via that torch path
  conv/gate/deltastep all compile, BUT they emit the a5 ir.Edf format
  (...a163456466 a5 ...) which is NOT byte-compatible with binary_bundle (missing
  the top-level 'binaries' field -> CompiledGraph.deserialize rejects it). So a5
  blobs cannot be dropped into the artifact; only the a6 ones above are real
  artifact-format EDF.
"""
import os, sys, hashlib, json, time, traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

from furiosa.native_common import compiler

TARGET_NPU = "renegade-8pe"
OUT_DIR = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen3-coder-next-fp8-rngd/_edf_blobs"
os.makedirs(OUT_DIR, exist_ok=True)

# --- real config ---
HIDDEN = 2048
VOCAB = 151936
N_Q_HEADS = 16
N_KV_HEADS = 2
HEAD_DIM = 256          # full-attn head_dim
DN_KEY_HEADS = 16
DN_VAL_HEADS = 32
DN_KEY_DIM = 128
DN_VAL_DIM = 128
CONV_K = 4
N_EXPERTS = 512
TOPK = 10
MOE_INTER = 512
SHARED_INTER = 512
EPS = 1e-6

# token count for a small representative prefill chunk
SEQ = 16

results = []


def hdr_is_a6(blob: bytes) -> bool:
    # a6 CompiledGraph: ... a1 63 'E' 'd' 'f' a6 65 'n' 'o' 'd' 'e' 's'
    # i.e. {cEdf: {6-key map} enodes...}. The 'a6' nibble = CBOR map of 6 entries.
    needle = bytes.fromhex("a163456466a6656e6f646573")  # a1 63 Edf a6 65 nodes
    return needle in blob[:64]


def compile_piece(name, mod, args, note=""):
    print(f"\n{'='*78}\n{name}   {note}\n{'='*78}")
    shapes = [tuple(a.shape) for a in args]
    dtypes = [str(a.dtype).replace("torch.", "") for a in args]
    rec = {
        "piece": name, "input_shapes": shapes, "input_dtypes": dtypes,
        "note": note, "status": None, "edf_hash": None, "bytes": None,
        "header_a6_confirmed": None, "npu_node": None, "error": None,
        "compile_s": None,
    }
    t0 = time.time()
    try:
        mod = mod.eval()
        r = compiler.compile(mod, tuple(args), TARGET_NPU, target_ir="edf")
        rec["compile_s"] = round(time.time() - t0, 3)
        cg = r.graphs[0]
        is_edf = cg.is_edf()
        blob = cg.serialize()
        a6 = hdr_is_a6(blob)
        h = hashlib.md5(blob).hexdigest()
        out = os.path.join(OUT_DIR, f"{h}.edf")
        with open(out, "wb") as f:
            f.write(blob)
        rec.update(status="ok", edf_hash=h, bytes=len(blob),
                   header_a6_confirmed=bool(a6), npu_node=bool(is_edf))
        print(f"  COMPILE OK in {rec['compile_s']}s -> CompiledGraph (n_graphs={len(r.graphs)})")
        print(f"  is_edf={is_edf}  bytes={len(blob)}  a6_header={a6}")
        print(f"  head={blob[:24].hex()}")
        print(f"  saved {out}")
    except Exception as e:
        rec["compile_s"] = round(time.time() - t0, 3)
        rec["status"] = "FAIL"
        tb = traceback.format_exc()
        # keep the most informative tail
        rec["error"] = (type(e).__name__ + ": " + str(e))[:600]
        print(f"  COMPILE FAILED after {rec['compile_s']}s: {type(e).__name__}")
        for ln in (str(e) + "\n" + tb).splitlines()[-14:]:
            print("   |", ln[:200])
    results.append(rec)
    return rec


# ============================================================
# (a) Linear projection shapes (same op, different dims)
# ============================================================
class Lin(nn.Module):
    def __init__(self, i, o, bias=False):
        super().__init__()
        self.fc = nn.Linear(i, o, bias=bias)
    def forward(self, x):
        return self.fc(x)


def linear_pieces():
    x = torch.randn(SEQ, HIDDEN)
    # full-attn projections
    compile_piece("lin.q_proj", Lin(HIDDEN, N_Q_HEADS * HEAD_DIM), (x,),
                  f"{HIDDEN}->{N_Q_HEADS*HEAD_DIM} (16 q heads x hd256)")
    compile_piece("lin.kv_proj", Lin(HIDDEN, 2 * N_KV_HEADS * HEAD_DIM), (x,),
                  f"{HIDDEN}->{2*N_KV_HEADS*HEAD_DIM} (k+v, 2 kv heads x hd256)")
    compile_piece("lin.o_proj", Lin(N_Q_HEADS * HEAD_DIM, HIDDEN), (torch.randn(SEQ, N_Q_HEADS*HEAD_DIM),),
                  f"{N_Q_HEADS*HEAD_DIM}->{HIDDEN}")
    # DeltaNet in_proj qkvz (q,k = key heads*key_dim; v,z = val heads*val_dim) and ba (b,a gates)
    qkvz_out = 2 * DN_KEY_HEADS * DN_KEY_DIM + 2 * DN_VAL_HEADS * DN_VAL_DIM
    compile_piece("lin.in_proj_qkvz", Lin(HIDDEN, qkvz_out), (x,),
                  f"{HIDDEN}->{qkvz_out} (q,k:16x128 + v,z:32x128)")
    ba_out = 2 * DN_VAL_HEADS
    compile_piece("lin.in_proj_ba", Lin(HIDDEN, ba_out), (x,),
                  f"{HIDDEN}->{ba_out} (b,a gate scalars per val head)")
    compile_piece("lin.dn_out_proj", Lin(DN_VAL_HEADS * DN_VAL_DIM, HIDDEN),
                  (torch.randn(SEQ, DN_VAL_HEADS * DN_VAL_DIM),),
                  f"{DN_VAL_HEADS*DN_VAL_DIM}->{HIDDEN}")
    # MoE expert gate/up/down (per-expert), router gate, shared expert
    compile_piece("lin.moe_gate", Lin(HIDDEN, MOE_INTER), (x,), f"{HIDDEN}->{MOE_INTER} expert gate_proj")
    compile_piece("lin.moe_up", Lin(HIDDEN, MOE_INTER), (x,), f"{HIDDEN}->{MOE_INTER} expert up_proj")
    compile_piece("lin.moe_down", Lin(MOE_INTER, HIDDEN), (torch.randn(SEQ, MOE_INTER),),
                  f"{MOE_INTER}->{HIDDEN} expert down_proj")
    compile_piece("lin.router_gate", Lin(HIDDEN, N_EXPERTS), (x,), f"{HIDDEN}->{N_EXPERTS} router")
    compile_piece("lin.shared_gate", Lin(HIDDEN, SHARED_INTER), (x,), f"{HIDDEN}->{SHARED_INTER} shared gate")


def lm_head_piece():
    # big: vocab 151936 -> ~1.2GB fp32 blob (slow). Same Linear op as above, larger.
    compile_piece("lin.lm_head", Lin(HIDDEN, VOCAB), (torch.randn(SEQ, HIDDEN),),
                  f"{HIDDEN}->{VOCAB} lm_head (tied to embedding)")


class Embed(nn.Module):
    def __init__(self, v, h):
        super().__init__()
        self.emb = nn.Embedding(v, h)
    def forward(self, ids):
        return self.emb(ids)


def embed_piece():
    ids = torch.randint(0, VOCAB, (SEQ,), dtype=torch.long)
    compile_piece("embedding", Embed(VOCAB, HIDDEN), (ids,),
                  f"Embedding({VOCAB},{HIDDEN}) gather")


# ============================================================
# (b) Full-attention SDPA block (GQA 16/2, hd256)
# ============================================================
class FullAttnSDPA(nn.Module):
    """q@k^T scaled + softmax + @v with GQA repeat (16 q / 2 kv heads)."""
    def __init__(self, nq, nkv, hd):
        super().__init__()
        self.nq, self.nkv, self.hd = nq, nkv, hd
        self.rep = nq // nkv
        self.scale = hd ** -0.5
    def forward(self, q, k, v):
        # q: (nq, S, hd) ; k,v: (nkv, S, hd)
        k = k.repeat_interleave(self.rep, dim=0)
        v = v.repeat_interleave(self.rep, dim=0)
        att = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, v)
        return out


def sdpa_piece():
    q = torch.randn(N_Q_HEADS, SEQ, HEAD_DIM)
    k = torch.randn(N_KV_HEADS, SEQ, HEAD_DIM)
    v = torch.randn(N_KV_HEADS, SEQ, HEAD_DIM)
    compile_piece("full_attn_sdpa", FullAttnSDPA(N_Q_HEADS, N_KV_HEADS, HEAD_DIM), (q, k, v),
                  "GQA 16q/2kv hd256: q@kT*scale -> softmax -> @v")


# ============================================================
# (c) MoE expert SwiGLU: down(silu(gate(x)) * up(x))
# ============================================================
class ExpertSwiGLU(nn.Module):
    def __init__(self, h, inter):
        super().__init__()
        self.gate = nn.Linear(h, inter, bias=False)
        self.up = nn.Linear(h, inter, bias=False)
        self.down = nn.Linear(inter, h, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def moe_swiglu_piece():
    x = torch.randn(SEQ, HIDDEN)
    compile_piece("moe_expert_swiglu", ExpertSwiGLU(HIDDEN, MOE_INTER), (x,),
                  f"down(silu(gate(x))*up(x)) inter={MOE_INTER}")


# ============================================================
# (d) conv1d + SiLU (depthwise, K=4) — DeltaNet short conv
# ============================================================
class DepthwiseConvSiLU(nn.Module):
    def __init__(self, ch, k):
        super().__init__()
        # depthwise: groups == channels
        self.conv = nn.Conv1d(ch, ch, kernel_size=k, groups=ch, padding=k - 1, bias=True)
        self.k = k
    def forward(self, x):
        # x: (ch, L) -> add batch dim
        x = x.unsqueeze(0)
        y = self.conv(x)[..., : x.shape[-1]]
        return F.silu(y).squeeze(0)


def conv_piece():
    # DeltaNet conv channels = q+k+v projected dims (mix dim). Use key+key+val heads dim as channels.
    ch = 2 * DN_KEY_HEADS * DN_KEY_DIM + DN_VAL_HEADS * DN_VAL_DIM
    x = torch.randn(ch, SEQ)
    compile_piece("dn_conv1d_silu", DepthwiseConvSiLU(ch, CONV_K), (x,),
                  f"depthwise conv1d K={CONV_K} ch={ch} + SiLU")


# ============================================================
# (e) RMSNorm / gated RMSNorm
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, h, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(h))
        self.eps = eps
    def forward(self, x):
        v = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(v + self.eps)
        return x * self.weight


class GatedRMSNorm(nn.Module):
    """DeltaNet output gated RMSNorm: rmsnorm(x) * silu(gate)."""
    def __init__(self, h, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(h))
        self.eps = eps
    def forward(self, x, gate):
        v = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(v + self.eps)
        return x * self.weight * F.silu(gate)


def norm_pieces():
    x = torch.randn(SEQ, HIDDEN)
    compile_piece("rmsnorm", RMSNorm(HIDDEN, EPS), (x,), f"RMSNorm hidden={HIDDEN}")
    g = torch.randn(SEQ, DN_VAL_HEADS * DN_VAL_DIM)
    xv = torch.randn(SEQ, DN_VAL_HEADS * DN_VAL_DIM)
    compile_piece("gated_rmsnorm", GatedRMSNorm(DN_VAL_HEADS * DN_VAL_DIM, EPS), (xv, g),
                  f"rmsnorm(x)*silu(gate) dim={DN_VAL_HEADS*DN_VAL_DIM}")


# ============================================================
# (f) DeltaNet gate (sigmoid / exp / softplus) standalone
# ============================================================
class DNGate(nn.Module):
    def __init__(self, nv):
        super().__init__()
        self.A_log = nn.Parameter(torch.randn(nv))
        self.dt_bias = nn.Parameter(torch.randn(nv))
    def forward(self, b_, a_):
        # b_: beta logits, a_: decay logits ; (S, nv)
        beta = torch.sigmoid(b_)
        sp = F.softplus(a_ + self.dt_bias)
        g = -torch.exp(self.A_log) * sp
        return beta, g


def gate_piece():
    b_ = torch.randn(SEQ, DN_VAL_HEADS)
    a_ = torch.randn(SEQ, DN_VAL_HEADS)
    compile_piece("dn_gate", DNGate(DN_VAL_HEADS), (b_, a_),
                  "sigmoid(beta), -exp(A_log)*softplus(a+dt_bias)")


# ============================================================
# (3) DeltaNet recurrent step — EXPECTED a6-PRODUCER FAILURE (no artifact EDF)
# ============================================================
class DeltaStep(nn.Module):
    """One recurrent step: state += k⊗delta ; out = einsum(state,q).
    The 4-D outer-product state accumulate is what the a6 producer rejects:
    via compiler.compile -> RuntimeError "conflict between concrete labels:
    Concrete(3) and Concrete(1)" (the einsum lowering cannot reconcile the
    state's dk/dv axes with the broadcast). So there is NO a6 CompiledGraph for
    this piece -> it cannot be a binary_bundle .edf. (The torch backend path
    furiosa.torch CompileModule.from_module DOES lower it, but only to the a5
    ir.Edf format, which is not the artifact format.)"""
    def forward(self, state, q_t, k_t, v_t, g_t, beta_t):
        g_t = g_t.exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta_t.unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out_t = (state * q_t.unsqueeze(-1)).sum(dim=-2)
        return state, out_t


def deltastep_piece():
    nv, dk, dv = DN_VAL_HEADS, DN_KEY_DIM, DN_VAL_DIM
    state = torch.randn(nv, dk, dv)
    q_t = torch.randn(nv, dk); k_t = torch.randn(nv, dk)
    v_t = torch.randn(nv, dv); g_t = torch.randn(nv); beta_t = torch.randn(nv)
    compile_piece("deltanet_recurrent_step", DeltaStep(),
                  (state, q_t, k_t, v_t, g_t, beta_t),
                  "EXPECTED a6 FAIL: state += k(x)delta ; out=einsum(state,q) "
                  "-> 'conflict between concrete labels' (no artifact EDF)")


def main():
    print(f"compiler full_version: {compiler.full_version()}")
    print(f"target_npu: {TARGET_NPU}   out_dir: {OUT_DIR}")

    # fast/important compute pieces first
    linear_pieces()
    sdpa_piece()
    moe_swiglu_piece()
    conv_piece()
    norm_pieces()
    gate_piece()
    # heavy vocab-151936 pieces last (~1.2GB fp32 blobs each, slow)
    lm_head_piece()
    embed_piece()
    # the one with no artifact EDF, run last so its failure doesn't block earlier blobs
    deltastep_piece()

    # ---- summary ----
    print(f"\n{'#'*78}\nSUMMARY\n{'#'*78}")
    ok = [r for r in results if r["status"] == "ok"]
    bad = [r for r in results if r["status"] != "ok"]
    for r in results:
        if r["status"] == "ok":
            print(f"  OK   {r['piece']:24s} {r['bytes']:>10} B  a6={r['header_a6_confirmed']}  npu={r['npu_node']}  {r['edf_hash']}")
        else:
            print(f"  FAIL {r['piece']:24s} {r['error']}")
    print(f"\n  total={len(results)}  ok={len(ok)}  fail={len(bad)}")

    summary_path = os.path.join(OUT_DIR, "_compile_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "compiler_version": compiler.full_version(),
            "target_npu": TARGET_NPU,
            "config": {
                "hidden": HIDDEN, "vocab": VOCAB, "n_q_heads": N_Q_HEADS,
                "n_kv_heads": N_KV_HEADS, "head_dim": HEAD_DIM,
                "dn_key_heads": DN_KEY_HEADS, "dn_val_heads": DN_VAL_HEADS,
                "dn_key_dim": DN_KEY_DIM, "dn_val_dim": DN_VAL_DIM,
                "conv_k": CONV_K, "n_experts": N_EXPERTS, "topk": TOPK,
                "moe_inter": MOE_INTER, "seq": SEQ,
            },
            "results": results,
        }, f, indent=2)
    print(f"\n  summary -> {summary_path}")


if __name__ == "__main__":
    main()
