"""DECODE(T=1) per-layer wall-clock: batched vs original, to decide if batching helps.
Measures steady-state (2nd call, post-compile) for one MoE layer and one DeltaNet layer."""
import torch, time, sys
sys.argv = sys.argv[:1]
from qcn.loader import QCNWeights
from qcn import moe as _moe
from qcn.deltanet_layer import DeltaNetLayer
from qcn.deltanet_layer_looped import DeltaNetLayerLooped

W = QCNWeights()
cfg = W.config
H = cfg["hidden_size"]
torch.manual_seed(0)
hidden = torch.randn(1, 1, H, dtype=torch.float32) * 0.1  # decode: T=1


def _moe_weights(layer=0):
    return _moe  # moe funcs take W global? check signature uses W param


def timeit(fn, n=2):
    # first call compiles; time the LAST call (steady state)
    dt = None
    for _ in range(n):
        t = time.time(); fn(); dt = time.time() - t
    return dt


# ---------- MoE: batched vs per-expert (T=1, top-10) ----------
print("== MoE layer-0 decode T=1 ==", flush=True)
gate_w = W.get("model.layers.0.mlp.gate.weight", torch.float32)
top_idx, top_val = _moe.host_router(hidden.reshape(1, H), gate_w,
                                    cfg["num_experts_per_tok"], cfg.get("norm_topk_prob", True))
# load activated expert weights once via the module's loader path (moe_forward_npu loads internally)
try:
    t_un = timeit(lambda: _moe.moe_forward_npu_unbatched(hidden.reshape(1, H), W, top_idx, top_val, layer=0))
    print(f"  per-expert (unbatched): {t_un:.1f}s", flush=True)
except Exception as e:
    print(f"  per-expert FAIL: {type(e).__name__}: {str(e)[:120]}", flush=True)
try:
    t_b = timeit(lambda: _moe.moe_forward_npu(hidden.reshape(1, H), W, top_idx, top_val, layer=0))
    print(f"  batched experts       : {t_b:.1f}s", flush=True)
except Exception as e:
    print(f"  batched FAIL: {type(e).__name__}: {str(e)[:120]}", flush=True)

# ---------- DeltaNet: batched vs looped (T=1 decode step) ----------
print("== DeltaNet layer-0 decode T=1 ==", flush=True)
dn_b = DeltaNetLayer(cfg)
dn_l = DeltaNetLayerLooped(cfg)
w0 = {k: W.get(f"model.layers.0.linear_attn.{k}", torch.float32)
      for k in ["in_proj_qkvz.weight", "in_proj_ba.weight", "conv1d.weight",
                "A_log", "dt_bias", "norm.weight", "out_proj.weight"]}
# need a prior state for a true decode step; use zeros-init (prefill of 1 token)
try:
    t_lp = timeit(lambda: dn_l.forward(hidden, w0))
    print(f"  looped (32-head loop) : {t_lp:.1f}s", flush=True)
except Exception as e:
    print(f"  looped FAIL: {type(e).__name__}: {str(e)[:140]}", flush=True)
try:
    t_bb = timeit(lambda: dn_b.forward(hidden, w0))
    print(f"  head-batched          : {t_bb:.1f}s", flush=True)
except Exception as e:
    print(f"  head-batched FAIL: {type(e).__name__}: {str(e)[:140]}", flush=True)
print("DONE", flush=True)
