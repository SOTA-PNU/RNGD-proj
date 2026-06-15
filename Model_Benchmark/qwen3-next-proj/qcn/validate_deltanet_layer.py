#!/usr/bin/env python3
# Validate qcn/deltanet_layer.DeltaNetLayer (REAL Qwen3-Coder-Next config + REAL
# layer-0 weights) vs HF Qwen3NextGatedDeltaNet.
#   PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj \
#   RNGD_DEV=rngd:0 /home/jun/furiosa/bin/python qcn/validate_deltanet_layer.py
import os, sys, torch
sys.path.insert(0, "/home/jun/furiosa/lib/python3.12/site-packages")
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet

from qcn.loader import QCNWeights
from qcn.deltanet_layer import DeltaNetLayer, load_layer0_weights

torch.manual_seed(0)
T = 32

# ---- real config from the downloaded model ----
W = QCNWeights()
c = W.config
cfg = Qwen3NextConfig(
    hidden_size=c["hidden_size"],
    linear_num_value_heads=c["linear_num_value_heads"],
    linear_num_key_heads=c["linear_num_key_heads"],
    linear_key_head_dim=c["linear_key_head_dim"],
    linear_value_head_dim=c["linear_value_head_dim"],
    linear_conv_kernel_dim=c["linear_conv_kernel_dim"],
    rms_norm_eps=c["rms_norm_eps"],
    hidden_act=c["hidden_act"],
)

# ---- real layer-0 weights ----
w = load_layer0_weights(W, torch.float32, layer=0)

# ---- HF reference: instantiate, load_state_dict the real layer-0 weights ----
hf = Qwen3NextGatedDeltaNet(cfg, layer_idx=0).eval()
sd = {
    "in_proj_qkvz.weight": w["in_proj_qkvz"],
    "in_proj_ba.weight":   w["in_proj_ba"],
    "conv1d.weight":       w["conv1d_weight"],   # [conv_dim,1,Kc]
    "A_log":               w["A_log"],
    "dt_bias":             w["dt_bias"],
    "norm.weight":         w["norm_weight"],
    "out_proj.weight":     w["out_proj"],
}
missing, unexpected = hf.load_state_dict(sd, strict=False)
# only the (unused-in-forward) buffers may be missing; assert real params loaded
for k in sd:
    assert k not in missing, f"failed to load {k}"
print("HF load_state_dict: missing(non-param ok)=", list(missing), " unexpected=", list(unexpected))

hidden = torch.randn(1, T, cfg.hidden_size)
with torch.no_grad():
    hf_ret = hf(hidden)
hf_out = hf_ret[0] if isinstance(hf_ret, tuple) else hf_ret
hf_out = hf_out.reshape(-1, cfg.hidden_size) if hf_out.dim() == 3 else hf_out
hf_out = hf_out[:T]

# ---- our NPU-orchestrated layer (uses default chunk_size=64 == HF default) ----
layer = DeltaNetLayer(c, dev=os.environ.get("RNGD_DEV", "rngd:0"), chunk_size=64)
out, new_state = layer.forward(hidden, w)
out = out[0] if out.dim() == 3 else out

maxerr = (out - hf_out).abs().max().item()
ok = torch.allclose(out, hf_out, atol=1e-2)

print("=" * 74)
print(f"REAL config: hidden={cfg.hidden_size} nk={cfg.linear_num_key_heads} "
      f"nv={cfg.linear_num_value_heads} hk={cfg.linear_key_head_dim} "
      f"hv={cfg.linear_value_head_dim} conv_dim={layer.conv_dim} "
      f"n_rep={layer.n_rep} T={T} chunk={layer.chunk_size}")
print("-" * 74)
all_npu = layer.all_on_npu()
# summarize stage counts (don't print all 32*1 chunk lines)
from collections import Counter
stage_kinds = Counter()
fallbacks = []
for name, d in layer.npu_stages:
    base = name.split("[")[0]
    stage_kinds[base] += 1
    if d != 0:
        fallbacks.append((name, d))
print("NPU stages (count, all dfg_delta==0 == ran on NPU):")
for base, n in stage_kinds.items():
    print(f"   {base:22s} x{n}")
print("-" * 74)
print("HOST stages (inherently host -- no DSL op / inherently sequential):")
print("   g=-exp(A_log)*softplus(a+dt_bias) : softplus has NO DSL op (host)")
print("   T-matrix precompute               : tri-inverse refine + cumsum + decay_mask")
print("-" * 74)
print(f"new_state shape : {tuple(new_state.shape)}  (expected [{cfg.linear_num_value_heads},"
      f"{cfg.linear_key_head_dim},{cfg.linear_value_head_dim}])")
print(f"out shape       : {tuple(out.shape)}   hf {tuple(hf_out.shape)}")
print(f"FULL LAYER allclose(atol=1e-2) : {ok}")
print(f"FULL LAYER maxerr              : {maxerr:.3e}")
print(f"total _dfg_inner calls         : {DeltaNetLayer.total_dfg_calls()} (0 == every NPU stage on NPU)")
print(f"DeltaNet ops fallbacks         : {fallbacks if fallbacks else 'NONE'}")
print(f"ALL_DELTANET_OPS_ON_NPU        : {all_npu and DeltaNetLayer.total_dfg_calls() == 0}")
print(f"OVERALL_PASS                   : {bool(ok and all_npu and DeltaNetLayer.total_dfg_calls() == 0)}")
print("=" * 74)
