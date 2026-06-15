#!/usr/bin/env python3
# Validate the LOOPED DeltaNet layer (qcn/deltanet_layer_looped.DeltaNetLayerLooped,
# the ACTIVE chunk-scan in model.py) with the internal chunk matmuls moved onto the
# FAST DPE (systolic) engine -- vs HF Qwen3NextGatedDeltaNet, real layer-0 weights,
# T=32. DPE is bf16 systolic so the bar is atol 1e-2 (NOT 1e-3).
#
# Runs ONE chunk-scan yaml per invocation (separate process per variant avoids
# cross-graph NPU device-state flakiness). Prints maxerr_vs_hf, chunk-scan dfg_delta
# (0 == on NPU), and the full-layer wall time so the caller can compute speedup.
#
#   PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj RNGD_DEV=rngd:3 \
#   /home/jun/furiosa/bin/python qcn/validate_deltanet_dpe.py <chunk_yaml>
import os, sys, time, torch
sys.path.insert(0, "/home/jun/furiosa/lib/python3.12/site-packages")
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet

from qcn.loader import QCNWeights
from qcn.deltanet_layer import load_layer0_weights
import qcn.deltanet_layer_looped as dll
from qcn.deltanet_layer_looped import DeltaNetLayerLooped
import qcn.deltanet_layer as dl

CHUNK_YAML = sys.argv[1] if len(sys.argv) > 1 else "dn_chunk_full.yaml"
DEV = os.environ.get("RNGD_DEV", "rngd:3")
torch.manual_seed(0)
T = 32

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
w = load_layer0_weights(W, torch.float32, layer=0)

hf = Qwen3NextGatedDeltaNet(cfg, layer_idx=0).eval()
sd = {
    "in_proj_qkvz.weight": w["in_proj_qkvz"], "in_proj_ba.weight": w["in_proj_ba"],
    "conv1d.weight": w["conv1d_weight"], "A_log": w["A_log"], "dt_bias": w["dt_bias"],
    "norm.weight": w["norm_weight"], "out_proj.weight": w["out_proj"],
}
missing, _ = hf.load_state_dict(sd, strict=False)
for k in sd:
    assert k not in missing, f"failed to load {k}"

hidden = torch.randn(1, T, cfg.hidden_size)
with torch.no_grad():
    hf_ret = hf(hidden)
hf_out = hf_ret[0] if isinstance(hf_ret, tuple) else hf_ret
hf_out = hf_out.reshape(-1, cfg.hidden_size) if hf_out.dim() == 3 else hf_out
hf_out = hf_out[:T]

# remap the looped layer's hardcoded "dn_chunk_full.yaml" to the chosen variant
_real_npu = dll._npu
def _remap(yaml, inputs, dev):
    if yaml == "dn_chunk_full.yaml":
        yaml = CHUNK_YAML
    return _real_npu(yaml, inputs, dev)
dll._npu = _remap

# warm-up (compile) -- not timed
DeltaNetLayerLooped(c, dev=DEV).forward(hidden, w)

layer = DeltaNetLayerLooped(c, dev=DEV, chunk_size=64)
out, new_state = layer.forward(hidden, w)
out = out[0] if out.dim() == 3 else out
maxerr = (out - hf_out).abs().max().item()
relmean = ((out - hf_out).abs() / (hf_out.abs() + 1e-6)).mean().item()
scan_fb = sum(d for n, d in layer.npu_stages if n.startswith("chunk-scan") and d != 0)

reps = 3
t0 = time.perf_counter()
for _ in range(reps):
    DeltaNetLayerLooped(c, dev=DEV, chunk_size=64).forward(hidden, w)
wall = (time.perf_counter() - t0) / reps

ok = maxerr <= 1e-2
print(f"CHUNK_YAML={CHUNK_YAML} dev={DEV} T={T}")
print(f"maxerr_vs_hf={maxerr:.4e} relmean={relmean:.4e} allclose_1e-2={ok} "
      f"chunk_dfg_delta={scan_fb} wall_ms={wall*1e3:.2f}")
print(f"RESULT_LINE yaml={CHUNK_YAML} works={bool(ok and scan_fb==0)} "
      f"maxerr={maxerr:.4e} dfg={scan_fb} wall_ms={wall*1e3:.2f}")
