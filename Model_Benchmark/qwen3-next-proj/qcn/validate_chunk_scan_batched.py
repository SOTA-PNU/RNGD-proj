#!/usr/bin/env python3
# Validate the HEAD-BATCHED DeltaNetLayer (rewritten _chunk_scan -> dn_chunk_full_mh,
# ALL 32 value-heads in ONE NPU dispatch/chunk) against the ORIGINAL per-head LOOPED
# DeltaNetLayerLooped (dn_chunk_full per head -> 32 dispatches/chunk) on REAL layer-0
# weights at T=32.  Same math => must match (maxerr < 1e-4, ideally ~1e-6).
#
# Also counts NPU dispatches (_NPU_DISPATCHES) BEFORE (looped) vs AFTER (batched) for
# one layer, and reports the chunk-scan reduction (~32x).  Keeps _dfg_inner (_CALLS)
# == 0 so everything stays on the NPU.
#
#   PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj \
#   RNGD_DEV=rngd:2 /home/jun/furiosa/bin/python qcn/validate_chunk_scan_batched.py
import os, sys, torch

from qcn.loader import QCNWeights
import qcn.deltanet_layer as dl
from qcn.deltanet_layer import DeltaNetLayer, load_layer0_weights
from qcn.deltanet_layer_looped import DeltaNetLayerLooped

torch.manual_seed(0)
T = 32
DEV = os.environ.get("RNGD_DEV", "rngd:2")

W = QCNWeights()
c = W.config
w = load_layer0_weights(W, torch.float32, layer=0)
hidden = torch.randn(1, T, c["hidden_size"])

def dispatch_snapshot():
    return dl._NPU_DISPATCHES["n"], dict(dl._NPU_DISPATCHES["by_yaml"])

# ---------------- LOOPED (current / BEFORE) ----------------
d0_total, _ = dispatch_snapshot()
fb0 = dl._CALLS["n"]
looped = DeltaNetLayerLooped(c, dev=DEV, chunk_size=64)
out_loop, state_loop = looped.forward(hidden, w)
out_loop = out_loop[0] if out_loop.dim() == 3 else out_loop
d1_total, by1 = dispatch_snapshot()
fb1 = dl._CALLS["n"]
disp_loop = d1_total - d0_total
chunk_loop = by1.get("dn_chunk_full.yaml", 0)
fb_loop = fb1 - fb0

# ---------------- BATCHED (rewritten / AFTER) ----------------
d2_total, by2a = dispatch_snapshot()
fb2 = dl._CALLS["n"]
batched = DeltaNetLayer(c, dev=DEV, chunk_size=64)
out_b, state_b = batched.forward(hidden, w)
out_b = out_b[0] if out_b.dim() == 3 else out_b
d3_total, by2 = dispatch_snapshot()
fb3 = dl._CALLS["n"]
disp_batched = d3_total - d2_total
chunk_batched = by2.get("dn_chunk_full_mh.yaml", 0) - by2a.get("dn_chunk_full_mh.yaml", 0)
fb_batched = fb3 - fb2

# ---------------- compare ----------------
maxerr = (out_b - out_loop).abs().max().item()
state_err = (state_b - state_loop).abs().max().item()
out_ok = torch.allclose(out_b, out_loop, atol=1e-4)

NC = (T + 63) // 64   # chunks per head at chunk_size=64, T=32 -> 1 chunk

print("=" * 74)
print(f"REAL layer-0 weights | nv={c['linear_num_value_heads']} "
      f"hk={c['linear_key_head_dim']} hv={c['linear_value_head_dim']} "
      f"T={T} chunk_size=64  NC(chunks/head)={NC}  dev={DEV}")
print("-" * 74)
print("OUTPUT match (batched vs looped, same math):")
print(f"  out   maxerr : {maxerr:.3e}   allclose(1e-4): {out_ok}")
print(f"  state maxerr : {state_err:.3e}")
print("-" * 74)
print("NPU DISPATCH COUNT (one whole layer):")
print(f"  BEFORE (looped)  total dispatches : {disp_loop}")
print(f"           chunk-scan dispatches    : {chunk_loop}  (dn_chunk_full.yaml, == nv*NC)")
print(f"  AFTER  (batched) total dispatches : {disp_batched}")
print(f"           chunk-scan dispatches    : {chunk_batched}  (dn_chunk_full_mh.yaml, == NC)")
red = (chunk_loop / chunk_batched) if chunk_batched else float('nan')
print(f"  chunk-scan reduction             : {red:.1f}x  ({chunk_loop} -> {chunk_batched})")
print(f"  whole-layer dispatch reduction   : {disp_loop} -> {disp_batched}")
print("-" * 74)
print("ON-NPU PROOF (_dfg_inner CPU-fallback delta; 0 == all on NPU):")
print(f"  looped  fallbacks : {fb_loop}")
print(f"  batched fallbacks : {fb_batched}")
all_npu = (fb_loop == 0 and fb_batched == 0)
print("-" * 74)
print(f"DISPATCH_BEFORE (chunk-scan) : {chunk_loop}")
print(f"DISPATCH_AFTER  (chunk-scan) : {chunk_batched}")
print(f"MAXERR_VS_CURRENT            : {maxerr:.3e}")
print(f"OVERALL_PASS                 : {bool(out_ok and all_npu and chunk_batched < chunk_loop)}")
print("=" * 74)
