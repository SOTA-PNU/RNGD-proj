#!/usr/bin/env python3
# =============================================================================
# build_artifact.py  --  package the qwen3-coder-next-fp8 HOST-LOOP system into
# a self-contained, loadable "artifact" directory.
#
# WHY a host-loop artifact (and not a furiosa-llm EDF artifact):
#   furiosa-llm `build` of model_type=qwen3_next is BLOCKED in 2026.2.0 -- its
#   compiler kernelizer cannot lower the DeltaNet (linear-attention) token mixer
#   through the FX -> kernelize path (the TP splitter hits an IndexError on the
#   KV-less DeltaNet, and the EDF runtime can't carry the recurrent state).  So
#   there is NO binary_bundle.zip of compiled EDF pipelines for this model.
#
#   Instead we run the model via a HOST LOOP (qcn/model.py: 48-layer streamed
#   forward) where every token-mixer matmul (DeltaNet chunk scan, full-attn
#   SDPA, MoE SwiGLU) is dispatched to the RNGD NPU through HAND-AUTHORED
#   TacticKernel YAMLs (tk_kernels/dn_*.yaml) via furiosa.torch.TacticKernelModule.
#   The compute is genuinely on-device; only the orchestration loop is on host.
#
#   This script packages that working system into an artifact dir that mirrors a
#   real furiosa-llm artifact's *shape* where sensible (artifact.json manifest
#   with model.model_metadata.model_type / hf_configs, config.json,
#   generation_config.json, tokenizer*), but is HONEST that runtime == host-loop
#   (artifact.json["runtime"] == "host-loop", model.runtime_kind == "host-loop"),
#   carries the NPU kernel YAMLs the host loop dispatches, names the qcn module
#   entry-point, and points at the FP8 weights in the HF cache instead of
#   bundling 75GB of safetensors.
#
# Usage:
#   PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj \
#   /home/jun/furiosa/bin/python qcn/build_artifact.py
#       [--out  <artifact dir>]   (default: rngd-npu/artifacts/qwen3-coder-next-fp8-rngd)
#       [--copy-weights]          (also copy/symlink the 75GB safetensors in)
#       [--link]                  (symlink HF snapshot files instead of copying)
# =============================================================================
import os
import sys
import json
import glob
import time
import shutil
import hashlib
import argparse
import subprocess

REPO = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj"
TK_DIR = os.path.join(REPO, "tk_kernels")
# EDF blob compiler + packager (used by --emit-edf to (re)build binary_bundle.zip)
EDF_COMPILE = os.path.join(TK_DIR, "compile_edf_blobs.py")
EDF_SPLIT = os.path.join(TK_DIR, "emit_dn_split_blobs.py")  # DeltaNet recurrent/conv/gate 분해 블롭
EDF_PACK = os.path.join(TK_DIR, "pack_edf_bundle.py")
DEFAULT_OUT = ("/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/"
               "qwen3-coder-next-fp8-rngd")
HF_SNAP_GLOB = ("/home/jun/.cache/huggingface/hub/"
                "models--Qwen--Qwen3-Coder-Next-FP8/snapshots/*/")

# --- the NPU kernels the host loop dispatches (model.py import chain) ---------
# deltanet_layer_looped: dn_linear(_dpe), dn_chunk_full(_dpe2), dn_conv1d,
#                        dn_l2norm, dn_gnorm, dn_gate
# attn_layer:            dn_linear(_dpe)
# moe (unbatched path):  dn_linear(_dpe), dn_gate
# We package BOTH the EinsumByVe baseline twins AND the EinsumByDpe fast twins so
# the artifact runs whether QCN_DPE is 0 or 1.
KERNELS = [
    "dn_linear.yaml",          # EinsumByVe matmul (baseline)
    "dn_linear_dpe.yaml",      # EinsumByDpe matmul (fast, QCN_DPE=1)
    "dn_chunk_full.yaml",      # DeltaNet chunk scan (baseline)
    "dn_chunk_full_dpe2.yaml", # DeltaNet chunk scan on DPE (QCN_DPE=1)
    "dn_conv1d.yaml",          # short causal conv tail
    "dn_l2norm.yaml",          # per-head L2 normalize of q/k
    "dn_gnorm.yaml",           # gated RMSNorm output of DeltaNet
    "dn_gate.yaml",            # sigmoid/silu gating (SwiGLU + gates)
]

# host-loop python sources that constitute the runtime (recorded in manifest)
QCN_SOURCES = [
    "model.py", "loader.py", "deltanet_layer_looped.py", "deltanet_layer.py",
    "attn_layer.py", "moe.py", "generate.py", "serve.py", "serve_mc.py",
]


def _snap():
    d = sorted(glob.glob(HF_SNAP_GLOB))
    assert d, f"HF snapshot not found under {HF_SNAP_GLOB}"
    return d[-1]


def _sha256(path, cap=None):
    """sha256 of a file; cap=bytes to hash only a prefix (for big weights)."""
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            if cap and n >= cap:
                break
    return h.hexdigest()


def _place(src, dst, link):
    """Copy or symlink src -> dst (resolving HF blob symlinks to real files)."""
    real = os.path.realpath(src)
    if os.path.lexists(dst):
        os.remove(dst)
    if link:
        os.symlink(real, dst)
    else:
        shutil.copy2(real, dst)
    return real


def _binary_bundle_from_existing(out_dir):
    """Build the artifact.json 'binary_bundle' section from an already-packed
    binary_bundle.zip + binary_bundle_manifest.json, or return None if absent."""
    bundle = os.path.join(out_dir, "binary_bundle.zip")
    mpath = os.path.join(out_dir, "binary_bundle_manifest.json")
    if not (os.path.exists(bundle) and os.path.exists(mpath)):
        return None
    bm = json.load(open(mpath))
    pieces_with = [v["piece"] for v in bm.get("blobs", {}).values() if v.get("piece")]
    n_ok = sum(1 for v in bm.get("blobs", {}).values() if v.get("deserialize_ok"))
    n = bm.get("n_blobs", len(bm.get("blobs", {})))
    # DeltaNet 분해 블롭(dn_recur_*)이 들어 있으면 compute-complete; 아니면 옛 partial.
    compute_complete = any(p.startswith("dn_recur_") for p in pieces_with)
    return {
        "kind": "edf-split (compute-complete)" if compute_complete else "partial-edf",
        "zip": "binary_bundle.zip",
        "manifest": "binary_bundle_manifest.json",
        "compression": bm.get("compression", "ZIP_STORED"),
        "format": bm.get("format"),
        "producer": ("tk_kernels/compile_edf_blobs.py (base compute) + "
                     "tk_kernels/emit_dn_split_blobs.py (DeltaNet recurrent/conv/gate split) "
                     "-> tk_kernels/pack_edf_bundle.py (zip + validate). "
                     "Regenerate with: qcn/build_artifact.py --emit-edf"),
        "n_blobs": n,
        "zip_bytes": os.path.getsize(bundle),
        "zip_sha256": _sha256(bundle),
        "validated": (f"{n_ok}/{n} deserialize via CompiledGraph.deserialize as "
                      "valid a6 CompiledGraph (is_edf=True)"),
        "pieces_with_edf": pieces_with,
        "pieces_without_edf": [] if compute_complete else [
            {"piece": "dn_conv1d_silu",
             "reason": "O136 is not an operator the 2026.2.0 compiler supports"},
            {"piece": "dn_gate",
             "reason": "aten::log1p not importable by the a6 producer"},
            {"piece": "deltanet_recurrent_step",
             "reason": ("DeltaNet outer-product recurrence panics with 'conflict "
                        "between concrete labels: Concrete(3) and Concrete(1)'; no "
                        "a6 EDF + no runtime recurrent-state slot -> stays host-loop")},
        ],
        "scope_note": bm.get("scope",
            ("COMPUTE-COMPLETE EDF: every compute piece incl. DeltaNet recurrence "
             "(split form) is a real a6 EDF blob; remaining limit is deploy-only "
             "(serve runtime has no cross-step recurrent-state pool) -> host-loop."
             if compute_complete else
             "PARTIAL EDF: real on-device EDF blobs for all compilable compute "
             "(matmuls/attention/MoE/norms); DeltaNet recurrence + orchestration "
             "stay host-loop.")),
    }


def emit_edf(out_dir, recompile=False):
    """(Re)generate binary_bundle.zip of REAL a6 CompiledGraph EDF blobs.

    The compute pieces the 2026.2.0 compiler CAN lower (all Linear projections,
    full-attention SDPA, MoE SwiGLU, RMSNorm/gated RMSNorm, lm_head, embedding)
    are compiled to a6 EDF via tk_kernels/compile_edf_blobs.py -> _edf_blobs/, then
    packed + round-trip-validated (CompiledGraph.deserialize) into a flat
    ZIP_STORED binary_bundle.zip + binary_bundle_manifest.json by
    tk_kernels/pack_edf_bundle.py.

    The DeltaNet recurrent step / conv1d / gate pieces cannot compile to a6 in
    2026.2.0 -> they have no blob and remain host-loop (see manifest scope_note).

    recompile=True re-runs the (slow, NPU) compile pass; otherwise it reuses
    existing _edf_blobs/*.edf and only (re)packs the zip + manifest.

    Returns (bundle_path, n_blobs) or (None, 0) if no blobs are available.
    """
    blob_dir = os.path.join(out_dir, "_edf_blobs")
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", REPO)
    env.setdefault("RNGD_DEV", "rngd:4")  # pick a FREE PE (0-3 often busy)

    existing = glob.glob(os.path.join(blob_dir, "*.edf"))
    if recompile or not existing:
        print("  [emit-edf] compiling a6 EDF blobs via compile_edf_blobs.py "
              "(slow, uses NPU) ...")
        rc = subprocess.call([sys.executable, EDF_COMPILE], env=env, cwd=REPO)
        if rc != 0:
            print(f"  [emit-edf] WARNING: compile_edf_blobs.py exited {rc} "
                  "(some pieces are EXPECTED to fail a6 -- see _MASTER_summary.json)")
        # DeltaNet recurrent/conv/gate 를 단일-op 분해 a6 블롭으로 추가(2026-06-15):
        # 통째 그래프는 a6 거부('concrete labels')지만 op별로 쪼개면 통과 → 25블롭
        # compute-complete. 이게 없으면 binary_bundle 이 17블롭 partial 로 되돌아간다.
        print("  [emit-edf] adding DeltaNet split blobs via emit_dn_split_blobs.py ...")
        rc2 = subprocess.call([sys.executable, EDF_SPLIT], env=env, cwd=REPO)
        if rc2 != 0:
            print(f"  [emit-edf] WARNING: emit_dn_split_blobs.py exited {rc2}")
        existing = glob.glob(os.path.join(blob_dir, "*.edf"))

    if not existing:
        print("  [emit-edf] no .edf blobs found -> skipping binary_bundle.zip")
        return None, 0

    print(f"  [emit-edf] packing {len(existing)} .edf blobs -> binary_bundle.zip "
          "and validating (CompiledGraph.deserialize) ...")
    rc = subprocess.call([sys.executable, EDF_PACK], env=env, cwd=REPO)
    if rc != 0:
        raise RuntimeError(
            f"pack_edf_bundle.py failed (rc={rc}): bundle did not validate")
    bundle = os.path.join(out_dir, "binary_bundle.zip")
    n = len(existing)
    print(f"  [emit-edf] binary_bundle.zip OK "
          f"({os.path.getsize(bundle)} B, {n} blobs)")
    return bundle, n


def build(out_dir, link=False, copy_weights=False, with_edf=False):
    snap = _snap()
    os.makedirs(out_dir, exist_ok=True)
    kdir = os.path.join(out_dir, "kernels")
    os.makedirs(kdir, exist_ok=True)

    print("=" * 78)
    print(f"BUILD host-loop artifact for qwen3-coder-next-fp8")
    print(f"  out  : {out_dir}")
    print(f"  snap : {snap}")
    print(f"  mode : {'symlink' if link else 'copy'}  copy_weights={copy_weights}")
    print("=" * 78)

    # ---- (2) NPU kernels --------------------------------------------------
    kernel_records = []
    for k in KERNELS:
        src = os.path.join(TK_DIR, k)
        assert os.path.exists(src), f"kernel missing: {src}"
        dst = os.path.join(kdir, k)
        _place(src, dst, link)
        kernel_records.append({
            "name": k,
            "path": f"kernels/{k}",
            "bytes": os.path.getsize(src),
            "sha256": _sha256(src),
            "engine": ("EinsumByDpe" if ("dpe" in k) else "EinsumByVe"),
        })
        print(f"  kernel  {k:26s} {os.path.getsize(src):>9d} B")

    # ---- (3) config / tokenizer / chat template from HF snapshot ----------
    snap_files = [
        "config.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json",
        "vocab.json", "merges.txt", "chat_template.jinja",
    ]
    placed_snap = []
    for f in snap_files:
        src = os.path.join(snap, f)
        if not os.path.exists(src):
            print(f"  (skip absent snapshot file: {f})")
            continue
        dst = os.path.join(out_dir, f)
        _place(src, dst, link)
        placed_snap.append(f)
        print(f"  snap    {f}")

    # load config for the manifest hf_configs
    cfg = json.load(open(os.path.join(out_dir, "config.json")))
    gen_cfg = {}
    gcp = os.path.join(out_dir, "generation_config.json")
    if os.path.exists(gcp):
        gen_cfg = json.load(open(gcp))

    # ---- (optional) weights: copy/symlink the FP8 shards + index ----------
    weights_ref = {
        "kind": "hf_cache_pointer",
        "hf_repo": "Qwen/Qwen3-Coder-Next-FP8",
        "snapshot": snap,
        "format": "fp8_e4m3_blockwise(128x128)",
        "note": ("75GB of FP8 weights are NOT bundled; the loader (qcn.loader."
                 "QCNWeights) mmaps + dequantizes them on demand from this HF "
                 "snapshot. Set QCN_SNAP to override."),
    }
    if copy_weights:
        wdir = os.path.join(out_dir, "weights")
        os.makedirs(wdir, exist_ok=True)
        shards = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
        idx = os.path.join(snap, "model.safetensors.index.json")
        if os.path.exists(idx):
            _place(idx, os.path.join(wdir, "model.safetensors.index.json"), link)
        for s in shards:
            _place(s, os.path.join(wdir, os.path.basename(s)), link)
        weights_ref = {
            "kind": "bundled",
            "dir": "weights",
            "format": "fp8_e4m3_blockwise(128x128)",
            "n_shards": len(shards),
        }
        print(f"  weights bundled {len(shards)} shards ({'symlink' if link else 'copy'})")

    # ---- (optional) EDF blobs: real a6 binary_bundle.zip ------------------
    # --emit-edf (re)generates binary_bundle.zip of the compute pieces that DO
    # compile to a6 EDF; the bundle is PARTIAL (DeltaNet recurrence stays host).
    binary_bundle_section = _binary_bundle_from_existing(out_dir)
    if with_edf:
        emit_edf(out_dir, recompile=True)
        binary_bundle_section = _binary_bundle_from_existing(out_dir)

    # ---- (1) artifact.json manifest --------------------------------------
    # Mirror the real furiosa-llm artifact.json shape where sensible, but be
    # honest that runtime == host-loop (no binary_bundle.zip of EDF pipelines).
    manifest = {
        "metadata": {
            "artifact_id": hashlib.sha256(
                (out_dir + str(time.time())).encode()).hexdigest()[:32],
            "name": "qwen3-coder-next-fp8-rngd",
            "timestamp": int(time.time()),
            "builder": "qcn/build_artifact.py",
            "furiosa_torch_required": True,
            "includes_composable_ir": False,
            "binary_bundle_kind": (
                "partial-edf" if binary_bundle_section else "none"),
            "binary_bundle_note": (
                ("A REAL binary_bundle.zip IS bundled: a6 CompiledGraph EDF blobs "
                 "(compiler.compile(...,'renegade-8pe',target_ir='edf'); same a6 "
                 "format as a furiosa-llm artifact, header a163456466 a6 "
                 "656e6f646573), each round-trip validated via "
                 "CompiledGraph.deserialize. They cover every Qwen3-Coder-Next "
                 "compute piece the 2026.2.0 compiler can lower (all Linear "
                 "projections, full-attention SDPA, MoE SwiGLU, RMSNorm/gated "
                 "RMSNorm, lm_head, embedding). The bundle is PARTIAL: "
                 "dn_conv1d_silu (O136), dn_gate (log1p), and "
                 "deltanet_recurrent_step (linear-attention recurrence -- "
                 "'conflict between concrete labels') do NOT compile to a6, so "
                 "that DeltaNet recurrent step + the cross-layer orchestration "
                 "stay a HOST LOOP (the recurrence can't compile AND the EDF "
                 "runtime has no recurrent-state slot). runtime stays 'host-loop' "
                 "even though real EDF compute blobs are now bundled.")
                if binary_bundle_section else
                "No binary_bundle.zip in this build (run with --emit-edf to "
                "generate one of the compute pieces that compile to a6 EDF)."),
            "build_blocked_reason": (
                "Full furiosa-llm `build` of model_type=qwen3_next remains "
                "unsupported in 2026.2.0: the DeltaNet (linear-attention) "
                "recurrent token mixer cannot be lowered to a single EDF pipeline "
                "and the EDF runtime carries no recurrent-state slot, so there is "
                "no whole-model binary_bundle.zip the furiosa-llm serve loader "
                "would run end-to-end. This artifact instead bundles a "
                "binary_bundle.zip of the compute subgraphs that DO compile to "
                "real a6 EDF and runs them under the working host loop."
                if binary_bundle_section else
                "furiosa-llm build of model_type=qwen3_next is unsupported in "
                "2026.2.0: the compiler kernelizer cannot lower the DeltaNet "
                "(linear-attention) token mixer via FX->kernelize, so no EDF "
                "binary_bundle.zip exists. This artifact ships the working "
                "host-loop runtime instead."),
        },
        # HONEST runtime marker -- this is the load-bearing distinction from a
        # real furiosa-llm artifact (which would have runtime == "edf"). Even
        # with real EDF compute blobs bundled, the DeltaNet recurrence stays on
        # the host, so runtime is still "host-loop".
        "runtime": "host-loop",
        **({"binary_bundle": binary_bundle_section} if binary_bundle_section else {}),
        "model": {
            "model_metadata": {
                "model_type": cfg.get("model_type", "qwen3_next"),
                "task": "generate",
                "runtime_kind": "host-loop",
                "hf_configs": cfg,            # full model config.json
                "model_weight_path": None,
                "trust_remote_code": False,
            },
            "weights": weights_ref,
            # host-loop "pipeline": the NPU kernels dispatched per layer, named
            # in dispatch order rather than as compiled EDF pipelines.
            "kernels": kernel_records,
            "runtime_module": {
                "entry_point": "qcn.model:QCNModel",
                "generate_method": "QCNModel.generate",
                "stream_method": "QCNModel.generate_stream",
                "openai_servers": ["qcn.serve:app", "qcn.serve_mc"],
                "sources": QCN_SOURCES,
                "kernel_dir": "kernels",
                "pythonpath": REPO,
                "env": {
                    "PYTHONPATH": REPO,
                    "RNGD_DEV": "rngd:2",
                    "QCN_DPE": "1 (use EinsumByDpe fast kernels; 0 for VE baseline)",
                    "QCN_SNAP": "(optional) override HF snapshot weights path",
                },
            },
        },
        "generator_config": {
            "generation_config": gen_cfg,
            "num_speculative_tokens": None,
        },
        "files": {
            "config": "config.json",
            "generation_config": ("generation_config.json"
                                  if "generation_config.json" in placed_snap else None),
            "tokenizer": [f for f in placed_snap if f.startswith("tokenizer")
                          or f in ("vocab.json", "merges.txt")],
            "chat_template": ("chat_template.jinja"
                              if "chat_template.jinja" in placed_snap else None),
            "kernels_dir": "kernels",
            "binary_bundle": ("binary_bundle.zip" if binary_bundle_section else None),
            "binary_bundle_manifest": ("binary_bundle_manifest.json"
                                       if binary_bundle_section else None),
            "readme": "README.md",
        },
        "version": {"major": 1, "minor": 0, "schema": "host-loop-artifact/v1"},
    }
    mpath = os.path.join(out_dir, "artifact.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  manifest {os.path.basename(mpath)} "
          f"({os.path.getsize(mpath)} B, {len(kernel_records)} kernels)")

    # ---- (4) README ------------------------------------------------------
    readme = _readme_text(out_dir, snap, kernel_records, cfg)
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(readme)
    print(f"  README.md written")

    print("-" * 78)
    print("ARTIFACT CONTENTS:")
    for name in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, name)
        if os.path.isdir(p):
            n = len(os.listdir(p))
            print(f"  {name}/   ({n} files)")
        else:
            print(f"  {name}   ({os.path.getsize(p)} B)")
    print("=" * 78)
    print(f"BUILD OK -> {out_dir}")
    return out_dir


def _readme_text(out_dir, snap, kernel_records, cfg):
    klist = "\n".join(f"  - kernels/{k['name']}  ({k['engine']})"
                      for k in kernel_records)
    return f"""# qwen3-coder-next-fp8-rngd  (host-loop artifact)

A self-contained, loadable artifact for **Qwen/Qwen3-Coder-Next-FP8** on the
Furiosa RNGD NPU.

## Why host-loop (not an EDF artifact)

`furiosa-llm build` of `model_type=qwen3_next` is **unsupported in SDK 2026.2.0**:
the compiler kernelizer cannot lower the DeltaNet (linear-attention) token mixer
through the FX -> kernelize path, so there is **no `binary_bundle.zip`** of
compiled EDF pipelines for this model.

Instead this artifact ships the working **host loop**: `qcn/model.py` runs the
48-layer forward on the host, and every token-mixer matmul (DeltaNet chunk scan,
full-attention SDPA, MoE SwiGLU) is dispatched to the RNGD NPU via the
hand-authored TacticKernel YAMLs in `kernels/` (loaded by
`furiosa.torch.TacticKernelModule`). The compute is genuinely on-device; only
the orchestration loop runs on the host. `artifact.json["runtime"] == "host-loop"`
makes this explicit.

## Contents

- `artifact.json` -- manifest (model_type, hf_configs, kernel list, entry-point,
  weights pointer). Mirrors the furiosa-llm artifact shape where sensible.
- `kernels/` -- the NPU TacticKernel YAMLs the host loop dispatches:
{klist}
- `config.json`, `generation_config.json` -- from the HF snapshot.
- `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`,
  `chat_template.jinja` -- the tokenizer + chat template.
- `weights` -- by default a *pointer* to the FP8 weights in the HF cache
  (75GB, not bundled). `qcn.loader.QCNWeights` mmaps + dequantizes per layer.

## Model

- model_type : `{cfg.get('model_type')}`
- layers     : {cfg.get('num_hidden_layers')}  (linear_attention + full_attention mix)
- hidden     : {cfg.get('hidden_size')}   vocab: {cfg.get('vocab_size')}
- MoE        : {cfg.get('num_experts')} experts, top-{cfg.get('num_experts_per_tok')}
- weights    : FP8 e4m3 blockwise (128x128) -> dequant on load

## Load + run

```bash
PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj \\
RNGD_DEV=rngd:2 QCN_DPE=1 \\
/home/jun/furiosa/bin/python qcn/run_artifact.py \\
    --artifact {out_dir} \\
    --prompt "def quicksort(arr):" --max-new 3
```

`run_artifact.py` reads `artifact.json`, points the host-loop modules'
kernel-base paths at this artifact's `kernels/`, resolves the tokenizer +
weights, instantiates `qcn.model.QCNModel`, and runs a greedy generation --
asserting the NPU CPU-fallback counters stayed 0 (proof the kernels ran
on-device).

> Device note: `RNGD_DEV=rngd:N` selects PE index N on npu0 here. If other
> processes (e.g. a serve) hold some of npu0's PEs you'll get an EBUSY at the
> first dispatch -- pick a FREE PE number instead (check with
> `ls -l /proc/<pid>/fd | grep npu0pe`; e.g. use `rngd:4`..`rngd:7` if PE0-3 are
> busy).

## Serve (OpenAI-compatible)

```bash
PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj \\
RNGD_DEV=rngd:2 QCN_DPE=1 \\
/home/jun/furiosa/bin/python -m qcn.serve   # or qcn.serve_mc
```

Built by `qcn/build_artifact.py`. HF snapshot: `{snap}`
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--link", action="store_true",
                    help="symlink HF snapshot files instead of copying")
    ap.add_argument("--copy-weights", action="store_true",
                    help="also bundle the 75GB FP8 safetensors shards")
    ap.add_argument("--emit-edf", action="store_true",
                    help="(re)compile the compute pieces to a6 EDF and (re)build "
                         "binary_bundle.zip + binary_bundle_manifest.json (slow, "
                         "uses the NPU via tk_kernels/compile_edf_blobs.py + "
                         "pack_edf_bundle.py). Without this flag an already-present "
                         "binary_bundle.zip is still referenced from artifact.json.")
    args = ap.parse_args()
    build(args.out, link=args.link, copy_weights=args.copy_weights,
          with_edf=args.emit_edf)


if __name__ == "__main__":
    main()
