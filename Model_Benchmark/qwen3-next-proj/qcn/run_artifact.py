#!/usr/bin/env python3
# =============================================================================
# run_artifact.py  --  LOAD + RUN the host-loop artifact built by
# build_artifact.py, proving it is a self-contained loadable+runnable unit.
#
# What "loading the artifact" means here:
#   1. read  <artifact>/artifact.json  (the manifest).
#   2. point the host-loop modules' kernel-base paths at <artifact>/kernels/ so
#      the NPU TacticKernel YAMLs are loaded FROM THE ARTIFACT (not the source
#      tree) -- this is what makes the artifact self-contained for compute.
#   3. resolve the weights (manifest weights pointer -> HF snapshot) and the
#      tokenizer/config (shipped in the artifact dir).
#   4. instantiate qcn.model.QCNModel and run a greedy generation.
#   5. assert the NPU CPU-fallback counters stayed 0  ==>  the kernels really
#      ran on the RNGD NPU (host loop only orchestrates).
#
# Usage:
#   PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj \
#   RNGD_DEV=rngd:2 QCN_DPE=1 \
#   /home/jun/furiosa/bin/python qcn/run_artifact.py \
#       --artifact /home/jun/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen3-coder-next-fp8-rngd \
#       --prompt "def quicksort(arr):" --max-new 3
# =============================================================================
import os
import sys
import json
import glob
import argparse

REPO = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj"
sys.path.insert(0, REPO)


def load_manifest(artifact_dir):
    mpath = os.path.join(artifact_dir, "artifact.json")
    assert os.path.exists(mpath), f"no artifact.json in {artifact_dir}"
    with open(mpath) as f:
        return json.load(f)


def _redirect_kernels(artifact_dir, kernel_records):
    """Repoint the host-loop modules' kernel-base constants at the artifact's
    kernels/ dir, and verify every kernel the manifest lists is present there.

    The modules read open(<BASE> + name) / open(os.path.join(TK, name)) at
    dispatch time, so overriding the module-level constant before any forward
    pass makes them load the ARTIFACT's kernel copies.  Modules touched:
      qcn.deltanet_layer._BASE   (shared by deltanet_layer_looped via import)
      qcn.attn_layer.TK
      qcn.moe.BASE
    """
    kdir = os.path.join(artifact_dir, "kernels")
    assert os.path.isdir(kdir), f"no kernels/ dir in artifact: {kdir}"
    # _BASE / BASE are used as string concatenation -> need a trailing slash.
    base_slash = kdir.rstrip("/") + "/"

    for rec in kernel_records:
        kp = os.path.join(artifact_dir, rec["path"])
        assert os.path.exists(kp), f"manifest kernel missing on disk: {kp}"

    import qcn.deltanet_layer as _dn
    import qcn.attn_layer as _attn
    import qcn.moe as _moe
    _dn._BASE = base_slash
    _attn.TK = kdir
    _moe.BASE = base_slash
    # deltanet_layer_looped imported _BASE by-value; rebind there too if present.
    try:
        import qcn.deltanet_layer_looped as _dnl
        if hasattr(_dnl, "_BASE"):
            _dnl._BASE = base_slash
    except Exception:
        pass
    return kdir


def _resolve_snapshot(manifest, artifact_dir):
    """Resolve the weights snapshot path from the manifest pointer.
    Honors $QCN_SNAP, then the manifest's recorded snapshot, then a fresh glob."""
    env = os.environ.get("QCN_SNAP")
    if env and os.path.isdir(env):
        return env
    wr = manifest.get("model", {}).get("weights", {})
    if wr.get("kind") == "bundled":
        wdir = os.path.join(artifact_dir, wr.get("dir", "weights"))
        if os.path.isdir(wdir):
            return wdir
    snap = wr.get("snapshot")
    if snap and os.path.isdir(snap):
        return snap
    g = sorted(glob.glob("/home/jun/.cache/huggingface/hub/"
                         "models--Qwen--Qwen3-Coder-Next-FP8/snapshots/*/"))
    assert g, "could not resolve weights snapshot"
    return g[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="artifact dir path")
    ap.add_argument("--prompt", default="def quicksort(arr):")
    ap.add_argument("--max-new", type=int, default=3)
    ap.add_argument("--chat", action="store_true")
    args = ap.parse_args()

    artifact_dir = os.path.abspath(args.artifact)
    dev = os.environ.get("RNGD_DEV", "rngd:2")

    print("=" * 78)
    print("LOAD + RUN host-loop artifact")
    print(f"  artifact : {artifact_dir}")
    print(f"  dev      : {dev}   QCN_DPE={os.environ.get('QCN_DPE', '0')}")
    print("=" * 78)

    # ---- 1. read manifest ------------------------------------------------
    manifest = load_manifest(artifact_dir)
    runtime = manifest.get("runtime")
    mm = manifest["model"]["model_metadata"]
    kernel_records = manifest["model"]["kernels"]
    entry = manifest["model"]["runtime_module"]["entry_point"]
    print(f"  manifest : model_type={mm['model_type']}  runtime={runtime!r}  "
          f"entry={entry}")
    print(f"             {len(kernel_records)} kernels listed")
    assert runtime == "host-loop", f"unexpected runtime: {runtime}"
    assert entry == "qcn.model:QCNModel", f"unexpected entry-point: {entry}"

    # ---- 2. point host-loop kernel paths at the artifact's kernels/ -------
    kdir = _redirect_kernels(artifact_dir, kernel_records)
    print(f"  kernels  : host-loop modules redirected -> {kdir}")

    # ---- 3. resolve weights + tokenizer ----------------------------------
    snap = _resolve_snapshot(manifest, artifact_dir)
    print(f"  weights  : {snap}")
    # use the artifact's own tokenizer/config dir for tokenization if complete,
    # else fall back to the snapshot (model.py reads tokenizer from the snap).
    art_has_tok = all(os.path.exists(os.path.join(artifact_dir, f))
                      for f in ("tokenizer.json", "tokenizer_config.json"))
    print(f"  tokenizer: artifact-local={art_has_tok}")

    # ---- 4. instantiate the model from the entry-point + run --------------
    # Import AFTER the kernel redirect so the model picks up the artifact paths.
    from qcn.model import QCNModel
    from qcn import deltanet_layer as _dn
    from qcn import attn_layer as _attn
    from qcn import moe as _moe

    # sanity: prove the model will load kernels from the ARTIFACT, not src tree.
    assert _dn._BASE.startswith(artifact_dir), _dn._BASE
    assert _attn.TK.startswith(artifact_dir), _attn.TK
    assert _moe.BASE.startswith(artifact_dir), _moe.BASE

    print("-" * 78)
    print(f"instantiating QCNModel(snap={os.path.basename(snap.rstrip('/'))}, dev={dev}) ...",
          flush=True)
    model = QCNModel(snap=snap, dev=dev)
    print(f"  loaded: {model.n_layers} layers, hidden={model.hidden}, "
          f"top_k={model.top_k}", flush=True)

    print("-" * 78)
    print(f"GENERATE prompt={args.prompt!r}  max_new={args.max_new}", flush=True)
    res = model.generate(args.prompt, max_new_tokens=args.max_new,
                         chat=args.chat, greedy=True, verbose=True)

    # ---- 5. NPU-exec proof ----------------------------------------------
    proof = {
        "deltanet_cpu_fallbacks": _dn._CALLS["n"],
        "attn_cpu_fallbacks":     _attn.CALLS["n"],
        "moe_cpu_fallbacks":      _moe.CALLS["n"],
        "deltanet_npu_stages":    len(model.dn.npu_stages),
        "attn_npu_stages":        len(_attn.NPU_STAGES),
        "moe_npu_stages":         len(_moe.NPU_STAGES),
    }
    all_npu = (proof["deltanet_cpu_fallbacks"] == 0 and
               proof["attn_cpu_fallbacks"] == 0 and
               proof["moe_cpu_fallbacks"] == 0)

    print("-" * 78)
    print("PROMPT    :", repr(res["prompt"]))
    print("GENERATED :", repr(res["generated_text"]))
    print("FULL TEXT :")
    print(res["full_text"])
    print("-" * 78)
    print("NPU-exec proof (CPU-fallback counters, MUST be 0):")
    print(f"  deltanet={proof['deltanet_cpu_fallbacks']}  "
          f"attn={proof['attn_cpu_fallbacks']}  moe={proof['moe_cpu_fallbacks']}")
    print(f"  NPU stages: deltanet={proof['deltanet_npu_stages']} "
          f"attn={proof['attn_npu_stages']} moe={proof['moe_npu_stages']}")
    print(f"  ALL_MIXERS_ON_NPU = {all_npu}")
    print("-" * 78)

    out = {
        "artifact": artifact_dir,
        "kernels_dir": kdir,
        "model_type": mm["model_type"],
        "runtime": runtime,
        "prompt": res["prompt"],
        "generated_text": res["generated_text"],
        "generated_ids": res["generated_ids"],
        "full_text": res["full_text"],
        "npu_proof": proof,
        "all_mixers_on_npu": all_npu,
        "dev": dev,
    }
    op = os.path.join(artifact_dir, "run_artifact_result.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"saved -> {op}")
    print("=" * 78)
    ok = all_npu and len(res["generated_ids"]) > 0
    print(f"ARTIFACT LOAD+RUN {'OK' if ok else 'FAILED'} "
          f"(generated {len(res['generated_ids'])} tokens, on_npu={all_npu})")
    print("=" * 78)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
