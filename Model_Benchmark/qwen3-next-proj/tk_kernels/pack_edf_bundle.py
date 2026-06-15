"""Package the compiled a6 EDF blobs into a real binary_bundle.zip + manifest.

Mirrors the REAL furiosa-llm binary_bundle.zip layout (measured from
qwen3-coder-30b-a3b-inst-tp8-65k-tc/binary_bundle.zip):
  - a FLAT zip of <32-hex-hash>.edf files at the zip root (no subdirs)
  - ZIP_STORED (compress_type == 0)  <- the real bundle stores, never deflates
  - each .edf == furiosa.native_common.compiler CompiledGraph.serialize() bytes
    whose header begins ...a163456466 a6 656e6f646573 (a6 = 6-key map WITH the
    top-level 'binaries' field, the artifact CompiledGraph format).

This script:
  (1) builds binary_bundle.zip from _edf_blobs/<hash>.edf
  (2) round-trip VALIDATES: opens the zip, reads each .edf, and runs
      CompiledGraph.deserialize() on it -- the test that distinguishes a real
      a6 EDF bundle the loader accepts from junk -- reporting how many of N OK.
  (3) writes binary_bundle_manifest.json: hash -> {piece, input_shapes, bytes,
      dpe_or_ve}.

Run:
  PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj RNGD_DEV=rngd:4 \
    /home/jun/furiosa/bin/python tk_kernels/pack_edf_bundle.py
"""
import os
import sys
import json
import glob
import zipfile

import torch  # noqa: F401  (import torch first per SDK requirement)
from furiosa.native_common.compiler import CompiledGraph

ARTIFACT = ("/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/"
            "qwen3-coder-next-fp8-rngd")
BLOB_DIR = os.path.join(ARTIFACT, "_edf_blobs")
BUNDLE = os.path.join(ARTIFACT, "binary_bundle.zip")
MANIFEST = os.path.join(ARTIFACT, "binary_bundle_manifest.json")
MASTER = os.path.join(BLOB_DIR, "_MASTER_summary.json")

A6_NEEDLE = bytes.fromhex("a163456466a6656e6f646573")  # a1 63 Edf a6 65 nodes

# Which compute engine each piece predominantly lands on. The matmul/contraction
# pieces (Linear projections, attention scores, expert SwiGLU, embedding gather,
# lm_head) are matrix-engine (DPE) work; the pure elementwise/reduction norms
# (RMSNorm, gated RMSNorm) are vector-engine (VE) work. This is a per-piece
# classification by op nature (the a6 CompiledGraph schedules ops across both
# engines internally; this records the dominant engine).
DPE_OR_VE = {
    "lin.q_proj": "dpe", "lin.kv_proj": "dpe", "lin.o_proj": "dpe",
    "lin.in_proj_qkvz": "dpe", "lin.in_proj_ba": "dpe", "lin.dn_out_proj": "dpe",
    "lin.moe_gate": "dpe", "lin.moe_up": "dpe", "lin.moe_down": "dpe",
    "lin.router_gate": "dpe", "lin.shared_gate": "dpe",
    "full_attn_sdpa": "dpe", "moe_expert_swiglu": "dpe",
    "lin.lm_head_repr": "dpe", "embedding_repr": "dpe",
    "rmsnorm": "ve", "gated_rmsnorm": "ve",
    # DeltaNet recurrent step split (2026-06-15): contraction/outer = matrix engine,
    # decay/delta/add = elementwise (vector engine); conv-shift + gate = vector.
    "dn_recur_contract": "dpe", "dn_recur_outer": "dpe",
    "dn_recur_decay": "ve", "dn_recur_delta": "ve", "dn_recur_add": "ve",
    "dn_conv1d_shift": "ve", "dn_gate_beta": "ve", "dn_gate_g": "ve",
}


def hdr_is_a6(blob: bytes) -> bool:
    return A6_NEEDLE in blob[:64]


def load_master():
    with open(MASTER) as f:
        m = json.load(f)
    # map edf_hash -> record (only the ok ones with a hash)
    by_hash = {}
    for r in m.get("results", []):
        if r.get("edf_hash"):
            by_hash[r["edf_hash"]] = r
    return m, by_hash


def build_zip():
    """Build binary_bundle.zip = flat ZIP_STORED of <hash>.edf at root."""
    edfs = sorted(glob.glob(os.path.join(BLOB_DIR, "*.edf")))
    assert edfs, f"no .edf blobs in {BLOB_DIR}"
    if os.path.exists(BUNDLE):
        os.remove(BUNDLE)
    # ZIP_STORED to match the real 30B bundle (compress_type==0, no deflate).
    with zipfile.ZipFile(BUNDLE, "w", compression=zipfile.ZIP_STORED) as z:
        for p in edfs:
            arcname = os.path.basename(p)  # flat: <hash>.edf at root, no subdir
            z.write(p, arcname=arcname)
    return edfs


def validate_roundtrip():
    """Open the zip, read each .edf, deserialize via CompiledGraph.deserialize.

    Returns (n_total, n_ok, per_entry list)."""
    results = []
    n_ok = 0
    with zipfile.ZipFile(BUNDLE, "r") as z:
        names = [n for n in z.namelist() if n.endswith(".edf")]
        for n in names:
            blob = z.read(n)
            tag = n[:-4]  # the <hash> (== hash_val the real loader passes as tag)
            entry = {
                "name": n, "bytes": len(blob),
                "a6_header": hdr_is_a6(blob),
                "deserialize_ok": False, "is_edf": None, "tag_used": None,
                "error": None,
            }
            # The real loader calls CompiledGraph.deserialize(bytes, tag).
            #   new_pipeline_builder.py:1343 -> tag = hash_val (the blob hash)
            #   pipeline/next_gen.py:324     -> tag = ""
            # Try the hash tag first (what builder.py uses), then "" fallback.
            last_err = None
            for tag_try in (tag, ""):
                try:
                    cg = CompiledGraph.deserialize(blob, tag_try)
                    entry["deserialize_ok"] = True
                    entry["tag_used"] = tag_try
                    try:
                        entry["is_edf"] = bool(cg.is_edf())
                    except Exception:
                        entry["is_edf"] = None
                    n_ok += 1
                    break
                except Exception as e:
                    last_err = (type(e).__name__ + ": " + str(e))[:300]
            if not entry["deserialize_ok"]:
                entry["error"] = last_err
            results.append(entry)
    return len(results), n_ok, results


def write_manifest(edfs, by_hash, validation):
    val_by_name = {v["name"]: v for v in validation}
    blobs = {}
    for p in sorted(edfs):
        name = os.path.basename(p)
        h = name[:-4]  # strip .edf
        rec = by_hash.get(h, {})
        v = val_by_name.get(name, {})
        blobs[h] = {
            "file": name,
            "piece": rec.get("piece"),
            "input_shapes": rec.get("input_shapes"),
            "in_dim": rec.get("in_dim"),
            "out_dim": rec.get("out_dim"),
            "note": rec.get("note"),
            "bytes": os.path.getsize(p),
            "dpe_or_ve": DPE_OR_VE.get(rec.get("piece"), "unknown"),
            "deserialize_ok": v.get("deserialize_ok"),
            "is_edf": v.get("is_edf"),
        }
    manifest = {
        "bundle": "binary_bundle.zip",
        "format": ("flat ZIP_STORED of <md5-hash>.edf at zip root; each .edf == "
                   "furiosa.native_common.compiler CompiledGraph.serialize() a6 "
                   "bytes (header a163456466 a6 656e6f646573), same format as a "
                   "real furiosa-llm artifact binary_bundle.zip"),
        "compression": "ZIP_STORED",
        "producer": ("tk_kernels/compile_edf_blobs.py (compiler.compile(mod,args,"
                     "'renegade-8pe',target_ir='edf')) -> tk_kernels/pack_edf_bundle.py"),
        "n_blobs": len(blobs),
        "scope": (
            "REAL a6 EDF blobs for every Qwen3-Coder-Next compute piece: all "
            "Linear projections (q/k/v/o, in_proj_qkvz/ba, dn_out_proj, MoE "
            "gate/up/down, router, shared, lm_head, embedding), full-attention "
            "SDPA, MoE expert SwiGLU, RMSNorm, gated RMSNorm, AND (2026-06-15) the "
            "DeltaNet linear-attention recurrence -- previously uncompilable, now "
            "added by SPLITTING each blocked piece into single-op a6 subgraphs: "
            "recurrent step -> dn_recur_decay/contract/delta/outer/add (faithful to "
            "fp64, fp32 rel ~2.9e-7); conv1d -> dn_conv1d_shift (host-pad + "
            "shift-mul-add + SiLU, avoids O136 Conv1d & constant_pad_nd); gate -> "
            "dn_gate_beta (sigmoid) + dn_gate_g (-exp(A_log)*log(1+exp), softplus "
            "without log1p). ROOT CAUSE the split fixes: the a6 compiler rejects a "
            "single graph that mixes multiple contraction patterns ('conflict "
            "between concrete labels') or holds multiple independent output "
            "subgraphs ('multiple internal subgraphs'); one-op-per-graph compiles. "
            "NOTE: the compute is now fully a6-bundled, but furiosa-llm serve still "
            "cannot DRIVE the recurrent decode -- it has no cross-step recurrent-"
            "state pool (paged-KV is append-only K/V, the recurrence is read-modify-"
            "write). So this remains a host-loop artifact; running these split blobs "
            "as a chained serve pipeline needs the vendor runtime (2026.3+)."),
        "blobs": blobs,
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def main():
    print("=" * 78)
    print("PACK + VALIDATE binary_bundle.zip for qwen3-coder-next-fp8-rngd")
    print("=" * 78)
    _, by_hash = load_master()
    edfs = build_zip()
    size = os.path.getsize(BUNDLE)
    print(f"  built {BUNDLE}")
    print(f"  {len(edfs)} .edf files, zip size {size} B ({size/1e6:.1f} MB)")

    n, n_ok, validation = validate_roundtrip()
    print(f"\n  ROUND-TRIP VALIDATION (CompiledGraph.deserialize):")
    for v in validation:
        flag = "OK  " if v["deserialize_ok"] else "FAIL"
        print(f"    {flag} {v['name']}  {v['bytes']:>10} B  a6={v['a6_header']}  "
              f"is_edf={v['is_edf']}  tag={v['tag_used']!r}"
              + (f"  ERR={v['error']}" if v['error'] else ""))
    print(f"\n  {n_ok}/{n} blobs deserialize as valid a6 CompiledGraph")

    manifest = write_manifest(edfs, by_hash, validation)
    print(f"\n  manifest -> {MANIFEST} ({len(manifest['blobs'])} blobs)")
    print("=" * 78)
    if n_ok != n:
        print("WARNING: not all blobs deserialized; bundle is suspect.")
        sys.exit(1)
    print(f"OK: binary_bundle.zip ({len(edfs)} blobs, {size/1e6:.1f} MB), "
          f"all {n_ok} deserialize as real a6 CompiledGraph.")


if __name__ == "__main__":
    main()
