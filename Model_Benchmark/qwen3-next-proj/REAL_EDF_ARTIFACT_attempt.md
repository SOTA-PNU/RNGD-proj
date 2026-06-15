# REAL furiosa-llm EDF artifact for qwen3_next — rigorous attempt + verdict

Date: 2026-06-12. SDK: `/home/jun/furiosa/lib/python3.12/site-packages/furiosa_llm/`
(2026.2.0). All claims below are backed by file:line or a probe I ran.

## TL;DR verdict

**PARTIAL → effectively NO for a self-contained, compiler-honest artifact.**

- The full-attention layers, MoE, embedding, gate/sigmoid/exp/log standalone
  elementwise — these **do** compile to real serializable EDF.
- The **DeltaNet recurrent body** (`state = state + k⊗delta; out = (state*q).sum`)
  **does not compile**: it triggers a hard Rust **panic in the global compiler**
  (`global-compiler/src/lib.rs:100`), surfaced as `furiosa.UnsupportedOpError:
  failed to compile the graph`. `allow_external_operators=True` /
  `allow_unlowered_operators=True` change **nothing** (measured, identical errors).
- There is **no ExternalOperator injection API** in furiosa-llm. The flag
  `allow_external_operators` exists only on the low-level compiler `Config`
  (`furiosa/native_torch/compiler.pyi:17,41`) and does not accept a pre-compiled
  EDF blob, a path, or a registered custom op as a DeltaNet substitute.
- A hand-authored `furiosa.torch` EDF (`ir.Edf`) is **not** byte-compatible with the
  artifact's `CompiledGraph` EDF (one extra top-level CBOR field, `binaries`), so it
  cannot be dropped into `binary_bundle.zip` as-is.

So a "real" artifact that faithfully runs all 48 layers on-NPU through the standard
build is not achievable in 2026.2.0. What *is* achievable is a hybrid bundle whose
12 full-attn/MoE layers are genuine CompiledGraphs — but the 36 DeltaNet layers have
no producible EDF, so the bundle would still be incomplete and serve would have no
valid blob for those tasks. The working solution remains the host loop (`qcn/`).

---

## 1. How `binary_bundle.zip` is produced (build path)

`furiosa_llm/artifact/builder.py`:
- `ArtifactBuilder.build()` → `_build_model_artifact()` (builder.py:172) only supports
  `_use_composable_kernel=True` (KERNELWISE); the block-wise path raises
  `ValueError("Block-wise artifact is not supported anymore.")` (builder.py:313,391).
- Pipelines are compiled by `next_gen.build_pipeline(..., "edf", ...)` (builder.py:268).
- `__preprocess_for_pipeline_save()` (builder.py:402) walks `pipeline.blobs`; for
  `TaskKind.EDF` it asserts the blob is a `CompiledGraph` and writes
  `data = [blob.serialize()]` to `<id>.edf` inside `binary_bundle.zip` (builder.py:452-455).
  → **The artifact stores `CompiledGraph.serialize()` bytes, one per task id.**

The actual compile is `furiosa.native_common.compiler.compile(...)`:
- `furiosa_llm/parallelize/pipeline/builder/converter.py:863` →
  `compile_gm_and_get_preprocessed_gm_hash` → `compile(..., target_ir="edf")`
  (converter.py:913), returning `CompileResult.graphs: List[CompiledGraph]`
  (`furiosa/native_common/compiler/__init__.pyi:21-31`).
- `new_pipeline_builder.py:1192` serializes each `CompiledGraph`; deserialized back
  at `new_pipeline_builder.py:1343` via `CompiledGraph.deserialize(...)`.

## 2. The ExternalOperator path — what it actually is

- `grep ExternalOperator furiosa_llm/` → **0 hits.** There is no external-operator or
  pre-compiled-EDF injection API at the furiosa-llm level.
- `allow_external_operators` and `allow_unlowered_operators` are **compiler `Config`
  booleans only** (`furiosa/native_torch/compiler.pyi:15-17, 38-41`). They are plain
  flags — they do **not** take a blob/path/op. Their effect (measured) on DeltaNet: none.
- They are reachable from a build via `compiler_config_overrides`
  (`build_with_override.py` → merged at `compiler_config.py:176-177`). The prior run
  `logs/build_override.log` used `{'allow_unlowered_operators': True,
  'allow_external_operators': True}` and **did not** reach `BUILD OK` (it hangs/dies in
  "Compilation Progress 0/9", no success line).

## 3. Can our hand-authored DeltaNet TK kernels become standalone EDF blobs?

Yes for the *easy* pieces, no for the recurrence. Probe:
`tk_kernels/edf_backend_probe.py` (uses `CompileModule.from_module`, i.e. the same
`furiosa.torch.export.PASSES` + `compiler.compile` that `torch.compile(backend=ft.backend)`
runs — `furiosa/torch/custom_ops/edf.py:464-481`).

Measured (`config = default` and `allow_external+unlowered`, **identical** outcomes):

| sub-module | result | ir.Edf serialize |
|---|---|---|
| matmul (control) | COMPILE OK | 28227 B, npu_node=True |
| Gate-only (sigmoid/exp/log standalone) | **COMPILE OK** | 44369 B, npu_node=True |
| DeltaStep (one recurrent step) | **FAIL** | — |

DeltaStep error (full trace captured):
```
Panic in compiler thread: Any { .. }
  global-compiler/src/lib.rs:100:26
→ furiosa.UnsupportedOpError: failed to compile the graph
  (furiosa/torch/custom_ops/edf.py:479)
```
The culprit is the 4-D outer-product state accumulate
`state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)` + `(state*q).sum(-2)` — the compiler
**panics** (does not gracefully reject) on this pattern. `allow_external_operators`
does not route around the panic.

Note: the SDK's own `furiosa/models/language/architecture/qwen3_next.py:460-466`
documents the same wall ("커널라이저가 standalone elementwise를 커널로 내보내지
못해 … 'ONNN is not an operator' 무한 반복"). My probe **refines** that: standalone
gate elementwise *does* compile through the proper torch backend; it is specifically
the **recurrent outer-product state update** that kills the global compiler.

## 4. Could DeltaNet layers be EDF blobs injected into the artifact?

Two hard blockers:

**(a) No producible blob.** The DeltaNet recurrent body cannot be compiled to *any*
EDF (panic above), so there is nothing to inject. Even a chunked/parallel scan form
would still need the same outer-product accumulate the compiler panics on.

**(b) Format mismatch even if we had a blob.** A `furiosa.torch` `ir.Edf` is NOT the
artifact `CompiledGraph`. Measured CBOR headers (probe `/tmp/cbor_cmp2.py`):
- torch `ir.Edf.serialize()` → `…a163456466 a5 656e…` → **a5** = 5-key CBOR map.
- artifact `<id>.edf` (real 30B bundle) → `…a163456466 a6 656e…` → **a6** = 6-key map.
- `CompiledGraph.deserialize(ir.Edf_blob)` → `RuntimeError: missing field 'binaries'`
  (`furiosa-llm-common/src/compiler/compile.rs:96`).

The two formats share the `cEdf` envelope (`nodes`/`Copy`/`Npu`/`task_binaries_map`/
`binary`) and differ by exactly **one top-level field** (`binaries`). So they are
*structurally close* but **not interchangeable** — the artifact loader
(`furiosa_llm/parallelize/pipeline/next_gen.py:324`, `CompiledGraph.deserialize`)
rejects an `ir.Edf` blob.

**Mechanically, the a6 producer is callable from Python:**
`furiosa.native_common.compiler.compile(MyModule(), args, target_ir="edf",
target_npu="renegade-8pe")` returns a real `CompileResult` whose `graphs[0]` is a
`CompiledGraph` with the correct `…a6 656e…` header (75450 B, `is_edf()=True`) —
measured (`/tmp/cg_make.py`). So a **hybrid bundle is buildable for any sub-graph the
compiler accepts** (full-attn, MoE, gate). It is *only* the DeltaNet recurrence that
has no valid CompiledGraph → the bundle is unavoidably incomplete.

## 5. The runtime recurrent-state problem (independent, second wall)

Even setting compilation aside, the artifact/serve runtime has no place to thread the
per-step recurrent state across decode calls: the paged-KV contract
(`CausalModelForwardInputs.kv_caches: List[Tuple[K,V]]`, identical shape per layer)
can't represent a `(nv, dk, dv)` DeltaNet state, and assigning an unused (K,V) slot to a
DeltaNet layer makes it a dead node → `graph_partitioner.py:130 IndexError`
(documented in `qwen3_next.py:497-501`; matches the [Qwen3-Next blocker] memory).

---

## What WOULD make it buildable (concrete asks for the vendor)

1. Compiler fix: lower (or at least not panic on) the DeltaNet recurrent outer-product
   state update — i.e. support `einsum`-style `state += k ⊗ delta` and
   `out = einsum(state, q)`. This is the single op pattern that panics
   `global-compiler/src/lib.rs:100`.
2. A real external-operator / pre-compiled-EDF injection API in furiosa-llm that lets a
   task's blob be supplied as an a6 `CompiledGraph` we author (today the `.edf` writer
   only ever takes compiler output; `binary_bundle` has no "bring-your-own-blob" hook).
3. A runtime recurrent-state slot in the forward-inputs contract (or composable-kernel
   metadata) so DeltaNet state persists across decode steps.

Until then: keep the **host loop** (`qcn/model.py` 48-layer forward + hand-authored TK
kernels via `furiosa.torch` for the matmul/gate/norm pieces, host-side recurrence).
The pieces that *do* compile (matmul EinsumByDpe, gate, l2norm, gnorm, conv1d) can each
be a real EDF used inside the host loop — that is already what the project does.

## Probes/artifacts produced by this investigation

- `tk_kernels/edf_extern_probe.py` — raw `compiler.compile` (no preprocessing) probe.
- `tk_kernels/edf_backend_probe.py` — proper `CompileModule.from_module` probe (the
  one with the real results table above).
- `/tmp/cbor_cmp2.py`, `/tmp/cg_make.py`, `/tmp/xdeser.py` — format/cross-deser probes
  (a5 vs a6, `missing field 'binaries'`, a6 producer reachable).
