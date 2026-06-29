# Static call graph: `furiosa-llm build`

This is the consolidated, source-verified static call graph for the
`furiosa-llm build` (AOT compile) path, the command that *produces* the serve
artifact. It is the build-side companion to [`static-callgraph.md`](static-callgraph.md)
(the serve path) and was reconciled from eight per-subsystem extractions
(`cli`, `builder-init/validate/resolve`, `builder-build/save`,
`pipeline-orchestration`, `tracing`, `partition/parallelize`, `compile`,
`native-boundary`) — every `file:line` below was re-read against the installed
SDK **2026.2.0** (`~/furiosa/lib/python3.12/site-packages/furiosa_llm/`, rev
`9f92da0`) and several were cross-checked against a live py-spy/gdb capture of
the running build (see [`../03-synthesis/BUILD-CALLGRAPH-WALKTHROUGH.md`](../03-synthesis/BUILD-CALLGRAPH-WALKTHROUGH.md)).

All `file:line` citations are package-relative under `furiosa_llm/` unless noted.
Line numbers are **call-site** lines (the line that executes the call), matching
how the serve doc cites; a few definition (`def`) lines are noted where useful.

## The two boundaries this graph crosses

`build` crosses **two** opaque boundaries, where serve crosses only one:

1. **Driver → Ray worker** (process boundary). `build` starts a local Ray
   cluster and runs the heavy work — FX tracing and AOT compile — inside Ray
   *actor* processes, not the driver. The driver blocks in `ray.get(...)`. The
   `.remote()` call is the driver→worker boundary, exactly as a PyO3 call is the
   Python→native boundary in serve.
   - `LocalPipelineGenerationActor` (`@ray.remote(num_cpus=24)`,
     `new_pipeline_builder.py:1398`) — pipeline build = **trace + partition +
     parallelize**.
   - `TaskCompileActor` (`@ray.remote(num_cpus=32)`,
     `new_pipeline_builder.py:1079`) — **compile** (EDF blob generation).
2. **Python → native** (PyO3 boundary, inside the worker). The compiler itself
   is `furiosa.native_common.compiler`, a PyO3 module that
   `furiosa/native_common/compiler/__init__.py` registers from the native
   library **`furiosa/native_llm_common.cpython-312-x86_64-linux-gnu.so`**
   (143 MB, **fully stripped** — `nm` reports no symbols; 518 dynsym exports,
   only `PyInit_*` / `tch_*_stream_*` / `llg_*` / `perf_signal_handler`). So,
   exactly like serve's `native_runtime.so`, the build's native compile frames
   come out of gdb as `native_llm_common.so!0xADDR` and can only be named
   provisionally (see the `??` naming in
   [`../03-synthesis/full-callgraphs/gdb_build.native_names.md`](../03-synthesis/full-callgraphs/gdb_build.native_names.md)).
   The graph **pre**processing crosses a *different* native lib first —
   `furiosa_torch_ext.torch_ext.preprocess` (`converter.py:28`).

> Unlike serve, **`build` never touches the NPU.** Neither the driver nor any
> Ray worker opens a `/dev/rngd/*` node, and the kernel trace records zero
> doorbell/DMA/ioctl to `furiosa_rngd` (evidence:
> `../02-dynamic/logs/build_fds.txt`, `kernel_trace_build.log`). Compilation is a
> pure host (CPU) activity.

---

## (A) Driver — CLI → ArtifactBuilder → orchestrate (single process)

The driver does argument parsing, config/metadata resolution and validation,
then *submits* the per-bucket work to Ray and waits. py-spy caught the driver
parked exactly here: `… build_pipeline (new_pipeline_builder.py:1583) → ray.wait`.

### A.1 CLI bootstrap

1. `furiosa_llm.cli.main.main` builds the argparse registry and registers the
   `build` subcommand; `build` and `convert` share one handler in
   `cli/convert.py` (`cli/main.py:9`, dispatch at `cli/main.py:28`).
2. `cli.convert.convert` (`cli/convert.py:124`) parses the args into config
   objects (`ModelConfig`/`ParallelConfig`/`BucketConfig`/`CompilerConfig`/
   `ArtifactConfig`), reads `--num-pipeline-builder-workers` and
   `--num-compile-workers` (both default **1**, `cli/convert.py:15`+),
   constructs `ArtifactBuilder(...)` (`cli/convert.py:197`) and calls
   `ArtifactBuilder.build(...)` (`cli/convert.py:214`).

### A.2 ArtifactBuilder.__init__ — load / validate / resolve

`ArtifactBuilder.__init__` (`artifact/builder.py:116`) does steps [1]–[3] of the
build (the cheap, fast prologue — all in the driver):

- **load** HF config (`transformers.AutoConfig`).
- **validate** — `artifact/validator.py`: `validate_hf_config` (model_type
  supported? required fields present?), parallel-config (`SUPPORTED_TP_SIZES =
  {4, 8, 32}`), bucket and artifact validation. `validate_resolved_buckets`
  (`validator.py:188`) calls **NATIVE** `native_llm_common.compute_limits`.
- **resolve** — `artifact/resolver.py`: `resolve_model_metadata`, `max_model_len`,
  device mesh, and `ResolvedBuckets.resolve`, which consults `presets.find_preset`.
  `find_preset` matches by `(model_type, hidden_size, intermediate_size)` and
  calls **NATIVE** `compiler.approx_per_layer_params_b` (reached via
  `hf_utils.py:189`). Observed in the log:
  `Found bucket preset for model_type=qwen2, hidden_size=1536, intermediate_size=8960`
  → `Filtered bucket preset by max_model_len=2048`.

### A.3 ArtifactBuilder.build — orchestration

`ArtifactBuilder.build` (`artifact/builder.py:315`):

1. **NATIVE** `compiler.compiler_git_short_hash()` (`builder.py:357`, imported at
   `builder.py:350`) — stamps the artifact with the compiler revision.
2. `self._model_metadata.ensure_model_and_update_weight_hash()` (`builder.py:362`)
   — hashes the weights (log: `Calculated the hashsum in 2.8s … size=2955 MB`).
3. `self._build_model_artifact(...)` (`builder.py:363`) — the heavy phase (B+C).
4. `ArtifactBuilder.__save_artifacts(...)` (`builder.py:393`) — phase (D).

`_build_model_artifact` (`builder.py:172`) builds the `ModelCreationInfo`,
fetches the param file cache, builds the compiler-config generator, then calls
`next_gen.build_pipeline(...)` (`builder.py:268`) — the entry into Ray.

### A.4 build_pipeline — Ray fan-out + wait (the driver→worker boundary)

`new_pipeline_builder.build_pipeline` (`new_pipeline_builder.py:1474`):

- spawns the pipeline-build actor:
  `LocalPipelineGenerationActor.options(num_cpus=…).remote()`
  (`new_pipeline_builder.py:1557`).
- submits one trace task per bucket:
  `actors[i % n].build_for_bucket.remote(bucket)` (`new_pipeline_builder.py:1576`)
  — **driver→worker boundary [★]**.
- **blocks** on `ray.get(done_ref)` (`new_pipeline_builder.py:1586`) — this is
  where the driver's MainThread sits the whole tracing phase.
- then converts local→global (`LocalToGlobalPipelineConverter.convert`) and calls
  `get_compiled_pipeline(...)` (`new_pipeline_builder.py:1623`).

`get_compiled_pipeline` (`new_pipeline_builder.py:1199`) does the same dance for
compile: spawns `TaskCompileActor.options(num_cpus=…).remote()`
(`new_pipeline_builder.py:1281`), submits
`actor.compile_task.remote(stage_id, stage, task, symbol_values)`
(`new_pipeline_builder.py:1312`) — **driver→worker boundary [★]** — and blocks on
`ray.get(done_ref)` (`new_pipeline_builder.py:1320`). After results return it
calls **NATIVE** `CompiledGraph.deserialize(...)` (`new_pipeline_builder.py:1343`,
imported at `:1237`) to turn the returned blob bytes back into graph objects.

---

## (B) Worker 1 — `LocalPipelineGenerationActor`: trace + partition + parallelize

Runs inside the Ray actor process. The whole chain below was confirmed by py-spy
on the live actor (`../02-dynamic/logs/pyspy_build_trace.txt`).

```
build_for_bucket                         new_pipeline_builder.py:1451   (Ray actor method)
└─ build_local_pipeline                  new_pipeline_builder.py:580
   └─ build_partitioned_graphmodule      new_pipeline_builder.py:412 (def)
      ├─ get_aten_graph_with_metadata    new_pipeline_builder.py:455   → tracing (B.1)
      └─ parallelize_and_partition_graphmodule  new_pipeline_builder.py:486 → partition/parallelize (B.2)
```

### B.1 Tracing (FX / TorchDynamo / make_fx) → serialize graph

`trace.get_aten_graph_with_metadata` (`trace.py:1230` call →
`trace.py:1201` def) → `_get_aten_graph_with_metadata` (`trace.py:1180` →
`:1098` def). This checks the `gm_cache`, instantiates the model, and traces it:

- `block_slicer.enable_marker_op` (`trace.py:943`) — turns on the
  `furiosa.module_marker` ops that mark block boundaries for later slicing.
- `trace_model` (`trace.py:891`) → `torch._dynamo.export` (`trace.py:855`) and
  `torch.fx.experimental.proxy_tensor.make_fx` (`trace.py:828`). (py-spy caught
  the worker deep in `torch/_dynamo/{symbolic_convert,guards,variables}.py` and
  in fake-tensor propagation — the dynamo + make_fx engine room.)
- On success the traced GraphModule is **serialized to the cache**:
  `_get_aten_graph_with_metadata` → `cache_entry.save` (`trace.py:1087`) →
  `export.graphmodule.save_gm` (`export/graphmodule.py:426`) →
  `_convert_gm_into_dict_and_get_some_metadata` (`export/graphmodule.py:285`)
  via `torch._export.serde.serialize.GraphModuleSerializer.serialize_graph`.
  (py-spy caught the worker here too — `_dataclass_to_dict` over the exported
  graph.) `trace.py:29` imports `furiosa_torch_ext.torch_ext.eliminate_dead_code`.

### B.2 Partition + parallelize + rewrite

`parallelize_and_partition_graphmodule` (`new_pipeline_builder.py:367` call) does
two things:

- **Partition**: `PartitionComposer.partition_gm` →
  `KernelwisePartitioner.partition_gm` (`graph_partitioner.py:97`) using
  `block_slicer.get_kernelwise_sliced_color_bitmap_with_marker`
  (`graph_partitioner.py:65`); the marker context manager is
  `block_slicer.enable_marker_op` (`block_slicer.py:1258`).
- **Parallelize**: `parallelize_graphmodule` (`new_pipeline_builder.py:147`) →
  `ModelRewriter.rewrite` (`model_rewriter/api.py:101`) →
  `ShardingPropagator.propagate` (`model_rewriter/sharding_prop/sharding_propagator.py:241`)
  → `run_node` (`sharding_propagator.py:123`) over the FX graph. This is where
  the tp=4 sharding/sided-effects are applied. `model_rewriter/api.py` imports
  `furiosa_torch_ext.torch_ext.eliminate_dead_code`.

---

## (C) Worker 2 — `TaskCompileActor`: compile (EDF blobs)

Runs inside a second Ray actor process. `compile_task` (`new_pipeline_builder.py:1110`):

```
compile_task                              new_pipeline_builder.py:1110   (Ray actor method)
├─ CompilerConfigContext.load_config_with_layer_range   compiler_config.py (import :63)
│     ├─ NATIVE compiler.create_llm_compiler_config_with_layer_range   compiler_config.py:122
│     └─ NATIVE compiler.create_default_compiler_config (fallback)      compiler_config.py:142
├─ deserialize_gm                         (rebuild GraphModule from the cached blob)
├─ generate_graph_metadata               → NATIVE compiler.GraphMetadataBuilder   converter.py:1461 (import) / 1463 (use)
├─ GraphModuleConverter.compile_gm_and_get_preprocessed_gm_hash
│     ├─ furiosa_torch_ext.torch_ext.preprocess(gm, example_input)     converter.py:886   [native torch-ext]
│     └─ NATIVE compiler.compile(preprocessed, example_input, …)        converter.py:913   ★ THE AOT COMPILE → EDF blob
└─ NATIVE CompiledGraph.serialize()  (per graph)                        new_pipeline_builder.py:1192
```

`converter.py:876` imports `compile` / `CompileResult` from
`furiosa.native_common.compiler`. The output of `compile` is the precommand/EDF
binary that serve later loads; nothing here runs on the NPU — it is the host
compiler turning the partitioned ATen graph into RNGD machine code.

---

## (D) Save artifacts

`ArtifactBuilder.__save_artifacts` (`builder.py:481`) ← called at `builder.py:393`:

- `__preprocess_for_pipeline_save` (`builder.py:403`) → **NATIVE**
  `CompiledGraph.serialize()` per blob (`builder.py:455`, and the composable
  variant at `:463`).
- `get_tokenizer` + model config are saved; everything is bundled into
  `binary_bundle.zip` + `artifact.json` in the output dir.

---

## Python → native boundary crossings (build path)

All native targets resolve into **`native_llm_common.cpython-312*.so`** (stripped)
via the `furiosa.native_common.compiler` PyO3 module, except `preprocess` /
`eliminate_dead_code` / `SIDE_EFFECT_OPS` which are in `furiosa_torch_ext`'s own
extension.

| Python symbol | Native target | Site |
|---|---|---|
| `validator.validate_resolved_buckets` | `native_llm_common.compute_limits` | `artifact/validator.py:188` |
| `presets.find_preset` (via `hf_utils`) | `compiler.approx_per_layer_params_b` | `artifact/hf_utils.py:189` |
| `ArtifactBuilder.build` | `compiler.compiler_git_short_hash` | `artifact/builder.py:357` |
| `CompilerConfigContext.load_config_with_layer_range` | `compiler.create_llm_compiler_config_with_layer_range` | `parallelize/compiler_config.py:122` |
| `CompilerConfigContext.load_config_with_layer_range` | `compiler.create_default_compiler_config` | `parallelize/compiler_config.py:142` |
| `generate_graph_metadata` | `compiler.GraphMetadataBuilder` | `parallelize/pipeline/builder/converter.py:1461` |
| `GraphModuleConverter.compile_gm_and_get_preprocessed_gm_hash` | `furiosa_torch_ext.torch_ext.preprocess` | `parallelize/pipeline/builder/converter.py:886` |
| `GraphModuleConverter.compile_gm_and_get_preprocessed_gm_hash` | **`compiler.compile`** (AOT) ★ | `parallelize/pipeline/builder/converter.py:913` |
| `TaskCompileActor.compile_task` | `compiler.CompiledGraph.serialize` | `parallelize/new_pipeline_builder.py:1192` |
| `get_compiled_pipeline` | `compiler.CompiledGraph.deserialize` | `parallelize/new_pipeline_builder.py:1343` |
| `ArtifactBuilder.__preprocess_for_pipeline_save` | `compiler.CompiledGraph.serialize` | `artifact/builder.py:455` |
| `trace` / `model_rewriter` | `furiosa_torch_ext.torch_ext.eliminate_dead_code` | `parallelize/trace.py:29`, `model_rewriter/api.py:5` |

## Driver → Ray-worker boundary crossings (build path) — new vs serve

| Driver symbol | Ray target (separate process) | Site |
|---|---|---|
| `build_pipeline` | `LocalPipelineGenerationActor.build_for_bucket.remote` | `new_pipeline_builder.py:1576` |
| `build_pipeline` (wait) | `ray.get(done_ref)` | `new_pipeline_builder.py:1586` |
| `get_compiled_pipeline` | `TaskCompileActor.compile_task.remote` | `new_pipeline_builder.py:1312` |
| `get_compiled_pipeline` (wait) | `ray.get(done_ref)` | `new_pipeline_builder.py:1320` |

---

## Open questions / verified dynamically

1. **Native compile is opaque per-function** — `native_llm_common.so` is
   stripped, so `compiler.compile` / `create_*_compiler_config` /
   `GraphMetadataBuilder` / `CompiledGraph.*` are reached purely via PyO3 and
   show up in gdb only as `native_llm_common.so!0xADDR`. Named provisionally from
   the compile-phase gdb snapshot of `TaskCompileActor`
   (`../03-synthesis/full-callgraphs/gdb_build.native_names.md`).
2. **Worker count vs Ray** — even with the default `--num-*-workers 1`, build
   always starts a local Ray cluster and runs the work in **one** actor per
   phase; the actor is a separate process (confirmed: `ray::LocalPipelineGenerationActor`
   / `ray::TaskCompileActor` in the process tree). Higher worker counts add more
   actors (more buckets/stages in flight), not in-process parallelism.
3. **No NPU** — confirmed by open-fd scan and the empty kernel trace; the only
   `furiosa-smi`/device access on the box during the window is the resident
   monitor, not the build.
