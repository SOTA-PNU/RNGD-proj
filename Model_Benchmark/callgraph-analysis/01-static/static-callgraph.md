# Static call graph: `furiosa-llm serve`

This document is a consolidated, deduplicated static call graph for the
`furiosa-llm serve` path, reconciled from five per-subsystem extractions
(`cli-bootstrap`, `server-handlers`, `engine-api`, `artifact-load`,
`build-path`, `native-boundary`).

All `file:line` citations use the package-relative path under
`site-packages/furiosa_llm/` unless noted. The one important convention wrinkle:
the `LLM.__init__` dispatch and `_init_from_artifact` / `_init_from_v3_engine`
bodies physically live in **`api.py`** (e.g. `api.py:216`, `api.py:383`), even
though some upstream extractions cited them as `server/app.py` per a
package-relative convention. Where the two disagree, the `api.py` line is the
real source location and is used here.

The native boundary is almost entirely one PyO3 shared object:
`furiosa.native_runtime` (`furiosa/native_runtime.cpython-312-x86_64-linux-gnu.so`,
a ~164MB statically-linked Rust crate embedding npu-tools / npu-compiler /
scheduler and talking to the `rngd` NPU driver via `ioctl`). A smaller helper
`furiosa.native_llm_common` handles artifact metadata, and the **build path**
crosses into `furiosa.native_common.compiler` (not used at serve time).

---

## (A) Serve startup + artifact load

This is the serve-time "build": load the prebuilt artifact, validate it,
construct the native engine, device-map, and warm up.

### A.1 CLI bootstrap -> uvicorn

1. `furiosa_llm.cli.main.main` builds the argparse registry
   (`argparse.ArgumentParser.add_subparsers`, `cli/main.py:11`) and registers the
   serve subcommand via `furiosa_llm.cli.serve.add_serve_args` (`cli/main.py:17`).
2. `add_serve_args` wires the dispatch target with
   `set_defaults(dispatch_function=serve)` (`cli/serve.py:381`), and reads the
   registered tool/reasoning parser keys
   (`ToolParserManager.tool_parsers` `cli/serve.py:87`,
   `ReasoningParserManager.reasoning_parsers` `cli/serve.py:97`) to build
   `--tool-call-parser` / `--reasoning-parser` metavars.
3. `main` parses args (`argparse.ArgumentParser.parse_args`, `cli/main.py:25`) and
   dispatches `args.dispatch_function(args)` -> `furiosa_llm.cli.serve.serve`
   (`cli/main.py:28`; statically bound through the `set_defaults` above).
4. `serve` -> `furiosa_llm.server.app.run_server` (`cli/serve.py:385`).
5. `run_server` -> `furiosa_llm.server.app.init_app` (`server/app.py:536`), then
   conditionally wraps middleware
   (`RequestLoggerMiddleware` when `--enable-payload-logging`, `server/app.py:539`;
   `AuthenticationMiddleware` when `--api-key`, `server/app.py:547`) and finally
   calls the blocking `uvicorn.run(app)` (`server/app.py:549`).

### A.2 init_app: LLM load + handler wiring

`init_app` is the bootstrap core (`server/app.py:389`+):

- `furiosa_llm.server.models.load_llm_from_args` (`server/app.py:389`)
  - `SchedulerConfig.load_from_args` (`server/models.py:40`) ->
    `PrefixCacheConfig.load_from_args` (`metadata/config_types.py:122`)
  - `furiosa_llm.api.LLM.__init__` (`server/models.py:42`) — the LLM construction.
- `ModelsResponse.from_llm` (`server/app.py:401`) -> `Model.from_llm`
  (`server/protocol.py:995`) builds the cached `/v1/models` payload.
- GENERATION_TASKS only: `llm.tokenizer.get_chat_template` (`server/app.py:407`),
  `build_model_config_adapter` (`server/app.py:412`) which builds
  `MultiModalConfig` (`server/app.py:101`) and `ModelConfigAdapter`
  (`server/app.py:106`).
- Serving-handler instantiation (task-conditional):
  - `OpenAIServingCompletion` (`server/app.py:413`),
    `OpenAIServingChat` (`server/app.py:414`),
    `OpenAIServingResponses` (`server/app.py:426`) — GENERATION_TASKS.
  - `OpenAIServingEmbedding` (`server/app.py:438`) — EMBED.
  - `ServingScores` (`server/app.py:440`) — SCORE.
  - `OpenAIServingTokenization` (`server/app.py:446`) — always.
- App assembly: `fastapi.FastAPI(lifespan=lifespan)` (`server/app.py:467`),
  `app.include_router(router)` (`server/app.py:468`) which registers every
  module-level `@router.get`/`@router.post` handler, conditional
  `Log4xx5xxMiddleware` (`server/app.py:470`), `CORSMiddleware`
  (`server/app.py:471`), and the Prometheus mount via
  `get_metrics_mount` (`server/app.py:479`) -> `starlette.routing.Mount` +
  `_make_prometheus_endpoint` (`server/metrics.py:49`).

### A.3 lifespan startup/shutdown

The `@asynccontextmanager lifespan` (defined in `init_app`):

- startup: `install_metrics_logging_thread` (`server/app.py:457`; skipped under
  `--disable-log-stats`) -> spawns a daemon `threading.Thread`
  (`server/metrics.py:113`); and `prewarm_server` (`server/app.py:460`).
- shutdown: `llm.engine.shutdown()` -> **native**
  `NativeLLMEngine.shutdown` (`server/app.py:465`).

`prewarm_server` warms the async backend (`anyio.sleep` `server/app.py:492`),
tokenizer (`llm.tokenizer.encode` `server/app.py:495`), the native pooling
binding (`NativePoolingOutput` `server/app.py:501`, POOLING_TASKS only), and the
chat template (`resolve_hf_chat_template` `server/app.py:505`,
`resolve_chat_template_content_format` `server/app.py:511`,
`apply_hf_chat_template` `server/app.py:518`), plus `get_score_prompt`
(`server/app.py:528`, SCORE only).

### A.4 LLM.__init__ -> artifact load (native)

`LLM.__init__` dispatches on `fxb`:

- `fxb is None` (default serve path) -> `LLM._init_from_artifact` (`api.py:216`).
- `fxb` provided -> `LLM._init_from_v3_engine` (`api.py:197`).

`_init_from_artifact` (`api.py:321`+):

1. `resolve_artifact_path` (`api.py:346`) ->
   `resolve_default_hf_revision` (`utils.py:378`) and
   `get_path_or_hf_download` (`utils.py:379`); the latter probes
   `huggingface_hub.hf_hub_download` (`utils.py:314`, **external**) for
   `artifact.json` then `huggingface_hub.snapshot_download` (`utils.py:315`,
   **external**) for the full snapshot.
2. `get_tokenizer` (`api.py:347`) -> `transformers.AutoTokenizer.from_pretrained`
   (`tokenizer/tokenizer.py:83`; HF tokenizers Rust backend).
3. **NATIVE** `furiosa.native_llm_common.NextGenArtifact.load_without_blob`
   (`api.py:349`) — deserializes `artifact.json` (+ safetensors headers) without
   the weight blob.
4. `resolve_devices` (`api.py:352`) -> `get_available_devices` (`device.py:201`)
   -> **NATIVE** `furiosa_smi_py.list_devices` (`utils.py:396`) when devices is
   `None`.
5. **NATIVE** `NextGenArtifact.override_with` (`api.py:354`) — loads the binary
   bundle + params and applies pp-stage / cache_dir overrides.
6. `compute_bucket_lengths` (`api.py:360`),
   `get_diff_sampling_params` (`api.py:378`) ->
   `try_load_generation_config` (`generation_config.py:42`) ->
   `transformers.GenerationConfig.from_pretrained` (`generation_config.py:28`).
7. Serialize config + tokenizer: `_serialize_obj` (`api.py:390`) ->
   **NATIVE** `pydantic_core.to_json` (`api.py:417`); and
   **NATIVE** `tokenizers.Tokenizer.to_str` via `backend_tokenizer.to_str()`
   (`api.py:392`).
8. **NATIVE** `furiosa.native_runtime.llm.NativeLLMEngine.__init__`
   (`api.py:383`) — constructs the runtime engine from `artifact_path` + devices;
   performs device mapping and warmup inside the `.so`. Skipped only when
   `skip_engine=True` (not set by the serve path).

The `_init_from_v3_engine` variant instead loads HF config/tokenizer directly
(`get_tokenizer` `api.py:266`, `AutoConfig.from_pretrained` `api.py:269`),
resolves devices (`api.py:297`), and constructs **NATIVE** `NativeLLMEngine`
(`api.py:301`).

> Note: `artifact/validator.py:188` `validate_resolved_buckets` ->
> **NATIVE** `furiosa.native_llm_common.compute_limits` is a *build-path*
> validation; it does **not** execute during serve-time `_init_from_artifact`.
> Likewise the `furiosa-ai/fake-llm` branch in `load_llm_from_args`
> (`server/models.py:20-30`) is test-only and bypasses the real engine.

---

## (B) Inference request path

HTTP request -> FastAPI route -> serving_* handler -> `AsyncLLMEngine` ->
**NATIVE** `NativeLLMEngine` -> NPU. Route handler bodies are bound to `router`
at import via decorators and registered at `app.include_router(router)`
(`server/app.py:468`); they call serving methods only at request time.

### B.1 Chat completions

`POST /v1/chat/completions` -> `create_chat_completion` (`server/app.py:191`):

- `utils.parse_request` validates `ChatCompletionRequest` (`server/app.py:190`).
- `OpenAIServingChat.create_chat_completion` (`serving_chat.py:143`):
  - standard path: `chat_utils.preprocess_chat` (`serving_chat.py:192`).
  - harmony/gpt_oss path: `_make_request_with_harmony` (`serving_chat.py:184`)
    -> `get_system_message` (`serving_chat.py:1376`),
    `get_developer_message` (`serving_chat.py:1384`),
    `parse_chat_input` (`serving_chat.py:1389`),
    `render_for_completion` (`serving_chat.py:1392`).
  - `ChatCompletionRequest.to_sampling_params` (`serving_chat.py:211`).
  - `AsyncLLMEngine.generate` — streaming branch `output_kind=DELTA`
    (`serving_chat.py:257`) or non-streaming `output_kind=FINAL`
    (`serving_chat.py:272`).
  - disconnect watcher `utils.handle_disconnect` (`serving_chat.py:261`) ->
    `_abort_request` -> `AsyncLLMEngine.abort` (`serving_chat.py:1363`).
  - streaming -> `chat_completion_stream_generator` (`serving_chat.py:263`);
    non-streaming -> `chat_completion_full_generator` (`serving_chat.py:279`)
    (`parse_chat_output` `serving_chat.py:1059`,
    `clamp_prompt_logprobs` `serving_chat.py:1253`).

`chat_utils.preprocess_chat` (`chat_utils.py:1447`):
`resolve_chat_template_content_format` (`chat_utils.py:1488`),
`parse_chat_messages_futures` (`chat_utils.py:1496`) ->
`_parse_chat_message_content` (`chat_utils.py:1287`),
`apply_hf_chat_template` (`chat_utils.py:1513`) ->
`resolve_hf_chat_template` (`chat_utils.py:1390`) +
`transformers.PreTrainedTokenizer.apply_chat_template` (`chat_utils.py:1411`),
and `transformers.PreTrainedTokenizer.encode` (`chat_utils.py:1537`).

### B.2 Completions

`POST /v1/completions` -> `create_completion` (`server/app.py:177`):
`parse_request` (`server/app.py:176`) ->
`OpenAIServingCompletion.create_completion` (`serving_completions.py:50`):
`CompletionRequest.to_sampling_params` (`serving_completions.py:69`),
`parse.parse_and_batch_prompt` (`serving_completions.py:79`),
one `AsyncLLMEngine.generate` per prompt (`serving_completions.py:112`),
`utils.merge_async_iterators` (`serving_completions.py:115`),
`handle_disconnect` (`serving_completions.py:123`),
then `completion_stream_generator` (`serving_completions.py:127`) or
`completion_full_generator` (`serving_completions.py:136`) ->
`request_outputs_to_completion_response` (`serving_completions.py:248`).
Abort: `_abort_request` -> `AsyncLLMEngine.abort` (`serving_completions.py:384`).

### B.3 Embeddings / score / tokenize / responses

- `POST /v1/embeddings` -> `OpenAIServingEmbedding.create_embedding`
  (`server/app.py:205`): `vllm_compat.preprocess_prompt`
  (`serving_embedding.py:70`), `AsyncLLMEngine.encode` (`serving_embedding.py:84`),
  `merge_async_iterators` (`serving_embedding.py:91`); abort ->
  `AsyncLLMEngine.abort` (`serving_embedding.py:132`).
- `tokenize` (`server/app.py:273`) -> `OpenAIServingTokenization.create_tokenize`
  -> `_tokenize_chat` (`serving_tokenization.py:65`) ->
  `preprocess_chat` (`serving_tokenization.py:87`); `detokenize`
  (`server/app.py:284`) -> `create_detokenize`.
- `create_responses` (`server/app.py:316`) ->
  `OpenAIServingResponses.create_responses`: `preprocess_chat`
  (`serving_responses.py:245`), `AsyncLLMEngine.generate`
  (`serving_responses.py:292`).

### B.4 AsyncLLMEngine -> native

`furiosa_llm.llm_engine.AsyncLLMEngine.generate`:
`vllm_compat.preprocess_prompt` (`llm_engine.py:600`, re-tokenize with
`add_special_tokens=False`), **NATIVE**
`NativeLLMEngine.stream_generate` (`llm_engine.py:611`), then
`outputs.NativeOutputConverter.convert_stream` (`llm_engine.py:626`).

`AsyncLLMEngine.encode`: `preprocess_prompt` (`llm_engine.py:658`),
`apply_prompt_truncation` (`llm_engine.py:660`), **NATIVE**
`NativeLLMEngine.encode` (`llm_engine.py:670`).
`AsyncLLMEngine.abort` -> **NATIVE** `NativeLLMEngine.abort_request`
(`llm_engine.py:684`).

`NativeOutputConverter.convert_stream` (`outputs.py:336`) async-iterates the
native generator, reads native attrs, calls `StreamDecoder.push_decode`
(`outputs.py:404`) -> `transformers.PreTrainedTokenizerBase.decode`
(`outputs.py:548`), and builds `RequestOutput` (`outputs.py:374`) yielded back to
the serving layer.

### B.5 Synchronous / offline API (engine-api)

The public `LLM` methods share the same native engine:
`LLM.generate` -> **NATIVE** `NativeLLMEngine.generate` (`api.py:466`) ->
`_generate_postprocess` (`api.py:467`, reads native `NativeCompletionOutput`
fields at `api.py:644`/`api.py:656`);
`LLM.stream_generate` -> **NATIVE** `NativeLLMEngine.stream_generate`
(`api.py:606`); `LLM.encode` (used by `embed`/`score`) -> **NATIVE**
`NativeLLMEngine.encode` (`api.py:773`) driven by
`run_sync(async_gather(...))` (`api.py:784`); `LLM.chat` renders the template
then delegates to `LLM.generate` (`api.py:539`); `LLM.shutdown` -> **NATIVE**
`NativeLLMEngine.shutdown` (`api.py:924`).

The sync `LLMEngine` wrapper schedules work on a background loop:
`add_request` -> `asyncio.run_coroutine_threadsafe` (`llm_engine.py:407`) ->
`_process_generation_request` -> **NATIVE** `NativeLLMEngine.stream_generate`
(`llm_engine.py:453`) -> `convert_stream` (`llm_engine.py:456`); pooling ->
`_process_encoding_request` -> **NATIVE** `NativeLLMEngine.encode`
(`llm_engine.py:465`); `abort_request` -> **NATIVE**
`NativeLLMEngine.abort_request` (`llm_engine.py:443`).

### B.6 Request-time native lifecycle / metrics

- `GET /health` -> `health` -> **NATIVE** `NativeLLMEngine.is_alive`
  (`server/app.py:119`).
- `GET /metrics` -> `_make_prometheus_endpoint.prometheus_app` -> **NATIVE**
  `get_metrics_as_prometheus_string` (`server/metrics.py:29`).
- `LogMetrics` daemon -> **NATIVE** `get_dp_metrics`
  (default_factory, `server/metrics.py:62`).

---

## (C) furiosa-llm build (AOT compile) path

This is **not** executed by serve; it is the path that produced the serve
artifact bundle. The native compiler boundary is `furiosa.native_common.compiler`
(PyO3 in `native_llm_common.so`).

1. `furiosa_llm.cli.convert.convert` builds configs and an `ArtifactBuilder`
   (`ArtifactBuilder.__init__` `cli/convert.py:197`) then calls
   `ArtifactBuilder.build` (`cli/convert.py:214`).
2. `ArtifactBuilder.__init__`: `load_hf_config` (`artifact/builder.py:146`),
   `resolver.resolve_model_metadata` (`artifact/builder.py:161`),
   `ResolvedBuckets.resolve` (`artifact/builder.py:168`; consults
   `presets.find_preset` `artifact/presets.py:404`).
3. `ArtifactBuilder.build`: **NATIVE** `compiler.compiler_git_short_hash`
   (`artifact/builder.py:357`), `ensure_model_and_update_weight_hash`
   (`artifact/builder.py:362`), `_build_model_artifact` (`artifact/builder.py:363`),
   `__save_artifacts` (`artifact/builder.py:393`).
4. `_build_model_artifact`: `ModelCreationInfo.__init__` (`artifact/builder.py:214`),
   `trace.get_param_file_with_cache` (`artifact/builder.py:233`),
   `ComposableKernelPipelineBuildConfigGenerator.__init__`
   (`artifact/builder.py:258`), `new_pipeline_builder.build_pipeline`
   (`artifact/builder.py:268`). The block-wise else-branch is dead
   ("Block-wise artifact is not supported anymore.").
5. `build_pipeline`:
   `get_needed_buckets` (`new_pipeline_builder.py:1532`);
   Ray fan-out `LocalPipelineGenerationActor.build_for_bucket`
   (`new_pipeline_builder.py:1576`) ->
   `build_local_pipeline` (`new_pipeline_builder.py:1451`) ->
   `build_partitioned_graphmodule` (`new_pipeline_builder.py:580`);
   `LocalToGlobalPipelineConverter.convert` (`new_pipeline_builder.py:1613`);
   `get_compiled_pipeline` (`new_pipeline_builder.py:1623`).
6. Tracing: `build_partitioned_graphmodule` ->
   `get_graph_partitioner` (`new_pipeline_builder.py:436`),
   `trace.get_aten_graph_with_metadata` (`new_pipeline_builder.py:455`) ->
   `_get_aten_graph_with_metadata` (`trace.py:1230`) ->
   `_trace_into_aten_graph_with_metadata` (`trace.py:1166`) ->
   `block_slicer.enable_marker_op` (`trace.py:943`) +
   `_get_aten_gm` (`trace.py:947`) -> `trace_model` (`trace.py:891`) ->
   `torch._dynamo.export` (`trace.py:855`) and
   `torch.fx.experimental.proxy_tensor.make_fx` (`trace.py:828`).
7. Partition + parallelize: `parallelize_and_partition_graphmodule`
   (`new_pipeline_builder.py:486`) ->
   `PartitionComposer.partition_gm` (`new_pipeline_builder.py:343`) ->
   `KernelwisePartitioner.partition_gm` (`graph_partitioner.py:97`) using
   `block_slicer.get_kernelwise_sliced_color_bitmap_with_marker`
   (`graph_partitioner.py:65`); and `parallelize_graphmodule`
   (`new_pipeline_builder.py:367`) -> `ModelRewriter.rewrite`
   (`new_pipeline_builder.py:147`).
8. Compile (Ray fan-out): `get_compiled_pipeline` ->
   `TaskCompileActor.compile_task` (`new_pipeline_builder.py:1312`):
   `CompilerConfigContext.load_config_with_layer_range`
   (`new_pipeline_builder.py:1127`),
   `deserialize_gm` (`new_pipeline_builder.py:1129`),
   `generate_graph_metadata` (`new_pipeline_builder.py:1153`),
   `GraphModuleConverter.compile_gm_and_get_preprocessed_gm_hash`
   (`new_pipeline_builder.py:1165`),
   **NATIVE** `compiler.CompiledGraph.serialize` (`new_pipeline_builder.py:1192`).
   Back in the driver, **NATIVE** `compiler.CompiledGraph.deserialize`
   (`new_pipeline_builder.py:1343`).
9. The actual compiler crossing: `compile_gm_and_get_preprocessed_gm_hash` ->
   `furiosa_torch_ext.torch_ext.preprocess` (`pipeline/builder/converter.py:886`)
   then **NATIVE** `furiosa.native_common.compiler.compile`
   (`pipeline/builder/converter.py:913`) — **THE** AOT compile step producing
   EDF/precommandgen blobs. Config: `load_config_with_layer_range` ->
   **NATIVE** `create_llm_compiler_config_with_layer_range`
   (`compiler_config.py:122`) / `create_default_compiler_config`
   (`compiler_config.py:142`). Metadata: `generate_graph_metadata` ->
   **NATIVE** `compiler.GraphMetadataBuilder` (`pipeline/builder/converter.py:1461`).
   Preset matching: `presets.find_preset` -> **NATIVE**
   `compiler.approx_per_layer_params_b` (`artifact/presets.py:416`).
10. `__save_artifacts` (`artifact/builder.py:393`): `__preprocess_for_pipeline_save`
    (`artifact/builder.py:498`) -> **NATIVE** `compiler.CompiledGraph.serialize`
    (`artifact/builder.py:455`); `get_tokenizer` (`artifact/builder.py:510`); bundles
    into `binary_bundle.zip` + `artifact.json`.

---

## Python -> native boundary crossings

These are the exact Python-symbol -> native-target edges the dynamic trace
layers (gdb/perf on the `.so`, eBPF on `ioctl`/`rngd`) should hook. Native targets
are PyO3 symbols inside `furiosa/native_runtime.cpython-312*.so` (engine /
device / scheduler), `furiosa/native_llm_common.cpython-312*.so` (artifact
metadata), or third-party Rust (`pydantic_core`, `tokenizers`, `huggingface_hub`,
`furiosa_smi_py`).

### Serve startup + artifact load (scenario A)

| Python symbol | Native target | Site |
|---|---|---|
| `LLM._init_from_artifact` | `furiosa.native_llm_common.NextGenArtifact.load_without_blob` | `api.py:349` |
| `LLM._init_from_artifact` | `furiosa.native_llm_common.NextGenArtifact.override_with` | `api.py:354` |
| `LLM._init_from_artifact` | `furiosa.native_runtime.llm.NativeLLMEngine.__init__` | `api.py:383` |
| `LLM._init_from_v3_engine` | `furiosa.native_runtime.llm.NativeLLMEngine.__init__` | `api.py:301` |
| `LLM._serialize_obj` | `pydantic_core.to_json` | `api.py:417` |
| `LLM._init_from_artifact` | `tokenizers.Tokenizer.to_str` (`backend_tokenizer.to_str()`) | `api.py:392` |
| `tokenizer.get_tokenizer` | `transformers.AutoTokenizer.from_pretrained` (HF tokenizers Rust) | `tokenizer/tokenizer.py:83` |
| `utils.get_path_or_hf_download` | `huggingface_hub.hf_hub_download` | `utils.py:314` |
| `utils.get_path_or_hf_download` | `huggingface_hub.snapshot_download` | `utils.py:315` |
| `utils.get_available_devices` | `furiosa_smi_py.list_devices` (PyO3 SMI) | `utils.py:396` |
| `server/app.lifespan` (shutdown) | `NativeLLMEngine.shutdown` | `server/app.py:465` |
| `prewarm_server` | `NativePoolingOutput.__new__` (force-load Rust numpy) | `server/app.py:501` |
| `server/app.show_version` | `furiosa.native_runtime.__version__` (+ ir/build attrs) | `server/app.py:136` |

### Inference (scenario B)

| Python symbol | Native target | Site |
|---|---|---|
| `AsyncLLMEngine.generate` | `NativeLLMEngine.stream_generate` | `llm_engine.py:611` |
| `AsyncLLMEngine.encode` | `NativeLLMEngine.encode` | `llm_engine.py:670` |
| `AsyncLLMEngine.abort` | `NativeLLMEngine.abort_request` | `llm_engine.py:684` |
| `LLMEngine._process_generation_request` | `NativeLLMEngine.stream_generate` | `llm_engine.py:453` |
| `LLMEngine._process_encoding_request` | `NativeLLMEngine.encode` | `llm_engine.py:465` |
| `LLMEngine.abort_request` | `NativeLLMEngine.abort_request` | `llm_engine.py:443` |
| `LLM.generate` | `NativeLLMEngine.generate` (blocking) | `api.py:466` |
| `LLM.stream_generate` | `NativeLLMEngine.stream_generate` | `api.py:606` |
| `LLM.encode` | `NativeLLMEngine.encode` | `api.py:773` |
| `NativeOutputConverter.convert_stream` | iterate `NativeRequestOutput` (attr reads) | `outputs.py:336` |
| `LLM._generate_postprocess.convert` | `NativeCompletionOutput` (attr reads) | `api.py:644` |
| `server/app.health` | `NativeLLMEngine.is_alive` | `server/app.py:119` |
| `_make_prometheus_endpoint.prometheus_app` | `get_metrics_as_prometheus_string` | `server/metrics.py:29` |
| `LogMetrics` | `get_dp_metrics` | `server/metrics.py:62` |

The hot per-token crossing for streaming serve is
`NativeLLMEngine.stream_generate` (yields `NativeRequestOutput`); inside the
`.so` this is what drives NPU execution (`ioctl` to `rngd`, DMA/PDMA descriptor
build, dispatch to PE units `pe_0/pe_1/pe_2`). None of this is an exported C-ABI
function — it is reached purely via PyO3 method calls — so the kernel layer must
hook `ioctl`/`rngd` and the native layer must hook inside the `.so`.

### Build (scenario C) — not on serve path

| Python symbol | Native target | Site |
|---|---|---|
| `GraphModuleConverter.compile_gm_and_get_preprocessed_gm_hash` | `furiosa.native_common.compiler.compile` | `pipeline/builder/converter.py:913` |
| `CompilerConfigContext.load_config_with_layer_range` | `compiler.create_llm_compiler_config_with_layer_range` | `compiler_config.py:122` |
| `CompilerConfigContext.load_config_with_layer_range` | `compiler.create_default_compiler_config` | `compiler_config.py:142` |
| `generate_graph_metadata` | `compiler.GraphMetadataBuilder` | `pipeline/builder/converter.py:1461` |
| `TaskCompileActor.compile_task` | `compiler.CompiledGraph.serialize` | `new_pipeline_builder.py:1192` |
| `get_compiled_pipeline` | `compiler.CompiledGraph.deserialize` | `new_pipeline_builder.py:1343` |
| `ArtifactBuilder.__preprocess_for_pipeline_save` | `compiler.CompiledGraph.serialize` | `artifact/builder.py:455` |
| `presets.find_preset` | `compiler.approx_per_layer_params_b` | `artifact/presets.py:416` |
| `ArtifactBuilder.build` | `compiler.compiler_git_short_hash` | `artifact/builder.py:357` |
| `validator.validate_resolved_buckets` | `furiosa.native_llm_common.compute_limits` | `artifact/validator.py:188` |

---

## Open questions / to verify dynamically

1. **Native syscall mapping.** `NativeLLMEngine` is a PyO3 Rust class; the NPU
   `ioctl`/DMA syscalls to `rngd` (building `dma_desc`/PDMA descriptors,
   dispatching to PE units `pe_0/pe_1/pe_2`, the scheduler) are internal to
   `native_runtime.so` and are **not** exported C-ABI symbols. Which Python method
   (`generate` vs `stream_generate` vs `encode` vs `__init__` warmup) triggers
   `ioctl` vs shared-memory DMA can only be confirmed with gdb/perf inside the
   `.so` + eBPF on `ioctl`/`rngd`.
2. **Shared Rust core packaged twice.** Both `native_runtime.so` and
   `native_llm_common.so` export `PyInit_native_llm_common` and the same torch
   stream / `perf_signal_handler` symbols; whether `furiosa.native_llm_common` at
   runtime resolves into `native_runtime.so` or the standalone
   `native_llm_common.so` should be confirmed at load time (e.g. `lsof`/`/proc/maps`).
3. **Guided decoding (`llg_*`).** The llguidance C-ABI symbols (44 of them) exist
   in `native_runtime.so` but no `furiosa_llm` Python file calls them; they are
   invoked internally when `SamplingParams.structured_outputs_backend`
   (guidance/xgrammar) is set. Whether structured-output requests change the
   inference native call path needs a runtime trace with a grammar-constrained
   request.
4. **NPU warmup / device-map in `NativeLLMEngine.__init__`.** The device mapping
   and warmup steps (`api.py:383`) are native and not statically inspectable;
   confirm the warmup `ioctl`/DMA pattern at startup dynamically.
5. **Artifact file I/O split.** `load_without_blob` vs `override_with`
   (`api.py:349` / `api.py:354`) — exactly which files each touches
   (`artifact.json` vs `binary_bundle.zip` vs `params-*.safetensors`) is inferred
   from naming; verify with strace/eBPF on `open`/`read` during startup.
6. **Request-time route -> serving edges.** All `@router` handler bodies call the
   serving methods only at request time; the static graph captured the startup
   registration and the representative `/health` -> native edge. Confirm the full
   per-route request fan-out (`create_chat_completion` etc.) with a live request.
7. **Tokenizer Rust dispatch.** `transformers` fast-tokenizer `encode`/`decode`
   likely dispatches into the Rust `tokenizers` lib, but that was not verified
   within `furiosa_llm`; trace if tokenization cost matters.
8. **Streaming attr reads cross the boundary.** Per-output reads on
   `NativeRequestOutput`/`NativeCompletionOutput`
   (`.outputs/.token_ids/.logprobs/.prompt_logprobs/.finish_reason`) are PyO3
   property accesses, recorded as notes not separate call edges — confirm their
   cost/frequency dynamically since they fire per decode step.
