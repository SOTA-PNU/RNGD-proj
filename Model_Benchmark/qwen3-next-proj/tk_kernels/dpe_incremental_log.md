# RECON-C: EinsumByVe -> EinsumByDpe, incremental error-driven derivation

Goal: convert the proven `dn_linear.yaml` (kind: EinsumByVe, slow VECTOR-engine
matmul) into a working `kind: EinsumByDpe` (fast systolic/DPE-MAC matmul) by
flipping `kind` and filling struct fields ONE VALIDATOR ERROR AT A TIME.

- Source (working VE):  `dn_linear.yaml`  y[t,o]=sum_i x[t,i]*W[o,i] == F.linear
- Target (this work):   `dn_linear_dpe.yaml`
- Probe runner:         `run_dpe_probe.py`  (compile+exec on rngd, spies _dfg_inner)
- Run env: `PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj RNGD_DEV=rngd:3 /home/jun/furiosa/bin/python run_dpe_probe.py`
  (rngd:2 was EBUSY during this session; rngd:3 was free.)

## RESULT: SUCCESS — full matmul runs on PURE NPU via the DPE engine.

- dfg_inner == 0 (no CPU fallback) for all tested shapes.
- ~3.8x faster than the EinsumByVe twin: **3.51 ms (DPE) vs 13.33 ms (VE)** for
  [128,512] @ [2048,512], 20-iter avg, rngd:3.
- Precision: DPE accumulates in reduced precision. maxabs <= 2.4e-3,
  relmean ~1.6%. `allclose` passes at atol/rtol=1e-2, FAILS at 1e-3. Expected
  tradeoff (matches the known "matmul ~0.23%+ reduced-precision" DPE behavior).
  Shapes verified: [128,512,2048] [128,256,128] [128,32,256] [256,2048,512].

---

## The serde map (from native_torch.so strings) that guided the fills

`struct OperatorTacticEinsumByDpe with 14 elements`, field order:
`input, filter, output_element_type, contraction, schedule_config, swap_inputs,
input_paddings, input_slidings, acc_major_mode, reduce_mode, separate_vector_ops,
force_mask_to_input`.
**IMPORTANT**: those 12 fields belong to the *LOWERED* `OperatorTacticEinsumByDpe`
(what the COMPILER emits). At the hand-authored TacticKernel DSL level, `kind:
EinsumByDpe` is just a tag on the same `{reads, ein_ops, vector_ops, write}` inner
struct as EinsumByVe. None of input/filter/contraction/swap_inputs/... are written
by hand here; the contraction is driven entirely by `ein_ops`.

Other relevant structs: `EinOps = { reduce, mul_source }`; the `reduce` sub-struct
needs `{ mode, input, axes, source }`; `ReduceMode` enum = Add|Max|Mul|Generic
("DPE allows only add and max for reducing"). Unary op enum has NO identity
(probe_unary.py: Exp/Sigmoid/Sqrt/Tanh/Sin/Cos/Log compile; Identity/Neg/Abs/Square
are "unknown enum variant"). Binary operands are Tensor-only (no Const variant).

---

## Error -> Fix sequence

### Iter 1 — flip kind only (`EinsumByVe` -> `EinsumByDpe`)
- parse: OK
- exec error (npu-ir/src/tactic_kernel/mod.rs:286):
  `Condition failed: self.inner.ein_ops.as_ref().is_some_and(|ein_ops|
   { ein_ops.reduce.mode.is_dpe_supported_mode() })`
- DIAGNOSIS: DPE reads `ein_ops` (VE used `vector_ops`). `ein_ops: ~` (null) is
  rejected; needs a populated reduce whose mode is Add or Max.
- FIX: add `ein_ops: { reduce: { mode: Add, tags: [...] }, mul_source: ~ }`.

### Iter 2 — first ein_ops guess
- parse error (graph_embedding.rs:37):
  `ein_ops: field errors (reduce: field errors (input: missing field,
   axes: missing field, source: missing field), mul_source: invalid value
   (string expected))`
- DIAGNOSIS: `reduce` needs fields `input`, `axes`, `source` (NOT `tags`).
  `mode` was accepted (not complained about). `mul_source` is a STRING field.
- FIX: `reduce: { mode: Add, input: 2, axes: {Tag:[{inner:"i"}]}, source: "" }`,
  `mul_source: ""`.

### Iter 3 — input as int, axes as Tag-map
- parse error:
  `reduce: field errors (input: invalid value (object expected),
   axes: invalid value (array expected))`
- DIAGNOSIS: `mode/source/mul_source` now accepted. `reduce.input` must be an
  OBJECT (a tensor descriptor), not a tensor index. `reduce.axes` must be a plain
  ARRAY, not the `{Tag: [...]}` wrapper.
- FIX: `input: {}` (empty object to probe), `axes: [ {inner:"i"} ]`.

### Iter 4 — empty input object, axes elements as {inner:..}
- parse error:
  `reduce: field errors (input: field errors (shape: missing field,
   element_type: missing field), axes: unknown enum variant)`
- DIAGNOSIS: `reduce.input` is the SAME TensorLike struct used by `reads[].input`
  and `write.input`: `{ shape: {inner:[...]}, element_type }`. Each `axes` element
  is an `AxisTag` enum variant (e.g. `LabelStride`), not a bare `{inner:..}`.
- FIX: `reduce.input.shape` = the PRE-reduction product tensor over [t,o,i] with
  `element_type: Float32`; `axes: [ { LabelStride: { label:{inner:"i"}, stride:1 } } ]`.

### Iter 5 — full reduce.input + LabelStride axes
- parse: OK  (ein_ops fully accepted)
- exec error (npu-ir/operators/src/tensor_kernel/mod.rs:390):
  `input_descriptors.len() (1) != vector_ops.inputs.len() (2)`
  (validator now RENDERS the DPE table: INPUT[0] broadcast o, INPUT[1] broadcast t,
   EinOps reduce:+i, VectorOps #0 mul #1 reduce, Write [t,o].)
- DIAGNOSIS: with `ein_ops` doing the contraction, the two reads collapse to ONE
  post-contraction descriptor. `vector_ops` must take EXACTLY 1 input, but the VE
  carryover declared 2 (`inputs: [0,1]`).

### Iter 6 — vector_ops inputs:[0], single Unary PassF
- parse error: `vector_ops: ... expr: field errors (operator: unknown enum variant)`
- DIAGNOSIS: `PassF` is not a valid Unary operator. (Structure Unary/operand OK.)
- FIX attempt: make vector_ops a no-op -> set `insts: []`.

### Iter 7 — inputs:[0], insts:[]
- parse: OK
- exec error (tensor_kernel/vector_ops/mod.rs:66):
  `&self.inputs ([DefId(0)]) != &self.input_tensors() ([])`
- DIAGNOSIS: declared input 0 but no inst references it. inputs must equal the
  tensors actually used by insts.
- FIX attempt: empty BOTH -> `inputs: []`, `insts: []`.

### Iter 8 — inputs:[], insts:[]
- parse: OK
- exec error (tensor_kernel/mod.rs:390):
  `input_descriptors.len() (1) != vector_ops.inputs.len() (0)`
- DIAGNOSIS: CONTRADICTION with iter 7. `input_descriptors` is FIXED at 1 (the
  EinOps output). So vector_ops MUST have exactly 1 input AND that input must be
  consumed by an inst. => need a single-input identity/passthrough inst.

### Iter 9 — Binary AddF with Const 0.0 (identity via add-zero)
- parse error: `expr: field errors (rhs: unknown enum variant)`
- DIAGNOSIS: Binary operands are Tensor-only; no `Const` operand variant exists.
  (Confirmed by probe_binary.py which only ever uses {Tensor: N}.)

### Iter 10 — vector_ops MulF over the 2 reads (inputs:[0,1])
- parse: OK
- exec error: `input_descriptors.len() (1) != vector_ops.inputs.len() (2)`
- DIAGNOSIS: re-confirms input_descriptors==1 regardless of read count. vector_ops
  truly sees only the single contracted result; cannot reference the raw reads.

### Iter 11 — Reduce(LocalReduceAddF) over EMPTY axes = identity  ✅
- `vector_ops: { inputs: [0], insts: [ { def:1, expr: Reduce{ operator:
   LocalReduceAddF, operand: Tensor 0, axes: {Tag: []} } } ] }`
- parse: OK
- exec: **OK, dfg=NPU (pure NPU), err=1.3e-3.**
- A Reduce over an empty axes list is a valid no-op identity: 1 input, references
  it, satisfies both `input_descriptors==1` AND `inputs==input_tensors`. SUCCESS.

---

## The minimal EinsumByDpe delta vs EinsumByVe (load-bearing fields)

1. `kind: EinsumByDpe` (was EinsumByVe).
2. `ein_ops` MUST be non-null and carry the contraction:
   ```yaml
   ein_ops:
     reduce:
       mode: Add                 # Add or Max only (DPE-supported)
       input:                    # TensorLike of the PRE-reduce product [t,o,i]
         shape: { inner: [ {tag:{LabelStride:{label:{inner:"t"},stride:1}}, size:{Var:T}},
                           {tag:{LabelStride:{label:{inner:"o"},stride:1}}, size:{Var:O}},
                           {tag:{LabelStride:{label:{inner:"i"},stride:1}}, size:{Var:I}} ] }
         element_type: Float32
       axes:                     # PLAIN ARRAY of AxisTag enum variants
         - LabelStride: { label: {inner:"i"}, stride: 1 }   # reduce/contract over "i"
       source: ""
     mul_source: ""              # STRING (not null)
   ```
3. `vector_ops` MUST have exactly 1 input (the contracted EinOps result) and a
   single identity passthrough inst (no DSL identity op exists, so use a Reduce
   over EMPTY axes):
   ```yaml
   vector_ops:
     inputs: [0]
     insts:
       - def: 1
         expr:
           Reduce: { operator: LocalReduceAddF, operand: {Tensor: 0}, axes: {Tag: []} }
         source: ""
   ```
4. `reads` and `write` are UNCHANGED from the EinsumByVe version (the broadcast
   read0 / read1 tiling still drives input vs filter).

## Notes / gotchas
- `EinsumByDpe should have at most one Reduce instruction at last position in
  VectorOps` (seen in serde) — consistent with our single trailing Reduce.
- `input_descriptors.len()` for DPE == 1 ALWAYS (the fused EinOps output). You can
  never reference the raw reads from vector_ops once ein_ops is present.
- There is NO identity Unary and NO constant operand in the naive_yaml DSL; the
  empty-axes Reduce is the canonical passthrough.
- Get the FULL inner error by reading the HEAD of the probe output: the tail shows
  a misleading `furiosa::dfg only runs on CPU device` / `allocator != nullptr`
  (CPU-fallback artifacts). The real cause is the `compiled_ops_err: ... verification
  of operator failed` block near the top.

---

## 2026-06-11 — applying the recipe to the DeltaNet chunk-scan (5 internal matmuls)

`dn_chunk_full.yaml` (gen by `gen_chunk_full.py`) has 5 EinsumByVe matmuls inside one
graph: op0 qk `ck,dk->cd` (reduce k), op3 attn_inter `ck,kv->cv` (reduce k), op4
v_prime `dk,kv->dv` (reduce k), op6 intra `cd,dv->cv` (reduce d), op9 kv `dk,dv->kv`
(reduce d). The other 7 ops are pure Elementwise (Mul/Sub/Add) and stay on the VE.

Extended `gen_chunk_full.py` with `emit(mh, dpe=False, dpe_max=None)`. Each matmul
converts cleanly with the proven recipe: pre-reduce product = `[write axes...,
contracted axis]` (qk -> [c,d,k] reduce k; intra -> [c,v,d] reduce d; etc.),
identity empty-axes-Reduce vector_ops. Generated `dn_chunk_full_dpe.yaml` (all 5 DPE)
and `dn_chunk_full_dpe2.yaml` (dpe_max=2).

### BLOCKER FOUND: at most 2 EinsumByDpe ops per TacticKernel graph

- Each of the 5 matmuls converts and runs CORRECTLY *in isolation* (single-op yaml,
  maxabs ~1e-3, relmean ~1-4%, the normal bf16-DPE precision). Verified all 5.
- Hybrid graphs (VE base, swap N op-bodies to DPE) on real chunk inputs:
  - **1 DPE op**  -> out maxabs ~8e-4   (fine)
  - **2 DPE ops** -> out maxabs ~1e-3   (fine, any pair)
  - **3 DPE ops** -> out maxabs ~6e-1   (GARBAGE) — step change, not gradual.
  - 4, 5 DPE ops -> same ~6e-1 garbage.
- Independent of WHICH ops and of data dependencies: `[0,3,4]`, `[3,4,9]`, `[0,6,9]`,
  `[0,3,9]` all break identically even though those ops don't feed each other and
  are separated by VE elementwise ops in program order.
- `_dfg_inner` stays **0** the whole time — the corrupt result is computed ON the
  NPU (no CPU fallback). It's a silent miscompile, not a fallback.
- Root cause (strings in native .so): compiler config has
  `fuse_mamma_to_single_einsum_by_dpe` + `separate_vector_ops_from_dpe`. The
  mamba/linear-attn lowering pass tries to FUSE multiple matmul-like DPE ops into a
  single systolic EinsumByDpe; with 3+ in one graph the fusion mis-schedules the
  systolic array and emits garbage. These are compiler knobs NOT exposed through the
  hand-authored naive_yaml DSL, so the only lever from the YAML side is to keep the
  per-graph DPE count <= 2.

### Result (full DeltaNet layer vs HF, real layer-0 weights, T=32, rngd)
- `dn_chunk_full.yaml`      (0 DPE / all VE):  maxerr_vs_hf = 8.94e-08  (matches HF)
- `dn_chunk_full_dpe.yaml`  (5 DPE):           maxerr_vs_hf ~ 5.2e-01   FAILS 1e-2
- `dn_chunk_full_dpe2.yaml` (2 DPE, op0+op3):  maxerr_vs_hf <= 1e-2     WORKS (see run)

Takeaway: the recipe itself is correct for every DeltaNet matmul; the DeltaNet chunk
graph just can't host more than 2 of them on the DPE at once. `dn_chunk_full_dpe2.yaml`
is the largest validating variant. To DPE-accelerate the *whole* scan you'd have to
split the chunk into multiple kernels (<=2 DPE matmuls each) — a graph-partitioning
change, not a YAML field change.
