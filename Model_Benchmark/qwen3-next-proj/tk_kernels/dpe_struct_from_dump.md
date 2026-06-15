# RECON-A: EinsumByDpe struct extracted from the Furiosa COMPILER's own lowering

Goal: get the real `EinsumByDpe` / `TuContraction` struct that the Furiosa compiler
emits when it lowers an ordinary matmul, so we can copy it into a hand-authored
`#naive_yaml` `SymTacticKernel` op (replacing our slow `kind: EinsumByVe`).

Date: 2026-06-11. Toolchain: `furiosa.native_torch` (npu-tools rev `3f23a71`).
All numbers below are **concrete values produced by the compiler**, not guesses.

---

## 1. How the struct was obtained (reproducible)

Bare `torch.export` of `aten::matmul` is **not** importable by `compiler.lower()`
(`dfg_import/aten_ops.rs:465 — Not importable, matmul`). The torch.compile path
*decomposes* it into `aten::mm`, which **is** importable. So the recipe is:

1. Run a plain matmul model through the real backend to let dynamo decompose it and
   capture the (importable) `ExportedProgram`:
   - monkeypatch `furiosa.native_torch.compiler.compile` to stash its `ep` argument.
   - `cm = torch.compile(MM(), backend=ft.backend); cm(a.to('rngd:2'), b.to('rngd:2'))`.
2. Re-lower that captured EP: `dfg = compiler.lower(ep, compiler_config=cfg)`.
3. `dfg.serialize_to_str()` → **base64 of an 8-byte length prefix + CBOR**.
   Decode with: `cbor2.loads(base64.b64decode(s)[8:])`.
4. Walk `Dfg.graph.operators.operators[*].option.SplitOperation.operation.Seq…` →
   `…operators[3].option.LowLevelTuContraction`.

Scripts (saved):
- `tk_kernels/dpe_capture_probe.py` — capture EP + lower + grep (the working one).
- `tk_kernels/dpe_lower_probe.py` — direct lower (fails: bare matmul not importable).
- `tk_kernels/dpe_dump_probe.py` — `cache_context(summary=True)` only writes
  `key.bin`/`output.bin`, **no IR-viewer json**, so it is not the dump path.
- decoded artifacts: `tk_kernels/_dpe_dfg/lowlevel_tucontraction_full.json`,
  `tk_kernels/_dpe_dfg/mm_huge_decoded.json`, plus per-case `*.dfg.txt` / `*.pprint.txt`.

`dfg.pprint()` is a Python-linter-friendly view that **collapses every tactic op
into a generic `SplitOperation`** and never prints the word `EinsumByDpe` — that is
why a plain text grep for `EinsumByDpe`/`EinsumByVe` over the serialized DFG returns
nothing. The real tactic lives in the CBOR under `LowLevelTuContraction`.

---

## 2. HEADLINE RESULT: the compiler lowers EVERY matmul to DPE, never VE

| case            | shapes            | tactic                | dpe_element_type | VE markers |
|-----------------|-------------------|-----------------------|------------------|------------|
| mm8             | 8×8 · 8×8         | LowLevelTuContraction | Bfloat16         | none       |
| mm32            | 32×32 · 32×32     | LowLevelTuContraction | Bfloat16         | none       |
| mm128           | 128×256 · 256×128 | LowLevelTuContraction | Bfloat16         | none       |
| mm_big1         | 128×2048·2048×512 | LowLevelTuContraction | Bfloat16         | none       |
| mm_big2         | 512×2048·2048×2048| LowLevelTuContraction | Bfloat16         | none       |
| mm_huge         | 256×4096·4096×4096| LowLevelTuContraction | Bfloat16         | none       |

So the question "does the compiler ever use DPE for matmul?" → **yes, always**, even
8×8, under every `TacticSortingPolicy`/`TacticHintConfig` tried (`Default`, `ForMiscModel`,
`ForLlmModelPrefill`, `NoConstraint`). The compiler **never** emits `EinsumByVe` for a
matmul; VE is only what *we* hand-write. Note: the compiler also **casts f32 inputs to
bf16** (`dpe_element_type = Bfloat16`, `trf_element_type = Bfloat16`) to feed the
systolic MAC array — DPE is a bf16 engine here, not f32. This is the precision tradeoff
of the fast path.

---

## 3. Field layouts (canonical serde order, from native_torch.so strings)

`OperatorTacticEinsumByDpe` (the **high-level** tactic op, what a YAML `kind:
EinsumByDpe` becomes) carries these extra fields, in order:

```
input, filter, output_element_type, contraction, schedule_config,
swap_inputs, input_paddings, input_slidings, acc_major_mode, reduce_mode,
separate_vector_ops, force_mask_to_input
```

`TuContraction` = "struct with 22 elements". Its axis-category fields (the ones the
lowered form actually populates), in serde order:

```
split, cluster, slice_dummy, peslice(=pe), ve_gat, gat, segment,
outer_acc, ve_acc, acc, shift_reuse_acc, feed_reuse, feed_reuse_acc,
dpe_reg_tile, dpe_elementwise, lat
```

plus (seen in the LowLevel form) `dpe_element_type, trf_element_type, mac_rows,
ve_reduce_labels`, and scalars `filter, reduce_mode, acc_major_mode,
input_zero_point, filter_zero_point, tail_padding, slidings, ve_pass`.

`TuContractionAxis` = "struct with 5 elements":

```
tag, is_input_tile, is_filter_tile, is_contraction     (5th = tag's inner payload)
```

Each axis `tag` is a `LabelStride { label: {inner: "<name>"}, stride: <int> }` plus a
`size`. `reduce_mode` ∈ {`Add`, `Max`} only ("**DPE allows only add and max for
reducing**" — string confirmed in the .so). `acc_major_mode` ∈ {`RowMajor`, `ColMajor`}.

The `"incompatible sequences"` execution error our EinsumByDpe hits sits in the .so
right next to `Einsum{ lhs, rhs, output_equation, contraction_order }` — i.e. the
runtime cannot build the DPE feed/accumulate **sequence** because `contraction` (the
22-field axis assignment) is empty in our hand-written op. The struct below is what
must be filled in.

---

## 4. CONCRETE STRUCT — minimal case: 8×8×8 matmul `ik = sum_j a[i,j]·b[j,k]`

`LowLevelTuContraction` emitted for `mm8` (axis labels: `0`=output-row i, `1`=output-col
k, `2`=contraction j/K):

```yaml
LowLevelTuContraction:
  input: 32            # tensor id of a[i,j]  (lhs / stationary-or-moving)
  filter: 33           # tensor id of b[j,k]  (rhs / weight)
  output: 53           # tensor id of out[i,k]
  reduce_mode: Add
  acc_major_mode: ColMajor
  input_zero_point: None
  filter_zero_point: None
  tail_padding: [0, 0]
  slidings: []
  vrfs: []
  inner:                       # the 22-element TuContraction axis assignment
    dpe_element_type: Bfloat16
    trf_element_type: Bfloat16
    split:          { axes: [] }
    chip:           { axes: [] }
    cluster:        { axes: [] }
    slice_dummy:    { axes: [] }
    pe:             { axes: [] }
    slice:                     # output row tile (i) -> filter/weight side
      axes:
        - tag: { LabelStride: { label: {inner: "0"}, stride: 1 } }
          size: 8
          is_input_tile: false
          is_filter_tile: true
          is_contraction: false
    ve_gat:         { axes: [] }
    gat:                       # the CONTRACTION axis (K=j) -> gathered/fed
      axes:
        - tag: { LabelStride: { label: {inner: "2"}, stride: 1 } }
          size: 8
          is_input_tile: false
          is_filter_tile: false
          is_contraction: true
    segment:        { axes: [] }
    outer_acc:      { axes: [] }
    ve_acc:         { axes: [] }
    acc:            { axes: [] }
    shift_reuse_acc:{ axes: [] }
    feed_reuse:     { axes: [] }
    feed_reuse_acc: { axes: [] }
    mac_rows:                  # output col tile (k) -> input side, drives MAC rows
      axes:
        - tag: { LabelStride: { label: {inner: "1"}, stride: 1 } }
          size: 8
          is_input_tile: true
          is_filter_tile: false
          is_contraction: false
    dpe_reg_tile:   { axes: [] }
    dpe_elementwise:{ axes: [] }
    lat:            { axes: [] }
    ve_reduce_labels: []
  ve_pass:                     # == separate_vector_ops ; empty-ish (no fused VE op)
    inputs: ["Sram"]
    pass_contexts: [ { ve_pass: { ... } } ]   # full blob in lowlevel_tucontraction_full.json
    mask_operands: []
```

Reading of the axis assignment for an 8×8×8 matmul `out[i,k] = Σ_j a[i,j]·b[j,k]`:
- label `"0"` (i, the M dim) → **`slice`**, `is_filter_tile=true`  (walks the weight rows)
- label `"1"` (k, the N dim) → **`mac_rows`**, `is_input_tile=true` (mapped onto the MAC array rows)
- label `"2"` (j, the K/contraction dim) → **`gat`**, `is_contraction=true` (fed/accumulated)

`reduce_mode: Add`, `acc_major_mode: ColMajor`, everything in **bf16**.

### Larger case (mm_huge 256×4096·4096×4096) — same shape of struct, more tiling
`acc_major_mode: RowMajor`, `reduce_mode: Add`, and the contraction is split across
**`gat` + `acc`(3 sub-axes) + `feed_reuse`/`feed_reuse_acc`** while the output dims use
`slice` + `outer_acc` + `mac_rows`(2) + `dpe_reg_tile`. Full values in
`_dpe_dfg/lowlevel_tucontraction_full.json`. The 22-field categories are exactly the
knobs the compiler turns to tile a big GEMM onto 8 PEs; for a hand-written kernel the
**minimal mm8 assignment above is the one to copy.**

---

## 5. Reconstructed `#naive_yaml` SymTacticKernel DPE op (matmul y[t,o]=Σ_i x[t,i]·W[o,i])

This mirrors `dn_linear.yaml` (same tensors/inputs/outputs) but swaps
`kind: EinsumByVe` for `kind: EinsumByDpe` and supplies the `contraction`
(`TuContraction`) the runtime needs. Label mapping for `y[t,o]=Σ_i x[t,i]·W[o,i]`:
`t`=output rows (M), `o`=output cols (N), `i`=contraction (K).

> Status: NOT yet executed. This is the struct to try; the high-level `EinsumByDpe`
> serde wrapper expects `input/filter/output_element_type/contraction/schedule_config/
> swap_inputs/input_paddings/input_slidings/acc_major_mode/reduce_mode/
> separate_vector_ops/force_mask_to_input` (Section 3). The compiler-proven axis
> assignment for the 22-field `contraction` is the mm8 mapping in Section 4.

```yaml
# kind: EinsumByDpe  --  y[t,o] = sum_i x[t,i]*W[o,i]  on the DPE systolic engine.
# Inputs: 0 = x[t,i] (input), 1 = W[o,i] (filter).  Output: 2 = y[t,o].
operators:
  operators:
    0:
      name: dn_linear_dpe
      option:
        SymTacticKernel:
          inputs: [0, 1]
          output: 2
          inner:
            inner:
              # reads/vector_ops/write keep the SAME framing as dn_linear.yaml's
              # EinsumByVe (the DSL still needs the tiled reads + write); the DPE
              # tactic is selected by `kind` + the `contraction` block below.
              # ... (reuse dn_linear.yaml reads/write verbatim) ...
            kind: EinsumByDpe
            # ---- the extra DPE fields (OperatorTacticEinsumByDpe order) ----
            output_element_type: Bfloat16     # DPE runs bf16 (compiler does f32->bf16)
            swap_inputs: false
            input_paddings: []
            input_slidings: []
            acc_major_mode: ColMajor
            reduce_mode: Add                  # DPE: Add or Max ONLY
            separate_vector_ops: ~            # ve_pass empty for a bare matmul
            force_mask_to_input: false
            contraction:                      # the 22-element TuContraction
              dpe_element_type: Bfloat16
              trf_element_type: Bfloat16
              split:          { axes: [] }
              cluster:        { axes: [] }
              slice_dummy:    { axes: [] }
              pe:             { axes: [] }
              slice:                          # output-row dim t  -> filter tile
                axes:
                  - tag: { LabelStride: { label: {inner: "t"}, stride: 1 } }
                    size: { Var: T }
                    is_input_tile: false
                    is_filter_tile: true
                    is_contraction: false
              ve_gat:         { axes: [] }
              gat:                            # contraction dim i  -> is_contraction
                axes:
                  - tag: { LabelStride: { label: {inner: "i"}, stride: 1 } }
                    size: { Var: I }
                    is_input_tile: false
                    is_filter_tile: false
                    is_contraction: true
              segment:        { axes: [] }
              outer_acc:      { axes: [] }
              ve_acc:         { axes: [] }
              acc:            { axes: [] }
              shift_reuse_acc:{ axes: [] }
              feed_reuse:     { axes: [] }
              feed_reuse_acc: { axes: [] }
              mac_rows:                       # output-col dim o  -> input/MAC rows
                axes:
                  - tag: { LabelStride: { label: {inner: "o"}, stride: 1 } }
                    size: { Var: O }
                    is_input_tile: true
                    is_filter_tile: false
                    is_contraction: false
              dpe_reg_tile:   { axes: [] }
              dpe_elementwise:{ axes: [] }
              lat:            { axes: [] }
              ve_reduce_labels: []
            sparsity: None
```

Open risks to validate when wiring this into the DSL (next step, not done here):
- The DSL's `SymTacticKernel` high-level path may expect the **22-field
  `TuContraction`** *without* the `LowLevel`-only `mac_rows`/`dpe_*` split — i.e. the
  hand-written op may only set `slice`/`gat`(contraction) and let the lowerer assign
  `mac_rows`/`acc`. Try the minimal `gat`(contraction)+`slice`+`mac_rows` first; if the
  parser rejects `mac_rows` at the high level, move `o` into `acc`/`outer_acc`.
- bf16: `output_element_type: Bfloat16` likely required (compiler always uses bf16 for
  DPE); an f32 DPE op may be unsupported, which would also explain "incompatible
  sequences" if we left it Float32.
- exact stride values for `Var`-sized axes come from the runtime layout, not the YAML
  (compiler used stride 1 for the contiguous mm8; for tiled cases strides encode the
  tile geometry).
