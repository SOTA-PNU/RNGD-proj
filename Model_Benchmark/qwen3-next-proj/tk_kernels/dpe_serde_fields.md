# RECON-B: `OperatorTacticEinsumByDpe` / `TuContraction` / `TuContractionAxis` serde schema

Goal: author the FAST DPE matmul YAML field-by-field. Source of truth =
`native_torch.cpython-312-x86_64-linux-gnu.so` (serde field arrays recovered from
`.data.rel.ro` via `.rela.dyn` RELATIVE relocations) **cross-validated against a real
lowered DPE matmul** (CBOR DFG dump decoded from
`tk_kernels/_dpe_dfg/mm8__Est_Misc.dfg.txt`, op `LowLevelTuContraction`).

`.so` = `/home/jun/furiosa/lib/python3.12/site-packages/furiosa/native_torch.cpython-312-x86_64-linux-gnu.so`
(BuildID 2333d91d..., npu-tools git checkout `3f23a71`).

---

## 0. CRITICAL: there are TWO "EinsumByDpe" things — don't confuse them

1. **High-level `kind: EinsumByDpe`** — an enum value of `SymTacticKernelInner.kind`.
   This is what the *current* hand-authored YAML uses (`tk_kernels/dn_linear_dpe.yaml`):
   the same `reads / ein_ops / vector_ops / write` body as the VE kernel, only the
   `kind:` string is swapped from `EinsumByVe` to `EinsumByDpe`. It PARSES but FAILS at
   execution — the SymTacticKernel body does not carry the DPE-specific `contraction`
   roles, so the lowerer cannot build the systolic schedule → "incompatible sequences".

2. **`OperatorTacticEinsumByDpe`** (serde struct, this report) — the **LOWERED**
   operator-tactic. In a real compile it is produced by the lowerer, serialized into the
   DFG as operator option `LowLevelTuContraction` (pprint name) / the serde struct
   `OperatorTacticEinsumByDpe`. Its body is `input/filter/output/contraction/...` with
   explicit per-axis MAC roles. This is the struct whose fields are below.

There are actually **two distinct serde field arrays** in the binary — verified by reading
the canonical de-duplicated rodata blob (file off 10945454) AND the reloc-reconstructed
FIELDS arrays:

**(A) `OperatorTacticEinsumByDpe` — the SERDE struct (12 fields)** — canonical rodata blob
@ off 10945454 reads literally:
```
input  filter  output_element_type  contraction  schedule_config  swap_inputs
input_paddings  input_slidings  acc_major_mode  reduce_mode  separate_vector_ops
force_mask_to_input
```
This is the struct named `struct OperatorTacticEinsumByDpe` in the deserializer — **NO
`vrfs`, NO `output`**. This is the one you write in a `#naive_yaml` operator option if you
hand-author the operator-tactic.

**(B) `LowLevelTuContraction` — the LOWERED runtime op (14 fields)** — reloc FIELDS array @
`.data.rel.ro` 0x6317128:
```
input  filter  vrfs  output  output_element_type  contraction  schedule_config
swap_inputs  input_paddings  input_slidings  acc_major_mode  reduce_mode
separate_vector_ops  force_mask_to_input
```
This is what a real compile emits into the DFG (op name `LowLevelTuContraction`); it adds
`vrfs` + explicit `output` tensor-id. The CBOR ground truth in §2/§4 is this (B) form.

The two share the same `contraction: TuContraction` (§2), `acc_major_mode`, `reduce_mode`,
etc. The table in §1 below lists the **(B)** 14-field form (matches the decoded matmul); for
the **(A)** serde struct drop rows `vrfs` and `output`.

---

## 1. `OperatorTacticEinsumByDpe` — fields, order, types

| # | field | type | matmul value | notes |
|---|-------|------|--------------|-------|
| 0 | `input`      | tensor id (u32)            | `0` (= A `[i,k]`)        | the activation / row operand |
| 1 | `filter`     | tensor id (u32)            | `1` (= B `[k,j]`)        | the weight / column operand |
| 2 | `vrfs`       | list                       | `[]`                     | extra VRF inputs (none for plain matmul) |
| 3 | `output`     | tensor id (u32)            | `2` (= C `[i,j]`)        |  |
| 4 | `output_element_type` | EnumElementType   | `Float32`                | acc dtype out; DPE acc is fp32 |
| 5 | `contraction`| `TuContraction` (22-elem)  | see §2                   | the per-axis MAC schedule |
| 6 | `schedule_config` | Option / struct       | `None`                   | optional scheduler hint |
| 7 | `swap_inputs`| bool                       | `false`                  | true ⇒ feed filter as rows |
| 8 | `input_paddings`  | list<[u32,u32]>       | `[]`                     | per-axis pad of `input` |
| 9 | `input_slidings`  | list                  | `[]`                     | conv-style sliding windows |
| 10| `acc_major_mode`  | `AccMajorMode` enum   | `ColMajor`               | see §3 (RowMajor/ColMajor) |
| 11| `reduce_mode`     | `ReduceMode` enum     | `Add`                    | **DPE: only Add or Max** (§3) |
| 12| `separate_vector_ops` | bool              | `false`                  | split VE ops off DPE |
| 13| `force_mask_to_input` | bool              | `false`                  | attention masking flag |

---

## 2. `TuContraction` — 22 fields (serde "struct TuContraction with 22 elements")

VERIFIED against real lowered matmul (`inner` of `LowLevelTuContraction`, mm8 = 8×8×8).
It is a **named-field map of 22 keys** (serde derive; the "with 22 elements" string is the
seq-visitor message). Each field after the two element-types is an **`Axes` struct**:
`{ axes: [TuContractionAxis, ...] }`. An empty group is `{axes: []}`.

| # | field | type | mm8 value | role |
|---|-------|------|-----------|------|
| 0 | `dpe_element_type` | EnumElementType | `Bfloat16` | DPE MAC operand dtype (matmul body runs bf16; output acc is fp32) |
| 1 | `trf_element_type` | EnumElementType | `Bfloat16` | tensor-register-file feed dtype |
| 2 | `split`            | Axes | `{axes:[]}` | inter-chip split axes |
| 3 | `chip`             | Axes | `{axes:[]}` | per-chip axes |
| 4 | `cluster`          | Axes | `{axes:[]}` | per-cluster axes |
| 5 | `slice_dummy`      | Axes | `{axes:[]}` | dummy slice axes |
| 6 | `pe`               | Axes | `{axes:[]}` | per-PE axes |
| 7 | `slice`            | Axes | `{axes:[ j-axis, i-axis ]}` | **output (non-reduced) axes** held in slice; mm8: labels "1","0" size 8, contr=false |
| 8 | `ve_gat`           | Axes | `{axes:[]}` | gather-on-VE axes |
| 9 | `gat`              | Axes | `{axes:[]}` | gather axes |
| 10| `segment`          | Axes | `{axes:[]}` | segmented-reduce axes |
| 11| `outer_acc`        | Axes | `{axes:[]}` | outer accumulation axes |
| 12| `ve_acc`           | Axes | `{axes:[]}` | VE accumulation axes |
| 13| `acc`              | Axes | `{axes:[]}` (mm8) / k-axis (mm128) | inner accumulator axes (large-k split) |
| 14| `shift_reuse_acc`  | Axes | `{axes:[]}` | shift-reuse accumulator |
| 15| `feed_reuse`       | Axes | `{axes:[]}` | feed-buffer reuse axes |
| 16| `feed_reuse_acc`   | Axes | `{axes:[]}` | feed-reuse + acc |
| 17| `mac_rows`         | Axes | `{axes:[]}` (mm8) / i-axis (mm128) | rows fed into the MAC array (the *input/row* dim) |
| 18| `dpe_reg_tile`     | Axes | `{axes:[]}` (mm8) / j-axis (mm128) | DPE register tiling (the *filter/col* dim) |
| 19| `dpe_elementwise`  | Axes | `{axes:[]}` | elementwise-on-DPE axes |
| 20| `lat`              | Axes | `{axes:[ k-axis ]}` | **the CONTRACTION axis** lives here; mm8: label "2" size 8 `is_contraction: true` |
| 21| `ve_reduce_labels` | list<Label> | `[]` | labels reduced on VE instead of DPE |

### Axis-role mapping for C[i,j] = Σ_k A[i,k]·B[k,j]
From mm8 (single tile) and mm128 (tiled) decoded dumps:
- **k (contraction)** → `lat` (always, `is_contraction: true`); when k is large it *also*
  appears in `acc` (mm128: lat k=16 + acc k=4 ⇒ k split 16×4=64-ish).
- **i (input/row)**  → `mac_rows` (`is_input_tile: true`); for tiny mm8 it degenerates
  into `slice`.
- **j (filter/col)** → `dpe_reg_tile` (`is_filter_tile: true`); for tiny mm8 it
  degenerates into `slice`.
- For the trivial 8×8×8 case the compiler put both output dims in `slice` and the only
  populated reduce group is `lat`. The minimal correct hand-authored contraction therefore
  needs: `lat`=[k axis, is_contraction:true], plus the two output dims somewhere in
  `slice`/`mac_rows`/`dpe_reg_tile`, everything else `{axes:[]}`.

---

## 3. `TuContractionAxis` — 5 fields (serde "struct TuContractionAxis with 5 elements")

VERIFIED (every axis object in the decoded dump has exactly these 5 keys, in this order):

| # | field | type | matmul value | notes |
|---|-------|------|--------------|-------|
| 0 | `tag` | `AxisTag` enum (`LabelStride{label:{inner:"k"}, stride:N}` \| `Broadcast{size}` \| `Dummy` \| `ModSkewedLabelStride` \| `LabelSymbolicStride` \| `SparseLabelStride`) | `LabelStride{label:{inner:"2"},stride:1}` | the axis identity |
| 1 | `size` | int (u64) or `Var: NAME` for symbolic | `8` (or `Var: K`) | extent |
| 2 | `is_input_tile`   | bool | `false`/`true` | belongs to `input` operand tiling |
| 3 | `is_filter_tile`  | bool | `false`/`true` | belongs to `filter` operand tiling |
| 4 | `is_contraction`  | bool | `true` for the k axis, else `false` | the summed dim |

(The earlier flat-string dump that showed `tag,size,is_input_tile,is_filter_tile,
is_contraction,symbol,valid_length,stride,inner` was concatenating TuContractionAxis with
the neighbouring `LabelSymbolicStride`/`AxisIndex` structs — the true axis struct is the
5 keys above, confirmed by the live CBOR.)

---

## 4. Enums

### `ReduceMode` (field `reduce_mode`)  — rodata @ off 12762482, reloc array @ 0x630de50
Variants in order: **`Add`, `Max`, `Mul`** (a 4th `GenericReduce` exists in the parent
`GenericEinOps`). Hard constraint string in the binary:

> `DPE allows only add and max for reducing`

⇒ for `EinsumByDpe`, `reduce_mode` MUST be `Add` (matmul) or `Max` (max-reduce). `Mul`/
`GenericReduce` are rejected at lowering. For C=ΣA·B use **`Add`**.

### `AccMajorMode` (field `acc_major_mode`) — rodata "AccMajorMode" @ off 10839955; reloc array @ 0x632dc90
Variants: **`RowMajor`, `ColMajor`**. The real lowered matmul uses **`ColMajor`** (output
written column-major from the systolic array). Use `ColMajor` for the plain matmul.
(`{RowMajor,ColMajor,None,Reshape}` in that reloc run — the trailing `None,Reshape` belong
to the adjacent `ReshapeEinsumMode`; AccMajorMode itself = `RowMajor` | `ColMajor`.)

### Head/Tail contraction & rounding (seen alongside, for context)
- `HeadContractionMode`/`TailContractionMode` variants: `ContractionByAdd`,
  `ContractionByMax` (reloc @ 0x63f68c8).
- `FxpShiftRoundingMode`: `AwayFromZero`, `ToNearestEven`, `KeepSticky`.

### Element types (field `output_element_type`, `dpe_element_type`, `trf_element_type`)
`EnumElementType` variants (from `.so`): `Float32, Float16, Float8e4m3, Float8e5m2,
Float4e2m1, Float4e2m1Decoded, Float64, RawInt4, RawInt5, RawInt8, RawInt9, Uint8, Int32,
Int4Decoded, Double, Int4, Bfloat16`. For fp32 matmul: `output_element_type: Float32`;
DPE MAC body runs `Bfloat16` (dpe/trf element types) — i.e. inputs are cast to bf16, the
accumulator/output is Float32. (This matches the pprint: inputs `LowLevelVePassOps` cast
f32→bf16 before the contraction, output is f32.)

---

## 5. Validation / error strings (DPE / contraction / sequence)

Recovered from `strings` on the `.so` (each verified present):

| string | meaning / when it fires |
|--------|--------------------------|
| `DPE allows only add and max for reducing` | `reduce_mode` ∉ {Add,Max} |
| `incompatible axis sequence: ` | axis label order between input/filter/contraction does not line up — **this is the "incompatible sequences" failure when `contraction` is unpopulated** |
| `reduced tags should exist` | the contraction (`lat`) axis set is empty / the reduce label is missing from the operands |
| `already reduced` | a label marked contraction appears already reduced |
| `unsupported element type for reduce` | reduce dtype not supported |
| `only f32 or i32 inputs are supported (current: …)` | DPE/VE reduce input dtype gate |
| `Invalid ReduceMode: ` | serde-time bad `reduce_mode` value |
| `output_element_type is invalid` | serde/validate bad `output_element_type` |
| `acc_major_mode` / `reduce_mode` are serde-validated enums (bad value ⇒ deser error) | |
| `* is not contained in TuContraction table.` | a referenced axis label is absent from the contraction's axis groups |
| `unsupported input types for LowLevelTuContraction (input: …)` | operand dtype not accepted by the lowered contraction |
| `EinsumByVe should have broadcast` | (VE path) reminder that read0 must be the broadcast read — load-bearing for the VE yaml, not DPE |

The root cause of the current "incompatible sequences" on `kind: EinsumByDpe`: the
high-level SymTacticKernel body carries no `contraction: TuContraction`, so when the
lowerer tries to assign axis roles it finds no `lat`/`is_contraction` axis ⇒ `reduced tags
should exist` / `incompatible axis sequence`. The fix is to emit the **lowered**
`OperatorTacticEinsumByDpe`/`LowLevelTuContraction` form with `contraction.lat = [k axis,
is_contraction:true]` populated, OR to make the compiler lower it for us (export+
`compiler.lower` with `tactic_sorting_policy=ByEstimation`) and reuse the produced tactic.

---

## 6. YAML value map for the plain matmul  C[i,j] = Σ_k A[i,k]·B[k,j]

(labels: i="0"? — in the decoded dump the compiler used "1"=i(tokens), "0"=j(out), "2"=k.
For a hand-authored kernel pick any 3 distinct labels, e.g. i, j, k.)

```yaml
OperatorTacticEinsumByDpe:        # operator option (lowered form)
  input: 0                        # A[i,k]
  filter: 1                       # B[k,j]
  vrfs: []
  output: 2                       # C[i,j]
  output_element_type: Float32
  contraction:                    # TuContraction (22 named groups)
    dpe_element_type: Bfloat16
    trf_element_type: Bfloat16
    split:           {axes: []}
    chip:            {axes: []}
    cluster:         {axes: []}
    slice_dummy:     {axes: []}
    pe:              {axes: []}
    slice:                        # the two OUTPUT (non-reduced) dims
      axes:
        - tag: {LabelStride: {label: {inner: "i"}, stride: 1}}
          size: {Var: I}
          is_input_tile: true
          is_filter_tile: false
          is_contraction: false
        - tag: {LabelStride: {label: {inner: "j"}, stride: 1}}
          size: {Var: J}
          is_input_tile: false
          is_filter_tile: true
          is_contraction: false
    ve_gat:          {axes: []}
    gat:             {axes: []}
    segment:         {axes: []}
    outer_acc:       {axes: []}
    ve_acc:          {axes: []}
    acc:             {axes: []}          # populate if K must be split across MAC depth
    shift_reuse_acc: {axes: []}
    feed_reuse:      {axes: []}
    feed_reuse_acc:  {axes: []}
    mac_rows:        {axes: []}          # (large case: i axis goes here, is_input_tile:true)
    dpe_reg_tile:    {axes: []}          # (large case: j axis goes here, is_filter_tile:true)
    dpe_elementwise: {axes: []}
    lat:                                  # THE CONTRACTION AXIS k
      axes:
        - tag: {LabelStride: {label: {inner: "k"}, stride: 1}}
          size: {Var: K}
          is_input_tile: false
          is_filter_tile: false
          is_contraction: true
    ve_reduce_labels: []
  schedule_config: None
  swap_inputs: false
  input_paddings: []
  input_slidings: []
  acc_major_mode: ColMajor
  reduce_mode: Add
  separate_vector_ops: false
  force_mask_to_input: false
```

Caveats before trusting this hand-author:
- The compiler-produced kernels put i in `mac_rows` and j in `dpe_reg_tile` only once the
  tile is big enough; for tiny shapes both output dims sat in `slice`. The minimum the
  lowerer demands is a non-empty `lat` with `is_contraction:true` and the output dims
  present somewhere — verify by round-trip (serialize the above, lower, diff vs a
  compiler-produced tactic).
- `size:` accepts either a literal int (as in the dumped 8) or `{Var: NAME}` for the
  dynamic/symbolic shapes used by the existing VE yamls.
- The element-type split (bf16 MAC, f32 out) is what the real lower chose; a pure-f32 DPE
  path may not exist — the VE→bf16 cast is inserted before the contraction.

---

## 7. Method / provenance (for reproduction)
- Field arrays: parsed ELF `.rela.dyn` R_X86_64_RELATIVE relocs, walked `.data.rel.ro` as
  `(reloc_ptr→rodata_str, len)` pairs, rebuilt contiguous 16-byte-stride runs =
  serde-derived `FIELDS`/variant arrays. Script: `/tmp/parse_serde2.py`.
- Ground-truth values: `tk_kernels/_dpe_dfg/mm8__Est_Misc.dfg.txt` is a base64 CBOR DFG
  (8-byte LE length header, then `cbor2.loads(bytes[8:])`). Decoder:
  `/tmp/decode_dfg5.py` / `/tmp/decode_full.py`. The op with both `reduce_mode` and
  `acc_major_mode` keys is the `LowLevelTuContraction`.
- Reference VE matmul that runs on NPU: `tk_kernels/dn_linear.yaml`.
- Prior (failing) DPE attempt: `tk_kernels/dn_linear_dpe.yaml` (just flips `kind:` — wrong
  representation, see §0).
