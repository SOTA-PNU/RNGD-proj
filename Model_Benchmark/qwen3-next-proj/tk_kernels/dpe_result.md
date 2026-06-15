# DPE matmul — does the systolic engine make matmul fast? (YES)

Payoff summary for the hand-authored `kind: EinsumByDpe` matmul
(`dn_linear_dpe.yaml`) vs the `kind: EinsumByVe` matmul (`dn_linear.yaml`),
for the SAME linear `y[t,o] = sum_i x[t,i]*W[o,i] == F.linear(x, W)`.

Run env (rngd:2):
```
PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj RNGD_DEV=rngd:2 \
  /home/jun/furiosa/bin/python bench_dpe_vs_ve.py
```
Date: 2026-06-11. Device: rngd:2 (the task-specified card; previously rngd:3 in RECON-C
because rngd:2 was EBUSY — both interchangeable, results consistent).

---

## 1. Does it RUN on the NPU? YES (pure NPU, no CPU fallback)

`dn_linear_dpe.yaml` compiles and executes on rngd:2 with `_dfg_inner == 0`
(zero CPU-fallback calls) across every shape tested. The systolic/DPE-MAC engine
does the whole matmul.

## 2. Validation vs torch F.linear (rngd:2)

| kernel | shape (T,I,O) | maxabs   | relmean | allclose@1e-2 | allclose@1e-3 | dfg |
|--------|---------------|----------|---------|---------------|---------------|-----|
| DPE    | 128,512,2048  | 1.30e-3  | 0.23%   | True          | False         | 0 (NPU) |
| DPE    | 128,256,128   | 8.26e-4  | 0.23%   | True          | True          | 0 (NPU) |
| DPE    | 256,2048,512  | 2.35e-3  | 0.23%   | True          | False         | 0 (NPU) |
| VE     | 128,512,2048  | 2.98e-7  | 0.00%   | True          | True          | 0 (NPU) |
| VE     | 128,256,128   | 2.68e-7  | 0.00%   | True          | True          | 0 (NPU) |
| VE     | 256,2048,512  | 5.36e-7  | 0.00%   | True          | True          | 0 (NPU) |

- DPE matches torch at **atol/rtol = 1e-2** (NOT 1e-3). The ~0.23% relmean / up to
  ~2.4e-3 maxabs is the expected DPE reduced-precision accumulation: the compiler
  feeds the systolic array in **bf16** (`dpe_element_type = trf_element_type =
  Bfloat16`, confirmed in the compiler dump, RECON-A). Not a bug — it is the price
  of the fast path.
- VE stays f32-exact (~1e-7) because it does MulF + LocalReduceAddF in float on the
  vector engine.

## 3. BENCHMARK — wall-clock, steady-state (2nd+ call post-compile), rngd:2

Realistic size: `x[128,2048] @ W[512,2048] -> [128,512]`, 20 timed iters after 3 warmups.
Each timed call is the full `cm(x_dev, W_dev) -> .to("cpu")` (host round-trip identical
for both kernels, so the comparison is apples-to-apples).

| kernel | mean (ms) | median (ms) | min (ms) |
|--------|-----------|-------------|----------|
| **DPE (EinsumByDpe)** | **2.895** | **2.889** | 2.576 |
| VE  (EinsumByVe)      | 5.661     | 5.649     | 5.441 |

### **SPEEDUP (VE / DPE): 1.96x (mean), 1.96x (median)** on rngd:2.

Why 1.96x here vs the 3.8x in RECON-C: this end-to-end timing includes the
host->device->host transfer that is the SAME for both kernels, which dilutes the
ratio. RECON-C's 3.8x (3.51 ms DPE vs 13.33 ms VE) timed device-resident execution
more directly. Both measurements agree on the conclusion: **DPE is materially
faster, and the gap widens as the matmul (not the transfer) dominates** — exactly
the regime that matters for the ~89s `[10,1,512,2048]` batched matmul that motivated
this work, where VE materializes the full `[.,o,i]` outer product and DPE does not.

## 4. Verdict

- DPE matmul **RUNS on pure NPU**, **matches torch at 1e-2**, and is **~2x faster
  end-to-end (≈3.8x compute-only)** than the VE matmul for the same op.
- Because `dn_linear` is used everywhere in the model, swapping the VE matmul for
  this DPE kernel is a model-wide fast-path — with the one caveat that callers must
  tolerate ~0.23% relmean (bf16 accumulation). For attention/QK and projection
  matmuls this is typically fine; for anything needing f32-exact reduction, keep VE.

---

## 5. How the working DPE kernel is built (the load-bearing delta vs VE)

The hand-authored DSL (`#naive_yaml` SymTacticKernel) does **NOT** require the full
22-field `TuContraction` block that the compiler emits in its lowered form (RECON-A).
At the high level, `kind: EinsumByDpe` is a tag on the same
`{reads, ein_ops, vector_ops, write}` inner struct as EinsumByVe; the contraction is
driven by `ein_ops`, and the lowerer assigns the slice/gat/mac_rows/acc tiling itself.
The minimal delta vs `dn_linear.yaml`:

1. `kind: EinsumByDpe` (was `EinsumByVe`).
2. `ein_ops` MUST be non-null and carry the contraction (was `ein_ops: ~`):
   ```yaml
   ein_ops:
     reduce:
       mode: Add                 # DPE allows only Add | Max
       input:                    # TensorLike of the PRE-reduce product [t,o,i]
         shape: { inner: [ t:Var T, o:Var O, i:Var I ] (LabelStride each) }
         element_type: Float32
       axes:
         - LabelStride: { label: {inner: "i"}, stride: 1 }   # contract over "i"
       source: ""
     mul_source: ""              # STRING, not null
   ```
3. `vector_ops` MUST have EXACTLY 1 input (the fused EinOps result — the two reads
   collapse to one post-contraction descriptor) and a single identity passthrough
   inst. The DSL has NO identity Unary op (PassF/Identity/Neg/Abs/Square all reject)
   and NO constant operand for Binary, so the only passthrough is a **Reduce over an
   EMPTY axes list**:
   ```yaml
   vector_ops:
     inputs: [0]
     insts:
       - def: 1
         expr:
           Reduce: { operator: LocalReduceAddF, operand: {Tensor: 0}, axes: {Tag: []} }
         source: ""
   ```
4. `reads` and `write` are UNCHANGED from the VE version.

Full error->fix derivation: `dpe_incremental_log.md` (11 iterations).
Compiler-dump struct (the lowered 22-field TuContraction, for reference): `dpe_struct_from_dump.md`.
Serde field map: `dpe_serde_fields.md`.

## Files
- `dn_linear_dpe.yaml`  — the working EinsumByDpe matmul kernel.
- `bench_dpe_vs_ve.py`  — validation + benchmark script (this report's numbers).
- `dn_linear.yaml`      — the EinsumByVe reference matmul (unchanged).
