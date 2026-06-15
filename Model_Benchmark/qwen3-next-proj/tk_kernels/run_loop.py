import os, sys, traceback
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
import torch  # first
from furiosa.native_torch.ir import Dfg
from furiosa.torch import debug

YAML = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/dn_loop.yaml"
dsl = open(YAML).read()
dfg = Dfg.parse(dsl)
print("PARSE OK; required_symbolic_params=", dfg.required_symbolic_params,
      "input_symbols=", dfg.input_symbols)

S = torch.zeros(4, dtype=torch.float32)
X = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=torch.float32)
print("inputs: S=", S.tolist(), " X=", X.tolist())

# Only bind REQUIRED symbolic params (input_symbols are auto-inferred from tensors).
sp = {}
for name in dfg.required_symbolic_params:
    try:
        sp[name] = int(name)
    except ValueError:
        sp[name] = 3
print("symbolic_params:", sp)

print("=== CPU DfgExecutor ===")
try:
    ex = debug.DfgExecutor(dfg)
    outs = ex.execute([S, X], symbolic_params=sp)
    print("CPU RUN OK")
    for i, o in enumerate(outs):
        print(f"  out[{i}] = {o.tolist()}  shape={tuple(o.shape)} dtype={o.dtype}")
    expected = [6.0, 6.0, 6.0, 6.0]
    if outs and outs[0].tolist() == expected:
        print("CORRECT: S accumulated to", expected)
    else:
        print("MISMATCH: expected", expected, "got", outs[0].tolist() if outs else None)
except Exception as e:
    print("CPU RUN FAIL:", type(e).__name__)
    print(traceback.format_exc())
