import os, traceback
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
import torch
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule
import furiosa.torch.custom_ops.dfg as dfgmod

BASE = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/tk_kernels/"

# --- monkeypatch _dfg_inner to detect CPU-offline fallback ---
_orig = dfgmod._dfg_inner
STATE = {"called": False}
def _spy(*a, **k):
    STATE["called"] = True
    return _orig(*a, **k)
dfgmod._dfg_inner = _spy

def run(yaml_name, inputs, expected, label):
    STATE["called"] = False
    print(f"\n========== {label}: {yaml_name} ==========")
    dsl = open(BASE + yaml_name).read()
    m = TacticKernelModule(dsl)
    cm = torch.compile(m, backend=ft.backend)
    dev_inputs = [t.to('rngd:0') for t in inputs]
    out = cm(*dev_inputs)
    if isinstance(out, (tuple, list)):
        out = out[0]
    out_cpu = out.detach().to('cpu').float()
    print("NPU out shape", tuple(out_cpu.shape), "dtype", out_cpu.dtype)
    print("  got:", out_cpu.flatten().tolist())
    print("  exp:", expected.flatten().tolist())
    match = torch.allclose(out_cpu, expected.float(), atol=1e-4)
    maxerr = (out_cpu - expected.float()).abs().max().item()
    print("  MATCH:", match, " max_abs_err:", maxerr)
    print("  _dfg_inner called (CPU fallback)?:", STATE["called"])
    print("  -> NPU-executed (not _dfg_inner):", (not STATE["called"]))
    return match, (not STATE["called"]), out_cpu

results = {}

# ---- DECAY: Sout[k,v] = S[k,v] * decay ----
K, V = 2, 3
S = torch.arange(K*V, dtype=torch.float32).reshape(K, V) + 1.0   # 1..6
decay = 0.5
decay_full = torch.full((K, V), decay, dtype=torch.float32)
exp_decay = S * decay
try:
    m, npu, o = run("dn_decay.yaml", [S, decay_full], exp_decay, "DECAY")
    results["decay"] = (m, npu, o, exp_decay, S, decay)
except Exception as e:
    print("DECAY FAIL:", type(e).__name__); print(traceback.format_exc()); results["decay"] = None

# ---- DELTA: delta[v] = (v_vec - kv) * beta ----
V2 = 4
v_vec = torch.tensor([1.0, 2.0, 3.0, 4.0])
kv    = torch.tensor([0.5, 0.5, 1.0, 2.0])
beta  = 2.0
beta_full = torch.full((V2,), beta, dtype=torch.float32)
exp_delta = (v_vec - kv) * beta
try:
    m, npu, o = run("dn_delta.yaml", [v_vec, kv, beta_full], exp_delta, "DELTA")
    results["delta"] = (m, npu, o, exp_delta, v_vec, kv, beta)
except Exception as e:
    print("DELTA FAIL:", type(e).__name__); print(traceback.format_exc()); results["delta"] = None

print("\n==================== SUMMARY ====================")
for k, v in results.items():
    if v is None:
        print(k, ": ERROR")
    else:
        print(k, ": match=", v[0], " npu_executed=", v[1])
