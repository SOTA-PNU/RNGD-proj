"""UNROLL LIMIT harness.

For one T-unrolled DeltaNet prefill YAML, at d_k=d_v=128 (single head):
  - measure compile time (torch.compile + first call triggers furiosa lowering)
  - count graph ops (parse YAML next_operator_index)
  - prove NPU execution (monkeypatch furiosa.torch.custom_ops.dfg._dfg_inner;
    NPU run => call count == 0)
  - check correctness vs HF torch_recurrent_gated_delta_rule (allclose atol=1e-3)

Usage:  python unroll_limit_test.py <T> <yaml_path> [stable]
  stable (default on): draw g_t < 0 so decay=exp(g)<1 (a real forgetting gate),
    which keeps the recurrent state bounded. With unconstrained g in [-0.5,0.5]
    decay can exceed 1 and the *math itself* (CPU ref included) explodes
    geometrically (|S| ~1e16 by T=32), so allclose is meaningless there.
Prints a single RESULT json line so the caller can parse it.
"""
import sys, time, json, re
import torch
import furiosa.torch as ft
from furiosa.torch import TacticKernelModule
from transformers.models.qwen3_next.modeling_qwen3_next import (
    torch_recurrent_gated_delta_rule,
)

T = int(sys.argv[1])
YAML = sys.argv[2]
STABLE = (len(sys.argv) < 4) or (sys.argv[3].lower() not in ("0", "no", "false", "unstable"))
K = V = 128
torch.manual_seed(0)

# ---- per-timestep inputs, same layout as gen_unroll.py ----
#   t: q=1+5t, k=2+5t, v=3+5t, beta=4+5t, decay=5+5t ; S0=0
# YAML kernel pre-scales nothing: q must be PRE-SCALED by 1/sqrt(d_k);
# decay materialized to [K,V]; beta materialized to [V].
scale = 1.0 / (K ** 0.5)
q_list, k_list, v_list, beta_s_list, g_list = [], [], [], [], []
for t in range(T):
    q_list.append(torch.randn(K, dtype=torch.float32))
    k_list.append(torch.randn(K, dtype=torch.float32))
    v_list.append(torch.randn(V, dtype=torch.float32))
    beta_s_list.append(float(torch.rand(1).item()))
    if STABLE:
        # g_t < 0  ->  decay = exp(g) in (0,1): a genuine forgetting gate,
        # bounded recurrent state. This is how the real model behaves.
        g_list.append(float((-torch.rand(1)).item()))        # g_t in (-1,0]
    else:
        g_list.append(float((torch.rand(1) - 0.5).item()))   # g_t in [-0.5,0.5]

S0 = torch.zeros(K, V, dtype=torch.float32)
inputs = [S0]
for t in range(T):
    q_scaled = q_list[t] * scale  # pre-scale here (kernel does not)
    decay_s = float(torch.exp(torch.tensor(g_list[t])).item())
    inputs.append(q_scaled)
    inputs.append(k_list[t])
    inputs.append(v_list[t])
    inputs.append(torch.full((V,), beta_s_list[t], dtype=torch.float32))
    inputs.append(torch.full((K, V), decay_s, dtype=torch.float32))

# ---- HF reference ----
# torch_recurrent_gated_delta_rule expects (B,S,H,D) layout and internally
# transposes to (B,H,S,D) and pre-scales q by 1/sqrt(d_k).
B, H, S = 1, 1, T
query = torch.stack(q_list, 0).view(B, S, H, K).contiguous()
key   = torch.stack(k_list, 0).view(B, S, H, K).contiguous()
value = torch.stack(v_list, 0).view(B, S, H, V).contiguous()
g     = torch.tensor(g_list, dtype=torch.float32).view(B, S, H).contiguous()
beta  = torch.tensor(beta_s_list, dtype=torch.float32).view(B, S, H).contiguous()
core_out_ref, final_state_ref = torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state=None, output_final_state=True
)
# core_out_ref: (B,S,H,V) -> per-t out[V]
out_ref = [core_out_ref[0, t, 0] for t in range(T)]          # list of [V]
final_S_ref = final_state_ref[0, 0]                          # [K,V]

# ---- NPU exec spy ----
import furiosa.torch.custom_ops.dfg as dfgmod
calls = {"n": 0}
_orig = dfgmod._dfg_inner
def spy(*a, **kw):
    calls["n"] += 1
    return _orig(*a, **kw)
dfgmod._dfg_inner = spy

# ---- graph op count from YAML ----
yaml_txt = open(YAML).read()
m = re.search(r"next_operator_index:\s*(\d+)", yaml_txt)
n_ops = int(m.group(1)) if m else -1

# ---- compile + run, timed ----
t0 = time.time()
mod = TacticKernelModule(yaml_txt)
cm = torch.compile(mod, backend=ft.backend)
res = cm(*[t.to('rngd:0') for t in inputs])
compile_run_s = time.time() - t0

# outputs order: out_0..out_{T-1} then final S  (see gen_unroll outputs list)
outs = [res[i].to('cpu') for i in range(T)]
final_S = res[T].to('cpu')

# ---- correctness ----
worst_out = 0.0
all_out_ok = True
for t in range(T):
    err = (outs[t] - out_ref[t]).abs().max().item()
    worst_out = max(worst_out, err)
    if not torch.allclose(outs[t], out_ref[t], atol=1e-3):
        all_out_ok = False
S_err = (final_S - final_S_ref).abs().max().item()
S_ok = torch.allclose(final_S, final_S_ref, atol=1e-3)

result = dict(
    T=T,
    stable=STABLE,
    K=K, V=V,
    n_ops=n_ops,
    compile_run_s=round(compile_run_s, 2),
    dfg_inner_calls=calls["n"],
    npu_ran=(calls["n"] == 0),
    out_allclose=all_out_ok,
    out_maxerr=worst_out,
    final_S_allclose=S_ok,
    final_S_maxerr=S_err,
    PASS=bool(all_out_ok and S_ok and calls["n"] == 0),
)
print("RESULT " + json.dumps(result))
