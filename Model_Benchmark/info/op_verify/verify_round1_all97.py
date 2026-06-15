"""Empirically verify every op in furiosa.torch.db.SUPPORTED_ATEN_OPS.

For each op: build a minimal graph that uses it, export+decompose via the
production path (core_aten_decompositions), confirm the op survives into the
final graph, AOT-compile it (CompileModule.from_exported), then run on RNGD NPU
and compare against CPU eager.
"""
import os, sys, json, time, traceback
import torch
import furiosa.torch
from furiosa.torch import CompileModule
from torch._decomp import core_aten_decompositions

torch.manual_seed(0)
DEV_IDX = int(os.environ.get("RNGD_IDX", "3"))
TABLE = dict(core_aten_decompositions())  # production-faithful decomposition table

def M(fn):
    class _M(torch.nn.Module):
        def forward(self, *a):
            return fn(*a)
    return _M().eval()

# ---- input factories ----
def f(*shape):   return torch.randn(*shape)
def fp(*shape):  return torch.rand(*shape) + 0.5          # strictly positive (log/sqrt/rsqrt)
def ii(*shape, hi=4): return torch.randint(0, hi, shape, dtype=torch.int32)
def il(*shape, hi=4): return torch.randint(0, hi, shape, dtype=torch.int64)
def bb(*shape):  return torch.randint(0, 2, shape, dtype=torch.bool)

# ---- test registry: (target_op, callable, args) ----
T = []
def reg(op, fn, *args): T.append((op, fn, args))

# Math unary
reg("abs",        lambda x: torch.abs(x), f(4,5))
reg("cos",        lambda x: torch.cos(x), f(4,5))
reg("erf",        lambda x: torch.erf(x), f(4,5))
reg("exp",        lambda x: torch.exp(x), f(4,5))
reg("log",        lambda x: torch.log(x), fp(4,5))
reg("neg",        lambda x: torch.neg(x), f(4,5))
reg("reciprocal", lambda x: torch.reciprocal(x), fp(4,5))
reg("rsqrt",      lambda x: torch.rsqrt(x), fp(4,5))
reg("sin",        lambda x: torch.sin(x), f(4,5))
reg("sqrt",       lambda x: torch.sqrt(x), fp(4,5))
reg("isnan",      lambda x: torch.isnan(x), f(4,5))

# Arithmetic binary
reg("add.Scalar", lambda x: x + 1.5, f(4,5))
reg("add.Tensor", lambda x,y: x + y, f(4,5), f(4,5))
reg("sub.Scalar", lambda x: x - 1.5, f(4,5))
reg("sub.Tensor", lambda x,y: x - y, f(4,5), f(4,5))
reg("mul.Scalar", lambda x: x * 2.0, f(4,5))
reg("mul.Tensor", lambda x,y: x * y, f(4,5), f(4,5))
reg("div.Scalar", lambda x: x / 2.0, f(4,5))
reg("div.Tensor", lambda x,y: x / y, f(4,5), fp(4,5))
reg("pow.Tensor_Scalar", lambda x: torch.pow(x, 2.5), fp(4,5))
reg("clamp",      lambda x: torch.clamp(x, -0.5, 0.5), f(4,5))

# Conv / Matmul
reg("convolution", lambda x,w,b: torch.nn.functional.conv2d(x, w, b, stride=1, padding=1), f(1,3,16,16), f(8,3,3,3), f(8))
reg("mm",  lambda a,b: torch.mm(a,b), f(4,6), f(6,5))
reg("bmm", lambda a,b: torch.bmm(a,b), f(2,4,6), f(2,6,5))

# Activation
reg("relu",         lambda x: torch.relu(x), f(4,5))
reg("leaky_relu",   lambda x: torch.nn.functional.leaky_relu(x, 0.1), f(4,5))
reg("sigmoid",      lambda x: torch.sigmoid(x), f(4,5))
reg("tanh",         lambda x: torch.tanh(x), f(4,5))
reg("_softmax",     lambda x: torch.softmax(x, -1), f(4,5))
reg("_log_softmax", lambda x: torch.log_softmax(x, -1), f(4,5))

# Comparison
reg("eq.Scalar", lambda x: x == 1.0, ii(4,5))
reg("eq.Tensor", lambda x,y: x == y, ii(4,5), ii(4,5))
reg("ne.Scalar", lambda x: x != 1.0, ii(4,5))
reg("ne.Tensor", lambda x,y: x != y, ii(4,5), ii(4,5))
reg("lt.Scalar", lambda x: x < 1.0, f(4,5))
reg("lt.Tensor", lambda x,y: x < y, f(4,5), f(4,5))
reg("le.Scalar", lambda x: x <= 1.0, f(4,5))
reg("le.Tensor", lambda x,y: x <= y, f(4,5), f(4,5))
reg("gt.Scalar", lambda x: x > 1.0, f(4,5))
reg("gt.Tensor", lambda x,y: x > y, f(4,5), f(4,5))
reg("ge.Scalar", lambda x: x >= 1.0, f(4,5))
reg("ge.Tensor", lambda x,y: x >= y, f(4,5), f(4,5))
reg("maximum",   lambda x,y: torch.maximum(x,y), f(4,5), f(4,5))
reg("minimum",   lambda x,y: torch.minimum(x,y), f(4,5), f(4,5))

# Logical
reg("logical_and", lambda a,b: torch.logical_and(a,b), bb(4,5), bb(4,5))
reg("logical_not", lambda a: torch.logical_not(a), bb(4,5))
reg("logical_xor", lambda a,b: torch.logical_xor(a,b), bb(4,5), bb(4,5))

# Bitwise
reg("bitwise_and.Scalar", lambda x: torch.bitwise_and(x, 3), ii(4,5))
reg("bitwise_and.Tensor", lambda x,y: torch.bitwise_and(x,y), ii(4,5), ii(4,5))
reg("bitwise_or.Scalar",  lambda x: torch.bitwise_or(x, 3), ii(4,5))
reg("bitwise_or.Tensor",  lambda x,y: torch.bitwise_or(x,y), ii(4,5), ii(4,5))
reg("bitwise_xor.Scalar", lambda x: torch.bitwise_xor(x, 3), ii(4,5))
reg("bitwise_xor.Tensor", lambda x,y: torch.bitwise_xor(x,y), ii(4,5), ii(4,5))

# Reduction
reg("sum.dim_IntList", lambda x: x.sum(dim=[1]), f(4,5))
reg("mean.dim",        lambda x: x.mean(dim=1), f(4,5))
reg("amax",            lambda x: torch.amax(x, dim=1), f(4,5))
reg("max.dim",         lambda x: torch.max(x, dim=1), f(4,5))
reg("argmax",          lambda x: torch.argmax(x, dim=1), f(4,5))
reg("any.dim",         lambda x: torch.any(x, dim=1), bb(4,5))
reg("cumsum",          lambda x: torch.cumsum(x, dim=1), f(4,5))
reg("var_mean.correction", lambda x: torch.var_mean(x, dim=1, correction=1), f(4,5))
reg("topk",            lambda x: torch.topk(x, 2, dim=1), f(4,5))

# Pooling
reg("avg_pool2d",                lambda x: torch.nn.functional.avg_pool2d(x, 2), f(1,3,16,16))
reg("max_pool2d_with_indices",   lambda x: torch.nn.functional.max_pool2d(x, 2, return_indices=True), f(1,3,16,16))
reg("_adaptive_avg_pool2d",      lambda x: torch.nn.functional.adaptive_avg_pool2d(x, 1), f(1,3,16,16))

# Shape / View
reg("view",              lambda x: x.view(2,10), f(4,5))
reg("view_copy",         lambda x: x.view(2,10), f(4,5))      # may appear via functionalization
reg("permute",           lambda x: x.permute(1,0), f(4,5))
reg("permute_copy",      lambda x: x.permute(1,0), f(4,5))
reg("transpose_copy.int",lambda x: x.transpose(0,1), f(4,5))
reg("t_copy",            lambda x: x.t(), f(4,5))
reg("squeeze.dim",       lambda x: x.squeeze(1), f(4,1,5))
reg("squeeze.dims",      lambda x: x.squeeze((1,2)), f(4,1,1,5))
reg("squeeze_copy.dim",  lambda x: x.squeeze(1), f(4,1,5))
reg("unsqueeze",         lambda x: x.unsqueeze(1), f(4,5))
reg("unsqueeze_copy",    lambda x: x.unsqueeze(1), f(4,5))
reg("expand",            lambda x: x.expand(3,4,5), f(1,4,5))
reg("expand_copy",       lambda x: x.expand(3,4,5), f(1,4,5))
reg("cat",               lambda a,b: torch.cat([a,b], dim=1), f(4,5), f(4,3))
reg("slice.Tensor",      lambda x: x[:, 1:4], f(4,5))
reg("slice_scatter",     lambda x,y: torch.slice_scatter(x, y, dim=1, start=1, end=4), f(4,5), f(4,3))

# Split
reg("split_with_sizes",      lambda x: torch.split(x, [2,3], dim=1), f(4,5))
reg("split_with_sizes_copy", lambda x: torch.split(x, [2,3], dim=1), f(4,5))

# Copy / Clone
reg("clone",   lambda x: torch.clone(x), f(4,5))
reg("copy",    lambda x,y: x.copy_(y), f(4,5), f(4,5))   # in-place -> functionalized
reg("copy_",   lambda x,y: x.copy_(y), f(4,5), f(4,5))
reg("_to_copy",lambda x: x.to(torch.float64).to(torch.float32), f(4,5))

# Indexing
reg("index.Tensor",  lambda x,idx: x[idx], f(6,5), il(3, hi=6))
reg("index_put",     lambda x,idx,v: x.index_put((idx,), v), f(6,5), il(3, hi=6), f(3,5))
reg("index_select",  lambda x,idx: torch.index_select(x, 0, idx), f(6,5), il(3, hi=6))
reg("gather",        lambda x,idx: torch.gather(x, 1, idx), f(4,5), il(4,2, hi=5))
reg("scatter.src",   lambda x,idx,src: x.scatter(1, idx, src), f(4,5), il(4,2, hi=5), f(4,2))

# Conditional
reg("where.self", lambda c,x,y: torch.where(c, x, y), bb(4,5), f(4,5), f(4,5))

# Creation
reg("full",      lambda x: torch.full((4,5), 3.0) + x, f(4,5))
reg("full_like", lambda x: torch.full_like(x, 2.0), f(4,5))
reg("fill.Scalar", lambda x: x.clone().fill_(7.0), f(4,5))

# Padding
reg("constant_pad_nd", lambda x: torch.nn.functional.pad(x, (1,2), value=0.0), f(4,5))


def aten_norm(target):
    s = str(target)
    if s.startswith("aten."):
        s = s[5:]
    return s

def covers(graph_ops, op):
    for g in graph_ops:
        if g == op or g == op + ".default":
            return True
        if g.endswith(".default") and g[:-8] == op:
            return True
    return False

def flatten(o):
    if isinstance(o, (tuple, list)):
        out = []
        for e in o: out += flatten(e)
        return out
    return [o]

def cmp(a, b):
    """Return (ok, max_err) comparing CPU ref vs NPU out (flattened)."""
    fa, fb = flatten(a), flatten(b)
    if len(fa) != len(fb):
        return False, float("nan")
    worst = 0.0
    for x, y in zip(fa, fb):
        x = x.to("cpu"); y = y.to("cpu")
        if x.shape != y.shape:
            return False, float("nan")
        if x.dtype.is_floating_point:
            d = (x.float() - y.float()).abs().max().item()
            worst = max(worst, d)
            if d > 1e-3:
                return False, worst
        else:
            if not torch.equal(x, y):
                return False, float("inf")
    return True, worst

dev = torch.device("rngd", DEV_IDX)
results = []
for op, fn, args in T:
    rec = {"op": op, "graph_ops": [], "present": False,
           "compile": None, "run": None, "max_err": None, "error": ""}
    try:
        m = M(fn)
        with torch.no_grad():
            ref = m(*[a.clone() for a in args])
            ep = torch.export.export(m, tuple(a.clone() for a in args)).run_decompositions(TABLE)
        gops = sorted({aten_norm(n.target) for n in ep.graph.nodes if n.op == "call_function"})
        rec["graph_ops"] = gops
        rec["present"] = covers(gops, op)
        try:
            cm = CompileModule.from_exported(ep)
            rec["compile"] = "OK"
        except Exception as e:
            rec["compile"] = "FAIL"
            rec["error"] = repr(e)[:400]
            results.append(rec);
            print(f"{op:28s} present={rec['present']!s:5s} compile=FAIL  {rec['error'][:90]}")
            continue
        try:
            cm.to(dev)
            with torch.no_grad():
                out = cm(*[a.to(dev) for a in args], device=dev)
            ok, err = cmp(ref, out)
            rec["run"] = "OK" if ok else "MISMATCH"
            rec["max_err"] = err
        except Exception as e:
            rec["run"] = "FAIL"
            rec["error"] = repr(e)[:400]
    except Exception as e:
        rec["compile"] = "EXPORT_FAIL"
        rec["error"] = repr(e)[:400]
    results.append(rec)
    print(f"{op:28s} present={rec['present']!s:5s} compile={rec['compile']!s:5s} run={rec['run']!s:8s} err={rec['max_err']}  {rec['error'][:70]}")

json.dump(results, open("/home/jun/.claude/jobs/220196a8/tmp/results.json","w"), indent=1)
# summary
npass = sum(1 for r in results if r["run"]=="OK")
print("\n=== SUMMARY ===")
print(f"total tests: {len(results)}")
print(f"run OK     : {npass}")
print(f"present&runOK: {sum(1 for r in results if r['present'] and r['run']=='OK')}")
print(f"NOT present in graph: {[r['op'] for r in results if not r['present']]}")
print(f"compile FAIL: {[r['op'] for r in results if r['compile'] in ('FAIL','EXPORT_FAIL')]}")
print(f"run FAIL/MISMATCH: {[r['op'] for r in results if r['run'] in ('FAIL','MISMATCH')]}")
