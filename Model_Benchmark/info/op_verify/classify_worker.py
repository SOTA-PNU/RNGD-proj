"""op 1개를 측정해 분류 라벨을 JSON 한 줄로 출력. subprocess 격리용(crash는 runner가 판정).
두 경로 측정:
  - AOT: export → run_decompositions → CompileModule.from_exported → NPU 실행
  - eager dispatch: RngdTensor(x.to(rngd)) + op → coverage 의 run_on_rngd / run_on_cpu
분류(primary): trace_unsupported / npu / host / compile_fail  (crash 는 runner)."""
import os, sys, json, torch, furiosa.torch
from furiosa.torch import CompileModule
from furiosa.torch.coverage import RNGDCoverageTrace
from torch._decomp import core_aten_decompositions
from torch.utils import _pytree as pytree
torch.manual_seed(0)
TABLE = dict(core_aten_decompositions())
DEV = torch.device("rngd", int(os.environ.get("RNGD_IDX", "0")))

def f(*s): return torch.randn(*s)
def fp(*s): return torch.rand(*s) + 0.5
def ii(*s, hi=4): return torch.randint(0, hi, s, dtype=torch.int32)
def il(*s, hi=4): return torch.randint(0, hi, s, dtype=torch.int64)
def bb(*s): return torch.randint(0, 2, s, dtype=torch.bool)
T = []
def reg(op, fn, *args): T.append((op, fn, args))

reg("abs", lambda x: torch.abs(x), f(4,5)); reg("cos", lambda x: torch.cos(x), f(4,5))
reg("erf", lambda x: torch.erf(x), f(4,5)); reg("exp", lambda x: torch.exp(x), f(4,5))
reg("log", lambda x: torch.log(x), fp(4,5)); reg("neg", lambda x: torch.neg(x), f(4,5))
reg("reciprocal", lambda x: torch.reciprocal(x), fp(4,5)); reg("rsqrt", lambda x: torch.rsqrt(x), fp(4,5))
reg("sin", lambda x: torch.sin(x), f(4,5)); reg("sqrt", lambda x: torch.sqrt(x), fp(4,5))
reg("isnan", lambda x: torch.isnan(x), f(4,5))
reg("add.Scalar", lambda x: x + 1.5, f(4,5)); reg("add.Tensor", lambda x,y: x + y, f(4,5), f(4,5))
reg("sub.Scalar", lambda x: x - 1.5, f(4,5)); reg("sub.Tensor", lambda x,y: x - y, f(4,5), f(4,5))
reg("mul.Scalar", lambda x: x * 2.0, f(4,5)); reg("mul.Tensor", lambda x,y: x * y, f(4,5), f(4,5))
reg("div.Scalar", lambda x: x / 2.0, f(4,5)); reg("div.Tensor", lambda x,y: x / y, f(4,5), fp(4,5))
reg("pow.Tensor_Scalar", lambda x: torch.pow(x, 2.5), fp(4,5)); reg("clamp", lambda x: torch.clamp(x, -0.5, 0.5), f(4,5))
reg("convolution", lambda x,w,b: torch.nn.functional.conv2d(x, w, b, stride=1, padding=1), f(1,3,16,16), f(8,3,3,3), f(8))
reg("mm", lambda a,b: torch.mm(a,b), f(4,6), f(6,5)); reg("bmm", lambda a,b: torch.bmm(a,b), f(2,4,6), f(2,6,5))
reg("relu", lambda x: torch.relu(x), f(4,5)); reg("leaky_relu", lambda x: torch.nn.functional.leaky_relu(x, 0.1), f(4,5))
reg("sigmoid", lambda x: torch.sigmoid(x), f(4,5)); reg("tanh", lambda x: torch.tanh(x), f(4,5))
reg("_softmax", lambda x: torch.softmax(x, -1), f(4,5)); reg("_log_softmax", lambda x: torch.log_softmax(x, -1), f(4,5))
reg("eq.Scalar", lambda x: x == 1.0, ii(4,5)); reg("eq.Tensor", lambda x,y: x == y, ii(4,5), ii(4,5))
reg("ne.Scalar", lambda x: x != 1.0, ii(4,5)); reg("ne.Tensor", lambda x,y: x != y, ii(4,5), ii(4,5))
reg("lt.Scalar", lambda x: x < 1.0, f(4,5)); reg("lt.Tensor", lambda x,y: x < y, f(4,5), f(4,5))
reg("le.Scalar", lambda x: x <= 1.0, f(4,5)); reg("le.Tensor", lambda x,y: x <= y, f(4,5), f(4,5))
reg("gt.Scalar", lambda x: x > 1.0, f(4,5)); reg("gt.Tensor", lambda x,y: x > y, f(4,5), f(4,5))
reg("ge.Scalar", lambda x: x >= 1.0, f(4,5)); reg("ge.Tensor", lambda x,y: x >= y, f(4,5), f(4,5))
reg("maximum", lambda x,y: torch.maximum(x,y), f(4,5), f(4,5)); reg("minimum", lambda x,y: torch.minimum(x,y), f(4,5), f(4,5))
reg("logical_and", lambda a,b: torch.logical_and(a,b), bb(4,5), bb(4,5))
reg("logical_not", lambda a: torch.logical_not(a), bb(4,5))
reg("logical_xor", lambda a,b: torch.logical_xor(a,b), bb(4,5), bb(4,5))
reg("bitwise_and.Scalar", lambda x: torch.bitwise_and(x, 3), ii(4,5)); reg("bitwise_and.Tensor", lambda x,y: torch.bitwise_and(x,y), ii(4,5), ii(4,5))
reg("bitwise_or.Scalar", lambda x: torch.bitwise_or(x, 3), ii(4,5)); reg("bitwise_or.Tensor", lambda x,y: torch.bitwise_or(x,y), ii(4,5), ii(4,5))
reg("bitwise_xor.Scalar", lambda x: torch.bitwise_xor(x, 3), ii(4,5)); reg("bitwise_xor.Tensor", lambda x,y: torch.bitwise_xor(x,y), ii(4,5), ii(4,5))
reg("sum.dim_IntList", lambda x: x.sum(dim=[1]), f(4,5)); reg("mean.dim", lambda x: x.mean(dim=1), f(4,5))
reg("amax", lambda x: torch.amax(x, dim=1), f(4,5)); reg("max.dim", lambda x: torch.max(x, dim=1), f(4,5))
reg("argmax", lambda x: torch.argmax(x, dim=1), f(4,5)); reg("any.dim", lambda x: torch.any(x, dim=1), bb(4,5))
reg("cumsum", lambda x: torch.cumsum(x, dim=1), f(4,5))
reg("var_mean.correction", lambda x: torch.var_mean(x, dim=1, correction=1), f(4,5))
reg("topk", lambda x: torch.topk(x, 2, dim=1), f(4,5))
reg("avg_pool2d", lambda x: torch.nn.functional.avg_pool2d(x, 2), f(1,3,16,16))
reg("max_pool2d_with_indices", lambda x: torch.nn.functional.max_pool2d(x, 2, return_indices=True), f(1,3,16,16))
reg("_adaptive_avg_pool2d", lambda x: torch.nn.functional.adaptive_avg_pool2d(x, 1), f(1,3,16,16))
reg("view", lambda x: x.view(2,10), f(4,5)); reg("view_copy", lambda x: x.view(2,10), f(4,5))
reg("permute", lambda x: x.permute(1,0), f(4,5)); reg("permute_copy", lambda x: x.permute(1,0), f(4,5))
reg("transpose_copy.int", lambda x: x.transpose(0,1), f(4,5)); reg("t_copy", lambda x: x.t(), f(4,5))
reg("squeeze.dim", lambda x: x.squeeze(1), f(4,1,5)); reg("squeeze.dims", lambda x: x.squeeze((1,2)), f(4,1,1,5))
reg("squeeze_copy.dim", lambda x: x.squeeze(1), f(4,1,5)); reg("unsqueeze", lambda x: x.unsqueeze(1), f(4,5))
reg("unsqueeze_copy", lambda x: x.unsqueeze(1), f(4,5)); reg("expand", lambda x: x.expand(3,4,5), f(1,4,5))
reg("expand_copy", lambda x: x.expand(3,4,5), f(1,4,5)); reg("cat", lambda a,b: torch.cat([a,b], dim=1), f(4,5), f(4,3))
reg("slice.Tensor", lambda x: x[:, 1:4], f(4,5))
reg("slice_scatter", lambda x,y: torch.slice_scatter(x, y, dim=1, start=1, end=4), f(4,5), f(4,3))
reg("split_with_sizes", lambda x: torch.split(x, [2,3], dim=1), f(4,5))
reg("split_with_sizes_copy", lambda x: torch.split(x, [2,3], dim=1), f(4,5))
reg("clone", lambda x: torch.clone(x), f(4,5)); reg("copy", lambda x,y: x.copy_(y), f(4,5), f(4,5))
reg("copy_", lambda x,y: x.copy_(y), f(4,5), f(4,5)); reg("_to_copy", lambda x: x.to(torch.float64).to(torch.float32), f(4,5))
reg("index.Tensor", lambda x,idx: x[idx], f(6,5), il(3, hi=6))
reg("index_put", lambda x,idx,v: x.index_put((idx,), v), f(6,5), il(3, hi=6), f(3,5))
reg("index_select", lambda x,idx: torch.index_select(x, 0, idx), f(6,5), il(3, hi=6))
reg("gather", lambda x,idx: torch.gather(x, 1, idx), f(4,5), il(4,2, hi=5))
reg("scatter.src", lambda x,idx,src: x.scatter(1, idx, src), f(4,5), il(4,2, hi=5), f(4,2))
reg("where.self", lambda c,x,y: torch.where(c, x, y), bb(4,5), f(4,5), f(4,5))
reg("full", lambda x: torch.full((4,5), 3.0) + x, f(4,5)); reg("full_like", lambda x: torch.full_like(x, 2.0), f(4,5))
reg("fill.Scalar", lambda x: x.clone().fill_(7.0), f(4,5))
reg("constant_pad_nd", lambda x: torch.nn.functional.pad(x, (1,2), value=0.0), f(4,5))

class Mod(torch.nn.Module):
    def __init__(s, fn): super().__init__(); s.fn = fn
    def forward(s, *a): return s.fn(*a)

def materialize(o):
    for t in pytree.tree_leaves(o):
        if isinstance(t, torch.Tensor): t.to("cpu")

def measure(op, fn, args):
    r = {"op": op, "trace": None, "aot": None, "npu_exec": None, "eager": None, "rngd": 0, "cpu": 0, "primary": None, "err": ""}
    # --- AOT 경로 ---
    try:
        ep = torch.export.export(Mod(fn).eval(), tuple(a.clone() for a in args)).run_decompositions(TABLE)
        r["trace"] = "ok"
        try:
            cm = CompileModule.from_exported(ep); r["aot"] = "ok"
            try:
                cm.to(DEV); cm(*[a.to(DEV) for a in args], device=DEV); r["npu_exec"] = "ok"
            except Exception as e:
                r["npu_exec"] = "fail"; r["err"] = repr(e)[:90]
        except Exception as e:
            r["aot"] = "fail" if "UnsupportedOp" in repr(e) else "exc"; r["err"] = r["err"] or repr(e)[:90]
    except Exception as e:
        r["trace"] = "fail"; r["err"] = repr(e)[:90]
    # --- eager dispatch 경로 (coverage) ---
    if r["trace"] == "ok":
        try:
            dev_args = [a.to(DEV) if isinstance(a, torch.Tensor) else a for a in args]
            with RNGDCoverageTrace(op) as t:
                materialize(fn(*dev_args))
            st = t.statistics(); r["rngd"] = st.total_run_on_rngd; r["cpu"] = st.total_run_on_cpu
            r["eager"] = "npu" if (r["rngd"] > 0 and r["cpu"] == 0) else ("host" if r["cpu"] > 0 else ("npu" if r["rngd"] > 0 else "none"))
        except Exception as e:
            r["eager"] = "exc"; r["err"] = r["err"] or repr(e)[:90]
    # --- primary 분류 ---
    if r["trace"] == "fail":
        p = "trace_unsupported"
    elif r["aot"] == "ok" and r["npu_exec"] == "ok":
        p = "npu"
    elif r["eager"] == "npu":
        p = "npu"
    elif r["eager"] == "host":
        p = "host"
    elif r["aot"] in ("fail", "exc"):
        p = "compile_fail"
    else:
        p = "other"
    r["primary"] = p
    return r

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--names":
        print(json.dumps([t[0] for t in T])); sys.exit(0)
    idx = int(sys.argv[1])
    op, fn, args = T[idx]
    print(json.dumps(measure(op, fn, args)))
