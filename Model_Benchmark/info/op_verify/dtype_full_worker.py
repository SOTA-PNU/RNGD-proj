"""op 1개를 10 dtype 으로 측정 → {op, dtype별 분류}. cls_worker.py 의 97개 op 정의 재사용 +
입력 텐서 dtype 캐스팅(int64=인덱스·bool 유지, 그 외 데이터 텐서만 target dtype). subprocess 격리용."""
import sys, os, json, torch, furiosa.torch
sys.path.insert(0, "/home/jun/.claude/jobs/220196a8/tmp")
import cls_worker as W   # T(97 op), Mod, materialize
from furiosa.torch import CompileModule
from furiosa.torch.coverage import RNGDCoverageTrace
from torch._decomp import core_aten_decompositions
TABLE = dict(core_aten_decompositions())
DEV = torch.device("rngd", int(os.environ.get("RNGD_IDX", "0")))
DT = [("float64",torch.float64),("float32",torch.float32),("float16",torch.float16),
      ("bfloat16",torch.bfloat16),("int64",torch.int64),("int32",torch.int32),
      ("int16",torch.int16),("int8",torch.int8),("uint16",torch.uint16),("uint32",torch.uint32)]

def cast(a, dt):
    if not isinstance(a, torch.Tensor): return a
    if a.dtype in (torch.int64, torch.bool): return a   # 인덱스/bool 역할은 유지
    return a.to(dt)                                       # float·int32 데이터 → target dtype

def classify(fn, args):
    # 0) torch eager 가 이 dtype 에서 op 를 받나
    try:
        W.Mod(fn).eval()(*[a.clone() if isinstance(a, torch.Tensor) else a for a in args])
    except Exception:
        return "na"
    # 1) trace
    try:
        ep = torch.export.export(W.Mod(fn).eval(), tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)).run_decompositions(TABLE)
    except Exception:
        return "trace_unsupported"
    # 2) AOT compile + NPU 실행
    aot = npu_exec = None
    try:
        cm = CompileModule.from_exported(ep); aot = "ok"
        try:
            cm.to(DEV); cm(*[a.to(DEV) if isinstance(a, torch.Tensor) else a for a in args], device=DEV); npu_exec = "ok"
        except Exception:
            npu_exec = "fail"
    except Exception:
        aot = "fail"
    if aot == "ok" and npu_exec == "ok":
        return "npu"
    # 3) eager dispatch (coverage) → npu / host
    try:
        with RNGDCoverageTrace("t") as t:
            W.materialize(fn(*[a.to(DEV) if isinstance(a, torch.Tensor) else a for a in args]))
        st = t.statistics()
        if st.total_run_on_rngd > 0 and st.total_run_on_cpu == 0: return "npu"
        if st.total_run_on_cpu > 0: return "host"
    except Exception:
        return "compile_fail"
    return "compile_fail"

def measure_op(fn, base_args):
    out = {}
    for dn, dt in DT:
        try:
            out[dn] = classify(fn, tuple(cast(a, dt) for a in base_args))
        except Exception:
            out[dn] = "crash"
    return out

if len(sys.argv) > 1 and sys.argv[1] == "--names":
    print(json.dumps([t[0] for t in W.T])); sys.exit(0)
idx = int(sys.argv[1])
op, fn, args = W.T[idx]
print(json.dumps({"op": op, **measure_op(fn, args)}))
