"""97 op × 10 dtype 분류 매트릭스. op별 subprocess(dtype_full_worker.py) 격리(crash 감지)."""
import subprocess, json, os, signal
PY = "/home/jun/furiosa/bin/python"
WK = "/home/jun/.claude/jobs/220196a8/tmp/dtype_full_worker.py"
ENV = dict(os.environ, RNGD_IDX=os.environ.get("RNGD_IDX", "0"))
DTS = ["float64","float32","float16","bfloat16","int64","int32","int16","int8","uint16","uint32"]
names = json.loads(subprocess.run([PY, WK, "--names"], capture_output=True, text=True, env=ENV).stdout.strip().split("\n")[-1])
N = len(names)
AB = {"npu":"N","host":"H","compile_fail":"C","trace_unsupported":"T","na":"-","crash":"X"}
rows = []
print(f"{N} ops × {len(DTS)} dtypes\n")
print(f"{'op':24s}" + "".join(f"{d[:7]:>8s}" for d in DTS))
for i in range(N):
    op = names[i]
    try:
        r = subprocess.run([PY, WK, str(i)], capture_output=True, text=True, timeout=300, env=ENV)
        if r.returncode == 0:
            js = [l for l in r.stdout.strip().split("\n") if l.startswith("{")]
            rec = json.loads(js[-1]) if js else {"op": op, **{d: "crash" for d in DTS}}
        else:
            rec = {"op": op, **{d: "crash" for d in DTS}}
    except subprocess.TimeoutExpired:
        rec = {"op": op, **{d: "crash" for d in DTS}}
    rows.append(rec)
    print(f"{op:24s}" + "".join(f"{AB.get(rec.get(d,'crash'),'?'):>8s}" for d in DTS))
json.dump({"dtypes": DTS, "rows": rows}, open("/home/jun/.claude/jobs/220196a8/tmp/dtype_full.json", "w"), indent=1)
# 집계
from collections import Counter
print("\n===== dtype별 분류 집계 =====")
for d in DTS:
    c = Counter(r.get(d) for r in rows)
    print(f"  {d:9s} npu={c.get('npu',0):2d} host={c.get('host',0):2d} compile_fail={c.get('compile_fail',0):2d} trace_unsup={c.get('trace_unsupported',0):2d} na={c.get('na',0):2d} crash={c.get('crash',0):2d}")
