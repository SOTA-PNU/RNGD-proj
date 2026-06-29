"""97개 op 를 각각 별도 subprocess(cls_worker.py)로 측정. 비정상 종료(signal/abort/timeout)는
crash 로 분류. 결과를 5분류(npu/host/compile_fail/trace_unsupported/crash)로 집계."""
import subprocess, json, os, signal
PY = "/home/jun/furiosa/bin/python"
WK = "/home/jun/.claude/jobs/220196a8/tmp/cls_worker.py"
ENV = dict(os.environ, RNGD_IDX=os.environ.get("RNGD_IDX", "0"))

names = json.loads(subprocess.run([PY, WK, "--names"], capture_output=True, text=True, env=ENV).stdout.strip().split("\n")[-1])
N = len(names)
print(f"측정 대상 op: {N}개\n")
results = []
for i in range(N):
    op = names[i]
    try:
        r = subprocess.run([PY, WK, str(i)], capture_output=True, text=True, timeout=180, env=ENV)
        if r.returncode == 0:
            js = [l for l in r.stdout.strip().split("\n") if l.startswith("{")]
            rec = json.loads(js[-1]) if js else {"op": op, "primary": "crash", "err": "no-output"}
        else:
            sig = -r.returncode if r.returncode < 0 else r.returncode
            signame = signal.Signals(-r.returncode).name if r.returncode < 0 else f"exit{r.returncode}"
            tail = (r.stderr or "").strip().split("\n")[-1][:100]
            rec = {"op": op, "primary": "crash", "trace": None, "aot": None, "npu_exec": None,
                   "eager": None, "rngd": 0, "cpu": 0, "err": f"{signame}: {tail}"}
    except subprocess.TimeoutExpired:
        rec = {"op": op, "primary": "crash", "err": "timeout(180s)", "trace": None, "aot": None, "npu_exec": None, "eager": None, "rngd": 0, "cpu": 0}
    results.append(rec)
    print(f"{i+1:3d}/{N}  {op:26s} -> {rec['primary']:18s} (trace={rec.get('trace')} aot={rec.get('aot')} npu_exec={rec.get('npu_exec')} eager={rec.get('eager')} r/c={rec.get('rngd')}/{rec.get('cpu')}) {rec.get('err','')[:50]}")

json.dump(results, open("/home/jun/.claude/jobs/220196a8/tmp/cls_results.json", "w"), indent=1)
from collections import Counter
cnt = Counter(r["primary"] for r in results)
print("\n===== 5분류 집계 =====")
for k in ["npu", "host", "compile_fail", "trace_unsupported", "crash", "other"]:
    if cnt.get(k): print(f"  {k:18s} {cnt[k]:3d}   {[r['op'] for r in results if r['primary']==k]}")
