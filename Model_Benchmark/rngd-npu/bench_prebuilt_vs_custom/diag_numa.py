#!/usr/bin/env python3
"""요청 하나를 흘리는 동안 0.5초마다 커널 카운터를 찍어, 스텝 사이 멈춤과 시각을 맞춰 본다.

  diag_numa.py <serve URL> <모델> [--prompt long] [--cap 16384] [--tag 이름]

찍는 것: /proc/vmstat 의 numa_hint_faults, numa_pages_migrated, pgmigrate_success, compact_stall,
thp_fault_alloc 등과 serve 프로세스의 minflt/majflt. 멈춤(조각 간격 0.1초 초과)이 난 표본 구간에서
카운터가 얼마나 늘었는지를 멈춤 없는 구간과 비교한다.
(2026-09-11: 넉 장짜리 모델에서 스텝 사이 멈춤이 몇 분 단위로 밀려왔다 빠졌다. smi 호출, max_tokens,
 라우터는 결정 실험으로 원인에서 빠졌다. 남은 후보가 커널의 자동 NUMA 균형)
"""
import argparse, json, os, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
from length_probe import stream_probe  # noqa: E402

KEYS = ("numa_hint_faults", "numa_hint_faults_local", "numa_pages_migrated", "numa_pte_updates",
        "pgmigrate_success", "compact_stall", "thp_fault_alloc", "thp_migration_success", "pgfault", "pgmajfault")


def vmstat():
    d = {}
    for line in open("/proc/vmstat"):
        k, v = line.split()
        if k in KEYS:
            d[k] = int(v)
    return d


def procflt(pid):
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return {"minflt": int(f[7]), "majflt": int(f[9])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("--prompt", default="long")
    ap.add_argument("--cap", type=int, default=16384)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    bench.ROUTER = a.url.rstrip("/")
    pid = subprocess.run(["pgrep", "-f", f"furiosa-llm serve .*{a.model}"], capture_output=True, text=True).stdout.split()[0]
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            s = {"t": time.perf_counter(), **vmstat(), **procflt(pid)}
            samples.append(s)
            stop.wait(0.5)
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    time.sleep(1.0)
    t_start = time.perf_counter()
    r = stream_probe(a.model, dict(bench.PROMPTS)[a.prompt], a.cap)
    time.sleep(1.0)
    stop.set(); th.join()

    t = r["times"]
    gaps = [(t_start + x, t_start + y) for x, y in zip(t, t[1:]) if y - x > 0.1]
    rows = []
    for s0, s1 in zip(samples, samples[1:]):
        if s1["t"] < t_start + t[0] or s0["t"] > t_start + t[-1]:
            continue
        stall = sum(min(b, s1["t"]) - max(a_, s0["t"]) for a_, b in gaps if b > s0["t"] and a_ < s1["t"])
        rows.append({"stall_s": round(stall, 3), **{k: s1[k] - s0[k] for k in KEYS + ("minflt", "majflt")}})
    hot = [x for x in rows if x["stall_s"] > 0.05]
    cold = [x for x in rows if x["stall_s"] == 0]

    def mean(xs, k):
        return round(sum(x[k] for x in xs) / len(xs), 1) if xs else None
    print(f"=== {a.model} {a.prompt} cap {a.cap}: {r['out_tokens']}tok {r['decode_tps']:.1f} tok/s, "
          f"멈춤 {len(gaps)}회 {sum(b - a_ for a_, b in gaps):.1f}s, 표본 {len(rows)}개(멈춤 구간 {len(hot)} / 조용한 구간 {len(cold)})")
    print(f"{'카운터(0.5초당 증가)':26s} {'멈춤 구간':>10s} {'조용한 구간':>10s}")
    for k in KEYS + ("minflt", "majflt"):
        print(f"{k:26s} {mean(hot, k)!s:>10s} {mean(cold, k)!s:>10s}")
    out = os.path.join(HERE, "results_len_diag", f"numa_{a.model}{('_' + a.tag) if a.tag else ''}.json")
    json.dump({"model": a.model, "prompt": a.prompt, "cap": a.cap, "run": {k: r[k] for k in ("out_tokens", "decode_tps", "total")},
               "stalls": len(gaps), "rows": rows}, open(out, "w", encoding="utf-8"), indent=1)
    print(f"    → {out}")


if __name__ == "__main__":
    main()
