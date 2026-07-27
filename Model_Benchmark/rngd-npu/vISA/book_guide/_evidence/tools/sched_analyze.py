#!/usr/bin/env python3
"""Reduce every dumped kernel schedule to real-hardware characteristics.

`--dump-schedule` is what the compiler believes the NPU will do: per-instruction
lifetimes in cycles, the engine each runs on, and the source line it came from.
It is the only cycle-level instrument in the public stack and needs no device.
"""
import json
import pathlib
import collections

D = pathlib.Path("/home/jun/.claude/jobs/46bc5c7e/tmp/ex_logs/sched")

# Filenames were made by turning "::" into "_", which is lossy because kernel names
# contain "_" too. Recover the real name from the authoritative kernel list.
NAMES = {}
for line in pathlib.Path("/home/jun/.claude/jobs/46bc5c7e/tmp/ok_kernels.txt").read_text().split():
    NAMES[line.replace(":", "_")] = line

rows = []
engine_cycles = collections.Counter()
engine_count = collections.Counter()

for p in sorted(D.glob("*.json")):
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    ins = d.get("instructions", [])
    if not ins:
        continue
    span = max(i["lifetime"]["end"] for i in ins)
    busy = collections.Counter()
    for i in ins:
        dur = i["lifetime"]["end"] - i["lifetime"]["begin"]
        for c in i.get("contexts", ["?"]):
            busy[c] += dur
            engine_cycles[c] += dur
            engine_count[c] += 1
    tensors = d.get("tensors", [])
    sram = sum(t.get("size", 0) for t in tensors if t.get("buffer_type") == "Sram")
    rows.append({
        "kernel": NAMES.get(p.stem, p.stem),
        "file": p.stem,
        "span": span,
        "n_inst": len(ins),
        "engines": dict(busy),
        "n_tensors": len(tensors),
        "sram_bytes": sram,
        "types": collections.Counter(i["tpe"] for i in ins),
    })

rows.sort(key=lambda r: -r["span"])
print(f"analyzed kernels: {len(rows)}")
print()
print("=== 사이클 스팬 분포 ===")
spans = sorted(r["span"] for r in rows)
if spans:
    n = len(spans)
    print(f"min={spans[0]:,}  p25={spans[n//4]:,}  median={spans[n//2]:,}  p75={spans[3*n//4]:,}  max={spans[-1]:,}")
print()
print("=== 가장 무거운 커널 12개 ===")
print(f"{'cycles':>10} {'inst':>5}  kernel")
for r in rows[:12]:
    print(f"{r['span']:>10,} {r['n_inst']:>5}  {r['kernel']}")
print()
print("=== 가장 가벼운 커널 8개 ===")
for r in rows[-8:]:
    print(f"{r['span']:>10,} {r['n_inst']:>5}  {r['kernel']}")
print()
print("=== 엔진별 총 점유 사이클 (전 커널 합) ===")
tot = sum(engine_cycles.values()) or 1
for e, c in engine_cycles.most_common():
    print(f"  {e:<16} {c:>12,} cycles  ({100*c/tot:5.1f}%)  inst={engine_count[e]}")
print()
print("=== 인스트럭션 종류 상위 ===")
allt = collections.Counter()
for r in rows:
    allt.update(r["types"])
for t, c in allt.most_common(12):
    print(f"  {t:<24} {c}")

out = pathlib.Path("/home/jun/.claude/jobs/46bc5c7e/tmp/ex_logs/sched_summary.json")
out.write_text(json.dumps({"kernels": rows, "engine_cycles": dict(engine_cycles)}, indent=1, default=str))
print(f"\nwrote {out}")
