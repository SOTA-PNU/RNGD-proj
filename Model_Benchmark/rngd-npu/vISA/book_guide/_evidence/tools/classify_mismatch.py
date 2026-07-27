#!/usr/bin/env python3
"""Classify a real-hardware value-mismatch failure.

A mismatch can mean two very different things:
  * ULP_ROUNDING  - the kernel is correct; NPU and host round the last bit differently
  * REAL_CORRUPTION - the data is simply wrong (stale buffer, unwritten destination)
Telling them apart is the whole point; "N tests failed" without this split is useless.
"""
import pathlib
import re
import struct
import sys

DETAIL = pathlib.Path("/home/jun/.claude/jobs/46bc5c7e/tmp/ex_logs/npu_matrix_detail")


def ulps_f32(a, b):
    ia = struct.unpack("<i", struct.pack("<f", a))[0]
    ib = struct.unpack("<i", struct.pack("<f", b))[0]
    if (ia < 0) != (ib < 0):
        return abs(ia) + abs(ib)
    return abs(ia - ib)


def classify(path):
    txt = path.read_text()
    m = re.search(r"left:\s*\[(.*?)\]\s*\n\s*right:\s*\[(.*?)\]", txt, re.S)
    if not m:
        # scalar assertion: `left: 24.0` / `right: 0.0`, optionally with a message
        s = re.search(r"assertion .*failed(?::\s*(.*))?\n\s*left:\s*([-\d.eE+]+)\s*\n\s*right:\s*([-\d.eE+]+)", txt)
        if not s:
            return None
        a, b = float(s.group(2)), float(s.group(3))
        u = None if float(a).is_integer() and float(b).is_integer() else ulps_f32(a, b)
        return {
            "verdict": "ULP_ROUNDING" if (u is not None and u <= 2) else "REAL_MISMATCH",
            "n": 1, "n_diff": 1, "frac": 1.0, "max_ulp": u,
            "max_abs": abs(a - b), "int_like": None,
            "sample": [(a, b)], "msg": (s.group(1) or "").strip(),
        }
    def parse(s):
        try:
            return [float(x) for x in s.split(",") if x.strip()]
        except ValueError:
            return None
    L, R = parse(m.group(1)), parse(m.group(2))
    if L is None or R is None or len(L) != len(R) or not L:
        return {"verdict": "UNPARSED"}

    diffs = [(a, b) for a, b in zip(L, R) if a != b]
    is_int = all(float(x).is_integer() for x in L[:200] + R[:200])
    ulps = [] if is_int else [ulps_f32(a, b) for a, b in diffs]
    maxulp = max(ulps) if ulps else None
    frac = len(diffs) / len(L)

    if not diffs:
        verdict = "NO_DIFF"
    elif ulps and maxulp <= 2:
        verdict = "ULP_ROUNDING"
    elif frac > 0.9:
        verdict = "REAL_CORRUPTION"
    else:
        verdict = "REAL_MISMATCH"

    return {
        "verdict": verdict,
        "n": len(L),
        "n_diff": len(diffs),
        "frac": frac,
        "max_ulp": maxulp,
        "max_abs": max(abs(a - b) for a, b in diffs),
        "int_like": is_int,
        "sample": [(round(a, 8), round(b, 8)) for a, b in diffs[:4]],
    }


targets = sys.argv[1:] or [p.stem for p in DETAIL.glob("*.txt")]
print(f"{'verdict':<17} {'diff/total':>12} {'maxULP':>10} {'maxabs':>12}  test")
for name in sorted(targets):
    p = DETAIL / f"{name}.txt"
    if not p.exists():
        continue
    r = classify(p)
    if not r or r["verdict"] == "NO_DIFF":
        continue
    mu = "-" if r.get("max_ulp") is None else str(r["max_ulp"])
    print(f"{r['verdict']:<17} {r['n_diff']:>5}/{r['n']:<6} {mu:>10} {r['max_abs']:>12.4g}  {name}")
    if r["verdict"] != "ULP_ROUNDING":
        print(f"{'':<17} 예시 npu/host: {r['sample'][:3]}")
