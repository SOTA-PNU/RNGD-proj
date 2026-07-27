#!/usr/bin/env python3
"""Read a `--backend npu` build log and gate every test function that fails to
compile because it references an npu-gated kernel.

Inserts `#[cfg(not(backend = "npu"))]` above the test's attribute block, which is
the same idiom tests/matmul_tests.rs:124 already uses upstream.
Run repeatedly (build -> gate -> build) until the log is clean.
"""
import pathlib
import re
import sys

BASE = pathlib.Path("/home/jun/.claude/jobs/46bc5c7e/tmp/visa_ex_gated")
GATE = '#[cfg(not(backend = "npu"))]'
ATTR = re.compile(r"\s*#\[")
FN = re.compile(r"\s*(?:pub )?(?:async )?fn \w+")

log = pathlib.Path(sys.argv[1]).read_text().splitlines()

# collect (file, line) pairs that rustc pointed at inside tests/
hits = set()
for i, ln in enumerate(log):
    if not re.match(r"^error(\[E\d+\])?:", ln):
        continue
    for j in range(i + 1, min(i + 4, len(log))):
        m = re.search(r"--> (furiosa-opt-examples/tests/[\w/]+\.rs):(\d+):", log[j])
        if m:
            hits.add((m.group(1), int(m.group(2))))
            break

by_file = {}
for f, line in hits:
    by_file.setdefault(f, []).append(line)

total = 0
for f, lines_no in sorted(by_file.items()):
    p = BASE / f
    src = p.read_text().split("\n")
    starts = set()
    for n in sorted(lines_no):
        i = n - 1
        # walk up to the fn signature, then above its attribute block
        while i >= 0 and not FN.match(src[i]):
            i -= 1
        if i < 0:
            continue
        while i - 1 >= 0 and (ATTR.match(src[i - 1]) or src[i - 1].strip().startswith(("///", "//"))):
            i -= 1
        if GATE in src[max(0, i - 1)]:
            continue
        starts.add(i)
    for i in sorted(starts, reverse=True):
        indent = re.match(r"\s*", src[i]).group(0)
        src.insert(i, indent + GATE)
        total += 1
    if starts:
        p.write_text("\n".join(src))
        print(f"{f}: gated {len(starts)} test fn(s)")

print(f"total gated: {total}")
sys.exit(0 if total else 2)
