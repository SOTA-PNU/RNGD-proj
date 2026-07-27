#!/usr/bin/env python3
"""Gate every #[device] kernel that fails to lower for the npu backend behind
`#[cfg(not(backend = "npu"))]`, so the whole furiosa-opt-examples crate builds
with `--backend npu`.

The repo already uses this idiom (tests/matmul_tests.rs:124), so the gate is
upstream-shaped rather than a hack. Emulation and typecheck are unaffected.

Module-path aware: several inline `pub mod`s in one file may define kernels with
the SAME fn name where only one of them fails, so matching on the bare name is
not enough.
"""
import pathlib
import re
import sys

BASE = pathlib.Path("/home/jun/.claude/jobs/46bc5c7e/tmp/visa_ex_gated/furiosa-opt-examples")
SRC = BASE / "src"
MATRIX = pathlib.Path("/home/jun/.claude/jobs/46bc5c7e/tmp/ex_logs/perkernel_matrix.txt")
GATE = '#[cfg(not(backend = "npu"))]'


def file_module(path):
    rel = path.relative_to(SRC)
    parts = list(rel.parts)
    if parts[-1] == "mod.rs":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return parts


def scan(path):
    """Yield (full_module_path, line_index_of_#[device]) for each kernel."""
    base = file_module(path)
    lines = path.read_text().split("\n")
    stack, depth, dev_at = [], 0, None
    for i, ln in enumerate(lines):
        m = re.match(r"\s*(?:pub )?mod (\w+)\s*\{", ln)
        if m:
            stack.append((m.group(1), depth))
            depth += ln.count("{") - ln.count("}")
            continue
        if re.match(r"\s*#\[device\b", ln):
            dev_at = i
        else:
            fn = re.match(r"\s*pub fn (\w+)", ln)
            if fn and dev_at is not None:
                full = "::".join(base + [s[0] for s in stack] + [fn.group(1)])
                yield full, dev_at, lines
                dev_at = None
            elif ln.strip() and not ln.strip().startswith(("#[", "///", "//")):
                dev_at = None
        depth += ln.count("{") - ln.count("}")
        while stack and depth <= stack[-1][1]:
            stack.pop()


fails = {l.split("|")[1] for l in MATRIX.read_text().splitlines() if l.startswith("FAIL|")}

found, inserted = set(), 0
for path in sorted(SRC.rglob("*.rs")):
    hits = [(full, at) for full, at, _ in scan(path) if full in fails]
    if not hits:
        continue
    lines = path.read_text().split("\n")
    # insert bottom-up so earlier line indices stay valid
    for full, at in sorted(hits, key=lambda h: -h[1]):
        indent = re.match(r"\s*", lines[at]).group(0)
        lines.insert(at, indent + GATE)
        inserted += 1
        found.add(full)
    path.write_text("\n".join(lines))

print(f"failing kernel paths : {len(fails)}")
print(f"gates inserted       : {inserted}")
missed = sorted(fails - found)
if missed:
    print(f"NOT FOUND ({len(missed)}):")
    for m in missed:
        print("   ", m)
sys.exit(1 if missed else 0)
