"""Reconstruct a caller->callee call graph from a viztracer chrome-trace JSON.

viztracer emits complete ('X') duration events per thread. Caller/callee nesting
is recovered from (ts, dur) containment within each tid: while the top-of-stack
event has ended, pop; the remaining top is the caller of the next event.

Outputs:
  <stem>_edges.tsv   caller \t callee \t count
  <stem>.dot         Graphviz of the top-N edges (native-boundary nodes highlighted)
  prints a summary (roots, deepest chains, native-boundary leaves)
"""
import json
import sys
from collections import defaultdict

NATIVE_HINTS = ("NativeLLMEngine", "native_runtime", "native_llm_common",
                "native_torch", "NextGenArtifact", "NativeRequestOutput", "engine.generate",
                "stream_generate", ".encode")


def short(name):
    # viztracer names look like "func (path/file.py:line)"; keep "file:func" compactly
    return name.strip()


def main(path):
    stem = path.rsplit(".", 1)[0]
    with open(path) as f:
        data = json.load(f)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    byt = defaultdict(list)
    for e in events:
        if e.get("ph") != "X":
            continue
        if "ts" not in e or "name" not in e:
            continue
        byt[e.get("tid", 0)].append(e)

    edges = defaultdict(int)
    nodes = set()
    roots = defaultdict(int)
    callee_count = defaultdict(int)
    max_depth = 0
    deepest = []

    for tid, evs in byt.items():
        evs.sort(key=lambda e: (e["ts"], -(e.get("dur", 0))))
        stack = []  # (name, end)
        for e in evs:
            ts = e["ts"]; dur = e.get("dur", 0) or 0; name = short(e["name"])
            while stack and stack[-1][1] <= ts:
                stack.pop()
            nodes.add(name)
            callee_count[name] += 1
            if stack:
                edges[(stack[-1][0], name)] += 1
            else:
                roots[name] += 1
            stack.append((name, ts + dur))
            if len(stack) > max_depth:
                max_depth = len(stack)
                deepest = [s[0] for s in stack]

    # write edges tsv
    with open(stem + "_edges.tsv", "w") as f:
        f.write("caller\tcallee\tcount\n")
        for (a, b), c in sorted(edges.items(), key=lambda kv: -kv[1]):
            f.write(f"{a}\t{b}\t{c}\n")

    # native-boundary leaves: nodes that look native and have few/no callees below
    boundary = sorted({n for n in nodes if any(h in n for h in NATIVE_HINTS)})

    # top-N dot
    topn = sorted(edges.items(), key=lambda kv: -kv[1])[:120]
    keep = set()
    for (a, b), _ in topn:
        keep.add(a); keep.add(b)

    def nid(n):
        return "n" + str(abs(hash(n)) % (10 ** 12))

    with open(stem + ".dot", "w") as f:
        f.write("digraph viz {\n  rankdir=LR;\n  node [shape=box, fontsize=9];\n")
        for n in keep:
            attr = ""
            if any(h in n for h in NATIVE_HINTS):
                attr = ", style=filled, fillcolor=lightyellow"
            lbl = n.replace('"', "'")
            f.write(f'  {nid(n)} [label="{lbl}"{attr}];\n')
        for (a, b), c in topn:
            f.write(f'  {nid(a)} -> {nid(b)} [label="{c}"];\n')
        f.write("}\n")

    print(f"events={sum(len(v) for v in byt.values())} nodes={len(nodes)} edges={len(edges)} max_depth={max_depth}")
    print("ROOTS:", dict(sorted(roots.items(), key=lambda kv: -kv[1])[:10]))
    print("NATIVE-BOUNDARY LEAVES:")
    for b in boundary:
        print("   ", b)
    print("DEEPEST CHAIN (one sample):")
    for i, n in enumerate(deepest[:40]):
        print("   " * 0 + "  " * i + "-> " + n)


if __name__ == "__main__":
    main(sys.argv[1])
