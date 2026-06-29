"""Exhaustive text call graph for the pdb/Python layer, from the viztracer edge list.

Input: viz_load_infer_edges.tsv (caller\tcallee\tcount) — the COMPLETE set of
caller->callee relations recovered from the deterministic viztracer trace
(load_and_infer.py: LLM(...) load + llm.generate() inference). ALL functions, no filter.

Outputs:
  viz_full_adjacency.txt   every function -> all its callees (with call counts)
  viz_full_calltree.txt    DFS-expanded call tree from root(s); each function expanded
                           once in full, later occurrences marked (^ expanded above)
  viz_full_reverse.txt     every function <- all its callers (with counts)
"""
import sys
from collections import defaultdict

def main(tsv, outdir):
    adj = defaultdict(list)            # caller -> [(callee,count)]
    indeg = defaultdict(int)
    outsum = defaultdict(int)
    insum = defaultdict(int)
    nodes = set()
    rev = defaultdict(list)
    with open(tsv) as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 3:
                continue
            a, b, c = parts[0], parts[1], int(parts[2])
            adj[a].append((b, c))
            rev[b].append((a, c))
            indeg[b] += 1
            outsum[a] += c
            insum[b] += c
            nodes.add(a); nodes.add(b)
    for a in adj:
        adj[a].sort(key=lambda kv: -kv[1])
    for b in rev:
        rev[b].sort(key=lambda kv: -kv[1])

    # ---- complete adjacency ----
    with open(f"{outdir}/viz_full_adjacency.txt", "w") as f:
        f.write(f"# COMPLETE Python call graph (viztracer, load + inference). {len(nodes)} functions, "
                f"{sum(len(v) for v in adj.values())} edges. ALL functions, no filter.\n"
                f"# CALLER  [out-calls=N]\\n    -> callee  (xCOUNT)\n\n")
        for a in sorted(adj, key=lambda k: -outsum[k]):
            f.write(f"{a}   [out={outsum[a]}]\n")
            for b, c in adj[a]:
                f.write(f"    -> {b}  (x{c})\n")
            f.write("\n")

    # ---- reverse index ----
    with open(f"{outdir}/viz_full_reverse.txt", "w") as f:
        f.write(f"# Reverse call graph: callee <- callers. {len(nodes)} functions.\n\n")
        for b in sorted(rev, key=lambda k: -insum[k]):
            f.write(f"{b}   [in={insum[b]}]\n")
            for a, c in rev[b]:
                f.write(f"    <- {a}  (x{c})\n")
            f.write("\n")

    # ---- DFS call tree from roots ----
    roots = sorted([n for n in nodes if indeg[n] == 0], key=lambda k: -outsum[k])
    if not roots:
        roots = ["builtins.exec"] if "builtins.exec" in nodes else [max(nodes, key=lambda k: outsum[k])]
    expanded = set()
    import sys as _s
    _s.setrecursionlimit(100000)
    with open(f"{outdir}/viz_full_calltree.txt", "w") as f:
        f.write(f"# DFS call tree from {len(roots)} root(s). Each function expanded in full on first "
                f"visit; '(^ expanded above)' marks a later occurrence (graph has shared/recursive nodes).\n"
                f"# Nx after an arrow = call count on that edge.\n\n")

        def walk(node, depth, incount):
            pad = "  " * depth
            tag = f"  (x{incount})" if incount is not None else ""
            if node in expanded and adj.get(node):
                f.write(f"{pad}{node}{tag}  (^ expanded above)\n")
                return
            f.write(f"{pad}{node}{tag}\n")
            if not adj.get(node):
                return
            expanded.add(node)
            for b, c in adj[node]:
                if depth > 4000:
                    f.write("  " * (depth + 1) + "... (depth cap)\n")
                    break
                walk(b, depth + 1, c)

        for r in roots:
            walk(r, 0, None)
        # any nodes never reached from roots
        unreached = sorted(nodes - expanded - set(n for n in nodes if not adj.get(n)))
        if unreached:
            f.write(f"\n# functions never reached from the above roots ({len(unreached)}):\n")
            for n in unreached:
                f.write(f"#   {n}\n")

    print(f"{len(nodes)} functions, {sum(len(v) for v in adj.values())} edges, {len(roots)} roots "
          f"-> viz_full_adjacency.txt / viz_full_calltree.txt / viz_full_reverse.txt")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
