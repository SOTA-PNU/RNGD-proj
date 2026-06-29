"""Exhaustive text call graph from `thread apply all bt` output — ALL functions, no filter.

Every frame of every thread is parsed. Stripped native_runtime.so frames (shown as
`?? from <so>` by gdb) are kept and labelled `<so>!0xADDR`; identical addresses within
one snapshot (fixed ASLR per process) merge to the same node.

Outputs (per input file <base>):
  <base>.archetypes.txt   every distinct full thread stack (outermost->innermost) + count + thread name
  <base>.calltree.txt     all stacks merged into one call tree (trie), node = function, with thread counts
  <base>.adjacency.txt    complete caller->callee adjacency (counts) across all threads
"""
import os
import re
import sys
from collections import defaultdict, Counter

FRAME_RE = re.compile(r'^#(\d+)\s+(.*)$')
ADDR_IN_RE = re.compile(r'^(0x[0-9a-f]+)\s+in\s+(.*)$')
FROM_RE = re.compile(r'from\s+(\S+)')


def frame_label(rest, addr):
    """rest = text after optional '0xADDR in '. Return a stable function label."""
    head = rest.split(' (')[0].strip()
    if head == '??' or head == '':
        mso = FROM_RE.search(rest)
        so = os.path.basename(mso.group(1)) if mso else 'unknown'
        so = re.sub(r'\.cpython-[^.]*', '', so)   # native_runtime.cpython-...so -> native_runtime.so
        return f"{so}!{addr if addr else '0x?'}"
    return head


def parse(path):
    """Return list of (thread_name, lwp, frames[outer->inner])."""
    with open(path, errors='replace') as f:
        text = f.read()
    threads = []
    cur = None
    for line in text.splitlines():
        th = re.match(r'^Thread \d+ \(.*?(?:LWP (\d+)\b)?.*?(?:"([^"]+)")?\):', line)
        if line.startswith('Thread ') and '(' in line:
            mname = re.search(r'"([^"]+)"', line)
            mlwp = re.search(r'LWP (\d+)', line)
            if cur:
                threads.append(cur)
            cur = [mname.group(1) if mname else '?', mlwp.group(1) if mlwp else '?', []]
            continue
        fm = FRAME_RE.match(line.strip())
        if fm and cur is not None:
            rest = fm.group(2)
            addr = None
            am = ADDR_IN_RE.match(rest)
            if am:
                addr = am.group(1)
                rest = am.group(2)
            cur[2].append(frame_label(rest, addr))
    if cur:
        threads.append(cur)
    # reverse each to outer->inner
    return [(n, l, fr[::-1]) for (n, l, fr) in threads if fr]


def main(path):
    base = path
    threads = parse(path)
    n_threads = len(threads)

    # ---- archetypes (full, no filter) ----
    arche = defaultdict(lambda: [0, '', []])
    for name, lwp, frames in threads:
        key = (name, tuple(frames))
        arche[key][0] += 1
        arche[key][1] = name
        if len(arche[key][2]) < 6:
            arche[key][2].append(lwp)
    with open(base + '.archetypes.txt', 'w') as f:
        f.write(f"# {path}\n# {n_threads} threads, {len(arche)} distinct full stack archetypes "
                f"(outermost -> innermost; native_runtime.so frames shown as so!0xADDR)\n\n")
        for (name, frames), (cnt, _, lwps) in sorted(arche.items(), key=lambda kv: -kv[1][0]):
            f.write(f"==== {cnt} thread(s)  name='{name}'  LWP e.g. {','.join(lwps)} ====\n")
            for i, fr in enumerate(frames):
                f.write("    " * i + ("-> " if i else "   ") + fr + "\n")
            f.write("\n")

    # ---- merged call tree (trie of outer->inner) ----
    # node id = path tuple; store child counts
    children = defaultdict(Counter)     # parent_path -> Counter(child_label)
    node_count = Counter()              # path -> threads passing through
    for name, lwp, frames in threads:
        prefix = ()
        for fr in frames:
            parent = prefix
            prefix = prefix + (fr,)
            children[parent][fr] += 1
            node_count[prefix] += 1

    def write_tree(f, prefix, depth):
        kids = children.get(prefix, {})
        for label, cnt in sorted(kids.items(), key=lambda kv: -kv[1]):
            childpath = prefix + (label,)
            f.write("  " * depth + f"{cnt:>4}x {label}\n")
            write_tree(f, childpath, depth + 1)

    with open(base + '.calltree.txt', 'w') as f:
        f.write(f"# {path}\n# merged call tree of ALL {n_threads} threads "
                f"(root = outermost frame; Nx = threads through this node)\n\n")
        write_tree(f, (), 0)

    # ---- complete adjacency (caller -> callee) ----
    adj = defaultdict(Counter)
    callee_of = defaultdict(Counter)
    for name, lwp, frames in threads:
        for a, b in zip(frames, frames[1:]):   # a (outer/caller) -> b (inner/callee)
            adj[a][b] += 1
            callee_of[b][a] += 1
    allfns = sorted(set(adj) | set(callee_of))
    with open(base + '.adjacency.txt', 'w') as f:
        f.write(f"# {path}\n# COMPLETE call-graph adjacency, ALL {len(allfns)} functions seen.\n"
                f"# format:  CALLER\\n    -> callee  (xCOUNT)\n\n")
        for fn in sorted(adj, key=lambda k: -sum(adj[k].values())):
            f.write(f"{fn}\n")
            for callee, c in sorted(adj[fn].items(), key=lambda kv: -kv[1]):
                f.write(f"    -> {callee}  (x{c})\n")
            f.write("\n")
        leaves = sorted(set(callee_of) - set(adj))
        f.write(f"# LEAF functions (no callee observed; innermost frames) — {len(leaves)}:\n")
        for fn in leaves:
            f.write(f"  {fn}\n")

    print(f"{os.path.basename(path)}: {n_threads} threads, {len(arche)} archetypes, "
          f"{len(allfns)} distinct functions -> .archetypes/.calltree/.adjacency.txt")


if __name__ == '__main__':
    for p in sys.argv[1:]:
        main(p)
