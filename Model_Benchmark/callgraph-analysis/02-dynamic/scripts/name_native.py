"""Assign PROVISIONAL (heuristic) names to the stripped native_runtime.so `??` frames
seen in gdb_infer.txt.

Why this is possible: gdb_load_10s / gdb_idle / gdb_infer were all taken on the SAME
process (pid 2967220), so a runtime address denotes the SAME function in all three.
Names are inferred from: (1) address region/cluster (= same code area in the .so),
(2) position in the call ladder (caller/callee), (3) terminal syscall (syscall=block,
epoll_wait=io reactor), (4) the owning thread name + RUST_LOG subsystem
(scheduler-eager => furiosa_generator::next_gen::scheduler eager loop). They are
GUESSES for readability, not symbolicated ground truth.

Outputs (in 03-synthesis/full-callgraphs/):
  gdb_infer.native_names.md         table: addr -> name + region + callee + count + presence + note
  gdb_infer.calltree.named.txt      gdb_infer call tree with addresses replaced by the names
"""
import os
import re
import sys
from collections import Counter, defaultdict

LOGS = "/home/jun/chacha/callgraph-analysis/02-dynamic/logs"
OUT = "/home/jun/chacha/callgraph-analysis/03-synthesis/full-callgraphs"

# addr_tail (everything after '0x000076e7') -> (name, region, note)
NAMES = {
    # ---- region 0x..e_ : furiosa async runtime / eager-scheduler core (256-thread park stack + reactor tail)
    "6eccb64f": ("furiosa.thread_entry", "core", "native thread trampoline; shared entry of 269 furiosa native threads (called by libc start_thread)"),
    "6ea7d6a2": ("sched.eager.run",        "core", "L1 eager-scheduler thread run"),
    "6ea5ef33": ("sched.eager.loop",       "core", "L2 scheduler main loop"),
    "6ea62bd4": ("sched.eager.step",       "core", "L3 scheduler step/dispatch"),
    "6ea5d931": ("sched.poll.4",           "core", "L4 nested poll"),
    "6ea7fbc5": ("sched.poll.5",           "core", "L5 nested poll"),
    "6ea669cd": ("sched.poll.6",           "core", "L6 nested poll"),
    "6ea6bccd": ("sched.poll.7",           "core", "L7 nested poll"),
    "6ea6f366": ("sched.poll.8",           "core", "L8 nested poll"),
    "6ea718b8": ("sched.poll.9",           "core", "L9 nested poll"),
    "6ea6b318": ("sched.park.enter",       "core", "L10 enter park"),
    "6ea6a455": ("sched.park.dispatch",    "core", "L11 park; branches to wait/epoll/io"),
    "6ea6f8de": ("sched.park.prepare_wait","core", "prepare blocking wait"),
    "6ea85a69": ("sched.wait.SYSCALL",     "core", "★ blocking-wait primitive -> syscall (NPU completion / futex); reached by 259 frames"),
    "6ea70c2e": ("worker.wait.SYSCALL",    "core", "alt caller of the syscall-wait (worker/device pools)"),
    "6ea6f85e": ("sched.park.branch",      "core", "park variant branch"),
    "6ea6715d": ("sched.wait.path2a",      "core", "alt wait path"),
    "6ea70a82": ("sched.wait.path2b",      "core", "alt wait path"),
    "6ea85afd": ("sched.wait.SYSCALL2",    "core", "-> syscall (variant wait primitive)"),
    "6ea74d41": ("io.to_reactor",          "core", "bridge into epoll reactor tail"),
    "6ea6728d": ("io.reactor.poll",        "core", "io reactor poll"),
    "6ea6d54f": ("io.reactor.turn",        "core", "io reactor turn"),
    "6ea82a7b": ("io.reactor.EPOLL",       "core", "★ epoll reactor wait -> epoll_wait"),
    # ---- region 0x..a2_ : io-driver thread pool (8 threads -> epoll reactor)
    "6a20d5e9": ("iodrv.thread_run",       "iodrv", "io-driver thread entry (8 threads)"),
    "6a24c678": ("iodrv.L1",               "iodrv", ""),
    "6a1ecdca": ("iodrv.L2",               "iodrv", ""),
    "6a22b137": ("iodrv.L3",               "iodrv", ""),
    "6a230909": ("iodrv.L4",               "iodrv", ""),
    "6a251bea": ("iodrv.L5",               "iodrv", ""),
    "6a250d32": ("iodrv.L6",               "iodrv", "-> io.to_reactor -> epoll"),
    # ---- region 0x..69_/0x..99_ : worker/device thread pool (4+1 threads -> worker.wait.SYSCALL)
    "6970ba09": ("wrk.thread_run",         "worker", "worker-pool thread entry (4 threads)"),
    "69a21f74": ("wrk.L1",                 "worker", ""),
    "69a89f02": ("wrk.L2",                 "worker", ""),
    "6992be53": ("wrk.L3",                 "worker", "-> worker.wait.SYSCALL"),
    "6970c439": ("wrk2.thread_run",        "worker", "worker variant entry (1 thread)"),
    "69a22624": ("wrk2.L1",                "worker", ""),
    "69a8a795": ("wrk2.L2",                "worker", ""),
    "6992cd52": ("wrk2.L3",                "worker", "-> worker.wait.SYSCALL"),
}

NAT_RE = re.compile(r'0x0+76e7([0-9a-f]+)\s+in\s+\?\?\s+\(\)\s+from\s+\S*native_runtime')


def native_addrs(path):
    s = set()
    with open(path, errors="replace") as f:
        for line in f:
            m = NAT_RE.search(line)
            if m:
                s.add(m.group(1))
    return s


def infer_callees():
    """addr_tail -> Counter(callee_label) and total count, from gdb_infer.txt frames."""
    path = f"{LOGS}/gdb_infer.txt"
    callee = defaultdict(Counter)
    total = Counter()
    with open(path, errors="replace") as f:
        text = f.read()
    for block in re.split(r'\nThread \d+ ', text):
        frames = []  # innermost -> outermost as listed
        for line in block.splitlines():
            fm = re.match(r'^#\d+\s+(.*)$', line.strip())
            if not fm:
                continue
            rest = fm.group(1)
            am = re.match(r'^(0x[0-9a-f]+)\s+in\s+(.*)$', rest)
            addr = None
            if am:
                addr = am.group(1); rest = am.group(2)
            head = rest.split(' (')[0].strip()
            if head == '??':
                mso = re.search(r'from\s+(\S+)', rest)
                so = os.path.basename(mso.group(1)) if mso else 'unknown'
                so = re.sub(r'\.cpython-[^.]*', '', so)
                nm = NAT_RE.search(am.group(1) + ' in ?? () from ' + (mso.group(1) if mso else '')) if am else None
                lbl = f"{so}!{addr}"
            else:
                lbl = head
            frames.append(lbl)
        # caller = next outer frame (i+1), callee = i
        for i in range(len(frames) - 1):
            inner = frames[i]; outer = frames[i + 1]
            mo = re.search(r'native_runtime\.so!0x0+76e7([0-9a-f]+)', outer)
            if mo:
                callee[mo.group(1)][inner] += 1
                total[mo.group(1)] += 1
    return callee, total


def main():
    pres = {k: native_addrs(f"{LOGS}/{k}.txt") for k in ("gdb_load_10s", "gdb_idle", "gdb_infer")}
    callee, total = infer_callees()

    def short(lbl):
        m = re.search(r'native_runtime\.so!0x0+76e7([0-9a-f]+)', lbl)
        if m and m.group(1) in NAMES:
            return NAMES[m.group(1)][0]
        return lbl

    rows = []
    for tail in pres["gdb_infer"]:
        name, region, note = NAMES.get(tail, (f"nrt.unnamed_{tail}", "?", "(not in name map)"))
        cset = ", ".join(f"{short(c)}(x{n})" for c, n in callee.get(tail, {}).most_common(3))
        presence = "".join(["L" if tail in pres["gdb_load_10s"] else "-",
                             "I" if tail in pres["gdb_idle"] else "-",
                             "F" if tail in pres["gdb_infer"] else "-"])
        rows.append((region, name, tail, presence, total.get(tail, 0), cset, note))
    rows.sort(key=lambda r: (r[0], r[1]))

    with open(f"{OUT}/gdb_infer.native_names.md", "w") as f:
        f.write("# gdb_infer.txt 의 `??` (native_runtime.so) 프레임 — 간이(provisional) 이름\n\n")
        f.write("`native_runtime.so` 는 스트립되어 함수명이 없고 런타임 주소만 보입니다. 아래 이름은 "
                "**추론치**입니다 — 근거: 주소 영역(클러스터)=같은 코드 구역, 콜래더 내 위치, 말단 syscall/epoll, "
                "스레드명/RUST_LOG 서브시스템. gdb_load_10s·gdb_idle·gdb_infer 는 **동일 프로세스(pid 2967220)** 라 "
                "주소가 세 스냅샷에서 같은 함수를 가리킵니다.\n\n")
        f.write("- 영역 `core` = 0x…ea/e 대역: furiosa 비동기 런타임/eager 스케줄러 코어 (256-스레드 파킹 사다리 + epoll reactor 꼬리)\n")
        f.write("- 영역 `iodrv` = 0x…a2 대역: io-driver 스레드 풀 (8 스레드 → epoll)\n")
        f.write("- 영역 `worker` = 0x…69/99 대역: worker/device 스레드 풀 (4+1 스레드 → syscall)\n")
        f.write("- 존재(Presence) 열: L=gdb_load, I=gdb_idle, F=gdb_infer 스냅샷에 등장\n\n")
        f.write("| 영역 | 간이 이름 | 주소(0x…76e7+) | 존재 | infer 통과수 | 주요 callee | 역할(추론) |\n")
        f.write("|---|---|---|---|---:|---|---|\n")
        for region, name, tail, presence, tot, cset, note in rows:
            f.write(f"| {region} | `{name}` | `…{tail}` | {presence} | {tot} | {cset or '—'} | {note} |\n")
        f.write(f"\n총 {len(rows)} 개 네이티브 주소. ★ = 실제 블로킹 지점(syscall/epoll). "
                "`SYSCALL` 접미사 = 그 프레임이 libc `syscall()` 을 직접 호출(= NPU 완료/futex 대기).\n")

    # named call tree: replace addresses in the existing calltree
    src = f"{OUT}/gdb_infer.calltree.txt"
    with open(src, errors="replace") as f:
        tree = f.read()
    def repl(m):
        tail = m.group(1)
        if tail in NAMES:
            return f"{NAMES[tail][0]} ⟨…{tail}⟩"
        return m.group(0)
    tree2 = re.sub(r'native_runtime\.so!0x0+76e7([0-9a-f]+)', repl, tree)
    tree2 = tree2.replace("merged call tree of ALL", "[names are provisional/inferred] merged call tree of ALL")
    with open(f"{OUT}/gdb_infer.calltree.named.txt", "w") as f:
        f.write(tree2)

    print(f"named {len(rows)} native addrs -> gdb_infer.native_names.md + gdb_infer.calltree.named.txt")
    unnamed = [t for t in pres["gdb_infer"] if t not in NAMES]
    if unnamed:
        print("UNNAMED (add to dict):", unnamed)


if __name__ == "__main__":
    main()
