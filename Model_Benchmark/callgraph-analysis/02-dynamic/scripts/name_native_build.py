"""Assign PROVISIONAL (heuristic) names to the stripped native_llm_common.so `??`
frames seen in the COMPILE-phase gdb snapshots of ray::TaskCompileActor.

native_llm_common.cpython-312-x86_64-linux-gnu.so (143 MB) is the PyO3 library
behind `furiosa.native_common.compiler` (compiler.compile / create_*_compiler_config
/ GraphMetadataBuilder / CompiledGraph.*). It is fully stripped (nm: no symbols),
so gdb shows only runtime addresses. gdb_build_compile_1..5 were all taken on the
SAME TaskCompileActor process (LWP 3105741), so an address denotes the SAME
function across the five snapshots (fixed ASLR per process) — exactly as
gdb_load/idle/infer did for serve's native_runtime.so.

Naming is by (1) ADDRESS REGION (the byte of the .so offset = a code area /
compiler pass), confirmed by two ground-truth ladders captured live:
  • Python MainThread:  _PyEval -> CPython call -> compile DRIVER (region 19)
                        -> sync (region 1d 0x..ddf/0x..ded) -> syscall (waits on pool)
  • active pool thread: tactic LEAF (region 1d 0x1da50e14) -> mid-lower (region 1b)
                        -> passA (region 1a) -> recursive operator VISITOR loop
                        (region 19: 0x19989eb9 <-> 0x19b897b9 <-> 0x19b67a4d, repeating)
  • 62 parked pool threads: nllc.so!0x1fbccdaa -> syscall (work-queue wait)
and (2) a handful of explicitly-named hot frames. They are GUESSES for
readability, not symbolicated truth.

Outputs (03-synthesis/full-callgraphs/):
  gdb_build.native_names.md        region table + hot frames + thread archetypes
  gdb_build.calltree.named.txt     compile call tree with addresses -> names
"""
import os
import re
from collections import Counter, defaultdict

CA = "/home/jun/RNGD-proj/Model_Benchmark/callgraph-analysis"
LOGS = f"{CA}/02-dynamic/logs"
OUT = f"{CA}/03-synthesis/full-callgraphs"
SNAPS = [f"gdb_build_compile_{k}" for k in (1, 2, 3, 4, 5)]
SO = "native_llm_common"

# region byte (offset[-8:][:2]) -> (role-prefix, description) for the compiler passes
REGION = {
    "19": ("lower.drv",    "lowering driver + recursive operator visitor (main lowering loop)"),
    "1a": ("lower.pA",     "lowering sub-pass A"),
    "1b": ("lower.mid",    "mid-level lowering / IR transform"),
    "1c": ("lower.cg",     "lowering / codegen sub-pass (region between mid and tactic)"),
    "1d": ("lower.tac",    "innermost tactic-selection / codegen (leaf) + driver sync primitive"),
    "1f": ("pool",         "compiler worker-thread pool entry/park"),
}
# explicitly-named hot frames (full address) -> (name, note). '*' marks a key boundary/leaf.
HOT = {
    "0x00007f0919513d61": ("compile.driver.enter",   "★ first native frame under the CPython compile() PyO3 call (compile orchestration entry)"),
    "0x00007f091ded6c05": ("compile.wait.SYSCALL",   "★ driver blocks here -> syscall (waits on worker pool to finish lowering)"),
    "0x00007f091ddf6109": ("compile.sync",           "driver sync/collect primitive (region 1d)"),
    "0x00007f091fbccdaa": ("pool.worker.park",       "★ worker-pool thread entry; 62 threads park here -> syscall (work-queue wait)"),
    "0x00007f0919989eb9": ("lower.visit.recurse.A",  "★ recursive operator-tree visitor (loops with .B/.C)"),
    "0x00007f0919b897b9": ("lower.visit.recurse.B",  "★ recursive operator-tree visitor (loops with .A/.C)"),
    "0x00007f0919b67a4d": ("lower.visit.recurse.C",  "★ recursive operator-tree visitor (loops with .A/.B)"),
    "0x00007f091982dd9a": ("lower.visit.driver",     "per-operator lowering driver (calls the recursion)"),
    "0x00007f091ddf26e4": ("lower.visit.sync",       "lowering visitor sync/branch (region 1d)"),
    "0x00007f091da50e14": ("lower.tactic.leaf",      "★ innermost lowering / tactic search LEAF — origin of 'failed to lower (no tactic)'"),
    "0x00007f091b2fe721": ("lower.mid.enter",        "entry into mid-level lowering pass"),
    "0x00007f091aee71ee": ("lower.pA.enter",         "entry into lowering sub-pass A"),
}

NAT_RE = re.compile(r'(0x[0-9a-f]+)\s+in\s+\?\?\s+\(\)\s+from\s+\S*' + SO)


def region_of(addr):
    return addr[-8:][:2]


def name_of(addr):
    if addr in HOT:
        return HOT[addr][0]
    pref = REGION.get(region_of(addr), ("nllc.r" + region_of(addr), ""))[0]
    return f"{pref}.{addr[-6:]}"


def native_addrs(path):
    s = set()
    try:
        for line in open(path, errors="replace"):
            m = NAT_RE.search(line)
            if m:
                s.add(m.group(1))
    except FileNotFoundError:
        pass
    return s


def frame_label(rest, addr):
    head = rest.split(' (')[0].strip()
    if head == '??' or head == '':
        mso = re.search(r'from\s+(\S+)', rest)
        so = os.path.basename(mso.group(1)) if mso else 'unknown'
        so = re.sub(r'\.cpython-[^.]*', '', so)
        return f"{so}!{addr if addr else '0x?'}"
    return head


def infer_callees(path):
    callee = defaultdict(Counter)
    total = Counter()
    try:
        text = open(path, errors="replace").read()
    except FileNotFoundError:
        return callee, total
    for block in re.split(r'\nThread \d+ ', text):
        frames = []
        for line in block.splitlines():
            fm = re.match(r'^#\d+\s+(.*)$', line.strip())
            if not fm:
                continue
            rest = fm.group(1)
            am = re.match(r'^(0x[0-9a-f]+)\s+in\s+(.*)$', rest)
            addr = None
            if am:
                addr = am.group(1); rest = am.group(2)
            frames.append(frame_label(rest, addr))
        for i in range(len(frames) - 1):
            inner, outer = frames[i], frames[i + 1]
            mo = re.search(SO + r'\.so!(0x[0-9a-f]+)', outer)
            if mo:
                callee[mo.group(1)][inner] += 1
                total[mo.group(1)] += 1
    return callee, total


def short(lbl):
    m = re.search(SO + r'\.so!(0x[0-9a-f]+)', lbl)
    return name_of(m.group(1)) if m else lbl


def main():
    pres = {k: native_addrs(f"{LOGS}/{k}.txt") for k in SNAPS}
    all_addrs = set().union(*pres.values()) if pres else set()
    callee = defaultdict(Counter); total = Counter()
    for k in SNAPS:
        c, t = infer_callees(f"{LOGS}/{k}.txt")
        for a, cc in c.items():
            callee[a].update(cc)
        total.update(t)

    rows = []
    for addr in all_addrs:
        reg = region_of(addr)
        cset = ", ".join(f"{short(c)}(x{n})" for c, n in (callee.get(addr) or Counter()).most_common(3))
        presence = "".join((str(i + 1) if addr in pres[k] else "-") for i, k in enumerate(SNAPS))
        note = HOT[addr][1] if addr in HOT else ""
        rows.append((reg, -total.get(addr, 0), name_of(addr), addr, presence, total.get(addr, 0), cset, note))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    by_reg = Counter(region_of(a) for a in all_addrs)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/gdb_build.native_names.md", "w") as f:
        f.write("# `furiosa-llm build` 컴파일 단계 `??`(native_llm_common.so) 프레임 — 간이(provisional) 이름\n\n")
        f.write("`native_llm_common.cpython-312*.so`(143 MB)는 **스트립**되어 함수명이 없고 런타임 주소만 보입니다. "
                "이 라이브러리가 `furiosa.native_common.compiler`(`compile` / `create_*_compiler_config` / "
                "`GraphMetadataBuilder` / `CompiledGraph.*`)의 실체이며, `TaskCompileActor`가 `compile()`을 호출하면 "
                "여기서 17분간 lowering을 돌다가 `failed to lower the operator O1089 (no tactic)`로 실패했습니다. "
                "아래 이름은 **추론치**입니다 — 근거: (1) 주소 영역(.so 오프셋의 상위 바이트 = 코드 구역/컴파일 패스), "
                "(2) 라이브로 잡은 두 콜래더(파이썬 MainThread = compile 드라이버→sync→syscall, 활성 풀 스레드 = "
                "tactic leaf→mid-lower→재귀 operator visitor), (3) 위쪽 파이썬 경계(py-spy: "
                "`compile_task → compile_gm_and_get_preprocessed_gm_hash → compile()@converter.py:913`). "
                "gdb_build_compile_1..5 는 **동일 TaskCompileActor 프로세스(LWP 3105741)** 라 주소가 같은 함수를 가리킵니다.\n\n")
        f.write("## 스레드 아키타입 (gdb 244 스레드)\n\n")
        f.write("| 아키타입 | 개수(대략) | 스택 요지 |\n|---|---|---|\n")
        f.write("| `compile.driver` (파이썬 MainThread) | 1 | Ray actor → `_PyEval` → CPython call → **native compile 드라이버(region 19)** → sync(region 1d) → **syscall**(풀 대기) |\n")
        f.write("| `lower.pool.parked` (컴파일러 워커풀) | ~62 | `clone3 → start_thread → pool.worker.park(0x…1fbccdaa) → syscall` (작업큐 대기) |\n")
        f.write("| `lower.pool.active` (컴파일러 워커풀) | 수~수십 | `tactic.leaf(1d) → mid-lower(1b) → passA(1a) → 재귀 operator visitor(19)` — 실제 lowering 수행 |\n")
        f.write("| `ray.infra` (event_engine·nexting·grpc·poll·gcs) | ~50 | Ray gRPC/이벤트루프/타이머 — 컴파일러 아님 |\n\n")
        f.write("## 주소 영역(region) = 컴파일 패스\n\n")
        f.write("| region(오프셋 상위바이트) | 역할(추론) | 주소 수 |\n|---|---|---:|\n")
        for r in sorted(REGION):
            f.write(f"| `0x..{r}xxxxxx` `{REGION[r][0]}` | {REGION[r][1]} | {by_reg.get(r,0)} |\n")
        other = sum(v for k, v in by_reg.items() if k not in REGION)
        f.write(f"| (기타 영역) | 보조/런타임 글루 | {other} |\n\n")
        f.write(f"## 주소별 간이 이름 (총 {len(rows)}개, region별 정렬; 통과수 내림차순)\n\n")
        f.write("- 존재(Presence): 1..5 = gdb_build_compile_{1..5} 스냅샷 등장. 통과수 = 5 스냅샷 합산 caller→callee 통과 횟수.\n")
        f.write("- ★ = 핵심 경계/리프 프레임. 이름은 추론치이며 심볼화된 사실이 아닙니다.\n\n")
        f.write("| region | 간이 이름 | 주소 | 존재 | 통과수 | 주요 callee | 비고 |\n")
        f.write("|---|---|---|---|---:|---|---|\n")
        for reg, _negtot, nm, addr, presence, tot, cset, note in rows:
            f.write(f"| `{reg}` | `{nm}` | `{addr}` | {presence} | {tot} | {cset or '—'} | {note} |\n")
        f.write(f"\n총 {len(rows)} 개 네이티브 주소. region 5개 + 명시 hot {len(HOT)}개로 명명, 나머지는 region 접두사+주소꼬리. "
                "모든 이름은 추론치입니다.\n")

    # named call tree from the representative compile snapshot
    src = f"{OUT}/gdb_build_compile_1.calltree.txt"
    if os.path.exists(src):
        tree = open(src, errors="replace").read()
        tree2 = re.sub(SO + r'\.so!(0x[0-9a-f]+)', lambda m: f"{name_of(m.group(1))} ⟨{m.group(1)[-8:]}⟩", tree)
        tree2 = tree2.replace("merged call tree of ALL", "[names are provisional/inferred] merged call tree of ALL")
        open(f"{OUT}/gdb_build.calltree.named.txt", "w").write(tree2)

    print(f"named {len(rows)} native addrs across regions {dict(by_reg)} -> gdb_build.native_names.md + gdb_build.calltree.named.txt")


if __name__ == "__main__":
    main()
