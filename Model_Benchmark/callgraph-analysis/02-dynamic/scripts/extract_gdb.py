"""Collapse `thread apply all bt` output into folded-stack archetypes.

For each Thread block, extract the frame symbols (outermost->innermost) and fold
into 'a;b;c'. Identical folds are counted to reveal the thread taxonomy, and
folds touching the furiosa/device/driver layers are printed in full.
"""
import re
import sys
from collections import Counter, defaultdict

INTEREST = ("furiosa", "device_runtime", "doorbell", "ioctl", "pdma", "generator",
            "scheduler", "native_", "NativeLLM", "serving", "llm_engine", "rngd",
            "dma", "submit", "io_uring", "sample", "tokeniz", "xgrammar", "llg_")

def parse(path):
    with open(path, errors="replace") as f:
        text = f.read()
    blocks = re.split(r"\nThread \d+ ", text)
    folds = Counter()
    fold_name = {}
    examples = {}
    for b in blocks:
        mname = re.search(r'"([^"]+)"', b[:200])
        tname = mname.group(1) if mname else "?"
        frames = []
        for m in re.finditer(r"^#\d+\s+(?:0x[0-9a-f]+ in\s+)?([A-Za-z_][\w:<>, ]*?)\s*(?:\(|$)",
                              b, re.M):
            sym = m.group(1).strip()
            # drop template/arg noise tails
            sym = re.sub(r"<.*", "", sym).strip()
            if sym:
                frames.append(sym)
        if not frames:
            continue
        frames = frames[::-1]  # outermost -> innermost
        fold = ";".join(frames)
        folds[fold] += 1
        fold_name[fold] = tname
        if fold not in examples:
            examples[fold] = ";".join(frames)
    return folds, fold_name, examples

def main(path):
    folds, fold_name, examples = parse(path)
    total = sum(folds.values())
    print(f"# {path}: {total} threads, {len(folds)} distinct stack archetypes\n")
    print("== TOP ARCHETYPES (count x thread-name : innermost<-...<-outermost) ==")
    for fold, c in folds.most_common(25):
        inner = fold.split(";")[-1]
        print(f"{c:4d}  [{fold_name[fold]:16.16s}] innermost={inner}")
    print("\n== STACKS TOUCHING furiosa/device/driver/generator/scheduler (full folds) ==")
    seen = 0
    for fold, c in folds.most_common():
        if any(h in fold for h in INTEREST):
            seen += 1
            print(f"\n--- {c}x  thread={fold_name[fold]} ---")
            for i, fr in enumerate(fold.split(";")):
                print("  " * min(i, 12) + fr)
            if seen >= 18:
                break
    if seen == 0:
        print("(none — all threads were in generic runtime/idle stacks at sample instant)")

if __name__ == "__main__":
    main(sys.argv[1])
