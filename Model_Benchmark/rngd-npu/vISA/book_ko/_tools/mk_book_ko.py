#!/usr/bin/env python3
"""book_ko 스캐폴드: docs/src 복제 + {{#include}} 인라인 해소."""
import re, shutil, sys
from pathlib import Path

REPO = Path("/home/jun/.claude/jobs/46bc5c7e/tmp/repo/furiosa-opt-main")
SRC = REPO / "docs" / "src"
DST = Path("/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vISA/book_ko")

INC = re.compile(r"\{\{#include\s+([^}]+?)\}\}")


def anchor_body(path: Path, name: str):
    """mdbook ANCHOR: name .. ANCHOR_END: name 구간 추출 (마커 줄 제외)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    beg = re.compile(r"ANCHOR:\s*" + re.escape(name) + r"\s*$")
    end = re.compile(r"ANCHOR_END:\s*" + re.escape(name) + r"\s*$")
    out, on = [], False
    for ln in lines:
        if end.search(ln):
            on = False
            continue
        if beg.search(ln):
            on = True
            continue
        if on:
            # 구간 안의 다른 ANCHOR 마커 줄은 mdbook 이 제거한다
            if re.search(r"ANCHOR(_END)?:", ln):
                continue
            out.append(ln)
    if not out:
        return None
    # 공통 들여쓰기 제거 (mdbook 동작과 동일)
    ind = min((len(l) - len(l.lstrip()) for l in out if l.strip()), default=0)
    return "\n".join(l[ind:] if l.strip() else "" for l in out)


def main():
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    # book.toml / mermaid 자산
    for f in ("book.toml", "mermaid-init.js", "mermaid.min.js"):
        shutil.copy2(REPO / "docs" / f, DST / f)
    shutil.copytree(SRC, DST / "src")

    resolved = unresolved = 0
    for md in sorted((DST / "src").rglob("*.md")):
        rel = md.relative_to(DST / "src")
        text = md.read_text(encoding="utf-8")
        if "{{#include" not in text:
            continue

        def sub(m):
            nonlocal resolved, unresolved
            spec = m.group(1).strip()
            if ":" in spec:
                p, name = spec.rsplit(":", 1)
            else:
                p, name = spec, None
            # 원본 md 위치 기준 상대경로 해석
            target = (SRC / rel).parent.joinpath(p).resolve()
            if not target.is_file():
                unresolved += 1
                print(f"  MISS file  {rel}  ->  {p}", file=sys.stderr)
                return m.group(0)
            if name is None:
                body = target.read_text(encoding="utf-8").rstrip("\n")
            else:
                body = anchor_body(target, name)
                if body is None:
                    unresolved += 1
                    print(f"  MISS anchor {rel} -> {p}:{name}", file=sys.stderr)
                    return m.group(0)
            resolved += 1
            return body

        md.write_text(INC.sub(sub, text), encoding="utf-8")

    print(f"resolved={resolved} unresolved={unresolved}")
    return 1 if unresolved else 0


sys.exit(main())
