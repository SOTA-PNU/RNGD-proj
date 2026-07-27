#!/usr/bin/env python3
"""mdbook 앵커 처리.

audit  <dir>            : 링크가 참조하는 앵커가 원본 헤딩에서 나오는지 검사
plan   <dir> <out.json> : 파일별 [헤딩index -> 주입할 slug] 계획 생성
inject <dir> <plan.json>: 번역본 헤딩 앞에 <a id="slug"></a> 주입
"""
import json, os, re, sys
from pathlib import Path

FENCE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)?(`{3,}|~{3,})")
HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
LINK = re.compile(r"\]\(([^)]*)\)")


def slugify(s: str) -> str:
    s = re.sub(r"`([^`]*)`", r"\1", s)          # 인라인 코드 백틱 제거
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)  # 링크는 텍스트만
    s = re.sub(r"[*_]", "", s)                   # 강조 기호
    s = s.lower()
    s = re.sub(r"[^\w\- ]", "", s, flags=re.UNICODE)
    return s.replace(" ", "-")


def headings(text: str):
    """(순번, level, 원문) 목록 — 코드펜스 안은 제외."""
    out, fence = [], None
    for ln in text.split("\n"):
        m = FENCE.match(ln)
        if fence is not None:
            if m and m.group(1)[0] == fence[0]:
                fence = None
            continue
        if m:
            fence = m.group(1)
            continue
        h = HEAD.match(ln)
        if h:
            out.append((len(h.group(1)), h.group(2)))
    return out


def body_links(text: str):
    out, fence = [], None
    for ln in text.split("\n"):
        m = FENCE.match(ln)
        if fence is not None:
            if m and m.group(1)[0] == fence[0]:
                fence = None
            continue
        if m:
            fence = m.group(1)
            continue
        out.extend(LINK.findall(ln))
    return out


def load(d: Path):
    return {p.relative_to(d): p.read_text(encoding="utf-8") for p in sorted(d.rglob("*.md"))}


def main():
    mode, d = sys.argv[1], Path(sys.argv[2])
    files = load(d)
    # 앵커는 헤딩 슬러그 + 명시적 <a id="..."> 둘 다에서 나온다 (원문 책도 후자를 쓴다)
    EXPLICIT = re.compile(r'<a\s+(?:id|name)\s*=\s*["\']([^"\']+)["\']')
    slugs = {
        f: {slugify(t) for _, t in headings(x)} | set(EXPLICIT.findall(x))
        for f, x in files.items()
    }

    if mode in ("audit", "plan"):
        # 참조된 앵커: (대상파일, slug)
        want = {}
        broken = []
        for f, text in files.items():
            for tgt in body_links(text):
                if "#" not in tgt or tgt.startswith("http"):
                    continue
                path, _, frag = tgt.partition("#")
                if not frag:
                    continue
                if path == "":
                    dest = f
                else:
                    dest = Path(os.path.normpath((f.parent / path).as_posix()))
                if dest not in files:
                    broken.append((str(f), tgt, "대상 파일 없음"))
                    continue
                if frag not in slugs[dest]:
                    broken.append((str(f), tgt, "앵커 없음"))
                    continue
                want.setdefault(dest, set()).add(frag)

        if mode == "audit":
            n = sum(len(v) for v in want.values())
            print(f"참조되는 앵커 {n}종 / {len(want)}개 파일에서 확인됨")
            if broken:
                print(f"\n원본 책 자체가 깨진 참조 {len(broken)}건:")
                for a, b, c in broken:
                    print(f"  {a}  ->  {b}   ({c})")
            else:
                print("깨진 참조 없음")
            return 0

        plan = {}
        for f, need in want.items():
            hs = headings(files[f])
            idxmap = {}
            for i, (_, t) in enumerate(hs):
                s = slugify(t)
                if s in need and s not in idxmap.values():
                    idxmap[i] = s
            plan[str(f)] = idxmap
        Path(sys.argv[3]).write_text(json.dumps(plan, ensure_ascii=False, indent=1))
        print(f"plan: {sum(len(v) for v in plan.values())} 앵커 / {len(plan)} 파일")
        return 0

    if mode == "inject":
        plan = json.loads(Path(sys.argv[3]).read_text())
        total = 0
        for f, idxmap in plan.items():
            p = d / f
            text = p.read_text(encoding="utf-8")
            lines = text.split("\n")
            out, fence, hi = [], None, 0
            for ln in lines:
                m = FENCE.match(ln)
                if fence is not None:
                    out.append(ln)
                    if m and m.group(1)[0] == fence[0]:
                        fence = None
                    continue
                if m:
                    fence = m.group(1)
                    out.append(ln)
                    continue
                if HEAD.match(ln):
                    s = idxmap.get(str(hi))
                    if s:
                        out.append(f'<a id="{s}"></a>')
                        total += 1
                    hi += 1
                out.append(ln)
            p.write_text("\n".join(out), encoding="utf-8")
        print(f"injected {total} anchors")
        return 0

    print("unknown mode")
    return 2


sys.exit(main())
