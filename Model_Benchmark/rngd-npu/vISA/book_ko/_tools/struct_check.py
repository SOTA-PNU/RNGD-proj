#!/usr/bin/env python3
"""번역 전/후 구조 동일성 검사.

usage:  struct_check.py baseline <dir> <out.json>
        struct_check.py compare  <dir> <baseline.json>
"""
import hashlib, json, re, sys
from collections import Counter
from pathlib import Path

FENCE = re.compile(r"^(\s*(?:[-*+]\s+|\d+[.)]\s+)?)(`{3,}|~{3,})(.*)$")
LINK = re.compile(r"\]\(([^)]*)\)")
INLINE = re.compile(r"`[^`\n]+`")
ALERT = re.compile(r">\s*\[!(\w+)\]")


def profile(text: str):
    lines = text.split("\n")
    heads, blocks, tbl = [], [], 0
    cur, fence, lang = None, None, None
    body_lines = []
    for ln in lines:
        m = FENCE.match(ln)
        if fence is not None:
            if m and m.group(2)[0] == fence[0] and len(m.group(2)) >= len(fence) and not m.group(3).strip():
                blocks.append([lang, hashlib.sha1("\n".join(cur).encode()).hexdigest()])
                cur, fence, lang = None, None, None
            else:
                cur.append(ln)
            continue
        if m and m.group(2):
            fence, lang, cur = m.group(2), m.group(3).strip(), []
            continue
        body_lines.append(ln)
        if ln.lstrip().startswith("#") and re.match(r"^\s*#{1,6}\s", ln):
            heads.append(len(ln.lstrip()) - len(ln.lstrip().lstrip("#")))
        if ln.lstrip().startswith("|"):
            tbl += 1
    body = "\n".join(body_lines)
    return {
        "heads": heads,
        "blocks": blocks,
        "open_fence": fence is not None,
        "links": sorted(LINK.findall(body)),
        "inline": sorted(INLINE.findall(body)),
        "tbl": tbl,
        "alerts": ALERT.findall(body),
    }


def collect(d: Path):
    return {
        str(p.relative_to(d)): profile(p.read_text(encoding="utf-8"))
        for p in sorted(d.rglob("*.md"))
    }


def main():
    mode, d = sys.argv[1], Path(sys.argv[2])
    if mode == "baseline":
        Path(sys.argv[3]).write_text(json.dumps(collect(d), ensure_ascii=False))
        print(f"baseline: {len(collect(d))} files")
        return 0
    base = json.loads(Path(sys.argv[3]).read_text())
    cur = collect(d)
    bad = warn = 0
    for f, b in base.items():
        c = cur.get(f)
        if c is None:
            print(f"❌ {f}: 파일 없음")
            bad += 1
            continue
        if c["open_fence"]:
            print(f"❌ {f}: 코드펜스가 닫히지 않음")
            bad += 1
        if b["heads"] != c["heads"]:
            print(f"❌ {f}: 헤딩 시퀀스 불일치 ({len(b['heads'])}개 -> {len(c['heads'])}개)")
            bad += 1
        if b["blocks"] != c["blocks"]:
            bb, cb = b["blocks"], c["blocks"]
            if len(bb) != len(cb):
                print(f"❌ {f}: 코드블록 개수 {len(bb)} -> {len(cb)}")
            else:
                for i, (x, y) in enumerate(zip(bb, cb)):
                    if x != y:
                        print(f"❌ {f}: 코드블록#{i} 변경 (lang {x[0]!r}->{y[0]!r})")
            bad += 1
        if b["links"] != c["links"]:
            miss = set(b["links"]) - set(c["links"])
            extra = set(c["links"]) - set(b["links"])
            print(f"❌ {f}: 링크 타깃 불일치 없어짐={sorted(miss)[:4]} 생김={sorted(extra)[:4]}")
            bad += 1
        if b["inline"] != c["inline"]:
            cb, cc = Counter(b["inline"]), Counter(c["inline"])
            lost = {k: (cb[k], cc[k]) for k in cb if cc[k] < cb[k]}
            gain = {k: (cb[k], cc[k]) for k in cc if cc[k] > cb[k]}
            if lost:  # 원문에 있던 인라인 코드가 사라짐 = 번역돼 버린 것
                shown = ", ".join(f"{k} {v[0]}→{v[1]}" for k, v in list(lost.items())[:6])
                print(f"❌ {f}: 인라인코드 소실 ({len(lost)}종) {shown}")
                bad += 1
            if gain:  # 산문 단어를 코드로 감싼 것 — 무해
                shown = ", ".join(f"{k} {v[0]}→{v[1]}" for k, v in list(gain.items())[:6])
                print(f"⚠️  {f}: 인라인코드 추가 ({len(gain)}종) {shown}")
                warn += 1
        if b["tbl"] != c["tbl"]:
            print(f"❌ {f}: 표 행수 {b['tbl']} -> {c['tbl']}")
            bad += 1
        if b["alerts"] != c["alerts"]:
            print(f"❌ {f}: alert 마커 {b['alerts']} -> {c['alerts']}")
            bad += 1
    tail = f' / 경고 {warn}건' if warn else ''
    print(f"\n{'✅ 구조 보존(오류 0)' if bad == 0 else f'❌ {bad} 건 위반'}{tail}  ({len(base)} 파일)")
    return 1 if bad else 0


sys.exit(main())
