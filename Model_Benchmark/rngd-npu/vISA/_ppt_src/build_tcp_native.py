#!/usr/bin/env python3
"""ISCA 2024 TCP 논문 분석 PPT 조립기."""
import json
import math
import os
import re
import sys

from pptx.util import Inches as _In

import deck
import deck as _D

SRC, OUT = sys.argv[1], sys.argv[2]
DIA = sys.argv[3] if len(sys.argv) > 3 else None      # 직접 그린 도해 index.json
FIGDIR = "/home/jun/.claude/jobs/46bc5c7e/tmp/tcp_paper"

PART_META = {
    "A1": ("0·1", "논문 소개와 서론", "무슨 논문이고, 왜 우리에게 중요하며, 어떤 문제를 풀겠다는 것인가"),
    "A2": ("2", "PRELIMINARIES — 논문의 어휘", "low-level einsum · lowered shape · tactic"),
    "A3": ("3", "SYSTEM-ON-CHIP", "PE 8개 · NoC · HBM3 · 칩 제원"),
    "A4": ("4", "MICRO-ARCHITECTURE (1)", "PE · 슬라이스 · 컨텍스트 · Fetch Network"),
    "A5": ("5", "MICRO-ARCHITECTURE (2)", "Contraction Engine · Vector Engine"),
    "A6": ("6", "PROGRAMMING INTERFACE", "컴파일러 스택과 IR"),
    "A7": ("7", "CASE STUDY (1)", "LLaMA-2 7B 매핑과 성능 비교"),
    "A8": ("8", "CASE STUDY (2)", "성능·전력 분석 그래프 5종"),
    "A9": ("9", "LESSONS LEARNED · CONCLUSION", "칩을 만들며 배운 것"),
    "A10": ("부록", "우리 서버 실측과의 대조", "논문이 말한 것과 우리가 측정한 것"),
}
ORDER = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]

DIA_STAT = []
SRC_LABEL = {}
for i in range(1, 16):
    SRC_LABEL[f"fig{i}"] = f"출처: Fig. {i} — TCP: A Tensor Contraction Processor for AI Workloads (ISCA 2024)"
SRC_LABEL["tableI"] = "출처: TABLE I — TCP (ISCA 2024)"
SRC_LABEL["tableII"] = "출처: TABLE II — TCP (ISCA 2024)"


def figpath(key):
    if not key:
        return None
    k = re.sub(r"[^A-Za-z0-9]", "", str(key))
    k = k.replace("Fig", "fig").replace("Table", "table").replace("TABLE", "table")
    for cand in (k, k.lower(), k.replace("table", "table").upper()):
        p = os.path.join(FIGDIR, f"isca_{cand}.png")
        if os.path.exists(p):
            return p, cand
    m = re.match(r"(fig|table)(.+)", k, re.I)
    if m:
        p = os.path.join(FIGDIR, f"isca_{m.group(1).lower()}{m.group(2)}.png")
        if os.path.exists(p):
            return p, f"{m.group(1).lower()}{m.group(2)}"
    return None


def _area(sl, code=False):
    top = _In(2.00) if sl.get("subtitle") else _In(1.66)
    h = _D.BODY_BOT - top - (_In(0.78) if sl.get("callout") else 0)
    if code:
        h = h * 0.38 - _In(0.24)
    return _D.BODY_W, max(h, _In(0.4))


def _ratio(items, w, h):
    return _D.Deck._fit(items, w, h)[1] if items else 0.0


def split(sl):
    lay = sl.get("layout", "bullets")
    bl = sl.get("bullets") or []
    if lay == "diagram":
        # 도해는 세로 공간이 곧 그림 안 글자 크기다. 슬라이드 하나를 통째로 주고
        # 설명 불릿은 바로 뒤 슬라이드로 뺀다(핵심 한 줄만 그림과 같은 장에 남긴다).
        a = dict(sl); a["bullets"] = []
        if not bl:
            return [a]
        b = {"layout": "bullets", "title": sl.get("title", "") + " — 설명",
             "subtitle": sl.get("subtitle"), "bullets": bl,
             "note": sl.get("note")}
        return [a] + split(b)
    if lay == "figure":
        r = figpath(sl.get("fig"))
        if not r or not bl:
            return [sl]
        from PIL import Image as _Img
        iw, ih = _Img.open(r[0]).size
        ar = iw / ih
        top = _In(2.00) if sl.get("subtitle") else _In(1.66)
        avail = _D.BODY_BOT - top - (_In(0.78) if sl.get("callout") else 0)
        cap_h = _In(0.24)
        if ar >= 2.55:                       # 그림이 위, 설명이 아래
            box_h = avail * 0.62 - cap_h
            w = min(_D.BODY_W, box_h * ar)
            bw, bh = _D.BODY_W, avail - (w / ar) - cap_h - _In(0.14)
        else:                                # 그림이 왼쪽, 설명이 오른쪽
            rx = _D.ML + _D.BODY_W * 0.54 + _In(0.30)
            bw, bh = _D.SW - _D.MR - rx, avail
        if bh < _In(0.35) or _ratio(bl, bw, bh) > 1.0:
            keep, rest = [], list(bl)
            while rest and (not keep or _ratio(keep + rest[:1], bw, bh) <= 1.0):
                keep.append(rest.pop(0))
            if not keep:                     # 설명을 아예 못 넣는 경우
                keep, rest = [], list(bl)
            a = dict(sl); a["bullets"] = keep; a.pop("callout", None)
            b = {"layout": "bullets", "title": sl.get("title", "") + " — 설명",
                 "subtitle": sl.get("subtitle"), "bullets": rest,
                 "callout": sl.get("callout"), "note": sl.get("note")}
            return [a] + (split(b) if rest else [])
        return [sl]
    if lay == "code" and (sl.get("code") or {}).get("lines"):
        out = [sl]
        if bl:
            w, h = _area(sl, code=True)
            if _ratio(bl, w, h) > 1.0:
                a = dict(sl); a["bullets"] = []; a.pop("callout", None)
                b = {"layout": "bullets", "title": sl.get("title", "") + " — 설명",
                     "subtitle": sl.get("subtitle"), "bullets": bl, "callout": sl.get("callout")}
                out = [a] + split(b)
        first = out[0]
        lines = first["code"]["lines"]
        top = _In(2.00) if first.get("subtitle") else _In(1.66)
        avail = _D.BODY_BOT - top - (_In(0.78) if first.get("callout") else 0)
        bh = avail if not (first.get("bullets") or []) else avail * 0.62
        if first["code"].get("caption"):
            bh -= _In(0.32)
        cap = _D.code_rows(_D.code_size(lines, bh), bh)
        if len(lines) > cap:
            chunks, k, part = [], 0, 1
            while k < len(lines):
                c = dict(first); c["code"] = dict(first["code"])
                c["code"]["lines"] = lines[k:k + cap]
                if part > 1:
                    c["title"] = first.get("title", "") + f" (계속 {part})"
                    c.pop("callout", None); c.pop("bullets", None); c["code"].pop("caption", None)
                chunks.append(c); k += cap; part += 1
            out = chunks + out[1:]
        return out
    if lay == "table":
        t = sl.get("table") or {}
        rows = t.get("rows") or []
        if len(rows) > 14:
            out, i, part = [], 0, 1
            while i < len(rows):
                s2 = dict(sl)
                s2["table"] = {"headers": t.get("headers"), "rows": rows[i:i + 12]}
                if part > 1:
                    s2["title"] = sl.get("title", "") + f" (계속 {part})"
                    s2.pop("callout", None); s2.pop("bullets", None)
                out.append(s2); i += 12; part += 1
            return out
        return [sl]
    if lay == "bullets" and bl:
        w, h = _area(sl)
        r = _ratio(bl, w, h)
        if r <= 1.0:
            return [sl]
        n = min(math.ceil(r + 0.15), 4)
        size = math.ceil(len(bl) / n)
        out = []
        for k in range(0, len(bl), size):
            s2 = dict(sl); s2["bullets"] = bl[k:k + size]
            if k:
                s2["title"] = sl.get("title", "") + " (계속)"
                s2.pop("callout", None); s2.pop("subtitle", None)
            out.append(s2)
        return out
    return [sl]


def render(d, sl, used):
    lay = sl.get("layout", "bullets")
    title = sl.get("title", "")
    sub = sl.get("subtitle")
    co = sl.get("callout")
    note = sl.get("note")
    bl = sl.get("bullets") or []
    if lay == "section":
        return None
    if lay == "diagram":
        r = d.diagram_svg(title, sl["svg"], bl, sub, co, note,
                          name="도해 " + str(sl.get("dia_id", "")))
        DIA_STAT.append((d.n, title, r[1], r[2]))
        return r[0]
    if lay == "figure":
        r = figpath(sl.get("fig"))
        if not r:
            return d.bullets(title, bl, sub, co, note)
        path, key = r
        used.add(key)
        return d.figure(title, path, SRC_LABEL.get(key), bl, sub, co, note)
    if lay == "big":
        s = d.big(title, sub or ""); d._note(s, note); return s
    if lay == "code" and (sl.get("code") or {}).get("lines"):
        c = sl["code"]
        return d.code(title, c["lines"], c.get("caption"), bl, sub, co, note)
    if lay == "table" and (sl.get("table") or {}).get("headers"):
        t = sl["table"]
        rows = [[str(c) for c in r] for r in (t.get("rows") or [])]
        if rows:
            return d.table(title, [str(h) for h in t["headers"]], rows, sub, co, note)
    if lay == "twocol" and sl.get("right"):
        heads = None
        if sub and "|" in sub:
            a, b = sub.split("|", 1); heads = [a.strip(), b.strip()]; sub = None
        return d.twocol(title, bl, sl["right"], heads, sub, co, note)
    return d.bullets(title, bl, sub, co, note)


def main():
    data = json.load(open(SRC, encoding="utf-8"))
    if isinstance(data, str):
        data = json.loads(data)
    by = {p["part"]: p.get("slides", []) for p in data if isinstance(p, dict)}

    # 직접 그린 도해로 해당 슬라이드를 통째로 갈아끼운다
    swapped, skipped = 0, []
    if DIA and os.path.exists(DIA):
        raw = json.load(open(DIA, encoding="utf-8"))
        if isinstance(raw, str):
            raw = json.loads(raw)
        flat = []
        for grp in raw:
            for x in grp.get("items", []):
                x = dict(x); x["part"] = grp["part"]; flat.append(x)
        for it in flat:
            lst = by.get(it["part"])
            if not lst or not (0 <= it["index"] < len(lst)):
                skipped.append(f"{it['part']}#{it['index']} 인덱스 벗어남")
                continue
            cur = lst[it["index"]]
            if cur.get("layout") in ("figure", "section"):
                skipped.append(f"{it['part']}#{it['index']} 논문그림/표지라 보존")
                continue
            lst[it["index"]] = {
                "layout": "diagram", "svg": it["svg"],
                "dia_id": f"{it['part']}#{it['index']:02d}",
                "title": it.get("title") or cur.get("title", ""),
                "subtitle": it.get("subtitle") or cur.get("subtitle"),
                "bullets": it.get("bullets") or [],
                "callout": it.get("callout"),
                "note": it.get("note") or cur.get("note"),
            }
            swapped += 1

    d = deck.Deck(
        "TCP 논문 완전 분석",
        "TCP: A Tensor Contraction Processor for AI Workloads\nFuriosaAI · ISCA 2024 (Industry Track) — 우리 서버 RNGD 의 아키텍처 원논문",
        "논문 전문 + 그림 15개 · 표 2개 전량 수록 · 2026-07",
    )
    used = set()
    n = 0
    for key in ORDER:
        num, ttl, stt = PART_META[key]
        d.part = ""
        d.section(num, ttl, stt)
        d.part = f"{num}부 · {ttl}" if key != "A10" else "부록 · 우리 서버 실측과의 대조"
        for sl in by.get(key, []):
            for piece in split(sl):
                if render(d, piece, used) is not None:
                    n += 1
    # 빠진 그림이 있으면 부록에 모아 넣는다 (누락 방지)
    allkeys = [f"fig{i}" for i in range(1, 16)] + ["tableI", "tableII"]
    missing = [k for k in allkeys if k not in used]
    if missing:
        d.part = "부록 · 논문 그림 전량"
        for k in missing:
            r = figpath(k)
            if r:
                path, kk = r
                d.figure(f"{'Fig. ' + kk[3:] if kk.startswith('fig') else 'TABLE ' + kk[5:]} (본문 미배치분)",
                         path, SRC_LABEL.get(kk), [], None, None, None)
                n += 1
    d.save(OUT)
    total = len(d.prs.slides._sldIdLst)
    print(f"부 {len(ORDER)}개, 본문 {n}장, 총 {total}장 → {OUT}")
    print(f"직접 그린 도해로 교체: {swapped}장")
    for s_ in skipped:
        print(f"   ⚠ {s_}")
    print(f"논문 그림 사용: {len(used)}/17" + (f"   부록으로 보완: {missing}" if missing else "   (전량 본문 배치)"))
    if DIA_STAT:
        mins = [x[3] for x in DIA_STAT]
        import statistics as _st
        print(f"\n네이티브 도해 {len(DIA_STAT)}장 · 도형 {sum(x[2] for x in DIA_STAT)}개")
        print(f"   그림 안 최소 글자: 중앙 {_st.median(mins):.1f}pt "
              f"(최소 {min(mins):.1f} / 최대 {max(mins):.1f})")
        small = [x for x in DIA_STAT if x[3] < 6.5]
        for pg, ti, n_, mp in small:
            print(f"   ⚠ p{pg} {mp:.1f}pt  {ti[:52]}")
    if deck.OVERFLOW:
        print(f"\n⚠️  넘침 {len(deck.OVERFLOW)}장")
        for pg, ti, ratio in sorted(deck.OVERFLOW, key=lambda x: -x[2])[:15]:
            print(f"   {ratio:4.2f}배  p{pg}  {ti[:58]}")
    else:
        print("\n✅ 글자 넘침 없음")


main()
