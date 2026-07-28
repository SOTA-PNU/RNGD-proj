#!/usr/bin/env python3
"""pptx 안 XML 의 자식 요소 순서를 ECMA-376 스키마 순서와 대조한다.

PowerPoint 는 순서에 엄격해서, 순서를 어기면 파일을 손상으로 보고 복구를 요구한다.
LibreOffice·python-pptx 는 통과시키므로 이런 검사가 없으면 못 잡는다.
"""
import re
import sys
import zipfile
from collections import Counter

from lxml import etree

A = "http://schemas.openxmlformats.org/drawingml/2006/main"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"

# 요소 -> 자식이 나타나야 하는 순서(그룹 단위). 같은 그룹 안은 순서 무관.
ORDER = {
    f"{{{A}}}rPr": [("ln",), ("noFill", "solidFill", "gradFill", "blipFill", "pattFill", "grpFill"),
                    ("effectLst", "effectDag"), ("highlight",), ("uLnTx", "uLn"),
                    ("uFillTx", "uFill"), ("latin",), ("ea",), ("cs",), ("sym",),
                    ("hlinkClick",), ("hlinkMouseOver",), ("rtl",), ("extLst",)],
    f"{{{A}}}pPr": [("lnSpc",), ("spcBef",), ("spcAft",), ("buClrTx", "buClr"),
                    ("buSzTx", "buSzPct", "buSzPts"), ("buFontTx", "buFont"),
                    ("buNone", "buAutoNum", "buChar"), ("tabLst",), ("defRPr",), ("extLst",)],
    f"{{{A}}}bodyPr": [("prstTxWarp",), ("noAutofit", "normAutofit", "spAutoFit"),
                       ("scene3d",), ("sp3d", "flatTx"), ("extLst",)],
    f"{{{A}}}ln": [("noFill", "solidFill", "gradFill", "pattFill"), ("prstDash", "custDash"),
                   ("round", "bevel", "miter"), ("headEnd",), ("tailEnd",), ("extLst",)],
    f"{{{P}}}spPr": [("xfrm",), ("custGeom", "prstGeom"),
                     ("noFill", "solidFill", "gradFill", "blipFill", "pattFill", "grpFill"),
                     ("ln",), ("effectLst", "effectDag"), ("scene3d",), ("sp3d",), ("extLst",)],
    f"{{{A}}}xfrm": [("off",), ("ext",)],
    f"{{{A}}}p": [("pPr",), ("r", "br", "fld"), ("endParaRPr",)],
    f"{{{P}}}txBody": [("bodyPr",), ("lstStyle",), ("p",)],
    f"{{{A}}}txBody": [("bodyPr",), ("lstStyle",), ("p",)],
    f"{{{P}}}sp": [("nvSpPr",), ("spPr",), ("style",), ("txBody",)],
    f"{{{P}}}cxnSp": [("nvCxnSpPr",), ("spPr",), ("style",)],
    f"{{{A}}}tc": [("txBody",), ("tcPr",)],
}
# rPr 과 같은 순서를 쓰는 것들
for t in ("defRPr", "endParaRPr"):
    ORDER[f"{{{A}}}{t}"] = ORDER[f"{{{A}}}rPr"]


def group_index(spec, local):
    for i, grp in enumerate(spec):
        if local in grp:
            return i
    return None


def check_tree(root, where, bad, unknown):
    for el in root.iter():
        spec = ORDER.get(el.tag)
        if spec is None:
            continue
        last = -1
        seq = []
        for ch in el:
            if not isinstance(ch.tag, str):
                continue
            local = etree.QName(ch).localname
            gi = group_index(spec, local)
            seq.append(local)
            if gi is None:
                unknown[(etree.QName(el).localname, local)] += 1
                continue
            if gi < last:
                bad.append((where, etree.QName(el).localname, tuple(seq)))
                break
            last = gi


def main(path):
    z = zipfile.ZipFile(path)
    bad, unknown = [], Counter()
    parts = [n for n in z.namelist() if n.endswith(".xml") and
             (n.startswith("ppt/slides/") or n.startswith("ppt/slideLayouts/")
              or n.startswith("ppt/slideMasters/") or n == "ppt/presentation.xml"
              or n.startswith("ppt/notesSlides/"))]
    for n in sorted(parts):
        try:
            root = etree.fromstring(z.read(n))
        except Exception as e:
            bad.append((n, "PARSE", str(e)))
            continue
        check_tree(root, n, bad, unknown)

    # 필수 관계·중복 id 점검
    ids = Counter()
    for n in parts:
        if not n.startswith("ppt/slides/slide"):
            continue
        root = etree.fromstring(z.read(n))
        for el in root.iter(f"{{{P}}}cNvPr"):
            ids[(n, el.get("id"))] += 1
    dup = [k for k, v in ids.items() if v > 1]

    print(f"검사한 XML 파트: {len(parts)}개")
    if bad:
        print(f"\n❌ 요소 순서 위반 {len(bad)}건")
        c = Counter((b[1], b[2]) for b in bad if b[1] != "PARSE")
        for (parent, seq), cnt in c.most_common(12):
            print(f"   {cnt:5d}회  <{parent}> 자식순서 {seq}")
        for b in bad:
            if b[1] == "PARSE":
                print("   파싱 실패:", b[0], b[2][:120])
    else:
        print("✅ 요소 순서 위반 없음")
    if dup:
        print(f"❌ 슬라이드 내 중복 shape id {len(dup)}건: {dup[:5]}")
    else:
        print("✅ shape id 중복 없음")
    if unknown:
        print(f"\n참고 — 스키마 표에 없는 자식 {len(unknown)}종 (오류 아닐 수 있음):")
        for k, v in unknown.most_common(8):
            print(f"   {v:5d}  <{k[0]}> 안의 <{k[1]}>")
    return 1 if (bad or dup) else 0


sys.exit(main(sys.argv[1]))
