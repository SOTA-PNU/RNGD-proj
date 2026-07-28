#!/usr/bin/env python3
"""그림·표 추출 v3 — 본문 글꼴(10pt) 라인만 경계로 쓴다.

그림 안 라벨은 4~6pt, 캡션은 9pt 라서 본문(>=9.5pt)만 경계로 삼으면
그림 영역이 정확히 잘린다.
"""
import json
import re
import sys

import fitz

SRC, OUT = sys.argv[1], sys.argv[2]
DPI = 300
BODY = 9.5
CAP = re.compile(r"(Fig\.?|Figure|TABLE|Table)\s*([IVXLC]+|\d+)\s*:", re.I)  # 콜론만 진짜 캡션
LCOL = (48, 303)
RCOL = (306, 566)
TOP, BOT = 46.0, 745.0

doc = fitz.open(SRC)
found = {}

for pno, page in enumerate(doc):
    W, H = page.rect.width, page.rect.height
    lines = []
    for blk in page.get_text("dict")["blocks"]:
        if blk.get("type") != 0:
            continue
        for ln in blk["lines"]:
            txt = "".join(sp["text"] for sp in ln["spans"])
            mx = max((sp["size"] for sp in ln["spans"]), default=0)
            lines.append({"t": txt.strip(), "bbox": ln["bbox"], "sz": mx})
    caps = []
    for ln in lines:
        m = CAP.match(ln["t"])
        if m and len(ln["t"]) > len(m.group(0)) + 3:
            caps.append({"kind": "table" if m.group(1).lower().startswith("tab") else "fig",
                         "num": m.group(2).upper(), "bbox": list(ln["bbox"]), "text": " ".join(ln["t"].split())})
    for c in caps:
        cx0, cy0, cx1, cy1 = c["bbox"]
        wide = (cx1 - cx0) > W * 0.55
        if wide:
            col = (LCOL[0], RCOL[1])
        else:
            col = LCOL if (cx0 + cx1) / 2 < W / 2 else RCOL

        colw = col[1] - col[0]

        def in_col(bb):
            return bb[2] > col[0] - 4 and bb[0] < col[1] + 4

        def is_body(ln):
            """경계로 쓸 수 있는 본문 줄 — 짧은 기호 조각은 제외한다."""
            bb = ln["bbox"]
            return ln["sz"] >= BODY and in_col(bb) and (bb[2] - bb[0]) >= 55

        def is_para(ln):
            """단 폭을 거의 채우는 본문 줄 (표 아래 경계용)."""
            bb = ln["bbox"]
            return ln["sz"] >= BODY and in_col(bb) and (bb[2] - bb[0]) >= colw * 0.78

        # 캡션이 여러 줄이면 전체를 캡션으로 본다
        cap_bot = cy1
        for _ in range(4):   # 캡션이 여러 줄이면 이어 붙인다
            for ln in lines:
                if in_col(ln["bbox"]) and cap_bot - 1 <= ln["bbox"][1] <= cap_bot + 4 \
                        and ln["bbox"][3] > cap_bot:
                    cap_bot = ln["bbox"][3]

        if c["kind"] == "fig":       # 그림은 캡션 위
            above = [ln for ln in lines if is_body(ln) and ln["bbox"][3] <= cy0 - 2]
            top = max([ln["bbox"][3] for ln in above], default=TOP)
            rect = (col[0] - 2, top + 4, col[1] + 2, cy0 - 2)
        else:                        # 표는 캡션 아래
            below = [ln for ln in lines if is_para(ln) and ln["bbox"][1] >= cap_bot + 2]
            bot = min([ln["bbox"][1] for ln in below], default=BOT)
            rect = (col[0] - 2, cap_bot + 2, col[1] + 2, bot - 4)

        if rect[3] - rect[1] < 24 or rect[2] - rect[0] < 60:
            continue
        key = (c["kind"], c["num"])
        area = (rect[2] - rect[0]) * (rect[3] - rect[1])
        if key not in found or area > found[key]["area"]:
            found[key] = {"kind": c["kind"], "num": c["num"], "page": pno, "wide": wide,
                          "rect": [round(v, 1) for v in rect], "area": area,
                          "cap_rect": [round(v, 1) for v in (col[0] - 2, cy0 - 2, col[1] + 2, cap_bot + 2)],
                          "text": c["text"][:220]}

ROM = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
order = sorted(found.values(),
               key=lambda it: (0 if it["kind"] == "fig" else 1,
                               ROM.get(it["num"], int(it["num"]) if it["num"].isdigit() else 99)))
zoom = DPI / 72.0
for it in order:
    pix = doc[it["page"]].get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=fitz.Rect(*it["rect"]))
    it["file"] = f"{OUT}_{it['kind']}{it['num']}.png"
    pix.save(it["file"])
    it["px"] = [pix.width, pix.height]
    print(f"{it['file']:20s} p{it['page']+1:2d} {'wide' if it['wide'] else '    '} "
          f"{pix.width:4d}x{pix.height:4d}px  {it['text'][:58]}")

json.dump(order, open(f"{OUT}_index.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n총 {len(order)}개")
