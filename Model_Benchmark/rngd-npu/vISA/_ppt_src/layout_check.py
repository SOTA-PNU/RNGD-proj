#!/usr/bin/env python3
"""슬라이드 레이아웃 점검 — 글자 넘침·도형 겹침·경계 이탈.

python-pptx 는 텍스트를 실제로 조판하지 않으므로, 글자 폭/높이를 추정해서 검사한다.
추정이라 완벽하진 않지만 '명백히 넘치는 것'은 잡는다.
"""
import sys
from collections import defaultdict

from pptx import Presentation
from pptx.util import Emu

EMU_IN = 914400.0
PT_EMU = 12700.0


def vlen(s):
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def est_size(shape):
    """(필요한 높이 pt, 가장 긴 줄의 폭 pt) 추정."""
    tf = shape.text_frame
    w_emu = shape.width or 0
    ml = tf.margin_left or 0
    mr = tf.margin_right or 0
    w_pt = max((w_emu - ml - mr) / PT_EMU, 10)
    total_h = 0.0
    max_w = 0.0
    for p in tf.paragraphs:
        sizes = [r.font.size.pt for r in p.runs if r.font.size is not None]
        fs = max(sizes) if sizes else 18.0
        mono = any((r.font.name or "") == "Consolas" for r in p.runs)
        per = fs * (0.55 if mono else 0.5)      # 반각 1칸의 폭
        # p.text 는 <a:br/> 를 '\v' 로 준다 — 강제 줄바꿈으로 센다
        segs = (p.text or "").split("\v")
        lines = 0
        for seg in segs:
            line_w = vlen(seg) * per
            max_w = max(max_w, line_w)
            lines += max(1, int(line_w / w_pt) + (1 if line_w % w_pt else 0)) if seg else 1
        ls = p.line_spacing if isinstance(p.line_spacing, float) else 1.2
        sb = p.space_before.pt if p.space_before is not None else 0
        sa = p.space_after.pt if p.space_after is not None else 0
        total_h += lines * fs * ls + sb + sa
    mt = (tf.margin_top or 0) / PT_EMU
    mb = (tf.margin_bottom or 0) / PT_EMU
    return total_h + mt + mb, max_w


def boxes(slide):
    out = []
    for sh in slide.shapes:
        if sh.left is None or sh.top is None:
            continue
        txt = sh.text_frame.text.strip() if sh.has_text_frame else ""
        out.append({
            "sh": sh, "txt": txt,
            "l": sh.left, "t": sh.top,
            "r": sh.left + (sh.width or 0), "b": sh.top + (sh.height or 0),
        })
    return out


def overlap(a, b):
    ox = min(a["r"], b["r"]) - max(a["l"], b["l"])
    oy = min(a["b"], b["b"]) - max(a["t"], b["t"])
    if ox <= 0 or oy <= 0:
        return 0.0
    inter = ox * oy
    amin = min((a["r"] - a["l"]) * (a["b"] - a["t"]), (b["r"] - b["l"]) * (b["b"] - b["t"]))
    return inter / amin if amin else 0.0


def main(path):
    prs = Presentation(path)
    W, H = prs.slide_width, prs.slide_height
    over_h, over_w, collide, outside = [], [], [], []
    for i, s in enumerate(prs.slides, 1):
        bs = boxes(s)
        for b in bs:
            sh = b["sh"]
            if b["l"] < -Emu(5000) or b["t"] < -Emu(5000) or b["r"] > W + Emu(20000) or b["b"] > H + Emu(20000):
                outside.append((i, b["txt"][:40], round(b["r"] / EMU_IN, 2), round(b["b"] / EMU_IN, 2)))
            if not b["txt"] or not sh.has_text_frame:
                continue
            need_h, need_w = est_size(sh)
            box_h = (sh.height or 0) / PT_EMU
            box_w = (sh.width or 0) / PT_EMU
            # 텍스트박스는 아래로 흘러넘쳐 아래 도형을 덮는다
            if need_h > box_h * 1.25 and need_h - box_h > 6:
                over_h.append((i, b["txt"][:44], round(need_h, 1), round(box_h, 1)))
            # 도형 라벨이 한 줄인데 폭을 넘으면 좌우로 삐져나온다
            if len(sh.text_frame.paragraphs) == 1 and need_w > box_w * 1.02 and box_h < 34:
                over_w.append((i, b["txt"][:44], round(need_w, 1), round(box_w, 1)))
        # 글자 있는 도형끼리 심하게 겹치는지
        tb = [b for b in bs if b["txt"]]
        for x in range(len(tb)):
            for y in range(x + 1, len(tb)):
                r = overlap(tb[x], tb[y])
                if r > 0.55:
                    collide.append((i, tb[x]["txt"][:26], tb[y]["txt"][:26], round(r, 2)))

    def rep(name, items, fmt):
        print(f"\n{'✅' if not items else '⚠️ '} {name}: {len(items)}건")
        for it in items[:18]:
            print("   " + fmt(it))
        if len(items) > 18:
            print(f"   … 외 {len(items)-18}건")

    print(f"슬라이드 {len(prs.slides._sldIdLst)}장 검사")
    rep("슬라이드 경계 이탈", outside, lambda x: f"p{x[0]:3d} R{x[2]} B{x[3]}  '{x[1]}'")
    rep("세로 넘침(아래 도형을 덮을 수 있음)", over_h,
        lambda x: f"p{x[0]:3d} 필요 {x[2]}pt / 박스 {x[3]}pt  '{x[1]}'")
    rep("가로 넘침(라벨이 도형 밖으로)", over_w,
        lambda x: f"p{x[0]:3d} 필요 {x[2]}pt / 박스 {x[3]}pt  '{x[1]}'")
    rep("글자 있는 도형끼리 겹침", collide,
        lambda x: f"p{x[0]:3d} {x[3]:.0%}  '{x[1]}' ↔ '{x[2]}'")
    bad = len(outside) + len(over_h) + len(over_w) + len(collide)
    print(f"\n{'✅ 레이아웃 문제 없음' if bad == 0 else f'총 {bad}건 확인 필요'}")
    return 1 if bad else 0


sys.exit(main(sys.argv[1]))
