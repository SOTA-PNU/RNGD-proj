#!/usr/bin/env python3
"""에이전트가 쓴 슬라이드 내용 + 손으로 그린 도해를 합쳐 최종 pptx 를 만든다."""
import json
import re
import sys

import deck
import dia
import dia2

SRC = sys.argv[1]
OUT = sys.argv[2]

PART_META = {
    "P0-P1": ("0 · 1", "딥러닝 연산의 정체는 결국 행렬곱",
              "전체 지도를 먼저 보고, 행렬곱·GEMM·einsum·텐서 축약까지 쌓는다"),
    "P2": ("2", "왜 빠른 계산기만으로는 안 되나",
           "메모리 벽 · 연산강도 · Roofline"),
    "P3": ("3", "Tensor Core란 무엇인가",
           "MAC 하나에서 시작해 시스톨릭 어레이까지"),
    "P4": ("4", "타일링 (Tiling)",
           "이 발표의 축. 데이터를 재사용하려고 계산 순서를 바꾸는 기술"),
    "P5": ("5", "RNGD / TCP 하드웨어 구조",
           "Chip · Cluster · Slice · Lane 과 8단계 파이프라인"),
    "P6": ("6", "vISA란 무엇인가",
           "어느 높이의 언어이고, 무엇을 노출하고 무엇을 감추는가"),
    "P7": ("7", "매핑 m![] — vISA의 심장",
           "타일링·배치·병렬화를 한 줄에 적는 문법"),
    "P8": ("8", "커널 완전 해부",
           "constant_add 부터 GEMM · MNIST 까지 실제로 돌려본 코드"),
    "P9": ("9", "실행 모델과 스케줄링",
           "컨텍스트 · 해저드 · 이중 버퍼링 · 사이클 읽는 법"),
    "P10": ("10", "그래서 vISA로 뭘 할 수 있나",
            "우리 서버 실측 기준으로만 답한다"),
    "APP": ("부록", "치트시트 · 용어 · 자주 하는 실수", ""),
}

ORDER = ["P0-P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "APP"]

# 도해를 어느 슬라이드 뒤에 끼울지: (부, 제목에서 찾을 키워드들, 그리는 함수)
# 키워드가 안 맞으면 그 부의 끝에 붙인다.
DIAGRAMS = [
    ("P0-P1", ["지도", "목표", "무엇을"], dia.roadmap),
    ("P2", ["roofline", "루프라인", "천장"], dia.roofline),
    ("P3", ["시스톨릭", "systolic", "배열", "어레이"], dia.systolic),
    ("P4", ["블록", "타일 크기", "재사용", "행렬곱"], dia.gemm_tiling),
    ("P4", ["이중", "버퍼", "겹치", "프리페치"], dia.double_buffer),
    ("P5", ["계층", "hierarchy", "chip", "cluster"], dia.hierarchy),
    ("P5", ["메모리", "memory", "hbm", "dm"], dia.memory_tiers),
    ("P5", ["파이프라인", "엔진", "8단계"], dia.pipeline),
    ("P6", ["계층", "사다리", "추상화", "어느 높이"], dia.ladder),
    ("P7", ["stride", "modulo", "스트라이드", "모듈로", "타일"], dia.mapping_grid),
    ("P10", ["성능", "dma", "사이클", "최적화"], dia.dma_share),
    # --- 추가 도해 ---
    ("P0-P1", ["축약", "contraction", "브로드캐스트"], dia2.contraction3),
    ("P0-P1", ["gemm", "행렬 × 행렬", "행렬곱"], dia2.gemm_ladder),
    ("P0-P1", ["einsum", "표기"], dia2.einsum_read),
    ("P2", ["비싸", "이동", "에너지", "비용"], dia2.mem_cost),
    ("P2", ["연산강도", "arithmetic", "intensity"], dia2.intensity),
    ("P3", ["mac", "곱셈", "누산"], dia2.mac_unit),
    ("P3", ["정밀도", "bf16", "fp8", "dtype"], dia2.precision_bits),
    ("P4", ["패딩", "나눠떨어", "padding"], dia2.padding_lane),
    ("P5", ["flit", "packet", "패킷"], dia2.flit_packet),
    ("P6", ["백엔드", "컴파일", "빌드"], dia2.compile_stages),
    ("P7", ["pair", "축 합", "쌍"], dia2.mapping_pair),
    ("P7", ["padding", "패딩", "resize", "리사이즈"], dia2.mapping_padding),
    ("P8", ["mnist", "커널 구조", "해부"], dia2.kernel_flow),
    ("P9", ["컨텍스트", "context"], dia2.contexts_timeline),
    ("P9", ["해저드", "hazard", "raw"], dia2.hazards),
    ("P10", ["매트릭스", "실기", "결과", "통과"], dia2.results_matrix),
]


def norm(s):
    return re.sub(r"\s+", "", (s or "").lower())


def render(d, sl):
    lay = sl.get("layout", "bullets")
    title = sl.get("title", "")
    sub = sl.get("subtitle")
    co = sl.get("callout")
    note = sl.get("note")
    bl = sl.get("bullets") or []
    if lay == "section":
        return None  # 부 표지는 우리가 직접 만든다
    if lay == "big":
        s = d.big(title, sub or "")
        d._note(s, note)
        return s
    if lay == "code" and (sl.get("code") or {}).get("lines"):
        c = sl["code"]
        return d.code(title, c["lines"], c.get("caption"), bl, sub, co, note)
    if lay == "table" and (sl.get("table") or {}).get("headers"):
        t = sl["table"]
        rows = [[str(c) for c in r] for r in (t.get("rows") or [])]
        if rows:
            return d.table(title, [str(h) for h in t["headers"]], rows, sub, co, note)
        return d.bullets(title, bl, sub, co, note)
    if lay == "twocol" and sl.get("right"):
        heads = None
        if sub and "|" in sub:
            a, b = sub.split("|", 1)
            heads = [a.strip(), b.strip()]
            sub = None
        return d.twocol(title, bl, sl["right"], heads, sub, co, note)
    return d.bullets(title, bl, sub, co, note)



# ---------------------------------------------------------------- 자동 분할
from pptx.util import Inches as _In
import deck as _D


def _area(sl, code=False):
    """이 슬라이드의 본문 영역 (폭, 높이)."""
    top = _In(2.00) if sl.get("subtitle") else _In(1.66)
    h = _D.BODY_BOT - top - (_In(0.78) if sl.get("callout") else 0)
    if code:
        h = h * 0.38 - _In(0.24)          # 코드 박스가 62% 를 쓴다
    return _D.BODY_W, max(h, _In(0.4))


def _ratio(items, w, h):
    if not items:
        return 0.0
    return _D.Deck._fit(items, w, h)[1]


def split(sl):
    """안 들어가면 여러 장으로 쪼갠다."""
    lay = sl.get("layout", "bullets")
    bl = sl.get("bullets") or []

    if lay == "code" and (sl.get("code") or {}).get("lines"):
        out = [sl]
        # 1) 설명 불릿이 코드와 같이 안 들어가면 별도 장으로 뺀다
        if bl:
            w, h = _area(sl, code=True)
            if _ratio(bl, w, h) > 1.0:
                a = dict(sl); a["bullets"] = []
                a.pop("callout", None)
                b = {"layout": "bullets", "title": sl.get("title", "") + " — 설명",
                     "subtitle": sl.get("subtitle"), "bullets": bl,
                     "callout": sl.get("callout")}
                out = [a] + split(b)
        # 2) 코드 자체가 길면 여러 장으로 나눈다
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
                c = dict(first)
                c["code"] = dict(first["code"])
                c["code"]["lines"] = lines[k:k + cap]
                if part > 1:
                    c["title"] = first.get("title", "") + f" (계속 {part})"
                    c.pop("callout", None)
                    c.pop("bullets", None)
                    c["code"].pop("caption", None)
                chunks.append(c)
                k += cap
                part += 1
            out = chunks + out[1:]
        return out

    if lay == "table":
        t = sl.get("table") or {}
        rows = t.get("rows") or []
        if len(rows) > 14:
            out, i, part = [], 0, 1
            while i < len(rows):
                chunk = rows[i:i + 12]
                s2 = dict(sl)
                s2["table"] = {"headers": t.get("headers"), "rows": chunk}
                if part > 1:
                    s2["title"] = sl.get("title", "") + f" (계속 {part})"
                    s2.pop("callout", None)
                    s2.pop("bullets", None)
                out.append(s2)
                i += 12
                part += 1
            return out
        return [sl]

    if lay in ("bullets",) and bl:
        w, h = _area(sl)
        r = _ratio(bl, w, h)
        if r <= 1.0:
            return [sl]
        import math as _m
        n = min(_m.ceil(r + 0.15), 4)
        size = _m.ceil(len(bl) / n)
        out = []
        for k in range(0, len(bl), size):
            s2 = dict(sl)
            s2["bullets"] = bl[k:k + size]
            if k:
                s2["title"] = sl.get("title", "") + " (계속)"
                s2.pop("callout", None)
                s2.pop("subtitle", None)
            out.append(s2)
        return out

    return [sl]


def main():
    data = json.load(open(SRC, encoding="utf-8"))
    if isinstance(data, str):
        data = json.loads(data)
    by = {p["part"]: p.get("slides", []) for p in data if isinstance(p, dict)}

    d = deck.Deck(
        "vISA로 무엇을 할 수 있나",
        "행렬곱·타일링·Tensor Core 기초부터\nFuriosaAI RNGD NPU 커널 작성까지",
        "부산대 RNGD 서버 · 실측 기반 · 2026-07",
    )
    total_content = 0
    for key in ORDER:
        slides = by.get(key, [])
        num, ttl, stt = PART_META[key]
        d.part = ""
        d.section(num, ttl, stt)
        d.part = f"{num}부 · {ttl}" if key != "APP" else "부록"
        # 도해 삽입 위치를 제목으로 먼저 정한다 (렌더 순서를 바꾸기 위해)
        plan, used = {}, set()
        for pk, kws, fn in DIAGRAMS:
            if pk != key:
                continue
            idx = None
            for i, sl in enumerate(slides):
                if i in used:
                    continue
                t = norm(sl.get("title"))
                if any(norm(k) in t for k in kws):
                    idx = i
            if idx is None:
                idx = max(len(slides) - 1, 0)
            used.add(idx)
            plan.setdefault(idx, []).append(fn)
        for i, sl in enumerate(slides):
            for piece in split(sl):
                if render(d, piece) is not None:
                    total_content += 1
            for fn in plan.get(i, []):
                fn(d)
        if not slides:
            for fns in plan.values():
                for fn in fns:
                    fn(d)
    out = d.save(OUT)
    n = len(d.prs.slides._sldIdLst)
    print(f"부 {len(ORDER)}개, 본문 {total_content}장, 총 {n}장 → {out}")


main()

import deck as _dk
if _dk.OVERFLOW:
    print(f"\n⚠️  글자 넘침 의심 {len(_dk.OVERFLOW)}장 (최소 크기로도 안 들어감):")
    for pg, ti, ratio in sorted(_dk.OVERFLOW, key=lambda x: -x[2])[:25]:
        print(f"   {ratio:4.2f}배  p{pg}  {ti[:60]}")
else:
    print("\n✅ 글자 넘침 없음")
