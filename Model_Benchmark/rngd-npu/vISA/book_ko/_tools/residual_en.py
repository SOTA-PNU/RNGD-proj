#!/usr/bin/env python3
"""번역 누락(영문 산문 잔존) 검출.

코드펜스·인라인코드·링크타깃·HTML·이미지경로를 제거한 뒤,
남은 텍스트가 연속 영단어 N개 이상이면 미번역으로 본다.
"""
import re, sys
from pathlib import Path

FENCE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)?(`{3,}|~{3,})")
MIN_WORDS = int(sys.argv[2]) if len(sys.argv) > 2 else 7

# 영문으로 남는 게 정상인 고유명사 — 이것만으로 이뤄진 줄은 무시
KEEP = r"""Fetch|Switch|Collect|Contraction|Vector|Cast|Transpose|Commit|Adapter|Engine|
Sequencer|Valid|Count|Generator|Tensor|Unit|Outer|Packet|Reducer|Time|Lane|Folder|
Intra|Inter|Slice|Chain|Stream|Chip|Cluster|Element|HBM|DM|SPM|TRF|VRF|flit|
TCP|RNGD|vISA|EDF|DPE|MAC|PE|DMA|PCIe|MoE|GEMM|GEMV|NPU|SRAM|SDK|API|ISA|IR|
Mode|Tag|Logic|Fxp|Narrow|Pair|Stage|Stages|Interface|Register|File|Files|Memory"""
KEEP_RE = re.compile(r"^(?:(?:%s)\W*)+$" % KEEP.replace("\n", ""), re.X)


def clean(ln: str) -> str:
    ln = re.sub(r"`[^`]*`", " ", ln)                 # 인라인 코드
    ln = re.sub(r"\]\([^)]*\)", "] ", ln)            # 링크 타깃
    ln = re.sub(r"<[^>]+>", " ", ln)                 # HTML 태그
    ln = re.sub(r"https?://\S+", " ", ln)
    ln = re.sub(r"[|>#*\-\[\]!]", " ", ln)           # 마크다운 기호
    return ln


def main():
    root = Path(sys.argv[1])
    total = 0
    for p in sorted(root.rglob("*.md")):
        fence, hits = None, []
        for i, ln in enumerate(p.read_text(encoding="utf-8").split("\n"), 1):
            m = FENCE.match(ln)
            if fence is not None:
                if m and m.group(1)[0] == fence[0]:
                    fence = None
                continue
            if m:
                fence = m.group(1)
                continue
            if re.search(r"[가-힣]", ln):             # 한글이 있으면 번역된 줄
                continue
            t = clean(ln).strip()
            if not t:
                continue
            words = re.findall(r"\b[A-Za-z][A-Za-z'’]{1,}\b", t)
            if len(words) < MIN_WORDS:
                continue
            if KEEP_RE.match(t.strip()):
                continue
            hits.append((i, ln.strip()[:110]))
        if hits:
            total += len(hits)
            print(f"\n── {p.relative_to(root)}  ({len(hits)}줄)")
            for i, s in hits:
                print(f"   {i:5d}: {s}")
    print(f"\n{'✅ 잔존 영문 산문 없음' if total == 0 else f'⚠️  미번역 의심 {total}줄'}")
    return 1 if total else 0


sys.exit(main())
