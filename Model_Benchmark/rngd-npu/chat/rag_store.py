#!/usr/bin/env python3
"""선택적 RAG(검색 증강 생성) — furiosa-apps/rag(kotaemon) 가 보여주는 패턴을 우리 채팅에 이식.

kotaemon 원본은 furiosa-llm 으로 embedding(Qwen3-Embedding-8B)·reranker(Qwen3-Reranker-8B)·LLM 을
각 카드에 serve 하고 문서를 임베딩해 검색한다. 우리 환경엔 임베딩/리랭커 아티팩트가 없고(카드도 귀함)
chat venv 에 sentence-transformers/sklearn 도 없으므로, **두 가지 백엔드를 갖춘 플러그형**으로 만든다:

  1. 기본(항상 동작): **TF-IDF 코사인 검색** — numpy 만으로, NPU·다운로드 없이 업로드 문서에서 검색.
  2. 선택(furiosa 방식): 환경변수 `CHAT_EMBED_URL`(OpenAI 호환 /v1) 이 있으면 **의미 임베딩**으로 검색,
     `CHAT_RERANK_URL`(furiosa TeiFastReranking /v1/rerank) 이 있으면 상위 후보를 **리랭킹**.

"rag 기능은 필요하면 사용할 수 있도록" — UI 토글로 켤 때만 동작하고, 켜져 있어도 문서가 없으면 무해.
"""
import math
import os
import re
import threading
from collections import Counter

import httpx

EMBED_URL = os.environ.get("CHAT_EMBED_URL", "").rstrip("/")     # 예: http://127.0.0.1:8021/v1
EMBED_MODEL = os.environ.get("CHAT_EMBED_MODEL", "")             # 비면 서버의 첫 모델 id 사용
RERANK_URL = os.environ.get("CHAT_RERANK_URL", "").rstrip("/")   # 예: http://127.0.0.1:8022/v1/rerank

_WORD = re.compile(r"[A-Za-z0-9_]+|[가-힣]+")  # 영문/숫자/한글 토큰


def _tokenize(s):
    return _WORD.findall(s.lower())


def _chunk(text, size=900, overlap=150):
    """문단 경계를 존중하며 ~size 글자 청크로 자른다(겹침 overlap)."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= size:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            if cur:
                chunks.append(cur)
            if len(p) <= size:
                cur = p
            else:  # 한 문단이 너무 길면 글자 윈도로 쪼갬
                for i in range(0, len(p), size - overlap):
                    chunks.append(p[i:i + size])
                cur = ""
    if cur:
        chunks.append(cur)
    return chunks


def _strip_html(raw):
    raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = re.sub(r"&nbsp;", " ", raw)
    raw = re.sub(r"&amp;", "&", raw)
    raw = re.sub(r"&lt;", "<", raw)
    raw = re.sub(r"&gt;", ">", raw)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", raw)).strip()


def _cos_sparse(a, b):
    """두 sparse(dict) 벡터 코사인. 노름은 미리 곱해 넣지 않고 여기서 계산."""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(w * b.get(t, 0.0) for t, w in a.items())
    if dot == 0.0:
        return 0.0
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class RagStore:
    """업로드/URL/붙여넣기 문서를 청킹·인덱싱하고 질문에 관련 청크를 돌려준다(스레드 안전)."""

    def __init__(self):
        self._lock = threading.RLock()
        self.docs = {}          # name -> {"chunk_ids": [...], "n": chars}
        self.chunks = []        # [{"text", "doc", "tf": Counter, "vec": dict|None, "emb": list|None}]
        self.idf = {}
        self._embed_ok = None   # None=미시도, True/False=가용성 캐시

    # ── 백엔드 가용성 ────────────────────────────────────────────────────
    @property
    def backend(self):
        return "embedding" if (EMBED_URL and self._embed_ok is not False) else "tf-idf"

    def _embed(self, texts):
        """furiosa 임베딩 서버(OpenAI 호환)로 임베딩. 실패하면 None 반환(→ TF-IDF 폴백)."""
        if not EMBED_URL:
            return None
        try:
            model = EMBED_MODEL
            if not model:
                ml = httpx.get(f"{EMBED_URL}/models", timeout=5).json()
                model = ml["data"][0]["id"]
            r = httpx.post(f"{EMBED_URL}/embeddings",
                           json={"model": model, "input": texts}, timeout=120)
            r.raise_for_status()
            vecs = [d["embedding"] for d in r.json()["data"]]
            self._embed_ok = True
            return vecs
        except Exception:
            self._embed_ok = False
            return None

    # ── 인덱싱 ──────────────────────────────────────────────────────────
    def _reindex(self):
        """전체 청크의 TF-IDF idf·tf-idf 벡터 재계산(문서 추가/삭제 후)."""
        n = len(self.chunks)
        df = Counter()
        for c in self.chunks:
            df.update(set(c["tf"]))
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        for c in self.chunks:
            c["vec"] = {t: f * self.idf.get(t, 0.0) for t, f in c["tf"].items()}

    def add(self, name, text):
        """문서 1건 추가(청킹+인덱싱). 임베딩 서버가 있으면 임베딩도 함께 저장."""
        text = (text or "").strip()
        if not text:
            return 0
        pieces = _chunk(text)
        if not pieces:
            return 0
        embs = self._embed(pieces)  # None 이면 TF-IDF 만
        with self._lock:
            base = name
            i = 2
            while name in self.docs:  # 이름 충돌 회피
                name = f"{base} ({i})"
                i += 1
            ids = []
            for j, p in enumerate(pieces):
                self.chunks.append({
                    "text": p, "doc": name, "tf": Counter(_tokenize(p)),
                    "vec": None, "emb": embs[j] if embs else None,
                })
                ids.append(len(self.chunks) - 1)
            self.docs[name] = {"chunk_ids": ids, "n": len(text)}
            self._reindex()
        return len(pieces)

    def add_url(self, url):
        url = (url or "").strip()
        if not re.match(r"^https?://", url):
            raise ValueError("http(s):// 로 시작하는 URL 이 필요합니다")
        r = httpx.get(url, timeout=30, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 (RNGD-Chat RAG)"})
        r.raise_for_status()
        text = _strip_html(r.text) if "html" in r.headers.get("content-type", "") else r.text
        name = re.sub(r"^https?://", "", url)[:60]
        return self.add(name, text)

    def add_file(self, path):
        """텍스트 계열 파일을 읽어 추가. PDF 는 pypdf 가 있으면 처리, 없으면 안내 예외."""
        from pathlib import Path
        p = Path(path)
        ext = p.suffix.lower()
        if ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(p))
                text = "\n\n".join(pg.extract_text() or "" for pg in reader.pages)
            except ImportError:
                raise ValueError("PDF 는 pypdf 가 필요합니다 — .txt/.md/코드 파일을 쓰거나 내용을 붙여넣으세요")
        else:
            text = p.read_text(errors="replace")
        return self.add(p.name, text)

    def remove(self, name):
        with self._lock:
            if name not in self.docs:
                return
            del self.docs[name]
            self.chunks = [c for c in self.chunks if c["doc"] != name]
            # chunk_ids 는 인덱스 기반이라 재구성
            for d in self.docs.values():
                d["chunk_ids"] = []
            for i, c in enumerate(self.chunks):
                self.docs[c["doc"]]["chunk_ids"].append(i)
            self._reindex()

    def clear(self):
        with self._lock:
            self.docs.clear()
            self.chunks.clear()
            self.idf.clear()

    # ── 검색 ────────────────────────────────────────────────────────────
    def _rerank(self, query, cands):
        """furiosa 리랭커(/v1/rerank)로 후보 재정렬. 실패하면 입력 순서 유지."""
        if not RERANK_URL or not cands:
            return cands
        try:
            r = httpx.post(RERANK_URL,
                           json={"query": query, "documents": [c["text"] for c in cands]},
                           timeout=60)
            r.raise_for_status()
            data = r.json()
            results = data.get("results") or data.get("data") or []
            order = [(item.get("index", i), item.get("relevance_score", item.get("score", 0.0)))
                     for i, item in enumerate(results)]
            order.sort(key=lambda x: x[1], reverse=True)
            return [cands[i] for i, _ in order if i < len(cands)]
        except Exception:
            return cands

    def retrieve(self, query, k=4):
        """질문에 가장 관련 있는 청크 top-k 를 (text, doc, score) 로 반환."""
        query = (query or "").strip()
        with self._lock:
            chunks = list(self.chunks)
            idf = dict(self.idf)
        if not query or not chunks:
            return []
        scored = []
        use_emb = bool(EMBED_URL) and self._embed_ok is not False and any(c["emb"] for c in chunks)
        if use_emb:
            # 혼합 인덱스 방지: emb 없는 청크(서버가 잠시 죽었을 때 추가된 것)를 백필해
            # 전부 임베딩으로 비교한다. 다 못 채우면 아래 TF-IDF 로 전체 일관 재점수.
            missing = [c for c in chunks if c["emb"] is None]
            if missing:
                vecs = self._embed([c["text"] for c in missing])
                if vecs and len(vecs) == len(missing):
                    for c, v in zip(missing, vecs):
                        c["emb"] = v
            qv = self._embed([query])
            if qv and all(c["emb"] is not None for c in chunks):
                import numpy as np
                q = np.array(qv[0], dtype="float32")
                qn = float(np.linalg.norm(q)) or 1.0
                for c in chunks:
                    v = np.array(c["emb"], dtype="float32")
                    s = float(q @ v) / (qn * (float(np.linalg.norm(v)) or 1.0))
                    scored.append((s, c))
        if not scored:  # TF-IDF (임베딩 불가/혼합 → 전체를 같은 척도로 일관 점수)
            qtf = Counter(_tokenize(query))
            qv = {t: f * idf.get(t, 0.0) for t, f in qtf.items()}
            for c in chunks:
                scored.append((_cos_sparse(qv, c["vec"] or {}), c))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [c for s, c in scored if s > 0][: max(k * 3, k)] or [c for _, c in scored[:k]]
        top = self._rerank(query, top)[:k]
        out = []
        for c in top:
            # 점수 재계산은 생략(리랭킹 후 순서가 의미) — 표시는 순위로
            out.append((c["text"], c["doc"]))
        return out

    def context(self, query, k=4):
        """retrieve 결과를 LLM 에 주입할 컨텍스트 문자열 + 출처 목록으로 포맷."""
        hits = self.retrieve(query, k)
        if not hits:
            return "", []
        blocks, sources = [], []
        for i, (text, doc) in enumerate(hits, 1):
            blocks.append(f"[{i}] (출처: {doc})\n{text}")
            if doc not in sources:
                sources.append(doc)
        ctx = "\n\n".join(blocks)
        return ctx, sources

    def summary(self):
        with self._lock:
            nd, nc = len(self.docs), len(self.chunks)
            names = list(self.docs.keys())
        return nd, nc, names
