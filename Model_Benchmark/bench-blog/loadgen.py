"""블로그 재현용 부하시험 코어 — OpenAI 호환 엔드포인트 공용(furiosa-llm / vLLM).

Furiosa 블로그(RNGD vs RTX PRO 6000, Qwen3-32B)의 방법론을 1장 vs 1대로 재현한다:
  - 배치(=동시 사용자) b ∈ {1,8,16,32,64,256} 를 스윕.
  - 각 배치에서 b개의 요청을 항상 in-flight 로 유지(closed-loop)하며 정상상태 구간을 측정.
  - 고정 입력길이(ISL)·고정 출력길이(OSL: max_tokens + ignore_eos)로 공정 비교.
  - per-request: TTFT(스트리밍 첫 토큰), decode 시간, 출력/입력 토큰 수.
  - per-batch: 집계 출력 TPS(구간), per-user 출력 TPS(p50), TTFT(p50/p90), 평균 전력(동시 샘플링).

핵심 지표(블로그):
  - "SLO당 사용자 수": per-user 출력 TPS ≥ SLO(20/30/40) 를 지키는 최대 동시성.
  - 전력효율: 집계 TPS / 평균 전력 = tokens/sec/W.
  - TTFT, 집계 처리량 곡선.

이 파일은 서버에 의존하지 않는다(엔드포인트 URL만 받음). 서버 기동은 run_rngd.sh / run_pro6000.sh.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from power import PowerSampler


# ----------------------------- 프롬프트 만들기 -----------------------------

_BASE_SENT = (
    "The quick brown fox jumps over the lazy dog while the engineer carefully measures "
    "throughput and latency of a large language model running on dedicated accelerator hardware. "
)


def build_prompt(target_tokens: int, model: Optional[str] = None) -> str:
    """대략 target_tokens 토큰이 되도록 프롬프트 생성.
    같은 모델·같은 텍스트면 두 플랫폼의 prompt_tokens 가 동일하므로 ISL 비교가 공정하다.
    (정확한 토큰 수는 서버 usage.prompt_tokens 로 측정해 기록한다 — 여기선 근사만.)
    transformers 가 있으면 정확히 잘라 맞춘다.
    """
    # 이 영문 본문은 토큰/단어 ≈ 1.06 (실측). 목표 토큰에 맞춰 단어 수 추정.
    approx_words = max(4, int(target_tokens / 1.06))
    words = (_BASE_SENT * (approx_words // len(_BASE_SENT.split()) + 2)).split()
    text = " ".join(words[:approx_words])
    if model:
        try:  # 가능하면 정확히 ISL 토큰으로 트림
            from transformers import AutoTokenizer  # type: ignore

            tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) >= target_tokens:
                text = tok.decode(ids[:target_tokens])
        except Exception:
            pass
    return text


# ----------------------------- 측정 단위 -----------------------------

@dataclass
class ReqMetric:
    ok: bool
    ttft_s: float = float("nan")
    decode_s: float = float("nan")      # 첫 토큰 이후 생성 시간
    total_s: float = float("nan")
    out_tokens: int = 0
    in_tokens: int = 0
    t_start: float = 0.0
    t_end: float = 0.0
    error: Optional[str] = None

    @property
    def out_tps(self) -> float:
        """per-user 출력 토큰 속도(decode 기준) = SLO 와 비교하는 값."""
        return self.out_tokens / self.decode_s if (self.out_tokens > 0 and self.decode_s > 0) else 0.0


async def _one_request(
    client: httpx.AsyncClient, base_url: str, model: str, prompt: str,
    max_tokens: int, ignore_eos: bool, endpoint: str,
) -> ReqMetric:
    is_chat = endpoint == "chat"
    url = f"{base_url}/chat/completions" if is_chat else f"{base_url}/completions"
    if is_chat:
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    else:
        payload = {"model": model, "prompt": prompt}
    payload.update({
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},  # 명시해야 마지막 usage chunk 옴(furiosa/vLLM 공통)
    })
    if ignore_eos:
        # OSL 정확 고정: ignore_eos + min_tokens=max_tokens → 정확히 max_tokens 토큰 생성
        # (furiosa-llm protocol.py:687/689, vLLM 동일 지원)
        payload["ignore_eos"] = True
        payload["min_tokens"] = max_tokens

    t0 = time.perf_counter()
    t_first = None
    n_chunks = 0
    usage = None
    try:
        async with client.stream("POST", url, json=payload) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode("utf-8", "ignore")[:200]
                return ReqMetric(ok=False, error=f"HTTP {r.status_code}: {body}", t_start=t0)
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                ch = choices[0]
                piece = ch.get("text") if not is_chat else (ch.get("delta") or {}).get("content")
                if piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    n_chunks += 1
        t_end = time.perf_counter()
    except Exception as e:  # noqa: BLE001
        return ReqMetric(ok=False, error=f"{type(e).__name__}: {e}", t_start=t0)

    if t_first is None:
        return ReqMetric(ok=False, error="no tokens", t_start=t0, t_end=t_end)
    out_tokens = (usage or {}).get("completion_tokens") or n_chunks
    in_tokens = (usage or {}).get("prompt_tokens") or 0
    return ReqMetric(
        ok=True, ttft_s=t_first - t0, decode_s=max(t_end - t_first, 1e-9),
        total_s=t_end - t0, out_tokens=out_tokens, in_tokens=in_tokens,
        t_start=t0, t_end=t_end,
    )


# ----------------------------- 배치별 closed-loop 측정 -----------------------------

@dataclass
class BatchResult:
    batch: int
    window_s: float
    n_completed: int
    n_error: int
    agg_out_tps: float                 # 구간 집계 출력 TPS (= 총 출력토큰/구간)
    per_user_out_tps_p50: float        # per-user 출력 TPS 중앙값
    per_user_out_tps_p10: float
    ttft_p50_s: float
    ttft_p90_s: float
    in_tokens_mean: float
    out_tokens_mean: float
    power_avg_w: Optional[float]
    power_max_w: Optional[float]
    errors_sample: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


async def run_batch(
    base_url: str, model: str, prompt: str, *, batch: int, max_tokens: int,
    ignore_eos: bool, endpoint: str, warmup_s: float, window_s: float,
    power_backend: str, power_devices: Optional[list[int]], unique_prompts: bool = True,
    min_completions: Optional[int] = None, max_window_mult: float = 3.0,
) -> BatchResult:
    """closed-loop 로 batch 개 요청을 항상 in-flight 유지.
    warmup_s 동안 워밍업(버림) → 측정: 최소 window_s 경과 AND 최소 min_completions 완료까지(상한
    window_s*max_window_mult). 긴 출력(OSL)이라 한 요청이 window_s 보다 길어도 완료를 보장한다."""
    if min_completions is None:
        min_completions = max(batch, 12)
    limits = httpx.Limits(max_connections=batch + 8, max_keepalive_connections=batch + 8)
    timeout = httpx.Timeout(connect=30.0, read=600.0, write=60.0, pool=600.0)
    metrics: list[ReqMetric] = []
    errors: list[str] = []
    warmup_done = [False]
    stop = [False]

    def make_prompt() -> str:
        # 요청마다 고유 prefix → prefix-caching 캐시히트로 처리량이 부풀려지는 것 방지(양 플랫폼 공통).
        return f"[uid {uuid.uuid4().hex} req] {prompt}" if unique_prompts else prompt

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        async def worker():
            while not stop[0]:
                m = await _one_request(client, base_url, model, make_prompt(), max_tokens, ignore_eos, endpoint)
                if not warmup_done[0]:
                    continue          # 워밍업 구간 결과는 버림
                if stop[0] and not m.ok:
                    break
                if m.ok and m.out_tokens > 0:
                    metrics.append(m)
                elif m.error:
                    errors.append(m.error)

        workers = [asyncio.create_task(worker()) for _ in range(batch)]

        # 워밍업
        await asyncio.sleep(warmup_s)
        warmup_done[0] = True

        # 측정 구간 — 전력 동시 샘플링. 최소 시간 + 최소 완료수 충족까지, 상한 내에서.
        t_win_start = time.perf_counter()
        sampler = PowerSampler(power_backend, devices=power_devices, interval_s=1.0).start()
        hard_cap = window_s * max_window_mult
        while True:
            elapsed = time.perf_counter() - t_win_start
            if elapsed >= hard_cap:
                break
            if elapsed >= window_s and len(metrics) >= min_completions:
                break
            await asyncio.sleep(0.5)
        stop[0] = True
        # in-flight 정리(현재 요청 끝나면 워커 종료)
        await asyncio.gather(*workers, return_exceptions=True)
        pstats = sampler.stop()
        t_win_end = time.perf_counter()

    win = t_win_end - t_win_start
    ok = [m for m in metrics if m.ok and m.out_tokens > 0]
    if not ok:
        return BatchResult(
            batch=batch, window_s=round(win, 2), n_completed=0, n_error=len(errors),
            agg_out_tps=0.0, per_user_out_tps_p50=0.0, per_user_out_tps_p10=0.0,
            ttft_p50_s=float("nan"), ttft_p90_s=float("nan"),
            in_tokens_mean=0.0, out_tokens_mean=0.0,
            power_avg_w=pstats.avg_w if pstats.n else None,
            power_max_w=pstats.max_w if pstats.n else None,
            errors_sample=errors[:3],
        )
    total_out = sum(m.out_tokens for m in ok)
    user_tps = sorted(m.out_tps for m in ok)
    ttfts = sorted(m.ttft_s for m in ok)

    def pct(xs, p):
        if not xs:
            return float("nan")
        k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
        return xs[k]

    return BatchResult(
        batch=batch, window_s=round(win, 2), n_completed=len(ok), n_error=len(errors),
        agg_out_tps=round(total_out / win, 1),
        per_user_out_tps_p50=round(statistics.median(user_tps), 2),
        per_user_out_tps_p10=round(pct(user_tps, 10), 2),
        ttft_p50_s=round(statistics.median(ttfts), 3),
        ttft_p90_s=round(pct(ttfts, 90), 3),
        in_tokens_mean=round(statistics.mean(m.in_tokens for m in ok), 1),
        out_tokens_mean=round(statistics.mean(m.out_tokens for m in ok), 1),
        power_avg_w=round(pstats.avg_w, 1) if pstats.n else None,
        power_max_w=round(pstats.max_w, 1) if pstats.n else None,
        errors_sample=errors[:3],
    )


# ----------------------------- 엔트리포인트 -----------------------------

async def main_async(args) -> dict:
    # 요청별 고유 prefix(약 14토큰)를 감안해 본문은 ISL-14 로 만든다. tokenizer 있으면 정확 트림.
    prompt = build_prompt(max(8, args.isl - 14), model=args.tokenizer or None)
    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    print(f"[loadgen] platform={args.platform} model={args.model} ISL~{args.isl} OSL={args.osl} "
          f"batches={batches} window={args.window}s endpoint={args.endpoint}", flush=True)

    results = []
    for b in batches:
        print(f"[loadgen] batch={b} 측정 중...", flush=True)
        r = await run_batch(
            args.base_url, args.model, prompt, batch=b, max_tokens=args.osl,
            ignore_eos=not args.no_ignore_eos, endpoint=args.endpoint,
            warmup_s=args.warmup, window_s=args.window,
            power_backend=args.power, power_devices=(
                [int(x) for x in args.power_devices.split(",")] if args.power_devices else None),
            unique_prompts=not args.no_unique_prompts,
        )
        results.append(r.to_dict())
        print(f"  -> agg={r.agg_out_tps} tok/s | per-user p50={r.per_user_out_tps_p50} tok/s | "
              f"TTFT p50={r.ttft_p50_s}s | power={r.power_avg_w}W | ok={r.n_completed} err={r.n_error}",
              flush=True)
        await asyncio.sleep(2.0)  # 배치 간 정리

    return {
        "meta": {
            "platform": args.platform,
            "label": args.label,
            "model": args.model,
            "isl_target": args.isl,
            "osl": args.osl,
            "ignore_eos": not args.no_ignore_eos,
            "endpoint": args.endpoint,
            "warmup_s": args.warmup,
            "window_s": args.window,
            "power_backend": args.power,
            "power_devices": args.power_devices,
            "base_url": args.base_url,
        },
        "batches": results,
    }


def main():
    ap = argparse.ArgumentParser(description="블로그 재현 부하시험 (OpenAI 호환)")
    ap.add_argument("--base-url", required=True, help="예: http://127.0.0.1:8004/v1")
    ap.add_argument("--model", required=True, help="/v1/models 의 모델 id")
    ap.add_argument("--platform", default="unknown", help="rngd | pro6000 (리포트 라벨)")
    ap.add_argument("--label", default="", help="추가 라벨(예: fp8)")
    ap.add_argument("--isl", type=int, default=1024, help="입력 토큰 목표(근사, 실제는 usage로 기록)")
    ap.add_argument("--osl", type=int, default=256, help="출력 토큰(고정, ignore_eos)")
    ap.add_argument("--batches", default="1,8,16,32,64,256", help="동시성(=사용자) 목록")
    ap.add_argument("--window", type=float, default=30.0, help="배치별 측정 구간(초)")
    ap.add_argument("--warmup", type=float, default=8.0, help="배치별 워밍업(초)")
    ap.add_argument("--endpoint", choices=["completions", "chat"], default="completions")
    ap.add_argument("--no-ignore-eos", action="store_true", help="ignore_eos 미지원 서버용(출력길이 가변)")
    ap.add_argument("--no-unique-prompts", action="store_true", help="요청별 고유 prefix 끄기(prefix-cache 영향 측정용)")
    ap.add_argument("--power", choices=["rngd", "gpu", "none"], default="none")
    ap.add_argument("--power-devices", default=None, help="측정 디바이스 인덱스(예: '0'). 미지정=전체")
    ap.add_argument("--tokenizer", default=None, help="정확 ISL 트림용 HF tokenizer id(선택)")
    ap.add_argument("--out", required=True, help="결과 JSON 경로")
    args = ap.parse_args()

    data = asyncio.run(main_async(args))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"[loadgen] 저장: {args.out}")


if __name__ == "__main__":
    main()
