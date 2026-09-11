#!/usr/bin/env python3
"""max_tokens 를 넉넉히 두면 모델이 몇 토큰에서 스스로 멈추는지 잰다.

bench.py 는 max_tokens=1024 로 잰다. 사고(thinking) 모델은 그 안에서 답을 못 끝내는 일이 잦아서
(2026-08-29 결과: 사고 모델 7종 중 6종이 한 번 이상 잘림) 적정 한도를 정하려면 잘리지 않은 길이가 필요하다.

bench.py 와 같은 네 프롬프트, 같은 temperature 0 으로 한도만 키워 다시 잰다. 기록하는 것:
  finish_reason   stop 이면 스스로 멈춤, length 면 한도에 걸림, wall 이면 시간 제한으로 끊음
  출력 토큰       usage.completion_tokens, 사고/본문 조각 수(kinds 문자열: t=사고, c=본문)
  조각별 도착 시각 길이에 따라 디코드 속도가 떨어지는지(attention 버킷이 커지는지) 보려고
  원문            사고와 본문 둘 다

동시 4요청(long 프롬프트)도 같은 한도로 잰다 — bench.py 와 같은 항목을 모두 재므로 이 결과가 곧 새 벤치다.
greedy 라 한도 안에서 스스로 멈춘 실행은 한도를 얼마로 두든 같은 결과가 나온다.

--control 모델은 long 프롬프트를 1024 로도 한 번 더 재고, 동시 4요청도 1024 로 한 번 더 잰다.
max_tokens 가 속도나 동시 처리량을 바꾸는지 가르는 대조군이다.

결과: results_len/<id>.json. 모델 하나가 끝나야 complete=true 가 되고, 그 전에 끊기면 다음 실행이 다시 잰다.
"""
import argparse, json, os, sys, time, threading
import urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402  PROMPTS, ROUTER, get, wait_ready 를 그대로 쓴다

OUT = os.path.join(HERE, "results_len")
CAP = 16384
# 요청 하나가 이보다 오래 걸리면 끊고 finish_reason=wall 로 적는다.
# gpt-oss-120b 는 단일 스트림이 1~3 tok/s 라(2026-08-29 실측) 한도를 다 쓰면 이 제한에 먼저 걸린다.
WALL = 1800
# serve 는 prompt + max_tokens > max_model_len 이면 잘라 주지 않고 거절한다(furiosa_llm/errors.py).
# 프롬프트는 템플릿 포함 120토큰 안쪽이라 256 을 비워 둔다.
CTX_MARGIN = 256
CONC = 4


def router_ctx():
    d = bench.get("/router/models")
    return {m["id"]: m["context"] for m in d.get("data", [])}


def stream_probe(model, prompt, max_tokens, wall=WALL):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(bench.ROUTER + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t0_wall = time.time()   # 커널 카운터 기록(diag_sampler.py)과 시각을 맞추려고
    ttft = None; times = []; kinds = []; text = []; think = []; usage = None; finish = None; by_wall = False
    try:
        r = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as e:
        return dict(max_tokens=max_tokens, error=f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}")
    with r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except Exception:
                    ev = {}
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices", []):
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    d = ch.get("delta") or {}
                    c = d.get("content") or ""
                    rz = d.get("reasoning_content") or d.get("reasoning") or ""
                    if c or rz:
                        now = time.perf_counter() - t0
                        if ttft is None:
                            ttft = now
                        times.append(round(now, 3))
                        kinds.append("t" if rz else "c")
                    if c:
                        text.append(c)
                    if rz:
                        think.append(rz)
            if time.perf_counter() - t0 > wall:
                by_wall = True
                break
    total = time.perf_counter() - t0
    kinds = "".join(kinds)
    out_tok = (usage or {}).get("completion_tokens") or len(times)
    return dict(max_tokens=max_tokens, finish_reason="wall" if by_wall else finish, t0_wall=round(t0_wall, 3),
                ttft=ttft, total=total, out_tokens=out_tok, in_tokens=(usage or {}).get("prompt_tokens"),
                usage=usage, think_chunks=kinds.count("t"), text_chunks=kinds.count("c"),
                t_answer=(times[kinds.index("c")] if "c" in kinds else None),
                decode_tps=(out_tok - 1) / (total - ttft) if (ttft and total > ttft and out_tok > 1) else None,
                kinds=kinds, times=times, text="".join(text), thinking="".join(think))


def concurrent(model, prompt, max_tokens):
    res = [None] * CONC

    def work(i):
        try:
            res[i] = stream_probe(model, prompt, max_tokens)
        except Exception as e:
            res[i] = {"error": f"{type(e).__name__}: {e}"}
    ts = [threading.Thread(target=work, args=(i,)) for i in range(CONC)]
    t0 = time.perf_counter()
    [t.start() for t in ts]; [t.join() for t in ts]
    wall = time.perf_counter() - t0
    ok = [r for r in res if r and not r.get("error")]
    tot = sum(r["out_tokens"] or 0 for r in ok)
    return {"max_tokens": max_tokens, "n": CONC, "wall_s": round(wall, 2), "out_tokens": tot,
            "agg_tps": round(tot / wall, 1) if wall else None,
            "per_req": [{"out_tokens": r["out_tokens"], "finish_reason": r["finish_reason"],
                         "total": round(r["total"], 2)} for r in ok],
            "errors": [r.get("error") for r in res if r and r.get("error")]}


def save(path, rec):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def line(name, r):
    if r.get("error"):
        return f"    {name:12s} 오류 {r['error'][:160]}"
    return (f"    {name:12s} {str(r['finish_reason']):6s} {r['out_tokens']:6d}tok "
            f"(사고 {r['think_chunks']} / 본문 {r['text_chunks']})  "
            f"{(r['decode_tps'] or 0):5.1f} tok/s  총 {r['total']:7.1f}s")


def probe_model(model, cap, ctx, control, path):
    rec = {"model": model, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cap": cap, "ctx": ctx,
           "complete": False}
    print(f"\n=== {model}  (max_tokens {cap}, ctx {ctx})", flush=True)
    rec["load_s"] = round(bench.wait_ready(model), 1)
    print(f"    적재 {rec['load_s']}s", flush=True)
    rec["runs"] = []
    for name, p in bench.PROMPTS:
        r = stream_probe(model, p, cap)
        r["prompt"] = name
        rec["runs"].append(r)
        print(line(name, r), flush=True)
        save(path, rec)
    if control:
        r = stream_probe(model, bench.PROMPTS[3][1], 1024)
        r["prompt"] = "long@1024"
        rec["control_long_1024"] = r
        print(line("long@1024", r), flush=True)
    for mt in ((1024, cap) if control else (cap,)):
        c = concurrent(model, bench.PROMPTS[3][1], mt)
        rec["concurrent" if mt == cap else f"conc_{mt}"] = c
        print(f"    동시{CONC} @{mt}: {c['wall_s']}s 합계 {c['out_tokens']}tok {c['agg_tps']} tok/s "
              f"finish={[x['finish_reason'] for x in c['per_req']]}", flush=True)
        save(path, rec)
    rec["complete"] = True
    rec["elapsed_s"] = round(time.time() - time.mktime(time.strptime(rec["started"], "%Y-%m-%d %H:%M:%S")), 1)
    save(path, rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--control", nargs="*", default=[], help="1024 대조군까지 잴 모델")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ctxs = router_ctx()
    for m in a.models:
        path = os.path.join(OUT, m.replace("/", "_") + ".json")
        if os.path.exists(path) and not a.force:
            try:
                if json.load(open(path, encoding="utf-8")).get("complete"):
                    print(f"건너뜀(완료): {m}", flush=True)
                    continue
            except Exception:
                pass
        ctx = ctxs.get(m)
        cap = a.cap
        if ctx:
            cap = min(cap, ctx - CTX_MARGIN)
        try:
            probe_model(m, cap, ctx, m in a.control, path)
        except Exception as e:
            rec = {"model": m, "error": f"{type(e).__name__}: {e}", "complete": False,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}
            print(f"    실패: {rec['error']}", flush=True)
            if not os.path.exists(path):
                save(path, rec)
        print(f"    → {path}", flush=True)


if __name__ == "__main__":
    main()
