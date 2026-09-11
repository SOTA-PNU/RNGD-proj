#!/usr/bin/env python3
"""length_probe.py 결과를 해석하기 전에 가려야 할 것을 잰다. 결과는 results_len_diag/ 에.

  stall <모델> [--runs N] [--prompt reason] [--caps 1024 16384] [--direct URL] [--smi 초] [--tag 이름]
      같은 프롬프트를 한도만 바꿔 번갈아 잰다. 출력이 한도 안에서 끝나면 greedy 라 두 한도의 출력이
      똑같으므로, 조각 사이 0.1초 넘는 멈춤 수가 다르면 한도 탓이다.
      (2026-09-11: 넉 장짜리 모델 일부에서 스텝 사이 0.1~1초 멈춤이 잦았다. 정상 스텝은 08-29 와 같았다)
      --direct  라우터를 거치지 않고 serve 에 바로 요청한다(라우터는 5초마다 furiosa-smi 를 부른다)
      --smi     측정하는 동안 furiosa-smi status 를 이 간격(초)으로 부른다. 0 이면 쉬지 않고.
                멈춤이 smi 실행 구간과 겹치는 비율을, smi 구간이 차지하는 시간 비율(우연 수준)과 비교한다.

  conc <모델> [--caps 1024 16384] [--prompt long] [--direct URL]
      동시 4요청을 한도만 바꿔 잰다. 동시 요청은 같은 프롬프트라도 요청마다 출력이 갈라지므로
      (배치가 수치를 바꿔 greedy 경로가 달라진다) 출력 길이 대신 "1024번째 조각 도달 시각"을 비교한다.
      (Qwen3-4B: 동시4 @1024 274 tok/s, @16384 225 tok/s — 출력 길이도 달랐다)
"""
import argparse, json, os, subprocess, sys, threading, time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
from length_probe import stream_probe  # noqa: E402

OUT = os.path.join(HERE, "results_len_diag")


def stalls(r, thr=0.1):
    t = r.get("times") or []
    gaps = [b - a for a, b in zip(t, t[1:])]
    big = [g for g in gaps if g > thr]
    base = sorted(g for g in gaps if g <= thr)
    return {"n": len(big), "lost_s": round(sum(big), 2),
            "step_ms": round(base[len(base) // 2] * 1000, 1) if base else None}


class SmiLoop:
    """furiosa-smi status 를 주기적으로 부르고 (시작, 끝) 구간을 perf_counter 로 남긴다."""

    def __init__(self, every):
        self.every, self.win, self.stop = every, [], threading.Event()
        self.th = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            a = time.perf_counter()
            subprocess.run(["furiosa-smi", "status"], capture_output=True, timeout=60)
            self.win.append((a, time.perf_counter()))
            if self.every > 0:
                self.stop.wait(self.every)

    def __enter__(self):
        self.th.start()
        return self

    def __exit__(self, *e):
        self.stop.set()
        self.th.join(timeout=90)


def overlap(r, t_start, wins, thr=0.1):
    """멈춤 중 smi 구간과 겹친 비율, 그리고 요청 시간 중 smi 구간이 차지한 비율."""
    t = r.get("times") or []
    if not t or not wins:
        return None
    lo, hi = t_start + t[0], t_start + t[-1]
    inside = [(max(a, lo), min(b, hi)) for a, b in wins if b > lo and a < hi]
    cover = sum(b - a for a, b in inside) / (hi - lo) if hi > lo else 0
    gaps = [(t_start + a, t_start + b) for a, b in zip(t, t[1:]) if b - a > thr]
    hit = sum(1 for a, b in gaps if any(x < b and y > a for x, y in inside))
    return {"stalls": len(gaps), "stalls_in_smi": hit, "smi_calls": len(inside),
            "smi_time_frac": round(cover, 3), "stall_in_smi_frac": round(hit / len(gaps), 3) if gaps else None}


def summary(r):
    s = stalls(r)
    o = r.get("smi_overlap")
    extra = (f"  smi {o['smi_calls']}회, 시간 {o['smi_time_frac']:.0%}, 멈춤 중 smi 겹침 "
             f"{o['stalls_in_smi']}/{o['stalls']}") if o else ""
    return (f"max_tokens {r['max_tokens']:5d}  {r.get('finish_reason')!s:6s} {r['out_tokens']:5d}tok  "
            f"{(r.get('decode_tps') or 0):5.1f} tok/s  총 {r['total']:6.1f}s  "
            f"멈춤 {s['n']:3d}회 {s['lost_s']:5.1f}s  정상 스텝 {s['step_ms']} ms{extra}")


def wait_direct(url, budget=3600):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < budget:
        try:
            with urllib.request.urlopen(url + "/v1/models", timeout=10) as r:
                if json.loads(r.read()).get("data"):
                    return time.perf_counter() - t0
        except Exception:
            pass
        time.sleep(10)
    raise TimeoutError(f"{url} 준비 실패")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stall", "conc"])
    ap.add_argument("model")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--prompt", default="reason")
    ap.add_argument("--caps", type=int, nargs="+", default=[1024, 16384])
    ap.add_argument("--direct")
    ap.add_argument("--smi", type=float, default=-1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    prompt = dict(bench.PROMPTS)[a.prompt]
    if a.direct:
        bench.ROUTER = a.direct.rstrip("/")
        load = wait_direct(bench.ROUTER)
    else:
        load = bench.wait_ready(a.model)
    os.makedirs(OUT, exist_ok=True)
    rec = {"mode": a.mode, "model": a.model, "prompt": a.prompt, "caps": a.caps, "direct": a.direct,
           "smi": a.smi, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "load_s": round(load, 1)}
    print(f"=== {a.mode} {a.model} ({a.prompt}) caps={a.caps} direct={a.direct} smi={a.smi}", flush=True)
    if a.mode == "stall":
        rec["runs"] = []
        for i in range(a.runs):
            for cap in a.caps:
                if a.smi >= 0:
                    with SmiLoop(a.smi) as sm:
                        t_start = time.perf_counter()
                        r = stream_probe(a.model, prompt, cap)
                    r["smi_overlap"] = overlap(r, t_start, sm.win)
                else:
                    r = stream_probe(a.model, prompt, cap)
                r["round"] = i
                r["stalls"] = stalls(r)
                rec["runs"].append(r)
                print("   ", summary(r), flush=True)
    else:
        rec["conc"] = {}
        # 같은 한도를 번갈아 여러 번 잴 수 있게(시간에 따라 밀려오는 멈춤과 한도의 효과를 가르려고) 순번을 키에 넣는다
        for i, cap in enumerate(a.caps):
            res = [None] * 4

            def work(i):
                res[i] = stream_probe(a.model, prompt, cap)
            ts = [threading.Thread(target=work, args=(i,)) for i in range(4)]
            t0 = time.perf_counter()
            [t.start() for t in ts]; [t.join() for t in ts]
            wall = time.perf_counter() - t0
            tot = sum(r["out_tokens"] for r in res)
            # 1024번째 조각이 올 때까지 걸린 시간 — 출력 길이와 무관하게 앞부분 속도를 비교한다
            t1k = [r["times"][min(1023, len(r["times"]) - 1)] for r in res]
            rec["conc"][f"{i}:{cap}"] = {"wall_s": round(wall, 2), "out_tokens": tot, "agg_tps": round(tot / wall, 1),
                                     "t_at_1024": [round(x, 2) for x in t1k], "runs": res}
            print(f"    동시4 @{cap}: {wall:6.1f}s 합계 {tot}tok {tot / wall:6.1f} tok/s  "
                  f"1024조각 도달 {[round(x, 1) for x in t1k]}  "
                  f"멈춤 {[stalls(r)['n'] for r in res]}", flush=True)
    path = os.path.join(OUT, f"{a.mode}_{a.model}{('_' + a.tag) if a.tag else ''}.json")
    json.dump(rec, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"    → {path}", flush=True)


if __name__ == "__main__":
    main()
