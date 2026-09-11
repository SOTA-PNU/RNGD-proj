#!/usr/bin/env python3
"""멈춤 시각과 커널 카운터(diag_sampler.py 기록)를 맞춰 본다.

  diag_corr.py [sampler.jsonl]

results_len/*.json 의 단일 요청마다 t0_wall + 조각 시각으로 멈춤(간격 0.1초 초과)의 절대 시각을 구하고,
0.5초 표본 구간마다 멈춤 시간과 카운터 증가량을 짝짓는다. 멈춤 구간과 조용한 구간의 평균 증가량,
그리고 멈춤 시간과 각 카운터 증가량의 상관계수를 찍는다.
"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results_len_diag", "sampler.jsonl")


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy), 3) if sx and sy else None


def main():
    import gzip
    # 커밋본은 sampler.jsonl.gz 다(원본은 9.6 MB 라 .gitignore)
    path = SAMPLER if os.path.exists(SAMPLER) else SAMPLER + ".gz"
    fh = gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")
    samp = [json.loads(l) for l in fh]
    samp.sort(key=lambda s: s["t"])
    keys = [k for k in samp[0] if k not in ("t",)]
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "results_len", "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        runs = (d.get("runs") or []) + ([d["control_long_1024"]] if d.get("control_long_1024") else [])
        for r in runs:
            t, t0 = r.get("times") or [], r.get("t0_wall")
            if not t or not t0:
                continue
            lo, hi = t0 + t[0], t0 + t[-1]
            # 멈춤 기준: 0.1초와 간격 중앙값 × 5 중 큰 값(gpt-oss 는 스텝 자체가 약 1초)
            dd = sorted(b - a for a, b in zip(t, t[1:]))
            thr = max(0.1, 5 * dd[len(dd) // 2]) if dd else 0.1
            gaps = [(t0 + a, t0 + b) for a, b in zip(t, t[1:]) if b - a > thr]
            for s0, s1 in zip(samp, samp[1:]):
                if s1["t"] <= lo or s0["t"] >= hi:
                    continue
                stall = sum(min(b, s1["t"]) - max(a, s0["t"]) for a, b in gaps if b > s0["t"] and a < s1["t"])
                row = {"model": d["model"], "stall_s": stall}
                for k in keys:
                    # 누적 카운터는 증가량, 노드 여유 메모리는 그 시점 값
                    row[k] = s1[k] if k.endswith("_free_kb") else s1[k] - s0[k]
                rows.append(row)
    if not rows:
        print("짝지을 요청이 없다 (t0_wall 이 있는 요청이 기록기와 겹치지 않음)")
        return
    hot = [r for r in rows if r["stall_s"] > 0.05]
    cold = [r for r in rows if r["stall_s"] == 0]
    print(f"표본 {len(rows)}개: 멈춤 구간 {len(hot)}, 조용한 구간 {len(cold)}, 모델 {sorted(set(r['model'] for r in hot))}")
    print(f"{'카운터':26s} {'멈춤 구간 평균':>14s} {'조용한 구간 평균':>16s} {'상관계수':>8s}")
    st = [r["stall_s"] for r in rows]
    for k in keys:
        h = sum(r[k] for r in hot) / len(hot) if hot else float("nan")
        c = sum(r[k] for r in cold) / len(cold) if cold else float("nan")
        print(f"{k:26s} {h:14.1f} {c:16.1f} {pearson(st, [r[k] for r in rows])!s:>8s}")


if __name__ == "__main__":
    main()
