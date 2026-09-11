#!/usr/bin/env python3
"""results_len/*.json(length_probe.py 결과)을 summary_len.json 으로 모으고, 한도 후보별 표를 찍는다.

summary.json(analyze.py)과 같은 필드를 채워 덱 생성기(_ppt_src)가 그대로 읽게 하고,
한도를 정하는 데 필요한 필드를 더한다:
  finish       실행별 stop / length / wall
  think_tok    usage.completion_tokens_details.reasoning_tokens (없으면 사고 조각 수)
  answer_tok   출력 - 사고
  tps_by_pos   디코드 속도를 출력 위치 구간별로 — 길어질수록 느려지는지(attention 버킷이 커지는지)
  loop         length/wall 로 끝난 실행의 꼬리가 같은 조각의 반복인지
"""
import json, glob, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import serve_load_seconds, looks_broken  # noqa: E402

RES = os.path.join(HERE, "results_len")
OUT = os.path.join(HERE, "summary_len.json")
POS = [(0, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 16384)]
CANDIDATES = [1024, 2048, 4096, 8192, 16384]


def think_tokens(r):
    det = (r.get("usage") or {}).get("completion_tokens_details") or {}
    v = det.get("reasoning_tokens")
    return v if v is not None else r.get("think_chunks")


def tps_by_pos(r):
    """조각 도착 시각으로 구간별 디코드 속도를 낸다. 조각 하나가 토큰 하나보다 조금 적게 오므로
    (한글은 바이트가 모자라면 서버가 조각을 합친다) 조각 수 대 토큰 수 비율로 보정한다."""
    t = r.get("times") or []
    n = len(t)
    if n < 64 or not r.get("out_tokens"):
        return {}
    scale = r["out_tokens"] / n
    out = {}
    for a, b in POS:
        lo, hi = max(a, 1), min(b, n)
        if hi - lo < 64:
            continue
        dt = t[hi - 1] - t[lo - 1]
        if dt > 0:
            out[f"{a}-{b}"] = round((hi - lo) * scale / dt, 1)
    return out


def loop_period(s, min_span=300, max_p=1500):
    """꼬리가 같은 조각의 반복이면 그 주기(글자)를, 아니면 None."""
    s = (s or "")[-6000:]
    for p in range(1, min(max_p, len(s) // 3) + 1):
        unit = s[-p:]
        reps = 1
        while (reps + 1) * p <= len(s) and s[-(reps + 1) * p:-reps * p] == unit:
            reps += 1
        if reps >= 3 and reps * p >= min_span:
            return p
    return None


def stalls(r, thr=0.1):
    """조각 사이 thr 초 넘는 멈춤. 2026-09-11 tp32 모델에서 스텝 사이 0.1~1초 멈춤이 잦았다
    (정상 스텝은 08-29 와 같은 14 ms). 벽시계 속도와 멈춤을 뺀 속도(정상 스텝 중앙값)를 따로 낸다."""
    t = r.get("times") or []
    gaps = [b - a for a, b in zip(t, t[1:])]
    big = [g for g in gaps if g > thr]
    base = sorted(g for g in gaps if g <= thr)
    step = base[len(base) // 2] if base else None
    scale = (r["out_tokens"] / len(t)) if t and r.get("out_tokens") else 1.0
    return {"n": len(big), "lost_s": round(sum(big), 2),
            "step_ms": round(step * 1000, 1) if step else None,
            "clean_tps": round(scale / step, 1) if step else None}


def time_at(r, cap):
    """이 실행을 max_tokens=cap 으로 돌렸다면 걸렸을 시간. greedy 라 앞부분은 똑같이 나온다."""
    t = r.get("times") or []
    if not t or not r.get("out_tokens"):
        return r.get("total")
    if r["out_tokens"] <= cap:
        return r["total"]
    k = int(len(t) * cap / r["out_tokens"])
    return t[max(k, 1) - 1]


def median(v):
    v = sorted(x for x in v if x is not None)
    return v[len(v) // 2] if v else None


def load_all():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        mid = d["model"]
        d["base"] = mid.split("@")[0]
        d["serve_load_s"], d["build_kind"] = serve_load_seconds(mid)
        if "runs" in d:
            for r in d["runs"] + ([d["control_long_1024"]] if d.get("control_long_1024") else []):
                if r.get("error"):
                    r["broken"] = "오류: " + r["error"][:80]
                    continue
                r["broken"] = looks_broken(r.get("text"), r.get("thinking"))
                r["think_tok"] = think_tokens(r)
                r["answer_tok"] = (r["out_tokens"] or 0) - (r["think_tok"] or 0)
                r["tps_by_pos"] = tps_by_pos(r)
                r["stalls"] = stalls(r)
                r["loop"] = (loop_period((r.get("thinking") or "") + (r.get("text") or ""))
                             if r.get("finish_reason") in ("length", "wall") else None)
            runs = [r for r in d["runs"] if not r.get("error")]
            ct = median([r["stalls"]["clean_tps"] for r in runs if (r.get("out_tokens") or 0) >= 64])
            d["clean_tps_med"] = round(ct, 1) if ct else None
            d["stall_n"] = sum(r["stalls"]["n"] for r in runs)
            d["stall_lost_s"] = round(sum(r["stalls"]["lost_s"] for r in runs), 1)
            ok = [r for r in runs if not r["broken"] or r["broken"].startswith("사고만")]
            d["thinking_only"] = sum(1 for r in runs if (r["broken"] or "").startswith("사고만"))
            d["garbage"] = sum(1 for r in runs if r["broken"] and not r["broken"].startswith("사고만"))
            d["ok_runs"], d["n_runs"] = len(ok), len(d["runs"])
            d["decode_tps_med"] = round(median([r.get("decode_tps") for r in runs]), 1) if runs else None
            tt = median([r.get("ttft") for r in runs])
            d["ttft_med"] = round(tt, 3) if tt else None
            d["natural_max"] = max([r["out_tokens"] for r in runs if r.get("finish_reason") == "stop"] or [0])
            d["hit_cap"] = [r["prompt"] for r in runs if r.get("finish_reason") in ("length", "wall")]
        rows.append(d)
    return rows


def cap_rows(rows, candidates=CANDIDATES):
    """한도 후보마다: 잘리는 실행 수(단일, 동시)와 전 모델 단일 4프롬프트에 드는 시간 합.
    rows 는 load_all() 결과나 summary_len.json 을 그대로 받는다."""
    out = []
    for cap in candidates:
        cut1 = n1 = cutc = nc = 0
        secs = 0.0
        cut_models = set()
        for d in rows:
            for r in d.get("runs", []):
                if r.get("error"):
                    continue
                n1 += 1
                # 한도를 넘겨 나왔거나, 측정 때부터 스스로 못 멈춘 실행(한도, 시간 제한에 걸림)은 잘린다
                c = (r["out_tokens"] or 0) > cap or r.get("finish_reason") in ("length", "wall")
                cut1 += c
                if c:
                    cut_models.add(d["model"])
                secs += time_at(r, cap) or 0
            for q in (d.get("concurrent") or {}).get("per_req", []):
                nc += 1
                cutc += (q["out_tokens"] or 0) > cap or q.get("finish_reason") in ("length", "wall")
        out.append({"cap": cap, "cut_single": cut1, "n_single": n1, "cut_conc": cutc, "n_conc": nc,
                    "single_min": round(secs / 60, 1), "cut_models": sorted(cut_models)})
    return out


def cap_table(rows):
    print(f"\n{'한도':>6s} {'잘림(단일)':>10s} {'잘림(동시)':>10s} {'단일 4개 시간 합(전 모델)':>26s}  잘리는 모델")
    for c in cap_rows(rows):
        print(f"{c['cap']:6d} {c['cut_single']:6d}/{c['n_single']:<3d} {c['cut_conc']:6d}/{c['n_conc']:<3d} "
              f"{c['single_min']:20.1f}분  {', '.join(c['cut_models'])}")


def main():
    rows = load_all()
    print(f"{'모델':34s} {'프롬프트':7s} {'끝':6s} {'출력':>6s} {'사고':>6s} {'답':>6s} {'tok/s':>6s} {'총s':>7s}  위치별 tok/s / 반복")
    for d in rows:
        if "runs" not in d:
            print(f"{d['model']:34s} 실패 {d.get('error', '')[:70]}")
            continue
        for r in d["runs"] + ([d["control_long_1024"]] if d.get("control_long_1024") else []):
            if r.get("error"):
                print(f"{d['model'][:34]:34s} {r.get('prompt', ''):7s} 오류 {r['error'][:60]}")
                continue
            s = r.get("stalls") or {}
            extra = f"정상 {s.get('clean_tps')}/s 멈춤 {s.get('n')}회 {s.get('lost_s')}s  "
            extra += " ".join(f"{k}:{v}" for k, v in (r.get("tps_by_pos") or {}).items())
            if r.get("loop"):
                extra += f"  반복 주기 {r['loop']}자"
            if r.get("broken"):
                extra += f"  [{r['broken']}]"
            print(f"{d['model'][:34]:34s} {r['prompt'][:9]:7s} {str(r.get('finish_reason')):6s} {r['out_tokens']:6d} "
                  f"{(r.get('think_tok') or 0):6d} {r['answer_tok']:6d} {(r.get('decode_tps') or 0):6.1f} "
                  f"{r['total']:7.1f}  {extra}")
        print(f"{'':34s} 요약: 벽시계 {d['decode_tps_med']} tok/s, 멈춤 뺀 {d['clean_tps_med']} tok/s, "
              f"멈춤 {d['stall_n']}회 {d['stall_lost_s']}s, 스스로 멈춘 최대 {d['natural_max']}, 한도 도달 {d['hit_cap']}")
        for k in ("conc_1024", "concurrent"):
            c = d.get(k)
            if c:
                print(f"{'':34s} {k:12s} {c['wall_s']:7.1f}s {c['out_tokens']:6d}tok {c['agg_tps']:6.1f} tok/s "
                      f"finish={[q['finish_reason'] for q in c['per_req']]} out={[q['out_tokens'] for q in c['per_req']]}")
    cap_table(rows)
    json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n→ {OUT} ({len(rows)}개)")


if __name__ == "__main__":
    main()
