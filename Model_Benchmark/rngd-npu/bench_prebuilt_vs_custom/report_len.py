#!/usr/bin/env python3
"""summary_len.json(analyze_len.py) 을 README 에 붙일 마크다운 표로 찍는다. 수치는 손으로 적지 않는다.

  python3 report_len.py [--cap 16384]
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "_ppt_src"))
from analyze_len import cap_rows, CANDIDATES  # noqa: E402

MODELS = [  # _ppt_src/gen_all.py 의 순서와 표시 이름
    ("gpt-oss-120b", "gpt-oss 120B", 4), ("Solar-Open-100B-NVFP4A16", "Solar-Open 100B", 4),
    ("K-EXAONE-236B-A23B-NVFP4A16", "K-EXAONE 236B", 4), ("Llama-3.3-70B-Instruct", "Llama 3.3 70B", 4),
    ("Qwen3-32B-FP8", "Qwen3 32B", 4), ("EXAONE-4.0-32B-FP8", "EXAONE 4.0 32B", 4),
    ("Qwen3-VL-32B-Instruct", "Qwen3-VL 32B", 4), ("Qwen3-Coder-30B-A3B-Instruct-FP8", "Qwen3-Coder 30B", 4),
    ("Qwen3-30B-A3B-Instruct-2507-FP8", "A3B Instruct 2507", 4),
    ("Qwen3-30B-A3B-Thinking-2507-FP8", "A3B Thinking 2507", 4), ("Qwen3-30B-A3B-FP8", "A3B 30B", 4),
    ("Llama-3.1-8B-Instruct", "Llama 3.1 8B", 1), ("Qwen3-8B-FP8", "Qwen3 8B", 1),
    ("Qwen3-4B-FP8", "Qwen3 4B", 1), ("Qwen2.5-0.5B-Instruct", "Qwen2.5 0.5B", 1),
]


def f(v, d=0, u=""):
    return "-" if v is None else f"{v:,.{d}f}{u}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int)
    a = ap.parse_args()
    rows = json.load(open(os.path.join(HERE, "summary_len.json"), encoding="utf-8"))
    old = {d["model"]: d for d in json.load(open(os.path.join(HERE, "summary.json"), encoding="utf-8"))}
    by = {d["model"]: d for d in rows}

    print("| 모델 | 카드 | 적재 | TTFT | tok/s | 멈춤 뺀 tok/s | 08-29 tok/s | 동시4 | 08-29 동시4 | 최장 출력 | 멈춤 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for mid, lab, cards in MODELS:
        d = by.get(mid)
        o = old.get(mid) or {}
        if not d or "runs" not in d:
            print(f"| {lab} | {cards} | 못 뜸 | - | - | - | - | - | - | - | - |")
            continue
        runs = [r for r in d["runs"] if not r.get("error")]
        top = max(runs, key=lambda r: r["out_tokens"])
        mark = " (반복)" if top.get("degenerate") else (" (잘림)" if top.get("finish_reason") != "stop" else "")
        print(f"| {lab} | {cards} | {f(d.get('serve_load_s'), 0, 's')} | {f(d.get('ttft_med'), 2)} | "
              f"{f(d.get('decode_tps_med'))} | {f(d.get('clean_tps_med'))} | {f(o.get('decode_tps_med'))} | "
              f"{f((d.get('concurrent') or {}).get('agg_tps'))} | {f((o.get('concurrent') or {}).get('agg_tps'))} | "
              f"{top['out_tokens']:,}{mark} | {d.get('stall_n', 0)}회 {f(d.get('stall_lost_s'), 0, 's')} |")

    print("\n| max_tokens | 잘리는 단일 요청 | 잘리는 동시 요청 | 단일 4종 시간 합 | 잘리는 모델 |")
    print("|---|---|---|---|---|")
    lab = {m: l for m, l, _ in MODELS}
    cands = sorted(set(CANDIDATES) | ({a.cap} if a.cap else set()))
    for c in cap_rows(rows, cands):
        print(f"| {c['cap']:,}{' ★' if c['cap'] == a.cap else ''} | {c['cut_single']} / {c['n_single']} | "
              f"{c['cut_conc']} / {c['n_conc']} | {c['single_min']:.1f}분 | "
              f"{', '.join(lab.get(m, m) for m in c['cut_models']) or '없음'} |")

    print("\n| 모델 | fact | code | reason | long | 동시4 (요청별) |")
    print("|---|---|---|---|---|---|")
    for mid, l, _ in MODELS:
        d = by.get(mid)
        if not d or "runs" not in d:
            continue
        cell = {}
        for r in d["runs"]:
            if r.get("error"):
                continue
            th = r.get("think_tok") or 0
            tag = " 반복" if r.get("degenerate") else ("" if r.get("finish_reason") == "stop" else f" {r.get('finish_reason')}")
            cell[r["prompt"]] = f"{r['out_tokens']:,}" + (f" (사고 {th:,})" if th else "") + tag
        conc = (d.get("concurrent") or {}).get("per_req") or []
        print(f"| {l} | " + " | ".join(cell.get(k, "-") for k in ("fact", "code", "reason", "long")) +
              f" | {', '.join(format(q['out_tokens'], ',') for q in conc) or '-'} |")


if __name__ == "__main__":
    main()
