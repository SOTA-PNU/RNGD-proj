"""rngd.json + pro6000.json → 블로그 주장 대조 리포트(+차트).

블로그(RNGD vs RTX PRO 6000, Qwen3-32B)의 핵심 주장을 1장 vs 1대 실측으로 대조한다:
  (A) raw per-device — SLO(20/30/40 TPS/user)당 최대 사용자 수. 블로그 1:1 근사치는 ~1.1x.
  (B) 전력정규화 — users/kW, tokens/sec/W. 블로그의 1.8~2.5x 'normalized for rack power' 주장의 근거.
  (C) TTFT — RNGD ≈ 절반 주장.
  (D) 집계 처리량 vs 배치 곡선.

핵심: 블로그의 1.8/1.9/2.0x 는 '랙 전력 정규화 후' 값이다(per-device 아님). 그래서 (A)는 ~1.1x,
(B)에서 ~2x 가 나오는 게 정상이며, 둘을 분리해서 보여줘야 블로그가 정확히 재현·설명된다.

사용:
  python compare.py results/rngd.json results/pro6000.json --out results/report.md
  python compare.py results/rngd.json                       # GPU 결과 전이면 RNGD 단독 요약
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

SLOS = [20, 30, 40]


def load(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def by_batch(data: dict) -> dict[int, dict]:
    return {b["batch"]: b for b in data["batches"]}


def users_at_slo(batches: list[dict], slo: float) -> float:
    """per-user p50 출력 TPS가 SLO 이상을 유지하는 최대 동시성(=사용자).
    배치 점들 사이는 선형보간해 per-user TPS == SLO 가 되는 지점의 (분수) 사용자 수를 돌려준다.
    배치 1조차 SLO 미달이면 0, 모든 배치가 SLO 이상이면 최대 배치(천장 미발견, '>=')."""
    pts = sorted(((b["batch"], b["per_user_out_tps_p50"]) for b in batches
                  if b.get("per_user_out_tps_p50")), key=lambda x: x[0])
    if not pts:
        return float("nan")
    if pts[0][1] < slo:
        return 0.0
    best = pts[0][0]
    for (b_lo, t_lo), (b_hi, t_hi) in zip(pts, pts[1:]):
        if t_hi >= slo:
            best = b_hi
            continue
        # t_lo >= slo > t_hi 사이에서 보간
        if t_lo == t_hi:
            return float(b_lo)
        frac = (t_lo - slo) / (t_lo - t_hi)
        return round(b_lo + (b_hi - b_lo) * frac, 1)
    return float(best)  # 전 구간 SLO 이상 → 최대 배치(천장 미발견)


def ratio(a: float, b: float) -> str:
    if not b or math.isnan(a) or math.isnan(b) or b == 0:
        return "—"
    return f"{a / b:.2f}x"


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def peak_eff(batches: list[dict]) -> tuple[Optional[int], float]:
    """tokens/sec/W 가 최대가 되는 배치와 그 효율."""
    best_b, best_e = None, 0.0
    for b in batches:
        p = b.get("power_avg_w")
        if p and b.get("agg_out_tps"):
            e = b["agg_out_tps"] / p
            if e > best_e:
                best_b, best_e = b["batch"], e
    return best_b, best_e


def build_report(rngd: dict, gpu: Optional[dict]) -> str:
    L = []
    rb = rngd["batches"]
    gb = gpu["batches"] if gpu else None
    meta_r = rngd["meta"]
    meta_g = gpu["meta"] if gpu else None

    L.append("# RNGD vs RTX PRO 6000 — Qwen3-32B 블로그 재현 리포트\n")
    L.append("> 출처 블로그: https://furiosa.ai/blog/rngd-rtx-pro-6000-real-world-efficiency-benchmark-qwen3")
    L.append(f"> 워크로드: ISL≈{meta_r['isl_target']} / OSL={meta_r['osl']} (고정), "
             f"배치=동시 사용자, 배치별 {meta_r['window_s']}s 정상상태 측정.\n")
    L.append("**중요 — 블로그 수치의 성격**: 블로그의 *1.8x/1.9x/2.0x*(SLO 20/30/40)는 본문에 "
             "*\"normalized for rack power\"*(랙 전력 정규화 후)로 명시돼 있습니다. 장비 1:1에 가장 가까운 "
             "본문 수치는 40 TPS에서 *서버당 46 vs 41명(≈1.12x)* 뿐입니다. 따라서 1장 vs 1대 재현에서는 "
             "**(A) raw 사용자 수는 ~1.1x, (B) 전력정규화(users/kW·tokens/s/W)에서 ~2x** 가 나오는 것이 "
             "블로그를 정확히 재현·설명하는 결과입니다.\n")
    L.append(f"- RNGD: {meta_r.get('label','')} (model `{meta_r['model'].split('/')[-1]}`)")
    if meta_g:
        L.append(f"- PRO 6000: {meta_g.get('label','')} (model `{meta_g['model']}`)")
    else:
        L.append("- PRO 6000: **결과 아직 없음** — `run_pro6000.sh` 실행 후 pro6000.json 을 넘기면 비교가 채워집니다.")
    L.append("")

    # ---- 표 1: 배치별 원시 측정 ----
    L.append("## 1. 배치별 측정 (raw, 장비 1개 기준)\n")
    head = "| 배치(=사용자) | RNGD 집계TPS | RNGD per-user TPS | RNGD TTFT p50 | RNGD 전력W"
    if gb:
        head += " | GPU 집계TPS | GPU per-user TPS | GPU TTFT p50 | GPU 전력W"
    head += " |"
    L.append(head)
    L.append("|" + "---|" * (5 + (4 if gb else 0)))
    gmap = by_batch(gpu) if gpu else {}
    for b in rb:
        row = (f"| {b['batch']} | {fmt(b['agg_out_tps'])} | {fmt(b['per_user_out_tps_p50'],2)} | "
               f"{fmt(b['ttft_p50_s'],2)}s | {fmt(b['power_avg_w'])}")
        if gb:
            g = gmap.get(b["batch"], {})
            row += (f" | {fmt(g.get('agg_out_tps'))} | {fmt(g.get('per_user_out_tps_p50'),2)} | "
                    f"{fmt(g.get('ttft_p50_s'),2)}s | {fmt(g.get('power_avg_w'))}")
        L.append(row + " |")
    L.append("")

    # ---- 표 2: (A) SLO당 최대 사용자 (raw per-device) ----
    L.append("## 2. (A) SLO당 최대 사용자 — raw, 장비 1개\n")
    L.append("per-user 출력 TPS(p50)가 SLO 이상을 유지하는 최대 동시성(보간). 블로그 1:1 근사 ≈1.1x.\n")
    L.append("| SLO (TPS/user) | RNGD 사용자 | GPU 사용자 | RNGD/GPU |")
    L.append("|---|---|---|---|")
    for slo in SLOS:
        ur = users_at_slo(rb, slo)
        ug = users_at_slo(gb, slo) if gb else float("nan")
        L.append(f"| {slo} | {fmt(ur,1)} | {fmt(ug,1)} | {ratio(ur,ug) if gb else '—'} |")
    L.append("")

    # ---- 표 3: (B) 전력정규화 ----
    L.append("## 3. (B) 전력 정규화 — users/kW & tokens/s/W (블로그 핵심)\n")
    rp, _ = peak_eff(rb)
    re_b = next((x for x in rb if x["batch"] == rp), None)
    L.append("### 3a. SLO당 전력당 사용자 (users per kW)\n")
    L.append("= (위 raw 사용자 수) ÷ (그 동작점 전력 kW). 블로그의 'normalized for rack power'에 대응.\n")
    L.append("| SLO | RNGD users/kW | GPU users/kW | RNGD/GPU |")
    L.append("|---|---|---|---|")
    def users_per_kw(batches, slo):
        u = users_at_slo(batches, slo)
        if math.isnan(u) or u <= 0:
            return float("nan")
        # 그 사용자 수에 해당하는 배치의 전력(가장 가까운 측정 배치)
        cand = min(batches, key=lambda b: abs(b["batch"] - u))
        p = cand.get("power_avg_w")
        return u / (p / 1000.0) if p else float("nan")
    for slo in SLOS:
        ur = users_per_kw(rb, slo)
        ug = users_per_kw(gb, slo) if gb else float("nan")
        L.append(f"| {slo} | {fmt(ur,1)} | {fmt(ug,1)} | {ratio(ur,ug) if gb else '—'} |")
    L.append("")
    L.append("### 3b. 에너지 효율 tokens/sec/W (집계 처리량 ÷ 전력)\n")
    L.append("| 배치 | RNGD tok/s/W | GPU tok/s/W | RNGD/GPU |")
    L.append("|---|---|---|---|")
    for b in rb:
        er = (b["agg_out_tps"] / b["power_avg_w"]) if (b.get("power_avg_w") and b.get("agg_out_tps")) else float("nan")
        eg = float("nan")
        if gb:
            g = gmap.get(b["batch"], {})
            if g.get("power_avg_w") and g.get("agg_out_tps"):
                eg = g["agg_out_tps"] / g["power_avg_w"]
        L.append(f"| {b['batch']} | {fmt(er,3)} | {fmt(eg,3) if gb else '—'} | {ratio(er,eg) if gb else '—'} |")
    L.append("")

    # ---- 표 4: (C) TTFT ----
    L.append("## 4. (C) TTFT 비교 (블로그: RNGD ≈ 절반)\n")
    L.append("| 배치 | RNGD TTFT p50 | RNGD p90 | GPU TTFT p50 | GPU p90 | GPU/RNGD(p50) |")
    L.append("|---|---|---|---|---|---|")
    for b in rb:
        g = gmap.get(b["batch"], {}) if gb else {}
        gp50 = g.get("ttft_p50_s")
        rr = ratio(gp50, b["ttft_p50_s"]) if (gb and gp50) else "—"
        L.append(f"| {b['batch']} | {fmt(b['ttft_p50_s'],2)}s | {fmt(b['ttft_p90_s'],2)}s | "
                 f"{fmt(gp50,2) if gb else '—'}{'s' if gb and gp50 else ''} | "
                 f"{fmt(g.get('ttft_p90_s'),2) if gb else '—'}{'s' if gb and g.get('ttft_p90_s') else ''} | {rr} |")
    L.append("")

    # ---- 헤드라인 ----
    L.append("## 5. 요약 — 블로그 주장과 대조\n")
    if gb:
        # 단일스트림 천장(b1 per-user TPS) — 어느 SLO 를 애초에 만족 가능한지 결정
        ceil_r = max((b["per_user_out_tps_p50"] for b in rb if b["batch"] == 1), default=0)
        ceil_g = max((b["per_user_out_tps_p50"] for b in gb if b["batch"] == 1), default=0)
        L.append(f"- **단일스트림 천장(b1 per-user TPS)**: RNGD {fmt(ceil_r,1)} vs GPU {fmt(ceil_g,1)}. "
                 f"이 값 미만의 SLO만 만족 가능(1카드/1대 기준).")
        L.append("- **(A) raw 사용자 / (B) users/kW — SLO별**:")
        for slo in SLOS:
            ur, ug = users_at_slo(rb, slo), users_at_slo(gb, slo)
            kr, kg = users_per_kw(rb, slo), users_per_kw(gb, slo)
            note = ""
            if ur == 0 and ug == 0:
                note = " (양쪽 모두 단일스트림 천장 미달 — 도달 불가)"
            elif ur == 0:
                note = " (RNGD 1카드 단일스트림 천장 미달)"
            L.append(f"    - SLO {slo}: raw {fmt(ur,1)} vs {fmt(ug,1)} = **{ratio(ur,ug)}**, "
                     f"users/kW {fmt(kr,1)} vs {fmt(kg,1)} = **{ratio(kr,kg)}**{note}")
        rp_eff = peak_eff(rb)[1]; gp_eff = peak_eff(gb)[1]
        L.append(f"- **(B) 피크 에너지효율 tokens/s/W**: RNGD {fmt(rp_eff,3)} vs GPU {fmt(gp_eff,3)} → "
                 f"**{ratio(rp_eff,gp_eff)}** ← 블로그의 전력효율 우위 핵심 지표.")
        L.append("- **(C) TTFT**: 위 표 참조(GPU/RNGD>1 이면 RNGD가 더 빠름 = 블로그의 '절반' 주장 부합).")
        L.append("\n> 해석: 1장 vs 1대 raw 비교에서 GPU가 단일스트림이 더 빨라 높은 SLO(30/40)의 raw 사용자 "
                 "수는 GPU가 많을 수 있습니다. 블로그의 RNGD 우위는 **(B) 전력 정규화(users/kW·tokens/s/W)** "
                 "에서 드러나야 하며, 이게 블로그가 강조한 'normalized for rack power' 1.8~2.x 의 본질입니다. "
                 "raw 처리량이 아니라 **와트당 처리량**으로 비교하세요.")
    else:
        L.append("GPU(pro6000.json) 결과가 있으면 여기서 raw·전력정규화 비율과 블로그 대조가 채워집니다.")
        L.append("\n현재 RNGD 단독 측정 요약:")
        for slo in SLOS:
            L.append(f"- SLO {slo} TPS/user → RNGD 최대 사용자 ≈ {fmt(users_at_slo(rb,slo),1)}")
        rp_b, rp_e = peak_eff(rb)
        L.append(f"- 피크 에너지효율: 배치 {rp_b} 에서 {fmt(rp_e,3)} tokens/s/W")
    L.append("")
    L.append("---")
    L.append("_방법·한계·재현법은 README.md 참조. 블로그가 시퀀스 길이·엔진·TP·정밀도·user 공식을 "
             "공개하지 않아 절대 수치 일치가 아닌 **방법론·비율 재현**임._")
    return "\n".join(L)


def make_charts(rngd: dict, gpu: Optional[dict], outdir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    except Exception:
        print("[compare] matplotlib 없음 → 차트 생략(표 리포트만). 필요시 pip install matplotlib")
        return []
    outdir.mkdir(parents=True, exist_ok=True)
    rb = rngd["batches"]
    series = [("RNGD", rb, "tab:blue")]
    if gpu:
        series.append(("PRO6000", gpu["batches"], "tab:green"))
    made = []

    # 실제 측정한 배치 값들(예: 1,8,16,32,64,256)을 그대로 눈금으로 사용.
    xticks = sorted({b["batch"] for _, bs, _ in series for b in bs})

    def plain_log2_x():
        """log2 x축 눈금을 2^k 거듭제곱 표기 대신 평범한 정수(1,8,16,…)로 표시."""
        ax = plt.gca()
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in xticks]))
        ax.xaxis.set_minor_locator(NullLocator())

    def line(key, ylabel, fname, slo_lines=False, ylog=False):
        plt.figure(figsize=(7, 4.5))
        for name, bs, c in series:
            xs = [b["batch"] for b in bs]
            ys = [b.get(key) for b in bs]
            plt.plot(xs, ys, "o-", label=name, color=c)
        if slo_lines:
            for slo in SLOS:
                plt.axhline(slo, ls="--", color="gray", lw=0.8)
                plt.text(xs[0], slo, f" SLO {slo}", va="bottom", fontsize=8, color="gray")
        plt.xlabel("batch (= concurrent users)"); plt.ylabel(ylabel)
        plt.xscale("log", base=2)
        plain_log2_x()
        if ylog:
            plt.yscale("log")
        plt.grid(True, alpha=0.3); plt.legend(); plt.title(ylabel)
        p = outdir / fname; plt.tight_layout(); plt.savefig(p, dpi=120); plt.close()
        made.append(str(p))

    line("per_user_out_tps_p50", "per-user output TPS (p50)", "per_user_tps.png", slo_lines=True)
    line("agg_out_tps", "aggregate output TPS", "agg_tps.png")
    line("ttft_p50_s", "TTFT p50 (s)", "ttft.png")
    # tokens/s/W
    plt.figure(figsize=(7, 4.5))
    for name, bs, c in series:
        xs = [b["batch"] for b in bs]
        ys = [(b["agg_out_tps"] / b["power_avg_w"]) if (b.get("power_avg_w") and b.get("agg_out_tps")) else None for b in bs]
        plt.plot(xs, ys, "o-", label=name, color=c)
    plt.xlabel("batch (= concurrent users)"); plt.ylabel("tokens/sec/W")
    plt.xscale("log", base=2); plain_log2_x(); plt.grid(True, alpha=0.3); plt.legend(); plt.title("energy efficiency (tokens/sec/W)")
    p = outdir / "tokens_per_watt.png"; plt.tight_layout(); plt.savefig(p, dpi=120); plt.close()
    made.append(str(p))
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rngd", help="results/rngd.json")
    ap.add_argument("gpu", nargs="?", default=None, help="results/pro6000.json (선택)")
    ap.add_argument("--out", default=None, help="리포트 md 경로(기본 rngd.json 옆 report.md)")
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()

    rngd = load(args.rngd)
    if rngd is None:
        raise SystemExit(f"RNGD 결과 없음: {args.rngd}")
    gpu = load(args.gpu)

    out = Path(args.out) if args.out else Path(args.rngd).parent / "report.md"
    report = build_report(rngd, gpu)
    out.write_text(report)
    print(report)
    print(f"\n[compare] 리포트 저장: {out}")
    if not args.no_charts:
        charts = make_charts(rngd, gpu, out.parent / "charts")
        if charts:
            print("[compare] 차트:", *charts, sep="\n  ")


if __name__ == "__main__":
    main()
