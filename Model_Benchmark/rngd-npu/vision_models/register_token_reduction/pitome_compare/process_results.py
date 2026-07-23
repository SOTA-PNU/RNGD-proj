#!/usr/bin/env python3
"""GPU서 전송된 pitome_compare 결과(results_acc.txt / results_tput.txt)를 파싱 →
   (1) 논문용 마크다운 표, (2) 정확도·throughput 그림(ACCV/fig_pitome_compare.png), (3) JSON 요약 출력.
compare.py 의 출력 포맷에 맞춰 파싱. 데이터 도착 후 자동 실행됨.
사용: python process_results.py"""
import re, os, json

HERE = os.path.dirname(os.path.abspath(__file__))
ACC = os.path.join(HERE, "results_acc.txt")
TPUT = os.path.join(HERE, "results_tput.txt")
FIG = os.path.abspath(os.path.join(HERE, "..", "..", "..", "ACCV", "fig_pitome_compare.png"))

C_BY_SIZE = {"small": 384, "base": 768, "large": 1024, "giant": 1536}


def parse_setup(path):
    """results 파일의 [setup] 줄에서 모델 차원 파싱: prefix·patches·blocks + 모델명→C."""
    line = open(path).readline()
    def g(k, d):
        m = re.search(rf"{k}=(\d+)", line); return int(m.group(1)) if m else d
    prefix, patches, blocks = g("prefix", 5), g("patches", 256), g("blocks", 12)
    C = 768
    for sz, c in C_BY_SIZE.items():
        if sz in line: C = c; break
    return {"prefix": prefix, "T0": prefix + patches, "blocks": blocks, "C": C}


def _blk_macs(N, C):
    """ViT 블록 1개의 곱셈-누산(MAC) 수: QKV+out proj+MLP(선형, 12NC²) + attention(2N²C)."""
    return 12 * N * C * C + 2 * N * N * C


def _total_macs(dims, r, nprot):
    """압축 설정(r)의 전체 MAC 수(patch-embed conv 포함). 토큰 스케줄은 결정론적."""
    T0, L, C = dims["T0"], dims["blocks"], dims["C"]
    patches = T0 - dims["prefix"]
    T, tot = T0, 0
    for _ in range(L):
        tot += _blk_macs(T, C)
        T -= min(r, max((T - nprot) // 2, 0))
    tot += patches * (14 * 14 * 3) * C          # patch14 임베딩 conv
    return tot


def flops_saved_pct(dims, r, nprot):
    """실제 FLOP 절감%(무압축 대비). 결정론적, GPU 불필요."""
    return 100.0 * (1 - _total_macs(dims, r, nprot) / _total_macs(dims, 0, nprot))


def gflops(dims, r, nprot):
    """이미지 1장당 GFLOPs. 논문 표준 관행(fvcore/thop: 곱셈-누산 1회=1)으로 셈 — ToMe·PiToMe와 동일 잣대."""
    return _total_macs(dims, r, nprot) / 1e9


def parse_acc(path):
    """compare.py acc 출력 → [{r,comp,tome,pitome,ours,delta}] + baseline(r=0)."""
    rows, base = [], None
    for ln in open(path):
        m = re.match(r"\s*(\d+)\s+([\d.]+)\s+(.*)", ln)
        if not m:
            continue
        r, comp, rest = int(m.group(1)), float(m.group(2)), m.group(3)
        nums = re.findall(r"-?\d+\.\d+", rest)
        if r == 0:
            base = float(nums[0]) if nums else None
        elif len(nums) >= 3:
            rows.append({"r": r, "comp": comp, "tome": float(nums[0]),
                         "pitome": float(nums[1]), "ours": float(nums[2]),
                         "delta": round(float(nums[2]) - float(nums[1]), 2)})
    return base, rows


def parse_tput(path):
    """compare.py tput 출력 → [{r,comp,tome,pitome,ours}] (im/s 정수)."""
    rows = []
    for ln in open(path):
        m = re.match(r"\s*(\d+)\s+([\d.]+)\s+(.*)", ln)
        if not m:
            continue
        r, comp, rest = int(m.group(1)), float(m.group(2)), m.group(3)
        nums = re.findall(r"\d+", rest)
        if len(nums) >= 3:
            rows.append({"r": r, "comp": comp, "tome": int(nums[0]),
                         "pitome": int(nums[1]), "ours": int(nums[2])})
    return rows


def md_table(base, acc, tput):
    out = []
    if base is not None:
        out.append(f"무압축(r=0) baseline = **{base:.2f}** (val leave-one-out k-NN, k=20)\n")
    if acc:
        out.append("| 토큰압축% | FLOP절감% | GFLOPs | ToMe | PiToMe | Ours | Δ(Ours−PiToMe) |")
        out.append("|---|---|---|---|---|---|---|")
        for x in acc:
            fl = f"{x['flop']:.1f}" if "flop" in x else "-"
            gf = f"{x['gflops']:.1f}" if "gflops" in x else "-"
            out.append(f"| {x['comp']:.1f} | {fl} | **{gf}** | {x['tome']:.2f} | {x['pitome']:.2f} | **{x['ours']:.2f}** | **{x['delta']:+.2f}** |")
        out.append("")
    if tput:
        out.append("**Throughput (im/s):**")
        out.append("| 압축률(comp%) | ToMe | PiToMe | Ours |")
        out.append("|---|---|---|---|")
        for x in tput:
            out.append(f"| {x['comp']:.1f} | {x['tome']} | {x['pitome']} | {x['ours']} |")
        out.append("")
    return "\n".join(out)


def make_fig(base, acc, tput):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return f"(그림 생략: matplotlib 없음 — {e})"
    # 색맹-안전 팔레트(전 그림 공통 identity): Ours=블루, ToMe=버밀리언, PiToMe=그린
    OURS, TOME, PITOME, INK = "#0072B2", "#D55E00", "#009E73", "#4d4d4d"
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.linewidth": 0.8,
                         "axes.unicode_minus": False})
    n = 1 + (1 if tput else 0)
    fig, axs = plt.subplots(1, n, figsize=(5.0 * n, 4.0))
    axs = [axs] if n == 1 else list(axs)
    if acc:
        xs = [x["flop"] for x in acc]                    # x축=FLOP 절감%(공정한 계산 예산)
        ax = axs[0]
        if base is not None:
            ax.axhline(base, ls=(0, (1, 1.6)), lw=1.0, color=INK, alpha=.6)
            ax.text(xs[0], base + 0.1, f"Uncompressed  {base:.1f}", ha="left", va="bottom", fontsize=8.5, color=INK, alpha=.8)
        ax.plot(xs, [x["tome"] for x in acc], "--s", color=TOME, lw=2.0, ms=6, mec="white", mew=0.8, label="ToMe", zorder=2)
        ax.plot(xs, [x["pitome"] for x in acc], "-.D", color=PITOME, lw=2.0, ms=6, mec="white", mew=0.8, label="PiToMe", zorder=3)
        ax.plot(xs, [x["ours"] for x in acc], "-o", color=OURS, lw=2.4, ms=6.5, mec="white", mew=0.8, label="Ours", zorder=4)
        ax.set_xlabel("FLOP reduction (%)"); ax.set_ylabel("kNN top-1 (%)")
        ax.set_title("Accuracy vs FLOP reduction")
        ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, fontsize=10, loc="lower left")
    if tput:
        xs = [x["flop"] for x in tput]                   # 좌우 x축 통일: FLOP 절감%
        ax = axs[-1]
        ax.plot(xs, [x["tome"] for x in tput], "--s", color=TOME, lw=2.0, ms=6, mec="white", mew=0.8, label="ToMe", zorder=2)
        ax.plot(xs, [x["pitome"] for x in tput], "-.D", color=PITOME, lw=2.0, ms=6, mec="white", mew=0.8, label="PiToMe", zorder=3)
        ax.plot(xs, [x["ours"] for x in tput], "-o", color=OURS, lw=2.4, ms=6.5, mec="white", mew=0.8, label="Ours", zorder=4)
        ax.set_xlabel("FLOP reduction (%)"); ax.set_ylabel("Throughput (im/s)")
        ax.set_title("GPU throughput")
        ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, fontsize=10, loc="upper left")
    for ax in axs:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(length=3, color="#c9ccd1")
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.tight_layout(w_pad=2.2); fig.savefig(FIG, dpi=200, bbox_inches="tight"); plt.close(fig)
    return FIG


def main():
    base, acc, tput = None, [], []
    if os.path.exists(ACC):
        base, acc = parse_acc(ACC)
        dims = parse_setup(ACC)
        for x in acc:   # FLOP(결정론적, 방법간 ±0.1%라 nprot=1 대표값). GFLOPs=MAC관행(=GMACs)
            x["flop"] = flops_saved_pct(dims, x["r"], 1)
            x["gflops"] = gflops(dims, x["r"], 1)
    if os.path.exists(TPUT):
        tput = parse_tput(TPUT)
        dimt = parse_setup(TPUT)
        for x in tput:   # 처리량 패널도 x축을 FLOP 절감%로 통일
            x["flop"] = flops_saved_pct(dimt, x["r"], 1)
    if not acc and not tput:
        print("[process] 결과 파일 없음/미완 — results_acc.txt / results_tput.txt 대기"); return
    table = md_table(base, acc, tput)
    figpath = make_fig(base, acc, tput)
    summary = {"baseline": base, "acc": acc, "tput": tput, "fig": figpath}
    with open(os.path.join(HERE, "results_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("=" * 60); print(table); print("=" * 60)
    print(f"[fig] {figpath}")
    # 핵심 한 줄: 최고 압축 지점 요약
    if acc:
        x = max(acc, key=lambda a: a["comp"])
        fl = f", FLOP {x['flop']:.0f}% 절감" if "flop" in x else ""
        print(f"[핵심] 토큰 {x['comp']:.0f}%{fl} 지점: Ours {x['ours']:.2f} vs PiToMe {x['pitome']:.2f} vs ToMe {x['tome']:.2f} "
              f"(Ours−PiToMe {x['delta']:+.2f})")
        print(f"[FLOP축] 우리 sweep = FLOP {acc[0]['flop']:.0f}~{x['flop']:.0f}% 절감 (PiToMe 논문 보고 40~60%와 같은 잣대)")


if __name__ == "__main__":
    main()
