#!/usr/bin/env python3
"""①(우리 포팅)과 ②(공식 repo)의 DeiT 결과를 나란히 대조하고 판정.

results/ 의 JSON 을 읽음:
  ours_port__<model>.json   (deit_compare.py 생성)
  official__<model>.json     (official_deit_driver.py 생성; 있으면)
같은 모델·같은 ratio 에서 top-1 과 (pitome−tome) 격차를 비교한다. 공식이 아직 없으면 우리 곡선만 +
공식 논문 공개 참조치를 보여준다.

판정: 같은 ratio 에서 |우리_pitome − 공식_pitome| 과 |baseline 차| 이 THRESH 안이면 '포팅=공식' 확인.
"""
import os, json, glob

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
THRESH = 0.5   # top-1 %p — GPU 비결정성/미세 구현차 허용폭

# 공식 논문(arXiv:2405.16148 Table5/6, off-the-shelf) 참조 — repo 미실행 시 눈대중용.
REF = {
    "deit_small_patch16_224": "공식 논문 DeiT-S: baseline 79.8, off-the-shelf PiToMe−ToMe ≈ +1.4 (79.1 vs 77.7)",
    "deit_tiny_patch16_224":  "공식 논문 DeiT-T: baseline 72.3, off-the-shelf PiToMe−ToMe ≈ +1.9 (70.8 vs 68.9)",
    "deit_base_patch16_224":  "공식 논문에 DeiT-B 없음 → 참조치 없음(우리 곡선 단독)",
}


def load(pat):
    d = {}
    for f in glob.glob(os.path.join(RES, pat)):
        j = json.load(open(f)); d[j["model"]] = j
    return d


def rows_by_ratio(j):
    return {r["ratio"]: r for r in j["rows"]}


def main():
    lines = []
    def P(s=""): print(s); lines.append(s)

    ours = load("ours_port__*.json")
    offi = load("official__*.json")
    if not ours and not offi:
        print("결과 JSON 없음. 먼저 bash run.sh (①) [+ bash run_official_pitome.sh (②)] 실행."); return

    for model in sorted(set(ours) | set(offi)):
        P("=" * 78)
        P(f"모델: {model}")
        P(f"  참조: {REF.get(model, '(없음)')}")
        o, f = ours.get(model), offi.get(model)
        if o: P(f"  ① 우리 포팅  baseline r=0 = {o['baseline_r0']}  (n_val={o['n_val']})")
        if f: P(f"  ② 공식 repo  baseline r=0 = {f['baseline_r0']}  (n_val={f['n_val']})")

        if o and f:                                              # 완전 대조
            db = abs(o["baseline_r0"] - f["baseline_r0"])
            P(f"\n  {'ratio':>6} {'comp%':>6} | {'①tome':>7} {'②tome':>7} {'Δt':>6} | {'①pito':>7} {'②pito':>7} {'Δp':>6}")
            ro, rf = rows_by_ratio(o), rows_by_ratio(f); worst = db
            for ratio in sorted(set(ro) & set(rf), reverse=True):
                a, b = ro[ratio], rf[ratio]
                dt, dp = abs(a["tome"] - b["tome"]), abs(a["pitome"] - b["pitome"])
                worst = max(worst, dp)
                P(f"  {ratio:6.3f} {a['comp']:6.1f} | {a['tome']:7.2f} {b['tome']:7.2f} {dt:6.2f} | "
                  f"{a['pitome']:7.2f} {b['pitome']:7.2f} {dp:6.2f}")
            verdict = "✅ 포팅=공식 일치" if worst < THRESH else "⚠️ 차이 있음 → 마진/에너지/병합순서 점검"
            P(f"\n  판정: 최대 top-1 차 {worst:.2f}%p (baseline포함) → {verdict}  (허용 {THRESH})")
        elif o:                                                  # 우리만
            P(f"\n  {'ratio':>6} {'comp%':>6} {'tome':>7} {'pitome':>7} {'Δ(P-T)':>8}")
            for r in o["rows"]:
                P(f"  {r['ratio']:6.3f} {r['comp']:6.1f} {r['tome']:7.2f} {r['pitome']:7.2f} {r['delta_PT']:+8.2f}")
            P("  (공식 repo 미실행 → 위 (pitome−tome) 를 참조치와 눈대중 대조. bash run_official_pitome.sh 로 실측 대조.)")
        else:
            P("  (우리 포팅 결과 없음 → bash run.sh 먼저.)")
    P("=" * 78)

    outp = os.path.join(RES, "comparison_report.txt")
    os.makedirs(RES, exist_ok=True)
    open(outp, "w").write("\n".join(lines) + "\n")
    print(f"\n[저장] {outp}")


if __name__ == "__main__":
    main()
