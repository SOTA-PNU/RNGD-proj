#!/usr/bin/env python3
"""쓸 수 있는 모델 정리 덱. 수치는 전부 ../<summary> 에서 읽는다.

서사: 어떤 모델이 있나 → 어떻게 쟀나(코드) → 출력 한도 → 시간 축 → 지연 → 처리량 → 적재 → 답변.
머리글은 "그림 N: 무엇을 그렸나"(L-38), 코드 캡션은 "코드 N: 파일, 무엇"(L-39).

같은 코드로 두 판을 만든다.
  기본  summary.json(analyze.py), 2026-08-29, max_tokens 1024 → 모델별-성능정리.pptx
  새판  DECK_SUMMARY=summary_len.json DECK_CAP=<적정 한도> DECK_DATE=2026-09-11 DECK_OUT=<파일>
        length_probe.py 결과. 출력 한도 그림과 표, 멈춤을 뺀 속도가 더 붙는다.
"""
import sys, os, json, itertools
sys.path.insert(0, "/home/jun/.claude/skills/diagram-deck/scripts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import deck
from check_shapes import compare, check_group_tight
SRC = os.path.dirname(os.path.abspath(__file__))
SUM = json.load(open(os.path.join(SRC, "..", os.environ.get("DECK_SUMMARY", "summary.json")), encoding="utf-8"))
BY = {d["model"]: d for d in SUM}
# 출력 길이(finish_reason)가 있으면 새판이다.
LEN = any(r.get("finish_reason") for d in SUM for r in d.get("runs") or [])
CAP = int(os.environ.get("DECK_CAP", "0")) or None
DATE = os.environ.get("DECK_DATE", "2026-08-29")
PROBE = max((r.get("max_tokens") or 0) for d in SUM for r in d.get("runs") or []) if LEN else None
MT = CAP if (LEN and CAP) else 1024
OLD = ({d["model"]: d for d in json.load(open(os.path.join(SRC, "..", "summary.json"), encoding="utf-8"))}
       if LEN else None)
FIG = itertools.count(1)
CODE = itertools.count(1)


def load(n):
    """SVG 를 읽으면서 조립 전에 먼저 확인한다 — 문제를 pptx 를 만든 뒤가 아니라 여기서 잡는다.

    ① XML 로 파싱되는가 (글자 안 & < > 이스케이프 누락이면 여기서 걸린다)
    ② 내용 폭이 828 인가, 높이가 234 이하인가 (다르면 슬라이드마다 축척이 달라진다, L-30)
    """
    import xml.etree.ElementTree as ET
    from svg2shapes import content_bbox
    svg = open(os.path.join(SRC, n + ".svg"), encoding="utf-8").read()
    try:
        ET.fromstring(svg)
    except ET.ParseError as e:
        raise SystemExit(f"[{n}.svg] XML 이 깨졌다: {e}\n  글자 안의 & < > 를 esc() 로 감쌌는지 보라")
    x0, y0, x1, y1 = content_bbox(svg)
    w, h = round(x1 - x0), round(y1 - y0)
    if w != 828 or h > 234:
        print(f"  ! {n}.svg 규격 벗어남 — 폭 {w}(828 이어야), 높이 {h}(234 이하여야)")
    return svg


def n_done():
    import gen_all
    return sum(1 for mid, _, _ in gen_all.MODELS if mid in BY and "runs" in BY[mid])


d = deck.Deck("RNGD 서버에서 쓸 수 있는 모델",
              ("채팅 모델 전종을 같은 입력, 같은 출력 한도로 재 본 처리량, 지연, 그리고 실제 답변" if LEN else
               "채팅 모델 전종을 같은 입력으로 재 본 처리량, 지연, 그리고 실제 답변"),
              f"RNGD 4장 서버, {DATE} 실측" + (f", max_tokens {MT:,}" if LEN else ""))
srcs = {}


def dia(title, name, head, items):
    svg = load(name)
    s, n, mp, _ = d.diagram_svg(title, svg, items, None, None, None,
                                name="도해 " + name, cols=1, head=head)
    srcs["도해 " + name] = svg
    print(f"{name}: 도형 {n}개, 최소 글자 {mp:.1f}pt")


def rec_of(mid):
    r = BY.get(mid)
    return r if r and "runs" in r else None


def lab_of(mid):
    import gen_all
    return next(lab for m, lab, _ in gen_all.MODELS if m == mid)


def length_items():
    """출력 한도 그림 아래 글머리. 숫자는 전부 두 summary 에서 센다."""
    import gen_all
    old_cut = old_n = 0
    for mid, _, _ in gen_all.MODELS:
        for r in (OLD.get(mid) or {}).get("runs") or []:
            old_n += 1
            old_cut += (r.get("out_tokens") or 0) >= 1024
    runs = [(mid, r) for mid, _, _ in gen_all.MODELS if rec_of(mid) for r in rec_of(mid)["runs"] if not r.get("error")]
    stop = [x for x in runs if x[1].get("finish_reason") == "stop"]
    top_mid, top = max(runs, key=lambda x: x[1]["out_tokens"])
    conc = [(mid, q["out_tokens"]) for mid, _, _ in gen_all.MODELS if rec_of(mid)
            for q in (rec_of(mid).get("concurrent") or {}).get("per_req", [])]
    cmid, cmax = max(conc, key=lambda x: x[1]) if conc else (None, 0)
    items = [
        {"t": f"2026-08-29 판은 max_tokens 1024 로 쟀고, 단일 요청 {old_n}개 중 {old_cut}개가 그 한도에 걸려 잘렸다. "
              f"잘린 것은 거의 전부 사고하는 모델이었고, 그중 여럿은 답변을 한 글자도 못 내고 사고만 하다 끝났다"},
        {"t": f"한도를 {PROBE:,} 로 풀자 단일 요청 {len(runs)}개 중 {len(stop)}개가 스스로 멈췄다. "
              f"가장 긴 것은 {lab_of(top_mid)} 의 {top['out_tokens']:,}토큰이고 그중 사고가 "
              f"{(top.get('think_tok') or 0):,}토큰이다. 동시 요청은 배치 탓에 출력이 요청마다 갈라져 "
              f"{lab_of(cmid)} 가 {cmax:,}토큰까지 나왔다"},
    ]
    if CAP:
        items.append({"t": f"새 한도 {CAP:,} 는 동시 요청까지 포함한 가장 긴 출력을 여유 있게 품는다. "
                           f"Qwen2.5 0.5B 는 서빙 컨텍스트가 4,096 이라 그 이상을 요청하면 서버가 400 으로 거절하므로, "
                           f"이 모델만 컨텍스트에서 프롬프트 몫을 뺀 값으로 줄여 보낸다"})
    return items


def stall_items():
    """멈춤을 뺀 속도 그림 아래 글머리."""
    import gen_all
    hit = [(mid, rec_of(mid)) for mid, _, _ in gen_all.MODELS if rec_of(mid) and rec_of(mid).get("stall_lost_s", 0) > 1]
    clean = [(mid, rec_of(mid)) for mid, _, c in gen_all.MODELS if rec_of(mid) and c == 4 and rec_of(mid).get("stall_lost_s", 0) <= 1]
    items = [
        {"t": "토큰이 오는 시각을 전부 기록해 조각 사이 간격을 봤다. 정상 간격으로 계산한 속도는 08-29 속도와 같다. "
              "모델이 느려진 것이 아니라 스텝 사이에 0.1~1초씩 멈추는 간격이 끼어든 것이다"},
        {"t": "멈춤은 몇 분 단위로 밀려왔다 빠진다. 같은 모델도 시간대에 따라 0회에서 수천 회까지 달랐고, "
              "동시 요청이 가장 크게 다쳤다. A3B Instruct 2507 의 동시 4요청은 16:49 에 35 tok/s 였는데 19:3x 에 다시 재니 "
              "141~143 tok/s 로 08-29(142)와 같았다. 서버 로그의 생성량도 같은 시간에 떨어졌으니 서버 쪽 멈춤이다"},
        {"t": "smi 호출, max_tokens, 라우터, CPU 경합, 카드 한 장의 열화, 온도, 노드 메모리 부족은 실험으로 배제했다. "
              "08-29 이후 바뀐 것은 09-01 재부팅 때 커널이 6.17 에서 7.0 으로 올라간 것이다"},
    ]
    return items, hit


def build():
    dia("모델 목록과 카드 점유", "lineup",
        head=f"그림 {next(FIG)}: 라우터에 등록된 채팅 모델 전종과 각각이 차지하는 카드 수",
        items=[
            {"t": "라우터(:8400)에 채팅 모델 15종이 등록돼 있다. 이 밖에 임베딩과 리랭커가 하나씩 더 있지만 대화용이 아니라 이 정리에서는 뺐다"},
            {"t": "열한 종은 카드 넉 장을 통째로 쓴다. 그래서 그중 하나만 떠 있을 수 있고, 다른 것을 부르면 쓰던 것을 내리고 새로 올린다"},
            {"t": "네 종은 카드 한 장이면 되므로 넷까지 동시에 떠 있을 수 있다. 여러 사람이 각자 다른 모델을 쓸 때 이 차이가 크다"},
        ])
    # 제원 표 — 이름이 아니라 각 모델의 config.json 과 라우터 REGISTRY 에서 뽑은 값이다.
    d.table("모델 제원",
            ["모델", "가중치", "디스크", "구조", "정밀도", "컨텍스트", "카드", "용도"],
            SPECS,
            subtitle="가중치는 카드에 올라간 실측치, 디스크는 HF 캐시 용량, 컨텍스트 단위는 토큰",
            note=("가중치는 serve 로그의 'Total size of parameters loaded' 실측이고, 디스크는 HF 캐시 폴더 용량이다. "
                  "파라미터 수는 모델 이름에 이미 들어 있다 — A3B, A23B 는 MoE 의 활성 파라미터다(전체 30B 중 3B 가 토큰마다 동작). "
                  "카드 한 장의 HBM 은 47.5 GiB 이므로, 가중치가 그보다 크면 여러 장에 나눠 올린다. "
                  "컨텍스트는 토큰 수이고 프롬프트와 생성분을 합쳐서 센다. "
                  "컨텍스트는 라우터 REGISTRY 의 ctx 로, 모델이 지원하는 최대와 다를 수 있다 — "
                  "Qwen2.5-0.5B 만 모델은 32768 을 지원하는데 4096 으로 서빙한다. "
                  "Solar-Open-100B 은 배포명이 NVFP4A16(활성 16비트)인데 config 의 input_activations 는 "
                  "4비트로 적혀 있어 이름과 어긋난다. 표에는 config 값을 적었다."))
    if LEN:
        d.code("측정 방법", [
            "# length_probe.py — bench.py 와 같은 네 프롬프트, 같은 greedy 로 한도만 키워 잰다",
            f"CAP = {PROBE}                        # 잘리지 않은 길이를 보려고 넉넉히",
            "CTX_MARGIN = 256                    # prompt + max_tokens > 컨텍스트 이면 서버가 400 으로 거절",
            "cap = min(CAP, ctx - CTX_MARGIN)    # 컨텍스트가 작은 모델(0.5B, 4096)만 줄어든다",
            "",
            "body = {'model': model, 'messages': [...], 'temperature': 0,",
            "        'max_tokens': cap, 'stream': True, 'stream_options': {'include_usage': True}}",
            "for 이벤트 in 스트림:",
            "    finish = choice['finish_reason']   # stop 이면 스스로 멈춤, length 면 한도에 걸림",
            "    times.append(지금 - t0)             # 조각마다 도착 시각, 멈춤을 찾으려고",
            "사고 토큰 = usage.completion_tokens_details.reasoning_tokens",
            "decode_tps = (출력토큰 - 1) / (total - ttft)",
        ], caption=f"코드 {next(CODE)}: length_probe.py, 한도를 정하고 끝난 이유와 조각 시각을 남기는 부분")
        dia("출력 길이와 한도", "lengths",
            head=f"그림 {next(FIG)}: 모델마다 스스로 멈춘 출력 길이와 기존 한도 1,024, 새 한도의 위치",
            items=length_items())
        import analyze_len
        cands = sorted(set(analyze_len.CANDIDATES) | ({CAP} if CAP else set()))
        rows = []
        for c in analyze_len.cap_rows(SUM, cands):
            rows.append([f"{c['cap']:,}" + (" (새 한도)" if c["cap"] == CAP else ""),
                         f"{c['cut_single']} / {c['n_single']}", f"{c['cut_conc']} / {c['n_conc']}",
                         f"{c['single_min']:.1f}분", ", ".join(lab_of(m) for m in c["cut_models"]) or "없음"])
        d.table("출력 한도 후보", ["max_tokens", "잘리는 단일 요청", "잘리는 동시 요청", "단일 4종 시간 합", "잘리는 모델"],
                rows,
                subtitle=f"{PROBE:,} 로 잰 결과에서 계산. greedy 라 한도 안에서 끝난 출력은 한도와 무관하게 같다",
                note=("잘리는 요청 = 그 한도보다 길게 나온 요청 + 측정 때부터 스스로 못 멈춘 요청. "
                      "시간 합은 전 모델의 단일 요청 네 개를 그 한도로 끊었을 때의 소요를 조각 시각에서 계산한 값이다. "
                      "공식 권장 출력 길이는 이보다 훨씬 크다(Qwen3 32768, Qwen3-Coder 65536, A3B Instruct 2507 16384, "
                      "K-EXAONE 사고 모드 16384). 이 네 짧은 프롬프트에는 그만큼이 필요 없다는 뜻이지 "
                      "어려운 문제에 충분하다는 뜻은 아니다."))
    else:
        d.code("측정 방법", [
            "# bench.py — 모델마다 (1) 준비될 때까지 기다리고 (2) 프롬프트 4종을 스트리밍으로 재고",
            "#            (3) 같은 프롬프트 4개를 동시에 던져 총처리량을 잰다",
            "PROMPTS = [사실 질문, 코드 작성, 계산 추론, 개념 설명]   # 네 종류 고정, 모든 모델에 같은 문장",
            "MAX_TOKENS = 1024                                     # 256 이면 사고하는 모델이 답까지 못 간다",
            "",
            "body = {'model': model, 'messages': [...],",
            "        'temperature': 0,                             # greedy, 무작위성 제거",
            "        'max_tokens': MAX_TOKENS, 'stream': True,",
            "        'stream_options': {'include_usage': True}}     # 토큰 수는 서버가 세 준 값을 쓴다",
            "",
            "t0 = time.perf_counter()",
            "for 이벤트 in 스트림:",
            "    본문 = delta.content;  사고 = delta.reasoning      # 사고하는 모델은 reasoning 으로 온다",
            "    if (본문 or 사고) and ttft is None:",
            "        ttft = time.perf_counter() - t0               # 첫 토큰까지 = TTFT",
            "total = time.perf_counter() - t0                      # 답이 끝날 때까지",
            "decode_tps = (출력토큰 - 1) / (total - ttft)           # 첫 토큰 뒤의 순수 생성 속도",
            "",
            "# 동시 요청: 같은 프롬프트 4개를 스레드로 동시에 → 합계 토큰 ÷ 전부 끝난 시간",
        ], caption=f"코드 {next(CODE)}: bench.py, 모든 모델에 같은 조건을 강제하는 부분과 시간을 재는 부분")
    dia("시간 측정 구간", "method",
        head=f"그림 {next(FIG)}: 한 요청의 시간 축에서 TTFT 와 디코드 구간이 각각 어디인지",
        items=[
            {"t": "TTFT 는 요청을 보낸 순간부터 첫 글자가 올 때까지다. 프롬프트를 읽는 시간이 여기 들어간다"},
            {"t": "디코드 속도는 첫 글자 이후 남은 토큰을 걸린 초로 나눈 값이라, 프롬프트 길이에 덜 휘둘린다"},
            {"t": "적재 시간은 따로 잰다. 요청이 아니라 모델을 카드에 올리는 시간이고, serve 로그에서 뽑는다"},
        ])
    import gen_all
    ft = {mid: gen_all.first_total(mid) for mid, _, _ in gen_all.MODELS if rec_of(mid)}
    tt = [rec_of(mid).get("ttft_med") for mid in ft if rec_of(mid).get("ttft_med")]
    inst, think = "Qwen3-30B-A3B-Instruct-2507-FP8", "Qwen3-30B-A3B-Thinking-2507-FP8"
    dia("응답 지연", "latency",
        head=f"그림 {next(FIG)}: 모델별 첫 토큰까지 걸린 시간과, 짧은 질문 하나가 끝날 때까지 걸린 시간",
        items=[
            {"t": f"TTFT 는 중앙값 기준 {min(tt):.2f}~{max(tt):.2f}초다. 체감 차이는 첫 글자가 아니라 답이 끝날 때까지에서 생긴다"},
            {"t": (f"별표가 붙은 사고하는 모델은 답을 내기 전에 속으로 길게 생각한다. 같은 30B 인데도 A3B Instruct 는 "
                   f"{ft.get(inst, 0):.1f}초, A3B Thinking 은 {ft.get(think, 0):.1f}초가 걸렸다")
                  + (f". 1024 한도였던 08-29 에는 A3B Thinking 이 사고 도중 잘려 14.5초로 짧게 나왔다" if LEN else "")},
            {"t": "짧게 묻고 짧게 받는 용도라면 사고 없는 모델이 훨씬 낫다. 어려운 추론을 맡길 때만 사고하는 쪽을 고른다"},
        ])
    four = [(mid, rec_of(mid)) for mid, _, c in gen_all.MODELS if rec_of(mid) and c == 4]
    one = [(mid, rec_of(mid)) for mid, _, c in gen_all.MODELS if rec_of(mid) and c == 1]
    best4 = max(four, key=lambda x: (x[1].get("concurrent") or {}).get("agg_tps") or 0)
    best1 = max(one, key=lambda x: (x[1].get("concurrent") or {}).get("agg_tps") or 0)
    dia("생성 처리량", "tput",
        head=f"그림 {next(FIG)}: 요청 하나를 처리하는 속도와, 같은 질문 4개를 동시에 던졌을 때의 전체 속도",
        items=[
            {"t": "요청 하나만 보면 큰 모델과 작은 모델의 차이가 크지 않다. 토큰마다 무게 전부를 한 번씩 읽어야 해서 "
                  "모델 크기보다 카드 수와 정밀도가 속도를 정한다"},
            {"t": f"동시에 여러 요청이 들어오면 갈린다. 카드 한 장짜리 {lab_of(best1[0])} 가 "
                  f"{best1[1]['concurrent']['agg_tps']:.0f} tok/s 로, 넉 장짜리 가운데 가장 빠른 {lab_of(best4[0])} 의 "
                  f"{best4[1]['concurrent']['agg_tps']:.0f} tok/s 보다 높다"},
            {"t": ("사람이 여럿 붙는 서비스라면 이 두 번째 숫자가 중요하다. 혼자 쓰는 도구라면 첫 번째 숫자와 앞 장의 응답 시간을 본다"
                   if not LEN else
                   "이번 측정은 스텝 사이 멈춤이 시간대에 따라 끼어, 넉 장짜리 모델 여럿이 두 숫자 모두 낮게 나왔다. 다음 장에 멈춤을 뺀 속도를 따로 적었다")},
        ])
    if LEN:
        items, hit = stall_items()
        srows = []
        for mid, lab, cards in gen_all.MODELS:
            rec = rec_of(mid)
            if not rec:
                continue
            o = (OLD.get(mid) or {}).get("decode_tps_med")
            srows.append([lab, str(cards), (rec.get("build_kind") or "-").replace("v2-artifact", "v2"),
                          f"{rec['decode_tps_med']:.0f}" if rec.get("decode_tps_med") else "-",
                          f"{rec['clean_tps_med']:.0f}" if rec.get("clean_tps_med") else "-",
                          f"{o:.0f}" if o else "-",
                          f"{rec.get('stall_n', 0)}회, {rec.get('stall_lost_s', 0):.0f}s"])
        d.table("스텝 사이 멈춤", ["모델", "카드", "아티팩트", "벽시계 tok/s", "멈춤 뺀 tok/s", "08-29 tok/s", "0.1초 넘는 멈춤"],
                srows,
                subtitle="단일 요청 네 개의 조각 도착 시각에서 계산. 멈춤 뺀 속도 = 정상 조각 간격 중앙값의 역수",
                note=" ".join(x["t"] + "." for x in items))
    dia("모델 적재 시간", "loading",
        head=f"그림 {next(FIG)}: 모델을 카드에 올리는 데 걸린 시간, serve 로그 기준",
        items=[
            {"t": "카드 넉 장을 쓰는 모델은 쓰던 것을 내리고 새로 올려야 해서, 모델을 바꾸면 수십 초에서 수 분을 기다린다"},
            {"t": "카드 한 장짜리는 십 초 안팎이라 갈아 끼우는 부담이 거의 없고, 넷까지 동시에 떠 있을 수 있어 기다림 자체가 잘 안 생긴다"},
            {"t": "한 번 올라간 모델은 계속 떠 있다. 자주 쓰는 모델이 정해져 있다면 이 시간은 하루에 몇 번만 치르는 비용이다"},
        ])
    # ── 숫자는 표로도 한 번 정리한다. 그림은 크기를 비교하게 하고, 표는 값을 읽게 한다.

    def cell(v, digits=0, unit=""):
        return "-" if v is None else f"{v:.{digits}f}{unit}"

    trows = []
    for mid, lab, cards in gen_all.MODELS:
        rec = BY.get(mid)          # ★ 'd' 를 쓰면 덱 객체를 가려 build() 전체가 깨진다
        if not rec or "runs" not in rec:
            why = {"Qwen3-30B-A3B-FP8": "배포 FXB 결함",
                   "K-EXAONE-236B-A23B-NVFP4A16": "적재 중 메모리 부족" if LEN else "측정 실패"}.get(mid, "측정 실패")
            trows.append([lab, str(cards), "못 뜸", "-", "-", "-", "-", why] + (["-"] if LEN else []))
            continue
        c = rec.get("concurrent") or {}
        state = "정상" if rec.get("ok_runs") == rec.get("n_runs") else f"{rec.get('garbage', 0)}개 깨짐"
        row = [lab + (" *" if mid in gen_all.THINK else ""), str(cards),
               cell(rec.get("serve_load_s"), 0, "s"), cell(rec.get("ttft_med"), 2),
               cell(rec.get("decode_tps_med"), 0), cell(c.get("agg_tps"), 0),
               cell(gen_all.first_total(mid), 1, "s"), state]
        if LEN:
            row.insert(7, f"{max(r['out_tokens'] for r in rec['runs'] if not r.get('error')):,}")
        trows.append(row)
    heads = ["모델", "카드", "적재", "TTFT", "tok/s", "동시4", "첫응답", "답변"]
    if LEN:
        heads.insert(7, "최장 출력")
    d.table("성능 요약", heads, trows,
            subtitle=f"같은 프롬프트 4종, temperature 0, max_tokens {MT:,}. 별표는 사고하는 모델",
            note=("적재 = serve 로그의 Loading LLM 부터 기동 완료까지. TTFT = 첫 토큰까지(초, 중앙값). "
                  "tok/s = 첫 토큰 이후 생성 속도(중앙값). 동시4 = 같은 질문 4개 동시 요청의 합계 속도. "
                  "첫응답 = 가장 짧은 프롬프트가 끝날 때까지. " + ("최장 출력 = 단일 요청 네 개 중 가장 긴 출력 토큰. " if LEN else "") +
                  "Qwen3-30B-A3B-FP8 은 배포 FXB 번들이 가중치를 30.2 GiB 다 읽은 뒤 "
                  "embed_tokens.weight 가 F32 인데 EDF 는 bf16 을 기대해 죽는다(50회 재현). "
                  "다운로드 문제가 아니라 배포본 결함이다."
                  + (" K-EXAONE 236B 는 적재 중 서버 메모리가 바닥나 OOM 으로 serve 가 죽었다. 가중치 136.6 GiB 가 "
                     "서버 메모리 125 GiB 보다 크고, 적재 중 serve 는 가중치의 약 2배를 호스트에 든다. "
                     "08-29 판(1024 한도)에서는 떴다." if LEN else "")))

    rows = []
    for mid, lab, _ in gen_all.MODELS:
        rs = (BY.get(mid) or {}).get("runs")
        if not rs:
            continue
        r = rs[0]
        t = (r.get("text") or "").strip().replace("\n", " ")
        for ch in ("·", "ㆍ", "‧"):
            t = t.replace(ch, ", ")
        bad = r.get("broken")
        if bad:
            t = "같은 문자만 반복" if "반복" in str(bad) else str(bad)
        elif not t:
            t = "사고만 하고 답까지 못 감"
        rows.append([lab + (" *" if mid in gen_all.THINK else ""), t[:46], str(r.get("out_tokens") or "-")])
    d.table("답변 요약", ["모델", "답변 첫 줄", "토큰"], rows,
            subtitle="질문: 대한민국의 수도는 어디이고, 그 도시가 수도가 된 역사적 배경을 세 문장으로. 별표는 사고하는 모델",
            note=f"답변 원문 전체는 {'results_len' if LEN else 'results'}/<모델>.json 에 저장돼 있다. "
                 "프롬프트 4종 중 첫 번째(사실 질문)의 답이다. 토큰 수에는 사고 토큰이 포함된다.")


# config.json 의 quantization_config 와 라우터 REGISTRY 에서 뽑은 제원. 이름으로 추측하지 않았다.
SPECS = [
    # 용량 두 가지: 카드에 올라간 가중치(serve 로그의 Total size of parameters loaded)와 디스크 캐시.
    # 파라미터 수는 모델 이름에 이미 있으므로 크기 칸에는 넣지 않는다.
    ["gpt-oss 120B",        "60.8 GiB", "62 G",  "MoE 128",  "MXFP4",       "131,072", "4", "대화"],
    ["Solar-Open 100B",     "61.8 GiB", "64 G",  "dense",    "NVFP4 W4A4",  "131,072", "4", "대화"],
    ["K-EXAONE 236B",       "136.6 GiB","142 G", "MoE 128",  "NVFP4 W4A16", "262,144", "4", "대화"],
    ["Llama 3.3 70B",       "131.5 GiB","134 G", "dense",    "bf16",        "131,072", "4", "대화"],
    ["Qwen3 32B",           "32.0 GiB", "36 G",  "dense",    "FP8 블록",     "40,960",  "4", "대화"],
    ["EXAONE 4.0 32B",      "30.8 GiB", "34 G",  "dense",    "FP8 블록",     "131,072", "4", "대화"],
    ["Qwen3-VL 32B",        "62.1 GiB", "66 G",  "dense",    "bf16",        "262,144", "4", "대화, 이미지"],
    ["Qwen3-Coder 30B",     "29.0 GiB", "31 G",  "MoE 128",  "FP8 블록",     "262,144", "4", "코드"],
    ["A3B Instruct 2507",   "29.0 GiB", "33 G",  "MoE 128",  "FP8 블록",     "262,144", "4", "대화"],
    ["A3B Thinking 2507",   "29.0 GiB", "33 G",  "MoE 128",  "FP8 블록",     "262,144", "4", "대화, 사고"],
    ["A3B 30B",             "30.2 GiB", "32 G",  "MoE 128",  "FP8 블록",     "40,960",  "4", "대화, 못 뜸"],
    ["Llama 3.1 8B",        "15.0 GiB", "20 G",  "dense",    "bf16",        "131,072", "1", "대화"],
    ["Qwen3 8B",            "8.8 GiB",  "9.7 G", "dense",    "FP8 블록",     "40,960",  "1", "대화"],
    ["Qwen3 4B",            "4.8 GiB",  "5.8 G", "dense",    "FP8 블록",     "40,960",  "1", "대화"],
    ["Qwen2.5 0.5B",        "950 MiB",  "974 M", "dense",    "bf16",        "4,096",   "1", "대화"],
    ["Qwen3-Embedding 8B",  "-",        "15 G",  "dense",    "bf16",        "40,960",  "1", "임베딩"],
    ["Qwen3-Reranker 8B",   "-",        "-",     "dense",    "bf16",        "40,960",  "1", "리랭크"],
]

PROMPTS = [
    ("사실 질문", "대한민국의 수도는 어디이고, 그 도시가 수도가 된 역사적 배경을 세 문장으로 설명해줘."),
    ("코드 작성", "파이썬으로 피보나치 수열의 n번째 항을 반복문으로 구하는 함수를 쓰고, 시간복잡도를 한 줄로 덧붙여줘."),
    ("계산 추론", "한 상자에 사과가 12개 들어간다. 사과 100개를 담으려면 상자가 몇 개 필요하고 마지막 상자에는 몇 개가 남는지 계산 과정을 보여줘."),
    ("개념 설명", "트랜스포머의 어텐션이 무엇인지 처음 배우는 사람에게 설명해줘. 비유를 하나 들고, 왜 순환신경망보다 병렬화에 유리한지도 말해줘."),
]
KEY = {"사실 질문": "fact", "코드 작성": "code", "계산 추론": "reason", "개념 설명": "long"}


def prompts_slide():
    """쓴 프롬프트를 요약하지 않고 원문 그대로 싣는다."""
    lines = []
    for i, (name, text) in enumerate(PROMPTS, 1):
        lines.append(f"# {i}. {name}")
        # 한 줄이 너무 길면 코드 상자에서 글자가 작아진다 — 40자쯤에서 접는다
        buf = text
        while buf:
            cut = 44 if len(buf) > 44 else len(buf)
            if cut < len(buf):
                sp = buf.rfind(" ", 0, cut + 1)
                cut = sp if sp > 20 else cut
            lines.append("  " + buf[:cut].strip())
            buf = buf[cut:].strip()
        lines.append("")
    d.code("사용한 프롬프트 전문", lines[:-1],
           caption="네 종류를 모든 모델에 똑같이 던졌다. 이 문장 그대로다",
           note=f"temperature 0, max_tokens {MT:,}, 스트리밍. 답변 원문은 이어지는 절에 그대로 싣는다.")


# 글머리 유형의 본문 크기(L-47). 답변 원문은 이 크기에 들어가게 나누고, 표지와 부 표지 부제도 이 크기로 맞춘다.
ANS_PT = 14
# 부제 없는 글머리 슬라이드의 본문 높이(pt): _chrome 이 돌려주는 top(0.42+0.72+0.30in)부터 BODY_BOT 까지
AVAIL_PT = (deck.BODY_BOT - deck.Inches(1.44)) / 12700.0 - 8


def _fits(items):
    return deck.Deck._needed_pt(items, deck.BODY_W, ANS_PT) <= AVAIL_PT


def _split(t, hdr):
    """한 모델의 답이 한 장에 안 들어가면 잘라서 이어 싣는다. 자르는 자리는 공백으로."""
    out = []
    while t:
        n = len(t)
        while n > 60 and not _fits([{"t": hdr}, {"t": t[:n], "lv": 1}]):
            n = int(n * 0.9)
        if n < len(t):
            sp = t.rfind(" ", int(n * 0.8), n)
            n = sp if sp > 0 else n
        out.append(t[:n].strip())
        t = t[n:].strip()
    return out


def answer_slides():
    """모델별 답변을 발췌 없이 싣는다. 슬라이드마다 ANS_PT 에 들어가는 만큼만 담는다(L-47)."""
    import gen_all
    for name, _ in PROMPTS:
        key = KEY[name]
        pages, cur = [], []
        for mid, lab, _ in gen_all.MODELS:
            rec = BY.get(mid)
            if not rec or "runs" not in rec:
                continue
            r = next((x for x in rec["runs"] if x.get("prompt") == key), None)
            if not r:
                continue
            t = (r.get("text") or "").strip().replace("\n", " ")
            for ch in ("·", "ㆍ", "‧"):
                t = t.replace(ch, ", ")
            if not t:
                t = "(답변 없음, 사고만 하고 예산이 끝났다)" if r.get("thinking") else "(빈 출력)"
            th = r.get("think_tok") or r.get("think_tokens")
            hdr = f"{lab}  ({r.get('out_tokens')}tok" + (f", 사고 {th}" if th else "") + ")"
            for k, piece in enumerate(_split(t, hdr)):
                pair = [{"t": hdr + ("  이어서" if k else "")}, {"t": piece, "lv": 1}]
                if cur and not _fits(cur + pair):
                    pages.append(cur)
                    cur = []
                cur += pair
        if cur:
            pages.append(cur)
        for i, items in enumerate(pages, 1):
            # 제목에 대시를 쓰지 않는다(L-26). 콜론으로 잇는다.
            d.bullets(f"답변 원문: {name}" + (f" ({i})" if len(pages) > 1 else ""), items, subtitle=None)


def unify_sizes(prs):
    """L-47: 같은 유형의 슬라이드는 같은 본문 크기. deck.py 는 장마다 들어가는 최대 크기를 골라서 흩어진다.
    유형 구분과 크기 측정은 검사기(check_layout)의 것을 그대로 쓴다 — 기준이 다르면 고쳐도 걸린다.
      글머리  ANS_PT. 답변 원문은 1단과 2단의 차이(1.2pt)를 지키며 옮기고, 표지와 부 표지는 부제만 줄인다
      표, 코드  그 유형에서 가장 작게 잡힌 크기(큰 쪽에 맞추면 내용이 많은 장이 넘친다)"""
    import check_layout as cl
    from pptx.util import Pt
    kinds = [(s, cl._slide_kind(s)) for s in prs.slides]
    tmin = min((cl._body_pt(s) for s, k in kinds if k == "표"), default=None)
    cmin = min((cl._body_pt(s) for s, k in kinds if k == "코드"), default=None)

    def body_runs(s):
        for sh in s.shapes:
            if sh.has_text_frame and (sh.top or 0) > cl.EMU_IN:
                for p in sh.text_frame.paragraphs:
                    for r in p.runs:
                        if r.font.size:
                            yield r
    for s, k in kinds:
        if k == "표" and tmin:
            for sh in s.shapes:
                if sh.has_table:
                    for row in sh.table.rows:
                        for c in row.cells:
                            for p in c.text_frame.paragraphs:
                                for r in p.runs:
                                    if r.font.size:
                                        r.font.size = Pt(tmin)
        elif k == "코드" and cmin:
            for r in body_runs(s):
                if r.font.name == "Consolas":
                    r.font.size = Pt(cmin)
        elif k == "글머리":
            body = cl._body_pt(s)
            if not body or body == ANS_PT:
                continue
            cover = any(r.font.size.pt >= 30 for sh in s.shapes if sh.has_text_frame
                        for p in sh.text_frame.paragraphs for r in p.runs if r.font.size)
            for r in body_runs(s):
                v = r.font.size.pt
                if not 9.6 <= v <= 24:
                    continue
                if cover:
                    if v > ANS_PT:
                        r.font.size = Pt(ANS_PT)
                else:
                    r.font.size = Pt(v - (body - ANS_PT))
    from collections import defaultdict
    by = defaultdict(set)
    for s, k in kinds:
        pt = cl._body_pt(s)
        if pt is not None:
            by[k].add(pt)
    print("유형별 본문 크기(L-47):", {k: sorted(v) for k, v in by.items()})


if __name__ == "__main__":
    build()
    prompts_slide()
    d.section("2", "답변 원문", "발췌하지 않고 모델이 낸 그대로")
    answer_slides()
    unify_sizes(d.prs)
    OUT = os.path.join(SRC, "..", os.environ.get("DECK_OUT", "모델별-성능정리.pptx"))
    d.save(OUT)
    print("넘침:", deck.OVERFLOW if deck.OVERFLOW else "없음")
    from pptx import Presentation
    prs = Presentation(OUT); worst = 0.0
    for sl in prs.slides:
        for shp in sl.shapes:
            if shp.shape_type == 6 and shp.name in srcs:
                diff, _, _ = compare(srcs[shp.name], shp); worst = max(worst, diff)
                print(f"되돌려 비교 {shp.name}: {diff*100:.2f}% ({'통과' if diff <= 0.02 else '실패'})")
    print(f"최대 {worst*100:.2f}%, 슬라이드 {len(prs.slides)}장, 저장 {os.path.abspath(OUT)}")
