#!/usr/bin/env python3
"""실시간 성능 대시보드 — furiosa-apps chat-playground 의 메트릭을 우리 Gradio 채팅에 이식.

furiosa 원본(chat-playground/backend/main.py)은 furiosa_smi_py + 전용 백엔드로 5개 지표
(TPS·TTFT·TPOT·E2E·Power/card)를 WebSocket 으로 흘려보낸다. 우리는 그 구조를 Gradio 에 맞게 옮긴다:

- **토큰 타이밍(TPS·TTFT·TPOT·E2E)**: 우리 스트리밍 생성(chat_app._generate)에서 직접 측정.
  토큰 수는 furiosa-llm serve 가 stream_options(include_usage) 로 주는 usage(정확) + 진행 중엔
  글자수/4 추정(라이브 스파크라인용).
- **전력·온도·사용률(Power/Temp/Util)**: `furiosa-smi` CLI 파싱 — chat venv 에 furiosa_smi_py 가
  없어도 동작한다(의존성 0). 전력은 카드별로 실제 변함(유휴 ~38W → 추론 ~128W)이라 좋은 실시간 신호.

디자인은 furiosa 원본 그대로: 검정 카드(#151515)·시안 제목(#76d6ff)·보라 라인(#cdbbff)·흰 monospace 숫자.
"""
import re
import subprocess
import threading
import time
from collections import deque

SMI = "furiosa-smi"
HIST = 40  # 스파크라인 표본 개수

# furiosa 디자인 토큰 (chat-playground/frontend/src 에서 추출)
CYAN = "#76d6ff"    # 메트릭 제목
PURPLE = "#cdbbff"  # 라인 차트
CARD_BG = "#151515"
CARD_BORDER = "#444"
NUM = "#ffffff"
MUTE = "#888"
RED = "#dc2626"     # 메모리 거의 가득(>=90%) 경고 막대


def _run(cmd, timeout=6):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


class Metrics:
    """스레드 안전 메트릭 저장소. 생성 코드가 요청 타이밍을 기록하고, 타이머가 HW 표본을 채운다."""

    def __init__(self):
        self._lock = threading.RLock()
        # 하드웨어 (타이머가 furiosa-smi 로 주기적으로 갱신)
        self.power = 0.0
        self.temp = 0.0
        self.util = 0.0
        # 0 으로 미리 채우지 않는다 — 채우면 유휴여도 deque 가 굴러 스파크라인이 매 틱 바뀌어
        # 재렌더가 멈추지 않는다(리뷰 지적). 표본이 쌓이는 만큼만 그린다(_sparkline len<2 가드).
        self.power_hist = deque(maxlen=HIST)
        self.util_hist = deque(maxlen=HIST)
        self.tps_hist = deque(maxlen=HIST)
        self.tpot_hist = deque(maxlen=HIST)
        # 마지막(또는 진행 중) 요청 결과
        self.ttft = 0.0   # ms
        self.e2e = 0.0    # ms
        self.tpot = 0.0   # ms/token
        self.tps = 0      # 마지막 요청 평균 tok/s (정확 — usage 기반)
        self.max_tps = 0  # 정확 평균 TPS 의 피크(라이브 추정치는 넣지 않음)
        self.tokens_last = 0
        # 라이브 TPS 윈도(진행 중 글자수를 모아 타이머가 토큰/초로 환산 — 추정치)
        self._window_chars = 0
        self._last_sample_t = time.time()
        self.last_activity = 0.0       # 마지막 토큰/생성시작 시각 — 대시보드 타이머 on/off 게이트
        self._smi_cache = (0.0, None)  # (ts, per-card dict)
        self.mem = {}                  # idx -> (used_gib, total_gib, pct) — 카드별 HBM 점유

    # ── 요청 타이밍: 생성 코드(_generate/_stream_reply)가 호출 ──────────────
    def start(self):
        """요청 시작. 반환된 rec 를 같은 요청의 first_token/add_chars/finish 에 넘긴다."""
        self.touch()
        return {"t0": time.time(), "first": None, "chars": 0}

    def touch(self):
        """활동 표시(생성 시작·토큰 도착). 대시보드 타이머가 이 시각 기준으로 자동 on/off."""
        with self._lock:
            self.last_activity = time.time()

    def first_token(self, rec):
        if rec.get("first") is None:
            rec["first"] = time.time()

    def add_chars(self, rec, nchars):
        """스트리밍 델타의 글자수를 누적(라이브 TPS 추정용). 토큰≈글자/4 로 환산하되
        델타마다 1로 올림하지 않아(리뷰 지적: 서브토큰 과대) 나머지를 이월한다."""
        if nchars <= 0:
            return
        rec["chars"] += nchars
        with self._lock:
            self._window_chars += nchars
            self.last_activity = time.time()

    def finish(self, rec, completion_tokens=None):
        """요청 종료 — 정확한 토큰수(usage)가 있으면 그걸로 TPOT/평균TPS 확정."""
        now = time.time()
        toks = int(completion_tokens) if completion_tokens else max(0, rec.get("chars", 0) // 4)
        ttft = (rec["first"] - rec["t0"]) * 1000 if rec.get("first") else 0.0
        e2e = (now - rec["t0"]) * 1000
        gen_ms = max(0.0, e2e - ttft)
        tpot = gen_ms / toks if toks > 0 else 0.0
        tps = round(toks / (gen_ms / 1000)) if gen_ms > 0 and toks > 0 else 0
        with self._lock:
            self.ttft, self.e2e, self.tpot, self.tps = ttft, e2e, tpot, tps
            self.tokens_last = toks
            if tpot > 0:
                self.tpot_hist.append(round(tpot, 1))
            if tps > 0:
                self.max_tps = max(self.max_tps, tps)   # 정확 평균만 피크에 반영

    # ── 하드웨어 표본: 대시보드 타이머가 호출 ──────────────────────────────
    def _read_smi(self, active_cards=None):
        """furiosa-smi info(전력·온도) + status(코어 사용률) 파싱. active_cards 가 주어지면
        그 카드들 중 최대값(작업 중인 카드)을, 없으면 전체 최대값을 본다."""
        now = time.time()
        ts, cached = self._smi_cache
        if cached is None or now - ts > 1.6:   # 타이머 틱(1.8s)당 최대 1회 spawn 되도록
            info = _run([SMI, "info"])
            status = _run([SMI, "status"])
            per = {}  # idx -> {"power":, "temp":, "util":}
            for line in info.splitlines():
                m = re.search(r"npu(\d+)\b", line)
                if not m:
                    continue
                idx = int(m.group(1))
                mt = re.search(r"([\d.]+)\s*°?C", line)
                mp = re.search(r"([\d.]+)\s*W", line)
                d = per.setdefault(idx, {"power": 0.0, "temp": 0.0, "util": 0.0})
                if mt:
                    d["temp"] = float(mt.group(1))
                if mp:
                    d["power"] = float(mp.group(1))
            # status: 디바이스 블록(+---+ 구분)마다 npuN + 최대 Core % + Memory(used/total GiB, %)
            for block in re.split(r"\+[-+]+\+", status):
                mi = re.search(r"npu(\d+)\b", block)
                if not mi:
                    continue
                idx = int(mi.group(1))
                d = per.setdefault(idx, {"power": 0.0, "temp": 0.0, "util": 0.0})
                cores = [float(x) for x in re.findall(r"Core\s*\d+:\s*([\d.]+)\s*%", block)]
                if cores:
                    d["util"] = max(cores)
                # 예: "43.56/47.50 GiB"  +  "(91.71%)"
                mm = re.search(r"([\d.]+)\s*/\s*([\d.]+)\s*GiB", block)
                if mm:
                    d["mem_used"] = float(mm.group(1))
                    d["mem_total"] = float(mm.group(2))
                mpct = re.search(r"\(([\d.]+)\s*%\)", block)
                if mpct:
                    d["mem_pct"] = float(mpct.group(1))
            cached = per
            self._smi_cache = (now, per)
        per = cached or {}
        idxs = [i for i in per if (active_cards is None or i in active_cards)] or list(per)
        if not idxs:
            return 0.0, 0.0, 0.0
        power = max(per[i]["power"] for i in idxs)
        temp = max(per[i]["temp"] for i in idxs)
        util = max(per[i]["util"] for i in idxs)
        return round(power, 1), round(temp, 1), round(util, 1)

    def sample(self, active_cards=None):
        """타이머가 호출: HW 읽고 라이브 TPS(윈도 글자수/4 / 경과초) 갱신, 히스토리 push.
        라이브 TPS 는 추정치라 max_tps(점선)에는 넣지 않는다 — 점선은 finish 의 정확 평균 피크만."""
        power, temp, util = self._read_smi(active_cards)
        per = self._smi_cache[1] or {}
        mem = {i: (round(d.get("mem_used", 0.0), 1), round(d.get("mem_total", 0.0), 1),
                   round(d.get("mem_pct", 0.0), 1)) for i, d in per.items()}
        now = time.time()
        with self._lock:
            elapsed = max(0.5, now - self._last_sample_t)
            self._last_sample_t = now
            live_tps = round((self._window_chars / 4) / elapsed) if self._window_chars else 0
            self._window_chars = 0
            self.power, self.temp, self.util = power, temp, util
            self.mem = mem
            self.power_hist.append(power)
            self.util_hist.append(util)
            self.tps_hist.append(live_tps)

    # ── JSON(클라이언트 폴링용) ──────────────────────────────────────────
    def metrics_json(self) -> dict:
        """대시보드 클라이언트가 폴링해 '값만 제자리 갱신'하도록 현재 수치를 dict 로.
        (HTML 통째 교체 대신 이 JSON 으로 텍스트/막대/스파크라인만 바꿔 깜빡임을 없앤다.)"""
        with self._lock:
            return {
                "tps": self.tps, "tpot": round(self.tpot, 1), "e2e": self.e2e, "ttft": self.ttft,
                "power": self.power, "temp": self.temp, "util": self.util, "max_tps": self.max_tps,
                "tps_hist": list(self.tps_hist), "tpot_hist": list(self.tpot_hist),
                "power_hist": list(self.power_hist),
                "mem": [[i, u, t, p] for i, (u, t, p) in sorted(self.mem.items())],
            }

    # ── 렌더(정적 구조 1회) ──────────────────────────────────────────────
    def render_dashboard(self) -> str:
        """값이 들어갈 자리(id)만 있는 정적 구조를 1회 렌더. 실제 수치는 클라이언트 JS 가
        /dash_metrics 를 폴링해 각 id 의 텍스트/막대/스파크라인만 바꿔 채운다(제자리 갱신=무깜빡임)."""
        cards = [
            _dcard("tps", "TPS", "tok/s", spark=True, dash=True),
            _dcard("tpot", "TPOT", "ms", spark=True),
            _dcard("e2e", "E2E", "s", spark=False),
            _dcard("ttft", "TTFT", "ms", spark=False),
            _dcard("power", "Power / card", "W", spark=True),
            _row2(_dcard("temp", "Temp", "°C", spark=False, mini=True),
                  _dcard("util", "NPU util", "%", spark=False, mini=True)),
            _mem_card_struct(),
        ]
        return (
            f'<div style="display:flex;flex-direction:column;gap:9px;">'
            f'<div style="color:{MUTE};font-size:12px;letter-spacing:.5px;'
            f'text-transform:uppercase;margin-bottom:1px;">Real-time performance</div>'
            + "".join(cards) + "</div>"
        )

    # ── 렌더(레거시: HTML 통째) — 호환용으로 남겨둠 ───────────────────────
    def render_html(self):
        with self._lock:
            ttft, e2e, tpot, tps = self.ttft, self.e2e, self.tpot, self.tps
            power, temp, util = self.power, self.temp, self.util
            tps_hist = list(self.tps_hist)
            tpot_hist = list(self.tpot_hist)
            power_hist = list(self.power_hist)
            max_tps = self.max_tps
            mem = dict(self.mem)
        cards = [
            _card("TPS", "tok/s", _fmt(tps), tps_hist, dash=max_tps or None),
            _card("TPOT", "ms", _fmt(tpot), tpot_hist),
            _card("E2E", "s", _fmt(e2e / 1000.0), None),
            _card("TTFT", "ms", _fmt(ttft), None),
            _card("Power / card", "W", _fmt(power), power_hist),
            _row2(_card("Temp", "°C", _fmt(temp), None, mini=True),
                  _card("NPU util", "%", _fmt(util), None, mini=True)),
        ]
        return (
            f'<div style="display:flex;flex-direction:column;gap:9px;">'
            f'<div style="color:{MUTE};font-size:12px;letter-spacing:.5px;'
            f'text-transform:uppercase;margin-bottom:1px;">Real-time performance</div>'
            + "".join(cards)
            + _mem_card(mem)            # 위 성능 카드 아래에 카드별 메모리 점유 추가
            + "</div>"
        )


def _fmt(v):
    """furiosa formatNumber 와 동일한 유효숫자 규칙(>=100 정수, >=10 소수1, >=1 소수2)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "0"
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 100:
        return f"{round(v)}"
    if a >= 10:
        return f"{round(v, 1):g}"
    if a >= 1:
        return f"{round(v, 2):g}"
    return f"{float('%.3g' % v):g}"


def _sparkline(values, color=PURPLE, w=210, h=46, dash=None):
    """deque/list → furiosa 풍 미니 라인차트(인라인 SVG, 의존성 0). 점선(dash)으로 max 표시 가능."""
    vals = [float(x) for x in values] if values else [0.0]
    if len(vals) < 2:
        vals = [0.0, 0.0]
    vmax = max(vals + ([float(dash)] if dash else []))
    vmin = min(vals)
    rng = (vmax - vmin) or 1.0
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        x = i / (n - 1) * w
        y = h - 3 - (v - vmin) / rng * (h - 6)
        pts.append(f"{x:.1f},{y:.1f}")
    poly = (f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
            f'stroke-width="1.5" />')
    dashln = ""
    if dash:
        dy = h - 3 - (float(dash) - vmin) / rng * (h - 6)
        dashln = (f'<line x1="0" y1="{dy:.1f}" x2="{w}" y2="{dy:.1f}" stroke="{color}" '
                  f'stroke-width="1" stroke-dasharray="4 4" opacity="0.55" />')
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
            f'style="width:100%;height:{h}px;display:block;">{poly}{dashln}</svg>')


def _card(title, unit, value, hist, dash=None, mini=False):
    spark = _sparkline(hist, dash=dash) if hist is not None else ""
    pad = "9px 12px" if mini else "11px 13px"
    head = (
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
        f'<span style="color:{CYAN};font-size:13px;font-weight:500;">{title}'
        f'<span style="color:{CYAN};"> ({unit})</span></span>'
        f'<span style="color:{NUM};font-family:monospace;font-size:26px;line-height:1;">{value}</span>'
        f'</div>'
    )
    return (f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};border-radius:8px;'
            f'padding:{pad};box-sizing:border-box;">{head}{spark}</div>')


def _row2(a, b):
    return (f'<div style="display:flex;gap:9px;">'
            f'<div style="flex:1;min-width:0;">{a}</div>'
            f'<div style="flex:1;min-width:0;">{b}</div></div>')


def _mem_bar(idx, used, total, pct):
    """카드 한 장의 메모리 한 줄: NPUn [막대] used/total  pct%. 위 카드들과 같은 팔레트.
    90% 이상이면 막대를 빨강(거의 가득)으로, 아니면 시안으로."""
    pct = max(0.0, min(100.0, pct if pct else (used / total * 100 if total else 0)))
    fill = RED if pct >= 90 else CYAN
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin:6px 0 0;">'
        f'<span style="color:{MUTE};font-size:12px;width:42px;flex:none;">NPU{idx}</span>'
        f'<div style="flex:1;height:8px;background:#000;border:1px solid #333;border-radius:4px;'
        f'overflow:hidden;">'
        f'<div style="width:{pct:.1f}%;height:100%;background:{fill};"></div></div>'
        f'<span style="color:{NUM};font-family:monospace;font-size:12px;width:92px;text-align:right;'
        f'flex:none;">{used:.1f}/{total:.1f}</span>'
        f'<span style="color:{MUTE};font-family:monospace;font-size:12px;width:38px;text-align:right;'
        f'flex:none;">{pct:.0f}%</span>'
        f'</div>'
    )


def _dcard(key, title, unit, spark=True, dash=False, mini=False):
    """클라이언트가 채울 빈 카드(값 span 은 id=mv-<key>, 스파크라인 polyline 은 id=mp-<key>)."""
    val = (f'<span id="mv-{key}" style="color:{NUM};font-family:monospace;'
           f'font-size:26px;line-height:1;">—</span>')
    head = (
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
        f'<span style="color:{CYAN};font-size:13px;font-weight:500;">{title}'
        f'<span style="color:{CYAN};"> ({unit})</span></span>{val}</div>'
    )
    sp = ""
    if spark:
        dl = (f'<line id="md-{key}" x1="0" y1="0" x2="210" y2="0" stroke="{PURPLE}" '
              f'stroke-width="1" stroke-dasharray="4 4" opacity="0" />') if dash else ""
        sp = (f'<svg viewBox="0 0 210 46" preserveAspectRatio="none" '
              f'style="width:100%;height:46px;display:block;">'
              f'<polyline id="mp-{key}" points="" fill="none" stroke="{PURPLE}" stroke-width="1.5" />'
              f'{dl}</svg>')
    pad = "9px 12px" if mini else "11px 13px"
    return (f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};border-radius:8px;'
            f'padding:{pad};box-sizing:border-box;">{head}{sp}</div>')


def _mem_card_struct():
    """카드별 메모리 막대 빈 구조(NPU0~3 고정 행, 처음엔 숨김). 클라이언트가 width/색/텍스트만 갱신.
    width 에 CSS transition 을 줘 차오름이 매끄럽게 보입니다."""
    rows = ""
    for i in range(4):
        rows += (
            f'<div id="mem-row-{i}" style="display:none;align-items:center;gap:8px;margin:6px 0 0;">'
            f'<span style="color:{MUTE};font-size:12px;width:42px;flex:none;">NPU{i}</span>'
            f'<div style="flex:1;height:8px;background:#000;border:1px solid #333;border-radius:4px;'
            f'overflow:hidden;">'
            f'<div id="mem-bar-{i}" style="width:0%;height:100%;background:{CYAN};'
            f'transition:width .5s ease;"></div></div>'
            f'<span id="mem-used-{i}" style="color:{NUM};font-family:monospace;font-size:12px;'
            f'width:92px;text-align:right;flex:none;">—</span>'
            f'<span id="mem-pct-{i}" style="color:{MUTE};font-family:monospace;font-size:12px;'
            f'width:38px;text-align:right;flex:none;">—</span></div>')
    head = (
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
        f'<span style="color:{CYAN};font-size:13px;font-weight:500;">NPU memory'
        f'<span style="color:{CYAN};"> (GiB)</span></span></div>'
    )
    return (f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};border-radius:8px;'
            f'padding:11px 13px;box-sizing:border-box;">{head}{rows}</div>')


def _mem_card(mem):
    """카드별 HBM 점유를 한 카드 안에 NPU0~3 막대로. 성능 카드와 동일한 검정 카드 디자인."""
    rows = [_mem_bar(i, *mem[i]) for i in sorted(mem) if mem[i][1] > 0]
    body = ("".join(rows) if rows
            else f'<div style="color:{MUTE};font-size:12px;margin-top:4px;">측정 대기 중…</div>')
    head = (
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
        f'<span style="color:{CYAN};font-size:13px;font-weight:500;">NPU memory'
        f'<span style="color:{CYAN};"> (GiB)</span></span></div>'
    )
    return (f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};border-radius:8px;'
            f'padding:11px 13px;box-sizing:border-box;">{head}{body}</div>')
