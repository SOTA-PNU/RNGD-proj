#!/usr/bin/env python3
"""RNGD NPU 채팅 UI — furiosa-llm serve 위의 ChatGPT 스타일 대화 인터페이스(다크).

- on-demand serve: 모델을 고르면 필요한 카드를 비우고 띄움. tp8 은 복제(dp)·레이어분할(pp)을
  골라(dp×pp ≤ 4장) 띄우고, tp32 는 4장 고정(dp·pp 비활성). 카드 회계는 실제 serve 의
  --devices/-pp/-dp 로 항상 정확히.
- 상태 LED: 🟢 떠 있음 / 🟡 전환중(이 dot만 깜빡) / 🔴 꺼짐·실패.
  도는 네모(progress 스피너)는 모든 이벤트에서 끔(show_progress="hidden") → LED 펄스만.
  LED 자동 갱신은 Timer 가 담당하되 '바뀔 때만' 갱신해 깜빡임이 없음.
- 질문은 입력 즉시 대화창에 뜨고, 답변은 토큰 단위로 흘러나옴(스트리밍).
- max_tokens 는 생성 시 (컨텍스트 - 프롬프트)로 자동 클램프 → 컨텍스트 초과 에러 안 남.
- 대화 사이드바(새 채팅·검색·최근·선택 삭제) + 서버 디스크 영구 저장.
- 전송 버튼(↑)은 생성 중 중지(■)로 바뀜. 메시지의 ↻ 아이콘으로 다시 생성.
"""
import os
import re
import json
import signal
import subprocess
import threading
import time
import warnings
import datetime as dt
from pathlib import Path

# 부팅 경고 정리: 5.50 에는 아직 대체 API(launch theme/css, Chatbot buttons,
# api_visibility)가 없어 마이그레이션이 불가하므로, "Gradio 6.0" 예고용
# DeprecationWarning 만 정밀하게 끈다(다른 경고는 그대로 보이게 둠).
warnings.filterwarnings("ignore", message=r".*Gradio 6\.0.*", category=DeprecationWarning)

import gradio as gr  # noqa: E402
import httpx  # noqa: E402
from openai import OpenAI  # noqa: E402

ARTIFACTS = Path.home() / "RNGD-proj/Model_Benchmark/rngd-npu/artifacts"
FURIOSA_LLM = str(Path.home() / "furiosa/bin/furiosa-llm")
LOG_DIR = Path(__file__).resolve().parent / "serve_logs"
CONV_DIR = Path(__file__).resolve().parent / "conversations"
LOG_DIR.mkdir(exist_ok=True)
CONV_DIR.mkdir(exist_ok=True)

# 키 -> 모델. kind: tp8(카드 1~4 dp) / tp32(4장 고정). ctx: max_model_len.
# Qwen2.5-Coder-1.5B 는 furiosa-llm 2026.2.0 이 출력이 깨지게 컴파일해(greedy 에서도
# 토큰 수프, untie·재빌드로도 안 고쳐짐 — info/README_build.md 8.2) 카탈로그에서 뺐다.
CATALOG = {
    "coder7":         dict(name="Qwen2.5-Coder-7B", port=8002, kind="tp8",
                           sub="qwen2.5-coder-7b-inst-tp8", extra=[], ctx=32768),
    "coder14":        dict(name="Qwen2.5-Coder-14B", port=8003, kind="tp8",
                           sub="qwen2.5-coder-14b-tp8", extra=[], ctx=32768),
    "qwen3-32b":      dict(name="Qwen3-32B-FP8", port=8004, kind="tp8",
                           sub="qwen3-32b-fp8-tp8", extra=["--reasoning-parser", "qwen3"], ctx=40960),
    "qwen3-32b-16k":  dict(name="Qwen3-32B-FP8-16k", port=8005, kind="tp8",
                           sub="qwen3-32b-fp8-tp8-16k", extra=["--reasoning-parser", "qwen3"], ctx=16384),
    "exaone-32b":     dict(name="EXAONE-4.0-32B-FP8", port=8011, kind="tp32",
                           sub="exaone-4.0-32b-fp8-tp32/snapshots/8c42cdea3e7339fe3e3aefc5c7cff1f66b320f31",
                           extra=["--reasoning-parser", "exaone4"], ctx=131072),
    "llama-70b":      dict(name="Llama-3.3-70B", port=8012, kind="tp32",
                           sub="llama-3.3-70b-inst-tp32/snapshots/2cbb7a6286be88e25072e56d3a64943e56408440",
                           extra=["--tool-call-parser", "llama3_json"], ctx=131072),
    "qwen3-32b-tp32": dict(name="Qwen3-32B-FP8-tp32", port=8013, kind="tp32",
                           sub="qwen3-32b-fp8-tp32/snapshots/1f5cf9426425998140e2dde6357ba0ee4f6820b2",
                           extra=["--reasoning-parser", "qwen3"], ctx=40960),
    # Qwen3-Coder-30B-A3B-FP8 은 2026.2.0 런타임이 FP8 MoE serve 미지원이라 제외.
}
DISPLAY2KEY = {m["name"]: k for k, m in CATALOG.items()}
DISPLAY_NAMES = [m["name"] for m in CATALOG.values()]
# 기본 선택 모델 = 가장 가벼운 정상 모델 coder7 (coder1.5 는 출력 깨짐으로 제외, 위 참고).
DEFAULT_MODEL = CATALOG["coder7"]["name"]
STARTUP_TIMEOUT = float(os.environ.get("CHAT_SERVE_TIMEOUT", "900"))


def _dd_choices():
    """드롭다운: '모델명 · tp8' / '· tp32'. 값은 모델명."""
    return [(f"{m['name']}  ·  {m['kind']}", m["name"]) for m in CATALOG.values()]


def _port_up(port):
    try:
        return httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=1.5).status_code == 200
    except Exception:
        return False


def _par_flags(kind, dp, pp):
    """serve 명령에 넣을 병렬화 플래그(-pp/-dp). 실측으로 검증된 형태만 쓴다(2026-06-09):
    - tp32 는 아티팩트가 tp32·4장 고정이라 플래그 없음.
    - pp=1 이면 플래그 없음 — dp 는 --devices 카드 수로 자동 추론(furiosa 기본·현행 동작, 검증됨).
    - pp>1 이면 -pp 를 명시(카드 수만으론 pp/dp 구분 불가). dp>1 이면 -dp 도 함께 못박는다.
    예) (8,1,1)→[]  (8,2,1)→[]  (8,1,2)→[-pp 2]  (8,1,4)→[-pp 4]  (8,2,2)→[-pp 2 -dp 2]"""
    if kind == "tp32" or pp <= 1:
        return []
    flags = ["-pp", str(pp)]
    if dp > 1:
        flags += ["-dp", str(dp)]
    return flags


class ServeManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._proc = {}
        self._state = {k: "down" for k in CATALOG}
        self._err = {}
        self._dev = {}
        self._par = {}     # key -> (dp, pp). tp8 의 복제 수·레이어분할 수(serve 명령에 -dp/-pp 로 반영)
        self._lru = []

    def _touch(self, key):
        if key in self._lru:
            self._lru.remove(key)
        self._lru.append(key)

    def _discover(self):
        """실행 중인 furiosa-llm serve 의 --port/--devices/-pp/-dp 를 읽어 카드 점유·병렬구성을
        실제와 맞춘다. pgrep 로 살아있는 것으로 확인된 키 집합을 돌려준다(HTTP 와 무관한 liveness)."""
        try:
            out = subprocess.run(["pgrep", "-af", "furiosa-llm serve"],
                                 capture_output=True, text=True, timeout=5).stdout
        except Exception:
            return set()
        port2info = {}
        for line in out.splitlines():
            mp, md = re.search(r"--port\s+(\d+)", line), re.search(r"--devices\s+(\S+)", line)
            if mp and md:
                mpp = re.search(r"(?:-pp|--pipeline-parallel-size)\s+(\d+)", line)
                mdp = re.search(r"(?:-dp|--data-parallel-size)\s+(\d+)", line)
                port2info[int(mp.group(1))] = (md.group(1),
                                               int(mpp.group(1)) if mpp else None,
                                               int(mdp.group(1)) if mdp else None)
        port2key = {m["port"]: k for k, m in CATALOG.items()}
        found = set()
        with self._lock:
            for port, (dev, pp, dp) in port2info.items():
                k = port2key.get(port)
                if k:
                    found.add(k)
                    if self._state.get(k) not in ("loading", "stopping"):
                        self._state[k] = "up"
                        self._dev[k] = dev
                        ncards = len([d for d in dev.split(",") if d.startswith("npu:")])
                        ppv = pp or 1
                        # -dp 미지정이면 카드 수/pp 로 역산(현행 dp 자동추론과 동일 규칙)
                        self._par[k] = (dp if dp else max(1, ncards // ppv), ppv)
        return found

    def _held_cards(self):
        held = set()
        for k, st in self._state.items():
            if st in ("up", "loading"):
                for d in self._dev.get(k, "").split(","):
                    if d.startswith("npu:"):
                        held.add(int(d.split(":")[1]))
        return held

    def _free_cards(self, n):
        held = self._held_cards()
        return [c for c in range(4) if c not in held][:n]

    def state(self, key):
        return self._state.get(key, "down")

    def error(self, key):
        return self._err.get(key, "")

    def device(self, key):
        return self._dev.get(key, "")

    def par(self, key):
        return self._par.get(key, (1, 1))

    def request(self, key, dp=1, pp=1):
        if not key or key not in CATALOG:
            return
        self._discover()
        kind = CATALOG[key]["kind"]
        if kind == "tp32":
            dp, pp, needed = 1, 1, 4          # tp32 는 4장 전부 — dp·pp 무의미
        else:
            dp = max(1, min(4, int(dp or 1)))
            pp = max(1, min(4, int(pp or 1)))
            if dp * pp > 4:                    # 카드 4장 한도 — pp 우선, dp 축소
                dp = max(1, 4 // pp)
            needed = dp * pp                    # tp8 = 카드당 8PE → 카드 수 = dp×pp
        with self._lock:
            cur = self._state.get(key)
            cur_cards = (len(self._dev.get(key, "").split(","))
                         if cur in ("up", "loading") and self._dev.get(key) else 0)
            # 이미 전환 중이면 새 전환을 시작하지 않음(중복 요청·터널 재진입 방지)
            if cur in ("loading", "stopping"):
                return
            # 같은 모델이 같은 (dp,pp) 로 이미 떠 있으면 재기동 없이 재사용
            if cur == "up" and cur_cards == needed and self._par.get(key) == (dp, pp):
                self._touch(key)
                return
            others = [o for o in CATALOG if o != key and self._state.get(o) in ("up", "loading")]
            reclaimable = (4 - len(self._held_cards())) + cur_cards
            victims = []
            for o in self._lru + others:
                if reclaimable >= needed:
                    break
                if o in others and o not in victims:
                    victims.append(o)
                    reclaimable += len(self._dev.get(o, "").split(",")) if self._dev.get(o) else 1
            if reclaimable < needed:
                self._state[key] = "error"
                self._err[key] = f"{needed}장 확보 불가"
                return
            for v in victims:
                self._state[v] = "stopping"
            self._par[key] = (dp, pp)
            self._state[key] = "loading"
            self._err.pop(key, None)
            self._touch(key)
        threading.Thread(target=self._transition, args=(key, victims, needed), daemon=True).start()

    def _transition(self, key, victims, needed):
        for v in victims:
            self._stop_blocking(v)
        with self._lock:
            old = self._proc.get(key)
        if old is not None:
            self._kill(old)
            with self._lock:
                self._proc.pop(key, None)
                self._dev.pop(key, None)
        elif key in self._dev:
            self._stop_blocking(key)
            with self._lock:
                self._state[key] = "loading"
        with self._lock:
            cards = self._free_cards(needed)
            if len(cards) < needed:
                self._state[key] = "error"
                self._err[key] = f"{needed}장 확보 실패"
                return
            dev = ",".join(f"npu:{c}" for c in cards)
            self._dev[key] = dev
        self._start_and_wait(key, dev)

    def _start_and_wait(self, key, dev):
        m = CATALOG[key]
        art = str(ARTIFACTS / m["sub"])
        port = m["port"]
        if not Path(art, "artifact.json").exists():
            with self._lock:
                self._state[key] = "error"
                self._err[key] = f"artifact 없음: {art}"
                self._dev.pop(key, None)
            return
        dp, pp = self._par.get(key, (1, 1))
        cmd = [FURIOSA_LLM, "serve", art, "--devices", dev, "--host", "0.0.0.0",
               "--port", str(port), "--enable-prefix-caching",
               *_par_flags(m["kind"], dp, pp), *m["extra"]]
        try:
            logf = open(LOG_DIR / f"{port}.log", "w")
            proc = subprocess.Popen(cmd, stdout=logf, stderr=logf, start_new_session=True)
        except Exception as e:
            with self._lock:
                self._state[key] = "error"
                self._err[key] = f"serve 실행 실패: {e}"
                self._dev.pop(key, None)
            return
        with self._lock:
            self._proc[key] = proc
        base = f"http://127.0.0.1:{port}/v1"
        deadline = time.time() + STARTUP_TIMEOUT
        while time.time() < deadline:
            if proc.poll() is not None:
                with self._lock:
                    self._state[key] = "error"
                    self._err[key] = f"serve 조기 종료(code {proc.returncode}) — serve_logs/{port}.log"
                    self._dev.pop(key, None)
                return
            try:
                if httpx.get(base + "/models", timeout=3.0).status_code == 200:
                    with self._lock:
                        self._state[key] = "up"
                    return
            except Exception:
                pass
            time.sleep(3.0)
        self._stop_blocking(key)
        with self._lock:
            self._state[key] = "error"
            self._err[key] = f"{int(STARTUP_TIMEOUT)}초 안에 준비 안 됨 — serve_logs/{port}.log"

    def _kill(self, proc):
        try:
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                try:
                    proc.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=10)
        except Exception:
            pass

    def _stop_blocking(self, key):
        with self._lock:
            proc = self._proc.get(key)
            port = CATALOG[key]["port"]
            if self._state.get(key) != "down":
                self._state[key] = "stopping"
        if proc is not None:
            self._kill(proc)
        elif _port_up(port):
            subprocess.run(["pkill", "-f", f"furiosa-llm serve.*--port {port}"], check=False)
            time.sleep(2.0)
        with self._lock:
            self._proc.pop(key, None)
            self._dev.pop(key, None)
            self._state[key] = "down"
            self._err.pop(key, None)

    def states(self):
        found = self._discover()    # pgrep 로 실제 살아있는 키
        with self._lock:
            snap = dict(self._state)
            procs = dict(self._proc)
        out = {}
        for k, m in CATALOG.items():
            s = snap.get(k, "down")
            if s in ("loading", "stopping"):
                out[k] = s
                continue
            if k in found:
                # 프로세스가 살아있으면 HTTP 프로브가 느려도 up 유지(busy/slow 오판 방지)
                out[k] = "up"
                continue
            if s == "up":
                # pgrep 에도 없고 상태가 up 이었으면, HTTP 로 한 번 더 확인 후에만 내림
                if _port_up(m["port"]):
                    out[k] = "up"
                    continue
                p = procs.get(k)
                new = "error" if (p is not None and p.poll() is not None) else "down"
                with self._lock:
                    self._state[k] = new
                    self._dev.pop(k, None)
                    if new == "error":
                        self._err[k] = f"serve 중단됨 — serve_logs/{m['port']}.log"
                out[k] = new
            else:
                out[k] = s
        return out


MGR = ServeManager()

# ── ChatGPT 풍 다크 테마 ───────────────────────────────────────────
BG = "#212121"       # 메인 배경
SIDE = "#171717"     # 사이드바 배경
ELEV = "#2f2f2f"     # 입력창·검색창·선택 항목
TXT = "#ececec"
MUTE = "#8e8e9e"
CSS = f"""
@keyframes ledpulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.18; }} }}
.led-pulse {{ animation: ledpulse 1.1s ease-in-out infinite; }}
.gradio-container {{ max-width:100% !important; padding:0 !important; background:{BG} !important; color:{TXT} !important; }}
/* 최외곽 래퍼 main.fillable.app 의 max-width:1536·margin(auto→32px)·padding:32px 가 양옆을 비움 → 0/100% 로 덮어 빈 공간 제거 */
.gradio-container .app, main.fillable, main.app {{ max-width:100% !important; margin:0 !important; padding:0 !important; }}
footer, .show-api, .built-with {{ display:none !important; }}
* {{ --color-accent:{ELEV} !important; --color-accent-soft:{ELEV} !important; }}
.gradio-container .prose, .gradio-container label, .gradio-container span {{ color:{TXT}; }}
input[type=range] {{ accent-color:#9a9aa6 !important; }}
input:focus, textarea:focus, .gr-box:focus-within {{ outline:none !important; box-shadow:none !important; }}

/* 사이드바 */
#sidebar {{ background:{SIDE} !important; border-right:1px solid #2a2a2a; padding:10px 8px !important; min-height:98vh; }}
#sidebar .gap, #sidebar .form {{ background:transparent !important; border:none !important; }}
#newchat-btn {{ background:transparent !important; border:1px solid #3a3a3a !important; color:{TXT} !important; border-radius:12px !important; font-weight:500; text-align:left; }}
#newchat-btn:hover {{ background:#2a2a2a !important; }}
#search-box {{ background:transparent !important; }}
#search-box input, #search-box textarea {{ background:{ELEV} !important; border:none !important; color:{TXT} !important; border-radius:12px !important; padding:9px 12px !important; }}
#sidebar .label-wrap, #recent-label p {{ color:{MUTE} !important; font-size:12px !important; font-weight:600; padding:6px 6px 2px !important; margin:0 !important; }}
#convo-list {{ border:none !important; background:transparent !important; box-shadow:none !important; }}
#convo-list label {{ display:block !important; width:100%; padding:8px 10px !important; margin:1px 0 !important; border-radius:8px !important; cursor:pointer; color:#cfcfd6 !important; font-size:14px; border:none !important; background:transparent !important; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
#convo-list label:hover {{ background:#2a2a2a !important; }}
#convo-list input[type=radio] {{ display:none !important; }}
#convo-list label:has(input:checked) {{ background:{ELEV} !important; color:#fff !important; }}
#del-btn {{ background:transparent !important; border:none !important; color:{MUTE} !important; font-size:12.5px !important; text-align:left; }}
#del-btn:hover {{ color:#ef6a6a !important; }}
#statusbox, #statusbox div {{ font-size:12.5px; line-height:1.7; color:{TXT} !important; }}
#sidebar .accordion, #sidebar .accordion * {{ border-color:#2a2a2a !important; }}
#userchip {{ border-top:1px solid #2a2a2a; margin-top:8px; padding:10px 6px 4px; color:{TXT}; font-size:13.5px; }}

/* 메인 */
#main {{ background:{BG} !important; }}
#topbar {{ padding:10px 18px 2px 18px !important; }}
#model-dd {{ max-width:340px !important; }}
#model-dd, #model-dd .wrap, #model-dd .secondary-wrap {{ background:transparent !important; border:none !important; box-shadow:none !important; min-height:0 !important; }}
#model-dd input {{ font-weight:700 !important; font-size:16px !important; color:{TXT} !important; cursor:pointer; }}
#model-dd .wrap:hover {{ background:#2a2a2a !important; border-radius:10px !important; }}
#settings-acc {{ border:none !important; background:transparent !important; }}

/* 챗봇 — GPT 풍 말풍선 (Gradio 5.50 실제 클래스: .user-row/.bot-row/.bubble/.message) */
#chatbot {{ background:transparent !important; border:none !important; max-width:100% !important; margin:0 !important; padding:0 24px !important; }}
#inputwrap {{ max-width:100% !important; margin:0 !important; }}
#chatbot .message-wrap, #chatbot .message-row {{ box-shadow:none !important; }}
#chatbot .message-bubble-border {{ border-color:transparent !important; }}
#chatbot .bot-row .message, #chatbot .bot-row .bubble, #chatbot .bot {{ background:transparent !important; border:none !important; color:{TXT} !important; }}
#chatbot .user-row .message, #chatbot .user-row .bubble, #chatbot .user {{ background:{ELEV} !important; border:none !important; color:#fff !important; border-radius:20px !important; }}
#chatbot .avatar-container, #chatbot .avatar-image {{ display:none !important; }}
#chatbot .message, #chatbot .message-content, #chatbot .bubble, #chatbot .message-row {{ opacity:1 !important; }}
#chatbot .user-row .message-content, #chatbot .user-row .message {{ color:#fff !important; }}
#chatbot .bot-row .message-content, #chatbot .bot-row .message {{ color:{TXT} !important; }}
/* 답변(봇)은 채팅 폭을 꽉 채우게 — Gradio 기본 width 제한 해제 (질문 말풍선은 우측 컴팩트 유지) */
#chatbot .bot-row, #chatbot .bot-row .message, #chatbot .bot-row .message-content, #chatbot .bot-row .prose {{ max-width:100% !important; width:100% !important; }}
/* 사고 과정 — ChatGPT식: 접힘 = 연회색 한 줄, 펼침 = 은은한 박스 안 회색 추론 */
#chatbot details {{ background:transparent !important; border:none !important; padding:0 !important; margin:2px 0 10px !important; }}
#chatbot details summary {{ color:#8e8e9e !important; font-size:13.5px !important; cursor:pointer; list-style:none; outline:none; user-select:none; }}
#chatbot details summary::-webkit-details-marker {{ display:none; }}
#chatbot details summary::marker {{ content:""; }}
#chatbot details summary::before {{ content:"▸"; margin-right:6px; color:#8e8e9e; font-size:11px; }}
#chatbot details[open] summary::before {{ content:"▾"; }}
#chatbot details[open] {{ background:#1a1a1a !important; border-radius:10px !important; padding:10px 14px !important; }}
#chatbot details[open] > *:not(summary) {{ color:#9a9aa6 !important; font-size:13px !important; line-height:1.6; }}

/* 입력 알약 */
#inputwrap {{ padding:8px 24px 18px !important; }}
#inputbar {{ background:{ELEV} !important; border:1px solid #3a3a3a !important; border-radius:28px !important; padding:6px 6px 6px 18px !important; align-items:center !important; }}
/* 겉 알약 안의 텍스트박스 내부 박스(block/wrap)를 완전히 투명화 → 박스 안 박스 제거 */
#inputbar .block, #inputbar .wrap, #inputbar label, #inputbar .input-container {{ background:transparent !important; border:none !important; box-shadow:none !important; border-radius:0 !important; padding:0 !important; }}
#inputbar textarea {{ background:transparent !important; border:none !important; color:{TXT} !important; box-shadow:none !important; font-size:15px; padding:8px 0 !important; }}
#send-btn, #stop-btn {{ border-radius:50% !important; min-width:38px !important; max-width:38px; width:38px; height:38px; padding:0 !important; font-size:18px; line-height:1; box-shadow:none !important; }}
#send-btn {{ background:{TXT} !important; color:#111 !important; border:none !important; }}
#send-btn:hover {{ background:#fff !important; }}
#stop-btn {{ background:{TXT} !important; color:#111 !important; border:none !important; }}
#hint {{ color:{MUTE}; font-size:11.5px; text-align:center; padding:6px 0 0; }}
"""
_COLOR = {"up": "#22c55e", "loading": "#eab308", "stopping": "#eab308", "down": "#6b7280", "error": "#ef4444"}

# 라이트/다크 어느 모드든 검정으로 보이게 테마 변수를 직접 다크로 고정
# (Base 테마 기본값이 흰색이라 검색창·입력창·아코디언·버튼이 희게 뜨는 것을 막음).
_DARKVARS = dict(
    body_background_fill=BG, body_text_color=TXT, body_text_color_subdued=MUTE,
    background_fill_primary=BG, background_fill_secondary=SIDE,
    block_background_fill=BG, block_border_color="#2a2a2a",
    block_label_background_fill=BG, block_label_text_color=MUTE, block_title_background_fill=BG,
    border_color_primary="#2a2a2a", border_color_accent="#2a2a2a",
    input_background_fill=ELEV, input_background_fill_focus=ELEV, input_background_fill_hover=ELEV,
    input_border_color="#3a3a3a", input_placeholder_color=MUTE,
    button_secondary_background_fill=ELEV, button_secondary_background_fill_hover="#3a3a3a",
    button_secondary_text_color=TXT, panel_background_fill=SIDE, panel_border_color="#2a2a2a",
    color_accent_soft=ELEV, code_background_fill="#1b1b1b",
)
THEME = gr.themes.Base(primary_hue="gray", secondary_hue="gray", neutral_hue="gray").set(
    **_DARKVARS, **{f"{k}_dark": v for k, v in _DARKVARS.items()})


def status_html():
    """LED + 모델명 + (떠 있을 때만) 사용 카드·병렬구성(dp·pp). 실제 serve 프로세스의
    --devices/-pp/-dp 에서 읽으므로(_discover) UI 선택이 진짜 적용됐는지 여기서 확인된다."""
    st = MGR.states()
    rows = []
    for k, m in CATALOG.items():
        s = st[k]
        cls = ' class="led-pulse"' if s in ("loading", "stopping") else ''
        dot = (f'<span{cls} style="display:inline-block;width:9px;height:9px;border-radius:50%;'
               f'background:{_COLOR[s]};margin-right:8px;vertical-align:middle;"></span>')
        dev = MGR.device(k)
        if m["kind"] != "tp32" and s in ("up", "loading"):
            dp_v, pp_v = MGR.par(k)
            dev = f"{dev} · dp{dp_v}·pp{pp_v}" if dev else dev
        info = f' <span style="color:{MUTE};">{dev}</span>' if s in ("up", "loading") and dev else ""
        if s == "error" and MGR.error(k):
            info = ' <span style="color:#ef4444;">실패</span>'
        rows.append(f'<div style="padding:1px 0;">{dot}{m["name"]}{info}</div>')
    return (f'<div style="color:{MUTE};margin-bottom:5px;">🟢 켜짐 · 🟡 전환중 · 🔴 꺼짐</div>'
            + "".join(rows))


# LED 자동 갱신은 '전환 중'에만 Timer 를 켜서 돌린다. 유휴 상태에선 Timer 를 꺼
# (active=False) 패널을 아예 다시 그리지 않으므로 '전체 깜빡임'이 없다.
# (gr.skip() 은 Timer.tick 에서 프런트 재렌더를 막지 못해, 켜져 있으면 매 틱 패널이
#  통째로 교체되며 깜빡이기 때문에 — 유휴 시엔 끄는 것이 정답.)
_LAST_STATUS = {"html": None}


def _transitioning():
    return any(MGR.state(k) in ("loading", "stopping") for k in CATALOG)


def status_tick():
    """Timer tick: 상태 HTML 갱신 + 전환이 끝났으면 Timer 를 스스로 끈다.
    출력 2개: (status, timer)."""
    h = status_html()
    changed = h != _LAST_STATUS["html"]
    _LAST_STATUS["html"] = h
    status_out = h if changed else gr.skip()
    timer_out = gr.skip() if _transitioning() else gr.Timer(active=False)
    return status_out, timer_out


def status_force():
    h = status_html()
    _LAST_STATUS["html"] = h
    return h


def load_status():
    """페이지 로드: 상태 1회 갱신 + 전환 중이면 Timer 켬. 출력 2개: (status, timer)."""
    return status_force(), gr.Timer(active=_transitioning())


# ── 대화 영구 저장 ────────────────────────────────────────────────
def _new_conv_id():
    return "c" + dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _title_of(messages):
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content", "").strip():
            t = msg["content"].strip().replace("\n", " ")
            return (t[:30] + "…") if len(t) > 30 else t
    return "(빈 대화)"


def _save_convo(cid, model_name, messages):
    if not cid or not any(msg.get("role") == "user" for msg in messages):
        return
    (CONV_DIR / f"{cid}.json").write_text(json.dumps(
        {"id": cid, "title": _title_of(messages), "model": model_name,
         "updated": dt.datetime.now().isoformat(timespec="seconds"), "messages": messages},
        ensure_ascii=False, indent=2))


def _load_convo(cid):
    try:
        return json.loads((CONV_DIR / f"{cid}.json").read_text()).get("messages", [])
    except Exception:
        return []


def _convo_choices(query=""):
    items = []
    for f in sorted(CONV_DIR.glob("c*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            d = json.loads(f.read_text())
            title = d.get("title") or d["id"]
            if not query or query.lower() in title.lower():
                items.append((title, d["id"]))
        except Exception:
            pass
    return items


# ── 생성 ──────────────────────────────────────────────────────────
def _client(base_url):
    return OpenAI(base_url=base_url, api_key="dummy", timeout=600)


def _resolve_model_id(base_url):
    return _client(base_url).models.list().data[0].id


def _think_block(think, label):
    """추론 텍스트를 ChatGPT식 접힘 헤더로. 라벨만 보이고, 누르면 전체 추론 펼침.
    Gradio 5.50 은 allow_tags=True + '블록' 포맷(태그가 줄 단독)일 때만 <details> 를 렌더한다."""
    return f"<details>\n<summary>{label}</summary>\n\n{think}\n\n</details>\n\n"


def _stream_reply(base_url, model_id, msgs, temperature, max_tokens):
    """답변 스트리밍. 추론(reasoning)은 ChatGPT처럼 연한 회색 접힘 줄로:
    추론 중엔 '💭 생각하는 중…', 끝나면 '💭 N초 동안 생각함'(클릭하면 전체 추론)."""
    stream = _client(base_url).chat.completions.create(
        model=model_id, messages=msgs, temperature=temperature, max_tokens=int(max_tokens), stream=True)
    think, body = "", ""
    t0 = time.time()
    think_secs = None
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        r = getattr(delta, "reasoning", None)
        if r:
            think += r
        if delta.content:
            if think and think_secs is None:           # 추론 끝, 답변 시작 → 걸린 시간 확정
                think_secs = max(1, round(time.time() - t0))
            body += delta.content
        if think:
            label = (f"💭 {think_secs}초 동안 생각함" if think_secs is not None else "💭 생각하는 중…")
            yield _think_block(think, label) + body
        else:
            yield body or "…"


# 전송/중지 버튼 토글 상태
_GEN = (gr.update(visible=False), gr.update(visible=True))     # 생성중: 전송 숨김, 중지 표시
_IDLE = (gr.update(visible=True), gr.update(visible=False))    # 대기: 전송 표시, 중지 숨김


def _generate(history, conv_id, model_name, dp, pp, system_prompt, temperature, max_tokens):
    """history(마지막 user) 뒤에 답변 스트리밍. 출력 7개:
    (chatbot, conv_id, txt, convo, status, send_btn, stop_btn)."""
    if not conv_id:
        conv_id = _new_conv_id()
    history = list(history)
    history.append({"role": "assistant", "content": ""})
    yield history, conv_id, "", gr.update(), gr.skip(), *_GEN     # 생성 시작 → 중지 버튼

    key = DISPLAY2KEY.get(model_name)
    if key is None:
        history[-1]["content"] = f"⚠️ 알 수 없는 모델: {model_name}"
        _save_convo(conv_id, model_name, history)
        yield history, conv_id, "", gr.update(choices=_convo_choices(), value=conv_id), status_force(), *_IDLE
        return
    m = CATALOG[key]
    port = m["port"]
    base_url = f"http://127.0.0.1:{port}/v1"

    if MGR.state(key) != "up" and not _port_up(port):
        MGR.request(key, dp, pp)
        t0 = time.time()
        while True:
            if MGR.state(key) == "up" or _port_up(port):
                break
            if MGR.state(key) == "error":
                history[-1]["content"] = f"⚠️ '{model_name}' 띄우기 실패: {MGR.error(key)}"
                _save_convo(conv_id, model_name, history)
                yield history, conv_id, "", gr.update(choices=_convo_choices(), value=conv_id), status_force(), *_IDLE
                return
            if MGR.state(key) == "down":   # 이전 모델 정리로 down 까지 떨어졌으면 다시 요청(idempotent)
                MGR.request(key, dp, pp)
            history[-1]["content"] = f"⏳ '{model_name}' 준비 중… ({int(time.time() - t0)}초). 무거운 모델은 수 분 걸립니다."
            yield history, conv_id, "", gr.update(), gr.skip(), gr.skip(), gr.skip()
            time.sleep(2.0)
            if time.time() - t0 > STARTUP_TIMEOUT + 30:
                history[-1]["content"] = f"⚠️ '{model_name}' 준비 시간 초과 — serve_logs/{port}.log"
                _save_convo(conv_id, model_name, history)
                yield history, conv_id, "", gr.update(choices=_convo_choices(), value=conv_id), status_force(), *_IDLE
                return

    try:
        model_id = _resolve_model_id(base_url)
    except Exception as e:
        history[-1]["content"] = f"⚠️ 서버 연결 실패: {e}"
        _save_convo(conv_id, model_name, history)
        yield history, conv_id, "", gr.update(choices=_convo_choices(), value=conv_id), status_force(), *_IDLE
        return

    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt and system_prompt.strip() else [])
    msgs += history[:-1]
    # 컨텍스트 초과 방지: prompt + max_tokens <= ctx. 프롬프트 토큰을 보수적으로 추정해 클램프.
    est_prompt = sum(len(mm.get("content", "")) for mm in msgs) // 3 + 16
    eff_max = max(16, min(int(max_tokens), m["ctx"] - est_prompt - 256))
    try:
        for partial in _stream_reply(base_url, model_id, msgs, temperature, eff_max):
            history[-1]["content"] = partial
            yield history, conv_id, "", gr.update(), gr.skip(), gr.skip(), gr.skip()
    except Exception as e:
        history[-1]["content"] = f"⚠️ 생성 중 에러: {e}"
        yield history, conv_id, "", gr.update(), gr.skip(), gr.skip(), gr.skip()
    _save_convo(conv_id, model_name, history)
    yield history, conv_id, "", gr.update(choices=_convo_choices(), value=conv_id), gr.skip(), *_IDLE


def respond(user_msg, history, conv_id, model_name, dp, pp, system_prompt, temperature, max_tokens):
    user_msg = (user_msg or "").strip()
    history = list(history or [])
    if not user_msg:
        yield history, conv_id, "", gr.update(), gr.skip(), gr.skip(), gr.skip()
        return
    if not conv_id:
        conv_id = _new_conv_id()
    history.append({"role": "user", "content": user_msg})
    # 질문을 입력 즉시 대화창에 노출 + 전송→중지 토글
    yield history, conv_id, "", gr.update(), gr.skip(), *_GEN
    yield from _generate(history, conv_id, model_name, dp, pp, system_prompt, temperature, max_tokens)


def regenerate(history, conv_id, model_name, dp, pp, system_prompt, temperature, max_tokens):
    history = list(history or [])
    if history and history[-1].get("role") == "assistant":
        history.pop()
    if not history or history[-1].get("role") != "user":
        yield history, conv_id, "", gr.update(), gr.skip(), gr.skip(), gr.skip()
        return
    yield from _generate(history, conv_id, model_name, dp, pp, system_prompt, temperature, max_tokens)


def delete_convo(selected_cid, cur_conv_id):
    """대화 목록에서 선택한 대화를 삭제. 열려 있던 대화면 화면도 비움."""
    if selected_cid:
        try:
            (CONV_DIR / f"{selected_cid}.json").unlink()
        except Exception:
            pass
    if selected_cid and selected_cid == cur_conv_id:
        return [], "", gr.update(choices=_convo_choices(), value=None)
    return gr.update(), cur_conv_id, gr.update(choices=_convo_choices(), value=None)


def new_chat():
    return [], _new_conv_id(), gr.update(value=None), ""


def load_chat(cid):
    if not cid:
        return gr.update(), gr.update()
    return _load_convo(cid), cid


def filter_convos(query):
    return gr.update(choices=_convo_choices(query or ""))


def _par_updates(model_name, dp, pp):
    """모델 종류에 맞춰 dp·pp 드롭다운을 재구성하고, 적용할 (dp,pp) 를 함께 돌려준다.
    - tp32: dp·pp 비활성(4장 고정, 둘 다 1·선택 불가)
    - tp8 : pp 에 맞춰 dp 선택지를 제한(dp×pp ≤ 4 장). 반환: (dp_update, pp_update, dp, pp)"""
    key = DISPLAY2KEY.get(model_name)
    is_tp32 = CATALOG.get(key, {}).get("kind") == "tp32"
    if is_tp32:
        return (gr.update(value=1, choices=[1], interactive=False),
                gr.update(value=1, choices=[1], interactive=False), 1, 1)
    pp = max(1, min(4, int(pp or 1)))
    max_dp = max(1, 4 // pp)                      # 카드 4장 한도: dp ≤ 4/pp
    dp = max(1, min(int(dp or 1), max_dp))
    return (gr.update(value=dp, choices=list(range(1, max_dp + 1)), interactive=True),
            gr.update(value=pp, choices=[1, 2, 3, 4], interactive=True), dp, pp)


def on_model_change(model_name, dp, pp):
    """모델 변경 → on-demand serve 즉시 시작 후 바로 반환(터널 연결 장시간 점유 X).
    dp·pp 컨트롤 재구성 + max_tokens 를 그 모델 최대치로 재설정.
    출력 5개: (status, maxtok, dp, pp, timer)."""
    key = DISPLAY2KEY.get(model_name)
    ctx = CATALOG.get(key, {}).get("ctx", 8192)
    dp_u, pp_u, dp_v, pp_v = _par_updates(model_name, dp, pp)
    MGR.request(key, dp_v, pp_v)
    timer_upd = gr.Timer(active=True) if _transitioning() else gr.skip()
    return status_force(), gr.update(maximum=ctx, value=ctx), dp_u, pp_u, timer_upd


def on_par_change(model_name, dp, pp):
    """dp/pp 변경 → 새 병렬 구성으로 재-serve. dp×pp>4 면 제약에 맞춰 자동 보정.
    max_tokens 는 건드리지 않음(모델 그대로). 출력 4개: (status, dp, pp, timer)."""
    key = DISPLAY2KEY.get(model_name)
    dp_u, pp_u, dp_v, pp_v = _par_updates(model_name, dp, pp)
    MGR.request(key, dp_v, pp_v)
    timer_upd = gr.Timer(active=True) if _transitioning() else gr.skip()
    return status_force(), dp_u, pp_u, timer_upd


def build_ui():
    _ctx0 = CATALOG[DISPLAY2KEY[DEFAULT_MODEL]]["ctx"]
    with gr.Blocks(title="RNGD NPU Chat", fill_height=True, css=CSS, theme=THEME) as demo:
        conv_id = gr.State("")
        with gr.Row(equal_height=False):
            # ── 사이드바 ──
            with gr.Column(scale=1, min_width=250, elem_id="sidebar"):
                new_btn = gr.Button("✏️  새 채팅", elem_id="newchat-btn")
                search = gr.Textbox(placeholder="🔍  검색", show_label=False, elem_id="search-box",
                                    lines=1, container=False)
                gr.Markdown("최근", elem_id="recent-label")
                convo_radio = gr.Radio(choices=_convo_choices(), show_label=False, value=None,
                                       elem_id="convo-list", container=False)
                del_btn = gr.Button("🗑  선택한 대화 삭제", size="sm", elem_id="del-btn")
                with gr.Accordion("모델 상태", open=True):
                    status = gr.HTML(value=status_force(), elem_id="statusbox")
                    refresh_btn = gr.Button("🔄 새로고침", size="sm")
                gr.HTML('<div id="userchip">🟢&nbsp; RNGD NPU · furiosa-llm</div>')
            # ── 메인 ──
            with gr.Column(scale=5, elem_id="main"):
                with gr.Row(elem_id="topbar"):
                    model_dd = gr.Dropdown(_dd_choices(), value=DEFAULT_MODEL, show_label=False,
                                           container=False, elem_id="model-dd", scale=1)
                with gr.Accordion("⚙  설정", open=False, elem_id="settings-acc"):
                    with gr.Row():
                        dp = gr.Dropdown([1, 2, 3, 4], value=1, label="복제 dp (처리량↑)", scale=1,
                                         info="모델을 카드마다 복제 — 동시 요청↑ (한 대화는 1장). tp8만")
                        pp = gr.Dropdown([1, 2, 3, 4], value=1, label="레이어 분할 pp (카드 분산)", scale=1,
                                         info="한 모델을 여러 장에 나눠 실행. dp×pp ≤ 4장. tp8만")
                        temp = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="temperature", scale=1)
                    maxtok = gr.Slider(64, _ctx0, value=_ctx0, step=256, label="max_tokens (답변 최대 길이)",
                                       info="모델 선택 시 그 모델 최대치로 자동 설정. 생성 시 컨텍스트에 맞게 자동 조정됨")
                    sys_box = gr.Textbox(label="시스템 프롬프트", lines=1,
                                         placeholder="예: 너는 한국어로 답하는 코딩 도우미야.")
                chatbot = gr.Chatbot(type="messages", elem_id="chatbot", height="68vh",
                                     show_label=False, show_copy_button=True,
                                     allow_tags=True)  # True 라야 <details> 사고과정이 렌더됨
                with gr.Column(elem_id="inputwrap"):
                    with gr.Row(elem_id="inputbar"):
                        txt = gr.Textbox(placeholder="무엇이든 부탁하세요", show_label=False, scale=9,
                                         lines=1, container=False, autofocus=True)
                        send = gr.Button("↑", elem_id="send-btn", scale=0)
                        stop = gr.Button("■", elem_id="stop-btn", scale=0, visible=False)
                    gr.HTML('<div id="hint">RNGD NPU 위 furiosa-llm · 답변은 모델에 따라 부정확할 수 있어요.</div>')

        SP = "hidden"  # 도는 네모(progress 스피너) 끔 → LED 펄스만, 스트리밍/질문 즉시 노출
        # 유휴 시엔 꺼두는 LED 자동갱신 Timer (전환 중에만 켜짐 → 유휴 패널 깜빡임 없음).
        timer = gr.Timer(2.5, active=False)
        chat_inputs = [txt, chatbot, conv_id, model_dd, dp, pp, sys_box, temp, maxtok]
        chat_outputs = [chatbot, conv_id, txt, convo_radio, status, send, stop]
        regen_inputs = [chatbot, conv_id, model_dd, dp, pp, sys_box, temp, maxtok]
        ev1 = txt.submit(respond, chat_inputs, chat_outputs, show_progress=SP)
        ev2 = send.click(respond, chat_inputs, chat_outputs, show_progress=SP)
        ev3 = chatbot.retry(regenerate, regen_inputs, chat_outputs, show_progress=SP)
        stop.click(lambda: _IDLE, None, [send, stop], cancels=[ev1, ev2, ev3], show_progress=SP)
        new_btn.click(new_chat, None, [chatbot, conv_id, convo_radio, txt], show_progress=SP)
        del_btn.click(delete_convo, [convo_radio, conv_id], [chatbot, conv_id, convo_radio], show_progress=SP)
        refresh_btn.click(status_force, None, [status], show_progress=SP)
        search.change(filter_convos, [search], [convo_radio], show_progress=SP)
        convo_radio.change(load_chat, [convo_radio], [chatbot, conv_id], show_progress=SP)
        model_dd.change(on_model_change, [model_dd, dp, pp], [status, maxtok, dp, pp, timer], show_progress=SP)
        dp.change(on_par_change, [model_dd, dp, pp], [status, dp, pp, timer], show_progress=SP)
        pp.change(on_par_change, [model_dd, dp, pp], [status, dp, pp, timer], show_progress=SP)
        # 전환 중에만 켜져 LED 를 갱신하고, 끝나면 status_tick 이 Timer 를 스스로 끈다.
        timer.tick(status_tick, None, [status, timer], show_progress=SP)
        # 터널 재접속/새 탭: 최신 상태 1회 갱신 + 전환 중이면 Timer 켜기.
        demo.load(load_status, None, [status, timer], show_progress=SP)
    return demo


if __name__ == "__main__":
    host = os.environ.get("CHAT_HOST", "0.0.0.0")
    port = int(os.environ.get("CHAT_PORT", "7860"))
    root_path = os.environ.get("CHAT_ROOT_PATH", "")
    share = os.environ.get("CHAT_SHARE", "0") == "1"
    _auth = os.environ.get("CHAT_AUTH", "")
    auth = tuple(_auth.split(":", 1)) if ":" in _auth else None
    build_ui().queue().launch(server_name=host, server_port=port, show_error=True,
                              root_path=root_path, share=share, auth=auth)
