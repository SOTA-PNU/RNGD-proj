#!/usr/bin/env python3
"""Furiosa RNGD Chat — furiosa-llm serve 위의 대화 인터페이스.

furiosa-apps(github.com/furiosa-ai/furiosa-apps) 의 디자인과 두 기능을 우리 채팅에 통합:
- **furiosa 인터페이스 디자인**: 순수 검정 테마 + 로고/Furiosa RNGD Chat/DEMO 헤더,
  빨강(전송)·시안(메트릭 제목)·보라(라인·배지) 강조 (chat-playground 원본 팔레트).
- **실시간 성능 대시보드(우측 컬럼)**: TPS·TTFT·TPOT·E2E·Power/card·Temp·Util.
  토큰 타이밍은 스트리밍 생성에서, 전력/온도/사용률은 furiosa-smi 파싱(npu_metrics.py).
- **RAG(선택)**: 사이드바에서 켜면 업로드 문서에서 근거를 찾아 컨텍스트로 주입+출처 각주.
  기본 TF-IDF(의존성·NPU 0), furiosa 임베딩/리랭커 서버 있으면 그걸 사용(rag_store.py).

기존 디테일은 그대로 유지(요구사항):
- on-demand serve: 모델을 고르면 필요한 카드를 비우고 띄움. tp8 은 복제(dp)·레이어분할(pp)을
  골라(dp×pp ≤ 4장) 띄우고, tp32 는 4장 고정(dp·pp 비활성). 카드 회계는 실제 serve 의
  --devices/-pp/-dp 로 항상 정확히.
- 상태 LED: 🟢 떠 있음 / 🟡 전환중(이 dot만 깜빡) / 🔴 꺼짐·실패.
- 질문은 입력 즉시 대화창에 뜨고, 답변은 토큰 단위로 흘러나옴(스트리밍).
- max_tokens 는 생성 시 (컨텍스트 - 프롬프트)로 자동 클램프 → 컨텍스트 초과 에러 안 남.
- 대화 사이드바(새 채팅·검색·최근·선택 삭제) + 서버 디스크 영구 저장.
- 전송 버튼(↑)은 생성 중 중지(■)로 바뀜. 메시지의 ↻ 아이콘으로 다시 생성.
"""
import os
import re
import json
import base64
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

import npu_metrics  # noqa: E402  실시간 성능 대시보드(TPS·TTFT·TPOT·E2E·Power) — furiosa chat-playground 이식
import rag_store    # noqa: E402  선택적 RAG(문서 검색 후 컨텍스트 주입) — furiosa rag(kotaemon) 패턴 이식

ARTIFACTS = Path.home() / "RNGD-proj/Model_Benchmark/rngd-npu/artifacts"
FURIOSA_LLM = str(Path.home() / "furiosa/bin/furiosa-llm")
HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "serve_logs"
CONV_DIR = HERE / "conversations"
LOG_DIR.mkdir(exist_ok=True)
CONV_DIR.mkdir(exist_ok=True)

# 실시간 대시보드·RAG 전역(서버 1개 = 공유 상태). MGR(ServeManager)와 같은 위상.
METRICS = npu_metrics.Metrics()
RAG = rag_store.RagStore()


def _logo_data_uri():
    """furiosa Symbol.png(칩 아이콘)을 base64 data URI 로 — 헤더에 인라인(파일 서빙 경로 무관)."""
    p = HERE / "assets" / "Symbol.png"
    try:
        return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
    except Exception:
        return ""


LOGO_URI = _logo_data_uri()

# 키 -> 모델. kind: tp8(카드 1~4 dp) / tp32(4장 고정). ctx: max_model_len.
# Qwen2.5-Coder-1.5B 는 furiosa-llm 2026.2.0 이 출력이 깨지게 컴파일해(greedy 에서도
# 토큰 수프, untie·재빌드로도 안 고쳐짐 — info/README_build.md 8.2) 카탈로그에서 뺐다.
CATALOG = {
    "coder7":         dict(name="Qwen2.5-Coder-7B-Inst", port=8002, kind="tp8",
                           sub="qwen2.5-coder-7b-inst-tp8", extra=[], ctx=32768),
    "coder14":        dict(name="Qwen2.5-Coder-14B-Inst", port=8003, kind="tp8",
                           sub="qwen2.5-coder-14b-inst-tp8", extra=[], ctx=32768),
    "coder14-base":   dict(name="Qwen2.5-Coder-14B tp8", port=8007, kind="tp8",
                           sub="qwen2.5-coder-14b-tp8", extra=[], ctx=32768),
    # Qwen3-Coder-30B-A3B (MoE, --max-model-len 65536 로 빌드). a3b-fp8 은 masquerade 로 FP8 MoE serve
    # 부활시킨 것(artifact.json model_type=qwen3 위장) — 30G 라 1장 OK, 사용자가 8000 에서 운용 중.
    # a3b(bf16)은 58G > 1장 47.5G 라 1장(dp1·pp1)이면 serve OOM → pp≥2(2장 레이어분할)로 띄워야 함.
    "a3b-fp8":        dict(name="Qwen3-Coder-30B-A3B-Inst-FP8 tp8", port=8000, kind="tp8",
                           sub="qwen3-coder-30b-a3b-inst-fp8-tp8-65k-tc", extra=[], ctx=65536),
    "a3b":            dict(name="Qwen3-Coder-30B-A3B-Inst tp8", port=8006, kind="tp8",
                           sub="qwen3-coder-30b-a3b-inst-tp8-65k-tc", extra=[], ctx=65536),
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
    # (Qwen3-Coder-30B-A3B-FP8 은 한때 FP8 MoE serve 미지원으로 제외했으나 masquerade 로 부활 — 위 a3b-fp8.)
}
DISPLAY2KEY = {m["name"]: k for k, m in CATALOG.items()}
DISPLAY_NAMES = [m["name"] for m in CATALOG.values()]
# 기본 선택 모델 = 가장 가벼운 정상 모델 coder7 (coder1.5 는 출력 깨짐으로 제외, 위 참고).
DEFAULT_MODEL = CATALOG["coder7"]["name"]
STARTUP_TIMEOUT = float(os.environ.get("CHAT_SERVE_TIMEOUT", "900"))


def _dd_choices():
    """드롭다운 라벨: '모델명 · tp8/tp32'. 단 이름에 이미 tp 표시가 있으면(예: '… tp8',
    'Qwen3-32B-FP8-tp32') 중복되니 안 붙인다. 값은 모델명(DISPLAY2KEY 키)."""
    out = []
    for m in CATALOG.values():
        name, kind = m["name"], m["kind"]
        label = name if kind.lower() in name.lower() else f"{name}  ·  {kind}"
        out.append((label, name))
    return out


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
        self._pending = {} # key -> (dp, pp). 전환 중에 들어온 새 dp/pp 요청 — 전환 끝나면 즉시 적용
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
            # 전환 중(loading/stopping): 같은 config 면 중복이라 무시. 다른 config 면 '대기 설정'에
            # 적어 두고 리턴 — 진행 중 전환이 정리되는 즉시 그 설정으로 다시 전환한다(변경 중 dp/pp
            # 를 바꿔도 바뀐 값이 반영되도록). 같은 요청 반복(터널 재진입)은 여전히 무해하게 흡수.
            if cur in ("loading", "stopping"):
                if (dp, pp) != self._par.get(key):
                    self._pending[key] = (dp, pp)
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
            self._pending.pop(key, None)   # 새 전환이 권위 — 묵은 대기 설정 비움
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
        # 전환 중에 dp/pp 변경 요청이 들어와 쌓였으면(=_pending), 지금 serve 를 내리고 새 설정으로
        # 즉시 다시 전환한다. (_start_and_wait 도 로딩 중 _pending 을 감지하면 일찍 빠져나온다.)
        with self._lock:
            pending = self._pending.pop(key, None)
            cur_par = self._par.get(key)
        if pending is not None and pending != cur_par:
            self._stop_blocking(key)            # 현재(옛 설정) serve 내리고 카드 반납 → state=down
            self.request(key, pending[0], pending[1])   # 새 설정으로 재전환(cur=down 이라 진행됨)

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
            # 로딩 중에 dp/pp 변경 요청(_pending)이 들어왔으면 이 로딩을 즉시 포기 →
            # _transition 이 옛 serve 를 내리고 새 설정으로 다시 띄운다(변경 중 설정변경 반영).
            with self._lock:
                pend = self._pending.get(key)
            if pend is not None and pend != self._par.get(key):
                return
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

# ── furiosa RNGD 테마 (furiosa-apps chat-playground 디자인: 순수 검정 + 빨강·시안·보라) ──
BG = "#000000"       # 메인 배경(furiosa = 순수 검정)
SIDE = "#0a0a0a"     # 사이드바
ELEV = "#1c1c1c"     # 입력창·검색창
CARD = "#151515"     # 카드·아코디언·선택 항목
BORDER = "#3a3a3a"   # 경계선(furiosa #444 계열)
TXT = "#e0e0e0"
MUTE = "#888888"
RED = "#dc2626"      # furiosa 강조(전송/Enter 버튼)
CYAN = "#76d6ff"     # 메트릭 제목
PURPLE = "#cdbbff"   # 라인·DEMO 배지
CSS = f"""
@keyframes ledpulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.18; }} }}
.led-pulse {{ animation: ledpulse 1.1s ease-in-out infinite; }}
.gradio-container {{ max-width:100% !important; padding:0 !important; background:{BG} !important; color:{TXT} !important; }}
/* 최외곽 래퍼 main.fillable.app 의 max-width:1536·margin(auto→32px)·padding:32px 가 양옆을 비움 → 0/100% 로 덮어 빈 공간 제거 */
.gradio-container .app, main.fillable, main.app {{ max-width:100% !important; margin:0 !important; padding:0 !important; }}
footer, .show-api, .built-with {{ display:none !important; }}
* {{ --color-accent:{CARD} !important; --color-accent-soft:{CARD} !important; }}
.gradio-container .prose, .gradio-container label, .gradio-container span {{ color:{TXT}; }}
input[type=range] {{ accent-color:{RED} !important; }}
input:focus, textarea:focus, .gr-box:focus-within {{ outline:none !important; box-shadow:none !important; }}

/* furiosa 헤더 (로고 + Furiosa RNGD Chat + DEMO 배지 | 모델) */
#furheader {{ background:{BG} !important; border-bottom:1px solid {BORDER}; padding:0 18px !important; min-height:62px; align-items:center !important; gap:0 !important; }}
#brand {{ display:flex; align-items:center; gap:11px; height:62px; }}
#brand img {{ height:26px; width:auto; }}
#brand .ttl {{ color:#fff; font-weight:700; font-size:1.2rem; letter-spacing:.4px; }}
#brand .demo {{ background:{PURPLE}; color:#000; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:700; letter-spacing:.5px; }}
#model-dd {{ max-width:480px !important; min-width:340px !important; }}
#model-dd, #model-dd .wrap, #model-dd .secondary-wrap {{ background:transparent !important; border:none !important; box-shadow:none !important; min-height:0 !important; }}
/* 긴 모델명이 잘리지 않게: 폭 넉넉히 + 한 줄 + 넘치면 …(끝 칩은 input 밖 secondary-wrap 라 안 가림) */
#model-dd input {{ font-weight:600 !important; font-size:13.5px !important; color:{TXT} !important; font-family:monospace !important; cursor:pointer; text-align:right; text-overflow:ellipsis; white-space:nowrap; overflow:hidden; padding-right:2px !important; }}
#model-dd .wrap:hover {{ background:{CARD} !important; border-radius:8px !important; }}

/* 사이드바 */
#sidebar {{ background:{SIDE} !important; border-right:1px solid {BORDER}; padding:10px 8px !important; min-height:96vh; }}
#sidebar .gap, #sidebar .form {{ background:transparent !important; border:none !important; }}
#newchat-btn {{ background:transparent !important; border:1px solid {BORDER} !important; color:{TXT} !important; border-radius:10px !important; font-weight:500; text-align:left; }}
#newchat-btn:hover {{ background:{CARD} !important; border-color:{RED} !important; }}
#search-box {{ background:transparent !important; }}
#search-box input, #search-box textarea {{ background:{ELEV} !important; border:none !important; color:{TXT} !important; border-radius:10px !important; padding:9px 12px !important; }}
#sidebar .label-wrap, #recent-label p {{ color:{MUTE} !important; font-size:12px !important; font-weight:600; padding:6px 6px 2px !important; margin:0 !important; }}
#convo-list {{ border:none !important; background:transparent !important; box-shadow:none !important; }}
#convo-list label {{ display:block !important; width:100%; padding:8px 10px !important; margin:1px 0 !important; border-radius:8px !important; cursor:pointer; color:#cfcfd6 !important; font-size:14px; border:none !important; background:transparent !important; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
#convo-list label:hover {{ background:{CARD} !important; }}
#convo-list input[type=radio] {{ display:none !important; }}
#convo-list label:has(input:checked) {{ background:{CARD} !important; color:#fff !important; box-shadow:inset 2px 0 0 {RED}; }}
#del-btn {{ background:transparent !important; border:none !important; color:{MUTE} !important; font-size:12.5px !important; text-align:left; }}
#del-btn:hover {{ color:{RED} !important; }}
#statusbox, #statusbox div {{ font-size:12.5px; line-height:1.7; color:{TXT} !important; }}
#sidebar .accordion, #sidebar .accordion * {{ border-color:{BORDER} !important; }}
#userchip {{ border-top:1px solid {BORDER}; margin-top:8px; padding:10px 6px 4px; color:{MUTE}; font-size:13px; }}
#settings-acc, #rag-acc {{ border:none !important; background:transparent !important; }}

/* 메인 */
#main {{ background:{BG} !important; }}

/* 우측 실시간 대시보드 */
#dashboard {{ background:{BG} !important; border-left:1px solid {BORDER}; padding:16px 12px !important; min-height:96vh; }}
#dashbox {{ background:transparent !important; }}

/* RAG 컨트롤 */
#rag-files {{ background:{ELEV} !important; border:1px dashed {BORDER} !important; border-radius:10px !important; }}
#rag-info {{ font-size:12px; color:{MUTE}; line-height:1.6; }}

/* 챗봇 — furiosa 풍 (질문=어두운 말풍선, 답변=투명 폭 꽉 채움) */
#chatbot {{ background:transparent !important; border:none !important; max-width:100% !important; margin:0 !important; padding:0 22px !important; }}
#inputwrap {{ max-width:100% !important; margin:0 !important; }}
#chatbot .message-wrap, #chatbot .message-row {{ box-shadow:none !important; }}
#chatbot .message-bubble-border {{ border-color:transparent !important; }}
#chatbot .bot-row .message, #chatbot .bot-row .bubble, #chatbot .bot {{ background:transparent !important; border:none !important; color:{TXT} !important; }}
#chatbot .user-row .message, #chatbot .user-row .bubble, #chatbot .user {{ background:{CARD} !important; border:1px solid {BORDER} !important; color:#fff !important; border-radius:14px !important; }}
#chatbot .avatar-container, #chatbot .avatar-image {{ display:none !important; }}
#chatbot .message, #chatbot .message-content, #chatbot .bubble, #chatbot .message-row {{ opacity:1 !important; }}
#chatbot .user-row .message-content, #chatbot .user-row .message {{ color:#fff !important; }}
#chatbot .bot-row .message-content, #chatbot .bot-row .message {{ color:{TXT} !important; }}
/* 답변(봇)은 채팅 폭을 꽉 채우게 — Gradio 기본 width 제한 해제 (질문 말풍선은 우측 컴팩트 유지) */
#chatbot .bot-row, #chatbot .bot-row .message, #chatbot .bot-row .message-content, #chatbot .bot-row .prose {{ max-width:100% !important; width:100% !important; }}
/* 사고 과정 — 접힘 = 연회색 한 줄, 펼침 = 은은한 박스 안 회색 추론 */
#chatbot details {{ background:transparent !important; border:none !important; padding:0 !important; margin:2px 0 10px !important; }}
#chatbot details summary {{ color:{MUTE} !important; font-size:13.5px !important; cursor:pointer; list-style:none; outline:none; user-select:none; }}
#chatbot details summary::-webkit-details-marker {{ display:none; }}
#chatbot details summary::marker {{ content:""; }}
#chatbot details summary::before {{ content:"▸"; margin-right:6px; color:{MUTE}; font-size:11px; }}
#chatbot details[open] summary::before {{ content:"▾"; }}
#chatbot details[open] {{ background:{CARD} !important; border-radius:10px !important; padding:10px 14px !important; }}
#chatbot details[open] > *:not(summary) {{ color:#9a9aa6 !important; font-size:13px !important; line-height:1.6; }}

/* 입력 알약 */
#inputwrap {{ padding:8px 22px 16px !important; }}
#inputbar {{ background:{ELEV} !important; border:1px solid {BORDER} !important; border-radius:26px !important; padding:6px 6px 6px 18px !important; align-items:center !important; }}
#inputbar:focus-within {{ border-color:{RED} !important; }}
/* 겉 알약 안의 텍스트박스 내부 박스(block/wrap)를 완전히 투명화 → 박스 안 박스 제거 */
#inputbar .block, #inputbar .wrap, #inputbar label, #inputbar .input-container {{ background:transparent !important; border:none !important; box-shadow:none !important; border-radius:0 !important; padding:0 !important; }}
#inputbar textarea {{ background:transparent !important; border:none !important; color:{TXT} !important; box-shadow:none !important; font-size:15px; padding:8px 0 !important; }}
#send-btn, #stop-btn {{ border-radius:50% !important; min-width:38px !important; max-width:38px; width:38px; height:38px; padding:0 !important; font-size:18px; line-height:1; box-shadow:none !important; }}
#send-btn {{ background:{RED} !important; color:#fff !important; border:none !important; }}
#send-btn:hover {{ background:#ef3b3b !important; }}
#stop-btn {{ background:#fff !important; color:#111 !important; border:none !important; }}
#hint {{ color:{MUTE}; font-size:11.5px; text-align:center; padding:6px 0 0; }}
"""
_COLOR = {"up": "#22c55e", "loading": "#eab308", "stopping": "#eab308", "down": "#6b7280", "error": "#ef4444"}

# Base 테마 기본값이 흰색이라 검색창·입력창·아코디언·버튼이 희게 뜨는 것을 막아 검정으로 고정.
_DARKVARS = dict(
    body_background_fill=BG, body_text_color=TXT, body_text_color_subdued=MUTE,
    background_fill_primary=BG, background_fill_secondary=SIDE,
    block_background_fill=BG, block_border_color=BORDER,
    block_label_background_fill=BG, block_label_text_color=MUTE, block_title_background_fill=BG,
    border_color_primary=BORDER, border_color_accent=BORDER,
    input_background_fill=ELEV, input_background_fill_focus=ELEV, input_background_fill_hover=ELEV,
    input_border_color=BORDER, input_placeholder_color=MUTE,
    button_secondary_background_fill=CARD, button_secondary_background_fill_hover="#262626",
    button_secondary_text_color=TXT, panel_background_fill=SIDE, panel_border_color=BORDER,
    color_accent_soft=CARD, code_background_fill="#0d0d0d",
)
THEME = gr.themes.Base(primary_hue="red", secondary_hue="gray", neutral_hue="gray").set(
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


def _stream_reply(base_url, model_id, msgs, temperature, max_tokens, rec=None):
    """답변 스트리밍. 추론(reasoning)은 ChatGPT처럼 연한 회색 접힘 줄로:
    추론 중엔 '💭 생각하는 중…', 끝나면 '💭 N초 동안 생각함'(클릭하면 전체 추론).
    rec 가 주어지면 대시보드용 메트릭(첫 토큰 시각·토큰 수·정확 completion_tokens)을 기록한다.
    stream_options(include_usage)로 furiosa-llm serve 가 주는 정확한 토큰수를 받는다."""
    stream = _client(base_url).chat.completions.create(
        model=model_id, messages=msgs, temperature=temperature, max_tokens=int(max_tokens),
        stream=True, stream_options={"include_usage": True})
    think, body = "", ""
    t0 = time.time()
    think_secs = None
    for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if not chunk.choices:
            if usage is not None and rec is not None:       # 마지막 usage 청크 → 정확한 토큰수
                rec["completion_tokens"] = getattr(usage, "completion_tokens", 0) or 0
            continue
        delta = chunk.choices[0].delta
        r = getattr(delta, "reasoning", None)
        if r:
            think += r
            if rec is not None:
                METRICS.first_token(rec)
                METRICS.add_chars(rec, len(r))   # 글자 누적 → 라이브 TPS 추정(최종은 usage 로 보정)
        if delta.content:
            if think and think_secs is None:           # 추론 끝, 답변 시작 → 걸린 시간 확정
                think_secs = max(1, round(time.time() - t0))
            body += delta.content
            if rec is not None:
                METRICS.first_token(rec)
                METRICS.add_chars(rec, len(delta.content))
        if think:
            label = (f"💭 {think_secs}초 동안 생각함" if think_secs is not None else "💭 생각하는 중…")
            yield _think_block(think, label) + body
        else:
            yield body or "…"


# 전송/중지 버튼 토글 상태
_GEN = (gr.update(visible=False), gr.update(visible=True))     # 생성중: 전송 숨김, 중지 표시
_IDLE = (gr.update(visible=True), gr.update(visible=False))    # 대기: 전송 표시, 중지 숨김


def _generate(history, conv_id, model_name, dp, pp, rag_on, rag_k, system_prompt, temperature, max_tokens):
    """history(마지막 user) 뒤에 답변 스트리밍. 출력 7개:
    (chatbot, conv_id, txt, convo, status, send_btn, stop_btn).
    rag_on 이면 업로드 문서에서 관련 청크를 찾아 컨텍스트로 주입하고 출처를 각주로 단다."""
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

    # ── RAG: 켜져 있고 문서가 있으면 마지막 질문으로 검색해 컨텍스트를 질문 직전에 주입 ──
    rag_sources = []
    if rag_on and RAG.summary()[1] > 0:
        user_q = next((mm["content"] for mm in reversed(history[:-1]) if mm.get("role") == "user"), "")
        ctx, rag_sources = RAG.context(user_q, int(rag_k))
        if ctx:
            rag_sys = {"role": "system", "content": (
                "다음은 사용자가 올린 문서에서 질문과 관련해 찾은 발췌입니다. 답변에 활용하고, "
                "사용한 부분은 [번호]로 인용하세요. 문서에 답이 없으면 일반 지식으로 답하되 그 점을 밝히세요.\n\n"
                + ctx)}
            msgs.insert(len(msgs) - 1, rag_sys)   # 마지막 user 턴 바로 앞에 삽입

    # 컨텍스트 초과 방지: prompt + max_tokens <= ctx. 프롬프트 토큰을 보수적으로 추정해 클램프.
    est_prompt = sum(len(mm.get("content", "")) for mm in msgs) // 3 + 16
    eff_max = max(16, min(int(max_tokens), m["ctx"] - est_prompt - 256))
    rec = METRICS.start()
    try:
        for partial in _stream_reply(base_url, model_id, msgs, temperature, eff_max, rec=rec):
            history[-1]["content"] = partial
            yield history, conv_id, "", gr.update(), gr.skip(), gr.skip(), gr.skip()
        if rag_sources:   # 답변 끝에 출처 각주(furiosa/kotaemon 식 근거 표시)
            history[-1]["content"] += (
                f"\n\n<span style=\"color:#888;font-size:12px;\">🔎 RAG 참조: "
                f"{', '.join(rag_sources)}</span>")
            yield history, conv_id, "", gr.update(), gr.skip(), gr.skip(), gr.skip()
    except Exception as e:
        history[-1]["content"] = f"⚠️ 생성 중 에러: {e}"
        yield history, conv_id, "", gr.update(), gr.skip(), gr.skip(), gr.skip()
    finally:
        METRICS.finish(rec, rec.get("completion_tokens"))
    _save_convo(conv_id, model_name, history)
    yield history, conv_id, "", gr.update(choices=_convo_choices(), value=conv_id), gr.skip(), *_IDLE


def respond(user_msg, history, conv_id, model_name, dp, pp, rag_on, rag_k, system_prompt, temperature, max_tokens):
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
    yield from _generate(history, conv_id, model_name, dp, pp, rag_on, rag_k, system_prompt, temperature, max_tokens)


def regenerate(history, conv_id, model_name, dp, pp, rag_on, rag_k, system_prompt, temperature, max_tokens):
    history = list(history or [])
    if history and history[-1].get("role") == "assistant":
        history.pop()
    if not history or history[-1].get("role") != "user":
        yield history, conv_id, "", gr.update(), gr.skip(), gr.skip(), gr.skip()
        return
    yield from _generate(history, conv_id, model_name, dp, pp, rag_on, rag_k, system_prompt, temperature, max_tokens)


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


# ── 실시간 대시보드 ──────────────────────────────────────────────────
# LED status 와 같은 전략: 타이머는 '생성 중(+여유 5초)'에만 켜고 유휴엔 끈다.
# (이 파일이 실측한 'always-on Timer + gr.skip() 은 매 틱 재렌더되어 깜빡인다'는 발견 때문 —
#  status_tick 주석 참고. dash_timer 를 생성 시 켜고 활동이 멎으면 스스로 끈다.)
_LAST_DASH = {"html": None}
DASH_IDLE_OFF = 5.0   # 마지막 토큰 후 이 시간(초) 지나면 타이머 자동 off


def _active_cards(model_name):
    """선택 모델이 실제 점유한 NPU 카드 인덱스 집합(없으면 None=전체). 대시보드가 '그 모델의
    카드' 전력/온도/사용률만 보도록 — 다른 모델 부하가 새어들지 않게."""
    key = DISPLAY2KEY.get(model_name)
    dev = MGR.device(key) if key else ""
    cards = {int(d.split(":")[1]) for d in dev.split(",") if d.startswith("npu:")}
    return cards or None


def dash_tick(model_name):
    """대시보드 타이머 tick: 선택 모델 카드의 HW 표본 갱신 + furiosa 스타일 HTML 렌더.
    활동이 멎으면(생성 종료 5초 후) 타이머를 스스로 끈다. 출력 2개: (dash, dash_timer)."""
    METRICS.sample(_active_cards(model_name))
    h = METRICS.render_html()
    html_out = gr.skip() if h == _LAST_DASH["html"] else h
    _LAST_DASH["html"] = h
    idle = (time.time() - METRICS.last_activity) > DASH_IDLE_OFF
    timer_out = gr.Timer(active=False) if idle else gr.skip()
    return html_out, timer_out


def start_dash():
    """생성 시작 시 호출: 활동 표시 + 대시보드 타이머 켬(생성 중 라이브 갱신)."""
    METRICS.touch()
    return gr.Timer(active=True)


# ── RAG 컨트롤 핸들러 ────────────────────────────────────────────────
def _rag_info_html(extra=""):
    nd, nc, names = RAG.summary()
    backend = "임베딩 서버" if RAG.backend == "embedding" else "TF-IDF(로컬)"
    head = (f'📚 <b>{nd}</b>개 문서 · <b>{nc}</b>개 청크 · 검색: {backend}'
            if nd else f'문서 없음 · 검색: {backend}')
    lst = "".join(f'<div style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">• {n}</div>'
                  for n in names[:8])
    more = f'<div>… 외 {nd - 8}개</div>' if nd > 8 else ""
    err = f'<div style="color:#dc2626;">{extra}</div>' if extra else ""
    return f'<div id="rag-info">{head}{err}<div style="margin-top:4px;">{lst}{more}</div></div>'


def rag_add_files(files):
    """업로드 파일들을 인덱싱. 출력 2개: (rag_info, file_clear)."""
    errs = []
    for f in files or []:
        path = f if isinstance(f, str) else getattr(f, "name", None)
        if not path:
            continue
        try:
            RAG.add_file(path)
        except Exception as e:
            errs.append(f"{Path(path).name}: {e}")
    return _rag_info_html("; ".join(errs)), None


def rag_add_url_fn(url):
    try:
        n = RAG.add_url(url)
        return _rag_info_html(f"" if n else "내용이 비어 추가 안 됨"), ""
    except Exception as e:
        return _rag_info_html(f"URL 실패: {e}"), url


def rag_add_text_fn(text):
    if text and text.strip():
        name = "붙여넣기 " + dt.datetime.now().strftime("%H:%M:%S")
        RAG.add(name, text)
        return _rag_info_html(), ""
    return _rag_info_html(), text


def rag_clear_fn():
    RAG.clear()
    return _rag_info_html()


def _header_html():
    img = f'<img src="{LOGO_URI}" alt="furiosa"/>' if LOGO_URI else "🔴 "
    return (f'<div id="brand">{img}'
            f'<span class="ttl">Furiosa RNGD Chat</span>'
            f'<span class="demo">DEMO</span></div>')


def build_ui():
    _ctx0 = CATALOG[DISPLAY2KEY[DEFAULT_MODEL]]["ctx"]
    with gr.Blocks(title="Furiosa RNGD Chat", fill_height=True, css=CSS, theme=THEME) as demo:
        conv_id = gr.State("")
        # ── furiosa 헤더: 로고 + Furiosa RNGD Chat + DEMO | 모델 드롭다운 ──
        with gr.Row(elem_id="furheader", equal_height=True):
            with gr.Column(scale=1, min_width=200):
                gr.HTML(_header_html())
            model_dd = gr.Dropdown(_dd_choices(), value=DEFAULT_MODEL, show_label=False,
                                   container=False, elem_id="model-dd", scale=0, min_width=340)
        with gr.Row(equal_height=False):
            # ── 사이드바: 대화 이력 + 모델 상태(LED) + 설정(dp/pp) + RAG ──
            with gr.Column(scale=1, min_width=240, elem_id="sidebar"):
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
                with gr.Accordion("⚙  설정 (dp·pp·생성)", open=False, elem_id="settings-acc"):
                    with gr.Row():
                        dp = gr.Dropdown([1, 2, 3, 4], value=1, label="복제 dp", scale=1,
                                         info="카드마다 복제 — 동시 요청↑. tp8만")
                        pp = gr.Dropdown([1, 2, 3, 4], value=1, label="레이어 분할 pp", scale=1,
                                         info="여러 장에 레이어 분산. dp×pp≤4. tp8만")
                    temp = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="temperature")
                    maxtok = gr.Slider(64, _ctx0, value=_ctx0, step=256, label="max_tokens (답변 최대 길이)",
                                       info="모델 선택 시 그 모델 최대치로 자동 설정. 생성 시 컨텍스트에 맞게 자동 조정됨")
                    sys_box = gr.Textbox(label="시스템 프롬프트", lines=1,
                                         placeholder="예: 너는 한국어로 답하는 코딩 도우미야.")
                with gr.Accordion("📎 문서 검색 (RAG)", open=False, elem_id="rag-acc"):
                    rag_on = gr.Checkbox(value=False, label="RAG 사용 (올린 문서에서 근거 검색)")
                    rag_files = gr.File(label="문서 업로드 (.txt·.md·코드·.pdf)", file_count="multiple",
                                        elem_id="rag-files", height=90)
                    with gr.Row():
                        rag_url = gr.Textbox(placeholder="https:// URL", show_label=False, scale=3, lines=1)
                        rag_url_btn = gr.Button("URL", size="sm", scale=1)
                    with gr.Row():
                        rag_paste = gr.Textbox(placeholder="텍스트 붙여넣기", show_label=False, scale=3, lines=1)
                        rag_paste_btn = gr.Button("추가", size="sm", scale=1)
                    rag_k = gr.Slider(1, 8, value=4, step=1, label="참조 청크 수 (top-k)")
                    rag_info = gr.HTML(value=_rag_info_html())
                    rag_clear = gr.Button("🗑 문서 비우기", size="sm")
                gr.HTML('<div id="userchip">RNGD NPU · furiosa-llm</div>')
            # ── 채팅 ──
            with gr.Column(scale=3, elem_id="main"):
                chatbot = gr.Chatbot(type="messages", elem_id="chatbot", height="74vh",
                                     show_label=False, show_copy_button=True,
                                     allow_tags=True)  # True 라야 <details> 사고과정이 렌더됨
                with gr.Column(elem_id="inputwrap"):
                    with gr.Row(elem_id="inputbar"):
                        txt = gr.Textbox(placeholder="무엇이든 부탁하세요", show_label=False, scale=9,
                                         lines=1, container=False, autofocus=True)
                        send = gr.Button("↑", elem_id="send-btn", scale=0)
                        stop = gr.Button("■", elem_id="stop-btn", scale=0, visible=False)
                    gr.HTML('<div id="hint">RNGD NPU 위 furiosa-llm · 답변은 모델에 따라 부정확할 수 있어요.</div>')
            # ── 우측 실시간 성능 대시보드 (furiosa chat-playground 이식) ──
            with gr.Column(scale=1, min_width=240, elem_id="dashboard"):
                dash = gr.HTML(value=METRICS.render_html(), elem_id="dashbox")

        SP = "hidden"  # 도는 네모(progress 스피너) 끔 → LED 펄스만, 스트리밍/질문 즉시 노출
        # 유휴 시엔 꺼두는 LED 자동갱신 Timer (전환 중에만 켜짐 → 유휴 패널 깜빡임 없음).
        timer = gr.Timer(2.5, active=False)
        # 대시보드 타이머: 유휴엔 꺼 두고(깜빡임·furiosa-smi 폴링 방지), 생성 시작 때 켜서
        # 1.8s마다 HW·TPS 갱신하다가 활동이 멎으면 dash_tick 이 스스로 끈다.
        dash_timer = gr.Timer(1.8, active=False)
        chat_inputs = [txt, chatbot, conv_id, model_dd, dp, pp, rag_on, rag_k, sys_box, temp, maxtok]
        chat_outputs = [chatbot, conv_id, txt, convo_radio, status, send, stop]
        regen_inputs = [chatbot, conv_id, model_dd, dp, pp, rag_on, rag_k, sys_box, temp, maxtok]
        ev1 = txt.submit(respond, chat_inputs, chat_outputs, show_progress=SP)
        ev2 = send.click(respond, chat_inputs, chat_outputs, show_progress=SP)
        ev3 = chatbot.retry(regenerate, regen_inputs, chat_outputs, show_progress=SP)
        stop.click(lambda: _IDLE, None, [send, stop], cancels=[ev1, ev2, ev3], show_progress=SP)
        # 생성 시작과 동시에 대시보드 타이머 켜기(라이브 갱신). 활동이 멎으면 dash_tick 이 끔.
        for _trig in (txt.submit, send.click, chatbot.retry):
            _trig(start_dash, None, dash_timer, show_progress=SP)
        new_btn.click(new_chat, None, [chatbot, conv_id, convo_radio, txt], show_progress=SP)
        del_btn.click(delete_convo, [convo_radio, conv_id], [chatbot, conv_id, convo_radio], show_progress=SP)
        refresh_btn.click(status_force, None, [status], show_progress=SP)
        search.change(filter_convos, [search], [convo_radio], show_progress=SP)
        convo_radio.change(load_chat, [convo_radio], [chatbot, conv_id], show_progress=SP)
        model_dd.change(on_model_change, [model_dd, dp, pp], [status, maxtok, dp, pp, timer], show_progress=SP)
        dp.change(on_par_change, [model_dd, dp, pp], [status, dp, pp, timer], show_progress=SP)
        pp.change(on_par_change, [model_dd, dp, pp], [status, dp, pp, timer], show_progress=SP)
        # RAG: 업로드/URL/붙여넣기/비우기 → 문서 인덱싱 + 정보 패널 갱신
        rag_files.upload(rag_add_files, [rag_files], [rag_info, rag_files], show_progress=SP)
        rag_url_btn.click(rag_add_url_fn, [rag_url], [rag_info, rag_url], show_progress=SP)
        rag_paste_btn.click(rag_add_text_fn, [rag_paste], [rag_info, rag_paste], show_progress=SP)
        rag_clear.click(rag_clear_fn, None, [rag_info], show_progress=SP)
        # 전환 중에만 켜져 LED 를 갱신하고, 끝나면 status_tick 이 Timer 를 스스로 끈다.
        timer.tick(status_tick, None, [status, timer], show_progress=SP)
        # 대시보드: 생성 중에만 켜진 타이머가 선택 모델 카드의 HW·TPS 를 갱신하고 멎으면 스스로 끔.
        dash_timer.tick(dash_tick, [model_dd], [dash, dash_timer], show_progress=SP)
        # 터널 재접속/새 탭: 최신 상태 1회 갱신 + 전환 중이면 Timer 켜기 + 대시보드 마지막 스냅샷.
        demo.load(load_status, None, [status, timer], show_progress=SP)
        demo.load(lambda: METRICS.render_html(), None, [dash], show_progress=SP)
    return demo


if __name__ == "__main__":
    host = os.environ.get("CHAT_HOST", "0.0.0.0")
    port = int(os.environ.get("CHAT_PORT", "7860"))
    root_path = os.environ.get("CHAT_ROOT_PATH", "")
    share = os.environ.get("CHAT_SHARE", "0") == "1"
    _auth = os.environ.get("CHAT_AUTH", "")
    auth = tuple(_auth.split(":", 1)) if ":" in _auth else None
    # default_concurrency_limit>1: 대시보드/LED 타이머가 스트리밍 생성과 동시에 돌아야
    # 생성 중에도 TPS·전력이 라이브로 갱신된다(큐가 직렬화되면 생성 끝까지 멈춰버림).
    build_ui().queue(default_concurrency_limit=12).launch(
        server_name=host, server_port=port, show_error=True,
        root_path=root_path, share=share, auth=auth)
