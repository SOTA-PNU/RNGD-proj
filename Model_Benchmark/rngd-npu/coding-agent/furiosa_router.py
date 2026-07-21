#!/usr/bin/env python3
"""
Furiosa NPU lazy-serving 라우터 (OpenCode 용)
=============================================
OpenAI 호환 엔드포인트 1개를 열어서, rngd-npu/artifacts 의 모든 빌드 아티팩트를
"모델"로 노출한다. OpenCode 모델 선택창(switch model)에 전부 뜬다.

어떤 모델 X 로 /v1/chat/completions 요청이 오면:
  1) X 를 서빙 중인 furiosa-llm serve 가 있으면 그쪽으로 프록시(스트리밍).
  2) 없으면 X 에 맞는 "올바른 옵션"(tool 파서·reasoning 파서·pp·devices)으로
     furiosa-llm serve 를 그 자리에서 띄운다(lazy). NPU 카드가 모자라면
     least-recently-used 백엔드를 내려서(evict) 카드를 확보한다.
  3) 준비되면 그 백엔드로 프록시.

→ OpenCode 에서 모델만 고르면, 첫 요청 때 알아서 올바르게 서빙되어 바로 쓰인다.

실행:
  python3 furiosa_router.py serve              # 라우터 :8400 기동
  python3 furiosa_router.py gen-config PATH    # opencode.json 생성(전 모델 등록)
  python3 furiosa_router.py list               # 등록 모델/플래그 출력
"""
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import atexit
import hmac

ART = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/artifacts"
LOGDIR = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/chat/serve_logs"
FURIOSA_LLM = "/home/jun/furiosa/bin/furiosa-llm"
ROUTER_PORT = 8400
BACKEND_PORT_BASE = 8410
ALL_CARDS = [0, 1, 2, 3]
# 백엔드 serve 준비 대기(초). fxb 허브 모델의 '첫' 기동은 HF 가중치 다운로드까지 포함될 수 있어
# 최대 모델 기준으로 여유 있게(K-EXAONE-236B ≈ 150GB — 실측 대역폭 ~150MiB/s 에서 ~1000s+).
# 다운로드 중 타임아웃되면 백엔드가 죽고 503 이 나며, 재시도 시 HF 가 이어받아 결국 성공한다.
# 필요하면 ROUTER_READY_TIMEOUT 환경변수로 조정.
READY_TIMEOUT = int(os.environ.get("ROUTER_READY_TIMEOUT", "2400"))
CARD_FREE_TIMEOUT = 90     # evict 후 카드 메모리 해제 대기(초)

os.makedirs(LOGDIR, exist_ok=True)   # chat/ 은 git 미추적이라 디렉토리가 없을 수 있음

# ── 모델 레지스트리 ────────────────────────────────────────────────────────
# 공식 모델 17종 전부 (claude-agent/available_model.md — huggingface.co/furiosa-ai, 2026.3).
# model_id = HF 저장소 basename. 두 종류의 아티팩트가 섞여 있다:
#   · prebuilt FXB 저장소  → ~/.cache/furiosa/llm/fxb 에서 해석 (fxb show 로 실측)
#   · v2 아티팩트 저장소   → 저장소에 artifact.json (serve 가 v2026.3 태그를 기본 revision 으로 받음)
# 필드:
#   path      : furiosa-ai/... HF ID (또는 ART 상대/절대 경로 — artifacts/ 시절 호환)
#   tp        : 아티팩트의 tensor_parallel_size(PE 수). cards = ceil(tp/8). tp<8 이면 카드 일부만
#               사용(예: tp4 → npu:X:0-3) — devices 는 _start 가 tp 로 계산.
#   cards     : 점유 NPU 칩 수 (스케줄링 단위 — 부분 점유도 1장으로 계산)
#   tool      : --tool-call-parser 값. None = 파서 없음(tool calling 비활성)
#   reasoning : --reasoning-parser 값 또는 None (thinking 모델만; 아니면 None — 주면 400)
#   extra     : 모델별 추가 serve 인자 (예: K-EXAONE 의 enable_thinking)
#   kind      : "chat"(기본) | "embedding" | "reranker" — chat 이 아니면 opencode/furio 목록에서 제외
#   ctx       : 클라이언트 컨텍스트 한도 힌트 = 아티팩트 max_position_embeddings (serve 엔 전달 안 됨)
# 파서 매핑 근거: available_model.md (2026.3 공식) / tp·ctx 근거: fxb show + artifact.json 실측(2026-07-16)
REGISTRY = {
    # ── agent-ready (tool calling 검증/신뢰 순) ──
    "gpt-oss-120b":                     dict(path="furiosa-ai/gpt-oss-120b",                     tp=32, cards=4, pp=1, tool="openai",     reasoning=None,         ctx=131072),
    "Solar-Open-100B-NVFP4A16":         dict(path="furiosa-ai/Solar-Open-100B-NVFP4A16",         tp=32, cards=4, pp=1, tool="solar_open", reasoning="solar_open", ctx=131072),
    "Qwen3-32B-FP8":                    dict(path="furiosa-ai/Qwen3-32B-FP8",                    tp=32, cards=4, pp=1, tool="hermes",     reasoning="qwen3",      ctx=40960),
    "Llama-3.3-70B-Instruct":           dict(path="furiosa-ai/Llama-3.3-70B-Instruct",           tp=32, cards=4, pp=1, tool="llama3_json", reasoning=None,        ctx=131072),
    # ── chat (파서는 있으나 신뢰도 낮거나 미실측) ──
    "EXAONE-4.0-32B-FP8":               dict(path="furiosa-ai/EXAONE-4.0-32B-FP8",               tp=32, cards=4, pp=1, tool="hermes",     reasoning="exaone4",    ctx=131072),
    "K-EXAONE-236B-A23B-NVFP4A16":      dict(path="furiosa-ai/K-EXAONE-236B-A23B-NVFP4A16",      tp=32, cards=4, pp=1, tool="hermes",     reasoning="deepseek_v3", ctx=262144,
                                             extra=["--default-chat-template-kwargs", '{"enable_thinking": true}']),
    "Qwen3-30B-A3B-Instruct-2507-FP8":  dict(path="furiosa-ai/Qwen3-30B-A3B-Instruct-2507-FP8",  tp=32, cards=4, pp=1, tool="hermes",     reasoning=None,         ctx=262144),
    "Qwen3-30B-A3B-Thinking-2507-FP8":  dict(path="furiosa-ai/Qwen3-30B-A3B-Thinking-2507-FP8",  tp=32, cards=4, pp=1, tool="hermes",     reasoning="qwen3",      ctx=262144),
    "Qwen3-30B-A3B-FP8":                dict(path="furiosa-ai/Qwen3-30B-A3B-FP8",                tp=32, cards=4, pp=1, tool="hermes",     reasoning="qwen3",      ctx=40960),
    "Qwen3-Coder-30B-A3B-Instruct-FP8": dict(path="furiosa-ai/Qwen3-Coder-30B-A3B-Instruct-FP8", tp=32, cards=4, pp=1, tool="hermes",     reasoning=None,         ctx=262144),
    "Qwen3-VL-32B-Instruct":            dict(path="furiosa-ai/Qwen3-VL-32B-Instruct",            tp=32, cards=4, pp=1, tool="hermes",     reasoning=None,         ctx=262144),
    "Llama-3.1-8B-Instruct":            dict(path="furiosa-ai/Llama-3.1-8B-Instruct",            tp=8,  cards=1, pp=1, tool="llama3_json", reasoning=None,        ctx=131072),
    "Qwen3-8B-FP8":                     dict(path="furiosa-ai/Qwen3-8B-FP8",                     tp=8,  cards=1, pp=1, tool="hermes",     reasoning="qwen3",      ctx=40960),
    "Qwen3-4B-FP8":                     dict(path="furiosa-ai/Qwen3-4B-FP8",                     tp=8,  cards=1, pp=1, tool="hermes",     reasoning="qwen3",      ctx=40960),
    "Qwen2.5-0.5B-Instruct":            dict(path="furiosa-ai/Qwen2.5-0.5B-Instruct",            tp=4,  cards=1, pp=1, tool="hermes",     reasoning=None,         ctx=32768),
    # ── 비-chat (furio/opencode 목록에서 제외 — /v1/embeddings·/v1/rerank 로 사용) ──
    "Qwen3-Embedding-8B":               dict(path="furiosa-ai/Qwen3-Embedding-8B",               tp=8,  cards=1, pp=1, tool=None,         reasoning=None,         ctx=40960, kind="embedding"),
    "Qwen3-Reranker-8B":                dict(path="furiosa-ai/Qwen3-Reranker-8B",                tp=8,  cards=1, pp=1, tool=None,         reasoning=None,         ctx=40960, kind="reranker"),
}
# 참고:
#   · reasoning 파서: available_model.md 매핑 그대로. gpt-oss 는 목록에 없음(harmony 가 serve 내부에서
#     reasoning 을 직접 처리 — 파서 불필요, 실측으로 reasoning 필드 출력 확인).
#   · K-EXAONE: enable_thinking 템플릿 kwargs 없이는 추론 불가(available_model.md 비고).
#   · Qwen3-Coder-30B-A3B: 파서는 available_model.md 대로 hermes 이나, 2026.3.0 에 전용 qwen3_coder
#     파서가 없어 tool calling 이 깨진다(실측: 모델의 XML tool 포맷을 hermes 가 파싱 못함 → tool_calls
#     빈 채로 원문이 content 로 누출). 에이전트 도구용 아님 — 코딩 채팅 전용. 도구호출은 gpt-oss-120b·
#     Solar-Open-100B·Qwen3-32B-FP8·Llama-3.3-70B 를 쓸 것.
#   · tp32 모델은 4장 전체 점유 — 한 번에 한 개만 서빙되고 요청 시 LRU 교체.
#   · v2 아티팩트 모델의 첫 사용은 HF 다운로드(수십~백GB) 포함 — READY_TIMEOUT 참고.

# 모델별 tool calling(에이전트) 지원 —
#   ok   : tool calling 실측 OK (gpt-oss·Solar 2026-07 실측, Qwen3-32B·Llama-70B 2026-06 실측)
#   weak : 파서 지정은 있으나 미실측이거나 신뢰도 낮음(a3b MoE 3B-active·소형 모델)
#   no   : 파서 없음 → 채팅 전용
TOOL_SUPPORT = {
    "gpt-oss-120b": "ok",
    "Solar-Open-100B-NVFP4A16": "ok",
    "Qwen3-32B-FP8": "ok",
    "Llama-3.3-70B-Instruct": "ok",
    "EXAONE-4.0-32B-FP8": "weak",
    "K-EXAONE-236B-A23B-NVFP4A16": "weak",
    "Qwen3-30B-A3B-Instruct-2507-FP8": "weak",
    "Qwen3-30B-A3B-Thinking-2507-FP8": "weak",
    "Qwen3-30B-A3B-FP8": "weak",
    "Qwen3-Coder-30B-A3B-Instruct-FP8": "no",   # 실측(2026-07-16): hermes 파서가 Qwen3-Coder XML tool 포맷을 못 읽음 → tool_calls 비고 content 로 원문 누출. 채팅 전용.
    "Qwen3-VL-32B-Instruct": "weak",
    "Llama-3.1-8B-Instruct": "weak",
    "Qwen3-8B-FP8": "weak",
    "Qwen3-4B-FP8": "weak",
    "Qwen2.5-0.5B-Instruct": "weak",
}

DEFAULT_MODEL = "gpt-oss-120b"   # 기본 — tool calling OK + 가중치가 이미 서버에 캐시됨

# 모델 표시명(picker)·컨텍스트 단일 출처 — 서버 opencode.json(gen_config)·/router/models·맥 install.sh 가 공유
NAME_HINT = {"ok": "", "weak": "  [tools~weak]", "no": "  [chat-only]"}
def model_display_name(m):
    kind = REGISTRY.get(m, {}).get("kind", "chat")
    if kind != "chat":
        return f"{m}  [{kind}]"
    return m + NAME_HINT.get(TOOL_SUPPORT.get(m, "ok"), "")


def artifact_path(reg):
    p = reg["path"]
    if p.startswith("/"):
        return p
    local = os.path.join(ART, p)
    # artifacts/ 에 실재하면 그 경로, 아니면 fxb 허브 ID 로 간주(furiosa-llm 이 캐시에서 해석)
    return local if os.path.isdir(local) else p


# ── NPU 카드 상태 ──────────────────────────────────────────────────────────
def npu_used_mem():
    """furiosa-smi status 파싱 → {npu_id: used_GiB}."""
    try:
        out = subprocess.run(["furiosa-smi", "status"], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return {}
    mem = {}
    for line in out.splitlines():
        m = re.search(r"npu(\d+)\b.*?(\d+\.\d+)\s*/\s*\d+\.\d+\s*GiB", line)
        if m:
            mem[int(m.group(1))] = float(m.group(2))
    return mem


def wait_cards_free(cards, timeout=CARD_FREE_TIMEOUT):
    deadline = time.time() + timeout
    while time.time() < deadline:
        mem = npu_used_mem()
        if all(mem.get(c, 0.0) < 2.0 for c in cards):
            return True
        time.sleep(2)
    return False


def free_port(start=BACKEND_PORT_BASE):
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free backend port")


# ── 백엔드(furiosa-llm serve 1개) ─────────────────────────────────────────
class Backend:
    def __init__(self, model_id, port, proc, cards):
        self.model_id = model_id
        self.port = port
        self.proc = proc
        self.cards = cards           # 점유 중인 npu id 리스트
        self.last_used = time.time()

    def alive(self):
        return self.proc.poll() is None


class Router:
    def __init__(self):
        self.running = {}            # model_id -> Backend
        self.lock = threading.RLock()
        atexit.register(self.shutdown_all)

    # 현재 비어 있는 카드: 내 백엔드가 점유 중이지도 않고, furiosa-smi 상 실제로도 비어
    # 있어야 free. (외부 serve 가 든 카드와 충돌 방지)
    def _free_cards(self):
        owned = set()
        for b in self.running.values():
            owned.update(b.cards)
        mem = npu_used_mem()
        return [c for c in ALL_CARDS if c not in owned and mem.get(c, 0.0) < 2.0]

    def _log(self, msg):
        print(f"[router {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def _stop(self, b):
        self._log(f"evict '{b.model_id}' (port {b.port}, cards {b.cards})")
        try:
            b.proc.terminate()
            try:
                b.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                b.proc.kill()
                b.proc.wait(timeout=15)
        except Exception as e:
            self._log(f"  terminate error: {e}")
        self.running.pop(b.model_id, None)
        wait_cards_free(b.cards)

    def _evict_until(self, need):
        # LRU 부터 내려서 need 장 확보
        while len(self._free_cards()) < need and self.running:
            victim = min(self.running.values(), key=lambda b: b.last_used)
            self._stop(victim)

    def _start(self, model_id):
        reg = REGISTRY[model_id]
        need = reg["cards"]
        self._evict_until(need)
        free = self._free_cards()
        if len(free) < need:
            raise RuntimeError(f"need {need} cards, only {len(free)} free")
        cards = free[:need]
        tp = reg.get("tp", 8 * need)
        if tp < 8:   # 카드 일부만 쓰는 아티팩트(예: tp4) — 코어 범위 표기
            devices = ",".join(f"npu:{c}:0-{tp - 1}" for c in cards)
        else:
            devices = ",".join(f"npu:{c}" for c in cards)
        port = free_port()
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", model_id)
        logpath = os.path.join(LOGDIR, f"router-{safe}.log")
        cmd = [
            FURIOSA_LLM, "serve", artifact_path(reg),
            "--served-model-name", model_id,
            "--devices", devices, "--host", "127.0.0.1", "--port", str(port),
            "--enable-prefix-caching",
        ]
        if reg["tool"]:
            cmd += ["--enable-auto-tool-choice", "--tool-call-parser", reg["tool"]]
        if reg["pp"] > 1:
            cmd += ["-pp", str(reg["pp"])]
        if reg["reasoning"]:
            cmd += ["--reasoning-parser", reg["reasoning"]]
        if reg.get("extra"):
            cmd += reg["extra"]
        self._log(f"start '{model_id}' devices={devices} pp={reg['pp']} tool={reg['tool']} reasoning={reg['reasoning']} → :{port}")
        logf = open(logpath, "w")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        b = Backend(model_id, port, proc, cards)
        self.running[model_id] = b
        try:
            self._wait_ready(b, logpath)
        except Exception:
            self._stop(b)
            raise
        self._log(f"ready '{model_id}' on :{port}")
        return b

    def _wait_ready(self, b, logpath):
        import httpx
        deadline = time.time() + READY_TIMEOUT
        url = f"http://127.0.0.1:{b.port}/v1/models"
        while time.time() < deadline:
            if not b.alive():
                tail = ""
                try:
                    with open(logpath) as f:
                        tail = "".join(f.readlines()[-15:])
                except Exception:
                    pass
                raise RuntimeError(f"serve process exited early:\n{tail}")
            try:
                if httpx.get(url, timeout=3).status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError(
            f"serve not ready within {READY_TIMEOUT}s — 첫 사용 모델이면 가중치 다운로드가 진행 중이었을 수 "
            f"있습니다. 재시도하면 다운로드를 이어받습니다(또는 ROUTER_READY_TIMEOUT 을 늘려 재기동).")

    def ensure(self, model_id):
        """model_id 가 서빙되도록 보장하고 백엔드 포트 반환(블로킹)."""
        if model_id not in REGISTRY:
            raise KeyError(model_id)
        # fast-path: 이미 떠 있고 살아있으면 락 없이 즉시 반환. 다른 모델의 콜드스타트가
        # self.lock 을 (최대 READY_TIMEOUT) 잡고 있어도 warm 모델 요청은 막히지 않는다.
        b = self.running.get(model_id)
        if b and b.alive():
            b.last_used = time.time()
            return b.port
        with self.lock:
            b = self.running.get(model_id)
            if b and b.alive():
                b.last_used = time.time()
                return b.port
            if b:  # 죽은 백엔드 정리
                self.running.pop(model_id, None)
            return self._start(model_id).port

    def status(self):
        # 락 없이 스냅샷(list 복사 후 순회) — ensure() 가 콜드스타트 동안 self.lock 을 잡고 있어도
        # 상태 조회는 블로킹되지 않는다. dict 읽기는 GIL 하에서 안전.
        return {mid: dict(port=b.port, cards=b.cards, alive=b.alive(),
                          idle_s=round(time.time() - b.last_used, 1))
                for mid, b in list(self.running.items())}

    def shutdown_all(self):
        for b in list(self.running.values()):
            try:
                b.proc.terminate()
            except Exception:
                pass


ROUTER = Router()


# ── FastAPI 앱 ─────────────────────────────────────────────────────────────
def build_app():
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.background import BackgroundTask
    from starlette.concurrency import run_in_threadpool

    # docs/openapi 자동 엔드포인트는 비활성(인증 미적용 + 0.0.0.0 노출 시 정보유출 방지)
    app = FastAPI(title="furiosa-router", docs_url=None, redoc_url=None, openapi_url=None)
    aclient = httpx.AsyncClient(timeout=httpx.Timeout(None))

    # 선택적 Bearer 인증: SDI_API_KEY(또는 FURIOSA_API_KEY) 가 설정돼 있으면 /v1·/router
    # 요청에 'Authorization: Bearer <key>' 를 요구. (원격 Mac/Win 클라이언트 노출 시 필수)
    API_KEY = os.environ.get("SDI_API_KEY") or os.environ.get("FURIOSA_API_KEY")

    @app.middleware("http")
    async def _auth(request, call_next):
        if API_KEY and request.url.path.startswith(("/v1", "/router")):
            # 상수시간 비교(타이밍 사이드채널로 키 유출 방지)
            if not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {API_KEY}"):
                return JSONResponse({"error": {"message": "missing or invalid API key"}}, status_code=401)
        return await call_next(request)

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list",
                "data": [{"id": m, "object": "model", "owned_by": "furiosa-npu"} for m in REGISTRY]}

    # sync 핸들러(async 아님) — FastAPI 가 threadpool 에서 돌리므로 furiosa-smi 호출·락 대기가
    # 이벤트 루프를 막지 않는다. (async 로 두면 콜드스타트 900s 동안 라우터 전체가 얼어붙음 — 실측)
    @app.get("/router/status")
    def router_status():
        return {"running": ROUTER.status(), "free_cards": ROUTER._free_cards()}

    @app.get("/router/models")
    async def router_models():
        # 표시명(힌트 포함)·컨텍스트 단일 출처 → 맥 install.sh 가 서버와 동일하게 설정
        return {"data": [{"id": m, "name": model_display_name(m), "context": reg["ctx"],
                          "kind": reg.get("kind", "chat")}
                         for m, reg in REGISTRY.items()]}

    def _sse_from_completion(data):
        # 비스트리밍 chat.completion → OpenAI 스트리밍(SSE) 청크로 변환
        cid = data.get("id", "chatcmpl-router")
        created = data.get("created", 0)
        model = data.get("model", "")
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        fr = choice.get("finish_reason")

        def chunk(delta, finish=None):
            o = {"id": cid, "object": "chat.completion.chunk", "created": created,
                 "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            return "data: " + json.dumps(o, ensure_ascii=False) + "\n\n"

        async def gen():
            yield chunk({"role": "assistant"})
            if msg.get("reasoning"):
                yield chunk({"reasoning": msg["reasoning"]})
            if msg.get("content"):
                yield chunk({"content": msg["content"]})
            for i, tc in enumerate(msg.get("tool_calls") or []):
                fn = tc.get("function", {})
                yield chunk({"tool_calls": [{"index": i, "id": tc.get("id") or f"call-router-{i}",
                                             "type": "function",
                                             "function": {"name": fn.get("name"),
                                                          "arguments": fn.get("arguments", "")}}]})
            yield chunk({}, finish=fr or "stop")
            yield "data: [DONE]\n\n"
        return gen()

    async def _proxy(request: Request, subpath: str):
        raw = await request.body()
        try:
            payload = json.loads(raw)
            model = payload.get("model")
        except Exception:
            return JSONResponse({"error": {"message": "invalid JSON body"}}, status_code=400)
        if model not in REGISTRY:
            return JSONResponse({"error": {"message": f"unknown model '{model}'. /v1/models 참고."}}, status_code=404)
        try:
            port = await run_in_threadpool(ROUTER.ensure, model)
        except Exception as e:
            return JSONResponse({"error": {"message": f"failed to serve '{model}': {e}"}}, status_code=503)
        url = f"http://127.0.0.1:{port}/v1/{subpath}"

        # de-stream(옵션): 특정 모델이 스트리밍 tool 파싱이 취약하면 REGISTRY 에 destream=True 를 주어,
        # 백엔드를 비스트리밍으로 호출→견고한 extract_tool_calls→결과를 SSE 로 재구성해 보낸다.
        # 2026.3.0 기준 현재 쓰는 파서(hermes·llama3_json·openai·solar_open)는 모두 스트리밍 tool 파싱을
        # 내장(extract_tool_calls_streaming)하므로 기본은 어떤 모델도 destream 을 켜지 않는다(전량 raw 패스스루).
        needs_destream = (
            subpath == "chat/completions"
            and bool(payload.get("stream"))
            and bool(REGISTRY.get(model, {}).get("destream"))
        )
        if needs_destream:
            body2 = dict(payload)
            body2["stream"] = False
            try:
                r = await aclient.post(url, json=body2, timeout=httpx.Timeout(None))
                data = r.json()
            except Exception as e:
                return JSONResponse({"error": {"message": f"backend error: {e}"}}, status_code=502)
            if r.status_code != 200:
                return JSONResponse(data, status_code=r.status_code)
            return StreamingResponse(_sse_from_completion(data), media_type="text/event-stream")

        req = aclient.build_request("POST", url, content=raw,
                                    headers={"Content-Type": "application/json"})
        resp = await aclient.send(req, stream=True)
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers={"Content-Type": resp.headers.get("content-type", "application/json")},
            background=BackgroundTask(resp.aclose),
        )

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await _proxy(request, "chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _proxy(request, "completions")

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        return await _proxy(request, "embeddings")

    @app.post("/v1/rerank")
    async def rerank(request: Request):
        return await _proxy(request, "rerank")

    return app


# ── opencode.json 생성 ─────────────────────────────────────────────────────
def gen_opencode_json(path):
    models = {m: {"name": model_display_name(m),
                  "limit": {"context": reg["ctx"], "output": 8192}}
              for m, reg in REGISTRY.items() if reg.get("kind", "chat") == "chat"}
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "furiosa": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "FuriosaNPU (router)",
                "options": {"baseURL": f"http://localhost:{ROUTER_PORT}/v1"},
                "models": models,
            }
        },
        "model": f"furiosa/{DEFAULT_MODEL}",
        "small_model": f"furiosa/{DEFAULT_MODEL}",
    }
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {path} with {len(models)} models (provider 'furiosa' → :{ROUTER_PORT})")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "list":
        for m, reg in REGISTRY.items():
            kind = reg.get("kind", "chat")
            agent = TOOL_SUPPORT.get(m, "ok") if kind == "chat" else f"-({kind})"
            print(f"  {m:32s} cards={reg['cards']} pp={reg['pp']} tool={str(reg['tool']):11s} "
                  f"reasoning={str(reg['reasoning']):11s} agent={agent}")
    elif cmd == "gen-config":
        gen_opencode_json(sys.argv[2])
    elif cmd == "serve":
        import uvicorn
        api_key = os.environ.get("SDI_API_KEY") or os.environ.get("FURIOSA_API_KEY")
        # 인증 on/off 는 SDI_API_KEY 설정 여부로 결정:
        #   키 있음 → 네트워크 개방 + Bearer 인증(사용자도 같은 키 필요)
        #   키 없음 → 네트워크 개방 + 무인증(승인된 사내망 사용자는 키 없이 접속)
        # 사내망 공유 서버라 기본은 0.0.0.0. 로컬 전용으로 닫으려면 SDI_BIND=127.0.0.1.
        host = os.environ.get("SDI_BIND") or "0.0.0.0"
        if host != "127.0.0.1" and not api_key:
            ROUTER._log(f"ℹ️  인증 OFF — :{ROUTER_PORT} 가 네트워크에 개방됩니다(키 불필요 모드, 승인된 사내망 전용). "
                        f"키를 요구하려면 SDI_API_KEY 를 설정하세요.")
        pidfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".router.pid")
        with open(pidfile, "w") as f:
            f.write(str(os.getpid()))
        atexit.register(lambda: os.path.exists(pidfile) and os.remove(pidfile))
        authmode = "on" if api_key else ("off(loopback)" if host == "127.0.0.1" else "OFF(open)")
        ROUTER._log(f"furiosa-router up on {host}:{ROUTER_PORT}  ({len(REGISTRY)} models, auth={authmode})  pid={os.getpid()}")
        uvicorn.run(build_app(), host=host, port=ROUTER_PORT, log_level="warning")
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
