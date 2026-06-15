"""Qwen3-Coder-Next on RNGD NPU — MULTI-CARD OpenAI 호환 서빙 (serve.py 의 업그레이드).

설계 한 줄 요약: **N개 워커 프로세스(카드당 1개) + 부모 FastAPI 가 라운드로빈 디스패치 +
SSE 스트리밍**. serve.py 는 단일 QCNModel + threading.Lock 로 모든 요청을 직렬화했지만,
여기서는 카드 N장에 모델 복제본 N개를 띄워 ~N배 처리량을 낸다(단일 요청 지연은 동일 ~35s/tok).

────────────────────────────────────────────────────────────────────────────
왜 멀티프로세싱인가 (단일 프로세스 멀티스레드가 아니라):
  qcn.attn_layer / qcn.moe / qcn.deltanet_layer_looped 는 디바이스(RNGD_DEV)와
  DPE 플래그(QCN_DPE)를 **import 시점에 모듈 전역으로** 읽는다
    - attn_layer.py:51  DEV = os.environ["RNGD_DEV"]
    - attn_layer.py:60  DPE = os.environ["QCN_DPE"]=="1"
    - moe.py:46 DEV / moe.py:52 DPE
    - deltanet_layer_looped.py:26 DPE
  한 프로세스 안에서는 모듈이 한 번만 import 되므로 카드를 여럿 잡을 수 없다. 그래서
  **워커마다 독립 프로세스**를 띄우고, 각 워커가 qcn.model 을 import 하기 *전에*
  os.environ["RNGD_DEV"]=rngd:k, os.environ["QCN_DPE"]=... 를 설정한다. 이러면 워커 k
  의 모든 NPU 연산이 카드 k 로 고정된다. 가중치(~75GB)는 mmap 이라 4개 리더가 OS 페이지
  캐시를 공유 → RAM 중복 없음.

프로세스 모델:
  parent(FastAPI/uvicorn) ── job_q(k) ──▶ worker k (carded QCNModel, 1 seq at a time)
                          ◀─ res_q ──────┘   (token/usage 스트림을 res_q 로 push)
  - 워커 k 전용 입력 큐 job_q[k] (multiprocessing.Queue): {req_id, kind, payload}
  - 공용 결과 큐 res_q (multiprocessing.Queue): {req_id, ev, ...}  ev∈token/usage/error/started
  - 부모는 결과 큐를 단일 reader thread 로 비우고 req_id 별 asyncio.Queue 로 fan-out.

스케줄러/큐:
  - 워커는 "한 번에 시퀀스 1개"만 처리(모델이 B==1, model.py:201). busy 플래그로 추적.
  - admit(): FREE 워커가 있으면 즉시 배정(라운드로빈 시작점), 없으면 asyncio 대기 큐에서
    FREE 가 생길 때까지 await. → 자연스러운 admission control + 백프레셔.
  - in-flight req_id 추적, /v1/* 취소(클라이언트 disconnect) 시 워커에 cancel 신호 + busy 해제.

견고성:
  - 워커가 죽으면(sentinel exit) 부모가 감지해 해당 req 에 error 푸시 + 워커 재기동(respawn).
  - 워커 응답 watchdog: 토큰 간 간격이 STALL_S 를 넘으면 그 요청을 실패시키고 워커를 재기동
    (느린/멈춘 워커가 서버 전체를 막지 않게). 다른 워커/요청은 영향 없음.

실행:
  PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj \
    /home/jun/furiosa/bin/python qcn/serve_mc.py
  환경변수:
    QCN_CARDS   사용할 카드 수 (기본: furiosa-smi 로 감지한 카드 수, 최대 4)
    QCN_DEVS    "0,1,2,3" 처럼 카드 인덱스 직접 지정(있으면 QCN_CARDS 무시)
    QCN_DPE     1(기본, 빠름) / 0(f32 정확)
    PORT        8900
    QCN_STALL_S 토큰 무응답 watchdog 초 (기본 600 = prefill 77s + 여유)
"""
import os
import sys
import json
import time
import uuid
import asyncio
import threading
import multiprocessing as mp
from typing import List, Optional, Union, Dict

# ⚡ 멀티카드 처리량의 진짜 병목은 NPU 가 아니라 HOST CPU 다(실측: host-loop 한 요청이
# torch.compile 디스패치+looped DeltaNet/attn/MoE glue 로 ~40코어 점유). 캡 없이 N개를
# 돌리면 128코어를 서로 뺏어 각자 ~N배 느려진다. 그래서 워커별 스레드를 코어/N 로 캡한다.
# OMP/MKL 은 torch import '전에' 설정해야 적용되므로 모듈 top 에서 QCN_THREADS(부모가 스폰
# 전에 설정→spawn 자식이 env 상속) 를 읽어 건다.
_qcn_threads = os.environ.get("QCN_THREADS")
if _qcn_threads:
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, _qcn_threads)

import torch  # noqa: F401  (furiosa backend 로드 전 torch 먼저 — serve.py 와 동일 규칙)
if _qcn_threads:
    try:
        torch.set_num_threads(int(_qcn_threads))
    except Exception:
        pass
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

MODEL_NAME = "qwen3-coder-next-fp8-rngd"
STALL_S = float(os.environ.get("QCN_STALL_S", "600"))  # 토큰 무응답 watchdog


# ============================================================================
# 카드 감지: furiosa-smi info --format json 의 dev_name(npu0..) → rngd 인덱스
# ============================================================================
def detect_cards() -> List[int]:
    """사용할 카드 인덱스 리스트. QCN_DEVS 가 있으면 그걸 쓰고, 없으면 furiosa-smi
    로 감지한 전체 카드(최대 QCN_CARDS)를 0..N-1 로 반환."""
    explicit = os.environ.get("QCN_DEVS", "").strip()
    if explicit:
        return [int(x) for x in explicit.replace(" ", "").split(",") if x != ""]
    n_detected = 4
    try:
        import subprocess
        out = subprocess.run(["furiosa-smi", "info", "--format", "json"],
                             capture_output=True, text=True, timeout=15)
        data = json.loads(out.stdout)
        n_detected = len(data)
    except Exception:
        pass
    want = int(os.environ.get("QCN_CARDS", str(n_detected)))
    n = max(1, min(want, n_detected))
    return list(range(n))


# ============================================================================
# 워커 프로세스: 카드 k 에 고정된 QCNModel 하나. job_q 에서 작업을 받아 res_q 로
# 토큰/usage 를 스트리밍한다. qcn.model 을 import 하기 *전에* env 를 박는 게 핵심.
# ============================================================================
def _worker_main(card_idx: int, dpe: str, job_q: mp.Queue, res_q: mp.Queue):
    # --- import 전에 디바이스/DPE 고정 (모듈 전역 env 캡처 때문) ---
    os.environ["RNGD_DEV"] = f"rngd:{card_idx}"
    os.environ["QCN_DPE"] = dpe
    sys.path.insert(0, "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj")

    # 워커별 torch 스레드 캡(런타임). 모듈 top 의 OMP/MKL env 와 함께 N 워커가 코어를 나눠 쓰게.
    _cap = os.environ.get("QCN_THREADS")
    if _cap:
        try:
            torch.set_num_threads(int(_cap))
        except Exception:
            pass

    try:
        from qcn.model import QCNModel
        model = QCNModel()  # dev = RNGD_DEV = rngd:card_idx
        # 토크나이저/캐시 워밍: 첫 요청 지연 줄이기 위해 미리 로드
        model.get_tokenizer()
        res_q.put({"worker": card_idx, "ev": "ready"})
    except Exception as e:  # 로드 실패 → 부모에 알리고 종료
        import traceback
        res_q.put({"worker": card_idx, "ev": "fatal",
                   "error": f"{e}\n{traceback.format_exc()}"})
        return

    while True:
        job = job_q.get()
        if job is None:  # 종료 sentinel
            break
        req_id = job["req_id"]
        p = job["payload"]
        try:
            res_q.put({"worker": card_idx, "req_id": req_id, "ev": "started"})
            stream = model.generate_stream(
                p["prompt"],
                max_new_tokens=p["max_new_tokens"],
                chat=p["chat"],
                greedy=p["greedy"],
                temperature=p["temperature"],
                top_p=p["top_p"],
            )
            for item in stream:
                if item["type"] == "token":
                    res_q.put({"worker": card_idx, "req_id": req_id, "ev": "token",
                               "token_id": item["token_id"], "text": item["text"],
                               "step": item["step"]})
                else:  # usage
                    res_q.put({"worker": card_idx, "req_id": req_id, "ev": "usage",
                               "prompt_ids": item["prompt_ids"],
                               "generated_ids": item["generated_ids"],
                               "generated_text": item["generated_text"],
                               "prefill_s": item["prefill_s"],
                               "per_token_s": item["per_token_s"],
                               "finish_reason": item["finish_reason"]})
            res_q.put({"worker": card_idx, "req_id": req_id, "ev": "done"})
        except Exception as e:
            import traceback
            res_q.put({"worker": card_idx, "req_id": req_id, "ev": "error",
                       "error": f"{e}\n{traceback.format_exc()}"})


# ============================================================================
# 부모 측 워커 핸들: 프로세스 + 전용 입력 큐 + busy/상태
# ============================================================================
class Worker:
    def __init__(self, card_idx: int, dpe: str, res_q: mp.Queue, ctx):
        self.card_idx = card_idx
        self.dpe = dpe
        self.res_q = res_q
        self.ctx = ctx
        self.job_q: mp.Queue = ctx.Queue()
        self.proc: Optional[mp.Process] = None
        self.busy_req: Optional[str] = None   # 현재 처리 중 req_id (없으면 FREE)
        self.ready = False
        self.start()

    def start(self):
        self.proc = self.ctx.Process(
            target=_worker_main,
            args=(self.card_idx, self.dpe, self.job_q, self.res_q),
            daemon=True,
        )
        self.proc.start()
        self.ready = False

    @property
    def free(self) -> bool:
        return self.busy_req is None and self.ready and self.alive

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def submit(self, req_id: str, payload: dict):
        self.busy_req = req_id
        self.job_q.put({"req_id": req_id, "payload": payload})

    def respawn(self):
        """죽었거나 stall 된 워커를 깨끗이 재기동. 새 job_q 로 교체(쌓인 작업 폐기)."""
        try:
            if self.proc is not None and self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=5)
        except Exception:
            pass
        self.job_q = self.ctx.Queue()  # 오염된 큐 폐기
        self.busy_req = None
        self.start()


# ============================================================================
# 스케줄러: 부모 단일 인스턴스. res_q reader thread + req 별 asyncio.Queue fan-out.
# ============================================================================
class Scheduler:
    def __init__(self):
        self.cards = detect_cards()
        self.dpe = os.environ.get("QCN_DPE", "1")
        self.ctx = mp.get_context("spawn")  # furiosa/torch fork-unsafe → spawn
        self.res_q: mp.Queue = self.ctx.Queue()
        # ⚡ 워커별 CPU 스레드 캡 = 코어수 / 워커수 (스폰 전에 env 설정 → 자식 상속).
        # 명시 QCN_THREADS 가 있으면 존중. 캡으로 N 워커가 코어를 나눠 써 진짜 병렬 처리량 확보.
        if not os.environ.get("QCN_THREADS"):
            try:
                _ncpu = len(os.sched_getaffinity(0))
            except Exception:
                _ncpu = os.cpu_count() or 128
            os.environ["QCN_THREADS"] = str(max(4, _ncpu // max(1, len(self.cards))))
        self.workers: List[Worker] = [
            Worker(c, self.dpe, self.res_q, self.ctx) for c in self.cards
        ]
        # req_id -> {"q": asyncio.Queue, "loop": loop, "worker": Worker,
        #            "last": ts, "started": bool, "cancelled": bool}
        self.reqs: Dict[str, dict] = {}
        self.lock = threading.Lock()
        self._rr = 0  # 라운드로빈 시작 인덱스
        self._free_event = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # 결과 큐 reader (백그라운드 스레드: blocking get 을 asyncio 로 브릿지)
        self._reader = threading.Thread(target=self._drain_results, daemon=True)
        self._reader.start()
        # watchdog (stall 감지 → 워커 재기동)
        self._wd = threading.Thread(target=self._watchdog, daemon=True)
        self._wd.start()

    def bind_loop(self, loop):
        self._loop = loop

    # ---------- 워커 선택 (라운드로빈으로 FREE 찾기) ----------
    def _pick_free(self) -> Optional[Worker]:
        n = len(self.workers)
        for off in range(n):
            w = self.workers[(self._rr + off) % n]
            if w.free:
                self._rr = (self._rr + off + 1) % n
                return w
        return None

    async def admit(self, req_id: str, payload: dict) -> "asyncio.Queue":
        """FREE 워커에 배정. 없으면 생길 때까지 await. req 별 asyncio.Queue 반환."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        while True:
            with self.lock:
                w = self._pick_free()
                if w is not None:
                    self.reqs[req_id] = {
                        "q": q, "loop": loop, "worker": w,
                        "last": time.time(), "started": False, "cancelled": False,
                    }
                    w.submit(req_id, payload)
                    return q
            # FREE 없음 → 이벤트 대기(다른 요청이 끝나면 set)
            self._free_event.clear()
            try:
                await asyncio.wait_for(self._free_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass  # 주기적으로 재시도(워커 재기동 등 상태 변화 흡수)

    def cancel(self, req_id: str):
        """클라이언트 disconnect 등으로 요청 취소. 워커는 한 시퀀스를 도중에 멈출 수
        없으므로(host 루프), 워커를 재기동해 즉시 FREE 로 만든다(다른 요청 보호)."""
        with self.lock:
            info = self.reqs.get(req_id)
            if not info:
                return
            info["cancelled"] = True
            w = info["worker"]
            if w.busy_req == req_id:
                w.respawn()  # 진행 중인 무거운 디코드를 끊고 카드 회수
            self.reqs.pop(req_id, None)
        self._notify_free()

    def _notify_free(self):
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._free_event.set)

    def _push(self, req_id: str, item: dict):
        info = self.reqs.get(req_id)
        if not info:
            return
        info["last"] = time.time()
        loop = info["loop"]
        loop.call_soon_threadsafe(info["q"].put_nowait, item)

    # ---------- 결과 큐 reader ----------
    def _drain_results(self):
        while True:
            try:
                msg = self.res_q.get()
            except (EOFError, OSError):
                break
            ev = msg.get("ev")
            if ev == "ready":
                wk = self.workers[self._idx_of(msg["worker"])]
                wk.ready = True
                self._notify_free()
                continue
            if ev == "fatal":
                # 워커 로드 실패 → 재기동 시도
                idx = self._idx_of(msg["worker"])
                print(f"[sched] worker card {msg['worker']} fatal: "
                      f"{msg['error'][:300]}", flush=True)
                self.workers[idx].respawn()
                continue

            req_id = msg.get("req_id")
            if req_id is None:
                continue
            with self.lock:
                info = self.reqs.get(req_id)
            if ev == "started":
                if info:
                    info["started"] = True
                    info["last"] = time.time()
                continue

            if ev == "token":
                self._push(req_id, {"type": "token", "token_id": msg["token_id"],
                                    "text": msg["text"], "step": msg["step"]})
            elif ev == "usage":
                self._push(req_id, {"type": "usage", **{k: msg[k] for k in (
                    "prompt_ids", "generated_ids", "generated_text",
                    "prefill_s", "per_token_s", "finish_reason")}})
            elif ev == "error":
                self._push(req_id, {"type": "error", "error": msg.get("error", "")})
                self._finish(req_id)
            elif ev == "done":
                self._push(req_id, {"type": "done"})
                self._finish(req_id)

    def _finish(self, req_id: str):
        """요청 종료: 워커 FREE 해제 + req 정리 + 대기자 깨우기."""
        with self.lock:
            info = self.reqs.pop(req_id, None)
            if info:
                w = info["worker"]
                if w.busy_req == req_id:
                    w.busy_req = None
        self._notify_free()

    def _idx_of(self, card_idx: int) -> int:
        for i, w in enumerate(self.workers):
            if w.card_idx == card_idx:
                return i
        return 0

    # ---------- watchdog: stall/dead 워커 회수 ----------
    def _watchdog(self):
        while True:
            time.sleep(5)
            now = time.time()
            for w in self.workers:
                # 죽은 워커(busy 상태로) → 진행 중 요청 실패 + 재기동
                if not w.alive and w.busy_req is not None:
                    rid = w.busy_req
                    print(f"[wd] worker card {w.card_idx} died on req {rid}; respawn",
                          flush=True)
                    self._push(rid, {"type": "error", "error": "worker died"})
                    self._finish(rid)
                    w.respawn()
                    continue
                # stall: busy 인데 STALL_S 동안 토큰/이벤트 무소식
                rid = w.busy_req
                if rid is not None:
                    with self.lock:
                        info = self.reqs.get(rid)
                    if info and (now - info["last"]) > STALL_S:
                        print(f"[wd] worker card {w.card_idx} stalled on req {rid} "
                              f"({now - info['last']:.0f}s); respawn", flush=True)
                        self._push(rid, {"type": "error", "error": "worker stalled"})
                        self._finish(rid)
                        w.respawn()

    # ---------- 상태 ----------
    def health(self) -> dict:
        ws = []
        for w in self.workers:
            ws.append({
                "card": f"rngd:{w.card_idx}",
                "alive": w.alive,
                "ready": w.ready,
                "busy": w.busy_req is not None,
                "req_id": w.busy_req,
            })
        n_free = sum(1 for w in self.workers if w.free)
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "cards": len(self.workers),
            "free": n_free,
            "in_flight": len(self.reqs),
            "dpe": self.dpe,
            "workers": ws,
        }


SCHED: Optional[Scheduler] = None  # lifespan 에서 생성


# ============================================================================
# FastAPI
# ============================================================================
from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # 서버 부팅 시 N개 워커(카드당 1개)를 spawn + 모델 로드. Scheduler 가 spawn
    # 컨텍스트를 쓰므로, 이 호출은 uvicorn 이 __main__ import 를 마친 뒤(또는
    # uvicorn qcn.serve_mc:app 진입점) 일어나 spawn-안전하다.
    global SCHED
    if SCHED is None:
        SCHED = Scheduler()
    SCHED.bind_loop(asyncio.get_running_loop())
    yield


app = FastAPI(title="Qwen3-Coder-Next on RNGD (multi-card)", lifespan=_lifespan)


# ---------- OpenAI 스키마 ----------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = MODEL_NAME
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 64
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: Optional[str] = MODEL_NAME
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 64
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False


def _build_payload(prompt: str, max_tokens: int, chat: bool,
                   temperature: float, top_p: float) -> dict:
    greedy = (not temperature) or temperature <= 0.0
    return {
        "prompt": prompt,
        "max_new_tokens": int(max_tokens),
        "chat": chat,
        "greedy": greedy,
        "temperature": float(temperature or 0.0),
        "top_p": float(top_p if top_p is not None else 1.0),
    }


async def _collect_full(req_id: str, q: "asyncio.Queue"):
    """비스트리밍: 토큰을 모아 (text, usage) 반환. error 면 예외."""
    text_parts = []
    usage = None
    while True:
        item = await q.get()
        t = item["type"]
        if t == "token":
            text_parts.append(item["text"])
        elif t == "usage":
            usage = item
        elif t == "error":
            raise RuntimeError(item.get("error", "worker error"))
        elif t == "done":
            break
    full = usage["generated_text"] if usage else "".join(text_parts)
    return full, usage


# ---------------- /health, /v1/models ----------------
@app.get("/health")
def health():
    if SCHED is None:
        return {"status": "starting"}
    return SCHED.health()


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [
        {"id": MODEL_NAME, "object": "model", "owned_by": "rngd-host-loop-mc"}]}


# ---------------- chat ----------------
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, raw: Request):
    user_msgs = [m for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(400, "no user message")
    prompt = user_msgs[-1].content
    payload = _build_payload(prompt, req.max_tokens, True, req.temperature, req.top_p)
    req_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    q = await SCHED.admit(req_id, payload)

    if req.stream:
        return StreamingResponse(
            _chat_sse(req_id, q, req.model or MODEL_NAME, created, raw),
            media_type="text/event-stream")

    # 비스트리밍: 모아서 한 방에
    try:
        full, usage = await _collect_full(req_id, q)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    pu = len(usage["prompt_ids"]) if usage else 0
    cu = len(usage["generated_ids"]) if usage else 0
    return {
        "id": req_id, "object": "chat.completion", "created": created,
        "model": req.model or MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full},
            "finish_reason": (usage or {}).get("finish_reason", "stop"),
        }],
        "usage": {"prompt_tokens": pu, "completion_tokens": cu,
                  "total_tokens": pu + cu},
    }


async def _chat_sse(req_id, q, model_name, created, raw: Request):
    """OpenAI chat SSE: 첫 청크는 role, 이후 content delta, 마지막 finish_reason
    + [DONE]. 청크 모양은 serving_chat.py:382-404 를 따른다."""
    def chunk(delta, finish=None):
        return {
            "id": req_id, "object": "chat.completion.chunk", "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
    # 첫 청크: role
    yield f"data: {json.dumps(chunk({'role': 'assistant'}))}\n\n"
    finish = "stop"
    try:
        while True:
            if await raw.is_disconnected():
                SCHED.cancel(req_id)
                return
            try:
                item = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            t = item["type"]
            if t == "token":
                yield f"data: {json.dumps(chunk({'content': item['text']}))}\n\n"
            elif t == "usage":
                finish = item.get("finish_reason", "stop")
            elif t == "error":
                yield f"data: {json.dumps(chunk({}, finish='error'))}\n\n"
                yield "data: [DONE]\n\n"
                return
            elif t == "done":
                break
    finally:
        pass
    yield f"data: {json.dumps(chunk({}, finish=finish))}\n\n"
    yield "data: [DONE]\n\n"


# ---------------- completions ----------------
@app.post("/v1/completions")
async def completions(req: CompletionRequest, raw: Request):
    prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
    payload = _build_payload(prompt, req.max_tokens, False, req.temperature, req.top_p)
    req_id = "cmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    q = await SCHED.admit(req_id, payload)

    if req.stream:
        return StreamingResponse(
            _cmpl_sse(req_id, q, req.model or MODEL_NAME, created, raw),
            media_type="text/event-stream")

    try:
        full, usage = await _collect_full(req_id, q)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    pu = len(usage["prompt_ids"]) if usage else 0
    cu = len(usage["generated_ids"]) if usage else 0
    return {
        "id": req_id, "object": "text_completion", "created": created,
        "model": req.model or MODEL_NAME,
        "choices": [{"index": 0, "text": full,
                     "finish_reason": (usage or {}).get("finish_reason", "stop")}],
        "usage": {"prompt_tokens": pu, "completion_tokens": cu,
                  "total_tokens": pu + cu},
    }


async def _cmpl_sse(req_id, q, model_name, created, raw: Request):
    """OpenAI text_completion SSE: text delta 청크 + [DONE]."""
    def chunk(text, finish=None):
        return {
            "id": req_id, "object": "text_completion", "created": created,
            "model": model_name,
            "choices": [{"index": 0, "text": text, "finish_reason": finish}],
        }
    finish = "stop"
    try:
        while True:
            if await raw.is_disconnected():
                SCHED.cancel(req_id)
                return
            try:
                item = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            t = item["type"]
            if t == "token":
                yield f"data: {json.dumps(chunk(item['text']))}\n\n"
            elif t == "usage":
                finish = item.get("finish_reason", "stop")
            elif t == "error":
                yield f"data: {json.dumps(chunk('', finish='error'))}\n\n"
                yield "data: [DONE]\n\n"
                return
            elif t == "done":
                break
    finally:
        pass
    yield f"data: {json.dumps(chunk('', finish=finish))}\n\n"
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    # spawn 컨텍스트를 강제(furiosa/torch 는 fork 불가). Scheduler 가 spawn 사용.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8900")))
