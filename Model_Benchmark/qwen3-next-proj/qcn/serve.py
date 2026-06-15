"""Qwen3-Coder-Next on RNGD NPU — OpenAI 호환 서빙 (A안: host 추론 루프 + API 래퍼).

native furiosa-llm serve 는 DeltaNet 의 cross-step 순환상태(read-modify-write)를
append-only paged-KV 런타임이 못 들어서 벤더 전용(2026.3+). 그래서 우리 host 추론
루프(qcn/model.py QCNModel, 상태를 host 가 보유)를 OpenAI 호환 HTTP API 로 감싼다.

- 모델 1회 로드(가중치는 레이어별 스트리밍이라 init 가벼움; 느린 건 토큰당 NPU 컴퓨트).
- 요청당 전역 lock 으로 직렬화(단일 NPU 상태/디바이스 충돌 방지). 비스트리밍.
- 엔드포인트: GET /v1/models, POST /v1/completions, POST /v1/chat/completions, GET /health.

실행: PYTHONPATH=<proj> RNGD_DEV=rngd:2 ~/furiosa/bin/python -m uvicorn qcn.serve:app --host 0.0.0.0 --port 8900
또는:  PYTHONPATH=<proj> RNGD_DEV=rngd:2 ~/furiosa/bin/python qcn/serve.py
"""
import torch  # noqa: F401  (furiosa backend 로드 전 torch 먼저)
import os, time, threading, asyncio, json
from typing import List, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# 프로덕션 서빙 = 속도 우선 → DPE(systolic matmul) 기본 ON (prefill 4.69x / decode 1.59x,
# bf16 ~0.23% rel — 실모델 FP8/bf16이라 충실). f32-정확 검증이 필요하면 QCN_DPE=0 으로 실행.
# (model 임포트 '전에' 설정해야 attn/moe/deltanet 의 QCN_DPE 분기가 반영됨.)
os.environ.setdefault("QCN_DPE", "1")

from qcn.model import QCNModel

MODEL_NAME = "qwen3-coder-next-fp8-rngd"

# --- 모델 1회 로드 + 요청 직렬화 lock ---
_model = None
_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        _model = QCNModel()  # 가중치 레이어별 스트리밍, dev=RNGD_DEV
    return _model


app = FastAPI(title="Qwen3-Coder-Next on RNGD")


# ---------- OpenAI 스키마 ----------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = MODEL_NAME
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 64
    temperature: Optional[float] = 0.0  # greedy(=0) 만 지원(현 구현)


class CompletionRequest(BaseModel):
    model: Optional[str] = MODEL_NAME
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 64
    temperature: Optional[float] = 0.0


def _run_generate(prompt_str: str, max_tokens: int, chat: bool):
    """blocking 생성 (lock 으로 직렬화). generate() dict 반환."""
    with _lock:
        m = get_model()
        return m.generate(prompt_str, max_new_tokens=max_tokens, chat=chat,
                          greedy=True, verbose=False)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": os.environ.get("RNGD_DEV", "rngd:0")}


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "rngd-host-loop"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if req.temperature and req.temperature > 0:
        raise HTTPException(400, "only greedy (temperature=0) is supported by the host-loop server")
    # 마지막 user 메시지를 프롬프트로(간단형); chat 템플릿은 generate(chat=True) 가 적용
    user_msgs = [m for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(400, "no user message")
    prompt = user_msgs[-1].content
    t0 = time.time()
    out = await asyncio.to_thread(_run_generate, prompt, req.max_tokens, True)
    dt = time.time() - t0
    return {
        "id": "chatcmpl-rngd", "object": "chat.completion", "created": int(t0),
        "model": req.model or MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": out["generated_text"]},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(out["prompt_ids"]),
            "completion_tokens": len(out["generated_ids"]),
            "total_tokens": len(out["prompt_ids"]) + len(out["generated_ids"]),
        },
        "timing": {"wall_s": round(dt, 1)},
    }


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if req.temperature and req.temperature > 0:
        raise HTTPException(400, "only greedy (temperature=0) is supported")
    prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
    t0 = time.time()
    out = await asyncio.to_thread(_run_generate, prompt, req.max_tokens, False)
    return {
        "id": "cmpl-rngd", "object": "text_completion", "created": int(t0),
        "model": req.model or MODEL_NAME,
        "choices": [{"index": 0, "text": out["generated_text"], "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": len(out["prompt_ids"]),
            "completion_tokens": len(out["generated_ids"]),
            "total_tokens": len(out["prompt_ids"]) + len(out["generated_ids"]),
        },
    }


if __name__ == "__main__":
    # 시작 시 모델 미리 로드(첫 요청 지연 줄이기)
    get_model()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8900")))
