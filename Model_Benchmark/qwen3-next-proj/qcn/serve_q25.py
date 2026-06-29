"""원본 bf16 Qwen2.5-72B-Instruct OpenAI 호환 서빙 (host 추론 루프 Q25Model).

표준 furiosa-llm serve가 bf16 72B를 못 띄우므로(inter-chip 가중치 바인딩/DramShapeGuide
미구현), host가 추론 루프를 들고 레이어별로 가중치를 NPU에 스트리밍하는 Q25Model을
OpenAI HTTP API로 감싼다. 요청당 전역 lock 직렬화(단일 NPU 상태). greedy만.

⚠️ bf16 dense 72B는 토큰마다 135GiB 스트리밍이라 매우 느림(~수백초/토큰). 정확도 우선.
실행: PYTHONPATH=<proj> RNGD_DEV=rngd:4 ~/furiosa/bin/python qcn/serve_q25.py  (PORT=8009)
"""
import torch  # noqa: torch first
import os, time, threading, asyncio
from typing import List, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

os.environ.setdefault("QCN_DPE", "1")  # model import 전에
from qcn.qwen25_model import Q25Model

MODEL_NAME = "qwen2.5-72b-inst"
_model = None
_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        _model = Q25Model()
    return _model


app = FastAPI(title="Qwen2.5-72B (bf16) on RNGD host-loop")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = MODEL_NAME
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 32
    temperature: Optional[float] = 0.0


def _run(prompt, max_tokens, chat):
    with _lock:
        return get_model().generate(prompt, max_new_tokens=max_tokens, chat=chat,
                                    greedy=True, verbose=False)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": os.environ.get("RNGD_DEV", "rngd:4"),
            "note": "bf16 72B host-loop; very slow (~수백초/토큰)"}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "rngd-host-loop"}]}


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if req.temperature and req.temperature > 0:
        raise HTTPException(400, "only greedy (temperature=0) supported")
    users = [m for m in req.messages if m.role == "user"]
    if not users:
        raise HTTPException(400, "no user message")
    t0 = time.time()
    out = await asyncio.to_thread(_run, users[-1].content, req.max_tokens, True)
    return {"id": "chatcmpl-q25", "object": "chat.completion", "created": int(t0),
            "model": req.model or MODEL_NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": out["generated_text"]},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(out["prompt_ids"]), "completion_tokens": len(out["generated_ids"]),
                      "total_tokens": len(out["prompt_ids"]) + len(out["generated_ids"])},
            "timing": {"wall_s": round(time.time() - t0, 1), "per_token_s": [round(x, 1) for x in out["per_token_s"]]}}


if __name__ == "__main__":
    get_model()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8009")))
