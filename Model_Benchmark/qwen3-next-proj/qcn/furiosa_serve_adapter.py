"""furiosa-llm serve 스택을 '우리 host 추론 루프'로 구동하는 어댑터.

배경(2026-06-15 조사): furiosa-llm 의 디코드 루프·KV풀·스케줄러·cross-step 상태는
전부 컴파일된 네이티브 엔진(`furiosa.native_runtime.llm.NativeLLMEngine`, Rust .so,
소스 없음) 안에 있어 우리가 거기에 순환상태 풀을 못 넣는다. **그러나** Python serve
층(`AsyncLLMEngine`·`server/app.py`·`OpenAIServingChat`)은 `llm.engine` 을
`Union[NativeLLMEngine, FakeNativeLLMEngine]` 로 **덕타이핑**만 한다(api.py:94). 그리고
네이티브 출력 클래스(`NativeRequestOutput`/`NativeCompletionOutput`)는 평범한 int/str
인자로 Python 에서 생성 가능하다(llm.pyi:53-83).

따라서 네이티브 엔진과 **같은 인터페이스**(stream_generate/encode/abort_request/
is_alive/shutdown/generate)를 가진 Python 엔진을 만들어 끼우면, furiosa-llm 자신의
`AsyncLLMEngine.generate` → `NativeOutputConverter.convert_stream`(outputs.py:335) 이
우리 host 루프를 그대로 구동한다. **cross-step 순환상태 풀 = 요청별 QCNModel state_cache
(이 어댑터가 요청 수명 동안 host 에 보유), sub-op 체이닝 = 우리 host 루프의 NPU 커널
순차 호출.** 즉 "순환상태 풀 + sub-op 체이닝"을 *네이티브가 아니라 Python serve 층*에
구현해 furiosa-llm serve 서버 스택을 재사용한다.

핵심 인터페이스(AsyncLLMEngine.generate, llm_engine.py:578-627):
  native_engine.stream_generate(batch_encoding, sampling_params, request_id)
    -> async generator of NativeRequestOutput(outputs=[NativeCompletionOutput(
         index, token_ids=<이번 스텝 신규 토큰들>, finish_reason)])
  convert_stream 이 token_ids 를 누적·디토큰해 RequestOutput 으로 변환.
"""
import asyncio
import torch
from furiosa.native_runtime.llm import NativeRequestOutput, NativeCompletionOutput


class HostLoopEngine:
    """NativeLLMEngine 과 같은 덕타입 인터페이스를 갖는 Python 엔진.
    QCNModel host 추론 루프를 구동하고, 요청별 순환상태를 host 에 보유한다."""

    def __init__(self, model, lock=None):
        self.model = model                       # QCNModel
        self.tok = model.get_tokenizer()
        self._abort = set()
        self._lock = lock or asyncio.Lock()      # 단일 NPU/모델 직렬화
        self._eos = set()
        if self.tok.eos_token_id is not None:
            self._eos.add(int(self.tok.eos_token_id))
        try:
            im = self.tok.convert_tokens_to_ids("<|im_end|>")
            if isinstance(im, int) and im >= 0:
                self._eos.add(im)
        except Exception:
            pass

    # ---- 네이티브 엔진과 동일한 표면 ----
    def is_alive(self) -> bool:
        return True

    def shutdown(self) -> None:
        pass

    def abort_request(self, request_id: str) -> None:
        self._abort.add(request_id)

    async def encode(self, *a, **k):
        raise NotImplementedError("pooling 미지원 (generation 전용 어댑터)")

    def generate(self, *a, **k):
        raise NotImplementedError("동기 generate 미사용; stream_generate 사용")

    async def stream_generate(self, batch_encoding, sampling_params, request_id=None):
        """furiosa-llm AsyncLLMEngine 이 호출. batch_encoding.input_ids = 프롬프트 토큰.
        스텝마다 NativeRequestOutput(신규 토큰 1개) yield, 마지막에 finish_reason."""
        ids = batch_encoding.input_ids
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        ids = [int(x) for x in ids]
        max_new = int(getattr(sampling_params, "max_tokens", None) or 16)
        temp = float(getattr(sampling_params, "temperature", 0.0) or 0.0)
        greedy = temp <= 0.0

        async with self._lock:
            # ---- prefill (heavy NPU work -> thread) ----
            ids_t = torch.tensor([ids]).long()
            logits, cache = await asyncio.to_thread(self.model.prefill, ids_t)
            next_logits = logits[0, -1]
            pos = len(ids)
            finish = "length"

            for step in range(max_new):
                if request_id in self._abort:
                    self._abort.discard(request_id)
                    finish = "abort"
                    break
                nxt = int(torch.argmax(next_logits)) if greedy else \
                    int(torch.multinomial(torch.softmax(next_logits.float() / temp, -1), 1))
                if nxt in self._eos:
                    finish = "stop"
                    break
                # 신규 토큰 1개를 델타로 yield (convert_stream 이 누적)
                yield NativeRequestOutput(
                    outputs=[NativeCompletionOutput(index=0, token_ids=[nxt],
                                                    finish_reason=None)],
                    num_cached_tokens=len(ids),
                )
                # ---- decode_step: cache(=순환상태 풀)를 host 가 보유·갱신 ----
                step_logits = await asyncio.to_thread(self.model.decode_step, nxt, pos, cache)
                next_logits = step_logits[0, -1]
                pos += 1

            # 종료 신호 (빈 토큰 + finish_reason)
            yield NativeRequestOutput(
                outputs=[NativeCompletionOutput(index=0, token_ids=[],
                                                finish_reason=finish)],
                num_cached_tokens=len(ids),
            )


def build_async_engine(model, prompt_max_seq_len=262144, max_seq_len_to_capture=65536):
    """furiosa-llm 의 진짜 AsyncLLMEngine 을 우리 HostLoopEngine 으로 구성해 반환.
    이걸 OpenAIServingChat / app.py 에 그대로 넘기면 furiosa-llm serve 스택이 구동된다."""
    from furiosa_llm.llm_engine import AsyncLLMEngine
    eng = HostLoopEngine(model)
    return AsyncLLMEngine(
        native_engine=eng,
        tokenizer=model.get_tokenizer(),
        task_type="generate",
        prompt_max_seq_len=prompt_max_seq_len,
        max_seq_len_to_capture=max_seq_len_to_capture,
        llm=None,
    )
