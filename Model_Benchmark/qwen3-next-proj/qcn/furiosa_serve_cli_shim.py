"""literal `furiosa-llm serve <host-loop-artifact>` CLI 가 우리 HostLoopEngine 을 쓰게 만드는
   Python shim. **네이티브 .so 를 바이너리 패치하지 않는다.**

배경(2026-06-15 정찰): native_runtime.so 에는 recurrent-state/state_pool/ExternalOperator/
plugin/register_hook 같은 익스포트 심볼·문자열이 0건이고, 디코드·KV·스케줄러는 전부 내부
mangled Rust 다. 즉 "call X; ret 0" 식으로 우리 코드를 끼울 합법적 콜백 훅이 없다(있으려면
주소 기반 inline detour 인데, 상태 처리는 단일 함수가 아니라 *구조적으로 부재한 서브시스템*
이라 한 호출부 패치로 안 된다). 대신 furiosa-llm 의 **Python serve 층**이 엔진을 덕타이핑
(api.py:94 `engine: Union[NativeLLMEngine, FakeNativeLLMEngine]`)하므로, `LLM` 생성만 가로채
우리 엔진을 꽂으면 *정식 serve CLI 경로*(app.run_server → load_llm_from_args → LLM →
AsyncLLMEngine.from_llm → OpenAIServingChat)가 그대로 우리 host 루프를 구동한다.

사용법(정식 CLI 를 우리 엔진으로):
  PYTHONPATH=<proj> RNGD_DEV=rngd:4 ~/furiosa/bin/python -c \
    "import qcn.furiosa_serve_cli_shim; from furiosa_llm.cli.main import main; \
     import sys; sys.argv=['furiosa-llm','serve','<host-loop-artifact-dir>']; main()"
"""
import os
import sys
import json

PROJ = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj"
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import furiosa_llm.api as _api


class _HostLoopMeta:
    """furiosa-llm 이 llm.model_metadata 에서 읽는 최소 필드."""
    task = "generate"
    trust_remote_code = True


_orig_init = _api.LLM.__init__


def _is_host_loop_artifact(path):
    aj = os.path.join(str(path), "artifact.json")
    if not os.path.isfile(aj):
        return False
    try:
        d = json.load(open(aj))
    except Exception:
        return False
    # runtime=host-loop 또는 entry_point 가 qcn.model 이면 우리 host-loop 아티팩트
    return d.get("runtime") == "host-loop" or "qcn.model" in str(d.get("entry_point", ""))


def _patched_init(self, model_id_or_path=None, *args, **kwargs):
    """host-loop 아티팩트면 HostLoopEngine 을 꽂고(네이티브 엔진·serde 게이트 우회),
    아니면 원래 furiosa-llm 동작."""
    path = model_id_or_path if model_id_or_path is not None else kwargs.get("model_id_or_path")
    if path is not None and _is_host_loop_artifact(path):
        from qcn.furiosa_serve_adapter import HostLoopEngine
        # pp4: QCN_PP set or QCN_CARDS>1 -> PipelineModel (layers split across cards).
        # Otherwise the proven single-card QCNModel. Both expose prefill/decode_step.
        _pp = os.environ.get("QCN_PP") or (int(os.environ.get("QCN_CARDS", "1")) > 1)
        if _pp:
            from qcn.pipeline import PipelineModel
            m = PipelineModel(artifact_dir=str(path))
            print(f"[cli-shim] pp 모드: PipelineModel({m.n_stages} stages {m.ranges})", flush=True)
        else:
            from qcn.model import QCNModel
            m = QCNModel()
        self.engine = HostLoopEngine(m)
        self.tokenizer = m.get_tokenizer()
        self.model_metadata = _HostLoopMeta()
        self.artifact_id = os.path.basename(str(path).rstrip("/"))  # Model.from_llm reads this
        self.prompt_max_seq_len = int(m.cfg_d.get("max_position_embeddings", 262144))
        self.max_seq_len_to_capture = 65536
        self.default_generation_config = {}
        # OpenAIServingChat -> ModelConfigAdapter(config_dict: dict) (chat_utils.py:167)
        # 이라 HF config 객체가 아니라 config.json **dict** 가 필요하다.
        try:
            self.model_config = json.load(open(os.path.join(str(path), "config.json")))
        except Exception:
            self.model_config = {}
        print(f"[cli-shim] host-loop 아티팩트 감지 -> HostLoopEngine 주입 (네이티브 엔진 우회): {path}",
              flush=True)
        return
    return _orig_init(self, model_id_or_path, *args, **kwargs)


def install():
    if getattr(_api.LLM.__init__, "_qcn_shimmed", False):
        return
    _patched_init._qcn_shimmed = True
    _api.LLM.__init__ = _patched_init
    print("[cli-shim] furiosa_llm.api.LLM.__init__ 패치 완료 (host-loop 아티팩트는 우리 엔진 사용)",
          flush=True)


# import 시 자동 설치
install()
