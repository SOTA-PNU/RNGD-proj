# PP 미지원 ForCausalLM 을 직접 등록해서 빌드 가능하게 만들기

`furiosa-llm build -pp N` (Pipeline Parallel) 은 모델을 transformer block 들 사이에서 잘라 여러 NPU 카드에 나눠 싣는 기능입니다. **현재 SDK 가 자르는 위치를 아는 architecture 만** 가능하고, 그 목록은 한 dict 안에 들어있습니다. 새 architecture 를 그 dict 에 직접 추가하면 PP 빌드를 시도할 수 있습니다.

본 문서는 그 전 과정을 5단계로 정리합니다.

---

## 0. 현재 PP 지원 architecture (= dict 안의 4개)

`MODEL_ARCH_TO_BLOCK_SPLITTER_AND_WEIGHT_NODE_PATTERN`
**위치:** `~/furiosa/lib/python3.12/site-packages/furiosa_llm/parallelize/block_slicer.py:676`

```python
MODEL_ARCH_TO_BLOCK_SPLITTER_AND_WEIGHT_NODE_PATTERN: Final[
    Dict[str, List[Tuple[Callable[[GraphModule, str], List[List[Tuple[str, str]]]], str]]]
] = {
    "GPTJForCausalLM":             [(get_first_embedding_edge_names, GPTJ_EMBEDDING_WEIGHT_PATTERN),
                                    (get_first_layernorm_edge_names, GPTJ_FIRST_LAYERNORM_WEIGHT_PATTERN)],
    "BertForQuestionAnswering":    [...],
    "RobertaForQuestionAnswering": [...],
    "LlamaForCausalLM":            [(get_first_embedding_edge_names, LLAMA_EMBEDDING_WEIGHT_PATTERN),
                                    (get_first_rms_norm_edge_names, LLAMA_FIRST_RMS_NORM_WEIGHT_PATTERN)],
}
```

이 dict 의 key 에 없는 architecture (`Qwen3ForCausalLM`, `Qwen2ForCausalLM`, `Exaone4ForCausalLM` 등) 는 `-pp N` 빌드 시 다음 라인에서 즉시 에러:

```python
# block_slicer.py:727
raise NotImplementedError(f"Block slicing for {original_model_type_name} is not supported.")
```

---

## 1. PP 빌드가 dict 를 사용하는 흐름

```
furiosa-llm build ... -pp 2
        │
        ▼
 ArtifactBuilder._build_model_artifact         (builder.py:172~)
        │
        ▼
 Pipeline build (= Model Tracing)              torch.fx 로 graph 추출
        │
        ▼
 get_block_boundary_edges(gm, model_type, ...) (block_slicer.py:709~)
        │
        ├──▶ dict.get(model_type.__name__)     ← 여기서 architecture 매칭
        │       │
        │       └─ 등록 안돼있으면 ▶▶ NotImplementedError (조기 실패)
        │
        └──▶ splitter_fn(gm, weight_pattern)   ← graph 안에서 자를 edge 들 찾음
                │
                ▼
        [[(src_node1, dst_node1)], [(src_node2, dst_node2)], ...]
                │  └─ 각 list 안의 (src, dst) 가 graph 의 한 "절단선"
                ▼
 get_blockwise_sliced_gms(gm, node_to_color)   block 단위로 graph 분할
        │
        ▼
 각 GPU 에 GraphModule 할당 → Compile → Save artifact
```

핵심: dict 가 알려주는 건 **"어디서 자르면 transformer block 사이가 되는가"** 의 단서 (= 특정 weight 가 들어가는 edge). splitter 함수가 그 단서를 graph 에서 실제 edge 로 변환합니다.

---

## 2. 필요한 두 가지 — `splitter_fn` + `weight_pattern`

dict 의 value 는 `(splitter_fn, weight_pattern)` 튜플의 리스트입니다. 한 architecture 에 보통 2개:

| # | 의도 | 사용하는 splitter |
|---|---|---|
| ① | **임베딩 직후** 를 첫 block 의 시작으로 표시 | `get_first_embedding_edge_names` |
| ② | **각 block 의 첫 layernorm/RMSNorm** 직전을 block 경계로 표시 | `get_first_layernorm_edge_names` 또는 `get_first_rms_norm_edge_names` |

### 2-1. splitter 함수 4종 (모두 `block_slicer.py` 안)

| 함수 | 라인 | 무엇을 하나 |
|---|---|---|
| `get_first_embedding_edge_names` | 658 | 임베딩 weight 가 들어가는 단 한 개의 edge 반환 — block 시작점 |
| `get_first_layernorm_edge_names` | 239 | 각 transformer block 의 첫 LayerNorm (GPT-J 류) edge 들 반환 |
| `get_first_rms_norm_edge_names` | 325 | 각 transformer block 의 첫 RMSNorm (Llama 류) edge 들 반환 |
| `get_attention_output_layernorm_edge_names` | 186 | Attention 출력 LayerNorm (BERT/RoBERTa 류) edge 들 반환 |

**시그너처는 공통**:
```python
def splitter_fn(gm: torch.fx.GraphModule, weight_pattern: str) -> List[List[Tuple[str, str]]]:
    """graph 안에서 weight_pattern 으로 weight 노드 찾고,
    그 weight 가 들어가는 edge (src_node_name, dst_node_name) 들을 반환."""
```

### 2-2. weight_pattern (정규식) — 등록된 예시

| Architecture | 무엇을 찾는 정규식 | 위치 |
|---|---|---|
| GPTJ embedding | `r"transformer\.wte(\.org_target)?\.weight"` | `block_slicer.py:51` |
| GPTJ first LN | `r"transformer\.h\.\d+\.ln_1(\.org_target)?\.weight"` | `block_slicer.py:49` |
| Llama embedding | `r"(model\.embed_tokens.*\.weight)|..."` | `block_slicer.py:72` |
| Llama first RMSNorm | `r"(model\.layers\.\d+\.input_layernorm\.weight)|..."` | `block_slicer.py:70` |

정규식의 핵심: **모델 state_dict 의 weight 이름 형식**을 정확히 잡아야 함. 가짜 변형(`.org_target` 같은 furiosa 내부 마커, `L__self___model_layers_...` 같은 fx-trace 결과 별명) 까지 모두 OR (`|`) 로 묶어둠.

---

## 3. 새 architecture 추가 — 5단계

대상 예: `Qwen3ForCausalLM` (Llama 와 매우 유사한 구조이므로 가장 만만한 후보)

### Step 1. 모델 구조 분석 — state_dict 의 weight 이름

```python
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-32B", torch_dtype="bfloat16", device_map="cpu")
for k in list(m.state_dict().keys())[:20]:
    print(k)
```

확인할 두 가지:
- **임베딩 weight** 이름 (예: `model.embed_tokens.weight`)
- **각 block 첫 norm** 이름 (예: `model.layers.0.input_layernorm.weight`)

Llama / Qwen2 / Qwen3 / Exaone4 는 거의 같은 이름 규칙을 따릅니다 (`model.layers.<N>.input_layernorm.weight`). 이 경우 Llama 패턴을 그대로 쓰면 됨.

### Step 2. weight pattern 작성

`block_slicer.py` 의 상수 정의 영역 (line 30~80) 끝에 추가:

```python
# block_slicer.py — Llama 정의 다음에 새 정의 추가
QWEN3_FIRST_RMS_NORM_WEIGHT_PATTERN = r"(model\.layers\.\d+\.input_layernorm(\.org_target)?\.weight)"
QWEN3_EMBEDDING_WEIGHT_PATTERN      = r"model\.embed_tokens(\.org_target)?\.weight"
```

⚠️ **fx-trace 시 weight 이름이 살짝 바뀔 수 있음** — 등록된 Llama 패턴에 `L__self___model_layers_\d+_input_layernorm__forward_method___self___weight` 같은 alt 가 OR 로 들어있는 이유. 처음엔 단순 패턴으로 시도하고, 빌드 중 weight 못 찾는다고 assertion 깨지면 그때 alt 추가.

### Step 3. 어떤 splitter 함수를 쓸지 결정

- 임베딩 → `get_first_embedding_edge_names` (4개 architecture 모두 공통)
- block 첫 norm 이 **RMSNorm** (Llama 류) → `get_first_rms_norm_edge_names`
- block 첫 norm 이 **LayerNorm** (GPT-J / BERT 류) → `get_first_layernorm_edge_names`

Qwen3 는 RMSNorm 사용 → `get_first_rms_norm_edge_names` 그대로 재사용.

### Step 4. dict 에 등록

`block_slicer.py:676` 의 dict 안에 한 줄 추가:

```python
MODEL_ARCH_TO_BLOCK_SPLITTER_AND_WEIGHT_NODE_PATTERN = {
    # 기존 4개 ...
    "LlamaForCausalLM": [...],

    # ── 추가 ──
    "Qwen3ForCausalLM": [
        (get_first_embedding_edge_names, QWEN3_EMBEDDING_WEIGHT_PATTERN),
        (get_first_rms_norm_edge_names,  QWEN3_FIRST_RMS_NORM_WEIGHT_PATTERN),
    ],
}
```

### Step 5. 빌드 시도 + 패턴 보정

```bash
RAY_memory_monitor_refresh_ms=0 \
  furiosa-llm build Qwen/Qwen3-32B-FP8 \
    ~/RNGD-proj/.../artifacts/qwen3-32b-fp8-tp8pp2  -tp 8 -pp 2
```

실패 시나리오 → 대처:

| 메시지 | 원인 / 수정 |
|---|---|
| `NotImplementedError: Block slicing for Qwen3ForCausalLM is not supported.` | dict 등록 안 됐거나 architecture 이름 오타. `block_slicer.py:716` 의 lookup 확인 |
| `AssertionError: len(embedding_weights) == 1` | 임베딩 weight pattern 이 0개 or 2개 매칭. fx-traced graph 의 실제 이름 확인 |
| `AssertionError` (splitter 내부) | 첫 norm pattern 이 block 수만큼 매칭 안 됨. `.org_target`, `L__self___...` 같은 alt 를 OR 로 추가 |
| trace 자체가 실패 (`AttributeError` 등) | architecture 가 Llama 와 다른 부분 (custom attention, MoE routing 등). 이건 dict 만으로 안 됨 — model wrapper 패치 필요 |

---

## 4. 등록된 weight 이름이 무엇인지 빠르게 추출하는 도구

빌드를 실제로 한 번 돌려 trace 된 weight 이름을 직접 봐야 정확한 패턴을 만들 수 있을 때:

```python
# 별도 스크립트 — graph 의 _param 노드 이름 덤프
import torch
from furiosa_llm.optimum import AutoModelForCausalLM  # furiosa-llm 의 trace 거친 모델

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-32B-FP8")  # 우리가 보고 싶은 모델
gm = model._exported_gm                                              # 내부 GraphModule
for n in gm.graph.nodes:
    if n.op == 'get_attr' and n.name.startswith("_param"):
        print(n.name, "→", get_original_name(n))   # get_original_name 은 block_slicer 안 helper
```

또는 더 간단히:

```python
import re, torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-32B", torch_dtype=torch.bfloat16)
for k in m.state_dict().keys():
    if re.search(r"(embed_tokens|input_layernorm).*weight$", k):
        print(k)
```

여기서 본 이름을 그대로 정규식으로 옮기면 됨.

---

## 5. 한계 — dict 만으로 안 되는 케이스

| 케이스 | 이유 |
|---|---|
| **MoE** (Qwen3_moe, Llama-4) | expert routing 노드가 block 안에 추가 → block 경계가 단순 norm 으로 안 정의됨. splitter 함수 새로 작성 필요 |
| **Multi-modal** (Llama-3.2 Vision, Gemma-3) | vision encoder 와 LLM 두 graph 가 합쳐져 있음. encoder 부분의 block 정의가 별개 |
| **custom attention** (Gemma sliding window, Mistral SWA, Phi 시리즈) | tracing 자체가 깨질 수 있음 — `furiosa_llm/models/` 안에 model wrapper 가 있어야 함. dict 추가 전에 wrapper 부터 |
| **shared embedding + lm_head tied weights** | 마지막 block 절단 위치가 흔들림. 후처리 필요 |

이런 경우는 SDK 자체 패치 (`furiosa_llm/models/<arch>/...`) 가 우선이고, dict 추가는 마지막 단계.

---

## 6. 정리 — 수정/참고 파일 한눈에

| 역할 | 경로 | 무엇을 |
|---|---|---|
| 본체 dict | `furiosa_llm/parallelize/block_slicer.py:676` | 새 architecture entry 추가 |
| weight pattern 상수 | `block_slicer.py:30~80` | `<ARCH>_EMBEDDING_WEIGHT_PATTERN`, `<ARCH>_FIRST_*_WEIGHT_PATTERN` 추가 |
| splitter 함수 (재사용) | `block_slicer.py:186, 239, 325, 658` | 그대로 사용 또는 신규 작성 |
| dict 호출처 (수정 X) | `block_slicer.py:716` | NotImplementedError 발생 지점 — 디버깅용 |
| TP/PP/device 검증 (수정 X) | `furiosa_llm/artifact/validator.py:234` | `pp ≥ 1`, `ceil(tp/8)*pp ≤ 8` 만 체크. dict 와 무관 |
| Bucket preset (선택) | `furiosa_llm/artifact/presets.py` | PRESET_REFS 에 `(arch_model_type, h, i)` 등록 시 자동 preset 매칭 |

위 4개 파일만 건드리면 dict-level PP 등록은 끝. 그 후 빌드 → 실패 메시지 따라 패턴 조정 → 통과 확인.
