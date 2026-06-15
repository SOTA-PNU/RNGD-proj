# RNGD 지원 연산자(op) 목록 재검증 + 실제 실행 검증

`furiosa.torch`가 지원한다고 알려진 ATen 연산자(op) 목록이 (1) 정확한지, (2) 목록에 있다고
**실제로 NPU에서 컴파일·실행이 되는지**를 이 머신에서 하나씩 직접 돌려 확인한 기록입니다.

- 검증일: 2026-06-09
- 환경: `furiosa-torch` **2026.2.0**, `torch` **2.10.0** (venv `~/furiosa/`), RNGD 4장
  (firmware 2026.2.1 / driver 2026.2.0, `furiosa-smi`로 확인). 실행은 비어 있는 카드
  `rngd:3`에서 했습니다.
- 출처(SDK 내부): 목록 정의는 `~/furiosa/lib/python3.12/site-packages/furiosa/torch/db/aten_config.py`,
  컴파일 경로는 `furiosa.torch.CompileModule.from_exported` 입니다.

---

## 한 줄 결론

목록(97개)은 **거의 정확하지만 1개가 틀렸고**, 더 중요한 건 **목록에 "지원"으로 올라
있어도 실제로는 안 되거나(2개) 특정 모양에서만 되는(6개) op가 있다**는 점입니다. 즉
`SUPPORTED_ATEN_OPS`는 "컴파일러 프론트엔드가 받겠다고 선언한 목록"이지, "무조건 EDF로
내려가 NPU에서 돈다"는 보장이 아닙니다.

---

## 1. 목록 재검증 — 97개 중 1개 오류

기존에 정리해 둔 목록은 `from furiosa.torch.extension import SUPPORTED_ATEN_OPS` 라고 적혀
있었는데, **두 가지가 틀렸습니다.**

1. **import 경로가 틀렸습니다.** `SUPPORTED_ATEN_OPS`는 `furiosa.torch.extension`에 **없습니다**
   (확인: `hasattr(furiosa.torch.extension, 'SUPPORTED_ATEN_OPS')` → `False`). 올바른 경로는
   **`from furiosa.torch.db import SUPPORTED_ATEN_OPS`** 입니다(`furiosa.torch.db.aten_config`에서 정의).
   - 참고: 그냥 `from furiosa.torch.db import ...` 만 하면 triton 더블 등록 에러가 날 수 있는데,
     `import torch` → `import furiosa.torch` 순서로 먼저 부르거나
     `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 을 주면 됩니다.

2. **목록 내용 1개 오류.** 직접 뽑은 97개와 기존 정리본을 `comm`으로 비교하니 딱 하나
   어긋났습니다.
   - 기존 정리본: `detach_copy` (Copy/Clone 칸)
   - 실제 목록: `copy_` (제자리 복사, in-place copy)
   - 즉 **`detach_copy`는 목록에 없고, 대신 `copy_`가 있습니다.** 나머지 96개는 정확히
     일치했습니다. (개수는 양쪽 다 97개라 헷갈리기 쉬웠습니다.)

> `IMPORTABLE_ATEN_OPS`(156개)도 같은 파일에서 확인했습니다. `supported`(97) ⊂ `importable`(156)
> 이고, 차이(59개)는 "받아서 분해(decomposition)하면 통과하는" op들입니다.

### 목록이 만들어지는 방식 (왜 "지원=실행"이 아닌가)

`aten_config.py`를 보면 목록은 이렇게 만들어집니다.

```python
if check_aten(native_compiler.is_supported_aten):   # 네이티브 컴파일러가 "받는다"고 답한 것
    supported_aten_ops.append(full_func_name)
```

즉 **네이티브 컴파일러의 프론트엔드가 "이 op는 입력으로 받을 수 있다"고 선언한 것**이
`SUPPORTED_ATEN_OPS`입니다. 그런데 받는 것과, 그걸 끝까지 **EDF 코드로 내려서 NPU에서 돌리는
것**은 별개라서, 아래처럼 "지원 목록엔 있는데 실제론 막히는" op가 생깁니다.

---

## 2. 실제 실행 검증 방법

op 하나마다 그 op만 쓰는 최소 그래프를 만들어 → `torch.export` → 분해
(`core_aten_decompositions`, furiosa-llm이 쓰는 그 경로) → `CompileModule.from_exported`(AOT
컴파일) → `rngd:3`에서 실행 → CPU 결과와 비교했습니다.

```python
import torch
import furiosa.torch                       # import 순서: torch 먼저, 그 다음 이걸
from furiosa.torch import CompileModule
from torch._decomp import core_aten_decompositions
TABLE = dict(core_aten_decompositions())

class Mod(torch.nn.Module):
    def forward(self, x):
        return <op 표현식>
ep = torch.export.export(Mod().eval(), (x,)).run_decompositions(TABLE)
cm = CompileModule.from_exported(ep)        # 여기서 막히면 그 op는 EDF로 못 내려가는 것
dev = torch.device("rngd", 3)
cm.to(dev)
out = cm(x.to(dev), device=dev)             # CPU 결과와 비교
```

검증은 한 번이 아니라 **3단계 + 교차검증**으로 했습니다: ①op 단독 그래프, ②실패한 건
`sigmoid()+add` 같은 실제 연산 그래프에 **끼워서** 재시험(단독 그래프가 곧이곧대로 안 되는
경우가 많아서), ③서로 다른 dtype·랭크·모양·API로 흔들어 보기. 마지막으로 8개 의심 op는
**독립 에이전트들이 각자 다른 방식으로 "되게 만들어 보라"는 반대 입장**으로 다시 돌려
교차 확인했습니다.

---

## 3. 결과 요약

| 분류 | 개수 | 내용 |
|---|---|---|
| **그냥 잘 됨** | ~89 | 단항/이항 사칙연산, 활성화, 비교/논리/비트, 대부분의 reduction·shape·view·split·copy, where, full 류 등. 컴파일+실행 OK, 결과 CPU와 일치. |
| **됨(단, 정밀도 낮음)** | 3 | `convolution`, `mm`, `bmm` — 정상 실행되지만 **상대오차 ≈ 0.23%**(BF16/TF32급 텐서엔진). 코사인 유사도 0.9999975. 틀린 게 아니라 NPU 정밀도입니다. |
| **조건부(특정 모양에서만)** | 6 | `cumsum`, `index.Tensor`, `index_select`, `gather`, `slice_scatter`, `max_pool2d_with_indices` — 아래 표 참고. |
| **목록엔 있지만 실제로 안 됨** | 2 | `isnan`, `constant_pad_nd` — 어떤 모양·dtype·끼워넣기로도 컴파일 실패. |

> 정밀도 주의: 일반 elementwise 연산은 CPU와 사실상 동일(상대오차 6e-8)하지만,
> **matmul 계열(conv/mm/bmm)만 상대오차 ~0.23%로 일정**하게 납니다. 크기를 키워도 상대오차는
> 그대로고 절대오차만 커집니다(값이 커지니까). NPU 텐서엔진의 정상적인 감소정밀도입니다.

---

## 4. "지원이라는데 안 되는/조건부인" op 상세 (핵심)

전부 `CompileModule.from_exported`에서 `UnsupportedOpError('failed to compile the graph')`로
막힙니다(NPU 실행 단계까지 가지도 못함). op 자체는 분해되지 않고 그래프에 그대로 남아 있어서,
**네이티브 EDF 백엔드에 그 op의 코드 생성이 없거나 모양 제약이 있는 것**이 원인입니다.

| op | 상태 | 되는 조건 / 안 되는 조건 | 근거(실측) |
|---|---|---|---|
| `isnan` | ❌ 완전 불가 | float16/32/bf16, 랭크 0~4, where/any/cast로 감싸기 등 11가지 다 실패 | `(x != x)`로 바꾸면(=`ne.Tensor`) 컴파일·실행 OK → 막는 건 `isnan` 노드 하나가 확실. **우회: NaN 검출은 `x != x`로** |
| `constant_pad_nd` | ❌ 완전 불가 | F.pad / 1D·2D·3D pad, 0/비0 value, conv 뒤, Linear 앞 등 12가지 다 실패 | 주변 op(conv/sigmoid/addmm)는 다 사는데 pad 노드 때문에 그래프 전체가 실패 |
| `cumsum` | ⚠️ 조건부 | **정수(int32/int64) 입력만** 됨. **float 입력은 전부 실패** | int64 1D/2D/3D 모두 OK(CPU와 bit-exact). float는 dim·랭크 불문 실패 |
| `index_select` | ⚠️ 조건부 | **맨 안쪽(마지막) 차원이 8의 배수**여야 됨 | rows는 4~32 다 OK, cols 스윕: 8·16 OK / 3·5·7·15·17 실패 |
| `index.Tensor`(`x[idx]`) | ⚠️ 조건부 | **맨 안쪽 차원이 4의 배수**면 됨(8보다 덜 빡빡) | cols 스윕: 4·8·12·16 OK / 3·5·7 실패. 단 `x[:, idx]`(비-선두축)은 실패 |
| `gather` | ⚠️ 조건부 | **사실상 1차원(벡터) gather만** 됨. 여러 행을 한꺼번에 모으는 일반 rank-2는 실패 | 1D dim0 OK / (R,C) 다중행은 정렬·dim 불문 전부 실패, 안쪽 축 gather도 실패 |
| `slice_scatter` | ⚠️ 사실상 불가 | **해당 축 전체를 덮어쓰는 경우만** 됨(=결과가 src와 동일, 의미 없음). 부분 덮어쓰기(진짜 쓸모 있는 것)는 실패 | full-overwrite는 OK, start/end로 일부만 바꾸면 전부 실패 |
| `max_pool2d_with_indices` | ⚠️ 사실상 불가 | **indices(int64) 출력은 절대 안 됨.** values만 써도 kernel≥2는 거의 실패(16×16+앞에 elementwise가 붙는 드문 경우만 우연히 됨). ResNet stem(56×56,k3s2p1) 실패 | `[info/README_vision_compile.md]`의 "중간 풀링이 그래프를 쪼갠다"와 같은 한계 |

### 발견: 8-타일 정렬 제약 (gather/index 계열)

`index_select`(안쪽 8의 배수)·`index.Tensor`(안쪽 4의 배수)가 **맨 안쪽 차원 크기에 따라**
되고 안 되고가 갈리는 건, RNGD가 **8-wide PE(연산 레인)**로 타일링하기 때문으로 보입니다.
feature 폭이 타일에 안 맞으면(예: 5, 7) 그 op의 코드 생성을 못 합니다. 그래서 같은 코드라도
`(8,16)`은 되고 `(6,5)`는 안 됩니다 — 처음에 결과가 엇갈렸던 이유가 이거였습니다.

**실무 함의**: gather/index/embedding-lookup·`constant_pad_nd`를 쓰는 임의의 모델을 그대로
올리면 feature 폭이 8(또는 4)의 배수가 아닐 때 컴파일이 막힐 수 있습니다. LLM 본체가 도는 건
prebuilt 아티팩트가 이런 부분을 다른 방식(전용 커널/패딩)으로 처리하기 때문이지, 이 op들이
범용으로 다 도는 게 아닙니다.

---

## 5. 재현 / 메모

- 목록 뽑기: `TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c "from furiosa.torch.db import SUPPORTED_ATEN_OPS as S; print(len(S))"` → 97
- 위 §2 레시피로 op별 최소 그래프를 컴파일·실행하면 그대로 재현됩니다.
- **실행 스크립트(재현 가능)**: `info/op_verify/` — `verify_round1_all97.py`(97개 전수),
  `verify_round2_embedded.py`, `verify_round3_harden.py`, `reconcile.py`, `precision_probe.py`,
  `shape_sweep.py`. 설명은 `info/op_verify/README.md`.
- **발표자료**: `ppt/RNGD_Op_Support.pptx` / `.pdf` (6장, 빌드 스크립트 `ppt/build_op_support.js`).
- 컴파일은 host(CPU) AOT라 NPU 없이도 되고, 실행만 `rngd:N`이 필요합니다(LLM이 점유 안 한
  카드 사용). 관련: `[info/README_vision_compile.md]`(비전 모델 컴파일), `[info/README_build.md]`(빌드).
