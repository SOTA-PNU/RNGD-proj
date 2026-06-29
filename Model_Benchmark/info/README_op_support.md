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

---

## 6. dtype별 op 지원 매트릭스

요청 dtype 10종(float64/32/16·bfloat16·int64/32/16/8·uint16/32)에 대해 op가 RNGD로
컴파일되는지 직접 측정했습니다(2026-06-09). 각 op는 `op(x+x)` 실연산 그래프로 감싸 단독그래프
degeneracy를 제거했고, op마다 그 op를 torch eager가 받는 dtype에만 시도했습니다.

> ⚠️ 이 측정은 **컴파일 게이트**(`CompileModule.from_exported`) 기준입니다. 측정 시점에
> furiosa-llm serve 2개가 4장 전체를 점유(메모리 ~45GiB/장)해 NPU 실행이 EBUSY였습니다.
> 단 dtype 가용성은 §1에서 보듯 컴파일 단계에서 결정되고(미지원 dtype은 `from_exported`에서
> `UnsupportedOpError`로 차단), §3에서 "컴파일 OK ≈ 실행 OK(정밀도 제외)"를 이미 확인했습니다.

### 6-1. dtype 요약 (등급)

| dtype | 등급 | 가용 op | 막히는 대표 op |
|---|---|---|---|
| **float32** | ✅ 완전 | 36 / 37 | cumsum |
| **bfloat16** | ✅ 완전 | 36 / 37 | cumsum (float32와 동일) |
| **int32** | ✅ 정수 강 | 34 / 36 | pow · conv2d |
| **int64** | ✅ 정수 강 | 33 / 36 | pow · conv2d · full_like |
| **int8** | 🔶 중간 | 28 / 36 | relu · max · argmax · where · cumsum · slice · pow |
| **float16** | 🔶 부분 | 20 / 37 | 비교(eq·lt) · neg · clamp · log · rsqrt · pow · relu · conv2d · mean · max · argmax · where · slice |
| **int16** | 🔶 약함 | 18 / 36 | 비교 · 단항 다수(abs·neg·log·sqrt·clamp) · relu · conv2d · max · cumsum · bitwise · where |
| **float64** | ❌ 미지원 | 0 | 전부 (dtype 게이트에서 차단) |
| **uint16** | ❌ 미지원 | 0 | 전부 |
| **uint32** | ❌ 미지원 | 0 | 전부 |

### 6-2. 핵심 발견

- **float64·uint16·uint32는 dtype 자체가 EDF 미지원**입니다. 어떤 연산도 컴파일되지 않고,
  `clone`·`full_like`처럼 상수·메타만 쓰는 연산이 상수 폴딩으로 우연히 통과하는 정도뿐입니다.
- **bfloat16 = float32 동급**(둘 다 36/37). 반면 **float16은 부분 지원**(20/37)으로,
  비교·일부 단항(neg·clamp·log·rsqrt)·pow·relu·conv2d·reduction(mean·max·argmax)·where에서
  막힙니다. → NPU가 bf16을 1급 부동소수로 다루고 fp16은 그렇지 않습니다.
- **정수는 int32(34)·int64(33)가 가장 강하고, int8(28) → int16(18) 순**입니다.
  int8은 `conv2d`까지 컴파일되지만(저정밀 양자화 경로), int16은 정수 중 가장 약합니다.
- **`cumsum`은 int64·int32에서만** 컴파일됩니다(float 전부·int16·int8 불가). §4의 "정수만"을
  정밀화한 결과입니다.
- **`pow`는 float32·bf16만**, **`conv2d`는 float32·bf16·int8만** 컴파일됩니다.
- **`logical_and`은 입력이 bool(또는 `x!=0`)이어야** 컴파일됩니다(numeric 직접 입력은 막힘).
- `slice` 행은 출력을 materialize해야 정확히 측정됩니다(뷰가 그래프 끝이면 단독그래프
  degeneracy로 false-negative). 보정 후 f32·bf16·i64·i32·i16=O, f16·i8=✗.

### 6-3. 전체 매트릭스

범례: **O** = 컴파일 성공 · **✗** = unsup(`UnsupportedOpError`) · **·** = **미정의**(torch eager가
그 dtype에서 op 자체를 정의하지 않음 — PPT 히트맵엔 `—`로 표기). `op(x+x)` embed 기준, `slice`·`logical_and`은 6-2의 보정값.

| op | float64 | float32 | float16 | bfloat16 | int64 | int32 | int16 | int8 | uint16 | uint32 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `add` | ✗ | O | O | O | O | O | O | O | · | · |
| `sub` | ✗ | O | O | O | O | O | O | O | · | · |
| `mul` | ✗ | O | O | O | O | O | O | O | · | · |
| `div` | ✗ | O | O | O | O | O | O | O | · | · |
| `pow` | ✗ | O | ✗ | O | ✗ | ✗ | ✗ | ✗ | · | · |
| `clamp` | ✗ | O | ✗ | O | O | O | ✗ | O | · | · |
| `abs` | ✗ | O | O | O | O | O | ✗ | O | · | · |
| `neg` | ✗ | O | ✗ | O | O | O | ✗ | O | · | · |
| `exp` | ✗ | O | O | O | O | O | O | O | · | · |
| `log` | ✗ | O | ✗ | O | O | O | ✗ | O | · | · |
| `sqrt` | ✗ | O | O | O | O | O | ✗ | O | · | · |
| `rsqrt` | ✗ | O | ✗ | O | O | O | ✗ | O | · | · |
| `sin` | ✗ | O | O | O | O | O | O | O | · | · |
| `erf` | ✗ | O | O | O | O | O | O | O | · | · |
| `sigmoid` | ✗ | O | O | O | O | O | O | O | · | · |
| `tanh` | ✗ | O | O | O | O | O | O | O | · | · |
| `softmax` | ✗ | O | O | O | · | · | · | · | · | · |
| `relu` | ✗ | O | ✗ | O | O | O | ✗ | ✗ | · | · |
| `mm` | ✗ | O | O | O | O | O | O | O | · | · |
| `conv2d` | ✗ | O | ✗ | O | ✗ | ✗ | ✗ | O | · | · |
| `eq` | ✗ | O | ✗ | O | O | O | ✗ | O | · | · |
| `lt` | ✗ | O | ✗ | O | O | O | ✗ | O | · | · |
| `maximum` | ✗ | O | O | O | O | O | O | O | · | · |
| `logical_and` | ✗ | O | ✗ | O | O | O | ✗ | O | ✗ | ✗ |
| `bitwise_and` | · | · | · | · | O | O | ✗ | ✗ | · | · |
| `sum` | ✗ | O | O | O | O | O | O | O | · | · |
| `mean` | ✗ | O | ✗ | O | · | · | · | · | · | · |
| `max.dim` | ✗ | O | ✗ | O | O | O | ✗ | ✗ | · | · |
| `argmax` | ✗ | O | ✗ | O | O | O | ✗ | ✗ | · | · |
| `cumsum` | ✗ | ✗ | ✗ | ✗ | O | O | ✗ | ✗ | · | · |
| `view` | ✗ | O | O | O | O | O | O | O | · | · |
| `cat` | ✗ | O | O | O | O | O | O | O | · | · |
| `permute` | ✗ | O | O | O | O | O | O | O | · | · |
| `slice` | ✗ | O | ✗ | O | O | O | O | ✗ | · | · |
| `where` | ✗ | O | ✗ | O | O | O | ✗ | ✗ | · | · |
| `clone` | ✗ | O | O | O | O | O | O | O | · | · |
| `to_float32` | ✗ | O | O | O | O | O | O | O | · | · |
| `full_like` | O | O | ✗ | O | ✗ | O | ✗ | O | · | · |

스크립트: `info/op_verify/dtype_matrix.py`(embed판) · `dtype_matrix_standalone.py`(단독판).
`bitwise_and`/`mean`/`softmax`의 `·`는 torch가 그 dtype에서 op 자체를 거부하는 경우입니다
(예: float에 bitwise, 정수에 mean/softmax).

---

## 7. op 실행 위치 분류 (npu / host / compile_fail / crash / trace_unsupported)

SUPPORTED 97개를 단순 가능/불가가 아니라 **어디서 어떻게 처리되는지**로 분류했습니다
(2026-06-09, NPU 4장 유휴 상태에서 측정). 각 op를 **별도 subprocess로 격리 실행**해
크래시를 감지하고, 두 경로로 측정했습니다.

**측정 경로 (2가지):**
- **AOT 빌드 경로** — `CompileModule.from_exported`. furiosa-llm build/serve가 쓰는 그 길.
  op가 EDF로 컴파일된 뒤 NPU 실행까지 되는가.
- **eager 런타임 경로** — `RngdTensor`(=`x.to("rngd")`) dispatch + `RNGDCoverageTrace`.
  furiosa가 op별로 `run_on_rngd`(NPU 실행) / `run_on_cpu`(CPU fallback)를 coverage에 기록합니다
  (`coverage.py`의 `log` / `log_cpu_fallback`, `backend/eager.py:112·119`). 미지원 op는
  `UnsupportedOpError` → `run_by_cpu`로 **CPU fallback**됩니다.

**분류 정의:**

| 분류 | 정의 |
|---|---|
| **npu** | AOT 컴파일+NPU 실행 성공 (eager coverage `run_on_rngd`). |
| **host** | EDF 미지원 → eager 경로에서 **CPU로 fallback 실행**(`run_on_cpu`). AOT 경로에선 `UnsupportedOpError`. |
| **compile_fail** | AOT 컴파일 실패 + 실행도 안 됨 (아래처럼 단독 op 그래프 한정). |
| **trace_unsupported** | `torch.export` 트레이스 단계 실패(FX 그래프 생성 불가). |
| **crash** | 측정 중 프로세스 비정상 종료(네이티브 abort/segfault/timeout). |

**결과:**

| 분류 | 단독 op 그래프 | 실연산 그래프 |
|---|:-:|:-:|
| **npu** | 82 | **89** |
| **host** | 8 | 8 |
| **compile_fail** | 7 | 0 |
| **trace_unsupported** | 0 | 0 |
| **crash** | 0 | 0 |

- **host 8개** (EDF 미지원 → CPU fallback): `isnan` · `cumsum` · `max_pool2d_with_indices` ·
  `slice_scatter` · `index.Tensor` · `index_select` · `gather` · `constant_pad_nd`. §4의 그 8개와 동일.
  eager 런타임에선 CPU에서 돌아 **결과는 나오지만**(host), AOT 빌드(serve)에선 `UnsupportedOpError`로
  막힙니다(compile_fail). → **host와 compile_fail은 같은 op의 두 경로 결과**입니다.
- **compile_fail 7개는 단독 op 그래프에서만** 나타납니다 (`expand`·`expand_copy`·`slice.Tensor`·
  `split_with_sizes`·`split_with_sizes_copy`·`copy`·`copy_`). `op(x+x)+후속연산` 실연산 그래프에
  넣으면 7개 모두 npu로 컴파일·실행됩니다 — §4의 단독그래프 degeneracy.
- **trace_unsupported 0**: 97개 모두 `torch.export` 트레이스 가능.
- **crash 0**: 네이티브 컴파일러 panic(`rust 'not yet implemented' at
  npu-compiler/.../vector_ops_compiler/memory_usage.rs`)이 발생해도 furiosa가 catch해 CPU
  fallback하므로 프로세스 abort가 일어나지 않았습니다(panic은 stderr로 관측되나 프로세스 생존).

**실무 함의:** 모델을 furiosa-llm `serve`(=AOT)로 올릴 때 그래프에 (분해 후) host 8개가 남으면
빌드가 막힙니다. 반면 `furiosa.torch` eager(torch.compile) 런타임에선 그 8개가 CPU fallback으로
돌아 결과는 나옵니다(느려짐).

스크립트: `info/op_verify/classify_worker.py`(op 1개 측정, AOT+eager 2경로) ·
`classify_runner.py`(97개 subprocess 격리 실행 + 5분류 집계).

### 7-1. op × dtype 실행위치 히트맵 (SUPPORTED 97개 전수)

§6(dtype)과 §7(실행위치)을 교차해 **SUPPORTED 97개 × 10 dtype = 970칸**을 분류했습니다.
각 op는 별도 subprocess로 격리(crash 감지), AOT(`from_exported`+NPU실행)+eager(coverage) 2경로,
NPU 4장 유휴. 입력은 데이터 텐서만 target dtype으로 캐스팅(인덱스=int64·bool은 역할 유지).
**단독 op 그래프 기준**입니다. 그래프는 `ppt/RNGD_Op_Support.pptx` 슬라이드 10 (Y=op, X=dtype, 분류별 색).

**dtype별 집계 (npu / host / compile_fail / na, 970칸):**

| dtype | npu | host | compile_fail | na |
|---|:-:|:-:|:-:|:-:|
| **float32** | 76 | 8 | 7 | 6 |
| **bfloat16** | 76 | 8 | 7 | 6 |
| **int32** | 72 | 11 | 7 | 7 |
| **int64** | 63 | 14 | 14 | 6 |
| **int8** | 54 | 29 | 7 | 7 |
| **int16** | 45 | 38 | 7 | 7 |
| **float16** | 43 | 41 | 7 | 6 |
| **float64** | 8 | 63 | 20 | 6 |
| **uint16** | 4 | 0 | 63 | 30 |
| **uint32** | 4 | 0 | 63 | 30 |

(trace_unsupported·crash = 970칸 중 0)

- **float64**: 거의 전부 **host** — EDF 미지원이나 eager 경로에서 CPU fallback으로 실행됨.
- **float32 · bfloat16**: 거의 전부 **npu** (동급). **uint16 · uint32**: 거의 전부 미지원(na 또는 compile_fail).
- **정수**: int32 ≳ int64 > int8 > int16. **float16**은 절반쯤 host.
- **`cumsum`은 int64·int32만 npu**, `pow`·`clamp`는 float32/bf16/정수만, `bitwise_*`는 int64/int32만 npu(int16/8은 host).
- **C(compile_fail)** 칸의 `expand`·`expand_copy`·`slice.Tensor`·`split_with_sizes(_copy)`·`copy`·`copy_`
  (및 `view`·`squeeze`·`permute` 류의 float64/int64)은 **단독 op 그래프 degeneracy**입니다 —
  `op(x+x)+후속연산` 실연산 그래프에 넣으면 npu(§7). 즉 실연산 기준 compile_fail은 거의 0.

> 보조: degeneracy를 제거한 38개 대표 op embed판 매트릭스도 측정했습니다
> (`info/op_verify/dtype_class_matrix.py`). 거기선 compile_fail 0, float64=거의 host, uint=na로 동일 결론.

스크립트: `info/op_verify/dtype_full_worker.py` · `dtype_full_runner.py` (97 op × 10 dtype, op별 subprocess 격리).

---

## 8. op 기능 카테고리 분류 (14종)

SUPPORTED 97개를 기능 카테고리로 분류했습니다. 카테고리 체계는 **furiosa 컴파일러
`native_runtime.so`의 IR op enum**에서 확인했습니다(`strings`로 추출): `activation` · `Conv` ·
`index` · `matmul` · `meta` · `norm` · `reduction` · `resize` · `shape` · `pad` · `Cast` · `Copy` ·
`select`, 그리고 그래프 노드 `Unary` / `Binary` / `SymExpr::Ternary` / `Reduce` / `CumulativeSum` /
`AttentionKernel`(=attn). 이를 op 의미와 합쳐 14개 카테고리로 매핑했습니다(합계 = 97).

- **`mn` = min/max** (`maximum`·`minimum`). native에 `AbsMinF/AbsMaxF`·`minmax` 별도 존재.
- **`attn`·`norm`·`resize`는 SUPPORTED 97개에 해당 op가 없습니다** (attention은 전용 커널,
  batch/layer norm은 분해되어 primitive로, interpolate/upsample은 미포함).

| 카테고리 | 수 | op | dtype 실행 경향 (§7-1, 970칸 실측) |
|---|:-:|---|---|
| **unary** | 14 | abs·cos·erf·exp·log·neg·reciprocal·rsqrt·sin·sqrt·isnan·clamp·pow.Tensor_Scalar·logical_not | f32/bf16 npu · fp16·int16 일부 host · `isnan` 전부 host |
| **activation** | 6 | relu·leaky_relu·sigmoid·tanh·_softmax·_log_softmax | f32/bf16 npu · fp16서 relu host |
| **binary** | 28 | add·sub·mul·div(.Scalar/.Tensor) · eq·ne·lt·le·gt·ge · logical_and/xor · bitwise_and/or/xor | f32/bf16 npu · bitwise=int전용 · 비교 .Scalar는 int8/16 host |
| **mn** | 2 | maximum·minimum | f32/bf16/int npu · float64 host |
| **matmul** | 2 | mm·bmm | f32/bf16/int npu(감소정밀도 ~0.23%) · float64 host |
| **conv** | 1 | convolution | f32/bf16/int8 npu · fp16·int16·int32/64 host |
| **reduction** | 9 | sum.dim_IntList·mean.dim·amax·max.dim·argmax·any.dim·var_mean.correction·topk·cumsum | f32/bf16 npu · `cumsum`=int32/64만 · max/argmax int16/8 host |
| **pool** | 3 | avg_pool2d·max_pool2d_with_indices·_adaptive_avg_pool2d | 대부분 host (NPU 직접 거의 안 됨) · max_pool 전 dtype host |
| **shape** | 18 | view(_copy)·permute(_copy)·transpose_copy·t_copy·squeeze(.dim/.dims/_copy)·unsqueeze(_copy)·expand(_copy)·cat·slice.Tensor·slice_scatter·split_with_sizes(_copy) | f32/int npu · expand/slice/split/copy_ 단독 C → 실연산 npu |
| **index** | 5 | index.Tensor·index_put·index_select·gather·scatter.src | index_put·scatter npu · index/index_select/gather 전 dtype host |
| **ternary** | 1 | where.self | f32/bf16/int npu · int16/8 host |
| **creation** | 3 | full·full_like·fill.Scalar | f32/int32 npu · fp16/int16 일부 host |
| **pad** | 1 | constant_pad_nd | 전 dtype host (NPU EDF 미지원) |
| **meta** | 4 | clone·copy·copy_·_to_copy | clone·_to_copy npu · copy·copy_ 단독 C → 실연산 npu |

그래프: `ppt/RNGD_Op_Support.pptx` 슬라이드 10(카테고리별로 묶고 **맨 오른쪽 열에 카테고리**를 표시한
히트맵) · 슬라이드 11(분류 표). 매핑 데이터: `info/op_verify/op_categories.json`.
