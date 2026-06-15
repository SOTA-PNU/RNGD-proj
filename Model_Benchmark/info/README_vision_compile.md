# RNGD에서 CNN·비전 모델 컴파일 — 직접 검증 결과

LLM이 아닌 비전/CNN 모델(MobileNet·ResNet·EfficientNet·YOLO 등)을 RNGD NPU로
컴파일할 수 있는지 **이 머신에서 직접 돌려 확인한** 기록입니다. 문서상으로는
Furiosa SDK 2024.2.0 릴리스가 MobileNetV1/V2·ResNet50/152·EfficientNet·YOLOv8m 등을
지원한다고 되어 있지만, 지금 깔려 있는 SDK(2026.2.0)에서 실제로 되는지는 별개라서
하나씩 컴파일해 봤습니다.

검증일: 2026-06-08 · 출처 경로는 모두 `~/furiosa/lib/python3.12/site-packages/` 기준(SDK 2026.2.0).

---

## 한 줄 결론

**컴파일·실행은 됩니다. 그런데 학습 가중치에서 결과가 틀립니다 — 이게 제일 중요합니다.**
MobileNetV1·MobileNetV2·EfficientNet-B0 셋 다 `furiosa.torch`로 컴파일되고 RNGD NPU에서
end-to-end로 돌긴 합니다(EDF를 파일로 저장해 재실행하는 것까지 됨). 그런데 **실제로 쓸 수
있느냐는 별개입니다**: 처음엔 랜덤 가중치로 검증해 "NPU=CPU top-1 일치"로 봤는데,

> ⚠️ **그 검증이 착시였습니다(2026-06-11 정정).** 랜덤(미학습) 가중치 네트워크는 입력이
> 무엇이든 출력이 거의 변하지 않는 **퇴화 상태**라 top-1이 trivially 맞았던 것입니다.
> **학습된(ImageNet) 가중치 + 실제 사진**으로 돌리면 NPU 출력이 CPU와 **상대오차 ~100%로
> 발산**해 모든 이미지를 엉뚱한 한 클래스("window screen")로 오분류합니다(CPU는 정답).
> 원인은 furiosa.torch의 **감소정밀도 로워링**이 학습 가중치(heavy-tailed)를 무너뜨리는
> 것이고, **Python API로 끌 방법이 없습니다.** 자세히는 아래 "한계 ③ — 정확도" 절.

즉 **현재 이 경로는 "컴파일/실행 파이프라인 입증"까지는 되지만, 실제 학습 모델 추론용으로는
정확도가 안 나옵니다.** 그 외에 **중간 풀링 레이어(MaxPool2d·AvgPool2d)가 있는 모델
(예: ResNet 계열)은 컴파일 자체가 막히고**, 경로도 문서와 다릅니다 — 아래에서 설명합니다.

---

## 실측 결과 매트릭스

`furiosa.torch` 경로(아래 3절)로 직접 컴파일·실행한 결과입니다.

"NPU 실행"과 "수치 정확도"는 **다른 축**이라 컬럼을 나눴습니다. 실행(end-to-end로 돈다)은
되는데, 학습 가중치 정확도(CPU와 같은 답)는 안 나옵니다.

| 모델 | 출처 | 컴파일 | NPU 실행 | 학습가중치 정확도 |
|---|---|:--:|:--:|---|
| **MobileNetV2** | torchvision | ✅ OK (1038 노드, 313초) | ✅ rngd, 6.7ms | ❌ **틀림** — 실제이미지 rel ~100%, 전부 "window screen" 오분류 (CPU는 정답) |
| **MobileNetV1** | timm `mobilenetv1_100` | ✅ OK (544 노드, 171초) | ✅ rngd, 6.5ms | ❓ 미측정(학습본 timm). 랜덤가중치만 일치(=퇴화 검증) |
| **EfficientNet-B0** | torchvision | ✅ OK (1122 노드, 608초) | ✅ rngd, 7.3ms | ❌ 같은 감소정밀도 문제로 추정(미측정). 랜덤가중치만 일치 |
| **ResNet50** | torchvision | ❌ FAIL (564초 후) | — | — `unsupported EDF node: Cpu(...)` — **stem MaxPool 제거해도 실패**(아래) |
| **ResNet152** | torchvision | ❌ FAIL (3152노드, 1629초) | — | — ResNet50과 동일(`unsupported EDF node: Cpu`) |
| **YOLOv8m** | ultralytics | ⚠️ 미확정 | — | — `forward`·`torch.export`는 OK. compile은 serve 보호 위해 중단 — 구조상 SPPF `MaxPool2d`+conv 헤드(최종 matmul 없음)로 막힐 것 |

- "NPU 실행"의 지연(ms)은 학습 가중치 EDF로 실측한 값입니다(컴파일은 되고 빠르게 돌긴 함).
- 랜덤 가중치 검증의 "max 오차 1e-11, top-1 일치"는 사실이지만 **퇴화 네트워크 착시**라
  정확도 입증이 아닙니다(아래 "한계 ③" 참고). MobileNetV2만 학습 가중치로 직접 측정해
  발산을 확인했고, 나머지 둘은 같은 컴파일 경로라 동일 문제로 봅니다(직접 측정은 안 함).

- "노드" = `torch.export` + 분해(decomposition) 후 그래프의 call_function 수.
- 컴파일 시간은 host CPU(128코어) AOT 컴파일 기준이고 모델/병렬도에 따라 크게 변합니다.

> 참고: 컴파일은 host(CPU·RAM)에서 일어나는 AOT 과정이라 NPU가 놀고 있어도 됩니다.
> 실제 추론만 NPU가 필요합니다. 이 점은 LLM 빌드와 같습니다(`info/README_build.md`).

---

## 경로가 3개인데, 되는 건 하나뿐입니다

비전 모델을 RNGD로 올리는 길은 겉보기에 셋이지만 실제로 쓸 수 있는 건 마지막 하나입니다.

### (1) `furiosa.models` 의 비전 모델 — 지금 SDK엔 없습니다

예전 Warboy 시절 `furiosa-models`는 ResNet50·SSD-MobileNet·YOLO 같은 양자화된 비전 모델을
바로 제공했습니다. 그런데 **2026.2.0의 `furiosa.models`는 LLM 전용으로 바뀌었습니다.**

- `furiosa.models`의 공개 심볼은 전부 언어모델입니다: `LlamaForCausalLM`, `Qwen2/3...`,
  `Exaone4...`, `Mistral...`, `Phi3...`, `GptOss...`, `Qwen3VL...`.
- **`furiosa/models/vision/__init__.py`는 0바이트(빈 파일)**, `vision/architecture/`도
  `__init__.py`만 있고 모델 코드가 없습니다. 즉 비전 서브패키지는 **껍데기(stub)**입니다.
- `furiosa/models/core/quantization/`도 LLM용 FP8·MXFP4(Linear 레이어 대상)만 있고
  CNN용 INT8 calibration/PTQ는 없습니다.

→ **문서가 말하는 "2024.2.0 비전 모델 라이브러리"는 현재 설치본에 들어 있지 않습니다.**

### (2) `furiosa-compiler` CLI — ONNX는 잘 읽힙니다. 컴파일 백엔드가 이 머신에서 깨져 있습니다

> **정정(2026-06-09).** 처음엔 "이 CLI가 표준 ONNX를 못 읽는다(prost가 ModelProto에서
> EOF로 죽는다)"고 적었는데, strace로 프로세스를 직접 따라가 보니 **그게 아니었습니다.**
> ONNX는 정상적으로 파싱되고, 진짜 문제는 그 뒤의 컴파일 백엔드가 이 설치본에서 동작
> 불능이라는 것입니다. 아래로 정정합니다.

`/usr/bin/furiosa-compiler`(시스템 dpkg 패키지 **v2025.3.0**)는 `--help`상 ONNX와
`dfg`·`cdfg`·`gir`·`lir`(furiosa 내부 IR)을 입력으로 받아 `renegade`(=RNGD 코드명) 타깃
EDF로 컴파일한다고 돼 있습니다. `/usr/bin/furiosa-compile`은 여기로 가는 **심볼릭 링크라
같은 바이너리**입니다.

그런데 ONNX를 넣으면 **무엇을 넣든** 이렇게 죽습니다:

```
$ furiosa-compiler resnet50.onnx --target-npu renegade -o out.edf
ERROR: io error: unexpected end of file
error: Invalid model
```

**핵심 — 이건 ONNX 탓이 아닙니다.** `strace`로 프로세스를 따라가 보면(2026-06-09 실측):

- `furiosa-compiler`(5.5MB Rust)는 ONNX를 **정상적으로 파싱**합니다. 별도 워커
  `/usr/bin/furiosa-compiler-bridge`(108MB)를 pipe로 띄워 컴파일 요청을 넘기는데, 그 요청
  바이트 안에 **Conv 노드·`kernel_shape` 속성·가중치 텐서(`W`)가 그대로 들어가 있습니다**
  (= 그래프 변환 성공). 빈 파일을 넣으면 "빈 그래프"가 정상적으로 들어갑니다.
- 진짜 실패는 그 다음 단계입니다. **백엔드 워커(bridge)가 요청을 끝까지(길이 프리픽스대로
  온전히) 받은 뒤 내부 역직렬화에서 실패**하고(바이너리에 `failed to deserialize FuriosaIR`,
  `FieldSet corrupted (this is a bug)` 문자열 존재), 자기 stderr에 `unexpected end of file`을
  찍습니다. 그래서 어떤 입력이든 **`.edf` 산출물이 단 하나도 안 만들어집니다.**
- 통제 실험으로 확정: **빈 파일(0바이트)·쓰레기 100바이트·정상 ONNX(opset 9/13/18, checker
  통과) 전부 토씨 하나 안 틀리고 같은 에러**(exit 255)였고, 없는 파일만 다른 에러
  (`No such file or directory`)를 냅니다. 즉 ONNX 내용·크기·opset과 **무관**하고, 확장자를
  `.dfg`로 바꿔도 같습니다. 결론은 **이 머신의 컴파일 백엔드 자체가 동작 불능**이라는 것입니다.

**왜 깨졌나 (유력 원인 — 단정 아님):** 버전 부정합입니다. 시스템 컴파일러는 dpkg
`furiosa-compiler **2025.3.0**`인데 드라이버·펌웨어·파이썬 SDK는 **2026.x**
(driver-rngd 2026.2.0, firmware 2026.2.1, furiosa-llm 2026.2.0)이고, `furiosa.native`
파이썬 모듈이 없어(ModuleNotFoundError) 네이티브 백엔드 설치가 불완전합니다. bridge가 요청
페이로드를 못 푸는 게 이 어긋남과 일치합니다. (다만 bridge 양쪽은 둘 다 2025.3.0이라
자기들끼리는 맞고 실패 지점이 IR 페이로드 역직렬화라, "버전 맞춰 재설치"는 **시도할
해결책**이지 입증된 원인은 아닙니다.)

→ **결론(이 경로 사용 불가)은 그대로지만 이유가 다릅니다.** "표준 ONNX가 포맷이 안 맞아서"가
아니라 **"2025.3.0 컴파일러가 2026.x SDK와 어긋나 컴파일 백엔드가 안 돈다"** 입니다.

**어느 걸 고쳐야 하나 — 펌웨어·드라이버·furiosa-llm 전부 아닙니다.** 버전을 다 찍어보면
구버전은 딱 하나, **독립 CLI인 `furiosa-compiler` .deb(2025.3.0-4)** 뿐입니다.

| 구성요소 | 버전 | |
|---|---|---|
| `furiosa-compiler` (CLI .deb, `/usr/bin/...`) | **2025.3.0-4** | ⚠️ 유일한 구버전 |
| 파이썬 SDK 컴파일러 (`furiosa.native_common.compiler.full_version()`) | **2026.2.0** | ✅ furiosa.torch가 쓰는 것 |
| furiosa-driver-rngd / firmware / furiosa-llm·torch | 2026.2.0 / 2026.2.1 / 2026.2.0 | ✅ 최신 |

즉 **파이썬 SDK 안에 이미 멀쩡한 2026.2.0 컴파일러가 내장**돼 있고(`furiosa.torch`가 그걸
씁니다), 깨진 건 옛 `.deb`로 남은 **CLI 잔재 하나**뿐입니다. CLI를 꼭 쓰려면 `apt`로
`furiosa-compiler` 패키지만 2026.x로 올리면 되지만(펌웨어/드라이버/llm 아님), **사실 그럴 필요도
없습니다** — 비전 모델을 RNGD에 올리는 **실제 동작 경로는 어차피 (3)의 `furiosa.torch`** 라서
ONNX도 CLI도 안 거칩니다.

### (3) `furiosa.torch` (torch.compile 백엔드) — 이게 실제로 되는 길입니다

`furiosa-llm`이 내부적으로 쓰는 컴파일러입니다(LLM을 RNGD에 올릴 때 쓰는 바로 그 경로 —
`furiosa_llm`의 `api.py`·`parallelize/trace.py`·`converter.py` 등이 `torch.export`/
`from_exported`/`furiosa.torch`를 호출). PyTorch 모델을 `torch.export`로 뜬 뒤 furiosa의 패스를
거쳐 EDF로 컴파일합니다. 이걸로 MobileNet·EfficientNet이 컴파일됐습니다.

호출 체인(실측): `CompileModule.from_exported(ep)`(`furiosa/torch/custom_ops/edf.py:465`)
→ `compiler.compile(exported)`(`edf.py:479`) → `from furiosa.native_torch import compiler`
(`furiosa/torch/compiler/__init__.py:7`) → `compile(ep) -> ir.Edf`(`compiler/__init__.py:60`).
최종 EDF를 만드는 건 **네이티브 `.so` 컴파일러(`full_version()` = 2026.2.0)** 이고, 그
입력 시그니처는 `ExportedProgram`/`torch.fx GraphModule`이라 **onnx 파라미터가 없습니다**
(docstring = "Compiles an `ExportedProgram` to EDF"). 깨진 `/usr/bin/furiosa-compiler`(2025.3.0,
§2)와는 완전히 별개의, SDK 내장 컴파일러입니다.

---

## 되는 방법 (furiosa.torch, 단계별)

핵심은 **batch_norm을 먼저 직접 분해**하는 것입니다. furiosa의 기본 분해 테이블에는
batch_norm이 없어서, 안 풀면 importer가 `_native_batch_norm_legit_no_training`을 못 받습니다.

```python
import torch
import furiosa.torch                       # PrivateUse1("rngd") 백엔드 등록 (import 순서 중요)
from furiosa.torch import CompileModule
from torch._decomp import core_aten_decompositions, get_decompositions

# 1) 분해 테이블 = core-aten + batch_norm (batch_norm 은 furiosa 기본 분해에 없음)
TABLE = dict(core_aten_decompositions())
TABLE.update(get_decompositions([
    torch.ops.aten._native_batch_norm_legit_no_training,
    torch.ops.aten._native_batch_norm_legit,
    torch.ops.aten.batch_norm,
    torch.ops.aten.native_batch_norm,
]))

# 2) 모델 준비 — 추론 모드 + grad off (grad 켜진 텐서는 importer가 거부)
import torchvision.models as M
m = M.mobilenet_v2(weights=None).eval()
for p in m.parameters():
    p.requires_grad_(False)
x = torch.randn(1, 3, 224, 224)

# 3) export → 분해 → 컴파일 (여기까지는 NPU 불필요, host AOT)
with torch.no_grad():
    ep = torch.export.export(m, (x,)).run_decompositions(TABLE)
    cm = CompileModule.from_exported(ep)   # 성공하면 EDF 생성
print(type(cm.edf).__name__)               # -> Edf

# 4) NPU에서 실행 — rngd:N (이 머신은 npu0~2가 LLM serve 점유, npu3이 빔)
dev = torch.device("rngd", 3)
cm.to(dev)
with torch.no_grad():
    out = cm(x.to(dev), device=dev)
print(out.to("cpu").argmax(-1))
```

- `furiosa.torch`를 import만 해도 됩니다(아까 실패하던 건 `furiosa.torch.coverage`
  서브모듈뿐, 코어는 정상). 단 **`import torch`를 먼저** 해야 합니다.
- 실행 디바이스 이름은 `"rngd"`(= NPU), 인덱스로 카드를 고릅니다. npu0~2는 LLM이 쓰는 중이라
  비전 모델 실행은 **npu3(`rngd:3`)** 으로 했습니다.

### 내부 해부 — `CompileModule`이 비전 모델에서 EDF를 뽑아내기까지

위 코드가 SDK 안에서 실제로 무슨 일을 하는지, 소스(`~/furiosa/lib/python3.12/site-packages/`,
`furiosa-torch 2026.2.0`)를 따라가며 확인한 내용입니다. `furiosa.torch`는 이 레포의 파일이
아니라 **furiosa venv에 설치된 패키지 폴더**(`site-packages/furiosa/torch/`)입니다.

전체 흐름:

```
비전 모델(nn.Module)
   │ ① .eval() + requires_grad_(False)      ← grad 켜진 텐서는 importer가 거부
   │ ② torch.export.export(m, (x,))         → ExportedProgram (FX 그래프, ONNX 아님)
   │ ③ .run_decompositions(TABLE)           → batch_norm을 primitive로 분해 ★비전 핵심
   ▼
CompileModule.from_exported(ep)              custom_ops/edf.py:465
   │ ④ for fx_pass in PASSES: ...           furiosa 전처리 FX 패스 (export/passes.py)
   │ ⑤ EdfModule(compiler.compile(ep))      edf.py:479 — 네이티브 컴파일러(2026.2.0)가 EDF 생성
   │ ⑥ ExportedProgramWeight(ep)            가중치를 별도 모듈로 등록 (export/exported_program.py:93)
   ▼
cm.edf  →  ir.Edf 객체                       edf.py:496 — ★EDF 추출 지점
```

단계별 의미:

- **②③이 비전 컴파일의 결정적 비결**입니다. ONNX 대신 `torch.export`로 FX 그래프를 뜨고,
  furiosa 기본 분해 테이블에 없는 `batch_norm`을 직접 분해 테이블에 넣어
  `mul/add/rsqrt` 같은 기본 연산으로 풀어줍니다. 이걸 안 하면 importer가
  `_native_batch_norm_legit_no_training`에서 거부 → conv-bn-relu 모델 전부 탈락.
- **⑤가 실제 EDF가 만들어지는 곳**: `compiler.compile(exported)` =
  `furiosa.native_torch.compiler.compile(ExportedProgram) -> ir.Edf`(네이티브 `.so`,
  `full_version()` = 2026.2.0). host CPU에서 AOT로 돌고 NPU는 필요 없습니다.
- **⑥ 가중치는 EDF와 분리 보관**됩니다. `CompileModule.forward`(edf.py:534)가 실행 때마다
  `module_weight.flatten_inputs(...)`로 **파라미터/버퍼를 EDF 그래프의 런타임 입력으로
  공급**합니다. 즉 EDF는 "연산 프로그램", 가중치는 "입력"으로 나뉘어 있습니다(아래
  EDF 저장·재사용 절에서 중요해집니다).
- 컴파일 결과물은 `cm.edf` 프로퍼티(`ir.Edf` 타입)로 꺼내며, `ir.Edf.serialize() -> bytes` /
  `ir.Edf.deserialize(bytes) -> Edf`로 파일 저장·복원이 됩니다(`native_torch/ir/__init__.pyi:75,79`).

---

## 지원 / 미지원 연산자 (실측 + SDK 목록)

권위 있는 목록은 `furiosa.torch.db.SUPPORTED_ATEN_OPS`(97개)·`IMPORTABLE_ATEN_OPS`(156개)로
직접 조회했습니다(`from furiosa.torch import db`).

**지원되어 잘 도는 연산**: `convolution`, `relu`, `sigmoid`, `mul`/`add`/`sub`/`div`,
`rsqrt`/`sqrt`, `mean`(= global avg pool), `mm`/`bmm`/`addmm`(Linear), `view`/`permute`/
`transpose`/`clone`, `_to_copy`, `constant_pad_nd` 등. → 표준 conv-bn-relu 분류기 본체는 OK.

**분해해야 받는 것**: `conv2d`·`batch_norm`·`linear`은 그대로는 미지원으로 뜨지만,
분해하면 각각 `convolution`·primitive 연산·`mm/addmm`이 되어 통과합니다. **batch_norm만
직접 분해**해 주면 됩니다(위 코드).

**목록엔 supported로 떠도 실제론 막히는 것 — 풀링**:
`max_pool2d`·`max_pool2d_with_indices`는 `db` 목록상 supported=True인데, **실제 컴파일하면
실패**합니다. 아래 한계 절 참고.

---

## 한계 ① — 중간 풀링 레이어가 그래프를 쪼갭니다 (제일 중요)

`MaxPool2d`나 `AvgPool2d` 같은 **공간 다운샘플 풀링 레이어가 그래프 중간에 있으면**
컴파일이 다음 에러로 막힙니다.

```
RuntimeError: multiple internal subgraphs are not supported
```
(큰 모델에서는 `unsupported EDF node: Cpu(...)` 형태로 나타나기도 합니다 — ResNet50 실측.)

격리 실험으로 원인을 좁혔습니다(작은 모델로 1줄만 바꿔가며):

| 작은 모델 구조 | 결과 |
|---|---|
| Conv→ReLU→**GlobalAvgPool→flatten→Linear** | ✅ OK |
| Conv→ReLU→**MaxPool(3,2,1)**→GAP→flatten→Linear | ❌ multiple internal subgraphs |
| Conv→ReLU→**AvgPool(3,2,1)**→GAP→flatten→Linear | ❌ multiple internal subgraphs |
| resnet형(residual add) — 중간 풀링 **없음** | ✅ OK |
| resnet형(residual add) — 중간 **MaxPool 있음** | ❌ multiple internal subgraphs |

즉 **max/avg를 가리지 않고, 공간을 줄이는 중간 풀링 레이어 자체**가 문제입니다.
반면 **맨 끝의 global pool(→1×1, 본질은 `mean`)→flatten→Linear**는 정상입니다.

- MobileNetV1/V2·EfficientNet은 다운샘플을 **strided convolution**으로 하고 풀링 레이어가
  없어서 통과합니다.
- ResNet 계열은 stem에 `MaxPool2d(3,2,1)`가 있어 막힙니다. YOLOv8은 `torch.export`까지는
  되지만(확인됨) SPPF에 `MaxPool2d`가 있고 탐지 헤드가 conv로 끝나(최종 matmul 없음) 같은
  제약에 걸릴 것으로 보입니다(compile은 serve 보호 위해 끝까지 안 돌림).

**compiler_config로 우회가 안 됩니다.** `Config(tactic_hint=TacticHintConfig.ForVisionModel)`,
`Config(allow_unlowered_operators=True)`, 둘 다 줘 봐도 똑같이 실패했습니다
(`furiosa.native_torch.compiler.Config`).

### ResNet은 stem 풀링만 빼도 안 됩니다 (실측)

처음엔 `model.maxpool = torch.nn.Identity()`로 stem 풀링만 빼면 될 거라 봤는데, **실제로
해보니 ResNet50은 여전히 `unsupported EDF node: Cpu(...)`로 실패**했습니다. 즉 ResNet에는
풀링 말고도 컴파일을 막는 요소가 더 있습니다.

그런데 같은 구조를 작게 만든 것들은 다 통과합니다:

| 작은 모델 | 결과 |
|---|---|
| residual basic block(중간 풀링 없음) + GAP + Linear | ✅ OK |
| **bottleneck(1×1→3×3→1×1) + projection shortcut** + GAP + Linear | ✅ OK |

bottleneck·projection·residual 구조 자체는 문제가 아닙니다. 그런데 **torchvision의 진짜
ResNet50/152(4스테이지·수십 블록)** 는 막힙니다. 원인이 되는 정확한 연산은 더 좁히지
못했고(에러는 깊은 융합 커널이 CPU 노드로 떨어지는 형태), **결론적으로 ResNet 계열은 현재
SDK에서 그대로는 컴파일되지 않습니다.** 풀링 제거 같은 가벼운 수술로는 안 되고, 모델을
RNGD가 받는 연산만으로 다시 짜야 할 수준입니다.

---

## 한계 ② — 그래프가 Linear(matmul)로 끝나야 한 덩어리가 됩니다

위 표에서 보듯, conv/pool **feature map으로 끝나는**(최종 matmul이 없는) 그래프는
"multiple internal subgraphs"가 됩니다. 분류기는 마지막이 `Linear`라 괜찮지만,
탐지/세그멘테이션처럼 conv 헤드로 끝나는 모델(YOLO 등)은 이 점도 걸립니다.

---

## 한계 ③ — 학습 가중치에서 NPU 결과가 틀립니다 (정확도, 제일 치명적)

**이게 이 문서에서 제일 중요한 한계입니다.** 컴파일도 되고 NPU에서 돌기도 하지만,
**학습된 가중치로는 NPU 출력이 CPU와 크게 어긋나 실제 분류가 틀립니다.**

### 어떻게 드러났나 — 랜덤 검증의 함정

처음 검증은 `weights=None`(랜덤·미학습) 가중치로 했고 "NPU=CPU, max 오차 1e-11, top-1
일치"가 나왔습니다. 그런데 랜덤 네트워크를 뜯어보니 **입력이 무엇이든 출력이 거의 안
변하는 퇴화 상태**였습니다(실측: 실제이미지 입력 vs 랜덤노이즈 입력의 NPU 출력 차이가
6.6e-10 = 사실상 동일, argmax 항상 같은 클래스). 즉 **출력이 입력을 거의 무시**하니
NPU가 CPU를 "그대로 베끼기"가 쉬웠을 뿐, 입력 의존 계산의 정확성은 검증되지 않았습니다.

### 학습 가중치 + 실제 사진으로 측정한 결과 (2026-06-11)

학습된 ImageNet 가중치(`weights="IMAGENET1K_V1"`)로 다시 컴파일하고 진짜 사진 5장
(`/tmp/dog.jpg` 등)을 넣어 NPU와 CPU를 대조했습니다 (`classify.py`):

| 이미지 | CPU(정답) | **NPU** | NPU 상대오차 |
|---|---|---|--:|
| dog.jpg | **Samoyed (83%)** | window screen (75%) | **110%** |
| cat.jpg | **Egyptian cat (78%)** | window screen (46%) | — |
| panda.jpg | **giant panda (100%)** | window screen (33%) | **102%** |
| coffee.jpg | **cup (78%)** | window screen (74%) | — |
| banana.jpg | **banana (62%)** | window screen (27%) | — |

- **CPU는 5장 다 정확히 맞힙니다**(가중치·전처리·라벨 파이프라인 정상). 그런데 **NPU는
  5장 전부 "window screen"** 으로 붕괴합니다. 같은 EDF, 같은 입력인데 NPU만 틀립니다.
- 랜덤 입력으로 재봐도 학습 가중치에선 max 오차 0.19(rel **2.56%**)로 이미 어긋납니다
  (랜덤 가중치 땐 1e-11이었음). 즉 **입력이 아니라 "학습 가중치" 자체가 트리거**입니다.

### 원인 — 감소정밀도 로워링이 학습 가중치를 무너뜨림 (바이너리/EDF로 규명)

- **가중치는 EDF에 안 구워집니다** — fp32 런타임 입력입니다(EDF placeholder 316개 전부
  Float32). 랜덤가중치 EDF와 학습가중치 EDF는 **컴파일된 프로그램 바이너리(521568바이트)가
  바이트 단위로 동일**하고, 차이는 가중치 상수 블록 798바이트뿐. → **같은 프로그램이 같은
  감소정밀도 로워링을 런타임 fp32 가중치에 적용**합니다.
- furiosa.torch는 conv/matmul을 **감소정밀도**(BF16/TF32급, `info/README_op_support.md`의
  "matmul ~0.23%"와 같은 계열)로 내립니다. **랜덤 가우시안 가중치는 이 정밀도를 견디지만
  (1e-11), 학습 가중치는 heavy-tailed**(예: depthwise conv `features.1.conv.0.0.weight`
  kurtosis 11.6)라 감소정밀도에서 누적 오차가 폭발해 출력이 붕괴합니다.
- **고칠 Python 옵션이 없습니다.** `CompileModule.from_exported(ep, compiler_config=...)`의
  `compiler_config`(=`furiosa.native_torch.compiler.Config`)는 필드 12개 중 **정밀도/양자화
  관련이 0개**입니다. 실제로 두 레버를 시도했지만(실측):
  - `Config(implicit_type_casting=False, ...)` → **컴파일 자체 실패**(`UnsupportedOpError`)
  - `Config(tactic_hint=ForVisionModel)` → 컴파일은 되나 **결과 동일하게 발산**(rel 1.1)
  정밀도 동작은 네이티브 컴파일러(`.so`, version 3f23a71250) 내부에 고정돼 있어 Python으로
  못 끕니다.
- **LLM은 왜 멀쩡한가**: furiosa-llm 기본 빌드는 **BF16 무손실**이고, FP8/MXFP4를 쓸 때도
  캘리브레이션이 아니라 **prequantized 체크포인트의 scale + 추론시 동적 per-block 양자화**로
  범위를 잡습니다. 비전 conv 경로엔 이런 scale/범위관리 메커니즘이 없어서 그대로 무너집니다.

### 결론

**현재 SDK(2026.2.0)에서 비전 모델의 RNGD 추론은 "컴파일·실행 파이프라인 입증"까지이고,
실제 학습 모델을 정확히 추론하는 용도로는 못 씁니다.** 정밀도를 제어할 Python 손잡이가
없으니, 정확한 비전 추론이 필요하면 벤더(furiosa) 측 네이티브 컴파일러의 정밀도/스케일
처리 개선이 있어야 합니다.

### 직접 해보는 법 (classify.py)

```bash
source ~/furiosa/bin/activate
cd ~/RNGD-proj/Model_Benchmark/rngd-npu
# 학습 가중치(IMAGENET1K_V1)로 컴파일 → 실제 사진 분류, NPU vs CPU top-5 대조
python classify.py mobilenet_v2 --npu 0 --images /tmp/dog.jpg /tmp/cat.jpg /tmp/panda.jpg
```
- 학습 가중치·전처리·1000 라벨은 torchvision `Weights` enum에서 가져오므로 별도 다운로드
  불필요(가중치 `.pth`만 첫 실행 시 download.pytorch.org에서 자동 — `curl -I`는 403이지만
  실제 GET은 됩니다). 테스트 이미지는 위 5장을 GitHub raw에서 받아 `/tmp`에 뒀습니다.
- 출력에서 **CPU top-1은 정답, NPU top-1은 전부 "window screen"** 으로 나오는 걸 확인할 수
  있습니다. `--reuse-edf <file.edf>`를 붙이면 "랜덤가중치 EDF + 학습가중치" 조합도 시험해
  볼 수 있습니다(역시 오답 — EDF가 가중치에 묶임).

---

## NPU 실제 실행 (end-to-end 입증) — ⚠️ 랜덤 가중치 기준 (정확도는 한계 ③ 참고)

> 아래 수치는 **랜덤(미학습) 가중치** 기준입니다. "NPU가 CPU 계산을 그대로 재현한다"는
> 좁은 의미의 동치성은 보이지만, **학습 가중치 정확도는 위 "한계 ③"처럼 깨집니다.**

컴파일만이 아니라 **RNGD에서 실제로 추론**까지 확인했습니다 (MobileNetV2, `rngd:3`).

```
RESULT mobilenet_v2: RAN_ON_NPU rngd:3 latency=9.5ms
  max_abs_err=1.277e-11 mean_abs_err=2.935e-12
  top1_npu=514 top1_cpu=514 top1_match=True
```

- 같은 입력에 대해 **CPU 결과와 최대 오차 1.3e-11**, top-1 일치 — **단 이건 랜덤(미학습)
  가중치라서**입니다. 랜덤 네트워크는 출력이 입력을 거의 무시하는 퇴화 상태라 NPU가 CPU를
  재현하기 쉬웠을 뿐, **학습 가중치에선 같은 컴파일 경로가 ~100% 발산합니다(한계 ③).**
  즉 이 수치는 "보존"이 아니라 "퇴화 케이스에서의 동치성"으로만 읽어야 합니다.
- 실행 시 `furiosa-smi status`에서 NPU 메모리/코어가 점유됩니다.

---

## EDF를 파일로 저장해두고 재사용 실행 (run_edf.py, 2026-06-10 실측)

위 실행은 컴파일 직후 메모리의 EDF를 바로 쓴 것이라, 매번 수백 초 컴파일을 다시 해야
합니다. 그래서 **EDF를 디스크에 저장해 두고, 컴파일 없이 다시 불러와 NPU에서 실행**하는
것까지 검증했습니다. 스크립트는 `rngd-npu/run_edf.py`.

### 계획 — 무엇을 저장해야 다시 실행되나

SDK 소스(`furiosa/torch/custom_ops/edf.py`)를 보면 `CompileModule` = `EdfModule`(컴파일된
EDF) + `ExportedProgramWeight`(가중치 모듈) 두 부분이고, `forward`(edf.py:534)가 실행 때마다
`module_weight.flatten_inputs(...)`로 **가중치를 EDF의 런타임 입력으로 공급**합니다.
즉 **EDF는 "연산 프로그램"이고 가중치는 따로 들어가는 "입력"**이라, EDF 파일 하나만으론
부족하고 두 가지를 저장해야 합니다:

1. `<model>.edf` — `cm.edf.serialize()` (ir.Edf → bytes)
2. `<model>.pt` — `torch.save(m.state_dict(), ...)` (가중치)

실제 파일 크기가 이 구조를 그대로 보여줍니다 — **EDF가 가중치보다 한 자릿수 작습니다**:

| 모델 | .edf (프로그램) | .pt (가중치) |
|---|--:|--:|
| mobilenetv1 | 0.7MB | 17MB |
| mobilenet_v2 | 1.6MB | 14MB |
| efficientnet_b0 | 3.2MB | 21MB |

### 복원(실행) 쪽 구현 — 컴파일 없이 CompileModule 재조립

```python
from furiosa.torch import CompileModule
from furiosa.torch.custom_ops.edf import EdfModule
from furiosa.torch.export import ExportedProgramWeight, PASSES
from furiosa.native_torch import ir

# 1) 저장된 EDF 로드 (수 ms, 컴파일 없음)
edf = ir.Edf.deserialize(open("mobilenet_v2.edf", "rb").read())

# 2) 가중치 모듈 재구성 — 컴파일 때와 "동일한" 파이프라인으로 ep 를 다시 만들어야
#    입력 순서/이름이 EDF 와 일치한다 (export → batch_norm 분해 → PASSES)
m.load_state_dict(torch.load("mobilenet_v2.pt", weights_only=True))
ep = torch.export.export(m, (x,)).run_decompositions(DECOMP)
for fx_pass in PASSES:          # from_exported 가 내부에서 하는 것과 동일 (edf.py:473)
    ep = fx_pass(ep)
cm = CompileModule(EdfModule(edf), ExportedProgramWeight(ep))   # 재조립 끝

# 3) NPU 실행 — 이후는 보통 때와 동일
cm.to(torch.device("rngd", 0));  out = cm(x.to("rngd:0"), device=torch.device("rngd", 0))
```

- `EdfModule(edf)`는 `ir.Edf` 객체 하나로 재구성됩니다. SDK 내부 로그도 "using
  pre-compiled edf"라서 **저장된 EDF 재사용이 의도된 경로**임을 알 수 있습니다(edf.py:138).
- 가중치 재구성에 `torch.export`를 다시 돌리는 비용(~10초 안팎)은 들지만, **수백 초
  컴파일은 완전히 생략**됩니다.

### 실측 결과 — 저장된 EDF로 NPU 실행 (⚠️ 랜덤 가중치 기준)

> 아래는 **랜덤(미학습) 가중치** EDF를 저장·복원해 돌린 것이라, "오차 1e-11, top-1 일치"는
> **퇴화 네트워크 착시**입니다(한계 ③). 저장/복원 **메커니즘 자체가 동작한다**는 입증으로만
> 보세요. 학습 가중치로는 결과가 틀립니다.

이 머신(npu0~3 전부 유휴)에서 `compile`(저장) → 별도 프로세스로 `run`(복원·실행) 순서로
직접 돌린 결과입니다:

| 모델 | 컴파일(1회만) | EDF 로드 | 가중치 재구성 | NPU 실행(cold→warm) | CPU 대조(랜덤가중치) |
|---|--:|--:|--:|--:|---|
| mobilenet_v2 (rngd:0) | 270.8s | 7ms | 14.5s | 9.6ms → **6.7ms** | 오차 1.7e-11, top-1 765=765 (퇴화) |
| mobilenetv1 (rngd:1) | 128.0s | 4ms | 8.5s | 5.7ms → **6.5ms** | 오차 2.6e-4, top-1 822=822 (퇴화) |
| efficientnet_b0 (rngd:2) | 642.8s | 13ms | 13.0s | 8.4ms → **7.3ms** | 오차 4.1e-16, top-1 728=728 (퇴화) |

- 검증된 것은 **"EDF 파일 저장→컴파일 없이 복원→NPU 실행" 파이프라인이 돈다**는 것과
  지연(warm 6~7ms)뿐입니다. 재실행 비용이 "수백 초 컴파일" → "~15초(가중치 재구성)+수 ms"로
  줄어드는 것도 사실입니다.
- 단, **랜덤 가중치 EDF를 학습 가중치로 재사용하면 결과가 틀립니다**(실측): 같은
  `mobilenet_v2.edf`에 학습 가중치만 끼워 dog.jpg를 넣으면 NPU가 "window screen"(오답),
  CPU는 Samoyed(정답). EDF는 **컴파일 시점 가중치에 묶인 감소정밀도 산출물**이라, 쓸
  가중치로 매번 새로 컴파일해야 하고 — 그래도 학습 가중치는 한계 ③대로 틀립니다.
- `top-1 765=765`의 765는 랜덤 가중치라 의미 없는 클래스 번호고, "NPU와 CPU가 같은 (무의미한)
  답"이라는 뜻일 뿐입니다.

### 직접 해보는 법

```bash
source ~/furiosa/bin/activate          # torchvision/timm 은 --no-deps 로 설치돼 있어야 함
cd ~/RNGD-proj/Model_Benchmark/rngd-npu

# 1) 컴파일 + 저장 (모델당 1회, 수백 초 — host CPU AOT 라 NPU 불필요)
python run_edf.py compile mobilenet_v2     # → mobilenet_v2.edf + mobilenet_v2.pt
python run_edf.py compile mobilenetv1
python run_edf.py compile efficientnet_b0

# 2) 저장된 EDF 로 실행 (컴파일 없음, NPU 필요)
python run_edf.py run mobilenet_v2 --npu 0
python run_edf.py run mobilenetv1 --npu 1
python run_edf.py run efficientnet_b0 --npu 2
```

주의할 점 두 가지(실측):

- **여러 컴파일을 동시에 돌리면 메모리 압박으로 조용히 죽을 수 있습니다.** 3개를 동시에
  띄웠더니 1개(mobilenetv1)가 출력 없이 사망했고(에러도 OOM 로그도 없음), 단독 재실행은
  정상이었습니다. 컴파일은 하나씩 돌리는 게 안전합니다(RAM 125GB 중 LLM serve 등이
  ~63GB 점유 상태였음).
- 시작 시 `WARN ... manual_seed_all is not implemented yet`가 뜨는데 무해합니다
  (`torch.manual_seed`가 rngd 백엔드 시딩까지 건드려서 나는 경고).

---

## 검증 환경

- SDK: `furiosa-llm` / `furiosa-torch` / `furiosa-models` **2026.2.0**,
  `furiosa-compiler` **2025.3.0**, NPU 펌웨어 2026.2.1.
- 하드웨어: RNGD **4장**(npu0~3). 검증 당시 npu0~2는 LLM serve(coder7·coder14·qwen3-32b)
  점유, **npu3 유휴** → 비전 모델 실행은 npu3 사용.
- furiosa venv torch: **2.10.0+cu128** (Python 3.12).
- 모델 정의용으로 `torchvision 0.25.0`·`timm 1.0.27`·`ultralytics 8.4.61`을 furiosa venv에
  `--no-deps`로 **임시 설치**(torch 불변)했다가 **검증 후 모두 제거**해 원상복구했습니다.
  `furiosa.torch` 자체는 이들 없이 동작하며, 재현하려면 `pip install --no-deps torchvision timm`만
  다시 깔면 됩니다(아래 스크립트 참고).

---

## 재현 방법

위 "되는 방법" 코드를 그대로 쓰면 됩니다. 모델만 바꿔(`M.resnet50` 등) 컴파일 성공/실패를
확인할 수 있습니다. 풀링이 있는 모델은 위 한계대로 막히는 게 정상입니다.

검증에 쓴 분해 테이블(batch_norm 포함)과 `rngd:N` 실행 코드가 핵심이고, 나머지는 표준
PyTorch입니다.

---

## 정리 — 문서 vs 실제

| 항목 | 문서(2024.2.0) | 이 머신(2026.2.0) 실측 |
|---|---|---|
| 비전 모델 라이브러리(`furiosa.models`) | 제공 | **빠짐(빈 stub), LLM 전용** |
| ONNX → `furiosa-compiler` CLI | 지원 | **ONNX 파싱은 됨, 컴파일 백엔드가 깨짐**(컴파일러 2025.3.0 ↔ SDK 2026.x 부정합 추정; 어떤 입력이든 `.edf` 0개) |
| MobileNetV1/V2 | 지원 | 🟡 **컴파일·NPU 실행은 됨, 학습 가중치 정확도는 틀림**(V2 실측 rel ~100% 오분류; 한계 ③) |
| EfficientNet | 지원 | 🟡 **컴파일·NPU 실행은 됨, 정확도 동일 문제 추정**(랜덤가중치만 검증) |
| ResNet50/152 | 지원 | ❌ **컴파일 막힘**(`unsupported EDF node: Cpu`) — 풀링 제거로도 안 됨 |
| YOLOv8m | 지원 | ⚠️ **미확정** — export는 되나 compile 미완(구조상 SPPF MaxPool로 막힐 것) |

**핵심**: RNGD에서 풀링 없는 strided-conv 계열 CNN의 **컴파일과 NPU 실행 자체는 됩니다**.
그러나 **학습 가중치에선 감소정밀도 로워링이 결과를 무너뜨려(한계 ③) 실제 추론 정확도가
안 나오고**, 이를 끌 Python 옵션이 없습니다. 더해 **중간 풀링이 있으면 컴파일이 막힙니다**.
즉 현재 SDK(2026.2.0)에서 비전 모델 RNGD 경로는 **"파이프라인 입증" 단계**이지 실사용
단계가 아닙니다 — 정확도까지 쓰려면 벤더 측 네이티브 컴파일러의 정밀도 처리 개선이
필요합니다.
