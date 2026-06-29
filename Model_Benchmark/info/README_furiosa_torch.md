# furiosa-torch 분석 + serve 가능 여부

작성 2026-06-17. SDK 2026.2.0 기준. 모든 사실은 설치된 모듈(`furiosa/torch/`)·native 라이브러리·
이 프로젝트 실측(`qwen3-next-proj/qcn/`)으로 검증했습니다.

요지를 먼저 적습니다.
- **furiosa-torch는 "PyTorch를 NPU에서 돌리는 장치 백엔드"입니다.** 모델을 서빙용 파일로 굽는
  도구가 아니라, 파이썬 프로세스 안에서 torch 연산을 NPU로 실행시키는 계층입니다.
- **furiosa-torch로 컴파일한 결과는 furiosa-llm `serve`로 못 띄웁니다.** 포맷이 다릅니다
  (furiosa-torch는 a5 `ir.Edf`, furiosa-llm 아티팩트는 a6 `CompiledGraph`+메타데이터). serve
  하려면 **별도 방법**(커스텀 서버, 또는 host 추론 루프 + 어댑터)이 필요하고, 이 프로젝트의
  `qcn/serve_q25.py`·`qcn/serve.py`가 바로 그 실례입니다.

관련 문서: [README_qwen3_coder_next.md](README_qwen3_coder_next.md)(furiosa-torch로 만든 host
루프), [README_op_support.md](README_op_support.md)(지원 연산), [README_vision_compile.md](README_vision_compile.md)
(비전 모델 컴파일), [ALL_about_build_serve.md](ALL_about_build_serve.md)(furiosa-llm build/serve 내부).

---

## 1. furiosa-torch가 무엇인가

`furiosa-torch`는 PyTorch에 **RNGD NPU를 장치로 붙여 주는** 패키지입니다. 두 가지를 합칩니다.

1. **`rngd` torch 장치**: `tensor.to("rngd:0")` 처럼 텐서를 NPU로 보낼 수 있게 등록합니다
   (`furiosa/torch/__init__.py` 가 `rename_privateuse1_backend("rngd")` + `_register_device_module`
   로 등록; `native_device.py` 는 디바이스 함수 재노출). 그래서 import 순서가
   중요합니다. `import torch` 를 먼저 하고 `import furiosa.torch` 를 해야 백엔드가 정상 등록됩니다.
2. **`torch.compile` 백엔드**: `torch.compile(model, backend=furiosa.torch.backend)` 로 모델의
   계산 그래프(FX)를 NPU가 실행하는 형태로 컴파일합니다(`furiosa/torch/backend/`).

비유하면, GPU에서 `model.cuda()` + `torch.compile` 을 쓰듯이, NPU에서 `.to("rngd")` +
`torch.compile(backend=furiosa.torch.backend)` 를 쓰는 것입니다. 차이는 NPU가 만능이 아니라
정해진 연산만 빠르게 한다는 점입니다(아래 4절).

핵심은 **"실행"이지 "저장·배포"가 아니라는 것**입니다. furiosa-torch는 모델을 파이썬 프로세스
안에서 NPU로 돌립니다. 이 결과를 따로 묶어 다른 서버가 읽는 "도시락(아티팩트)"으로 만드는
기능은 없습니다(그건 furiosa-llm build의 일입니다, 5절).

---

## 2. 구성과 API (실측)

`import torch; import furiosa.torch as ft` 후 노출되는 핵심 요소들입니다.

| 요소 | 역할 |
|---|---|
| `ft.backend` (`backend.torch_compile._Backend`) | `torch.compile(m, backend=ft.backend)` 의 백엔드 |
| `ft.RngdTensor` | NPU에 올라간 텐서 타입 |
| `ft.CompilableModule` / `CompileModule` / `DfgModule` / `LowerModule` / `EdfModule` | 컴파일 단계별 모듈 래퍼(아래 3절) |
| `ft.TacticKernelModule(dsl)` | 손으로 작성한 저수준 NPU 커널 DSL 텍스트(`#naive_yaml`/`#tactic_kernel_dsl`, `Dfg.parse`로 파싱)를 모듈로 |
| `ft.SUPPORTED_ATEN_OPS`(97) / `IMPORTABLE_ATEN_OPS`(156) / `SKIPPED_ATEN_OPS`(82) | NPU가 지원·임포트·건너뛰는 ATen 연산 목록 |
| `ft.DECOMP_RULES_DB` / `STD_DECOMPOSITIONS` / `register_decomposition` | 복잡한 연산을 지원 연산으로 쪼개는 규칙 |
| `ft.export` (패키지; `ExportedProgram` 은 `export.exported_program` 서브모듈에) | **torch.export(PT2)** 통합 + NPU용 `PASSES`(서빙 아티팩트 저장 기능이 아니라 in-process 컴파일용) |
| `ft.custom_ops` (`dfg.py`, `edf.py`, `cat_stack.py` 등) | 커스텀 연산·DFG/EDF 처리·CPU 폴백 경로(`dfg._dfg_inner`) |
| `ft.db` | 연산 지원 데이터베이스 |
| `ft.profiler` / `ft.coverage` / `ft.debug` | 프로파일·연산 커버리지·디버그 |

native 쪽은 `furiosa/native_torch.cpython-...so`(105MB) 한 덩어리이고, Rust 크레이트
`furiosa-torch`(torch 그래프 → EDF IR lowering) + `npu-executor`(실행) + `furiosa-hal2`(하드웨어
추상화)를 담습니다(`ALL_about_build_serve.md` Part 4). 이 `.so`는 PyTorch C++(`libc10.so`)에
링크돼 있어서 **빌드/실행 환경에서만** 임포트됩니다.

`export`가 "서빙 export"가 아니라 **torch.export 통합**이라는 점을 분명히 합니다. 안에 든 건
`ExportedProgram`·`ExportedProgramWeight`·`PASSES`(NPU용 패스)뿐이고, save/serve/artifact 류
함수는 없습니다. 즉 PT2 `torch.export` 로 뽑은 프로그램을 NPU로 컴파일·실행하는 용도이지,
furiosa-llm이 읽는 배포 파일을 만드는 게 아닙니다.

---

## 3. 컴파일 파이프라인 (torch → NPU)

`torch.compile(model, backend=furiosa.torch.backend)` 가 호출되면 모델이 먼저 `ExportedProgram`
(torch.export 형태)으로 추적되고, 그걸 NPU 실행 모듈로 컴파일합니다. **핵심 경로는 곧장 EDF로
갑니다**(실측, backend/torch_compile.py:126 → custom_ops/edf.py:464-481).

```
 PyTorch 모델
   │  torch.compile 이 ExportedProgram(EP)으로 추적
   ▼
 CompileModule.from_exported(EP)
   │   ① PASSES 적용(furiosa.torch.export.PASSES: NPU용 그래프 변환)
   │   ② 네이티브 compiler.compile(EP) 한 방에 호출 → ir.Edf (a5)
   ▼
 EdfModule  (NPU가 실행하는 EDF, a5 ir.Edf 포맷)
   │
   ▼
 NPU에서 실행 (RngdTensor 입출력; dfg._dfg_inner 는 CPU 폴백 경로라 호출 0이면 순수 NPU)
```

`*Module` 들은 **순차 체인이 아니라 같은 EP에 대한 병렬 진입점**입니다(전부 `from_exported`).
용도가 다릅니다(실측, custom_ops/edf.py·dfg.py, compiler/__init__.py).
- **`CompileModule.from_exported`**: PASSES → `compiler.compile` → `ir.Edf` → `EdfModule`. NPU
  실행용. **torch.compile 백엔드가 쓰는 경로**입니다.
- **`LowerModule.from_exported`**: PASSES → `compiler.lower(EP)` → `ir.Dfg` → `DfgModule`. DFG까지만
  낮춰 들여다보는 경로(EDF 전 단계 검사용).
- **`CompilableModule`**: 원본 EP를 그대로 CPU에서 실행하는 래퍼(레퍼런스 비교용).
- 즉 dfg→primitive→kernelized→…→edf 같은 세부 12단계는 **`compiler.compile` 내부(네이티브)** 에서
  일어나고, 파이썬 쪽 `*Module` 은 그 진입점/결과 래퍼일 뿐입니다.

중요한 두 가지(이 프로젝트 실측):
- **결과물은 in-process `EdfModule`** 입니다. 즉 같은 파이썬 프로세스 안에서 `cm(*inputs)` 로
  부르면 NPU에서 돌아가는, torch `nn.Module` 같은 객체입니다. 파일로 떨어지는 "아티팩트"가
  아닙니다.
- NPU 실행 검증법: `furiosa.torch.custom_ops.dfg._dfg_inner` 가 **CPU 전용 폴백 경로**라,
  이걸 감싸 호출 횟수가 0이면 그 연산이 전부 NPU에서 돌았다는 뜻입니다(이 프로젝트가 쓴 spy
  기법, `qcn/deltanet_layer.py:45-52`).

---

## 4. 능력과 한계 (무엇이 컴파일되나)

- **지원 연산 97종**(`SUPPORTED_ATEN_OPS`). 행렬곱(matmul/linear), 어텐션 SDPA, elementwise,
  reduce, softmax, RMSNorm 구성요소, conv 일부 등 트랜스포머·CNN의 주요 연산이 포함됩니다.
  목록과 실행 가능 여부의 차이(목록에 있어도 조건부·불가한 것)는 [README_op_support.md](README_op_support.md)
  에 정리돼 있습니다(예: `isnan`·`constant_pad_nd` 완전 불가, `gather`/`cumsum` 등 6개 조건부).
- **정밀도**: matmul은 DPE(systolic) 엔진에서 bf16으로 돌아 torch 대비 약 0.23% 상대오차가
  납니다(하드웨어 지문). 그 외 대부분은 더 정확합니다.
- **모델 단위 실측**(비전, [README_vision_compile.md](README_vision_compile.md)): MobileNet·
  EfficientNet은 컴파일·실행 OK(MobileNetV2는 NPU 실행까지), ResNet·YOLO는 풀링/Cpu 노드로
  막힙니다. 즉 "모든 PyTorch 모델"이 아니라 **지원 연산으로 떨어지는 그래프**만 NPU로 갑니다.
- **손수 커널**: 컴파일러가 자동으로 못 내리는 연산도 `TacticKernelModule(dsl)` 로 직접
  TK-graph를 작성하면 NPU에서 실행됩니다(이 프로젝트가 Gated DeltaNet을 이렇게 NPU에서 돌렸음,
  [README_qwen3_coder_next.md](README_qwen3_coder_next.md)).

요약하면 furiosa-torch는 **유연**합니다(임의 PyTorch 계산을 NPU로 시도, 손수 커널까지 가능).
대신 **저수준**입니다(연산 지원 한계를 직접 다뤄야 하고, 서빙·배치·KV 캐시 같은 운영 기능은
없음).

---

## 5. furiosa-llm과의 관계 (둘은 층이 다름)

헷갈리기 쉬운데, 둘은 경쟁이 아니라 **층**입니다.

```
 [ furiosa-llm ]  (고수준: 서빙 프레임워크)
   build: 모델 → trace → 분할/색칠 → compile → 아티팩트(a6) → serve(스케줄러·연속배칭·KV·OpenAI API)
            └─ 이 compile 단계가 내부적으로 ───┐
                                               ▼
 [ furiosa-torch ] (저수준: torch→NPU 컴파일·실행 엔진)
   torch 그래프 → DFG → lower → EDF, NPU 실행
```

- **furiosa-llm build는 내부적으로 furiosa-torch를 씁니다**(native_torch.so가 torch 그래프를
  EDF IR로 낮추는 역할). 하지만 furiosa-llm은 거기에 파이프라인 분할·스케줄러·KV 캐시·서빙
  계약을 더해 **a6 `CompiledGraph` + 메타데이터(`NextGenArtifact`)** 라는 서빙용 도시락을
  만듭니다.
- **furiosa-torch 단독**은 그 도시락을 만들지 않습니다. in-process로 컴파일·실행할 뿐입니다.

그래서 "PyTorch 모델을 NPU에서 돌린다"는 두 길이 있습니다.
1. **furiosa-llm build** → 서빙용 아티팩트(a6) → `furiosa-llm serve`(빠른 native 서빙). 단
   지원 모델·tp 제약이 있습니다(예: 대형 bf16의 inter-chip, qwen3_next의 DeltaNet 등은 막힘).
2. **furiosa-torch** → in-process 컴파일·실행. 제약이 적고 손수 커널까지 되지만, 서빙은 직접
   짜야 합니다(아래 6절).

---

## 6. furiosa-torch로 만든 것을 serve 할 수 있나 (핵심 질문)

### 6-1. furiosa-llm `serve` 로는 불가 (포맷 비호환, 실측)

furiosa-llm `serve <아티팩트>` 는 **`NextGenArtifact`** 를 읽습니다. 이건 `model`·`metadata`·
`generator_config`·`pipelines`(컴파일된 a6 `CompiledGraph` 묶음 = `binary_bundle.zip`)·parallel
config 를 갖춘 구조입니다(`load`/`load_without_blob` 으로 적재).

그런데 furiosa-torch가 만드는 건 **a5 `ir.Edf`** 입니다. 이 프로젝트 실측으로,
`furiosa.torch.custom_ops.edf.CompileModule.from_module` 가 내는 EDF는 헤더가 `a163456466 a5`
이고 **최상위 `binaries` 필드가 없어서**, furiosa-llm의 `CompiledGraph.deserialize` 가
**거부**합니다(반면 furiosa-llm `furiosa.native_common.compiler.compile` 은 a6를 냄). 또한
furiosa-torch 산출물에는 서빙에 필요한 메타데이터(model_metadata·버킷·스케줄러 설정 등)가
아예 없습니다.

따라서 **furiosa-torch로 컴파일한 결과는 furiosa-llm serve로 그대로 못 띄웁니다.** 포맷도
다르고, 서빙이 요구하는 메타·파이프라인 구조도 없습니다. 둘 사이를 잇는 변환기도 없습니다.

### 6-2. 그럼 어떻게 serve 하나 (다른 방법, 이 프로젝트가 실증)

furiosa-torch 모델은 **직접 서버를 짜서** 서빙합니다. 방법은 두 가지이고, 둘 다 이 프로젝트에
구현·검증돼 있습니다.

**(A) 커스텀 in-process 서버 (권장·검증됨).** furiosa-torch로 NPU에서 도는 모델을 파이썬
프로세스가 들고, 그 위에 FastAPI로 OpenAI 호환 HTTP를 씌웁니다. host가 추론 루프(토큰 생성)를
돌리고 무거운 계산만 NPU에 보냅니다.
- 실례: `qcn/serve.py`(Qwen3-Coder-Next 80B), `qcn/serve_q25.py`(Qwen2.5-72B bf16). 둘 다
  `furiosa.torch` 백엔드로 NPU 계산을 돌리는 host 추론 루프를 OpenAI API로 감싼 것입니다.
  즉 **"furiosa-torch 모델을 serve하는 법"의 실제 답**이 이미 이 레포에 있습니다.

**(B) furiosa-llm의 서버 코드 재사용 (어댑터 주입).** furiosa-llm의 Python 서빙 엔진
(`AsyncLLMEngine`·`OpenAIServingChat`)은 엔진 객체를 덕타이핑으로 받으므로, furiosa-torch 기반
host 루프를 같은 인터페이스(`HostLoopEngine`)로 감싸 끼우면 furiosa-llm의 **서버 스택**(OpenAI
라우팅·스트리밍)을 그대로 쓸 수 있습니다. 단 이건 furiosa-llm의 **native 엔진(아티팩트 로드)**을
쓰는 게 아니라 **서버 코드만** 빌려 쓰는 것입니다.
- 실례: `qcn/furiosa_serve_adapter.py`(`HostLoopEngine`+`build_async_engine`), `qcn/furiosa_serve_cli_shim.py`.
  상세 [README_qwen3_coder_next.md](README_qwen3_coder_next.md) §7.

정리: **(A)와 (B) 모두 "서버는 우리가, NPU 계산은 furiosa-torch가"** 라는 같은 구조입니다.
furiosa-torch 자체에 서빙 기능이 없으므로 어느 쪽이든 서버 계층은 직접 마련해야 합니다.

### 6-3. 무엇을 기대할 수 있나 (성능·용도)

- furiosa-torch 기반 host 서빙은 **연속 배칭·페이지드 KV 같은 native 서빙 최적화가 없습니다.**
  토큰을 host 루프가 한 단계씩 돌리고, 큰 모델은 가중치를 레이어별로 NPU에 스트리밍하므로
  **느립니다**(예: bf16 Qwen2.5-72B는 가중치 135GiB를 토큰마다 스트리밍해 수백 초/토큰).
- 그래서 furiosa-torch 서빙의 가치는 **속도가 아니라 "되게 함"** 입니다. furiosa-llm build가
  못 굽는 모델(대형 bf16, qwen3_next DeltaNet 등)을 정확도 그대로 NPU에서 돌리고 API로 낼 수
  있다는 점입니다. 빠른 처리량이 목표라면 furiosa-llm build로 굽는 native 서빙(가능한 모델에
  한해)이나 prebuilt 아티팩트가 맞습니다.

---

## 7. 한 줄 결론

- **furiosa-torch** = PyTorch를 NPU에서 컴파일·실행하는 저수준 장치/컴파일 백엔드(서빙 기능 없음,
  in-process a5 EDF, 97개 ATen 연산 + 손수 커널).
- **serve**: furiosa-torch 산출물은 **furiosa-llm serve로 못 띄웁니다**(a5 vs a6, 메타·파이프라인
  부재). 서빙하려면 **직접 서버**(커스텀 OpenAI 서버, 또는 furiosa-llm 서버 코드에 host-loop
  엔진 주입)가 필요하며, 그 구현이 이 레포의 `qcn/serve.py`·`qcn/serve_q25.py`·
  `qcn/furiosa_serve_adapter.py` 입니다.

근거: 모듈 `furiosa/torch/`(API 실측), native `native_torch.so`(Rust furiosa-torch+npu-executor),
a5/a6 포맷 실측(`qwen3-next-proj/tk_kernels/compile_edf_blobs.py` 도크스트링), `NextGenArtifact`
구성(`furiosa.native_llm_common`), 이 프로젝트 서빙 구현(`qcn/`).
