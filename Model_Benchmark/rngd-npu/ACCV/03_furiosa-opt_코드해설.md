# 03. furiosa-opt 코드 해설 — 어느 코드를 왜 쓰는가

이 문서는 우리 실험이 furiosa-opt(와 관련 SDK)의 **어느 경로의 어느 코드**를 쓰는지, 그 코드가 **무슨 뜻인지**, **왜 쓰는지**를 배경지식이 적은 사람도 따라올 수 있게 설명한 기술 해설입니다. 코드 인용은 furiosa-opt 공개 저장소(<https://github.com/furiosa-ai/furiosa-opt>)와 이 프로젝트의 실측 스크립트 기준입니다.

## 0. furiosa-opt가 뭔가

furiosa-opt는 RNGD NPU(퓨리오사의 추론용 칩)를 **사람이 직접 저수준으로 프로그래밍**할 수 있게 공개한 도구입니다. 보통은 PyTorch 같은 고수준 도구가 칩의 세부를 알아서 처리해 주는데, furiosa-opt는 칩의 계산 장치(엔진)·메모리·정밀도를 **손으로** 다룰 수 있게 열어 줍니다. 비유하면, 자동 변속기(고수준) 대신 수동 변속기(furiosa-opt)를 주는 것입니다 — 손이 많이 가지만 원하는 대로 제어할 수 있습니다.

**중요한 구분(정직하게).** 우리 핵심 실험은 대부분 *고수준 경로*(`furiosa.torch`)로 돌아갑니다. furiosa-opt(저수준 vISA)는 두 가지 역할입니다:
1. **칩이 숫자를 어떻게 다루는지 알려주는 "설명서"** — 이걸 보고 PC에서 칩을 흉내내는 모델(Q_npu)을 정확히 만듭니다.
2. **정밀도를 직접 통제해 복구를 칩에서 시연하는 "보너스 도구"** — 실험 E8(후속)에서 씁니다.

아래 A는 실제로 실행하는 코드, B~F는 furiosa-opt에서 근거·도구로 쓰는 코드입니다.

---

## A. 실제 실험을 돌리는 코드 (고수준 `furiosa.torch` 경로)

### A-1. `rngd-npu/vision_models/classify.py` — 학습 모델을 칩과 CPU에서 분류
- **무슨 코드:**
  - `make_model()` (47–52행): `M.mobilenet_v2(weights=IMAGENET1K_V1)`로 **학습된 가중치**를 불러옵니다.
  - `export_decompose()` (55–62행): `torch.export` 후 **batch_norm을 직접 분해**합니다. 이걸 안 하면 칩 컴파일러가 그 연산을 못 받습니다.
  - `CompileModule.from_exported(ep)` (98행): 모델을 칩용 프로그램(EDF)으로 컴파일.
  - `cm(x.to(dev), device=dev)` (110행): 칩(`rngd:N`)에서 추론. CPU 결과(`m(x)`)와 top-5 비교.
  - `--reuse-edf` (91–95행): 이미 만든 EDF를 그대로 쓰고 **가중치만 갈아끼웁니다**(`ir.Edf.deserialize` + `CompileModule(EdfModule(edf), ExportedProgramWeight(ep))`).
- **무슨 뜻(쉽게):** "학습된 모델을 칩에 올려 진짜 사진을 분류하고, 정답(CPU)과 비교하는" 도구입니다.
- **왜 쓰나:** 붕괴(M1)와 복구(M5)를 **실제 칩에서** 측정하는 메인 도구라서. 컴파일 결과(EDF) 안에는 가중치가 안 들어가고 **가중치는 실행할 때 따로 들어가는 입력**이라(`--reuse-edf`가 이걸 이용), 설정만 바꿔 가며 **재컴파일 없이** 빠르게 비교할 수 있습니다.

### A-2. `rngd-npu/run_edf.py` — EDF 저장·재사용
- **무슨 코드:** `do_compile()`는 `cm.edf.serialize()`로 EDF를 파일로 저장(+`state_dict`), `do_run()`은 `ir.Edf.deserialize()`로 불러와 재컴파일 없이 실행.
- **왜 쓰나:** 컴파일은 수백 초 걸리는데, 한 번 저장해 두면 이후 실행은 수 초입니다. M4→M5에서 스케일을 바꿔 가며 여러 번 칩에 올릴 때 시간을 크게 아낍니다.

---

## B. 칩이 숫자를 다루는 방식 (Q_npu 흉내 모델의 근거)

### B-1. Contraction 엔진 — "곱하기는 잘게, 더하기는 넓게"
- **어디:** `docs/src/computing-tensors/contraction-engine/index.md` (furiosa-opt 책).
- **인용:** *"the Multiplier widens to the contraction output type (`i4`/`i8` -> `i32`, `f8`/`bf16` -> `f32`)"* (56행). 또 *"the Outer stage caps `Lane ≤ 8` and `Packet ≤ 64 B` (on RNGD)"* (62행).
- **무슨 뜻(쉽게):** 칩의 곱셈-덧셈 장치는 입력 숫자는 **잘게(bf16/int8)** 받지만, 곱한 결과를 더할 때는 **넓은 그릇(i32/f32)** 에 모읍니다. 즉 **더하다 넘쳐서** 망가지는 게 아닙니다 — 망가지는 지점은 **곱하기 직전에 입력을 잘게 깎는 단계**입니다. 그리고 한 번에 곱하는 칸이 8개로 제한됩니다(`Lane ≤ 8`).
- **왜 쓰나:** 이 한 줄이 우리 논문의 **메커니즘을 확정**합니다. 흉내 모델 Q_npu를 "입력을 bf16으로 깎고 → f32로 넓게 더한다"로 만들어야 칩과 맞습니다(M3). 또 `Lane ≤ 8` 제한이 우리가 채널을 8개 묶음 단위로 다루는 "8-tile" 제약의 근거입니다(`README_op_support.md`의 8-tile 발견과 일치).

---

## C. 정밀도를 직접 고르는 단추 — Cast 엔진

- **어디:** `furiosa-opt-std/src/engine/cast.rs` + 책 `docs/src/computing-tensors/cast-engine.md`.
- **인용(코드):** `pub fn cast<OutD: Scalar, OutPacket: M>(self) -> CastTensor<...>` (cast.rs 37–44행). 책 설명: *"narrows `f32`/`i32` pipeline results to lower-precision types (e.g., `bf16`)"* + 지원 변환표:

  | 입력 | 가능한 출력 |
  |---|---|
  | `i32` | `i4`, `i8`, `i16` |
  | `f32` | `f8e5m2`, `f8e4m3`, `f16`, `bf16` |

- **무슨 뜻(쉽게):** `.cast::<OutD>()`는 "이 숫자들을 어떤 정밀도로 깎을지 **내가 직접 고르는** 단추"입니다. 예를 들어 `i32 → i8`로 깎을 수 있습니다.
- **왜 쓰나:** 고수준 `furiosa.torch` 경로에는 이 단추가 **숨겨져 있어**(정밀도/양자화 설정 칸이 0개) 무조건 bf16으로 깎여 붕괴했습니다. vISA에서는 이 `.cast()`로 **`int8`로 깎되 채널별 스케일과 함께** 깎을 수 있습니다. 01번 문서에서 말한 "부동소수 스케일은 효과 없고, int8 + 클립이 진짜 레버"라는 복구를, 칩에서 직접 구현하는 도구가 바로 이 Cast 엔진입니다(보너스 실험 E8).

## D. 튀는 값 자르기·정수 셈 — Vector 엔진

- **어디:** `furiosa-opt-std/src/engine/vector/op/semantics.rs` + 책 `docs/src/computing-tensors/vector-engine/`.
- **인용:** `vector_clip(ClipBinaryOpF32::Max, 0.0f32)`(값 자르기/클램프), `vector_fxp(FxpBinaryOp::AddFxp, 100)`(고정소수점 셈). intra/inter-slice reduce(합·최댓값 모으기)도 있음.
- **무슨 뜻(쉽게):** `vector_clip`은 "이 값보다 크면 잘라"라는 가위, `vector_fxp`는 정수 기반(고정소수점) 사칙연산입니다.
- **왜 쓰나:** 우리 복구의 두 레버 중 하나가 **outlier(튀는 값) 클립**입니다. 이 가위가 칩에 직접 있어, int8 변환 전에 튀는 값을 잘라 누적 오차를 줄일 수 있습니다(E8).

## E. 행렬곱/합성곱을 손으로 짜는 실제 모양 — GEMM 커널 예제

- **어디:** `base-template/src/kernel/gemm_kernel.rs`.
- **인용(핵심 줄):**
  ```rust
  a: &HbmTensor<bf16, Chip, m![I, K]>,           // 입력 두 개를 bf16으로 받아
  ...
  .contract_outer(...).contract_packet(...).contract_time(...).contract_lane(...)  // 곱하고 넓게 더하고
  .cast::<bf16, m![J % 8 # 16]>()                 // ★ 마지막에 정밀도를 골라 깎는다
  .commit(0)
  ```
- **무슨 뜻(쉽게):** 행렬곱(GEMM)을 칩 위에서 단계별로 짜는 실제 코드입니다. 합성곱(conv)도 본질은 같은 곱-합이라 같은 방식으로 표현됩니다. 중요한 건 **맨 끝의 `.cast()` 가 "정밀도를 고르는 자리"** 라는 점입니다.
- **왜 쓰나:** 보너스 실험 E8에서 conv를 **명시적 정밀도(int8 + 우리 스케일)** 로 다시 짤 때의 청사진입니다. "어디서 정밀도가 결정되는가(=`.cast`)"를 보여 주므로, 우리 복구를 칩 코드 어디에 끼워야 하는지가 분명해집니다.

## F. NPU 없이 검증하기 — simulation 백엔드

- **어디:** `furiosa-opt-std/src/runtime/simulation/backend.rs`, 기본 백엔드 설정은 `furiosa-opt-std/build.rs`(빌드 시 simulation을 기본으로 주입). 백엔드 종류는 `furiosa-opt-std/src/runtime/mod.rs`(`typecheck`/`simulation`/`emulation`/`npu`).
- **무슨 뜻(쉽게):** 커널을 **실제 칩 없이 PC에서 숫자로 돌려** 결과가 맞는지 확인하는 모드입니다.
- **왜 쓰나:** 복구 아이디어(M4)와 conv 커널(E8)이 **수치적으로 맞는지** 칩을 점유하지 않고 먼저 확인할 수 있습니다. 칩(npu0~2)이 LLM 서빙에 바쁠 때도 실험을 돌릴 수 있어, 날짜에 안 묶이고 진도를 빼는 데 핵심입니다.

---

## 정리 — 무엇을 어디에 쓰는지 한 표

| furiosa-opt / SDK 코드 | 무슨 뜻 | 우리 실험에서 왜 |
|---|---|---|
| `classify.py`, `run_edf.py` (furiosa.torch) | 학습 모델을 칩·CPU에서 분류·비교, EDF 저장·재사용 | **M1 붕괴·M5 복구를 실제 칩에서 측정** (메인 실행) |
| `contraction-engine/index.md`(widening, Lane≤8) | 곱하기는 잘게, 더하기는 넓게; 한 번에 8칸 | **Q_npu 흉내 모델의 근거**(M3) + 8-tile 제약 |
| `engine/cast.rs` `.cast::<OutD>()` | 정밀도를 직접 고르는 단추(i32→i8 등) | 고수준이 숨긴 정밀도 통제 → **int8 복구의 핵심 도구**(E8) |
| `engine/vector` `vector_clip`/`vector_fxp` | 튀는 값 자르기 + 정수 셈 | 복구 레버 중 **클립**을 칩에서 구현(E8) |
| `base-template/.../gemm_kernel.rs` | 곱-합을 손으로 짜고 끝에 `.cast`로 정밀도 결정 | **E8에서 conv를 명시 정밀도로 재구현**하는 청사진 |
| `runtime/simulation` + `build.rs` | NPU 없이 PC에서 수치 검증 | 복구·커널이 맞는지 **칩 없이 확인**(M4·E8) |

**한 줄 정직 요약:** 핵심 결과(붕괴·진단·흉내·복구, M0~M4)는 furiosa-opt 손코딩 없이 고수준 경로와 PC만으로 냅니다. furiosa-opt는 ① 칩의 정밀도 동작을 알려주는 **근거**(B)와 ② 칩에서 직접 정밀도를 통제해 복구를 시연하는 **보너스 도구**(C·D·E·F)로 씁니다. 더 깊은 분석은 `Model_Benchmark/info/README_virtual_isa.md`에 있습니다.
