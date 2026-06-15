# Pipeline build vs Compile — `furiosa-llm build`의 두 단계

`furiosa-llm build` 한 명령으로 보이지만, 안에서는 **완전히 다른 성격의 두 단계**가
연달아 실행됩니다. 한쪽은 그래프를 그리는 단계이고 다른 쪽은 그 그래프를 NPU 명령어로
번역하는 단계예요. 두 단계의 시간·메모리 특성이 크게 달라서, 빌드를 모니터링하고
디버깅할 때 둘을 구분하는 게 중요합니다.

이 문서에서는 두 단계가 무엇을 하는지, 왜 다른지, 어떻게 식별할 수 있는지를 정리합니다.

(소스 인용은 모두 `~/furiosa/lib/python3.12/site-packages/furiosa_llm/`을 기준,
SDK 2026.2.0.)

---

## 한 줄 비유

| 단계 | 비유 |
|---|---|
| **Pipeline build** ("tracing") | 모델을 보고 "**무엇을 계산할지**" 그래프(설계도)를 그리는 단계 |
| **Compile** | 그 설계도를 "**NPU가 실제로 실행할 명령어**" 로 번역하는 단계 |

조금 더 풀어 쓰면:
- Pipeline build = "**컴파일러에게 줄 입력(IR)을 만든다**"
- Compile = "**IR을 NPU 바이너리(EDF)로 변환한다**"

---

## 전체 흐름에서 어디

```
furiosa-llm build (한 명령)
  │
  ├─ ❶ HF config 로드 + 검증 + resolve (수 초)
  │
  ├─ ❷ Pipeline build  ─────────  ★ 메모리 위험 구간
  │     "Model Tracing Progress: x/Y"
  │     - 각 버킷마다 PyTorch FX 트레이싱
  │     - tp/pp 병렬화, 그래프 최적화
  │     - Ray actor: LocalPipelineGenerationActor.build_for_bucket
  │     → 출력: Pipeline 객체 (그래프 IR + 메타데이터)
  │
  ├─ ❸ Compile  ────────────────  ★ 시간 길지만 메모리 안전
  │     "Compilation Progress: x/Y"
  │     - 그래프 supertask 단위로 분할
  │     - 각 supertask → NPU 컴파일러 → EDF 바이너리
  │     → 출력: 여러 개의 EDF 파일
  │
  └─ ❹ 저장
        artifact.json + binary_bundle.zip + config.json + tokenizer
```

❷와 ❸이 이 문서의 주제입니다. 진입 코드: `builder.py:268` `next_gen.build_pipeline(...)`
한 번 호출 안에서 ❷·❸이 순서대로 일어납니다 (워커 풀은 별도: `num_pipeline_builder_workers`,
`num_compile_workers`).

---

## ❷ Pipeline build 단계 ("tracing")

### 하는 일

1. HF weight 파일을 메모리에 적재 (FP8 32GB → 트레이싱 중 부분 dequant 시 60GB+)
2. **버킷마다** PyTorch FX tracer로 모델 forward를 따라가며 계산 그래프(FX IR) 구성
   - 정의된 버킷이 N개면 트레이싱도 N번 (각 버킷의 shape를 입력 가정으로)
3. tensor/pipeline 병렬화 적용 — 그래프를 device mesh에 따라 분할
4. paged-attention 적용, decomposition, constant 임베딩 등 그래프 변환
5. 모든 버킷의 그래프를 하나의 Pipeline 객체로 묶음

### 핵심 개념

- **FX 그래프** — PyTorch의 정적 그래프 표현(IR). `torch.fx.GraphModule`. 노드는 연산,
  엣지는 데이터 흐름. 이 단계의 결과물이 곧 컴파일러 입력.
- **트레이싱** — 더미 입력 텐서로 모델을 한 번 실행하면서 모든 연산을 기록 → FX 그래프 완성.
- **버킷** — `(batch_size, context_length)` 조합. 버킷 하나당 그래프 하나가 만들어짐.
  N개 버킷 = N번 트레이싱 = IR이 N배.
- **Ray actor** — `LocalPipelineGenerationActor`가 이 단계의 워커. 하나의 actor가 한 인스턴스
  안에서 여러 버킷의 트레이싱을 누적 처리.

### 왜 메모리가 무거운가

- 전체 모델 weight를 한 메모리 공간에 유지해야 함 (32B 빌드 시 60~100GB 차지 가능)
- 트레이싱 중 활성화 텐서가 그래프 안에 *심볼릭으로* 잡혀서, 큰 버킷(예: (32, 32k))의 IR이
  그 자체로 매우 큼
- N개 버킷의 그래프 IR이 한 actor 인스턴스 안에 누적 — 마지막 버킷쯤 가면 누적 IR이
  weight보다 더 커질 수도 있음
- → **OOM 사망의 거의 100%가 이 단계**

### 메모리 절약 옵션

- `--max-model-len` 축소 → preset 버킷 중 큰 것들 자동 제외 (`artifact/presets.py:425` `filter_preset_by_max_model_len`)
  - 예: 40960 → 16384로 줄이면 40k·32k 버킷 빠짐 → 트레이싱 횟수와 IR 크기 모두 감소
- `--num-pipeline-builder-workers` 1로 유지 (기본값) — 늘리면 actor가 병렬로 떠서 RAM 곱으로 증가
- `-pb`/`-db`로 버킷 수동 지정해 꼭 필요한 모양만 트레이싱

### 출력

- Pipeline 객체 — 그래프 IR + 메타데이터(파라미터 위치, 디바이스 매핑, bucket 정보)
- 아직 NPU에서 실행 불가 — 단지 "**컴파일러에 넘길 IR**" 상태

### 코드 위치

- 진입 로그: `builder.py:248` (`Attention buckets: ...` 출력)
- 호출: `builder.py:268-295` `next_gen.build_pipeline(...)`
- 본체: `parallelize/new_pipeline_builder.py`
- 트레이싱: `parallelize/trace.py`

### 시간 특성

- iter 시간은 비교적 짧음 (~40~60초/iter)
- iter 수는 버킷 수에 비례 (기본 preset이 105~123개)
- 전체 시간은 buckets·model size에 비례

---

## ❸ Compile 단계

### 하는 일

1. Pipeline build에서 만든 그래프를 작은 단위인 **supertask**로 분할
   - 보통 transformer block 1개가 1 supertask (`CompilerConfig.num_blocks_per_supertask=1`,
     `types/config.py:103`)
2. 각 supertask를 **furiosa NPU 컴파일러**(`furiosa.native_common.compiler`)에 입력
3. 컴파일러는 supertask의 그래프를 NPU 명령어 시퀀스로 lowering
4. 결과를 **EDF**(Executable DataFlow) 바이너리로 저장
5. 모든 supertask가 컴파일되면 그 EDF 파일들이 `binary_bundle.zip`(또는
   `--no-bundle-binaries` 시 개별 `.edf` 파일들)의 본체가 됨

### 핵심 개념

- **EDF (Executable DataFlow)** — NPU가 직접 실행하는 명령어 포맷. 메모리 layout, dataflow
  스케줄, 연산 분배 등이 다 박힌 바이너리.
- **Supertask** — 그래프의 한 덩어리. 컴파일·실행 단위.
- **NPU 컴파일러** — furiosa의 자체 컴파일러(`furiosa.native_common.compiler`). FX 그래프
  서브셋을 받아 EDF로 lowering. 이 컴파일러 자체는 정적 분석·최적화 중심.

### 왜 메모리가 가벼운가 (상대적으로)

- 컴파일 워커는 현재 처리 중인 supertask + 그 파라미터만 메모리에 둠
- 전체 weight·전체 그래프를 들고 있을 필요 없음
- supertask 하나씩 처리하므로 누적 메모리도 거의 없음
- → 우리 32B 실측: 컴파일 단계 RAM 사용 **~20GB / 125GB** (Pipeline build의 1/4~1/5)

### 출력

- 여러 개의 EDF 바이너리 (= NPU에서 바로 실행 가능)
- 각 binary는 특정 (bucket, device-shard) 조합에 대응
- 묶음: `binary_bundle.zip` (기본) — `builder.py:443-477`

### 코드 위치

- 진행 로그: `parallelize/pipeline/builder/converter.py` 의 컴파일 루프 (`Compilation Progress`)
- 컴파일러 호출: `furiosa.native_common.compiler` (외부 패키지)

### 시간 특성

- iter 시간은 길어짐 (~100~200초/iter) — 컴파일러가 정적 분석·최적화에 시간 사용
- iter 수는 supertask 수에 비례 (32B + 16k 빌드 시 134개)
- 전체 시간은 model size에 비례 (큰 모델일수록 supertask·연산 많음)

---

## 비교 정리

| 항목 | ❷ Pipeline build | ❸ Compile |
|---|---|---|
| 로그 패턴 | `Model Tracing Progress: x/Y` | `Compilation Progress: x/Y` |
| 무엇을 만드나 | FX 그래프 IR (= 컴파일러 입력) | EDF 바이너리 (= NPU 실행 가능) |
| 입력 | HF weight + bucket 정의 | Pipeline build의 출력 |
| 단위 반복 | 버킷마다 트레이싱 | supertask마다 컴파일 |
| 워커 옵션 | `--num-pipeline-builder-workers` (기본 1) | `--num-compile-workers` (기본 1) |
| 워커 인스턴스 | `LocalPipelineGenerationActor` (Ray) | 컴파일러 풀 |
| 메모리 부담 | ★ 매우 큼 (weight + 누적 IR) | 낮음 (supertask 단위) |
| OOM 위험 | ★ 매우 높음 — 거의 모든 OOM 여기서 발생 | 거의 없음 |
| iter 시간 | 짧음 (40~60초) | 김 (100~200초) |
| 전체 시간 비중 | ~1/4 | ~3/4 |
| 메모리 절약 lever | `--max-model-len`, 버킷 수동 축소 | (보통 조정 불필요) |

---

## 왜 이 구분이 중요한가

### 1) OOM 진단

빌드가 죽었을 때 마지막 로그가 `Model Tracing Progress`인지 `Compilation Progress`인지로
원인을 좁힐 수 있습니다.

- Tracing 중 사망 → 메모리 문제 → `--max-model-len` 축소 또는 swap·oomd 조치
- Compile 중 사망 → 매우 드뭄. 봤다면 다른 원인(디스크 부족, 컴파일러 버그 등) 의심

### 2) 시간 예측

`Model Tracing Progress: 50/105` 같은 진행률은 **트레이싱만의 비율**입니다. 50%면
전체로 보면 ~12.5%일 수도 있어요 (트레이싱이 전체의 1/4이라). compile 진입 후
`Compilation Progress`가 뜨면 그제서야 전체 진행률에 가까운 숫자.

### 3) 워커 옵션 선택

`--num-pipeline-builder-workers`를 늘리면 트레이싱이 빨라지지만 메모리는 곱셈. 32B처럼
큰 모델이면 1로 두는 게 안전. `--num-compile-workers`는 늘려도 메모리 부담 적어서
compile 단계를 빠르게 끝내고 싶을 때 후보.

### 4) "위험 구간 통과" 신호

`Compilation Progress` 로그가 한 번이라도 뜨면 메모리 위험 구간은 통과한 상태입니다.
그 뒤에 갑자기 죽는 일은 거의 없으므로, 시간만 기다리면 됩니다.

---

## 우리 32B FP8 빌드 실측 (`--max-model-len 16384`)

| 단계 | iter 수 | iter 시간 | 총 시간(추정) | 피크 RAM |
|---|---:|---:|---:|---:|
| Pipeline build (tracing) | 105 | ~45초 | ~1h 18m | 70~80GB |
| Compile | 134 | ~150초 | ~5h 35m | ~20GB |
| **합계** | — | — | **~7h** | — |

(`--max-model-len 40960` default로 했을 때는 트레이싱 iter 수 123, 피크 RAM 100GB+로
OOM 사망. 16k로 줄이고 큰 버킷 제외되면서 트레이싱 양과 메모리 둘 다 감소.)

---

## `furiosa-llm build`의 컴파일러 == `furiosa-compiler` 명령인가?

자주 헷갈리는 질문이에요. **결론부터 말하면: 같은 컴파일러 *코드베이스*에서 나왔지만,
서로 다른 *빌드 산출물*이고 이 머신에서는 버전·진입 방식까지 다릅니다.**
즉 `furiosa-llm build`가 `/usr/bin/furiosa-compiler` 명령을 불러 쓰는 건 **아닙니다.**

### 같은 점 — 컴파일러 코어(계보)는 하나

둘 다 furiosa 내부 모노레포 `npu-tools`(cargo git 체크아웃 루트 해시
`npu-tools-dec42ebd45bb55cf`)의 **`npu-compiler` / `npu-ir` 크레이트 패밀리**를 정적으로
박아 빌드했습니다. 그래프를 NPU 명령어로 낮추는(lowering) 핵심 로직은 같은 프로젝트예요.

#### `npu-tools`는 어디서 나온 이름인가

만든 이름이 아니라, 산출물 안에 박힌 **cargo git 체크아웃 경로 문자열**에서 그대로 읽은
것입니다. `.so`·CLI 둘 다 이런 문자열을 담고 있어요:

```
/root/.cache/cargo/git/checkouts/npu-tools-dec42ebd45bb55cf/<commit>/crates/npu-compiler/...
```

이건 Rust cargo가 **git 저장소를 의존성으로** 가져올 때 만드는 표준 구조
(`~/.cargo/git/checkouts/<저장소이름>-<URL해시>/<커밋>/`)입니다. 따라서 `npu-tools`는
컴파일러 코어 크레이트(`npu-ir`, `npu-compiler`, `scheduler`, `command-gen`,
`tactic-populator`, `global-compiler` …)가 든 **Furiosa 사내 Rust 저장소**이고,
`dec42ebd45bb55cf`는 cargo가 그 저장소 URL을 해시한 식별자입니다. 저장소는 사실 **둘**:

| 저장소 | 역할 | 증거 문자열 |
|---|---|---|
| **`furiosa-runtime`** | `furiosa-compiler` CLI의 소스 집 | CLI: `/workspace/source/furiosa-runtime/furiosa-compiler/src/api.rs` |
| **`npu-tools`** | 컴파일러 코어 크레이트. cargo git 의존성으로 가져다 씀 | 양쪽: `cargo/git/checkouts/npu-tools-…/crates/…` |

(한계: `npu-tools`는 비공개 사내 저장소라 소스를 직접 못 열고, 바이너리에 박힌 경로로
이름·역할을 추정한 것입니다.)

#### 그럼 npu-tools를 어디서 보나 — 소스는 못 보고, "흔적/버전"만

박힌 경로 `/root/.cache/cargo/...`·`/workspace/source/furiosa-runtime/...`는 **Furiosa
빌드 컨테이너의 경로**라 이 머신에 없습니다(`ls` 시 `Permission denied`/`No such file`).
설치된 건 컴파일된 `.so`/바이너리뿐이라, 이 머신에서 확인 가능한 건 세 가지입니다:

```bash
SP=~/furiosa/lib/python3.12/site-packages

# ① 산출물에 박힌 npu-tools 커밋
strings -n6 $SP/furiosa/native_llm_common.cpython-312-x86_64-linux-gnu.so \
  | grep -oE 'npu-tools-[a-f0-9]+/[a-f0-9]+' | sort -u   # → .../5c885c7
strings -n6 /usr/bin/furiosa-compiler \
  | grep -oE 'npu-tools-[a-f0-9]+/[a-f0-9]+' | sort -u   # → .../8c5160b

# ② 런타임 버전 직접 호출 (가장 확실)
~/furiosa/bin/python -c "import furiosa.native_common.compiler as c; \
  print(c.full_version()); print('ir_hash:', c.compiler_git_short_hash())"
#   Furiosa Compiler 2026.2.0 (rev: 9f92da069, built_at: 2026-04-28T...)
#   ir_hash: 5c885c73ee
#   → 바인딩(furiosa-compiler-python) 커밋=9f92da069, npu-tools/npu-ir 코어=5c885c73ee
#   → 이 ir_hash가 artifact.json의 furiosa_compiler_version (builder.py:357)

# ③ 읽을 수 있는 컴파일러 API 타입 스텁
cat $SP/furiosa/native_common/compiler/__init__.pyi   # compile(), find_compiler_config() ...
```

실제 컴파일러 Rust 소스(`npu-tools`/`furiosa-runtime`)를 보려면 Furiosa에 사내 저장소
접근 권한을 요청해야 합니다.

### 다른 점 ① — 진입 방식(프런트엔드)이 다름

| | `furiosa-llm build`의 compile | `/usr/bin/furiosa-compiler` |
|---|---|---|
| 형태 | **in-process 파이썬 바인딩**(PyO3 `.so`) | **독립 실행 Rust 바이너리**(CLI) |
| 호출 | `from furiosa.native_common.compiler import compile` → `compile(...)` | 셸에서 `furiosa-compiler foo.onnx -o out.edf` |
| 입력 | 이미 트레이싱·전처리된 Torch FX 그래프(IR) | ONNX 또는 IR 파일(dfg/cdfg/gir/lir) |
| 실행 범위 | `skip_trace=True, skip_preprocess=True` → **로워링 백엔드만** (그래프→EDF) | 프런트엔드 포함 **전체 파이프라인** (ONNX→DFG→CDFG→LIR→EDF) |

`furiosa-llm`은 트레이싱·전처리를 파이썬/Torch 쪽
(`furiosa_torch_ext.torch_ext.preprocess`)에서 먼저 끝낸 뒤, 컴파일러에는 "낮추기만 해"라고
지시합니다. 반면 CLI는 모델 파일 하나를 받아 앞단부터 전부 돌립니다.

### 다른 점 ② — 이 설치본에서는 버전·커밋도 다름

| 산출물 | 누가 쓰나 | SDK 버전 | `npu-tools` 커밋 |
|---|---|---|---|
| `furiosa/native_llm_common.cpython-312-*.so` | `furiosa-llm build`의 compile 단계 | **2026.2.0** | **5c885c7** |
| `/usr/bin/furiosa-compiler` | 사람이 직접 치는 CLI | **2025.3.0** | **8c5160b** |
| `furiosa/native_torch.cpython-312-*.so` | `furiosa.torch`(비전/CNN 컴파일) | 2026.2.0 | 3f23a71 |

같은 저장소(`dec42ebd45bb55cf`)의 다른 커밋이라, **바이트 동일이 아니고 정확한 컴파일된
로직도 다릅니다.** 참고로 `furiosa.torch` 경로(비전 컴파일, [`README_vision_compile.md`])까지
치면 컴파일러 진입점은 사실 **셋**이고, 셋 다 같은 `npu-compiler` 코어 크레이트를 공유하되
각자 다른 `.so`/바이너리로 따로 빌드돼 있습니다.

### "그럼 build가 내부적으로 furiosa-compiler를 호출하나?" — "어느 furiosa-compiler냐"에 따라

- **CLI 명령 `/usr/bin/furiosa-compiler`를 부르나 → 아니요.** `furiosa_llm` 전체를 grep해도
  그 바이너리로의 `subprocess`/`Popen`/`os.system` 셸 호출이 **없습니다**(subprocess는
  `cli/collect_env.py` 환경정보 수집 한 곳뿐, 무관). build가 그 명령을 실행하는 일은 없어요.
- **컴파일러 엔진(=`furiosa-compiler-python` 크레이트)을 부르나 → 예, in-process 라이브러리로.**
  `from furiosa.native_common.compiler import compile` → `native_llm_common.so`(PyO3)의
  `compile()`를 같은 프로세스 안에서 직접 호출합니다. 이 `.so`에 박힌 버전 흔적이 그 증거:
  `__git_short_hash__ 9f92da069`(이 바인딩 자체, 2026-04-28) + `__ir_git_short_hash__ 5c885c73ee`
  (그 안에 박은 npu-ir 코어) + `sysmodules → furiosa.native_common.compiler`(PyO3 등록 모듈명).

즉 CLI와 이 라이브러리는 **둘 다 `npu-tools` 코어를 감싼 형제 래퍼**일 뿐, 한쪽이 다른 쪽을
부르는 관계가 아닙니다. 같은 엔진을 ①터미널 실행파일(CLI)과 ②파이썬 import용 라이브러리(.so)로
따로 포장한 것이고, furiosa-llm은 ②를 씁니다.

### 한 줄 정리

> 코어 알고리즘은 한 뿌리(`npu-tools`의 `npu-compiler`)지만, `furiosa-llm build`는 그걸
> **라이브러리(로워링 백엔드)** 로 in-process 호출하고 `furiosa-compiler`는 **독립 CLI(풀
> 파이프라인)** 입니다. 게다가 이 머신에선 버전(2026.2.0 vs 2025.3.0)도 달라요.

### 코드/증거 위치

| 항목 | 위치 |
|---|---|
| compile 호출부 | `furiosa_llm/parallelize/pipeline/builder/converter.py:913` (`compiled = compile(...)`) |
| compile import | 같은 파일 `:876-882` (`from furiosa.native_common.compiler import compile`, 없으면 `furiosa-native-llm-common is required` 에러) |
| 바인딩 shim | `furiosa/native_common/compiler/__init__.py` (PyO3가 `native_llm_common.so`를 등록) |
| 버전 기록 | `furiosa_llm/artifact/builder.py:350-357` (`furiosa_compiler_version=compiler_git_short_hash()`) |
| CLI 정체 | `furiosa-compiler --version` → `v2025.3.0 (4467f6a699-modified)`; `--help` SOURCE = `onnx, dfg, cdfg, gir, lir` |
| CLI 자립성 | `ldd /usr/bin/furiosa-compiler` → libc/libgcc만 (파이썬 `.so` 미링크) |
| 커밋 증거 | `.so` strings → `npu-tools-dec42ebd45bb55cf/5c885c7/crates/npu-compiler/...`; CLI strings → `.../8c5160b/crates/npu-ir/...` + `/workspace/source/furiosa-runtime/furiosa-compiler/src/api.rs` |

(검증: 위 사실은 SDK 2026.2.0 설치본에서 `strings`·`ldd`·`--version`·소스 grep으로 직접
대조했고, 반증 시도 3건 모두 결론을 뒤집지 못했습니다.)

---

## 참고

- 빌드 옵션·검증·OOM 트러블슈팅 → [`README_build.md`](README_build.md)
- `presets.py` 버킷 4종 자세히 (prefill/decode/append/tokenwise, fmt) → [`README_preset.md`](README_preset.md)
- HF `config.json` 필드 자세히 (`max_position_embeddings`, `quantization_config` 등) → [`README_config.md`](README_config.md)
- 빌드 끝난 artifact 서빙·테스트 방법 (`furiosa-llm serve` + curl/SDK) → [`README_runcode.md`](README_runcode.md)
- 다운로드부터 등록까지 단계별 튜토리얼 → [`docs/COMPILING_MODELS.md`](docs/COMPILING_MODELS.md)
- 코드 빠른 참조 위치:

| 항목 | 파일:라인 |
|---|---|
| ArtifactBuilder.build (전체 진입) | `artifact/builder.py:315` |
| Pipeline build 호출 | `artifact/builder.py:268-295` |
| 워커 기본값 (모두 1) | `artifact/builder.py:319-320` |
| Pipeline build 본체 | `parallelize/new_pipeline_builder.py` |
| Tracing | `parallelize/trace.py` |
| Compile 루프 (Compilation Progress 출력) | `parallelize/pipeline/builder/converter.py` |
| 버킷 preset 필터링 | `artifact/presets.py:425` (`filter_preset_by_max_model_len`) |
| EDF 저장 / bundle | `artifact/builder.py:443-477` |
