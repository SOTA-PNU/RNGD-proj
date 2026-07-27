# 소개

FuriosaAI 의 Tensor Contraction Processor(TCP)는 추론 워크로드를 겨냥한 대규모 병렬 AI 가속기다.
메모리 레이아웃과 하드웨어 스케줄링을 추상화해 감추는 PyTorch, XLA 같은 고수준 프레임워크와 달리, TCP 는 저수준 커널 API 의 바이트 단위 추론을 요구하지 않으면서 프로그래머에게 직접적인 제어를 노출한다.

TCP 의 Virtual Instruction Set Architecture(Virtual ISA, 또는 vISA)는 이 제어를 노출하는 프로그래밍 인터페이스다.
프로그래머가 텐서 단위로 사고하면서도 메모리 할당과 텐서 유닛 스케줄링을 직접 관리하게 해준다.
이 매뉴얼은 그 인터페이스를 소개하며, vISA 를 직접 작성하는 프로그래머와 vISA 를 생성하는 컴파일러 개발자라는 두 독자를 대상으로 한다.
두 독자 모두 기본적인 Rust 지식을 전제한다.
필요하면 [언어 매뉴얼](https://doc.rust-lang.org/book/)을 보라.

> [!WARNING]
> **알파 테스트 빌드: 실험적 소프트웨어**
>
> 이 소프트웨어는 기술 평가와 내부 테스트만을 위한 초기의, 실험적이고 불완전한 빌드다.
>
> 이 소프트웨어를 프로덕션 작업, 중요한 업무, 중요한 데이터에 사용하기 전에는 반드시 Furiosa 엔지니어와 상의해야 한다.
>
> 여러분의 피드백은 우리 개발에 필수적이다. 꼭 전해 달라.

<a id="installation"></a>
## 설치

세 가지를 설치한다.

1. **Rust 툴체인(고정됨)**: Furiosa 옵티마이저는 rustc 드라이버이며, 특정 nightly 에 ABI 가 묶여 있다.

   ```bash
   rustup toolchain install nightly-2026-05-01
   ```

   같은 채널이 [`rust-toolchain.toml`](https://github.com/furiosa-ai/furiosa-opt/blob/main/rust-toolchain.toml) 에 고정되어 있다. 그 파일을 포함한 프로젝트로 cd 하면 cargo 가 자동으로 활성화한다.

2. **[`cargo-furiosa-opt`](./appendix/cargo-furiosa-opt.md)**: 알맞은 `--cfg backend="..."` 를 주입하고 NPU 용 커널을 미리 컴파일하는 cargo 서브커맨드.

   ```bash
   cargo +nightly-2026-05-01 install cargo-binstall
   cargo +nightly-2026-05-01 binstall cargo-furiosa-opt
   ```

3. **Furiosa SDK + 실물 NPU** *(`--backend npu` 에만 해당)*: NPU 백엔드는 SDK 의 커널 드라이버와 PE 런타임(`furiosa-driver-rngd`, `furiosa-smi` 등, [SDK 문서](https://developer.furiosa.ai/latest/en/) 참고)을 통해 실제 하드웨어로 디스패치한다.

   **`emulation` 과 `typecheck` 백엔드는 SDK 가 필요 없다.** 이들은 NPU 의존성 없이 호스트 측에서 실행되므로, 커널을 개발하거나 평가만 하려는 고객은 SDK 를 설치할 필요가 전혀 없다.

## 첫 프로그램

[`cargo-generate`](https://cargo-generate.github.io/cargo-generate/) 로 `base-template` 스타터에서 새 프로젝트를 만들어라. 이 스타터에는 [Quick Start](./quick-start.md) 장에서 다루는 다섯 가지 예제가 들어 있다.

```bash
cargo install cargo-generate
cargo generate furiosa-ai/furiosa-opt base-template
cd base-template
```

<a id="layout"></a>
### 레이아웃

```text
base-template/
├── Cargo.toml                               # `[package.metadata.furiosa-opt]` marks it a kernel package
├── README.md
├── rust-toolchain.toml
└── src/
    ├── lib.rs                                # `pub mod kernel;`
    ├── kernel/                               # every #[device] function lives here
    │   ├── mod.rs                            # `pub mod {constant_add,...}_kernel;`
    │   ├── constant_add_kernel.rs            # `#[device] fn constant_add_kernel(...)`
    │   ├── elementwise_mul_kernel.rs
    │   ├── dot_product_kernel.rs
    │   ├── gemv_kernel.rs
    │   └── gemm_kernel.rs
    ├── constant_add.rs                       # host binary that `launch()`es its kernel
    ├── elementwise_mul.rs
    ├── dot_product.rs
    ├── gemv.rs
    └── gemm.rs
```

다음 레이아웃 규칙을 그대로 지켜라.

- 패키지는 `Cargo.toml` 에 `[package.metadata.furiosa-opt]` 를 선언해 커널 컴파일에 참여한다.
- 호스트 프로그램은 `src/*.rs` 파일로 직접 두어야 하며, `Cargo.toml` 의 명시적인 `[[bin]] path = "src/<name>.rs"` 항목으로 등록한다.
- 호스트 프로그램을 `src/bin/`, `examples/`, `tests/` 로 옮기지 마라. rustc 플러그인은 `src/` 를 루트로 하는 cargo 타깃을 스캔하고 그 밖의 위치는 건너뛴다.

다섯 커널 예제는 모두 `src/kernel/` 아래에 있으며 `src/kernel/mod.rs` 와 `src/lib.rs` 를 통해 재수출된다.
각 바이너리의 `main()` 은 `launch(kernel, ...)` 만 호출한다.
호스트 측 기준값과의 값 비교는 같은 파일 안의 `#[cfg(test)] mod tests` 블록에 있다.

### 예제 실행하기

```bash
# Host-side emulation (default; no NPU hardware required).
cargo furiosa-opt run --release --bin gemm

# Mapping/shape verification only — kernel body runs against phantom (empty) tensors.
cargo furiosa-opt --backend typecheck run --release --bin gemm

# Real NPU dispatch (requires the SDK and a physical NPU; see Installation step 3).
cargo furiosa-opt --backend npu run --release --bin gemm
```

### 기준값과 대조하기

```bash
# Full numeric comparison on emulated values.
cargo furiosa-opt test --release --bin gemm

# Under typecheck the comparison loop trivially passes: `actual` is the
# phantom-empty Vec, so the per-element assertion has zero iterations.
cargo furiosa-opt --backend typecheck test --release --bin gemm
```

### 커널 추가하기

1. `#[device(...)] pub fn <name>_kernel(...)` 를 담은 `src/kernel/<name>_kernel.rs` 를 넣는다.
2. `src/kernel/mod.rs` 에 `pub mod <name>_kernel;` 를 덧붙인다.
3. `launch(<name>_kernel, ...)` 를 호출하는 호스트 프로그램으로 `src/<name>.rs` 를 추가한다.
4. `Cargo.toml` 에 `path = "src/<name>.rs"` 를 가진 대응 `[[bin]]` 항목을 등록한다.
5. `cargo furiosa-opt run --release --bin <name>` 로 커널을 실행한다.

## 개발 도구

Furiosa IR Optimizer 는 TCP 장치에서 vISA 프로그램을 개발, 테스트, 최적화하기 위한 유틸리티를 제공한다.
프로그래머가 vISA 를 손으로 쓰든 컴파일러가 생성하든, 개발자에게 프로그램 동작에 대한 세밀한 제어를 주어 [Furiosa SDK 의 컴파일러](https://developer.furiosa.ai/latest/en/overview/software_stack.html#furiosa-compiler)를 보완한다.

<a id="backends"></a>
### 백엔드

vISA 프로그램은 `furiosa-opt-std` API 를 쓰는 Rust 프로그램이다. `cargo furiosa-opt` 는 `--cfg backend="..."` 를 설정해 어떤 백엔드가 커널을 평가할지 고른다.

| 백엔드     | 기본값? | 실행되는 것                                   | 쓰는 때                                                                             |
|-------------|----------|---------------------------------------------|--------------------------------------------------------------------------------------|
| `typecheck` |          | 커널 본문이 팬텀(빈) 텐서로 실행됨 | 매핑/모양 오류를 빠르게 잡을 때(값 계산은 생략)                       |
| `emulation`  | 예      | 실제 버퍼 위에서 호스트 측 완전 해석 | 개발 기본값. 수치적 정확성을 검증                              |
| `npu`       |          | 하드웨어(또는 NVP 시뮬레이터)에서 컴파일된 EDF | 하드웨어 경로까지 포함한 종단 간 실행                                               |

```bash
# Default: emulation backend, no NPU hardware needed.
cargo furiosa-opt run --release

# Fast mapping/shape verification (kernel body runs with phantom tensors).
cargo furiosa-opt --backend typecheck run --release

# Real NPU dispatch (requires the SDK and a physical NPU).
cargo furiosa-opt --backend npu run --release
```

`cargo check` 는 (어떤 백엔드에서든) 타입 검사기만 돌린다. 커널 함수 본문을 실행하지 **않으므로** `Collect output packet must be exactly 32 bytes` 같은 매핑 단언에는 도달할 수 없다. 그런 경우에는 `--backend typecheck run` 을 쓴다.

`cargo furiosa-opt` 는 모든 cargo 플래그를 그대로 전달하므로, `cargo run`, `cargo test`, `cargo check`, `cargo build` 에 모두 직접 대응하는 명령이 있다.

```bash
cargo furiosa-opt build              # cargo build with emulation backend
cargo furiosa-opt --backend npu test # cargo test on real NPU
```

전체 명령 레퍼런스는 [`cargo furiosa-opt` 부록](./appendix/cargo-furiosa-opt.md)을 보라.


### 언어 서버

`furiosa-rust-analyzer-proxy` 는 `rust-analyzer` 의 프록시로, 표준 Rust IDE 기능에 매핑 표현식 지원을 강화해 제공한다.
평소의 `rust-analyzer` 경험을 유지하면서 `Stride<Symbol<A>, 8>` 같은 장황한 타입을 `m![A / 8]` 같은 읽기 쉬운 매핑 표현식으로 단순화한다.

설치와 설정은 [Language Server 부록](./appendix/language-server.md)을 보라.

### Schedule Viewer

Schedule Viewer 는 실행 타임라인을 시각화해 성능 병목을 찾는 데 도움을 준다.
`furiosa-opt` 로 스케줄 JSON 파일을 내보낸 뒤 `furiosa-schedule-viewer` 로 연다.

설치와 사용법은 [Schedule Viewer 부록](./appendix/schedule-viewer.md)을 보라.

## 책의 구성

각 장은 앞 장 위에 쌓인다. 텐서 매핑과 이동이 데이터 모델을 세우고, 텐서 연산이 파이프라인 엔진을 다루며, 스케줄링과 커널 예제가 이들을 실제 프로그램으로 조립하는 방법을 보여준다.

- **[Quick Start](./quick-start.md)**: vISA 프로그래밍이 어떻게 동작하는지를 원소 단위 연산과 텐서 축약을 다루는 예제로 소개한다.
- **[텐서 매핑](./mapping-tensors/index.md)**: 논리 텐서가 물리 메모리로 매핑되는 방식. 축 배치, 스트라이드, 패딩, 타일링.
- **[텐서 이동](./moving-tensors/index.md)**: 데이터가 Fetch, Commit, DMA 엔진을 통해 메모리 계층(HBM, DM)과 Tensor Unit 사이를 오가는 방식.
- **[텐서 연산](./computing-tensors/index.md)**: Tensor Unit 파이프라인(Switch, Collect, Contraction, Vector, Cast, Transpose)이 매 사이클 데이터를 변환하는 방식.
- **[스케줄링](./scheduling.md)**: 연산이 컨텍스트에 걸쳐 정렬되고 동시에 실행되는 방식.
- **[커널 예제](./kernel-examples/index.md)**: 매핑, 이동, 연산, 스케줄링이 실제 커널로 결합되는 모습을 보여주는 종단 간 예제.

## 라이선스

이 문서와 `furiosa-opt` 저장소 전체는 [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) 으로 배포된다.
