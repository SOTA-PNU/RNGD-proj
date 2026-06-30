# 00 · 환경 구축 + 첫 커널 돌리기

이 문서는 vISA 개발환경 설치 안내입니다. NPU 없이도 커널을 컴파일·시뮬레이션할 수 있는 환경을 만들고, 끝나면 `constant_add` 커널이 도는 걸 봅니다.

> 핵심 사실 먼저: **개발·검증은 NPU·SDK 없이 됩니다.** `simulation`(호스트에서 실제 값 계산)과 `typecheck`(모양/매핑만 검사) 백엔드는 하드웨어가 필요 없어요. NPU는 `--backend npu` 일 때만 필요합니다. (출처: `docs/src/introduction.md:40-44`)

---

## 1. 필요한 것 3가지

### (1) Rust nightly — 버전 고정 필수
`cargo-furiosa-opt`는 rustc 드라이버라 **특정 nightly에 ABI가 묶여 있습니다.** 반드시 이 버전:

```bash
rustup toolchain install nightly-2026-05-01
```

저장소/실험 폴더에 `rust-toolchain.toml`(channel = nightly-2026-05-01)이 있어서, 그 폴더로 `cd` 하면 cargo가 자동으로 이 toolchain을 씁니다. 따로 `+nightly...` 안 붙여도 됩니다. (출처: `rust-toolchain.toml:1-3`, `docs/src/introduction.md:25-31`)

### (2) OS 의존성 (Ubuntu jammy/noble/resolute)

```bash
sudo apt install libclang-dev gcc-aarch64-linux-gnu
```

- `libclang-dev` — 빌드 시 `furiosa-opt-std/build.rs`가 `bindgen`을 돌리는데 `libclang.so`가 필요. **모든 백엔드에서 빌드할 때 필요** (npu 모듈이 항상 컴파일되므로).
- `gcc-aarch64-linux-gnu` — `--backend npu`로 NPU용 `.bin`을 만들 때 `aarch64-linux-gnu-{gcc,as,ld,objcopy}`를 호출. (시뮬레이션만 할 거면 당장은 없어도 되지만, 깔아두면 편함) (출처: `README.md:28-33`)

### (3) 비공개 컴파일러 바이너리 `cargo-furiosa-opt`
실제 vISA→EDF 컴파일러는 소스가 없고 **미리 빌드된 바이너리**로만 배포됩니다(x86_64 리눅스만). `cargo-binstall`로 받습니다:

```bash
cargo install cargo-binstall
cargo binstall cargo-furiosa-opt
```

> 이건 GitHub 릴리스의 63MB tgz를 받아 깝니다. 소스 빌드는 막혀 있어요(`disabled-strategies=["compile"]`). (출처: `cargo-furiosa-opt/Cargo.toml:17-21`)

### (선택) 새 프로젝트 뼈대 만들기용 `cargo-generate`

```bash
cargo install cargo-generate
# 빈 프로젝트를 처음부터 만들고 싶을 때:
cargo generate furiosa-ai/furiosa-opt base-template
```

이 커리큘럼은 이미 `experiments/`에 뼈대를 만들어 뒀으니, 당장은 없어도 됩니다.

---

## 2. 첫 커널 돌리기 (NPU 불필요)

```bash
cd experiments

# 시뮬레이션 실행 (기본 백엔드, 실제 값 계산)
cargo furiosa-opt run --release --bin constant_add
#  → "Constant Add: kernel ran" 출력되면 성공

# 호스트 레퍼런스와 수치 비교 (정답 검증)
cargo furiosa-opt test --release --bin constant_add
#  → test result: ok 나오면 성공

# 모양/매핑만 빠르게 검사 (커널 본문을 빈 텐서로 실행)
cargo furiosa-opt --backend typecheck run --release --bin constant_add
```

> 첫 빌드는 **인터넷이 필요**합니다. `furiosa-mapping`의 build.rs가 비공개 `libfuriosa_mapping_impl.a`(15MB)를 GitHub 릴리스에서 받아 SHA256 검증 후 정적 링크하거든요. (출처: `furiosa-mapping/build.rs:18-63`)

---

## 3. (이 머신 한정) 실제 NPU에서 돌리기

이 서버엔 RNGD 4장 + SDK가 있으니 `--backend npu`도 됩니다.

```bash
# 칩 0,1 을 골라 NPU에서 실행
FURIOSA_OPT_NPUS=0,1 cargo furiosa-opt --backend npu run --release --bin gemm
```

- `FURIOSA_OPT_NPUS` = 쓸 칩 ID(콤마 구분). 비우면 칩 0. (출처: `furiosa-opt-std/src/runtime/npu/ffi.rs:35-60`)
- 토폴로지 제약: 칩마다 **양쪽 half-cluster를 pe0-7로 통째 묶는 것만** 됩니다. pe0-3 같은 좁은 분할은 "Invalid device ID"로 실패. (출처: `ffi.rs:49-83`)

---

## 4. IDE 지원 (강력 추천)

매핑 타입은 `Stride<Symbol<A>, 8>`처럼 길게 나오는데, **`furiosa-rust-analyzer-proxy`**가 이걸 `m![A / 8]`처럼 읽기 좋게 바꿔서 보여줍니다. rust-analyzer 앞단에 끼우는 프록시예요.

```bash
cargo binstall furiosa-rust-analyzer-proxy   # 또는 릴리스에서 직접 다운로드
```

설정법: `docs/src/appendix/language-server.md` (= `reference/book/appendix/language-server.md`).

---

## 5. (선택) 책을 로컬에서 읽기

```bash
cargo install mdbook mdbook-mermaid mdbook-pdf
# 상위 저장소 클론 후:
cd furiosa-opt && mdbook serve docs --open
```

읽기만 할 거면 `reference/book/`의 마크다운을 그냥 열어도 됩니다.

---

## 6. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `error: toolchain 'nightly-2026-05-01' is not installed` | `rustup toolchain install nightly-2026-05-01` |
| `cargo furiosa-opt: command not found` | `cargo binstall cargo-furiosa-opt` 안 됨 → `cargo-binstall` 먼저 설치, PATH에 `~/.cargo/bin` 확인 |
| 빌드 중 `libclang` 못 찾음 | `sudo apt install libclang-dev` |
| build.rs 가 `.a` 다운로드 실패(네트워크) | 오프라인이면 `FURIOSA_MAPPING_IMPL_LOCAL_PREBUILT=/경로/libfuriosa_mapping_impl.a` 환경변수로 로컬 `.a` 지정 (출처: `furiosa-mapping/build.rs:18-30`) |
| 커널을 추가했는데 플러그인이 무시함 | `[[bin]] path="src/<name>.rs"` 를 `Cargo.toml`에 명시했는지 확인. `src/bin/` 아래면 스캔 안 됨 (출처: `docs/src/introduction.md:63,79`) |
| `cargo check`는 통과인데 매핑 단언을 못 봄 | check는 본문 실행 안 함 → `--backend typecheck run` 사용 (출처: `docs/src/introduction.md:133`) |
| ABI 관련 이상한 rustc 패닉 | toolchain이 정확히 `nightly-2026-05-01`인지 확인(다른 nightly면 드라이버와 안 맞음) |

---

준비 끝났으면 **`curriculum/01_mental_model.md`** 로 갑니다. (아직 작성 중이면 README의 학습 경로 표를 참고하세요.)
