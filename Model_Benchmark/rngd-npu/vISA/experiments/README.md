# vISA 실험장 (experiments)

이 폴더는 **직접 돌려보는 vISA 커널 모음**입니다. 퓨리오사 공식 `base-template`을 그대로 가져와(검증된 코드) 실행 가능한 cargo 프로젝트로 만든 것이고, 커리큘럼 모듈을 따라가며 여기에 커널을 추가·수정합니다.

> ⚠️ 먼저 `../00_SETUP.md` 를 끝내야 합니다. 툴체인(nightly-2026-05-01)과 `cargo-furiosa-opt`(닫힌 컴파일러 바이너리)가 깔려 있어야 아래 명령이 돕니다. **NPU는 필요 없습니다** — 기본 `simulation` 백엔드는 호스트 CPU에서 돌아가요.

## 들어있는 커널 (기본 5종, base-template 원본)

| bin 이름 | 무엇을 하나 | 배우는 개념 | 관련 모듈 |
|---|---|---|---|
| `constant_add` | 벡터의 모든 원소에 1 더하기 | Fetch→Collect→Vector→Commit, `to_dm`/`to_hbm` | 03 |
| `elementwise_mul` | 두 벡터 원소별 곱 | `sub` 컨텍스트로 VRF 프리로드 | 03 |
| `dot_product` | 두 벡터 내적 | Contraction 엔진, TRF | 04 |
| `gemv` | 행렬×벡터 | Switch 브로드캐스트 | 04 |
| `gemm` | 행렬×행렬 | 출력 타일을 슬라이스에 분산, Lane Folder | 04 |

## 실행 방법

```bash
cd experiments

# 1) 시뮬레이션으로 실행 (기본 백엔드, NPU 불필요, 실제 값 계산)
cargo furiosa-opt run --release --bin gemm

# 2) 모양/매핑만 빠르게 검사 (커널 본문을 빈 텐서로 실행 → 매핑 단언까지 확인)
cargo furiosa-opt --backend typecheck run --release --bin gemm

# 3) 호스트 레퍼런스와 수치 비교 (정답 검증)
cargo furiosa-opt test --release --bin gemm

# 참고: 그냥 cargo 로도 시뮬레이션이 됩니다 (build.rs 가 기본 backend=simulation 주입)
cargo run --release --bin constant_add
```

`cargo furiosa-opt` 는 `--cfg backend="..."` 만 끼워 넣고 나머지 cargo 플래그는 그대로 전달합니다. 그래서 `run`/`test`/`check`/`build` 전부 백엔드별로 됩니다.

> `cargo check` 는 타입검사만 하고 커널 본문을 **실행하지 않습니다**. 그래서 `"Collect output packet must be exactly 32 bytes"` 같은 **런타임 매핑 단언**까지 보려면 `--backend typecheck run` 을 쓰세요. (출처: `docs/src/introduction.md:133`)

## 새 커널 추가하는 법 (load-bearing 규칙 주의)

1. `src/kernel/<이름>_kernel.rs` 에 `#[device]` 커널 작성
2. `src/kernel/mod.rs` 에 `pub mod <이름>_kernel;` 추가
3. `src/<이름>.rs` 에 호스트 프로그램(`launch(...)`) 작성
4. **`Cargo.toml` 에 `[[bin]] name="<이름>" path="src/<이름>.rs"` 명시** ← 이게 핵심

플러그인은 `src/` 에 뿌리내린 cargo 타겟만 스캔하고 `src/bin/`·`examples/`·`tests/` 아래는 **조용히 건너뜁니다**. 그래서 `[[bin]] path="src/..."` 선언이 반드시 필요합니다. (출처: `docs/src/introduction.md:63,79`)

## 더 큰 예제는 어디에?

MNIST(완전 검증된 실제 NN), matmul 16384, Qwen2.5-0.5B 트랜스포머 분해 등 큰 예제는 상위 저장소에 있습니다. `00_SETUP.md` 에서 클론하는 `furiosa-opt` 저장소의 `furiosa-opt-examples/` 에서 돌리세요. 읽기용 복사본은 `../reference/examples/` 에도 있습니다.

```bash
# (furiosa-opt 저장소를 클론한 경우)
cd furiosa-opt
cargo furiosa-opt test --release -p furiosa-opt-examples --test mnist_tests   # 시뮬레이션 검증
```
