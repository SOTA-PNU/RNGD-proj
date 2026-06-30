# 11 · 마무리 실습과 숙달

이 문서는 vISA 커리큘럼 모듈 11입니다. 지금까지 배운 걸 모아 **직접 새 커널을 짜서 검증**합니다. 그리고 vISA의 현실(비공개 컴파일러, EDF 포맷 차이)과 우리 프로젝트(DeltaNet 등)와의 연결, 디버깅 도구(Schedule Viewer)까지 정리해 "이제 혼자 할 수 있다"로 마무리합니다.

## 학습 목표
- [ ] 빈 파일에서 새 `#[device]` 커널 + 호스트 프로그램 + `[[bin]]` 등록을 막힘없이 한다
- [ ] `typecheck → simulation → test` 검증 루프를 몸에 익힌다
- [ ] vISA의 `.bin` 이 furiosa-llm `.edf` 와 다른 포맷·다른 경로임을 설명한다
- [ ] DeltaNet 같은 우리 과제를 vISA로 풀 때 뭐가 되고 뭐가 막히는지 안다
- [ ] Schedule Viewer로 커널 스케줄을 들여다본다

---

## 1. 마무리 실습의 핵심 — "툴체인이 검증해 준다"

지금부터는 제가 정답 코드를 다 드리지 않습니다. 대신 **검증된 예제에서 최소한만 바꿔** 새 커널을 만들고, 툴체인에게 "이게 맞아?"를 물어보는 흐름을 연습합니다. vISA의 가장 큰 장점이 여기 있습니다. 매핑이 틀리면 컴파일이 안 되고(`typecheck`), 값이 틀리면 테스트가 잡습니다(`test`). 그래서 모르는 걸 **겁내지 말고 일단 바꿔서 돌려보는** 게 가장 빠른 학습입니다.

새 커널을 추가하는 절차는 항상 같습니다 (`docs/src/introduction.md:105`, [`../CHEATSHEET.md`](../CHEATSHEET.md) 참고).

1. `experiments/src/kernel/<이름>_kernel.rs` 에 `#[device]` 커널을 쓴다
2. `experiments/src/kernel/mod.rs` 에 `pub mod <이름>_kernel;` 한 줄 추가
3. `experiments/src/<이름>.rs` 에 호스트 프로그램(`launch(...)` + `#[cfg(test)]` 참조 검증)을 쓴다
4. `experiments/Cargo.toml` 에 `[[bin]] name="<이름>" path="src/<이름>.rs"` 를 명시한다 ← **빠지면 플러그인이 무시합니다**(`src/bin/`·`tests/`·`examples/` 아래는 스캔 안 됨)

그리고 검증은 항상 이 순서입니다.

```bash
cd ~/RNGD-proj/Model_Benchmark/rngd-npu/vISA/experiments
cargo furiosa-opt --backend typecheck run --release --bin <이름>   # ① 매핑/모양이 합법인가
cargo furiosa-opt run --release --bin <이름>                       # ② 시뮬레이션으로 실제 값 계산
cargo furiosa-opt test --release --bin <이름>                      # ③ 호스트 참조와 수치 비교
```

> 컴파일이 안 되거나 typecheck가 막으면, 그 오류 메시지가 곧 교재입니다. "어느 파이프라인 전이가 불법인지", "어느 축이 안 나눠떨어지는지"를 그대로 알려 줍니다. 읽고 고치는 게 실력입니다.

---

## 2. 마무리 실습 프로젝트

### 과제 A (쉬움) — 상수 더하기 커널을 새 bin으로 (프로젝트 뼈대 잡기 연습)

목표: `constant_add` 와 똑같은 패턴으로, 내가 고른 상수 K를 더하는 **새 bin** `add_const` 를 만들고 4단계 등록·검증을 끝까지 해보기. 패턴 자체는 검증된 것이라 매핑 걱정이 없습니다 — 목적은 "새 커널을 처음부터 등록하는 손맛"입니다.

`experiments/src/kernel/add_const_kernel.rs` (출처: `base-template/src/kernel/constant_add_kernel.rs` 의 최소 변형):

```rust
use furiosa_opt_std::prelude::*;

axes![A = 2048];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![A / 8 # 256];

#[device(chip = 1)]
pub fn add_const_kernel(ctx: &mut Context, input: &HbmTensor<i32, Chip, m![A]>) -> HbmTensor<i32, Chip, m![A]> {
    let dm = input.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma, 0);
    let result = ctx
        .main
        .begin(dm.view())
        .fetch::<i32, m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, 7)   // ← 여기 상수를 내가 고른 값으로
        .vector_final()
        .commit::<m![A % 8]>(1 << 12);
    result.to_hbm(&mut ctx.tdma, 1 << 28)
}
```

호스트(`experiments/src/add_const.rs`)는 `constant_add.rs` 를 복사해서 `constant_add_kernel` → `add_const_kernel` 로 바꾸고, 참조식의 `wrapping_add(1)` 을 `wrapping_add(7)` 로 맞추면 됩니다. `mod.rs` 에 `pub mod add_const_kernel;`, `Cargo.toml` 에 `[[bin]] name="add_const" path="src/add_const.rs"` 추가. 그다음 위 ①②③ 검증.

성공하면, 새 커널을 0에서 등록하는 전 과정을 한 번 손에 익힌 겁니다.

### 과제 B (중간) — 두 벡터를 더하는 커널 (sub 컨텍스트 + VRF)

목표: `elementwise_mul`(두 벡터 곱)을 **덧셈**으로 바꾼 `vec_add` 를 만들기. 핵심 변화는 딱 한 줄, Vector 연산을 `MulInt` 에서 `AddFxp` 로 바꾸는 것입니다. 이걸로 "한 피연산자는 스트림, 다른 하나는 VRF에 미리 싣는" 패턴을 내 손으로 재현합니다.

`experiments/src/kernel/vec_add_kernel.rs` (출처: `base-template/src/kernel/elementwise_mul_kernel.rs` 의 최소 변형):

```rust
// ... elementwise_mul_kernel.rs 와 동일하게 두되, 함수 이름을 vec_add_kernel 로,
//     그리고 main 컨텍스트의 이 줄만 바꿉니다:
        // .vector_fxp(FxpBinaryOp::MulInt, &rhs_vrf)   // 원래(곱)
        .vector_fxp(FxpBinaryOp::AddFxp, &rhs_vrf)      // 바꾼 것(합)
```

호스트(`vec_add.rs`)는 `elementwise_mul.rs` 를 복사해 이름만 바꾸고, 참조식 `wrapping_mul(b)` → `wrapping_add(b)` 로. 등록 후 ①②③ 검증.

> 만약 `typecheck` 가 `AddFxp` + VRF 피연산자 조합을 거부하면, 그 오류를 읽는 것도 학습입니다. `FxpBinaryOp` 의 어떤 변종이 즉값(immediate)·VRF 피연산자 중 무엇을 받는지 `furiosa-opt-std/src/engine/vector/op/mod.rs` 에서 확인하세요. 거부당하지 않으면 `test` 로 수치까지 맞는지 봅니다. **이게 vISA를 배우는 정상 루프입니다 — 가설을 세우고, 툴체인에게 묻고, 메시지로 배웁니다.**

### 과제 C (도전) — 실제 RMSNorm 읽고 돌리고 변형하기

처음부터 RMSNorm을 짜는 건 타일링·재배치가 많아 어렵습니다. 대신 **실제 예제를 읽고 → 돌리고 → 한 군데 바꿔보는** 게 현실적입니다.

- 읽기: `furiosa-opt-examples/src/transformer/common/norm.rs`, `.../embedding/rms_norm.rs`. RMSNorm이 `residual add(VE) → variance → reciprocal sqrt(`FpUnaryOp::Sqrt`) → per-channel weight 곱` 으로 분해되는 걸 따라가세요. 07 모듈에서 본 Vector 엔진 패턴이 그대로 나옵니다.
- 돌리기: 상위 `~/furiosa-opt` 저장소에서 트랜스포머 테스트를 시뮬레이션/타입체크로 돌려봅니다(`cargo furiosa-opt --backend typecheck test -p furiosa-opt-examples`).
- 변형: `eps` 상수나 타일 크기 한 군데를 바꿔 typecheck가 어떻게 반응하는지 관찰하세요.

---

## 3. 꼭 알아야 할 현실 — 비공개 컴파일러와 EDF 포맷

vISA를 제대로 익히려면 그 경계도 알아야 합니다. (자세히는 [`../../../info/README_virtual_isa.md`](../../../info/README_virtual_isa.md) §2, §7)

- **컴파일러는 비공개입니다.** 우리가 쓰는 `cargo-furiosa-opt` 는 미리 빌드된 바이너리(`npu_opt` 플러그인)이고 소스가 없습니다(`cargo-furiosa-opt/src/main.rs` 는 5줄짜리 stub). vISA 언어·타입시스템·시뮬레이터·예제·매뉴얼만 공개됐고, 실제 `MIR→VISA→LIR→EDF` 코드젠은 블랙박스입니다.
- **vISA의 "EDF"는 furiosa-llm의 `.edf` 와 다른 포맷입니다.** vISA가 만드는 `.bin` 은 pert-ipc 명령어 스트림이고 `libdevice_runtime.so` 가 로드합니다. furiosa-llm의 `.edf` 는 CBOR 그래프 IR이라 **서로 못 읽습니다**(직접 까서 확인). 둘은 같은 칩(`librenegade.so`/`/dev/rngd`)에서 돌지만 **다른 실행 경로**입니다.
- **그래서**: vISA로 짠 커널을 우리가 쓰던 furiosa-llm serve 파이프라인이나 masquerade(artifact.json model_type 위장)에 끼워 넣을 수 **없습니다**. vISA는 "정식 serve를 우회해 칩에 커널을 직접 올리는 별도의 길"이지, "serve에 커스텀 연산을 주입하는 길"이 아닙니다.
- **알파입니다.** 예제에 TODO·`todo!()` 가 남아 있고(`attention.rs` 미완성, `run_qwen` ignored 등), x86_64 리눅스·nightly-2026-05-01 에 고정돼 있습니다.

---

## 4. 우리 프로젝트와의 연결 — DeltaNet/recurrent

vISA가 우리 [[qwen3-next-blocker]] 작업에 주는 의미를 정리합니다. (출처: 분석문서 §8 + 퓨리오사 공식 답변)

**열리는 것**
- op 지원 DB·model_type·serve 게이트를 **전부 우회**합니다. gather/scatter가 1급 기본 연산(`dma_gather`/`dma_scatter`)이라, 고수준 컴파일러가 거부하던 연산을 엔진 직접 호출로 표현할 수 있습니다.
- DeltaNet류 recurrent의 재료가 다 있습니다 — Contraction 엔진, Vector 엔진의 게이팅 비선형(`Sigmoid`/`Exp` 등), 데이터 의존 분기(Tag/VCG), 그리고 **상태를 TRF/VRF/DM의 고정 주소에 두고 호스트 루프로 도는** 방식.

**막히는 것 (그래서 "다리"입니다)**
- **정적 shape 강제** — 가변 길이 시퀀스/상태는 최악 크기로 미리 패딩해야 합니다(동적 shape는 퓨리오사 1년 로드맵).
- **온칩 순환상태 부재** — 매 단계 host 왕복 없이 칩 안에서 상태를 들고 도는 Persistent Kernel은 미구현(퓨리오사 2026 목표). 그래서 지금은 우리 host-loop처럼 host 오케스트레이션을 직접 짜야 합니다. 흥미롭게도 **퓨리오사가 말한 Persistent Kernel = 우리 host-loop의 칩 내재화** 입니다. 즉 우리 아키텍처가 퓨리오사의 중간 단계 설계와 같은 방향이고, 기능이 도착하면 갈아끼우는 구조입니다.
- **LLM 서빙 스택이 전무** — 디코드 루프·KV/상태 풀·샘플러·연속배치·토크나이저가 없어 전부 자작해야 합니다.

한 줄로: vISA는 **저수준에서 연산을 손으로 뚫는 새 길**이지만, "serve로 바로 배포"가 아니라 "별도 device-runtime로 커널 직접 구동 + 호스트 하네스 자작"이 현실적인 그림입니다.

---

## 5. 도구 — Schedule Viewer와 LSP

- **Schedule Viewer** (퓨리오사 공식, 2026-06): 컴파일된 커널의 엔진 타임라인·병렬 연산·텐서 배치/shape/SRAM 수명을 GUI로 봅니다. vISA(`.bin`) 경로 전용 가시화 도구예요.
  ```bash
  cargo furiosa-opt compiler build --dump-schedule <out.json>
  cargo install furiosa-schedule-viewer
  furiosa-schedule-viewer          # http://127.0.0.1:9254
  ```
  커널이 왜 느린지(엔진이 놀고 있는지, 이중 버퍼링이 겹치는지)를 눈으로 확인할 때 씁니다.
- **LSP 프록시** (`furiosa-rust-analyzer-proxy`): `Stride<Symbol<A>, 8>` 같은 장황한 타입을 `m![A / 8]` 로 보여 줍니다. 매핑을 많이 다룰수록 필수입니다(설정: `docs/src/appendix/language-server.md`).

---

## 6. 숙달 체크리스트 & 다음 단계

[`../PROGRESS.md`](../PROGRESS.md) 의 "숙달 신호"와 같습니다. 다 체크되면 "안다"고 봐도 됩니다.

- [ ] 빈 파일에서 새 `#[device]` 커널 + 호스트 + `[[bin]]` 등록을 막힘없이 한다 (과제 A·B)
- [ ] 타입 오류 메시지를 보고 어느 파이프라인 전이/매핑이 불법인지 바로 짚는다
- [ ] 임의의 einsum(예: `BHQK`)을 보고 어떤 축을 Slice/Lane/Time/Packet에 둘지 설계한다
- [ ] softmax/RMSNorm을 Vector 엔진 op로 어떻게 조립하는지 트레이스한다 (07·10)
- [ ] vISA `.bin` ≠ furiosa-llm `.edf` 와 그 의미(serve 우회 불가)를 설명한다
- [ ] DeltaNet을 vISA로 풀 때 되는 것·막히는 것(정적 shape, Persistent Kernel 부재)을 설명한다

다음 단계:
- 책에서 아직 안 깊게 본 챕터를 정독: `docs/src/computing-tensors/contraction-engine/2d-convolution.md`(CNN), `docs/src/kernel-examples/mixture-of-experts.md`(MoE 라우팅).
- 실제 작업 도전: `furiosa-opt-examples` 의 matmul 변형을 직접 한 단계 바꿔보거나, MNIST forward(`src/mnist/mod.rs`)를 읽고 작은 MLP를 변형해 보세요.
- 우리 과제로 가져오기: DeltaNet 한 단계(recurrent update)을 vISA 커널 하나로 표현해보고 `typecheck` 까지 통과시키는 걸 목표로.

수고하셨습니다. 여기까지 왔으면 vISA를 "읽고·돌리고·직접 짜는" 출발선에 선 겁니다. 🎯

➡️ 처음으로: [README](../README.md) · 진도표: [PROGRESS](../PROGRESS.md)
