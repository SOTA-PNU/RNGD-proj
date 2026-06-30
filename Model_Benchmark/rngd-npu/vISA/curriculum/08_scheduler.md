# 08 · 스케줄러와 동시성

이 문서는 vISA 커리큘럼 모듈 08입니다. main/sub/tdma/pdma 컨텍스트가 어떻게 병렬로 도는지, 해저드(RAW/WAR/WAW)와 수동 주소 배정, 이중 버퍼링으로 연산과 데이터 이동을 겹치는 법을 배웁니다.
*선행: 03~07 (특히 03의 sub 컨텍스트) · 예상 시간: 반나절*

## 학습 목표

- [ ] 네 컨텍스트(main/sub/tdma/pdma)의 역할과 병렬 규칙을 안다
- [ ] 스케줄러가 작성 순서를 지키고 재배치하지 않음을 안다
- [ ] RAW/WAR/WAW 해저드를 주소로 어떻게 잡는지 안다
- [ ] TRF 반쪽 이중 버퍼링 패턴을 안다

## 1. 개념

## 큰 그림: vISA 스케줄러는 두 가지만 본다

vISA에서 프로그램을 하드웨어 실행 일정으로 바꿀 때, 스케줄러가 보는 정보는 딱 두 가지입니다. (1) 우리가 코드에 적은 연산의 "순서"와 (2) 우리가 직접 지정한 "메모리 주소"입니다(`docs/src/scheduler.md:4`). 이게 핵심입니다. 보통의 GPU/CPU 컴파일러는 알아서 연산을 재배치(reorder)하고 메모리도 알아서 할당해 주지만, vISA 스케줄러는 **적은 순서를 그대로 따르고(재배치 안 함)**, 메모리도 **자동으로 안 잡아 줍니다**. 그래서 동시성과 메모리 안전성의 책임이 상당 부분 프로그래머에게 있습니다. 대신 그만큼 하드웨어 동작이 예측 가능해집니다.

예제(`docs/src/scheduler.md:12`)를 보면:

```rust,ignore
let t0 = load_from_host();  // O0
let t1 = load_from_host();  // O1
let t2 = t0.op();           // O2
let t3 = t1.op();           // O3
let t4 = t2.op();           // O4
let t5 = t4.op();           // O5
```

최종 실행 순서는 적힌 그대로 `O0 → O1 → O2 → O3 → O4 → O5`입니다. 스케줄러는 이 "쓰여진 순서(written order)"를 권위 있는 시퀀스로 취급합니다. 즉 코드 줄 순서가 곧 명령 발행 순서입니다.

## 실행 컨텍스트(Execution Context): 병렬성이 나오는 곳

그럼 다 직렬이면 어떻게 빨라질까요? 답은 "컨텍스트"입니다. 하드웨어는 **서로 독립적으로 동시에 돌 수 있는 실행 스트림**을 여러 개 노출합니다. 문서는 이를 개념적으로 세 개 — main, sub, DMA — 로 설명합니다(`docs/src/scheduler.md:33`, `docs/src/computing-tensors/index.md:64`). 그런데 실제 코드(`furiosa-opt-std/src/context.rs:57`)의 `Context` 구조체를 열어 보면 필드가 네 개입니다:

```rust
pub struct Context {
    pub main: TuContext<{ Tu::Main }>,   // 메인 텐서 유닛
    pub sub:  TuContext<{ Tu::Sub }>,    // 서브 텐서 유닛
    pub tdma: DmaContext<{ Dma::Tensor }>, // 텐서 DMA (HBM <-> DM)
    pub pdma: DmaContext<{ Dma::Pcie }>,   // PCIe DMA (호스트 <-> HBM)
}
```

정리하면 이렇습니다. **main**과 **sub**는 둘 다 "텐서 유닛(Tensor Unit)" 파이프라인을 구동하는 컨텍스트입니다(`furiosa-opt-std/src/context.rs:20`의 `enum Tu { Main, Sub }`). **DMA**는 다시 두 엔진으로 나뉩니다(`furiosa-opt-std/src/context.rs:29`의 `enum Dma { Tensor, Pcie }`): `tdma`(텐서 DMA, 칩 안에서 HBM↔DM 옮기는 일)와 `pdma`(PCIe DMA, 호스트 메모리↔HBM 옮기는 일)입니다. 그래서 "세 컨텍스트"라고 말하지만 자원으로 보면 main/sub/tdma/pdma 네 갈래가 동시에 굴러갈 수 있습니다.

각 컨텍스트가 무엇을 할 수 있는지(`docs/src/computing-tensors/index.md:69`):
- **main**: 텐서 유닛의 8개 엔진(Fetch→Switch→Collect→Contraction→Vector→Cast→Transpose→Commit) 전부를 구동합니다. 커널의 주 연산이 여기서 돕니다.
- **sub**: 같은 파이프라인의 "부분집합"을 구동합니다. **Contraction 엔진과 몇몇 기능이 빠집니다**(`docs/src/computing-tensors/index.md:74`). 보통은 main이 현재 배치를 계산하는 동안 다음 피연산자를 TRF/VRF로 미리 당겨오는(prefetch) 역할입니다.
- **DMA(tdma/pdma)**: 텐서 유닛 바깥의 DMA 엔진만 굴립니다. HBM↔DM, HBM↔SPM, DM↔SPM 같은 대용량 이동을 둘 중 어느 텐서 유닛 컨텍스트와도 무관하게 동시에 합니다.

### 왜 Rust 필드로 나눠 놨을까 (WHY)

`Context::acquire()`(`furiosa-opt-std/src/context.rs:72`)는 전역 싱글톤을 `Mutex`로 감싸 하나만 돌려줍니다. 그런데 우리는 한 함수 안에서 `ctx.tdma`로 DMA를 돌리고, `ctx.sub.begin(...)`로 서브 작업을 시작하고, `ctx.main.begin(...)`로 메인 작업을 동시에 적습니다. 이게 가능한 이유는 Rust의 "필드별 분리 빌림(disjoint borrows)" 덕분입니다. `ctx.main`, `ctx.sub`, `ctx.tdma`, `ctx.pdma`는 서로 다른 필드라서 각각 `&mut`로 독립적으로 빌릴 수 있습니다. 즉 **소스 코드에서 서로 다른 컨텍스트를 동시에 빌리는 모양**이 그대로 **하드웨어에서 서로 다른 컨텍스트가 동시에 도는 것**과 대응됩니다. 타입 시스템이 동시성의 표현 수단인 셈입니다.

## 동시성 규칙: 같은 컨텍스트는 줄 서고, 다른 컨텍스트는 같이 간다

스케줄러는 각 컨텍스트를 독립적인 연산 스트림으로 봅니다(`docs/src/scheduler.md:35`). 규칙은 단순합니다.
- **같은 컨텍스트의 연산들은 직렬화(serialize)** 됩니다. 예: 행렬곱 두 개가 모두 main 컨텍스트를 필요로 하면 차례로 실행됩니다.
- **다른 컨텍스트의 연산들은 병렬(parallel)** 로 돕니다. 예: 행렬곱(main)과 DMA 전송(DMA)은 서로 다른 컨텍스트라 동시에 돕니다(`docs/src/scheduler.md:39`).

연산이 직렬화되는 원인은 두 가지뿐입니다(`docs/src/scheduler.md:37`):
1. **자원 충돌(resource conflict)**: 두 연산이 같은 컨텍스트를 써야 할 때, 뒤엣것이 기다립니다.
2. **메모리 의존성(memory dependency)**: 공유된 주소에서 데이터 위험(hazard)이 생길 때.

### 데이터 위험(hazard) 3종 — RAW/WAR/WAW

서로 다른 컨텍스트라서 원래는 동시에 돌 수 있어도, 같은 메모리 주소를 건드리면 순서를 지켜야 합니다(`docs/src/scheduler.md:43`):
- **RAW (read-after-write)**: 읽기는 앞선 쓰기의 결과를 봐야 한다. (예: sub가 VRF에 써넣은 값을 main이 읽음 → main은 sub가 끝날 때까지 기다림.)
- **WAR (write-after-read)**: 아직 읽는 중인 데이터를 덮어쓰면 안 된다.
- **WAW (write-after-write)**: 같은 주소에 대한 쓰기는 적힌 순서대로 실행돼야 한다.

스케줄러는 **프로그램에 적힌 메모리 주소를 분석해서** 이 위험들을 탐지하고(`docs/src/scheduler.md:49`), 필요한 곳에 자동으로 대기(implicit wait)를 끼워 넣습니다(`docs/src/scheduler.md:51`). 그래서 프로그래머가 동기화 배리어(barrier)를 손으로 넣지 않아도 됩니다. 다만 **주소만큼은 여전히 직접 지정**해야 합니다 — 위험 분석의 입력이 바로 그 주소이기 때문입니다.

이것이 "자동 동기화(main이 sub를 기다린다)"의 정확한 메커니즘입니다. main이 무조건 sub를 기다리는 게 아니라, **둘이 같은 주소에서 RAW가 생길 때** 스케줄러가 main 쪽에 대기를 넣어 주는 것입니다. 주소가 안 겹치고 위험이 없으면 둘은 그냥 같이 돕니다.

### 실제 커널에서 보는 자동 동기화 — elementwise_mul

`base-template/src/kernel/elementwise_mul_kernel.rs`를 줄 단위로 읽어 봅시다(이 파일은 `experiments/` 프로젝트에도 동일하게 들어 있어 바로 실행 가능합니다):

```rust
// 1) tdma로 두 피연산자를 HBM -> DM 으로 이동. 서로 다른 기준 주소!
let lhs_dm = lhs.to_dm::<...>(&mut ctx.tdma, 0);        // DM 주소 0
let rhs_dm = rhs.to_dm::<...>(&mut ctx.tdma, 1 << 12);  // DM 주소 4096

// 2) sub 컨텍스트: rhs 를 VRF(주소 0)로 적재 — main 과 동시에 돈다고 주석에 명시
let rhs_vrf = ctx.sub.begin(rhs_dm.view())
    .fetch::<...>().collect::<...>()
    .to_vrf(0);

// 3) main 컨텍스트: lhs 의 각 원소를 VRF 의 rhs 와 곱함
let result = ctx.main.begin(lhs_dm.view())
    .fetch::<...>().collect::<...>()
    .vector_init().vector_intra_slice_tag(TagMode::Zero)
    .vector_fxp(FxpBinaryOp::MulInt, &rhs_vrf)  // <-- VRF 를 여기서 "읽음"
    .vector_final()
    .commit::<...>(1 << 13);                      // DM 주소 8192 로 결과 쓰기

result.to_hbm(&mut ctx.tdma, 1 << 28)             // 결과 DM -> HBM
```

여기서 두 가지 동기화가 일어납니다. 첫째, `tdma`가 `rhs_dm`(DM 주소 4096)에 써 넣은 값을 sub가 `fetch`로 읽으므로 RAW가 생기고, sub는 그 DMA가 끝나길 기다립니다. 둘째, sub가 `to_vrf(0)`로 VRF에 써 넣은 `rhs_vrf`를 main이 `vector_fxp(..., &rhs_vrf)`로 읽으므로 또 RAW가 생기고, **main은 sub의 VRF 채우기가 끝날 때까지 기다립니다**. 이게 바로 "main이 sub를 기다린다"의 구체적 사례입니다. 코드에는 어떤 `wait()`도 없지만 주소 분석만으로 스케줄러가 알아서 끼워 넣습니다. 한편 두 DM 주소를 0과 4096으로 "일부러 다르게" 둔 이유(주석: "use distinct base addresses to avoid overlap")는 겹치면 위험/덮어쓰기가 생기기 때문입니다.

gemm/gemv 커널도 같은 모양입니다(`base-template/src/kernel/gemm_kernel.rs:25`, `gemv_kernel.rs:24`): sub가 한 피연산자를 TRF에 채우고, main이 Contraction으로 그 TRF를 읽습니다. sub는 Contraction을 못 하니(부분집합) 적재만 맡고, 계산은 main이 하는 자연스러운 분업입니다.

## 수동 SRAM 주소 지정 — 자동 할당이 없다

vISA는 **텐서마다 정확한 메모리 주소가 필요하고, 그 주소를 지금은 프로그래머가 직접 지정**해야 합니다(`docs/src/scheduler.md:24`). 자동 할당기(allocator)가 없습니다. 주소를 받는 API들을 보면:
- `to_dm(dma, address)` — HBM→DM, 두 번째 인자가 DM 주소(`u64`). (`furiosa-opt-std/src/tensor/memory.rs:423`)
- `to_vrf(address)` — Collect→VRF, raw 주소(`u64`). (`furiosa-opt-std/src/engine/collect.rs:72`)
- `to_trf(TrfAddress)` — Collect→TRF, 단 raw 숫자가 아니라 `enum TrfAddress { FirstHalf, SecondHalf, Full }`. (`furiosa-opt-std/src/engine/collect.rs:56`, enum은 `tensor/memory.rs:101`)
- `commit(address)` — 파이프라인 결과를 DM에 쓰기, raw 주소. (`furiosa-opt-std/src/engine/commit.rs:27`)
- `to_hbm(dma, address)` — DM→HBM. (`furiosa-opt-std/src/tensor/memory.rs:387`)

`Address`는 그냥 `u64`입니다(`furiosa-opt-std/src/tensor/memory.rs:21`). 즉 우리가 "이 텐서는 DM 0번지, 저 텐서는 4096번지" 하고 손으로 번지수를 박는 것입니다.

### 왜 주소가 겹치면 안 되나 (WHY)

스케줄러의 위험 분석은 "같은 주소면 의존성"이라는 규칙으로 돕니다. 그런데 서로 다른 두 텐서가 **우연히 같은(또는 겹치는) 주소**를 쓰면 두 가지 문제가 동시에 터집니다. (1) 실제 하드웨어 메모리에서 한 텐서가 다른 텐서를 **덮어써** 데이터가 깨지고, (2) 스케줄러는 그걸 "의도된 의존성"으로 오해해 동시에 돌릴 것을 직렬화하거나, 반대로 의도치 않은 WAR/WAW로 결과가 손상됩니다. 자동 할당기가 없으니 "겹치지 않게 배치"하는 건 전적으로 프로그래머 몫입니다. elementwise_mul이 DM을 0/4096/8192로 띄워 둔 것, gemm이 0/`1<<12`로 나눈 것이 다 이 이유입니다.

### TRF는 절반으로 나뉜다, VRF는 자유 분할

TRF(Tensor Register File)는 각 뱅크를 두 절반으로 나눕니다(`docs/src/computing-tensors/register-files.md:90`). `TrfAddress`의 세 모드: `Full`은 128행 전부(65,536바이트), `FirstHalf`는 0–63행, `SecondHalf`는 64–127행이며 절반 모드는 슬라이스당 40KB로 용량이 제한됩니다(enum 용량은 `furiosa-opt-std/src/tensor/memory.rs:111`). 반면 VRF는 하드웨어가 강제로 절반을 나누지 않습니다(`docs/src/scheduler.md:60`). 각 슬라이스의 8KB VRF를 여러 텐서가 자유롭게 쪼개 쓰고, 이중 버퍼링을 원하면 프로그래머가 "겹치지 않는 영역"을 직접 잡아 줘야 합니다(이게 또 "수동 주소"의 한 단면입니다).

## 이중 버퍼링(Double-Buffering) — 기다림을 없애는 패턴

elementwise_mul처럼 "sub가 채우고 같은 회차에 main이 읽으면" RAW 때문에 main이 sub를 기다립니다. 이 기다림을 없애는 게 이중 버퍼링입니다(`docs/src/scheduler.md:55`). 아이디어: TRF를 두 절반으로 나눠, **sub는 한쪽 절반에 "다음" 배치를 채우고, 동시에 main은 다른 쪽 절반에서 "지금" 배치를 읽습니다.** 회차마다 두 절반을 번갈아(swap) 씁니다.

```rust,ignore
// 루프 전에 첫 절반을 미리 채움
let mut trf = ctx.sub.begin(weights[0].view()).fetch::<...>().collect::<...>()
    .to_trf(TrfAddress::FirstHalf);

for i in 0..N {
    // main 이 현재 절반을 읽는 동안 sub 는 "반대쪽" 절반에 다음 배치를 미리 적재
    let other_half = if i % 2 == 0 { TrfAddress::SecondHalf } else { TrfAddress::FirstHalf };
    let next_trf = (i + 1 < N).then(|| {
        ctx.sub.begin(weights[i + 1].view()).fetch::<...>().collect::<...>()
            .to_trf(other_half)
    });

    ctx.main.begin(input[i].view()).contract_outer::<...>(&trf)...; // 현재 절반 사용

    if let Some(t) = next_trf { trf = t; }
}
```

왜 이게 겹쳐 도느냐(WHY): sub의 쓰기와 main의 읽기가 **서로 다른 TRF 절반(다른 주소)** 을 건드리므로 WAR 위험이 없고(`docs/src/scheduler.md:91`), 동시에 sub와 main은 **다른 하드웨어 컨텍스트**라 자원 충돌도 없습니다. 위험도 없고 충돌도 없으니 스케줄러가 둘을 자동으로 완전히 겹쳐 돌립니다. 만약 같은 절반을 둘 다 쓰면 WAR가 생겨 다시 직렬화될 겁니다. 그래서 "절반 번갈아 쓰기"가 핵심입니다. (TRF 시퀀서가 한 절반에서 로드하는 동안 다른 절반을 채울 수 있다는 하드웨어 근거: `docs/src/computing-tensors/register-files.md:92`. 단 두 절반이 같은 뱅크를 공유해서, 같은 사이클에 같은 뱅크를 치면 읽기가 우선권을 갖고 읽기 캐시·뱅크 교대로 충돌을 완화합니다 — `register-files.md:98`.)

또 하나의 미묘한 점(`docs/src/computing-tensors/index.md:80`): Vector 엔진과 Cast 엔진은 한 번에 한 컨텍스트만 구동할 수 있는 "하나의 스케줄링 단위"라서, sub가 Vector 일을 하는 동안 main은 Cast 엔진 대신 Commit 엔진의 타입 캐스팅을 써서 직렬화를 피합니다. 스케줄러는 또 DM 접근 패턴이 하드웨어 메모리 충돌을 일으킬 것 같으면 텐서 유닛 연산을 방어적으로 DMA 컨텍스트에 넘기기도 합니다(`docs/src/computing-tensors/index.md:84`).

## launch() — 커널을 띄우는 진입점

호스트 프로그램(`base-template/src/elementwise_mul.rs:14`)은 이렇게 커널을 호출합니다:

```rust
let mut ctx = Context::acquire();
let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0).await;          // 호스트->HBM 은 pdma!
let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;
let _out_hbm = launch(elementwise_mul_kernel, (&mut ctx, &lhs_hbm, &rhs_hbm)).await;
```

`launch`(`furiosa-opt-std/src/runtime/mod.rs:195`)는 시그니처가 `launch<F, P>(_f: F, args: P) -> F::Output`이고 내부는 `F::execute(args).await`가 전부입니다. 함수 값 자체(`_f`)는 버려지고, 그 "타입"만 트레잇 디스패치에 쓰입니다. 그래서 `#[device]` 매크로가 만들어 준 snake_case 상수(`elementwise_mul_kernel`)를 그대로 넘길 수 있어 편합니다. 인자는 튜플 `(&mut ctx, &lhs_hbm, &rhs_hbm)`로 묶어 전달합니다. 어느 백엔드에서 실제로 평가되는지는 빌드 시 `--cfg backend=...`가 결정합니다(`docs/src/introduction.md:114`): `cargo` 그대로면 함수 본문을 CPU에서 돌리고, `cargo furiosa-opt`면 선택한 백엔드(simulation/typecheck/npu)로 돕니다(`furiosa-opt-std/src/runtime/mod.rs:175` 주석). 여기서 호스트→HBM 적재가 `ctx.pdma`(PCIe DMA)인 점을 주목하세요. 커널 안에서 HBM↔DM은 `ctx.tdma`였습니다. 두 DMA 엔진의 역할이 이렇게 나뉩니다.

## 가장 중요한(그리고 헷갈리는) 진실: 시뮬레이션/타입체크는 주소를 "검사하지 않는다"

여기서 초심자가 거의 항상 오해하는 지점을 짚겠습니다. 위에서 "주소가 겹치면 위험하다"고 했는데, **그 위험 분석은 비공개(closed) NPU 컴파일러/스케줄러가 하는 일**입니다. 공개된 호스트 백엔드 두 개는 물리적 SRAM을 모델링하지 않습니다.
- **simulation** 백엔드는 매 연산을 매핑 대수로 인터프리트하는데, 데이터는 각 텐서가 들고 있는 자기 버퍼(`MathRawTensor`, `ArrayD`)에 저장됩니다(`furiosa-opt-std/src/runtime/simulation/backend.rs:9`, `furiosa-opt-std/src/tensor/raw.rs:5`). 주소는 그냥 텐서 핸들에 같이 들고 다니는 메타데이터일 뿐이고, 주소로 색인되는 공용 메모리 풀이 없습니다. `fetch`도 주소가 아니라 `self.inner`(그 텐서 자기 버퍼)에서 읽습니다(`furiosa-opt-std/src/engine/fetch.rs` fetch_impl).
- **typecheck** 백엔드는 축(axes)만 들고 값 버퍼가 아예 없습니다(`PhantomRawTensor`, `furiosa-opt-std/src/tensor/raw.rs:7`). 매핑/모양만 검증합니다.

결론(검증된 사실): **두 호스트 백엔드에서는 주소를 일부러 겹쳐도 데이터가 깨지지 않고, 테스트도 그대로 통과합니다.** 충돌은 조용히 무시됩니다. 즉 simulation은 "값이 맞는가"를, typecheck는 "모양이 맞는가"를 검증할 뿐, "스케줄/주소 안전성"은 검증하지 못합니다. 주소 배치의 정확성과 동시성(타이밍)은 진짜 NPU(`--backend npu`)에서만 강제됩니다. 이건 약점이 아니라 알고 써야 할 경계선입니다 — 시뮬레이션이 통과했다고 주소 배치가 안전하다고 착각하면 안 됩니다. (emulation 백엔드는 버퍼 기반이지만 현재 핵심 메서드가 `todo!()` 플레이스홀더라 사실상 미구현입니다 — `furiosa-opt-std/src/runtime/emulation/backend.rs:14`.)

이 사실을 알면 아래 실험들의 "예측 vs 관찰"이 왜 흥미로운지 바로 이해됩니다: 주소를 겹쳐도 시뮬레이션은 멀쩡히 통과하고, 그게 곧 "동시성/주소는 타이밍 개념이라 호스트 백엔드에 안 보인다"는 가장 큰 교훈입니다.

## 실행 명령 요약

프로젝트는 `Model_Benchmark/rngd-npu/vISA/experiments/`에 있고 5개 bin(`constant_add, elementwise_mul, dot_product, gemv, gemm`)이 등록돼 있습니다(`experiments/Cargo.toml`). 명령 형태(`docs/src/introduction.md:85`):
- 시뮬레이션 실행: `cargo furiosa-opt run --release --bin elementwise_mul`
- 타입체크만: `cargo furiosa-opt --backend typecheck run --release --bin elementwise_mul`
- 레퍼런스 대조 검증: `cargo furiosa-opt test --release --bin elementwise_mul`

(주의: `cargo check`는 함수 본문을 실행하지 않아 매핑 단언까지 못 갑니다 — `docs/src/introduction.md:133`. 그리고 cargo-furiosa-opt 서브커맨드 설치가 필요합니다 — `docs/src/introduction.md:33`. nightly-2026-05-01 툴체인은 `experiments/rust-toolchain.toml`에 고정돼 있습니다.)

## 2. 핵심 API · 패턴

| 이름 | 쓰는 법 | 설명 | 출처 |
|---|---|---|---|
| `Context (필드: main, sub, tdma, pdma)` | `ctx.main / ctx.sub : TuContext, ctx.tdma / ctx.pdma : DmaContext` | 네 필드가 네 갈래의 동시 실행 자원. 서로 다른 필드라 Rust의 분리 빌림으로 동시에 &mut 가능 = 동시성 표현 수단. | `furiosa-opt-std/src/context.rs:57` |
| `Context::acquire` | `let mut ctx = Context::acquire(); // Mutex 보호 전역 싱글톤 반환` | 디바이스 컨텍스트는 하나뿐. DerefMut<Target=Context> 가드를 돌려준다. | `furiosa-opt-std/src/context.rs:72` |
| `enum Tu / enum Dma` | `Tu::Main \| Tu::Sub ; Dma::Tensor \| Dma::Pcie` | const 제네릭으로 컨텍스트 종류를 타입에 박는다(TuContext<{Tu::Main}> 등). | `furiosa-opt-std/src/context.rs:20` |
| `TuContext::begin` | `ctx.main.begin(dm_tensor.view()) -> BeginTensor (이후 fetch/collect/... 체인)` | main/sub 둘 다에서 텐서 유닛 파이프라인을 시작하는 진입점. | `furiosa-opt-std/src/context.rs:87` |
| `launch` | `launch(kernel_fn, (&mut ctx, &a_hbm, &b_hbm)).await -> F::Output` | 함수 값은 버려지고 타입만 트레잇 디스패치에 사용. 내부는 F::execute(args).await. 실제 평가 백엔드는 --cfg backend 가 결정. | `furiosa-opt-std/src/runtime/mod.rs:195` |
| `HbmTensor::to_dm` | `.to_dm::<Cluster, Slice, Elem>(&mut ctx.tdma, address: u64)` | HBM→DM 적재. 두 번째 인자가 수동 DM 주소. 자동 할당 없음. | `furiosa-opt-std/src/tensor/memory.rs:423` |
| `CollectTensor::to_vrf` | `.to_vrf::<Element>(address: Address) -> VrfTensor` | Collect→VRF. raw u64 주소. 충돌 검사 없음(self.inner 데이터 + 주소 메타데이터). | `furiosa-opt-std/src/engine/collect.rs:72` |
| `CollectTensor::to_trf` | `.to_trf::<Lane, Element>(address: TrfAddress) -> TrfTensor` | Collect→TRF. raw 숫자가 아니라 enum(FirstHalf/SecondHalf/Full). 이중 버퍼링용 절반 선택. | `furiosa-opt-std/src/engine/collect.rs:56` |
| `enum TrfAddress` | `TrfAddress::FirstHalf \| SecondHalf \| Full ; .capacity()` | Full=65536B(0~127행), FirstHalf=0~63행, SecondHalf=64~127행, 절반=32768B(40KB cap). | `furiosa-opt-std/src/tensor/memory.rs:101` |
| `*::commit` | `.commit::<Element>(address: Address) -> DmTensor` | 파이프라인 결과를 DM 주소에 쓰기. 입력 텐서의 DM 주소와 겹치지 않게 잡아야 함. | `furiosa-opt-std/src/engine/commit.rs:27` |
| `DmTensor::to_hbm` | `.to_hbm(&mut ctx.tdma, address) ; HostTensor::to_hbm(&mut ctx.pdma, address).await` | 커널 내 DM→HBM 은 tdma, 호스트 프로그램의 host→HBM 적재는 pdma 를 쓴다. | `furiosa-opt-std/src/tensor/memory.rs:387` |
| `type Address` | `pub type Address = u64;` | DM/VRF 주소는 단순 u64. TRF만 enum. | `furiosa-opt-std/src/tensor/memory.rs:21` |

## 3. 실험 (직접 돌리기)

> 실험은 NPU 없이 `simulation`·`typecheck`로 돌아갑니다. 실행법은 [`../experiments/README.md`](../experiments/README.md), MNIST는 `cargo furiosa-opt test`(npu 전용).

### 실험 08.1 — 기준: elementwise_mul 을 시뮬레이션과 타입체크로 돌려 보기
*난이도 1/5 · 기반: `Model_Benchmark/rngd-npu/vISA/experiments/src/kernel/elementwise_mul_kernel.rs`*

**목표** — 스케줄러가 sub(VRF 적재)와 main(벡터 곱)을 한 커널에서 어떻게 엮는지 실제로 통과시켜 감을 잡는다.

```bash
cd /home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vISA/experiments && cargo furiosa-opt run --release --bin elementwise_mul && cargo furiosa-opt --backend typecheck run --release --bin elementwise_mul && cargo furiosa-opt test --release --bin elementwise_mul
```
**관찰** — run 은 'Elementwise Mul: kernel ran' 출력, test 는 레퍼런스(out[i]=lhs[i]*rhs[i])와 전 원소 일치로 통과. typecheck 도 모양 검증만으로 통과. 세 가지 모두 성공해야 이후 실험의 기준선이 된다.

**심화** — 코드에서 sub 블록과 main 블록의 컨텍스트를 짚어 보고, 어디에서 RAW(main 이 rhs_vrf 를 읽음)가 생겨 main 이 sub 를 기다리게 되는지 줄 번호로 표시해 보라.

### 실험 08.2 — sub 프리페치 주소 바꿔 보기: VRF 주소 0 -> 0x1000
*난이도 2/5 · 기반: `Model_Benchmark/rngd-npu/vISA/experiments/src/kernel/elementwise_mul_kernel.rs`*

**목표** — 수동 SRAM 주소가 호스트 백엔드에서 어떻게 취급되는지 확인한다(주소는 메타데이터).

```bash
# elementwise_mul_kernel.rs 의 `.to_vrf(0)` 를 `.to_vrf(1 << 12)` 로 수정 후:
cd /home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vISA/experiments && cargo furiosa-opt test --release --bin elementwise_mul
```
**관찰** — 예측: VRF 주소만 바꿨으니 다른 텐서와 겹치지 않는 한 결과는 동일. 관찰: 테스트 여전히 통과. 교훈: simulation 은 값을 텐서 자기 버퍼로 추적하므로 주소는 메타데이터일 뿐이다(furiosa-opt-std/src/runtime/simulation/backend.rs).

**심화** — 원래 0으로 되돌린 뒤 git diff 로 변경을 확인하고, 이 주소가 진짜로 의미를 갖는 시점은 --backend npu 라는 점을 메모하라.

### 실험 08.3 — 일부러 주소 충돌시키기: rhs_dm 을 lhs_dm 과 같은 0번지로
*난이도 2/5 · 기반: `Model_Benchmark/rngd-npu/vISA/experiments/src/kernel/elementwise_mul_kernel.rs`*

**목표** — '주소 충돌 = 실패'라는 통념을 호스트 백엔드에서 검증하고, 충돌이 어디서 진짜로 잡히는지 이해한다.

```bash
# `let rhs_dm = rhs.to_dm::<...>(&mut ctx.tdma, 1 << 12);` 의 주소 `1 << 12` 를 `0` 으로 바꿔 lhs_dm 과 충돌시킨다. 그 뒤:
cd /home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vISA/experiments && cargo furiosa-opt test --release --bin elementwise_mul && cargo furiosa-opt --backend typecheck run --release --bin elementwise_mul
```
**관찰** — 예측(흔한 오해): lhs 가 rhs 를 덮어써서 테스트 실패. 실제 관찰: simulation/test 도, typecheck 도 그대로 통과한다. 이유: 두 백엔드는 주소로 색인되는 공용 SRAM 을 모델링하지 않아 충돌이 조용히 무시된다(furiosa-opt-std/src/tensor/raw.rs:5, runtime/simulation/backend.rs:9). 진짜 RAW/WAR/WAW 강제는 비공개 NPU 스케줄러(--backend npu)에서만 일어난다.

**심화** — 끝나면 주소를 1<<12 로 복구하라. 이 실험이 주는 메타 교훈을 한 문장으로 적어 보라: '시뮬레이션 통과 != 주소 배치 안전'.

### 실험 08.4 — sub 의 프리페치를 main 으로 옮겨 동시성 제거해 보기
*난이도 3/5 · 기반: `Model_Benchmark/rngd-npu/vISA/experiments/src/kernel/elementwise_mul_kernel.rs`*

**목표** — 컨텍스트 분리가 곧 병렬성이라는 점, 그리고 그 병렬성(타이밍)이 호스트 백엔드에는 보이지 않는다는 점을 체감한다.

```bash
# rhs_vrf 적재 블록의 `ctx.sub` 를 `ctx.main` 으로 바꿔 VRF 적재와 벡터 곱이 같은 main 컨텍스트에 놓이게 한다. 그 뒤:
cd /home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vISA/experiments && cargo furiosa-opt test --release --bin elementwise_mul
```
**관찰** — 예측: 값은 그대로(둘 다 main 이면 직렬화될 뿐 결과는 동일). 관찰: 테스트 통과. 교훈: simulation 은 '값이 맞는가'만 검증하고 sub/main 오버랩 손실(타이밍)은 보여 주지 않는다 — 컨텍스트 선택의 성능 효과는 NPU 에서만 드러난다.

**심화** — 이 변경이 docs/src/scheduler.md:39 의 '같은 컨텍스트는 직렬화'와 어떻게 대응되는지 설명하고, 원복하라.

### 실험 08.5 — gemm 의 sub->TRF->main 분업 관찰 + 대조용 모양 오류 내 보기
*난이도 3/5 · 기반: `Model_Benchmark/rngd-npu/vISA/experiments/src/kernel/gemm_kernel.rs`*

**목표** — sub 가 Contraction 을 못 해 TRF 적재만 맡고 main 이 contract 로 읽는 분업을 확인하고, 호스트 백엔드가 '무엇은' 잡는지(모양) vs '무엇은 못 잡는지'(주소) 대조한다.

```bash
cd /home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vISA/experiments && cargo furiosa-opt test --release --bin gemm && cargo furiosa-opt --backend typecheck run --release --bin gemm
```
**관찰** — 정상 통과 확인. gemm_kernel.rs:25 의 ctx.sub.begin(...).to_trf(TrfAddress::Full) 가 TRF 적재, ctx.main.begin(...).contract_outer(&b_trf) 가 그 TRF 를 읽는 RAW 구조임을 코드에서 확인한다.

**심화** — gemm_kernel.rs 의 b_dm 주소 `1 << 12` 를 0(=a_dm 과 충돌)으로 바꿔도 test 가 통과함을 확인(주소는 호스트 백엔드서 무검사). 반대로 collect 의 Element 매핑을 일부러 틀리게 바꾸면 --backend typecheck run 이 매핑 단언으로 잡아낸다 — '주소는 못 잡고 모양은 잡는다'의 대조를 직접 만들어 보라. 끝나면 원복.

## 4. 연습문제 (손으로, 컴파일 없이)

**Q1.** 다음을 보고 최종 실행 순서를 적어라. let t0=load(); //O0\nlet t1=load(); //O1\nlet t2=t0.op(); //O2\nlet t3=t1.op(); //O3\nlet t4=t2.op(); //O4\nlet t5=t4.op(); //O5

<details><summary>정답/힌트</summary>

O0 → O1 → O2 → O3 → O4 → O5. 스케줄러는 쓰여진 순서를 재배치 없이 그대로 따른다(docs/src/scheduler.md:21).

</details>

**Q2.** 행렬곱 A(main 컨텍스트)와 DMA 전송 B(DMA 컨텍스트), 그리고 또 다른 행렬곱 C(main 컨텍스트)가 있다. A,B,C 중 어느 쌍이 병렬이고 어느 쌍이 직렬인가?

<details><summary>정답/힌트</summary>

A와 B는 다른 컨텍스트라 병렬. A와 C는 같은 main 컨텍스트라 직렬(자원 충돌). B와 C도 다른 컨텍스트라 병렬(주소 의존성이 없다면). docs/src/scheduler.md:39.

</details>

**Q3.** 주소 0x100 에 sub 가 write 한 뒤 main 이 같은 0x100 을 read 한다. 이건 무슨 위험(hazard)이며 스케줄러는 어떻게 처리하는가?

<details><summary>정답/힌트</summary>

RAW(read-after-write). 스케줄러가 main 의 read 앞에 암묵적 대기를 삽입해 sub 의 write 가 끝난 뒤 읽게 한다 = 'main 이 sub 를 기다린다'.

</details>

**Q4.** 이중 버퍼링 루프에서 i=0 에 sub 가 FirstHalf 를 미리 채웠다. for 루프 안 표현식 `if i % 2 == 0 { SecondHalf } else { FirstHalf }` 기준으로, i=0,1,2 에서 sub 가 '다음 배치'를 채우는 절반은 각각 어디인가? 그리고 왜 main 의 현재 읽기와 충돌하지 않는가?

<details><summary>정답/힌트</summary>

i=0→SecondHalf, i=1→FirstHalf, i=2→SecondHalf. main 은 현재 절반(i=0 이면 FirstHalf)을 읽고 sub 는 반대 절반을 채워 서로 다른 주소라 WAR 없음 + 다른 컨텍스트라 자원 충돌 없음 → 완전 오버랩(docs/src/scheduler.md:91).

</details>

**Q5.** elementwise_mul_kernel 에서 rhs_dm 의 주소를 lhs_dm 과 같은 0 으로 바꾸고 `cargo furiosa-opt test --release --bin elementwise_mul` 을 돌리면 통과할까 실패할까? 이유는?

<details><summary>정답/힌트</summary>

통과한다. simulation 백엔드는 데이터를 각 텐서 자기 버퍼(MathRawTensor)로 추적하고 주소는 메타데이터일 뿐이라 충돌이 데이터를 손상시키지 않는다. 진짜 충돌 강제는 --backend npu 에서만(furiosa-opt-std/src/runtime/simulation/backend.rs:9).

</details>

**Q6.** 행렬곱(컨트랙션)을 ctx.sub.begin(...).contract_outer(...) 로 sub 컨텍스트에서 돌리려 한다. 무엇이 문제인가?

<details><summary>정답/힌트</summary>

sub 는 main 의 부분집합이라 Contraction 엔진을 구동하지 못한다(docs/src/computing-tensors/index.md:74). 컨트랙션은 main, sub 는 TRF/VRF 적재용.

</details>

**Q7.** 호스트 드라이버에서 lhs.to_hbm(&mut ctx.tdma, 0) 처럼 적었다. 무엇이 잘못됐고 어떻게 고치나?

<details><summary>정답/힌트</summary>

호스트 메모리↔HBM 적재는 PCIe DMA 라 ctx.pdma 를 써야 한다. tdma 는 칩 내 HBM↔DM 용. base-template/src/elementwise_mul.rs:12 참고.

</details>

## 5. 흔한 함정

- 메모리 주소를 직접 안 적으면 안 된다 — 자동 할당기가 없다. to_dm/to_vrf/commit 등은 모두 주소 인자를 요구하며, 텐서들이 겹치지 않게 배치하는 책임은 전적으로 프로그래머에게 있다.  
  ↳ 출처 `docs/src/scheduler.md:24, furiosa-opt-std/src/tensor/memory.rs:21`
- 주소를 일부러 겹쳐도 simulation/typecheck 는 통과한다. 이 두 호스트 백엔드는 주소로 색인되는 공용 SRAM 을 모델링하지 않아 충돌이 조용히 무시된다. '시뮬레이션 통과 = 주소 안전'이라고 착각하면 NPU 에서 데이터 손상으로 터진다. RAW/WAR/WAW 강제는 비공개 NPU 스케줄러에서만 일어난다.  
  ↳ 출처 `furiosa-opt-std/src/runtime/simulation/backend.rs:9, furiosa-opt-std/src/tensor/raw.rs:5, docs/src/scheduler.md:49`
- 다른 컨텍스트라고 무조건 병렬은 아니다. 같은 주소에서 RAW/WAR/WAW 가 생기면 스케줄러가 대기를 끼워 직렬화한다. 반대로 같은 컨텍스트는 주소가 안 겹쳐도 자원 충돌로 직렬화된다.  
  ↳ 출처 `docs/src/scheduler.md:37`
- sub 컨텍스트는 Contraction 엔진을 구동하지 못한다. 행렬곱/어텐션 같은 컨트랙션을 sub 에 넣으려 하지 말 것 — sub 는 적재(TRF/VRF 프리페치) 담당이고 컨트랙션은 main 몫이다.  
  ↳ 출처 `docs/src/computing-tensors/index.md:74`
- 이중 버퍼링에서 sub 와 main 이 같은 TRF 절반을 쓰면 WAR 가 생겨 오버랩이 깨지고 직렬화된다. 회차마다 FirstHalf/SecondHalf 를 반드시 번갈아 써야 겹쳐 돈다.  
  ↳ 출처 `docs/src/scheduler.md:91`
- VRF 는 하드웨어가 절반을 나눠 주지 않는다. VRF 이중 버퍼링을 원하면 8KB 안에서 겹치지 않는 영역을 프로그래머가 직접 잡아야 한다(TRF 처럼 자동 절반 모드가 없음).  
  ↳ 출처 `docs/src/scheduler.md:60`
- cargo check 는 함수 본문을 실행하지 않아 매핑/모양 단언까지 도달하지 못한다. 모양 검증을 빠르게 하려면 cargo furiosa-opt --backend typecheck run 을 써야 한다.  
  ↳ 출처 `docs/src/introduction.md:133`
- 커널 내 HBM↔DM 은 ctx.tdma, 호스트↔HBM 적재는 ctx.pdma 로 서로 다르다. 호스트 드라이버에서 to_hbm 에 tdma 를 넘기는 식으로 혼동하면 안 된다.  
  ↳ 출처 `base-template/src/elementwise_mul.rs:12, base-template/src/kernel/elementwise_mul_kernel.rs:16`
- bin 소스는 src/ 바로 아래에 두고 Cargo.toml 에 [[bin]] path 를 명시해야 한다. src/bin/, examples/, tests/ 아래 파일은 rustc 플러그인이 조용히 건너뛴다.  
  ↳ 출처 `docs/src/introduction.md:79`

## 6. 핵심 정리 & 다음

기억할 사실:
- 하드웨어는 동시에 돌 수 있는 실행 컨텍스트를 노출한다. 문서상 개념은 세 개(main, sub, DMA)지만 DMA가 텐서 DMA와 PCIe DMA로 갈리므로 코드의 Context 구조체에는 main/sub/tdma/pdma 네 필드가 있다. (`docs/src/scheduler.md:33, furiosa-opt-std/src/context.rs:57`)
- main 컨텍스트는 텐서 유닛의 8개 엔진(Fetch→Switch→Collect→Contraction→Vector→Cast→Transpose→Commit) 전부를 구동한다. (`docs/src/computing-tensors/index.md:73`)
- sub 컨텍스트는 main의 부분집합만 구동한다 — Contraction 엔진과 일부 기능이 빠진다. 그래서 sub는 보통 TRF/VRF로 피연산자를 미리 적재하는 역할이다. (`docs/src/computing-tensors/index.md:74`)
- tdma는 텐서 DMA(칩 내 HBM↔DM, HBM↔SPM, DM↔SPM), pdma는 PCIe DMA(호스트 메모리↔HBM)다. enum Dma { Tensor, Pcie }로 구분된다. (`furiosa-opt-std/src/context.rs:29, docs/src/computing-tensors/index.md:71`)
- 같은 컨텍스트의 연산은 직렬화되고, 다른 컨텍스트의 연산은 병렬로 실행된다. 직렬화 원인은 자원 충돌과 메모리 의존성 두 가지뿐이다. (`docs/src/scheduler.md:35`)
- 스케줄러는 적힌 순서를 권위 있는 시퀀스로 취급하며 연산을 재배치하지 않는다. (`docs/src/scheduler.md:5`)
- 스케줄러는 프로그램에 적힌 메모리 주소를 분석해 RAW/WAR/WAW 위험을 탐지하고 필요한 곳에 암묵적 대기를 자동 삽입한다(수동 배리어 불필요). 단 주소 자체는 프로그래머가 명시해야 한다. (`docs/src/scheduler.md:49`)
- Vector 엔진과 Cast 엔진은 한 번에 한 컨텍스트만 구동하는 하나의 스케줄링 단위다. sub가 Vector를 쓰는 동안 main은 Cast 대신 Commit 엔진의 타입 캐스팅을 써서 직렬화를 피한다. (`docs/src/computing-tensors/index.md:80`)

➡️ 다음: [09_tiling_partitioning.md](./09_tiling_partitioning.md)
