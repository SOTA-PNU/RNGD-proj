# 10 · MNIST 실전 해부 — vISA로 만든 모델이 RNGD에서 어떻게 연산되나

> furiosa-opt-examples 의 MNIST 예제(`furiosa-opt-examples/src/mnist/mod.rs`)를 우리 서버에서
> 실제로 컴파일·실행하고, 소스와 컴파일러 스케줄 덤프를 1:1로 대조해 구조를 해부한 문서.
> **이 문서의 수치는 실기 실행 로그와 컴파일러 스케줄 덤프에서 그대로 옮긴 값이다** — 소스는 파일:줄로, 사이클은 `cargo furiosa-opt compile forward --dump-schedule`
> 결과(총 17,953 cycle, 인스트럭션 22개)에서 그대로 옮겼다. 추정·근사는 그렇게 명시했다.
> 단 **두 종류를 구분해서 읽을 것**: **실기 실행 결과**(진짜 NPU 에서 돈 것 — 정답률 10/10, 0.12s)와
> **컴파일러 스케줄 모델값**(사이클·인스트럭션 분해 — §7). 후자는 NPU 를 켜지 않고 얻은 값이다.

## 0. 검증 사실 (먼저)

| 항목 | 값 | 근거 |
|---|---|---|
| 실행 결과 | typecheck ⚠️(빌드·매핑 검증은 통과, 테스트 단정은 phantom 값이라 실패) / emulation ✅ / **npu ✅** (10/10 정답) | npu: `cargo furiosa-opt --backend npu test --release` → `test_mnist ... ok` (0.12s) / typecheck: `ex_logs/tc2.log` `mnist_tests.rs:47` unwrap None |
| forward EDF | 68,106 B (단일 커널) | `target_npu/furiosa-opt/kernel/visa_mnist/visa_mnist::mnist::forward.edf` |
| 스케줄 span | 17,953 cycle (@1GHz ≈ 18.0 µs) | `--dump-schedule` 최종 end |
| DRAM io | 419,968 B | `--dump-summary` summary.json `dram_usage_by_kind.io` |
| forward 소스 | `mnist.rs:181-192`, 2줄 | 아래 §2 |

> 주의: 위 "@1GHz" 는 클럭 라벨이 덤프에 없어 문서(dma-engine.md:330)의 1 GHz 예시로 환산한 값이다.
> **사이클 수 자체는 클럭과 무관한 스케줄 모델값**이고, µs 환산만 클럭 가정에 의존한다.

### 0.1 이 예제가 왜 기준점인가 (크레이트 전수 조사 결과)

- 저장소에서 **end-to-end 학습 모델은 MNIST 하나뿐**이다. Qwen 2.5 0.5B(24레이어)는 가중치가 전부
  `zero()`, Llama 3.1 예제는 함수가 `// TODO` 로 끊긴 미완성 골격이다.
- `mnist::forward` 는 소스에서 추출한 커널 **200개 중 실기 컴파일에 성공한 137개**에 든다(63개 실패).
  즉 **여기서 되는 게 전부 되는 게 아니다** — `transformer::{embedding,attention,decoder,head}::forward`,
  `vrf_add::vrf_add` 는 실기 컴파일에 실패한다. 다중 칩/클러스터 커널 6종은 5종이 컴파일 실패고,
  `cluster_chip_shuffle_slice::chip_shuffle` 1종은 컴파일은 통과하지만 실기 커널 로드에서 abort 한다.
- 게이팅한 크레이트로 돌린 **실기 테스트 89개** 중 `mnist` 는 **1/1 통과**(이미지 10장 전부 정답).
- 실패 유형·전수 매트릭스: [12-예제-전수실행](./12-예제-전수실행.md), [13-NPU-실기-매트릭스](./13-NPU-실기-매트릭스.md)

## 1. 모델 — 무엇을 분류하나

2층 완전연결(FC) MLP. 28×28 MNIST 숫자를 10개 클래스로 분류한다.
가중치는 `data/mnist/mnist.safetensors`(859,220 B)에 들어 있다(실측 형상):

| 텐서 | dtype | shape | 의미 |
|---|---|---|---|
| `hw.fc1.weight` | bf16 | [256, 800] | 1층 가중치 (입력 784→800 패딩, 은닉 256) |
| `fc1.bias` | bf16 | [256] | 1층 bias |
| `hw.fc2.weight` | bf16 | [16, 256] | 2층 가중치 (은닉 256, 출력 10→16 패딩) |
| `hw.fc2.bias` | bf16 | [16] | 2층 bias |
| `hw.image_{i}` | bf16 | [800] | 입력 이미지 (784→800 패딩) |
| `label_{i}` | i32 | [1] | 정답 라벨 |

**패딩 이유**: 784(=28²)와 10은 하드웨어 타일 경계(8/16/32의 배수)에 안 맞는다.
`mnist.py`가 784→800, 10→16 으로 zero-pad 해서 저장한다(`mnist.py:124-131`, `F.pad`).
그래서 소스의 축은 `X = 800`(입력), `H = 256`(은닉), `C = 16`(출력)이다(`mnist.rs:3`).

수식:
```
hidden = ReLU(fc1_W · image + fc1_b)     # [800] → [256]
logits =      fc2_W · hidden + fc2_b     # [256] → [16]
predict = argmax(logits[:10])            # ← 호스트에서
```

## 2. forward — 전체 골격

`mnist.rs:181-192`, `#[device]` 함수 본문은 두 줄뿐이다:
```rust
#[device(chip = 1)]
pub fn forward(ctx, input, fc1_weight, fc1_bias, fc2_weight, fc2_bias) -> HbmTensor<bf16, Chip, m![C]> {
    let hidden = fc1_relu(ctx, input, fc1_weight, fc1_bias);   // 1층 + bias + ReLU
    fc2(ctx, hidden, fc2_weight, fc2_bias)                     // 2층 + bias
}
```
이 두 줄이 컴파일되면 **22개 하드웨어 인스트럭션의 단일 EDF**가 된다.
`argmax`는 커널에 없다 — 호스트(`tests/mnist_tests.rs`)가 logits를 받아 `max_by`로 고른다.

**축 정의** (`mnist.rs:5-6`):
- `Chip = m![1]` — 칩 1개
- `Cluster = m![1 # 2]` — 논리 클러스터 1개를 물리 2개로 패딩(절반 유휴)

## 3. 한 레이어의 뼈대 — fc1_matmul (`mnist.rs:8-34`)

가장 순수한 행렬곱. `image · W` 를 계산한다. 6단계 파이프라인:

```rust
let input_dm  = input.to_dm(&mut ctx.tdma);    // :13  HBM → DM(SRAM)
let weight_dm = weight.to_dm(&mut ctx.tdma);   // :14

let input_trf = ctx.sub.begin(input_dm.view()) // :16  sub 컨텍스트
    .fetch::<m![1], m![X]>()                    // :19  Fetch 엔진
    .collect::<m![X / 16], m![X % 16]>()        // :20  Collect (32B flit 정규화)
    .to_trf();                                  // :21  → TRF (레지스터파일 상주)

ctx.main.begin(weight_dm.view())               // :23  main 컨텍스트
    .fetch::<m![X / 16], m![X % 16]>()          // :25  weight 스트리밍
    .collect::<m![X / 16], m![X % 16]>()        // :26
    .contract_outer::<...>(&input_trf)          // :27  ┐ DPE 4단계:
    .contract_packet::<m![1]>()                 // :28  │  Broadcast+Multiply
    .contract_time::<m![1]>()                   // :29  │  → Packet reduce
    .contract_lane::<m![1], m![1 # 8]>(...)     // :30  ┘  → Time reduce → Lane
    .cast::<bf16, m![1 # 16]>()                 // :31  f32 누산 → bf16
    .commit_trim::<m![1 # 16]>()                // :32  packet 잘라내기
    .commit()                                   // :33  → DM
```

**핵심 구조 3가지** (책 contraction-engine/index.md:3-11 그대로):
1. **두 피연산자의 역할이 다르다.** `input`은 TRF에 **상주**(sub 컨텍스트), `weight`는 **스트리밍**(main 컨텍스트)해서 흘러 들어간다. 이게 systolic array가 곱-누산하는 방식이다.
2. **축약은 4단계로 분해된다.** `contract_outer`(Broadcast+Multiply) → `packet` → `time` → `lane`(Reduce 3단). matmul이든 conv든 이 4단계를 탄다.
3. **정밀도 경계는 `.cast()`.** Contraction은 bf16 피연산자를 f32로 확장해 누산하고(cast.rs `ContractionCast for bf16 → f32`), `.cast::<bf16>`로 되좁힌다.

## 4. bias 준비 — 연산보다 데이터 이동이 더 복잡 (`mnist.rs:36-65`)

`fc1_bias_prepared`는 bias 하나 준비하는 데 **transpose + switch(InterTranspose) + reshape** 를 쓴다:
```rust
.transpose::<m![H % 8], m![1 # 16]>()          // :46  flit 안 축 재정렬
unsafe { bias_dm_1.reshape() }                 // :49  타입만 재해석(런타임 0)
.switch::<m![H], m![Dummy8]>(InterTranspose {slice1:8, slice0:1, time0:1})  // :54  슬라이스↔시간 축 교환
.transpose::<m![Dummy8 / 4], m![Dummy8 % 4 # 16]>()  // :60
```
**왜 이렇게 복잡한가**: bias는 `[256]` 1D인데, matmul 출력은 슬라이스/레인에 흩어진 레이아웃이다.
Vector Engine에서 둘을 더하려면 bias를 **matmul 출력과 똑같은 물리 배치**로 옮겨야 한다.
`reshape`는 `unsafe`이고 런타임 비용 0(타입 재해석만) — 실제 데이터 이동은 `transpose`/`switch`가 한다.

> 이것이 vISA의 실체다: 코드 줄 수의 대부분이 "연산"이 아니라 "데이터를 엔진이 원하는 모양으로 옮기기"다.

## 5. ReLU — 곱셈이 아니라 Vector Engine의 클립 (`mnist.rs:67-89`)

`fc1_relu`가 matmul 결과와 bias를 **동시에** 받아 bias-add + ReLU 를 한 번에 처리한다:
```rust
ctx.main.begin_interleaved::<I, ...>(matmul.view(), bias_dm_4.view())  // :77  두 텐서 동시 입력
    .fetch::<m![I], m![1 # 4]>()
    .fetch_cast::<f32>()                        // :79  f32로 올려서 계산
    .collect::<m![I], m![1 # 8]>()
    .vector_init()                              // :81  Vector Engine 시작
    .vector_intra_slice_unzip::<I, ...>()       // :82  interleave된 두 값을 분리
    .vector_clip_zip(ClipBinaryOpF32::Add)      // :83  matmul + bias
    .vector_clip(ClipBinaryOpF32::Max, 0.0f32)  // :84  ★ ReLU = max(x, 0)
    .vector_final()
    .cast::<bf16, m![1 # 16]>()                 // :86  다시 bf16
```
**ReLU = `vector_clip(Max, 0.0)`.** 0보다 작은 값을 0으로 자르는 가위질이다.
별도 커널이 아니라 **bias-add에 융합**되어 한 번의 Vector Engine 통과로 끝난다.
`begin_interleaved`(`I=2`)가 두 입력(matmul, bias)을 교차로 넣고, `unzip`이 다시 분리해서 더한다.

## 6. 2층 — 같은 뼈대 반복 (`mnist.rs:91-179`)

`fc2`는 fc1과 구조가 같다: `fc2_matmul`(256→16 행렬곱) + `fc2_bias_prepared` + Vector Engine.
차이는 **fc2에는 ReLU가 없다**는 것(`mnist.rs:170-173`에 `vector_clip(Max,0)` 줄이 없음) — 마지막 층은 logits를 그대로 낸다.
그리고 fc2는 입력을 다시 배치하는 `fc2_input_prepared`(`:119-131`, `Broadcast1` switch)가 추가로 필요하다.
마지막에 `logits.to_hbm`(`:178`)으로 결과를 HBM에 쓴다.

## 7. 하드웨어 스케줄 — 22개 인스트럭션 (스케줄 모델 덤프)

`--dump-schedule` 결과. 소스라인이 그대로 붙어 나온다:

> **이 절의 사이클은 컴파일러의 스케줄 모델값이다.** `compile --dump-schedule` 은 AOT 산출물이라
> **NPU 를 점유하지 않는다**(실기 테스트와 동시 실행 가능). 실기에서 실제로 측정한 것은 §0 의
> `test_mnist ... ok` (0.12s, 10/10 정답) 쪽이다. 둘을 섞어 인용하지 말 것.

| # | 엔진 | cyc | % | 소스 | 하는 일 |
|--:|---|--:|--:|---|---|
| 1 | DmaLoad | 1,662 | 9.3% | `:14` | **fc1 weight [256×800] → DM** |
| 8 | Main | 273 | 1.5% | `:41` | fc1 bias transpose |
| 15 | Main | 343 | 1.9% | `:50` | fc1 bias switch |
| 24 | Main | 4,871 | **27.1%** | `:139` | fc2 bias 준비 |
| 25 | DmaLoad | 5,497 | **30.6%** | `:13` | **fc1 input [800] → DM** (슬라이스 256개로 복제) |
| 32 | Sub | 463 | 2.6% | `:16` | fc1 input → TRF |
| 37 | Main | 315 | 1.8% | `:23` | **fc1 matmul (DPE)** |
| 42 | Main | 283 | 1.6% | `:76` | fc1 bias-add + ReLU |
| 47 | Main | 1,033 | 5.8% | `:123` | fc2 input 재배치 |
| 51 | Sub | 327 | 1.8% | `:99` | fc2 input → TRF |
| 57 | Main | 281 | 1.6% | `:106` | **fc2 matmul (DPE)** |
| 62 | Main | 283 | 1.6% | `:164` | fc2 bias-add |
| 65 | DmaStore | 3,576 | **19.9%** | `:178` | logits → HBM |
| — | Core sync | 600 | 3.3% | | 클러스터 동기화 |

> **정정** — `#1`/`#25` 의 담당은 이전 판에서 서로 바뀌어 적혀 있었다. 스케줄 덤프의 소스 귀속은
> `#1` = `mnist.rs:14` `weight.to_dm` (1,662 cyc), `#25` = `mnist.rs:13` `input.to_dm` (5,497 cyc)다
> (§3 코드블록의 줄번호와 일치). **즉 더 비싼 로드는 가중치가 아니라 입력 쪽이다.**
> 형상 근거: 입력은 HBM 에서 `m![X]` 1D(`:10`)인데 DM 에서는 `m![H], m![X]`(`:13`) — 슬라이스 축 `H`=256 으로 복제된다.
> 가중치는 HBM 에서 이미 `m![H, X]`(`:11`)라 형상 그대로 옮겨진다.

(표에서 8개 인스트럭션을 생략했다: `reshape` 4개(`:49`/`:64`/`:138`/`:152`) + 0-cycle `Core`(`:178`) = **0-cycle 5개** + bias·fc2 weight 의 **DmaLoad 3개**
(611·572·447 cyc, `:40`/`:97`/`:137`). 아래 엔진별 합계는 22개 **전부**를 포함하므로 표만 더하면 안 맞는다.)

**엔진별 합계** (겹쳐 실행되어 합이 100% 초과):

| 엔진 | cyc | span 대비 |
|---|--:|--:|
| DmaLoad | 8,789 | 49.0% |
| Main (DPE+Vector) | 7,682 | 42.8% |
| DmaStore | 3,576 | 19.9% |
| Sub (TRF적재) | 790 | 4.4% |
| Core (sync) | 600 | 3.3% |

`--dump-summary` 요약: main 42.8% / io 68.9% / sub 4.4%, computation_cycle 8,472.

**2차 독립 확인** — 커널 130개를 한꺼번에 스캔한 별도 분석기(`sched_scan.sh` + `sched_analyze.py`,
[13-NPU-실기-매트릭스](./13-NPU-실기-매트릭스.md))가 위 값을 **자릿수까지 그대로** 재현했다:
`span 17,953 / 22 inst`, **DmaEngine 12,365 (68.9%)**, MainContext 7,682, **VectorEngine 1,162**,
SubContext 790, PeCore 600.

- `DmaEngine 12,365` = 위 표의 DmaLoad 8,789 + DmaStore 3,576. `io 68.9%` 와 같은 수다.
- `VectorEngine 1,162` 은 **MainContext 7,682 에 더할 항목이 아니다.** Main 인스트럭션 4개
  (`:23` 315 + `:76` 283 + `:106` 281 + `:164` 283)가 MainContext 와 VectorEngine 컨텍스트를
  **동시에** 점유한 합이다 — 즉 fc1/fc2 의 matmul·bias-add·ReLU 커밋이 그 4개다.

## 8. 스케줄이 드러내는 두 가지 구조적 사실

**① 두 레이어가 겹쳐 실행된다.** 스케줄에서 가장 무거운 두 명령이 동시에 돈다:
- `#24` fc2 bias 준비 (Main, begin 5295)
- `#25` fc1 **input** 로드 (DmaLoad, begin 5295 — `:13`, 위 정정 참조)

DMA 엔진과 연산 엔진이 **다른 컨텍스트(tdma/main)라 병렬**이라, 스케줄러가 "fc1 계산 중에 fc2 준비를 미리 당겨" 겹친다.
소스는 순차(`fc1_relu` → `fc2`)로 짰지만 하드웨어는 파이프라인으로 겹친다.

**② 이 모델은 연산이 제 몫을 한다.** Main 42.8% vs io 68.9% 로 균형잡혀 있다.
→ 앞서 본 gemm(512×512)은 io 95.7% / 연산 0.9% 로 완전히 write-bound였다.
차이는 **출력 크기**다: gemm은 512×512 큰 행렬을 쓰느라 DmaStore가 지배했지만,
MNIST는 출력이 [16]으로 작아 write 부담이 없고(DmaStore 19.9%) 연산이 살아난다.
**MNIST가 vISA 파이프라인이 제대로 도는 걸 보여주는 더 좋은 예제인 이유가 이것이다.**

MNIST 가 이 스택에서 어디쯤인지 이제 수치로 말할 수 있다. 커널 **130개**의 스케줄을 뽑아 보면
커널별 DMA 사이클 비중의 **중앙값이 82.8%**, 130개 중 **107개(82%)가 50% 이상**, **54개는 90% 이상**이다.
MNIST 의 **68.9%** 는 그 중앙값보다 **낮은** 쪽이다 — 다만 여전히 50% 이상인 107개 안에 들고,
DMA 지배를 벗어난 23개(50% 미만)에는 못 든다.
상세는 [13-NPU-실기-매트릭스](./13-NPU-실기-매트릭스.md).

## 9. 어느 엔진이 무엇을 했나 — 한눈에

| 연산 | 담당 엔진 | vISA 호출 |
|---|---|---|
| HBM ↔ DM 이동 | DMA (tdma) | `to_dm`, `to_hbm` |
| TRF 상주 | Fetch+Collect (sub) | `fetch.collect.to_trf` |
| 행렬곱 (곱-누산) | **Contraction (DPE)** | `contract_outer/packet/time/lane` |
| bias 더하기 | **Vector Engine** | `vector_clip_zip(Add)` |
| ReLU | **Vector Engine** | `vector_clip(Max, 0.0)` |
| 정밀도 변환 | Cast | `.cast::<bf16>` |
| 데이터 재배치 | Switch / Transpose | `switch`, `transpose`, `reshape` |
| argmax | (호스트 CPU) | `mnist_tests.rs` `max_by` |

## 10. 한 줄 요약

**furiosa-opt는 MNIST를 "행렬곱 두 번"이 아니라,
HBM→DM→TRF 메모리 계층을 통과시키며 DPE로 곱-누산하고,
Vector Engine의 클립으로 bias-add와 ReLU를 붙이고,
Switch/Transpose로 bias를 맞추고, 그 흐름을 스케줄러가 겹쳐 실행하는 방식으로 연산한다.**
forward 전체가 단일 68KB EDF 커널이고, argmax만 호스트에서 한다.

## 11. 어떻게 실행했나 — 실제 절차

이 예제는 벤더 리포(`furiosa-opt-examples`)에 있지만, 우리 서버 래퍼가 **0.4 세대**라 0.3 핀 벤더 트리를
그대로는 `--backend npu` 로 못 돌린다(typecheck/emulation 은 동작).
그래서 예제 파일들을 0.4 규격의 **독립 프로젝트**로 재구성해 격리 실행했다(사용자 원본 `/home/jun/yik`은 안 건드림).

> **함정 — MNIST 만 돌리려도 크레이트를 쪼개야 한다.** `--backend npu` 는 패키지 안의 *모든* `#[device]`
> 함수를 빌드 시점에 EDF 로 낮춘다. **테스트가 그 함수를 부르든 말든 상관없고, 하나라도 못 낮추면 크레이트
> 전체가 죽는다.** 벤더 예제 크레이트를 그대로 `--backend npu` 로 돌리면 **63개 에러로 빌드 실패, 테스트
> 0개 실행**이다 — `test_mnist` 도 시작조차 못 한다. 아래처럼 MNIST 만 담은 단일 모듈 프로젝트로 격리한 것이
> 사실상 그 우회다. 원 크레이트를 유지하며 우회하려면 실패 커널에만 `#[cfg(not(backend = "npu"))]` 게이트를
> 넣는 쪽이 낫다 → [12-예제-전수실행](./12-예제-전수실행.md).

### 11.1 프로젝트 구성 (한 번)
```bash
W=/home/jun/.claude/jobs/46bc5c7e/tmp/visa_mnist        # 격리 작업 디렉터리
# 벤더 main 브랜치에서 소스·가중치를 가져와 배치:
#   src/mnist.rs          ← furiosa-opt-examples/src/mnist/mod.rs
#   tests/mnist_tests.rs  ← furiosa-opt-examples/tests/mnist_tests.rs (모듈명만 visa_mnist 로 치환)
#   data/mnist/mnist.safetensors ← furiosa-opt-examples/data/mnist/mnist.safetensors
#   src/lib.rs, Cargo.toml, rust-toolchain.toml ← base-template 규격으로 신규 작성
```
Cargo.toml 의 핵심 3줄(이게 없으면 실행 불가):
```toml
[package.metadata.furiosa-opt]        # ← 0.4 래퍼가 요구하는 커널 패키지 마커
[dependencies]
furiosa-opt-std = "0.4"               # ← 0.4 (설치 래퍼 세대와 일치)
safetensors = "0.4"                   # ← 가중치 로드
```

### 11.2 3단계 실행 (실제로 한 순서)
```bash
. "$HOME/.cargo/env"                  # cargo 가 어느 셸 PATH 에도 없으므로 필수
cd $W

# ① 매핑 검증 (NPU 불필요) — 모든 커널이 하드웨어 제약을 만족하나
export CARGO_TARGET_DIR=$W/target_tc
cargo furiosa-opt --backend typecheck test --release
#   → 빌드·매핑 검증은 통과. 단 test_mnist 자체는 FAILED 다 — phantom 텐서라 logits 가 비어
#     `mnist_tests.rs:47` 의 `max_by(...).unwrap()` 이 None 에서 패닉한다(오류 아님, 값 미계산의 결과)

# ② 호스트 수치 검증 (NPU 불필요) — 10장 분류가 실제로 정답인가
export CARGO_TARGET_DIR=$W/target_emu
cargo furiosa-opt --backend emulation test --release
#   → test_mnist ... ok  (10/10 정답. 호스트 CPU 가 커널을 해석 실행)

# ③ 실기 (NPU 필요) — 진짜 RNGD 에서 도나
#   먼저 npu0 유휴 확인:  furiosa-smi status | grep npu0   → 0.00/47.50 GiB
export CARGO_TARGET_DIR=$W/target_npu
cargo furiosa-opt --backend npu test --release
#   → test_mnist ... ok  (0.12s, 실기 10/10 정답)
```

### 11.3 프로파일 (NPU 불필요, 카드 안 잡음)
```bash
cargo furiosa-opt compile forward --dump-schedule prof/fwd_sched.json --dump-summary prof/fwd_sum
#   → §7 의 22-인스트럭션 사이클 표가 여기서 나옴
```

### 11.4 실행 시 하드웨어가 로드하는 것 (npu 백엔드)
`--backend npu test` 는 두 단계다:
1. **빌드타임**: `furiosa-opt-driver`(rustc 플러그인)가 `#[device] forward` 를 찾아 vISA→LIR→EDF 로 낮춰
   `forward.edf`(68,106 B) 생성 → `target_npu/furiosa-opt/kernel/visa_mnist/` 에 저장.
2. **런타임**: 테스트 바이너리가 그 EDF 를 로드해 칩에 올리고, `mnist.safetensors` 가중치를 HBM 으로 DMA 한 뒤
   이미지 10장을 하나씩 `launch(forward, ...)` 로 실행, 결과 logits 를 호스트로 읽어 argmax→라벨 대조.
   (NPU 네이티브 호출은 `furiosa-opt-std-0.4.0/vendor/.../libdevice_runtime.a`(10.7 MB, 정적 링크)를 통함.)

## 12. 어느 위치의 어떤 파일이 쓰였나

### 12.1 프로젝트 파일 (직접 만들거나 벤더에서 가져온 것)
위치: `/home/jun/.claude/jobs/46bc5c7e/tmp/visa_mnist/`

| 파일 | 크기 | 출처 | 역할 |
|---|--:|---|---|
| `src/mnist.rs` | 6,956 B | 벤더 `furiosa-opt-examples/src/mnist/mod.rs` | **커널 정의** (fc1/fc2/forward, §2~6) |
| `tests/mnist_tests.rs` | 1,867 B | 벤더 `.../tests/mnist_tests.rs` | **실행 진입점** (가중치 로드→launch→argmax→assert) |
| `data/mnist/mnist.safetensors` | 859,220 B | 벤더 `.../data/mnist/mnist.safetensors` | **학습 가중치**(fc1/fc2 W·b) + 이미지 10장 + 라벨 |
| `src/lib.rs` | 110 B | 신규 작성 | `pub mod mnist;` (register_tool) |
| `Cargo.toml` | 481 B | 신규 작성 | 0.4 의존성 + `[package.metadata.furiosa-opt]` 마커 |
| `rust-toolchain.toml` | 78 B | base-template | `channel = "nightly-2026-05-01"` 핀 |

### 12.2 툴체인·크레이트 (설치된 외부 의존)
| 파일/디렉터리 | 크기 | 역할 |
|---|--:|---|
| `~/.cargo/bin/cargo-furiosa-opt` | 3.5 MB | 래퍼(0.4 세대, `--backend`·커널컴파일 트리거) |
| `~/.cargo/bin/furiosa-opt-driver` | 111 MB | rustc 플러그인(vISA→EDF 로우어링) |
| `~/.cargo/registry/src/.../furiosa-opt-std-0.4.0/` | — | vISA 표준 라이브러리(엔진·매핑·백엔드) |
| `.../furiosa-opt-std-0.4.0/vendor/x86_64-unknown-linux-gnu/libdevice_runtime.a` | 10.7 MB | NPU 네이티브 런타임(정적 링크, 실기 실행 시) |
| `nightly-2026-05-01` 툴체인 | — | 드라이버 ABI 가 이 nightly 에 고정됨 |

### 12.3 컴파일 산출물 (실행에 로드된 것)
위치: `.../visa_mnist/target_npu/furiosa-opt/kernel/visa_mnist/`

| 파일 | 크기 | 역할 |
|---|--:|---|
| `visa_mnist::mnist::forward.edf` | 68,106 B | **실기에서 칩에 올라간 커널** (forward 전체) |
| `visa_mnist::mnist::forward.bin` | 53,229 B | 커널 바이너리 |
| `visa_mnist::mnist::forward.hash` | 64 B | 캐시 무효화용 해시 |

### 12.4 프로파일 산출물 (§7 표의 출처)
위치: `.../visa_mnist/prof/`

| 파일 | 역할 |
|---|---|
| `prof/fwd_sched.json` | **22-인스트럭션 스케줄** (엔진별 begin/end 사이클, 소스라인 귀속) |
| `prof/fwd_sum/summary.log`, `summary.json` | 요약 통계(main/io/sub %, computation_cycle, dram io 바이트) |
| `prof/fwd_sum/dot/{lir,edf}.dot` | IR 그래프(그래프뷰용) |

---

### 관련 문서
- [05-축약엔진-DPE](./05-축약엔진-DPE.md) — fc1/fc2 matmul 의 4단계 상세
- [06-벡터-캐스트-전치](./06-벡터-캐스트-전치.md) — ReLU=vector_clip, bias transpose
- [07-스케줄링](./07-스케줄링.md) — 두 레이어 겹침, main/sub/tdma 동시성
- [00-개요와-환경](./00-개요와-환경.md) — 0.4 세대 래퍼·`[package.metadata.furiosa-opt]` 마커 상세
