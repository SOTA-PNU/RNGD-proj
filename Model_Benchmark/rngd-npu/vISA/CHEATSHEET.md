# vISA 빠른 참조

이 문서는 vISA API·매핑 연산자·명령어 빠른 참조입니다.

> 이름은 모두 `furiosa-opt` v0.2.0 소스에서 확인했습니다. 정확한 시그니처는 rustdoc(<https://developer.furiosa.ai/furiosa-opt/rustdoc/furiosa_opt_std/>)이나 `reference/examples/`를 보세요.

## 하드웨어 상수 (RNGD)
| 항목 | 값 | 비고 |
|---|---|---|
| Cluster / chip | 2 | `fetch.rs:50` 단언 |
| Slice / cluster | 256 | `fetch.rs:51` 단언 |
| Lane / slice | 8 | Contraction MAC 한 행 |
| flit | 32 byte | Collect가 정규화하는 단위 |
| packet | 64 byte | Contraction Outer 입력 |
| DM | 512 KB/slice (총 256 MB) | 주 작업 메모리 |
| TRF | 8 KB/lane (Full=65,536 B/slice) | Contraction 레지스터, `FirstHalf`/`SecondHalf`/`Full` |
| VRF | 8 KB/slice | Vector 레지스터, 자유 분할 |
| Sequencer | 8 entries / 65,536 iters | 루프 한도 |

## 백엔드 & 명령어
| 백엔드 | 무엇 | NPU? |
|---|---|---|
| `simulation`(기본) | 호스트에서 실제 값 계산 | ❌ |
| `typecheck` | 빈 텐서로 본문 실행 → 매핑/모양 단언 | ❌ |
| `emulation` | 버퍼 에뮬(문서엔 없음) | ❌ |
| `npu` | 컴파일된 `.bin`을 칩에 디스패치 | ✅ SDK 필요 |
```bash
cargo furiosa-opt run  --release --bin NAME            # 시뮬레이션 실행
cargo furiosa-opt --backend typecheck run --release --bin NAME
cargo furiosa-opt test --release --bin NAME            # 레퍼런스 대비 검증
FURIOSA_OPT_NPUS=0,1 cargo furiosa-opt --backend npu run --release --bin NAME
```
`cargo check`는 본문 실행 안 함 → 매핑 단언 보려면 `--backend typecheck run`.

## 매핑 언어 `m![]` (`mapping-expressions.md`)
| 연산자 | 이름 | 뜻 | 예 |
|---|---|---|---|
| `A` | Symbol | 축 A 그대로 (크기=A) | `m![A]` |
| `,` | Pair | 두 축 합침 (왼쪽이 상위), 우결합 | `m![A, B]` = `Pair<A,Pair...>` |
| `/ n` | Stride | n으로 나눈 **바깥(블록) 인덱스** (크기=A/n) | `m![A / 8]` |
| `% n` | Modulo | n으로 나눈 **안쪽 위치** (크기=n) | `m![A % 8]` |
| `# n` | Padding | 하드웨어 단위 수 n으로 패딩(남는 칸=임의값) | `m![A / 8 # 256]` |
| `= n` | Resize | 논리 크기를 n으로 자름(축소) | `m![D = 2]` |
| `1` | Identity | 1칸짜리, Pair의 항등원 (0.3·0.4 에서는 `Broadcast{size:1}`) | `m![1]` |
| `{ X }` | Escape | 타입 별칭 X를 매핑에 끼움 | `m![{ L }, { R }]` |
| `B' = B - A` | Skew | 대각 접근(wavefront) — **책에만 있음, 크레이트 미구현** | `m![A, B' = 4]` |
| `$(e1:n1,...)` | Sliding | 겹치는 블록(conv) 선형결합 — **책에만 있음, 크레이트 미구현** | — |

> ⚠ **Skew·Sliding 은 배포된 크레이트에 없다.** 벤더 책(`reference/book/mapping-tensors/mapping-expressions.md:379-438`)에는
> 설명이 있지만 `furiosa-mapping-types`·`furiosa-mapping-macro` **0.2.0·0.3.0·0.4.0 소스 전체에서 식별자 0건**이다(실측).
>
> `Mapping` 변종도 버전마다 다르다 — **0.2.0**(이 저장소 `experiments/Cargo.toml` 이 거는 버전)은
> `Identity, Symbol, Stride, Modulo, Resize, Padding, Pair`, **0.3.0·0.4.0** 은
> `Symbol, Stride, Modulo, Resize, Padding, Pair, Broadcast` 다.

- `m![A / 8, A % 8]` ≡ `m![A]` (stride·modulo 분해, A가 8로 나눠떨어질 때).
- 디바이스 텐서 매핑 순서: `<dtype, Chip, Cluster, Slice, (Lane), (Time), Packet/Element>`.
- 예: `DmTensor<bf16, m![1], m![1 # 2], m![A / 8 # 256], m![A % 8]>` = 칩1·클러스터2중1·256슬라이스에 8개씩.
- `axes![A = 2048, B = 512];` 로 축·크기 선언(중복 이름은 컴파일 오류).
- 나눠떨어짐은 컴파일타임 const 단언으로 검사(`Stride`: `L::SIZE % SIZE == 0`).

## 메모리 텐서 타입 & 이동 (`tensor/memory.rs`)
| 타입 | 위치 |
|---|---|
| `HostTensor<D, E>` | 호스트 |
| `HbmTensor<D, Chip, E>` | HBM |
| `DmTensor<D, Chip, Cluster, Slice, E>` | DM(SRAM) |
| `TrfTensor<D, Chip, Cluster, Slice, Lane, E>` | TRF |
| `VrfTensor<D, ...>` | VRF |

이동(타입 바뀌며 정렬 단언 동반):
```rust
let dm  = hbm.to_dm::<Cluster, Slice, Element>(&mut ctx.tdma, addr);   // HBM→DM
let hbm = dm.to_hbm(&mut ctx.tdma, addr);                              // DM→HBM
let trf = chain.to_trf(TrfAddress::Full);    // Full | FirstHalf | SecondHalf
let vrf = chain.to_vrf(addr);
let emb = table.dma_gather(&indices, addr, true);   // 임베딩/KV 룩업
// dma_scatter, tile(zero-copy view), view() 도 있음
```
호스트:
```rust
let mut ctx = Context::acquire();
let h = HostTensor::<i32, m![A]>::rand(&mut rng);   // 또는 ::uninit()
let in_hbm = h.to_hbm(&mut ctx.pdma, 0).await;
let out = launch(kernel, (&mut ctx, &in_hbm)).await;
let v: Vec<i32> = out.to_host::<m![A]>(&mut ctx.pdma).await.to_buf();
```

## 텐서유닛 파이프라인 (메서드 체인, 순서 고정)
`begin → fetch → [switch] → collect → [contraction | vector | cast | transpose] → commit`
```rust
ctx.main.begin(dm.view())
  .fetch::<dtype, Time, Packet>()
  .switch(SwitchConfig::Broadcast01 { slice1, slice0, time0 })   // 선택
  .collect::<Time2, Packet2>()
  // --- 연산 엔진 (택1 계열) ---
  .contract_outer::<...>(&trf).contract_packet::<...>().contract_time::<...>().contract_lane::<...>(LaneMode::Interleaved)
  // 또는 vector_init().vector_*().vector_final()
  .cast::<OutD, OutPacket>()      // dtype 변환 (선택)
  .commit::<Element>(addr);
```
- `begin`은 `ctx.main` 또는 `ctx.sub`(보통 프리페치). collect 전에 commit 부르면 **타입 오류**.

### Switch 토폴로지 (`engine/switch.rs`)
`Broadcast01{slice1,slice0,time0}` · `Broadcast1{slice1,slice0}` · `Transpose{slice1,slice0}` · `InterTranspose{slice1,slice0,time0}` · `CustomBroadcast{ring_size}` (+ Transpose+Broadcast1 결합형)

### Contraction (`engine/contraction/`)
- `contract_outer(&trf)` — Outer/Stream Adapter, 정지된 RHS(TRF) × 스트림 LHS
- `contract_packet` — 공간(패킷) 축약 (하드웨어 리덕션 트리)
- `contract_time` — 시간(타일 반복) 누적
- `contract_lane(LaneMode::{Interleaved|Sequential})` — 8 레인 접기
- Reduce 연산: `AddSat`, `Max`, `Min`

### Vector 엔진 (`engine/vector/`)
진입/종료: `vector_init()` … `vector_final()`. 태그: `vector_intra_slice_tag(TagMode::{Zero|AxisToggle{axis}|ValidCount|...})`
- 고정소수점: `vector_fxp(FxpBinaryOp, imm)` — `AddFxp/AddFxpSat/SubFxp/SubFxpSat/MulFxp/MulInt/LeftShift(Sat)/LogicRightShift/ArithRightShift(Round)`
- 부동소수점: `vector_fp_unary(FpUnaryOp)` — `Exp/NegExp/Sqrt/Tanh/Sigmoid/Erf/Log/Sin/Cos` · `vector_fp_binary(FpBinaryOp)` — `AddF/SubF/MulF/MaskMulF/DivF` · `vector_fp_ternary(FpTernaryOp)` — `FmaF/MaskFmaF` · `vector_fp_div`
- 변환: `vector_fp_to_fxp` / `vector_fxp_to_fp`
- 리덕션: `vector_intra_slice_reduce` (슬라이스 안), `vector_inter_slice_reduce` (슬라이스 간)
- 기타: `vector_clip(_zip)`, `vector_logic(_zip)`, `vector_filter`, `vector_stash`, `vector_narrow_clip/split`, `vector_widen_concat/pad`, `vector_*_zip`(두 입력), `*_with_mode`(반올림/포화 모드)
- → softmax = max(inter_slice_reduce Max) → SubF → Exp → sum(reduce AddSat) → DivF 조합

## 커널 선언 & 스케줄
```rust
#[device(chip = 1)]
pub fn my_kernel(ctx: &mut Context, x: &HbmTensor<...>) -> HbmTensor<...> { ... }
```
- 컨텍스트: `ctx.main`(주 연산) · `ctx.sub`(프리페치) · `ctx.tdma`(텐서 DMA) · `ctx.pdma`(PCIe DMA). 서로 다른 컨텍스트는 **병렬**, 같은 컨텍스트는 직렬.
- 스케줄러는 **재배치 안 함**(작성 순서 그대로). 해저드(RAW/WAR/WAW)는 주소 분석으로 자동 대기 삽입. **주소는 사람이 배정**(겹치면 안 됨).
- 이중 버퍼링: TRF `FirstHalf`↔`SecondHalf` 번갈아 → sub가 채우는 동안 main이 읽음.
- ⚠️ 같은 뱅크 64회 충돌 → **클러스터 리셋(치명)**. (`memory-performance.md`)

## dtype (`scalar.rs`, `cast.rs`)
`i4 · i8 · i32 · bf16 · f8e4m3 · f32`. Fetch/Contraction 시 합법 변환만(예: bf16→f32 누적, i8→i32). `cast()`로 명시 변환.

## 새 bin 추가 (필수 절차)
1. `src/kernel/x_kernel.rs` 작성 → 2. `src/kernel/mod.rs`에 `pub mod x_kernel;` → 3. `src/x.rs` 호스트 → 4. `Cargo.toml`에 `[[bin]] name="x" path="src/x.rs"`. (`src/bin/`·`tests/`·`examples/`는 스캔 안 됨)
