# 이 서버의 검증된 사실 (작성 에이전트는 반드시 이대로 쓸 것)

## 툴체인 세대 (중요 — 앞선 오류를 반복하지 말 것)
- 설치된 래퍼: `~/.cargo/bin/cargo-furiosa-opt`, `--version` → "0.3.0" 을 출력하지만
  **바이너리는 로컬 v0.4.0 타르볼과 md5 동일(457f51daad8171abbad51a8602566a0e, 3,538,576 B, 7/20 설치)**.
  즉 **0.4 세대 래퍼**다. `--version` 문자열로 세대를 판단하지 말 것.
- `cargo furiosa-opt --help` 실측: `--backend` possible values = **[typecheck, emulation, npu]**,
  **default = emulation**. `simulation` 백엔드는 **없다**(0.3 시절 이름). `--backend simulation` 은 거부됨.
- 커널 패키지 마커: 현재 래퍼는 **`[package.metadata.furiosa-opt]`** 를 Cargo.toml 에 요구한다.
  구세대의 `src/furiosa-opt.tag` 0바이트 마커만 있으면 **"no kernel packages found" 로 거부**된다.
- 커널 산출물 명명: **`pkg::mod::fn.{bin,edf,hash}`** (구세대는 `kernel__mod__fn.*`).

## 사용자 프로젝트 /home/jun/yik 의 현실
- `furiosa-opt-std = "0.3"` 을 핀하고 `src/furiosa-opt.tag` 를 쓴다 → **현재 래퍼로는 `--backend npu` 불가**.
  (실측: `cargo furiosa-opt compile gemm_kernel` → "no kernel packages found".)
  npu 를 쓰려면 (1) Cargo.toml 에 `[package.metadata.furiosa-opt]` 추가, (2) 커널을 0.4 API 로 포팅해야 한다.
- `typecheck` / `emulation` 백엔드는 잘 돈다(NPU 불필요).
- 5개 예제 bin: constant_add, elementwise_mul, dot_product, gemv, gemm.
- `default-run` 이 없어 `--bin <name>` 없이 `run` 하면 cargo 가 "could not determine which binary" 로 죽는다.

## 0.3 → 0.4 API 차이 (실측)
- `contract_outer` 제네릭: 0.3 은 4개, **0.4 는 5개** (`::<Time, Packet, _, _, _>`).
- 0.4 는 to_dm/to_hbm/to_trf/commit 의 **명시적 주소 인자를 제거**했다 (0.3 은 `to_dm(&mut ctx.tdma, 0)`, 0.4 는 `to_dm(&mut ctx.tdma)`).
- 버퍼 접근자: 0.3 `to_buf()`, **0.4 `into_vec()`**.
- `runtime::Npu` (0.3) → `backend::` 모듈 (0.4).

## 사용자의 vISA 커리큘럼
- /home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vISA/ 의 reference/book 은 **v0.2.0** 기준.
  현재 문서(v0.4.0)와 백엔드 이름·API 가 다르다. 커리큘럼의 `--backend simulation` 언급은 현재 래퍼에서 안 통한다.

## 알려진 문서/코드 오류 (그대로 옮기지 말 것)
- `yik/src/kernel/gemm_kernel.rs:8` 주석 "Each slice handles a 16 × 16 output tile" 은 **틀렸다**.
  Slice = m![I/32, J/32] = 16×16 **격자 = 256 슬라이스**, 각 슬라이스가 담당하는 타일은 m![I%32, J%32] = **32×32**.
  검산: 256 슬라이스 × 1024(32×32) = 262,144 = I·J. (256×256 ≠ I·J.)
- 같은 커널의 "Switch Engine distributes..." 주석 — 실제로 `switch()` 는 호출되지 않는다. 분배는 to_dm 의 Slice 타입 파라미터가 한다.

## TCL vs vISA 계층 경계 (실측)
- TCL(furiosa.tcl)의 배치 어휘는 `tcl.context(layout={Chip, Cluster, Split})` 가 전부.
  Slice / Lane / Packet / TRF / VRF / commit-trim 은 TCL 공개 API 에 **0건** — 즉 TCL 은 클러스터 경계에서 멈추고,
  **슬라이스 내부 데이터패스(Lane≤8, Packet≤64B, 레지스터파일)는 vISA 만 노출**한다.
- 계측: `furiosa-tcc` 는 덤프/프로파일 옵션이 **전무**. `cargo furiosa-opt compile` 은
  `--dump-schedule/--dump-ir/--dump-visa/--dump-graph/--dump-summary` 를 제공(NPU 점유 없이 사이클 단위 스케줄).

## 클럭 도메인 (moving-tensors 문서 실측)
- HBM 채널 컨트롤러 0.75 GHz(버스트 8회 → 실효 6 GHz), DRAM 타이밍 표는 1.5 GHz, DMA/코어 예제는 1 GHz 기준.
- HBM 1.5 TB/s per chip (32ch × 48 GB/s), DMA 엔진 256 GB/s.

## 하드웨어 사양 (computing-tensors 문서 실측)
- chip당 512 spatial cells, Lane ≤ 8, Packet ≤ 64 B(=bf16 32개), cluster당 256 slice, chip당 2 cluster.
- Contraction(DPE) 은 bf16/f8 → f32, i4/i8 → i32 로 확장 누산. Vector(VE) 는 i32/f32 만.
- DM(SRAM): 총 256 MB, slice당 512 KB. TRF 8 KB/lane(8 lane/slice). VRF 8 KB/slice.

---
---

# ▣ 2026-07-24 실기(real NPU) 실측으로 추가·정정된 사실

> 위쪽(1~51행)은 **행 번호가 다른 문서에서 참조되고 있어 그대로 보존**한다. 아래는 전부 추가분이다.
> 근거 로그: `/home/jun/.claude/jobs/46bc5c7e/tmp/ex_logs/`.
> 상세는 [12-예제-전수실행](./12-예제-전수실행.md)(3개 백엔드 전수 조사·게이팅 레시피)과
> [13-NPU-실기-매트릭스](./13-NPU-실기-매트릭스.md)(실기 전용 매트릭스·결함 유형·사이클 특성)에 있다.

## ★ N1. `--backend npu` 의 제1 규칙 (이걸 모르면 무조건 막힌다)

> **`--backend npu` 는 패키지 안의 *모든* `#[device]` 함수를 빌드 시점에 EDF 로 낮춘다.
> 테스트가 그 함수를 부르든 말든 상관없다. 하나라도 못 낮추면 크레이트 전체가 죽는다.**

- 벤더 예제 크레이트(`furiosa-opt-examples`)를 그대로 `--backend npu` 로 돌리면
  **커널 에러 63개로 빌드 실패, 테스트는 0개 실행**된다.
- `typecheck`·`emulation` 은 커널을 **호출될 때** 처리하므로 아무도 안 부르는 커널은 그냥 지나간다.
  그래서 두 백엔드는 통과하는데 `npu` 만 죽는다.
- 검증된 우회법 2가지:
  1. **서브셋 패키지** — 되는 모듈만 골라 새 크레이트 (실기 테스트 21개)
  2. **게이팅**(권장) — 실패 커널에만 `#[cfg(not(backend = "npu"))]`.
     **상류가 이미 쓰는 관용구**(`tests/matmul_tests.rs:124`). 실기 테스트 **89개**로 확대.
- 이것이 MNIST 를 돌릴 때 `visa_mnist` 라는 **별도 패키지**가 필요했던 진짜 이유다.

## N2. 실기 실행은 열려 있다 (기존 "npu0 점유로 blocked" 기술은 폐기)

- `furiosa-smi status` 실측: **npu0~npu3 전부 `alive`, `0.00/47.50 GiB`** (실행 전·후 동일).
- 이 세션에서 **실기 테스트 89개를 실제로 실행**했다. 자원 누수 0.
- 라우터(`furiosa_router.py`)는 상주하지만 **백엔드 모델이 미기동**이면 NPU 를 잡지 않는다.
- → `OPTIMIZATION-SURFACE.md` L4 #20 이 적었던 "npu0 이 PID 1215564 / 메모리 97.86% 로 점유 중이라
  실기 불가" 는 **더 이상 사실이 아니다**(해당 문서에서 정정함).

## N3. 실기 테스트 매트릭스 (테스트마다 별도 프로세스로 격리 실행)

| 항목 | 값 |
|---|---|
| NPU 백엔드에 존재하는 테스트 | **89** |
| PASS | **80** |
| FAIL | 5 |
| ABORT(프로세스 즉사) | 3 |
| `#[ignore]` 미실행 | 1 |
| **실기 정상 동작으로 봐야 하는 수** | **83 / 89 (93.3%)** |

`83` = 80(통과) + 2(커널은 옳고 테스트 기준이 과함, N5-④) + 1(`#[ignore]` 지만 강제 실행하면 통과).

바이너리별: `switch_assertions` **18/18**, `contract_outer_assertions` **13/13**, `at_primitives` 4/4,
`vector_engine` 32/36, `fetch_assertions`·`memory_op`·`param` 각 2/2, **`mnist` 1/1**,
`binary_add`·`contract_element_types_answer`·`fetch_commit`·`scatter_gather`·`transpose` 각 1/1,
`tile` 1/2, `broadcast` 0/1, `reshape` 0/2, `shuffle_slice` 0/1.

### ★ N3-1. 격리 실행이 필수다 — 안 하면 숫자가 3배 틀린다

| 실행 방식 | vector_engine 통과 | 실패 |
|---|--:|--:|
| 한 프로세스에 전부 (평범한 `cargo test`) | 10 | 25 |
| **테스트마다 새 프로세스** | **33** | **3** |

> 두 행의 모수는 똑같이 36 이다(`#[ignore]` 1건 포함). 격리 열의 **33** = 정규 통과 **32**(N3 표의
> `vector_engine 32/36`) + `#[ignore]` 라 실행되지 않아 프로세스가 `ok` 로 끝난 1건.
> 단일 프로세스 행은 그 1건을 어느 열에도 넣지 않아 10+25=35 로 보인다.

기전: hang 커널 하나가 HAL `-110`(ETIMEDOUT)을 유발하면 **그 프로세스의 이후 모든 커널 실행이
전부 `-110` 으로 실패**한다. 멀쩡한 커널 22개가 "실패"로 집계된다.
`--test-threads=1` 로 직렬화해도 소용없다(동시성 문제가 아니라 **프로세스 상태 오염**).
→ **실기 결과를 믿으려면 프로세스를 격리하라.** 한 번에 돌린 "N개 실패"는 상한이 아니라 **과대계상**이다.

## ★ N4. 사이클은 DMA 가 지배한다 (커널 130개, 스케줄 모델)

`cargo furiosa-opt compile --dump-schedule` 로 실기 컴파일되는 커널 130개를 합산.
**이 값은 컴파일러의 스케줄 모델 예측이며 실측 벽시계가 아니다** — 단, `mnist::forward` 에 대해
17,953 cycle / 22 instruction 을 재현해 독립 기록과 정확히 일치함을 확인했다
(분석기 검증 기준점 — 두 기록이 같은 스케줄 모델 산출물이므로 **벽시계 대조는 아니다**).

| 엔진 | 총 사이클 | 비중 | 인스트럭션 |
|---|--:|--:|--:|
| **DmaEngine** | **75,464,336** | **96.5%** | 470 |
| PeCore | 2,586,167 | 3.3% | 1,557 |
| MainContext | 58,883 | 0.1% | 108 |
| InterChipTransfer | 38,018 | 0.0% | 2 |
| VectorEngine | 14,770 | 0.0% | 50 |
| SubContext | 9,737 | 0.0% | 27 |

- **인스트럭션 수는 PeCore 가 1,557개로 최다인데 사이클은 3.3%.** DmaEngine 은 470개로 96.5%.
  → DMA 인스트럭션 하나가 연산 인스트럭션 하나보다 두 자릿수 이상 비싸다.
- 합산의 착시가 아니다: **커널 130개 중 107개(82%)가 DMA 에 50% 이상**, 54개는 90% 이상,
  **중앙값 82.8%**.
- 사이클 스팬: min 16 / p25 4,612 / **median 10,532** / p75 23,503 / max 10,845,036.
  최대 커널(`at_primitives::vrf::multi_vrf_at`)은 **인스트럭션 12개로 1,080만 사이클** — 명령 수와 비용이 비례하지 않는다.
- `--dump-schedule` 은 **NPU 를 점유하지 않는다** → 실기 테스트와 동시 실행 가능.

> **함의**: vISA 가 배타적으로 노출하는 슬라이스 내부 데이터패스(Lane/Packet/TRF/DPE/VE)의
> 최적화 상한은 **3.3%**(PeCore 총 점유)다. 실기에서 중요한 건 **데이터가 어떻게 놓이고 움직이는가**다.

## N5. 실기 결함 4유형 (6개 테스트가 여기 걸린다)

| 유형 | 건수 | 정체 | 위험도 |
|---|--:|---|---|
| ① 커널 로더 범위초과 → 프로세스 abort | 3 | `device-runtime-c/src/kernel.rs:137` | 높음(즉사) |
| ② **조용한 데이터 오배치** | 2 | 에러 없이 틀린 위치/값 | **가장 높음**(안 들킴) |
| ③ 커널 hang → HAL 타임아웃 | 1 | `os error -110`(ETIMEDOUT) | 높음(연쇄 오염) |
| ④ ~~반올림·순서 차이~~ | (2) | 커널은 옳음, 테스트 기준이 과함 | **버그 아님** |

- **①** `reshape` ×2 + `chip_shuffle`. 요구 크기가 실제의 **약 1.5배**(50,560/33,792 · 56,576/37,888)
  → 단일 크기계산 결함의 서명. 발생 지점은 **커널 로드**(`furiosa_kernel_load`)이고
  `cannot unwind` 로 **테스트 바이너리 전체가 abort**된다. **세 건 모두 에뮬레이션에서는 통과.**
- **②(a)** `broadcast::test_view_broadcast`: HBM→HBM 브로드캐스트 DMA 가 **목적지에 아무것도 쓰지 않는다.**
  2048/2048 전부 불일치, 2회 실행에서 값이 완전히 동일(결정적), 그 값을 f32 로 재해석하면
  `-0.283, 0.251, …` → **묵은 잔류 데이터를 읽는 것**. 에뮬레이션에서는 통과.
- **②(b)** `tile_tests::test_tile_window_commit_host`: **데이터는 온전한데 쓰인 위치만 틀렸다.**
  `result[32..64]` 에 가야 할 `input[0..32]` 가 `result[8..40]` 에 착지.
  산술: 32 elements × 4 B = 128 B 여야 하는데 실제 착지는 8 elements = **32 B**
  → 32 가 "요소 수" 대신 **"바이트 수"로 적용**된 것(정황 근거이며 런타임 소스 확인은 아님).
- **③** `vector_engine::normal::test_ve_stash_fp_fp`: **단독 실행에서도 60초 타임아웃.**
  NPU 자체는 손상되지 않는다(직후 `alive / 0.00 GiB`). 프로세스 종료 시 회복.
- **④** 1 ULP 1건(512개 중 489개 비트 동일, 최대 ULP 1, 최대 상대오차 1.742e-07) + **리듀스 순서** 1건.
  후자는 처음엔 오염처럼 보였다(387/512 불일치, `i32::MAX` 출현). 원인은 `saturating_add` 가
  **결합법칙을 만족하지 않기** 때문 — 포화 불가능한 작은 값(−50..50)을 넣으니 **실기에서 통과(0.12s)**.
  → **비결합 연산(`saturating_add`, 부동소수 덧셈)은 순서 독립 입력으로 재검증해야 한다.**

> **②가 왜 최악인가**: ①③은 시끄럽게 죽어서 바로 안다. ②는 **숫자가 조용히 틀린다.**
> 정답을 검증하는 테스트가 없으면 그냥 지나간다. **실기로 옮길 때 값 검증은 필수다.**

## N6. 커널 컴파일 매트릭스 — 그리고 개수 4종이 공존한다

**소스 추출 200개 중 137 OK / 63 FAIL** (68.5%).

> ### ★ 개수를 "통일"하려 하지 말 것 — 넷 다 맞다
> - **207** = 툴 자체 집계(`Finished 1 compiled, 206 filtered out`). 제네릭 단형화 포함. 가장 정확.
> - **200** = 소스 텍스트에서 `#[device]`+`pub fn` 으로 추출한 수. 매트릭스의 분모.
> - **143** = 게이팅 후 실기 빌드된 수. **207 − 게이트 64 = 143** 으로 정합.
> - **63 vs 64**: 게이팅에는 보정 전 목록(64)을 썼고 그중 1건이 거짓 FAIL(N7)이라 진짜 FAIL 은 63.

### N6-1. 실패 분류 (전수 조사 + 적대적 재검증) — **보정 전 64개 목록 기준**

| 분류 | 수 | 의미 |
|---|--:|---|
| **REAL_LOWERING_GAP** | **24** | 멀쩡한 커널인데 백엔드가 아직 못 낮춘다 ← 진짜 공백 |
| INTENTIONAL_NEGATIVE | 23 | 잘못된 매핑을 문서화하려 만든 표본. **실패가 정상** |
| **COMPILER_ICE** | **13** | 문구가 `internal compiler error` — 컴파일러 자체 버그 |
| GENERIC_NOT_MONOMORPHIZED | 2 | 제네릭 device 함수라 구체 래퍼 필요 |
| UNCLEAR | 2 | 근거 부족 |

> 표의 합은 **64** 다(24+23+13+2+2). 분류는 게이팅에 쓴 **보정 전 목록**으로 돌렸고, 그중 1건이
> 거짓 FAIL(N7)이라 진짜 FAIL 은 **63** 이다. 산수 오류가 아니다.

> **앞선 추정 정정**: `12-예제-전수실행.md` §5.1 에서 `*_assertions` **28개**를 "의도적 표본"으로 묶었으나,
> 전수 분류 결과 **의도적인 건 23개뿐**이고 나머지 40개는 진짜 공백이거나 컴파일러 버그다.
> **이름으로 판단하면 틀린다** — `contract_outer_assertions::lane_size::valid_size_{1,2,4}` 는
> 이름이 `valid_*` 인데 실기 컴파일에 **실패**한다.

### N6-2. 실패에서 읽히는 패턴 3가지

1. **정렬(alignment)이 반복해서 발목을 잡는다.** `not aligned by 8`(DMA 시퀀서 8B 정렬 요구),
   `tail_size % min_align`, `incorrect buffer size`(부분 충전 레인 그룹의 꼬리 패딩 누락).
   → **N4 의 "DMA 가 사이클을 지배한다"와 같은 곳이 문제다.**
   *도는 커널은 DMA 에 사이클을 쓰고, 안 도는 커널은 DMA 정렬에 막힌다.*
2. **다중 칩/클러스터 경로가 통째로 막혀 있다** — 6종이 두 가지 기전으로 막힌다.
   - **npu 컴파일 실패 5종**: `matmul_chip_reduce`·`matmul_cluster_reduce`·`cluster_shuffle`·
     `chip_slice`·`cluster_slice`
   - **컴파일 성공 1종**: `cluster_chip_shuffle_slice::chip_shuffle` 은 컴파일은 **OK** 인데
     **실기 커널 로드에서 abort**한다(N5-①의 56,576/37,888). → N9(`compile` 성공 ≠ 실기 실행 성공)의 실례.
   → **이 서버의 RNGD 4장을 vISA 로 함께 쓰는 예제는 현재 하나도 실기에서 안 돈다.**
   (단 "전부 컴파일이 안 된다"고 쓰면 틀린다.)
3. **분기(branch)가 없다.** `visa: Branch conversion is not yet implemented` — 호스트 측 `for`/`if` 로
   타일링·누적을 표현하는 큰 커널(`matmul_16384`)은 실기로 못 간다.

### N6-3. 실기로 못 가는 주요 커널 (테스트가 실제로 쓰는 28개 중)

- `matmul` **7종 전부** — `matmul_16384`(분기 미구현), `chip_reduce`/`cluster_reduce`(정렬),
  `4096`(`IndexAccess … must have axis with given tag`), `split_reduce`(`commit_trim packet mismatch`),
  `wo_broadcast`(`Collect time mismatch`), `split_reduce2`(ICE)
- `transformer::{embedding,attention,decoder,head}::forward` **4종 전부 ICE** (Qwen 24레이어 경로)
- **`vrf_add::vrf_add`** — 대표 VRF 예제가 실기에 못 올라감(`tail_size % min_align (4) != 0`)
- `contract_outer_assertions::lane_size::valid_size_{1,2,4}` — **부분 충전 레인 그룹(8레인 미만) 봉쇄.**
  완전 충전된 8레인 변형만 낮춰진다
- `switch_assertions::alignment::aligned_fetch_packet_i4` — i4 서브바이트 DRAM 크기 오산(256 vs 240).
  동일한 i8/bf16 커널은 정상
- `memory_op::{dm_pcopy,dm_view_pcopy,hbm_chip_shuffle}`, `view::{simpl,padding}`

> **28개 밖**: `view::nested::view_nested`·`tile::tile_computed_offset` 도 npu 컴파일 FAIL 이지만
> **이 둘을 부르는 테스트는 없다**(`view_tests.rs` 는 `simpl`/`padding`, `tile_tests.rs` 는
> `tile_simple`/`tile_window_commit` 만 쓴다). 63개 FAIL 목록에는 들어가고 28개에는 안 들어간다.

## ★ N7. 방법론 함정 — `compile <FILTER>` 는 부분문자열 매칭이다

```bash
cargo furiosa-opt compile -p <pkg> "switch_assertions::inter_transpose::invalid_time0"
#   Compiling [3/3] ...invalid_time0_mismatch   ← 이름이 접두사인 다른 커널까지 선택된다
#   error: ...invalid_time0_mismatch: ...       ← 에러는 그 커널 것
```

"출력에 error 가 있으면 FAIL" 로 집계하면 **멀쩡한 커널이 FAIL 로 기록된다.**

- 접두사 충돌 후보: 200개 중 **8개** / 실제 오염된 판정: **1건**(`invalid_time0` — 실제로는 컴파일 성공)
- 엄밀 재판정 규칙: 에러 줄이 `error: furiosa-opt: <정확한 커널명>:` 로 **그 커널을 지목할 때만** FAIL

## N8. 상류 `#[ignore]` 주석 3건이 낡았다 (실측 반박)

| 테스트 | 상류 사유 | 실측 |
|---|---|---|
| `matmul_tests::test_matmul_4096` | `Failing on cpu` | **에뮬레이션 통과**(2s) |
| `matmul_tests::test_matmul_with_chip_reduce` | `Failing on cpu` | **에뮬레이션 통과**(4s) |
| `vector_engine::normal::test_ve_elementwise_vrf` | `Failing on cpu` | **에뮬레이션 + 실기(NPU) 모두 통과**(0.14s) |

사유가 실제 동작과 다른 것:
- `test_matmul_16384`: `takes too much time to run` → 실제로는 **4초 만에** `commit.rs:57` 에서 실패
- `attention` 테스트: `incomplete kernel with todo!()` → 크레이트 전체에 **`todo!()` 매크로 0건**.
  실제로는 `src/attention.rs:101` 의 **주석** `// TODO: Complete the function definition.`
- `transformer::run_qwen`: `DmaCommandScatter lowering not yet implemented` → 실제 실패는 128초 뒤
  `collect.rs:125`. 벤치 경로(`kernel_sim`)로도 독립 확인 — attention/decoder 가 `switch.rs:87` 에서
  `OutTime does not match expected layout` / `OutTime must preserve 'time2'` 로 패닉

## N9. `compile` 성공 ≠ 실기 실행 성공

`reshape` 가 반례다: 커널 컴파일은 성공(137 OK 에 포함)하지만 **실기 로드에서 abort**한다(N5-①).
실기 검증은 반드시 **실제 실행**까지 해야 한다.

## N10. 예제 크레이트 규모와 "모델"의 실체

- 테스트 함수 **114개**(27개 파일), `#[device]` 커널 **207개**(42개 파일), 벤치 진입점 2개
- 3개 백엔드 전수: `typecheck` 97P/7F/10I(7개 실패는 **phantom 텐서 vacuity** = 오류 아님),
  `emulation` **104P/0F/10I**(강제 실행 3건 포함 시 실질 107), `npu` 그대로는 **빌드 실패**
- **end-to-end 학습 모델은 MNIST 하나뿐.** Qwen 2.5 0.5B(24레이어)는 차원은 실제지만
  테스트가 넣는 가중치가 전부 `zero()` → 데이터플로우 검증용. Llama 3.1 예제는 `// TODO` 로 끊긴 미완성 골격

## N11. 실기에 코드 올릴 때 지킬 것 (검증된 규칙)

1. 실행 전후로 `furiosa-smi status` 확인 — 누수 감시
2. **테스트마다 새 프로세스** — 안 그러면 N3-1 연쇄 오염에 속는다
3. **값 검증 필수** — N5-② 조용한 오배치는 크래시 없이 지나간다
4. **타임아웃 걸기**(`timeout 150`) — hang 커널이 있으면 영영 안 끝난다
5. `CARGO_TARGET_DIR` 별도 지정 — 사용자 `target/` 오염 방지
6. 비결합 연산은 순서 독립 입력으로 재검증 — N5-④

## N12. 이 세션의 재사용 도구 (`/home/jun/.claude/jobs/46bc5c7e/tmp/`)

| 경로 | 용도 |
|---|---|
| `npu_matrix.sh` | 89개 테스트 격리 실행기(상태·시간·HAL·불일치 기록) |
| `classify_mismatch.py` | 값 불일치를 ULP / 진짜 오염으로 자동 분류 |
| `sched_scan.sh` + `sched_analyze.py` | 커널 사이클 덤프 + 엔진별 분해 |
| `gate_kernels.py` | 매트릭스 FAIL 커널에 게이트 삽입(인라인 모듈 경로 추적) |
| `gate_tests.py` | 빌드 로그 읽어 실패 테스트 게이트(반복 수렴) |
| `mk_npu_subset.sh` | 서브셋 패키지 생성 |
| `visa_ex_gated/` | 게이팅된 크레이트(실기 89개 실행 가능) |
| `GROUND_TRUTH_BRIEF.md` | 위 사실들의 원본 브리프 |
