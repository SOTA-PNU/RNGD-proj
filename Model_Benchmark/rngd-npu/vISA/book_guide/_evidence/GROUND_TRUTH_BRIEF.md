# 실측 브리프 — book_guide 최신화용 (2026-07-24 세션 확정 사실)

이 파일은 book_guide 문서들을 최신화하는 에이전트가 **그대로 인용해야 하는** 실측값이다.
모든 수치는 이 서버에서 실제로 실행한 로그에서 나왔다. 추정·전언은 없다.
원본 로그: `/home/jun/.claude/jobs/46bc5c7e/tmp/ex_logs/`

---

## A. 가장 중요한 변경 — 실기(real NPU)가 이제 열려 있다

### A-1. NPU 카드는 이 세션 내내 유휴였고, 실기 실행을 반복했다
- `furiosa-smi status` 실측: **npu0~npu3 전부 `alive`, `0.00/47.50 GiB`** (실행 전·후 모두)
- 이 세션에서 **실기 테스트 89개를 실제로 돌렸다.** 자원 누수 0.
- **따라서 "npu0이 점유 중이라 실기 불가"는 더 이상 사실이 아니다.**
  (구 문서에 `PID 1215564 / 메모리 97.86% 점유`라 적힌 것은 폐기.)
- 라우터(`furiosa_router.py`)는 떠 있지만 **백엔드 모델이 미기동**이라 NPU를 잡지 않는다.

### A-2. `--backend npu`의 진짜 제약 (CHIP0 고정보다 이게 먼저다)
> **`--backend npu`는 패키지 안의 *모든* `#[device]` 함수를 빌드 시점에 EDF로 낮춘다.**
> 테스트가 그 함수를 부르든 말든 상관없다. **하나라도 못 낮추면 크레이트 전체가 죽는다.**
- 벤더 예제 크레이트를 그대로 `--backend npu`로 돌리면 **63개 에러로 빌드 실패**, 테스트 0개 실행.
- 우회법 2가지가 검증됐다(상세는 `12-예제-전수실행.md` §7, §10):
  1. **서브셋 패키지** — 되는 모듈만 골라 새 크레이트 (실기 테스트 21개)
  2. **게이팅** (권장) — 실패 커널에만 `#[cfg(not(backend = "npu"))]`.
     상류가 이미 쓰는 관용구(`tests/matmul_tests.rs:124`). 실기 테스트 **89개**.

---

## B. 실기 테스트 매트릭스 (테스트마다 별도 프로세스로 격리 실행)

| 항목 | 값 |
|---|---|
| NPU 백엔드에 존재하는 테스트 | **89** |
| PASS | **80** |
| FAIL | 5 |
| ABORT (프로세스 즉사) | 3 |
| `#[ignore]`로 미실행 | 1 |
| **실기에서 정상 동작으로 봐야 하는 수** | **83 / 89 (93.3%)** |

`83`의 근거: 80(그대로 통과) + 2(커널은 옳은데 테스트 기준이 과함, §D-4) + 1(`#[ignore]`지만 강제 실행하면 통과).

바이너리별 실기 통과: `switch_assertions` **18/18**, `contract_outer_assertions` **13/13**,
`at_primitives` 4/4, `vector_engine` 32/36, `fetch_assertions`·`memory_op`·`param` 각 2/2,
**`mnist` 1/1**(이미지 10장 전부 정답), `binary_add`·`contract_element_types_answer`·`fetch_commit`·
`scatter_gather`·`transpose` 각 1/1, `tile` 1/2, `broadcast` 0/1, `reshape` 0/2, `shuffle_slice` 0/1.

### B-1. 격리 실행이 필수다 — 안 하면 숫자가 3배 틀린다
| 실행 방식 | vector_engine 통과 | 실패 |
|---|--:|--:|
| 한 프로세스에 전부 (평범한 `cargo test`) | 10 | 25 |
| **테스트마다 새 프로세스** | **33** | **3** |

기전: `test_ve_stash_fp_fp`가 hang → HAL이 `-110`(ETIMEDOUT) 반환 → **그 프로세스의 이후 모든 커널
실행이 전부 `-110`으로 실패**. 멀쩡한 커널 22개가 "실패"로 집계된다.
`--test-threads=1`로 직렬화해도 소용없다(동시성 문제가 아니라 프로세스 상태 오염).

---

## C. 실기 하드웨어 특성 — 사이클의 96.5%가 DMA

`cargo furiosa-opt compile --dump-schedule`로 **커널 130개**의 스케줄을 뽑아 합산했다.
(이 덤프는 AOT 컴파일러 산출물이라 **NPU를 점유하지 않는다** — 실기 테스트와 동시 실행 가능.)

| 엔진 | 총 사이클 | 비중 | 인스트럭션 |
|---|--:|--:|--:|
| **DmaEngine** | **75,464,336** | **96.5%** | 470 |
| PeCore | 2,586,167 | 3.3% | 1,557 |
| MainContext | 58,883 | 0.1% | 108 |
| InterChipTransfer | 38,018 | 0.0% | 2 |
| VectorEngine | 14,770 | 0.0% | 50 |
| SubContext | 9,737 | 0.0% | 27 |

**인스트럭션 수는 PeCore가 1,557개로 최다인데 사이클은 3.3%다.** DmaEngine은 470개로 96.5%.
→ DMA 인스트럭션 하나가 연산 인스트럭션 하나보다 두 자릿수 이상 비싸다.

### C-1. 합산의 착시가 아니다 (커널 하나하나가 그렇다)
| 커널별 DMA 사이클 비중 | 커널 수 |
|---|--:|
| 90% 이상 | **54 / 130** |
| 50% 이상 | **107 / 130 (82%)** |
| 50% 미만 | 23 / 130 |
| **중앙값** | **82.8%** |

### C-2. 사이클 스팬 분포
| min | p25 | median | p75 | max |
|--:|--:|--:|--:|--:|
| 16 | 4,612 | **10,532** | 23,503 | 10,845,036 |

1 GHz 기준 중앙값 ≈ 10.5 µs, 최대 ≈ 10.8 ms.
가장 무거운 커널: `at_primitives::vrf::multi_vrf_at` 10,845,036 cycle인데 **인스트럭션은 12개뿐**.
→ 명령 수와 비용이 전혀 비례하지 않는다.

### C-3. `mnist::forward` 실측 (독립 기록과 일치 확인됨)
**17,953 cycle / 22 instruction.** 엔진 분해: DmaEngine **12,365 (68.9%)**, MainContext 7,682,
VectorEngine 1,162, SubContext 790, PeCore 600. DRAM io 419,968 B. edf 68,106 B.
→ 이 값은 `11-MNIST-실행결과.md`에 독립적으로 기록된 수치와 **정확히 일치**한다(분석기 검증용 기준점).

### C-4. 예전의 "82,449 cycle / DmaStore 88%" 는 이제 재현·확장됐다
`OPTIMIZATION-SURFACE.md`가 "미확인 — 재현 컴파일 미실시"로 남겨둔 그 예는,
이번에 **130개 커널로 확장 재현**됐다. 결론이 강화됐다: 개별 커널이 아니라 **스택 전반의 특성**이다.

---

## D. 실기 결함 4유형 (6개 테스트가 여기 걸린다)

### D-1. ① 커널 로더 범위초과 → 프로세스 abort (3건)
```
panicked at device-runtime-c/src/kernel.rs:137:22:
range end index 50560 out of range for slice of length 33792
panicked at core/src/panicking.rs:225:5: panic in a function that cannot unwind
  10: furiosa_kernel_load
```
| 테스트 | 요구 | 실제 | 비율 |
|---|--:|--:|--:|
| `reshape_tests::test_reshape` | 50,560 | 33,792 | 1.496 |
| `reshape_tests::test_reshape_different_num_axes` | 50,560 | 33,792 | 1.496 |
| `shuffle_slice_tests::test_chip_shuffle` | 56,576 | 37,888 | 1.493 |

**세 건 모두 요구 크기가 실제의 약 1.5배** = 단일 크기계산 결함의 서명.
- 발생 지점은 **커널 로드**(`furiosa_kernel_load`). 연산 결과가 틀린 게 아니라 EDF를 올리다 죽는다.
- `cannot unwind` → **테스트 바이너리 전체 abort**. 같은 파일의 다른 테스트도 같이 죽는다.
- **세 테스트 모두 에뮬레이션에서는 통과한다.** 매핑은 유효하고 NPU 런타임 쪽 문제.
- **컴파일은 성공한다** → `compile` 성공이 실기 실행을 보장하지 않는다.

### D-2. ② 조용한 데이터 오배치 (2건, 가장 위험 — 에러 없이 틀린다)

**(a) `broadcast::test_view_broadcast` — 브로드캐스트 DMA가 목적지에 안 쓴다**
- 테스트: i32 `0..512`를 HBM→HBM 브로드캐스트로 `[A=512,B=4]`로 펼침. 기대 `[0,0,0,0,1,1,1,1,...]`
- 실기: 앞 4개가 `-1097810306, 1048624774, -1089814802, 1046945792`. **2048/2048 전부 불일치**,
  최대 절대오차 2.139e+09, **0인 원소가 하나도 없음**(기대값엔 4개가 0).
- **2회 실행에서 앞 세 값이 완전히 동일** → 난수 쓰레기가 아니라 결정적으로 특정 영역을 읽는다.
- 그 값들을 **f32 비트로 재해석**하면 `-0.283, 0.251, -0.542, 0.226` — 작은 실수들.
  → **이전에 그 HBM 영역에 쓰였던 f32 데이터의 잔류물**을 읽고 있다.
- 결론: 브로드캐스트 DMA가 **목적지에 아무것도 쓰지 않는다.** 에뮬레이션에서는 통과.

**(b) `tile_tests::test_tile_window_commit_host` — 커밋 창이 엉뚱한 곳에 쓰인다**
- 단언만 보면 `result[32]`가 24.0(기대 0.0)이라 원인 불명 → **목적지 전체를 출력하는 프로브**를 만들어 실기 실행.
- 드러난 패턴: `result[0..8]`=0(정상), **`result[8..40]`에 `input[0..32]`(=0,1,2,…,31)가 들어감**,
  `result[32..64]`는 앞 8칸만 24..31.
- **데이터는 전혀 손상되지 않았다.** 순서대로 온전하다. **쓰인 위치만 틀렸다.**
- 산술: 의도 오프셋 32 elements × 4 B(f32) = 128 B인데 실제 착지는 8 elements = **32 B**.
  → **32라는 값이 "요소 수"가 아니라 "바이트 수"로 적용됐다** (32 B ÷ 4 B = 8).
  (단, 이 해석은 산술 정황이며 런타임 소스를 읽어 확인한 것은 아니다.)

### D-3. ③ 커널 hang → HAL 타임아웃 (1건)
`vector_engine::normal::test_ve_stash_fp_fp`. **단독 실행에서도 60초 타임아웃 후 실패.**
```
furiosa_kernel_run: HAL error on ClusterId(npu0pe0-3): Unknown error -110 (os error -110)
```
`-110` = 리눅스 `ETIMEDOUT`. NPU 자체는 손상되지 않는다(직후 `alive / 0.00 GiB`). 프로세스 종료 시 회복.

### D-4. ④ 커널이 옳은데 실패로 잡힌 2건 — **버그가 아니다**

**(a) 1 ULP 반올림 — `zip::test_ve_group_pair_ternary_selective`**
512개 중 **489개(95.5%)가 비트 단위 동일**, 불일치 23개, **최대 ULP 1**, 최대 상대오차 1.742e-07.
가장 나쁜 원소조차 `npu=4.0194473` vs `host=4.019448` = f32 마지막 한 자리.
테스트가 `tests/common.rs`의 `assert_f32_vec_eq`로 **비트 일치**를 요구해서 실패한 것.

**(b) 리듀스 순서 — `reduce::test_ve_intra_slice_reduce_split_time_packet`**
처음엔 진짜 오염처럼 보였다: 512개 중 **387개(75%) 불일치**, 최대 오차 2.7e9, `2147483647`(=`i32::MAX`)까지.
그러나 이 커널은 i32 **`saturating_add` 리듀스**(R16=16개)다.
> **`saturating_add`는 결합법칙이 성립하지 않는다.** `(MAX+1)+(-1) = MAX-1` 이지만 `MAX+(1+(-1)) = MAX`.
> 즉 더하는 순서가 다르면 결과가 정당하게 달라진다.
입력이 `rand()` i32라 거의 모든 레인이 포화 → 결과가 전적으로 순서에 좌우된다.
**결정적 실험**: 같은 커널에 포화가 불가능한 작은 값(−50..50)을 넣으니 **실기에서 통과(0.12s)**.
→ 데이터 오염이 아니라 **하드웨어 병렬 리듀스 순서가 호스트 순차 기준과 다른 것.** 커널은 옳다.

> **교훈**: 비결합 연산(`saturating_add`, 부동소수 덧셈)을 검증할 때는 **순서 독립적인 입력**으로
> 한 번 더 확인해야 한다. 안 하면 "실기에서 i32 리듀스가 깨진다"는 틀린 결론을 낸다.

---

## E. NPU로 못 가는 쪽 — 커널 컴파일 매트릭스

### E-1. 집계 (엄밀 재판정 후)
**소스에서 추출한 200개 커널 중 137 OK / 63 FAIL** (68.5% 성공).

> **수치 주의 — 세 가지 커널 개수가 공존한다. 서로 "고치려" 하지 말 것.**
> - **207** = 툴 자체 집계(`Finished 1 compiled, 206 filtered out`). 제네릭 단형화 포함. 가장 정확.
> - **200** = 소스 텍스트에서 `#[device]`+`pub fn`으로 추출한 수. 매트릭스의 분모.
> - **143** = 게이팅 후 실기 빌드된 수. 207 − 게이트 64 = 143. 정합적이다.
> - 63 vs 64: 게이팅에는 보정 전 목록(64)을 썼고, 그중 1건이 거짓 FAIL(§E-3)이라 진짜 FAIL은 63.

### E-2. 실패 63개 분류 (전수 조사 + 적대적 재검증)
| 분류 | 수 | 의미 |
|---|--:|---|
| **REAL_LOWERING_GAP** | **24** | 멀쩡한 커널인데 백엔드가 아직 못 낮춘다 ← 진짜 공백 |
| INTENTIONAL_NEGATIVE | 23 | 잘못된 매핑을 문서화하려고 만든 표본. **실패가 정상** |
| **COMPILER_ICE** | **13** | 에러 문구가 `internal compiler error` — 컴파일러 자체 버그 |
| GENERIC_NOT_MONOMORPHIZED | 2 | 제네릭 device 함수라 구체 래퍼 필요 |
| UNCLEAR | 2 | 근거 부족 |

> **이전 추정을 정정한다.** `12-예제-전수실행.md` §5.1에서 `*_assertions`의 **28개**를 "의도적 표본"으로
> 묶었는데, 전수 분류 결과 **의도적인 건 23개뿐**이고 나머지 40개는 진짜 공백이거나 컴파일러 버그다.
> **이름으로 판단하면 틀린다**: `contract_outer_assertions::lane_size::valid_size_{1,2,4}`는
> 이름이 `valid_*`인데 실기 컴파일에 실패한다.

### E-3. 방법론 함정 — `compile <FILTER>`는 부분문자열 매칭이다
```bash
cargo furiosa-opt compile -p <pkg> "switch_assertions::inter_transpose::invalid_time0"
#   Compiling [3/3] ...invalid_time0_mismatch   ← 이름이 접두사인 다른 커널까지 선택된다
#   error: ...invalid_time0_mismatch: ...       ← 에러는 그 커널 것
```
"출력에 error가 있으면 FAIL"로 집계하면 **멀쩡한 커널이 FAIL로 기록된다.**
- 접두사 충돌 후보: 200개 중 **8개**
- 실제 오염된 판정: **1건** (`switch_assertions::inter_transpose::invalid_time0` — 실제로는 컴파일 성공)
- 엄밀 재판정 규칙: 에러 줄이 `error: furiosa-opt: <정확한 커널명>:` 로 **그 커널을 지목할 때만** FAIL

### E-4. 테스트가 실제로 쓰는데 실기로 못 가는 커널 28개 — 대표 사례
| 커널 | 분류 | 실기에서 막힌 지점(에러 원문 요지) |
|---|---|---|
| `matmul::matmul_16384` | REAL_GAP | `visa: Branch conversion is not yet implemented` |
| `matmul::matmul_chip_reduce` | REAL_GAP | `strides([8,128,4,...]) is not aligned by 8` (DMA 시퀀서 8B 정렬 요구) |
| `matmul::matmul_cluster_reduce` | REAL_GAP | `tail_size % min_align (1) != 0` |
| `matmul::matmul_4096` | REAL_GAP | `IndexAccess ... must have axis with given tag` (slice_tile 에필로그) |
| `matmul::matmul_with_split_reduce` | REAL_GAP | `mir: commit_trim packet mismatch. Expected A % 4 # 8 ..., got A % 8` |
| `matmul::matmul_wo_broadcast` | REAL_GAP | `mir: Collect time mismatch. Expected: A / 4 % 4, got: A / 4` |
| `transformer::{embedding,attention,decoder,head}::forward` | **ICE** ×4 | Qwen 24레이어 전 단계가 컴파일러 내부 오류 |
| `vrf_add::vrf_add` | REAL_GAP | `tail_size % min_align (4) != 0` — **대표 VRF 예제가 실기에 못 올라감** |
| `contract_outer_assertions::lane_size::valid_size_{1,2,4}` | REAL_GAP ×3 | `incorrect buffer size`: 부분 충전 레인 그룹의 꼬리 패딩을 DRAM 크기 계산이 누락 (256 vs 228/232/240) |
| `switch_assertions::alignment::aligned_fetch_packet_i4` | REAL_GAP | i4(4비트) 서브바이트 DRAM 크기 오산 (256 vs 240) |
| `memory_op::{dm_pcopy,dm_view_pcopy,hbm_chip_shuffle}` | REAL_GAP ×3 | |
| `cluster_chip_shuffle_slice::{cluster_shuffle,chip_slice,cluster_slice}` | REAL_GAP ×3 | |
| `view::padding::view_padding` | REAL_GAP | `visa: tile index with non-empty packing is not supported` |
| `tile::tile_computed_offset` | REAL_GAP | 루프변수로 계산한 오프셋이 상수로 폴딩되지 않음 |
| `attention::compile_llama3_1_...` | REAL_GAP | (미완성 커널) |
| `view::simpl::view_simpl`, `matmul::matmul_with_split_reduce2` | ICE | |

### E-5. 읽히는 패턴 3가지 (중요)
1. **정렬(alignment)이 반복해서 발목을 잡는다.** `not aligned by 8`, `tail_size % min_align`,
   `incorrect buffer size`(꼬리 패딩 누락) — DMA 시퀀서의 8바이트 정렬 요구와 부분 충전 그룹의
   꼬리 패딩 계산이 여러 커널을 동시에 막는다. **§C의 "DMA가 사이클을 지배한다"와 같은 곳이 문제다.**
2. **다중 칩/클러스터 경로가 통째로 막혀 있다.** `matmul_chip_reduce`·`matmul_cluster_reduce`·
   `cluster_shuffle`·`chip_slice`·`cluster_slice`·`chip_shuffle`이 전부 실패.
   → **이 서버의 RNGD 4장을 vISA로 함께 쓰는 예제는 현재 하나도 실기에서 안 돈다.**
3. **분기(branch)가 없다.** `Branch conversion is not yet implemented` — 호스트 측 `for`/`if`로
   타일링·누적을 표현하는 큰 커널은 실기로 못 간다.

---

## F. 상류 `#[ignore]` 주석 3건이 낡았다 (실측 반박)

| 테스트 | 상류가 적어둔 사유 | 실측 |
|---|---|---|
| `matmul_tests::test_matmul_4096` | `Failing on cpu` | **에뮬레이션 통과** (2s) |
| `matmul_tests::test_matmul_with_chip_reduce` | `Failing on cpu` | **에뮬레이션 통과** (4s) |
| `vector_engine::normal::test_ve_elementwise_vrf` | `Failing on cpu` | **에뮬레이션 통과 + 실기(NPU)에서도 통과 (0.14s)** |

추가로 사유가 실제 동작과 다른 것:
- `test_matmul_16384`: `takes too much time to run` → 실제로는 **4초 만에** `commit.rs:57`에서 실패
- `attention` 테스트: `incomplete kernel with todo!()` → 크레이트 전체에 **`todo!()` 매크로는 0건**
  (`grep -rn "todo!\|unimplemented!" src` = 0). 실제로는 `src/attention.rs:101`에 **주석** `// TODO: Complete the function definition.`
- `transformer::run_qwen`: `DmaCommandScatter lowering not yet implemented` → 실제 실패는
  128초 뒤 `collect.rs:125`. 벤치 경로(`kernel_sim`)로도 독립 확인: attention/decoder가
  `switch.rs:87`에서 `OutTime does not match expected layout` / `OutTime must preserve 'time2'`로 패닉.

---

## G. 3개 백엔드 전수 실행 결과 (참고 — 12번 문서에 상술됨)

| 백엔드 | 결과 | 비고 |
|---|---|---|
| `typecheck` | 97 passed / 7 failed / 10 ignored | 7개 실패는 전부 **phantom 텐서 vacuity**(`left: []`) = 오류 아님. 성공 신호는 "빌드 성공" |
| `emulation` | **104 passed / 0 failed / 10 ignored** | 완전 통과. ignored 중 3개는 강제 실행하면 통과(§F) → 실질 107 |
| `npu` (그대로) | **빌드 실패, 테스트 0개 실행** | 63개 커널 에러 (§A-2) |
| `npu` (게이팅) | 143 커널 컴파일, 배치 58 passed / 격리 80 passed | §B |

크레이트 규모: 테스트 함수 **114개**(27개 파일), `#[device]` 커널 **207개**(42개 파일), 벤치 진입점 2개.
저장소에서 **end-to-end 학습 모델은 MNIST 하나뿐**. Qwen 2.5 0.5B(24레이어)는 가중치가 전부 `zero()`이고,
Llama 3.1 예제는 함수가 `// TODO`로 끊긴 미완성 골격.

---

## H. 실기에 코드 올릴 때 반드시 지킬 것 (검증된 규칙)

1. **실행 전후로 `furiosa-smi status` 확인** — 누수 감시
2. **테스트마다 새 프로세스** — 안 그러면 §B-1의 연쇄 오염에 속는다
3. **값 검증을 반드시 붙일 것** — §D-2의 조용한 오배치는 크래시 없이 지나간다
4. **타임아웃을 걸 것** — hang 커널이 있으면 영영 안 끝난다 (`timeout 150`)
5. **`CARGO_TARGET_DIR`를 따로 지정** — 사용자 `target/` 오염 방지
6. **비결합 연산은 순서 독립 입력으로 재검증** — §D-4(b)

---

## I. 이 세션에서 만든 재사용 도구

| 경로 | 용도 |
|---|---|
| `tmp/npu_matrix.sh` | 89개 테스트 격리 실행기 (상태·시간·HAL·불일치 기록) |
| `tmp/classify_mismatch.py` | 값 불일치를 ULP / 진짜 오염으로 자동 분류 |
| `tmp/sched_scan.sh` + `sched_analyze.py` | 커널 사이클 덤프 + 엔진별 분해 |
| `tmp/gate_kernels.py` | 매트릭스 FAIL 커널에 게이트 삽입 (인라인 모듈 경로 추적) |
| `tmp/gate_tests.py` | 빌드 로그 읽어 실패 테스트 게이트 (반복 수렴) |
| `tmp/mk_npu_subset.sh` | 서브셋 패키지 생성 |
| `tmp/visa_ex_gated/` | 게이팅된 크레이트 (실기 89개 실행 가능) |

---

## J. 변하지 않은 사실 (기존 문서가 맞게 적은 것 — 건드리지 말 것)
- 래퍼는 **0.4 세대**(`--version`은 "0.3.0" 출력, md5가 v0.4.0 타르볼과 동일)
- 백엔드는 `[typecheck, emulation, npu]`, **기본 `emulation`**. `simulation`은 **없음**(0.3 시절 이름)
- 커널 패키지 마커는 **`[package.metadata.furiosa-opt]`** (구세대 `src/furiosa-opt.tag`는 안 통함)
- 산출물 명명 `pkg::mod::fn.{bin,edf,hash}`
- `/home/jun/yik`은 0.3 핀 + tag → **현재 래퍼로 `--backend npu` 불가**. typecheck/emulation은 동작
- `gemm_kernel.rs:8`의 "16×16 output tile" 주석은 틀렸다(격자가 16×16, 타일은 32×32)
- TCL은 클러스터 경계에서 멈추고 슬라이스 내부(Lane/Packet/TRF/VRF)는 vISA만 노출
- 하드웨어: chip당 512 spatial cells, Lane ≤ 8, Packet ≤ 64 B, cluster당 256 slice, chip당 2 cluster,
  DM 총 256 MB(slice당 512 KB), TRF 8 KB/lane, VRF 8 KB/slice, HBM 1.5 TB/s, DMA 256 GB/s
