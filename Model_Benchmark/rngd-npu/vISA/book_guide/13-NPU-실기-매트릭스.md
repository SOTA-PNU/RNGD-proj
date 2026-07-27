# 13 · NPU 실기 완전 매트릭스 — 진짜 하드웨어에서 뭐가 도나

> **이 문서는 실기(real RNGD hardware)만 다룬다.** typecheck·emulation 이야기는 [12-예제-전수실행](./12-예제-전수실행.md)에 있다.
> 여기 있는 건 전부 `--backend npu`로 **실제 NPU에서 돌린** 결과다.
> 모든 테스트를 **개별 프로세스로 격리 실행**했다 — 이유는 §3에 있다. 그게 없으면 숫자가 거짓말을 한다.

---

## 0. 한 줄 결론

**NPU 백엔드에 존재하는 89개 테스트 중 80개가 그대로 실기에서 통과한다(89.9%).**
여기에 **커널은 옳은데 테스트 기준이 과한 2건**과 `#[ignore]`지만 강제 실행하면 통과하는 1건을 더하면
**83 / 89 (93.3%)** 가 실기에서 정상 동작한다(§1.1).
실패 6개는 딱 **4가지 결함 유형**으로 압축되고, 그중 **3가지는 하드웨어/런타임 쪽 진짜 결함**이다.

| | |
|---|---|
| 실기 통과 (그대로) | **80 / 89** (89.9%) |
| 실기 정상 동작 (§1.1 보정) | **83 / 89** (93.3%) |
| 실기 실패 | 6 (§2 ①②③) |
| `#[ignore]`로 안 돎 | 1 |
| 실기 바이너리로 컴파일되는 커널 | **137 / 200** (68.5%) |
| **실기 사이클의 지배 요인** | **DMA — 커널 130개 중 107개에서 50% 이상, 중앙값 82.8%** (컴파일러 스케줄 모델) |

> **가장 중요한 한 줄**: 실기에서 이 커널들은 **연산이 아니라 데이터 이동에 묶여 있다.**
> 이건 vISA로 무엇을 최적화해야 하는지를 바꾸는 사실이다(§5).

---

## 1. 실기에서 도는 것 — 89개 전수 매트릭스

89개를 **하나씩 새 프로세스로** 실기 실행한 결과다.

| 테스트 바이너리 | 실기 통과 | 비고 |
|---|:--:|---|
| `switch_assertions_tests` | **18 / 18** | — |
| `contract_outer_assertions_tests` | **13 / 13** | — |
| `at_primitives_tests` | **4 / 4** | — |
| `fetch_assertions_tests` · `memory_op_tests` · `param_tests` | **2 / 2** each | — |
| **`mnist_tests`** | **1 / 1** | 학습 가중치로 이미지 10장 전부 정답 |
| `binary_add_tests` · `contract_element_types_answer_tests` | **1 / 1** each | — |
| `fetch_commit_tests` · `scatter_gather_tests` · `transpose_tests` | **1 / 1** each | — |
| `vector_engine_tests` | 32 / 36 | 실패 4건 중 **2건은 커널이 옳음**(§2.4) |
| `tile_tests` | 1 / 2 | FAIL `test_tile_window_commit_host` (§2.2) |
| `broadcast` | 0 / 1 | FAIL `test_view_broadcast` (§2.2) |
| `reshape_tests` | 0 / 2 | ABORT ×2 (§2.1) |
| `shuffle_slice_tests` | 0 / 1 | ABORT `test_chip_shuffle` (§2.1) |
| **합계** | **80 / 89** | |

### 1.1 "80"은 과소평가다 — 실제로는 83이 정상 동작

원시 집계 80에는 **커널이 옳은데도 실패로 잡힌 것**이 섞여 있다.

| 항목 | 수 | 설명 |
|---|--:|---|
| 그대로 통과 | 80 | |
| 커널은 옳으나 테스트가 너무 엄격 | **+2** | 1 ULP 반올림 1건, 리듀스 순서 1건 (§2.4) |
| `#[ignore]`라 안 돌았지만 **강제 실행하면 통과** | **+1** | `test_ve_elementwise_vrf` — 사유 `"Failing on cpu"`가 낡음 |
| **실기에서 정상 동작** | **83 / 89 (93.3%)** | |
| **진짜 결함에 걸린 것** | **6** | §2 의 ①②③ (4개 기전, §1.2) |

> `test_ve_elementwise_vrf`는 `#[ignore = "Failing on cpu"]`가 붙어 있는데,
> **실기에서 강제 실행하니 0.14초에 통과**했다(에뮬레이션에서도 통과함은 12번 문서 §6.1에서 이미 확인).
> 상류 주석이 낡았다는 세 번째 사례다.

### 1.2 실기 결함은 6개 테스트 / **4개 기전**

번호는 §2 의 결함 유형에 맞춘다. §2 의 ②(조용한 데이터 오배치)는 **기전이 둘로 갈리므로** (a)/(b) 로 나눴다.

| 기전 | 걸린 테스트 | 건수 | §2 유형 |
|---|---|--:|:--:|
| 커널 로더 범위초과 → 프로세스 abort | `reshape` ×2, `chip_shuffle` | 3 | ① |
| 브로드캐스트 DMA 가 목적지에 미기록 | `test_view_broadcast` | 1 | **②**(a) |
| 커밋 창 오프셋 단위 오류(요소 vs 바이트) | `test_tile_window_commit_host` | 1 | **②**(b) |
| 커널 hang → HAL `-110` 타임아웃 | `test_ve_stash_fp_fp` | 1 | ③ |
| | | **합 6** | |

§2 의 **④(반올림·순서 차이, 2건)는 여기 없다** — 커널이 옳으므로 결함으로 세지 않는다(§2.4).

---

## 2. 실기 실패 6건 — 4가지 결함 유형

실패를 "6개 실패"라고만 적으면 쓸모가 없다. 성격이 전부 다르다.

| 유형 | 건수 | 정체 | 위험도 |
|---|--:|---|---|
| ① **로더 범위초과 → 프로세스 abort** | 3 | `device-runtime-c/src/kernel.rs:137` | 높음 (즉사) |
| ② **조용한 데이터 오배치** | 2 | 에러 없이 틀린 위치/값 | **가장 높음** (안 들킴) |
| ③ **커널 hang → HAL 타임아웃** | 1 | `os error -110` (ETIMEDOUT) | 높음 (연쇄 오염) |
| ④ ~~반올림·순서 차이~~ | (2) | 커널은 옳음, 테스트 기준이 과함 | **버그 아님** |

> **④를 ②와 섞지 않는 것이 이 문서의 핵심 작업이다.** 값 불일치를 자동 분류하는
> `classify_mismatch.py`를 만들어 ULP·순서 아티팩트와 진짜 오배치를 갈랐다.
> 안 그랬으면 "실기 실패 5건"이라 적었을 텐데, 실제 결함은 3건이다.

### 2.1 ① 커널 로더 범위초과 — 3건

```
thread '<unnamed>' panicked at device-runtime-c/src/kernel.rs:137:22:
range end index 50560 out of range for slice of length 33792
thread '<unnamed>' panicked at core/src/panicking.rs:225:5:
panic in a function that cannot unwind
  10: furiosa_kernel_load
```

| 테스트 | 요구 크기 | 실제 크기 | 비율 |
|---|--:|--:|--:|
| `reshape_tests::test_reshape` | 50,560 | 33,792 | 1.496 |
| `reshape_tests::test_reshape_different_num_axes` | 50,560 | 33,792 | 1.496 |
| `shuffle_slice_tests::test_chip_shuffle` | 56,576 | 37,888 | 1.493 |

**세 건 모두 요구 크기가 실제의 약 1.5배**다. 우연이 아니라 **단일 크기계산 결함**의 서명이다.

- 발생 지점: **커널 로드 단계**(`furiosa_kernel_load`). 연산 결과가 틀린 게 아니라 EDF를 올리다 죽는다.
- 파급: `panic in a function that cannot unwind` → **테스트 프로세스 전체가 abort.**
  같은 바이너리의 다른 테스트도 같이 죽는다.
- **대조**: 세 테스트 모두 **에뮬레이션에서는 통과한다.** 매핑은 유효하고 **NPU 런타임 쪽 문제**다.
- 컴파일은 성공한다 — 즉 `cargo furiosa-opt compile` 성공이 실기 실행을 보장하지 않는다.

### 2.2 ② 조용한 데이터 오염 — 2건 (가장 위험)

에러도, HAL 오류도, 크래시도 없다. 그냥 **틀린 값을 정상인 척 돌려준다.**

#### `broadcast::test_view_broadcast` — 전 원소 쓰레기

테스트가 하는 일은 단순하다. i32 `0..512`를 HBM에 올리고, HBM→HBM 브로드캐스트 DMA로
`[A=512, B=4]`로 펼친 뒤 되읽는다. 기대값은 `[0,0,0,0, 1,1,1,1, 2,2,2,2, ...]`.

```rust
let input = HostTensor::<i32, m![A]>::from_vec((0..512).collect::<Vec<_>>());
let hbm1 = input.to_hbm::<m![1], m![A]>(&mut ctx.pdma).await;
let hbm2 = hbm1.to_hbm::<{ Dma::Tensor }, m![A, B]>(&mut ctx.tdma);   // ← 브로드캐스트
let output = hbm2.to_host::<m![A, B]>(&mut ctx.pdma).await;
```

실기 결과:

| | 값 |
|---|---|
| 기대 (앞 8개) | `0, 0, 0, 0, 1, 1, 1, 1` |
| **실기 (앞 4개)** | `-1097810306, 1048624774, -1089814802, 1046945792` |
| 불일치 | **2048 / 2048 (전부)** |
| 최대 절대오차 | 2.139e+09 |
| 0인 원소 | **0개** (기대값엔 4개가 0) |

**정체 규명:**
- **결정적이다.** 2회 반복 실행에서 앞 세 값이 `-1097810306, 1048624774, -1089814802`로 **완전히 동일**.
  → 난수 쓰레기가 아니라 특정 영역을 일관되게 읽고 있다.
- 그 값들을 **f32 비트로 재해석**하면 `-0.283, 0.251, -0.542, 0.226` — 작은 실수들이다.
  → **이전에 그 HBM 영역에 쓰였던 f32 데이터의 잔류물**을 읽고 있다.
- 결론: **브로드캐스트 DMA가 목적지에 아무것도 쓰지 않는다.** 되읽기는 묵은 값을 가져온다.
- **에뮬레이션에서는 통과한다.** 조용한 정합성 붕괴 — 실기와 에뮬의 결과가 갈리는데 아무도 안 알려준다.

#### `tile_tests::test_tile_window_commit_host` — 커밋 창이 엉뚱한 곳에 쓰인다

첫 증상은 이것뿐이라 원인을 알 수 없다:

```
assertion `left == right` failed: result[32] should equal input[0]
  left: 24.0    right: 0.0
```

그래서 **단언 대신 목적지 전체를 출력하는 프로브**를 만들어 실기에서 돌렸다.
그러자 오염 패턴이 정확히 드러났다.

| 위치 | 기대 | **실기 실측** |
|---|---|---|
| `result[0..8]` | 0 (미기록) | 0 ✅ |
| `result[8..40]` | 0 (미기록) | **`input[0..32]` = 0,1,2,…,31** ← 여기 쓰였다 |
| `result[32..64]` | `input[0..32]` | 앞 8칸만 24..31, 나머지 0 |

**데이터는 전혀 손상되지 않았다.** `0,1,2,…,31`이 순서대로 온전하다.
**쓰인 위치만 틀렸다** — 목적지 오프셋이 **32가 아니라 8**이었다.

산술이 정확히 맞는다:

```
의도한 오프셋 : 32 elements × 4 B(f32) = 128 B
실제 착지점   :  8 elements           =  32 B
→ 32라는 값이 "요소 수"가 아니라 "바이트 수"로 적용됐다.  32 B ÷ 4 B = 8 elements
```

즉 **커밋 창 목적지 오프셋의 단위 혼동(요소 vs 바이트)** 이다. 우연의 일치로 보기엔
비율(32→8 = ÷4 = f32 크기)이 너무 정확하다.

- **에뮬레이션에서는 통과한다.** 실기에서만 갈린다.
- 크래시도 경고도 없다. 정답 검증이 없었으면 **그냥 지나갔다.**

> **이 유형이 왜 최악인가**: ①·③은 시끄럽게 죽어서 바로 안다. ②는 **숫자가 조용히 틀린다.**
> MNIST처럼 정답을 검증하는 테스트가 없으면 **틀린 채로 넘어간다.**
> 실기로 무언가를 옮길 때 **반드시 값 검증을 붙여야 하는 이유**가 이것이다.

### 2.3 ③ 커널 hang — `os error -110`

`vector_engine::normal::test_ve_stash_fp_fp` 하나. **단독 실행에서도 60초 타임아웃 후 실패**한다.

```
furiosa_kernel_run: HAL error on ClusterId(npu0pe0-3): Unknown error -110 (os error -110)
```

`-110`은 리눅스 **`ETIMEDOUT`**. 커널 실행이 하드웨어에서 끝나지 않는다.

**연쇄 오염이 진짜 문제다** — §3 참조.
NPU 자체는 손상되지 않는다: 직후 `furiosa-smi status`에서 npu0~npu3 모두 `alive / 0.00 GiB`.
프로세스가 끝나면 회복된다.

### 2.4 ④ 커널이 옳은데 실패로 잡힌 2건

값이 다르다고 다 버그가 아니다. 이 둘은 **커널이 맞고 테스트 기준이 과한** 경우다.

#### (a) 1 ULP 반올림 — `zip::test_ve_group_pair_ternary_selective`

| 원소 수 | 비트 일치 | 불일치 | 최대 ULP | 최대 상대오차 |
|--:|--:|--:|--:|--:|
| 512 | **489 (95.5%)** | 23 | **1** | 1.742e-07 |

가장 나쁜 원소조차 `npu=4.0194473` vs `host=4.019448` — **f32의 마지막 한 자리**다.
테스트가 `assert_f32_vec_eq`로 **비트 일치**를 요구한다:

```rust
// tests/common.rs — NaN만 봐주고 나머지는 완전 일치를 요구
fn eq(&self, other: &Self) -> bool {
    self.0 == other.0 || (self.0.is_nan() && other.0.is_nan())
}
```

#### (b) 리듀스 순서 — `reduce::test_ve_intra_slice_reduce_split_time_packet`

이건 처음엔 **진짜 오염처럼 보였다**: 512개 중 **387개(75%)가 불일치**, 최대 오차 2.7e9,
`2147483647`(= `i32::MAX`)까지 나온다. ULP로 설명할 수 없는 규모다.

그런데 이 커널이 하는 연산은 i32 **`saturating_add` 리듀스**(R16=16개)다.

> **`saturating_add`는 결합법칙이 성립하지 않는다.**
> `(MAX + 1) + (-1) = MAX - 1` 이지만 `MAX + (1 + (-1)) = MAX`.
> 즉 **더하는 순서가 다르면 결과가 정당하게 달라진다.**

입력이 `rand()`로 만든 i32(크기 ~2³¹)라 16개를 더하면 **거의 모든 레인이 포화**한다.
그러면 결과는 전적으로 순서에 좌우된다 — 75% 불일치는 순서 민감성이 예측하는 바로 그 값이다.

**결정적 실험**으로 확정했다. 같은 커널에 **포화가 불가능한 작은 값**(−50..50)을 넣었다:

```rust
// PROBE: 포화가 절대 일어나지 않는 입력. 실패가 순서 때문이라면 이건 반드시 통과한다.
let input = HostTensor::<i32, m![R, A]>::from_vec(
    (0..n).map(|i| ((i % 101) as i32) - 50).collect::<Vec<_>>());
```

```
test ..._split_time_packet_smallvals ... ok        (실기, 0.12s)
```

**통과했다.** → 실패는 데이터 오염이 아니라 **하드웨어의 병렬 리듀스 순서가 호스트의 순차 기준과
다르기 때문**이다. 커널은 옳다.

> 이 실험이 없었으면 "실기에서 i32 리듀스가 깨진다"는 **틀린 결론**을 문서에 적었을 것이다.
> 비결합 연산(`saturating_add`, 부동소수 덧셈)을 검증할 때는 **순서 독립적인 입력**으로
> 한 번 더 확인해야 한다.

---

## 3. 왜 "개별 프로세스 격리"가 필수인가 — 숫자가 3배 달라진다

이 문서의 매트릭스는 **테스트 하나당 프로세스 하나**로 돌렸다. 그냥 한 번에 돌리면 **숫자가 거짓말을 한다.**

| 실행 방식 | vector_engine 통과 | 실패 |
|---|--:|--:|
| 한 프로세스에 전부 (평범한 `cargo test`) | 10 | 25 |
| **테스트마다 새 프로세스** | **33** | **3** |

**같은 하드웨어, 같은 커널인데 통과 수가 10 → 33으로 3.3배 달라진다.**

### 3.1 메커니즘

1. `test_ve_stash_fp_fp`가 NPU에서 hang → HAL이 `-110`(ETIMEDOUT) 반환
2. **그 프로세스의 이후 모든 커널 실행이 전부 `-110`으로 실패**
3. 멀쩡한 커널 22개가 "실패"로 집계됨

`--test-threads=1`로 직렬화해도 **소용없다**(동시성 문제가 아니라 프로세스 상태 오염이므로).
확인: 단일 스레드 실행에서도 앞 10개만 통과하고 11번째부터 무너졌다.

### 3.2 결정적 증거

| 커널 | 단독 실행 | 판정 |
|---|---|---|
| `normal::test_ve_stash_fp_fp` | **60초 타임아웃 실패** | 🔴 진짜 hang |
| `reduce::test_ve_intra_slice_reduce_add_f32` | **0.13초 통과** | 🟢 연쇄 피해자 |

> ### 실무 규칙
> **실기 테스트 결과를 믿으려면 프로세스를 격리하라.**
> hang 커널 하나가 뒤따르는 모든 커널을 오염시키므로, 한 번에 돌린 "N개 실패"는
> **상한이 아니라 과대계상**이다. 진짜 실패를 알려면 하나씩 새 프로세스로 돌려야 한다.

---

## 4. 실기 하드웨어 특성 — 사이클은 어디로 가나

`cargo furiosa-opt compile --dump-schedule`은 컴파일러가 예측한 **인스트럭션별 사이클 수명과 엔진**을
JSON으로 내놓는다. **NPU 장치를 점유하지 않으므로** 실기 테스트와 동시에 돌릴 수 있다.
커널 **130개**의 스케줄을 뽑았다 — 실기 컴파일에 성공하는 137개 전부는 아니다.

> **검증**: 이 분석기가 `mnist::forward`에 대해 내놓은 값은 **17,953 cycle / 22 instruction**,
> 엔진 분해는 `DmaEngine 12,365 / MainContext 7,682 / SubContext 790`.
> [11-MNIST-실행결과](./11-MNIST-실행결과.md)에 독립적으로 기록된 수치와 **정확히 일치**한다.

### 4.1 사이클은 거의 전부 DMA로 간다

**커널 130개 전체 합산:**

| 엔진 | 총 사이클 | 비중 | 인스트럭션 |
|---|--:|--:|--:|
| **DmaEngine** | **75,464,336** | **96.5%** | 470 |
| PeCore | 2,586,167 | 3.3% | 1,557 |
| MainContext | 58,883 | 0.1% | 108 |
| InterChipTransfer | 38,018 | 0.0% | 2 |
| VectorEngine | 14,770 | 0.0% | 50 |
| SubContext | 9,737 | 0.0% | 27 |

인스트럭션 **수**는 PeCore가 1,557개로 제일 많은데 **사이클**은 3.3%다.
반대로 DmaEngine은 470개 인스트럭션으로 **96.5%를 잡아먹는다.**
→ **DMA 인스트럭션 하나가 연산 인스트럭션 하나보다 두 자릿수 이상 비싸다.**

### 4.2 커널 하나하나가 그렇다 (합산의 착시가 아니다)

거대 커널 몇 개가 평균을 끌어올린 게 아니라는 확인:

| 커널별 DMA 사이클 비중 | 커널 수 |
|---|--:|
| 90% 이상 | **54 / 130** |
| 50% 이상 | **107 / 130 (82%)** |
| 50% 미만 | 23 / 130 |
| **중앙값** | **82.8%** |

**중앙 커널조차 사이클의 83%를 데이터 이동에 쓴다.**

### 4.3 사이클 스팬 분포

| min | p25 | median | p75 | max |
|--:|--:|--:|--:|--:|
| 16 | 4,612 | **10,532** | 23,503 | 10,845,036 |

1 GHz 기준 중앙값 ≈ **10.5 µs**, 최대 ≈ **10.8 ms**.

가장 무거운 커널들:

| 사이클 | 인스트럭션 | 커널 |
|--:|--:|---|
| 10,845,036 | 12 | `at_primitives::vrf::multi_vrf_at` |
| 10,845,036 | 12 | `vector_engine::normal::ve_elementwise_multi_vrf` |
| 10,842,104 | 10 | `vector_engine::normal::ve_elementwise_vrf` |
| 6,946,088 | 16 | `cluster_chip_shuffle_slice::chip_shuffle` |
| 6,840,200 | 258 | `tile::tile_simple` |

**인스트럭션 12개짜리가 1,080만 사이클을 쓴다.** 명령 수와 비용이 전혀 비례하지 않는다 —
DMA 한 방이 압도적으로 비싸다는 같은 이야기다.

참고로 `mnist::forward`는 17,953 cycle / 22 instruction으로 중앙값 근처의 평범한 커널이고,
그 안에서도 DmaEngine이 12,365 cycle(68.9%)을 차지한다.

---

## 5. 그래서 실기에서 무엇을 최적화해야 하나

§4의 결론은 명확하다: **DMA가 사이클을 지배한다.**

지난 세션에서 vISA **"연산 최적화"** 주제가 세 번 무너진 이유가 여기서 정량적으로 설명된다.

- vISA가 **배타적으로** 노출하는 건 슬라이스 내부 데이터패스(Lane/Packet/TRF/DPE/Vector)다.
- 그런데 실기 사이클의 대부분은 **슬라이스 바깥의 데이터 이동**에서 나온다.
- 슬라이스 내부를 아무리 잘 짜도 전체에서 차지하는 몫이 작으면 **Amdahl 상한**에 갇힌다.

이건 "vISA가 쓸모없다"가 아니라 **"어디를 봐야 하는지"** 를 알려준다.

| 봐야 할 곳 | 근거 |
|---|---|
| **HBM↔DM 전송량·정렬** | 사이클의 96.5%가 DmaEngine, 중앙 커널도 82.8% |
| **레이아웃(내부 런 길이)** | 지난 세션 실측: 64B→256B 정렬 변경으로 DmaStore 72,731→1,552 cycle (**46.9배**) |
| 슬라이스 내부 연산 | 중요하지만 **상한이 3.3%** (PeCore 총 점유) |

> 정렬 실험의 상세는 [OPTIMIZATION-SURFACE](./OPTIMIZATION-SURFACE.md) 참조.

### 5.1 §7이 같은 곳을 가리킨다

실기로 **컴파일조차 안 되는** 커널들의 실패 사유도 정렬이다 —
`not aligned by 8`, `tail_size % min_align`, `incorrect buffer size`(꼬리 패딩 누락).

> **도는 커널은 DMA에 사이클을 쓰고, 안 도는 커널은 DMA 정렬에 막힌다.**
> 실기에서 중요한 건 슬라이스 안이 아니라 **데이터가 어떻게 놓이고 움직이는가**다.

---

## 6. 실기로 내 코드를 올리는 법 (검증된 레시피)

### 6.1 전제 조건 — 이걸 모르면 무조건 막힌다

> **`--backend npu`는 패키지 안의 *모든* `#[device]` 함수를 빌드 시점에 EDF로 낮춘다.**
> 테스트가 그 함수를 부르든 말든 상관없다. **하나라도 못 낮추면 크레이트 전체가 죽는다.**

벤더 예제 크레이트를 그대로 `--backend npu`로 돌리면 **63개 에러로 빌드가 죽는다**. 테스트는 하나도 못 돈다.

### 6.2 두 가지 해법

| | 서브셋 패키지 | **게이팅** (권장) |
|---|---|---|
| 방법 | 되는 모듈만 골라 새 크레이트 | 안 되는 커널에만 `#[cfg(not(backend="npu"))]` |
| 자르는 단위 | 모듈 | **커널** |
| 실기 테스트 수 | 21 | **89** |
| 원본과의 차이 | 구조가 다름 | **게이트 삽입만** |
| 상류 관용구인가 | 아니오 | **예** (`tests/matmul_tests.rs:124`가 이미 사용) |

### 6.3 게이팅 절차

```bash
# 1) 커널별로 실기 컴파일 가능 여부를 전수 조사 (NPU 불필요, 병렬 가능)
cat kernels.txt | xargs -P 12 -I{} bash -c '
  out=$(cargo furiosa-opt compile -p <pkg> "{}" 2>&1)
  echo "$(echo "$out" | grep -q "^error" && echo FAIL || echo OK)|{}"'

# 2) FAIL 커널에 게이트 삽입 (인라인 모듈 경로까지 추적해야 동명 함수를 구분한다)
python3 gate_kernels.py

# 3) 빌드 → 실패 테스트에 게이트 → 반복 (에러 0까지 보통 2~3회)
cargo furiosa-opt --backend npu test -p <pkg> --release --no-run
python3 gate_tests.py <build.log>

# 4) 실기 실행 — 반드시 테스트마다 새 프로세스로 (§3)
while read bin test; do
  cargo furiosa-opt --backend npu test -p <pkg> --release --test "$bin" -- --exact "$test"
done < all_tests.txt
```

### 6.4 실기 실행 시 반드시 지킬 것

1. **실행 전후로 `furiosa-smi status` 확인** — 누수 감시
2. **테스트마다 새 프로세스** — 안 그러면 §3의 연쇄 오염에 속는다
3. **값 검증을 반드시 붙일 것** — §2.2의 조용한 오염은 크래시 없이 지나간다
4. **타임아웃을 걸 것** — hang 커널이 있으면 영영 안 끝난다 (`timeout 150`)

---

## 7. NPU로 못 가는 쪽 — 컴파일 실패 63개

실기에서 도는 것만 보면 반쪽이다. **커널 200개 중 63개는 실기 바이너리로 아예 컴파일되지 않는다.**
이 63개를 전수 분류했다(그룹별 담당 + 적대적 재검증).

### 7.1 분류 결과 — "일부러 실패하는 것"은 절반도 안 된다

| 분류 | 수 | 의미 |
|---|--:|---|
| **REAL_LOWERING_GAP** | **24** | 멀쩡한 커널인데 백엔드가 아직 못 낮춘다 ← **진짜 공백** |
| INTENTIONAL_NEGATIVE | 23 | 잘못된 매핑을 문서화하려고 만든 표본. **실패가 정상** |
| **COMPILER_ICE** | **13** | 에러 문구가 `internal compiler error` — **컴파일러 자체 버그** |
| GENERIC_NOT_MONOMORPHIZED | 2 | 제네릭 device 함수라 구체 래퍼가 필요 |
| UNCLEAR | 2 | 근거 부족 |

표 합이 **64**로 63과 안 맞는다. 분류는 **보정 전 64개 목록**(§7.4) 기준이고, 그중 1건이
거짓 FAIL이어서 진짜 FAIL은 63이다. 재분류는 하지 않았다.

> **이전 추정을 정정한다.** [12번 문서 §5.1](./12-예제-전수실행.md)에서 나는 실패의 상당수가
> `contract_outer_assertions`·`switch_assertions`의 의도적 표본이라고 적었다.
> 전수 분류해 보니 **의도적인 건 23개뿐**이고, 나머지 40개는 진짜 공백이거나 컴파일러 버그다.
> 특히 `contract_outer_assertions::lane_size::valid_size_{1,2,4}`처럼 **이름이 `valid_*`인데 실패**하는
> 것들이 있어, 이름만 보고 판단하면 틀린다.

### 7.2 테스트가 실제로 쓰는데 실기로 못 가는 커널 — 28개

의도적 표본은 아무도 안 부르지만, **28개는 실제 테스트가 호출한다.** 즉 진짜로 아쉬운 손실이다.

대표적인 것:

| 커널 | 분류 | 실기에서 막힌 지점 |
|---|---|---|
| `matmul::matmul_16384` | REAL_GAP | `visa: Branch conversion is not yet implemented` — 타일링 루프의 분기를 NPU 제어흐름으로 못 바꾼다 |
| `matmul::matmul_chip_reduce` | REAL_GAP | `strides([8,128,4,...]) is not aligned by 8` — DMA 시퀀서가 8B 정렬을 요구 |
| `matmul::matmul_cluster_reduce` | REAL_GAP | `tail_size % min_align (1) != 0` — i8 4개 꼬리가 8B flit 정렬에 안 맞음 |
| `transformer::{embedding,attention,decoder,head}::forward` | **ICE** ×4 | Qwen 24레이어 전 단계가 컴파일러 내부 오류 |
| `vrf_add::vrf_add` | REAL_GAP | `tail_size % min_align (4) != 0` — 대표 VRF 예제가 실기에 못 올라감 |
| `contract_outer_assertions::lane_size::valid_size_{1,2,4}` | REAL_GAP ×3 | `incorrect buffer size`: 부분 충전된 레인 그룹의 꼬리 패딩을 DRAM 크기 계산이 누락 |
| `switch_assertions::alignment::aligned_fetch_packet_i4` | REAL_GAP | i4(4비트) 텐서의 서브바이트 DRAM 크기 오산 (256 vs 240) |
| `memory_op::{dm_pcopy,dm_view_pcopy,hbm_chip_shuffle}` | REAL_GAP ×3 | |
| `cluster_chip_shuffle_slice::{cluster_shuffle,chip_slice,cluster_slice}` | REAL_GAP ×3 | 다중 클러스터/칩 재배치 |

### 7.3 읽히는 패턴 세 가지

1. **정렬(alignment)이 반복해서 발목을 잡는다.** `not aligned by 8`, `tail_size % min_align`,
   `incorrect buffer size` — DMA 시퀀서의 8바이트 정렬 요구와 부분 충전 그룹의 꼬리 패딩 계산이
   여러 커널을 동시에 막는다. §4의 "DMA가 사이클을 지배한다"와 **같은 곳**이 문제다.
2. **다중 칩/클러스터 경로가 통째로 막혀 있다.** 단 기전이 두 가지다.
   `matmul_chip_reduce`·`matmul_cluster_reduce`·`cluster_shuffle`·`chip_slice`·`cluster_slice`
   **5종은 실기 컴파일 자체가 실패**하고, `cluster_chip_shuffle_slice::chip_shuffle`
   **1종은 컴파일은 통과하지만 실기 커널 로드에서 abort**한다(§2.1). 후자는 `compile` 성공이
   실기 실행을 보장하지 않는다는 교과서적 실례다.
   → **이 서버의 RNGD 4장을 vISA로 함께 쓰는 예제는 지금 하나도 실기에서 안 돈다.**
3. **분기(branch)가 없다.** `Branch conversion is not yet implemented` —
   호스트 측 `for`/`if`로 타일링·누적을 표현하는 큰 커널은 실기로 못 간다.

### 7.4 방법론 주의 — `compile <FILTER>`는 부분문자열 매칭이다

매트릭스를 만들 때 실수할 뻔한 지점이다.

```bash
cargo furiosa-opt compile -p <pkg> "switch_assertions::inter_transpose::invalid_time0"
#   Compiling [3/3] ...invalid_time0_mismatch      ← 이름이 접두사인 다른 커널까지 함께 선택된다
#   error: ...invalid_time0_mismatch: ...          ← 에러는 그 커널 것
```

필터가 **부분문자열**이라 `invalid_time0`을 지정하면 `invalid_time0_mismatch`·`invalid_time0_size`도
같이 컴파일된다. "출력에 error가 있으면 FAIL"로 집계하면 **멀쩡한 커널이 FAIL로 기록된다.**

- 접두사 충돌 후보: 200개 중 **8개**
- 실제로 오염된 판정: **1건** (`invalid_time0` — 실제로는 컴파일 성공)
- 재판정 방법: 에러 줄이 `error: furiosa-opt: <정확한 커널명>:` 로 **그 커널을 지목할 때만** FAIL
- **보정 후: 137 OK / 63 FAIL** (보정 전 136/64)

> 게이팅(§6)에는 보정 전 목록을 썼으므로 `invalid_time0` 하나가 불필요하게 제외됐다.
> 실기 결과에는 영향이 없다(그 커널을 부르는 테스트가 없다).

---

## 8. 정직한 한계

- **성능 비교를 하지 않았다.** 사이클은 **컴파일러의 스케줄 모델**이지 실측 벽시계가 아니다.
  실기 벽시계는 cargo 오버헤드(최소 3.5초)에 묻혀 커널 시간을 분리하지 못했다.
- **§4의 사이클은 정적 예측이다.** 실제 하드웨어 카운터를 읽은 게 아니다.
  다만 `mnist::forward`가 독립 기록과 정확히 일치하므로 모델 자체는 신뢰할 만하다.
- **게이팅한 64개 커널을 고쳐보지 않았다.** 실기로 못 간다는 것까지만 확인했다.
- **`test_ve_stash_fp_fp`의 hang 원인을 소스 수준에서 규명하지 않았다.** 어느 매핑이
  하드웨어를 멈추는지는 후속 과제다.
- **②의 두 결함이 같은 뿌리인지 확인하지 않았다.** 둘 다 "잘못된 위치"지만
  broadcast는 DMA 목적지 미기록, tile은 오프셋 단위 혼동으로 증상이 다르다.
- **tile의 "바이트 vs 요소" 해석은 산술 정황이지 소스 확인이 아니다.** 32→8이 정확히 f32 크기(4)로
  나눠떨어지는 건 강한 근거지만, 런타임 소스를 읽어 확인한 것은 아니다.
- 배치=1, 단일 칩. **다중 칩/클러스터 경로는 6종 중 5종이 실기 컴파일에서 막히고, 유일하게
  컴파일되는 `chip_shuffle`은 실기 커널 로드에서 abort한다**(§7.3, §2.1) —
  이 서버의 RNGD 4장을 vISA로 함께 쓰는 예제는 현재 하나도 실기에서 돌지 않는다.
- **§7의 분류는 에이전트 조사 + 적대적 재검증 결과다.** 8개 그룹 중 7개에서 검증자가
  표현·범위 오류를 지적해 반영했지만, 개별 커널 사유를 전수 수작업 재확인하지는 않았다.
- **상류에 보고하지 않았다.** 보고 가치가 있는 실측(순서는 우선순위가 아니다):
  - 커널 로더 범위초과 (§2.1)
  - 브로드캐스트 DMA 목적지 미기록 (§2.2)
  - 커밋 창 오프셋 단위 혼동 (§2.2)
  - 커널 hang → HAL `-110` (§2.3)
  - 낡은 `#[ignore]` 사유 3건
  - 컴파일러 ICE 13건 (§7.1)

---

## 9. 재현

```bash
. "$HOME/.cargo/env"
BASE=/home/jun/.claude/jobs/46bc5c7e/tmp

# 게이팅 크레이트 (12번 문서 §10에서 만든 것)
cd $BASE/visa_ex_gated

# NPU 백엔드에 존재하는 테스트 열거
for b in <각 테스트 바이너리>; do
  cargo furiosa-opt --backend npu test -p furiosa-opt-examples --release --test "$b" -- --list \
    | grep ': test$' | sed "s/: test$//" | sed "s|^|$b\t|"
done > npu_all_tests.txt        # → 89개

# 전수 격리 실행 (핵심: 하나당 프로세스 하나 + 타임아웃)
$BASE/npu_matrix.sh             # → ex_logs/npu_matrix.tsv

# 실패 유형 자동 분류 (ULP vs 진짜 오염)
python3 $BASE/classify_mismatch.py

# 사이클 특성 (NPU 불필요, 동시 실행 가능)
$BASE/sched_scan.sh
python3 $BASE/sched_analyze.py
```

### 산출물

| 경로 | 내용 |
|---|---|
| `tmp/npu_matrix.sh` | 89개 격리 실행기 (상태·시간·HAL·불일치 기록) |
| `tmp/classify_mismatch.py` | 값 불일치를 ULP / 진짜 오염으로 자동 분류 |
| `tmp/sched_scan.sh` · `sched_analyze.py` | 사이클 덤프 + 엔진별 분해 |
| `tmp/ex_logs/npu_matrix.tsv` | **실기 매트릭스 원본 (89행)** |
| `tmp/ex_logs/npu_matrix_detail/` | 테스트별 실기 출력 전문 |
| `tmp/ex_logs/sched/*.json` | 커널별 스케줄 덤프 |
| `tmp/ex_logs/sched_summary.json` | 사이클 요약 |

---

### 관련 문서
- [12-예제-전수실행](./12-예제-전수실행.md) — 조사 전 과정(3개 백엔드), 게이팅을 만든 경위
- [11-MNIST-실행결과](./11-MNIST-실행결과.md) — MNIST 실기 판정
- [OPTIMIZATION-SURFACE](./OPTIMIZATION-SURFACE.md) — 어디를 고쳐야 하나
- [_GROUND_TRUTH](./_GROUND_TRUTH.md) — 이 서버의 버전 현실
