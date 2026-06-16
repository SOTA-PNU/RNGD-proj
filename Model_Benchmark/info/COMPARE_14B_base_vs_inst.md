# Qwen2.5-Coder-14B — base vs instruct 벤치 비교 (RNGD NPU, tp8)

> 같은 아키텍처(Qwen2.5-Coder-14B)의 **base**와 **Instruct** 두 모델을 Furiosa RNGD NPU에서 측정해 비교한 문서입니다.
> 측정 도구: `run_all/orchestrator.py` (furiosa-llm `serve` + `/chat/completions`).

## 한눈에 보기

| 항목 | base (`qwen2.5-coder-14b-tp8`) | instruct (`qwen2.5-coder-14b-inst-tp8`) | 판정 |
|---|---|---|---|
| 토큰당 디코드 지연 (itl p50) | 0.0325 s | 0.0326 s | **사실상 동일** |
| 단일스트림 출력 TPS | 30.8 tok/s | 30.7 tok/s | **사실상 동일** |
| 합성 프롬프트 출력 길이 | 512 토큰까지 끝까지 생성 | 평균 ~104 토큰에서 **조기 종료(EOS)** | 행동 차이 |
| 집계 throughput(고동시성) | ~1,800 tok/s | ~1,410 tok/s | **속도차 아님** — 출력 길이→배치 점유율(§2) |
| SWE-bench 패치 생성 | **0개** 생성 | **40개** 생성(단 포맷 invalid) | instruct 우위 |
| SWE-bench 해결(resolved) | 0 / 50 | 0 / 50 | 둘 다 0(14B 한계) |

**한 줄 결론:** 두 모델은 **연산량·메모리가 같아** 속도가 사실상 동일합니다(단일스트림 ITL 0.0325 s 일치 — `binary_bundle.zip` 크기도 같음, 단 md5는 다름). 실질 차이는 *행동*입니다 — **instruct는 지시를 따르고 적절히 멈추지만(대화형)**, **base는 그렇지 못합니다**(합성 프롬프트엔 끝까지 생성, chat 포맷엔 즉시 종료). 그래서 패치를 "생성하라"는 SWE-bench에서 instruct만 패치를 만들어 냅니다(품질은 14B 한계로 아직 낮음).

---

## 0. 비교 대상 & 측정 환경

| | base | instruct |
|---|---|---|
| 아티팩트 | `artifacts/qwen2.5-coder-14b-tp8` | `artifacts/qwen2.5-coder-14b-inst-tp8` |
| 결과 폴더 | `results/Qwen2.5-Coder-14B-tp8/` | `results/Qwen2.5-Coder-14B-inst-tp8/` |
| 측정일 | 2026-06-09 | 2026-06-04 ~ 05 |
| 파라미터 | qwen2 48레이어, **bf16 (W16A16KV16)**, ≈28 GB | (동일) |
| 병렬화 | **tp8 / pp1** (RNGD 1장 = npu:0) | (동일) |
| 컴파일 바이너리 | `binary_bundle.zip` = 45,967,849 bytes (md5 `68ff2f4a…`) | 45,967,849 bytes (md5 `2e4fe841…`) |

> 두 아티팩트의 `binary_bundle.zip`은 **크기는 같지만 md5는 다릅니다**(artifact_id도 상이). 같은 그래프라도 zip 타임스탬프/압축으로 바이트가 달라질 수 있어 "바이너리 동일"로 단정하진 않습니다. 다만 **연산량(레이어·shape·dtype)이 동일**하므로 하드웨어 처리 속도는 같아야 정상이고, 실제로 단일스트림 ITL(§1)이 일치해 이를 뒷받침합니다.

⚠️ **데이터 출처 주의 2가지**
1. instruct 결과 폴더 안의 JSON은 내부 `model` 필드를 `.../qwen2.5-coder-14b-tp8`(base 경로)로 기록하고 있습니다. 작업자 확인에 따라 **당시 그 경로에 instruct 가중치가 있던 측정본**으로 간주합니다.
2. instruct의 `swebench/eval_result.json`은 2026-06-09 base 벤치의 `eval_swebench.sh`(모든 폴더 일괄 재채점)가 **덮어써서** 현재 `report: null`입니다. 단 instruct의 resolved=0은 ① 2026-06-05 채점본(`completed 4 / resolved 0 / empty 7 / error 39`)과 ② 예측 요약(패치 40개 전부 invalid)으로 확정돼 비교엔 지장 없습니다.

---

## 1. 디코드 속도 — 동일 (단일 스트림, tps 태스크)

동시성 1, 입력 ~256토큰, `max_tokens=512`, 측정 30회.

| 지표 | base | instruct |
|---|---|---|
| TTFT p50 | 0.0512 s | 0.0498 s |
| TTFT p95 | 0.0548 s | 0.0513 s |
| **ITL p50 (토큰당 지연)** | **0.03248 s** | **0.03258 s** |
| ITL p95 | 0.03261 s | 0.03278 s |
| 출력 TPS/요청 p50 | 30.84 tok/s | 30.69 tok/s |
| 집계 출력 TPS | 30.77 tok/s | 30.25 tok/s |

→ **단일 스트림에서 토큰당 속도(ITL)가 동일**합니다(연산량이 같으니 예상된 결과). 단, 동시성이 커지면 운영점 차이로 ±5~8% 갈립니다(동시성 256에선 instruct가 오히려 약간 느림 — sweep ITL 0.0992 vs base 0.0917). 단일 스트림은 출력 길이가 throughput에 영향을 주지 않아 두 모델이 깔끔하게 일치합니다.

---

## 2. 처리량(throughput) — 격차는 디코드 속도가 아니라 **출력 길이가 만든 배치 점유율 차이**

### 2-1. sweep (동시성 × 입력길이, 입력 256토큰 기준 집계 출력 TPS)

| 동시성 | base | instruct |
|---:|---:|---:|
| 1 | 30.6 | 30.2 |
| 2 | 57.9 | 56.1 |
| 4 | 108.3 | 105.5 |
| 8 | 221.1 | 212.5 |
| 16 | 427.2 | 384.2 |
| 32 | 728.4 | 636.7 |
| 64 | 1,305.3 | 1,083.5 |
| 128 | 1,847.2 | 1,461.8 |
| 256 | 1,854.9 | 1,445.4 |

- 최고치: base **1,854.9** tok/s(동시성 256), instruct **1,461.8** tok/s(동시성 128).
- 동시성 1~8에서는 거의 같고, **동시성이 커질수록 base가 높게** 나옵니다.

### 2-2. 원인: base는 끝까지 생성, instruct는 조기 종료

memsweep(동시성 256, 입력 256, `max_tokens=512`, 512요청)에서 **총 출력 토큰**을 보면 분명합니다.

| combo | base 총출력토큰 | instruct 총출력토큰 | base 벽시계 | instruct 벽시계 | base ITL | instruct ITL |
|---|---:|---:|---:|---:|---:|---:|
| baseline | **262,144** | **53,466** | 146.8 s | 38.0 s | 0.092 s | 0.094 s |
| max_model_len=40960 | 262,144 | 53,466 | 144.1 s | 37.8 s | 0.090 s | 0.096 s |
| max_batch_size=256 | 262,144 | 53,466 | 146.3 s | 37.7 s | 0.092 s | 0.093 s |

- base 262,144 = 512요청 × **512토큰(꽉 채움)** — 합성 프롬프트엔 멈춤 신호가 약해 `max_tokens` 한도까지 계속 생성합니다. (반대로 chat 포맷 입력에선 즉시 멈춥니다 → §3)
- instruct 53,466 = 512요청 × **평균 ~104토큰** — 합성 프롬프트에 "응답"하고 **EOS로 멈춥니다**(대화형 모델 특성).
- **토큰당 지연(ITL)은 0.09s대로 거의 같습니다**(base 0.092 vs instruct 0.094).

집계 throughput은 `총출력토큰 ÷ 벽시계`입니다. 토큰당 디코드 속도(ITL)는 거의 같지만, **출력이 짧은 instruct는 256-요청 배치가 일찍 비워져 평균 점유율이 떨어집니다** — 즉 순수 측정 오류가 아니라 출력 길이가 만든 *실제* 시스템 throughput 차이입니다. 핵심은 **base가 토큰을 더 빨리 찍어내는 게 아니라는 점**(ITL 동일)이고, 격차의 대부분은 출력 길이에서 옵니다.

### 2-3. memsweep — serve 옵션은 둘 다 영향 거의 없음

20개 옵션 조합(`max_model_len`/`max_batch_size`/`max_num_batched_tokens`) 전부에서:

| | base 집계 TPS 범위 | instruct 집계 TPS 범위 |
|---|---|---|
| 전 조합 | 1,771 ~ 1,820 (best: `max_model_len=40960` 1,819.7) | 1,406 ~ 1,417 (best: `max_batch_size=256` 1,416.5) |

→ 두 모델 모두 옵션을 바꿔도 처리량이 거의 평평합니다(동시성 256에서 이미 포화). 모델 내부 격차의 원인 역시 §2-2(출력 길이)와 동일합니다.

> 공정한 처리량 비교가 필요하면 **동일 조건(같은 날짜·같은 입력·출력 길이 강제 고정)** 으로 재측정해야 합니다. 현재 두 데이터는 측정일(6/4 vs 6/9)도 다릅니다.

---

## 3. 코딩 품질 — SWE-bench Lite (oracle subset, 50문제)

`temperature=0`, `max_tokens=1024`. 모델이 만든 패치를 Docker로 채점.

| 지표 | base | instruct |
|---|---:|---:|
| 대상 문제 | 50 | 50 |
| **비어있지 않은 패치 생성** | **0** | **40** |
| 그중 포맷 invalid | 0 | 40 |
| 생성 오류(n_error) | 7 | 7 |
| **해결(resolved)** | **0 / 50** | **0 / 50** |

- **base는 패치를 한 개도 만들지 못했습니다.** SWE-bench 채점 파이프라인이 입력에 chat 템플릿을 씌우자, base는 거기에 **응답하도록 학습되지 않아 즉시 EOS를 내고 0토큰을 생성**했습니다(생성 시간 ≈0.2초, 50건 전부 빈 출력). chat/instruction 포맷을 다루지 못하는 base의 한계입니다.
- **instruct는 40개의 패치를 생성**했습니다(지시는 따름). 다만 전부 git diff로 적용 가능한 포맷이 아니어서(invalid) 결국 resolved 0입니다.
- 두 모델 다 **resolved 0**: SWE-bench Lite는 14B급에는 어려운 과제이고, 패치 포맷/정확도가 부족합니다. (참고: 같은 파이프라인에서 작은 모델도 0 resolved가 흔합니다 — `REPORT.md` 리더보드 참고.)

→ **품질축의 실질 결론:** instruct는 "패치를 만들라"는 지시를 수행하고 base는 못 합니다. 이 차이가 base/instruct의 본질입니다. 단 *정답률*로 둘을 가르려면 14B로는 부족하니, 정확도가 목적이면 더 큰 모델(32B/70B)이 필요합니다.

---

## 4. 종합

| 관점 | 결론 |
|---|---|
| 속도(디코드) | **동일** — 단일스트림 ITL 0.0325 s 일치(연산량 동일, 바이너리 크기 같음·md5만 상이) |
| 메모리/적재 | **동일** — bf16 ≈28 GB, RNGD 1장(tp8) |
| 처리량 절대값 | 비교 부적절 — 출력 길이가 배치 점유율을 바꾼 영향 + 측정일(6/4 vs 6/9) 차이(§2) |
| 행동 | instruct=**대화형**(지시 수행·EOS로 멈춤), base=지시 미수행(합성엔 끝까지·chat엔 즉시 종료) |
| 코딩 과제 수행 | instruct만 패치 생성. 정답률은 둘 다 0(14B 한계) |

**용도 가이드**
- **코딩 에이전트·패치 생성·대화형** → **instruct** (지시를 따르고 멈춤).
- **순수 코드 완성·FIM·이어쓰기** → **base** (멈추지 않고 생성).
- 둘의 *토큰당 속도와 메모리는 같으므로*, 선택 기준은 오직 "지시 수행이 필요한가"입니다.

---

## 부록 — 데이터 출처 / 재현

- base 측정 로그: `results/_run_logs/14b_base_full_run.log`
- 원본 결과:
  - base `results/Qwen2.5-Coder-14B-tp8/{tps,sweep,memsweep,swebench}/`
  - instruct `results/Qwen2.5-Coder-14B-inst-tp8/{tps,sweep,memsweep,swebench}/`
- 재측정 명령(둘 다 동일 조건으로 다시 보고 싶을 때):
  ```bash
  cd Model_Benchmark/rngd-npu && source ~/furiosa/bin/activate
  python -u run_all/orchestrator.py configs/models.yaml \
    --tasks tps,sweep,memsweep,swebench --models qwen2.5-coder-14b-tp8        # base
  # instruct는 configs/models.yaml에 inst 아티팩트 항목 추가 후 --models 로 지정
  ```
- 숫자 추출 스크립트(이 문서 표의 원천): 측정 JSON에서 `summary.aggregate_output_tps`, `summary.itl_s_p50`, `summary.total_output_tokens`, swebench 예측요약의 `n_nonempty_patch`/`n_invalid_patch`/`n_error`를 직접 읽어 산출.
