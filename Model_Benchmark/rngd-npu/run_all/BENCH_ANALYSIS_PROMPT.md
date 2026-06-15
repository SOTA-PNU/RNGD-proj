# RNGD Benchmark Analysis Prompt

아래 프롬프트와 함께 RNGD NPU 벤치마크 결과 파일을 제공한다.

---

첨부한 RNGD NPU 벤치마크 결과를 분석하여 Markdown 기술 보고서로 정리해 주세요.

## 분석 대상

가능한 경우 아래 파일을 모두 사용하세요.

- `configs/models.yaml`
- `artifacts/**/artifact.json`
- `results/**/tps/*.json`
- `results/**/sweep/*.json`
- `results/**/memsweep/memsweep_summary.json`
- `results/**/memsweep/memsweep_*.json`
- `results/**/swebench/*.json`
- `results/**/swebench/eval_result.json`
- `results/**/swebench/preds/*.jsonl`
- `results/_server_logs/*.log`
- `results/_server_logs/**/*.log`
- `results/_run_logs/*.log`
- `REPORT.md`

`REPORT.md`는 참고 자료로만 사용하고, 원본 JSON과 서버 로그를 우선 기준으로 삼으세요.

## 분석 원칙

- 제공된 원본 데이터에서 직접 확인하거나 계산할 수 있는 사실만 작성하세요.
- 추측, 주관적인 추천, 근거 없는 원인 분석은 제외하세요.
- 실패한 측정값은 `0 TPS`로 처리하지 말고 `측정 실패`로 표시하세요.
- 비교는 공통으로 성공한 동일 `prompt_len × concurrency` 조건의 결과끼리 수행하세요.
- 한쪽 모델만 성공한 포인트는 속도 차이가 아니라 처리 범위 차이로 분류하세요.
- 비교 대상의 조건이 다르면 표에 명시하세요.
- 데이터가 없거나 비교가 불가능하면 `판단 불가: <이유>`로 표시하세요.
- 증감률은 기준값과 계산식을 명시하세요.
- SWE-bench는 요청 수, 컨텍스트 필터링 수, 실제 추론 수, 실제 평가 수, resolved 수를 구분하세요.
- SWE-bench 분모가 다르면 resolved 개수만으로 비교하지 마세요.
- memsweep은 OFAT(One-Factor-at-a-Time) 방식이므로 여러 옵션의 결합 효과를 추정하지 마세요.
- 서로 다른 모델 계열, 정밀도, dense/MoE 구조, TP, RNGD 수의 차이를 특정 원인 하나의 효과로 단정하지 마세요.
- 동일 모델의 BF16/FP8 쌍이 없다면 양자화 자체의 성능 효과를 단정하지 마세요.
- 같은 모델과 태스크에 결과 파일이 여러 개 있으면 최신의 완결된 실행 결과를 우선 사용하세요. 설정이 다른 과거 실행 결과를 한 표에 섞지 말고, 선택한 파일과 측정 시각을 명시하세요.
- 결과 디렉터리명만으로 모델을 추정하지 마세요. `configs/models.yaml`의 `name`과 `id`, 결과 JSON 내부의 `model` 필드를 함께 사용해 매핑하세요.
- 동일 내용을 반복하지 마세요.

## 계산 기준

- sweep의 `prompt_tokens_target`은 러너가 생성한 대략적인 목표 길이입니다. 실제 토크나이즈된 입력 길이로 표현하지 말고 `목표 prompt 길이`로 표시하세요.
- `단일 TPS`는 `tps` 태스크의 concurrency `1`에서 측정된 `output_tps_per_request_p50`을 의미합니다.
- sweep 표의 `요청당 TPS`는 각 포인트의 `output_tps_per_request_p50`, `aggregate TPS`는 `aggregate_output_tps`를 의미합니다.
- `itl_s_p50`, `itl_s_p95`는 모든 token interval 원본의 직접 percentile이 아니라 요청별 median ITL 값들을 다시 요약한 지표입니다. 표에는 `runner-reported ITL p50/p95`로 표시하세요.
- 현재 TPS 러너의 `output_tokens`는 tokenizer usage가 아니라 스트리밍 `delta.content` 이벤트 수를 기반으로 집계됩니다. 따라서 TPS는 `runner-reported output TPS`로 표기하고, tokenizer 기준의 정밀한 tokens/s로 과장하지 마세요.
- sweep 포인트는 `error`가 없고 `failures == 0`일 때만 `완전 성공`으로 분류하세요.
- 일부 요청만 성공한 sweep 포인트는 `부분 실패`로 분리하고 완전 성공 포인트에 포함하지 마세요.
- sweep 성공률은 `완전 성공 포인트 수 / 시도한 전체 sweep 포인트 수 × 100`으로 계산하세요.
- 요청 실패율도 별도로 `전체 failures / (전체 successes + 전체 failures) × 100`으로 계산하세요.
- `최초 실패 조합`은 설정 파일의 `prompt_lens`, `batch_sizes` 순서대로 탐색했을 때 처음으로 완전 성공이 아닌 포인트입니다.
- memsweep 조합은 `status == "ok"`일 때만 성공으로 분류하세요. 성공 범위는 연속 범위로 추정하지 말고 실제로 성공한 테스트 값을 나열하세요.
- concurrency 증가에 따른 TPS 증가율은 인접한 두 concurrency 값 사이에서 `(다음 aggregate TPS - 이전 aggregate TPS) / 이전 aggregate TPS × 100`으로 계산하세요. 임의의 둔화 기준을 만들지 말고 증가율 표를 제공하세요.
- 서버 기동 로그 또는 결과가 없으면 서버 기동 실패로 단정하지 말고 `판단 불가`로 표시하세요.
- SWE-bench resolved 비율의 기본 분모는 harness 결과의 `report.total_instances`입니다. `resolved_instances / total_instances × 100`으로 계산하세요.
- SWE-bench는 가능한 경우 다음 값을 함께 표시하세요.
  - 요청 수: 추론 summary의 `n_requested_instances`
  - 컨텍스트 필터링 수: 추론 summary의 `n_filtered_context`
  - 실제 추론 시도 수: 추론 summary의 `n_instances`
  - 추론 오류 수: 추론 summary의 `n_error`
  - harness 제출 수: 평가 결과의 `report.submitted_instances`
  - harness 완료 수: 평가 결과의 `report.completed_instances`
  - resolved 수: 평가 결과의 `report.resolved_instances`
  - resolved 비율: `report.resolved_instances / report.total_instances × 100`

## 보고서 작성 방식

- 긴 설명보다 표를 우선 사용하세요.
- 각 실험의 첫 부분에 핵심 결과를 3~5줄로 요약하세요.
- 표 아래에는 데이터에서 확인되는 주요 관찰 사항만 짧은 bullet로 작성하세요.
- 수치는 단위를 명시하고 소수점 자릿수를 일관되게 유지하세요.
- 모델명은 축약형을 사용하되, 첫 등장 시 전체 이름과 매핑하세요.
- 실패 원인이 로그에서 확인되면 `OOM`, `timeout`, `context limit 초과`, `서버 기동 실패`, `기타`로 분류하세요.
- 표가 너무 커지면 핵심 요약표를 먼저 제공하고 상세표는 Appendix로 분리하세요.
- Executive Summary는 1페이지 분량으로 유지하세요.

## 공통 환경 정보

다음 내용을 표로 정리하세요.

- 측정 일시
- 모델명
- 모델 계열
- 파라미터 규모
- 정밀도: BF16 또는 FP8
- 구조: dense 또는 MoE
- TP
- 사용 RNGD 수
- artifact max model len
- 서버 기동 성공 여부
- 사용한 sweep 설정
- 사용한 memsweep 설정
- 데이터 누락 및 측정 실패 현황

## 실험 1. Max Model Len 차이가 실사용과 코딩 테스트에 미치는 영향

주 비교 대상:

- `Qwen3-32B-FP8-tp8`: max model len `40960`
- `Qwen3-32B-FP8-tp8-16k`: max model len `16384`

다음 질문에 각각 답하세요.

- 공통으로 성공한 `prompt_len × concurrency` 포인트에서 TTFT p50/p95, ITL p50/p95, 요청당 TPS, aggregate TPS 차이는 몇 %인가
- 각 모델이 오류 없이 처리한 최대 prompt 길이는 얼마인가
- prompt 길이별로 오류 없이 처리한 최대 concurrency는 얼마인가
- 요청 실패가 최초로 발생한 `prompt_len × concurrency` 조합은 무엇인가
- 각 모델의 전체 sweep 성공률은 얼마인가
- memsweep `--max-model-len` 변경 시 동일 요청 조건에서 TPS와 실패율은 어떻게 변하는가
- SWE-bench 요청 수, 컨텍스트 필터링 수, 실제 추론 시도 수, 추론 오류 수, harness 제출 수, harness 완료 수, resolved 수와 resolved 비율은 각각 얼마인가

공통 성공 구간의 성능 비교와 한쪽 모델만 성공한 처리 범위 비교를 분리해서 작성하세요.

## 실험 2. 파라미터 수에 따른 성능 차이

주 비교 대상:

- `Qwen2.5-Coder-1.5B-tp8`: BF16, TP 8, RNGD 1장
- `Qwen2.5-Coder-7B-tp8`: BF16, TP 8, RNGD 1장
- `Qwen2.5-Coder-14B-tp8`: BF16, TP 8, RNGD 1장

다음 질문에 각각 답하세요.

- 동일 `prompt_len × concurrency` 조건에서 TTFT, ITL, 요청당 TPS, aggregate TPS는 모델별로 얼마인가
- 파라미터 수 증가에 따른 각 지표의 증감률은 얼마인가
- prompt 길이별 peak aggregate TPS와 해당 concurrency는 얼마인가
- prompt 길이별 오류 없는 최대 concurrency는 얼마인가
- 모델별 전체 sweep 성공률은 얼마인가
- concurrency 증가에 따른 aggregate TPS 증가율은 인접 구간별로 얼마인가
- SWE-bench 실제 추론 시도 수, 추론 오류 수, harness 제출 수, harness 완료 수, resolved 수, resolved 비율은 얼마인가
- 모델 크기 증가에 따라 resolved 비율 1%p를 얻기 위해 감소한 TPS는 얼마인가

resolved 비율 1%p당 TPS 감소량은 concurrency `1`의 `output_tps_per_request_p50`을 기준으로 아래 계산식을 사용하세요.

```text
(기준 모델 TPS - 비교 모델 TPS) / (비교 모델 resolved 비율 - 기준 모델 resolved 비율)
```

분모가 `0` 이하이거나 SWE-bench 결과가 없으면 `판단 불가`로 표시하세요.

## 실험 3. 전체 모델별 성능 및 처리 범위 비교

`configs/models.yaml`에서 `enabled: true`인 모든 모델을 비교하세요.

다음 질문에 각각 답하세요.

- 각 모델의 서버 기동 성공 여부는 무엇인가
- 모델별 단일 요청 TTFT, ITL, output TPS는 얼마인가
- prompt 길이별 peak aggregate TPS와 해당 concurrency는 얼마인가
- prompt 길이별 오류 없는 최대 concurrency는 얼마인가
- 오류 없이 처리한 최대 prompt 길이는 얼마인가
- 전체 sweep 성공률은 얼마인가
- memsweep에서 오류 없이 동작한 옵션 범위는 무엇인가
- SWE-bench 요청 수, 필터링 수, 실제 추론 시도 수, 추론 오류 수, harness 제출 수, harness 완료 수, resolved 수와 resolved 비율은 얼마인가
- RNGD 수와 TP가 다른 모델은 어떤 조건으로 측정되었는가

실험 3은 활성화된 전체 모델의 관측 결과를 한눈에 비교하는 섹션입니다. 모델별 측정 조건과 결과를 정리하되, 조건이 다른 모델 간 차이의 원인을 단정하지 마세요.

## 출력 형식

다음 순서를 지키세요.

1. `Executive Summary`
   - 실험별 핵심 결과 요약표
   - 중요한 측정 실패 또는 비교 제한 사항

2. `실험 환경`
   - 모델 사양 표
   - sweep 및 memsweep 설정 표
   - SWE-bench 실행 조건

3. `데이터 누락 및 실패 현황`
   - 누락된 파일
   - 실패한 측정 포인트 수
   - 확인 가능한 실패 원인

4. `실험 1 결과`
   - 핵심 요약
   - 공통 성공 포인트 성능 비교표
   - prompt 길이별 처리 범위 표
   - memsweep max-model-len 비교표
   - SWE-bench 비교표
   - 객관적으로 확인된 사실

5. `실험 2 결과`
   - 핵심 요약
   - 동일 조건 모델별 성능 비교표
   - 파라미터 증가에 따른 증감률 표
   - prompt 길이별 peak TPS 및 최대 concurrency 표
   - SWE-bench 비교표
   - resolved 비율 1%p 증가당 TPS 감소량 표
   - 객관적으로 확인된 사실

6. `실험 3 결과`
   - 핵심 요약
   - 전체 모델 단일 요청 성능 표
   - prompt 길이별 peak aggregate TPS 표
   - 처리 범위 및 sweep 성공률 표
   - memsweep 성공 옵션 범위 표
   - SWE-bench 결과 표
   - TP 및 RNGD 수가 다른 모델의 측정 조건 표
   - 객관적으로 확인된 사실

7. `현재 결과만으로 판단할 수 없는 항목`
   - 판단할 수 없는 질문
   - 판단이 불가능한 이유
   - 추가로 필요한 데이터

8. `Appendix`
   - 필요할 때만 상세 `prompt_len × concurrency` 매트릭스
