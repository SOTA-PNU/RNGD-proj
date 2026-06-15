# `run_all.sh` 가 실행하는 파이썬 파일들

`run_all.sh` 는 벤치마크 전체 파이프라인을 단계별로 묶은 wrapper. **모든 py 파일은 `rngd-npu/run_all/` 폴더에 모여있고**, sh 가 거기를 호출합니다.

```bash
bash run_all.sh                   # 전체 (preflight → smoke → gen → embed → swebench → report)
STAGE=gen bash run_all.sh         # 단계별 실행
CONFIG=configs/models.yaml bash run_all.sh
```

| STAGE | 실행 |
|---|---|
| `preflight` | (bash) `preflight.sh` |
| `smoke` | `run_all/orchestrator.py --tasks tps --models Qwen2.5-0.5B` |
| `gen` | `run_all/orchestrator.py --tasks tps,sweep,memsweep` |
| `embed` | `run_all/orchestrator.py --tasks embed,rerank` |
| `swebench` | `run_all/orchestrator.py --tasks swebench` → `eval_swebench.sh` → `run_all/swebench_eval.py` |
| `report` | `run_all/analyze.py` → `run_all/report.py` |
| `all` | 위 전부 |

---

## 폴더 구조

```
rngd-npu/
├── run_all.sh                 단계별 wrapper
├── eval_swebench.sh           SWE-bench Docker 채점 wrapper
├── preflight.sh               NPU·SDK·docker·HF 캐시 점검
├── setup.sh                   측정용 의존성 설치
├── configs/models.yaml        모델 목록 + 태스크 인자
├── run_all/
│   ├── orchestrator.py        모델 × 태스크 매트릭스 메인
│   ├── analyze.py             결과 JSON → CSV 집계
│   ├── report.py              결과 JSON → REPORT.md
│   ├── swebench_eval.py       SWE-bench Docker 채점
│   └── runners/
│       ├── server.py          furiosa-llm serve up/down
│       ├── tps.py             tps / sweep
│       ├── memory_sweep.py    memsweep
│       ├── embed_bench.py     embed / rerank
│       └── swebench_run.py    swebench inference + eval
└── results/                   결과 (자동 생성)
```

각 py 는 `REPO_ROOT = Path(__file__).resolve().parent.parent` 로 `rngd-npu/` 를 가리켜 `configs/`·`results/` 를 한 단계 위에서 찾습니다.

---

## 1. `orchestrator.py` — 메인 오케스트레이터

모델 × 평가축 매트릭스. 각 모델마다 NPU 서버를 한 번 띄우고 여러 태스크를 실행 후 내림.

| 태스크 | 측정 | runner |
|---|---|---|
| `tps` | concurrency=1, stream → TTFT / ITL / output TPS | `runners/tps.py` |
| `sweep` | concurrency × prompt_len 매트릭스 | `runners/tps.py` |
| `memsweep` | 서버 인자(`--max-model-len`, `--max-batch-size`, `--max-num-batched-tokens`) OFAT 스윕 | `runners/memory_sweep.py` |
| `embed` | 임베딩 throughput (batch size 별) | `runners/embed_bench.py` |
| `rerank` | 리랭커 throughput | `runners/embed_bench.py` |
| `swebench` | SWE-bench Lite oracle 추론 | `runners/swebench_run.py` |

**결과**: `results/<모델>/<태스크>/<timestamp>.json`

---

## 1b. 실행 방식 — 순차 실행과 카드 사용

오케스트레이터는 선택된 모델을 한 번에 하나씩 순차로 돕니다(orchestrator.py:270-278의
`for m in selected` 루프). 모델마다 서버를 올리고(`with server:`) 태스크를 끝낸 뒤 내린 다음
다음 모델로 넘어갑니다.

디바이스 기본값은 `npu:0` 한 장입니다(configs/models.yaml:146, orchestrator.py:275). tp=8로
빌드한 1장 모델은 모두 이 기본값을 그대로 써서 npu:0만 사용합니다. 4장을 명시하는 모델은
tp=32 prebuilt 3개(Qwen3-32B·EXAONE-4.0-32B·Llama-3.3-70B)뿐이고, 이들은 모델별 serve_args에
`--devices npu:0,npu:1,npu:2,npu:3`을 넣습니다(configs/models.yaml:103·111·119).

따라서 1장 모델을 측정하는 동안 나머지 세 장(npu1·2·3)은 놀게 됩니다. 카드가 4장이므로
1장 모델 4개를 카드별로 동시에 올려 시간을 줄이는 것은 하드웨어상 가능합니다. `FuriosaServer`가
devices·host·port를 인스턴스마다 받기 때문에(server.py:21-59), 포트(공통 인자가 8000 고정,
configs/models.yaml:141)와 `--devices`만 인스턴스마다 다르게 주면 됩니다.

다만 태스크에 따라 병렬화가 적절한지 갈립니다.

| 태스크 | 측정 대상 | 4장 병렬 | 이유 |
|---|---|---|---|
| tps / sweep / memsweep | TTFT·ITL·처리량 | 권장 안 함 | 지연·처리량을 클라이언트(호스트)에서 측정합니다. 한 호스트에서 여러 서버와 부하생성기를 동시에 돌리면 CPU·PCIe 경합으로 NPU가 아니라 호스트 포화를 재게 됩니다. sweep는 동시 요청을 256까지 올려 특히 민감합니다 |
| swebench 추론 | 정답 패치율(temperature=0) | 안전·이득 | 점수가 호스트 경합과 무관하고, 느려져도 결과가 같아 시간만 줄어듭니다 |

tp=32 모델 3개는 한 모델이 카드 4장을 모두 쓰므로 병렬 대상이 아니며, 1장 모델과도 겹치면
안 됩니다.

이 원칙대로, swebench **전용 단계**(`--tasks swebench`)는 단일 카드 모델을 카드별로 동시에
돌립니다(orchestrator.py `_run_swebench_parallel`). 카드 풀(기본 npu:0~3)을 자원 큐로 두고
비는 카드에 모델을 하나씩 배정하며, 카드마다 포트를 8000부터 분리합니다. 단일 카드 모델이
풀보다 많으면 카드가 빌 때마다 다음 모델이 물려받습니다. tp=32 모델은 단일 카드 단계가 끝난
뒤 단독 순차로 돕니다. tps·sweep·memsweep은 지연·처리량 측정이라 여전히 순차입니다.

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `SWEBENCH_PARALLEL` | `1` | swebench 전용 단계의 카드별 병렬. `0`이면 기존 순차로 폴백 |
| `SWEBENCH_PARALLEL_DEVICES` | `npu:0,npu:1,npu:2,npu:3` | 단일 카드 모델에 배정할 카드 풀 |

병렬은 swebench 전용 단계에서만 켜집니다. `--tasks`에 tps·sweep 등이 섞이면 그 실행은
기존대로 순차입니다. 또한 카드와 포트(8000~)를 쓰므로, 다른 측정이 npu:0이나 포트 8000을
점유하는 동안에는 같이 돌리면 안 됩니다.

---

## 2. `swebench_eval.py` — SWE-bench 채점

```bash
python run_all/swebench_eval.py                   # 전체
python run_all/swebench_eval.py --models Llama    # 필터
python run_all/swebench_eval.py --max-workers 12
```

추론과 채점은 NPU/Docker 의존성이 달라서 분리.

### 알려진 운영 이슈 — Docker 소켓 권한 거부

이번 실행에서는 enabled 9개 모델 모두 추론은 됐지만 채점이 전건 실패했습니다.
원인은 채점 컨테이너가 Docker 데몬 소켓에 붙지 못한 권한 거부(PermissionError 13)입니다.
그 결과 각 `results/<모델>/swebench/eval_result.json`이 `returncode: 1`, `report: null`로
남고, resolved 점수는 채점되지 못했습니다(예: `results/Qwen3-32B-FP8-tp8/swebench/eval_result.json`).

해결은 실행 사용자가 docker 소켓에 접근할 수 있게 하는 것입니다. 사용자를 docker 그룹에
추가하거나(`sudo usermod -aG docker $USER` 후 재로그인), rootless docker로 돌리면 됩니다.

---

## 3. `analyze.py` / `report.py`

```bash
python run_all/analyze.py --csv out.csv     # JSON 들 → CSV
python run_all/report.py                    # JSON 들 → REPORT.md
```

`report.py` 상단의 임계값:
- `SLA_TTFT_P95_S = 10.0`
- `EFFICIENT_FRAC = 0.90`
- `SWEEP_PROMPT_LEN = 1024`

### `report.py` 알려진 한계

REPORT.md를 만드는 `report.py`에 아래 세 가지 표기 버그가 있습니다(이번 실행에서 확인).
REPORT.md는 자동 생성물이라 직접 고치지 말고 `report.py`를 고쳐야 합니다.

- 표1 NPU 열·표7 'NPU 카드' 카드 수 오표기는 **2026-06-05 수정됨.** 이전에는 `model_meta`를
  yaml `id`로만 키잉해서(report.py), 결과·표가 `name`으로 식별되는 tp32·로컬 아티팩트의
  `--devices`를 못 찾아 전부 npu:0(1장)으로 폴백했습니다. 이제 id·name 둘 다로 키잉하고
  카드 수를 'N장'으로 표기합니다(report.py:186-194). tp32 3개(Qwen3-32B·EXAONE-4.0-32B·
  Llama-3.3-70B)는 4장, 나머지는 1장으로 정확히 나옵니다.
- 8절 '측정 제외 모델(하드웨어 제약, tp32)' 목록을 `enabled: false`만 보고 만들어서
  (report.py:413), tp32도 아니고 결과도 있는 furiosa-ai/Qwen2.5-0.5B·Llama-3.1-8B를
  잘못 포함합니다.
- 표6 Embedding/Reranker는 batch=4도 측정하지만, row dict가 `{1, 16, 64}`만 가져서
  (report.py:335) batch=4 컬럼이 표에서 누락됩니다.

---

## 데이터 흐름

```
configs/models.yaml
      │
      ▼
 run_all/orchestrator.py
   ├─ runners/server.py       (serve up/down)
   ├─ runners/tps.py          (tps, sweep)
   ├─ runners/memory_sweep.py (memsweep)
   ├─ runners/embed_bench.py  (embed, rerank)
   └─ runners/swebench_run.py (inference)
      │
      ▼
 results/<model>/<task>/<ts>.json
      │
      ├──▶ run_all/swebench_eval.py  (Docker 채점)
      ├──▶ run_all/analyze.py        → summary.csv
      └──▶ run_all/report.py         → REPORT.md
```
