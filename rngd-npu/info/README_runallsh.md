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

## 2. `swebench_eval.py` — SWE-bench 채점

```bash
python run_all/swebench_eval.py                   # 전체
python run_all/swebench_eval.py --models Llama    # 필터
python run_all/swebench_eval.py --max-workers 12
```

추론과 채점은 NPU/Docker 의존성이 달라서 분리.

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
