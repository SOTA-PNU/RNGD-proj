# furiosa-llm serve · build — 전체 콜그래프 분석 (gdb · pdb · eBPF)

분석 대상 두 가지:

```bash
furiosa-llm serve /home/jun/chacha/qwen2.5-coder-7b-inst-tp8 --port 12345   # (A) serve
furiosa-llm build <Qwen2.5-Coder-1.5B-Instruct> <out> -tp 4 --max-model-len 2048   # (B) build
```

두 명령을 각각 **실제로 실행**하고 **gdb**(네이티브), **Python 디버거 계층**(serve=viztracer/py-spy, build=py-spy — pdb와 동일한 인터프리터 트레이싱 메커니즘), **eBPF**(bpftrace로 `furiosa_rngd` 커널 드라이버 후킹)로 계측한 뒤, 내부 코드 API 정적 분석과 결합하여 콜그래프를 재구성한다. **serve** 는 로드/추론을 Python ↔ 네이티브(Rust) ↔ 커널 ↔ 하드웨어 4계층으로, **build** 는 AOT 컴파일을 드라이버 ↔ Ray 워커 ↔ 네이티브 컴파일러 3계층으로(빌드는 NPU 미접촉) 그렸다.

## ⭐ 먼저 볼 것

**serve**
1. **[`03-synthesis/CALLGRAPH-WALKTHROUGH.md`](03-synthesis/CALLGRAPH-WALKTHROUGH.md)** — serve 통합 워크스루
2. **[`03-synthesis/G1-startup-load-fullstack.svg`](03-synthesis/G1-startup-load-fullstack.svg)** — 기동/로드 풀스택
3. **[`03-synthesis/G2-inference-fullstack.svg`](03-synthesis/G2-inference-fullstack.svg)** — 추론 요청 풀스택

**build**
1. **[`03-synthesis/BUILD-CALLGRAPH-WALKTHROUGH.md`](03-synthesis/BUILD-CALLGRAPH-WALKTHROUGH.md)** — build 통합 워크스루(메인 문서)
2. **[`03-synthesis/GB1-build-fullstack.svg`](03-synthesis/GB1-build-fullstack.svg)** — 빌드 풀스택 콜그래프(드라이버→Ray 워커→네이티브)
3. **[`01-static/build-static-callgraph.md`](01-static/build-static-callgraph.md)** — build 정적 file:line 레퍼런스
4. **[`03-synthesis/full-callgraphs/gdb_build.native_names.md`](03-synthesis/full-callgraphs/gdb_build.native_names.md)** — 컴파일러 `??`(native_llm_common.so) 간이 명명

## TL;DR (실측)

- 토폴로지: **tp=8, pp=1, dp=4** — 4× RNGD 카드 전부, 카드당 8 PE(`pe0-3`+`pe4-7` 융합), DP 4 복제.
- 로드: 아티팩트(schema 3.0) → `NativeLLMEngine` 생성 → 가중치 14.2GiB mmap(0.99s) → KV 31.3GiB/카드(585,555 blocks) → XGrammar/LLGuidance → Eager 스케줄러. ~18s에 ready.
- 추론: HTTP → uvloop → FastAPI → `OpenAIServingChat` → (네이티브 토크나이저) → `NativeLLMEngine.stream_generate` → Rust 스케줄러/제너레이터 → 커널 `doorbell_sq_write`(PE 커맨드 링 제출) → NPU 연산 → `npu_doorbell_handler` IRQ + PDMA → 토큰 SSE.
- 네이티브 함수단위 그래프 불가(`native_runtime.so` 스트립) → RUST_LOG 모듈 흐름 + gdb 스레드 분류 + eBPF 커널 카운트로 재구성.

## TL;DR — build (실측)

- 대상: Qwen2.5-Coder-1.5B-Instruct, **tp=4**, preset 버킷, 워커 기본 1.
- 경계 **둘**: ① 드라이버→Ray 워커(프로세스), ② Python→네이티브(PyO3). 무거운 일은 Ray 액터에서 — 드라이버는 `build_pipeline → ray.get`(`new_pipeline_builder.py:1586`)에서 파킹.
- 단계: 트레이싱(`ray::LocalPipelineGenerationActor`, dynamo+make_fx+샤딩, 49 태스크) → 컴파일(`ray::TaskCompileActor`, `compile()` @ `converter.py:913` → `native_llm_common.so`, 78 태스크).
- **NPU 미접촉**: device open 0, DMA 0. 커널 트래픽은 상주 furiosa-smi(`tokio-runtime-w`) 노이즈뿐.
- 결과: 이 (모델,tp)는 컴파일 stage_0 에서 `failed to lower the operator O1089 (no tactic)` 로 실패(계측 무관). 그 17분간의 gdb 스냅샷이 컴파일러를 한창 도는 상태로 잡음 → `native_llm_common.so`(143MB, 스트립) `??` 프레임을 region(컴파일 패스)·콜래더로 **간이 명명**.

## 디렉터리 구성

```
callgraph-analysis/
├── README.md                         이 파일
├── 01-static/                        정적 분석 (코드 API, 멀티에이전트 추출)
│   ├── static-callgraph.md           [serve] A 기동/로드 · B 추론 · C 빌드 + 네이티브 경계표 (file:line)
│   ├── build-static-callgraph.md     [build] 드라이버·Ray워커·네이티브 컴파일러 (file:line, SDK 2026.2.0 검증)
│   ├── startup-load.dot/.svg         정적 기동/로드 그래프
│   ├── inference.dot/.svg            정적 추론 경로 그래프
│   └── build.dot/.svg                정적 빌드(AOT 컴파일) 경로 그래프
├── 02-dynamic/                       동적 캡처 (라이브 실행)
│   ├── scripts/                      재현 스크립트
│   │   ├── kernel_trace.bt           bpftrace eBPF 커널 트레이스 (furiosa_rngd)
│   │   ├── run_A.sh                  라이브 serve + bpftrace + gdb + (perf/pyspy) + 추론
│   │   ├── run_B.sh                  viztracer 결정론적 Python 콜그래프 + 로드 커널 트레이스
│   │   ├── run_C.sh / run_D.sh       추가 perf/py-spy 캡처
│   │   ├── load_and_infer.py         [serve] viztracer 대상(로드+추론 드라이버)
│   │   ├── parse_viz.py              viztracer JSON → 콜그래프 edge/dot
│   │   ├── extract_gdb.py            gdb bt → 폴디드 스택 아키타입
│   │   ├── build_run_A.sh            [build] bpftrace + build + gdb(드라이버·워커) + py-spy(워커)
│   │   ├── build_run_B.sh            [build] viztracer 드라이버 트레이스(선택)
│   │   ├── build_driver.py           [build] in-process build 드라이버(viztracer 대상)
│   │   └── name_native_build.py      [build] native_llm_common.so ?? → 간이 명명
│   └── logs/                         캡처 산출물 (아래 "증거 색인" 참조)
└── 03-synthesis/                     합성 (정적+동적 결합, 메인 산출물)
    ├── CALLGRAPH-WALKTHROUGH.md      [serve] 통합 워크스루
    ├── BUILD-CALLGRAPH-WALKTHROUGH.md [build] 통합 워크스루
    ├── G1-startup-load-fullstack.dot/.svg/.png
    ├── G2-inference-fullstack.dot/.svg/.png
    ├── GB1-build-fullstack.dot/.svg/.png   [build] 풀스택(드라이버→Ray워커→네이티브)
    └── full-callgraphs/             ⭐ 전수 텍스트 콜그래프 (모든 함수, 필터 없음)
        ├── gdb_{load_10s,idle,infer}.{calltree,archetypes,adjacency}.txt   [serve] 네이티브
        ├── viz_full_{calltree,adjacency,reverse}.txt                       [serve] Python(pdb 계층)
        ├── gdb_build_{driver,trace_worker,compile_1}.{calltree,archetypes,adjacency}.txt  [build] 네이티브
        └── gdb_build.native_names.md · gdb_build.calltree.named.txt        [build] 컴파일러 ?? 간이 명명
```

## 주요 증거 파일 (`02-dynamic/logs/`)

| 파일 | 내용 |
|---|---|
| `serve.log`, `serve_C.log`, `serve_D.log` | 네이티브 RUST_LOG (tp8/dp4, 디바이스 매핑, 스케줄러, KV/weights) |
| `kernel_trace.log` / `_B` / `_C` | eBPF 커널 이벤트 (open/ioctl/mgmt_admin/DMA/doorbell, 카운트 맵) |
| `gdb_idle.txt`, `gdb_infer.txt`, `gdb_load_10s.txt`, `*_archetypes.txt` | gdb 네이티브 스레드 백트레이스 + 분류 |
| `viz_load_infer_edges.tsv`, `viz_callgraph_summary.txt`, `viz_load_infer.dot/.svg` | viztracer 결정론적 Python 콜그래프 (원시 1.5GB JSON은 용량상 삭제, `run_B.sh`로 재생성) |
| `perf_infer.log`, `pyspy_infer.log` | perf/py-spy(--native) 시도 기록 — 이 호스트에서 유효 샘플 미수집(워크스루 §7); 네이티브는 gdb로 대체 |
| `npu_fds.txt` | serve가 연 `/dev/rngd/*` 노드 (tp8 실증) |
| `chat_response*.json`, `completion_stream.txt`, `long_gen*.json` | 추론 정상 동작 증거 |
| `furiosa_rngd_symbols.txt`, `furiosa_rngd_tracepoints.txt` | 드라이버 심볼/트레이스포인트 목록 |

## 재현

- **serve**: `02-dynamic/scripts/`의 `run_A.sh`(메인) → `run_B.sh`(viztracer) → `run_D.sh`(perf) 순.
- **build**: `BUILD_MODEL=<HF경로> bash 02-dynamic/scripts/build_run_A.sh` → `python 02-dynamic/scripts/name_native_build.py`.

bpftrace/perf/gdb 부착에 root 필요(`sudo`). 상세는 각 워크스루 §6. build 동적 산출물은 `02-dynamic/logs/`의 `build.log`·`pyspy_build_{driver,trace,compile}.txt`·`gdb_build_*.txt`·`kernel_trace_build.log`·`build_fds.txt`.

## 도구 / 환경
furiosa-llm (venv `~/furiosa`, Python 3.12) · gdb 12 · bpftrace · perf 6.17 · py-spy 0.4.2 · viztracer 1.1.1 · graphviz 2.43. 커널 `furiosa_rngd`, 4× RNGD (firmware 2026.2.1).
