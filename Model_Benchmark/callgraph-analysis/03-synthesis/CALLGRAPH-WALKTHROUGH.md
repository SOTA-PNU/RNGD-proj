# `furiosa-llm serve` 전체 콜그래프 — gdb · pdb(Python tracing) · eBPF 통합 분석

대상 명령:

```bash
furiosa-llm serve /home/jun/chacha/qwen2.5-coder-7b-inst-tp8 --port 12345
```

이 문서는 위 명령을 **실제로 실행**하고 세 가지 계측(gdb / Python 디버거 계층 / eBPF 커널 후킹)으로 캡처한 결과 + 내부 코드 정적 분석을 결합해, **빌드(=serve 시점의 아티팩트 로드·엔진 구성·디바이스 매핑·워밍업)** 와 **추론 요청 처리**의 전체 콜그래프를 Python ↔ 네이티브(Rust) ↔ 커널(furiosa_rngd) ↔ 하드웨어 4계층으로 재구성한다.

> 메인 그래프 2장: [`G1-startup-load-fullstack.svg`](G1-startup-load-fullstack.svg) (기동/로드), [`G2-inference-fullstack.svg`](G2-inference-fullstack.svg) (추론). 정적 구조 그래프는 [`../01-static/`](../01-static/).

---

## 0. 실행 요약 (관측된 실제 토폴로지)

| 항목 | 값 (serve.log / bpftrace 실측) |
|---|---|
| 아티팩트 스키마 | 3.0 (`furiosa_llm_common::artifact`) |
| 병렬화 | **tp=8, pp=1, dp=4** (텐서병렬 8 × 데이터병렬 4) |
| 디바이스 매핑 | DpId(0)→`npu0pe0-3,pe4-7`, DpId(1)→`npu1…`, (2)→`npu2…`, (3)→`npu3…` (4 카드 전부, 카드당 8 PE를 2개 융합 그룹으로) |
| 가중치 | 14.2 GiB, 0.99 s 에 적재 (`backing_file`, safetensors mmap) |
| KV 캐시 | 31.3 GiB/카드, 585,555 blocks, block_size=1024, prefix-cache on |
| 구조화 출력 | XGrammar + LLGuidance(`llg_*`) 백엔드 |
| 스케줄러 | Eager, DP 라우팅=PrefixAware, npu_queue_limit=1 |
| 기동 시간 | ~18 s (로드→ready), 추론 3요청 모두 200 OK |
| 이벤트 루프 | **uvloop** (libuv는 `libtorch_cpu.so`에 번들) |
| 네이티브 스레드 | 프로세스당 ~656 스레드 (tokio 128 + scheduler-eager ~260 + rayon + blas + libuv + main) |

검증된 추론 결과(예): `/v1/chat/completions` 가 정상 Fibonacci 함수 160토큰 생성, `/v1/completions` 스트리밍이 토큰 단위 SSE 반환. (`02-dynamic/logs/chat_response.json`, `completion_stream.txt`)

---

## 1. 계측 방법론 — gdb / pdb / eBPF 를 무엇이 담당했는가

요청은 "gdb·pdb·eBPF 기반 커널 후킹"이었다. 각 계층을 실제로 담당한 도구와 그 이유:

| 요청한 계층 | 실제 사용 도구 | 담당 범위 | 비고 |
|---|---|---|---|
| **pdb** (Python 디버거 계층) | **viztracer**(결정론적 전수 추적) + **py-spy**(샘플링) | Python 콜그래프 전체 (import→LLM 구성→generate) | pdb는 대화형 중단점 도구라 콜그래프 캡처에 부적합. pdb와 동일한 `sys.settrace`/프로파일 메커니즘을 쓰는 viztracer로 **전수 콜트리**(11.27M 이벤트, 깊이 318)를, py-spy로 라이브 서버 스택을 캡처. |
| **gdb** | `sudo gdb -p <pid> -batch "thread apply all bt"` | 네이티브 스레드 구조 / C·Rust 프레임 / 메인 uvloop 스택 | 로드 중·idle·추론 중 3회 스냅샷. native_runtime.so가 **스트립**이라 내부 프레임은 주소(`??`)로 나오나 **스레드 분류·경계 프레임**은 확정. |
| **eBPF 커널 후킹** | **bpftrace** (kprobe + `renegade_pdma` 트레이스포인트) | `furiosa_rngd` 드라이버 호출: open/ioctl/mgmt_admin/DMA/doorbell/IRQ | 커널 경계의 ground-truth. 비특권 eBPF는 차단(`unprivileged_bpf_disabled=2`)이라 root로 실행. |
| 보조 (이 호스트 미수집) | **perf** (`-g --call-graph dwarf`) | (의도) 네이티브+커널 샘플 콜스택 | `-p` attach 시 유효 샘플 0건(rc=255). py-spy `--native`도 656 스레드를 못 따라가 중단(샘플 300s+ 지연). **네이티브 레이어는 gdb로 대체** (§7). |
| 디바이스 매핑 | `/proc/<pid>/fd`, `furiosa-smi`, lsof | 어떤 `/dev/rngd/*` 노드를 여는지 | tp8/dp4 실증. |

환경 제약(실측):
- `ptrace_scope=1` → 내가 띄운 프로세스/자식에는 gdb·py-spy 부착 가능. 그 외엔 root 필요 → 본 분석은 sudo로 부착.
- `unprivileged_bpf_disabled=2`, `perf_event_paranoid=4` → **비특권 eBPF/perf 전면 차단**. bpftrace/perf 모두 root.
- `native_runtime.so`: 163 MB, **fully stripped** (`.symtab` 0개, `.dynsym` 56개 = `llg_*` llguidance C-ABI뿐). ⇒ 네이티브 함수단위 그래프는 바이너리로 복원 불가 → **RUST_LOG 모듈 경로**가 네이티브 콜그래프의 권위 있는 레벨.

---

## 2. 그래프 G1 — 기동 / 로드 (serve 시점의 "빌드")

전체: [`G1-startup-load-fullstack.svg`](G1-startup-load-fullstack.svg)

### 2.1 CLI → uvicorn (Python)
```
furiosa-llm(console_script)
└─ cli.main.main()                         cli/main.py:9   (argparse: build|serve|collect-env|version)
   └─ cli.serve.serve(args)                cli/serve.py:384
      └─ server.app.run_server(args)       server/app.py:533
         ├─ init_app(args)                 server/app.py:376   → app 구성
         └─ uvicorn.run(app, host, port)   server/app.py:549   → 이벤트 루프(uvloop) 진입
```

### 2.2 init_app: 엔진 적재 + 핸들러 와이어링 (Python)
```
init_app(args)                              server/app.py:376
├─ LLM(model, devices=None, ...)            api.py:115        ★ Python→네이티브 경계의 시작
├─ OpenAIServingChat / Completion / Embedding / Responses / Tokenization   app.py:413-446
└─ FastAPI(lifespan=…)                      app.py:467        라우트 등록 + 시작/종료 훅
```

### 2.3 LLM 로드 → 네이티브 엔진 (Python→Rust 경계)
```
LLM.__init__                                api.py:115
└─ LLM._init_from_artifact                  api.py:321
   ├─ resolve_artifact_path                 api.py:346
   ├─ NextGenArtifact.load_without_blob     api.py:349   ⟹ native_llm_common  [경계]
   ├─ NextGenArtifact.override_with         api.py:354   ⟹ native_llm_common  [경계, viztracer 확증]
   ├─ resolve_devices(devices)              api.py:352 → device.py
   └─ NativeLLMEngine.__init__(artifact, devices)   api.py:383   ⟹ native_runtime  [경계 ★]
```
viztracer가 잡은 **네이티브 경계 리프**(이 아래는 Python 프레임 없음): `NextGenArtifact.override_with`, `NativeLLMEngine.generate/shutdown`, `tokenizers.Tokenizer.encode_batch`, `pydantic_core.SchemaValidator.validate_json/python`. 로드 경로는 `parallelize/pipeline/next_gen.py`의 `SymbolVal.validate_model`·`Stage.serialize_tasks`를 pydantic으로 대량 검증(아티팩트에 직렬화된 파이프라인 태스크 그래프).

### 2.4 네이티브 런타임 (RUST_LOG 모듈 흐름 — serve.log 실측)
```
furiosa::llm::engine            아티팩트 로드(schema 3.0), cfg 파싱 tp=8 pp=1 dp=4
└─ device_runtime::context      디바이스 open; Device([npu:k:0-3,4-7]) 별 memory-dump 스레드 기동
   └─ furiosa_generator::next_gen::hf_compat_next_gen   타깃 모델 로드(1.40 s)
      ├─ next_gen::pipeline::resolve    할당 plan: weights 14.2GiB, IO 2.0GiB, KV 31.3GiB/dev
      ├─ backing_file                   14.2GiB 파라미터 적재(0.99 s, mmap)
      ├─ structured_output::manager     XGrammar + LLGuidance 초기화
      └─ next_gen::generator (×4 DP)    DpId(k)→npu k; 
         ├─ scheduler::task_selector    AOT 파이프라인 와이어링(jit=off)
         ├─ scheduler::memory_manager   KVCacheManager 585,555 blocks
         └─ Eager scheduler started     (4 DP 그룹 각각)
```

### 2.5 커널 (furiosa_rngd — bpftrace 실측 카운트)
```
openat /dev/rngd/npu{0..3}{mgmt,bar0/2/4,pe0-3,pe4-7,ch0-7,ch*r,dmar,p2pmem}
└─ npu_pdma_open ×72 · npu_dmar_ncdev_open ×8 · npu_bar_ncdev_open
   └─ npu_mgmt_admin_cmd ×820            펌웨어 init/config (comm=tokio-runtime-w)
      └─ npu_dmar_ncdev_ioctl ×16        PE-group 설정 (cmd 0xC0046706, 0xC0C0…; 4카드×2그룹)
         └─ 가중치 DMA:  doorbell_sq_write → npu_dma_transfer_start ~8000× (kworker/u515)
            → npu_doorbell_handler IRQ            14.2GiB → HBM
```

---

## 3. 그래프 G2 — 추론 요청 처리

전체: [`G2-inference-fullstack.svg`](G2-inference-fullstack.svg)

### 3.1 HTTP 수신 → 서버 (Python, uvloop 메인 스레드)
gdb로 확인한 메인 스레드(Thread 1, "furiosa-llm"):
```
epoll_pwait ← uv.io_poll ← uv_run(libtorch_cpu.so의 libuv) ← uvloop Loop._run
← run_forever ← run_until_complete ← _PyEval_EvalFrameDefault ← Py_RunMain
```
요청 처리:
```
FastAPI route @app.post('/v1/chat/completions')          server/app.py
└─ OpenAIServingChat.create_chat_completion              server/serving_chat.py
   ├─ chat_utils: 채팅 템플릿 렌더(content-format 'string')
   ├─ tokenizers.Tokenizer.encode_batch                  [경계] 네이티브 HF 토크나이저
   └─ AsyncLLMEngine.generate  (llm_engine.py:578) / LLM
      └─ NativeLLMEngine.stream_generate(prompt_ids, sp)  api.py:606 / llm_engine.py:611
                                                          [경계 ★] 네이티브 async generator
```
(비스트리밍/offline 경로는 `LLM.generate` api.py:419 → `self.engine.generate` api.py:466.)

### 3.2 네이티브 디코드 루프 (Rust; gdb 스레드 분류)
추론 시 호스트는 대부분 NPU 완료를 대기(파킹)하고, 실제 연산은 NPU에서 수행:
```
next_gen::generator           요청 enqueue → DP 라우팅(PrefixAware)
└─ scheduler (Eager)          task_selector: 버킷에 맞는 AOT 파이프라인 선택
   ├─ memory_manager          KV 블록 할당(block=1024), prefix-cache 조회
   ├─ structured_output       (xgrammar/llg_*) 토큰별 로짓 마스크(제약 시)
   └─ executor                커맨드 디스크립터 작성 → PE 커맨드 링 제출
      └─ poll CQ → 로짓 수집 → 다음 토큰 샘플 → (토큰별 반복)
```
gdb 스냅샷 순간의 스레드 분류(656 스레드): `scheduler-eager` ×260(native_runtime.so 안 7프레임 깊이로 `syscall` 대기), `tokio-runtime-w` ×128, `furiosa-llm` futex 대기 ×126, `libuv-worker` ×4, blas_thread_server, rayon 풀.

### 3.3 커널 (furiosa_rngd — 추론 구간 bpftrace 실측)
```
executor → doorbell_sq_write                          PE 커맨드 링에 제출 ★ 추론 명령 경로
   · 스텝별 다수는 네이티브 엔진 워커 comm=tokio-runtime-w (Run C 추론 윈도우 6,624건)
   · 고수준/메인 comm=furiosa-llm 32건
   → NPU 가 prefill/decode 실행 (8 PE × 4 DP)
   → npu_doorbell_handler IRQ  16,416건                완료 인터럽트
   → npu_pdma_dma_transfer_* + dma_wait(kworker)       KV/액티베이션 DMA
   → doorbell_cq_read                                  네이티브가 완료 큐 read
```
주: 공유 박스에 furiosa-smi 모니터링이 상주하여 doorbell 트래픽 일부(comm=furiosa-smi, Run C 윈도우 9,760건)는 추론과 무관한 노이즈. 엔진 라우팅은 DP 단위(`furiosa_llm.server.metrics`가 `[Engine k]`별 throughput 로깅; 예: Engine 0 = 32.2 tok/s, Running 1 req).

### 3.4 응답 (스택 역방향)
```
NativeRequestOutput / NativeCompletionOutput  [경계] native→Python (viztracer 확증)
└─ _generate_postprocess / 스트리밍 델타 빌드        serving_chat.py
   └─ FastAPI StreamingResponse  'data: {…}\n\n' (SSE, 토큰별)
      └─ uvloop write → curl 수신
```

---

## 4. 계층별 심화

### 4.1 Python 계층 (정적 + viztracer + py-spy)
- 정적 콜그래프 전문: [`../01-static/static-callgraph.md`](../01-static/static-callgraph.md) (A 기동/로드, B 추론, C 빌드, 네이티브 경계 표).
- viztracer 결정론적 그래프 — 파싱 산출물: 엣지 `viz_load_infer_edges.tsv`(5.8 MB), 요약 `viz_callgraph_summary.txt`, 그래프 `viz_load_infer.dot/.svg`(상위 120 엣지). 원시 1.5 GB JSON은 용량상 삭제했으며 `run_B.sh`로 재생성 가능.
- **Python→네이티브 경계(전수 확증)**: `NativeLLMEngine.__init__/generate/stream_generate/encode/shutdown`, `NextGenArtifact.load_without_blob/override_with`, `tokenizers.Tokenizer.encode_batch`, `pydantic_core.SchemaValidator.*`, `furiosa.native_torch.enable_logging`.
- **전수 텍스트 콜그래프(모든 함수, 필터 없음)**: [`full-callgraphs/viz_full_calltree.txt`](full-callgraphs/viz_full_calltree.txt)(루트부터 전체 콜트리), `viz_full_adjacency.txt`(함수→callee+횟수), `viz_full_reverse.txt`(callee←caller). 17,670 함수 / 41,331 엣지. 폴더 설명: [`full-callgraphs/README.md`](full-callgraphs/README.md).

### 4.2 네이티브 계층 (Rust, `native_runtime.so` — 스트립)
- 함수명 복원 불가(스트립). 권위 있는 신호 = **RUST_LOG 모듈 경로**(serve.log): `furiosa::llm::engine`, `furiosa_generator::next_gen::{hf_compat_next_gen, pipeline::resolve, generator, scheduler::{request_management::task_selector, memory_manager}}`, `device_runtime::context`, `furiosa_generator::backing_file`, `structured_output::manager`.
- 스레딩 모델(gdb): tokio 멀티스레드 런타임 + DP별 `scheduler-eager` 스레드 + rayon(데이터병렬 CPU) + OpenBLAS(`blas_thread_server`) + uvloop libuv 워커.
- **전수 텍스트 콜그래프(모든 함수, 모든 스레드/프레임)**: [`full-callgraphs/gdb_{load_10s,idle,infer}.calltree.txt`](full-callgraphs/) (병합 콜트리), `.archetypes.txt`(전체 스레드 스택 종류), `.adjacency.txt`(caller→callee). 스트립된 furiosa 프레임은 `native_runtime.so!0xADDR`, tokio/std/parking_lot/rayon/blas/libuv 는 심볼명. (load 5종/idle 30종/infer 29종 스택, 63~101 함수)
- **`??` 네이티브 주소 간이 이름**: [`full-callgraphs/gdb_infer.native_names.md`](full-callgraphs/gdb_infer.native_names.md) (38개 주소→추론 이름+근거), [`gdb_infer.calltree.named.txt`](full-callgraphs/gdb_infer.calltree.named.txt) (이름 적용 콜트리). 3 스냅샷 동일 프로세스라 주소=동일 함수; `core`(eager 스케줄러 파킹), `iodrv`, `worker` 3개 풀로 식별. 추론치임.
- 노출된 C-ABI(`nm -D`)는 `llg_*`(llguidance 문법) 56개뿐 → 구조화 출력 경로만 외부 심볼.

### 4.3 커널 계층 (`furiosa_rngd` 드라이버)
- 디바이스(major): `rngd_mgmt`=234(mgmt/bar/pe/dmar/p2pmem), `rngd_pdma`=510(ch0-7, ch*r), `rngd!ttyNPU`=511. 노드는 `/dev/rngd/` 아래.
- 드라이버 함수 859개 + 자체 커널 트레이스포인트 서브시스템 **`renegade_pdma`**(`npu_pdma_open`, `dma_transfer_sync/async_start/stop`, `prepare_request`, `wait_request`, `iopoll`, `direct_io` 등 30+).
- 후킹한 fops/명령 경로: `npu_bar_ncdev_ioctl/mmap/open`, `npu_dmar_ncdev_ioctl`, `npu_mgmt_admin_cmd`, `doorbell_sq_write/cq_read`, `npu_dma_transfer_start`, `npu_doorbell_handler`, `npu_dma_isr`.
- 실측 카운트 — 로드(Run A): opens(pdma 72/pdma_remote 64/dmar 8), mgmt_admin 820, dmar ioctl 16(2종×8), weight DMA(`dma_transfer_start`) ~8,000(kworker). 추론(Run C 윈도우): `doorbell_sq_write` tokio-runtime-w 6,624 + furiosa-llm 32 (+ furiosa-smi 9,760 노이즈), `npu_doorbell_handler` IRQ 16,416. 전체 raw: `02-dynamic/logs/kernel_trace*.log`.
- **주의(귀속)**: 동일 박스에 furiosa-smi 모니터링이 상주(`comm=furiosa-smi`, `tokio-runtime-w`)하여 mgmt_admin/doorbell 상당량이 모니터링 트래픽. 추론 명령 제출의 실체는 네이티브 엔진 워커(`tokio-runtime-w`)의 `doorbell_sq_write`이며, 완료는 `npu_doorbell_handler` IRQ. (kworker DMA는 IRQ/워크큐 컨텍스트라 pid 귀속이 약함 — `comm`으로만 구분.)

---

## 5. 증거 색인 (어떤 파일이 무엇을 증명하나)

| 주장 | 증거 파일 |
|---|---|
| tp8/dp4, 디바이스 매핑, KV/weights 크기, 스케줄러 | `02-dynamic/logs/serve.log`, `serve_C.log` |
| 연 디바이스 노드(tp8 실증) | `02-dynamic/logs/npu_fds.txt` |
| 커널 open/ioctl/DMA/doorbell 카운트·순서 | `02-dynamic/logs/kernel_trace.log`(Run A), `kernel_trace_B.log`(로드 전용), `kernel_trace_C.log`(추론) |
| 네이티브 스레드 구조 / 메인 uvloop 스택 | `02-dynamic/logs/gdb_idle.txt`, `gdb_infer.txt`, `gdb_load_10s.txt`, `*_archetypes.txt` |
| Python 전수 콜그래프 + 경계 리프 | `02-dynamic/logs/viz_*` |
| (시도) 네이티브+커널 샘플 | `02-dynamic/logs/perf_infer.log`, `pyspy_infer.log` — 이 호스트에서 유효 샘플 미수집(§7); 네이티브는 gdb로 대체 |
| 추론 정상 동작 | `chat_response.json`, `chat_response2.json`, `completion_stream.txt`, `long_gen*.json` |
| 드라이버 심볼/트레이스포인트 목록 | `02-dynamic/logs/furiosa_rngd_symbols.txt`, `furiosa_rngd_tracepoints.txt` |

---

## 6. 재현 방법

```bash
# 정적 콜그래프 (멀티에이전트 추출)는 01-static/ 에 생성됨.
# 동적 캡처 (root 필요: bpftrace/perf/gdb 부착):
cd /home/jun/chacha/callgraph-analysis/02-dynamic/scripts
bash run_A.sh     # 라이브 serve + bpftrace + gdb 스냅샷 + perf/py-spy + 추론요청
bash run_B.sh     # viztracer 결정론적 Python 콜그래프 + 로드 전용 커널 트레이스
bash run_C.sh     # 수정된 perf(dwarf) + py-spy(--native) on 지속 생성
# 합성/렌더:
cd ../../03-synthesis && dot -Tsvg G1-startup-load-fullstack.dot -o G1-startup-load-fullstack.svg
python ../02-dynamic/scripts/parse_viz.py ../02-dynamic/logs/viz_load_infer.json
python ../02-dynamic/scripts/extract_gdb.py ../02-dynamic/logs/gdb_idle.txt
```
bpftrace 스크립트: `02-dynamic/scripts/kernel_trace.bt`.

---

## 7. 한계 / 캐비엇
1. **네이티브 함수단위 그래프 불가** — `native_runtime.so`가 스트립. 모듈 단위(RUST_LOG)까지가 한계. (디버그 심볼 빌드가 있으면 gdb/perf로 함수단위 복원 가능.)
2. **추론 시 호스트는 대부분 대기** — 연산은 NPU에서 수행되므로 CPU 백트레이스는 "완료 대기" 프레임이 지배적(정상). 명령/완료의 실체는 커널 doorbell/IRQ 카운트로 포착.
3. **perf / py-spy(--native) 이 호스트에서 미작동** — `perf record -g --call-graph dwarf -p <pid>`는 유효 샘플 0건(rc=255; `perf_event_paranoid=4` + 656 스레드 dwarf 언와인딩 추정), py-spy `--native`는 656 스레드를 100Hz로 못 따라가 샘플이 300s+ 지연되어 중단. ⇒ 네이티브 레이어는 **gdb 스레드 백트레이스**로, Python 레이어는 **viztracer 전수 추적**으로 대체(둘 다 성공). 즉 요청한 3계층(gdb/pdb/eBPF)은 전부 확보되며, perf/py-spy 플레임그래프만 환경 제약으로 미생성.
4. **모니터링 노이즈 귀속** — furiosa-smi 상주로 일부 커널 이벤트는 serve가 아님. `comm` 으로 분리.
5. **DMA tracepoint 카운트는 로드+모니터링 혼재** — kworker 컨텍스트라 pid 귀속이 약함. 핵심 추론 신호는 `comm=furiosa-llm` doorbell.
