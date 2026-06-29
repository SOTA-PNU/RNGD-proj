# `furiosa-llm build` 전체 콜그래프 — gdb · pdb(py-spy) · eBPF 통합 분석

대상 명령(실제 실행):

```bash
furiosa-llm build <Qwen2.5-Coder-1.5B-Instruct> <out> -tp 4 --max-model-len 2048
```

이 문서는 위 명령을 **실제로 실행**하면서 세 가지 계측(gdb / 파이썬 디버거 계층 / eBPF)으로 캡처한 결과에 내부 코드 정적 분석을 결합해, `furiosa-llm build`(HF 모델 → RNGD 아티팩트 AOT 컴파일)의 전체 콜그래프를 재구성한 것입니다. serve 분석([`CALLGRAPH-WALKTHROUGH.md`](CALLGRAPH-WALKTHROUGH.md))과 **같은 방식**으로 작성했고, 빌드에만 있는 두 가지 특징을 함께 다룹니다.

> 메인 그래프: [`GB1-build-fullstack.svg`](GB1-build-fullstack.svg). 정적 file:line 레퍼런스: [`../01-static/build-static-callgraph.md`](../01-static/build-static-callgraph.md).

## serve 와 다른 두 가지 (먼저 알아두면 좋은 것)

1. **빌드는 경계를 둘 넘습니다.** serve 는 `Python → 네이티브(Rust)` 한 번이지만, 빌드는
   - **드라이버 → Ray 워커**(프로세스 경계): 무거운 일(FX 트레이싱·AOT 컴파일)은 드라이버가 아니라 **Ray 액터 프로세스** 안에서 돕니다. 드라이버는 `ray.get(...)` 에서 멈춰 기다립니다. `.remote()` 호출이 곧 경계입니다(serve 의 PyO3 호출에 해당).
   - **Python → 네이티브**(PyO3 경계, 워커 안): 컴파일러 본체는 `furiosa.native_common.compiler` 이고, 그 실체는 `furiosa/native_llm_common.cpython-312*.so`(143 MB, **스트립**)입니다.
2. **빌드는 NPU 를 한 번도 건드리지 않습니다.** 드라이버·워커 어느 것도 `/dev/rngd/*` 를 열지 않고, 커널 트레이스에 빌드발(發) doorbell/DMA 가 0 입니다. 컴파일은 순수 호스트(CPU) 작업입니다. (serve 의 4계층 중 **커널/하드웨어 계층이 빠진** 셈입니다.)

---

## 0. 실행 요약 (관측값)

| 항목 | 값 (build.log / py-spy / gdb / bpftrace 실측) |
|---|---|
| 모델 | Qwen2.5-Coder-1.5B-Instruct (`qwen2`, 28 layers, heads 12 / kv 2, bf16 = `W16A16KV16`) |
| 토폴로지 | **tp=4** (= preset 의 `SUPPORTED_TP_SIZES={4,8,32}` 중 1.5B 가능 최소값), pp=1 |
| 버킷 | preset 사용(`Found bucket preset for model_type=qwen2 …` → `Filtered … by max_model_len=2048`) |
| 워커 | pipeline-builder 1 / compile 1 (기본값) — 그래도 Ray 로컬 클러스터를 띄움 |
| 단계 | **트레이싱 49 태스크(~22분) → 컴파일 78 태스크(stage_0에서 실패)** |
| 결과 | **실패(rc=1).** `failed to lower the operator O1089 (no tactic)` — stage_0 (Embedding → TransformerBlock0 의 QkvProjection). 총 40분 52초, 최대 RSS 6.99 GB |
| 드라이버 | 1 프로세스 ~120 스레드, 빌드 내내 `build_pipeline → ray.get` 에서 파킹 |
| 워커1 | `ray::LocalPipelineGenerationActor` (`@ray.remote num_cpus=24`) — 트레이싱+파티션+병렬화 |
| 워커2 | `ray::TaskCompileActor` (`@ray.remote num_cpus=32`, 244 스레드) — 컴파일(네이티브 lowering 풀) |
| NPU | **미접촉.** 디바이스 open 0, DMA 0, bar mmap 0. 윈도 내 `furiosa_rngd` 트래픽은 전부 상주 furiosa-smi 모니터(`comm=tokio-runtime-w`, doorbell/mgmt 7356건) 노이즈 |

> 빌드가 **실패해도 콜그래프는 온전합니다.** 오히려 컴파일러가 stage_0 lowering 을 17분간 돌다 포기했기 때문에, 그 17분 사이에 찍은 gdb 스냅샷이 **네이티브 컴파일러를 한창 돌고 있는 상태**로 잡혔습니다(아래 3.3). 즉 "no tactic" 실패는 분석에 오히려 유리한 데이터였습니다. (`failed to lower … (no tactic)` 는 이 SDK 에서 알려진 컴파일러 에러 계열입니다 — `info/README_build.md`, `README_config.md`, `README_tp32_build.md`.)

---

## 1. 계측 방법론 — gdb / pdb / eBPF 가 무엇을 담당했나

요청은 serve 와 동일하게 "gdb · pdb · eBPF 기반". 빌드는 멀티 프로세스(드라이버 + Ray 워커들)라서 계측을 **프로세스별로** 붙였습니다.

| 요청한 계층 | 실제 도구 | 담당 | 비고 |
|---|---|---|---|
| **pdb**(Python 디버거 계층) | **py-spy**(dump+record) | 드라이버·워커의 **Python 콜스택**(심볼화) | pdb 와 같은 인터프리터 스택을 읽습니다. 드라이버는 한 스택(=ray.get 경계까지)이 결정적이라 dump 로 충분, 워커는 트레이싱 국면을 dump 여러 장 + flamegraph 로 캡처. **viztracer 를 워커에 못 쓰는 이유**: 일이 Ray 워커(별도 프로세스)에서 도는데 viztracer 는 자기가 띄운 프로세스만 따라갑니다(=serve 의 "네이티브는 viztracer 로 못 본다"와 같은 한계). 그래서 워커 Python 은 py-spy 로 잡았습니다. |
| **gdb** | `sudo gdb -p <pid> -batch "thread apply all bt"` | 드라이버 + 워커의 **네이티브 프레임/스레드 구조** | 트레이싱 워커 1장, 컴파일 워커 5장(같은 프로세스), 드라이버 1장. `native_llm_common.so` 가 **스트립**이라 컴파일러 프레임은 주소(`??`)로 나오며 → **간이 명명**(3.3). |
| **eBPF 커널 후킹** | **bpftrace**(`furiosa_rngd`) | 빌드가 NPU 를 건드리는지 | ground-truth: **건드리지 않음**(open/DMA/mmap 0). |
| 디바이스 fd 스캔 | `/proc/<pid>/fd` | rngd 노드 open 여부 | 드라이버·워커 모두 0 → NPU 미접촉 확증. |

환경 제약(serve 와 동일): `ptrace_scope=1`(내가 띄운 프로세스엔 sudo 로 gdb/py-spy 부착 가능), `unprivileged_bpf_disabled=2`(bpftrace 는 root). `native_llm_common.so`: 143 MB, **fully stripped**(`nm` 심볼 0, dynsym 518 = `PyInit_*`/`tch_*_stream_*`/`llg_*`/`perf_signal_handler` 뿐) → 네이티브 함수단위 그래프는 바이너리로 복원 불가 → **주소 클러스터 + 콜래더 + Python 경계**로 간이 명명.

---

## 2. 그래프 GB1 — 빌드 풀스택

전체: [`GB1-build-fullstack.svg`](GB1-build-fullstack.svg). 4개 박스 = 드라이버 / Ray 워커1 / Ray 워커2 / 네이티브. 두 굵은 화살표(초록·보라)가 드라이버→워커 경계, 노란 박스가 PyO3 네이티브 경계입니다.

### 2.1 드라이버 (Python, 1 프로세스 — py-spy 로 확정)

py-spy 가 빌드 내내 드라이버 MainThread 를 정확히 이 스택에서 잡았습니다(=결정적 드라이버 콜그래프, `ray.get` 에서 블록):
```
<module> furiosa-llm:6
└─ cli.main.main                         cli/main.py:28
   └─ cli.convert.convert (build 핸들러)  cli/convert.py:124 → builder.build :214
      └─ ArtifactBuilder.build            artifact/builder.py:315
         ├─ ensure_model_and_update_weight_hash :362  (가중치 해시; "Calculated the hashsum 2.8s")
         ├─ ArtifactBuilder._build_model_artifact   builder.py:172 → build_pipeline :268
         │  └─ build_pipeline             new_pipeline_builder.py:1474
         │     └─ ray.get(done_ref)       new_pipeline_builder.py:1586  ⟹ [경계 ★ 드라이버는 여기서 정지]
         └─ ArtifactBuilder.__save_artifacts  builder.py:481  (실패 시 미도달)
```
`__init__`(`builder.py:116`)에서 load/validate/resolve(아키텍처 지원·`SUPPORTED_TP_SIZES`·버킷·`presets.find_preset`)를 마친 뒤 build 로 들어옵니다. 드라이버 gdb 스냅샷은 ~120 스레드가 대부분 Ray RPC poll/epoll/syscall 로 워커를 기다리는 모습(=serve 메인 스레드가 NPU 완료를 기다리던 것과 같은 자리).

### 2.2 드라이버 → Ray 워커 (프로세스 경계)

```
build_pipeline       → LocalPipelineGenerationActor.options(num_cpus=24).remote()   :1557   (워커1 생성)
                     → actor.build_for_bucket.remote(bucket)                         :1576   ① 트레이싱 제출 [★]
                     → ray.get(done_ref)                                             :1586         (대기)
get_compiled_pipeline→ TaskCompileActor.options(num_cpus=32).remote()                :1281   (워커2 생성)
                     → actor.compile_task.remote(stage, task, …)                     :1312   ② 컴파일 제출 [★]
                     → ray.get(done_ref)                                             :1320         (대기)
                     → CompiledGraph.deserialize(...)                                :1343   ⟹ 네이티브
```

### 2.3 네이티브 (워커 안, RUST 로깅/스트립 .so)

컴파일러 본체는 `native_llm_common.so` 의 `furiosa.native_common.compiler`. 핵심 경계는 `compile()`(`converter.py:913`)이며, 그 직전에 `furiosa_torch_ext.torch_ext.preprocess`(`converter.py:886`, 별도 .so)가 그래프를 다듬습니다.

---

## 3. 단계별 심화

### 3.1 드라이버 (정적 + py-spy)
- 결정적 콜스택은 위 2.1. 전수 텍스트: [`full-callgraphs/gdb_build_driver.*`](full-callgraphs/)(120 스레드, 24 아키타입 — Ray client/server poll·epoll·syscall 지배). Python 측: [`../02-dynamic/logs/pyspy_build_driver.txt`](../02-dynamic/logs/pyspy_build_driver.txt).
- 정적 file:line 전부: [`../01-static/build-static-callgraph.md`](../01-static/build-static-callgraph.md) (A 드라이버 / B 워커1 / C 워커2 / D 저장 + 네이티브·Ray 경계표).

### 3.2 워커1 — `LocalPipelineGenerationActor`: 트레이싱 + 파티션 + 병렬화 (py-spy 로 확정)
py-spy 가 트레이싱 국면의 워커를 여러 하위 단계에서 잡았습니다(`../02-dynamic/logs/pyspy_build_trace.txt`, flamegraph `pyspy_build_trace.svg`):
```
build_for_bucket                         new_pipeline_builder.py:1451
└─ build_local_pipeline                  :580
   └─ build_partitioned_graphmodule      :412
      ├─ get_aten_graph_with_metadata    :455 → trace.py:1230 → _get_aten… :1180
      │   ├─ trace_model → torch._dynamo.export (trace.py:855) / make_fx (trace.py:828)
      │   │      └─ (py-spy 실측: _dynamo/{symbolic_convert,guards,variables}.py, fake-tensor 전개 — 다이나모 엔진룸)
      │   └─ save → export.graphmodule.save_gm  (trace.py:1087 → graphmodule.py:426)  (트레이스 그래프를 캐시에 직렬화)
      └─ parallelize_and_partition_graphmodule   :486
         ├─ KernelwisePartitioner.partition_gm   graph_partitioner.py:97 (color bitmap :65, marker block_slicer.py:1258)
         └─ parallelize_graphmodule :367 → ModelRewriter.rewrite (model_rewriter/api.py:101)
                → ShardingPropagator.propagate (sharding_propagator.py:241)   (tp=4 샤딩 적용)
```
즉 트레이싱 국면의 워커는 **순수 Python**(torch dynamo + make_fx + furiosa 파티셔너/리라이터)입니다. 이 국면 gdb 스냅샷에서 네이티브 프레임은 Ray(`_raylet.so`)뿐, furiosa 컴파일러 프레임은 아직 없습니다 — 컴파일러는 다음 국면(3.3)에서야 등장합니다.

### 3.3 워커2 — `TaskCompileActor`: 컴파일 + `??` 네이티브 간이 명명 (★ 핵심)
py-spy 가 잡은 경계: `compile_task (new_pipeline_builder.py:1165) → compile_gm_and_get_preprocessed_gm_hash → compile() (converter.py:913)` — 여기서 **PyO3 로 native_llm_common.so 진입**, 17분간 lowering 후 "no tactic"으로 실패. gdb 스냅샷 5장(같은 프로세스 LWP 3105741)에 컴파일러가 한창 도는 모습이 잡혔습니다(snap1 에 native 프레임 5008개).

`native_llm_common.so` 가 스트립이라 컴파일러 프레임은 `native_llm_common.so!0xADDR`(`??`)로만 보이므로, serve 와 동일하게 **간이(provisional) 이름**을 부여했습니다 — 표: [`full-callgraphs/gdb_build.native_names.md`](full-callgraphs/gdb_build.native_names.md)(802 주소, region 6 + hot 12), 이름 적용 콜트리: [`full-callgraphs/gdb_build.calltree.named.txt`](full-callgraphs/gdb_build.calltree.named.txt).

라이브로 잡은 두 콜래더가 명명의 근거입니다:
```
[파이썬 MainThread]  _PyEval → CPython call → compile.driver.enter(0x…19513d61, region 19)
                     → … compile 오케스트레이션 … → compile.wait.SYSCALL(0x…1ded6c05, region 1d) → syscall  (워커풀 대기)
[활성 lowering 풀스레드] lower.tactic.leaf(0x…1da50e14, region 1d) ★ 'no tactic' 발원지
                     → mid-lower(region 1b) → passA(region 1a)
                     → 재귀 operator visitor  recurse.A(0x…19989eb9) ↔ recurse.C(0x…19b67a4d) ↔ recurse.B(0x…19b897b9) … 깊게 반복
[파킹된 풀스레드 ~62개] clone3 → start_thread → pool.worker.park(0x…1fbccdaa) → syscall  (작업큐 대기)
```
**주소 영역(region) = 컴파일 패스**로 묶어 이름을 줬습니다(serve 의 core/iodrv/worker 클러스터와 같은 발상):

| region | 간이 역할 | 주소 수 |
|---|---|---:|
| `0x..19xxxxxx` `lower.drv` | lowering 드라이버 + 재귀 operator visitor (메인 lowering 루프) | 78 |
| `0x..1axxxxxx` `lower.pA` | lowering 서브패스 A | 70 |
| `0x..1bxxxxxx` `lower.mid` | 중간 lowering / IR 변환 | 244 |
| `0x..1cxxxxxx` `lower.cg` | lowering/codegen 서브패스 | 55 |
| `0x..1dxxxxxx` `lower.tac` | 최내곽 tactic 선택/codegen(leaf) + 드라이버 sync | 354 |
| `0x..1fxxxxxx` `pool` | 컴파일러 워커풀 진입/파킹 | 1 |

스레드 구조(gdb 244 스레드): `compile.driver`(파이썬 MainThread 1) + `lower.pool`(~62 파킹 + 활성 수~수십, native_llm_common.so 의 자체 병렬 풀) + `ray.infra`(event_engine·nexting·grpc·poll ~50, 컴파일러 아님). **이름은 전부 추론치**이며 심볼화된 사실이 아닙니다(스트립). 단, 존재(Presence) 열로 안정 프레임(5장 모두 등장 = 드라이버/파킹)과 일시 프레임(1~2장 = 활성 lowering)을 구분했습니다.

### 3.4 커널 / NPU — 없음 (대조)
bpftrace(`../02-dynamic/logs/kernel_trace_build.log`) 집계: device open 0, bar mmap 0, dma_transfer 0, pdma 0. doorbell/mgmt 7356건은 전부 `comm=tokio-runtime-w`(상주 furiosa-smi 모니터)로, 빌드 프로세스(`furiosa-llm build`·Ray 워커)가 아닙니다. fd 스캔(`build_fds.txt`)도 드라이버·워커 모두 rngd 노드 0. ⇒ **빌드는 호스트 CPU 전용**이고 컴파일 결과(EDF/precommand blob)는 나중에 serve 가 NPU 에 올립니다.

---

## 4. 계층 정리
- **Python(py-spy = pdb 계층)**: 드라이버 결정적 스택(2.1) + 워커1 트레이싱 래더(3.2) + 워커2 컴파일 경계(3.3). 네이티브 경계 leaf = `ray.get`(드라이버→워커) 와 `compile()`(Python→네이티브). 전수 텍스트는 gdb 콜트리로 이어집니다.
- **네이티브(gdb, `native_llm_common.so` 스트립)**: 함수명 복원 불가 → region 클러스터(컴파일 패스) + 콜래더 + Python 경계로 간이 명명. 전수: [`full-callgraphs/gdb_build_compile_1.*`](full-callgraphs/)(244 스레드, 145 아키타입, 711 함수) + 간이이름표/네임드 콜트리.
- **커널**: 해당 없음(빌드는 NPU 미접촉).

## 5. 증거 색인

| 주장 | 증거 파일 (`../02-dynamic/logs/` 또는 `full-callgraphs/`) |
|---|---|
| 토폴로지·버킷·해시·단계·실패원인(O1089 no tactic) | `build.log` |
| 드라이버 결정적 콜스택(→ ray.get 경계) | `pyspy_build_driver.txt` · `gdb_build_driver.*` |
| 워커1 트레이싱 래더(dynamo/make_fx/sharding) | `pyspy_build_trace.txt` · `pyspy_build_trace.svg` · `gdb_build_trace_worker.*` |
| 워커2 컴파일 경계(compile_task→compile@913) | `pyspy_build_compile.txt` |
| 워커2 네이티브 컴파일러 프레임(5008개) + `??` 명명 | `gdb_build_compile_1..5.txt` · `gdb_build.native_names.md` · `gdb_build.calltree.named.txt` |
| 빌드가 NPU 미접촉 | `build_fds.txt`(rngd fd 0) · `kernel_trace_build.log`(open/DMA 0, doorbell=smi 노이즈) |
| 프로세스 트리(드라이버+Ray 액터들) | `build_pstree.txt` |

## 6. 재현
```bash
# 정적 콜그래프(멀티에이전트 추출)는 ../01-static/build-static-callgraph.md
# 동적 캡처 (root 필요):
cd ../02-dynamic/scripts
BUILD_MODEL=<HF경로> bash build_run_A.sh     # bpftrace + build(RUST_LOG) + gdb(드라이버·워커 트레이싱/컴파일) + py-spy(워커)
# 풀콜그래프/명명:
python gdb_full_callgraph.py ../../03-synthesis/full-callgraphs/gdb_build_compile_1.txt   # (드라이버/워커도 동일)
python name_native_build.py                  # native_llm_common.so ?? -> 간이 이름
cd ../../03-synthesis && dot -Tsvg GB1-build-fullstack.dot -o GB1-build-fullstack.svg
```
> 참고: 위 실측은 시간 절약을 위해 진행 중이던 빌드에 bpftrace/gdb/py-spy 를 붙여 잡았습니다(스크립트는 처음부터 붙이도록 작성). viztracer 기반 드라이버 전수 트레이스(`build_run_B.sh`)는 빌드가 40분+ 걸리고 실패로 끝나며, 드라이버 결정적 스택은 이미 py-spy 로 확보돼 생략했습니다.

## 7. 한계 / 캐비엇
1. **이 (모델,tp) 조합은 컴파일 실패** — `O1089 (no tactic)` at stage_0. 1.5B/tp4 가 네이티브 lowerer 에서 막힌 것으로, **계측 때문이 아닙니다**(gdb attach 는 프로세스를 잠깐 멈출 뿐 컴파일러 판정 로직을 바꾸지 않음; "no tactic"은 순수 컴파일러 결정). 콜그래프(드라이버→Ray→트레이싱→컴파일→네이티브)는 실패와 무관하게 전부 캡처됨.
2. **네이티브 함수단위 그래프 불가** — `native_llm_common.so` 스트립. region(패스) 단위까지가 한계. (디버그 심볼 빌드가 있으면 함수단위 복원 가능.)
3. **워커 Python 은 py-spy(샘플링)** — viztracer 가 Ray 워커(별도 프로세스)를 못 따라가서. 트레이싱이 같은 스택을 오래 반복하므로 dump 다수 + flamegraph 로 충분히 덮였으나, serve 의 viztracer 전수(결정론)와 달리 샘플링입니다.
4. **모니터 노이즈 귀속** — 커널 윈도의 doorbell/mgmt 는 전부 상주 furiosa-smi(`comm=tokio-runtime-w`). 빌드 귀속은 fd 스캔(0)과 comm 분리로 배제.
