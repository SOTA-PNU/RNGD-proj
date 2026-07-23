# 전수 텍스트 콜그래프 — gdb (네이티브) · pdb/viztracer (Python)

특정 interest 필터 없이 **관측된 모든 함수**에 대한 콜그래프를 텍스트로 펼친 것.

## gdb — 네이티브 콜그래프 (스냅샷 3개)

`sudo gdb -p <serve_pid> -batch "thread apply all bt"` 의 전수 파싱. **모든 스레드의 모든 프레임** 포함.
스트립된 furiosa 코드 프레임은 `native_runtime.so!0xADDR` 로 표기(스냅샷 내에서 동일 주소=동일 함수로 병합). tokio/std/parking_lot/rayon/OpenBLAS/libuv 등 런타임 프레임은 심볼명 그대로.

| 스냅샷 | 시점 |
|---|---|
| `gdb_load_10s.*` | 로드 ~10초 (가중치 적재/엔진 초기화 중, 261 스레드) |
| `gdb_idle.*` | 로드 완료·요청 전 (steady-state, 681 스레드) |
| `gdb_infer.*` | 추론 요청 처리 중 (697 스레드) |

각 스냅샷당 3가지 뷰:
- `*.archetypes.txt` — **서로 다른 전체 스레드 스택**(outermost→innermost) 전부 + 개수 + 스레드명/LWP 예시. (idle 30종, infer 29종, load 5종)
- `*.calltree.txt` — 모든 스레드를 **하나의 콜트리로 병합**(root=outermost, `Nx`=해당 노드를 지나는 스레드 수).
- `*.adjacency.txt` — **완전한 caller→callee 인접 리스트**(호출 횟수 포함) + 리프 함수 목록. (스냅샷당 60~101개 함수)

해석 요점: 추론 시점에도 호스트 스레드 대부분은 NPU 완료를 `syscall`/`epoll_wait`/`futex`로 대기(연산은 NPU에서). `scheduler-eager` 스레드는 `native_runtime.so` 안 6~7프레임 깊이로 진입 후 `syscall` 대기, tokio 워커는 `...worker::run → park → condvar/epoll`.

### `??` 네이티브 프레임 간이 이름 (provisional naming)
`native_runtime.so` 스트립으로 함수명이 없어 `native_runtime.so!0xADDR` 로만 보이는 프레임에 **추론 이름**을 부여:
- `gdb_infer.native_names.md` — 주소→간이 이름 매핑표(38개) + 근거(주소 클러스터, 콜래더 위치, 말단 syscall/epoll, 스레드명/RUST_LOG, 3 스냅샷 존재 여부).
- `gdb_infer.calltree.named.txt` — 이름이 적용된 콜트리(예: `furiosa.thread_entry → sched.eager.run → … → sched.park.dispatch → sched.wait.SYSCALL → syscall`).
- 3 스냅샷(load/idle/infer)이 **동일 프로세스**라 주소=동일 함수. 식별된 3개 스레드-풀: `core`(eager 스케줄러 파킹 사다리 + epoll reactor), `iodrv`(io-driver 8스레드→epoll), `worker`(4+1스레드→syscall). **이름은 추론치이며 심볼화된 사실이 아님.**

## pdb / viztracer — Python 콜그래프 (전수, 결정론적)

`load_and_infer.py`(= `LLM(artifact)` 로드 + `llm.generate()` 추론)를 viztracer로 **전수 추적**(11.27M 이벤트)한 뒤 caller→callee 엣지(41,331개, 함수 17,670개)로부터 생성. pdb와 동일한 인터프리터 트레이싱 계층의 **모든 Python 함수**.

- `viz_full_calltree.txt` (6.1MB) — 루트 `builtins.exec` 에서 DFS로 펼친 **전체 콜트리**. 각 함수는 최초 방문 시 완전 전개, 이후 재등장은 `(^ expanded above)` 로 표기. 화살표 뒤 `xN`=해당 엣지 호출 횟수.
- `viz_full_adjacency.txt` (4.4MB) — **모든 함수 → 모든 callee**(호출 횟수). 함수별 총 out-call 수 내림차순.
- `viz_full_reverse.txt` (5.2MB) — 역방향(callee ← caller). "이 함수는 누가 부르나" 조회용.

네이티브 경계: `NativeLLMEngine.*`, `NextGenArtifact.*`, `tokenizers.Tokenizer.encode_batch`, `pydantic_core.SchemaValidator.*` 등은 leaf(그 아래 Python 프레임 없음) → 여기서 Rust(`native_runtime.so`)로 진입하며, 그 이후는 위 gdb 콜그래프로 이어짐.

## 두 계층의 연결
Python 콜트리의 네이티브-경계 leaf  ⟹  gdb 콜트리의 `native_runtime.so!0xADDR` 체인  ⟹  커널(`02-dynamic/logs/kernel_trace*.log`, `renegade_pdma`/doorbell/ioctl). 전체 통합 그래프는 상위 `../CALLGRAPH-WALKTHROUGH.md` 의 G1/G2.

## build — `furiosa-llm build` 전수 콜그래프 (gdb · py-spy)

빌드는 멀티 프로세스(드라이버 + Ray 워커)라 프로세스별로 gdb 를 떴습니다. 전수 텍스트:

| 스냅샷 | 시점 / 프로세스 |
|---|---|
| `gdb_build_driver.*` | 드라이버(120 스레드) — `build_pipeline → ray.get` 에서 파킹(Ray RPC poll/epoll 지배) |
| `gdb_build_trace_worker.*` | `ray::LocalPipelineGenerationActor` 트레이싱 중(torch dynamo, 순수 Python) |
| `gdb_build_compile_1.*` | `ray::TaskCompileActor` 컴파일 중(네이티브 `native_llm_common.so` 5008 프레임) |

각 스냅샷당 `.archetypes/.calltree/.adjacency.txt`. 원시 5장: `gdb_build_compile_1..5.txt`(같은 프로세스 LWP 3105741).

### `??` 네이티브 간이 명명 (build)
`native_llm_common.cpython-312*.so`(143 MB, 스트립)가 `furiosa.native_common.compiler` 의 실체. 함수명이 없어 `native_llm_common.so!0xADDR` 로만 보이는 컴파일러 프레임에 **추론 이름**을 부여:
- `gdb_build.native_names.md` — 802 주소를 **region(=컴파일 패스)** 6개 + hot 12개로 명명(스레드 아키타입·콜래더·Python 경계 근거). `compile.driver.enter`/`compile.wait.SYSCALL`/`pool.worker.park`(5장 모두) vs `lower.visit.recurse.*`/`lower.tactic.leaf`(일시).
- `gdb_build.calltree.named.txt` — 이름 적용 콜트리(재귀 operator visitor 루프가 보임). **이름은 추론치이며 심볼화된 사실이 아님.**

Python 측(py-spy)은 `../../02-dynamic/logs/pyspy_build_{driver,trace,compile}.txt`(드라이버 결정적 스택 + 워커 트레이싱/컴파일 경계). 빌드는 NPU 미접촉이라 커널 계층은 없습니다(대조: serve 의 doorbell/DMA).

## 재생성
gdb: `python ../../02-dynamic/scripts/gdb_full_callgraph.py <gdb_*.txt>`
viztracer: `python ../../02-dynamic/scripts/viz_full_callgraph.py ../../02-dynamic/logs/viz_load_infer_edges.tsv .`
(원시 viztracer JSON은 `../../02-dynamic/scripts/run_B.sh` 로 재생성)
build 명명: `python ../../02-dynamic/scripts/name_native_build.py`
