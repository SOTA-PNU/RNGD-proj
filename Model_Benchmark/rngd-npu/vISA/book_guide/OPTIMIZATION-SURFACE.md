# 최적화·커스터마이즈 지도 (Optimization Surface)

> **1줄 요약**: RNGD 스택에서 "내가 직접 손댈 수 있는 최적화 지점"을 서빙설정(L1) → TCL커널저작(L2) → 벤더TCL커널편집(L3) → vISA손코딩(L4) → 폐쇄영역(L5) 5계층 25개 지점으로 정리하고, 각 지점의 **파일경로·바꿀것·기대효과·측정법·난이도·상태**를 표로 못박는다. 시간 낭비를 막기 위해 `blocked` 는 왜 막혔는지(폐쇄소스/API부재/버전스큐)까지 밝힌다.

이 문서는 책 해설이 아니라 **별도 산출물**이다. 책과의 느슨한 대응은 다음과 같다.

**느슨하게 대응하는 책 섹션**
- `computing-tensors/*` → L2/L4 (DPE·VE 엔진 선택, TacticKind)
- `moving-tensors/*` → L3 (집합통신 DMA 엔진, tcl.Dram 사전 분할), 계측(DmaStore 지배 형상)
- `mapping-tensors/*` → L4 (`m![]` 매핑 표현식 Chip/Cluster/Slice/Lane/Packet)
- `getting-started/*`, `command-line/*` → 계측 절(`cargo furiosa-opt compile --dump-*`)

**데이터 출처**: `/home/jun/.claude/jobs/46bc5c7e/tmp/visa_book_guide/_data_optmap.json`(customizationMap 25지점 + nextAction + biggestTrap) 및 `_GROUND_TRUTH.md`. 수치·경로는 원본 그대로 옮기되, 이 세션에서 `ls`/`grep`으로 존재를 확인한 것과 미확인인 것을 구분 표기했다.

---

## 목차
- [0. 가장 먼저 할 일 (2분 보호 조치)](#0-가장-먼저-할-일-2분-보호-조치)
- [계층·상태 범례](#계층상태-범례)
- [계측 도구: NPU 안 잡고 사이클 귀속 보기](#계측-도구-npu-안-잡고-사이클-귀속-보기)
- [L1 — 서빙 설정](#l1--서빙-설정)
- [L2 — TCL 커널 저작](#l2--tcl-커널-저작)
- [L3 — 벤더 TCL 커널 편집](#l3--벤더-tcl-커널-편집)
- [L4 — vISA 손코딩](#l4--visa-손코딩)
- [L5 — 폐쇄 영역](#l5--폐쇄-영역)
- [nextAction 요약](#nextaction-요약)
- [⚠ biggestTrap — 마지막 경고](#-biggesttrap--마지막-경고)

---

## 0. 가장 먼저 할 일 (2분 보호 조치)

**어떤 최적화보다 먼저** 아래 두 파일을 git 추적 트리로 대피시켜라. `pip install` 한 번이면 영구 소실된다. 이미 2026.3.0 업그레이드가 `presets.py`의 사용자 등록 8건(qwen3_next 포함)을 지웠다(근거: `_data_optmap.json` nextAction).

| # | 파일 | 크기/상태 | 왜 위험한가 |
|---|------|-----------|-------------|
| 1 | `/home/jun/furiosa/lib/python3.12/site-packages/furiosa/models/language/architecture/qwen3_next.py` | **37,499 B (실측 확인)**, exec 비트 있음 | 사용자 직접 작성. `furiosa_models` RECORD 에 없음 = pip 미관리 = 재설치 시 삭제 대상 |
| 2 | `/home/jun/furiosa/lib/python3.12/site-packages/furiosa_llm/artifact/presets.py.bak-20260616` | **실측 확인** | 사용자 프리셋 백업. 이미 원본 `presets.py` 는 업그레이드로 사용자 등록이 지워짐 |

```bash
# /home/jun/RNGD-proj 는 git repo. 거기로 복사·커밋(읽기 전용 원칙 — 벤더 트리는 안 건드림).
mkdir -p /home/jun/RNGD-proj/_rescue
cp /home/jun/furiosa/lib/python3.12/site-packages/furiosa/models/language/architecture/qwen3_next.py \
   /home/jun/RNGD-proj/_rescue/
cp /home/jun/furiosa/lib/python3.12/site-packages/furiosa_llm/artifact/presets.py.bak-20260616 \
   /home/jun/RNGD-proj/_rescue/
cd /home/jun/RNGD-proj && git add _rescue && git commit -m "rescue: qwen3_next.py + presets bak (pip-unmanaged)"
```

> **경로 정정(중요)**: 작업 지시서에 적힌 `/home/jun/lib/python3.12/site-packages/furiosa/furiosa_llm/artifact/presets.py.bak-20260616` 는 **존재하지 않는다**(이 세션 `ls` 확인). 실제 경로는 위 표의 `/home/jun/furiosa/lib/.../furiosa_llm/artifact/presets.py.bak-20260616`. `find` 로 재확인함.

---

## 계층·상태 범례

**계층(L)**: 위로 갈수록 쉽고 재현 가능, 아래로 갈수록 강력하지만 위험/봉쇄가 커진다.
- **L1 서빙 설정** — 재컴파일 없이(또는 아티팩트 재빌드로) 런타임 노브. 가장 안전.
- **L2 TCL 커널 저작** — 내가 새 TCL 커널을 짜서 EDF까지 뽑는 경로. furiosa/tcl 은 순수 파이썬이라 저작·컴파일은 지금 가능.
- **L3 벤더 TCL 커널 편집** — 이미 배포된 벤더 커널(gpt_oss/qwen3_moe/exaone_moe…)의 설정·플래그를 바꾸는 것.
- **L4 vISA 손코딩** — `/home/jun/yik` 처럼 `m![]` 매핑까지 손으로. 완전 제어. 실칩 실행은 CHIP0 고정 + npu0 유휴 + 커널 게이팅이 전제(#20, 2026-07-24 open).
- **L5 폐쇄 영역** — 컴파일러 바이너리 내부. 수정 불가. 시간 쓰지 말 것.

**상태**
- `open` — 지금 착수 가능, 오라클/측정 경로가 있음. **(11지점)**
- `partial` — 일부만 가능하거나 미검증 전제가 남음. **(6지점)**
- `blocked` — 폐쇄소스/API부재/버전스큐로 막힘. **(8지점)**

> **2026-07-24 변경**: #20(vISA 실칩 실행)이 **blocked → open** 으로 바뀌었다.
> 당시 근거였던 "npu0 점유"가 해소됐고, 이 세션에 실기 테스트 89개를 실제로 실행했다(#20 행 참조).

---

## 계측 도구: NPU 안 잡고 사이클 귀속 보기

이 스택에서 **NPU 점유 없이** 성능을 들여다볼 수 있는 유일한 창구가 `cargo furiosa-opt compile --dump-*` 다. `furiosa-tcc`(TCL→EDF 컴파일러)는 덤프/프로파일 옵션이 **전무**하다(`_GROUND_TRUTH.md:41-42`).

**이 세션 실측**(`cargo furiosa-opt compile --help`, exit 0, 카드 미점유):
```
--dump-visa <FILE>     중간 vISA 를 파일로
--dump-ir   <FILE>     중간 LIR 을 파일로
--dump-schedule <FILE> 스케줄을 JSON 으로   ← 소스라인 귀속 사이클
--dump-graph <FILE>    IR 그래프를 JSON 으로
--dump-summary <DIR>   컴파일 요약을 디렉터리로
```
> 출처: 이 세션 `cargo furiosa-opt compile --help` 실행 결과. 백엔드는 `[typecheck, emulation, npu]`, default `emulation`(`_GROUND_TRUTH.md:7-8`) — 덤프에 `npu` 불필요.

**핵심 효용 — `--dump-schedule` 은 사이클을 소스라인에 귀속시킨다.**
이 세션에서 확인된 계측 예(작업 지시 기준, 특정 커널):
- 총 **82,449 cycle** 중 **`DmaStore` 88%**, 실제 축약 **`Main` 0.94%**.
- 해석: 이 커널은 **연산이 아니라 I/O(결과 write-back)가 지배**하는 형상. 여기서 DPE/VE 태틱을 아무리 바꿔도(L2/L3) 총 지연은 거의 안 움직인다. 먼저 DMA/레이아웃(L3의 `dma_preference`, `tcl.Dram` 분할)을 손대야 한다.

### ★ 위 예는 재현·확장됐다 — 개별 커널이 아니라 스택 전반의 특성이다 (2026-07-24)

당초 이 수치는 "재현 컴파일 미실시(미확인)"로 남겨뒀다. 이후 실기 컴파일에 성공한 **커널 130개**의
스케줄을 덤프해 합산했고, 결론이 개별 커널 사례가 아니라 **전반적 특성**임이 확인됐다.

| 엔진 | 총 사이클 | 비중 | 인스트럭션 |
|---|--:|--:|--:|
| **DmaEngine** | **75,464,336** | **96.5%** | 470 |
| PeCore | 2,586,167 | 3.3% | 1,557 |
| MainContext | 58,883 | 0.1% | 108 |
| InterChipTransfer | 38,018 | 0.0% | 2 |
| VectorEngine | 14,770 | 0.0% | 50 |
| SubContext | 9,737 | 0.0% | 27 |

- **인스트럭션 수는 PeCore 가 1,557개로 최다인데 사이클은 3.3%.** DmaEngine 은 470개로 96.5% —
  DMA 인스트럭션 하나가 연산 인스트럭션 하나보다 두 자릿수 이상 비싸다.
- 합산의 착시가 아니다: **커널 130개 중 107개(82%)가 DMA 에 50% 이상**, 54개는 90% 이상, **중앙값 82.8%**.
- 사이클 스팬: min 16 / p25 4,612 / **median 10,532** / p75 23,503 / max 10,845,036.
- 검증: `mnist::forward` 를 17,953 cycle / 22 instruction 으로 재현해
  [11-MNIST-실행결과](./11-MNIST-실행결과.md) 의 독립 기록과 정확히 일치함을 확인했다.
- 이 값들은 **컴파일러의 스케줄 모델 예측**이며 실측 벽시계가 아니다(모델 신뢰 근거는 위 검증).

> ### ★ 이 문서의 우선순위에 대한 함의
> **L4(vISA 손코딩)가 배타적으로 노출하는 슬라이스 내부 데이터패스(Lane/Packet/TRF/DPE/VE)의
> 최적화 상한은 3.3%다**(PeCore 총 점유, 스케줄 모델 기준). 사이클의 96.5% 는 슬라이스 바깥의 데이터 이동에 있다.
> → **"연산을 더 잘 짜는" 방향보다 DMA 전송량·레이아웃·정렬이 먼저다.**
> 같은 결론이 반대편에서도 나온다: 실기 컴파일에 **실패**하는 커널 63개에서 반복해 나오는 사유도 정렬이다
> (`not aligned by 8`, `tail_size % min_align`, `incorrect buffer size`) —
> *도는 커널은 DMA 에 사이클을 쓰고, 안 도는 커널은 DMA 정렬에 막힌다.*
> 상세: [13-NPU-실기-매트릭스](./13-NPU-실기-매트릭스.md) §4, §7.

### 두 번째 창구 — `--dump-summary <DIR>` (이 세션 실측)

`--dump-summary` 는 디렉터리에 **파일 20개**를 쏟는다. `summary.log`(사람이 읽는 사이클 요약) ·
`summary.json`(같은 내용 + 스케줄러 내부 통계) · `target_plot.json`·`lir_plot.json`·`rlir_plot.json` ·
`fir/lir.fir` · `populator/populator_log.json` · `sfr/*.sfr.yaml`(8개) · `dot/{edf,lir}.dot` ·
`json/lir_ir_viewer.json` · `text_form/{lir.fir.txt,lir.desc.json}`.

**사이클 분해가 `--dump-schedule` 보다 정밀하다.** `mnist::forward` 실측:

| 항목 | 값 |
|---|--:|
| `total_instruction_cycle` (main+sub+io) | **23,337** |
| `total_execution_cycle` (main+sub+io−hidden) | **17,953** |
| `io` / `io_only` | 68.874% (12,365) / **38.311% (6,878)** |
| `main` / `main_only` | 42.790% (7,682) / 12.226% (2,195) |
| `sub` / `sub_only` | 4.400% (790) / 4.400% (790) |

- 인스트럭션 사이클 합 23,337 과 실행 사이클 17,953 의 차이 **5,384 가 hidden**(겹쳐 감춰진 부분)이다.
  `--dump-schedule` 의 엔진별 점유 합이 100% 를 넘는 이유가 여기서 설명된다.
- **어디를 최적화할지 정할 때 정직한 숫자는 `*_only`(겹치지 않은 순수 점유)다.** io 총 점유는 68.874% 지만
  순수 io 는 **38.311%** 다. "DMA 가 지배한다"를 엄밀히 말하려면 이 값을 쓴다.
- 위 값들도 **컴파일러 산출물**(스케줄 모델 예측)이며 실기 실행 측정이 아니다. NPU 를 점유하지 않는다.
- `json/lir_ir_viewer.json` 이 **L2 #8 이 diff 하라고 지목한 그 `*_ir_viewer.json`** 이다 —
  얻는 방법이 곧 이 옵션이다.
- `sfr/*.sfr.yaml` 은 **명령별 SFR(Special Function Register) 설정**을 노출한다(이 커널 8개,
  `RenegadeCommand#O3 …#O18`). 스케줄보다 한 단 아래인 레지스터 수준 정보다.
- `summary.json` 에 `operator_schedule_heuristic`·`beam_size`·`total_states_visited` 키가 **존재하지만
  이 커널에서는 전부 `None`** 이다(`schedule_method` 는 빈 문자열). 다른 커널·설정에서 값이 채워지는지는
  **미확인** — 키가 있다는 사실만 실측이다.

**오염 방지 규칙**: 반드시 `CARGO_TARGET_DIR=/home/jun/.claude/jobs/46bc5c7e/tmp/...` 로 지정해 사용자 `target/` 을 오염시키지 말 것.
```bash
cd /home/jun/yik && \
CARGO_TARGET_DIR=/home/jun/.claude/jobs/46bc5c7e/tmp/cargo_probe \
  cargo furiosa-opt compile --backend emulation --dump-schedule /tmp/sched.json <kernel>
```
> 단, `/home/jun/yik` 은 `furiosa-opt-std="0.3"` 핀 + `src/furiosa-opt.tag` 라 현재 0.4 래퍼로 `--backend npu` 는 "no kernel packages found"로 거부됨(`_GROUND_TRUTH.md:14-16`). `typecheck`/`emulation` 덤프는 동작.

---

## L1 — 서빙 설정

가장 안전한 계층. 재컴파일 없는 런타임 노브가 대부분이나, `compiler_config_overrides` 만은 아티팩트 재빌드가 필요하다.

| # | 파일·지점 | 무엇을 바꾸나 | 기대효과 | 측정법 | 난이도 | 상태 |
|---|-----------|---------------|----------|--------|--------|------|
| 1 | `rngd-npu/run_all/runners/memory_sweep.py:67-68` → `furiosa-llm serve --max-num-batched-tokens` (4096/16384) | 프리필 청크 크기. 서버 로그에 `max_num_batched_tokens: Some(4096)` 로 실제 등록되는 **유일하게 확인된 스케줄러 플래그** | TTFT↔처리량 트레이드오프 이동. 측정 8조합 편차 baseline 2251.05 대비 ±3.4%(재현성 밴드 안) | memsweep 재실행 후 aggregate TPS·TTFT p50/p95 비교. **먼저 서버 로그 SchedulerConfig 값 변경을 어서션** | hours (조합당 ~49s, 8조합 ~7분, 카드 1장) | **open** |
| 2 | `furiosa-llm serve --max-batch-size` (memsweep_000 vs 004 대조) | **지금은 바꾸지 말 것.** 두 로그 설정 차이 0(동일 Resolve 47 pipeline / max_executable_len=131072 / SchedulerConfig{max_concurrency:None}) | 관측상 효과 없음. 플래그 적용 여부 자체가 미확인(2251→2232 = -0.8%, 노이즈) | 재실행 시 파이프라인 해석·SchedulerConfig·/v1/models 한계에 변화 있는지 어서션. 없으면 "적용 확인 불가"로 보고 | hours | **blocked** (플래그 무흔적 = 적용 검증 불가) |
| 3 | `/usr/bin/furiosa-smi governor -p performance\|powersave -i 2,3` (root 필요) | 주파수 프로파일 최고 고정/최저 고정 | **재컴파일 없이 즉시 적용되는 유일한 실질 DVFS 레버.** 저동시성=토큰당 에너지↑, 고동시성=처리량 손실 | `bench-blog/loadgen.py`+`power.py` 배치 스윕. `furiosa-smi` 에 power 하위명령 없어 `power.py:54-57` 이 info 텍스트를 1Hz·1W 스크레이핑. 유휴 39~40W/카드 감산 | days (거버너2 × 배치6, npu2/npu3 로 6~10 NPU시간) | **open** |
| 4 | `furiosa_llm/artifact/types/config.py:114` (`compiler_config_overrides`) → `builder.py:295`; 병합은 `parallelize/compiler_config.py:177-178` 이 **마지막 = 항상 우선** | ~96개 컴파일러 Config 키를 **빌드 시점** 덮어쓰기 (tactic_hint / dma_preference / lowering_mode / use_attention_kernel …) | 스케줄링 전반 변경 | 키당 아티팩트 재빌드 후 서빙 벤치. **런타임 플래그 아님** — `/home/jun/.cache/furiosa/llm` 캐시 85GB, 지점당 전체 재빌드 비용 | days~weeks | **partial** (키당 재빌드 비용이 탐색을 제약) |
| 5 | `rngd-npu/tp32/converter_tp32_broadcast.patch`, `build_tp32.py` (`FURIOSA_TP32_BCAST=ALL`) | 4카드 단일 인스턴스(tp32) 빌드 — 모든 가중치 칩마다 복제 | 자체 빌드로는 "한 칩에 들어가는 크기"(예: 7B bf16)만 가능. 진짜 tp32 는 벤더 프리빌트에만 | `info/README_all_change.md:89` 에 "tp32(MoE bf16) stage_0 임베딩 컴파일 실패 → 4장 단일 인스턴스 빌드 불가" 기록 | weeks | **blocked** (사용자 빌드로는 대형 tp32 불가) |

**L1 초심자 설명**: L1 은 "모델은 그대로 두고 서버 옵션만 돌리는" 층이다. #1 은 확실히 먹는 노브, #2 는 "먹는지조차 모르는" 노브(그래서 blocked), #3 은 전기·주파수 레버, #4 는 컴파일러에게 주는 힌트 뭉치인데 **빌드 때만** 반영된다는 함정, #5 는 "4장을 한 덩어리로" 쓰려는 시도인데 큰 모델은 막힌다.

---

## L2 — TCL 커널 저작

> **이 계층의 두 열쇠**(지시서 강조):
> 1. **7종 TacticKind** — 엔진 스케줄을 강제 지정하는 손잡이.
> 2. **EdfModule 실행 경로** — `TclModule` 부재를 우회하는 미탐색 실행로(30분 실험).

### L2 열쇠 A — 7종 TacticKind (이 세션 실측 확인)
`/home/jun/furiosa/lib/python3.12/site-packages/furiosa/tcl/_primitive/_tactic_kind.py:7-13` (파일 실측):
```python
class TacticKind(Enum):          # Rust TacticKind enum 과 일치
    EinsumByDpe = auto()   # 축약을 DPE(Contraction 엔진)로
    ReduceByVe  = auto()   # 리덕션을 VE(Vector 엔진)로
    Interleaving = auto()
    Elementwise = auto()
    TensorOperation = auto()
    EinsumByVe  = auto()   # 축약을 VE 로 (DPE 대신)
    FilterCompaction = auto()
```
`@tcl.tensor_operation(tactic_kind=...)` 로 강제 지정한다(`_tensor_operation.py:2428-2459`). **파이썬 측 검증이 없어** 동일 본체로 7종 전부 방출 가능 — 탐색 공간이 사실상 미개척(배포 커널 전체 override 사용처 7곳뿐).

### L2 열쇠 B — EdfModule 실행 경로 (이 세션 실측 확인)
`TclModule` 은 `furiosa/torch` 어디에도 없다(이 세션 `grep -rn TclModule` = 0건). 하지만 `EdfModule` 은 있다:
- `furiosa/torch/custom_ops/edf.py:131` → `class EdfModule(torch.nn.Module)` (실측 확인)
- `furiosa/torch/__init__.py:37-46` `__all__` 에 `"EdfModule"` 노출됨 (실측 확인; 같은 리스트에 `TacticKernelModule` 은 있으나 `TclModule` 없음)

**미탐색 우회로**(`_data_optmap.json` L2 #11, status `open`):
```text
@tcl.kernel 저작 → furiosa-tcc 로 .edf 생성 → ir.Edf.deserialize → EdfModule(edf).to("rngd:N")
```
- 성공하면 **별도 venv 없이** 사용자 TCL 커널을 NPU 실행 가능 = L2 봉쇄 해제.
- 확인할 것 하나: tcc 2026.3.0 산출 EDF가 2026.2.0 `ir.Edf` 로 역직렬화되는지. 디바이스 문자열 `rngd:N`, `import torch` 다음 `import furiosa.torch` 순서 필수(`rngd-npu/run_edf.py:27-28`).
- **effort: hours (30분 내외 결정적 실험, 미실시)**. `ir.Edf.deserialize` 경로 자체는 이 세션 미검증(미확인).

### L2 지점표

| # | 파일·지점 | 무엇을 바꾸나 | 기대효과 | 측정법 | 난이도 | 상태 |
|---|-----------|---------------|----------|--------|--------|------|
| 6 | `tcl/_tensor_operation.py:2428-2459` `@tcl.tensor_operation(tactic_kind=...)` | 7 TacticKind 중 강제 지정 (검증 없음) | HW 엔진 스케줄 선택 변경. 벤더 in-source 근거: "태틱 없으면 flash attention 성능 소폭 저하"(`kernels/common/math.py:63-65`), "자동 추론이 컴파일 불가 모델 유발"(`_tensor_operation.py:2418-2419`) | 동일 본체 × 7태틱 × chip count 컴파일 성공/실패·지연 | days(저작·컴파일)/weeks(NPU실행) | **partial** |
| 7 | `tcl/_lang/_context.py:79-127` `tcl.context(layout={Chip: axis})` | 연산자별 칩 축 배치 (파생 `tcl.Axis`→`reshape` 인수분해→외축을 Chip 지명, `gpt_oss/optimized/full_attention_bf16.py` 참조) | 4칩 텐서병렬의 핵심. 축 선택이 리덕션 축이 칩 내부에 남는지=집합통신 종류를 결정 | 칩 수별 지연·처리량. **호출 키워드는 반드시 `compiler_hints=`** — docstring(`_context.py:232-236`)의 `context=` 는 TypeError | hours per kernel | **partial** |
| 8 | `tcl/_lang/_context.py:221-248` `heuristic_hint={...}` (23키: `avoid_dpe_bottleneck`, `utilize_dpe_feed_buffer`, `utilize_dpe_accumulator`, `io_bound_mode`, `min_total_util`, `force_trf_half_mode` …) | 연산자별 스케줄 휴리스틱 (키 검증 없음) | 스케줄 휴리스틱 변경. 단 배포 커널 사용처 0건 → 참조값 전무, 값 타입/범위 미검증 | 힌트 하나만 뒤집고 `.fir`/`*_ir_viewer.json` 덤프·`Dfg.to_dot()` diff — 그 `*_ir_viewer.json` 은 `--dump-summary <DIR>` 의 `json/lir_ir_viewer.json` 이다(계측 절 참조). 오타 시 Rust serde 파싱 오류 = 저렴한 타입 프로브 | weeks(키당 컴파일1, 오라클 없음, 다수 null 가능) | **open** |
| 9 | `tcl/kernel.py:82-141` `Kernel.from_edsl(...).dsl → .tc → furiosa-tcc` (`/home/jun/furiosa/bin/furiosa-tcc`, 145MB 실측) | `@tcl.kernel` 저작 → DSL 방출 → `furiosa-tcc --target-npu renegade-8pe-4chip --output x.edf`. 유효 타깃: renegade / -2pe/-4pe/-8pe/-8pe-2chip/-8pe-4chip/renegade-s | 자동 커널화기 우회 직접 EDF 생산. furiosa/tcl 순수 파이썬 = furiosa-torch 의존 없어 저작·컴파일 즉시 가능 | 주의1: `from_edsl` 은 `#tactic_kernel_dsl` 헤더 미방출 → 직접 앞에 붙여야("No parse header found" 방지). 주의2: 생성 EDF 가 아티팩트 a6 CompiledGraph 컨테이너와 호환되는지 미검증 | hours(저작·컴파일), 실행 별개 | **partial** |
| 10 | `tcl/torch/__init__.py:37-46` — `TclModule` 부재 / 버전 스큐 | **아무것도 in-place 업그레이드하지 말 것.** `furiosa_models 2026.2.0` 이 `furiosa-torch==2026.2.0` 하드핀, `furiosa_llm 2026.3.0` 이 `furiosa-models ~=2026.2.0` 요구 → 현 스큐는 **벤더 의도 구성** | TCL 커널을 NPU 실행하는 문서화 경로(`TclModule`)가 설치본에 없음. `furiosa/kernels/qwen3/naive/*.py` 3종도 TclModule 참조 = 벤더 커널 실행 경로 전체 봉쇄 | 업그레이드 강행 시 `qwen3_next.py`(37,499B, RECORD 부재) 영구 소실 → **§0 대피 선행 필수**, 실험은 별도 venv 에서만 | days(별도 venv) | **blocked** (버전 스큐 = 벤더 의도) |
| 11 | `tcl/torch/custom_ops/edf.py:131-141, 260, 430` — `EdfModule` | TclModule 우회: `.edf → ir.Edf.deserialize → EdfModule(edf).to('rngd:N')`. EdfModule 은 2026.2.0 에 이미 존재(실측: `edf.py:131`) | 성공 시 별도 venv 없이 사용자 TCL 커널 NPU 실행 → **L2 전체 봉쇄 해제** | tcc 2026.3.0 EDF 가 2026.2.0 ir.Edf 로 역직렬화되는지만 확인. `rngd:N`, import 순서(`run_edf.py:27-28`) | **hours — 30분 결정적 실험, 미실시** | **open** (최우선 저비용 실험) |

**L2 함정 요약**: (a) `_context.py` 키워드는 `compiler_hints=` (문서가 안내하는 `context=` 는 TypeError). (b) `from_edsl` 은 파스 헤더를 안 붙여준다. (c) 절대 in-place 업그레이드 금지 — §0 대피 먼저. (d) #10(blocked)과 #11(open)은 한 몸: #11 이 뚫리면 #10 우회.

---

## L3 — 벤더 TCL 커널 편집

이미 배포된 벤더 커널의 **설정 dict·플래그**를 바꾸는 층. 커널 로직을 새로 짜지 않고 룩업테이블/불리언만 뒤집는 저비용 개구부가 많다. 다만 벤더 소스 재배포는 금지(LicenseRef-Proprietary, 라이선스 본문 미설치 — `_data_optmap.json` L3 #18).

| # | 파일·지점 | 무엇을 바꾸나 | 기대효과 | 측정법 | 난이도 | 상태 |
|---|-----------|---------------|----------|--------|--------|------|
| 12 | `kernels/gpt_oss/common/entry_function.py:202-219` (BB/BGi/BG 수기 튜닝 표) | MoE 블록 크기 룩업 dict. `gpt_oss {32:(4,16,128)…}` vs `qwen3_moe {32:(4,32,160)…}` 상이. 벤더 `TODO(OPT-3587)`="자동 유도로 교체" 명시 | MoE 블록 게더 효율 변화. **벤더 스스로 수기 튜닝 인정 = 정당성 최고 개구부** | T별 처리량·지연. 설정만 바꾸므로 커널 코드 수정 불필요 | hours(수정)/days(측정, 재빌드) | **open** |
| 13 | `kernels/gpt_oss/config.py:47-106`, `qwen3_moe/config.py:48-69` | 모델별 컴파일러 설정 dict: `tactic_hint {IOBound:200/ComputeBound:100}`, `dma_preference (IO0.8/compute1.2)`, `lowering_mode ('Optimal' if token_size<8 else 'Heuristic')`, `enable_vrf_half_mode`, `use_attention_kernel` 임계 4종 | 프리필/디코드 스케줄 전략 전환. `use_attention_kernel` 임계는 벤더가 s=32768,i=1024 에서 오히려 퇴행한다고 주석 | 프리필/디코드 분리 측정. 값 공간(정수 100/200 의미, dma_preference 범위) 미문서화 | days | **open** |
| 14 | `furiosa-tcc` Config 키 `separate_vector_ops_from_dpe`, `fuse_mamma_to_single_einsum_by_dpe` (금일 grep: 바이너리에 각 7회, 배포 설정 사용처 0건) | 두 키를 `compiler_config` 로 전달 → ">2 EinsumByDpe 무증상 오컴파일"이 사라지는지 | 사라지면 "설정이 정합성을 좌우" / 안 사라지면 진짜 컴파일러 결함 확정 (어느 쪽이든 4순위 논문 성립, 문구만 달라짐) | `dn_chunk_full_dpe.yaml`(5DPE, maxerr~5.2e-01) 재컴파일 후 HF 기준 maxerr 재측정. `dn_chunk_full.yaml`(0DPE, 8.94e-08)=양성대조 | hours — **4순위 논문 착수 전 필수 선행** | **open** |
| 15 | `kernels/gpt_oss/optimized/entry_function.py:100`, `common/wiring_utils.py:248` (`apply_collective_dq`) | 집합통신 부분합을 MxFp8 block-32 양자화. 완전 배관됐으나 기본 False, 디코드 배선 하드코딩 False | 칩간 인터커넥트 트래픽↓ vs 정확도↓. 벤더 주석: "large tw case 에서만 유의미" | 플래그 1개 전환 후 처리량·출력 정확도(동일 프롬프트 로짓 비교) 동시 측정 | hours(전환)/days(측정) | **open** |
| 16 | `kernels/exaone_moe/k_exaone_w4fa16kv16/optimized.py:578,672 …` (`dma_engine_type=PDMA`) | 집합통신 DMA 엔진. `tcl.all_gather/reduce_scatter` 기본 TDMA. EXAONE·Solar·qwen3-vl-text 는 hot all_gather 를 PDMA override, gpt_oss·qwen3_moe 는 전혀 안 함 | 벤더 코드 내 설명되지 않은 불일치. 키워드 1개 A/B | 동일 커널 PDMA vs TDMA 지연 비교 | hours | **open** |
| 17 | `kernels/{llama,qwen3}/naive/*` (금일 검증: llama Chip=0/collectives=0/tcl.Dram=0; qwen3 Chip=0/collectives=1/tcl.Dram=0) | llama/qwen3 밀집 모델에 gpt_oss/qwen3_moe 플레이북(칩 힌트+집합통신+tcl.Dram 사전분할) 이식 | 단일 칩 코드를 4칩 텐서병렬로. Llama-3.1-8B 고동시성 열세의 직접 원인 후보 | 포팅 전후 동시성 스윕. 단 tcl.Dram 사전분할 대응 가중치 직렬화 writer 가 furiosa/kernels/ 에 없음(미검증) | weeks~months | **partial** |
| 18 | `furiosa/kernels/` naive vs optimized 쌍 (금일: gpt_oss/qwen3_moe/qwen3_vl 3개만 쌍 존재) | naive vs optimized 직접 A/B 측정 | **현재 불가**: (a) 런타임 스위치 부재(furiosa_llm 이 furiosa.kernels 를 파이썬 import 안 함, 선택은 각 __init__.py 하드코딩), (b) gpt_oss naive=20B/optimized=120B 로 동일 계산 아님, (c) 벤더 site-packages 직접 수정 필요 | qwen3_moe 만 전후 쌍에 가까움. LicenseRef-Proprietary — 인용 가능, 재배포 금지 | days~weeks | **blocked** (런타임 스위치 부재+동일계산 아님) |

**L3 초심자 설명**: #12/#13 은 "벤더가 손으로 넣어둔 숫자표"를 바꾸는 것 — 벤더 스스로 `TODO`/주석으로 임시임을 인정해서 **가장 명분 좋은** 지점이다. #15/#16 은 불리언 한 개 A/B(집합통신 양자화·DMA 엔진). #17 은 "밀집 모델을 4칩용으로 다시 배선"하는 큰 공사. #18 은 "naive vs optimized 어느 게 빠른가"를 궁금해하지만 **비교 대상이 애초에 같은 계산이 아니라** 막혔다.

---

## L4 — vISA 손코딩

`/home/jun/yik` 에서 `m![]` 매핑 표현식으로 Chip/Cluster/Slice/Lane/Packet 5계층을 직접 지정하는 층. 완전 제어를 얻는다. 실칩 실행은 **CHIP0 고정 + npu0 유휴 + 패키지 전체 `#[device]` 게이팅**을 전제로 하며, 그 전제가 충족되면 실행된다(2026-07-24 실측, #20).

| # | 파일·지점 | 무엇을 바꾸나 | 기대효과 | 측정법 | 난이도 | 상태 |
|---|-----------|---------------|----------|--------|--------|------|
| 19 | `/home/jun/yik` (furiosa-opt-std 0.3.0 핀, 예제 5종; 백엔드 typecheck/emulation) | 손코딩 vISA 에서 `m![]` 매핑(Chip/Cluster/Slice/Lane/Packet)을 직접. TCL 의 read→dpe.exec→ve.exec→write.to 4단계와 동일한 fetch→collect→contract_*→commit 사슬을 매핑 인자까지 전부 명시 | 완전한 스케줄 제어. 검증된 verbosity: 동일 GEMM 이 TCL 3행/매핑0개 vs vISA 본체 ~20행/`m![]` ~20개(파일 전체 27개) | `typecheck`/`emulation` 은 **NPU 미점유** → 지금 자유 사용. TCL vs vISA 생산성·성능 비교 근거 | days | **open** |
| 20 | `cargo-furiosa-opt --backend npu` (`runtime/npu/ffi.rs:34-35`) | 실칩 실행. `--backend npu` 는 **CHIP 0 만 하드타깃**, 환경변수 우회 없음 (실측 재확인: HAL 오류 메시지가 `ClusterId(npu0pe0-3)` 로 npu0 지목) | 카드 1장 점유 | **2026-07-24 해소.** 실행 시점 `furiosa-smi status` = npu0~npu3 전부 `alive / 0.00 GiB` → **실기 테스트 89개 실제 실행, 80개 통과**. 누수 0. 전제: 라우터 백엔드 모델이 미기동이어야 npu0 이 빈다 | hours | **open** (조건: npu0 유휴 + 게이팅 필요, 아래 주의 참조) |
| 21 | `/home/jun/yik` — `cargo-furiosa-opt` 바이너리 자체 (저장소엔 182B 스텁만; 실물 `~/.cargo/bin/cargo-furiosa-opt` 3,538,576B 실측) | **없음. 폐쇄 프리빌트.** | vISA 컴파일 파이프라인 내부 수정 불가 | 해당 없음 | blocked | **blocked** (폐쇄 바이너리) |

> ### ★ 실기(`--backend npu`) 실행의 제1 전제 — CHIP0 고정보다 이게 먼저 막는다
> **`--backend npu` 는 패키지 안의 *모든* `#[device]` 함수를 빌드 시점에 EDF 로 낮춘다.
> 테스트가 그 함수를 부르든 말든 상관없다. 하나라도 못 낮추면 크레이트 전체가 죽는다.**
> 벤더 예제 크레이트를 그대로 돌리면 **커널 에러 63개로 빌드 실패, 테스트 0개 실행**이다.
> 검증된 우회법은 실패 커널에만 `#[cfg(not(backend = "npu"))]` 를 붙이는 **게이팅**
> (상류가 이미 쓰는 관용구 — `tests/matmul_tests.rs:124`). 이걸로 실기 테스트가 21개 → **89개**로 늘었다.
> 절차·스크립트는 [12-예제-전수실행](./12-예제-전수실행.md) §10, 실기 결과는
> [13-NPU-실기-매트릭스](./13-NPU-실기-매트릭스.md).
>
> 또한 실기 실행에는 다음이 필수다(경험적으로 확정):
> **① 테스트마다 새 프로세스**(hang 커널 하나가 HAL `-110` 으로 뒤 커널을 전부 오염 — 격리하면
> 통과 수가 10 → 33 으로 3배 달라진다), **② 값 검증 필수**(에러 없이 조용히 틀린 값을 주는 결함이
> 실존), **③ 타임아웃**(`timeout 150`), **④ 실행 전후 `furiosa-smi status`** 로 누수 확인.
>
> 한편 `/home/jun/yik` 자체는 `furiosa-opt-std="0.3"` 핀 + `src/furiosa-opt.tag` 라
> **현재 0.4 래퍼로는 `--backend npu` 불가**(`_GROUND_TRUTH.md:14-16`). 실기를 쓰려면 0.4 로 포팅해야 한다.

> **주의(GROUND_TRUTH)**: `yik/src/kernel/gemm_kernel.rs:8` 의 "16×16 output tile" 주석은 **틀렸다** — Slice=`m![I/32, J/32]`=16×16 격자=256 슬라이스, 각 슬라이스 담당 타일은 `m![I%32, J%32]`=**32×32**. 검산 256×1024=262,144=I·J (`_GROUND_TRUTH.md:32-34`). 또 `switch()` 는 실제로 호출되지 않는다(분배는 `to_dm` 의 Slice 타입 파라미터). 0.3→0.4 API 차이(`contract_outer` 제네릭 4→5, 주소 인자 제거, `to_buf()`→`into_vec()`)는 `_GROUND_TRUTH.md:21-26` 참조.

**L4 초심자 설명**: L4 는 "컴파일러가 자동으로 하던 배치를 내가 손으로 다 적는" 최하층이다. 지금 **분석·시뮬레이션(typecheck/emulation)은 얼마든지** 할 수 있어서 "TCL 3줄 vs vISA 20줄" 같은 생산성 비교 논문 자료로는 즉시 쓸 수 있다(#19 open). **실칩(#20)도 이제 열렸다** — 하드웨어가 무조건 0번 카드만 쓰는 건 여전하지만, 그 카드가 유휴이면 바로 돌아간다(2026-07-24 실측: 실기 테스트 89개 실행, 80개 통과). 다만 실기의 진짜 관문은 카드 점유가 아니라 **패키지 안의 모든 `#[device]` 커널이 낮춰져야 한다**는 것이고(위 ★ 블록), 라우터에 모델이 올라가 있으면 npu0 이 잠기니 그때는 서버 정지가 전제가 된다.

**L4 에서 무엇을 최적화할지에 대한 스케줄 모델 기준 경고**: 위 계측 절이 보여주듯 사이클의 **96.5% 가 DMA** 이고 PeCore 는 3.3% 다(컴파일러 스케줄 모델 예측이며 실측 벽시계가 아니다 — 위 계측 절 참조). L4 가 배타적으로 주는 슬라이스 내부 제어는 **그 3.3% 안에서만** 효과가 있다. "vISA 로 연산을 더 잘 짜서 빠르게 한다"는 방향은 Amdahl 상한에 갇히고, 실제 레버는 DMA 전송량·레이아웃·정렬이다.

---

## L5 — 폐쇄 영역

컴파일러 바이너리 내부. **수정 불가**. 여기 표는 "왜 시간 쓰면 안 되는가"를 못박는 용도다. 예외적으로 #24(masquerade)만 관측 가능한 우회가 있다.

| # | 파일·지점 | 무엇을 바꾸나 | 왜 막혔나 / 관측 | 상태 |
|---|-----------|---------------|-------------------|------|
| 22 | `/home/jun/furiosa/bin/furiosa-tcc`(145MB), `native_torch*.so`(105MB) — 태틱 적법성·비용모델·타일링·스케줄링 | **없음.** 파이썬은 제안만, 결정은 전부 이 안 | 모든 컴파일러 내부 실패가 단일 예외로 세탁: `torch/compiler/compiler.py:66-68` 이 어떤 예외든 `UnsupportedOpError('failed to compile the graph')` 로 변환. `global-compiler/src/lib.rs:100` 패닉이 이 경로로 표면화 | **blocked** (폐쇄소스) |
| 23 | `allow_external_operators` / `allow_unlowered_operators` (`native_torch/compiler.pyi:17,41`, 각 모델 config.py) | **없음.** 둘 다 단순 Config 불리언, 연산자 주입 API 부재. 벤더 배포는 전부 False | DeltaNet 조사에서 값 바꿔도 무변화(`logs/build_override.log` 21분10초 동안 0/9 정체, BUILD OK 미방출). **이 경로에 시간 쓰지 말 것** | **blocked** (API 부재) |
| 24 | furiosa-llm model_type 게이트 2단(로드 `next_gen.rs:238`, 엔진 `hf_compat_next_gen.rs:367`), `binary_bundle.zip` | 직접 수정 불가. 유일 레버는 `artifact.json` 의 `model_metadata.model_type` 재작성(masquerade) | 확인 사례: qwen3_moe(FP8)→qwen3 마스커레이드로 Qwen3-Coder-30B-A3B-FP8 서빙이 단일 카드 **62.7 tok/s** 부활. mini qwen3_next 아티팩트는 `/v1/completions` 200 OK. `artifacts/mini-qwen3-next-served/artifact.json` 디스크 잔존=재현 가능. **논문에선 Rust 심볼·주소 인용 말고 관측 동작만**(라이선스·역공학 위험) | **partial** (masquerade 우회만 가능) |
| 25 | `CausalModelForwardInputs.kv_caches` (paged-KV 계약) — 순환 상태 슬롯 부재 | **없음.** (nv,dk,dv) DeltaNet 상태 슬롯이 계약에 없음 | 미사용 (K,V) 슬롯 배정 시 데드노드 → `graph_partitioner.py` IndexError. 0-스케일 keep-alive 는 상수 폴딩으로 제거. 충실한 DeltaNet 을 서빙 경로에 넣는 사용자 우회로 없음 — 벤더 런타임 변경 필요 | **blocked** (계약 부재) |

**L5 초심자 설명**: 여기는 "닫힌 방"이다. 컴파일러가 내부에서 무슨 결정을 하든 밖에서 못 바꾸고, 실패하면 죄다 `failed to compile the graph` 한 줄로 뭉개진다(#22). 불리언 두 개(#23)는 겉보기엔 노브 같지만 실제로 아무것도 안 바뀐다 — 21분 태우고 확인됨. 유일하게 "먹는" 트릭은 #24 마스커레이드: 아티팩트 메타데이터의 model_type 을 다른 이름으로 속여 화이트리스트를 통과시키는 것(62.7 tok/s 부활 사례). 단 논문·공개물에는 바이너리 내부 심볼·주소를 쓰지 말 것.

---

## nextAction 요약

`_data_optmap.json` nextAction 이 지목한 **즉시 실행 순서**:
1. **§0 보호 조치(2분)** — `qwen3_next.py`(37,499B) + `presets.py.bak-20260616` 를 `/home/jun/RNGD-proj` git 트리로 커밋. pip 한 번에 소실.
2. 그다음 최우선 작업은 **NPU 0시간·컴파일 0회·카드 무관**한 것: 이미 디스크에 있는 요청단위 기록(`tps_20260518_131427.json` 50건 + 요청당 255 ITL 등)으로 TTFT·ITL·요청당 TPS 의 부트스트랩 95% CI 를 산출 → 1순위 논문의 "n=1, 오차막대 없음" 반론을 그 자리에서 해소.
3. 병렬로 IEIE 추계 2026 일반세션 마감(2026-10-19, 현재 유일 검증된 열린 마감) 캘린더 고정.

**저비용 실험 우선순위(이 지도 기준)**:
- **L2 #11 (EdfModule, hours/미실시)** — 뚫리면 L2 전체 봉쇄 해제. 30분 결정적 실험.
- **L3 #14 (separate_vector_ops/fuse_mamma, hours)** — 4순위 논문 착수 전 필수 선행.
- **L3 #12 (MoE BB/BGi/BG 표, hours)** — 벤더가 임시라 인정한 최고 명분 개구부.
- **L1 #1 / #3, L4 #19** — 지금 바로 측정 가능(카드1 또는 NPU 미점유).

---

## ⚠ biggestTrap — 마지막 경고

**"RNGD가 GPU 대비 3배 전력 효율" 류의 배치-1 단일 지점 주장을 논문 헤드라인으로 쓰지 말 것.** 사용자 자신의 데이터가 이를 반박한다(원문 `_data_optmap.json` biggestTrap 그대로):

- **134.3W vs 402.2W 는 배치 1 행일 뿐.** 전체 스윕에서 RNGD 134.3→155.9→175.9→193.3→204.0→209.5W, PRO6000 402.2→451.1→480.0→505.2→562.3→599.9W 로 올라가 **토큰당 에너지 우위가 2.03배(b1)→1.33배(b32)→1.07배(b64)→1.08배(b256) 로 붕괴**한다.
- **최적 대 최적은 1.950 vs 1.695 = 1.15배**에 불과(`results/report.md:73`). 벤더 헤드라인 지표(SLO 정규화 users/kW)에서는 **방향 역전** — SLO20 에서 RNGD 46.8 vs GPU 88.7 = 0.53배, RNGD 는 단일 스트림 상한 25.23 TPS/user 로 **SLO30 도달조차 못 함**.

세 가지 가중 요인:
1. **독립 실험 아님** — 이 데이터셋은 2026-04-02 FuriosaAI 마케팅 블로그의 명시적 **재현 키트**(`bench-blog/README.md:1-4`, `results/report.md:3` 출처 URL). "최초의 독립 전력효율 측정" 신규성 주장 불가. K-Perf 가 2026-06-04 이미 RNGD 실측 공표 → "국내 최초 실측"도 거짓. **살아남는 주장은 "최초의 학술 논문 형태 제3자 재현"뿐.**
2. **전 측정이 1카드 단일 디바이스 샘플링**(`rngd.json` meta `power_devices='0'`, label `1card/tp8`). 4카드 전력도 A6000 전력도 없음. 4카드 섀시 회계(유휴 3장 × ~39.5W)면 **배치 1 제외 전 구간에서 RNGD 가 진다.**
3. **b=256 행은 정상상태 아님** — TTFT p50 이 **142.72초**로 측정 루프 하드캡 120초 초과. 중앙값 요청이 루프 종료 후 첫 토큰 수신. **b=256 행은 헤드라인 표에서 제외하거나 비정상상태로 명시.**

**안전하고 더 강한 대안**: 배수가 아니라 **곡선**을 제시하고, 회계 경계(장치/섀시/SLO)를 **주장 문구 안에** 명시하며, **부분 비재현(partial non-reproduction) 자체를 결과로** 내세운다.

---

### 이 문서에서 미확인으로 남긴 것

**해소된 항목 (2026-07-24)**
- ~~`--dump-schedule` 82,449 cycle / DmaStore 88% / Main 0.94% 예 — 재현 컴파일 미실시~~
  → **해소.** 커널 130개로 확장 재현했고 DMA 지배가 전반적 특성임을 확인(위 계측 절).
- ~~L4 #20 npu0 점유 PID 1215564 / 메모리 97.86%~~
  → **반박됨.** 실측 시 npu0~npu3 전부 `alive / 0.00 GiB` 였고 실기 테스트 89개를 실행했다.
  해당 행의 상태를 `blocked` → `open` 으로 정정했다.

**여전히 미확인**
- L2 #11 `ir.Edf.deserialize` 경로 — `EdfModule`(edf.py:131) 존재는 실측 확인했으나 deserialize 왕복 자체는 미검증(30분 실험 미실시).
- L3 #17 tcl.Dram 사전분할 대응 가중치 직렬화 writer 존재 여부 — furiosa/kernels/ 에 없음(원본 표기, 이 세션 재확인 안 함).
- `compiler_config_overrides ~96키`, `heuristic_hint 23키`, `furiosa-tcc` 두 키 각 7회 grep — `_data_optmap.json` 수치를 옮긴 것으로 이 세션 개수 재검증 안 함.
- `summary.json` 의 `operator_schedule_heuristic`·`beam_size`·`total_states_visited` — 키 존재는 실측,
  `mnist::forward` 에서 전부 `None` 인 이유와 다른 커널에서 채워지는지는 미확인(`schedule_method` 빈 문자열도 의미 미확인).