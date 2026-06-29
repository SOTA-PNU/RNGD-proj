# RNGD 4장(tp32) 빌드: 왜 실패했고 어떻게 되게 만들었나

## 한 줄 요약 (먼저 솔직하게)

**진짜 tp32(가중치를 4칩에 1/4씩 쪼개 큰 모델을 올리는 것)는 공개 SDK로는 안 됩니다.**
공개 컴파일러는 칩 사이 가중치 "분할(TP)"을 구현하지 않았고(`Inter-chip TP is not
implemented yet`), 칩 사이 "복제(Broadcast)"만 됩니다.

그래프 메타데이터의 `tp_config.inputs`에 가중치 배치를 `Broadcast`로 주입하면 빌드가
끝까지 진행되긴 합니다(Qwen2.5-Coder-7B tp32 빌드 완료). 하지만 빌드 결과물을 디코딩해 보면
**모든 가중치가 4칩에 통째로 복제**되어 있어(칩당 = 전체, 1/4 아님), 메모리 이득이 0입니다.
즉 tp8과 메모리가 똑같아서 **큰 모델을 못 올립니다**. 어텐션·KV 캐시만 칩 사이로 분할되므로
순수 dp는 아니지만, 메모리 이득도 처리량 이득도 없어 실용성은 낮습니다(6절·아래 표 참고).

벤더가 레포에 넣어둔 미리 빌드된 tp32 아티팩트(qwen3-32b 등)는 더 새로운 내부 컴파일러
(`d19a92a2f2`)로 만든 것으로, 가중치가 실제로 1/4씩 분할되어 있습니다. 큰 모델용 진짜 tp32가
필요하면 그 prebuilt 아티팩트를 쓰거나, 그 컴파일러가 공개되기를 기다려야 합니다.

| | dp(복제) | 진짜 tp32(벤더) | 이 문서의 빌드(Broadcast) |
|---|---|---|---|
| 가중치/칩 | 전체 | **1/4** | 전체 |
| 큰 모델 적재 | ❌ | ✅ | ❌ |
| 처리량 | N배 | 1배 | 1배(칩간통신 손해) |

---

## 1. 증상: tp32로 빌드하면 임베딩에서 바로 멈춥니다

작은 모델(Qwen2.5-Coder-1.5B)을 옵션 없이 tp32로 빌드해도 똑같이 실패합니다. 즉 특정 모델
문제가 아니라 tp32 자체의 문제입니다.

```
Compilation failed for stage id: stage_0 ...
LayerRange(start=Embedding(), end=TransformerBlock(idx=0, QkvProjection))
Error: fail to compile: Graph input#0 must have Broadcast or Fixed DramShapeGuide (Name: embedding_table)
```

- 에러가 나는 위치: `furiosa_llm/parallelize/new_pipeline_builder.py` (컴파일 단계 stage_0)
- stage_0 = 임베딩부터 첫 블록의 Q/K/V 투영까지 묶인 첫 번째 컴파일 조각입니다.
- 핵심 문구: `embedding_table`(임베딩 가중치)이 **Broadcast 또는 Fixed** 배치를 가져야 하는데
  그렇지 않다는 뜻입니다.

재현 환경: `furiosa-llm 2026.2.0-release (rev 9f92da0)`, RNGD 4장(32 PE). 같은 에러를 최신
공개 버전 `2026.2.1`에서도 그대로 확인했습니다.

---

## 2. DramShapeGuide가 무엇이고, 왜 임베딩에서 막히나

### 2-1. 배치(placement)와 DramShapeGuide

여러 칩·여러 PE에 텐서를 어떻게 나눠 올릴지를 "배치(placement)"라고 부릅니다. RNGD 컴파일러는
각 가중치가 **칩 사이로 어떻게 놓이는지**를 보고 DRAM 배치 안내(DramShapeGuide)를 정합니다.
종류는 세 가지입니다(`furiosa_llm/parallelize/compiler_config.py`의 `DramShapeKind`).

- `FREE`(자유): 칩 사이 배치가 정해지지 않은 상태
- `BROADCAST`(복제): 모든 칩에 똑같이 올림
- `FIXED`(고정): 정해진 규칙으로 칩 사이에 나눠 올림(예: 가로축을 4칩으로 4등분)

컴파일러는 임베딩 입력이 `FREE`이면 거부합니다. `BROADCAST`나 `FIXED`여야 합니다.

### 2-2. tp8은 되고 tp32는 안 되는 이유

- **tp8 = 1칩 × 8 PE**: 모든 분할이 한 칩 안(intra-chip)에서만 일어납니다. "칩 사이" 축이
  없으니 임베딩 배치도 문제없이 정해지고 빌드가 됩니다.
- **tp32 = 4칩 × 8 PE**: 이제 "칩 사이" 축이 생깁니다. 이 칩 사이 배치를 정해 줘야 하는데,
  공개 컴파일러는 임베딩 같은 가중치에 대해 이 값을 비워(`FREE`) 둡니다. 그래서 거부됩니다.

### 2-3. 이 배치는 파이썬이 아니라 닫힌 컴파일러가 정합니다

처음에는 파이썬 쪽 `GraphMetadata.input_dram_shape_guide`를 고치면 될 줄 알았지만, 이 값은
실제 빌드에 쓰이지 않는 죽은 코드였습니다(직렬화 결과를 빌더가 버립니다). 실제 배치는 닫힌
네이티브 컴파일러(`furiosa/native_llm_common...so`)가 `target_npu="renegade-8pe-4chip"`
문자열만 보고 내부에서 정합니다. 파이썬 `mppp` 계층은 가중치를 단일 멀티칩 장치에
`Replicate`로 올릴 뿐, 칩 사이 분할을 직접 지정하지 않습니다.

확인된 사실:
- 가중치 배치를 정하는 파이썬 코드가 없습니다. 인트라칩(8 PE)·인터칩(4칩) 분할 모두 닫힌
  컴파일러가 결정합니다.
- 공개 컴파일러는 임베딩처럼 조회(lookup)·원소별 가중치의 칩 사이 축을 `FREE`로 두고, 행렬곱
  가중치(Q/K/V 등)도 마찬가지로 `FREE`로 둡니다(아래 4-3에서 실측).
- 같은 네이티브 바이너리 안에 `Inter-chip TP is not implemented yet`(런타임)와
  `Chip: Broadcast=4`(칩 4개에 복제) 문자열이 함께 있습니다. 즉 **칩 사이 "복제"는 되지만 칩
  사이 "텐서 분할(TP)"은 아직 구현되지 않았다**는 뜻입니다.

### 2-4. 벤더의 tp32 아티팩트는 더 새 컴파일러로 만든 것입니다

레포에 들어 있는 미리 빌드된 tp32 아티팩트(`artifacts/qwen3-32b-fp8-tp32` 등)의
`artifact.json`을 보면 우리 것과 버전이 다릅니다.

| 항목 | 우리 SDK / 벤더 tp8 | 벤더 tp32 |
|------|---------------------|-----------|
| furiosa_llm | `9f92da0` | `b62dbc1` |
| 컴파일러 | `5c885c73ee` | `d19a92a2f2` |
| composable_ir | false | **true** |
| 임베딩 칩 사이 배치 | (1칩) | **LabelStride로 4칩 분할**(가로 5120 → 칩당 1280) |
| `.precommandgen`(칩 사이 DMA) | 없음 | **있음** |

즉 벤더의 tp32는 칩 사이 분할(`precommandgen`)을 지원하는 **더 새로운 내부 컴파일러
`d19a92a2f2`**로 만든 것이고, 이 컴파일러는 어떤 공개 릴리스에도 들어 있지 않습니다(우리가
설치할 수 있는 최신은 `2026.2.1`이고, 거기에는 없습니다).

---

## 3. 시도한 방법과 결과

| 시도 | 방법 | 결과 |
|------|------|------|
| H1 | 최신 공개판 `2026.2.1`로 업그레이드(격리 venv) | ❌ 임베딩에서 동일 에러 |
| H2 | `CompilerConfig(includes_composable_ir=True)` | ❌ 실패하지만 에러가 `fail to compile **into precommandgen**`으로 바뀜(벤더와 같은 컴파일 경로는 탐, 임베딩 배치는 여전히 FREE) |
| H3 | 그래프 메타데이터 `tp_config.inputs`에 배치를 직접 주입 | ✅ 통과 |

H1·H2로 알 수 있는 점: 공개판은 버전을 올려도, 벤더와 같은 컴파일 경로(`precommandgen`)를
켜도, 가중치 칩 사이 배치를 스스로 만들지 못합니다.

---

## 4. 해결 방법(H3): 배치를 메타데이터에 직접 적어 넣기

### 4-1. 어디에 끼워 넣나

빌드가 컴파일러에 넘기는 그래프 메타데이터는 `tp_config.inputs`라는 칸을 가지고 있고, 여기에
`텐서이름 → Free|Broadcast|Fixed`를 적으면 컴파일러가 그대로 따릅니다. 이 메타데이터를 만드는
곳이 한 군데뿐이라 거기만 손보면 됩니다.

- 파일: `furiosa_llm/parallelize/pipeline/builder/converter.py`
- 함수: `generate_graph_metadata` (약 1441~1467줄)
- 원래는 `return graph_metadata_builder.build()` 한 줄. 여기서 만들어진 YAML 문자열에
  `tp_config.inputs[가중치이름] = "Broadcast"`를 추가하고 다시 직렬화해서 돌려줍니다.

실제 패치는 `rngd-npu/tp32/converter_tp32_broadcast.patch`에 있습니다. 환경변수
`FURIOSA_TP32_BCAST`가 설정됐을 때만, 그리고 칩이 2개 이상일 때만 동작하므로 평소 빌드(tp8)와
서빙에는 전혀 영향이 없습니다.

> 참고: 컴파일은 Ray 작업자(별도 프로세스)에서 일어나므로, 실행 중에 코드를 바꿔치기하는
> 방식(monkeypatch)은 작업자에 전달되지 않습니다. 그래서 SDK 파일 자체를 수정해야 합니다
> (원본은 `converter.py.tp32fix.bak`로 백업해 두었습니다).

### 4-2. 컴파일러가 정말로 따르는지 확인

`embedding_table`에만 `Broadcast`를 주입하고 빌드하니, 에러가
`input#0 embedding_table`에서 **`input#7 rope_table`로 넘어갔습니다.** 즉 컴파일러가 주입한
값을 무시하지 않고 실제로 따릅니다(임베딩은 통과, 다음 비어 있는 입력에서 다시 멈춤).

### 4-3. 모든 가중치가 비어 있습니다

이름을 하나씩 추가하며 빌드하니 막히는 입력이 계속 바뀝니다.

`embedding_table` → `rope_table` → `norm_weight`·스케일·바이어스 → `attn_Q_proj_weight`(행렬곱
가중치) ...

즉 공개 컴파일러는 임베딩뿐 아니라 **행렬곱 가중치까지 전부** 칩 사이 배치를 비워 둡니다.
(벤더의 더 새 컴파일러는 행렬곱 가중치를 자동으로 칩 사이 분할합니다.) 그래서 공개판에서는
모든 가중치를 `Broadcast`(복제)로 채워 줘야 통과합니다. 이것이 `FURIOSA_TP32_BCAST=ALL`
모드입니다.

### 4-4. 끝까지 빌드 성공

- **Qwen2.5-Coder-1.5B**: stage_0(임베딩+QKV)은 통과했지만 stage_1(어텐션)에서
  `cannot divide target axis size (2) by num_npus(4)`로 멈췄습니다. 이 모델은
  `num_key_value_heads=2`라 어텐션을 4칩에 나눌 수 없습니다(2 ÷ 4 불가). tp32 시험용으로는
  부적합한 모델이라는 뜻입니다.
- **Qwen2.5-Coder-7B**(`num_key_value_heads=4`, 4로 나눠떨어짐): **13개 컴파일 조각 전부
  통과, tp32 빌드 끝까지 성공**(약 39분). 산출물 검증:
  - `artifact.json`: `tensor_parallel_size=32`, `pipeline_parallel_size=1`,
    devices `npu:0:0-7,npu:1:0-7,npu:2:0-7,npu:3:0-7`(4칩 × 8 PE)
  - `binary_bundle.zip`: 컴파일된 `.edf` 13개
  - 보관 위치: `rngd-npu/artifacts/qwen2.5-coder-7b-tp32-replicated-demo`

참고로 벤더가 미리 빌드해 둔 tp32 모델(qwen3-32b, llama-3.3-70b, exaone-32b)은 모두
`num_key_value_heads=8`로, 4로 나눠떨어집니다. 위 나눗셈 제약과 일치합니다.

### 4-5. 빌드 결과물 검증: 사실은 "복제"였습니다 (중요)

빌드가 됐다고 끝이 아니라, 산출물 내부(`binary_bundle.zip`의 `.edf`)를 CBOR로 디코딩해서
가중치가 칩 사이로 진짜 분할됐는지 확인했습니다. 결과는 **전부 복제(Broadcast)**였습니다.

| 가중치 | 칩 사이 배치 | 칩당 크기 | 전체 | 진짜 분할이면(1/4) |
|--------|-------------|-----------|------|--------------------|
| `attn_Q_proj_weight` | Broadcast | 3584 | 3584 | 896 |
| `mlp_gate_proj_weight` | Broadcast | 18944 | 18944 | 4736 |
| `embedding_table` | Broadcast | 152064 | 152064 | 38016 |

칩당 크기가 전체와 같습니다(1/4이 아님). 즉 **4칩 각각이 모델 전체를 들고 있어 메모리 이득이
없습니다.** 반면 벤더 tp32는 같은 종류의 가중치가 `LabelStride`로 칩당 1/4(예: Q가 8192 →
2048)만 들고 있어 칩당 메모리가 4배 줄어듭니다.

단, KV 캐시와 어텐션 계산은 제 빌드에서도 칩 사이로 분할됩니다(`k_cache`가 kv-head 축
`LabelStride`). 그래서 "순수 dp"는 아니지만, 가중치가 복제라 큰 모델을 못 올리는 건 마찬가지고,
한 요청을 4칩이 나눠 처리하며 칩간 통신만 늘어 처리량은 dp보다 못합니다. **메모리 이득도,
처리량 이득도 없는 형태**라는 점을 분명히 해 둡니다.

### 4-6. 진짜 분할(Fixed)을 주입해도 컴파일러가 못 만듭니다 (결정적)

"그러면 Broadcast 말고 진짜 분할(Fixed/LabelStride)을 주입하면 되지 않나?"를 끝까지
실험했습니다. `tp_config.inputs`는 단순 `Broadcast`뿐 아니라 파라미터가 있는 `Fixed` 분할도
형식상 받아들입니다(round-trip 확인). 그래서 임베딩에 분할을 주입해 봤습니다.

1. **inter 축만 주입**(임베딩을 hidden축 4분할): 게이트는 통과(더 이상 "Broadcast 또는 Fixed"
   에러 없음)했지만 다음 단계에서 `failed to lower the operator#O1118`로 실패.
2. **벤더와 100% 동일한 full Fixed 스펙**(inter+intra 축 전부, 벤더 tp32 임베딩 디코딩값을
   1.5B 치수로 스케일)을 composable/precommandgen 경로(벤더와 같은 경로)로 주입: 그래도
   `fail to compile into precommandgen: failed to lower the operator#O1118`로 동일 실패.

즉 **메타데이터(분할 지시)는 받아들여지지만, 공개 컴파일러가 "칩 사이로 분할된 연산"을 실제
NPU 코드로 변환(lower)하지 못합니다.** 게이트가 아니라 코드 생성 단계에서 막히며, 이것이
`Inter-chip TP is not implemented yet`의 실체입니다. 벤더가 만든 것과 똑같은 분할 지시를 줘도
우리 컴파일러로는 안 됩니다.

**그래서 공개 SDK에서 컴파일되는 칩 사이 배치는 `Broadcast`(복제)뿐이고, 그건 메모리 이득이
없습니다. 진짜 tp32(가중치 분할)는 벤더 내부 컴파일러(`d19a92a2f2`)가 있어야만 됩니다.**

### 4-7. radare2로 바이너리 패치까지 시도했지만 안 됩니다 (결정적)

"컴파일러 .so를 직접 패치해서 막힌 걸 뚫으면 안 되나?"도 끝까지 해봤습니다.

먼저 컴파일러 바이너리(`native_llm_common.so`)에 칩 사이 기계장치가 **부분적으로 실재**함을
확인했습니다(`inter_chip_exchange`, `inter_chip_command`, `inter_chip_cluster`, repartition
명령, 그리고 약 9.5KB짜리 실제 inter-chip exchange lowering 함수). 즉 "통째로 없다"가 아닙니다.

radare2로 분석한 결과, 칩 사이 exchange 함수(`fcn.0354f2a0`) 안에 "축이 하나라도 split(분할)
이면 에러로 점프"하는 **게이트**(`0x354f50a`, `Inter chip exchange with split axes is not
supported`)가 있고, 바로 다음이 실제 성공 경로였습니다. 그래서 그 점프를 NOP(`90 90 90 90 90`)
으로 덮어 게이트를 우회했습니다.

**그런데 패치 후 다시 빌드해도 결과가 똑같았습니다**(`failed to lower the operator#O1118`,
동일 지점). 실제로 막히는 연산 O1118은 임베딩 가중치를 host에서 분할된 DRAM 배치로 싣는
**Bridge(가중치 적재) 연산**인데, 제가 우회한 게이트(exchange 경로)와는 **다른 경로**라서
효과가 없었습니다. 즉 칩 사이 분할 lowering이 **여러 연산에서 제각각 미완성**이라, 게이트 하나
뚫어봤자 다음 미완성 지점에서 막힙니다(게이트 하나의 문제가 아님).

설령 모든 지점을 억지로 우회해도 그 뒤 코드는 non-split을 가정하고 짜여 있어 **엉터리 결과나
크래시**가 나고, 서빙은 런타임의 `Inter-chip TP is not implemented yet`에 또 막힙니다.
**결론: 바이너리 패치로도 진짜 tp32는 안 됩니다. 빠진 구현을 패치로 만들어낼 수는 없습니다.**
(패치한 .so는 격리된 실험용 사본이고, 끝나고 원본으로 되돌렸습니다.)

---

## 5. 재현 방법

전제: SDK에 `converter_tp32_broadcast.patch`가 적용되어 있어야 합니다(이미 적용·백업해
두었습니다). 빌드는 NPU 장치를 점유하지 않으므로(오프라인 컴파일) 서빙 중에도 돌릴 수 있습니다.

```bash
# 한 칩에 들어가고 KV 헤드 수가 4의 배수인 모델만 됩니다.
FURIOSA_TP32_BCAST=ALL ~/furiosa/bin/python \
  Model_Benchmark/rngd-npu/tp32/build_tp32.py \
  Qwen/Qwen2.5-Coder-7B-Instruct  ./out_tp32
```

원래대로 되돌리려면 백업을 복원하면 됩니다.

```bash
cp ~/furiosa/.../converter.py.tp32fix.bak ~/furiosa/.../converter.py
```

---

## 6. 한계와 주의사항

1. **복제(Broadcast)만 됩니다.** 공개 컴파일러는 칩 사이 "분할(TP)"이 없어서, 모든 가중치를
   칩마다 통째로 복제하는 방식만 빌드됩니다. 따라서 **한 칩에 들어가는 크기의 모델**만 됩니다.
   한 칩에 안 들어가는 큰 모델(예: 72B bf16)은 이 방법으로 못 만듭니다. 그런 모델의 진짜 분할
   tp32는 벤더의 내부 컴파일러(`d19a92a2f2`)가 필요하고, 그게 레포의 미리 빌드된 tp32
   아티팩트입니다.
2. **KV 헤드 수가 칩 수(4)로 나눠떨어져야** 합니다(어텐션을 4칩에 분배하기 때문). 7B·14B·32B·
   72B(8헤드), 7B(4헤드)는 되고, 1.5B·0.5B·3B(2헤드)는 안 됩니다.
3. **서빙(추론) 정확성은 아직 검증하지 못했습니다.** tp32 서빙은 4장(32 PE)이 전부 비어 있어야
   하는데 지금은 사용자 서버가 NPU를 쓰고 있어 띄워 볼 수 없었습니다. 이 글은 "빌드가 된다"까지
   확인한 것이고, 복제 방식 tp32의 추론 결과가 맞는지는 NPU가 비었을 때 따로 확인해야 합니다.
4. 패치는 환경변수가 있을 때만 동작하는 opt-in이라, 기존 tp8 빌드와 서빙에는 영향이 없습니다.

---

## 7. 출처

- 에러·빌드 로그: `furiosa_llm/parallelize/new_pipeline_builder.py`(컴파일 실패 지점),
  1.5B·7B 빌드 실측 로그
- DramShapeKind(FREE/BROADCAST/FIXED) 정의: `furiosa_llm/parallelize/compiler_config.py`
- 패치 지점: `furiosa_llm/parallelize/pipeline/builder/converter.py`의 `generate_graph_metadata`
- 네이티브 문자열: `furiosa/native_llm_common...so`(`Chip: Broadcast=4`),
  `furiosa/native_runtime...so`(`Inter-chip TP is not implemented yet`)
- 벤더 tp32 비교: `artifacts/qwen3-32b-fp8-tp32`(컴파일러 `d19a92a2f2`, composable_ir true,
  임베딩 LabelStride 4칩 분할)
- 적용 파일: `rngd-npu/tp32/converter_tp32_broadcast.patch`, `rngd-npu/tp32/build_tp32.py`
