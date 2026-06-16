# Qwen3-Coder-Next를 furiosa-llm으로 다루기 (개념을 잇는 실전편)

작성일 2026-06-15.

이 문서는 앞의 두 개념 문서를 이어받아, 실제 모델 **Qwen3-Coder-Next-FP8(80B)**을 FuriosaAI
RNGD NPU에서 어떻게 다루는지를 설명합니다. 왜 표준 방식이 막히는지, 우리가 어떻게 우회해서
실제로 NPU에서 코드를 생성하고 서빙까지 했는지, 그리고 실제 명령까지 한 곳에 모았습니다.

먼저 읽으면 좋은 개념 문서:
- 모델 구조(어텐션 기초부터 Gated DeltaNet·Gated Attention): [README_attention_and_gated_deltanet.md](README_attention_and_gated_deltanet.md)
- furiosa-llm CLI(build/serve)가 하는 일(예시 모델로): [README_furiosa_llm_cli_explained.md](README_furiosa_llm_cli_explained.md)

깊은 SDK 내부(파일·줄 단위), 전체 변경 이력은 같은 폴더의 `ALL_about_build_serve.md`,
`README_all_change.md`를 참고하세요. 이 문서는 그 둘을 빼고 "이 모델을 어떻게 다루나"에
집중합니다.

---

## 1. Qwen3-Coder-Next는 어떤 모델인가

개념 문서에서 쌓은 내용으로 보면, 이 모델은 어텐션 발전의 거의 끝에 있는 하이브리드입니다.

- 종류: `qwen3_next` (Qwen3NextForCausalLM)
- 크기: 약 80B(800억) 파라미터, FP8로 약 80GB(40개 safetensors 조각)
- 층: 48개. 그중
  - **36개 층**: Gated DeltaNet (선형 어텐션 + delta rule + 망각 게이트). 긴 글을 싸게 처리.
  - **12개 층**(4개마다 1개): Gated Attention이 붙은 보통의 softmax 어텐션. 정확한 전역 참조 담당.
  - 모든 층에 MoE(전문가 512명 중 토큰마다 10명 + 공유 전문가 1명). 전체는 크지만 한 번에
    일하는 부분(활성 파라미터)은 약 3B로 작음.
- 어텐션 세부:
  - DeltaNet: 키 머리 16개, 밸류 머리 32개, 머리 크기 128, 짧은 conv(커널 4).
  - 풀 어텐션: 쿼리 16/키밸류 2(GQA), 머리 크기 256, 부분 RoPE(25%), 출력 게이트(Gated Attention).
- hidden 2048, vocab 151936, 컨텍스트 최대 262144.

핵심은 개념 문서 11절에서 말한 성질입니다. **Gated DeltaNet 층은 KV 캐시(뒤에 덧붙이기)가
아니라, 고정 크기 상태 행렬을 매 토큰마다 읽고 고쳐 다시 쓰는(read-modify-write) 메모리를
씁니다.** 이 한 가지가 표준 NPU 서빙을 막는 근본 원인입니다.

---

## 2. 왜 표준 furiosa-llm build/serve로는 안 되나

예시 문서(B)에서 본 표준 흐름에 비춰 보면, 막히는 곳이 분명해집니다. 막힘은 "컴파일이 안 된다"가
아니라 대부분 "서빙 구조가 안 맞는다"입니다.

### 2-1. serve 쪽 벽 (벤더만 풀 수 있음)

1. **모델 종류 미등록.** furiosa의 서빙 런타임(컴파일된 폐쇄 바이너리)은 자기가 아는 모델
   종류 목록(llama, qwen2, qwen3 등)에만 반응합니다. 이 목록에 `qwen3_next`가 없어서, 정식
   `furiosa-llm serve`는 아티팩트를 보자마자 "모르는 모델"이라며 거부합니다.
2. **KV 캐시 계약이 append-only.** 서빙 런타임은 모든 층이 "뒤에 덧붙이기만 하는" 키·밸류
   캐시를 쓴다고 가정합니다. Gated DeltaNet의 "읽고 고쳐 쓰는 상태 행렬"을 담을 자리가
   런타임에 아예 없습니다.
3. **순환상태 풀 없음.** 위 2번과 같은 뿌리입니다. 매 토큰마다 상태를 제자리에서 갱신하는
   기능이 런타임에 없습니다.

이 셋은 모두 컴파일된 바이너리(.so) 안에 있어서 우리가 파이썬으로 못 고칩니다. 이건 벤더
(FuriosaAI)가 다음 버전(2026.3 이상)에서 추가해야 풀립니다.

### 2-2. build 쪽 벽 (이건 우리가 파이썬으로 고칠 수 있음)

흥미로운 점은, **NPU 컴파일러 자체는 이 모델의 계산을 다 컴파일할 수 있다**는 것입니다.
예전에는 "NPU 컴파일러가 DeltaNet 계산을 못 한다"고 생각했는데, 직접 실험해 보니 그게
아니었습니다.

진짜 막힘은 B 문서 3단계의 **나눠 칠하기(partition)** 에 있었습니다. 빌드는 색칠된 조각마다
컴파일을 한 번씩 합니다(SDK의 `converter.py`, 조각이 단일 그래프인지 확인하는
`assert len(compiled.graphs)==1`는 converter.py:1052). 그런데 DeltaNet 본체 전체가 한 색으로
뭉쳐 한 조각이 되면, 그 조각 안에 여러 종류의 계산이 섞여서 컴파일러가 거부합니다(`conflict
between concrete labels`, `multiple internal subgraphs`).

해결의 열쇠는 개념이 아니라 **그래프를 어떻게 쪼개느냐**였습니다. **"계산 한 개 = 그래프 한 개"로
잘게 쪼개면**, 표준 모델의 깔끔한 조각들과 똑같은 모양이 되어 전부 컴파일됩니다(아래 5절에서
실측 증거). 이 쪼개기는 파이썬 쪽(커스텀 partitioner)에서 할 수 있습니다.

정리하면, 표준 furiosa-llm으로 **계산은 다 컴파일할 수 있지만**(파이썬 partitioner 추가 시),
**정식 serve가 돌릴 수 있는 단일 아티팩트는 못 만듭니다**(serve 런타임의 순환상태 풀이 없어서).

---

## 3. 우리가 푼 방법 한눈에

표준 serve가 막히니, 우리는 두 축으로 풀었습니다.

1. **host 추론 루프**: 토큰 생성 루프를 NPU 런타임이 아니라 우리 파이썬(host)이 직접 돌립니다.
   순환 상태(읽고 고쳐 쓰는 행렬)는 host가 들고 있고, 무거운 행렬 계산만 NPU에 보냅니다. 이러면
   런타임의 순환상태 풀이 없어도 됩니다(상태를 host가 가지니까).
2. **계산을 NPU 커널로 직접 작성 + 잘게 쪼개 컴파일**: DeltaNet의 각 계산을 손수 작성한 NPU
   커널로 만들고, 컴파일이 막히던 부분은 "계산 한 개 = 그래프 한 개"로 쪼개 전부 NPU 바이너리
   (a6 EDF)로 만들었습니다.

이 둘을 합쳐, 실제 80B 모델이 RNGD NPU에서 올바른 코드를 생성하고 OpenAI 호환으로 서빙까지
됩니다. 아래에서 하나씩 봅니다.

---

## 4. 방법 1: host 추론 루프 (이게 실제로 모델을 돌리는 본체)

host 추론 루프는 이렇게 동작합니다.

```
입력 토큰들 → embed(host) →
 for 층 i in 0..47:
    입력 정규화(host) →
    토큰 믹서:
       DeltaNet 층이면  → 손수 작성한 NPU 커널들로 계산 (host가 상태 S와 conv 상태 보유)
       풀 어텐션 층이면 → 투영·SDPA를 NPU 커널로 계산 (host가 KV 보유)
    → 잔차 합 → 사후 정규화(host) → MoE(라우팅은 host, 전문가 계산은 NPU) → 잔차 합
 → 최종 정규화(host) → lm_head(host) → 다음 토큰 뽑기 → 반복
```

핵심은 **상태를 host가 들고 있다**는 점입니다. DeltaNet의 "읽고 고쳐 쓰는" 상태 행렬과 짧은
conv 상태를 파이썬 텐서로 보관하면서, 토큰마다 갱신합니다. 무거운 행렬 곱셈은 NPU에 보내고
결과만 받습니다. 가중치는 80GB라 한꺼번에 NPU에 못 올리므로, **층마다 필요한 가중치만
host에서 풀어서(FP8 역양자화) NPU로 보내고 계산 후 버립니다**(스트리밍). 그래서 47.5GB짜리
카드 한 장으로도 80B 모델이 돕니다(느리지만 정확).

이 host 추론 루프가 NPU에서 실제로 도는지(파이썬 CPU로 몰래 떨어지는 게 아닌지)는 실측으로
4중 확인했습니다. 토큰을 만드는 동안 (1) CPU 폴백 카운터가 0이고, (2) `furiosa-smi ps`가 그
프로세스를 NPU 코어에 묶어서 보여 주며, (3) host CPU가 바쁜 건 폴백이 아니라 가중치
역양자화·glue 작업이고, (4) DPE(빠른 행렬 엔진) 결과가 NPU 하드웨어 특유의 약 0.23% 오차
지문을 보였습니다.

코드 위치: `qwen3-next-proj/qcn/model.py`(48층 루프), `loader.py`(FP8 역양자화),
`deltanet_layer*.py` · `attn_layer.py` · `moe.py`(층별 NPU 커널 호출), `generate.py`(생성 CLI).

---

## 5. 방법 2: 계산을 NPU 커널로 직접 작성하고, 막히면 쪼개기

DeltaNet의 계산은 손수 작성한 NPU 커널(`furiosa.torch.TacticKernelModule`)로 NPU에서 돌렸고,
HuggingFace 원본과 소수점 7자리까지 일치시켰습니다. 더 나아가, 빌드용 컴파일러
(`furiosa.native_common.compiler.compile`)로 a6 EDF 바이너리까지 뽑으려 할 때 막히던 세 조각을 쪼개서
전부 통과시켰습니다.

쪼개기 전후를 실측한 결과입니다.

| 조각 | 통째로 컴파일 | 한 계산씩 쪼개서 |
|---|---|---|
| DeltaNet 순환 한 스텝 | 실패 (`conflict between concrete labels`) | 5조각으로 쪼개니 전부 성공 |
| conv1d(짧은 합성곱) | 실패 (`O136` 미지원) | host에서 미리 패딩 + 곱셈·덧셈으로 바꾸니 성공 |
| 게이트(softplus 포함) | 실패 (`log1p` 미지원) | sigmoid와 log(1+exp)로 나누니 성공 |

순환 한 스텝을 쪼갠 모양(원본과 fp64에서 똑같음, fp32 상대오차 약 0.0000003):
```
state = state × α          (감쇠: dn_recur_decay)
kv    = k @ state          (읽기: dn_recur_contract)
delta = (v − kv) × β       (차이: dn_recur_delta)
state = state + k ⊗ delta  (쓰기: dn_recur_outer + dn_recur_add)
out   = q @ state          (출력: dn_recur_contract 재사용)
```

**여기서 얻은 교훈이 핵심입니다.** 막힌 이유는 계산 자체가 NPU에서 안 되어서가 아니라,
한 그래프에 여러 종류 계산이 섞이면 컴파일러가 거부했기 때문입니다. "계산 한 개 = 그래프
한 개"로 쪼개면 통과합니다. 이건 표준 모델(Qwen2.5-Coder-7B)의 어텐션·MLP 조각이 원래 단일
계산이라 잘 컴파일됐던 것과 같은 원리입니다.

---

## 6. binary_bundle: 계산은 전부 NPU 바이너리로 (25조각, 검증됨)

위에서 쪼갠 조각들을 포함해, 이 모델의 모든 계산을 a6 EDF 바이너리로 만들어 `binary_bundle.zip`에
담았습니다. 이건 B 문서에서 본 표준 모델의 binary_bundle.zip과 **똑같은 형식**입니다(같은
컴파일러, 같은 압축·배치).

- 25개 EDF 블롭, 약 515MB.
- 전부 SDK의 `CompiledGraph.deserialize`로 다시 읽혀 a6로 검증됨(25/25 통과).
- 들어 있는 계산: 모든 Linear 투영, 풀 어텐션 SDPA, MoE SwiGLU, 정규화, lm_head/embedding,
  그리고 **DeltaNet 순환·conv·게이트의 쪼갠 조각들**.

즉 **계산 측면에서는 표준 모델만큼 완성**(compute-complete)됐습니다. 위치는
`rngd-npu/artifacts/qwen3-coder-next-fp8-rngd/binary_bundle.zip`이고, 도시락 설명서
(artifact.json)의 `kind`가 `edf-split (compute-complete)`로 표시됩니다.

남은 한계는 6절의 컴파일이 아니라, 2-1절의 serve 런타임(순환상태 풀)입니다. 즉 **계산은 다
NPU 바이너리로 만들 수 있지만, 그걸 정식 serve로 자동 연결해 돌리는 부분만 벤더 몫**입니다.

---

## 7. 서빙: 4가지 경로

이 모델을 OpenAI 호환으로 서빙하는 길은 네 가지입니다. 앞 세 가지는 지금 됩니다.

### 7-1. 우리 단일카드 서버 (`qcn/serve.py`)

가장 단순하고 안정적입니다. 요청을 하나씩 순서대로 처리하고, greedy(가장 확률 높은 토큰)만
지원합니다. 엔드포인트는 `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`.

### 7-2. 우리 멀티카드 스트리밍 서버 (`qcn/serve_mc.py`)

카드별로 일꾼(워커)을 띄우고, 요청 큐 + 실시간 스트리밍 + 샘플링(temperature/top_p)을
지원합니다. 다만 **동시 처리량은 크게 늘지 않습니다.** 병목이 NPU가 아니라 host(가중치
역양자화 + glue 작업으로 요청당 약 40코어 점유)라서, 워커별 스레드 수를 제한해도 host가
포화됩니다. 진짜 선형 확장은 파이프라인 병렬(층을 카드별로 나누기)이 필요하고 아직
미구현입니다.

### 7-3. furiosa-llm의 serve 서버 코드를 우리 엔진으로 구동 (`qcn/furiosa_serve_adapter.py`)

이게 질문 "serve 런타임에 순환상태 풀과 계산 연결을 만들면 furiosa-llm serve를 쓸 수 있나?"에
대한 실전 답입니다. 정답은 **네이티브 런타임(.so)에는 못 넣지만, furiosa-llm의 파이썬 serve
층은 우리 엔진으로 갈아끼울 수 있다**는 것입니다.

이유는 이렇습니다. furiosa-llm의 파이썬 서빙 엔진(`AsyncLLMEngine`)은 실제 일을 `llm.engine`에게
넘기기만 하는데, `llm.engine`의 타입이 느슨해서(덕타이핑, api.py:94) 우리가 만든 파이썬 엔진을
끼울 수 있습니다. 게다가 네이티브가 돌려주는 출력 객체(`NativeRequestOutput` 등)도 파이썬에서
직접 만들 수 있습니다(llm.pyi). 그래서 네이티브 엔진과 같은 인터페이스를 가진 우리
`HostLoopEngine`을 끼우면, **furiosa-llm 자신의 `AsyncLLMEngine`이 우리 host 추론 루프를
구동**합니다. 이때 "순환상태 풀"은 요청마다 host가 들고 있는 상태가 되고, "계산 연결"은 우리
host 루프가 됩니다.

실제로 실증했습니다. furiosa-llm의 `AsyncLLMEngine.generate`가 우리 `HostLoopEngine`을 구동해
`furiosa_llm.outputs.RequestOutput`을 스트리밍했고, 계산은 전부 NPU에서 돌았습니다(CPU 폴백 0).
완전한 OpenAI HTTP 경로(`OpenAIServingChat`)도 같은 방식으로 연결됩니다.

### 7-4. 정식 `furiosa-llm serve <아티팩트>` CLI 그대로

이건 **그대로는 안 됩니다.** CLI가 아티팩트로 네이티브 엔진을 만들려 하고, 2-1절의 모델 종류
게이트가 `qwen3_next`를 거부하기 때문입니다. (코더가 목적이고 DeltaNet이 꼭 아니어도 된다면,
이미 빌드되는 MoE 코더인 Qwen3-Coder-30B-A3B를 모델 종류만 `qwen3`로 위장해 정식 serve로
띄우는 인접 대안이 있습니다.)

### 7-5. 그럼 네이티브 런타임에 호출부만 넣고 우리가 구현하면(바이너리 패치) 되지 않나

자주 나오는 발상입니다. 네이티브 런타임(.so)에 "여기서 우리 함수를 부르고 돌아와라" 식으로
호출부(`call X; ret`)만 박고, 그 `X`(순환상태 풀, sub-op 연결)를 우리가 구현하면 정식 serve가
되지 않겠냐는 것입니다. 발상의 방향(런타임이 못 하는 일을 우리 코드에 맡기기)은 정확하지만,
**이 문제에 바이너리 패치는 맞지 않습니다.** 실측 근거 다섯 가지입니다.

1. **끼울 훅이 없습니다.** `native_runtime.so`의 익스포트 심볼은 `llg_*`(문법 제약 디코딩)·
   `tch_*`(torch 스트림 IO)·`PyInit_*`뿐이고, `recurrent_state`·`state_pool`·`ExternalOperator`·
   `plugin`·`register_hook` 같은 문자열은 0건입니다(strings 실측). 디코드·KV·스케줄러는 전부
   내부 mangled Rust라, 합법적으로 바인딩할 콜백 진입점이 아예 없습니다.
2. **"비우고 채울 함수"가 없습니다.** `ret 0`으로 대체할 기존 함수가 있다는 전제인데, 런타임에는
   순환상태를 다루는 기능 자체가 없습니다. 빠진 건 잘못된 함수 하나가 아니라 **구조적으로 부재한
   서브시스템**(상태 할당 + 매 스텝 read-modify-write + 레이아웃 라우팅)이라, 호출부 하나를
   패치한다고 생겨나지 않습니다.
3. **한 스텝이 통째 EDF 실행입니다.** 런타임은 매 스텝 컴파일된 모델 그래프(EDF)를 KV 모양
   입출력으로 돌립니다. 우리 `X`가 끼려면 런타임의 비공개 Rust 구조체(요청 상태·KV 블록
   포인터·텐서 레이아웃)를 알아야 하는데, 이건 안정된 ABI가 없어 구조체 오프셋을 역설계해
   하드코딩해야 합니다.
4. **그래도 host 사전계산이 필요합니다.** 청크 prefill의 삼각역행렬·cumsum은 host가 미리
   계산해야 하므로(알고리즘적), 패치를 해도 host 왕복을 못 피합니다.
5. **유지가 안 됩니다.** 주소 기반 패치는 SDK가 다시 빌드될 때마다 깨집니다.

**그런데 발상의 본질은 .so를 안 건드리고 이미 달성됩니다.** "런타임의 역할(디코드 루프 + 상태
보유)을 우리 코드가 한다"를 **Python 층에서** 하면 됩니다(7-3절의 어댑터). 더 나아가 **정식
`furiosa-llm serve` CLI가 우리 엔진을 쓰게 만드는 것도 가능**합니다. CLI는 `LLM(아티팩트)`를
만들 때 네이티브 엔진을 생성하는데, `LLM` 생성만 가로채는 약 30줄짜리 Python shim
(`qcn/furiosa_serve_cli_shim.py`)을 끼우면, host-loop 아티팩트일 때 우리 `HostLoopEngine`을
대신 꽂습니다(네이티브 엔진 생성을 건너뛰므로 serde 게이트도 자동 우회).

실증(2026-06-15): shim 설치 후 정식 CLI의 서버 구성 경로를 그대로 재현하니
`LLM.engine = HostLoopEngine` → `furiosa_llm.llm_engine.AsyncLLMEngine.from_llm` 통과 →
**`OpenAIServingChat`(정식 OpenAI HTTP 핸들러) 구성 성공** → `AsyncLLMEngine.generate`로
`furiosa_llm.outputs.RequestOutput` 토큰 생성, CPU 폴백 0(all_on_npu). **바이너리 패치 없이
정식 serve 스택 전체가 우리 host 루프로 동작**했습니다.

정리하면, "네이티브에 호출부를 박는다"는 어렵고 깨지기 쉬운 길이고, 같은 목적을 Python
shim으로 더 안전하게 달성합니다. 다만 이때도 디코드 루프는 우리 host 루프이고, 네이티브
런타임의 연속 배칭·스케줄러를 그대로 쓰는 건 아닙니다(그건 여전히 벤더 몫). 우리 모델은 어차피
host-bound라 그 차이의 throughput 영향은 작습니다.

실행(정식 CLI를 우리 엔진으로):
```bash
PYTHONPATH=$PROJ RNGD_DEV=rngd:16 $PY -c \
  "import qcn.furiosa_serve_cli_shim; from furiosa_llm.cli.main import main; \
   import sys; sys.argv=['furiosa-llm','serve','$ART']; main()"
```

---

## 8. 실제 명령 모음

환경 변수 두 개를 먼저 알아 둡니다.
- `RNGD_DEV=rngd:N`: 사용할 **글로벌 PE 인덱스**입니다(카드 번호 아님). 이 머신은 4카드 ×
  8PE = 32PE이고, `0~7=npu0, 8~15=npu1, 16~23=npu2, 24~31=npu3`입니다(실측: `rngd:9`는
  `furiosa-smi ps`에 `npu1:1`로 보임). 빈 PE를 `furiosa-smi ps`로 확인해 쓰세요(npu0의 0~3은
  자주 바쁨).
- `QCN_DPE=1`: 빠른 DPE(systolic 행렬 엔진) 사용. prefill 약 4.7배, decode 약 1.6배 빠릅니다.
  정확도 검증이 필요하면 `QCN_DPE=0`(fp32).

경로 변수:
```bash
PY=/home/jun/furiosa/bin/python
PROJ=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj
ART=/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen3-coder-next-fp8-rngd
```

### 빌드 / 컴파일

```bash
# (1) 모든 계산 조각을 a6 EDF로 컴파일
PYTHONPATH=$PROJ RNGD_DEV=rngd:4 $PY $PROJ/tk_kernels/compile_edf_blobs.py
PYTHONPATH=$PROJ RNGD_DEV=rngd:5 $PY $PROJ/tk_kernels/emit_dn_split_blobs.py   # DeltaNet 분해 조각

# (2) binary_bundle.zip으로 묶고 검증 (25/25 통과 확인)
PYTHONPATH=$PROJ RNGD_DEV=rngd:5 $PY $PROJ/tk_kernels/pack_edf_bundle.py

# (3) self-contained 아티팩트(도시락) 빌드. --emit-edf면 위 (1)(2)까지 실행해
#     25조각 compute-complete 번들을 만듭니다.
PYTHONPATH=$PROJ $PY $PROJ/qcn/build_artifact.py --out $ART --emit-edf

# (4) 아티팩트로 실행
PYTHONPATH=$PROJ RNGD_DEV=rngd:4 QCN_DPE=1 $PY $PROJ/qcn/run_artifact.py \
    --artifact $ART --prompt "def quicksort(arr):" --max-new 3
```

### 생성 (CLI)

```bash
PYTHONPATH=$PROJ RNGD_DEV=rngd:2 $PY $PROJ/qcn/generate.py --prompt "def add(a, b):" --max-new 24
```

### 서빙

```bash
# 단일카드 (포트 8900)
PYTHONPATH=$PROJ RNGD_DEV=rngd:2 $PY $PROJ/qcn/serve.py

# 멀티카드 스트리밍 (환경변수로 카드/PE 지정)
PYTHONPATH=$PROJ $PY $PROJ/qcn/serve_mc.py
#   QCN_DEVS: 사용할 글로벌 PE 인덱스. 값이 PE 인덱스라 "0,1,2,3"은 npu0의 4 PE(한 카드)에
#   몰립니다. 물리 4카드에 1개씩 분산하려면 "0,8,16,24"를 쓰세요.
#   QCN_DPE=1(기본 빠름)/0(정확), PORT=8900

# 호출 예시
curl -s localhost:8900/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"qwen3-coder-next-fp8-rngd",
  "messages":[{"role":"user","content":"Write a Python factorial function."}],
  "max_tokens":32, "temperature":0}'
```

furiosa-llm 서버 코드를 우리 엔진으로 구동하는 어댑터 사용 예(7-3절):
```python
import asyncio
from qcn.model import QCNModel
from qcn.furiosa_serve_adapter import build_async_engine
from furiosa_llm import SamplingParams

async def main():
    m = QCNModel()
    engine = build_async_engine(m)   # furiosa_llm의 진짜 AsyncLLMEngine
    async for out in engine.generate("def add(a, b):",
                                     SamplingParams(max_tokens=8, temperature=0.0), "req-1"):
        print(out.outputs[0].text)

asyncio.run(main())
```

---

## 9. 성능 · 옵션 · 자주 겪는 문제

성능(실측, DPE 켠 상태):
- prefill(프롬프트 처리): 약 77초에서 143초(프롬프트 길이·PE 점유 상태에 따라 다름).
- decode(토큰 1개 생성): 약 34초에서 44초. host가 병목입니다(가중치 역양자화 + glue).
- DPE는 VE(벡터 엔진) 대비 prefill 약 4.7배, decode 약 1.6배 빠릅니다. DPE는 bf16이라 약
  0.23% 오차가 있는데, 실제 모델이 FP8/bf16이라 오히려 자연스럽습니다.

자주 겪는 문제와 해결:
- `furiosa::dfg only runs on CPU device`: 재컴파일 한도 초과로 CPU 폴백. model.py가 한도를
  크게 올려 둠. 별도 코드를 짠다면 같은 설정 필요.
- 대화 입력에서 AttributeError: 일부 transformers 버전이 BatchEncoding을 돌려줘서 생김.
  코드에서 `input_ids` 추출로 처리됨.
- DPE 결과가 이상함: 한 그래프에 DPE 계산이 3개 이상이면 잘못 컴파일됨. DeltaNet 스캔은 DPE
  2개짜리 커널을 씀. 출력 축이 1이면 거부되므로 32의 배수로 패딩.
- 빌드 후 결과가 안 바뀜: 그래프 캐시 `~/.cache/furiosa/llm/graphmodules/*Qwen3Next*` 삭제.
- PE 점유 충돌: `RNGD_DEV`를 빈 PE로. `furiosa-smi ps`로 확인.

---

## 10. 무엇이 우리 손이고, 무엇이 벤더 몫인가 (정직한 경계)

우리가 파이썬으로 할 수 있는 것:
- 모든 계산을 a6 EDF로 컴파일(쪼개기), binary_bundle 묶기, host 추론 루프, host-loop 아티팩트,
  furiosa-llm serve 서버 코드에 우리 엔진 끼우기, (원하면) 빌드용 커스텀 partitioner.

오직 벤더(FuriosaAI, 2026.3 이상)만 할 수 있는 것(컴파일된 .so, 소스 없음):
1. 서빙 런타임에 `qwen3_next` 모델 종류 등록(모델 종류 게이트 통과).
2. 네이티브 선형 어텐션/Gated DeltaNet 커널 또는 순환을 한 그래프로 도는 Loop 기능.
3. 런타임의 **cross-step 순환상태 풀**(읽고 고쳐 쓰기). 이게 없으면 정식 `furiosa-llm serve`로
   DeltaNet 디코드를 자동으로 못 돌립니다.

한 줄 요약: **계산은 우리 손으로 다 끝낼 수 있고, 정식 native serve만 벤더 게이트입니다.** 그
사이에는 host 추론 루프(7-1, 7-2)와 furiosa-llm serve 어댑터(7-3)가 완전한 대안입니다.

---

## 11. 파일 맵

```
qwen3-next-proj/
  qcn/
    model.py                  48층 host 추론 루프 (prefill/decode/generate/generate_stream)
    loader.py                 safetensors mmap + FP8 역양자화
    deltanet_layer*.py        DeltaNet 층 (NPU 커널 호출)
    attn_layer.py             풀 어텐션 층 (Gated Attention 포함)
    moe.py                    MoE (라우팅 host, 전문가 계산 NPU)
    generate.py               생성 CLI
    serve.py                  단일카드 OpenAI 서버
    serve_mc.py               멀티카드 스트리밍 서버
    furiosa_serve_adapter.py  furiosa-llm AsyncLLMEngine에 끼우는 HostLoopEngine
    furiosa_serve_cli_shim.py 정식 furiosa-llm serve CLI가 우리 엔진을 쓰게 하는 Python shim(.so 패치 없음)
    build_artifact.py         self-contained 아티팩트 빌드 (--emit-edf)
    run_artifact.py           아티팩트 적재·실행
  tk_kernels/
    compile_edf_blobs.py      기본 계산 -> a6 EDF
    emit_dn_split_blobs.py    DeltaNet 분해 -> a6 EDF
    pack_edf_bundle.py        binary_bundle.zip 묶기 + 검증
    dn_*.yaml                 손수 작성한 NPU 커널
rngd-npu/artifacts/qwen3-coder-next-fp8-rngd/   빌드 산출 아티팩트
```

더 깊은 내용:
- furiosa SDK 내부(파일·줄 단위): `ALL_about_build_serve.md`
- 전체 변경 이력: `README_all_change.md`
- 모델 구조 개념: [README_attention_and_gated_deltanet.md](README_attention_and_gated_deltanet.md)
- CLI 개념: [README_furiosa_llm_cli_explained.md](README_furiosa_llm_cli_explained.md)
