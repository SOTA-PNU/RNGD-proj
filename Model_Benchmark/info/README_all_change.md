# Qwen3-Coder-Next RNGD 프로젝트 — 전체 변경 기록

> 이 문서는 "Qwen/Qwen3-Coder-Next 모델을 RNGD 4장에서 서빙한다" 프로젝트에서
> 발생하는 **모든 변경 사항**을 기록합니다. 무엇을, 어디서, 어떻게, 왜 바꿨는지를
> 시간순으로 남깁니다. 기존 코드 수정은 최소화하고, 필수 수정만 합니다.
>
> 시작일: 2026-06-10 · SDK: furiosa-llm 2026.2.0 · 장비: RNGD 4장 (펌웨어 2026.2.1)

---

## 변경 이력 (시간순)

### 2026-06-10 — 프로젝트 시작, 사전 조사 (코드 변경 없음)

이 날은 조사만 진행했고 **SDK·프로젝트 코드 변경은 없습니다.**
이 문서(README_all_change.md) 신규 생성이 유일한 파일 추가입니다.

### 2026-06-10 — 미니 합성 모델 하니스 생성 (신규 파일, 기존 코드 변경 없음)

**왜:** 실제 모델(159GB)을 받기 전에, 장난감 크기 랜덤 가중치 모델로 빌드 파이프라인을
분 단위로 반복 검증하기 위함. transformers 5.1.0의 qwen3_next 레퍼런스로 CPU 기준
출력과 수치 비교도 가능.

**무엇을:**
- `Model_Benchmark/qwen3-next-proj/make_mini_model.py` 신규 — 미니 모델 생성 스크립트
- `mini_models/mini-qwen3` (161.6M), `mini-qwen3-moe` (163.2M), `mini-qwen3-next` (174.8M)
  생성 — 실모델의 구조적 특성(MoE, DeltaNet 비율 3:1, partial rotary 등)을 보존하고 크기만 축소
- 토크나이저는 Qwen/Qwen2.5-Coder-1.5B-Instruct 것 재사용 (vocab 151936 동일 계열)

### 2026-06-10 — [SDK 수정 #1] presets.py 에 미니 스모크 preset 추가

**파일:** `~/furiosa/lib/python3.12/site-packages/furiosa_llm/artifact/presets.py`

**왜:** CLI 는 `-pb`/`-db` 만 받는데 resolver 는 4종 버킷 전부(또는 전부 비움)를
요구함 → 버킷을 비우면 preset 매칭 필요. 미니 모델(h=512, i=1536)은 qwen3 의
경우 임베딩용 pooling preset(디코드 버킷 없음)에 log-거리 매칭돼 빌드 실패
(`No matching bucket preset found`, resolver.py:90 실측). 미니 전용 preset 이 필요.

**무엇을 어떻게:**
1. `MINI_SMOKE_PRESET` 추가 (PRESET_REFS 직전) — prefill (1,128) / decode (1,1024) /
   append (1,256,128) / tokenwise (128,) 최소 구성
2. `PRESET_REFS` 끝에 `(qwen3, 512, 1536)`, `(qwen3_moe, 512, 1536)`,
   `(qwen3_next, 512, 1536)` 3개 ref 추가 → 모두 `MINI_SMOKE_PRESET` 참조
3. 기존 엔트리는 변경 없음. 실모델 매칭에 영향 없음을 검증
   (`find_preset('qwen3', 5120, 25600)` 여전히 32B preset 반환 확인)

---

## 사전 조사 결과 요약 (2026-06-10)

### 대상 모델: Qwen/Qwen3-Coder-Next

HF API 실측 (huggingface.co/api/models/Qwen/Qwen3-Coder-Next, 2026-06-10 조회):

| 항목 | 값 |
|---|---|
| 파라미터 | 79.7B (BF16, 159.4GB, safetensors 40개) |
| model_type | `qwen3_next` (`Qwen3NextForCausalLM`) |
| 레이어 | 48개 — **36개 Gated DeltaNet(선형 어텐션) + 12개 일반 어텐션** (`full_attention_interval=4`) |
| MoE | **512 experts, top-10**, moe_intermediate=512, shared expert 512, 전 레이어 MoE |
| 어텐션 | 16 heads / 2 KV heads / head_dim 256 / partial rotary 0.25 |
| hidden / vocab / 최대길이 | 2048 / 151936 / 262144 |
| 라이선스 | Apache 2.0 |

### SDK 현황 (furiosa-llm 2026.2.0, `~/furiosa/lib/python3.12/site-packages`)

- `furiosa/models/language/architecture/` 에 **qwen3_next 없음**
  (있는 것: llama, qwen2, qwen3, qwen3_moe, qwen3_vl, exaone, exaone4, exaone_moe, gpt_oss, mistral, mllama4, phi3)
- venv의 transformers **5.1.0에는 qwen3_next 레퍼런스 구현 있음** → 참조 구현으로 활용 가능
- `furiosa_llm/` 파이썬 전체에 mamba/SSM/선형어텐션 상태 관리 코드 **없음** (grep 검증, 2026-06-10)

### 네이티브 런타임 제약 (strings 직접 확인, 2026-06-10)

- `furiosa/native_runtime.so`, `native_llm_common.so`, `native_torch.so` 3개 바이너리 모두에서
  `qwen3_next`/`Qwen3Next`/`deltanet`/`gated_delta`/`linear_attn`/`mamba` 문자열 **0건**
- 기존 실측(README_build.md 12절, 2026-06-04): serve 시 네이티브 게이트
  (`furiosa-generator/src/next_gen/hf_compat_next_gen.rs:367`)가
  `(model_type, task, weight, act/kv)` 조합을 화이트리스트 검사 → 미지원 조합은
  `PanicException: Unsupported model metadata` 로 즉시 종료
- MoE 커널은 **BF16 전용 2종만 존재** (`blockwise_moe_w16a16`, `default_blockwise_moe_qwen3_w16a16`)
  → FP8 MoE serve 불가 (Qwen3-Coder-30B-A3B-FP8 실측 패닉 사례)

### 선례 (이전 세션 실측 기록)

| 사례 | 결과 | 의미 |
|---|---|---|
| Qwen2.5-Coder 7B/14B (qwen2, preset만 추가) | 빌드+serve+측정 성공 | dense 신규 모델은 preset 추가만으로 가능 |
| Qwen3-Coder-30B-A3B-FP8 (qwen3_moe) | 빌드 성공, **serve 패닉** | MoE×FP8 커널 부재. "빌드 성공 ≠ serve 성공" |
| tp32 빌드 (MoE bf16) | stage_0 임베딩 컴파일 실패 | 4장 단일 인스턴스 빌드 불가 |
| serve `-pp` (tp8 아티팩트 레이어 분할) | 2026-06-09 실측 동작 | 1장 초과 모델도 tp8 빌드 + serve pp로 가능성 |

### 4장 메모리 타당성 (개산)

- tp32 빌드 불가 → tp8(1장) 아티팩트 + serve `-pp 4` 경로만 가능
- BF16 159GB ÷ 4장 ≈ **40GB/장 + KV캐시·워크스페이스** → 48GB HBM 대비 매우 빡빡 (검증 필요)
- FP8로 줄이면 ~80GB지만 **FP8 MoE serve 불가**(위 게이트) → 2026.2.0에서는 BF16 외길

### 핵심 리스크 (조사 시점 평가)

1. **(최대) DeltaNet 순환 상태**: 36개 레이어가 토큰마다 갱신되는 고정 크기 상태
   (레이어당 32×128×128 + conv state)를 요구. 런타임은 paged KV cache 계약만 지원 —
   상태 텐서 지원 여부가 프로젝트 성패를 가름. (심층 분석 진행 중)
2. **런타임 게이트**: `qwen3_next`는 게이트 enum에 없음 → serve 시 메타데이터 통과 전략 필요
3. **512 experts**: 기존 검증은 128 experts까지. 컴파일 시간·메모리 스케일 미지
4. **호스트 RAM**: 125GB RAM + 200GB swap에서 160GB weight 트레이싱 OOM 위험

---

### 2026-06-10 — 미니 모델 빌드/서빙 검증 + ⭐ 메타데이터 위장 돌파구

**실측 결과 (시간순):**

1. **mini-qwen3 (dense) 빌드 성공** — preset 추가 후 전체 빌드 ~2분 (트레이싱 3버킷 + EDF 컴파일 6 supertask).
   serve(npu:0) + `/v1/completions` 생성 성공. 미니 하니스 검증 완료.
2. **mini-qwen3-moe (BF16 MoE) 빌드 성공, serve 패닉** —
   `Unsupported model metadata { model_type: Some(Qwen3Moe), ..., weight: BF16 }`
   (`hf_compat_next_gen.rs:367`). **기존 가설("BF16 MoE는 serve 될 여지") 반증** —
   2026.2.0 런타임 게이트는 MoE를 양자화 무관하게 전부 거부.
3. ⭐ **위장(masquerade) 실험 성공** — mini-qwen3-moe 아티팩트의 `artifact.json` 에서
   `model_metadata.model_type` 을 `qwen3` 으로, `hf_configs` 를 dense mini 것으로 교체
   (KV 차원 동일: 2L/kv2/hd64). **serve 부팅 + NPU 토큰 생성 성공.**
   - **증명된 것:** 런타임 게이트는 메타데이터 문자열만 검사하고, MoE 연산은 이미
     EDF 바이너리에 전부 컴파일되어 있어 런타임 MoE 커널이 필요 없음.
   - **함의:** (a) 2026.2.0에서 MoE serve가 사실상 가능, (b) qwen3_next 도 같은 경로로
     serve 가능성 — 단 KV 캐시 계약이 표준과 동일해 보여야 함 (DeltaNet 상태가 관건).
4. (진행 중) 실물 `Qwen3-Coder-30B-A3B-Instruct-FP8-tp8-65k` 아티팩트(기존 serve 패닉으로
   사장)를 같은 방식으로 위장 → serve 테스트. 실제 학습 가중치라 출력 품질로 수치
   정확성까지 검증 가능. 위장 사본: `qwen3-next-proj/artifacts/qwen3-coder-30b-masq`
   (params 디렉터리는 하드링크, 디스크 추가 소모 없음. 원본 무변경.)

**위장 방법 (재현 절차):**
```python
# artifact.json 에서
d['model']['model_metadata']['model_type'] = 'qwen3'          # 게이트 통과용
hf = d['model']['model_metadata']['hf_configs']
hf['model_type'] = 'qwen3'; hf['architectures'] = ['Qwen3ForCausalLM']
# MoE 전용 키 제거 (decoder_sparse_step, moe_intermediate_size, num_experts*, ...)
# 주의: KV 캐시 차원(num_hidden_layers, num_key_value_heads, head_dim)은 절대 변경 금지
#       — 런타임이 hf_configs 기준으로 캐시를 할당하므로 실제 그래프와 일치해야 함
```

### 2026-06-10 — ⭐⭐ 실물 30B 코더 위장 serve 성공 + 속도 측정

**`Qwen3-Coder-30B-A3B-Instruct-FP8` (model_type→qwen3 위장) 단일 카드(npu:0, tp8) 실측:**

| 항목 | 결과 |
|---|---|
| serve 부팅 | ✅ 성공 (이전 패닉 → 정상) |
| 코드 생성 품질 | ✅ 정상 (Fibonacci/quicksort 정확·docstring 포함) |
| 단일 스트림 속도 | **62.7 tok/s** (quicksort 128토큰, temp 0) |

이전 세션에서 "serve 불가"로 사장됐던 30B MoE 코더가 **위장만으로 1장에서 실서비스 수준
속도로 부활.** 기존 Qwen2.5-Coder-14B(30.7 tok/s)보다 빠르고 더 강력한 MoE 코더.
4장이면 dp 복제로 처리량 4배 확보 가능. → **사용자 본래 목표("더 좋은 코더 모델")의
즉시 달성 가능한 경로 확보.**

### 2026-06-10 — 런타임/빌드 계약 심층 규명 (서브에이전트 2종, .so strings + 코드 실증)

**① serve 게이트 메커니즘 (확정):**
- 게이트는 `NativeLLMEngine()` 생성 시(api.py:383) **`artifact.json`의
  `model_metadata.model_type` 문자열 하나만** 검사 (`hf_compat_next_gen.rs`)
- 허용 enum: **`{llama, exaone4, qwen2, qwen3, qwen3_moe, gpt_oss, embed, score}`**
  (native_runtime.so strings 직접 확인). `qwen3_next` 없음.
- 게이트는 그래프 구조를 보지 않음 → **컴파일된 EDF에 연산이 다 들어있으면
  model_type만 통과시키면 실행됨** (위장이 동작하는 근본 이유, 실증 완료)
- KV 캐시 할당: 아티팩트 `pipeline_metadata_list[*].attention_buckets[*].kv_cache_size`
  버킷 스펙 기반. 단 hf_configs의 `num_hidden_layers/num_key_value_heads/head_dim`로
  shape 정합성 검증 → **이 KV 차원은 위장 시 절대 바꾸면 안 됨** (실제 그래프와 일치 필수)
- serve `-pp`: 순수 기하학적 분할(스테이지 수 × 디바이스). model_type 무관.

**② 빌드 KV 캐시 계약 = qwen3_next의 진짜 벽 (확정):**
- `CausalModelForwardInputs.kv_caches: List[Tuple[Tensor, Tensor]]`
  (`specs/inputs.py:14-89`), `__post_init__`이 **각 레이어를 (K,V) 2-튜플로 강제 검증**
- `create_kv_caches()` (`utils.py:423-467`)는 **전 레이어 동일 shape**
  `(num_blocks, block_size, num_kv_heads, head_dim)` 의 (K,V)만 생성.
  레이어별 다른 shape·추가 상태 텐서를 만드는 **확장 지점 없음**
- 캐시 쓰기는 in-place 페이지드 `cache[block_idx, slot] = x` (append 전용).
  DeltaNet의 read-modify-write 순환 상태와 **접근 패턴 자체가 다름**
- **결론:** DeltaNet의 conv state (B,8192,4) + recurrent state (B,32,128,128 fp32)를
  레이어당 추가로 흘려보내려면 `specs/inputs.py` 데이터클래스 + `create_kv_caches` +
  `LLMKVCacheWriter` + `parallelize/`의 TensorOrigin 파싱까지 **전부 수정 필요**
  (서브에이전트 2종 독립 확인). "마법 같은 확장 지점 없음."

**③ DeltaNet 연산 자체의 컴파일 불가 항목 (qwen3next-ref 분석):**
- chunked prefill 커널: fp32 `cumsum` (RNGD는 int32/64만), `F.pad`→`constant_pad_nd`
  (전면 불가), 부분 `slice_scatter` (full-axis만 가능), `groups=8192` depthwise conv1d
- decode 커널은 순수 elementwise+reduction이라 상대적으로 우호적

**종합 판정:** 충실한(faithful) Qwen3-Coder-Next의 2026.2.0 빌드는
(a) DeltaNet 연산 미지원 + (b) 페이지드-KV 전용 상태 계약 두 벽으로 **현 SDK 단독으론 막힘**.
빌드 파이프라인 다수 수정(parallelize/ 내부 포함) 또는 차기 SDK/벤더 지원 필요.
→ 다음 단계로 **미니 qwen3_next 아키텍처를 실제 작성·빌드해 위 분석을 실증**하고,
병행해서 **30B-A3B 코더 위장 경로를 사용자에게 제공할 결과물로 확정**.

### 2026-06-10 — qwen3_next 아키텍처 구현 + 빌드 게이트 우회 (SDK 수정 #2~#4)

**목표:** "직접 가벼운 모델을 실행해서 로그를 들여다본다" — 미니 qwen3_next 를 실제
빌드해 SDK 파이프라인이 어디서 막히는지 실증.

**[SDK 수정 #2] 신규 아키텍처 파일**
`~/furiosa/.../furiosa/models/language/architecture/qwen3_next.py` (신규, ~600줄)
- `Qwen3NextForCausalLM` / `Qwen3NextModel` — HF 클래스명과 정확히 일치 (resolver 가
  `getattr(furiosa.models, cls.__name__)` 로 찾음, modeling.py:173)
- full attention (gated, partial rotary 0.25, head_dim q/k norm), Gated DeltaNet
  (순환형, **probe: 상태 0-초기화·비유지**), MoE (routed + shared expert + gate) 구현
- full attn·MoE 는 SDK 코어 레이어 재사용(NPU 커널), DeltaNet conv1d·순환규칙은 torch

**[SDK 수정 #3] 패키지 등록**
`~/furiosa/.../furiosa/models/language/__init__.py` — import 2줄 + `__all__` 2개 추가.
검증: `furiosa.models.Qwen3NextForCausalLM` resolve True, CausalModelServer 서브클래스 True.

**[SDK 수정 #4] 빌드 게이트 우회 (실험적 model_type 허용)**
`~/furiosa/.../furiosa_llm/metadata/hf_utils.py:validate_model_support`
- **첫 실증 실패:** 게이트 없이는 빌드가 트레이싱 전에 즉사 —
  `ValueError: No compiler configuration available for model_type='qwen3_next'`
  (hf_utils.py:213). 네이티브 `find_compiler_config` 테이블에 qwen3_next 없음.
- **수정:** `_EXPERIMENTAL_MODEL_TYPES = {"qwen3_next"}` 추가. 이 집합의 model_type 은
  find_compiler_config 가 None 이어도 raise 대신 경고 후 통과. per-kernel 컴파일은
  unknown type 에 대해 `create_default_compiler_config()` 로 자동 폴백
  (compiler_config.py:138-142 확인)하므로 빌드 진행 가능.
- 기존 동작 보존: 목록에 없는 model_type 은 종전대로 raise. serve 게이트와는 무관.

### 2026-06-10 — qwen3_next 빌드 실패 계단 실증 (로그 기반)

미니 qwen3_next 빌드를 반복하며 막히는 지점을 하나씩 통과시켜 **실제 벽의 위치**를
실측했습니다 (사용자 요구: "직접 실행해서 로그를 들여다보고").

| 단계 | 결과 | 통과 방법 |
|---|---|---|
| ① 모델 클래스 resolve | ✅ | `__init__.py` 등록 (HF 클래스명 일치) |
| ② 빌드 진입 게이트 | ❌→✅ | `validate_model_support` 가 `find_compiler_config`=None 으로 즉사 → `_EXPERIMENTAL_MODEL_TYPES` 우회 |
| ③ **FX 트레이싱** | ✅ | hidden_states 2D 가정 수정. **DeltaNet 순환·conv1d·gated norm·MoE·gated attention 전부 트레이싱 성공** (= 연산 자체는 그래프화 가능) |
| ④ TP 그래프 분할 | ❌→(시도) | DeltaNet 이 배정된 (K,V) 캐시를 안 써서 dead node→`graph_partitioner.py:130 IndexError`. 캐시 0-배율 keep-alive 로 우회 시도 |
| ⑤ EDF 컴파일 | (측정 중) | — |

**핵심 발견:** qwen3_next 의 벽은 "연산 미지원"이 아니라(③ 트레이싱 성공),
**(a) 빌드 진입 게이트(②, 우회 가능)**, **(b) 페이지드-KV 전용 그래프 구조(④)** 였음.
④는 파이프라인이 "모든 레이어가 자기 (K,V) 입력을 소비"하길 강제하는 구조적 제약의
실측 증거 — DeltaNet 처럼 KV 를 안 쓰는 레이어는 dead node 를 만들어 분할기를 깨뜨림.
설령 keep-alive 로 빌드를 통과시켜도, serve 런타임(폐쇄 Rust)은 DeltaNet 순환 상태를
유지·관리하지 못하므로 **자기회귀 디코딩의 정확성은 별도 문제로 남음**.

### 2026-06-10 — ⭐⭐⭐ 30B-A3B 코더 4장 dp 서빙 처리량 실측 (최종 결과물)

`Qwen3-Coder-30B-A3B-Instruct-FP8` (qwen3 위장) 을 **4장 dp** 로 서빙:

| 동시성 | 합산 tok/s | 스트림당 | 출력 |
|---:|---:|---:|---|
| 1 | 63.2 | 63.2 | 정상(prime/fib/quicksort 정확) |
| 8 | 429.7 | 53.7 | 정상 |
| 32 | 1036.2 | 32.4 | 정상 |

기존 dense Qwen2.5-Coder-14B(단일 30.7 tok/s) 대비 단일 2배·합산 30배+.
**본래 목표("RNGD 4장으로 더 강력한 코더") 달성.** 운영 권장: 처리량은 4장 dp,
저지연은 1장 단독. 종합 결론·근거는 [README_qwen3_next_TECH.md] 참조.

### 2026-06-10 — qwen3_next probe 정리 + 위장 도구화

- `qwen3_next.py` 의 무효 keep-alive 해킹 제거(상수폴딩으로 무효 확인) → 솔직한
  주석으로 대체. 아키텍처 파일은 차기 SDK 대비 + ③ 트레이싱 성공 증거로 보존.
- `masquerade_artifact.py` 신규 — 위장 절차 재사용 도구화(하드링크 사본 + model_type
  교체 + MoE 키 제거, KV 차원 보존 가드).
- 종합 보고서 `README_qwen3_next_TECH.md` 신규.

### 2026-06-10 (2차 세션) — ⭐ TP 분할기 벽, 우리 손으로 통과 (사용자 질문 "vendor 몫 말고는 해결 가능하지 않나?"의 실증 답변)

사용자의 지적대로 빌드 측 분할기는 열린 Python — 직접 수정해 실증했습니다.

**근본 원인 규명 (워크플로 시뮬레이션 + 실측 디버그 로그로 이중 검증):**
- kernelwise 분할기는 레이어 i 마다 색 3개(2i=attn 앞 tokenwise, 2i+1=attention,
  2i+2=attn 뒤 tokenwise)를 칠하는데, **attention 색은 모듈 경로가
  `*.self_attn.attn` 인 모듈에만 시딩**(block_slicer.py:936-937, 1013-1016; 경로 정규식은
  graph_partitioner.py:50-58 하드코딩)
- DeltaNet 레이어(`linear_attn`)는 그 경로가 없어 attn 색 미시딩 → 파티션 ID 희소
  (실측 `ids=[0,2,4,6,7,8]`) → PartitionComposer 가 "관측 ID 개수" 길이의 조밀 리스트를
  "원시 ID 값"으로 인덱싱(graph_partitioner.py:119-131) → IndexError
- 이전 결론("dead KV 입력이 원인")은 **공동 증상이었지 원인이 아님** — dead 노드는
  파티션 0 으로 흡수되어 무해함이 시뮬레이션으로 확인됨

**적용한 해법 (SDK 수정 #5 — qwen3_next.py 내부만, 분할기 무수정):**
1. DeltaNet 순환 본체를 `_GatedDeltaNetCore` 로 분리해 `<layer>.self_attn.attn` 경로에
   배치 (마커 정규식에 부합) — 디코더의 모듈명도 `linear_attn`→`self_attn`
2. HF 가중치 이름 `linear_attn.*` → `self_attn.*` 리매핑 (transform_weights)
3. `make_example_inputs` @staticmethod 오버라이드 — KV 캐시를 full-attention 레이어
   수만큼만 선언 (호출부가 `model_cls.make_example_inputs` 형태라 서브클래스 적용됨,
   metadata/utils.py:70,83). 길이 검증은 모델 밖 어디에도 없음(에이전트 전수조사)

**결과 (실측):** 파티션 ID **연속 [0..8] 달성, 분할기 통과** ✅ → 다음 벽 도달:
`transform.py:116 replicate_nodes_with_multiple_colors` 의 KeyError — 명시된 가정
"다색 노드의 부모는 모두 같은 색"(transform.py:113 주석)을 DeltaNet 그래프가 위반.

**교훈 — 그래프 캐시 함정:** `~/.cache/furiosa/llm/graphmodules` 는 모델 가중치/설정
기준으로 키가 잡혀 **SDK Python 코드 수정을 반영하지 않음**. 아키텍처 코드를 고친 뒤
재빌드 전 `rm ~/.cache/furiosa/llm/graphmodules/*Qwen3Next*` 필수 (이번에 2회의
빌드가 옛 캐시로 잘못 평가됐음).

**임시 계측:** 진단용 디버그 로그를 graph_partitioner.py·block_slicer.py 에 넣었다가
**원인 확정 후 전부 제거** (현재 0건, SDK 임포트 정상 확인).

**serve 측 정밀 사실 (서브에이전트 실험, npu:1/2):** KV 버퍼 수·메모리 예산은
hf_configs 가 아니라 **아티팩트 파이프라인 텐서 목록**에서 파생(nhl=1/2/48 모두 동일
동작·동일 출력). `layer_types` 에 `linear_attention` 값이 있으면 Rust 파서 패닉,
prefix cache 기본 ON 은 하이브리드에서 조용한 오답 위험 → feasibility 2-1절에 반영.

### 2026-06-10 (3차 세션) — SDK 학습 종합 문서 ALL_about_build_serve.md 작성

사용자 요청: furiosa-llm build/serve 전 과정을 file:line·코드역할·원리·.so 역할까지
실측 기반으로 정리.

- 위장 30B 아티팩트를 `rngd-npu/artifacts/qwen3-coder-30b-a3b-inst-fp8-tp8-65k-tc/` 로
  이동(사용자가 옮김). 문서는 이 경로 기준으로 작성.
- **실측 수행:** ① 미니 qwen3_moe 풀로그 빌드(`logs/build_trace_full.log` — 3버킷 트레이싱·
  6 컴파일유닛·스테이지 LayerRange 분해 캡처) ② 실물 30B serve 부팅 로그
  (`logs/serve_30b_tc_newpath.log` — KVCachePlan·가중치 29.2GiB·스케줄러) ③ 라이브 serve
  엔드투엔드 생성 검증 ④ .so 3종 strings 크레이트 추출 ⑤ 6-에이전트 file:line 트레이싱.
- **신규 문서:** `info/ALL_about_build_serve.md` (632줄) — Part1 build, Part1B 30b vs next
  빌드차이, Part2 serve, Part3 .so 3총사, Part4 부록(캐시·함정).
- **검증한 핵심 사실:** 양자화 표기 `W8fA16KV16`=weight FP8/act BF16/kv BF16
  (`optimum/types.py:178-183` __str__, suffix `8f`=FP8). serve KV 블록 수는 아티팩트 KV
  텐서에서 도출(hf_config 무관, 실험확정). build 컴파일러=native_llm_common.so(CompiledGraph,
  12 pass npu-compiler), torch→EDF=native_torch.so(libc10 링크, serve 불가), serve 엔진=
  native_runtime.so(furiosa-generator). 인용 file:line 전수 재확인(app.py:186/533/549,
  serving_chat.py:192/211/256, llm_engine.py:453/483/578/611, api.py:343/349/381).

### 2026-06-10 (4차 세션) — radare2 게이트 분석 + qwen3_next serve 통과 시연 + 문서 교정

사용자 요청: ALL_about_build_serve.md 재검토 + 핵심 .so 를 radare2(github 소스 빌드)로
분석 + qwen3-coder-next serve 통과(공부 목적). 위장 30B 는 사용자가
`rngd-npu/artifacts/qwen3-coder-30b-a3b-inst-fp8-tp8-65k-tc/` 로 이동.

- **문서 재검토(2-에이전트 워크플로)**: ALL_about_build_serve.md 의 줄번호/사실 교정 적용 —
  find_preset(presets.py:395, 로직 407-422), validate_model_support(hf_utils.py:197),
  find_compiler_config(hf_utils.py:217), add_marker_op_hooks(block_slicer.py:1107),
  resolve_model_metadata 정의(resolver.py:246), _make_example_inputs 정의(causal.py:393),
  SchedulerConfig 12필드(config_types.py:84-96, 부팅로그 11 직렬화), ModelConfig
  seed_for_random_weight. (컴파일 스테이지·convert.py·builder.py·trace.py·.so 역할은 검증 정확.)
- **radare2 6.1.7 소스 빌드**: `qwen3-next-proj/radare2/`(설치는 sudo 필요해 빌드트리에서
  직접 실행, LD_LIBRARY_PATH 로 libr 로드).
- **serve 게이트 바이너리 해부(실측)**: 게이트가 **2겹** — ①load 게이트(native_llm_common.so,
  serde enum ModelType 역직렬화, `artifact/types/next_gen.rs:238`, api.py:349) ②engine 게이트
  (native_runtime.so, hf_compat_next_gen.rs:367). 허용 변형 6개
  {llama,exaone4,qwen2,qwen3,qwen3_moe,gpt_oss}. 매처 = **첫 바이트 점프 테이블**(0x00aca0d4)
  → 변형별 단말 블록(`lea 변형문자열; mov 길이; jmp [0x0888c620] compare 테일콜`),
  매처 함수 0x1fcd350~0x1fcd402, qwen3 단말 0x1fcd3cc. VARIANTS 배열 0x00019b00(fat-pointer,
  동적 재배치). model_type 은 enum + 구조로더(qwen3_32b/qwen3_30b_a3b) 동시 선택.
- **qwen3_next serve 통과 실측**: `model_type=qwen3_next` 그대로는 serde "unknown variant"
  거부 → `qwen3`(위장)로 바꾸자 게이트 통과·토큰 생성 성공. (mini-qwen3-as-next vs
  mini-qwen3-next-served.)
- **바이너리 패치 경로 규명(미실행)**: 리터럴 qwen3_next 통과는 (단말+라우팅 패치)×2바이너리
  +재배치+구조로더 때문에 고위험·저효용 → 분석으로 경로만 문서화, **안전한 마스커레이드를
  실 통과법으로 확정**. .so 2개 분석 후 `cmp` 로 PRISTINE 확인(읽기전용, 원본 무변경).
- **신규 문서**: `info/README_qwen3_next_TECH.md` (게이트 2겹·매처 디스어셈블·통과법·
  재현 r2 명령). ALL_about_build_serve.md Part2 에 1번째 게이트 + 교차링크 추가.
- 안전조치: `native_{runtime,llm_common}.so.orig` 백업 생성(가역). NPU 4장 해제.

### 2026-06-10 (4차 세션 추가) — 위장 NPU 영향 답변 + 바이너리 패치 실험(사본)

- **Q: 위장 통과가 NPU에 부정적 영향?** → **하드웨어 위험 없음.** EDF(NPU 명령)는 빌드 때
  그대로, 위장은 artifact.json model_type 문자열만 바꿈. serve 시 model_type 은 스케줄러/
  구조 프리셋만 선택(KV는 아티팩트 텐서에서 도출). 단점은 소프트웨어 레벨(프리셋 불일치
  시 처리량 비최적, 하이브리드 모델의 prefix 캐시 오답 위험, layer_types linear_attention
  파서 패닉)뿐. 30B 실측 정상.
- **Q: 사본 복사 후 radare2 패치로 qwen3_next 통과?** → 실제 시도. native_llm_common.so
  사본에 qwen3 단말(0x1fcd3cc)을 "qwen3_next"(10B, 케이브 0x1fcd402)로 패치
  (`wx 7177656e335f6e657874 @0x1fcd402; wx 2f000000 @0x1fcd3cf; wx 0a @0x1fcd3d4`),
  import 경로 교체 후 load 테스트 → **여전히 거부**. **매처가 길이-우선 디스패치라
  len-10 입력이 단말 도달 전 에러로 빠짐을 실증.** 단말 패치만으론 불가. 테스트 후 .orig
  복원 → 두 .so PRISTINE 확인. 결론: 길이 디스패치+라우팅+gate2까지 정밀 RE 필요해
  고위험 → 마스커레이드 권장. README_qwen3_next_TECH.md 5-1절에 상세.

### 2026-06-10 (5차 세션) — ⭐⭐ transform.py 통과 + 빌드가 NPU 컴파일러 백엔드까지 도달

사용자 요청: ⑤ transform.py 벽 통과 시도. **통과 성공** + 그 이후 op-import 단계까지 전부
뚫어 빌드가 실제 NPU 컴파일러 백엔드에 진입.

**[SDK 수정] transform.py 노드 복제 버그 우회**
`~/furiosa/.../furiosa_llm/parallelize/pipeline/builder/transform.py:replicate_nodes_with_multiple_colors`
- 원인 실측: 다색 노드 `to_dtype`(임베딩 캐스트, 색(0,2))의 부모 `embedding_default`가
  **단색(0)**이라 복제본이 없어 `node_to_replicated_node[parent][color]` KeyError.
  코드 가정("다색 노드의 부모는 전부 다색")이 dense 트랜스포머에서만 성립.
- 수정: `_resolve_parent` 헬퍼 — 부모가 단색이라 해당 색 복제본이 없으면 **원본 부모를
  그대로 참조**(값이 파티션 경계를 넘는 것은 split/파이프라인 I/O가 처리). 다색인데 색
  없으면 종전대로 에러. → **⑤ 통과.**

**[SDK 수정] qwen3_next.py DeltaNet 을 NPU-안전 op 으로 재작성** (컴파일 통과용)
빌드가 ⑥ 컴파일에 진입하며 DeltaNet 의 view-heavy op 들이 `aten.as_strided` 등으로
lowering 돼 거부됨. 하나씩 실측·수정:
1. **`repeat_interleave` → `torch.stack`+reshape** (repeat_interleave 는 expand→as_strided).
2. **융합 투영 `in_proj_qkvz`/`in_proj_ba` → 별도 Linear 6개**(`in_proj_q/k/v/z/b/a`).
   per-head view+split 이 strided 슬라이스→as_strided 를 만들기 때문. transform_weights 에서
   융합 가중치를 per-head 로 잘라 채움.
3. **융합 depthwise `conv1d` → q/k/v 별도 depthwise conv 3개**(`conv1d_q/k/v`). cat+split
   회피. conv1d.weight 도 채널 분리.
4. **`F.softplus` → `torch.log(torch.exp(x)+1)`** (softplus 가 aten.log1p 로 분해, log1p 미지원).

→ as_strided·log1p 전부 제거, op-import 전부 통과. 빌드가 **stage_0 컴파일 `[1/12]
dfg→primitive`** 까지 도달. **현재 벽: 내부 컴파일러 op "O945 is not yet supported"**
(stage_0 = 임베딩 + DeltaNet 레이어0 전처리; conv1d 또는 그 패딩(`constant_pad_nd` 미지원)
의심 — 진단 중). op 지원 문서: convolution 은 지원하나 conv1d decomp 부재·constant_pad_nd 불가.

**진전 요약:** ④ TP분할 → ⑤ transform.py(노드복제) → op-import(as_strided/log1p) →
⑥ NPU 컴파일러 백엔드 진입(O945에서 정지). 역대 가장 깊이 도달.

**[진단] 컴파일러 백엔드 미지원 op이 stage_0에 다수 (실측, 2026-06-10)**
conv1d 를 일시 우회한 진단 빌드 → **O945 사라지고 새 미지원 op O759 등장**(둘 다 stage_0).
즉 **DeltaNet 전처리만으로도 NPU 컴파일러 백엔드가 못 받는 내부 op이 여럿**:
- O945 = depthwise conv1d (또는 그 causal 패딩; convolution 자체는 지원되나 conv1d
  decomp 부재·constant_pad_nd 미지원).
- O759 = conv 외 전처리 op 중 하나(l2norm/stack-repeat/softplus-log/transpose 후보).
- 그리고 **핵심 순환 루프(_GatedDeltaNetCore, attention 스테이지)는 아직 컴파일도 안 됨**
  (stage_0에서 먼저 막힘).
→ op-import(Python FX)는 다 통과해도 **NPU 컴파일러 백엔드(Rust npu-compiler)의 내부
op 지원이 DeltaNet 의 특이 연산을 여러 군데서 못 받음**. 각 O-번호를 역추적해 회피
재작성해야 하고 끝점 불확실(근본 한계 가능성). 진단 후 conv 코드는 실제 버전으로 복원.

### 2026-06-10 (6차 세션) — ⭐⭐⭐ 끝까지 추적: NPU 컴파일러 커널라이저가 근본 벽

목표 "더 파고들어서 끝을 보자". transform.py 이후 ⑥ NPU 컴파일러 백엔드를 op 단위로
끝까지 파고든 결과, **근본 한계를 정확히 규명**.

**도구·방법:** `FURIOSA_COMPILE_DUMP_PATH` 로 IR 덤프 → `primitive_ir_viewer.json["operators"][N]`
로 "ONNN is not an operator" 의 N 을 실제 연산명+shape 로 해석(서브에이전트 규명).

**op 단위 추적 (각 수정 → 다음 벽, 전부 stage_0=DeltaNet 전처리):**
| O-op | 정체 | 수정 | 결과 |
|---|---|---|---|
| O945 | depthwise conv1d 의 causal 패딩(`constant_pad_nd`) | conv padding=0 + cat(zeros) 좌패딩 | 통과 |
| O957 | beta sigmoid [128,8] (좁은 마지막 차원) | transpose 먼저 → [8,128] | 여전히 실패 |
| O982 | beta sigmoid [8,128] (2D) | unsqueeze 로 3D | reshape_remover 가 size-1 제거 → 실패 |
| O1057/O1069 | beta sigmoid (stack-2 [8,2,128] 3D) | stack-2 mid | mid 너무 작아 실패 |
| O1565 | beta sigmoid **[8,64,128]** (conv sigmoid 와 동일 shape!) | mid=64 | **여전히 실패** |
| O1288 | gating 우회 후 **`mul_1` [128,512] (평범한 2D Mul)** | — | 실패 |

**결정적 결론:** shape·2D/3D·mid-size 와 **무관하게** standalone elementwise(Sigmoid,
**Mul** 까지)가 `[2/10] primitive→kernelized` 에서 거부됨. 즉 **RNGD 2026.2.0 NPU 컴파일러의
커널라이저는 standalone elementwise 연산을 커널로 내보내지 못한다 — elementwise 는 matmul/
conv 커널에 융합(fuse)될 때만 가능**. qwen3_moe 가 컴파일되는 건 그 elementwise(silu 등)가
matmul 에 융합되기 때문. DeltaNet(선형 어텐션 순환)은 gating·순환 스캔·gated norm 이 거의
전부 **standalone elementwise/reduction** 이라 op 마다 끝없이 거부 → **근본적 컴파일 불가**.

**이것이 진짜 "끝":** 빌드측 열린 Python 단계(①~⑤+op-import)는 전부 우리가 뚫었으나,
⑥ 폐쇄 Rust NPU 컴파일러(npu-compiler)의 커널라이저가 linear-attention 의 연산 패턴
자체를 다루지 못함. **벤더의 컴파일러 지원(linear-attention/recurrent 커널)이 필수** —
이는 serve 런타임이 순환 상태를 관리 못 하는 것과 **같은 뿌리**(NPU 스택이 트랜스포머
전용). Python 재작성으론 극복 불가. qwen3_next.py 의 gating 은 충실한 버전으로 복원하고
이 한계를 주석으로 명기.

### 2026-06-10 (7차 세션) — BF16 30B 위장 serve 검증 + DeltaNet 커널 주입 가능성 정직 검증

**#1 BF16 30B-A3B 빌드 확인:** `qwen3-coder-30b-a3b-inst-tp8-65k` 빌드 완료 확인
(qwen3_moe, W16A16KV16 BF16). compiled_graphs 캐시 존재.

**#2 BF16 위장본 생성+검증:** `qwen3-coder-30b-a3b-inst-tp8-65k-tc` 에 artifact.json 위장
(model_type qwen3_moe→qwen3, MoE 키 제거) + 나머지 파일 SRC→DST 이동(용량). KV 차원
(48L/kv4/hd128) 보존. **실측 검증:** 1장 serve는 OOM(BF16 60GB>47.5GB), **2장 pp2 serve
성공 + 정상 코드 생성**(reverse_string). 위장 게이트는 BF16 MoE도 통과 확인.

**#3 "컴파일러에 커널 넣으면 되나?" 정직 검증 (3-에이전트 + radare2 + 실측):**
- ⚠️ **지난 결론 정정:** `compiler.pyi` `is_supported_aten`에 sigmoid·mul·exp·log·rsqrt·
  cumsum·constant_pad_nd **다 있음** → "standalone elementwise 불가"는 부정확. 실패는 op
  미지원이 아니라 **DeltaNet 그래프 위치(context: Sub)에서 tactic 미할당**(elementwise를
  융합할 matmul 앵커 부재). 즉 부재한 건 op 커널이 아니라 **융합 gated-delta 스캔 tactic**.
- **radare2 커널 주입 = 불가:** npu-compiler(git 3f23a71)가 .so에 정적링크된 닫힌 컴파일
  코드. op지원=컴파일타임 enum+match+phf 정적테이블, 런타임 플러그인/dlopen 없음. 새 커널=
  새 lowering+tactic+TU/VE/DPE 코드젠(재배치 기계어 수천 바이트)→스트립 105MB에 손패치 불가.
- **config 플래그 = 불가(실측):** `allow_unlowered_operators=true`→빌드 **hang**(unlowered op
  은 DRAM IR에 남아 실행불가). `allow_external_operators`=이미 컴파일된 EDF 블롭만.
- **순수 Python matmul 재정식화 = 불가:** chunked로 cumsum→삼각matmul, pad→cat, mask→상수
  까지 되나 데이터의존 exp() decay 게이트·l2norm rsqrt가 상수로 안 접혀 standalone 잔존.
- **유일 정공법:** 벤더 npu-tools 소스로 gated linear-attention 스캔 tactic 커널 + 런타임
  순환상태 버퍼 추가(2026.3+). qwen3.5/3.6도 같은 구조라 동일 커널로 커버.
- 신규 문서: `info/README_qwen3_next_TECH.md` (구조 공부 + 커널 위치 + 정직한 결론).

### 2026-06-10~11 (8차 세션) — 🚀🚀 한계 돌파: Gated DeltaNet 레이어를 RNGD NPU에서 실제 계산

7차의 "벤더 전용" 결론을 사용자 독려("한계 한번 뚫어보자")로 더 파고든 결과, **틀렸음을
실증으로 뒤집음.** Rust 소스 없이도 커널을 손수 작성해 NPU에서 돌릴 수 있었고, 실제 DeltaNet
레이어를 NPU에서 계산해 HuggingFace 레퍼런스와 ~1e-7로 일치시킴.

**(A) TacticKernelModule 발견 (Rust 소스 없이 커널 작성 경로):**
- `furiosa.torch.TacticKernelModule(yaml_dsl)` → `Dfg.parse` → `torch.ops.furiosa.dfg` 커스텀op
  → `torch.compile(m, backend=furiosa.torch.backend)` → EDF 컴파일 + NPU 로드·실행.
- Furiosa 자신이 `models/core/operators/tk_graphs/moe_blockwise_compute_wg_idx.yaml` MoE
  work-group-index 커널을 이 방식으로 작성·사용 중. DSL = `#naive_yaml` (tensors/inputs/
  outputs/operators; SymTacticKernel + Loop 노드).
- 실측: 커스텀 elementwise add 커널을 작성→rngd:0서 정답, CPU폴백 0회(NPU 실행 검증).
- ⚠️ 저수준 `compiler.compile(ExportedProgram)`은 `furiosa::dfg` 재import 실패 → 고수준
  dynamo backend 경로만 AOT-EDF. 입력없는 SymArange-only는 NPU 컴파일서 segfault(실입력 ≥1 필요).

**(B) DeltaNet 빌딩블록 fp32 커널 — 전부 NPU 검증:**
- `dn_einsum_f32.yaml`: contraction `Σₖ S[k,v]·k[k]` (EinsumByVe+tiles broadcast read0+
  LocalReduceAddF) — torch.einsum 일치, err 2.4e-7.
- `dn_rank1.yaml`: rank-1 외적 `S+=k⊗δ` — **EinsumByVe를 Reduce inst 없이** 쓰면 외적(단일
  Elementwise 2-broadcast는 EDF가 Cpu노드로 떨궈 실패) + Elementwise AddF. 정확 일치.
- `dn_decay.yaml`·`dn_delta.yaml`: `S·decay`, `(v−kv)·β` — 정확 일치. 스칼라는 caller가
  `torch.full`로 동형화(진짜 [1] broadcast 미지원) 또는 `{ConstFloat}`.
- DSL핵: fp32는 `MulF/AddF/SubF`+`LocalReduceAddF`(*Fxp는 정수전용). broadcast read=read0.
  contraction은 공유라벨이 출력에 없으면 축약축. 텐서id 전역 flat, op-local Tensor:0/1 분리.

**(C) ⭐ 풀 delta-rule 스텝 + 레이어 — NPU 실행 성공:**
- `dn_step.yaml`: delta-rule **한 스텝 전체(7-op 융합 DFG)** — S1=S·decay → kv=Σₖ S1·k →
  delta=(v−kv)·β → Sout=S1+k⊗delta → out=Σₖ Sout·q. rngd:0서 Sout·out 둘 다 torch ref 일치
  (err<1.2e-7), CPU폴백 0. 첫 시도 성공.
- **host-loop (T=8)**: 스텝커널을 8토큰에 상태 스레딩 → HF `torch_recurrent_gated_delta_rule`
  일치(err 6e-8), 8/8 NPU 실행. (q는 1/√d_k 사전스케일 필요 — HF는 루프 전 query*scale.)
- **`dn_prefill_unroll4.yaml`**: 스텝 7-op를 **T=4 언롤 → 28-op 단일 DFG/EDF, 한 번의 forward**
  → HF ref 일치(err 1.2e-7). 단일-forward T-토큰 prefill 커널 성립. `gen_unroll.py`가 T 파라미터화.

**(D) 유일 잔존 vendor-lock = 네이티브 `Loop` 노드** (`dn_loop2.yaml`):
- 정적 `UnlabeledShape{[1]}`(정수 sizes)로 첫 벽(SymExpr 상수) 통과 발견. 하지만
  `loop_impl.rs:126`이 loop_index를 **스케줄러산 SpmShape 스칼라**로 요구 → naive_yaml 불가.
  CPU·NPU 양쪽 동일 게이트. **그러나 언롤/host-loop이 Loop 노드를 불필요화** → 우회됨.

**정정된 최종 결론:** 컴퓨트 벽은 **완전 돌파**(radare2/config/Python-reform "불가"는 맞으나
**TacticKernelModule 손작성으론 가능**). 남은 건 **엔지니어링**: ①실차원 스케일(검증은 d=4 미니,
실제 d=128·32헤드·36레이어) ②긴 seq 청크화(언롤은 그래프가 T에 선형) ③furiosa-llm serve 통합
(prefill 단일forward 가능, decode cross-step 상태는 paged-KV 한계→커스텀 host루프 or 벤더 상태풀).
- 산출물: `qwen3-next-proj/tk_kernels/` (커널 YAML 8종 + 드라이버 .py). 문서:
  `README_qwen3_next_TECH.md` 7·8차 절.

### 2026-06-11 (9차 세션) — 🚀 실차원·멀티헤드·청크 형태 스케일 (남은 일 ①②③ 해결)

8차 돌파(미니 d=4)를 실모델 규모로 확장. 전부 NPU 검증:
- **실차원**: `dn_step.yaml`이 d_k=d_v=128서 **무변경** 동작(symbolic Var:K,V 자동추론). `scale_test_d128.py` torch 일치 오차 8.6e-6, CPU폴백 0.
- **멀티헤드**: `dn_step_mh.yaml` H=4·d=128 4헤드 동시 NPU 실행, HF식 per-head 일치(오차 1.5e-5). head=batch축(모든 shape 최외곽 라벨, reduce/tile 안 함). 한 번에 성공.
- **언롤 한계**: `gen_unroll.py` 파라미터화, `dn_prefill_unroll{8..128}.yaml`. 컴파일은 T=128(896op)까지 NPU OK이나 시간 초선형(T64≈110s,T128≈449s). 정확도는 fp32 누적으로 **T~8 초과 drift** → 긴 seq는 청크 필수.
- **청크 형태 한 청크**: `dn_chunk.yaml` chunk-parallel gated delta rule, **3 matmul 코어 전부 NPU**, 실config C=64/d=128 오차 7.6e-6. **행렬-행렬 einsum `ck,dk→cd`가 NPU 내려감**(8차 일반화). 새 gotcha: no-reduce EinsumByVe는 양쪽 1D broadcast(외적)만 NPU, broadcast-1D×full-2D는 Cpu노드→graph분할 실패→2D materialize+Elementwise MulF 우회.

### 2026-06-11 (10차 세션) — 🚀 완전한 DeltaNet 레이어 컴퓨트 조각 전부 NPU 검증

레이어를 이루는 모든 컴퓨트를 NPU에서 만들고 HF와 대조 완료(④ 핵심):
- **멀티청크 스캔**(`dn_chunk_full.yaml`, 12-op): **inter-chunk 상태 carry 포함**(attn_inter·v_prime·v_new·S_next). host-loop NC=3 청크 → HF `torch_chunk_gated_delta_rule` 일치 out 1.5e-8·state 3e-8, **carry 실검증**(S_prev≠0), CPU폴백 0. 삼각역행렬 정련은 S_prev 무의존이라 host 사전계산, **S_prev 닿는 4 matmul만 NPU**. q,k **L2정규화 필수**(안 하면 state 1e14 폭발).
- **conv1d+SiLU**(`dn_conv1d.yaml`, 1.4e-6) · **l2norm**(`dn_l2norm.yaml`, 6e-8) · **gated RMSNorm**(`dn_gnorm.yaml`, 2.9e-6) — 전부 NPU.
- DSL 실측: 유효 Unary=Exp/Sigmoid/Sqrt만(native rsqrt 없음→√s/s), Binary=MulF/AddF/SubF/DivF. reduction은 reduce축 외곽+생존축 내곽≥128일 때만 NPU. 융합 inst ≤2개.
- → **레이어 모든 컴퓨트(투영·conv·l2norm·청크스캔·gated norm) NPU 검증 완료.** 남은 일: 완전 레이어 forward 조립(HF Qwen3NextGatedDeltaNet 대조) + serve 통합.

### 2026-06-11 (11차 세션) — 🏆 캡스톤: 완전한 DeltaNet 레이어가 NPU에서 HF와 ~1e-7 일치 (적대적 검증)

조각들을 `full_layer.py`(host-오케스트레이션)로 **완전한 단일헤드 Gated DeltaNet 레이어**로 조립 →
**HF `Qwen3NextGatedDeltaNet` 전체와 대조 성공**, 별도 에이전트 적대적 재검증 통과.
- 결과: allclose=True **maxerr 3.9e-7**(T=32/2청크), T=64/4청크 2.7e-7. 총 `_dfg_inner=0`.
- NPU 스테이지: conv1d+SiLU·l2norm(q,k)·beta=sigmoid·**멀티청크 스캔(상태 carry, 4청크 검증)**·gated RMSNorm(z-게이팅 실재). 전부 dfg_delta=0.
- 적대적 검증: spy 진짜 CPU경로(count=0=NPU)·진짜 HF torch 경로·오차 1e-7(오염 시 1.2 민감)·미달 reduction은 에러(가짜통과 불가).
- caveat(검증): host 실행 = in_proj/out_proj matmul(NPU-compilable)·softplus 스칼라(log 값오류로 native 불가)·청크 사전계산(tri-inverse/cumsum). 단일헤드·소차원.
- → **DeltaNet 고유 컴퓨트는 완전 NPU 실행.** 남은 일: 투영/softplus/사전계산 NPU화·멀티헤드/실차원 스케일·full-model 통합(full-attn+MoE는 표준)·serve(decode cross-step 상태).

### 2026-06-11 (12차 세션) — 🏁 투영까지 NPU(96.74%) + serve 경로 확정

- **(A) 최대한-NPU 레이어**: `dn_linear.yaml`(nn.Linear=EinsumByVe matmul 'ti,oi→to') 신규 → `full_layer_npu.py`에서 in_proj/out_proj까지 NPU. HF 대조 maxerr 1.6e-6, 총 `_dfg_inner=0`, **matmul FLOP 96.74% NPU**. host 잔여 3.26%는 순차 삼각역행렬 T 사전계산 + softplus 스칼라(본질적 한계)뿐.
- **(B) serve 경로 확정**(코드 정독): ❌KV-위장(런타임이 슬롯을 append식 소유, RMW 불가, aliasing Python 훅 없음, SSM 기구 전무) / ✅furiosa-llm serve 밖 커스텀 host 추론루프(host가 상태 RMW 보유, NPU per-step 커널 — 권장) / ⚠️serve 안은 벤더 전용(2026.3+).
- → **DeltaNet 레이어 컴퓨트 사실상 완성(96.74% NPU·HF 일치). 배포는 host 추론루프로 가능.** 남은 큰 빌드 = full-model host 추론루프(컴퓨트 전부 증명됨, 실가중치 엮기).

### 2026-06-11 (13차 세션) — 🏆🏆 실제 Qwen3-Coder-Next-FP8(80B)가 RNGD NPU에서 코드 생성

사용자 목표(청사진→253GB 가능성→실가중치→serve)로 실제 80B 모델을 NPU에서 돌림.
- **청사진**: `info/README_qwen3_next_RUN.md`. **용량 검토**: FP8 80.4GB(253GB 디스크 OK, 125GB RAM은 mmap/레이어 스트리밍).
- **다운로드**: Qwen/Qwen3-Coder-Next-FP8 75GB(40 shard). **로더** `qcn/loader.py`(safetensors mmap + FP8 blockwise dequant).
- **컴포넌트(실가중치 HF 대조, 전부 NPU `_dfg_inner=0`)**: DeltaNet `qcn/deltanet_layer.py`(16/32헤드 d128) 8.9e-8 · full-attn `qcn/attn_layer.py`(GQA16/2 hd256 partial-RoPE 게이트, matmul 100% NPU) 5.96e-7 · MoE `qcn/moe.py`(512expert top10+shared) 1.79e-7.
- **전체 모델** `qcn/model.py`(48레이어 스트리밍): 첫 4레이어 실HF 대조 ~1e-6, 전 mixer NPU.
- **생성** `qcn/generate.py`: `def quicksort(arr):` → 올바른 quicksort 코드 생성. prefill 360s, decode 24토큰 55.8s/tok, NPU stages deltanet 36000·attn 10824·moe 59336, CPU폴백 0. prefill↔decode 일관성 1e-7.
- **#4 serve(A안 완료)** `qcn/serve.py`(FastAPI OpenAI 호환): `/v1/completions`(`def add(a,b):`→`\n    return a`)·`/v1/chat/completions`(chat 템플릿, →`` ```python\ntotal ``) 둘 다 NPU에서 작동. native furiosa-llm serve(B안)는 벤더 전용(append-only KV) 확정 → A안이 현실적 프로덕션 경로.
- → **🏆 청사진→253GB 가능성→실가중치 NPU 코드생성→OpenAI 호환 서빙까지 end-to-end 완성.**

### 2026-06-11 (14차 세션) — 🚀 성능: 배치는 퇴행, EinsumByDpe(systolic matmul)가 진짜 레버

- **배치 최적화 = 음성 결과**: head/expert를 batch-axis로(dn_chunk_full_mh·dn_linear_be) 디스패치 32×·29.8× 줄이고 bit-identical 정확하나 **wall-clock 16× 퇴행** — RNGD matmul=벡터엔진 EinsumByVe(outer product materialize 후 reduce)라 배치가 일감만 N배(systolic 아님). 검증된 per-head/per-expert로 복원(model.py).
- **🚀 EinsumByDpe 돌파**: "미해결 frontier"였던 DPE(MAC/systolic) 엔진을 깸. 3각도 정찰(컴파일러 DFG CBOR 디코드·serde reloc 역설계·점진 에러추적 11회). 레시피: kind VE→DPE, contraction을 `ein_ops.reduce`로, vector_ops=빈-axes Reduce identity. `dn_linear_dpe.yaml` NPU 실행·torch 1e-2 일치·**VE 대비 compute 3.8×**.
- **모델 전체 DPE 적용**(`QCN_DPE=1`): prefill 360.8s→**76.9s(4.69×)**, decode 55.8→**35.15 s/tok(1.59×)**, 투영 matmul 9.06×. 생성 여전히 올바른 quicksort(byte-identical), 4레이어 HF atol 1e-2. serve.py는 DPE 기본 ON.
- 제약: DPE per-graph 2개 cap(3개+ 융합 오컴파일), 출력축 32배수 pad, bf16 ~0.23% rel. decode는 host-bound(cumsum/tri-inverse) 부분 때문에 1.59×에 그침.
- 산출물: `dn_linear_dpe.yaml`·`dn_chunk_full_dpe2.yaml`·`dpe_*.md`·`bench_dpe_vs_ve.py`·`validate_*_dpe.py`·`perf_dpe.json`. `dn_chunk_full_mh`·`dn_linear_be`는 배치 음성결과 참고용.


### 2026-06-11 (15차 세션) — 📚 문서 통합 + 셸/자원 정리

- **qwen3_next md 4개 → 2개 통합**(사용자 요청, 내용 손실 0): `README_qwen3_next_feasibility.md`·`README_deltanet_kernel_study.md`·`README_radare2_gate_analysis.md`·`README_full_model_blueprint.md` → **`README_qwen3_next_TECH.md`**(기술 전말: 빌드벽·radare2 게이트·TacticKernelModule·DPE 등 6절+부록) + **`README_qwen3_next_RUN.md`**(실행·사용 가이드: 용량·아키텍처·로더·generate/serve 커맨드·성능). 적대적 검증으로 ~230개 구체사실(maxerr·perf·file:line·radare2 주소·레시피·제약) 전수 점검, 2개 누락 복원. 원본 4개 삭제, 참조(메모리·ALL_about·본 로그) 갱신.
- **#2 답(furiosa-llm 빌드 아티팩트 가능?)**: ❌ 불가. 정식 빌드는 DeltaNet 커널화 벽으로 막히고, 위장(-tc)은 DeltaNet 가중치를 qwen3_moe 어텐션으로 계산→garbage. `qcn/` host루프+손수커널이 유일 경로.
- 셸 정리: 고아 모니터 sleep 정리, NPU 4장·포트 해제. code-server 인터랙티브 터미널은 보존.

### 2026-06-15 (16차 세션) — 🔬 NPU 실행 4중 재증명 + DeltaNet 분해로 a6 번들 완성 + SDK 문서 감사

- **#1 NPU 실행 재증명(4중, 실측)**: 80B를 `rngd:4`에서 3토큰 생성하며 `furiosa-smi`+host CPU 동시 샘플링. (1) CPU-폴백 카운터(`_dfg_inner` spy) **5760 디스패치 전부 0**(deltanet/attn/moe), (2) `furiosa-smi ps`가 생성 구간 내내 PID↔`npu0:4` 바인딩·종료 즉시 해제, (3) host CPU ~3415%(≈34코어)는 weight스트리밍+FP8역양자화+torch글루(폴백 아님, 1번이 증명), (4) DPE matmul **0.23% rel** = bf16 systolic 하드웨어 지문(CPU fp32면 VE처럼 ~3e-7). power/util 평탄(39W)은 지표 둔감+host-bound — 4000회 matmul 대조에서도 평탄. 드라이버/샘플러: `tmp/qcn_npu_proof.py`·`npu_sampler.py`·`fast_sampler.py`. → TECH §7-1.
- **#3 DeltaNet 분해 돌파**: a6 불가였던 3조각의 진짜 원인은 연산이 아니라 **그래프 구성** — 한 그래프 내 복수 contraction('conflict between concrete labels')·복수 독립출력('multiple internal subgraphs')·미지원op(Conv1d→O136, softplus→log1p)를 거부. "연산 1개=그래프 1개"로 쪼개니 전부 통과(통과한 SDPA/Linear와 동일 구조). recurrent step→`dn_recur_decay/contract/delta/outer/add`(fp64 정확·fp32 rel 2.9e-7), conv1d→`dn_conv1d_shift`(host-pad+shift-mul-add+SiLU), gate→`dn_gate_beta`+`dn_gate_g`(log(1+exp)). 실측: `tmp/dn_decompose_probe.py`·`probe2`.
- **번들 통합**: 8개 분해 블롭을 실 config로 컴파일해 추가(`tk_kernels/emit_dn_split_blobs.py` 신규 → `pack_edf_bundle.py` 재패킹). binary_bundle **17→25블롭**, kind `partial-edf`→**`edf-split (compute-complete)`**, `pieces_without_edf` 3→0, **25/25 a6 역직렬화 검증**(515.3MB, sha256 `d740ea47…`). artifact.json·manifest 갱신. **남은 한계는 컴파일이 아니라 deploy**(serve 런타임 순환상태 풀 없음, 벤더 2026.3+).
- **#2 SDK 문서 감사**: build/serve 실행 파일·모델→NPU 변환·아티팩트 서빙 기록이 충분히 상세한지 SDK 소스 대조 재확인 → **3영역 모두 THOROUGH**(`ALL_about_build_serve.md`가 전 체인을 file:line으로 실측 기록, ~40개 인용 대부분 정확). `presets.py`/`api.py` 편집 이후 stale였던 인용 6건 수정: `ALL_about`(LLM.__init__ `api.py:115`/호출 `:216`, load_llm_from_args `models.py:12`, `api.py:354 override_with` 추가), `BUILD_FLOW`/`BUILD_COMPIL`/`README_config`(presets.py `:425`), `README_preset`/`README_build`(PRESET_REFS `:277`, 7종→15항목/11 preset).
- **이론 공부(gated DeltaNet/attention)**: 신규 `README_gated_deltanet_STUDY.md` — delta rule/SGD·청크 WY/UT·게이트 delta(식10)·gated attention·Qwen3-Next 구조·paged-KV 불가 이유를 식 유도+논문/코드 file:line으로 정리.

## 파일 변경 목록 (누적)

| # | 일자 | 파일 | 변경 | 사유 |
|---|---|---|---|---|
| 1 | 2026-06-10 | `info/README_all_change.md` | **신규** | 프로젝트 전체 변경 기록 (사용자 요구) |
| 2 | 2026-06-10 | `qwen3-next-proj/make_mini_model.py` | **신규** | 미니 합성 모델 생성 하니스 |
| 3 | 2026-06-10 | `qwen3-next-proj/mini_models/{qwen3,qwen3-moe,qwen3-next}` | **신규** | 파이프라인 검증용 장난감 모델 3종 |
| 4 | 2026-06-10 | `~/furiosa/.../furiosa_llm/artifact/presets.py` | **수정** (MINI_SMOKE_PRESET + ref 3개) | 미니 preset 매칭. 기존 엔트리 무변경 |
| 5 | 2026-06-10 | `~/furiosa/.../furiosa/models/language/architecture/qwen3_next.py` | **신규** | qwen3_next 아키텍처(빌드 ④에서 막힘, probe·차기SDK 대비) |
| 6 | 2026-06-10 | `~/furiosa/.../furiosa/models/language/__init__.py` | **수정** (import+`__all__` 2줄) | qwen3_next 클래스 등록 |
| 7 | 2026-06-10 | `~/furiosa/.../furiosa_llm/metadata/hf_utils.py` | **수정** (`_EXPERIMENTAL_MODEL_TYPES` 게이트 우회) | qwen3_next 빌드 진입 허용. 목록 외 model_type 은 종전대로 |
| 8 | 2026-06-10 | `qwen3-next-proj/masquerade_artifact.py` | **신규** | serve 게이트 위장 도구 |
| 9 | 2026-06-10 | `info/README_qwen3_next_TECH.md` | **신규** | 종합 결론 보고서 |
| 10 | 2026-06-10 | `qwen3-next-proj/artifacts/{mini-qwen3-tp8, mini-qwen3-moe-tp8}` | **신규 빌드** | dense·MoE 미니 아티팩트 |
| 11 | 2026-06-10 | `qwen3-next-proj/artifacts/{mini-qwen3-moe-masq, qwen3-coder-30b-masq}` | **신규** (위장 사본) | 게이트 우회 실증. 원본 무변경(하드링크) |
| 12 | 2026-06-10 | `~/furiosa/.../furiosa_llm/parallelize/pipeline/builder/transform.py` | **수정** (`replicate_nodes_with_multiple_colors` 단색부모 fallback) | 5차: ⑤노드복제 통과. 다색노드의 단색부모는 복제본 없어 원본참조. 가역 |
| 13 | 2026-06-10 | `~/furiosa/.../furiosa/models/language/architecture/qwen3_next.py` | **수정**(누적, 6차 충실복원) | DeltaNet NPU-안전 재작성(별도투영6·별도conv3·cat패딩·softplus=log(exp+1)·stack-repeat·게이트 transpose). 컴파일 추적 흔적, 커널주입 한계 주석 |
| 14 | 2026-06-10 | `info/ALL_about_build_serve.md` | **신규** | 3차: build/serve 전과정 file:line 실측 정리 |
| 15 | 2026-06-10 | `info/README_qwen3_next_TECH.md` | **신규** | 4차: serve 게이트 2겹 radare2 해부 |
| 16 | 2026-06-10 | `qwen3-next-proj/radare2/` | **신규**(소스빌드) | radare2 6.1.7 게이트 분석용 |
| 17 | 2026-06-11 | `qwen3-next-proj/artifacts/qwen3-coder-30b-a3b-inst-tp8-65k-tc/` | **신규**(위장+파일이동) | 7차: BF16 30B `model_type qwen3` 위장. SRC엔 artifact.json만 남김. 2장 pp2 serve 검증 |
| 18 | 2026-06-11 | `qwen3-next-proj/build_with_override.py` | **신규** | 7차: compiler_config_overrides 주입 빌드 테스트(allow_unlowered → hang 실증) |
| 19 | 2026-06-11 | `info/README_qwen3_next_TECH.md` | **신규**(7·8차 갱신) | DeltaNet 구조+커널위치+TacticKernelModule 돌파 기록 |
| 20 | 2026-06-11 | `qwen3-next-proj/tk_kernels/*.yaml` (8종) + 드라이버 `.py` | **신규** | 8차: 손작성 TK-graph 커널. `dn_step`(스텝 7-op)·`dn_prefill_unroll4`(언롤 단일EDF)·`dn_einsum_f32`·`dn_rank1`·`dn_decay`·`dn_delta`·`dn_gate`·`dn_loop2` + `gen_unroll.py` 등. 전부 NPU 검증 |
| 21 | 2026-06-11 | `qwen3-next-proj/tk_kernels/dn_step_mh.yaml`·`dn_chunk.yaml`·`dn_prefill_unroll{8..128}.yaml`·`scale_test_d128.py`·`gen_chunk.py`·`mh_test.py`·`unroll_limit_test.py` | **신규** | 9차: 실차원 d=128·멀티헤드 H=4·청크형태 NPU 검증 |
| 22 | 2026-06-11 | `qwen3-next-proj/tk_kernels/dn_chunk_full.yaml`·`dn_conv1d.yaml`·`dn_l2norm.yaml`·`dn_gnorm.yaml` + `gen_chunk_full.py`·`run_dn_chunk_full.py`·`gen_dn_layer.py`·`test_dn_layer.py`·`probe_unary.py`·`probe_binary.py` | **신규** | 10차: 멀티청크 스캔(inter-chunk carry)+conv1d/l2norm/gated RMSNorm, 전부 NPU·HF 대조 |
| 23 | 2026-06-11 | `qwen3-next-proj/tk_kernels/full_layer.py` | **신규** | 11차 캡스톤: 완전한 DeltaNet 레이어 조립, HF 전체 대조 maxerr 3.9e-7, DeltaNet 고유 op 전부 NPU. 적대적 검증 통과 |
| 24 | 2026-06-11 | `qwen3-next-proj/tk_kernels/dn_linear.yaml`·`full_layer_npu.py`·`test_dn_linear.py` | **신규** | 12차: nn.Linear NPU matmul 커널 → in_proj/out_proj까지 NPU. 레이어 96.74% FLOP NPU, HF maxerr 1.6e-6 |
| 25 | 2026-06-11 | `info/README_qwen3_next_RUN.md` | **신규** | 13차: 전체 모델 host 추론 청사진 + 실행 진행 |
| 26 | 2026-06-11 | `qwen3-next-proj/qcn/` (`loader.py`·`deltanet_layer.py`·`attn_layer.py`·`moe.py`·`model.py`·`generate.py` + validate/sample) | **신규** | 13차: 실제 Qwen3-Coder-Next-FP8 전체 모델 NPU 추론. quicksort 코드 생성 성공 |
| — | 2026-06-11 | HF 캐시 `Qwen3-Coder-Next-FP8` (75GB) | **다운로드** | 실가중치(SDK 무관, ~/.cache/huggingface) |
| 27 | 2026-06-11 | `qwen3-next-proj/qcn/serve.py` | **신규** | 13차 #4: OpenAI 호환 서빙(A안). /v1/completions·/v1/chat/completions NPU 작동 |

> **SDK 수정 6건(#4~#7, #12, #13) 모두 가역적**입니다. 되돌리는 법은 feasibility 보고서 6절 표 참조.
> 원본 모델/아티팩트 가중치는 일절 수정하지 않았습니다. tk_kernels(#20)는 SDK 외부 신규 산출물.
