# Qwen3-Coder-Next on RNGD — 실현 가능성 결론 보고서

작성 2026-06-10 · SDK furiosa-llm 2026.2.0 · RNGD 4장(각 47.5GiB HBM, 펌웨어 2026.2.1)
변경 이력 전체: [README_all_change.md](README_all_change.md)

---

## 🚀 갱신 (2026-06-11) — 아래 "최종 결론"은 부분 반증됨

아래 6차 결론("NPU 컴파일러 커널라이저가 근본 벽, 벤더 전용")은 **컴퓨트 측면에서 뒤집혔습니다.**
`furiosa.torch.TacticKernelModule`로 TK-graph 커널을 손수 작성해 **Gated DeltaNet 레이어를
RNGD NPU에서 실제 계산**, HuggingFace 레퍼런스와 ~1e-7 일치(host-loop T=8 + 언롤 단일EDF T=4).
즉 **컴퓨트 벽은 돌파**됐고, 남은 건 실차원 스케일·긴 seq 청크화·serve 통합(엔지니어링) +
네이티브 Loop 노드/serve cross-step 상태풀(벤더). 상세: **[README_deltanet_kernel_study.md](README_deltanet_kernel_study.md)
8차 절**. (아래 6차 분석은 "FX→kernelize 경로"에 한정해 여전히 유효 — 그 경로로는 막히지만
TK-graph 직접작성으론 통과.)

---

## ⚠️ 최종 결론 (2026-06-10, 끝까지 추적 완료 — 위 갱신으로 일부 반증)

빌드 파이프라인을 **op 단위로 끝까지** 밀어붙인 결과, qwen3_next 의 진짜 벽은
**RNGD NPU 컴파일러(폐쇄 Rust `npu-compiler`)의 커널라이저**임이 확정됐습니다:

- 빌드측 **열린 Python 단계는 전부 통과**시킴: ②빌드 게이트 우회, ④TP 분할(자력),
  ⑤transform.py 노드복제(자력), op-import(as_strided/log1p/constant_pad_nd 제거).
- 그러나 ⑥ `[2/10] primitive→kernelized` 에서 **standalone elementwise 연산(Sigmoid,
  심지어 평범한 Mul `[128,512]`)을 커널로 못 냄** — shape·2D/3D·mid-size 무관(`O945→O957→
  O982→O1057→O1565→O1288` 순으로 끝없이 막힘, IR 덤프로 실측 해석).
- 원리: NPU 커널라이저는 **elementwise 를 matmul/conv 커널에 융합**할 때만 처리. qwen3_moe
  가 컴파일되는 이유. DeltaNet(선형 어텐션)은 gating·순환 스캔·gated norm 이 거의 전부
  **standalone elementwise/reduction** 이라 근본적으로 컴파일 불가.
- 이는 serve 런타임이 순환 상태를 관리 못 하는 것과 **같은 뿌리**(NPU 스택이 트랜스포머
  matmul/conv 중심). **벤더의 linear-attention/recurrent 커널 지원이 없으면 불가** —
  Python 재작성으로 극복 불가능.

추적 상세는 README_all_change.md "6차 세션" 표 참고.

---

## TL;DR

- **`Qwen/Qwen3-Coder-Next` (80B, model_type `qwen3_next`) 의 충실한 빌드·서빙은
  2026.2.0 에서 불가능**합니다. 막는 벽은 "연산 미지원"이 아니라 **구조**입니다:
  (a) TP 그래프 분할기가 *모든 레이어가 페이지드 (K,V) 캐시를 소비*하도록 강제 →
  KV 를 안 쓰는 DeltaNet 레이어(48개 중 36개)는 dead node 를 만들어 분할기가 깨짐,
  (b) serve 런타임(폐쇄 Rust 바이너리)이 DeltaNet 순환 상태를 유지·관리할 수 없음.
  둘 다 미니 모델로 **실측 확인**.
- **대신, "RNGD 4장으로 더 강력한 오픈소스 코더"라는 본래 목표는 달성**했습니다.
  메타데이터 위장(masquerade) 기법으로 이전에 serve 패닉으로 사장됐던
  **Qwen3-Coder-30B-A3B-Instruct-FP8** (Qwen3세대 MoE 코더)를 부활시켰고,
  4장 dp 서빙에서 **단일 63 tok/s, 동시 32명 합산 1036 tok/s** + 정상 코드 출력을
  실측했습니다. (기존 dense Qwen2.5-Coder-14B 단일 30.7 tok/s 대비 큰 향상)

---

## 1. Qwen3-Coder-Next 모델 특성 (HF 실측)

| 항목 | 값 |
|---|---|
| 파라미터 / 크기 | 79.7B / BF16 159.4GB (safetensors 40개) |
| model_type | `qwen3_next` (`Qwen3NextForCausalLM`) |
| 레이어 | 48 = **36 Gated DeltaNet(선형 어텐션) + 12 gated full attention** (`full_attention_interval=4`) |
| MoE | 512 experts, top-10, moe_inter 512, shared expert 512, 전 레이어 |
| 어텐션 | 16 heads / 2 KV heads / head_dim 256 / partial rotary 0.25 / q·k norm |
| 컨텍스트 | 262144 |

DeltaNet 레이어는 토큰마다 갱신되는 **고정 크기 순환 상태**(레이어당 recurrent
`(num_v_heads, head_k, head_v)` + causal-conv 상태 `(conv_dim, kernel-1)`)를 유지해야
자기회귀 디코딩이 성립합니다. 이는 트랜스포머의 append-only KV 캐시와 **접근 패턴이
근본적으로 다릅니다**(read-modify-write vs append).

## 2. furiosa-llm build / serve 유기적 관계 (실측 규명)

```
furiosa-llm build <model> <out> -tp 8
  └ ArtifactBuilder
      ① validate_model_support (metadata/hf_utils.py)         ← model_type 게이트(빌드측)
         └ native find_compiler_config(model_type, task) 가 None 이면 즉시 ValueError
      ② resolve (presets.find_preset 로 버킷 채움)
      ③ Pipeline build = FX 트레이싱 (parallelize/new_pipeline_builder.py)
         └ 각 버킷마다 모델 forward 를 그래프(IR)로
      ④ TP 그래프 분할 (parallelize/graph_partitioner.py)      ← qwen3_next 가 깨진 곳
         └ 레이어별 (K,V) 입력을 파티션 단위로 매핑
      ⑤ Compile = supertask 단위로 NPU 컴파일러 → EDF 바이너리
      ⑥ artifact.json + binary_bundle.zip + params/ 저장

furiosa-llm serve <artifact> --devices ... [-dp/-pp]
  └ api.py:_init_from_artifact
      ⓐ NativeLLMEngine() 생성 시 model_type 게이트(serve측)          ← 위장으로 통과
         └ artifact.json 의 model_metadata.model_type 만 화이트리스트 검사
            허용: {llama, exaone4, qwen2, qwen3, qwen3_moe, gpt_oss, embed, score}
            (furiosa-generator/.../hf_compat_next_gen.rs:367)
      ⓑ KV 캐시 할당: pipeline_metadata 버킷 kv_cache_size 기준, hf_configs
         (num_hidden_layers/num_key_value_heads/head_dim) 로 shape 검증
      ⓒ -dp = 1장용 tp8 아티팩트를 여러 장에 복제(처리량↑),
         -pp = 스테이지를 장에 분할(큰 모델용). 둘 다 model_type 무관(기하학적).
```

게이트가 둘(④ 빌드 / ⓐ serve)이라는 점이 핵심입니다. **연산은 ⑤에서 EDF 로 다
구워지므로, 게이트만 통과하면 런타임은 그래프를 그대로 실행**합니다 — 이것이 위장이
동작하는 이유이자, qwen3_next 가 ④에서 막히는 이유입니다.

### 2-1. serve 측 KV 캐시 바인딩의 정밀 메커니즘 (2026-06-10 추가 실험으로 정정)

서브에이전트가 mini 아티팩트의 hf_configs `num_hidden_layers` 를 1/2/48 로 바꿔가며
직접 serve 실험(npu:1·npu:2, temperature=0)한 결과:

- **KV 캐시 버퍼의 수·모양·메모리 예산은 전적으로 아티팩트 `pipelines[].tensors` 중
  `origin.name=="kv_caches"` 텐서 목록에서 파생** (KVCachePlan·블록 수·출력이
  nhl=1/2/48 모두 바이트 동일). 본 문서 초판의 "런타임이 hf_configs 기준으로 캐시를
  할당·검증한다"는 서술은 **반증됨** — num_hidden_layers 교차검증은 없음.
- hf_configs 는 ①필수 필드 존재 파싱(model_type, num_hidden_layers, num_attention_heads,
  max_position_embeddings, hidden_size, vocab_size, eos_token_id, architectures),
  ②model_type 화이트리스트 게이트, ③`layer_types` 값 검증, ④(JIT 켰을 때만) 컴파일러
  설정에 쓰임.
- ⚠️ **`layer_types` 함정**: Rust hf_config 파서는 `sliding_attention`/`full_attention`
  값만 허용 — **`linear_attention` 이 남아 있으면 "unexpected layer_types" 패닉**.
  실제 Qwen3-Coder-Next config 의 layer_types 에는 linear_attention 이 36개 들어있으므로
  위장 시 반드시 제거하거나 전부 full_attention 으로 재작성해야 함.
- ⚠️ **prefix cache 함정**: serve 기본값이 prefix cache ON. 페이지드 KV 만 복원하므로
  비-KV 순환 상태를 가진 하이브리드 모델에서는 캐시 히트 시 **조용히 틀린 출력**이
  됨. 하이브리드 serve 실험 시 prefix cache 비활성화 필요.
- 함의: 레이어 일부만 KV 캐시를 갖는 하이브리드 아티팩트(12/48만 선언)도 serve 측
  바인딩 관점에서는 **수용 가능** — 막는 것은 바인딩이 아니라 순환 상태의 프레임 간
  유지(아래 4절)다.

## 3. qwen3_next 빌드 실패 계단 (미니 모델 실측)

미니 합성 qwen3_next(4레이어=3 DeltaNet+1 full, 8 experts, ~175M)로 한 단계씩 통과:

| 단계 | 결과 | 비고 |
|---|---|---|
| ① 클래스 resolve | ✅ | `furiosa/models/.../qwen3_next.py` 작성 + `__init__` 등록 |
| ② 빌드 게이트 | ❌→✅ | `find_compiler_config(qwen3_next)=None` → 즉사. `_EXPERIMENTAL_MODEL_TYPES` 로 우회(per-kernel 컴파일은 default config 폴백) |
| ③ **FX 트레이싱** | ✅ | **DeltaNet 순환규칙·depthwise conv1d·gated norm·512→8 MoE·gated attention 전부 그래프화 성공.** = 연산 자체는 트레이싱 가능 |
| ④ **TP 분할** | ❌→✅ (자력 통과, 2차 세션) | 근본 원인 = attention 색이 `*.self_attn.attn` 경로 모듈에만 시딩되는 하드코딩(block_slicer.py:936-937) → DeltaNet 레이어의 attn 색 미시딩 → 파티션 ID 희소(`[0,2,4,6,7,8]` 실측) → PartitionComposer 인덱싱 IndexError. **해법(모델 코드만 수정):** DeltaNet 순환 본체를 `self_attn.attn` 경로의 서브모듈로 재배치 + 가중치 이름 리매핑 + `make_example_inputs` 오버라이드로 미사용 KV 미선언 → **ID 연속 [0..8], 분할기 통과** ✅ |
| ⑤ 파티션 경계 노드 복제 | ❌ (현재 벽) | `transform.py:116 replicate_nodes_with_multiple_colors` KeyError — 코드에 명시된 가정 "다색 노드의 부모는 전부 같은 색"(transform.py:113)을 DeltaNet 그래프가 위반 |

**결론(2차 세션 갱신):** "전 레이어 KV 소비 강제"는 구조적 전제가 **아니라 자력 통과
가능한 벽**이었음 — 사용자의 문제 제기("빌드 측은 우리가 고칠 수 있지 않나")가 옳았고
실증됨. 빌드 측 계단(④⑤⑥...)은 전부 열린 Python 이라 계속 오를 수 있으나, 각 단계가
"dense 트랜스포머 + 페이지드 어텐션" 가정을 내장하고 있어 남은 계단 수는 미지수.
그리고 끝까지 올라 빌드를 완성해도 **serve 런타임(폐쇄 Rust)이 DeltaNet 순환 상태를
디코드 스텝 간 유지하지 못하는 문제는 그대로** — 이것만이 진짜 벤더 몫이다
(2-1절: KV 바인딩 자체는 텐서 목록 기반이라 하이브리드도 수용 가능, 막는 것은 상태
지속성뿐).

추가 교훈: `~/.cache/furiosa/llm/graphmodules` 캐시는 SDK Python 코드 수정을 키에
반영하지 않음 — 아키텍처 수정 후 `rm .../graphmodules/*Qwen3Next*` 필수.

## 4. 충실한 qwen3_next 에 필요한 작업 (차기 SDK / 벤더 몫)

1. `parallelize/` 분할기·`specs/inputs.py`(`CausalModelForwardInputs.kv_caches` 2-튜플
   강제)·`create_kv_caches`(균일 shape) 를 확장해 **레이어별 이종 상태 텐서**(conv/
   recurrent)를 그래프 입출력으로 표현.
2. **DeltaNet prefill 커널의 RNGD-안전 재작성** — chunked 경로의 fp32 `cumsum`,
   `F.pad`(constant_pad_nd), 부분 `slice_scatter`, 큰 groups depthwise conv1d 회피
   (recurrent 경로는 ③에서 트레이싱됨).
3. **런타임(Rust)에 순환 상태 풀 + qwen3_next enum + 스케줄 preset 추가** — 폐쇄
   바이너리라 벤더(Furiosa)만 가능. 게이트의 미지원 메시지도 "차기 버전에서 지원"을
   명시함.

→ 1·2는 우리 쪽에서 가능하나 3 없이는 serve 불가. **2026.3+ 또는 벤더 지원 필요.**

## 5. 달성한 대안 — Qwen3-Coder-30B-A3B-Instruct-FP8 부활 (위장)

이전 세션에서 "serve 패닉(qwen3_moe×FP8 커널 부재)"으로 사장된 모델을, 빌드된
아티팩트의 `model_type` 을 `qwen3`(dense, 게이트 허용)로 위장해 부활.
도구: `qwen3-next-proj/masquerade_artifact.py`.

**4장 dp 서빙 실측 (2026-06-10, max_ctx 65536):**

| 동시성 | 합산 처리량 | 스트림당 | 비고 |
|---:|---:|---:|---|
| 1 | 63.2 tok/s | 63.2 | 정상 코드 출력(prime/fib/quicksort 정확) |
| 8 | 429.7 tok/s | 53.7 | |
| 32 | 1036.2 tok/s | 32.4 | |

dense qwen3 스케줄러 preset 으로 MoE 그래프를 구동 — 짧은 생성에서 정상 확인.
장문맥·고동시성 안정성은 추가 검증 권장. **"빌드 성공 ≠ serve 성공"의 반대편:
"serve 게이트만 통과시키면 컴파일된 그래프는 실행된다"를 실증.**

### 권장 운영안
- 코더 모델: **Qwen3-Coder-30B-A3B-Instruct-FP8 (위장) + 4장 dp** — 처리량 최대.
- 단일 응답 지연이 중요하면 1장 단독(63 tok/s)도 충분히 빠름.

## 5-1. 자주 묻는 질문 (2026-06-10 사용자 Q&A 누적)

**Q. FX 트레이싱은 어떻게 구현했나?**
구현체는 `furiosa/models/language/architecture/qwen3_next.py` 하나입니다. SDK 가
버킷마다 `torch._dynamo.export`(export/serve/base.py:56)로 모델 forward 를 따라가며
정적 그래프를 뜨는데, 모델이 순수 torch 연산으로만 쓰여 있으면 자동으로 그래프화
됩니다. DeltaNet 순환은 `for i in range(seq_len)` 루프 → 버킷별 고정 길이로 **정적
언롤**(프리필 128토큰 버킷이면 128회 펼쳐짐), conv1d 는 `aten.conv1d(groups=conv_dim)`
로 기록. SDK 는 hidden_states 를 2D `(tokens, hidden)` 로 넘기므로 DeltaNet 진입 시
3D 로 reshape 하는 어댑터가 필요했음(실측 그래프 브레이크 후 수정).

**Q. TP 분할은 vendor 없이 해결 못 하나?**
빌드 측은 **해결 가능** — 실제로 통과시켰습니다(3절 ④). 분할기·블록슬라이서·트랜스폼
전부 열린 Python(`furiosa_llm/parallelize/`)이고, 이번에 모델 코드 수정만으로(분할기
본체 무수정) 통과. 단 다음 단계들(⑤ 노드 복제 등)도 dense-트랜스포머 가정을 내장하고
있어 계단이 더 남아 있고, **끝까지 올라도 serve 의 순환 상태 유지만은 폐쇄 Rust
런타임이라 벤더 몫**입니다.

**Q. 30B-A3B 코더는 왜 serve 가 안 됐고 어떻게 위장으로 통과했나? 직접 띄우려면?**
- 안 된 이유: serve 부팅 시 `NativeLLMEngine` 생성(api.py:383)에서 Rust 게이트
  (`hf_compat_next_gen.rs:367`)가 `model_metadata` 조합을 화이트리스트 검사 —
  qwen3_moe 는 FP8·BF16 모두 거부(2026.2.0 에서 MoE serve 미개방).
- 통과 원리: 연산은 이미 EDF 에 컴파일돼 있고 게이트는 메타데이터 문자열만 봄 →
  `model_type` 을 `qwen3`(dense, 허용)로 바꾸면 부팅·실행·정상 출력(5절 실측).
- 직접 serve (이미 위장된 사본 사용):
  ```bash
  ~/furiosa/bin/furiosa-llm serve \
    /home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/artifacts/qwen3-coder-30b-masq \
    --devices npu:0 --host 0.0.0.0 --port 8000          # 1장 (단일 63 tok/s)
  # 4장 처리량(합산 ~1036 tok/s):
  #   --devices "npu:0,npu:1,npu:2,npu:3" -dp 4
  curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"<위 경로 그대로>","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
  ```
  새 아티팩트 위장은 `qwen3-next-proj/masquerade_artifact.py --copy` 사용(원본 보존).

**Q. qwen3_next 런타임 지원을 퓨리오사에 요청하려면 어떤 .so 가 대상인가?**
아래 표(6-1절) 참조.

## 5-2. 벤더 요청 대상 네이티브 바이너리 (6-1)

| pip 패키지 (2026.2.0) | .so 파일 (site-packages/furiosa/) | 역할 | qwen3_next 에 필요한 작업 |
|---|---|---|---|
| `furiosa-native-runtime` | `native_runtime.cpython-312-x86_64-linux-gnu.so` | serve 엔진 전체: model_type 게이트(`furiosa-generator/src/next_gen/hf_compat_next_gen.rs:367`), 스케줄러·preset, KV 캐시 풀(`host_kv_cache_pool.rs`), prefix cache | ① 게이트 enum 에 qwen3_next 추가 ② **선형어텐션 순환상태(conv+recurrent)의 요청별 풀 관리** ← 최대 작업 ③ 스케줄 preset ④ prefix cache 의 상태 인지 |
| `furiosa-native-llm-common` | `native_llm_common.cpython-312-x86_64-linux-gnu.so` | 빌드측 게이트 `find_compiler_config(model_type, task)` + 컴파일러 설정 테이블(`hf_config.rs` 파서 포함 — layer_types 의 `linear_attention` 값 미지원) | ① 컴파일러 설정 테이블에 qwen3_next 추가 ② hf_config 파서의 layer_types 에 linear_attention 허용 |
| `furiosa-torch` | `native_torch.cpython-312-x86_64-linux-gnu.so` | NPU 컴파일러(op lowering) | (참고) DeltaNet chunked prefill 용 fp32 cumsum·constant_pad_nd·부분 slice_scatter 지원 시 prefill 성능 개선 — 순환식 prefill 은 현 op 셋으로도 가능 |

요청 문구 예: "2026.2.0 의 furiosa-native-runtime / furiosa-native-llm-common 에
model_type `qwen3_next`(Qwen3-Next/Qwen3-Coder-Next, hybrid Gated DeltaNet + gated
attention + MoE) 지원 추가 계획이 있는지, 특히 **디코드 스텝 간 순환 상태(레이어당
conv state + recurrent state) 풀 관리**가 로드맵에 있는지 문의. 빌드 측 Python
아키텍처는 자체 구현 보유."

## 6. 우리가 SDK 에 가한 변경 (최소·문서화)

| 파일 | 변경 | 되돌리기 |
|---|---|---|
| `furiosa_llm/artifact/presets.py` | MINI_SMOKE_PRESET + ref 3개 (미니 빌드용) | 해당 블록 삭제 |
| `furiosa/models/language/architecture/qwen3_next.py` | 신규 아키텍처(빌드 ④에서 막힘, probe) | 파일 삭제 |
| `furiosa/models/language/__init__.py` | qwen3_next import/`__all__` 2줄 | 해당 줄 삭제 |
| `furiosa_llm/metadata/hf_utils.py` | `_EXPERIMENTAL_MODEL_TYPES` 빌드 게이트 우회 | 블록 삭제 |

원본 모델·아티팩트 가중치는 무변경. 위장 사본은 하드링크라 디스크 추가소모 없음.
