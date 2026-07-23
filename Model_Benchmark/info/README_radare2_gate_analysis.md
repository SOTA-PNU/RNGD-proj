# radare2로 들여다본 serve 게이트 — 바이너리 레벨 정밀 분석 (공부용)

> furiosa-llm 의 `.so` 는 Rust 로 짠 PyO3 확장이라 소스를 못 봅니다. 이 문서는
> **radare2 6.1.7**(소스 빌드)로 `native_llm_common.so`·`native_runtime.so` 를 직접
> 디스어셈블해, "왜 `qwen3_next` 는 serve 가 안 되고 어떻게 통과시키는가"를 바이트
> 단위로 규명한 기록입니다. **모든 주소·바이트는 실측**(2026-06-10).
>
> - 분석 도구: `Model_Benchmark/qwen3-next-proj/radare2/` (소스 빌드, `binr/radare2/radare2`)
>   - 실행: `LD_LIBRARY_PATH=<libr 경로들> ./binr/radare2/radare2 -2 <so>` (`-2` = stderr 억제)
> - 대상 .so (정확 경로):
>   - `/home/jun/furiosa/lib/python3.12/site-packages/furiosa/native_llm_common.cpython-312-x86_64-linux-gnu.so` (143MB)
>   - `/home/jun/furiosa/lib/python3.12/site-packages/furiosa/native_runtime.cpython-312-x86_64-linux-gnu.so` (163MB)
> - ⚠️ 두 .so 는 분석 전 `.orig` 로 백업했고, radare2 는 **읽기 전용**(`-w` 미사용)으로만
>   돌려 **원본 무변경**(분석 후 `cmp` 로 PRISTINE 확인).

관련 문서: [ALL_about_build_serve.md](ALL_about_build_serve.md)(Part 2 serve),
[README_qwen3_next_feasibility.md](README_qwen3_next_feasibility.md)(2-1 KV 바인딩),
[masquerade-serve-gate 메모리].

---

## 0. 한눈 요약

- **serve 게이트는 2겹**입니다:
  1. **load 게이트** — `native_llm_common.so` 가 아티팩트 로드 시 `model_type` 문자열을
     serde enum `ModelType` 으로 역직렬화. 미등록 값이면 즉시 에러.
     (`api.py:349 NextGenArtifact.load_without_blob` → Rust
     `furiosa-llm-common/src/artifact/types/next_gen.rs:238`)
  2. **engine 게이트** — `native_runtime.so` 가 엔진 생성 시 model metadata 재검증.
     (`api.py:383 NativeLLMEngine(...)` → `furiosa-generator/src/next_gen/hf_compat_next_gen.rs:367`)
- **허용 변형(generate)**: `llama, exaone4, qwen2, qwen3, qwen3_moe, gpt_oss` (6개).
  (`embed`, `score` 는 pooling 용 별도.) `qwen3_next` 는 **없음** → serde 가 거부.
- 게다가 `model_type` 은 enum 변형뿐 아니라 **구조 로더**도 고름
  (`qwen3`→`qwen3_32b`(dense), `qwen3_moe`→`qwen3_30b_a3b`(MoE)). 차원 기반.
- ⇒ `qwen3_next` 라벨을 통과시키는 **안전한 방법은 마스커레이드**(model_type 을
  `qwen3`/`qwen3_moe` 로 교체). 바이너리 패치로 리터럴 `qwen3_next` 를 통과시키는 건
  가능하지만 고위험(아래 5절).

---

## 1. 거부 현장 실측 — "unknown variant"

`mini-qwen3` dense 아티팩트의 `model_type` 을 `qwen3_next` 로 바꿔 serve 시도:

```
RuntimeError: unknown variant `qwen3_next`,
  expected one of `llama`, `exaone4`, `qwen2`, `qwen3`, `qwen3_moe`, `gpt_oss`
  at line 1 column 281
Location: furiosa-llm-common/src/artifact/types/next_gen.rs:238:34
```

- Python 트레이스백: `api.py:349 NextGenArtifact.load_without_blob(artifact_path)`.
- 즉 **가중치 로드 전, 아티팩트 메타(JSON)를 파싱하는 단계**에서 serde 가 거부.
- "expected one of ..." 6개 목록이 곧 enum `ModelType` 의 generate 변형들.

---

## 2. 변형 테이블 찾기 (`izz` / `/`)

`native_runtime.so` 에서 문자열 검색:

```
[r2]> izz~ModelType
0x00d7ebfc  ... ModelTypellamaexaone4qwen2qwen3qwen3_moegpt_ossembedscore ...
```

`native_llm_common.so` 에서 각 변형의 정확 주소(`/ qwen3_moe`, `/ qwen3`):

| 변형 | 주소(native_llm_common) | 길이 |
|---|---|---|
| (블롭 시작 `llama`) | 0x00aca217 | 5 |
| `exaone4` | 0x00aca21c | 7 |
| `qwen2` | 0x00aca223 | 5 |
| `qwen3` | 0x00aca228 | 5 |
| `qwen3_moe` | 0x00aca22d | 9 |
| `gpt_oss` | 0x00aca236 | 7 |
| `embed`/`score` | 0x00aca23d / 0x00aca242 | 5 / 5 |

- 변형들은 **널 구분자 없이 한 덩어리**로 packed (`llamaexaone4qwen2qwen3qwen3_moe...`).
  serde 는 (포인터, 길이) 쌍으로 각 조각을 가리킴.

---

## 3. serde `unknown_variant` 의 VARIANTS 배열 (`pxq`)

에러 메시지의 "expected one of ..." 는 serde `unknown_variant(value, VARIANTS)` 가 출력.
그 `VARIANTS: &[&str]` 배열을 llama 포인터(0x00aca217)로 검색해 찾음:

```
[r2]> /x 17a2ac0000000000      ; llama 포인터(LE)
0x00019b00 hit
[r2]> pxq 96 @ 0x00019b00
0x00019b00  0x0000000000aca217  0x0000000008608808   ; (name_ptr=llama, payload_ptr)
0x00019b10  0x0000000000000008  0x0000000000aca21c   ; 0x8, (name_ptr=exaone4)
0x00019b20  0x0000000008608818  0x0000000000000008
0x00019b30  0x0000000000aca223  0x0000000008608828   ; qwen2
0x00019b40  0x0000000000000008  0x0000000000aca228   ; qwen3
...
```

- 엔트리 = 24바이트 `(name_ptr, payload_ptr, 0x08)`. payload(0x086088xx)는
  `{ptr=0(재배치), len}` 형태의 fat-pointer 풀이고, `ptr` 은 **로드 시 동적 재배치**로
  채워짐(파일에는 0). 길이만 정적으로 보임(7,5,5,9,7...).
- ⇒ 파일에서 포인터를 정적 패치해도 **로더가 재배치로 덮어씀** → 데이터 패치가 어려운 이유.

---

## 4. 진짜 매처 디스어셈블 (`aar` → `pd`) — 첫 바이트 점프 테이블

`aar`(참조 분석)로 `qwen3` 문자열(0x00aca228)을 `lea` 하는 코드를 찾음:

```
[r2]> aar ; axt 0x00aca228
(nofunc) 0x01fcd3cc  lea rdi, [0x00aca228]   ; ← 매처 안 "qwen3" 단말 블록
```

그 함수(0x1fcd350~0x1fcd402)를 디스어셈블하면 **serde deserialize_identifier 의
첫 바이트 점프 테이블 디스패치**가 드러남:

```asm
; --- 핫 패스: 입력 문자열의 첫 바이트로 분기 ---
0x1fcd380  mov    rdx, rsi                 ; rdx = 입력 길이
0x1fcd383  movzx  eax, byte [rdi]          ; eax = 입력[0] (첫 바이트)
0x1fcd386  lea    rcx, [0x00aca0d4]        ; 점프 테이블 베이스(256 × i32)
0x1fcd38d  movsxd rax, dword [rcx+rax*4]   ; off = jumptable[첫바이트]
0x1fcd391  add    rax, rcx                 ; target = base + off
0x1fcd394  jmp    rax                      ; → 해당 변형 핸들러로

; --- 변형별 "단말 블록": (변형문자열, 길이) 싣고 compare 로 테일콜 ---
0x1fcd396  lea rdi,[0x00aca217] ; mov esi,5  ; jmp [0x0888c620]   ; llama
0x1fcd3a8  lea rdi,[0x00aca22d] ; mov esi,9  ; jmp [0x0888c620]   ; qwen3_moe
0x1fcd3ba  lea rdi,[0x00aca223] ; mov esi,5  ; jmp [0x0888c620]   ; qwen2
0x1fcd3cc  lea rdi,[0x00aca228] ; mov esi,5  ; jmp [0x0888c620]   ; qwen3  ★
0x1fcd3de  lea rdi,[0x00aca21c] ; mov esi,7  ; jmp [0x0888c620]   ; exaone4
0x1fcd3f0  lea rdi,[0x00aca236] ; mov esi,7  ; jmp [0x0888c620]   ; gpt_oss
0x1fcd402  int3
```

해석(비유: **우편 분류기**):
1. **첫 글자로 1차 분류** — 점프 테이블(0x00aca0d4)이 입력 첫 바이트('l','q','e','g'…)별로
   해당 핸들러로 보냄. (`q` 로 시작하는 qwen2/qwen3/qwen3_moe 는 추가로 길이·뒷바이트로
   2차 분류됨.)
2. **단말 블록** — 각 변형마다 "기대 문자열 + 길이"를 레지스터에 싣고 공용 비교 함수
   `[0x0888c620]` 으로 **테일콜**(jmp). 비교 함수가 입력 == 기대인지 보고 매치면 그 변형의
   discriminant 를, 아니면 에러를 **호출자에게 바로 반환**(tailcall 이라 되돌아오지 않음).
3. 그래서 `qwen3_next`(길이 10) 는 어느 단말에도 안 맞아 **에러 단말**로 떨어짐 → 1절의
   "unknown variant".

핵심: **매칭은 packed 블롭이 아니라 이 단말 블록들의 (lea 주소, mov 길이)** 로 이뤄짐.
블롭 텍스트만 바꾸면 에러 메시지만 바뀔 뿐 매칭은 안 바뀜.

---

## 5. 리터럴 `qwen3_next` 를 바이너리 패치로 통과시키려면 (이론 + 위험)

**개념**: `qwen3_next` 가 qwen3 단말로 라우팅되어 qwen3 의 discriminant 를 반환하게 만들면,
이후 구조 로더가 `qwen3`(dense)로 처리 → dense EDF 실행. 필요한 패치:

1. **단말 패치** — qwen3 단말(0x1fcd3cc)의 `mov esi, 5` → `mov esi, 10` (1바이트: 0x05→0x0A),
   `lea rdi,[0xaca228 "qwen3"]` → 10바이트 `"qwen3_next"` 문자열을 가리키게(코드 케이브에
   문자열 기록 + disp32 수정).
2. **라우팅 패치** — 첫 바이트 'q' 의 2차 분류(길이/뒷바이트)가 길이-10 입력을 qwen3 단말로
   보내도록 분기 수정. (현재는 길이 5/9 만 q-서브트리에 존재.)
3. **2겹 모두** — 위를 `native_runtime.so` 의 동일 매처에도 복제(engine 게이트).

**왜 안 했나(고위험·저효용)**:
- 두 바이너리(143MB·163MB)가 **심볼 전부 스트립**, 함수 경계를 radare2 분석으로 추정해야 함.
- 디스패치 트리의 2차 분류 노드를 정확히 짚어 좌표 맞춰 패치해야 하고, 어긋나면 다른
  model_type 의 로딩까지 깨짐.
- VARIANTS fat-pointer 는 **동적 재배치**라 정적 패치가 덮어써짐(3절).
- 설령 두 게이트를 다 뚫어도, 우리가 가진 EDF 는 dense/MoE 라 **리터럴 qwen3_next 의
  실제 연산(DeltaNet)이 들어있지 않음** — 라벨만 qwen3_next 일 뿐 내용은 그대로.
- ⇒ 학습 가치 대비 SDK 손상 위험이 커서, **분석으로 경로를 규명**하는 선에서 멈춤.

---

## 5-1. 바이너리 패치 실험 (사본에 직접 시도, 2026-06-10 실측)

사용자 제안대로 **원본을 복사해 사본을 radare2(`-w`)로 패치**하고, import 경로에 교체해
gate-1(load)에서 `qwen3_next` 가 통과하는지 검증했습니다(원본은 `.orig` 복원점 유지).

**가설**: qwen3 단말(0x1fcd3cc)이 비교하는 문자열을 `"qwen3"`(5) → `"qwen3_next"`(10) 로
바꾸면, 매처가 입력 `qwen3_next` 를 이 단말로 보내 qwen3 discriminant 를 반환할 것이다.

**패치 (radare2 `-w`, 사본에):**
```
wx 7177656e335f6e657874 @ 0x1fcd402   # 코드 케이브(int3 14B)에 "qwen3_next" 기록
wx 2f000000             @ 0x1fcd3cf   # qwen3 단말 lea disp32 → 케이브(0x1fcd402)
wx 0a                   @ 0x1fcd3d4   # mov esi,5 → mov esi,10
```
검증: `pd 2 @ 0x1fcd3cc` → `lea rdi,[0x1fcd402]; mov esi,0xa`, `ps @ 0x1fcd402` → `qwen3_next`. ✅

**결과**: 사본을 import 경로에 넣고
`NextGenArtifact.load_without_blob(<model_type=qwen3_next 아티팩트>)` 테스트 →
**여전히 `unknown variant 'qwen3_next'`로 거부** ❌. 즉 **단말 패치는 효과 없음.**

**무엇을 배웠나**: 매처는 입력이 변형 단말에 닿기 **전에 길이/구조로 먼저 분기**한다
(serde 가 흔히 생성하는 `match len { 5 => …, 7 => …, 9 => … }` 길이-우선 디스패치).
`qwen3_next`(길이 10)는 길이 버킷이 없어 **단말에 도달하기도 전에 에러 경로로** 빠진다.
따라서 리터럴 통과에는 **길이 디스패치 + 라우팅 + 단말**을 좌표 맞춰 고쳐야 하고, 이를
`native_runtime.so`(gate-2)에도 복제해야 하며, 스트립된 143MB/163MB 바이너리에서
함수 경계를 추정해야 하므로 **고위험·고비용**이다. (다른 model_type 로딩까지 깨질 수 있음.)

**안전성**: 패치는 전부 **사본**에 했고, 테스트 후 즉시 `.orig` 로 복원 → 두 `.so` 모두
`cmp` 로 **PRISTINE** 확인. qwen3/위장 아티팩트 정상 로드, qwen3_next 정상 거부 확인.

> 교훈: serve 게이트의 serde enum 매처는 **길이-우선 + 구조적**이라 한 군데만 고쳐선 안
> 뚫린다. 안전·확실한 통과는 여전히 **마스커레이드**(아래). 바이너리 패치는 가능하나
> (길이 디스패치까지 정밀 RE 필요) ROI 대비 SDK 손상 위험이 커서 권장하지 않는다.

---

## 6. 그래서 통과는 이렇게 (마스커레이드, 실측 성공)

게이트가 `model_type` **문자열만** 보고 연산은 EDF 에 이미 구워져 있으니, 라벨을 허용
변형으로 바꾸면 통과합니다(바이너리 무패치, 가역).

**실측(2026-06-10):**

| 아티팩트 | model_type | serve |
|---|---|---|
| `mini-qwen3-as-next` | `qwen3_next` | ❌ serde "unknown variant" (1절) |
| `mini-qwen3-next-served` | `qwen3`(위장) | ✅ 게이트 통과 + 토큰 생성 |

```bash
# qwen3_next 라벨 아티팩트를 통과시키는 변환 (KV 차원은 절대 불변)
python - <<'PY'
import json; p='artifact.json'; d=json.load(open(p)); md=d['model']['model_metadata']
md['model_type']='qwen3'                          # ← 게이트 통과(허용 변형 + dense 구조로더)
md['hf_configs']['model_type']='qwen3'
md['hf_configs']['architectures']=['Qwen3ForCausalLM']
json.dump(d,open(p,'w'))
PY
furiosa-llm serve <artifact> --devices npu:0 --port 8000   # → 부팅·생성 정상
```

- MoE 코더(예: Qwen3-Coder-30B-A3B)면 `qwen3_moe` 로 위장 → `qwen3_30b_a3b` 구조로 라우팅
  (더 충실). 도구: `qwen3-next-proj/masquerade_artifact.py`.
- ⚠️ `hf_configs.layer_types` 에 `linear_attention` 값이 있으면 Rust hf_config 파서가 패닉
  → 위장 시 제거/재작성. KV 차원(`num_hidden_layers/num_key_value_heads/head_dim`)은 불변 필수.

---

## 7. 재현용 radare2 명령 모음

```bash
# 환경 (소스 빌드 radare2)
cd Model_Benchmark/qwen3-next-proj/radare2
LIBS=$(find $PWD/libr -name '*.so' | xargs -n1 dirname | sort -u | tr '\n' ':')
R2="LD_LIBRARY_PATH=$LIBS $PWD/binr/radare2/radare2 -2"
SO=~/furiosa/lib/python3.12/site-packages/furiosa/native_llm_common.cpython-312-x86_64-linux-gnu.so

# 변형 테이블·VARIANTS·매처
eval $R2 -q -c 'izz~ModelType' $SO                       # 변형 블롭
eval $R2 -q -c 'e search.in=io.maps; / qwen3_moe' $SO    # 변형 주소
eval $R2 -q -c 'e search.in=io.maps; /x 17a2ac0000000000; pxq 96 @ hit0_0' $SO  # VARIANTS 배열
eval $R2 -q -c 'e anal.in=io.maps; aar; axt 0x00aca228' $SO          # 매처 단말 찾기
eval $R2 -q -c 'e anal.in=io.maps; s 0x1fcd350; pd 60' $SO           # 매처 디스어셈블
```

> 정리: serve 게이트는 **컴파일된 serde enum 매처**(첫 바이트 점프 테이블 + 변형별 단말)
> 이고, `model_type` 문자열로 enum 변형과 구조 로더를 동시에 고른다. `qwen3_next` 는 enum 에
> 없어 거부되며, 안전·실측된 통과법은 **허용 변형으로의 위장**이다. 바이너리 패치 경로는
> 규명했으나(5절) 스트립·재배치·2겹·구조로더 때문에 고위험이라 권장하지 않는다.
