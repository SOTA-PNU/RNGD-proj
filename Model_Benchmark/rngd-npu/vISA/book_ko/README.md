# furiosa-opt 공식 책 한국어판

FuriosaAI, *Programming Tensor Contraction Processors* (<https://developer.furiosa.ai/furiosa-opt/book/>)
전문 번역. 저본은 `furiosa-opt` 저장소 `main` 의 `docs/src` 46개 절이며, 같은 트리의
`furiosa-opt-std` 버전은 **0.4.0** 이다.

- 여기는 **원문 번역**이다. 원문에 없는 말을 넣지 않았다.
- 우리 서버 기준 해설·실측·최적화 지도는 [`../book_guide/`](../book_guide/) 에 따로 있다.
- 용어는 [`../GLOSSARY.md`](../GLOSSARY.md) 를 따랐다(단 아래 §5 참고).

## 1. 원문과 다른 점 — 3가지뿐

**(1) `{{#include}}` 50곳을 실제 소스로 인라인 치환했다.**
원문은 mdbook 이 빌드 시점에 형제 크레이트(`furiosa-opt-std`, `furiosa-mapping-types`, `base-template`)
소스를 끌어와 코드블록을 채운다. 이 사본에는 그 크레이트들이 없으므로 미리 채워 넣었다.
그래서 크레이트 없이도 코드가 그대로 보이고, `mdbook build` 가 include 오류 없이 끝난다.
출처는 저본과 같은 커밋의 0.4.0 소스이고, mdbook 과 동일한 `ANCHOR:` 구간 추출·공통 들여쓰기 제거를 적용했다.

**(2) 앵커 80개에 `<a id="...">` 를 심었다.**
헤딩을 한국어로 옮기면 GitHub/mdbook 이 만드는 슬러그가 바뀌어 문서 간 앵커 링크가 전부 깨진다.
원문 헤딩에서 계산한 슬러그를 명시적 id 로 헤딩 바로 앞에 넣어 링크를 살렸다.
헤딩 순서가 원문과 1:1 이므로 위치 대응은 기계적으로 결정된다.

**(3) 산문만 한국어로 바꿨다.** 그 외는 바이트 단위로 원문이다:
코드블록 내용(영어 주석 포함), 인라인 코드, 링크 타깃, 헤딩 개수·레벨·순서,
표의 행 수, `> [!NOTE]` / `> [!WARNING]` 마커, 이미지 경로, 수식, HTML.

## 2. 원본 책 자체가 깨져 있는 링크 15건 — 고치지 않았다

번역 중 발견했다. 목적지를 추측해서 고치면 없는 사실을 만드는 셈이라 **원문 그대로 두고 여기에만 적는다.**
영문 원본과 이 번역본이 **같은 15건**으로 일치한다(§3 의 `anchors.py audit` 로 양쪽을 대조했다).

| 가리키는 곳 | 참조한 문서 | 건수 |
|---|---|--:|
| `mapping-tensors/spatial-temporal-dimensions.md#tensor-unit-stream` | `kernel-examples/index` | 5 |
| `mapping-tensors/spatial-temporal-dimensions.md#hbm-and-sram` | `kernel-examples/index` | 4 |
| `moving-tensors/sequencer.md#configuration` | `contraction-engine/outer` | 1 |
| `moving-tensors/fetch-engine.md#type-casting` | `vector-engine/index` | 1 |
| `moving-tensors/fetch-engine.md#masking` | `vector-engine/intra-slice-reduce` | 1 |
| `#r-in-time-and-slice-over-padded` (자기 문서) | `vector-engine/vcg` | 1 |
| `#4-cumsum-implementation-on-npu` (자기 문서) | `kernel-examples/mixture-of-experts` | 1 |
| `quick-start.md#vector-register-file-vrf` | `kernel-examples/tiling` | 1 |

대상 문서에 그런 헤딩도, 그런 `<a id>` 도 없다. 책이 개편되면서 절이 옮겨지거나 이름이 바뀐 흔적으로 보인다
(예: `mapping-tensors/` 에는 내용이 한 줄뿐인 껍데기 파일 `memory-stream.md`·`tensor-functions.md` 가 남아 있다).

`#4-cumsum-implementation-on-npu` 만은 원인이 분명하다. 링크는 `...-on-npu` 인데 실제 헤딩은
`### 4. \`Cumsum\` Implementation on **TCP**`(슬러그 `#4-cumsum-implementation-on-tcp`)다.
단어 하나 차이의 오타로 보이나, 그래도 원문을 고치지는 않았다.

> 앵커를 셀 때는 헤딩 슬러그만 보면 안 된다. **원본 책은 이미 명시적 `<a id="...">` 를 5곳에 쓴다**
> (`mathematical-tensor-move`, `transitioning-to-the-inter-slice-reducer`, `argument-modes`,
> `padding-strategy`, `interface`). 이걸 빠뜨리면 멀쩡한 참조 8건을 깨진 것으로 오판하게 된다.
> §1-(2) 에서 쓴 기법도 결국 원본 책이 이미 쓰던 것과 같은 기법이다.

## 3. 검증 — 실행 결과

번역 전 영문(include 해소본)에서 파일별 구조 지문(헤딩 레벨 시퀀스, 코드블록 SHA1, 링크 타깃 목록,
인라인 코드 목록, 표 행 수, alert 마커)을 뜬 뒤 번역 후 대조했다.

| 검사 | 결과 |
|---|---|
| 구조 보존 (46파일) | **오류 0**, 경고 1 (`lane-folder.md` 에서 산문 단어 `Lane` 하나를 인라인 코드로 감쌈 — 소실 아님) |
| 코드블록 | 288개 전부 SHA1 동일 |
| 링크 타깃 | 전부 동일 |
| 앵커 참조 | 해소 84종 / 미해소 15건 — **영문 원본과 완전히 같은 수치** |
| 미번역 영문 산문 | 4줄 (전부 의도된 표기법 보존 — 아래 참고) |
| `mdbook build` | **ERROR 0 / WARN 0 / html 47쪽** (영문 대조군과 동일) |

의도적으로 영문으로 남긴 4줄: `switch-engine.md` 의 라우팅 표 3줄은 바로 위 문단이
`<packet>: from <source>, to <action>` 을 표기법으로 정의하고 `input`/`left`/`right`/`output` 을
토큰으로 못박았기 때문이고, `intra-slice-chain.md` 표 머리 1줄은 위 문단이 `Stage`·`Method`·`Way`·`Operand`
를 한국어로 풀어 설명한 뒤 그대로 열 이름으로 쓰기 때문이다.

재현:

```bash
T=/home/jun/.claude/jobs/46bc5c7e/tmp        # 도구 위치(임시 디렉터리, 회수되면 사라짐)
D=<이 저장소>/Model_Benchmark/rngd-npu/vISA/book_ko/src

python3 $T/struct_check.py compare $D $T/baseline.json   # 구조 보존
python3 $T/residual_en.py $D                             # 미번역 영문 잔존
python3 $T/anchors.py audit $D                           # 앵커 참조 해소
$T/build_check.sh $(dirname $D)                          # mdbook 실제 빌드
```

규모: 46개 문서 **13,040행**(영문 12,960행 + 주입 앵커 80행), 한글 81,765자, 이미지 18개.

## 4. 빌드

```bash
cargo install mdbook mdbook-mermaid   # mermaid 블록 6개 때문에 전처리기도 필요
mdbook serve .                        # http://localhost:3000
```

`mdbook-mermaid` 없이 빌드하려면 `book.toml` 의 `[preprocessor.mermaid]` 절을 지운다
(다이어그램 6개가 코드블록으로 보일 뿐 나머지는 정상).

## 5. ⚠️ 원문 부록의 백엔드 설명은 낡았다

`appendix/cargo-furiosa-opt.md` 는 `--backend` 의 기본값을 `simulation` 이라 하고
`simulation` 을 선택지로 넣는다. **실제 설치본에는 `simulation` 이 없다.**

```console
$ cargo furiosa-opt --help
      --backend <BACKEND>
          Possible values:
          - typecheck: Validate kernel mapping and shapes without computing values
          - emulation: Run kernels on the host CPU over Npu's physical buffer storage
          - npu:       Run compiled kernels on the NPU

          [default: emulation]
```

즉 백엔드는 **`typecheck` / `emulation` / `npu` 3종이고 기본값은 `emulation`** 이다.
번역본은 원문을 고치지 않으므로 해당 절에는 여전히 `simulation` 이 적혀 있다. 읽을 때 감안하라.

`../GLOSSARY.md` 의 "backend" 항목(`simulation`(호스트 실값, 기본)…)도 같은 출처에서 온 오류다.
자세한 배경은 [`../book_guide/_GROUND_TRUTH.md`](../book_guide/_GROUND_TRUTH.md) 에 있다.

## 6. 원문 대조

같은 경로·같은 파일명이므로 원문과 1:1 로 대조된다.
원문은 <https://developer.furiosa.ai/furiosa-opt/book/> 또는 저장소의 `docs/src/` 에 있다.
