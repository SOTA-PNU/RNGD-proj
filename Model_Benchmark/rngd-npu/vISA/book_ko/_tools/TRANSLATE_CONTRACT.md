# 번역 계약서 — furiosa-opt 책 한국어판

대상 트리: `/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vISA/book_ko/src/`
현재 그 트리의 md 파일들은 **영문 원문**이다. **제자리에서(in-place) 한국어로 고쳐 쓴다.** 새 파일을 만들지 않는다.

기계 검증기가 번역 전 구조 지문과 대조한다. 아래 "절대 보존"을 하나라도 어기면 실패로 잡힌다.

---

## 1. 절대 보존 — 바이트 단위로 손대지 말 것

1. **코드펜스 안의 모든 것.** ` ``` ` ~ ` ``` ` 사이는 한 글자도 바꾸지 않는다.
   - `rust` / `text` / `bash` / `mermaid` / `jsonc` 전부 해당.
   - 코드 안의 **영어 주석도 그대로 둔다.** (검증기가 SHA1 로 대조한다)
   - `# ` 로 시작하는 rust 숨김 줄, `{{#...}}`, 들여쓰기, 빈 줄 모두 유지.
   - 펜스 줄 자체(` ```rust,ignore `)의 정보 문자열도 그대로.
2. **인라인 코드** `` `...` `` 안의 내용. 백틱 안은 절대 번역하지 않는다.
   - `` `TrfTensor` ``, `` `m![A / 2]` ``, `` `--backend npu` `` 등.
   - 검증기가 인라인 코드 스팬 목록을 통째로 대조한다. **개수·내용 둘 다 같아야 한다.**
3. **링크 타깃** `](...)` 의 괄호 안. 상대경로·앵커(`#foo`)·URL 모두 원문 그대로.
   - 링크 **텍스트**(대괄호 안)는 번역 대상이다. 타깃만 고정.
   - 앵커가 깨져 보여도 고치지 마라. 원문 그대로 둔다(따로 기록해 둔 사안이다).
4. **헤딩 개수·순서·레벨.** `#` 개수와 등장 순서가 1:1 로 같아야 한다.
   헤딩을 합치거나 나누거나 추가하지 않는다. 헤딩 **텍스트만** 바꾼다.
5. **표 구조.** `|` 로 시작하는 줄 수가 같아야 한다. 행·열을 늘리거나 줄이지 않는다.
6. **경고 블록 마커** `> [!NOTE]` / `> [!WARNING]` / `> [!TIP]` 는 영문 그대로. 본문만 번역.
7. **이미지 경로** `![alt](images/x.png)` 의 경로. alt 텍스트는 번역해도 된다.
8. **HTML 태그**, **수식**(`\\(...\\)`, `$...$`) 원문 그대로.

## 2. 번역 대상

문단, 목록 항목, 헤딩 텍스트, 표 셀의 산문, 경고 블록 본문, 이미지 alt.

- **문체: 평서형 '다'체.** "~한다", "~이다". ("~합니다" 쓰지 말 것.)
  기술 매뉴얼 톤. 원문이 짧으면 짧게. 설명을 덧붙이거나 의견을 넣지 않는다.
- **원문에 없는 내용을 만들지 않는다.** 요약·생략도 하지 않는다. 문장 대 문장으로 옮긴다.
- 영어 문장이 통째로 남으면 안 된다. 단, 고유명사·식별자는 영문 유지(아래 §3).

## 3. 용어 — `vISA/GLOSSARY.md` 기준

### 영문 그대로 둘 것 (번역 금지)

- **엔진·유닛 이름**: Fetch Engine, Switch Engine, Collect Engine, Contraction Engine,
  Vector Engine, Cast Engine, Transpose Engine, Commit Engine, Fetch Adapter, Commit Adapter,
  Sequencer, Valid Count Generator(VCG), Tensor Unit, Outer, Packet Reducer, Time Reducer,
  Lane Folder, Intra-Slice Chain, Inter-Slice Reducer, Stream Adapter
- **계층·축 이름**: Chip, Cluster, Slice, Lane, Time, Packet, Element
- **메모리**: HBM, DM, SPM, TRF, VRF, flit, packet
- **타입·식별자·매크로**: `m![]`, `axes![]`, `#[device]`, TrfTensor 등 전부
- **제품·기술명**: TCP, RNGD, vISA, EDF, DPE, MAC, PE, DMA, PCIe, MoE

### 한국어로 옮길 것

| 영문 | 한국어 |
|---|---|
| mapping / mapping expression | 매핑 / 매핑 표현식 |
| spatial dimension / temporal dimension | 공간 차원 / 시간 차원 |
| contraction | 축약 |
| reduce / reduction | 축약(리듀스) — 문맥상 첫 등장만 병기, 이후 "리듀스" |
| broadcast | 브로드캐스트 |
| stride / modulo / padding / resize / identity / escape | 스트라이드 / 모듈로 / 패딩 / 리사이즈 / 항등 / 이스케이프 |
| hazard (RAW/WAR/WAW) | 해저드 (RAW/WAR/WAW 는 영문 유지) |
| double-buffering | 더블 버퍼링 |
| execution context | 실행 컨텍스트 |
| tiling | 타일링 |
| scheduler / scheduling | 스케줄러 / 스케줄링 |
| backend | 백엔드 |
| kernel | 커널 |
| lowering | 로워링 |
| throughput / latency / bandwidth | 처리량 / 지연 / 대역폭 |
| type casting | 타입 캐스팅 |
| trimming | 절단(트리밍) |
| masking | 마스킹 |
| zero-point subtraction | 제로포인트 감산 |
| valid count packing | valid count 패킹 |
| interface | 인터페이스 |
| constraints | 제약 |
| performance | 성능 |
| architecture | 구조 |
| examples | 예제 |
| variants | 변형 |
| optimizations | 최적화 |
| stages | 단계 |
| parameters | 매개변수 |

### 헤딩 처리 원칙

- 헤딩이 **고유 하드웨어·API 이름**이면 영문 유지: `## Fetch Adapter`, `### Pair Mode`, `### Tag`.
- 헤딩이 **일반 명사**면 번역: `## Interface` → `## 인터페이스`, `## Performance` → `## 성능`.
- 섞인 경우 고유명사만 살린다: `## Transitioning to the Inter-Slice Reducer`
  → `## Inter-Slice Reducer 로 넘어가기`.
- 헤딩에 인라인 코드가 있으면 그 백틱 부분은 그대로: `#### \`R\` as \`Time\`` → `#### \`R\` 을 \`Time\` 으로`.

## 4. 하지 말 것

- 문서 맨 위/아래에 "번역본입니다" 류의 머리말·꼬리말을 붙이지 않는다.
- 역주를 달지 않는다. 원문에 없는 각주·괄호 설명을 추가하지 않는다.
- 파일을 새로 만들거나 이름을 바꾸지 않는다.
- 원문의 사실관계를 "고치지" 않는다. 틀려 보여도 그대로 옮긴다.

## 5. 마치기 전 자체 점검

담당 파일마다 확인한다.

```bash
f=<담당 파일 절대경로>
# 코드펜스 짝이 맞는지 (짝수여야 함)
grep -c '^\s*```' "$f"
# 헤딩 개수 (원문과 같아야 함 — 아래 표의 값)
grep -cE '^#{1,6} ' "$f"
```

영문 문장이 통째로 남았는지 눈으로 훑는다(코드·식별자 제외).
