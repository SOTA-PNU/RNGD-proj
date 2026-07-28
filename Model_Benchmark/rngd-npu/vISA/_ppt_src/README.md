# 발표자료 생성 소스

`../vISA-입문-발표자료.pptx` (331장) 를 만든 스크립트다.

| 파일 | 역할 |
|---|---|
| `ppt_content.json` | 슬라이드 내용 (11개 부) |
| `deck.py` | 렌더러 — 16:9, 자동 글자크기, 한글 East-Asian 폰트, 넘침 검출 |
| `dia.py` / `dia2.py` | 손으로 그린 도해 27종 |
| `build_ppt.py` | 조립 — 도해 삽입 위치 결정, 넘치는 슬라이드·코드 자동 분할 |
| `ooxml_check.py` | **OOXML 스키마 검증** — 요소 순서·중복 id |
| `layout_check.py` | 레이아웃 검증 — 경계 이탈·글자 넘침·도형 겹침 |

## 다시 만들기

```bash
python3 -m venv /tmp/pptenv && /tmp/pptenv/bin/pip install python-pptx
cd _ppt_src
/tmp/pptenv/bin/python build_ppt.py ppt_content.json ../vISA-입문-발표자료.pptx
/tmp/pptenv/bin/python ooxml_check.py  ../vISA-입문-발표자료.pptx   # 전부 ✅ 여야 한다
/tmp/pptenv/bin/python layout_check.py ../vISA-입문-발표자료.pptx   # 전부 ✅ 여야 한다
```

## ⚠️ PowerPoint 가 "복구" 를 요구할 때 — 이미 겪은 함정 3가지

python-pptx 로 XML 을 직접 손대면 LibreOffice 는 열리는데 PowerPoint 는 손상으로 판정한다.
아래 셋은 실제로 발생했고 `ooxml_check.py` 가 잡는다.

1. **`a:rPr` 자식 순서** — `latin -> ea -> cs` 여야 한다. 순서를 어기면 PowerPoint 가 복구를 요구한다.
   (한글 폰트를 넣으려고 `a:ea` 를 삽입할 때 실수하기 쉽다.)
2. **좌표는 정수** — EMU 는 `xsd:long` 이다. 계산 중 나온 실수를 그대로 쓰면
   `x="2670596.6"` 같은 값이 박힌다. `deck.E()` 를 통과시킨다.
3. **줄바꿈은 `<a:br/>`** — run 의 text 에 `\n` 을 넣으면 `<a:t>` 안에 날 개행이 들어가고,
   PowerPoint 는 그것을 줄바꿈으로 안 친다(한 줄로 이어 그린다). `deck._runs()` 가 처리한다.

## 도해 27종

전체 지도 · 축약 3단계 · 내적→GEMV→GEMM · einsum 읽는 법 · 메모리 비용 계단 · 연산강도 비교 ·
Roofline · MAC 구조 · 시스톨릭 어레이 · 정밀도 비트 배치 · GEMM 타일링 격자 · 패딩·부분충전 ·
이중 버퍼링 · 하드웨어 계층 · 8단계 파이프라인 · 메모리 계층 · flit/packet · 추상화 사다리 ·
컴파일 단계 · 매핑 Pair · 매핑 Padding/Resize · Stride/Modulo=타일링 · 커널 데이터 흐름 ·
컨텍스트 타임라인 · 해저드 3종 · DMA 지배 막대 · 실측 결과 매트릭스

## TCP 계층도 (HTML → PDF)

`TCP-계층도.html` → `../TCP-계층도.pdf` (A4 가로 10쪽)

논문의 하드웨어 5계층(Chip / PE / TU / slice / operation unit)과 소프트웨어 2용어
(lowered shape / tactic)를 개념 하나에 그림 한 장씩 SVG 로 그린 것이다.

```bash
/tmp/pptenv/bin/pip install weasyprint
/tmp/pptenv/bin/python -c "from weasyprint import HTML; HTML('TCP-계층도.html').write_pdf('../TCP-계층도.pdf')"
```

- 브라우저·wkhtmltopdf 없이 WeasyPrint 만으로 변환된다. 한글은 시스템의 Noto Sans CJK KR 을 임베드한다.
- 각 `.page` 는 높이 186mm 고정이고 `.art` 가 104mm 를 차지한다. 설명이 길어지면 다음 쪽으로 흘러넘치므로
  변환 후 **쪽수가 10인지** 반드시 확인할 것(설계상 10쪽).

## PDF → PPTX 변환

`pdf2pptx.py` — 쪽마다 고해상도 PNG 로 렌더해 슬라이드 하나에 전면 배치한다.

```bash
/tmp/pptenv/bin/python pdf2pptx.py ../TCP-계층도.pdf ../TCP-계층도.pptx 200
```

- 슬라이드 크기를 **원본 쪽 크기와 같게** 잡는다(A4 가로 → 11.69 × 8.27 in). 그래서 여백도 잘림도 없다.
- 슬라이드에는 **그림 도형 하나만** 둔다. 숨은 텍스트 상자를 쓰지 않아 손상 위험이 가장 낮다.
- 쪽 텍스트는 **발표자 노트**에 넣어 검색·복사가 된다. 제목은 `p:cSld/@name` 으로 붙인다.
- 16:9 로 바꾸려면 슬라이드 크기만 바꾸면 되지만 좌우에 흰 여백이 생긴다(원본이 A4 비율이라 그렇다).

## 직접 그린 도해 50개

글 위주 슬라이드 50장을 SVG 도해로 바꿔 **그림 객체**로 넣었다.

| 파일 | 내용 |
|---|---|
| `tcp_dia_raw.json` | 도해 원본 SVG 마크업 + 요약 설명 (부·슬라이드 인덱스별) |
| `tcp_dia/*.png` | 렌더·트림된 그림 (220 DPI) |
| `index.json` | 각 도해의 png 경로·픽셀 크기·인치 크기 |
| `make_diagrams.py` | SVG 검증 + 렌더 |

```bash
PY=~/.cache/diagram-deck/venv/bin/python
$PY make_diagrams.py tcp_dia_raw.json tcp_dia          # SVG → 트림된 PNG
$PY build_tcp.py tcp_content.json ../TCP논문-ISCA2024-분석.pptx tcp_dia/index.json
bash ~/.claude/skills/diagram-deck/scripts/verify.sh ../TCP논문-ISCA2024-분석.pptx
```

도해는 `diagram-deck` 스킬(<https://github.com/hyunjun1234/claude-skills>)의 규약과 도구로 만들었다.
그림은 **테두리 없는 독립 객체**로 들어가고, 렌더 시 흰 여백을 잘라내므로 선택 박스가 그림에 딱 맞는다.

## TCP 논문 분석 덱 — 도해는 네이티브 도형이다 (2026-07-27 개정)

50개 도해를 PNG 로 굽지 않고 **PowerPoint 도형**으로 넣는다. 상자·화살표·글자가 각각
편집 가능한 도형이고, 도해 하나가 그룹 하나로 묶인다(그룹 안으로 들어가면 개별 편집).

```bash
PY=~/.cache/diagram-deck/venv/bin/python
$PY build_tcp_native.py tcp_content.json out.pptx tcp_dia_raw.json
bash ~/.claude/skills/diagram-deck/scripts/verify.sh out.pptx
```

- `tcp_dia_raw.json` — 도해 50개의 SVG 원본과 설명. **이것이 유일한 원본이다** (PNG 없음).
- `svg2shapes.py` — SVG → PowerPoint 네이티브 도형 변환기.
- `check_shapes.py` — pptx 안 도형을 SVG 로 되돌려 원본과 픽셀 비교(검증 루프).
  슬라이드 통째 미리보기(`slide_to_svg`)도 여기 있다 — LibreOffice 없이 배치를 눈으로 볼 때 쓴다.
- 배치: 도해는 슬라이드 폭을 다 쓰고, 설명 불릿은 바로 뒤 "— 설명" 슬라이드로 뺀다.
  옆에 나란히 놓으면 그림이 반폭으로 줄어 안쪽 라벨이 5pt 가 된다.
