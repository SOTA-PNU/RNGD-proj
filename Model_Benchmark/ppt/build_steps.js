/* furiosa-llm build 7단계 (1.1~1.7) — 이해하기 쉬운 1장
 * 출처: info/ALL_about_build_serve.md Part 1 (1.1~1.7) + 실측 build_trace_full.log
 * 스타일: 기존 build_serve_cli.js / build_model_framework.js (Paperlogy) 준수. */
const pptx = new (require("pptxgenjs"))();
pptx.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pptx.layout = "W";
pptx.author = "RNGD build steps";

const F = {
  black: "Paperlogy 9 Black", xbold: "Paperlogy 8 ExtraBold",
  bold: "Paperlogy 7 Bold", semi: "Paperlogy 6 SemiBold",
  med: "Paperlogy 5 Medium", reg: "Paperlogy 4 Regular",
};
const C = {
  ink: "222222", ink2: "45515e", mut: "8e8e93",
  blue: "1456f0", blue2: "3b82f6", blueBg: "eef4ff",
  white: "ffffff", border2: "e5e7eb", bg2: "f4f5f7",
  ok: "16a34a", warn: "d97706", warnBg: "fff7e6", err: "dc2626",
  vl: "7c3aed", glossBg: "f5f3ff", codeInk: "0b3d91",
};
const M = 0.5, CW = 13.333 - 2 * M;

(() => {
  const s = pptx.addSlide();
  s.background = { color: C.white };

  // 머리 라벨
  s.addText("FURIOSA-LLM · build 내부 동작 (RNGD SDK 2026.2.0)", {
    x: M, y: 0.30, w: 11, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 12, color: C.mut, charSpacing: 0.6,
  });
  // 제목
  s.addText("furiosa-llm build 한 줄이 하는 일 — 7단계", {
    x: M, y: 0.58, w: CW, h: 0.56, margin: 0,
    fontFace: F.bold, fontSize: 27, color: C.ink, charSpacing: -0.6,
  });
  // 부제
  s.addText([
    { text: "명령 한 줄을 넣으면 ", options: { fontFace: F.med, fontSize: 12.5, color: C.ink2 } },
    { text: "1.1 → 1.7", options: { fontFace: F.semi, fontSize: 12.5, color: C.blue } },
    { text: " 순서로 진행돼, 마지막엔 ", options: { fontFace: F.med, fontSize: 12.5, color: C.ink2 } },
    { text: "./out/ 에 NPU가 그대로 실행할 모델 꾸러미(아티팩트)", options: { fontFace: F.semi, fontSize: 12.5, color: C.ink } },
    { text: " 가 생깁니다.", options: { fontFace: F.med, fontSize: 12.5, color: C.ink2 } },
  ], { x: M, y: 1.16, w: CW, h: 0.34, margin: 0, lineSpacingMultiple: 1.2 });

  // 명령 코드 라인
  s.addShape(pptx.ShapeType.roundRect, { x: M, y: 1.60, w: CW, h: 0.38, rectRadius: 0.06, fill: { color: C.bg2 }, line: { type: "none" } });
  s.addText([
    { text: "$ furiosa-llm build ", options: { fontFace: F.semi, fontSize: 11.5, color: C.blue } },
    { text: "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8  ./out  ", options: { fontFace: F.semi, fontSize: 11.5, color: C.ink } },
    { text: "-tp 8  --max-model-len 65536", options: { fontFace: F.reg, fontSize: 11.5, color: C.ink2 } },
  ], { x: M + 0.18, y: 1.60, w: CW - 0.3, h: 0.38, margin: 0, valign: "middle" });

  // ===== 본 표 (7단계) =====
  const HFS = 10.5, FS = 10.2, RH = 0.52;
  const cellBase = (opts) => Object.assign({ valign: "middle", margin: [3, 6, 3, 6] }, opts);

  const hdr = [
    { text: "단계", options: cellBase({ fill: { color: C.blue }, color: C.white, fontFace: F.semi, fontSize: HFS, align: "center" }) },
    { text: "하는 일 — 쉽게 풀어서", options: cellBase({ fill: { color: C.blue }, color: C.white, fontFace: F.semi, fontSize: HFS, align: "left" }) },
    { text: "한 줄 비유", options: cellBase({ fill: { color: C.blue }, color: C.white, fontFace: F.semi, fontSize: HFS, align: "left" }) },
  ];

  // 단계 셀(번호+이름), 본문(리치), 비유
  const stepCell = (no, name, bg) => ({
    text: [
      { text: no + "\n", options: { fontFace: F.bold, fontSize: 12, color: C.blue, breakLine: true } },
      { text: name, options: { fontFace: F.semi, fontSize: 9.5, color: C.ink } },
    ],
    options: cellBase({ fill: { color: bg }, align: "center" }),
  });
  const bodyCell = (runs) => ({ text: runs, options: cellBase({ fill: { color: C.white }, fontFace: F.reg, fontSize: FS, color: C.ink2, align: "left" }) });
  const anaCell = (t) => ({ text: t, options: cellBase({ fill: { color: C.white }, fontFace: F.med, fontSize: 9.6, color: C.vl, align: "left" }) });
  const b = (t) => ({ text: t, options: { fontFace: F.semi, color: C.ink } });   // 강조 run
  const n = (t) => ({ text: t, options: { fontFace: F.reg, color: C.ink2 } });   // 보통 run

  const RB = "f0f5ff";  // 단계 셀 배경(연파랑)
  const rows = [
    hdr,
    [ stepCell("1.1", "명령 접수", RB),
      bodyCell([ n("내가 친 옵션("), b("-tp 8, --max-model-len"), n(" 등)을 프로그램이 알아듣는 "), b("설정 6묶음"), n("으로 옮겨 적습니다.") ]),
      anaCell("주문을 정식 양식으로 옮겨 적는 웨이터") ],
    [ stepCell("1.2", "검증", RB),
      bodyCell([ n("설정이 규칙에 맞나 "), b("빠르게 검문"), n("합니다(tp는 4·8·32만, 모델 정보 필수 항목 있나). 틀리면 "), b("몇 시간 빌드 전에 즉시 거부"), n(".") ]),
      anaCell("공연장 입구의 검표 — 표 틀리면 바로 돌려보냄") ],
    [ stepCell("1.3", "기본값 채우기", RB),
      bodyCell([ n("안 준 값을 자동 결정: 모델 신원("), b("qwen3_moe·FP8"), n("), 컨텍스트 길이, NPU 배치, 그리고 "), b("버킷"), n("(미리 구울 고정 길이 목록).") ]),
      anaCell("신원조회 + 옷 치수표 정하기") ],
    [ stepCell("1.4", "가중치 적재\n+ 양자화", RB),
      bodyCell([ n("모델의 숫자("), b("가중치"), n(")를 불러와 "), b("FP8로 압축"), n("해 용량을 줄입니다. 결과 파일 이름이 곧 명세서("), b("48L·W8fA16KV16"), n(").") ]),
      anaCell("256색 그림을 16색으로 인쇄 — 중요한 글자만 풀컬러") ],
    [ stepCell("1.5", "트레이싱\n(설계도 뽑기)", RB),
      bodyCell([ n("모델을 한 번 따라 돌려 "), b("계산 순서를 그래프(설계도)"), n("로 기록합니다. 버킷(길이)마다 한 장씩, Ray 일꾼이 수행.") ]),
      anaCell("악보(계산그래프)와 가사집(가중치)을 따로 인쇄") ],
    [ stepCell("1.6", "컴파일\n(NPU 기계어)", RB),
      bodyCell([ n("설계도를 NPU가 직접 실행하는 기계어 "), b("EDF"), n("로 번역. 레이어를 조각내 "), b("12단계"), n("를 거칩니다("), b("native_llm_common.so"), n(").") ]),
      anaCell("EDF = NPU 전용 기계어 + 배선도") ],
    [ stepCell("1.7", "저장 / 패키징", RB),
      bodyCell([ n("번역 결과(EDF)를 "), b("binary_bundle.zip"), n("으로 묶고 "), b("artifact.json·가중치·토크나이저"), n("를 ./out/ 한 폴더에 저장.") ]),
      anaCell("악기 파트보(EDF)를 한 폴더로 압축·납품") ],
  ];

  s.addTable(rows, {
    x: M, y: 2.14, w: CW, colW: [1.55, 7.28, 3.50],
    border: { type: "solid", color: C.border2, pt: 0.5 }, rowH: RH, valign: "middle",
  });

  // ===== 용어 풀이 박스 =====
  const GY = 6.18;
  s.addShape(pptx.ShapeType.roundRect, { x: M, y: GY, w: CW, h: 0.86, rectRadius: 0.05, fill: { color: C.glossBg }, line: { color: "e6e0ff", width: 1 } });
  s.addText("어려운 말 풀이", {
    x: M + 0.16, y: GY + 0.06, w: 2.2, h: 0.24, margin: 0, fontFace: F.semi, fontSize: 10, color: C.vl,
  });
  const g = (term, desc) => ([
    { text: term + " ", options: { fontFace: F.semi, fontSize: 9.3, color: C.ink } },
    { text: desc + "    ", options: { fontFace: F.reg, fontSize: 9.3, color: C.ink2 } },
  ]);
  s.addText([
    ...g("양자화(FP8)", "가중치 숫자를 적은 비트로 줄여 용량·속도↑ (W8fA16KV16=가중치 FP8, 활성·KV는 BF16)"),
    ...g("· 버킷", "NPU가 미리 컴파일해 두는 고정 입력 길이(128·256·1024 …). NPU는 동적 크기에 약해 치수별로 따로 굽습니다."),
    { text: "\n", options: { breakLine: true, fontSize: 4 } },
    ...g("트레이싱", "모델을 한 번 실행해 계산 순서를 그래프로 기록"),
    ...g("· EDF", "NPU가 직접 읽는 실행 포맷(기계어+배선도)"),
    ...g("· supertask/stage", "컴파일 단위. 레이어 1개 = [앞 tokenwise | attention | 뒤 tokenwise] 3조각으로 잘림"),
  ], { x: M + 0.16, y: GY + 0.28, w: CW - 0.32, h: 0.54, margin: 0, lineSpacingMultiple: 1.04, valign: "top" });

  // 출처
  s.addText("출처: info/ALL_about_build_serve.md Part 1 (1.1~1.7) · 실측 build_trace_full.log · 코드 cli/convert.py · builder.py · validator.py · resolver.py · trace.py · converter.py", {
    x: M, y: 7.17, w: CW, h: 0.24, margin: 0, fontFace: F.reg, fontSize: 8.3, color: C.mut,
  });
})();

pptx.writeFile({ fileName: "RNGD_Build_Steps.pptx" }).then((f) => console.log("WROTE", f));
