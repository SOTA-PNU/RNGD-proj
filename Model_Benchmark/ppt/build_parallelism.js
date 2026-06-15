/* TP / PP / DP 병렬화 구조 차이 (16:9, 1장) — 그림 설명
 * 근거: 본 세션 furiosa-llm 2026.2.0 소스 검증 (api.py·artifact/types/config.py·device.py).
 *  - tp = 한 레이어의 텐서(행렬)를 여러 PE가 나눠 동시 계산 (빌드 때 바이너리에 고정)
 *  - pp = 모델 레이어를 스테이지로 나눠 카드별 배치, 토큰이 순차 통과
 *  - dp = 같은 아티팩트를 통째 복제 + 라우터가 요청 분배 (serve 런타임)
 * 스타일: 기존 build_serve_cli.js / build_model_framework.js (Paperlogy) 준수. */
const pptx = new (require("pptxgenjs"))();
pptx.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pptx.layout = "W";
pptx.author = "RNGD Parallelism";

const F = {
  black: "Paperlogy 9 Black", xbold: "Paperlogy 8 ExtraBold",
  bold: "Paperlogy 7 Bold", semi: "Paperlogy 6 SemiBold",
  med: "Paperlogy 5 Medium", reg: "Paperlogy 4 Regular",
};
const C = {
  ink: "222222", ink2: "45515e", mut: "8e8e93",
  blue: "1456f0", blueBg: "eef4ff", blueLn: "9dbcff",
  vl: "7c3aed", vlBg: "f3e8ff", vlLn: "c9a9f5",
  ok: "16a34a", okBg: "e8ffea", okLn: "9cdcab",
  white: "ffffff", border2: "e5e7eb", bg2: "f0f0f0", gray: "cfd4da",
};
const M = 0.5, CW = 13.333 - 2 * M;

function arrowDown(s, x, y, h, color, dash) {
  s.addShape(pptx.ShapeType.line, { x, y, w: 0, h, line: { color, width: 1.75, endArrowType: "triangle", dashType: dash || "solid" } });
}
function arrowLine(s, x, y, w, h, color) {
  s.addShape(pptx.ShapeType.line, { x, y, w, h, line: { color, width: 1.75, endArrowType: "triangle" } });
}
function box(s, x, y, w, h, fill, line, r) {
  s.addShape(pptx.ShapeType.roundRect, { x, y, w, h, rectRadius: r == null ? 0.07 : r, fill: { color: fill }, line: line ? { color: line, width: 1 } : { type: "none" } });
}
function txt(s, t, x, y, w, h, opt) {
  s.addText(t, Object.assign({ x, y, w, h, margin: 0, align: "center", valign: "middle", fontFace: F.semi, fontSize: 9, color: C.ink }, opt || {}));
}

const s = pptx.addSlide();
s.background = { color: C.white };

s.addText("RNGD 병렬화 — TP · PP · DP 구조 차이", {
  x: M, y: 0.30, w: 11, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 12, color: C.mut, charSpacing: 0.6,
});
s.addText("텐서를 쪼개나(TP) · 레이어를 쪼개나(PP) · 모델을 복제하나(DP)", {
  x: M, y: 0.58, w: CW, h: 0.6, margin: 0, fontFace: F.bold, fontSize: 26, color: C.ink, charSpacing: -0.5,
});
s.addText("같은 4장(32 PE)이라도 '무엇을' 나누느냐가 다릅니다. TP는 한 연산을, PP는 레이어를, DP는 요청을 나눕니다.", {
  x: M, y: 1.18, w: CW, h: 0.4, margin: 0, fontFace: F.med, fontSize: 12.5, color: C.ink2,
});

// ===== 3 컬럼 =====
const colW = (CW - 2 * 0.22) / 3;       // ≈ 3.964
const colY = 1.80, colH = 4.74;
const cols = [
  { x: M, kind: "TP", full: "텐서 병렬 (Tensor Parallel)", c: C.blue, bg: C.blueBg, ln: C.blueLn },
  { x: M + colW + 0.22, kind: "PP", full: "파이프라인 병렬 (Pipeline Parallel)", c: C.vl, bg: C.vlBg, ln: C.vlLn },
  { x: M + 2 * (colW + 0.22), kind: "DP", full: "데이터 병렬 (Data Parallel)", c: C.ok, bg: C.okBg, ln: C.okLn },
];

cols.forEach((col) => {
  // 컬럼 카드 + 헤더 바
  box(s, col.x, colY, colW, colH, C.white, C.border2, 0.09);
  s.addShape(pptx.ShapeType.roundRect, { x: col.x, y: colY, w: colW, h: 0.52, rectRadius: 0.09, fill: { color: col.c }, line: { type: "none" } });
  // 헤더 바 아래 모서리 가리개
  s.addShape(pptx.ShapeType.rect, { x: col.x, y: colY + 0.34, w: colW, h: 0.18, fill: { color: col.c }, line: { type: "none" } });
  s.addText([
    { text: col.kind + "  ", options: { fontFace: F.xbold, fontSize: 14, color: C.white } },
    { text: "· " + col.full, options: { fontFace: F.semi, fontSize: 9.5, color: C.white } },
  ], { x: col.x + 0.1, y: colY, w: colW - 0.2, h: 0.52, margin: 0, align: "center", valign: "middle" });
});

const cx = (i) => cols[i].x + colW / 2;

/* ---------------- TP 다이어그램 ---------------- */
(() => {
  const x0 = cols[0].x, c = cx(0);
  txt(s, "한 레이어의 가중치 행렬 W", x0 + 0.2, 2.46, colW - 0.4, 0.26, { fontFace: F.semi, fontSize: 9.5, color: C.blue });
  // W 띠 (4 분할)
  const wx = x0 + 0.3, ww = colW - 0.6, wy = 2.76, wh = 0.5;
  box(s, wx, wy, ww, wh, C.blueBg, C.blue, 0.05);
  for (let k = 1; k < 4; k++) s.addShape(pptx.ShapeType.line, { x: wx + (ww / 4) * k, y: wy, w: 0, h: wh, line: { color: C.blue, width: 1, dashType: "dash" } });
  for (let k = 0; k < 4; k++) txt(s, "W" + k, wx + (ww / 4) * k, wy, ww / 4, wh, { fontFace: F.bold, fontSize: 9, color: C.blue });
  // 화살표 W조각 → PE
  for (let k = 0; k < 4; k++) arrowDown(s, wx + (ww / 4) * (k + 0.5), wy + wh + 0.02, 0.34, C.blue);
  // NPU 카드 + PE 4
  const ky = 3.66, kh = 0.92;
  box(s, x0 + 0.3, ky, colW - 0.6, kh, C.bg2, C.gray, 0.06);
  txt(s, "NPU 1장 (8 PE)", x0 + 0.3, ky + 0.02, colW - 0.6, 0.24, { fontFace: F.semi, fontSize: 8, color: C.mut });
  const px = x0 + 0.45, pw = colW - 0.9, ph = 0.42, py = ky + 0.32;
  const each = (pw - 3 * 0.1) / 4;
  for (let k = 0; k < 4; k++) { const bx = px + k * (each + 0.1); box(s, bx, py, each, ph, C.white, C.blue, 0.05); txt(s, "PE" + k, bx, py, each, ph, { fontFace: F.semi, fontSize: 8.5, color: C.blue }); }
  txt(s, "행렬을 PE 수만큼 쪼개 동시 계산\n(예: tp8 → 8조각). 한 요청을 여러 PE가 분담 → 가속", x0 + 0.2, 4.74, colW - 0.4, 0.62, { fontFace: F.reg, fontSize: 8.5, color: C.ink2, lineSpacingMultiple: 1.12 });
})();

/* ---------------- PP 다이어그램 ---------------- */
(() => {
  const x0 = cols[1].x, c = cx(1);
  const bw = colW - 1.0, bx = c - bw / 2;
  txt(s, "입력 토큰", c - 0.7, 2.46, 1.4, 0.24, { fontFace: F.semi, fontSize: 8.5, color: C.mut });
  arrowDown(s, c, 2.70, 0.18, C.vl);
  // Stage 0
  box(s, bx, 2.90, bw, 0.62, C.vlBg, C.vl, 0.06);
  txt(s, [{ text: "카드 0 — Stage 0\n", options: { fontFace: F.bold, fontSize: 9.5, color: C.vl } }, { text: "Layer 0–23", options: { fontFace: F.reg, fontSize: 8.5, color: C.ink2 } }], bx, 2.90, bw, 0.62, {});
  arrowDown(s, c, 3.54, 0.30, C.vl);
  // Stage 1
  box(s, bx, 3.86, bw, 0.62, C.vlBg, C.vl, 0.06);
  txt(s, [{ text: "카드 1 — Stage 1\n", options: { fontFace: F.bold, fontSize: 9.5, color: C.vl } }, { text: "Layer 24–47", options: { fontFace: F.reg, fontSize: 8.5, color: C.ink2 } }], bx, 3.86, bw, 0.62, {});
  arrowDown(s, c, 4.50, 0.18, C.vl);
  txt(s, "출력", c - 0.7, 4.68, 1.4, 0.24, { fontFace: F.semi, fontSize: 8.5, color: C.mut });
  txt(s, "레이어를 스테이지로 나눠 카드별 배치.\n토큰이 카드→카드 순차 통과 (1장 초과 모델 적재용)", x0 + 0.2, 5.0, colW - 0.4, 0.6, { fontFace: F.reg, fontSize: 8.5, color: C.ink2, lineSpacingMultiple: 1.12 });
})();

/* ---------------- DP 다이어그램 ---------------- */
(() => {
  const x0 = cols[2].x, c = cx(2);
  // 라우터
  const rw = colW - 0.8, rx = c - rw / 2, ry = 2.70;
  box(s, rx, ry, rw, 0.5, C.okBg, C.ok, 0.06);
  txt(s, "라우터 (요청 분배 · round-robin)", rx, ry, rw, 0.5, { fontFace: F.semi, fontSize: 8.5, color: C.ok });
  // 분기 화살표
  const repY = 3.74, repW = (colW - 0.7) / 2 - 0.12, repH = 0.96;
  const lx = x0 + 0.25, rxx = x0 + colW - 0.25 - repW;
  arrowLine(s, c, ry + 0.5, (lx + repW / 2) - c, repY - (ry + 0.5) - 0.02, C.ok);
  arrowLine(s, c, ry + 0.5, (rxx + repW / 2) - c, repY - (ry + 0.5) - 0.02, C.ok);
  // 복제본 2
  [[lx, "복제본 0", "npu:0"], [rxx, "복제본 1", "npu:1"]].forEach(([rX, t1, t2]) => {
    box(s, rX, repY, repW, repH, C.okBg, C.ok, 0.06);
    txt(s, [{ text: t1 + "\n", options: { fontFace: F.bold, fontSize: 9.5, color: C.ok } }, { text: "전체 모델 (" + t2 + ")", options: { fontFace: F.reg, fontSize: 8, color: C.ink2 } }], rX, repY, repW, repH, {});
  });
  txt(s, "같은 아티팩트를 통째로 복제.\n요청을 나눠 처리 → 동시 처리량 ≈ N배", x0 + 0.2, 4.84, colW - 0.4, 0.6, { fontFace: F.reg, fontSize: 8.5, color: C.ink2, lineSpacingMultiple: 1.12 });
})();

// ===== 컬럼별 특성 3줄 =====
const feats = [
  ["빌드 때 고정 (바이너리에 박힘 → serve가 못 바꿈)", "단일 요청 가속 O (한 연산을 PE가 분담)", "RNGD 1장 = 8 PE  ·  tp≤8 = 1장"],
  ["빌드 + serve 둘 다 (serve = 블록 재그룹)", "1장 초과 모델 적재 · KV 풀 확장", "⚠ 단일 요청 가속 아님 (파이프라인 버블)"],
  ["serve 런타임만 (재컴파일 불필요)", "동시 요청 처리량 ↑ (복제 수만큼)", "아티팩트엔 안 박힘 → 가용 PE로 자동 결정"],
];
const fy = 5.62;
cols.forEach((col, i) => {
  const fx = col.x + 0.12;
  feats[i].forEach((line, j) => {
    s.addText([
      { text: "● ", options: { fontFace: F.bold, fontSize: 7.5, color: col.c } },
      { text: line, options: { fontFace: F.reg, fontSize: 8.3, color: C.ink2 } },
    ], { x: fx, y: fy + j * 0.275, w: colW - 0.24, h: 0.27, margin: 0, align: "left", valign: "middle" });
  });
});

// ===== 하단 요약 =====
box(s, M, 6.56, CW, 0.42, C.bg2, null, 0.07);
s.addText([
  { text: "예) RNGD 4장(32 PE): ", options: { fontFace: F.semi, fontSize: 9.5, color: C.ink } },
  { text: "tp8(1장 분할) × dp4(4장 복제) = 32 PE 전부 사용", options: { fontFace: F.reg, fontSize: 9.5, color: C.ink2 } },
  { text: "   ·   dp × pp ≤ 카드 수   ·   ", options: { fontFace: F.reg, fontSize: 9.5, color: C.mut } },
  { text: "tp는 빌드 때, pp·dp는 serve 때 조정", options: { fontFace: F.semi, fontSize: 9.5, color: C.ink } },
], { x: M + 0.2, y: 6.56, w: CW - 0.4, h: 0.42, margin: 0, align: "center", valign: "middle" });

s.addText("개념: TP=한 연산(행렬)을 PE들이 분담 / PP=레이어를 카드별 스테이지로 분할(순차) / DP=모델 복제+라우터로 요청 분산. 근거: furiosa-llm 2026.2.0 소스(api.py·artifact/types/config.py·device.py) 본 세션 검증.", {
  x: M, y: 7.12, w: CW, h: 0.28, margin: 0, fontFace: F.reg, fontSize: 8, color: C.mut,
});

pptx.writeFile({ fileName: "RNGD_Parallelism.pptx" }).then((f) => console.log("WROTE", f));
