/* Furiosa 블로그 재현 — RNGD vs RTX PRO 6000 (Qwen3-32B) 1장 vs 1대 실측 (16:9)
 * build.js (Brandlogy/Paperlogy) 스타일 준수. NPU=파랑, GPU=주황.
 * 데이터 출처: bench-blog/results/report.md (rngd.json + pro6000.json 실측, ISL1024/OSL256).
 * 블로그: https://furiosa.ai/blog/rngd-rtx-pro-6000-real-world-efficiency-benchmark-qwen3 */
const pptx = new (require("pptxgenjs"))();
pptx.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pptx.layout = "W";
pptx.author = "RNGD vs PRO6000";
const TOTAL = 9;

const F = {
  black: "Paperlogy 9 Black", xbold: "Paperlogy 8 ExtraBold",
  bold: "Paperlogy 7 Bold", semi: "Paperlogy 6 SemiBold",
  med: "Paperlogy 5 Medium", reg: "Paperlogy 4 Regular",
};
const C = {
  ink: "222222", ink2: "45515e", mut: "8e8e93",
  blue: "1456f0", blue2: "3b82f6", blue3: "60a5fa", blueLt: "bfdbfe",
  npu: "1456f0", gpu: "f97316", gpuLt: "fed7aa", gpuDk: "c2410c",
  pink: "ea5ec1", white: "ffffff", border: "f2f3f5", border2: "e5e7eb",
  bg2: "f0f0f0", dark: "181e25", okBg: "e8ffea", ok: "16a34a",
  warn: "d97706", warnBg: "fef3c7", err: "dc2626",
  codeTx: "e5e9ef", codeMut: "8ea0b5", codeAc: "5fc6ff",
};
const shStd = () => ({ type: "outer", color: "000000", opacity: 0.08, blur: 6, offset: 2, angle: 90 });
const shGlow = () => ({ type: "outer", color: "2c1e74", opacity: 0.16, blur: 15, offset: 0, angle: 90 });
const M = 0.5, CW = 13.333 - 2 * M;

function frame(s, chapter, page, source) {
  s.background = { color: C.white };
  s.addText(chapter.toUpperCase(), { x: M, y: 0.4, w: 9, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 12, color: C.mut, charSpacing: 0.8 });
  s.addText(`${page} / ${TOTAL}`, { x: M, y: 7.05, w: 3, h: 0.25, margin: 0, fontFace: F.med, fontSize: 10, color: C.mut });
  if (source) s.addText(source, { x: 13.333 - M - 8, y: 7.05, w: 8, h: 0.25, margin: 0, fontFace: F.reg, fontSize: 9.5, color: C.mut, align: "right" });
}
function title(s, head, sub) {
  s.addText(head, { x: M, y: 1.0, w: CW, h: 0.7, margin: 0, fontFace: F.bold, fontSize: 31, color: C.ink, charSpacing: -0.6, lineSpacingMultiple: 1.18 });
  if (sub) s.addText(sub, { x: M, y: 1.73, w: CW, h: 0.4, margin: 0, fontFace: F.med, fontSize: 15, color: C.ink2, lineSpacingMultiple: 1.4 });
}
function card(s, x, y, w, h, opt = {}) {
  s.addShape(pptx.ShapeType.roundRect, { x, y, w, h, rectRadius: opt.r || 0.13, fill: { color: opt.fill || C.white }, line: opt.line === null ? { type: "none" } : { color: opt.line || C.border, width: 1 }, shadow: opt.shadow });
}
function tag(s, x, y, text, fill, txtColor, fs) {
  const w = 0.26 + text.length * 0.092;
  s.addShape(pptx.ShapeType.roundRect, { x, y, w, h: 0.28, rectRadius: 0.14, fill: { color: fill }, line: { type: "none" } });
  s.addText(text, { x, y, w, h: 0.28, margin: 0, align: "center", valign: "middle", fontFace: F.semi, fontSize: fs || 9.5, color: txtColor || C.white });
  return w;
}
function accent(s, x, y, h, color) {
  s.addShape(pptx.ShapeType.roundRect, { x, y, w: 0.07, h, rectRadius: 0.03, fill: { color }, line: { type: "none" } });
}
function legendNG(s, x, y) {
  s.addShape(pptx.ShapeType.rect, { x, y: y + 0.04, w: 0.22, h: 0.13, fill: { color: C.npu }, line: { type: "none" } });
  s.addText("RNGD (1장·tp8)", { x: x + 0.3, y: y - 0.04, w: 1.9, h: 0.28, margin: 0, fontFace: F.semi, fontSize: 9.5, color: C.ink2 });
  s.addShape(pptx.ShapeType.rect, { x: x + 2.2, y: y + 0.04, w: 0.22, h: 0.13, fill: { color: C.gpu }, line: { type: "none" } });
  s.addText("RTX PRO 6000 (1대)", { x: x + 2.5, y: y - 0.04, w: 2.2, h: 0.28, margin: 0, fontFace: F.semi, fontSize: 9.5, color: C.ink2 });
}
function insightCards(s, rx, ry, rw, rh, cards) {
  const ih = (rh - (cards.length - 1) * 0.16) / cards.length;
  cards.forEach(([tg, ti, d, ac], i) => {
    const y = ry + i * (ih + 0.16);
    card(s, rx, y, rw, ih, { shadow: shStd() });
    accent(s, rx, y + 0.18, ih - 0.36, ac);
    s.addText([{ text: tg + "   ", options: { fontFace: F.bold, fontSize: 11.5, color: ac } }, { text: ti, options: { fontFace: F.bold, fontSize: 12.5, color: C.ink } }], { x: rx + 0.26, y: y + 0.15, w: rw - 0.5, h: 0.3, margin: 0 });
    s.addText(d, { x: rx + 0.26, y: y + 0.5, w: rw - 0.52, h: ih - 0.64, margin: 0, fontFace: F.reg, fontSize: 9.6, color: C.ink2, lineSpacingMultiple: 1.34 });
  });
}
const LBL = ["1", "8", "16", "32", "64", "256"];
const lineOpts = (rh, maxv, catTitle) => ({
  chartColors: [C.npu, C.gpu], lineSize: 2.75, lineSmooth: true, lineDataSymbol: "circle", lineDataSymbolSize: 5, showValue: false,
  valAxisMinVal: 0, valAxisMaxVal: maxv,
  catAxisTitle: catTitle, showCatAxisTitle: !!catTitle, catAxisTitleFontSize: 9, catAxisTitleColor: C.mut, catAxisTitleFontFace: F.med,
  catAxisLabelFontFace: F.med, catAxisLabelFontSize: 9, catAxisLabelColor: C.ink2,
  valAxisLabelFontFace: F.reg, valAxisLabelFontSize: 8.5, valAxisLabelColor: C.mut,
  valGridLine: { style: "solid", color: C.border2, size: 0.5 }, valAxisLineColor: C.border2, catAxisLineColor: C.border2, showLegend: false,
});

/* ===================== 1 — Cover ===================== */
(() => {
  const s = pptx.addSlide();
  s.background = { color: C.white };
  s.addText("FURIOSA BLOG REPRODUCTION · Qwen3-32B · 2026.06", { x: M, y: 0.55, w: 11, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 12, color: C.mut, charSpacing: 1 });
  s.addText("RNGD vs RTX PRO 6000\n블로그 효율 벤치마크 직접 재현", { x: M, y: 1.2, w: 12.4, h: 1.7, margin: 0, fontFace: F.bold, fontSize: 41, color: C.ink, charSpacing: -0.8, lineSpacingMultiple: 1.12 });
  s.addText("Furiosa 블로그의 Qwen3-32B 비교를 내 장비 RNGD 1장 vs RTX PRO 6000 1대로 같은 조건 측정", { x: M, y: 2.92, w: 12, h: 0.4, margin: 0, fontFace: F.med, fontSize: 15, color: C.ink2 });
  const hx = M, hy = 3.55, hw = 12.333, hh = 1.86;
  s.addShape(pptx.ShapeType.roundRect, { x: hx, y: hy, w: hw, h: hh, rectRadius: 0.22, fill: { color: C.blue }, line: { type: "none" }, shadow: shGlow() });
  s.addShape(pptx.ShapeType.roundRect, { x: hx + 0.4, y: hy + 0.3, w: 1.16, h: 0.32, rectRadius: 0.16, fill: { color: C.white }, line: { type: "none" } });
  s.addText("핵심 결론", { x: hx + 0.4, y: hy + 0.3, w: 1.16, h: 0.32, margin: 0, align: "center", valign: "middle", fontFace: F.semi, fontSize: 10, color: C.blue });
  s.addText([{ text: "raw 속도는 GPU, 효율은 RNGD — ", options: { fontFace: F.med, color: "dbe6ff" } }, { text: "같은 일을 1/3 전력으로", options: { fontFace: F.bold, color: C.white } }], { x: hx + 0.4, y: hy + 0.62, w: hw - 0.8, h: 0.55, margin: 0, fontSize: 24 });
  s.addText("단일 카드 raw 처리량은 PRO 6000이 최대 2.6배 빠르지만(984 vs 377 tok/s), RNGD는 전력이 1/3(≈180W vs ≈500W)이라 와트당 처리량(tokens/s/W)이 저~중부하에서 최대 2.03배. 블로그의 1.8~2x 사용자 우위는 4장+랙 전력 정규화에서 나오는 값.", { x: hx + 0.4, y: hy + 1.2, w: hw - 0.8, h: 0.56, margin: 0, fontFace: F.med, fontSize: 11, color: "dbe6ff", lineSpacingMultiple: 1.3 });
  const chips = [
    ["측정 모델", "Qwen3-32B FP8", "RNGD weight-FP8 · GPU 공식 W8A8"],
    ["워크로드", "ISL 1024 / OSL 256", "배치(=동시 사용자) 1~256 스윕"],
    ["비교 장비", "1장 vs 1대", "RNGD tp8 1카드 · RTX PRO 6000 1대"],
  ];
  const cw = (12.333 - 2 * 0.2) / 3;
  chips.forEach(([l, n, d], i) => {
    const x = M + i * (cw + 0.2);
    card(s, x, 5.62, cw, 1.16, { shadow: shStd() });
    s.addText(l, { x: x + 0.22, y: 5.76, w: cw - 0.44, h: 0.24, margin: 0, fontFace: F.semi, fontSize: 11, color: C.mut });
    s.addText(n, { x: x + 0.22, y: 5.97, w: cw - 0.44, h: 0.42, margin: 0, fontFace: F.bold, fontSize: 19, color: C.blue });
    s.addText(d, { x: x + 0.22, y: 6.44, w: cw - 0.44, h: 0.3, margin: 0, fontFace: F.reg, fontSize: 9, color: C.ink2 });
  });
  s.addText("RNGD: furiosa-llm 2026.2.0 · npu 1카드   |   GPU: vLLM · RTX PRO 6000 96GB (Blackwell)", { x: 13.333 - M - 9.5, y: 7.05, w: 9.5, h: 0.25, margin: 0, fontFace: F.reg, fontSize: 9.5, color: C.mut, align: "right" });
})();

/* ===================== 2 — 블로그 주장 vs 우리 재현 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Context", 2, "출처: furiosa.ai/blog · RNGD vs RTX Pro 6000 (Qwen3)");
  title(s, "블로그는 무엇을 주장했나 — 그리고 우리가 검증한 것", "블로그의 1.8~2x는 본문에 \"normalized for rack power\"(랙 전력 정규화 후)로 명시됨");
  const gy = 2.45, gh = 1.18;
  card(s, M, gy, 12.333, gh, { fill: C.blue, line: null, r: 0.16, shadow: shGlow() });
  s.addText("블로그 핵심 주장 (4x RNGD 서버 기준)", { x: M + 0.35, y: gy + 0.16, w: 6, h: 0.26, margin: 0, fontFace: F.semi, fontSize: 11, color: "dbe6ff" });
  s.addText([
    { text: "RNGD가 RTX PRO 6000보다 1.8x · 1.9x · 2.0x 더 많은 사용자", options: { fontFace: F.bold, color: C.white } },
    { text: "  (SLO 20·30·40 TPS/user, 랙 전력 정규화) · 전력 8장 3kW vs 6.6kW · 랙당 2.5x", options: { fontFace: F.med, color: "dbe6ff" } },
  ], { x: M + 0.35, y: gy + 0.46, w: 11.6, h: 0.6, margin: 0, fontSize: 14.5, lineSpacingMultiple: 1.3 });
  const cy = gy + gh + 0.22, ch = 6.85 - cy;
  const cols = [
    ["공개된 것", C.ok, ["Qwen3-32B · 배치 b8~b256", "SLO = per-user 20/30/40 TPS", "사용자 수 = 그 SLO에서 받는 동시 사용자", "전력 3kW vs 6.6kW · 칩 180W"]],
    ["블로그가 안 밝힌 것", C.warn, ["입력/출력 토큰 길이", "서빙 엔진(양측 모두)", "한 모델당 카드 수(TP)", "\"사용자 수\" 계산식"]],
    ["우리 재현 방식", C.blue, ["1장 vs 1대 (per-device)", "ISL 1024 / OSL 256 고정", "양쪽 FP8 · 공통 loadgen", "raw + 전력정규화 둘 다 측정"]],
  ];
  const cw = (12.333 - 2 * 0.2) / 3;
  cols.forEach(([h, ac, items], i) => {
    const x = M + i * (cw + 0.2);
    card(s, x, cy, cw, ch, { shadow: shStd() });
    accent(s, x, cy + 0.24, 0.4, ac);
    s.addText(h, { x: x + 0.26, y: cy + 0.22, w: cw - 0.5, h: 0.34, margin: 0, fontFace: F.bold, fontSize: 15, color: C.ink });
    s.addText(items.map((t) => ({ text: t, options: { fontFace: F.reg, fontSize: 11.5, color: C.ink2, bullet: { code: "2022", indent: 14 }, breakLine: true } })), { x: x + 0.3, y: cy + 0.74, w: cw - 0.55, h: ch - 0.95, margin: 0, lineSpacingMultiple: 1.5 });
  });
})();

/* ===================== 3 — 측정 환경·방법 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Method", 3, "출처: bench-blog/loadgen.py · run_rngd.sh · run_pro6000.sh");
  title(s, "측정 환경 — 같은 클라이언트, 다른 장비", "OpenAI 호환 /v1/completions 만 호출 → 서버 계층만 바꿔 RNGD·GPU를 같은 코드로 측정");
  const fy = 2.5, fh = 1.55;
  card(s, M, fy, 3.3, fh, { shadow: shStd() });
  accent(s, M, fy + 0.26, fh - 0.52, C.ink2);
  s.addText("공통 loadgen", { x: M + 0.3, y: fy + 0.22, w: 2.9, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 13, color: C.ink });
  s.addText("closed-loop 배치 스윕\n고정 ISL/OSL · 고유 프롬프트\nTTFT·per-user·집계 TPS + 전력 동시측정", { x: M + 0.3, y: fy + 0.56, w: 2.85, h: fh - 0.7, margin: 0, fontFace: F.reg, fontSize: 9.6, color: C.ink2, lineSpacingMultiple: 1.36 });
  s.addText("→", { x: M + 3.3, y: fy, w: 0.5, h: fh, margin: 0, align: "center", valign: "middle", fontFace: F.bold, fontSize: 22, color: C.mut });
  const bw = 4.3, bx0 = M + 3.86;
  [["RNGD 경로", "furiosa-llm serve", "qwen3-32b-fp8-tp8 · npu 1카드 · furiosa-smi 전력", C.npu], ["GPU 경로", "vllm serve", "Qwen/Qwen3-32B-FP8 W8A8 · 1대 · nvidia-smi 전력", C.gpu]].forEach(([t, eng, d, ac], i) => {
    const x = bx0 + i * (bw + 0.2);
    card(s, x, fy, bw, fh, { shadow: shStd(), line: ac });
    accent(s, x, fy + 0.26, fh - 0.52, ac);
    s.addText(t, { x: x + 0.3, y: fy + 0.2, w: bw - 0.5, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 11, color: ac });
    s.addText(eng, { x: x + 0.3, y: fy + 0.48, w: bw - 0.5, h: 0.34, margin: 0, fontFace: F.bold, fontSize: 15, color: C.ink });
    s.addText(d, { x: x + 0.3, y: fy + 0.86, w: bw - 0.55, h: fh - 0.98, margin: 0, fontFace: F.reg, fontSize: 10, color: C.ink2, lineSpacingMultiple: 1.32 });
  });
  const dy = fy + fh + 0.24, dh = 6.85 - dy;
  const items = [
    ["배치 = 동시 사용자", "b1·8·16·32·64·256 동시 요청을 항상 in-flight 유지(closed-loop). 배치 = 그 순간 동시 사용자 수.", C.blue],
    ["고정 ISL/OSL", "입력 1024 / 출력 256 토큰 고정(ignore_eos+min_tokens로 정확히 256). 양쪽 같은 프롬프트 → 공정.", C.blue2],
    ["per-user TPS · SLO", "각 사용자가 받는 출력속도(tok/s). 블로그처럼 20/30/40 TPS를 SLO 기준선으로 사용.", C.blue3],
    ["전력 동시 샘플링", "측정 구간 동안 1Hz로 카드 전력(W) 수집 → tokens/s/W·users/kW 계산.", C.pink],
  ];
  const cw = (12.333 - 3 * 0.2) / 4;
  items.forEach(([ti, d, ac], i) => {
    const x = M + i * (cw + 0.2);
    card(s, x, dy, cw, dh, { shadow: shStd() });
    accent(s, x, dy + 0.22, 0.4, ac);
    s.addText(ti, { x: x + 0.24, y: dy + 0.2, w: cw - 0.44, h: 0.6, margin: 0, fontFace: F.bold, fontSize: 13.5, color: C.ink, lineSpacingMultiple: 1.1 });
    s.addText(d, { x: x + 0.24, y: dy + 0.86, w: cw - 0.46, h: dh - 1.0, margin: 0, fontFace: F.reg, fontSize: 10.5, color: C.ink2, lineSpacingMultiple: 1.46 });
  });
})();

/* ===================== 4 — 처리량 (aggregate TPS) ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Result · Throughput", 4, "출처: results/report.md · 집계 출력 TPS");
  title(s, "처리량 — raw는 GPU가 빠르다", "단일 PRO 6000은 동시성이 늘수록 계속 확장(984 tok/s), RNGD 1장은 b32에서 ~377로 포화");
  const ry = 2.45, rh = 6.85 - ry, chw = 7.5;
  card(s, M, ry, chw, rh, { shadow: shStd() });
  s.addText("배치별 집계 출력 처리량 (tok/s)", { x: M + 0.25, y: ry + 0.14, w: 3.5, h: 0.28, margin: 0, fontFace: F.semi, fontSize: 12.5, color: C.ink });
  legendNG(s, M + chw - 4.5, ry + 0.16);
  s.addChart(pptx.ChartType.line, [
    { name: "RNGD", labels: LBL, values: [25.5, 171.2, 289.7, 376.9, 368.4, 372.7] },
    { name: "PRO6000", labels: LBL, values: [37.6, 288.7, 487.9, 741.1, 953.3, 984.1] },
  ], { x: M + 0.1, y: ry + 0.5, w: chw - 0.3, h: rh - 0.9, ...lineOpts(rh, 1100, "배치 (동시 사용자)") });
  insightCards(s, M + chw + 0.24, ry, 12.333 - chw - 0.24, rh, [
    ["b1–b16", "저부하 격차 작음", "단일·소수 사용자에서는 차이가 작다(b1 25.5 vs 37.6). RNGD 1장도 충분.", C.npu],
    ["b32", "RNGD 포화 시작", "RNGD는 b32에서 377 tok/s로 천장. 1카드 8 PE의 배치 한계.", C.mut],
    ["b64–b256", "GPU 계속 확장", "PRO 6000은 984까지 확장(b256). 피크 처리량 GPU가 RNGD의 2.6배.", C.gpu],
  ]);
})();

/* ===================== 5 — per-user TPS & SLO ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Result · Per-user SLO", 5, "출처: results/report.md · per-user 출력 TPS(p50)");
  title(s, "per-user 속도 & SLO 사용자 수", "블로그의 20/30/40은 '사용자 1명당 보장 tok/s' 기준선 — RNGD 1장 단일스트림 천장은 25 tok/s");
  const ry = 2.45, rh = 6.85 - ry, chw = 7.5;
  card(s, M, ry, chw, rh, { shadow: shStd() });
  s.addText("배치별 per-user 출력 TPS (p50)", { x: M + 0.25, y: ry + 0.14, w: 3.5, h: 0.28, margin: 0, fontFace: F.semi, fontSize: 12.5, color: C.ink });
  legendNG(s, M + chw - 4.5, ry + 0.16);
  s.addChart(pptx.ChartType.line, [
    { name: "RNGD", labels: LBL, values: [25.23, 19.43, 16.21, 11.92, 9.38, 9.35] },
    { name: "PRO6000", labels: LBL, values: [38.20, 35.69, 31.74, 22.77, 15.84, 5.60] },
  ], { x: M + 0.1, y: ry + 0.5, w: chw - 0.3, h: rh - 0.9, ...lineOpts(rh, 45, "배치 (동시 사용자)") });
  s.addText("SLO 20·30·40 TPS/user 기준선 = 각 사용자 최소 보장속도", { x: M + 0.3, y: ry + rh - 0.34, w: chw - 0.6, h: 0.26, margin: 0, fontFace: F.reg, fontSize: 8.6, color: C.mut, italic: true });
  // 오른쪽: users@SLO 미니표 + 해석
  const rx = M + chw + 0.24, rw = 12.333 - chw - 0.24;
  card(s, rx, ry, rw, 2.55, { shadow: shStd() });
  s.addText("SLO당 최대 동시 사용자 (raw, 장비 1개)", { x: rx + 0.24, y: ry + 0.18, w: rw - 0.4, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 12.5, color: C.ink });
  const rows = [["SLO", "RNGD", "PRO6000"], ["20 TPS", "7.3명", "44.8명"], ["30 TPS", "0", "19.1명"], ["40 TPS", "0", "0"]];
  const colx = [rx + 0.26, rx + 1.9, rx + 3.4]; let ty = ry + 0.62;
  rows.forEach((r, ri) => {
    r.forEach((cell, ci) => s.addText(cell, { x: colx[ci], y: ty, w: 1.6, h: 0.36, margin: 0, fontFace: ri === 0 ? F.semi : F.bold, fontSize: ri === 0 ? 10 : 12.5, color: ri === 0 ? C.mut : (ci === 1 ? C.npu : ci === 2 ? C.gpu : C.ink2) }));
    if (ri === 0) s.addShape(pptx.ShapeType.line, { x: rx + 0.26, y: ty + 0.34, w: rw - 0.5, h: 0, line: { color: C.border2, width: 1 } });
    ty += ri === 0 ? 0.46 : 0.5;
  });
  insightCards(s, rx, ry + 2.71, rw, rh - 2.71, [
    ["천장", "단일스트림 25 vs 38", "RNGD 1장은 한 명에게도 25 tok/s가 최대 → 30·40 SLO는 1장으론 도달 불가. GPU는 단일스트림이 빨라(38) 더 높은 SLO·더 많은 사용자.", C.gpu],
    ["블로그처럼 하려면", "RNGD 멀티카드 필요", "블로그는 4x RNGD. 카드를 묶으면(tp16/32) 단일스트림이 빨라져 30·40 SLO도 도달. 1장 비교의 한계.", C.blue],
  ]);
})();

/* ===================== 6 — 전력 & 에너지효율 (RNGD 핵심) ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Result · Efficiency", 6, "출처: results/report.md · 전력(W) & tokens/s/W");
  title(s, "전력과 에너지효율 — 여기가 RNGD의 무대", "RNGD는 1/3 전력으로 동작 → 와트당 처리량(tokens/s/W)이 저~중부하에서 최대 2.03배");
  legendNG(s, 13.333 - M - 4.75, 2.2);
  // 위: 두 차트 (전력 bar | 효율 line)
  const ry = 2.6, chh = 2.5, cw = (12.333 - 0.24) / 2;
  card(s, M, ry, cw, chh, { shadow: shStd() });
  s.addText("배치별 카드 전력 (W)", { x: M + 0.25, y: ry + 0.13, w: 4, h: 0.26, margin: 0, fontFace: F.semi, fontSize: 11.5, color: C.ink });
  s.addChart(pptx.ChartType.bar, [
    { name: "RNGD", labels: LBL, values: [134, 156, 176, 193, 204, 210] },
    { name: "PRO6000", labels: LBL, values: [402, 451, 480, 505, 562, 600] },
  ], {
    x: M + 0.1, y: ry + 0.48, w: cw - 0.3, h: chh - 0.66, barDir: "col", barGapWidthPct: 45, chartColors: [C.npu, C.gpu],
    valAxisMinVal: 0, valAxisMaxVal: 700, catAxisLabelFontFace: F.med, catAxisLabelFontSize: 8.4, catAxisLabelColor: C.ink2,
    valAxisLabelFontFace: F.reg, valAxisLabelFontSize: 8, valAxisLabelColor: C.mut,
    valGridLine: { style: "solid", color: C.border2, size: 0.5 }, valAxisLineColor: C.border2, catAxisLineColor: C.border2, showLegend: false,
  });
  const cx = M + cw + 0.24;
  card(s, cx, ry, cw, chh, { shadow: shStd() });
  s.addText("에너지효율 (tokens/s/W) — 높을수록 효율적", { x: cx + 0.25, y: ry + 0.13, w: 5, h: 0.26, margin: 0, fontFace: F.semi, fontSize: 11.5, color: C.ink });
  s.addChart(pptx.ChartType.line, [
    { name: "RNGD", labels: LBL, values: [0.190, 1.098, 1.647, 1.950, 1.806, 1.779] },
    { name: "PRO6000", labels: LBL, values: [0.093, 0.640, 1.016, 1.467, 1.695, 1.640] },
  ], { x: cx + 0.1, y: ry + 0.48, w: cw - 0.3, h: chh - 0.66, ...lineOpts(chh, 2.2, "") });
  // 아래: KPI 3개 가로
  const ky = ry + chh + 0.22, kh = 6.85 - ky, kw = (12.333 - 2 * 0.2) / 3;
  const kpis = [["1/3", "전력", "RNGD ≈180W vs GPU ≈500W 평균 — 같은 일을 1/3 전력으로", C.npu], ["2.03x", "효율 @ 저부하(b1)", "tokens/s/W 0.190 vs 0.093 — 단일·소수 사용자에서 격차 최대", C.blue], ["1.15~1.7x", "효율 @ 중부하", "b16~b32에서 RNGD가 와트당 1.6~2.0배. 고부하선 격차 축소", C.blue2]];
  kpis.forEach(([n, l, d, ac], i) => {
    const x = M + i * (kw + 0.2);
    card(s, x, ky, kw, kh, { shadow: shStd() });
    accent(s, x, ky + 0.2, kh - 0.4, ac);
    s.addText(n, { x: x + 0.28, y: ky + 0.16, w: kw - 0.5, h: 0.6, margin: 0, fontFace: F.black, fontSize: 30, color: ac });
    s.addText(l, { x: x + 0.28, y: ky + 0.8, w: kw - 0.5, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 13, color: C.ink });
    s.addText(d, { x: x + 0.28, y: ky + 1.12, w: kw - 0.55, h: kh - 1.25, margin: 0, fontFace: F.reg, fontSize: 10, color: C.ink2, lineSpacingMultiple: 1.36 });
  });
})();

/* ===================== 7 — TTFT ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Result · Latency", 7, "출처: results/report.md · TTFT p50 (b1~b32 구간)");
  title(s, "TTFT — 중부하까진 RNGD가 더 빠르다", "b8~b32에서 RNGD 첫 토큰이 더 빠름. 단 b64↑에서는 RNGD가 큐 포화로 급증");
  const ry = 2.45, rh = 6.85 - ry, chw = 7.5;
  card(s, M, ry, chw, rh, { shadow: shStd() });
  s.addText("TTFT p50 (초)", { x: M + 0.25, y: ry + 0.14, w: 3, h: 0.28, margin: 0, fontFace: F.semi, fontSize: 12.5, color: C.ink });
  legendNG(s, M + chw - 4.5, ry + 0.16);
  s.addChart(pptx.ChartType.bar, [
    { name: "RNGD", labels: ["1", "8", "16", "32"], values: [0.27, 0.76, 0.76, 0.78] },
    { name: "PRO6000", labels: ["1", "8", "16", "32"], values: [0.19, 1.52, 2.02, 2.53] },
  ], {
    x: M + 0.1, y: ry + 0.52, w: chw - 0.3, h: rh - 0.9, barDir: "col", barGapWidthPct: 45, chartColors: [C.npu, C.gpu],
    showValue: true, dataLabelFontSize: 8, dataLabelFontFace: F.semi, dataLabelPosition: "outEnd", dataLabelColor: C.ink, dataLabelFormatCode: "0.00\"s\"",
    valAxisMinVal: 0, valAxisMaxVal: 3, catAxisTitle: "배치 (동시 사용자)", showCatAxisTitle: true, catAxisTitleFontSize: 9, catAxisTitleColor: C.mut, catAxisTitleFontFace: F.med,
    catAxisLabelFontFace: F.med, catAxisLabelFontSize: 9, catAxisLabelColor: C.ink2, valAxisLabelFontFace: F.reg, valAxisLabelFontSize: 8, valAxisLabelColor: C.mut,
    valGridLine: { style: "solid", color: C.border2, size: 0.5 }, valAxisLineColor: C.border2, catAxisLineColor: C.border2, showLegend: false,
  });
  insightCards(s, M + chw + 0.24, ry, 12.333 - chw - 0.24, rh, [
    ["b8–b32", "RNGD 첫 토큰 빠름", "RNGD 0.76~0.78s vs GPU 1.52~2.53s. 대화형 응답 체감 RNGD 우위(최대 3.25x).", C.npu],
    ["b64 이상", "RNGD 큐 포화", "b64 RNGD 8.4s, b256 143s로 급증(1장 배치 한계). GPU는 b256도 15.6s.", C.gpu],
    ["시사점", "적정 부하대가 핵심", "RNGD 1장은 ~b32 이내 운영이 적합. 그 이상은 카드 추가 또는 GPU.", C.blue],
  ]);
})();

/* ===================== 8 — 종합 비교 (raw + 전력정규화) ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Summary", 8, "출처: results/report.md · 표 2·3");
  title(s, "종합 — raw vs 전력 정규화", "raw 1:1에서는 GPU 우위. 블로그의 RNGD 우위는 '전력 정규화' 관점에서 봐야 한다");
  const cy = 2.5, ch = 2.05, cw = (12.333 - 0.24) / 2;
  // 왼쪽: raw users@SLO
  card(s, M, cy, cw, ch, { shadow: shStd() });
  s.addText("(A) raw 최대 사용자 / SLO", { x: M + 0.26, y: cy + 0.16, w: 5, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 13.5, color: C.ink });
  [["SLO", "RNGD", "PRO6000", "RNGD/GPU"], ["20 TPS", "7.3", "44.8", "0.16x"], ["30 TPS", "0", "19.1", "—"], ["40 TPS", "0", "0", "—"]].forEach((r, ri) => {
    const yy = cy + 0.56 + ri * (ri === 0 ? 0.4 : 0.42);
    r.forEach((cell, ci) => s.addText(cell, { x: M + 0.26 + ci * 1.45, y: yy, w: 1.45, h: 0.34, margin: 0, fontFace: ri === 0 ? F.semi : F.bold, fontSize: ri === 0 ? 10 : 12, color: ri === 0 ? C.mut : ci === 1 ? C.npu : ci === 2 ? C.gpu : C.ink2 }));
  });
  // 오른쪽: 전력정규화 (tokens/s/W)
  const rx = M + cw + 0.24;
  card(s, rx, cy, cw, ch, { shadow: shStd() });
  s.addText("(B) 에너지효율 tokens/s/W (RNGD/GPU)", { x: rx + 0.26, y: cy + 0.16, w: 5.5, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 13.5, color: C.ink });
  [["배치", "1", "8", "16", "32", "64", "256"], ["RNGD/GPU", "2.03x", "1.72x", "1.62x", "1.33x", "1.07x", "1.08x"]].forEach((r, ri) => {
    const yy = cy + 0.58 + ri * 0.5;
    r.forEach((cell, ci) => s.addText(cell, { x: rx + 0.26 + ci * 0.86, y: yy, w: 0.9, h: 0.36, margin: 0, align: ci === 0 ? "left" : "center", fontFace: ri === 0 ? F.semi : F.bold, fontSize: ri === 0 ? 9.5 : 11.5, color: ri === 0 ? C.mut : (parseFloat(cell) >= 1.5 ? C.ok : C.ink2) }));
  });
  s.addText("저~중부하에서 RNGD가 와트당 1.6~2.0배 효율. 고부하에선 GPU 처리량이 커져 효율 격차 축소(~1.1x).", { x: rx + 0.26, y: cy + ch - 0.5, w: cw - 0.5, h: 0.4, margin: 0, fontFace: F.reg, fontSize: 9.5, color: C.ink2, lineSpacingMultiple: 1.3 });
  // 아래: 왜 블로그와 다른가
  const by = cy + ch + 0.24, bh = 6.85 - by;
  card(s, M, by, 12.333, bh, { fill: C.blue, line: null, r: 0.16, shadow: shGlow() });
  s.addText("왜 1:1 raw로는 블로그의 1.8~2x가 안 나오나", { x: M + 0.35, y: by + 0.2, w: 8, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 12, color: "dbe6ff" });
  const reasons = [
    ["① 카드 수", "블로그는 4x RNGD. 멀티카드는 단일스트림·처리량이 1장보다 크게 빠르다 → 30·40 SLO 도달, 사용자↑."],
    ["② 랙 전력 정규화", "1.8~2x는 \"normalized for rack power\". 15kW 랙에 RNGD 서버 5대 vs GPU 2대 → 전력당 사용자에서 벌어짐."],
    ["③ per-card 효율은 입증됨", "그럼에도 1장 실측에서 RNGD는 전력 1/3·와트당 최대 2배 → 블로그의 '효율' 논지 자체는 성립."],
  ];
  const rw = (12.333 - 0.7 - 2 * 0.3) / 3;
  reasons.forEach(([t, d], i) => {
    const x = M + 0.35 + i * (rw + 0.3);
    s.addText(t, { x, y: by + 0.6, w: rw, h: 0.32, margin: 0, fontFace: F.bold, fontSize: 14, color: C.white });
    s.addText(d, { x, y: by + 0.98, w: rw, h: bh - 1.2, margin: 0, fontFace: F.med, fontSize: 10.5, color: "dbe6ff", lineSpacingMultiple: 1.4 });
  });
})();

/* ===================== 9 — 결론 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Conclusion", 9, "출처: bench-blog · 1장 vs 1대 실측");
  title(s, "결론 — 무엇을 증명했나", "블로그의 '효율' 논지는 1장 실측으로도 성립. '1.8~2x 사용자'는 멀티카드+랙 전력에서 재현");
  const colW = (12.333 - 0.24) / 2, cy = 2.5, chh = 2.5;
  // RNGD 강점
  card(s, M, cy, colW, chh, { shadow: shStd(), line: C.npu });
  accent(s, M, cy + 0.26, chh - 0.52, C.npu);
  s.addText("RNGD가 이기는 지점", { x: M + 0.3, y: cy + 0.22, w: colW - 0.5, h: 0.34, margin: 0, fontFace: F.bold, fontSize: 16, color: C.npu });
  s.addText([
    "전력 1/3 (≈180W vs ≈500W)", "와트당 효율 최대 2.03배 (저~중부하)", "b8~b32 TTFT 더 빠름 (최대 3.25배)", "저전력·고밀도 → 랙당 더 많은 서버",
  ].map((t) => ({ text: t, options: { fontFace: F.med, fontSize: 12, color: C.ink2, bullet: { code: "2022", indent: 14 }, breakLine: true } })), { x: M + 0.34, y: cy + 0.72, w: colW - 0.6, h: chh - 0.9, margin: 0, lineSpacingMultiple: 1.62 });
  // GPU 강점
  card(s, M + colW + 0.24, cy, colW, chh, { shadow: shStd(), line: C.gpu });
  accent(s, M + colW + 0.24, cy + 0.26, chh - 0.52, C.gpu);
  s.addText("PRO 6000이 이기는 지점", { x: M + colW + 0.54, y: cy + 0.22, w: colW - 0.5, h: 0.34, margin: 0, fontFace: F.bold, fontSize: 16, color: C.gpu });
  s.addText([
    "raw 처리량 최대 2.6배 (984 vs 377 tok/s)", "단일스트림 1.5배 (38 vs 25 tok/s)", "높은 SLO(30·40) 사용자 수", "고부하(b64↑) 안정적 TTFT",
  ].map((t) => ({ text: t, options: { fontFace: F.med, fontSize: 12, color: C.ink2, bullet: { code: "2022", indent: 14 }, breakLine: true } })), { x: M + colW + 0.58, y: cy + 0.72, w: colW - 0.6, h: chh - 0.9, margin: 0, lineSpacingMultiple: 1.62 });
  // 한 줄 결론 배너
  const by = cy + chh + 0.24, bh = 6.85 - by;
  card(s, M, by, 12.333, bh, { fill: C.dark, line: null, r: 0.16, shadow: shGlow() });
  s.addText("한 줄 결론", { x: M + 0.4, y: by + 0.22, w: 3, h: 0.28, margin: 0, fontFace: F.semi, fontSize: 11, color: C.codeAc });
  s.addText([
    { text: "단일 카드 raw 성능은 RTX PRO 6000이 빠르다. 하지만 RNGD는 같은 작업을 1/3 전력으로 처리해 와트당 효율이 최대 2배 — ", options: { fontFace: F.med, color: C.codeTx } },
    { text: "블로그가 강조한 '효율' 우위는 1장 실측으로도 방향이 확인된다.", options: { fontFace: F.bold, color: C.white } },
    { text: " 블로그의 1.8~2x '사용자 수'를 재현하려면 RNGD를 4장으로 묶고 랙 전력 기준으로 정규화해야 한다.", options: { fontFace: F.med, color: C.codeTx } },
  ], { x: M + 0.4, y: by + 0.56, w: 12.333 - 0.8, h: bh - 0.8, margin: 0, fontSize: 15, lineSpacingMultiple: 1.5 });
})();

pptx.writeFile({ fileName: "/home/jun/RNGD-proj/Model_Benchmark/ppt/RNGD_vs_PRO6000.pptx" })
  .then(() => console.log(`OK: ${TOTAL} slides`))
  .catch((e) => { console.error(e); process.exit(1); });
