/* RNGD NPU vs NVIDIA GPU — 코드 생성 모델 서빙 벤치마크 덱 (16:9)
 * Design.md (Brandlogy) 준수. 테스트 번호 체계: 테스트 1~5.
 * 데이터 출처: rngd-npu/REPORT.md · bench-gpu/REPORT.md · README_npu_gpu_result.md
 * NPU = 파랑, GPU = 주황으로 일관 표기. */
const pptx = new (require("pptxgenjs"))();
pptx.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pptx.layout = "W";
pptx.author = "RNGD Benchmark";
const TOTAL = 19;

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
  codeTx: "e5e9ef", codeMut: "8ea0b5", codeAc: "5fc6ff",
};
const shStd = () => ({ type: "outer", color: "000000", opacity: 0.08, blur: 6, offset: 2, angle: 90 });
const shGlow = () => ({ type: "outer", color: "2c1e74", opacity: 0.16, blur: 15, offset: 0, angle: 90 });
const M = 0.5, CW = 13.333 - 2 * M;

function frame(s, chapter, page, source) {
  s.background = { color: C.white };
  s.addText(chapter.toUpperCase(), {
    x: M, y: 0.4, w: 9, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 12, color: C.mut, charSpacing: 0.8,
  });
  s.addText(`${page} / ${TOTAL}`, {
    x: M, y: 7.05, w: 3, h: 0.25, margin: 0, fontFace: F.med, fontSize: 10, color: C.mut,
  });
  s.addText(source, {
    x: 13.333 - M - 8, y: 7.05, w: 8, h: 0.25, margin: 0,
    fontFace: F.reg, fontSize: 9.5, color: C.mut, align: "right",
  });
}
function title(s, head, sub) {
  s.addText(head, {
    x: M, y: 1.0, w: CW, h: 0.7, margin: 0,
    fontFace: F.bold, fontSize: 31, color: C.ink, charSpacing: -0.6, lineSpacingMultiple: 1.18,
  });
  s.addText(sub, {
    x: M, y: 1.73, w: CW, h: 0.4, margin: 0,
    fontFace: F.med, fontSize: 15, color: C.ink2, lineSpacingMultiple: 1.4,
  });
}
function card(s, x, y, w, h, opt = {}) {
  s.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h, rectRadius: opt.r || 0.13,
    fill: { color: opt.fill || C.white },
    line: opt.line === null ? { type: "none" } : { color: opt.line || C.border, width: 1 },
    shadow: opt.shadow,
  });
}
function tag(s, x, y, text, fill, txtColor, fs) {
  const w = 0.26 + text.length * 0.092;
  s.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h: 0.28, rectRadius: 0.14, fill: { color: fill }, line: { type: "none" },
  });
  s.addText(text, {
    x, y, w, h: 0.28, margin: 0, align: "center", valign: "middle",
    fontFace: F.semi, fontSize: fs || 9.5, color: txtColor || C.white,
  });
  return w;
}
function accent(s, x, y, h, color) {
  s.addShape(pptx.ShapeType.roundRect, { x, y, w: 0.07, h, rectRadius: 0.03, fill: { color }, line: { type: "none" } });
}
function codeCard(s, x, y, w, h, label, lines, fs) {
  s.addShape(pptx.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.1, fill: { color: C.dark }, line: { type: "none" }, shadow: shStd() });
  let ty = y + 0.16;
  if (label) {
    s.addText(label, { x: x + 0.24, y: ty, w: w - 0.48, h: 0.24, margin: 0, fontFace: F.semi, fontSize: 10, color: C.codeMut });
    ty += 0.32;
  }
  s.addText(lines.map((ln) => ({
    text: ln.t, options: { fontFace: F.reg, fontSize: fs || 10, color: ln.c || C.codeTx, breakLine: true },
  })), { x: x + 0.24, y: ty, w: w - 0.48, h: y + h - ty - 0.14, margin: 0, lineSpacingMultiple: 1.3, valign: "top" });
}
// 방법 스트립: 무엇을 / 입력 / 출력·분석 3카드
function methodStrip(s, y, items) {
  const h = 1.2, w = (CW - 2 * 0.2) / 3;
  items.forEach(([lab, txt, ac], i) => {
    const x = M + i * (w + 0.2);
    card(s, x, y, w, h, { shadow: shStd() });
    tag(s, x + 0.2, y + 0.16, lab, ac);
    s.addText(txt, {
      x: x + 0.2, y: y + 0.5, w: w - 0.4, h: h - 0.62, margin: 0,
      fontFace: F.reg, fontSize: 9.8, color: C.ink2, lineSpacingMultiple: 1.34,
    });
  });
}
function resultLabel(s, y) {
  s.addText("결과", { x: M, y, w: 2, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 13, color: C.blue });
}
// NPU/GPU 범례 칩
function legendNG(s, x, y) {
  s.addShape(pptx.ShapeType.rect, { x, y: y + 0.04, w: 0.22, h: 0.13, fill: { color: C.npu }, line: { type: "none" } });
  s.addText("NPU (RNGD)", { x: x + 0.3, y: y - 0.04, w: 1.7, h: 0.28, margin: 0, fontFace: F.semi, fontSize: 9.5, color: C.ink2 });
  s.addShape(pptx.ShapeType.rect, { x: x + 1.95, y: y + 0.04, w: 0.22, h: 0.13, fill: { color: C.gpu }, line: { type: "none" } });
  s.addText("GPU (A6000)", { x: x + 2.25, y: y - 0.04, w: 1.7, h: 0.28, margin: 0, fontFace: F.semi, fontSize: 9.5, color: C.ink2 });
}

/* ===================== 1 — Cover ===================== */
(() => {
  const s = pptx.addSlide();
  s.background = { color: C.white };
  s.addText("FURIOSA RNGD vs NVIDIA A6000 · 2026.05", {
    x: M, y: 0.55, w: 9, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 12, color: C.mut, charSpacing: 1,
  });
  s.addText("RNGD NPU vs NVIDIA GPU\n코드 생성 모델 서빙 벤치마크", {
    x: M, y: 1.25, w: 12.4, h: 1.7, margin: 0, fontFace: F.bold, fontSize: 42, color: C.ink,
    charSpacing: -0.8, lineSpacingMultiple: 1.12,
  });
  s.addText("같은 모델·같은 조건으로 Furiosa RNGD와 NVIDIA A6000에서 5개 테스트를 측정·비교", {
    x: M, y: 2.95, w: 12, h: 0.4, margin: 0, fontFace: F.med, fontSize: 15, color: C.ink2,
  });
  const hx = M, hy = 3.62, hw = 12.333, hh = 1.78;
  s.addShape(pptx.ShapeType.roundRect, { x: hx, y: hy, w: hw, h: hh, rectRadius: 0.22, fill: { color: C.blue }, line: { type: "none" }, shadow: shGlow() });
  s.addShape(pptx.ShapeType.roundRect, { x: hx + 0.4, y: hy + 0.3, w: 1.16, h: 0.32, rectRadius: 0.16, fill: { color: C.white }, line: { type: "none" } });
  s.addText("핵심 결론", { x: hx + 0.4, y: hy + 0.3, w: 1.16, h: 0.32, margin: 0, align: "center", valign: "middle", fontFace: F.semi, fontSize: 10, color: C.blue });
  s.addText([
    { text: "8B 모델 단일 속도는 ", options: { fontFace: F.med, color: "dbe6ff" } },
    { text: "NPU가 30% 빠르다", options: { fontFace: F.bold, color: C.white } },
  ], { x: hx + 0.4, y: hy + 0.62, w: hw - 0.8, h: 0.55, margin: 0, fontSize: 25 });
  s.addText("Llama-3.1-8B 단일 출력 NPU 54.5 vs GPU 41.9 tok/s · 동시 사용자 32명 부근에서 GPU 역전 · 코드 정확도(SWE-bench)는 두 디바이스 모두 0% — 정확도는 하드웨어가 아닌 모델 크기가 결정",
    { x: hx + 0.4, y: hy + 1.18, w: hw - 0.8, h: 0.5, margin: 0, fontFace: F.med, fontSize: 11, color: "dbe6ff", lineSpacingMultiple: 1.3 });
  const chips = [
    ["측정 테스트", "5종", "속도 · 동시성 · serve옵션 · SWE-bench · 임베딩"],
    ["측정 디바이스", "2종", "Furiosa RNGD NPU · NVIDIA A6000 GPU"],
    ["비교 모델", "2종", "Qwen2.5-0.5B(smoke) · Llama-3.1-8B"],
  ];
  const cw = (12.333 - 2 * 0.2) / 3;
  chips.forEach(([l, n, d], i) => {
    const x = M + i * (cw + 0.2);
    card(s, x, 5.62, cw, 1.16, { shadow: shStd() });
    s.addText(l, { x: x + 0.22, y: 5.76, w: cw - 0.44, h: 0.24, margin: 0, fontFace: F.semi, fontSize: 11, color: C.mut });
    s.addText(n, { x: x + 0.22, y: 5.97, w: cw - 0.44, h: 0.42, margin: 0, fontFace: F.bold, fontSize: 22, color: C.blue });
    s.addText(d, { x: x + 0.22, y: 6.42, w: cw - 0.44, h: 0.3, margin: 0, fontFace: F.reg, fontSize: 9, color: C.ink2 });
  });
  s.addText("NPU: furiosa-llm 2026.2.0 · RNGD 1카드   |   GPU: vLLM 0.10.0 · RTX A6000 48GB", {
    x: 13.333 - M - 9, y: 7.05, w: 9, h: 0.25, margin: 0, fontFace: F.reg, fontSize: 9.5, color: C.mut, align: "right",
  });
})();

/* ===================== 2 — 검증 목표 & 평가 기준 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Objective", 2, "");
  title(s, "무엇을, 어떤 기준으로 검증했나",
    "RNGD NPU와 A6000 GPU에서 코드 생성 모델 서빙을 4개 축으로 측정·비교");
  const gy = 2.39, gh = 1.05;
  card(s, M, gy, 12.333, gh, { fill: C.blue, line: null, r: 0.16, shadow: shGlow() });
  s.addText("검증 목표", { x: M + 0.35, y: gy + 0.16, w: 3, h: 0.26, margin: 0, fontFace: F.semi, fontSize: 11, color: "dbe6ff" });
  s.addText("Furiosa RNGD NPU가 코드 생성 워크로드에서 NVIDIA GPU 대비 어느 지점에 강하고 어느 지점에 약한지를, 같은 모델·같은 조건의 정량 측정으로 가려낸다.", {
    x: M + 0.35, y: gy + 0.42, w: 11.6, h: 0.5, margin: 0, fontFace: F.med, fontSize: 13.5, color: C.white, lineSpacingMultiple: 1.35,
  });
  const crit = [
    ["속도", "토큰 생성 속도", "단일 요청에서 첫 토큰 지연(TTFT)과 초당 생성 토큰 수(tok/s)를 NPU·GPU 각각 측정", "테스트 1", C.blue],
    ["동시성·배치", "동시 접속 확장성", "동시 요청을 1→128로 늘리며 합산 처리량을 측정 — NPU·GPU의 교차점을 찾는다", "테스트 2 · 3", C.blue2],
    ["정확도", "SWE-bench", "실제 GitHub 이슈를 코드 패치로 해결하는 정확도 — 하드웨어 의존성 검증", "테스트 4", C.blue3],
    ["환경", "서빙·자동화", "furiosa-llm·vLLM 서빙 구성과 모델 추가만으로 도는 측정 파이프라인 구축", "테스트 5 · 전반", C.pink],
  ];
  const cy = gy + gh + 0.2, ch = 6.85 - cy, cw = (12.333 - 3 * 0.2) / 4;
  crit.forEach(([tg, ti, d, task, ac], i) => {
    const x = M + i * (cw + 0.2);
    card(s, x, cy, cw, ch, { shadow: shStd() });
    accent(s, x, cy + 0.24, 0.42, ac);
    s.addText(`0${i + 1}`, { x: x + 0.24, y: cy + 0.22, w: cw - 0.4, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 15, color: ac });
    s.addText(ti, { x: x + 0.24, y: cy + 0.62, w: cw - 0.44, h: 0.34, margin: 0, fontFace: F.bold, fontSize: 15, color: C.ink });
    s.addText(tg, { x: x + 0.24, y: cy + 0.96, w: cw - 0.44, h: 0.26, margin: 0, fontFace: F.semi, fontSize: 10.5, color: ac });
    s.addText(d, { x: x + 0.24, y: cy + 1.34, w: cw - 0.46, h: ch - 2.05, margin: 0, fontFace: F.reg, fontSize: 11, color: C.ink2, lineSpacingMultiple: 1.5 });
    s.addShape(pptx.ShapeType.line, { x: x + 0.24, y: cy + ch - 0.66, w: cw - 0.48, h: 0, line: { color: C.border, width: 1 } });
    s.addText(`해당 테스트 · ${task}`, { x: x + 0.24, y: cy + ch - 0.54, w: cw - 0.46, h: 0.42, margin: 0, fontFace: F.semi, fontSize: 9.5, color: ac });
  });
})();

/* ===================== 3 — 측정 환경 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Environment", 3, "");
  title(s, "측정 환경 — 같은 클라이언트, 다른 디바이스",
    "측정 도구는 OpenAI 호환 API만 호출 → 서버 계층만 바꿔 NPU·GPU를 같은 코드로 측정");
  const fy = 2.5, fh = 1.62;
  // 공통 클라이언트
  card(s, M, fy, 3.3, fh, { shadow: shStd() });
  accent(s, M, fy + 0.26, fh - 0.52, C.ink2);
  s.addText("측정 클라이언트 (공통)", { x: M + 0.3, y: fy + 0.24, w: 2.9, h: 0.34, margin: 0, fontFace: F.bold, fontSize: 13, color: C.ink });
  s.addText("httpx 비동기 스트리밍\n/v1/chat/completions 호출\ntps·sweep·swebench 코드 100% 공유", { x: M + 0.3, y: fy + 0.62, w: 2.85, h: fh - 0.8, margin: 0, fontFace: F.reg, fontSize: 10, color: C.ink2, lineSpacingMultiple: 1.4 });
  s.addText("→", { x: M + 3.3, y: fy, w: 0.5, h: fh, margin: 0, align: "center", valign: "middle", fontFace: F.bold, fontSize: 22, color: C.mut });
  // NPU / GPU 분기
  const bw = 4.3, bx0 = M + 3.86;
  [
    ["NPU 경로", "furiosa-llm serve", "RNGD NPU · prebuilt 아티팩트 실행", C.npu],
    ["GPU 경로", "vllm serve", "A6000 GPU · 원본 HF 모델 실행", C.gpu],
  ].forEach(([t, eng, d, ac], i) => {
    const x = bx0 + i * (bw + 0.2);
    card(s, x, fy, bw, fh, { shadow: shStd(), line: ac });
    accent(s, x, fy + 0.26, fh - 0.52, ac);
    s.addText(t, { x: x + 0.3, y: fy + 0.22, w: bw - 0.5, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 11, color: ac });
    s.addText(eng, { x: x + 0.3, y: fy + 0.5, w: bw - 0.5, h: 0.34, margin: 0, fontFace: F.bold, fontSize: 15, color: C.ink });
    s.addText(d, { x: x + 0.3, y: fy + 0.88, w: bw - 0.55, h: fh - 1.0, margin: 0, fontFace: F.reg, fontSize: 10.5, color: C.ink2, lineSpacingMultiple: 1.35 });
  });
  const dy = fy + fh + 0.22, dh = 6.85 - dy;
  const items = [
    ["서빙", "OpenAI 호환 API", "furiosa-llm·vLLM 모두 동일한 /v1 endpoint를 제공 → 측정 도구가 디바이스를 구분하지 않는다."],
    ["측정", "스트리밍 클라이언트", "응답을 토큰 단위 스트림으로 받아 TTFT·토큰 간 지연을 실측. 동시 요청은 비동기로 발생."],
    ["채점", "Docker harness", "SWE-bench 채점에만 Docker 사용 — NPU·GPU 동일하게 인스턴스별 격리 컨테이너에서 테스트 실행."],
  ];
  const iw = (12.333 - 2 * 0.2) / 3;
  items.forEach(([tg, ti, d], i) => {
    const x = M + i * (iw + 0.2);
    card(s, x, dy, iw, dh, { shadow: shStd() });
    tag(s, x + 0.24, dy + 0.22, tg, [C.blue, C.blue2, C.pink][i]);
    s.addText(ti, { x: x + 0.24, y: dy + 0.58, w: iw - 0.46, h: 0.32, margin: 0, fontFace: F.bold, fontSize: 13.5, color: C.ink });
    s.addText(d, { x: x + 0.24, y: dy + 0.92, w: iw - 0.46, h: dh - 1.1, margin: 0, fontFace: F.reg, fontSize: 10.5, color: C.ink2, lineSpacingMultiple: 1.45 });
  });
})();

/* ===================== 4 — 측정 대상 & 하드웨어 제약 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Targets", 4, "출처: 각 모델 artifact.json · model.parallel_config");
  title(s, "7종 중 4종만 RNGD 2-카드에서 측정 가능",
    "prebuilt 아티팩트의 tensor-parallel 크기가 필요 NPU 수를 고정한다");
  const cx = M, cy = 2.39, cw = 7.5, ch = 4.46;
  card(s, cx, cy, cw, ch, { shadow: shStd() });
  s.addText("모델별 필요 PE (RNGD 1카드 = 8 PE)", { x: cx + 0.25, y: cy + 0.18, w: cw - 0.5, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 14, color: C.ink });
  const labels = ["Qwen2.5-0.5B", "Llama-3.1-8B", "Qwen3-Embed-8B", "Qwen3-Rerank-8B", "Qwen3-32B-FP8", "EXAONE-4.0-32B", "Llama-3.3-70B"];
  s.addChart(pptx.ChartType.bar, [{ name: "필요 PE", labels, values: [4, 8, 8, 8, 32, 32, 32] }], {
    x: cx + 0.1, y: cy + 0.55, w: cw - 0.3, h: ch - 1.0, barDir: "bar",
    chartColors: [C.blue, C.blue, C.blue, C.blue, C.pink, C.pink, C.pink],
    valAxisMinVal: 0, valAxisMaxVal: 40,
    showValue: true, dataLabelFontSize: 9.5, dataLabelColor: C.ink, dataLabelFontFace: F.semi, dataLabelPosition: "outEnd",
    catAxisLabelFontFace: F.med, catAxisLabelFontSize: 9.5, catAxisLabelColor: C.ink2,
    valAxisLabelFontFace: F.reg, valAxisLabelFontSize: 9, valAxisLabelColor: C.mut,
    valAxisLineColor: C.border2, catAxisLineColor: C.border2,
    valGridLine: { style: "solid", color: C.border2, size: 0.5 }, showLegend: false,
    chartArea: { fill: { color: C.white } },
  });
  s.addText("파랑 = 2카드(16 PE)로 측정 가능 · 분홍 = 4카드 필요 → 측정 불가", { x: cx + 0.25, y: cy + ch - 0.36, w: cw - 0.5, h: 0.26, margin: 0, fontFace: F.reg, fontSize: 9, color: C.mut });
  const rx = M + cw + 0.24, rw = 12.333 - cw - 0.24;
  const rows = [
    ["측정 대상 4종", "Qwen2.5-0.5B · Llama-3.1-8B · Qwen3-Embedding-8B · Qwen3-Reranker-8B — 모두 1카드(≤8 PE)", C.blue],
    ["측정 제외 3종", "Qwen3-32B-FP8 · EXAONE-4.0-32B-FP8 · Llama-3.3-70B — 아티팩트가 tp=32(4카드)로 컴파일", C.pink],
    ["NPU↔GPU 비교 모델", "생성 모델 Qwen2.5-0.5B·Llama-3.1-8B 2종을 GPU에서도 동일 측정 — 이 덱의 비교 대상", C.gpu],
  ];
  const rh = (4.46 - 2 * 0.18) / 3;
  rows.forEach(([t, d, ac], i) => {
    const y = cy + i * (rh + 0.18);
    card(s, rx, y, rw, rh, { shadow: shStd() });
    accent(s, rx, y + 0.22, rh - 0.44, ac);
    s.addText(t, { x: rx + 0.28, y: y + 0.2, w: rw - 0.5, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 14, color: C.ink });
    s.addText(d, { x: rx + 0.28, y: y + 0.54, w: rw - 0.5, h: rh - 0.72, margin: 0, fontFace: F.reg, fontSize: 10.5, color: C.ink2, lineSpacingMultiple: 1.35 });
  });
})();

/* ===================== 5 — 비교 조건 정렬 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Fair Comparison", 5, "출처: rngd-npu·bench-gpu configs/models.yaml");
  title(s, "공정 비교 — 엔진 외 모든 변수를 일치",
    "서빙 엔진(furiosa-llm vs vLLM)만 다르고, 나머지 측정 조건은 NPU·GPU 동일");
  const ty = 2.42, tw = 7.7, th = 3.5;
  card(s, M, ty, tw, th, { shadow: shStd() });
  s.addText("조건 정렬 표", { x: M + 0.25, y: ty + 0.16, w: tw - 0.5, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 13, color: C.ink });
  const rows = [
    ["항목", "NPU (RNGD)", "GPU (A6000)"],
    ["측정 모델", "Qwen2.5-0.5B · Llama-3.1-8B", "동일 (원본 HF)"],
    ["정밀도", "bf16 (양자화 없음)", "bf16 (양자화 없음)"],
    ["컨텍스트 — Qwen", "4,096 (artifact 한도)", "4,096 (--max-model-len)"],
    ["컨텍스트 — Llama", "32,768 (artifact 한도)", "32,768 (--max-model-len)"],
    ["prefix caching", "ON", "ON (--enable-prefix-caching)"],
    ["sweep 그리드", "동시성 1–128 × prompt 256/1K/4K", "동일"],
    ["SWE-bench", "Lite oracle · single-shot · 50건", "동일"],
  ];
  const colX = [0, 2.55, 4.95], colW = [2.55, 2.4, 2.35];
  const txx = M + 0.25, trh = 0.355, t0 = ty + 0.5;
  rows.forEach((r, ri) => {
    const y = t0 + ri * trh;
    if (ri === 0) s.addShape(pptx.ShapeType.rect, { x: txx, y, w: tw - 0.5, h: trh, fill: { color: C.border }, line: { type: "none" } });
    else if (ri % 2 === 0) s.addShape(pptx.ShapeType.rect, { x: txx, y, w: tw - 0.5, h: trh, fill: { color: "fafafa" }, line: { type: "none" } });
    r.forEach((c, ci) => {
      s.addText(c, {
        x: txx + colX[ci] + 0.06, y, w: colW[ci] - 0.1, h: trh, margin: 0, valign: "middle",
        fontFace: ri === 0 ? F.semi : (ci === 0 ? F.semi : F.reg), fontSize: 9.4,
        color: ri === 0 ? C.ink : (ci === 0 ? C.ink : C.ink2),
      });
    });
  });
  // 우측: 검증 + 주의
  const rx = M + tw + 0.24, rw = 12.333 - tw - 0.24;
  card(s, rx, ty, rw, 2.04, { fill: C.blue, line: null, r: 0.16, shadow: shGlow() });
  s.addText("정렬이 실제로 됐다는 증거", { x: rx + 0.28, y: ty + 0.2, w: rw - 0.56, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 13, color: C.white });
  s.addText("sweep에서 prompt=4096 셀은 NPU·GPU 모두 전건 실패한다 (prompt 4096 + max_tokens 256 > 4096 한도). 양쪽이 똑같이 실패했다는 것이 4K 컨텍스트가 동일하게 적용됐다는 직접 증거다.", {
    x: rx + 0.28, y: ty + 0.54, w: rw - 0.56, h: 1.4, margin: 0, fontFace: F.med, fontSize: 10.5, color: "dbe6ff", lineSpacingMultiple: 1.42,
  });
  const wy = ty + 2.04 + 0.16, wh = ty + th - wy;
  card(s, rx, wy, rw, wh, { shadow: shStd() });
  accent(s, rx, wy + 0.2, wh - 0.4, C.pink);
  s.addText("주의 — 성능 비교만", { x: rx + 0.28, y: wy + 0.18, w: rw - 0.5, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 12.5, color: C.ink });
  s.addText("A6000과 RNGD는 가격·소비전력 제품군이 다르다. 이 덱은 동일 SW 조건의 처리 성능만 비교하며, 전력당·비용당 성능(TCO)은 별도 측정 과제다.", {
    x: rx + 0.28, y: wy + 0.5, w: rw - 0.52, h: wh - 0.66, margin: 0, fontFace: F.reg, fontSize: 9.8, color: C.ink2, lineSpacingMultiple: 1.4,
  });
  // 하단 스트립
  card(s, M, 6.32, 12.333, 0.5, { fill: C.bg2, line: null });
  s.addText([
    { text: "처음 GPU 측정은 컨텍스트가 8배(32K) 차이 나 불공정했다  ", options: { fontFace: F.med, fontSize: 10.5, color: C.ink2 } },
    { text: "→ NPU artifact 한도(4K·32K)에 맞춰 재측정한 결과가 이 덱의 데이터다.", options: { fontFace: F.bold, fontSize: 10.5, color: C.blue } },
  ], { x: M + 0.3, y: 6.32, w: 11.7, h: 0.5, margin: 0, valign: "middle" });
})();

/* ===================== 6 — 자동화 파이프라인 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Pipeline", 6, "");
  title(s, "측정 파이프라인 — 모델 추가만으로 자동화",
    "configs/models.yaml에 모델을 등록하면 서버 기동부터 리포트까지 자동 실행 (NPU·GPU 동일)");
  const steps = [
    ["01", "모델 정의", "models.yaml에 모델 id·역할·serve 옵션 등록"],
    ["02", "서버 기동", "furiosa-llm / vllm serve 실행, 헬스체크 통과까지 대기"],
    ["03", "테스트 실행", "테스트 1~5를 순차 측정"],
    ["04", "결과 저장", "results/<모델>/<테스트>/*.json 자동 기록"],
    ["05", "집계·리포트", "analyze · report로 비교표·종합 리포트 생성"],
  ];
  const sy = 2.55, sh = 1.95, sw = (12.333 - 4 * 0.22) / 5;
  steps.forEach(([n, t, d], i) => {
    const x = M + i * (sw + 0.22);
    const feat = i >= 1 && i <= 3;
    card(s, x, sy, sw, sh, { shadow: shStd() });
    s.addShape(pptx.ShapeType.ellipse, { x: x + 0.24, y: sy + 0.24, w: 0.5, h: 0.5, fill: { color: feat ? C.blue : C.blue3 }, line: { type: "none" } });
    s.addText(n, { x: x + 0.24, y: sy + 0.24, w: 0.5, h: 0.5, margin: 0, align: "center", valign: "middle", fontFace: F.bold, fontSize: 13, color: C.white });
    s.addText(t, { x: x + 0.22, y: sy + 0.86, w: sw - 0.4, h: 0.32, margin: 0, fontFace: F.bold, fontSize: 13.5, color: C.ink });
    s.addText(d, { x: x + 0.22, y: sy + 1.2, w: sw - 0.42, h: sh - 1.36, margin: 0, fontFace: F.reg, fontSize: 9.8, color: C.ink2, lineSpacingMultiple: 1.4 });
    if (i < 4) s.addText("›", { x: x + sw - 0.04, y: sy, w: 0.3, h: sh, margin: 0, align: "center", valign: "middle", fontFace: F.bold, fontSize: 20, color: C.mut });
  });
  const by = sy + sh + 0.22, bh = 6.85 - by;
  card(s, M, by, 12.333, bh, { fill: C.bg2, line: null, shadow: shStd() });
  accent(s, M + 0.02, by + 0.24, bh - 0.48, C.blue);
  s.addText("측정 코드는 NPU·GPU 100% 공유 — server 계층만 다름", { x: M + 0.32, y: by + 0.22, w: 11.6, h: 0.32, margin: 0, fontFace: F.bold, fontSize: 14, color: C.ink });
  s.addText("한 모델의 서버를 띄워 테스트 1·2·4를 측정하고, 테스트 3(memsweep)은 serve 옵션을 바꿔 서버를 재기동하며 측정한다. NPU는 furiosa-llm, GPU는 vLLM으로 서버만 갈아끼울 뿐 tps·sweep·swebench·embed 측정 코드는 같다 — 그래서 두 디바이스 결과를 같은 표에 올릴 수 있다. 새 모델은 models.yaml에 한 줄 추가하면 전 과정이 자동 반복된다.", {
    x: M + 0.32, y: by + 0.6, w: 11.7, h: bh - 0.8, margin: 0, fontFace: F.reg, fontSize: 11, color: C.ink2, lineSpacingMultiple: 1.5,
  });
})();

/* ===================== 7 — 측정 테스트 5종 한눈에 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Test Overview", 7, "");
  title(s, "측정 테스트 5종 한눈에",
    "각 모델·각 디바이스에 대해 아래 5개 테스트를 순차 수행 — 이후 슬라이드는 테스트 번호 순");
  const tests = [
    ["1", "토큰 생성 속도", "단일 요청 스트리밍으로 첫 토큰 지연·생성 속도 측정", "지표 TTFT · tok/s", C.blue],
    ["2", "동시성 스케일링", "동시 요청을 1→128로 늘리며 처리량·지연 변화 측정", "지표 합산 TPS · 실패율", C.blue2],
    ["3", "serve 옵션", "serve 인자를 바꿔가며 처리량 영향 측정 (memsweep)", "지표 조합별 합산 TPS", C.blue3],
    ["4", "SWE-bench", "실제 GitHub 이슈를 코드 패치로 해결, 테스트 통과 채점", "지표 resolved %", C.pink],
    ["5", "임베딩·리랭커", "검색 보조 모델의 배치별 처리량 측정", "지표 inputs/s", C.mut],
  ];
  const cy = 2.39, ch = 4.46, cw = (12.333 - 4 * 0.18) / 5;
  tests.forEach(([n, name, d, metric, ac], i) => {
    const x = M + i * (cw + 0.18);
    card(s, x, cy, cw, ch, { shadow: shStd() });
    s.addShape(pptx.ShapeType.roundRect, { x, y: cy, w: cw, h: 0.62, rectRadius: 0.13, fill: { color: ac }, line: { type: "none" } });
    s.addShape(pptx.ShapeType.rect, { x, y: cy + 0.32, w: cw, h: 0.3, fill: { color: ac }, line: { type: "none" } });
    s.addText(`테스트 ${n}`, { x: x + 0.2, y: cy, w: cw - 0.4, h: 0.62, margin: 0, valign: "middle", fontFace: F.bold, fontSize: 14, color: C.white });
    s.addText(name, { x: x + 0.2, y: cy + 0.8, w: cw - 0.4, h: 0.6, margin: 0, fontFace: F.bold, fontSize: 14, color: C.ink, lineSpacingMultiple: 1.15 });
    s.addText(d, { x: x + 0.2, y: cy + 1.5, w: cw - 0.4, h: 1.7, margin: 0, fontFace: F.reg, fontSize: 10, color: C.ink2, lineSpacingMultiple: 1.45 });
    s.addShape(pptx.ShapeType.line, { x: x + 0.2, y: cy + ch - 0.62, w: cw - 0.4, h: 0, line: { color: C.border, width: 1 } });
    s.addText(metric, { x: x + 0.2, y: cy + ch - 0.5, w: cw - 0.4, h: 0.4, margin: 0, fontFace: F.semi, fontSize: 9.3, color: ac, lineSpacingMultiple: 1.25 });
  });
})();

/* ===================== 8 — 테스트 1: 토큰 생성 속도 (NPU vs GPU) ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Test 1 · Speed", 8, "출처: rngd-npu·bench-gpu results · tps task (concurrency=1)");
  title(s, "테스트 1 — 토큰 생성 속도 (NPU vs GPU)",
    "단일 요청 스트리밍 — 8B는 NPU가, 0.5B는 GPU가 빠르다");
  methodStrip(s, 2.36, [
    ["무엇을", "동시 사용자 1명일 때 체감 속도 — 코드 자동완성·대화형 보조의 핵심 지표.", C.blue],
    ["입력", "고정 프롬프트를 /v1/chat/completions에 스트리밍 요청. max_tokens 256, 워밍업 5 + 측정 50회.", C.blue],
    ["출력·분석", "요청별 TTFT·ITL(토큰 간 간격)·출력 tok/s를 NPU·GPU 각각 p50으로 집계.", C.blue],
  ]);
  resultLabel(s, 3.74);
  const ry = 4.06, rh = 6.85 - ry, kw = (12.333 - 0.24) / 2;
  const panels = [
    ["Llama-3.1-8B-Instruct", "코드 생성 후보", 54.5, 41.9, "tok/s", "NPU +30%", C.npu,
      "TTFT  NPU 32.6ms / GPU 35.9ms      ITL  NPU 18.3ms / GPU 23.9ms"],
    ["Qwen2.5-0.5B-Instruct", "smoke · 검증용", 84.5, 249.0, "tok/s", "GPU 3.0×", C.gpu,
      "TTFT  NPU 30.6ms / GPU 13.5ms      ITL  NPU 11.9ms / GPU 4.0ms"],
  ];
  panels.forEach(([name, role, npu, gpu, unit, badge, winColor, sub], i) => {
    const x = M + i * (kw + 0.24);
    card(s, x, ry, kw, rh, { shadow: shStd() });
    s.addText(name, { x: x + 0.3, y: ry + 0.2, w: kw - 1.8, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 14.5, color: C.ink });
    s.addText(role, { x: x + 0.3, y: ry + 0.48, w: kw - 1.8, h: 0.26, margin: 0, fontFace: F.med, fontSize: 10, color: C.mut });
    tag(s, x + kw - 1.4, ry + 0.24, badge, winColor, C.white, 10);
    // NPU / GPU 숫자 블록
    const half = (kw - 0.6) / 2;
    [["NPU", npu, C.npu], ["GPU", gpu, C.gpu]].forEach(([lab, val, ac], j) => {
      const bx = x + 0.3 + j * half;
      s.addText(lab, { x: bx, y: ry + 0.92, w: half - 0.2, h: 0.24, margin: 0, fontFace: F.semi, fontSize: 11, color: ac });
      s.addText([
        { text: String(val), options: { fontFace: F.bold, fontSize: 40, color: ac } },
        { text: " " + unit, options: { fontFace: F.semi, fontSize: 12, color: C.ink2 } },
      ], { x: bx, y: ry + 1.18, w: half - 0.1, h: 0.7, margin: 0 });
    });
    s.addShape(pptx.ShapeType.line, { x: x + 0.3 + half, y: ry + 0.95, w: 0, h: 0.95, line: { color: C.border2, width: 1 } });
    s.addText("단일 요청 출력 토큰 처리량", { x: x + 0.3, y: ry + 1.92, w: kw - 0.6, h: 0.26, margin: 0, fontFace: F.med, fontSize: 10.5, color: C.mut });
    s.addShape(pptx.ShapeType.line, { x: x + 0.3, y: ry + 2.24, w: kw - 0.6, h: 0, line: { color: C.border, width: 1 } });
    s.addText(sub, { x: x + 0.3, y: ry + 2.36, w: kw - 0.6, h: 0.34, margin: 0, fontFace: F.semi, fontSize: 10, color: C.ink2 });
  });
  card(s, M, 6.32, 12.333, 0.5, { fill: C.bg2, line: null });
  s.addText([
    { text: "왜 갈리나  ", options: { fontFace: F.bold, fontSize: 10.5, color: C.blue } },
    { text: "RNGD는 수 B 이상 모델에서 연산 파이프라인 효율이 살아난다 — 8B는 NPU 우위, 0.5B는 너무 작아 메모리 대역폭 싸움이 되어 GPU 우위.", options: { fontFace: F.med, fontSize: 10.5, color: C.ink2 } },
  ], { x: M + 0.3, y: 6.32, w: 11.7, h: 0.5, margin: 0, valign: "middle" });
})();

/* ===================== 9 — 테스트 2: 동시성 스케일링 (NPU vs GPU) ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Test 2 · Concurrency", 9, "출처: results · sweep task (Llama-3.1-8B, prompt 1024)");
  title(s, "테스트 2 — 동시성 스케일링 (NPU vs GPU)",
    "동시 요청 1→128 — c16까지 NPU 우세, c32 교차, c128은 GPU가 66% 앞선다");
  const ry = 2.45, rh = 6.85 - ry, chw = 7.5;
  card(s, M, ry, chw, rh, { shadow: shStd() });
  s.addText("동시성별 합산 처리량 (tok/s)", { x: M + 0.25, y: ry + 0.14, w: 3.0, h: 0.28, margin: 0, fontFace: F.semi, fontSize: 12.5, color: C.ink });
  legendNG(s, M + chw - 4.05, ry + 0.16);
  const cc = ["1", "2", "4", "8", "16", "32", "64", "128"];
  s.addChart(pptx.ChartType.line, [
    { name: "NPU (RNGD)", labels: cc, values: [52, 100, 192, 335, 637, 1091, 1647, 2192] },
    { name: "GPU (A6000)", labels: cc, values: [41, 80, 151, 272, 580, 1122, 2048, 3634] },
  ], {
    x: M + 0.1, y: ry + 0.5, w: chw - 0.3, h: rh - 0.9,
    chartColors: [C.npu, C.gpu], lineSize: 2.75, lineSmooth: true,
    lineDataSymbol: "circle", lineDataSymbolSize: 5, showValue: false,
    valAxisMinVal: 0, valAxisMaxVal: 4000,
    catAxisTitle: "동시 요청 수", showCatAxisTitle: true, catAxisTitleFontSize: 9, catAxisTitleColor: C.mut, catAxisTitleFontFace: F.med,
    catAxisLabelFontFace: F.med, catAxisLabelFontSize: 9, catAxisLabelColor: C.ink2,
    valAxisLabelFontFace: F.reg, valAxisLabelFontSize: 8.5, valAxisLabelColor: C.mut,
    valGridLine: { style: "solid", color: C.border2, size: 0.5 },
    valAxisLineColor: C.border2, catAxisLineColor: C.border2, showLegend: false,
  });
  const rx = M + chw + 0.24, rw = 12.333 - chw - 0.24;
  const cards = [
    ["c1–c16", "NPU 우세 구간", "동시 16명까지 NPU가 빠르다 (c8: 335 vs 272 tok/s, +23%). 소수 사용자 코드 어시스턴트·IDE 플러그인에 적합.", C.npu],
    ["c32", "교차점", "동시 32명 부근에서 역전 (1091 vs 1122). 이 지점이 NPU·GPU 선택 기준선.", C.mut],
    ["c64–c128", "GPU 우세 구간", "GPU의 배치 처리량이 앞선다 (c128: 3634 vs 2192, +66%). 대규모 동시 트래픽 공용 API에 적합.", C.gpu],
  ];
  const ih = (rh - 2 * 0.16) / 3;
  cards.forEach(([tg, ti, d, ac], i) => {
    const y = ry + i * (ih + 0.16);
    card(s, rx, y, rw, ih, { shadow: shStd() });
    accent(s, rx, y + 0.18, ih - 0.36, ac);
    s.addText([
      { text: tg + "   ", options: { fontFace: F.bold, fontSize: 12, color: ac } },
      { text: ti, options: { fontFace: F.bold, fontSize: 13, color: C.ink } },
    ], { x: rx + 0.26, y: y + 0.16, w: rw - 0.5, h: 0.3, margin: 0 });
    s.addText(d, { x: rx + 0.26, y: y + 0.5, w: rw - 0.5, h: ih - 0.66, margin: 0, fontFace: F.reg, fontSize: 9.8, color: C.ink2, lineSpacingMultiple: 1.36 });
  });
})();

/* ===================== 10 — 테스트 3: serve 옵션 (memsweep) NPU vs GPU ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Test 3 · Serve Options", 10, "출처: results · memsweep task (Llama-3.1-8B, 동시성 128)");
  title(s, "테스트 3 — serve 옵션 튜닝 (NPU vs GPU)",
    "서버 옵션을 한 축씩 바꿔(OFAT) 합산 처리량 영향 측정");
  methodStrip(s, 2.36, [
    ["무엇을", "KV cache·배치 관련 serve 옵션이 처리량에 주는 영향을 NPU·GPU 각각 측정.", C.blue3],
    ["입력", "baseline + max-model-len·배치 슬롯·max-num-batched-tokens 한 축씩 변경, 조합마다 서버 재기동.", C.blue3],
    ["출력·분석", "조합별 합산 처리량 비교 → 어떤 옵션이 유의미한지, 디바이스별 민감도 차이 판단.", C.blue3],
  ]);
  resultLabel(s, 3.74);
  const ry = 4.06, rh = 6.85 - ry, chw = 7.5;
  card(s, M, ry, chw, rh, { shadow: shStd() });
  s.addText("serve 옵션별 합산 처리량 (tok/s)", { x: M + 0.25, y: ry + 0.13, w: 3.1, h: 0.26, margin: 0, fontFace: F.semi, fontSize: 12, color: C.ink });
  legendNG(s, M + chw - 4.05, ry + 0.14);
  s.addChart(pptx.ChartType.bar, [
    { name: "NPU", labels: ["baseline", "max-model-len 4096", "max-num-batched 16384", "배치 슬롯 8"], values: [2251, 2272, 2139, 2232] },
    { name: "GPU", labels: ["baseline", "max-model-len 4096", "max-num-batched 16384", "배치 슬롯 8"], values: [3616, 3671, 3613, 306] },
  ], {
    x: M + 0.1, y: ry + 0.46, w: chw - 0.3, h: rh - 0.82, barDir: "bar", barGapWidthPct: 40,
    chartColors: [C.npu, C.gpu], valAxisMinVal: 0, valAxisMaxVal: 4000,
    showValue: true, dataLabelFontSize: 8, dataLabelFontFace: F.semi, dataLabelPosition: "outEnd", dataLabelColor: C.ink,
    catAxisLabelFontFace: F.med, catAxisLabelFontSize: 8.4, catAxisLabelColor: C.ink2,
    valAxisLabelFontFace: F.reg, valAxisLabelFontSize: 8, valAxisLabelColor: C.mut,
    valGridLine: { style: "solid", color: C.border2, size: 0.5 },
    valAxisLineColor: C.border2, catAxisLineColor: C.border2, showLegend: false,
  });
  const rx = M + chw + 0.24, rw = 12.333 - chw - 0.24;
  const cards = [
    ["정상 옵션은 튜닝 효과 작음", "max-model-len·max-num-batched-tokens를 바꿔도 처리량 변동은 양쪽 모두 수 % 이내. 기본값 서빙으로 충분하다.", C.blue],
    ["배치 슬롯 축소 — 결정적 차이", "슬롯을 8로 줄이면 GPU(vLLM)는 처리량이 92% 붕괴(3616→306). NPU(furiosa-llm)는 −1%로 거의 무영향.", C.pink],
    ["운영 시사점", "GPU는 max-num-seqs를 충분히 크게 둬야 한다. NPU는 이 옵션에 둔감 — 튜닝 부담이 작다.", C.gpu],
  ];
  const ih = (rh - 2 * 0.16) / 3;
  cards.forEach(([t, d, ac], i) => {
    const y = ry + i * (ih + 0.16);
    card(s, rx, y, rw, ih, { shadow: shStd() });
    accent(s, rx, y + 0.16, ih - 0.32, ac);
    s.addText(t, { x: rx + 0.26, y: y + 0.14, w: rw - 0.5, h: 0.46, margin: 0, fontFace: F.bold, fontSize: 11.5, color: C.ink, lineSpacingMultiple: 1.1 });
    s.addText(d, { x: rx + 0.26, y: y + 0.58, w: rw - 0.5, h: ih - 0.72, margin: 0, fontFace: F.reg, fontSize: 9.5, color: C.ink2, lineSpacingMultiple: 1.34 });
  });
})();

/* ===================== 11 — 테스트 4 SWE-bench ①: 정의 & 종류 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Test 4 · SWE-bench ①", 11, "출처: SWE-bench (Princeton NLP) · arXiv 2310.06770");
  title(s, "테스트 4 — SWE-bench ①: 정의 & 종류",
    "GitHub 이슈를 패치로 해결하고 실제 repo 테스트 통과 여부로 채점하는 벤치마크");
  const lw = 5.4, ly = 2.39, lh = 4.46;
  card(s, M, ly, lw, lh, { shadow: shStd() });
  s.addText("채점 방식", { x: M + 0.26, y: ly + 0.2, w: lw - 0.5, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 13.5, color: C.ink });
  const flow = [
    ["입력", "GitHub 이슈 본문 + 해당 시점 repo 코드", C.blue],
    ["모델 출력", "이슈를 해결하는 unified diff (패치)", C.blue2],
    ["적용", "Docker 컨테이너에서 repo에 패치 적용", C.blue3],
    ["채점", "테스트 실행 — FAIL→PASS 통과 시 resolved", C.pink],
  ];
  const fh = (lh - 0.6) / 4;
  flow.forEach(([t, d, ac], i) => {
    const y = ly + 0.6 + i * fh;
    s.addShape(pptx.ShapeType.ellipse, { x: M + 0.28, y: y + 0.16, w: 0.36, h: 0.36, fill: { color: ac }, line: { type: "none" } });
    s.addText(`${i + 1}`, { x: M + 0.28, y: y + 0.16, w: 0.36, h: 0.36, margin: 0, align: "center", valign: "middle", fontFace: F.bold, fontSize: 11, color: C.white });
    s.addText(t, { x: M + 0.8, y: y + 0.12, w: lw - 1.0, h: 0.26, margin: 0, fontFace: F.bold, fontSize: 12.5, color: C.ink });
    s.addText(d, { x: M + 0.8, y: y + 0.37, w: lw - 1.05, h: fh - 0.42, margin: 0, fontFace: F.reg, fontSize: 10, color: C.ink2, lineSpacingMultiple: 1.3 });
    if (i < 3) s.addText("↓", { x: M + 0.36, y: y + 0.5, w: 0.2, h: fh - 0.34, margin: 0, align: "center", fontFace: F.bold, fontSize: 11, color: C.border2 });
  });
  const rx = M + lw + 0.24, rw = 12.333 - lw - 0.24;
  card(s, rx, ly, rw, 2.66, { shadow: shStd() });
  s.addText("데이터셋 종류", { x: rx + 0.26, y: ly + 0.18, w: rw - 0.5, h: 0.28, margin: 0, fontFace: F.semi, fontSize: 13, color: C.ink });
  const ds = [
    ["SWE-bench (full)", "2,294건 · 원본 12개 repo"],
    ["SWE-bench Lite", "300건 · 저비용 평가용 subset"],
    ["SWE-bench Verified", "500건 · 사람 검증, 신뢰도 최상"],
    ["SWE-bench Multimodal", "517건 · 시각 요소 포함(JS/UI)"],
  ];
  ds.forEach(([a, b], i) => {
    const y = ly + 0.52 + i * 0.5;
    if (i % 2 === 1) s.addShape(pptx.ShapeType.rect, { x: rx + 0.2, y, w: rw - 0.4, h: 0.5, fill: { color: "fafafa" }, line: { type: "none" } });
    s.addText(a, { x: rx + 0.32, y, w: 2.7, h: 0.5, margin: 0, valign: "middle", fontFace: F.semi, fontSize: 11, color: C.ink });
    s.addText(b, { x: rx + 3.05, y, w: rw - 3.25, h: 0.5, margin: 0, valign: "middle", fontFace: F.reg, fontSize: 10.5, color: C.ink2 });
  });
  const sy = ly + 2.66 + 0.16, sh = 6.85 - sy;
  card(s, rx, sy, rw, sh, { fill: C.blue, line: null, r: 0.16, shadow: shGlow() });
  s.addShape(pptx.ShapeType.roundRect, { x: rx + 0.28, y: sy + 0.22, w: 1.0, h: 0.3, rectRadius: 0.15, fill: { color: C.white }, line: { type: "none" } });
  s.addText("본 측정", { x: rx + 0.28, y: sy + 0.22, w: 1.0, h: 0.3, margin: 0, align: "center", valign: "middle", fontFace: F.semi, fontSize: 9.5, color: C.blue });
  s.addText("SWE-bench Lite · oracle · single-shot", { x: rx + 0.28, y: sy + 0.56, w: rw - 0.56, h: 0.32, margin: 0, fontFace: F.bold, fontSize: 14.5, color: C.white });
  s.addText("Lite 300건 중 12개 repo(astropy·django·matplotlib·sympy 등)에 고르게 50건 추출. context는 oracle(정답 파일 제공), 호출은 single-shot(1회 생성) — 에이전트 반복 없이 순수 코드 편집 능력을 본다. NPU·GPU 동일 구성.", {
    x: rx + 0.28, y: sy + 0.92, w: rw - 0.56, h: sh - 1.1, margin: 0, fontFace: F.med, fontSize: 10.3, color: "dbe6ff", lineSpacingMultiple: 1.42,
  });
})();

/* ===================== 12 — 테스트 4 SWE-bench ②: 시작 가이드 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Test 4 · SWE-bench ②", 12, "출처: swebench.com/guides/quickstart · docs/SWEBENCH_SETUP.md");
  title(s, "테스트 4 — SWE-bench ②: 시작 가이드",
    "Docker 기반 채점 — 인스턴스별 격리 컨테이너에서 repo 테스트 실행");
  const lw = 7.0, ly = 2.39;
  codeCard(s, M, ly, lw, 1.66, "① 설치", [
    { t: "$ git clone https://github.com/SWE-bench/SWE-bench.git", c: C.codeAc },
    { t: "$ cd SWE-bench && pip install -e .", c: C.codeAc },
    { t: "$ docker --version    # Docker 동작 확인", c: C.codeMut },
  ], 10.5);
  codeCard(s, M, ly + 1.82, lw, 2.64, "② 채점 실행", [
    { t: "$ python -m swebench.harness.run_evaluation \\", c: C.codeAc },
    { t: "    --dataset_name princeton-nlp/SWE-bench_Lite \\", c: C.codeTx },
    { t: "    --predictions_path predictions.jsonl \\", c: C.codeTx },
    { t: "    --max_workers 8 \\", c: C.codeTx },
    { t: "    --namespace swebench \\", c: C.codeTx },
    { t: "    --run_id my_run", c: C.codeTx },
    { t: "# --namespace swebench → prebuilt 이미지 pull", c: C.codeMut },
    { t: "# 결과: <model>.<run_id>.json (resolved 카운트)", c: C.codeMut },
  ], 10.5);
  const rx = M + lw + 0.24, rw = 12.333 - lw - 0.24;
  card(s, rx, ly, rw, 2.0, { shadow: shStd() });
  s.addText("요구사항", { x: rx + 0.26, y: ly + 0.18, w: rw - 0.5, h: 0.28, margin: 0, fontFace: F.semi, fontSize: 13, color: C.ink });
  const req = ["x86_64 Linux · Docker", "디스크 수십~120GB (인스턴스 이미지)", "Python 3.10+ · swebench 패키지"];
  req.forEach((r, i) => {
    s.addShape(pptx.ShapeType.ellipse, { x: rx + 0.3, y: ly + 0.58 + i * 0.42, w: 0.1, h: 0.1, fill: { color: C.blue }, line: { type: "none" } });
    s.addText(r, { x: rx + 0.52, y: ly + 0.46 + i * 0.42, w: rw - 0.8, h: 0.34, margin: 0, valign: "middle", fontFace: F.reg, fontSize: 10.8, color: C.ink2 });
  });
  card(s, rx, ly + 2.16, rw, 2.3, { shadow: shStd() });
  s.addText("예측 파일 (predictions.jsonl)", { x: rx + 0.26, y: ly + 2.32, w: rw - 0.5, h: 0.28, margin: 0, fontFace: F.semi, fontSize: 13, color: C.ink });
  s.addText("한 줄 = 한 인스턴스. 모델이 만든 패치를 model_patch에 담는다:", {
    x: rx + 0.26, y: ly + 2.6, w: rw - 0.5, h: 0.5, margin: 0, fontFace: F.reg, fontSize: 10, color: C.ink2, lineSpacingMultiple: 1.4,
  });
  codeCard(s, rx + 0.26, ly + 3.06, rw - 0.52, 1.2, null, [
    { t: '{"instance_id": "astropy__..."', c: C.codeTx },
    { t: ' "model_name_or_path": "8B",', c: C.codeTx },
    { t: ' "model_patch": "diff --git ..."}', c: C.codeAc },
  ], 9.5);
})();

/* ===================== 13 — 테스트 4 SWE-bench ③: 측정 방법 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Test 4 · SWE-bench ③", 13, "예시: astropy__astropy-12907");
  title(s, "테스트 4 — SWE-bench ③: 측정 방법",
    "oracle 프롬프트를 로컬 서버(furiosa-llm/vLLM)에 보내 diff를 받고 harness로 채점");
  const y = 2.39, h = 4.46, w = (12.333 - 2 * 0.22) / 3;
  card(s, M, y, w, h, { shadow: shStd() });
  tag(s, M + 0.22, y + 0.2, "입력", C.blue);
  s.addText("oracle 프롬프트", { x: M + 0.22, y: y + 0.54, w: w - 0.44, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 13, color: C.ink });
  s.addText("이슈 설명 + 수정 대상 파일 코드", { x: M + 0.22, y: y + 0.84, w: w - 0.44, h: 0.3, margin: 0, fontFace: F.reg, fontSize: 9.8, color: C.ink2 });
  codeCard(s, M + 0.22, y + 1.18, w - 0.44, h - 1.4, null, [
    { t: "<issue>", c: C.codeMut },
    { t: "separability_matrix 가", c: C.codeTx },
    { t: "nested CompoundModel 에서", c: C.codeTx },
    { t: "분리성을 잘못 계산함", c: C.codeTx },
    { t: "</issue>", c: C.codeMut },
    { t: "", c: C.codeTx },
    { t: "[start of separable.py]", c: C.codeMut },
    { t: "def _separable(transform):", c: C.codeAc },
    { t: "    ...", c: C.codeTx },
  ], 9.5);
  card(s, M + w + 0.22, y, w, h, { shadow: shStd() });
  tag(s, M + w + 0.44, y + 0.2, "출력", C.blue2);
  s.addText("모델 생성 diff", { x: M + w + 0.44, y: y + 0.54, w: w - 0.44, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 13, color: C.ink });
  s.addText("Llama-3.1-8B single-shot 응답", { x: M + w + 0.44, y: y + 0.84, w: w - 0.44, h: 0.3, margin: 0, fontFace: F.reg, fontSize: 9.8, color: C.ink2 });
  codeCard(s, M + w + 0.44, y + 1.18, w - 0.44, h - 1.4, null, [
    { t: "--- a/.../separable.py", c: C.codeMut },
    { t: "+++ b/.../separable.py", c: C.codeMut },
    { t: "@@ -304,6 +304,10 @@", c: C.codeAc },
    { t: "  elif isinstance(", c: C.codeTx },
    { t: "      transform, Compound):", c: C.codeTx },
    { t: "+   if isinstance(", c: "7ee787" },
    { t: "+       transform.left,...):", c: "7ee787" },
    { t: "+     sepleft =", c: "7ee787" },
    { t: "+       _separable(...)", c: "7ee787" },
  ], 9.5);
  card(s, M + 2 * (w + 0.22), y, w, h, { shadow: shStd() });
  tag(s, M + 2 * (w + 0.22) + 0.22, y + 0.2, "채점", C.pink);
  s.addText("Docker harness", { x: M + 2 * (w + 0.22) + 0.22, y: y + 0.54, w: w - 0.44, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 13, color: C.ink });
  s.addText("패치 적용 후 테스트 실행", { x: M + 2 * (w + 0.22) + 0.22, y: y + 0.84, w: w - 0.44, h: 0.3, margin: 0, fontFace: F.reg, fontSize: 9.8, color: C.ink2 });
  const gx = M + 2 * (w + 0.22) + 0.22, gw = w - 0.44;
  const judge = [
    ["resolved", "패치 적용 + 테스트 통과", C.ok],
    ["unresolved", "적용됐으나 테스트 미통과", C.mut],
    ["적용실패", "malformed diff — 적용 불가", C.pink],
  ];
  judge.forEach(([t, d, ac], i) => {
    const jy = y + 1.2 + i * 1.04;
    card(s, gx, jy, gw, 0.9, { fill: "fafafa", line: C.border });
    accent(s, gx, jy + 0.16, 0.58, ac);
    s.addText(t, { x: gx + 0.22, y: jy + 0.12, w: gw - 0.4, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 12, color: C.ink });
    s.addText(d, { x: gx + 0.22, y: jy + 0.42, w: gw - 0.4, h: 0.4, margin: 0, fontFace: F.reg, fontSize: 9.5, color: C.ink2, lineSpacingMultiple: 1.3 });
  });
})();

/* ===================== 14 — 테스트 4 SWE-bench ④: 결과 (NPU vs GPU) ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Test 4 · SWE-bench ④", 14, "출처: SWE-bench Lite oracle 50건 · Docker harness 채점");
  title(s, "테스트 4 — SWE-bench ④: 결과 (NPU vs GPU)",
    "두 디바이스 모두 resolved 0% — 같은 모델은 디바이스가 달라도 같은 정확도");
  // verdict 배너
  const vy = 2.39, vh = 0.8;
  card(s, M, vy, 12.333, vh, { fill: C.blue, line: null, r: 0.16, shadow: shGlow() });
  s.addShape(pptx.ShapeType.roundRect, { x: M + 0.32, y: vy + 0.24, w: 0.92, h: 0.32, rectRadius: 0.16, fill: { color: C.white }, line: { type: "none" } });
  s.addText("결론", { x: M + 0.32, y: vy + 0.24, w: 0.92, h: 0.32, margin: 0, align: "center", valign: "middle", fontFace: F.semi, fontSize: 10, color: C.blue });
  s.addText("NPU·GPU 4개 조합 전부 resolved 0 — 정확도는 디바이스가 아니라 모델 크기·방식이 결정한다", {
    x: M + 1.4, y: vy, w: 11.2, h: vh, margin: 0, valign: "middle", fontFace: F.bold, fontSize: 14.5, color: C.white,
  });
  // 결과 표
  const ty = vy + vh + 0.2, tw = 7.6, tch = 6.85 - ty;
  card(s, M, ty, tw, tch, { shadow: shStd() });
  s.addText("채점 결과 — 4개 조합", { x: M + 0.25, y: ty + 0.16, w: tw - 0.5, h: 0.28, margin: 0, fontFace: F.semi, fontSize: 13, color: C.ink });
  const rows = [
    ["모델 · 디바이스", "resolved", "미해결", "적용실패", "평가건수"],
    ["Llama-3.1-8B · NPU", "0", "14", "36", "50"],
    ["Llama-3.1-8B · GPU", "0", "18", "29", "47"],
    ["Qwen2.5-0.5B · NPU", "0", "1", "2", "3"],
    ["Qwen2.5-0.5B · GPU", "0", "1", "2", "3"],
  ];
  const colX = [0, 2.55, 3.65, 4.8, 6.05], colW = [2.55, 1.1, 1.15, 1.25, 1.05];
  const txx = M + 0.25, trh = 0.52, t0 = ty + 0.52;
  rows.forEach((r, ri) => {
    const y = t0 + ri * trh;
    if (ri === 0) s.addShape(pptx.ShapeType.rect, { x: txx, y, w: tw - 0.5, h: trh, fill: { color: C.border }, line: { type: "none" } });
    else if (ri % 2 === 0) s.addShape(pptx.ShapeType.rect, { x: txx, y, w: tw - 0.5, h: trh, fill: { color: "fafafa" }, line: { type: "none" } });
    r.forEach((c, ci) => {
      s.addText(c, {
        x: txx + colX[ci] + 0.06, y, w: colW[ci] - 0.08, h: trh, margin: 0, valign: "middle",
        fontFace: ri === 0 || ci === 0 ? F.semi : F.reg, fontSize: ci === 0 ? 10 : 10.5,
        color: ri === 0 ? C.ink : (ci === 1 ? C.pink : (ci === 0 ? C.ink : C.ink2)),
        align: ci === 0 ? "left" : "center",
      });
    });
  });
  s.addText("미해결 = 적용됐으나 테스트 실패 · 적용실패 = malformed diff. Qwen은 4K 컨텍스트라 47건이 평가 불가(제외) → 3건만 채점.", {
    x: M + 0.25, y: t0 + 5 * trh + 0.08, w: tw - 0.5, h: 0.6, margin: 0, fontFace: F.reg, fontSize: 8.6, color: C.mut, lineSpacingMultiple: 1.4,
  });
  // 우측 해석
  const rx = M + tw + 0.24, rw = 12.333 - tw - 0.24;
  const notes = [
    ["resolved 0 — 모델 한계", "8B·0.5B를 single-shot으로 돌리면 어느 하드웨어에서도 SWE-bench Lite를 못 푼다.", C.pink],
    ["적용실패율 60~70%", "NPU Llama 36/50, GPU Llama 29/47 — 모델이 구조적으로 올바른 diff를 잘 못 만든다.", C.mut],
    ["디바이스 독립 확인", "같은 모델의 NPU·GPU 결과가 거의 일치 → NPU 전환에 따른 정확도 손해는 없다.", C.blue],
  ];
  const nh = (tch - 2 * 0.16) / 3;
  notes.forEach(([t, d, ac], i) => {
    const y = ty + i * (nh + 0.16);
    card(s, rx, y, rw, nh, { shadow: shStd() });
    accent(s, rx, y + 0.16, nh - 0.32, ac);
    s.addText(t, { x: rx + 0.26, y: y + 0.15, w: rw - 0.48, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 12, color: C.ink });
    s.addText(d, { x: rx + 0.26, y: y + 0.47, w: rw - 0.5, h: nh - 0.62, margin: 0, fontFace: F.reg, fontSize: 9.8, color: C.ink2, lineSpacingMultiple: 1.4 });
  });
})();

/* ===================== 15 — 테스트 4 SWE-bench ⑤: 심층 분석 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Test 4 · SWE-bench ⑤", 15, "출처: 예측 jsonl 분석 · SWE-bench·SWE-RL 논문(공개 GPU 수치)");
  title(s, "테스트 4 — SWE-bench ⑤: 왜 0%인가",
    "single-shot 8B의 한계 — 같은 방식이면 GPU·프런티어 모델도 한 자릿수");
  const vy = 2.39, vh = 0.8;
  card(s, M, vy, 12.333, vh, { fill: C.blue, line: null, r: 0.16, shadow: shGlow() });
  s.addShape(pptx.ShapeType.roundRect, { x: M + 0.32, y: vy + 0.24, w: 0.92, h: 0.32, rectRadius: 0.16, fill: { color: C.white }, line: { type: "none" } });
  s.addText("결론", { x: M + 0.32, y: vy + 0.24, w: 0.92, h: 0.32, margin: 0, align: "center", valign: "middle", fontFace: F.semi, fontSize: 10, color: C.blue });
  s.addText("정확도를 가르는 건 하드웨어가 아니라 모델 크기·스캐폴드 — 개선은 32B/70B 또는 에이전트로만",
    { x: M + 1.4, y: vy, w: 11.2, h: vh, margin: 0, valign: "middle", fontFace: F.bold, fontSize: 14, color: C.white });
  // 좌: 두 실패 유형
  const ty = vy + vh + 0.2, lw = 5.7, lh = 6.85 - ty;
  s.addText("Llama-3.1-8B 두 실패 유형 — 모두 모델 역량 문제", { x: M + 0.04, y: ty, w: lw, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 12.5, color: C.ink });
  const types = [
    ["적용실패 (NPU 36 · GPU 29)", "모델이 diff의 @@ hunk 헤더 줄 수를 잘못 계산 → patch 도구가 거부. 8B가 정확한 unified diff 포맷을 못 맞춘다.", C.pink],
    ["미해결 (NPU 14 · GPU 18)", "문법이 맞는 패치가 적용됐으나 테스트 미통과 — 수정 위치·내용이 틀림. 단발 8B 코드 추론의 한계.", C.mut],
  ];
  const th = (lh - 0.36 - 0.16) / 2;
  types.forEach(([t, d, ac], i) => {
    const y = ty + 0.36 + i * (th + 0.16);
    card(s, M, y, lw, th, { shadow: shStd() });
    accent(s, M, y + 0.18, th - 0.36, ac);
    s.addText(t, { x: M + 0.3, y: y + 0.16, w: lw - 0.55, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 12.5, color: C.ink });
    s.addText(d, { x: M + 0.3, y: y + 0.5, w: lw - 0.58, h: th - 0.64, margin: 0, fontFace: F.reg, fontSize: 10, color: C.ink2, lineSpacingMultiple: 1.4 });
  });
  // 우: 공개 GPU 수치 표
  const rx = M + lw + 0.22, rw = 12.333 - lw - 0.22;
  card(s, rx, ty, rw, lh, { shadow: shStd() });
  s.addText("공개 SWE-bench 결과 — 단발 생성(에이전트 없음)", { x: rx + 0.24, y: ty + 0.15, w: rw - 0.48, h: 0.28, margin: 0, fontFace: F.semi, fontSize: 12, color: C.ink });
  const rows = [
    ["모델 (규모)", "벤치 · N", "resolved"],
    ["Llama-3.1-8B (본 측정·8B)", "Lite · 50", "0 %"],
    ["GPT-4 (프런티어)", "full · 2294", "1.7 %"],
    ["Claude 2 (프런티어)", "full · 2294", "4.8 %"],
    ["Llama-3.3-70B (70B)", "Verified · 500", "5.4 %"],
  ];
  const colX = [0, 3.15, 4.75], colW = [3.15, 1.6, 1.05];
  const txx = rx + 0.24, trh = 0.40, t0 = ty + 0.52;
  rows.forEach((r, ri) => {
    const y = t0 + ri * trh;
    if (ri === 0) s.addShape(pptx.ShapeType.rect, { x: txx, y, w: rw - 0.48, h: trh, fill: { color: C.border }, line: { type: "none" } });
    else if (ri === 1) s.addShape(pptx.ShapeType.roundRect, { x: txx, y, w: rw - 0.48, h: trh, rectRadius: 0.04, fill: { color: "fdeef7" }, line: { color: C.pink, width: 1 } });
    else if (ri % 2 === 1) s.addShape(pptx.ShapeType.rect, { x: txx, y, w: rw - 0.48, h: trh, fill: { color: "fafafa" }, line: { type: "none" } });
    r.forEach((c, ci) => {
      s.addText(c, {
        x: txx + colX[ci] + 0.08, y, w: colW[ci] - 0.12, h: trh, margin: 0, valign: "middle",
        fontFace: ri === 0 ? F.semi : (ci === 0 || ci === 2 ? F.bold : F.reg), fontSize: ri === 0 ? 9.5 : 9.6,
        color: ri === 0 ? C.ink : (ci === 2 ? (ri === 1 ? C.pink : C.blue) : (ci === 0 ? C.ink : C.ink2)),
        align: ci === 2 ? "right" : "left",
      });
    });
  });
  card(s, rx + 0.24, ty + lh - 0.86, rw - 0.48, 0.7, { fill: C.bg2, line: null });
  s.addText([
    { text: "핵심   ", options: { fontFace: F.bold, fontSize: 9.5, color: C.blue } },
    { text: "같은 single-shot이면 8B보다 큰 GPT-4·Claude 2·70B도 1.7~5.4%. 정확도는 더 큰 모델(4카드 32B·70B)이나 에이전트 스캐폴드로만 오른다.", options: { fontFace: F.med, fontSize: 9.3, color: C.ink2 } },
  ], { x: rx + 0.42, y: ty + lh - 0.86, w: rw - 0.84, h: 0.7, margin: 0, valign: "middle", lineSpacingMultiple: 1.32 });
})();

/* ===================== 16 — 테스트 5: 임베딩 / 리랭커 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Test 5 · Embedding", 16, "출처: rngd-npu/results · embed / rerank task");
  title(s, "테스트 5 — 임베딩 · 리랭커",
    "검색 보조 모델(Qwen3-Embedding-8B · Qwen3-Reranker-8B)의 처리량 측정 — NPU 단독");
  methodStrip(s, 2.39, [
    ["무엇을", "임베딩/리랭킹 처리량 — SWE-bench 검색 보조(RAG)에 쓰이는 모델.", C.mut],
    ["입력", "/v1/embeddings·/v1/rerank에 batch {1·4·16·64}개 입력.", C.mut],
    ["출력·분석", "초당 처리 건수(inputs/s), 배치 크기별 효율.", C.mut],
  ]);
  resultLabel(s, 3.78);
  const ry = 4.12, rh = 6.85 - ry;
  const kp = [["Qwen3-Embedding-8B", "1.17", "inputs/s", "batch 1·16·64 모두 동일 — 배치 이득 없음"],
    ["Qwen3-Reranker-8B", "1.17", "pairs/s", "쿼리당 100문서 ≈ 85초 — 항목당 약 0.85초"]];
  const kw = 3.9;
  kp.forEach(([n, v, u, note], i) => {
    const x = M + i * (kw + 0.2);
    card(s, x, ry, kw, rh, { shadow: shStd() });
    s.addText(n, { x: x + 0.28, y: ry + 0.24, w: kw - 0.56, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 12.5, color: C.mut });
    s.addText([
      { text: v, options: { fontFace: F.bold, fontSize: 40, color: C.blue } },
      { text: "  " + u, options: { fontFace: F.semi, fontSize: 13, color: C.ink2 } },
    ], { x: x + 0.28, y: ry + 0.58, w: kw - 0.56, h: 0.7, margin: 0 });
    s.addText(note, { x: x + 0.28, y: ry + 1.36, w: kw - 0.56, h: 0.5, margin: 0, fontFace: F.reg, fontSize: 9.8, color: C.ink2, lineSpacingMultiple: 1.35 });
  });
  const dx = M + 2 * (kw + 0.2), dw = 12.333 - 2 * (kw + 0.2);
  card(s, dx, ry, dw, rh, { shadow: shStd() });
  accent(s, dx, ry + 0.26, rh - 0.52, C.pink);
  s.addText("처리량 이상 — 별도 점검 필요", { x: dx + 0.3, y: ry + 0.24, w: dw - 0.55, h: 0.34, margin: 0, fontFace: F.bold, fontSize: 14, color: C.ink });
  s.addText("두 모델 모두 항목당 약 0.85초 — 8B 생성 모델의 단일 prefill(수십 ms) 대비 수십 배 느리다. batch size를 키워도 처리량이 1.17/s로 고정돼 배칭이 동작하지 않는다. prebuilt 아티팩트가 단일 배치(batch=1)로 컴파일된 것으로 추정 — 대량 검색에 부적합하며 배칭 지원 아티팩트 재컴파일이 후속 과제. GPU 측은 미측정(embedding 모델 보류).", {
    x: dx + 0.3, y: ry + 0.64, w: dw - 0.6, h: rh - 0.9, margin: 0, fontFace: F.reg, fontSize: 10.5, color: C.ink2, lineSpacingMultiple: 1.5,
  });
})();

/* ===================== 17 — 측정 모델 노트: Qwen2.5-0.5B ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Model Note · 0.5B", 17, "출처: results · Qwen2.5-0.5B-Instruct (NPU·GPU)");
  title(s, "측정 모델 노트 — Qwen2.5-0.5B-Instruct",
    "속도는 GPU가 압도하지만 컨텍스트 4,096 한계로 코드 작업엔 부적합 — smoke 전용");
  const vy = 2.39, vh = 0.82;
  card(s, M, vy, 12.333, vh, { fill: C.blue, line: null, r: 0.16, shadow: shGlow() });
  s.addShape(pptx.ShapeType.roundRect, { x: M + 0.32, y: vy + 0.25, w: 1.16, h: 0.32, rectRadius: 0.16, fill: { color: C.white }, line: { type: "none" } });
  s.addText("smoke", { x: M + 0.32, y: vy + 0.25, w: 1.16, h: 0.32, margin: 0, align: "center", valign: "middle", fontFace: F.semi, fontSize: 10, color: C.blue });
  s.addText("자동화 파이프라인(서빙→측정→채점)이 정상 동작하는지 빠르게 확인하는 검증용 모델 — models.yaml에 role: smoke로 지정된, 코드 생성 후보가 아닌 모델.", {
    x: M + 1.66, y: vy, w: 10.4, h: vh, margin: 0, valign: "middle", fontFace: F.med, fontSize: 11.5, color: C.white, lineSpacingMultiple: 1.3,
  });
  const cy = vy + vh + 0.18, cBot = 6.30, ch = cBot - cy;
  const cw = (12.333 - 0.22) / 2;
  const cols = [
    ["속도 — 0.5B는 GPU가 압도", C.gpu, [
      ["249 / 84", "단일 출력 tok/s (GPU / NPU) — 작은 모델은 메모리 대역폭 싸움 → GPU 3배."],
      ["11.4K / 4.1K", "동시성 c128 합산 tok/s (GPU / NPU) — 전 구간 GPU 우세."],
      ["13 / 31 ms", "TTFT p50 (GPU / NPU) — 첫 토큰도 GPU가 빠르다."],
    ]],
    ["한계 — 컨텍스트 4,096 토큰", C.pink, [
      ["4,096", "max_context_len — NPU prebuilt 아티팩트의 고정 버킷. GPU도 동일하게 맞춤."],
      ["0 / 3", "SWE-bench — oracle 프롬프트 47건이 4096 초과로 평가 불가, 3건만 채점."],
      ["sweep 4096", "prompt=4096 셀은 NPU·GPU 모두 전 동시성 실패 — 버킷 초과."],
    ]],
  ];
  cols.forEach(([head, ac, rows], ci) => {
    const x = M + ci * (cw + 0.22);
    card(s, x, cy, cw, ch, { shadow: shStd() });
    s.addText(head, { x: x + 0.26, y: cy + 0.16, w: cw - 0.5, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 12.5, color: ac });
    const rh = (ch - 0.56) / 3, ry0 = cy + 0.54;
    rows.forEach(([v, d], ri) => {
      const y = ry0 + ri * rh;
      if (ri > 0) s.addShape(pptx.ShapeType.line, { x: x + 0.26, y, w: cw - 0.52, h: 0, line: { color: C.border, width: 1 } });
      accent(s, x + 0.26, y + (rh - 0.42) / 2, 0.42, ac);
      s.addText(v, { x: x + 0.42, y: y + 0.06, w: 2.05, h: rh - 0.12, margin: 0, valign: "middle", fontFace: F.bold, fontSize: 16, color: ac });
      s.addText(d, { x: x + 2.55, y: y + 0.06, w: cw - 2.55 - 0.3, h: rh - 0.12, margin: 0, valign: "middle", fontFace: F.reg, fontSize: 9.6, color: C.ink2, lineSpacingMultiple: 1.32 });
    });
  });
  card(s, M, 6.32, 12.333, 0.5, { fill: C.bg2, line: null });
  s.addText([
    { text: "시사점  ", options: { fontFace: F.bold, fontSize: 10.5, color: C.blue } },
    { text: "smoke·단문 고처리량 데모엔 적합 · 코드 생성·긴 컨텍스트 작업엔 부적합 — 실사용하려면 아티팩트를 더 큰 컨텍스트 버킷으로 재컴파일해야 한다.", options: { fontFace: F.med, fontSize: 10.5, color: C.ink2 } },
  ], { x: M + 0.3, y: 6.32, w: 11.7, h: 0.5, margin: 0, valign: "middle" });
})();

/* ===================== 18 — 종합: NPU vs GPU 디바이스 성격 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Synthesis", 18, "출처: README_npu_gpu_result.md");
  title(s, "종합 — NPU와 GPU는 성격이 다른 디바이스",
    "RNGD = 지연(latency) 최적화 · A6000 = 처리량(throughput) 최적화");
  // 좌우 디바이스 성격 카드
  const cy = 2.42, ch = 2.5, cw = (12.333 - 0.22) / 2;
  const devs = [
    ["RNGD NPU", "지연(latency) 머신", C.npu, [
      "8B 단일 출력 속도 +30% (54.5 vs 41.9 tok/s)",
      "토큰 간 지연(ITL)·첫 토큰(TTFT) 모두 짧음",
      "동시성 c16까지 합산 처리량 우세",
      "serve 옵션(배치 슬롯)에 둔감 — 튜닝 부담 작음",
    ]],
    ["NVIDIA A6000 GPU", "처리량(throughput) 머신", C.gpu, [
      "동시성 c128 합산 처리량 +66% (3634 vs 2192)",
      "0.5B 같은 소형 모델 전 구간 우세",
      "동시 32명 이상 대규모 트래픽에 강함",
      "max-num-seqs 충분히 키워야 처리량 유지",
    ]],
  ];
  devs.forEach(([name, role, ac, items], ci) => {
    const x = M + ci * (cw + 0.22);
    card(s, x, cy, cw, ch, { shadow: shStd(), line: ac });
    s.addShape(pptx.ShapeType.roundRect, { x, y: cy, w: cw, h: 0.56, rectRadius: 0.13, fill: { color: ac }, line: { type: "none" } });
    s.addShape(pptx.ShapeType.rect, { x, y: cy + 0.28, w: cw, h: 0.28, fill: { color: ac }, line: { type: "none" } });
    s.addText([
      { text: name + "   ", options: { fontFace: F.bold, fontSize: 14, color: C.white } },
      { text: role, options: { fontFace: F.semi, fontSize: 11, color: "ffffff" } },
    ], { x: x + 0.3, y: cy, w: cw - 0.6, h: 0.56, margin: 0, valign: "middle" });
    items.forEach((it, i) => {
      const y = cy + 0.72 + i * 0.42;
      s.addShape(pptx.ShapeType.ellipse, { x: x + 0.32, y: y + 0.07, w: 0.1, h: 0.1, fill: { color: ac }, line: { type: "none" } });
      s.addText(it, { x: x + 0.54, y, w: cw - 0.8, h: 0.36, margin: 0, valign: "middle", fontFace: F.reg, fontSize: 10.3, color: C.ink2 });
    });
  });
  // 워크로드 권장 표
  const ty = cy + ch + 0.2, th = 6.85 - ty;
  card(s, M, ty, 12.333, th, { shadow: shStd() });
  s.addText("워크로드별 권장 디바이스", { x: M + 0.25, y: ty + 0.14, w: 11.8, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 13, color: C.ink });
  const rows = [
    ["워크로드", "권장", "근거"],
    ["IDE 자동완성·소수 사용자 코드 어시스턴트 (동시 ≤16)", "RNGD NPU", "8B 단일 +30%, c16까지 처리량 우세"],
    ["대화형 챗봇 — 낮은 지연 우선", "RNGD NPU", "TTFT·ITL 모두 NPU가 짧음"],
    ["대규모 공용 API (동시 ≥32)", "A6000 GPU", "c32 이후 처리량 역전, c128 +66%"],
    ["코드 정확도가 핵심", "디바이스 무관", "8B는 둘 다 0% — 32B/70B·에이전트 필요"],
  ];
  const colX = [0, 6.2, 8.0], colW = [6.2, 1.8, 3.83];
  const txx = M + 0.25, trh = (th - 0.5) / 5, t0 = ty + 0.48;
  rows.forEach((r, ri) => {
    const y = t0 + ri * trh;
    if (ri === 0) s.addShape(pptx.ShapeType.rect, { x: txx, y, w: 12.333 - 0.5, h: trh, fill: { color: C.border }, line: { type: "none" } });
    else if (ri % 2 === 0) s.addShape(pptx.ShapeType.rect, { x: txx, y, w: 12.333 - 0.5, h: trh, fill: { color: "fafafa" }, line: { type: "none" } });
    r.forEach((c, ci) => {
      const isNpu = c === "RNGD NPU", isGpu = c === "A6000 GPU";
      s.addText(c, {
        x: txx + colX[ci] + 0.08, y, w: colW[ci] - 0.12, h: trh, margin: 0, valign: "middle",
        fontFace: ri === 0 ? F.semi : (ci === 1 ? F.bold : F.reg), fontSize: ri === 0 ? 10 : 10.2,
        color: ri === 0 ? C.ink : (ci === 1 ? (isNpu ? C.npu : (isGpu ? C.gpu : C.mut)) : C.ink2),
      });
    });
  });
})();

/* ===================== 19 — 결론 & 권장 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "Conclusion", 19, "출처: rngd-npu·bench-gpu REPORT.md · README_npu_gpu_result.md");
  title(s, "결론 — 8B 코드 서빙, 동시성으로 디바이스를 고른다",
    "소수~중간 규모는 RNGD NPU, 대규모 동시 트래픽은 A6000 GPU");
  // 핵심 결론 배너
  const vy = 2.39, vh = 1.04;
  card(s, M, vy, 12.333, vh, { fill: C.blue, line: null, r: 0.16, shadow: shGlow() });
  s.addShape(pptx.ShapeType.roundRect, { x: M + 0.34, y: vy + 0.34, w: 1.5, h: 0.36, rectRadius: 0.18, fill: { color: C.white }, line: { type: "none" } });
  s.addText("최종 결론", { x: M + 0.34, y: vy + 0.34, w: 1.5, h: 0.36, margin: 0, align: "center", valign: "middle", fontFace: F.semi, fontSize: 10.5, color: C.blue });
  s.addText("8B급 코드 생성 모델을 동시 16~32명 이하로 서빙하면 RNGD NPU가 GPU보다 빠르고 안정적이다. 동시 사용자가 수십 명을 넘으면 GPU의 배치 처리량이 앞선다. 코드 정확도는 두 디바이스가 동일 — NPU 전환에 따른 정확도 손해는 없다.", {
    x: M + 2.0, y: vy, w: 10.1, h: vh, margin: 0, valign: "middle", fontFace: F.med, fontSize: 12, color: C.white, lineSpacingMultiple: 1.4,
  });
  // 3개 권장 카드
  const ay = vy + vh + 0.2, ah = 6.85 - ay;
  const acts = [
    ["채택", "소수 사용자 = RNGD NPU", "8B 단일 속도 +30%, c16까지 처리량 우세. 코드 자동완성·대화형 보조에 적합하며 serve 옵션은 기본값으로 충분.", C.npu, true],
    ["분담", "대규모 트래픽 = A6000 GPU", "동시 32명 이상에서 GPU 처리량이 역전(c128 +66%). 공용 API는 GPU, 저지연 구간은 NPU로 분담 가능.", C.gpu, false],
    ["과제", "정확도·임베딩 후속 측정", "정확도는 4카드 32B·70B 측정 필요. NPU 임베딩 1.17/s 이상 현상은 배칭 아티팩트 재컴파일로 점검.", C.pink, false],
  ];
  const aw = (12.333 - 2 * 0.2) / 3;
  acts.forEach(([tg, ti, d, ac, feat], i) => {
    const x = M + i * (aw + 0.2);
    card(s, x, ay, aw, ah, { fill: feat ? C.blue : C.white, line: feat ? null : C.border, shadow: feat ? shGlow() : shStd(), r: feat ? 0.16 : 0.13 });
    if (feat) {
      s.addShape(pptx.ShapeType.roundRect, { x: x + 0.24, y: ay + 0.24, w: 0.9, h: 0.32, rectRadius: 0.16, fill: { color: C.white }, line: { type: "none" } });
      s.addText(tg, { x: x + 0.24, y: ay + 0.24, w: 0.9, h: 0.32, margin: 0, align: "center", valign: "middle", fontFace: F.semi, fontSize: 10, color: C.blue });
    } else {
      tag(s, x + 0.24, ay + 0.24, tg, ac);
    }
    s.addText(ti, { x: x + 0.24, y: ay + 0.66, w: aw - 0.48, h: 0.62, margin: 0, fontFace: F.bold, fontSize: 14.5, color: feat ? C.white : C.ink, lineSpacingMultiple: 1.12 });
    s.addText(d, { x: x + 0.24, y: ay + 1.32, w: aw - 0.5, h: ah - 1.5, margin: 0, fontFace: F.reg, fontSize: 10.3, color: feat ? "dbe6ff" : C.ink2, lineSpacingMultiple: 1.46 });
  });
})();

pptx.writeFile({ fileName: "/home/jun/RNGD-proj/Model_Benchmark/ppt/RNGD_Benchmark.pptx" })
  .then(() => console.log(`OK: ${TOTAL} slides`))
  .catch((e) => { console.error(e); process.exit(1); });
