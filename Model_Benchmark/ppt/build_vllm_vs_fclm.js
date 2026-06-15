/* vLLM vs furiosa-llm — 실행·운영 관점 비교 (16:9)
 * 기존 build_model_prep.js (Paperlogy) 스타일 준수.
 * 내용 출처: 두 패키지 설치본 소스 직접 분석
 *   - furiosa-llm 2026.2.0  ~/furiosa/.../furiosa_llm/
 *   - vLLM 0.10.0           bench-gpu/.venv/.../vllm/
 *   + repo 실제 사용 스크립트(bench-blog/run_rngd.sh, bench-gpu/runners/server.py, configs/models.yaml,
 *     info/README_runcode.md / BUILD_FLOW.md). 각 슬라이드 하단에 파일:라인 표기. */
const pptx = new (require("pptxgenjs"))();
pptx.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pptx.layout = "W";
pptx.author = "RNGD vLLM vs FCLM";
const TOTAL = 11;

const F = {
  black: "Paperlogy 9 Black", xbold: "Paperlogy 8 ExtraBold",
  bold: "Paperlogy 7 Bold", semi: "Paperlogy 6 SemiBold",
  med: "Paperlogy 5 Medium", reg: "Paperlogy 4 Regular",
};
const C = {
  ink: "222222", ink2: "45515e", mut: "8e8e93",
  blue: "1456f0", blue2: "3b82f6", blue3: "60a5fa", blueLt: "bfdbfe", blueBg: "eef3ff",
  vio: "6d28d9", vio2: "8b5cf6", vioLt: "ddd6fe", vioBg: "f5f3ff",
  white: "ffffff", border: "f2f3f5", border2: "e5e7eb", bg2: "f0f0f0",
  dark: "181e25", codeTx: "e5e9ef", codeMut: "8ea0b5", codeAc: "5fc6ff", codeRed: "ff8087", codeGrn: "8ce0a6",
  ok: "16a34a", okBg: "e8ffea", err: "dc2626", errBg: "fde8e8", warn: "d97706", warnBg: "fef3c7",
};
const shStd = () => ({ type: "outer", color: "000000", opacity: 0.08, blur: 6, offset: 2, angle: 90 });
const M = 0.5, CW = 13.333 - 2 * M;

function frame(s, chapter, page, source) {
  s.background = { color: C.white };
  s.addText(chapter.toUpperCase(), { x: M, y: 0.4, w: 9, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 12, color: C.mut, charSpacing: 0.8 });
  s.addText(`${page} / ${TOTAL}`, { x: 13.333 - M - 3, y: 7.08, w: 3, h: 0.25, margin: 0, fontFace: F.med, fontSize: 10, color: C.mut, align: "right" });
  if (source) s.addText(source, { x: M, y: 7.08, w: 9.5, h: 0.25, margin: 0, fontFace: F.reg, fontSize: 8.5, color: C.mut });
}
function title(s, head, sub) {
  s.addText(head, { x: M, y: 0.92, w: CW, h: 0.7, margin: 0, fontFace: F.bold, fontSize: 29, color: C.ink, charSpacing: -0.6, lineSpacingMultiple: 1.16 });
  if (sub) s.addText(sub, { x: M, y: 1.66, w: CW, h: 0.4, margin: 0, fontFace: F.med, fontSize: 14, color: C.ink2, lineSpacingMultiple: 1.4 });
}
function card(s, x, y, w, h, opt = {}) {
  s.addShape(pptx.ShapeType.roundRect, { x, y, w, h, rectRadius: opt.r || 0.12, fill: { color: opt.fill || C.white }, line: opt.line === null ? { type: "none" } : { color: opt.line || C.border2, width: 1 }, shadow: opt.shadow });
}
function tag(s, x, y, text, fill, txtColor, fs) {
  const w = 0.34 + text.length * 0.105;
  s.addShape(pptx.ShapeType.roundRect, { x, y, w, h: 0.32, rectRadius: 0.16, fill: { color: fill }, line: { type: "none" } });
  s.addText(text, { x, y, w, h: 0.32, margin: 0, align: "center", valign: "middle", fontFace: F.semi, fontSize: fs || 10.5, color: txtColor || C.white });
  return w;
}
function accent(s, x, y, h, color) {
  s.addShape(pptx.ShapeType.roundRect, { x, y, w: 0.07, h, rectRadius: 0.03, fill: { color }, line: { type: "none" } });
}
function codeCard(s, x, y, w, h, label, lines, fs) {
  s.addShape(pptx.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.1, fill: { color: C.dark }, line: { type: "none" }, shadow: shStd() });
  let ty = y + 0.18;
  if (label) { s.addText(label, { x: x + 0.26, y: ty, w: w - 0.5, h: 0.24, margin: 0, fontFace: F.semi, fontSize: 10, color: C.codeMut }); ty += 0.34; }
  s.addText(lines.map((ln) => ({ text: ln.t, options: { fontFace: F.reg, fontSize: fs || 10.5, color: ln.c || C.codeTx, breakLine: true } })),
    { x: x + 0.26, y: ty, w: w - 0.5, h: y + h - ty - 0.16, margin: 0, lineSpacingMultiple: 1.3, valign: "top" });
}
// 두 진영 헤더 칩
function engineHead(s, x, y, w, name, sub, col, colLt) {
  s.addShape(pptx.ShapeType.roundRect, { x, y, w, h: 0.5, rectRadius: 0.1, fill: { color: col }, line: { type: "none" } });
  s.addText(name, { x: x + 0.2, y, w: w - 0.4, h: 0.5, margin: 0, valign: "middle", fontFace: F.bold, fontSize: 14, color: C.white });
  if (sub) s.addText(sub, { x: x + 0.2, y, w: w - 0.3, h: 0.5, margin: 0, valign: "middle", align: "right", fontFace: F.med, fontSize: 10, color: colLt });
}

/* ───────────── 1. 표지 ───────────── */
(() => {
  const s = pptx.addSlide();
  s.background = { color: C.dark };
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 13.333, h: 0.12, fill: { color: C.blue }, line: { type: "none" } });
  s.addText("실행 · 운영 관점 비교", { x: M, y: 2.0, w: CW, h: 0.4, margin: 0, fontFace: F.semi, fontSize: 15, color: C.codeAc, charSpacing: 1.2 });
  s.addText([
    { text: "vLLM", options: { fontFace: F.black, fontSize: 52, color: C.white } },
    { text: "  vs  ", options: { fontFace: F.med, fontSize: 30, color: C.codeMut } },
    { text: "furiosa-llm", options: { fontFace: F.black, fontSize: 52, color: C.blue3 } },
  ], { x: M, y: 2.5, w: CW, h: 1.2, margin: 0 });
  s.addText("같은 OpenAI 호환 API, 정반대 실행 모델 — 설치 · 빌드 · serve · 호출 · 튜닝을 실제 명령어로 비교", {
    x: M, y: 3.95, w: CW, h: 0.5, margin: 0, fontFace: F.med, fontSize: 15, color: C.codeTx, lineSpacingMultiple: 1.4 });
  s.addText([
    { text: "furiosa-llm 2026.2.0", options: { fontFace: F.semi, fontSize: 12, color: C.blue3, breakLine: false } },
    { text: "   ·   ", options: { fontFace: F.reg, fontSize: 12, color: C.codeMut, breakLine: false } },
    { text: "vLLM 0.10.0 (V1)", options: { fontFace: F.semi, fontSize: 12, color: C.codeTx, breakLine: false } },
    { text: "   ·   설치본 소스 + repo 실제 스크립트 기반", options: { fontFace: F.reg, fontSize: 12, color: C.codeMut } },
  ], { x: M, y: 5.2, w: CW, h: 0.4, margin: 0 });
  s.addText("2026-06-09", { x: M, y: 6.7, w: 4, h: 0.3, margin: 0, fontFace: F.med, fontSize: 11, color: C.codeMut });
})();

/* ───────────── 2. 핵심 차이: 라이프사이클 ───────────── */
(() => {
  const s = pptx.addSlide();
  frame(s, "한눈에 · 실행 모델", 2, "출처: api.py · cli/convert.py · cli/serve.py · BUILD_FLOW.md · vllm/entrypoints/llm.py");
  title(s, "한 줄 차이 — 2단계 AOT vs 1단계 JIT", "furiosa는 먼저 컴파일해 “아티팩트”를 만든 뒤 그걸 띄웁니다. vLLM은 띄우는 순간 로드·컴파일까지 한 번에.");

  // furiosa row
  const y1 = 2.6;
  s.addText("furiosa-llm", { x: M, y: y1 - 0.05, w: 2.2, h: 0.4, margin: 0, fontFace: F.bold, fontSize: 14, color: C.blue });
  const steps1 = [
    ["HF 모델", C.bg2, C.ink], ["furiosa-llm build", C.blue, C.white], ["아티팩트\n(EDF 바이너리)", C.blueBg, C.blue],
    ["furiosa-llm serve", C.blue2, C.white], ["서버", C.dark, C.white],
  ];
  let x = M, bw = 2.05, gap = 0.42, by = y1 + 0.35;
  steps1.forEach(([t, f, c], i) => {
    card(s, x, by, bw, 0.85, { fill: f, line: null, shadow: shStd(), r: 0.1 });
    s.addText(t, { x, y: by, w: bw, h: 0.85, margin: 0, align: "center", valign: "middle", fontFace: i % 2 ? F.semi : F.bold, fontSize: i % 2 ? 11.5 : 12.5, color: c, lineSpacingMultiple: 1.1 });
    if (i < steps1.length - 1) s.addText("→", { x: x + bw, y: by, w: gap, h: 0.85, margin: 0, align: "center", valign: "middle", fontFace: F.bold, fontSize: 18, color: C.mut });
    x += bw + gap;
  });
  s.addText("느린 오프라인 빌드 1회  →  저장된 바이너리 로드(빠름) · shape·tp·양자화가 빌드에 “박힘”", {
    x: M, y: by + 0.92, w: CW, h: 0.3, margin: 0, fontFace: F.med, fontSize: 10.5, color: C.blue });

  // vllm row
  const y2 = 4.95;
  s.addText("vLLM", { x: M, y: y2 - 0.05, w: 2.2, h: 0.4, margin: 0, fontFace: F.bold, fontSize: 14, color: C.vio });
  const steps2 = [["HF 모델 (또는 로컬)", C.bg2, C.ink], ["vllm serve  =  로드 + torch.compile + CUDA그래프", C.vio, C.white], ["서버", C.dark, C.white]];
  const bw2 = [2.9, 6.4, 2.05]; x = M; const by2 = y2 + 0.35;
  steps2.forEach(([t, f, c], i) => {
    card(s, x, by2, bw2[i], 0.85, { fill: f, line: null, shadow: shStd(), r: 0.1 });
    s.addText(t, { x, y: by2, w: bw2[i], h: 0.85, margin: 0, align: "center", valign: "middle", fontFace: i === 1 ? F.bold : F.semi, fontSize: i === 1 ? 12.5 : 12, color: c, lineSpacingMultiple: 1.1 });
    if (i < steps2.length - 1) s.addText("→", { x: x + bw2[i], y: by2, w: gap, h: 0.85, margin: 0, align: "center", valign: "middle", fontFace: F.bold, fontSize: 18, color: C.mut });
    x += bw2[i] + gap;
  });
  s.addText("별도 빌드 단계 없음  →  매 기동마다 가중치 로드·JIT 컴파일(시작 느림) · tp·양자화·shape를 기동마다 자유 선택", {
    x: M, y: by2 + 0.92, w: CW, h: 0.3, margin: 0, fontFace: F.med, fontSize: 10.5, color: C.vio });
})();

/* ───────────── 3. 설치 / 환경 ───────────── */
(() => {
  const s = pptx.addSlide();
  frame(s, "1. 설치 · 환경", 3, "출처: furiosa-llm METADATA(Requires-Dist) · vLLM 0.10.0+cu126 · info/README_build.md");
  title(s, "설치와 전제 — 전용 SDK vs pip 한 줄", "");
  const cw = (CW - 0.4) / 2, y = 2.5, h = 4.0;
  // furiosa
  engineHead(s, M, y, cw, "furiosa-llm", "RNGD NPU 전용", C.blue, C.blueLt);
  card(s, M, y + 0.62, cw, h - 0.62, { shadow: shStd() });
  codeCard(s, M + 0.25, y + 0.85, cw - 0.5, 1.5, "설치 (Furiosa SDK 저장소)", [
    { t: "# NPU 드라이버 · 펌웨어 · 컴파일러 선설치 필요", c: C.codeMut },
    { t: "pip install furiosa-llm \\", c: C.codeGrn },
    { t: "  --extra-index-url <furiosa-pypi>", c: C.codeTx },
    { t: "furiosa-smi info   # NPU 인식 확인", c: C.codeAc },
  ], 10);
  const fb = [
    "torch 2.10 · transformers 5.1 등 버전 고정(pin)",
    "furiosa-native-runtime / -compiler 동반 설치",
    "NPU 카드 + 드라이버 없으면 serve 불가",
  ];
  fb.forEach((t, i) => { s.addText("•", { x: M + 0.28, y: y + 2.55 + i * 0.42, w: 0.2, h: 0.4, margin: 0, fontFace: F.bold, fontSize: 12, color: C.blue }); s.addText(t, { x: M + 0.5, y: y + 2.5 + i * 0.42, w: cw - 0.75, h: 0.45, margin: 0, fontFace: F.reg, fontSize: 11, color: C.ink2, lineSpacingMultiple: 1.25 }); });
  // vllm
  const x2 = M + cw + 0.4;
  engineHead(s, x2, y, cw, "vLLM", "NVIDIA GPU (CUDA)", C.vio, C.vioLt);
  card(s, x2, y + 0.62, cw, h - 0.62, { shadow: shStd() });
  codeCard(s, x2 + 0.25, y + 0.85, cw - 0.5, 1.5, "설치 (PyPI)", [
    { t: "# CUDA 런타임만 있으면 됨", c: C.codeMut },
    { t: "pip install vllm", c: C.codeGrn },
    { t: "", c: C.codeTx },
    { t: "nvidia-smi   # GPU 인식 확인", c: C.codeAc },
  ], 10);
  const vb = [
    "PyPI 한 줄(+ROCm/TPU/CPU 등 다중 백엔드 휠)",
    "임의 HF 체크포인트 즉시 사용 가능",
    "GPU만 있으면 어디서나 동일하게 동작",
  ];
  vb.forEach((t, i) => { s.addText("•", { x: x2 + 0.28, y: y + 2.55 + i * 0.42, w: 0.2, h: 0.4, margin: 0, fontFace: F.bold, fontSize: 12, color: C.vio }); s.addText(t, { x: x2 + 0.5, y: y + 2.5 + i * 0.42, w: cw - 0.75, h: 0.45, margin: 0, fontFace: F.reg, fontSize: 11, color: C.ink2, lineSpacingMultiple: 1.25 }); });
})();

/* ───────────── 4. 모델 준비 (build) ───────────── */
(() => {
  const s = pptx.addSlide();
  frame(s, "2. 모델 준비", 4, "출처: cli/convert.py(build 플래그) · BUILD_FLOW.md · BUILD_COMPIL.md · vLLM은 별도 준비 단계 없음");
  title(s, "모델 준비 — furiosa만 “빌드”가 필요", "vLLM에는 이 단계 자체가 없습니다. 그래서 furiosa는 첫 준비가 무겁고, 대신 serve가 가볍습니다.");
  const cw = (CW - 0.4) / 2, y = 2.45;
  engineHead(s, M, y, cw, "furiosa-llm", "AOT 컴파일 → 아티팩트", C.blue, C.blueLt);
  codeCard(s, M, y + 0.62, cw, 1.9, "furiosa-llm build", [
    { t: "furiosa-llm build Qwen/Qwen3-32B-FP8 \\", c: C.codeGrn },
    { t: "  ./artifacts/qwen3-32b-fp8-tp8 \\", c: C.codeTx },
    { t: "  -tp 8 \\            # PE 8개 = 카드 1장", c: C.codeAc },
    { t: "  --prefill-buckets ... --decode-buckets ...", c: C.codeTx },
    { t: "  # 수 분~수십 분 (그래프 분할+컴파일)", c: C.codeMut },
  ], 10.5);
  s.addText("산출물: artifact.json + EDF 바이너리 + param.safetensors + tokenizer  →  디스크에 저장, 재사용", {
    x: M, y: y + 2.66, w: cw, h: 0.6, margin: 0, fontFace: F.med, fontSize: 10.5, color: C.blue, lineSpacingMultiple: 1.3 });
  card(s, M, y + 3.35, cw, 0.95, { fill: C.warnBg, line: null });
  accent(s, M + 0.0, y + 3.5, 0.65, C.warn);
  s.addText([{ text: "tp · 양자화 · shape(버킷)가 여기서 “박힘”", options: { fontFace: F.bold, fontSize: 11.5, color: C.warn, breakLine: true } },
    { text: "→ 나중에 바꾸려면 이 build를 다시 돌려야 함", options: { fontFace: F.reg, fontSize: 10.5, color: C.ink2 } }],
    { x: M + 0.25, y: y + 3.45, w: cw - 0.45, h: 0.78, margin: 0, valign: "middle", lineSpacingMultiple: 1.25 });

  const x2 = M + cw + 0.4;
  engineHead(s, x2, y, cw, "vLLM", "준비 단계 없음", C.vio, C.vioLt);
  card(s, x2, y + 0.62, cw, 1.9, { fill: C.vioBg, line: null, shadow: shStd() });
  s.addText("(해당 없음)", { x: x2, y: y + 0.62, w: cw, h: 0.6, margin: 0, align: "center", valign: "middle", fontFace: F.bold, fontSize: 16, color: C.vio2 });
  s.addText("모델 경로(HF id)를 serve에 바로 넘기면 됩니다.\n가중치 로드·컴파일은 모두 기동 시점에 일어납니다.", {
    x: x2 + 0.3, y: y + 1.25, w: cw - 0.6, h: 1.1, margin: 0, align: "center", fontFace: F.reg, fontSize: 12, color: C.ink2, lineSpacingMultiple: 1.4 });
  s.addText("장점 vs 비용", { x: x2, y: y + 2.66, w: cw, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 10.5, color: C.mut });
  card(s, x2, y + 3.0, cw, 1.3, { line: null, fill: C.vioBg });
  s.addText([
    { text: "+  사전 빌드 필요 없음 — 어떤 모델이든 즉시 시도\n", options: { fontFace: F.med, fontSize: 11, color: C.ok, breakLine: true } },
    { text: "+  tp·양자화·shape를 기동마다 자유 변경\n", options: { fontFace: F.med, fontSize: 11, color: C.ok, breakLine: true } },
    { text: "−  대신 매 기동마다 컴파일/그래프 캡처 비용 발생", options: { fontFace: F.med, fontSize: 11, color: C.err } },
  ], { x: x2 + 0.28, y: y + 3.12, w: cw - 0.5, h: 1.1, margin: 0, valign: "middle", lineSpacingMultiple: 1.45 });
})();

/* ───────────── 5. 서버 실행 (serve) ───────────── */
(() => {
  const s = pptx.addSlide();
  frame(s, "3. 서버 실행", 5, "출처: bench-blog/run_rngd.sh:41 · bench-gpu/runners/server.py:59 · configs/models.yaml · cli/serve.py");
  title(s, "서버 실행 — 양쪽 다 OpenAI 호환 HTTP 서버", "명령 모양은 닮았지만, furiosa는 “아티팩트 경로 + --devices”, vLLM은 “HF id + GPU” 입니다.");
  const cw = (CW - 0.4) / 2, y = 2.5;
  engineHead(s, M, y, cw, "furiosa-llm serve", "입력 = 빌드된 아티팩트", C.blue, C.blueLt);
  codeCard(s, M, y + 0.62, cw, 2.55, "실제 사용 예 (run_rngd.sh)", [
    { t: "furiosa-llm serve ./artifacts/qwen3-32b-fp8-tp8 \\", c: C.codeGrn },
    { t: "  --devices npu:0 \\        # 어느 NPU 카드", c: C.codeAc },
    { t: "  --host 0.0.0.0 --port 8000 \\", c: C.codeTx },
    { t: "  --max-concurrency 32 \\", c: C.codeTx },
    { t: "  --reasoning-parser qwen3 \\", c: C.codeTx },
    { t: "  --enable-prefix-caching", c: C.codeTx },
  ], 10.5);
  s.addText("입력이 “모델 이름”이 아니라 “빌드 산출물 폴더” · NPU 카드는 --devices npu:N 으로 지정", {
    x: M, y: y + 3.3, w: cw, h: 0.6, margin: 0, fontFace: F.med, fontSize: 10.5, color: C.blue, lineSpacingMultiple: 1.3 });

  const x2 = M + cw + 0.4;
  engineHead(s, x2, y, cw, "vllm serve", "입력 = HF 모델 id", C.vio, C.vioLt);
  codeCard(s, x2, y + 0.62, cw, 2.55, "실제 사용 예 (runners/server.py + models.yaml)", [
    { t: "CUDA_VISIBLE_DEVICES=2 \\   # 어느 GPU", c: C.codeAc },
    { t: "vllm serve Qwen/Qwen3-32B \\", c: C.codeGrn },
    { t: "  --host 0.0.0.0 --port 8000 \\", c: C.codeTx },
    { t: "  --max-num-seqs 32 \\", c: C.codeTx },
    { t: "  --tensor-parallel-size 2 \\  # 기동 시 결정", c: C.codeTx },
    { t: "  --enable-prefix-caching", c: C.codeTx },
  ], 10.5);
  s.addText("입력이 “HF 모델 id”(없으면 자동 다운로드) · GPU는 CUDA_VISIBLE_DEVICES 로 지정", {
    x: x2, y: y + 3.3, w: cw, h: 0.6, margin: 0, fontFace: F.med, fontSize: 10.5, color: C.vio, lineSpacingMultiple: 1.3 });

  card(s, M, 6.35, CW, 0.55, { fill: C.blueBg, line: null });
  s.addText([{ text: "공통: ", options: { fontFace: F.bold, fontSize: 11, color: C.ink, breakLine: false } },
    { text: "둘 다 기동 후 ", options: { fontFace: F.reg, fontSize: 11, color: C.ink2, breakLine: false } },
    { text: "GET /v1/models", options: { fontFace: F.semi, fontSize: 11, color: C.blue, breakLine: false } },
    { text: " 로 헬스 체크 → 같은 OpenAI HTTP 엔드포인트(/v1/chat/completions 등) 제공", options: { fontFace: F.reg, fontSize: 11, color: C.ink2 } }],
    { x: M + 0.25, y: 6.35, w: CW - 0.5, h: 0.55, margin: 0, valign: "middle" });
})();

/* ───────────── 6. 오프라인 Python API ───────────── */
(() => {
  const s = pptx.addSlide();
  frame(s, "4. 오프라인 Python API", 6, "출처: furiosa_llm api.py:90/115/419 (LLM, generate) · vllm/entrypoints/llm.py (LLM, generate)");
  title(s, "코드로 직접 호출 — from … import LLM, SamplingParams", "클래스 이름·메서드까지 닮았습니다. 차이는 LLM()에 “무엇을 넘기느냐”입니다.");
  const cw = (CW - 0.4) / 2, y = 2.55;
  engineHead(s, M, y, cw, "furiosa-llm", "LLM(아티팩트 경로)", C.blue, C.blueLt);
  codeCard(s, M, y + 0.62, cw, 3.4, null, [
    { t: "from furiosa_llm import LLM, SamplingParams", c: C.codeGrn },
    { t: "", c: C.codeTx },
    { t: "# 인자 = 빌드된 아티팩트 경로", c: C.codeMut },
    { t: "llm = LLM(\"./artifacts/qwen3-32b-fp8-tp8\",", c: C.codeTx },
    { t: "          devices=\"npu:0\")", c: C.codeTx },
    { t: "sp = SamplingParams(temperature=0.7,", c: C.codeTx },
    { t: "                    max_tokens=256)", c: C.codeTx },
    { t: "out = llm.generate(\"프롬프트\", sp)", c: C.codeAc },
    { t: "# (HF id로 쓰려면 fxb= 로 v3 엔진 경로)", c: C.codeMut },
  ], 10.5);

  const x2 = M + cw + 0.4;
  engineHead(s, x2, y, cw, "vLLM", "LLM(model=HF id)", C.vio, C.vioLt);
  codeCard(s, x2, y + 0.62, cw, 3.4, null, [
    { t: "from vllm import LLM, SamplingParams", c: C.codeGrn },
    { t: "", c: C.codeTx },
    { t: "# 인자 = HF 모델 id (런타임 로드)", c: C.codeMut },
    { t: "llm = LLM(model=\"Qwen/Qwen3-32B\",", c: C.codeTx },
    { t: "          tensor_parallel_size=2)", c: C.codeTx },
    { t: "sp = SamplingParams(temperature=0.7,", c: C.codeTx },
    { t: "                    max_tokens=256)", c: C.codeTx },
    { t: "out = llm.generate([\"프롬프트\"], sp)", c: C.codeAc },
    { t: "# tp·양자화 등을 생성자 인자로 바로 지정", c: C.codeMut },
  ], 10.5);
})();

/* ───────────── 7. 클라이언트 호출 ───────────── */
(() => {
  const s = pptx.addSlide();
  frame(s, "5. 클라이언트 호출", 7, "출처: server/protocol.py:1-2 · chat_utils.py:2-3 (“Adapted from vLLM”) · sampling 필드 드롭은 §본문 검증");
  title(s, "클라이언트는 동일 — 단, furiosa는 일부 필드를 조용히 버림", "furiosa 서버 코드 상단에 “Adapted from vLLM” 이 명시돼 있을 만큼 API 표면이 같습니다.");
  const y = 2.5;
  codeCard(s, M, y, CW, 1.85, "동일한 OpenAI 클라이언트 — 서버만 바꾸면 됨 (base_url)", [
    { t: "from openai import OpenAI", c: C.codeGrn },
    { t: "client = OpenAI(base_url=\"http://HOST:8000/v1\", api_key=\"-\")   # furiosa·vLLM 동일", c: C.codeTx },
    { t: "client.chat.completions.create(model=MODEL, messages=[...], temperature=0.7, stream=True)", c: C.codeAc },
  ], 11);
  const y2 = 4.6, cw = (CW - 0.4) / 2;
  card(s, M, y2, cw, 2.0, { fill: C.okBg, line: null, shadow: shStd() });
  accent(s, M, y2 + 0.2, 1.6, C.ok);
  s.addText("양쪽 모두 그대로 동작", { x: M + 0.28, y: y2 + 0.18, w: cw - 0.5, h: 0.35, margin: 0, fontFace: F.bold, fontSize: 14, color: C.ok });
  s.addText("model · messages/prompt · temperature · top_p · top_k · max_tokens · stop_token_ids · stream · logprobs · 구조화 출력(JSON/grammar)",
    { x: M + 0.28, y: y2 + 0.62, w: cw - 0.55, h: 1.3, margin: 0, fontFace: F.reg, fontSize: 11.5, color: C.ink2, lineSpacingMultiple: 1.4 });

  card(s, M + cw + 0.4, y2, cw, 2.0, { fill: C.errBg, line: null, shadow: shStd() });
  accent(s, M + cw + 0.4, y2 + 0.2, 1.6, C.err);
  s.addText("furiosa가 받지만 “무시”하는 필드", { x: M + cw + 0.68, y: y2 + 0.18, w: cw - 0.5, h: 0.35, margin: 0, fontFace: F.bold, fontSize: 14, color: C.err });
  s.addText([
    { text: "presence_penalty · frequency_penalty · seed · logit_bias · stop(문자열)\n", options: { fontFace: F.semi, fontSize: 11.5, color: C.err, breakLine: true } },
    { text: "→ 요청은 200 OK지만 효과 없음(to_sampling_params에서 드롭). n>1·투기적 디코딩은 아예 미지원.", options: { fontFace: F.reg, fontSize: 11, color: C.ink2 } },
  ], { x: M + cw + 0.68, y: y2 + 0.62, w: cw - 0.55, h: 1.3, margin: 0, lineSpacingMultiple: 1.4 });
})();

/* ───────────── 8. 튜닝 노브 매핑 ───────────── */
(() => {
  const s = pptx.addSlide();
  frame(s, "6. 튜닝 노브", 8, "출처: configs/models.yaml(실측 매핑) · cli/serve.py · EngineArgs(llm_engine.py:54-211) · vllm EngineArgs");
  title(s, "튜닝 노브 대응표 — 이름까지 일부러 맞춰 놓음", "벤치 비교용으로 정렬해 둔 매핑(models.yaml)을 그대로 옮긴 표입니다.");
  const rows = [
    ["동시 처리 시퀀스 수", "--max-concurrency / --max-batch-size", "--max-num-seqs", "개념 동일"],
    ["배치 토큰 예산", "--max-num-batched-tokens", "--max-num-batched-tokens", "키 이름 동일"],
    ["최대 컨텍스트 길이", "--max-model-len (버킷 상한)", "--max-model-len", "furiosa는 버킷 범위 내"],
    ["프리픽스 캐싱", "--enable-prefix-caching", "--enable-prefix-caching", "둘 다 기본 ON"],
    ["텐서 병렬(TP)", "빌드에 고정 (serve에선 무시)", "--tensor-parallel-size (기동 시)", "★ 가장 큰 차이"],
    ["메모리 점유율", "(등가 없음) · --spare-blocks-ratio", "--gpu-memory-utilization", "NPU엔 직접 등가 없음"],
    ["NPU 전용", "--devices npu:N · --npu-queue-limit · 버킷", "—", "하드웨어 종속"],
  ];
  const y = 2.45, rh = 0.5, x0 = M;
  const cols = [3.2, 4.0, 3.05, 2.08];
  const hd = ["항목", "furiosa-llm", "vLLM", "비고"];
  let cx = x0;
  hd.forEach((t, i) => {
    s.addShape(pptx.ShapeType.rect, { x: cx, y, w: cols[i], h: 0.5, fill: { color: i === 1 ? C.blue : i === 2 ? C.vio : C.dark }, line: { color: C.white, width: 1 } });
    s.addText(t, { x: cx, y, w: cols[i], h: 0.5, margin: 0, align: "center", valign: "middle", fontFace: F.bold, fontSize: 12, color: C.white });
    cx += cols[i];
  });
  rows.forEach((r, ri) => {
    const ry = y + 0.5 + ri * rh; cx = x0;
    const star = r[3].includes("★");
    r.forEach((t, ci) => {
      s.addShape(pptx.ShapeType.rect, { x: cx, y: ry, w: cols[ci], h: rh, fill: { color: star ? C.warnBg : ri % 2 ? C.bg2 : C.white }, line: { color: C.border2, width: 0.5 } });
      const isCode = ci === 1 || ci === 2;
      s.addText(t, { x: cx + 0.08, y: ry, w: cols[ci] - 0.16, h: rh, margin: 0, align: ci === 0 ? "left" : ci === 3 ? "center" : "left", valign: "middle",
        fontFace: ci === 0 ? F.semi : isCode ? F.reg : F.med, fontSize: isCode ? 9.5 : 10, color: star && ci === 3 ? C.warn : ci === 0 ? C.ink : C.ink2, lineSpacingMultiple: 1.1 });
      cx += cols[ci];
    });
  });
  s.addText("※ tp는 furiosa에선 빌드에 박혀 serve 플래그가 무시됨(다음 장). pp·dp는 serve에서 변경 가능.", {
    x: M, y: y + 0.5 + rows.length * rh + 0.12, w: CW, h: 0.3, margin: 0, fontFace: F.med, fontSize: 10.5, color: C.blue });
})();

/* ───────────── 9. CLI 플래그 지형 ───────────── */
(() => {
  const s = pptx.addSlide();
  frame(s, "7. CLI 플래그", 9, "출처: furiosa-llm cli/serve.py (전체 플래그 추출) · vLLM EngineArgs/cli_args");
  title(s, "furiosa serve = vLLM 플래그 미러링 + NPU 전용 추가", "furiosa는 vLLM 사용자가 그대로 옮겨올 수 있게 같은 플래그를 많이 채택했습니다.");
  const y = 2.5, cw = (CW - 0.4) / 2;
  // 공통(미러)
  card(s, M, y, cw, 4.0, { shadow: shStd() });
  s.addShape(pptx.ShapeType.roundRect, { x: M, y, w: cw, h: 0.5, rectRadius: 0.1, fill: { color: C.ink }, line: { type: "none" } });
  s.addText("양쪽 공통 (furiosa가 vLLM서 채택)", { x: M + 0.2, y, w: cw - 0.4, h: 0.5, margin: 0, valign: "middle", fontFace: F.bold, fontSize: 13, color: C.white });
  const shared = ["--host / --port", "--max-model-len", "--max-num-batched-tokens", "--enable-prefix-caching",
    "--tensor-parallel-size *", "--pipeline-parallel-size", "--data-parallel-size", "--reasoning-parser",
    "--tool-call-parser", "--structured-outputs-backend", "--served-model-name", "--api-key / --chat-template"];
  shared.forEach((t, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    s.addText("✓", { x: M + 0.25 + col * (cw / 2 - 0.1), y: y + 0.7 + row * 0.53, w: 0.25, h: 0.4, margin: 0, fontFace: F.bold, fontSize: 11, color: C.ok });
    s.addText(t, { x: M + 0.5 + col * (cw / 2 - 0.1), y: y + 0.7 + row * 0.53, w: cw / 2 - 0.45, h: 0.4, margin: 0, fontFace: F.reg, fontSize: 10.5, color: C.ink2 });
  });
  s.addText("* --tensor-parallel-size 는 plumbing만 같고, prebuilt 아티팩트에선 무시됨", { x: M + 0.25, y: y + 3.62, w: cw - 0.4, h: 0.3, margin: 0, fontFace: F.med, fontSize: 9.5, color: C.warn });

  // NPU 전용
  const x2 = M + cw + 0.4;
  card(s, x2, y, cw, 4.0, { shadow: shStd(), fill: C.blueBg, line: null });
  s.addShape(pptx.ShapeType.roundRect, { x: x2, y, w: cw, h: 0.5, rectRadius: 0.1, fill: { color: C.blue }, line: { type: "none" } });
  s.addText("furiosa 전용 (vLLM엔 없음)", { x: x2 + 0.2, y, w: cw - 0.4, h: 0.5, margin: 0, valign: "middle", fontFace: F.bold, fontSize: 13, color: C.white });
  const only = [
    ["--devices npu:N", "어느 NPU 카드/PE에 올릴지"],
    ["--prefill-buckets / --decode-buckets", "정적 shape 버킷 지정"],
    ["--max-batch-size / --min-batch-size", "NPU 배치 경계"],
    ["--npu-queue-limit", "NPU 큐 깊이"],
    ["--spare-blocks-ratio", "KV 예약 블록 비율"],
    ["--enable-jit-compilation", "기동 시 재컴파일(opt-in)"],
    ["--scheduler-kind / -loop-type", "네이티브 스케줄러 선택"],
  ];
  only.forEach(([f, d], i) => {
    s.addText(f, { x: x2 + 0.28, y: y + 0.68 + i * 0.46, w: cw - 0.5, h: 0.4, margin: 0, fontFace: F.semi, fontSize: 10.5, color: C.blue });
    s.addText(d, { x: x2 + 0.28, y: y + 0.68 + i * 0.46, w: cw - 0.45, h: 0.4, margin: 0, align: "right", fontFace: F.reg, fontSize: 9.5, color: C.ink2 });
  });
})();

/* ───────────── 10. 운영 함정 ───────────── */
(() => {
  const s = pptx.addSlide();
  frame(s, "8. 운영 함정", 10, "출처: README_runcode.md:499 · server/models.py:38 · api.py:383 · artifact/builder.py · presets.py");
  title(s, "실전에서 부딪히는 차이 5가지", "주로 furiosa의 “정적·사전 컴파일” 성격에서 옵니다.");
  const items = [
    ["tp는 serve에서 못 바꾼다", C.err, "prebuilt 아티팩트는 tp가 바이너리에 박혀 있어 --tensor-parallel-size 가 무시됨(WARNING). 바꾸려면 그 값으로 build 재실행. (vLLM은 기동 플래그로 즉시 변경)"],
    ["프롬프트 길이 = 버킷 상한", C.warn, "정적 shape이라 최대 prefill 버킷을 넘는 프롬프트는 거부됨. (vLLM은 동적 shape이라 --max-model-len 까지 자유)"],
    ["gpu-memory-utilization 등가 없음", C.blue, "NPU는 메모리 자동 프로파일링 노브가 없음 — --spare-blocks-ratio 가 가장 근접. (vLLM은 0.9로 HBM 점유율 직접 조절)"],
    ["첫 준비는 무겁고, 기동은 가볍다", C.ok, "furiosa: build 한 번(수십 분) → serve는 로드만(빠름·결정적). vLLM: build 없음 → 매 기동마다 컴파일/캡처 비용."],
    ["아티팩트 이동성", C.vio, "furiosa 아티팩트는 특정 모델·tp·SDK 버전에 묶임. vLLM은 HF 체크포인트만 있으면 어느 GPU에서나 동일 기동."],
  ];
  const y = 2.45, h = 0.86;
  items.forEach(([nm, col, desc], i) => {
    const yy = y + i * (h + 0.04);
    card(s, M, yy, CW, h, { shadow: i === 0 ? shStd() : undefined });
    accent(s, M, yy + 0.16, h - 0.32, col);
    s.addText(`${i + 1}`, { x: M + 0.2, y: yy, w: 0.5, h, margin: 0, align: "center", valign: "middle", fontFace: F.black, fontSize: 22, color: col });
    s.addText(nm, { x: M + 0.8, y: yy + 0.13, w: 3.7, h: h - 0.26, margin: 0, valign: "middle", fontFace: F.bold, fontSize: 13.5, color: C.ink, lineSpacingMultiple: 1.05 });
    s.addText(desc, { x: M + 4.65, y: yy + 0.1, w: CW - 4.85, h: h - 0.2, margin: 0, valign: "middle", fontFace: F.reg, fontSize: 10.8, color: C.ink2, lineSpacingMultiple: 1.28 });
  });
})();

/* ───────────── 11. 정리 / 의사결정 ───────────── */
(() => {
  const s = pptx.addSlide();
  frame(s, "정리", 11, "");
  title(s, "정리 — 무엇을 언제 쓰나", "");
  const cw = (CW - 0.4) / 2, y = 2.4, h = 3.0;
  card(s, M, y, cw, h, { shadow: shStd() });
  s.addShape(pptx.ShapeType.roundRect, { x: M, y, w: cw, h: 0.55, rectRadius: 0.1, fill: { color: C.blue }, line: { type: "none" } });
  s.addText("furiosa-llm 이 맞는 경우", { x: M + 0.2, y, w: cw - 0.4, h: 0.55, margin: 0, valign: "middle", fontFace: F.bold, fontSize: 14, color: C.white });
  [
    "Furiosa RNGD NPU에 배포가 전제",
    "모델·구성이 고정 — 한 번 빌드해 오래 서빙",
    "빠르고 결정적인 기동, 사전 최적화가 중요",
    "AOT로 굳힌 정적 파이프라인이 잘 맞는 워크로드",
  ].forEach((t, i) => { s.addText("•", { x: M + 0.3, y: y + 0.78 + i * 0.52, w: 0.2, h: 0.4, margin: 0, fontFace: F.bold, fontSize: 13, color: C.blue }); s.addText(t, { x: M + 0.54, y: y + 0.72 + i * 0.52, w: cw - 0.8, h: 0.5, margin: 0, fontFace: F.reg, fontSize: 11.5, color: C.ink2, lineSpacingMultiple: 1.25 }); });

  const x2 = M + cw + 0.4;
  card(s, x2, y, cw, h, { shadow: shStd() });
  s.addShape(pptx.ShapeType.roundRect, { x: x2, y, w: cw, h: 0.55, rectRadius: 0.1, fill: { color: C.vio }, line: { type: "none" } });
  s.addText("vLLM 이 맞는 경우", { x: x2 + 0.2, y, w: cw - 0.4, h: 0.55, margin: 0, valign: "middle", fontFace: F.bold, fontSize: 14, color: C.white });
  [
    "GPU 환경 · 다양한 모델을 자주 바꿔 실험",
    "tp·양자화·shape를 기동마다 유연하게",
    "투기적 디코딩·전체 샘플링 기능이 필요",
    "엔진 내부를 읽고·고치고·검증해야 함",
  ].forEach((t, i) => { s.addText("•", { x: x2 + 0.3, y: y + 0.78 + i * 0.52, w: 0.2, h: 0.4, margin: 0, fontFace: F.bold, fontSize: 13, color: C.vio }); s.addText(t, { x: x2 + 0.54, y: y + 0.72 + i * 0.52, w: cw - 0.8, h: 0.5, margin: 0, fontFace: F.reg, fontSize: 11.5, color: C.ink2, lineSpacingMultiple: 1.25 }); });

  card(s, M, 5.75, CW, 0.95, { fill: C.dark, line: null, shadow: shStd() });
  s.addText([
    { text: "한 줄 요약  ", options: { fontFace: F.bold, fontSize: 13, color: C.codeAc, breakLine: false } },
    { text: "furiosa-llm = 미리 컴파일해 얼린 아티팩트 + 네이티브 런타임(NPU)   ·   vLLM = 기동마다 로드·컴파일하는 열린 Python 런타임(GPU)",
      options: { fontFace: F.med, fontSize: 12, color: C.codeTx } },
  ], { x: M + 0.3, y: 5.75, w: CW - 0.6, h: 0.95, margin: 0, valign: "middle", lineSpacingMultiple: 1.3 });
})();

pptx.writeFile({ fileName: "RNGD_vLLM_vs_FCLM.pptx" }).then((f) => console.log("saved:", f));
