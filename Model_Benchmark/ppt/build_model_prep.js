/* Furiosa RNGD 모델 준비 — SDK 내부 구조와 빌드 에러 분석 (16:9)
 * 기존 build.js (Brandlogy/Paperlogy) 스타일 준수.
 * 모든 내용은 SDK 소스(~/furiosa/lib/python3.12/site-packages/furiosa_llm/)를
 * 직접 읽어 확인한 사실 + 실제 빌드 로그(journalctl·tmux) 기반.
 * 출처는 각 슬라이드 하단에 파일:라인으로 표기. */
const pptx = new (require("pptxgenjs"))();
pptx.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pptx.layout = "W";
pptx.author = "RNGD Model Prep";
const TOTAL = 13;

const F = {
  black: "Paperlogy 9 Black", xbold: "Paperlogy 8 ExtraBold",
  bold: "Paperlogy 7 Bold", semi: "Paperlogy 6 SemiBold",
  med: "Paperlogy 5 Medium", reg: "Paperlogy 4 Regular",
};
const C = {
  ink: "222222", ink2: "45515e", mut: "8e8e93",
  blue: "1456f0", blue2: "3b82f6", blue3: "60a5fa", blueLt: "bfdbfe",
  white: "ffffff", border: "f2f3f5", border2: "e5e7eb", bg2: "f0f0f0",
  dark: "181e25", codeTx: "e5e9ef", codeMut: "8ea0b5", codeAc: "5fc6ff", codeRed: "ff8087",
  ok: "16a34a", okBg: "e8ffea", err: "dc2626", errBg: "fde8e8", warn: "d97706", warnBg: "fef3c7",
};
const shStd = () => ({ type: "outer", color: "000000", opacity: 0.08, blur: 6, offset: 2, angle: 90 });
const M = 0.5, CW = 13.333 - 2 * M;

function frame(s, chapter, page, source) {
  s.background = { color: C.white };
  s.addText(chapter.toUpperCase(), {
    x: M, y: 0.4, w: 9, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 12, color: C.mut, charSpacing: 0.8,
  });
  s.addText(`${page} / ${TOTAL}`, {
    x: 13.333 - M - 3, y: 7.08, w: 3, h: 0.25, margin: 0, fontFace: F.med, fontSize: 10, color: C.mut, align: "right",
  });
  if (source) s.addText(source, {
    x: M, y: 7.08, w: 9, h: 0.25, margin: 0, fontFace: F.reg, fontSize: 9, color: C.mut,
  });
}
function title(s, head, sub) {
  s.addText(head, {
    x: M, y: 0.92, w: CW, h: 0.7, margin: 0,
    fontFace: F.bold, fontSize: 30, color: C.ink, charSpacing: -0.6, lineSpacingMultiple: 1.16,
  });
  if (sub) s.addText(sub, {
    x: M, y: 1.66, w: CW, h: 0.4, margin: 0,
    fontFace: F.med, fontSize: 14.5, color: C.ink2, lineSpacingMultiple: 1.4,
  });
}
function card(s, x, y, w, h, opt = {}) {
  s.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h, rectRadius: opt.r || 0.12,
    fill: { color: opt.fill || C.white },
    line: opt.line === null ? { type: "none" } : { color: opt.line || C.border2, width: 1 },
    shadow: opt.shadow,
  });
}
function tag(s, x, y, text, fill, txtColor, fs) {
  const w = 0.3 + text.length * 0.1;
  s.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h: 0.3, rectRadius: 0.15, fill: { color: fill }, line: { type: "none" },
  });
  s.addText(text, {
    x, y, w, h: 0.3, margin: 0, align: "center", valign: "middle",
    fontFace: F.semi, fontSize: fs || 10, color: txtColor || C.white,
  });
  return w;
}
function accent(s, x, y, h, color) {
  s.addShape(pptx.ShapeType.roundRect, { x, y, w: 0.07, h, rectRadius: 0.03, fill: { color }, line: { type: "none" } });
}
function codeCard(s, x, y, w, h, label, lines, fs) {
  s.addShape(pptx.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.1, fill: { color: C.dark }, line: { type: "none" }, shadow: shStd() });
  let ty = y + 0.18;
  if (label) {
    s.addText(label, { x: x + 0.26, y: ty, w: w - 0.5, h: 0.24, margin: 0, fontFace: F.semi, fontSize: 10, color: C.codeMut });
    ty += 0.34;
  }
  s.addText(lines.map((ln) => ({
    text: ln.t, options: { fontFace: F.reg, fontSize: fs || 10.5, color: ln.c || C.codeTx, breakLine: true },
  })), { x: x + 0.26, y: ty, w: w - 0.5, h: y + h - ty - 0.16, margin: 0, lineSpacingMultiple: 1.32, valign: "top" });
}
function bullets(s, x, y, w, h, items, fs) {
  s.addText(items.map((it) => {
    const o = typeof it === "object" ? it : { t: it };
    return {
      text: o.t,
      options: {
        bullet: { code: "2022", indent: 16 }, breakLine: true,
        fontFace: o.b ? F.semi : F.reg, color: o.c || C.ink2,
        fontSize: o.fs || fs || 14,
      },
    };
  }), { x, y, w, h, margin: 0, lineSpacingMultiple: 1.45, paraSpaceAfter: 9, valign: "top" });
}
function infoTable(s, x, y, w, header, rows, opt = {}) {
  const fs = opt.fs || 11;
  const head = header.map((h) => ({
    text: h, options: { fontFace: F.semi, fontSize: fs, color: C.white, fill: { color: opt.headFill || C.blue }, align: "center", valign: "middle" },
  }));
  const body = rows.map((r) => r.map((c, i) => {
    const cell = typeof c === "object" ? c : { t: String(c) };
    return {
      text: cell.t,
      options: {
        fontFace: F.reg, fontSize: fs, color: cell.c || C.ink2,
        fill: { color: cell.fill || C.white },
        align: i === 0 ? "left" : "center", valign: "middle",
      },
    };
  }));
  s.addTable([head, ...body], {
    x, y, w, colW: opt.colW, border: { type: "solid", color: C.border2, pt: 0.5 },
    rowH: opt.rowH || 0.38, margin: [2, 4, 2, 4], valign: "middle",
  });
}

/* ===================== 1 — Cover ===================== */
(() => {
  const s = pptx.addSlide();
  s.background = { color: C.white };
  s.addText("FURIOSA RNGD · furiosa-llm 2026.2.0 · 2026.05", {
    x: M, y: 0.6, w: 11, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 12, color: C.mut, charSpacing: 1,
  });
  s.addText("Furiosa RNGD 모델 준비\nSDK 내부 구조와 빌드 에러 분석", {
    x: M, y: 1.5, w: 12.4, h: 1.9, margin: 0, fontFace: F.bold, fontSize: 40, color: C.ink,
    charSpacing: -0.8, lineSpacingMultiple: 1.14,
  });
  s.addText("furiosa-llm build 의 내부 동작(검증·preset·버킷·컴파일)을 SDK 소스로 직접 확인하고,\n모델을 NPU 아티팩트로 준비하며 실제로 마주친 빌드 에러를 정리합니다.", {
    x: M, y: 3.5, w: 12, h: 0.9, margin: 0, fontFace: F.med, fontSize: 15, color: C.ink2, lineSpacingMultiple: 1.45,
  });
  accent(s, M, 5.2, 1.2, C.blue);
  s.addText([
    { text: "다룰 내용\n", options: { fontFace: F.semi, fontSize: 13, color: C.ink, breakLine: true } },
    { text: "빌드 3단계 흐름 · tp/pp 규칙 · preset 매칭 원리와 함정 · 버킷 시스템 · 빌드 2단계(트레이싱/컴파일) · 빌드 에러 4종 · 성공·실패 매트릭스", options: { fontFace: F.reg, fontSize: 12.5, color: C.ink2 } },
  ], { x: M + 0.25, y: 5.2, w: 11.8, h: 1.2, margin: 0, lineSpacingMultiple: 1.4, valign: "top" });
  s.addText("출처: SDK 소스 직접 분석 (artifact/builder.py · validator.py · resolver.py · presets.py · parallelize/block_slicer.py) + 실측 빌드 로그", {
    x: M, y: 6.95, w: 12.4, h: 0.3, margin: 0, fontFace: F.reg, fontSize: 9, color: C.mut,
  });
})();

/* ===================== 2 — 큰 그림 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "SDK 구조 · 전체 흐름", 2, "출처: artifact/builder.py:116(__init__) · :315(build)");
  title(s, "furiosa-llm build 한 줄이 하는 일", "명령 하나 안에서 두 성격의 작업이 순서대로 — 입구 검증/채움(가벼움)과 실제 빌드(무거움)");
  const y = 2.5, h = 3.4, w = (CW - 2 * 0.3) / 3;
  const cols = [
    ["1. 준비", C.blue, "__init__ 단계 (수 초)", [
      "① Validate — 설정·HF config·버킷이 규칙에 맞는지 검사",
      "② Resolve — 안 준 값(max_model_len·버킷)을 config·preset에서 채움",
    ]],
    ["2. 빌드", C.warn, "build 단계 (수 시간, 메모리 위험)", [
      "③ Tracing — 버킷마다 FX 그래프 생성 (weight+IR 메모리 ★)",
      "④ Compile — supertask마다 NPU 컴파일러로 EDF 바이너리",
    ]],
    ["3. 저장", C.ok, "결과물", [
      "artifact.json (메타·parallel_config)",
      "binary_bundle.zip (EDF) · config · tokenizer",
    ]],
  ];
  cols.forEach(([lab, col, head, items], i) => {
    const x = M + i * (w + 0.3);
    card(s, x, y, w, h, { shadow: shStd() });
    accent(s, x, y + 0.25, 0.55, col);
    s.addText(lab, { x: x + 0.28, y: y + 0.24, w: w - 0.5, h: 0.4, margin: 0, fontFace: F.bold, fontSize: 17, color: C.ink });
    s.addText(head, { x: x + 0.28, y: y + 0.74, w: w - 0.5, h: 0.5, margin: 0, fontFace: F.med, fontSize: 11, color: col });
    bullets(s, x + 0.28, y + 1.35, w - 0.55, h - 1.5, items, 11);
  });
  s.addText("OOM·실패는 거의 전부 ③ Tracing 단계 — ④ Compile(\"Compilation Progress\") 로그가 뜨면 메모리 위험 구간은 통과한 것", {
    x: M, y: 6.2, w: CW, h: 0.5, margin: 0, fontFace: F.med, fontSize: 12, color: C.ink2, align: "center",
  });
})();

/* ===================== 3 — Validate (tp/pp) ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "SDK 구조 · ① 검증", 3, "출처: artifact/validator.py:234-267 · device.py:6 (NUM_PES_PER_NPU=8)");
  title(s, "tp / pp 규칙 — 입구에서 막는다", "validate_parallel_config 가 비싼 빌드 전에 잘못된 병렬 설정을 즉시 거부");
  const y = 2.45;
  card(s, M, y, 6.0, 3.5, { shadow: shStd() });
  s.addText("검증 규칙", { x: M + 0.3, y: y + 0.22, w: 5, h: 0.35, margin: 0, fontFace: F.bold, fontSize: 15, color: C.blue });
  bullets(s, M + 0.3, y + 0.75, 5.4, 2.6, [
    { t: "tensor parallel(tp) ∈ {4, 8, 32} — 그 외 값은 거부", b: true },
    "pipeline parallel(pp) ≥ 1",
    { t: "필요 디바이스 = ceil(tp/8) × pp ≤ 8", b: true },
    "RNGD 1장 = 8 PE (NUM_PES_PER_NPU)",
    "tp=4 → 1장 안 PE 4개만 / tp=8 → 1장 풀",
  ], 12.5);
  infoTable(s, 6.8, y + 0.15, CW - 6.3, ["tp", "pp", "카드 수", "비고"], [
    ["4", "1~8", "1~8", "PE 4개만"],
    ["8", "1~8", "1~8", "1장 풀 PE"],
    [{ t: "32", c: C.blue }, "1~2", "4~8", "4장에 분산"],
  ], { fs: 11.5, rowH: 0.55, colW: [0.9, 1.1, 1.3, 2.43] });
  s.addText("prebuilt 32B·70B 아티팩트가 tp=32 — 그래서 RNGD 4장(32 PE)이 있어야 서빙됨 (2장에선 \"Required PEs: 32\" 거부)", {
    x: 6.8, y: y + 2.5, w: CW - 6.3, h: 0.9, margin: 0, fontFace: F.reg, fontSize: 11.5, color: C.ink2, lineSpacingMultiple: 1.4,
  });
})();

/* ===================== 4 — Resolve + find_preset ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "SDK 구조 · ② 채움", 4, "출처: artifact/resolver.py:34-122 · presets.py:372 find_preset");
  title(s, "버킷 자동 선택 — find_preset", "버킷을 직접 안 주면 모델에 맞는 preset을 SDK가 골라줌");
  card(s, M, 2.45, CW, 1.7, { shadow: shStd(), fill: C.bg2, line: null });
  s.addText("2단계 매칭", { x: M + 0.3, y: 2.6, w: 4, h: 0.35, margin: 0, fontFace: F.bold, fontSize: 14, color: C.blue });
  s.addText([
    { text: "1단계  ", options: { fontFace: F.semi, fontSize: 13, color: C.blue } },
    { text: "model_type 으로 후보 필터 (llama·qwen2·qwen3·qwen3_moe·exaone4 …)\n", options: { fontFace: F.reg, fontSize: 13, color: C.ink2, breakLine: true } },
    { text: "2단계  ", options: { fontFace: F.semi, fontSize: 13, color: C.blue } },
    { text: "(hidden_size, intermediate_size)로 layer당 파라미터 수 계산 → log-distance가 가장 가까운 preset 선택", options: { fontFace: F.reg, fontSize: 13, color: C.ink2 } },
  ], { x: M + 0.3, y: 3.05, w: CW - 0.6, h: 0.95, margin: 0, lineSpacingMultiple: 1.4, valign: "top" });
  bullets(s, M + 0.1, 4.45, CW - 0.2, 2.2, [
    { t: "크기가 정확히 같지 않아도 됨 — 가장 가까운 항목이 잡힘 (같은 architecture의 fine-tune 모델은 대체로 자동 매칭)", },
    { t: "예: qwen3_moe(2048,6144)=Qwen3-Coder-30B vs (6144,8192)=Qwen3-Coder-480B → per-layer 0.042B vs 0.252B, log-distance로 명확히 구분", },
    { t: "preset이 아예 없는 model_type(mistral·phi3·gpt_oss 등)은 -pb/-db로 버킷을 직접 줘야 빌드 가능", c: C.ink2 },
  ], 12.5);
})();

/* ===================== 5 — preset 동점 함정 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "SDK 구조 · preset 함정", 5, "출처: presets.py:393 min(candidates) · PRESET_REFS 등록 순서 (실측 확인)");
  title(s, "함정 — (h,i)가 같으면 \"등록 순서\"가 결정한다", "log-distance가 동점일 때 min()은 먼저 등록된 항목을 고름");
  codeCard(s, M, 2.4, 6.3, 3.0, "PRESET_REFS — qwen3_moe (2048, 6144) 3개 중복", [
    { t: "# 먼저 등록된 일반 preset (decode 8192)", c: C.codeMut },
    { t: "QWEN_3_30B_A3B_PRESET        ← 먼저 → 선택됨", c: C.codeRed },
    { t: "QWEN_3_CODER_30B_A3B_PRESET  (decode 256K)", c: C.codeTx },
    { t: "QWEN_3_CODER_30B_A3B_PRESET  (중복)", c: C.codeMut },
    { t: "", c: C.codeTx },
    { t: "→ Qwen3-Coder가 8192 preset에 매칭됨", c: C.codeAc },
  ], 11);
  card(s, 7.0, 2.4, CW - 6.5, 3.0, { shadow: shStd() });
  accent(s, 7.0, 2.65, 0.5, C.err);
  s.addText("증상", { x: 7.3, y: 2.62, w: 4, h: 0.35, margin: 0, fontFace: F.bold, fontSize: 14, color: C.err });
  bullets(s, 7.3, 3.15, CW - 7.0, 2.2, [
    "Qwen3-Coder-30B(256K 모델)이 컨텍스트 8192로만 빌드됨",
    "두 preset의 (h,i)가 완전히 동일 → log-distance=0 동점",
    { t: "해결: 일반 entry 제거 → Coder preset(256K)로 통일", c: C.ok, b: true },
  ], 12);
  s.addText("교훈 — preset을 새로 추가할 때 기존과 (hidden, intermediate)가 겹치면, 의도한 preset을 PRESET_REFS에서 먼저 등록해야 한다", {
    x: M, y: 5.75, w: CW, h: 0.6, margin: 0, fontFace: F.med, fontSize: 12, color: C.ink2, align: "center", lineSpacingMultiple: 1.35,
  });
})();

/* ===================== 6 — 버킷 4종 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "SDK 구조 · 버킷", 6, "출처: metadata/config_types.py:141 AttentionBucket · resolver.py");
  title(s, "버킷 — AOT 컴파일이라 모양을 미리 굽는다", "(batch, context) 조합 하나하나를 그래프로 컴파일해 두고, 요청을 가장 가까운 버킷에 라우팅");
  const y = 2.5, h = 3.3, w = (CW - 3 * 0.25) / 4;
  const items = [
    ["prefill", C.blue, "kv_cache = 0", "프롬프트 전체를 한 번에 — 첫 토큰 직전. 보통 짧게(128~1024)"],
    ["decode", C.ok, "input_ids = 1", "한 토큰씩 생성. 생성 모델 필수 — 최대 컨텍스트를 여기서 정함"],
    ["append", C.warn, "1<input<attn", "prefix 캐시 일부 재사용 + 새 토큰 묶음 추가 (CLI 없음, API)"],
    ["tokenwise", C.mut, "정수 1개", "composable kernel용 시퀀스 길이 (CLI 없음, API)"],
  ];
  items.forEach(([nm, col, cond, desc], i) => {
    const x = M + i * (w + 0.25);
    card(s, x, y, w, h, { shadow: shStd() });
    accent(s, x, y + 0.25, 0.5, col);
    s.addText(nm, { x: x + 0.26, y: y + 0.24, w: w - 0.45, h: 0.4, margin: 0, fontFace: F.bold, fontSize: 15, color: C.ink });
    s.addText(cond, { x: x + 0.26, y: y + 0.78, w: w - 0.45, h: 0.3, margin: 0, fontFace: F.med, fontSize: 10.5, color: col });
    s.addText(desc, { x: x + 0.26, y: y + 1.2, w: w - 0.5, h: h - 1.4, margin: 0, fontFace: F.reg, fontSize: 11, color: C.ink2, lineSpacingMultiple: 1.4, valign: "top" });
  });
  s.addText("모든 요청 모양을 다 구울 필요 없음 — 대표 모양만 빌드. --max-model-len 으로 큰 버킷을 제외해 빌드 메모리를 줄일 수 있음", {
    x: M, y: 6.1, w: CW, h: 0.5, margin: 0, fontFace: F.med, fontSize: 12, color: C.ink2, align: "center",
  });
})();

/* ===================== 7 — build 2단계 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "SDK 구조 · 빌드 2단계", 7, "출처: parallelize/new_pipeline_builder.py · pipeline/builder/converter.py");
  title(s, "Tracing vs Compile — 성격이 정반대", "같은 build 명령 안의 두 단계가 메모리·시간 특성이 완전히 다름");
  const y = 2.45, w = (CW - 0.4) / 2, h = 3.0;
  card(s, M, y, w, h, { shadow: shStd() });
  accent(s, M, y + 0.25, 0.55, C.warn);
  s.addText("③ Tracing", { x: M + 0.3, y: y + 0.22, w: 4, h: 0.4, margin: 0, fontFace: F.bold, fontSize: 17, color: C.ink });
  s.addText("\"Model Tracing Progress: x/Y\"", { x: M + 0.3, y: y + 0.72, w: w - 0.5, h: 0.3, margin: 0, fontFace: F.med, fontSize: 11, color: C.warn });
  bullets(s, M + 0.3, y + 1.15, w - 0.55, h - 1.3, [
    "버킷마다 PyTorch FX로 그래프 생성",
    "전체 weight + 모든 버킷 IR을 한 메모리에 누적",
    { t: "★ OOM 위험 매우 높음 (32B는 100GB+ 도달)", c: C.err, b: true },
    "시간 비중 ~1/4",
  ], 12);
  card(s, M + w + 0.4, y, w, h, { shadow: shStd() });
  accent(s, M + w + 0.4, y + 0.25, 0.55, C.ok);
  s.addText("④ Compile", { x: M + w + 0.7, y: y + 0.22, w: 4, h: 0.4, margin: 0, fontFace: F.bold, fontSize: 17, color: C.ink });
  s.addText("\"Compilation Progress: x/Y\"", { x: M + w + 0.7, y: y + 0.72, w: w - 0.5, h: 0.3, margin: 0, fontFace: F.med, fontSize: 11, color: C.ok });
  bullets(s, M + w + 0.7, y + 1.15, w - 0.55, h - 1.3, [
    "supertask(보통 block 1개)마다 NPU 컴파일 → EDF",
    "현재 supertask만 메모리에 — 누적 거의 없음",
    { t: "메모리 안정 (32B 실측 ~20GB)", c: C.ok, b: true },
    "시간 비중 ~3/4 (오래 걸리지만 안전)",
  ], 12);
  s.addText("실측: 32B 빌드 OOM 4회 모두 Tracing 단계 — \"Compilation Progress\"가 한 번 뜨면 그 뒤로는 시간만 기다리면 됨", {
    x: M, y: 5.7, w: CW, h: 0.5, margin: 0, fontFace: F.med, fontSize: 12, color: C.ink2, align: "center",
  });
})();

/* ===================== 8 — 에러 ① embedding_table ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "빌드 에러 · ①", 8, "출처: 실측 빌드 로그 (Qwen3-32B · Qwen2.5-Coder-32B, BF16 tp=32)");
  title(s, "에러 ① — BF16 32B를 tp=32로 빌드", "Tracing은 통과하지만 Compile 진입 직후 컴파일러가 거부");
  codeCard(s, M, 2.45, CW, 1.5, "Compile 단계 에러", [
    { t: "RuntimeError: Compilation error: fail to compile:", c: C.codeRed },
    { t: "  Graph input#0 must have Broadcast or Fixed", c: C.codeRed },
    { t: "  DramShapeGuide (Name: embedding_table)", c: C.codeRed },
  ], 11);
  card(s, M, 4.15, CW, 2.0, { shadow: shStd() });
  bullets(s, M + 0.3, 4.35, CW - 0.6, 1.7, [
    { t: "32B dense 모델의 BF16 임베딩을 tp=32로 분할할 때 컴파일러가 DRAM shape guide를 만들지 못함 — SDK 한계", b: true },
    { t: "재현: Qwen3-32B(BF16,tp32) · Qwen2.5-Coder-32B(BF16,tp32) 둘 다 동일", },
    { t: "회피: 같은 모델의 FP8 변종 사용 — 양자화 메타가 임베딩에 함께 들어가 컴파일 가능 (FP8 빌드는 성공)", c: C.ok },
  ], 12.5);
})();

/* ===================== 9 — 에러 ② reshape mapping ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "빌드 에러 · ②", 9, "출처: 실측 빌드 로그 (Qwen3-Coder-30B-A3B-FP8, tp=32) + config.json");
  title(s, "에러 ② — FP8을 tp=32로 분할 (block 정렬 깨짐)", "FP8 양자화 블록(128×128)이 tp 분할 후 차원과 안 맞음");
  codeCard(s, M, 2.45, 6.3, 1.5, "Compile stage_2 에러", [
    { t: "RuntimeError: Compilation error:", c: C.codeRed },
    { t: "  impossible to apply", c: C.codeRed },
    { t: "  reshape mapping", c: C.codeRed },
  ], 11);
  card(s, 7.0, 2.45, CW - 6.5, 1.5, { shadow: shStd(), fill: C.bg2, line: null });
  s.addText([
    { text: "hidden 2048 ÷ tp 32 = 64\n", options: { fontFace: F.semi, fontSize: 14, color: C.err, breakLine: true } },
    { text: "< FP8 block_size 128 → 정렬 불가", options: { fontFace: F.med, fontSize: 13, color: C.ink2 } },
  ], { x: 7.3, y: 2.75, w: CW - 7.0, h: 1.0, margin: 0, lineSpacingMultiple: 1.4, valign: "middle" });
  infoTable(s, M, 4.25, CW, ["tp", "hidden ÷ tp", "block(128) 정렬", "결과"], [
    ["8", "256", { t: "≥ 128 OK", c: C.ok }, { t: "성공", c: C.ok }],
    ["16", "128", { t: "= 128 경계", c: C.warn }, "이론상 가능"],
    ["32", "64", { t: "< 128 깨짐", c: C.err }, { t: "실패", c: C.err }],
  ], { fs: 12, rowH: 0.5 });
  s.addText("hidden이 작은 MoE 모델(Qwen3-Coder-30B, hidden 2048)은 tp=32에서 FP8 블록 정렬이 깨짐 — tp=8로는 정상 빌드", {
    x: M, y: 6.15, w: CW, h: 0.5, margin: 0, fontFace: F.med, fontSize: 12, color: C.ink2, align: "center",
  });
})();

/* ===================== 10 — 에러 ③ OOM ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "빌드 에러 · ③", 10, "출처: journalctl (커널 OOM killer) + free/swapon 실측");
  title(s, "에러 ③ — OOM, swap이 있어도 못 막는다", "Tracing의 working set이 RAM을 넘으면 커널 OOM killer가 발동");
  codeCard(s, M, 2.4, CW, 1.2, "journalctl", [
    { t: "tmux-spawn-...scope: A process of this unit has been", c: C.codeRed },
    { t: "killed by the OOM killer.", c: C.codeRed },
  ], 11);
  const y = 3.85, w = (CW - 0.4) / 2;
  card(s, M, y, w, 2.2, { shadow: shStd() });
  s.addText("왜 swap이 안 통하나", { x: M + 0.3, y: y + 0.2, w: w - 0.5, h: 0.35, margin: 0, fontFace: F.bold, fontSize: 14, color: C.err });
  bullets(s, M + 0.3, y + 0.7, w - 0.55, 1.4, [
    "Tracing 메모리는 계속 쓰이는 active working set",
    "active 페이지는 swap으로 못 빠짐",
    { t: "실측: swap 207GB 중 1.5GB만 쓰고 OOM", b: true },
  ], 11.5);
  card(s, M + w + 0.4, y, w, 2.2, { shadow: shStd() });
  s.addText("진짜 해결책", { x: M + w + 0.7, y: y + 0.2, w: w - 0.5, h: 0.35, margin: 0, fontFace: F.bold, fontSize: 14, color: C.ok });
  bullets(s, M + w + 0.7, y + 0.7, w - 0.55, 1.4, [
    { t: "--max-model-len 축소 → working set 자체를 줄임", b: true, c: C.ok },
    "FP8로 weight 절반 (60GB→30GB)",
    "systemd-oomd 꺼도 커널 OOM은 별개 (못 끔)",
  ], 11.5);
})();

/* ===================== 11 — 에러 ④ 기타 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "빌드 에러 · ④", 11, "출처: 실측 로그 (EXAONE 빌드) · vocab_embedding.py:20 pad_vocab_size");
  title(s, "에러 ④ — 그 외 자주 막히는 것들", "컴파일 이전 단계에서 빌드를 멈추는 환경·모델 제약");
  const y = 2.5, h = 1.55, w = (CW - 0.3) / 2;
  const items = [
    ["디스크 풀", C.err, "No space left on device → 빌드가 import 단계에서 즉시 종료(임시 디렉토리 못 만듦). 앞 모델 다운로드가 디스크를 채운 게 원인"],
    ["vocab_size 패딩", C.warn, "SDK가 vocab을 64의 배수로 패딩(예 32016→32064)하는데 weight loader가 패딩을 못 받아 mismatch — CodeLlama 빌드 불가"],
    ["SSH 끊김", C.mut, "빌드는 수 시간 — 직접 셸이면 연결이 끊길 때 같이 죽음. tmux 세션 안에서 빌드해야 살아남음"],
    ["pp 미지원 architecture", C.blue, "-pp는 block_slicer.py에 등록된 4종(Llama·GPTJ·Bert·Roberta)만 가능. 그 외는 NotImplementedError → dp로 우회"],
  ];
  items.forEach(([nm, col, desc], i) => {
    const x = M + (i % 2) * (w + 0.3);
    const yy = y + Math.floor(i / 2) * (h + 0.25);
    card(s, x, yy, w, h, { shadow: shStd() });
    accent(s, x, yy + 0.22, 0.45, col);
    s.addText(nm, { x: x + 0.28, y: yy + 0.2, w: w - 0.5, h: 0.35, margin: 0, fontFace: F.bold, fontSize: 14.5, color: C.ink });
    s.addText(desc, { x: x + 0.28, y: yy + 0.62, w: w - 0.55, h: h - 0.75, margin: 0, fontFace: F.reg, fontSize: 11, color: C.ink2, lineSpacingMultiple: 1.36, valign: "top" });
  });
})();

/* ===================== 12 — 성공/실패 매트릭스 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "정리 · 매트릭스", 12, "출처: 본 캠페인 실측 빌드 결과");
  title(s, "빌드 성공 · 실패 매트릭스", "같은 모델도 dtype·tp 조합에 따라 결과가 갈림");
  infoTable(s, M, 2.5, CW, ["모델", "dtype", "tp", "결과", "원인 / 비고"], [
    ["Qwen2.5-Coder-1.5B/7B/14B", "BF16", "8", { t: "성공", c: C.ok }, "1장 적재 (max 32K)"],
    ["Qwen3-32B-FP8", "FP8", "8", { t: "성공", c: C.ok }, "1장 적재 (max 40960)"],
    ["Qwen3-Coder-30B-A3B-FP8", "FP8", "8", { t: "성공", c: C.ok }, "MoE, 1장 적재"],
    ["Qwen3-32B / Qwen2.5-Coder-32B", "BF16", "32", { t: "실패", c: C.err }, "embedding_table (에러①)"],
    ["Qwen3-Coder-30B-A3B-FP8", "FP8", "32", { t: "실패", c: C.err }, "reshape mapping (에러②)"],
    ["EXAONE-4.0-32B", "BF16", "32", { t: "실패", c: C.err }, "OOM + 디스크 풀 (에러③④)"],
    ["CodeLlama-70B", "BF16", "32", { t: "실패", c: C.err }, "vocab 패딩 mismatch (에러④)"],
  ], { fs: 11.5, rowH: 0.46, colW: [3.5, 1.0, 0.7, 1.0, CW - 6.2] });
  s.addText("일관된 패턴: FP8 + 작은 tp(8)는 성공 / BF16 32B + tp=32는 컴파일 미지원", {
    x: M, y: 6.5, w: CW, h: 0.4, margin: 0, fontFace: F.semi, fontSize: 12.5, color: C.blue, align: "center",
  });
})();

/* ===================== 13 — 정리 ===================== */
(() => {
  const s = pptx.addSlide();
  frame(s, "정리 · 교훈", 13, "");
  title(s, "정리 — 모델 준비 체크리스트", "");
  const y = 2.4, h = 1.5, w = (CW - 0.3) / 2;
  const items = [
    ["성공 경로", C.ok, "FP8 양자화 + tp=8 → 1장 적재 + Tracing 메모리 절반. 32B급은 이 조합이 표준"],
    ["BF16 32B tp=32 회피", C.err, "embedding_table 컴파일 미지원 — 메모리를 늘려도 안 됨. FP8 변종으로 전환"],
    ["preset 순서 주의", C.blue, "(hidden, intermediate)가 겹치는 preset은 의도한 것을 먼저 등록 (동점 시 first 선택)"],
    ["OOM은 메모리 축소로", C.warn, "swap·oomd 끄기로는 못 막음. --max-model-len을 줄여 working set 자체를 낮춰야"],
  ];
  items.forEach(([nm, col, desc], i) => {
    const x = M + (i % 2) * (w + 0.3);
    const yy = y + Math.floor(i / 2) * (h + 0.25);
    card(s, x, yy, w, h, { shadow: shStd() });
    accent(s, x, yy + 0.22, 0.45, col);
    s.addText(nm, { x: x + 0.28, y: yy + 0.2, w: w - 0.5, h: 0.35, margin: 0, fontFace: F.bold, fontSize: 15, color: C.ink });
    s.addText(desc, { x: x + 0.28, y: yy + 0.64, w: w - 0.55, h: h - 0.78, margin: 0, fontFace: F.reg, fontSize: 11.5, color: C.ink2, lineSpacingMultiple: 1.36, valign: "top" });
  });
  s.addText("\"빌드 가능\" ≠ \"우리 머신서 서빙 가능\" — 빌드는 host RAM, 서빙은 NPU HBM. 둘을 따로 확인해야 한다", {
    x: M, y: 5.85, w: CW, h: 0.6, margin: 0, fontFace: F.semi, fontSize: 13, color: C.blue, align: "center", lineSpacingMultiple: 1.35,
  });
})();

pptx.writeFile({ fileName: "RNGD_Model_Prep.pptx" }).then((f) => console.log("saved:", f));
