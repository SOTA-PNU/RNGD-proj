/* RNGD 지원 ATen op 검증 — 목록 재검증 + 실제 NPU 실행 검증 (16:9, 6장)
 * 모든 수치는 이 머신에서 직접 실행한 실측값(추정 없음).
 * 출처: furiosa.torch.db.aten_config(목록) · CompileModule.from_exported(컴파일) ·
 *       info/op_verify/*.py (실행 스크립트) · furiosa-torch 2026.2.0 / torch 2.10.0 / RNGD 4장.
 * 기존 build_model_framework.js / build_serve_cli.js (Paperlogy) 스타일 준수. */
const pptx = new (require("pptxgenjs"))();
pptx.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pptx.layout = "W";
pptx.author = "RNGD Op Support Verification";

const F = {
  black: "Paperlogy 9 Black", xbold: "Paperlogy 8 ExtraBold",
  bold: "Paperlogy 7 Bold", semi: "Paperlogy 6 SemiBold",
  med: "Paperlogy 5 Medium", reg: "Paperlogy 4 Regular",
};
const C = {
  ink: "222222", ink2: "45515e", mut: "8e8e93",
  blue: "1456f0", blue2: "3b82f6", blueBg: "eef4ff",
  white: "ffffff", border2: "e5e7eb", bg2: "f0f0f0",
  dark: "181e25", codeTx: "e5e9ef", codeMut: "8ea0b5", codeAc: "5fc6ff", codeGr: "7ee787", codeRed: "ff8087",
  ok: "16a34a", okBg: "e8ffea", warn: "d97706", warnBg: "fef3c7", err: "dc2626", errBg: "fde8e8",
  vl: "7c3aed", vlBg: "f3e8ff",
};
const M = 0.5, CW = 13.333 - 2 * M;

function chapter(s, t) {
  s.addText(t, { x: M, y: 0.32, w: CW, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 12, color: C.mut, charSpacing: 0.6 });
}
function title(s, head, sub, y) {
  y = y || 0.62;
  s.addText(head, { x: M, y, w: CW, h: 0.62, margin: 0, fontFace: F.bold, fontSize: 26, color: C.ink, charSpacing: -0.6, lineSpacingMultiple: 1.1 });
  if (sub) s.addText(sub, { x: M, y: y + 0.6, w: CW, h: 0.5, margin: 0, fontFace: F.med, fontSize: 12.5, color: C.ink2, lineSpacingMultiple: 1.32 });
}
function card(s, x, y, w, h, opt = {}) {
  s.addShape(pptx.ShapeType.roundRect, { x, y, w, h, rectRadius: opt.r || 0.1, fill: { color: opt.fill || C.white }, line: opt.line === null ? { type: "none" } : { color: opt.line || C.border2, width: 1 } });
}
function accent(s, x, y, h, color) {
  s.addShape(pptx.ShapeType.roundRect, { x, y, w: 0.07, h, rectRadius: 0.03, fill: { color }, line: { type: "none" } });
}
function chip(s, x, y, text, fill, txt, fs) {
  const w = 0.4 + text.length * 0.105;
  s.addShape(pptx.ShapeType.roundRect, { x, y, w, h: 0.42, rectRadius: 0.21, fill: { color: fill }, line: { type: "none" } });
  s.addText(text, { x, y, w, h: 0.42, margin: 0, align: "center", valign: "middle", fontFace: F.semi, fontSize: fs || 11.5, color: txt });
  return w;
}
function srcline(s, t) {
  s.addText(t, { x: M, y: 7.16, w: CW, h: 0.26, margin: 0, fontFace: F.reg, fontSize: 8.3, color: C.mut, lineSpacingMultiple: 1.1 });
}
// dark code block. lines = [[{t,c,f}, ...], ...]
function codeBlock(s, x, y, w, h, lines, fs) {
  fs = fs || 9.5;
  card(s, x, y, w, h, { fill: C.dark, line: null, r: 0.09 });
  const runs = [];
  lines.forEach((segs, li) => {
    if (segs.length === 0) { runs.push({ text: " ", options: { fontSize: fs, breakLine: true } }); return; }
    segs.forEach((sg, i) => runs.push({
      text: sg.t,
      options: { fontFace: sg.f || F.reg, fontSize: fs, color: sg.c || C.codeTx, breakLine: i === segs.length - 1 },
    }));
  });
  s.addText(runs, { x: x + 0.2, y: y + 0.16, w: w - 0.4, h: h - 0.3, margin: 0, valign: "top", lineSpacingMultiple: 1.18 });
}

/* ===================== 1. 개요 / 결론 ===================== */
(() => {
  const s = pptx.addSlide(); s.background = { color: C.white };
  chapter(s, "FURIOSA.TORCH · ATen 연산자(OP) 지원 검증 · 2026-06-09");
  title(s,
    "RNGD 지원 op — 목록 재검증 + NPU 실제 실행 검증",
    "furiosa.torch.db.SUPPORTED_ATEN_OPS 97개를 직접 추출해 목록을 다시 확인하고, op마다 최소 그래프를 만들어 RNGD에서 컴파일·실행까지 돌려 검증했습니다.");

  // 결론 카드
  const cy = 2.12, ch = 0.92;
  card(s, M, cy, CW, ch, { fill: C.bg2, line: null });
  accent(s, M, cy + 0.18, ch - 0.36, C.blue);
  s.addText("한 줄 결론", { x: M + 0.26, y: cy + 0.14, w: 2, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 13, color: C.blue });
  s.addText([
    { text: "‘지원 목록(SUPPORTED)’은 컴파일러가 ", options: { fontFace: F.med, fontSize: 13, color: C.ink2 } },
    { text: "“받겠다”고 선언한 목록일 뿐, NPU 실행 보장이 아닙니다.", options: { fontFace: F.semi, fontSize: 13, color: C.ink } },
    { text: "  97개 중 89개는 실제로 돌고, 6개는 특정 모양에서만, 2개는 목록에 있어도 안 됩니다.", options: { fontFace: F.med, fontSize: 13, color: C.ink2 } },
  ], { x: M + 0.26, y: cy + 0.44, w: CW - 0.5, h: 0.4, margin: 0, valign: "middle", lineSpacingMultiple: 1.2 });

  // 숫자 chips
  let cx = M, chy = 3.3;
  cx += chip(s, cx, chy, "97개 검증", C.blue, C.white) + 0.18;
  cx += chip(s, cx, chy, "89 실행 OK", C.ok, C.white) + 0.18;
  cx += chip(s, cx, chy, "6 조건부", C.warn, C.white) + 0.18;
  cx += chip(s, cx, chy, "2 불가", C.err, C.white) + 0.18;
  cx += chip(s, cx, chy, "목록 오류 1", C.vl, C.white) + 0.18;

  // 정정 2건
  const ty = 4.06, th = 1.62, halfW = (CW - 0.3) / 2;
  card(s, M, ty, halfW, th, { fill: C.white });
  accent(s, M, ty + 0.2, th - 0.4, C.err);
  s.addText("정정 ① — import 경로", { x: M + 0.28, y: ty + 0.18, w: halfW - 0.4, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 13, color: C.ink });
  s.addText([
    { text: "from furiosa.torch.extension import SUPPORTED_ATEN_OPS", options: { fontFace: F.semi, fontSize: 11, color: C.err, strike: true } },
    { text: "\n→ extension 모듈엔 없음(hasattr=False). 올바른 경로:", options: { fontFace: F.med, fontSize: 11, color: C.ink2 } },
    { text: "\nfrom furiosa.torch.db import SUPPORTED_ATEN_OPS", options: { fontFace: F.semi, fontSize: 11.5, color: C.ok } },
  ], { x: M + 0.28, y: ty + 0.52, w: halfW - 0.5, h: 1.0, margin: 0, valign: "top", lineSpacingMultiple: 1.28 });

  const rx = M + halfW + 0.3;
  card(s, rx, ty, halfW, th, { fill: C.white });
  accent(s, rx, ty + 0.2, th - 0.4, C.vl);
  s.addText("정정 ② — 목록 내용 1개", { x: rx + 0.28, y: ty + 0.18, w: halfW - 0.4, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 13, color: C.ink });
  s.addText([
    { text: "기존 정리본: ", options: { fontFace: F.med, fontSize: 11.5, color: C.ink2 } },
    { text: "detach_copy", options: { fontFace: F.semi, fontSize: 11.5, color: C.err, strike: true } },
    { text: "   →   실제 목록: ", options: { fontFace: F.med, fontSize: 11.5, color: C.ink2 } },
    { text: "copy_", options: { fontFace: F.semi, fontSize: 11.5, color: C.ok } },
    { text: " (제자리 복사)", options: { fontFace: F.reg, fontSize: 11, color: C.mut } },
    { text: "\n나머지 96개는 정확히 일치. 양쪽 다 97개라 개수로는 안 드러남.", options: { fontFace: F.med, fontSize: 11, color: C.ink2 } },
  ], { x: rx + 0.28, y: ty + 0.56, w: halfW - 0.5, h: 1.0, margin: 0, valign: "top", lineSpacingMultiple: 1.3 });

  srcline(s, "환경: furiosa-torch 2026.2.0 / torch 2.10.0 (venv ~/furiosa) · RNGD 4장(firmware 2026.2.1, driver 2026.2.0, furiosa-smi 확인) · 실행 카드 rngd:3 · 목록 정의 furiosa/torch/db/aten_config.py");
})();

/* ===================== 2. 방법 + 실행 코드 ===================== */
(() => {
  const s = pptx.addSlide(); s.background = { color: C.white };
  chapter(s, "검증 방법 · 어떻게 op를 찾고, 어떻게 NPU에서 돌렸나");
  title(s,
    "op 목록 추출 → 그래프화 → 컴파일(EDF) → NPU 실행 → CPU 대조",
    "op 하나마다 그 op만 쓰는 최소 모듈을 만들어 furiosa-llm 과 같은 분해 경로로 컴파일하고 rngd:3 에서 실행해 CPU 결과와 비교했습니다.");

  const cy = 2.06, codeW = 7.35, codeH = 4.62;
  codeBlock(s, M, cy, codeW, codeH, [
    [{ t: "# 1) 지원 op 목록(97개) — 권위 있는 출처", c: C.codeMut }],
    [{ t: "from", c: C.codeRed, f: F.semi }, { t: " furiosa.torch.db ", c: C.codeTx }, { t: "import", c: C.codeRed, f: F.semi }, { t: " SUPPORTED_ATEN_OPS", c: C.codeAc }],
    [{ t: "len(SUPPORTED_ATEN_OPS)  ", c: C.codeTx }, { t: "# = 97   (IMPORTABLE = 156)", c: C.codeMut }],
    [{ t: "# 정의: aten_config.py → native_compiler.is_supported_aten()", c: C.codeMut }],
    [],
    [{ t: "# 2) op 1개당 최소 그래프 → 컴파일 → NPU 실행", c: C.codeMut }],
    [{ t: "import", c: C.codeRed, f: F.semi }, { t: " torch, furiosa.torch     ", c: C.codeTx }, { t: "# 순서: torch 먼저", c: C.codeMut }],
    [{ t: "from", c: C.codeRed, f: F.semi }, { t: " furiosa.torch ", c: C.codeTx }, { t: "import", c: C.codeRed, f: F.semi }, { t: " CompileModule", c: C.codeAc }],
    [{ t: "from", c: C.codeRed, f: F.semi }, { t: " torch._decomp ", c: C.codeTx }, { t: "import", c: C.codeRed, f: F.semi }, { t: " core_aten_decompositions", c: C.codeAc }],
    [{ t: "TABLE = ", c: C.codeTx }, { t: "dict", c: C.codeAc }, { t: "(core_aten_decompositions())  ", c: C.codeTx }, { t: "# = furiosa-llm 경로", c: C.codeMut }],
    [],
    [{ t: "ep = torch.export.export(Mod().eval(), (x,))", c: C.codeTx }],
    [{ t: "         .run_decompositions(TABLE)", c: C.codeTx }],
    [{ t: "cm = CompileModule.from_exported(ep)   ", c: C.codeGr }, { t: "# ← EDF 컴파일", c: C.codeMut }],
    [{ t: "cm.to(torch.device(", c: C.codeTx }, { t: "\"rngd\"", c: C.codeGr }, { t: ", 3))", c: C.codeTx }],
    [{ t: "out = cm(x.to(dev), device=dev)        ", c: C.codeGr }, { t: "# ← NPU 실행", c: C.codeMut }],
    [{ t: "# out  vs  Mod()(x)  (CPU eager) 비교", c: C.codeMut }],
  ], 10);

  // 오른쪽: 판정 + 강건성
  const rx = M + codeW + 0.3, rw = CW - codeW - 0.3;
  const boxes = [
    ["판정 3축", C.blue, [
      ["present", "분해 후 그래프에 그 op 노드가 실제로 남았는가"],
      ["compile", "from_exported 가 EDF 생성에 성공하는가"],
      ["run", "NPU 결과가 CPU eager 와 일치하는가"],
    ]],
    ["검증 3단계 + 교차검증", C.ok, [
      ["①", "op 단독 그래프"],
      ["②", "실패 시 sigmoid+add 실연산 그래프에 끼워서 재시험"],
      ["③", "dtype·랭크·모양·API 바꿔가며 흔들기"],
      ["④", "의심 8개를 독립 에이전트가 “되게 만들어보라” 반대검증"],
    ]],
  ];
  let by = cy;
  boxes.forEach(([t, col, rows], bi) => {
    const bh = bi === 0 ? 1.66 : 2.66;
    card(s, rx, by, rw, bh, { fill: C.white });
    accent(s, rx, by + 0.18, bh - 0.36, col);
    s.addText(t, { x: rx + 0.26, y: by + 0.14, w: rw - 0.4, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 13, color: C.ink });
    const rt = [];
    rows.forEach(([k, v], i) => {
      rt.push({ text: k + "  ", options: { fontFace: F.semi, fontSize: 11, color: col } });
      rt.push({ text: v + (i < rows.length - 1 ? "\n" : ""), options: { fontFace: F.reg, fontSize: 10.5, color: C.ink2, breakLine: false } });
    });
    s.addText(rt, { x: rx + 0.26, y: by + 0.5, w: rw - 0.46, h: bh - 0.6, margin: 0, valign: "top", lineSpacingMultiple: 1.26 });
    by += bh + 0.3;
  });

  srcline(s, "실행 스크립트(재현 가능): info/op_verify/verify_round1_all97.py · verify_round2_embedded.py · verify_round3_harden.py · reconcile.py · precision_probe.py · shape_sweep.py  ·  총 4라운드, 130+ 케이스");
})();

/* ===================== 3. 분류별 결과 ===================== */
(() => {
  const s = pptx.addSlide(); s.background = { color: C.white };
  chapter(s, "결과 · 분류별");
  title(s, "분류별 결과 — 16개 카테고리 / 97개 op", null, 0.56);

  const ST = { ok: [C.ok, C.okBg], part: [C.warn, C.warnBg], no: [C.err, C.errBg] };
  // [카테고리, 개수, 상태라벨, 상태키, 대표 op / 비고]
  const data = [
    ["Math unary", "11", "10 OK · 1 불가", "part", "abs·cos·exp·log·sqrt·rsqrt·neg…  /  isnan 불가"],
    ["Arithmetic binary", "10", "OK", "ok", "add·sub·mul·div(.Scalar는 .Tensor로 정규화)·pow·clamp"],
    ["Conv / Matmul", "3", "OK (감소정밀도)", "ok", "convolution·mm·bmm  — 상대오차 ~0.23%"],
    ["Activation", "6", "OK", "ok", "relu·leaky_relu·sigmoid·tanh·softmax·log_softmax"],
    ["Comparison", "14", "OK", "ok", "eq·ne·lt·le·gt·ge(.Scalar/.Tensor)·maximum·minimum"],
    ["Logical", "3", "OK", "ok", "logical_and·logical_not·logical_xor"],
    ["Bitwise", "6", "OK", "ok", "bitwise_and·or·xor (.Scalar/.Tensor)"],
    ["Reduction", "9", "8 OK · 1 조건부", "part", "sum·mean·amax·max·argmax·any·var_mean·topk  /  cumsum 정수만"],
    ["Pooling", "3", "2 OK · 1 조건부", "part", "avg_pool2d·adaptive_avg_pool2d(→mean)  /  max_pool2d_with_indices 조건부"],
    ["Shape / View", "16", "15 OK · 1 조건부", "part", "view·permute·transpose·expand·squeeze·cat·slice…  /  slice_scatter 조건부"],
    ["Split", "2", "OK", "ok", "split_with_sizes · split_with_sizes_copy"],
    ["Copy / Clone", "4", "OK", "ok", "clone · copy · copy_ · _to_copy"],
    ["Indexing", "5", "2 OK · 3 조건부", "part", "index_put·scatter OK  /  index·index_select·gather 조건부"],
    ["Conditional", "1", "OK", "ok", "where.self"],
    ["Creation", "3", "OK", "ok", "full · full_like · fill"],
    ["Padding", "1", "불가", "no", "constant_pad_nd"],
  ];
  const fs = 9.6;
  const head = ["카테고리", "개수", "상태", "대표 op / 비고"].map((h, i) => ({
    text: h, options: { fontFace: F.semi, fontSize: fs, color: C.white, fill: { color: C.blue }, align: i === 1 ? "center" : "left", valign: "middle", margin: [1, 5, 1, 6] },
  }));
  const body = data.map(([cat, n, lbl, k, note]) => {
    const [c, bg] = ST[k];
    return [
      { text: cat, options: { fontFace: F.semi, fontSize: fs, color: C.ink, align: "left", valign: "middle", fill: { color: C.white }, margin: [1, 5, 1, 6] } },
      { text: n, options: { fontFace: F.semi, fontSize: fs, color: C.ink2, align: "center", valign: "middle", fill: { color: C.white } } },
      { text: lbl, options: { fontFace: F.semi, fontSize: fs, color: c, align: "left", valign: "middle", fill: { color: bg }, margin: [1, 5, 1, 6] } },
      { text: note, options: { fontFace: F.reg, fontSize: fs, color: C.ink2, align: "left", valign: "middle", fill: { color: C.white }, margin: [1, 5, 1, 6] } },
    ];
  });
  s.addTable([head, ...body], {
    x: M, y: 1.34, w: CW, colW: [2.05, 0.7, 1.95, CW - 4.7],
    border: { type: "solid", color: C.border2, pt: 0.5 }, rowH: 0.305, valign: "middle",
  });
  srcline(s, "근거: verify_round1_all97.py 가 97개를 모두 export→compile→run. .Scalar 사칙연산·_copy 뷰 변형 등은 export 시 동등 형태(.Tensor·일반 뷰)로 정규화되어 그 형태로 검증됨.");
})();

/* ===================== 4. 정밀도 ===================== */
(() => {
  const s = pptx.addSlide(); s.background = { color: C.white };
  chapter(s, "분류별 특징 · 수치 정밀도 (실측)");
  title(s,
    "elementwise는 CPU와 사실상 동일, matmul 계열만 ~0.23%",
    "matmul(conv·mm·bmm)은 정상 실행되지만 텐서엔진 감소정밀도라 상대오차가 ~0.23%로 일정합니다. 크기를 키워도 상대오차는 그대로고 절대오차만 커집니다.");

  const fs = 11;
  const head = ["케이스", "max 절대오차", "상대오차 (L2)", "코사인 유사도"].map((h, i) => ({
    text: h, options: { fontFace: F.semi, fontSize: fs, color: C.white, fill: { color: C.blue }, align: i === 0 ? "left" : "center", valign: "middle", margin: [2, 6, 2, 7] },
  }));
  // [케이스, abs, rel, cos, 강조여부]
  const rows = [
    ["sigmoid(x)*2  (elementwise)", "2.4e-07", "6.2e-08", "1.0000000", false],
    ["mm  8×8", "0.019", "0.00232", "0.9999976", true],
    ["mm  64×64", "0.074", "0.00233", "0.9999973", true],
    ["mm  256×256", "0.185", "0.00235", "0.9999979", true],
    ["conv2d  3×3", "0.058", "0.00232", "0.9999974", true],
  ];
  const body = rows.map(([c, a, r, cs, hot]) => ([
    { text: c, options: { fontFace: F.semi, fontSize: fs, color: hot ? C.warn : C.ok, align: "left", valign: "middle", fill: { color: hot ? C.warnBg : C.okBg }, margin: [2, 6, 2, 7] } },
    { text: a, options: { fontFace: F.reg, fontSize: fs, color: C.ink, align: "center", valign: "middle", fill: { color: C.white } } },
    { text: r, options: { fontFace: F.semi, fontSize: fs, color: hot ? C.warn : C.ink2, align: "center", valign: "middle", fill: { color: C.white } } },
    { text: cs, options: { fontFace: F.reg, fontSize: fs, color: C.ink2, align: "center", valign: "middle", fill: { color: C.white } } },
  ]));
  s.addTable([head, ...body], {
    x: M, y: 2.34, w: CW, colW: [4.9, 2.5, 2.5, CW - 9.9],
    border: { type: "solid", color: C.border2, pt: 0.5 }, rowH: 0.46, valign: "middle",
  });

  const ky = 5.3, kh = 1.3;
  card(s, M, ky, CW, kh, { fill: C.bg2, line: null });
  accent(s, M, ky + 0.2, kh - 0.4, C.warn);
  s.addText("읽는 법", { x: M + 0.26, y: ky + 0.16, w: 2, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 13, color: C.warn });
  s.addText([
    { text: "matmul 계열의 ‘오차’는 ", options: { fontFace: F.med, fontSize: 12.5, color: C.ink2 } },
    { text: "틀린 게 아니라 NPU 텐서엔진의 정상적인 감소정밀도", options: { fontFace: F.semi, fontSize: 12.5, color: C.ink } },
    { text: "입니다. 상대오차 ~0.23% 고정·코사인 0.99999+ 로 방향이 사실상 일치합니다. 나머지 단항/이항/활성화/비교/shape 연산은 CPU와 거의 비트 단위로 같습니다(상대오차 ~1e-7).", options: { fontFace: F.med, fontSize: 12.5, color: C.ink2 } },
  ], { x: M + 0.26, y: ky + 0.48, w: CW - 0.5, h: 0.7, margin: 0, valign: "top", lineSpacingMultiple: 1.28 });

  srcline(s, "근거: info/op_verify/precision_probe.py (rngd:3). 입력 randn, FP32. 상대오차 = ‖npu−cpu‖₂ / ‖cpu‖₂.");
})();

/* ===================== 5. 조건부 / 불가 상세 ===================== */
(() => {
  const s = pptx.addSlide(); s.background = { color: C.white };
  chapter(s, "결과 · 목록엔 ‘지원’인데 실제로는");
  title(s, "불가 2개 · 조건부 6개 — 되는/안 되는 조건 (실측)", null, 0.56);

  const fs = 9.6;
  const head = ["op", "판정", "되는 조건 / 안 되는 조건", "실측 근거"].map((h, i) => ({
    text: h, options: { fontFace: F.semi, fontSize: fs, color: C.white, fill: { color: C.blue }, align: i === 1 ? "center" : "left", valign: "middle", margin: [1, 5, 1, 6] },
  }));
  // [op, 판정라벨, 판정색, 조건, 근거]
  const data = [
    ["isnan", "불가", C.err, C.errBg, "float16/32/bf16·랭크 0~4·where/any/cast 등 11가지 전부 컴파일 실패", "(x != x) 로 바꾸면 컴파일·실행 OK → 막는 건 isnan 노드 하나"],
    ["constant_pad_nd", "불가", C.err, C.errBg, "F.pad·1D/2D/3D pad·0/비0 value·conv 뒤·Linear 앞 등 12가지 전부 실패", "주변 op(conv/sigmoid/addmm)는 사는데 pad 노드 때문에 그래프 실패"],
    ["cumsum", "조건부", C.warn, C.warnBg, "정수(int32/int64) 입력만 OK · float 입력은 dim·랭크 불문 전부 실패", "int64 1D/2D/3D CPU와 bit-exact · float 18케이스 실패"],
    ["index_select", "조건부", C.warn, C.warnBg, "맨 안쪽(마지막) 차원이 8의 배수일 때만 OK", "cols 8·16 OK / 3·5·7·15·17 FAIL (rows 4~32 무관)"],
    ["index.Tensor (x[idx])", "조건부", C.warn, C.warnBg, "맨 안쪽 차원이 4의 배수면 OK · x[:, idx](비-선두축)는 실패", "cols 4·8·12·16 OK / 3·5·7 FAIL"],
    ["gather", "조건부", C.warn, C.warnBg, "사실상 1차원(벡터) gather만 OK · 다중행 rank-2는 실패", "1D OK / (8,C)·(R,8) 다중행 정렬·dim 불문 전부 FAIL"],
    ["slice_scatter", "조건부", C.warn, C.warnBg, "해당 축 전체 덮어쓰기만 OK(=결과가 src와 동일, 무의미) · 부분 덮기 실패", "full-overwrite OK / start·end 부분 변경 전부 실패"],
    ["max_pool2d_with_indices", "조건부", C.warn, C.warnBg, "indices(int64) 출력은 절대 불가 · values만 써도 kernel≥2는 거의 실패", "ResNet stem(56×56,k3s2p1)·8×8·plain pool 실패"],
  ];
  const body = data.map(([op, lbl, c, bg, cond, ev]) => ([
    { text: op, options: { fontFace: F.semi, fontSize: fs, color: C.ink, align: "left", valign: "middle", fill: { color: C.white }, margin: [1, 5, 1, 6] } },
    { text: lbl, options: { fontFace: F.semi, fontSize: fs, color: c, align: "center", valign: "middle", fill: { color: bg } } },
    { text: cond, options: { fontFace: F.reg, fontSize: fs, color: C.ink2, align: "left", valign: "middle", fill: { color: C.white }, margin: [1, 5, 1, 6] } },
    { text: ev, options: { fontFace: F.reg, fontSize: fs, color: C.ink2, align: "left", valign: "middle", fill: { color: C.white }, margin: [1, 5, 1, 6] } },
  ]));
  s.addTable([head, ...body], {
    x: M, y: 1.34, w: CW, colW: [2.25, 0.95, 5.0, CW - 8.2],
    border: { type: "solid", color: C.border2, pt: 0.5 }, rowH: 0.6, valign: "middle",
  });
  s.addText([
    { text: "공통: ", options: { fontFace: F.semi, fontSize: 10, color: C.ink } },
    { text: "전부 CompileModule.from_exported 에서 UnsupportedOpError('failed to compile the graph') 로 막힘(NPU 실행 단계 전). op 자체는 분해되지 않고 그래프에 남아 있음 → EDF 백엔드에 해당 op의 코드 생성이 없거나 모양 제약이 있음.", options: { fontFace: F.reg, fontSize: 10, color: C.ink2 } },
  ], { x: M, y: 6.66, w: CW, h: 0.4, margin: 0, valign: "middle", lineSpacingMultiple: 1.18 });
  srcline(s, "근거: verify_round2_embedded.py · verify_round3_harden.py · reconcile.py (rngd:3).");
})();

/* ===================== 6. 8-정렬 실측 sweep ===================== */
(() => {
  const s = pptx.addSlide(); s.background = { color: C.white };
  chapter(s, "결과 · 조건부의 정체 (실측 sweep)");
  title(s,
    "gather/index 는 ‘맨 안쪽 차원 정렬’에 따라 성패가 갈린다",
    "같은 코드라도 텐서 모양만 바꾸면 컴파일이 되고 안 됩니다. 맨 안쪽(feature) 차원 크기로 경계가 정확히 갈립니다.");

  // OK/FAIL grid helper
  function grid(x, y, w, label, sub, items) {
    card(s, x, y, w, 1.66, { fill: C.white });
    s.addText(label, { x: x + 0.22, y: y + 0.14, w: w - 0.4, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 12.5, color: C.ink });
    s.addText(sub, { x: x + 0.22, y: y + 0.44, w: w - 0.4, h: 0.3, margin: 0, fontFace: F.med, fontSize: 9.5, color: C.ink2 });
    let gx = x + 0.22; const gy = y + 0.82, cw = 0.52;
    items.forEach(([n, ok]) => {
      s.addShape(pptx.ShapeType.roundRect, { x: gx, y: gy, w: cw, h: 0.6, rectRadius: 0.06, fill: { color: ok ? C.okBg : C.errBg }, line: { color: ok ? C.ok : C.err, width: 0.75 } });
      s.addText([
        { text: n + "\n", options: { fontFace: F.semi, fontSize: 10.5, color: C.ink } },
        { text: ok ? "OK" : "FAIL", options: { fontFace: F.semi, fontSize: 8.5, color: ok ? C.ok : C.err } },
      ], { x: gx, y: gy, w: cw, h: 0.6, margin: 0, align: "center", valign: "middle", lineSpacingMultiple: 0.95 });
      gx += cw + 0.1;
    });
  }

  const y1 = 2.04, y2 = 3.86;
  grid(M, y1, CW, "index_select(x, 0, idx) · 안쪽 차원(cols) 스윕  (rows=8)",
    "rows 는 4~32 전부 OK(무관). cols 가 8의 배수일 때만 OK.",
    [["3", false], ["5", false], ["7", false], ["8", true], ["15", false], ["16", true], ["17", false]]);

  grid(M, y2, (CW - 0.3) / 2, "index.Tensor  x[idx] · cols 스윕  (rows=6)",
    "4의 배수면 OK.",
    [["3", false], ["4", true], ["5", false], ["7", false], ["8", true], ["12", true], ["16", true]]);

  const rx = M + (CW - 0.3) / 2 + 0.3;
  grid(rx, y2, (CW - 0.3) / 2, "gather · 차원/모양 스윕",
    "사실상 1D(벡터)만 OK.",
    [["1D", true], ["(8,4)", false], ["(8,8)", false], ["(8,16)", false], ["(R,8)", false]]);

  const ky = 5.72, kh = 1.18;
  card(s, M, ky, CW, kh, { fill: C.bg2, line: null });
  accent(s, M, ky + 0.18, kh - 0.36, C.blue);
  s.addText("함의", { x: M + 0.26, y: ky + 0.14, w: 2, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 13, color: C.blue });
  s.addText([
    { text: "임의의 모델을 그대로 올리면 gather·index·embedding-lookup·constant_pad_nd 의 feature 폭이 8(또는 4)의 배수가 아닐 때 컴파일이 막힐 수 있습니다. ", options: { fontFace: F.med, fontSize: 12, color: C.ink2 } },
    { text: "“지원 op”라도 모양에 따라 안 됩니다.", options: { fontFace: F.semi, fontSize: 12, color: C.ink } },
    { text: "  (참고: furiosa-smi status — 디바이스당 코어 8개, Core 0~7)", options: { fontFace: F.reg, fontSize: 11, color: C.mut } },
  ], { x: M + 0.26, y: ky + 0.46, w: CW - 0.5, h: 0.6, margin: 0, valign: "top", lineSpacingMultiple: 1.26 });

  srcline(s, "근거: info/op_verify/shape_sweep.py · reconcile.py (rngd:3). 표기 OK=컴파일+실행 성공, FAIL=UnsupportedOpError.");
})();

pptx.writeFile({ fileName: "RNGD_Op_Support.pptx" }).then((f) => console.log("WROTE", f));
