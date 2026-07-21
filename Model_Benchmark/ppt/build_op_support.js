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

/* ===================== 7. dtype 요약 ===================== */
(() => {
  const s = pptx.addSlide(); s.background = { color: C.white };
  chapter(s, "dtype별 op 지원 · 요약 (실측, 2026-06-09)");
  title(s,
    "10개 dtype 중 7개만 컴파일 가능 — bf16 = f32, int8 > int16, fp16은 부분",
    "각 op를 op(x+x) 실연산 그래프로 감싸 dtype별 RNGD 컴파일을 측정했습니다. (serve가 4장 점유 중이라 실행은 EBUSY → 컴파일 게이트 기준. dtype 가용성은 컴파일 단계에서 결정됨.)");

  const fs = 10.5;
  const GR = { ok: [C.ok, C.okBg], warn: [C.warn, C.warnBg], err: [C.err, C.errBg] };
  const head = ["dtype", "등급", "가용 op", "막히는 대표 op"].map((h, i) => ({
    text: h, options: { fontFace: F.semi, fontSize: fs, color: C.white, fill: { color: C.blue }, align: i === 2 ? "center" : "left", valign: "middle", margin: [2, 6, 2, 7] },
  }));
  const data = [
    ["float32", "✅ 완전", "36 / 37", "cumsum", "ok"],
    ["bfloat16", "✅ 완전", "36 / 37", "cumsum  (float32와 동일)", "ok"],
    ["int32", "✅ 정수 강", "34 / 36", "pow · conv2d", "ok"],
    ["int64", "✅ 정수 강", "33 / 36", "pow · conv2d · full_like", "ok"],
    ["int8", "🔶 중간", "28 / 36", "relu · max · argmax · where · cumsum · slice", "warn"],
    ["float16", "🔶 부분", "20 / 37", "비교 · neg · clamp · log · rsqrt · pow · relu · conv2d · mean · max · argmax", "warn"],
    ["int16", "🔶 약함", "18 / 36", "비교 · 단항 다수 · relu · conv2d · max · cumsum · bitwise · where", "warn"],
    ["float64", "❌ 미지원", "0", "전부 (dtype 게이트에서 차단)", "err"],
    ["uint16", "❌ 미지원", "0", "전부", "err"],
    ["uint32", "❌ 미지원", "0", "전부", "err"],
  ];
  const body = data.map(([dt, g, av, blk, k]) => {
    const [c, bg] = GR[k];
    return [
      { text: dt, options: { fontFace: F.semi, fontSize: fs, color: C.ink, align: "left", valign: "middle", fill: { color: C.white }, margin: [2, 6, 2, 7] } },
      { text: g, options: { fontFace: F.semi, fontSize: fs, color: c, align: "left", valign: "middle", fill: { color: bg }, margin: [2, 5, 2, 6] } },
      { text: av, options: { fontFace: F.semi, fontSize: fs, color: C.ink2, align: "center", valign: "middle", fill: { color: C.white } } },
      { text: blk, options: { fontFace: F.reg, fontSize: fs - 0.5, color: C.ink2, align: "left", valign: "middle", fill: { color: C.white }, margin: [2, 6, 2, 7] } },
    ];
  });
  s.addTable([head, ...body], {
    x: M, y: 2.18, w: CW, colW: [1.55, 1.6, 1.2, CW - 4.35],
    border: { type: "solid", color: C.border2, pt: 0.5 }, rowH: 0.36, valign: "middle",
  });

  const ky = 6.16, kh = 0.86;
  card(s, M, ky, CW, kh, { fill: C.bg2, line: null });
  accent(s, M, ky + 0.16, kh - 0.32, C.blue);
  s.addText([
    { text: "핵심: ", options: { fontFace: F.bold, fontSize: 11.5, color: C.blue } },
    { text: "NPU는 ", options: { fontFace: F.med, fontSize: 11.5, color: C.ink2 } },
    { text: "bf16을 1급 부동소수", options: { fontFace: F.semi, fontSize: 11.5, color: C.ink } },
    { text: "로 다루고(=float32 동급) fp16은 비교·일부 단항·relu·conv·reduction에서 막힙니다. 정수는 ", options: { fontFace: F.med, fontSize: 11.5, color: C.ink2 } },
    { text: "int32/int64가 강하고 int8(conv2d까지) > int16", options: { fontFace: F.semi, fontSize: 11.5, color: C.ink } },
    { text: ". float64·uint16·uint32는 EDF가 dtype 자체를 거부합니다.", options: { fontFace: F.med, fontSize: 11.5, color: C.ink2 } },
  ], { x: M + 0.26, y: ky + 0.16, w: CW - 0.5, h: kh - 0.3, margin: 0, valign: "middle", lineSpacingMultiple: 1.22 });

  srcline(s, "근거: info/op_verify/dtype_matrix.py (op(x+x) embed, CompileModule.from_exported). 가용 = O / (전체−N/A). furiosa-torch 2026.2.0.");
})();

/* ===================== 8. 전체 매트릭스 ===================== */
(() => {
  const s = pptx.addSlide(); s.background = { color: C.white };
  chapter(s, "dtype별 op 지원 · 전체 매트릭스 (실측)");
  title(s, "op × dtype 컴파일 매트릭스 — 38 op × 10 dtype", null, 0.52);

  const DTS = ["f64", "f32", "f16", "bf16", "i64", "i32", "i16", "i8", "u16", "u32"];
  // 'O' 컴파일OK · 'x' unsup · '-' N/A
  const R = [
    ["add","x O O O O O O O - -"],["sub","x O O O O O O O - -"],["mul","x O O O O O O O - -"],
    ["div","x O O O O O O O - -"],["pow","x O x O x x x x - -"],["clamp","x O x O O O x O - -"],
    ["abs","x O O O O O x O - -"],["neg","x O x O O O x O - -"],["exp","x O O O O O O O - -"],
    ["log","x O x O O O x O - -"],["sqrt","x O O O O O x O - -"],["rsqrt","x O x O O O x O - -"],
    ["sin","x O O O O O O O - -"],["erf","x O O O O O O O - -"],["sigmoid","x O O O O O O O - -"],
    ["tanh","x O O O O O O O - -"],["softmax","x O O O - - - - - -"],["relu","x O x O O O x x - -"],
    ["mm","x O O O O O O O - -"],["conv2d","x O x O x x x O - -"],["eq","x O x O O O x O - -"],
    ["lt","x O x O O O x O - -"],["maximum","x O O O O O O O - -"],["logical_and","x O x O O O x O x x"],
    ["bitwise_and","- - - - O O x x - -"],["sum","x O O O O O O O - -"],["mean","x O x O - - - - - -"],
    ["max.dim","x O x O O O x x - -"],["argmax","x O x O O O x x - -"],["cumsum","x x x x O O x x - -"],
    ["view","x O O O O O O O - -"],["cat","x O O O O O O O - -"],["permute","x O O O O O O O - -"],
    ["slice","x O x O O O O x - -"],["where","x O x O O O x x - -"],["clone","x O O O O O O O - -"],
    ["to_float32","x O O O O O O O - -"],["full_like","O O x O x O x O - -"],
  ];
  // 좌표 기반 직접 그리드 (addTable colW 충돌 회피)
  const x0 = M, y0 = 1.30, labW = 1.95, dcw = (CW - labW) / 10;
  const hH = 0.30, rH = (6.92 - (y0 + hH)) / R.length;
  // 헤더
  s.addShape(pptx.ShapeType.rect, { x: x0, y: y0, w: labW, h: hH, fill: { color: C.blue }, line: { type: "none" } });
  s.addText("op", { x: x0 + 0.08, y: y0, w: labW - 0.1, h: hH, margin: 0, valign: "middle", align: "left", fontFace: F.semi, fontSize: 8.5, color: C.white });
  DTS.forEach((d, j) => {
    const cx = x0 + labW + j * dcw;
    s.addShape(pptx.ShapeType.rect, { x: cx, y: y0, w: dcw, h: hH, fill: { color: C.blue }, line: { color: C.white, width: 0.5 } });
    s.addText(d, { x: cx, y: y0, w: dcw, h: hH, margin: 0, valign: "middle", align: "center", fontFace: F.semi, fontSize: 8, color: C.white });
  });
  // 데이터 행
  const STY = { O: [C.ok, C.okBg, "O"], x: [C.err, C.errBg, "✗"], "-": [C.mut, C.bg2, "—"] };
  R.forEach(([op, cells], i) => {
    const ry = y0 + hH + i * rH;
    s.addShape(pptx.ShapeType.rect, { x: x0, y: ry, w: labW, h: rH, fill: { color: i % 2 ? "fafbfc" : C.white }, line: { color: C.border2, width: 0.3 } });
    s.addText(op, { x: x0 + 0.08, y: ry, w: labW - 0.1, h: rH, margin: 0, valign: "middle", align: "left", fontFace: F.semi, fontSize: 7, color: C.ink });
    cells.split(" ").forEach((v, j) => {
      const [fg, bg, gl] = STY[v];
      const cx = x0 + labW + j * dcw;
      s.addShape(pptx.ShapeType.rect, { x: cx, y: ry, w: dcw, h: rH, fill: { color: bg }, line: { color: C.white, width: 0.4 } });
      s.addText(gl, { x: cx, y: ry, w: dcw, h: rH, margin: 0, valign: "middle", align: "center", fontFace: F.semi, fontSize: 7.5, color: fg });
    });
  });
  s.addText([
    { text: "O", options: { fontFace: F.semi, fontSize: 9.5, color: C.ok } },
    { text: " 컴파일 성공    ", options: { fontFace: F.reg, fontSize: 9.5, color: C.ink2 } },
    { text: "✗", options: { fontFace: F.semi, fontSize: 9.5, color: C.err } },
    { text: " unsup(UnsupportedOpError)    ", options: { fontFace: F.reg, fontSize: 9.5, color: C.ink2 } },
    { text: "—", options: { fontFace: F.semi, fontSize: 9.5, color: C.mut } },
    { text: " 미정의(torch에 그 op·dtype 조합 자체가 없음)", options: { fontFace: F.reg, fontSize: 9.5, color: C.ink2 } },
  ], { x: M, y: 6.9, w: CW, h: 0.26, margin: 0, valign: "middle" });
  srcline(s, "근거: info/op_verify/dtype_matrix.py · dtype_matrix_standalone.py (rngd:3, 컴파일 게이트). slice·logical_and은 출력 materialize/bool 입력 보정값.");
})();

/* ===================== 9. op 실행 위치 분류 ===================== */
(() => {
  const s = pptx.addSlide(); s.background = { color: C.white };
  chapter(s, "op 실행 위치 분류 · 97개 전수 (실측, 2026-06-09)");
  title(s,
    "97개 op 분류 — npu 89 · host 8 · compile_fail 0 · trace_unsupported 0 · crash 0",
    "op마다 별도 subprocess로 격리 실행(크래시 감지). AOT 빌드 경로(CompileModule.from_exported)와 eager 런타임 경로(RngdTensor + coverage)를 둘 다 측정. (실연산 그래프 기준)");

  // 숫자 chips
  let cx = M, cy = 2.04;
  cx += chip(s, cx, cy, "npu 89", C.ok, C.white) + 0.16;
  cx += chip(s, cx, cy, "host 8", C.warn, C.white) + 0.16;
  cx += chip(s, cx, cy, "compile_fail 0", C.mut, C.white) + 0.16;
  cx += chip(s, cx, cy, "trace_unsupported 0", C.mut, C.white) + 0.16;
  cx += chip(s, cx, cy, "crash 0", C.mut, C.white) + 0.16;

  // 좌: 분류 정의
  const ly = 2.78, lw = 6.0, lh = 3.4;
  card(s, M, ly, lw, lh, { fill: C.white });
  accent(s, M, ly + 0.2, lh - 0.4, C.blue);
  s.addText("분류 정의", { x: M + 0.26, y: ly + 0.16, w: lw - 0.4, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 13, color: C.ink });
  const defs = [
    ["npu", C.ok, "AOT 컴파일 + NPU 실행 성공 (coverage run_on_rngd)"],
    ["host", C.warn, "EDF 미지원 → eager 경로서 CPU fallback 실행. AOT 경로선 UnsupportedOpError"],
    ["compile_fail", C.err, "AOT 컴파일 실패 + 실행 불가 (단독 op 그래프 한정)"],
    ["trace_unsupported", C.vl, "torch.export 트레이스 단계 실패"],
    ["crash", C.ink2, "프로세스 비정상 종료(네이티브 abort/segfault)"],
  ];
  const rt = [];
  defs.forEach(([k, c, v], i) => {
    rt.push({ text: k + "  ", options: { fontFace: F.semi, fontSize: 11.5, color: c } });
    rt.push({ text: v + (i < defs.length - 1 ? "\n" : ""), options: { fontFace: F.reg, fontSize: 11, color: C.ink2, breakLine: false } });
  });
  s.addText(rt, { x: M + 0.26, y: ly + 0.54, w: lw - 0.46, h: lh - 0.7, margin: 0, valign: "top", lineSpacingMultiple: 1.42 });

  // 우: host 8개 + 의미
  const rx = M + lw + 0.3, rw = CW - lw - 0.3;
  card(s, rx, ly, rw, lh, { fill: C.warnBg, line: null });
  accent(s, rx, ly + 0.2, lh - 0.4, C.warn);
  s.addText("host 8개 — EDF 미지원, CPU fallback", { x: rx + 0.26, y: ly + 0.16, w: rw - 0.4, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 13, color: C.warn });
  s.addText("isnan · cumsum · max_pool2d_with_indices · slice_scatter · index.Tensor · index_select · gather · constant_pad_nd",
    { x: rx + 0.26, y: ly + 0.52, w: rw - 0.5, h: 0.9, margin: 0, fontFace: F.semi, fontSize: 11.5, color: C.ink, lineSpacingMultiple: 1.3 });
  s.addText([
    { text: "두 경로 결과가 다릅니다. ", options: { fontFace: F.semi, fontSize: 11, color: C.ink } },
    { text: "eager(torch.compile) 런타임에선 CPU에서 돌아 결과는 나오지만(host), AOT build/serve 경로에선 UnsupportedOpError로 막힙니다(compile_fail). 모델 serve 시 이 8개가 그래프에 남으면 빌드가 막힙니다.", options: { fontFace: F.reg, fontSize: 11, color: C.ink2 } },
  ], { x: rx + 0.26, y: ly + 1.5, w: rw - 0.5, h: 1.0, margin: 0, valign: "top", lineSpacingMultiple: 1.28 });
  s.addText([
    { text: "degeneracy 7개: ", options: { fontFace: F.semi, fontSize: 10.5, color: C.ink2 } },
    { text: "expand·expand_copy·slice.Tensor·split_with_sizes(_copy)·copy·copy_ 는 단독 op 그래프에선 compile_fail이나, 실연산 그래프에 넣으면 npu (그래서 실연산 기준 compile_fail=0).", options: { fontFace: F.reg, fontSize: 10.5, color: C.ink2 } },
  ], { x: rx + 0.26, y: ly + 2.5, w: rw - 0.5, h: 0.8, margin: 0, valign: "top", lineSpacingMultiple: 1.26 });

  srcline(s, "근거: info/op_verify/classify_worker.py(op 1개 AOT+eager 2경로 측정) · classify_runner.py(97개 subprocess 격리 + 집계). coverage.py·backend/eager.py. NPU 4장 유휴 상태 실행.");
})();

/* ===================== 10. op × dtype 히트맵 · 카테고리별 정렬·색 ===================== */
(() => {
  const s = pptx.addSlide(); s.background = { color: C.white };
  chapter(s, "op × dtype 분류 히트맵 · 97개, 카테고리별 정렬·색 (실측, 2026-06-09)");
  title(s, "op × dtype 히트맵 — 97개 (카테고리별로 묶음, 맨 오른쪽 열 = 카테고리)", null, 0.48);
  s.addText("셀 = (op,dtype) 분류(하단 범례). op 를 카테고리별로 정렬하고 맨 오른쪽 열에 카테고리를 묶어 표시했습니다. 단독 op 그래프 기준 — C(compile_fail)의 shape·meta 계열은 degeneracy로 실연산 그래프에선 npu(§7).",
    { x: M, y: 0.98, w: CW, h: 0.4, margin: 0, fontFace: F.med, fontSize: 9.5, color: C.ink2, lineSpacingMultiple: 1.22 });

  const COLS = ["f64","f32","f16","bf16","i64","i32","i16","i8","u16","u32"];
  const CC = { unary:"dbeafe", activation:"ccfbf1", binary:"ede9fe", mn:"e0e7ff", matmul:"fee2e2", conv:"ffedd5", reduction:"dcfce7", pool:"ecfccb", shape:"e2e8f0", index:"fce7f3", ternary:"f0e6d2", creation:"fef9c3", pad:"fae8ff", meta:"e5e7eb" };
  const RC = [
["unary","abs","H N N N N N H N - -"],["unary","cos","H N N N N N N N C C"],["unary","erf","H N N N N N N N C C"],
["unary","exp","H N N N N N N N C C"],["unary","log","H N N N N N N N C C"],["unary","neg","H N H N N N H N - -"],
["unary","reciprocal","H N H N N N N N C C"],["unary","rsqrt","H N H N N N N N C C"],["unary","sin","H N N N N N N N C C"],
["unary","sqrt","H N N N N N N N C C"],["unary","isnan","H H H H H H H H C C"],["unary","pow.Tensor_Scalar","H N H N H H H H C C"],
["unary","clamp","H N H N H H H H C C"],["unary","logical_not","N N N N N N N N N N"],
["activation","relu","H N H N N N H H - -"],["activation","leaky_relu","H N H N - - - - - -"],["activation","sigmoid","H N N N N N N N C C"],
["activation","tanh","H N N N N N N N C C"],["activation","_softmax","H N N N - - - - - -"],["activation","_log_softmax","H N N N - - - - - -"],
["binary","add.Scalar","H N H N N N N N C C"],["binary","add.Tensor","H N N N N N N N - -"],["binary","sub.Scalar","H N H N N N N N C C"],
["binary","sub.Tensor","H N N N N N N N - -"],["binary","mul.Scalar","H N H N N N N N C C"],["binary","mul.Tensor","H N N N N N N N C C"],
["binary","div.Scalar","H N H N N N N N C C"],["binary","div.Tensor","H N N N N N N N C C"],["binary","eq.Scalar","H N H N N N H H C C"],
["binary","eq.Tensor","H N H N N N H N C C"],["binary","ne.Scalar","H N H N N N H H C C"],["binary","ne.Tensor","H N H N N N H N C C"],
["binary","lt.Scalar","H N H N N N H H C C"],["binary","lt.Tensor","H N H N N N H N - -"],["binary","le.Scalar","H N H N N N H H C C"],
["binary","le.Tensor","H N H N N N H N - -"],["binary","gt.Scalar","H N H N N N H H C C"],["binary","gt.Tensor","H N H N N N H N - -"],
["binary","ge.Scalar","H N H N N N H H C C"],["binary","ge.Tensor","H N H N N N H N - -"],["binary","logical_and","N N N N N N N N N N"],
["binary","logical_xor","N N N N N N N N N N"],["binary","bitwise_and.Scalar","- - - - N N H H C C"],["binary","bitwise_and.Tensor","- - - - N N H H C C"],
["binary","bitwise_or.Scalar","- - - - N N H H C C"],["binary","bitwise_or.Tensor","- - - - N N H H C C"],["binary","bitwise_xor.Scalar","- - - - N N H H C C"],
["binary","bitwise_xor.Tensor","- - - - N N H H C C"],["mn","maximum","H N N N N N N N - -"],["mn","minimum","H N N N N N N N - -"],
["matmul","mm","H N N N N N N N - -"],["matmul","bmm","H N N N N N N N - -"],["conv","convolution","H N H N H H H N - -"],
["reduction","sum.dim_IntList","H N N N N N N N C C"],["reduction","mean.dim","H N H N - - - - - -"],["reduction","amax","H N N N N N N N - -"],
["reduction","max.dim","H N H N N N H H - -"],["reduction","argmax","H N H N N N H H - -"],["reduction","any.dim","N N N N N N N N N N"],
["reduction","cumsum","H H H H N N H H C C"],["reduction","var_mean.correction","H N H N - - - - - -"],["reduction","topk","H N H N H H H H - -"],
["pool","avg_pool2d","H N H N H - - - - -"],["pool","max_pool2d_with_indices","H H H H H H H H - -"],["pool","_adaptive_avg_pool2d","H N H N - - - - - -"],
["shape","view","C N N N C N N N C C"],["shape","view_copy","C N N N C N N N C C"],["shape","permute","C N N N N N N N C C"],
["shape","permute_copy","C N N N N N N N C C"],["shape","transpose_copy.int","C N N N N N N N C C"],["shape","t_copy","C N N N N N N N C C"],
["shape","squeeze.dim","C N N N C N N N C C"],["shape","squeeze.dims","C N N N C N N N C C"],["shape","squeeze_copy.dim","C N N N C N N N C C"],
["shape","unsqueeze","C N N N C N N N C C"],["shape","unsqueeze_copy","C N N N C N N N C C"],["shape","expand","C C C C C C C C C C"],
["shape","expand_copy","C C C C C C C C C C"],["shape","cat","N N N N N N N N C C"],["shape","slice.Tensor","C C C C C C C C C C"],
["shape","slice_scatter","H H H H H H H H C C"],["shape","split_with_sizes","C C C C C C C C C C"],["shape","split_with_sizes_copy","C C C C C C C C C C"],
["index","index.Tensor","H H H H H H H H C C"],["index","index_put","H N N N N N N N - -"],["index","index_select","H H H H H H H H C C"],
["index","gather","H H H H H H H H - -"],["index","scatter.src","H N N N N N N N - -"],["ternary","where.self","H N N N N N H H - -"],
["creation","full","C N N N N N N N C C"],["creation","full_like","N N H N H N H H C C"],["creation","fill.Scalar","N N H N H N H H C C"],
["pad","constant_pad_nd","H H H H H H H H C C"],["meta","clone","N N N N N N N N C C"],["meta","copy","C C C C C C C C C C"],
["meta","copy_","C C C C C C C C C C"],["meta","_to_copy","C N N N N N N N C C"],
  ];
  const STY = { N:[C.ok,C.okBg,"N"], H:[C.warn,C.warnBg,"H"], C:[C.err,C.errBg,"C"], "-":[C.mut,C.bg2,"—"], T:[C.vl,C.vlBg,"T"], X:[C.ink2,"e5e7eb","X"] };
  const RH = (6.86 - 1.7) / 49;
  function grid(x0, gw, rows, y0) {
    const labW = 1.32, catW = 0.82, dcw = (gw - labW - catW) / 10, hH = 0.2;
    // 헤더: op | dtype들 | category
    s.addShape(pptx.ShapeType.rect, { x: x0, y: y0, w: labW, h: hH, fill: { color: C.blue }, line: { type: "none" } });
    s.addText("op", { x: x0+0.05, y: y0, w: labW-0.07, h: hH, margin:0, valign:"middle", align:"left", fontFace:F.semi, fontSize:6, color:C.white });
    COLS.forEach((d,j) => {
      const cx = x0+labW+j*dcw;
      s.addShape(pptx.ShapeType.rect, { x:cx, y:y0, w:dcw, h:hH, fill:{color:C.blue}, line:{color:C.white,width:0.4} });
      s.addText(d, { x:cx, y:y0, w:dcw, h:hH, margin:0, valign:"middle", align:"center", fontFace:F.semi, fontSize:5.4, color:C.white });
    });
    const catX = x0+labW+10*dcw;
    s.addShape(pptx.ShapeType.rect, { x:catX, y:y0, w:catW, h:hH, fill:{color:C.blue}, line:{color:C.white,width:0.4} });
    s.addText("category", { x:catX, y:y0, w:catW, h:hH, margin:0, valign:"middle", align:"center", fontFace:F.semi, fontSize:5.6, color:C.white });
    // 행: op | cells (op 라벨은 흰/줄무늬, 카테고리색 제거)
    rows.forEach(([cat,op,cells],i) => {
      const ry = y0+hH+i*RH;
      s.addShape(pptx.ShapeType.rect, { x:x0, y:ry, w:labW, h:RH, fill:{color: i%2?"fafbfc":C.white}, line:{color:C.border2,width:0.25} });
      s.addText(op, { x:x0+0.04, y:ry, w:labW-0.05, h:RH, margin:0, valign:"middle", align:"left", fontFace:F.semi, fontSize:5.2, color:C.ink });
      cells.split(" ").forEach((v,j) => {
        const [fg,bg,gl] = STY[v];
        const cx = x0+labW+j*dcw;
        s.addShape(pptx.ShapeType.rect, { x:cx, y:ry, w:dcw, h:RH, fill:{color:bg}, line:{color:C.white,width:0.3} });
        s.addText(gl, { x:cx, y:ry, w:dcw, h:RH, margin:0, valign:"middle", align:"center", fontFace:F.semi, fontSize:5.6, color:fg });
      });
    });
    // 맨 오른쪽 카테고리 열: 같은 카테고리 연속 행을 하나의 박스로 묶음
    let i = 0;
    while (i < rows.length) {
      const cat = rows[i][0]; let j = i;
      while (j < rows.length && rows[j][0] === cat) j++;
      const yA = y0+hH+i*RH, h = (j-i)*RH;
      s.addShape(pptx.ShapeType.rect, { x:catX, y:yA, w:catW, h, fill:{color: CC[cat] || C.white}, line:{color:C.white,width:1} });
      s.addText(cat, { x:catX, y:yA, w:catW, h, margin:0, valign:"middle", align:"center", fontFace:F.semi, fontSize: (j-i)>=2?7.5:5.6, color:C.ink });
      i = j;
    }
  }
  grid(M, 6.05, RC.slice(0,48), 1.5);   // 좌: unary+activation+binary (48)
  grid(M+6.28, 6.05, RC.slice(48), 1.5); // 우: mn~meta (49)
  let lx = M;
  const leg = [["N",C.ok,C.okBg,"npu"],["H",C.warn,C.warnBg,"host (CPU fallback)"],["C",C.err,C.errBg,"compile_fail (단독그래프→실연산 npu)"],["—",C.mut,C.bg2,"미정의 (torch에 op·dtype 조합 없음)"]];
  leg.forEach(([g,fg,bg,t]) => {
    s.addShape(pptx.ShapeType.rect, { x:lx, y:6.98, w:0.2, h:0.2, fill:{color:bg}, line:{color:fg,width:0.6} });
    s.addText(g, { x:lx, y:6.98, w:0.2, h:0.2, margin:0, align:"center", valign:"middle", fontFace:F.semi, fontSize:7, color:fg });
    s.addText(t, { x:lx+0.25, y:6.96, w:3.4, h:0.24, margin:0, valign:"middle", fontFace:F.reg, fontSize:8.5, color:C.ink2 });
    lx += 0.25 + t.length*0.082 + 0.42;
  });
  srcline(s, "근거: info/op_verify/dtype_full_*.py + op_categories.json. op는 14개 카테고리로 정렬(슬라이드11). 카테고리 출처: furiosa native_runtime.so IR op enum.");
})();

/* ===================== 11. op 카테고리 분류 표 ===================== */
(() => {
  const s = pptx.addSlide(); s.background = { color: C.white };
  chapter(s, "op 카테고리 분류 · furiosa 컴파일러 IR 기준");
  title(s, "SUPPORTED 97개 → 14개 카테고리",
    "카테고리 출처: native_runtime.so 의 op IR enum (activation·conv·index·matmul·meta·norm·reduction·resize·shape·pad + Unary/Binary/SymExpr::Ternary/Reduce/AttentionKernel). attn·norm·resize 는 SUPPORTED 97개에 해당 op 없음. mn = min/max.");

  const CC = { unary:"dbeafe", activation:"ccfbf1", binary:"ede9fe", mn:"e0e7ff", matmul:"fee2e2", conv:"ffedd5", reduction:"dcfce7", pool:"ecfccb", shape:"e2e8f0", index:"fce7f3", ternary:"f0e6d2", creation:"fef9c3", pad:"fae8ff", meta:"e5e7eb" };
  const fs = 9;
  const head = ["카테고리","수","대표 op","dtype 실행 경향 (npu / host)"].map((h,i) => ({
    text: h, options: { fontFace:F.semi, fontSize:fs, color:C.white, fill:{color:C.blue}, align:i===1?"center":"left", valign:"middle", margin:[1,5,1,6] },
  }));
  const data = [
    ["unary",14,"abs·cos·exp·log·sqrt·rsqrt·neg·sin·erf·reciprocal·clamp·pow·isnan·logical_not","f32/bf16 npu · fp16·int16 일부 host · isnan 전부 host"],
    ["activation",6,"relu·leaky_relu·sigmoid·tanh·_softmax·_log_softmax","f32/bf16 npu · fp16서 relu host"],
    ["binary",28,"add·sub·mul·div(.Scalar/.Tensor)·비교 eq~ge·logical_and/xor·bitwise_*","f32/bf16 npu · bitwise=int전용 · 비교 .Scalar는 int8/16 host"],
    ["mn",2,"maximum·minimum","f32/bf16/int npu · float64 host"],
    ["matmul",2,"mm·bmm","f32/bf16/int npu(감소정밀도 ~0.23%) · float64 host"],
    ["conv",1,"convolution","f32/bf16/int8 npu · fp16·int16·int32/64 host"],
    ["reduction",9,"sum·mean·amax·max·argmax·any·var_mean·topk·cumsum","f32/bf16 npu · cumsum=int32/64만 npu · max/argmax int16/8 host"],
    ["pool",3,"avg_pool2d·max_pool2d_with_indices·_adaptive_avg_pool2d","대부분 host (NPU 직접 거의 안 됨) · max_pool 전 dtype host"],
    ["shape",18,"view·permute·transpose·squeeze·unsqueeze·expand·cat·slice·slice_scatter·split","f32/int npu · expand/slice/split/copy_ 단독 C → 실연산 npu"],
    ["index",5,"index.Tensor·index_put·index_select·gather·scatter.src","index_put·scatter npu · index/index_select/gather 전 dtype host"],
    ["ternary",1,"where.self","f32/bf16/int npu · int16/8 host"],
    ["creation",3,"full·full_like·fill.Scalar","f32/int32 npu · fp16/int16 일부 host"],
    ["pad",1,"constant_pad_nd","전 dtype host (NPU EDF 미지원)"],
    ["meta",4,"clone·copy·copy_·_to_copy","clone·_to_copy npu · copy·copy_ 단독 C → 실연산 npu"],
  ];
  const body = data.map(([cat,n,ops,trend]) => ([
    { text: cat, options: { fontFace:F.semi, fontSize:fs, color:C.ink, align:"left", valign:"middle", fill:{color:CC[cat]}, margin:[1,5,1,6] } },
    { text: String(n), options: { fontFace:F.semi, fontSize:fs, color:C.ink2, align:"center", valign:"middle", fill:{color:C.white} } },
    { text: ops, options: { fontFace:F.reg, fontSize:fs-0.5, color:C.ink2, align:"left", valign:"middle", fill:{color:C.white}, margin:[1,5,1,6] } },
    { text: trend, options: { fontFace:F.reg, fontSize:fs-0.5, color:C.ink2, align:"left", valign:"middle", fill:{color:C.white}, margin:[1,5,1,6] } },
  ]));
  s.addTable([head, ...body], {
    x: M, y: 2.0, w: CW, colW: [1.2, 0.5, 5.55, CW-7.25],
    border: { type:"solid", color:C.border2, pt:0.5 }, rowH: 0.34, valign:"middle",
  });
  s.addText([
    { text: "미사용 카테고리(SUPPORTED 97개에 op 없음): ", options: { fontFace:F.semi, fontSize:10, color:C.ink } },
    { text: "attn (attention 전용 커널) · norm (batch/layer norm → 분해됨) · resize (interpolate/upsample 미포함).", options: { fontFace:F.reg, fontSize:10, color:C.ink2 } },
  ], { x: M, y: 6.94, w: CW, h: 0.3, margin: 0, valign: "middle" });
  srcline(s, "근거: furiosa native_runtime.so IR op enum (strings 추출) + op 의미. 분류·매핑 info/op_verify/op_categories.json. dtype 경향은 §7-1(970칸 실측) 요약.");
})();

pptx.writeFile({ fileName: "RNGD_Op_Support.pptx" }).then((f) => console.log("WROTE", f));
