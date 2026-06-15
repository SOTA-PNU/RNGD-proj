/* Furiosa SW 스택 · 모델 프레임워크 (최상단 계층) — RNGD가 실행하는 모델 ↔ furiosa.models 클래스 (16:9, 1장)
 * 기존 build_model_prep.js (Brandlogy/Paperlogy) 스타일 준수.
 * 모든 내용은 SDK 소스(~/furiosa/lib/python3.12/site-packages/)를 직접 읽고
 * `import furiosa.models` 로 클래스 11종 실재를 확인한 사실 기반.
 * 연결 규칙: furiosa-llm 은 HF config 의 architectures 이름과 "똑같은 이름"의 클래스를
 *   furiosa.models 에서 getattr 로 찾아 NPU용으로 실행한다 (optimum/modeling.py:173). */
const pptx = new (require("pptxgenjs"))();
pptx.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pptx.layout = "W";
pptx.author = "RNGD Model Framework";

const F = {
  black: "Paperlogy 9 Black", xbold: "Paperlogy 8 ExtraBold",
  bold: "Paperlogy 7 Bold", semi: "Paperlogy 6 SemiBold",
  med: "Paperlogy 5 Medium", reg: "Paperlogy 4 Regular",
};
const C = {
  ink: "222222", ink2: "45515e", mut: "8e8e93",
  blue: "1456f0", blue2: "3b82f6", blue3: "60a5fa", blueLt: "bfdbfe", blueBg: "eef4ff",
  white: "ffffff", border: "f2f3f5", border2: "e5e7eb", bg2: "f0f0f0",
  dark: "181e25", codeTx: "e5e9ef", codeMut: "8ea0b5", codeAc: "5fc6ff", codeRed: "ff8087",
  ok: "16a34a", okBg: "e8ffea", err: "dc2626", warn: "d97706", warnBg: "fef3c7",
  vl: "7c3aed", vlBg: "f3e8ff",
};
const shStd = () => ({ type: "outer", color: "000000", opacity: 0.08, blur: 6, offset: 2, angle: 90 });
const M = 0.5, CW = 13.333 - 2 * M;

function title(s, head, sub) {
  s.addText(head, {
    x: M, y: 0.66, w: CW, h: 0.7, margin: 0,
    fontFace: F.bold, fontSize: 29, color: C.ink, charSpacing: -0.6, lineSpacingMultiple: 1.14,
  });
  if (sub) s.addText(sub, {
    x: M, y: 1.36, w: CW, h: 0.62, margin: 0,
    fontFace: F.med, fontSize: 13.5, color: C.ink2, lineSpacingMultiple: 1.36,
  });
}
function card(s, x, y, w, h, opt = {}) {
  s.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h, rectRadius: opt.r || 0.1,
    fill: { color: opt.fill || C.white },
    line: opt.line === null ? { type: "none" } : { color: opt.line || C.border2, width: 1 },
    shadow: opt.shadow,
  });
}
function accent(s, x, y, h, color) {
  s.addShape(pptx.ShapeType.roundRect, { x, y, w: 0.07, h, rectRadius: 0.03, fill: { color }, line: { type: "none" } });
}
function chip(s, x, y, text, fill, txt, fs) {
  const w = 0.34 + text.length * 0.092;
  s.addShape(pptx.ShapeType.roundRect, { x, y, w, h: 0.34, rectRadius: 0.17, fill: { color: fill }, line: { type: "none" } });
  s.addText(text, { x, y, w, h: 0.34, margin: 0, align: "center", valign: "middle", fontFace: F.semi, fontSize: fs || 10, color: txt });
  return w;
}

/* ===================== 모델 프레임워크 (1장) ===================== */
(() => {
  const s = pptx.addSlide();
  s.background = { color: C.white };

  // 상단 챕터 + 출처
  s.addText("FURIOSA 소프트웨어 스택 · 모델 프레임워크 (최상단 계층)", {
    x: M, y: 0.34, w: 11, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 12, color: C.mut, charSpacing: 0.6,
  });

  title(
    s,
    "모델 프레임워크 — RNGD가 실행하는 모델 ↔ furiosa.models 클래스",
    "Furiosa SW 스택의 최상단. furiosa-llm 은 HF config 의 architectures 이름과 똑같은 이름의 클래스를 furiosa.models 에서 찾아(getattr) NPU용으로 실행합니다."
  );

  // 연결 규칙 (가벼운 카드 + 코드 강조)
  const ry = 2.08, rh = 0.86;
  card(s, M, ry, CW, rh, { fill: C.bg2, line: null, r: 0.1 });
  accent(s, M, ry + 0.2, rh - 0.4, C.blue);
  s.addText("연결 규칙", { x: M + 0.26, y: ry + 0.16, w: 2.2, h: 0.3, margin: 0, fontFace: F.bold, fontSize: 13, color: C.blue });
  s.addText([
    { text: "model.config.architectures[0]  ", options: { fontFace: F.semi, fontSize: 12.5, color: C.ink } },
    { text: "(예: ", options: { fontFace: F.reg, fontSize: 12.5, color: C.ink2 } },
    { text: "\"LlamaForCausalLM\"", options: { fontFace: F.semi, fontSize: 12.5, color: C.blue } },
    { text: ")   ⟶   ", options: { fontFace: F.reg, fontSize: 12.5, color: C.mut } },
    { text: "getattr(furiosa.models, name)", options: { fontFace: F.semi, fontSize: 12.5, color: C.ink } },
    { text: "   ⟶   ", options: { fontFace: F.reg, fontSize: 12.5, color: C.mut } },
    { text: "CausalModelServer", options: { fontFace: F.semi, fontSize: 12.5, color: C.ok } },
    { text: " → NPU 실행", options: { fontFace: F.reg, fontSize: 12.5, color: C.ink2 } },
  ], { x: M + 0.26, y: ry + 0.46, w: CW - 0.5, h: 0.4, margin: 0, valign: "middle" });

  // ===== 표: furiosa.models 클래스 ↔ 모델 (Dense / MoE / VL 그룹) =====
  const KIND = {
    Dense: { c: C.blue, fill: C.blueBg },
    MoE: { c: C.warn, fill: C.warnBg },
    VL: { c: C.vl, fill: C.vlBg },
  };
  // [클래스, model_type, 종류, 예시, canonical?]
  const data = [
    ["LlamaForCausalLM", "llama", "Dense", "Llama 3.1 8B·70B, 3.3 70B, DeepSeek-R1-Distill-Llama, CodeLlama", true],
    ["Qwen2ForCausalLM", "qwen2", "Dense", "Qwen2.5 0.5~32B, Qwen2.5-Coder 1.5~32B, DeepSeek-R1-Distill-Qwen, QwQ-32B", true],
    ["Qwen3ForCausalLM", "qwen3", "Dense", "Qwen3 4B·8B·32B(FP8), Qwen3-Embedding·Reranker", true],
    ["Exaone4ForCausalLM", "exaone4", "Dense", "EXAONE-4.0-32B", true],
    ["ExaoneForCausalLM", "exaone", "Dense", "EXAONE 3.0·3.5·Deep", false],
    ["MistralForCausalLM", "mistral", "Dense", "Mistral 계열 (preset 없음 → 버킷 수동)", false],
    ["Phi3ForCausalLM", "phi3", "Dense", "Phi-3 계열 (preset 없음 → 버킷 수동)", false],
    ["Qwen3MoeForCausalLM", "qwen3_moe", "MoE", "Qwen3-30B-A3B-Instruct, Qwen3-Coder-30B-A3B", true],
    ["GptOssForCausalLM", "gpt_oss", "MoE", "gpt-oss 20B·120B", true],
    ["ExaoneMoeForCausalLM", "exaone_moe", "MoE", "EXAONE MoE 계열", false],
    ["Qwen3VLForConditionalGeneration", "qwen3_vl", "VL", "Qwen3-VL (이미지+텍스트 멀티모달)", false],
  ];

  const fs = 10.5;
  const header = ["furiosa.models 클래스", "model_type", "종류", "RNGD에서 쓰는 모델 (예시)"].map((h, i) => ({
    text: h, options: { fontFace: F.semi, fontSize: fs, color: C.white, fill: { color: C.blue }, align: i === 3 ? "left" : (i === 0 ? "left" : "center"), valign: "middle" },
  }));
  const body = data.map(([cls, mt, kind, ex, canon]) => {
    const k = KIND[kind];
    const clsText = canon
      ? [{ text: "★ ", options: { fontFace: F.semi, fontSize: fs, color: C.blue } }, { text: cls, options: { fontFace: F.semi, fontSize: fs, color: C.ink } }]
      : [{ text: cls, options: { fontFace: F.semi, fontSize: fs, color: C.ink } }];
    return [
      { text: clsText, options: { align: "left", valign: "middle", fill: { color: C.white } } },
      { text: mt, options: { fontFace: F.reg, fontSize: fs, color: C.ink2, align: "center", valign: "middle", fill: { color: C.white } } },
      { text: kind, options: { fontFace: F.semi, fontSize: fs, color: k.c, align: "center", valign: "middle", fill: { color: k.fill } } },
      { text: ex, options: { fontFace: F.reg, fontSize: fs, color: C.ink2, align: "left", valign: "middle", fill: { color: C.white } } },
    ];
  });
  s.addTable([header, ...body], {
    x: M, y: 3.10, w: CW, colW: [3.05, 1.25, 0.95, CW - 5.25],
    border: { type: "solid", color: C.border2, pt: 0.5 },
    rowH: 0.29, margin: [2, 5, 2, 5], valign: "middle",
  });

  // 범례 + 보조 설명
  s.addText([
    { text: "★ ", options: { fontFace: F.semi, fontSize: 10.5, color: C.blue } },
    { text: "= furiosa-llm 공식 지원 집합(CANONICAL_MODEL_IDS)    ·    ", options: { fontFace: F.reg, fontSize: 10.5, color: C.ink2 } },
    { text: "Dense ", options: { fontFace: F.semi, fontSize: 10.5, color: C.blue } },
    { text: "전 파라미터 사용   ", options: { fontFace: F.reg, fontSize: 10.5, color: C.ink2 } },
    { text: "MoE ", options: { fontFace: F.semi, fontSize: 10.5, color: C.warn } },
    { text: "전문가 일부만 활성   ", options: { fontFace: F.reg, fontSize: 10.5, color: C.ink2 } },
    { text: "VL ", options: { fontFace: F.semi, fontSize: 10.5, color: C.vl } },
    { text: "비전-언어    ·    임베딩·리랭킹은 Qwen3 를 pooling/score 태스크로 사용", options: { fontFace: F.reg, fontSize: 10.5, color: C.ink2 } },
  ], { x: M, y: 6.72, w: CW, h: 0.3, margin: 0, valign: "middle" });

  // 출처
  s.addText("출처: furiosa_llm/optimum/modeling.py:138(CANONICAL_MODEL_IDS)·:173(get_models_lang_class)·:431(AutoModelForCausalLM) · furiosa.models.language.architecture.* · `import furiosa.models` 로 클래스 11종 실재 확인 (furiosa-llm 2026.2.0)", {
    x: M, y: 7.13, w: CW, h: 0.28, margin: 0, fontFace: F.reg, fontSize: 8.5, color: C.mut,
  });
})();

pptx.writeFile({ fileName: "RNGD_Model_Framework.pptx" }).then((f) => console.log("WROTE", f));
