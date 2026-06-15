/* RNGD 모델 벤치마크 핵심 결과 + 빌드 artifact 목록 — 2장 (16:9)
 * 출처: rngd-npu/REPORT.md §1(모델별 핵심 지표) · rngd-npu/artifacts/ 실측(du + params 정밀도 + binary_bundle 유무)
 * 스타일: build_14b_diff.js (Paperlogy) 준수. NPU=파랑. */
const pptx = new (require("pptxgenjs"))();
pptx.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pptx.layout = "W";
pptx.author = "RNGD Benchmark";

const F = {
  black: "Paperlogy 9 Black", xbold: "Paperlogy 8 ExtraBold",
  bold: "Paperlogy 7 Bold", semi: "Paperlogy 6 SemiBold",
  med: "Paperlogy 5 Medium", reg: "Paperlogy 4 Regular",
};
const C = {
  ink: "222222", ink2: "45515e", mut: "8e8e93",
  blue: "1456f0", blueBg: "eef4ff", blue2: "3b82f6",
  white: "ffffff", border2: "e5e7eb", bg2: "f0f0f0", bg3: "f7f8fa",
  ok: "16a34a", okBg: "e8ffea", warn: "b45309", amberBg: "fef3c7",
  grey: "6b7280", greyBg: "eceef1",
};
const M = 0.5, CW = 13.333 - 2 * M;

function eyebrow(s, txt) {
  s.addText(txt, { x: M, y: 0.32, w: CW, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 12, color: C.mut, charSpacing: 0.6 });
}
function head(s, t, y) {
  s.addText(t, { x: M, y: y || 0.62, w: CW, h: 0.6, margin: 0, fontFace: F.bold, fontSize: 27, color: C.ink, charSpacing: -0.6 });
}
function source(s, txt) {
  s.addText(txt, { x: M, y: 7.04, w: CW, h: 0.34, margin: 0, fontFace: F.reg, fontSize: 8.3, color: C.mut, lineSpacingMultiple: 1.08 });
}

/* ============ Slide 1 — 모델 벤치마크 핵심 결과 ============ */
(() => {
  const s = pptx.addSlide();
  s.background = { color: C.white };
  eyebrow(s, "FURIOSA RNGD · furiosa-llm · 코드 생성 모델 벤치마크 (prompt_len=1024)");
  head(s, "RNGD 모델 벤치마크 — 핵심 결과");
  s.addText([
    { text: "단일 스트림 디코드 속도와 동시성 ", options: { fontFace: F.med, fontSize: 13, color: C.ink2 } },
    { text: "peak 합산 처리량", options: { fontFace: F.bold, fontSize: 13, color: C.blue } },
    { text: " 요약 · peak 합산 TPS@c = 가장 높은 총처리량과 그때의 동시 요청 수 · 실패 0 기준", options: { fontFace: F.med, fontSize: 13, color: C.ink2 } },
  ], { x: M, y: 1.22, w: CW, h: 0.4, margin: 0, lineSpacingMultiple: 1.2 });

  const HFS = 12, FS = 12, mg = [3, 5, 3, 7];
  const h = (t, al) => ({ text: t, options: { fill: { color: C.blue }, color: C.white, fontFace: F.semi, fontSize: HFS, align: al || "center", valign: "middle", margin: mg } });
  // 일반 행 셀
  const m0 = (t, bold) => ({ text: t, options: { fontFace: bold ? F.bold : F.semi, fontSize: FS, color: bold ? C.blue : C.ink, align: "left", valign: "middle", fill: { color: bold ? C.blueBg : C.white }, margin: mg } });
  const cc = (t, fill, col) => ({ text: t, options: { fontFace: F.med, fontSize: FS, color: col || C.ink2, align: "center", valign: "middle", fill: { color: fill || C.white }, margin: mg } });
  const pk = (t, fill, col) => ({ text: t, options: { fontFace: F.bold, fontSize: FS, color: col || C.ink, align: "center", valign: "middle", fill: { color: fill || C.white }, margin: mg } });
  // 흐린 행(동시성 한계)
  const g0 = (t) => ({ text: t, options: { fontFace: F.semi, fontSize: FS, color: C.grey, align: "left", valign: "middle", fill: { color: C.bg3 }, margin: mg } });
  const gc = (t) => ({ text: t, options: { fontFace: F.med, fontSize: FS, color: C.mut, align: "center", valign: "middle", fill: { color: C.bg3 }, margin: mg } });

  // [모델, 카드, TTFT, 단일TPS, peak합산TPS@c, 효율배치]
  const rows = [
    [h("모델"), h("카드"), h("TTFT p50(s)"), h("단일 TPS"), h("peak 합산 TPS @동시성"), h("효율 배치")],
    [m0("Qwen2.5-0.5B Instruct"), cc("1장"), cc("0.031"), cc("84.5"), pk("4,120  @c128", C.blueBg, C.blue), cc("c128")],
    [m0("Qwen2.5-Coder-1.5B  ★종합1위", true), cc("1장"), cc("0.020"), pk("95.5", null, C.blue), pk("3,443  @c256", C.blueBg, C.blue), cc("c256")],
    [m0("Qwen2.5-Coder-7B Instruct"), cc("1장"), cc("0.032"), cc("50.3"), pk("2,225  @c256"), cc("c256")],
    [m0("Llama-3.1-8B Instruct"), cc("1장"), cc("0.033"), cc("54.5"), pk("2,192  @c128"), cc("c128")],
    [m0("Qwen2.5-Coder-14B Instruct"), cc("1장"), cc("0.050"), cc("30.7"), pk("1,074  @c256"), cc("c128")],
    [m0("EXAONE-4.0-32B  FP8"), cc("4장"), cc("0.221"), cc("30.4"), pk("809  @c256"), cc("c256")],
    [m0("Llama-3.3-70B Instruct"), cc("4장"), cc("0.247"), cc("24.5"), pk("383  @c128"), cc("c128")],
    [g0("Qwen3-32B  FP8"), gc("4장"), gc("—"), gc("—"), gc("25  @c1"), gc("c1")],
    [g0("Qwen3-32B  FP8"), gc("1장"), gc("—"), gc("—"), gc("5  @c1"), gc("c1")],
  ];
  s.addTable(rows, {
    x: M, y: 1.9, w: CW, colW: [4.0, 0.85, 1.55, 1.35, 2.93, 1.65],
    rowH: [0.42, 0.44, 0.46, 0.44, 0.44, 0.44, 0.44, 0.44, 0.42, 0.42], valign: "middle",
    border: { type: "solid", color: C.border2, pt: 1 },
  });

  // 인사이트 스트립
  const sy = 6.42;
  s.addShape(pptx.ShapeType.roundRect, { x: M, y: sy, w: CW, h: 0.5, rectRadius: 0.06, fill: { color: C.bg2 }, line: { type: "none" } });
  s.addText([
    { text: "핵심   ", options: { fontFace: F.bold, fontSize: 11, color: C.blue } },
    { text: "종합 1위 Qwen2.5-Coder-1.5B(1장) — 동시성 256까지 무손실 3,443 tok/s.  ", options: { fontFace: F.med, fontSize: 11, color: C.ink2 } },
    { text: "Qwen3-32B FP8(흐림)은 동시성↑ 시 실패 급증 → 1·4장 모두 c1만 안정", options: { fontFace: F.med, fontSize: 11, color: C.grey } },
  ], { x: M + 0.18, y: sy, w: CW - 0.36, h: 0.5, margin: 0, valign: "middle", lineSpacingMultiple: 1.05 });

  source(s, "출처: rngd-npu/REPORT.md §1(모델별 핵심 지표) · 측정 prompt_len=1024 · 단일 TPS=동시 1요청 디코드 속도, peak 합산 TPS@c=최대 총처리량과 그 동시성 · Qwen2.5-0.5B/Llama-3.1-8B는 furiosa-ai HF prebuilt");
})();

/* ============ Slide 2 — 빌드한 artifact 목록 ============ */
(() => {
  const s = pptx.addSlide();
  s.background = { color: C.white };
  eyebrow(s, "FURIOSA RNGD · furiosa-llm 컴파일 산출물 · rngd-npu/artifacts/");
  head(s, "빌드한 artifact 목록");
  s.addText([
    { text: "로컬 빌드 ", options: { fontFace: F.bold, fontSize: 13, color: C.blue } },
    { text: "7개 + ", options: { fontFace: F.med, fontSize: 13, color: C.ink2 } },
    { text: "prebuilt(HF) ", options: { fontFace: F.bold, fontSize: 13, color: C.grey } },
    { text: "3개 = 총 10개 · 약 426 GB · 정밀도는 params 파일명(W8f=FP8 / W16=bf16)으로 확정", options: { fontFace: F.med, fontSize: 13, color: C.ink2 } },
  ], { x: M, y: 1.22, w: CW, h: 0.4, margin: 0, lineSpacingMultiple: 1.2 });

  const HFS = 11.5, FS = 11.5, mg = [3, 5, 3, 7];
  const h = (t, al) => ({ text: t, options: { fill: { color: C.blue }, color: C.white, fontFace: F.semi, fontSize: HFS, align: al || "center", valign: "middle", margin: mg } });
  const af = (t) => ({ text: t, options: { fontFace: F.reg, fontSize: 9.8, color: C.ink, align: "left", valign: "middle", fill: { color: C.white }, margin: mg } });
  const md = (t) => ({ text: t, options: { fontFace: F.med, fontSize: FS, color: C.ink2, align: "left", valign: "middle", fill: { color: C.white }, margin: mg } });
  const cc = (t) => ({ text: t, options: { fontFace: F.med, fontSize: FS, color: C.ink2, align: "center", valign: "middle", fill: { color: C.white }, margin: mg } });
  const sz = (t) => ({ text: t, options: { fontFace: F.bold, fontSize: FS, color: C.ink, align: "right", valign: "middle", fill: { color: C.white }, margin: [3, 9, 3, 5] } });
  const tLocal = () => ({ text: "로컬빌드", options: { fontFace: F.semi, fontSize: 10.5, color: C.blue, align: "center", valign: "middle", fill: { color: C.blueBg }, margin: mg } });
  const tPre = () => ({ text: "prebuilt", options: { fontFace: F.semi, fontSize: 10.5, color: C.grey, align: "center", valign: "middle", fill: { color: C.greyBg }, margin: mg } });

  const rows = [
    [h("아티팩트 (디렉토리)", "left"), h("모델 · 정밀도 · 컨텍스트", "left"), h("TP · 카드"), h("크기"), h("빌드 유형")],
    [af("qwen2.5-coder-7b-inst-tp8"), md("Qwen2.5-Coder-7B Instruct · bf16"), cc("tp8 · 1장"), sz("15 GB"), tLocal()],
    [af("qwen2.5-coder-14b-inst-tp8"), md("Qwen2.5-Coder-14B Instruct · bf16"), cc("tp8 · 1장"), sz("28 GB"), tLocal()],
    [af("qwen2.5-coder-14b-tp8"), md("Qwen2.5-Coder-14B base · bf16"), cc("tp8 · 1장"), sz("28 GB"), tLocal()],
    [af("qwen3-32b-fp8-tp8"), md("Qwen3-32B · FP8"), cc("tp8 · 1장"), sz("33 GB"), tLocal()],
    [af("qwen3-32b-fp8-tp8-16k"), md("Qwen3-32B · FP8 · 16k ctx"), cc("tp8 · 1장"), sz("33 GB"), tLocal()],
    [af("qwen3-coder-30b-a3b-inst-fp8-tp8-65k-tc"), md("Qwen3-Coder-30B-A3B Inst · FP8 · 65k"), cc("tp8 · 1장"), sz("30 GB"), tLocal()],
    [af("qwen3-coder-30b-a3b-inst-tp8-65k-tc"), md("Qwen3-Coder-30B-A3B Inst · bf16 · 65k"), cc("tp8 · 1장"), sz("58 GB"), tLocal()],
    [af("exaone-4.0-32b-fp8-tp32"), md("EXAONE-4.0-32B · FP8"), cc("tp32 · 4장"), sz("34 GB"), tPre()],
    [af("qwen3-32b-fp8-tp32"), md("Qwen3-32B · FP8"), cc("tp32 · 4장"), sz("34 GB"), tPre()],
    [af("llama-3.3-70b-inst-tp32"), md("Llama-3.3-70B Instruct · bf16"), cc("tp32 · 4장"), sz("133 GB"), tPre()],
  ];
  s.addTable(rows, {
    x: M, y: 1.9, w: CW, colW: [4.5, 3.45, 1.5, 1.05, 1.83],
    rowH: [0.42, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], valign: "middle",
    border: { type: "solid", color: C.border2, pt: 1 },
  });

  // 미완 스트립
  const sy = 6.5;
  s.addShape(pptx.ShapeType.roundRect, { x: M, y: sy, w: CW, h: 0.44, rectRadius: 0.06, fill: { color: C.amberBg }, line: { type: "none" } });
  s.addText([
    { text: "미완 2개 (제외)   ", options: { fontFace: F.bold, fontSize: 10.5, color: C.warn } },
    { text: "qwen3-coder-30b-a3b-inst-fp8-tp8-65k (18 MB) · qwen3-coder-30b-a3b-inst-tp8-65k (16 MB) — artifact.json만 있고 빌드 산출물 0 → 위 -tc 변형이 완성본",
      options: { fontFace: F.med, fontSize: 10, color: C.ink2 } },
  ], { x: M + 0.18, y: sy, w: CW - 0.36, h: 0.44, margin: 0, valign: "middle", lineSpacingMultiple: 1.0 });

  source(s, "출처: rngd-npu/artifacts/ 실측 · 크기=du -sh · 정밀도=params-*.safetensors 파일명(W8f→FP8, W16→bf16) · 빌드유형=binary_bundle.zip(로컬빌드)/blobs·snapshots(HF prebuilt) · 합계 로컬 225 GB + prebuilt 201 GB ≈ 426 GB");
})();

pptx.writeFile({ fileName: "RNGD_Bench_Artifacts.pptx" }).then((f) => console.log("WROTE", f));
