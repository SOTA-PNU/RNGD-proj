/* Qwen2.5-Coder-14B base vs instruct — 차이점 1장 (16:9)
 * 숫자 출처: results/Qwen2.5-Coder-14B-tp8 (base, 2026-06-09) vs
 *           results/Qwen2.5-Coder-14B-inst-tp8 (instruct, 2026-06-04), 실측 + 6-에이전트 교차검증.
 * 스타일: build_serve_cli.js (Paperlogy) 준수. */
const pptx = new (require("pptxgenjs"))();
pptx.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pptx.layout = "W";
pptx.author = "RNGD 14B base vs instruct";

const F = {
  black: "Paperlogy 9 Black", xbold: "Paperlogy 8 ExtraBold",
  bold: "Paperlogy 7 Bold", semi: "Paperlogy 6 SemiBold",
  med: "Paperlogy 5 Medium", reg: "Paperlogy 4 Regular",
};
const C = {
  ink: "222222", ink2: "45515e", mut: "8e8e93",
  blue: "1456f0", blueBg: "eef4ff",
  white: "ffffff", border2: "e5e7eb", bg2: "f0f0f0",
  ok: "16a34a", warn: "b45309", warnBg: "fff7 e6".replace(" ", ""), amberBg: "fef3c7",
  vl: "7c3aed",
};
const M = 0.5, CW = 13.333 - 2 * M;

(() => {
  const s = pptx.addSlide();
  s.background = { color: C.white };

  // eyebrow
  s.addText("QWEN2.5-CODER-14B · base vs instruct · RNGD NPU (tp8, 1장)", {
    x: M, y: 0.32, w: CW, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 12, color: C.mut, charSpacing: 0.6,
  });
  // title
  s.addText("base 와 instruct, 무엇이 다른가", {
    x: M, y: 0.62, w: CW, h: 0.6, margin: 0, fontFace: F.bold, fontSize: 27, color: C.ink, charSpacing: -0.6,
  });
  // subtitle (근본원인 한 줄)
  s.addText([
    { text: "측정으로 확인된 차이만 정리. 근본 원인은 하나 — ", options: { fontFace: F.med, fontSize: 13, color: C.ink2 } },
    { text: "instruction-tuning 유무", options: { fontFace: F.bold, fontSize: 13, color: C.blue } },
    { text: ". 속도·메모리·연산량은 동일합니다.", options: { fontFace: F.med, fontSize: 13, color: C.ink2 } },
  ], { x: M, y: 1.24, w: CW, h: 0.42, margin: 0, lineSpacingMultiple: 1.2 });

  // ===== 차이점 표 =====
  const HFS = 12.5, FS = 12.5, CFS = 11;
  const hcell = (t, al) => ({ text: t, options: { fill: { color: C.blue }, color: C.white, fontFace: F.semi, fontSize: HFS, align: al || "left", valign: "middle", margin: [3, 6, 3, 8] } });
  const c0 = (t) => ({ text: t, options: { fontFace: F.semi, fontSize: FS, color: C.ink, align: "left", valign: "middle", fill: { color: C.white }, margin: [3, 6, 3, 8] } });
  const cb = (t) => ({ text: t, options: { fontFace: F.bold, fontSize: FS, color: C.ink, align: "center", valign: "middle", fill: { color: C.amberBg }, margin: [3, 4, 3, 4] } });
  const ci = (t) => ({ text: t, options: { fontFace: F.bold, fontSize: FS, color: C.blue, align: "center", valign: "middle", fill: { color: C.blueBg }, margin: [3, 4, 3, 4] } });
  const cw = (t) => ({ text: t, options: { fontFace: F.reg, fontSize: CFS, color: C.ink2, align: "left", valign: "middle", fill: { color: C.white }, margin: [3, 6, 3, 8] } });

  const rows = [
    [hcell("측정 항목"), hcell("base", "center"), hcell("instruct", "center"), hcell("원인 (간단)")],
    [c0("SWE-bench 패치 생성"), cb("0 개"), ci("40 개 (포맷 invalid)"), cw("instruct만 ‘패치를 만들라’는 지시를 수행")],
    [c0("chat 프롬프트 출력"), cb("0 토큰 · 즉시 종료"), ci("정상 생성 (gen 33.7s)"), cw("base는 chat/지시 포맷 미학습 → 입력 받자마자 EOS")],
    [c0("합성 프롬프트 출력"), cb("512 토큰 (안 멈춤)"), ci("~104 토큰 (EOS)"), cw("instruct만 ‘적절히 멈추기(EOS)’를 학습")],
    [c0("집계 throughput\n(동시성 256)"), cb("~1,800 tok/s"), ci("~1,410 tok/s"), cw("디코드 속도차 아님 — 짧은 출력이 배치를 일찍 비워 점유율↓")],
  ];

  s.addTable(rows, {
    x: M, y: 1.92, w: CW, colW: [2.85, 2.55, 2.75, 4.18],
    rowH: [0.42, 0.74, 0.74, 0.74, 0.86], valign: "middle",
    border: { type: "solid", color: C.border2, pt: 1 },
  });

  // ===== 공통(차이 없음) 스트립 =====
  const stripY = 6.18;
  s.addShape(pptx.ShapeType.roundRect, { x: M, y: stripY, w: CW, h: 0.62, rectRadius: 0.06, fill: { color: C.bg2 }, line: { type: "none" } });
  s.addText([
    { text: "동일 (차이 없음)   ", options: { fontFace: F.bold, fontSize: 11, color: C.ok } },
    { text: "모델 크기 bf16 ≈28 GB · tp8 1장(npu:0) · 단일스트림 토큰당 지연 ITL 0.0325 s · SWE-bench 해결률 0/50 · 연산 그래프(바이너리 크기 동일, md5만 상이)",
      options: { fontFace: F.med, fontSize: 11, color: C.ink2 } },
  ], { x: M + 0.18, y: stripY, w: CW - 0.36, h: 0.62, margin: 0, valign: "middle", lineSpacingMultiple: 1.1 });

  // 출처
  s.addText("출처: results/Qwen2.5-Coder-14B-tp8(base, 2026-06-09) vs results/Qwen2.5-Coder-14B-inst-tp8(instruct, 2026-06-04) · tps/sweep/memsweep/swebench 실측 · 6-에이전트 교차검증(숫자 오류 0건)", {
    x: M, y: 7.06, w: CW, h: 0.3, margin: 0, fontFace: F.reg, fontSize: 8.5, color: C.mut,
  });
})();

pptx.writeFile({ fileName: "RNGD_14B_base_vs_inst.pptx" }).then((f) => console.log("WROTE", f));
