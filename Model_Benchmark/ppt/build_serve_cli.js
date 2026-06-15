/* furiosa-llm serve 옵션 레퍼런스 (16:9, 1장)
 * 출처: `furiosa-llm serve --help` (furiosa-llm 2026.2.0) 실제 출력.
 * 기존 build_model_framework.js / build_model_prep.js (Paperlogy) 스타일 준수. */
const pptx = new (require("pptxgenjs"))();
pptx.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pptx.layout = "W";
pptx.author = "RNGD serve CLI";

const F = {
  black: "Paperlogy 9 Black", xbold: "Paperlogy 8 ExtraBold",
  bold: "Paperlogy 7 Bold", semi: "Paperlogy 6 SemiBold",
  med: "Paperlogy 5 Medium", reg: "Paperlogy 4 Regular",
};
const C = {
  ink: "222222", ink2: "45515e", mut: "8e8e93",
  blue: "1456f0", blue2: "3b82f6", blueBg: "eef4ff",
  white: "ffffff", border2: "e5e7eb", bg2: "f0f0f0",
  ok: "16a34a", warn: "d97706", warnBg: "fef3c7", err: "dc2626",
  vl: "7c3aed",
};
const M = 0.5, CW = 13.333 - 2 * M;

function title(s, head, sub) {
  s.addText(head, {
    x: M, y: 0.62, w: CW, h: 0.6, margin: 0,
    fontFace: F.bold, fontSize: 27, color: C.ink, charSpacing: -0.6, lineSpacingMultiple: 1.1,
  });
  if (sub) s.addText(sub, {
    x: M, y: 1.22, w: CW, h: 0.42, margin: 0,
    fontFace: F.med, fontSize: 12.5, color: C.ink2, lineSpacingMultiple: 1.3,
  });
}

/* ===================== serve 옵션 표 (1장) ===================== */
(() => {
  const s = pptx.addSlide();
  s.background = { color: C.white };

  s.addText("FURIOSA-LLM CLI · serve (OpenAI 호환 서버 실행)", {
    x: M, y: 0.32, w: 11, h: 0.3, margin: 0, fontFace: F.semi, fontSize: 12, color: C.mut, charSpacing: 0.6,
  });
  title(
    s,
    "furiosa-llm serve 옵션 한눈에",
    "엔진을 띄우는 유일한 CLI. 위치인자 model(HF id 또는 아티팩트 경로) + 옵션들. 기본값은 (기본 …) 으로 표기."
  );

  // 사용법 코드 라인
  s.addShape(pptx.ShapeType.roundRect, { x: M, y: 1.74, w: CW, h: 0.36, rectRadius: 0.06, fill: { color: C.bg2 }, line: { type: "none" } });
  s.addText([
    { text: "$ furiosa-llm serve ", options: { fontFace: F.semi, fontSize: 11, color: C.blue } },
    { text: "<model>", options: { fontFace: F.semi, fontSize: 11, color: C.ink } },
    { text: " [--port 8000] [-tp 8] [-dp 4] [--enable-prefix-caching] …", options: { fontFace: F.reg, fontSize: 11, color: C.ink2 } },
  ], { x: M + 0.18, y: 1.74, w: CW - 0.3, h: 0.36, margin: 0, valign: "middle" });

  const HFS = 8.6;   // group header fontSize
  const FS = 7.6;    // body fontSize
  const RH = 0.168;

  const hdr = (t) => [{
    text: t, options: { colspan: 2, fill: { color: C.blue }, color: C.white, fontFace: F.semi, fontSize: HFS, align: "left", valign: "middle", margin: [1, 4, 1, 5] },
  }];
  const row = (flag, desc) => [
    { text: flag, options: { fontFace: F.semi, fontSize: FS, color: C.ink, align: "left", valign: "middle", fill: { color: C.white }, margin: [1, 4, 1, 5] } },
    { text: desc, options: { fontFace: F.reg, fontSize: FS, color: C.ink2, align: "left", valign: "middle", fill: { color: C.white }, margin: [1, 4, 1, 5] } },
  ];

  // ---------- LEFT ----------
  const left = [
    hdr("기본 / 접속"),
    row("model  (위치인자)", "HF 모델 id 또는 Furiosa 아티팩트 경로 · 서버당 1개"),
    row("--revision", "HF 리비전(브랜치/태그/커밋). 기본 main · furiosa-ai 모델은 release 태그"),
    row("--served-model-name", "API에 노출할 모델 이름 (기본: model 인자)"),
    row("--host", "바인드 호스트 (기본 0.0.0.0)"),
    row("--port", "바인드 포트 (기본 8000)"),
    hdr("병렬화 / 디바이스"),
    row("-tp, --tensor-parallel-size", "텐서 병렬 수 (기본 4)"),
    row("-pp, --pipeline-parallel-size", "파이프라인 스테이지 수"),
    row("-dp, --data-parallel-size", "데이터 병렬 수 (미지정 시 가용 PE로 추론)"),
    row("--devices", "사용 디바이스. npu:0 / npu:0:0 / npu:0:0-3"),
    row("--data-parallel-routing-policy", "dp 라우팅: round_robin / prefix_aware"),
    hdr("버킷 / 길이 / 배치"),
    row("-pb, --prefill-buckets", "prefill 버킷 (기본: 아티팩트 값)"),
    row("-db, --decode-buckets", "decode 버킷 (기본: 아티팩트 값)"),
    row("--max-prompt-len", "캡처할 최대 프롬프트 길이 (초과 버킷 무시)"),
    row("--max-model-len", "최대 지원 시퀀스 길이 (초과 버킷 무시)"),
    row("--max/min-batch-size", "npu 요청당 최대/최소 배치 (벗어난 버킷 무시)"),
    hdr("채팅 / 툴 / 추론 파서"),
    row("--chat-template", "채팅 템플릿 파일로 덮어쓰기 (기본: tokenizer)"),
    row("--chat-template-content-format", "auto / string / openai (기본 auto)"),
    row("--enable-auto-tool-choice", "툴 자동 선택 (--tool-call-parser 필요)"),
    row("--tool-call-parser", "hermes / llama4_json / llama3_json / openai"),
    row("--reasoning-parser", "deepseek_r1 / exaone4 / qwen3 (추론 모델용)"),
    row("--response-role", "chat 응답 role (기본 assistant)"),
    row("--structured-outputs-backend", "구조화 출력 엔진: auto / guidance / xgrammar"),
  ];

  // ---------- RIGHT ----------
  const right = [
    hdr("스케줄러 / 성능"),
    row("--scheduler-kind", "특화 전략 스케줄러 선택 (기본: 범용)"),
    row("--npu-queue-limit", "NPU 큐 한도 override (기본: 아티팩트)"),
    row("--max-processing-samples", "최대 처리 샘플 override"),
    row("--spare-blocks-ratio", "여유 블록 비율 (기본 0.0 · ↑시 성능↑/OOM 위험)"),
    row("--estimation-time-limit-ms", "배칭 전략 추정 시간 한도 (ms)"),
    row("--enable-prefix-caching", "prefix 캐시 켜기/끄기"),
    row("--max-concurrency", "iter당 최대 동시 decode (↑ throughput, TTFT↑)"),
    row("--max-num-batched-tokens", "iter당 최대 배치 토큰 (↑ TPOT, TTFT↑)"),
    row("--prefix-cache-lookahead-requests", "prefix 캐시 lookahead 요청 수 (실험)"),
    row("--max-io-memory-mb", "I/O 텐서용 NPU 메모리 MB (0–49152). 작으면 OOM"),
    hdr("인증 / 로깅 / 보안"),
    row("--api-key", "Bearer 토큰 인증 키 (기본 None = 인증 끔)"),
    row("--allowed-origins", "허용 origin (json list, 기본 [\"*\"])"),
    row("--disable-log-stats", "통계 로깅 끄기"),
    row("--enable-payload-logging", "POST 페이로드 로깅 (민감정보 노출 위험)"),
    hdr("멀티모달 / Responses API"),
    row("--allowed-local-media-path", "로컬 미디어 접근 경로 (미설정 시 차단)"),
    row("--allowed-media-domains", "원격 미디어 허용 도메인 (SSRF 방어)"),
    row("--image/video-limit-per-prompt", "프롬프트당 이미지/비디오 개수 제한"),
    row("--interleave-mm-strings", "멀티모달 placeholder 원위치 삽입"),
    row("--enable-responses-api-store", "Responses API 인메모리 저장 (멀티턴)"),
    row("--responses-api-store-*", "저장 max-entries(기본 10000) · ttl(기본 3600s)"),
    hdr("uvicorn · JIT(실험) · v3"),
    row("--uvicorn-log-level", "debug~critical / trace (기본 info)"),
    row("--disable-uvicorn-access-log", "uvicorn 접근 로그 끄기"),
    row("--enable-jit-compilation", "JIT 컴파일 켜기 (실험)"),
    row("--jit-threshold / -max-workers / -unit-size", "트리거 5 / 워커 15 / 유닛 8"),
    row("--fxb", "FXB 파일·디렉터리 경로 → v3 엔진 사용"),
  ];

  const TY = 2.28;
  s.addTable(left, {
    x: M, y: TY, w: 6.02, colW: [2.18, 3.84],
    border: { type: "solid", color: C.border2, pt: 0.5 }, rowH: RH, valign: "middle",
  });
  s.addTable(right, {
    x: M + 6.18, y: TY, w: 6.13, colW: [2.42, 3.71],
    border: { type: "solid", color: C.border2, pt: 0.5 }, rowH: RH, valign: "middle",
  });

  // 출처
  s.addText("출처: `furiosa-llm serve --help` 실제 출력 (furiosa-llm 2026.2.0) · serve 진입 cli/serve.py:381 · 엔진 구동 server/serving_completions.py:44 (AsyncLLMEngine)", {
    x: M, y: 7.16, w: CW, h: 0.26, margin: 0, fontFace: F.reg, fontSize: 8.5, color: C.mut,
  });
})();

pptx.writeFile({ fileName: "RNGD_Serve_CLI.pptx" }).then((f) => console.log("WROTE", f));
