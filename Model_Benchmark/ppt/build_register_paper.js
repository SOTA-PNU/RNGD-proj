/* "Don't Merge the Registers" — 논문 발표용 덱 (16:9, Brandlogy/Paperlogy, Design.md 준수).
 * 발표 흐름: Motivation → Background(kNN·ViT·모델) → Related Work(ToMe/PiToMe) → 추론삽입 코드 →
 *   Problem → Method(개념·코드 n_protect 1→5) → Experiments → Results(라인·ablation) → Conclusion.
 * 수치 출처: rngd-npu/ACCV/논문_핵심정리.md · register_token_reduction/{results,ablation/results}. 코드: tome_reg.py. */
const pptx = new (require("pptxgenjs"))();
pptx.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pptx.layout = "W"; pptx.author = "Register Token Reduction";
const TOTAL = 14;
const F = { black:"Paperlogy 9 Black", xbold:"Paperlogy 8 ExtraBold", bold:"Paperlogy 7 Bold",
  semi:"Paperlogy 6 SemiBold", med:"Paperlogy 5 Medium", reg:"Paperlogy 4 Regular", mono:"Consolas" };
const C = { ink:"222222", ink2:"45515e", mut:"8e8e93",
  blue:"1456f0", blue2:"3b82f6", blue3:"60a5fa", blueLt:"bfdbfe", blueBg:"eef4ff",
  white:"ffffff", border2:"e5e7eb", bg2:"f4f5f7",
  dark:"181e25", gold:"e9a23b", goldBg:"fdf3e0", gray:"c9cdd6",
  ok:"2e7d32", okBg:"e8ffea", err:"c00000", errBg:"fdecec",
  codeTx:"e5e9ef", codeMut:"8ea0b5", codeAc:"5fc6ff", codeGold:"ffcf6b", codeGreen:"7ee2a8" };
const M = 0.5, CW = 13.333 - 2*M;
const shStd = () => ({ type:"outer", color:"000000", opacity:0.08, blur:6, offset:2, angle:90 });
function frame(s, chapter, page, source){ s.background = { color: C.white };
  s.addText(chapter.toUpperCase(), { x:M, y:0.4, w:9, h:0.3, margin:0, fontFace:F.semi, fontSize:12, color:C.mut, charSpacing:0.8 });
  s.addText(`${page} / ${TOTAL}`, { x:M, y:7.05, w:3, h:0.25, margin:0, fontFace:F.med, fontSize:10, color:C.mut });
  s.addText(source, { x:13.333-M-8, y:7.05, w:8, h:0.25, margin:0, fontFace:F.reg, fontSize:9.5, color:C.mut, align:"right" }); }
function title(s, head, sub){
  s.addText(head, { x:M, y:1.0, w:CW, h:0.66, margin:0, fontFace:F.bold, fontSize:28, color:C.ink, charSpacing:-0.5, lineSpacingMultiple:1.14 });
  if (sub) s.addText(sub, { x:M, y:1.72, w:CW, h:0.5, margin:0, fontFace:F.med, fontSize:14, color:C.ink2, lineSpacingMultiple:1.4 }); }
function card(s,x,y,w,h,opt={}){ s.addShape(pptx.ShapeType.roundRect,{ x,y,w,h,rectRadius:opt.r||0.1,
  fill:{color:opt.fill||C.white}, line:opt.line===null?{type:"none"}:{color:opt.line||C.border2,width:1}, shadow:opt.shadow }); }
function accent(s,x,y,h,color){ s.addShape(pptx.ShapeType.roundRect,{ x,y,w:0.07,h,rectRadius:0.03,fill:{color},line:{type:"none"} }); }
function chip(s,x,y,text,fill,txt,fs){ const w=0.32+text.length*0.098;
  s.addShape(pptx.ShapeType.roundRect,{ x,y,w,h:0.32,rectRadius:0.16,fill:{color:fill},line:{type:"none"} });
  s.addText(text,{ x,y,w,h:0.32,margin:0,align:"center",valign:"middle",fontFace:F.semi,fontSize:fs||10,color:txt }); return w; }
function tok(s,x,y,fill,star,prot){ const w=0.6;
  s.addShape(pptx.ShapeType.roundRect,{ x,y,w,h:w,rectRadius:0.08,fill:{color:fill}, line:{color:prot?C.ok:"888888",width:prot?2.6:1.1} });
  if(star) s.addText("★",{ x,y,w,h:w,margin:0,align:"center",valign:"middle",fontFace:F.bold,fontSize:14,color:C.white }); }
function tokRow(s,cx,y,cols,stars=[],prot=[]){ const w=0.6,gap=0.16,n=cols.length,x0=cx-(n*w+(n-1)*gap)/2;
  cols.forEach((c,i)=>tok(s,x0+i*(w+gap),y,c,stars.includes(i),prot.includes(i))); }
function arrowR(s,x,y,w,color){ s.addShape(pptx.ShapeType.rightArrow,{ x,y,w,h:0.32,fill:{color},line:{type:"none"} }); }
function arrowD(s,x,y,h,color){ s.addShape(pptx.ShapeType.downArrow,{ x,y,w:0.32,h,fill:{color},line:{type:"none"} }); }
function codeCard(s,x,y,w,h,label,lines,fs){
  s.addShape(pptx.ShapeType.roundRect,{ x,y,w,h,rectRadius:0.1,fill:{color:C.dark},line:{type:"none"},shadow:shStd() });
  let ty=y+0.16;
  if(label){ s.addText(label,{ x:x+0.24,y:ty,w:w-0.48,h:0.24,margin:0,fontFace:F.semi,fontSize:10,color:C.codeMut }); ty+=0.32; }
  s.addText(lines.map(ln=>({ text:ln.t, options:{ fontFace:F.mono, fontSize:fs||10.5, color:ln.c||C.codeTx, breakLine:true } })),
    { x:x+0.24,y:ty,w:w-0.48,h:y+h-ty-0.14,margin:0,lineSpacingMultiple:1.32,valign:"top" }); }
const G=C.gray, GOLD=C.gold, SRC="DINOv2-reg B/14 · ImageNet-1k val 50k · kNN k=20";

/* ── 1. 표지 (논문 제목) ── */
(()=>{ const s=pptx.addSlide(); s.background={color:C.dark};
  s.addShape(pptx.ShapeType.roundRect,{x:M,y:2.1,w:0.09,h:2.9,rectRadius:0.04,fill:{color:C.blue3},line:{type:"none"}});
  s.addText("Don't Merge the Registers", { x:0.85,y:2.15,w:11.8,h:1.0,margin:0,fontFace:F.black,fontSize:48,color:C.white,charSpacing:-1 });
  s.addText("Register-Aware Token Reduction for Vision Encoders with Registers", { x:0.86,y:3.25,w:11.8,h:0.6,margin:0,fontFace:F.med,fontSize:21,color:C.blueLt });
  s.addText("재학습 없이 register 토큰을 지켜, 극단 압축에서 정확도를 유지한다", { x:0.86,y:4.05,w:11.8,h:0.5,margin:0,fontFace:F.med,fontSize:15,color:"aeb6c2" });
  s.addText([{text:"ACCV 2026 (준비 중)",options:{fontFace:F.semi,fontSize:12,color:C.blue3}},{text:"    ·    Anonymous submission    ·    발표용 요약 덱",options:{fontFace:F.reg,fontSize:12,color:C.mut}}],
    { x:0.86,y:6.45,w:11.8,h:0.4,margin:0 });
})();

/* ── 2. Motivation ── */
(()=>{ const s=pptx.addSlide(); frame(s,"01 · Motivation",2,"foundation 인코더 = 재학습 없이 쓰는 특징 추출기");
  title(s,"동기 — 강력한 비전 인코더는 '토큰이 많아' 느리다","토큰을 줄이면 빨라지지만, 세게 줄이면 정확도가 무너지는 문제가 있다");
  const ry=2.5,rh=6.85-ry; const cw=(CW-2*0.3)/3;
  const cards=[["강력하지만 비싸다","DINOv2·CLIP 같은 인코더는 사진을 수백 개 토큰으로 처리 → 추론 비용이 크다.",C.blue,"수백 개 토큰"],
    ["토큰 줄이기 = 가속","비슷한 토큰을 합쳐 개수를 줄이면(재학습 없이) 빨라진다 — 잘 알려진 방법.",C.blue2,"재학습 없음"],
    ["하지만 극단서 붕괴","90% 넘게 줄이면 기존 방법은 정확도가 급락한다. 왜? → 이 발표의 주제.",C.err,"이게 문제"]];
  cards.forEach(([t,d,ac,tag],i)=>{ const x=M+i*(cw+0.3);
    card(s,x,ry,cw,rh-1.1,{shadow:shStd()}); accent(s,x,ry+0.22,rh-1.5,ac);
    chip(s,x+0.26,ry+0.24,tag,i===2?C.errBg:C.blueBg,ac,10);
    s.addText(t,{x:x+0.26,y:ry+0.72,w:cw-0.5,h:0.6,margin:0,fontFace:F.bold,fontSize:15,color:C.ink,lineSpacingMultiple:1.1});
    s.addText(d,{x:x+0.26,y:ry+1.42,w:cw-0.5,h:rh-2.7,margin:0,fontFace:F.reg,fontSize:12,color:C.ink2,lineSpacingMultiple:1.42}); });
  card(s,M,ry+rh-0.95,CW,0.8,{fill:C.blueBg,line:null});
  s.addText([{text:"핵심 질문   ",options:{fontFace:F.bold,fontSize:13,color:C.blue}},{text:"세게 압축해도 정확도를 지키려면, '무엇을' 지키며 합쳐야 하는가?",options:{fontFace:F.med,fontSize:13.5,color:C.ink}}],
    {x:M+0.35,y:ry+rh-0.95,w:CW-0.7,h:0.8,margin:0,valign:"middle"});
})();

/* ── 3. Background: kNN ── */
(()=>{ const s=pptx.addSlide(); frame(s,"02 · Background",3,"kNN = k-Nearest Neighbors (k개 최근접 이웃)");
  title(s,"평가 지표 — kNN 정확도 (분류기 학습 없이)","모델이 만든 특징 벡터만으로, 가장 비슷한 이웃들의 다수결로 라벨을 맞힌다");
  const ry=2.5,rh=6.85-ry;
  card(s,M,ry,7.2,rh,{shadow:shStd()});
  s.addText("동작 방식",{x:M+0.26,y:ry+0.16,w:3,h:0.3,margin:0,fontFace:F.semi,fontSize:13,color:C.ink});
  card(s,M+0.4,ry+0.9,1.1,1.1,{fill:C.blueBg,line:C.blue2});
  s.addText("?",{x:M+0.4,y:ry+0.9,w:1.1,h:1.1,margin:0,align:"center",valign:"middle",fontFace:F.black,fontSize:32,color:C.blue});
  s.addText("새 이미지",{x:M+0.3,y:ry+2.05,w:1.3,h:0.28,margin:0,align:"center",fontFace:F.med,fontSize:10,color:C.ink2});
  arrowR(s,M+1.7,ry+1.34,0.6,C.blue3);
  ["개 87%","개 82%","고양이"].forEach((t,i)=>{ const cy=ry+0.95+i*0.5;
    card(s,M+2.45,cy,1.5,0.4,{fill:i<2?C.okBg:C.errBg,line:null});
    s.addText(`이웃 ${i+1}: ${t}`,{x:M+2.45,y:cy,w:1.5,h:0.4,margin:0,align:"center",valign:"middle",fontFace:F.med,fontSize:9.5,color:i<2?C.ok:C.err}); });
  s.addText("가장 비슷한 k개",{x:M+2.4,y:ry+2.45,w:1.6,h:0.28,margin:0,align:"center",fontFace:F.reg,fontSize:9.5,color:C.mut});
  arrowR(s,M+4.15,ry+1.34,0.55,C.blue3);
  card(s,M+4.9,ry+0.95,1.95,1.0,{fill:C.blue,line:null});
  s.addText([{text:"다수결 → ",options:{fontFace:F.semi,fontSize:12,color:C.blueLt}},{text:"개",options:{fontFace:F.black,fontSize:22,color:C.white}}],
    {x:M+4.9,y:ry+0.95,w:1.95,h:1.0,margin:0,align:"center",valign:"middle"});
  s.addText("kNN 정확도 = 이렇게 맞힌 비율(%)",{x:M+0.26,y:ry+2.95,w:6.7,h:0.4,margin:0,align:"center",fontFace:F.semi,fontSize:13,color:C.ink});
  const rx=M+7.5,rw=12.333-7.5; card(s,rx,ry,rw,rh,{fill:C.blueBg,line:null,shadow:shStd()}); accent(s,rx,ry+0.25,rh-0.5,C.blue);
  s.addText("왜 이 지표인가",{x:rx+0.28,y:ry+0.22,w:rw-0.5,h:0.34,margin:0,fontFace:F.bold,fontSize:15,color:C.blue});
  [["학습이 필요 없다","별도 분류기를 훈련하지 않고, 모델의 '특징' 자체만 본다."],
   ["특징 품질을 정직하게","분류기 훈련은 손상을 가릴 수 있다. kNN은 특징이 나쁘면 그대로 드러난다."],
   ["우리 실험에 적합","토큰을 합쳐도 특징이 잘 보존되는지 왜곡 없이 측정한다."]]
   .forEach(([t,d],i)=>{ const y=ry+0.75+i*1.28;
     s.addText("· "+t,{x:rx+0.28,y,w:rw-0.5,h:0.3,margin:0,fontFace:F.semi,fontSize:12.5,color:C.ink});
     s.addText(d,{x:rx+0.42,y:y+0.33,w:rw-0.64,h:0.8,margin:0,fontFace:F.reg,fontSize:10.5,color:C.ink2,lineSpacingMultiple:1.34}); });
})();

/* ── 4. Background: ViT ── */
(()=>{ const s=pptx.addSlide(); frame(s,"02 · Background",4,"ViT = Vision Transformer");
  title(s,"ViT 구조 — 사진을 '토큰'으로 바꿔 처리한다","사진 → 패치로 자름 → 벡터(토큰) → 블록 여러 개 통과 → 요약(CLS)");
  const ry=2.5;
  const boxes=[["사진",C.blueBg,C.blue2],["패치로\n자름",C.bg2,"888888"],["토큰\n(벡터)",C.bg2,"888888"],
    ["Transformer\n블록 × N",C.blue,C.white],["CLS\n요약",C.goldBg,C.gold],["특징\n벡터",C.okBg,C.ok]];
  const bw=1.5,gap=0.52,x0=(13.333-(boxes.length*bw+(boxes.length-1)*gap))/2;
  boxes.forEach(([t,fill,ec],i)=>{ const x=x0+i*(bw+gap),dark=(ec===C.white);
    card(s,x,ry+0.35,bw,1.05,{fill,line:dark?null:ec,shadow:shStd(),r:0.09});
    s.addText(t,{x,y:ry+0.35,w:bw,h:1.05,margin:0,align:"center",valign:"middle",fontFace:F.semi,fontSize:12.5,color:dark?C.white:C.ink,lineSpacingMultiple:1.05});
    if(i<boxes.length-1) arrowR(s,x+bw+0.12,ry+0.71,gap-0.24,C.blue3); });
  const by=ry+2.05,bh=6.85-by-0.42; card(s,M,by,CW,bh,{fill:C.bg2,line:null});
  s.addText("Transformer 블록 1개 = 이 두 단계 (여러 개를 쌓음)",{x:M+0.3,y:by+0.18,w:9,h:0.3,margin:0,fontFace:F.semi,fontSize:13,color:C.ink});
  const iy=by+0.7,ih=bh-0.95;
  [["Attention (어텐션)","토큰들이 서로 정보를 주고받음 — '어떤 조각이 서로 관련있나'를 섞는다.",C.blue],
   ["MLP","각 토큰을 더 풍부한 표현으로 변환한다.",C.blue2]].forEach(([t,d,ac],i)=>{ const x=M+0.4+i*(CW/2);
    card(s,x,iy,CW/2-0.7,ih,{shadow:shStd()}); accent(s,x,iy+0.2,ih-0.4,ac);
    s.addText(t,{x:x+0.28,y:iy+0.18,w:CW/2-1.1,h:0.34,margin:0,fontFace:F.bold,fontSize:14,color:ac});
    s.addText(d,{x:x+0.28,y:iy+0.6,w:CW/2-1.1,h:ih-0.78,margin:0,fontFace:F.reg,fontSize:11.5,color:C.ink2,lineSpacingMultiple:1.4});
    if(i===0) s.addText("+",{x:x+CW/2-0.74,y:iy,w:0.5,h:ih,margin:0,align:"center",valign:"middle",fontFace:F.black,fontSize:24,color:C.mut}); });
  s.addText("핵심: ViT는 처음부터 끝까지 '토큰(벡터)들'을 다룬다 → 토큰 수를 줄이면 계산이 준다.",{x:M,y:6.55,w:CW,h:0.3,margin:0,align:"center",fontFace:F.med,fontSize:11.5,color:C.blue});
})();

/* ── 5. Background: 모델 (CLIP=대조군 정정) ── */
(()=>{ const s=pptx.addSlide(); frame(s,"02 · Background",5,"주력 = DINOv2-reg · 대조군 = register 없는 DINOv2·CLIP");
  title(s,"우리가 쓰는 모델 — 그리고 'register(메모장) 토큰'","공개 사전학습 모델을 재학습 없이 그대로 사용. register가 있는 인코더가 주 대상");
  const ry=2.5,rh=6.85-ry, cw=(CW-0.4)/2;
  card(s,M,ry,cw,rh,{shadow:shStd()});
  s.addText("역할별 모델",{x:M+0.28,y:ry+0.16,w:cw-0.5,h:0.34,margin:0,fontFace:F.bold,fontSize:15,color:C.ink});
  [["주력: DINOv2-reg","register 4개를 가진 자기지도 인코더. 여기서 우리 방법이 큰 이득. (base 완료, small/large 진행)",C.blue,"MAIN"],
   ["대조군 A: register 없는 DINOv2","register가 없어 이득이 사라져야 정상 → 이득이 register 덕분임을 증명하는 음성 대조군.",C.ink2,"CONTROL"],
   ["대조군 B: CLIP","register 없음(대조 성질만). ⚠ 코드상 최종 projection 누락으로 절대값은 무의미 — 대조용으로만.",C.mut,"CONTROL"]]
   .forEach(([t,d,ac,tg],i)=>{ const y=ry+0.68+i*1.18;
     accent(s,M+0.28,y+0.02,0.94,ac); chip(s,M+0.46,y,tg, i===0?C.blueBg:C.bg2, ac,8.5);
     s.addText(t,{x:M+1.3,y:y-0.02,w:cw-1.55,h:0.36,margin:0,valign:"middle",fontFace:F.bold,fontSize:12.5,color:ac});
     s.addText(d,{x:M+0.46,y:y+0.38,w:cw-0.72,h:0.72,margin:0,fontFace:F.reg,fontSize:10.5,color:C.ink2,lineSpacingMultiple:1.32}); });
  const rx=M+cw+0.4; card(s,rx,ry,cw,rh,{fill:C.goldBg,line:null,shadow:shStd()});
  s.addText("register = '메모장' 토큰",{x:rx+0.28,y:ry+0.16,w:cw-0.5,h:0.34,margin:0,fontFace:F.bold,fontSize:15,color:C.gold});
  tokRow(s,rx+cw/2,ry+0.82,[G,G,GOLD,G,G,G,GOLD,G],[2,6]);
  s.addText("★금색 = register (사진 전체 요약을 담는 소수의 특별한 토큰)",{x:rx+0.28,y:ry+1.62,w:cw-0.5,h:0.5,margin:0,align:"center",fontFace:F.med,fontSize:11,color:C.ink,lineSpacingMultiple:1.3});
  card(s,rx+0.28,ry+2.32,cw-0.56,1.45,{fill:C.white,line:null,shadow:shStd()});
  s.addText([{text:"왜 중요? ",options:{fontFace:F.bold,fontSize:11.5,color:C.gold}},
    {text:"학습 때 register를 넣으면(DINOv2-reg) 흩어진 '전체 요약'이 이 몇 개 토큰에 모인다. 공개 모델을 그대로 쓰되, 우리는 추론에서 이 토큰을 지킨다.",options:{fontFace:F.reg,fontSize:10.5,color:C.ink2}}],
    {x:rx+0.5,y:ry+2.5,w:cw-1.0,h:1.15,margin:0,lineSpacingMultiple:1.34});
  s.addText("출처: Darcet 외, \"Vision Transformers Need Registers\", ICLR 2024",{x:rx+0.28,y:ry+rh-0.4,w:cw-0.5,h:0.3,margin:0,fontFace:F.reg,fontSize:9,color:C.mut});
})();

/* ── 6. Related Work: ToMe / PiToMe ── */
(()=>{ const s=pptx.addSlide(); frame(s,"03 · Related Work",6,"ToMe: ICLR 2023 · PiToMe: NeurIPS 2024");
  title(s,"기존 방법 — 토큰 '합치기'를 추론에 끼우는 두 논문","둘 다 재학습 없이 forward에 삽입. 차이는 '무엇을 지킬지' 고르는 기준");
  const ry=2.5,rh=6.85-ry, cw=(CW-0.4)/2;
  const P=[["ToMe","Token Merging: Your ViT But Faster","비슷한 토큰 두 개를 bipartite soft matching으로 짝지어 벡터 평균으로 합침 + size-가중. 블록마다 삽입, 재학습 없음. 우리 실험의 baseline.",C.blue,"보호: CLS만"],
    ["PiToMe","Spectrum-Preserving Token Merging","같은 합치기 틀에서, 지킬 토큰을 '에너지 점수(고립도)'로 고름. 우리 ablation에서 energy 프록시로 비교.",C.blue2,"보호: 에너지 높은 토큰"]];
  P.forEach(([t,full,d,ac,keep],i)=>{ const x=M+i*(cw+0.4);
    card(s,x,ry,cw,rh,{shadow:shStd()}); accent(s,x,ry+0.24,rh-0.5,ac);
    s.addText(t,{x:x+0.3,y:ry+0.22,w:cw-0.6,h:0.4,margin:0,fontFace:F.black,fontSize:20,color:ac});
    s.addText(full,{x:x+0.3,y:ry+0.68,w:cw-0.6,h:0.34,margin:0,fontFace:F.med,fontSize:11,color:C.mut,italic:true});
    s.addText(d,{x:x+0.3,y:ry+1.15,w:cw-0.6,h:1.5,margin:0,fontFace:F.reg,fontSize:12,color:C.ink2,lineSpacingMultiple:1.42});
    chip(s,x+0.3,ry+rh-0.95,keep, i===0?C.blueBg:C.bg2, ac,10.5); });
  card(s,M,ry+rh+0.02,CW,0,{}); // spacer noop
  s.addText("→ 두 방법 다 'register 구조'는 보지 않는다. 우리는 여기에 register 보호를 더한다.",{x:M,y:6.52,w:CW,h:0.3,margin:0,align:"center",fontFace:F.med,fontSize:12,color:C.blue});
})();

/* ── 7. 추론 삽입 + 코드 before/after ── */
(()=>{ const s=pptx.addSlide(); frame(s,"03 · Related Work",7,"training-free = 재학습 없이 forward에만 삽입 · tome_reg.py");
  title(s,"'추론 시점'에 끼우는 이유 — 코드 before / after","재학습을 안 하므로, 토큰은 forward 중 블록 사이에서만 줄일 수 있다 (가중치 불변)");
  const ry=2.5,rh=6.85-ry, cw=(CW-0.35)/2;
  codeCard(s,M,ry,cw,rh-0.7,"BEFORE · 일반 ViT forward",[
    {t:"t = pos_embed(patch_embed(x))"},
    {t:"for blk in model.blocks:"},
    {t:"    t = blk(t)          # 그냥 통과",c:C.codeMut},
    {t:"t = model.norm(t)"},
    {t:"return t[:, 0]          # CLS 특징"},
  ],12);
  codeCard(s,M+cw+0.35,ry,cw,rh-0.7,"AFTER · 합치기 삽입 (reduced_forward)",[
    {t:"t = pos_embed(patch_embed(x))"},
    {t:"size = ones(B, T, 1)    # 대표 원본 수",c:C.codeMut},
    {t:"for blk in model.blocks:"},
    {t:"    t = blk(t)"},
    {t:"    t, size = merge_step(", c:C.codeGold},
    {t:"        t, size, r, n_protect)  # ★삽입", c:C.codeGold},
    {t:"t = model.norm(t)"},
    {t:"return t[:, 0]"},
  ],12);
  card(s,M,ry+rh-0.62,CW,0.55,{fill:C.blueBg,line:null});
  s.addText([{text:"딱 한 줄 추가 ",options:{fontFace:F.bold,fontSize:12.5,color:C.blue}},{text:"(merge_step) — 블록마다 r개 병합. 가중치는 하나도 안 바뀐다 = training-free. 이 삽입 방식이 곧 ToMe.",options:{fontFace:F.med,fontSize:12.5,color:C.ink}}],
    {x:M+0.35,y:ry+rh-0.62,w:CW-0.7,h:0.55,margin:0,valign:"middle"});
})();

/* ── 8. Problem ── */
(()=>{ const s=pptx.addSlide(); frame(s,"04 · Problem",8,"극단 압축(>90%)에서 발생");
  title(s,"문제 — 기존 방법이 'register(메모장)'까지 합쳐버린다","register는 겉보기엔 평범해 보여, 세게 압축하면 합쳐져 사라진다 → 정확도 급락");
  const ry=2.6,rh=6.85-ry; card(s,M,ry,CW,rh-0.05,{shadow:shStd()});
  tokRow(s,4.4,ry+0.55,[G,GOLD,G,G,GOLD,G,G,G],[1,4]);
  arrowD(s,4.24,ry+1.32,0.5,C.err);
  s.addText("기존 ToMe: 메모장도 '평범한 조각'으로 보고 합침",{x:5.4,y:ry+1.38,w:7.3,h:0.4,margin:0,valign:"middle",fontFace:F.semi,fontSize:13,color:C.err});
  tokRow(s,4.4,ry+2.0,[G,G,G,G]);
  s.addText("→ ★메모장 사라짐",{x:6.4,y:ry+2.06,w:3.5,h:0.5,margin:0,valign:"middle",fontFace:F.bold,fontSize:15,color:C.err});
  card(s,M+0.4,ry+2.95,CW-0.8,0.95,{fill:C.errBg,line:null});
  s.addText([{text:"결과:  ",options:{fontFace:F.bold,fontSize:14,color:C.err}},{text:"사진 전체 요약이 날아가 정확도가 급락한다. 특히 90% 넘게 줄이는 극단 압축에서 붕괴에 가깝다.",options:{fontFace:F.med,fontSize:13,color:C.ink}}],
    {x:M+0.7,y:ry+2.95,w:CW-1.4,h:0.95,margin:0,valign:"middle"});
})();

/* ── 9. Method 개념 ── */
(()=>{ const s=pptx.addSlide(); frame(s,"05 · Method",9,"training-free plug-in (재학습 없음)");
  title(s,"해결 — register(메모장)만 '보호'하고 나머지만 합친다","같은 합치기 알고리즘에서 register는 병합 대상에서 제외, 나머지 패치만 과감히 합친다");
  const ry=2.55,rh=6.85-ry; card(s,M,ry,CW,2.28,{shadow:shStd()});
  tokRow(s,4.4,ry+0.34,[G,GOLD,G,G,GOLD,G,G,G],[1,4],[1,4]);
  s.addText("초록 테두리 = 보호(안 합침)",{x:7.6,y:ry+0.28,w:3.4,h:0.32,margin:0,valign:"middle",fontFace:F.med,fontSize:11,color:C.ok});
  arrowD(s,4.24,ry+1.08,0.44,C.ok);
  tokRow(s,4.4,ry+1.5,[GOLD,G,GOLD,G,G],[0,2],[0,2]);
  s.addText("→ 메모장 유지",{x:7.0,y:ry+1.56,w:3.5,h:0.44,margin:0,valign:"middle",fontFace:F.bold,fontSize:13,color:C.ok});
  const sy=ry+2.48,sh=rh-2.48; const sw=(CW-2*0.3)/3;
  [["1. 식별","register가 있으면 그 토큰들 (없으면 값이 큰 토큰 top-k)",C.blue],
   ["2. 보호","이 토큰들은 병합에서 제외 — 그대로 통과",C.gold],
   ["3. 병합","나머지 중복 패치만 size-가중 평균으로 강하게 합침",C.ok]]
   .forEach(([t,d,ac],i)=>{ const x=M+i*(sw+0.3);
    card(s,x,sy,sw,sh,{shadow:shStd()}); accent(s,x,sy+0.2,sh-0.4,ac);
    s.addText(t,{x:x+0.26,y:sy+0.2,w:sw-0.5,h:0.34,margin:0,fontFace:F.bold,fontSize:15,color:ac});
    s.addText(d,{x:x+0.26,y:sy+0.62,w:sw-0.5,h:sh-0.8,margin:0,fontFace:F.reg,fontSize:12,color:C.ink2,lineSpacingMultiple:1.4}); });
})();

/* ── 10. Method 코드 변화 (n_protect 1→5) ── */
(()=>{ const s=pptx.addSlide(); frame(s,"05 · Method",10,"우리 변화 = 보호 토큰 수 n_protect 하나 · tome_reg.py");
  title(s,"코드로 본 우리 방식 — 'n_protect 1 → 5'가 전부","합치기는 ToMe 그대로. 우리는 보호할 prefix 개수만 바꾼다 (CLS만 → CLS+register)");
  const ry=2.5,rh=6.85-ry, cw=(CW-0.35)/2;
  codeCard(s,M,ry,cw,rh-0.7,"merge_step — 보호 부분은 병합서 제외",[
    {t:"def merge_step(x, size, r, n_protect):"},
    {t:"  xp = x[:, :n_protect]  # 보호(그대로)",c:C.codeGold},
    {t:"  xr = x[:, n_protect:]  # 여기만 병합",c:C.codeGreen},
    {t:"  a, b = xr[:, ::2], xr[:, 1::2]"},
    {t:"  # 비슷한 a→b 짝지어 size-가중 평균",c:C.codeMut},
    {t:"  ... (ToMe와 동일)"},
    {t:"  return cat([xp, unm, b_merged]), ..."},
  ],11.5);
  codeCard(s,M+cw+0.35,ry,cw,rh-0.7,"호출부 — 같은 함수, n_protect만 다름",[
    {t:"# ToMe (baseline)"},
    {t:"reduced_forward(m, x, r,"},
    {t:"    n_protect=1)      # CLS만 보호",c:C.codeMut},
    {t:""},
    {t:"# Ours (register-aware)"},
    {t:"reduced_forward(m, x, r,"},
    {t:"    n_protect=m.num_prefix_tokens)", c:C.codeGold},
    {t:"    # = 5 (CLS + register 4)", c:C.codeGold},
  ],11.5);
  card(s,M,ry+rh-0.62,CW,0.55,{fill:C.okBg,line:null});
  s.addText([{text:"방법의 전부 = 1 → 5. ",options:{fontFace:F.bold,fontSize:12.5,color:C.ok}},{text:"새 학습·새 손실·새 네트워크 없음. 그런데 극단 압축에서 +8%p (다음 장).",options:{fontFace:F.med,fontSize:12.5,color:C.ink}}],
    {x:M+0.35,y:ry+rh-0.62,w:CW-0.7,h:0.55,margin:0,valign:"middle"});
})();

/* ── 11. Experiments ── */
(()=>{ const s=pptx.addSlide(); frame(s,"06 · Experiments",11,"GPU: RTX PRO 6000 · ImageNet-1k val 50k");
  title(s,"실험 설계 — 이득의 원천이 register임을 못박기","같은 병합에서 '보호 대상만' 바꿔 비교. 공개 모델 그대로, 재학습 없음");
  const ry=2.5,rh=6.85-ry,cw=(CW-0.4)/2;
  card(s,M,ry,cw,rh,{shadow:shStd()});
  s.addText("설정",{x:M+0.28,y:ry+0.16,w:3,h:0.3,margin:0,fontFace:F.bold,fontSize:14,color:C.ink});
  [["모델","DINOv2-ViT-B/14 (register 4개), 공개 가중치"],["데이터","ImageNet-1k 검증셋 5만 장 전체"],
   ["지표","kNN top-1 (k=20, 재학습 없음)"],["압축률","37% → 92% 여러 단계"],
   ["비교","ToMe(보호=CLS만) vs Ours(보호=CLS+register)"]]
   .forEach(([k,v],i)=>{ const y=ry+0.62+i*0.72; chip(s,M+0.28,y,k,C.blueBg,C.blue,10);
     s.addText(v,{x:M+1.75,y:y-0.02,w:cw-2.05,h:0.4,margin:0,valign:"middle",fontFace:F.med,fontSize:11,color:C.ink}); });
  const rx=M+cw+0.4; card(s,rx,ry,cw,rh,{shadow:shStd()});
  s.addText("'진짜 register 덕분인가' 검증 장치",{x:rx+0.28,y:ry+0.16,w:cw-0.5,h:0.3,margin:0,fontFace:F.bold,fontSize:14,color:C.ink});
  [["① Ablation (보호 대상만 교체)","보호 개수는 같게 두고 register 대신 random·energy(PiToMe식)·highnorm 보호 → register만 효과면 '개수'가 아님이 증명.",C.gold],
   ["② 음성 대조군 (register 없는 모델)","register 없는 DINOv2·CLIP엔 어떤 보호도 효과 없어야 정상.",C.blue2],
   ["③ Dense 작업","분할(ADE20k)에서도 도움되는지 확인.",C.ok]]
   .forEach(([t,d,ac],i)=>{ const y=ry+0.6+i*((rh-0.6-2*0.16)/3+0.16), ih=(rh-0.6-2*0.16)/3;
    card(s,rx+0.2,y,cw-0.4,ih,{fill:C.bg2,line:null}); accent(s,rx+0.2,y+0.14,ih-0.28,ac);
    s.addText(t,{x:rx+0.44,y:y+0.12,w:cw-0.82,h:0.32,margin:0,fontFace:F.bold,fontSize:11.5,color:ac});
    s.addText(d,{x:rx+0.44,y:y+0.44,w:cw-0.82,h:ih-0.56,margin:0,fontFace:F.reg,fontSize:9.8,color:C.ink2,lineSpacingMultiple:1.3}); });
})();

/* ── 12. Results ① 라인 ── */
(()=>{ const s=pptx.addSlide(); frame(s,"07 · Results",12,SRC+" · single seed");
  title(s,"결과 ① — 세게 압축할수록 격차가 커진다","모든 압축률에서 Ours 우세, 92%에서 +8.07% (붕괴 직전을 견딤)");
  const ry=2.5,rh=6.85-ry,chw=8.2; card(s,M,ry,chw,rh,{shadow:shStd()});
  s.addText("압축률별 kNN 정확도 (%)",{x:M+0.25,y:ry+0.14,w:5,h:0.28,margin:0,fontFace:F.semi,fontSize:12.5,color:C.ink});
  s.addChart(pptx.ChartType.line,[
    {name:"Ours (register 보호)",labels:["0%","37%","55%","74%","83%","92%"],values:[76.35,75.71,75.30,74.13,73.28,71.98]},
    {name:"ToMe (보호 안 함)",labels:["0%","37%","55%","74%","83%","92%"],values:[76.35,74.70,72.93,70.34,67.62,63.91]},
  ],{ x:M+0.15,y:ry+0.5,w:chw-0.35,h:rh-0.9, chartColors:[C.ok,C.err], lineSize:3, lineDataSymbol:"circle", lineDataSymbolSize:6,
    valAxisMinVal:60, valAxisMaxVal:78, showValue:false,
    catAxisTitle:"토큰 줄인 정도", showCatAxisTitle:true, catAxisTitleFontSize:9, catAxisTitleColor:C.mut, catAxisTitleFontFace:F.med,
    catAxisLabelFontFace:F.med, catAxisLabelFontSize:10, catAxisLabelColor:C.ink2,
    valAxisLabelFontFace:F.reg, valAxisLabelFontSize:9, valAxisLabelColor:C.mut,
    valGridLine:{style:"solid",color:C.border2,size:0.5}, valAxisLineColor:C.border2, catAxisLineColor:C.border2,
    showLegend:true, legendPos:"b", legendFontFace:F.med, legendFontSize:10, legendColor:C.ink2 });
  const rx=M+chw+0.24,rw=12.333-chw-0.24;
  [["단조 증가","격차가 +1.0 → +8.1로 압축률 따라 커진다.",C.blue],
   ["극단서 견고","92%서 Ours 71.98(무압축 −4.4)인데 ToMe는 63.91로 붕괴(−12.4).",C.ok],
   ["의미","극단 압축이 우리 영역 — 남들이 무너지는 지점에서 앞선다.",C.gold]]
   .forEach(([t,d,ac],i)=>{ const ih=(rh-2*0.16)/3, y=ry+i*(ih+0.16);
    card(s,rx,y,rw,ih,{shadow:shStd()}); accent(s,rx,y+0.16,ih-0.32,ac);
    s.addText(t,{x:rx+0.24,y:y+0.16,w:rw-0.45,h:0.3,margin:0,fontFace:F.bold,fontSize:13,color:ac});
    s.addText(d,{x:rx+0.24,y:y+0.5,w:rw-0.45,h:ih-0.62,margin:0,fontFace:F.reg,fontSize:10.5,color:C.ink2,lineSpacingMultiple:1.34}); });
})();

/* ── 13. Results ② ablation ── */
(()=>{ const s=pptx.addSlide(); frame(s,"07 · Results",13,"91% 압축 · 보호 개수 동일 · 보호 대상만 교체");
  title(s,"결과 ② — '개수'가 아니라 'register라서' 좋다","같은 개수를 보호해도 register만 효과. random·energy·highnorm은 전부 ToMe 동급");
  const ry=2.5,rh=6.85-ry,chw=8.2; card(s,M,ry,chw,rh,{shadow:shStd()});
  s.addText("91% 압축 kNN 정확도 (%) — 보호 대상별",{x:M+0.25,y:ry+0.14,w:6,h:0.28,margin:0,fontFace:F.semi,fontSize:12.5,color:C.ink});
  s.addChart(pptx.ChartType.bar,[
    {name:"정확도",labels:["Ours (register)","energy (PiToMe식)","highnorm","random","ToMe (보호 최소)"],values:[71.98,64.04,63.84,63.19,63.90]},
  ],{ x:M+0.15,y:ry+0.5,w:chw-0.35,h:rh-0.9, barDir:"bar", chartColors:[C.ok], barGapWidthPct:45, valAxisMinVal:60, valAxisMaxVal:74,
    showValue:true, dataLabelFontSize:11, dataLabelFontFace:F.bold, dataLabelPosition:"outEnd", dataLabelColor:C.ink, dataLabelFormatCode:"0.00",
    catAxisLabelFontFace:F.med, catAxisLabelFontSize:10.5, catAxisLabelColor:C.ink2,
    valAxisLabelFontFace:F.reg, valAxisLabelFontSize:8.5, valAxisLabelColor:C.mut,
    valGridLine:{style:"solid",color:C.border2,size:0.5}, valAxisLineColor:C.border2, catAxisLineColor:C.border2, showLegend:false });
  const rx=M+chw+0.24,rw=12.333-chw-0.24; card(s,rx,ry,rw,rh,{fill:C.okBg,line:null,shadow:shStd()}); accent(s,rx,ry+0.25,rh-0.5,C.ok);
  s.addText("이 그래프가 말하는 것",{x:rx+0.26,y:ry+0.22,w:rw-0.5,h:0.34,margin:0,fontFace:F.bold,fontSize:15,color:C.ok});
  [["register만 우뚝","다음 best(energy 64.0)보다 +7.9. 나머지는 다 ToMe(63.9) 수준."],
   ["'개수' 반박","같은 개수를 보호해도 아무거나/유사도/노름은 효과 0 → 'register라서'가 증명."],
   ["정직한 범위","register 없는 모델은 어떤 보호도 무효 = 이미 압축에 강건. 이득은 register 인코더에 특정."]]
   .forEach(([t,d],i)=>{ const y=ry+0.72+i*1.3;
     s.addText("· "+t,{x:rx+0.26,y,w:rw-0.5,h:0.3,margin:0,fontFace:F.semi,fontSize:12.5,color:C.ink});
     s.addText(d,{x:rx+0.4,y:y+0.32,w:rw-0.62,h:0.85,margin:0,fontFace:F.reg,fontSize:10.5,color:C.ink2,lineSpacingMultiple:1.34}); });
})();

/* ── 14. Conclusion ── */
(()=>{ const s=pptx.addSlide(); frame(s,"08 · Conclusion",14,"Don't Merge the Registers");
  title(s,"결론 & 일반화","한 모델 트릭이 아니라, register를 가진 인코더군 전반으로");
  const ry=2.55,rh=6.85-ry; const cw=(CW-2*0.3)/3;
  [["적용 범위","register를 가진 인코더 전반(DINOv2-reg·DINOv3 등, register가 표준이 되는 추세). register 없는 모델도 '큰 토큰' 보호로 확장 가능.",C.blue],
   ["원리 확장","'register를 압축의 사전지식(prior)으로 쓴다'는 발상은 가지치기·적응 계산 등 다른 효율 기법으로도 확장 가능.",C.blue2],
   ["큰 그림","해석가능성 용도이던 register를 '속도(효율)'의 도구로 재해석.",C.gold]]
   .forEach(([t,d,ac],i)=>{ const x=M+i*(cw+0.3);
    card(s,x,ry,cw,2.55,{shadow:shStd()}); accent(s,x,ry+0.22,2.11,ac);
    s.addText(t,{x:x+0.26,y:ry+0.22,w:cw-0.5,h:0.34,margin:0,fontFace:F.bold,fontSize:15,color:ac});
    s.addText(d,{x:x+0.26,y:ry+0.66,w:cw-0.5,h:1.8,margin:0,fontFace:F.reg,fontSize:11.5,color:C.ink2,lineSpacingMultiple:1.42}); });
  card(s,M,ry+2.8,CW,rh-2.8,{fill:C.dark,line:null});
  s.addText([{text:"한 줄 정리   ",options:{fontFace:F.bold,fontSize:15,color:C.blue3}},{text:"공개 인코더를 재학습 없이, register만 지켜 합치면 극단 압축에서 표준 ToMe보다 앞선다 (+8%p).",options:{fontFace:F.med,fontSize:15,color:C.white}}],
    {x:M+0.4,y:ry+2.8,w:CW-0.8,h:rh-2.8,margin:0,valign:"middle",lineSpacingMultiple:1.3});
})();

pptx.writeFile({ fileName: "/home/jun/RNGD-proj/Model_Benchmark/ppt/Register_Token_Reduction.pptx" })
  .then(f=>console.log("saved", f)).catch(e=>console.error("ERR",e));
