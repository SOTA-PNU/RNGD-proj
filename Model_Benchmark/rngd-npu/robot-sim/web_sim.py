#!/usr/bin/env python3
"""브라우저에서 보는 로봇 내비게이션 시뮬레이터(헤드리스 서버에서도 100% 보임).

에피소드를 한 번 돌려 로봇 궤적·LiDAR 를 기록하고, 그걸 재생하는 **자기완결 HTML 캔버스
애니메이션**을 만듭니다. 로봇이 장애물을 피해 목표로 가는 모습을 창처럼 띄워 봅니다.
turtle 창은 디스플레이가 필요하지만, 이건 챗 UI 처럼 브라우저 터널로 그대로 보입니다.

예시
  # 서버 없이(가짜 LLM) trap 시나리오를 그려서 바로 띄우기(브라우저 터널로 보기)
  python3 web_sim.py --mock good --scenario trap --serve 7900
  # → 브라우저에서 http://<터널주소>:7900 접속

  # 실제 NPU 모델로(먼저 ../chat/serve_models.sh coder7 등으로 띄운 뒤, openai 있는 venv 로)
  ../chat/.venv/bin/python web_sim.py --model coder7 --scenario trap --serve 7900

  # 파일만 만들고 직접 열기
  python3 web_sim.py --mock good --scenario all --out sim.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))

from sim_record import MODELS, record_episode  # noqa: E402
import scenarios as SC                          # noqa: E402

HTML = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{ --bg:#0a0a0a; --card:#151515; --red:#dc2626; --cyan:#76d6ff; --purple:#cdbbff;
         --mute:#8a8a8a; --bd:#262626; --txt:#e8e8e8; }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);
       font-family:ui-sans-serif,system-ui,'Segoe UI',Roboto,'Apple SD Gothic Neo',sans-serif}
  header{display:flex;align-items:center;gap:12px;padding:14px 20px;border-bottom:1px solid var(--bd)}
  header .logo{width:14px;height:14px;border-radius:3px;background:var(--red)}
  header b{font-size:15px;letter-spacing:.3px}
  header .badge{margin-left:8px;font-size:11px;color:var(--mute);border:1px solid var(--bd);
                border-radius:6px;padding:2px 7px}
  .wrap{display:flex;gap:18px;padding:18px 20px;flex-wrap:wrap}
  .stage{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:12px}
  canvas{display:block;border-radius:8px;background:#070707}
  .side{min-width:240px;flex:1}
  .row{display:flex;gap:10px;align-items:center;margin:10px 0;flex-wrap:wrap}
  button{background:#1b1b1b;color:var(--txt);border:1px solid var(--bd);border-radius:9px;
         padding:8px 14px;font-size:13px;cursor:pointer}
  button:hover{border-color:var(--red)}
  button.play{background:var(--red);border-color:var(--red);color:#fff;font-weight:600}
  input[type=range]{width:100%}
  select{background:#1b1b1b;color:var(--txt);border:1px solid var(--bd);border-radius:9px;padding:7px}
  .kv{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #1c1c1c;font-size:13px}
  .kv span:first-child{color:var(--mute)}
  .ok{color:#34d399} .fail{color:var(--red)}
  .hint{color:var(--mute);font-size:12px;margin-top:8px;line-height:1.6}
</style></head>
<body>
<header><div class="logo"></div><b>Furiosa RNGD · Robot Nav Sim</b>
  <span class="badge" id="badge">scenario</span></header>
<div class="wrap">
  <div class="stage"><canvas id="cv" width="760" height="760"></canvas></div>
  <div class="side stage">
    <div class="row">
      <button class="play" id="play">⏸ 일시정지</button>
      <button id="restart">↻ 처음부터</button>
      <select id="scn"></select>
    </div>
    <div class="row"><span style="color:var(--mute);font-size:12px;width:46px">속도</span>
      <input type="range" id="speed" min="0.25" max="4" step="0.25" value="1"></div>
    <div class="row"><span style="color:var(--mute);font-size:12px;width:46px">위치</span>
      <input type="range" id="scrub" min="0" max="1" step="1" value="0"></div>
    <div id="stats"></div>
    <div class="hint" id="hint"></div>
  </div>
</div>
<script>
const EPISODES = /*__EPISODES__*/;
let cur = 0, M = EPISODES[0].meta, F = EPISODES[0].frames;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const DPR = Math.max(1, window.devicePixelRatio||1);
function fitDPR(){ const s=760; cv.width=s*DPR; cv.height=s*DPR; cv.style.width=s+'px'; cv.style.height=s+'px';
  ctx.setTransform(DPR,0,0,DPR,0,0); }
fitDPR();

let scale=1, ox=0, oy=0, PX=760;
function computeScale(){ const m=24; scale=Math.min((PX-2*m)/M.width,(PX-2*m)/M.height);
  ox=(PX-M.width*scale)/2; oy=(PX-M.height*scale)/2; }
function wx(x){ return ox + x*scale; }
function wy(y){ return oy + (M.height-y)*scale; }   // y-up → 화면 y-down

function drawGrid(){
  ctx.fillStyle='#070707'; ctx.fillRect(0,0,PX,PX);
  ctx.strokeStyle='#141414'; ctx.lineWidth=1;
  for(let g=0; g<=M.width+0.001; g+=2){ ctx.beginPath(); ctx.moveTo(wx(g),wy(0)); ctx.lineTo(wx(g),wy(M.height)); ctx.stroke(); }
  for(let g=0; g<=M.height+0.001; g+=2){ ctx.beginPath(); ctx.moveTo(wx(0),wy(g)); ctx.lineTo(wx(M.width),wy(g)); ctx.stroke(); }
  ctx.strokeStyle='#333'; ctx.strokeRect(wx(0),wy(M.height),M.width*scale,M.height*scale);
}
function drawObstacles(){
  for(const o of M.obstacles){
    const cx=wx(o.cx), cy=wy(o.cy), r=o.r*scale;
    const g=ctx.createRadialGradient(cx-r*0.3,cy-r*0.3,r*0.2,cx,cy,r);
    g.addColorStop(0,'#3a3a3a'); g.addColorStop(1,'#1c1c1c');
    ctx.fillStyle=g; ctx.beginPath(); ctx.arc(cx,cy,r,0,7); ctx.fill();
    ctx.strokeStyle='#444'; ctx.lineWidth=1; ctx.stroke();
  }
}
function drawGoal(){
  const gx=wx(M.goal[0]), gy=wy(M.goal[1]);
  // 도달 허용 반경 링
  ctx.strokeStyle='rgba(205,187,255,0.45)'; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.arc(gx,gy,M.goal_tol*scale,0,7); ctx.stroke();
  // 타겟 마커(겹친 원 + 중심점)
  ctx.strokeStyle='#cdbbff'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.arc(gx,gy,8,0,7); ctx.stroke();
  ctx.fillStyle='#cdbbff'; ctx.beginPath(); ctx.arc(gx,gy,3,0,7); ctx.fill();
  ctx.beginPath(); ctx.moveTo(gx-11,gy); ctx.lineTo(gx+11,gy);
  ctx.moveTo(gx,gy-11); ctx.lineTo(gx,gy+11); ctx.stroke();
  ctx.fillStyle='#cdbbff'; ctx.font='12px sans-serif'; ctx.fillText('GOAL', gx+13, gy-9);
}
function drawStart(){
  const sx=wx(M.start[0]), sy=wy(M.start[1]);
  ctx.strokeStyle='#76d6ff'; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(sx,sy,5,0,7); ctx.stroke();
  ctx.fillStyle='#76d6ff'; ctx.font='12px sans-serif'; ctx.fillText('START', sx+9, sy+4);
}
function drawTrail(upto){
  ctx.strokeStyle='rgba(220,38,38,0.7)'; ctx.lineWidth=2; ctx.beginPath();
  for(let i=0;i<=upto && i<F.length;i++){ const f=F[i]; const X=wx(f.x),Y=wy(f.y);
    if(i===0) ctx.moveTo(X,Y); else ctx.lineTo(X,Y); } ctx.stroke();
}
function lerp(a,b,t){ return a+(b-a)*t; }
function drawRobot(f){
  const X=wx(f.x), Y=wy(f.y), rr=M.robot_radius*scale;
  // LiDAR 부채살
  ctx.lineWidth=1;
  for(let i=0;i<(f.lidar||[]).length;i++){
    const wa=f.h+f.ang[i], d=f.lidar[i];
    const ex=wx(f.x+d*Math.cos(wa)), ey=wy(f.y+d*Math.sin(wa));
    const hit=d < M.max_range-0.05;
    ctx.strokeStyle= hit ? 'rgba(118,214,255,0.30)' : 'rgba(118,214,255,0.06)';
    ctx.beginPath(); ctx.moveTo(X,Y); ctx.lineTo(ex,ey); ctx.stroke();
    if(hit){ ctx.fillStyle='rgba(118,214,255,0.8)'; ctx.beginPath(); ctx.arc(ex,ey,2,0,7); ctx.fill(); }
  }
  // 몸체
  ctx.fillStyle='rgba(220,38,38,0.18)'; ctx.beginPath(); ctx.arc(X,Y,rr,0,7); ctx.fill();
  ctx.strokeStyle='#dc2626'; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(X,Y,rr,0,7); ctx.stroke();
  // 방향 삼각형(화면은 y-down 이라 각도 부호 반전)
  const sa=-f.h, L=rr*1.7;
  ctx.fillStyle='#ff5a5a'; ctx.beginPath();
  ctx.moveTo(X+L*Math.cos(sa), Y+L*Math.sin(sa));
  ctx.lineTo(X+rr*0.9*Math.cos(sa+2.5), Y+rr*0.9*Math.sin(sa+2.5));
  ctx.lineTo(X+rr*0.9*Math.cos(sa-2.5), Y+rr*0.9*Math.sin(sa-2.5));
  ctx.closePath(); ctx.fill();
}
function frameAt(t){
  const i=Math.min(F.length-1, Math.floor(t)), j=Math.min(F.length-1,i+1), a=t-i;
  const A=F[i], B=F[j];
  // 각도 보간(wrap 처리)
  let dh=B.h-A.h; while(dh>Math.PI)dh-=2*Math.PI; while(dh<-Math.PI)dh+=2*Math.PI;
  return {x:lerp(A.x,B.x,a), y:lerp(A.y,B.y,a), h:A.h+dh*a, lidar:A.lidar, ang:A.ang};
}
function dist(f){ return Math.hypot(M.goal[0]-f.x, M.goal[1]-f.y); }
function stats(idx){
  const f=F[Math.min(F.length-1,Math.round(idx))];
  const st = M.success ? '<span class="ok">✅ 미션 성공</span>'
                       : '<span class="fail">❌ '+M.reason+'</span>';
  if(M.house){                                   // 집 미션 전용 패널
    const phase = f.phase==='home' ? '복귀 중' : '집 안 탐색 중';
    const found = f.found ? '<span class="ok">발견함</span>' : '아직';
    const truth = M.present ? '실제로 있음' : '실제로 없음';
    document.getElementById('stats').innerHTML =
      kv('모델', M.model)+kv('미션', '물건 확인 후 복귀')
      +kv('찾는 물건', JSON.stringify(M.objective))
      +kv('정답(물건 존재)', truth)
      +kv('결과', st)+kv('현재 단계', phase)+kv('탐색 중 발견', found)
      +kv('스텝', Math.round(idx)+' / '+(F.length-1))
      +kv('코드 재작성', M.replans+'회')+kv('첫 빌드', M.code_valid_first?'성공':'실패');
    return;
  }
  const tgt = M.vision ? kv('찾는 사람', JSON.stringify(M.target)) : '';
  const seen = (M.vision && f.cam) ? kv('카메라에 보임', f.cam.length+'명') : '';
  document.getElementById('stats').innerHTML =
    kv('모델', M.model)+kv('시나리오', M.scenario)+tgt+kv('결과', st)
    +kv('스텝', Math.round(idx)+' / '+(F.length-1))
    +kv(M.vision?'목표 사람까지':'목표까지', dist(f).toFixed(2)+' m')+seen
    +kv('코드 재작성', M.replans+'회')+kv('첫 빌드', M.code_valid_first?'성공':'실패');
}
function kv(k,v){ return '<div class="kv"><span>'+k+'</span><span>'+v+'</span></div>'; }

// ── 사람찾기(카메라) 전용 그리기 ──
const SHIRT={red:'#dc2626',blue:'#3b82f6',green:'#22c55e',yellow:'#eab308'};
function drawFOV(f){
  if(!M.vision && !M.house) return;
  const X=wx(f.x),Y=wy(f.y), rng=(M.cam_range||9)*scale, half=(M.cam_fov||1.03)/2;
  const a0=-f.h-half, a1=-f.h+half;             // 화면 y-down → 각도 부호 반전
  ctx.beginPath(); ctx.moveTo(X,Y); ctx.arc(X,Y,rng,a0,a1); ctx.closePath();
  ctx.fillStyle='rgba(118,214,255,0.07)'; ctx.fill();
  ctx.strokeStyle='rgba(118,214,255,0.25)'; ctx.lineWidth=1; ctx.stroke();
}
function personColor(feat){ return SHIRT[feat.shirt]||'#9ca3af'; }
function drawPeople(){
  if(!M.vision||!M.people) return;
  for(const p of M.people){
    const X=wx(p.x),Y=wy(p.y), r=0.45*scale;
    if(p.is_target){ ctx.beginPath(); ctx.arc(X,Y,r+6,0,7);
      ctx.strokeStyle='#cdbbff'; ctx.lineWidth=2.5; ctx.stroke(); }   // target 하이라이트
    ctx.beginPath(); ctx.arc(X,Y,r,0,7); ctx.fillStyle=personColor(p.features); ctx.fill();
    ctx.strokeStyle='#000'; ctx.lineWidth=1; ctx.stroke();
    if(p.features.cap){ ctx.beginPath();                              // 모자 = 위 작은 삼각형
      ctx.moveTo(X-r*0.7,Y-r*0.7); ctx.lineTo(X+r*0.7,Y-r*0.7); ctx.lineTo(X,Y-r*1.6);
      ctx.closePath(); ctx.fillStyle='#111'; ctx.fill(); }
  }
}
function drawDetections(f){
  if(!M.vision||!f.cam) return;
  for(const d of f.cam){                                            // 그 순간 카메라에 '보이는' 사람으로 선
    const wa=f.h+d.bearing;
    const ex=wx(f.x+d.distance*Math.cos(wa)), ey=wy(f.y+d.distance*Math.sin(wa));
    ctx.strokeStyle='rgba(118,214,255,0.5)'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(wx(f.x),wy(f.y)); ctx.lineTo(ex,ey); ctx.stroke(); ctx.setLineDash([]);
  }
}

// ── 집 미션(house) 전용 그리기 ──
const ITEMCOLOR={red:'#dc2626',blue:'#3b82f6',green:'#22c55e',yellow:'#eab308'};
function drawWalls(){
  if(!M.walls) return;
  for(const w of M.walls){
    ctx.save();
    ctx.translate(wx(w.cx), wy(w.cy)); ctx.rotate(-w.th);   // 화면 y-down → 각도 반전
    const L=w.L*scale, T=Math.max(2.5, w.T*scale);
    ctx.fillStyle='#2c2c2c'; ctx.fillRect(-L/2,-T/2,L,T);
    ctx.strokeStyle='#454545'; ctx.lineWidth=1; ctx.strokeRect(-L/2,-T/2,L,T);
    ctx.restore();
  }
}
function drawRoute(){
  if(!M.waypoints||M.waypoints.length<2) return;
  ctx.strokeStyle='rgba(205,187,255,0.16)'; ctx.lineWidth=1.5; ctx.beginPath();
  M.waypoints.forEach((p,i)=>{ const X=wx(p[0]),Y=wy(p[1]); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y); });
  ctx.stroke();
}
function drawHome(){
  const h=M.home||M.start; const X=wx(h[0]),Y=wy(h[1]);
  ctx.strokeStyle='#76d6ff'; ctx.lineWidth=2; ctx.strokeRect(X-7,Y-7,14,14);
  ctx.fillStyle='#76d6ff'; ctx.font='12px sans-serif'; ctx.fillText('HOME', X+10, Y+4);
}
function drawItems(){
  if(!M.items) return;
  for(const it of M.items){
    const X=wx(it.x), Y=wy(it.y), s=Math.max(4,0.32*scale);
    if(it.is_target){ ctx.strokeStyle='#cdbbff'; ctx.lineWidth=2.5;
      ctx.strokeRect(X-s-4,Y-s-4,2*s+8,2*s+8); }                   // 진짜 목표 물건 하이라이트
    ctx.fillStyle=ITEMCOLOR[it.features.color]||'#9ca3af';
    ctx.fillRect(X-s,Y-s,2*s,2*s);
    ctx.strokeStyle='#000'; ctx.lineWidth=1; ctx.strokeRect(X-s,Y-s,2*s,2*s);
    ctx.fillStyle='#cfcfcf'; ctx.font='10px sans-serif';
    ctx.fillText(''+(it.features.label||''), X+s+3, Y+3);
  }
}
function drawScan(f){
  if(!M.house||!f.scan) return;
  for(const d of f.scan){                                          // 카메라에 지금 '보이는' 물건으로 선
    const wa=f.h+d.bearing;
    const ex=wx(f.x+d.distance*Math.cos(wa)), ey=wy(f.y+d.distance*Math.sin(wa));
    const isT = M.objective && Object.keys(M.objective).every(k=>d.features[k]===M.objective[k]);
    ctx.strokeStyle= isT ? 'rgba(205,187,255,0.85)' : 'rgba(118,214,255,0.45)';
    ctx.lineWidth= isT ? 2 : 1; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(wx(f.x),wy(f.y)); ctx.lineTo(ex,ey); ctx.stroke(); ctx.setLineDash([]);
  }
}

let t=0, playing=true, speed=1, last=0;
function render(){
  computeScale(); drawGrid(); drawObstacles();
  const f=frameAt(t);
  if(M.house){ drawWalls(); drawRoute(); drawHome(); drawFOV(f); drawItems();
    drawTrail(Math.round(t)); drawScan(f); drawRobot(f); stats(t); }
  else {
    drawFOV(f); drawPeople(); drawStart();
    if(!M.vision) drawGoal();
    drawTrail(Math.round(t)); drawDetections(f); drawRobot(f); stats(t);
  }
  document.getElementById('scrub').value=Math.round(t);
}
function loop(ts){
  if(!last) last=ts; const dt=(ts-last)/1000; last=ts;
  if(playing){ t += dt*10*speed; if(t>=F.length-1){ t=F.length-1; playing=false; setPlayBtn(); } }
  render(); requestAnimationFrame(loop);
}
function setPlayBtn(){ document.getElementById('play').textContent = playing?'⏸ 일시정지':'▶ 재생'; }

function loadEpisode(k){
  cur=k; M=EPISODES[k].meta; F=EPISODES[k].frames; t=0; playing=true; last=0;
  document.getElementById('badge').textContent=M.scenario+'  ·  '+M.model;
  document.getElementById('scrub').max=F.length-1;
  document.title='Robot Sim · '+M.scenario;
  setPlayBtn();
  document.getElementById('hint').innerHTML= M.house ?
    ('집(TurtleBot3 House) 도면을 방마다 돌며(보라 경로) 카메라로 물건을 스캔합니다. 보라 네모는 찾는 '
     +'물건, 회색/파랑은 헷갈리게 놓은 decoy 입니다. NPU LLM 이 짠 컨트롤러가 경로 추종·스캔·복귀·판정을 '
     +'수행하고, 실패하면 스스로 코드를 고칩니다. 빨간 선은 지나온 궤적입니다.')
    : ('LiDAR(파란 부채살)로 장애물을 감지하고, NPU LLM 이 짠 컨트롤러 코드로 매 주기 움직입니다. '
     +'빨간 선은 지나온 궤적입니다.');
}
// 컨트롤
document.getElementById('play').onclick=()=>{ if(t>=F.length-1)t=0; playing=!playing; if(playing)last=0; setPlayBtn(); };
document.getElementById('restart').onclick=()=>{ t=0; playing=true; last=0; setPlayBtn(); };
document.getElementById('speed').oninput=(e)=>{ speed=parseFloat(e.target.value); };
document.getElementById('scrub').oninput=(e)=>{ t=parseInt(e.target.value); playing=false; setPlayBtn(); };
const scn=document.getElementById('scn');
EPISODES.forEach((e,i)=>{ const o=document.createElement('option'); o.value=i;
  o.textContent=e.meta.scenario+(e.meta.success?'  ✅':'  ❌'); scn.appendChild(o); });
scn.onchange=(e)=>loadEpisode(parseInt(e.target.value));
loadEpisode(0); requestAnimationFrame(loop);
</script></body></html>
"""


def build_html(episodes: list) -> str:
    title = "Robot Sim · " + (episodes[0]["meta"]["scenario"] if episodes else "sim")
    payload = json.dumps(episodes, ensure_ascii=False)
    return HTML.replace("__TITLE__", title).replace("/*__EPISODES__*/", payload)


def main():
    ap = argparse.ArgumentParser(description="브라우저용 로봇 내비게이션 시뮬레이터(HTML 캔버스)")
    ap.add_argument("--model", help=f"모델 키 {list(MODELS)}")
    ap.add_argument("--port", type=int, help="furiosa-llm serve 포트 직접 지정")
    ap.add_argument("--mock", choices=["good", "buggy"], help="서버 없이 가짜 LLM")
    ap.add_argument("--scenario", default="trap",
                    help=f"시나리오 또는 all. {', '.join(SC.list_scenarios())}")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-replans", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--out", default="sim.html", help="HTML 저장 경로")
    ap.add_argument("--serve", type=int, metavar="PORT",
                    help="HTML 을 이 포트로 띄워(브라우저 터널로 보기). 예: --serve 7900")
    args = ap.parse_args()

    if args.scenario in ("all", "suite"):
        names = SC.DEFAULT_SUITE
    elif "," in args.scenario:                      # 콤마 목록: "find_person,trap,..."
        names = [s.strip() for s in args.scenario.split(",") if s.strip()]
    else:
        names = [args.scenario]
    episodes = []
    for name in names:
        print(f"▶ {name} 기록 중…")
        meta, frames = record_episode(
            name, mock=args.mock, model=args.model, port=args.port, seed=args.seed,
            max_replans=args.max_replans, max_tokens=args.max_tokens, quiet=True)
        print(f"  {'✅ 도달' if meta['success'] else '❌ '+meta['reason']}  "
              f"프레임 {len(frames)}개, 재작성 {meta['replans']}회")
        episodes.append({"meta": meta, "frames": frames})

    html = build_html(episodes)
    out = os.path.abspath(args.out)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nHTML 저장: {out}  ({len(html)//1024} KB)")

    if args.serve:
        import http.server
        import socketserver
        html_bytes = html.encode("utf-8")

        # 어떤 경로로 들어와도(루트 / 포함) 시뮬레이터 HTML 을 바로 준다 — /sim.html 안 쳐도 됨.
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)

            def log_message(self, *a):   # 접속 로그 조용히
                pass

        socketserver.TCPServer.allow_reuse_address = True   # 재기동 시 TIME_WAIT 회피
        with socketserver.TCPServer(("0.0.0.0", args.serve), Handler) as httpd:
            print(f"\n🌐 브라우저에서 열기:  http://127.0.0.1:{args.serve}/   (루트로 바로 보임)")
            print(f"   맥북: alpacon tunnel furiosa-npu-e6ec40 -l {args.serve} -r {args.serve}  먼저 실행 → 위 주소")
            print("   (Ctrl+C 로 종료.)")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n종료.")
    else:
        print("브라우저에서 이 파일을 열거나, --serve 7900 으로 띄워 터널로 보세요.")


if __name__ == "__main__":
    main()
