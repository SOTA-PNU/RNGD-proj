# 로봇 내비게이션 LLM 시뮬레이터 (robot-sim)

로봇이 **자기 위치와 목표 위치만** 알고, RNGD NPU 위에 떠 있는 **코딩 LLM에게 직접 질문해
자기를 움직일 코드를 받아, 그 코드를 자기 자신에게 적용**해 목표까지 스스로 찾아가는 과정을
시뮬레이션으로 검증하는 도구입니다.

핵심은 **코딩 성능**입니다. 로봇은 길을 외워서 가는 게 아니라, LLM이 짜 준 컨트롤러 코드를
실행해 움직이고, 막히면 *무슨 일이 있었는지를 적어 다시 코드를 고쳐 달라고* 합니다. 그래서
"한 번에 맞는 코드를 짜는가", "실패를 보고 스스로 고쳐 성공시키는가"를 숫자로 볼 수 있습니다.

```
          ┌─────────────── 폐루프(closed loop) ───────────────┐
 현재위치·목표·LiDAR ─▶  LLM에 코드 요청  ─▶  plan(state) 코드 수신
          ▲                                              │
          │                                              ▼
   실패 원인 적어 수리 요청  ◀── 충돌/정체/예외  ──  로봇이 그 코드로 매 스텝 이동
          │                                              │
          └──────────────── 목표 도달 시 성공 ◀──────────┘
```

---

## 빠른 시작

### 1) 서버 없이 동작 확인 (가짜 LLM, 시스템 python 만으로 가능)

추가 설치 없이 표준 라이브러리만으로 돕니다. 시뮬레이터 자체가 제대로 도는지 먼저 확인하세요.

```bash
cd Model_Benchmark/rngd-npu/robot-sim

# 정상 컨트롤러를 주는 가짜 LLM 으로 다섯 시나리오 전부 주행
python3 run_sim.py --mock good --scenario all

# '목표 반대로 도는' 버그 코드를 처음 주고, 수리 요청부터 정상 코드를 주는 가짜 LLM
#  → 자가수리(self-debug) 루프가 실제로 도는 걸 서버 없이 시연
python3 run_sim.py --mock buggy --scenario single
```

`--mock good` 은 5/5 도달, `--mock buggy` 는 1차 충돌 후 수리 1회로 도달하도록 맞춰 두었습니다.
가짜 LLM 은 시뮬레이터(폐루프·충돌판정·수리요청·지표)가 맞게 도는지 확인하는 용도입니다.

### 2) 실제 RNGD NPU 서버에 붙여 코딩 성능 측정

먼저 챗 서버 쪽 스크립트로 모델을 띄웁니다(코딩 성능이 관건이라 coder 계열을 권합니다).

```bash
cd Model_Benchmark/rngd-npu
./chat/serve_models.sh coder7          # Qwen2.5-Coder-7B-Inst (포트 8002, 카드 1장)
# 준비되면 serve_logs/8002.log 에 'Uvicorn running' 이 뜹니다.
```

그다음 **openai 패키지가 있는 venv** 로 시뮬레이터를 실행합니다(`chat/.venv` 또는 `~/furiosa`).

```bash
cd robot-sim
../chat/.venv/bin/python run_sim.py --model coder7 --scenario all --report coder7.json
../chat/.venv/bin/python run_sim.py --model a3b-fp8 --scenario trap --middleware threaded
```

등록된 모델 키와 시나리오는 `python3 run_sim.py --list` 로 볼 수 있습니다.

---

## 창으로 보기 (시각 시뮬레이터)

ASCII 말고 **로봇이 움직이는 걸 창에서** 보고 싶으면 두 가지가 있습니다.

### 1) 브라우저 시뮬레이터 — `web_sim.py` (헤드리스 서버에서도 보임, 권장)

에피소드를 돌려 로봇 궤적·LiDAR 를 기록하고, 그걸 재생하는 **HTML 캔버스 애니메이션**(다크 테마,
LiDAR 부채살·궤적·HUD·재생/속도/위치 컨트롤, 시나리오 드롭다운)을 만듭니다. 챗 UI 처럼 브라우저
터널로 그대로 보이므로 디스플레이가 없는 원격 서버에서도 됩니다.

```bash
# 서버 없이(가짜 LLM) 다섯 시나리오를 7900 포트로 띄우기
python3 web_sim.py --mock good --scenario all --serve 7900
#  → 7900 을 터널로 포워딩하고 브라우저에서  http://<서버주소>:7900/sim.html  접속

# 실제 NPU 모델로
../chat/.venv/bin/python web_sim.py --model coder7 --scenario trap --serve 7900

# 파일만 만들어 직접 열기(서버 없이)
python3 web_sim.py --mock good --scenario trap --out sim.html
```

### 2) turtle 창 — `turtle_sim.py` (모니터 있는 PC에서)

파이썬 표준 `turtle` 로 진짜 OS 창을 띄워 로봇이 움직이는 걸 보여줍니다. turtle 은 디스플레이가
필요하므로 **모니터가 있는 PC(노트북/맥에 이 저장소를 받아서)** 에서 실행하세요. 헤드리스/원격이면
자동으로 위 브라우저 버전을 안내합니다.

```bash
python3 turtle_sim.py --mock good --scenario trap          # 디스플레이 있는 PC에서
python3 turtle_sim.py --model coder7 --scenario slalom --lidar
```

둘 다 `--mock good|buggy`, `--model <키>`/`--port`, `--scenario`, `--seed` 옵션을 `run_sim.py` 와
똑같이 받습니다.

---

## 라이브로 보기 (3D) — 맵에서 모델·dp/pp·task 고르면 그때부터 NPU 에 요청 (`live_sim.py`)

`web_sim.py` 는 에피소드를 **미리 다 돌려 녹화한 걸 재생**합니다. 반면 `live_sim.py` 는 **집(House)을
3D 로 그려 놓고 로봇을 현관에 세워둔 뒤, 사용자가 task 를 고른 그 순간부터 실제로 NPU 서버에 코드를
요청**하며 한 걸음씩 진행하고, 그 과정(요청 → 받은 코드 → 빌드 결과 → 주행 → 실패 → 수리요청 → 새
코드 → 완료)을 브라우저로 **실시간 스트리밍(SSE)** 합니다. "움직이는 기능만 있는 로봇이, 목적을 받은
순간부터 LLM 과 대화하며 코드를 고쳐 task 를 해내는 과정"을 3D 로 지켜볼 수 있습니다.

```bash
# (실제 모델은 화면에서 고르므로 미리 serve 안 해도 됩니다 — 고르면 dp/pp 로 그때 띄웁니다)
cd robot-sim && ../chat/.venv/bin/python live_sim.py --port 7910
#   맥북:  alpacon tunnel furiosa-npu-e6ec40 -l 7910 -r 7910  → http://127.0.0.1:7910
```

- **3D 뷰**: Three.js 로 집을 입체로 렌더(벽·바닥·로봇·카메라 화각·궤적). 마우스로 **회전·확대·이동**
  할 수 있습니다. three.js 는 `vendor/` 에 같이 넣어 둬 인터넷/CDN 없이 터널에서도 뜹니다.
- **센서 모니터(3D 아래, 별도)**: 3D 맵과 따로 **📡 LiDAR 스캔**(로봇 기준 360° 방사형 스코프 — 빔·거리링·
  정면 화살표)과 **📷 카메라 시야**(전방 화각 프레임에 검출된 물건을 방위각 위치·거리 크기·색·라벨로 표시)를
  실시간으로 보여 줍니다. **로봇이 실제로 받는 LiDAR·카메라 신호 그 자체**를 따로 보는 창입니다.
- **2-LLM(로봇 두뇌 + 서버 코더) 선택**: 화면 오른쪽에서 **로봇 두뇌**(작은 모델)와 **서버 코더**(큰 모델)를
  각각 고릅니다. 로봇 두뇌가 task 를 받아 "이런 동작을 하는/고치는 컨트롤러를 만들어 줘"를 **자연어**로 서버
  코더에게 보내고, 코더가 `plan(state)` 코드를 만들어 돌려줍니다(역할 분리: 두뇌=플래너, 코더=코딩). 로그에
  보라색 🧠 칸으로 두뇌의 자연어 지시가, 그 아래에 코더가 만든 코드가 보입니다. 로봇 두뇌를 **'(없음)'** 으로
  두면 예전처럼 **단일-LLM**(코더에 직접, 규칙 기반 프롬프트)으로 동작합니다. (코드는 `core/brain.py`,
  CLI 는 `run_sim.py --brain <모델>`.) 실제 모델은 두뇌·코더 각각 NPU 카드를 쓰므로 빈 카드가 2장+ 필요합니다.
- **dp/pp 선택(챗서비스와 동일)**: **서버 코더**의 dp(복제)·pp(레이어분할)를 설정합니다. 챗서비스의 모델 매니저
  (`chat/chat_app.py` 의 CATALOG·ServeManager)를 **그대로 재사용**하므로 같은 카드 회계·dp/pp 규칙을 따릅니다
  (tp8 은 dp×pp ≤ 4장, tp32 는 4장 고정, 일부는 pp 고정). 고른 모델이 안 떠 있으면 **▶ 시작** 때 그 dp/pp 로
  **온디맨드 serve** 하고(상태 배지로 로딩→서빙 표시), 떠 있으면 즉시 재사용합니다.
- **task(House 맵)**: 빨간 컵 확인·복귀 / 파란 컵 확인·복귀 / 노란 우산 확인·복귀(없음) / 집 안쪽 지점까지
  이동(짧은 코드). 물건은 **로봇 카메라가 발견할 때** 비로소 3D 로 나타나고, 끝나면 실제 배치를 공개해
  판정이 맞았는지 보여 줍니다.
- 모델 선택에서 `mock(정상/자가수리)` 을 고르면 서버 없이 **흐름만** 볼 수 있습니다(배관 확인용).
- task 가 끝나면 **결과 정리**(대화 흐름 + 시작↔최종 코드 diff + 결과)가 화면에 나오고, 동시에
  **`robot-sim/results/`** 폴더에 `<시각>_<task>_<모델>.json` 과 `.md`(사람이 읽는 정리)로 저장됩니다.
- ■ 중단을 누르면 즉시 멈춥니다. 할 일 설명의 괄호 속 '정답'은 **관찰자(우리)만** 보며, 로봇/LLM 에는
  찾을 물건 종류(`objective`)만 주고 있는지 여부·위치는 알려주지 않습니다(로봇이 스스로 탐색해 알아냄).

> 참고: 위 "실측" 절대로, 현재 서빙 코더 모델은 긴 코드 생성이 깨지기 쉬워, 물건검색 같은 긴 컨트롤러는
> 실모델로 한 번에 빌드되기 어려울 수 있습니다(그 실패·수리 과정도 화면에 그대로 보입니다). 짧은 코드인
> '집 안쪽 지점까지 이동' task 가 실모델로 성공할 가능성이 가장 큽니다.

---

## 센서는 무엇으로 구현됐나 (AI 모델·장비)

**중요: 센서 쪽에는 AI 모델이 안 들어갑니다.** 이 시뮬의 센서는 전부 **수학(기하)** 이고, AI 모델은 오직
**LLM 2개**(로봇 두뇌 + 서버 코더)뿐입니다.

- **LiDAR** — AI 모델 아님, **거리 측정**. 헤드리스 sim 은 광선-기하(`core/world.py` `raycast`/`lidar`,
  16빔·6m). 실제 TurtleBot3 waffle 장비는 **HLS-LFCD LDS**(360° 라이다, 모델 `hls_lfcd_lds`, Gazebo 에서
  360 samples·3.5m·10Hz, `/scan`) — `turtlebot3_waffle/model.sdf:135~162`.
- **카메라 + YOLO** — **실제로 쓰는 YOLO 모델은 없습니다.** 시뮬 카메라 장비는 **Intel RealSense R200**
  (Gazebo, HFOV 1.02974rad·30Hz·`/camera/image_raw`, `model.sdf:374~401`). 검출(detector)은 ROS2 패키지에
  두 백엔드가 있는데(`turtlebot3_llm_nav/.../perception.py`): 동작하는 건 `GroundTruthDetector`(**모델 없이**
  Gazebo 의 정답 좌표로 검출 흉내 — 오라클), `YoloDetector` 는 **STUB**(주석에 "쓸 거면 ultralytics
  **YOLOv8**(`yolov8n.pt`)" 라고만 적혀 있고 `NotImplementedError`). 헤드리스 sim 은 아예 픽셀 없이 FOV+가림
  기하로 검출(`core/world.py` `scan_view`).
- **오도메트리(위치)** — AI 모델 아님. 실로봇은 바퀴 오도메트리(차동구동) `/odom`, 헤드리스는 정확 적분.
- **IMU** — waffle 에 `tb3_imu`(200Hz) 있음(`model.sdf:48`), 현재 task 엔 미사용.

요약: **LiDAR=거리센서·카메라=RealSense R200(영상검출은 미구현/오라클)·오도메트리=바퀴**, 모두 장비/수학이며
**AI 모델은 LLM 둘뿐**. "카메라가 YOLO 로 사람·물건을 알아본다"는 건 아직 **자리만 비워둔(STUB)** 상태입니다.

---

## 동작 원리 (한 에피소드)

1. **첫 코드 요청** — 현재 위치·목표·LiDAR 값을 적어 LLM에게 `plan(state)` 컨트롤러를 부탁합니다.
   프롬프트에 함수 시그니처와 상태(state) 명세, 제약(장애물은 미리 모름·지역최소 주의)을 줍니다.
   (프롬프트 본문은 코드 생성 신뢰도를 위해 영어로 둡니다 — `prompts.py`.)
2. **코드 적용** — 받은 응답에서 파이썬 코드 블록만 뽑아 제한된 샌드박스에서 실행해 `plan` 함수를
   얻습니다. 빌드(문법·API)가 실패하면 그 자체가 코드 결함이므로 곧장 수리로 넘어갑니다(`executor.py`).
3. **주행** — 그 컨트롤러로 월드를 굴립니다. 매 제어주기마다 `plan(state)` 를 호출해 `(v, w)`
   (선속도·각속도)를 받고, 유니사이클 모델로 로봇을 움직입니다(`world.py`).
4. **실패 시 자가수리** — 충돌·이탈·정체·예외·스텝초과로 멈추면, *어디서 무슨 일이 있었는지*를
   적어 LLM에게 "고쳐 달라"고 다시 요청합니다(`prompts.repair_user`). 로봇은 **그 자리에서** 새
   코드로 이어서 갑니다(처음으로 되돌아가지 않습니다). 목표 도달 또는 재작성 한도까지 반복합니다.

> 로봇이 위치를 알아도 장애물은 모릅니다. 움직이며 얻는 LiDAR(360° 광선거리)만으로 회피해야 하므로,
> 단순히 "목표 쪽으로 직진"하는 코드는 오목한 벽(U자) 같은 **지역최소**에 갇힙니다. 이를 빠져나오는
> 탈출 로직(벽타기·버그 알고리즘 등)을 LLM이 `state['memory']` 를 써서 *코딩*해야 풀립니다.

---

## 시나리오 (난이도별, 코딩의 다른 면을 시험)

| 이름 | 내용 | 보는 능력 |
|------|------|-----------|
| `open`   | 장애물 없음 | 목표추종·각도제어(angle wrap)가 맞는가(기본기) |
| `single` | 정중앙 큰 장애물 1개 | **정면 대칭** 장애물 회피(좌우 반발이 0이 되는 함정) |
| `slalom` | 지그재그 장애물 | 연속 회피 + 목표 복귀 |
| `trap`   | 왼쪽으로 열린 U자 벽 | **지역최소 탈출**(정체 감지 + 우회 코딩) ← 성능차가 가장 큼 |
| `random` | 시드 고정 무작위 장애물밭 | 일반화 |
| `find_person` | **카메라로 '특정 사람'(빨강+모자) 찾기** — 같은 빨강 decoy 오인 유발 | 식별·탐색·실패수리(TurtleBot3 카메라 과제의 헤드리스판, `../turtlebot3-llm/`) |
| `house_search` | **집(TurtleBot3 House) 안을 돌며 '빨간 컵'이 있는지 확인하고 현관 복귀** — 빨간 컵 실제로 있음 | 자율 탐색(경로 없음) + 카메라 스캔 + 복귀 + 판정(present) + decoy 무시 |
| `house_search_absent` | 같은 집인데 **빨간 컵이 없음** — decoy(빨강 책·파란 컵)만 있음 | 끝까지 검색해 **올바로 'absent' 판정**(대충 검색하면 decoy 보고 오판) |

> `house_search*` 는 다른 시나리오와 성격이 다릅니다(아래 별도 절 참고). 일반 시나리오는 한 점→한 점
> 도달이지만, 집 미션은 **여러 방을 돌며 물건을 인식하고 돌아오는** 더 긴 임무라 `--scenario all`(open~random
> 묶음)에는 들어가지 않고 이름으로 따로 돌립니다.

---

## 집(House) 안 물건 확인 후 복귀 미션

`house_search` 는 [emanual 의 TurtleBot3 Gazebo Simulation](https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/#gazebo-simulation)
에 나오는 **House 월드**에서 하는 미션을, 디스플레이 없이 바로 돌려 보는 **헤드리스 판**입니다.
임무는 *"집을 방마다 돌며 특정 물건(빨간 컵)이 있는지 카메라로 확인하고 현관으로 돌아오기"* 입니다.

### 도면은 실제 House 를 그대로 옮긴 것

집 벽은 실제 Gazebo House 모델
(`../turtlebot3-llm/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_house/model.sdf`)의
벽(Wall_*) 21개를 그대로 파싱해 만듭니다(`house_world.py`). 이때 각 벽은 link 의 `<pose>` 와 그 안의
`<collision>` 하위 `<pose>`(길이축 오프셋)를 **합성**해야 제 위치가 됩니다 — 21개 중 11개가 0이 아닌
길이축 오프셋(최대 3.34 m)을 가지므로, 둘을 합쳐야 실제 도면과 맞습니다. 좌표는 (+8, +6) 평행이동해
16×12 m 평면에 올리고, 현관(home)은 실제 스폰 기본값 (-2.0, -0.5) → (6.0, 5.5) 입니다
(출처: `turtlebot3_gazebo/launch/turtlebot3_house.launch.py` 의 `x_pose`/`y_pose` 기본값).

### 자율주행 — 미리 정한 경로 없음

이 미션은 **자율주행**입니다. 미리 짜 둔 이동 경로(waypoints)를 주지 않습니다 — 로봇은 LiDAR 와
카메라만으로 **스스로 집을 돌아다니며** 물건을 찾고 현관으로 돌아와야 합니다.

- LLM 이 코딩하는 `plan(state)` 컨트롤러가 **탐색 + 스캔 + 복귀 + 판정**을 모두 합니다. state 에는
  `lidar`·`scan`(지금 보이는 물건들)·`home`(현관 좌표)·`objective`(찾는 물건)·`memory` 만 들어오고
  **경로(waypoints)는 없습니다**. 끝나면 `{'done': True, 'present': bool}` 를 돌려줍니다.
- 기본 예시 컨트롤러(`prompts.py` 의 `HOUSE_SCAFFOLD`)는 **LiDAR 벽추종(wall-following)으로 방들을
  자율 탐색**하고(문을 알아서 통과), 지나온 자리를 **빵부스러기(breadcrumb)** 로 기억해 두었다가 그걸
  역추적해 현관으로 안전하게 복귀합니다 — 미리 준 지도가 없어도 됩니다.
- anti-guess 게이트는 컨트롤러의 자기보고가 아니라 **로봇이 집의 닿는 영역을 실제로 얼마나
  돌아다녔는지(coverage)** 를 에이전트가 독립 측정해, '대충 보고만으로 absent' 같은 추측을 막습니다.

### 헷갈리게 놓은 물건(decoy)과 자가수리

진짜 목표는 **빨간 컵** 하나이고, **빨간 책**(색만 같음)·**파란 컵**(라벨만 같음)을 decoy 로 둡니다.
그래서 *색만 보고* 판정하면 틀립니다(false_report). 실패하면 무엇이 잘못됐는지 적어 NPU LLM 에게
코드를 고쳐 달라고 다시 요청합니다(self-debug). 실패 유형은 다음과 같습니다:

| 실패 | 뜻 |
|------|------|
| `missed_object` | 물건이 있는데 'absent' 라 단정(끝까지 안 찾음) |
| `false_report` | 물건이 없는데 'present' 라 단정(decoy 를 보고 오인) |
| `not_home` | 현관에 돌아오지 않고 미션 종료를 선언 |
| `no_report` | 끝까지 돌았는데 종료(`done`) 선언을 안 함 |
| `searched_too_little` | **제대로 찾지도 않고** 판정을 내림(추측) |
| `stuck` / `collision` | 정체 / 벽 충돌 |

### 추측으로는 통과 못 함 (anti-guess 게이트)

로봇은 현관에서 시작하므로, "그냥 `done, present=False` 를 내면" 검색 없이도 정답(absent)을 맞힐 위험이
있습니다. 이를 막으려고 **에이전트가 독립적으로** 두 가지를 추적합니다(`agent.py`):

- `present` 판정은 로봇이 **진짜로 목표 물건을 카메라로 본 적이 있을 때만** 인정합니다.
- `absent` 판정은 로봇이 **경로 웨이포인트의 60% 이상을 실제로 방문했을 때만** 인정합니다.

둘 다 컨트롤러의 자기보고가 아니라 시뮬레이터가 직접 센 값이라, 추측으로는 통과할 수 없습니다. 실제로
"항상 present" · "항상 absent" 컨트롤러는 모두 `searched_too_little` 로 실패함을 확인했습니다.

### 돌려 보기

```bash
cd Model_Benchmark/rngd-npu/robot-sim

# 콘솔: 빨간 컵 찾기(있음) / 없는 경우 / 버그→자가수리
python3 run_sim.py --mock good  --scenario house_search          # 있음 → present 로 성공
python3 run_sim.py --mock good  --scenario house_search_absent   # 없음 → absent 로 성공
python3 run_sim.py --mock buggy --scenario house_search          # '검색 없이 없다' 버그 → 수리 → 성공

# 브라우저: 집 도면·방·문·물건·카메라 화각·경로·궤적·판정을 움직이는 애니메이션으로
python3 web_sim.py --mock good --scenario house_search --serve 7900
#  → 7900 을 터널로 포워딩하고 http://127.0.0.1:7900/ 접속

# 실제 NPU 코더 모델로(먼저 ../chat/serve_models.sh coder7)
../chat/.venv/bin/python run_sim.py --model coder7 --scenario house_search
```

> 실모델 주의: 집 미션 컨트롤러는 일반 시나리오보다 길어서, 아래 "실측" 절의 **긴 코드 생성 손상**
> 한계가 더 잘 드러납니다. 그래서 폐루프·도면·판정 자체의 검증은 mock 으로 하고(위), 실모델 검증은
> 서빙 측 코드 생성 품질이 좋아지면 코드 변경 없이 바로 됩니다. 같은 미션을 실제 Gazebo House 에서
> 돌리는 ROS2 패키지는 `../turtlebot3-llm/` 에 있습니다.

---

## 측정 지표

### 코딩 성능 (핵심)

- **success / success_rate** — 로봇이 목표에 도달했는가(코드가 임무를 완수했는가).
- **first-try success** — 수리 없이 첫 코드로 성공한 횟수(한 번에 맞히는 능력).
- **avg replans** — 목표까지 코드를 평균 몇 번 고쳐 썼는가(적을수록 좋음).
- **code_valid_first** — 첫 코드가 문법·API를 지켜 바로 실행됐는가.
- **collisions / exceptions** — 주행 중 충돌·런타임 예외 횟수.
- **path-eff** — (직선거리 / 실제주행거리). 1.0 에 가까울수록 효율적 경로.

### 연동(미들웨어) 성능 — SOAFEE/ROS2 분석용

- **hop_sense_plan / plan_compute / hop_plan_ctrl (ms)** — perception→planning→control 홉별 지연.
- **cycle Hz** — 제어 한 사이클의 주파수.
- **TTFT / TPS / tokens** — LLM 코드 생성·수리의 지연·처리량(첫 토큰까지 시간, 초당 토큰).
  실서버 모드에서 `furiosa-llm serve` 가 주는 정확한 토큰 수(usage)를 사용합니다.

`--report out.json` 으로 위 지표를 JSON 으로 저장해 모델 간 비교에 씁니다.

---

## 실측: 실제 NPU 모델로 돌려본 결과 (2026-06-18)

폐루프 전체가 **실제 RNGD NPU 서버를 상대로 끝까지 동작**하는 것을 확인했습니다: 모델에 코드 요청 →
스트리밍 응답 수신(TTFT·TPS 측정) → 코드 블록 추출 → 샌드박스 빌드 → 월드 롤아웃 → 실패 시 자가수리
→ 지표 집계까지, 각 단계를 살아 있는 서버(coder7·coder14·a3b-fp8)로 검증했습니다.

다만 **지금 furiosa-llm serve 로 띄운 코더 모델들은 길고 여러 줄짜리 코드를 생성할 때 출력이 깨집니다.**
짧은 코드(예: `return 0.5 * 2.0 + 1.25`)는 세 모델 모두 완벽하지만, `plan()` 같은 10줄 이상 코드를
만들면 대략 10~14번째 줄부터 다음과 같은 손상이 누적됩니다(temperature 0 에서도, 동시요청 없이도 재현):

- 들여쓰기 드리프트(4칸 → 5·6칸으로 밀림 → `IndentationError`/`TabError`)
- 유니코드 오치환: 전각 숫자 `atan２`, 스마트따옴표 `"`(U+201C)
- 빈/잘못된 숫자: `return {'v': . , 'w': . }`, `2.0` → `123`, `**( )`
- 토큰 중간 공백: `state ['pos']`, `v _max`, 식별자 중간 띄어쓰기

이 패턴이 **coder7(7B)·coder14(14B 밀집)·a3b-fp8(30B MoE, 활성 3B)에서 똑같이** 나타나므로, 특정 모델
크기 문제라기보다 **이 서빙 스택의 디코딩 단계가 긴 코드의 공백·특수 토큰을 잘 못 다루는** 쪽에
가깝습니다(짧은 출력은 멀쩡하고, 긴 코드일수록 심해지는 점이 근거). 채팅(산문)에서는 들여쓰기에
민감하지 않아 잘 안 드러나고, 들여쓰기가 곧 문법인 파이썬 코드에서 드러납니다.

그래서 **현재 스택에서는 한 번에 빌드되는 완전한 컨트롤러를 실모델로 받기가 어렵습니다.** 폐루프
자체의 검증은 위 mock(`--mock good` 5/5 도달, `--mock buggy` 자가수리)으로 하시고, 실제 모델 검증은
이 서빙 측 코드 생성 품질이 개선되면(또는 더 깨끗하게 생성하는 서빙/모델로 바꾸면) 코드 변경 없이
바로 됩니다. 시뮬레이터의 프롬프트는 이 한계를 감안해 **짧은 코드 + 실패 유형별 점진적 수리**로
설계해, 모델이 잘하는 '짧고 깨끗한 코드' 쪽으로 유도합니다(`prompts.py`).

> 이 자체가 과제에 쓸모 있는 결과입니다 — "로봇을 코딩으로 움직이려면 서빙 스택이 **긴 코드도
> 손상 없이** 생성해야 한다"는 요구사항을 구체적으로 짚어 주기 때문입니다.

---

## 과제(SDV / SOAFEE / ROS2)와의 연결

대상 과제는 **"SDV와의 연동을 위한 SOAFEE/ROS2 기반 미들웨어 연동 분업 기술 구현"** 으로,
- 자율주행소프트웨어(Autoware) 분업을 위한 **컴포넌트 분리 및 컨테이너화**
- 컨테이너 간 연동에 따른 **성능 분석** 및 ROS2 기반 미들웨어 **최적화**
- End-to-End 자율주행에 대한 컨테이너 분리 **성능 분석**

을 다룹니다. 이 저장소에서 맡은 역할은 그중 **LLM 서비스**입니다. 시뮬레이터는 이 역할을 다음과
같이 자율주행 파이프라인에 끼워 넣어 검증합니다(`middleware.py`).

| Autoware/ROS2 개념 | 시뮬레이터 구현 |
|--------------------|-----------------|
| perception 노드 | `PerceptionNode` — LiDAR+pose 관측을 `/sensors` 로 발행 |
| **planning 노드(= 우리 LLM 서비스)** | `PlanningNode` — LLM이 짠 `plan(state)` 컨트롤러를 실행해 `/cmd_vel` 발행 |
| control 노드 | `ControlNode` — `/cmd_vel` 을 받아 구동 입력 `(v, w)` 확정 |
| ROS2 미들웨어(DDS) | 인프로세스 pub/sub `Bus` |
| 컴포넌트 분리/컨테이너화 | `--middleware threaded` — 각 노드를 별도 스레드+큐로 분리(컨테이너/프로세스 격리의 1차 근사) |
| 컨테이너 간 연동 성능 | 홉별 지연(ms)·사이클 Hz 측정 |

`--middleware sync` 는 한 프로세스 직렬 실행(전송지연 ≈ 0, 구조만 확인), `--middleware threaded` 는
노드를 스레드로 떼어 **진짜 큐잉/문맥전환 지연**을 측정합니다. 실제 컨테이너(ROS2 `rclpy` + DDS,
또는 SOAFEE 분리 배포)로 바꿀 자리는 `middleware.py` 의 `_planning_loop`/`_control_loop` 큐 입출력
지점입니다(여기를 rclpy publisher/subscriber 로 교체).

> 정리하면, 이 시뮬레이터는 **"planning 컨테이너를 코딩 LLM으로 구현했을 때, End-to-End로 목표에
> 도달하는가"** 와 **"컨테이너를 분리하면 연동 지연이 얼마나 드는가"** 를 함께 확인하는 발판입니다.

---

## 파일 구성

폴더는 **로봇이 기본으로 가진 코드(`core/`)** 와 **그걸 구동하는 실행 앱(최상위)** 로 나뉩니다.
실행 앱은 `core/` 를 import 경로에 넣어(각 앱의 `sys.path.insert(... "core")`) 그대로 돌아갑니다 —
실행 명령은 예전 그대로(`python3 run_sim.py …`, `… live_sim.py --port …`)입니다.

**`core/` — 로봇 기본 코드(task 받기 전부터 가진 이동·인지·폐루프·예시 컨트롤러)**

| 파일 | 역할 |
|------|------|
| `core/world.py` | 2D 연속 월드 · LiDAR 광선추적 · 유니사이클 로봇 이동 · 카메라 · 충돌 (로봇의 '움직이는 기능') |
| `core/executor.py` | LLM 코드 추출 · 제한 샌드박스 실행 · plan() 시간제한 호출 |
| `core/agent.py` | 폐루프 오케스트레이션(observe→plan→step) + 자가수리(에피소드 실행) |
| `core/prompts.py` | 코드 생성/수리 프롬프트 + **기본 예시 컨트롤러(`HOUSE_SCAFFOLD` 등)** |
| `core/llm_client.py` | NPU 서버(OpenAI 호환) 클라이언트 + 가짜 LLM(good/buggy) |
| `core/house_world.py` | 실제 House SDF 벽 파싱 → 2D 도면 + 물건 배치(집 미션) |
| `core/scenarios.py` | 난이도별 시나리오(open/single/slalom/trap/random/find_person/house_search) |
| `core/middleware.py` | ROS2식 노드 분리(perception/planning/control) + 홉 지연 측정 |
| `core/metrics.py` | 에피소드/배치 지표 · 표·JSON 리포트 |
| `core/viz.py` | ASCII 경로 그림(항상) + PNG(matplotlib 있을 때) |
| `core/sim_record.py` | 에피소드를 돌리며 로봇 자세·LiDAR 프레임 기록(두 뷰어 공용) |

**최상위 — 실행 앱(엔트리 포인트)**

| 파일 | 역할 |
|------|------|
| `run_sim.py` | CLI 러너(모델·시나리오·미들웨어 선택, 리포트) |
| `web_sim.py` | 브라우저 캔버스 애니메이션 뷰어(에피소드 녹화 후 재생, 헤드리스 OK) |
| `live_sim.py` | **라이브 3D 인터랙티브 서버** — Three.js 3D 집 + 챗서비스식 모델·dp/pp 선택(CATALOG·ServeManager 재사용) + task 선택 시 NPU 요청·코드수정 과정 실시간 스트리밍(FastAPI/SSE) |
| `turtle_sim.py` | turtle 창 뷰어(디스플레이 있는 PC에서 실제 창) |
| `vendor/three/` | 로컬 번들 three.js(0.160) + OrbitControls — 라이브 3D 뷰가 CDN 없이 뜨도록 |
| `results/` | 라이브 앱이 task 종료 시 결과 정리(JSON+.md)를 저장하는 폴더 |

---

## 안전 / 주의

이 도구는 **LLM이 생성한 파이썬 코드를 실행**합니다. `executor.py` 가 내장함수 화이트리스트와
`import` 제한(math 만 허용)으로 막아 두지만, 임의 코드 실행은 본질적으로 위험합니다. **신뢰
가능한 로컬 연구용 시뮬레이터**로만 쓰고, 외부에서 받은 신뢰할 수 없는 응답을 그대로 넣지 마세요.

---

## 출처

- OpenAI 호환 호출 방식(base_url·api_key·스트리밍·usage 토큰)은 같은 저장소의 챗 서버와 동일합니다:
  `Model_Benchmark/rngd-npu/chat/chat_app.py:684-733`, `chat/npu_metrics.py`.
- 모델 포트·키 매핑은 `chat/chat_app.py` 의 CATALOG(73~110행)와 `chat/serve_models.sh` 를 따릅니다.
- 과제 주제 문구는 "SDV와의 연동을 위한 SOAFEE/ROS2 기반 미들웨어 연동 분업 기술 구현" 및 그 세부
  과제(Autoware 컴포넌트 분리·컨테이너화, 컨테이너 간 연동 성능 분석, End-to-End 자율주행 분석)를
  그대로 옮긴 것입니다.
- 광선-원/광선-사각경계 교차, 유니사이클(차동구동) 운동학, 포텐셜필드·버그 알고리즘은 로보틱스
  표준 계산기하·경로계획 기법입니다.
