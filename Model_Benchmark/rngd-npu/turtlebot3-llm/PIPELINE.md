# 카메라로 '특정 사람' 찾기 — 로봇이 NPU LLM과 대화하며 코드를 고쳐 목표를 달성하는 파이프라인

이 문서는 다음을 자세히 설명합니다.

> TurtleBot3(카메라 장착, waffle)에게 **"빨간 옷에 모자 쓴 사람에게 가"** 처럼 *특정 사람*에게 도달하라고
> 시켰을 때, 단순 센서 주행으로는 **실패가 잦습니다**(엉뚱한 사람, 가려짐, 놓침 등). 이때 로봇이
> 우리 **RNGD NPU의 코딩 LLM 서버**와 **대화를 주고받으며 자기 제어 코드를 고쳐**, 결국 맞는 사람에게
> 도달하는 전체 파이프라인.

구현은 두 층으로 되어 있습니다.
- **진짜 TurtleBot3**: ROS2 패키지 `turtlebot3_llm_nav/` (Gazebo + waffle 카메라에서 실행).
- **헤드리스 검증판**: `../robot-sim/` 의 `find_person` 시나리오 (디스플레이 없는 서버에서 지금 돌려 보고
  브라우저로 볼 수 있는 같은 폐루프). 둘은 **똑같은 `plan(state)` 계약**을 공유해, 헤드리스에서 검증한
  파이프라인을 그대로 실로봇으로 옮길 수 있습니다.

출처: 위 토픽·카메라 사양은 클론한 [robotis-git/turtlebot3_simulations](https://github.com/robotis-git/turtlebot3_simulations)
의 `turtlebot3_waffle/model.sdf`·`spawn_turtlebot3.launch.py` 에서 확인했습니다. 폐루프 코딩 하니스는
`../robot-sim/` 에서 가져왔습니다.

> 이 문서는 **사람찾기** 임무를 자세히 설명합니다. 같은 폐루프 위에 올린 두 번째 임무인 **집 안 물건
> 확인 후 복귀**(House 월드)는 맨 아래 §10 에 있습니다.

---

## 1. 한눈에 보는 구조

```
        ┌──────────────────────── 로봇(TurtleBot3 waffle) ────────────────────────┐
        │                                                                          │
  /camera/image_raw ─┐                                                             │
  /scan ─────────────┤  Perception   ──►  state(관측)  ──►  plan(state)  ──►  /cmd_vel ─► 구동
  /odom ─────────────┘  (사람 검출)        (dict)            (LLM이 짠 코드)                  │
        │                                       ▲                  │                        │
        └───────────────────────────────────────┼──────────────────┼────────────────────────┘
                                                 │                  │ 실패 감지
                              ┌──────────────────┴──────────────────▼─────────────────┐
                              │   "무슨 일이 있었는지"를 적어 코드 수리를 요청          │
                              │   (carmera에 뭐가 보였나 / 왜 실패했나)                 │
                              ▼                                                        │
                  ┌────────────────────────────┐    OpenAI 호환 HTTP    ┌─────────────┴───┐
                  │  RNGD NPU · furiosa-llm     │ ◄──────────────────►  │  LLM 코딩 대화   │
                  │  serve (coder LLM)          │   /v1/chat/...        │  (생성·수리)     │
                  └────────────────────────────┘                       └─────────────────┘
```

핵심 아이디어: **로봇의 '행동'을 사람이 미리 짠 고정 알고리즘이 아니라, LLM이 그 자리에서 짜 주고
실패하면 LLM과 대화로 고치는 코드**로 둡니다. 그래서 가장 중요한 게 **코딩 성능**입니다.

---

## 2. 제어 한 주기(control cycle) — 실로봇 기준

`turtlebot3_llm_nav` 노드는 약 10 Hz로 다음을 반복합니다(코드: `llm_nav_node.py`).

1. **관측 수집**: 최신 토픽으로 `state`(dict)를 만듭니다.
   - `/odom` → `pos`, `heading`
   - `/scan` → `lidar`(거리), `lidar_angles`(로봇 기준 각). 다운샘플해 LLM이 보기 쉽게.
   - `/camera/image_raw` → **사람 검출** `camera` = `[{bearing, distance, features, conf}, ...]`
     (Perception 모듈이 만듭니다 — 3절).
   - `target`(찾을 사람 특징, 예 `{"shirt":"red","cap":true}`) + `memory`(주기 간 유지되는 내부 상태).
2. **컨트롤러 실행**: LLM이 짠 `plan(state)` 를 시간제한 안에서 호출 → `{'v','w'}` 를 받습니다.
3. **구동**: `(v, w)` 를 클램프해 `/cmd_vel`(이 빌드는 `TwistStamped`)로 발행. **안전 게이트**: 정면
   LiDAR가 위험거리 미만이면 코드와 무관하게 강제 정지.
4. **실패 감지**: 충돌·정체·놓침·오인·예외를 감시(5절). 실패면 **수리 루프**로.

> LLM 호출(코드 생성/수리)은 무겁고 가끔이라, 제어 타이머와 **다른 스레드**에서 돕니다. 생성 중엔
> 로봇을 멈춰 두고, 새 코드가 준비되면 이어서 갑니다. (제어 주기는 절대 안 막힘.)

---

## 3. 왜 '특정 사람에게 가'가 어려운가 — 그리고 perception

장애물 회피하며 *좌표*로 가는 건 LiDAR로 쉽습니다. 하지만 **카메라로 '특정 사람'을 찾아가는 건**
다음 때문에 실패가 잦습니다. 이게 이 과제의 본질입니다.

| 어려움 | 구체 상황 | 결과 실패 |
|--------|-----------|-----------|
| **식별 모호성** | 빨간 옷 입은 사람이 둘 — 한 명만 target인데 옷 색만 보면 구분 못 함 | 엉뚱한 사람 도착(`wrong_person`) |
| **부분 관측** | target이 카메라 화각(±~29°) 밖이거나 다른 사람·벽에 가려 안 보임 | 목표 놓침(`lost_target`) |
| **외형 변화** | 거리·조명에 따라 특징 신뢰도(conf)↓ — 멀면 모자 유무가 안 보임 | 성급한 오인 |
| **탐색 필요** | 처음엔 아무도 안 보임 → 방을 둘러봐야 함. 가만있으면 영영 못 찾음 | 정체(`stuck`/`lost_target`) |
| **동적/군중** | 사람이 움직이거나 여럿이 겹침 | 추적 실패 |

**Perception(`perception.py`)** 은 카메라 프레임을 위 `camera` 검출 목록으로 바꾸는 모듈이며, 두 백엔드를
**같은 인터페이스**(`detect() -> [{bearing,distance,features,conf}]`)로 제공합니다.

- **`GroundTruthDetector`(시뮬레이션 오라클)**: Gazebo의 사람 위치를 `/people_ground_truth`(JSON) 로 받아
  로봇 좌표계로 변환하고, **카메라 화각 안**에 든 사람만 검출로 내보냅니다(bearing/distance 정확, conf는
  거리·각도로). → **실제 비전 검출기 없이도 전체 파이프라인을 Gazebo에서 돌려 볼 수 있게** 해 줍니다.
- **`YoloDetector`(실검출기 자리)**: YOLO/얼굴인식 같은 실제 모델을 끼우는 스텁. 픽셀→bearing(HFOV
  1.02974 rad), bbox 높이→거리(사람 1.7 m) 변환식은 이미 들어 있고, 모델 추론만 채우면 됩니다.

> 즉 perception은 "누가 어디 보이고 어떤 특징인가"까지만 줍니다. **"그 중 누구가 target이고 어떻게 갈
> 것인가"는 LLM이 짠 `plan` 의 몫**입니다 — 바로 그 판단 코드를 대화로 고치는 게 이 파이프라인입니다.

---

## 4. `plan(state)` 계약 — LLM이 짜는 것

LLM은 매번 **짧은 파이썬 함수 하나**를 돌려줍니다(프롬프트: `prompts.py`).

```python
def plan(state):
    # state['camera']  : 지금 보이는 사람들 [{bearing, distance, features, conf}, ...]  (화각 밖/가려진 사람은 없음)
    # state['target']  : 찾을 사람 특징 dict, 예 {'shirt':'red','cap':True}
    # state['lidar'], state['lidar_angles'] : 장애물 회피용
    # state['memory']  : 주기 간 유지(검색 방향, 카운터 등)
    # 좌표(goal)는 안 줍니다 — 카메라로 찾아야 합니다.
    ...
    return {'v': 선속도, 'w': 각속도}
```

이 코드는 **샌드박스**(`executor.py`)에서 실행됩니다: 던더(`__...__`) 접근·위험 내장함수 차단, 빌드/호출
시간제한, `math` 만 허용. (LLM이 짠 코드를 그대로 로봇에 적용하므로 안전장치가 필수입니다. 단 이는
신뢰 가능한 로컬·자기 생성 코드 전용 가드이지 완전한 보안 경계는 아닙니다.)

---

## 5. 실패 → 대화 → 코드 수정 → 달성 (이 문서의 핵심)

노드는 각 실패를 **유형**으로 분류하고, 유형에 맞는 **수리 프롬프트**로 LLM과 대화합니다. 한 번에 다
시키지 않고 **딱 필요한 만큼만** 보강을 요청해, 작은 coder 모델도 **짧고 깨끗한 코드**를 내게 유도합니다
(긴 코드를 생성하면 현재 NPU serve 스택이 출력을 깨뜨리는 실측 한계가 있어서입니다 — 9절).

### 실패 유형 → 수리 매핑

| 실패(reason) | 어떻게 감지 | LLM에게 주는 수리 지시(요지) |
|--------------|-------------|------------------------------|
| `wrong_person` | target과 **일부만** 맞는 사람에 근접·정착 | "target의 **모든** 특징을 비교해라. 옷색만 같은 가짜는 무시/장애물 취급." |
| `lost_target` | target이 한 번도 화각에 안 들어옴 + 탐색 멈춤 | "안 보이면 한 방향으로 계속 회전하며 새 위치로 전진해 방을 훑어라(메모리에 방향 저장)." |
| `stuck` | 진전 없이 제자리 맴돔 | "안 보이면 탐색, 보이면 LiDAR로 장애물 피하며 접근." |
| `collision` | 정면 LiDAR가 위험거리 미만 | "정면 광선이 ~1.5 m 미만이면 감속하고 트인 쪽으로 틀어라." |
| `exception` | `plan` 이 예외/NaN 반환 | "빈 camera·없는 key를 .get으로 방어하고 항상 dict를 반환해라." |
| `no_valid_code` | 코드 빌드 실패(문법/들여쓰기) | "4칸 들여쓰기로 깔끔히 다시. 반드시 dict 반환." |

### 실제 대화 예시 (find_person 시나리오)

이 시나리오엔 **빨간 옷 사람이 둘**입니다 — target(빨강+모자), decoy(빨강+모자X, 더 가까이 정면).

**① 초기 요청 (로봇 → LLM)**
```
[system] 너는 카메라+LiDAR 로봇의 plan(state) 컨트롤러를 짠다. target 특징과 일치하는 '특정 사람'에게
         가라. camera 검출에서 target의 모든 특징이 맞는 사람만 진짜 목표다. … (짧게, 4칸 들여쓰기)
[user]   Robot at (3.0,12.0), heading 0.0. TARGET = {'shirt':'red','cap':True}.
         Camera sees now: [bearing +0deg, 7.0m, {'shirt':'red','cap':False}, conf 0.7]
                          [bearing +20deg, 14.9m, {'shirt':'red','cap':True}, conf 0.3] …
         Write plan(). Keep it short.
```

**② LLM 응답 — 첫 코드 (흔한 실수: 옷색만 봄)**
```python
def plan(state):
    cam = state['camera']
    reds = [p for p in cam if p['features'].get('shirt') == state['target']['shirt']]
    p = (reds or cam)[0]                 # ← 가장 정면의 '빨강'에게 직진 = decoy
    return {'v': 0.6*state['v_max'], 'w': 1.8*p['bearing']}
```
→ 로봇이 정면의 **decoy(빨강·모자X)** 에게 다가가 멈춥니다.

**③ 실패 감지 (로봇)**: decoy 옆에 2초 이상 머묾 → `wrong_person`.

**④ 수리 요청 (로봇 → LLM)** — 무슨 일이 있었는지 + 유형별 지시:
```
[user] Your previous plan() did NOT reach the correct target person.
       Outcome: wrong_person. The robot stopped at the WRONG person (features {'shirt':'red','cap':False});
       the target must match ALL of {'shirt':'red','cap':True}.
       Robot at (10.0,12.0)… Camera sees now: [+0deg,{'shirt':'red','cap':False}], [+25deg,{'shirt':'red','cap':True}]…
       Hint: compare ALL keys in target against each detection's features; only approach a FULL match.
       Treat look-alikes (same shirt, no cap) as obstacles/ignore. Output the corrected plan().
       <이전 코드 첨부>
```

**⑤ LLM 응답 — 고친 코드 (전체 특징 일치 + 안 보이면 탐색)**
```python
def plan(state):
    cam, tgt, mem = state['camera'], state['target'], state['memory']
    match = None
    for p in cam:
        f = p.get('features', {})
        if all(f.get(k) == v for k, v in tgt.items()):   # ← 모든 특징 일치만 진짜 목표
            match = p; break
    if match:
        return {'v': 0.6*state['v_max'], 'w': 2.0*match['bearing']}
    side = mem.get('side', 1); mem['side'] = side          # ← 안 보이면 한 방향 탐색
    return {'v': 0.0, 'w': side*0.7*state['w_max']}
```
→ 이제 decoy를 무시하고 **빨강+모자 target** 에게 접근 → **목표 달성**.

이 한 번의 왕복(요청→실행→실패→수리→실행)이 폐루프의 한 사이클입니다. 다른 유형(가려짐 →
`lost_target` → "탐색 추가", 충돌 → "회피 강화")도 같은 방식으로 **대화로 코드를 키워** 갑니다.
재작성 횟수에 상한을 둬 무한루프를 막습니다.

> **검증(헤드리스, 지금 실행 가능):**
> `cd ../robot-sim && python3 run_sim.py --mock buggy --scenario find_person`
> → `wrong_person`(step ~82) → 수리 1회 → **목표 도달**. 위 ②~⑤가 실제로 재현됩니다.
> 브라우저로 보기: `python3 web_sim.py --mock buggy --scenario find_person --serve 7900`.

---

## 6. '코딩 성능'이 관건인 이유 + 측정

이 파이프라인의 성패는 결국 **LLM이 (a) 한 번에 맞는 컨트롤러를 짜는가, (b) 실패 설명을 읽고 스스로
고쳐 성공시키는가**에 달려 있습니다. 그래서 지표를 그렇게 잡습니다(헤드리스 하니스 `metrics.py`):

- `success` (임무 완수), `first-try`(수리 없이 성공), `replans`(고친 횟수), `code_valid_first`(첫 코드가
  빌드됐나), `wrong_person`/`lost_target`/`collision` 횟수, LLM의 TTFT·TPS·토큰.

모델별로 이 숫자를 비교하면 **어떤 코더 모델이 로봇을 더 잘 '코딩으로' 운전하는지**를 정량 비교할 수 있습니다.

---

## 7. 컨테이너 분리(SOAFEE / ROS2 / Autoware)와의 연결

과제 주제(SDV·SOAFEE·ROS2 미들웨어 분업)와 이렇게 맞물립니다.

| Autoware/ROS2 개념 | 이 파이프라인 |
|--------------------|---------------|
| perception 컨테이너 | 카메라 사람검출(`perception.py`) — `/camera`→검출 |
| **planning 컨테이너(= 우리 LLM 서비스)** | LLM이 짠 `plan(state)` 를 실행하는 노드 — '특정 사람' 판단·접근 |
| control 컨테이너 | `(v,w)`→`/cmd_vel` 구동 |
| ROS2 미들웨어(DDS) | 노드 간 토픽 |
| **LLM 코드 생성/수리 컨테이너** | RNGD NPU의 `furiosa-llm serve` (OpenAI 호환) |
| 컨테이너 간 연동 성능 분석 | 토픽 홉 지연 + LLM 수리 지연(TTFT/TPS)을 함께 측정 |

즉 이 작업은 **"planning 컨테이너를 *LLM이 코딩으로* 구현하고, 실패 시 NPU와의 대화로 그 코드를
고쳐 가는" 분업 구조**를 구체화한 것입니다. 헤드리스 하니스의 `--middleware threaded` 가 perception→
planning→control 노드 분리·홉 지연을 측정해 컨테이너화 성능 분석으로 이어집니다.

---

## 8. 실로봇으로 옮기기 (요약)

1. ROS2 Jazzy + Gazebo Harmonic + TurtleBot3 가 있는 PC(디스플레이 필요)에서 upstream과 이 패키지를 빌드.
2. NPU 서버에서 코더 모델 serve: `./chat/serve_models.sh coder7` (또는 coder14/coder32 등).
3. 사람(actor)이 있는 월드로 Gazebo+waffle 기동(예시: `turtlebot3_llm_nav/worlds/`).
4. `ros2 launch turtlebot3_llm_nav llm_person_nav.launch.py target:='{"shirt":"red","cap":true}' llm_port:=8002`
5. 헤드리스에서 검증한 것과 **같은 `plan(state)`·같은 수리 루프**가 실로봇에서 돕니다.

자세한 빌드/실행은 `turtlebot3_llm_nav/README.md` 참고.

---

## 9. 솔직한 한계 (실측)

- **현재 NPU serve의 코더 모델들은 긴 다중행 코드를 생성하면 출력이 깨집니다**(들여쓰기 드리프트·전각
  숫자·빈 숫자 등 — `../robot-sim/README.md` "실측 결과" 참고). 그래서 이 파이프라인의 프롬프트는 의도적으로
  **짧은 코드 + 실패유형별 점진 수리**로 설계했습니다(모델이 잘하는 짧고 깨끗한 코드 쪽으로). 이 서빙
  품질이 개선되면 더 복잡한 컨트롤러도 한 번에 받을 수 있습니다.
- **헤드리스 검증판은 '오라클' 검출**(사람 위치를 알고 화각·가림만 시뮬)을 씁니다 — 실제 카메라 인식의
  오류(오검출·미검출·외형변화)는 `YoloDetector` 에 실제 모델을 끼워야 완전히 재현됩니다. 단, 식별 모호성·
  가려짐·탐색 같은 **행동/판단 차원의 실패와 그 코드 수리 루프**는 헤드리스에서 그대로 검증됩니다.
- 이 서버는 ROS2/Gazebo 미설치(헤드리스)라 실 Gazebo 실행은 다른 PC가 필요합니다. 그래서 **개념·코드는
  완성**돼 있고, **즉시 검증은 헤드리스판**으로, **실행은 ROS2 PC**로 나눠 두었습니다.

---

## 10. 또 하나의 임무: 집 안 물건 확인 후 복귀 (object_search)

위 사람찾기와 **같은 폐루프·같은 `plan(state)` 계약** 위에, 더 긴 임무를 하나 더 올렸습니다:
[emanual 의 TurtleBot3 Gazebo **House**](https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/#gazebo-simulation)
에서 *"집을 방마다 돌며 특정 물건(빨간 컵)이 있는지 확인하고 현관으로 돌아오기"*.

```
전역 플래너(주어짐, Nav2 역할)            로컬 컨트롤러(LLM이 코딩)
  방 순회 웨이포인트 경로  ──►  plan(state): 경로 추종 + 카메라 스캔 + 현관 복귀 + present/absent 판정
                                      │ 끝나면 {'done':True,'present':bool}
        실패 감지 ◄──────────────────┘
   (missed_object / false_report / not_home / no_report / searched_too_little / stuck / collision)
        │ 수리 프롬프트
        ▼  RNGD NPU 코딩 LLM 과 대화로 코드 수정 → 다시 시도
```

- **planning ↔ control 분담:** 방을 도는 경로는 *주어집니다*(전역 플래너 역할 — 실제 시스템에선 Nav2).
  로봇(LLM)은 그 경로를 따라가는 **로컬 컨트롤러**를 코딩합니다. Autoware 의 planning↔control 분리와
  같은 구조라, 이 과제의 "컴포넌트 분리" 관점에도 맞습니다.
- **헷갈리게 놓은 물건(decoy):** 진짜 목표는 빨간 컵 하나, decoy 로 빨간 책(색만 같음)·파란 컵(라벨만
  같음)을 둡니다. *색만 보고* 판정하면 틀립니다(false_report) → 모든 특징이 일치할 때만 목표로 셈.
- **추측 방지(anti-guess):** 로봇이 현관에서 시작하므로 "그냥 없다고 단정"하면 검색 없이 정답을 맞힐 수
  있습니다. 그래서 노드/에이전트가 **직접** 추적합니다 — `present` 는 진짜로 목표를 본 적이 있을 때만,
  `absent` 는 경로의 60% 이상을 실제로 방문했을 때만 인정합니다. 추측 컨트롤러는 `searched_too_little`
  로 실패합니다(헤드리스에서 실측 확인).

**구성(사람찾기와 같은 2층):**
- **실 TurtleBot3(ROS2):** 노드 `object_search_node.py`, 물체검출 `object_perception.py`(GT 오라클
  `/objects_ground_truth` + YOLO STUB), 프롬프트 `house_prompts.py`, 런치 `llm_house_search.launch.py`,
  월드 셋업 `worlds/objects.snippet.sdf`·`worlds/objects.README.md`.
- **헤드리스 검증판:** `../robot-sim` 의 `house_search` / `house_search_absent`. 도면은 실제 House 의
  `model.sdf` 벽을 그대로 파싱해 만든 것이라 방·문 배치가 같고, 브라우저로 바로 봅니다
  (`python3 web_sim.py --mock good --scenario house_search --serve 7900`).

실행/셋업 자세히는 `turtlebot3_llm_nav/README.md` 의 "집 물건찾기 노드" 절과
`turtlebot3_llm_nav/worlds/objects.README.md` 를 보세요. §9 의 한계(긴 코드 생성 손상, 오라클 검출,
실행은 ROS2 PC 필요)는 이 임무에도 그대로 적용됩니다.
