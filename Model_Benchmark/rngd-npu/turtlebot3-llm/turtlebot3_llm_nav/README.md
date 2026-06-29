# turtlebot3_llm_nav

TurtleBot3(waffle)을 **LLM 이 직접 짠 컨트롤러 코드**로 움직이는 ROS2(rclpy) 패키지입니다. LLM 서버는
NPU 위에서 도는 furiosa-llm serve(OpenAI 호환)를 씁니다. 두 가지 임무를 같은 폐루프로 제공합니다.

| 노드(entry point) | 임무 | 런치 |
|---|---|---|
| `llm_nav_node` | 카메라로 본 **특정 사람**에게 다가가기 | `llm_person_nav.launch.py` |
| `object_search_node` | **집(House)을 돌며 특정 물건 확인 후 현관 복귀** | `llm_house_search.launch.py` |

아래는 먼저 **사람찾기**(`llm_nav_node`)를 설명하고, 마지막 절에서 **집 물건찾기**(`object_search_node`)를
설명합니다. 둘은 `executor.py`(샌드박스)·`llm_client.py`(NPU/mock)를 공유하고, 같은 `plan(state)` 폐루프
구조를 씁니다.

## 무엇을 하나요

LLM 에게 "사람을 향해 가는 컨트롤러 함수 `plan(state)` 를 짧게 써 줘" 라고 부탁합니다. 노드는
그 함수를 받아 매 제어주기(약 10 Hz)마다 실행하고, 결과 속도를 로봇에 보냅니다. 잘못되면(부딪힘,
사람을 놓침, 엉뚱한 사람에게 감, 멈춰버림, 코드 오류) 무엇이 잘못됐는지 적은 **수리 요청**을 다시
보내 고친 코드를 받습니다. 이 "코드 생성 → 실행 → 실패 감지 → 수리" 의 반복이 폐루프(closed loop)입니다.

이 구조는 `rngd-npu/robot-sim/` 의 폐루프 하니스(executor.py / llm_client.py)를 ROS2·Gazebo·카메라
사람찾기 task 로 옮긴 것입니다.

## 폐루프 한눈에

```
/odom /scan /camera/image_raw /camera/camera_info (+/people_ground_truth)
        │  매 틱 state 조립
        ▼
   state = {pos, heading, lidar, lidar_angles, camera(=사람검출들), target, memory, v_max, w_max, dt}
        │
        ▼
   LLM ── plan(state) 코드 생성 ──► 샌드박스 build ──► 매 틱 호출 ──► (v,w) clamp ──► /cmd_vel
        ▲                                                              │
        └────────── 수리 프롬프트(실패유형별) ◄──── 실패 감지 ◄─────────┘
        (충돌 / 길잃음 / 엉뚱한사람 / 정체 / 예외 / 타임아웃 / 코드없음, replan 상한)
```

`plan(state)` 는 `{'v': 선속도, 'w': 각속도}` 를 돌려줍니다. 노드가 한계값으로 잘라
`/cmd_vel` 로 보냅니다.

## 토픽 배선 (이 저장소 waffle 브리지에서 확인한 값)

| 방향 | 토픽 | 타입 |
|---|---|---|
| 구독 | `/scan` | sensor_msgs/LaserScan |
| 구독 | `/camera/image_raw` | sensor_msgs/Image |
| 구독 | `/camera/camera_info` | sensor_msgs/CameraInfo |
| 구독 | `/odom` | nav_msgs/Odometry |
| 구독 | `/people_ground_truth` | std_msgs/String (JSON, ground_truth 검출기 전용) |
| 발행 | `/cmd_vel` | **geometry_msgs/TwistStamped** |

> 주의: 이 ros_gz / Jazzy 브리지(`turtlebot3_gazebo/params/turtlebot3_waffle_bridge.yaml`)는
> `/cmd_vel` 을 **TwistStamped** 로 브리지합니다(평범한 Twist 아님). 그래서 기본값을
> `cmd_vel_stamped:=true` 로 두었습니다. 평범한 Twist 를 쓰는 환경이면 `cmd_vel_stamped:=false`
> 로 바꾸세요.
>
> 카메라는 Intel RealSense R200, 1920x1080, horizontal_fov 1.02974 rad 입니다
> (`turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf` 에서 확인).

## 사람 인식: ground-truth vs 실제 검출기

`perception.py` 는 검출기 백엔드를 **두 개** 같은 인터페이스로 제공합니다.

- **ground_truth (기본, 시뮬레이션용 오라클)** — 실제 영상인식이 **아닙니다**. 시뮬레이터가
  `/people_ground_truth` 로 알려준 사람들의 실제 좌표를 받아, 카메라 화각(±FOV/2) 안에 드는
  사람만 검출로 만듭니다. 진짜 검출기 없이도 폐루프 전체를 Gazebo 에서 바로 돌려볼 수 있습니다.
- **yolo (실제 검출기 자리, STUB)** — 카메라 Image 에 YOLO 사람검출 + 옷색/얼굴 특징추출을 붙이는
  자리입니다. 인터페이스(`detect()` 가 `{bearing, distance, features, conf}` 리스트 반환)와
  픽셀→각도/거리 투영 헬퍼(`pixel_to_bearing`, `estimate_distance`)는 이미 들어 있고, 모델 추론
  부분만 `TODO` 로 비워 두었습니다. 화면 x 좌표와 카메라 HFOV(1.02974)로 bearing 을, bbox 세로
  높이와 사람 키 가정(1.7 m)으로 거리를 핀홀 추정합니다.

검출 하나의 형식은 두 백엔드가 같습니다:
`{'bearing': 좌우각도[rad], 'distance': 거리[m], 'features': {...}, 'conf': 0~1}`.

## 빌드

ROS2 워크스페이스의 `src/` 아래에 이 패키지를 두고 colcon 으로 빌드합니다.

```bash
# 예: 워크스페이스가 ~/ros2_ws 라면
mkdir -p ~/ros2_ws/src
ln -s ~/RNGD-proj/Model_Benchmark/rngd-npu/turtlebot3-llm/turtlebot3_llm_nav ~/ros2_ws/src/
cd ~/ros2_ws
colcon build --packages-select turtlebot3_llm_nav
source install/setup.bash
```

LLM 서버에 붙으려면 워크스페이스 파이썬에 `openai`, `httpx` 가 있어야 합니다(mock 모드는 불필요).

## 실행

세 가지를 켭니다: ① 시뮬레이터(사람 있는 월드) ② (ground_truth 면) 사람 pose 발행 ③ LLM 서버.

```bash
# 0) LLM 서버(NPU). chat 의 serve_models.sh 로 coder7(포트 8002) 띄우기
cd ~/RNGD-proj/Model_Benchmark/rngd-npu/chat && ./serve_models.sh coder7

# 1) 시뮬레이터 (사람 actor 가 있는 월드 — worlds/README.md 참고)
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# 2) ground_truth 검출기용 사람 pose 발행 (actor 좌표와 일치시키기)
ros2 topic pub -r 2 /people_ground_truth std_msgs/msg/String \
  "{data: '[{\"id\":\"p1\",\"x\":1.0,\"y\":0.0,\"features\":{\"shirt\":\"red\"}},{\"id\":\"p2\",\"x\":2.0,\"y\":1.5,\"features\":{\"shirt\":\"blue\"}}]'}"

# 3) 사람찾기 노드 (빨간 셔츠 사람을 목표로)
ros2 launch turtlebot3_llm_nav llm_person_nav.launch.py target:='{"shirt":"red"}' llm_port:=8002
```

서버 없이 노드 동작만 확인하려면 mock LLM 을 씁니다:

```bash
ros2 launch turtlebot3_llm_nav llm_person_nav.launch.py llm_mock:=good
# llm_mock:=buggy 로 하면 첫 코드에 부호버그를 넣어 self-debug(수리) 루프를 재현합니다.
```

## 주요 파라미터

| 파라미터 | 기본 | 설명 |
|---|---|---|
| `llm_port` | 8002 | furiosa-llm serve 포트 (chat CATALOG: coder7=8002, coder14=8003, a3b-fp8=8000 …) |
| `llm_mock` | "" | "good"/"buggy" 면 서버 없이 mock LLM |
| `target` | `{"shirt":"red"}` | 찾을 사람 특징(JSON). 검출의 features 가 이걸 모두 포함하면 그 사람이 목표 |
| `detector` | ground_truth | `ground_truth`(오라클) 또는 `yolo`(실검출기, STUB) |
| `max_replans` | 5 | 수리(코드 재생성) 횟수 상한 |
| `goal_tol` | 0.6 | 목표 사람까지 이 거리[m] 안이면 도착 성공 |
| `v_max` / `w_max` | 0.22 / 1.8 | waffle 선속도/각속도 한계 |
| `control_hz` | 10.0 | 제어주기 |
| `collision_dist` | 0.22 | 정면 최소 scan 이 이 아래면 충돌 처리(강제 정지) |
| `cmd_vel_stamped` | true | `/cmd_vel` 을 TwistStamped 로(이 브리지 기본). Twist 면 false |

## 실패 감지(요약)

- **collision** — 정면 lidar 최소거리가 `collision_dist` 아래 → 즉시 정지 + 수리.
- **lost_target** — 목표 특징과 맞는 사람이 연속 `lost_ticks` 동안 안 보임.
- **wrong_person** — 목표가 안 보이는데 다른 사람에게 `goal_tol` 안으로 접근.
- **stuck** — 위치도 안 변하고 목표 거리도 안 줄어듦(`stuck_ticks`).
- **exception / timeout** — plan() 실행 중 예외 또는 시간초과.
- **no_valid_code** — LLM 응답에 정상 `plan(state)` 가 없음(빌드 실패).

각 유형마다 `prompts.repair_user()` 가 다른 수리 메시지를 보내 self-debug 를 유도합니다.

## 안전 메모

`executor.py` 의 샌드박스는 던더 접근·위험 내장함수를 막고 import 는 `math` 만 허용하며,
빌드/호출에 시간제한을 둡니다. 다만 in-process `exec` 는 완전한 격리가 아니므로 **우리 시스템이
생성한 신뢰 가능한 코드**에만 쓰세요. 또 노드는 plan() 결과와 무관하게, 정면 충돌이 임박하면
무조건 정지하는 안전 게이트를 둡니다.

## 집 물건찾기 노드 (`object_search_node`)

[emanual 의 TurtleBot3 Gazebo **House**](https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/#gazebo-simulation)
에서, *집을 방마다 돌며 특정 물건(빨간 컵)이 있는지 확인하고 현관으로 복귀*하는 임무입니다. 사람찾기와
같은 폐루프지만, 한 점 도달이 아니라 **여러 방을 도는 경로 추종 + 물건 스캔 + 복귀 + present/absent 판정**
이라는 더 긴 임무입니다.

**전역 경로 ↔ 로컬 컨트롤러 분담(Autoware planning↔control):** 방을 도는 웨이포인트 경로(`waypoints`
파라미터)는 **주어집니다**(실제 시스템에선 Nav2 전역 플래너 역할). LLM 은 그 경로를 따라가는 로컬
컨트롤러 `plan(state)` 를 코딩합니다 — 순수 추종 + 카메라 스캔 + 현관 복귀 + 판정. state 에 `waypoints`·
`home`·`objective`·`scan`(지금 보이는 물건들)·`memory` 가 들어오고, 끝나면 `{'done':True,'present':bool}`
을 돌려줍니다. (계약은 헤드리스 `../../robot-sim/prompts.py` 의 HOUSE_* 와 동일 — `house_prompts.py`.)

**물체 인식:** `object_perception.py` 가 사람찾기의 `perception.py` 와 같은 구조로 두 백엔드를 줍니다.
- `ground_truth`(기본) — `/objects_ground_truth`(std_msgs/String, JSON)로 받은 물건 좌표를 카메라 화각
  안의 것만 검출로 만드는 **시뮬레이션 오라클**(실제 인식 아님).
- `yolo`(STUB) — 실제 영상 기반 물체검출 자리. `detect()` 인터페이스와 투영 헬퍼만 채워 두고 추론은 TODO.

**미션 성공/실패 판정:** 노드는 `/objects_ground_truth` 전체 목록으로 **정답(물건 존재 여부)**을 알고,
로봇이 `{'done':True,'present':...}` 를 선언하면 ① 현관 복귀(`goal_tol`) ② 판정 정확을 확인합니다. 추측
통과를 막으려고, 헤드리스와 같은 **anti-guess 게이트**를 노드가 직접 추적합니다 — `present` 는 로봇이
진짜로 목표를 카메라로 봤을 때만, `absent` 는 경로 웨이포인트의 60% 이상을 실제로 방문했을 때만 인정합니다.
틀리거나(missed_object/false_report), 복귀 전 종료(not_home), 검색 부족(searched_too_little)이면 수리
프롬프트로 코드를 고쳐 다시 받습니다.

**실행** (자세한 월드 셋업은 [`worlds/objects.README.md`](worlds/objects.README.md)):

```bash
# 0) LLM 서버(NPU): coder7(포트 8002)
cd ~/RNGD-proj/Model_Benchmark/rngd-npu/chat && ./serve_models.sh coder7

# 1) House 월드에 물건(빨간 컵 + decoy) 넣고 띄우기 (objects.snippet.sdf 참고)
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py     # 물건 넣은 월드 복사본

# 2) 물건 pose 발행(좌표는 snippet 과 일치)
ros2 topic pub -r 2 /objects_ground_truth std_msgs/msg/String \
  "{data: '[{\"id\":\"cup_red\",\"x\":-5.0,\"y\":-2.0,\"features\":{\"label\":\"cup\",\"color\":\"red\"}}]'}"

# 3) 집 물건찾기 노드
ros2 launch turtlebot3_llm_nav llm_house_search.launch.py \
     objective:='{"label":"cup","color":"red"}' llm_port:=8002
# 서버 없이 폐루프만 확인:  llm_mock:=good  (buggy 면 '검색 없이 없다' 버그→수리 재현)
```

**주요 파라미터:** `objective`(찾을 물건 JSON), `waypoints`(전역 경로 JSON `[[x,y],...]`, 비우면 노드
기본 순회 경로), `home`(현관 좌표, 기본 `[-2.0,-0.5]` = House 스폰 기본값), `detector`(ground_truth|yolo),
`cam_range`(물건 인지 거리, 기본 4.0 m), 그 외 `llm_port`/`llm_mock`/`goal_tol`/`v_max`/`w_max`/
`max_replans`/`cmd_vel_stamped` 는 사람찾기와 동일합니다.

> 디스플레이 없이 같은 미션을 지금 바로 보려면, 헤드리스 `../../robot-sim` 의 `house_search` 를 쓰세요
> (`python3 web_sim.py --mock good --scenario house_search --serve 7900`). 그쪽 도면은 이 House 의
> `model.sdf` 벽을 그대로 파싱해 만든 것이라 방·문 배치가 같습니다.

## 출처 / 기반

- TurtleBot3 시뮬레이션(waffle 모델·토픽·런치): 상위 저장소
  `turtlebot3_simulations` (`turtlebot3_gazebo/`) — ROBOTIS, Apache-2.0.
  본 저장소에 클론되어 있습니다:
  `RNGD-proj/Model_Benchmark/rngd-npu/turtlebot3-llm/turtlebot3_simulations/`.
  토픽/타입은 `turtlebot3_gazebo/params/turtlebot3_waffle_bridge.yaml`,
  카메라 사양은 `models/turtlebot3_waffle/model.sdf` 에서 확인했습니다.
- LLM 폐루프 하니스(코드 추출·샌드박스·OpenAI 스트리밍 클라이언트): `rngd-npu/robot-sim/`
  (`executor.py`, `llm_client.py`).
- LLM 서버/포트 카탈로그: `rngd-npu/chat/`(`chat_app.py` 의 CATALOG, `serve_models.sh`).
