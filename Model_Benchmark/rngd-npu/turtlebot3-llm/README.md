# TurtleBot3 × NPU LLM — 카메라 코딩 폐루프 (사람찾기 · 집 물건찾기)

TurtleBot3(ROBOTIS의 ROS2/Gazebo 시뮬레이터)에서, 로봇이 카메라로 받은 임무를 **단순 센서 주행으로는
자주 실패**할 때, **RNGD NPU의 코딩 LLM 서버와 대화하며 자기 제어 코드를 고쳐** 임무를 완수하는 폐루프를
구현·검증한 묶음입니다. ([[../robot-sim]] 의 2D 폐루프 하니스를 카메라/ROS2로 확장한 것입니다.)

두 가지 임무(task)를 같은 폐루프 위에 올렸습니다.

1. **특정 사람 찾아가기** — 카메라로 **특정 사람**(예: 빨간 옷 + 모자)에게 도달. 같은 빨강 decoy 오인,
   가려짐, 놓침 같은 실패를 LLM 과 대화하며 코드로 고칩니다. (노드 `llm_nav_node`)
2. **집 안 물건 확인 후 복귀** — [emanual 의 TurtleBot3 Gazebo **House** 월드](https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/#gazebo-simulation)
   에서 *집을 방마다 돌며 특정 물건(빨간 컵)이 있는지 확인하고 현관으로 돌아오기*. 전역 플래너가 준 방
   순회 경로를 따라가며 스캔→복귀→present/absent 판정하는 컨트롤러를 LLM 이 코딩합니다. (노드 `object_search_node`)

## 구성

| 위치 | 무엇 |
|------|------|
| **`PIPELINE.md`** | ⭐ 카메라→특정사람→실패→**NPU LLM 대화→코드수정→달성** 파이프라인 **상세 설명**(실제 대화 예시 포함). 먼저 읽으세요. |
| `turtlebot3_simulations/` | upstream 클론([robotis-git/turtlebot3_simulations](https://github.com/robotis-git/turtlebot3_simulations)) — 토픽·waffle 카메라·월드 기준점 |
| `turtlebot3_llm_nav/` | **ROS2(rclpy) 통합 패키지** — `/scan`·`/camera`·`/odom` → LLM `plan(state)` → `/cmd_vel` 폐루프. 사람찾기 + 자가수리. ROS2 Jazzy+Gazebo Harmonic+TurtleBot3 머신에서 실행 |
| `../robot-sim/` (`find_person`) | **헤드리스 검증판** — 디스플레이 없는 서버에서 지금 돌려 보고 브라우저로 볼 수 있는 같은 폐루프 |

`turtlebot3_llm_nav`(사람찾기 `llm_nav_node`, 집 물건찾기 `object_search_node`)와 `robot-sim`
(`find_person`, `house_search`)은 **똑같은 `plan(state)` 계약**을 공유합니다 — 헤드리스에서 검증한
파이프라인을 그대로 실로봇으로 옮길 수 있습니다.

### 집 물건찾기 트랙(ROS2)에 들어 있는 것

| 파일 | 무엇 |
|------|------|
| `turtlebot3_llm_nav/object_search_node.py` | 집 미션 폐루프 노드. `/scan`·`/camera`·`/odom`·`/objects_ground_truth` → `plan(state)` → `/cmd_vel`. 미션 성공 = present/absent 판정이 맞고 현관 복귀까지(정답은 `/objects_ground_truth` 로 대조). 헤드리스와 **같은 anti-guess 게이트**(present 는 실제로 봤을 때만, absent 는 경로 60%+ 방문했을 때만 인정) |
| `turtlebot3_llm_nav/object_perception.py` | 물체 검출기 — `ObjectGroundTruthDetector`(시뮬레이션 오라클, `/objects_ground_truth`) + `YoloObjectDetector`(실검출기 STUB) |
| `turtlebot3_llm_nav/house_prompts.py` | 집 미션 `plan(state)` 프롬프트 — 헤드리스 `robot-sim/prompts.py` 의 계약과 **동일** |
| `launch/llm_house_search.launch.py` | 집 미션 노드 런치(objective·waypoints·llm_port/mock 등) |
| `worlds/objects.snippet.sdf` · `worlds/objects.README.md` | House 월드에 물건(빨간 컵 target + decoy) 넣기 + `/objects_ground_truth` 발행법 |

웨이포인트(전역 경로)는 파라미터로 줍니다. 기본값은 turtlebot3_house 의 **대략적 방 순회 경로**라,
실제 월드/스폰에 맞춰 바꾸거나 Nav2 전역 플래너 경로로 대체하면 됩니다(실제 시스템의 planning 역할).

## 지금 바로 보기 (헤드리스, 디스플레이 불필요)

이 서버는 ROS2/Gazebo가 없고 헤드리스라, 사람찾기 폐루프는 `robot-sim` 의 `find_person` 으로 검증·관람합니다.

```bash
cd ../robot-sim
# 콘솔: 빨강 decoy 오인 → 'wrong_person' → LLM 수리 → 빨강+모자 target 도달
python3 run_sim.py --mock buggy --scenario find_person

# 브라우저: 사람·카메라 화각(FOV)·검출선·target 하이라이트가 움직이는 애니메이션
python3 web_sim.py --mock buggy --scenario find_person --serve 7900
#  → 7900 을 터널로 포워딩하고 http://127.0.0.1:7900/ 접속
```

집 물건찾기도 같은 방식으로 헤드리스에서 바로 봅니다:
```bash
# 콘솔: 집을 돌며 빨간 컵 확인 후 복귀(있음/없음/버그→자가수리)
python3 run_sim.py --mock good  --scenario house_search
python3 run_sim.py --mock good  --scenario house_search_absent
python3 run_sim.py --mock buggy --scenario house_search

# 브라우저: 집 도면·방·문·물건·카메라 화각·경로·판정 애니메이션
python3 web_sim.py --mock good --scenario house_search --serve 7900
```

실제 NPU 코더 모델로(서버에서 `./chat/serve_models.sh coder7` 먼저):
```bash
../chat/.venv/bin/python run_sim.py --model coder7 --scenario find_person
../chat/.venv/bin/python run_sim.py --model coder7 --scenario house_search
```

## 실제 TurtleBot3로 (디스플레이 있는 ROS2 PC)

요약 (자세히는 `turtlebot3_llm_nav/README.md`):
```bash
# 1) ROS2 Jazzy + Gazebo Harmonic + TurtleBot3 + 이 패키지 colcon 빌드
# 2) NPU 서버에서 코더 모델 serve:  ./chat/serve_models.sh coder7
# 3) 사람(actor) 있는 월드로 waffle 기동(예시: turtlebot3_llm_nav/worlds/)
# 4) ros2 launch turtlebot3_llm_nav llm_person_nav.launch.py \
#        target:='{"shirt":"red","cap":true}' llm_port:=8002
```

집 물건찾기는 House 월드로(자세히는 `turtlebot3_llm_nav/worlds/objects.README.md`):
```bash
# 1) House 월드에 물건(빨간 컵 + decoy) 넣고 띄우기 (objects.snippet.sdf)
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py     # (물건 넣은 월드 복사본)
# 2) 물건 pose 발행:  ros2 topic pub -r 2 /objects_ground_truth std_msgs/msg/String "{data: '[...]'}"
# 3) ros2 launch turtlebot3_llm_nav llm_house_search.launch.py \
#        objective:='{"label":"cup","color":"red"}' llm_port:=8002
```

## 핵심 포인트
- **로봇의 '행동'을 LLM이 코딩으로 구현**하고, 실패하면 **무슨 일이 있었는지 적어 NPU와 대화로 코드를
  고칩니다**(실패 유형별 점진 수리). 그래서 가장 중요한 게 **코딩 성능**입니다.
- 식별 모호성(빨강 둘 중 모자 쓴 한 명)·가려짐·탐색 같은 **'특정 사람 찾기'의 실패와 그 수리 루프**가
  핵심이며, 헤드리스판에서 그대로 재현·측정됩니다.
- **집 물건찾기**는 여기에 *여러 방을 도는 경로 추종 + 물건 식별(색만 같은 decoy 무시) + 현관 복귀 +
  present/absent 판정*까지 더한 더 긴 임무입니다. 전역 경로는 주어지고(planning), 그 위의 로컬
  컨트롤러를 LLM 이 코딩합니다(control) — Autoware 의 planning↔control 분담과 같은 구조입니다.
- 한계: 현재 NPU serve 코더 모델은 긴 코드 생성이 깨져, 프롬프트를 짧은 코드+점진 수리로 설계했습니다
  (`../robot-sim/README.md` 실측 참고). 실 Gazebo 실행은 ROS2 PC 필요.
