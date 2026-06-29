# worlds — '집 안 물건 확인 후 복귀' 미션 셋업 (turtlebot3_house)

이 미션은 로봇이 **집(turtlebot3_house)을 돌며 특정 물건이 있는지 카메라로 확인하고 현관으로
복귀**하는 것이라, 집 월드에 물건 몇 개가 놓여 있어야 합니다. 진짜 목표(빨간 컵) 하나와, 한
특징만 같은 **decoy**(빨간 책·파란 컵)를 섞어, '색만 보고' 판정하면 틀리도록 둡니다.

> 헤드리스로 먼저 보기: 디스플레이/ROS2 없이 같은 미션을 `rngd-npu/robot-sim` 에서 바로 돌려
> 브라우저로 볼 수 있습니다 — `python3 web_sim.py --mock good --scenario house_search --serve 7900`.
> 아래는 **실제 Gazebo turtlebot3_house** 에서 돌리는 방법입니다.

## 1) House 월드에 물건 넣기 (objects.snippet.sdf)

[`objects.snippet.sdf`](./objects.snippet.sdf) 에 작은 색 박스 모형 3개(빨간 컵 = 목표,
빨간 책·파란 컵 = decoy)가 있습니다.

1. upstream house 월드를 복사합니다.
   ```bash
   cp $(ros2 pkg prefix turtlebot3_gazebo)/share/turtlebot3_gazebo/worlds/turtlebot3_house.world \
      ~/turtlebot3_house_objects.world
   ```
2. `objects.snippet.sdf` 안의 `<model>...</model>` 블록들을 그 월드의 `</world>` **바로 위**에
   붙여 넣습니다. (좌표는 house 네이티브 = 원점이 집 중앙. 방배치에 맞춰 옮겨도 됩니다.)
3. 그 월드로 house 를 띄웁니다. upstream `turtlebot3_house.launch.py` 는 월드 경로가 하드코딩이라,
   그 런치를 복사해 `world` 변수만 `~/turtlebot3_house_objects.world` 로 바꾸는 것이 가장 쉽습니다.
   ```bash
   export TURTLEBOT3_MODEL=waffle
   ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py   # (월드 경로 바꾼 복사본)
   ```

> 메모(검증): waffle 카메라는 Intel RealSense R200, 1920x1080, horizontal_fov 1.02974 rad
> (`turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf`). 현관 스폰 기본값은 x=-2.0, y=-0.5
> (`turtlebot3_gazebo/launch/turtlebot3_house.launch.py`).

## 2) ground-truth 검출기용 물건 pose 발행 (/objects_ground_truth)

`detector:=ground_truth` 모드는 실제 영상인식 대신, 물건 **ground-truth pose** 를 JSON 으로 받는
토픽 `/objects_ground_truth` (std_msgs/String) 를 봅니다(좌표는 위 snippet 의 `<pose>` 와 동일):

```json
[{"id":"cup_red","x":-5.0,"y":-2.0,"features":{"label":"cup","color":"red"}},
 {"id":"book_red","x":2.5,"y":1.5,"features":{"label":"book","color":"red"}},
 {"id":"cup_blue","x":0.5,"y":-3.0,"features":{"label":"cup","color":"blue"}}]
```

물건이 고정이라 위 좌표를 그대로 반복 publish 하면 됩니다:

```bash
ros2 topic pub -r 2 /objects_ground_truth std_msgs/msg/String \
  "{data: '[{\"id\":\"cup_red\",\"x\":-5.0,\"y\":-2.0,\"features\":{\"label\":\"cup\",\"color\":\"red\"}},{\"id\":\"book_red\",\"x\":2.5,\"y\":1.5,\"features\":{\"label\":\"book\",\"color\":\"red\"}},{\"id\":\"cup_blue\",\"x\":0.5,\"y\":-3.0,\"features\":{\"label\":\"cup\",\"color\":\"blue\"}}]'}"
```

**'없음(absent)' 시나리오**를 보려면 `cup_red` 항목만 빼고(빨간 컵을 월드에서도 제거) 발행하세요.
그러면 끝까지 검색해도 진짜 목표가 없으니, 로봇은 'absent' 로 올바로 판정해야 합니다(대충 검색하면
decoy 를 보고 present 라 오인 = false_report).

## 3) 미션 노드 띄우기

```bash
# 서버 없이 폐루프 검증(정상 컨트롤러: 경로추종→스캔→복귀→판정)
ros2 launch turtlebot3_llm_nav llm_house_search.launch.py llm_mock:=good

# 실제 NPU 코더 모델로(먼저 chat/serve_models.sh coder7 로 포트 8002 serve)
ros2 launch turtlebot3_llm_nav llm_house_search.launch.py \
     objective:='{"label":"cup","color":"red"}' llm_port:=8002
```

`waypoints:='[[x,y],...]'` 로 방 순회 경로를 줄 수 있습니다(기본값은 house 의 대략적 순회 경로 —
실제 방배치/스폰에 맞춰 바꾸거나 Nav2 전역 플래너 경로로 대체). 노드는 경로를 따라가며 스캔하고,
현관(home) 복귀 후 present/absent 를 판정합니다. 틀리거나(missed_object/false_report) 복귀 전에
끝내면(not_home) NPU LLM 에게 코드를 고쳐 달라고 다시 요청합니다(self-debug).
