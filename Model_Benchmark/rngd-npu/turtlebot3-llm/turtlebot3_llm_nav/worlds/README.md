# worlds — 사람(actor)이 있는 월드 만들기

이 task 는 로봇이 **카메라로 본 특정 사람**에게 다가가는 것이라, Gazebo 월드에 사람이
있어야 합니다. Gazebo Harmonic 은 `<actor>` 로 사람 모형을 넣습니다. 여기서는 upstream
`turtlebot3_world.world` 에 사람 몇 명을 추가하는 **최소** 예시를 둡니다.

## 1) actor 스니펫 (people.snippet.sdf)

[`people.snippet.sdf`](./people.snippet.sdf) 에 사람 3명을 서로 다른 위치에 둔 스니펫이
있습니다. 쓰는 법:

1. upstream 월드를 복사합니다.
   ```bash
   cp $(ros2 pkg prefix turtlebot3_gazebo)/share/turtlebot3_gazebo/worlds/turtlebot3_world.world \
      ~/turtlebot3_world_people.world
   ```
2. `people.snippet.sdf` 안의 `<actor>...</actor>` 블록들을 그 월드의 `</world>` **바로 위**에
   붙여 넣습니다.
3. 그 월드로 시뮬레이터를 띄웁니다(아래 2번).

> 메모(검증): waffle 카메라는 Intel RealSense R200, 1920x1080, horizontal_fov 1.02974 rad
> (`turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf` 에서 확인).
> 사람 위치는 로봇 스폰 위치(turtlebot3_world.launch.py 기본 x=-2.0, y=-0.5) 앞쪽 화각 안에
> 두면 카메라에 바로 잡힙니다.

## 2) 사람 있는 월드로 시뮬레이터 띄우기

```bash
export TURTLEBOT3_MODEL=waffle
# (A) 가장 간단: 기본 월드로 띄우고, 위에서 만든 월드 파일 경로를 직접 넣어 실행하려면
#     upstream 의 gz_sim 런치에 world 인자를 주는 방식을 쓰거나, turtlebot3_world.launch.py 의
#     world 경로를 ~/turtlebot3_world_people.world 로 바꿔 실행하면 됩니다.
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

(upstream `turtlebot3_world.launch.py` 는 월드 경로가 하드코딩이라, 사람월드를 쓰려면 그
런치를 복사해 `world` 변수만 `~/turtlebot3_world_people.world` 로 바꾸는 것이 가장 쉽습니다.)

## 3) ground-truth 검출기용 사람 pose 발행

`detector:=ground_truth` 모드는 실제 영상인식 대신, 시뮬레이터의 **사람 ground-truth pose**
를 JSON 으로 받는 토픽 `/people_ground_truth` (std_msgs/String) 를 봅니다. 형식:

```json
[{"id":"p1","x":1.0,"y":0.0,"features":{"shirt":"red","height":1.7}},
 {"id":"p2","x":2.0,"y":1.5,"features":{"shirt":"blue","height":1.8}}]
```

가장 간단한 발행 방법(actor 가 고정 위치라면, 위 좌표를 그대로 한 번씩 publish):

```bash
ros2 topic pub -r 2 /people_ground_truth std_msgs/msg/String \
  "{data: '[{\"id\":\"p1\",\"x\":1.0,\"y\":0.0,\"features\":{\"shirt\":\"red\"}},{\"id\":\"p2\",\"x\":2.0,\"y\":1.5,\"features\":{\"shirt\":\"blue\"}}]'}"
```

actor 가 움직이면, Gazebo 의 `/world/<world>/dynamic_pose/info` (또는 모델 pose) 토픽을
ros_gz_bridge 로 받아 사람 좌표를 추출해 위 JSON 으로 다시 publish 하는 작은 노드를 두면
됩니다(상자 actor 의 이름→features 매핑은 사용자가 정의). 이 부분은 시뮬레이터 셋업에
의존하므로 패키지에 포함하지 않았습니다 — 위 정적 publish 로 충분히 폐루프를 검증할 수 있습니다.
```
