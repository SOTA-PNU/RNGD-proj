# 맥북(Apple Silicon)에 Gazebo + ROS2 + TurtleBot3 환경 구축 — LLM 은 NPU 서버

> 구조: **무거운 그래픽(Gazebo)은 맥(GPU)에서, LLM 추론은 NPU 서버(HTTP)에서.**
> 아래 명령은 공식 문서 + 커뮤니티 설치 저장소에서 확인한 것이지만, **작성 환경엔 맥이 없어 직접 실행 검증은
> 못 했습니다**(특히 source 빌드는 Xcode/버전에 따라 손이 갑니다). 빠른 확인은 맨 아래 "스모크 테스트"부터.

## 0. 서버 쪽 — **새로 구축할 것 없음** ✅
NPU·furiosa-llm·아티팩트·serve 스크립트가 이미 다 있습니다. 쓸 때 **모델 하나만 serve(띄우기)** 하면 됩니다:
```bash
# (NPU 서버에서) 코더 모델 하나 serve — 예: coder7 → 포트 8002
cd ~/RNGD-proj/Model_Benchmark/rngd-npu/chat && ./serve_models.sh coder7
```
끝. 빌드/설치는 서버에서 할 게 없습니다.

---

# 맥북에서 (한 번만) 환경 구축

## 1. 사전 준비
```bash
# Xcode Command Line Tools  (⚠️ 이 설치본은 Xcode 16.2 를 요구 — 더 최신이면 빌드 실패 보고 있음)
xcode-select --install
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -license

# Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
(echo; echo 'eval "$(/opt/homebrew/bin/brew shellenv)"') >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

## 2. ROS2 Jazzy + Gazebo Harmonic (한 줄, 네이티브·GPU 사용)
[IOES-Lab 설치기](https://github.com/IOES-Lab/ROS2_Jazzy_MacOS_Native_AppleSilicon)가 ROS2 Jazzy + Gazebo
Harmonic 를 **소스 빌드**로 깔아 줍니다(M3 기준 합쳐 ~30–45분).
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/IOES-Lab/ROS2_Jazzy_MacOS_Native_AppleSilicon/main/install.sh)"
```
- 기본 설치 위치: `$HOME/ros2_jazzy` (Gazebo 는 `gz_harmonic`).
- **환경 활성화**(매 셸마다): `source $HOME/ros2_jazzy/activate_ros`  ← ROS2 가 파이썬 venv 안에서 돌아서,
  `setup.bash` 직접 source 말고 **이 스크립트로** 켜야 합니다. (설치기가 `ros` 별칭도 제안)
- 확인: `gz sim --version`, `ros2 --help` 가 뜨면 OK.

## 3. ros_gz 브리지 빌드 (Gazebo↔ROS2 — 맥에선 brew/apt 없음, 소스 빌드)
```bash
mkdir -p $HOME/ros_gz_ws/src && cd $HOME/ros_gz_ws/src
git clone https://github.com/IOES-Lab/ROS_GZ_MacOS_Native_AppleSilicon.git
git clone --branch 2.1.2 https://github.com/swri-robotics/gps_umd.git
git clone --branch 0.0.1 https://github.com/rudislabs/actuator_msgs.git
git clone https://github.com/ros-perception/vision_msgs.git && cd vision_msgs && git checkout 1adca4d && cd ..
# macOS 패치
cd $HOME/ros_gz_ws/src/vision_msgs
git apply ../ROS_GZ_MacOS_Native_AppleSilicon/patches/vision_msgs_rviz_jazzy.patch
cd $HOME/ros_gz_ws
# Qt5 환경
export CMAKE_PREFIX_PATH=$(brew --prefix qt@5)/lib:$(brew --prefix qt@5)/lib/cmake:/opt/homebrew/opt:${CMAKE_PREFIX_PATH}
export PATH=$(brew --prefix qt@5)/bin:$PATH
ln -s $(brew --prefix qt@5)/mkspecs /opt/homebrew/mkspecs 2>/dev/null || true
ln -s $(brew --prefix qt@5)/plugins /opt/homebrew/plugins 2>/dev/null || true
# 빌드
source $HOME/ros2_jazzy/activate_ros
python3.11 -m colcon build --symlink-install \
    --packages-skip-by-dep python_qt_binding \
    --cmake-args -DBUILD_TESTING=OFF \
    -DCMAKE_OSX_SYSROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
    -Wno-dev --event-handlers console_cohesion+
# 활성화 스크립트에 워크스페이스 추가(매번 자동 source)
echo "source $HOME/ros_gz_ws/install/setup.zsh" >> $HOME/ros2_jazzy/activate_ros
```
> ⚠️ IOES-Lab README 안내: 빌드 중 `gz-msgs10` 의 TINYXML2 참조를 cmake 파일에서 손봐야 하는 단계가 있음
> (sudo 필요). 빌드 로그를 보고 안내대로 처리하세요. — 이 ros_gz 빌드가 맥 셋업에서 제일 까다로운 부분입니다.

## 4. 우리 TurtleBot3-LLM 패키지 올리기
서버의 두 패키지를 맥으로 복사 후 워크스페이스에 둡니다.
```bash
# (맥에서) 서버에서 두 패키지 받기
mkdir -p ~/tb3_ws/src
rsync -a -e "ssh -p 10022" \
  jun@164.125.19.138:'~/RNGD-proj/Model_Benchmark/rngd-npu/turtlebot3-llm/turtlebot3_simulations' \
  jun@164.125.19.138:'~/RNGD-proj/Model_Benchmark/rngd-npu/turtlebot3-llm/turtlebot3_llm_nav' \
  ~/tb3_ws/src/

# 노드의 런타임 파이썬 의존성(openai)을 ROS2 가 쓰는 인터프리터에 설치
source $HOME/ros2_jazzy/activate_ros
python3 -m pip install openai httpx

# 빌드
cd ~/tb3_ws
rosdep install --from-paths src --ignore-src -r -y    # ros_gz_bridge/image, cv_bridge 등 당김(있으면 생략됨)
python3.11 -m colcon build --symlink-install
echo "source ~/tb3_ws/install/setup.zsh" >> $HOME/ros2_jazzy/activate_ros
```

---

# 실행 (맥에서 Gazebo, 요청은 NPU 서버가 처리)

```bash
source $HOME/ros2_jazzy/activate_ros        # ROS2 + ros_gz + 우리 패키지 전부 활성화

# (a) NPU 서버의 serve 포트를 맥 로컬로 터널 (NpuClient 가 http://127.0.0.1:<port>/v1 고정)
ssh -p 10022 -N -L 8002:127.0.0.1:8002 jun@164.125.19.138 &

# (b) 집 월드 — 맥 GPU 로 렌더(매끄러움)
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py &     # 로봇 스폰 x=-2.0,y=-0.5

# (c) 물건찾기 LLM 노드 — llm_port 를 터널된 포트로 → 추론은 NPU 서버
ros2 launch turtlebot3_llm_nav llm_house_search.launch.py \
    objective:='{"label":"cup","color":"red"}' \
    llm_port:=8002 detector:=ground_truth &

# (d) (ground_truth 모드) 물건 위치 발행 — 빨강컵 target + decoy
ros2 topic pub -r 2 /objects_ground_truth std_msgs/msg/String \
  "{data: '[{\"id\":\"cup_red\",\"x\":-5.0,\"y\":-2.0,\"features\":{\"label\":\"cup\",\"color\":\"red\"}},{\"id\":\"book_red\",\"x\":2.5,\"y\":1.5,\"features\":{\"label\":\"book\",\"color\":\"red\"}},{\"id\":\"cup_blue\",\"x\":0.5,\"y\":-3.0,\"features\":{\"label\":\"cup\",\"color\":\"blue\"}}]'}"
```
- (서버) 먼저 `./chat/serve_models.sh coder7` 로 8002 serve 떠 있어야 함(0번 참고).
- 'absent' 시나리오: 위 (d)에서 `cup_red` 항목 빼고(월드에서도 제거) 발행.
- **2-LLM(로봇 두뇌+서버 코더)**: 두 모델을 각각 serve(예 8002 코더, 8003 두뇌)하고 두 포트를 터널 →
  노드에 두뇌/코더 포트를 함께 넘기게 확장(현재 object_search_node 는 단일 llm_port; 2-LLM 은 robot-sim
  쪽 brain 연동을 ROS2 노드로 옮기면 됨).

## ⚡ 스모크 테스트 — Gazebo·NPU·터널 없이 노드만 (가장 빠른 확인)
`llm_mock:=good` 은 내장 컨트롤러를 써서 **rclpy 만으로** 폐루프가 도는지 봅니다(openai·터널 불필요).
```bash
ros2 launch turtlebot3_llm_nav llm_house_search.launch.py llm_mock:=good
```

---

## 핵심 파라미터 (object_search_node / launch 공통)
`llm_port`(기본 8002) · `llm_mock`(`""`|`good`|`buggy`) · `objective`(예 `{"label":"cup","color":"red"}`) ·
`detector`(`ground_truth`|`yolo`) · `home`(기본 `[-2.0,-0.5]`) · `waypoints` · `max_replans`(5) · `goal_tol` ·
`v_max` · `w_max` · `control_hz` · `cam_range` · `cmd_vel_stamped`(기본 `true` — 이 ros_gz 브리지는 TwistStamped).

## 주의 / 대안
- **Xcode 16.2 고정**: 더 최신이면 빌드 깨짐(다운그레이드 필요). DDS 는 CycloneDDS 제외 → Fast-DDS 기본.
- **GPU/렌더**: macOS 엔 OGRE2 가 선호하는 OpenGL 경로가 없어 Apple Metal 경로로 감(소스빌드/런타임 실패가
  여기서 잦음). 그래도 맥 GPU 라 GPU 없는 서버보다 훨씬 부드럽습니다.
- **더 쉬운 대안(소스빌드 싫으면)**: ① **RoboStack(conda)** — `conda` 환경에 `ros-humble-desktop`+`ros-gz`
  바이너리(빌드 거의 없음, 단 Jazzy 아닌 Humble 위주) ② **Docker(Ubuntu arm64)** — 설치 제일 쉬우나 GPU 가속
  안 돼 소프트 렌더(느림). GPU 부드러움 우선이면 위 네이티브, 설치 편의 우선이면 RoboStack.
- 같은 ROS2 패키지·`plan(state)` 계약이라 **맥 Gazebo 에서 검증한 걸 실 TurtleBot3 로 그대로 이식**.

출처: [IOES-Lab/ROS2_Jazzy_MacOS_Native_AppleSilicon](https://github.com/IOES-Lab/ROS2_Jazzy_MacOS_Native_AppleSilicon),
[IOES-Lab/ROS_GZ_MacOS_Native_AppleSilicon](https://github.com/IOES-Lab/ROS_GZ_MacOS_Native_AppleSilicon),
[Gazebo docs — Installing Gazebo with ROS](https://gazebosim.org/docs/latest/ros_installation/),
[idesign0/gz-macOS](https://github.com/idesign0/gz-macOS).
