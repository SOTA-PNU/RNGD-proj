# 진짜 Gazebo 화면을 웹브라우저로 보기 (헤드리스 서버에서)

> **상태**: 이 서버(부산대 NPU)는 Gazebo·ROS2 미설치 + **3D GPU 없음**(ASPEED BMC 2D만) + 작성자(AI)는
> 비번 없는 sudo 불가라 **아래 레시피는 직접 실행/검증해야 하는 '시작 레시피'** 입니다. GPU가 없어 소프트웨어
> 렌더(llvmpipe)로 떨어지므로 3D Gazebo 화면은 **느리고 끊깁니다.** 매끄럽게 보려면 GPU 있는 ROS2 PC를 권장.

이 서버 실측(2026-06-25): GNOME 데스크톱은 실제로 떠 있음(`gdm`→`Xwayland :1024`, HDMI 화면). 그래픽 환경은
있으나 ① Gazebo/ROS2 미설치 ② `/dev/dri/card1`=ASPEED(2D, render 노드 없음)→3D 가속 없음, 소프트 GL만
(`swrast_dri.so` 존재) ③ Xvfb·x11vnc·websockify·noVNC 미설치.

---

## ⭐ 방법 D (가장 좋음) — Gazebo 는 **GPU 있는 맥북(M-시리즈)**, LLM 은 **NPU 서버**

이게 사용자가 말한 그림이자 **권장 구성**입니다. NPU 서버엔 3D GPU 가 없지만 **맥북엔 진짜 GPU(Metal)** 가
있으니, **무거운 그래픽(Gazebo GUI)은 맥에서 GPU 가속으로 매끄럽게**, **LLM 추론은 NPU 서버가 HTTP 로** —
정확히 2겹 분담입니다. (역할 분리: Gazebo·ROS2·렌더 = 맥, plan(state) 코드 생성/수정 = NPU.)

### D-1. 맥에 Gazebo + ROS2 설치 (Apple Silicon)
공식적으로 ROS2 는 macOS 미지원이지만, **Apple Silicon 네이티브 소스빌드 설치**가 커뮤니티에 정리돼 있습니다
(M3 기준 각 ~15분):
- ROS2 Jazzy + Gazebo Harmonic 네이티브: <https://github.com/IOES-Lab/ROS2_Jazzy_MacOS_Native_AppleSilicon>
- ros_gz 브리지(ROS2↔Gazebo): <https://github.com/IOES-Lab/ROS_GZ_MacOS_Native_AppleSilicon>
- (대안) ABI 충돌 피한 소스빌드 Gazebo 스택: <https://github.com/idesign0/gz-macOS>
- 더 쉬운 길: **Ubuntu 24.04 arm64 VM**(Parallels=3D 가속 일부 지원 / UTM·Docker=소프트렌더). 네이티브가 GPU 가속엔 유리.

설치 후 이 저장소의 `turtlebot3_simulations`(turtlebot3_house.world) + `turtlebot3_llm_nav` 를 colcon 빌드.

### D-2. 맥에서 Gazebo 집 + LLM 노드 실행, LLM 은 NPU 서버로
```bash
# (서버) 코더 모델 serve — 예: coder7 → 포트 8002
#   cd ~/RNGD-proj/Model_Benchmark/rngd-npu/chat && ./serve_models.sh coder7

# (맥) NPU 서버의 그 포트를 SSH 터널로 당겨오기
ssh -p 10022 -N -L 8002:localhost:8002 jun@164.125.19.138 &

# (맥) Gazebo 집 월드 — 맥 GPU 로 렌더(매끄러움)
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py &

# (맥) 물건찾기 LLM 노드 — llm_port 를 터널된 포트로 지정 → 요청이 NPU 서버에서 처리됨
ros2 launch turtlebot3_llm_nav llm_house_search.launch.py \
     objective:='{"label":"cup","color":"red"}' llm_port:=8002
```
- `object_search_node` 는 `http://127.0.0.1:<llm_port>/v1` 로 OpenAI 호환 요청을 보냅니다(`llm_client.py`).
  터널이 그걸 NPU 서버로 넘기므로 **추론은 NPU, 화면·시뮬은 맥** 으로 깔끔히 나뉩니다.
- 2-LLM(로봇 두뇌+서버 코더)까지 원하면 두뇌·코더 각각의 serve 포트를 터널하고 노드에서 둘을 가리키게 확장.

### D-3. 왜 이게 최선인가
- **GPU**: 맥(Metal)에서 Gazebo 가속 → 부드러운 3D. 서버(GPU 없음)나 비가속 VM 의 소프트렌더 버벅임 회피.
- **NPU**: LLM 은 빌드된 아티팩트로 NPU 가 처리(맥엔 모델 불필요). 네트워크엔 추론 트래픽만 흐름.
- **그대로 실로봇 이식**: 같은 ROS2 패키지·`plan(state)` 계약이라, 맥 시뮬에서 검증한 걸 실 TurtleBot3 로 옮기기 쉬움.
> 주의: macOS ROS2 는 커뮤니티 소스빌드라 설치에 시간·버전 충돌 가능(위 gz-macOS 가 그 대비). 빠르고 확실한 건
> Ubuntu(네이티브 PC 또는 GPU 패스스루 VM). M5 는 CPU·GPU 가 강력해 네이티브가 잘 되면 가장 쾌적.

---

## 방법 A — Gazebo GUI를 가상화면에 띄우고 noVNC로 브라우저 스트리밍 (진짜 화면)

### A-1. 설치 (직접, sudo 필요 · 수백 MB)
```bash
# ROS2 Jazzy + Gazebo Harmonic + TurtleBot3 + ros_gz  (Ubuntu 24.04 기준)
sudo apt update
sudo apt install -y ros-jazzy-desktop ros-jazzy-ros-gz \
    ros-jazzy-turtlebot3 ros-jazzy-turtlebot3-msgs
# 이 저장소의 turtlebot3_simulations(turtlebot3_house.world 포함)를 colcon 빌드:
#   cd ~/ros2_ws/src && ln -s <repo>/turtlebot3-llm/turtlebot3_simulations . && cd ~/ros2_ws
#   colcon build && source install/setup.bash
# 가상 디스플레이 + 화면 스트리밍 도구
sudo apt install -y xvfb x11vnc python3-websockify novnc
```

### A-2. 실행 (가상화면 → Gazebo → noVNC)
```bash
# 1) 가상 디스플레이(보이지 않는 화면) + 소프트웨어 GL(이 서버엔 GPU 없음)
export DISPLAY=:99
Xvfb :99 -screen 0 1280x800x24 +extension GLX +render -noreset &
export LIBGL_ALWAYS_SOFTWARE=1          # GPU 없음 → llvmpipe 소프트 렌더
export GALLIUM_DRIVER=llvmpipe

# 2) ROS2 + 집 월드(GUI 가 :99 가상화면에 렌더됨)
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py &   # gzserver + gzclient(-g)

# 3) 가상화면(:99)을 VNC 로 열고 → 웹(noVNC)으로 변환
x11vnc -display :99 -nopw -forever -shared -rfbport 5900 &
websockify --web=/usr/share/novnc 6080 localhost:5900 &

echo "브라우저:  http://127.0.0.1:6080/vnc.html   (맥북은 alpacon tunnel -l 6080 -r 6080 먼저)"
```
> 끊김이 심하면 해상도/품질을 낮추거나(`-screen 0 1024x600x24`), GUI 없이 `gz sim -s`(서버만) + 방법 C 사용.

### A-3. 이미 떠 있는 데스크톱(:1024)을 그대로 스트리밍하려면
Xvfb 대신 `x11vnc -display :1024 ...` (단 그 화면에 Gazebo GUI가 떠 있어야 보임; Wayland 세션이라 권한/호환 주의).

---

## 방법 C — 픽셀 말고 '상태'만 가져와 우리 웹뷰어로 재렌더 (헤드리스 친화, 권장)

GPU·X 없이 헤드리스에서 가볍게, 게다가 **진짜 센서 데이터**로 봅니다. Gazebo(`gz sim -s`, GUI 없음)나
실로봇의 `/scan`·`/camera`·`/odom`·`/tf` 를 구독하는 작은 ROS2 브리지가 그 값을 JSON 으로 흘리면, `robot-sim`
의 `live_sim` 3D 뷰어(이미 SSE 로 벽·로봇·라이다·카메라를 그림)가 그대로 표시합니다. = 지금 헤드리스 웹 시뮬과
같은 화면을, 실제 Gazebo/로봇 데이터로. (이 브리지는 미구현 — 필요하면 만들 수 있습니다.)

---

## 한 줄 정리
- 헤드리스라 **Gazebo가 불가능한 건 아님**(Xvfb+소프트GL). 진짜 벽은 **미설치 + GPU 없음(느림)**.
- **Gazebo 화면 웹서빙도 가능**(방법 A=noVNC). 단 이 서버는 GPU 없어 버벅임 → GPU 있는 ROS2 PC 권장,
  헤드리스 이 서버에선 **방법 C(상태 스트리밍)** 가 실용적.
