#!/usr/bin/env python3
"""llm_house_search.launch.py — '집 안 물건 확인 후 복귀' LLM 미션 노드 실행.

전제(이 런치는 노드만 띄웁니다 — 시뮬레이터/LLM 서버는 먼저 켜 두세요):
  1) turtlebot3_house 가 떠 있어야 합니다(emanual Gazebo Simulation 의 House):
       export TURTLEBOT3_MODEL=waffle
       ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py
  2) detector:=ground_truth 면, 물건 pose 를 JSON 으로 /objects_ground_truth(std_msgs/String)에
     발행하는 퍼블리셔가 있어야 합니다(worlds/objects.snippet.sdf · worlds/README 참고).
  3) LLM 서버(furiosa-llm serve, OpenAI 호환)가 떠 있어야 합니다(mock 제외). 예:
       cd ~/RNGD-proj/Model_Benchmark/rngd-npu/chat && ./serve_models.sh coder7   # 포트 8002

예시:
  # 서버 없이 폐루프 검증(정상 컨트롤러)
  ros2 launch turtlebot3_llm_nav llm_house_search.launch.py llm_mock:=good
  # 실제 NPU 코더 모델로 + 찾는 물건/경로 지정
  ros2 launch turtlebot3_llm_nav llm_house_search.launch.py \
       objective:='{"label":"cup","color":"red"}' llm_port:=8002

waypoints 는 집 방배치에 맞춘 전역 경로(JSON [[x,y],...])입니다. 기본값은 turtlebot3_house 의
대략적 방 순회 경로이며, 실제 월드/스폰 위치에 맞춰 바꾸거나 Nav2 전역 플래너 경로로 대체하세요.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument("llm_port", default_value="8002",
                              description="furiosa-llm serve 포트(chat CATALOG: coder7=8002)"),
        DeclareLaunchArgument("llm_mock", default_value="",
                              description="'good'/'buggy' 면 서버 없이 mock LLM 사용"),
        DeclareLaunchArgument("objective", default_value='{"label": "cup", "color": "red"}',
                              description="찾을 물건 특징(JSON: label/color 등)"),
        DeclareLaunchArgument("detector", default_value="ground_truth",
                              description="ground_truth | yolo"),
        DeclareLaunchArgument("waypoints", default_value="",
                              description="전역 경로 JSON [[x,y],...]. 비우면 노드 기본 경로 사용"),
        DeclareLaunchArgument("home", default_value="[-2.0, -0.5]",
                              description="복귀할 현관 좌표(JSON [x,y]) — house 스폰 기본값"),
        DeclareLaunchArgument("max_replans", default_value="5"),
        DeclareLaunchArgument("goal_tol", default_value="0.4"),
        DeclareLaunchArgument("v_max", default_value="0.22"),
        DeclareLaunchArgument("w_max", default_value="1.8"),
        DeclareLaunchArgument("control_hz", default_value="10.0"),
        DeclareLaunchArgument("cam_range", default_value="4.0"),
        DeclareLaunchArgument("cmd_vel_stamped", default_value="true",
                              description="이 ros_gz 브리지는 TwistStamped. 평범한 Twist 면 false"),
    ]

    params = {
        "llm_port": LaunchConfiguration("llm_port"),
        "llm_mock": LaunchConfiguration("llm_mock"),
        "objective": LaunchConfiguration("objective"),
        "detector": LaunchConfiguration("detector"),
        "home": LaunchConfiguration("home"),
        "max_replans": LaunchConfiguration("max_replans"),
        "goal_tol": LaunchConfiguration("goal_tol"),
        "v_max": LaunchConfiguration("v_max"),
        "w_max": LaunchConfiguration("w_max"),
        "control_hz": LaunchConfiguration("control_hz"),
        "cam_range": LaunchConfiguration("cam_range"),
        "cmd_vel_stamped": LaunchConfiguration("cmd_vel_stamped"),
    }

    # waypoints 는 빈 문자열이면 노드 기본값을 쓰도록 파라미터를 넘기지 않습니다.
    def _launch_setup(context, *a, **k):
        wp = LaunchConfiguration("waypoints").perform(context)
        p = dict(params)
        if wp.strip():
            p["waypoints"] = wp
        return [Node(package="turtlebot3_llm_nav", executable="object_search_node",
                     name="object_search_node", output="screen", parameters=[p])]

    from launch.actions import OpaqueFunction
    return LaunchDescription(args + [OpaqueFunction(function=_launch_setup)])
