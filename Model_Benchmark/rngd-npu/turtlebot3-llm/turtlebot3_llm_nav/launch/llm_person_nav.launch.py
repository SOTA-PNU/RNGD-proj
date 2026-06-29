#!/usr/bin/env python3
"""llm_person_nav.launch.py — LLM 사람찾기 노드 실행.

전제(이 런치는 노드만 띄웁니다 — 시뮬레이터/LLM 서버는 먼저 켜 두세요):
  1) TurtleBot3 + Gazebo 가 떠 있어야 합니다. 예(이 저장소 upstream 런치):
       export TURTLEBOT3_MODEL=waffle
       ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
     사람(actor)이 있는 월드를 쓰려면 worlds/README 를 보고 actor 를 넣은 .world 로 띄우세요.
  2) detector:=ground_truth 면, 사람 pose 를 JSON 으로 /people_ground_truth(std_msgs/String)에
     발행하는 퍼블리셔가 있어야 합니다(worlds/README 의 people_gt_publisher 예시 참고).
  3) LLM 서버(furiosa-llm serve, OpenAI 호환)가 떠 있어야 합니다(mock 모드 제외). 예:
       cd ~/RNGD-proj/Model_Benchmark/rngd-npu/chat && ./serve_models.sh coder7   # 포트 8002

예시:
  ros2 launch turtlebot3_llm_nav llm_person_nav.launch.py target:='{"shirt":"red"}' llm_port:=8002
  ros2 launch turtlebot3_llm_nav llm_person_nav.launch.py llm_mock:=good   # 서버 없이 검증
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
        DeclareLaunchArgument("target", default_value='{"shirt": "red"}',
                              description="찾을 사람 특징(JSON)"),
        DeclareLaunchArgument("detector", default_value="ground_truth",
                              description="ground_truth | yolo"),
        DeclareLaunchArgument("max_replans", default_value="5"),
        DeclareLaunchArgument("goal_tol", default_value="0.6"),
        DeclareLaunchArgument("v_max", default_value="0.22"),
        DeclareLaunchArgument("w_max", default_value="1.8"),
        DeclareLaunchArgument("control_hz", default_value="10.0"),
        DeclareLaunchArgument("cmd_vel_stamped", default_value="true",
                              description="이 ros_gz 브리지는 TwistStamped. 평범한 Twist 면 false"),
    ]

    node = Node(
        package="turtlebot3_llm_nav",
        executable="llm_nav_node",
        name="llm_nav_node",
        output="screen",
        parameters=[{
            "llm_port": LaunchConfiguration("llm_port"),
            "llm_mock": LaunchConfiguration("llm_mock"),
            "target": LaunchConfiguration("target"),
            "detector": LaunchConfiguration("detector"),
            "max_replans": LaunchConfiguration("max_replans"),
            "goal_tol": LaunchConfiguration("goal_tol"),
            "v_max": LaunchConfiguration("v_max"),
            "w_max": LaunchConfiguration("w_max"),
            "control_hz": LaunchConfiguration("control_hz"),
            "cmd_vel_stamped": LaunchConfiguration("cmd_vel_stamped"),
        }],
    )

    return LaunchDescription(args + [node])
