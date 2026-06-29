"""turtlebot3_llm_nav — LLM이 컨트롤러 코드를 직접 써서 TurtleBot3(waffle)을
카메라로 본 '특정 사람'에게 다가가게 하는 ROS2 폐루프 패키지.

핵심 아이디어(robot-sim 의 폐루프 하니스를 ROS2/Gazebo 로 옮긴 것):
  LLM 이 plan(state) 라는 짧은 파이썬 컨트롤러를 만든다 →
  노드가 매 제어주기마다 ROS2 토픽으로 state 를 모아 plan() 을 돌린다 →
  실패(충돌·길잃음·엉뚱한 사람·정체·예외)를 감지하면 수리 프롬프트를 보내
  고친 코드를 받아 계속한다. (replan 횟수 상한)

모듈
  perception   : 카메라 Image → 사람 검출 리스트(ground-truth / 실검출기 백엔드)
  executor     : LLM 코드 추출 + 던더차단 샌드박스 exec + 시간제한
  llm_client   : OpenAI 호환(furiosa-llm serve) 스트리밍 클라이언트 + mock
  prompts      : 사람찾기 task 의 SYSTEM/scaffold/repair 프롬프트(영어)
  llm_nav_node : 위를 묶는 rclpy 노드(구독/상태조립/폐루프/cmd_vel 발행)
"""

__all__ = [
    "perception",
    "executor",
    "llm_client",
    "prompts",
    "llm_nav_node",
]
__version__ = "0.1.0"
