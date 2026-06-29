"""로봇의 '두뇌'(작은 LLM, 플래너/매니저) — 2-LLM 계층 구조의 엣지측.

구조(역할 분리, 사용자 선택 'A + 가'):
  [로봇 두뇌(작은 LLM, NPU 1장)]  ──자연어 지시──▶  [서버 코더(큰 LLM, 다른 NPU)]
        - task 해석                                      - plan(state) 코드 생성/수정
        - 무엇을 요청/수정할지 '자연어'로 결정             - 그 코드를 로봇에 돌려줌
        - 받은 코드 적용·평가는 로봇 몸(agent.py)이 함

핵심: **두뇌는 코드를 직접 쓰지 않습니다.** 상황을 읽고, 코더 모델에게 "이런 동작을 하는
컨트롤러를 만들어/고쳐 줘"라고 사람 말처럼(자연어) 적어 보내는 일만 합니다(역할 '가').
"""
from __future__ import annotations

import math


# 두뇌(플래너) 시스템 프롬프트 — 영어로 두어 모델 신뢰도를 높입니다(사용자 로그는 한국어).
BRAIN_SYSTEM = """You are the on-board PLANNER of a mobile robot — the robot's small 'brain'. You do NOT
write code. A separate, more powerful CODING model writes the actual plan(state) controller. Your job is
to read the TASK and the current SITUATION and write a SHORT, clear, plain-language INSTRUCTION telling
the coding model what behaviour the controller should have (or, when something failed, what to fix).

Rules:
- PLAIN NATURAL LANGUAGE only. Do NOT write Python or code blocks — describe the behaviour in words.
- Be concrete and short (2-4 sentences). Cover: the goal, how the robot should move/explore, what to
  watch for (e.g. look-alike decoys), how to handle obstacles/walls, and when the task is finished.
- When fixing, name what went wrong and the specific behaviour change needed (start with "Fix the controller").
"""


def situation_summary(state: dict) -> str:
    """두뇌에게 줄 짧은 상황 요약(코더용 상세 스펙이 아니라 사람이 읽는 한 줄 수준)."""
    x, y = state.get("pos", (0.0, 0.0))
    parts = [f"robot at ({x:.1f}, {y:.1f})"]
    if state.get("objective"):
        parts.append(f"looking for {state['objective']}")
    scan = state.get("scan")
    if scan is not None:
        if scan:
            parts.append("camera now sees: " + ", ".join(str(d.get("features", {})) for d in scan[:4]))
        else:
            parts.append("camera sees nothing right now")
    lidar, ang = state.get("lidar"), state.get("lidar_angles")
    if lidar and ang:
        front = min([d for d, a in zip(lidar, ang) if abs(a) < 0.4] or [state.get("max_range", 5.0)])
        parts.append(f"nearest wall ahead ~{front:.1f} m")
    if state.get("home"):
        hx, hy = state["home"][0], state["home"][1]
        parts.append(f"home is at ({hx:.1f}, {hy:.1f})")
    if state.get("goal"):
        parts.append(f"goal point is at ({state['goal'][0]:.1f}, {state['goal'][1]:.1f})")
    return "; ".join(parts) + "."


class RobotBrain:
    """로봇 두뇌(플래너) — 작은 LLM 클라이언트를 감싸, 코더에게 보낼 '자연어 지시'를 생성합니다."""

    def __init__(self, client, temperature: float = 0.3, max_tokens: int = 320):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = getattr(client, "name", "brain")

    def _ask(self, user: str):
        text, m = self.client.complete(BRAIN_SYSTEM, user,
                                       temperature=self.temperature, max_tokens=self.max_tokens)
        return (text or "").strip(), m

    def initial_request(self, task_text: str, state: dict):
        """task 를 받아 코더에게 보낼 첫 지시(자연어)를 만듭니다."""
        user = (f"TASK: {task_text}\n\n"
                f"SITUATION (start): {situation_summary(state)}\n\n"
                "Write a short plain-language instruction for the coding model describing the controller "
                "behaviour it should implement for this task. No code.")
        return self._ask(user)

    def repair_request(self, reason: str, detail: str, state: dict, prev_code: str):
        """실패를 보고 코더에게 보낼 수리 지시(자연어)를 만듭니다."""
        user = (f"The controller you ordered did NOT complete the task.\n"
                f"FAILURE: {reason} — {detail}\n"
                f"SITUATION now: {situation_summary(state)}\n\n"
                f"The coding model's previous controller was:\n```python\n{prev_code[:1200]}\n```\n\n"
                "Write a SHORT plain-language instruction telling the coding model what to fix "
                "(start with 'Fix the controller'). Describe the behaviour change, not code.")
        return self._ask(user)


def task_text_for(world) -> str:
    """world 로부터 두뇌에게 줄 high-level task 설명(자연어)을 만듭니다."""
    if getattr(world, "house_task", False):
        obj = world.objective or {}
        return (f"Autonomously explore the house with the LiDAR and camera to decide whether a "
                f"{obj} is present, then return to home and report present or absent. There is no map "
                f"or route. Beware decoy objects that match only one feature (e.g. same colour, different label).")
    # 일반 내비/사람찾기
    if getattr(world, "vision_task", False):
        return (f"Use the camera to find and drive to the specific target person matching "
                f"{world.target}, ignoring look-alike decoys, while avoiding obstacles with the LiDAR.")
    return ("Drive the robot from its current position to the goal point, avoiding obstacles sensed with "
            "the LiDAR (watch out for getting stuck in concave/U-shaped walls).")
