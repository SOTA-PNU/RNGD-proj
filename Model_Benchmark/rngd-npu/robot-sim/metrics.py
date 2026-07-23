"""에피소드·배치 성능 지표와 리포트(표/JSON) 렌더링.

코딩 성능을 보는 핵심 숫자
  * success        : 로봇이 목표에 도달했는가(코드가 실제로 임무를 완수했는가)
  * replans        : 목표까지 코드를 몇 번 고쳐 썼는가(적을수록 1발에 맞춘 것)
  * code_valid_1st : 첫 코드가 컴파일/실행되었는가(문법·API 준수)
  * collisions / exceptions : 주행 중 충돌·런타임 예외 횟수
연동(SOAFEE/ROS2) 분석 숫자
  * 컨트롤 사이클 홉 지연(perception→planning→control), 사이클 Hz
  * LLM 코드생성/수리 지연(TTFT·TPS·토큰)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class EpisodeResult:
    scenario: str
    model: str
    success: bool = False
    terminate_reason: str = ""        # goal / collision / out_of_bounds / stuck / step_budget / no_valid_code
    steps: int = 0                    # 실제 시뮬 스텝 수
    path_length: float = 0.0          # 주행 거리(m)
    straight_dist: float = 0.0        # 시작-목표 직선거리(m) — 경로효율 비교용
    min_clearance: float = 1e9        # 주행 중 장애물 최소 여유(m)
    collisions: int = 0
    exceptions: int = 0
    replans: int = 0                  # 코드 재작성 횟수(초기 생성 제외)
    code_valid_first: bool = False    # 첫 코드가 빌드+실행 성공했는가
    llm_calls: int = 0
    llm_total_s: float = 0.0
    llm_total_tokens: int = 0
    ttft_s: float = 0.0               # 첫 LLM 호출 TTFT
    tps: float = 0.0                  # 평균 TPS(가중)
    # 미들웨어(연동) 분석
    cycle_hops: Optional[Dict[str, float]] = None   # 평균 홉 지연(ms)
    cycle_hz: float = 0.0
    path: List[Tuple[float, float]] = field(default_factory=list)

    def path_efficiency(self) -> float:
        """직선거리 / 실제주행거리(1.0 에 가까울수록 효율적). 도달 못 하면 0."""
        if not self.success or self.path_length <= 0:
            return 0.0
        return self.straight_dist / self.path_length


def summarize(results: List[EpisodeResult]) -> dict:
    """여러 에피소드를 한 줄 요약(성공률·평균 재작성·평균 효율 등)."""
    n = len(results)
    if n == 0:
        return {}
    succ = [r for r in results if r.success]
    def avg(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else 0.0
    return {
        "episodes": n,
        "success": len(succ),
        "success_rate": len(succ) / n,
        "avg_replans": avg(r.replans for r in results),
        "first_try_success": sum(1 for r in results if r.success and r.replans == 0),
        # 코드가 실제로 생성된 에피소드만 분모로(서버/LLM 호출 실패는 '나쁜 첫 코드'가 아니므로 제외)
        "code_valid_first_rate": avg(1.0 if r.code_valid_first else 0.0
                                     for r in results if r.terminate_reason != "llm_error"),
        "avg_steps_success": avg(r.steps for r in succ),
        "avg_path_eff": avg(r.path_efficiency() for r in succ),
        "total_collisions": sum(r.collisions for r in results),
        # 실제 측정된 값만 평균(실패 호출의 0.0 이 평균을 끌어내리지 않게 — avg_tps 와 동일 기준)
        "avg_ttft_s": avg(r.ttft_s for r in results if r.ttft_s > 0),
        "avg_tps": avg(r.tps for r in results if r.tps > 0),
        "total_llm_tokens": sum(r.llm_total_tokens for r in results),
    }


# ── 리포트 렌더링 ─────────────────────────────────────────────────
def render_table(results: List[EpisodeResult]) -> str:
    """에피소드별 한 줄 표(콘솔용)."""
    head = (f"{'scenario':<12} {'result':<8} {'reason':<13} "
            f"{'steps':>5} {'eff':>5} {'rpln':>4} {'1st':>3} {'coll':>4} "
            f"{'ttft':>6} {'tps':>6} {'tok':>6}")
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(
            f"{r.scenario:<12} {('OK' if r.success else 'FAIL'):<8} {r.terminate_reason:<13} "
            f"{r.steps:>5} {r.path_efficiency():>5.2f} {r.replans:>4} "
            f"{('Y' if r.code_valid_first else 'n'):>3} {r.collisions:>4} "
            f"{r.ttft_s:>6.2f} {r.tps:>6.1f} {r.llm_total_tokens:>6}")
    s = summarize(results)
    if s:
        lines.append("-" * len(head))
        lines.append(
            f"success {s['success']}/{s['episodes']} ({s['success_rate']*100:.0f}%)  "
            f"first-try {s['first_try_success']}  "
            f"avg replans {s['avg_replans']:.2f}  "
            f"avg path-eff {s['avg_path_eff']:.2f}  "
            f"avg TTFT {s['avg_ttft_s']:.2f}s  avg TPS {s['avg_tps']:.1f}")
    return "\n".join(lines)


def dump_json(results: List[EpisodeResult], path: str, meta: Optional[dict] = None):
    out = {
        "meta": meta or {},
        "summary": summarize(results),
        "episodes": [
            {k: v for k, v in asdict(r).items() if k != "path"}  # 경로 좌표는 용량 커서 제외
            for r in results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return path
