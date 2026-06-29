# ACCV 2026 논문 프로젝트

이 폴더는 ACCV 2026(제18회 아시아 컴퓨터비전 학회)에 낼 논문 "Silent Precision Collapse"의 주제·실험 계획·기술 해설을 모아 둔 작업 폴더입니다.

**논문 한 줄 요약.** 학습이 끝난 이미지 분류 신경망을 퓨리오사 RNGD NPU(저정밀 추론용 칩)에 올리면 정확도가 거의 0으로 무너집니다. 이 논문은 그 원인을 밝히고, 채널마다 숫자 크기를 알맞게 조절하는 값(per-channel 스케일)과 튀는 값을 자르는 기준(clip)을 **최적화**해서, 다시 학습시키지 않고도 정확도를 거의 원래대로 되살립니다.

## 문서 구성
- [01_논문주제.md](01_논문주제.md) — 무슨 문제를 왜 푸는지, 왜 좋은 주제인지 (쉬운 말로)
- [02_실험계획.md](02_실험계획.md) — 무엇을·어떤 순서로·어떻게 돌리는지 (명령어·코드 포함)
- [03_furiosa-opt_코드해설.md](03_furiosa-opt_코드해설.md) — furiosa-opt의 어느 코드를 왜 쓰는지, 중학생도 이해할 수준의 해설

## 한눈에
- **학회:** ACCV 2026, 일본 오사카, 12월. 본 트랙 제출 마감 7월 5일. 노리는 트랙은 "Optimization Methods".
- **결정권자 핵심 2명:** Jiwen Lu(양자화·하드웨어 공동설계 전문가), Hyunjung Shim(원인 먼저 진단하고 고치는 방식 선호). 둘의 취향에 정확히 맞춘 주제다.
- **상세 근거 문서(이 레포):** `Model_Benchmark/info/README_virtual_isa.md`(furiosa-opt 분석), `README_vision_compile.md`(비전 모델 NPU 실측), `README_op_support.md`(연산·정밀도 실측).
- **실험에 쓰는 코드:** `Model_Benchmark/rngd-npu/vision_models/classify.py`, `rngd-npu/run_edf.py`(둘 다 이미 동작 확인됨).
