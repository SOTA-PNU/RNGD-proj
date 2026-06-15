# op_verify — RNGD 지원 op 실행 검증 스크립트

`furiosa.torch.db.SUPPORTED_ATEN_OPS`(97개)가 실제로 NPU에서 컴파일·실행되는지 직접 돌려
확인한 스크립트입니다. 결과 요약은 상위 폴더의 `README_op_support.md`, 발표자료는
`../../ppt/RNGD_Op_Support.pptx` 입니다.

- 환경: `~/furiosa/bin/python` (furiosa-torch 2026.2.0 / torch 2.10.0), 실행 카드 `rngd:3`.
- 실행: `cd` 후 `/home/jun/furiosa/bin/python <스크립트>.py` (stderr 시끄러우면 `2>/dev/null`).

| 스크립트 | 하는 일 |
|---|---|
| `verify_round1_all97.py` | 97개 op 전부 export→분해→`CompileModule.from_exported`→`rngd:3` 실행, CPU와 비교. present/compile/run 3축 판정 + `results.json` 저장 |
| `verify_round2_embedded.py` | 1라운드에서 막힌 op를 `sigmoid()+add` 실연산 그래프에 끼워 재시험 + `_copy` 오버로드 + matmul 정밀도 재판정 |
| `verify_round3_harden.py` | 의심 op를 dtype·랭크·모양·API 바꿔가며 다지기 (max_pool values-only, copy 등) |
| `reconcile.py` | 라운드 간·교차검증 간 결과가 엇갈린 케이스를 같은 카드에서 나란히 재현 |
| `precision_probe.py` | matmul/conv vs elementwise 정밀도 측정 (상대오차·코사인) |
| `shape_sweep.py` | gather/index 계열의 맨 안쪽 차원 정렬 의존성(8/4의 배수) 스윕 |

핵심 결론: 목록(SUPPORTED)은 컴파일러가 "받겠다"고 선언한 것일 뿐 실행 보장이 아님 →
97개 중 89개 실행 OK(matmul 3개는 ~0.23% 감소정밀도), 6개 조건부, 2개(`isnan`·`constant_pad_nd`) 불가.
