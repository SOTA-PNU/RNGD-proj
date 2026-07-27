# _evidence — book_guide 수치의 원본 근거 (2026-07-24 세션)

book_guide 문서들이 인용하는 실측값의 **원본 데이터와 재현 도구**다.
원래는 job 임시 디렉터리(`/home/jun/.claude/jobs/46bc5c7e/tmp/`)에만 있었는데,
그 경로는 정리되면 사라지므로 근거를 잃지 않도록 저장소로 옮겼다.

## 무엇이 들어 있나

| 경로 | 내용 |
|---|---|
| `GROUND_TRUTH_BRIEF.md` | 문서 최신화에 사용한 실측 브리프. 모든 수치의 1차 출처 |
| `logs/npu_matrix.tsv` | **실기 매트릭스 원본.** 89행 = 테스트별 (상태, 소요ms, 바이너리, 테스트명, HAL오류수, 불일치수, load, 패닉위치) |
| `logs/perkernel_matrix_fixed.txt` | 커널 200개 개별 컴파일 판정(`OK\|이름` / `FAIL\|이름`). 접두사 충돌 보정 후 = 137 OK / 63 FAIL |
| `logs/sched_summary.json` | 커널 130개 스케줄 요약(커널별 span·엔진별 사이클·인스트럭션 종류·SRAM) |
| `logs/ve_isolated.log` | vector_engine 36개 개별 격리 실행 결과(연쇄 오염 판별 근거) |
| `tools/` | 재현 스크립트 7종 |

## 핵심 수치를 직접 재확인하는 법

```bash
cd "$(dirname "$0")"

# 실기 매트릭스: PASS=80 FAIL=5 ABORT=3 OTHER=1, 합 89
awk -F'\t' '/^(PASS|FAIL|ABORT|OTHER)/{c[$1]++;n++} END{for(k in c) printf "%s=%d ",k,c[k]; print " total="n}' logs/npu_matrix.tsv

# 커널 컴파일: OK=137 FAIL=63
awk -F'|' '{c[$1]++} END{printf "OK=%d FAIL=%d\n",c["OK"],c["FAIL"]}' logs/perkernel_matrix_fixed.txt

# 엔진별 사이클: DmaEngine 96.5% / PeCore 3.3%
python3 -c "
import json;d=json.load(open('logs/sched_summary.json'));e=d['engine_cycles'];t=sum(e.values())
[print(f'  {k:<18}{v:>12,}  {100*v/t:5.1f}%') for k,v in sorted(e.items(),key=lambda x:-x[1])]
print('kernels:',len(d['kernels']))"

# 커널별 DMA 지배율: 중앙값 82.8%, 107/130 이 50% 이상
python3 -c "
import json,statistics as st
d=json.load(open('logs/sched_summary.json'))
r=[(k['engines'].get('DmaEngine',0)/max(1,sum(k['engines'].values()))) for k in d['kernels']]
print(f'  n={len(r)} median={st.median(r):.1%} ge50={sum(1 for x in r if x>=.5)} ge90={sum(1 for x in r if x>=.9)}')"

# vector_engine 격리 실행: PASS 33 / FAIL 3
printf "  PASS=%s FAIL=%s\n" "$(grep -c 'test result: ok' logs/ve_isolated.log)" "$(grep -c 'test result: FAILED' logs/ve_isolated.log)"
```

## 주의

- `sched_summary.json` 의 사이클은 **컴파일러 스케줄 모델 예측**이며 실측 벽시계가 아니다.
  실제 실행 결과는 `npu_matrix.tsv` 쪽이다. 두 계열을 섞어 읽지 말 것.
- `tools/` 의 스크립트는 경로가 job 임시 디렉터리로 하드코딩돼 있다. 재실행하려면 경로를 고쳐야 한다.
- 전체 원본 로그(빌드 로그 등 수십 MB)는 옮기지 않았다. 여기 있는 것은 문서가 인용하는 수치의 근거만이다.
