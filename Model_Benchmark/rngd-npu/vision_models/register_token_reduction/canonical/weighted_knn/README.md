# 정통 kNN 가중투표 재채점 (weighted_knn)

이 폴더는 **정통(train-갤러리) kNN을 DINOv2 공식식 '온도가중 투표'로 다시 채점**하는 독립 도구입니다.
논문 정통 baseline이 다수결로 **80.87**(공인 82.0에 ~1점 못 미침)인데, 그 격차의 **가장 큰 원인이
투표 방식**(우리는 비가중 다수결, 공식은 온도가중)임을 확인·보정하려는 용도입니다.

---

## ⚠️ 전제조건 — 이 폴더만 단독으론 못 돕니다

이 도구는 **`pitome_compare` 엔진(`compare.py`)과 그 데이터·캐시**를 재사용합니다(특징추출 코드를 읽기
전용으로 import). 따라서 **다음이 있어야** 실행됩니다:

| 필요한 것 | 위치 | 없으면 |
|---|---|---|
| 엔진 `compare.py` | `../../pitome_compare/` (또는 `--engine`/`ENGINE`로 지정) | 즉시 에러(친절 안내) |
| 특징 캐시 `feat_cache/*.pt` | `pitome_compare/feat_cache/` | 캐시 없으면 데이터로 재추출(수 시간) |
| 데이터 `imagenet_val/`·`imagenet_train/` | `pitome_compare/` | 캐시도 없으면 실행 불가(에러) |

**즉, 정통 실험(`run_base_canonical.sh`)을 이미 돌린 그 GPU 서버에서 쓰는 게 정상 시나리오입니다** —
거기엔 엔진·데이터·(GC=1이었다면)캐시가 이미 있으니까요.

### 올바른 배치 방법
- **가장 쉬움**: 이 `weighted_knn/` 폴더를 그 GPU 서버의 **기존 `canonical/` 폴더 안에** 넣습니다
  (`register_token_reduction/canonical/weighted_knn/` 구조 유지). 그러면 기본 경로 `../../pitome_compare`가
  맞아떨어져 그대로 돌아갑니다.
- **다른 위치에 둘 경우**: 엔진 경로만 알려주면 됩니다 →
  `python weighted_knn.py --engine /경로/pitome_compare` 또는 `ENGINE=/경로/pitome_compare bash run_weighted_knn.sh`
- ❌ **완전히 빈 서버에 이 폴더만** 보내면 엔진·데이터가 없어 안 됩니다(그때는 `pitome_compare` 전체가 필요).

---

## 왜 별도 폴더인가
기존 검증 코드(`../../pitome_compare/compare.py`, `../run_base_canonical.sh`)와 그 결과는 **논문의 현재
수치**입니다. 그래서 원본을 **전혀 수정하지 않고**, `compare.py`를 import(읽기 전용)만 해서 **kNN 채점만**
새로 합니다. 원본 파일·기존 결과는 그대로 보존됩니다.

## 무엇을 하나
- 캐시된 특징을 **재사용** → 재추출(수 시간) 없음. kNN만 다시 채점하므로 **설정당 ~수초**.
- 각 (방법, 축소율)에서 **majority(=논문 현재 수치)와 weighted(=공식식)를 나란히** 출력 → 격차가 얼마나 닫히는지 바로 보임.
- 실행 시작에 **전제조건 점검(preflight)**: 엔진·캐시·데이터 유무를 사람이 읽을 수 있게 찍고, 아무것도 없으면 명확히 중단.

## 실행 (GPU 서버, 위 배치 후)
```bash
cd canonical/weighted_knn
bash run_weighted_knn.sh                 # 전체 곡선(r=0,8,12,16,18,20)
RLIST="" bash run_weighted_knn.sh        # baseline(r=0)만 빨리 (공인 82.0 대조용)
python weighted_knn.py --r_list 8 12 16 18 20   # 직접 실행
```
결과는 `results_weighted_knn.txt`에 저장됩니다.

## 결과 해석 / 다음 단계
- **r=0 의 weighted 열이 ~82** 로 오르면 → "격차 대부분이 투표 방식"임이 확인됩니다(버그 아님).
- 각 r 에서 majority→weighted 로 바꿔도 **세 방법(ToMe/PiToMe/Ours)의 순서·교차점이 유지**되면 → 결론이 투표 방식에도 불변.
- 두 조건이 맞으면(예상됨) 논문 `tab:canonical` 을 weighted 수치로 갱신하고, 정통 절 격차 설명을
  "**가중 kNN로 공인값을 재현**"으로 강화할 수 있습니다. (선택 — 카메라레디용. 현재 제출본은 이미 방어 가능.)

## 옵션
- `--k`(기본 20), `--temp`(기본 0.07): 이웃 수·온도. 공식은 k∈{10,20,100,200} 중 최적을 보고하지만 여기선 대조 목적상 k=20 고정.
- `--engine`, `--cache_dir`: 엔진/캐시 경로 직접 지정(다른 위치·머신용).
