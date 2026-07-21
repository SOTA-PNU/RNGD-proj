# PiToMe 같은-예산 비교 실험 (GPU용 자료)

이 문서는 우리 방법(Ours, register 보호)을 **가장 가까운 경쟁자 PiToMe**와 **같은 압축률(같은 예산)에서 정확도로 정면 비교**하고, 그 정확도 이득이 속도를 깎지 않는지 **throughput**으로 확인하는 실험 자료 모음입니다. NPU 서버에서 만들었고, 실제 실행은 GPU 서버에서 합니다.

## 무엇을·왜 비교하나 (핵심 한 줄)
우리 논문의 셀링포인트는 "**빠르게**"가 아니라 "**같은 속도인데 더 정확하게**"입니다. 그래서 이 실험은 두 가지를 보입니다.
1. **정확도(주력):** 같은 압축률에서 `Ours > PiToMe > ToMe` — register 위치를 보호 기준으로 쓰는 게 PiToMe의 에너지 판단보다 낫다.
2. **throughput(보조):** 세 방법의 초당 처리량(im/s)이 사실상 같다 → "정확도 이득은 **공짜**(속도 대가 없음)"임을 못박음.

## 세 방법 (공정 비교 설계)
세 방법 모두 **블록 뒤·같은 metric(post-block 토큰)·블록당 같은 제거량 r** 로 통제했습니다. 오직 **"무엇을 지키고 무엇을 합칠지"** 규칙만 다릅니다.

| 이름 | 보호 대상 | 병합 방식 |
|---|---|---|
| `tome` | CLS 1개 | size-가중 bipartite soft matching(BSM) |
| `pitome` | **CLS 1개만** (register 개념 없음) | 에너지 `E=elu(cos−m).mean` 로 고에너지 2r개=병합·나머지=보호 |
| `ours` | CLS + register 전부 | 동일 BSM |

- **PiToMe가 CLS만 보호**하는 게 핵심입니다. PiToMe에는 register라는 개념이 없어서, DINOv2-reg의 register 토큰을 그냥 "합칠 수 있는 패치"로 보고 에너지 점수로 없앨 수 있습니다 — 바로 우리 논문이 노리는 취약점입니다.
- **Ours는 register를 위치로 지킵니다.** "무엇이 중요한지 판단"할 필요 없이 register 위치가 정답이라는 것이 우리 주장이고, 이 비교가 그 증거입니다.

## PiToMe 구현의 충실성 (공식 코드 그대로)
`compare.py`의 `pitome_step` 은 공식 리포 소스를 **줄 단위로 포팅**했습니다 (추정 아님, 원문 대조).
- 에너지식 `energy = F.elu(sim − margin).mean(-1)` — 공식 `algo/pitome/merge.py` 의 `pitome_vision`.
  (논문 Eq.4 `f_m(cos)` 의 코드판. 저에너지=보호, 고에너지=병합.)
- 층별 margin `m = 0.75 − 0.75·(l/L)` — 공식 `algo/pitome/patch/deit.py` 의 `margins = [.75 - .75*(i/num_layers) ...]`.
- 병합집합을 짝/홀로 쪼개 `a→best-b` 로 합치고 size-가중 평균 — 공식 `pitome` + `merge_wavg`.
- 출처: <https://github.com/hchautran/PiToMe> (`algo/pitome/merge.py`, `algo/pitome/patch/deit.py`), 논문 PiToMe (NeurIPS 2024, arXiv 2405.16148).

> 통제상 우리는 세 방법 모두 metric=post-block 토큰·삽입=블록 뒤로 **고정**했습니다(공정성). 공식 리포의 attention-key metric·proportional-attention·블록 내부 삽입 같은 "전체 파이프라인"은 `run_official_pitome.sh` 로 **그들 모델(DeiT)에서 따로 교차검증**합니다(아래).

## 평가 프로토콜
지표는 **표준 kNN top-1 정확도**(k=20)입니다. 기본 설정은 **val leave-one-out k-NN**: **갤러리 = 쿼리 = ImageNet-val 5만 장**이며, 각 이미지는 **자기 자신만 빼고(leave-one-out)** 이웃을 찾습니다(라벨 누수 없음 — 표준 기법). 논문의 다른 전 실험과 **동일 프로토콜**이라 나란히 실을 수 있습니다.
- **왜 val leave-one-out(50k)인가**: 방법 논문의 기여는 절대값이 아니라 **같은 조건에서의 차이(Δ)** 입니다. 그래서 남의 논문 수치(정통 128만 갤러리 기준 82.0)를 인용해 우리 값 옆에 놓지 **않고**, ToMe·PiToMe·Ours를 **전부 같은 50k 하네스에서 우리가 직접 재측정**합니다(PiToMe는 공식 코드 포팅). 무압축(r=0) 값이 곧 "DINOv2 kNN을 50k val(leave-one-out)로 재측정한 baseline"입니다. 절대값이 공인값보다 낮은 것은 갤러리가 작아 같은-클래스 이웃이 적기 때문(프로토콜 차이)일 뿐, 상대 비교는 완전히 유효합니다.
- **논문 표기(신빙성)**: "k-NN top-1(k=20), ImageNet-val을 갤러리로 쓰는 leave-one-out, 모든 방법 동일 적용, 경쟁 방법은 우리가 재측정"을 명시. (새 지표가 아니라 표준 kNN의 갤러리 설정만 val로 둔 것.) 정통 128만 갤러리로 승급하면 절대값이 ~82로 오르나 상대 순서는 유지 예상.
- **승급 옵션**: `--gallery train`(gallery=ImageNet train) 지원 — 시간·자원 여유 시 정통 프로토콜로 재실행(무압축 ~82 재현, 출처 [dinov2 MODEL_CARD](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)). train 다운로드·특징추출 비용 큼(`feat_cache/` 캐시로 재개).

## 실행 방법
```bash
bash run.sh            # 0)의존성 1)데이터(val) 2)val leave-one-out k-NN 정확도 3)throughput 까지 한 번에
```
개별 실행:
```bash
python prepare_data.py --split val                                  # val 5만 (최초 1회)
python compare.py --mode acc  --gallery val --r_list 8 12 16 18 20  # val leave-one-out k-NN(기본), r=0 baseline 자동 포함
python compare.py --mode tput --batch 128 --r_list 0 8 12 16 18 20  # throughput(im/s), r=0=무압축 기준선
# (승급) python prepare_data.py --split train --per_class 1300 && python compare.py --mode acc --gallery train
```
- 기본 모델 `vit_base_patch14_reg4_dinov2.lvd142m` (register 4개). ViT-S/L 은 `--model` 로 교체. 입력은 224×224 고정(모든 이미지 동일 256 패치).
- 결과는 콘솔 + `results_acc.txt` / `results_tput.txt` 로 저장됩니다.

## 읽는 법 (기대 결과)
- **정확도 표**: 같은 `comp%`(압축률) 행에서 `ours` 열이 `pitome`·`tome` 보다 높고, 압축이 셀수록 격차가 벌어지면 성공. `Δ(ours-pitome)` 열이 그 차이입니다. 절대값이 아니라 **세 방법의 Δ가 기여**(모두 같은 프로토콜, 우리가 직접 측정). (val leave-one-out 예비 ablation: 91% 압축 시 energy 계열 ≈ tome, ours 만 +7~8%p.)
- **throughput 표**: 세 방법 im/s 가 서로 비슷하면(±몇 %) → "정확도 이득이 공짜"라는 주장 성립. r=0(무압축) 대비 압축 시 몇 배 빨라지는지도 같이 보입니다. (합성 배치 측정 = 데이터셋 무관·표준 방식.)

## (선택) 공식 리포 교차검증
```bash
bash run_official_pitome.sh   # 공식 repo clone + 그들 env + DeiT-base 에서 pitome/tome 직접 실행
```
공식 pitome 곡선이 우리 `compare.py` 의 pitome 곡선과 **같은 추세**면 포팅이 옳다는 확인입니다. (DeiT는 register가 없어 여기 절대수치는 그들 논문 재현용이고, register head-to-head는 `compare.py`가 담당합니다.)

## 파일
- `compare.py` — 정확도(정통 kNN, gallery=train) + throughput 본체 (tome/pitome/ours, 공식 selection 포팅, 특징 캐싱).
- `prepare_data.py` — ImageNet val(query)·train(gallery) 다운로드(HF resized-256 미러, `--split`, 토큰 불필요).
- `run.sh` — 전체 파이프라인.
- `run_official_pitome.sh` — 공식 리포 독립 교차검증.
- `requirements.txt` — torch/timm/datasets 등.

## 관련
- 방법 본체·다른 실험: `../` (register_token_reduction), 다전략 ablation: `../eval_v2/`(`pick_extra`가 random/energy/highnorm 등 "무엇을 보호할지 판단"하는 대안들 — ours가 이들을 이김).
- 논문 원고: `../../ACCV/main.md`, 핵심 정리: `../../ACCV/논문_핵심정리.md`.
