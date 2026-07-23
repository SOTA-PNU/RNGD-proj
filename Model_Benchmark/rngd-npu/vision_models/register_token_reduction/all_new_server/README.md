# all_new_server — 새 GPU 서버에서 faithful + train 전체로 실험하기

이 문서는 **기존 환경이 전혀 없는 새 GPU 서버(A6000 4장 기준)에서**, 지금까지 val 로 돌렸던 실험을 **정식(faithful) 방식 + ImageNet train 전체(1.28M) 갤러리**로 다시 돌리기 위한 자족(self-contained) 번들의 사용 설명서입니다. 이 폴더만 새 서버로 통째로 옮기면, 아래 순서대로 환경 설치 → 데이터 준비 → 실험 실행까지 됩니다.

기존 A100 2장 서버에는 데이터·가중치가 이미 캐시돼 있어서 문제가 없었지만, 새 서버에는 그게 없습니다. 그래서 새 서버에서 자주 걸리는 세 가지(데이터 경로 제각각·오프라인 모드로 가중치 못 받음·train 데이터 없음)를 이 번들이 미리 처리합니다.

---

## 1. 무엇을 돌리나요 (val → train 매핑)

핵심은 하나입니다: **"val 로 하던 정확도 비교를 train 전체 갤러리로"**.

| 실험 | 기존(A100) | 새 서버에서 | 스크립트 |
|---|---|---|---|
| **헤드라인 head-to-head** (ToMe·PiToMe·Ours kNN) | val-LOO, faithful | **train 갤러리, faithful** | `run_train_gallery.sh` |
| ablation (register vs random/energy/highnorm) | val-LOO, faithful | **train 갤러리, faithful** | `run_ablation_regcount.sh` |
| reg-count 스윕 + 부트스트랩 CI | val-LOO, faithful | **train 갤러리, faithful** | `run_ablation_regcount.sh` |
| (대조용) 위 전부를 val-LOO 로 재현 | val-LOO, faithful | val-LOO, faithful | `run_val_sanity.sh` |

- **train 갤러리 kNN** = 갤러리(비교 대상)로 ImageNet train 128만 장, 쿼리로 val 5만 장을 씁니다. DINOv2 공식 kNN(≈82.0)과 같은 정통 프로토콜입니다.
- **val-LOO kNN** = 갤러리=쿼리=val 5만(자기 자신 제외). 논문 전 실험에서 통일해 쓴 방식이며, 여기서는 **새 서버 환경이 A100 결과를 그대로 내는지 확인하는 대조용**으로만 씁니다(빠름, 수십 분).
- **faithful(정식) harness** = 선행연구(Bolya ToMe, PiToMe) 그대로의 방식입니다: proportional attention(log(size) bias) + key-metric(attention key 평균) + attention↔MLP 사이 병합. PiToMe 는 공식 소스(`pitome_bsm`/`pitome`/margin 스케줄)를 그대로 이식했습니다.
- 검색 mAP·dense(ADE20k)는 이 번들 범위 밖입니다. 검색 mAP 는 val 자기검색이라 "train 갤러리로 옮기기"가 다른 프로토콜이고(대조용으로 `run_val_sanity.sh` 에 val 버전만 포함), dense 는 ImageNet 이 아니라 ADE20k 라서 별개입니다.

---

## 2. 요구 사양

- 리눅스 + NVIDIA GPU(권장 A6000 48GB × 4). GPU 1장으로도 됩니다(그만큼 순차).
- Python 3.9+ (venv 사용).
- **디스크 여유 넉넉히**: val ~7GB + train(1.28M JPG) ~수십 GB + **train 특징 캐시가 큼**(설정당 ~2GB, 전부 켜면 수백 GB). 여유 있으면 **250GB+ 권장**. 디스크가 빠듯하면 `export GALLERY_CACHE=0` (특징 캐시 끔 → 재개 불가 대신 디스크 대폭 절약), 또는 모델별 실험을 끝낼 때마다 `$DATA_ROOT/feat_cache/*` 를 지우세요. train 다운로드는 **NVMe** 를 강하게 권합니다(HDD 면 읽기가 병목).
- 인터넷(첫 1회): HF 미러에서 데이터·가중치를 받습니다. **HF 토큰은 필요 없습니다**(non-gated 미러 `evanarlian/imagenet_1k_resized_256`).

---

## 3. 빠른 시작

```bash
# 0) 이 폴더를 새 서버로 옮긴 뒤, 폴더 안에서:
cd all_new_server

# (선택) 데이터/캐시를 큰 디스크에 두려면 먼저 지정:
export DATA_ROOT=/mnt/big/rtr_data          # 기본값은 ./data

# 1) 환경 설치 (서버당 1회)
bash setup_env.sh
source .venv/bin/activate                    # 이후 실험은 이 venv 안에서

# 2) 전부 자동으로 (데이터→가중치→val 대조→train 헤드라인→ablation)
bash run_all.sh
```

`run_all.sh` 가 부담되면 아래처럼 단계별로 돌려도 됩니다:

```bash
source .venv/bin/activate
source config.sh                             # 환경변수 로드(수동 단계용)

python prepare_data.py --split val                       # 쿼리 5만
python prepare_data.py --split train --per_class 1300    # 갤러리 1.28M (수 시간)
python warmup_models.py                                  # 가중치 캐시(온라인 1회)

bash run_val_sanity.sh 0                      # (권장) 환경 대조: A100 수치와 일치하나?
bash run_train_gallery.sh s b l              # ★ 헤드라인: S/B/L 을 GPU0/1/2 병렬
bash run_ablation_regcount.sh b 3            # (선택) ablation+reg-count 를 GPU3
```

---

## 4. 스크립트 설명

- **`setup_env.sh`** — venv 생성 + `requirements.txt` 설치 + 검증(torch·CUDA 인식 여부 출력). CUDA 미인식이면 서버 CUDA 에 맞는 torch 휠 재설치 안내를 띄웁니다.
- **`config.sh`** — 모든 실행 스크립트가 맨 처음 읽는 공통 설정(데이터 경로·모델 목록·`HF_HUB_OFFLINE=0`·워커 수). **바꿀 건 보통 `DATA_ROOT` 뿐**입니다.
- **`prepare_data.py`** — ImageNet val/train 을 non-gated 미러에서 받아 `DATA_ROOT/imagenet_{val,train}` 에 저장(`labels.csv`+`images/`+`DONE`). 이미 `DONE` 이면 건너뜁니다.
- **`warmup_models.py`** — DINOv2-reg S/B/L 가중치를 미리 받아 캐시(새 서버 오프라인 함정 방지).
- **`run_val_sanity.sh`** — val-LOO faithful 5종 재현(빠름). 새 환경 검증용.
- **`run_train_gallery.sh`** — ★ 메인. `compare_faithful.py --gallery train` 으로 ToMe·PiToMe·Ours 를 train 갤러리에서. 태그(`s b l`)마다 GPU 하나씩 병렬. `SEQUENTIAL=1` 이면 순차.
- **`run_ablation_regcount.sh`** — train 갤러리에서 ablation + reg-count(+CI).
- **`run_all.sh`** — 위 전 과정을 순서대로.
- **`engine/`** — 실제 연산 코드. `compare.py`(train/val 갤러리 엔진), `compare_faithful.py`(정식 forward 주입), `faithful_tome_h2h.py`·`faithful_pitome_h2h.py`·`eval_ablation_faithful.py`·`reg_count_sweep_faithful.py`·`retrieval_map_faithful.py`(A100 에서 검증된 val 스크립트 원본), `ablation_train_faithful.py`·`regcount_train_faithful.py`(위 검증된 forward/엔진을 train 갤러리로 잇는 신규 얇은 래퍼), `tome_core.py`(공용 코어).

> 참고: `engine/` 안의 val 스크립트들은 A100 에서 검증된 원본을 **그대로** 복사했습니다. `compare.py` 만 데이터 위치를 `DATA_ROOT` 로 읽도록 최소 수정했고(원본은 자기 폴더에서 읽음), 신규 `*_train_faithful.py` 두 개는 검증된 `forward`·kNN 엔진만 재사용합니다.

---

## 5. 예상 시간·산출물

- **데이터**: val 수 분, train 1.28M 다운로드+리사이즈 저장은 회선·디스크에 따라 **수 시간**. 한 번 받으면 재사용.
- **train 갤러리 실험**: 모델당 대략 ToMe/PiToMe/Ours × r 마다 train 128만 장 특징추출이 있어 **모델당 수 시간~하루**(A6000 는 A100 보다 조금 느림). `--gallery_cache 1` 로 특징을 디스크에 캐시하므로 **중단해도 이어서** 됩니다.
- **결과 위치**: `results/`
  - `canonical_faithful_{s,b,l}.txt` — 헤드라인 표(comp% 별 ToMe/PiToMe/Ours, Δ(ours−pitome)).
  - `ablation_train_faithful_{tag}.txt`, `regcount_train_faithful_{tag}.txt`.
  - `val_*.txt` — val-LOO 대조.

---

## 6. 결과 해석 (기대치)

**val 대조**(`run_val_sanity.sh`, ViT-B, 50k)가 A100 과 일치해야 합니다:
- 무압축 kNN ≈ **76.33**.
- 정식 ToMe 대비 Ours: 압축이 셀수록 벌어져 **+10.29@91%**.
- 정식 PiToMe 대비 Ours: **전 구간 우세**(온건 +0.66 ~ 극단 +8.16).
- ablation: 91% 압축에서 Ours 만 유지(72.7), random/energy/highnorm ≈ ToMe(62 대).

**train 갤러리**(`run_train_gallery.sh`)에서 볼 점:
- 무압축(r=0)은 세 방법이 같아야 하고, DINOv2 공식 kNN(≈82)에 가깝게 나옵니다.
- 압축구간에서 **Ours > ToMe(전 구간), Ours > PiToMe(극단, 교차점 토큰 ~74% 부근)** 순서가 val-LOO faithful 과 같은 그림이면, 헤드라인을 train-갤러리로 승격할 근거가 완성됩니다.

---

## 7. 문제 해결 (새 서버 함정)

1. **`torch.cuda.is_available()` 가 False** → 서버 CUDA 에 맞는 torch 휠 재설치:
   `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` (버전은 서버에 맞게).
2. **timm 가중치 다운로드 실패(오프라인 오류)** → `warmup_models.py` 를 먼저 돌리세요. 실험 스크립트는 `config.sh` 를 통해 `HF_HUB_OFFLINE=0` 으로 실행되므로, 워밍업만 끝나면 이후는 캐시로 동작합니다.
3. **데이터가 없다는 assert** (`val 미준비`/`train 미준비`) → `prepare_data.py --split val` / `--split train --per_class 1300` 을 먼저. 반드시 `config.sh` 를 source 한 셸(또는 `run_*.sh`)에서 실행해야 같은 `DATA_ROOT` 를 봅니다.
4. **디스크 가득** → `export DATA_ROOT=/큰디스크/경로` 로 바꾼 뒤 처음부터. 특징 캐시(`$DATA_ROOT/feat_cache`)도 큽니다.
5. **train 다운로드가 너무 느림** → NVMe 로 옮기고 `WORKERS=16 bash run_...` 로 워커를 늘려 보세요.

---

## 8. 파일 목록

```
all_new_server/
├── README.md                     ← 이 문서
├── requirements.txt              ← 의존성(torch/timm/datasets/…)
├── setup_env.sh                  ← venv + 설치 + 검증
├── config.sh                     ← 공통 환경변수(DATA_ROOT 등)
├── prepare_data.py               ← val 5만 + train 1.28M 다운로드
├── warmup_models.py              ← DINOv2-reg S/B/L 가중치 캐시
├── run_all.sh                    ← 전 과정 순서 실행
├── run_val_sanity.sh             ← val-LOO faithful 재현(환경 대조)
├── run_train_gallery.sh          ← ★ train-갤러리 faithful 헤드라인(S/B/L 병렬)
├── run_ablation_regcount.sh      ← train-갤러리 ablation + reg-count
└── engine/                       ← 실제 연산 코드(검증본 복사 + 신규 train 래퍼 2개)
```
