# extra_models — DINOv3 · ViT-5 추가 실험 (faithful, 전체 데이터셋)

이 폴더는 논문 헤드라인(DINOv2-reg) 외에 **다른 아키텍처의 register 모델**에서도 우리의
register-보호 토큰축소가 성립하는지 보려고, DINOv3(-S+/-B)와 ViT-5를 **공식 소스에서 그대로**
가져와 같은 faithful 프로토콜로 돌리는 독립 번들입니다. 기존 검증 코드는 수정하지 않고
`engine/`(kNN·데이터 파이프라인)만 읽기전용으로 재사용합니다.

---

## 🚀 빠른 실행 — GPU 서버에서 이것만 하면 끝

**전제**: 리눅스 + **NVIDIA GPU** 서버, 인터넷 접속, 디스크 여유 ~60GB+(ImageNet train 1.28M).
이 폴더(`extra_models/`)를 GPU 서버에 두고, 그 안에서:

```bash
cd extra_models
bash run_all.sh        # 환경설치 → selfcheck → 데이터(val 50k+train 1.28M) → DINOv3-S+/B → ViT-5-B
```

이 한 줄이 **전부 자동**으로 합니다: 파이썬 venv 설치 → 어댑터 정확성 게이트(selfcheck) →
ImageNet 다운로드 → 세 모델 실험. (train 1.28M 다운로드+추출 때문에 **수 시간** 걸립니다.
`tmux`/`screen` 안에서 돌리길 권장.) 큰 디스크를 쓰려면 먼저 `export DATA_ROOT=/큰디스크/경로`.

**끝나면 나오는 결과** (이 파일들을 회수해서 저(Claude)에게 주시면 논문 §일반성에 반영):
```
results/extra_dinov3_base_train_faithful.txt      # 각 r 에서 ours vs noreg 정확도 + Δ
results/extra_dinov3_splus_train_faithful.txt
results/extra_vit5_base_train_faithful.txt
results/extra_*_regsweep.txt                       # k=0..4 register 보호 스윕
```

> ✅ 이 어댑터들은 **로컬 CPU에서 공식 forward와 cosine=1.0으로 이미 검증**됐습니다(아래 표).
> `run_all.sh`는 실험 전에 서버에서도 selfcheck를 다시 돌려 PASS를 확인하니, 그 로그에
> `check1 ... cosine=1.000000 -> PASS`가 뜨는지만 봐 주세요.

**수동/개별 실행**을 원하면 아래 [실행 순서(수동)](#실행-순서-수동) 참고. 로컬에서 서버로 전송까지
한 번에 하려면 `launch_on_gpu.sh`(scp+ssh 포함, 서버 비번 입력) 사용.

---

## 모델과 공식 출처 (임의 환경 생성 없음)

| 모델 | 로드 방법 | 출처 |
|---|---|---|
| DINOv3-S+/16 | `timm.create_model('hf_hub:timm/vit_small_plus_patch16_dinov3.lvd1689m', pretrained=True)` | timm 비게이트 미러(공식 LVD-1689M 가중치) |
| DINOv3-B/16 | `timm.create_model('hf_hub:timm/vit_base_patch16_dinov3.lvd1689m', pretrained=True)` | timm 비게이트 미러 |
| ViT-5-B/16 | 공식 repo `wangf3014/ViT-5`의 `models_vit5.py`로 빌드 + HF `FengWang3211/ViT-5/vit5_base_patch16_224.pth` 로드 | 저자 공식 repo·체크포인트 |

- DINOv3를 공식 게이트 가중치로 쓰려면: `--dinov3_hub`(`torch.hub.load('facebookresearch/dinov3', ...)`), 가중치 URL은 `DINOV3_WEIGHTS`.
- register 개수는 세 모델 모두 **4개**(DINOv2와 동일)로 확인됨. 토큰 순서만 다름:
  - DINOv3: `[CLS, reg×4, patch...]` (rope는 patch만, prefix 건너뜀)
  - ViT-5: `[CLS, patch..., reg×4]` (patch=rope / register=rope_reg(theta=100) / CLS=rope 없음)

## ★ 설계 결정 — rope 모델에서 "faithful"이 자명하지 않다 (논문에 명시)

DINOv2-reg에서는 register가 patch와 **동질**(공유 절대 위치임베딩)이라 "ToMe가 register를 patch에
병합해 없앤다"가 자연스러운 baseline이었습니다. **DINOv3·ViT-5는 rotary(rope) 어텐션**이라
register가 patch와 다른 위치공간에 있습니다. register를 patch pool에 섞어 병합하면 rope 정렬이
깨져, 그 결과의 성능 저하가 "register 정보 손실" 때문인지 "rope 오정렬" 때문인지 분리되지 않습니다.

그래서 이 번들은 **rope를 깨지 않는, model-exact한 두 전략만** 비교합니다:

- **`ours`** — CLS + register 전부 보호, **patch만** size-가중 병합.
  - r=0에서 공식 forward와 **일치**(`selfcheck.py`가 cosine≈1로 검증). 즉 우리 forward는 모델을 정확히 재현.
  - 병합 후 생존 patch는 **원래 grid 위치의 rope를 유지**(pos-tracking): 병합된 토큰은 흡수하는
    dst 토큰의 위치를 물려받음. rope는 매 블록 생존 patch 위치로 재-gather.
- **`noreg`** — register를 시퀀스에서 **제거**(같은 가중치의 '레지스터 없는 모델'), patch만 병합.
  - 토큰을 지우는 건 rope-안전(남은 patch의 rope 불변)하므로 오정렬 아티팩트 없음.
  - `Δ = ours − noreg` = **압축 하에서 register가 기여하는 정확도**. rope 효과와 섞이지 않음.
- (옵션) **`--regsweep`** — k=0..4 register 보호 스윕. 논문의 reg-count 실험을 타 아키텍처에서 재현.

이 정의는 선행연구(ToMe/PiToMe)를 깎아내리지 않습니다. "register-비인지 병합이 rope 모델에선
아예 정의가 애매하다"는 건 우리 기법의 **필요성**을 보여주는 관찰이지, 남 논문의 흠이 아닙니다.

> 만약 저자(사용자)가 "rope 오정렬을 감수하고 naive-ToMe(CLS만 보호, register도 병합) baseline을
> 넣자"고 결정하면, 그 변형은 `run_extra.py`에 전략 추가로 확장 가능합니다. 기본값은 위의
> 깨끗한 `ours` vs `noreg`입니다.

## 실행 순서 (수동)

`run_all.sh`가 아래를 전부 자동으로 합니다. 단계별로 직접 돌리고 싶을 때만 아래를 따르세요.

```bash
cd extra_models
bash setup_env.sh && source .venv/bin/activate     # torch/timm>=1.0.20/datasets/einops
# 데이터(용량 큼): val 50k + train 1.28M — 큰 디스크면 config.sh 의 DATA_ROOT 를 그 경로로
python prepare_data.py --split val
python prepare_data.py --split train --per_class 1300

# 1) 어댑터 정확성 게이트 — 반드시 PASS 후 진행
python selfcheck.py --model dinov3_base
python selfcheck.py --model dinov3_splus

# 2) DINOv3 실험(train 갤러리)
bash run_dinov3.sh                                  # 결과: results/extra_dinov3_*

# 3) ViT-5 — 공식 repo·ckpt 준비 후
git clone https://github.com/wangf3014/ViT-5 /path/ViT-5
huggingface-cli download FengWang3211/ViT-5 vit5_base_patch16_224.pth --local-dir /path/vit5_ckpt
export VIT5_REPO=/path/ViT-5  VIT5_CKPT=/path/vit5_ckpt/vit5_base_patch16_224.pth
bash run_vit5.sh                                    # 결과: results/extra_vit5_*
```

## ✅ 어댑터 검증 완료 (로컬 CPU, 공식 가중치/소스, 2026-07-04)

세 모델 어댑터를 **로컬 CPU에서 실제로 실행**해 `selfcheck.py`가 전부 PASS함을 확인했습니다
(임시 CPU venv + 공식 timm 미러 가중치 + 공식 ViT-5 repo clone). 즉 우리 rope-aware faithful
forward가 **r=0에서 공식 forward와 정확히 일치**(정보 손실 0)합니다:

| 모델 | check1 (r=0 == 공식) | check2 (r=12 병합) | check3 (noreg) |
|---|---|---|---|
| DINOv3-B  | **PASS** cosine=1.000000, rel_l2=0 | PASS | PASS |
| DINOv3-S+ | **PASS** cosine=1.000000, rel_l2=0 | PASS | PASS |
| ViT-5-B   | **PASS** cosine=1.000000, rel_l2=0 | PASS | PASS |

또 `run_extra.py`의 특징추출→kNN 배관도 실제 모델 forward로 end-to-end 동작 확인(무작위 상회).
**전체 규모 kNN(train 1.28M 갤러리 + val 50k, 표준 정확도 수치)만 GPU가 필요**하고, 이 머신은
Furiosa NPU 박스(NVIDIA GPU 없음)라 그 부분은 GPU 서버에서 사용자가 실행합니다(`GPU_RUN_PROMPT.md`).

### 검증 중 발견·수정한 실제 이슈 (모두 코드에 반영됨)
1. **DINOv3 rope broadcast**: timm `apply_rot_embed_cat`은 배치별 rope를 head 축으로 broadcast해야 해
   `[B,1,Pcur,rope_dim]` 형태 필요(처음 `[B,Pcur,dim]`으로 넘겨 H vs B 충돌). → `unsqueeze(1)`로 수정.
2. **DINOv3 register 제거(noreg) 시 prefix 불일치**: `EvaAttention.num_prefix_tokens`가 5로 고정돼 있어
   register를 빼면 rope가 엉뚱한 토큰에 걸림. → forward 동안 현재 npt로 동기화 후 복원.
3. **ViT-5 rope는 2D-공간 그리드 락 + `.cuda()` 하드코딩**: `VisionRotaryEmbedding`은 매번
   `ft_seq_len=sqrt(현재 토큰수)`로 정사각 그리드를 재구성(병합 시퀀스에 직접 못 씀)하고 `freqs_cos/sin`
   버퍼도 없음(즉석 계산). → 전체 그리드 freqs 테이블을 device-무관하게 재구성 후 **생존 patch의 원래
   2D-grid 위치로 gather**(pos-tracking)하도록 ViT-5 forward를 공식과 일치하게 재구현(interleaved
   rotate_half, transpose 순서, qk_norm 포함). selfcheck는 CPU 검증 위해 rope의 `.cuda()`를 패치.

GPU 서버에서도 실험 전 `selfcheck.py`를 한 번 돌려 PASS를 재확인하세요(환경/버전 차이 방어).

## 결과 → 논문 반영

- `results/extra_dinov3_{base,splus}_train_faithful.txt`, `results/extra_vit5_base_train_faithful.txt`
  → "다양한 아키텍처 일반성" 절: **ours가 압축 하에서 정확도 하락을 작게 유지**, `Δ(ours−noreg)>0`.
- `results/extra_*_regsweep.txt` → reg-count 스윕이 타 아키텍처에서도 단조 증가하는지.
- 프로토콜은 헤드라인과 동일(train 갤러리 1.28M, val 50k 쿼리, 표준 kNN, faithful). 외부 논문 수치
  인용 없이 같은 잣대로 직접 측정.
