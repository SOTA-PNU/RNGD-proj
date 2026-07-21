# deit_compare — 우리 PiToMe 포팅이 공식과 같은지 DeiT에서 교차검증 (GPU 서버 자동)

이 폴더는 우리 논문이 쓴 **PiToMe 포팅**(그리고 ToMe)이 **공식 PiToMe와 실제로 같은지**를 DeiT에서 확인하는 자동 도구 모음입니다. 논문 헤드라인 비교는 DINOv2-reg에서 하지만, 공식 PiToMe repo는 옛 라이브러리(`timm==0.4.12`)에 묶여 DINOv2-reg를 못 올립니다. 그래서 **둘 다 돌릴 수 있는 공통 모델 DeiT**에서 맞춰봅니다.

## 빠른 시작 (GPU 서버 · 한 방)
```bash
cd deit_compare
bash run_all.sh
```
이 한 줄이 **①우리 포팅 + ②공식 repo 실측 + 대조 리포트**를 자동으로 다 합니다. 마지막에 "①↔② 나란히 대조표 + 판정(✅/⚠️)"이 찍힙니다.

- **필요한 것**: `conda`(② 격리 env 자동 생성용), CUDA GPU, 인터넷(repo 클론·체크포인트·val 미러). 
- **HF 토큰은 필요 없습니다.** gated 데이터를 안 씁니다(비-gated val 미러 + FB 체크포인트 + 공식 *알고리즘 코드*만 사용). → 아래 "HF 토큰" 참고(선택).
- **대략 시간**: 첫 실행 ~30~60분(② conda env·토치 설치 포함), 이후 ~30~50분. ① 만 따로면 ~10~20분.

부분 실행:
```bash
bash run.sh                 # ① 우리 포팅만 (현재 env, timm>=1.0) — 가볍게 먼저
bash run_official_pitome.sh # ② 공식 repo 실측만 (전용 conda env, timm==0.4.12)
python compare_report.py    # 저장된 결과로 대조표만 다시 출력
```

## 왜 필요한가 (한 문단)
우리는 공식 PiToMe 알고리즘을 우리 코드로 옮겨 적었습니다(포팅). "옳게 옮겼는가"를 증명하려면, 공식이 돌아가는 모델(DeiT)에서 **우리 포팅**과 **공식 코드**를 같은 조건으로 돌려 결과가 겹치는지 보면 됩니다. DeiT에서 일치하면, 같은 코드를 쓰는 DINOv2-reg 논문 수치도 신뢰할 수 있습니다.

## 어떻게 "공식 결과"를 자동으로 얻나 (핵심)
`run_official_pitome.sh` → `official_deit_driver.py`는 공식 repo를 클론한 뒤, **그들의 실제 알고리즘 코드**(`algo.pitome.patch.deit.apply_patch`)를 그들 env(`timm==0.4.12`)에서 불러 DeiT에 적용하고, **우리 로컬 val**로 평가합니다. 즉 "공식 코드가 낸 수치"를 얻으면서도, 그들의 **gated imagenet-1k 다운로드(수십 GB)·HF 로그인**을 통째로 건너뜁니다. 검증된 공식 API를 그대로 씁니다:
- `apply_patch(model)` → `model.ratio = <보존비율>` 설정(패치는 ratio를 안 받고 내부에서 마진 스케줄 `0.75→0` 자동).
- `logits, flop = model(x)` (패치된 forward는 튜플 반환).
- 전처리 = 공식 `main_ic` 그대로: Resize(256,bicubic)→CenterCrop(224)→Norm. **우리 포팅(①)도 같은 전처리로 통일**해 사과-대-사과.

## 파일
- `deit_compare.py` — **①** 우리 포팅(`compare.py`의 `merge_step`·`pitome_step` **그대로**)을 공식 스케줄로 DeiT에 적용, 분류 top-1 측정 → `results/ours_port__*.json`.
- `official_deit_driver.py` — **②** 공식 `apply_patch`를 불러 DeiT 실행 → `results/official__*.json`.
- `compare_report.py` — ①↔② 나란히 대조표 + 판정. 공식이 없으면 우리 곡선 + 공개 참조치.
- `run_all.sh` — ①+②+리포트 한 방. / `run.sh` — ①만. / `run_official_pitome.sh` — ②만(conda env 자동).
- `prepare_data.py` — ImageNet val 준비(비-gated 미러, 표준 라벨순서). `pitome_compare/imagenet_val` 있으면 재사용.
- `requirements.txt` — ① 쪽(우리 env, timm≥1.0) 의존성.

## 안전장치 (자체검증)
- **r=0 무압축 정확도**를 공식 baseline과 대조해 `✅일치`/`⚠️불일치`를 찍습니다(DeiT-S=79.8, DeiT-T=72.3). 라벨순서·전처리·헤드가 틀리면 여기서 바로 드러납니다.
- 대조표 **판정**: 같은 ratio에서 `|우리_pitome − 공식_pitome|`(및 baseline 차)이 0.5%p 안이면 `✅ 포팅=공식 일치`.

## HF 토큰 (선택 — 필수 아님)
gated 데이터를 안 쓰므로 **원래 토큰이 필요 없습니다.** 다만 HF에서 val 미러를 받을 때 익명 다운로드가 rate-limit에 걸리면, 토큰을 넣어 완화할 수 있습니다:
> **여기 붙여넣기**: `deit_compare/hf_token.txt` 파일을 만들어 토큰 한 줄만 넣으면 `run_official_pitome.sh`가 자동으로 `HF_TOKEN`으로 씁니다.

## 중요한 주의점
- **공식 논문에 DeiT-B는 없습니다.** 분류 백본은 DeiT-T/S만 보고돼 있어, 대조는 **DeiT-S**(baseline 79.8, PiToMe−ToMe≈+1.4)와 DeiT-T(72.3, ≈+1.9)로 합니다. DeiT-B도 돌릴 수는 있으나(파이프라인만 확인) 공개 참조치가 없습니다.
- **② 는 별도 conda env**(옛 timm 0.4.12)를 자동 생성합니다 — 우리 최신 env와 안 섞입니다. `salesforce-lavis`(ITR 전용, 무거움)와 `wandb`는 설치에서 제외합니다(알고리즘 import엔 불필요).
- **CUDA 걱정 불필요**: 공식 env의 `pytorch-cuda=11.8`은 conda가 CUDA 런타임을 자체 번들하므로 최신 드라이버 서버에서도 GPU로 돕니다.

## 결과 해석
r=0가 공식 baseline과 맞고, 대조표에서 우리 pitome/tome이 공식과 0.5%p 안으로 겹치면 → **우리 포팅이 공식 PiToMe를 충실히 구현했음**이 실측으로 확인되고, DINOv2-reg 논문의 PiToMe 비교 수치가 정당화됩니다. 어긋나면(⚠️) 포팅의 마진 스케줄·에너지식·병합 순서를 공식과 재대조하면 됩니다.
