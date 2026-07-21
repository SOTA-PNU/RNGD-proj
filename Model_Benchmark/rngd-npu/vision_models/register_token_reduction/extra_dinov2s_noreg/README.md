# DINOv2 no-reg를 train-갤러리(1.28M)로 재측정 — tab:extra 프로토콜 정합

이 폴더는 논문 표 `tab:extra`의 **DINOv2 no-reg 행**을 나머지 표와 **같은 프로토콜**로 맞추기 위한 실험입니다.

## 무엇이 문제였나 (프로토콜 불일치)

`tab:extra` 캡션은 "표준 train-갤러리 kNN top-1"입니다. 그런데 각 행의 실제 출처를 보면:

| 행 | 갤러리 | 출처 |
|----|--------|------|
| DINOv2-B/S **Ours** | **train 1.28M** | `canonical/faithful_results/canonical_faithful_{base,small}.txt` |
| DINOv3-S+/B, ViT-5 **Ours·no-reg** | **train 1.28M** (`gallery=train`) | `extra_models/results/*_train_faithful.txt` |
| DINOv2-B **no-reg** (기존 75.85…) | **50k val self-kNN** ✗ | `results/dinov2_noreg_control.txt` (`eval_imagenet.py --n 50000`, self 제외) |
| DINOv2-S **no-reg** | (없음) | 미측정 |

즉 **DINOv2 no-reg만 50k val 프로토콜**이라 캡션·다른 행(train-갤러리)과 어긋납니다. 절대 수치 기준도 달라(train-갤러리가 더 높음) 직접 비교가 부정확합니다.

## 고침

DINOv2 no-reg를 DINOv3/ViT-5와 **똑같이 train-갤러리(1.28M)**로 다시 잽니다. 검증된 `canonical/run_canonical_faithful.sh`(= `compare_faithful.py --gallery train`)를 **레지스터 없는 DINOv2 체크포인트**로 호출만 합니다(원본 코드 수정 없음).

- no-reg 모델은 `prefix=1`(레지스터 0)이라 `ours`=`tome`(보호할 레지스터가 없음) → **no-reg 값 = `tome` 열**.
- DINOv2-S: `vit_small_patch14_dinov2.lvd142m` (표에 빠졌던 행)
- DINOv2-B: `vit_base_patch14_dinov2.lvd142m` (기존 50k-val 값을 train-갤러리로 **교체**)

## 실행 (GPU 서버)

```bash
cd .../vision_models/register_token_reduction
CUDA_VISIBLE_DEVICES=0 bash extra_dinov2s_noreg/run_dinov2s_noreg.sh small   # DINOv2-S no-reg
CUDA_VISIBLE_DEVICES=1 bash extra_dinov2s_noreg/run_dinov2s_noreg.sh base    # DINOv2-B no-reg (재측정 권장)
# 또는 순차: bash extra_dinov2s_noreg/run_dinov2s_noreg.sh both
```

- CUDA GPU + `imagenet_train`(1.28M) 준비 필요.
- **매우 무거움**: small ~8–10h, base ~20–25h(canonical faithful과 동일 규모). 2 GPU면 병렬.
- 결과: `canonical/faithful_results/canonical_faithful_{small_noreg,base_noreg}.txt`

## 반영

각 로그의 `tome` 열에서 r=0(무압축)/12(~55%)/16(~74%)/20(~92%)를 읽어:
- `tab:extra`의 **DINOv2-S no-reg 행 추가**
- **DINOv2-B no-reg 행 교체**(50k-val 75.85… → train-갤러리 값)

로그를 그대로 보내주면 표를 채우고 교체합니다.

## 참고: 이미 있는 값 (Ours, train-갤러리)

| 토큰 축소 | DINOv2-S Ours | DINOv2-B Ours |
|-----------|---------------|---------------|
| 무압축 | 77.41 | 80.87 |
| ~55% | 75.80 | 80.15 |
| ~74% | 74.23 | 79.41 |
| ~92% | 69.85 | 77.28 |
