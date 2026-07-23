# ablation — register keep-prior 결정적 ablation (GPU 패키지)

이 폴더는 GPU 서버에 그대로 복사해 실행하는, "register-aware 토큰 압축" 논문의 **결정적 ablation** 실험입니다. 상위 `register_token_reduction/`의 본실험(`eval_imagenet.py`, ToMe vs Ours만)이 **하지 않는** 비교 — "register를 보호하는 게 정말 특별한가, 아니면 아무 토큰이나/에너지 기준으로 보호해도 같은가?" — 를 채웁니다.

## 왜 필요한가 (리뷰어가 반드시 묻는 베이스라인)
ToMe보다 Ours(register 보호)가 좋다는 것만으로는 부족합니다. 심사위원은 "그건 그냥 토큰을 더 보호해서, 또는 ToMe가 극단압축서 약해서 아니냐"라고 묻습니다. 그래서 **같은 보호 개수**로 다음을 비교해 register의 특수성을 격리합니다.

| 전략 | 보호 대상 | 의미 |
|---|---|---|
| `tome` | CLS만 | 보호 없음(바닥) |
| `ours` | CLS + register | 우리 방법 |
| `random` | CLS + 무작위 patch 4개 | "그냥 4개 더 보호"와 다른가 |
| `energy` | CLS + 저에너지 patch 4개 | PiToMe식 keep-prior와 다른가 |
| `highnorm` | CLS + 고노름 patch 4개 | register 없는 모델용 대리 |

병합 메커니즘은 정식 size-가중 ToMe(`tome_reg.py`, Bolya ICLR'23)로 **고정**하고 보호 집합만 바꿉니다.

## 실행
```bash
bash run.sh      # 의존성 → ImageNet val 50k 다운 → ablation(DINOv2-reg/CLIP/plain) → 그림
```
개별 실행:
```bash
export IMAGENET_VAL="$(pwd)/imagenet_val"
python prepare_data.py --per_class 50
python eval_ablation.py --model vit_base_patch14_reg4_dinov2.lvd142m --n 50000 --batch 128 --k 20 --r_list 8 12 16 18 20
python make_sweep_figure.py results/ablation_vit_base_patch14_reg4_dinov2_n50000.json
```

## 산출물
- `results/ablation_<model>_n<N>.json` — 압축률별 전략별 kNN top-1.
- `results/ablation_<model>_n<N>.png` — 압축률 vs 정확도 sweep 그림(논문 Figure).

## 회수
실행 후 `results/`를 로컬로 가져오세요:
```bash
scp -r -P <port> jun@<this-host>:~/register_token_reduction/ablation/results ./
```

## 주의
- **kNN의 k는 클래스당 이미지 수 이하**여야 신뢰됩니다(50장/클래스 → k=20 적정). 작은 표본에서 k가 크면 정확도가 눌리고 노이즈가 큽니다(로컬 CPU 소표본 결과가 그래서 노이즈가 컸음).
- register 없는 모델(CLIP/plain)은 `ours`가 `tome`와 같아져(degenerate) 의미가 없으므로 `highnorm`을 "ours의 대리"로 봅니다.
- 기대: **극단 압축(>90%)에서 `ours`(또는 highnorm)가 `random`·`energy`를 이김** → register/고노름 keep-prior의 특수성 입증. 중간 압축선 `tome`이 비슷하거나 약간 나을 수 있음(우리 영역=극단 압축).

## 파일
- `eval_ablation.py` — 5-way ablation 본체(전략별 CLS 특징 추출 → kNN, JSON 저장).
- `tome_reg.py` — size-가중 bipartite soft matching 병합(상위 폴더와 동일).
- `prepare_data.py` — ImageNet val 다운로드(HF non-gated, 토큰 불필요).
- `make_sweep_figure.py` — 압축률 sweep 그림.
- `run.sh` / `requirements.txt`.
