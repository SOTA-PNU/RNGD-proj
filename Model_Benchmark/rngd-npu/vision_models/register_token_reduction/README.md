# register_token_reduction — GPU 서버용 실험 패키지

이 폴더는 **GPU 서버에 복사해 그대로 실행**하면 되는, "register-aware 토큰 압축" 논문의 본실험(Phase 1) 패키지입니다.

## 한 줄 주제
foundation 비전 인코더(특히 register를 가진 DINOv2-reg)에서, **register/큰 토큰을 보호**하며 중복 패치를 병합하면 **극단 압축(>90%)에서 표준 ToMe보다 정확도가 높고 압축에 견고**하다 — 재학습 없음.

## 빠른 실행
```bash
bash run.sh        # 의존성 → ImageNet val 다운 → 평가
```

## 핵심 결과(이미 CPU에서 검증된 추세, GPU서 풀스케일 재현 목표)
정식 size-가중 ToMe에서 **보호 토큰 수(n_protect)만** 바꿔 비교 (DINOv2-reg, prefix=cls+register4):
- `tome` = n_protect 1 (CLS만), `ours` = n_protect 5 (CLS+register)
- CPU 800장 예비: 92% 압축서 ours 58.75 vs tome 55.00 (**+3.75%**), ours는 74%→92%서 거의 안 떨어짐(견고). 중간(74%) 압축선 tome이 약간 나음 → **우리 영역은 극단 압축**.

## 파일
- `tome_reg.py` — size-가중 bipartite soft matching 병합(ToMe, Bolya ICLR'23 arXiv:2210.09461) + n_protect 보호. `reduced_forward()`가 블록마다 병합하며 forward.
- `eval_imagenet.py` — 압축률·보호전략별 CLS 특징 추출 → ImageNet val kNN 정확도. 디바이스 자동(cuda).
- `prepare_data.py` — ImageNet val 다운로드(HF non-gated, 토큰 불필요).
- `run.sh` / `requirements.txt`.

## 사용 예
```bash
python eval_imagenet.py --n 50000 --batch 128 --r_list 0 8 12 16 18 20            # DINOv2-reg 풀
python eval_imagenet.py --model vit_base_patch14_dinov2.lvd142m --n 50000          # register 없는 모델(고노름 보호는 TODO)
python eval_imagenet.py --model vit_base_patch16_clip_224.openai --n 50000         # CLIP
```

## Phase 1 할 일 (GPU)
1. 풀 ImageNet 50k로 위 표 확정(여러 모델·압축률).
2. **PITOME 비교**(에너지 점수 baseline) + **무작위 보호 대조군** 추가 → register 보호가 에너지/무작위보다 나은지 ablation.
3. **proportional attention**(log-size 비아스) 추가로 ToMe baseline 완성도↑(현재 병합 size-가중까지만).
4. **dense 작업**(분할 mIoU/깊이) — 패치 토큰 보존이 중요한 곳에서 더 큰 이득 기대(unmerge 복원).
5. 비-register 모델용 **고노름 토큰 보호**(per-image) 변형.

> NPU(RNGD)는 보조: 토큰 줄인 모델 실칩 지연 측정(forward-only 컴파일됨). 상위 폴더 도구 참고.
