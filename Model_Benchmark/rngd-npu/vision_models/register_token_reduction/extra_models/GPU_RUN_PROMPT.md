# GPU 서버 실행 안내 (사용자용 — GPU 실행/전송은 사용자 몫)

이 번들(`extra_models/`)을 GPU 서버로 옮겨 아래 순서로 실행하세요. 저는 코드만 준비했고, 실제
GPU 실행·서버 전송은 하지 않습니다.

## 0) 전송

```bash
# 로컬 → GPU 서버 (예: jun@164.125.249.13:10022)
scp -P 10022 -r extra_models jun@164.125.249.13:~/extra_models
```

## 1) 환경

```bash
cd ~/extra_models
bash setup_env.sh && source .venv/bin/activate
# torch.cuda.is_available() 가 False 면 서버 CUDA 에 맞는 torch 휠 재설치(안내는 setup_env.sh 출력)
```

## 2) 데이터 (용량 큼 — 큰 디스크 경로면 config.sh 의 DATA_ROOT 수정)

```bash
python prepare_data.py --split val
python prepare_data.py --split train --per_class 1300
```

## 3) 어댑터 정확성 게이트 (반드시 PASS 확인)

```bash
python selfcheck.py --model dinov3_base     # check1 cosine>0.999 = forward 가 모델 정확 재현
python selfcheck.py --model dinov3_splus
```
- FAIL 이면 README 의 "⚠️ selfcheck 리스크" 항목대로 `models_extra.py` 의 rope 접근을 고친 뒤 재검.
  (ViT-5 는 rope 테이블 버퍼 이름이 repo 와 다를 수 있어 이 단계가 특히 중요)

## 4) 실험

```bash
bash run_dinov3.sh          # DINOv3-B, S+ (train 갤러리) + reg-count 스윕

# ViT-5 (공식 repo·ckpt 준비)
git clone https://github.com/wangf3014/ViT-5 ~/ViT-5
huggingface-cli download FengWang3211/ViT-5 vit5_base_patch16_224.pth --local-dir ~/vit5_ckpt
export VIT5_REPO=~/ViT-5 VIT5_CKPT=~/vit5_ckpt/vit5_base_patch16_224.pth
bash run_vit5.sh
```

## 5) 결과 회수

`results/extra_dinov3_*` , `results/extra_vit5_*` 를 로컬로 가져오면 제가 논문(일반성 절)에 반영합니다.
```bash
scp -P 10022 -r jun@164.125.249.13:~/extra_models/results ./extra_models_results
```
