# Llama-3.3-70B FP8 양자화 (GPU 서버용)

폴더를 그대로 GPU 서버로 복사한 뒤 `./run.sh` 한 번으로 환경 구축 → 다운로드 → FP8 양자화까지 자동 진행됩니다.

## 빠른 시작

```bash
# 1) 이 폴더를 GPU 서버로
rsync -avh quantize_llama70b/ user@gpu-server:~/

# 2) GPU 서버에서
cd ~/quantize_llama70b
hf auth login            # 처음 한 번 (meta-llama gated 모델 접근용)
./run.sh                 # 전체 자동
```

결과: `./out/meta-llama--Llama-3.3-70B-Instruct-fp8/` (~70 GB)

## 단계별 실행

```bash
bash setup.sh                                      # ① venv + torch(CUDA) + transformers
bash download.sh meta-llama/Llama-3.3-70B-Instruct # ② 원본 BF16 다운로드 (~141 GB)
source ./venv/bin/activate
python quantize.py meta-llama/Llama-3.3-70B-Instruct  # ③ FP8 변환 (~70 GB)
```

다른 모델도 가능:
```bash
./run.sh Qwen/Qwen2.5-72B-Instruct
python quantize.py meta-llama/Llama-3.1-8B-Instruct -o ./out/llama8b-fp8
```

## 폴더 구성

```
quantize_llama70b/
├── run.sh              setup → download → quantize 한 번에
├── setup.sh            venv 만들고 torch(CUDA) + transformers 설치
├── download.sh         원본 가중치 다운로드 (safetensors 만, original/*.pth 제외)
├── quantize.py         FineGrainedFP8 양자화 메인 스크립트
├── requirements.txt    의존성 (torch 제외 — setup.sh 가 CUDA 매칭 설치)
├── README.md           본 문서
└── out/                양자화 결과 (자동 생성)
```

## 하드웨어 요구사항

| 항목 | 필요 |
|---|---|
| GPU | A100 80GB ×2 / H100 ×1 (BF16 70B = 141GB → `device_map="auto"` 가 자동 분산) |
| 디스크 | 원본 BF16 ~141 GB + 결과 FP8 ~70 GB → 여유 **250 GB+** |
| RAM | 32 GB+ (가중치는 GPU 로 올라가지만 일부 버퍼) |
| CUDA driver | 11.8 또는 12.x |
| Python | 3.10+ |

## 양자화 설정

[Furiosa 공식 문서](https://developer.furiosa.ai/latest/en/furiosa_llm/model-preparation.html) 따라:

```python
FineGrainedFP8Config(
    activation_scheme="dynamic",     # 동적 activation 스케일
    weight_block_size=(128, 128),    # 128×128 블록 단위 양자화
)
```

변경 옵션 (`quantize.py --help`):
- `--activation static` — static activation 양자화 (calibration 필요할 수 있음)
- `--block 64` — 더 작은 블록 (정확도 ↑, 메모리 ↓)

## 양자화 후 — NPU 서버로 옮기기

```bash
# GPU 서버 → NPU 서버
rsync -avh ./out/meta-llama--Llama-3.3-70B-Instruct-fp8/ \
    jun@npu-server:~/.cache/huggingface/hub/models--Llama-3.3-70B-Instruct-FP8/

# NPU 서버에서 furiosa-llm build
RAY_memory_monitor_refresh_ms=0 \
  furiosa-llm build ~/.cache/huggingface/hub/models--Llama-3.3-70B-Instruct-FP8 \
    ~/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/llama-3.3-70b-fp8-tp8pp2 \
    -tp 8 -pp 2     # FP8 70 GB 가 1장(48GB)에 안 들어가서 PP 필요
```

> Llama 는 `block_slicer.py` 에 PP 등록돼 있어서 `-pp 2` 가능.

## 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `No GPU or XPU found` | GPU 없는 머신. `nvidia-smi` 확인 후 GPU 머신에서 실행 |
| `Access denied. This repository requires approval` | meta-llama gated — HF 페이지에서 access 신청, `hf auth login` 다시 |
| 디스크 부족 | 원본 (`original/*.pth`) 까지 받지 않도록 `download.sh` 가 `--exclude` 사용 — 그래도 부족하면 `HF_HOME` 환경변수로 큰 디스크로 캐시 옮기기 |
| OOM (CUDA) | `device_map="auto"` 가 자동 분산 — GPU 추가 또는 더 큰 메모리 GPU 필요 |

## 대안 — 양자화 안 하고 받기

이미 양자화된 공개 체크포인트가 있으면 GPU 양자화 단계 스킵 가능:

```bash
hf download RedHatAI/Llama-3.3-70B-Instruct-FP8-block --exclude "original/*"
```

`furiosa-llm/optimum/modeling.py:84` 가 이 repo 를 Llama-3.3-70B FP8 의 인증된 변종으로 가리킵니다.
