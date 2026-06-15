# PRO 6000 서버 vLLM 셋업 (Blackwell sm_120)

RTX PRO 6000 은 Blackwell 세대(compute capability **sm_120**, 96GB)라서, 구형 vLLM/PyTorch 휠
(예: 기존 `bench-gpu` 의 vLLM 0.10.0 **cu126**)은 `CUDA error: no kernel image is available` /
`sm_120 is not compatible` 로 죽습니다. Blackwell 은 **CUDA 12.8 이상**이 필요합니다.

우리 대상 모델 `Qwen/Qwen3-32B-FP8` 은 **dense block-wise FP8(block 128, e4m3, W8A8)** 이고,
이 경로는 vLLM 에 sm_120 CUTLASS 커널로 머지돼(PR #22131) 단일 PRO 6000 96GB 에서 잘 돕니다.
(터지는 건 주로 MoE/NVFP4 경로 — 우리 모델은 dense라 무관.)

> 정직성 노트: "정확히 어느 vLLM stable 버전부터 stock cu129 휠이 sm_120 을 포함하는가"는 출처마다
> 시점이 달라 단정 못 합니다. 아래 **3단 폴백**이 실무상 가장 안전합니다.

---

## 0. 드라이버 확인

```bash
nvidia-smi   # CUDA 12.8+ 지원 드라이버여야 함. 카드/드라이버/CUDA 버전 확인.
```

## 1. (1순위) 최신 stock vLLM — 되면 끝

```bash
cd ~/bench-blog            # 이 디렉토리(레포 Model_Benchmark/bench-blog) 를 GPU 서버에 복사해 둔 위치
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -U vllm        # 기본 cu129 휠 (Blackwell은 CUDA 12.8+ 필요)
pip install httpx          # loadgen 의존성 (matplotlib 는 compare.py 차트용, 선택)

# 동작 확인 — (12, 0) 이 떠야 Blackwell 커널 포함된 빌드
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"
```
`(12, 0)` 이 나오면 2번으로. `no kernel image` 또는 capability 미인식이면 폴백:

## 2. (2순위) 명시적 cu128 휠

> ⚠ **torch/nccl 을 손으로 핀하지 마세요.** `torch==X` 를 따로 깔고 그 위에 `vllm` 을 깔면 vllm 이
> torch 를 다른 버전으로 끌어올리면서 **NCCL 심볼 불일치**(`undefined symbol: ncclDevCommDestroy`)가
> 납니다. **vLLM 하나만 깔아 torch+nccl 을 통째로 맞추는 게 안전합니다.**

```bash
# 깨끗한 venv 에서 vLLM 만 — torch+nccl 을 vLLM 이 일관되게 끌어옴
pip install -U vllm --extra-index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"   # (12,0) 확인
# 그래도 NCCL 심볼 에러가 나면 nccl 만 최신으로:
#   pip install -U nvidia-nccl-cu12
```

## 3. (최후수단) 소스 빌드 — 확실히 동작

```bash
export TORCH_CUDA_ARCH_LIST="12.0"          # Blackwell. (5090 호환도 원하면 "8.9;12.0")
git clone https://github.com/vllm-project/vllm.git && cd vllm
pip install -r requirements/build.txt
pip install --no-build-isolation -e .
```

---

## 흔한 에러 → 해결

| 증상(로그/import) | 원인 | 해결 |
|---|---|---|
| `import torch` 시 `undefined symbol: ncclDevCommDestroy` (또는 다른 nccl 심볼) | **torch ↔ nccl 버전 불일치** (설치된 nvidia-nccl-cu12 가 torch 가 요구하는 것보다 구버전). torch/nccl 을 따로 핀했을 때 흔함 | `pip install -U nvidia-nccl-cu12` → 그래도면 venv 새로 만들어 `pip install -U vllm` 하나만 |
| serve 시 `CUDA error: no kernel image is available` / `sm_120 ... not compatible` | torch/vLLM 빌드에 **Blackwell(sm_120) 커널 미포함** | 2순위(cu128) → 안 되면 3순위(소스 빌드, `TORCH_CUDA_ARCH_LIST=12.0`) |
| `torch.cuda.get_device_capability()` 가 `(12,0)` 이 아님/`NOCUDA` | 위 커널 미포함 또는 드라이버 문제 | 드라이버(CUDA 12.8+) 확인 후 재설치 |
| EngineCore `Failed core proc(s): {}` 로 즉사 | EngineCore 서브프로세스 하드 크래시 — 보통 위 nccl/sm_120 중 하나 | `results/pro6000_serve.log` 상단의 진짜 원인 확인(run_pro6000.sh 가 자동 추출) |
| FP8/quant/cutlass 관련 에러 | vLLM 이 너무 구버전이라 **sm_120 block-FP8 커널 미포함**(PR #22131) | `pip install -U vllm` 로 최신화. 급하면 `MODEL=Qwen/Qwen3-32B`(bf16)로 파이프라인부터 확인 |
| `out of memory` (FP8 33GB 인데 OOM) | 커널 미스로 full-precision 폴백돼 VRAM 폭증 | sm_120 빌드 재점검(=no kernel 케이스). 임시로 `GPUUTIL=0.85` |

> `run_pro6000.sh` 는 실행 전 `torch.cuda.get_device_capability()` 를 preflight 로 찍고, 실패 시
> 로그에서 위 키워드(`no kernel image`, `ncclDevCommDestroy`, `out of memory` 등)를 자동 추출합니다.

---

## 4. serve 동작 점검 (run_pro6000.sh 가 자동으로 하지만, 수동 점검용)

```bash
# 공식 W8A8 FP8 체크포인트(= RNGD FP8 과 정밀도 매칭). --quantization 플래그 불필요(config 가 fp8 명시).
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-32B-FP8 \
  --tensor-parallel-size 1 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --max-num-seqs 256 --port 8000
# 다른 셸에서:
curl -s 127.0.0.1:8000/v1/models | python3 -m json.tool
```

### 메모리 (1장 96GB) — 충분
- weight FP8 ≈ **33GB**, 남는 ~63GB 가 KV+activation. bf16 KV 로도 32K 컨텍스트 수십~수백 동시 시퀀스 여유.
- FP8 모델이 예상보다 훨씬 큰 메모리를 먹으면(예: 80GB+) **커널 미스로 full-precision 폴백** 신호 → 설치(아키 플래그) 재점검.

### 주의
- **`--kv-cache-dtype fp8`** 로 KV 절약 가능하나 일부 환경서 fp8 KV silent corruption 보고 있음 → 정확도 민감하면 기본(bf16)으로 두고 비교(run_pro6000.sh 기본 `KVDTYPE=auto`).
- **INT8 불가**(sm_120) — FP8 로 가야 함.
- 멀티 Blackwell(NVLink 없는 2장)일 때만 `--disable-custom-all-reduce` 필요. **1장이면 불필요.**
- reasoning 플래그(`--reasoning-parser qwen3` 등)는 chat 파싱용 — 우리 부하시험은 `/v1/completions`(raw prompt)라 **불필요**. run_pro6000.sh 는 안 붙입니다.

---

## 5. 실행

```bash
# (설치된 .venv 활성화 상태에서)
GPU=0 ./run_pro6000.sh                 # 결과 results/pro6000.json
# 그 다음 RNGD 서버에서 만든 rngd.json 을 가져와 비교:
python compare.py results/rngd.json results/pro6000.json --out results/report.md
```

## 출처
- vLLM GPU 설치(CUDA 12.9 기본/12.8 최소): https://docs.vllm.ai/en/stable/getting_started/installation/gpu/
- Qwen3-32B-FP8 모델카드(FP8 block128, vllm≥0.8.5): https://huggingface.co/Qwen/Qwen3-32B-FP8
- vLLM PR #22131 (SM120 block FP8 CUTLASS, RTX PRO 6000): https://github.com/vllm-project/vllm/pull/22131
- vLLM 포럼: RTX 6000 Blackwell 96GB 셋업/FP8 권장/INT8 불가: https://discuss.vllm.ai/t/support-for-rtx-6000-blackwell-96gb-card/1707
- vLLM issue #35432 (prebuilt 휠 sm_120 실패): https://github.com/vllm-project/vllm/issues/35432
- `vllm bench serve` (대안 부하도구): https://docs.vllm.ai/en/stable/cli/bench/serve/
