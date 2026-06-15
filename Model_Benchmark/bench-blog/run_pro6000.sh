#!/usr/bin/env bash
# RTX PRO 6000 측 블로그 재현 — vLLM 으로 공식 Qwen/Qwen3-32B-FP8(W8A8) 을 1장에 serve 하고 loadgen 측정.
# 결과: results/pro6000.json  (compare.py 에서 rngd.json 과 합쳐 리포트)
#
# 전제: Blackwell(sm_120)용 vLLM 설치 완료. 설치는 setup_gpu.md 참고.
#   - 동작 확인: python -c "import torch;print(torch.cuda.get_device_capability())"  -> (12, 0)
#
# 사용:
#   ./run_pro6000.sh                          # GPU0, ISL1024/OSL256, b1..256
#   GPU=1 ISL=2048 OSL=512 ./run_pro6000.sh
#   MODEL=Qwen/Qwen3-32B-FP8 ./run_pro6000.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-Qwen/Qwen3-32B-FP8}"   # 공식 W8A8 FP8(block128, e4m3) — RNGD FP8 과 정밀도 매칭
GPU="${GPU:-0}"
PORT="${PORT:-8000}"
ISL="${ISL:-1024}"; OSL="${OSL:-256}"
BATCHES="${BATCHES:-1,8,16,32,64,256}"
WINDOW="${WINDOW:-30}"; WARMUP="${WARMUP:-8}"
MAXSEQS="${MAXSEQS:-256}"               # vLLM --max-num-seqs (동시 시퀀스 상한; furiosa --max-concurrency 대응)
GPUUTIL="${GPUUTIL:-0.90}"
MAXLEN="${MAXLEN:-32768}"
KVDTYPE="${KVDTYPE:-auto}"              # 'fp8' 로 KV 절약 가능하나 정확도 민감하면 auto(bf16) 권장
OUT="${OUT:-$HERE/results/pro6000.json}"
LOG="${LOG:-$HERE/results/pro6000_serve.log}"
mkdir -p "$HERE/results"

# venv 가 있으면 활성화
[ -f "$HERE/.venv/bin/activate" ] && source "$HERE/.venv/bin/activate"
command -v vllm >/dev/null || { echo "vllm 없음 — setup_gpu.md 로 먼저 설치하세요."; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi 없음 — NVIDIA 드라이버 필요."; exit 1; }

# 실패 시 진짜 원인(로그 상단)을 추려서 보여준다 — vLLM 은 래퍼 traceback 이 길어 tail 만으론 원인이 안 보임.
show_failure() {
  echo "── 근본 원인 후보(로그에서 추출) ──"
  grep -nE "no kernel image|not compatible|CUDA error|out of memory|OutOfMemory|RuntimeError|ImportError|ModuleNotFound|capability|sm_1[0-9]+|AssertionError|ValueError|flashinfer|NVML|Unsupported|Cannot|Failed to" "$LOG" 2>/dev/null | head -25
  echo "── 로그 끝 40줄 ──"; tail -40 "$LOG"
}

# preflight: 이 torch 빌드가 Blackwell(sm_120) 커널을 잡는지 확인 (구형 휠은 여기서 cap 미인식/NOCUDA)
CAP=$(CUDA_VISIBLE_DEVICES="$GPU" python3 -c "import torch;print('%d.%d'%torch.cuda.get_device_capability() if torch.cuda.is_available() else 'NOCUDA')" 2>/dev/null)
echo "  preflight: torch $(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null) / device capability = ${CAP:-?}"
if [ "$CAP" = "NOCUDA" ] || [ -z "$CAP" ]; then
  echo "!! torch 가 GPU 를 못 잡습니다(드라이버/CUDA 빌드 문제). setup_gpu.md 의 설치(2~3순위)로 재설치하세요."
  exit 1
fi
case "$CAP" in 12.*) : ;; *) echo "  ⚠ capability $CAP — PRO 6000 은 12.0 이어야 정상. 다른 GPU 이거나 빌드 문제일 수 있음.";; esac

echo "▶ vLLM serve: $MODEL on GPU $GPU  port=$PORT  (max-num-seqs=$MAXSEQS, kv=$KVDTYPE)"
EXTRA=()
[ "$KVDTYPE" != "auto" ] && EXTRA+=(--kv-cache-dtype "$KVDTYPE")
CUDA_VISIBLE_DEVICES="$GPU" nohup vllm serve "$MODEL" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GPUUTIL" \
  --max-num-seqs "$MAXSEQS" \
  --host 0.0.0.0 --port "$PORT" \
  "${EXTRA[@]}" \
  > "$LOG" 2>&1 &
SERVE_PID=$!
cleanup() { echo "▷ vllm 종료(pid $SERVE_PID)"; kill "$SERVE_PID" 2>/dev/null; wait "$SERVE_PID" 2>/dev/null; }
trap cleanup EXIT

# health 대기 (첫 실행은 모델 다운로드 ~33GB 로 오래 걸릴 수 있어 1800초)
echo -n "  서버 준비 대기(첫 실행은 다운로드로 길어질 수 있음)"
code=000
for i in $(seq 1 1800); do
  code=$(curl -s -m3 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
  [ "$code" = "200" ] && { echo " OK"; break; }
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then echo; echo "!! vllm 프로세스 종료됨"; show_failure; exit 1; fi
  [ $((i % 10)) -eq 0 ] && echo -n "."; sleep 1
done
[ "$code" = "200" ] || { echo; echo "!! 준비 실패(시간 초과)"; show_failure; exit 1; }

SERVED=$(curl -s -m5 "http://127.0.0.1:$PORT/v1/models" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
echo "  모델 id: $SERVED"

python3 "$HERE/loadgen.py" \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$SERVED" \
  --platform pro6000 --label "Qwen3-32B-FP8 (1x RTX PRO 6000, vLLM, W8A8)" \
  --isl "$ISL" --osl "$OSL" --batches "$BATCHES" \
  --window "$WINDOW" --warmup "$WARMUP" --endpoint completions \
  --tokenizer "$MODEL" \
  --power gpu --power-devices "$GPU" \
  --out "$OUT"

echo "✅ PRO 6000 결과: $OUT"
