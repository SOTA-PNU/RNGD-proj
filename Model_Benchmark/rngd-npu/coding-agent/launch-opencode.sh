#!/usr/bin/env bash
# OpenCode 를 Furiosa NPU 라우터 백엔드로 실행
# ---------------------------------------------------------------------------
# 라우터(:8400)가 안 떠 있으면 먼저 띄우고, 라우터용 opencode.json 이 있는
# 폴더에서 OpenCode TUI 를 연다. 모델 선택창(switch model)에 rngd-npu/artifacts
# 의 모든 모델이 뜨고, 고르면 그 모델이 올바른 옵션으로 자동 서빙된다.
#
# (주의) 벤더 furiosa-opencode.py / opencode.sh 는 단일 모델 opencode.json 을
#        덮어쓰므로 라우터 구성에선 쓰지 않는다. 이 스크립트를 쓰자.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE=~/RNGD-proj/Model_Benchmark/rngd-npu/coding-agent
export PATH="$HOME/.opencode/bin:$PATH"

if ! curl -fsS --max-time 2 http://localhost:8400/v1/models >/dev/null 2>&1; then
  echo "[..] 라우터(:8400)가 없어 기동합니다"
  bash "$HERE/serve-router.sh" start
  for i in $(seq 1 20); do
    curl -fsS --max-time 2 http://localhost:8400/v1/models >/dev/null 2>&1 && break
    sleep 1
  done
fi

cd "$HERE/opencode"      # 이 폴더의 opencode.json(전 모델 → 라우터)을 OpenCode 가 읽는다
exec opencode "$@"       # TUI. 첫 모델 사용 시 라우터가 콜드스타트(큰 모델은 수십 초~분)
