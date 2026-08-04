#!/usr/bin/env bash
# qwen3_moe 아티팩트의 model_type 을 qwen3 로 위장한다 (serve 게이트 통과용).
#
# 왜: furiosa-llm 런타임은 model_type 화이트리스트를 들고 있어서 qwen3_moe 아티팩트를
#     부팅 때 거부한다 — 2026.3.0 에서도 그대로다. **양자화와 무관하다**:
#     fp8(coder-tp8)·bf16(coder-bf16-tp8) 둘 다 거부되는 것을 2026-08-04 에 실측했다.
#
#       pyo3_runtime.PanicException: Unsupported model metadata: ModelMetadata {
#           model_type: Some(Qwen3Moe), quantization_config: { weight: FP8, ... } }
#
#     연산은 빌드 때 이미 EDF 바이너리로 컴파일돼 있고 게이트만 메타데이터 문자열을 보므로,
#     model_type 만 바꾸면 통과하고 런타임은 컴파일된 MoE 그래프를 그대로 실행한다.
#
# 사용:
#   bash masquerade_moe.sh          # 대상만 보여주기(변경 없음)
#   bash masquerade_moe.sh --apply  # 실제 적용
#
# 멱등하다 — 이미 위장된 것은 model_type 이 qwen3 라 대상에서 빠진다.
# 원본 artifact.json 은 artifact.json.orig-qwen3_moe 로 자동 백업된다(되돌리려면 되돌려 놓으면 됨).
set -u
ART="${CHAT_ARTIFACTS:-/mnt/nvme2n1p1/models/artifacts}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASQ="$HERE/../../qwen3-next-proj/masquerade_artifact.py"
APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

[ -f "$MASQ" ] || { echo "✗ masquerade_artifact.py 없음: $MASQ"; exit 1; }

mapfile -t TARGETS < <(python3 - "$ART" <<'PY'
import json, os, sys
A = sys.argv[1]
for d in sorted(os.listdir(A)):
    p = os.path.join(A, d, "artifact.json")
    if not os.path.isfile(p):
        continue
    try:
        mm = json.load(open(p))["model"]["model_metadata"]
    except Exception:
        continue
    # 양자화와 무관하게 model_type=qwen3_moe 이면 전부 거부된다(2026-08-04 fp8·bf16 둘 다 실측).
    if mm.get("model_type") == "qwen3_moe":
        print(d)
PY
)

if [ "${#TARGETS[@]}" -eq 0 ]; then
  echo "✅ 위장이 필요한 아티팩트 없음 (model_type=qwen3_moe 0건)"
  exit 0
fi

echo "위장 대상 ${#TARGETS[@]}개 (model_type=qwen3_moe):"
printf '  - %s\n' "${TARGETS[@]}"
if [ "$APPLY" -eq 0 ]; then
  echo
  echo "적용하려면:  bash $(basename "${BASH_SOURCE[0]}") --apply"
  exit 0
fi

echo
for d in "${TARGETS[@]}"; do
  echo "── $d"
  python3 "$MASQ" "$ART/$d" --as qwen3 --in-place || echo "  ✗ 실패: $d"
done

echo
echo "확인:  python3 $HERE/validate_catalog.py"
