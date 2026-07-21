#!/usr/bin/env bash
# furiosa-llm 에 qwen3_coder tool 파서를 설치(등록)한다.
# furiosa-llm 재설치/업그레이드 후 다시 실행하면 복구된다(멱등).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TP=$(python3 -c "import furiosa_llm.server.tool_parsers as m, os; print(os.path.dirname(m.__file__))")
echo "[..] tool_parsers 폴더: $TP"

cp "$HERE/qwen3_coder_tool_parser.py" "$TP/qwen3_coder_tool_parser.py"
echo "[ok] 파서 파일 복사"

# __init__.py 에 import + __all__ 멱등 추가
python3 - "$TP/__init__.py" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
imp = "from .qwen3_coder_tool_parser import Qwen3CoderToolParser"
if imp not in src:
    # 마지막 from .*_tool_parser import 줄 뒤에 삽입
    lines = src.splitlines()
    last = max(i for i, l in enumerate(lines) if l.startswith("from .") and "import" in l)
    lines.insert(last + 1, imp)
    src = "\n".join(lines) + ("\n" if not src.endswith("\n") else "")
    # __all__ 에도 추가
    src = src.replace('"OpenAIToolParser",', '"OpenAIToolParser",\n    "Qwen3CoderToolParser",', 1)
    open(p, "w").write(src)
    print("[ok] __init__.py 에 등록 추가")
else:
    print("[ok] 이미 등록돼 있음")
PY

# 검증
python3 -c "
import furiosa_llm.server.tool_parsers as m
ks = list(m.ToolParserManager.tool_parsers.keys())
assert 'qwen3_coder' in ks, f'등록 실패: {ks}'
print('[ok] 등록된 tool 파서:', ks)
"
