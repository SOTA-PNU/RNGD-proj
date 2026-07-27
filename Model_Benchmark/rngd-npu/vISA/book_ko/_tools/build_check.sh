#!/usr/bin/env bash
# 사용법: build_check.sh <book디렉터리>
# mermaid 전처리기(미설치)만 빼고 mdbook 으로 빌드해 오류/경고를 요약한다.
MB=/home/jun/.claude/jobs/46bc5c7e/tmp/mdbook-inst/bin/mdbook
SRC="$1"; W=$(mktemp -d)
cp -r "$SRC"/. "$W"/
python3 - "$W/book.toml" <<'PY'
import re, sys
p = sys.argv[1]
t = open(p).read()
open(p, 'w').write(re.sub(r'\[preprocessor\.mermaid\][^\[]*', '', t))
PY
out=$("$MB" build "$W" 2>&1); rc=$?
err=$(printf '%s\n' "$out" | grep -c '^ERROR')
warn=$(printf '%s\n' "$out" | grep -c '^ WARN')
pages=$(find "$W/book" -name '*.html' 2>/dev/null | wc -l)
printf 'rc=%s  ERROR=%s  WARN=%s  html=%s\n' "$rc" "$err" "$warn" "$pages"
if [ "$err" -gt 0 ]; then
  echo "--- ERROR 유형별 ---"
  printf '%s\n' "$out" | grep '^ERROR' | sed 's/(.*//' | sort | uniq -c | sort -rn | head -10
fi
rm -rf "$W"
[ "$err" -eq 0 ]
