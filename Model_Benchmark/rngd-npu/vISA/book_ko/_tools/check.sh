#!/usr/bin/env bash
# book_ko 최종 게이트. 각 검사는 이 작업에서 실제로 문제가 났던 지점을 그대로 인코딩한다.
B="$(cd "$(dirname "$0")/.." && pwd)"     # book_ko/
T="$B/_tools"
S="$B/src"
FAIL=0
ok(){ printf '  ✅ %s\n' "$1"; }
bad(){ printf '  ❌ %s\n' "$1"; FAIL=$((FAIL+1)); }
hr(){ printf '%s\n' "============================================================"; }

hr; echo "1) 파일 수·번역 반영"
n=$(find "$S" -name '*.md' | wc -l)
[ "$n" = 46 ] && ok "md 46개" || bad "md $n 개 (46 이어야 함)"
k=$(grep -rlP '[가-힣]' "$S" --include='*.md' | wc -l)
[ "$k" = 46 ] && ok "46개 전부 한국어 반영" || bad "$k/46 만 한국어"

hr; echo "2) 구조 보존 (영문 기준 지문과 대조)"
if python3 "$T/struct_check.py" compare "$S" "$T/baseline.json" > /tmp/_sc.$$ 2>&1; then
  ok "$(tail -1 /tmp/_sc.$$ | sed 's/^ *//')"
else
  bad "구조 위반"; grep '❌' /tmp/_sc.$$ | head -10
fi
rm -f /tmp/_sc.$$

hr; echo "3) {{#include}} 미해소 잔존 (0 이어야 함)"
c=$(grep -rc '{{#include' "$S" --include='*.md' 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
[ "$c" = 0 ] && ok "include 잔존 없음 (전부 실소스로 채워짐)" || bad "$c 곳 미해소"

hr; echo "4) 앵커 — 헤딩 슬러그 + 명시적 <a id> 양쪽 모두 인정"
a=$(python3 "$T/anchors.py" audit "$S" | head -1)
echo "  $a"
got=$(echo "$a" | grep -oE '[0-9]+종' | tr -d '종')
[ "$got" = 84 ] && ok "해소 84종 (영문 원본과 동일)" || bad "해소 $got 종 (84 여야 함)"
br=$(python3 "$T/anchors.py" audit "$S" | grep -c '앵커 없음')
[ "$br" = 15 ] && ok "원본 책 유래 미해소 15건 (영문 원본과 동일)" || bad "미해소 $br 건 (15 여야 함)"
inj=$(grep -rho '<a id=' "$S" --include='*.md' | wc -l)
[ "$inj" = 85 ] && ok "<a id> 85개 (원본 5 + 주입 80)" || bad "<a id> $inj 개 (85 여야 함)"

hr; echo "5) 미번역 영문 산문 (의도된 표기법 4줄만 허용)"
r=$(python3 "$T/residual_en.py" "$S" | tail -1)
cnt=$(echo "$r" | grep -oE '[0-9]+줄' | tr -d '줄')
if [ -z "$cnt" ]; then ok "잔존 없음"
elif [ "$cnt" -le 4 ]; then ok "$cnt 줄 (표기법 보존분 이하)"
else bad "$cnt 줄 — 늘어남"; fi

hr; echo "6) mdbook 실제 빌드"
out=$("$T/build_check.sh" "$B" 2>&1 | head -1)
echo "  $out"
echo "$out" | grep -q 'ERROR=0  WARN=0  html=47' && ok "ERROR 0 / WARN 0 / 47쪽" || bad "빌드 결과 이상"

hr; echo "7) 이미지·자산"
img=$(find "$S" \( -name '*.png' -o -name '*.webp' \) | wc -l)
[ "$img" = 18 ] && ok "이미지 18개" || bad "이미지 $img 개"
[ -f "$B/book.toml" ] && ok "book.toml 존재" || bad "book.toml 없음"

hr; echo "8) 규모"
printf "  문서 %s개 / %s행 / 한글 %s자\n" \
  "$(find "$S" -name '*.md' | wc -l)" \
  "$(cat $(find "$S" -name '*.md') | wc -l)" \
  "$(cat $(find "$S" -name '*.md') | grep -oP '[가-힣]' | wc -l)"

hr
[ "$FAIL" = 0 ] && echo "✅ 전체 통과" || echo "❌ 실패 $FAIL 건"
exit $FAIL
