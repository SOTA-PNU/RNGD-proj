#!/usr/bin/env python3
"""
sdi code 설정 생성기 — 서버(/v1/models)에서 모델 목록을 받아 opencode.json 을 만든다.
표준 라이브러리만 사용. API 키는 config 의 apiKey 에 리터럴로 넣고 파일은 0600 으로 생성한다
(env 파일 source 로 인한 셸 코드주입 위험 제거). 클라이언트는 OPENCODE_CONFIG 로 이 파일을 가리킨다.

  SDI_SERVER=http://10.125.19.138:8400 SDI_API_KEY=<key> python3 gen-config.py <출력경로>
"""
import json
import os
import sys
import urllib.request

SERVER = (os.environ.get("SDI_SERVER") or "http://10.125.19.138:8400").rstrip("/")
KEY = os.environ.get("SDI_API_KEY", "")
OUT = sys.argv[1] if len(sys.argv) > 1 else "opencode.json"
BASE = SERVER + "/v1"

req = urllib.request.Request(BASE + "/models")
if KEY:
    req.add_header("Authorization", "Bearer " + KEY)
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        ids = [m["id"] for m in json.load(r).get("data", [])]
except Exception as e:
    print(f"[fail] 서버 도달 실패 {BASE}/models : {e}", file=sys.stderr)
    sys.exit(1)
if not ids:
    print(f"[fail] 서버에 모델이 없습니다 ({BASE}/models)", file=sys.stderr)
    sys.exit(1)


# 컨텍스트 한도는 모델명 휴리스틱 추정(서버 /v1/models 가 ctx 를 안 내려줌). 대소문자 무시.
# ⚠️ 서버 REGISTRY 의 ctx 와 수동 동기 필요 — 명명 규칙 밖 모델 추가 시 어긋날 수 있음.
def ctx(mid):
    m = mid.lower()
    return 16384 if "16k" in m else (65536 if "a3b" in m else 32768)


# 브랜드/명령 이름은 환경변수로 교체 가능(추후 'sdi' 외 이름으로 리브랜딩 시)
PROVIDER = os.environ.get("SDI_PROVIDER", "sdi")
BRAND = os.environ.get("SDI_BRAND", "SDI Code (Furiosa NPU)")
models = {mid: {"name": mid, "limit": {"context": ctx(mid), "output": 8192}} for mid in ids}
default = "Qwen3-32B-FP8" if "Qwen3-32B-FP8" in ids else ids[0]
# 키는 선택: 서버가 인증 OFF 면 KEY 가 비고 apiKey 를 넣지 않는다(무인증 접속).
opts = {"baseURL": BASE}
if KEY:
    opts["apiKey"] = KEY
cfg = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
        PROVIDER: {
            "npm": "@ai-sdk/openai-compatible",
            "name": BRAND,
            "options": opts,
            "models": models,
        }
    },
    "model": f"{PROVIDER}/{default}",
    "small_model": f"{PROVIDER}/{default}",
}
os.makedirs(os.path.dirname(os.path.abspath(OUT)) or ".", exist_ok=True)
# 키가 들어가므로 소유자 전용(0600)으로 생성
fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(f"[ok] {OUT} 작성(0600): 모델 {len(ids)}개 (서버 {SERVER}, 기본 {default})")
