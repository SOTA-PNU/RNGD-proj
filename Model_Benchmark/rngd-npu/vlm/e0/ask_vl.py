#!/usr/bin/env python3
"""E0 forward 테스트: 서빙 중인 VL 엔드포인트(OpenAI 호환)에 이미지+질문을 보내 답을 확인.

사용: python ask_vl.py [--port 8010] [--model qwen3-vl-32b-inst] [--image e0/test_42.png] [--q "..."]
정답 체크: test_42.png 는 빨간 숫자 42 + 파란 글자 RNGD-VLM. 답에 "42"가 있으면 비전 타워가 글자를 봤다는 뜻.
"""
import argparse, base64, json, sys, urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="qwen3-vl-32b-inst")
    ap.add_argument("--image", default="/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vlm/e0/test_42.png")
    ap.add_argument("--q", default="What red number is shown in the image? Answer with just the number.")
    a = ap.parse_args()

    b64 = base64.b64encode(open(a.image, "rb").read()).decode()
    payload = {
        "model": a.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": a.q},
        ]}],
        "max_tokens": 64, "temperature": 0.0,
    }
    url = f"http://{a.host}:{a.port}/v1/chat/completions"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.load(r)
        ans = out["choices"][0]["message"]["content"]
        print("ANSWER:", ans)
        print("E0 PASS" if "42" in ans else "E0 응답은 받았으나 정답(42) 불일치 — 비전 경로 점검 필요")
    except Exception as e:
        print("REQUEST FAILED:", type(e).__name__, str(e)[:300])
        sys.exit(1)


if __name__ == "__main__":
    main()
