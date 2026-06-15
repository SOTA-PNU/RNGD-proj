#!/usr/bin/env python3
"""quantize.py — Llama-3.3-70B (또는 다른 HF 모델) FineGrainedFP8 양자화.

Furiosa 공식 문서의 워크플로 (https://developer.furiosa.ai/latest/en/furiosa_llm/model-preparation.html)
를 그대로 따름. GPU 필수.

사용:
    python quantize.py                                # 기본 meta-llama/Llama-3.3-70B-Instruct
    python quantize.py <hf-id-or-path>
    python quantize.py meta-llama/Llama-3.3-70B-Instruct -o ./fp8-out
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def _safe(s: str) -> str:
    return s.replace("/", "--").replace(":", "-")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL,
                    help=f"HF model id 또는 로컬 경로 (default: {DEFAULT_MODEL})")
    ap.add_argument("-o", "--out",
                    help=f"저장 경로 (default: {SCRIPT_DIR}/out/<model>-fp8)")
    ap.add_argument("--activation", choices=["dynamic", "static"], default="dynamic",
                    help="FP8 activation 양자화 방식 (default: dynamic)")
    ap.add_argument("--block", type=int, default=128,
                    help="weight block size NxN (default: 128)")
    ap.add_argument("--force", action="store_true", help="기존 출력 폴더 덮어쓰기")
    args = ap.parse_args()

    save_dir = (Path(args.out).expanduser().resolve()
                if args.out else SCRIPT_DIR / "out" / f"{_safe(args.model)}-fp8")

    if save_dir.exists() and (save_dir / "config.json").exists() and not args.force:
        print(f"⚠️  이미 존재: {save_dir}")
        if input("덮어쓰시겠습니까? [y/N] ").strip().lower() != "y":
            sys.exit(0)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"── 입력 ──")
    print(f"  모델       : {args.model}")
    print(f"  저장 위치  : {save_dir}")
    print(f"  activation : {args.activation}")
    print(f"  block      : ({args.block}, {args.block})")
    print()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config
    except ImportError as e:
        print(f"❌ {e} — './setup.sh' 를 먼저 실행하고 venv 활성화하세요.", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print("❌ CUDA GPU 미감지 — FineGrainedFP8 양자화는 GPU 필수.", file=sys.stderr)
        sys.exit(2)
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}      : {p.name} ({p.total_memory/1e9:.0f} GB)")
    print()

    quant_cfg = FineGrainedFP8Config(
        activation_scheme=args.activation,
        weight_block_size=(args.block, args.block),
    )

    print("── 모델 적재 + FP8 양자화 (transformers 가 자동 처리) ──")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",                  # 여러 GPU 자동 분산
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print(f"   → 적재·양자화: {time.time()-t0:.1f}s")

    print("\n── 토크나이저 ──")
    tok = AutoTokenizer.from_pretrained(args.model)

    print(f"\n── 저장 → {save_dir} ──")
    t0 = time.time()
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    print(f"   → 저장: {time.time()-t0:.1f}s")

    print("\n✅ 완료")
    sz = subprocess.run(["du", "-sh", str(save_dir)], capture_output=True, text=True)
    if sz.returncode == 0:
        print(f"   크기                  : {sz.stdout.split()[0]}")
    cfg_path = save_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        print(f"   model_type            : {cfg.get('model_type')}")
        if cfg.get("quantization_config"):
            print(f"   quantization_config   : {json.dumps(cfg['quantization_config'], ensure_ascii=False)}")

    print(f"\nNPU 서버로 옮긴 후 빌드:")
    print(f"  rsync -avh {save_dir}/ jun@<NPU호스트>:~/.cache/huggingface/hub/models--<name>-fp8/")
    print(f"  RAY_memory_monitor_refresh_ms=0 furiosa-llm build <path> <out> -tp 8 [-pp 2]")


if __name__ == "__main__":
    main()
