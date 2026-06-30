#!/usr/bin/env python3
"""Qwen3-VL (Qwen3VL / Qwen3VLMoe) 를 furiosa-llm 으로 빌드한다.

CLI `furiosa-llm build` 가 막는 두 지점을 우회한다(2026-06-30 실측):
  1) validate_hf_config 가 top-level 에서 max_position_embeddings/num_hidden_layers/
     hidden_size/intermediate_size 를 찾는데 VL config 는 이들을 text_config 안에 둠
     → config.json 을 패치(해당 4필드를 text_config 값으로 top-level 복사)한 로컬 dir 사용.
  2) task auto 해석이 Qwen3VL*ForConditionalGeneration 을 CAUSAL_LM 매핑에서 못 찾아 None
     → ModelConfig(task="generate") 로 명시(=GENERATE).

무프리셋(qwen3_vl 버킷 preset 없음)은 -pb/-db 수동 버킷으로 우회.

사용:
  python build_vl.py <snapshot_dir> <out_dir> [--name N] [--tp 8] [--pp 1] \
                     [--prefill 1,8192] [--decode 1,8192] [--max-model-len 8192]
serve 시 레이어 분할은 빌드가 아니라 serve 때 `-pp 4` 로(아티팩트는 tp8 로 빌드).
"""
import argparse, json, os, sys


def patch_config(snap: str, fix: str) -> str:
    """text_config 의 필수 4필드를 top-level 로 끌어올린 로컬 모델 dir 생성(safetensors 는 심링크)."""
    os.makedirs(fix, exist_ok=True)
    for f in os.listdir(snap):
        if f == "config.json":
            continue
        dst = os.path.join(fix, f)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.realpath(os.path.join(snap, f)), dst)
    c = json.load(open(os.path.join(snap, "config.json")))
    tc = c.get("text_config", {})
    need = ["max_position_embeddings", "num_hidden_layers", "hidden_size", "intermediate_size"]
    lifted = {}
    for k in need:
        if k not in c and k in tc:
            c[k] = tc[k]
            lifted[k] = tc[k]
    json.dump(c, open(os.path.join(fix, "config.json"), "w"), indent=2)
    print(f"[patch] {fix}\n[patch] lifted: {lifted}", flush=True)
    return fix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--name", default="qwen3-vl")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--prefill", default="1,8192")   # batch,ctx
    ap.add_argument("--decode", default="1,8192")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    fix = a.out_dir.rstrip("/") + "-cfgfix"   # artifacts/ 안에 깔끔히
    patch_config(a.snapshot_dir, fix)

    from furiosa_llm.artifact import (
        ArtifactBuilder, ModelConfig, ParallelConfig, BucketConfig,
        CompilerConfig, ArtifactConfig,
    )

    mc = ModelConfig(trust_remote_code=True, task="generate", max_model_len=a.max_model_len)
    pc = ParallelConfig(tensor_parallel_size=a.tp, pipeline_parallel_size=a.pp)
    pb = [tuple(map(int, a.prefill.split(",")))]
    db = [tuple(map(int, a.decode.split(",")))]
    bc = BucketConfig(prefill_buckets=pb, decode_buckets=db)
    cc = CompilerConfig()
    ac = ArtifactConfig(bundle_binaries=True)

    print(f"[build] model={fix} tp={a.tp} pp={a.pp} pb={pb} db={db} maxlen={a.max_model_len}", flush=True)
    builder = ArtifactBuilder(
        fix, a.name,
        model_config=mc, parallel_config=pc, bucket_config=bc,
        compiler_config=cc, artifact_config=ac,
    )
    builder.build(a.out_dir, num_pipeline_builder_workers=a.workers, num_compile_workers=a.workers)
    print("[build] DONE ->", a.out_dir, flush=True)


if __name__ == "__main__":
    sys.exit(main())
