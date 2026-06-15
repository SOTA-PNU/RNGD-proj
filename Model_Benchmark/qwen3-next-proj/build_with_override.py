#!/usr/bin/env python3
"""qwen3_next 미니 빌드 + 컴파일러 config 오버라이드 주입 테스트.

목적(공부): NPU 컴파일러의 `allow_unlowered_operators` / `allow_external_operators`
플래그를 켜면, 커널화 안 되는 standalone elementwise op(sigmoid/mul 등)을 통과시킬 수
있는지 실측. compiler_config_overrides 는 compiler_config.py:176-177 에서 config 에 병합됨.

사용: ~/furiosa/bin/python build_with_override.py <model_path> <out> '<json_overrides>'
"""
import sys, json
from furiosa_llm.artifact import (
    ArtifactBuilder, ArtifactConfig, BucketConfig, CompilerConfig, ModelConfig, ParallelConfig,
)

model_path = sys.argv[1]
out = sys.argv[2]
overrides = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
print(f"[override-build] overrides = {overrides}")

builder = ArtifactBuilder(
    model_path,
    "mini-qwen3-next-ovr",
    model_config=ModelConfig(trust_remote_code=False, max_model_len=2048),
    parallel_config=ParallelConfig(tensor_parallel_size=8, pipeline_parallel_size=1),
    bucket_config=BucketConfig(prefill_buckets=[], decode_buckets=[]),
    compiler_config=CompilerConfig(compiler_config_overrides=overrides),
    artifact_config=ArtifactConfig(),
)
builder.build(out, num_pipeline_builder_workers=1, num_compile_workers=1)
print("BUILD OK")
