#!/usr/bin/env python3
"""RNGD 4장(32 PE) tp32 아티팩트 빌드 도우미.

배경: furiosa-llm 공개 SDK(2026.2.0/2026.2.1)는 tp32(4칩 x 8PE) 빌드 시 모든 가중치의
'칩 사이' 배치를 FREE로 두는데, 컴파일러 게이트는 Broadcast 또는 Fixed를 요구해서
embedding 단계(stage_0)에서 바로 실패합니다("Graph input#0 must have Broadcast or Fixed
DramShapeGuide"). converter_tp32_broadcast.patch 를 적용하면 환경변수 FURIOSA_TP32_BCAST 로
칩 사이 배치를 Broadcast(칩마다 복제)로 강제할 수 있어 빌드가 끝까지 진행됩니다.

전제 조건(자세한 설명은 info/README_tp32_build.md):
  1) furiosa-llm SDK에 converter_tp32_broadcast.patch 가 적용되어 있어야 합니다.
  2) FURIOSA_TP32_BCAST=ALL 로 모든 가중치를 복제합니다(= 모델을 칩마다 통째로 올림).
     따라서 "한 칩에 들어가는 크기"의 모델만 빌드됩니다(예: 7B bf16).
  3) num_key_value_heads 가 칩 수(4)로 나눠떨어져야 합니다(어텐션을 4칩에 분배하므로).

사용 예:
  FURIOSA_TP32_BCAST=ALL python build_tp32.py Qwen/Qwen2.5-Coder-7B-Instruct ./out_tp32
"""
import os, sys, time
from furiosa_llm.artifact import ArtifactBuilder, ModelConfig, ParallelConfig
import furiosa_llm


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    model, out = sys.argv[1], sys.argv[2]
    max_len = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    if not os.environ.get("FURIOSA_TP32_BCAST"):
        print("[경고] FURIOSA_TP32_BCAST 가 설정되지 않았습니다. 'ALL' 로 설정하세요.", file=sys.stderr)
        print("       예: FURIOSA_TP32_BCAST=ALL python build_tp32.py <model> <out>", file=sys.stderr)
    print(f"[tp32] furiosa_llm={furiosa_llm.__version__}  model={model}  out={out}", flush=True)
    t0 = time.time()
    builder = ArtifactBuilder(
        model_id_or_path=model,
        name=os.path.basename(out.rstrip("/")),
        model_config=ModelConfig(max_model_len=max_len),
        parallel_config=ParallelConfig(tensor_parallel_size=32, pipeline_parallel_size=1),
    )
    builder.build(out)
    print(f"[tp32] BUILD SUCCEEDED in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
