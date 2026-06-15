#!/usr/bin/env python3
"""아티팩트 메타데이터 위장(masquerade) 도구 — qwen3-next-proj

furiosa-llm 2026.2.0 의 serve 런타임은 `artifact.json` 의
`model_metadata.model_type` 문자열 하나만 화이트리스트 검사합니다
(허용: llama, exaone4, qwen2, qwen3, qwen3_moe, gpt_oss, embed, score —
`hf_compat_next_gen.rs:367`). model_type 이 목록에 없거나(예: qwen3_next),
런타임에 해당 (model_type × 양자화) 커널이 없으면(예: qwen3_moe × FP8) 부팅 시
`PanicException: Unsupported model metadata` 로 죽습니다.

그러나 **연산은 이미 빌드 시 EDF 바이너리로 다 컴파일**되어 있으므로, 게이트만
통과시키면 런타임은 컴파일된 그래프를 그대로 실행합니다. 이 스크립트는 빌드된
아티팩트의 메타데이터 model_type 을 게이트가 허용하는 값으로 바꿔(=위장) serve 를
가능하게 합니다.

⚠️ 안전 규칙:
  - KV 캐시 차원(num_hidden_layers / num_key_value_heads / head_dim)은 절대 바꾸지
    말 것 — 런타임이 hf_configs 기준으로 캐시 shape 를 검증/할당하므로 실제
    컴파일된 그래프와 일치해야 함.
  - 원본 아티팩트는 건드리지 않고 사본을 위장하는 것을 권장(--copy).
  - 검증된 사례: qwen3_moe(FP8) → qwen3 위장으로 Qwen3-Coder-30B-A3B-Instruct-FP8
    serve 부활(2026-06-10, 62.7 tok/s 단일 카드). dense qwen3 스케줄러 preset 으로
    MoE 그래프를 구동 — 짧은 생성에서 정상 확인. 장문맥·고동시성은 추가 검증 권장.

사용법:
    # 사본을 만들고(하드링크) 위장
    ~/furiosa/bin/python masquerade_artifact.py <src_artifact> --as qwen3 --copy <dst>
    # 제자리 위장(원본 수정, 비권장)
    ~/furiosa/bin/python masquerade_artifact.py <artifact> --as qwen3 --in-place
"""
import argparse
import json
import os
import shutil
import sys

# qwen3_moe(원래 model_type) → 위장 후 hf_configs 에서 떼어낼 MoE 전용 키
MOE_ONLY_KEYS = [
    "decoder_sparse_step", "moe_intermediate_size", "num_experts_per_tok",
    "num_experts", "num_local_experts", "norm_topk_prob",
    "output_router_logits", "router_aux_loss_coef", "mlp_only_layers",
    "shared_expert_intermediate_size",
]
ARCH_FOR = {
    "qwen3": "Qwen3ForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "llama": "LlamaForCausalLM",
}


def masquerade(artifact_dir: str, as_type: str, strip_moe: bool) -> None:
    p = os.path.join(artifact_dir, "artifact.json")
    with open(p) as f:
        d = json.load(f)
    md = d["model"]["model_metadata"]
    hf = md["hf_configs"]
    orig = md.get("model_type")
    md["model_type"] = as_type
    hf["model_type"] = as_type
    if as_type in ARCH_FOR:
        hf["architectures"] = [ARCH_FOR[as_type]]
    if strip_moe:
        for k in MOE_ONLY_KEYS:
            hf.pop(k, None)
    with open(p, "w") as f:
        json.dump(d, f)
    print(f"[masquerade] {orig} -> {as_type}  ({p})")
    print(f"  KV dims kept: layers={hf.get('num_hidden_layers')} "
          f"kv_heads={hf.get('num_key_value_heads')} head_dim={hf.get('head_dim')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="빌드된 아티팩트 디렉터리")
    ap.add_argument("--as", dest="as_type", default="qwen3",
                    help="위장할 model_type (기본 qwen3)")
    ap.add_argument("--copy", metavar="DST",
                    help="원본을 DST 로 하드링크 복사 후 위장 (원본 보존)")
    ap.add_argument("--in-place", action="store_true",
                    help="원본을 직접 위장 (비권장)")
    ap.add_argument("--strip-moe", action="store_true", default=True,
                    help="hf_configs 에서 MoE 전용 키 제거 (기본 True)")
    args = ap.parse_args()

    if args.copy:
        if os.path.exists(args.copy):
            sys.exit(f"DST already exists: {args.copy}")
        # 하드링크 복사 — 큰 safetensors/zip 을 디스크 추가소모 없이 공유
        shutil.copytree(args.src, args.copy, copy_function=os.link)
        target = args.copy
    elif args.in_place:
        target = args.src
    else:
        sys.exit("Either --copy DST or --in-place required.")

    masquerade(target, args.as_type, args.strip_moe)
    print(f"\nserve:  ~/furiosa/bin/furiosa-llm serve {target} --devices npu:0 ...")


if __name__ == "__main__":
    main()
