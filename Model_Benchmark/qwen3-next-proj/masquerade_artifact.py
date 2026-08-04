#!/usr/bin/env python3
"""아티팩트 메타데이터 위장(masquerade) 도구 — qwen3-next-proj

furiosa-llm 2026.2.0 의 serve 런타임은 `artifact.json` 의
`model_metadata.model_type` 문자열 하나만 화이트리스트 검사합니다
(허용: llama, exaone4, qwen2, qwen3, qwen3_moe, gpt_oss, embed, score —
`hf_compat_next_gen.rs:367`). model_type 이 목록에 없거나(예: qwen3_next),
런타임에 그 model_type 용 커널이 없으면(예: qwen3_moe — 양자화와 무관) 부팅 시
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
    (--in-place 여도 artifact.json.orig-<원래타입> 백업을 자동으로 남긴다.)
  - 검증된 사례: qwen3_moe(FP8) → qwen3 위장으로 Qwen3-Coder-30B-A3B-Instruct-FP8
    serve 부활(2026-06-10, 62.7 tok/s 단일 카드). dense qwen3 스케줄러 preset 으로
    MoE 그래프를 구동 — 짧은 생성에서 정상 확인. 장문맥·고동시성은 추가 검증 권장.

**2026.3.0 에서도 게이트는 그대로다** (2026-08-04 실측 — legacy_moe_build/README §6 의
"미실측" 항목 해소). 직접 빌드한 tp8 v2 아티팩트 `coder-tp8` 을 그대로 serve 하면:

    pyo3_runtime.PanicException: Unsupported model metadata: ModelMetadata {
        model_type: Some(Qwen3Moe), ...
        quantization_config: Some(QuantizationConfig { weight: FP8, ... }) }

즉 **qwen3_moe 는 여전히 거부**되고 위장이 필요하다. **양자화와 무관하다** — fp8(coder-tp8)과
bf16(coder-bf16-tp8) 둘 다 같은 PanicException 으로 막히는 것을 실측했다(처음엔 fp8 만
문제라고 봤다가 정정). qwen3/llama/exaone4 계열은 애초에 대상이 아니다.
위장 대상: `coder-tp8` `coder-bf16-tp8` `a3b-tp8` `a3b-inst-2507-tp8` `a3b-think-2507-tp8`.

(2026-08-04 수정) `--copy` 는 하드링크 복사라 예전에는 사본을 위장하면 **원본 artifact.json
까지 같이 바뀌었다**(같은 inode 를 제자리 truncate). 지금은 임시파일 → os.replace 로 써서
링크를 끊으므로 원본이 보존된다.

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
    # 원본 보존용 백업(멱등 — 이미 있으면 덮어쓰지 않는다. 되돌리기: 이 파일을 artifact.json 으로 복사).
    bak = p + f".orig-{orig}"
    if not os.path.exists(bak):
        shutil.copyfile(p, bak)
        print(f"[backup] {bak}")
    # ⚠️ 반드시 임시파일 → os.replace 로 쓴다.
    #    --copy 는 하드링크 복사라 dst/artifact.json 이 src 와 같은 inode 다. 여기서 open(p,"w") 로
    #    제자리 truncate 하면 **원본까지 같이 바뀐다**(2026-08-04 발견). os.replace 는 디렉터리
    #    엔트리를 갈아끼우므로 링크가 끊기고 원본이 그대로 남는다. 제자리 쓰기도 원자적이 된다.
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, p)
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
