#!/usr/bin/env python3
"""미니 합성 모델 생성기 — qwen3-next-proj

실제 160GB 모델을 받기 전에, 장난감 크기(수백 MB)의 랜덤 가중치 모델로
furiosa-llm build/serve 파이프라인을 빠르게 검증하기 위한 도구입니다.

사용법:
    ~/furiosa/bin/python make_mini_model.py qwen3        # dense 베이스라인
    ~/furiosa/bin/python make_mini_model.py qwen3_moe    # MoE 경로 검증 (BF16)
    ~/furiosa/bin/python make_mini_model.py qwen3_next   # 본 프로젝트 대상

출력: ./mini_models/mini-<type>/  (config.json + safetensors + Qwen 토크나이저)
토크나이저는 HF 캐시의 Qwen/Qwen2.5-0.5B-Instruct 것을 재사용합니다 (vocab 151936 동일 계열).
"""
import sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

OUT_BASE = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj/mini_models"
TOKENIZER_SRC = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # HF 캐시에 이미 있음

# 공통: vocab은 실제 Qwen 토크나이저와 맞춤(151936). 나머지는 컴파일 검증에 충분한 최소 크기.
COMMON = dict(
    vocab_size=151936,
    hidden_size=512,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=4096,
    rope_theta=5000000.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    torch_dtype="bfloat16",
)

CONFIGS = {
    # dense 베이스라인 — 파이프라인 자체 검증용
    "qwen3": dict(
        COMMON,
        model_type="qwen3",
        architectures=["Qwen3ForCausalLM"],
        num_hidden_layers=2,
        intermediate_size=1536,
        head_dim=64,
    ),
    # MoE 경로 검증 — BF16 MoE serve 가능성 확인용 (실제 30B-A3B와 같은 구조, 크기만 축소)
    "qwen3_moe": dict(
        COMMON,
        model_type="qwen3_moe",
        architectures=["Qwen3MoeForCausalLM"],
        num_hidden_layers=2,
        intermediate_size=1536,
        head_dim=64,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        mlp_only_layers=[],
        output_router_logits=False,
        router_aux_loss_coef=0.001,
    ),
    # 본 프로젝트 대상 — 실제 Qwen3-Coder-Next config 의 모든 구조적 특성을 보존, 크기만 축소
    # (full_attention_interval=4 → 레이어 4개면 linear 3 + full 1)
    "qwen3_next": dict(
        COMMON,
        model_type="qwen3_next",
        architectures=["Qwen3NextForCausalLM"],
        num_hidden_layers=4,
        intermediate_size=1536,
        head_dim=128,           # 실모델은 256 (h 대비 큰 head_dim 특성 유지)
        partial_rotary_factor=0.25,
        full_attention_interval=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_value_head_dim=64,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        mlp_only_layers=[],
        output_router_logits=False,
        router_aux_loss_coef=0.001,
    ),
}


def main(kind: str):
    cfg_dict = CONFIGS[kind]
    out_dir = f"{OUT_BASE}/mini-{kind.replace('_', '-')}"
    config = AutoConfig.for_model(**cfg_dict)
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{kind}] params={n_params/1e6:.1f}M -> {out_dir}")
    model.save_pretrained(out_dir, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_SRC)
    tok.save_pretrained(out_dir)
    print(f"[{kind}] saved.")


if __name__ == "__main__":
    main(sys.argv[1])
