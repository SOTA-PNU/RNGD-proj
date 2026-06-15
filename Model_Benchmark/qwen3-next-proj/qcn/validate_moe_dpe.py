"""Validate the DPE-wired npu_linear on the projection-heavy MoE layer-0.

Runs the UNBATCHED NPU MoE path (moe_forward_npu_unbatched) -- which routes EVERY
expert's gate/up/down projection AND the shared-expert SwiGLU + shared_gate through
the shared npu_linear helper -- with QCN_DPE=1 (dn_linear_dpe.yaml, EinsumByDpe).
Compares against the real HF Qwen3NextSparseMoeBlock at bf16 DPE tolerance (atol 1e-2).

Run: PYTHONPATH=.../qwen3-next-proj RNGD_DEV=rngd:2 QCN_DPE=1 \
     /home/jun/furiosa/bin/python -m qcn.validate_moe_dpe
"""
import torch
import torch.nn.functional as F
from qcn.loader import QCNWeights
from qcn import moe as M


def main():
    torch.manual_seed(0)
    LAYER = 0
    W = QCNWeights()
    H = W.config["hidden_size"]
    top_k = W.config["num_experts_per_tok"]
    norm = W.config["norm_topk_prob"]
    print("=" * 72)
    print(f"MoE layer {LAYER} DPE validation  (QCN_DPE active = {M.DPE}, yaml={M.LINEAR_YAML})")
    print(f"experts={W.config['num_experts']} top_k={top_k} "
          f"moe_inter={W.config['moe_intermediate_size']} hidden={H}")

    # one decode token -> exactly top_k=10 active experts (tractable on NPU)
    T = 1
    hidden = (torch.randn(8, H) * 0.5)[:T]
    gate_w = W.get(f"model.layers.{LAYER}.mlp.gate.weight", torch.float32)
    tv, ti = M.host_router(hidden, gate_w, top_k, norm)
    activated = sorted(torch.unique(ti).tolist())
    print(f"active experts = {len(activated)}  (each runs gate/up/down proj via DPE npu_linear)")
    print("-" * 72)

    # ---- exact host oracle (== unbatched VE path to ~1e-7) ----
    host_ref = M.moe_forward_host_ref(hidden, W, ti, tv, LAYER)

    # ---- real HF Qwen3NextSparseMoeBlock ----
    blk, cfg = M.build_hf_reference(W, activated, LAYER)
    with torch.no_grad():
        hf_res = blk(hidden.unsqueeze(0))      # [1,T,H] -> [1,T,H] (may be tuple)
    hf_out = (hf_res[0] if isinstance(hf_res, (tuple, list)) else hf_res).squeeze(0).float()

    # ---- UNBATCHED NPU path with DPE npu_linear on EVERY projection ----
    M.NPU_STAGES.clear(); M.CALLS["n"] = 0; M.DISPATCH["n"] = 0
    M.FLOPS["npu"] = 0; M.FLOPS["host"] = 0
    npu_out, n_act = M.moe_forward_npu_unbatched(hidden, W, ti, tv, LAYER)
    dfg = M.CALLS["n"]
    bad = [(n, d) for n, d in M.NPU_STAGES if d != 0]

    err_hf  = (npu_out - hf_out).abs().max().item()
    rel_hf  = err_hf / (hf_out.abs().max().item() + 1e-9)
    err_ora = (npu_out - host_ref).abs().max().item()
    rel_ora = err_ora / (host_ref.abs().max().item() + 1e-9)
    ok_hf   = torch.allclose(npu_out, hf_out, atol=1e-2)
    ok_ora  = torch.allclose(npu_out, host_ref, atol=1e-2)

    print(f"NPU dispatches (unbatched DPE)     : {M.DISPATCH['n']}")
    print(f"stages that fell back to CPU       : {bad if bad else 'NONE'}")
    print(f"_dfg_inner                         : {dfg}  (0 == every matmul on NPU/DPE)")
    print("-" * 72)
    print(f"maxerr  DPE-NPU vs HF block        : {err_hf:.3e}   (rel {rel_hf:.3e})")
    print(f"maxerr  DPE-NPU vs host oracle      : {err_ora:.3e}   (rel {rel_ora:.3e})")
    print(f"allclose(atol=1e-2) vs HF          : {ok_hf}")
    print(f"allclose(atol=1e-2) vs oracle      : {ok_ora}")
    print("-" * 72)
    overall = bool(ok_hf and ok_ora and dfg == 0)
    print(f"OVERALL_PASS (atol1e-2 + dfg==0)   : {overall}")
    print("=" * 72)
    return overall


if __name__ == "__main__":
    main()
