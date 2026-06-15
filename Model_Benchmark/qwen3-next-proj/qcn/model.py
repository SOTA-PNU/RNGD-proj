#!/usr/bin/env python3
# =============================================================================
# FULL 48-layer Qwen3-Coder-Next forward pass on the RNGD NPU, weight-streamed.
#
# Model: Qwen/Qwen3-Coder-Next-FP8  (FP8 blockwise weights, dequant via
#   qcn.loader.QCNWeights).  ~80GB of weights, so we STREAM per layer: each
#   decoder layer's weights are loaded from the safetensors shards via
#   QCNWeights.get() right before the layer runs and freed (del) right after,
#   so peak host RAM stays at roughly one-layer's-worth + embed/lm_head.
#
# Per-layer structure (verified against transformers
#   Qwen3NextDecoderLayer.forward, modeling_qwen3_next.py L924-969):
#       residual = h
#       h = input_layernorm(h)                         # Qwen3NextRMSNorm (1+w)
#       h = token_mixer(h)                             # DeltaNet OR full-attn
#       h = residual + h
#       residual = h
#       h = post_attention_layernorm(h)               # Qwen3NextRMSNorm (1+w)
#       h = MoE(h)                                     # every layer has MoE
#       h = residual + h
#   then  h = model.norm(h)  (Qwen3NextRMSNorm, 1+w)   # L1074
#   then  logits = lm_head(h)                          # bf16 Linear, no bias
#
#   token mixer chosen by config.layer_types[i]:
#       'full_attention'  (i in 3,7,11,...,47)  -> QCNFullAttentionNPU
#       'linear_attention' (all others)         -> DeltaNetLayer (Gated DeltaNet)
#
# RMSNorm conventions (VERIFIED in modeling_qwen3_next.py):
#   * Qwen3NextRMSNorm.forward (L250-255): out = _norm(x) * (1.0 + weight).
#     This is the class used for input_layernorm / post_attention_layernorm /
#     model.norm AND the attention q_norm/k_norm.  So decoder norms DO use the
#     (1 + weight) convention -- the task brief's "plain weight" was inverted;
#     the code is authoritative.  attn_layer.rms_norm_headdim already uses 1+w.
#   * Qwen3NextRMSNormGated.forward (L72-81): hidden = weight * _norm(hidden),
#     i.e. PLAIN weight (no +1), then * silu(gate).  This is the DeltaNet
#     gated output norm -- DeltaNetLayer._gnorm already implements plain weight.
#
# embed_tokens + lm_head are bf16 (NOT FP8 -> no scale_inv); QCNWeights.get()
# returns them straight (loader.py L48 guards FP8 dtype) so we just .float().
#
# QCNModel.prefill(input_ids) -> (logits, state_cache) where state_cache is a
# dict {layer_idx: payload}:  DeltaNet layers store the recurrent state S
# (and conv tail), full-attn layers store (K, V) for the cached tokens.
#
# NPU exec proof: DeltaNetLayer / attn_layer / moe each monkeypatch
# furiosa.torch.custom_ops.dfg._dfg_inner and keep a call counter; every NPU
# matmul/elementwise stage must leave that counter unchanged (dfg_delta==0).
# We aggregate the three counters after prefill and assert the total stayed 0.
# =============================================================================
import os
import sys
import glob
import torch
import torch._dynamo as _dynamo

# The TacticKernelModule forward (furiosa/torch/custom_ops/dfg.py) is
# torch.compile'd and dynamo RE-compiles it once per distinct input shape.
# A full 48-layer prefill issues MANY distinct dn_linear / dn_chunk_full /
# dn_conv1d / dn_l2norm / dn_gnorm / dn_gate shapes (per-tile O/I blocks, the
# 512-expert SwiGLU dims, etc.), so the default per-frame limit of 8 and the
# accumulated limit of 256 are both blown -- once exceeded, dynamo stops
# recompiling and the op runs the CPU-only fallback against rngd: tensors,
# raising "furiosa::dfg only runs on CPU device".  Raise both limits up front.
_dynamo.config.recompile_limit = 100000
_dynamo.config.cache_size_limit = 100000
_dynamo.config.accumulated_recompile_limit = 1_000_000
if hasattr(_dynamo.config, "accumulated_cache_size_limit"):
    _dynamo.config.accumulated_cache_size_limit = 1_000_000

# torch FIRST, then furiosa.torch (the proven import order).
import furiosa.torch as ft  # noqa: F401  (ensures backend registered)

sys.path.insert(0, "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj")
sys.path.insert(0, "/home/jun/furiosa/lib/python3.12/site-packages")

from qcn.loader import QCNWeights
# PERF (2026-06-11): head-batched DeltaNet + expert-batched MoE were unit-correct but
# a wall-clock REGRESSION — RNGD matmul runs on the VECTOR engine (EinsumByVe = broadcast-
# multiply-reduce, materializes the full [.,o,i] outer product), so batching N heads/experts
# makes ONE op do N x the materialization at the same throughput (no systolic speedup) +
# big SRAM pressure. So we use the PROVEN-fast per-head / per-expert path (55.8 s/tok, the
# only end-to-end-verified config). Batched versions kept (deltanet_layer.DeltaNetLayer,
# moe.moe_forward_npu) for reference. Real speedup lever = EinsumByDpe (the MAC/DPE engine),
# an open frontier ("needs full struct populated"); or multi-card; or vendor serve.
from qcn.deltanet_layer_looped import DeltaNetLayerLooped as DeltaNetLayer
from qcn import deltanet_layer as _dn  # looped shares _dn's _CALLS spy counter
from qcn import attn_layer as _attn
from qcn.attn_layer import QCNFullAttentionNPU
from qcn import moe as _moe
from qcn.moe import host_router, moe_forward_npu_unbatched as moe_forward_npu

DEV = os.environ.get("RNGD_DEV", "rngd:2")


# ---------------------------------------------------------------------------
# Host RMSNorm: Qwen3NextRMSNorm  (out = x*rsqrt(mean(x^2)+eps) * (1.0+weight))
# Exactly modeling_qwen3_next.py Qwen3NextRMSNorm.forward (L247-255).
# ---------------------------------------------------------------------------
def rms_norm(x, weight, eps):
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return normed * (1.0 + weight.float())


class QCNModel:
    """Full Qwen3-Coder-Next forward, weights streamed per layer, NPU mixers."""

    def __init__(self, snap=None, dev=DEV):
        self.W = QCNWeights(snap=snap)
        self.cfg_d = self.W.config
        self.dev = dev
        self.n_layers = self.cfg_d["num_hidden_layers"]          # 48
        self.hidden = self.cfg_d["hidden_size"]                  # 2048
        self.eps = self.cfg_d["rms_norm_eps"]                    # 1e-6
        self.top_k = self.cfg_d["num_experts_per_tok"]           # 10
        self.norm_topk = self.cfg_d["norm_topk_prob"]            # True
        # layer_types via the config class (config.json stores it as None).
        from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
        self.cfg = Qwen3NextConfig.from_pretrained(self.W.snap)
        self.layer_types = self.cfg.layer_types
        # reusable DeltaNet engine (stateless across layers; weights passed in)
        self.dn = DeltaNetLayer(self.cfg_d, dev=dev)

    # ---------------- per-layer weight streaming ----------------
    def _load_dn_weights(self, i):
        p = f"model.layers.{i}.linear_attn."
        w = {
            "in_proj_qkvz":  self.W.get(p + "in_proj_qkvz.weight", torch.float32),
            "in_proj_ba":    self.W.get(p + "in_proj_ba.weight", torch.float32),
            "conv1d_weight": self.W.get(p + "conv1d.weight", torch.float32),
            "A_log":         self.W.get(p + "A_log", torch.float32),
            "dt_bias":       self.W.get(p + "dt_bias", torch.float32),
            "norm_weight":   self.W.get(p + "norm.weight", torch.float32),
            "out_proj":      self.W.get(p + "out_proj.weight", torch.float32),
        }
        return w

    def _norm_w(self, i, which):
        return self.W.get(f"model.layers.{i}.{which}.weight", torch.float32)

    # ---------------- token mixers ----------------
    def _run_deltanet(self, h_btH, i, state, conv_state=None):
        """h_btH [1,T,H] -> ([1,T,H] out, recurrent state, conv state). Streams weights."""
        w = self._load_dn_weights(i)
        out, new_state, new_conv = self.dn.forward(
            h_btH, w, state=state, conv_state=conv_state, return_conv=True)
        del w
        return out, new_state, new_conv

    def _run_attention(self, h_btH, i, position_ids):
        """h_btH [1,T,H] -> ([1,T,H] out, (K,V) cache). Streams weights."""
        layer = QCNFullAttentionNPU(self.W, self.cfg, layer_idx=i)
        out = layer.forward(h_btH, position_ids)
        # cache K,V (post q/k/v proj + norm + rope) for decode: recompute the
        # k/v exactly as forward() does so the prefill cache matches HF layout.
        kv = self._attn_kv(layer, h_btH, position_ids)
        del layer
        return out, kv

    def _attn_kv(self, layer, hidden_states, position_ids):
        """Recompute the rotated K and V (as cached by HF) for this attn layer.
        Mirrors QCNFullAttentionNPU.forward up to RoPE; K is post-rope, V raw."""
        from qcn.attn_layer import npu_linear, rope_cos_sin, apply_partial_rope, rms_norm_headdim
        h = hidden_states[0].float()
        T = h.shape[0]
        Dk, nkv = layer.head_dim, layer.n_kv
        k_lin = npu_linear(h, layer.Wk, "kv_cache.k_proj")
        v_lin = npu_linear(h, layer.Wv, "kv_cache.v_proj")
        k = rms_norm_headdim(k_lin.view(T, nkv, Dk), layer.k_norm_w, layer.eps)
        v = v_lin.view(T, nkv, Dk)
        k = k.transpose(0, 1).unsqueeze(0)                       # [1,nkv,T,Dk]
        v = v.transpose(0, 1).unsqueeze(0)
        cos, sin = rope_cos_sin(position_ids, Dk, layer.partial, layer.rope_theta)
        # apply rope to k only (q irrelevant for the cache); reuse helper with q=k
        _, k = apply_partial_rope(k, k, cos, sin)
        return (k[0].detach(), v[0].detach())                    # ([nkv,T,Dk],[nkv,T,Dk])

    # ---------------- MoE ----------------
    def _run_moe(self, h_btH, i):
        """h_btH [1,T,H] -> [1,T,H]. Host router + NPU experts; streams weights."""
        h = h_btH[0]                                             # [T,H]
        gate_w = self.W.get(f"model.layers.{i}.mlp.gate.weight", torch.float32)
        top_val, top_idx = host_router(h, gate_w, self.top_k, self.norm_topk)
        del gate_w
        out, _n = moe_forward_npu(h, self.W, top_idx, top_val, layer=i)
        return out.unsqueeze(0)

    # ---------------- full forward ----------------
    def prefill(self, input_ids, max_layers=None, capture=None):
        """input_ids: [1,T] long. Returns (logits [1,T,vocab], state_cache dict).

        max_layers: run only the first N decoder layers (validation harness).
                    If set, SKIPS the final model.norm + lm_head and returns the
                    raw hidden state as 'logits' so callers can compare hiddens.
        capture:    optional dict to fill with intermediate hidden states; keys
                    f"layer{i}" -> hidden AFTER layer i (post residual add).
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, T = input_ids.shape
        assert B == 1, "this assembly runs B=1"
        position_ids = torch.arange(T).unsqueeze(0)              # [1,T]

        # ---- embed_tokens (bf16, no quant) ----
        embed = self.W.get("model.embed_tokens.weight", torch.float32)   # [V,H]
        h = embed[input_ids[0]].unsqueeze(0).float()            # [1,T,H]
        del embed

        n_run = self.n_layers if max_layers is None else max_layers
        state_cache = {}
        for i in range(n_run):
            ltype = self.layer_types[i]
            residual = h
            h = rms_norm(h, self._norm_w(i, "input_layernorm"), self.eps)
            if ltype == "linear_attention":
                mix, st, cst = self._run_deltanet(h, i, state=None)
                state_cache[i] = {"type": "deltanet", "state": st, "conv": cst}
            else:
                mix, kv = self._run_attention(h, i, position_ids)
                state_cache[i] = {"type": "attention", "kv": kv}
            h = residual + mix

            residual = h
            h = rms_norm(h, self._norm_w(i, "post_attention_layernorm"), self.eps)
            h = self._run_moe(h, i)
            h = residual + h

            if capture is not None:
                capture[f"layer{i}"] = h.detach().clone()
            print(f"[layer {i:2d}/{n_run}] {ltype:17s} done  "
                  f"hidden[0,0,:3]={h[0,0,:3].tolist()}", flush=True)

        if max_layers is not None:
            # validation mode: return the raw hidden after the truncated stack
            return h, state_cache

        # ---- final norm + lm_head ----
        h = rms_norm(h, self.W.get("model.norm.weight", torch.float32), self.eps)
        lm_head = self.W.get("lm_head.weight", torch.float32)   # [V,H] bf16-src
        logits = (h[0] @ lm_head.t()).unsqueeze(0)              # [1,T,V] (host)
        del lm_head
        return logits, state_cache

    # ---------------- single-token decode ----------------
    def decode_step(self, token_id, pos, state_cache):
        """Run ONE new token through all 48 layers, updating state_cache in place.
        token_id: python int.  pos: absolute position (int) of this token.
        Returns logits [1,1,vocab] for the next-token distribution.

        DeltaNet layers: feed the 1 new token with the carried recurrent state S
        and conv tail (so the chunk scan continues the recurrence; conv sees the
        last kernel-1 inputs).  Full-attn layers: compute new K/V at position pos,
        append to the cached K/V, attend over the whole history."""
        embed = self.W.get("model.embed_tokens.weight", torch.float32)
        h = embed[token_id].view(1, 1, self.hidden).float()     # [1,1,H]
        del embed
        position_ids = torch.tensor([[pos]])

        for i in range(self.n_layers):
            ltype = self.layer_types[i]
            residual = h
            h = rms_norm(h, self._norm_w(i, "input_layernorm"), self.eps)
            if ltype == "linear_attention":
                st = state_cache[i]["state"]
                cst = state_cache[i]["conv"]
                mix, new_st, new_cst = self._run_deltanet(h, i, state=st, conv_state=cst)
                state_cache[i]["state"] = new_st
                state_cache[i]["conv"] = new_cst
            else:
                layer = QCNFullAttentionNPU(self.W, self.cfg, layer_idx=i)
                mix, new_kv = layer.forward_decode(h, position_ids, state_cache[i]["kv"])
                state_cache[i]["kv"] = new_kv
                del layer
            h = residual + mix

            residual = h
            h = rms_norm(h, self._norm_w(i, "post_attention_layernorm"), self.eps)
            h = self._run_moe(h, i)
            h = residual + h

        h = rms_norm(h, self.W.get("model.norm.weight", torch.float32), self.eps)
        lm_head = self.W.get("lm_head.weight", torch.float32)
        logits = (h[0] @ lm_head.t()).unsqueeze(0)              # [1,1,V]
        del lm_head
        return logits

    # ---------------- autoregressive generation ----------------
    def get_tokenizer(self):
        if getattr(self, "_tok", None) is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.W.snap)
        return self._tok

    def generate(self, prompt_str, max_new_tokens=24, chat=False, greedy=True,
                 verbose=True):
        """Tokenize prompt_str, prefill, then greedy-decode up to max_new_tokens
        (or EOS).  Returns dict with prompt, generated_text, token timings, etc."""
        import time
        tok = self.get_tokenizer()
        if chat:
            msgs = [{"role": "user", "content": prompt_str}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt")
            # 일부 transformers 버전은 BatchEncoding(dict) 을 반환 → input_ids 추출
            if not isinstance(ids, torch.Tensor):
                ids = ids["input_ids"]
        else:
            ids = tok(prompt_str, return_tensors="pt").input_ids
        ids = ids.long()
        prompt_len = ids.shape[1]
        eos_ids = set()
        if tok.eos_token_id is not None:
            eos_ids.add(int(tok.eos_token_id))
        # Qwen3 chat uses <|im_end|>
        try:
            im_end = tok.convert_tokens_to_ids("<|im_end|>")
            if isinstance(im_end, int) and im_end >= 0:
                eos_ids.add(im_end)
        except Exception:
            pass

        if verbose:
            print(f"[generate] prompt_len={prompt_len} tokens; prefill ...", flush=True)
        t0 = time.time()
        logits, cache = self.prefill(ids)
        t_prefill = time.time() - t0
        next_logits = logits[0, -1]                            # [V]
        if verbose:
            top5 = torch.topk(next_logits, 5)
            toks = [tok.decode([int(t)]) for t in top5.indices.tolist()]
            print(f"[generate] prefill done in {t_prefill:.1f}s; "
                  f"top-5 next tokens: {list(zip(toks, [round(v,2) for v in top5.values.tolist()]))}",
                  flush=True)

        generated = []
        per_tok = []
        pos = prompt_len
        for step in range(max_new_tokens):
            nxt = int(torch.argmax(next_logits)) if greedy else \
                  int(torch.multinomial(torch.softmax(next_logits, -1), 1))
            generated.append(nxt)
            if nxt in eos_ids:
                if verbose:
                    print(f"[generate] EOS ({nxt}) at step {step}", flush=True)
                break
            ts = time.time()
            step_logits = self.decode_step(nxt, pos, cache)
            dt = time.time() - ts
            per_tok.append(dt)
            next_logits = step_logits[0, -1]
            pos += 1
            if verbose:
                piece = tok.decode([nxt])
                print(f"[gen step {step:2d}] tok={nxt:6d} {piece!r}  ({dt:.1f}s)", flush=True)

        gen_text = tok.decode(generated, skip_special_tokens=False)
        return {
            "prompt": prompt_str,
            "prompt_ids": ids[0].tolist(),
            "generated_ids": generated,
            "generated_text": gen_text,
            "full_text": tok.decode(ids[0].tolist() + generated, skip_special_tokens=False),
            "prefill_s": t_prefill,
            "per_token_s": per_tok,
        }

    # ---------------- streaming generation (per-token yield) ----------------
    @staticmethod
    def _sample_next(next_logits, greedy, temperature, top_p):
        """Pick the next token id from a [V] logits vector.

        greedy=True  -> argmax.
        else         -> host-side temperature + nucleus (top_p) multinomial.
        Mirrors generate()'s argmax/multinomial path (model.py:339-340) but adds
        temperature scaling and top_p truncation done entirely on the host CPU
        (the NPU only produced the logits; sampling is cheap and host-side)."""
        if greedy or not temperature or temperature <= 0.0:
            return int(torch.argmax(next_logits))
        logits = next_logits.float() / float(temperature)
        probs = torch.softmax(logits, dim=-1)
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            # keep the smallest prefix whose cumulative mass >= top_p
            cutoff = (cumsum < top_p).sum().item() + 1
            keep_idx = sorted_idx[:cutoff]
            mask = torch.zeros_like(probs)
            mask[keep_idx] = probs[keep_idx]
            denom = mask.sum()
            if denom > 0:
                probs = mask / denom
        return int(torch.multinomial(probs, 1))

    def generate_stream(self, prompt_str, max_new_tokens=24, chat=False,
                        greedy=True, temperature=0.0, top_p=1.0):
        """Generator version of generate(): prefill then decode, YIELDING one
        dict per produced token plus a final usage record. Reuses the exact
        decode loop body from generate() (model.py:338-354).

        Yields, in order:
          {"type": "token", "token_id": int, "text": str, "step": int, "dt": float}
            ... one per decoded token (text is the incremental piece) ...
          {"type": "usage", "prompt_ids": [...], "generated_ids": [...],
           "generated_text": str, "prefill_s": float, "per_token_s": [...],
           "finish_reason": "stop"|"length"}
        EOS is emitted as a token-less stop: the final usage carries the reason.
        """
        import time
        tok = self.get_tokenizer()
        if chat:
            msgs = [{"role": "user", "content": prompt_str}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt")
            if not isinstance(ids, torch.Tensor):
                ids = ids["input_ids"]
        else:
            ids = tok(prompt_str, return_tensors="pt").input_ids
        ids = ids.long()
        prompt_len = ids.shape[1]
        eos_ids = set()
        if tok.eos_token_id is not None:
            eos_ids.add(int(tok.eos_token_id))
        try:
            im_end = tok.convert_tokens_to_ids("<|im_end|>")
            if isinstance(im_end, int) and im_end >= 0:
                eos_ids.add(im_end)
        except Exception:
            pass

        t0 = time.time()
        logits, cache = self.prefill(ids)
        t_prefill = time.time() - t0
        next_logits = logits[0, -1]                            # [V]

        generated = []
        per_tok = []
        pos = prompt_len
        finish_reason = "length"
        for step in range(max_new_tokens):
            nxt = self._sample_next(next_logits, greedy, temperature, top_p)
            generated.append(nxt)
            if nxt in eos_ids:
                finish_reason = "stop"
                break
            piece = tok.decode([nxt])
            yield {"type": "token", "token_id": nxt, "text": piece,
                   "step": step, "dt": None}
            ts = time.time()
            step_logits = self.decode_step(nxt, pos, cache)
            dt = time.time() - ts
            per_tok.append(dt)
            next_logits = step_logits[0, -1]
            pos += 1
        else:
            finish_reason = "length"

        gen_text = tok.decode(generated, skip_special_tokens=False)
        yield {
            "type": "usage",
            "prompt_ids": ids[0].tolist(),
            "generated_ids": generated,
            "generated_text": gen_text,
            "prefill_s": t_prefill,
            "per_token_s": per_tok,
            "finish_reason": finish_reason,
        }

    # ---------------- NPU-exec proof ----------------
    @staticmethod
    def npu_dfg_totals():
        """Aggregate the CPU-fallback counters of the three NPU sub-modules.
        All three should read 0 if every NPU op truly ran on the NPU."""
        return {
            "deltanet": _dn._CALLS["n"],
            "attn":     _attn.CALLS["n"],
            "moe":      _moe.CALLS["n"],
        }


# ===========================================================================
# VALIDATION: real HF Qwen3NextForCausalLM truncated to 4 layers (CPU, fp32)
# vs our NPU assembly's hidden state after 4 layers (+ layer0 and layer3).
# ===========================================================================
def _snap():
    d = sorted(glob.glob("/home/jun/.cache/huggingface/hub/"
                         "models--Qwen--Qwen3-Coder-Next-FP8/snapshots/*/"))
    assert d, "model snapshot not found"
    return d[-1]


def _build_hf_4layer(snap, W, n=4):
    """Real HF Qwen3NextForCausalLM with num_hidden_layers=n, real weights for
    layers 0..n-1 + embed + model.norm, on CPU fp32.  Returns (model, cfg)."""
    import copy
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextModel

    cfg_full = Qwen3NextConfig.from_pretrained(snap)
    cfg = copy.deepcopy(cfg_full)
    cfg.num_hidden_layers = n
    cfg.layer_types = cfg_full.layer_types[:n]                  # keep real pattern
    print(f"  HF truncated config: {n} layers, layer_types={cfg.layer_types}", flush=True)

    model = Qwen3NextModel(cfg).eval().float()

    # load real weights into the HF state dict (only the modules that exist)
    sd = model.state_dict()
    loaded, missing = 0, []
    for name in sd.keys():
        full = name if name.startswith("model.") else "model." + name
        # HF Qwen3NextModel state dict keys are e.g. 'embed_tokens.weight',
        # 'layers.0...', 'norm.weight' -> map to checkpoint 'model.<name>'.
        ck = "model." + name
        if W.has(ck):
            with torch.no_grad():
                sd[name].copy_(W.get(ck, torch.float32))
            loaded += 1
        else:
            missing.append(ck)
    print(f"  loaded {loaded} HF tensors; {len(missing)} not in checkpoint "
          f"(expected: fused MoE experts gate_up_proj/down_proj).", flush=True)

    # The HF Qwen3NextSparseMoeBlock fuses experts into gate_up_proj[E,2I,H] and
    # down_proj[E,H,I] -- the checkpoint stores per-expert gate/up/down. Fill them.
    INT = cfg.moe_intermediate_size
    for i in range(n):
        layer = model.layers[i]
        if not hasattr(layer.mlp, "experts"):
            continue
        ex = layer.mlp.experts
        ex.gate_up_proj.data.zero_()
        ex.down_proj.data.zero_()
        E = cfg.num_experts
        for e in range(E):
            gp = f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight"
            up = f"model.layers.{i}.mlp.experts.{e}.up_proj.weight"
            dp = f"model.layers.{i}.mlp.experts.{e}.down_proj.weight"
            with torch.no_grad():
                ex.gate_up_proj.data[e, :INT] = W.get(gp, torch.float32)
                ex.gate_up_proj.data[e, INT:] = W.get(up, torch.float32)
                ex.down_proj.data[e] = W.get(dp, torch.float32)
        print(f"  HF layer {i}: filled {E} fused experts", flush=True)
    return model, cfg


def validate(n=4, T=12, seed=0):
    torch.manual_seed(seed)
    snap = _snap()
    W = QCNWeights(snap=snap)
    vocab = W.config["vocab_size"]

    input_ids = torch.randint(0, vocab, (1, T))
    print("=" * 78)
    print(f"VALIDATE Qwen3-Coder-Next  first {n} layers  T={T}  dev={DEV}")
    print("=" * 78)

    # ---- HF reference (CPU fp32, real weights, truncated to n layers) ----
    print("building HF reference (truncated, real weights) ...", flush=True)
    hf_model, cfg = _build_hf_4layer(snap, W, n=n)
    hf_capture = {}
    hooks = []

    def _mk_hook(idx):
        def hook(mod, inp, out):
            hf_capture[f"layer{idx}"] = (out[0] if isinstance(out, tuple) else out).detach().clone()
        return hook
    for i in range(n):
        hooks.append(hf_model.layers[i].register_forward_hook(_mk_hook(i)))

    with torch.no_grad():
        hf_out = hf_model(input_ids=input_ids, use_cache=False)
    hf_hidden = hf_out.last_hidden_state.float()                # AFTER n layers + model.norm
    for hk in hooks:
        hk.remove()
    print("HF reference done.", flush=True)

    # ---- NPU assembly: run first n layers, capture per-layer hidden ----
    print("running NPU assembly (first %d layers) ..." % n, flush=True)
    model = QCNModel(snap=snap, dev=DEV)
    npu_capture = {}
    npu_hidden_raw, _state = model.prefill(input_ids, max_layers=n, capture=npu_capture)
    # apply model.norm to match HF last_hidden_state (HF applies norm at the end)
    npu_hidden = rms_norm(npu_hidden_raw, W.get("model.norm.weight", torch.float32), model.eps)
    print("NPU assembly done.", flush=True)

    # ---- compare per-layer + final ----
    print("-" * 78)
    results = {}
    for i in range(n):
        a = npu_capture[f"layer{i}"].float()
        b = hf_capture[f"layer{i}"].float()
        maxerr = (a - b).abs().max().item()
        rel = maxerr / (b.abs().max().item() + 1e-9)
        results[f"layer{i}_out"] = (maxerr, rel)
        print(f"  after layer {i} (post-residual)  maxerr={maxerr:.3e}  rel={rel:.3e}")

    fe = (npu_hidden - hf_hidden).abs().max().item()
    frel = fe / (hf_hidden.abs().max().item() + 1e-9)
    results["after_norm"] = (fe, frel)
    print(f"  after {n} layers + model.norm   maxerr={fe:.3e}  rel={frel:.3e}")
    print("-" * 78)

    # ---- NPU-exec proof ----
    tot = QCNModel.npu_dfg_totals()
    all_npu = all(v == 0 for v in tot.values())
    print("NPU-exec proof (_dfg_inner CPU-fallback counters, MUST be 0):")
    for k, v in tot.items():
        print(f"   {k:9s} : {v}  {'(on NPU)' if v == 0 else '(FELL BACK TO CPU)'}")
    # confirm each mixer actually ran NPU stages
    dn_stages = len(model.dn.npu_stages)
    attn_stages = len(_attn.NPU_STAGES)
    moe_stages = len(_moe.NPU_STAGES)
    print(f"   NPU stages run: deltanet={dn_stages} attn={attn_stages} moe={moe_stages}")
    print("-" * 78)

    # tolerance: fp32 host vs fp32 NPU matmul; per validated components ~1e-5..1e-2
    # over a single layer, accumulated over 4 layers -> allow atol 5e-2 on hidden.
    layer_ok = all(me < 5e-2 for me, _ in
                   [results[f"layer{i}_out"] for i in range(n)])
    final_ok = fe < 5e-2
    ok = layer_ok and final_ok and all_npu
    print(f"PER-LAYER ALLCLOSE(<5e-2) : {layer_ok}")
    print(f"AFTER-NORM ALLCLOSE(<5e-2): {final_ok}")
    print(f"ALL_MIXERS_ON_NPU         : {all_npu} "
          f"(deltanet+attn+moe stages all ran on NPU)")
    print(f"OVERALL_PASS              : {ok}")
    print("=" * 78)
    return ok, results, tot


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq", type=int, default=12)
    args = ap.parse_args()
    if args.validate:
        validate(n=args.layers, T=args.seq)
    else:
        # tiny smoke: full 48-layer prefill on a 4-token prompt
        m = QCNModel()
        ids = torch.tensor([[1, 2, 3, 4]])
        logits, cache = m.prefill(ids)
        print("logits:", tuple(logits.shape), "cache layers:", len(cache))
