#!/usr/bin/env python3
"""원본 bf16 Qwen2.5-72B-Instruct 를 RNGD NPU에서 돌리는 host 추론 루프(Q25Model).

배경: bf16 72B(135GiB)는 furiosa-llm 표준 serve가 전부 막힘(tp32 빌드=inter-chip
DramShapeGuide 미구현, pp4 serve=인터칩 가중치 바인딩 실패, pp2=2장 초과). radare2/
allow_inter_chip_dram 플래그로도 안 됨(실측, 미구현 native 기능). 그래서 Qwen3-Coder-Next
80B에서 검증한 방식대로, host가 추론 루프를 들고 레이어별로 가중치를 NPU에 스트리밍한다.
Qwen2.5는 표준 dense 트랜스포머라 DeltaNet보다 단순하고, qcn 인프라(npu_linear·SDPA·
QCNWeights·HostLoopEngine)를 그대로 재사용한다.

인터페이스(HostLoopEngine/serve 어댑터 호환): get_tokenizer / prefill(ids)->(logits,cache)
/ decode_step(tok,pos,cache)->logits / generate / npu_dfg_totals.

⚠️ 느림(bf16 dense는 토큰마다 135GiB 스트리밍). 정확도 우선. 실행:
  PYTHONPATH=<proj> RNGD_DEV=rngd:4 QCN_DPE=1 ~/furiosa/bin/python qcn/qwen25_model.py --smoke
"""
import os, sys, glob, time
import torch

sys.path.insert(0, "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj")
import furiosa.torch as ft  # noqa
from qcn.loader import QCNWeights
from qcn import attn_layer as _attn
from qcn.attn_layer import npu_linear, npu_matmul_AB, rope_cos_sin, apply_partial_rope

DEV = os.environ.get("RNGD_DEV", "rngd:4")
SNAP_GLOB = "/home/jun/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/*/"


def _snap():
    d = sorted(glob.glob(SNAP_GLOB))
    assert d, "Qwen2.5-72B snapshot not found"
    return d[-1]


def rms_norm(x, weight, eps):
    """Qwen2RMSNorm: out = x*rsqrt(mean(x^2)+eps) * weight  (plain weight, NOT 1+w)."""
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return normed * weight.float()


class Q25Model:
    """Qwen2.5-72B host 추론 루프. 가중치 레이어별 스트리밍, token-mixer/MLP는 NPU."""

    def __init__(self, snap=None, dev=DEV):
        self.snap = snap or _snap()
        self.W = QCNWeights(snap=self.snap)
        c = self.W.config
        self.dev = dev
        self.n_layers = c["num_hidden_layers"]            # 80
        self.hidden = c["hidden_size"]                    # 8192
        self.n_q = c["num_attention_heads"]               # 64
        self.n_kv = c["num_key_value_heads"]              # 8
        self.head_dim = c.get("head_dim") or (self.hidden // self.n_q)  # 128
        self.n_rep = self.n_q // self.n_kv                # 8
        self.eps = c.get("rms_norm_eps", 1e-6)
        self.theta = c.get("rope_theta", 1000000.0)
        self.scale = self.head_dim ** -0.5
        _attn.DEV = dev                                   # attn_layer 헬퍼가 쓰는 디바이스
        self._tok = None

    def get_tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.snap)
        return self._tok

    # ---------------- attention (GQA, full RoPE, qkv bias) ----------------
    def _attn(self, h, i, position_ids, kv_cache=None):
        """h: [T,H] float. returns (out [T,H], (K,V) cache). kv_cache=(Kprev,Vprev) for decode."""
        p = f"model.layers.{i}.self_attn."
        T = h.shape[0]
        Dk, nq, nkv = self.head_dim, self.n_q, self.n_kv
        q = npu_linear(h, self.W.get(p + "q_proj.weight"), "q") + self.W.get(p + "q_proj.bias").float()
        k = npu_linear(h, self.W.get(p + "k_proj.weight"), "k") + self.W.get(p + "k_proj.bias").float()
        v = npu_linear(h, self.W.get(p + "v_proj.weight"), "v") + self.W.get(p + "v_proj.bias").float()
        q = q.view(T, nq, Dk).transpose(0, 1).unsqueeze(0)      # [1,nq,T,Dk]
        k = k.view(T, nkv, Dk).transpose(0, 1).unsqueeze(0)     # [1,nkv,T,Dk]
        v = v.view(T, nkv, Dk).transpose(0, 1).unsqueeze(0)
        cos, sin = rope_cos_sin(position_ids, Dk, 1.0, self.theta)   # full RoPE
        q, k = apply_partial_rope(q, k, cos, sin)
        k = k[0]; v = v[0]; q = q[0]                            # [heads,T,Dk]
        if kv_cache is not None:
            Kp, Vp = kv_cache
            k = torch.cat([Kp, k], dim=1); v = torch.cat([Vp, v], dim=1)   # append on T
        Ktot = k.shape[1]
        new_cache = (k.detach(), v.detach())
        # GQA + per-head SDPA (matmul on NPU, softmax on host) + causal mask
        out = torch.empty(nq, T, Dk)
        for hd in range(nq):
            qh = q[hd]                                          # [T,Dk]
            kv_idx = hd // self.n_rep
            kh = k[kv_idx]; vh = v[kv_idx]                      # [Ktot,Dk]
            scores = npu_matmul_AB(qh.contiguous(), kh.contiguous(), f"qk[{hd}]") * self.scale  # [T,Ktot]
            if T > 1:  # prefill causal mask (decode: T=1 sees all)
                off = Ktot - T
                m = torch.full((T, Ktot), float("-inf"))
                for r in range(T):
                    m[r, : off + r + 1] = 0.0
                scores = scores + m
            attn = torch.softmax(scores, dim=-1)
            out[hd] = npu_matmul_AB(attn.contiguous(), vh.t().contiguous(), f"av[{hd}]")  # [T,Dk]
        out = out.transpose(0, 1).reshape(T, nq * Dk)          # [T,H]
        out = npu_linear(out, self.W.get(p + "o_proj.weight"), "o")
        return out, new_cache

    # ---------------- MLP (SwiGLU) ----------------
    def _mlp(self, h, i):
        p = f"model.layers.{i}.mlp."
        g = npu_linear(h, self.W.get(p + "gate_proj.weight"), "gate")
        u = npu_linear(h, self.W.get(p + "up_proj.weight"), "up")
        act = torch.nn.functional.silu(g) * u
        return npu_linear(act, self.W.get(p + "down_proj.weight"), "down")

    # ---------------- one layer ----------------
    def _layer(self, h, i, position_ids, kv_cache=None):
        nw = self.W.get(f"model.layers.{i}.input_layernorm.weight")
        a, cache = self._attn(rms_norm(h, nw, self.eps), i, position_ids, kv_cache)
        h = h + a
        pw = self.W.get(f"model.layers.{i}.post_attention_layernorm.weight")
        h = h + self._mlp(rms_norm(h, pw, self.eps), i)
        return h, cache

    # ---------------- prefill ----------------
    def prefill(self, input_ids, max_layers=None):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, T = input_ids.shape
        assert B == 1
        position_ids = torch.arange(T).unsqueeze(0)
        embed = self.W.get("model.embed_tokens.weight")
        h = embed[input_ids[0]].float()                        # [T,H]
        del embed
        n_run = self.n_layers if max_layers is None else max_layers
        cache = {}
        for i in range(n_run):
            h, cache[i] = self._layer(h, i, position_ids)
            print(f"[prefill {i:2d}/{n_run}] h[0,:3]={h[0,:3].tolist()}", flush=True)
        if max_layers is not None:
            return h.unsqueeze(0), cache
        h = rms_norm(h, self.W.get("model.norm.weight"), self.eps)
        lm = self.W.get("lm_head.weight")
        logits = (h @ lm.t()).unsqueeze(0)                     # [1,T,V] (host)
        return logits, cache

    # ---------------- decode ----------------
    def decode_step(self, token_id, pos, cache):
        embed = self.W.get("model.embed_tokens.weight")
        h = embed[int(token_id)].view(1, self.hidden).float()  # [1,H]
        del embed
        position_ids = torch.tensor([[pos]])
        for i in range(self.n_layers):
            h, cache[i] = self._layer(h, i, position_ids, kv_cache=cache[i])
        h = rms_norm(h, self.W.get("model.norm.weight"), self.eps)
        lm = self.W.get("lm_head.weight")
        return (h @ lm.t()).unsqueeze(0)                       # [1,1,V]

    # ---------------- generate (smoke) ----------------
    def generate(self, prompt, max_new_tokens=8, chat=True, greedy=True, verbose=True):
        tok = self.get_tokenizer()
        if chat:
            ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          add_generation_prompt=True, return_tensors="pt")
            if not isinstance(ids, torch.Tensor):
                ids = ids["input_ids"]
        else:
            ids = tok(prompt, return_tensors="pt").input_ids
        ids = ids.long()
        t0 = time.time()
        logits, cache = self.prefill(ids)
        nxt_logits = logits[0, -1]
        gen, per = [], []
        pos = ids.shape[1]
        eos = {tok.eos_token_id}
        for s in range(max_new_tokens):
            nxt = int(torch.argmax(nxt_logits))
            gen.append(nxt)
            if nxt in eos:
                break
            ts = time.time()
            nxt_logits = self.decode_step(nxt, pos, cache)[0, -1]
            per.append(time.time() - ts)
            pos += 1
            if verbose:
                print(f"[gen {s}] {nxt} {tok.decode([nxt])!r} ({per[-1]:.1f}s)", flush=True)
        return {"generated_text": tok.decode(gen, skip_special_tokens=True),
                "prefill_s": logits is not None and (time.time() - t0), "per_token_s": per,
                "prompt_ids": ids[0].tolist(), "generated_ids": gen}

    @staticmethod
    def npu_dfg_totals():
        return {"attn": _attn.CALLS["n"]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prompt", default="What is the capital of France? Answer in one word.")
    ap.add_argument("--max-new", type=int, default=4)
    args = ap.parse_args()
    m = Q25Model()
    print(f"Q25Model loaded: {m.n_layers}L hidden{m.hidden} q{m.n_q}/kv{m.n_kv} hd{m.head_dim} dev={m.dev}", flush=True)
    if args.smoke:
        out = m.generate(args.prompt, max_new_tokens=args.max_new)
        print("GENERATED:", repr(out["generated_text"]), flush=True)
        print("per_token_s:", [round(x, 1) for x in out["per_token_s"]], flush=True)
        print("CPU-fallback:", Q25Model.npu_dfg_totals(), flush=True)
