#!/usr/bin/env python3
"""DeltaNet recurrent step / conv1d / gate 를 '연산 1개 = 그래프 1개'로 쪼개
   a6 EDF 블롭으로 컴파일해 아티팩트의 _edf_blobs/ 에 추가하고 _MASTER_summary.json
   을 갱신한다. 이전에 a6 불가로 누락됐던 3조각(deltanet_recurrent_step,
   dn_conv1d_silu, dn_gate)을 분해형 블롭으로 대체한다.

   분해 근거(실측 dn_decompose_probe.py): a6 컴파일러는 한 그래프 안에 복수
   contraction 패턴('conflict between concrete labels')이나 복수 독립 출력
   서브그래프('multiple internal subgraphs')가 있으면 거부한다. 연산을 하나씩
   쪼개면 통과한 Linear/SDPA/SwiGLU 와 같은 단일패턴 그래프가 되어 a6 통과한다.
   쪼갠 시퀀스는 원본 recurrent step 과 fp64 정확(fp32 rel ~2.9e-7).

run:
  PYTHONPATH=/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj RNGD_DEV=rngd:5 \
    /home/jun/furiosa/bin/python tk_kernels/emit_dn_split_blobs.py
"""
import os, json, time, hashlib, traceback
import torch, torch.nn as nn, torch.nn.functional as F
from furiosa.native_common import compiler

TARGET = "renegade-8pe"
ART = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen3-coder-next-fp8-rngd"
BLOB_DIR = os.path.join(ART, "_edf_blobs")
MASTER = os.path.join(BLOB_DIR, "_MASTER_summary.json")

# real config (per-step, all 32 value heads, head_dim 128)
NV, DK, DV = 32, 128, 128
CONV_CH, SEQ, K = 8192, 16, 4
GATE_NV = 32

A6 = bytes.fromhex("a163456466a6656e6f646573")
new_recs = []

def emit(piece, mod, args, note):
    t0 = time.time()
    try:
        r = compiler.compile(mod.eval(), tuple(args), TARGET, target_ir="edf")
        cg = r.graphs[0]; blob = cg.serialize(); h = hashlib.md5(blob).hexdigest()
        open(os.path.join(BLOB_DIR, f"{h}.edf"), "wb").write(blob)
        rec = {"piece": piece, "status": "ok", "edf_hash": h, "bytes": len(blob),
               "header_a6_confirmed": A6 in blob[:64], "npu_node": bool(cg.is_edf()),
               "input_shapes": [list(a.shape) for a in args], "in_dim": None,
               "out_dim": None, "note": note, "compile_s": round(time.time()-t0, 2),
               "error": None}
        print(f"  OK   {piece:22s} a6={rec['header_a6_confirmed']} {len(blob):>9}B {h}")
    except Exception as e:
        rec = {"piece": piece, "status": "FAIL", "edf_hash": None, "bytes": None,
               "header_a6_confirmed": None, "npu_node": None,
               "input_shapes": [list(a.shape) for a in args], "note": note,
               "compile_s": round(time.time()-t0, 2),
               "error": (type(e).__name__+": "+str(e))[:300]}
        print(f"  FAIL {piece:22s} {rec['error']}")
    new_recs.append(rec)
    return rec

# ---- recurrent-step split (5 ops) ----
class Decay(nn.Module):
    def forward(self, state, g): return state * g.exp().view(state.shape[0], 1, 1)
class Contract(nn.Module):            # bmm(vec[nv,1,dk], state[nv,dk,dv]) -> [nv,dv]
    def forward(self, state, vec): return torch.bmm(vec.unsqueeze(1), state).squeeze(1)
class Delta(nn.Module):
    def forward(self, v, kv, beta): return (v - kv) * beta.unsqueeze(-1)
class Outer(nn.Module):               # bmm(k[nv,dk,1], delta[nv,1,dv]) -> [nv,dk,dv]
    def forward(self, k, delta): return torch.bmm(k.unsqueeze(2), delta.unsqueeze(1))
class Add(nn.Module):
    def forward(self, state, upd): return state + upd

# ---- conv1d short-conv (host-pad + shift-mul-add + SiLU) ----
class ConvShift(nn.Module):
    def __init__(self, ch, k, L):
        super().__init__()
        self.w = nn.Parameter(torch.randn(ch, k)); self.b = nn.Parameter(torch.randn(ch))
        self.k, self.L = k, L
    def forward(self, xp):
        out = self.b.unsqueeze(-1)
        for j in range(self.k):
            out = out + self.w[:, j:j+1] * xp[:, j:j+self.L]
        return F.silu(out)

# ---- gate split (beta, g) — 각각 단일 출력 ----
class GateBeta(nn.Module):
    def forward(self, b): return torch.sigmoid(b)
class GateG(nn.Module):                # g = -exp(A_log) * log(1+exp(a+dt))  (softplus 대체)
    def __init__(self, nv):
        super().__init__()
        self.A_log = nn.Parameter(torch.randn(nv)); self.dt = nn.Parameter(torch.randn(nv))
    def forward(self, a):
        return -torch.exp(self.A_log) * torch.log(1.0 + torch.exp(a + self.dt))

def main():
    print(f"compiler {compiler.full_version()}  ->  {BLOB_DIR}")
    print("emitting DeltaNet split a6 blobs (recurrent / conv1d / gate):")
    st = torch.randn(NV, DK, DV)
    emit("dn_recur_decay",    Decay(),    (st, torch.randn(NV)),                         "state * alpha (per-head scalar decay)")
    emit("dn_recur_contract", Contract(), (st, torch.randn(NV, DK)),                     "bmm vec@state contraction (kv_mem AND readout)")
    emit("dn_recur_delta",    Delta(),    (torch.randn(NV, DV), torch.randn(NV, DV), torch.randn(NV)), "(v - kv) * beta")
    emit("dn_recur_outer",    Outer(),    (torch.randn(NV, DK), torch.randn(NV, DV)),    "bmm k(x)delta outer product")
    emit("dn_recur_add",      Add(),      (st, torch.randn(NV, DK, DV)),                 "state + outer_update")
    emit("dn_conv1d_shift",   ConvShift(CONV_CH, K, SEQ), (torch.randn(CONV_CH, SEQ+K-1),), "depthwise short-conv K=4 as host-pad+shift-mul-add+SiLU (replaces O136 Conv1d)")
    emit("dn_gate_beta",      GateBeta(), (torch.randn(SEQ, GATE_NV),),                  "beta = sigmoid(b)")
    emit("dn_gate_g",         GateG(GATE_NV), (torch.randn(SEQ, GATE_NV),),              "g = -exp(A_log)*log(1+exp(a+dt)) (softplus via log(1+exp))")

    # ---- merge into _MASTER_summary.json ----
    m = json.load(open(MASTER))
    replaced = {"dn_conv1d_silu", "dn_gate", "deltanet_recurrent_step"}
    kept = [r for r in m["results"] if r.get("piece") not in replaced]
    m["results"] = kept + new_recs
    ok = [r for r in m["results"] if r.get("status") == "ok"]
    bad = [r for r in m["results"] if r.get("status") != "ok"]
    m["a6_edf_produced"] = [r["piece"] for r in ok]
    m["no_a6_edf"] = [{"piece": r["piece"], "reason": r.get("error")} for r in bad]
    m["counts"] = {"total": len(m["results"]), "ok": len(ok), "fail": len(bad)}
    m["description"] = (m.get("description", "") +
        " | 2026-06-15: DeltaNet recurrent step/conv1d/gate split into single-op "
        "a6 EDF blobs (decay/contract/delta/outer/add, conv-shift, gate beta/g) "
        "and added; the 3 previously-uncompilable pieces are now in the bundle.")
    json.dump(m, open(MASTER, "w"), indent=2, ensure_ascii=False)
    print(f"\n  master summary updated: ok={len(ok)} fail={len(bad)} "
          f"(replaced {sorted(replaced)} with {len(new_recs)} split blobs)")
    print(f"  new ok pieces: {[r['piece'] for r in new_recs if r['status']=='ok']}")
    if bad:
        print(f"  still failing: {[r['piece'] for r in bad]}")

if __name__ == "__main__":
    main()
