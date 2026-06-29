"""pp4 pipeline-parallel host-loop for Qwen3-Coder-Next.

Splits the 48 decoder layers into N stages (default 4) across the 4 physical RNGD
cards.  Each stage is its OWN process pinned to one card (RNGD_DEV set before
`qcn.model` import, because the modules capture the device at import time).  A
request flows stage0 -> stage1 -> ... -> stageN-1; the host activation `h`
([1,T,H] prefill / [1,1,H] decode) is marshalled worker->worker over mp.Queue.
Each stage owns the recurrent/KV state for ITS layer slice, so the DeltaNet
read-modify-write recurrence stays local to the stage across decode steps.

Stage 0 also does embed_tokens; the last stage also does final RMSNorm + lm_head.
The compute on each stage is the SAME `QCNModel._run_layer_range` the proven
single-card path uses, so pp4 output matches single-card (modulo ~0.23% NPU fp).

`PipelineModel` exposes `prefill` / `decode_step` / `generate` / `get_tokenizer`
with the same signatures as `QCNModel`, so it is a drop-in for run_artifact, the
HostLoopEngine adapter, and the official `furiosa-llm serve` CLI shim.

Why one PE per card (not tp8): `set_fusion(8)` meshes all 8 PE of a card but the
hand-authored TacticKernels (SRAM-tiled for 1 PE) fail to lower under fusion
(`dn_l2norm` -> UnsupportedOpError: unsupported EDF node Cpu(...)).  The model is
also host-bound (decode ~40s/token from weight dequant), so tp8 adds ~no speedup.
pp4 is the axis that actually uses the 4 cards.  See README_qwen3_coder_next.md.

Usage:
    PYTHONPATH=$PROJ python qcn/pipeline.py --prompt "def add(a, b):" --max-new 4
    QCN_CARDS=4 (stages), QCN_PE="0,8,16,24" (one PE per physical card), QCN_DPE=1
"""
import os
import sys
import time
import argparse
import multiprocessing as mp

PROJ = "/home/jun/RNGD-proj/Model_Benchmark/qwen3-next-proj"
DEFAULT_ART = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/artifacts/qwen3-coder-next-fp8-rngd"
# global PE index for one PE on each physical card (0-7=card0,8-15=card1,...).
DEFAULT_PE = [0, 8, 16, 24]


def _stage_worker(stage_idx, n_stages, lo, hi, dev, artifact_dir,
                  in_q, next_q, result_q, ready_q, dpe):
    """One pipeline stage, pinned to `dev`, owning decoder layers [lo, hi)."""
    os.environ["RNGD_DEV"] = dev            # MUST precede qcn.model import
    os.environ["QCN_DPE"] = dpe
    sys.path.insert(0, PROJ)
    # die if the parent (coordinator/serve) dies, even on kill -9 (else we orphan
    # and keep the card busy, breaking the next run with an allocator/EBUSY error).
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, 9)   # PR_SET_PDEATHSIG=1, SIGKILL=9
    except Exception:
        pass
    try:
        import torch  # noqa: F401  (torch first, then furiosa.torch — proven order)
        import qcn.run_artifact as ra
        manifest = ra.load_manifest(artifact_dir)
        ra._redirect_kernels(artifact_dir, manifest["model"]["kernels"])  # use artifact kernels
        snap = ra._resolve_snapshot(manifest, artifact_dir)
        from qcn.model import QCNModel
        model = QCNModel(snap=snap)         # dev = RNGD_DEV
        ready_q.put(("ready", stage_idx, dev, lo, hi))
    except Exception as e:
        import traceback
        ready_q.put(("fatal", stage_idx, f"{e}\n{traceback.format_exc()}"))
        return

    state_cache = {}                         # this stage's layer-slice state
    while True:
        msg = in_q.get()
        if msg is None:
            break
        kind = msg[0]
        try:
            if kind == "reset":
                state_cache.clear()
                (next_q.put(msg) if next_q is not None else result_q.put(("reset_done",)))
                continue
            if kind == "prefill":
                h = model.embed_tokens(msg[1]) if stage_idx == 0 else msg[1]
                T = h.shape[1]
                pos_ids = torch.arange(T).unsqueeze(0)
                h = model._run_layer_range(h, lo, hi, state_cache,
                                           position_ids=pos_ids, decode=False)
                fwd = ("prefill", h)
            else:  # decode: msg = ("decode", token_or_h, pos)
                pos = msg[2]
                h = model.embed_tokens(int(msg[1])) if stage_idx == 0 else msg[1]
                pos_ids = torch.tensor([[pos]])
                h = model._run_layer_range(h, lo, hi, state_cache,
                                           position_ids=pos_ids, decode=True)
                fwd = ("decode", h, pos)
            if next_q is not None:
                next_q.put(fwd)
            else:
                result_q.put(("logits", model.final_logits(h)))
        except Exception as e:
            import traceback
            result_q.put(("error", f"stage{stage_idx}: {e}\n{traceback.format_exc()}"))


class PipelineModel:
    """pp-parallel drop-in for QCNModel (prefill / decode_step / generate)."""

    def __init__(self, snap=None, artifact_dir=None, n_stages=None, pe=None, dpe=None):
        self.artifact_dir = artifact_dir or os.environ.get("QCN_ART", DEFAULT_ART)
        self.n_stages = int(n_stages or os.environ.get("QCN_CARDS", 4))
        pe_env = os.environ.get("QCN_PE")
        self.pe = pe or ([int(x) for x in pe_env.split(",")] if pe_env else DEFAULT_PE[:self.n_stages])
        assert len(self.pe) >= self.n_stages, f"need >= {self.n_stages} PE indices, got {self.pe}"
        dpe = dpe or os.environ.get("QCN_DPE", "1")

        # read layer count from the artifact config
        import json
        cfg = json.load(open(os.path.join(self.artifact_dir, "config.json")))
        self.cfg_d = cfg                         # QCNModel-compatible (cli-shim reads this)
        self.n_layers = cfg["num_hidden_layers"]
        self.hidden = cfg["hidden_size"]
        per = self.n_layers // self.n_stages
        self.ranges = [(k * per, (k + 1) * per if k < self.n_stages - 1 else self.n_layers)
                       for k in range(self.n_stages)]

        ctx = mp.get_context("spawn")
        self.in_qs = [ctx.Queue() for _ in range(self.n_stages)]
        self.result_q = ctx.Queue()
        ready_q = ctx.Queue()
        self.procs = []
        for k in range(self.n_stages):
            lo, hi = self.ranges[k]
            next_q = self.in_qs[k + 1] if k < self.n_stages - 1 else None
            p = ctx.Process(target=_stage_worker, daemon=True,
                            args=(k, self.n_stages, lo, hi, f"rngd:{self.pe[k]}",
                                  self.artifact_dir, self.in_qs[k], next_q,
                                  self.result_q, ready_q, dpe))
            p.start()
            self.procs.append(p)

        ready = 0
        t0 = time.time()
        while ready < self.n_stages:
            ev = ready_q.get()
            if ev[0] == "ready":
                ready += 1
                print(f"[pp] stage {ev[1]} ready  dev={ev[2]}  layers[{ev[3]},{ev[4]})", flush=True)
            else:
                raise RuntimeError(f"[pp] stage {ev[1]} FAILED to load:\n{ev[2]}")
        print(f"[pp] {self.n_stages} stages up in {time.time()-t0:.1f}s "
              f"({self.n_layers} layers split {self.ranges})", flush=True)
        self._tok = None

    # ---- QCNModel-compatible surface ----
    def prefill(self, input_ids, max_layers=None, capture=None):
        if hasattr(input_ids, "dim") and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        self.in_qs[0].put(("prefill", input_ids))
        tag, payload = self.result_q.get()
        if tag == "error":
            raise RuntimeError(payload)
        return payload, {"__pp_handle__": True}   # state lives in the stage workers

    def decode_step(self, token_id, pos, state_cache=None):
        self.in_qs[0].put(("decode", int(token_id), pos))
        tag, payload = self.result_q.get()
        if tag == "error":
            raise RuntimeError(payload)
        return payload

    def reset(self):
        self.in_qs[0].put(("reset",))
        self.result_q.get()

    def get_tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            src = self.artifact_dir if os.path.exists(
                os.path.join(self.artifact_dir, "tokenizer.json")) else None
            self._tok = AutoTokenizer.from_pretrained(src or self.artifact_dir)
        return self._tok

    def generate(self, prompt_str, max_new_tokens=8, chat=False, greedy=True,
                 temperature=0.0, top_p=1.0, verbose=True):
        import torch
        tok = self.get_tokenizer()
        if chat:
            msgs = [{"role": "user", "content": prompt_str}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        else:
            ids = tok(prompt_str, return_tensors="pt").input_ids
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        prompt_ids = ids[0].tolist()
        t0 = time.time()
        self.reset()
        logits, _ = self.prefill(ids)
        nxt = int(torch.argmax(logits[0, -1]))
        pos = len(prompt_ids)
        prefill_s = time.time() - t0
        out_ids = [nxt]
        if verbose:
            print(f"[pp] prefill {prefill_s:.1f}s  tok0={nxt} {tok.decode([nxt])!r}", flush=True)
        for step in range(1, max_new_tokens):
            ts = time.time()
            logits = self.decode_step(nxt, pos)
            nxt = int(torch.argmax(logits[0, -1]))
            pos += 1
            out_ids.append(nxt)
            if verbose:
                print(f"[pp] step {step}  {time.time()-ts:.1f}s  tok={nxt} {tok.decode([nxt])!r}", flush=True)
        gen_text = tok.decode(out_ids)
        return {"prompt": prompt_str, "prompt_ids": prompt_ids,
                "generated_ids": out_ids, "generated_text": gen_text,
                "full_text": prompt_str + gen_text, "prefill_s": prefill_s}

    def shutdown(self):
        for q in self.in_qs:
            try:
                q.put(None)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=DEFAULT_ART)
    ap.add_argument("--prompt", default="def add(a, b):")
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--stages", type=int, default=int(os.environ.get("QCN_CARDS", 4)))
    ap.add_argument("--chat", action="store_true")
    args = ap.parse_args()
    print("=" * 78)
    print(f"pp{args.stages} host-loop generate  artifact={args.artifact}")
    print("=" * 78)
    m = PipelineModel(artifact_dir=args.artifact, n_stages=args.stages)
    res = m.generate(args.prompt, max_new_tokens=args.max_new, chat=args.chat, greedy=True)
    print("-" * 78)
    print("PROMPT    :", repr(res["prompt"]))
    print("GENERATED :", repr(res["generated_text"]))
    print("FULL TEXT :\n" + res["full_text"])
    print("-" * 78)
    m.shutdown()


if __name__ == "__main__":
    main()
