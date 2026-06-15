"""Qwen3-Coder-Next-FP8 가중치 로더 + FP8 blockwise dequant.

- safetensors mmap 으로 레이어별 온디맨드 로드(전체 80GB 를 RAM 에 안 올림).
- FP8(e4m3) blockwise(128x128) dequant: w_bf16 = w_fp8.float() * scale_block (브로드캐스트).
  (lm_head, embed_tokens 는 비양자화 — scale 없음, 그대로.)
"""
import json, glob, os
import torch
from safetensors import safe_open

REPO_SNAP_GLOB = "/home/jun/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next-FP8/snapshots/*/"
BLOCK = 128


def _snap():
    d = sorted(glob.glob(REPO_SNAP_GLOB))
    assert d, "model snapshot not found (download in progress?)"
    return d[-1]


class QCNWeights:
    """index 를 읽고, 텐서명->shard 매핑으로 필요한 텐서만 mmap 로드."""

    def __init__(self, snap=None):
        self.snap = snap or _snap()
        idx = json.load(open(os.path.join(self.snap, "model.safetensors.index.json")))
        self.weight_map = idx["weight_map"]
        self.config = json.load(open(os.path.join(self.snap, "config.json")))
        self._handles = {}  # shard file -> safe_open handle

    def _h(self, fname):
        if fname not in self._handles:
            self._handles[fname] = safe_open(os.path.join(self.snap, fname), framework="pt", device="cpu")
        return self._handles[fname]

    def has(self, name):
        return name in self.weight_map

    def raw(self, name):
        """원시 텐서(FP8 이면 e4m3 dtype 그대로). shard 가 아직 안 받아졌으면 KeyError/IOError."""
        fname = self.weight_map[name]
        return self._h(fname).get_tensor(name)

    def get(self, name, dtype=torch.float32):
        """dequant 된 텐서를 반환. weight_scale_inv 가 있으면 FP8 blockwise dequant."""
        w = self.raw(name)
        scale_name = name + "_scale_inv"
        if self.has(scale_name) and w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            scale = self.raw(scale_name).float()  # [ceil(out/128), ceil(in/128)]
            return _dequant_blockwise(w, scale, dtype)
        return w.to(dtype)


def _dequant_blockwise(w_fp8, scale, out_dtype=torch.float32, block=BLOCK):
    """w_fp8: [O, I] float8; scale: [ceil(O/block), ceil(I/block)] fp32.
    각 [block, block] 타일에 해당 스케일을 곱한다."""
    w = w_fp8.float()
    O, I = w.shape
    sO, sI = scale.shape
    # 스케일을 원소 단위로 확장: repeat_interleave 후 [O,I] 로 크롭
    s_full = scale.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:O, :I]
    return (w * s_full).to(out_dtype)


if __name__ == "__main__":
    W = QCNWeights()
    print("snap:", W.snap)
    print("model_type:", W.config["model_type"], "layers:", W.config["num_hidden_layers"])
    # 받아진 텐서 하나로 dequant 테스트 (layer0 DeltaNet in_proj)
    name = "model.layers.0.linear_attn.in_proj_qkvz.weight"
    try:
        w = W.get(name, torch.float32)
        raw = W.raw(name)
        print(f"{name}: raw dtype={raw.dtype} shape={tuple(raw.shape)} -> dequant {tuple(w.shape)} {w.dtype}")
        print(f"  raw[0,:4]={raw.float()[0,:4].tolist()}  dequant[0,:4]={w[0,:4].tolist()}")
        sc = W.raw(name + "_scale_inv")
        print(f"  scale_inv shape={tuple(sc.shape)} dtype={sc.dtype} [0,:3]={sc.float()[0,:3].tolist()}")
    except Exception as e:
        print(f"  (shard for {name} not downloaded yet: {type(e).__name__}: {e})")
