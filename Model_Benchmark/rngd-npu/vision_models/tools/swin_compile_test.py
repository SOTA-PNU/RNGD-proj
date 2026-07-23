#!/usr/bin/env python3
"""make-or-break #1: 윈도우 어텐션(roll/window-partition/merge)이 furiosa.torch로 컴파일되나?
SwinIR(SR)은 timm에 없지만 코어 연산이 같은 timm Swin 분류 모델로 프록시 테스트.
미지원 op는 lowering서 빨리 실패 → fail-fast. 사용: python tools/swin_compile_test.py"""
import time, warnings
warnings.filterwarnings("ignore")
import torch, furiosa.torch
from furiosa.torch import CompileModule
from torch._decomp import core_aten_decompositions, get_decompositions
import timm

D = dict(core_aten_decompositions())
D.update(get_decompositions([torch.ops.aten._native_batch_norm_legit_no_training,
    torch.ops.aten._native_batch_norm_legit, torch.ops.aten.batch_norm, torch.ops.aten.native_batch_norm]))

MODELS = ["swin_tiny_patch4_window7_224", "swinv2_tiny_window8_256"]

for name in MODELS:
    try:
        m = timm.create_model(name, pretrained=False).eval()
        for p in m.parameters(): p.requires_grad_(False)
        cfg = timm.data.resolve_model_data_config(m)
        h = cfg["input_size"][1]
        x = torch.randn(1, 3, h, h)
        t = time.time()
        with torch.no_grad():
            ep = torch.export.export(m, (x,)).run_decompositions(D)
        nodes = sum(1 for n in ep.graph.nodes if n.op == "call_function")
        with torch.no_grad():
            cm = CompileModule.from_exported(ep)
        print(f"[{name}] COMPILE_OK nodes={nodes} {time.time()-t:.0f}s", flush=True)
    except Exception as e:
        c = e
        while getattr(c, "__cause__", None) is not None: c = c.__cause__
        msg = str(c).splitlines()[0] if str(c) else type(c).__name__
        print(f"[{name}] FAIL after {time.time()-t:.0f}s :: {msg[:160]}", flush=True)
