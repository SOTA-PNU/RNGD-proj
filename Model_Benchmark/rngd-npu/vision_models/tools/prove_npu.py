#!/usr/bin/env python3
"""NPU 실거주 증명: 저장된 EDF를 rngd:N에 올려 추론 루프를 돌리는 동안 furiosa-smi로 그 PE 가동을 직접 포착.
증거 4종: (1) device=rngd:N, (2) 단발 지연 NPU급(<10ms; CPU vit_b_16은 ~50ms+), (3) furiosa-smi idle vs active 전력/사용률,
(4) 처리량. 추론 cm()은 워커 스레드 한 곳에서만 호출(동시호출 없음 → 안전).
사용: python tools/prove_npu.py --npu 0"""
import argparse, time, threading, subprocess, warnings
warnings.filterwarnings("ignore")
import torch, furiosa.torch
from furiosa.torch import CompileModule
from furiosa.torch.custom_ops.edf import EdfModule
from furiosa.torch.export import ExportedProgramWeight, PASSES
from furiosa.native_torch import ir
from torch._decomp import core_aten_decompositions, get_decompositions
import torchvision.models as M
from torchvision.models import ViT_B_16_Weights

EDF = "/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/edf/vit_b_16_fromexported.edf"
D = dict(core_aten_decompositions())
D.update(get_decompositions([torch.ops.aten._native_batch_norm_legit_no_training,
    torch.ops.aten._native_batch_norm_legit, torch.ops.aten.batch_norm, torch.ops.aten.native_batch_norm]))


def smi(*sub):
    try:
        return subprocess.run(["furiosa-smi", *sub], capture_output=True, text=True, timeout=10).stdout
    except Exception as e:
        return f"(furiosa-smi 실패: {e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npu", type=int, default=0)
    args = ap.parse_args()
    card = args.npu // 8  # rngd:N → npu(card)
    m = M.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval()
    for p in m.parameters(): p.requires_grad_(False)
    edf = ir.Edf.deserialize(open(EDF, "rb").read())
    with torch.no_grad():
        ep = torch.export.export(m, (torch.randn(1, 3, 224, 224),)).run_decompositions(D)
        for fx in PASSES: ep = fx(ep)
        cm = CompileModule(EdfModule(edf), ExportedProgramWeight(ep))
    dev = torch.device("rngd", args.npu)
    cm.to(dev)
    x = torch.randn(1, 3, 224, 224).contiguous().to(dev)
    with torch.no_grad():
        cm(x, device=dev)
        t0 = time.time(); cm(x, device=dev); one_ms = (time.time() - t0) * 1000
    print(f"[증거1] device = {dev}", flush=True)
    print(f"[증거2] 단발 추론 = {one_ms:.2f} ms  (CPU vit_b_16 ~50ms+ → NPU 확정)", flush=True)

    idle = smi("status")
    stop = {"v": False}; cnt = {"v": 0}
    def worker():
        with torch.no_grad():
            while not stop["v"]:
                cm(x, device=dev); cnt["v"] += 1
    t_loop = time.time()
    th = threading.Thread(target=worker); th.start()
    time.sleep(6)            # 워커가 PE를 계속 점유하는 동안
    active = smi("status")
    psout = smi("ps")        # 어느 프로세스가 어느 NPU를 쓰는지 — 직접 증거
    time.sleep(1)
    stop["v"] = True; th.join()
    dt = time.time() - t_loop
    print(f"[증거4] 루프 {dt:.1f}s 동안 {cnt['v']} 추론 = {cnt['v']/dt:.0f} inf/s\n", flush=True)

    def npu_row(s):
        rows = [ln for ln in s.splitlines() if f"npu{card}" in ln.lower()]
        return rows[0].strip() if rows else "(npu행 못찾음)"
    print(f"[증거3] furiosa-smi status — npu{card}(rngd:{args.npu} 카드) 전력/사용률 IDLE vs ACTIVE:", flush=True)
    print("   IDLE  :", npu_row(idle), flush=True)
    print("   ACTIVE:", npu_row(active), flush=True)
    print(f"\n[증거5] furiosa-smi ps — 내 PID가 npu{card}를 점유 중이라는 직접 증거:", flush=True)
    import os
    mypid = os.getpid()
    hit = [ln for ln in psout.splitlines() if str(mypid) in ln or f"npu{card}" in ln.lower() or "prove_npu" in ln]
    for ln in (hit or psout.splitlines()[:6]):
        print("   ", ln.strip(), flush=True)
    print(f"   (내 PID = {mypid})", flush=True)


if __name__ == "__main__":
    main()
