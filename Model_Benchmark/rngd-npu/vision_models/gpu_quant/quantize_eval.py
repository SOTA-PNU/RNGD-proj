#!/usr/bin/env python3
"""GPU 양자화 평가: ViT/DeiT 6모델을 FP32/FP16/BF16/INT8/FP8로 양자화하고
ImageNet val에서 top-1 정확도 + GPU 지연/throughput 측정.

NPU(furiosa.torch)는 비전 양자화 공개 API가 없어, 양자화 비교는 GPU에서 수행한다.
이 폴더는 GPU 서버에 복사해 그대로 실행하면 된다(README 참고).

사용:
  python quantize_eval.py                       # 전체(6모델 × 전 precision)
  python quantize_eval.py --models vit_base_patch16_224.augreg_in1k --batches 1 32 128
  python quantize_eval.py --modes fp32 fp16 int8_weight int8_dynamic fp8
출력: results_gpu_quant.json + 표
"""
import argparse, time, json, os, csv, warnings
warnings.filterwarnings("ignore")
import torch
import timm
from PIL import Image

MODELS = [
    "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "vit_small_patch16_224.augreg_in1k",
    "vit_base_patch16_224.augreg_in1k",
    "deit_tiny_patch16_224.fb_in1k",
    "deit_small_patch16_224.fb_in1k",
    "deit_base_patch16_224.fb_in1k",
]
# (mode, base dtype) — int8/fp8은 torchao 권장대로 bf16 베이스 위에서
MODE_DTYPE = {
    "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16,
    "int8_weight": torch.bfloat16, "int8_dynamic": torch.bfloat16, "fp8": torch.bfloat16,
}
VAL_DIR = os.path.join(os.path.dirname(__file__), "imagenet_val")


def load_val(model, n):
    cfg = timm.data.resolve_model_data_config(model)
    tf = timm.data.create_transform(**cfg, is_training=False)
    rows = list(csv.DictReader(open(f"{VAL_DIR}/labels.csv")))[:n]
    xs, ys = [], []
    for r in rows:
        p = f"{VAL_DIR}/images/{r['filename']}"
        if os.path.exists(p):
            xs.append(tf(Image.open(p).convert("RGB"))); ys.append(int(r["label_idx"]))
    return torch.stack(xs), torch.tensor(ys)


def quantize(model, mode):
    """양자화 적용. 지원 안 되면 예외 → 호출부에서 skip."""
    base = MODE_DTYPE[mode]
    model = model.to(base)
    if mode in ("fp32", "fp16", "bf16"):
        return model
    from torchao.quantization import quantize_  # torch>=2.4 + torchao 필요
    if mode == "int8_weight":
        from torchao.quantization import int8_weight_only
        quantize_(model, int8_weight_only())
    elif mode == "int8_dynamic":
        from torchao.quantization import int8_dynamic_activation_int8_weight
        quantize_(model, int8_dynamic_activation_int8_weight())
    elif mode == "fp8":
        from torchao.quantization import float8_weight_only
        quantize_(model, float8_weight_only())
    else:
        raise ValueError(mode)
    return model


@torch.no_grad()
def measure(model, X, Y, device, batch, base_dtype, iters):
    model.eval()
    # 정확도 (전체 val)
    correct = total = 0
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(device).to(base_dtype)
        out = model(xb).float().cpu()
        correct += (out.argmax(-1) == Y[i:i + batch]).sum().item(); total += xb.shape[0]
    acc = round(100 * correct / total, 2)
    # 지연/throughput
    xb = X[:batch].to(device).to(base_dtype)
    if xb.shape[0] < batch:
        xb = xb[torch.arange(batch) % xb.shape[0]]
    for _ in range(5): model(xb)
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters): model(xb)
    if device.type == "cuda": torch.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000
    return acc, round(ms, 3), round(batch / (ms / 1000), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--modes", nargs="+", default=["fp32", "fp16", "bf16", "int8_weight", "int8_dynamic", "fp8"])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 32, 128])
    ap.add_argument("--n_val", type=int, default=10000)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results_gpu_quant.json"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("⚠ CUDA 미감지 — GPU 서버에서 실행하세요. (CPU로도 정확도는 측정되나 속도 의미없음)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}", flush=True)
    if not os.path.exists(f"{VAL_DIR}/labels.csv"):
        print(f"⚠ {VAL_DIR}/labels.csv 없음 — 먼저 `python prepare_imagenet.py` 실행"); return

    results = []
    for mn in args.models:
        X, Y = load_val(timm.create_model(mn, pretrained=False), args.n_val)
        print(f"\n=== {mn}  (val {len(X)}장) ===", flush=True)
        for mode in args.modes:
            for b in args.batches:
                rec = {"model": mn, "mode": mode, "batch": b}
                try:
                    m = timm.create_model(mn, pretrained=True).eval().to(device)
                    m = quantize(m, mode)
                    acc, ms, ips = measure(m, X, Y, device, b, MODE_DTYPE[mode], args.iters)
                    rec.update(top1=acc, ms_per_batch=ms, img_per_s=ips, ok=True)
                    print(f"  {mode:13s} b={b:<4d} top1={acc:5.2f}%  {ms:8.3f} ms/batch  {ips:8.1f} img/s", flush=True)
                    del m
                    if device.type == "cuda": torch.cuda.empty_cache()
                except Exception as e:
                    rec.update(ok=False, err=f"{type(e).__name__}: {str(e)[:100]}")
                    print(f"  {mode:13s} b={b:<4d} SKIP ({rec['err']})", flush=True)
                results.append(rec)
                json.dump(results, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
