#!/usr/bin/env python3
"""GPU 서버에서 ImageNet-1k val 균형 부분집합을 HF에서 받아 ./imagenet_val/ 에 저장(토큰 불필요).
non-gated 미러 evanarlian/imagenet_1k_resized_256, split 'val'(클래스당 50장, 라벨=표준 ILSVRC2012/torchvision 순서).
사용: python prepare_data.py --per_class 50   (기본 50=풀 50k)"""
import argparse, os, csv
from datasets import load_dataset

OUT = os.path.join(os.path.dirname(__file__), "imagenet_val")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class", type=int, default=50)
    ap.add_argument("--max_side", type=int, default=384)
    args = ap.parse_args()
    os.makedirs(f"{OUT}/images", exist_ok=True)
    ds = load_dataset("evanarlian/imagenet_1k_resized_256", split="val", streaming=True)
    names = ds.features["label"].names if hasattr(ds, "features") and ds.features.get("label") else None
    per, rows, idx = {}, [], 0
    for ex in ds:
        lab = int(ex["label"])
        if per.get(lab, 0) >= args.per_class:
            continue
        img = ex["image"].convert("RGB")
        if max(img.size) > args.max_side:
            r = args.max_side / max(img.size); img = img.resize((int(img.size[0]*r), int(img.size[1]*r)))
        fn = f"{idx:06d}.jpg"; img.save(f"{OUT}/images/{fn}", quality=92)
        rows.append((fn, lab, names[lab] if names else str(lab))); per[lab] = per.get(lab, 0)+1; idx += 1
        if idx % 2000 == 0: print(f"  saved {idx}", flush=True)
        if len(per) == 1000 and all(v >= args.per_class for v in per.values()): break
    with open(f"{OUT}/labels.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["filename", "label_idx", "label_name"]); w.writerows(rows)
    open(f"{OUT}/DONE", "w").close()
    print(f"[done] {idx} images, {len(per)} classes -> {OUT}")


if __name__ == "__main__":
    main()
