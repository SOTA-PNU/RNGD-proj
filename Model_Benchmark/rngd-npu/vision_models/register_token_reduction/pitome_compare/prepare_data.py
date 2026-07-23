#!/usr/bin/env python3
"""GPU 서버에서 ImageNet-1k 를 HF non-gated 미러(evanarlian/imagenet_1k_resized_256)에서 받아 저장(토큰 불필요).
정통 kNN 프로토콜용: gallery=train(전체 128만), query=val(5만). 라벨=표준 ILSVRC2012/torchvision 순서.
사용:
  python prepare_data.py --split val                      # query: val 5만(클래스당 50)
  python prepare_data.py --split train --per_class 1300   # gallery: train 전체(클래스당 ~최대 1300 => 사실상 전량)
※ resized-256 미러라 224 kNN 엔 충분하고 원본 165GB 보다 훨씬 작게 받습니다."""
import argparse, os, csv
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--per_class", type=int, default=0, help="클래스당 최대 장수. 0=해당 split 전량")
    ap.add_argument("--max_side", type=int, default=384)
    args = ap.parse_args()
    cap = args.per_class if args.per_class > 0 else 10**9
    OUT = os.path.join(os.path.dirname(__file__), f"imagenet_{args.split}")
    os.makedirs(f"{OUT}/images", exist_ok=True)
    ds = load_dataset("evanarlian/imagenet_1k_resized_256", split=args.split, streaming=True)
    names = ds.features["label"].names if hasattr(ds, "features") and ds.features.get("label") else None
    per, rows, idx = {}, [], 0
    for ex in ds:
        lab = int(ex["label"])
        if per.get(lab, 0) >= cap:
            continue
        img = ex["image"].convert("RGB")
        if max(img.size) > args.max_side:
            r = args.max_side / max(img.size); img = img.resize((int(img.size[0]*r), int(img.size[1]*r)))
        fn = f"{idx:07d}.jpg"; img.save(f"{OUT}/images/{fn}", quality=92)
        rows.append((fn, lab, names[lab] if names else str(lab))); per[lab] = per.get(lab, 0)+1; idx += 1
        if idx % 5000 == 0: print(f"  saved {idx}", flush=True)
        if args.per_class > 0 and len(per) == 1000 and all(v >= cap for v in per.values()): break
    with open(f"{OUT}/labels.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["filename", "label_idx", "label_name"]); w.writerows(rows)
    open(f"{OUT}/DONE", "w").close()
    print(f"[done] split={args.split} {idx} images, {len(per)} classes -> {OUT}")


if __name__ == "__main__":
    main()
