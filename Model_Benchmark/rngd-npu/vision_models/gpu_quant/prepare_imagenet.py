#!/usr/bin/env python3
"""GPU 서버에서 ImageNet-1k val 균형 부분집합을 HF에서 받아 ./imagenet_val/ 에 저장.
non-gated 미러(evanarlian/imagenet_1k_resized_256, split 'val', 클래스당 50장, 라벨=표준 ILSVRC2012/torchvision 순서).
기본 10장/클래스 = 10000장. HF 토큰 불필요.

사용: python prepare_imagenet.py            # 10/class
      python prepare_imagenet.py --per_class 5
"""
import argparse, os, csv
from datasets import load_dataset

OUT = os.path.join(os.path.dirname(__file__), "imagenet_val")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class", type=int, default=10)
    ap.add_argument("--max_side", type=int, default=384)
    args = ap.parse_args()
    os.makedirs(f"{OUT}/images", exist_ok=True)

    ds = load_dataset("evanarlian/imagenet_1k_resized_256", split="val", streaming=True)
    names = None
    try:
        info = load_dataset("evanarlian/imagenet_1k_resized_256", split="val", streaming=True)
        names = info.features["label"].names
    except Exception:
        pass

    per_class = {}
    rows = []
    idx = 0
    for ex in ds:
        lab = int(ex["label"])
        if per_class.get(lab, 0) >= args.per_class:
            continue
        img = ex["image"].convert("RGB")
        if max(img.size) > args.max_side:
            r = args.max_side / max(img.size)
            img = img.resize((int(img.size[0] * r), int(img.size[1] * r)))
        fn = f"{idx:05d}.jpg"
        img.save(f"{OUT}/images/{fn}", quality=92)
        lname = names[lab] if names else str(lab)
        rows.append((fn, lab, lname))
        per_class[lab] = per_class.get(lab, 0) + 1
        idx += 1
        if idx % 1000 == 0:
            print(f"  saved {idx} (classes filled: {sum(1 for v in per_class.values() if v>=args.per_class)})", flush=True)
        if len(per_class) >= 1000 and all(v >= args.per_class for v in per_class.values()) and len(per_class) == 1000:
            break

    with open(f"{OUT}/labels.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["filename", "label_idx", "label_name"]); w.writerows(rows)
    with open(f"{OUT}/META.txt", "w") as f:
        f.write(f"dataset=evanarlian/imagenet_1k_resized_256 split=val\n")
        f.write(f"per_class={args.per_class} total={idx} classes={len(per_class)}\n")
        f.write("label_idx = standard ILSVRC2012 / torchvision ViT_B_16_Weights order\n")
    open(f"{OUT}/DONE", "w").close()
    print(f"[done] {idx} images, {len(per_class)} classes -> {OUT}")


if __name__ == "__main__":
    main()
