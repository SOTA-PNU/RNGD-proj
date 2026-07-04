#!/usr/bin/env python3
"""[새 서버] ImageNet-1k 를 non-gated HF 미러에서 받아 DATA_ROOT/imagenet_{split} 로 저장합니다(HF 토큰 불필요).
  - val   : 쿼리 5만(클래스당 50). 논문 val-LOO 와 동일 구성.
  - train : 갤러리 전체(클래스당 최대 1300 ≒ 전량 1.28M). 정통 train-갤러리 kNN 용.
라벨은 표준 ILSVRC2012/torchvision 순서. 미러는 resized-256 이라 224 실험에 충분하고 원본(165GB)보다 훨씬 작습니다.
저장 위치는 config.sh 가 export 하는 $DATA_ROOT(기본 <bundle>/data). 이미 DONE 이면 건너뜁니다.
사용:
  python prepare_data.py --split val
  python prepare_data.py --split train --per_class 1300
"""
import argparse, os, csv
from datasets import load_dataset

DEFAULT_PER = {"val": 50, "train": 1300}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "train"], required=True)
    ap.add_argument("--per_class", type=int, default=0, help="0=split 기본값(val 50 / train 1300≒전량)")
    ap.add_argument("--max_side", type=int, default=384)
    ap.add_argument("--out", default=None, help="미지정 시 $DATA_ROOT/imagenet_<split>")
    args = ap.parse_args()
    cap = args.per_class if args.per_class > 0 else DEFAULT_PER[args.split]
    cap = cap if cap > 0 else 10**9
    root = args.out or os.path.join(
        os.environ.get("DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")),
        f"imagenet_{args.split}")
    if os.path.exists(f"{root}/DONE"):
        print(f"[skip] 이미 준비됨: {root}"); return
    os.makedirs(f"{root}/images", exist_ok=True)
    ds = load_dataset("evanarlian/imagenet_1k_resized_256", split=args.split, streaming=True)
    names = ds.features["label"].names if hasattr(ds, "features") and ds.features.get("label") else None
    per, rows, idx = {}, [], 0
    for ex in ds:
        lab = int(ex["label"])
        if per.get(lab, 0) >= cap:
            continue
        img = ex["image"].convert("RGB")
        if max(img.size) > args.max_side:
            rr = args.max_side / max(img.size); img = img.resize((int(img.size[0]*rr), int(img.size[1]*rr)))
        fn = f"{idx:07d}.jpg"; img.save(f"{root}/images/{fn}", quality=92)
        rows.append((fn, lab, names[lab] if names else str(lab))); per[lab] = per.get(lab, 0)+1; idx += 1
        if idx % 5000 == 0:
            print(f"  saved {idx}", flush=True)
        if cap < 10**9 and len(per) == 1000 and all(v >= cap for v in per.values()):
            break
    with open(f"{root}/labels.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["filename", "label_idx", "label_name"]); w.writerows(rows)
    open(f"{root}/DONE", "w").close()
    print(f"[done] split={args.split} {idx} images, {len(per)} classes -> {root}")


if __name__ == "__main__":
    main()
