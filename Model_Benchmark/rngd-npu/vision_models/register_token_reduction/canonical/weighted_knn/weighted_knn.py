#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
정통(train-갤러리) kNN을 'DINOv2 공식식 온도가중 투표'로 재채점하는 **독립 스크립트**.

목적: 논문의 정통 baseline(다수결 80.87)이 공인값 82.0에 조금 못 미치는 가장 큰 원인이
      "우리가 비가중 다수결을 쓴 것"임을 확인/보정한다. 공식 DINOv2 kNN은 가까운 이웃에
      가중치(w=exp(cos/T))를 주는 온도가중 투표를 쓴다.

설계 원칙(사용자 지시): **기존 코드(compare.py 등)를 수정하지 않는다.** 이 파일은 compare.py의
      특징추출 함수(extract_split·make_tf·reduced_forward)를 '읽기 전용'으로 import 만 하고,
      kNN 채점만 여기서 새로(가중/다수결 둘 다) 수행한다.

★전제조건: 이 도구는 단독으로 못 돈다. **pitome_compare 엔진 + 그 데이터/캐시**가 있어야 한다
      (보통은 정통 실험을 돌린 GPU 서버에 이미 있음). 엔진 위치는 아래 순서로 찾는다:
        ① 환경변수 ENGINE  ② --engine 인자  ③ 기본값 ../../pitome_compare
      찾은 pitome_compare 안의 imagenet_val/·imagenet_train/(데이터)와 feat_cache/(캐시)를 쓴다.
      캐시가 있으면 재추출 없이 ~수초, 없으면 데이터로 새로 추출(느림).

출력: (strat, r) 별로 majority(=논문 현재 수치)와 weighted(=공식식)를 나란히 찍는다.
      r=0 baseline 의 weighted 값이 82.0 에 얼마나 가까워지는지가 핵심.
"""
import os, sys, argparse, torch
import torch.nn.functional as F
import timm

HERE = os.path.dirname(os.path.abspath(__file__))


def load_engine(engine_arg):
    """pitome_compare 엔진을 찾아 읽기 전용 import. 못 찾으면 친절한 에러로 종료."""
    cands = [engine_arg, os.environ.get("ENGINE"), os.path.join(HERE, "..", "..", "pitome_compare")]
    for c in cands:
        if not c:
            continue
        c = os.path.abspath(c)
        if os.path.exists(os.path.join(c, "compare.py")):
            sys.path.insert(0, c)
            import compare as eng   # 원본 미변경, 읽기 전용 재사용
            return eng, c
    sys.exit(
        "[에러] pitome_compare 엔진(compare.py)을 못 찾았습니다.\n"
        "  이 도구는 pitome_compare 엔진과 그 데이터/캐시가 있어야 돕니다(단독 실행 불가).\n"
        "  다음 중 하나로 해결하세요:\n"
        "   ① weighted_knn 폴더를 register_token_reduction/canonical/ 안에 두기(기본 경로 ../../pitome_compare)\n"
        "   ② python weighted_knn.py --engine /경로/pitome_compare\n"
        "   ③ ENGINE=/경로/pitome_compare python weighted_knn.py\n"
        f"  (찾아본 위치: {[os.path.abspath(c) for c in cands if c]})")


def preflight(eng, cache_dir):
    """데이터·캐시 상태를 점검해 사람이 읽을 상태를 찍고, 아무것도 없으면 중단."""
    data_ok = os.path.exists(f"{eng.HERE}/imagenet_val/DONE")
    npt = len([f for f in os.listdir(cache_dir) if f.endswith(".pt")]) if os.path.isdir(cache_dir) else 0
    print(f"[전제] 엔진={eng.HERE}", flush=True)
    print(f"[전제] 캐시={cache_dir}  (.pt {npt}개)  |  데이터(imagenet_val/DONE)={'있음' if data_ok else '없음'}", flush=True)
    if npt == 0 and not data_ok:
        sys.exit("[에러] 캐시도 데이터도 없습니다. 이 GPU 서버에서 정통 실험(run_base_canonical.sh)을 먼저 돌렸어야 합니다.\n"
                 "       (또는 pitome_compare/prepare_data.py 로 imagenet_val/train 준비.)")
    if npt == 0:
        print("[주의] 캐시가 없어 특징을 새로 추출합니다(수 시간 걸릴 수 있음). 캐시가 있으면 ~수초.", flush=True)


@torch.no_grad()
def knn_both(Gf, Gy, Qf, Qy, k, dev, temp, chunk=256):
    """한 번의 유사도 계산으로 majority(비가중 다수결)와 weighted(온도가중) 정확도를 동시 산출.
    Gf/Gy=갤러리 특징·라벨, Qf/Qy=쿼리 특징·라벨. 반환: (majority_acc%, weighted_acc%)."""
    G = F.normalize(Gf.to(dev).float(), dim=-1).half(); Gy = Gy.to(dev)
    Qn = F.normalize(Qf.float(), dim=-1).half()
    C = int(Gy.max().item()) + 1
    cm = cw = 0
    for i in range(0, len(Qn), chunk):
        s = Qn[i:i+chunk].to(dev) @ G.T
        sim, idx = s.topk(k, dim=1)
        ytop = Gy[idx]
        maj = torch.mode(ytop, dim=1).values                 # 비가중 다수결(=논문 현재 프로토콜)
        w = (sim.float() / temp).exp()                       # 온도가중치 exp(cos/T)
        probs = torch.zeros(ytop.shape[0], C, device=dev)
        probs.scatter_add_(1, ytop, w)
        wtd = probs.argmax(1)                                # DINOv2 공식식 가중 투표
        y = Qy[i:i+s.shape[0]].to(dev)
        cm += (maj == y).sum().item(); cw += (wtd == y).sum().item()
    n = len(Qn); return 100 * cm / n, 100 * cw / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--temp", type=float, default=0.07, help="가중 투표 온도(DINOv2 공식 기본 0.07)")
    ap.add_argument("--r_list", type=int, nargs="+", default=[8, 12, 16, 18, 20])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--engine", default=None, help="pitome_compare 경로(미지정시 ENGINE 환경변수 → ../../pitome_compare)")
    ap.add_argument("--cache_dir", default=None, help="특징 캐시(미지정시 <engine>/feat_cache)")
    args = ap.parse_args()

    eng, engine_dir = load_engine(args.engine)
    cache_dir = args.cache_dir or os.path.join(engine_dir, "feat_cache")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = timm.create_model(args.model, pretrained=True, num_classes=0, img_size=224).eval().to(dev)
    tf = eng.make_tf(m)
    nprefix = getattr(m, "num_prefix_tokens", 1); npatch = m.patch_embed.num_patches; L = len(m.blocks)
    mtag = args.model.split(".")[0]
    strategies = ["tome", "pitome", "ours"]

    print(f"[setup] {args.model} dev={dev} prefix={nprefix} patches={npatch} blocks={L} k={args.k} temp={args.temp}", flush=True)
    preflight(eng, cache_dir)
    print(f"[proto] 정통 kNN(gallery=train, query=val) 재채점 — majority(논문) vs weighted(공식식)", flush=True)
    print(f"\n{'r':>3} {'comp%':>6} {'strat':>7} {'majority':>9} {'weighted':>9}   Δ(w-m)", flush=True)

    rows = [0] + list(args.r_list)
    for r in rows:
        for st in (["ours"] if r == 0 else strategies):      # r=0 은 세 방법 동일 → 1회만
            Qf, Qy = eng.extract_split(m, "val", r, st, nprefix, tf, args.batch, args.workers, dev, cache_dir, mtag)
            Gf, Gy = eng.extract_split(m, "train", r, st, nprefix, tf, args.batch, args.workers, dev, cache_dir, mtag)
            maj, wtd = knn_both(Gf, Gy, Qf, Qy, args.k, dev, args.temp)
            comp = 0.0 if r == 0 else 100 * (1 - (nprefix + max(npatch - L * r, 1)) / (nprefix + npatch))
            tag = "  ← baseline (공인 82.0 과 대조)" if r == 0 else ""
            print(f"{r:>3} {comp:6.1f} {st:>7} {maj:9.2f} {wtd:9.2f}   {wtd-maj:+.2f}{tag}", flush=True)

    print("\n해석: r=0 weighted 가 82.0 에 근접하면 → '격차 대부분이 투표방식(가중 여부)'임이 확인됨.", flush=True)
    print("      각 r 에서 majority→weighted 로 바뀌어도 세 방법의 상대 순서가 유지되면 → 결론이 투표방식에 불변.", flush=True)
    print("      두 조건이 맞으면 tab:canonical 을 weighted 열로 갱신하고 격차 문장을 '가중 kNN로 82.0 재현'으로 강화 가능.", flush=True)


if __name__ == "__main__":
    main()
