#!/usr/bin/env python3
"""Phase 0b: cosine proxy를 '실제 분류 정확도'로 강화(CPU, 재학습 없음).
DINOv2-reg의 CLS 특징을 {full, tome(CLS만보호), ours(register+고노름보호)}로 추출 → ImageNet val 라벨로
leave-one-out kNN 정확도 비교. ours가 tome보다 높으면 'register 보호가 실제 정확도를 살린다' 입증.
사용: python phase0b_knn_accuracy.py --n 1200"""
import argparse, warnings, csv, os
warnings.filterwarnings("ignore")
import torch, timm
import torch.nn.functional as F
from PIL import Image
from phase0_register_tome import merge_step, run_reduced  # 동일 병합 재사용

VAL="/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/imagenet_val"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=1200); ap.add_argument("--k",type=int,default=20)
    args=ap.parse_args()
    name="vit_base_patch14_reg4_dinov2.lvd142m"
    m=timm.create_model(name,pretrained=True,num_classes=0,img_size=224).eval()
    nprefix=getattr(m,"num_prefix_tokens",1)
    cfg=timm.data.resolve_model_data_config(m); cfg["input_size"]=(3,224,224)
    tf=timm.data.create_transform(**cfg,is_training=False)
    rows=list(csv.DictReader(open(f"{VAL}/labels.csv")))[:args.n]
    imgs,ys=[],[]
    for r in rows:
        p=f"{VAL}/images/{r['filename']}"
        if os.path.exists(p): imgs.append(tf(Image.open(p).convert('RGB'))); ys.append(int(r['label_idx']))
    Y=torch.tensor(ys); N=len(imgs)
    print(f"[setup] {name} N={N} reg-prefix={nprefix}",flush=True)

    def knn_acc(F_feat):
        Fn=F.normalize(F_feat,dim=-1); S=Fn@Fn.T; S.fill_diagonal_(-2)
        idx=S.topk(args.k,dim=1).indices
        pred=torch.mode(Y[idx],dim=1).values
        return 100*(pred==Y).float().mean().item()

    configs=[("full",0),("tome",16),("ours",16),("tome",20),("ours",20)]
    feats={}
    for strat,mpb in configs:
        key=f"{strat}_m{mpb}" if strat!="full" else "full"
        if key in feats: continue
        embs=[]
        with torch.no_grad():
            for x in imgs:
                t0=m._pos_embed(m.patch_embed(x.unsqueeze(0))).squeeze(0)
                embs.append(run_reduced(m,t0,strat,mpb,nprefix,k_protect=4))
        feats[key]=torch.stack(embs)
        npatch=t0.shape[0]-nprefix; final=nprefix+max(npatch-len(m.blocks)*mpb,1) if mpb else nprefix+npatch
        red=100*(1-final/(nprefix+npatch))
        print(f"  [{key}] reduce={red:.0f}%  kNN acc={knn_acc(feats[key]):.2f}%",flush=True)
    print("\n해석: 같은 압축률에서 ours_m16>tome_m16, ours_m20>tome_m20 이면 register 보호가 실제 분류정확도를 살림.",flush=True)

if __name__=="__main__": main()
