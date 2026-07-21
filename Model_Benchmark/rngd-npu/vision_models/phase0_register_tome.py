#!/usr/bin/env python3
"""Phase 0 가설검증(CPU): 토큰 병합 시 '무엇을 보호하느냐'만 바꿔 비교.
같은 bipartite soft matching 병합 알고리즘에서:
  - tome    : CLS(+구조적 prefix)만 보호 (표준 ToMe류 — 큰 토큰 안 지킴)
  - ours    : CLS+register + 큰(고노름) 토큰 보호
  - random  : CLS + 무작위 k개 보호 (대조군: '아무거나 보호'와 다른가)
지표: 병합 후 최종 임베딩이 '병합 안 한 full 모델' 임베딩과 얼마나 같은가(cosine, 높을수록 보존).
가설: ours > tome, ours > random — 특히 압축률 높을수록. 큰 토큰 강한 DINOv2서 격차 큼.
사용: python phase0_register_tome.py"""
import warnings; warnings.filterwarnings("ignore")
import torch, timm
import torch.nn.functional as F
from PIL import Image

IMG="/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/test_images"
IMGS=["brambling.jpg","tabby_cat.jpg","convertible.jpg","orange.jpg","dog1.jpg","astronaut.jpg"]

def merge_step(x, m, protected):
    """x:[T,D] 한 이미지. m개 토큰 병합. protected: 보호할 인덱스(LongTensor). 반환 새 x."""
    T,D=x.shape
    prot=torch.zeros(T,dtype=torch.bool); prot[protected]=True
    cand=(~prot).nonzero(as_tuple=True)[0]
    if m<=0 or len(cand)<=m+1: return x
    A=cand[0::2]; Bs=cand[1::2]
    if len(A)==0 or len(Bs)==0: return x
    xn=F.normalize(x,dim=-1)
    sim=xn[A]@xn[Bs].T
    best_sim,best_j=sim.max(dim=1)
    mm=min(m,len(A))
    merge_A=best_sim.argsort(descending=True)[:mm]
    keepA=torch.ones(len(A),dtype=torch.bool); keepA[merge_A]=False
    Bsum=x[Bs].clone(); Bcnt=torch.ones(len(Bs))
    for a in merge_A.tolist():
        j=best_j[a].item(); Bsum[j]=Bsum[j]+x[A[a]]; Bcnt[j]+=1
    Bmerged=Bsum/Bcnt[:,None]
    return torch.cat([x[protected], Bmerged, x[A[keepA]]],dim=0)

@torch.no_grad()
def run_reduced(m_model, tokens0, strategy, m_per_blk, nprefix, k_protect):
    """tokens0:[T,D] (pos-embed 더해진 초기). 블록마다 병합. 최종 CLS 임베딩 반환."""
    x=tokens0
    for blk in m_model.blocks:
        x=blk(x.unsqueeze(0)).squeeze(0)   # [T,D]
        if m_per_blk>0:
            prefix=list(range(nprefix))
            if strategy=="tome":
                prot=torch.tensor(prefix[:1])           # CLS만 (구조 무지)
            elif strategy=="ours":
                # prefix(cls+reg) + 비-prefix 중 고노름 top-k
                nb=x[nprefix:].norm(dim=-1)
                hi=(nprefix+torch.topk(nb,min(k_protect,len(nb))).indices)
                prot=torch.cat([torch.tensor(prefix),hi]).unique()
            elif strategy=="random":
                nonp=torch.arange(1,x.shape[0])
                ridx=nonp[torch.randperm(len(nonp))[:k_protect]]
                prot=torch.cat([torch.tensor([0]),ridx]).unique()
            else:
                prot=torch.tensor(prefix[:1])
            x=merge_step(x,m_per_blk,prot)
    x=m_model.norm(x.unsqueeze(0)).squeeze(0)
    return x[0]   # CLS

def run_model(name):
    torch.manual_seed(0)
    m=timm.create_model(name,pretrained=True,num_classes=0,img_size=224).eval()
    cfg=timm.data.resolve_model_data_config(m); cfg["input_size"]=(3,224,224)
    tf=timm.data.create_transform(**cfg,is_training=False)
    nprefix=getattr(m,"num_prefix_tokens",1)
    print(f"\n=== {name}  (prefix tokens={nprefix}) ===")
    print(f"{'reduce%':>7} {'tome':>8} {'ours':>8} {'random':>8} {'full':>6}")
    xs=[tf(Image.open(f"{IMG}/{f}").convert('RGB')) for f in IMGS]
    for m_per_blk in [8,16,20]:
        accum={"tome":[],"ours":[],"random":[]}
        npatch=None
        for x in xs:
            with torch.no_grad():
                t0=m._pos_embed(m.patch_embed(x.unsqueeze(0))).squeeze(0)  # [T,D]
                npatch=t0.shape[0]-nprefix
                ref=run_reduced(m,t0,"full",0,nprefix,0)
                for strat in accum:
                    emb=run_reduced(m,t0,strat,m_per_blk,nprefix,k_protect=4)
                    accum[strat].append(F.cosine_similarity(emb,ref,dim=0).item())
        nblk=len(m.blocks); final=nprefix+max(npatch-nblk*m_per_blk,1)
        red=100*(1-final/(nprefix+npatch))
        print(f"{red:6.1f}% {sum(accum['tome'])/6:8.4f} {sum(accum['ours'])/6:8.4f} {sum(accum['random'])/6:8.4f} {1.0:6.2f}")
    print("  (값=full 대비 임베딩 cosine. ours>tome 이면 가설 지지)")

for mn in ["vit_base_patch14_dinov2.lvd142m","vit_base_patch14_reg4_dinov2.lvd142m","vit_base_patch16_clip_224.openai"]:
    run_model(mn)
print("\n해석: ours가 tome보다 일관되게 높고(특히 고압축), random보다 높으면 → '큰 토큰을 콕 보호하는 게 이득' 입증 → 주제 성립.")
