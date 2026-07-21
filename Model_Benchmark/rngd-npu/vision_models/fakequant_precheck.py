#!/usr/bin/env python3
"""주제 핵심 사전검증(CPU fake-quant, GPU 투자 전): 활성을 INT8/4/3/2로 흉내내 양자화할 때
(a) 표준 per-tensor 단일스케일 vs (b) per-token 스케일 vs (c) 상위 k개 outlier 토큰만 FP 유지
→ DINOv2/CLIP 임베딩이 FP32 대비 얼마나 보존되나(cosine).
기대: 표준은 저비트서 붕괴, per-token/keep-outlier가 회복 → '큰 토큰이 범인 + 따로 다루면 복구' 입증.
사용: python fakequant_precheck.py"""
import warnings; warnings.filterwarnings("ignore")
import torch, timm
from PIL import Image
import torch.nn.functional as Fn

IMG="/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/test_images"
IMGS=["brambling.jpg","tabby_cat.jpg","convertible.jpg","orange.jpg","dog1.jpg","astronaut.jpg"]

def fq_per_tensor(x, bits):
    # 대칭 per-tensor: 전체 한 스케일
    qmax=2**(bits-1)-1
    s=x.abs().amax()/qmax
    return torch.round(x/s).clamp(-qmax-1,qmax)*s

def fq_per_token(x, bits):
    # per-token: 토큰(dim=-2)마다 스케일  x:[B,T,D]
    qmax=2**(bits-1)-1
    s=x.abs().amax(dim=-1,keepdim=True)/qmax
    s=torch.clamp(s,min=1e-8)
    return torch.round(x/s).clamp(-qmax-1,qmax)*s

def fq_keep_outlier(x, bits, k=4):
    # 상위 k개 큰 토큰만 FP 유지, 나머지는 per-tensor 양자화
    n=x.norm(dim=-1)  # [B,T]
    out=x.clone()
    for b in range(x.shape[0]):
        idx=torch.topk(n[b],k).indices
        mask=torch.ones(x.shape[1],dtype=torch.bool); mask[idx]=False
        out[b,mask]=fq_per_tensor(x[b,mask],bits)
        # 아웃라이어 토큰은 그대로(FP)
    return out

def run(model_name):
    m=timm.create_model(model_name,pretrained=True,num_classes=0,img_size=224).eval()
    cfg=timm.data.resolve_model_data_config(m); cfg["input_size"]=(3,224,224)
    tf=timm.data.create_transform(**cfg,is_training=False)
    xs=torch.stack([tf(Image.open(f"{IMG}/{f}").convert("RGB")) for f in IMGS])
    # 각 블록 출력(활성)에 fake-quant 삽입
    def hookify(fn):
        hs=[]
        for blk in m.blocks:
            o=blk.forward
            def mk(o):
                def f(x,*a,**k):
                    out=o(x,*a,**k)
                    t=out if isinstance(out,torch.Tensor) else out[0]
                    t=fn(t)
                    return t if isinstance(out,torch.Tensor) else (t,*out[1:])
                return f
            blk.forward=mk(o); hs.append((blk,o))
        return hs
    def restore(hs):
        for blk,o in hs: blk.forward=o
    with torch.no_grad():
        ref=m(xs).float()
    print(f"\n=== {model_name} ===")
    print(f"{'bits':>4} | {'per-tensor':>11} {'per-token':>11} {'keep-outlier':>12}")
    for bits in [8,4,3,2]:
        row={}
        for name,fn in [("pt",lambda x:fq_per_tensor(x,bits)),
                        ("ptok",lambda x:fq_per_token(x,bits)),
                        ("ko",lambda x:fq_keep_outlier(x,bits,4))]:
            hs=hookify(fn)
            with torch.no_grad(): out=m(xs).float()
            restore(hs)
            cos=Fn.cosine_similarity(out,ref,dim=-1).mean().item() if not torch.isnan(out).any() else float('nan')
            row[name]=cos
        print(f"{bits:>4} | {row['pt']:>11.4f} {row['ptok']:>11.4f} {row['ko']:>12.4f}")

for mn in ["vit_base_patch14_dinov2.lvd142m","vit_base_patch16_clip_224.openai"]:
    run(mn)
print("\n해석: per-tensor가 저비트(3/2)서 붕괴하고 per-token/keep-outlier가 살아나면 → '큰 토큰이 범인+따로다루면복구' 입증.")
