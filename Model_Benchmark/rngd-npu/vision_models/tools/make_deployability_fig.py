#!/usr/bin/env python3
"""논문 중심 figure: 닫힌 저정밀 NPU에서 비전 트랜스포머 배포가능성 지도 (4-panel, 실측값).
A 배포 taxonomy · B register 인과 · C 메커니즘(토큰노름) · D 해자(sim≠silicon)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

NAVY,RED,GREEN,GREY,AMBER="#1F3864","#C00000","#2E7D32","#888888","#E9A23B"
FIG="/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vision_models/results/figures"
import os; os.makedirs(FIG,exist_ok=True)
fig,ax=plt.subplots(2,2,figsize=(13.5,9))

# A 배포 taxonomy
a=ax[0,0]
rows=[("ViT-B/16 (supervised)","survive"),("ViT/DeiT tiny·small·base","survive"),
      ("DINOv2 (SSL, no-register)","nan"),("DINOv2 + registers","survive"),
      ("Swin / SwinV2 (window-attn)","compilefail")]
col={"survive":GREEN,"nan":RED,"compilefail":GREY}
lab={"survive":"runs (FP32-parity)","nan":"compiles, then NaN","compilefail":"compile rejected"}
for i,(name,st) in enumerate(rows):
    a.barh(i,1,color=col[st]); a.text(0.02,i,name,va="center",fontsize=11,color="white",weight="bold")
a.set_yticks([]); a.set_xticks([]); a.set_ylim(-0.6,len(rows)-0.4); a.invert_yaxis()
a.set_title("(A) Deployability map of ViTs on a closed low-precision NPU",fontsize=12,color=NAVY,weight="bold")
a.legend(handles=[Patch(color=col[k],label=lab[k]) for k in ["survive","nan","compilefail"]],
         loc="lower right",fontsize=9.5,framealpha=0.95)

# B register 인과
b=ax[0,1]
bars=b.bar(["DINOv2\n(no register)","DINOv2\n+ 4 registers"],[0,1.0],color=[RED,GREEN],width=0.55)
b.text(0,0.05,"NaN\n(6/6 dead)",ha="center",va="bottom",fontsize=12,color=RED,weight="bold")
b.text(1,1.0,"cosine 1.0000\n(perfect)",ha="center",va="bottom",fontsize=11,color=GREEN)
b.set_ylim(0,1.25); b.set_ylabel("NPU vs FP32 embedding cosine",fontsize=11)
b.set_title("(B) Causal proof: registers are the only difference",fontsize=12,color=NAVY,weight="bold")
b.grid(True,axis="y",alpha=0.3)

# C 메커니즘 토큰노름
c=ax[1,0]
blk=[0,3,6,9,11]
plain=[3.3,2.8,7.5,556.0,240.7]; reg=[6.7,27.5,28.0,219.9,272.1]
c.plot(blk,plain,"o-",color=RED,lw=2.4,ms=8,label="DINOv2 (no register)")
c.plot(blk,reg,"s--",color=GREEN,lw=2.2,ms=8,label="DINOv2 + registers")
c.set_yscale("log"); c.set_xlabel("transformer block",fontsize=11)
c.set_ylabel("max token L2-norm (log)",fontsize=11)
c.annotate("artifact token\n556 = 38.7x median\n(overflows low-precision)",xy=(9,556),xytext=(4.2,560),
           fontsize=9.5,color=RED,arrowprops=dict(arrowstyle="->",color=RED))
c.set_title("(C) Mechanism: an extreme artifact token (absorbed by registers)",fontsize=12,color=NAVY,weight="bold")
c.legend(fontsize=10); c.grid(True,alpha=0.3,which="both")

# D 해자 sim != silicon
d=ax[1,1]
xs=["real NPU\n(TCP silicon)","sim bf16\n(CPU)","sim fp16\n(CPU)"]
nan=[6,0,0]
bars=d.bar(xs,nan,color=[RED,GREEN,GREEN],width=0.55)
for bar,v,cos in zip(bars,nan,[None,0.9998,1.0]):
    d.text(bar.get_x()+bar.get_width()/2,v+0.1,f"{v}/6 NaN"+("" if cos is None else f"\ncos {cos}"),
           ha="center",va="bottom",fontsize=10.5,color=RED if v else GREEN,weight="bold")
d.set_ylim(0,7); d.set_ylabel("images producing NaN (out of 6)",fontsize=11)
d.set_title("(D) Moat: simulation cannot reproduce the silicon NaN",fontsize=12,color=NAVY,weight="bold")
d.grid(True,axis="y",alpha=0.3)

fig.suptitle("Compiles, Then Dies — Vision-Transformer deployability on a closed reduced-precision NPU (real-silicon measurement)",
             fontsize=13.5,color=NAVY,weight="bold",y=1.0)
fig.tight_layout(rect=[0,0,1,0.97])
out=f"{FIG}/deployability_map.png"; fig.savefig(out,dpi=150,bbox_inches="tight")
print("saved:",out)
