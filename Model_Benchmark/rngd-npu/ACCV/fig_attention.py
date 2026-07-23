#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig_attention: CLS 어텐션이 각 패치에 몰리는 정도를 색블록 그리드로 표시.
규칙: 그림엔 라벨만(설명은 캡션/본문). 이미지 오버레이·register 박스 없음.
색 기준: 많이 몰림=빨강, 중간=노랑, 적게=파랑(RdYlBu_r). 네 패널 공통 스케일."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NPZ="/home/jun/.claude/jobs/02d85653/tmp/attn_maps.npz"
d=np.load(NPZ)
panels=[("Uncompressed","full"),("ToMe","tome"),("PiToMe","pitome"),("Ours","ours")]
titlecol={"Uncompressed":"#222222","ToMe":"#D55E00","PiToMe":"#7d3ac1","Ours":"#0072B2"}

# 공통 스케일(패널 간 색 의미 통일). 로그로 대비 강조.
grids={k:np.log1p(d[k]*1000.0) for _,k in panels}
vmax=max(g.max() for g in grids.values()); vmin=min(g.min() for g in grids.values())

fig,axs=plt.subplots(1,4,figsize=(12,3.4))
for ax,(name,key) in zip(axs,panels):
    im=ax.imshow(grids[key],cmap="RdYlBu_r",vmin=vmin,vmax=vmax,interpolation="nearest")
    ax.set_title(name,fontsize=15,fontweight="bold",color=titlecol[name],pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor("#cccccc"); s.set_linewidth(1.0)

# 공통 컬러바(라벨만: high/low)
cbar=fig.colorbar(im,ax=axs,fraction=0.018,pad=0.015,ticks=[vmin,(vmin+vmax)/2,vmax])
cbar.ax.set_yticklabels(["low","mid","high"],fontsize=11)
cbar.outline.set_edgecolor("#cccccc")
plt.savefig("fig_attention.pdf",bbox_inches="tight"); plt.savefig("fig_attention.png",dpi=150,bbox_inches="tight")
print("saved fig_attention.pdf/png")
