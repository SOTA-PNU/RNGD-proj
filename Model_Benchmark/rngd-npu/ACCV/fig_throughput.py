#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig_throughput: 토큰 축소율에 따른 처리량(im/s). 세 방법이 거의 겹침 = 레지스터 보호가 속도를 안 깎음.
규칙: 그림엔 라벨만. 설명은 캡션/본문."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

comp=[0,36.8,55.2,73.6,82.8,92.0]
tome=[355,405,451,503,543,574]; pitome=[351,396,440,493,531,562]; ours=[350,406,451,504,544,575]
OURS="#0072B2"; TOME="#D55E00"; PITOME="#009E73"

fig,ax=plt.subplots(figsize=(5.2,4.0))
ax.plot(comp,tome,"--s",color=TOME,lw=2.0,ms=6,mec="white",mew=0.8,label="ToMe")
ax.plot(comp,pitome,":D",color=PITOME,lw=2.0,ms=6,mec="white",mew=0.8,label="PiToMe")
ax.plot(comp,ours,"-o",color=OURS,lw=2.4,ms=6.5,mec="white",mew=0.8,label="Ours")
ax.set_xlabel("Token reduction (%)",fontsize=13)
ax.set_ylabel("Images / second",fontsize=13)
ax.set_xticks([0,37,55,74,83,92])
ax.grid(axis="y",alpha=.25)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
ax.legend(frameon=False,fontsize=12,loc="upper left")
plt.tight_layout()
plt.savefig("fig_throughput.pdf",bbox_inches="tight"); plt.savefig("fig_throughput.png",dpi=150,bbox_inches="tight")
print("saved fig_throughput.pdf/png")
