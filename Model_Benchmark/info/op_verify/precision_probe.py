"""matmul/conv 정밀도 측정: elementwise는 FP32-exact, matmul 계열만 ~0.23% 상대오차."""
import torch, furiosa.torch
from furiosa.torch import CompileModule
from torch._decomp import core_aten_decompositions
TABLE=dict(core_aten_decompositions()); DEV=torch.device("rngd",3)
def compile_run(fn,args):
    class M(torch.nn.Module):
        def forward(self,*a): return fn(*a)
    m=M().eval()
    with torch.no_grad():
        ep=torch.export.export(m,tuple(args)).run_decompositions(TABLE)
        cm=CompileModule.from_exported(ep); cm.to(DEV)
        out=cm(*[a.to(DEV) for a in args],device=DEV)
    return m(*args), out.to("cpu")
def stats(name,ref,out):
    ref=ref.flatten().float(); out=out.flatten().float()
    cos=torch.nn.functional.cosine_similarity(ref,out,dim=0).item()
    abse=(ref-out).abs().max().item()
    rel=((ref-out).norm()/ref.norm()).item()
    print(f"{name:22s} max_abs={abse:.4g}  rel_l2={rel:.4g}  cos={cos:.7f}")
for n in [8,64,256]:
    a=torch.randn(n,n); b=torch.randn(n,n)
    r,o=compile_run(lambda x,y:torch.mm(x,y),[a,b]); stats(f"mm {n}x{n}",r,o)
x=torch.randn(1,3,32,32); w=torch.randn(8,3,3,3); bs=torch.randn(8)
r,o=compile_run(lambda x,w,b:torch.nn.functional.conv2d(x,w,b,1,1),[x,w,bs]); stats("conv 3x3",r,o)
x=torch.randn(256,256)
r,o=compile_run(lambda x:torch.sigmoid(x)*2.0,[x]); stats("sigmoid*2 (elemwise)",r,o)
