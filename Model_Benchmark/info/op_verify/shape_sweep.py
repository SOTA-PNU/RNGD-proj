"""gather/index 계열의 모양 의존성 측정: 맨 안쪽(feature) 차원 정렬에 따라 컴파일 성패가 갈림."""
import torch, furiosa.torch
from furiosa.torch import CompileModule
from torch._decomp import core_aten_decompositions
TABLE=dict(core_aten_decompositions()); DEV=torch.device("rngd",3)
def ok(fn,args):
    class M(torch.nn.Module):
        def forward(self,*a): return fn(*a)
    try:
        with torch.no_grad():
            ep=torch.export.export(M().eval(),tuple(a.clone() for a in args)).run_decompositions(TABLE)
            cm=CompileModule.from_exported(ep); cm.to(DEV); cm(*[a.to(DEV) for a in args],device=DEV)
        return "OK"
    except Exception: return "FAIL"

print("index_select(x,0,idx): rows sweep (cols=16)")
for rows in [4,6,7,8,9,15,16,17,24,32]:
    print(f"  rows={rows:3d}  {ok(lambda x,i: torch.index_select(x,0,i), [torch.randn(rows,16), torch.tensor([0,1,2])])}")
print("index_select(x,0,idx): rows=8, cols sweep")
for c in [3,5,7,8,15,16,17]:
    print(f"  cols={c:3d}  {ok(lambda x,i: torch.index_select(x,0,i), [torch.randn(8,c), torch.tensor([0,1,2])])}")
print("index.Tensor x[idx]: rows=6, cols sweep")
for c in [3,4,5,7,8,12,16]:
    print(f"  cols={c:3d}  {ok(lambda x,i:x[i],[torch.randn(6,c),torch.tensor([0,2,4])])}")
print("gather dim1 x(8,C) idx(8,2): C sweep")
for c in [4,5,8,15,16]:
    print(f"  C={c:3d}  {ok(lambda x,i:torch.gather(x,1,i),[torch.randn(8,c),torch.randint(0,c,(8,2))])}")
print("gather dim0 x(R,8) idx(2,8): R sweep")
for R in [4,6,8]:
    print(f"  R={R:3d}  {ok(lambda x,i:torch.gather(x,0,i),[torch.randn(R,8),torch.randint(0,R,(2,8))])}")
