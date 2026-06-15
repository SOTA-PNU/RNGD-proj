"""Round 3: harden the 7 genuine failures with ALTERNATIVE embeddings, and test
the not-yet-embedded max_pool2d_with_indices and copy. If an op fails across
multiple realistic graph shapes, it is genuinely unsupported by the EDF backend."""
import os, json, torch
import furiosa.torch
from furiosa.torch import CompileModule
from torch._decomp import core_aten_decompositions
torch.manual_seed(0)
DEV=torch.device("rngd", int(os.environ.get("RNGD_IDX","3")))
TABLE=dict(core_aten_decompositions())
def M(fn):
    class _M(torch.nn.Module):
        def forward(self,*a): return fn(*a)
    return _M().eval()
def f(*s): return torch.randn(*s)
def il(*s,hi=4): return torch.randint(0,hi,s,dtype=torch.int64)
def norm(t):
    s=str(t); return s[5:] if s.startswith("aten.") else s
def go(label, fn, args):
    try:
        m=M(fn)
        with torch.no_grad():
            ep=torch.export.export(m,tuple(a.clone() for a in args)).run_decompositions(TABLE)
        g=sorted({norm(n.target) for n in ep.graph.nodes if n.op=="call_function"})
        try:
            cm=CompileModule.from_exported(ep)
        except Exception as e:
            print(f"{label:42s} compile=FAIL  {repr(e)[:60]}  graph={g}"); return
        cm.to(DEV)
        with torch.no_grad(): out=cm(*[a.to(DEV) for a in args],device=DEV)
        print(f"{label:42s} compile+run=OK  graph={g}")
    except Exception as e:
        print(f"{label:42s} EXPORT/RUN_FAIL  {repr(e)[:60]}")

W=torch.nn.Parameter(torch.randn(10,5), requires_grad=False)
print("--- the 7 suspected-genuine failures, alternative graph shapes ---")
go("isnan->where",        lambda x: torch.where(torch.isnan(x), torch.zeros_like(x), x*2.0), [f(4,5)])
go("isnan->sum",          lambda x: torch.isnan(torch.sigmoid(x)).sum().float(), [f(4,5)])
go("cumsum(dim0)",        lambda x: torch.cumsum(x,0), [f(4,5)])
go("constant_pad_nd(4D)", lambda x: torch.nn.functional.pad(torch.sigmoid(x),(1,1,1,1))*2.0, [f(1,3,8,8)])
go("constant_pad_nd(1D)", lambda x: torch.constant_pad_nd(x,[1,2],0.0), [f(4,5)])
go("index_select(emb)",   lambda i: torch.index_select(W,0,i)+1.0, [il(3,hi=10)])
go("index.Tensor(emb)",   lambda i: W[i]+1.0, [il(3,hi=10)])
go("embedding(nn)",       (lambda emb: (lambda i: emb(i)))(torch.nn.Embedding(10,5)), [il(1,3,hi=10)])
go("gather(dim1)",        lambda x,i: torch.gather(torch.sigmoid(x),1,i)+1.0, [f(4,5), il(4,2,hi=5)])
go("slice_scatter(plain)",lambda x,y: torch.slice_scatter(x,y,1,1,4), [f(4,5),f(4,3)])

print("\n--- not-yet-embedded: max_pool2d_with_indices, copy ---")
go("max_pool2d_with_indices@embed", lambda x:(lambda v,i:(v+1.0,i))(*torch.nn.functional.max_pool2d(torch.sigmoid(x),2,return_indices=True)), [f(1,3,16,16)])
go("max_pool2d(no_indices)@embed",  lambda x: torch.nn.functional.max_pool2d(torch.sigmoid(x),2)+1.0, [f(1,3,16,16)])
go("copy_@embed",         lambda x,y: (torch.sigmoid(x)).clone().copy_(torch.sigmoid(y))+1.0, [f(4,5),f(4,5)])
