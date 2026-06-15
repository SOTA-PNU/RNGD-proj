"""Reconcile contradictions between my harness and the adversarial workflow.
Run each contested construction on the SAME card, report compile+run truth."""
import torch, furiosa.torch
from furiosa.torch import CompileModule
from torch._decomp import core_aten_decompositions
import torch.nn.functional as F
torch.manual_seed(0)
TABLE=dict(core_aten_decompositions()); DEV=torch.device("rngd",3)
def t(label, fn, args):
    class M(torch.nn.Module):
        def forward(self,*a): return fn(*a)
    m=M().eval()
    try:
        with torch.no_grad():
            ep=torch.export.export(m,tuple(a.clone() for a in args)).run_decompositions(TABLE)
        g=sorted({(str(n.target)[5:] if str(n.target).startswith('aten.') else str(n.target)) for n in ep.graph.nodes if n.op=='call_function'})
        try: cm=CompileModule.from_exported(ep)
        except Exception as e:
            print(f"  {label:46s} COMPILE_FAIL  graph={g}"); return
        cm.to(DEV)
        with torch.no_grad(): out=cm(*[a.to(DEV) for a in args],device=DEV)
        ref=m(*[a.clone() for a in args])
        o=out[0].to('cpu') if isinstance(out,(tuple,list)) else out.to('cpu')
        r=ref[0] if isinstance(ref,(tuple,list)) else ref
        d=(o.float()-r.float()).abs().max().item() if o.shape==r.shape else float('nan')
        print(f"  {label:46s} COMPILE+RUN OK  maxdiff={d:.2e}  graph={g}")
    except Exception as e:
        print(f"  {label:46s} EXPORT/RUN_FAIL  {repr(e)[:60]}")

sig=torch.sigmoid
print("== max_pool2d VALUES-ONLY, kernel=2, my size vs agent size ==")
t("maxpool k2 16x16 sigmoid-then-pool (MY case)", lambda x: F.max_pool2d(sig(x),2)+1.0, [torch.randn(1,3,16,16)])
t("maxpool k2 8x8 sigmoid-then-pool (AGENT size)", lambda x: F.max_pool2d(sig(x),2)+1.0, [torch.randn(1,3,8,8)])
t("maxpool k2 16x16 plain input pool",           lambda x: F.max_pool2d(x,2),        [torch.randn(1,3,16,16)])
t("maxpool k2 8x8 plain input pool",             lambda x: F.max_pool2d(x,2),        [torch.randn(1,3,8,8)])
t("maxpool k3s2p1 56x56 (resnet stem)",          lambda x: F.max_pool2d(x,3,2,1),    [torch.randn(1,8,56,56)])

print("== index.Tensor: input vs intermediate ==")
t("x[idx] INPUT standalone (8,4)",   lambda x,i: x[i],          [torch.randn(8,4), torch.tensor([0,2,5,7])])
t("x[idx] INPUT standalone (6,5)",   lambda x,i: x[i],          [torch.randn(6,5), torch.tensor([0,2,4])])
t("x[idx] INPUT embedded +1",        lambda x,i: x[i]+1.0,      [torch.randn(8,4), torch.tensor([0,2,5,7])])
t("sigmoid(x)[idx] INTERMEDIATE +1 (MY case)", lambda x,i: sig(x)[i]+1.0, [torch.randn(6,5), torch.tensor([0,2,4])])
t("sigmoid(x)[idx] INTERMEDIATE +1 (agent size)", lambda x,i: sig(x)[i]+1.0, [torch.randn(8,4), torch.tensor([0,2,5,7])])

print("== index_select dim0: input vs intermediate ==")
t("index_select(x,0,idx) INPUT",          lambda x,i: torch.index_select(x,0,i),     [torch.randn(8,16), torch.tensor([0,2,5,7])])
t("sigmoid then index_select(.,0,idx) (MY)", lambda x,i: torch.index_select(sig(x),0,i)+1.0, [torch.randn(6,5), torch.tensor([0,2,4])])
t("index_select(sigmoid(x),0,idx) agent-size", lambda x,i: torch.index_select(sig(x),0,i)+1.0, [torch.randn(8,16), torch.tensor([0,2,5,7])])

print("== gather: 1D-effective vs general rank2 ==")
t("gather 1D dim0 (agent OK)",     lambda x,i: torch.gather(x,0,i),       [torch.randn(10), torch.tensor([4,0,3,1,2,0])])
t("gather(sigmoid 1D) dim0 +2",    lambda x,i: torch.gather(sig(x),0,i)+2.0, [torch.randn(10), torch.tensor([4,0,3,1,2,0])])
t("gather rank2 dim1 (MY case)",   lambda x,i: torch.gather(x,1,i),       [torch.randn(4,5), torch.randint(0,5,(4,2))])
