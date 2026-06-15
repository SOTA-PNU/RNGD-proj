"""Round 2: (A) re-test the round-1 failures EMBEDDED in a real compute graph,
(B) force the explicit _copy overloads, (C) re-judge matmul/pool 'mismatch' with
reduced-precision tolerance."""
import os, json, torch
import furiosa.torch
from furiosa.torch import CompileModule
from torch._decomp import core_aten_decompositions
torch.manual_seed(0)
DEV = torch.device("rngd", int(os.environ.get("RNGD_IDX","3")))
TABLE = dict(core_aten_decompositions())

def M(fn):
    class _M(torch.nn.Module):
        def forward(self, *a): return fn(*a)
    return _M().eval()
def f(*s): return torch.randn(*s)
def fp(*s): return torch.rand(*s)+0.5
def il(*s, hi=4): return torch.randint(0,hi,s,dtype=torch.int64)

def aten_norm(t):
    s=str(t); return s[5:] if s.startswith("aten.") else s
def flat(o):
    if isinstance(o,(tuple,list)):
        r=[]; [r.extend(flat(e)) for e in o]; return r
    return [o]
def run(label, fn, args, rtol=2e-2, atol=2e-2):
    rec={"label":label,"graph":[],"compile":None,"run":None,"abs":None,"rel":None,"err":""}
    try:
        m=M(fn)
        with torch.no_grad():
            ref=m(*[a.clone() for a in args])
            ep=torch.export.export(m,tuple(a.clone() for a in args)).run_decompositions(TABLE)
        rec["graph"]=sorted({aten_norm(n.target) for n in ep.graph.nodes if n.op=="call_function"})
        try:
            cm=CompileModule.from_exported(ep); rec["compile"]="OK"
        except Exception as e:
            rec["compile"]="FAIL"; rec["err"]=repr(e)[:160]
            print(f"{label:34s} compile=FAIL  graph={rec['graph']}  {rec['err'][:70]}"); return rec
        cm.to(DEV)
        with torch.no_grad():
            out=cm(*[a.to(DEV) for a in args], device=DEV)
        fa,fb=flat(ref),flat(out)
        worst_a=worst_r=0.0; ok=True
        for x,y in zip(fa,fb):
            x=x.to("cpu"); y=y.to("cpu")
            if x.shape!=y.shape: ok=False; worst_a=worst_r=float("nan"); break
            if x.dtype.is_floating_point:
                da=(x.float()-y.float()).abs()
                worst_a=max(worst_a, da.max().item())
                worst_r=max(worst_r, (da/(x.float().abs()+1e-6)).max().item())
                if not torch.allclose(x.float(),y.float(),rtol=rtol,atol=atol): ok=False
            else:
                if not torch.equal(x,y): ok=False; worst_a=float("inf")
        rec["run"]="OK" if ok else "MISMATCH"; rec["abs"]=worst_a; rec["rel"]=worst_r
    except Exception as e:
        rec["compile"]="EXPORT_FAIL"; rec["err"]=repr(e)[:160]
    print(f"{label:34s} compile={rec['compile']!s:4s} run={rec['run']!s:8s} abs={rec['abs']} rel={rec['rel']}  graph={rec['graph']}")
    return rec

R=[]
A=lambda x: torch.sigmoid(x)   # compute anchor that materializes
print("===== (A) round-1 failures, EMBEDDED in a compute graph =====")
R.append(run("isnan@embed",        lambda x: torch.isnan(A(x)).to(torch.float32)*2.0+x, [f(4,5)]))
R.append(run("cumsum@embed",       lambda x: torch.cumsum(A(x),1)+x, [f(4,5)]))
R.append(run("expand@embed",       lambda x: A(x).expand(3,4,5)+1.0, [f(1,4,5)]))
R.append(run("slice.Tensor@embed", lambda x: A(x)[:,1:4]+1.0, [f(4,5)]))
R.append(run("slice_scatter@embed",lambda x,y: torch.slice_scatter(A(x),A(y),1,1,4)+1.0, [f(4,5),f(4,3)]))
R.append(run("split_with_sizes@embed", lambda x: (lambda a,b:(a+1.0,b+1.0))(*torch.split(A(x),[2,3],1)), [f(4,5)]))
R.append(run("index.Tensor@embed", lambda x,i: A(x)[i]+1.0, [f(6,5), il(3,hi=6)]))
R.append(run("index_select@embed", lambda x,i: torch.index_select(A(x),0,i)+1.0, [f(6,5), il(3,hi=6)]))
R.append(run("gather@embed",       lambda x,i: torch.gather(A(x),1,i)+1.0, [f(4,5), il(4,2,hi=5)]))
R.append(run("constant_pad_nd@embed", lambda x: torch.nn.functional.pad(A(x),(1,2),value=0.0)+1.0, [f(4,5)]))
R.append(run("permute@embed",      lambda x: A(x).permute(1,0)+1.0, [f(4,5)]))
R.append(run("transpose@embed",    lambda x: A(x).transpose(0,1)+1.0, [f(4,5)]))
R.append(run("t@embed",            lambda x: A(x).t()+1.0, [f(4,5)]))
R.append(run("view@embed",         lambda x: A(x).view(2,10)+1.0, [f(4,5)]))

print("\n===== (B) explicit _copy overloads, embedded =====")
aten=torch.ops.aten
R.append(run("view_copy@explicit",   lambda x: aten.view_copy(A(x),[2,10])+1.0, [f(4,5)]))
R.append(run("permute_copy@explicit",lambda x: aten.permute_copy(A(x),[1,0])+1.0, [f(4,5)]))
R.append(run("transpose_copy@explicit",lambda x: aten.transpose_copy(A(x),0,1)+1.0, [f(4,5)]))
R.append(run("t_copy@explicit",      lambda x: aten.t_copy(A(x))+1.0, [f(4,5)]))
R.append(run("squeeze_copy.dim@explicit", lambda x: aten.squeeze_copy(A(x),1)+1.0, [f(4,1,5)]))
R.append(run("unsqueeze_copy@explicit",   lambda x: aten.unsqueeze_copy(A(x),1)+1.0, [f(4,5)]))
R.append(run("expand_copy@explicit", lambda x: aten.expand_copy(A(x),[3,4,5])+1.0, [f(1,4,5)]))
R.append(run("split_with_sizes_copy@explicit", lambda x:(lambda a,b:(a+1.,b+1.))(*aten.split_with_sizes_copy(A(x),[2,3],1)), [f(4,5)]))

print("\n===== (C) matmul/pool reduced-precision re-judge (rtol=2e-2) =====")
R.append(run("mm",  lambda a,b: torch.mm(a,b), [f(8,16),f(16,8)]))
R.append(run("bmm", lambda a,b: torch.bmm(a,b), [f(2,8,16),f(2,16,8)]))
R.append(run("convolution", lambda x,w,b: torch.nn.functional.conv2d(x,w,b,1,1), [f(1,3,16,16),f(8,3,3,3),f(8)]))
R.append(run("avg_pool2d", lambda x: torch.nn.functional.avg_pool2d(x,2), [f(1,3,16,16)]))
R.append(run("topk", lambda x: torch.topk(x,2,dim=1), [f(4,8)]))

json.dump(R, open("/home/jun/.claude/jobs/220196a8/tmp/results2.json","w"), indent=1)
print("\n=== ROUND2 SUMMARY ===")
print("compile FAIL:", [r["label"] for r in R if r["compile"] not in ("OK",)])
print("run not-OK  :", [r["label"] for r in R if r["compile"]=="OK" and r["run"]!="OK"])
