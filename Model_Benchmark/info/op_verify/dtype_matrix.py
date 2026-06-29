"""dtype 매트릭스 — degeneracy 제거판: 모든 op를 OP(x+x) 로 감싸 실연산 그래프로 측정.
add 는 지원 dtype 전부에서 컴파일되므로 dtype 게이트를 바꾸지 않으면서 단독그래프 문제만 제거."""
import os, json, torch, furiosa.torch
from furiosa.torch import CompileModule
from torch._decomp import core_aten_decompositions
torch.manual_seed(0)
TABLE = dict(core_aten_decompositions())
TRY_RUN = os.environ.get("TRY_RUN","0")=="1"
DEV = torch.device("rngd", int(os.environ.get("RNGD_IDX","3")))
DT = [("float64",torch.float64),("float32",torch.float32),("float16",torch.float16),
      ("bfloat16",torch.bfloat16),("int64",torch.int64),("int32",torch.int32),
      ("int16",torch.int16),("int8",torch.int8),("uint16",torch.uint16),("uint32",torch.uint32)]
def mk(dt, shape=(4,8)):
    if dt.is_floating_point: return torch.randn(*shape).to(dt)
    if dt in (torch.uint8,torch.uint16,torch.uint32): return torch.randint(0,5,shape,dtype=dt)
    return torch.randint(-3,4,shape,dtype=dt)
class Mod(torch.nn.Module):
    def __init__(s,fn): super().__init__(); s.fn=fn
    def forward(s,*a): return s.fn(*a)
def run_cell(fn, dt, mkargs):
    try:
        args=mkargs(dt); Mod(fn).eval()(*[a.clone() for a in args])   # eager 게이트
    except Exception: return "-"
    try:
        ep=torch.export.export(Mod(fn).eval(),tuple(a.clone() for a in args)).run_decompositions(TABLE)
        cm=CompileModule.from_exported(ep)
    except Exception as e:
        return "x" if "UnsupportedOp" in repr(e) else "E"
    if not TRY_RUN: return "O"
    try:
        cm.to(DEV); cm(*[a.to(DEV) for a in args],device=DEV); return "R"
    except Exception as e:
        return "O*" if ("EBUSY" in repr(e) or "busy" in repr(e)) else "Erun"
# embed: 입력을 x+x 로 한 번 통과시켜 실연산 anchor 부여
A = lambda x: x + x
one = lambda mkf: (lambda dt: (mkf(dt),))
OPS = [
  ("add",       lambda x: A(x)+A(x),                 one(lambda dt: mk(dt))),
  ("sub",       lambda x: A(x)-A(x),                 one(lambda dt: mk(dt))),
  ("mul",       lambda x: A(x)*A(x),                 one(lambda dt: mk(dt))),
  ("div",       lambda x: A(x)/A(x),                 one(lambda dt: mk(dt))),
  ("pow",       lambda x: torch.pow(A(x),2),         one(lambda dt: mk(dt))),
  ("clamp",     lambda x: torch.clamp(A(x),0,3),     one(lambda dt: mk(dt))),
  ("abs",       lambda x: torch.abs(A(x)),           one(lambda dt: mk(dt))),
  ("neg",       lambda x: torch.neg(A(x)),           one(lambda dt: mk(dt))),
  ("exp",       lambda x: torch.exp(A(x)),           one(lambda dt: mk(dt))),
  ("log",       lambda x: torch.log(torch.abs(A(x))+1), one(lambda dt: mk(dt))),
  ("sqrt",      lambda x: torch.sqrt(torch.abs(A(x))),  one(lambda dt: mk(dt))),
  ("rsqrt",     lambda x: torch.rsqrt(torch.abs(A(x))+1),one(lambda dt: mk(dt))),
  ("sin",       lambda x: torch.sin(A(x)),           one(lambda dt: mk(dt))),
  ("erf",       lambda x: torch.erf(A(x)),           one(lambda dt: mk(dt))),
  ("sigmoid",   lambda x: torch.sigmoid(A(x)),       one(lambda dt: mk(dt))),
  ("tanh",      lambda x: torch.tanh(A(x)),          one(lambda dt: mk(dt))),
  ("softmax",   lambda x: torch.softmax(A(x),-1),    one(lambda dt: mk(dt))),
  ("relu",      lambda x: torch.relu(A(x)),          one(lambda dt: mk(dt))),
  ("mm",        lambda a,b: torch.mm(A(a),A(b)),     lambda dt: (mk(dt,(4,8)),mk(dt,(8,4)))),
  ("conv2d",    lambda x,w: torch.nn.functional.conv2d(A(x),A(w),None,1,1), lambda dt: (mk(dt,(1,3,8,8)),mk(dt,(4,3,3,3)))),
  ("eq",        lambda x: A(x)==A(x),                one(lambda dt: mk(dt))),
  ("lt",        lambda x: A(x)<A(x),                 one(lambda dt: mk(dt))),
  ("maximum",   lambda x: torch.maximum(A(x),A(x)),  one(lambda dt: mk(dt))),
  ("logical_and",lambda x: torch.logical_and(A(x),A(x)), one(lambda dt: mk(dt))),
  ("bitwise_and",lambda x: torch.bitwise_and(A(x),A(x)), one(lambda dt: mk(dt))),
  ("sum",       lambda x: A(x).sum(dim=1),           one(lambda dt: mk(dt))),
  ("mean",      lambda x: torch.mean(A(x),dim=1),    one(lambda dt: mk(dt))),
  ("max.dim",   lambda x: torch.max(A(x),dim=1),     one(lambda dt: mk(dt))),
  ("argmax",    lambda x: torch.argmax(A(x),dim=1),  one(lambda dt: mk(dt))),
  ("cumsum",    lambda x: torch.cumsum(A(x),dim=1),  one(lambda dt: mk(dt))),
  ("view",      lambda x: A(x).view(-1),             one(lambda dt: mk(dt))),
  ("cat",       lambda x: torch.cat([A(x),A(x)],1),  one(lambda dt: mk(dt))),
  ("permute",   lambda x: A(x).permute(1,0),         one(lambda dt: mk(dt))),
  ("slice",     lambda x: A(x)[:,1:5],               one(lambda dt: mk(dt))),
  ("where",     lambda x: torch.where(A(x)>0,A(x),A(x)), one(lambda dt: mk(dt))),
  ("clone",     lambda x: A(x).clone(),              one(lambda dt: mk(dt))),
  ("to_float32",lambda x: A(x).to(torch.float32),    one(lambda dt: mk(dt))),
  ("full_like", lambda x: torch.full_like(A(x),1),   one(lambda dt: mk(dt))),
]
rows=[]
hdr=f"{'op':12s}"+"".join(f"{n[:8]:>9s}" for n,_ in DT)
print(hdr); print("-"*len(hdr))
for name,fn,mkargs in OPS:
    c={dn:run_cell(fn,dt,mkargs) for dn,dt in DT}
    rows.append({"op":name,**c})
    print(f"{name:12s}"+"".join(f"{c[dn]:>9s}" for dn,_ in DT))
json.dump({"dtypes":[d for d,_ in DT],"rows":rows,"try_run":TRY_RUN},
          open("/home/jun/.claude/jobs/220196a8/tmp/dtype_matrix_embed.json","w"),indent=1)
print("\n범례: O=컴파일OK R=NPU실행OK O*=컴파일OK(실행 EBUSY) x=unsup -=N/A(torch미지원) E=기타에러")
