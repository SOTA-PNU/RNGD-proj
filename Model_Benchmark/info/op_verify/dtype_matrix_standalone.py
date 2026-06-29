"""op x dtype 매트릭스: 각 op를 10개 dtype으로 RNGD 컴파일(+가능시 실행) 시도.
셀 의미:  O = 컴파일 성공(EDF lower OK) ·  x = unsup(UnsupportedOpError) ·
          - = N/A(torch eager 자체가 그 dtype에서 op 미지원) ·  R = NPU 실행까지 OK ·  E = 기타 에러"""
import os, json, torch, furiosa.torch
from furiosa.torch import CompileModule
from torch._decomp import core_aten_decompositions
torch.manual_seed(0)
TABLE = dict(core_aten_decompositions())
TRY_RUN = os.environ.get("TRY_RUN", "0") == "1"
DEV = torch.device("rngd", int(os.environ.get("RNGD_IDX", "3")))

DT = [("float64",torch.float64),("float32",torch.float32),("float16",torch.float16),
      ("bfloat16",torch.bfloat16),("int64",torch.int64),("int32",torch.int32),
      ("int16",torch.int16),("int8",torch.int8),("uint16",torch.uint16),("uint32",torch.uint32)]

def mk(dt, shape=(4,8)):
    if dt.is_floating_point: return torch.randn(*shape).to(dt)
    if dt in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        return torch.randint(0, 5, shape, dtype=dt)
    return torch.randint(-3, 4, shape, dtype=dt)

class Mod(torch.nn.Module):
    def __init__(self, fn): super().__init__(); self.fn = fn
    def forward(self, *a): return self.fn(*a)

def cell(fn, dt, mkargs):
    # 1) torch eager 가 이 dtype 을 받는가?
    try:
        args = mkargs(dt)
        with torch.no_grad():
            ref = Mod(fn).eval()(*[a.clone() for a in args])
    except Exception:
        return "-"   # N/A: torch 자체가 op-dtype 미지원
    # 2) RNGD 컴파일
    try:
        with torch.no_grad():
            ep = torch.export.export(Mod(fn).eval(), tuple(a.clone() for a in args)).run_decompositions(TABLE)
            cm = CompileModule.from_exported(ep)
    except Exception as e:
        return "x" if "UnsupportedOp" in repr(e) else "E"
    if not TRY_RUN:
        return "O"
    # 3) NPU 실행 (카드 비었을 때만)
    try:
        cm.to(DEV)
        with torch.no_grad():
            cm(*[a.to(DEV) for a in args], device=DEV)
        return "R"
    except Exception as e:
        return "O*" if "EBUSY" in repr(e) or "busy" in repr(e) else "Erun"

# op 정의: (이름, fn, 입력생성). 입력은 dtype 인자를 받아 dtype 맞춰 생성.
one = lambda mkf: (lambda dt: (mkf(dt),))
OPS = [
  # --- 산술 (numeric 전반) ---
  ("add  (x+x)",      lambda x: x + x,                 one(lambda dt: mk(dt))),
  ("sub  (x-x)",      lambda x: x - x,                 one(lambda dt: mk(dt))),
  ("mul  (x*x)",      lambda x: x * x,                 one(lambda dt: mk(dt))),
  ("div  (x/x)",      lambda x: x / x,                 one(lambda dt: mk(dt))),
  ("pow  (x**2)",     lambda x: torch.pow(x, 2),       one(lambda dt: mk(dt))),
  ("clamp",           lambda x: torch.clamp(x, 0, 3),  one(lambda dt: mk(dt))),
  ("abs",             lambda x: torch.abs(x),          one(lambda dt: mk(dt))),
  ("neg",             lambda x: torch.neg(x),          one(lambda dt: mk(dt))),
  # --- 초월/실수 전용 ---
  ("exp",             lambda x: torch.exp(x),          one(lambda dt: mk(dt))),
  ("log",             lambda x: torch.log(torch.abs(x) + 1),  one(lambda dt: mk(dt))),
  ("sqrt",            lambda x: torch.sqrt(torch.abs(x)),     one(lambda dt: mk(dt))),
  ("rsqrt",           lambda x: torch.rsqrt(torch.abs(x)+1),  one(lambda dt: mk(dt))),
  ("sin",             lambda x: torch.sin(x),          one(lambda dt: mk(dt))),
  ("erf",             lambda x: torch.erf(x),          one(lambda dt: mk(dt))),
  ("sigmoid",         lambda x: torch.sigmoid(x),      one(lambda dt: mk(dt))),
  ("tanh",            lambda x: torch.tanh(x),         one(lambda dt: mk(dt))),
  ("softmax",         lambda x: torch.softmax(x, -1),  one(lambda dt: mk(dt))),
  ("relu",            lambda x: torch.relu(x),         one(lambda dt: mk(dt))),
  # --- matmul / conv ---
  ("mm",              lambda a,b: torch.mm(a, b),      lambda dt: (mk(dt,(4,8)), mk(dt,(8,4)))),
  ("conv2d",          lambda x,w: torch.nn.functional.conv2d(x,w,None,1,1), lambda dt: (mk(dt,(1,3,8,8)), mk(dt,(4,3,3,3)))),
  # --- 비교 / 논리 / 비트 ---
  ("eq  (x==x)",      lambda x: x == x,                one(lambda dt: mk(dt))),
  ("lt  (x<x)",       lambda x: x < x,                 one(lambda dt: mk(dt))),
  ("maximum",         lambda x: torch.maximum(x, x),   one(lambda dt: mk(dt))),
  ("logical_and",     lambda x: torch.logical_and(x, x), one(lambda dt: mk(dt))),
  ("bitwise_and",     lambda x: torch.bitwise_and(x, x), one(lambda dt: mk(dt))),
  # --- reduction ---
  ("sum",             lambda x: x.sum(dim=1),          one(lambda dt: mk(dt))),
  ("mean",            lambda x: x.float().mean(dim=1) if False else torch.mean(x, dim=1), one(lambda dt: mk(dt))),
  ("max.dim",         lambda x: torch.max(x, dim=1),   one(lambda dt: mk(dt))),
  ("argmax",          lambda x: torch.argmax(x, dim=1),one(lambda dt: mk(dt))),
  ("cumsum",          lambda x: torch.cumsum(x, dim=1),one(lambda dt: mk(dt))),
  # --- shape / 이동 / 생성 ---
  ("view",            lambda x: x.view(-1),            one(lambda dt: mk(dt))),
  ("cat",             lambda x: torch.cat([x, x], 1),  one(lambda dt: mk(dt))),
  ("permute",         lambda x: x.permute(1, 0),       one(lambda dt: mk(dt))),
  ("slice",           lambda x: x[:, 1:5],             one(lambda dt: mk(dt))),
  ("where",           lambda x: torch.where(x > 0, x, x), one(lambda dt: mk(dt))),
  ("clone",           lambda x: x.clone(),             one(lambda dt: mk(dt))),
  ("to_float32",      lambda x: x.to(torch.float32),   one(lambda dt: mk(dt))),
  ("full_like",       lambda x: torch.full_like(x, 1), one(lambda dt: mk(dt))),
]

rows = []
hdr = f"{'op':16s}" + "".join(f"{n[:8]:>9s}" for n,_ in DT)
print(hdr); print("-"*len(hdr))
for name, fn, mkargs in OPS:
    cells = {}
    for dn, dt in DT:
        cells[dn] = cell(fn, dt, mkargs)
    rows.append({"op": name, **cells})
    print(f"{name:16s}" + "".join(f"{cells[dn]:>9s}" for dn,_ in DT))
json.dump({"dtypes":[d for d,_ in DT], "rows":rows, "try_run":TRY_RUN},
          open("/home/jun/.claude/jobs/220196a8/tmp/dtype_matrix.json","w"), indent=1)
print("\n범례: O=컴파일OK  R=NPU실행OK  O*=컴파일OK(실행은 EBUSY)  x=unsup  -=N/A(torch미지원)  E=기타에러")
