import os, sys, traceback
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
import torch  # must be first
from furiosa.native_torch.ir import Dfg

def try_parse(path, label):
    dsl = open(path).read()
    print(f"\n===== {label}: parse {path} =====")
    try:
        dfg = Dfg.parse(dsl)
        print("PARSE OK")
        try:
            print("  required_symbolic_params:", dfg.required_symbolic_params)
        except Exception as e:
            print("  (req params err)", e)
        try:
            print("  input_symbols:", dfg.input_symbols)
        except Exception as e:
            print("  (input_symbols err)", e)
        return dfg
    except Exception as e:
        print("PARSE FAIL:", type(e).__name__)
        msg = str(e)
        print("  msg:", msg[:2000])
        return None

if __name__ == "__main__":
    for p in sys.argv[1:]:
        try_parse(p, os.path.basename(p))
