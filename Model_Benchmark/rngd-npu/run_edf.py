#!/usr/bin/env python3
"""저장된 EDF 파일로 RNGD에서 비전 모델을 실행하는 검증 스크립트.

compile_vision.py 가 "컴파일이 되는가"를 확인했다면, 이 스크립트는 그 산출물(EDF)을
디스크에 저장해 두고 **컴파일 없이 다시 불러와 NPU에서 실행**하는 것까지 확인합니다.
핵심 가치: 컴파일(수백 초, host AOT)은 1회만, 이후 실행은 저장된 EDF 재사용(수 초).

동작 원리 (furiosa/torch/custom_ops/edf.py 실측 기반):
  - CompileModule = EdfModule(컴파일된 EDF) + ExportedProgramWeight(가중치 모듈)
  - forward 시 module_weight.flatten_inputs() 가 파라미터/버퍼를 EDF 그래프의
    런타임 입력으로 공급한다 → EDF 파일만으론 부족하고 같은 가중치가 필요
  - EdfModule(edf: ir.Edf) 는 Edf 객체 하나로 재구성 가능(로그도 "using pre-compiled edf")
  - ir.Edf.serialize() -> bytes / ir.Edf.deserialize(bytes) -> Edf

그래서 저장은 2개 파일: <model>.edf (직렬화된 EDF) + <model>.pt (state_dict).

사용:
  source ~/furiosa/bin/activate            # torchvision/timm 필요 (--no-deps 설치)
  python run_edf.py compile mobilenet_v2   # 컴파일 → mobilenet_v2.edf + .pt 저장 (수백 초)
  python run_edf.py run mobilenet_v2 --npu 0   # 저장된 EDF 로드 → rngd:0 실행 + CPU 대조 (수 초)
"""
import argparse, time, warnings
warnings.filterwarnings("ignore")

import torch
import furiosa.torch                        # PrivateUse1("rngd") 백엔드 등록 (torch 다음에 import)
from furiosa.torch import CompileModule
from furiosa.torch.custom_ops.edf import EdfModule          # EDF 재구성용
from furiosa.torch.export import ExportedProgramWeight, PASSES  # 가중치 모듈 + 전처리 패스
from furiosa.native_torch import ir                          # ir.Edf.deserialize
from torch._decomp import core_aten_decompositions, get_decompositions

# compile_vision.py 와 동일: core-aten + batch_norm 직접 분해
DECOMP = dict(core_aten_decompositions())
DECOMP.update(get_decompositions([
    torch.ops.aten._native_batch_norm_legit_no_training,
    torch.ops.aten._native_batch_norm_legit,
    torch.ops.aten.batch_norm,
    torch.ops.aten.native_batch_norm,
]))


def build(key):
    """compile_vision.py 와 동일 규칙. 컴파일 성공이 확인된 모델만 대상."""
    if key == "mobilenetv1":
        import timm
        return timm.create_model("mobilenetv1_100", pretrained=False), (1, 3, 224, 224)
    import torchvision.models as M
    return getattr(M, key)(weights=None), (1, 3, 224, 224)


def prepare(key, load_state=None):
    """모델 생성(+state_dict 로드) → eval/grad off → export → 분해. 컴파일·실행 공통."""
    torch.manual_seed(0)
    m, shape = build(key)
    if load_state:
        m.load_state_dict(torch.load(load_state, weights_only=True))
    m = m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    x = torch.randn(*shape)
    with torch.no_grad():
        ep = torch.export.export(m, (x,)).run_decompositions(DECOMP)
    return m, x, ep


def do_compile(key):
    m, x, ep = prepare(key)
    t = time.time()
    with torch.no_grad():
        cm = CompileModule.from_exported(ep)          # furiosa FX 패스 → 2026.2.0 컴파일러 → EDF
    dt = time.time() - t
    edf_bytes = cm.edf.serialize()                    # ir.Edf -> bytes
    with open(f"{key}.edf", "wb") as f:
        f.write(edf_bytes)
    torch.save(m.state_dict(), f"{key}.pt")           # 실행 때 가중치 재구성용
    print(f"[{key}] COMPILE_OK compile={dt:.1f}s  saved {key}.edf ({len(edf_bytes)/1e6:.1f}MB) + {key}.pt")


def do_run(key, npu):
    # 1) 저장된 EDF 로드 (컴파일 없음)
    t = time.time()
    with open(f"{key}.edf", "rb") as f:
        edf = ir.Edf.deserialize(f.read())
    t_edf = time.time() - t

    # 2) 가중치 모듈 재구성 — 컴파일 때와 동일 파이프라인(export→분해→PASSES)으로
    #    ExportedProgramWeight 를 만들어야 입력 순서/이름이 EDF 와 일치한다
    t = time.time()
    m, x, ep = prepare(key, load_state=f"{key}.pt")
    for fx_pass in PASSES:                            # from_exported 가 내부에서 하는 것과 동일
        ep = fx_pass(ep)
    weight = ExportedProgramWeight(ep)
    cm = CompileModule(EdfModule(edf), weight)        # 컴파일 없이 실행 모듈 복원
    t_prep = time.time() - t

    # 3) CPU 정답
    with torch.no_grad():
        ref = m(x)

    # 4) NPU 실행 (1회차 = 디바이스 로드 포함, 2회차 = warm)
    dev = torch.device("rngd", npu)
    cm.to(dev)
    with torch.no_grad():
        t = time.time(); out = cm(x.to(dev), device=dev); t_cold = (time.time() - t) * 1000
        t = time.time(); out = cm(x.to(dev), device=dev); t_warm = (time.time() - t) * 1000
    out = out.to("cpu").float()
    diff = (out - ref.float()).abs()
    print(f"[{key}] RAN_FROM_SAVED_EDF rngd:{npu}  edf_load={t_edf*1000:.0f}ms  weight_prep={t_prep:.1f}s")
    print(f"  latency cold={t_cold:.1f}ms warm={t_warm:.1f}ms  "
          f"max_abs_err={diff.max():.3g}  top1_npu={int(out.argmax(-1))} top1_cpu={int(ref.float().argmax(-1))} "
          f"top1_match={int(out.argmax(-1)) == int(ref.float().argmax(-1))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compile", "run"])
    ap.add_argument("model")
    ap.add_argument("--npu", type=int, default=0, help="run 시 사용할 rngd 인덱스")
    args = ap.parse_args()
    if args.cmd == "compile":
        do_compile(args.model)
    else:
        do_run(args.model, args.npu)


if __name__ == "__main__":
    main()
