#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extra_models/models_extra.py
================================================================================
DINOv3(-S+/-B) 와 ViT-5 를 **공식 소스에서 그대로 로드**하고, 그 위에서 우리의
faithful 토큰 축소(ToMe recipe: proportional-attention + attention-key metric +
attention↔MLP 사이 병합)를 돌리기 위한 어댑터.

★ 임의로 모델을 지어내지 않습니다:
  - DINOv3: timm 의 **비게이트 미러**(공식 LVD-1689M 가중치) 로드.
      timm/vit_small_plus_patch16_dinov3.lvd1689m  (ViT-S+/16, SwiGLU FFN)
      timm/vit_base_patch16_dinov3.lvd1689m        (ViT-B/16,  MLP FFN)
    (또는 --dinov3-hub 로 공식 facebookresearch/dinov3 torch.hub 사용; 가중치 게이트)
  - ViT-5: **공식 repo(wangf3014/ViT-5)** 의 models_vit5.py 를 import 해서 모델을
    빌드하고, 공식 체크포인트(HF: FengWang3211/ViT-5/vit5_base_patch16_224.pth)를 로드.

────────────────────────────────────────────────────────────────────────────────
왜 DINOv2 처럼 "모델명만 교체"가 안 되나 (공식 소스 확인):
  두 모델 다 **RoPE(rotary) 어텐션**이고 register 를 "특수 토큰"으로 다룹니다.
    - DINOv3(EvaAttention, timm/models/eva.py): 토큰 순서 [CLS, reg×4, patch...],
      rope 는 prefix(CLS+reg 5개)를 건너뛰고 patch 에만 적용. EvaAttention.forward 는
      attn_mask 인자를 받으므로 proportional 편향을 그대로 넣을 수 있음.
    - ViT-5(wangf3014/ViT-5): 토큰 순서 [CLS, patch..., reg×4], patch 는 self.rope,
      register 는 **별도 rope_reg(2×2 grid, theta=100)**, CLS 는 rope 없음. 어텐션에
      편향 인자가 없어 재구현 필요. RMSNorm + qk_norm(RMSNorm on q,k).

★★ 설계 결정 (README 참조, 논문에 명시할 것):
  DINOv2 에서는 register 가 patch 와 **동질**(공유 절대 위치임베딩)이라 "ToMe 가
  register 를 patch 에 병합해 없앤다"가 자연스러웠다. rope 모델에서는 register 가
  patch 와 다른 위치공간이라, register 를 patch pool 에 섞어 병합하면 rope 정렬이
  깨진다. 따라서:
    • Ours(레지스터 보호): CLS+register 를 병합에서 제외(prefix 고정), **patch 만**
      size-가중 병합. 이 forward 는 **모델 정확**하다(r=0 에서 공식 forward 와 일치).
      → selfcheck.py 가 이를 검증한다.
    • 병합 후 생존 patch 는 **원래 grid 위치의 rope 를 유지**한다(pos-tracking).
      (병합된 토큰은 흡수하는 dst 토큰의 위치를 물려받는다.)
  즉 이 번들은 "레지스터 보호(Ours)가 rope 기반 register 모델에서도 극단 patch 병합을
  견디는가"를 **모델 정확한 forward** 로 측정한다. 무보호 baseline(strat='noreg')은 register 를
  patch pool 에 합치는(=rope 정렬 깨짐) 대신, **register 를 입력 시퀀스에서 아예 제거**하는
  정의를 쓴다(= 같은 가중치의 '레지스터 없는 모델'). rope 를 깨지 않으면서 "register 없음"을
  재현하므로, Δ = Ours − noreg 은 **압축 하에서 register 의 기여**를 뜻한다(보호-vs-무보호가
  아니라 register 유무). 두 arm 다 최종 특징은 CLS([:,0])이며 patch 수·병합량은 동일하다.
────────────────────────────────────────────────────────────────────────────────

이 파일은 GPU 서버에서 실행됩니다(로컬은 CPU/timm 없음). 실행 전 반드시 selfcheck.py 로
r=0 forward 가 공식 forward 와 일치하는지 확인하세요.
"""
import sys, math, os
import torch
import torch.nn.functional as F


# ============================================================================ #
#  공통: size-가중 이분 소프트매칭 병합 (ToMe, 위치 추적 포함)
# ============================================================================ #
@torch.no_grad()
def _bsm_merge(x, size, metric, r, pos):
    """patch 토큰(x, size)과 그 원래 grid 위치(pos)에 대해 ToMe 병합 r개.
    x:[B,P,C], size:[B,P,1], metric:[B,P,d], pos:[B,P](원래 patch index).
    반환: 병합 후 x,size,pos. (prefix/보호 토큰은 이 함수 밖에서 처리)
    dst 가 src 를 흡수하고, dst 의 위치를 유지 → 생존 토큰의 rope 위치가 정의됨."""
    B, P, C = x.shape
    r = min(r, P // 2)
    if r <= 0:
        return x, size, pos
    a, b = metric[:, ::2], metric[:, 1::2]
    an = F.normalize(a, dim=-1); bn = F.normalize(b, dim=-1)
    scores = an @ bn.transpose(-1, -2)
    node_max, node_idx = scores.max(-1)
    edge = node_max.argsort(-1, descending=True)[..., None]
    unm_idx, src_idx = edge[:, r:, :], edge[:, :r, :]
    dst_idx = node_idx[..., None].gather(1, src_idx)
    xa, xb = x[:, ::2], x[:, 1::2]; sa, sb = size[:, ::2], size[:, 1::2]
    pa, pb = pos[:, ::2], pos[:, 1::2]
    unm = xa.gather(1, unm_idx.expand(-1, -1, C)); unm_s = sa.gather(1, unm_idx.expand(-1, -1, 1))
    unm_p = pa.gather(1, unm_idx.squeeze(-1))
    src = xa.gather(1, src_idx.expand(-1, -1, C)); src_s = sa.gather(1, src_idx.expand(-1, -1, 1))
    xb2 = (xb * sb).scatter_add(1, dst_idx.expand(-1, -1, C), src * src_s)
    sb2 = sb.scatter_add(1, dst_idx.expand(-1, -1, 1), src_s)
    x_out = torch.cat([unm, xb2 / sb2], 1)
    s_out = torch.cat([unm_s, sb2], 1)
    p_out = torch.cat([unm_p, pb], 1)          # dst(=b) 는 자기 위치 유지
    return x_out, s_out, p_out


# ============================================================================ #
#  DINOv3 (timm EVA)  —  EvaAttention 재사용(attn_mask=size bias, rope 내부)
# ============================================================================ #
def load_dinov3(size="base", hub=False, weights=None, device="cuda", img_size=224):
    """size ∈ {'splus'(S+), 'base'(B)}. hub=True 면 공식 facebookresearch/dinov3(게이트)."""
    if hub:
        name = {"splus": "dinov3_vits16plus", "base": "dinov3_vitb16"}[size]
        m = torch.hub.load("facebookresearch/dinov3", name, weights=weights)
    else:
        import timm
        name = {"splus": "hf_hub:timm/vit_small_plus_patch16_dinov3.lvd1689m",
                "base":  "hf_hub:timm/vit_base_patch16_dinov3.lvd1689m"}[size]
        m = timm.create_model(name, pretrained=True, num_classes=0, img_size=img_size)
    return m.eval().to(device)


@torch.no_grad()
def dinov3_forward_faithful(m, img, r, n_reg_keep=None, proportional=True):
    """timm-EVA DINOv3 위 faithful forward.
    n_reg_keep: 보호(및 시퀀스 유지)할 register 수. None/4=Ours(모델 정확, r=0서 공식 forward 일치).
                0=register 제거(rope-안전 무보호 baseline: 같은 가중치의 '레지스터 없는 모델').
    CLS+register 보호, patch 만 병합. EvaAttention(x, rope, attn_mask) 재사용:
    rope 는 병합 후 생존 patch 위치로 재-gather, attn_mask 는 proportional log(size) 편향."""
    npt = getattr(m, "num_prefix_tokens", 5)          # 1 CLS + 4 reg = 5
    n_reg_total = npt - 1
    if n_reg_keep is None:
        n_reg_keep = n_reg_total
    x = m.patch_embed(img)
    x, rope_full = m._pos_embed(x)                    # rope_full: patch 용 rope [P0, hd] (prefix 제외)
    x = m.norm_pre(x)
    if n_reg_keep < n_reg_total:                      # register 제거(뒤쪽 register 부터). patch rope 불변.
        x = torch.cat([x[:, :1 + n_reg_keep], x[:, 1 + n_reg_total:]], 1)
        npt = 1 + n_reg_keep
    B, N, C = x.shape
    H = m.blocks[0].attn.num_heads; hd = C // H
    P0 = N - npt
    size = torch.ones(B, N, 1, dtype=x.dtype, device=x.device)
    pos = torch.arange(P0, device=x.device).unsqueeze(0).expand(B, P0).clone()  # 생존 patch 의 원래 index
    if rope_full is not None and rope_full.dim() == 3:
        rope_full = rope_full[0]                      # [P0, hd] (배치 공통)
    # EvaAttention 은 self.num_prefix_tokens 로 rope 를 건너뛸 prefix 를 정한다. register 를 제거하면(npt<5)
    # 그 값이 안 맞아 rope 가 엉뚱한 토큰에 걸리므로, 현재 npt 로 동기화 후 복원.
    _orig_npt = [blk.attn.num_prefix_tokens for blk in m.blocks]
    for blk in m.blocks:
        blk.attn.num_prefix_tokens = npt
    try:
      for blk in m.blocks:
        Ncur = x.shape[1]
        xn = blk.norm1(x)
        # (1) 병합 metric = content key(pre-rope) 평균. base 는 qkv_bias=False 라 qkv(x) 로 충분.
        qkv = blk.attn.qkv(xn).reshape(B, Ncur, 3, H, hd)
        k_metric = qkv[:, :, 1].permute(0, 2, 1, 3).mean(1)     # [B,Ncur,hd]
        # (2) 현재 생존 patch 위치의 rope (배치별 다름) 를 gather. EvaAttention 은 apply_rot_embed_cat 로
        #     q[:,:,npt:] ([B,H,Pcur,hd]) 에 rope 를 곱하는데, 배치별 rope 가 head 축으로 broadcast 되려면
        #     [B,1,Pcur,rope_dim] 형태여야 함(안 그러면 H vs B 차원 충돌). unsqueeze(1) 로 head 축 추가.
        if rope_full is not None:
            rope_now = rope_full[pos].unsqueeze(1)     # [B, 1, Pcur, rope_dim]
        else:
            rope_now = None
        bias = size.log().reshape(B, 1, 1, Ncur) if proportional else None
        xa = blk.attn(xn, rope=rope_now, attn_mask=bias)
        x = x + blk.drop_path1(blk.gamma_1 * xa if getattr(blk, "gamma_1", None) is not None else xa)
        # (3) patch 만 병합. register 는 n_reg_keep 개만 시퀀스에 남아 보호됨(위 슬라이스). CLS+reg 는 항상 보호.
        prefix, patches = x[:, :npt], x[:, npt:]
        sp, sr = size[:, :npt], size[:, npt:]
        patches, sr, pos = _bsm_merge(patches, sr, k_metric[:, npt:], r, pos)
        x = torch.cat([prefix, patches], 1); size = torch.cat([sp, sr], 1)
        x = x + blk.drop_path2(blk.gamma_2 * blk.mlp(blk.norm2(x)) if getattr(blk, "gamma_2", None) is not None else blk.mlp(blk.norm2(x)))
      x = m.norm(x)
    finally:
      for blk, o in zip(m.blocks, _orig_npt):
        blk.attn.num_prefix_tokens = o
    return x[:, 0]                                     # CLS 특징


# ============================================================================ #
#  ViT-5 (공식 wangf3014/ViT-5)  —  어텐션 재구현(qk_norm + rope + rope_reg + size bias)
# ============================================================================ #
def load_vit5(size="base", ckpt=None, repo_dir=None, device="cuda", img_size=224):
    """공식 repo(wangf3014/ViT-5) 의 models_vit5.py 로 빌드 + 공식 .pth 로드.
    repo_dir: git clone 한 ViT-5 폴더(=sys.path 에 추가). ckpt: 공식 .pth 경로."""
    if repo_dir:
        sys.path.insert(0, repo_dir)
    import models_vit5                                 # 공식 정의 그대로
    build = {"small": "vit5_small", "base": "vit5_base", "large": "vit5_large"}[size]
    m = getattr(models_vit5, build)(img_size=img_size, num_classes=0)
    if ckpt:                                           # ckpt=None 이면 random init(어댑터 수학 검증용)
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("model", sd.get("state_dict", sd))
        m.load_state_dict(sd, strict=False)            # head 등은 무시(num_classes=0)
    return m.eval().to(device)


def _rotate_half_v5(x):
    """ViT-5 rope.py 의 rotate_half 와 동일(연속 쌍 interleave): out[2d]=-x[2d+1], out[2d+1]=x[2d]."""
    x1 = x[..., 0::2]; x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def _vit5_rope_tables(rope_mod, n_full, device, dtype):
    """VisionRotaryEmbedding.forward 의 2D-grid freqs 테이블을 device-무관하게 재구성.
    반환 cos,sin: 각 [n_full, D](D=head_dim). 병합 후엔 이 전체 그리드 테이블을 원래 위치로 gather.
    (공식 forward 는 .cuda() 하드코딩 + ft_seq_len=sqrt(현재 토큰수)라 병합 시퀀스에 직접 못 씀)"""
    if rope_mod is None:
        return None, None
    ft = int(round(n_full ** 0.5))
    t = torch.arange(ft, device=device, dtype=torch.float32) / ft * rope_mod.pt_seq_len
    fr = torch.einsum('..., f -> ... f', t, rope_mod.freqs.to(device).float())   # [ft, nf]
    fr = fr.repeat_interleave(2, dim=-1)                                          # [ft, 2nf]
    A = fr[:, None, :].expand(ft, ft, -1); Bm = fr[None, :, :].expand(ft, ft, -1)
    frc = torch.cat([A, Bm], dim=-1).reshape(ft * ft, -1)                         # [ft*ft, D]
    return frc.cos().to(dtype), frc.sin().to(dtype)


def _vit5_apply(x, cos, sin):
    """x:[B,Nsub,H,hd], cos/sin:[B,Nsub,1,hd] or [1,Nsub,1,hd]. 공식과 동일: x*cos + rotate_half(x)*sin."""
    return x * cos + _rotate_half_v5(x) * sin


@torch.no_grad()
def vit5_forward_faithful(m, img, r, n_reg_keep=None, proportional=True):
    """ViT-5 위 faithful forward. 토큰 순서 [CLS, patch..., reg×nreg] (공식과 동일).
    n_reg_keep: 유지·보호할 register 수. None/4=Ours(모델 정확, r=0서 공식 일치). 0=register 제거 baseline.
    공식 Attention.forward 를 정확히 미러: qkv→qk_norm→rope(patch, transpose 前 [B,N,H,hd])/rope_reg(reg)
    →transpose→q*scale·qk^T·softmax·@v·proj. + size bias(log) pre-softmax(r=0서 0). patch 만 병합, CLS+reg 보호.
    rope 는 병합 후 생존 patch 의 '원래 2D-grid 위치' 테이블을 gather(pos-tracking)."""
    nreg_total = getattr(m, "num_registers", 4)
    nreg = nreg_total if n_reg_keep is None else n_reg_keep
    x = m.patch_embed(img)
    B, P0, C = x.shape
    if getattr(m, "pos_embed", None) is not None:      # 공식: patch 에만, cat 前
        x = x + m.pos_embed
    cls = m.cls_token.expand(B, -1, -1)
    x = torch.cat([cls, x], 1)
    if nreg > 0 and getattr(m, "reg_token", None) is not None:
        x = torch.cat([x, m.reg_token.expand(B, -1, -1)[:, :nreg]], 1)   # [CLS, patch, reg×nreg]
    H = m.blocks[0].attn.num_heads; hd = C // H
    dev, dt = x.device, x.dtype
    size = torch.ones(B, x.shape[1], 1, dtype=dt, device=dev)
    pos = torch.arange(P0, device=dev).unsqueeze(0).expand(B, P0).clone()   # 생존 patch 원래 index
    a0 = m.blocks[0].attn
    cosP, sinP = _vit5_rope_tables(getattr(a0, "rope", None), P0, dev, dt)          # [P0, hd]
    cosR, sinR = _vit5_rope_tables(getattr(a0, "rope_reg", None), nreg_total, dev, dt)  # [nreg_total, hd]
    for blk in m.blocks:
        a = blk.attn
        N = x.shape[1]; Pcur = N - 1 - nreg
        xn = blk.norm1(x)
        qkv = a.qkv(xn).reshape(B, N, 3, H, hd)
        q, k, v = qkv.unbind(dim=2)                    # 각 [B,N,H,hd] (공식과 동일, transpose 前)
        if getattr(a, "qk_norm", False):
            dtq = q.dtype
            q = a.q_norm(q).to(dtq); k = a.k_norm(k).to(dtq)
        metric = k.mean(dim=2)                         # [B,N,hd] (pre-rope content key)
        # rope(patch): 생존 위치 gather → [B,Pcur,1,hd]
        if cosP is not None and Pcur > 0:
            cg = cosP[pos].unsqueeze(2); sg = sinP[pos].unsqueeze(2)
            q = torch.cat([q[:, :1], _vit5_apply(q[:, 1:1+Pcur], cg, sg), q[:, 1+Pcur:]], 1)
            k = torch.cat([k[:, :1], _vit5_apply(k[:, 1:1+Pcur], cg, sg), k[:, 1+Pcur:]], 1)
        # rope_reg(register, 보호=위치 고정 0..nreg-1) → [1,nreg,1,hd]
        if cosR is not None and nreg > 0:
            cr = cosR[:nreg][None, :, None, :]; sr = sinR[:nreg][None, :, None, :]
            q = torch.cat([q[:, :1+Pcur], _vit5_apply(q[:, 1+Pcur:], cr, sr)], 1)
            k = torch.cat([k[:, :1+Pcur], _vit5_apply(k[:, 1+Pcur:], cr, sr)], 1)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)   # [B,H,N,hd]
        attn = (q * a.scale) @ k.transpose(-2, -1)     # [B,H,N,N]
        if proportional:
            attn = attn + size.log().reshape(B, 1, 1, N)
        attn = attn.softmax(-1)
        xa = a.proj((attn @ v).transpose(1, 2).reshape(B, N, C))
        ls = getattr(blk, "layer_scale", False)
        x = x + blk.drop_path(blk.gamma_1 * xa if ls else xa)
        # patch 만 병합 (CLS 앞·reg 뒤 보호)
        cls_t, pat, reg_t = x[:, :1], x[:, 1:1+Pcur], x[:, 1+Pcur:]
        sc, spat, sreg = size[:, :1], size[:, 1:1+Pcur], size[:, 1+Pcur:]
        pat, spat, pos = _bsm_merge(pat, spat, metric[:, 1:1+Pcur], r, pos)
        x = torch.cat([cls_t, pat, reg_t], 1); size = torch.cat([sc, spat, sreg], 1)
        mlp_out = blk.mlp(blk.norm2(x))
        x = x + blk.drop_path(blk.gamma_2 * mlp_out if ls else mlp_out)
    return m.norm(x)[:, 0]


# ============================================================================ #
#  디스패치
# ============================================================================ #
def get_model_and_forward(key, device="cuda", img_size=224, **kw):
    """key ∈ {'dinov3_splus','dinov3_base','vit5_base'} → (model, nprefix, forward_fn)."""
    if key == "dinov3_splus":
        m = load_dinov3("splus", device=device, img_size=img_size, **kw)
        return m, getattr(m, "num_prefix_tokens", 5), dinov3_forward_faithful
    if key == "dinov3_base":
        m = load_dinov3("base", device=device, img_size=img_size, **kw)
        return m, getattr(m, "num_prefix_tokens", 5), dinov3_forward_faithful
    if key == "vit5_base":
        m = load_vit5("base", device=device, img_size=img_size, **kw)
        return m, 1 + getattr(m, "num_registers", 4), vit5_forward_faithful
    raise ValueError(f"unknown model key: {key}")
