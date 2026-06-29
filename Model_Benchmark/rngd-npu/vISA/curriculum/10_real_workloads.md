# 10 · 실전 워크로드 — 트랜스포머·MoE·MNIST

이 문서는 vISA 커리큘럼 모듈 10입니다. 실제 LLM(Qwen2.5-0.5B)이 vISA 커널로 어떻게 분해되는지, MoE 라우팅을 프리미티브로 어떻게 조립하는지, 그리고 완전 검증된 MNIST를 봅니다.
*선행: 04·06·07 (Contraction·Vector) · 예상 시간: 하루*

## 학습 목표

- [ ] 디코더가 embedding→attention→decoder→head로 분해되는 구조를 안다
- [ ] masked GQA softmax 커널을 트레이스한다
- [ ] MNIST end-to-end를 (NPU 백엔드로) 검증한다
- [ ] 무엇이 검증됐고 무엇이 TODO인지 구분한다

## 1. 개념

## 0. 먼저 큰 그림부터

지금까지 vISA의 "문법"(매핑 `m![...]`, Fetch/Collect/Contraction/Vector 엔진, 스케줄러)을 배웠다면, 이 장은 "그래서 이걸로 진짜 트랜스포머 LLM을 어떻게 짜느냐"를 다룹니다. 결론부터 말하면, vISA에는 **Qwen2.5-0.5B 디코더 전체가 손으로 쓴 커널로 들어있습니다** (`furiosa-opt-examples/src/transformer/`). 한 줄짜리 행렬곱이 아니라, 임베딩 → 24개 디코더 레이어 → LM 헤드까지 PyTorch 모델 하나가 통째로 vISA로 번역돼 있어요. 이걸 읽으면 "추상적인 매핑 문법"이 "실제 LLM의 어느 연산"에 대응되는지가 한눈에 들어옵니다.

다만 한 가지 현실을 먼저 못 박아 둡니다. **이 예제들을 진짜 NPU에서 끝까지 돌리려면 닫힌(closed) 컴파일러가 필요**합니다. 저장소에 들어있는 `cargo-furiosa-opt`는 소스가 아니라 껍데기입니다. 실제 파일을 보면

```rust
// cargo-furiosa-opt/src/main.rs (전부입니다)
fn main() {
    eprintln!("Install via: cargo binstall cargo-furiosa-opt");
    eprintln!("See https://github.com/furiosa-ai/furiosa-opt for details.");
    std::process::exit(1);
}
```

진짜 바이너리는 `cargo binstall cargo-furiosa-opt`로 받는 미리 빌드된 rustc 드라이버이고, 특정 nightly(`nightly-2026-05-01`)에 ABI가 고정돼 있습니다(`docs/src/introduction.md:25`). 그래서 우리는 NPU 없이도 **두 가지 백엔드로 학습**합니다. 이게 이 장 실습의 토대입니다.

## 1. 세 가지 백엔드 — NPU 없이 배우는 법

`cargo furiosa-opt`는 `--cfg backend="..."`를 주입해서 커널을 "어떻게 평가할지"를 고릅니다(`docs/src/introduction.md:114`).

| 백엔드 | NPU 필요? | 무엇이 실행되나 | 언제 쓰나 |
|---|---|---|---|
| `typecheck` | X | 커널 본문이 빈(phantom) 텐서로 실행 | 매핑/형상 오류를 빠르게 잡을 때(값 계산은 건너뜀) |
| `simulation` (기본) | X | 호스트에서 전체 해석 실행 | 개발 기본값, 수치 정확성까지 검증 |
| `npu` | O | 컴파일된 EDF를 하드웨어에서 | 끝까지(하드웨어 경로 포함) |

핵심: **simulation과 typecheck는 SDK도 NPU도 필요 없습니다**(`introduction.md:42`). 그리고 중요한 함정 하나 — "공개 SDK에는 호스트용 NPU 시뮬레이터가 (오늘 기준) 없다"고 문서가 명시합니다(`introduction.md:44`). 즉 `--backend simulation`은 "NPU 동작을 흉내내는 사이클 시뮬레이터"가 아니라 **호스트에서 매핑 의미를 그대로 해석해서 수치를 내는 인터프리터**입니다.

또 하나 자주 헷갈리는 점: `cargo check`는 어떤 백엔드든 **타입 검사만** 하고 커널 본문을 실행하지 않습니다. 그래서 `Collect output packet must be exactly 32 bytes` 같은 매핑 단언(assert)에 도달하지 못합니다. 매핑 단언까지 확인하려면 `--backend typecheck run`을 써야 합니다(`introduction.md:133`).

명령 형태는 cargo를 그대로 감쌉니다:
```bash
cargo furiosa-opt run  --release --bin gemm          # 기본 simulation 실행
cargo furiosa-opt --backend typecheck run --release  # 형상만 검증
cargo furiosa-opt test --release --bin gemm          # 레퍼런스와 수치 비교
```

## 2. "composable kernel" — 모델을 단계로 쪼개는 사고방식

`transformer/mod.rs`의 헤더가 전체 설계를 요약합니다(`src/transformer/mod.rs:1`). Qwen2.5-0.5B는 **4PE, W16A16KV16(전부 bf16)** 설정으로, 네 개의 `#[device]` 커널 단계로 나뉩니다.

| 모듈 | 담당 레이어 구간 |
|---|---|
| `embedding` | `embedding..transformer_0:qkv_projection` |
| `attention` | `transformer_N:attention` (prefill, S=1024) |
| `decoder` | `transformer_N:output_projection..transformer_N+1:qkv` |
| `head` | `transformer_-1:output_projection..output` (+ LM head) |

여기서 가장 중요한 발상은 **단계 경계를 "레이어 경계"가 아니라 "재사용 가능한 파이프라인 경계"에 둔다**는 점입니다. 예를 들어 `decoder` 커널 하나는 레이어 N의 뒷부분(o_proj→MLP)을 끝내고 곧바로 레이어 N+1의 앞부분(QKV 투영→RoPE)을 시작합니다(`decoder/mod.rs:63`). 왜 이렇게 어긋나게 자를까요? 그래야 "잔차 연결 + 정규화 + 다음 투영"이 한 커널 안에서 데이터를 메모리에 내렸다 다시 올리는 왕복 없이 이어지기 때문입니다. PyTorch의 `Qwen2DecoderLayer.forward()` 한 덩어리를, 하드웨어에서 데이터가 흐르기 좋은 모양으로 다시 자른 거예요.

공통 블록은 `common/`에 모여 재사용됩니다: `o_proj`(출력 투영), `mlp`(SwiGLU), `norm`(잔차+RMSNorm), `rope`(회전 위치 임베딩). 이 네 개가 decoder와 head에서 공유됩니다(`transformer/mod.rs:12`).

## 3. axes.rs — 모델 상수를 매핑 축으로 (`src/transformer/axes.rs`)

vISA에서 텐서 형상은 컴파일타임 상수 축으로 표현됩니다. `axes![...]` 매크로가 Qwen2.5-0.5B의 하이퍼파라미터를 그대로 축 이름으로 박아둡니다:

```rust
axes![
    D = 64,            // head_dim
    G = 7,             // gqa_ratio = 14 / 2  (Q헤드14 / KV헤드2)
    H = 896,           // hidden_size = 14 * 64
    K = 128,           // kv_proj_size = N * D
    M = 4864,          // mlp_intermediate_size
    N = 2,             // num_kv_heads
    Q = 896,           // q_proj_size
    R = 2,             // rope_rot (2x2 회전행렬 축)
    S_decode = 128,    // 시퀀스 길이(decode)
    S_prefill = 1024,  // 쿼리 시퀀스 길이(prefill)
    T = 1024,          // key/value 시퀀스 길이
    W_vocab = 151936,  // vocab_size
    C_kvcache = 1124,  // kv_cache_len = 1024 + 100 padding
    ...
];
```

`X=16, Y=4, Z=2, W_kbcast=32, W_norm=64` 같은 축은 모델 파라미터가 아니라 **하드웨어 타일링을 위한 브로드캐스트/복제 상수**입니다. 예컨대 `Y=4`는 4PE에 걸쳐 복제할 때, `W_norm=64`는 정규화 가중치를 S/2회 복제할 때 쓰입니다. 즉 axes에는 "모델의 차원"과 "하드웨어에 맞추기 위한 패딩/복제 인자"가 같이 들어있고, 이걸 구분해서 읽는 게 vISA 코드를 이해하는 첫 단추입니다. `D=64`인데 `G=7`(=14/2)인 이유: GQA(Grouped Query Attention)에서 Q헤드 14개가 KV헤드 2개를 7개씩 공유하기 때문입니다.

## 4. embedding 단계 (`src/transformer/embedding/`)

PyTorch로 치면 `embed_tokens → input_layernorm → q/k/v_proj → rotary_emb`입니다(`embedding/mod.rs:48`). 파이프라인을 따라가 봅시다.

### 4.1 임베딩 룩업 = gather (`embedding/embedding.rs`)
임베딩은 본질적으로 "토큰 ID로 큰 표에서 행을 골라오는" 연산입니다. 그래서 행렬곱이 아니라 **DMA gather**로 구현합니다:
1. `input_ids`를 SRAM으로 올림
2. 토큰 ID를 **바이트 오프셋으로 변환** — `vector_fxp(FxpBinaryOp::MulInt, 1792)`. 왜 1792? `H=896 × sizeof(bf16)=2 = 1792`, 즉 임베딩 표 한 행의 바이트 크기입니다. gather가 byte-offset 모드라 row index를 직접 바이트 위치로 바꿔주는 거예요(`embedding/embedding.rs:36`).
3. `embedding_table.dma_gather(&ids_hbm, 0x0, true)` — 마지막 인자 `scaled=true`가 "인덱스를 바이트 오프셋으로 해석하라"는 뜻(`embedding.rs:51`).

이 패턴(인덱스→스케일→gather)은 RoPE 테이블 룩업과 KV 캐시 scatter에서도 똑같이 반복됩니다. 한 번 익히면 계속 보입니다.

### 4.2 RMSNorm (`embedding/rms_norm.rs`)
`y = x * (1/sqrt(mean(x²)+eps)) * weight`. 벡터 엔진 파이프라인으로 표현되는데 핵심만 보면:
- `vector_fp_binary(MulF(Mul0), Stash)`로 x²를 만들고 `vector_intra_slice_reduce(Add)`로 합산
- `vector_fp_div(896.0f32)`로 H로 나눠 평균을 냄 (896=H)
- `vector_clip(Add, 6.25e-8f32)`로 epsilon 더함
- 따로 `Sqrt` → `1/x`로 역수 RMS를 만들어 VRF에 두고, 원본 x에 곱함

여기서 배울 점: **수치 안정성 상수(eps)와 차원 상수(896)가 코드에 직접 박힌다**는 것. 그리고 `hidden_copy`를 만들어 두는 이유 — 분산(variance) 계산과 최종 스케일링이 같은 입력을 **독립적으로 두 번** 읽어야 해서 복사본을 둡니다(`rms_norm.rs:90`).

### 4.3 Q/K/V 투영 (`embedding/proj.rs`)
세 투영이 미묘하게 다릅니다. V/K는 같은 패턴(가중치를 TRF에 올리고 활성을 스트림), **Q는 거꾸로** 활성을 TRF에 올립니다(`proj.rs:8`). 왜? Q 가중치 `[896,896]`가 K/V `[128,896]`보다 커서, 작은 쪽(활성)을 TRF FirstHalf에 두는 게 유리하기 때문입니다. 이 "큰 피연산자를 스트리밍하고 작은 쪽을 TRF에 고정" 전략은 o_proj, lm_head에서도 반복됩니다. 그리고 bias는 `vector_clip(ClipBinaryOpF32::Add, &bias_vrf)`로 matmul 결과에 융합(fuse)해서 더합니다 — 별도 패스가 아니라 contraction 직후 벡터 단계에서 한 번에. (참고: 이 커널 경로는 K/V bias를 생략합니다 — 주석에 명시, `decoder/mod.rs:78`.)

### 4.4 RoPE (`common/rope.rs`)
회전 위치 임베딩을 "2×2 회전행렬과의 작은 행렬곱(einsum)"으로 봅니다. `rope_table`은 `[P, D/2, R, R]` 모양으로, 위치마다 cos/sin을 담은 2×2 행렬을 미리 계산해 둔 표입니다(`axes.rs:11`, R=2). 흐름:
1. position_ids → 바이트 오프셋(×256) → `rope_table.dma_gather`로 위치별 회전계수 가져오기(`rope.rs:73`)
2. 회전계수를 TRF FirstHalf에 올려 **Q와 K가 재사용**
3. K를 회전쌍 레이아웃으로 TRF SecondHalf에 올리고 `contract_outer`로 2×2 × K쌍 수행 → K 회전 완료
4. 같은 패턴으로 Q 회전

배울 점: RoPE처럼 "보기엔 복잡한 삼각함수 연산"도, **표 룩업 + 작은 행렬곱**으로 재구성하면 NPU의 contraction 엔진에 그대로 태울 수 있다는 것. 이게 transformer.md 문서가 말하는 "RoPE는 회전행렬을 준비하면 einsum으로 귀결된다"의 실제 구현입니다(`docs/src/kernel-examples/transformer.md:130`).

## 5. attention 단계 (`src/transformer/attention/`)

`attention/mod.rs`가 prefill용으로 batch=1, S=1024를 처리합니다. 분해는 **KV 캐시 쓰기 → Q×Kᵀ → softmax → score×V**(`attention/mod.rs:49`). scale은 `head_dim^-0.5 = 0.125`이고 attn_weight 안에서 적용됩니다.

### 5.1 KV 캐시 scatter (`attention/kv_cache.rs`)
K/V를 캐시의 "지정 위치"에 흩뿌리는(scatter) 연산입니다. 임베딩 gather의 거울상이에요. 인덱스를 바이트 오프셋으로 바꾸고(`MulInt, 256` — K행 바이트 크기 = K=128 × 2), HBM에 내린 뒤 `v_scatter_reshaped.dma_scatter::<m![T], _, _>(&v_idx_hbm, out_v_cache, true)`로 캐시에 씁니다(`kv_cache.rs:83`). 이 `dma_scatter`가 바로 `run_qwen` 테스트가 #[ignore]된 이유의 핵심입니다(뒤 8장).

### 5.2 Q×Kᵀ (`attention/attn_weight.rs`)
K를 두 개의 T-반쪽(half)으로 나눠 TRF FirstHalf/SecondHalf에 올리고, cascade 누적으로 점수를 만듭니다(`attn_weight.rs:54`). T=1024를 한 번에 못 올리니 512씩 쪼개 두 번 contraction하고 같은 점수 버퍼에 누적하는 거예요. GQA의 G=7 그룹 구조가 매핑 축 `m![S%8, G, T]`에 그대로 박혀 있습니다.

### 5.3 softmax — 이 장의 read-trace 대상 (`attention/softmax.rs`)
softmax는 vISA의 벡터 엔진/분기(branch) 기능을 가장 잘 보여주는 커널이라 자세히 봅니다. PyTorch로는 `softmax(scores + mask, dim=-1)`을 헤드별로 하는 것이고, GQA라 **G=7 그룹마다 독립적으로** 돕니다. 수치 안정 공식: max-subtract → exp → sum → div(`softmax.rs:6`).

핵심 단계:
1. **마스크를 분기 상태로 변환**: attention_mask(i32)를 `vector_intra_slice_tag(TagMode::Comparison([...Equal{boundary:0}...]))`로 읽어 VRF에 분기 로그를 만듭니다(`softmax.rs:33`). 이게 "마스크 텐서를 branch log에 쓴 뒤 분기 연산으로 처리한다"는 문서 설명의 실물입니다(`transformer.md:197`).
2. **그룹 루프**: `GROUP_SRAM_ADDRS: [u64; 7]` 7개 워크스페이스 주소를 두고, 각 그룹 타일을 떼어내 처리(`softmax.rs:55`). 이게 "연산 1개 = 그래프 1개"식으로 손으로 펼친 GQA입니다.
3. **마스킹**: `vector_logic(LogicBinaryOpF32::BitAnd, -3.3895314e38f32)` — 마스크된 위치를 아주 큰 음수로 채워 exp 후 0이 되게 함(`softmax.rs:73`). `TagMode::Vrf`가 1단계에서 만든 분기 상태를 암묵적으로 읽습니다.
4. **행별 max**: `vector_intra_slice_reduce(Max)`로 행 최대값을 VRF에 (`softmax.rs:87`).
5. **sum(exp(x-max))**: `SubF(max)` → `Exp` → `intra_slice_reduce(Add)` → VRF (`softmax.rs:101`).
6. **정규화**: `SubF(max)` → `Exp` → `fp_div(sum)` → bf16 캐스트 후 그룹 타일 주소에 commit (`softmax.rs:119`).

배울 점: (a) max-subtract라는 수치 트릭이 하드웨어 벡터 연산 시퀀스로 1:1 매핑된다는 것, (b) 마스크가 if-else가 아니라 **태그/분기 상태 + BitAnd로 거대음수 주입**으로 표현된다는 것, (c) GQA의 "그룹 독립성"이 Rust `for` 루프 + 주소 분리로 펼쳐진다는 것. softmax 하나만 정독해도 vISA 벡터 엔진의 표현력이 손에 잡힙니다.

### 5.4 score×V (`attention/attn_output.rs`)
V도 두 T-반쪽으로 나눠 TRF에 올리고, score 반쪽 × V 반쪽 조합 4번을 누적해 `[N,G,D]`를 만든 뒤 hidden_size H 레이아웃으로 reshape해서 HBM에 씁니다(`attn_output.rs:169`). 마지막에 두 T-반쪽 결과를 `begin_interleaved::<I,...>` + `vector_clip_zip(Add)`로 합칩니다 — `I=2`(dual-input interleave)가 여기서 쓰이는 "두 텐서를 지퍼처럼 끼워 더하는" 패턴입니다.

## 6. decoder / head 단계와 공통 블록

### 6.1 decoder (`src/transformer/decoder/mod.rs`)
6단계: O투영 → (잔차+post-attn norm) → MLP → (잔차+input norm, 다음 레이어 시작) → QKV 투영 → RoPE. PyTorch `Qwen2DecoderLayer.forward()`와 1:1로 주석에 적혀 있습니다(`decoder/mod.rs:53`). 입력/출력이 `arg0..arg15`, `out#0..out#3`으로 번호 매겨진 게 보이죠 — 이건 호스트(테스트)가 미리 할당한 HBM 버퍼에 커널이 결과를 쓰는 구조입니다.

### 6.2 o_proj (`common/o_proj.rs`)
출력 투영 `[H→H, bias=False]`. 가중치 `[896,896]`가 TRF FirstHalf에 들어가기엔 커서 **입력을 TRF에 두고 가중치를 SRAM에서 스트리밍하는 "역방향 matmul"**을 씁니다(`o_proj.rs:21`). embedding은 56폭, o_proj는 112폭 타일링(H/112=8)을 쓴다는 차이도 주석에 명시(`o_proj.rs:5`). "타일 폭"이 성능/정합성에 영향을 주는 튜닝 노브라는 감각을 여기서 얻습니다.

### 6.3 MLP = SwiGLU (`common/mlp.rs`)
`x = silu(gate(x)) * up(x); x = down(x)`, SiLU(x)=x·sigmoid(x)(`mlp.rs:3`). gate/up 투영을 각각 두 TRF 반쪽으로 누적해 계산하고, gate에 `Sigmoid`를 적용한 뒤 up과 곱하고(`vector_fp_binary(MulF(Mul0), &up_vrf)`, `mlp.rs:229`), down 투영으로 H로 되돌립니다. M=4864라는 중간 차원이 여기저기 `M/76`, `M/19%4`로 쪼개지는 게 보이는데, 이게 "큰 중간 텐서를 SRAM/TRF 용량에 맞춰 잘게 나눠 흘리는" 전형입니다.

### 6.4 norm 3종 (`common/norm.rs`)
- `residual_norm`: post-attention. (정규화 결과, 잔차합) 둘 다 반환 (`norm.rs:27`)
- `residual_norm_post`: post-MLP. 잔차합을 out_hidden에 씀 (`norm.rs:66`)
- `final_norm`: 마지막 레이어 후, **출력 타일링이 다름** — attention/MLP 소비자는 H-연속(`m![S%2,H%224]`), lm_head 소비자는 S-연속(`m![H%14,S]`)이 필요해서 final_norm만 S-연속으로 냅니다(`norm.rs:178`). "다음 단계가 원하는 레이아웃에 맞춰 정규화 출력 모양을 바꾼다"는 게 vISA 커널 조립의 현실적 디테일입니다. 잔차합은 `begin_interleaved` + `vector_clip_zip(Add)`로 두 텐서를 더해서 만듭니다(`norm.rs:42`).

### 6.5 head + lm_head (`src/transformer/head/`)
head 커널은 decoder 뒷부분 + 최종 RMSNorm + LM 헤드입니다(`head/mod.rs:39`). lm_head는 `logits = hidden @ embedding_tableᵀ`인데, **가중치 묶기(tie_word_embeddings=True)** 때문에 임베딩 표를 그대로 LM 헤드 가중치로 씁니다(`lm_head.rs:43`). vocab W=151936이 너무 커서 `C_lmhead=8192` 청크로 19번(`for chunk_idx in 0..19`) 나눠 처리하며, **더블 버퍼링**(짝/홀 청크마다 다른 SRAM 주소)으로 로드와 계산을 겹칩니다(`lm_head.rs:74`). 8192×19=155648 ≥ 151936이라 vocab을 덮습니다. 거대 vocab을 청크+더블버퍼로 흘리는 이 패턴이 LLM 출력층의 표준 처리법입니다.

## 7. MNIST — 유일하게 끝까지 검증된 신경망 (`src/mnist/mod.rs`, `tests/mnist_tests.rs`)

트랜스포머는 거대해서 "조각별로는 맞다"를 보이지만, **엔드투엔드 수치 정합이 완전히 검증된 진짜 NN은 MNIST**입니다. 구조는 단순한 FC 분류기: 784→256(ReLU)→10(`src/mnist/README.md`). 그래서 vISA를 처음 배울 때 "전체가 돌아가는 한 바퀴"를 보기에 딱 좋습니다.

구조:
- `fc1_matmul`: 입력 `[X=800]`을 TRF에 올리고 가중치 `[H=256, X]`와 contraction (`mnist/mod.rs:8`)
- `fc1_bias_prepared`: bias를 transpose/switch로 matmul 출력 레이아웃에 맞춤 (`mod.rs:35`)
- `fc1_relu`: `begin_interleaved`로 matmul+bias를 더하고 `vector_clip(Max, 0.0)`로 **ReLU**(0과의 max) (`mod.rs:81`)
- `fc2`+`fc2_bias_prepared`: 두 번째 FC로 `[C=16]` 로짓 (10개만 의미, 16은 패딩)
- `forward`: `fc1_relu` → `fc2` (`mod.rs:161`)

테스트(`tests/mnist_tests.rs`)는 진짜 학습된 가중치를 **safetensors**로 읽어옵니다: `mnist.py`(PyTorch 레퍼런스)가 학습/추출하고, Rust 테스트가 그 가중치를 HBM에 올려 10개 이미지를 추론한 뒤 argmax 예측을 라벨과 비교합니다(`mnist_tests.rs:33`). 이게 "PyTorch 레퍼런스 ↔ vISA 커널" 정합을 직접 거는 모범 사례입니다.

⚠️ 중요한 함정: 이 테스트는 `#[cfg_attr(not(backend="npu"), ignore="fc1_bias_prepared's reshape-around-padding trips CPU-sim verify_transpose")]`로 막혀 있습니다(`mnist_tests.rs:8`). 즉 **simulation/typecheck에서는 기본적으로 건너뜁니다**. 패딩을 둘러싼 reshape가 CPU 시뮬의 transpose 검증을 건드리기 때문이에요. 진짜 수치 검증은 NPU에서 합니다. typecheck로 돌리면 비교 루프가 phantom(빈) 값이라 0회 반복으로 "trivially pass"합니다. README가 적은 `cargo furiosa-opt test --test mnist_tests`는 NPU(또는 `-- --ignored` 강제) 맥락이라는 점을 기억하세요.

## 8. scatter / gather — KV 캐시의 토대 (`src/scatter_gather.rs`)

`scatter_minimal`/`gather_minimal`은 KV 캐시 동작을 최소 단위로 격리한 테스트 커널입니다. K=512 키를 C=612 캐시에 흩뿌립니다(희소: C>K). PyTorch로는 `cache[index] = data`(dim 0 index_put)와 같습니다(`tests/scatter_gather_tests.rs:3`). 

여기 두 가지 현실적 제약이 주석에 적혀 있어 꼭 알아야 합니다(`scatter_gather.rs:6`):
1. `renegade-8pe`(num_slices=256) 설정에서 **gather 커널은 파티셔닝 검사에 걸립니다** — `power_of_two_aligned(C/2)==num_slices`인데 C/2=306이 256이 아니라 512로 올림되기 때문. 즉 캐시 크기/슬라이스 수가 안 맞으면 컴파일 자체가 안 됩니다. "비-2의거듭제곱, K보다 큰 캐시"가 이 제약을 일부러 스트레스합니다.
2. LIR 런타임이 **in-place `&mut output`을 지원하지 않습니다**. 그래서 `gather_minimal`은 결과를 기존 `&mut HBM`에 쓰지 않고 새 주소로 `to_hbm`해서 값으로 반환합니다(`scatter_gather.rs:42`).

`test_scatter_minimal`은 #[ignore]가 없어서 **simulation에서 그대로 수치 검증이 됩니다** — 이 장에서 NPU 없이 "값까지 맞는지" 확인할 수 있는 가장 깔끔한 실제 테스트입니다. API 시그니처: `DmTensor::dma_scatter::<Key,...>(&index_hbm, &mut output, scaled)`(`furiosa-opt-std/src/tensor/memory.rs:730`)와 `HbmTensor::dma_gather::<...>(&index, addr, scaled)`(`memory.rs:407`). scatter는 key 축이 source에 온전히 포함돼야 한다는 assert가 있습니다(Chip/Element에 쪼개지면 indirect DMA가 주소를 못 잡음, `memory.rs:738`).

## 9. 무엇이 검증됐고 무엇이 TODO인가 — 정직한 지도

vISA의 LLM 예제는 "완성된 제품"이 아니라 "진행 중인 분해 작업"입니다. 정확히 어디까지 됐는지 구분해야 합니다.

- **완전 검증(수치까지)**: MNIST(NPU), scatter_minimal(simulation). 
- **단계별로 존재하지만 엔드투엔드 미검증**: Qwen2.5-0.5B의 embedding/attention/decoder/head 커널들. `tests/transformer_tests.rs`의 `run_qwen`은 24레이어를 호스트에서 오케스트레이션하지만 **`#[ignore = "DmaCommandScatter lowering not yet implemented"]`** 로 막혀 있습니다(`transformer_tests.rs:266`). 즉 KV 캐시 scatter의 하드웨어 lowering이 아직 안 돼서 끝까지 못 돕니다. run_embedding/run_attention/run_decoder/run_head는 #[test]가 아니라 헬퍼 async fn이고, 가중치도 `HostTensor::uninit()`(미초기화 플레이스홀더)이라 "형상/배선 검증용"입니다(주석: "TODO: Fill with actual weight values", `transformer_tests.rs:21`).
- **명백한 스텁(미완성)**: `src/attention.rs`는 Llama-3.1 8PE/4chip prefill 첫 블록을 노린 커널인데, embedding gather와 norm weight 로드까지만 있고 루프 본문이 **`// TODO: Complete the function definition.`** 로 끝납니다(`attention.rs:92`). 이름(`compile_llama3_1_mlperf_..._b1_s1024`)만 거창하고 실제로는 미완성이라는 걸 알아야 헛다리를 안 짚습니다.

이 지도를 머리에 넣고 있으면 "왜 transformer 예제를 simulation으로 돌렸는데 안 끝나지?"에서 헤매지 않습니다. 답은 "원래 #[ignore]이고, scatter lowering이 미구현"이기 때문입니다.

## 10. 툴링과 숙련의 현실

### 10.1 cargo furiosa-opt = 미리 빌드된 rustc 드라이버
앞서 봤듯 저장소의 `cargo-furiosa-opt`는 껍데기이고, 진짜는 binstall로 받는 닫힌 바이너리입니다(`cargo-furiosa-opt/Cargo.toml`의 `disabled-strategies=["quick-install","compile"]`이 "소스 컴파일 불가"를 박아둠). NPU 디바이스 바이너리(`*.bin`)를 낼 땐 `aarch64-linux-gnu-{gcc,as,ld,objcopy}`를 부르고, `furiosa-opt-std/build.rs`가 bindgen으로 `libclang.so`를 씁니다(`README.md:32`). 즉 컴파일러 내부(EDF 코드젠)는 비공개입니다.

### 10.2 Schedule Viewer (벤더 스케줄 시각화)
`docs/src/moving-tensors/memory-performance.md:136`이 "Schedule Viewer를 보면 어떤 연산이 병렬로 도는지, 실제 컨텍스트 배정이 맞는지 확인할 수 있다"고 안내합니다(introduction.md#schedule-viewer 앵커로 링크하지만 이 스냅샷의 introduction.md엔 해당 섹션이 비어 있어 사실상 죽은 링크입니다). 우리 분석/벤더 답변 기준 사용법은 `cargo furiosa-opt compiler build --dump-schedule`로 스케줄을 덤프한 뒤 `furiosa-schedule-viewer` GUI로 여는 것입니다(콜그래프 도구의 후속). 스케줄러 자체는 "텍스트 작성 순서 + 명시적 메모리 주소" 두 입력으로 동작하며 **순서를 재배치하지 않습니다**. 같은 컨텍스트는 직렬화, 다른 컨텍스트(`ctx.main`/`ctx.sub`)는 병렬입니다(`docs/src/scheduler.md:4`). 그래서 `ctx.sub`로 TRF 로드를 하고 `ctx.main`으로 contraction을 하면 TRF 절반이 달라(WAR 해저드 없음) 자동으로 겹쳐 돕니다(`scheduler.md:91`).

### 10.3 Language Server (`docs/src/appendix/language-server.md`)
`furiosa-rust-analyzer-proxy`는 rust-analyzer를 감싸서, `Stride<Symbol<A>, 8>` 같은 장황한 내부 타입을 `m![A/8]` 매핑 표기로 바꿔 보여줍니다(`appendix/language-server.md:3`). Hover/Inlay Hints/Signature Help/Call Hierarchy 등 표준 IDE 기능을 매핑 표기로 변환해 줍니다. 팁: `rust-analyzer.inlayHints.maxLength`를 `null`로 두면 긴 힌트가 `_`로 잘리는 걸 줄일 수 있습니다(`language-server.md:88`). 함정: 사용자가 `Symbol<T>` 같은 이름을 직접 정의하면 LSP가 그걸 `m![T]`로 오인 표시할 수 있는데, 이건 화면 표시 문제일 뿐 동작엔 영향 없습니다(`language-server.md:103`). vISA의 매핑 타입은 사람이 읽기엔 끔찍하게 길어서, 이 프록시 없이 손으로 디버깅하는 건 사실상 비현실적입니다 — 숙련의 일부가 이 도구에 의존한다는 뜻입니다.

### 10.4 Makefile (`Makefile`)
주요 타깃: `make check`(cargo check), `make clippy`(`-D warnings`), `make test`(workspace release), `make test-typecheck`(`--cfg backend="typecheck"`로 전체 테스트), `make mdbook-serve/build/test`(문서). typecheck 변형이 별도로 있는 게 포인트 — 매핑/형상 검증을 NPU 없이 CI에 거는 표준 경로입니다(`Makefile:57`).

### 10.5 ⛔ vISA .bin ≠ furiosa-llm .edf (우리 프로젝트의 핵심 결론)
이건 furiosa-opt 저장소가 아니라 우리 분석/벤더 공식 답변에서 나온 사실이라 별도로 표시합니다(MEMORY: virtual-isa). vISA 컴파일러가 내는 디바이스 바이너리(`.bin`, pert-ipc 포맷)는 furiosa-llm serve가 먹는 `.edf`(CBOR 그래프) 포맷과 **다릅니다**. 그래서 vISA로 만든 커널을 furiosa-llm serve에 masquerade로 끼워 넣을 수 없습니다 — "같은 칩, 다른 경로"입니다. 또 우리가 진짜 필요로 하는 온칩 순환 상태(Persistent Kernel)와 동적 shape는 둘 다 벤더 로드맵(올해~1년)이라, vISA는 "우리가 이미 한 host-loop의 더 저수준 재구현"일 뿐 새로 풀어주는 serve 경로는 없습니다. 정리하면: **vISA = 커널을 손으로 짜서 NPU 연산력에 직접 접근하는 길이지, furiosa-llm serve 게이트를 우회하는 길은 아니다.**

## 11. 한 문장 요약
vISA는 Qwen2.5-0.5B 디코더를 임베딩/어텐션/디코더/헤드 네 개의 손으로 쓴 커널로 분해해 두었고(공통 블록 o_proj/mlp/norm/rope 재사용), MNIST만이 엔드투엔드로 완전 검증된 NN이며, 트랜스포머 엔드투엔드(run_qwen)는 scatter lowering 미구현으로 #[ignore], Llama 스텁(attention.rs)은 TODO, 컴파일러는 닫힌 미리빌드 바이너리이고, NPU 없이는 simulation(수치)·typecheck(형상)로 학습하며, .bin은 serve용 .edf와 다른 포맷이라는 것 — 이게 vISA로 LLM을 다루는 현실의 전부입니다.

## 2. 핵심 API · 패턴

| 이름 | 쓰는 법 | 설명 | 출처 |
|---|---|---|---|
| `HbmTensor::dma_gather` | `pub fn dma_gather<Cluster2, Slice2, Element2, Element3>(&self, index: &HbmTensor<i32,...>, address: Address, scaled: bool) -> DmTensor<...>` | DRAM 표에서 index가 가리키는 행을 SRAM으로 모음(index_select). scaled=true면 index를 바이트 오프셋으로 해석. 임베딩/RoPE 룩업의 기반. | `furiosa-opt-std/src/tensor/memory.rs:407` |
| `DmTensor::dma_scatter` | `pub fn dma_scatter<Key, Element2, Element3>(&self, index: &HbmTensor<i32,...>, output: &mut HbmTensor<...>, scaled: bool)` | SRAM 값을 index 위치로 DRAM에 흩뿌림(KV 캐시 쓰기). key 축이 source에 온전히 포함돼야 함(assert). run_qwen #[ignore]의 원인인 DmaCommandScatter lowering이 이 연산. | `furiosa-opt-std/src/tensor/memory.rs:730` |
| `HbmTensor::to_dm` | `pub fn to_dm<Cluster, Slice, Element2>(&self, _dma: &mut DmaContext<{Dma::Tensor}>, address: Address) -> DmTensor<...>` | HBM→SRAM 적재. 이때 Cluster 분산이 결정됨. assert_dma_layout으로 DMA 쓰기폭 정합 검사. | `furiosa-opt-std/src/tensor/memory.rs:423` |
| `DmTensor::to_hbm` | `pub fn to_hbm<Element2>(&self, _dma: &mut DmaContext<{Dma::Tensor}>, address: Address) -> HbmTensor<...>` | SRAM→HBM 스필. 새 주소의 HbmTensor를 값으로 반환(기존 &mut에 쓰지 않음). | `furiosa-opt-std/src/tensor/memory.rs:712` |
| `Builder::contract_outer` | `.contract_outer::<OutTime, OutPacket, Lane, TrfElement>(&trf_tensor)` | TRF에 올린 피연산자와 fetch한 입력의 외적 contraction(matmul/einsum의 핵심). 뒤에 contract_packet/time/lane으로 reduce 범위를 지정. | `furiosa-opt-std/src/engine/contraction/outer/mod.rs:74` |
| `Builder::to_trf` | `.to_trf(TrfAddress::FirstHalf \| SecondHalf \| Full)` | 수집한 타일을 Tensor Register File에 적재. FirstHalf/SecondHalf를 나눠 쓰면 sub/main 컨텍스트가 WAR 없이 겹쳐 돎. | `furiosa-opt-std/src/engine/collect.rs:56` |
| `launch` | `pub async fn launch<F, P>(_f: F, args: P) -> F::Output` | #[device] 커널을 호스트에서 구동. 테스트가 ctx와 HBM 텐서 튜플을 넘겨 호출. | `furiosa-opt-std/src/runtime/mod.rs:195` |
| `Context::acquire` | `pub fn acquire() -> impl DerefMut<Target = Context>` | 디바이스 컨텍스트 획득(ctx.main/ctx.sub/ctx.tdma/ctx.pdma 제공). 모든 테스트의 시작점. | `furiosa-opt-std/src/context.rs:72` |
| `axes! 매크로` | `axes![D = 64, H = 896, ...];` | 모델 하이퍼파라미터와 하드웨어 타일링 상수를 컴파일타임 매핑 축으로 선언. m![...] 안에서 사용. | `furiosa-opt-examples/src/transformer/axes.rs:3` |
| `#[device(chip = N)]` | `#[device(chip = 1)] pub fn forward(ctx: &mut Context, ...)` | vISA 커널 진입점 표시(prelude의 furiosa_opt_macro::device). chip 수를 지정. | `furiosa-opt-std/src/lib.rs:49` |

## 3. 실험 (직접 돌리기)

> 실험은 NPU 없이 `simulation`·`typecheck`로 돌아갑니다. 실행법은 [`../experiments/README.md`](../experiments/README.md), MNIST는 `cargo furiosa-opt test`(npu 전용).

### 실험 10.1 — scatter_minimal을 시뮬레이션으로 실행하고 수치까지 검증
*난이도 1/5 · 기반: `furiosa-opt-examples/tests/scatter_gather_tests.rs`*

**목표** — NPU 없이 simulation 백엔드가 '값까지' 맞추는 걸 직접 확인하고, KV 캐시의 토대인 dma_scatter 동작을 본다(이 장에서 #[ignore]가 없는 유일한 수치 검증 테스트).

```bash
cargo furiosa-opt test --release --test scatter_gather_tests -- --nocapture
```
**관찰** — test_scatter_minimal이 1 passed. 의미상 cache[:512]는 data로 채워지고 cache[512:612]는 0 — PyTorch의 cache[index]=data와 동일. 실패하면 assert_eq!(actual, expected)에서 어느 인덱스가 틀렸는지 출력.

**심화** — tests의 PyTorch 등가 주석대로 index를 arange가 아니라 역순/홀짝으로 바꿔 expected를 손으로 예측한 뒤 다시 돌려보기.

### 실험 10.2 — Qwen2.5-0.5B 24레이어 분해를 typecheck로 형상 검증
*난이도 2/5 · 기반: `furiosa-opt-examples/tests/transformer_tests.rs`*

**목표** — embedding→attention→(decoder×23)→head로 이어지는 전체 매핑 배선이 형상상 일관적인지 NPU 없이 확인한다.

```bash
cargo furiosa-opt --backend typecheck test --release --test transformer_tests -- --ignored run_qwen
```
**관찰** — typecheck에서는 커널 본문이 phantom 텐서로 실행되어 매핑/형상 단언만 검사됨. DmaCommandScatter lowering(시뮬/런타임 한계)은 typecheck에서 안 건드리므로 형상 검증 목적엔 적합. 통과하면 24층 배선이 형상상 정합. 만약 매핑 오류가 나면 메시지의 축 이름(예: T, S, G)으로 어느 단계가 안 맞는지 추적.

**심화** — simulation으로 같은 명령(--backend 생략)을 돌려, run_qwen이 'DmaCommandScatter lowering not yet implemented'로 #[ignore]된 이유(scatter 미구현)를 직접 확인.

### 실험 10.3 — axes.rs를 깨뜨려 typecheck가 매핑 오류를 잡는지 본다 (find-the-error)
*난이도 3/5 · 기반: `furiosa-opt-examples/src/transformer/axes.rs`*

**목표** — vISA의 컴파일타임 형상 검사가 얼마나 촘촘한지 체감한다. 축 값을 비정합으로 바꾸면 어디서 터지는지 관찰.

```bash
# axes.rs에서 T = 1024 를 T = 1000 으로 임시 변경 후:
cargo furiosa-opt --backend typecheck test --release --test transformer_tests -- --ignored run_qwen
# 확인 후 git checkout -- furiosa-opt-examples/src/transformer/axes.rs 로 원복
```
**관찰** — T가 더 이상 512/8 등으로 나눠떨어지지 않아 attn_weight/softmax의 m![...] 분해에서 매핑/나눗셈 단언 오류가 발생. 에러 메시지가 어느 collect/fetch에서 났는지 보고, 형상이 '딱 떨어져야' 하는 vISA의 규칙을 확인.

**심화** — 대신 G=7을 G=6으로 바꿔 softmax의 GROUP_SRAM_ADDRS(7개 고정 배열)와 G 축 불일치가 어떻게 드러나는지 비교.

### 실험 10.4 — MNIST 커널 매핑을 typecheck로 검증하고 PyTorch 레퍼런스 살펴보기
*난이도 2/5 · 기반: `furiosa-opt-examples/tests/mnist_tests.rs`*

**목표** — 유일하게 엔드투엔드 검증된 NN의 구조(FC→ReLU→FC)를 형상 레벨에서 통과시키고, simulation에서 왜 #[ignore]되는지 함정을 직접 본다.

```bash
cargo furiosa-opt --backend typecheck test --release --test mnist_tests -- --ignored --nocapture
```
**관찰** — typecheck에서 forward 커널의 매핑이 검사되고, 10장 비교 루프는 phantom(빈) 로짓이라 0회 반복으로 trivially pass. 진짜 수치 검증은 NPU에서만 됨(simulation은 fc1_bias_prepared의 패딩 reshape가 verify_transpose를 건드려 #[ignore]). 이 차이를 직접 확인하는 게 목표.

**심화** — src/mnist/mnist.py를 venv에서 실행(python mnist.py)해 mnist.safetensors를 만들고, Rust 테스트가 어떤 텐서 이름(hw.fc1.weight 등)을 읽는지 대응시켜 보기.

### 실험 10.5 — softmax 커널 read-trace + 마스크 상수 바꿔보기 (predict-then-run)
*난이도 3/5 · 기반: `furiosa-opt-examples/src/transformer/attention/softmax.rs`*

**목표** — 벡터 엔진 기반 마스크드 softmax(max-sub→exp→sum→div)와 분기(TagMode) 처리를 코드로 따라가고, 거대음수 상수를 바꾸면 매핑이 그대로인지 확인.

```bash
# softmax.rs:73 의 -3.3895314e38f32 를 -1.0e30f32 로 바꾼 뒤:
cargo furiosa-opt --backend typecheck test --release --test transformer_tests -- --ignored run_qwen
# 원복: git checkout -- furiosa-opt-examples/src/transformer/attention/softmax.rs
```
**관찰** — 상수만 바꾸는 건 매핑 형상에 영향이 없으므로 typecheck는 그대로 통과(값 검증은 phantom이라 스킵). 이를 통해 'typecheck는 수치를 보지 않는다'는 백엔드 차이를 체감. 동시에 코드를 읽으며 GROUP_SRAM_ADDRS 7개 루프가 GQA 7그룹과 1:1 대응함을 확인.

**심화** — vector_logic(BitAnd, ...) 대신 max-subtract 단계(softmax.rs:79~90)를 종이에 트레이스해서, 마스크된 위치가 exp 후 왜 0이 되는지 설명해 보기.

## 4. 연습문제 (손으로, 컴파일 없이)

**Q1.** axes.rs에서 G=7, N=2, D=64일 때 num_q_heads는 몇 개인가? 그리고 hidden_size H=896이 어떻게 나오는지 식으로 쓰라.

<details><summary>정답/힌트</summary>

num_q_heads = G×N = 7×2 = 14. H = num_q_heads × D = 14 × 64 = 896.

</details>

**Q2.** 임베딩 룩업에서 토큰 ID에 1792를 곱한다(vector_fxp MulInt 1792). 이 1792가 어떤 의미의 바이트 수인지, H와 dtype으로 설명하라.

<details><summary>정답/힌트</summary>

임베딩 표 한 행의 바이트 크기 = H × sizeof(bf16) = 896 × 2 = 1792. dma_gather가 scaled=true(바이트 오프셋) 모드라 행 인덱스를 바이트 위치로 바꾸는 것.

</details>

**Q3.** softmax.rs는 마스크된 위치를 -3.3895314e38f32로 채운다(vector_logic BitAnd). 이렇게 거대한 음수를 넣으면 이후 exp와 정규화에서 그 위치의 확률이 어떻게 되는지, 왜 max-subtract 단계가 먼저 필요한지 설명하라.

<details><summary>정답/힌트</summary>

exp(거대음수 - max) ≈ 0 → 정규화 후 확률 0. max-subtract는 exp 입력을 ≤0으로 만들어 오버플로(inf)를 막는 수치 안정화. 큰 양수 점수가 있어도 exp가 폭발하지 않음.

</details>

**Q4.** run_qwen 테스트는 const NUM_LAYERS = 24로 embedding 1회, attention 24회, decoder 23회, head 1회를 호출한다. 코드의 루프(for _ in 1..NUM_LAYERS)가 decoder/attention을 각각 몇 번 도는지, 왜 decoder가 attention보다 1회 적은지 설명하라.

<details><summary>정답/힌트</summary>

루프는 1..24 = 23회 → decoder 23, 그 안의 attention 23 + 루프 밖 첫 attention 1 = 24. 첫 레이어의 QKV는 embedding 커널이 만들고, head가 마지막 레이어 뒷부분을 담당하므로 decoder(레이어 사이 이음)는 23회면 충분.

</details>

**Q5.** lm_head는 vocab W=151936을 C_lmhead=8192 청크로 19회 처리한다. 8192×19은 얼마이며 vocab을 덮는가? 마지막 청크에서 남는(패딩) 원소 수는?

<details><summary>정답/힌트</summary>

8192×19 = 155648 ≥ 151936 → 덮음. 남는 패딩 = 155648 − 151936 = 3712원소.

</details>

**Q6.** 다음 명령 중 NPU 없이 '수치 정합까지' 검증되는 것은? (a) cargo furiosa-opt test --test scatter_gather_tests (b) cargo furiosa-opt --backend typecheck test --test transformer_tests -- --ignored run_qwen (c) cargo furiosa-opt test --test mnist_tests. 이유와 함께 고르라.

<details><summary>정답/힌트</summary>

(a). scatter_minimal은 #[ignore] 없이 simulation에서 수치 비교(assert_eq). (b)는 typecheck라 값 미검증·형상만. (c)는 not(backend=npu)에서 #[ignore]라 기본 스킵, 진짜 수치 검증은 NPU 필요.

</details>

**Q7.** decoder/mod.rs의 forward는 한 커널 안에서 '레이어 N의 끝(o_proj→MLP)'과 '레이어 N+1의 시작(QKV→RoPE)'을 같이 한다. 왜 레이어 경계가 아니라 이렇게 어긋나게 잘랐는지 한 가지 이유를 쓰라.

<details><summary>정답/힌트</summary>

잔차+정규화 결과(다음 레이어 입력)를 HBM에 내렸다 다시 올리는 왕복 없이 곧바로 다음 QKV 투영으로 이어가기 위함 — 데이터가 SRAM/TRF에 머무는 동안 파이프라인을 길게 가져가 메모리 트래픽을 줄임.

</details>

## 5. 흔한 함정

- 트랜스포머 엔드투엔드(run_qwen)는 simulation으로 돌려도 끝나지 않는다 — 원래 #[ignore = "DmaCommandScatter lowering not yet implemented"]이기 때문. 강제 실행해도 scatter lowering 미구현으로 막힌다.  
  ↳ 출처 `furiosa-opt-examples/tests/transformer_tests.rs:266`
- MNIST 테스트는 simulation/typecheck에서 기본 SKIP된다. #[cfg_attr(not(backend="npu"), ignore)]로 막혀 있고, fc1_bias_prepared의 패딩 reshape가 CPU-sim의 verify_transpose를 건드려 강제 실행 시 오히려 실패할 수 있다. 진짜 수치 검증은 NPU에서.  
  ↳ 출처 `furiosa-opt-examples/tests/mnist_tests.rs:8`
- src/attention.rs는 이름(compile_llama3_1_mlperf_...)만 거창하고 본문이 '// TODO: Complete the function definition.'로 끝나는 미완성 스텁이다. 동작하는 Llama 커널로 착각 금지.  
  ↳ 출처 `furiosa-opt-examples/src/attention.rs:92`
- transformer_tests의 run_embedding/run_decoder 등은 가중치를 HostTensor::uninit()(미초기화)으로 채운다. 형상/배선 검증용이지 실제 추론 결과가 의미 있지 않다('TODO: Fill with actual weight values').  
  ↳ 출처 `furiosa-opt-examples/tests/transformer_tests.rs:21`
- gather 커널은 설정에 따라 컴파일 자체가 안 될 수 있다. renegade-8pe(num_slices=256)에서 power_of_two_aligned(C/2)==num_slices 검사 실패(C/2=306→512). 캐시 크기/슬라이스 수 정합이 필수.  
  ↳ 출처 `furiosa-opt-examples/src/scatter_gather.rs:6`
- cargo check는 어떤 백엔드든 커널 본문을 실행하지 않아 'Collect output packet must be exactly 32 bytes' 같은 매핑 단언에 도달하지 못한다. 형상 단언까지 보려면 --backend typecheck run을 써야 한다.  
  ↳ 출처 `docs/src/introduction.md:133`
- Schedule Viewer 링크(introduction.md#schedule-viewer)는 이 문서 스냅샷에서 해당 섹션이 비어 있어 죽은 앵커다. 실제 사용은 cargo furiosa-opt compiler build --dump-schedule 후 furiosa-schedule-viewer로 연다(벤더/프로젝트 분석 기준).  
  ↳ 출처 `docs/src/moving-tensors/memory-performance.md:136`
- 저장소의 cargo-furiosa-opt 소스는 '설치 안내만 출력하고 종료'하는 껍데기다. 실제 컴파일러는 닫힌 미리빌드 바이너리(cargo binstall)이고 소스 컴파일이 막혀 있다(disabled-strategies). EDF 코드젠은 비공개.  
  ↳ 출처 `cargo-furiosa-opt/src/main.rs:1`
- norm 변형은 출력 타일링이 다르다. final_norm만 S-연속(m![H%14,S])으로 내보내는데(lm_head용), 이를 무시하고 residual_norm처럼 H-연속을 기대하면 lm_head 입력 형상이 어긋난다.  
  ↳ 출처 `furiosa-opt-examples/src/transformer/common/norm.rs:178`
- 이 decoder/embedding 커널 경로는 K/V bias를 의도적으로 생략한다(주석 명시). PyTorch Qwen2의 q/k/v_proj는 bias=True지만 vISA 경로는 Q bias만 융합한다.  
  ↳ 출처 `furiosa-opt-examples/src/transformer/decoder/mod.rs:78`

## 6. 핵심 정리 & 다음

기억할 사실:
- Qwen2.5-0.5B vISA 커널은 4PE, W16A16KV16(가중치/활성/KV 전부 bf16) 설정으로 작성됨 (`furiosa-opt-examples/src/transformer/mod.rs:1`)
- 모델 상수: hidden_size H=896(=14×64), head_dim D=64, num_q_heads=14, num_kv_heads N=2, gqa_ratio G=7, mlp_intermediate M=4864, vocab W=151936, max_position P=32768 (`furiosa-opt-examples/src/transformer/axes.rs:3`)
- prefill 시퀀스 S_prefill=1024, kv_cache_len C_kvcache=1124(=1024+100 패딩), decode 토큰 길이 S_decode=128 (`furiosa-opt-examples/src/transformer/axes.rs:14`)
- attention scale = head_dim^-0.5 = 64^-0.5 = 0.125, GQA는 14 Q헤드를 2 KV헤드가 7개씩 공유하므로 softmax를 그룹당 독립 수행 (`furiosa-opt-examples/src/transformer/attention/mod.rs:50`)
- 임베딩 룩업은 행렬곱이 아니라 DMA gather이며, 토큰 ID를 H×sizeof(bf16)=1792 바이트 오프셋으로 변환 후 dma_gather(scaled=true) 수행 (`furiosa-opt-examples/src/transformer/embedding/embedding.rs:36`)
- lm_head는 tie_word_embeddings=True라 임베딩 표를 가중치로 재사용하고, vocab 151936을 C_lmhead=8192 청크로 19회 더블버퍼링 처리(8192×19=155648≥151936) (`furiosa-opt-examples/src/transformer/head/lm_head.rs:43`)
- TRF 적재 전략: 큰 피연산자(o_proj/Q proj의 [896,896] 가중치)는 SRAM에서 스트리밍하고 작은 쪽(활성)을 TRF FirstHalf에 고정하는 '역방향 matmul'을 씀 (`furiosa-opt-examples/src/transformer/common/o_proj.rs:21`)
- 공개 SDK에는 오늘 기준 호스트용 NPU 시뮬레이터가 없음; simulation 백엔드는 사이클 시뮬이 아니라 호스트 인터프리터, typecheck는 매핑/형상만 검증 (`docs/src/introduction.md:44`)

➡️ 다음: [11_capstone.md](./11_capstone.md)
