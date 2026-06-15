# bench-blog — Furiosa "RNGD vs RTX PRO 6000" Qwen3-32B 벤치 재현 키트

Furiosa 블로그 [*RNGD vs RTX PRO 6000 real-world efficiency benchmark (Qwen3)*](https://furiosa.ai/blog/rngd-rtx-pro-6000-real-world-efficiency-benchmark-qwen3)
의 실험을 **내 장비(RNGD 1장 vs RTX PRO 6000 1대)** 로 재현해 그 주장을 직접 검증하는 키트입니다.
RNGD 서버와 PRO 6000 서버에서 각각 같은 부하시험을 돌려 JSON 을 만들고, `compare.py` 로 합쳐
블로그 주장과 대조한 리포트를 뽑습니다.

---

## 0. 블로그가 실제로 뭘 쟀나 (분석 결과)

블로그 본문을 정독·교차검증한 결과입니다. **본문이 공개한 것과 안 한 것을 구분**했습니다.

**모델·하드웨어**
- 모델: **Qwen3-32B** (정밀도 미공개 — RNGD furiosa-llm 은 fine-grained FP8, 공식 `Qwen/Qwen3-32B-FP8` 도 W8A8 block-128).
- 처리량 그래프는 **4x RNGD 서버 vs 4x RTX PRO 6000 서버**, 전력 비교는 **8장 서버**(혼재).
- 한 모델을 serve 하는 데 쓴 **TP(카드 수)는 미공개**.

**워크로드·지표**
- 배치 **b8, b16, b32, b64, b256** 스윕. SLO **20/40 TPS/user** 범위 최적화.
- 핵심 지표 = *"target SLO per user 를 지키면서 동시에 서비스 가능한 사용자 수"* — 단, **"사용자 수"의 정확한 계산식(집계TPS÷SLO 인지, per-user SLO 유지 최대 동시성인지)은 본문 미정의**.
- **입력/출력 토큰 길이, 서빙 엔진(양측), 양자화 정밀도는 전부 미공개.**

**핵심 수치 (본문 인용)**
| 주장 | 값 | 조건 |
|---|---|---|
| SDK 2월→3월 처리량 | b64 1,200→1,500 TPS(+25%), b32 750→1,100 TPS(+47%) | RNGD |
| 서비스 용량(b32) | 5.8 → 47.5 users (8.2x) | SDK 개선분 |
| **RNGD/RTX 사용자 배수** | **1.8x / 1.9x / 2.0x** (SLO 20/30/40) | **"normalized for rack power"** |
| 서버당 사용자(40 TPS) | RNGD **46** vs RTX **41** | 1:1 근사(전력정규화 전) ≈ **1.12x** |
| TTFT (30 TPS SLO) | RTX 2.7~4.4s vs RNGD 1.1~2.1s | RNGD ≈ 절반 |
| 전력 | 8x RNGD **3kW** vs 8x RTX **6.6kW**, 칩 **180W** | |
| 랙(15kW, 30 TPS) | RNGD 5대 474명 vs RTX 2대 → **2.5x/랙** | |

### ⚠️ 가장 중요한 해석
블로그의 **1.8~2.5x 는 전부 "전력/랙 정규화"** 값입니다. 장비 1:1 에 가장 가까운 본문 수치는
**40 TPS 에서 서버당 46 vs 41명 ≈ 1.12x** 뿐입니다. 즉 **RNGD 의 진짜 우위는 처리량 자체(~1.1x)가
아니라 "와트당 처리량"(~2x, 칩 180W vs RTX ~600W급)** 입니다. 그래서 이 키트는 결과를 **두 층위**로
나눠서 보여줍니다:
- **(A) raw 1:1** — SLO당 사용자 수 → ~1.1x 나오면 정상.
- **(B) 전력 정규화** — users/kW, tokens/sec/W → ~2x 나오면 블로그 재현 성공.
  - `tokens/sec/W`(= 집계 처리량 ÷ 전력 = 와트당 토큰, 사실상 토큰/줄)는 **높을수록** 에너지 효율이 좋다는 뜻입니다. 차트에서 위에 있는 쪽이 와트당 더 많은 토큰을 냅니다. (반대로 TTFT 는 낮을수록, per-user TPS 는 SLO 이상을 유지할수록 좋습니다.)

---

## 1. 이 키트의 방법론 (블로그 미공개 부분은 합리적 기본값)

| 항목 | 선택 | 근거 |
|---|---|---|
| 모델/정밀도 | RNGD `qwen3-32b-fp8-tp8`(1장), GPU 공식 `Qwen/Qwen3-32B-FP8`(W8A8) | 양쪽 FP8. (RNGD 로컬빌드는 weight-FP8+bf16-act — 정밀도 註 참고) |
| ISL/OSL | **1024 / 256** (변경 가능) | 블로그 미공개 → 전형적 chat 부하. `ISL=`,`OSL=` 로 변경 |
| 프롬프트 내용 | 의미 없는 **합성 더미 문장**("The quick brown fox…")을 ISL 토큰만큼 반복·트림 | 처리량/지연/전력 측정이라 내용은 무의미, **길이만** 고정하면 됨. 양 플랫폼 각자 토크나이저로 트림해 `prompt_tokens` 일치 (`loadgen.py` `_BASE_SENT`/`build_prompt`) |
| 배치=사용자 | **1,8,16,32,64,256** | 블로그 b8~b256 + 단일스트림 천장용 b1 |
| per-user TPS | 스트리밍 decode 기준 = 출력토큰 ÷ (첫토큰~끝) | SLO 비교 대상 |
| "사용자@SLO" | per-user p50 TPS ≥ SLO 인 최대 동시성(보간) | 블로그 정의 미공개 → 방어가능한 정의 채택 |
| 전력 | RNGD `furiosa-smi info` Power 컬럼, GPU `nvidia-smi power.draw`, 1Hz | 측정구간 평균 |
| 출력길이 고정 | `ignore_eos + min_tokens=max_tokens` | 정확히 OSL 토큰(furiosa·vLLM 공통 지원) |
| 캐시 왜곡 방지 | 요청마다 고유 prefix | prefix-caching 캐시히트 처리량 부풀림 차단 |

### 프롬프트가 어떻게 ISL(~1024) 토큰이 되나

`_BASE_SENT`("The quick brown fox…")은 그 자체로 27단어짜리 짧은 문장이지, 1024토큰이 아닙니다.
`build_prompt()`(`loadgen.py:43-63`)가 **반복 → 슬라이스 → (가능하면) 토크나이저 트림**으로 길이를 맞춥니다(ISL=1024 예시, 실측 토큰 수 병기):

1. **목표 보정**: 요청마다 붙는 고유 prefix를 빼고 본문은 `1024-14 = 1010`토큰을 목표(`loadgen.py:276`).
2. **단어 수 추정**: 이 영문 토큰/단어 비율은 실측 **1.04~1.07**(짧은 문장 1.07, 긴 본문 1.04) → 코드는 어림값 **1.06** 사용 → `int(1010/1.06) ≈ 952`단어(`loadgen.py:50`).
3. **문장 반복(타일링)**: `_BASE_SENT`(27단어)를 `952//27+2 = 37`번 이어붙여 풀을 만든 뒤 앞 952단어만 슬라이스(`loadgen.py:51-52`).
4. **토크나이저 트림(목표 초과 시에만)**: `--tokenizer`가 있으면 `AutoTokenizer`로 토큰화해 `len(ids) >= target`일 때만 `ids[:target]`로 자름(`loadgen.py:53-62`). **기본 설정에선 952단어가 실측 987토큰(<1010)이라 트림이 발동하지 않고 본문은 987에서 멈춤** — 1.06이 실제(~1.04)보다 약간 과대추정이라 단어가 모자란 결과.
5. **고유 prefix 부착**: 요청 직전 `[uid <uuid hex 32자> req] `를 앞에 붙임(`make_prompt`, `loadgen.py:195-197`). 이 prefix는 가정한 14토큰이 아니라 **실측 35토큰**(랜덤 hex가 잘게 쪼개짐) → 본문 987 + 35 = **최종 1021토큰**.

토큰은 글자도 단어도 아닌 **BPE 서브워드**다: "the"·"language"=1토큰이지만 "throughput"→`['through','put']`, "accelerator"→`['accel','erator']`처럼 긴 단어는 2토큰. 단, 문장 안에서는 앞 공백이 붙어(" throughput"=1토큰) 대부분 1토큰으로 줄어 평균이 1에 가깝다.

이 1024는 **목표값(추정)**이고, 진짜 고정값은 서버가 보고한 `usage.prompt_tokens`로 측정해 `in_tokens_mean`에 기록합니다(`loadgen.py:145,264`).
실제 결과(`results/*.json`)에선 전 배치 **약 1020토큰**(목표 1024 대비 트림 미발동+prefix 35토큰의 합으로 ~1021) — 두 플랫폼이 동일 텍스트·토크나이저라 입력 길이가 같게 맞춰져 공정 비교가 됩니다.

### 배치를 늘리는 동시 테스트 = closed-loop(고정 동시성)

"**항상 batch개 요청을 in-flight로 유지**"하는 closed-loop 방식입니다(open-loop 도착률 모델 아님).

- **배치 스윕**(`main_async`, `loadgen.py:282`): `[1,8,16,32,64,256]`을 순차로 돌며 각 b마다 `run_batch(batch=b)`; 배치 사이 `sleep(2)`로 서버 정리.
- **워커 b개**(`loadgen.py:212`): 각 워커가 `요청 1개 → 끝나면 즉시 다음`을 반복(`loadgen.py:200-202`)해 동시 요청 수를 항상 batch로 유지. httpx 동시연결도 `batch+8`로 개방(`loadgen.py:188`).
  - ⚠ 헷갈리기 쉬운 점: 워커 **1개의 while loop는 순차**(한 유저=한 번에 한 요청)지만, `asyncio.create_task`로 워커를 **batch개 동시에** 띄우고 각 워커가 `await`(네트워크 I/O)에서 이벤트 루프에 양보하므로 **워커들끼리는 동시**다. 동시성은 *loop가 아니라 워커 개수*에서 나옴(단일 스레드 I/O 동시성, 실제 추론 묶음 처리는 서버의 continuous batching). 바깥 배치 스윕 `for`만 순차.
- **워밍업**(`loadgen.py:215-216`): `warmup_s`(8s) 동안 결과를 버려 콜드스타트/램프업 제외.
- **count-based 측정창**(`loadgen.py:219-228`): 1Hz 전력 샘플링과 함께 `경과 ≥ window_s(30s) AND 완료수 ≥ max(batch,12)`까지 측정, 상한은 `window_s×3(90s)`. OSL=256 요청 하나가 30s보다 길어도 완료를 보장해 ok=0을 방지.
- **드레인**(`loadgen.py:229-231`): `stop=True` 후 워커들이 진행 중 요청만 마치고 종료.
- **집계**(`loadgen.py:235-269`): `agg_out_tps=총출력토큰/창길이`, `per_user p50`(요청별 out_tokens/decode_s 중앙값), TTFT p50/p90, 평균/최대 전력.

---

## 2. 실행 방법

### A) RNGD 서버 (이 레포가 있는 머신)
```bash
cd Model_Benchmark/bench-blog
./run_rngd.sh                      # 빈 카드 자동 serve → results/rngd.json
# 옵션: CARD=3 ISL=1024 OSL=256 BATCHES=1,8,16,32,64,256 WINDOW=45 ./run_rngd.sh
```

### B) PRO 6000 서버 (별도 GPU 머신)
1. 이 `bench-blog/` 폴더를 GPU 서버로 복사.
2. `setup_gpu.md` 보고 Blackwell 용 vLLM 설치(중요 — 구형 휠은 sm_120 미지원).
3. ```bash
   GPU=0 ./run_pro6000.sh           # vllm serve Qwen/Qwen3-32B-FP8 → results/pro6000.json
   ```

### C) 비교 리포트
```bash
# 두 JSON 을 한 곳에 모은 뒤:
python compare.py results/rngd.json results/pro6000.json --out results/report.md
#  → report.md (표) + charts/*.png (per-user TPS·집계TPS·tokens/s/W·TTFT)
```
GPU 결과 전이라도 `python compare.py results/rngd.json` 로 RNGD 단독 요약을 볼 수 있습니다.

---

## 3. 파일
| 파일 | 역할 |
|---|---|
| `loadgen.py` | 공통 부하시험 코어(OpenAI 호환). closed-loop 배치 스윕, 고정 ISL/OSL, TTFT/per-user TPS/집계 TPS + 전력 동시측정 |
| `power.py` | 전력 샘플러 — `furiosa-smi info`(RNGD) / `nvidia-smi`(GPU) 파싱, 1Hz 백그라운드 |
| `run_rngd.sh` | RNGD: 빈 카드에 qwen3-32b-fp8-tp8 serve → loadgen → `results/rngd.json` |
| `run_pro6000.sh` | PRO 6000: vllm serve Qwen/Qwen3-32B-FP8 → loadgen → `results/pro6000.json` |
| `compare.py` | 두 JSON → (A)raw users@SLO + (B)users/kW·tokens/s/W + TTFT + 차트 → `report.md` |
| `setup_gpu.md` | Blackwell(sm_120) vLLM 설치 3단 폴백 |

---

## 4. 결과 해석
- **(A) raw users@SLO 가 ~1.1x, (B) users/kW·tokens/s/W 가 ~2x** 면 블로그(전력정규화 1.8~2.0x)를
  방향·크기 모두 재현한 것입니다. raw 가 2x 나오길 기대하면 안 됩니다(그건 블로그도 전력정규화 후 값).
- per-user TPS 곡선이 SLO(20/30/40) 선과 만나는 동시성이 "그 SLO 의 최대 사용자". RNGD 곡선이
  더 오래 SLO 위에 머물고(사용자↑), 같은 처리량을 더 적은 와트로 내면(tokens/s/W↑) 블로그 주장이 성립.

## 5. 한계 (정직하게)
- **1장 vs 1대** 재현이라 블로그의 4x/8x·랙 절대수치(474명 등)는 재현 대상 아님 — **비율·방법론**만 재현.
- 블로그가 **시퀀스 길이·서빙 엔진(양측)·TP·정밀도·user 공식**을 공개 안 해 **절대 수치 일치는 불가**. 위 기본값은 합리적 선택일 뿐.
- RNGD 로컬 `qwen3-32b-fp8-tp8` 은 **weight-only FP8 + bf16 activation/KV**(artifact.json 기준). GPU 공식 FP8 는 W8A8. 더 엄격한 정밀도 매칭을 원하면 RNGD 도 prebuilt [`furiosa-ai/Qwen3-32B-FP8`](https://huggingface.co/furiosa-ai/Qwen3-32B-FP8)(W8A8) 사용 — 단 그건 tp32(4장)라 "1장" 의도와 어긋남.
- 블로그 수치의 1차 출처는 **Furiosa 자체 블로그 단일 소스**(독립 제3자 교차검증 없음).
- 전력은 카드 단위(furiosa-smi/nvidia-smi)만 측정 — 호스트(CPU/팬/PSU) 전력은 미포함. 블로그의 3kW/6.6kW 는 서버 전체(호스트 포함)로 추정되므로, 카드 단위 tokens/s/W 비교가 더 보수적입니다.
