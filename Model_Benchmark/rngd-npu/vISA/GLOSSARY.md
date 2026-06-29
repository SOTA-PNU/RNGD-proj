# vISA 용어집

이 문서는 vISA 용어집입니다. 모듈을 읽다 막히는 단어를 여기서 찾으세요.

## 칩·하드웨어
- **TCP (Tensor Contraction Processor)** — 퓨리오사 NPU 아키텍처. 곧 RNGD 칩. "텐서 축약(contraction)"에 최적화.
- **RNGD** — TCP를 구현한 실제 칩(우리 서버에 4장).
- **vISA (Virtual ISA, Virtual Instruction Set Architecture)** — TCP의 프로그래밍 인터페이스. 텐서 단위로 생각하되 메모리/스케줄은 직접 관리하는 "중간 높이" 언어. Rust로 작성.
- **Chip / Cluster / Slice / Lane** — 하드웨어 4계층. 칩 안에 클러스터 2개, 클러스터당 슬라이스 256개, 슬라이스당 레인 8개. 슬라이스 하나가 텐서유닛 하나를 돌림.
- **Tensor Unit (TU)** — 슬라이스마다 있는 고정 파이프라인. 한 사이클에 데이터를 단계별로 변환.
- **MAC (Multiply-Accumulate)** — 곱하고 더하는 기본 연산기. Lane이 MAC 배열의 한 행.
- **PE (Processing Element)** — SDK/드라이버가 칩을 부르는 단위. vISA에선 칩마다 pe0-7(양쪽 half-cluster 융합)로 묶임.

## 파이프라인 8단계 (엔진)
- **Fetch** — DM에서 데이터를 스트림으로 끌어옴(패딩·dtype 변환 가능).
- **Switch** — 슬라이스 사이로 데이터를 분배/전치(브로드캐스트 등). 유일하게 슬라이스를 가로지르는 엔진.
- **Collect** — 스트림을 **32바이트 flit**로 정규화.
- **Contraction** — 축약(곱-합). Outer→Packet→Time→Lane 하위 단계로 구성. TRF의 정지 피연산자를 읽음.
- **Vector** — 원소별 연산(활성함수, 정규화, 사칙). VRF를 매 사이클 읽음.
- **Cast** — dtype 변환(예: f32 누적값 → bf16).
- **Transpose** — flit 안에서 전치(within-flit 제약).
- **Commit** — 결과를 DM에 씀(절단/ReLU/valid-count 패킹 가능).

## 메모리 계층
- **HBM** — 패키지 위 대용량(48GB). 가중치·활성 장기 저장.
- **DM (Data Memory)** — 온칩 SRAM, 슬라이스당 512KB. 주 작업 메모리.
- **SPM** — 온칩 SRAM, 임시·중간값(컴파일러 관리, API 아직 제한적).
- **TRF (Tensor Register File)** — Contraction용, 레인당 8KB. `FirstHalf/SecondHalf/Full`로 주소 지정(더블버퍼링).
- **VRF (Vector Register File)** — Vector용, 슬라이스당 8KB. 자유 분할.
- **flit** — Collect가 만드는 32바이트 데이터 묶음.
- **packet** — Contraction Outer가 받는 64바이트(=flit 2개).

## 매핑(가장 중요한 개념)
- **mapping (매핑)** — "텐서 인덱스 ↔ 버퍼 위치"를 잇는 규칙. Rust **타입**으로 표현(`m![]`). `M` 트레잇이 크기(`SIZE`)와 `map()` 함수를 줌.
- **axes!** — 축 이름·크기 선언. `axes![A = 2048]`.
- **m![]** — 매핑 표현식. 연산자: `,`(Pair) `/`(Stride) `%`(Modulo) `#`(Padding) `=`(Resize) `1`(Identity) `{ }`(Escape).
- **Symbol** — 축 하나(`m![A]`). **Pair** — 축 합침(`m![A,B]`, 왼쪽 상위, 우결합). **Stride `/n`** — 바깥(블록) 인덱스. **Modulo `%n`** — 안쪽 위치. **Padding `#n`** — 하드웨어 단위로 패딩(남는 칸은 쓰레기값). **Resize `=n`** — 논리 크기 축소(절단). **Identity `m![1]`** — 1칸, Pair 항등원. **Escape `{X}`** — 별칭 끼우기.
- **Skew (`B' = B - A`)** — 대각 접근(wavefront). **Sliding (`$(...)`)** — 겹치는 블록(conv) 선형결합.
- **Spatial dimension(공간 차원)** — Chip/Cluster/Slice/Lane처럼 **하드웨어 유닛에 펼쳐지는** 축.
- **Temporal dimension(시간 차원)** — `Time`(파이프라인 반복 회차), `Packet`(한 회차 안 원소). 시간에 걸쳐 처리.

## 실행·스케줄
- **#[device(chip = N)]** — 디바이스 커널 함수 표시 매크로. 백엔드에 따라 본문이 CPU 시뮬 또는 NPU EDF 실행으로 갈림.
- **launch(kernel, (&mut ctx, &args))** — 호스트에서 커널 실행.
- **Context** — `ctx.main`(주 연산)·`ctx.sub`(프리페치)·`ctx.tdma`(텐서 DMA)·`ctx.pdma`(PCIe DMA). 다른 컨텍스트끼리 병렬.
- **Hazard (해저드)** — 같은 주소 데이터 의존. RAW(쓰고 읽기)/WAR(읽는 중 덮기)/WAW(연달아 쓰기). 스케줄러가 주소 분석으로 자동 대기.
- **Double-buffering** — TRF 반쪽씩 번갈아 써서 프리페치와 연산을 겹침.
- **Sequencer** — 파이프라인 반복을 구동(8 entries / 65536 iters 한도).
- **VCG (Valid Count Generator)** — 유효 원소 개수로 분기/태그를 만드는 유닛.

## 백엔드·산출물
- **backend** — `simulation`(호스트 실값, 기본)·`typecheck`(모양만)·`emulation`·`npu`(실하드웨어).
- **EDF / `.bin`** — vISA 컴파일러의 최종 산출물(`MIR→VISA→LIR→EDF`). 실제 파일은 `.bin`(pert-ipc 명령어 스트림). `libdevice_runtime.so`가 로드.
- **`.edf` (furiosa-llm)** — ⚠️ 이름은 같지만 **다른 것**. furiosa-llm의 `.edf`는 CBOR 그래프 IR. vISA `.bin`과 포맷 호환 안 됨.
- **DPE** — Dot-Product Engine. Contraction의 핵심(우리 측정상 matmul에 ~0.23% 정밀도 지문).

## 연산·패턴
- **Tensor contraction** — 행렬곱의 일반화. 공유 축으로 곱-합. **Broadcast → Multiply → Reduce** 3단계.
- **einsum** — 축약을 간결히 쓰는 표기(예: `IK, KJ → IJ` = GEMM).
- **GEMM / GEMV / dot product** — 행렬×행렬 / 행렬×벡터 / 벡터 내적.
- **TopK / one_hot / histogram / cumsum** — MoE 라우팅을 프리미티브로 조립할 때 쓰는 패턴.
- **ReduceScatter / split-K** — 큰 행렬곱을 여러 슬라이스/클러스터/칩에 쪼개 부분합을 모으는 분할 전략.
- **Tiling** — 512KB/slice를 넘는 텐서를 시간(순차)·공간(병렬)으로 타일 분할.
