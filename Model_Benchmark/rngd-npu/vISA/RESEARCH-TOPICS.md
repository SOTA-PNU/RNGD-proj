# vISA 논문 주제 조사 결과

> 부산대 RNGD 서버에서 2026-07 에 수행한 vISA 실기 조사([`book_guide/`](./book_guide/))를 근거로
> 논문이 될 만한 주제 7개를 뽑아, 각각 **조사 담당**과 **적대적 심사 담당**을 붙여 검증한 기록이다.
> 주제별 상세는 논문 작성 순서(초록 → 문제의식 → 기여 → 방법 → 평가 → 반론 → 학회)로 정리했다.

## 0. 이 문서를 읽는 법

### 0.1 어떻게 조사했나

주제마다 두 사람을 붙였다.

1. **조사 담당** — 로컬 문서(`book_guide/`, `book_ko/`, `_evidence/logs/`)를 직접 읽어 수치를 확인하고,
   웹 검색으로 선행연구를 찾아 신규성을 판정한다. 기억으로 인용하지 말고 **실제로 찾은 것만** 적게 했다.
2. **적대적 심사 담당** — 그 제안을 **떨어뜨리려고** 읽는다. 조사 담당이 적은 수치를 파일에서 재확인하고,
   이미 출판된 논문을 찾아내고, reject 사유를 나열한다.

총 14명, 실패 0. 검색으로 확인된 선행연구 75건, 제기된 치명적 반론 77건.

### 0.2 ★ 결과를 읽을 때의 편향 고지

**7개 주제 전부 심사에서 `survives = false` 가 나왔다.** 그런데 나는 심사 담당에게
"떨어뜨리려고 읽어라"라고 **명시적으로 지시했다**. 그러니 전멸이라는 결과 자체는
그 지시의 산물이기도 하며, "이 주제들은 전부 가망 없다"는 뜻으로 읽으면 안 된다.

읽어야 할 것은 판정이 아니라 **구체적 반론**과 **살리는 길(§n.9)** 이다.
실제로 심사 담당 여럿이 "이 형태로는 안 되지만 이렇게 바꾸면 된다"는 경로를 스스로 제시했고,
그중 몇 개는 **이미 확보된 근거만으로 1.5~2개월이면 워크숍 논문이 된다**고 적었다.

### 0.3 근거의 층위 — 이 구분을 놓치면 전부 무너진다

| 층위 | 뜻 | 예 |
|---|---|---|
| **실측** | 실제 하드웨어에서 돌려 관측 | 테스트 89개 격리 실행 → 80 통과, MNIST 10장 정답 |
| **도구 산출** | 컴파일러/드라이버가 뱉은 값 | 커널 200개 중 137 컴파일 성공, `furiosa-smi` 카드 정보 |
| **모델 예측** | 컴파일러 스케줄 모델의 계산값 (**벽시계 아님**) | 사이클의 96.5% 가 DMA, `mnist::forward` 17,953 cycle |
| **문서 서술** | 책이 그렇게 적었을 뿐, 소스·실기 대조 안 함 | 일부 엔진 내부 동작 |
| **미확인** | 확인하지 못했다고 명시한 것 | `total_instruction_cycle` 의 설명 안 되는 2,500 사이클 |

**성능과 관련된 거의 모든 수치가 "모델 예측" 층에 있다.** 이 문서의 주제 2와 4가 그 위에 서 있고,
심사에서 가장 세게 얻어맞은 지점도 정확히 여기다.

---

## 1. 먼저 알아야 할 사실 — 벤더가 이미 논문을 냈다

조사 중 확인한 것으로, 주제 선정 전체를 좌우한다.

| 논문 | 위치 |
|---|---|
| Tensor Contraction Processor 아키텍처 | **ACM/IEEE ISCA 2024** |
| "FuriosaAI RNGD: A Tensor Contraction Processor for Sustainable AI Computing" | **IEEE Micro** (Hot Chips 2024 특집) |

**따라가는 결론 두 가지.**

1. **아키텍처 개관 각도는 이미 선점됐다.** "RNGD 는 이런 칩이다"류 논문은 낼 자리가 없다.
   회로 스위치 기반 fetch 네트워크, 입력 브로드캐스트, 버퍼 기반 재사용 같은 핵심 기여가 이미 그 논문들에 있다.
2. **그러나 `m![]` 매핑 표현식·타입 수준 인코딩·합법성 판정은 그 논문들에 없다.**
   조사 담당이 IEEE Micro 논문 PDF 8쪽을 전문 확인했다. 즉 **프로그래밍 모델 쪽은 비어 있다.**

우리가 낼 수 있는 자리는 아키텍처가 아니라 **그 위에서 프로그램을 어떻게 쓰고 무엇이 실제로 되는가** 쪽이다.

출처: [FuriosaAI TCP 블로그](https://furiosa.ai/blog/tensor-contraction-processor-ai-chip-architecture) ·
[IEEE Xplore](https://ieeexplore.ieee.org/document/10929037/) ·
[IEEE Micro PDF](https://web.ist.utl.pt/nuno.lopes/pubs/tcp-micro25.pdf) ·
[Chips and Cheese 해설](https://chipsandcheese.com/p/furiosaais-rngd-at-hot-chips-2024-accelerating-ai-with-a-more-flexible-primitive)

---

## 2. 조사 중 확인된 사실 정정 — Skew·Sliding 은 존재하지 않는다

심사 담당이 제기해 **내가 직접 재확인**한 것이다. 우리 문서를 고쳐야 한다.

```console
$ R=~/.cargo/registry/src/index.crates.io-*/
$ grep -rl "Skew\|Sliding" $R/furiosa-mapping-types-0.4.0 $R/furiosa-mapping-macro-0.4.0 $R/furiosa-opt-std-0.4.0 | wc -l
0
$ grep -rl "Skew\|Sliding" $R/furiosa-mapping-types-0.3.0 $R/furiosa-mapping-macro-0.3.0 $R/furiosa-opt-std-0.3.0 | wc -l
0
$ grep -n "^    [A-Z][A-Za-z]* {" $R/furiosa-mapping-types-0.4.0/src/lib.rs   # Mapping enum 변종
Symbol / Stride / Modulo / Resize / Padding / Pair / Broadcast      ← 7개뿐
```

| 확인한 것 | 결과 |
|---|---|
| `Mapping` enum 변종 (0.4.0) | **7개**: Symbol · Stride · Modulo · Resize · Padding · Pair · **Broadcast** |
| `Skew` / `Sliding` 구현 | **0.3.0 · 0.4.0 어디에도 없다** (문자열 0건, 매크로 렉서에 `$` 토큰 없음) |
| 현재 책(46개 절, `book_ko/`) | **Skew·Sliding 을 다루지 않는다** (해당 절이 삭제됨) |
| 옛 영문 스냅샷 `reference/book/mapping-expressions.md:379` | **"Skewed axis" 절이 남아 있다** ← 출처 |

즉 **Skew·Sliding 은 옛 판 책에만 있던 서술이고, 구현된 적이 없으며, 현재 책에서도 빠졌다.**
`book_guide/02-매핑-텐서.md:18` 이 이미 "소스 대조 안 함"으로 미확인 표시해 둔 항목이 이렇게 해소됐다(부정으로).

**고쳐야 할 곳** (아직 안 고쳤다):

- `vISA/GLOSSARY.md:38` — "**Skew (`B' = B - A`)** — 대각 접근(wavefront). **Sliding (`$(...)`)** — 겹치는 블록(conv) 선형결합."
- `vISA/CHEATSHEET.md`, `vISA/curriculum/02_mapping.md`, `vISA/curriculum/06_computing_engines_1.md`
- `GLOSSARY.md:36` 의 연산자 목록에 **`Broadcast` 가 빠져 있다**

> 이것이 이번 조사가 준 부수 소득이다. 논문 주제를 검증하려고 소스를 판 결과, 우리 학습 문서의
> 틀린 서술 하나가 잡혔다. (앞서 같은 방식으로 `GLOSSARY.md` 의 backend 항목도 정정한 바 있다.)

---

## 3. 결론 요약

| # | 주제 | 초기 | 심사 후 | 핵심 반론 (한 줄) | 살리는 길 | 기간 |
|---|---|:--:|:--:|---|---|---|
| 1 | 매핑 대수 형식화·합법성 판정 | viable | **weak** | 차별점이던 Sliding 이 미구현. 엔트리 병합 규칙 = CuTe coalesce 재발견 | '대수 형식화'를 버리고 **명세-구현 적합성 차등 퍼징**으로 축 회전 | 9~12개월 (축소 시 6~8) |
| 2 | 스케줄 모델 정확도 검증 | viable | **weak** | **벽시계 계측 수단이 확보되지 않았다.** 모델을 모델로 검증할 수는 없다 | 계측 go/no-go 게이트를 먼저 통과 (4주) | 5~8개월 |
| 3 | 로워링 공백 실증 특성화 | viable | **weak** | 단일 벤더·사전출시 소프트웨어. 코퍼스가 벤더 예제뿐 | 두 번째 컴파일러 스택 추가. **워크숍이면 1.5개월** | 6~9개월 |
| 4 | DMA 지배 → 매핑 최적화 | weak | **not-a-paper** | 관찰 자체가 기존 결론이고, 근거가 실측이 아니라 모델 예측이다 | **비용모델 감사**로 선회 (주제 2와 합류) | 6~9개월 |
| 5 | 툴체인 평가 방법론 | weak | **not-a-paper** | 방법론 단독으로 내려면 툴체인 2개 이상 + 반복 통계가 전제 | 다른 논문의 **§방법론 + 아티팩트**로 (2~4주) | 6~9개월(단독) |
| 6 | 정렬·레인 제약 타입화 | weak | **not-a-paper** | 기여가 얇고 기존 shape-type 연구와 겹친다 | 워크숍/툴 페이퍼로 축소 | 3~5개월 |
| 7 | 동적 형상 (VCG·MoE) | weak | **not-a-paper** | 실험이 **벤더 컴파일러 버그에 막혀 있다** (transformer 4종 ICE 등) | 표현력 형식화만 떼어 워크숍 | 1.5~2개월(축소) |

### 3.1 내 판단 — 7편을 쓰려 하지 말고 하나로 묶어라

심사 결과를 관통하는 신호가 하나 있다. **개별 주제는 전부 얇은데, 근거는 모아 놓으면 두껍다.**
같은 서버·같은 툴체인·같은 코퍼스에서 나온 것이라 하나의 논문으로 묶을 때 오히려 강해진다.

**권장 A — 통합 실측 논문 (현실적 1순위, ISPASS 또는 IISWC, 6~9개월)**

> *"상용 사전출시 NPU 텐서 eDSL 툴체인의 실측 특성화"*
> 본문 = 주제 3(로워링 공백 63/200 + 실기 89 매트릭스) · §방법론 = 주제 5(게이팅·격리·값검증) ·
> 성능 절 = 주제 4의 선회안(비용모델 감사) · 가능하면 주제 2(벽시계 대조)까지.

묶어야 하는 이유: 주제 3은 "왜 이 코퍼스만?"에 약하고, 주제 5는 "방법론만으로?"에 약한데,
**서로가 서로의 약점을 메운다.** 방법론이 결과를 바꿨다는 것을 결과로 증명하는 구조가 된다
(프로세스 격리로 통과 수가 10 → 33 으로 3배 달라진 사례).

**권장 B — 워크숍 논문 (즉시 착수 가능, 1.5~2개월)**

이미 확보된 근거만으로 쓸 수 있다. LATTE / C4ML / CGO 워크숍 급.
권장 A 의 예고편 역할을 하면서 조기 피드백을 받는다.

**권장 C — 장기 고위험 (CGO 또는 CC, 9~12개월)**

주제 1을 심사 담당이 제안한 대로 **적합성 차등 퍼징**으로 회전시킨 것.
결정적 자산은 **`M` 트레잇이 순수 CPU 오라클이라는 점**이다 — `map()` 은 하드웨어 없이 `cargo test` 로 돌고,
같은 표현식이 폐쇄 소스 백엔드로도 내려간다. **공짜 오라클이 딸린 차등 퍼징 환경**이고,
실제로 이번 조사에서 조용한 데이터 오배치 2건을 잡아낸 것도 이 구조 덕이다.
분모가 벤더 예제 200개가 아니라 **생성된 매핑 수만 개**가 되면 "단일 벤더" 반론이 크게 약해진다.

### 3.2 주제 간 의존관계 — 순서를 틀리면 헛수고가 된다

```
주제 2 (스케줄 모델이 맞는가?)
        │
        │  모델이 틀렸다면 아래 주장이 전부 무너진다
        ▼
주제 4 (DMA 가 96.5% 니 매핑을 최적화하자)
```

**주제 4의 "DMA 96.5%" 는 주제 2가 검증하려는 바로 그 모델이 낸 값이다.**
모델 검증 없이 그 위에 최적화 논문을 얹으면, 심사에서 "당신 전제가 미검증"으로 한 방에 무너진다.
그래서 **주제 2가 주제 4보다 먼저**다. 심사 담당이 주제 4를 `not-a-paper` 로 내린 이유이기도 하다.

---

# 주제별 상세

> 아래는 조사·심사 담당이 낸 내용을 논문 작성 순서로 정리한 것이다. 요약이 아니라 원문 기반이다.

---

## 주제 1. 매핑 표현식 대수의 형식 의미론과 합법성 판정·합성

> **판정: `viable` → 적대적 심사 후 `weak`**  (약함 — 독립 논문보다 한 절이 낫다)
> 조사자가 찾은 선행연구 15건 · 심사자가 제기한 치명적 반론 11건 ·
> "이미 출판됨" 지적 10건 · 근거 없이 단정한 문장 13건

### 1.1 초록 — 한 문장 주장

vISA 매핑 표현식은 "버퍼 위치 → 텐서 인덱스"의 부분·다치 관계를 주는 작은 대수이며, 이 대수의 표시의미론을 기계화하고 파이프라인 단계별 하드웨어 실현가능성(sequencer 엔트리 8개·반복 65,536·flit 32B·정렬·용량)을 판정하는 결정 절차를 세우면, 타입체크는 통과하지만 실기 낮추기에서 63/200이 실패하고 그중 2건은 오류 없이 데이터를 잘못 놓는 현행 벤더 컴파일러의 불완전성을 형식적으로 메우고 자동 수리까지 할 수 있다.

### 1.2 문제의식과 선행연구의 빈틈

검색으로 확인한 바: (1) 벤더 자신의 논문(ISCA 2024 TCP, IEEE Micro 45(3) 2025 RNGD)은 순수 아키텍처·컴파일러 개관이고 m![] 매핑 표현식·타입 수준 인코딩·합법성 판정은 **한 글자도 없다** — PDF 8쪽 전문 확인. 즉 이 대수 자체는 미출판이다. (2) 그러나 "레이아웃 대수의 형식화"라는 문제 부류는 이미 붐빈다: CuTe 레이아웃 대수에 대해 2026년에만 형식화 논문 2편(Cecka의 표현·대수, Carlisle/Shah/Stern의 범주론적 기초)이 나왔고, Shah의 2024 노트는 composition/complement/division 의 **well-defined 조건**(=합법성 판정)을 이미 준다. (3) 합성 쪽도 Hexcute 가 "레이아웃 합성을 제약 프로그래밍으로 형식화하고 타입추론 기반 알고리즘으로 푼다"를 GPU 에서 이미 했다. 남는 진짜 공백은 세 가지다: (a) CuTe 레이아웃은 좌표→오프셋 **함수**이고 well-definedness 를 위해 비퇴화성을 요구하는데, vISA 의 Sliding 은 하나의 버퍼 인덱스가 여러 텐서 인덱스로 가는 **관계**를 1급으로 둔다(책이 명시); (b) Padding 이 Top(임의값)/Zero(0 보장)/Bottom(접근 시 UB) 3종 격자라 슬롯에 대한 지식·권한 주석이 붙는데 CuTe·Timeloop 계열에 대응물이 없다; (c) 합법성이 메모리 레이아웃 하나가 아니라 **고정 파이프라인 단계들의 실현가능성 논리곱**이다 — sequencer 설정으로의 컴파일(엔트리 ≤8, 병합 규칙 s1==n2·s2, 반복 ≤65,536, 패킷 크기 ∈{1,2,4,8,16,32}B, 최내곽 엔트리 조건), Collect 의 32B flit 정규화, Buf/스트림 축 분해의 공통세분 존재(gcd) 조건, 공간 차원 정확일치. 이 조합을 판정 문제로 세운 선행연구는 못 찾았다.

### 1.3 제안한 기여

- 매핑 대수의 완전한 표시의미론 기계화: M::SIZE + map(i)->Option<Index> 의 함수적 조각과 Sliding 의 관계적 조각(S,e ⊢ si ~ ti)을 하나의 의미론으로 통합하고, Broadcast<N>·Padding 3종(Top/Zero/Bottom)·Resize·Skew 를 모두 포함해 Rocq/Lean 으로 증명. 벤더 문서 자체가 M 트레잇(함수)과 Sliding 절(비일대일)에서 서로 모순되는 지점을 해소하는 것이 출발점이다.
- 매핑 동치의 기호적 결정 절차: 현행 정의는 'SIZE 같고 모든 i 에서 map(i) 같음'이라 소박하게는 O(SIZE)(실제 커널에서 10^5~10^6)다. 책이 제시한 6개 항등식(Pair 항등원, stride-modulo 분해, Pair 사영, 결합법칙, 멱등, %1≡1)이 완전한 등식 이론인지 판정하고, 열거 없는 정규형 알고리즘과 완전성 증명을 준다.
- 파이프라인 실현가능성 판정: 매핑 표현식 → sequencer Config(엔트리 리스트) 확장, 인접 병합 s1==n2·s2 의 합류성·정규형 증명, 그리고 '병합 후 엔트리 ≤8 ∧ 각 size ≤65,536 ∧ 최내곽 엔트리가 (s∈{0,1} ∧ n%packet_size==0) 또는 packet_size==1' 을 다항시간에 판정. Buf/스트림 공통세분 존재(gcd 기반) 판정도 같은 틀에 넣는다.
- 불합법 매핑의 자동 수리: 등식 이론 안에서 의미보존 재작성으로 합법 매핑을 탐색하고, 불가능하면 최소 변경(패딩 추가/분해 변경/sequencer 호출 분할)을 제안. Hexcute 류의 전면 합성 대신 '수리'로 범위를 좁혀 einsum→매핑 합성과의 중복을 피한다.
- 벤더 컴파일러에 대한 차등 검사 실증: 예제 커널 200개에서 --backend typecheck 는 커널 낮추기 오류 0건인데 --backend npu 는 63개 실패(137 OK, 68.5%)라는 불완전성 격차를 정량화하고, 특히 컴파일도 실기 로드도 통과하면서 조용히 데이터를 잘못 놓는 2건을 검사기가 사전에 잡아내는지로 평가한다.

### 1.4 방법

1) 구문·의미론 확정: book_ko/src/mapping-tensors/mapping-expressions.md 의 생성자 정의와 furiosa-mapping-types-0.4.0/src/dsl.rs, furiosa-mapping-macro-0.4.0 parser.lalrpop 을 대조해 실제 구현된 생성자 집합을 확정한다(Skew·Sliding·간접 시퀀싱은 아직 소스 대조 미완). 의미 영역은 Index = 축-원자 → 값의 합(Term{Atom, stride, modulo})이며 Pair 가 부분 인덱스를 더한다는 점에서 준아핀(quasi-affine) 인덱스 대수에 해당하므로, 다면체 모델의 정수 집합/관계(Presburger)로 임베딩해 결정성을 얻는다. Sliding 은 $(e1:n1,...,ed:nd) 의 크기 1+Σ(size(ek)-1)·nk 와 si=Σ sik·nk 관계를 그대로 Presburger 관계로 쓴다.
2) 하드웨어 제약을 단계별 술어로 형식화: Fetch/Collect(32B flit 정규화: 32B 경계 패딩 → flit 분할, 바깥 flit 수를 Time 으로 흡수), Sequencer(엔트리 확장·병합·한도), 공간 차원(Chip==NUM_CHIPS, Cluster==2, Slice==256, Lane≤8), 용량(DM 512KB/slice, TRF 8KB/lane, VRF 8KB/slice), 정렬(시작주소가 size_of::<D>() 배수, DMA 8B 정렬). 각 술어는 매핑 표현식에서 순수하게 계산 가능해야 한다(하드웨어 불필요).
3) 판정 알고리즘: 엔트리 병합은 인접쌍 국소 재작성이므로 합류성을 증명해 최대병합 정규형을 O(k) 에 얻고, 실현가능성을 그 정규형 위의 부등식 검사로 환원한다. 축 분해 호환성은 두 분해의 공통세분 존재 문제로 환원해 gcd/격자 판정으로 푼다.
4) 구현: Rust 로 checker 를 짜서 m![] 타입에서 직접 돌린다. SIZE/map 이 순수 CPU 코드라 NPU·래퍼 없이 cargo test 로 오라클 대조가 된다.
5) 수리: 등식 이론을 e-graph(egg)로 포화시키고 실현가능성 술어를 비용함수로 추출한다.
6) 일반화: 같은 술어 틀에 CuTe 의 complement/composition/division well-definedness 조건과 NKI 타일 제약(partition ≤128, free ≤64K SBUF/4K PSUM)을 인스턴스로 넣어 단일 벤더 반론에 답한다.

### 1.5 평가 설계

베이스라인은 (B1) 벤더의 타입 수준 검사(const assert + --backend typecheck), (B2) 벤더 백엔드의 낮추기(--backend npu), (B3) 실기 실행 결과(furiosa-smi 격리 실행 89 테스트). 지표: (1) **합법성 판정 정확도** — 커널 200개에 대해 검사기 판정 vs B2 결과의 혼동행렬. 특히 24개 REAL_LOWERING_GAP 을 사전에 예측하는가, 23개 INTENTIONAL_NEGATIVE 를 올바르게 거부하는가, 13개 COMPILER_ICE 를 '컴파일러 버그(우리 기준 합법)'로 분리하는가. (2) **B1 대비 정밀도 향상** — B1 은 63건 중 0건을 잡는다(typecheck 커널 낮추기 오류 0건). 검사기가 잡는 건수/63 이 직접적인 개선치다. (3) **B2 가 놓치는 것을 잡는가** — 컴파일·로드 통과 후 조용히 틀리는 2건(broadcast 목적지 미기록 2048/2048 불일치, tile 커밋이 result[32..64] 대신 result[8..40] 착지)과 로드 abort 3건(50,560 vs 33,792 / 56,576 vs 37,888)을 정적으로 거부하는가. 여기서 이기면 논문이 선다. (4) **수리 성공률** — 24개 REAL_GAP 중 의미보존 재작성만으로 --backend npu 컴파일을 통과시킨 비율, 그리고 통과한 커널의 실기 값 일치 여부. (5) **비용** — 검사기 판정 시간 vs 벤더 컴파일 시간, 그리고 소박한 열거식 동치 검사(O(SIZE)) 대비 속도. (6) **성능 영향(부수)** — --dump-schedule 로 수리 전후 사이클 비교. 단 PeCore 는 전체의 3.3%, DmaEngine 이 96.5%(커널 130개 합산, 중앙값 82.8%)이므로 성능 주장은 반드시 DMA 사이클 기준으로만 한다. (7) **일반화** — 같은 프레임워크로 CuTe well-definedness 조건 재유도, NKI 타일 제약 인스턴스화. 방법론 주의: 실기 결과는 반드시 테스트마다 별도 프로세스로 격리한다(단일 프로세스면 vector_engine 이 10통과/25실패, 격리하면 33/3 — 행 커널 하나가 HAL -110 으로 프로세스를 오염시킨다).

### 1.6 이미 확보된 근거

- 대수의 표시의미론이 이미 반쯤 문서화돼 있다: M 트레잇은 const SIZE 와 map(i)->Option<Index> 둘로 의미를 정하고, 동치는 'SIZE 같음 + 모든 i 에서 map(i) 같음'으로 정의되며, 6개 항등식(Pair 항등원, stride-modulo 분해 E ≡ m![{E}/n, {E}%n], Pair 사영, 결합법칙, 멱등 /1·#SIZE·=SIZE, E%1≡m![1])이 명시돼 있다. 출처: book_ko/src/mapping-tensors/mapping-expressions.md:18-48, 601-617.
- 각 생성자의 SIZE/map 구현식이 소스와 문자 그대로 대조 확인됐다(furiosa-mapping-types-0.4.0/src/dsl.rs): Symbol 66-83, Pair 220-244(map(i)=L::map(i/R::SIZE)+R::map(i%R::SIZE)), Stride 104-127, Modulo 133-156, Resize 162-183, Padding 193-214. Stride/Modulo 는 assert!(L::SIZE % SIZE == 0)을 const 평가로 강제해 m![J/7](J=512)은 컴파일 실패한다. 우선순위·결합성도 parser.lalrpop:41-81 로 확인(','가 가장 느슨·우결합, / % = # 은 동일 우선순위 좌결합 → C%32/2 = (C%32)/2). 출처: book_guide/02-매핑-텐서.md:59-145.
- Sequencer 하드웨어 한도가 정확한 수치로 문서화돼 있다: 설정당 엔트리 최대 8개, 엔트리당 size ≤ 65,536, 패킷 크기 ∈ {1,2,4,8,16,32} 바이트, 최내곽 엔트리 n:s 는 (s==0||s==1)&&n%packet_size==0 이거나 packet_size==1. 병합 규칙은 (n1:s1),(n2:s2) → (n1·n2:s2) iff s1==n2·s2. 실제 워크드 예제: m![W/16,H%2,H/2,C/2,C%2,N/2,N%2,W/8%2,W%8] 이 엔트리 9개를 만들고 병합으로 6개가 된다(H%2(2:32)와 H/2(4:64)는 32≠4×64라 병합 불가). 출처: book_ko/src/moving-tensors/sequencer.md:468-509, 598-609.
- 축 분해 호환성이라는 별도 판정 조건이 있다: Buf m![A%5, A/5] 와 스트림 m![A%3, A/3] 은 A=15 로 원소 수가 같은데도 gcd(5,3)=1 이라 공통 세분이 없어 설정이 거부된다. 출처: sequencer.md:612-640.
- Collect 는 임의 크기 패킷을 정확히 32바이트 flit 로 정규화한다(32B 경계까지 패딩 → flit 경계에서 분할, 안쪽 32B 는 Packet2, 바깥 flit 수는 Time2 로 흡수). 하위 엔진 전부(Contraction/Vector/Cast/Transpose/Commit)가 32B flit 만 소비한다. 출처: book_ko/src/computing-tensors/collect-engine.md:3-8, 136-178.
- 공간 차원 제약이 하드웨어 개수와 정확일치를 요구한다: Chip::SIZE==NUM_CHIPS, Cluster::SIZE==2, Slice::SIZE==256, Lane::SIZE<=8. 용량은 DmTensor ≤512KB/slice, TrfTensor ≤8KB/lane, VrfTensor ≤8KB/slice. 시작 주소가 size_of::<D>() 배수가 아니면 read-modify-write 로 약 50× 느려진다. 출처: book_ko/src/mapping-tensors/spatial-temporal-dimensions.md:56-79.
- ★ 타입 수준 검사의 불완전성이 정량화돼 있다: 벤더 예제 크레이트를 --backend typecheck 로 빌드하면 커널 낮추기 오류 0건인데, 같은 크레이트를 --backend npu 로 빌드하면 소스 추출 커널 200개 중 63개 실패(137 OK, 68.5%)하고 그 63개 때문에 크레이트 전체 빌드가 죽어 테스트가 0개 실행된다. 출처: book_guide/02-매핑-텐서.md:375-382, _GROUND_TRUTH.md N1·N6.
- 실패 63개의 전수 분류가 이미 돼 있다(적대적 재검증 포함, 보정 전 64개 목록 기준): REAL_LOWERING_GAP 24, INTENTIONAL_NEGATIVE 23, COMPILER_ICE 13, GENERIC_NOT_MONOMORPHIZED 2, UNCLEAR 2. 즉 '일부러 실패하는 표본'은 절반이 안 되고 나머지 40개는 진짜 공백이거나 컴파일러 자체 버그다. 출처: _GROUND_TRUTH.md N6-1, 13-NPU-실기-매트릭스.md §7.1.
- 실패 메시지가 대부분 매핑 모양이다: 'mir: Collect time mismatch. Expected: A / 4 % 4, got: A / 4', 'mir: commit_trim packet mismatch. Expected A % 4 # 8 or a trimming of it, got A % 8', 'lir: incorrect buffer size at T7: buffer.size() (256) != num_chips * intra_chip_size (228)', 'strides([8,128,4,...]) is not aligned by 8'(DMA 시퀀서 8B 정렬), 'tail_size % min_align (1) != 0'. 출처: book_guide/02-매핑-텐서.md:376-381, 13-NPU-실기-매트릭스.md:475-480.
- ★ Padding 생성자의 의미론과 백엔드 크기 계산이 실제로 어긋나는 구체 사례가 있다: contract_outer_assertions::lane_size::valid_size_{1,2,4} 는 출력 매핑이 각각 m![A, 1 # 8], m![A, R/4 # 8], m![A, R/2 # 8] 로 셋 다 타입 수준에서는 256 B 인데 백엔드는 228/232/240 으로 센다 — 부분 충전 레인 그룹의 꼬리 패딩이 DRAM 크기 계산에서 누락된다. 이름이 valid_* 인데 실기 컴파일에 실패한다. 출처: book_guide/02-매핑-텐서.md:382, _GROUND_TRUTH.md N6-3.
- ★ 컴파일 성공이 실기 정확성을 보장하지 않는 사례 2건이 값 수준으로 확인됐다(가장 강한 동기): broadcast::test_view_broadcast 은 HBM→HBM 브로드캐스트 DMA 가 목적지에 아무것도 쓰지 않아 2048/2048 전부 불일치하고 2회 실행에서 값이 동일(결정적 잔류 데이터). tile_tests::test_tile_window_commit_host 는 데이터는 온전한데 result[32..64] 에 가야 할 input[0..32] 가 result[8..40] 에 착지 — 32 가 '요소 수' 대신 '바이트 수'로 적용된 서명(32 elem × 4 B = 128 B 여야 하는데 실제 착지 8 elem = 32 B). 둘 다 에뮬레이션에서는 통과한다. 출처: _GROUND_TRUTH.md N5-②.
- compile 성공 ≠ 실기 로드 성공: reshape ×2 와 chip_shuffle 은 컴파일 137 OK 에 들어가는데 커널 로드에서 요구 크기가 실제의 약 1.5배(50,560/33,792, 56,576/37,888)라 프로세스가 abort 한다. 출처: _GROUND_TRUTH.md N5-①, N9.
- 실기 실행 경로와 오라클이 이미 확보돼 있다: NPU 백엔드 테스트 89개를 격리 실행해 80 PASS / 5 FAIL / 3 ABORT / 1 ignore(실질 83/89, 93.3%)를 기록했고 자원 누수 0. 재사용 도구(npu_matrix.sh, classify_mismatch.py, gate_kernels.py, sched_scan.sh)도 있다. 출처: _GROUND_TRUTH.md N2·N3·N12.
- 매핑의 SIZE/map 은 순수 CPU 코드라 NPU·래퍼 없이 cargo test 로 검증된다 — 즉 형식 의미론의 실행 가능 오라클을 하드웨어 없이 만들 수 있다. 실제 단위 테스트 예가 문서화돼 있다(m![I/32,J/32]::SIZE==256, m![I%32,J%32]::SIZE==1024, m![J/64,J%64] 가 모든 i 에서 m![J] 와 일치). 출처: book_guide/02-매핑-텐서.md §8.
- 사이클 귀속 실측: 실기 컴파일되는 커널 130개 합산에서 DmaEngine 75,464,336 사이클(96.5%, 인스트럭션 470개) vs PeCore 2,586,167(3.3%, 1,557개), 커널 130개 중 107개(82%)가 DMA 50% 이상, 중앙값 82.8%. → 레이아웃·이동이 지배하므로 매핑 합법성/수리의 성능 서사는 DMA 기준으로만 세워야 하고, 슬라이스 내부 데이터패스 최적화 상한은 3.3%다. 출처: _GROUND_TRUTH.md N4.
- 벤더 공식 논문에 이 대수가 없음을 1차 확인: IEEE Micro 45(3), 2025(Hot Chips 2024 테마) 'FuriosaAI RNGD: A Tensor Contraction Processor for Sustainable AI Computing' PDF 8쪽 전문을 읽었다. 매핑 표현식·타입 수준 인코딩·합법성 판정은 전무하고, 컴파일러 서술은 '축 순열로 구성된 tactic space 탐색', 'fetch unit 이 N차원 루프 방식으로 메모리를 읽는다' 수준이다. ISCA 2024 TCP 논문도 아키텍처·설계공간 탐색 개관이다.

### 1.7 아직 없는, 반드시 해야 할 실험

- Skew(B'=B-A)·Sliding($(...))·'간접 시퀀싱'의 0.4.0 실제 구현 소스 대조. 현재 book_guide/02-매핑-텐서.md:18 이 '책 서술만 인용, 소스 대조 안 함'이라고 명시하고, mapping-expressions.md:563 의 '간접 시퀀싱' 절은 제목만 있고 본문이 비어 있다. 이 세 개가 실제로 구현돼 있는지가 논문의 흥미로운 조각 전부를 좌우한다 — 미구현이면 평가 불가.
- M 트레잇의 함수적 시그니처(map(i)->Option<Index>)와 Sliding 의 관계적 의미론(S,e ⊢ si ~ ti, 공간 인덱스 4가 {4_N},{2_N,1_F},{2_F} 로 동시에 매핑) 사이의 모순 해소. 어느 쪽이 진짜 의미론인지 구현으로 확정하고, 함수적 조각/관계적 조각의 경계를 형식적으로 그어야 한다.
- 표시의미론의 기계화(Rocq 또는 Lean): Symbol, Pair, Broadcast<N>(Identity 는 Broadcast<1> 의 특수화), Stride, Modulo, Resize, Padding<Top|Zero|Bottom>, Skew, Sliding 전부. 현재 작성된 형식 정의는 0줄이다.
- 동치 판정의 결정성·복잡도 증명과 6개 항등식의 완전성 판정. 소박한 정의는 O(SIZE)이고 실제 커널의 SIZE 는 262,144(gemm) 수준이라 열거는 쓸 수 없다. index.md:96 은 '정규형으로 정규화되고 기호적으로 검증된다'고 주장하지만 그 절차는 공개돼 있지 않다 — 우리가 독립적으로 세우고 증명해야 한다.
- 엔트리 병합 재작성의 합류성 증명과 최대병합 정규형 알고리즘, 그 위에서 '병합 후 ≤8' 판정의 다항시간 알고리즘. 그리고 실현가능성 술어 전체(엔트리·반복·패킷 크기·최내곽 조건·공통세분·용량·정렬)의 결정 절차 및 복잡도.
- 검사기 구현 후 커널 200개 차등 검사: 혼동행렬, 특히 24 REAL_LOWERING_GAP 예측률, 23 INTENTIONAL_NEGATIVE 거부율, 13 ICE 분리율. 주의: compile <FILTER> 는 부분문자열 매칭이라 접두사 충돌 8건이 있고 실제 오염 판정 1건(invalid_time0)이 있었다 — 'error: furiosa-opt: <정확한 커널명>:' 규칙으로 엄밀 재판정해야 한다.
- ★ 정적 검사기가 조용한 오배치 2건(broadcast 미기록, tile 32 B/elem 혼동)과 로더 abort 3건(1.5배 크기 오산)을 사전에 거부하는지 검증. 이게 논문의 결정적 결과다. 실패하면 '벤더 검사와 같은 것을 다시 짰다'가 된다.
- 수리 절차 구현과 성공률 측정: 24개 REAL_GAP 중 의미보존 재작성으로 --backend npu 를 통과시킨 비율, 통과 커널의 실기 값 일치, --dump-schedule 기준 DMA 사이클 변화.
- 일반화 실험(단일 벤더 반론 대응): 같은 술어 프레임워크로 (a) CuTe 의 complement/composition/logical division well-definedness 조건을 재유도하거나 (b) NKI 타일 제약(partition ≤128, free ≤64K SBUF / 4K PSUM) 또는 Gemmini 제약을 인스턴스화. 최소 1개 제2 타깃 없이는 통과 못 한다.
- 실 LLM 커널에서 엔트리 8개 한도가 실제로 구속력이 있는지 측정 — transformer 4종이 전부 ICE 라 이 서버 예제로는 측정이 막혀 있다. 별도 커널을 직접 작성해야 할 수 있다.
- 다중 칩/클러스터 경로 검증 불가 문제 해결: 이 서버에서 다중 칩/클러스터 예제 6종이 하나도 실기에서 안 돈다(5종 컴파일 실패, 1종 로드 abort). Chip/Cluster 차원의 합법성 이론은 현재 실기 근거를 붙일 수 없다.

### 1.8 ★ 심사 반론

**치명적 반론 (reject 사유)**

- ★ 논문의 유일한 비-CuTe 차별점(관계적 Sliding)이 구현에 존재하지 않는다 — 제안서가 스스로 '미구현이면 평가 불가'라고 적은 분기가 이미 확정됐다. 실측: /mnt/nvme1n1p2/home_jun/.cargo/registry/src/index.crates.io-*/furiosa-mapping-{types,macro}-0.{3,4}.0 및 furiosa-opt-std/furiosa-opt-lower 전체에 대해 grep -rli 'sliding|skew|LinearComb' → 0건. lib.rs 의 pub enum Mapping 은 정확히 7개 변종(Symbol/Stride/Modulo/Resize/Padding/Pair/Broadcast)뿐이고, parser.lalrpop 토큰 집합에 '$'(Dollar)도 아포스트로피도 없다. Skew·Sliding·간접 시퀀싱은 벤더 책 산문에만 있고 0.3·0.4 어디에도 없다. 남는 대수(Pair=concatenation, Stride/Modulo=logical division, Broadcast=stride-0 mode)는 CuTe 레이아웃의 진부분언어다.
- ★ sequencer 엔트리 병합 규칙은 CuTe coalescence 그 자체이고, '합류성·정규형·의미보존' 정리는 이미 출판돼 있다. `(n1:s1),(n2:s2)→(n1·n2:s2) iff s1==n2·s2` 는 CuTe 의 coalesce 조건(다음 모드의 stride == 앞 모드의 size×stride)과 안팎 규약만 다를 뿐 동일하다. Carlisle/Shah/Stern(arXiv:2601.05972)은 Φ_coal(L,S̄)=Φ_L 을 이미 증명했고, 나아가 **relative coalesce** — 결과 shape 가 주어진 S̄ 를 refine 해야 한다는 제약 하의 병합 — 을 정의한다. 이것이 제안서가 '선행연구를 못 찾았다'고 한 Buf/스트림 공통세분(gcd) 조건과 같은 문제다. 기여 3의 두 간판 정리가 제안서 자신이 인용한 논문 안에 있다.
- ★ 동치 판정의 복잡도 동기가 성립하지 않는다. AxisName::SIZE 는 `const SIZE: usize` 이고 axes! 가 컴파일 시점에 고정한다 — 이 언어에는 기호적(파라메트릭) 축 크기가 아예 없다. 따라서 모든 M::SIZE 가 구체 상수이고, '소박한 O(SIZE) 열거'는 SIZE=262,144 에서도 순수 Rust CPU 코드로 밀리초다(제안서 스스로 SIZE/map 이 NPU 없이 cargo test 로 돈다고 적었다). 열거가 이미 완전한 결정 절차인데 그 위에 정규형·완전성 증명을 얹는 것은 존재하지 않는 병목을 푸는 것이다.
- ★ Padding 의 Top/Zero/Bottom '3종 격자'는 논문이 기계화하겠다는 의미론에 들어 있지 않다. dsl.rs:211-213 의 `impl M for Padding<L,SIZE,KIND> { fn map(i)->Option<Index> { L::map(i) } }` — KIND 는 map 에 한 번도 나오지 않고 to_value() 의 태그로만 백엔드에 넘어간다. 즉 세 종류는 책 자신의 동치 정의(SIZE 같음 + 모든 i 에서 map 같음) 하에서 **구별 불가능**하다. 결과: (a) '지식·권한 주석'은 의미론이 아니라 폐쇄 소스 백엔드가 읽는 태그다, (b) 벤더의 동치 관계는 이미 백엔드에 대해 불건전하다 — 그러면 '기존 표시의미론을 기계화하고 그 동치를 결정한다'는 기여는 틀린 대상을 기계화하는 것이 된다. 새 의미론을 발명해야 하는 순간 '벤더 문서를 형식화했다'는 프레이밍이 무너지고, 들여다볼 수 없는 폐쇄 소스 오라클을 상대로 그 발명을 정당화해야 한다.
- ★ 결정적 실험(조용한 오배치 2건·로더 abort 3건을 정적으로 거부)이 원리적으로 성립하지 않는다. 이 5건은 매핑 합법성 위반이 아니라 백엔드/로더의 크기 계산 결함이다(N5-① 요구크기 1.5배는 furiosa_kernel_load 에서 발생, N5-②(b)의 'byte vs element' 진단은 _GROUND_TRUTH 가 '정황 근거이며 런타임 소스 확인은 아님'이라고 명시). m![] 표현식 위의 정적 검사기는, 매핑이 합법인데 백엔드 산술이 틀린 커널을 거부할 수 없다 — 백엔드 크기 계산을 재구현해 불일치를 보고하지 않는 한. 그건 '합법성 판정'이 아니라 '두 번째 컴파일러를 짜고 내 쪽이 옳다고 주장'이다. 제안서는 '여기서 이기면 논문이 선다'고 쓰면서 이길 기전을 제시하지 않는다. ICE 13건도 마찬가지로 합법성 신호가 아니다.
- 간판 숫자 63/200 이 주장하는 것을 측정하지 않는다. 63 = INTENTIONAL_NEGATIVE 23(실패가 정상 — 벤더가 일부러 만든 음성 표본을 '잡았다'고 세는 셈) + COMPILER_ICE 13(제안서 인용원 스스로 'ICE 는 사용자가 매핑을 고쳐서 못 피한다'고 적음) + GENERIC_NOT_MONOMORPHIZED 2(빌드 시스템 산물) + UNCLEAR 2 + REAL_LOWERING_GAP 24. 그리고 그 24의 실제 사유(13-매트릭스 §7.2)는 `Branch conversion is not yet implemented`(제어흐름, 레이아웃 아님), i4 서브바이트 크기 오산, 다중칩 재배치 미구현 등 **미구현 백엔드 기능**이 상당수다 — 이들을 거부하는 검사기는 합법 프로그램을 거부하는 것(위양성)이고 주장과 정반대다. 정직한 분모는 24보다 훨씬 작고 아마 10 안팎이다. 단일 벤더 예제 크레이트에서 열 건 남짓은 워크숍 규모의 실증이다.
- 베이스라인 B1 이 불공정하다. _GROUND_TRUTH N1: `--backend npu` 는 패키지의 모든 #[device] 함수를 빌드 시점에 낮추는 반면 typecheck/emulation 은 **호출될 때만** 처리한다. 따라서 'typecheck 가 63건 중 0건을 잡는다'는 검사 능력의 차이가 아니라 상당 부분 '보지도 않았다'를 측정한 것이다. N1 을 읽은 리뷰어는 이 비교를 즉시 무효화한다. 공정한 baseline 은 typecheck 를 같은 200개에 강제로 방문시킨 뒤 재측정해야 성립한다.
- 성능/귀속 서사의 근거가 실측이 아니라 폐쇄 소스 컴파일러 자신의 비용 모델이다. 제안서는 'DmaEngine 96.5% — 사이클 귀속 **실측**'이라 쓰지만 출처 N4 는 굵은 글씨로 '이 값은 컴파일러의 스케줄 모델 예측이며 실측 벽시계가 아니다'라고 하고, 유일한 교차검증(mnist 17,953 cycle)조차 '두 기록이 같은 스케줄 모델 산출물이므로 벽시계 대조는 아니다'라고 못박는다. 즉 이 기계에 대한 유일한 정량적 특성화가, 논문이 버그투성이라고 공격하는 바로 그 산출물에서 나온다.
- 직접 경쟁하는 2026년 선행연구를 놓쳤다(관련연구 14편에 하나도 없다). 특히 Axe(arXiv:2601.19092, Tianqi Chen 그룹, 2026-01)는 '논리 텐서 좌표를 named axes 를 통해 다축 물리 공간으로 보내는 하드웨어 인지 추상'으로 tiling/sharding/replication/offset 을 device mesh 부터 thread 까지 통일한다 — vISA 의 Chip/Cluster/Slice/Lane/Packet named-axis 매핑과 같은 착상을 컴파일러·다기기 평가까지 붙여 이미 출판했다. 그리고 AWS Trainium 은 Lean 으로 ~50만 줄 규모 ISA 형식화 + CompCert 스타일 검증 컴파일러를 굴리며 그 명세로 실제 하드웨어·시뮬레이터 버그를 잡고 있다 — 제안서가 지정한 '제2 타깃'이 바로 그 칩이다.
- 단일 벤더·단일 칩·단일 툴체인 세대이고, 이론의 상당 부분이 검증 불가 영역에 있다. 다중 칩/클러스터 예제 6종이 전부 실기에서 안 돌아(5종 컴파일 실패, 1종 로드 abort) Chip::SIZE==NUM_CHIPS·Cluster::SIZE==2 를 다루는 공간 합법성 이론에 실기 근거를 붙일 수 없다. transformer 4종은 전부 ICE 라 '엔트리 8개 한도가 실제 LLM 커널에서 구속력이 있는가'라는 핵심 질문도 이 서버에서 측정이 막혀 있다. 제안서의 일반화 계획도 '실기 접근이 없으므로 제약 술어를 문서에서 형식화했다 수준에 그칠 수 있다'고 자인하는데, 그건 일반화가 아니라 관련연구 표다.
- 법적·윤리적 게이트가 미해결인 채로 일정에 들어가 있다. 제안서는 목표를 '벤더의 비공개 내부(레이아웃 정규화 절차)를 역설계해 공개'로 규정하고, 악용 가능한 결함 서명 5건을 싣겠다고 하며, NDA·벤더 관계 확인을 '발표 전에' 하겠다고 미룬다. 아티팩트 평가는 벤더 크레이트 재배포를 요구한다. 기술적 성패와 무관하게 투고 시점에 논문을 죽일 수 있다.

**이미 출판되어 신규성이 없는 부분**

- Categorical Foundations for CuTe Layouts — Carlisle, Shah, Stern, arXiv:2601.05972, 2026-01. coalesce 의 의미보존(Φ_coal(L,S̄)=Φ_L)을 증명하고 **relative coalesce**(주어진 S̄ 를 refine 해야 한다는 제약 하의 병합)를 정의한다 → 제안서의 기여 3(엔트리 병합 합류성·정규형)과 '공통세분 존재 판정'이 둘 다 여기에 이미 있다.
- CuTe Layout Representation and Algebra — Cris Cecka (NVIDIA Research), arXiv:2603.02298, 2026-03. concatenation/coalescence/composition/complementation/division/tiling/inversion 대수 + '아키텍처가 규정한 레이아웃의 컴파일 타임 검증'. 구현에 남은 vISA 대수(Pair/Stride/Modulo/Broadcast)는 이 대수의 진부분집합이다.
- A note on the algebra of CuTe Layouts — Jay Shah, Colfax Research, 2024-01. complementation/composition/logical division 의 well-definedness 조건 = 레이아웃 합법성 판정.
- Axe: A Simple Unified Layout Abstraction for Machine Learning Compilers — Hou, Jin, Wang, Chen, Cai, Yang, Ye, Ding, Lai, T. Chen, arXiv:2601.19092, 2026-01. **논리 텐서 좌표 → named axes 다축 물리 공간** 매핑으로 tiling/sharding/replication/offset 을 device mesh~thread 까지 통일하고 DSL·컴파일러까지 붙였다. vISA 의 Chip/Cluster/Slice/Lane/Packet named-axis 매핑과 같은 착상. 제안서 미인용.
- AWS Trainium 용 CompCert 스타일 검증 컴파일러 (~50만 줄 Lean) — de Moura, 'The Lean Theorem Prover: Design, Evolution, and Impact', FLoC 2026 자료에 기술. 상용 NPU 의 ISA 를 Lean 으로 형식화해 HW/검증 엔지니어가 공유하는 시뮬레이터로 쓰고, 하드웨어·시뮬레이터 버그를 잡으며 종단 컴파일 의미보존 증명을 목표로 한다. 제안서의 '제2 타깃'(Trainium/NKI)에서 이미 훨씬 큰 규모로 수행 중. 제안서 미인용.
- DESIL: Detecting Silent Bugs in MLIR Compiler Infrastructure — PACMPL(OOPSLA) 2025, doi 10.1145/3763161. 차등 검사로 **조용한(silent) 버그**를 잡는 첫 기법, silent 23 + crash 19 발견. 제안서의 '차등 검사로 조용한 오배치를 잡는다' 방법론이 이미 있고 규모도 크다(제안서는 단일 벤더 2건). 미인용.
- Glenside / Pure Tensor Program Rewriting via Access Patterns — Smith 외, MAPL 2021. equality saturation 으로 프로그램 조각을 가속기 호출로 매핑하고 **데이터 레이아웃 변환을 자동 발견**한다 → 제안서 기여 4(egg 포화 + 실현가능성 비용함수 추출)를 선점. 미인용.
- ACT: Automatically Generating Compiler Backends from Tensor Accelerator ISA Descriptions — arXiv:2510.09932, 2025. 가속기 ISA 기술로부터 백엔드 자동 생성. '한 벤더 매뉴얼을 손으로 술어화'하는 접근의 대안으로 리뷰어가 반드시 든다. 미인용.
- Joint Program and Layout Transformations ... based on Constraint Programming — arXiv:2104.04731, 2021. 특수 하드웨어의 레이아웃 제약을 제약 프로그래밍으로 만족시키는 합법 레이아웃 생성. 'hexcute 이전'에 이미 레이아웃 합법성=CP 정식화가 있었음을 보여 준다. 미인용.
- TensorLift / ATLAAS — arXiv:2604.13523, 2026. RTL 로부터 텐서 ISA 의미론 자동 추출(156 파일, 121만 줄 처리). 제안서도 인접으로 인용했으나, '문서로부터 손으로 형식화'가 왜 더 나은지에 대한 답이 없다.

**근거 없이 단정한 문장 (수정 필요)**

- 'vISA 의 Sliding 은 하나의 버퍼 인덱스가 여러 텐서 인덱스로 가는 관계를 1급으로 둔다(책이 명시)' — 책 산문에는 있으나 0.3.0·0.4.0 어느 크레이트에도 구현이 없다(grep 0건, Mapping enum 7변종, 렉서에 '$' 토큰 없음). '1급'이라는 단정에 실행 가능한 근거가 없다.
- 'Padding 이 Top/Zero/Bottom 3종 격자라 슬롯에 대한 지식·권한 주석이 붙는다' — dsl.rs 의 Padding::map(i)=L::map(i) 는 KIND 를 완전히 무시한다. 세 종류는 책 자신의 동치 정의 하에서 구별 불가능하며, 격자는 의미론이 아니라 to_value() 태그다.
- '현행 정의는 소박하게는 O(SIZE)(실제 커널에서 10^5~10^6)다' 를 결정 절차의 동기로 쓴 것 — 사실이지만 무의미하다. 축 크기가 전부 const 라 열거가 밀리초에 끝나고 그 자체로 완전한 결정 절차다. 이것이 어디서든 병목이라는 근거가 제시되지 않았다.
- '사이클 귀속 **실측**: DmaEngine 96.5%' — 출처 N4 는 '컴파일러의 스케줄 모델 예측이며 실측 벽시계가 아니다'라고 명시한다. 실측으로 라벨링한 것은 근거 왜곡이다.
- 'B1 은 63건 중 0건을 잡는다(typecheck 커널 낮추기 오류 0건)' 을 typecheck 의 정밀도 부족 근거로 쓴 것 — N1 에 따르면 typecheck 는 호출된 커널만 처리하므로 커버리지와 정밀도가 교락돼 있다.
- '실패 메시지가 대부분 매핑 모양이다' — 인용원인 02-매핑-텐서.md:379 자신이 괄호로 '다만 전부는 아니다 — 63개 중 13개는 internal compiler error, 2개는 제네릭 미단형화'라고 정정하고, §7.2 는 branch conversion 미구현·i4 서브바이트 오산을 나열한다. '대부분'은 과장이다.
- '검사기가 조용한 오배치 2건·로더 abort 3건을 정적으로 거부하는가. 여기서 이기면 논문이 선다' — 매핑 수준 검사기가 백엔드/로더 크기 계산 버그를 잡을 수 있는 기전이 전혀 제시되지 않았다. N5-②(b) 의 'byte vs element' 진단 자체가 원문에서 '정황 근거, 런타임 소스 미확인'으로 유보돼 있다.
- '이 조합을 판정 문제로 세운 선행연구는 못 찾았다' / '대응물을 못 찾았다' — 검색이 Axe(2601.19092), Glenside, ACT(2510.09932), DESIL(OOPSLA 2025), Trainium/Lean 을 놓쳤고, CuTe 의 relative coalesce 가 공통세분 조건에 정면 대응한다는 것도 놓쳤다. '못 찾았다'를 '없다'로 쓴 문장들이다.
- '6개 항등식이 완전한 등식 이론인지 판정하고 열거 없는 정규형 알고리즘과 완전성 증명을 준다' — 이것이 열려 있거나 어렵다는 근거가 없다. 생성자 7개·상수 크기의 준아핀 레이아웃 언어에서 정규형은 거의 기계적이고, 벤더는 이미 정규형을 갖고 있다고 문서에 적어 두었다(index.md:96).
- '24개 REAL_GAP 중 의미보존 재작성으로 --backend npu 를 통과시킨 비율' — 그 24개의 상당수가 branch conversion 미구현·다중칩 재배치·i4 서브바이트 등 어떤 매핑 재작성으로도 못 고치는 항목이다. 실제 달성 가능한 분모가 명시되지 않았고 매우 작을 것이다.
- 'IEEE Micro 논문 저자에 Nuno P. Lopes 같은 형식검증 연구자가 이미 있다 — 접점이 있다' — 저자 목록은 협업 채널이 아니다. 벤더 관계 리스크를 완화하는 근거처럼 제시됐으나 근거가 아니다.
- '9~12개월 1인 / Rocq·Lean 기계화 3~4개월' — 대조군인 ATL(PLDI 2024)이나 Trainium/Lean(~50만 줄)은 다인·다년 규모다. 생성자 9개 전부 + 판정 절차 + 수리 + 제2 타깃을 1인 3~4개월에 기계화한다는 산정에 근거가 없다.
- 'mapping-expressions.md:601-617' 인용 — 파일은 616줄이다(사소하지만 인용 검증이 느슨하다는 신호). 마찬가지로 02-매핑-텐서.md:18 은 Sliding 출처를 'mapping-expressions.md:409-468' 로 적었으나 실제 Sliding 절은 565줄 이후다.

**조사자 스스로 적은 위험**

- ★ 신규성 압박이 가장 큰 위험. 2026년에 CuTe 레이아웃 대수 형식화 논문이 2편(Cecka; Carlisle/Shah/Stern) 나왔고, Shah 의 2024 노트는 이미 well-definedness 조건을 준다. 리뷰어는 '니치 칩에 CuTe 대수를 다시 형식화한 것'으로 볼 수 있다. Sliding 의 비단사성·Padding 3종 격자·파이프라인 단계 실현가능성이라는 차별점을 논문 제목·초록에서부터 못 박지 않으면 죽는다.
- 합성(einsum→매핑) 각도는 Hexcute 가 이미 '레이아웃 합성을 제약 프로그래밍으로 형식화하고 타입추론으로 푼다'로 선점했다. 전면 합성을 주장하면 직접 경쟁이 되고 GPU 대비 성능 비교를 요구받는다. '수리(repair)'로 범위를 좁히는 것이 안전하다.
- 벤더가 이미 '매핑 표현식은 정규형으로 정규화되고 기호적으로 검증된다'(index.md:96)고 적어 두었다. 즉 우리 기여는 '없던 것을 만든 것'이 아니라 '비공개 절차를 독립 재구성하고 증명한 것'이 된다. 이 프레이밍을 인정하고 대신 '벤더 절차가 실제로 불완전함을 63/200 과 조용한 오배치 2건으로 보인다'를 전면에 세워야 한다.
- 오라클이 오염돼 있다. 대조 대상인 벤더 컴파일러가 폐쇄 소스이고 ICE 13건에 조용한 오배치 2건이 있어, 검사기와 컴파일러가 불일치할 때 '누가 틀렸는가'가 자동으로 결정되지 않는다. 실기 실행을 최종 심판으로 써야 하는데 실기도 프로세스 오염(HAL -110 연쇄)으로 노이즈가 크다.
- 가장 흥미로운 생성자(Skew·Sliding·간접 시퀀싱)가 가장 근거가 약하다. 책 서술만 있고 0.4.0 소스 대조가 안 됐으며 '간접 시퀀싱' 절은 본문이 비어 있다. 미구현으로 밝혀지면 '함수적 조각만 남은 대수'가 되고, 그건 CuTe 와 훨씬 더 겹친다.
- 성능 서사가 약하다. PeCore 는 전체 사이클의 3.3%뿐이고 DmaEngine 이 96.5%다. '검사기가 있으면 빨라진다'는 주장은 성립하지 않고, '안 되던 커널이 되게 한다 / 조용히 틀리던 것을 잡는다'만 주장할 수 있다. ASPLOS·ISCA 리뷰어는 성능 숫자를 요구할 것이므로 PL 계열(CGO/CC/OOPSLA)이 맞다.
- 단일 벤더·단일 칩. 제2 타깃 인스턴스화가 없으면 '엔지니어링 노트'로 강등될 실질 위험이 있다. 그런데 제2 타깃(Trainium NKI, Gemmini)에 실기 접근이 없으므로 '제약 술어를 문서에서 형식화했다' 수준에 그칠 수 있고, 그건 약한 일반화다.
- 툴체인 세대 스큐가 실험을 갉아먹는다. 래퍼는 --version 이 0.3.0 을 출력하지만 실제로는 0.4 세대이고, 사용자 프로젝트는 0.3 핀 + src/furiosa-opt.tag 라 현재 래퍼가 'no kernel packages found' 로 거부한다. 재현성 서술에 이 함정을 명시하지 않으면 아티팩트 평가에서 문제가 된다.
- 국내 벤더 종속. FuriosaAI 스택의 미공개 내부(레이아웃 정규화 절차)를 역설계해 공개하는 형태가 되므로, 발표 전에 벤더와의 관계·NDA 여부를 확인해야 한다. 반대로 벤더 공동저자를 얻으면 논문이 훨씬 강해진다(IEEE Micro 논문 저자에 Nuno P. Lopes 같은 형식검증 연구자가 이미 있다 — 접점이 있다).

### 1.9 살리는 길 — 무엇을 바꿔야 하는가

"제안대로는 살릴 수 없다. '대수를 형식화한다'를 버리고 '명세와 구현의 괴리를 측정한다'로 축을 90도 돌려야 한다. 구체적으로:\n\n1) **먼저 잘라낼 것(검증 완료 사실 기반)**: Skew·Sliding·간접 시퀀싱은 0.3·0.4 어디에도 없다(grep 0건, Mapping enum 7변종, 렉서에 '$' 없음). 전부 삭제하라 — 이걸 남기면 첫 리뷰어가 5분 만에 확인하고 논문 전체의 신뢰를 잃는다. 동치 결정 절차도 삭제하라(축 크기가 const 라 열거가 이미 완전 절차다). 남은 대수가 CuTe 레이아웃의 진부분언어임을 **초록에서 먼저 인정**하고, 신규성을 대수가 아니라 실증에 걸어라.\n\n2) **새 논지**: '상용 NPU 의 타입 수준 레이아웃 언어에 대한 명세-구현 적합성(conformance) 연구'. 결정적 자산은 M 트레잇이 순수 CPU 오라클이라는 점이다 — map() 은 하드웨어 없이 cargo test 로 돌고, to_value() 는 같은 표현식을 폐쇄 소스 백엔드로 보낸다. 즉 **공짜 오라클이 딸린 차등 퍼징 설정**이고, 실제로 조용한 버그 2건을 만들어 낸 것도 이 구조다. 생성기로 매핑 표현식을 뽑아 map() 예측 vs (typecheck / npu 낮추기 / 에뮬레이션 / 실기 값) 4-way 를 대조하라. 200개 예제가 아니라 수만 개 생성 매핑이 분모가 되면 단일 벤더 반론이 크게 약해진다.\n\n3) **검사기의 역할 재정의**: 합법성 판정기가 아니라 '문서에서 순수하게 유도되는 단계별 술어의 적합성 검사기'로 한정하라(sequencer 엔트리≤8·병합·반복≤65,536·패킷크기·최내곽 조건, Collect 32B flit, 공간 정확일치, 용량, 정렬). 평가는 혼동행렬 한 장이 아니라 **두 방향**으로 보고하라: (a) 문서상 합법인데 백엔드가 거부 = 명세 버그, (b) 문서상 불법인데 백엔드가 수용 = 건전성 버그. INTENTIONAL_NEGATIVE 23 을 양성 대조군으로 쓰고, REAL_GAP 24 중 branch conversion·다중칩·i4 처럼 매핑과 무관한 항목은 분모에서 명시적으로 제외하라(정직한 분모는 10 안팎이라고 먼저 써라).\n\n4) **간판 주장 교체**: 'index.md:96 의 벤더 불변식 — 매핑 표현식은 정규형으로 정규화되고 기호적으로 검증된다 — 이 거짓임을 N개의 반례로 보인다.' 이건 반증 가능하고 Rocq 가 전혀 필요 없다.\n\n5) **Padding 발견을 논문의 중심에 두되 최소 재현으로 다시 세워라**: Padding::map 이 KIND 를 무시하므로 언어 자신의 동치 관계가 Top/Zero/Bottom 을 구별하지 못하는데 백엔드 크기 계산은 구별한다 — '자기 컴파일러에 대해 불건전한 문서화된 의미론'. valid_size_{1,2,4} 라는 벤더 테스트에 얹지 말고 20줄짜리 최소 커널로 재현하라. 이게 사실이면 CuTe 계열에 대응물이 없는 유일하게 남은 진짜 차별점이다.\n\n6) **숫자 라벨 정정**: 스케줄 모델 산출물은 전부 '모델 예측'으로 표기하라. DMA 96.5% 를 '실측'이라 쓴 문장은 그대로 두면 데스크 리젝 사유다. 성능 서사를 유지하려면 소수 커널이라도 벽시계를 재고, 못 재면 성능 프레이밍을 통째로 삭제하라. 프로세스 격리(10/25 → 33/3)와 부분문자열 필터 오염(8건 후보/1건 실오염)은 반대로 **방법론 기여**로 전면에 세워라 — 이건 검증됐고 재현 도구도 있다.\n\n7) **관련연구 재작성**: Axe(2601.19092), Glenside(MAPL 2021), DESIL(OOPSLA 2025), ACT(2510.09932), Trainium/Lean 을 반드시 넣고, CuTe relative coalesce 가 공통세분 조건과 같음을 스스로 밝혀라. 리뷰어가 먼저 찾으면 끝이고, 저자가 먼저 인정하면 정직성 점수가 된다.\n\n8) **투고처·일정 하향**: CGO/OOPSLA/PLDI 는 이 내용으로 도달 불가. CC 또는 MAPS/LATTE 워크숍의 experience/measurement 논문, 혹은 컴파일러 테스팅 각도로 ISSTA/ICSE-SEIP. 4~6개월.\n\n9) **형식 각도를 굳이 고집한다면** 유일하게 방어 가능한 판본은 '기호적 축 크기의 도입'이다 — 이 언어에 없는 파라메트릭 shape 을 추가해야 비로소 합법성이 진짜 결정 문제가 되고 Presburger 결정성 주장이 의미를 갖는다. 단, 그러려면 이 칩에서 파라메트릭 커널을 원하는 사람이 있음을 먼저 보여야 하는데 transformer 4종이 전부 ICE 라 그 증거를 못 만든다. 권하지 않는다.\n\n10) **벤더/NDA 를 집필 전에 정리하라.** 결함 서명 공개와 크레이트 재배포가 아티팩트 요건에 걸린다. 공동저자를 얻으면 위 2)~5)가 훨씬 강해지고, 못 얻으면 5건 공개 범위를 먼저 합의해야 한다."

### 1.10 후보 학회와 소요 기간

- CGO (Code Generation and Optimization) — 컴파일러 백엔드·판정 절차·차등 검사 조합에 가장 맞다. 1지망.
- CC (Compiler Construction) — 규모가 작고 대수 형식화 + 결정 절차에 우호적. 2지망.
- OOPSLA / PACMPL — 기계화 증명을 완주하면. 단 '단일 칩' 반론이 가장 세게 들어온다.
- PLDI — 수리(repair)까지 완성하고 제2 타깃 일반화가 붙었을 때만.
- LCTES / MAPS(PLDI workshop) / CTSTA — 초기 결과 선공개용.
- ASPLOS — DMA 사이클 개선 수치를 붙일 수 있을 때만. 현재 근거로는 무리.
- MICRO Industry Track / IEEE Micro — 벤더 공저가 성사되면 가장 현실적인 큰 무대.

**소요 기간 추정**: 9~12개월 (1인 기준). 내역: 구현 소스 대조·의미론 확정 1개월, Rocq/Lean 기계화 3~4개월, 판정 절차 설계·증명 2개월, 검사기 구현 + 커널 200개 차등 검사 1.5개월(도구 상당수 재사용 가능), 수리 절차 1.5개월, 제2 타깃 일반화 2개월, 집필 1개월. 다만 첫 1개월의 소스 대조에서 Skew/Sliding/간접 시퀀싱이 미구현으로 밝혀지면 주제를 '함수적 조각의 파이프라인 실현가능성'으로 축소해야 하고 그때는 6~8개월 규모의 CC/CGO 논문이 된다.

**신규성 확신도**: 중간(60% 정도). 근거를 나눠 쓴다. **높은 쪽**: 이 대수 자체는 확실히 미출판이다 — 벤더의 ISCA 2024 TCP 논문과 IEEE Micro 45(3) 2025 RNGD 논문을 직접 확인했고(후자는 PDF 8쪽 전문 판독), 둘 다 아키텍처·설계공간 탐색 서술뿐 m![] 매핑 표현식·타입 수준 인코딩·합법성 판정은 전무하다. FuriosaAI/RNGD/TCP + mapping expression/virtual ISA 로 검색해도 관련 논문이 안 나온다. **낮은 쪽**: '레이아웃 대수의 형식 의미론 + 합법성 판정 + 합성'이라는 문제 부류는 이미 붐빈다. CuTe 만 해도 well-definedness 조건을 주는 기술노트(Shah 2024)와 형식화 논문 2편(Cecka arXiv:2603.02298, Carlisle/Shah/Stern arXiv:2601.05972, 둘 다 2026)이 있고, 합성 쪽은 Hexcute(arXiv:2504.16214)가 '레이아웃 합성 = 제약 프로그래밍 + 타입추론'을 이미 했다. 따라서 '형식 의미론을 정의한다' 단독으로는 신규성이 없다고 봐야 한다. 살아남는 신규성은 세 가지로 좁혀진다: (1) Sliding 이 하나의 버퍼 인덱스를 여러 텐서 인덱스로 보내는 **관계**라는 점 — CuTe 레이아웃은 함수이고 well-definedness 를 위해 비퇴화성을 요구하므로 이 조각은 CuTe 대수로 표현되지 않는다; (2) Padding 의 Top/Zero/Bottom 3종 격자(임의값/0 보장/접근 시 UB)라는 지식·권한 주석 — 대응물을 못 찾았고, 실제 백엔드 버그(타입은 256 B, 백엔드는 228/232/240)가 정확히 여기서 난다; (3) 합법성이 단일 메모리 레이아웃이 아니라 sequencer(엔트리 ≤8, 병합 s1==n2·s2, 반복 ≤65,536, 패킷 ∈{1,2,4,8,16,32}B, 최내곽 조건) + Collect(32B flit) + 공간 정확일치 + 공통세분(gcd) 의 **논리곱**이라는 정식화 — 이 형태의 판정 문제를 다룬 선행연구는 못 찾았다. 여기에 '벤더 컴파일러가 실제로 불완전하다'는 실증(200개 중 63 실패, typecheck 는 0건 검출, 컴파일·로드 통과 후 조용히 틀리는 2건)이 붙어야 논문이 선다. 이 실증 없이 형식화만 내면 CuTe 형식화 논문들의 재탕으로 읽힐 것이다.

### 1.11 검색으로 확인한 관련 연구

| 제목 | 학회·연도 | 이 주제와의 관계 |
|---|---|---|
| CuTe Layout Representation and Algebra | arXiv:2603.02298 (Cris Cecka, NVIDIA Research), 2026-03 | 겹침. 계층적 레이아웃 표현과 concatenation/coalescence/composition/complementation/division/tiling/inversion 대수를 정의하고 '아키텍처가 규정한 레이아웃의 컴파일 타임 검증'을 명시적으로 내세운다. 즉 '매핑 대수 + 컴파일 타임 합법성'이라는 조합이 이미 GPU 쪽에 있다. 차이: CuTe 레이아웃은 좌표→오프셋 함수이고, vISA 의 Sliding 같은 비단사 관계나 Padding 3종 격자, sequencer 엔트리 예산 같은 파이프라인 단계 제약은 없다. |
| Categorical Foundations for CuTe Layouts | arXiv:2601.05972 (Jack Carlisle, Jay Shah, Reuben Stern), 2026-01 | 겹침. 레이아웃 대수의 형식 의미론을 범주·오퍼라드로 세우고 layout diagram 그래픽 계산을 도입하며, composition/complement/logical division 이 대응 연산과 호환됨을 증명한다. '이 대수의 형식 의미론을 정의한다'는 기여 자체가 다른 대수에서 이미 수행됐다는 증거. |
| A note on the algebra of CuTe Layouts | Colfax Research technical note (Jay Shah), 2024-01 | 겹침. complementation·composition·logical division 이 well-defined 가 되는 조건을 명시한다 — 이것이 사실상 '레이아웃 합법성 판정'이다. 우리의 '합법성 판정' 기여를 그대로 겨냥하는 선행연구이므로 반드시 차별점을 세워야 한다. |
| Hexcute: A Compiler Framework for Automating Layout Synthesis in GPU Programs | arXiv:2504.16214 (Zhang, Ding, Sun, Hu, Shpeisman, Pekhimenko), 2025-04 (최신 2026-01) | 겹침. '레이아웃 합성을 제약 프로그래밍 문제로 형식화하고 타입추론 기반 알고리즘으로 푼다'. 우리 주제의 'einsum 으로부터 매핑을 합성할 수 있는가' 부분과 정면으로 겹친다. CUTLASS 대비 코드 1.27~7.94배 감소, Triton 대비 MoE 6.46배, vLLM end-to-end 2.60배까지 보고. |
| Graphene: An IR for Optimized Tensor Computations on GPUs | ASPLOS 2023 (Hagedorn, Fan, Chen, Cecka, Garland, Grover) | 인접. 다차원 데이터와 스레드를 1급 텐서로 두고 계층적 타일 분해로 데이터↔스레드 매핑을 표현한다. vISA 의 Chip/Cluster/Slice/Lane/Packet 계층 매핑과 같은 발상이지만 형식 의미론·합법성 판정은 목표가 아니다. |
| CoSA: Scheduling by Constrained Optimization for Spatial Accelerators | ISCA 2021 (Huang, Kang, Dinh, Norell, Kalaiah, Demmel, Wawrzynek, Shao) | 인접. DNN 스케줄 공간을 알고리즘·아키텍처 제약을 가진 MIP 로 정식화해 한 번에 푼다. '하드웨어 제약 하에서 매핑을 합성한다'는 목표가 같지만 대상은 Timeloop 류 루프네스트 매핑이고, flit 정규화·sequencer 엔트리 예산·패딩 종류 같은 단계별 실현가능성은 다루지 않는다. |
| Exocompilation for Productive Programming of Hardware Accelerators (Exo) | PLDI 2022 (Ikarashi, Bernstein, Reinking, Genc, Ragan-Kelley) | 기반. 하드웨어 명령·특수 메모리·설정 상태를 사용자 라이브러리로 외부화하고, 스케줄을 언어 내 합성 가능한 재작성으로 두며, effect 분석으로 프로그램 동치와 메모리 안전을 보장한다. '재작성이 의미를 보존함을 보장한다'는 우리 수리 절차의 직접적 기반이다. |
| Exo 2: Growing a Scheduling Language | arXiv:2411.07211, 2024-11 | 기반. 사용자 확장 가능한 스케줄링 언어. 매핑 재작성을 사용자 정의 스케줄 연산으로 노출하는 설계 참고. |
| A Verified Compiler for a Functional Tensor Language (ATL) | PACMPL / PLDI 2024 | 기반. 명시적 구문 트리와 형식 의미론의 deep embedding 위에서 Coq 로 의미보존 재작성을 검증한다. 우리의 기계화 방식(deep embedding + 검증된 재작성)의 직접 모델. |
| TensorRight: Automated Verification of Tensor Graph Rewrites | POPL 2025, PACMPL vol.9, Article 29 (Arora, Lu, Jain, Xu, Houshmand, Phothilimthana, Lesani, Narayanan, Srinivasa Murthy, Bodík, Sabne, Mendis) | 인접. 임의 랭크·크기 텐서에 대해 그래프 재작성의 건전성을 자동 검증하는 첫 시스템. aggregated-axis 정의로 무한 축을 다루고 충분 랭크를 계산한다. 우리는 그래프 재작성이 아니라 레이아웃 실현가능성을 다루지만, '기호적 축에 대한 자동 판정'이라는 기술은 그대로 빌릴 수 있고 동시에 '이미 자동 검증은 있다'는 반론 재료이기도 하다. |
| FuriosaAI RNGD: A Tensor Contraction Processor for Sustainable AI Computing | IEEE Micro 45(3), 2025 (Hot Chips 2024 theme article) — Choi 외, Nuno P. Lopes, Sungjoo Yoo | 기반이자 신규성 근거. PDF 8쪽 전문을 읽어 확인한 결과 매핑 표현식 언어·타입 수준 인코딩·합법성 판정은 전혀 없다. 컴파일러 서술은 '축 순열 tactic space 탐색', 'fetch unit 이 N차원 루프 방식으로 읽음' 수준. 즉 이 대수는 미출판이다. |
| TCP: A Tensor Contraction Processor for AI Workloads (Industrial Product) | ISCA 2024 (H. Kim 외), doi 10.1109/ISCA59077.2024.00069 | 기반. 같은 아키텍처의 1세대 논문. 컴파일러가 텐서 모양과 루프 순서의 설계공간을 탐색한다는 서술은 있으나 매핑 대수의 구문·의미론은 없다. |
| Einsum Trees: An Abstraction for Optimizing the Execution of Tensor Expressions | ASPLOS 2025 | 인접. einsum 을 이진 축약 트리로 분해하고 flop·중간텐서 증가 없이 데이터 레이아웃을 최적화한다. 'einsum 으로부터 레이아웃을 정한다'는 축이 겹치나 하드웨어 합법성 판정은 없다. |
| ATLAAS / TensorLift: Automatic Tensor-Level Abstraction of Accelerator Semantics (MLIR Semantic Lifting from RTL) | arXiv:2604.13523, 2026 | 인접. '대부분 가속기의 ISA 의미론은 형식적으로 정의된 적이 없고 RTL·비공식 문서·설계자 머릿속에 흩어져 있다'는 문제의식이 우리와 같다. 다만 접근은 RTL 로부터의 자동 리프팅이고 우리는 문서화된 매핑 대수의 형식화 + 판정이다. 폐쇄 소스 백엔드를 다루는 대안 방법론으로 리뷰어가 언급할 가능성이 높다. |
| Scheduling Languages: A Past, Present, and Future Taxonomy | arXiv:2410.19927, 2024 | 기반. Halide/TVM/Exo/Triton 계열 스케줄 언어의 분류 서베이. 우리 위치 설정(스케줄 언어가 아니라 레이아웃/매핑 대수)에 인용용. |


---

## 주제 2. 컴파일러 스케줄 모델의 예측 정확도 검증

> **판정: `viable` → 적대적 심사 후 `weak`**  (약함 — 독립 논문보다 한 절이 낫다)
> 조사자가 찾은 선행연구 10건 · 심사자가 제기한 치명적 반론 12건 ·
> "이미 출판됨" 지적 7건 · 근거 없이 단정한 문장 10건

### 2.1 초록 — 한 문장 주장

상용 NPU 컴파일러가 스스로 출하하는 사이클 예측값(특히 엔진 겹침을 분해한 hidden/*_only 항)은 실기 벽시계와 대조된 적이 없으며, RNGD 커널 130개를 실측하면 예측 오차를 엔진별로 귀속시키고 보정할 수 있다 — 단 이 주장은 아직 벽시계 데이터가 0이므로 전부 미래형이다.

### 2.2 문제의식과 선행연구의 빈틈

검색으로 확인한 범위에서 선행연구의 검증 대상은 셋 중 하나다. (1) 연구용 시뮬레이터를 실기와 맞추는 것(Accel-Sim→Volta, SCALE-Sim TPU→TPU v4), (2) 저자가 직접 만든 해석적 모델의 정확도(NPUMeter→Ascend 910 <5%, Sparseloop, ZigZag), (3) 컴파일러 cost model을 학습 모델로 대체하고 그 정확도를 보고(XLA learned cost model MAPE 5.5%, Ansor/TenSet, LENS 2.15%). **벤더가 제품 컴파일러에 이미 넣어 출하 중인 자체 사이클 리포트를 제3자가 실측으로 감사한 연구는 찾지 못했다.** 특히 어느 선행연구도 "겹쳐서 숨은 사이클(hidden)"과 "가려지지 않은 순수 점유(*_only)"라는 겹침 분해를 실측으로 검증하지 않는다 — 시뮬레이터와 학습 모델에는 그런 라벨 자체가 없기 때문이다. 또 RNGD/TCP는 ISCA 2024·Hot Chips 2024 벤더 아키텍처 논문뿐이고 독립적인 컴파일러·비용모델 평가가 없다. 다만 이 gap은 "아무도 안 했다"이지 "학계가 중요하게 여긴다"는 뜻은 아니다.

### 2.3 제안한 기여

- 출하 중인 상용 NPU 컴파일러의 자체 사이클 리포트를 실기 벽시계로 감사한 첫 제3자 검증. 검증 대상이 연구용 시뮬레이터나 학습 대체모델이 아니라 '벤더가 사용자에게 보여 주는 그 숫자'라는 점이 선행연구와의 차이다.
- 단일 MAPE 대신 **컨텍스트별 오차 귀속**: 잔차를 io_only/main_only/sub_only/hidden/프롤로그/dram_usage_io 로 회귀해 스케줄 모델의 '어느 항이' 틀렸는지 지목한다. 특히 hidden(겹침 예측)이 낙관적인지 비관적인지는 컴파일러 스케줄러가 직접 고칠 수 있는 형태의 결론이다.
- 겹침 예측의 직접 검증 실험: 이중 버퍼링 TRF 절반 교대 vs 같은 절반(WAR 직렬화) 처럼 **겹침 기회만 다른 짝 커널**을 만들어, 모델이 예측한 Δhidden 과 실측 Δtime 이 일치하는지 본다. 상관계수가 아니라 인과 형태의 검증이다.
- 벤더 프로파일러 없이 µs급 커널을 닫힌 NPU에서 재는 계측 방법론(프로세스 내 반복 상각 + DVFS 고정 + 클럭 도메인 식별). 다른 폐쇄형 가속기에 재사용 가능한 절차로 기술.
- 출하 리포트의 내부 불일치를 재현 가능한 아티팩트로 문서화: total_instruction_cycle 이 (main+sub+io)로 표기됐으나 2,500 사이클 어긋남, util 필드가 130개 커널 전부 0.0, 덤프에 클럭 도메인 라벨 없음.

### 2.4 방법

(1) 계측 하네스: cargo 오버헤드(≥3.5s)를 제거해야 한다. 커널 1회 호출을 N=10^3~10^5 회 프로세스 내에서 반복해 (T_total − T_setup)/N 으로 커널당 지연을 뽑는다. 런타임 API가 컴파일된 커널의 반복 호출을 허용하는지가 전제이며 **현재 미확인**이다. 불가하면 대안은 온디바이스 타이머/성능카운터인데 존재 여부도 미확인. (2) 클럭 도메인 식별: furiosa-smi governor 로 performance/powersave 를 고정해 주파수를 바꾸고, 사이클(불변) 대비 실측 시간(가변)의 스케일링으로 스케줄 사이클이 어느 도메인(0.75/1/1.5 GHz)인지 실험적으로 확정한다. 벤더 스펙상 코어는 1.0 GHz 지만 DMA/HBM 항이 같은 도메인인지는 별개다. (3) 130개 커널 전수 측정: 각 커널 median + IQR, 열 드리프트·런간 분산 보고. 4장 유휴이므로 카드간 재현성도 교차 확인. (4) 회귀: measured_time ~ total_execution_cycle 의 OLS 기울기(=실효 클럭)·절편(=런치 오버헤드)·R²·잔차 분포. (5) 오차 귀속: 잔차를 컨텍스트 분해항으로 다중회귀 → 어느 엔진 항이 편향을 만드는지. (6) 보정 모델: *_only 분해 + dram_usage_io + 프롤로그 상수를 입력으로 하는 선형/GBDT 보정기를 leave-one-kernel-family-out 교차검증. (7) 짝 커널 겹침 검증(위 기여 3). (8) 2,500 사이클 항 규명: 이건 하드웨어 없이 가능하다 — 130개 전 커널의 summary.json lir_stats/rlir_stats(total_cycle, computation_cycle, sync_cycle, spill_io_cycle 등)를 뽑아 gap 이 상수인지·비례인지·특정 항과 일치하는지 회귀한다. 가장 싸고 즉시 착수 가능한 단계이므로 여기부터 시작해야 한다.

### 2.5 평가 설계

비교 대상 = 커널당 실측 벽시계(유휴 RNGD, N회 반복 중앙값). 예측자 베이스라인 7종을 같은 홀드아웃에서 겨룬다: (a) 인스트럭션 개수, (b) total_instruction_cycle(겹침 미고려), (c) **total_execution_cycle — 벤더의 자체 예측, 이것이 이겨야 할 주 베이스라인**, (d) Σ*_only, (e) DMA 바이트 루프라인(dram_usage_io ÷ 1.5 TB/s), (f) 제안 보정모델(*_only 분해 + 프롤로그 + dram io 선형), (g) 덤프 전 피처 GBDT(학습 모델 상한 참조, Ansor/TenSet 계열). 지표: MAPE, 중앙 상대오차, Pearson/Spearman R, OLS 기울기·절편, 잔차의 커널군별 편향. 성공 기준은 "R² 가 높다"가 아니라 **(f)가 (c)를 홀드아웃에서 유의하게 이기고, 그 개선이 특정 엔진 항으로 설명될 것**. 2차 평가(논문 급을 가르는 지점): 보정된 예측이 실제 **스케줄 선택**을 개선하는가 — 겹침 기회가 다른 변형 2~3개 중 가장 빠른 것을 고르는 top-1 정확도를 벤더 예측 대비 측정. 여기까지 가면 CGO/LCTES, 예측 정확도만이면 ISPASS/IISWC.

### 2.6 이미 확보된 근거

- 제시된 수치는 전부 문서와 일치했다 — 틀린 곳 없음. 07-스케줄링.md §6.8 의 summary.log 실측 발췌: total_instruction_cycle 23337, total_execution_cycle 17953, estimated_execution_cycle 17953, main 42.790%(7682), sub 4.400%(790), io 68.874%(12365), main_only 12.226%(2195), sub_only 4.400%(790), io_only 38.311%(6878), sync/spill_io/sync_only/spill_io_only 전부 0, dram_usage_io 419968B, peak_memory 440803328.
- 2,500 사이클 미설명 항 확인: 7,682+790+12,365 = 20,837 vs 표기 23,337. 이미 07-스케줄링.md §11 에 '미확인'으로 등재돼 있다. 보강 사실 — 로그의 나머지 라벨 sync·spill_io 가 둘 다 0 이므로 출하 리포트의 라벨 집합만으로는 이 2,500 을 설명할 방법이 실제로 없다.
- 17,953 의 3중 독립 일치(07-스케줄링.md §6.8c): --dump-schedule 파싱 max(lifetime.end), 11-MNIST-실행결과.md 독립 기록, summary.log total_execution_cycle 이 총계·main·sub·io 네 값 모두 자릿수까지 동일. 컴파일러 측 근거는 견고하다.
- 백분율 분모가 total_execution_cycle 임을 소수 3자리까지 검산 완료(§6.8d, 0 아닌 6줄 전부 재현).
- 커널 130개 스케줄 전수(§6.7): DmaEngine 75,464,336 cyc(96.5%, inst 470), PeCore 2,586,167(3.3%, inst 1,557), MainContext 58,883, InterChipTransfer 38,018, VectorEngine 14,770, SubContext 9,737. 커널별 DMA 비중 중앙값 82.8%, 50% 이상이 107/130.
- 스팬 분포(§6.7): min 16 / p25 4,612 / median 10,532 / p75 23,503 / max 10,845,036. 최대 커널 at_primitives::vrf::multi_vrf_at 는 인스트럭션 12개뿐 → 명령 수와 비용이 무관.
- 프롤로그 구조(§6.7): 첫 인스트럭션 DmaLoad 면 begin=2003(98개), DmaDtod 면 2000(11개), Sub 면 0(21개). 비-DMA 시작 커널이 0에서 출발하므로 런타임 초기화가 아니라 DMA 기동 지연. 130개 중 16개는 이 프롤로그가 스팬의 50% 초과(3,050~3,762 스팬에서 2,003 이 53~66%).
- **벽시계는 단 한 번도 측정된 적이 없다** — 13-NPU-실기-매트릭스.md §8 '정직한 한계'에 명시: '성능 비교를 하지 않았다… 실기 벽시계는 cargo 오버헤드(최소 3.5초)에 묻혀 커널 시간을 분리하지 못했다.' 사용자 진술과 일치.
- _evidence/logs/npu_matrix.tsv 의 시간 컬럼(3,710~4,551 ms)은 커널 시간이 아니다. npu_matrix.sh 를 읽어 확인: date +%s%N 로 `cargo furiosa-opt --backend npu test` 전체 호출을 감싼 프로세스 벽시계다. 즉 기존 로그에 재활용 가능한 지연 데이터는 없다.
- 내가 직접 계산: sched_summary.json 의 130개 커널이 perkernel_matrix_fixed.txt 의 실기 컴파일 성공 137개에 **전부 포함**된다(130/130). 즉 측정 후보 표본은 130개로 확보돼 있다. 그러나 그중 **109개가 스팬 100k 사이클 미만(1 GHz 기준 100 µs 미만)**, 1M 사이클(1 ms) 초과는 11개뿐이다 → 3.5 s 계측 바닥과 5자릿수 차이.
- 실기 동작 범위(13번 문서 §1): 89개 테스트 중 80 그대로 통과, 보정 83/89(93.3%). 커널은 200개 중 137개가 실기 바이너리로 컴파일. 다중칩 경로는 6종 중 5종이 컴파일 실패, 유일 컴파일되는 chip_shuffle 도 커널 로드에서 abort → 배치=1·단일칩 범위로 강제된다.
- 클럭 도메인이 덤프에 라벨돼 있지 않다(03-텐서-이동.md 미확인 항목). 문서상 도메인 3종 공존: HBM 채널 컨트롤러 0.75 GHz(버스트 8 → 실효 6 GHz), DRAM 타이밍표 1.5 GHz, DMA/코어 예제 1 GHz. 벤더 스펙(검색)은 RNGD 코어 1.0 GHz, TSMC 5nm, HBM3 1.5 TB/s.
- 덤프에 유효한 utilization 신호가 없다: instructions[].util 이 130개 커널 전부 0.0(07-스케줄링.md §10). DmaStore 72,731 사이클의 비용모델 근거도 미확인으로 남아 있다.
- 스케줄러 내부 키 operator_schedule_heuristic·beam_size·total_states_visited 는 mnist::forward 에서 전부 None, schedule_method 는 빈 문자열 — 스케줄러 탐색 방식은 블랙박스.

### 2.7 아직 없는, 반드시 해야 할 실험

- **[게이트 실험]** µs급 커널 계측 하네스 구축. furiosa device-runtime API 가 컴파일된 커널의 프로세스 내 반복 호출(고정 입력, N=10^3~10^5)을 지원하는지 먼저 확인. 지원하면 상각 측정, 불가하면 온디바이스 타이머/성능카운터 존재 여부 조사. **둘 다 안 되면 이 주제는 성립하지 않는다.** 4주 내 go/no-go.
- 스케줄 사이클의 클럭 도메인 실험적 확정. furiosa-smi governor performance/powersave(root 필요) 로 주파수를 바꿔 가며 동일 커널을 재고, 시간이 스케일링되는 비율로 도메인을 역산. DMA 지배 커널과 연산 지배 커널이 서로 다른 계수로 스케일링되면 단일 클럭 환산 자체가 불가능하다는 뜻이며 이는 그 자체로 결과다.
- 커널 130개 벽시계 전수 측정. 커널당 반복 N회, 중앙값·IQR·런간 분산·카드 4장 교차 재현성·열 드리프트 보고.
- measured_time vs total_execution_cycle 회귀 — 기울기(실효 클럭), 절편(런치 오버헤드), R², 잔차 분포. 베이스라인 7종(평가 항목 참조) 동시 산출.
- 잔차의 컨텍스트별 귀속 회귀: 잔차 ~ io_only + main_only + sub_only + hidden + 프롤로그상수 + dram_usage_io. 어느 항의 계수가 유의하게 0에서 벗어나는지가 논문의 핵심 표.
- 겹침(hidden) 직접 검증: 이중 버퍼링 TRF FirstHalf/SecondHalf 교대 vs 동일 절반(WAR 직렬화) 짝 커널을 작성해, 모델이 예측한 Δhidden 이 실측 Δtime 과 일치하는지 확인. 상관이 아니라 개입 실험.
- **[하드웨어 불필요, 즉시 착수]** 2,500 사이클 항 규명: 130개 전 커널의 summary.json lir_stats/rlir_stats(total_cycle, skip_aware_total_cycle, computation_cycle, main/sub/io/sync/spill_io_cycle)를 덤프해 gap = total_instruction_cycle − (main+sub+io) 가 상수인지·스팬 비례인지·특정 항(computation_cycle 등)과 일치하는지 회귀. mnist 1개에서만 본 2,500 이 일반 현상인지부터 확인해야 한다.
- 보정 모델의 홀드아웃 평가: leave-one-kernel-family-out 교차검증으로 벤더 자체 예측(total_execution_cycle) 대비 개선폭 유의성 검정.
- 스케줄 선택 실험(2차 평가): 겹침 구조만 다른 변형 2~3개 중 가장 빠른 것을 고르는 top-1 정확도를 벤더 예측 vs 보정 모델로 비교.
- 일반성 확보: 벤더 예제 크레이트 130개는 벤더가 고른 마이크로벤치라 선택 편향이 있다. 최소 한 개의 독립 워크로드 계열(직접 작성한 트랜스포머 블록 등)을 추가해 결론이 예제 밖에서도 유지되는지 확인.
- 런치 오버헤드 분리: 프롤로그 ~2,000 사이클이 호스트/드라이버 오버헤드인지 온칩 DMA 기동인지 구분. 이걸 못 가르면 100 µs 미만 커널 109개에서 '컴파일러 모델'이 아니라 '런타임'을 측정하게 된다.

### 2.8 ★ 심사 반론

**치명적 반론 (reject 사유)**

- **결과가 0이다 — 이건 논문이 아니라 제안서다.** 벽시계 데이터가 단 한 건도 없고 thesis 전체가 미래형이다. 게다가 프로젝트 전체가 '반복 호출 API 또는 온칩 타이머 중 하나가 존재한다'는 **미확인 전제 하나**에 100% 걸려 있다. 심사위원 입장에서 판단할 대상 자체가 없다. 현재 확보된 것은 전부 컴파일러 산출물이고, 그건 신규성이 아니라 준비물이라는 점은 제안자 본인도 인정한다.
- **VPUNN(Intel, arXiv 2205.04586, 2022)이 프레임을 이미 가져갔다.** '벤더가 제품 VPU 컴파일러에 넣어 쓰는 state-of-the-art cost model을 저수준 하드웨어 프로파일링 실측과 대조하고, 그것을 능가하는 보정/대체 모델을 만들어 컴파일러에 재투입해 FPS 개선까지 보인다' — 기여 1(감사)과 기여 6(보정)과 2차 평가(스케줄 선택 개선)를 한 논문이 이미 다 한다. related_work에 이 논문이 아예 없다는 것 자체가 서베이 부실의 증거다.
- **'겹침 분해를 실측 검증한 선행연구 없음'이라는 gap 문장이 틀렸다.** NPUMeter(ACM TACO, Ascend 910)는 파이프라인 단계 간 **overlap과 stall 상호작용을 명시적으로 모델링**해 실기에서 평균 <5%로 검증한다. StableHLO 크로스아키텍처 논문(arXiv 2604.12090, 2026)은 예측 오차를 **compute/communication overlap**·메모리 계층·인터커넥트에 귀속시킨다. '라벨 이름이 hidden/*_only 다'는 것은 선행연구와의 차이가 아니라 벤더 덤프의 명명 규칙일 뿐이다.
- **감사 대상이 '출하 중인 상용 컴파일러'가 아니다 — 전제가 무너진다.** 문서 근거상 furiosa-opt(vISA)는 버전 문자열 0.3.0을 찍는 0.4 세대 **저수준 프리뷰 Rust 계층**이고, `--dump-summary`는 디버그 덤프다. 200개 커널 중 63개가 실기로 낮춰지지 않고(31.5% 실패), 다중칩 경로는 6종 중 6종이 실기에서 못 돈다. 프로덕션 경로는 furiosa-llm/torch→EDF다(09-도구와-계층 §6). 즉 '벤더가 사용자에게 보여 주는 그 숫자'가 아니라 '프리뷰 툴의 디버그 출력'이며, 벤더는 '지원 대상 아닌 내부 출력'이라 한 줄로 답하면 끝난다.
- **검증하겠다는 hidden 자체가 well-defined 하지 않다.** summary.log가 `total_instruction_cycle : 23337 (main + sub + io)`라 표기했는데 7,682+790+12,365=20,837로 2,500 어긋난다. 그런데 hidden은 정의상 total_instruction − total_execution 이므로 **그 미설명 2,500이 hidden 5,384 안에 그대로 섞여 있다.** 논문의 중심 물체가 산술이 닫히지 않은 파생량이다. 실측으로 'hidden 예측이 낙관적/비관적'을 논해도 그것이 겹침 모델의 오차인지 회계 아티팩트인지 분리할 방법이 없다.
- **표본이 계측 하한과 5자릿수 어긋나고, 남는 것은 런타임 상수다.** 130개 중 109개가 100µs 미만, 중앙값 10.5µs(1GHz 가정). 프롤로그 2,003 사이클이 16개 커널에서 스팬의 50~66%를 차지한다. 상각 측정이 성공해도 µs급 커널에서 재는 것의 상당 비중이 드라이버·DMA 기동 상수다. '컴파일러 스케줄 모델 검증'이 '런타임 오버헤드 측정'으로 전락하는 것이 우연이 아니라 구조적이다.
- **베이스라인 7종이 사실상 같은 컬럼의 재배열이다.** (a)~(e)가 전부 동일한 컴파일러 덤프의 열이고 (f)는 그 열들의 선형결합이다. '더 많은 특징을 쓰는 회귀가 단일 특징을 이긴다'는 결과는 자명해서 정보량이 거의 0이다. 진짜 베이스라인은 덤프 밖에서 온 독립 예측기(루프라인, 순수 바이트 모델, 외부 학습모델)여야 하는데 (e)만 겨우 그렇고 그마저 1.5TB/s 나눗셈이다.
- **표본 130으로 학습 보정기를 훈련·검증한다는 계획은 통계적으로 빈약하다.** TenSet(수십만)·Ansor(25,000 프로그램) 대비 두세 자릿수 작다. leave-one-kernel-family-out을 해도 family 수가 적어 '유의하게 이긴다'를 주장할 검정력이 없다. GBDT(g)는 130 샘플에서 사실상 과적합 시연이 된다.
- **기여 5(2,500 사이클 불일치)는 버그 리포트이지 연구 기여가 아니다.** 게다가 mnist 단일 커널의 관측이고, 자기 문서 §6.8(e)가 이미 '두 덤프의 분해 축이 다르다(context 축 vs engine 축, PeCore 600·VectorEngine 1,162에 대응하는 summary.log 줄이 없다)'고 적어 뒀다. lir_stats/rlir_stats에 computation_cycle 등 미확인 라벨도 남아 있다. 즉 '라벨로 설명 불가'가 아니라 '아직 안 봤다'이다.
- **계측 방법론 기여(4)가 벤더 프로파일러 존재로 소멸할 수 있다.** Furiosa SDK 문서에 프로파일러(FURIOSA_PROFILER_OUTPUT_PATH 트레이스, Chrome trace 시각화)와 furiosa-bench `--trace-output`이 존재한다. vISA 커널 경로에 적용되는지 확인도 없이 '벤더 프로파일러 없이'를 기여로 내세우면 리뷰어 한 줄에 무너진다.
- **결과의 유효기간이 없다.** 0.4 프리뷰 컴파일러의 덤프 수치는 다음 릴리스에 바뀐다. 심지어 `--version`이 실제 세대(0.4)와 다른 0.3.0을 찍는 래퍼라 논문에 적을 버전 표기조차 불안정하다. '재현 가능한 아티팩트'를 약속해도 대조군 자체가 사라진다.
- **2차 평가(스케줄 선택 top-1)의 표본 크기가 계획에 없다.** 겹침 구조만 다른 변형 2~3개를 만들 수 있는 커널이 몇 개인지 미확인이다. 소수 커널 × 변형 3개면 top-1 정확도 분모가 한 자릿수라 CGO급 주장이 성립하지 않는다. 그런데 제안은 '여기까지 가면 CGO'라고 스스로 말한다 — 즉 논문의 급을 가르는 지점에 설계가 없다.

**이미 출판되어 신규성이 없는 부분**

- Towards Optimal VPU Compiler Cost Modeling by using Neural Networks to Infer Hardware Performances (VPUNN) — arXiv 2205.04586, Intel, 2022. **가장 치명적.** 제품 VPU 컴파일러가 쓰는 기존 cost model 을 저수준 하드웨어 프로파일링 실측 대비 평가해 열등함을 보이고, 실측 학습 보정 모델로 대체해 컴파일러에 재투입, 네트워크별 FPS 개선까지 보고한다. 본 제안의 기여 1(출하 cost model 감사)·기여 6(보정 모델)·2차 평가(스케줄 선택 개선)를 모두 포함한다. related_work 에 없음.
- NPUMeter: Automatic Operator Optimization for Ascend NPU with Accurate Analytical Performance Models — ACM TACO (DOI 10.1145/3820380). 상용 NPU(Ascend 910)에서 **실행 단계 간 overlap·stall 상호작용을 명시적으로 모델링**해 실기 평균 오차 <5% 로 검증하고 DSE 까지 수행. 제안이 주장하는 '겹침 분해를 실측 검증한 선행연구 없음'을 직접 반박한다.
- SCALE-Sim TPU: Validating and Extending SCALE-Sim for TPUs — arXiv 2603.22535, 2026. TPU v4 실측으로 사이클↔지연 선형 매핑을 세우고 비수축 연산에 경량 학습 지연모델(중앙 오차 <3%)을 얹는 절차가 본 제안의 (4)(6) 단계와 사실상 동형. 제안 자신이 인정하는 최근작.
- Evaluating Cross-Architecture Performance Modeling of Distributed ML Workloads Using StableHLO — arXiv 2604.12090, 2026. 예측 대 실측 비교 + **오차를 compute/communication overlap·메모리 계층·인터커넥트에 귀속**. '오차 귀속' 이라는 기여 2의 축이 이미 최신 논문에 존재.
- A Learned Performance Model for Tensor Processing Units / Learned TPU Cost Model for XLA — Kaufman et al., ML for Systems @ NeurIPS 2019 (및 후속 MLSys 2021). 프로덕션 컴파일러의 cost model 오차를 실측 대비 정량화(MAPE 5.5% vs 베이스라인 29.9%). '프로덕션 컴파일러 추정이 실측과 얼마나 다른가'는 이미 답이 있다.
- Ansor (OSDI 2020) / TenSet (NeurIPS 2021 D&B) — 실측 기반 cost model 학습·데이터셋 표준. 130 커널 규모가 '작다'는 비판의 기준점이자, GBDT 상한 베이스라인이 이미 확립돼 있어 (g)는 기여가 아니라 재현이다.
- MAESTRO / Sparseloop / ZigZag / SqueezeJet-2 계열 — 해석적 가속기 성능모델을 실칩·RTL 실측 대비 검증하는 관행(오차 0.1~8%, MAESTRO 96.1% 정확도)이 이미 표준. '모델을 실측으로 검증한다'는 행위 자체에 신규성이 없다는 근거.

**근거 없이 단정한 문장 (수정 필요)**

- '_evidence/logs/npu_matrix.tsv 의 시간 컬럼(3,710~4,551 ms)' — **틀렸다.** 실제 89행의 시간 범위는 **1,892 ~ 67,289 ms**다(head 4행만 보고 범위를 적은 것으로 보인다). 다만 '그 컬럼이 date +%s%N 로 cargo 호출 전체를 감싼 프로세스 벽시계'라는 결론은 _evidence/tools/npu_matrix.sh 를 읽어 확인했다 — 맞다.
- '출하 중인 상용 NPU 컴파일러의 자체 사이클 리포트' / '벤더가 사용자에게 보여 주는 그 숫자' — 근거 없음. 문서상 대상은 버전 문자열 0.3.0을 찍는 0.4 세대 vISA 프리뷰 툴체인의 `--dump-summary` 디버그 출력이며, 프로덕션 서빙 경로(furiosa-llm/torch→EDF)의 cost model 이라는 증거가 어디에도 없다.
- '출하 리포트의 라벨 집합만으로는 이 2,500 을 설명할 방법이 실제로 없다' — 성립하지 않는다. 같은 덤프의 summary.json 에 lir_stats/rlir_stats(total_cycle, skip_aware_total_cycle, computation_cycle, main/sub/io/sync/spill_io_cycle)가 있고, dump-schedule 엔진 축에는 summary.log 에 대응 줄이 없는 PeCore 600·VectorEngine 1,162 가 있다(07 §6.8e). 제안 자신의 method (8)이 바로 그 라벨들을 덤프하겠다고 하므로 **내부 모순**이다.
- '17,953 의 3중 독립 일치 → 컴파일러 측 근거는 견고하다' — '독립'이 과장이다. 셋 다 같은 컴파일 산출물에서 나온 값이다(dump-schedule 파싱, 그 파싱을 기록한 11번 문서, 같은 컴파일의 summary.log). 내부 일관성의 증거일 뿐 정확성의 증거가 아니며, 이 논문이 검증하려는 것이 바로 그 정확성이다.
- '벤더 프로파일러 없이 µs급 커널을 재는 계측 방법론' — 벤더 프로파일러 부재가 확인되지 않았다. Furiosa SDK 문서에 트레이스 프로파일러(FURIOSA_PROFILER_OUTPUT_PATH)와 furiosa-bench --trace-output 이 존재한다. 문서에서 확인된 것은 'furiosa-tcc(TCL)에 덤프/프로파일 옵션이 없다'뿐이며, 이는 런타임 프로파일러 부재와 다른 진술이다.
- '어느 선행연구도 겹침 분해를 실측으로 검증하지 않는다' — NPUMeter가 실행 단계 간 overlap/stall 상호작용을 모델링해 상용 NPU 실기로 검증했고, StableHLO 논문(2026)이 오차를 compute/communication overlap 에 귀속한다. 라벨 이름이 다를 뿐 축은 이미 존재한다.
- '컴파일러가 hidden 항을 출하한다' — summary.log 에 hidden 이라는 **숫자 필드는 없다.** 괄호 안 수식 표기 `(main + sub + io - hidden)` 뿐이고, 5,384 는 두 총계의 차로 제안자/문서가 파생한 값이다. '벤더가 분해해 출하하는 항'이라는 표현은 과장이다.
- 커널 스팬 median 10,532 — 130개 값의 통상적 중앙값은 **10,132**(65·66번째 값 9,732과 10,532의 평균)다. 문서는 nearest-rank 규약을 썼다. p25 4,612 도 같은 규약(보간 시 4,658). 오류라기보다 규약 문제지만, 논문 표에 그대로 옮기면 지적받는다.
- '실기 컴파일 성공 137개에 130개가 전부 포함' — 내가 재계산해 **확인했다(130/130 포함, OK 137 / FAIL 63)**. 이 항목은 사실이다.
- 그 외 검증 완료(전부 파일과 일치): summary.log 발췌 수치 전항(23337/17953/7682/790/12365/2195/6878/419968B/440803328), 백분율 분모 검산, 엔진 총계(DmaEngine 75,464,336 / PeCore 2,586,167 / MainContext 58,883 / InterChipTransfer 38,018 / VectorEngine 14,770 / SubContext 9,737), DMA 비중 중앙값 82.8%·107/130·54/130, 스팬 min 16/max 10,845,036, multi_vrf_at 인스트럭션 12개, <100k 109개·>1M 11개, 프롤로그 2003(98)/2000(11)/0(21)·16개 커널 50% 초과, util 전부 0.0, DmaStore 72,731, 클럭 도메인 0.75/1/1.5 GHz 병존, 13번 문서 §8 '3.5초에 묻혀 분리 불가', 89개 중 80 PASS·83/89 보정, 137/200 컴파일.

**조사자 스스로 적은 위험**

- **최대 위험 — 계측이 아직 불가능하다.** 벽시계 데이터가 0이고, 기존 시도는 실패로 문서화돼 있다(cargo 오버헤드 3.5 s). 표본 130개 중 109개가 100 µs 미만이라 5자릿수의 계측 여유가 필요하다. 반복 호출 API도 온칩 타이머도 존재가 확인되지 않았다. 논문 전체가 미해결 계측 문제 하류에 있다.
- **런타임 오버헤드가 신호를 삼킬 위험.** 프롤로그 ~2,000 사이클(130개 중 16개는 스팬의 50% 초과)이 호스트 측 오버헤드라면, µs급 커널에서 재는 것은 스케줄 모델의 오차가 아니라 드라이버 지연이다. 그러면 '컴파일러 모델 검증'이라는 프레임 자체가 무너진다.
- **일반성 공격 (타당하다).** 단일 벤더·단일 칩·배치 1·단일 칩 경로 한정(다중칩 예제는 6종 중 6종이 실기에서 못 돈다). 워크로드는 벤더가 고른 예제 커널 130개. 리뷰어의 '이게 FuriosaAI 밖에서 무엇을 가르치나'에 답할 준비가 없다면 ISCA/ASPLOS/MICRO 는 불가능하다.
- **'GPU에서 이미 다 했다' 반론은 방법론에 관해서는 대체로 옳다.** Accel-Sim 이 마이크로벤치 + 파라미터 튜너 + 상관 시각화라는 검증 파이프라인을 이미 정립했고, SCALE-Sim TPU 는 TPU v4 실측으로 사이클↔지연 선형 매핑 + 비수축 연산용 학습 모델(중앙 오차 <3%)까지 했다. 새 칩에 그대로 적용하면 replication 이다. 살아남으려면 '겹침 분해 검증'처럼 GPU/TPU 검증에 대응물이 없는 축이 결과의 중심이어야 한다.
- **선행연구 밀도가 높다.** NPUMeter(Ascend 910, 평균 오차 <5%), XLA learned cost model(MAPE 5.5%), LENS(NPU 지연 예측 평균 오차 2.15%), Ansor/TenSet. 리뷰어가 '알려진 결과, 새 칩'으로 요약할 여지가 크다.
- **'DMA가 지배한다'는 그 자체로 기여가 아니다.** 96.5%/중앙값 82.8% 는 인상적이지만 roofline(2009) 이후 가속기 문헌에서 가장 많이 반복된 결론이다. 실측이 이걸 확인해도 새 사실이 아니다.
- **클럭 도메인 모호성이 문제를 ill-posed 로 만들 수 있다.** 덤프에 도메인 라벨이 없고 문서상 0.75/1/1.5 GHz 가 공존한다. 사이클→시간 환산이 단일 계수로 성립하지 않으면 '스케줄 모델의 시간 예측 정확도'라는 질문 자체를 다시 세워야 한다(다만 그 사실 자체는 보고 가치가 있다).
- **2,500 사이클 gap 이 사소한 회계 아티팩트일 가능성.** 프롤로그를 한 번 더 더했다든가 하는 결과로 판명되면 '버그 리포트'이지 논문 소재가 아니다. 게다가 mnist 한 커널에서만 관측됐다 — 일반 현상인지도 아직 모른다.
- **유효기간이 짧다.** 벤더 컴파일러 업데이트 한 번이면 수치가 전부 바뀐다. 재현 가능한 절차와 아티팩트를 함께 내지 않으면 결과의 수명이 없다.
- **도구 신호 부족.** util 필드가 130개 전부 0.0, 스케줄러 탐색 키(beam_size 등)가 전부 None. 오차의 원인을 컴파일러 내부로 되짚을 관측 창구가 생각보다 좁다.

### 2.9 살리는 길 — 무엇을 바꿔야 하는가

"현재 형태로는 살릴 수 없다. 프레임을 바꿔야 한다. 구체적으로 다섯 가지.\n\n**(1) 게이트를 4주가 아니라 1주로 당기고, 통과 못 하면 폐기한다.** 문서 09 §6에 이미 경로가 있다: `EdfModule` 이 컴파일된 EDF 를 torch.nn.Module 로 감싸고 `.to(device)` 로 올린 뒤 `torch.ops.furiosa.edf(...)` 로 실행한다. 이건 **파이썬 루프에서 반복 호출 가능한 경로**다 — cargo 를 거칠 이유가 없다. 동시에 FURIOSA_PROFILER_OUTPUT_PATH / furiosa-bench --trace-output 이 EDF 경로에 붙는지 확인하라. 붙으면 기여 4(계측 방법론)는 **버리고** 벤더 트레이스를 그냥 쓰는 게 옳다. 두 경로 모두 실패하면 주제 종료.\n\n**(2) '첫 제3자 감사' 프레임을 버려라.** VPUNN 이 2022년에 출하 cost model 대 실측을 했고, 감사 대상은 프로덕션 컴파일러가 아니라 0.4 프리뷰 vISA 툴의 디버그 덤프다. 두 방향 모두에서 방어 불가다. 대신 답할 수 있는 질문으로 좁혀라: **'겹침을 분해해 보고하는 스케줄 모델이 실제 겹침을 맞히는가, 그리고 어디서 깨지는가.'** VPUNN·NPUMeter·SCALE-Sim TPU 를 intro 에서 정면으로 인용하고, 델타를 '새 모델을 만들지 않고 기존 분해 라벨의 인과적 타당성을 개입 실험으로 검사한다'로 명시하라.\n\n**(3) 개입 실험을 논문의 척추로 올려라(현재 기여 3 → 기여 1).** 상관·MAPE 는 선행연구가 포화시켰지만 **Δ예측 대 Δ실측**은 비어 있다. 결정적 이점은 통계가 아니라 물리다: 짝 커널의 차분을 보면 프롤로그 2,003·런치 오버헤드 같은 상수항이 **소거**되므로, 109개 µs급 커널이 만드는 최대 위험(런타임이 신호를 삼킴)과 2,500 사이클 회계 오염이 동시에 빠진다. 개입 축은 이미 문서에 있다: TRF 절반 교대 vs 동일 절반, 그리고 13번 문서 §378 의 **64B→256B 정렬로 DmaStore 72,731→1,552(46.9배)** — 정렬 스윕은 준비된 개입군이다.\n\n**(4) 커널 표본을 '주어진 130개'에서 '설계한 파라메트릭 계열'로 바꿔라.** 같은 구조를 10µs~10ms 로 스케일링하는 커널 계열을 직접 작성해, 런치 오버헤드를 **가정하지 않고 회귀 절편으로 분리**하라. 130개 벤더 마이크로벤치는 특성화 부록으로 내리고, 독립 워크로드(트랜스포머 블록 등) 최소 1계열을 본문에 넣어라. 130 샘플 GBDT 는 빼라 — 과적합 시연이다.\n\n**(5) 2,500 사이클은 하드웨어 없이 이번 주에 끝내고 결론을 문장 두 개로 줄여라.** lir_stats vs rlir_stats 회계 차이(스케줄 전 LIR 합 대 스케줄 후 컨텍스트 합)로 판명될 가능성이 높다. 그러면 기여 5 는 삭제하고, hidden 정의가 그 항으로 오염돼 있다는 사실만 방법론 절의 한 문단으로 남겨라(이게 오히려 개입 실험을 정당화한다).\n\n**목표 학회와 킬 크라이테리아.** ISPASS/IISWC 단편, 혹은 CGO practical-experience/artifact 트랙. ISCA/ASPLOS/MICRO 는 논외다. 미리 정해 둘 것: 실측 벽시계가 total_execution_cycle 의 아핀 함수로 R²>0.95 이고 잔차에 엔진 귀속 구조가 없으면, 정직한 결과물은 '벤더 스케줄 모델은 이 범위에서 맞는다'는 **1페이지 음성 결과**다. 그건 KCC/KSC 나 워크숍으로 내고 접어라 — 억지로 풀 페이퍼로 늘리지 마라."

### 2.10 후보 학회와 소요 기간

- ISPASS — 계측·검증 연구의 정확한 자리. 현실적 1순위
- IISWC — 워크로드/하드웨어 특성화. 130개 커널 특성화를 앞세우면 적합
- CGO — 보정이 스케줄 선택을 개선한다는 2차 결과가 나올 경우
- LCTES — 임베디드/가속기 컴파일러 트랙
- PACT — 컴파일러+아키텍처 경계 주제로 가능
- MICRO/ASPLOS/ISCA — 겹침 모델 오차가 구조적이고 칩을 넘어 일반화되는 결과가 나올 때만. 현재 근거로는 불가
- 한국정보과학회 KCC/KSC — 계측 하네스 실패 시 회수 경로

**소요 기간 추정**: 5~8개월. 단계별: (0) 하드웨어 불필요한 2,500-사이클 gap 전수 회귀 — 2주, 즉시 착수 가능. (1) **계측 하네스 go/no-go 게이트 — 4주.** 반복 호출 API 또는 온칩 타이머 둘 중 하나가 확인되지 않으면 여기서 중단해야 한다(주제 소멸). (2) 클럭 도메인 확정 + 130개 전수 측정 — 6주. (3) 회귀·오차 귀속·보정 모델 + 교차검증 — 6주. (4) 겹침 짝 커널 개입 실험 + 독립 워크로드 추가 — 4주. (5) 집필 — 6주. 게이트를 통과 못 하면 런타임 역공학에 2개월 이상이 추가되며 실패 확률이 실질적으로 높다.

**신규성 확신도**: 중간~낮음. 8회 검색에서 "벤더가 제품 컴파일러로 출하하는 자체 사이클 리포트의 겹침 분해(hidden/*_only)를 실기 실측으로 검증한 연구"는 찾지 못했고, RNGD/TCP 에 대한 독립 컴파일러 평가도 전무하다. 이 두 가지는 진짜 빈자리다. 그러나 확신을 낮추는 이유가 셋 있다. (1) 상위 프레임인 "성능 모델 정확도를 실기로 검증한다"는 포화 상태다 — NPUMeter(상용 NPU, <5%), SCALE-Sim TPU(TPU v4 실측 + 학습 보정, 2026), Accel-Sim(GPU 검증 방법론), XLA/Ansor/TenSet/LENS. 특히 SCALE-Sim TPU 는 본 계획의 절차(실측 회귀 → 사이클↔지연 매핑 → 학습 보정)와 거의 동형이며 최근작이다. (2) 새로움이 '대상 칩이 다르다'에만 걸리면 replication 이다. (3) 결정적으로 **아직 실측 데이터가 0이라 신규성을 뒷받침할 결과가 하나도 없다** — 현재 확보된 것은 전부 컴파일러 산출물이고, 그건 신규성이 아니라 준비물이다. 신규성이 살려면 겹침 예측 오차가 구조적 패턴을 보이거나, 사이클→시간 환산이 단일 계수로 성립하지 않는다는 식의 '컴파일러 리포트를 그대로 믿으면 안 되는 이유'가 실측으로 나와야 한다. 그 전까지는 신규성 주장이 불가능하다.

### 2.11 검색으로 확인한 관련 연구

| 제목 | 학회·연도 | 이 주제와의 관계 |
|---|---|---|
| NPUMeter: Automatic Operator Optimization for Ascend NPU with Accurate Analytical Performance Models | ACM TACO (DOI 10.1145/3820380, 게재연도는 검색으로 확정 못 함) | 겹침. 상용 NPU(Ascend 910)에서 해석적 성능모델을 실기 검증해 평균 오차 <5% 를 보고하고 DSE 까지 한다. '상용 NPU + 해석적 모델 + 실기 검증'이라는 골격이 거의 같다. 차이는 저자가 모델을 새로 만든 것이지 벤더 컴파일러의 출하 리포트를 감사한 게 아니라는 점, 그리고 겹침 분해 검증이 없다는 점. |
| SCALE-Sim TPU: Validating and Extending SCALE-Sim for TPUs | arXiv preprint 2026 (2603.22535) | 겹침. TPU v4 실측으로 시뮬레이션 사이클과 하드웨어 지연의 선형 상관을 확인하고 사이클→지연 매핑을 세운 뒤, 비수축 elementwise 연산에는 텐서 크기/모양만으로 중앙 상대오차 <3% 의 경량 학습 지연모델을 붙인다. 본 주제의 '회귀 + 보정 모델' 계획과 사실상 동일한 절차다. 최근작이므로 반드시 인용·차별화해야 한다. |
| Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling | ISCA 2020 (제목·PDF 는 검색 확인, 학회명은 통상 표기이며 검색 결과가 명시하진 않음) | 기반. 마이크로벤치 + 자동 파라미터 튜너 + 상관 시각화라는 '모델을 실기에 맞춰 검증'하는 표준 방법론을 정립했다(Volta 대비 오차 79% 감소). 'GPU에서 이미 다 했다' 반론의 근거가 되는 논문이며, 본 연구의 방법론은 이것의 NPU 이식으로 읽힐 위험이 있다. |
| Learned TPU Cost Model for XLA Tensor Programs | ML for Systems Workshop @ NeurIPS 2019 | 인접. 프로덕션 컴파일러(XLA)의 cost model 을 학습 모델로 대체하고 실측 대비 MAPE 5.5%(베이스라인 29.9%)를 보고. '프로덕션 컴파일러의 비용 추정이 실측과 얼마나 다른가'를 다루지만 해법이 대체이지 감사·분해가 아니다. |
| Ansor: Generating High-Performance Tensor Programs for Deep Learning | OSDI 2020 (arXiv 2006.06762) | 기반. 학습 cost model(GBDT)을 실측으로 재학습하는 루프. ResNet-50 25,000 프로그램으로 예측 대비 실측 산점도를 평가한 절차가 본 연구의 학습 상한 베이스라인(g)에 해당. |
| TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers | NeurIPS 2021 Datasets & Benchmarks | 인접. 학습 비용모델용 대규모 실측 데이터셋. 본 연구의 130개 커널 실측 데이터셋이 '규모가 작다'는 비판의 기준점이 된다. |
| Latency Prediction for LLM Inference on NPU Systems (LENS) | arXiv preprint 2026 (2606.18042) | 인접. 마이크로아키텍처·컴파일러 정보 없이 상용 NPU 여러 벤더에서 LLM 추론 지연을 예측(평균 오차 2.15%). '컴파일러 최적화가 예측 불가능하다'는 문제의식이 같지만 커널이 아니라 E2E 레벨이고, 컴파일러 내부 리포트를 쓰지 않는다. |
| Sparseloop: An Analytical Approach To Sparse Tensor Accelerator Modeling | MICRO 2022 (arXiv 2205.05826) | 인접. 해석적 가속기 모델의 검증 관행(여러 설계 대비 평균 오차 0.1~8%)을 대표. 이 분야에서 '몇 % 오차면 통과'인지의 기준선을 제공한다. |
| ZigZag: A Memory-Centric Rapid DNN Accelerator Design Space Exploration Framework | IEEE Transactions on Computers 2021 (arXiv 2007.11360) | 인접. 비용모델 검증 방법론 3종(테이프아웃 칩 실측 대비 / 합성후 추출 데이터 대비 / 타 DSE 프레임워크 대비)의 표준을 제시. 본 연구는 첫 번째 유형이다. |
| FuriosaAI RNGD: A Tensor Contraction Processor for Sustainable AI Computing | Hot Chips 2024 / IEEE Micro (IEEE Xplore 10929037) | 기반. 대상 하드웨어의 벤더 아키텍처 논문(TSMC 5nm, 1.0 GHz, INT8 512 TOPS, HBM3 1.5 TB/s). 함께 TCP: A Tensor Contraction Processor for AI Workloads (ISCA 2024) 가 아키텍처 원논문이다. **RNGD 에 대한 독립적인 컴파일러·비용모델 평가는 검색으로 하나도 찾지 못했다** — 이것이 이 주제의 유일하게 확실한 빈자리다. |


---

## 주제 3. 텐서 eDSL 로워링 공백의 실증적 특성화

> **판정: `viable` → 적대적 심사 후 `weak`**  (약함 — 독립 논문보다 한 절이 낫다)
> 조사자가 찾은 선행연구 10건 · 심사자가 제기한 치명적 반론 12건 ·
> "이미 출판됨" 지적 10건 · 근거 없이 단정한 문장 13건

### 3.1 초록 — 한 문장 주장

텐서 eDSL 의 "타입체크 통과 = 표현 가능" 계약과 백엔드의 "실기 바이너리로 낮출 수 있음" 계약은 서로 독립이며, 상용 NPU 벤더 예제 커널 200개 전수 컴파일·실기 실행으로 그 간극을 정량화하면 (a) 프론트엔드가 유효하다고 선언한 것의 상당수가 낮춰지지 않고 (b) 무효하다고 선언한 것의 40%가 문제없이 낮춰지며 (c) 낮춰진 것조차 로드·실행에서 조용히 틀리는, 세 겹의 계약 불일치가 드러난다.

### 3.2 문제의식과 선행연구의 빈틈

검색 8회로 확인한 범위에서, 선행연구는 세 갈래로 나뉘고 어느 쪽도 "표현은 되는데 낮출 수 없는 영역"을 대상으로 삼지 않는다. ① 컴파일러 결함 실증(FSE'21 Shen 603건, ISSRE'21 Du 2,717건)은 **이슈 트래커 사후 마이닝**이라 "이 컴파일러가 지금 무엇을 못 낮추는가"라는 정적 커버리지 질문을 던지지 않는다. ② 퍼징(Tzer OOPSLA'22, NNSmith ASPLOS'23, MLIR lowering-space ASE'25, DESIL'25)은 **새 버그를 찾는 것이 목표**이고, 컴파일러가 "not yet implemented"로 정직하게 거부하는 케이스는 오히려 노이즈로 버린다. ③ 가속기 필드 스터디(Mind the Gap 2511.11601, Ascend 2607.08215)는 **연산자/프레임워크 층**에서 불일치를 보고하고 eDSL 로워링 층은 다루지 않는다. 특히 "의도적 음성 표본(invalid_*)이 섞인 벤더 테스트 스위트에서 진짜 로워링 공백을 분리해내는 문제" 자체를 다룬 연구는 찾지 못했다 — 그런데 이건 벤더 예제를 실증 코퍼스로 쓰려는 모든 후속 연구가 반드시 부딪히는 문제다.

### 3.3 제안한 기여

- **프론트엔드-백엔드 계약 불일치의 정량화**(가장 날카로운 결과, 신규성 높음): 매핑 단언 커널 71개 중 `invalid_*`/`*_mismatch` 로 명명된 42개는 25 FAIL / **17 OK** — 즉 '잘못됐다고 선언한' 커널의 40%를 백엔드가 아무 불평 없이 낮춘다. 반대로 `valid_*` 29개 중 3개(`lane_size::valid_size_{1,2,4}`)는 낮춰지지 않는다. 타입상태·verify 단계의 유효성 계약과 로워링 계약이 71개 중 20개(28%)에서 갈린다. 이름·타입만 보고 판단하면 틀린다는 것을 수치로 보인다.
- **로워링 공백의 3단계 층화 특성화**: 컴파일 200개(137/63) → 실기 로드 → 실기 실행의 3개 관문을 같은 커널 집합에 대해 전수 통과시켜, '컴파일 성공 ≠ 로드 성공 ≠ 값 정확'을 단일 코퍼스에서 동시에 측정. `chip_shuffle` 은 컴파일 OK 후 로드 abort(56,576 vs 37,888), `reshape` ×2 는 50,560 vs 33,792 — **세 건 모두 요구/실제 비율 1.49~1.50** 로 단일 크기계산 결함의 서명을 이룬다.
- **실기 테스트 결과를 신뢰 불가로 만드는 측정 함정 2종의 발견과 정량화**(방법론 기여, 다른 가속기로 이전 가능): ① 프로세스 내 연쇄 오염 — hang 커널 1개(`ve_stash_fp_fp`, HAL -110/ETIMEDOUT)가 같은 프로세스 후속 커널을 전부 오염시켜 통과 수가 10 → 33(**3.3배**)으로 달라진다. `--test-threads=1` 로도 안 잡힌다. ② 필터 부분문자열 오염 — `compile <FILTER>` 가 접두사 일치라 남의 에러를 뒤집어쓴다(충돌 후보 8, 실오염 1건, 136/64 → **137/63** 보정).
- **비결합 연산 아티팩트와 진짜 데이터 오염을 가르는 판별 프로토콜**: 값 불일치 4건 중 2건은 결함이 아니었다. 1 ULP(512개 중 489개 비트 일치, 최대 상대오차 1.742e-07)와 `saturating_add` 리듀스 순서(387/512 불일치, 최대오차 2.7e9 — 겉보기엔 완전 붕괴)를 **포화 불가능한 입력(-50..50)으로 재실행하는 결정적 실험**으로 분리. 이 프로토콜이 없으면 'i32 리듀스가 실기에서 깨진다'는 틀린 결론을 낸다.
- **상용 NPU 런타임의 조용한 정합성 붕괴 2건, 기전까지 규명**: 브로드캐스트 DMA 가 목적지에 아무것도 쓰지 않아 이전 f32 잔류물을 되읽음(2048/2048 불일치, 2회 반복 실행에서 앞 3값 완전 동일 = 결정적), 커밋 창 오프셋의 요소-vs-바이트 단위 혼동(의도 32 elem = 128 B → 실착지 8 elem = 32 B, 정확히 ÷sizeof(f32)). 둘 다 **에뮬레이션에서는 통과**하고 크래시·경고가 없다.

### 3.4 방법

단일 벤더 스택(FuriosaAI furiosa-opt 0.4 / RNGD)의 예제 크레이트를 코퍼스로 삼아 3단계 관문 전수 통과. (1) **정적 관문** — `#[device]` 커널 200개를 `cargo furiosa-opt compile <FILTER>` 로 개별 호출(크레이트 단위 컴파일은 첫 63개 에러에서 죽으므로 커널 단위 격리가 필수), 에러 줄이 해당 커널을 정확히 지목할 때만 FAIL 판정해 접두사 오염 제거. (2) **분류** — 63(보정 전 64)개 실패를 REAL_LOWERING_GAP / INTENTIONAL_NEGATIVE / COMPILER_ICE / GENERIC_NOT_MONOMORPHIZED / UNCLEAR 로 라벨링. 핵심은 의도적 음성 표본의 분리이고, 명명 규칙이 아니라 **호출 테스트 존재 여부 + 에러 발생 단계(visa/mir/lir)** 로 판정. (3) **동적 관문** — FAIL 커널에만 `#[cfg(not(backend="npu"))]` 게이트를 자동 삽입(상류가 이미 쓰는 관용구)해 크레이트 전체를 실기 빌드(143 커널), NPU 테스트 89개를 **테스트당 프로세스 1개 + 타임아웃**으로 격리 실행. 값 불일치는 ULP 분포·순서 독립성 프로브로 자동 분류. 대조군은 같은 소스의 typecheck·emulation 백엔드(로워링 에러 0건 / 104 pass 0 fail)로, 세 백엔드의 차이가 곧 '표현 가능성 - 로워링 가능성 - 실행 가능성' 의 차분이 된다.

### 3.5 평가 설계

비교 대상은 다른 컴파일러가 아니라 **같은 소스에 대한 3개 백엔드 자신**이다. 지표: (a) 로워링 수율 = 낮춰진 커널 / 표현 가능한 커널 (137/200 = 68.5%), 대조 typecheck 200/200; (b) 실행 수율 = 실기 통과 / NPU 백엔드 테스트 (80/89 = 89.9%, 결함 보정 시 83/89), 대조 emulation 104/104; (c) 계약 불일치율 = 프론트 유효성 선언과 백엔드 판정이 갈리는 비율 (20/71 = 28%); (d) 측정 신뢰도 = 격리 실행 대비 비격리 실행의 실패 과대계상 배율 (3.3×); (e) 결함 유형별 검출 난이도 = abort(즉사) 3 / silent(무증상) 2 / hang 1. 베이스라인 부재가 최대 약점이므로, 최소한 **동일 프로토콜을 두 번째 컴파일러 스택**(furiosa 자체 TCL 경로, 또는 IREE/TVM-BYOC/Triton 비-NVIDIA 백엔드)에 적용해 수율·유형 분포가 재현되는지 보여야 한다. 로워링 공백 카탈로그의 유용성은 "게이팅 파이프라인이 실기 실행 가능 테스트를 21 → 89개로 넓혔다"로 직접 평가 가능하다.

### 3.6 이미 확보된 근거

- 커널 단위 컴파일 매트릭스 200행 전수 확인: **OK 137 / FAIL 63** (직접 grep -c 로 카운트 검증). 출처 `_evidence/logs/perkernel_matrix_fixed.txt`
- 실기 격리 실행 매트릭스 89행 전수 확인: **PASS 80 / FAIL 5 / ABORT 3 / OTHER(=#[ignore]) 1**. 플래그 집계도 일치 — hal=1 이 1건(`test_ve_stash_fp_fp`), load=1 이 3건(reshape ×2, chip_shuffle), mismatch=1 이 4건(broadcast, tile_window_commit, reduce_split_time_packet, ternary_selective). 출처 `_evidence/logs/npu_matrix.tsv`
- **내가 직접 세어 확인한 신규 수치(문서에는 §4 각주로만 언급됨)**: `invalid_*`/`*_mismatch` 명명 커널 42개 중 FAIL 25 / **OK 17**; `::valid_*` 명명 커널 29개 중 OK 26 / **FAIL 3**(전부 `contract_outer_assertions::lane_size::valid_size_{1,2,4}`). 즉 71개 단언 커널 중 20개(28%)에서 명명·프론트엔드 계약과 백엔드 판정이 어긋난다
- 로드 실패 3건의 크기 비율 서명: 50,560/33,792 = 1.496 (reshape ×2), 56,576/37,888 = 1.493 (chip_shuffle). 셋 다 `device-runtime-c/src/kernel.rs:137`. 출처 문서 13 §2.1 + npu_matrix.tsv 의 load=1 행
- 프로세스 격리 효과: vector_engine 을 한 프로세스로 돌리면 10 pass / 25 fail, 테스트마다 새 프로세스면 **33 pass / 3 fail**(3.3배). 단일 스레드로 직렬화해도 해결 안 됨. 출처 문서 12 §10.5, 13 §3, `_evidence/logs/ve_isolated.log`(격리 실행 원본, FAIL 은 ulp=1 2건뿐)
- 조용한 오배치 2건의 기전: ① broadcast DMA 목적지 미기록 — 2048/2048 불일치, 2회 실행에서 앞 3값 완전 동일(결정적), f32 재해석 시 -0.283/0.251/-0.542 로 이전 잔류물임이 드러남. ② 커밋 창 오프셋 단위 혼동 — 데이터 `0..31` 이 순서 온전한 채 오프셋 32(=128 B) 대신 8(=32 B)에 착지, 정확히 ÷4 = sizeof(f32). 둘 다 에뮬레이션 통과. 출처 문서 13 §2.2
- 비결합 리듀스 판별 실험: `saturating_add` R16 리듀스가 랜덤 i32 입력에서 387/512 불일치·최대오차 2.7e9 였으나, 포화 불가능한 입력(-50..50)으로 바꾸면 실기에서 0.12초 통과. → 데이터 오염이 아니라 하드웨어 병렬 리듀스 순서 차이. 출처 문서 13 §2.4(b)
- 1 ULP 사례 정량: 512 원소 중 489(95.5%) 비트 일치, ULP 분포 {0:489, 1:23}, 최대 상대오차 1.742e-07, 최악 원소 npu=4.0194473 vs host=4.019448. 테스트가 `assert_f32_vec_eq` 로 비트 일치를 요구해 실패. 출처 문서 12 §8.1 / 13 §2.4(a)
- 3개 백엔드 대조: 같은 크레이트가 typecheck 는 커널 낮추기 에러 0건, emulation 은 **104 passed / 0 failed / 10 ignored**, npu 는 **63개 에러로 빌드 사망 → 테스트 0개 실행**. 원인은 npu 백엔드가 호출 여부와 무관하게 패키지 내 모든 `#[device]` 를 AOT 로 낮추기 때문. 출처 문서 12 §4
- 실패 분류 24 REAL_GAP / 23 INTENTIONAL / 13 ICE / 2 GENERIC / 2 UNCLEAR = 64 (보정 전 목록 기준, 137/63 보정 후 재분류는 안 됨). 출처 문서 13 §7.1 — **단, 문서 §8 이 스스로 '에이전트 조사 + 적대적 재검증이며 개별 커널 사유를 전수 수작업 재확인하지 않았다'고 명시**
- ICE 13건이 **세 IR 단계에 걸쳐 있음**을 에러 문자열로 확인: `visa: internal compiler error`(09장), `mir: internal compiler error: T366 is used before definition` / `no entry found for key` / `not yet implemented`(08장, transformer 4단계), `lir: internal compiler error: entered unreachable code`(04장, i4_contract). 단계별 건수 표는 없음
- 필터 부분문자열 오염 보정: 접두사 충돌 후보 8/200, 실제 오염 판정 1건(`inter_transpose::invalid_time0`), 136/64 → **137/63**. 출처 문서 13 §7.4 + 매트릭스 파일 자체가 보정본

### 3.7 아직 없는, 반드시 해야 할 실험

- **두 번째 컴파일러 스택에 동일 프로토콜 적용**(가장 중요, 이거 없으면 weak 로 떨어진다). 후보: 같은 하드웨어의 furiosa TCL/furiosa-llm 경로(같은 칩·다른 컴파일러 = 가장 깨끗한 대조), IREE 또는 TVM-BYOC 의 비-NVIDIA 백엔드, Triton on AMD/Intel NPU. 로워링 수율·실패 유형 분포·계약 불일치율이 재현되는지가 논문의 생사를 가른다
- **벤더 테스트 스위트가 아닌 코퍼스 확보**. 현재 200개는 의도적 음성 표본 23개를 포함한 '단언 스위트'라 수율 68.5% 는 현실 워크로드에 대한 커버리지가 아니다. 독립 작성한 실제 커널 집합(attention 변형, MLP, norm, conv, KV-cache gather 등 50~100개)을 같은 파이프라인에 통과시켜 편향 없는 수율을 재측정해야 한다
- **63개 실패 분류의 인간 2인 이상 독립 재검증 + 일치도(Cohen's κ) 보고**. 현재 라벨은 LLM 에이전트 조사 결과이고 문서가 스스로 한계로 적었다. 실증 연구 트랙 리뷰어가 가장 먼저 치는 지점이다. 137/63 보정 후 재분류도 함께
- **보정 후 63개 기준 재분류 + 단계별(visa/mir/lir) ICE 건수 표 작성**. 지금은 ICE 13건이 세 단계에 흩어져 있다는 것만 일화적으로 확인됨. 단계별 분포는 '어느 IR 층이 가장 미성숙한가'라는 핵심 질문에 답한다
- **하드웨어 성능 카운터로 DMA 지배 주장 재확인**. 현재 96.5% DmaEngine / 커널별 중앙값 82.8% 는 **컴파일러의 정적 스케줄 모델**이지 실측이 아니다(문서 §8 이 인정). mnist::forward 17,953 cycle 이 독립 기록과 일치한다는 교차검증은 있으나 모델 1점 검증일 뿐. 벽시계도 cargo 오버헤드 3.5초에 묻혀 분리 못 했다
- **상류(FuriosaAI) 이슈 제기 및 벤더 확인 획득**. 로더 범위초과, 브로드캐스트 DMA 미기록, 커밋 오프셋 단위 혼동, 커널 hang, ICE 13건, 낡은 #[ignore] 3건 — 아직 하나도 보고 안 함. 벤더가 confirm/fix 한 건수는 실증 결함 연구의 표준 외적 타당성 지표이고, 지금 이게 0 이다
- **로워링 공백의 회피 가능성 분석**: REAL_GAP 24개 중 소스를 다시 써서 낮출 수 있는 것이 몇 개인가. '컴파일러가 고쳐야 할 근본 공백' vs '사용자가 우회 가능한 표현 선택'을 가르면 공백 카탈로그의 실용 가치가 생긴다. 지금은 '못 간다'까지만 확인
- **SDK 버전 간 드리프트 측정**: 같은 매트릭스를 2개 이상 SDK 버전에서 돌려 공백이 줄고 있는지·새로 생기는지 측정. 스냅샷 1회 측정이라는 비판에 대한 유일한 방어
- **연쇄 오염 3.3배 효과가 furiosa 특이 현상이 아님을 입증**: 다른 가속기 런타임(Ascend/XDNA/TPU)에서도 hang 커널 1개가 프로세스 내 후속 실행을 오염시키는지 확인. 되면 '가속기 테스트 하네스 설계 규칙'이라는 일반 주장이 서고, 안 되면 벤더 버그 1건으로 격하
- **ULP/순서 판별기를 일반 오라클로 승격 + 오탐/미탐률 측정**: 현재 `classify_mismatch.py` 는 4건에만 적용됐다. 알려진 결함/알려진 아티팩트가 섞인 합성 세트로 precision/recall 을 재야 도구로서 주장 가능
- **게이팅 파이프라인의 도구화 및 아티팩트 배포**: 커널 매트릭스 → 자동 게이트 삽입 → 격리 실행까지 재사용 가능한 도구로 패키징. CGO/ISPASS 급은 아티팩트 뱃지가 사실상 필수

### 3.8 ★ 심사 반론

**치명적 반론 (reject 사유)**

- 【논문의 척추가 부러진다 — 1번 기여의 전제가 증거와 정면으로 모순】 thesis 는 '타입체크 통과 = 표현 가능'이라는 프론트엔드 계약을 백엔드 계약과 대조한다고 주장하지만, 12-예제-전수실행.md §5.2-5.3 이 스스로 이렇게 적었다: 'typecheck·emulation 은 커널을 호출될 때 처리한다 → 아무도 안 부르는 invalid_* 는 그냥 지나간다 → 빌드 성공', 그리고 'grep -rn "invalid_|_mismatch" tests/ → 0건'. 즉 **typecheck 백엔드는 42개 invalid_* 커널을 애초에 쳐다본 적이 없다.** 프론트엔드는 그것들을 유효하다고도 무효하다고도 선언하지 않았다. 단지 도달하지 않았을 뿐이다. 따라서 '프론트엔드 유효성 계약 vs 백엔드 로워링 계약이 28% 갈린다'는 두 계약의 비교가 아니라, **벤더 테스트 작성자가 Rust 식별자에 붙인 이름 문자열 vs 컴파일러**의 비교다. 리뷰어 한 명만 이 두 줄을 읽으면 가장 날카로운 기여가 통째로 증발한다
- 【자기 방법론과 자기 헤드라인 수치가 서로를 부정한다】 method 는 '명명 규칙이 아니라 호출 테스트 존재 여부 + 에러 발생 단계(visa/mir/lir)로 판정'한다고 명시하고, 13-NPU-실기-매트릭스.md §7.1 도 '이름만 보고 판단하면 틀린다'고 경고한다. 그런데 contributions[0] 의 헤드라인 42개 중 17 OK / 25 FAIL 은 **순수하게 이름 문자열 grep 으로만 얻은 수치**다(내가 재현 확인: grep -E "invalid_|_mismatch" → 25 FAIL / 17 OK). 논문이 스스로 '틀린 방법'이라고 선언한 방법으로 최고 기여를 만들었다. 이건 major revision 이 아니라 reject 사유다
- 【'전수(exhaustive)'가 전수가 아니다 — 분모가 확정되지 않았다】 12-예제-전수실행.md §2 는 커널 수가 세 가지라고 적는다: 소스 추출 200, **툴 자체 집계 207**('Finished 1 compiled, 206 filtered out'), 서브에이전트 독립 집계 208. 문서는 §2·§4.1·§10 에서 **207 을 채택한다**고 선언해 놓고 매트릭스는 200행이다. 7개 커널이 측정에서 빠졌고 그 상태는 미지다. 만약 7개가 전부 FAIL 이면 로워링 수율은 68.5% 가 아니라 66.2% 다. 논문 전면에 세울 숫자의 분모가 ±3.5% 흔들린다는 것을 리뷰어가 보면 '전수'라는 단어 자체를 못 쓰게 한다
- 【n=1 — 컴파일러 1개, 칩 1종, SDK 1버전, 벤더 1곳】 200개·89개라는 표본 크기는 착시다. 독립 관측 단위는 컴파일러 **1개**다. '이 벤더 컴파일러가 미성숙하다'와 '텐서 eDSL 로워링이 일반적으로 이런 성질을 갖는다'를 구분할 수단이 전혀 없다. 게다가 실패의 상당수가 `Branch conversion is not yet implemented`, `not yet implemented` 류 — 즉 컴파일러가 **정직하게 미구현을 알린 것**이다. 다음 릴리스에서 사라지면 논문의 측정 대상 자체가 증발한다. 버전 드리프트 측정이 0회이므로 '2026년 7월 furiosa-opt 0.4 릴리스 노트'라는 조롱을 방어할 수 없다
- 【베이스라인이 없을 뿐 아니라, 있다고 주장하는 대조군 3개가 전부 가짜다】 evaluation 은 '같은 소스에 대한 3개 백엔드 자신'을 대조군이라 하지만: (a) typecheck 200/200 은 **아무도 측정한 적 없는 숫자**다 — 문서에 있는 건 '크레이트 빌드 성공'뿐이고 커널 단위 typecheck 매트릭스는 존재하지 않으며, 위 1번 이유로 미호출 커널에는 공허하다. (b) emulation '104/104' 는 실제로는 104 passed / 0 failed / **10 ignored** 이고, §6 이 그 10개 중 3개는 그냥 통과한다고 밝혔다. 정직하게 쓰면 104/114 다. (c) npu 대조는 '같은 소스'가 아니라 **저자가 직접 게이트를 삽입해 만든 크레이트**(143 커널)에서 나온 89개다. 분모가 방법에 내생적이다. 결국 진짜 대조군은 0개다
- 【기여의 절반이 벤더 SW 버그 리포트다 — 학술 델타가 아니다】 로더 범위초과 abort 3건, 브로드캐스트 DMA 미기록, 커밋 창 오프셋 요소-vs-바이트 혼동, 커널 hang(HAL -110). 디버깅으로는 훌륭하고 증거도 탄탄하지만(1.496/1.493 비율, 32→8=÷4, 2회 실행 앞 3값 동일 — 전부 파일에서 확인됨) 학술 기여로는 일화 4건이다. 게다가 13번 §8 이 '상류에 보고하지 않았다'고 적었다 — 벤더 confirm/fix 0건. 실증 결함 연구에서 외적 타당성 지표가 정확히 0 이다
- 【DMA 96.5% 지배 주장은 실측이 아니라 컴파일러 자신의 예측이다】 13-NPU-실기-매트릭스.md §8 이 직접 인정한다: '§4의 사이클은 정적 예측이다. 실제 하드웨어 카운터를 읽은 게 아니다', '실기 벽시계는 cargo 오버헤드(최소 3.5초)에 묻혀 커널 시간을 분리하지 못했다'. mnist::forward 17,953 cycle 교차검증은 **모델 1점 검증**이라 모델의 편향을 잡지 못한다. 컴파일러의 스케줄 모델로 그 컴파일러가 낳은 코드를 평가하는 것은 순환이다. 이 상태로 ISPASS/IISWC 에 '하드웨어 특성화'를 주장하면 방법론 섹션에서 desk-reject 위험이 있다
- 【분류 라벨이 논문의 절반을 지탱하는데 신뢰도가 측정되지 않았다】 24 REAL_GAP / 23 INTENTIONAL / 13 ICE / 2 GENERIC / 2 UNCLEAR = **64 ≠ 63**. 문서가 스스로 '재분류는 하지 않았다', '§7의 분류는 에이전트 조사 + 적대적 재검증 결과이며 개별 커널 사유를 전수 수작업 재확인하지 않았다'고 적었다. 인간 재검증 0명, κ 미보고. 더 나쁜 건 라벨이 서로 배타적이지 않다는 정황이다 — `switch_assertions::inter_transpose::invalid_time0_mismatch` 는 이름상 INTENTIONAL_NEGATIVE 후보인데 09-도구와-계층.md:154 에서 `visa: internal compiler error` 를 내므로 COMPILER_ICE 이기도 하다. 분할이 partition 이 아니면 24/23/13 표는 무의미하다
- 【코퍼스 편향이 헤드라인 숫자를 무효화한다】 68.5% 는 '실제 워크로드의 68.5%'가 아니라 '벤더 단언 스위트의 68.5%'다. 이 스위트는 (a) 일부러 틀리게 만든 표본 23개를 포함하고 (b) 나머지도 매핑 규칙 커버리지용 마이크로 커널이며 (c) 아예 호출되지 않는 커널이 42개 이상이다. 실모델 코드의 분포와의 관계가 미지다. risks 에서 저자도 인정하지만, 인정한다고 리뷰어가 봐주지 않는다 — 그냥 '그럼 왜 이 숫자를 Abstract 에 넣었나'가 된다
- 【'GPU/NPU 에서 이미 한 얘기' 반론에 답이 없다】 아래 already_published 참조. 컴파일 성공률과 실기 기능 정확도를 함께 재는 것(AscendCraft 98.1% vs 90.4%, NPUKernelBench, CANN Bench, NPUEval), '컴파일러는 통과하는데 런타임 크래시'(Hawk 2607.01590), 프론트엔드 수용 vs 백엔드 미구현으로 인한 false rejection 분류(Solidity 2512.18182), silent bug(DESIL), 로워링 경로 의존 실패(ASE'25), 가속기 간 출력 불일치(Mind the Gap), 단일 벤더 NPU 필드 스터디(Ascend 2607.08215) — 개별 관찰이 전부 선점됐다. 남은 것은 '조합'인데, 조합만으로 CGO/ISPASS 를 통과시키려면 그 조합이 **새 결론**을 낳아야 한다. 지금 조합이 낳은 유일한 새 결론이 28% 계약 불일치인데 그게 1번 반론으로 무너진다
- 【3.3배 격리 효과 수치가 like-for-like 가 아니다】 ve_isolated.log 를 직접 세었다: `ok. 1 passed` 32줄 + `0 passed; 0 failed; 1 ignored` 1줄 + `FAILED` 3줄 = 36개. 문서의 '33 pass' 는 **#[ignore] 된 test_ve_elementwise_vrf 를 pass 로 계산**한 것이다. 반면 비격리 쪽은 10 pass / 25 fail = 35개. 분모가 35 vs 36 으로 다르고 분자에 ignored 가 섞였다. 진짜 배율은 32/10 = 3.2× 다. 방법론 기여를 주장하는 논문이 방법론 기여의 헤드라인 숫자를 이렇게 세면 안 된다
- 【학회 적합성 리스크가 구조적이다】 성능 개선 0, 새 기법 0, 도구 릴리스 0(현재), 벤더 confirm 0. ISCA/MICRO/ASPLOS/PLDI/CGO 는 이 상태로는 desk-reject. ISPASS/IISWC 는 특성화를 받지만 '정적 모델을 하드웨어 특성으로 제시'를 싫어한다. ISSTA/ASE 실증 트랙은 인간 재검증+κ 없는 LLM 라벨링을 안 받는다. 즉 **현재 형태에 맞는 학회가 사실상 워크숍뿐**이다

**이미 출판되어 신규성이 없는 부분**

- Hawk: Harnessing Hardware-Aware Knowledge for High-Performance NPU Kernel Generation — arXiv 2607.01590 (2026-07). 명시적으로 '유사 NPU 커널에서 코드를 옮겨오면 **컴파일러는 통과하지만 일관되게 런타임 크래시와 성능 저하를 유발한다**'고 보고. 이 제안의 핵심 프레이밍인 '컴파일 성공 ≠ 실행 성공(상용 NPU)'이 이미 published 문장으로 존재한다
- AscendCraft: Automatic Ascend NPU Kernel Generation via DSL-Guided Transcompilation — arXiv 2601.22760 (2026-01). **컴파일 성공률 98.1% vs 기능 정확도 90.4%** 를 나란히 보고. 즉 '컴파일 관문/실행 관문 2단계 수율'이라는 측정 축이 상용 NPU(Ascend)에서 이미 정량화됐다. 이 제안의 관문 (1)-(3) 중 (1),(3) 이 선점됨
- AscendKernelGen: A Systematic Study of LLM-Based Kernel Generation for Neural Processing Units + NPUKernelBench — arXiv 2601.07160 (2026-01). NPU 커널에 대해 **컴파일·정확성·성능을 난이도 계층별로 평가하는 벤치마크**를 제시하고 실기에서 참조 구현과 출력을 비교. 'compilation rate(CR) + correctness(Acc)' 라는 이중 지표가 이미 프레임워크로 릴리스됨. 이 제안의 '3단계 층화 특성화' 기여가 상당 부분 겹친다
- CANN Bench: Benchmarking Agent Generated Kernels against Real NPU and Algorithmic Limits — arXiv 2607.20518 (2026-07). 실기 NPU 대상 커널 벤치마크. 제안이 need_experiments 에서 요구하는 '두 번째 스택'의 후보이자, 동시에 '실기에서 커널 집합을 전수 돌려 컴파일/정확도를 잰다'는 아이디어의 선행
- Understanding Typing-Related Bugs in Solidity Compiler — arXiv 2512.18182 (2025-12). **false rejection = 컴파일러가 well-typed 이고 의미적으로 유효한 프로그램을 잘못 거부하는 현상**을 원인별로 분류하고, 그 원인 중 하나로 '**미구현 기능에 대한 false positive**'를 명시. 즉 '프론트엔드 계약 vs 백엔드 실제 수용 범위의 불일치'라는 **개념과 분류 체계 자체가 이미 published** 다(언어만 Solidity). 이 제안의 개념적 신규성 주장을 정면으로 깎는다
- Mind the Gap: Revealing Inconsistencies Across Heterogeneous AI Accelerators — arXiv 2511.11601 (2025-11). 제안이 스스로 '가장 겹침'이라 적음. 연산자 커버리지 차이 + 5% 초과 출력 불일치를 4~5종 가속기에서 대규모로 보고
- On the Limitations of Non-GPU AI Accelerators for Large-Model Inference (Huawei Ascend field study) — arXiv 2607.08215 (2026-07). 단일 벤더 NPU 에 대해 '불완전한 연산자/기능 지원, 저수준 커널의 수치 결함, 미성숙한 그래프 컴파일'을 8개 범주로 정리. 프레이밍이 거의 동일하고 3개월 먼저 나왔다
- DESIL: Detecting Silent Bugs in MLIR Compiler Infrastructure — arXiv 2504.01379 (2025). '크래시 없이 틀린 결과'라는 이 제안 §2.2 의 범주를 이미 정의·검출
- Finding Bugs in MLIR Compiler Infrastructure via Lowering Space Exploration — ASE 2025. '로워링 경로에 따라 되기도 안 되기도 한다'는 문제의식 선점
- NPUEval: Optimizing NPU Kernels with LLMs and Open Source Compilers — arXiv 2507.14403 (2025-07). AMD NPU 커널 102개에 대해 컴파일 성공률과 실기 기능 정확도를 함께 측정. 제안의 measurement axes 와 동일

**근거 없이 단정한 문장 (수정 필요)**

- thesis: '텐서 eDSL 의 타입체크 통과 = 표현 가능 계약' — 이 계약이 존재한다는 증거가 없다. 12번 문서 §5.3 은 typecheck 가 미호출 커널을 아예 처리하지 않는다고 명시한다. 표현 가능성이 200개 전부에 대해 확인된 적이 없다
- evaluation (a): '대조 typecheck 200/200' — 커널 단위 typecheck 매트릭스는 어느 파일에도 없다. 문서에 있는 것은 '크레이트 빌드 성공' 1건과 '97 passed / 7 failed / 10 ignored'(§4.1) 뿐이다. 200/200 은 측정되지 않은 숫자다
- evaluation (b): '대조 emulation 104/104' — 실제 기록은 104 passed / 0 failed / **10 ignored**(§4.2). 게다가 §6 에서 그 10개 중 3개는 개별 실행 시 통과함이 확인됐다. 104/104 는 ignored 10개를 분모에서 지운 표기다
- contributions[0]: '프론트엔드가 무효하다고 선언한 커널의 40%를 백엔드가 아무 불평 없이 낮춘다' — 42개 중 17 OK 라는 카운트 자체는 파일에서 재현됨(확인 완료). 그러나 '프론트엔드가 무효하다고 선언'했다는 부분에 근거가 없다. 선언 주체는 프론트엔드가 아니라 함수 이름을 지은 벤더 개발자다. 어느 계층도 이 42개를 검사한 적이 없다
- have_evidence[9] / method: '커널 200개 전수' — 12번 문서 §2 가 툴 자체 집계 207 을 채택한다고 선언했고 §10 도 207 로 적는다. 매트릭스는 200행. 7개 커널의 컴파일 결과가 존재하지 않는다. '전수'는 성립하지 않으며 68.5% 의 분모가 미확정이다
- have_evidence[4] / contributions[2]: '33 pass / 3 fail (3.3배)' — ve_isolated.log 실측은 32 passed + 1 ignored + 3 FAILED. ignored 를 pass 로 계산했다. 비격리 쪽 10+25=35 와 총계도 안 맞는다(35 vs 36). 정확히는 3.2배이고 두 실행의 테스트 집합이 동일한지 검증되지 않았다
- evaluation 말미: '게이팅 파이프라인이 실기 실행 가능 테스트를 21 → 89개로 넓혔다' — 12번 문서가 기록한 것은 21 → **58개+**(§10, 일괄 실행)이고, 89 는 게이팅 크레이트에서 `--list` 로 열거한 **다른 측정**(13번 §9)이다. 58 과 89 를 하나의 개선 폭으로 합치는 것은 두 실험을 섞는 것이다
- method: '에러 발생 단계(visa/mir/lir)로 판정' — 단계별 판정을 체계적으로 적용했다는 증거가 없다. 저자 스스로 need_experiments 에서 '단계별 건수 표는 없음'이라 적었다. 지금 있는 것은 04/08/09장의 일화적 에러 문자열 5~6개뿐이다(이 문자열들 자체는 확인됨)
- have_evidence[10]: '24/23/13/2/2 = 64' — 합이 63과 안 맞는다는 것을 문서가 인정. 추가로 라벨 배타성이 검증되지 않았다: invalid_time0_mismatch 는 INTENTIONAL 후보이면서 동시에 visa ICE 다(09장:154). 중복 계상 가능성이 열려 있다
- evaluation (b): '결함 보정 시 83/89' — 13번 §1.1 에 근거가 있으나, 89 라는 분모가 저자의 게이팅으로 만들어진 집합이므로 방법에 내생적이다. '실행 수율 89.9%' 를 독립 지표처럼 제시할 수 없다
- contributions[1]: '컴파일 200개(137/63) → 실기 로드 → 실기 실행의 3개 관문을 **같은 커널 집합**에 대해 전수 통과' — 같은 집합이 아니다. 컴파일 관문은 커널 200개, 실행 관문은 테스트 89개이고, 89개 테스트가 어느 커널들을 덮는지의 매핑 표가 없다. 137개 중 몇 개가 실기 실행으로 검증됐는지 알 수 없다
- risks/evaluation 전반: 'DMA 96.5% / 커널별 중앙값 82.8%' — 파일이 아니라 sched_summary.json(컴파일러 스케줄 덤프)에서 나온 값이고 13번 §8 이 정적 예측임을 인정. 하드웨어 카운터 실측 0회. 성능·병목 주장에 사용 불가
- have_evidence[6] '커밋 창 오프셋 단위 혼동' — 13번 §8 이 직접 적었다: 'tile 의 바이트 vs 요소 해석은 산술 정황이지 소스 확인이 아니다'. '기전까지 규명'(contributions[4])은 과장이다. 브로드캐스트 쪽도 '목적지에 아무것도 쓰지 않는다'는 추론이며 DMA 디스크립터를 확인한 것은 아니다

**조사자 스스로 적은 위험**

- **단일 벤더·단일 칩·단일 SDK 버전**. n=1 이다. 200개 커널·89개 테스트라는 수치가 커 보이지만 독립 표본은 컴파일러 1개다. '이 컴파일러가 미성숙한 것이지 텐서 eDSL 로워링 일반의 성질이 아니다'는 반론에 지금은 답할 수 없다. 두 번째 스택 없이는 리뷰어 다수가 여기서 리젝한다
- **코퍼스 선택 편향이 치명적**. 대상이 워크로드가 아니라 벤더의 *단언 테스트 스위트*다. 일부러 틀리게 만든 커널 23개가 섞여 있고, 나머지도 매핑 규칙 커버리지용 마이크로 커널이라 실제 모델 코드의 분포와 무관하다. 68.5% 라는 수율 숫자를 논문 전면에 세우면 '무엇의 68.5%인가'로 무너진다. 오히려 계약 불일치율(28%)이 편향에 덜 취약하다
- **'GPU 에서 이미 다 한 얘기' 반론**. 컴파일 성공 ≠ 실행 성공, 에뮬레이터와 실기의 값 차이, 순서 의존 리듀스 — 전부 GPU/DL 컴파일러 문헌에 개별적으로 존재한다. Mind the Gap(2511.11601)은 이종 가속기 간 출력 불일치를, Ascend 필드 스터디(2607.08215)는 불완전한 연산자 지원·저수준 커널의 수치 결함을 이미 보고했다. 차별점은 '**eDSL 로워링 층에서, 의도적 음성 표본을 분리해가며, 전수로**' 뿐이고 이 차별점을 명시적으로 방어하지 못하면 '기존 관찰의 새 하드웨어 재확인'으로 읽힌다
- **분류 라벨의 신뢰도**. 24/23/13/2/2 는 LLM 에이전트 조사 + 적대적 재검증 결과이고, 문서가 스스로 '개별 커널 사유를 전수 수작업 재확인하지 않았다'고 적었다. 게다가 137/63 보정 이후 재분류를 안 해 합이 64로 안 맞는다. 실증 연구 트랙에서는 이것만으로 major revision 사유다
- **결함이 아니라 미성숙을 측정했을 위험**. `Branch conversion is not yet implemented`, `not yet implemented` 류는 컴파일러가 정직하게 미구현을 알린 것이다. 이걸 '공백'이라 부르는 건 맞지만, 다음 SDK 릴리스에서 사라지면 논문의 측정 대상 자체가 증발한다. 버전 드리프트 측정이 없으면 '특정 시점 스냅샷의 릴리스 노트'라는 비판을 받는다
- **방법론 기여와 특성화 기여가 서로를 약화시킨다**. 연쇄 오염 3.3배·순서 독립성 프로브·접두사 오염 보정은 진짜 이전 가능한 방법론인데, 지금 프레이밍은 '로워링 공백 특성화'라 이것들이 부록으로 밀린다. 반대로 방법론을 전면에 세우면 사례가 1개뿐이라 약하다. **어느 쪽을 주장으로 삼을지 결정하지 않으면 둘 다 얕은 논문이 된다** — 개인적으로는 방법론(가속기 컴파일러/런타임 테스트 결과를 신뢰 가능하게 측정하는 프로토콜)을 주장으로, 로워링 공백 특성화를 그 프로토콜이 낳은 결과로 배치하는 쪽이 방어 가능성이 높다
- **성능 주장이 전무**. 아키텍처·컴파일러 주류 학회(ISCA/MICRO/ASPLOS/PLDI)는 속도 개선 없는 측정 보고를 잘 안 받는다. DMA 96.5% 도 정적 모델이라 성능 주장으로 못 쓴다. 학회 선택을 잘못하면 내용과 무관하게 데스크 리젝
- **결함 6건 중 3건이 '벤더 SW 버그 리포트'로 축소 해석될 수 있다**. 로더 범위초과·브로드캐스트 미기록·오프셋 단위 혼동은 훌륭한 디버깅이지만 학술 기여로는 일화다. 이 3건에서 **일반 원리**(에뮬레이션-실기 등가성 검사의 부재가 구조적으로 이 유형을 놓친다)를 뽑아내지 못하면 엔지니어링 노트로 읽힌다

### 3.9 살리는 길 — 무엇을 바꿔야 하는가

"결론: 지금 프레이밍으로는 살릴 수 없다. 기여 1번(계약 불일치 28%)을 버리고 논문을 다시 세워야 한다. 순서대로.\n\n【0. 즉시 폐기】 '프론트엔드 유효성 계약 vs 백엔드 로워링 계약' 프레이밍 전체. 근거가 자기 문서(12번 §5.2-5.3)에 의해 반증된다. typecheck 200/200 과 emulation 104/104 도 Abstract·Evaluation 에서 삭제하거나 각각 '크레이트 빌드 성공 1건', '104 passed / 10 ignored, 그중 3개는 개별 실행 시 통과'로 정정 표기.\n\n【1. 주장을 방법론으로 갈아끼운다(risks 의 마지막 항목이 옳다)】 제목과 주장을 '가속기 컴파일러·런타임의 테스트 결과를 신뢰 가능하게 측정하는 프로토콜'로 바꾼다. 이유: (a) 로워링 공백 특성화는 Hawk/AscendCraft/NPUKernelBench/Ascend 필드스터디에 이미 선점됐고, (b) 반면 **측정 함정 3종**은 어디에도 없다 — 프로세스 내 연쇄 오염(hang 1개 → 후속 전부 오염, --test-threads=1 로도 안 잡힘), 필터 접두사 오염(136/64 → 137/63), 비결합 연산 아티팩트 vs 진짜 오염 판별(saturating_add 를 -50..50 재실행으로 가름). 이 셋은 내가 검색으로 선행을 찾지 못했고, '이 절차 없이 낸 숫자는 3.2배 틀린다'는 것을 실측으로 보인다. 이게 유일하게 방어 가능한 델타다.\n\n【2. 측정 함정 3종을 각각 '오라클/하네스 설계 규칙'으로 승격】 각각에 대해 (i) 실패 모드의 일반 조건, (ii) 탐지 절차, (iii) 미적용 시 오차 크기를 제시. 특히 순서-독립성 프로브를 일반 오라클로 만들고, **알려진 결함 + 알려진 아티팩트를 섞은 합성 세트로 precision/recall 을 측정**해야 도구 주장이 선다(현재 4건 적용은 사례일 뿐).\n\n【3. 두 번째 스택은 선택이 아니라 필수】 다만 우선순위를 바꿔라. 같은 하드웨어의 furiosa TCL 경로는 '다른 컴파일러 같은 런타임'이라 **런타임 함정(연쇄 오염·로더)의 재현**에는 강하지만 '로워링 공백 일반성'에는 약하다. 방법론을 주장으로 삼기로 했으므로 오히려 **다른 벤더 런타임 1개**(AMD XDNA/NPUEval 하네스 또는 Ascend CANN)에서 'hang 커널 1개가 프로세스 내 후속을 오염시키는가'만 재현해도 주장이 선다. 이건 2~3주짜리 실험이고 6~9개월 계획보다 훨씬 싸다.\n\n【4. 분모를 확정하라】 200 vs 207 vs 208 을 반드시 해소. 누락된 7개를 컴파일해 매트릭스를 207행으로 채우고, 커널 열거 방법(소스 추출 규칙 vs 툴 집계)을 논문에 명시. 이것 없이 '전수'라는 단어를 쓰면 안 된다. 마찬가지로 89개 테스트와 137개 컴파일 성공 커널의 **커버리지 매핑 표**를 만들어라 — 지금은 137개 중 실행으로 검증된 것이 몇 개인지 아무도 모른다.\n\n【5. 라벨링을 구제하라】 63개 기준으로 재분류 + 인간 2인 독립 + Cohen's κ 보고 + **라벨 배타성 명시**(invalid_time0_mismatch 같은 INTENTIONAL∩ICE 케이스의 처리 규칙). 그리고 REAL_GAP 24개는 '에러 문자열 원문'을 부록에 전부 실어 독자가 재판정할 수 있게 하라. LLM 에이전트가 라벨링했다는 사실을 숨기지 말고 방법으로 기술하되, 인간 검증층을 반드시 얹어라.\n\n【6. DMA 96.5% 는 논문에서 빼거나 '컴파일러 스케줄 모델 예측'이라고 제목·캡션·본문 세 곳 모두에 못 박아라】 하드웨어 카운터를 못 읽으면 성능·병목 절을 통째로 삭제하는 편이 낫다. 정적 모델을 특성화 결과로 제시하는 순간 ISPASS/IISWC 방법론 심사에서 죽는다.\n\n【7. 벤더 confirm 을 최소 1건 확보하라】 6건(로더 범위초과, 브로드캐스트 미기록, 커밋 오프셋 단위 혼동, hang, ICE 13, 낡은 #[ignore] 3)을 지금 당장 상류에 올려라. 제출 전 confirm 1~2건이면 '엔지니어링 노트' 반론의 절반이 막힌다. 이건 비용이 거의 0이고 효과가 가장 크다.\n\n【8. 현실적 경로】 위 0·2·4·7 만 하면(약 1.5~2개월) LCPC/C4ML 급 워크숍에 '가속기 컴파일러 테스트의 3가지 측정 함정과 보정 프로토콜'로 즉시 낼 수 있다. 3·5 를 더하면(+3개월) ISPASS/CC 도전 가능. **로워링 공백 카탈로그를 주장으로 삼는 버전은 어떤 조합으로도 살아나지 않는다** — 2026년 상반기에 같은 관찰이 최소 4편(2601.07160, 2601.22760, 2607.01590, 2607.08215) 나왔기 때문이다. 마지막으로: 검색은 영어 키워드 기반이고 2026-05~07 프리프린트가 폭증 중이므로 제출 직전 재검색이 필수다."

### 3.10 후보 학회와 소요 기간

- ISPASS (IEEE Int'l Symp. on Performance Analysis of Systems and Software) — 특성화 논문이 본령이고 성능 개선을 요구하지 않는다. 현재 형태에 가장 가까운 1순위
- IISWC (IEEE Int'l Symp. on Workload Characterization) — 마찬가지로 측정·특성화 트랙. 코퍼스 편향만 정리하면 적합
- CC (Int'l Conf. on Compiler Construction, ETAPS) — 컴파일러 로워링 계약 불일치라는 프레이밍이면 정확히 맞고 규모 요구가 CGO/PLDI 보다 완만하다
- CGO — 두 번째 스택 + 도구화 아티팩트가 붙으면 도전 가능. 아티팩트 평가 뱃지가 사실상 필수
- ISSTA / ASE / ICSE (SEIP·실증 트랙) — 결함 분류·오라클 설계·측정 프로토콜을 주장으로 세울 경우. 인간 재검증 + κ 보고가 전제
- MLSys — eDSL·컴파일러 신뢰성 각도로 가능하나 성능 없는 특성화는 경쟁력 낮음
- LCPC 또는 C4ML/CTSTA 류 워크숍 — 현재 근거만으로 즉시 낼 수 있는 현실적 최소 경로. 여기서 피드백 받고 본학회로 키우는 순서를 권장

**소요 기간 추정**: 6~9개월. 내역: 두 번째 컴파일러 스택 전수 매트릭스 2~3개월(같은 하드웨어의 TCL 경로면 1.5개월, IREE/TVM-BYOC 면 3개월) / 비-벤더 코퍼스 50~100 커널 작성 및 3관문 통과 1.5개월 / 63건 분류의 인간 2인 독립 재검증 + κ 0.5개월 / 상류 이슈 6건 제기 및 벤더 응답 대기 1~3개월(병렬 진행 가능, 논문 제출 전 confirm 1건 이상 목표) / 게이팅·격리·판별 파이프라인 도구화 + 아티팩트 1개월 / 집필·수정 1.5개월. 단, 워크숍 제출용으로만 정리한다면 **1.5개월**로 지금 근거만으로도 가능하다.

**신규성 확신도**: **중간 정도. 프레이밍은 미점유이나 구성 요소는 각각 거의 다 발표되어 있다.** 검색 8회(DL 컴파일러 버그 실증, 텐서 컴파일러 로워링/DSL 표현력, NPU 컴파일러 신뢰성, 퍼징, 에뮬레이터-실기 차분 테스트, MLIR 미구현 로워링, NPU 커널 벤치마크, 벤더 특정 검색) 결과, "**텐서 eDSL 이 표현은 되지만 낮출 수 없는 영역을 체계적으로 특성화한 연구**"에 정확히 대응하는 published 논문은 찾지 못했다. 특히 '의도적 음성 표본이 섞인 벤더 단언 스위트에서 진짜 로워링 공백을 분리한다'는 문제 정의는 어디에도 없었고, 프론트엔드 유효성 선언과 백엔드 로워링 판정이 71개 중 20개(28%)에서 갈린다는 정량은 내가 로그에서 직접 세어 확인한 신규 수치다. 그러나 낙관은 금물이다: (1) Mind the Gap(2511.11601)이 이종 가속기 간 연산자 커버리지 차이 + 출력 불일치를 이미 대규모로 보고했고, (2) Ascend 필드 스터디(2607.08215, 2026-07)가 단일 벤더 NPU 의 '불완전한 연산자 지원 + 미성숙한 그래프 컴파일 + 저수준 커널 수치 결함'을 8개 범주로 정리해 우리 프레이밍과 정신적으로 매우 가깝다. (3) DESIL 은 silent bug 를, ASE'25 lowering-space 논문은 로워링 경로 의존 실패를 각각 선점했다. 즉 **개별 관찰은 거의 다 이웃이 있고, 남은 신규성은 "eDSL 로워링 층에서, 계약 불일치를 축으로, 전수로" 라는 조합뿐**이다. 이 조합이 리뷰어에게 충분한 델타로 읽히려면 두 번째 스택에서 같은 계약 불일치가 재현되어야 한다. 또 한 가지 유의: 검색은 영어 키워드 기반이고 최근 3개월(2026-05~07) arXiv 프리프린트가 빠르게 늘고 있어, 제출 직전 재검색이 필수다.

### 3.11 검색으로 확인한 관련 연구

| 제목 | 학회·연도 | 이 주제와의 관계 |
|---|---|---|
| A Comprehensive Study of Deep Learning Compiler Bugs | ESEC/FSE 2021 (Shen, Ma, Chen, Tian, Cheung, Chen) | 기반 — DL 컴파일러 버그를 root cause/symptom/**stage** 로 분류한 표준 참조(TVM 318 + Glow 145 + nGraph 140 = 603건). 우리도 단계(visa/mir/lir)별 분류를 쓰므로 분류 체계를 그대로 상속·확장할 수 있다. 결정적 차이: 이 논문은 **이슈 트래커 사후 마이닝**이고 우리는 **컴파일러를 직접 전수 구동한 현재 시점 커버리지 측정**이다. 또 이들은 '고쳐진 버그'만 보므로 '아직 미구현이라 정직하게 거부되는 영역'은 표본에 없다 |
| An Empirical Study on Common Bugs in Deep Learning Compilers | ISSRE 2021 (Du et al.) | 인접 — TVM/Glow/nGraph/PlaidML/TC 5개 컴파일러의 버그 리포트 2,717건 대규모 분석(환경·호환성·메모리·문서·의미론 5대 원인). 규모 면에서 우리를 압도하지만 역시 리포트 기반이고 실기 실행 검증이 없다. 우리 논문의 '왜 리포트 마이닝으로는 로워링 공백이 안 보이는가' 논거의 상대 |
| Coverage-Guided Tensor Compiler Fuzzing with Joint IR-Pass Mutation (Tzer) | OOPSLA 2022 (PACMPL) | 기반 — TVM 저수준 IR 을 커버리지 유도로 변이해 미지 버그 49건 검출. 우리와 목적이 반대다: 퍼저는 새 입력을 합성해 크래시를 찾고, 우리는 **주어진 실제 커널 집합에 대해 무엇이 낮춰지는가**를 전수로 잰다. 퍼징이 도달 못 하는 것(의도적 음성 표본 구분, 실기 로드/실행 관문)이 우리 영역 |
| NNSmith: Generating Diverse and Valid Test Cases for Deep Learning Compilers | ASPLOS 2023 | 기반 — 그래프 수준 생성 퍼징, Tzer 대비 고유 커버리지 123배. '취약한 타입 시스템, 특정 데이터 레이아웃 지원 미흡'을 결함 유형으로 지목했는데 이는 우리가 실측한 정렬·서브바이트(i4)·부분 충전 레인 그룹 실패와 **정확히 같은 계열**이다. 우리 결과를 이 언어로 해석하면 자연스럽게 연결된다 |
| Mind the Gap: Revealing Inconsistencies Across Heterogeneous AI Accelerators | arXiv 2511.11601, 2025-11 | **가장 겹침** — 4,000개 실모델에서 10만 변종을 합성해 엔터프라이즈급 가속기 4~5종에 돌려 **연산자 커버리지 차이와 5% 초과 출력 불일치**를 보고. 원인으로 연산자 구현 차이·예외값 처리·명령 스케줄링을 든다. 우리의 '에뮬레이션 통과 vs 실기 불일치'와 주제가 직결. 남는 차별점: (a) 이들은 프레임워크/연산자 층, 우리는 eDSL 로워링 층, (b) 이들은 컴파일 거부를 다루지 않음, (c) 이들은 반올림/비결합 아티팩트와 진짜 오염을 결정적 실험으로 가르지 않음. **이 논문을 정면으로 인용하고 차별점을 명시하지 않으면 리뷰어가 중복으로 본다** |
| On the Limitations of Non-GPU AI Accelerators for Large-Model Inference: A Field Study of MoE and Multimodal Serving on Huawei Ascend | arXiv 2607.08215, 2026-07 (Zheng Yu) | **매우 겹침(정신적으로)** — 단일 벤더 NPU(Ascend 910)에 대한 필드 스터디로 **불완전한 연산자/기능 지원, 취약한 병렬성, 저수준 커널의 수치 결함, 미성숙한 그래프 컴파일** 등 8개 한계 범주를 제시. 벤더 플러그인에 12개 소스 패치를 넣어야 했고 정확도 보전을 위해 고성능 기능을 껐다고 보고. 우리 주제의 '단일 벤더라도 논문이 된다'는 존재 증명인 동시에 **가장 위험한 이웃**: 프레이밍이 유사하다. 차별점은 우리 쪽이 (a) 서빙 시스템이 아니라 커널·eDSL 층, (b) 정성 기술이 아니라 전수 매트릭스, (c) 의도적 음성 표본 분리 문제를 명시적으로 다룸 |
| Finding Bugs in MLIR Compiler Infrastructure via Lowering Space Exploration | ASE 2025 (IEEE/ACM, 40th) | 인접 — MLIR 의 **lowering 경로 공간**을 탐색해 미지 버그 38건(오컴파일 8, 크래시 30) 검출. '로워링 경로에 따라 되기도 안 되기도 한다'는 문제의식이 우리와 같지만, 이들은 경로를 바꿔가며 버그를 유도하고 우리는 고정된 단일 로워링 경로가 어떤 프로그램을 거부하는지를 잰다 |
| DESIL: Detecting Silent Bugs in MLIR Compiler Infrastructure | arXiv 2504.01379, 2025 | 인접 — MLIR 의 **silent bug**(크래시 없이 틀린 결과) 검출. 우리 §2.2 의 '조용한 데이터 오배치 2건'이 정확히 이 범주이고, 우리 쪽 강점은 실기 하드웨어에서 관측했고 기전(브로드캐스트 미기록 / 요소-바이트 혼동)까지 규명했다는 점. 오라클 설계 논거를 이 논문에서 빌려올 수 있다 |
| NPUEval: Optimizing NPU Kernels with LLMs and Open Source Compilers | arXiv 2507.14403, 2025-07 | 인접 — AMD NPU 대상 커널 102개 벤치마크로 **컴파일 성공률과 실기 기능 정확도를 함께** 측정. 우리와 측정 축이 같지만 목적이 LLM 코드생성 평가이고, 컴파일 실패를 '모델 능력 부족'으로 귀속시키지 '컴파일러의 로워링 공백'으로 특성화하지 않는다. 우리 프로토콜을 적용할 두 번째 스택 후보로도 유용 |
| Exocompilation for Productive Programming of Hardware Accelerators (Exo) | PLDI 2022 (Ikarashi et al.) | 기반 — 가속기용 eDSL 설계의 대표. 'DSL 의 표현력과 사용자 제어를 어디까지 열 것인가'라는 설계 질문의 원전이고, 우리 결과(표현 가능 영역과 낮출 수 있는 영역이 28% 어긋난다)는 이 설계 질문에 대한 실증 반례로 배치할 수 있다. eDSL 층 자체를 실증한 선행이 사실상 없다는 점이 우리 신규성의 근거이기도 하다 |


---

## 주제 4. DMA 지배 실측과 매핑 수준 최적화의 상한

> **판정: `weak` → 적대적 심사 후 `not-a-paper`**  (논문 아님 — 현재 형태로는 엔지니어링 노트)
> 조사자가 찾은 선행연구 9건 · 심사자가 제기한 치명적 반론 9건 ·
> "이미 출판됨" 지적 12건 · 근거 없이 단정한 문장 8건

### 4.1 초록 — 한 문장 주장

"사이클의 96.5%가 DMA이므로 매핑·타일링·데이터배치를 자동 탐색해 DMA 트래픽을 줄이자"는 제안은, 주장 자체는 MLSys'21 Data Movement Is All You Need 및 DAMOV와 정면으로 겹치고 수단은 Timeloop/ZigZag/Ansor/MLIR-AIR와 겹치며, 결정적으로 근거가 된 130개 커널이 실워크로드가 아니라 컴파일러 단위테스트 스위트(실제 DNN은 mnist 1개)이고 그 스케줄 모델의 DMA 비용이 전송 바이트에 반응하지 않아서 "트래픽 감소"를 측정할 수단조차 없다 — 지금 형태로는 논문이 아니고, 살릴 길은 "프로덕션 NPU 컴파일러 비용모델 감사"로의 선회다.

### 4.2 문제의식과 선행연구의 빈틈

검색으로 확인한 범위에서 진짜로 비어 있는 칸은 제안된 주제가 아니라 그 옆칸이다. (1) TCP(ISCA'24)는 회로스위치 fetch network·입력 브로드캐스트·버퍼 재사용이라는 독자적 재사용 구조를 갖고, vISA는 Chip/Cluster/Slice/Lane/Packet 이라는 `m![]` 매핑 축을 노출하는데, Timeloop/ZigZag/CoSA 계열의 mapspace 정식화는 전부 loop-nest/systolic 템플릿 전제라 이 매핑 공간을 표현하지 못한다 — TCP형 ISA에 대한 mapspace 정식화는 검색에서 못 찾았다. (2) 상용 NPU 컴파일러가 자기 스케줄 모델로 내놓는 사이클 귀속의 충실도를 실기 벽시계와 대조해 감사한 연구를 못 찾았다(cost model fidelity 문헌은 대부분 자기 모델을 self-validate 한다). 반면 "데이터 이동이 지배한다는 특성화" 와 "이동량 최소화 매핑 탐색"은 둘 다 이미 published 다 — 이 두 칸이 제안의 본체다.

### 4.3 제안한 기여

- (제안대로 갈 경우의 기여 — 약함) 상용 NPU 컴파일러 스케줄 모델 기준 엔진별 사이클 귀속을 커널 130개로 전수 집계하고, 슬라이스 내부 연산 최적화의 Amdahl 상한이 3.3%임을 정량화
- (선회안 A, 강함) 벤더 스케줄 모델의 DMA 비용이 전송 바이트에 비실질적으로 반응함을 3중 증거(f32/bf16 동일 72,731 사이클, 대역폭 루프라인 대비 35.5배 이격, util 필드 전 커널 0.0)로 보이고, 이 모델을 목적함수로 쓰는 매핑 오토튜너가 얼마나 잘못된 지점에 수렴하는지를 실기 벽시계와 대조해 측정
- (선회안 A) `--dump-summary` 의 hidden/*_only 분해를 이용해 '엔진 점유 합 116%' 같은 중복계상을 제거한, 겹침 인식 병목 귀속 방법론 제시 — 총점유 68.9% 대 순수점유 38.3% 라는 2배 격차가 최적화 여지 추정을 얼마나 바꾸는지
- (선회안 B) TCP의 Chip/Cluster/Slice/Lane/Packet + 회로스위치 fetch network 를 표현하는 mapspace 정식화와, 그 위의 정렬·런길이 인식 탐색 — Timeloop/ZigZag 템플릿이 표현 못 하는 축이 무엇인지 명시
- (공통) 실기 컴파일 실패 63건의 사유가 DMA 정렬(`not aligned by 8`, `tail_size % min_align`)에 집중된다는, 성능과 컴파일가능성이 같은 제약에서 만난다는 관찰

### 4.4 방법

현재 제안의 방법(스케줄 덤프 집계 → DMA 비중 산출 → 매핑/타일링 자동 탐색)은 두 단계 모두 재사용 부품이다. 살리려면 방법의 무게중심을 옮겨야 한다. 선회안 A: (1) 130개 커널 각각에 대해 `--dump-summary` 의 `dram_usage_io`(실제 전송 바이트)와 DmaEngine 사이클을 짝지어 회귀 — 모델이 바이트에 반응하는지 여부를 정면으로 검정. (2) 같은 커널을 실기 4카드에서 벽시계 측정해 모델 예측 대 실측의 순위상관(Spearman)과 절대오차 분포를 산출 — 오토튜너에 필요한 건 절대값이 아니라 순위이므로 순위가 깨지는지가 핵심 판정. (3) 정렬/런길이/타일크기를 스윕한 변형 커널 집합을 만들어, 모델이 시사하는 최적점과 실기 최적점의 괴리를 측정. (4) 그 괴리를 목적함수 노이즈로 모델링해 매핑 탐색의 수렴 손실을 정량화. 선회안 B: `m![]` 매핑 표현식의 유효공간을 정식화하고(정렬 제약을 탐색공간의 1급 제약으로), 이를 Timeloop mapspace 와 표현력 비교한 뒤 탐색기를 붙인다.

### 4.5 평가 설계

비교 대상은 "DMA 사이클 감소량"이 아니라 두 축이어야 한다. (축1, 모델 감사) 베이스라인 = 벤더 스케줄 모델 예측. 지표 = 실기 벽시계 대비 Spearman 순위상관, MAPE, 그리고 "모델이 고른 top-1 매핑이 실기 top-k 안에 드는 비율". ZigZag가 자기 모델을 5~7.5% 오차로 self-validate 한 것과 대비하면 이 값이 얼마나 나쁜지가 곧 기여다. (축2, 탐색) 베이스라인 = (a) 벤더 기본 매핑, (b) 랜덤 탐색, (c) Timeloop/ZigZag 스타일 이동량 최소화 목적함수, (d) Ansor 계열 레이아웃 재작성. 지표 = 실기 벽시계 지연/처리량과 실측 DRAM 전송 바이트, 탐색 예산 대비 수렴곡선. 워크로드는 반드시 실제 모델(LLaMA prefill/decode, attention 블록)이어야 한다 — 현재 130개 스위트로 평가하면 리뷰어가 첫 문단에서 기각한다.

### 4.6 이미 확보된 근거

- 엔진별 총 사이클 재계산 확인(sched_summary.json 직접 집계): 총 78,171,911 사이클, DmaEngine 75,464,336 = 96.536%, PeCore 2,586,167 = 3.308%, MainContext 0.075%, InterChipTransfer 0.049%, VectorEngine 0.019%, SubContext 0.012%. 제시된 수치와 일치.
- 커널별 DMA 비중 분포 재계산 확인: 중앙값 82.825%, 평균 76.499%, ≥50% 107/130, ≥90% 54/130, ≥99% 19/130, DMA 0%인 커널 1개. 제시된 수치와 일치.
- mnist::forward 재확인(sched_summary.json): span 17,953 / 22 instr / DmaEngine 12,365 / MainContext 7,682 / SubContext 790 / VectorEngine 1,162 / PeCore 600. 07-스케줄링.md §6.8이 인용한 summary.log 의 io 68.874%(12365/17953), io_only 38.311%(6878/17953)은 분모 검산이 소수 3자리까지 재현됨.
- 17,953 사이클이 --dump-schedule 분석기 / 11-MNIST-실행결과 독립기록 / summary.log total_execution_cycle 세 경로에서 일치(07-스케줄링.md §6.8c) — 분석기 자체의 정확성 근거는 확보돼 있다.
- 【중대 문제, 내가 새로 계산】 130개 커널의 정체는 워크로드가 아니라 컴파일러 단위테스트다. 접두사 분포: vector_engine 42, switch_assertions 33, contract_outer_assertions 13, tile 9, scatter_gather 6, contract_element_types 5, at_primitives 4 … mnist 1. 실제 DNN은 mnist::forward 단 1개이고 llama/attention 계열은 0개(perkernel_matrix_fixed.txt 에서 attention::compile_llama3_1_... 는 FAIL). 커널당 인스트럭션 중앙값 5개, 121/130이 20개 이하.
- 【중대 문제, 내가 새로 계산】 96.5% 헤드라인은 소수 마이크로테스트가 만든다. DMA 사이클 상위 3개(at_primitives::vrf::multi_vrf_at, vector_engine::normal::ve_elementwise_multi_vrf, ve_elementwise_vrf — 전부 10~12 instr짜리 VRF 스트레스 테스트)가 전체 DMA 사이클의 43.1%, 상위 10개가 93.75%. 상위 3개 제외 시 총 사이클이 78.2M→45.6M 로 붕괴. 다만 중앙값·≥50%·≥90% 통계는 견고해서(상위 20개 제외해도 88.8%) 정성적 결론 자체는 살아남는다.
- 【중대 문제, 내가 새로 계산】 스케줄 모델의 DMA 비용이 데이터량에 반응하지 않는 정황: sram_bytes 와 DmaEngine 사이클의 log-log 피어슨 상관 r = -0.029 (n=119, 사실상 무상관). sram_bytes 는 전송량이 아니라 SRAM 할당량이라 약한 프록시지만, JSON에 있는 유일한 용량 필드다.
- 위 정황을 뒷받침하는 문서 내 독립 기록 3건: (a) f32 변형이 bf16과 바이트 2배인데 DmaStore 사이클이 72,731로 동일(03-텐서-이동.md:297, 세 덤프 재현), (b) 512KB를 256 B/cyc 로 옮기면 이상적 2,048 사이클인데 모델값 72,731 — 35.5배 이격(03-텐서-이동.md:334), (c) util.total_util/base_util/efficiency 가 커널 130개 전부·모든 인스트럭션에서 0.0(07-스케줄링.md:267). 문서 스스로 '절대 사이클은 상대 비교용으로만'이라고 못박고 있다.
- 【수치 정정】 '470 instr' 는 469다. types 히스토그램 전수 합계: DmaLoad 153 + DmaDtod 186 + DmaStore 113 + DmaGather 11 + DmaStos 2 + DmaOther 2 + DmaInterChip 1 + DmaScatter 1 = 469. 더 심각한 건 문서 표의 SubContext 인스트럭션 27개가 types 의 Sub 144개와 어긋나고 VectorEngine 50에 대응하는 type 이 아예 없다는 점 — 즉 문서 표의 '인스트럭션' 열은 JSON types 필드에서 나온 값이 아니다. 논문에 이 표를 실으려면 출처를 다시 맞춰야 한다.
- 【미출처】 13-NPU-실기-매트릭스.md:378 의 '64B→256B 정렬 변경으로 DmaStore 72,731→1,552 cycle (46.9배)' 는 book_guide 전체 grep 에서 그 한 줄에만 존재하고 _evidence/ 에 뒷받침 로그가 없다. 논문에서 가장 값어치 있는 레버(구체적 레이아웃 조작 → 큰 효과)인데 현재 근거가 없다.

### 4.7 아직 없는, 반드시 해야 할 실험

- 실워크로드 확보가 최우선. LLaMA prefill/decode, attention 블록, MoE 등 실제 추론 그래프에 대해 --dump-schedule/--dump-summary 를 뽑아야 한다. 현재 attention::compile_llama3_1_... 는 컴파일 FAIL 이므로 이 실패부터 해소해야 한다. 이게 안 되면 이 주제는 어떤 형태로도 논문이 안 된다.
- 실기 벽시계 측정. RNGD 4카드에서 동일 커널의 실행시간을 측정해 스케줄 모델 예측과의 Spearman 순위상관·MAPE·top-1 일치율을 산출. 현재 근거는 전부 컴파일러 산출물이고 실측이 0건이다.
- DMA 비용모델의 바이트 의존성 정면 검정. 커널별 dram_usage_io(현재 mnist 419,968B 하나만 기록됨)를 130개 전부 뽑아 DmaEngine 사이클과 회귀. 무상관이면 '트래픽 감소' 목적함수 자체가 이 모델 위에서 무의미함을 증명하는 것이고, 그게 곧 선회안 A의 핵심 결과다.
- 46.9배 정렬 실험(72,731→1,552)의 재현. 현재 미출처. 정렬·런길이를 스윕한 커널 변형 집합을 만들어 모델값과 실기 벽시계를 동시에 측정.
- hidden/*_only 분해의 전수 확대. 현재 io_only 38.311% 는 mnist 1개에만 있다. 130개(또는 실워크로드 전부)에 대해 --dump-summary 를 돌려 '총점유 대 순수점유' 격차 분포를 만들어야 '최적화 여지'를 정직하게 말할 수 있다.
- 매핑 탐색기 자체. mapspace 정식화, 정렬 제약을 1급 제약으로 둔 탐색 알고리즘, 베이스라인(벤더 기본/랜덤/Timeloop형 이동량 최소화/Ansor형 레이아웃 재작성) 4종 구현·비교. 현재 이 부분은 0% 진행이다.
- Timeloop/ZigZag mapspace 와의 표현력 비교. TCP의 Chip/Cluster/Slice/Lane/Packet + 회로스위치 fetch network 중 기존 템플릿이 표현 못 하는 축을 구체적으로 짚어야 신규성 주장이 선다.

### 4.8 ★ 심사 반론

**치명적 반론 (reject 사유)**

- 【근거 population 붕괴 — 내가 새로 계산】 '커널 130개'는 독립 데이터 130점이 아니다. (span, n_inst, engines, sram_bytes) 서명으로 중복 제거하면 **독립 스케줄은 66개뿐**이다(중복군 19개, 최대군은 17개가 완전 동일 — span 22,664 / 5 instr). DMA 1·2위인 at_primitives::vrf::multi_vrf_at 과 vector_engine::normal::ve_elementwise_multi_vrf 는 레코드가 바이트 단위로 동일하고, tile::tile_simple / tile::tile_view_in_loop / typelevel_const / unevaluated_const 4개도 완전 동일하다(span 6,840,200 / 258 instr). 따라서 논문에 실릴 예정인 '107/130', '54/130', '19/130', '중앙값 82.8%' 는 전부 중복 가중된 수치다. 중복 제거 후: n=66, 중앙값 79.1%, ≥50% 53/66, ≥90% 18/66. 결론의 부호는 안 바뀌지만, 리뷰어가 JSON을 10분만 열면 Table 1의 모든 N/130 이 무너진다. 이건 해석 차이가 아니라 집계 오류다.
- 【헤드라인 분모가 시간이 아니다 — 내가 새로 계산】 96.5%의 분모는 sum(engine_cycles)=78,171,911 로, **엔진별 점유의 단순 합**이지 실행 시간이 아니다. sum(span)=76,754,924 이고 engine/span=1.0185, 개별 커널 최대 1.34배까지 초과한다(문서 스스로 §6.8a에서 'io 68.874 + main 42.790 + sub 4.400 = 116.064%' 라고 인정). span 을 분모로 하면 커널별 DMA 비중 **중앙값은 82.8%가 아니라 60.5%**(평균 62.3%)다. 즉 '96.5% / 82.8%' 라는 두 헤드라인 숫자 모두 겹침 미보정 분모로 부풀려져 있고, 정직한 분모(io_only)를 쓰는 순간 mnist에서 38.3%로 떨어진다는 걸 제안 스스로 적어 놓았다. 96.5%로 열고 38.3%로 평가하는 논문은 첫 리뷰에서 죽는다.
- 【실측 부재가 아니라 계측 불가 — 내가 새로 계산】 제안은 '실측 0건'이라 했지만 사실은 npu_matrix.tsv 에 **실기 실행 89행(PASS 80)이 이미 있다**. 문제는 기록된 유일한 시간이 테스트 프로세스 전체 벽시계 ms(PASS 평균 6,525 ms)라는 것이다. 리프 이름으로 68개를 매칭해 Spearman(process_ms, model_span)을 계산하면 **0.137** 이고, 모델이 가장 비싸다고 한 multi_vrf_at(10,845,036 cyc ≈ 10.8 ms @1GHz)이 실기에서는 **가장 빠른 테스트(4,354 ms)**다. 즉 모델이 예측한 커널 지연은 측정 가능한 시간의 0.25%라 현재 하네스로는 보이지도 않는다. 선회안 A의 핵심 방법(벽시계 대비 Spearman/MAPE/top-k)은 '측정만 하면 된다'가 아니라 **존재하지 않는 온디바이스 커널 타이밍 계측을 새로 만들어야** 성립하고, 그 인프라 비용이 제안의 6~9개월 산정에 반영돼 있지 않다.
- 【선회안 A의 방법이 표준 프로토콜이다 — 신규성 없음】 '비용모델 예측 대 실측을 Spearman/MAPE/top-k 로 감사한다'는 건 학습형 비용모델 논문의 **기본 평가 절차**다. TenSet(NeurIPS'21 D&B)이 52M 레코드·6개 하드웨어로 이 프로토콜을 정립했고 top-k accuracy 를 표준 지표로 못박았으며, TLP(ASPLOS'23)·LOOPer(Spearman 0.74)·Kaufman TPU learned cost model 이 모두 같은 지표를 쓴다. PyTorchSim(MICRO'25)은 NPU 시뮬레이터를 **실제 TPUv3 실리콘 대비 MAE 11.5%로 검증**했다. '한 벤더 NPU 컴파일러에 이 표준 프로토콜을 한 번 적용했다'는 벤치마킹 리포트지 연구 기여가 아니다. 리뷰어 질문 한 줄: '당신 방법 중 TenSet/PyTorchSim이 안 한 게 무엇인가?'
- 【선회안 B의 두 축이 모두 선점됨】 (1) 'Timeloop/ZigZag 의 loop-nest 템플릿이 broadcast/fetch network 재사용을 표현 못 한다' — MAESTRO(MICRO'19, IEEE Micro Top Picks'20)의 data-centric mapping representation 이 정확히 multicast/broadcast 와 시공간 재사용을 표현하려고 만들어졌고, 'A Formalism of DNN Accelerator Flexibility'(SIGMETRICS'22, arXiv:2206.02987)가 템플릿 밖 유연성 축을 정식화했으며, Union(PACT'21)이 MLIR 위에서 매핑 탐색기를 통합했다. 제안의 유일한 방어 논거가 이미 published 다. (2) '정렬 제약을 탐색공간의 1급 제약으로' — ROLLER(OSDI'22)의 rTile 이 하드웨어 정렬에 맞춘 타일 형상을 탐색공간 구성 원리로 삼고(비가분·하드웨어 정렬 타일 + 패딩), Heron(ASPLOS'23)이 DLA 제약을 정적 분석으로 자동 생성해 constrained space 를 만들고 제약 만족 문제 위에서 유전 알고리즘을 돌린다. 제안된 방법의 문장 그대로다.
- 【'NPU mapspace 특성화' 자리도 이미 IISWC에 있다】 제안이 1순위 투고처로 꼽은 IISWC에 'Demystifying Map Space Exploration for NPUs'(Kao, Parashar, Tsai, Krishna, IISWC 2022)가 있다. 새 매퍼를 제안하지 않고 매핑 축들이 성능에 어떻게 기여하는지, 서로 다른 탐색 기법이 map-space 를 어떻게 항해하는지를 apples-to-apples 로 비교한 first-of-its-kind 특성화 연구다. 제안의 선회안 B가 노리는 정확히 그 칸이다. 게다가 'Multi-level ML-Guided Autotuning for Code Generation on a Deep Learning Accelerator'(LCTES 2025)가 제안이 2순위로 꼽은 LCTES에 이미 있다.
- 【실워크로드 0 — 이건 보완이 아니라 전제 실패】 130개 중 실제 DNN은 mnist 1개고, perkernel_matrix_fixed.txt 를 직접 확인하니 attention::compile_llama3_1_mlperf_..._prefill(7행)뿐 아니라 **transformer::attention::forward(139행)도 FAIL** 이다. 즉 이 칩의 컴파일러 위에서 트랜스포머 블록이 하나도 낮춰지지 않는다. 'LLM 추론이 DMA-bound 다'라는 주장을 뒷받침할 데이터가 0인 게 아니라, 그 데이터를 만들 능력 자체가 아직 없다. 접두사 분포(vector_engine 42, switch_assertions 33, contract_outer_assertions 13 …)와 인스트럭션 중앙값 5개는 이게 워크로드 스위트가 아니라 컴파일러 어서션 테스트임을 스스로 말한다. 리뷰어는 Table 1의 커널 이름만 보고 기각한다.
- 【기여가 연구가 아니라 사내 엔지니어링 노트다】 '한 벤더 컴파일러의 덤프를 집계했더니 DMA가 크더라' + '그 벤더 비용모델이 이상하더라'는, 어느 학회에도 일반화되지 않는 단일 벤더 단일 칩 관측이다. 일반화 방어선으로 제시된 'TCP는 텐서 컨트랙션 네이티브라 기존 mapspace로 표현 안 된다'는 명제는 현재 근거에 한 줄도 없고 정식화도 안 돼 있으며, 위 MAESTRO/Formalism 반례로 이미 약해져 있다. 이걸 못 세우면 남는 건 '메모리가 병목이다'라는 2021년 문장이다.
- 【자기잠식 그대로】 유일한 실워크로드 mnist에서 겹침 보정 순수 점유는 io_only 38.311%(6,878/17,953)다. 노출된 DMA를 전부 0으로 만들어도 Amdahl 상한은 1.62배. 그런데 같은 커널에서 main_only 12.226%(2,195)가 두 번째로 크므로 '연산 최적화 상한 3.3%'라는 논문의 킬러 문장조차 겹침 보정 후에는 성립하지 않는다(PeCore 600/17,953=3.34% 는 총점유 기준). 제안의 서사 축(3.3% 상한 → 그러니 매핑으로 가라)이 유일한 실데이터에서 자기모순이다.

**이미 출판되어 신규성이 없는 부분**

- Demystifying Map Space Exploration for NPUs — Sheng-Chun Kao, Angshuman Parashar, Po-An Tsai, Tushar Krishna, IISWC 2022 (arXiv:2210.03731, DOI 10.1109/IISWC55918.2022.00031). 새 매퍼를 내지 않고 NPU 매핑 축이 성능·효율에 어떻게 기여하는지와 서로 다른 탐색 기법의 map-space 항해를 apples-to-apples 로 비교한 first-of-its-kind 특성화. 제안의 선회안 B(그리고 1순위 투고처 IISWC)가 노린 칸을 그대로 차지하고 있다.
- MAESTRO: A Data-Centric Approach to Understand Reuse, Performance, and Hardware Cost of DNN Mappings — Kwon, Chatarasi, et al., MICRO 2019 / IEEE Micro Top Picks 2020 (arXiv:1805.02566). 3개 directive 로 된 data-centric mapping representation 이 **multicast/broadcast 재사용을 명시적으로 표현**한다. 제안이 'Timeloop/ZigZag loop-nest 템플릿으로는 TCP의 입력 브로드캐스트·fetch network 를 표현 못 한다'고 세운 유일한 신규성 논거를 정면으로 무력화한다.
- A Formalism of DNN Accelerator Flexibility — SIGMETRICS 2022 (arXiv:2206.02987). 고정 템플릿 밖의 가속기 매핑 유연성 축을 정식화. '기존 정식화가 이 매핑 공간을 표현하지 못한다'는 주장은 이 논문을 먼저 반박해야 한다.
- ROLLER: Fast and Efficient Tensor Compilation for Deep Learning — Zhu et al., OSDI 2022. rTile 추상화가 **하드웨어 정렬 특성에 맞는 텐서 형상을 탐색공간 구성 원리 자체로** 삼는다(비가분이지만 하드웨어 정렬된 타일 크기 + 축 융합 + 패딩). 제안의 '정렬 제약을 탐색공간의 1급 제약으로 둔다'는 기여를 선점.
- Heron: Automatically Constrained High-Performance Library Generation for Deep Learning Accelerators — Jun Bi et al., ASPLOS 2023 (DOI 10.1145/3582016.3582061). compute 정적 분석으로 DLA 고유 제약을 자동 생성해 constrained space 를 만들고, 구체적 해가 아니라 제약 만족 문제 위에서 진화 탐색을 돌린다. 3개 DLA에서 SOTA 4종 대비 평균 2.71배. 선회안 B의 방법 설명과 문장 단위로 겹친다.
- TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers — Zheng, Liu et al., NeurIPS 2021 Datasets & Benchmarks. 6개 하드웨어 52M 성능 레코드로 비용모델 평가 지표(top-k accuracy, pairwise accuracy, RMSE)를 정립. 후속 TLP(ASPLOS'23)가 top-k 를 표준으로 채택. 선회안 A의 '지표 = Spearman/MAPE/top-1 일치율'은 이 라인의 기성 프로토콜이다.
- PyTorchSim: A Comprehensive, Fast, and Accurate NPU Simulation Framework — MICRO 2025 (DOI 10.1145/3725843.3756045). NPU 성능 모델을 **실제 Google TPUv3 실리콘과 대조해 MAE 11.5%로 검증**. '상용 NPU 비용모델을 실기와 대조한 연구를 못 찾았다'는 제안의 gap 주장에 대한 반례.
- LOOPer: A Learned Automatic Code Optimizer for Polyhedral Compilers (arXiv:2403.11522) / A Deep Learning Based Cost Model for Automatic Code Optimization (MLSys 2021, arXiv:2104.04955) / Learned TPU Cost Model for XLA Tensor Programs (Kaufman et al., MLforSystems@NeurIPS 2019). 비용모델 예측 대 실측을 Spearman(0.74, 0.95 등)·MAPE·nDCG 로 감사하는 것이 표준 절차임을 보여주는 3편. 선회안 A는 이 절차를 새 대상에 한 번 적용하는 것 이상이 아니다.
- From Principles to Practice: A Systematic Study of LLM Serving on Multi-core NPUs — Zhu, Feng, Feng, Xia (arXiv:2510.05632, 2025-10). 멀티코어 NPU에서 텐서 병렬 전략·코어 배치·메모리 관리·PD 분리/융합을 다층 시뮬레이터로 체계 분석해 SOTA 대비 1.32~6.03배. '상용 NPU에서 실제 LLM 서빙 워크로드로 배치·메모리를 체계 연구' 칸이 이미 차 있다.
- Multi-level Machine Learning-Guided Autotuning for Efficient Code Generation on a Deep Learning Accelerator — LCTES 2025 (DOI 10.1145/3735452.3735538). 제안이 선회안 B의 대안 투고처로 지목한 LCTES에, DL 가속기 코드 생성 오토튜닝이라는 같은 주제로 이미 실려 있다.
- Union: A Unified HW-SW Co-Design Ecosystem in MLIR for Evaluating Tensor Operations on Spatial Accelerators — PACT 2021 (arXiv:2109.07419). MLIR 위에서 매핑 탐색기와 비용모델을 플러그인화해 공간 가속기 위 텐서 연산을 평가. 'MLIR 기반 상용 NPU 매핑'이라는 자리를 MLIR-AIR 와 함께 이미 채운다.
- (제안이 이미 인용한 것 재확인) Data Movement Is All You Need (MLSys 2021) — 주장 동일. DAMOV (IEEE Access 2021, 345 앱 / 77K 함수) — 특성화 방법론 동일. Timeloop (ISPASS'19) / ZigZag (TC'21, 자기모델 5~7.5% 오차 검증) / Ansor (OSDI'20, 레이아웃 재작성으로 ResNet-50 ~40%) / MLIR-AIR (TRETS'26, 상용 NPU matmul 78.7% compute efficiency) — 수단 전부 선점.

**근거 없이 단정한 문장 (수정 필요)**

- '실기 컴파일 실패 63건의 사유가 DMA 정렬(not aligned by 8, tail_size % min_align)에 집중된다' — perkernel_matrix_fixed.txt 는 필드가 정확히 2개(`OK|이름` / `FAIL|이름`)뿐이고 **실패 사유 컬럼이 아예 없다**(63개 FAIL 전부 NF=2). 정렬 에러 문자열은 문서 산문에 matmul_chip_reduce / matmul_cluster_reduce / vrf_add 등 개별 사례로만 등장한다. 더 나쁜 건 같은 저장소의 05-축약엔진-DPE.md:301 이 정반대를 명시한다는 것이다: '나머지 5개는 실패 지점이 서로 다르다(한 계열로 묶으면 트리아지가 틀린다)' — Branch conversion is not yet implemented, IndexAccess 검증 실패, 그리고 컴파일러 ICE 까지 섞여 있다. '정렬에 집중'은 근거 없는 일반화이며 저장소가 스스로 경고한 오류다.
- '64B→256B 정렬 변경으로 DmaStore 72,731→1,552 cycle (46.9배)' — vISA 트리 전체 grep(md/txt/json/log) 결과 **13-NPU-실기-매트릭스.md:378 한 줄에만 존재**하고 _evidence/ 어디에도 뒷받침 로그가 없다. 출처 표기가 '지난 세션 실측'뿐이다. 논문에서 가장 값어치 있는 레버(레이아웃 조작 → 46.9배)인데 재현 근거가 0이다. 이 수치가 살아 있는 한 이 논문은 인용 불가 데이터를 핵심 주장으로 쓴다.
- '실기 벽시계 실측이 0건이다' — 틀렸다. _evidence/logs/npu_matrix.tsv 에 실기 실행 89행이 있다(PASS 80 / FAIL 5 / ABORT 3 / OTHER 1). 다만 기록된 시간이 테스트 프로세스 전체 소요 ms(PASS 평균 6,525 ms, 범위 1,892~67,289)라 커널 지연으로 쓸 수 없다. 주장의 방향은 맞지만 문장이 사실과 다르고, 이 파일의 존재가 오히려 '벽시계를 재면 된다'는 fix 가 즉시 실행 불가임을 보여준다(모델 예측 10.8 ms 커널이 4.35 초 프로세스 안에 파묻힌다).
- 'sram_bytes 와 DmaEngine 사이클의 log-log 상관 r=-0.029 → 모델의 DMA 비용이 데이터량에 반응하지 않는다' — 값 자체는 재현했다(r=-0.0291, n=119). 그러나 추론이 성립하지 않는다. sram_bytes 는 전송량이 아니라 SRAM 할당량이고, 결정적 반례가 데이터 안에 있다: **tile::tile_simple 은 sram_bytes=0 인데 DmaEngine 6,671,008 사이클**이다(DmaDtod 32회). 전송이 DM↔DM 이면 sram_bytes 가 0으로 잡히므로 이 필드는 전송량의 약한 프록시가 아니라 **무효한 프록시**다. 무상관은 모델의 결함이 아니라 프록시의 결함으로도 완전히 설명된다. 이 한 줄로 '3중 증거' 중 하나가 빠진다.
- 'DmaEngine 470 instr' — sched_summary.json 의 types 히스토그램 전수 합계는 **469**다(DmaDtod 186 + DmaLoad 153 + DmaStore 113 + DmaGather 11 + DmaStos 2 + DmaOther 2 + DmaInterChip 1 + DmaScatter 1). 더 심각한 건 문서 표(13-NPU-실기-매트릭스.md:313-318)의 인스트럭션 열 전체가 JSON과 안 맞는다는 점이다: 표의 SubContext 27 vs types 의 Sub 144, 표의 VectorEngine 50 에 대응하는 type 은 **아예 존재하지 않는다**(types 키는 Core/Dma*/Sub/Main 뿐). 즉 이 표의 인스트럭션 열은 출처 불명이다. n_inst 합은 2,278, Core 타입만 1,557 이다.
- 'TCP의 Chip/Cluster/Slice/Lane/Packet + 회로스위치 fetch network 는 기존 mapspace 템플릿으로 표현 못 한다' — 제안 스스로 '근거에 전혀 등장하지 않고 정식화도 안 돼 있다'고 인정한 명제인데, contributions 와 novelty_confidence 는 이걸 선회안 B의 신규성 근거로 쓴다. 게다가 MAESTRO 의 data-centric representation(multicast/broadcast 명시)과 'A Formalism of DNN Accelerator Flexibility' 가 반례로 존재한다. 표현력 비교 실험을 하기 전에는 이 문장을 논문에 쓸 수 없다.
- '커널 130개' 라는 표본 크기 자체 — 독립 스케줄은 66개다(중복군 19개). '130개 전수 집계'라는 표현은 논문에서 표본 크기를 두 배 부풀린 주장이 된다.
- 'DMA 인스트럭션 하나가 연산 인스트럭션 하나보다 두 자릿수 이상 비싸다'(13-...md:322) — 인스트럭션 열 자체가 JSON과 안 맞으므로(위 항목) 이 비율 계산의 분모가 검증되지 않았다. 또한 DmaEngine 사이클의 43.1%가 동일 레코드 3개(그중 2개는 완전 중복)에서 나오므로 '평균 DMA 인스트럭션 비용'은 정의상 의미가 없다.

**조사자 스스로 적은 위험**

- 【치명】 근거의 워크로드 대표성이 없다. 130개 중 실제 DNN은 mnist 1개, 나머지는 switch_assertions/contract_outer_assertions 같은 컴파일러 어서션 테스트다. 'DMA가 96.5%'는 이 칩이 LLM 추론에서 DMA-bound 라는 뜻이 전혀 아니다. 리뷰어는 Table 1에서 커널 이름만 보고 기각한다.
- 【치명】 목적함수를 측정할 수단이 없다. 스케줄 모델의 DMA 비용이 바이트에 반응하지 않는 정황이 3중으로 있다(f32=bf16 동일 사이클, 루프라인 대비 35.5배, util 전부 0.0, sram_bytes 상관 -0.029). 이 모델 위에서 'DMA 트래픽을 줄였다'를 보여도 아무 의미가 없고, 실기 측정 없이는 반증도 불가능하다.
- 【신규성】 주장 절반은 MLSys'21 Data Movement Is All You Need 가 이미 했다 — 데이터 이동 지배 + Amdahl 논거 + 이동량 22.91% 감소로 1.30배. 방법론 절반은 DAMOV(IEEE Access'21)가 345개 앱 77K 함수 규모로 이미 했다. 130개 단위테스트로 그 옆에 서기는 어렵다.
- 【신규성】 '매핑/타일링 자동 탐색으로 off-chip 이동 최소화'는 Timeloop(ISPASS'19), ZigZag(TC'21), CoSA, Ansor(OSDI'20)의 정의 그 자체다. Ansor는 이미 레이아웃 재작성으로 ResNet-50에서 40%를 얻었다. 더 나쁘게도 MLIR-AIR(TRETS'26)는 상용 NPU에서 타일링·비동기·통신중첩·명시적 DMA 채널을 다 하며 matmul 78.7% compute efficiency 를 보고했다 — '상용 NPU라서 새롭다'는 방어선이 이미 뚫려 있다.
- 【일반성】 단일 벤더·단일 칩이다. 반론에 대한 유일한 방어는 'TCP는 fixed-size matmul 로 분해하지 않는 텐서 컨트랙션 네이티브 구조라 기존 mapspace 템플릿으로 표현되지 않는다'인데, 이 논거는 현재 근거에 전혀 등장하지 않고 정식화도 안 돼 있다. 이걸 세우지 못하면 '어느 칩이든 메모리가 병목이다'라는 이미 아는 얘기가 된다.
- 【GPU에서 이미 다 한 얘기】 실제로 상당 부분 그렇다. Amdahl/roofline 논거, 데이터 이동 지배, 레이아웃 자동 탐색 전부 GPU 문헌에 있다. NPU 고유의 차별점으로 남는 건 (a) 정렬 실패가 성능이 아니라 컴파일가능성을 막는다는 점, (b) 소프트웨어 관리 스크래치패드에서 오버랩 구조가 hidden/*_only 로 명시적으로 노출된다는 점 정도인데, 둘 다 아직 논문 수준으로 파고들지 않았다.
- 【자기잠식】 유일한 실워크로드 mnist에서 정직한 수치는 io_only 38.3%다. 즉 모든 노출 DMA를 0으로 만들어도 Amdahl 상한은 1.62배다. 96.5%로 논문을 열고 38.3%로 평가하면 리뷰어가 정확히 이 모순을 짚는다.
- 【방어 가능한 부분】 상위 마이크로테스트 3개를 빼도 중앙값·≥50%·≥90% 통계는 견고하다(상위 20개 제외해도 88.8%). 'DMA가 비싸다'는 정성적 결론 자체는 살아남는다. 문제는 그 결론이 새롭지 않다는 것이지 틀렸다는 게 아니다.

### 4.9 살리는 길 — 무엇을 바꿔야 하는가

현재 제안(및 두 선회안 모두)은 살릴 수 없다. 선회안 A는 TenSet/TLP/LOOPer/PyTorchSim이 정립한 표준 평가 프로토콜의 재적용이고, 선회안 B는 MAESTRO(broadcast 표현)·ROLLER(정렬 1급 제약)·Heron(제약 자동생성+제약 탐색)·IISWC'22 Demystifying(NPU mapspace 특성화)이 네 조각을 나눠 갖고 있다. 그래도 굳이 살린다면 순서는 이렇다.

【0단계 — 논문 논의 이전에 반드시 처리】 (a) sched_summary.json 을 서명 기준 중복 제거하라. 130 → 66. 모든 N/130 통계를 다시 계산하고 논문에서 '130개 커널'이라는 표현을 폐기하라. (b) 헤드라인 분모를 sum(engine_cycles)에서 total_execution_cycle(또는 span)로 바꿔라. 96.5%/82.8% 대신 span 기준 중앙값 60.5%, 그리고 겹침 보정 io_only 를 쓰라. 두 수치를 섞어 쓰는 순간 리뷰어가 §6.8b를 그대로 인용해 기각한다. (c) 13-...md:378 의 46.9배(72,731→1,552)와 '63건 FAIL이 정렬에 집중'은 근거가 없으므로 재현하기 전까지 문서와 슬라이드에서 삭제하라. (d) sram_bytes 상관 r=-0.029 는 tile::tile_simple(sram_bytes=0, DMA 6.67M cyc) 반례 때문에 증거로 쓸 수 없다. 폐기하고 dram_usage_io 실측으로 대체하라.

【1단계 — 유일하게 남은 진짜 문제로 주제를 갈아라】 지금 이 저장소가 가진 것 중 선행연구가 채우지 않은 칸은 'DMA가 96.5%'가 아니라 **'텐서 컨트랙션 네이티브 ISA 위에서 트랜스포머 블록이 낮춰지지 않는다'** 는 사실이다. attention::compile_llama3_1_..._prefill 과 transformer::attention::forward 가 둘 다 FAIL 이고, 커널 200개 중 63개가 --backend npu 에서만 막히며, emulation 은 104 passed / 0 failed 다. 'typecheck·emulation 통과 ≠ 실기 lowering 가능'이라는 이 격차 자체가 컴파일러 논문의 소재다. 해야 할 일: 63개 FAIL 전부에 대해 컴파일러 에러 원문을 수집해 사유 컬럼을 만들고(현재 perkernel_matrix_fixed.txt 는 2필드뿐), 정렬/분기/다중칩/ICE 로 트리아지한 뒤, 각 사유가 vISA 의 어떤 매핑 표현식(m![] 축, to_dm 정렬, tail_size)에서 발생하는지를 정식화하라. 그러면 기여가 '메모리가 병목이다'(이미 안다)에서 **'이 ISA의 매핑 표현공간에서 유효/무효 경계가 어디인가, 그리고 그 경계가 왜 실제 트랜스포머 블록을 배제하는가'**(안 알려짐)로 바뀐다. 이건 Heron/ROLLER 가 다루는 '유효 탐색공간 제약'과 같은 문제지만, 그들이 전제한 '컴파일은 되는데 느리다'가 아니라 '컴파일 자체가 안 된다'는 더 강한 제약 체제라 차별화가 선다. 투고처: CGO / LCTES, 성공하면 PACT.

【2단계 — 계측을 먼저 만들어라, 논문은 그 다음】 벽시계 감사를 하려면 커널 단위 온디바이스 타이밍이 필요하다. 현재 있는 건 테스트 프로세스 전체 ms(평균 6,525 ms)뿐이고, 모델이 가장 비싸다고 한 커널조차 예측 10.8 ms 라 프로세스 시간의 0.25%다 — 지금 하네스로는 원리적으로 측정 불가다. 하드웨어 카운터나 커널 반복 실행 + 워밍업 기반 마이크로벤치 하네스를 먼저 만들어야 하고, 이건 제안의 '2개월'이 아니라 그 자체로 별개 작업이다. 이 인프라가 서기 전에는 비용모델 감사 논문을 계획에 넣지 마라.

【3단계 — 그래도 비용모델 감사를 쓰겠다면】 신규성 방어선은 '했다'가 아니라 '한 결과가 기존 문헌과 정성적으로 다르다'여야 한다. ZigZag 5~7.5%, PyTorchSim 11.5% MAE, LOOPer Spearman 0.74, TenSet 계열 top-k 가 이미 있으므로, 당신 결과가 예컨대 'Spearman 0.1대 / top-1이 실기 top-10에도 못 든다' 처럼 **질적으로 다른 실패 체제**를 보여야 논문이 된다. f32/bf16 동일 72,731 사이클과 루프라인 대비 35.5배 이격이 그 방향을 시사하긴 하나, 현재는 단일 커널 일화이고 util 필드가 전부 0.0이라 원인 규명이 불가능하다. 최소 요구: (i) 실워크로드 컴파일 성공, (ii) 커널 단위 실기 타이밍, (iii) dram_usage_io 130개(중복 제거 후 66개) 전수 수집 후 DmaEngine 사이클과 회귀. 이 셋 중 하나라도 없으면 쓰지 마라.

【쓰지 말아야 할 것】 '96.5%가 DMA' 로 논문을 여는 모든 버전. 그 문장은 (a) 분모가 시간이 아니고, (b) 표본이 66개 중복 포함 단위테스트이며, (c) 상위 3개(그중 2개는 동일 레코드)가 43.1%를 만들고, (d) 유일한 실워크로드에서 정직한 값은 38.3%다. 네 가지가 전부 같은 표에서 확인 가능하다. 이 문장을 motivation 으로도 쓰지 마라 — 다른 논문의 도입부에 들어가도 같은 반박을 받는다.

### 4.10 후보 학회와 소요 기간

- (선회안 A: 컴파일러 비용모델 감사) IISWC — 워크로드/모델 특성화의 자연스러운 집. 지금 근거 성숙도에 가장 현실적.
- (선회안 A) ISPASS — 성능분석·측정 방법론. Timeloop이 나온 곳이라 mapspace 논의와도 붙는다.
- (선회안 B: mapspace 정식화 + 탐색) CGO 또는 LCTES — 컴파일러 쪽 기여로 프레이밍할 경우.
- (선회안 B가 실기 성능까지 붙을 경우) PACT, 그 다음 MICRO/HPCA. 단 현재 근거로는 무리.
- (현 상태 그대로라면) 학회 본회의보다 MICRO/HPCA 병설 워크숍 또는 arXiv 기술보고서. 또는 다른 논문의 motivation 절.

**소요 기간 추정**: 현 형태 유지 시: 논문이 안 되므로 기간 산정 무의미. 선회안 A(비용모델 감사)로 IISWC/ISPASS 급을 노릴 경우 6~9개월 — 실워크로드 컴파일 실패 해소 1~2개월, 실기 벽시계 측정 인프라 및 130+실모델 전수 측정 2개월, 바이트 의존성 검정과 정렬 스윕 재현 1~2개월, 오토튜너 수렴 손실 정량화 2개월, 집필 1개월. 선회안 B(TCP mapspace 정식화 + 탐색기)까지 가면 추가 6개월, 총 12~15개월.

**신규성 확신도**: 신규성 없음에 대한 확신도 높음(제안된 형태 기준). 근거: 8회 검색에서 주장 쪽(Data Movement Is All You Need, MLSys'21), 방법론 쪽(DAMOV, IEEE Access'21), 수단 쪽(Timeloop ISPASS'19 / ZigZag TC'21 / Ansor OSDI'20), 그리고 '상용 NPU에서 실제로 그걸 한' 쪽(MLIR-AIR, TRETS'26)이 각각 명확히 확인됐다. 네 칸이 모두 채워져 있고 제안은 그 교집합이다. 반대로 선회안에 대한 신규성 확신도는 중간이다: (a) 상용 NPU 컴파일러 스케줄 모델을 실기 벽시계와 대조해 감사한 논문은 검색에서 못 찾았고(cost model fidelity 문헌은 대부분 자기 모델 self-validation 이며, 검색 중 '한 컴파일러의 해석적 비용함수 중앙값 오차 31%, 절대 오라클이 아닌 순위도구로만 타당' 이라는 언급을 보긴 했으나 그 출처를 특정하지 못해 related_work 에 넣지 않았다), (b) TCP형 텐서컨트랙션 네이티브 ISA의 mapspace 정식화도 못 찾았다. 다만 '못 찾았다'는 '없다'가 아니고, 두 선회안 모두 현재 근거가 0에 가깝다. 마지막으로 정직하게: 이 조사에서 가장 중요한 발견은 선행연구가 아니라 자기 근거의 결함이다 — 130개가 워크로드가 아니라 단위테스트라는 점, 그리고 스케줄 모델의 DMA 비용이 바이트에 반응하지 않는다는 점. 이 둘은 선행연구와 무관하게 현재 주제를 무너뜨린다.

### 4.11 검색으로 확인한 관련 연구

| 제목 | 학회·연도 | 이 주제와의 관계 |
|---|---|---|
| Data Movement Is All You Need: A Case Study on Optimizing Transformers | MLSys 2021 (arXiv:2007.00072) | 겹침 — 주장이 동일하다. 데이터 이동이 병목이고 Amdahl 때문에 연산 최적화가 막힌다는 논거를 그대로 편다. 데이터 이동 22.91% 감소, BERT 인코더 1.30배. 이 주제의 '핵심 질문'은 이 논문의 도입부다. |
| DAMOV: A New Methodology and Benchmark Suite for Evaluating Data Movement Bottlenecks | IEEE Access 2021 (arXiv:2105.03725) | 겹침 — 방법론이 동일하다. 데이터 이동 병목의 '원인별 분류' 체계를 345개 앱·77K 함수 특성화로 세우고 144개 함수 벤치마크로 배포했다. 130개 단위테스트 기반 특성화가 정면으로 비교당할 상대. |
| From Loop Nests to Silicon: Mapping AI Workloads onto AMD NPUs with MLIR-AIR | ACM TRETS 2026 (arXiv:2510.14871, 2025-10) | 겹침 — 가장 위험한 인접작. 상용 NPU에서 타일링·비동기 실행·통신 중첩을 컴파일러가 관리하고, air.channel/air.memcpy 로 레이아웃과 메모리공간을 명시해 DMA를 제어한다. matmul 78.7% compute efficiency, 손최적화 MLIR-AIE 수준. '상용 NPU 매핑 최적화'라는 자리를 이미 차지하고 있다. |
| Timeloop: A Systematic Approach to DNN Accelerator Evaluation | ISPASS 2019 | 기반 — 아키텍처 템플릿으로부터 완전한 mapspace 를 자동 구성하고 해석적 비용모델로 평가·탐색하는 원형. '매핑 자동 탐색' 기여를 주장하려면 이 mapspace 정식화 대비 무엇이 새로운지를 반드시 답해야 한다. |
| ZigZag: Enlarging Joint Architecture-Mapping Design Space Exploration for DNN Accelerators (및 ZigZag: A Memory-Centric Rapid DNN Accelerator DSE Framework) | IEEE Trans. Computers 2021 / arXiv:2007.11360 | 겹침 — 메모리 중심 DSE. W/I/O 오퍼랜드·메모리계층·시공간 매핑을 분리한 uneven mapping 으로 메모리 접근수·요구대역폭을 직접 산출해 최적화한다. 즉 '이동량을 목적함수로 매핑을 탐색'하는 일은 이미 되어 있다. 자기 모델을 실측 대비 5~7.5% 오차로 검증한 점도 이 주제의 비용모델 문제와 대비된다. |
| Ansor: Generating High-Performance Tensor Programs for Deep Learning | OSDI 2020 (arXiv:2006.06762) | 인접 — 템플릿 없는 오토스케줄러. 전역 레이아웃 탐색은 안 하지만 가중치 텐서 레이아웃 재작성 + 다단 타일링으로 ResNet-50에서 약 40% 개선을 이미 얻었다. '레이아웃·타일링 자동 탐색'의 표준 베이스라인. |
| TCP: A Tensor Contraction Processor for AI Workloads (Industrial Product) | ISCA 2024 | 기반 — 대상 칩 자체. 회로스위치 기반 fetch network, 입력 브로드캐스트, 입력버퍼 기반 재사용으로 데이터 재사용을 이미 아키텍처 차원에서 다루고 있다. 5nm, 256MB SRAM, 1.5TB/s HBM3. 벤더가 이미 재사용 서사를 발표했다는 점이 신규성 주장의 걸림돌이자, 동시에 'TCP mapspace는 기존 템플릿으로 표현 안 된다'는 선회 논거의 근거이기도 하다. |
| FuriosaAI RNGD: A Tensor Contraction Processor for Sustainable AI Computing | IEEE Micro 2025 (Hot Chips 2024 theme article) | 기반 — 같은 칩의 제품 레벨 기술 문서. 이 주제가 다루는 하드웨어의 공식 레퍼런스. |
| DeFiNES: Enabling Fast Exploration of the Depth-first Scheduling Space for DNN Accelerators through Analytical Modeling | arXiv:2212.05344 (2022) — 학회 확정 미확인 | 인접 — 레이어 융합/depth-first 스케줄 공간을 해석적 모델로 빠르게 탐색. 온칩에 안 들어가는 텐서의 오프칩 이동을 줄이는 방향으로, 제안된 '매핑 수준 상한' 논의와 같은 문제를 다룬다. 학회·연도를 확정하지 못해 이 항목은 참고 수준으로만. |


---

## 주제 5. 사전출시 NPU 툴체인의 재현가능 평가 방법론

> **판정: `weak` → 적대적 심사 후 `not-a-paper`**  (논문 아님 — 현재 형태로는 엔지니어링 노트)
> 조사자가 찾은 선행연구 15건 · 심사자가 제기한 치명적 반론 12건 ·
> "이미 출판됨" 지적 13건 · 근거 없이 단정한 문장 13건

### 5.1 초록 — 한 문장 주장

사전출시 NPU 툴체인에서 "무엇이 실기에서 도는가"라는 수치는 하드웨어가 아니라 평가 하네스가 결정한다 — 빌드 결합·장치세션 오염·선택자 별칭·오라클 강도라는 네 레버가 같은 하드웨어·같은 커널에서 보고값을 0↔80, 10↔32, 136↔137, 통과↔조용한 오류로 뒤집는다.

### 5.2 문제의식과 선행연구의 빈틈

8회 검색으로 확인한 결과, 네 위험의 **개별 기전은 전부 이미 published 다**: (2) 순서의존 오염은 iDFlakies(ICST'19)가 정확히 victim/polluter 로 형식화했고 데이터셋의 50.5%가 order-dependent 다. (4) 허용오차로 인한 거짓 FAIL/거짓 PASS 는 2026년 커널 벤치 논문들(Correctness Illusion, FastKernels 의 reference-vs-reference 비결정성 기반 tolerance 보정)이 더 체계적으로 다뤘다. 컴파일/정확성/성능을 독립 축으로 나누는 것은 CANN Bench(2026)가 이미 한다. 측정 편향 일반론은 ASPLOS'09, 체크리스트화는 Benchmarking Crimes(2018)·SIGPLAN 가이드라인이 선점했다. 못 찾은 것 세 가지만이 진짜 gap 이다: (a) **AOT 가 패키지 내 모든 device 함수를 빌드시점에 낮추는 탓에 "평가 단위"가 빌드에 의해 파괴되는 현상** 자체를 다룬 논문, (b) **커널 hang 이 HAL/드라이버 세션을 오염시켜 같은 프로세스의 후속 커널을 전부 실패시키는 '장치상태형 순서의존'의 정량화** — SE 쪽 OD 연구는 힙·파일시스템 공유상태만 본다, (c) **컴파일 성공 → 커널 로드 성공 → 값 정확 의 3단 감쇠를 실기에서 실측한 NPU 연구** (CANN Bench 는 로드 단계를 분리하지 않는다). 이 셋만으로 독립 논문 한 편을 지탱하기는 얇다.

### 5.3 제안한 기여

- 측정 위험 분류체계(H1 빌드 결합 / H2 장치세션 오염 / H3 선택자 별칭 / H4 오라클 강도)와 각 위험이 보고 수치에 주는 델타의 정량화 — '통과 수는 하드웨어의 성질이 아니라 하네스의 함수'라는 명제를 숫자로 고정한다
- 재사용 가능한 하네스 2종: 테스트당 1프로세스 + timeout 150s + HAL(-110)·값불일치·kernel_load 태깅 실행기(npu_matrix.sh), 값 불일치를 ULP_ROUNDING / REAL_CORRUPTION / REAL_MISMATCH 로 자동 3분류하는 오라클(classify_mismatch.py)
- 컴파일→로드→값정확 3단 감쇠 프로토콜과 그 실측(커널 200개 중 137개 컴파일, 테스트 89개 중 3건 로드단계 abort, 2건 조용한 데이터 오배치) — 'compile 성공 = 동작'이라는 흔한 가정의 반례를 단계별로 계량
- AOT 전역 결합 회피 레시피의 형식화: 커널 단위 컴파일 매트릭스 → 실패분에만 cfg 게이트 삽입 → 실기 실행. 평가 가능 테스트 수를 0 → 89 로 여는 절차이자, 동시에 게이팅이 남은 결과에 주는 낙관 편향을 명시적으로 회계처리하는 방법
- 보고 규약 제안: 'N passed' 대신 (격리 단위, 선택자 매칭 규칙, 허용오차 정책, 오라클 강도, 게이트된 항목 수)를 함께 보고하도록 요구하는 사전출시 툴체인용 체크리스트

### 5.4 방법

평가 프로토콜을 실험 대상으로 삼는 요인설계. 프로토콜 사다리 L0~L3 을 정의한다. L0(관행): 크레이트 통째 빌드 + `cargo test` 1회 + 통과/실패만 집계. L1: 커널 단위 컴파일 매트릭스 + cfg 게이팅으로 빌드 결합 해소(H1). L2: 테스트당 1프로세스 + 타임아웃 + 장치상태 사후검사(furiosa-smi)로 오염 차단(H2). L3: 정확 이름 매칭(H3) + 값 오라클 + ULP/비결합연산 순서 민감도 분류(H4). 같은 하드웨어·같은 커널 집합에 L0→L3 을 차례로 적용해 보고 수치가 어떻게 이동하는지를 측정하고, 각 이동분을 위험별로 귀속시킨다. H2 의 인과는 대조실험으로 확정한다: hang 커널 단독 실행(60초 타임아웃 실패) vs 피해 커널 단독 실행(0.13초 통과), 그리고 --test-threads=1 직렬화가 무효임을 보여 '동시성이 아니라 프로세스 수명 내 장치상태 오염'임을 분리한다. H4 는 결정적 프로브로 확정한다: saturating_add 리듀스 실패를 포화 불가능한 소값 입력(-50..50)으로 재실행해 통과시키면 '데이터 오염'이 아니라 '병렬 리듀스 순서'임이 증명된다. 일반성은 툴체인 축을 늘려 확보한다(furiosa-opt 외 최소 1~2개 스택에서 같은 사다리 반복).

### 5.5 평가 설계

비교 대상은 시스템이 아니라 **평가 프로토콜**이다. 베이스라인 = L0(관행적 하네스), 대조 = L1/L2/L3. 지표: (a) 보고 가능한 테스트 수 Δ (0 → 89), (b) 통과 수·통과율 Δ 와 그 재실행 분산 (vector_engine 10 → 32), (c) 오분류율 — 거짓 FAIL 건수(선택자 별칭 1/200, 연쇄 피해자 22/36, ULP·순서 아티팩트 2/6)와 거짓 PASS 건수(값 오라클 없을 때 놓치는 조용한 오배치 2건), (d) 결함 유형별 회수율 — L0 대비 L3 이 몇 개의 서로 다른 결함 기전을 드러내는가(4기전 6건), (e) 비용 — 프로세스 격리의 벽시계 오버헤드(cargo 진입비용 최소 3.5초 × 테스트 수). 설계: {툴체인 ≥2} × {L0..L3} × {반복 k≥5, 순서 무작위화}. 부차적으로 문헌 감사 축을 둔다: 최근 NPU/커널 벤치 하네스 N편이 H1~H4 중 몇 개를 방어하는지 코딩해 '이 위험이 실제로 유통 중인 수치를 오염시킨다'를 보인다(Mytkowicz 의 133편 서베이에 해당).

### 5.6 이미 확보된 근거

- 실기 매트릭스 89행 전수: PASS=80, FAIL=5, ABORT=3, OTHER(ignored)=1 = 89. 직접 재계산으로 확인. 출처 /home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vISA/book_guide/_evidence/logs/npu_matrix.tsv
- 커널 단위 컴파일 판정 200행: OK=137 / FAIL=63 (접두사 충돌 보정 후). 직접 재계산 확인. 출처 _evidence/logs/perkernel_matrix_fixed.txt. 단 보정 전 136/64 의 원본은 이 세트에 없고 서술로만 남아 있다
- 격리 실행 하네스가 코드로 실재: 테스트당 1프로세스 + `timeout 150` + HAL(-110)·`assertion left == right`·`furiosa_kernel_load` 카운트를 TSV 로 태깅. 출처 _evidence/tools/npu_matrix.sh
- 값 불일치 자동 분류기가 코드로 실재: ULP≤2 → ULP_ROUNDING, 불일치율>0.9 → REAL_CORRUPTION, 그 외 REAL_MISMATCH. 정수형은 ULP 계산에서 제외. 출처 _evidence/tools/classify_mismatch.py
- 격리 실행 시 vector_engine 36개 결과: 실제 통과 32, 실패 3, ignored no-op 1. npu_matrix.tsv 의 vector_engine 집계(PASS 32 / FAIL 3 / OTHER 1)와 정확히 교차일치. 출처 _evidence/logs/ve_isolated.log + npu_matrix.tsv
- 실기 실패 6건의 유형 분해가 TSV 와 일치: ABORT 3(reshape×2, shuffle_slice×1 = 커널 로더 범위초과, 요구/실제 비율 1.496·1.496·1.493), FAIL 5(broadcast 1·tile 1 = 조용한 오배치, vector_engine 3 중 1건이 HAL -110 hang, 2건이 ULP/순서 아티팩트). 출처 logs/npu_matrix.tsv + 13-NPU-실기-매트릭스.md §2
- 스케줄 모델 130 커널 집계 재계산 확인: DmaEngine 75,464,336 cycle(96.5%), PeCore 2,586,167(3.3%); 커널별 DMA 비중 중앙값 82.8%, ≥50% 107/130, ≥90% 54/130. 출처 _evidence/logs/sched_summary.json
- 선택자 별칭(H3)의 기전과 규모: 필터가 부분문자열이라 `invalid_time0` 지정이 `invalid_time0_mismatch` 까지 컴파일해 남의 에러를 뒤집어쓴다. 접두사 충돌 후보 200개 중 8개, 실제 오염 판정 1건. 출처 _evidence/GROUND_TRUTH_BRIEF.md L201-205 및 13번 문서 §7.4

### 5.7 아직 없는, 반드시 해야 할 실험

- **최소 1개, 가능하면 2개의 다른 툴체인에서 H1~H4 재현.** 후보: AMD XDNA/Peano(NPUEval 하네스), Ascend CANN(CANN Bench), Tenstorrent Metalium. 이게 임계경로다 — 없으면 '단일 벤더 버그 리포트'라는 반론을 못 막는다
- **문헌 감사.** 최근 NPU/커널 벤치·컴파일러 평가 논문 N(≥40)편을 코딩해 각각이 H1~H4 중 몇 개를 방어하는지 집계. Mytkowicz 의 '133편 중 0편' 에 해당하는 카운터가 없으면 이 논문은 '아무도 안 틀렸는데 혼자 조심한 이야기' 가 된다
- **반복 실행 통계.** 현재 모든 델타가 1회 실행이다. 격리/비격리 각 k≥5회 + 테스트 순서 무작위화(Tuscan 류)로 10→32 델타의 분산과 신뢰구간을 제시. 오염의 시작점(11번째 테스트)이 순서에 따라 어떻게 이동하는지도
- **H2 의 최소 격리 단위 규명.** -110 이후 HAL 상태가 프로세스 재초기화만으로 회복되는지, 디바이스 리셋·컨텍스트 재생성 같은 더 싼 수단으로 대체 가능한지. 프로세스 격리 비용(cargo 진입 최소 3.5초 × 89 ≈ 5분+)의 정당화가 여기 달려 있다
- **조용한 오배치 2건의 소스 수준 확증.** tile 의 '32 elements 를 32 bytes 로 해석' 은 산술 정황(32→8 = ÷4 = sizeof f32)일 뿐 런타임 소스로 확인하지 않았다. broadcast 미기록도 마찬가지
- **미보존 로그 복원.** 이 논문의 간판 델타인 0→58(빌드 63 에러 vs 게이팅 후) 과 10→25(비격리) 의 원본 로그가 _evidence 에 없다. 재현성 논문이 자기 근거를 못 재현하면 치명적이다. 재실행해 아카이브
- **오라클 정책 민감도.** classify_mismatch.py 의 ULP 임계 2, 불일치율 0.9 가 하드코딩이다. 임계를 스윕해 판정이 얼마나 바뀌는지(=이 논문의 오라클 자체가 H4 를 앓지 않는지) 보여야 한다
- **게이팅의 낙관 편향 측정.** 게이트로 제외한 64개가 남은 80/89 를 얼마나 좋아 보이게 만드는지. 제외 커널을 고쳐 살렸을 때 통과율이 어디로 가는지의 상한/하한
- **집계 규칙 자체의 감사.** 사용자가 준 '10→33' 은 실측상 '10→32(+ignored no-op 1)' 이다(아래 risks). 문서 전체의 카운팅 규칙을 기계 검증하는 스크립트가 필요하다

### 5.8 ★ 심사 반론

**치명적 반론 (reject 사유)**

- **논지 자체가 이미 published 다 (2026).** Harness-Bench(arXiv 2605.27922, 2026)는 '보고 수치는 시스템이 아니라 하네스가 결정한다'를 {모델}×{하네스} 요인설계로 측정하고, run-level pass rate 는 같은데 majority-pass/all-replicate 카운트가 갈리는 사례까지 제시한다. 후속으로 'Rethinking the Evaluation of Harness Evolution for Agents'(arXiv 2607.12227, 2026)도 있다. 도메인이 에이전트냐 NPU 커널이냐만 다르고 thesis 문장 구조·실험 설계·결론이 동일하다. 리뷰어가 이 두 편을 알면 '도메인 치환' 이상을 요구한다.
- **H2 의 완화책은 이미 벤치마크 표준이다.** KernelBench(Ouyang et al., 2025)는 '각 제출 솔루션을 전용 서브프로세스에서 컴파일·실행해 evaluator 상태를 격리, 한 솔루션이 뒤 솔루션에 영향 주지 못하게 한다'를 설계에 명시하고 있다. 즉 이 논문이 '발견'이라 부르는 것은 2025년 시점 커널 생성 벤치의 기본 설계 결정이다. 남는 건 '3.2배'라는 숫자뿐인데 그건 n=1 이다.
- **H2 의 인과 전체가 폴루터 단 1건에 걸려 있다.** npu_matrix.tsv 89행 중 hal>0 인 행은 `vector_engine::normal::test_ve_stash_fp_fp` **단 하나**(hal=1, 67,289ms)다. 오염 기전은 폴루터 1개·피해자 1군집·1회 실행에서만 관측됐다. 이건 현상(phenomenon)이 아니라 버그(bug) 한 건이다. 'HAL 세션형 순서의존'을 새 결함 범주로 주장하려면 최소한 서로 다른 폴루터 여러 개가 필요한데, 데이터에는 하나도 더 없다.
- **간판 델타 10→33 의 원본 로그가 아카이브에 없다.** _evidence/logs/ 에는 격리 실행(ve_isolated.log)만 있고 **비격리 실행(10 통과/25 실패)의 로그가 존재하지 않는다.** `--test-threads=1` 무효 실험('11번째부터 무너졌다')도, '피해자 22개' 도 마찬가지로 로그가 없다. 재현성을 논제로 삼는 논문이 자기 헤드라인 수치를 재현할 수 없다 — 이 한 줄로 REP/ICPE 는 desk-reject 사유가 된다.
- **핵심 기여물인 오라클이 아예 실행 불가능하다.** classify_mismatch.py 는 `DETAIL = /home/jun/.claude/jobs/46bc5c7e/tmp/ex_logs/npu_matrix_detail` 를 하드코딩하는데 그 디렉터리는 이미 소멸했고(현재 해당 tmp 에는 `2nd/`, `a83.tgz` 만 존재) _evidence 에 복사되지 않았다. npu_matrix.sh 도 같은 소멸 경로를 cd 한다. 따라서 ULP_ROUNDING / REAL_CORRUPTION / REAL_MISMATCH 판정, '조용한 오배치 2건', '512개 중 489개 비트 동일·최대 ULP 1' 같은 값 검증 주장은 **아카이브만으로는 단 한 건도 재검증할 수 없다.** 아티팩트 배지 심사 불통과가 아니라, 논문 본문 주장의 1차 근거가 없는 상태다.
- **아카이브의 자체 검증 레시피가 이 논문이 고발하는 바로 그 오류를 범한다.** _evidence/README.md 의 확인 명령은 `grep -c 'test result: ok' logs/ve_isolated.log` → "PASS=33" 을 출력하는데, 실제로 그 33행 중 1행은 `0 passed; 0 failed; 1 ignored` 인 no-op 이다(직접 확인). npu_matrix.tsv 는 이를 OTHER 로 올바르게 분류해 PASS=32 다. '집계 규칙이 수치를 바꾼다'는 명제를 증명하는 저자 자신의 아티팩트가 그 명제의 피해자다. 리뷰어는 이를 아이러니가 아니라 부주의로 읽는다.
- **'커널'의 분모가 논문 안에서 셋이고, 문서가 대놓고 화해를 거부한다.** 200(소스 텍스트 추출) / 207(툴 자체 집계, 제네릭 단형화 포함) / 143(게이팅 후 실기 빌드) 가 병존하고 09-도구와-계층.md:165-166 은 "서로 다른 정당한 수다 — **맞추려 하지 말 것**" 이라고 적는다. 각각은 타당하지만, 하필 '보고 규약을 제안하겠다'는 논문이 자기 보고 규약을 세 번 바꾼다. §5 의 체크리스트가 이 논문 자신에게 적용되지 않는다.
- **측정축이 통째로 비어 있다.** §4 의 사이클(DmaEngine 96.5%)은 _evidence/README.md 가 스스로 "**컴파일러 스케줄 모델 예측이며 실측 벽시계가 아니다**" 라고 명기한 AOT 산출물이다. 실기 벽시계는 테스트당 총 3.7~8.2초로 cargo 진입비용에 묻혔고, 하드웨어 성능 카운터·전력·대역폭 실측은 0건이다. ISPASS/IISWC 는 '측정 논문'을 표방하면서 측정한 하드웨어 양이 하나도 없는 원고를 통과시키지 않는다.
- **베이스라인이 저자가 직접 한 번 잘못 돌린 실행뿐이다.** L0 는 published 하네스가 아니라 self-constructed straw man 이다. 정당한 베이스라인은 KernelBench / MultiKernelBench(arXiv 2507.17773) 같은 기존 공개 하네스를 같은 커널 집합에 적용해 델타를 재는 것인데, 그 비교가 없다. 특히 MultiKernelBench 는 이미 GPU/NPU/TPU 285 태스크에 대해 device management·compilation·execution·timing·**resource cleanup** 을 백엔드 추상화 5메서드로 규정하고 compilation success → correctness → performance 를 단계 지표로 쓴다. 즉 '3단 감쇠'와 '2번째 툴체인 축' 둘 다 이미 존재하는 인프라가 대신 하고 있다.
- **게이팅이 만드는 선택 편향이 결과 수치에 그대로 남아 있다.** 63개 커널을 제외한 뒤 얻은 80/89(93.3%)를 '실기 동작률'로 보고하는데, 이는 H1~H4 와 같은 등급의 5번째 위험(H5: survivorship)이며 논문이 이를 자기 지표에 적용하지 않았다. 위험 분류체계를 제안하는 논문이 자기 대표 수치에서 그 분류체계에 걸린다.
- **H1·H3 은 연구 결과가 아니라 벤더 도구의 결함 보고다.** H1(`--backend npu` 가 패키지 전체 `#[device]` 를 낮춘다)은 릴리스 노트 한 줄로 사라질 툴 제약이고, H3(`compile <FILTER>` 부분문자열 매칭)은 200건 중 1건(0.5%) 영향에 정확 매칭으로 즉시 소멸한다. 논문 기여 5개 중 2.5개가 다음 버전에서 워크어라운드 기록이 된다.
- **신선도가 근거 전체를 인질로 잡는다.** furiosa-opt v0.4 사전출시 스냅샷 단일 버전, 단일 벤더, 단일 칩(npu0 하드타깃), 단일 크레이트(벤더 자체 예제), 1회 실행. 워크로드조차 실제 워크로드가 아니라 벤더 테스트 스위트다. 게재 시점에 63 에러·13 ICE·로더 범위초과 버그가 고쳐져 있으면 남는 건 방법론 문장뿐인데, 그 문장은 위 1·2번에서 이미 선점됐다.

**이미 출판되어 신규성이 없는 부분**

- Harness-Bench: Measuring Harness Effects across Models in Realistic Agent Workflows — arXiv 2605.27922 (2026). **thesis 직격.** {대상}×{하네스} 요인설계로 '하네스가 보고 수치를 바꾼다'를 정량화하고, run-level pass rate 는 동일한데 majority-pass/all-replicate 집계가 갈리는 사례까지 제시. 이 제안의 명제·설계·결론이 도메인만 바꾼 재현이다.
- Rethinking the Evaluation of Harness Evolution for Agents — arXiv 2607.12227 (2026). 하네스 진화가 평가 결과에 미치는 영향의 방법론적 재검토. 위와 함께 '하네스 효과'가 이미 독립 연구 주제임을 보인다.
- KernelBench: Can LLMs Write Efficient GPU Kernels? — Ouyang et al., 2025 (ICML). **H2 완화책 선점.** 각 솔루션을 전용 서브프로세스에서 컴파일·실행해 evaluator 상태를 격리, 한 솔루션이 뒤 솔루션에 영향 주지 못하게 한다고 설계에 명시. 제안의 '재사용 가능한 하네스' 기여가 2025년 표준 설계에 못 미친다.
- MultiKernelBench: A Multi-Platform Benchmark for Kernel Generation — arXiv 2507.17773 (2025). **contribution 3·need_experiments 1 동시 선점.** GPU(CUDA/Triton)·NPU(AscendC/TileLang)·TPU(Pallas)·Intel(SYCL) 285 태스크, 14 카테고리. 백엔드 추상화 5메서드가 device management / compilation / execution / timing / **resource cleanup** 을 규정하고, compilation success → correctness → performance 를 단계 지표로 집계. '컴파일→실행→정확 3단'과 '툴체인 ≥2 일반화'가 이미 구현·배포돼 있다.
- CANN Bench: Benchmarking Agent Generated Kernels against Real NPU and Algorithmic Limits — arXiv 2607.20518 (2026). 실기 Ascend NPU 에서 컴파일·기능정확성·성능 3축 + 재현 키트 배포. 제안서 스스로 인정.
- iDFlakies: A Framework for Detecting and Partially Classifying Flaky Tests — ICST 2019. victim/polluter 형식화, 422개 중 50.5% order-dependent. H2 를 결함 범주로서 선점.
- Systematically Producing Test Orders to Detect Order-Dependent Flaky Tests (Tuscan) — ISSTA 2023, DOI 10.1145/3597926.3598083. 47개 스위트 289개 OD 테스트에서 평균 104.7 순서로 97.2% 검출. 제안의 '순서 무작위화 반복' 설계가 여기 종속.
- Repairing Order-Dependent Flaky Tests via Test Generation — ICSE 2022. OD 결함의 자동 수리까지 진행됨. 즉 이 분야는 '발견'이 아니라 '수리' 단계.
- Detecting and Evaluating Order-Dependent Flaky Tests in JavaScript — arXiv 2501.12680 (2025). victim/brittle 분류를 다른 런타임으로 이식한 사례 — 즉 '매체만 바꾼 OD 연구'가 이미 별도 논문으로 유통 중이고, 그 델타 크기가 이 제안의 델타(힙→HAL)와 동급이다. 신규성 방어선이 무너지는 지점.
- The Correctness Illusion in LLM-Generated GPU Kernels — arXiv 2606.20128 (2026) / FASTKERNELS: Benchmarking GPU Kernel Generation in Production — arXiv 2605.23215 (2026). H4(거짓 PASS·허용오차)를 ULP≤2 하드코딩보다 원칙적으로 처리. 제안서 인정.
- Producing Wrong Data Without Doing Anything Obviously Wrong! — ASPLOS 2009 / Benchmarking Crimes — arXiv 1801.02381 (2018) / SIGPLAN Empirical Evaluation Checklist (2018–). 측정 편향·체크리스트 프레이밍 선점. 제안서 인정.
- Questionable practices in machine learning — arXiv 2407.12220. ML 평가 악습 목록화. '체크리스트 제안' 기여의 또 다른 선례.
- Hardware Cost Evaluation in Systems Security — ACM REP 2025, DOI 10.1145/3736731.3746155. **제안이 겨냥한 ACM REP 에서 이미 '평가 관행 감사 + 체크리스트' 형식의 논문이 게재됐다.** 이 형식이 REP 에서 통과하려면 무엇이 필요한지의 기준선이자, 같은 자리를 이미 차지한 경쟁 논문.

**근거 없이 단정한 문장 (수정 필요)**

- '격리 안 하면 vector_engine 10 통과 / 25 실패' — **_evidence 에 이 실행의 로그가 전혀 없다.** logs/ 는 npu_matrix.tsv, perkernel_matrix_fixed.txt, sched_summary.json, ve_isolated.log 4개뿐이고 비격리 실행 산출물은 0개다. 이 논문의 대표 델타(3.2~3.3배) 전체가 서술로만 존재한다.
- '`--test-threads=1` 로 직렬화해도 소용없다 — 단일 스레드에서도 앞 10개만 통과하고 11번째부터 무너졌다'(13번 §3.1) — 로그 없음. H2 를 '동시성이 아니라 프로세스 수명 내 장치상태 오염'으로 분리하는 **유일한** 실험인데 근거가 아카이브에 없다.
- '멀쩡한 커널 22개가 실패로 집계됨' — 25−3 의 산술 유도일 뿐, 어느 22개인지 식별한 데이터가 없다. iDFlakies 류 문헌이 요구하는 victim 목록이 부재.
- 제안서 evaluation (a) '보고 가능한 테스트 수 Δ (0 → 89)' — 문서와 불일치. OPTIMIZATION-SURFACE.md:260 은 게이팅 효과를 '**21개 → 89개**'로 적고, '테스트 0개'는 게이팅 전 빌드 실패 상태를 가리킨다. 서로 다른 세 기준선(0 / 21 / 89)이 한 지표에 뒤섞였다.
- 제안서가 인용한 '0→58 델타' — 문서에 그런 수치가 없다. 12-예제-전수실행.md:633 의 58 은 '7장 서브셋 20개 → 58개'로 **커널 서브셋 폭**이지 테스트 수가 아니다. 제안서가 근거 문서를 오독했다.
- '격리 시 33 통과' (README.md:31, 08-커널-예제.md:437, 13번 §3, GROUND_TRUTH_BRIEF B-1 전부 33) — **실측은 32.** ve_isolated.log 36행 중 `test result: ok` 33행이지만 그중 `vector_engine::normal::test_ve_elementwise_vrf` 는 `0 passed; 0 failed; 1 ignored`. npu_matrix.tsv 도 vector_engine PASS=32 / FAIL=3 / OTHER=1 로 32를 지지한다. 문서 4곳이 전부 1을 틀렸다.
- 'hang 커널 단독 실행 = 60초 타임아웃 실패'(13번 §3.2 표) — tsv 실측은 `FAIL, 67,289ms, hal=1` 이다. npu_matrix.sh 의 TIMEOUT 판정은 rc=124(=timeout 150) 인데 이 행은 TIMEOUT 이 아니라 FAIL 이다. 즉 '60초 타임아웃'은 이 매트릭스 실행의 결과가 아닌 다른 실행의 서술이며, 그 로그도 없다.
- 'ABORT 3건의 요구/실제 비율 1.496·1.496·1.493' — tsv 에는 `load=1` 과 패닉 위치 `device-runtime-c/src/kernel.rs:137` 만 있다. 비율의 출처인 per-test detail 로그(npu_matrix_detail/)가 아카이브에 없어 3건 중 1건(1.493, 03-텐서-이동.md:261)만 서술로 확인된다.
- '조용한 데이터 오배치 2건' — tsv 근거는 `mismatch=1` 플래그 2개(broadcast, tile)뿐. 이를 ULP 라운딩이 아닌 REAL_CORRUPTION 으로 판정한 classify_mismatch.py 의 입력이 소멸해 **재판정 불가**. 제안서 스스로 인정하듯 tile 의 '32 elements→32 bytes' 는 32→8=÷4=sizeof(f32) 라는 산술 정황일 뿐 런타임 소스 확증이 없다.
- '1 ULP 1건(512개 중 489개 비트 동일, 최대 ULP 1, 최대 상대오차 1.742e-07)'(_GROUND_TRUTH.md:168) — 원본 detail 로그 부재로 재계산 불가.
- 'cargo 진입비용 최소 3.5초'(13번:524) — tsv 최소 PASS 소요가 3,710ms 이므로 상한으로서는 정합하나, **빈 테스트 대조 실행이 없어** 3.5초가 cargo 오버헤드인지 커널 실행 포함인지 분리되지 않았다. 프로세스 격리 비용(≈5분)의 정당화 논거가 여기 걸려 있다.
- 제안서 내부 불일치: thesis 는 '10↔32', risks 는 '3.2배', 원 문서는 '10→33, 3.3배'. 같은 문단 안에서 배율이 세 값이다.
- '커널 hang → HAL 세션 오염'의 기전 자체 — 관측된 것은 '한 프로세스 안에서 -110 이후 후속 실패' 라는 상관뿐이고, HAL/드라이버의 어느 상태가 오염됐는지 소스·드라이버 수준 확증이 없다. need_experiments 도 이를 미해결로 남긴다.

**조사자 스스로 적은 위험**

- **단일 벤더·단일 칩·단일 크레이트.** furiosa-opt v0.4 예제 하나가 전부다. 리뷰어는 '방법론 논문이 아니라 벤더 버그 리포트' 로 읽는다. 지금 상태로는 이 반론을 막을 수 없다
- **H2 는 GPU 에서 이미 상식.** CUDA sticky error / OOM 후 컨텍스트 오염 때문에 커널 생성 벤치들은 이미 서브프로세스 격리를 관행으로 쓴다(검색에서 서브프로세스 격리+타임아웃이 표준 설계 패턴으로 서술됨). 새로운 건 3.2배라는 숫자뿐이고 그건 n=1 이다
- **H2 는 SE 문헌이 선점.** iDFlakies(ICST'19)가 victim/polluter 로 형식화하고 422개 데이터셋의 50.5%가 OD 임을 보였다. 매체가 힙 대신 HAL 세션이라는 것만 다르다
- **H4 는 2026년 커널 벤치 논문들이 더 체계적으로 다룸.** FastKernels 는 reference-vs-reference 비결정성으로 dtype 별 허용오차를 보정하고, Correctness Illusion 은 손으로 고른 atol/rtol 이 실측 오차 봉투보다 1~3자릿수 느슨하다고 비판한다. 우리 쪽은 ULP≤2 하드코딩이라 오히려 더 원시적이다
- **H3 은 명백한 도구 사용 오류.** 200개 중 1건(0.5%) 오염이고 정확 매칭으로 즉시 사라진다. 논문 소재로는 각주 수준
- **신선도 위험.** 사전출시 툴체인은 몇 달이면 고쳐진다. 게재 시점에 63 에러·13 ICE·2 조용한 오배치가 전부 없어져 있으면 근거가 통째로 휘발한다. 방법론으로 프레이밍하면 완화되지만, 근거가 그 버전 하나에만 묶여 있으면 완화되지 않는다
- **자기모순 리스크(실측 확인됨).** 준 수치 '격리 시 33 통과' 는 ve_isolated.log 의 `test result: ok` 33행 중 1행이 `0 passed; 0 failed; 1 ignored` no-op 이라 실제 통과는 32다. npu_matrix.tsv 도 PASS 32 / OTHER 1 로 32를 지지한다. 집계 규칙이 결과를 바꾼다고 주장하는 논문이 자기 집계에서 1을 틀리면 리뷰에서 그대로 반사된다
- **재현성 논문인데 근거 보존이 불완전.** 0/58/10-25 델타의 원본 로그가 _evidence 에 없고, tools/ 스크립트는 사라진 job 임시경로가 하드코딩돼 그대로는 안 돈다(README 가 스스로 인정). 아티팩트 배지 심사를 통과 못 한다
- **성능 주장이 없다.** §4 의 사이클은 컴파일러 스케줄 모델 예측이지 하드웨어 카운터가 아니고, 벽시계는 cargo 오버헤드에 묻혔다. ISPASS/IISWC 계열이 기대하는 실측 축이 비어 있다
- **기여의 절반이 '레시피'다.** 게이팅·서브셋은 훌륭한 엔지니어링 노트지만, 벤더가 다음 릴리스에서 per-kernel 빌드를 지원하면 논문 기여가 아니라 워크어라운드 기록이 된다

### 5.9 살리는 길 — 무엇을 바꿔야 하는가

독립 방법론 논문은 포기하라. 근거의 품질(직접 재계산으로 tsv 89행·perkernel 200행·sched 130커널이 모두 정확히 일치)은 좋지만, 그 근거가 지탱하는 것은 '방법론'이 아니라 '한 툴체인의 실측 특성화'다. 세 갈래만 현실적이다.

**경로 A (권장, 2~4주 — 지금 근거로 충족).** 「RNGD vISA 툴체인 실기 특성화」를 본 논문으로 세우고, H1~H4 를 §3 Methodology + Threats to Validity 로 흡수한다. 본문 주장은 실측 사실(200→137 컴파일, 89개 격리 실행 80 통과, ABORT 3건이 전부 커널 로드 단계, 사이클 96.5% DMA — 단 스케줄 모델 예측임을 본문에서 명기)에 두고, 하네스 이야기는 '이 수치를 얻으려면 이렇게 해야 했다'로 배치. 이 형태면 위 fatal 대부분이 threats 절로 강등된다. IISWC/워크샵.

**경로 B (독립 논문을 굳이 밀 경우, 6~9개월 — 그래도 viable 이 상한).** 다음이 전부 충족돼야 리뷰 테이블에 오른다.
1. **Harness-Bench(2605.27922)·MultiKernelBench(2507.17773)·KernelBench 와의 델타를 §1 에서 명시.** 특히 '서브프로세스 격리는 이미 표준인데 왜 논문인가'에 답해야 한다. 유일하게 방어 가능한 답: *격리가 표준인 이유는 GPU 경험칙이고, 그 비용/최소 격리 단위를 실기에서 잰 사람이 없다.* → 논문을 '격리의 정당화'가 아니라 **'격리 비용의 최소화'**로 재정의하라(디바이스 리셋 vs 컨텍스트 재생성 vs 프로세스 재시작의 회복 성공률·비용 곡선). 이건 아직 아무도 안 했고, 3.7초×89 라는 실제 비용 데이터가 이미 있다.
2. **n 을 늘려라.** 현재 폴루터가 89행 중 1건이다. 인위적 폴루터 주입(고의 hang 커널을 여러 종류로 합성)으로 폴루터 클래스별 오염 반경을 측정하면 n=1 문제가 사라지고 인과도 통제된다. 이게 가장 싸고 효과 큰 실험이다.
3. **툴체인 ≥2.** AMD XDNA/NPUEval 보다 **MultiKernelBench 백엔드 추가**가 훨씬 싸다 — 이미 GPU/NPU/TPU 추상화가 있으니 거기에 furiosa 백엔드를 붙이고 같은 사다리를 돌리면 '단일 벤더' 반론과 '베이스라인 없음' 반론이 동시에 죽는다. 임계경로를 2~3개월에서 수 주로 줄이는 유일한 수.
4. **아카이브 복구 없이는 아무것도 하지 마라.** 비격리 실행(10/25), `--test-threads=1` 실행, npu_matrix_detail/ 전체를 재실행해 _evidence 에 넣고, npu_matrix.sh·classify_mismatch.py 의 하드코딩 경로를 상대경로화. 현재 상태로는 아티팩트 심사 전에 자기 주장을 재현할 수 없다.
5. **자기 집계부터 고쳐라.** 33→32 정정(README·08·13·BRIEF 4곳), 200/207/143 의 정의를 표 하나로 고정하고 '맞추지 말 것'을 '왜 다른지'로 교체. 게이팅 편향을 H5 로 승격해 80/89 옆에 항상 '63 커널 제외' 를 병기. ULP≤2·불일치율 0.9 임계 스윕.
6. **문헌 감사 ≥40편.** Mytkowicz 의 '133편 중 0편'에 대응하는 카운터가 없으면 '아무도 안 틀렸는데 혼자 조심한 이야기'다. 단 KernelBench·MultiKernelBench 가 이미 방어하고 있으므로, 감사 결과가 '대부분 방어함'으로 나올 위험이 크다 — 그 경우 논문을 접는 판단 근거로 쓰라.
7. **측정축을 하나라도 채워라.** 하드웨어 카운터가 없으면 최소한 스케줄 모델 예측 vs 실기 벽시계의 **상관/괴리**를 재라. 그 자체가 '예측을 실측으로 보고하는 관행'에 대한 정당한 H5 가 된다.

**경로 C (프레이밍 전환).** SE 쪽으로 돌려 「Device-State Order Dependence: OD flakiness beyond heap and filesystem」로 ICST/ISSTA 산업트랙. iDFlakies/Tuscan 과 정면 비교가 강제되지만 기여가 가장 선명하다. 전제는 역시 B-2(폴루터 다수 확보)와 B-4(로그 복구). 매체가 다르다는 것만으로는 부족하고, **'HAL 세션 오염은 기존 OD 검출·수리 기법이 원리적으로 못 잡는다'**를 보여야 한다(테스트 재생성으로 수리 불가, cleanup 훅으로 복구 불가 등). 그걸 보이면 진짜 논문이 되고, 못 보이면 경로 A 로 돌아가라.

### 5.10 후보 학회와 소요 기간

- IISWC — 워크로드/툴체인 실측 특성화. 12·13번 문서의 실측(200→137 컴파일, 89 격리 매트릭스, DMA 96.5% 지배)을 본문으로 하고 방법론을 §3 으로 붙이는 형태가 가장 현실적인 착지점
- ISPASS — 측정·분석 전문. 다만 실측 벽시계·하드웨어 카운터 축이 비어 있어 지금 상태로는 약함
- ICPE (측정/재현성 트랙) 또는 ACM REP — 방법론 단독으로 밀 경우의 정공법. 대신 툴체인 ≥2개 + 반복 통계가 전제
- LATTE / C4ML / CGO 워크샵 — 컴파일러 툴체인 평가 프로토콜로 프레이밍할 때. 초기 노출용
- ICST / ISSTA 산업트랙 — '장치상태형 order-dependent 결함' 으로 프레이밍을 SE 쪽으로 돌릴 경우. iDFlakies 계열과 정면 비교가 강제되지만 기여가 가장 선명해지는 경로
- MLSys / ASPLOS 워크샵 — 사전출시 가속기 평가 경험보고(experience report) 형식

**소요 기간 추정**: 방법론 단독 논문으로 밀 경우 6~9개월. 임계경로는 2번째 툴체인 접근 확보(AMD XDNA 또는 Ascend, 2~3개월)와 문헌 감사 40편 코딩(1.5개월)이며, 반복 실행 통계·오라클 민감도·로그 재아카이빙은 병렬로 1개월. 반면 다른 논문(예: 'RNGD vISA 툴체인 실측 특성화')의 §방법론 + 아티팩트로 넣을 경우 2~4주면 충분하고, 현재 근거만으로도 충족된다 — 나는 이쪽을 권한다.

**신규성 확신도**: 낮음~중간. 네 위험의 개별 기전은 전부 published 선행연구에 1:1 대응된다 — (1)은 서베이 수준에서 '연산자 커버리지/폴백 보고 필요' 로 언급되고, (2)는 iDFlakies(ICST 2019)가 형식화했으며, (3)은 도구 사용 오류이고, (4)는 FastKernels·Correctness Illusion(2026)이 더 원칙적으로 다룬다. 8회 검색에서 '이 넷을 사전출시 NPU 툴체인 맥락으로 묶은' 논문은 찾지 못했으므로 조합·맥락의 신규성은 남아 있으나, 조합만으로 top-tier 기준을 넘기는 어렵다고 본다. 조각별로 보면 신규성이 가장 높은 것은 (2)의 변종 — 커널 hang 이 HAL 세션을 오염시켜 후속 커널을 전부 -110 으로 만드는 '장치상태형 순서의존' 을 정량화(10→32, 3.2배)한 논문을 찾지 못했다 — 과 '컴파일 성공 / 로드 성공 / 값 정확' 3단 감쇠의 실측 분리다. 다만 전자는 GPU 실무에서 sticky error 로 널리 알려진 현상이라 '문헌에 없다 ≠ 새롭다' 를 조심해야 한다. 검색 실패 가능성도 명시해 둔다: 벤더 내부 QA 문서나 비공개 엔지니어링 보고서는 검색 범위 밖이며, 이런 종류의 지식은 논문보다 그쪽에 축적돼 있을 공산이 크다.

### 5.11 검색으로 확인한 관련 연구

| 제목 | 학회·연도 | 이 주제와의 관계 |
|---|---|---|
| Producing Wrong Data Without Doing Anything Obviously Wrong! | ASPLOS 2009 | 기반. 실험 셋업의 무해해 보이는 변경이 결론을 뒤집는다는 원형 논증. 특히 ASPLOS/PACT/PLDI/CGO 133편 서베이에서 측정 편향을 고려한 논문이 0편임을 보인 구조가, 이 주제가 반드시 갖춰야 할 '남들이 실제로 틀리고 있다'의 표준이다 |
| iDFlakies: A Framework for Detecting and Partially Classifying Flaky Tests | ICST 2019 (pp.312-322) | 겹침. 위험 (2)를 정면으로 선점한다. victim/polluter 형식화 + 422개 flaky 데이터셋 중 50.5%가 order-dependent. '격리하면 통과, 순서대로 돌리면 실패'는 이 문헌의 정의 그 자체다. 우리 쪽 차별점은 공유상태가 힙/파일시스템이 아니라 NPU HAL 세션이라는 점뿐 |
| Systematically Producing Test Orders to Detect Order-Dependent Flaky Tests (Tuscan) | 2023 (Li et al., GMU 공개본) | 인접. 순서의존 결함을 체계적으로 드러내는 테스트 순서 생성. need_experiments 의 '순서 무작위화 반복 실행' 설계를 여기서 빌려야 한다 |
| The Correctness Illusion in LLM-Generated GPU Kernels | arXiv 2606.20128, 2026-07 (Dipankar Sarkar) | 겹침. 위험 (4)의 거짓 PASS 쪽을 선점. 고정 shape·소표본 allclose 검사가 체계적으로 낙관 편향되며 손으로 고른 atol/rtol 이 실측 오차 봉투보다 1~3자릿수 느슨하다고 주장. '값 검증 없이 통과로 세면 놓친다'는 우리 주장의 강화판 |
| FastKernels: Benchmarking GPU Kernel Generation in Production | arXiv 2605.23215, 2026 | 겹침. reference-vs-reference 수치 비결정성으로부터 dtype 별 허용오차를 보정하고, 기준을 고의로 틀린 베이스라인으로 교체해 품질 절벽을 관측. 우리의 ULP≤2 하드코딩보다 원칙적인 오라클 보정 방법론 |
| CANN Bench: Benchmarking Agent Generated Kernels against Real NPU and Algorithmic Limits | arXiv 2607.20518, 2026-07 | 겹침. 컴파일·기능정확성·성능을 독립 3축으로 다루고 재현 키트(스펙+골든구현+공개 테스트+성능 베이스라인)를 배포. 실기 Ascend NPU 대상. 다만 '커널 로드 단계' 를 별도 축으로 분리하지는 않아, 우리 (3단 감쇠)의 여지는 남는다 |
| NPUEval: Optimizing NPU Kernels with LLMs and Open Source Compilers | arXiv 2507.14403, 2025 | 인접. AMD NPU 대상 102개 연산자를 실기에서 기능정확성 + 벡터화 효율로 평가. 2번째 툴체인 축을 붙일 때 가장 현실적인 대상 |
| A Comprehensive Study of Deep Learning Compiler Bugs | ESEC/FSE 2021 | 인접. TVM/Glow/nGraph 603개 버그를 근본원인·증상·컴파일 단계별로 분류. 우리 §7의 63개 실패 분류(REAL_LOWERING_GAP 24 / INTENTIONAL 23 / ICE 13 …)는 이 방법론의 소규모 반복이며, 그 사실이 신규성을 깎는다 |
| Silent bugs in deep learning frameworks: an empirical study of Keras and TensorFlow | Empirical Software Engineering (EMSE) 2023 | 겹침. '크래시 없이 조용히 틀린 결과' 를 결함 범주로 확립. 위험 (4)의 조용한 데이터 오배치 2건이 새 범주가 아님을 보여준다 |
| Demystifying the Silence of Correctness Bugs in PyTorch Compiler | arXiv 2604.08720, 2026 | 인접. 컴파일러 계층의 조용한 정확성 버그를 다룬 최신 연구. 우리의 broadcast 미기록·오프셋 단위 혼동과 같은 계열 |
| Benchmarking Crimes: An Emerging Threat in Systems Security | arXiv 1801.02381, 2018 (van der Kouwe et al.) | 기반. 22개 '벤치마킹 범죄' 목록화 + tier-1 논문이 평균 5개를 범한다는 감사. 우리가 만들려는 'H1~H4 체크리스트' 의 직접 선례이자, 체크리스트만으로는 부족하고 감사 데이터가 있어야 함을 보여주는 사례 |
| SIGPLAN Empirical Evaluation Guidelines / A Checklist Manifesto for Empirical Evaluation | SIGPLAN 온라인 가이드라인, 2018– (Checklist Manifesto: Berger·Blackburn·Hauswirth·Hicks, 2019) | 기반. 커뮤니티 표준 체크리스트. 새 체크리스트를 제안하려면 이것과의 델타를 명시해야 한다 |
| Gernot's List of Systems Benchmarking Crimes | 온라인 리소스(비심사), 2020– (Heiser) | 기반. 측정 실무 규범의 사실상 표준 목록. 논문이 아니므로 인용 가중치는 낮지만, '이런 목록은 이미 있다' 는 반론의 근거가 된다 |
| MLPerf Inference Benchmark | ISCA 2020 | 인접. ML 시스템 벤치의 재현성·아키텍처 중립성 요구를 정립. 사용자가 지목한 'MLPerf 논쟁' 축의 1차 문헌이지만, 검색으로는 이를 정면 비판한 별도 논문을 찾지 못했다 |
| Hardware Acceleration for Neural Networks: A Comprehensive Survey | arXiv 2512.23914, 2025 | 인접. 가속기 결과가 소프트웨어 버전·컴파일 설정·측정 방법에 민감하며 재현 가능한 비교에는 이들 전부의 명시 보고가 필요하다고 정리. 엣지 NPU 에서 연산자 커버리지와 폴백 동작을 반드시 보고해야 한다는 권고는 우리 H1 과 같은 방향인데, 이미 서베이 수준에서 언급된다는 뜻이기도 하다 |


---

## 주제 6. 정렬·레인 제약을 타입 수준으로 끌어올리기

> **판정: `weak` → 적대적 심사 후 `not-a-paper`**  (논문 아님 — 현재 형태로는 엔지니어링 노트)
> 조사자가 찾은 선행연구 8건 · 심사자가 제기한 치명적 반론 11건 ·
> "이미 출판됨" 지적 9건 · 근거 없이 단정한 문장 10건

### 6.1 초록 — 한 문장 주장

vISA의 정렬·레인·용량 제약 중 편집 시점 타입으로 올릴 수 있는 것은 "커널 경계 HBM 매핑 타입에 부분 충전 패딩 그룹(k # N, k<N)이 노출되는가" 같은 지역 구문 술어뿐이고, 레인 제약은 매핑 전체에 걸친 관계형 성질이며 실기 무증상 오동작은 원리적으로 타입 밖이다 — 따라서 이 주제는 독립 논문이 아니라 "타입이 어디까지 사주는가"를 실측한 특성화 논문의 한 절이다.

### 6.2 문제의식과 선행연구의 빈틈

검색으로 확인한 공백은 좁다. (1) Dahlia(PLDI'20)는 HLS/FPGA의 뱅크·포트 자원 경합을 time-sensitive affine 타입으로 막지만, 온칩 레지스터파일의 레인 부분 활성이나 DRAM 경계 패딩 노출은 대상이 아니다. (2) TileLang(2025)은 타일 레이아웃·스레드 매핑을 "타입 속성 + 제약 해"로 다루지만 목적이 성능 자동화이고 GPU 대상이며, "표현 불가 제약"을 분류하지 않는다. (3) shape checking 계열(Gradual Tensor Shape Checking, Staged Shape-Dependent Types)은 차원 일치만 보고 정렬·패딩·레지스터파일 용량은 범위 밖이다. (4) Exo(PLDI'22)는 스케줄 재작성의 등가성·메모리 안전을 보장하지만 "벤더 백엔드 낮추기가 실패할 것"을 편집 시점에 예측하는 문제는 다루지 않는다. (5) AscendCraft(2026)는 NPU의 32B UB 정렬·크기 granularity 제약을 명시적으로 인정하면서 타입이 아니라 LLM 패스 + 컴파일러 피드백 루프로 처리한다 — 정확히 그 자리에 타입 기반 해법이 비어 있다. 못 찾은 것: "프론트엔드 타입 통과 + 에뮬레이션 통과 + 백엔드 낮추기 실패/무증상 오동작"이라는 간극을 실기 코퍼스로 정량화한 연구는 검색에서 하나도 안 나왔다. 다만 이 공백의 상당 부분이 벤더 컴파일러 결함이라 "공백"이 곧 "연구 가치"는 아니다.

### 6.3 제안한 기여

- 제약 3계층 분류표: vISA의 정렬·레인·용량 제약을 (a) 지역 구문 술어로 타입화 가능(HBM 경계 부분 충전 노출, TRF 용량 선형 부등식), (b) 매핑 전체에 걸친 관계형 리파인먼트가 필요(레인 부분 활성), (c) 원리적 타입 불가(백엔드 크기 계산 결함, 실기 무증상 오동작)로 나눈 실증 분류
- 결정적 반례쌍: Lane 타입 파라미터가 문자 그대로 동일한(TrfTensor<..., m![1], ...>) 두 커널이 정반대 결과를 낸다 — mnist::fc1_matmul/fc2_matmul 은 실기 통과, contract_outer_assertions::lane_size::valid_size_1 은 컴파일 실패. 레인 제약이 지역적으로 타입화 불가하고 커널 경계 HBM 타입과의 관계로만 결정됨을 소스 수준에서 증명
- 꼬리 패딩 누락의 닫힌 식: intra_chip_size = (N-1)·P·S + k·S (N=그룹수, P=패딩폭, S=원소바이트, k=유효폭) 로 백엔드 DRAM 크기 오산을 특징짓고, 이를 const generic 술어로 인코딩한 편집 시점 검사기 구현
- 코퍼스 위 정량 평가: 벤더 예제 200 커널 컴파일 매트릭스 + 실기 89 테스트에 대한 검출률/오탐률/진단 지연(cargo check vs --backend npu 왕복) 측정
- 부정 결과: 실기에서 가장 위험한 3종 결함(브로드캐스트 HBM→HBM DMA 무기록, commit_view 오프셋 오착지, chip_shuffle 커널 로드 abort)은 타입 강화로 하나도 줄지 않음 — 타입 규율의 상한을 실측으로 제시

### 6.4 방법

기존 프론트엔드가 이미 타입 수준이라는 점(m![] 매핑 타입 + CanApplyXxx 마커 트레이트 typestate)에서 출발해, 그 위에 얇은 리파인먼트 레이어를 얹는다. ① m![] 매핑 타입에서 (그룹수 N, 유효폭 k, 패딩폭 P, 원소바이트 S)를 뽑는 연관 상수(associated const)를 매크로 확장 시점에 생성. ② HbmTensor/DmTensor 의 경계 생성자(to_hbm/to_hbm_view/#[device] 시그니처)에 `where` 바운드 또는 `const { assert!(...) }` 로 "부분 충전 그룹 노출 금지" 술어를 건다. ③ TRF 용량은 Σ(lane 데이터) ≤ 65536 / 32768(TrfAddress 별)의 선형 부등식이라 const 산술로 직접 표현 — 현재 mir 단계에서 나는 진단을 cargo check 로 앞당긴다. ④ 관계형 제약(TRF Lane × 커밋 × HBM 경계)은 증거 토큰(witness type)을 contract_lane→commit→to_hbm 체인으로 흘려 지역화를 시도하고, 지역화가 실패하는 지점을 논문의 핵심 부정 결과로 명시. ⑤ Rust 표현력 한계는 세 인코딩(매크로 확장 시점 계산 / const assertion / typenum 트레이트)을 나란히 구현해 에러 메시지 품질·컴파일 시간으로 비교한다. ⑥ 표현 불가 집합은 왜 불가한지 논증한다 — 백엔드 상수 전파 이후에만 결정되는 것과, 애초에 타입이 아니라 런타임 결함인 것을 구분.

### 6.5 평가 설계

베이스라인 2개: (a) 현행 furiosa-opt 0.4.0 프론트엔드(typecheck + 에뮬레이션), (b) --backend npu 전체 컴파일. 대상 코퍼스: 벤더 예제 200 커널(전수 컴파일 매트릭스) + 실기 89 테스트(현재 80 통과). 지표 5종 — ① 편집 시점 검출률: 백엔드에서 실패하는 커널 중 cargo check 단계에서 잡히는 비율(현재 0%). ② 오탐률(false reject): 실기 통과 80 커널 중 잘못 거부하는 건수 — 0 이어야 하고, 여기가 MNIST 반례 때문에 어려운 지점. ③ 진단 지연: cargo check 초 vs --backend npu 컴파일 왕복 초. ④ 표현력 분류 커버리지: 관측된 제약 종수 중 (a)/(b)/(c) 각 계층 비율. ⑤ 에러 위치 정확도: 현행은 백엔드 내부 텐서 ID("incorrect buffer size at T7")를 주는 반면, 타입 수준은 사용자 소스의 어느 타입 파라미터인지 지목 가능 — 사용자 소스 라인 지목 성공률로 측정. ⑤가 실질적으로 가장 방어 가능한 지표다.

### 6.6 이미 확보된 근거

- TRF 용량 상한이 컴파일러가 뱉는 숫자로 확정: `mir: TRF data (524288 bytes = 8 lanes x 65536 bytes) exceeds register file capacity (65536 bytes)` (Full) 및 `(65536 bytes = 8 lanes x 8192 bytes) ... capacity (32768 bytes)` (Half) → Full 64 KB/slice = 8 KB/lane, Half 32 KB/slice = 4 KB/lane. 책의 원시 80 KB/slice(8 lane × 2 bank × 128 row × 320bit)와는 32/40 비율로 화해됨(80×32/40=64). 출처: /home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vISA/book_guide/04-연산-분배와-레지스터.md:269-285, book_ko/src/computing-tensors/register-files.md:106-119
- 부분 충전 레인 실패값 3점 확인: `incorrect buffer size at T7: buffer.size() (256) != num_chips * intra_chip_size (228 / 232 / 240)` — valid_size_{1,2,4}. 출처: 04-연산-분배와-레지스터.md:254-260, 03-텐서-이동.md:241
- 실패값의 산술 구조를 소스로 확정: contract_outer_assertions.rs 에서 A=8, 출력 원소는 i32(4 B), 출력 매핑은 m![A, k # 8] (k=1/2/4/8). 선언 크기 = 8행 × 8원소 × 4 B = 256 B. 백엔드 계산 = 7×32 + 4k = 228/232/240 → 마지막 그룹의 꼬리 패딩(32-4k B)만 누락. k=8 인 valid_size_8 은 누락분 0 이라 컴파일·실기 모두 통과. 출처: /home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vISA/reference/examples/furiosa-opt-examples/src/contract_outer_assertions.rs:1-115
- Lane<8 금지 아님이 소스로 확인, 그리고 프롬프트보다 강함: mnist::fc1_matmul 은 `TrfTensor<bf16, Chip, Cluster, m![H], m![1], m![X]>`, fc2_matmul 은 `TrfTensor<bf16, ..., m![C, 1 # 16], m![1], m![H]>` — Lane = m![1]. mnist_tests 실기 1/1 통과(이미지 10장 전부 정답). 반면 lane_size::valid_size_1 도 Lane = m![1] 인데 컴파일 실패. 출처: reference/examples/furiosa-opt-examples/src/mnist/mod.rs:17,96 및 book_guide/13-NPU-실기-매트릭스.md:40
- 두 커널의 유일한 구조적 차이가 HBM 경계 타입임을 확인: MNIST 는 부분 충전 패딩(m![1 # 4], m![1 # 8], m![1 # 16])을 DM/TU 내부에만 두고 HBM 반환 타입은 `HbmTensor<bf16, Chip, m![H]>` / `m![C]` (패딩 없음, mnist/mod.rs:86,158). lane_size 는 `output: &mut HbmTensor<i32, Chip, m![A, 1 # 8]>` 로 부분 충전을 커널 경계에 노출하고 to_hbm_view 로 쓴다 → 프롬프트의 '부분 충전 그룹을 HBM 출력 타입에 노출하는 패턴이 막힌다'가 소스 수준에서 뒷받침됨
- 프론트엔드가 이미 타입 수준 강제를 한다: 파이프라인 단계가 typestate(CanApplyXxx 마커 트레이트 그래프)로 컴파일 타임에 못박히고, fetch_mask→fetch_cast 역순은 타입이 막는다. 출처: 04-연산-분배와-레지스터.md:20-21,42-43
- 프론트엔드 검증의 비대칭: to_trf 는 `verify_to_trf::<D, Lane, Time, Packet, Element>(&address)` 를 호출하지만 to_vrf/to_vrf_at 은 verify 호출이 아예 없다 → VRF 쪽은 백엔드 StoVrf 낮추기에서 `kernel-declared fn_output_shape does not match vrf_shape` 로만 걸린다. 출처: book_ko/src/computing-tensors/register-files.md:38-51 vs 181-190, 04-...md:319-331
- 실기 무증상 오동작 3종(타입으로 못 잡는 부류)이 실측됨: ① 브로드캐스트 HBM→HBM DMA 가 목적지에 아무것도 안 씀 — 2048/2048 전부 불일치, 이전 f32 데이터 잔류물이 읽힘, 에러 없음, 에뮬레이션은 통과. ② commit_view 타일 윈도가 result[32..64] 대신 result[8..40] 에 착지(요소 대신 바이트로 적용된 정황). ③ chip_shuffle 은 컴파일 OK 인데 커널 로드에서 `range end index 56576 out of range for slice of length 37888` 로 프로세스 abort. 출처: 03-텐서-이동.md:171-176,247-256,257-263
- 코퍼스 규모: --backend npu 전수 컴파일 200 커널, --dump-schedule 130 커널, 실기 테스트 89개 중 80 통과. 출처: 03-텐서-이동.md:21-22, 04-...md:440, 13-NPU-실기-매트릭스.md

### 6.7 아직 없는, 반드시 해야 할 실험

- 최소 반례 격리(가장 중요): TRF/Lane/contract 를 전부 제거하고 `HbmTensor<i32, Chip, m![A, 1 # 8]>` 를 커널 경계에 노출하는 것만으로 같은 `incorrect buffer size` 가 나는지 확인. 이게 성립해야 '부분 충전 HBM 노출'이 원인이라는 주장이 선다. 현재는 lane_size 3건의 정황만 있다
- 공식 확정을 위한 격자 스캔: (그룹수 N, 유효폭 k, 패딩폭 P, 원소바이트 S)를 각각 2~3값씩 스캔해 intra_chip_size = (N-1)·P·S + k·S 를 확정. 현재 확보된 것은 (N=8, P=8, S=4, k∈{1,2,4,8}) 4점뿐 — 한 축만 움직인 셈이라 공식이라 부를 근거가 부족
- MNIST 1건 말고 통계적 확인: '부분 충전이 DM/TU 내부에만 있고 HBM 경계에 없으면 항상 통과'를 200 커널 매트릭스 전체에 대해 자동 판정(각 커널의 #[device] 시그니처 HBM 매핑을 파싱해 부분 충전 여부 라벨링 → 컴파일 가부와 교차표). 이 교차표가 논문의 핵심 근거인데 아직 0건
- 제안 검사기의 검출률·오탐률: 백엔드 FAIL 커널 전체 중 몇 %를 cargo check 에서 잡는지, 실기 통과 80 커널 중 몇 건을 잘못 거부하는지. 미측정
- 진단 지연 정량화: cargo check 시간 vs --backend npu 컴파일 왕복 시간 실측(커널당 중앙값). '편집 시점에 잡는 것'의 실익이 초 단위로 얼마인지 없이는 동기가 약하다
- Rust 표현력 한계 실측: 매핑에서 패딩·나눗셈·gcd 를 뽑는 const 산술이 stable/nightly 어디까지 되는지, 세 인코딩(매크로 시점 계산 / const assert / typenum)의 에러 메시지 품질과 컴파일 시간 오버헤드
- 벤더 버전 간 재실행: 0.5/0.6 백엔드에서 lane_size 가 고쳐지면 이 제약은 소멸한다. 최소 2개 백엔드 버전에서 같은 매트릭스를 돌려 '하드웨어 제약'과 '백엔드 결함'을 분리해야 한다. 현재는 0.4.0 단일 버전
- 일반성 논증: Ascend UB 32B 정렬 / TPU 8×128 타일링 / GPU tensor core 레이아웃 중 최소 1개에 같은 3계층 분류를 적용해 이식 가능함을 보여야 단일 벤더 반론에 답할 수 있다
- TRF 용량 제약을 편집 시점으로 옮겼을 때 실제로 잡히는 커널이 몇 건인지 — 현재 확인된 것은 의도적 negative 표본 invalid_to_trf_{full,half} 2건뿐이고, 진짜 사용자 실수 사례가 코퍼스에 있는지 미확인
- VRF 측 검증 부재의 영향 측정: to_vrf 에 verify 를 추가하면 vrf 커널 6건 중 실패 2건이 편집 시점에 잡히는지 실제로 구현해 확인

### 6.8 ★ 심사 반론

**치명적 반론 (reject 사유)**

- 【치명타 1 — 핵심 술어가 자기 코퍼스에 의해 이미 반증됨】 제안의 계층 (a) 지역 술어는 "커널 경계 HBM 매핑에 부분 충전 패딩 그룹(k # N, k<N)이 노출되면 거부"다. 이 술어는 벤더 예제 코퍼스 안에서 최소 3건을 오거부한다. `/home/jun/RNGD-proj/Model_Benchmark/rngd-npu/vISA/reference/examples/furiosa-opt-examples/src/switch_assertions.rs:301` `broadcast1::valid_basic` 의 출력 타입은 `&mut HbmTensor<i8, Chip, m![C / 16, 1 # 4, C % 4, A, C / 4 % 4, B]>` — `1 # 4` 는 4칸 중 1칸만 유효한 부분 충전이고, `to_hbm_view` 로 커널 경계에 그대로 노출된다. 같은 파일 `:320` `valid_degenerate`(`m![C / 4, 1 # 4, A, C % 4, B]`), `:503` `broadcast01::valid_broadcast_with_padding`(`m![C / 4, 1 # 4, A, C / 2 % 2, C % 2, B]`) 도 동일하다. 셋 다 컴파일 OK(`book_guide/_evidence/logs/perkernel_matrix_fixed.txt:101,102,92`) 이고 **실기 PASS**(`_evidence/logs/npu_matrix.tsv:38,39,35`). 즉 제안이 스스로 '0 이어야 한다'고 못박은 오탐률이 코퍼스 안에서 이미 3/80 이다. 이건 튜닝으로 줄일 문제가 아니라 술어가 현상을 잘못 지목했다는 뜻이다 — 실패한 lane_size 3건은 패딩 그룹이 **최내측(마지막) 축**이고 통과한 3건은 패딩 그룹이 **중간 축**이다. 제안은 이 축 위치 변수를 인지조차 못 하고 있다.
- 【치명타 2 — '결정적 반례쌍'이 반례가 아니다】 기여 2번은 'Lane 타입 파라미터가 동일한 두 커널이 정반대 결과'라고 주장하지만, 제안 스스로 '유일한 차이가 HBM 경계 타입'이라고 인정한다. 그러면 그건 반례가 아니라 지역 술어에 **부합하는** 사례다. 리뷰어가 즉시 지적한다: '당신 논문의 규칙이 두 커널을 정확히 구분하는데 왜 그게 지역 타입화 불가의 증명인가?' 게다가 '유일한 차이'라는 진술 자체가 소스로 반증된다(unsupported_claims 참조). 헤드라인 기여가 논리적으로 뒤집혀 있다.
- 【치명타 3 — 3계층 분류의 계층 (b)가 공집합】 논지는 '레인 제약은 매핑 전체에 걸친 관계형 성질'인데, 실측된 어떤 실패도 TRF Lane 파라미터의 관계형 제약이 아니다. `lir: incorrect buffer size at T7: DramShape { inner: []|[A_1=8:8] }` 는 **커밋 결과의 DRAM 버퍼 크기 계산**에서 나고, 문서 자신도 '부분 충전 레인 그룹의 꼬리 패딩을 DRAM 크기 계산이 누락한다'(`book_guide/03-텐서-이동.md:241`)라고 적는다. 대상은 HBM 매핑 타입 하나다. 관계형 리파인먼트가 필요하다는 계층 (b)에는 사례가 0건이고, 논문의 중심 논지('타입으로 못 올리는 관계형 계층이 존재한다')를 뒷받침할 실증이 없다. 증거 없는 계층 분류는 분류표가 아니다.
- 【치명타 4 — 스펙이 아니라 버그를 타입으로 승격】 프론트엔드는 256 B, 백엔드는 228 B 로 같은 텐서를 다르게 센다. 이건 하드웨어 제약이 아니라 **두 크기 함수의 불일치**다. 정상 해법은 백엔드 버그리포트 한 장이고, 정상 결론은 '벤더가 고쳐라'다. 논문은 그 버그를 사용자 소스의 타입 규칙으로 굳혀 사용자에게 우회를 강요한다. 0.5 릴리스가 고치는 순간 검사기는 정당한 커널만 거부하는 도구가 된다. 리뷰어가 가장 먼저 찌를 지점이고, 제안 자신도 risks 1번에 적어놓고 해결책을 제시하지 못한다.
- 【치명타 5 — 대상 모집단이 논문 규모가 아니다】 실측: 200 커널 중 컴파일 FAIL 63, OK 137(`perkernel_matrix_fixed.txt`). 그런데 FAIL 63 중 25건은 `invalid_*`/`*_mismatch` 로 **일부러 잘못 만든 negative 표본**이다(`book_guide/04-연산-분배와-레지스터.md:176-181`, '42개 중 FAIL 25 / OK 17'). 남은 실질 실패 38건 중 이 논문의 검사기가 겨냥하는 정렬/패딩 계열은 `lane_size::valid_size_{1,2,4}`(한 가족) + `aligned_fetch_packet_i4` + `view::{padding,nested,simpl}` + `ve_intra_slice_reduce_split_slice_time` 정도로 최대 8건, 독립 원인으로는 사실상 2~3개다. TRF 용량 검사가 잡을 표본은 의도적 negative 2건(`invalid_to_trf_{full,half}`)뿐이고, 진짜 사용자 실수 사례는 0건이다. '검출률/오탐률' 지표의 분모가 한 자릿수라 어떤 수치도 통계적 의미를 못 갖는다.
- 【치명타 6 — 신규성이 모든 계층에서 선점됨】 상위 주장('소모성 하드웨어 자원 제약을 타입으로'): Dahlia, PLDI 2020. '타일 레이아웃/스레드 매핑을 타입 속성 + 제약 해로': TileLang, arXiv 2504.17577 (2025). '레인 활성을 Rust 타입으로 추적': warp-types 프로토타입. '텐서 shape 를 리파인먼트/liquid 타입으로': US 11275671 (Microsoft, dynamically shaped tensors using liquid types) + Gradual Tensor Shape Checking(2022/2023) + Staged Shape-Dependent Types(2026). 그리고 검색으로 새로 확인된 직격탄: **ascend-rs (2026, ascend-rs.org) — 'Memory-Safe NPU Kernel Programming in Rust'**, Rust 소유권/타입으로 NPU 커널 안전성을 컴파일 타임에 강제하고 Ascend 910B3 실기 413 테스트를 돌린다. 'Rust 타입으로 NPU 커널 제약을 컴파일 타임에 잡는다'라는 이 논문의 포지션 문장이 통째로 겹친다. 남는 신규성은 'furiosa-opt 0.4.0 의 특정 버그 하나의 부호화'다.
- 【치명타 7 — 아티팩트·재현 불가, 단일 벤더·단일 버전】 furiosa-opt 0.4.0 은 비공개 SDK이고, 하드웨어는 이 서버의 RNGD 4장뿐이다. CGO/PLDI 계열 Tool 트랙은 아티팩트 평가가 사실상 필수인데, 리뷰어가 검사기를 돌려볼 방법이 없다. 이식성 논증도 0건 — Ascend/TPU 로 3계층 분류를 옮겼다는 증거가 하나도 없고, 그 작업은 다른 하드웨어 접근권을 요구한다. '단일 칩 하나의 컴파일러 버그를 위한 타입 규칙'이라는 요약이 리뷰 요약란에 그대로 적힐 것이다.
- 【치명타 8 — 베이스라인이 없다】 evaluation 의 '베이스라인 (a) 현행 프론트엔드, (b) --backend npu 전체 컴파일'은 베이스라인이 아니라 **측정 대상**이다. 진짜 베이스라인은 두 개다: ① AscendCraft(arXiv 2601.22760) 식 컴파일러 피드백 루프 — 같은 문제를 타입 없이 푸는 기존 접근인데 비교가 없다. ② `--backend npu` 를 IDE 저장 시 백그라운드로 돌리는 것 — 이게 이기면 타입 확장 자체가 불필요하다. 제안의 유일한 실익 주장인 '진단 지연'을 이 베이스라인과 비교한 초 단위 실측이 need_experiments 에 '미측정'으로 남아 있다. 지연 차이가 커널당 수 초라면 논문 동기 전체가 소멸한다.
- 【치명타 9 — 기여가 연구가 아니라 엔지니어링】 제안 스스로 인정하듯 프론트엔드는 이미 타입 수준이다(m![] 매핑 타입 + `CanApplyXxx` typestate). TRF 용량은 이미 `verify_to_trf`(`book_ko/src/computing-tensors/register-files.md:38-51`)와 mir 검증이 강제한다. 델타는 '기존 런타임/백엔드 검사 술어를 const 산술로 앞당김'이고, 이건 벤더 컴파일러 프론트엔드 개선 커밋이지 논문이 아니다. VRF 쪽 개선(`to_vrf` 에 verify 추가)은 더 노골적으로 그냥 버그픽스다 — `to_vrf`/`to_vrf_at` 에 verify 호출이 없다는 것(`register-files.md:178-190`)은 발견이 아니라 누락된 한 줄이다. 가장 방어 가능하다는 지표 ⑤(에러 위치 정확도)조차 '사용자 소스 라인을 지목할 수 있다'는 정성 주장이고, 사용자 스터디 없이는 '진단 품질이 개선됐다'를 주장할 수 없다.
- 【치명타 10 — 약속한 부정 결과가 자명하다】 '브로드캐스트 HBM→HBM DMA 무기록 / commit_view 오프셋 오착지 / chip_shuffle 커널 로드 abort 는 타입으로 못 잡는다'는 것은 실측 결과가 아니라 정의다. 셋 다 런타임 데이터 정확성·런타임 바운즈 결함이고, 어떤 정적 타입 시스템도 벤더 런타임의 오작동을 예측한다고 주장하지 않는다. '원리적 타입 불가'라는 계층 (c)에 이걸 채우는 것은 자명한 명제를 실측으로 포장하는 것이고, 리뷰어는 이를 기여로 세지 않는다.
- 【치명타 11 — '닫힌 식'이 한 가족에서만 나왔고 같은 에러에 원인이 최소 둘】 `intra_chip_size = (N-1)·P·S + k·S` 를 지지하는 점은 (N=8, P=8, S=4, k∈{1,2,4}) 3점뿐 — k 축 하나만 움직였다. 유일해 보이는 다른 점 `aligned_fetch_packet_i4`(256 vs 240, `switch_assertions.rs:33-34`, `m![A, B # 64]`, A=8/B=32)는 산술로는 식에 맞지만(7·64·0.5 + 32·0.5 = 240), 정작 문서 자신은 원인을 '**i4 가 4비트 폭을 반영 못 하는 DRAM 크기 오산**'으로 다르게 진단한다(`book_guide/03-텐서-이동.md:137`, `13-NPU-실기-매트릭스.md:480`). 게다가 형제 `aligned_fetch_packet_i8`/`_bf16` 은 HBM 타입에 패딩이 아예 없어서(`switch_assertions.rs:67-68,105-106`, `m![A, B]`) 축을 움직인 대조군이 못 된다. 즉 동일한 `incorrect buffer size` 문자열 뒤에 최소 두 개의 서로 다른 원인이 섞여 있고, 논문은 그걸 하나의 닫힌 식으로 통합했다. 특성화 논문이 저지를 수 있는 최악의 실수다.

**이미 출판되어 신규성이 없는 부분**

- Predictable Accelerator Design with Time-Sensitive Affine Types (Dahlia) — PLDI 2020. '소모성 하드웨어 자원(뱅크·포트) 제약을 타입으로 컴파일 타임 거부'라는 상위 주장을 확립. 이 논문의 프레이밍 문장이 그대로 여기 있다.
- TileLang: A Composable Tiled Programming Model for AI Systems — arXiv 2504.17577 (2025). 타일 레이아웃·스레드 매핑을 타입 속성 + 대수적 제약 해로 다룸. '레이아웃/매핑을 타입으로'가 선점됨.
- ascend-rs: Memory-Safe NPU Kernel Programming in Rust — 2026 (ascend-rs.org, Rust→MLIR→AscendC 파이프라인, Ascend 910B3 실기 413 테스트 통과). **가장 위험한 선행**: 'Rust 타입 시스템으로 NPU 커널 제약을 컴파일 타임에 강제한다'는 포지션이 정확히 겹치고, 실기 검증 규모(413)가 이 제안(89)의 4배 이상이다. 리뷰어가 알고 있으면 즉시 reject 사유.
- warp-types: Session-typed GPU divergence (compile-time prevention of shuffle-from-inactive-lane) — 미심사 프로토타입, 2026. '레인 활성 상태를 Rust PhantomData 세션 타입으로 추적'. 이 제안의 '레인 제약 타입화' 문구와 축자적으로 겹침.
- Systems, methods and media for dynamically shaped tensors using liquid types — US Patent 11,275,671 (Microsoft). 텐서 shape 를 liquid/refinement 타입으로 컴파일 타임 검증. 'shape 리파인먼트' 계층 선점 + 특허라 구현 자유도 이슈까지 있음.
- Compile-Time Tensor Shape Checking via Staged Shape-Dependent Types — arXiv 2604.23807 (2026, Suwa & Igarashi) / Gradual Tensor Shape Checking — arXiv 2203.08402 (2022, Springer LNCS 2023). shape 계층 선점.
- AscendCraft: Automatic Ascend NPU Kernel Generation via DSL-Guided Transcompilation — arXiv 2601.22760 (2026). NPU 32B UB 정렬·크기 granularity 제약을 명시적으로 다루되 LLM 다중 패스 + 컴파일러 피드백 루프로 해결. '같은 문제를 타입 없이 이미 실용 수준으로 푼' 접근이라 베이스라인 부재의 근거이자 동기 약화 요인.
- Rigel: Reverse-Engineering the Metal 4.1 Tensor Compute Path on the Apple M4 Max GPU — arXiv 2606.12765 (2026). '벤더 스펙이 숨기거나 모순되는 사실을 마이크로벤치로 복원'하는 실기 특성화 논문. 제안이 권고하는 재프레이밍('vISA 실기 특성화') 자리의 직접 경쟁자이자, 그 장르에서 요구되는 실측 밀도의 기준선을 제시한다.
- Characterizing Real-World Bugs in Tile Programs for Automated Bug Detection — arXiv 2605.19652 (2026). 재프레이밍 시 직접 비교 대상.

**근거 없이 단정한 문장 (수정 필요)**

- "두 커널의 유일한 구조적 차이가 HBM 경계 타입임을 확인" — 소스로 반증. `mnist/mod.rs:30` 은 `contract_lane::<m![1], m![1 # 8]>` (Time = m![1]) 인데 `contract_outer_assertions.rs:48` 은 `contract_lane::<m![A], m![1 # 8]>` (Time = m![A], A=8) 이다. 더해서 MNIST 는 contract_lane 뒤에 `.cast::<bf16, m![1 # 16]>()` 를 끼워 패딩을 소비하고(`mnist/mod.rs:31`) lane_size 는 곧바로 commit 한다. 원소 타입도 bf16 vs i32, 쓰기 경로도 `to_hbm` vs `to_hbm_view` 로 다르다. 구조적 차이가 최소 4개다.
- "레인 제약은 매핑 전체에 걸친 관계형 성질이며 지역적으로 타입화 불가" — 근거 0. 관측된 실패는 전부 커널 경계 HBM 매핑 하나에 대한 지역 술어 후보이고, TRF Lane 파라미터를 원인으로 지목하는 컴파일러 메시지·소스 증거가 없다. 오히려 문서(`03-텐서-이동.md:241`)는 원인을 DRAM 크기 계산으로 적는다. 논지의 중심 명제가 증거 없이 단정됐다.
- "부분 충전 그룹을 HBM 출력 타입에 노출하는 패턴이 막힌다" — 코퍼스가 반증. `switch_assertions.rs:301,320,503` 의 세 커널이 `1 # 4` 를 커널 경계 HBM 타입에 노출하고 `to_hbm_view` 로 쓰는데 컴파일 OK + 실기 PASS(`npu_matrix.tsv:35,38,39`). '노출 자체가 금지'라는 명제는 거짓이다.
- "꼬리 패딩 누락의 닫힌 식" — 3점(k 축 1개)에서 유도한 식을 '닫힌 식'이라 부른다. N·P·S 축을 독립적으로 움직인 데이터가 없고, 유일한 타축 후보(i4)는 문서가 다른 원인으로 진단한다. need_experiments 2번이 이 부족을 인정하지만, contributions 3번은 이미 확정된 것처럼 적혀 있다 — 기여 목록과 실험 목록이 서로 모순된다.
- "편집 시점 검출률 현재 0%" — 형식적으로는 참이지만 오해를 부른다. 정렬/패딩 위반의 상당수는 이미 프론트엔드 `verify_*` 나 typestate 가 잡고 있고, 남은 것이 백엔드로 새는 것이다. '0%' 는 '이 특정 백엔드 버그 계열의 0%' 이지 '정렬 제약의 0%' 가 아니다.
- "실기 89 테스트에 대한 검출률/오탐률" 을 지표로 세우는 것 — 89개 중 이 논문의 검사기가 관여하는 커널은 실질적으로 lane_size 계열 1개(valid_size_8 PASS)와 broadcast 패딩 계열 3개뿐이다. 나머지 85개는 검사기가 손대지 않으므로 오탐률 분모로 쓰면 수치가 인위적으로 좋아진다.
- "코퍼스 위 정량 평가: 벤더 예제 200 커널" — 200 은 컴파일 시도 커널 수이고(`perkernel_matrix_fixed.txt` 200행 = OK 137 / FAIL 63, 검증됨), 이 중 42개는 벤더가 일부러 잘못 만든 assertion 표본이다. 논문에서 '200 커널 코퍼스'라 쓰면 리뷰어가 유효 표본을 되물을 때 방어할 수 없다.
- "에러 위치 정확도 ... 사용자 소스 라인 지목 성공률" — 측정 프로토콜이 없다. 무엇이 '정답 라인'인지 정의하는 그라운드 트루스(누가 라벨링하는가, 커널당 결함 위치가 유일한가)가 제시되지 않았고, 이 지표를 '가장 방어 가능'하다고 자평한다.
- "프론트엔드 검증의 비대칭 ... to_vrf 는 verify 호출이 아예 없다" — 이 항목은 **소스로 확인됨**(`book_ko/src/computing-tensors/register-files.md:38-51` 의 `verify_to_trf::<D, Lane, Time, Packet, Element>(&address)` vs `:178-190` 의 verify 없는 `to_vrf`/`to_vrf_at`). 근거는 사실이나, 이것이 '연구 기여'라는 함의는 근거 없다 — 누락된 함수 호출 한 줄이다.
- TRF 용량 수치(Full 65,536 B / Half 32,768 B, 8 KB·4 KB per lane), 부분 충전 실패값 228/232/240, 실기 89개 중 80 PASS(`npu_matrix.tsv` 실측: PASS 80 / FAIL 5 / ABORT 3 / OTHER 1), MNIST 실기 1/1 통과(`npu_matrix.tsv:mnist_tests test_mnist PASS`, `13-NPU-실기-매트릭스.md:40` '이미지 10장 전부 정답'), 브로드캐스트 DMA 2048/2048 불일치·commit_view 오착지·chip_shuffle abort(`npu_matrix.tsv` 의 mismatch=1 / load=1 행으로 교차확인) — 이 수치들은 **전부 파일에서 확인되어 근거가 확실하다**. 문제는 수치가 아니라 그 수치로부터 끌어낸 인과 해석이다.

**조사자 스스로 적은 위험**

- 핵심 관측(228/232/240)이 하드웨어 제약이 아니라 백엔드 DRAM 크기 계산 결함이다. 프론트엔드는 256 이라 하고 백엔드는 228 이라 한다 — 두 크기 함수의 불일치일 뿐이다. 이걸 타입으로 굳히면 '버그를 스펙으로 승격'하는 셈이고, 벤더가 0.5에서 고치면 논문의 근거가 통째로 증발한다. 리뷰어가 가장 먼저 찌를 지점
- 프론트엔드가 이미 타입 수준이다(m![] 매핑 타입 + typestate). '타입으로 올린다'의 델타가 '검사 술어 하나 추가'로 축소되어 보인다. 실제로 TRF 용량은 이미 verify_to_trf/mir 이 강제하고 있어, 기여가 '진단 위치를 앞당김'에 그친다
- Dahlia(PLDI'20)가 '소모성 하드웨어 자원 제약 → affine 타입'의 원형을 이미 세웠고, TileLang(2025)은 레이아웃/스레드 매핑을 타입 속성 + 제약 해로 푼다. '하드웨어 제약을 타입으로'라는 상위 주장은 신규성을 못 얻는다
- GPU 쪽에 warp-types(레인 활성 마스크를 세션 타입으로 추적, shuffle-from-inactive-lane 컴파일 타임 차단) 프로토타입이 이미 존재한다. 미심사이지만 '레인 제약의 타입화'라는 문구가 정확히 겹쳐, 리뷰어가 알고 있으면 곤란하다
- 단일 벤더·단일 칩·단일 컴파일러 버전(furiosa-opt 0.4.0). SDK 가 공개가 아니면 아티팩트 평가가 불가능하고 재현이 안 된다. 상위 학회에서 치명적
- 타입 강화가 가장 필요한 지점에서 안 통한다: 브로드캐스트 DMA 무기록, commit_view 오프셋 오착지, 커널 로드 abort — 실기에서 실제로 사람을 다치게 하는 3종은 타입으로 원리적으로 못 잡는다. 논문의 약속과 실효가 어긋난다
- MNIST 반례 때문에 지역 술어만으로는 오탐이 난다. 'HBM 경계 부분 충전 금지'를 강하게 걸면 정당한 커널을 거부할 위험이 있고, 약하게 걸면 검출률이 떨어진다. 이 트레이드오프를 정량화하지 못하면 논문이 성립하지 않는다
- '단일 칩이라 일반성 없다'에 대한 방어가 현재 없다. Ascend/TPU 로 분류를 이식했다는 증거가 하나도 없고, 이식 작업은 다른 하드웨어 접근권을 요구한다
- 정직하게 쓰면 부정 결과 논문('타입은 여기까지')이 되는데, 그건 신규성이 아니라 측정 품질로 평가받는다. 200 커널/89 실기는 특성화 논문 기준으로는 작은 코퍼스다

### 6.9 살리는 길 — 무엇을 바꿔야 하는가

【결론: 단독 논문으로는 못 살린다. 제안 자신의 권고(§타입 확장 절로 강등)가 맞고, 오히려 그보다 더 축소해야 한다.】\n\n■ 즉시 폐기할 것\n1. '타입으로 올린다' 프레이밍 전부. Dahlia/TileLang/warp-types/ascend-rs 4중 선점이라 어떤 각도로도 신규성이 안 선다. 특히 ascend-rs(2026, Rust NPU 커널 컴파일 타임 안전, 실기 413테스트)와 나란히 놓이면 이 제안은 열등한 중복이다.\n2. 기여 2번 '결정적 반례쌍'. 반례가 아니고(지역 술어가 두 커널을 정확히 구분한다), '유일한 차이'라는 진술도 소스로 거짓이다(contract_lane Time 파라미터 m![1] vs m![A], 중간 cast 유무, 원소 타입, to_hbm vs to_hbm_view).\n3. 기여 3번의 '닫힌 식'. 3점 1축에서 나온 것을 식이라 부를 수 없고, 같은 에러 문자열 뒤에 최소 두 원인(꼬리 패딩 누락 / i4 서브바이트)이 섞여 있다.\n4. 기여 5번 '부정 결과'. 런타임 데이터 오류를 정적 타입이 못 잡는다는 것은 자명 명제다.\n5. 검사기 구현·검출률/오탐률 평가 전체. 유효 표본이 한 자릿수라 어떤 수치도 방어 불가.\n\n■ 그래도 남길 가치가 있는 단 하나\n제안이 스스로 못 찾은 진짜 관측이 코퍼스 안에 있다. **부분 충전 패딩 그룹의 '축 위치'가 성패를 가른다**: 패딩 그룹이 HBM 매핑의 최내측 축일 때만 백엔드가 꼬리 패딩을 누락한다(FAIL: contract_outer_assertions.rs:29,60,91 `m![A, k # 8]`, view.rs:43 `m![[A,B] # 64]`, switch_assertions.rs:33 `m![A, B # 64]`), 중간 축이면 후속 축이 그 공간을 메워 통과한다(PASS: switch_assertions.rs:301,320,503 의 `..., 1 # 4, ...`). 이건 소스 grep 5분이면 라벨링되고, 검증 가능하며, 제안의 술어보다 훨씬 날카롭다. 이 한 문단이 살아남을 전부다.\n\n■ 살리려면 이렇게 재구성하라\n(1) **논문 단위를 바꾼다.** 'vISA 실기 특성화 — 프론트엔드 타입체크와 에뮬레이션이 사주지 않는 것' 을 본 논문으로 세우고, 타입 이야기는 §Discussion 한 절(1페이지)로 강등한다. 경쟁 기준선은 Rigel(arXiv 2606.12765)과 tile-bug characterization(arXiv 2605.19652)이다.\n(2) **가설을 코퍼스로 먼저 죽여라.** 200 커널의 `#[device]` 시그니처를 파싱해 (패딩 그룹 유무, 축 위치, 원소 바이트, 그룹 수)를 자동 라벨링하고 컴파일 가부와 교차표를 만든다. 이건 need_experiments 3번이고, 하루면 끝나며, 지금 0건이다. 이 표 없이 쓴 문장은 전부 반증당한다 — 실제로 내가 5분 만에 3건 반증했다.\n(3) **최소 반례를 격리하라.** TRF/contract 를 다 걷어내고 `HbmTensor<i32, Chip, m![A, 1 # 8]>` 만 노출하는 12줄 커널로 같은 에러가 나는지, 그리고 패딩 축을 중간으로 옮기면 통과하는지 확인한다. 이게 성립해야 인과 주장이 선다.\n(4) **격자 스캔으로 원인을 분리하라.** (N, P, S, k) 를 독립적으로 2~3값씩 스캔해 '꼬리 패딩 누락'과 'i4 서브바이트 오산'이 같은 결함인지 다른 결함인지 판정한다. 같은 에러 문자열에 두 원인이 섞여 있다는 것을 밝히는 것만으로도 특성화 논문의 한 절이 된다.\n(5) **버그와 제약을 분리하라.** 최소 2개 백엔드 버전(0.4 + 0.5/0.6)에서 같은 매트릭스를 돌려 '하드웨어 제약'과 '벤더 결함'을 갈라라. 이 분리가 이 자료의 유일한 지속 가능한 가치다 — 0.5에서 lane_size 가 고쳐지면 그건 논문의 소멸이 아니라 '결함이었음의 증명'이 되게 설계하라.\n(6) **출구를 낮춰라.** 이 자료의 현실적 도착지는 ARRAY @ PLDI 워크숍 포스터, 또는 KCC/KSC 경험 논문이다. CGO Tool 트랙은 비공개 SDK 때문에 아티팩트에서 막히고, 상위 학회는 단일 벤더·단일 버전·8건 표본으로 통과하지 않는다.\n(7) **가장 정직한 대안**: 검사기를 논문으로 만들지 말고 furiosa 벤더에 버그리포트 3장(lane_size 꼬리 패딩, i4 서브바이트, to_vrf verify 누락)으로 제출하고, 논문은 '실기 매트릭스' 자체로만 쓴다. 89 실기 + 200 컴파일 + 결함 유형 분류는 그 자체로 국내 경험 논문 한 편 분량이며, 타입 이야기를 붙이는 순간 오히려 약해진다.

### 6.10 후보 학회와 소요 기간

- CGO Tool/Practical Papers (검사기 구현 + 코퍼스 평가 형태로)
- LCTES (임베디드/가속기 컴파일러, 실측 중심 수용 폭이 넓음)
- ARRAY @ PLDI (배열/텐서 타입 워크숍 — 현재 완성도에 가장 맞는 자리)
- IISWC 또는 ISPASS (특성화 논문으로 재프레이밍할 경우)
- 한국정보과학회 KCC/KSC, 정보과학회논문지 시스템및이론 (실측 기반 경험 논문으로 국내 우선 출구)

**소요 기간 추정**: 3~5개월 (검사기 구현 1.5 + 200커널 라벨링·교차표 측정 1 + 이식성 논증 1 + 집필 0.5~1.5). 단 벤더 백엔드 버전 고정이 전제이고, 0.5 릴리스가 lane_size 를 고치면 근거 재수집으로 +2개월

**신규성 확신도**: 낮음~중간. 상위 주장('하드웨어 제약을 타입으로')은 Dahlia(PLDI'20)로 이미 확립됐고, '레이아웃을 타입으로'는 TileLang(2025), '레인 활성을 타입으로'는 warp-types 프로토타입이 선점했다. 검색 10회로 '부분 충전 패딩 그룹의 DRAM 경계 노출'이나 'NPU TRF 레인 부분 활성의 타입화'를 직접 다룬 논문은 찾지 못했으므로 구체 규칙 자체는 미발표로 보이지만, 그 규칙이 하드웨어 제약이 아니라 벤더 백엔드 크기 계산 결함(프론트엔드 256 vs 백엔드 228)에 가깝다는 점이 신규성의 가치를 떨어뜨린다. 신규성이 있는 쪽은 오히려 부정 결과다 — Lane 타입 파라미터가 동일한 두 커널이 정반대 결과를 낸다는 반례쌍은 검색된 어느 선행연구에도 없고, '지역 타입 규율의 한계'를 실기로 보인 사례로는 새롭다. 다만 그 부정 결과 하나로 독립 논문을 세우기엔 코퍼스(200 커널/89 실기)와 일반성(단일 벤더·단일 버전)이 부족하다. 권고: 이 주제를 단독으로 밀지 말고, 'vISA 실기 특성화 — 프론트엔드 타입/에뮬레이션이 사주지 않는 것' 논문의 §타입 확장 절로 넣어라. 그 상위 논문이라면 viable 이다.

### 6.11 검색으로 확인한 관련 연구

| 제목 | 학회·연도 | 이 주제와의 관계 |
|---|---|---|
| Predictable Accelerator Design with Time-Sensitive Affine Types (Dahlia) | PLDI 2020 | 겹침 — '소모성 하드웨어 자원(메모리 뱅크·포트) 제약을 타입 시스템으로 컴파일 타임에 거부'라는 이 주제의 상위 주장을 이미 확립. HLS/FPGA 대상이라 NPU 레지스터파일 레인·DRAM 경계 패딩은 안 다루지만, '하드웨어 제약을 타입으로'의 신규성은 여기서 이미 소진됨 |
| TileLang: A Composable Tiled Programming Model for AI Systems | arXiv 2504.17577 (2025), OpenReview 심사 중 | 겹침 — 타일의 스레드 매핑을 '타입 속성'으로 보고 타일 프리미티브의 레이아웃/매핑 제약을 타입 추론(대수적 제약 해)으로 푼다. 목적이 검증이 아니라 성능 자동화이고 GPU(NVIDIA/AMD) 대상이라는 점이 차이 |
| Exocompilation for Productive Programming of Hardware Accelerators (Exo) | PLDI 2022 | 기반 — 커스텀 하드웨어 명령·특수 메모리·설정 상태를 사용자 라이브러리로 외부화하고, effect analysis 로 스케줄 재작성의 등가성·메모리 안전을 보장. 다만 '벤더 백엔드 낮추기 실패를 편집 시점에 예측'하는 문제는 다루지 않아 이 주제의 자리는 남아 있음 |
| Compile-Time Tensor Shape Checking via Staged Shape-Dependent Types | arXiv 2604.23807 (2026), Suwa & Igarashi, Kyoto Univ. | 인접 — staged 계산으로 텐서 shape 일관성을 사실상 정적으로 보장. shape(차원 일치)만 다루고 정렬·패딩·레지스터파일 용량은 범위 밖. 이 주제가 'shape 다음 단계'라는 위치를 잡을 근거 |
| Gradual Tensor Shape Checking | arXiv 2203.08402 (2022) / Springer LNCS 2023 | 인접 — shape 검사에 gradual typing 을 도입. 역시 shape 전용이며 하드웨어 자원 제약은 다루지 않음 |
| AscendCraft: Automatic Ascend NPU Kernel Generation via DSL-Guided Transcompilation | arXiv 2601.22760 (2026) | 인접·대비군 — Ascend NPU 의 엄격한 정렬 제약과 온칩 버퍼 크기 granularity 를 문제로 명시하지만, 해법이 타입이 아니라 LLM 다중 패스 lowering + 컴파일러 피드백 루프(Alignment and Padding Refinement 패스, DataCopyPad 치환)다. '같은 문제, 타입 아닌 해법'이라 이 주제의 baseline 으로 쓸 수 있음 |
| warp-types: Session-typed GPU divergence — compile-time prevention of shuffle-from-inactive-lane bugs | 미심사 프로토타입 (crates.io / GitHub, 2026 접근). H200·RTX 4000 Ada 실기 검증 주장 | 겹침 — '레인 활성 상태를 Rust 타입(PhantomData 기반 세션 타입)으로 추적해 비활성 레인 접근을 컴파일 타임에 차단'. 이 주제의 '레인 제약을 타입으로'와 문구가 정확히 겹친다. 학회 논문은 아니지만 신규성 주장 시 반드시 언급·구분해야 함 |
| Characterizing Real-World Bugs in Tile Programs for Automated Bug Detection | arXiv 2605.19652 (2026) | 인접 — 타일 프로그램의 실세계 버그 분류. 이 주제를 '타입 강화'가 아니라 '결함 특성화'로 재프레이밍할 경우의 직접 비교 대상 |


---

## 주제 7. 정적 형상 NPU 위의 동적·비정형 텐서 (VCG·MoE 블록 실행)

> **판정: `weak` → 적대적 심사 후 `not-a-paper`**  (논문 아님 — 현재 형태로는 엔지니어링 노트)
> 조사자가 찾은 선행연구 8건 · 심사자가 제기한 치명적 반론 10건 ·
> "이미 출판됨" 지적 14건 · 근거 없이 단정한 문장 9건

### 7.1 초록 — 한 문장 주장

정적 형상 NPU가 비정형 워크로드를 다루는 두 축(하드웨어 valid-count 술어 생성 + 소프트웨어 블록 정규화) 중, 전자(VCG)는 문헌에 미기재된 진짜 신규 관찰이지만 단일 벤더 고정 하드웨어이고 사이클 기여도가 0%대이며, 후자(MoE 블록 실행)는 MegaBlocks·vLLM moe_align_block_size 로 이미 published 다 — 현 상태로는 "정적 형상 가속기 프로그래밍" 논문의 한 절이지 독립 논문이 아니다.

### 7.2 문제의식과 선행연구의 빈틈

검색으로 확인한 공백은 좁고 하나뿐이다. (a) TCP/RNGD 의 공식 발표물(ISCA 2024 "TCP: A Tensor Contraction Processor for AI Workloads", IEEE Micro 2025 Hot Chips 기고)을 실제로 열어 확인한 결과 valid count generator, valid_size, time filter, packet clipper, 패딩 제외 리듀스, MoE 블록 실행에 대한 서술이 **전혀 없다**. 즉 VCG 라는 기제 자체는 문헌에 기록된 적이 없다. (b) 선행 dynamic-shape 컴파일러(CoRa, DietCode, Nimble, DISC, SoD²)는 전부 소프트웨어 측 패딩 최소화이고, **고정 용량 하드웨어 술어 생성기(술어 3+1개)에 아핀 매핑 표현식을 낮출 때 무엇이 표현 가능하고 무엇이 불가능한가**를 형식화한 연구는 못 찾았다. 이것이 유일하게 방어 가능한 공백이다. 반대로 (c) MoE 블록 단위 실행의 공백은 **없다**. 고정 블록 B, expert별 ceil(Count_e/B), 블록당 expert_id(-1=skip), cumsum 기반 scatter 주소는 MegaBlocks(MLSys 2023)와 vLLM moe_align_block_size 가 이미 하는 것과 동일하다. (d) "정적 형상 NPU 에서 MoE 동적 shape" 라는 프레이밍조차 NPUMoE(arXiv 2604.18788, 2026)가 선점했다.

### 7.3 제안한 기여

- (살아남는 것) 아핀 매핑 표현식 → 고정 용량 하드웨어 술어 생성기 낮추기의 표현력 경계를 형식화: 리듀스 축 R 을 Slice/Time/Packet 에 stride/modulo 인수로 분배할 때 valid_size(s,t) 로 정확히 포착되는 배치 집합의 특성화와, 5가지 표현 불가 클래스(Slice 내 역순, Time-Slice-Time interleave, TimeMajor 과도패딩 PADDED−R::SIZE > slice_span, Packet 내 major-R/축 공유, Slice+Packet 분할)의 반례 증명
- (살아남는 것) 술어가 파이프라인 단을 통과할 때의 valid-count 전파 대수: narrow_split → (min(v,4), max(v−4,0)), narrow_trim → min(v,4), widen_concat → v_low+v_high, widen_pad → 불변. 이 접두(prefix) 불변식이 깨지는 지점이 곧 표현 불가 경계라는 통일적 설명
- (조건부) 하드웨어 valid-count 태깅 vs 소프트웨어 identity 패딩(Fetch 마스킹)의 실측 A/B — 성능 이득이 아니라 **적용 가능성 차이**로서의 정량화(exp 등 비가역 변환이 리듀스 앞에 오면 identity 원소가 존재하지 않아 소프트웨어 경로가 원천 봉쇄됨)
- (약함) MoE 블록 실행의 정적 형상 이식 — 단, MegaBlocks/vLLM 대비 신규성은 '제어 흐름을 branch logger cumsum + O(E²) all-pairs rank 로 완전 제거' 라는 구현 제약 대응뿐이며, 알고리즘적 신규성은 사실상 0
- (정직한 부산물) 정적 형상 가속기에서 비정형 워크로드를 막는 실제 장벽 목록 — 표현력이 아니라 컴파일러 낮추기 실패(정렬, pack alias, ICE, 분기 미구현)가 지배적임을 실기 63건 실패 분류로 보인 것

### 7.4 방법

두 층으로 나눈다. (1) 형식화 층: 매핑 표현식 R # PADDED / n % m 의 Slice/Time/Packet 배치를 술어 생성기 설정(sequencer, slice_mask, slice_thres, time_thres, mode)으로 낮추는 함수를 정의하고, 이 낮추기가 전사(surjective)가 아닌 이유 — 즉 slice별로 달라지는 valid_size 를 t 단일 함수 packet_clipper(t) 가 표현 못 하고, 비단조 slice 유효성을 단일 slice_thres 가 표현 못 함 — 을 정리로 세운다. SliceMajor 3-case(below/boundary/above), TimeMajor 2-case 정당성 증명은 이미 문서에 있으므로 형식화만 하면 된다. (2) 측정 층: 같은 리듀스 커널을 (a) VCG 경로, (b) identity 패딩 경로, (c) 소프트웨어 명시 마스크 곱 경로 세 가지로 작성해 --dump-schedule 사이클과 실기 실행 시간·정확도를 A/B 한다. 이어 varlen 어텐션(prefill 길이 분포를 OpenOrca 등 실측 분포에서 추출)과 MoE(Qwen3 E=128/K=8, gpt-oss K=4)를 두 정책으로 구현해 padding FLOP 낭비율과 실측 처리량을 비교한다. 비교 축은 같은 서버의 A6000 vLLM 벤치(Model_Benchmark/gpu)를 외부 기준선으로 붙인다. 일반성 방어는 RNGD 단일 칩으로는 불가능하므로, 술어 생성기를 파라미터화한 추상 모델(술어 슬롯 수, slice 의존 가능 여부, 접두 제약 유무)을 정의하고 RVV vl/tail, SVE predicate, TPU 마스킹을 그 모델의 특수점으로 배치해야 한다.

### 7.5 평가 설계

비교 대상: (기준선1) identity 패딩 — Fetch 마스킹으로 패딩칸에 리듀스 항등원 채우기, (기준선2) 소프트웨어 마스크 곱, (기준선3) 패딩 그대로 두고 전량 계산, (기준선4, 외부) A6000 3장 vLLM 의 FlashAttention varlen + fused MoE. 지표: ① 유효 FLOP 비율(1 − 패딩 FLOP/총 FLOP), ② 컴파일러 스케줄 모델 사이클 및 엔진별 분해, ③ 실기 wall-clock 과 tokens/J, ④ 표현 가능 배치 비율(무작위/실제 모델 형상에서 뽑은 매핑 후보 중 VCG 로 낮춰지는 비율 — 이게 "표현력" 주장의 유일한 정량 지표다), ⑤ 정확도(비결합 AddSat·f32 순서 의존 때문에 비트 일치가 아니라 ULP 허용 기준). MoE 는 T(prefill 길이)·K·B 를 스윕해 실측 G 와 최악 상한 E+floor((T·K−E)/B) 의 격차를 그린다 — 이 격차 곡선이 "블록 크기 B 선택" 을 논문화할 유일한 여지다.

### 7.6 이미 확보된 근거

- VCG 사양 확정(vcg.md:30-52). valid_size(s,t) ∈ {0,…,8} — 즉 9개 값이며 8원소 flit 당 하나. 규칙은 valid_size = (모든 time filter 가 valid) ? packet_clipper.valid_size(t) : 0. packet_clipper.valid_size(t) = clamp(axis_size − idx(t), 0, packet_span) 로 t 에만 의존하고 slice s 에는 의존하지 않는다 — 이 slice 독립성이 표현력 한계의 근원이다.
- 하드웨어 용량 확정(vcg.md:851-863): packet clipper 1개, time filter 3개 → 한 invocation 에서 최대 4개 패딩 축만 추적. 각 필터/클리퍼의 sequencer 항목은 8개. 중요: time filter 는 리듀스 축 R 전용이 아니라 패딩된 비축약 축도 슬롯을 먹는다(vcg.md:493, [H,C,W]=[5,5,19] 예제는 W 가 clipper, C·H 가 filter 0·1).
- 표현 불가 클래스 5종과 그 반례가 문서에 확정되어 있다(vcg.md:621-849): Slice 내 sub-expr 역순(비단조 slice 유효성), Time-Slice-Time interleave(slice마다 유효 스텝 수 상이), TimeMajor 과도패딩 PADDED_SIZE − R::SIZE > slice_span, Packet 에 major-R 또는 타 축 공유(접두 성질 파괴), Slice+Packet 분할(R=2045, R#2048/8 × %8 에서 slice 0–254 는 valid_size=8 이 필요한데 slice 255 는 5 가 필요 — 단 R::SIZE % packet_span = 0 인 퇴화 경우는 지원).
- SliceMajor/TimeMajor 정당성 증명이 이미 문서에 있다(vcg.md:266-270, 349-352). r = r_slice + idx (SliceMajor) / r = idx·slice_span + r_slice (TimeMajor) 분해로 3-case·2-case 논증. TimeMajor 에는 '전부 무효' 영역이 없다는 따름정리 포함.
- valid-count 전파 규칙 확정(vcg.md:869-878): narrow_split → (min(v,4), max(v−4,0)), narrow_trim → min(v,4), widen_concat → v_low+v_high, widen_pad → 불변. trim 은 매핑이 v ≤ 4 를 정적으로 보장해야 안전.
- 경쟁 기제가 이미 문서화되어 있다 — identity 패딩(Fetch 마스킹으로 Add→0, Max→−inf/i32::MIN, Min→+inf/i32::MAX 를 미리 채움). 적용 불가 조건도 명시: exp(x)+… 처럼 비가역 변환이 앞서면 exp(p)=0 인 p 가 없어 못 씀(book_guide/06-벡터-캐스트-전치.md §3④, intra-slice-reduce.md:190-210). 이것이 VCG 의 유일한 명시적 우위 논거다.
- MoE 블록 실행 사양 확정(mixture-of-experts.md:262-325): G = Σ_e ceil(Count_e/B), 최악 (T·K−E)/B + E. Scatter_Idx = Global_Offset·B + Local_Offset, Expert_IDs = filter_compaction(Grid ≥ 0). Branchless TopK = (Score<<16)|Index 비트 패킹 + 음수에 ^0x7fff0000 비교 트릭 + O(E²) all-pairs rank(E=128 이면 토큰당 16,384회 비교). cumsum 은 VE branch logger 로 구현(mixture-of-experts.md:380-401). 모델 파라미터 E≈128, K: llama4=1, gpt-oss=4, qwen3=8(:48-50).
- 【반증 근거】리포지토리에 MoE 커널이 없다. 책의 MoE 코드 블록은 전부 rust,ignore 의사코드이고, 가이드 스스로 '미확인 청사진' 으로 분류한다(book_guide/08-커널-예제.md:361, 372, 463). 즉 기제 (2)는 로컬에 실행 근거가 0 이다.
- 【반증 근거】VCG 의 가장 흥미로운 배치(R 을 Slice+Time 분할)가 실기 낮추기에서 거부된다. ve_intra_slice_reduce_split_slice_time (S=15 → S#16, Slice /4 · Time %4) → 'visa: while lowering TensorUnit / caused by: cannot reduce pack alias' (book_guide/06 §3⑤). 이 커널은 테스트조차 없어 에뮬레이션 검증도 없다.
- 【반증 근거】Vector Engine 은 사이클 예산에서 사실상 0 이다. 커널 130개 합산 스케줄 모델에서 DmaEngine 75,464,336 사이클(96.5%, 470 instr) vs VectorEngine 14,770 사이클(0.0%, 50 instr). mnist::forward 단독 17,953 사이클 중 DMA 12,365(68.9%), VE 1,162 (book_guide/06 §10). 단 이는 --dump-schedule 예측이지 실기 측정이 아니다.
- 【반증 근거】실기 실행 가능 범위가 좁다. 게이팅된 벤더 예제 89개 중 80 PASS(정상 판정 83/89=93.3%)지만, transformer::{embedding,attention,decoder,head}::forward 4종이 전부 ICE 이고 matmul 7종 전부 실기 불가(분기 미구현, DMA 8B 정렬, commit_trim packet mismatch 등). npu 백엔드 전체 FAIL 63건 중 의도적 음성 표본은 23건뿐이고 40건은 진짜 공백/컴파일러 버그(book_guide/13-NPU-실기-매트릭스.md §7.1-7.2, _GROUND_TRUTH.md N6).
- 【선행연구 확인】ISCA 2024 TCP 논문과 IEEE Micro 2025 Hot Chips 기고(직접 PDF 열람)에 VCG·valid_size·time filter·packet clipper·MoE 블록 실행 서술이 전무하다. IEEE Micro 본문은 VE 를 'intra-slice/inter-slice reduction, 8-way throughput per slice' 수준으로만 기술한다. 즉 VCG 는 미공개 기제다.

### 7.7 아직 없는, 반드시 해야 할 실험

- VCG vs identity 패딩 vs 소프트웨어 마스크 3-way A/B — 동일 리듀스 커널을 세 정책으로 작성, --dump-schedule 사이클과 실기 wall-clock 을 측정. 이것이 '하드웨어 valid-count 태깅이 소프트웨어 마스킹 대비 얼마나 이득인가' 라는 핵심 질문에 답하는 유일한 실험이며 현재 단 한 건도 없다. 예상 결과가 '사이클 차이 ≈ 0'(VE 0.0%) 이라는 점까지 미리 각오해야 한다.
- 표현 가능 비율 측정 — 실제 모델(Qwen3/gpt-oss/llama4)의 리듀스 축 크기 분포에서 매핑 후보를 열거하고, 그중 VCG 로 낮춰지는 비율 vs 5가지 표현 불가 클래스에 걸리는 비율을 센다. 표현력 주장을 뒷받침할 유일한 정량 지표.
- '표현 가능 ≠ 낮추기 가능' 격차 정량화 — cannot reduce pack alias 로 막힌 Slice+Time 분할처럼, VCG 문서상 합법인데 --backend npu 가 거부하는 배치가 얼마나 되는지 전수 조사. 이게 크면 논문 주장이 '하드웨어 능력' 이 아니라 '컴파일러 미성숙' 이 된다.
- MoE 커널을 실제로 작성해 실기에 올리기 — 현재 리포에 없다. branchless TopK(O(E²), E=128), branch logger cumsum, filter_compaction, block scatter/gather 를 vISA std API 로 구현하고 실기 실행. transformer 4종이 전부 ICE 인 상태라 이 실험 자체가 성립할지 불확실하다.
- MoE 라우팅 오버헤드 분해 — O(E²) all-pairs rank 와 cumsum 이 expert GEMM 대비 몇 %인지. E=128, K=8, T=수천 이면 라우팅이 지배할 수 있다. MegaBlocks/vLLM 대비 이 오버헤드가 크면 '정적 형상 대가' 라는 정직한 결과가 나온다.
- 블록 크기 B 스윕과 실측 G vs 최악 상한 E+floor((T·K−E)/B) 격차 — 실제 라우팅 분포(OpenOrca 등)에서 최악 상한이 얼마나 과대한지. 컴파일러가 최악 기준으로 메모리를 잡으므로 이 격차가 곧 낭비다.
- varlen 어텐션으로의 일반화 검증 — VCG 는 리듀스 축 R 하나 + 패딩된 비축약 축 최대 3개만 추적한다. 배치 내 시퀀스마다 길이가 다른 varlen 어텐션은 slice 별로 valid_size 가 달라야 하는데 packet_clipper.valid_size(t) 는 slice 독립이다. 즉 varlen 어텐션이 애초에 VCG 로 표현 가능한지부터 확인해야 한다 — 안 되면 논문의 '일반화' 주장이 붕괴한다.
- 외부 기준선 정렬 — 같은 서버 A6000×3 vLLM(Model_Benchmark/gpu)에서 동일 모델·동일 시퀀스 길이 분포로 FlashAttention varlen / fused MoE 를 돌려 tokens/s, tokens/J 를 붙인다. 없으면 '단일 칩 이야기' 반론을 막을 수 없다.
- 추상 술어 생성기 모델의 정의와 기존 기제 배치 — 술어 슬롯 수, slice 의존 허용 여부, 접두 제약 유무를 파라미터로 두고 RVV vl/tail-agnostic, SVE predicate, TPU 마스킹, CoRa 의 소프트웨어 접근을 같은 모델의 점으로 배치. 일반성 방어의 유일한 수단이며 아직 착수 전.

### 7.8 ★ 심사 반론

**치명적 반론 (reject 사유)**

- 【신규성 근거 자체가 무너졌다 — ISCA'24 원문을 내가 읽었다】 제안서가 '이미지 PDF라 열람 실패, 미확인'으로 남긴 TCP ISCA 2024 본문을 https://web.ist.utl.pt/nuno.lopes/pubs/tcp-isca24.pdf 에서 받아 pdftotext 로 추출해 읽었다(/tmp/tcp_isca24.txt, 1172줄). 결과: (i) §IV.D Vector Engine 기능 목록에 'predicated operations' 가 명시돼 있다 — 술어 연산은 이미 published 다. (ii) §IV.C Fetch Unit 에 'a fetch process unit for data preparation like type conversion and padding'. (iii) §IV.F Commit Unit 에 'the commit process unit ... can remove padding from data for compaction during storage' — 패딩 삽입·제거가 둘 다 논문에 있다. (iv) §V.B 는 '마지막 차원이 dot-product engine 폭에 정렬 안 되면 utilization 이 떨어지고, 2의 거듭제곱이 아닌 축을 slice 로 나누면 성능이 떨어지므로 컴파일러가 in-slice non-last dimension 으로 배치한다'며 패딩/정렬 비용과 컴파일러 tactic 선택을 이미 다룬다. (v) §VI 'Supporting Dynamic Shapes and Control Flow ... TCP's software supports dynamic shapes by modifying control registers on the fly'. 즉 '패딩 제외 리듀스·동적 형상에 대한 서술이 전혀 없다'는 제안서의 gap(a) 는 사실이 아니다. 미공개인 것은 VCG 라는 이름과 5개 설정 필드(sequencer/slice_mask/slice_thres/time_thres/mode)의 세부값뿐이며, 이는 벤더 레지스터 문서화이지 연구 기여가 아니다.
- 【'유일하게 방어 가능한 공백'에 직접 선행연구가 있다】 Walter, Hannig, Teich, "Loop Control Management in Tightly Coupled Processor Arrays (TCPAs)", arXiv:2603.28645 (2026-03-30). 폴리헤드럴 반복 공간 표현에서 제어 조건을 유도하고, 그것을 **유한 개수의 하드웨어 평가기**(constant/affine inequality evaluator + 설정 가능한 AND/OR)에 낮추며, 제어 신호를 15–45배 줄이고, '쌍별로는 호환되지만 전역적으로 비호환인 제어 조건'이라는 **표현/할당 불가능성**까지 특징짓는다. 이것이 바로 제안서의 살아남는 기여 #1 (아핀 매핑 → 고정 용량 술어 생성기 낮추기의 표현력 경계)과 같은 연구 대상이다. 게다가 이 그룹은 Symbolic Loop Compilation (ACM TECS 2021), Symbolic Mapping of Loop Programs onto Processor Arrays (JSPS 2014), Hierarchical Partitioning for Piecewise Linear Algorithms 로 12년 이상 '아핀 분할 + 경계(border) 셀 제어 생성' 을 해 왔다. 제안서의 5가지 표현 불가 클래스는 전부 단조성/접두 폐포 붕괴의 사례이고, 이 프레임워크의 특수점으로 재기술 가능하다. 리뷰어는 'known framework 의 special case' 로 리젝한다.
- 【VCG 의 유일한 능력 우위 논거가 한 줄로 반박된다】 제안서 기여 #3 과 위험 #4 가 모두 기대는 근거는 '리듀스 앞에 exp 같은 비가역 변환이 오면 exp(p)=0 인 p 가 없어 identity 패딩이 원천 봉쇄된다'(intra-slice-reduce.md:210, book_guide/06:226 확인)이다. 이는 유한 정밀도에서 거짓이다. f32 에서 exp(x) 는 x ≲ −104 에서 정확히 0 으로 언더플로하고, −inf 는 exp(−inf)=0 을 정확히 준다. GPU softmax 와 FlashAttention 의 마스킹이 정확히 이 방법(패딩/마스크 위치를 exp **이전에** −inf 로 채움)을 20년째 쓴다. 즉 identity 패딩은 exp-합에도 적용된다. 이 논거가 죽으면, VE 가 전체 사이클의 0.0%(book_guide/06:446 확인)인 상황에서 VCG 에 남는 것은 성능 이득도 능력 우위도 아닌 '컴파일러가 대신 해 줘서 편하다'뿐이다. 이건 논문 주장이 아니다.
- 【기제의 핵심 동작이 실기에서 한 번도 올바르게 검증된 적이 없다】 book_guide/_evidence/logs/ve_isolated.log 전수 확인: 리듀스 테스트 11건 중 R 을 두 차원에 쪼갠 유일한 커널 `test_ve_intra_slice_reduce_split_time_packet` 은 실기에서 **FAILED (ulp=1)** 이다. 그런데 이 커널의 R=16 은 8의 배수라 packet clipper 가 항상 8 을 반환한다 — **패딩이 아예 발생하지 않는 케이스**다. 실제로 패딩이 걸리는 유일한 배치(`split_slice_time`, S=15 → S#16)는 `cannot reduce pack alias` 로 컴파일 자체가 안 되고 테스트도 없다(book_guide/06:234 확인). 정리하면: VCG 의 존재 이유인 '패딩 제외'가 실기에서 성공적으로 돌아간 사례가 이 리포지토리에 **0건**이다. 제안서는 split_slice_time 만 '테스트 없음'으로 적었고 이 더 심각한 사실을 적지 않았다. 이 상태로 '하드웨어 valid-count 태깅' 논문을 쓰면 리뷰어 한 명이 '패딩된 R 을 실기에서 한 번이라도 맞게 돌린 적 있느냐' 물었을 때 답이 없다.
- 【측정값이 모델 예측일 뿐 아니라, 측정할 도구가 아예 없다】 인용된 모든 수치(DmaEngine 75,464,336 / 96.5% / 470 instr, VectorEngine 14,770 / 0.0% / 50 instr, mnist::forward 17,953 / DMA 12,365 68.9% / VE 1,162)는 `--dump-schedule` 정적 스케줄 모델이다(book_guide/06:445-448, 09:224-228, 11:53-89 전부 실제 확인 — 수치는 정확하다). 유일하게 실시간처럼 보이는 값인 _evidence/logs/npu_matrix.tsv 2열(3710/4281/4354)은 **테스트당 프로세스 wall-clock**(컴파일·로드 포함)이지 커널 시간이 아니다. 즉 커널 단위 타이머도, 에너지 카운터도, tokens/s 도 없다. 그런데 evaluation 은 '실기 wall-clock 과 tokens/J' 를 지표 ③으로 세웠다. 계측기가 없는 성능 지표는 CGO/MICRO/ASPLOS 에서 자동 리젝이다.
- 【MoE 절반은 이미 published 정도가 아니라 정적 형상 가속기 위에서도 published 다】 제안서는 MegaBlocks/vLLM 만 인정했지만 더 나쁘다. TPU 는 XLA 정적 형상 가속기인데, MaxText 의 **Megablox grouped matmul(GMM)** 이 ragged expert 배치에 대해 '패딩 영역 계산을 피하며 ragged 차원 위로 map' 하는 것을 프로덕션에서 하고 있고, **Ragged Paged Attention (arXiv 2604.15464, 2026)** 은 2025년 2월부터 vLLM TPU 백엔드에 들어가 5배 처리량을 보고했다. 즉 '정적 형상 가속기에서 가변 길이/희소 MoE 를 블록으로 다룬다'는 프레이밍은 NPUMoE 이전에 TPU 생태계가 이미 점유했다. 남는 차별점 'branchless 로 제어 흐름 완전 제거'는 O(E²) all-pairs rank(E=128, 토큰당 16,384 비교, mixture-of-experts.md:200-224 확인)로 O(E log K) 선택을 대체하는 것 — 리뷰어에게는 기여가 아니라 **하드웨어 제약이 강요한 비용**으로 읽힌다. 게다가 구현체가 리포에 0건(book_guide/08:372 '미확인 청사진' 확인)이라 그 비용조차 측정 못 한다.
- 【표방한 일반화 대상(varlen attention)이 스펙상 표현 불가능일 가능성이 높고, 제안서도 안다】 vcg.md:356-358 확인: `packet_clipper.valid_size(t)` 는 t 에만 의존하고 slice s 에 의존하지 않으며, 문서 스스로 '이 slice 독립성이 VCG 가 표현할 수 있는 배치를 제한한다'고 적었다. varlen 은 배치 내 시퀀스(= slice)마다 유효 길이가 달라야 한다. 그러면 논문의 결론은 '이 하드웨어 블록은 정작 원하는 워크로드를 못 한다' 가 된다. 그건 한 벤더 레지스터 블록에 대한 부정적 관찰이지 아키텍처 논문의 결과가 아니다.
- 【단일 벤더·단일 칩·설계공간 0차원】 VCG 는 우리가 설계하지 않았고 RTL 도 시뮬레이터도 없다. 'time filter 3개 + clipper 1개' 라는 유일한 설계점만 존재하므로 '왜 4슬롯인가', 'slice 의존을 허용하면 얼마나 좋아지나', '접두 제약을 풀면 무엇이 표현 가능해지나' 에 답할 수단이 원리적으로 없다. 표현력 경계 논문인데 파라미터 공간이 점 하나다. 제안서가 제시한 방어책(추상 술어 생성기 모델에 RVV vl/tail, SVE predicate, TPU 마스킹을 배치)은 **측정 없는 분류표**이므로, 반론을 막는 게 아니라 '분류표만 있고 실험이 없다'는 두 번째 리젝 사유를 추가할 뿐이다.
- 【베이스라인 부재 + 외부 베이스라인의 교란】 리포에 소프트웨어 마스킹 경로도, identity 패딩 경로도 구현체가 없다. 즉 칩 **안**의 대조군이 0개다. 제안서가 붙이려는 외부 기준선(A6000×3 vLLM)은 공정, 메모리 시스템, 전력 봉투가 모두 다르므로 VCG 의 효과를 분리해 주지 못한다 — 오히려 '이건 칩 비교지 기제 평가가 아니다'라는 반론을 부른다.
- 【기여가 연구가 아니라 리버스 엔지니어링 문서화다】 살아남는다고 주장한 두 기여(#1 표현력 경계 형식화, #2 valid-count 전파 대수)의 정당성 증명은 이미 벤더 문서(vcg.md:266-272 SliceMajor 3-case, 349-354 TimeMajor 2-case, 872-878 전파 규칙)에 **완성된 형태로 존재**한다. 제안서 method 스스로 '이미 문서에 있으므로 형식화만 하면 된다'고 적었다. 벤더 산문의 논증을 정리(theorem) 형태로 옮겨 적는 것은 번역이지 결과가 아니다. 게다가 크레이트에 공개 심볼이 없어(book_guide/06:239 확인) 그 산문이 실제 하드웨어와 일치하는지 검증할 수단조차 없고, 같은 책이 f32→f16 캐스트를 '지원'이라 적어 놓고 0.4.0 크레이트에 f16 이 없는 전례가 이미 있다(book_guide/06:470 확인).

**이미 출판되어 신규성이 없는 부분**

- TCP: A Tensor Contraction Processor for AI Workloads (Industrial Product) — ISCA 2024. §IV.D 에 VE 의 'predicated operations', §IV.C 에 fetch 단 padding, §IV.F 에 commit 단 padding 제거, §V.B 에 미정렬 차원의 utilization 손실과 컴파일러 tactic 회피, §VI 에 'dynamic shapes by modifying control registers on the fly'. 제안서가 '전무하다'고 단정한 항목들이 published 다.
- Dominik Walter, Frank Hannig, Jürgen Teich — "Loop Control Management in Tightly Coupled Processor Arrays (TCPAs)", arXiv:2603.28645, 2026. 폴리헤드럴 반복 공간 → 유한 하드웨어 평가기로의 제어 조건 유도·축약(15–45×)과 전역 비호환성 특징화. 제안서의 '살아남는 기여 #1' 과 동일 대상.
- Frank Hannig, Jürgen Teich 외 — Symbolic Loop Compilation for Tightly Coupled Processor Arrays (ACM TECS, 2021) / Symbolic Mapping of Loop Programs onto Processor Arrays (J. Signal Processing Systems, 2014) / Hierarchical Partitioning for Piecewise Linear Algorithms (PARO). 아핀 분할과 경계 셀 제어 생성의 12년 계보 — '아핀 매핑에서 하드웨어 술어를 합성' 은 표준 문제다.
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts — MLSys 2023. 블록 희소 재정식화 + dropless. 제안서도 인정.
- vLLM `moe_align_block_size` / fused MoE — 블록 정렬, expert-token 레이아웃, num_tokens_post_pad. 알고리즘 절차가 동일.
- Megablox / grouped matmul (GMM) on TPU — MaxText 프로덕션(`sparse_matmul=True, megablox=True`). **정적 형상 XLA 가속기** 위에서 ragged expert 배치를 패딩 계산 없이 처리. '정적 형상 가속기 위 MoE 블록 실행' 이라는 각도를 NPUMoE 보다 먼저, 실제 배포 규모로 점유.
- Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU — arXiv:2604.15464, 2026 (vLLM TPU 백엔드에 2025-02 통합, 토큰 처리량 5배). 정적 형상 가속기에서의 varlen/ragged 어텐션 — 제안서의 'varlen 일반화' 각도를 선점.
- TurboGR: An Accelerated Training System for Large-Scale Generative Recommendation — arXiv:2605.13433, 2026. **Ascend NPU** 용 jagged fusion operator(어텐션·RAB), jagged embedding lookup 가속, dynamic jagged load balancing. '정적 형상 NPU 에서 가변 길이 jagged 워크로드의 패딩 낭비 제거'를 실기 학습 시스템으로 완결.
- Fast On-device LLM Inference with NPUs (llm.npu) — ASPLOS 2025. NPU 의 정적 실행 모델을 전제로, 가변 길이 프롬프트를 데이터 의존성을 보존한 채 고정 크기 청크로 재구성해 패딩을 최소화. 문제 정의('NPU 는 형상 불변을 가정하는 정적 실행 모델')가 제안서와 동일.
- Efficient Mixture-of-Experts LLM Inference with Apple Silicon NPUs (NPUMoE) — arXiv:2604.18788, 2026. 제안서도 인정. static tiers for expert capacity, grouped expert execution, 1.32–5.55× 지연 개선.
- The CoRa Tensor Compiler — MLSys 2022. ragged tensor 컴파일과 패딩 최소화.
- Dispatch-Aware Ragged Attention for Pruned Vision Transformers — arXiv:2604.15408, 2026. ragged 버퍼 패킹 + varlen 커널 디스패치.
- RVV vl/vsetvl tail 처리 및 SVE predication — '술어로 꼬리 원소를 무효화' 라는 개념의 기존 구현. RVV 에서 마스크 술어가 unmasked vsetvl 대비 35% 오버헤드라는 수치까지 보고됨(Closer in the Gap, arXiv:2605.10860 등).
- DataMaestro: A Versatile and Efficient Data Streaming Engine ... (arXiv:2504.14091, 2025) 및 AGU descriptor 표현력 계열 연구 — 프로그래머블 아핀 주소 생성 descriptor 가 N-D 접근을 무엇까지 표현하는가. VCG 는 구조적으로 sequencer(AGU) + 비교기이므로 이 계보와 직접 겹친다.

**근거 없이 단정한 문장 (수정 필요)**

- gap (a) 의 'TCP/RNGD 의 공식 발표물(ISCA 2024 + IEEE Micro 2025)을 **실제로 열어 확인한 결과** ... 서술이 전혀 없다' — 같은 제안서의 novelty_confidence 와 related_work 항목이 'ISCA 2024 TCP 원논문 본문은 슬라이드 PDF 가 이미지 기반이라 열람 실패, 미확인' 이라고 적었다. 내부 모순이며, 신규성 주장 전체가 읽지 않은 논문 위에 서 있었다.
- '패딩 제외 리듀스에 대한 서술이 전무하다' — 반증됨. ISCA'24 §IV.D 'predicated operations', §IV.C fetch unit 'type conversion and padding', §IV.F commit unit 'can remove padding from data for compaction during storage', §V.B 정렬/utilization 논의, §VI 'supports dynamic shapes by modifying control registers on the fly' 가 모두 published 다.
- '고정 용량 하드웨어 술어 생성기에 아핀 매핑 표현식을 낮출 때 표현 가능/불가능을 형식화한 연구는 못 찾았다' — Walter/Hannig/Teich, arXiv:2603.28645 (2026) 및 TCPA 계열(TECS 2021, JSPS 2014)이 정확히 그것이다. '못 찾았다'는 검색 실패이지 공백이 아니다.
- 'exp 등 비가역 변환이 리듀스 앞에 오면 identity 원소가 존재하지 않아 소프트웨어 경로가 원천 봉쇄됨' — 유한 정밀도에서 거짓. f32 exp 는 x ≲ −104 에서 0 으로 언더플로하고 exp(−inf)=0. 이 문장은 벤더 문서(intra-slice-reduce.md:210)에서 그대로 옮겨 온 것인데, 옮겨 오기 전에 검증하지 않았다. VCG 의 '유일한 명시적 우위 논거'가 이것이다.
- '하드웨어 valid-count 태깅' 이 실기에서 동작한다는 암묵적 전제 — ve_isolated.log 전수 확인 결과, 패딩이 실제로 발생하는 R 배치가 실기에서 통과한 사례가 0건이다. 유일한 split 커널(R=16)은 패딩이 없는 경우이고 그마저 FAILED(ulp=1) 이며, 패딩이 있는 유일한 배치(S=15→S#16)는 컴파일 불가다. 제안서는 후자만 언급하고 전자의 함의를 적지 않았다.
- evaluation 지표 ③ '실기 wall-clock 과 tokens/J' — 이를 산출할 계측이 리포에 없다. npu_matrix.tsv 의 시간 열은 테스트 프로세스 전체 wall-clock(컴파일·로드 포함)이지 커널 실행 시간이 아니다. 지표를 세우기 전에 계측기부터 없다는 사실이 have_evidence 에 없다.
- 'VE 는 사이클 예산에서 사실상 0(0.0%)' 를 근거로 'VCG 3-way A/B 의 예상 결과가 사이클 차이 ≈ 0' 이라고 추론한 것 — 이 0.0% 는 VCG·패딩 리듀스·MoE 를 하나도 포함하지 않는 벤더 예제 130개 혼합에 대한 값이다. 대상 워크로드가 없는 프로파일로 대상 워크로드의 결과를 예측하는 비논리다(위험 항목으로 적어 둔 것은 정직하나, 논거로 쓸 수는 없다).
- 'valid_size ∈ {0,…,8} 즉 9개 값' 등 VCG 사양 자체 — 문서와 일치함을 확인했으나(vcg.md:30-52), 근거는 `rust,ignore` 의사코드 한 편뿐이고 크레이트에 대응 공개 심볼이 없다. '사양 확정' 이라는 표현은 과하다. 정확히는 '벤더 산문에 기술됨, 실행 가능한 형태로 검증 불가'.
- '책의 MoE 코드는 전부 rust,ignore' — 확인했으나 정확히는 MoE 문서의 rust,ignore 블록은 4개이고 나머지는 코드 블록이 아닌 산문/불릿이다. 사소하나 '전부 의사코드' 라는 표현보다 '실행 가능한 코드가 0줄' 이 정확하다.

**조사자 스스로 적은 위험**

- 기제 (2) MoE 블록 실행은 신규성이 사실상 0 이다. 고정 블록 B, expert별 ceil(Count_e/B), 블록당 expert_id 와 -1 skip, cumsum 기반 정렬/주소 계산은 MegaBlocks(MLSys 2023)와 vLLM 의 moe_align_block_size 커널이 이미 하는 것과 절차까지 동일하다. 리뷰어가 이걸 지적하면 논문의 절반이 날아간다. 게다가 로컬에 구현체조차 없다.
- '이미 GPU 에서 다 한 얘기다' 반론이 강하다. 패딩 낭비 제거는 CoRa(MLSys 2022)가 CPU/GPU 에서 컴파일러로 해결했고, varlen 어텐션은 FlashAttention varlen 이, dropless MoE 는 MegaBlocks 가 해결했다. '정적 형상 NPU 에서 MoE 가 어렵다' 는 프레이밍마저 NPUMoE(2026)가 선점했다. 남는 차별점은 '동적 연산까지 NPU 위에 유지' 하나뿐인데, 그걸 뒷받침할 실행 근거가 없다.
- '단일 벤더·단일 칩이라 일반성이 없다' 반론이 치명적이다. VCG 는 우리가 설계한 것이 아니라 벤더 고정 하드웨어이고, 설계 공간 탐색(술어 슬롯을 4개가 아니라 8개로 하면? slice 의존을 허용하면?)을 할 수 없다. RTL 도 시뮬레이터도 없으므로 '왜 이 설계인가' 에 답할 수 없다.
- 핵심 정량화 질문의 답이 '무의미' 일 가능성이 높다. 스케줄 모델상 VE 는 전체 사이클의 0.0%, DMA 가 96.5% 다. VCG 로 리듀스 연산을 줄여도 wall-clock 이 안 변한다면, 이 기제의 가치는 성능이 아니라 '표현력/정확성'(exp 뒤 identity 패딩 불가) 뿐이며 이는 성능 논문의 지지 근거가 못 된다.
- '하드웨어 valid-count 태깅' 은 개념적으로 새롭지 않다. RVV 의 vl/vsetvl tail 처리와 SVE predicate 이 정확히 같은 일을 수십 년 전부터 한다(RVV 에서 마스크 술어가 unmasked vsetvl 대비 35% 오버헤드라는 수치도 이미 보고됨). 새로운 것은 '아핀 매핑 표현식으로부터 컴파일러가 술어 설정을 합성' 하는 부분과 그 표현력 경계인데, 이는 좁고 벤더 종속적이다.
- 실험 실행 자체가 벤더 컴파일러 버그에 막혀 있다. transformer 4종 ICE, matmul 7종 실기 불가, VCG 의 Slice+Time 분할이 cannot reduce pack alias 로 거부. 필요한 실험 중 상당수가 우리 연구 역량이 아니라 벤더 릴리스에 의존한다 — 논문 일정을 통제할 수 없다.
- 책 문서를 근거로 쓰는 데 따르는 위험: 문서와 실제 크레이트가 이미 어긋난 전례가 있다(책은 f32→f16 캐스트를 지원이라 적지만 0.4.0 크레이트에 f16 타입도 impl 도 없다). VCG 서술도 크레이트에 공개 심볼이 없는 의사코드뿐이므로, 형식화의 근거가 벤더 산문 한 편이라는 점을 리뷰어가 문제 삼을 수 있다.
- 최악 Grid 상한 (T·K−E)/B + E 는 신규 결과가 아니라 ceil(c/B) ≤ (c−1)/B + 1 의 자명한 따름정리다. 정확히는 E + floor((T·K−E)/B) 이며 (T·K−E) 가 B 로 나누어떨어지고 모든 expert 가 B 로 나눈 나머지 1 인 토큰 수를 가질 때만 tight 하다. 논문 기여로 세우면 안 된다.

### 7.9 살리는 길 — 무엇을 바꿔야 하는가

["【결론: 현 thesis 는 버려라. 자산은 하드웨어 기제가 아니라 실패 데이터셋이다.】", "", "1) **논문의 대상을 바꿔라.** 살아남는 것은 contributions[4]('정직한 부산물')뿐이며, 그것이 실은 가장 강하다. 새 thesis: \"상용 정적 형상 텐서 가속기에서 비정형 워크로드를 막는 것은 하드웨어 표현력이 아니라 컴파일러 낮추기다 — 커널 200개 / 실기 89건 / 실패 63건 전수 분류가 보이는 것.\" 근거는 이미 100% 확보돼 있고(13-NPU-실기-매트릭스 §7.1 의 REAL_LOWERING_GAP 24 / INTENTIONAL_NEGATIVE 23 / COMPILER_ICE 13, 실기 80·정상 83 / 89 전수 매트릭스, npu_matrix.tsv 원본 89행 — 내가 전부 대조해 수치 일치 확인) 재현 가능하다. 상용 NPU 툴체인의 실패를 이 해상도로 공개한 논문은 없다. 벤더가 허락 안 하기 때문이다. 이건 진짜 공백이다. 1.5–2개월이면 워크샵/experience 짧은 논문이 된다.", "", "2) **착수 전 2일짜리 반증 실험 두 개를 먼저 돌려라. 둘 중 하나라도 지면 VCG 각도는 폐기다.**", "   (a) exp-합 리듀스를 identity 패딩(-inf 또는 x ≲ −104 로 fetch 마스킹)으로 작성해 VCG 경로와 비트 일치하는지 확인. 일치하면 기여 #3('적용 가능성 차이')은 삭제하고, VCG 의 능력 우위 주장을 논문에서 빼라. 내 판단으로는 일치한다.", "   (b) vcg.md:20-26 의 대표 배치(R=43 # 48 을 Slice /8 · Time /2%4 · Packet %2 로 3분할)를 실제로 작성해 `--backend npu` 로 컴파일하고 실기에서 정답을 내는지 확인. 현재 리포에는 **패딩이 실제로 걸리는 R 배치가 실기에서 통과한 사례가 0건**이다(패딩 없는 R=16 split 은 FAILED, 패딩 있는 S#16 split 은 컴파일 불가). 이게 안 되면 논문의 기제가 존재하지 않는 것이고, 그 사실 자체를 (1)의 데이터셋에 넣어라 — 그게 더 강한 결과다.", "", "3) **표현력 형식화를 굳이 살리려면, 위치를 바꿔라.** 'nobody did this' 는 이제 못 쓴다. Walter/Hannig/Teich 의 제어 조건 유도 프레임워크를 인용하고, 델타를 정확히 진술하라: 그들의 평가기는 아핀 부등식 일반이고 재구성 가능한 반면, VCG 는 비교기 3 + clamp 1 이며 **packet clipper 가 slice 에 의존할 수 없다**는 한 가지 제약이 추가돼 있다. 논문의 질문은 '무엇이 표현 불가한가'(→ 이미 알려진 틀의 사례)가 아니라 **'slice 독립성이라는 이 한 제약이 실제 모델 형상 분포에서 얼마나 비싼가'** 여야 한다. 그건 측정 가능하고(evaluation 지표 ④ 표현 가능 배치 비율), 알려진 틀 안의 새로운 점이 된다. 그래도 워크샵(LATTE/C4ML/EMC²)급이다.", "", "4) **MoE 는 통째로 빼라.** MegaBlocks·vLLM 에 더해 TPU Megablox/GMM 이 정적 형상 가속기에서 프로덕션으로 하고 있다. 기여 0이고 구현도 0이며, 3–4개월을 태워도 리뷰어에게 돌아오는 것은 'O(E²) all-pairs rank 는 왜 쓰나' 라는 질문뿐이다. 꼭 쓰려면 기제가 아니라 **비용 측정**으로만 써라: '분기 없는 하드웨어에서 top-k 라우팅의 대가는 토큰당 16,384회 비교이며 expert GEMM 대비 X%' — 이건 정직한 negative result 다.", "", "5) **varlen 일반화는 주장하지 마라.** packet_clipper.valid_size(t) 의 slice 독립성이 스펙에 박혀 있는 한, 배치 내 시퀀스별 유효 길이는 표현 불가다. 표현 가능함을 증명하기 전에 abstract 에 varlen 을 쓰면 리뷰어가 스펙 한 줄로 리젝한다. 대신 '이 기제가 varlen 에 닿지 못하는 이유'를 형식적으로 보이면 그건 결과다.", "", "6) **계측기를 먼저 만들어라.** `--dump-schedule` 은 모델이고, 같은 문서가 그 모델을 실기와 대조한 적이 없음을 스스로 인정한다. 커널 단위 타이머(또는 반복 실행 기반 추정)와 전력 계측 없이는 성능 문장을 한 줄도 쓰지 마라. 모델 사이클을 결과로 보고하면 그 자체가 리젝 사유다.", "", "7) **투고처 현실화.** 실기 성능 수치가 없는 한 CGO/MICRO/ASPLOS 는 불가다. (1)의 실패 분류 + (2b)의 반증 결과 + (3)의 slice 독립성 비용 측정을 묶어 LATTE/C4ML/EMC² 워크샵 또는 컴파일러 experience 트랙 short paper. 벤더 컴파일러 릴리스에 걸린 실험(transformer 4종 ICE, matmul 7종, pack alias)에 논문 일정을 걸지 마라 — 통제 밖이다.", "", "8) **책 문서를 1차 근거로 쓰는 습관을 끊어라.** 근거 검증에서 제안서의 수치는 거의 전부 파일과 일치했다(사이클, 실기 매트릭스, VCG 사양, 전파 규칙, 표현 불가 5종 — 모두 대조 완료). 문제는 정확도가 아니라 **층위**다: 벤더 산문에서 옮긴 명제(exp identity 원소 부재)가 사실이 아니었고, 벤더 산문이 크레이트와 어긋난 전례(f16)도 이미 있다. 논문에 들어가는 모든 벤더 명제는 실행 또는 반례로 한 번 통과시켜라."]

### 7.10 후보 학회와 소요 기간

- 현 상태로는 독립 투고 부적합. 진행 중인 '정적 형상 텐서 가속기 프로그래밍' 큰 논문의 한 절(표현력 한계 절)로 넣는 것이 최선
- 표현력 형식화만 떼면: LATTE / C4ML / EMC² 등 컴파일러·ML시스템 워크샵 (실측 없이도 받아들여지는 유일한 급)
- 실측이 붙고 varlen 일반화가 성립하면: CGO 또는 PACT (컴파일러 낮추기 + 표현력 경계 각도)
- MoE·varlen 엔드투엔드 실기 수치가 A6000 기준선과 함께 나오면: MICRO / ASPLOS 의 경험 논문(experience/case-study) 트랙 — 단 단일 벤더 일반성 반론을 추상 술어 모델로 막았을 때만
- MLSys — 단, MegaBlocks·NPUMoE 와 직접 비교당하는 자리이므로 MoE 각도로는 권하지 않음

**소요 기간 추정**: weak → viable 로 올리는 데 6–9개월(VCG 3-way A/B 와 표현 가능 비율 측정에 1–2개월, MoE 커널 신규 작성·실기 안착에 3–4개월, varlen 일반화 검증과 A6000 기준선 정렬에 2–3개월). 단 이 일정은 우리 통제 밖이다 — transformer 4종 ICE, matmul 7종 실기 불가, VCG Slice+Time 분할이 cannot reduce pack alias 로 거부되는 상태라 상당수 실험이 벤더 컴파일러 릴리스에 걸려 있다. 벤더 수정이 안 되면 무기한. 반면 '표현력 형식화 + 실기 낮추기 실패 63건 분류' 만으로 워크샵 페이퍼를 쓰는 축소 경로는 이미 확보된 근거로 1.5–2개월이면 가능하며, 현실적으로 이쪽을 권한다.

**신규성 확신도**: 중간. VCG 라는 기제 자체가 미공개라는 점은 **확인했다** — TCP/RNGD 의 IEEE Micro 2025 기고 PDF 전문을 직접 읽어 VCG·valid_size·time filter·packet clipper·MoE 블록 실행 언급이 없음을 확인했고, 'valid count/valid size flit tagging' 키워드 검색으로도 대응 논문을 찾지 못했다. 다만 이는 '아무도 안 했다' 가 아니라 '이 벤더가 안 썼다' 에 가깝다. 개념 층위에서는 RVV vl/tail 처리와 SVE predicate 이 같은 일을 하고, 컴파일러가 타일 경계에서 술어를 합성하는 것도 tile-based 컴파일러의 표준 문제다(검색에서 'control-flow synthesis 가 부분 타일에 대해 일관성 없는 술어를 만든다' 는 버그 연구가 나왔다). 즉 신규한 것은 기제가 아니라 **고정 용량 술어 생성기에 대한 표현력 경계의 형식화**이며, 이 좁은 각도로도 정확히 일치하는 선행연구는 못 찾았다. 반대로 기제 (2) MoE 블록 실행은 **이미 published 라고 확신한다** — MegaBlocks(MLSys 2023)와 vLLM moe_align_block_size 가 블록 정렬·expert_id per block(-1 skip)·cumsum 주소 계산까지 동일하며, 여기에는 신규성이 없다. 확인 못 한 항목: ISCA 2024 TCP 원논문 본문(슬라이드 PDF 가 이미지 기반이라 열람 실패) — 여기에 VCG 가 이미 서술돼 있으면 남은 신규성마저 사라진다. 이것을 먼저 확인하기 전에는 신규성 확신을 '중간' 이상으로 올릴 수 없다.

### 7.11 검색으로 확인한 관련 연구

| 제목 | 학회·연도 | 이 주제와의 관계 |
|---|---|---|
| The CoRa Tensor Compiler: Compilation for Ragged Tensors with Minimal Padding | MLSys 2022 (arXiv 2110.10221) | 겹침 — 이 주제의 전제('패딩은 낭비이고 컴파일러가 제거해야 한다')를 정면으로 다룬 논문. 다만 CPU/GPU 대상 소프트웨어 컴파일러이고, 하드웨어 술어 생성기 낮추기는 다루지 않는다. PyTorch 대비 인코더 1.6x, ARM CPU MHA 1.86x geomean. |
| MegaBlocks: Efficient Sparse Training with Mixture-of-Experts | MLSys 2023 | 겹침 — 기제 (2)와 직접 경쟁. MoE 를 블록 희소 연산으로 재정식화하고 expert 당 가변 토큰 수를 블록 희소 GEMM 으로 처리하는 dropless MoE. Tutel 대비 최대 40%, Megatron-LM 대비 2.4x. '토큰 드롭 vs 패딩 낭비' 트레이드오프를 없앤 것이 기여인데, 이는 본 주제의 블록 단위 실행이 주장하려는 바와 같다. |
| Efficient Mixture-of-Experts LLM Inference with Apple Silicon NPUs (NPUMoE) | arXiv 2604.18788, 2026 | 겹침 — 프레이밍이 거의 동일. '전문가 라우팅이 동적 텐서 형상을 만들어 NPU 의 형상 고정 제약과 충돌하고, top-k/scatter/gather 가 NPU 비친화적이며, 작은 전문가 커널 다발이 디스패치 오버헤드를 만든다' 라는 문제 정의가 본 주제와 같다. 차이점은 NPUMoE 가 동적 연산을 CPU/GPU 로 폴백시키는 반면 본 주제는 NPU 위에 유지한다는 것 — 여기가 유일한 차별점이므로 반드시 실측으로 방어해야 한다. |
| SoD²: Statically Optimizing Dynamic Deep Neural Network Execution | ASPLOS 2024 | 인접 — 동적 DNN 실행을 정적으로 최적화하는 프레임워크. 동적 형상 문제를 컴파일 시점으로 밀어내는 접근이라 본 주제의 '정적 형상 제약 하 동적 워크로드' 와 문제 공간을 공유하나, 하드웨어 술어 기제는 다루지 않는다. |
| DietCode / Nimble 계열 dynamic-shape 컴파일 (MoonPoly 의 비교 대상으로 확인) | MoonPoly, ACM TACO 2025 (Nimble MLSys 2021 / DietCode MLSys 2022 를 baseline 으로 명시) | 기반 — DietCode 는 정적으로 알려진 범위 내 동적 차원까지 auto-scheduling 확장, Nimble 은 동적 신경망 컴파일. 둘 다 동적 차원마다 범위를 요구해 임의 형상은 못 다룬다는 한계가 보고됨. 본 주제는 '범위 대신 하드웨어 술어' 라는 다른 축이므로 직접 경쟁은 아니고 배경. |
| FuriosaAI RNGD: A Tensor Contraction Processor for Sustainable AI Computing | IEEE Micro (Hot Chips 2024 theme article), 2025 | 기반 — 대상 하드웨어의 공식 문헌. PDF 전문을 확인한 결과 VE 를 'intra-slice/inter-slice reduction, slice 당 8-way throughput' 수준으로만 기술하고 VCG·valid_size·time filter·packet clipper·MoE 블록 실행은 **전혀 언급하지 않는다**. 신규성 주장의 근거이자, 동시에 '벤더가 논문화하지 않은 이유' 를 스스로 물어야 할 지점. |
| TCP: A Tensor Contraction Processor for AI Workloads | ISCA 2024 | 기반 — TCP 아키텍처 원논문(H. Kim et al., IEEE Micro 2025 기고의 참고문헌 7번으로 서지 확인). 공개 슬라이드는 이미지 기반이라 본문 확인 실패 — VCG 관련 서술 유무는 **미확인**이며, 논문 착수 전 반드시 원문을 구해 확인해야 한다. |
| FlashAttention varlen (flash_attn_varlen_func) 의 가변 길이 처리와 부하 불균형 | FlashAttention 계열, 2022–2026 (기술 문서/블로그로 확인, 정식 논문 서지는 미확인) | 인접 — 패딩 입력 대비 최대 90% 빠르다는 보고와, causal/varlen 에서 워크타일 길이 차이로 SM 부하 불균형이 생긴다는 알려진 문제. GPU 측 varlen 해법의 현재 수준이며 본 주제의 varlen 일반화 주장이 넘어야 할 기준선. 정식 논문 인용은 원문 확인 후 붙일 것. |
---

## 11. 전 주제 공통 위험

심사 77건을 관통하는 반론 다섯 가지다. 어떤 주제를 고르든 이건 반드시 답해야 한다.

### 11.1 단일 벤더·단일 칩

가장 자주, 가장 세게 나온 반론이다. RNGD 하나에서 관찰한 것을 일반 결론처럼 쓰면 즉시 reject 된다.

**대응 세 가지 중 하나는 반드시 해야 한다.**
- 두 번째 스택을 붙인다 (같은 하드웨어의 TCL 경로면 1.5개월, AMD XDNA·Ascend·IREE/TVM-BYOC 면 2~3개월)
- 제약을 **추상 술어**로 올려 다른 가속기(CuTe well-definedness, NKI 타일 제약)에 인스턴스화해 보인다
- 분모를 키운다 (벤더 예제 200개 → 생성된 매핑 수만 개)

### 11.2 사전출시(alpha) 소프트웨어를 평가하는 것의 공정성

책 자신이 첫 장에서 "Alpha Test Build: Experimental Software" 라고 경고한다.
"미완성 소프트웨어의 버그를 세어 논문을 썼다"는 인상은 치명적이다.

**대응**: 발견한 결함은 논문 제출 전에 **상류에 이슈로 제기하고 응답을 받아라.**
confirm 이 하나라도 붙으면 "우리가 트집 잡은 것"이 아니라 "실제 문제였다"가 된다.
또한 버전을 고정해 명시하고, "이 시점의 스냅샷"임을 초록에 적어라.

### 11.3 ★ 벽시계 계측이 없다

**이 문서에서 가장 중요한 공백이다.** 우리가 가진 성능 수치는 전부 컴파일러 스케줄 모델 예측이다.
실제 실행 시간을 잰 적이 없다. 그래서

- 주제 2는 "무엇으로 검증할 것인가"가 미해결이고 (반복 호출 API 또는 온칩 타이머 필요)
- 주제 4는 전제가 미검증이고
- 주제 3·5도 성능을 말하는 순간 같은 벽에 부딪힌다

**이건 4주짜리 go/no-go 게이트다.** 계측 수단이 없다면 성능 계열 주제는 전부 접고
정확성·표현력 계열(주제 1·3·6)로 가야 한다.

### 11.4 실험이 벤더 릴리스에 걸려 있다

주제 7이 대표적이다. transformer 예제 4종이 컴파일러 ICE 로 죽고, matmul 7종이 실기로 못 가고,
VCG 의 Slice+Time 분할이 `cannot reduce pack alias` 로 거부된다.
**우리가 아무리 노력해도 벤더가 고치기 전에는 그 실험을 못 한다.**
일정에 우리 통제 밖 구간이 있는 주제는 우선순위를 낮춰야 한다.

### 11.5 벤더가 이미 아키텍처 논문을 갖고 있다

§1 참고. 아키텍처 각도로 가면 ISCA 2024 / IEEE Micro 와 정면으로 겹친다.
**프로그래밍 모델·툴체인·검증 각도로만 가라.**

---

## 12. 지금 당장 할 일 (순서대로)

| 순서 | 할 일 | 기간 | 하드웨어 | 왜 먼저인가 |
|:--:|---|---|:--:|---|
| 1 | `summary.log` 의 설명 안 되는 **2,500 사이클** 전수 회귀 — 커널 130개에서 같은 gap 이 어떤 구조로 나타나는지 | 2주 | 불필요 | 주제 2의 착수점이자 가장 싼 실험. 지금 바로 된다 |
| 2 | **벽시계 계측 수단 확인 (go/no-go)** — 반복 호출 API 나 온칩 타이머가 있는가 | 4주 | 필요 | 없으면 성능 계열 주제 3개가 동시에 죽는다. 가장 먼저 알아야 할 사실 |
| 3 | `GLOSSARY.md` · `CHEATSHEET.md` · `curriculum/` 의 **Skew·Sliding 기술 정정**, `Broadcast` 추가 | 1일 | 불필요 | §2 참고. 틀린 채로 두면 계속 잘못 배운다 |
| 4 | 발견한 결함 **상류 이슈 제기** (로더 abort 3, 조용한 오배치 2, hang 1, ICE 13) | 착수 즉시 · 응답 1~3개월 | 불필요 | §11.2. 응답 대기가 기니 일찍 시작해야 한다 |
| 5 | **워크숍 논문 초고** (권장 B) — 이미 있는 근거만으로 | 1.5~2개월 | 불필요 | 조기 피드백. 권장 A 의 뼈대가 된다 |
| 6 | 두 번째 컴파일러 스택 전수 매트릭스 | 1.5~3개월 | 필요 | §11.1. 권장 A 의 최대 관문 |

1·3번은 **오늘 시작할 수 있고 하드웨어도 필요 없다.**

---

## 부록 A. 조사 원본

| 자산 | 위치 |
|---|---|
| 조사·심사 원본 JSON (7주제 × 2단계, 358 KB) | `~/.claude/jobs/46bc5c7e/tmp/research_topics_raw.json` ※ job 임시 경로 — 회수되면 사라진다 |
| 근거가 된 실측 문서 | [`book_guide/`](./book_guide/) 17편 |
| 원본 로그·재현 스크립트 | [`book_guide/_evidence/`](./book_guide/_evidence/) |
| 책 원문 한국어판 | [`book_ko/`](./book_ko/) 46개 절 |
| 용어 | [`GLOSSARY.md`](./GLOSSARY.md) — 단 §2 의 정정 사항 반영 전 |

## 부록 B. 이 문서가 하지 않은 것

- **선행연구 검색은 완전하지 않다.** 웹 검색으로 확인한 75건뿐이고, 유료 DB(ACM DL, IEEE Xplore 전문)는
  못 봤다. "신규성 있음" 판정은 전부 잠정이다.
- **어떤 실험도 새로 하지 않았다.** 전부 기존 `book_guide/` 근거와 소스 대조, 웹 검색으로만 판단했다.
  예외는 §2 의 Skew·Sliding 소스 확인 하나다(이번에 직접 했다).
- **주제를 실제로 착수해 보지 않았다.** 기간 추정은 조사 담당의 추정이며 검증되지 않았다.
