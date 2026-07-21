# ACCV 작업 현재 상태 (두 세션 공유용 스냅샷)

> 이 파일은 지금 병행 중인 두 클로드 세션이 상태를 공유하려고 두는 **살아있는 스냅샷**입니다(자동 로드 아님 — 세션에 "이 파일 읽어줘"라고 해야 반영됩니다). **규약:** ① 작업 시작 전 읽는다 ② 상태가 바뀌면 해당 줄을 **덮어써 갱신**(길게 쌓지 말 것). 상세 이력은 메모리 `accv2026-paper-plan.md`.
> **마지막 갱신: 2026-07-04 (02d85653: ★★★**train-갤러리를 헤드라인으로 승격 + pitome_reg 일반성 통합 = 대규모 재구조화(사용자 결정)**. 표준 DINOv2 프로토콜(gallery=train 1.28M, 무압축 **80.87≈공인 82**)이 주 결과; val-LOO는 일관성 확인으로 강등(**sec:canonical→sec:consistency, tab:canonical→tab:consistency=val-LOO 3자**). **전 분류표 train화**: tab:main(Δ+0.9→+7.3), tab:pitome(O−P +0.6→+6.2), tab:ablation(대안 ±0.5 무이득), tab:aux(reg-count train CI @92%[+7.0,+7.6]·mAP는 val). ★**신규 tab:generality+§일반성**: PiToMe 병합 위에 레지스터 보호(``PiToMe+reg'') reg이득 +0.48→+5.07 전구간>0 = 병합기 무관 keep-rule = PiToMe와 나란한 포지셔닝 캡스톤. 초록·서론·결론·한계·효율·fig캡션 train화. 그림3종 train 재렌더·배치(result 우측=reg스윕 70→77.3). **KO+EN 완료·검증(양파일 9/9표+tab:generality·4/4그림·dangling0)**. +2회 검토워크플로(수치/톤/과대해석 + 실험별 why/how/result/의의): ★**'merger-agnostic/병합기 무관' 과대해석을 '시험한 두 병합기(ToMe·PiToMe)에서 성립'으로 스코프**(초록·서론·제목·본문·동기문장 한/영 전부), '정확도 유지'→'하락 작게 유지', dense 확증→뒷받침·'이 밀집설정'·무보호'이하', consistency '같은데 더큼' 모순 수정, ablation 흐름 정리. EN 초록 문법·중복문 정리. 출처: `pitome_compare/results_pitome_reg_train_faithful.txt`·`canonical/faithful_results/canonical_faithful_base.txt`·`ablation/results/ablation_train_faithful_base.txt`·`robustness_50k/reg_count_train_faithful_base.txt`. ⚠️**그림·본문 재구조화 진행중 — 408은 main.md/main_en.md/fig 손대기 전 이 줄 필독**. 이하는 이전 이력(그림 v2~v4·검토5회):**
> (이전) **그림 4종 v2 재작업 + 3회 검토 워크플로우 완료**(사용자 /loop·ultracode). **v2 개선**: ① 글자 대폭 확대(1000px 캔버스가 지면 폭 331pt로 줄면 12px→~4pt였음 → 축·라벨 ~2배로, method "너무 작다" 해결) ② fig_result 왼쪽에 **PiToMe 3번째 선 추가**(새 실험 반영, 캡션 한/영 갱신) + Δ축 단위 "points"→**"percentage points"** ③ 겹치는 선은 끝-라벨→**범례**로(pitome 처리량 패널, result·pitome 전반) ④ fig_ablation 범례로 "non-register keep-rules(random·energy·high-norm·ToMe)" 명시. **v4 사용자 후속수정**: ⑤ method 정확도 pill(62.4/72.6)·괄호(proportional 등)·수치주석 제거·"→ MLP" 줄바꿈 ⑥ pitome·result 범례/주석에 **흰 배경 플레이트**(격자선-글자 겹침 해결) ⑦ 범례에 "PiToMe (energy)" 괄호설명 추가 ⑧ ★**fig_result 오른쪽 패널 = 음성대조군 Δ(no-reg Δ≈0이 구성상 자명) → 레지스터 개수 스윕으로 교체**(k0=ToMe 62.42→k4=Ours 72.63, 95%CI[+9.8,+10.6], ∼91%; 사용자 결정. 캡션 한/영 갱신, 음성대조군은 tab:control로만 유지). **energy→pitome 개명은 사용자 확인 후 '유지'로 결정**(정적 프록시라 실제 PiToMe 64.47과 값·정의 다름, 개명 시 tab:pitome 모순+PiToMe 비하). **검토 5회 완료**(병렬감사+적대검증 워크플로): P1=KO/EN 패리티 1건, P2=0건, P3=**5건**(★dense 과장·동적재선택 하네스구분·EN초록 문법·ViT-S 텐션), P4=**2건**(★★**tab:ablation 캡션 과장 = dense와 같은 부류**: "레지스터만 베이스라인 넘고 나머지 노이즈 바닥"인데 실제 energy/high-norm이 55/74/83/91%서 ToMe를 +0.15~0.43 넘음 + ±0.4를 tab:control ≤0.05 노이즈와 혼동 → tab:ablation캡션·fig캡션·초록·본문 한/영 6곳 "레지스터만 큰 폭(압축확대) 이득, 대안은 ±0.4 근처 계통적이득 없음"으로 교정 / 공식식→공식 오타), P5=**0건 수렴**(전 표·그림캡션 재sweep, 적대검증서 후보 전부 기각) → **검토 종료**. ★교훈: "only/every/노이즈바닥" 류 보편수식어는 표 전행 대조 필수(P3·P4서 dense·ablation 두 캡션 과장 적발). 최종 구조 8/8표·4/4그림·dangling0. **실험_방법_정리.md 검토**: 수치 전수 로그대조(exp1-12) — exp8 canonical %라벨 91.2→**92.0 교정**+Ours−ToMe(base +7.28/small +4.37)로 정리, 군더더기 제거(PiToMe 중복설명 통합·메타 2줄 삭제). 구조 8/8표·4/4그림·dangling0 재확인. figbuild PNG↔ACCV md5 동일. ⚠️**그림은 원래 408 담당이나 사용자 지시로 02d85653 인수** — 408은 fig_*.pdf/png·main.md 그림블록·그림캡션 건드리기 전 이 줄 확인. 작업물 `jobs/02d85653/tmp/figbuild/`. 이전: faithful 10/10·톤 중립화 완료. 408: tab:ablation·aux train 열. 상세 §faithful)**

## 세션 분담 (충돌 방지) — 2026-07-03: **02d85653 main.md/main_en.md faithful 전면 편집 ✅ 완료(lock 해제) → 408 재개 가능**
> ✅ **완료(02d85653, 2026-07-03)**: main.md·main_en.md 6표(tab:main·pitome·ablation·aux·dense·canonical) + 초록·서론·방법·PiToMe절·ablation절·aux절·dense절·canonical절·한계·결론을 **faithful 수치로 전면 교체**. 통제-harness 설명 삭제, PiToMe "전구간 우세"로 플립(교차점·"온건 PiToMe 우세" 제거), aux의 "정식 ToMe" 행/문단 제거(이제 tab:main이 곧 faithful ToMe), retrieval within-set 각주 추가, 헤드라인 92%→91%(faithful comp), canonical은 train이라 92% 유지. 양 파일 LaTeX 8/8표·3/3그림·전 ref 해결·통제수치 0 검증.
> ⚠️ **남은 것**: ① train ablation·reg-count 도착 시 tab:ablation·aux에 train 열 추가(현재 val-LOO faithful). ② eval_v2 파생(다중인코더 S/L·선형프로브·동적재선택)은 통제 유지 → 본문에 "보조 3-seed/통제 하네스" 라벨 부착함(faithful화 미결정). 408이 문장 다듬을 때 이 라벨 유지.

- ✅ **02d85653의 main.md 참고문헌 배치 완료(2026-07-02)** — pool 32편 전부 관련연구·설정에 배치, 인용 15→**47/47**(전 bib 항목 배치·미해결 0). 02d85653은 현재 main.md 미편집 → **408 재개 가능**(이 파일 확인 후).
- **408e1fd3**: GPU 정통(canonical) 결과 감시 + dense 재실행 반영 + 그림(`ACCV/fig_*.png`). `canonical/`·`dense/`. (main.md 구조편집은 완료 상태)
- **02d85653**: `main.bib`·참고문헌 배치(main.md 인용)·`All_result.md`·PPT·개념 Q&A. GPU 실행·전송 안 함(사용자 몫).
- 같은 줄을 둘이 동시에 고칠 땐 각자 소유 영역만 덮어쓰고, 상대 영역은 건드리기 전 이 파일 재확인.

## 참고문헌 (02d85653 담당, 2026-07-02)
- **main.bib 6→47편** 확충. 신규 41편 전부 1차 출처(arXiv/OpenReview/DBLP/공식 proceedings) 웹검증 완료(발굴+검증 워크플로, 확정 41 / 거부 0). 주제 5구획(토큰축소·레지스터/이상치·파운데이션인코더·효율어텐션·데이터셋) 정리. 중복키 0·인용해결 100%.
- **`ACCV/참고문헌_지도.md` 신규**: 47편 각각이 뒷받침하는 주장·섹션 + 연결됨/pool 상태 + 배치 우선순위 안내. 408이 본문 재구성 시 pool 32편 배치용 지도.
- **현재 인용된 15키**: `tome·pitome·registers·massive·fna·regcache`(기존) + `dynamicvit·evit·vit·dinov2·dino·clip·siglip·imagenet·ade20k`(신규 연결). main.md의 모든 `\cite`가 bib에서 해결됨(동시편집 후 재검증, 미해결 0).
- ⚠️ **main.md 인라인 인용 9곳(위 신규 키)은 02d85653이 추가**했고 408의 `tab:positioning`과 충돌 없이 통합됨. **main.md 소유는 408**이므로 유지·이동·삭제는 408 재량(지도 참고). **02d85653은 이후 main.md 안 건드림 — main.bib·참고문헌_지도.md만 유지.**
- ✅ **갭 해소: 인용 47/47** — 참고문헌_배치_제안 블록 전부 반영(관련연구 §토큰축소 +10, §레지스터/이상치 +9, §효율 +5, 설정·데이터셋 +8). 모든 `\cite`가 bib에서 해결됨(재검증, 깨진 cite 0). ⚠️ LaTeX 툴체인 이 환경 미설치라 **실제 컴파일 검증은 사용자 환경 필요**. `참고문헌_배치_제안.md`·`참고문헌_지도.md`는 이력으로 보관.

## 영어본 상태 (2026-07-02) — ✅ 동기화 완료
- ✅ **`main_en.md` 신규 = 현재 영어본**, 한국어 `main.md`(2026-07-02)와 **완전 대등**: 47인용 동일·표 수치 동일·구조 6절/7표/4그림/18라벨 동일·`\cite`·`\ref` 전부 해결(결정적 검증 통과). bib 공유(`\bibliography{main}`).
- `main_en_backup.md`는 2026-07-01 옛 스냅샷으로 보존(상단 STALE 배너). 동기화 이력·체크리스트 = `영어본_동기화_상태.md`.
- ⚠️ **이후 유지**: 정통·dense 결과 도착 시 한국어 main.md와 **함께 main_en.md도** 갱신(현재 대등). LaTeX 툴체인 미설치라 실제 컴파일은 사용자 환경 필요.

## 구조 감사 결과 (408e1fd3, ACCV2024 관례 대조 — 2026-07-02)
판정: **근본 재구성 불필요**. 뼈대(서론→관련연구→방법→실험→결론, main-before-ablation)는 ACCV/LNCS 관례 부합(확인 ACCV2024 exemplar=CNN-MoD arXiv:2409.17016 + VoMix 2408.17062 + DTEM 2412.10569). surgical 보완만.
- ⚠️ **teaser: 논문 본문 미포함 확정(2026-07-02 사용자)** — `main.md`·`main_en.md`에서 fig_teaser 그림블록+참조 **제거**(양 파일 begin/end 20/20·dangling 0 검증). 파일 `ACCV/fig_teaser.png`·PPT 덱은 발표용으로 보관(논문엔 안 씀).
- 📌 **제출: 오늘 마감 = 50k 버전으로 초록 등록**(정통 base 결과 대기 안 함). 제출본=현 main.md/main_en.md(val-LOO 50k, teaser 없음, 인용 47/47). 정통·dense는 등록 후 반영(camera-ready/검증).
- ✅ **DONE(408e1fd3, main.md 담당)**: 위 구조 편집 전부 main.md에 반영·검증(LaTeX 21/21·dangling ref 0):
  1. teaser Fig.1 삽입(`fig:teaser`, 서론 최상단) + 본문 참조.
  2. `적용 범위와 한계`를 `\subsection`→`\section`(sec:limitations)으로 승격(실험↔결론 사이 독립 절).
  3. 기여 5→4 bullet(진단/방법/결과/경쟁자·인과, PiToMe bullet 기여화·극단 스코프 보존).
  4. 방법에 보호집합 공식표기 $\mathcal{P}=\{\text{CLS}\}\cup\{\text{레지스터}\}$.
  5. 관련연구에 positioning 표(`tab:positioning`, ToMe/DynamicViT/EViT/PiToMe/RegCache/FNA/Ours × 4속성, bib키 검증).
- SKIP(관례상 불필요): 실험 절 재정렬·Datasets/Impl 전용절·appendix·broader-impact.
- ✅ **DONE(408e1fd3): 4개 그림 PPT 네이티브 덱** `ACCV/ACCV_그림_네이티브.pptx`(빌더 `build_figures_ppt.py`) — teaser=도형, 주결과·PiToMe·ablation=python-pptx 네이티브 라인차트(데이터 내장, PNG 아님), 색 Okabe-Ito 통일. soffice PDF 4슬라이드 눈검증 완료. **02d85653의 `build_accv_ppt.py`는 안 건드림**(별도 파일).

## ACCV 관례 정합 패스 (02d85653 · 2026-07-03, accv2024 19편 워크플로 분석 기반)
- **판정**: 근본 재구성 불필요, surgical만. 이미 잘 지킨 강점 확인(섹션순서 19/19 정합·CI 희소관례(19편중 2편)·음성대조군·3-way 직접비교(SOTA표 억지 금지 권장 부합)·한계 별도섹션·our-position 문장·기여목록).
- **✅ 적용(양 파일 main.md·main_en.md)**: ① **방법 개념도 신설 `fig_method.png`**(PIL 생성·Read 눈검증, ToMe=레지스터 소멸→62.4% vs Ours=보호(파란테두리)→72.6% 도식)를 방법 섹션에 삽입→**Fig.1**(18/19편 method 다이어그램 관례 충족 + 앞쪽 개념도). ② 실험 소절 질문형 제목("레지스터가 실제로 특별한가?(음성대조군)"·"왜 레지스터를 보호하는가?(ablation)"). ③ 메커니즘+효율을 "심화 분석" 아래 볼드 run-in 통합(subsection 11→9). ④ 실험설정에 "데이터셋과 지표"·"구현 세부" run-in. ⑤ tab:pitome 방향표시(GFLOPs↓)+차선 밑줄(PiToMe)+"최고 볼드 차선 밑줄" 캡션. ⑥ tab:aux 캡션 부트스트랩 B=2000 명기. ⑦ Figure~→Fig.~ 통일. ⑧ 관련연구 1문단 our-position 문장 추가. ⑨ (정합보정) 영어 efficiency 92→91.
- **✅ 추가 사용자 지시(2026-07-03) 반영**: **NPU/온칩 언급 전부 제거**(초록·효율절·결론, 양 파일 — 효율절 제목도 "FLOP 절감과 GPU 처리량"으로, "측정 지연이 아니라 FLOP" 폐기). **"코드 게재시 공개" 문장 철회**(accepted 19편엔 실제 GitHub 링크 9편·"released after publication/soon" 2편뿐이라 투고본 표현 부적절 + 공개약속은 사용자 결정사항이라 뺌).
- **⚠️ 미적용(의도적)**: **티저 Fig.1 재삽입 안 함** — 2026-07-02 사용자 "본문 미포함 확정" 존중. 단 방법 Fig.1이 앞쪽 개념도 역할 겸함. (fig_teaser.png는 존재 → 사용자 원하면 2줄 재삽입 가능, 권고로 남김). en-dash 마커(low)는 미적용.
- 검증: 양 파일 8/8표·4/4그림·전 ref 해결·통제수치 0 회귀. 도구: `jobs/02d85653/tmp/make_fig_method.py`, 워크플로 결과 `.../tasks/w6ivoolx9.output`.

## ⚠️ 프레이밍 전환: "이겼다" → "빠진 재료" (02d85653 · 2026-07-03, 사용자 지시)
- **사유(리뷰 위험)**: 남의 기법(PiToMe)을 재이식해 "우리가 이겼다"는 수사는 공정성 공격·SOTA레이스 오해를 부름. 우리 진짜 기여=진단+간단한 플러그인.
- **적용(main.md·main_en.md·실험_방법_정리.md 전부)**: "레지스터 보호가 PiToMe를 전 구간 능가" → **"PiToMe도 레지스터를 명시적으로 보호하지 않아 극단압축서 무너지며, 같은 하네스에 레지스터 보호를 더하면 그 붕괴를 막는다"**. 초록·기여bullet·PiToMe절·ablation·결론·canonical·fig캡션·방법절 모두. **표·수치는 그대로**(Δ +8.16 등 유지), 수사만 낮춤.
- **핵심 문장 추가**: "우리는 새 병합 알고리즘이 아니라 강한 경쟁자도 놓친 레지스터 보호(=어떤 병합기에도 얹는 직교 keep-prior)를 제안한다"(방법절·PiToMe절 결론). "직접 대결/head-to-head/surpasses/능가" 표현 제거.
- **2차 중립화(사용자 추가 지시)**: 경쟁자 결함 서술도 위험 → **"PiToMe가 무너진다/못한다"(경쟁자 주어 부정문) 전부 제거**, "**레지스터 보호를 더하면 격차가 커진다**"(우리 방법 주어 긍정문)로. PiToMe는 "가장 강한 학습 없는 베이스라인"으로만 중립 언급. 표의 "레지스터 구조 이용 ✗"는 positioning표처럼 중립 사실이라 유지. "무보호 ToMe/베이스라인 하락"은 우리 ablation이라 정당·유지.
- ⚠️ 아래 LOCKED "주제"의 "Ours가 PiToMe 전 구간 우세" 서술도 이 톤(우리 방법 주어·붕괴/실패 단어 금지)으로 읽을 것. 데이터·표·수치 전부 불변, 프레이밍만 완화.

## 포지셔닝 확정: "ToMe 프레임워크의 새 keep-rule"(PiToMe와 나란함) — 02d85653 · 2026-07-03
- **사실 확인**: ToMe(Bolya 등, Meta, ICLR'23)와 PiToMe(Tran 등, hchautran, NeurIPS'24)는 **다른 저자**. PiToMe는 **ToMe의 후속**(이름 참조 + 코드상 앞절반층 `pitome_bsm`=ToMe 이분소프트매칭 그대로 재사용 + `merge_wavg` size가중, 거기에 에너지 selection만 추가). 즉 PiToMe=**"ToMe 병합 프레임워크 + 무엇을 합칠지 새 규칙(에너지)"**.
- **우리 포지셔닝(정직·선례 있음)**: Ours=**"ToMe 병합 프레임워크 + 무엇을 지킬지 새 규칙(레지스터 keep-rule)"** = PiToMe와 **완전히 같은 형태**. "남의 기법에 더한 약점"이 아니라 이 분야 표준 기여형(선례=PiToMe NeurIPS'24). 병합 엔진 표준 유지=통제비교 강점.
- **적용(main.md·main_en.md)**: 관련연구 토큰축소 문단을 "이 병합 프레임워크에서 핵심문제=어떤 토큰 합치고 지킬지; PiToMe는 ToMe 매칭 그대로+에너지 규칙; 우리는 같은 프레임워크에 레지스터 규칙"으로 재구성. 방법·PiToMe절의 **"어떤 병합기에도 얹는 직교 플러그인(any merger)" 과장 삭제** → "ToMe 프레임워크 안의 keep-rule(PiToMe와 같은 자리)"로. ⚠️ 우리 기여=레지스터 keep-rule은 **ToMe 위에서만 실증**(PiToMe+보호 미실험)이라 "범용 플러그인" 주장 금지.
- (선택·미결정) PiToMe+레지스터보호 실험(코드상 pitome n_protect 1→nprefix, 작은수정)으로 "범용성" 추가 실증 가능 — 나중 결정. → ✅ **완료·통합**: `results_pitome_reg_train_faithful.txt`, tab:generality(reg이득 +0.48→+5.07).

## ⚠️ 논문 대개정: 서론 서사 재작성 + 실험 순서 재배열 (02d85653 · 2026-07-05, 사용자 /loop 검토)
- **양 파일(main.md·main_en.md) 동시 개정**. ⚠️ 408은 실험 소절/서론 손대기 전 이 줄 필독(순서 바뀜).
- **①서론 도입부 재작성**: '모델·입력 커짐→자기어텐션 O(n²)→추론 느림→토큰축소 등장→표준 병합이 레지스터(전역 허브) 이르게 파괴→우리 연구' 서사. CLIP/SigLIP '자기지도' 오류 수정(언어지도 대조학습). 기여 5개로 분리(+신규 "일반성" bullet: DINOv3/ViT-5·ADE20k·부트스트랩 CI).
- **②개념 글로스**(첫 등장부): 토큰/패치, 클래스토큰(CLS), 이분 소프트 매칭·크기, **비례 어텐션**(그동안 미정의), kNN/갤러리, FLOP≠토큰 축소율.
- **③실험 소절 인과사슬 재배열**: 설정→주결과→PiToMe→**음성대조군→ablation→심화분석(메커니즘·어텐션sink)**→두병합기→다른인코더→밀집→보조검증→일관성. (스크립트 reorder.py/reorder_en.py로 헤더 이동, 내용 보존·정렬diff 확인, \ref 자동재번호라 참조 무결). 근거: 주결과 다음 독자질문="정말 레지스터 때문?"을 일반성보다 먼저; 파생(밀집=메커니즘 예측 검증)은 부모 뒤로.
- **④다리문장 3개**(주결과→PiToMe, PiToMe→음성대조군, ablation→심화분석) **⑤톤**: 트릭/잣대/미세구조 순화, 13.9 출처(val-LOO91%) 명시, DINOv3 값-모델 라벨 이름순, keep-prior 글로스, 선형프로브 문단을 ablation→보조검증(지표다양성) 이동. 5-에이전트 검토 워크플로 기반. 구조 무결(양파일 표10/10·그림5/5·소절11:11 일치). 백업 jobs/02d85653/tmp/main*.bak_pass{3,5}.
- **⑥적대 재검토(3-에이전트)로 재배열 잔여 수정 완료(2026-07-05)**: 보조검증 서두에 선형프로브 추가(로드맵)·"지표 다양성" 중복라벨을 (선형 프로브)/(검색 mAP)로 분리·통합결론 mAP문단으로 이동(전방참조 해소), 심화분석 다리를 효율까지 확장, 밀집 다리(DINOv2 복귀), 13.9 like-for-like 캐비엇, 초록 KO/EN 정합(model-dependent 괄호절·포지셔닝 1회), EN 현수수식어 수정. 최종 검증 verdict=MINOR, medium 전부 반영. 남은 건 low(최상급 반복 등)뿐. **논문 개정 수렴 완료.**

## 레지스터 근거 강화 + fig_pitome 수정 + 어텐션맵 신규 (02d85653 · 2026-07-04, 사용자 /goal)
- **①"왜 레지스터를 보호하는가" 문단 신규**(main.md·main_en.md 방법절): register=이미지무관 학습토큰·전역정보 응축(선형프로브 분류↑위치↓)·큰ViT서 전역표현이 register에 지배[[clsregdecouple 신규 bib=Lappe&Giese NeurIPS'25 arXiv:2505.05892]]·고노름 attention sink(massive)→균질병합이 극단압축서 이르게 흡수→CLS-readout이어도 붕괴. 인용 registers·clsregdecouple·massive.
- **②어텐션맵 그림 신규 `fig_attention.{png,pdf}`**(§심화분석, KO+EN, Fig~\ref{fig:attention}): DINOv2-reg 실측(CPU, dog 이미지). **무압축서 CLS 어텐션의 ~39%가 4개 register에 쏠림=attention sink 정량**, 극단압축(92%)서 ToMe는 register 병합→sink 흩어짐 vs Ours 보존. 정성 예시(단일이미지·마지막블록) 명시. 생성기 jobs/02d85653/tmp/attnmap.py·build_attn_html.py. **2026-07-05 개정**: 사용자 지적(패널명 "Full model"이 혼동)→"Uncompressed(no reduction)"로, 라벨 "registers absorb 39% of CLS attention", ToMe·Ours에 "~92% merged" 부제. 폰트를 method 그림과 동일(HTML→render.mjs chromium, Helvetica/Arial). 캡션 "무압축/uncompressed"와 일관. **2026-07-05 재개정(사용자 지적: 히트맵이 거꾸로 읽힘=ToMe가 오히려 진하고 Ours 옅음 + register가 지도밖)**: 원인=ToMe는 register(sink)를 패치로 병합해 그 어텐션이 지도에 찍혀 진해보이고 Ours는 register를 지도밖에 보존해 패치가 옅어짐. 해결=이미지 아래 텍스트 제거 후 **레지스터 4칸 명시 표시**(kept=inferno색·merged=회색빗금). 실측: register #4 하나가 CLS어텐션 37~40% 독점(massive-activation 문헌 일치), Uncompressed·Ours는 1칸 밝음(sink alive)·ToMe는 4칸 전부 병합. 캡션 한/영에 셀 설명+ToMe 히트맵 진해보임 근거화. 생성기 attnmap.py(reg_cells)·build_attn_html.py.
- **③fig_pitome_compare 수정**: 좌측 범례 `PiToMe (energy)`→`PiToMe (cls + energy)`(PiToMe=CLS보호+에너지선택, 맞음). 우측 처리량 3선 겹침(Ours≈ToMe 576vs574 완전포개짐)→대시(ToMe)·점선(PiToMe)·실선(Ours)+그리는순서로 셋 다 가시화. x축 FLOP savings 유지(이미 §효율서 정당화: 공정 계산비교+토큰수는 과대평가+PiToMe 40-60% 잣대 호환). 재렌더·배포 완료.

## ✅ 새 모델(DINOv3·ViT-5) 결과 도착·논문 통합 완료 (02d85653 · 2026-07-05)
- GPU 서버 실험 완료·결과 수신(extra_models/results/). **selfcheck 3모델 GPU서도 cosine=1.000000 PASS**(실환경 어댑터 검증). 결과: 무압축 DINOv3-B 81.63/S+ 77.94/ViT-5 82.40. **레지스터 제거(no-reg) 시 DINOv3 무작위 붕괴(48%서 14.8/19.6, 95%서 5.6/7.9), ViT-5 완만(71→61)**. Ours는 압축하 유지(95%서 75.1/70.3/78.8). reg-count 스윕 단조증가(DINOv3-B 1.8→54.8, ViT-5 42.3→70.7).
- **논문 통합**: 신규 `\subsection{다른 인코더 계열로의 일반성: DINOv3·ViT-5}` + **tab:extra**(Ours vs no-reg, ~48/72/95%) KO+EN. 초록에도 한 절 추가. 신규 bib **dinov3**(Siméoni+ 2025 arXiv:2508.10104)·**vit5**(Wang+ "Mid-2020s" arXiv:2602.08071) 웹검증 후 추가. baseline이 "레지스터 제거"라 DINOv2 ToMe와 직접대응 아님(레지스터 기여 격리)·단일seed 명시. 결과 paper_results/에 복사. 구조 검증 표10/10·그림5/5·인용해결.

## 추가 아키텍처 실험 번들 준비 (02d85653 · 2026-07-04, 사용자 /goal)
- **요청**: "dinov3-s+, b 와 vit5 모델 추가 실험. 환경 지어내지 말고 공식 repo에서. faithful + 전체 데이터셋. 따로 폴더." → 코드 준비 완료(GPU 실행은 사용자 몫).
- **위치**: `vision_models/register_token_reduction/extra_models/` (독립 번들, 기존 검증코드 미수정·`engine/` kNN만 읽기전용 재사용).
- **공식 출처**: DINOv3-S+/B = timm 비게이트 미러 `vit_{small_plus,base}_patch16_dinov3.lvd1689m`(공식 LVD-1689M 가중치). ViT-5-B = 공식 repo `wangf3014/ViT-5` + HF `FengWang3211/ViT-5` .pth.
- **★ 핵심 발견(사용자 판단 필요)**: DINOv3·ViT-5는 **rope(rotary) 어텐션 + register가 특수토큰**이라 "ToMe가 register를 patch에 병합" baseline이 DINOv2처럼 자명하지 않음(register 병합 시 rope 정렬 붕괴 → 정보손실과 오정렬이 뒤섞임). → **model-exact한 두 전략만** 구현: `ours`(CLS+reg 보호, patch만 병합; r=0==공식, selfcheck 검증) vs `noreg`(register 제거=같은 가중치의 레지스터 없는 모델, rope 안전). Δ=ours−noreg=압축 하 register 기여. + `--regsweep` k=0..4.
- **파일**: `models_extra.py`(로더+rope-aware faithful forward, pos-tracking rope), `run_extra.py`, `selfcheck.py`(어댑터 정확성 게이트), `run_dinov3.sh`·`run_vit5.sh`, `README.md`, `GPU_RUN_PROMPT.md`.
- **✅ 어댑터 로컬 CPU 검증 완료(2026-07-04)**: 임시 CPU venv + 공식 timm미러 가중치 + 공식 ViT-5 repo clone으로 **실제 실행**. selfcheck 3모델 전부 PASS — **DINOv3-B·DINOv3-S+·ViT-5-B 모두 check1(r=0==공식) cosine=1.000000, rel_l2=0**(우리 forward가 공식 모델 정확 재현), check2(병합)·check3(noreg)도 PASS. run_extra 특징추출→kNN 배관 end-to-end 동작 확인. **전체 규모 kNN(train 1.28M+val 50k)만 GPU 필요**(이 머신 Furiosa NPU라 GPU 없음)→GPU 서버서 사용자 실행.
- **실행 중 발견·수정한 실제 버그(모두 반영)**: ①DINOv3 rope는 배치별을 head축 broadcast 위해 `[B,1,P,dim]` 필요(unsqueeze) ②register 제거 시 EvaAttention.num_prefix_tokens 동기화 ③**ViT-5 rope는 2D-공간그리드 락 + `.cuda()` 하드코딩 + freqs_cos/sin 버퍼 없음** → 전체 그리드 테이블 재구성 후 생존 patch 원래 2D위치로 gather(pos-tracking)하도록 공식과 일치 재구현.
- **✅ 5-감사 워크플로 의도부합 검증(2026-07-04)**: 공식소스 MATCHES / faithful-recipe MATCHES(proportional+key metric+attn↔mlp 병합, DINOv2 harness와 동일) / 레지스터보호만 OK(n_reg_keep 단일 노브, ours·noreg·regsweep 동일 forward) / 모델보존 OK(DINOv3 진짜 EvaAttention 호출, ViT-5는 어텐션 body 재-표현하나 학습가중치는 전부 재사용·rope는 모델 freqs로 재구성·cosine=1.0, 강제사유=stock rope의 .cuda()+sqrt(N)락) / 전체데이터셋·핸드오프 OK. **적대검증 실버그 1건: einops 누락→ViT-5 selfcheck·run_vit5 ImportError**(수정: requirements.txt에 einops 추가). +문서잔재 정리(setup_env run_all.sh→실제 엔트리, launch_on_gpu는 대용량 다운로드 前 selfcheck fail-fast, noreg 주석 정정).
- **✅ 사용자 결정(2026-07-04): "검증된 코드로 마무리"** — full-scale GPU 실행(train 1.28M+val 50k)은 **사용자가 직접** 수행. 이유: 로컬은 GPU 없는 Furiosa NPU(물리 불가), 원격 GPU 서버 SSH는 Claude Code 권한 분류기가 차단(auto 모드). 실행기 `launch_on_gpu.sh` 준비됨(`! bash …/launch_on_gpu.sh` → 서버 비번 입력 시 전송·데이터·DINOv3/ViT-5 자동 실행). **결과 나오면 논문 §일반성에 ours 곡선+Δ(ours−noreg) 반영** — 그때 이 항목 갱신.

## 확정된 결정 (LOCKED)
- **주제**: register-aware 토큰 축소. 헤드라인 = **극단 압축 영역**. (통제 harness 기준) Ours는 ToMe 전구간 우세(92%서 +7.87), **공식 PiToMe는 극단서만 역전**(교차점 토큰~74%·FLOP~35%, 92%서 +3.2); 온건압축은 PiToMe 경쟁력. "전구간 우위" 주장 금지. ⚠️ **faithful 전환 시 이 항목 재검토**: 정식 harness선 Ours가 **PiToMe 전 구간 우세**(온건 +0.66 ~ 극단 +8.16), "온건 PiToMe 우세"·"전구간 우위 금지"가 사라짐 → §faithful 참조.
- **모델**: DINOv2-reg S/B/L(헤드라인 B). 재학습 없음(timm pretrained, training-free).
- **평가 프로토콜**: 주 = **val leave-one-out kNN**(전 표 통일, 무압축 76.33). + 헤드라인 **정통 train-갤러리 kNN 재현 완료(무압축 80.9)** = sec:canonical.
- **FLOP**: fvcore MAC 관행(무압축 23.5 GFLOPs). **그림 4종(result·pitome_compare·ablation·method) HTML→벡터PDF 재작업 완료(2026-07-04, faithful 수치, CVD-안전·직접레이블·겹침 제거)**. 논문 참조 .pdf. (옛 python-pptx 덱 `ACCV_그림_네이티브.pptx`는 발표용 보존)

## 실험 상태
- **val-LOO 50k 완료**: main(tab:main)·ablation·eval_v2(S/B/L·linear-probe·dynamic, 3-seed)·robustness_50k(정식ToMe·실제PiToMe·reg개수+부트스트랩CI·검색mAP, 로그 `robustness_50k/*_50k.log`)·pitome_compare(Run2).
- **main.md = 50k 최신**(§aux: 정식ToMe +2.2/+4.8/+10.3, mAP +5.1/+8.1/+13.1, CI [+2.0,+2.5]/[+3.8,+4.4]/[+7.5,+8.3], 전부 50k 명시).
- **✅ 정통(train-갤러리) base(ViT-B) 완료·논문반영(2026-07-02)**: 무압축 **80.87**(공인82 근접, val-LOO 76.33 격차 회복). 순서·교차점(~74%) val-LOO와 완전 일치(Ours>ToMe 전구간 92%+6.3; Ours>PiToMe 극단 92%+3.0). main.md·main_en.md에 §프로토콜검증(sec:canonical)+tab:canonical 삽입, 결론 '향후과제'→'재현함' 플립. 결과 `canonical/base_results/MERGED_canonical_base.txt`.
- **✅ dense 전량 재실행 완료·논문반영(2026-07-02)**: N_TRAIN=20210(전량) 결과 도착. 무압축 mIoU 23.10→**29.79**. Ours>ToMe 전 축소율 유지(+1.2~+2.6)이나 **격차 비단조**(중간압축 최대 ${\sim}74\%$서 +2.58, 91%선 +1.92) → 옛 '91%서 +3.1 단조증가' 문구 **삭제·정정**(한/영 표·산문·초록 동기화, LaTeX 22/22). 대안(random/energy/highnorm) 여전히 ≤ToMe(register-only 유지). 결과 `dense/results/`(옛 2000판 `results_ntrain2000_archive/`로 보존). **⏳ 남은 대기**: 정통 L(시간되면). **✅ 정통 S 도착(2026-07-03)**: 무압축 77.41, 교차점~74% 동일(Δours-pitome 37%−0.08/55%−0.04/74%+0.98/83%+1.68/92%+3.17), Ours>ToMe 전구간(+0.04~+0.87, B보다 작음=S 약함 프레이밍과 일치) → 정통 검증 2번째 크기로 확장. `canonical/small_results/`. **✅ weighted-kNN 재채점 도착(2026-07-03, canonical/weighted_knn/results/)**: r=0 다수결 80.87→가중 **81.42(+0.55)**, 즉 투표방식이 최대 단일요인이나 **82.0 완전 도달은 아님**(잔차 ~0.58=미러·224·best-k). ★순서·교차점 가중서도 완전 보존(Δours-pitome 92% +2.97→+3.13, 오히려 소폭↑) → 결론이 투표방식에 불변 확인. **추가 재실행 불요 판정.** **✅ deit_compare 진단 확정**: 우리 PiToMe 포팅 vs 공식 repo(timm0.4.12 실행), baseline 일치(77.27/77.2·69.34/69.33=파이프라인OK)·고압축서 최대 0.63(S)·1.25(T)%p 벌어짐. **원인=selection 버그 아님**: 에너지 selection이 없는 **tome도 pito만큼/더** 벌어짐 → 공유 병합 harness 차이(우리=post-block+feature metric / 공식=in-block+key-metric)뿐. 이는 우리 논문의 **통제(matched) vs faithful** 구분 그대로이고, faithful 케이스는 §aux `robustness_50k/faithful_tome_h2h.py`(prop-attn+key+attn↔MLP병합)로 이미 검증(Ours>정식ToMe +2.2/+4.8/+10.3). PiToMe 에너지 selection은 충실(pitome>tome 재현). **추가 재실행 불요**. (참고: 절대치 ~2.5%p 낮은 건 val 미러 손실, 양팔 공통.)

## 핵심 수치 (val-LOO, ViT-B, 무압축 76.33)
- @92%: ToMe 63.99 / PiToMe 68.70 / **Ours 71.86** (Δ +7.87 vs ToMe, +3.16 vs PiToMe).
- ablation 91%: register만 효과(random/energy/highnorm≈ToMe). reg개수 sweep **비단조**(k=1 미미·k=2 점프) — "하나씩 누적" 주장 금지.
- robustness 50k: 정식ToMe Δ+10.29@92%, mAP+13.11@92%, 부트스트랩 CI 전구간 하한>0.

## 파일 위치
- 실험: `Model_Benchmark/rngd-npu/vision_models/register_token_reduction/{pitome_compare,ablation,eval_v2,dense,robustness_50k,canonical}`
- 논문: `ACCV/main.md`(현재·50k), `ACCV/main.bib`(47편·검증), `ACCV/참고문헌_지도.md`(배치 안내), 그림 `ACCV/fig_{result,pitome_compare,ablation,method}.{pdf,png}`(2026-07-04 재작업·논문참조=.pdf), HTML소스 `jobs/02d85653/tmp/figbuild/fig_*.html`+`render.mjs`, `ACCV/논문_핵심정리.md`.
- 개념/선행연구 설명: `ACCV/논문_핵심정리.md`(우리 논문 5단), **`ACCV/선행연구_쉬운설명.md`(NEW·408, 인용 선행논문 쉬운설명 누적 — Darcet 'ViT Need Registers' ICLR2024/2309.16588 등)**.
- 정통 실행: `canonical/run_base_canonical.sh`(ViT-B)·`run_full_canonical.sh`(S/B/L), 엔진 `pitome_compare/compare.py`(--gallery {val,train}, --gallery_cache, 타이밍).

## 다음 할 일
1. base 정통 결과 도착 → 82 재현·순서 검증 → 논문 "프로토콜 검증" 절/표/문장 추가.
2. ✅ dense 전량(20210) 반영 완료(위).
3. (시간되면) S/L 정통.

## 산문 정리 (408e1fd3, 2026-07-02 /loop 5회 + 영어 미러)
- main.md·main_en.md 둘 다 괄호 남발·필러 정리(괄호 한 171→137·영 163→131, ~20%↓). LaTeX 20/20·dangling 0·인용 44 대등. teaser 제거 완료.
- ✅ **인용 47 유지**: MAE·BEiT·iBOT를 서론에서 '인코더를 사전학습 방식으로 분류'하며 자연스럽게 재인용(필러 문장 아님, 괄호 나열도 제거). 양 파일 cites 47·mae/beit/ibot 포함 확인.

## ★★ faithful(정식) harness 전면 전환 (02d85653 · 2026-07-03) — 진행중
**결정(사용자)**: 통제 harness(post-block+feature)는 "왜 원본과 다른가" 설명 부담만 늘림 → **선행연구 그대로의 faithful(prop-attn + key-metric + attn↔MLP사이 병합; Bolya ToMe/PiToMe 공식)을 헤드라인으로 승급.** 통제-harness 설명은 논문서 삭제 예정.
**배경**: deit_compare(공식 PiToMe repo를 DeiT서 실행)로 우리 통제 pitome 포팅이 공식 selection과 일치함은 검증됨(차이=harness뿐). 그런데 통제 유지 시 계속 변명 필요 → faithful 전환이 정답.
**엔진 신뢰도**: faithful forward(`forward_faithful`)의 tome/ours가 **3개 독립 실행(tome_h2h·pitome_h2h·ablation)서 소수점까지 일치** → 검증됨. 무압축=76.32(val-LOO 76.33과 동일, harness독립).

**✅ 관문 통과 — 완료된 faithful 결과 (전부 50k, ViT-B-reg, val-LOO kNN):**
- **정식 ToMe** (`faithful_tome_h2h.py`): Ours vs ToMe = +1.0/+2.2/+4.8/+6.7/**+10.29**@91% (통제 +7.87보다↑).
- **ablation** (`ablation/eval_ablation_faithful.py`): register만 효과 유지·강화 — 91%서 Ours 72.67 vs random/energy/highnorm 62.2~62.8(=ToMe 노이즈바닥) = **+9.8 대안대비**(통제 +8보다↑). 결과 `ablation/results/ablation_FAITHFUL_*.json`.
- **정식 PiToMe** (`faithful_pitome_h2h.py`, 공식 pitome_bsm+pitome+margin스케줄 이식): **Ours>PiToMe 전 구간**(온건 +0.66/+1.24, 극단 **+8.16**@91%). ★통제선 "PiToMe 온건우세(−0.4)"였는데 **faithful선 Ours가 온건도 이김** + 극단 +3.2→+8.16. PiToMe 정상(64.47>ToMe 62.41). 로그 `faithful_pitome_50k.log`.
  - (초기 버그: bsm경로 누락→PiToMe 붕괴 58.72/+13.92. 공식 merge.py verbatim 이식으로 수정 완료.)

**✅ 추가 완료 — faithful 결과 4건 더 도착·검증(02d85653 · 2026-07-03 11:08 로컬 도착, 내용 실측):**
- **retrieval mAP**(val-LOO, B, `robustness_50k/retrieval_map_faithful_50k.log`): Ours−ToMe mAP **+2.56/+5.14/+9.40/+12.47/+16.40**(36.8→91.2%). 전구간 Ours 우세(특징 자체 우수). ※통제판(+5.1/+8.1/+13.1) 대체.
- **reg-count+부트스트랩CI**(val-LOO, B, `reg_count_sweep_faithful_50k.log`): k0..4, @91% 62.42→72.63, **CI Δ+10.21 [+9.84,+10.58] 유의**(전 r 하한>0). k3≈k4 **비단조 tail 유지**(하나씩 누적 주장 금지 근거 재확인). k0/k4 = faithful ToMe(62.38/72.67)와 일치=엔진 정합.
- **dense**(ADE20k, `dense/results/dense_miou_FAITHFUL_*.json`): reg4 무압축 29.4; **Ours>ToMe 전 축소율**(+0.5~**+5.70@91%**); random/energy/highnorm≈ToMe(register-only 유지). no-reg 대조모델선 전략차 소멸(=효과의 원천=register 재확인). Δ 비단조성은 통제 dense와 동일 성격.
- **canonical small**(정통 train 1.28M, `canonical/faithful_results/canonical_faithful_small.txt`): 무압축 **77.41**(통제 small과 동일=harness독립 확인). **Ours>ToMe·>PiToMe 전구간**, @92% ours69.85/tome65.48/pitome**57.96**(Δours-pitome **+11.89**). ★faithful+train서 **PiToMe가 ToMe보다도 아래로 붕괴**(val-LOO보다 더 강한 그림) — Ours 우위 더 뚜렷.

**사용자 결정 B(2026-07-03): 1·2·3·5를 train 갤러리로, 4(retrieval)만 val 유지.**
- retrieval 근거: **self-retrieval within-set 지표**(kNN처럼 top-k 지름길 없어 train 1.28M선 전체정렬 O(N²)≈1.6조쌍 비현실적 + 대형갤러리 표준 없음) → val-LOO가 자연스러운 자리. **408 할 일: main.md/main_en.md tab:aux에 각주 1줄**("retrieval mAP은 self-retrieval within-set 지표라 eval-set(val)에 보고; kNN-계열만 train 갤러리로 검증").

**✅ 1+2 canonical BASE (train 1.28M, faithful) 완료·검증(2026-07-03 13:24, `canonical/faithful_results/canonical_faithful_base.txt`):**
- 무압축 r0 **80.87**(=통제 canonical base와 동일=harness독립 확인, DINOv2 공식~82 근접).
- r별 tome/pitome/**ours**: 36.8% 79.64/79.89/**80.53** · 55.2% 78.52/79.16/**80.15** · 73.6% 75.99/76.94/**79.41** · 82.8% 73.79/74.73/**78.67** · **92.0% 70.00/71.08/77.28**.
- **Ours>ToMe 전구간**(+0.89~**+7.28@92%**), **Ours>PiToMe 전구간**(+0.64~**+6.21@92%**). train은 절대치 높아 격차는 val-LOO(+10.29/+8.16)보다 작으나 **순서 동일**. base선 PiToMe가 ToMe 약간 위(small은 아래 붕괴).
- ⇒ **faithful 헤드라인이 val-LOO·train 양쪽서 성립** 확정. sec:canonical 한/영에 base faithful 표·문장 반영 대상(408).

**✅ train ablation·reg-count 완료·검증(02d85653 · 2026-07-04 00:43 도착):**
- **ablation TRAIN**(`ablation/results/ablation_train_faithful_base.txt`): train 갤러리·faithful서도 register만 이득 — @92% ours **77.28** vs random 69.53/energy 70.19/highnorm 70.20(≈무보호 tome 70.00). val-LOO 결론 정통 프로토콜 재현. (73.6%: ours 79.41 vs 대안 76.0~76.3)
- **reg-count TRAIN+CI**(`robustness_50k/reg_count_train_faithful_base.txt`): k0→k4 **전 r 단조↑**(val-LOO 비단조 tail보다 깨끗), @92% k0 70.00→k4 77.28 Δ+7.29 [+6.96,+7.61] 유의, 전 r 하한>0. k0/k4가 canonical·ablation과 정확 일치(정합✅).
- ⇒ **faithful 10/10 전부 완료, 남은 실험 없음.** 결정 B(1·2·3·5 train) 데이터 완비. **408 할 일: tab:ablation·tab:aux에 train 열 추가**(val-LOO 옆에 train-갤러리 수치).

**(완료됨) ⏳ 남은 train-faithful 실행 = 3·5 (canonical 종료로 2 GPU 비어 실행 가능):**
- **3 ablation(train)**: `pitome_compare/ablation_train_faithful.py --gallery train --cache_dir pitome_compare/feat_cache_faithful` (canonical 캐시 재사용: tome·ours 재추출 skip → random/energy/highnorm만).
- **5 reg-count(train)+CI**: `pitome_compare/regcount_train_faithful.py --gallery train --cache_dir pitome_compare/feat_cache_faithful` (k0=tome·k4=ours 캐시 재사용 → k1/k2/k3만). 둘 다 같은 캐시 폴더 동시 사용 안전(키 분리). 두 장 병렬 ~11h.
- 신규 2스크립트는 양 레이아웃 호환(pitome_compare 형제import / all_new_server 같은폴더), 구문검사 통과.
- **결과 도착 후 논문 반영**: tab:main·pitome를 canonical base faithful로 보강, tab:ablation·aux(reg-count/CI)를 train-faithful로 교체(+retrieval val 각주). tab:dense는 faithful(도착분)로.
- (참고) **eval_v2(S/L·선형프로브·dynamic)**: 통제판 유지 시 faithful 불요(미결정).

**재실행 불요(harness무관)**: tab:control(Δ=0 구성상)·mechanism(병합추적)·efficiency(FLOP=토큰수).

**🔎 코드 출처 3중 검증 완료(02d85653 · 2026-07-03) — "공식 코드로 돌리나?" 최종 확인:**
- **정직한 구분**: DINOv2 본실험(main/pitome/canonical faithful)은 공식 repo를 *직접 실행*이 아니라 **공식 알고리즘을 우리 harness에 이식(port)**. 이유=공식 PiToMe repo가 timm==0.4.12 고정 → DINOv2-reg(timm≥1.0) 로드 불가. deit_compare만 *진짜* 공식 repo(git clone hchautran/PiToMe, 그들 env)를 DeiT서 직접 실행.
- (a) **소스 대조(WebFetch, 공식 algo/pitome/merge.py)**: 우리 `faithful_pitome_h2h.py`/`compare_faithful.py`의 pitome()·pitome_bsm()·merge_wavg()·energy(elu(sim−margin).mean)·merge split(indices[:2r] 짝/홀)·dst(max sim)·margin schedule(0.75→0)·bsm 층스케줄(앞 ceil(L/2)) **전부 공식과 일치**.
- (b) **실측 대조(deit_compare)**: 우리 포팅 baseline 77.27 ≈ 공식 77.2, PiToMe−ToMe 격차 재현(공식 +0.14~+0.54 ↔ 우리 +0.21~+0.72). 절대편차(≤1.25%p)는 tome·pito 대칭 발생=harness(post-block vs in-block)뿐, selection 버그 아님.
- (c) **내부 일관성**: tome/ours가 faithful_tome_h2h·pitome_h2h·ablation서 소수점까지 일치.
- ⇒ 논문 표현은 이미 "**공통 하네스에 공식 알고리즘 이식 + 통제 비교**"로 정직화됨(over-claim '소스대로 실행' 금지). ToMe도 deit_compare ②tome(=공식 algo.tome) 일치로 커버.

**🆕 새 서버 번들 준비 완료(02d85653 · 2026-07-03)**: A6000×4 대여 서버(기존 환경 無)에서 "val로 돌린 것들을 train 전체 갤러리+faithful로" 그대로 돌릴 자족 번들 = `vision_models/register_token_reduction/all_new_server/`. 구성: `setup_env.sh`(venv+pip+검증)·`config.sh`(DATA_ROOT 등 공통env, HF_HUB_OFFLINE=0 강제)·`prepare_data.py`(val5만+train1.28M non-gated 미러)·`warmup_models.py`(S/B/L 가중치 캐시)·`run_train_gallery.sh`(★ compare_faithful --gallery train, S/B/L을 GPU0/1/2 병렬)·`run_ablation_regcount.sh`(train갤러리 ablation+reg-count+CI)·`run_val_sanity.sh`(val-LOO faithful 재현=환경대조)·`run_all.sh`·`engine/`(검증된 val 스크립트 원본 복사 + compare.py만 DATA_ROOT 반영 + 신규 얇은 래퍼 `ablation_train_faithful.py`·`regcount_train_faithful.py`=검증forward/엔진 재사용). 새 서버 3함정 해결(경로 통일·오프라인 해제·train 다운로드). 전 스크립트 py_compile/bash -n 통과. GPU 실행은 사용자 몫. 검색mAP·dense는 범위 밖(별개 프로토콜/데이터셋).

**결과 다 모이면 할 일**: tab:main·tab:pitome·tab:ablation·tab:dense·tab:aux(정식ToMe/mAP/CI)를 faithful 수치로 교체 + 통제-harness 설명 한/영 삭제. Ours 정의가 전 표서 단일(faithful+register)이 됨.
- (참고) deit_compare 도구는 `vision_models/.../deit_compare/`에 보존.

## 문체 규칙 (2026-07-03 사용자 지시 · 두 세션 공통)
- **줄표(—/--/---)로 부가정보 덧붙이기 금지.** 한 문장으로 통합하거나 문장을 나눌 것. 논문 본문(main.md/main_en.md) 포함.
  예: "발견했다 --- 레지스터, 아니면 이상치"(❌) → "레지스터나 고노름 이상치 토큰에 담김을 발견했다"(⭕). 사용자가 서론 초안의 줄표를 직접 지적함. 앞으로 문장 작성 시 준수. (메모리 doc-writing-style에도 기록)

## 정직성 가드
- val-LOO는 시간이 아니라 **방법론적 이유**(상대Δ 불변·전실험 일관·exhaustive 실현가능)로 정당화 + **정통 재현으로 검증**. "시간없어서"를 논문에 쓰거나 가짜 사유 만들지 말 것.

## 2026-07-03 반영 판단
- ✅ **weighted-kNN → 반영**: 정통 절(sec:canonical) 한/영에 '공식식 온도가중 재채점 시 무압축 80.87→81.42(+0.55), 순서·교차점 불변' 한 문장 추가 = 결론이 투표방식에도 불변임을 실측으로 강화(82.0 완전도달은 아님도 정직 반영).
- ✅ **정통 ViT-S → 반영**: 같은 절에 'ViT-S로 확장해도 같은 그림(무압축 77.4, 92%서 Ours가 PiToMe +3.2·ToMe 앞섬, 이득만 작음)' 추가 = 정통 검증 2번째 크기로 확장. 표는 ViT-B 유지(산문만).
- ✅ **deit_compare → (A)양성 확정, 정직화 반영**: 원인=우리 엔진은 블록'끝'에서 hidden-state 코사인으로 병합, 공식은 블록'중간'에서 attention-key(k.mean)로 병합 — 이 구조차가 tome·pito에 **대칭** 적용돼 상대Δ(pito−tome) 보존(우리 포팅이 오히려 pito−tome 격차 소폭 큼=경쟁자 안 약화). 실제 편차 2개: ①병합 위치/메트릭 ②PiToMe 전반부 pitome_bsm 하이브리드 생략(후반부 pitome 경로는 공식과 정확 일치, 마진·size·CLS 동일). DINOv2 상대비교 **위협 없음**(세 방법 동일 엔진→공통 편차 상쇄). → 수치는 논문 미반영, 대신 main.md·main_en.md의 '소스대로 포팅/following the source' → '공통 하네스에 이식 + 통제된 비교(동일 병합 기하, 유일 변수=선택·보호 규칙)'로 정직화(4곳).
- 검증: main.md·main_en.md LaTeX 22/22·dangling 0·cites 47 유지.

## 2026-07-03 pitome_reg train 래퍼 (408)
- **문제**: 02d85653이 만든 `robustness_50k/faithful_pitome_reg_h2h.py`(일반성 검증: PiToMe 병합 위에 register 얹어 pitome_reg>pitome이면 병합기 무관 keep-rule)는 **val-LOO 전용**(tome_core.load_model_and_data=val 5만 통째 메모리+knn=LOO). train 1.28M엔 부족(지연로딩·train kNN·캐시 없음).
- **해결(새 파일, 원본 미변경)**: `pitome_compare/pitome_reg_train_faithful.py` + `run_pitome_reg_train_faithful.sh`. 원본 `forward_faithful`만 import 재사용(⚠️import 시 원본 최상단 `int(sys.argv[1])` 크래시 회피 위해 argv 잠깐 중립화), `compare` 엔진의 지연로딩·캐시·knn_gallery/knn_loo 재사용. **ablation_train_faithful.py와 동일 패턴**(compare.reduced_forward 주입). 4-arm tome/pitome/pitome_reg/ours, reg@PiTo·reg@ToMe 열 출력. py_compile+--help(argv체인) 검증 통과.
- **★캐시 재사용으로 대폭 단축(검증)**: canonical faithful(compare_faithful.py)이 tome/pitome/ours 를 이미 train 1.28M 으로 `pitome_compare/feat_cache_faithful/` 에 저장했고, 그 엔진의 forward·merge(`_make_official_pitome_merge`·tome merge)가 내 래퍼가 쓰는 `faithful_pitome_h2h` 버전과 **AST 완전 동일**함을 확인 → 특징 byte-identical → **캐시 100% 재사용 안전**. mtag/키 규약(`{mtag}__{strat}__r{r}__{split}.pt`)도 동일(둘 다 compare.extract_split·compare.main). ⇒ 래퍼 기본 `--cache_dir=feat_cache_faithful` 로 바꿈 → tome/pitome/ours **캐시 히트(무추출)**, **실제 신규 추출 = pitome_reg 한 팔뿐 → ~20h가 아니라 ~5h**(feat_cache_faithful 이 GPU 서버에 살아있으면; 없으면 자동 재추출). 실행: `bash run_pitome_reg_train_faithful.sh`.
- **정합 기준**: tome/ours가 canonical_faithful_base.txt(train)와 일치해야 엔진 정합. reg@PiTo 전부>0이면 일반성 확정.

## 2026-07-03 PiToMe 프레이밍 = 형제·측정 (408, 사용자 지시)
- 사용자 지시: PiToMe는 ToMe의 후속(형제)이니 **경쟁자가 아니라 같은 프레임워크의 다른 규칙**으로 두고, 비교는 '이렇게 측정됐다'는 **중립 보고**로. ToMe는 우리가 딛고 선 **토대**로 소개(PiToMe가 ToMe를 소개한 톤).
- 적용(한/영 대칭, LaTeX 28/28·dangling0·인용47): ①관련연구 ToMe 소개='학습 없이 BSM으로 병합, 이후 연구가 딛고 서는 프레임워크를 세움'. ②PiToMe 절 제목 '가장 강한 베이스라인과의 비교'→'같은 프레임워크의 다른 규칙: PiToMe와 같은 예산에서'. ③절 도입·초록·기여 bullet에서 '가장 강한 베이스라인/strongest'(경쟁) 전부 제거(잔여 0)→'PiToMe=ToMe+에너지, Ours=ToMe+레지스터, 같은 예산서 두 규칙이 어떻게 측정되는지 잰다'. ④'격차/we beat'→'두 규칙의 정확도 차'(중립). ⑤ablation 도입 '가장 강한 베이스라인까지 이김'→'같은 예산 측정을 봤으니'.
- ⚠️ **이는 02d85653의 faithful 리라이트가 강화했던 'Ours가 PiToMe 전구간 우세=핵심주장' 톤을 의도적으로 완화**한 것. 수치·결과는 그대로(사실), 프레이밍만 '측정 보고'로. **다시 경쟁 톤으로 되돌리지 말 것**(사용자 명시).
- 겸사: AI티 줄표 제거(줄표로 부가정보 붙이던 곳 한/영 6곳→문장 분리). [[doc-writing-style]] 규칙.
- ✅ 확인: DINOv3·ViT-5 일반성 절은 실제 결과파일(extra_models/results/*.txt) 뒷받침됨(날조 아님, 수치 일치 검증). 정상.

## 2026-07-03 ★논문 전면 개편 진행중 (408, 사용자 /goal) — 02d85653 main.md 편집 보류 요망
전역: 괄호 최소화 · 줄표 부연 금지 · 표/그림 중복 금지 · 인용 1회 · 주관적 문구 삭제 · 현학적 중복어 삭제. (규칙 [[doc-writing-style]])
체크리스트(상태): 
- [x] 0 abstract 재작성  - [x] 1 tab:positioning 삭제  - [x] 2 intro 재작성  - [x] 3 fig/table→섹션 지도(사용자에 보고)
- [x] 4 그림 리뉴얼  - [x] 5 fig/table 위치  - [x] 6 tab:control 삭제(ablation에 흡수)
- [x] 7 소제목 라벨 제거  - [x] 8 인용 1회화(49→29)  - [x] 9 Ours(레지스터)→Ours+캡션 쉽게
- [x] 10 확인: PiToMe=ToMe+에너지선택 → Ours=ToMe+레지스터보호 대응 맞음  - [x] 11 tab:extra DINOv2  - [x] 12 dense mIoU/밀집/읽기층 설명+공정하다 삭제+분류 실험
- [x] 13 보조검증 재구성(tab:aux 삭제)+val-LOO 설명+consistency 캡션 정리  - [x] 14 섹션 분포 검토
- 표/그림 중복(같은 데이터 table+fig): tab:main+fig:result, tab:pitome+fig:pitome, tab:ablation+fig:ablation → 각 하나로 통합 예정.

- [x] EN(main_en.md) 미러 완료: 0~13 전부 EN 반영, 양 파일 표7·그림2 동기·LaTeX 18/18·dangling0·인용50. (fig:attention 그림 자체 리뉴얼만 잔여)
