# NPU 상태줄 진실성 — 죽은 백엔드 회수 + 클라이언트 모델 기억

날짜: 2026-07-23

## 증상 (사용자 실측)

라우터 재시작 후에도 `? for shortcuts` 위 모델 목록이 이전 세션 상태로 남음:

```
▶ ● gpt-oss-120b(빨강) · ● Llama-3.1-8B-Instruct@pp2 npu2,3(초록)
  · ● Qwen3-4B-FP8 npu0(초록) · ● Qwen3-8B-FP8 npu1(초록)
```

서버 모델은 이미 정리했는데도 3개가 카드 배정과 함께 초록으로 뜨고, gpt-oss-120b(기본 모델)가 매 실행마다 빨강으로 뜬다.

## 근본 원인 (실측 규명)

1. **초록 잔상**: `/router/status`(`Router.status()`)가 `state=self.state.get(mid, ...)` 로 `self.state["up"]` 을 그대로 보고 — `b.alive()==False`(serve 프로세스 죽음)여도 "up". 실제 `/router/status` 에 `alive:false, state:up` 확인. 게다가 죽은 백엔드가 `_free_cards()` 의 `owned` 에 계속 잡혀 `free_cards:[]`.
   - 자식 serve 는 라우터와 무관하게 죽을 수 있다(OOM·외부 pkill·serve-router 재시작이 자식만 종료). 라우터가 이를 감지·회수하지 못하는 게 핵심.
2. **gpt-oss-120b 항상 표시**: `npuDisplayModels(models, currentModelId)` 가 현재 모델이 라이브 목록에 없으면 `{state:'down'}` 을 합성(npuStatus.ts) → 빨강. 현재 모델은 기본값 `OPENAI_MODEL`(=gpt-oss-120b)이라 한 번도 안 올려도 항상 current.
3. **기억의 출처**: 지금 "이전 모델 기억"은 라우터 잔상(버그)일 뿐. 실제 클라이언트 메모리는 없음.

## 설계

### 서버 (`furiosa_router.py`)

- `_free_cards()`: **살아있는** 백엔드의 카드만 `owned` 로 계산. 죽은 백엔드 카드는 즉시 free 후보(smi 로 실제 여유도 재확인). 읽기 전용 → 즉시 정확.
- `status()`: 죽은 백엔드를 running 보고에서 **제외**. loading/error/stopping 전이 상태(state 항목)는 유지(콜드스타트 LED). 읽기 전용, 락 없이 안전.
- `_reap_dead()`: 락 하에 죽은 백엔드를 `running`/`state` 에서 제거·로그. 기존 smi 갱신 데몬(5초)에서 호출 → 실제 정리·카드 해제.

결과: `/router/status.running` = 실제로 살아 서빙 중인 것만. `free_cards` 정확.

### 클라이언트 (openclaude 포크)

- `npuRecentModels.ts`(신규): 라우터 base 별로 **실제 up 이었던 모델 id** 를 config dir(`OPENCLAUDE_CONFIG_DIR`)의 `npu-recent-models.json` 에 원자적 저장(discoveryCache 패턴). 최근순, 상한 N. 모든 fs 는 try/catch — 폴링을 절대 깨지 않음.
- `npuStatus.ts`: 폴 성공마다 up 모델 id 를 기억에 추가, `currentNpuStatus.remembered` 로 노출.
- `npuDisplayModels(models, currentModelId, remembered)`: 목록 = 라이브 ∪ (기억됐지만 라이브 아님 → `down`/빨강). 현재 모델은 **라이브·pending·기억된 경우에만** 표시 → 한 번도 안 올린 기본 gpt-oss 는 목록에서 사라짐.
- 상태줄 방향키 선택(footer)·`/model` 색칠은 이 합집합 목록을 그대로 사용.

### 최종 동작

- 서버 정리 후 재실행: 기억된 3개가 **빨강**(라이브 아님, 실상태 반영), gpt-oss(기본·미로딩) 안 뜸.
- 모델 로딩 시: 라이브 초록, 기억에 추가.
- LRU 로 밀려난 모델: 라이브에서 빠지고 기억으로 남아 빨강 — 재선택 시 다시 로딩.

## 테스트

- 서버: `status()`/`_free_cards()` 죽은 백엔드 반영, `_reap_dead()` 제거 — 유닛 + 실기(자식 kill 후 회수 확인).
- 클라이언트: `npuRecentModels` 저장/로드/상한, `npuDisplayModels` 기억 합집합·현재모델 조건부 표시(기존 "미로딩 current 빨강" 테스트 갱신), `buildNpuSegments` 색칠. pty 로 실제 TUI 확인.
