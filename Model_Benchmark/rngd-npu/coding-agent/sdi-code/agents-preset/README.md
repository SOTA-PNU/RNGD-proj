# sdi agents 프리셋 — 우리 NPU 모델용 멀티에이전트 팀 (OMO 없이, 오프라인)

OpenCode 네이티브 멀티에이전트 기능만으로, 우리 NPU 모델에 역할별 에이전트를 배치한 프리셋입니다.
("oh my opencode/OMO" 같은 클라우드 멀티모델 플러그인 없이, 같은 아이디어를 로컬 NPU로 구현)

| 에이전트 | mode | 모델(NPU) | 역할 | 도구 |
|---|---|---|---|---|
| `planner` | subagent | sdi/Qwen3-32B-FP8 | 작업 분해·설계 | 읽기전용(edit/bash deny) |
| `coder` | primary | sdi/Qwen2.5-Coder-32B-Instruct | 구현 | 전체 |
| `reviewer` | subagent | sdi/Qwen3-32B-FP8 | 버그·성능·컨벤션 리뷰 | 읽기전용 |
| `explorer` | subagent | sdi/Qwen3-32B-FP8 | 코드 검색·구조 파악 | 읽기전용 |
| `tester` | subagent | sdi/Qwen2.5-Coder-32B-Instruct | 테스트 작성·실행 | 전체 |
| `docs` | subagent | sdi/Qwen3-32B-FP8 | 문서·docstring·README | edit O, bash deny |
| `security` | subagent | sdi/Qwen3-32B-FP8 | 보안 취약점 집중 점검 | 읽기전용 |
| `debugger` | subagent | sdi/Qwen3-32B-FP8 | 버그 재현·근본원인 진단(안 고침) | edit deny, bash O |
| `refactor` | subagent | sdi/Qwen2.5-Coder-32B-Instruct | 동작 보존 구조 개선 | 전체 |
| `committer` | subagent | sdi/Qwen3-32B-FP8 | diff 보고 커밋 메시지·git | edit deny, bash O |

> 이건 **스타터 예시 10종**입니다. 에이전트는 정해진 목록이 아니라 **무제한 커스텀** — 아래 같은 역할을 .md 로 더 만들면 됩니다. mode 만 정해진 enum(`primary`/`subagent`/`all`).
>
> **더 만들 만한 역할(메뉴)**: `frontend`(UI), `db`(SQL·마이그레이션), `devops`(CI·Docker·인프라), `api-designer`(API 설계), `optimizer`(성능), `migrator`(버전/프레임워크 이전), `i18n`(번역), `data`(분석 스크립트) 등. 역할 × 모델(추론=Qwen3-32B / 코딩=Coder-32B·14B) × 권한(읽기전용/전체) 조합으로 자유 구성. 단 **모델 가짓수만 NPU 4장 안에서 적게** 유지.

## 설치 (둘 중 하나)

1. **프로젝트별**: 작업할 repo 안에 복사 → `cp *.md <프로젝트>/.opencode/agents/`
2. **전역**: `cp *.md ~/.config/opencode/agents/` (모든 프로젝트에서 사용)

> ⚠️ sdi 의 provider 이름이 `sdi` 가 아니면(리브랜딩으로 `SDI_CMD` 변경) 각 .md 의 `model:` 접두사도 맞춰 바꾸세요(예: `acme/Qwen3-32B-FP8`).

## 사용

- TUI(`sdi`)에서: 서브에이전트는 `@reviewer 이 변경 봐줘`, primary 전환은 `Tab`(build ↔ coder ↔ plan…).
- 자동: primary 가 작업에 맞게 `Task` 도구로 서브에이전트(planner/reviewer/explorer)에 위임.
- 비대화형: `sdi run --agent reviewer "..."`, `sdi run --agent coder "..."`.

## NPU 카드 예산 (중요)

각 에이전트가 자기 `model` 을 쓰므로, 동시에 활성화되면 그 모델들이 NPU에 떠야 합니다.
- 이 프리셋(7종): `Qwen3-32B-FP8`(planner/reviewer/explorer/docs/security 공유, **1장**) + `Qwen2.5-Coder-32B`(coder/tester 공유, **2장**) = **3장**(둘 다 상주, 1장 여유, 스래싱 없음). **에이전트가 7개라도 모델은 2종이라 카드는 그대로** — 카드는 *모델 가짓수*에만 달림.
- **더 가볍게**: 모든 `.md` 의 `model:` 을 `sdi/Qwen3-32B-FP8` 로 통일 → **1장**만 사용(역할/권한으로만 구분).
- ⚠️ 서로 다른 모델 에이전트를 **너무 많이** 병렬로 쓰면 4장을 넘겨 콜드스타트/교체(스래싱) 발생 — 소수 모델로 유지 권장.

## 검증 (실측)

- `agent list` 에 planner/coder/reviewer/explorer 표시(내장 build/plan 과 함께)
- `sdi run --agent reviewer` → 응답 정상, 라우터에 reviewer 모델(Qwen3-32B-FP8) 가동 확인(= per-agent 모델 라우팅 동작)
