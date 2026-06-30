# 학습 진도 체크리스트

이 문서는 vISA 학습 진도 체크리스트입니다. 각 모듈을 끝낼 때마다 체크하세요 — "실험 통과"는 실제로 돌려서 기대 결과를 본 것만 해당합니다(읽기만 한 건 ❌).

## 준비
- [ ] `00_SETUP.md` 완료 — `cargo furiosa-opt run --release --bin constant_add` 가 "kernel ran" 출력
- [ ] `cargo furiosa-opt test --release --bin constant_add` 가 `ok`

## 모듈
- [ ] **01 큰 그림** — vISA가 뭐고 왜 쓰는지, 하드웨어 계층(Chip/Cluster/Slice/Lane)·파이프라인 8단계·메모리 5계층을 말로 설명할 수 있다
- [ ] **02 매핑 & 텐서** — `m![A / 8 # 256]` 같은 매핑을 보고 "어느 슬라이스에 몇 개씩"인지 읽을 수 있다 / `--backend typecheck`로 합법·불법 매핑을 구분해봤다
- [ ] **03 원소 단위 연산** — constant_add·elementwise_mul·binary_add·vrf_add 실험 통과 / sub 컨텍스트가 VRF에 미리 싣는 이유를 안다
- [ ] **04 텐서 축약** — dot_product·gemv·gemm 실험 통과 / contract_outer→packet→time→lane 의 의미를 안다
- [ ] **05 텐서 옮기기** — Fetch 패딩·Commit 옵션·DMA·뱅크충돌(치명적)을 이해 / fetch_commit·reshape·tile·view 실험 통과
- [ ] **06 연산 엔진 I** — Switch 토폴로지·Collect(32B 플릿)·TRF/VRF 뱅킹·Contraction 내부(Outer/Packet/Time/Lane, 2D conv) 이해
- [ ] **07 연산 엔진 II** — Vector 엔진 op셋으로 softmax/layernorm 흐름을 따라갈 수 있다 / vector_engine 테스트 통과 / Cast·Transpose 제약을 안다
- [ ] **08 스케줄러** — main/sub/tdma/pdma·해저드·주소 충돌을 이해 / 일부러 주소 충돌 내서 실패를 봤다
- [ ] **09 타일링 & 분할** — 시간 vs 공간 분할·split-K·chip/cluster reduce를 안다 / matmul reduce 변형 실험 통과
- [ ] **10 실전 사례** — MNIST 전 과정 검증 통과 / Qwen2.5-0.5B 분해(embedding→attn→decoder→head)를 따라갔다 / softmax 커널을 트레이스했다
- [ ] **11 마무리 실습** — 새 커널(예: layernorm 또는 작은 recurrent)을 직접 짜서 시뮬레이션 검증 통과 / 비공개 컴파일러·EDF 현실·Schedule Viewer를 이해

## 숙달 신호 (이 정도면 "안다")
- [ ] 빈 파일에서 시작해 새 `#[device]` 커널 + 호스트 프로그램 + `[[bin]]` 등록까지 막힘없이 한다
- [ ] 타입 오류 메시지를 보고 "어느 파이프라인 전이가 불법인지" 바로 짚는다
- [ ] 임의의 einsum(예: `BHQK`)을 보고 어떤 축을 Slice/Lane/Time/Packet에 매핑할지 설계한다
- [ ] 우리 작업(DeltaNet recurrent 등)을 vISA로 어떻게 풀지, 무엇이 막히는지(정적 shape·Persistent Kernel 부재) 설명한다
