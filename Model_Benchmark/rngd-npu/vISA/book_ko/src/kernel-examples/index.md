# 커널 예제

이 장에서는 매핑, 이동, 연산, 스케줄링을 조합해 완결된 동작 커널을 만드는 방법을 보인다.
앞선 장들에서는 매핑 표현식이 TCP 하드웨어 계층에 작업을 어떻게 분배하는지, 각 구성요소가 부분 결과를 어떻게 축약하는지 설명했다.
[입문 튜토리얼](../quick-start.md)에서는 큰 텐서의 시간 분할과 공간 분할을 간단히 소개했다.
아래 표는 각 계층에서 쓸 수 있는 병렬성과 축약을 정리한 것이다.

| 차원 | 종류 | 정의되는 곳 | 축약되는 곳 |
|-----------|------|------------|------------|
| `Chip` | 공간 | [HBM, SRAM](../mapping-tensors/spatial-temporal-dimensions.md#hbm-and-sram), [Stream](../mapping-tensors/spatial-temporal-dimensions.md#tensor-unit-stream) | [DMA](../moving-tensors/dma-engine.md) + [Vector](../computing-tensors/vector-engine/index.md) |
| `Cluster` | 공간 | [SRAM](../mapping-tensors/spatial-temporal-dimensions.md#hbm-and-sram), [Stream](../mapping-tensors/spatial-temporal-dimensions.md#tensor-unit-stream) | [DMA](../moving-tensors/dma-engine.md) + [Vector](../computing-tensors/vector-engine/index.md) |
| `Slice` | 공간 | [SRAM](../mapping-tensors/spatial-temporal-dimensions.md#hbm-and-sram), [Stream](../mapping-tensors/spatial-temporal-dimensions.md#tensor-unit-stream) | [Vector](../computing-tensors/vector-engine/index.md) |
| `Lane` | 공간 | [TRF](../mapping-tensors/spatial-temporal-dimensions.md#hbm-and-sram) | [Contraction](../computing-tensors/contraction-engine/index.md) |
| `Time` | 시간 | [Stream](../mapping-tensors/spatial-temporal-dimensions.md#tensor-unit-stream) | [Contraction](../computing-tensors/contraction-engine/index.md) |
| `Packet` | 공간 | [Stream](../mapping-tensors/spatial-temporal-dimensions.md#tensor-unit-stream) | [Contraction](../computing-tensors/contraction-engine/index.md) |

위 표의 `Chip` 행과 `Cluster` 행은 칩 간·클러스터 간 축약 패턴에 해당한다.
DMA 브로드캐스트 뒤에 Vector Engine 이진 덧셈을 수행하는 예는 [Chip/Cluster Reduce](./chip-cluster-reduce.md)를 참고한다.


예제는 단일 엔진 패턴에서 시작해 여러 엔진을 조합한 패턴을 거쳐 완전한 모델 구현으로 이어진다.

- [타일링](./tiling.md): 타일 크기 선택, 메모리 배치, 누적 전략.


- [Fetch and Commit Engine](./fetch-commit-engine.md): 축 순열, full-flit commit, 꼬리 패딩, 텐서 분할.
  메모리와 연산 사이에서 데이터 배치 변환이 필요할 때 사용한다.
- [Split Reduce](./split-reduce.md): 여러 텐서 인스턴스에 걸쳐 리듀스하기 위한 인터리브 fetch.
  리듀스 차원이 타일 하나로 누적할 수 있는 크기를 넘을 때 사용한다.
- [Chip/Cluster Reduce](./chip-cluster-reduce.md): 칩 간 ReduceScatter 와 AllReduce.
  연산을 여러 칩이나 클러스터에 분산해야 할 때 사용한다.
- [Transformer](./transformer.md): prefill 단계와 decode 단계를 갖춘 Llama 3 70B 구현.
  타일링, 다중 칩 리듀스, 메모리 관리를 결합한 완전한 모델이다.
- [Mixture of Experts](./mixture-of-experts.md): 분기 없는 TopK 라우팅과 블록 단위 희소 연산.
  희소 연산 패턴으로 동적 라우팅을 구현한 완전한 모델이다.
