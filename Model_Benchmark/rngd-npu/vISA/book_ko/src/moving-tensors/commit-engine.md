# Commit Engine

Commit Engine 은 Tensor Unit 스트림 패킷을 DM 에 쓰며, [Fetch Engine](./fetch-engine.md) 의 역이다.
이것이 Commit Sequencer 다: 모든 슬라이스에서 독립적으로 돌며 각각 자기 로컬 DM 파티션에 쓰는 [수학적 텐서 이동](../mapping-tensors/tensor-semantics.md#mathematical-tensor-move) 이다.

## 인터페이스

`TuTensor` 는 Tensor Unit 파이프라인 끝에서 `Chip`, `Cluster`, `Slice`, `Time`, `Packet` 차원을 지닌다.
그 `Time` 은 계산의 시간 전개를 반영하고, `Packet` 은 출력 스트림의 원소 레이아웃이다.

`.commit()` 은 스트림을 DM 의 `DmTensor` 에 쓴다.

```rust,ignore
impl<'l, const T: Tu, P: CanApplyCommit, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Commits to data memory.
    #[primitive(TuTensor::commit)]
    pub fn commit<Element: M>(self) -> DmTensor<D, Chip, Cluster, Slice, Element, B> {
        verify_commit::<D, Time, Packet, Element>();
        DmTensor::new(self.inner.transpose(false), None)
    }

    /// Commits to data memory at `address`.
    #[primitive(TuTensor::commit_at)]
    pub fn commit_at<Element: M>(self, address: Address) -> DmTensor<D, Chip, Cluster, Slice, Element, B> {
        verify_commit::<D, Time, Packet, Element>();
        DmTensor::new(self.inner.transpose(false), Some(address))
    }

    /// Commits to a mutable tensor view in data memory.
    #[primitive(TuTensor::commit_view)]
    pub fn commit_view<Element: M>(self, mut dst: DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element, B>) {
        verify_commit::<D, Time, Packet, Element>();
        dst.inner.transpose(self.inner.view(), false);
    }
}
```

`.commit()` 은 `Chip`, `Cluster`, `Slice` 차원을 그대로 보존하는데, 각 슬라이스가 자기 DM 파티션에 독립적으로 쓰기 때문이다.
출력 `Element` 매핑이 `Time` 과 `Packet` 을 대체하며, 스트림이 DM 에 어떻게 배치되는지를 정의한다.
`Element` 는 Commit Sequencer 와 [Commit Adapter](../computing-tensors/commit-adapter.md) 를 함께 설정하고, 입력 스트림에 대해 `Time` 축 순서를 바꿔 커밋 중에 전치를 수행할 수 있다.
`Element` 매핑이 성능에 미치는 영향은 [최적화](#optimizations) 를 참고한다.

다음 예제는 캐스팅된 누적 결과를 `bf16` 으로 DM 에 커밋한다.
출력 `DmTensor` 는 256 슬라이스에 걸쳐 16 개의 시간 스텝 × 8 개 `bf16` 원소를 저장한다.
여기서 `D = bf16`, `Element = m![M, N # 16]` 다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![P = 256, M = 16, N = 8];

fn cast_commit<'l, const T: Tu>(
    input: ContractTensor<'l, T, f32, m![1], m![1 # 2], m![P], m![M], m![N]>,
) -> DmTensor<bf16, m![1], m![1 # 2], m![P], m![M, N # 16]> {
    // Cast f32 to bf16 (Cast Engine), then commit to DM (Commit Engine).
    // Input: M = 16 time steps, N = 8 f32 elements per packet (32 bytes).
    // After cast: N = 8 bf16 elements padded to 16 (32 bytes).
    // After trim: N # 16 trimmed into N = 8.
    // The sequencer writes across P = 256 slices.
    input.cast::<bf16, m![N # 16]>().commit_trim::<m![N]>().commit()
}
#
# let mut ctx = Context::acquire();
#
# let c: ContractTensor<'_, _, f32, m![1], m![1 # 2], m![P], m![M], m![N]> = ContractTensor::new(&mut ctx.main, Tensor::zero());
# let _o = cast_commit(c);
```

<a id="constraints"></a>
## 제약

- **하드웨어 차원**: `Chip::SIZE`, `Cluster::SIZE`, `Slice::SIZE` 는 하드웨어 구성과 일치해야 한다([Sequencer](./sequencer.md#architecture) 참고).
- **주소 정렬**: 모든 시퀀서 스트라이드는 8 바이트의 배수여야 한다.
- **쓰기 단위 정렬**: `D[valid_size]` 는 8, 16, 24, 32 바이트 중 하나여야 한다([Commit Adapter 의 절단(트리밍)](../computing-tensors/commit-adapter.md#trimming) 단계 참고).

## 다중 쓰기 패킷

패킷 축이 DM 에서 연속이 아닐 수 있으므로 패킷 하나를 쓰는 데 여러 번의 하드웨어 쓰기가 필요할 수 있다.
쓰기당 원소 개수 `write_size = gcd(valid_size, access_size)` 는 컴파일러가 유도하며, `valid_size` 는 [Commit Adapter](../computing-tensors/commit-adapter.md) 에서, `access_size` 는 [Sequencer 구조](./sequencer.md#access-size) 에서 온다.
[sub 컨텍스트](../computing-tensors/index.md#execution-context) 에서 `D[write_size]` 는 8 바이트로 고정된다.
총 사이클 수는 `Time::SIZE * (valid_size / write_size)` 다.
이 나눗셈은 언제나 딱 떨어진다: main 컨텍스트에서는 `valid_size == write_size` 이므로 각 패킷이 한 사이클에 커밋된다.
sub 컨텍스트에서는 `write_size` 가 8 바이트로 고정되고 `valid_size` 가 8, 16, 24, 32 바이트 중 하나이므로(절단 제약에서 온다) `valid_size / write_size` 는 언제나 1, 2, 3, 4 중 하나다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![M = 4, K = 2, W = 8, N = 16, L = 32];

// Compiler-generated configuration: [
//   M -> 4 : 64,  (64 == 2 * 32,  contiguous)
//   K -> 2 : 32,   (32  == 32 * 1,  contiguous)
//   M -> 32 : 1    (packet dimension, contiguous)
// ] : 8
// access_size = 64; valid_size = 8; write_size = gcd(64, 8) = 8; writes per packet = 1
fn no_transpose<'l, const T: Tu>(
    input: CastTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![M, K], m![L]>,
) -> DmTensor<i8, m![1], m![1 # 2], m![1 # 256], m![M, K, L]> {
    input.commit_trim::<m![L]>().commit()
}

// Compiler-generated configuration: [
//   M -> 4 : 8,   (8  != 2 * 32, NOT contiguous)
//   K -> 2 : 32,  (32 != 8 * 1,  NOT contiguous)
//   W -> 8 : 1    (packet dimension, contiguous)
// ] : 32
// access_size = 8; valid_size = 8; write_size = gcd(8, 8) = 8; writes per packet = 1
fn transpose<'l, const T: Tu>(
    input: ContractTensor<'l, T, f32, m![1], m![1 # 2], m![1 # 256], m![M, K], m![W]>,
) -> DmTensor<f32, m![1], m![1 # 2], m![1 # 256], m![K, M, W]> {
    input.commit_trim::<m![W]>().commit()
}

// Compiler-generated configuration: [
//   M -> 4 : 8,   (8  != 2 * 32, NOT contiguous)
//   K -> 2 : 32,  (32 != 8 * 1,  NOT contiguous)
//   N -> 8 : 1    (trimmed packet dimension, contiguous)
// ] : 16
// access_size = 8; valid_size = 8 (trimmed from 16); write_size = gcd(8, 8) = 8; writes per packet = 1
fn transpose_with_trimming<'l, const T: Tu>(
    input: CastTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![M, K], m![N # 32]>,
) -> DmTensor<i8, m![1], m![1 # 2], m![1 # 256], m![K, M, N]> {
    input.commit_trim::<m![N]>().commit()
}

#
# let mut ctx = Context::acquire();
# let a: CastTensor<'_, _, i8, m![1], m![1 # 2], m![1 # 256], m![M, K], m![L]> = CastTensor::new(&mut ctx.main, Tensor::zero());
# let _o = no_transpose(a);
# let b: ContractTensor<'_, _, f32, m![1], m![1 # 2], m![1 # 256], m![M, K], m![W]> = ContractTensor::new(&mut ctx.main, Tensor::zero());
# let _o = transpose(b);
# let c: CastTensor<'_, _, i8, m![1], m![1 # 2], m![1 # 256], m![M, K], m![N # 32]> = CastTensor::new(&mut ctx.main, Tensor::zero());
# let _o = transpose_with_trimming(c);
```

## 슬라이스 비트맵

슬라이스 비트맵은 클러스터 하나 전체를 덮는 256비트 마스크로(슬라이스당 1비트, 클러스터당 256 슬라이스), 어느 슬라이스가 커밋 데이터를 받을지를 통제한다.
예를 들어 `bitmap = 00000000...01` 은 슬라이스 `0` 에만 커밋을 허용하고, `bitmap = 11111111...10` 은 슬라이스 `0` 을 뺀 모든 슬라이스에 커밋을 허용한다.


<a id="optimizations"></a>
## 최적화

세 가지 요인이 Commit Sequencer 의 처리량을 결정한다.

- **연속 주소**: 각 슬라이스 안에서 연속된 DM 주소에 쓰면 병렬 뱅크 접근이 가능해진다(DMN 당 128 B/사이클, DMN 인터리빙 시 256 B/사이클).
  같은 뱅크를 연속으로 64 회 이상 치는 패턴은 [DM 뱅크 기아](./memory-performance.md#bank-starvation) 를 유발한다.
- **공간 병렬성**: 쓰기를 모든 활성 슬라이스에 걸쳐 분산하면 처리량이 최대가 된다.
- **정렬된 쓰기**(불변식): 쓰기 주소와 쓰기 단위가 언제나 8 바이트 정렬이므로 부분 뱅크 쓰기는 결코 일어나지 않는다.
  시퀀서 스트라이드는 8 바이트의 배수이고([제약](#constraints) 참고), [Commit Adapter 의 절단(트리밍)](../computing-tensors/commit-adapter.md#trimming) 단계가 `D[valid_size]` 를 8 바이트의 배수로 유지한다.
