# 텐서 의미론

텐서는 HBM, 온칩 DM, 또는 파이프라인 스트림에 존재하고, 연산이 그것을 변환한다.
이 장은 그 수학적 의미를 정의한다: 텐서 변수가 수학적 텐서를 *담는다*는 것이 무엇을 뜻하는지, 그리고 연산이 수학적 함수를 *명세한다*는 것이 무엇을 뜻하는지 정의한다.
이 정의들은 vISA 프로그램을 텐서 수준에서 추론할 수 있게 해준다: 어떤 매핑이나 메모리 계층을 쓰든, 출력이 올바른 수학적 텐서를 담으면 그 함수는 옳다.

## 텐서 담기

텐서 변수는, 각 원소가 각 차원의 매핑이 만들어내는 부분 인덱스를 합해 얻은 텐서 인덱스에서의 \\(T\\) 값을 저장할 때 수학적 텐서 \\(T\\) 를 *담는다*.

`HostTensor<D, E>` 가 가장 단순한 경우다: 단일 매핑 `E` 가 버퍼 위치와 텐서 인덱스 사이의 대응을 전부 결정한다.
예를 들어 `A = 8`, `B = 512` 인 `HostTensor<bf16, m![A, B]>` 는 4,096 개의 `bf16` 원소를 A-major, B-minor 순서로 저장한다.
이 텐서는 다음일 때 텐서 \\(T\\) 를 담는다:
- `E::map(i) = Some(ti)` 인 모든 버퍼 인덱스 `i` 에 대해,
- `i` 번째 원소가 `ti` 에서의 \\(T\\) 값을 저장한다.

`HbmTensor<D, Chip, Element>` 는 단일 매핑을 둘로 쪼개 이를 확장한다: `Chip` 은 칩 인덱스를 부분 텐서 인덱스로 매핑하고, `Element` 는 칩별 원소 인덱스를 나머지 부분 인덱스로 매핑하며, 각각이 서로 겹치지 않는 축 부분집합을 담당하므로 그 합이 전체 텐서 인덱스를 복원한다.
이 텐서는 다음일 때 \\(T\\) 를 담는다:
- `Chip::map(i) = Some(ti)` 이고 `Element::map(j) = Some(tj)` 인 모든 칩 인덱스 `i` 와 원소 인덱스 `j` 에 대해,
- `i` 번째 칩의 `j` 번째 원소가 인덱스 `ti + tj` 에서의 \\(T\\) 를 저장한다.

다른 모든 텐서 타입은 같은 규칙을 더 많은 차원에 적용한다: 각 원소는 그 모든 매핑 매개변수가 반환한 부분 인덱스의 합에서의 \\(T\\) 를 저장한다.

## 함수 명세하기

함수를 명세한다는 것은 그 출력이 입력에 대해 무엇을 담는지 선언하는 것이다.
예를 들어 `elementwise_add` 함수는 다음과 같은 뜻에서 수학적 연산 \\(f(T_1, T_2) = T_1 + T_2\\) 를 명세한다:
- 모든 텐서 \\(T_1\\) 과 \\(T_2\\) 에 대해,
- `lhs` 가 \\(T_1\\) 을 담고 `rhs` 가 \\(T_2\\) 를 담으면,
- 반환값은 \\(T_1 + T_2\\) 를 담는다.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, B = 512];

fn elementwise_add(
    lhs: &HbmTensor<bf16, m![A], m![B]>,
    rhs: &HbmTensor<bf16, m![A], m![B]>,
) -> HbmTensor<bf16, m![A], m![B]> {
    // ... computes elementwise add ...
    # todo!("elementwise add lhs and rhs")
}
```

<a id="mathematical-tensor-move"></a>
*수학적 텐서 이동*은 \\(f(T) = T\\) 를 명세한다: 표현 방식과 무관하게 출력이 입력과 같은 수학적 텐서를 담는다.
`.to_dm()` 은 수학적 텐서 이동이다.
예를 들어 `.to_dm()` 메서드는 다음과 같은 뜻에서 \\(f(T) = T\\) 를 명세한다:
- `hbm` 이 \\(T\\) 를 담으면,
- 반환값은 \\(T\\) 를 담는다.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, B = 512];

fn hbm_to_dm(
    ctx: &mut Context,
    hbm: &HbmTensor<bf16, m![A], m![B]>,
) -> DmTensor<bf16, m![A], m![1], m![B / 2], m![B % 2]> {
    hbm.to_dm(&mut ctx.tdma)
}
```
