# Switch Engine

다른 모든 Tensor Unit 엔진이 자기 DM 파티션 위에서 슬라이스마다 따로 도는 것과 달리, Switch Engine 은 256 슬라이스 링 네트워크를 통해 슬라이스를 가로질러 데이터를 옮긴다. 한 슬라이스의 값을 그룹 전체에 브로드캐스트하거나, 슬라이스끼리 값을 맞바꾸거나, 어느 슬라이스가 어느 값을 갖는지를 치환한다.

## 인터페이스

`FetchTensor::switch()` 는 `SwitchTensor` 를 만들며, `Chip`, `Cluster`, `Packet` 과 바탕 데이터 값을 그대로 보존한다.
선택한 설정을 반영하기 위해 `Slice` 와 `Time` 매핑만 바뀐다.

```rust,ignore
impl<'l, const T: Tu, P: CanApplySwitch, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Applies switching network routing only. The packet passes through
    /// unchanged, no padding, no reshaping. Use `collect` afterwards to
    /// normalize the packet to flit-sized chunks.
    #[primitive(TuTensor::switch)]
    pub fn switch<OutSlice: M, OutTime: M>(
        self,
        config: SwitchConfig,
    ) -> SwitchTensor<'l, T, D, Chip, Cluster, OutSlice, OutTime, Packet, B> {
        verify_switch::<Slice, Time, OutSlice, OutTime>(&config);
        SwitchTensor::new(self.ctx, self.inner.transpose(true))
    }
}
```

커널 작성자는 `OutSlice`, `OutTime`, 그리고 설정과 그 매개변수를 고르는 `SwitchConfig` 인자를 선택한다.
`SwitchConfig` 는 흔한 패턴을 위한 미리 정의된 변형([`Broadcast01`](#broadcast01), [`Broadcast1`](#broadcast1), [`Transpose`](#transpose), [`InterTranspose`](#intertranspose), [`TransposedBroadcast1`](#transposedbroadcast1)) 중 하나이거나, 더 일반적인 패턴을 위한 [`CustomBroadcast`](#custom-configurations) 이다.
각 변형 이름의 숫자 접미사는 `Slice` 밖으로 빠져나가는 슬라이스 하위 차원을 나열한다.
예를 들어 `Broadcast01` 은 `slice0` 과 `slice1` 을 모두 브로드캐스트하고, `Broadcast1` 은 `slice1` 만 브로드캐스트한다.

컴파일러는 `OutSlice` 와 `OutTime` 이 설정이 요구하는 차원 구조와 맞는지 검증하며(아래 설정별 절마다 입출력 도식으로 요구 구조를 보인다), 맞지 않으면 컴파일이 실패한다.
모든 설정은 `InSlice::SIZE == OutSlice::SIZE` 도 요구한다. 스위치는 전체 슬라이스 개수를 보존한다.

<a id="regular-configurations"></a>
## 정규 설정

각 정규 설정은 칩 위의 256 슬라이스를 각각 `ring_size` 슬라이스로 이루어진 병렬 **서브링**으로 분할한다.
`ring_size` 는 설정의 매개변수(보통 `slice1 × slice0`)에서 유도되며, 분할 단위와 사이클 비용을 모두 결정한다.
링 토폴로지와 라우터별 결정 로직은 아래 [구조](#architecture) 를, 사이클 인자 분해는 [성능](#performance) 을 보라.

정규 설정은 흔한 패턴 여섯 가지를 다룬다.
이 패턴들로 표현할 수 없는 임의의 `Slice` 하위 차원 치환은 대신 [`CustomBroadcast`](#custom-configurations) 를 요구한다.

브로드캐스트 축을 도입하는 설정(`Broadcast01`, `Broadcast1`, `TransposedBroadcast1` 의 `X` 또는 `Y`)은 그 축이 새로운 축이기를 요구한다. 입력 `Slice` 나 입력 `Time` 에 이미 나타나서는 안 된다.

| 설정 | 용도 | `ring_size` |
|---|---|---|
| [`Forwarding`](#forwarding) | 각 슬라이스의 데이터를 그대로 통과시킨다(슬라이스 간 교환 없음) | `1` |
| [`Broadcast01`](#broadcast01) | 안쪽 `Slice` 하위 차원 둘(`slice1` 과 `slice0`)을 서브링의 모든 슬라이스로 브로드캐스트한다 | `slice1 × slice0` |
| [`Broadcast1`](#broadcast1) | `slice0` 을 `Slice` 에 남긴 채 `slice1` 을 브로드캐스트한다 | `slice1 × slice0` |
| [`Transpose`](#transpose) | `Slice` 안에서 `slice1` 과 `slice0` 을 맞바꾼다 | `slice1 × slice0` |
| [`InterTranspose`](#intertranspose) | `Slice` 하위 차원(`slice1`)과 `Time` 하위 차원(`time1`)을 맞바꾼다 | `slice1 × slice0` |
| [`TransposedBroadcast1`](#transposedbroadcast1) | `slice1` 을 가장 안쪽 `Slice` 로 옮기면서 `slice0` 을 `Time` 으로 브로드캐스트한다(`Transpose` 후 `Broadcast1` 과 동등) | `slice1 × slice0` |

<a id="forwarding"></a>
### Forwarding

`Forwarding` 은 `Slice` 와 `Time` 매핑을 그대로 둔다. 모든 라우터가 자기 슬라이스의 입력을 그대로 출력하며, 슬라이스를 가로지르는 이동이 없다.

`SwitchConfig` 에는 `Forwarding` 변형이 없다.
슬라이스 간 교환이 필요 없으면 `.switch()` 를 건너뛰고 `FetchTensor` 에 직접 [`.collect()`](./collect-engine.md) 를 호출한다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 256, B = 64, C = 32];

fn forwarding<'l, const T: Tu>(
    input: FetchTensor<'l, T, f32, m![1], m![1 # 2], m![A], m![B], m![C]>,
) -> CollectTensor<'l, T, f32, m![1], m![1 # 2], m![A], m![B, C / 8], m![C % 8]> {
    input.collect()
}
# 
# let mut ctx = Context::acquire();
# 
# let f: FetchTensor<'_, _, f32, m![1], m![1 # 2], m![A], m![B], m![C]> = FetchTensor::new(&mut ctx.main, Tensor::zero());
# let _o = forwarding(f);
```

<a id="broadcast01"></a>
### Broadcast01

`Broadcast01` 은 각 슬라이스의 데이터를 서브링의 모든 슬라이스로 브로드캐스트하며, 안쪽 `Slice` 하위 차원 둘(`slice0` 과 `slice1`)을 `Time` 으로 옮긴다.

입력 차원 구조(바깥에서 안쪽으로, 왼쪽에서 오른쪽으로):

```text
┌──────────────────────────┬───────────────┐
│          Slice           │      Time     │
├────────┬────────┬────────┼───────┬───────┤
│ slice2 │ slice1 │ slice0 │ time1 │ time0 │
└────────┴────────┴────────┴───────┴───────┘
```

스위칭 후 `slice1` 과 `slice0` 은 서브링 전체로 브로드캐스트하기 위해 `Slice` 에서 `Time` 으로 옮겨간다.
`X` 와 `Y` 로 표시된 새 브로드캐스트 차원 둘이 비워진 `Slice` 자리를 채운다.

```text
┌──────────────────────┬─────────────────────────────────┐
│        Slice         │              Time               │
├────────┬──────┬──────┼───────┬────────┬───────┬────────┤
│ slice2 │  X   │  Y   │ time1 │ slice1 │ time0 │ slice0 │
└────────┴──────┴──────┴───────┴────────┴───────┴────────┘
```

서브링은 `ring_size = slice1 × slice0` 개 슬라이스에 걸치며, 고정된 `slice2` 에서 `(slice1, slice0)` 조합마다 하나씩이다.
모든 슬라이스가 자기 패킷을 서브링을 따라 보내고 모든 슬라이스가 `ring_size` 개 패킷을 전부 받으므로, 각 출력 슬라이스는 브로드캐스트 그룹 전체의 데이터를 갖게 되며, 출력 `Slice` 의 새 `X`, `Y` 축을 따라 인덱싱된다.

#### 예제

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 256, B = 64, C = 63, D = 2, X = 2, Y = 2];

fn broadcast01<'l, const T: Tu>(
    input: FetchTensor<'l, T, f32, m![1], m![D], m![A], m![B], m![C # 64]>,
) -> SwitchTensor<'l, T, f32, m![1], m![D], m![A / 4, X, Y], m![B / 4, A / 2 % 2, B % 4, A % 2], m![C # 64]> {
    input.switch::<m![A / 4, X, Y], m![B / 4, A / 2 % 2, B % 4, A % 2]>(
        SwitchConfig::Broadcast01 {
            slice1: 2,
            slice0: 2,
            time0: 4
        }
    )
}
# 
# let mut ctx = Context::acquire();
# 
# let f: FetchTensor<'_, _, f32, m![1], m![D], m![A], m![B], m![C # 64]> = FetchTensor::new(&mut ctx.main, Tensor::zero());
# let _o= broadcast01(f);
```

`slice1 = 2`(브로드캐스트 `X` 의 크기), `slice0 = 2`(브로드캐스트 `Y` 의 크기), `time0 = 4` 일 때 컴파일러는 `slice2 = 64`, `time1 = 16`, `ring_size = 4` 를 유도한다(서브링 64 개가 256 슬라이스에 걸친다).

하위 차원은 `slice2 = A / 4`, `slice1 = A / 2 % 2`, `slice0 = A % 2`, `time1 = B / 4`, `time0 = B % 4` 로 정해지며, `OutSlice = m![A / 4, X, Y]` 와 `OutTime = m![B / 4, A / 2 % 2, B % 4, A % 2]` 가 된다.

[사이클 추정](#performance): `ring_size × Time::SIZE × flits_per_packet = 4 × 64 × 8 = 2048`, 여기서 `Time::SIZE = 64` 이고 `flits_per_packet = sizeof(f32) × Packet::SIZE / 32 = 4 × 64 / 32 = 8` 이다.

<a id="broadcast1"></a>
### Broadcast1

입력 차원 구조(바깥에서 안쪽으로, 왼쪽에서 오른쪽으로):

```text
┌──────────────────────────┬────────┐
│           Slice          │  Time  │
├────────┬────────┬────────┼────────┤
│ slice2 │ slice1 │ slice0 │ time0  │
└────────┴────────┴────────┴────────┘
```

스위칭 후 `slice0` 은 `Slice` 에 남고 `slice1` 은 서브링 전체로 브로드캐스트하기 위해 `Slice` 에서 `Time` 으로 옮겨간다.
`X` 로 표시된 새 브로드캐스트 차원이 `slice1` 이 비운 `Slice` 자리를 채운다.

```text
┌────────────────────────┬────────────────┐
│          Slice         │      Time      │
├────────┬──────┬────────┼───────┬────────┤
│ slice2 │  X   │ slice0 │ time0 │ slice1 │
└────────┴──────┴────────┴───────┴────────┘
```

서브링은 `ring_size = slice1 × slice0` 개 슬라이스에 걸치며, 이는 `Broadcast01` 과 같은 물리적 범위다.
다만 브로드캐스트는 `slice1` 을 따라서만 일어난다. 각 출력 슬라이스는 같은 `slice0` 위치의 소스들로부터 `slice1` 개 패킷을 받으며, 가장 안쪽 `Time` 을 따라 순차적으로 배치된다.
출력 `Slice` 의 새 `X` 축(크기 `slice1`)은 이렇게 모인 데이터를 복제하고, `slice0` 자체는 원래 위치에 보존된다.

#### 예제

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 256, B = 64, C = 63, X = 4];

fn broadcast1<'l, const T: Tu>(
    input: FetchTensor<'l, T, i8, m![1], m![1 # 2], m![A], m![B], m![C # 64]>,
) -> SwitchTensor<'l, T, i8, m![1], m![1 # 2], m![A / 32, X, A % 8], m![B, A / 8 % 4], m![C # 64]> {
    input.switch::<m![A / 32, X, A % 8], m![B, A / 8 % 4]>(
        SwitchConfig::Broadcast1 {
            slice1: 4,
            slice0: 8,
        }
    )
}
# 
# let mut ctx = Context::acquire();
# 
# let f: FetchTensor<'_, _, i8, m![1], m![1 # 2], m![A], m![B], m![C # 64]> = FetchTensor::new(&mut ctx.main, Tensor::zero());
# let _o = broadcast1(f);
```

`slice1 = 4`(브로드캐스트 `X` 의 크기)와 `slice0 = 8` 일 때 컴파일러는 `slice2 = 8` 과 `ring_size = 32` 를 유도한다(서브링 8 개가 256 슬라이스에 걸친다).

하위 차원은 `slice2 = A / 32`, `slice1 = A / 8 % 4`, `slice0 = A % 8`, `time0 = B` 로 정해지며, `OutSlice = m![A / 32, X, A % 8]` 와 `OutTime = m![B, A / 8 % 4]` 가 된다.

[사이클 추정](#performance): `ring_size × Time::SIZE × flits_per_packet = 32 × 64 × 2 = 4096`, 여기서 `Time::SIZE = 64` 이고 `flits_per_packet = sizeof(i8) × Packet::SIZE / 32 = 1 × 64 / 32 = 2` 이다.

<a id="transpose"></a>
### Transpose

`Transpose` 는 `Slice` 의 가장 안쪽 부분에서 `slice1` 과 `slice0` 을 맞바꾼다.

입력과 출력의 `Slice` 순서:

```text
┌──────────────────────────┐         ┌──────────────────────────┐
│           Slice          │         │           Slice          │
├────────┬────────┬────────┤   ──►   ├────────┬────────┬────────┤
│ slice2 │ slice1 │ slice0 │         │ slice2 │ slice0 │ slice1 │
└────────┴────────┴────────┘         └────────┴────────┴────────┘
```

각 서브링은 `slice0 × slice1` 개 슬라이스에 걸쳐 데이터를 돌리며, 서브링의 모든 슬라이스가 자기 교환 상대가 이전에 갖고 있던 값을 갖게 된다.

`Transpose` 는 입력 `Time` 과 출력 `Time` 이 (정규화 후) 일치할 것을 요구한다.

#### 예제

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 256, B = 64, C = 63];

fn transpose<'l, const T: Tu>(
    input: FetchTensor<'l, T, i8, m![1], m![1 # 2], m![A], m![B], m![C # 64]>,
) -> SwitchTensor<'l, T, i8, m![1], m![1 # 2], m![A / 64, A % 2, A / 2 % 32], m![B], m![C # 64]> {
    input.switch::<m![A / 64, A % 2, A / 2 % 32], m![B]>(SwitchConfig::Transpose {
        slice1: 32,
        slice0: 2,
    })
}
# 
# let mut ctx = Context::acquire();
# 
# let f: FetchTensor<'_, _, i8, m![1], m![1 # 2], m![A], m![B], m![C # 64]> = FetchTensor::new(&mut ctx.main, Tensor::zero());
# let _o = transpose(f);
```

`slice1 = 32` 와 `slice0 = 2` 일 때 컴파일러는 `slice2 = 4` 와 `ring_size = 64` 를 유도한다(서브링 4 개가 256 슬라이스에 걸친다).

하위 차원은 `slice2 = A / 64`, `slice1 = A / 2 % 32`, `slice0 = A % 2`, `time0 = B` 로 정해지며, `OutSlice = m![A / 64, A % 2, A / 2 % 32]` 와 `OutTime = m![B]` 가 된다(slice0 과 slice1 이 맞바뀌고 `Time` 은 그대로다).

[사이클 추정](#performance): `ring_size × Time::SIZE × flits_per_packet = 64 × 64 × 2 = 8192`, 여기서 `Time::SIZE = 64` 이고 `flits_per_packet = sizeof(i8) × Packet::SIZE / 32 = 1 × 64 / 32 = 2` 이다.

<a id="intertranspose"></a>
### InterTranspose

`InterTranspose` 는 `Slice` 와 `Time` 사이에서 차원 하나를 맞바꾼다. `slice1` 이 `Time` 으로 옮겨가고 `time1` 이 `Slice` 로 옮겨간다(일반 `Transpose` 는 `Slice` 안에 머문다).

입력 차원 구조(바깥에서 안쪽으로, 왼쪽에서 오른쪽으로):

```text
┌──────────────────────────┬───────────────────────┐
│           Slice          │         Time          │
├────────┬────────┬────────┼───────┬───────┬───────┤
│ slice2 │ slice1 │ slice0 │ time2 │ time1 │ time0 │
└────────┴────────┴────────┴───────┴───────┴───────┘
```

스위칭 후 `slice1` 과 `time1` 은 `Slice`/`Time` 경계를 넘어 자리를 맞바꾼다.

```text
┌─────────────────────────┬────────────────────────┐
│          Slice          │          Time          │
├────────┬───────┬────────┼───────┬───────┬────────┤
│ slice2 │ time1 │ slice0 │ time2 │ time0 │ slice1 │
└────────┴───────┴────────┴───────┴───────┴────────┘
```

각 서브링은 `slice1 × slice0` 개 슬라이스에 걸쳐 `time1` 개 시간 스텝에 걸쳐 데이터를 돌리며, 이전에 `slice1` 로 인덱싱되던 각 슬라이스의 값이 `time1` 로 인덱싱되고 그 반대도 마찬가지가 된다.

`InterTranspose` 는 크기 제약 세 가지를 강제한다.

- `InSlice` 는 256 슬라이스 전체에 걸친다. 즉 `slice2 × slice1 × slice0 == 256`.
- 맞바뀌는 차원의 크기가 일치한다. 즉 `time1.SIZE == slice1`.
- `time2` 분해가 정수로 떨어지도록 `InTime::SIZE` 는 `slice1 × time0` 으로 나누어떨어진다.

#### 예제

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 256, B = 8, C = 32];

fn inter_transpose<'l, const T: Tu>(
    input: FetchTensor<'l, T, i8, m![1], m![1 # 2], m![A], m![B], m![C # 32]>,
) -> SwitchTensor<'l, T, i8, m![1], m![1 # 2], m![A / 32, B / 2 % 2, A % 16], m![B / 4, B % 2, A / 16 % 2], m![C # 32]> {
    input.switch::<m![A / 32, B / 2 % 2, A % 16], m![B / 4, B % 2, A / 16 % 2]>(
        SwitchConfig::InterTranspose {
            slice1: 2,
            slice0: 16,
            time0: 2,
        })
}
# 
# let mut ctx = Context::acquire();
# 
# let f: FetchTensor<'_, _, i8, m![1], m![1 # 2], m![A], m![B], m![C # 32]> = FetchTensor::new(&mut ctx.main, Tensor::zero());
# let _o = inter_transpose(f);
```

`slice1 = 2`, `slice0 = 16`, `time0 = 2` 일 때 컴파일러는 `slice2 = 8`, `time2 = 2`, `ring_size = 32` 를 유도한다(서브링 8 개가 256 슬라이스에 걸친다).

하위 차원은 `slice2 = A / 32`, `slice1 = A / 16 % 2`, `slice0 = A % 16`, `time2 = B / 4`, `time1 = B / 2 % 2`, `time0 = B % 2` 로 정해지며, `OutSlice = m![A / 32, B / 2 % 2, A % 16]` 와 `OutTime = m![B / 4, B % 2, A / 16 % 2]` 가 된다(slice1 과 time1 이 `Slice` 와 `Time` 사이에서 맞바뀐다).

[사이클 추정](#performance): `ring_size × Time::SIZE × flits_per_packet = 32 × 8 × 1 = 256`, 여기서 `Time::SIZE = 8` 이고 `flits_per_packet = sizeof(i8) × Packet::SIZE / 32 = 1 × 32 / 32 = 1` 이다.

<a id="transposedbroadcast1"></a>
### TransposedBroadcast1

`TransposedBroadcast1` 은 `slice0` 을 가장 안쪽 `Time` 위치로 브로드캐스트하고 `slice1` 을 가장 안쪽 `Slice` 위치로 옮기며, 이는 `Transpose` 를 적용한 뒤 `Broadcast1` 을 적용하는 것과 동등하다.
다음과 같이 구성된 입력 텐서는

```text
┌──────────────────────────┬────────┐
│           Slice          │  Time  │
├────────┬────────┬────────┼────────┤
│ slice2 │ slice1 │ slice0 │ time0  │
└────────┴────────┴────────┴────────┘
```

아래 출력이 된다. 여기서 `slice0` 은 서브링 전체로 브로드캐스트하기 위해 가장 안쪽 `Time` 위치로 옮겨가고, `slice1` 은 가장 안쪽 `Slice` 위치로 이동하며, 브로드캐스트 차원이 `slice1` 이 비운 가운데 자리를 채운다.

```text
┌────────────────────────┬────────────────┐
│          Slice         │      Time      │
├────────┬──────┬────────┼───────┬────────┤
│ slice2 │  Y   │ slice1 │ time0 │ slice0 │
└────────┴──────┴────────┴───────┴────────┘
```

각 서브링은 `slice0 × slice1` 개 슬라이스에 걸쳐 데이터를 돌리며, 모든 슬라이스가 자기 교환 상대의 값을 갖게 되고 그 값은 가장 안쪽 `Time` 하위 차원의 `slice0` 위치들에 걸쳐 브로드캐스트된다.

#### 예제

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 256, B = 16, C = 32, Y = 8];

fn transposed_broadcast1<'l, const T: Tu>(
    input: FetchTensor<'l, T, i8, m![1], m![1 # 2], m![A], m![B], m![C # 32]>,
) -> SwitchTensor<'l, T, i8, m![1], m![1 # 2], m![A / 64, Y, A / 8 % 8], m![B, A % 8], m![C # 32]> {
    input.switch::<m![A / 64, Y, A / 8 % 8], m![B, A % 8]>(
        SwitchConfig::TransposedBroadcast1 {
            slice1: 8,
            slice0: 8,
        }
    )
}
# 
# let mut ctx = Context::acquire();
# 
# let f: FetchTensor<'_, _, i8, m![1], m![1 # 2], m![A], m![B], m![C # 32]> = FetchTensor::new(&mut ctx.main, Tensor::zero());
# let _o = transposed_broadcast1(f);
```

`slice1 = 8` 과 `slice0 = 8`(브로드캐스트 `Y` 의 크기)일 때 컴파일러는 `slice2 = 4` 와 `ring_size = 64` 를 유도한다(서브링 4 개가 256 슬라이스에 걸친다).

하위 차원은 `slice2 = A / 64`, `slice1 = A / 8 % 8`, `slice0 = A % 8`, `time0 = B` 로 정해지며, `OutSlice = m![A / 64, Y, A / 8 % 8]` 와 `OutTime = m![B, A % 8]` 가 된다.

[사이클 추정](#performance): `ring_size × Time::SIZE × flits_per_packet = 64 × 16 × 1 = 1024`, 여기서 `Time::SIZE = 16` 이고 `flits_per_packet = sizeof(i8) × Packet::SIZE / 32 = 1 × 32 / 32 = 1` 이다.

<a id="architecture"></a>
## 구조

위에서 소개한 설정들을 실행하기 위해, Switch Engine 은 칩 위의 256 슬라이스 전부를 하나의 물리적 링(슬라이스마다 라우터 하나)으로 배치하고, 각각 `ring_size` 개 슬라이스로 이루어진 `256 / ring_size` 개의 병렬 서브링으로 분할한다(아래에 그런 서브링 하나를 보인다).
정규 설정에서는 컴파일러가 설정의 매개변수(`slice1`, `slice0`, `time0`)에서 `ring_size` 를 유도하고, `CustomBroadcast` 에서는 커널 작성자가 `ring_size` 를 직접 지정한다.

```text
   ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
   │      Router 0      │◀──▶│      Router 1      │◀──▶│        ...         │◀──▶│ Router ring_size-1 │
   └────────────────────┘    └────────────────────┘    └────────────────────┘    └────────────────────┘
             ▲                                                                                          ▲
             └──────────────────────────────────────────────────────────────────────────────────────────┘
                                            wrap-around (links are bidirectional)
```

Switch Engine 은 *스눕 비트맵*으로 설정한다. 항목 256 개(슬라이스마다 하나)로 이루어지며, 각 항목은 그 출력 슬라이스에 데이터가 도착해야 할 소스 슬라이스들을 지정한다.
정규 설정에는 내장 비트맵 생성기가 딸려 있다.
반면 [`CustomBroadcast`](#custom-configurations) 는 커널 작성자의 입출력 매핑으로부터 컴파일러가 임의의 비트맵을 합성하게 한다.


모든 라우터는 자기 비트맵 항목을 근거로, 들어오는 패킷마다 세 가지 동작 중 어떤 조합을 취할지 결정한다.

- **Output**: 패킷의 소스 슬라이스가 여기로 전달되도록 선택된 경우, 패킷을 로컬 슬라이스의 하류 파이프라인으로 넘긴다.
- **Forward right**: 패킷을 오른쪽 이웃의 라우터로 넘긴다.
- **Forward left**: 패킷을 왼쪽 이웃의 라우터로 넘긴다.

각 서브링에서 가장 왼쪽 라우터는 자기 데이터를 오른쪽으로 보내고, 가장 오른쪽 라우터는 자기 데이터를 왼쪽으로 보낸다.
`ring_size > 2` 일 때 중간 라우터들은 들어온 왼쪽 이웃 데이터를 출력하면서 오른쪽으로 전달한다.
모든 라우터는 이웃에서 도착한 데이터도 함께 출력한다.

아래 트레이스는 2 슬라이스 서브링(`ring_size = 2`) 하나에서 2 슬라이스 브로드캐스트 패턴을 돌릴 때의 라우터별 실행을 보인다.
나머지 서브링 127 개는 동일하게 동작하므로 생략한다.
각 링크는 1 사이클의 통과 지연을 가지므로, 가장 왼쪽 라우터는 사이클 0 에 시작하고 가장 오른쪽 라우터는 왼쪽 이웃의 첫 패킷이 도착한 사이클 1 에 시작한다.
`axes![A = 256, B = 2, C = 32]`, `Slice = m![A]`, `Time = m![B]`, `Packet = m![C]` 일 때, 보인 서브링은 슬라이스 0 과 1 을 담는다. 가장 왼쪽 슬라이스는 패킷 `[0, 1]` 을, 가장 오른쪽 슬라이스는 `[2, 3]` 을 갖는다.
각 칸은 `<packet>: from <source>, to <action>` 으로 읽으며, source 는 `input`/`left`/`right` 중 하나이고 action 은 `output`/`right`/`left` 중 하나 이상이다.

| 사이클 | 가장 왼쪽 슬라이스                    | 가장 오른쪽 슬라이스                                    | 출력 데이터                                                 |
| ----- | --------------------------------- | -------------------------------------------------- | ----------------------------------------------------------- |
| 0     | 0: from input, to (output, right) |                                                    | `Leftmost: [0]`<br>`Rightmost: []`                          |
| 1     | 1: from input, to (output, right) | 0: from left, to output<br> 2: from input, to left | `Leftmost: [0, 1]`<br>`Rightmost: [0]`                      |
| 2     | 2: from right, to (output, right) | 1: from left, to output<br> 3: from input, to left | `Leftmost: [0, 1, 2]`<br>`Rightmost: [0, 1]`                |
| 3     | 3: from right, to (output, right) | 2: from left, to output                            | `Leftmost: [0, 1, 2, 3]`<br>`Rightmost: [0, 1, 2]`          |
| 4     |                                   | 3: from left, to output                            | `Leftmost: [0, 1, 2, 3]`<br>`Rightmost: [0, 1, 2, 3]`       |

트레이스가 끝나면 서브링의 두 슬라이스 모두 패킷 네 개를 전부 갖게 되어 브로드캐스트가 완료된다.

비트맵은 항목의 모양으로 변환을 인코딩한다.

- **브로드캐스트 모양**: 같은 소스 데이터를 받는 여러 출력 슬라이스는 동일한 비트맵 항목을 갖는다.
- **`Slice`-to-`Time` 모양**: 한 출력 슬라이스가 여러 소스 슬라이스를 나열하면, 그 출력이 연속된 시간 스텝에 걸쳐 그것들 전부로부터 데이터를 모은다는 뜻이다.

예를 들어 위 [`Broadcast01` 예제](#broadcast01) 를 재현하는 비트맵은 다음과 같다.

| 비트맵 인덱스 | `(A / 4, A % 4)`                           | `A`                        | 링 그룹                 |
| ------------ | ------------------------------------------ | -------------------------- | -------------------------- |
| 0            | `(0, 0)`, `(0, 1)`, `(0, 2)`, `(0, 3)`     | `0`, `1`, `2`, `3`         | `0`, `1`, `2`, `3`         |
| 1            | `(0, 0)`, `(0, 1)`, `(0, 2)`, `(0, 3)`     | `0`, `1`, `2`, `3`         | `0`, `1`, `2`, `3`         |
| 2            | `(0, 0)`, `(0, 1)`, `(0, 2)`, `(0, 3)`     | `0`, `1`, `2`, `3`         | `0`, `1`, `2`, `3`         |
| 3            | `(0, 0)`, `(0, 1)`, `(0, 2)`, `(0, 3)`     | `0`, `1`, `2`, `3`         | `0`, `1`, `2`, `3`         |
| 4            | `(1, 0)`, `(1, 1)`, `(1, 2)`, `(1, 3)`     | `4`, `5`, `6`, `7`         | `4`, `5`, `6`, `7`         |
| …            | …                                          | …                          | …                          |
| 255          | `(63, 0)`, `(63, 1)`, `(63, 2)`, `(63, 3)` | `252`, `253`, `254`, `255` | `252`, `253`, `254`, `255` |

0-3 행이 동일한 항목을 갖는 이유는 슬라이스 `{0, 1, 2, 3}` 이 모두 입력 슬라이스 `{0, 1, 2, 3}` 으로부터 데이터를 받기 때문이고(브로드캐스트 모양), 각 행이 소스 네 개를 전부 나열하는 이유는 `slice1` 과 `slice0` 이 `Slice` 에서 `Time` 으로 접히기 때문이다(`Slice`-to-`Time` 모양).
이 패턴은 서브링 64 개에 대해 4 행마다 되풀이된다.
`SwitchConfig::Broadcast01` 은 이 비트맵을 자동으로 생성한다.

<a id="performance"></a>
## 성능

스위치 연산은 대략 `ring_size × Time::SIZE × flits_per_packet` 사이클이 걸린다.
모든 서브링이 병렬로 진행하므로 이 링당 사이클 수가 곧 칩 전체 지연이기도 하다.
세 인자는 다음과 같다.

- `ring_size`: flit 하나가 서브링 하나를 통과하는 데 드는 사이클.
  `ring_size` 가 크면 링당 비용이 높아지는 대신 링당 더 많은 슬라이스에 닿고, `ring_size` 가 작으면 링당 비용이 낮아지는 대신 클러스터를 더 많은 병렬 링으로 분할한다.
- `Time::SIZE`: 입력 텐서의 시간 스텝 개수.
  서브링 통과는 시간 스텝마다 한 번씩 되풀이된다.
- `flits_per_packet`: 패킷당 flit 수로, `D[Packet::SIZE]` 의 크기를 32 바이트로 나눈 값이다.
  통과는 패킷의 flit 마다도 한 번씩 되풀이된다.

<a id="custom-configurations"></a>
## 커스텀 설정

커스텀 설정은 임의의 차원 치환이나 부분 차원 추출처럼 어떤 정규 설정으로도 표현되지 않는 이동 패턴을 다룬다.
이 유연성에는 [설정 오버헤드](#configuration-overhead) 와 이 절 끝에 나열한 [제약](#constraints) 이 따른다.


`SwitchConfig::CustomBroadcast` 변형은 필드 하나를 갖는다.

```rust,ignore
/// Routes data across slices using a custom snoop bitmap.
/// The bitmap is computed by the compiler from the input shape and
/// topology parameters.
CustomBroadcast {
    /// Ring group size for the custom routing.
    ring_size: usize,
},
```

정규 설정은 스눕 비트맵용 내장 생성기를 제공하는 반면, `CustomBroadcast` 는 커널 작성자의 입출력 매핑과 `ring_size` 로부터 컴파일러가 비트맵을 직접 합성하게 한다.

### 지원하는 변환 패턴

커스텀 비트맵은 정규 설정이 다룰 수 없는 패턴 두 가지를 다룬다.

- **브로드캐스트를 동반한 자유 전치**: `Transpose` 나 `TransposedBroadcast1` 의 고정된 형태를 넘어서는, 분할 차원의 임의 치환과 브로드캐스트.
- **부분 차원 추출**: 브로드캐스트 도중 한 차원의 값 일부만 `Time` 으로 옮긴다. 반면 `Broadcast01` 같은 정규 설정은 항상 차원 전체를 옮긴다.

아래 예제들이 이 패턴들을 보여준다.

<a id="configuration-overhead"></a>
### 설정 오버헤드

커스텀 스눕 비트맵을 쓰는 일은 설정 데이터를 Switch Engine 의 Special Function Register(SFR)로 스트리밍하는 것이며, 이 SFR 쓰기는 그동안 DMA Engine 과 서브 컨텍스트를 모두 점유한다.
비트맵이 로드되는 동안 DMA 컨텍스트와 서브 컨텍스트는 다른 어떤 연산도 돌릴 수 없으므로, 비용은 고정 사이클 스톨이 아니라 스케줄링 병렬성 감소로 드러난다.

### 예제 1: 임의 치환

이 예제는 가장 안쪽 슬라이스 하위 차원 네 개(`A / 4, A % 4, B / 4, B % 4`)를 `[3, 2, 1, 0]` 으로 뒤집으며, 이는 어떤 정규 설정으로도 표현되지 않는 패턴이다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 16, B = 16, C = 8, D = 8, E = 8];

fn arbitrary_permutation<'l, const T: Tu>(
    input: FetchTensor<'l, T, f32, m![1], m![1 # 2], m![A, B], m![C], m![D, E]>,
) -> SwitchTensor<'l, T, f32, m![1], m![1 # 2], m![B % 4, B / 4, A % 4, A / 4], m![C], m![D, E]> {
    input.switch::<m![B % 4, B / 4, A % 4, A / 4], m![C]>(
        SwitchConfig::CustomBroadcast { ring_size: 256 }
    )
}
# 
# let mut ctx = Context::acquire();
# 
# let f: FetchTensor<'_, _, f32, m![1], m![1 # 2], m![A, B], m![C], m![D, E]> = FetchTensor::new(&mut ctx.main, Tensor::zero());
# let _o = arbitrary_permutation(f);
```

출력 `Slice = m![B % 4, B / 4, A % 4, A / 4]` 는 입력 슬라이스 모양 `[0, 1, 2, 3]` 을 `[3, 2, 1, 0]` 으로 치환하며, 이는 어떤 정규 설정도 다루지 못하지만 커스텀 비트맵은 다룬다.

| 비트맵 인덱스 | `(B % 4, B / 4, A % 4, A / 4)` | `(A, B)`   | 링 그룹 |
| ------------ | ------------------------------ | ---------- | ---------- |
| 0            | `(0, 0, 0, 0)`                 | `(0, 0)`   | `0`        |
| 1            | `(0, 0, 0, 1)`                 | `(4, 0)`   | `64`       |
| 2            | `(0, 0, 0, 2)`                 | `(8, 0)`   | `128`      |
| 3            | `(0, 0, 0, 3)`                 | `(12, 0)`  | `192`      |
| 4            | `(0, 0, 1, 0)`                 | `(0, 1)`   | `1`        |
| 5            | `(0, 0, 1, 1)`                 | `(4, 1)`   | `65`       |
| …            | …                              | …          | …          |
| 255          | `(3, 3, 3, 3)`                 | `(15, 15)` | `255`      |

[사이클 공식](#performance) 에 대입하면 `cycles ≈ ring_size × Time::SIZE × flits_per_packet = 256 × 8 × 8 = 16384` 이다.

- `ring_size = 256`
- `Time::SIZE = C::SIZE = 8`
- `flits_per_packet = sizeof(f32) × Packet::SIZE / 32 = 4 × 64 / 32 = 8` (`Packet = m![D, E]`, `D::SIZE × E::SIZE = 64`)

최대값인 `ring_size = 256` 이 필요한 이유는 이 치환이 되풀이 구조 없이 모든 슬라이스에 걸친 의존성을 만들기 때문이다. 입력 슬라이스와 출력 슬라이스가 링 인덱스에서 임의로 멀리 떨어질 수 있으므로, 더 작은 서브링은 그런 쌍을 적어도 하나는 담지 못한다.

### 예제 2: 다차원 브로드캐스트

예제 1 의 순수 치환과 달리, 이 예제는 인접하지 않은 차원 둘(`A % 2` 와 `B % 2`)을 `Slice` 에서 `Time` 으로 옮기면서 원래 위치에서 브로드캐스트한다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 16, B = 16, C = 8, D = 8, E = 8, X = 2, Y = 2];

fn multi_axis_broadcast<'l, const T: Tu>(
    input: FetchTensor<'l, T, f32, m![1], m![1 # 2], m![A, B], m![C], m![D, E]>,
) -> SwitchTensor<'l, T, f32, m![1], m![1 # 2], m![A / 2, X, B / 2, Y], m![C, A % 2, B % 2], m![D, E]> {
    input.switch::<m![A / 2, X, B / 2, Y], m![C, A % 2, B % 2]>(
        SwitchConfig::CustomBroadcast { ring_size: 32 }
    )
}
# 
# let mut ctx = Context::acquire();
# 
# let f: FetchTensor<'_, _, f32, m![1], m![1 # 2], m![A, B], m![C], m![D, E]> = FetchTensor::new(&mut ctx.main, Tensor::zero());
# let _o = multi_axis_broadcast(f);
```

출력은 `A % 2` 와 `B % 2` 를 `Slice` 에서 `Time` 으로 옮기고, 브로드캐스트 차원 `X` 와 `Y` 를 통해 원래 위치에서 브로드캐스트한다.
`Broadcast01` 도 비슷한 형태를 지원하지만 브로드캐스트 차원(`slice0`, `slice1`)이 입력 슬라이스에서 인접하기를 요구하므로, 인접하지 않은 차원이 `Slice` 에서 `Time` 으로 옮겨가는 것은 표현하지 못한다.
대신 커스텀 비트맵이 이를 표현한다.

| 비트맵 인덱스 | `(A / 2, A % 2, B / 2, B % 2)`                                 | `(A, B)`                                       | 링 그룹                 |
| ------------ | -------------------------------------------------------------- | ---------------------------------------------- | -------------------------- |
| 0            | `(0, 0, 0, 0)`, `(0, 0, 0, 1)`, `(0, 1, 0, 0)`, `(0, 1, 0, 1)` | `(0, 0)`, `(0, 1)`, `(1, 0)`, `(1, 1)`         | `0`, `1`, `16`, `17`       |
| 1            | `(0, 0, 0, 0)`, `(0, 0, 0, 1)`, `(0, 1, 0, 0)`, `(0, 1, 0, 1)` | `(0, 0)`, `(0, 1)`, `(1, 0)`, `(1, 1)`         | `0`, `1`, `16`, `17`       |
| 2            | `(0, 0, 1, 0)`, `(0, 0, 1, 1)`, `(0, 1, 1, 0)`, `(0, 1, 1, 1)` | `(0, 2)`, `(0, 3)`, `(1, 2)`, `(1, 3)`         | `2`, `3`, `18`, `19`       |
| 3            | `(0, 0, 1, 0)`, `(0, 0, 1, 1)`, `(0, 1, 1, 0)`, `(0, 1, 1, 1)` | `(0, 2)`, `(0, 3)`, `(1, 2)`, `(1, 3)`         | `2`, `3`, `18`, `19`       |
| …            | …                                                              | …                                              | …                          |
| 255          | `(7, 0, 7, 0)`, `(7, 0, 7, 1)`, `(7, 1, 7, 0)`, `(7, 1, 7, 1)` | `(14, 14)`, `(14, 15)`, `(15, 14)`, `(15, 15)` | `238`, `239`, `254`, `255` |

[사이클 공식](#performance) 에 대입하면 `cycles ≈ ring_size × Time::SIZE × flits_per_packet = 32 × 8 × 8 = 2048` 이다.

- `ring_size = 32` (가장 바깥 `A / 2` 분할은 서브링 간 교환이 필요 없으므로, 각 서브링 안의 가장 안쪽 32 슬라이스만 통신한다)
- `Time::SIZE = 8`
- `flits_per_packet = sizeof(f32) × Packet::SIZE / 32 = 4 × 64 / 32 = 8` (`Packet = m![D, E]`, `D::SIZE × E::SIZE = 64`)

<a id="example-3-partial-axis-extraction-slicing"></a>
### 예제 3: 부분 축 추출(슬라이싱)

옮기는 차원의 모든 값을 포함하는 예제 1 · 2 와 달리, 여기서는 `C # 4` 의 유효한 값 3 개만 `Slice` 에서 `Time` 으로 옮겨가고 그 패딩 칸은 남는다.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 16, B = 4, C = 3, D = 8, E = 8, X = 4];

fn partial_axis_extraction<'l, const T: Tu>(
    input: FetchTensor<'l, T, f32, m![1], m![1 # 2], m![A, B, C # 4], m![D], m![E]>,
) -> SwitchTensor<'l, T, f32, m![1], m![1 # 2], m![A, B, X], m![D, C], m![E]> {
    input.switch::<m![A, B, X], m![D, C]>(
        SwitchConfig::CustomBroadcast { ring_size: 4 }
    )
}
# 
# let mut ctx = Context::acquire();
# 
# let f: FetchTensor<'_, _, f32, m![1], m![1 # 2], m![A, B, C # 4], m![D], m![E]> = FetchTensor::new(&mut ctx.main, Tensor::zero());
# let _o = partial_axis_extraction(f);
```

출력은 `C # 4` 를 `Slice` 에서 `Time` 으로 옮기고 비워진 `Slice` 자리에 브로드캐스트 축 `X` 를 놓으며, 순수 패딩 칸인 네 번째 값(`C # 4 = 3`)은 버려지므로 유효한 값 `C` 세 개만 추출된다.
`Broadcast1` 도 비슷한 형태를 지원하지만 항상 차원 전체를 옮기므로 부분집합은 표현하지 못한다.

이 부분 추출이 허용되는 이유는 오직 버려지는 값이 패딩이기 때문이다. `C # 4` 는 4 로 패딩된 자리에 유효 원소 `C = 3` 개를 담으므로, 이를 `C` 로 잘라내는 것(`C # 4 → C`)은 실제 데이터를 하나도 버리지 않는다.
유효 범위를 잘라내면 살아 있는 입력을 버리게 되며, 이는 [버려지는 값은 패딩이어야 한다](#dropped-values-must-be-padding) 제약이 금지한다.
대신 커스텀 비트맵이 이 패딩 전용 추출을 표현한다.

| 비트맵 인덱스 | `(A, B, C # 4)` 소스                  | 링 그룹          |
| ------------ | ---------------------------------------- | ------------------- |
| 0            | `(0, 0, 0)`, `(0, 0, 1)`, `(0, 0, 2)`    | `0`, `1`, `2`       |
| 1            | `(0, 0, 0)`, `(0, 0, 1)`, `(0, 0, 2)`    | `0`, `1`, `2`       |
| 2            | `(0, 0, 0)`, `(0, 0, 1)`, `(0, 0, 2)`    | `0`, `1`, `2`       |
| 3            | `(0, 0, 0)`, `(0, 0, 1)`, `(0, 0, 2)`    | `0`, `1`, `2`       |
| 4            | `(0, 1, 0)`, `(0, 1, 1)`, `(0, 1, 2)`    | `4`, `5`, `6`       |
| …            | …                                        | …                   |
| 255          | `(15, 3, 0)`, `(15, 3, 1)`, `(15, 3, 2)` | `252`, `253`, `254` |

[사이클 공식](#performance) 에 대입하면 `cycles ≈ ring_size × Time::SIZE × flits_per_packet = 4 × 24 × 1 = 96` 이다.

- `ring_size = 4` (바깥 `A, B` 분할은 서브링 간 교환이 필요 없으므로, 각 서브링 안의 가장 안쪽 4 슬라이스(`C # 4` 그룹 하나)만 통신한다)
- `Time::SIZE = D::SIZE × C::SIZE = 8 × 3 = 24`
- `flits_per_packet = sizeof(f32) × Packet::SIZE / 32 = 4 × 8 / 32 = 1` (`Packet = m![E]`, `E::SIZE = 8`)

비트맵은 패딩 전용 추출을 곧바로 보여준다. `bitmap[0] = {0, 1, 2}` 는 출력 슬라이스 0 이 자기 `C # 4` 그룹 안의 유효한 입력 슬라이스 3 개로부터 받되 패딩 칸인 인덱스 3 은 건너뛴다는 뜻이다. `{0, 1, 2, 3}` 을 읽으면 그 패딩을 실제 데이터인 양 끌어들이게 된다.

<a id="constraints"></a>
### 제약

커스텀 설정에는 이 유연성을 한정하는 제약 일곱 가지가 따른다.

#### 브로드캐스트 축은 새로운 축이어야 한다

[정규 설정](#regular-configurations) 도입부와 같은 규칙이다. 출력 `Slice` 에 도입되는 각 브로드캐스트 축은 입력 `Slice` 나 입력 `Time` 에 나타나서는 안 된다.

예를 들어 `axes![A = 256, B = 64, C = 32]` 와 입력 `Slice = m![A], Time = m![B], Packet = m![C # 32]` 에서 출력 `Slice = m![A / 4, B / 32, A % 4]` 는 `B` 가 입력 `Time` 에 이미 나타나므로 이 제약을 위반한다.

#### 각 브로드캐스트 축은 정확히 한 번만 쓴다

각 브로드캐스트 축은 출력 `Slice` 에 정확히 한 번만 나타나야 한다.
같은 축을 두 출력 위치에 되풀이하는 것은 라우팅 비트맵에 대해 정의된 의미가 없다.

예를 들어 출력 `Slice = m![A / 4, X, X]`(여기서 `X` 는 두 번 쓰인 새 축)는 이 제약을 위반한다.

#### 브로드캐스트 축에는 패딩이 없어야 한다

출력 `Slice` 의 브로드캐스트 축은 패딩을 가져서는 안 된다(`Axis # N` 형태 불가).
브로드캐스트 축에 패딩이 있으면 패딩된 위치의 라우팅 목적지가 정의되지 않은 채 남는다.

예를 들어 출력 `Slice = m![A / 4, X # 4, Y]` 는 `X` 가 패딩을 가진 브로드캐스트 축이므로 이 제약을 위반한다.

#### 순서 보존

`Slice` 에서 `Time` 으로 옮겨가는 축은 입력 슬라이스 차원에서의 상대 순서를 보존해야 하며, 그렇지 않으면 `SwitchConfig::CustomBroadcast` 의 검증기가 커널 컴파일 시점에 패닉한다.
각 라우터는 버퍼링이 최소(패킷 하나)이고 로컬로 출력할지 전달할지를 즉시 결정해야 하므로, 여러 패킷을 버퍼링해 재정렬할 여지가 없다.

예를 들어 `axes![A = 16, B = 16, C = 8, D = 8, E = 8]` 와 `dtype = i8` 에서, 입력 `Slice = m![A, B], Time = m![C], Packet = m![D, E]` 를 출력 `Slice = m![A, B / 4, 4], Time = m![C, B % 2, B / 2], Packet = m![D, E]` 로 매핑하는 것은 이 제약을 위반한다.
여기서 `B % 2` 와 `B / 2` 는 입력 슬라이스 배열에 견주어 뒤집힌 순서로 나타난다.
출력 `Time = m![C, B / 2, B % 2]` 는 `B / 2, B % 2` 가 입력 순서와 맞으므로 유효하다.

#### 가장 안쪽 시간 위치

`Slice` 에서 `Time` 으로 옮겨가는 축은 출력 `Time` 의 가장 안쪽 위치를 차지해야 한다.
파이프라인에서 다른 슬라이스의 데이터는 각 패킷 안에서 마지막에 도착하므로, `Slice`-to-`Time` 하위 차원은 자연스럽게 가장 안쪽 시간 차원에 놓인다.
다른 곳에 두려면 전체 시간 시퀀스를 버퍼링하고 재정렬해야 하는데, 하드웨어는 그렇게 하지 못한다.

예를 들어 위와 같은 `axes!` 와 `dtype` 에서, 입력 `Slice = m![A, B], Time = m![C], Packet = m![D, E]` 를 출력 `Slice = m![A / 2, 2, B / 2, 2], Time = m![A % 2, C, B % 2], Packet = m![D, E]` 로 매핑하는 것은 이 제약을 위반한다.
여기서 `A % 2` 와 `B % 2` 는 상대 순서를 올바르게 보존하지만 `C` 가 그 사이에 놓인다.

> [!NOTE]
> `Broadcast01` 은 `time0` 매개변수로 이 제약을 우회하지만, 커스텀 설정에는 그런 수단이 없어 제약을 엄격히 따라야 한다.

<a id="dropped-values-must-be-padding"></a>
#### 버려지는 값은 패딩이어야 한다

한 차원에서 `Slice` 에서 `Time` 으로 옮겨가는 값의 개수가 그 차원이 걸치는 값보다 적을 때([예제 3](#example-3-partial-axis-extraction-slicing) 처럼 부분 추출), 남겨지는 값은 모두 패딩이어야 한다.
유효한 값을 버리면 살아 있는 입력을 조용히 버리게 되므로, 검증기가 커널 컴파일 시점에 이를 거부한다.

예를 들어 `axes![A = 16, B = 4, C = 3, D = 8, E = 8, X = 4]` 와 입력 `Slice = m![A, B, C # 4]` 에서 `C # 4 → C` 추출은 버려지는 네 번째 값이 패딩 칸이므로 허용된다.
전부 유효한 축을 잘라내는 것, 예컨대 `B = 4 → B = 3` 은 버려지는 값이 실제 데이터를 담으므로 이 제약을 위반한다.

#### 링 크기

`ring_size` 매개변수는 2 의 거듭제곱이어야 한다.
컴파일러는 입출력 매핑(가장 바깥의 직접 캐스트가 아닌 경계)에서 기대되는 `ring_size` 도 유도하며, 이와 맞지 않는 사용자 지정 값은 거부한다.

