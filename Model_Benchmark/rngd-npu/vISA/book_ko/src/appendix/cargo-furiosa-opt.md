# `cargo-furiosa-opt`

`cargo-furiosa-opt` 은 Furiosa NPU 컴파일러 툴체인을 위한 얇은 cargo 래퍼다.
`cargo` 를 실행할 자리라면 어디서든 대신 `cargo furiosa-opt` 를 실행한다: 모든 cargo 인자는 그대로 전달되고, 래퍼는 백엔드를 선택해 빌드에 필요한 커널을 컴파일한다.

설치는 [설치](../introduction.md#installation) 를 참고한다.

## 사용법

```text
cargo furiosa-opt [--backend <backend>] <command> [args]
cargo furiosa-opt compile [FILTER]... [options]
```

첫 번째 형태는 cargo 패스스루다: `<command>` 는 임의의 cargo 서브커맨드(`build`, `test`, `run`, …)이고 `[args]` 는 cargo 로 그대로 전달된다.
두 번째 형태는 커널을 직접 컴파일한다. [직접 컴파일](#direct-compilation-cargo-furiosa-opt-compile) 을 참고한다.

## 래퍼가 더하는 것

순수 cargo 와 비교하면 `cargo furiosa-opt` 는 cargo 호출 주변에서 몇 가지 일을 한다:

- **백엔드를 선택한다.** 지정한 백엔드로 빌드/테스트/실행한다. [`--backend`](#--backend-backend) 를 참고한다.
- **필요한 커널을 자동으로 빌드한다.** `--backend npu` 에서는 `cargo furiosa-opt` 가 cargo 빌드에 앞서 필요한 `#[device]` 함수를 커널로 컴파일해, 결과 바이너리가 런타임에 그것을 로드할 수 있게 한다. [자동 커널 빌드](#automatic-kernel-builds) 를 참고한다.

<a id="--backend-backend"></a>
## `--backend` _backend_

각 커널을 평가할 백엔드를 선택한다.
뒤에 오는 모든 인자는 cargo 로 그대로 전달되므로, `--backend` 는 cargo 서브커맨드 **앞에** 와야 한다:

```bash
cargo furiosa-opt --backend typecheck run
cargo furiosa-opt --backend npu test my_test
```

**기본값:** `simulation`

**가능한 값:**

- `typecheck`: 타입 수준의 매핑/모양 검증. 값 수준 루프는 팬텀 텐서로 단락된다. NPU 하드웨어가 필요 없다.
- `simulation`: 텐서 연산을 호스트 측에서 해석한다. NPU 하드웨어가 필요 없다.
- `emulation`: NPU 디바이스 경로를 호스트 측에서 시뮬레이션한다. NPU 하드웨어가 필요 없다.
- `npu`: 컴파일된 커널로 실제 NPU 에 디스패치한다. 자동 커널 컴파일을 유발한다. 물리 NPU 와 Furiosa SDK 가 필요하다.

작업 중심의 개요는 [백엔드](../introduction.md#backends) 를 참고한다.

<a id="automatic-kernel-builds"></a>
## 자동 커널 빌드

`--backend npu` 에서는 `cargo furiosa-opt` 가 cargo 로 넘기기 전에 커널 사전 컴파일 단계를 실행한다.
이 사전 단계는 의미가 있을 때만 실행된다. 다음 **두 가지**가 모두 성립하지 않으면 건너뛴다:

- cargo 서브커맨드가 코드를 빌드하거나 실행한다: `build`, `check`, `test`, `run`, `bench`, `doc`.
- 호출이 `-h` / `--help` 질의가 아니다.

실행될 때 사전 단계는 빌드가 실제로 필요로 하는 커널만 컴파일한다:

- cargo 의 유닛 그래프를 읽어 — `-p` / `--package` 와 워크스페이스 선택을 존중하며 — 그 명령이 어떤 커널 패키지를 빌드하는지 찾는다. 크레이트의 `Cargo.toml` 이 `[package.metadata.furiosa-opt]` 를 선언하면 그 크레이트는 커널 패키지다([레이아웃](../introduction.md#layout) 참고).
- 명령이 테스트, 예제, 바이너리처럼 특정 실행 가능 타깃으로 확정되면, 컴파일러는 각 타깃을 스캔해 거기서 도달 가능한 `#[device]` 함수만 컴파일한다.
- 그렇지 않으면 선택된 패키지의 모든 커널을 컴파일하는 쪽으로 되돌아간다.

컴파일은 커널 단위로 캐시되므로 바뀌지 않은 커널은 다음 실행에서 다시 컴파일되지 않는다.
산출물은 출력 디렉터리 아래에 쓰인다([`FURIOSA_OPT_OUT_DIR`](#environment-variables) 참고).

<a id="direct-compilation-cargo-furiosa-opt-compile"></a>
## 직접 컴파일: `cargo furiosa-opt compile`

`cargo furiosa-opt compile` 은 cargo 패스스루 없이 `#[device]` 함수를 직접 컴파일한다.
항상 NPU 커널을 빌드하므로 `--backend` 를 받지 않는다.

```bash
# Compile every #[device] function in every kernel package.
cargo furiosa-opt compile

# Compile only the functions matching a filter, in one package.
cargo furiosa-opt compile transpose_simple -p my_kernels

# Compile a single function and dump its schedule for the Schedule Viewer.
cargo furiosa-opt compile transpose::transpose_simple \
  --dump-schedule schedule.json
```


<a id="filter"></a>
### `[FILTER]...`

컴파일할 `#[device]` 함수 집합을 지정한다.
필터는 전체 경로 형태(`abc::def::foo`)의 `#[device]` 함수 이름에 부분 문자열로 매칭된다. 어느 한 필터에라도 매칭되면 그 함수는 컴파일된다.
생략하면 모든 디바이스 함수를 컴파일한다.

### `-p`, `--package` _name_

컴파일을 지정한 이름의 커널 패키지로 제한한다.
여러 커널 패키지를 선택하려면 이 옵션을 반복해도 된다.
생략하면 모든 커널 패키지를 컴파일한다.

### `--message-format` _format_

컴파일러로 전달되는 진단 형식(예: `json`). 도구가 커널 컴파일 실패 진단을 기계적으로 파싱할 수 있게 한다.

### `--dump-visa` _file_

중간 vISA 를 파일로 덤프한다.
단일 커널을 컴파일할 때만 사용해야 한다.


<a id="--dump-schedule-file"></a>
### `--dump-schedule` _file_

[Schedule Viewer](./schedule-viewer.md) 용으로 스케줄을 JSON 으로 덤프한다.
단일 커널을 컴파일할 때만 사용해야 한다.


<a id="environment-variables"></a>
## 환경 변수

### `FURIOSA_OPT_OUT_DIR`

커널 출력 디렉터리.
기본값은 `<workspace target>/furiosa-opt/kernel` 이다.

