# Schedule Viewer

Schedule Viewer 는 [`cargo furiosa-opt`](./cargo-furiosa-opt.md) 가 생성한 스케줄 JSON 파일을 읽어 대화형 실행 타임라인으로 보여준다.
어떤 연산이 병렬로 실행되는지, 각 연산이 어떤 컨텍스트나 자원을 점유하는지, 어떤 연산이 파이프라인 진행을 막는지 살펴보는 데 쓴다.

## 시작하기

### 뷰어 설치와 실행

crates.io 에서 `furiosa-schedule-viewer` 바이너리를 설치한다:

```bash
cargo install furiosa-schedule-viewer
```

로컬 웹 UI 를 시작하려면 뷰어를 실행한다:

```bash
furiosa-schedule-viewer
```

기본적으로 서버는 `127.0.0.1:9254` 에 바인딩하고 기본 브라우저에서 페이지를 연다.
서버가 수신할 주소를 바꾸려면 `--host` 와 `--port` 를 쓴다:

```bash
furiosa-schedule-viewer --host 127.0.0.1 --port 9254
```

### 스케줄 JSON 파일 생성

[`--dump-schedule`](./cargo-furiosa-opt.md#--dump-schedule-file) 은 컴파일된 스케줄을 JSON 파일로 쓴다.
[`cargo furiosa-opt compile`](./cargo-furiosa-opt.md#direct-compilation-cargo-furiosa-opt-compile) 로 단일 커널을 컴파일할 때 쓸 수 있다:

```bash
cargo furiosa-opt compile <device-function> \
  --dump-schedule <path-to-json-file>
```

함수 이름은 위치 인자 [필터](./cargo-furiosa-opt.md#filter) 로 준다.
함수 이름이 모호하면 `kernel::gemm_kernel::gemm_kernel` 처럼 전체 Rust 경로를 쓴다.
이 명령은 여전히 일반 커널 산출물을 내보내며, JSON 파일은 Schedule Viewer 의 입력이다.

## 사용법

드롭 존을 클릭하거나 스케줄 JSON 파일을 페이지로 끌어다 놓으면 시각화된다.

### 텐서와 연산자 살펴보기

아무 노드나 클릭하면 그 노드를 살펴볼 수 있다. 왼쪽 사이드바에 이름, 수명, 컨텍스트, 연결된 노드 같은 세부 정보가 표시된다.

노드 위에 마우스를 올리거나 노드를 선택하면 관련 노드가 강조된다.
예를 들어 연산자를 선택하면 그 입력 텐서와 출력 텐서가 강조된다. 텐서를 선택하면 그 텐서에 연결된 연산자가 강조된다.

텐서와 연산자는 실제 하드웨어 명령어를 기준으로 하므로, vISA 에서 정의한 텐서·연산자와 다를 수 있다.
가능한 경우 Schedule Viewer 는 vISA 텐서의 이름과 모양, 그리고 연산자 설명을 보여준다.

### 스케줄 확대하기

스케줄을 확대해 특정 구간을 살펴볼 수 있다.

사이클 범위만 조정하려면 위쪽 타임라인을 가로질러 드래그하거나 오른쪽 위의 **Cycle Range** 입력을 쓴다.
메모리 주소 범위를 조정하려면 **Enable Brush** 를 클릭한 다음 스케줄 뷰를 가로질러 드래그한다. 브러시는 사이클 범위와 메모리 주소 범위를 모두 설정한다.

전체 스케줄 뷰로 되돌리려면 타임라인 배경을 클릭하거나 오른쪽 위의 **Reset** 을 클릭한다.
