# gdb_infer.txt 의 `??` (native_runtime.so) 프레임 — 간이(provisional) 이름

`native_runtime.so` 는 스트립되어 함수명이 없고 런타임 주소만 보입니다. 아래 이름은 **추론치**입니다 — 근거: 주소 영역(클러스터)=같은 코드 구역, 콜래더 내 위치, 말단 syscall/epoll, 스레드명/RUST_LOG 서브시스템. gdb_load_10s·gdb_idle·gdb_infer 는 **동일 프로세스(pid 2967220)** 라 주소가 세 스냅샷에서 같은 함수를 가리킵니다.

- 영역 `core` = 0x…ea/e 대역: furiosa 비동기 런타임/eager 스케줄러 코어 (256-스레드 파킹 사다리 + epoll reactor 꼬리)
- 영역 `iodrv` = 0x…a2 대역: io-driver 스레드 풀 (8 스레드 → epoll)
- 영역 `worker` = 0x…69/99 대역: worker/device 스레드 풀 (4+1 스레드 → syscall)
- 존재(Presence) 열: L=gdb_load, I=gdb_idle, F=gdb_infer 스냅샷에 등장

| 영역 | 간이 이름 | 주소(0x…76e7+) | 존재 | infer 통과수 | 주요 callee | 역할(추론) |
|---|---|---|---|---:|---|---|
| core | `furiosa.thread_entry` | `…6eccb64f` | -IF | 269 | sched.eager.run(x256), iodrv.thread_run(x8), wrk.thread_run(x4) | native thread trampoline; shared entry of 269 furiosa native threads (called by libc start_thread) |
| core | `io.reactor.EPOLL` | `…6ea82a7b` | -IF | 9 | epoll_wait(x9) | ★ epoll reactor wait -> epoll_wait |
| core | `io.reactor.poll` | `…6ea6728d` | -IF | 9 | io.reactor.turn(x9) | io reactor poll |
| core | `io.reactor.turn` | `…6ea6d54f` | -IF | 9 | io.reactor.EPOLL(x9) | io reactor turn |
| core | `io.to_reactor` | `…6ea74d41` | -IF | 8 | io.reactor.poll(x8) | bridge into epoll reactor tail |
| core | `sched.eager.loop` | `…6ea5ef33` | -IF | 256 | sched.eager.step(x256) | L2 scheduler main loop |
| core | `sched.eager.run` | `…6ea7d6a2` | -IF | 256 | sched.eager.loop(x256) | L1 eager-scheduler thread run |
| core | `sched.eager.step` | `…6ea62bd4` | -IF | 256 | sched.poll.4(x256) | L3 scheduler step/dispatch |
| core | `sched.park.branch` | `…6ea6f85e` | -IF | 2 | sched.wait.path2a(x1), io.reactor.poll(x1) | park variant branch |
| core | `sched.park.dispatch` | `…6ea6a455` | -IF | 256 | sched.park.prepare_wait(x254), sched.park.branch(x2) | L11 park; branches to wait/epoll/io |
| core | `sched.park.enter` | `…6ea6b318` | -IF | 256 | sched.park.dispatch(x256) | L10 enter park |
| core | `sched.park.prepare_wait` | `…6ea6f8de` | -IF | 254 | sched.wait.SYSCALL(x254) | prepare blocking wait |
| core | `sched.poll.4` | `…6ea5d931` | -IF | 256 | sched.poll.5(x256) | L4 nested poll |
| core | `sched.poll.5` | `…6ea7fbc5` | -IF | 256 | sched.poll.6(x256) | L5 nested poll |
| core | `sched.poll.6` | `…6ea669cd` | -IF | 256 | sched.poll.7(x256) | L6 nested poll |
| core | `sched.poll.7` | `…6ea6bccd` | -IF | 256 | sched.poll.8(x256) | L7 nested poll |
| core | `sched.poll.8` | `…6ea6f366` | -IF | 256 | sched.poll.9(x256) | L8 nested poll |
| core | `sched.poll.9` | `…6ea718b8` | -IF | 256 | sched.park.enter(x256) | L9 nested poll |
| core | `sched.wait.SYSCALL` | `…6ea85a69` | -IF | 259 | syscall(x259) | ★ blocking-wait primitive -> syscall (NPU completion / futex); reached by 259 frames |
| core | `sched.wait.SYSCALL2` | `…6ea85afd` | -IF | 1 | syscall(x1) | -> syscall (variant wait primitive) |
| core | `sched.wait.path2a` | `…6ea6715d` | -IF | 1 | sched.wait.path2b(x1) | alt wait path |
| core | `sched.wait.path2b` | `…6ea70a82` | -IF | 1 | sched.wait.SYSCALL2(x1) | alt wait path |
| core | `worker.wait.SYSCALL` | `…6ea70c2e` | -IF | 5 | sched.wait.SYSCALL(x5) | alt caller of the syscall-wait (worker/device pools) |
| iodrv | `iodrv.L1` | `…6a24c678` | -IF | 8 | iodrv.L2(x8) |  |
| iodrv | `iodrv.L2` | `…6a1ecdca` | -IF | 8 | iodrv.L3(x8) |  |
| iodrv | `iodrv.L3` | `…6a22b137` | -IF | 8 | iodrv.L4(x8) |  |
| iodrv | `iodrv.L4` | `…6a230909` | -IF | 8 | iodrv.L5(x8) |  |
| iodrv | `iodrv.L5` | `…6a251bea` | -IF | 8 | iodrv.L6(x8) |  |
| iodrv | `iodrv.L6` | `…6a250d32` | -IF | 8 | io.to_reactor(x8) | -> io.to_reactor -> epoll |
| iodrv | `iodrv.thread_run` | `…6a20d5e9` | -IF | 8 | iodrv.L1(x8) | io-driver thread entry (8 threads) |
| worker | `wrk.L1` | `…69a21f74` | -IF | 4 | wrk.L2(x4) |  |
| worker | `wrk.L2` | `…69a89f02` | -IF | 4 | wrk.L3(x4) |  |
| worker | `wrk.L3` | `…6992be53` | -IF | 4 | worker.wait.SYSCALL(x4) | -> worker.wait.SYSCALL |
| worker | `wrk.thread_run` | `…6970ba09` | -IF | 4 | wrk.L1(x4) | worker-pool thread entry (4 threads) |
| worker | `wrk2.L1` | `…69a22624` | -IF | 1 | wrk2.L2(x1) |  |
| worker | `wrk2.L2` | `…69a8a795` | -IF | 1 | wrk2.L3(x1) |  |
| worker | `wrk2.L3` | `…6992cd52` | -IF | 1 | worker.wait.SYSCALL(x1) | -> worker.wait.SYSCALL |
| worker | `wrk2.thread_run` | `…6970c439` | -IF | 1 | wrk2.L1(x1) | worker variant entry (1 thread) |

총 38 개 네이티브 주소. ★ = 실제 블로킹 지점(syscall/epoll). `SYSCALL` 접미사 = 그 프레임이 libc `syscall()` 을 직접 호출(= NPU 완료/futex 대기).
