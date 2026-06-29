# `furiosa-llm build` 컴파일 단계 `??`(native_llm_common.so) 프레임 — 간이(provisional) 이름

`native_llm_common.cpython-312*.so`(143 MB)는 **스트립**되어 함수명이 없고 런타임 주소만 보입니다. 이 라이브러리가 `furiosa.native_common.compiler`(`compile` / `create_*_compiler_config` / `GraphMetadataBuilder` / `CompiledGraph.*`)의 실체이며, `TaskCompileActor`가 `compile()`을 호출하면 여기서 17분간 lowering을 돌다가 `failed to lower the operator O1089 (no tactic)`로 실패했습니다. 아래 이름은 **추론치**입니다 — 근거: (1) 주소 영역(.so 오프셋의 상위 바이트 = 코드 구역/컴파일 패스), (2) 라이브로 잡은 두 콜래더(파이썬 MainThread = compile 드라이버→sync→syscall, 활성 풀 스레드 = tactic leaf→mid-lower→재귀 operator visitor), (3) 위쪽 파이썬 경계(py-spy: `compile_task → compile_gm_and_get_preprocessed_gm_hash → compile()@converter.py:913`). gdb_build_compile_1..5 는 **동일 TaskCompileActor 프로세스(LWP 3105741)** 라 주소가 같은 함수를 가리킵니다.

## 스레드 아키타입 (gdb 244 스레드)

| 아키타입 | 개수(대략) | 스택 요지 |
|---|---|---|
| `compile.driver` (파이썬 MainThread) | 1 | Ray actor → `_PyEval` → CPython call → **native compile 드라이버(region 19)** → sync(region 1d) → **syscall**(풀 대기) |
| `lower.pool.parked` (컴파일러 워커풀) | ~62 | `clone3 → start_thread → pool.worker.park(0x…1fbccdaa) → syscall` (작업큐 대기) |
| `lower.pool.active` (컴파일러 워커풀) | 수~수십 | `tactic.leaf(1d) → mid-lower(1b) → passA(1a) → 재귀 operator visitor(19)` — 실제 lowering 수행 |
| `ray.infra` (event_engine·nexting·grpc·poll·gcs) | ~50 | Ray gRPC/이벤트루프/타이머 — 컴파일러 아님 |

## 주소 영역(region) = 컴파일 패스

| region(오프셋 상위바이트) | 역할(추론) | 주소 수 |
|---|---|---:|
| `0x..19xxxxxx` `lower.drv` | lowering driver + recursive operator visitor (main lowering loop) | 78 |
| `0x..1axxxxxx` `lower.pA` | lowering sub-pass A | 70 |
| `0x..1bxxxxxx` `lower.mid` | mid-level lowering / IR transform | 244 |
| `0x..1cxxxxxx` `lower.cg` | lowering / codegen sub-pass (region between mid and tactic) | 55 |
| `0x..1dxxxxxx` `lower.tac` | innermost tactic-selection / codegen (leaf) + driver sync primitive | 354 |
| `0x..1fxxxxxx` `pool` | compiler worker-thread pool entry/park | 1 |
| (기타 영역) | 보조/런타임 글루 | 0 |

## 주소별 간이 이름 (총 802개, region별 정렬; 통과수 내림차순)

- 존재(Presence): 1..5 = gdb_build_compile_{1..5} 스냅샷 등장. 통과수 = 5 스냅샷 합산 caller→callee 통과 횟수.
- ★ = 핵심 경계/리프 프레임. 이름은 추론치이며 심볼화된 사실이 아닙니다.

| region | 간이 이름 | 주소 | 존재 | 통과수 | 주요 callee | 비고 |
|---|---|---|---|---:|---|---|
| `19` | `lower.drv.b881e4` | `0x00007f0919b881e4` | ---45 | 838 | lower.drv.b881e4(x382), lower.drv.b882dd(x261), lower.drv.b88340(x161) |  |
| `19` | `lower.drv.82e3b0` | `0x00007f091982e3b0` | ---45 | 750 | lower.drv.b881e4(x373), lower.drv.b88340(x239), lower.drv.b882dd(x107) |  |
| `19` | `lower.visit.recurse.A` | `0x00007f0919989eb9` | 1--4- | 638 | lower.visit.recurse.C(x638) | ★ recursive operator-tree visitor (loops with .B/.C) |
| `19` | `lower.visit.recurse.C` | `0x00007f0919b67a4d` | 1--4- | 638 | lower.visit.recurse.B(x422), lower.drv.b8996a(x138), lower.drv.b89a05(x78) | ★ recursive operator-tree visitor (loops with .A/.B) |
| `19` | `lower.drv.81d495` | `0x00007f091981d495` | ---45 | 598 | lower.drv.988e9a(x516), lower.drv.988eac(x77), lower.drv.b858ca(x5) |  |
| `19` | `lower.drv.b882dd` | `0x00007f0919b882dd` | ---45 | 598 | lower.drv.81d495(x598) |  |
| `19` | `lower.drv.b88340` | `0x00007f0919b88340` | ---45 | 522 | lower.visit.sync(x522) |  |
| `19` | `lower.drv.988e9a` | `0x00007f0919988e9a` | ---45 | 516 | lower.drv.b882dd(x230), lower.drv.b88340(x122), lower.drv.b881e4(x81) |  |
| `19` | `lower.visit.recurse.B` | `0x00007f0919b897b9` | 1--4- | 422 | lower.visit.recurse.A(x320), lower.drv.989d7a(x102) | ★ recursive operator-tree visitor (loops with .A/.C) |
| `19` | `lower.drv.9b0ce0` | `0x00007f09199b0ce0` | ---45 | 257 | lower.drv.b9a7ee(x253), lower.drv.b9a6fc(x4) |  |
| `19` | `lower.drv.b858ca` | `0x00007f0919b858ca` | ---45 | 257 | lower.drv.9b0ce0(x257) |  |
| `19` | `lower.drv.b9a7ee` | `0x00007f0919b9a7ee` | ---45 | 253 | lower.drv.b9ad97(x229), lower.drv.b9aca8(x24) |  |
| `19` | `lower.drv.b9ad97` | `0x00007f0919b9ad97` | ---45 | 229 | lower.drv.b8f49d(x227), lower.drv.b8f13f(x1), lower.drv.b8f335(x1) |  |
| `19` | `lower.drv.b8f49d` | `0x00007f0919b8f49d` | ---45 | 227 | lower.tac.ed6746(x227) |  |
| `19` | `lower.visit.driver` | `0x00007f091982dd9a` | 1--4- | 202 | lower.visit.recurse.A(x192), lower.drv.989d7a(x10) | per-operator lowering driver (calls the recursion) |
| `19` | `lower.drv.989d7a` | `0x00007f0919989d7a` | 1--4- | 150 | lower.pA.eef59b(x150) |  |
| `19` | `lower.drv.b87e6e` | `0x00007f0919b87e6e` | ---45 | 139 | lower.drv.b858ca(x139) |  |
| `19` | `lower.drv.81d27b` | `0x00007f091981d27b` | 1--4- | 138 | lower.visit.recurse.A(x100), lower.drv.989d7a(x38) |  |
| `19` | `lower.drv.b8996a` | `0x00007f0919b8996a` | 1--4- | 138 | lower.drv.81d27b(x138) |  |
| `19` | `lower.drv.b89a05` | `0x00007f0919b89a05` | 1--4- | 78 | lower.visit.sync(x78) |  |
| `19` | `lower.drv.988eac` | `0x00007f0919988eac` | ---45 | 77 | lower.drv.b858ca(x77) |  |
| `19` | `lower.drv.82e3c2` | `0x00007f091982e3c2` | ---45 | 27 | lower.drv.b858ca(x27) |  |
| `19` | `lower.drv.988a6d` | `0x00007f0919988a6d` | 1--4- | 26 | lower.visit.recurse.A(x26) |  |
| `19` | `lower.drv.ba9b02` | `0x00007f0919ba9b02` | 1--4- | 26 | lower.drv.988a6d(x26) |  |
| `19` | `lower.drv.b9aca8` | `0x00007f0919b9aca8` | ---4- | 24 | lower.drv.ba9b02(x24) |  |
| `19` | `lower.drv.98a468` | `0x00007f091998a468` | 1-3-- | 16 | lower.drv.b67c2d(x14), lower.drv.b67ca1(x2) |  |
| `19` | `lower.drv.b67c2d` | `0x00007f0919b67c2d` | 1-3-- | 14 | lower.drv.b8a2fc(x9), lower.drv.b8a0a9(x5) |  |
| `19` | `lower.drv.82db7a` | `0x00007f091982db7a` | 1-3-- | 12 | lower.drv.98a468(x9), lower.drv.98a34c(x3) |  |
| `19` | `lower.drv.b87de7` | `0x00007f0919b87de7` | ---45 | 9 | lower.drv.b858ca(x9) |  |
| `19` | `lower.drv.b8a2fc` | `0x00007f0919b8a2fc` | 1-3-- | 9 | lower.visit.sync(x5), lower.tac.df2663(x4) |  |
| `19` | `compile.driver.enter` | `0x00007f0919513d61` | 12345 | 5 | lower.drv.4ff3cb(x5) | ★ first native frame under the CPython compile() PyO3 call (compile orchestration entry) |
| `19` | `lower.drv.4ff3cb` | `0x00007f09194ff3cb` | 12345 | 5 | lower.drv.508041(x5) |  |
| `19` | `lower.drv.508041` | `0x00007f0919508041` | 12345 | 5 | lower.drv.50dc22(x5) |  |
| `19` | `lower.drv.50dc22` | `0x00007f091950dc22` | 12345 | 5 | lower.drv.511a92(x5) |  |
| `19` | `lower.drv.511a92` | `0x00007f0919511a92` | 12345 | 5 | lower.drv.56126e(x5) |  |
| `19` | `lower.drv.55f461` | `0x00007f091955f461` | 12345 | 5 | lower.drv.8242f9(x5) |  |
| `19` | `lower.drv.56126e` | `0x00007f091956126e` | 12345 | 5 | lower.drv.55f461(x5) |  |
| `19` | `lower.drv.8242f9` | `0x00007f09198242f9` | 12345 | 5 | lower.drv.824a4b(x5) |  |
| `19` | `lower.drv.824a4b` | `0x00007f0919824a4b` | 12345 | 5 | lower.drv.b67465(x5) |  |
| `19` | `lower.drv.9004e4` | `0x00007f09199004e4` | 1-3-- | 5 | lower.drv.bc52f8(x5) |  |
| `19` | `lower.drv.98a34c` | `0x00007f091998a34c` | 1-3-- | 5 | lower.drv.9004e4(x5) |  |
| `19` | `lower.drv.b67465` | `0x00007f0919b67465` | 12345 | 5 | lower.drv.82058e(x4), lower.drv.820def(x1) |  |
| `19` | `lower.drv.b8a0a9` | `0x00007f0919b8a0a9` | 1-3-- | 5 | lower.drv.98a468(x3), lower.drv.98a34c(x2) |  |
| `19` | `lower.drv.bc52f8` | `0x00007f0919bc52f8` | 1-3-- | 5 | lower.drv.b98e89(x3), lower.drv.b9912e(x2) |  |
| `19` | `lower.drv.82058e` | `0x00007f091982058e` | 1-345 | 4 | lower.drv.ac5f72(x4) |  |
| `19` | `lower.drv.ab5eb9` | `0x00007f0919ab5eb9` | 1-345 | 4 | lower.drv.baac6a(x4) |  |
| `19` | `lower.drv.ac5f72` | `0x00007f0919ac5f72` | 1-345 | 4 | lower.drv.ab5eb9(x4) |  |
| `19` | `lower.drv.b9a6fc` | `0x00007f0919b9a6fc` | ----5 | 4 | lower.mid.1476b2(x2), lower.mid.147f46(x2) |  |
| `19` | `lower.drv.ba9f15` | `0x00007f0919ba9f15` | 1-345 | 4 | lower.drv.ba24c8(x2), lower.drv.ba5728(x2) |  |
| `19` | `lower.drv.baac6a` | `0x00007f0919baac6a` | 1-345 | 4 | lower.drv.ba9f15(x4) |  |
| `19` | `lower.drv.b98e89` | `0x00007f0919b98e89` | 1-3-- | 3 | lower.pA.ee8761(x3) |  |
| `19` | `lower.drv.7d7e2f` | `0x00007f09197d7e2f` | ---45 | 2 | compile.sync(x2) |  |
| `19` | `lower.drv.7d8a01` | `0x00007f09197d8a01` | 1-3-- | 2 | compile.sync(x2) |  |
| `19` | `lower.drv.817c2c` | `0x00007f0919817c2c` | ---45 | 2 | lower.drv.b881e4(x2) |  |
| `19` | `lower.drv.81a92c` | `0x00007f091981a92c` | 1-3-- | 2 | lower.drv.98a468(x2) |  |
| `19` | `lower.drv.82afed` | `0x00007f091982afed` | ---45 | 2 | lower.drv.817c2c(x2) |  |
| `19` | `lower.drv.82bdba` | `0x00007f091982bdba` | 1-3-- | 2 | lower.drv.81a92c(x2) |  |
| `19` | `lower.drv.988cb5` | `0x00007f0919988cb5` | ---45 | 2 | lower.drv.b675ec(x2) |  |
| `19` | `lower.drv.a23d0d` | `0x00007f0919a23d0d` | 1-3-- | 2 | lower.drv.b85293(x2) |  |
| `19` | `lower.drv.b675ec` | `0x00007f0919b675ec` | ---45 | 2 | lower.drv.7d7e2f(x2) |  |
| `19` | `lower.drv.b67ca1` | `0x00007f0919b67ca1` | 1-3-- | 2 | lower.drv.7d8a01(x2) |  |
| `19` | `lower.drv.b85293` | `0x00007f0919b85293` | 1-3-- | 2 | lower.drv.98a468(x2) |  |
| `19` | `lower.drv.b9912e` | `0x00007f0919b9912e` | 1---- | 2 | lower.drv.ba9364(x2) |  |
| `19` | `lower.drv.b9cbca` | `0x00007f0919b9cbca` | 1-3-- | 2 | lower.drv.a23d0d(x2) |  |
| `19` | `lower.drv.ba24c8` | `0x00007f0919ba24c8` | 1-3-- | 2 | lower.drv.b9cbca(x2) |  |
| `19` | `lower.drv.ba5728` | `0x00007f0919ba5728` | ---45 | 2 | lower.drv.988cb5(x2) |  |
| `19` | `lower.drv.ba9364` | `0x00007f0919ba9364` | 1---- | 2 | lower.drv.ba9b02(x2) |  |
| `19` | `lower.drv.820def` | `0x00007f0919820def` | -2--- | 1 | lower.mid.bc268d(x1) |  |
| `19` | `lower.drv.8f29a5` | `0x00007f09198f29a5` | ----5 | 1 | __GI___libc_free(x1) |  |
| `19` | `lower.drv.8f32a2` | `0x00007f09198f32a2` | ----5 | 1 | lower.drv.8f29a5(x1) |  |
| `19` | `lower.drv.944a3f` | `0x00007f0919944a3f` | ---4- | 1 | lower.drv.98d548(x1) |  |
| `19` | `lower.drv.94ba12` | `0x00007f091994ba12` | ---4- | 1 | __GI___libc_malloc(x1) |  |
| `19` | `lower.drv.98d548` | `0x00007f091998d548` | ---4- | 1 | lower.drv.9aa4e0(x1) |  |
| `19` | `lower.drv.9aa4e0` | `0x00007f09199aa4e0` | ---4- | 1 | lower.drv.9b9b1e(x1) |  |
| `19` | `lower.drv.9b9b1e` | `0x00007f09199b9b1e` | ---4- | 1 | lower.drv.94ba12(x1) |  |
| `19` | `lower.drv.b8f13f` | `0x00007f0919b8f13f` | ---4- | 1 | lower.drv.944a3f(x1) |  |
| `19` | `lower.drv.b8f335` | `0x00007f0919b8f335` | ----5 | 1 | lower.drv.8f32a2(x1) |  |
| `19` | `lower.drv.5e6670` | `0x00007f09195e6670` | 1---- | 0 | — |  |
| `1a` | `lower.pA.eef59b` | `0x00007f091aeef59b` | 1--4- | 150 | lower.pA.enter(x150) |  |
| `1a` | `lower.pA.enter` | `0x00007f091aee71ee` | 1--4- | 150 | lower.mid.08c5a7(x107), lower.mid.08d0c3(x23), lower.mid.08c8b6(x10) | entry into lowering sub-pass A |
| `1a` | `lower.pA.ee9c0d` | `0x00007f091aee9c0d` | 1--4- | 66 | lower.mid.3b6a5e(x39), lower.mid.3b6a87(x17), lower.mid.3b6cb5(x4) |  |
| `1a` | `lower.pA.ee9ec1` | `0x00007f091aee9ec1` | 1--4- | 22 | lower.cg.817557(x21), lower.cg.817375(x1) |  |
| `1a` | `lower.pA.ee9b85` | `0x00007f091aee9b85` | 1--4- | 17 | lower.pA.eebbc3(x13), lower.pA.eebc69(x2), lower.pA.eebbef(x1) |  |
| `1a` | `lower.pA.eebbc3` | `0x00007f091aeebbc3` | 1--4- | 13 | lower.pA.eeeaf6(x12), lower.pA.eeeab3(x1) |  |
| `1a` | `lower.pA.eeeaf6` | `0x00007f091aeeeaf6` | 1---- | 12 | lower.mid.1a7ec0(x9), lower.mid.1a7f46(x1), lower.mid.1a7528(x1) |  |
| `1a` | `lower.pA.be4cd4` | `0x00007f091abe4cd4` | 1---- | 8 | lower.pA.d34087(x8) |  |
| `1a` | `lower.pA.d34087` | `0x00007f091ad34087` | 1---- | 8 | lower.mid.090145(x7), lower.mid.08f692(x1) |  |
| `1a` | `lower.pA.dfe8b6` | `0x00007f091adfe8b6` | 1---- | 4 | __cxa_thread_atexit_impl(x4) |  |
| `1a` | `lower.pA.dd3d04` | `0x00007f091add3d04` | 1-3-- | 3 | lower.pA.debfbb(x3) |  |
| `1a` | `lower.pA.debfbb` | `0x00007f091adebfbb` | 1-3-- | 3 | lower.pA.e09fe9(x3) |  |
| `1a` | `lower.pA.def8c9` | `0x00007f091adef8c9` | 1-3-- | 3 | lower.pA.be888f(x2), lower.pA.be86d1(x1) |  |
| `1a` | `lower.pA.e09fe9` | `0x00007f091ae09fe9` | 1-3-- | 3 | lower.pA.def8c9(x3) |  |
| `1a` | `lower.pA.ee8761` | `0x00007f091aee8761` | 1-3-- | 3 | lower.mid.150473(x3) |  |
| `1a` | `lower.pA.bc9050` | `0x00007f091abc9050` | --3-- | 2 | lower.pA.db2585(x2) |  |
| `1a` | `lower.pA.be888f` | `0x00007f091abe888f` | --3-- | 2 | lower.pA.c2e994(x2) |  |
| `1a` | `lower.pA.c2e994` | `0x00007f091ac2e994` | --3-- | 2 | lower.pA.bc9050(x2) |  |
| `1a` | `lower.pA.db2585` | `0x00007f091adb2585` | --3-- | 2 | lower.pA.db2a3d(x2) |  |
| `1a` | `lower.pA.db2a3d` | `0x00007f091adb2a3d` | --3-- | 2 | lower.pA.db2f1a(x1), lower.pA.db2f64(x1) |  |
| `1a` | `lower.pA.e98e8c` | `0x00007f091ae98e8c` | 1---- | 2 | __GI___libc_realloc(x2) |  |
| `1a` | `lower.pA.ee9c2c` | `0x00007f091aee9c2c` | 1---- | 2 | lower.tac.1abc84(x2) |  |
| `1a` | `lower.pA.eebc69` | `0x00007f091aeebc69` | 1---- | 2 | lower.mid.87139d(x1), lower.mid.87146a(x1) |  |
| `1a` | `lower.pA.b427e3` | `0x00007f091ab427e3` | 1---- | 1 | lower.pA.b43b61(x1) |  |
| `1a` | `lower.pA.b43b61` | `0x00007f091ab43b61` | 1---- | 1 | lower.pA.eefb36(x1) |  |
| `1a` | `lower.pA.baef10` | `0x00007f091abaef10` | ---4- | 1 | lower.pA.bc4e5e(x1) |  |
| `1a` | `lower.pA.bbf176` | `0x00007f091abbf176` | ---4- | 1 | lower.pA.c821c5(x1) |  |
| `1a` | `lower.pA.bc05f3` | `0x00007f091abc05f3` | ---4- | 1 | lower.pA.baef10(x1) |  |
| `1a` | `lower.pA.bc0ecd` | `0x00007f091abc0ecd` | ---4- | 1 | lower.pA.bc05f3(x1) |  |
| `1a` | `lower.pA.bc4e5e` | `0x00007f091abc4e5e` | ---4- | 1 | lower.pA.bbf176(x1) |  |
| `1a` | `lower.pA.bc82e3` | `0x00007f091abc82e3` | 1---- | 1 | lower.pA.da85b1(x1) |  |
| `1a` | `lower.pA.be7685` | `0x00007f091abe7685` | 1---- | 1 | lower.pA.e7ae14(x1) |  |
| `1a` | `lower.pA.be86d1` | `0x00007f091abe86d1` | 1---- | 1 | lower.pA.c2f88a(x1) |  |
| `1a` | `lower.pA.bfcd0e` | `0x00007f091abfcd0e` | ----5 | 1 | lower.mid.0cc677(x1) |  |
| `1a` | `lower.pA.c153cf` | `0x00007f091ac153cf` | ----5 | 1 | __memcpy_avx512_unaligned_erms(x1) |  |
| `1a` | `lower.pA.c16ff3` | `0x00007f091ac16ff3` | 1---- | 1 | lower.pA.c2dedc(x1) |  |
| `1a` | `lower.pA.c177ef` | `0x00007f091ac177ef` | --3-- | 1 | lower.pA.c40eba(x1) |  |
| `1a` | `lower.pA.c17b69` | `0x00007f091ac17b69` | 1---- | 1 | lower.pA.c440c6(x1) |  |
| `1a` | `lower.pA.c2dedc` | `0x00007f091ac2dedc` | 1---- | 1 | lower.pA.da9922(x1) |  |
| `1a` | `lower.pA.c2f88a` | `0x00007f091ac2f88a` | 1---- | 1 | lower.mid.0152f2(x1) |  |
| `1a` | `lower.pA.c40eba` | `0x00007f091ac40eba` | --3-- | 1 | lower.mid.015cb6(x1) |  |
| `1a` | `lower.pA.c440c6` | `0x00007f091ac440c6` | 1---- | 1 | lower.pA.bc82e3(x1) |  |
| `1a` | `lower.pA.c821c5` | `0x00007f091ac821c5` | ---4- | 1 | __GI___libc_free(x1) |  |
| `1a` | `lower.pA.d7a1f5` | `0x00007f091ad7a1f5` | 1---- | 1 | lower.pA.be7685(x1) |  |
| `1a` | `lower.pA.d891e3` | `0x00007f091ad891e3` | 1---- | 1 | lower.pA.d7a1f5(x1) |  |
| `1a` | `lower.pA.da693a` | `0x00007f091ada693a` | 1---- | 1 | lower.pA.d891e3(x1) |  |
| `1a` | `lower.pA.da85b1` | `0x00007f091ada85b1` | 1---- | 1 | lower.pA.c16ff3(x1) |  |
| `1a` | `lower.pA.da9922` | `0x00007f091ada9922` | 1---- | 1 | lower.pA.da693a(x1) |  |
| `1a` | `lower.pA.db2f1a` | `0x00007f091adb2f1a` | --3-- | 1 | lower.mid.00f110(x1) |  |
| `1a` | `lower.pA.db2f64` | `0x00007f091adb2f64` | --3-- | 1 | lower.mid.01543d(x1) |  |
| `1a` | `lower.pA.dfb597` | `0x00007f091adfb597` | 1---- | 1 | lower.tac.9cfaf5(x1) |  |
| `1a` | `lower.pA.e7ae14` | `0x00007f091ae7ae14` | 1---- | 1 | lower.pA.e98e8c(x1) |  |
| `1a` | `lower.pA.e8bbba` | `0x00007f091ae8bbba` | 1---- | 1 | lower.pA.e98e8c(x1) |  |
| `1a` | `lower.pA.ee2eeb` | `0x00007f091aee2eeb` | ---4- | 1 | lower.cg.ca3704(x1) |  |
| `1a` | `lower.pA.eebbef` | `0x00007f091aeebbef` | 1---- | 1 | lower.cg.ca3682(x1) |  |
| `1a` | `lower.pA.eebd05` | `0x00007f091aeebd05` | 1---- | 1 | lower.cg.ca3682(x1) |  |
| `1a` | `lower.pA.eeeab3` | `0x00007f091aeeeab3` | ---4- | 1 | lower.mid.0cef4a(x1) |  |
| `1a` | `lower.pA.eef885` | `0x00007f091aeef885` | 1---- | 1 | lower.pA.c71f47(x1) |  |
| `1a` | `lower.pA.eefb36` | `0x00007f091aeefb36` | 1---- | 1 | lower.pA.eef885(x1) |  |
| `1a` | `lower.pA.f66dc4` | `0x00007f091af66dc4` | 1---- | 1 | lower.pA.e8bbba(x1) |  |
| `1a` | `lower.pA.fec612` | `0x00007f091afec612` | --3-- | 1 | lower.mid.36fe30(x1) |  |
| `1a` | `lower.pA.fede37` | `0x00007f091afede37` | 1---- | 1 | lower.pA.c17b69(x1) |  |
| `1a` | `lower.pA.fee87e` | `0x00007f091afee87e` | --3-- | 1 | lower.pA.c177ef(x1) |  |
| `1a` | `lower.pA.ff4df8` | `0x00007f091aff4df8` | 1---- | 1 | lower.mid.024dc1(x1) |  |
| `1a` | `lower.pA.ff6a78` | `0x00007f091aff6a78` | ----5 | 1 | lower.mid.19d89d(x1) |  |
| `1a` | `lower.pA.ff6a92` | `0x00007f091aff6a92` | 1---- | 1 | lower.mid.19d60f(x1) |  |
| `1a` | `lower.pA.ff6f8d` | `0x00007f091aff6f8d` | 1---- | 1 | lower.pA.c71f47(x1) |  |
| `1a` | `lower.pA.ffe4b5` | `0x00007f091affe4b5` | 1---- | 1 | lower.pA.ff6f8d(x1) |  |
| `1a` | `lower.pA.ffe60f` | `0x00007f091affe60f` | 1---- | 1 | lower.pA.ff4df8(x1) |  |
| `1a` | `lower.pA.c71f47` | `0x00007f091ac71f47` | 1---- | 0 | — |  |
| `1b` | `lower.mid.08c5a7` | `0x00007f091b08c5a7` | 1--4- | 107 | lower.pA.ee9c0d(x66), lower.pA.ee9ec1(x22), lower.pA.ee9b85(x17) |  |
| `1b` | `lower.mid.3b6a5e` | `0x00007f091b3b6a5e` | 1---- | 39 | lower.mid.4b1b09(x39) |  |
| `1b` | `lower.mid.4b1b09` | `0x00007f091b4b1b09` | 1---- | 39 | lower.cg.ca3682(x39) |  |
| `1b` | `lower.mid.08d0c3` | `0x00007f091b08d0c3` | 1---- | 23 | lower.tac.ef01b7(x23) |  |
| `1b` | `lower.mid.492150` | `0x00007f091b492150` | 1---- | 19 | lower.mid.57b3df(x8), lower.mid.57cc1c(x4), lower.mid.57dc95(x3) |  |
| `1b` | `lower.mid.281529` | `0x00007f091b281529` | 1--4- | 17 | lower.cg.ca3682(x17) |  |
| `1b` | `lower.mid.3b6a87` | `0x00007f091b3b6a87` | 1--4- | 17 | lower.mid.281529(x17) |  |
| `1b` | `lower.mid.27f960` | `0x00007f091b27f960` | ---4- | 13 | lower.mid.37f237(x13) |  |
| `1b` | `lower.mid.37f237` | `0x00007f091b37f237` | ---4- | 13 | lower.mid.38c34c(x8), lower.mid.38f728(x2), lower.mid.38dca3(x1) |  |
| `1b` | `lower.mid.491ba7` | `0x00007f091b491ba7` | 1---- | 13 | lower.mid.564acc(x13) |  |
| `1b` | `lower.mid.564acc` | `0x00007f091b564acc` | 1---- | 13 | lower.mid.4b684c(x11), lower.mid.4b6641(x2) |  |
| `1b` | `lower.mid.4b684c` | `0x00007f091b4b684c` | 1---- | 11 | lower.cg.fae1d3(x11) |  |
| `1b` | `lower.mid.08c8b6` | `0x00007f091b08c8b6` | 1---- | 10 | lower.mid.0d0313(x5), lower.mid.0d07b4(x4), lower.mid.0d07d9(x1) |  |
| `1b` | `lower.mid.1a6fa8` | `0x00007f091b1a6fa8` | 1---- | 9 | lower.mid.18e4ba(x8), lower.mid.18e223(x1) |  |
| `1b` | `lower.mid.1a7ec0` | `0x00007f091b1a7ec0` | 1---- | 9 | lower.mid.1a6fa8(x9) |  |
| `1b` | `lower.mid.0a6fe9` | `0x00007f091b0a6fe9` | 1---- | 8 | lower.pA.be4cd4(x8) |  |
| `1b` | `lower.mid.140575` | `0x00007f091b140575` | 1---- | 8 | lower.mid.16283e(x8) |  |
| `1b` | `lower.mid.16283e` | `0x00007f091b16283e` | 1---- | 8 | lower.mid.0a6fe9(x8) |  |
| `1b` | `lower.mid.18e4ba` | `0x00007f091b18e4ba` | 1---- | 8 | lower.mid.140575(x8) |  |
| `1b` | `lower.mid.38c34c` | `0x00007f091b38c34c` | ---4- | 8 | lower.mid.5b3b21(x6), lower.mid.5b3a77(x2) |  |
| `1b` | `lower.mid.57b3df` | `0x00007f091b57b3df` | 1---- | 8 | lower.mid.4cda50(x5), lower.mid.4cd7f6(x1), lower.mid.4cd5d1(x1) |  |
| `1b` | `lower.mid.090145` | `0x00007f091b090145` | 1---- | 7 | lower.mid.0d4421(x7) |  |
| `1b` | `lower.mid.0d4421` | `0x00007f091b0d4421` | 1---- | 7 | lower.mid.47e11f(x3), lower.mid.47ea22(x2), lower.mid.47e268(x1) |  |
| `1b` | `lower.mid.371136` | `0x00007f091b371136` | 1---- | 7 | lower.mid.588f91(x4), lower.mid.588ed3(x3) |  |
| `1b` | `lower.mid.5b3b21` | `0x00007f091b5b3b21` | ---4- | 6 | lower.mid.595753(x5), lower.mid.5956b5(x1) |  |
| `1b` | `lower.mid.0d0313` | `0x00007f091b0d0313` | 1---- | 5 | lower.mid.16a2da(x4), lower.mid.16a2a5(x1) |  |
| `1b` | `lower.mid.36f8d2` | `0x00007f091b36f8d2` | 1-3-- | 5 | lower.mid.370ea8(x2), lower.mid.371136(x2), lower.mid.370e41(x1) |  |
| `1b` | `lower.mid.4cda50` | `0x00007f091b4cda50` | 1---- | 5 | lower.tac.5136bb(x3), lower.tac.5137b3(x2) |  |
| `1b` | `lower.mid.595753` | `0x00007f091b595753` | ---4- | 5 | lower.tac.5c7cd7(x3), lower.tac.5c7d2a(x2) |  |
| `1b` | `lower.mid.011560` | `0x00007f091b011560` | 1---- | 4 | lower.mid.36f8d2(x4) |  |
| `1b` | `lower.mid.08c71f` | `0x00007f091b08c71f` | 1---- | 4 | lower.mid.0c9d98(x4) |  |
| `1b` | `lower.mid.0c9d98` | `0x00007f091b0c9d98` | 1---- | 4 | lower.mid.011560(x4) |  |
| `1b` | `lower.mid.0d07b4` | `0x00007f091b0d07b4` | 1---- | 4 | lower.mid.0c9f6c(x3), lower.mid.0ca07d(x1) |  |
| `1b` | `lower.mid.16a2da` | `0x00007f091b16a2da` | 1---- | 4 | lower.pA.dfe8b6(x4) |  |
| `1b` | `lower.mid.3b6cb5` | `0x00007f091b3b6cb5` | 1--4- | 4 | lower.mid.87139d(x3), lower.mid.872699(x1) |  |
| `1b` | `lower.mid.57cc1c` | `0x00007f091b57cc1c` | 1---- | 4 | lower.tac.05ea77(x4) |  |
| `1b` | `lower.mid.588f91` | `0x00007f091b588f91` | 1---- | 4 | lower.mid.58b45d(x4) |  |
| `1b` | `lower.mid.58b45d` | `0x00007f091b58b45d` | 1---- | 4 | lower.mid.58d7ce(x2), lower.mid.58de9a(x1), lower.mid.58d990(x1) |  |
| `1b` | `lower.mid.87139d` | `0x00007f091b87139d` | 1---- | 4 | lower.mid.9d5fcd(x3), lower.mid.9d621b(x1) |  |
| `1b` | `lower.mid.08d2b8` | `0x00007f091b08d2b8` | 1---- | 3 | lower.mid.0c9702(x3) |  |
| `1b` | `lower.mid.0c6976` | `0x00007f091b0c6976` | 1---- | 3 | lower.mid.371136(x3) |  |
| `1b` | `lower.mid.0c9702` | `0x00007f091b0c9702` | 1---- | 3 | lower.mid.0ca679(x2), lower.mid.0ca583(x1) |  |
| `1b` | `lower.mid.0c9f6c` | `0x00007f091b0c9f6c` | 1---- | 3 | lower.mid.0c6976(x3) |  |
| `1b` | `lower.mid.150473` | `0x00007f091b150473` | 1-3-- | 3 | lower.pA.dd3d04(x3) |  |
| `1b` | `lower.mid.360c39` | `0x00007f091b360c39` | 1---- | 3 | lower.mid.361bcf(x2), lower.mid.361e22(x1) |  |
| `1b` | `lower.mid.36234a` | `0x00007f091b36234a` | 1---- | 3 | lower.mid.360c39(x3) |  |
| `1b` | `lower.mid.39fc31` | `0x00007f091b39fc31` | 1--4- | 3 | lower.cg.ca3682(x3) |  |
| `1b` | `lower.mid.3b68e8` | `0x00007f091b3b68e8` | 1--4- | 3 | lower.mid.39fc31(x3) |  |
| `1b` | `lower.mid.47c4e2` | `0x00007f091b47c4e2` | 1---- | 3 | lower.tac.197d53(x3) |  |
| `1b` | `lower.mid.47e11f` | `0x00007f091b47e11f` | 1---- | 3 | lower.mid.47f424(x1), lower.mid.48016b(x1), lower.mid.47f92e(x1) |  |
| `1b` | `lower.mid.57dc95` | `0x00007f091b57dc95` | 1---- | 3 | lower.mid.36234a(x3) |  |
| `1b` | `lower.mid.588ed3` | `0x00007f091b588ed3` | 1---- | 3 | lower.mid.58f59f(x1), lower.mid.58f5ee(x1), lower.mid.58eb8c(x1) |  |
| `1b` | `lower.mid.9d5fcd` | `0x00007f091b9d5fcd` | 1---- | 3 | lower.mid.a07911(x2), lower.mid.a06a55(x1) |  |
| `1b` | `lower.mid.08cae6` | `0x00007f091b08cae6` | ---4- | 2 | lower.mid.09bdc5(x1), lower.mid.09bd18(x1) |  |
| `1b` | `lower.mid.0ca07d` | `0x00007f091b0ca07d` | 1---- | 2 | lower.cg.d9cf95(x1), lower.cg.d9cfc0(x1) |  |
| `1b` | `lower.mid.0ca679` | `0x00007f091b0ca679` | 1---- | 2 | lower.mid.371136(x2) |  |
| `1b` | `lower.mid.1476b2` | `0x00007f091b1476b2` | ----5 | 2 | lower.mid.14b346(x1), lower.mid.14b0d0(x1) |  |
| `1b` | `lower.mid.147f46` | `0x00007f091b147f46` | ----5 | 2 | lower.mid.0ce320(x1), lower.mid.0ce237(x1) |  |
| `1b` | `lower.mid.361bcf` | `0x00007f091b361bcf` | 1---- | 2 | lower.tac.5855f7(x2) |  |
| `1b` | `lower.mid.362a57` | `0x00007f091b362a57` | ---4- | 2 | lower.mid.387497(x1), lower.mid.386641(x1) |  |
| `1b` | `lower.mid.370ea8` | `0x00007f091b370ea8` | 1-3-- | 2 | lower.tac.9cfaf5(x1), lower.tac.9cf956(x1) |  |
| `1b` | `lower.mid.38f728` | `0x00007f091b38f728` | ---4- | 2 | lower.mid.362a57(x2) |  |
| `1b` | `lower.mid.3b47a6` | `0x00007f091b3b47a6` | 1---- | 2 | lower.tac.13c098(x2) |  |
| `1b` | `lower.mid.3b6b09` | `0x00007f091b3b6b09` | 1--4- | 2 | lower.mid.3ebd6d(x2) |  |
| `1b` | `lower.mid.3ebd6d` | `0x00007f091b3ebd6d` | 1--4- | 2 | lower.cg.ca3682(x2) |  |
| `1b` | `lower.mid.43719a` | `0x00007f091b43719a` | 1---- | 2 | lower.mid.47c4e2(x2) |  |
| `1b` | `lower.mid.47ea22` | `0x00007f091b47ea22` | 1---- | 2 | lower.mid.43719a(x2) |  |
| `1b` | `lower.mid.492651` | `0x00007f091b492651` | 1---- | 2 | lower.mid.3b47a6(x2) |  |
| `1b` | `lower.mid.492835` | `0x00007f091b492835` | 1---- | 2 | lower.tac.4104e0(x1), lower.tac.4105e0(x1) |  |
| `1b` | `lower.mid.4b6641` | `0x00007f091b4b6641` | 1---- | 2 | lower.mid.4cbfda(x1), lower.mid.4cc4e2(x1) |  |
| `1b` | `lower.mid.57f5c3` | `0x00007f091b57f5c3` | 1---- | 2 | lower.tac.1a0956(x1), lower.tac.1a0438(x1) |  |
| `1b` | `lower.mid.58d7ce` | `0x00007f091b58d7ce` | 1---- | 2 | lower.mid.30058b(x1), lower.mid.2ffe81(x1) |  |
| `1b` | `lower.mid.5b3a77` | `0x00007f091b5b3a77` | ---4- | 2 | lower.tac.5c841f(x2) |  |
| `1b` | `lower.mid.61b645` | `0x00007f091b61b645` | 1---- | 2 | lower.mid.62412d(x2) |  |
| `1b` | `lower.mid.61f9e4` | `0x00007f091b61f9e4` | 1---- | 2 | lower.mid.61b645(x2) |  |
| `1b` | `lower.mid.62412d` | `0x00007f091b62412d` | 1---- | 2 | lower.mid.625425(x1), lower.mid.623751(x1) |  |
| `1b` | `lower.mid.626614` | `0x00007f091b626614` | 1---- | 2 | lower.mid.61f9e4(x2) |  |
| `1b` | `lower.mid.626821` | `0x00007f091b626821` | 1---- | 2 | lower.mid.626614(x2) |  |
| `1b` | `lower.mid.6657b0` | `0x00007f091b6657b0` | 1--4- | 2 | lower.mid.661b6c(x1), lower.mid.661be6(x1) |  |
| `1b` | `lower.mid.a026cf` | `0x00007f091ba026cf` | 1--4- | 2 | lower.mid.6657b0(x2) |  |
| `1b` | `lower.mid.a06a55` | `0x00007f091ba06a55` | 1--4- | 2 | lower.mid.a026cf(x2) |  |
| `1b` | `lower.mid.a07911` | `0x00007f091ba07911` | 1---- | 2 | lower.mid.a13641(x1), lower.mid.a13a94(x1) |  |
| `1b` | `lower.mid.00f110` | `0x00007f091b00f110` | --3-- | 1 | lower.pA.fec612(x1) |  |
| `1b` | `lower.mid.0152f2` | `0x00007f091b0152f2` | 1---- | 1 | lower.pA.fede37(x1) |  |
| `1b` | `lower.mid.01543d` | `0x00007f091b01543d` | --3-- | 1 | lower.pA.fee87e(x1) |  |
| `1b` | `lower.mid.015cb6` | `0x00007f091b015cb6` | --3-- | 1 | lower.tac.9cee48(x1) |  |
| `1b` | `lower.mid.019f94` | `0x00007f091b019f94` | 1---- | 1 | lower.mid.1206af(x1) |  |
| `1b` | `lower.mid.08c94a` | `0x00007f091b08c94a` | 1---- | 1 | lower.pA.f66dc4(x1) |  |
| `1b` | `lower.mid.08f692` | `0x00007f091b08f692` | 1---- | 1 | lower.pA.ffe60f(x1) |  |
| `1b` | `lower.mid.09bd18` | `0x00007f091b09bd18` | ---4- | 1 | lower.mid.225755(x1) |  |
| `1b` | `lower.mid.09bdc5` | `0x00007f091b09bdc5` | ---4- | 1 | lower.pA.bc0ecd(x1) |  |
| `1b` | `lower.mid.09eba5` | `0x00007f091b09eba5` | ----5 | 1 | __GI___libc_free(x1) |  |
| `1b` | `lower.mid.0ac2f2` | `0x00007f091b0ac2f2` | ----5 | 1 | lower.pA.c153cf(x1) |  |
| `1b` | `lower.mid.0ca583` | `0x00007f091b0ca583` | 1---- | 1 | lower.pA.ffe4b5(x1) |  |
| `1b` | `lower.mid.0cc677` | `0x00007f091b0cc677` | ----5 | 1 | lower.mid.0ac2f2(x1) |  |
| `1b` | `lower.mid.0ce237` | `0x00007f091b0ce237` | ----5 | 1 | lower.pA.bfcd0e(x1) |  |
| `1b` | `lower.mid.0ce320` | `0x00007f091b0ce320` | ----5 | 1 | lower.mid.09eba5(x1) |  |
| `1b` | `lower.mid.0cef4a` | `0x00007f091b0cef4a` | ---4- | 1 | lower.mid.14bf71(x1) |  |
| `1b` | `lower.mid.0d07d9` | `0x00007f091b0d07d9` | 1---- | 1 | lower.mid.0ca07d(x1) |  |
| `1b` | `lower.mid.1206af` | `0x00007f091b1206af` | 1---- | 1 | lower.tac.a6ed32(x1) |  |
| `1b` | `lower.mid.12f7f9` | `0x00007f091b12f7f9` | ----5 | 1 | __GI___libc_free(x1) |  |
| `1b` | `lower.mid.14b0d0` | `0x00007f091b14b0d0` | ----5 | 1 | lower.pA.ff6a78(x1) |  |
| `1b` | `lower.mid.14b346` | `0x00007f091b14b346` | ----5 | 1 | lower.mid.17fb3b(x1) |  |
| `1b` | `lower.mid.14bf71` | `0x00007f091b14bf71` | ---4- | 1 | lower.pA.ee2eeb(x1) |  |
| `1b` | `lower.mid.150f7f` | `0x00007f091b150f7f` | 1---- | 1 | lower.pA.ff6a92(x1) |  |
| `1b` | `lower.mid.16a2a5` | `0x00007f091b16a2a5` | 1---- | 1 | lower.pA.dfb597(x1) |  |
| `1b` | `lower.mid.17fb3b` | `0x00007f091b17fb3b` | ----5 | 1 | lower.mid.12f7f9(x1) |  |
| `1b` | `lower.mid.18e223` | `0x00007f091b18e223` | 1---- | 1 | lower.pA.b427e3(x1) |  |
| `1b` | `lower.mid.19d89d` | `0x00007f091b19d89d` | ----5 | 1 | lower.tac.9e453a(x1) |  |
| `1b` | `lower.mid.1a750e` | `0x00007f091b1a750e` | 1---- | 1 | lower.mid.150f7f(x1) |  |
| `1b` | `lower.mid.1a7528` | `0x00007f091b1a7528` | 1---- | 1 | lower.mid.18f900(x1) |  |
| `1b` | `lower.mid.1a7f46` | `0x00007f091b1a7f46` | 1---- | 1 | lower.tac.1f119c(x1) |  |
| `1b` | `lower.mid.1bc2b7` | `0x00007f091b1bc2b7` | ---4- | 1 | lower.mid.1bf856(x1) |  |
| `1b` | `lower.mid.225755` | `0x00007f091b225755` | ---4- | 1 | lower.mid.1bc2b7(x1) |  |
| `1b` | `lower.mid.277ec1` | `0x00007f091b277ec1` | ---4- | 1 | lower.tac.f0e86d(x1) |  |
| `1b` | `lower.mid.27f1f5` | `0x00007f091b27f1f5` | ---4- | 1 | lower.mid.38017a(x1) |  |
| `1b` | `lower.mid.27fb52` | `0x00007f091b27fb52` | ---4- | 1 | lower.mid.3b8379(x1) |  |
| `1b` | `lower.mid.2a6655` | `0x00007f091b2a6655` | 1---- | 1 | lower.mid.34c911(x1) |  |
| `1b` | `lower.mid.2a8ed0` | `0x00007f091b2a8ed0` | 1---- | 1 | lower.tac.ed48a7(x1) |  |
| `1b` | `lower.mid.2db4b8` | `0x00007f091b2db4b8` | 1---- | 1 | __GI___libc_malloc(x1) |  |
| `1b` | `lower.mid.2fcdf5` | `0x00007f091b2fcdf5` | 1---- | 1 | lower.mid.2a8ed0(x1) |  |
| `1b` | `lower.mid.2ffe81` | `0x00007f091b2ffe81` | 1---- | 1 | lower.tac.a995e5(x1) |  |
| `1b` | `lower.mid.30058b` | `0x00007f091b30058b` | 1---- | 1 | lower.mid.2fcdf5(x1) |  |
| `1b` | `lower.mid.34c911` | `0x00007f091b34c911` | 1---- | 1 | lower.mid.48367e(x1) |  |
| `1b` | `lower.mid.360624` | `0x00007f091b360624` | ---4- | 1 | lower.mid.50fbb0(x1) |  |
| `1b` | `lower.mid.36076a` | `0x00007f091b36076a` | 1---- | 1 | lower.tac.4dbfba(x1) |  |
| `1b` | `lower.mid.361e22` | `0x00007f091b361e22` | 1---- | 1 | lower.tac.588880(x1) |  |
| `1b` | `lower.mid.36fe30` | `0x00007f091b36fe30` | --3-- | 1 | lower.mid.3702a3(x1) |  |
| `1b` | `lower.mid.3702a3` | `0x00007f091b3702a3` | --3-- | 1 | lower.mid.36f8d2(x1) |  |
| `1b` | `lower.mid.370e41` | `0x00007f091b370e41` | 1---- | 1 | lower.mid.36854d(x1) |  |
| `1b` | `lower.mid.3731f1` | `0x00007f091b3731f1` | ---4- | 1 | lower.tac.b9f0fd(x1) |  |
| `1b` | `lower.mid.379827` | `0x00007f091b379827` | ---4- | 1 | lower.mid.3731f1(x1) |  |
| `1b` | `lower.mid.38017a` | `0x00007f091b38017a` | ---4- | 1 | lower.tac.b9f0fd(x1) |  |
| `1b` | `lower.mid.382783` | `0x00007f091b382783` | ---4- | 1 | lower.tac.4db9ea(x1) |  |
| `1b` | `lower.mid.386641` | `0x00007f091b386641` | ---4- | 1 | lower.mid.360624(x1) |  |
| `1b` | `lower.mid.387497` | `0x00007f091b387497` | ---4- | 1 | lower.mid.382783(x1) |  |
| `1b` | `lower.mid.38d4e1` | `0x00007f091b38d4e1` | ---4- | 1 | lower.mid.5bc3ef(x1) |  |
| `1b` | `lower.mid.38db7c` | `0x00007f091b38db7c` | ---4- | 1 | lower.mid.379827(x1) |  |
| `1b` | `lower.mid.38dca3` | `0x00007f091b38dca3` | ---4- | 1 | lower.mid.3939eb(x1) |  |
| `1b` | `lower.mid.3939eb` | `0x00007f091b3939eb` | ---4- | 1 | lower.tac.9f2dcd(x1) |  |
| `1b` | `lower.mid.3b6c75` | `0x00007f091b3b6c75` | 1---- | 1 | lower.mid.4b19b9(x1) |  |
| `1b` | `lower.mid.3b79b0` | `0x00007f091b3b79b0` | ---4- | 1 | lower.tac.19ecbb(x1) |  |
| `1b` | `lower.mid.3b8379` | `0x00007f091b3b8379` | ---4- | 1 | lower.mid.3b79b0(x1) |  |
| `1b` | `lower.mid.4341e7` | `0x00007f091b4341e7` | 1---- | 1 | lower.tac.391f24(x1) |  |
| `1b` | `lower.mid.479a4e` | `0x00007f091b479a4e` | 1---- | 1 | lower.mid.47ae7e(x1) |  |
| `1b` | `lower.mid.47ae7e` | `0x00007f091b47ae7e` | 1---- | 1 | lower.mid.4341e7(x1) |  |
| `1b` | `lower.mid.47b313` | `0x00007f091b47b313` | 1---- | 1 | lower.tac.195f7b(x1) |  |
| `1b` | `lower.mid.47b81a` | `0x00007f091b47b81a` | 1---- | 1 | lower.tac.065159(x1) |  |
| `1b` | `lower.mid.47dee5` | `0x00007f091b47dee5` | 1---- | 1 | lower.mid.51cebe(x1) |  |
| `1b` | `lower.mid.47e268` | `0x00007f091b47e268` | 1---- | 1 | lower.mid.56eb91(x1) |  |
| `1b` | `lower.mid.47f424` | `0x00007f091b47f424` | 1---- | 1 | lower.mid.47b81a(x1) |  |
| `1b` | `lower.mid.47f92e` | `0x00007f091b47f92e` | 1---- | 1 | lower.mid.47b313(x1) |  |
| `1b` | `lower.mid.48016b` | `0x00007f091b48016b` | 1---- | 1 | lower.mid.47c4e2(x1) |  |
| `1b` | `lower.mid.4814f3` | `0x00007f091b4814f3` | 1---- | 1 | lower.mid.479a4e(x1) |  |
| `1b` | `lower.mid.48367e` | `0x00007f091b48367e` | 1---- | 1 | lower.mid.4b4f24(x1) |  |
| `1b` | `lower.mid.491fb0` | `0x00007f091b491fb0` | 1---- | 1 | lower.mid.4f85f6(x1) |  |
| `1b` | `lower.mid.4922c9` | `0x00007f091b4922c9` | 1---- | 1 | lower.tac.1caad1(x1) |  |
| `1b` | `lower.mid.492873` | `0x00007f091b492873` | 1---- | 1 | lower.mid.4766c0(x1) |  |
| `1b` | `lower.mid.4b19b9` | `0x00007f091b4b19b9` | 1---- | 1 | lower.cg.ca3682(x1) |  |
| `1b` | `lower.mid.4b4f24` | `0x00007f091b4b4f24` | 1---- | 1 | lower.cg.fac8d6(x1) |  |
| `1b` | `lower.mid.4cbfda` | `0x00007f091b4cbfda` | 1---- | 1 | lower.tac.aee621(x1) |  |
| `1b` | `lower.mid.4cc4e2` | `0x00007f091b4cc4e2` | 1---- | 1 | __GI___libc_free(x1) |  |
| `1b` | `lower.mid.4cd5d1` | `0x00007f091b4cd5d1` | 1---- | 1 | lower.cg.d9babd(x1) |  |
| `1b` | `lower.mid.4cd7f6` | `0x00007f091b4cd7f6` | 1---- | 1 | lower.tac.ac204f(x1) |  |
| `1b` | `lower.mid.4cdabe` | `0x00007f091b4cdabe` | 1---- | 1 | lower.tac.512afa(x1) |  |
| `1b` | `lower.mid.4e732f` | `0x00007f091b4e732f` | 1---- | 1 | lower.mid.4f28da(x1) |  |
| `1b` | `lower.mid.4e7ab0` | `0x00007f091b4e7ab0` | 1---- | 1 | lower.mid.4814f3(x1) |  |
| `1b` | `lower.mid.4f28da` | `0x00007f091b4f28da` | 1---- | 1 | __memcmp_evex_movbe(x1) |  |
| `1b` | `lower.mid.4f85f6` | `0x00007f091b4f85f6` | 1---- | 1 | lower.mid.2a6655(x1) |  |
| `1b` | `lower.mid.511f1a` | `0x00007f091b511f1a` | 1---- | 1 | lower.mid.256c60(x1) |  |
| `1b` | `lower.mid.512038` | `0x00007f091b512038` | 1---- | 1 | lower.mid.511f1a(x1) |  |
| `1b` | `lower.mid.51cebe` | `0x00007f091b51cebe` | 1---- | 1 | lower.mid.512038(x1) |  |
| `1b` | `lower.mid.56dc6a` | `0x00007f091b56dc6a` | 1---- | 1 | lower.mid.4e732f(x1) |  |
| `1b` | `lower.mid.56eb91` | `0x00007f091b56eb91` | 1---- | 1 | lower.mid.4e7ab0(x1) |  |
| `1b` | `lower.mid.57a675` | `0x00007f091b57a675` | 1---- | 1 | lower.mid.580ba0(x1) |  |
| `1b` | `lower.mid.57db53` | `0x00007f091b57db53` | 1---- | 1 | lower.mid.587414(x1) |  |
| `1b` | `lower.mid.580ba0` | `0x00007f091b580ba0` | 1---- | 1 | lower.tac.5db598(x1) |  |
| `1b` | `lower.mid.587414` | `0x00007f091b587414` | 1---- | 1 | lower.tac.51af4c(x1) |  |
| `1b` | `lower.mid.589230` | `0x00007f091b589230` | 1---- | 1 | lower.mid.2db4b8(x1) |  |
| `1b` | `lower.mid.58d990` | `0x00007f091b58d990` | 1---- | 1 | lower.mid.589230(x1) |  |
| `1b` | `lower.mid.58de9a` | `0x00007f091b58de9a` | 1---- | 1 | lower.mid.enter(x1) |  |
| `1b` | `lower.mid.58eb8c` | `0x00007f091b58eb8c` | 1---- | 1 | lower.tac.4e5cdc(x1) |  |
| `1b` | `lower.mid.58f59f` | `0x00007f091b58f59f` | 1---- | 1 | lower.mid.599873(x1) |  |
| `1b` | `lower.mid.58f5ee` | `0x00007f091b58f5ee` | 1---- | 1 | lower.mid.36076a(x1) |  |
| `1b` | `lower.mid.5956b5` | `0x00007f091b5956b5` | ---4- | 1 | lower.tac.4db6a5(x1) |  |
| `1b` | `lower.mid.599873` | `0x00007f091b599873` | 1---- | 1 | lower.mid.56dc6a(x1) |  |
| `1b` | `lower.mid.622476` | `0x00007f091b622476` | 1---- | 1 | lower.mid.66b905(x1) |  |
| `1b` | `lower.mid.623751` | `0x00007f091b623751` | 1---- | 1 | lower.mid.61eec6(x1) |  |
| `1b` | `lower.mid.625425` | `0x00007f091b625425` | 1---- | 1 | lower.mid.625537(x1) |  |
| `1b` | `lower.mid.625537` | `0x00007f091b625537` | 1---- | 1 | lower.mid.6258d5(x1) |  |
| `1b` | `lower.mid.6258d5` | `0x00007f091b6258d5` | 1---- | 1 | lower.mid.622476(x1) |  |
| `1b` | `lower.mid.644f88` | `0x00007f091b644f88` | 1---- | 1 | lower.mid.6523cb(x1) |  |
| `1b` | `lower.mid.65cb36` | `0x00007f091b65cb36` | 1---- | 1 | __GI___libc_malloc(x1) |  |
| `1b` | `lower.mid.661107` | `0x00007f091b661107` | 1---- | 1 | lower.mid.644f88(x1) |  |
| `1b` | `lower.mid.661be6` | `0x00007f091b661be6` | ---4- | 1 | lower.mid.66f798(x1) |  |
| `1b` | `lower.mid.661c84` | `0x00007f091b661c84` | 1---- | 1 | lower.mid.65cb36(x1) |  |
| `1b` | `lower.mid.66b905` | `0x00007f091b66b905` | 1---- | 1 | lower.mid.622660(x1) |  |
| `1b` | `lower.mid.66f798` | `0x00007f091b66f798` | ---4- | 1 | lower.mid.9bfc91(x1) |  |
| `1b` | `lower.mid.7f9739` | `0x00007f091b7f9739` | 1---- | 1 | lower.mid.801544(x1) |  |
| `1b` | `lower.mid.801544` | `0x00007f091b801544` | 1---- | 1 | lower.mid.8105bb(x1) |  |
| `1b` | `lower.mid.8105bb` | `0x00007f091b8105bb` | 1---- | 1 | lower.mid.7cb33d(x1) |  |
| `1b` | `lower.mid.87146a` | `0x00007f091b87146a` | 1---- | 1 | lower.mid.9b400e(x1) |  |
| `1b` | `lower.mid.9b400e` | `0x00007f091b9b400e` | 1---- | 1 | lower.mid.a8a72b(x1) |  |
| `1b` | `lower.mid.9b8bd1` | `0x00007f091b9b8bd1` | 1---- | 1 | lower.mid.7f9739(x1) |  |
| `1b` | `lower.mid.9bfb4c` | `0x00007f091b9bfb4c` | 1---- | 1 | lower.mid.a0a506(x1) |  |
| `1b` | `lower.mid.9bfc91` | `0x00007f091b9bfc91` | ---4- | 1 | lower.mid.997bcc(x1) |  |
| `1b` | `lower.mid.9d621b` | `0x00007f091b9d621b` | 1---- | 1 | lower.mid.9b8bd1(x1) |  |
| `1b` | `lower.mid.9da9b8` | `0x00007f091b9da9b8` | 1---- | 1 | lower.mid.626821(x1) |  |
| `1b` | `lower.mid.9fda84` | `0x00007f091b9fda84` | ---4- | 1 | lower.mid.a01389(x1) |  |
| `1b` | `lower.mid.9fdddf` | `0x00007f091b9fdddf` | 1---- | 1 | lower.mid.661c84(x1) |  |
| `1b` | `lower.mid.9fe5c3` | `0x00007f091b9fe5c3` | 1---- | 1 | lower.tac.1f0159(x1) |  |
| `1b` | `lower.mid.9ff5e1` | `0x00007f091b9ff5e1` | 1---- | 1 | lower.mid.9da9b8(x1) |  |
| `1b` | `lower.mid.a01389` | `0x00007f091ba01389` | ---4- | 1 | lower.mid.a06a55(x1) |  |
| `1b` | `lower.mid.a0a506` | `0x00007f091ba0a506` | 1---- | 1 | lower.mid.661107(x1) |  |
| `1b` | `lower.mid.a0a8e4` | `0x00007f091ba0a8e4` | 1---- | 1 | lower.mid.a1813a(x1) |  |
| `1b` | `lower.mid.a13641` | `0x00007f091ba13641` | 1---- | 1 | __GI___libc_malloc(x1) |  |
| `1b` | `lower.mid.a13a94` | `0x00007f091ba13a94` | 1---- | 1 | lower.mid.a0a8e4(x1) |  |
| `1b` | `lower.mid.a1813a` | `0x00007f091ba1813a` | 1---- | 1 | lower.mid.9bfb4c(x1) |  |
| `1b` | `lower.mid.a84123` | `0x00007f091ba84123` | 1---- | 1 | lower.mid.626821(x1) |  |
| `1b` | `lower.mid.a8a72b` | `0x00007f091ba8a72b` | 1---- | 1 | lower.mid.a84123(x1) |  |
| `1b` | `lower.mid.bc268d` | `0x00007f091bbc268d` | -2--- | 1 | lower.mid.d176d9(x1) |  |
| `1b` | `lower.mid.cba07f` | `0x00007f091bcba07f` | -2--- | 1 | lower.tac.ed8d41(x1) |  |
| `1b` | `lower.mid.cc55de` | `0x00007f091bcc55de` | -2--- | 1 | lower.mid.cba07f(x1) |  |
| `1b` | `lower.mid.d16231` | `0x00007f091bd16231` | -2--- | 1 | lower.mid.cc55de(x1) |  |
| `1b` | `lower.mid.d16d4f` | `0x00007f091bd16d4f` | -2--- | 1 | lower.mid.e608b2(x1) |  |
| `1b` | `lower.mid.d176d9` | `0x00007f091bd176d9` | -2--- | 1 | lower.mid.d16d4f(x1) |  |
| `1b` | `lower.mid.e608b2` | `0x00007f091be608b2` | -2--- | 1 | lower.mid.d16231(x1) |  |
| `1b` | `lower.mid.enter` | `0x00007f091b2fe721` | 1---- | 1 | lower.tac.aa2f1a(x1) | entry into mid-level lowering pass |
| `1b` | `lower.mid.024dc1` | `0x00007f091b024dc1` | 1---- | 0 | — |  |
| `1b` | `lower.mid.18f900` | `0x00007f091b18f900` | 1---- | 0 | — |  |
| `1b` | `lower.mid.19d60f` | `0x00007f091b19d60f` | 1---- | 0 | — |  |
| `1b` | `lower.mid.1bf856` | `0x00007f091b1bf856` | ---4- | 0 | — |  |
| `1b` | `lower.mid.256c60` | `0x00007f091b256c60` | 1---- | 0 | — |  |
| `1b` | `lower.mid.36854d` | `0x00007f091b36854d` | 1---- | 0 | — |  |
| `1b` | `lower.mid.4766c0` | `0x00007f091b4766c0` | 1---- | 0 | — |  |
| `1b` | `lower.mid.50fbb0` | `0x00007f091b50fbb0` | ---4- | 0 | — |  |
| `1b` | `lower.mid.5bc3ef` | `0x00007f091b5bc3ef` | ---4- | 0 | — |  |
| `1b` | `lower.mid.61eec6` | `0x00007f091b61eec6` | 1---- | 0 | — |  |
| `1b` | `lower.mid.622660` | `0x00007f091b622660` | 1---- | 0 | — |  |
| `1b` | `lower.mid.6523cb` | `0x00007f091b6523cb` | 1---- | 0 | — |  |
| `1b` | `lower.mid.661b6c` | `0x00007f091b661b6c` | 1---- | 0 | — |  |
| `1b` | `lower.mid.7cb33d` | `0x00007f091b7cb33d` | 1---- | 0 | — |  |
| `1b` | `lower.mid.872699` | `0x00007f091b872699` | ---4- | 0 | — |  |
| `1b` | `lower.mid.997bcc` | `0x00007f091b997bcc` | ---4- | 0 | — |  |
| `1c` | `lower.cg.ca3682` | `0x00007f091cca3682` | 1--4- | 65 | lower.cg.ca12c1(x64), lower.cg.ca12f1(x1) |  |
| `1c` | `lower.cg.ca12c1` | `0x00007f091cca12c1` | 1--4- | 64 | lower.tac.13a8cd(x63), lower.tac.13a9ca(x1) |  |
| `1c` | `lower.cg.ca1c6d` | `0x00007f091cca1c6d` | 1--4- | 58 | lower.mid.492150(x19), lower.mid.491ba7(x13), lower.mid.27f960(x13) |  |
| `1c` | `lower.cg.817557` | `0x00007f091c817557` | 1--4- | 21 | lower.cg.81b39e(x21) |  |
| `1c` | `lower.cg.81b39e` | `0x00007f091c81b39e` | 1--4- | 21 | lower.cg.901d2b(x21) |  |
| `1c` | `lower.cg.901bdd` | `0x00007f091c901bdd` | 1--4- | 21 | lower.cg.968e2e(x11), lower.cg.96aa70(x3), lower.cg.96a574(x2) |  |
| `1c` | `lower.cg.901d2b` | `0x00007f091c901d2b` | 1--4- | 21 | lower.cg.901bdd(x21) |  |
| `1c` | `lower.cg.9790a6` | `0x00007f091c9790a6` | 1---- | 12 | lower.cg.97a4f9(x11), lower.cg.97a369(x1) |  |
| `1c` | `lower.cg.968e2e` | `0x00007f091c968e2e` | 1---- | 11 | lower.cg.9790a6(x11) |  |
| `1c` | `lower.cg.97a4f9` | `0x00007f091c97a4f9` | 1---- | 11 | lower.cg.985795(x11) |  |
| `1c` | `lower.cg.97ccbf` | `0x00007f091c97ccbf` | 1---- | 11 | __cxa_thread_atexit_impl(x11) |  |
| `1c` | `lower.cg.985795` | `0x00007f091c985795` | 1---- | 11 | lower.cg.97ccbf(x11) |  |
| `1c` | `lower.cg.fae1d3` | `0x00007f091cfae1d3` | 1---- | 11 | lower.cg.fad865(x9), lower.cg.fadbdd(x2) |  |
| `1c` | `lower.cg.fad865` | `0x00007f091cfad865` | 1---- | 9 | lower.tac.55d6c9(x9) |  |
| `1c` | `lower.cg.ed2205` | `0x00007f091ced2205` | 1---- | 5 | lower.tac.1bdcc5(x5) |  |
| `1c` | `lower.cg.96aa70` | `0x00007f091c96aa70` | 1---- | 3 | lower.cg.99bd25(x2), lower.cg.99bedc(x1) |  |
| `1c` | `lower.cg.ee34cc` | `0x00007f091cee34cc` | 1---- | 3 | __GI___libc_realloc(x3) |  |
| `1c` | `lower.cg.96a574` | `0x00007f091c96a574` | 1---- | 2 | lower.tac.068c6e(x2) |  |
| `1c` | `lower.cg.96a5c0` | `0x00007f091c96a5c0` | 1---- | 2 | lower.tac.54c683(x2) |  |
| `1c` | `lower.cg.99bd25` | `0x00007f091c99bd25` | 1---- | 2 | lower.tac.05ece6(x2) |  |
| `1c` | `lower.cg.ca13f6` | `0x00007f091cca13f6` | 1---- | 2 | lower.tac.188a51(x1), lower.tac.188a7c(x1) |  |
| `1c` | `lower.cg.ee36be` | `0x00007f091cee36be` | 1---- | 2 | lower.cg.ee34cc(x2) |  |
| `1c` | `lower.cg.fadbdd` | `0x00007f091cfadbdd` | 1---- | 2 | lower.tac.5b7ca8(x2) |  |
| `1c` | `lower.cg.817375` | `0x00007f091c817375` | 1---- | 1 | lower.cg.9674b1(x1) |  |
| `1c` | `lower.cg.9674b1` | `0x00007f091c9674b1` | 1---- | 1 | lower.cg.9677d7(x1) |  |
| `1c` | `lower.cg.9677d7` | `0x00007f091c9677d7` | 1---- | 1 | lower.drv.5e6670(x1) |  |
| `1c` | `lower.cg.969065` | `0x00007f091c969065` | 1---- | 1 | lower.cg.9790a6(x1) |  |
| `1c` | `lower.cg.969ea8` | `0x00007f091c969ea8` | 1---- | 1 | lower.cg.9b0749(x1) |  |
| `1c` | `lower.cg.96aceb` | `0x00007f091c96aceb` | ---4- | 1 | lower.cg.97b7fb(x1) |  |
| `1c` | `lower.cg.97a369` | `0x00007f091c97a369` | 1---- | 1 | lower.cg.97b5db(x1) |  |
| `1c` | `lower.cg.97b5db` | `0x00007f091c97b5db` | 1---- | 1 | lower.tac.54b2e4(x1) |  |
| `1c` | `lower.cg.97b7fb` | `0x00007f091c97b7fb` | ---4- | 1 | lower.tac.58535f(x1) |  |
| `1c` | `lower.cg.99bedc` | `0x00007f091c99bedc` | 1---- | 1 | lower.cg.99cfbf(x1) |  |
| `1c` | `lower.cg.99cfbf` | `0x00007f091c99cfbf` | 1---- | 1 | lower.tac.05ece6(x1) |  |
| `1c` | `lower.cg.9b0749` | `0x00007f091c9b0749` | 1---- | 1 | lower.drv.5e6670(x1) |  |
| `1c` | `lower.cg.9b4b86` | `0x00007f091c9b4b86` | 1---- | 1 | lower.cg.ec6674(x1) |  |
| `1c` | `lower.cg.a37c06` | `0x00007f091ca37c06` | 1---- | 1 | lower.cg.a440bf(x1) |  |
| `1c` | `lower.cg.a581d2` | `0x00007f091ca581d2` | 1---- | 1 | lower.cg.a64538(x1) |  |
| `1c` | `lower.cg.a64538` | `0x00007f091ca64538` | 1---- | 1 | lower.tac.398b0d(x1) |  |
| `1c` | `lower.cg.ca12f1` | `0x00007f091cca12f1` | 1---- | 1 | lower.tac.1b198b(x1) |  |
| `1c` | `lower.cg.ca27c8` | `0x00007f091cca27c8` | ---4- | 1 | lower.tac.f08f0e(x1) |  |
| `1c` | `lower.cg.ca27ef` | `0x00007f091cca27ef` | ---4- | 1 | lower.tac.188a51(x1) |  |
| `1c` | `lower.cg.ca3704` | `0x00007f091cca3704` | ---4- | 1 | lower.cg.ca3682(x1) |  |
| `1c` | `lower.cg.cb2328` | `0x00007f091ccb2328` | 1---- | 1 | lower.cg.a34561(x1) |  |
| `1c` | `lower.cg.d9babd` | `0x00007f091cd9babd` | 1---- | 1 | lower.cg.a37c06(x1) |  |
| `1c` | `lower.cg.d9cfc0` | `0x00007f091cd9cfc0` | 1---- | 1 | __GI___libc_free(x1) |  |
| `1c` | `lower.cg.ec6674` | `0x00007f091cec6674` | 1---- | 1 | lower.cg.ee34cc(x1) |  |
| `1c` | `lower.cg.ed22d9` | `0x00007f091ced22d9` | 1---- | 1 | lower.tac.b8e7a1(x1) |  |
| `1c` | `lower.cg.ee34ea` | `0x00007f091cee34ea` | 1---- | 1 | __GI___libc_malloc(x1) |  |
| `1c` | `lower.cg.fac8d6` | `0x00007f091cfac8d6` | 1---- | 1 | lower.tac.52232f(x1) |  |
| `1c` | `lower.cg.a34561` | `0x00007f091ca34561` | 1---- | 0 | — |  |
| `1c` | `lower.cg.a440bf` | `0x00007f091ca440bf` | 1---- | 0 | — |  |
| `1c` | `lower.cg.ca14f4` | `0x00007f091cca14f4` | 1---- | 0 | — |  |
| `1c` | `lower.cg.d9cf95` | `0x00007f091cd9cf95` | 1---- | 0 | — |  |
| `1c` | `lower.cg.db09b0` | `0x00007f091cdb09b0` | 1---- | 0 | — |  |
| `1d` | `lower.visit.sync` | `0x00007f091ddf26e4` | 1-345 | 995 | lower.drv.82e3b0(x750), lower.visit.driver(x202), lower.drv.82e3c2(x27) | lowering visitor sync/branch (region 1d) |
| `1d` | `lower.tac.df235b` | `0x00007f091ddf235b` | 12345 | 640 | lower.visit.sync(x390), lower.tac.df2663(x250) |  |
| `1d` | `lower.tac.df40ea` | `0x00007f091ddf40ea` | 12345 | 640 | lower.tac.df235b(x640) |  |
| `1d` | `lower.tac.df76df` | `0x00007f091ddf76df` | 12345 | 640 | lower.tac.df40ea(x640) |  |
| `1d` | `lower.tac.ef526f` | `0x00007f091def526f` | 12345 | 640 | lower.tac.df76df(x640) |  |
| `1d` | `compile.wait.SYSCALL` | `0x00007f091ded6c05` | 12345 | 258 | syscall(x258) | ★ driver blocks here -> syscall (waits on worker pool to finish lowering) |
| `1d` | `lower.tac.df2663` | `0x00007f091ddf2663` | -23-- | 254 | lower.tac.df4e29(x254) |  |
| `1d` | `lower.tac.df4e29` | `0x00007f091ddf4e29` | -23-- | 254 | compile.wait.SYSCALL(x254) |  |
| `1d` | `lower.tac.ed6746` | `0x00007f091ded6746` | ---45 | 227 | syscall(x227) |  |
| `1d` | `lower.tac.13a8cd` | `0x00007f091d13a8cd` | 1--4- | 63 | lower.cg.ca1c6d(x58), lower.cg.ca13f6(x2), lower.cg.ca14f4(x1) |  |
| `1d` | `lower.tac.ef01b7` | `0x00007f091def01b7` | 1---- | 23 | syscall(x23) |  |
| `1d` | `lower.tac.a7bebc` | `0x00007f091da7bebc` | 1--4- | 15 | lower.tac.aacec7(x7), lower.tac.aac8af(x5), lower.tac.aad154(x2) |  |
| `1d` | `lower.tac.066253` | `0x00007f091d066253` | 1---- | 12 | lower.tac.54d6c5(x10), lower.tac.54d622(x2) |  |
| `1d` | `lower.tac.54d6c5` | `0x00007f091d54d6c5` | 1---- | 10 | lower.tac.a8e99d(x5), lower.tac.a8ea0b(x5) |  |
| `1d` | `lower.tac.9e6fbc` | `0x00007f091d9e6fbc` | 1--45 | 10 | __GI___libc_realloc(x10) |  |
| `1d` | `lower.tac.55d6c9` | `0x00007f091d55d6c9` | 1---- | 9 | lower.tac.55e55d(x3), lower.tac.55f2b4(x3), lower.tac.55f1ef(x1) |  |
| `1d` | `lower.tac.aacec7` | `0x00007f091daacec7` | 1--4- | 9 | lower.tac.8fb792(x7), lower.tac.8fb698(x2) |  |
| `1d` | `lower.tac.a817d5` | `0x00007f091da817d5` | 1--4- | 8 | lower.tac.a7bebc(x8) |  |
| `1d` | `lower.tac.a9c157` | `0x00007f091da9c157` | 1--4- | 8 | lower.tac.aacec7(x2), lower.tac.aac8af(x2), lower.tac.aacc41(x2) |  |
| `1d` | `lower.tac.aac8af` | `0x00007f091daac8af` | 1---- | 8 | lower.tac.a9c157(x6), lower.tac.a9cb8d(x1), lower.tac.a9bee3(x1) |  |
| `1d` | `lower.tac.05ee54` | `0x00007f091d05ee54` | 1---- | 7 | lower.tac.066253(x7) |  |
| `1d` | `lower.tac.8fb792` | `0x00007f091d8fb792` | 1--4- | 7 | lower.tac.bbb7a6(x4), lower.tac.bbb839(x3) |  |
| `1d` | `lower.tac.aa2f1a` | `0x00007f091daa2f1a` | 1---- | 7 | lower.tac.a817d5(x6), lower.tac.a81858(x1) |  |
| `1d` | `lower.tac.1bdcc5` | `0x00007f091d1bdcc5` | 1---- | 6 | lower.cg.ed2205(x5), lower.cg.ed22d9(x1) |  |
| `1d` | `lower.tac.a82272` | `0x00007f091da82272` | 1---- | 6 | lower.tac.a7bebc(x6) |  |
| `1d` | `lower.tac.a8e761` | `0x00007f091da8e761` | 1---- | 6 | lower.tac.a905dd(x3), lower.tac.a903bf(x1), lower.tac.a905fd(x1) |  |
| `1d` | `lower.tac.a8e99d` | `0x00007f091da8e99d` | 1---- | 6 | lower.tac.a8e761(x6) |  |
| `1d` | `lower.tac.53ee41` | `0x00007f091d53ee41` | 1---- | 5 | lower.tac.aa2f1a(x5) |  |
| `1d` | `lower.tac.9e48ba` | `0x00007f091d9e48ba` | 1---- | 5 | lower.tac.9e6fbc(x5) |  |
| `1d` | `lower.tac.a8ea0b` | `0x00007f091da8ea0b` | 1---- | 5 | lower.tac.a9132d(x3), lower.tac.a9136b(x2) |  |
| `1d` | `compile.sync` | `0x00007f091ddf6109` | 1-345 | 4 | compile.wait.SYSCALL(x4) | driver sync/collect primitive (region 1d) |
| `1d` | `lower.tac.05ea77` | `0x00007f091d05ea77` | 1---- | 4 | lower.tac.05ee54(x4) |  |
| `1d` | `lower.tac.bbb7a6` | `0x00007f091dbbb7a6` | 1---- | 4 | lower.tac.9e48ba(x4) |  |
| `1d` | `lower.tac.05ece6` | `0x00007f091d05ece6` | 1---- | 3 | lower.tac.066253(x3) |  |
| `1d` | `lower.tac.197d53` | `0x00007f091d197d53` | 1---- | 3 | lower.tac.05ee54(x3) |  |
| `1d` | `lower.tac.508e1f` | `0x00007f091d508e1f` | 1---- | 3 | lower.tac.5d03ea(x2), lower.tac.5d1cd4(x1) |  |
| `1d` | `lower.tac.5136bb` | `0x00007f091d5136bb` | 1---- | 3 | lower.tac.560e64(x3) |  |
| `1d` | `lower.tac.542d99` | `0x00007f091d542d99` | 1---- | 3 | lower.tac.53ee41(x3) |  |
| `1d` | `lower.tac.55e55d` | `0x00007f091d55e55d` | 1---- | 3 | lower.tac.5c97d6(x3) |  |
| `1d` | `lower.tac.55f2b4` | `0x00007f091d55f2b4` | 1---- | 3 | lower.tac.5c6ac2(x2), lower.tac.5c6a52(x1) |  |
| `1d` | `lower.tac.55fcfb` | `0x00007f091d55fcfb` | 1---- | 3 | lower.tac.561251(x2), lower.tac.560fdd(x1) |  |
| `1d` | `lower.tac.560e64` | `0x00007f091d560e64` | 1---- | 3 | lower.tac.55fcfb(x3) |  |
| `1d` | `lower.tac.5c6ac2` | `0x00007f091d5c6ac2` | 1--4- | 3 | lower.tac.a6bd61(x2), lower.tac.a6b898(x1) |  |
| `1d` | `lower.tac.5c7cd7` | `0x00007f091d5c7cd7` | ---4- | 3 | lower.tac.a6cc74(x1), lower.tac.a6cc25(x1), lower.tac.a6ca29(x1) |  |
| `1d` | `lower.tac.5c97d6` | `0x00007f091d5c97d6` | 1---- | 3 | lower.tac.605441(x3) |  |
| `1d` | `lower.tac.605441` | `0x00007f091d605441` | 1---- | 3 | lower.tac.508e1f(x3) |  |
| `1d` | `lower.tac.9e2534` | `0x00007f091d9e2534` | 1--4- | 3 | lower.tac.9e6fbc(x3) |  |
| `1d` | `lower.tac.a7b072` | `0x00007f091da7b072` | 1--4- | 3 | lower.tac.a99aff(x2), lower.tac.a9969a(x1) |  |
| `1d` | `lower.tac.a8be27` | `0x00007f091da8be27` | 1---- | 3 | lower.tac.aa133c(x2), lower.tac.aa0fc5(x1) |  |
| `1d` | `lower.tac.a905dd` | `0x00007f091da905dd` | 1---- | 3 | lower.tac.a8fae6(x1), lower.tac.a8fdb5(x1), lower.tac.a8fd12(x1) |  |
| `1d` | `lower.tac.a9132d` | `0x00007f091da9132d` | 1---- | 3 | lower.tac.a8be27(x2), lower.tac.a8bd42(x1) |  |
| `1d` | `lower.tac.a916c3` | `0x00007f091da916c3` | 1---- | 3 | lower.tac.a99579(x1), lower.tac.a9954a(x1), lower.tac.a99563(x1) |  |
| `1d` | `lower.tac.a92cf2` | `0x00007f091da92cf2` | 1---- | 3 | lower.tac.ac27e8(x2), lower.tac.ac2808(x1) |  |
| `1d` | `lower.tac.a9609a` | `0x00007f091da9609a` | 1--4- | 3 | lower.tac.82f17a(x2), lower.tac.82f0c4(x1) |  |
| `1d` | `lower.tac.a99aff` | `0x00007f091da99aff` | 1---- | 3 | lower.tac.aac61f(x2), lower.tac.aac5e0(x1) |  |
| `1d` | `lower.tac.aad154` | `0x00007f091daad154` | 1--4- | 3 | __GI___libc_free(x3) |  |
| `1d` | `lower.tac.adb45c` | `0x00007f091dadb45c` | 1---- | 3 | lower.tac.ab0112(x1), lower.tac.ab0028(x1), lower.tac.ab038c(x1) |  |
| `1d` | `lower.tac.bbb839` | `0x00007f091dbbb839` | 1--4- | 3 | lower.tac.a9c157(x2), lower.tac.a9c035(x1) |  |
| `1d` | `lower.tac.068c6e` | `0x00007f091d068c6e` | 1---- | 2 | lower.tac.066253(x2) |  |
| `1d` | `lower.tac.13c098` | `0x00007f091d13c098` | 1---- | 2 | lower.tac.38f263(x2) |  |
| `1d` | `lower.tac.188a51` | `0x00007f091d188a51` | 1--4- | 2 | lower.tac.33519b(x1), lower.tac.3351fb(x1) |  |
| `1d` | `lower.tac.1abc84` | `0x00007f091d1abc84` | 1---- | 2 | lower.tac.1b5d80(x1), lower.tac.1b5df4(x1) |  |
| `1d` | `lower.tac.38f263` | `0x00007f091d38f263` | 1---- | 2 | lower.cg.ee36be(x2) |  |
| `1d` | `lower.tac.4dbfba` | `0x00007f091d4dbfba` | 1---- | 2 | lower.tac.4de6ea(x2) |  |
| `1d` | `lower.tac.4de6ea` | `0x00007f091d4de6ea` | 1---- | 2 | lower.tac.4e244d(x2) |  |
| `1d` | `lower.tac.4e244d` | `0x00007f091d4e244d` | 1---- | 2 | lower.tac.5f9b74(x1), lower.tac.5f99ec(x1) |  |
| `1d` | `lower.tac.50d595` | `0x00007f091d50d595` | 1---- | 2 | __GI___libc_malloc(x2) |  |
| `1d` | `lower.tac.5137b3` | `0x00007f091d5137b3` | 1---- | 2 | lower.tac.58fc74(x2) |  |
| `1d` | `lower.tac.5203f7` | `0x00007f091d5203f7` | 1---- | 2 | lower.tac.5ef000(x1), lower.tac.5eefb2(x1) |  |
| `1d` | `lower.tac.52232f` | `0x00007f091d52232f` | 1---- | 2 | lower.tac.5203f7(x2) |  |
| `1d` | `lower.tac.542d52` | `0x00007f091d542d52` | 1---- | 2 | lower.tac.53ee41(x2) |  |
| `1d` | `lower.tac.54c683` | `0x00007f091d54c683` | 1---- | 2 | lower.tac.54c8e0(x2) |  |
| `1d` | `lower.tac.54c8e0` | `0x00007f091d54c8e0` | 1---- | 2 | lower.tac.542d99(x1), lower.tac.542ac3(x1) |  |
| `1d` | `lower.tac.54d41f` | `0x00007f091d54d41f` | 1---- | 2 | lower.tac.adb45c(x2) |  |
| `1d` | `lower.tac.54d622` | `0x00007f091d54d622` | 1---- | 2 | lower.tac.a9fa8f(x2) |  |
| `1d` | `lower.tac.558d06` | `0x00007f091d558d06` | 1---- | 2 | lower.tac.50d595(x2) |  |
| `1d` | `lower.tac.561251` | `0x00007f091d561251` | 1---- | 2 | lower.tac.542d52(x1), lower.tac.542936(x1) |  |
| `1d` | `lower.tac.58535f` | `0x00007f091d58535f` | 1--4- | 2 | lower.tac.587798(x2) |  |
| `1d` | `lower.tac.5855f7` | `0x00007f091d5855f7` | 1---- | 2 | lower.tac.542d99(x1), lower.tac.542d52(x1) |  |
| `1d` | `lower.tac.587798` | `0x00007f091d587798` | 1--4- | 2 | lower.tac.585169(x1), lower.tac.585215(x1) |  |
| `1d` | `lower.tac.58fc74` | `0x00007f091d58fc74` | 1---- | 2 | lower.tac.58ee61(x1), lower.tac.58ed0e(x1) |  |
| `1d` | `lower.tac.5b7ca8` | `0x00007f091d5b7ca8` | 1---- | 2 | lower.tac.563097(x1), lower.tac.563622(x1) |  |
| `1d` | `lower.tac.5c6a52` | `0x00007f091d5c6a52` | 1--4- | 2 | lower.tac.a6ca29(x1), lower.tac.a6c768(x1) |  |
| `1d` | `lower.tac.5c7d2a` | `0x00007f091d5c7d2a` | ---4- | 2 | lower.tac.a6bf4a(x1), lower.tac.a6b898(x1) |  |
| `1d` | `lower.tac.5c841f` | `0x00007f091d5c841f` | ---4- | 2 | lower.tac.5c6a52(x1), lower.tac.5c6ac2(x1) |  |
| `1d` | `lower.tac.5d03ea` | `0x00007f091d5d03ea` | 1---- | 2 | lower.tac.ae443c(x1), lower.tac.ae43e2(x1) |  |
| `1d` | `lower.tac.5ef65c` | `0x00007f091d5ef65c` | 1---- | 2 | lower.tac.558d06(x2) |  |
| `1d` | `lower.tac.82f17a` | `0x00007f091d82f17a` | 1---- | 2 | lower.tactic.leaf(x1), lower.tac.a50e4b(x1) |  |
| `1d` | `lower.tac.858792` | `0x00007f091d858792` | 1---- | 2 | lower.tac.8f78fe(x1), lower.tac.8f79c6(x1) |  |
| `1d` | `lower.tac.860157` | `0x00007f091d860157` | ---4- | 2 | lower.tac.84d972(x1), lower.tac.84d78d(x1) |  |
| `1d` | `lower.tac.8fb698` | `0x00007f091d8fb698` | 1---- | 2 | lower.tac.a916c3(x2) |  |
| `1d` | `lower.tac.9cfaf5` | `0x00007f091d9cfaf5` | 1---- | 2 | lower.cg.db09b0(x1), lower.mid.019f94(x1) |  |
| `1d` | `lower.tac.a45f57` | `0x00007f091da45f57` | 1---- | 2 | lower.tac.82cb63(x1), lower.tac.82ca87(x1) |  |
| `1d` | `lower.tac.a6b898` | `0x00007f091da6b898` | ---4- | 2 | lower.tac.a6cf8e(x1), lower.tac.a6d0a2(x1) |  |
| `1d` | `lower.tac.a6bd61` | `0x00007f091da6bd61` | 1---- | 2 | lower.tac.a6ce19(x2) |  |
| `1d` | `lower.tac.a6ca29` | `0x00007f091da6ca29` | 1--4- | 2 | lower.tac.a7e91a(x1), lower.tac.a7e2f1(x1) |  |
| `1d` | `lower.tac.a6ce19` | `0x00007f091da6ce19` | 1---- | 2 | lower.tac.858792(x2) |  |
| `1d` | `lower.tac.a80a28` | `0x00007f091da80a28` | 1--4- | 2 | lower.tac.a81858(x1), lower.tac.a817d5(x1) |  |
| `1d` | `lower.tac.a81858` | `0x00007f091da81858` | 1---- | 2 | lower.tac.a88157(x1), lower.tac.a8870c(x1) |  |
| `1d` | `lower.tac.a8fd12` | `0x00007f091da8fd12` | 1---- | 2 | lower.tac.a82272(x2) |  |
| `1d` | `lower.tac.a9136b` | `0x00007f091da9136b` | 1---- | 2 | lower.tac.a8bbae(x1), lower.tac.a8be27(x1) |  |
| `1d` | `lower.tac.a920dd` | `0x00007f091da920dd` | 1---- | 2 | lower.tac.aac070(x1), lower.tac.aac0ba(x1) |  |
| `1d` | `lower.tac.a9954a` | `0x00007f091da9954a` | 1--4- | 2 | lower.tac.a91d82(x2) |  |
| `1d` | `lower.tac.a99563` | `0x00007f091da99563` | 1---- | 2 | lower.tac.a920dd(x2) |  |
| `1d` | `lower.tac.a995fc` | `0x00007f091da995fc` | 1--4- | 2 | lower.tac.a7b072(x2) |  |
| `1d` | `lower.tac.a9fa8f` | `0x00007f091da9fa8f` | 1---- | 2 | lower.tac.aa2bad(x2) |  |
| `1d` | `lower.tac.aa133c` | `0x00007f091daa133c` | 1---- | 2 | lower.tac.aa797b(x1), lower.tac.aa7990(x1) |  |
| `1d` | `lower.tac.aa2bad` | `0x00007f091daa2bad` | 1---- | 2 | lower.tac.a82272(x2) |  |
| `1d` | `lower.tac.aa797b` | `0x00007f091daa797b` | 1---- | 2 | lower.tac.a92cf2(x2) |  |
| `1d` | `lower.tac.aac61f` | `0x00007f091daac61f` | 1---- | 2 | lower.tac.aa9905(x1), lower.tac.aa9adf(x1) |  |
| `1d` | `lower.tac.aacc41` | `0x00007f091daacc41` | 1---- | 2 | __GI___libc_malloc(x2) |  |
| `1d` | `lower.tac.aafcab` | `0x00007f091daafcab` | 1--4- | 2 | lower.tac.b748f5(x2) |  |
| `1d` | `lower.tac.ab0112` | `0x00007f091dab0112` | 1---- | 2 | lower.tac.a45f57(x2) |  |
| `1d` | `lower.tac.ab0a09` | `0x00007f091dab0a09` | 1---- | 2 | lower.tac.aaf5ac(x1), lower.tac.aafcab(x1) |  |
| `1d` | `lower.tac.b748f5` | `0x00007f091db748f5` | 1--4- | 2 | lower.tac.ab4375(x1), lower.tac.ab4366(x1) |  |
| `1d` | `lower.tac.b9f0fd` | `0x00007f091db9f0fd` | ---4- | 2 | lower.tac.9cefc9(x1), lower.tac.9ceec8(x1) |  |
| `1d` | `lower.tac.f08f0e` | `0x00007f091df08f0e` | ---4- | 2 | lower.tac.f0fd97(x2) |  |
| `1d` | `lower.tac.f0fd97` | `0x00007f091df0fd97` | ---4- | 2 | lower.tac.f092e3(x1), lower.mid.277ec1(x1) |  |
| `1d` | `lower.tac.065159` | `0x00007f091d065159` | 1---- | 1 | lower.tac.19718e(x1) |  |
| `1d` | `lower.tac.13a9ca` | `0x00007f091d13a9ca` | ---4- | 1 | lower.tac.18c6c5(x1) |  |
| `1d` | `lower.tac.14ac44` | `0x00007f091d14ac44` | 1---- | 1 | lower.tac.3992f5(x1) |  |
| `1d` | `lower.tac.188a7c` | `0x00007f091d188a7c` | 1---- | 1 | lower.cg.cb2328(x1) |  |
| `1d` | `lower.tac.18c6c5` | `0x00007f091d18c6c5` | ---4- | 1 | lower.tac.484be6(x1) |  |
| `1d` | `lower.tac.18ecea` | `0x00007f091d18ecea` | 1---- | 1 | lower.tac.456129(x1) |  |
| `1d` | `lower.tac.1941e8` | `0x00007f091d1941e8` | 1---- | 1 | lower.tac.54d41f(x1) |  |
| `1d` | `lower.tac.195f7b` | `0x00007f091d195f7b` | 1---- | 1 | lower.tac.196bc8(x1) |  |
| `1d` | `lower.tac.196bc8` | `0x00007f091d196bc8` | 1---- | 1 | lower.tac.19a588(x1) |  |
| `1d` | `lower.tac.19718e` | `0x00007f091d19718e` | 1---- | 1 | lower.tac.18ecea(x1) |  |
| `1d` | `lower.tac.198cad` | `0x00007f091d198cad` | 1---- | 1 | lower.tac.1941e8(x1) |  |
| `1d` | `lower.tac.19a588` | `0x00007f091d19a588` | 1---- | 1 | lower.tac.198cad(x1) |  |
| `1d` | `lower.tac.19ecbb` | `0x00007f091d19ecbb` | ---4- | 1 | lower.tac.54b139(x1) |  |
| `1d` | `lower.tac.1a0438` | `0x00007f091d1a0438` | 1---- | 1 | lower.tac.51aef5(x1) |  |
| `1d` | `lower.tac.1a0956` | `0x00007f091d1a0956` | 1---- | 1 | lower.cg.9b4b86(x1) |  |
| `1d` | `lower.tac.1b198b` | `0x00007f091d1b198b` | 1---- | 1 | lower.tac.3a94da(x1) |  |
| `1d` | `lower.tac.1b39da` | `0x00007f091d1b39da` | 1---- | 1 | lower.tac.189203(x1) |  |
| `1d` | `lower.tac.1b5d80` | `0x00007f091d1b5d80` | 1---- | 1 | lower.tac.1bdcc5(x1) |  |
| `1d` | `lower.tac.1b5df4` | `0x00007f091d1b5df4` | 1---- | 1 | lower.tac.46bc94(x1) |  |
| `1d` | `lower.tac.1c8fa9` | `0x00007f091d1c8fa9` | 1---- | 1 | lower.tac.1cdff5(x1) |  |
| `1d` | `lower.tac.1ca275` | `0x00007f091d1ca275` | 1---- | 1 | lower.cg.a581d2(x1) |  |
| `1d` | `lower.tac.1caad1` | `0x00007f091d1caad1` | 1---- | 1 | __GI___libc_malloc(x1) |  |
| `1d` | `lower.tac.1cdff5` | `0x00007f091d1cdff5` | 1---- | 1 | lower.tac.1ca275(x1) |  |
| `1d` | `lower.tac.1f0159` | `0x00007f091d1f0159` | 1---- | 1 | lower.tac.14ac44(x1) |  |
| `1d` | `lower.tac.1f119c` | `0x00007f091d1f119c` | 1---- | 1 | lower.tac.1c8fa9(x1) |  |
| `1d` | `lower.tac.335194` | `0x00007f091d335194` | 1---- | 1 | lower.tac.334abf(x1) |  |
| `1d` | `lower.tac.391f24` | `0x00007f091d391f24` | 1---- | 1 | lower.tac.54d41f(x1) |  |
| `1d` | `lower.tac.398b0d` | `0x00007f091d398b0d` | 1---- | 1 | lower.cg.ee34ea(x1) |  |
| `1d` | `lower.tac.3992f5` | `0x00007f091d3992f5` | 1---- | 1 | lower.tac.359b4e(x1) |  |
| `1d` | `lower.tac.3a94da` | `0x00007f091d3a94da` | 1---- | 1 | lower.tac.34ae40(x1) |  |
| `1d` | `lower.tac.4105e0` | `0x00007f091d4105e0` | 1---- | 1 | lower.tac.1b39da(x1) |  |
| `1d` | `lower.tac.456129` | `0x00007f091d456129` | 1---- | 1 | __memcpy_avx512_unaligned_erms(x1) |  |
| `1d` | `lower.tac.46bc94` | `0x00007f091d46bc94` | 1---- | 1 | lower.tac.335194(x1) |  |
| `1d` | `lower.tac.484be6` | `0x00007f091d484be6` | ---4- | 1 | __GI___libc_free(x1) |  |
| `1d` | `lower.tac.4da51c` | `0x00007f091d4da51c` | ---4- | 1 | lower.tac.5c4e8c(x1) |  |
| `1d` | `lower.tac.4daeef` | `0x00007f091d4daeef` | ---4- | 1 | lower.tac.4da51c(x1) |  |
| `1d` | `lower.tac.4db6a5` | `0x00007f091d4db6a5` | ---4- | 1 | lower.tac.4dd7ec(x1) |  |
| `1d` | `lower.tac.4db9ea` | `0x00007f091d4db9ea` | ---4- | 1 | lower.tac.4daeef(x1) |  |
| `1d` | `lower.tac.4dcfd0` | `0x00007f091d4dcfd0` | ---4- | 1 | lower.tac.99c0ac(x1) |  |
| `1d` | `lower.tac.4dd7ec` | `0x00007f091d4dd7ec` | ---4- | 1 | lower.tac.5c4a5d(x1) |  |
| `1d` | `lower.tac.4e5cdc` | `0x00007f091d4e5cdc` | 1---- | 1 | lower.tac.4dbfba(x1) |  |
| `1d` | `lower.tac.503d18` | `0x00007f091d503d18` | 1---- | 1 | lower.tac.507be2(x1) |  |
| `1d` | `lower.tac.50d092` | `0x00007f091d50d092` | ---4- | 1 | lower.tac.507be2(x1) |  |
| `1d` | `lower.tac.50d5d7` | `0x00007f091d50d5d7` | 1---- | 1 | lower.tac.503d18(x1) |  |
| `1d` | `lower.tac.510994` | `0x00007f091d510994` | 1---- | 1 | lower.tac.5e539e(x1) |  |
| `1d` | `lower.tac.512afa` | `0x00007f091d512afa` | 1---- | 1 | lower.tac.aa5c54(x1) |  |
| `1d` | `lower.tac.51aef5` | `0x00007f091d51aef5` | 1---- | 1 | lower.tac.a9e8a3(x1) |  |
| `1d` | `lower.tac.51af4c` | `0x00007f091d51af4c` | 1---- | 1 | lower.tac.543083(x1) |  |
| `1d` | `lower.tac.51ddda` | `0x00007f091d51ddda` | 1---- | 1 | lower.tac.51fdcd(x1) |  |
| `1d` | `lower.tac.51fdcd` | `0x00007f091d51fdcd` | 1---- | 1 | lower.tac.aa5b1c(x1) |  |
| `1d` | `lower.tac.54215b` | `0x00007f091d54215b` | 1---- | 1 | lower.tac.a995fc(x1) |  |
| `1d` | `lower.tac.542936` | `0x00007f091d542936` | 1---- | 1 | lower.tac.a9fc03(x1) |  |
| `1d` | `lower.tac.542ac3` | `0x00007f091d542ac3` | 1---- | 1 | lower.tac.557eed(x1) |  |
| `1d` | `lower.tac.543083` | `0x00007f091d543083` | 1---- | 1 | lower.tac.aa2f1a(x1) |  |
| `1d` | `lower.tac.54317e` | `0x00007f091d54317e` | 1---- | 1 | lower.tac.a9609a(x1) |  |
| `1d` | `lower.tac.54b139` | `0x00007f091d54b139` | ---4- | 1 | lower.tac.4dcfd0(x1) |  |
| `1d` | `lower.tac.54b2e4` | `0x00007f091d54b2e4` | 1---- | 1 | lower.tac.51ddda(x1) |  |
| `1d` | `lower.tac.557eed` | `0x00007f091d557eed` | 1---- | 1 | lower.tac.50d5d7(x1) |  |
| `1d` | `lower.tac.55c662` | `0x00007f091d55c662` | 1---- | 1 | lower.tac.4f6377(x1) |  |
| `1d` | `lower.tac.55e14f` | `0x00007f091d55e14f` | 1---- | 1 | lower.tac.a8e99d(x1) |  |
| `1d` | `lower.tac.55e417` | `0x00007f091d55e417` | 1---- | 1 | lower.tac.55c662(x1) |  |
| `1d` | `lower.tac.55f1ef` | `0x00007f091d55f1ef` | 1---- | 1 | lower.tac.52232f(x1) |  |
| `1d` | `lower.tac.560fdd` | `0x00007f091d560fdd` | 1---- | 1 | lower.tac.54215b(x1) |  |
| `1d` | `lower.tac.563097` | `0x00007f091d563097` | 1---- | 1 | lower.tac.54317e(x1) |  |
| `1d` | `lower.tac.563622` | `0x00007f091d563622` | 1---- | 1 | lower.tac.542d99(x1) |  |
| `1d` | `lower.tac.584a6a` | `0x00007f091d584a6a` | 1---- | 1 | __memcpy_avx512_unaligned_erms(x1) |  |
| `1d` | `lower.tac.585169` | `0x00007f091d585169` | 1---- | 1 | lower.tac.58bc4a(x1) |  |
| `1d` | `lower.tac.585215` | `0x00007f091d585215` | ---4- | 1 | lower.tac.ab08ed(x1) |  |
| `1d` | `lower.tac.588880` | `0x00007f091d588880` | 1---- | 1 | lower.tac.58535f(x1) |  |
| `1d` | `lower.tac.58bc4a` | `0x00007f091d58bc4a` | 1---- | 1 | lower.tac.5ef65c(x1) |  |
| `1d` | `lower.tac.58ed0e` | `0x00007f091d58ed0e` | 1---- | 1 | lower.tac.ab0a09(x1) |  |
| `1d` | `lower.tac.58ee61` | `0x00007f091d58ee61` | 1---- | 1 | lower.tac.58fb59(x1) |  |
| `1d` | `lower.tac.58f2e5` | `0x00007f091d58f2e5` | 1---- | 1 | lower.tac.aae1b1(x1) |  |
| `1d` | `lower.tac.58fb59` | `0x00007f091d58fb59` | 1---- | 1 | lower.tac.58f2e5(x1) |  |
| `1d` | `lower.tac.5c4a5d` | `0x00007f091d5c4a5d` | ---4- | 1 | lower.tac.50d092(x1) |  |
| `1d` | `lower.tac.5d1cd4` | `0x00007f091d5d1cd4` | 1---- | 1 | lower.tac.aefe0d(x1) |  |
| `1d` | `lower.tac.5d95dd` | `0x00007f091d5d95dd` | 1---- | 1 | lower.tac.5e6f42(x1) |  |
| `1d` | `lower.tac.5db598` | `0x00007f091d5db598` | 1---- | 1 | lower.tac.5dbb21(x1) |  |
| `1d` | `lower.tac.5dbb21` | `0x00007f091d5dbb21` | 1---- | 1 | lower.tac.5e1afa(x1) |  |
| `1d` | `lower.tac.5e1afa` | `0x00007f091d5e1afa` | 1---- | 1 | lower.tac.5ef65c(x1) |  |
| `1d` | `lower.tac.5e539e` | `0x00007f091d5e539e` | 1---- | 1 | lower.tac.ab0a09(x1) |  |
| `1d` | `lower.tac.5eace3` | `0x00007f091d5eace3` | 1---- | 1 | lower.tac.5d95dd(x1) |  |
| `1d` | `lower.tac.5eefb2` | `0x00007f091d5eefb2` | 1---- | 1 | lower.tac.510994(x1) |  |
| `1d` | `lower.tac.5ef000` | `0x00007f091d5ef000` | 1---- | 1 | lower.tac.5eace3(x1) |  |
| `1d` | `lower.tac.5f99ec` | `0x00007f091d5f99ec` | 1---- | 1 | lower.tac.c0f920(x1) |  |
| `1d` | `lower.tac.5f9b74` | `0x00007f091d5f9b74` | 1---- | 1 | lower.tac.584a6a(x1) |  |
| `1d` | `lower.tac.82553a` | `0x00007f091d82553a` | --3-- | 1 | lower.tac.8f3b14(x1) |  |
| `1d` | `lower.tac.82ca87` | `0x00007f091d82ca87` | 1---- | 1 | lower.tac.8f7fb6(x1) |  |
| `1d` | `lower.tac.82cb63` | `0x00007f091d82cb63` | 1---- | 1 | lower.tac.9e2534(x1) |  |
| `1d` | `lower.tac.839feb` | `0x00007f091d839feb` | 1---- | 1 | lower.tac.a88a4b(x1) |  |
| `1d` | `lower.tac.83b1c7` | `0x00007f091d83b1c7` | ---4- | 1 | lower.tac.9e2534(x1) |  |
| `1d` | `lower.tac.84d78d` | `0x00007f091d84d78d` | ---4- | 1 | lower.tac.860157(x1) |  |
| `1d` | `lower.tac.84d972` | `0x00007f091d84d972` | ---4- | 1 | lower.tac.8602a6(x1) |  |
| `1d` | `lower.tac.8578f0` | `0x00007f091d8578f0` | 1---- | 1 | __GI___libc_malloc(x1) |  |
| `1d` | `lower.tac.8602a6` | `0x00007f091d8602a6` | ---4- | 1 | __GI___libc_malloc(x1) |  |
| `1d` | `lower.tac.8f79c6` | `0x00007f091d8f79c6` | 1---- | 1 | lower.tac.a7b072(x1) |  |
| `1d` | `lower.tac.8f9957` | `0x00007f091d8f9957` | ---4- | 1 | lower.tac.acf448(x1) |  |
| `1d` | `lower.tac.99c0ac` | `0x00007f091d99c0ac` | ---4- | 1 | __memset_avx512_unaligned_erms(x1) |  |
| `1d` | `lower.tac.9cee48` | `0x00007f091d9cee48` | --3-- | 1 | lower.tac.b63236(x1) |  |
| `1d` | `lower.tac.9ceec8` | `0x00007f091d9ceec8` | ---4- | 1 | lower.tac.8f9957(x1) |  |
| `1d` | `lower.tac.9cefc9` | `0x00007f091d9cefc9` | ---4- | 1 | lower.tac.bafef5(x1) |  |
| `1d` | `lower.tac.9cf956` | `0x00007f091d9cf956` | --3-- | 1 | lower.tac.a44fd3(x1) |  |
| `1d` | `lower.tac.9d4cc0` | `0x00007f091d9d4cc0` | 1---- | 1 | lower.tac.b63e8a(x1) |  |
| `1d` | `lower.tac.9e453a` | `0x00007f091d9e453a` | ----5 | 1 | lower.tac.9e6fbc(x1) |  |
| `1d` | `lower.tac.9e492a` | `0x00007f091d9e492a` | 1---- | 1 | lower.tac.9e6fbc(x1) |  |
| `1d` | `lower.tac.9f2dcd` | `0x00007f091d9f2dcd` | ---4- | 1 | lower.tac.f08f0e(x1) |  |
| `1d` | `lower.tac.a44fd3` | `0x00007f091da44fd3` | --3-- | 1 | lower.tac.82553a(x1) |  |
| `1d` | `lower.tac.a50e4b` | `0x00007f091da50e4b` | 1---- | 1 | lower.tac.b697e9(x1) |  |
| `1d` | `lower.tac.a6bf4a` | `0x00007f091da6bf4a` | ---4- | 1 | lower.tac.a7046e(x1) |  |
| `1d` | `lower.tac.a6c768` | `0x00007f091da6c768` | ---4- | 1 | lower.tac.a80490(x1) |  |
| `1d` | `lower.tac.a6cc25` | `0x00007f091da6cc25` | ---4- | 1 | lower.tac.a7d62c(x1) |  |
| `1d` | `lower.tac.a6cc74` | `0x00007f091da6cc74` | ---4- | 1 | lower.tac.a82b5b(x1) |  |
| `1d` | `lower.tac.a6cf8e` | `0x00007f091da6cf8e` | ---4- | 1 | lower.tac.860157(x1) |  |
| `1d` | `lower.tac.a6d0a2` | `0x00007f091da6d0a2` | ---4- | 1 | lower.tac.a80a28(x1) |  |
| `1d` | `lower.tac.a6ed32` | `0x00007f091da6ed32` | 1---- | 1 | __GI___libc_free(x1) |  |
| `1d` | `lower.tac.a7046e` | `0x00007f091da7046e` | ---4- | 1 | lower.tac.a9bbd8(x1) |  |
| `1d` | `lower.tac.a7b492` | `0x00007f091da7b492` | 1---- | 1 | lower.tac.a99aff(x1) |  |
| `1d` | `lower.tac.a7d62c` | `0x00007f091da7d62c` | ---4- | 1 | lower.tac.a9954a(x1) |  |
| `1d` | `lower.tac.a7e2f1` | `0x00007f091da7e2f1` | ---4- | 1 | lower.tac.a995fc(x1) |  |
| `1d` | `lower.tac.a7e91a` | `0x00007f091da7e91a` | 1---- | 1 | lower.tac.a817d5(x1) |  |
| `1d` | `lower.tac.a7eabb` | `0x00007f091da7eabb` | 1---- | 1 | lower.tac.a813bf(x1) |  |
| `1d` | `lower.tac.a80490` | `0x00007f091da80490` | ---4- | 1 | lower.tac.a9609a(x1) |  |
| `1d` | `lower.tac.a813bf` | `0x00007f091da813bf` | 1---- | 1 | lower.tac.a8276b(x1) |  |
| `1d` | `lower.tac.a81d39` | `0x00007f091da81d39` | ---4- | 1 | lower.tac.a7bebc(x1) |  |
| `1d` | `lower.tac.a8276b` | `0x00007f091da8276b` | 1---- | 1 | lower.tac.a93a60(x1) |  |
| `1d` | `lower.tac.a82b5b` | `0x00007f091da82b5b` | ---4- | 1 | lower.tac.a8389b(x1) |  |
| `1d` | `lower.tac.a82ed2` | `0x00007f091da82ed2` | 1---- | 1 | lower.tac.a82272(x1) |  |
| `1d` | `lower.tac.a8389b` | `0x00007f091da8389b` | ---4- | 1 | lower.tac.a81d39(x1) |  |
| `1d` | `lower.tac.a8870c` | `0x00007f091da8870c` | 1---- | 1 | lower.tac.9e2534(x1) |  |
| `1d` | `lower.tac.a8bbae` | `0x00007f091da8bbae` | 1---- | 1 | lower.tac.839feb(x1) |  |
| `1d` | `lower.tac.a8bd42` | `0x00007f091da8bd42` | 1---- | 1 | lower.tac.aa797b(x1) |  |
| `1d` | `lower.tac.a8d156` | `0x00007f091da8d156` | ---4- | 1 | lower.tac.aac081(x1) |  |
| `1d` | `lower.tac.a8eae6` | `0x00007f091da8eae6` | 1---- | 1 | lower.tac.a82272(x1) |  |
| `1d` | `lower.tac.a8f5ad` | `0x00007f091da8f5ad` | 1---- | 1 | lower.tac.a80a28(x1) |  |
| `1d` | `lower.tac.a8fae6` | `0x00007f091da8fae6` | 1---- | 1 | lower.tac.a92cf2(x1) |  |
| `1d` | `lower.tac.a8fdb5` | `0x00007f091da8fdb5` | 1---- | 1 | __GI___libc_free(x1) |  |
| `1d` | `lower.tac.a903bf` | `0x00007f091da903bf` | 1---- | 1 | lower.tac.a8f5ad(x1) |  |
| `1d` | `lower.tac.a905fd` | `0x00007f091da905fd` | 1---- | 1 | lower.tac.a8fd12(x1) |  |
| `1d` | `lower.tac.a90a0c` | `0x00007f091da90a0c` | 1---- | 1 | lower.tac.a8eae6(x1) |  |
| `1d` | `lower.tac.a91fbf` | `0x00007f091da91fbf` | 1---- | 1 | lower.tac.a91d80(x1) |  |
| `1d` | `lower.tac.a92a47` | `0x00007f091da92a47` | 1---- | 1 | lower.tac.a9609a(x1) |  |
| `1d` | `lower.tac.a99539` | `0x00007f091da99539` | ---4- | 1 | lower.tac.a8d156(x1) |  |
| `1d` | `lower.tac.a995e5` | `0x00007f091da995e5` | 1---- | 1 | lower.tac.a7b492(x1) |  |
| `1d` | `lower.tac.a9969a` | `0x00007f091da9969a` | ---4- | 1 | lower.tac.a99539(x1) |  |
| `1d` | `lower.tac.a9bbd8` | `0x00007f091da9bbd8` | ---4- | 1 | lower.tac.a9949c(x1) |  |
| `1d` | `lower.tac.a9c035` | `0x00007f091da9c035` | 1---- | 1 | lower.tac.a97310(x1) |  |
| `1d` | `lower.tac.a9cb8d` | `0x00007f091da9cb8d` | 1---- | 1 | lower.tac.a92a47(x1) |  |
| `1d` | `lower.tac.a9e8a3` | `0x00007f091da9e8a3` | 1---- | 1 | lower.tac.aac8af(x1) |  |
| `1d` | `lower.tac.aa0fc5` | `0x00007f091daa0fc5` | 1---- | 1 | lower.tac.9e48ba(x1) |  |
| `1d` | `lower.tac.aa5b1c` | `0x00007f091daa5b1c` | 1---- | 1 | lower.tac.eba608(x1) |  |
| `1d` | `lower.tac.aa5c54` | `0x00007f091daa5c54` | 1---- | 1 | lower.tac.ac58eb(x1) |  |
| `1d` | `lower.tac.aa7990` | `0x00007f091daa7990` | 1---- | 1 | lower.tac.ac27fe(x1) |  |
| `1d` | `lower.tac.aa9905` | `0x00007f091daa9905` | 1---- | 1 | lower.tac.a99563(x1) |  |
| `1d` | `lower.tac.aa9adf` | `0x00007f091daa9adf` | 1---- | 1 | lower.tac.8578f0(x1) |  |
| `1d` | `lower.tac.aac0ba` | `0x00007f091daac0ba` | 1---- | 1 | lower.tac.a91fbf(x1) |  |
| `1d` | `lower.tac.aac5e0` | `0x00007f091daac5e0` | 1---- | 1 | lower.tac.857aa8(x1) |  |
| `1d` | `lower.tac.aac9a7` | `0x00007f091daac9a7` | 1---- | 1 | lower.tac.a916c3(x1) |  |
| `1d` | `lower.tac.aae1b1` | `0x00007f091daae1b1` | 1---- | 1 | lower.tac.ab0112(x1) |  |
| `1d` | `lower.tac.aaedd1` | `0x00007f091daaedd1` | 1---- | 1 | lower.tac.b7878f(x1) |  |
| `1d` | `lower.tac.aaf5ac` | `0x00007f091daaf5ac` | 1---- | 1 | lower.tac.aaedd1(x1) |  |
| `1d` | `lower.tac.ab0028` | `0x00007f091dab0028` | 1---- | 1 | lower.tac.ab0c27(x1) |  |
| `1d` | `lower.tac.ab038c` | `0x00007f091dab038c` | 1---- | 1 | lower.tac.9d4cc0(x1) |  |
| `1d` | `lower.tac.ab08ed` | `0x00007f091dab08ed` | ---4- | 1 | lower.tac.aafcab(x1) |  |
| `1d` | `lower.tac.ab346f` | `0x00007f091dab346f` | 1---- | 1 | lower.tac.a82ed2(x1) |  |
| `1d` | `lower.tac.ac204f` | `0x00007f091dac204f` | 1---- | 1 | lower.tac.ab346f(x1) |  |
| `1d` | `lower.tac.ac58eb` | `0x00007f091dac58eb` | 1---- | 1 | __memcpy_avx512_unaligned_erms(x1) |  |
| `1d` | `lower.tac.ac6bfe` | `0x00007f091dac6bfe` | --3-- | 1 | lower.tac.acda3d(x1) |  |
| `1d` | `lower.tac.acda3d` | `0x00007f091dacda3d` | --3-- | 1 | __GI___libc_malloc(x1) |  |
| `1d` | `lower.tac.acf448` | `0x00007f091dacf448` | ---4- | 1 | lower.tac.b6a2ea(x1) |  |
| `1d` | `lower.tac.ada9a2` | `0x00007f091dada9a2` | 1---- | 1 | lower.tac.a7eabb(x1) |  |
| `1d` | `lower.tac.adac30` | `0x00007f091dadac30` | 1---- | 1 | lower.tac.af8307(x1) |  |
| `1d` | `lower.tac.ae43e2` | `0x00007f091dae43e2` | 1---- | 1 | lower.tac.ada9a2(x1) |  |
| `1d` | `lower.tac.ae443c` | `0x00007f091dae443c` | 1---- | 1 | lower.tac.adac30(x1) |  |
| `1d` | `lower.tac.aee452` | `0x00007f091daee452` | 1---- | 1 | lower.tac.81f9d1(x1) |  |
| `1d` | `lower.tac.aee621` | `0x00007f091daee621` | 1---- | 1 | lower.tac.aee452(x1) |  |
| `1d` | `lower.tac.aefe0d` | `0x00007f091daefe0d` | 1---- | 1 | lower.tac.adb45c(x1) |  |
| `1d` | `lower.tac.af8307` | `0x00007f091daf8307` | 1---- | 1 | lower.tac.af99e4(x1) |  |
| `1d` | `lower.tac.b63236` | `0x00007f091db63236` | --3-- | 1 | lower.tac.ac6bfe(x1) |  |
| `1d` | `lower.tac.b63e8a` | `0x00007f091db63e8a` | 1---- | 1 | lower.tac.8337bc(x1) |  |
| `1d` | `lower.tac.b697e9` | `0x00007f091db697e9` | 1---- | 1 | lower.tac.acb2ea(x1) |  |
| `1d` | `lower.tac.b6a2ea` | `0x00007f091db6a2ea` | ---4- | 1 | lower.tac.a668f5(x1) |  |
| `1d` | `lower.tac.b7878f` | `0x00007f091db7878f` | 1---- | 1 | lower.tac.9e492a(x1) |  |
| `1d` | `lower.tac.bafef5` | `0x00007f091dbafef5` | ---4- | 1 | lower.tac.83b1c7(x1) |  |
| `1d` | `lower.tac.c0f920` | `0x00007f091dc0f920` | 1---- | 1 | __GI___libc_free(x1) |  |
| `1d` | `lower.tac.eba608` | `0x00007f091deba608` | 1---- | 1 | lower.tac.f08f21(x1) |  |
| `1d` | `lower.tac.ed8d41` | `0x00007f091ded8d41` | -2--- | 1 | __pthread_clockjoin_ex(x1) |  |
| `1d` | `lower.tac.f07142` | `0x00007f091df07142` | ---4- | 1 | lower.tac.f075a8(x1) |  |
| `1d` | `lower.tac.f092c0` | `0x00007f091df092c0` | ---4- | 1 | __memcpy_avx512_unaligned_erms(x1) |  |
| `1d` | `lower.tac.f092e3` | `0x00007f091df092e3` | ---4- | 1 | lower.tac.f07142(x1) |  |
| `1d` | `lower.tac.f0bfbc` | `0x00007f091df0bfbc` | ---4- | 1 | lower.tac.f092c0(x1) |  |
| `1d` | `lower.tac.f0e86d` | `0x00007f091df0e86d` | ---4- | 1 | lower.tac.f0bfbc(x1) |  |
| `1d` | `lower.tac.189203` | `0x00007f091d189203` | 1---- | 0 | — |  |
| `1d` | `lower.tac.334abf` | `0x00007f091d334abf` | 1---- | 0 | — |  |
| `1d` | `lower.tac.33519b` | `0x00007f091d33519b` | 1---- | 0 | — |  |
| `1d` | `lower.tac.3351fb` | `0x00007f091d3351fb` | ---4- | 0 | — |  |
| `1d` | `lower.tac.34ae40` | `0x00007f091d34ae40` | 1---- | 0 | — |  |
| `1d` | `lower.tac.359b4e` | `0x00007f091d359b4e` | 1---- | 0 | — |  |
| `1d` | `lower.tac.4104e0` | `0x00007f091d4104e0` | 1---- | 0 | — |  |
| `1d` | `lower.tac.4f6377` | `0x00007f091d4f6377` | 1---- | 0 | — |  |
| `1d` | `lower.tac.507be2` | `0x00007f091d507be2` | 1--4- | 0 | — |  |
| `1d` | `lower.tac.5c4e8c` | `0x00007f091d5c4e8c` | ---4- | 0 | — |  |
| `1d` | `lower.tac.5e6f42` | `0x00007f091d5e6f42` | 1---- | 0 | — |  |
| `1d` | `lower.tac.81f9d1` | `0x00007f091d81f9d1` | 1---- | 0 | — |  |
| `1d` | `lower.tac.82f0c4` | `0x00007f091d82f0c4` | ---4- | 0 | — |  |
| `1d` | `lower.tac.8337bc` | `0x00007f091d8337bc` | 1---- | 0 | — |  |
| `1d` | `lower.tac.857aa8` | `0x00007f091d857aa8` | 1---- | 0 | — |  |
| `1d` | `lower.tac.8f3b14` | `0x00007f091d8f3b14` | --3-- | 0 | — |  |
| `1d` | `lower.tac.8f78fe` | `0x00007f091d8f78fe` | 1---- | 0 | — |  |
| `1d` | `lower.tac.8f7fb6` | `0x00007f091d8f7fb6` | 1---- | 0 | — |  |
| `1d` | `lower.tac.a668f5` | `0x00007f091da668f5` | ---4- | 0 | — |  |
| `1d` | `lower.tac.a88157` | `0x00007f091da88157` | 1---- | 0 | — |  |
| `1d` | `lower.tac.a88a4b` | `0x00007f091da88a4b` | 1---- | 0 | — |  |
| `1d` | `lower.tac.a91d80` | `0x00007f091da91d80` | 1---- | 0 | — |  |
| `1d` | `lower.tac.a91d82` | `0x00007f091da91d82` | 1--4- | 0 | — |  |
| `1d` | `lower.tac.a93a60` | `0x00007f091da93a60` | 1---- | 0 | — |  |
| `1d` | `lower.tac.a97310` | `0x00007f091da97310` | 1---- | 0 | — |  |
| `1d` | `lower.tac.a9949c` | `0x00007f091da9949c` | ---4- | 0 | — |  |
| `1d` | `lower.tac.a99579` | `0x00007f091da99579` | 1---- | 0 | — |  |
| `1d` | `lower.tac.a9bee3` | `0x00007f091da9bee3` | 1---- | 0 | — |  |
| `1d` | `lower.tac.a9fc03` | `0x00007f091da9fc03` | 1---- | 0 | — |  |
| `1d` | `lower.tac.aac070` | `0x00007f091daac070` | 1---- | 0 | — |  |
| `1d` | `lower.tac.aac081` | `0x00007f091daac081` | ---4- | 0 | — |  |
| `1d` | `lower.tac.aace80` | `0x00007f091daace80` | ---4- | 0 | — |  |
| `1d` | `lower.tac.ab0c27` | `0x00007f091dab0c27` | 1---- | 0 | — |  |
| `1d` | `lower.tac.ab4366` | `0x00007f091dab4366` | ---4- | 0 | — |  |
| `1d` | `lower.tac.ab4375` | `0x00007f091dab4375` | 1---- | 0 | — |  |
| `1d` | `lower.tac.ac27e8` | `0x00007f091dac27e8` | 1---- | 0 | — |  |
| `1d` | `lower.tac.ac27fe` | `0x00007f091dac27fe` | 1---- | 0 | — |  |
| `1d` | `lower.tac.ac2808` | `0x00007f091dac2808` | 1---- | 0 | — |  |
| `1d` | `lower.tac.acb2ea` | `0x00007f091dacb2ea` | 1---- | 0 | — |  |
| `1d` | `lower.tac.af99e4` | `0x00007f091daf99e4` | 1---- | 0 | — |  |
| `1d` | `lower.tac.b8e7a1` | `0x00007f091db8e7a1` | 1---- | 0 | — |  |
| `1d` | `lower.tac.ed48a7` | `0x00007f091ded48a7` | 1---- | 0 | — |  |
| `1d` | `lower.tac.f075a8` | `0x00007f091df075a8` | ---4- | 0 | — |  |
| `1d` | `lower.tac.f08f21` | `0x00007f091df08f21` | 1---- | 0 | — |  |
| `1d` | `lower.tactic.leaf` | `0x00007f091da50e14` | 1---- | 0 | — | ★ innermost lowering / tactic search LEAF — origin of 'failed to lower (no tactic)' |
| `1f` | `pool.worker.park` | `0x00007f091fbccdaa` | 12345 | 310 | syscall(x310) | ★ worker-pool thread entry; 62 threads park here -> syscall (work-queue wait) |

총 802 개 네이티브 주소. region 5개 + 명시 hot 12개로 명명, 나머지는 region 접두사+주소꼬리. 모든 이름은 추론치입니다.
