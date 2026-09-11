#!/usr/bin/env python3
"""측정하는 동안 0.5초마다 메모리 회수와 NUMA 관련 커널 카운터를 JSONL 로 남긴다.

  diag_sampler.py <출력.jsonl>        # 끌 때는 프로세스를 종료한다

length_probe.py 의 요청마다 t0_wall(벽시계 시작 시각)과 조각 시각이 있으므로, 나중에 멈춤이 난 시각의
카운터 증가량을 조용한 시각과 비교할 수 있다. 가설: NUMA 노드 1 의 여유 메모리가 바닥(2026-09-11 144 MB,
페이지 캐시 43 GB)이라, 넉 장짜리 serve 의 노드 1 쪽 스레드가 할당할 때마다 직접 회수로 멈춘다.
"""
import json, sys, time

KEYS = ("allocstall_normal", "allocstall_movable", "pgscan_direct", "pgsteal_direct", "pgscan_kswapd",
        "pgsteal_kswapd", "compact_stall", "numa_hint_faults", "numa_pages_migrated", "pgmajfault", "workingset_refault_file")


def vmstat():
    d = {}
    for line in open("/proc/vmstat"):
        k, v = line.split()
        if k in KEYS:
            d[k] = int(v)
    return d


def psi(kind):
    some = open(f"/proc/pressure/{kind}").readline().split()
    return int(some[-1].split("=")[1])      # total= 누적 마이크로초


def node_free():
    out = {}
    for n in (0, 1):
        for line in open(f"/sys/devices/system/node/node{n}/meminfo"):
            if "MemFree" in line:
                out[f"node{n}_free_kb"] = int(line.split()[-2])
    return out


def main():
    path = sys.argv[1]
    with open(path, "a", encoding="utf-8") as fh:
        while True:
            rec = {"t": round(time.time(), 3), **vmstat(), "psi_mem_us": psi("memory"), "psi_io_us": psi("io"),
                   "psi_cpu_us": psi("cpu"), **node_free()}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            time.sleep(0.5)


if __name__ == "__main__":
    main()
