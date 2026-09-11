#!/usr/bin/env python3
"""모델 가중치 파일의 페이지 캐시만 내린다(root 불필요). NUMA 노드 메모리 고갈이 멈춤 원인인지 가리는 실험용.

  diag_evict.py [--dry-run]

drop_caches 는 root 가 필요하지만 posix_fadvise(DONTNEED) 는 읽을 수 있는 파일이면 누구나 쓸 수 있고,
그 파일의 깨끗한 캐시 페이지만 내린다. 서빙 중인 모델의 가중치는 이미 NPU 에 올라가 있어 해가 없고,
다음 적재 때 디스크에서 다시 읽느라 조금 느려질 뿐이다.
(2026-09-11: 노드 1 여유 144 MB, 페이지 캐시 43 GB. 넉 장짜리 Qwen 계열에서 스텝 사이 멈춤이 잦았다)
"""
import os, sys, time

ROOTS = ["/mnt/nvme2n1p1/models/hf/hub", "/mnt/nvme2n1p1/models/artifacts", "/mnt/nvme2n1p1/models/furiosa"]
MIN = 64 << 20   # 64 MiB 넘는 파일만


def node_free():
    out = {}
    for n in (0, 1):
        for line in open(f"/sys/devices/system/node/node{n}/meminfo"):
            if "MemFree" in line:
                out[n] = int(line.split()[-2]) >> 10   # MiB
    return out


def main():
    dry = "--dry-run" in sys.argv
    before = node_free()
    n = size = 0
    for root in ROOTS:
        for dp, _, fs in os.walk(root):
            for f in fs:
                p = os.path.join(dp, f)
                try:
                    if os.path.islink(p) or os.path.getsize(p) < MIN:
                        continue
                    size += os.path.getsize(p)
                    n += 1
                    if not dry:
                        fd = os.open(p, os.O_RDONLY)
                        try:
                            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        finally:
                            os.close(fd)
                except OSError:
                    pass
    time.sleep(2)
    after = node_free()
    print(f"{'(시험) ' if dry else ''}파일 {n}개, {size / 2**30:.0f} GiB 대상. 노드 여유 MiB: "
          f"node0 {before[0]} → {after[0]}, node1 {before[1]} → {after[1]}")


if __name__ == "__main__":
    main()
