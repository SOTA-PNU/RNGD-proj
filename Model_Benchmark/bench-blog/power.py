"""디바이스 전력(W) 샘플러 — RNGD(furiosa-smi)와 NVIDIA GPU(nvidia-smi) 공용.

furiosa-smi 는 기계판독용 서브커맨드가 없어서 `furiosa-smi info` 텍스트의 Power 컬럼을 파싱한다.
nvidia-smi 는 `--query-gpu=power.draw` 로 바로 숫자를 얻는다.

백그라운드 스레드로 ~1Hz 샘플링하다가 stop() 하면 평균/최대/표본을 돌려준다. 측정 구간(부하
정상상태) 동안만 켜서 그 구간의 평균 전력을 쓰면 된다.
"""
from __future__ import annotations

import re
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field


@dataclass
class PowerStats:
    samples_w: list[float] = field(default_factory=list)
    backend: str = "none"
    target: str = ""

    @property
    def n(self) -> int:
        return len(self.samples_w)

    @property
    def avg_w(self) -> float:
        return statistics.mean(self.samples_w) if self.samples_w else float("nan")

    @property
    def max_w(self) -> float:
        return max(self.samples_w) if self.samples_w else float("nan")

    @property
    def min_w(self) -> float:
        return min(self.samples_w) if self.samples_w else float("nan")

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "target": self.target,
            "n_samples": self.n,
            "avg_w": round(self.avg_w, 2) if self.n else None,
            "max_w": round(self.max_w, 2) if self.n else None,
            "min_w": round(self.min_w, 2) if self.n else None,
        }


# ---- 백엔드별 1회 측정 ----

_FURIOSA_ROW = re.compile(
    r"\|\s*rngd\s*\|\s*(npu\d+)\s*\|.*?\|\s*[\d.]+\s*°?C\s*\|\s*([\d.]+)\s*W\s*\|",
    re.IGNORECASE,
)


def _read_furiosa(devices: list[int] | None) -> float | None:
    """furiosa-smi info 의 Power 컬럼 합(지정 디바이스만, None이면 전체)을 W로 반환."""
    try:
        out = subprocess.run(
            ["furiosa-smi", "info"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return None
    total = 0.0
    found = False
    for m in _FURIOSA_ROW.finditer(out):
        idx = int(re.sub(r"\D", "", m.group(1)))
        if devices is not None and idx not in devices:
            continue
        total += float(m.group(2))
        found = True
    return total if found else None


def _read_nvidia(devices: list[int] | None) -> float | None:
    """nvidia-smi power.draw 합(지정 GPU만)을 W로 반환."""
    cmd = ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
    if devices is not None:
        cmd += ["-i", ",".join(str(d) for d in devices)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    vals = []
    for line in out.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            vals.append(float(line))
        except ValueError:
            pass
    return sum(vals) if vals else None


_READERS = {"rngd": _read_furiosa, "gpu": _read_nvidia, "nvidia": _read_nvidia}


class PowerSampler:
    """with PowerSampler('rngd', devices=[0]) as ps: ... ; ps.stats() 로 구간 평균전력.

    backend: 'rngd' | 'gpu' | 'none'
    devices: 측정할 디바이스 인덱스 목록(None이면 전체). 1장 벤치면 그 카드만 지정해 정확히.
    """

    def __init__(self, backend: str, devices: list[int] | None = None, interval_s: float = 1.0):
        self.backend = backend
        self.devices = devices
        self.interval_s = interval_s
        self._reader = _READERS.get(backend)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats = PowerStats(backend=backend, target=str(devices if devices is not None else "all"))

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            w = self._reader(self.devices) if self._reader else None
            if w is not None:
                self._stats.samples_w.append(w)
            # interval 맞춰 sleep (측정시간 보정)
            dt = self.interval_s - (time.time() - t0)
            if dt > 0:
                self._stop.wait(dt)

    def start(self):
        if self._reader is None:  # backend 'none' → 샘플링 안 함
            return self
        self._stop.clear()
        self._stats = PowerStats(backend=self.backend, target=self._stats.target)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> PowerStats:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s * 2 + 1)
        return self._stats

    def stats(self) -> PowerStats:
        return self._stats

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def probe(backend: str, devices: list[int] | None = None) -> float | None:
    """1회 즉시 측정 (셋업 점검용)."""
    reader = _READERS.get(backend)
    return reader(devices) if reader else None


if __name__ == "__main__":
    import sys
    be = sys.argv[1] if len(sys.argv) > 1 else "rngd"
    devs = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
    print(f"{be} devices={devs} 즉시 전력 = {probe(be, devs)} W")
