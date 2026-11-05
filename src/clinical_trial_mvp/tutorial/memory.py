"""Sample the GPU device on its host, or one explicitly identified Mac server PID."""

import subprocess
import threading
import math


class PeakMemory:
    def __init__(self, *, gpu_uuid=None, server_pid=None, interval=.1, reader=None):
        self.gpu_uuid, self.server_pid, self.interval = gpu_uuid, server_pid, interval
        self.reader = reader or self._read
        self.values, self.error = [], None
        self.stop = threading.Event()
        self.thread = None

    def _read(self):
        if self.gpu_uuid:
            output = subprocess.check_output(["nvidia-smi", "--id", self.gpu_uuid,
                "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True,
                stderr=subprocess.DEVNULL, timeout=3)
            return float(output.strip()) * 1024**2
        if self.server_pid:
            try:
                output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(self.server_pid)],
                    text=True, stderr=subprocess.DEVNULL, timeout=3)
            except subprocess.CalledProcessError:
                return None  # The observed edge process may have exited between samples.
            return float(output.strip()) * 1024
        return None

    def _sample(self):
        try:
            value = self.reader()
            if value is not None:
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Invalid memory sample")
                self.values.append(value)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.error = type(exc).__name__

    def _loop(self):
        while not self.stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self._sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=4)
        self._sample()

    def result(self):
        return {"peak_bytes": max(self.values) if self.values and not self.error else None,
                "scope": "device VRAM (includes idle allocation)" if self.gpu_uuid else (
                    "server process RSS; not unified GPU memory" if self.server_pid else "unavailable"),
                "sample_count": len(self.values), "sampling_interval_seconds": self.interval,
                "error": self.error}
