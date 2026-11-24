"""
Thread-safe CSV recorder for UAVibes enrollment sessions.

Each session writes one CSV file.  The recorder is reusable: call start()
to open a new file, record() to append rows, stop() to flush and close.

CSV columns
-----------
timestamp_s  — seconds since recorder.start() was called
phase        — label for the current segment (e.g. "all_four_ramp")
acc.x/y/z    — accelerometer in g
gyro.x/y/z   — gyroscope in deg/s
motor.m1-m4  — motor PWM 0-65535 (0 if firmware does not expose motor vars)
"""

import csv
import time
import threading
from pathlib import Path

COLUMNS = [
    "timestamp_s",
    "phase",
    "acc.x", "acc.y", "acc.z",
    "gyro.x", "gyro.y", "gyro.z",
    "motor.m1", "motor.m2", "motor.m3", "motor.m4",
]


class DataRecorder:
    """Write IMU + motor samples to a CSV file, thread-safely."""

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._file    = None
        self._writer  = None
        self._path:   Path | None = None
        self._phase   = "unknown"
        self._start_t = 0.0
        self._count   = 0

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self, path: Path, phase: str = "static_sweep") -> None:
        """Open a new CSV file and write the header."""
        with self._lock:
            if self._file is not None:
                self._file.flush()
                self._file.close()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file    = open(path, "w", newline="", encoding="utf-8")
            self._writer  = csv.DictWriter(
                self._file, fieldnames=COLUMNS, extrasaction="ignore"
            )
            self._writer.writeheader()
            self._path    = path
            self._phase   = phase
            self._start_t = time.time()
            self._count   = 0

    def stop(self) -> tuple[Path | None, int]:
        """Flush, close, and return (path, sample_count)."""
        with self._lock:
            if self._file:
                self._file.flush()
                self._file.close()
                self._file   = None
                self._writer = None
            return self._path, self._count

    # ── runtime API ───────────────────────────────────────────────────────────

    def set_phase(self, phase: str) -> None:
        """Change the phase label stamped on subsequent rows."""
        with self._lock:
            self._phase = phase

    def record(self, ts: float, data: dict) -> None:
        """
        Append one sample row.  Called from the cflib callback thread.
        ts  — wall-clock timestamp from ImuLogger callback
        data — dict with acc.x/y/z, gyro.x/y/z, [motor.m1-m4]
        """
        with self._lock:
            if self._writer is None:
                return
            row = {
                "timestamp_s": f"{ts - self._start_t:.6f}",
                "phase":       self._phase,
            }
            row.update(data)
            self._writer.writerow(row)
            self._count += 1

    # ── read-only state ───────────────────────────────────────────────────────

    @property
    def sample_count(self) -> int:
        with self._lock:
            return self._count

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._file is not None
