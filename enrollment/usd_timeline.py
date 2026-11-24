"""
PC-side event log for SD-deck (micro-SD, ~1 kHz onboard) recordings.

Every script that drives the drone while onboard logging is running
(test_usd_logging.py, tap_test.py, main.py's --usd static sweep) writes one
of these: the wall-clock time of every phase change and pwm command it
issues.  usd_convert.py later aligns this timeline to the SD log's own
timebase (via the usd_start/usd_stop event pair) to reconstruct the
phase and motor.m1-m4 columns, which the SD log itself does not contain
(the firmware's motor log variables read 0 under motorPowerSet override).
"""

import csv
import time
from pathlib import Path


class Timeline:
    """PC-side event log: wall-clock time of every command we issue."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(["wall_time_s", "event", "value"])
        self.path = path

    def log(self, event: str, value="") -> None:
        self._w.writerow([f"{time.time():.6f}", event, value])
        self._f.flush()

    def close(self) -> None:
        self._f.close()


class UsdPhaseRecorder:
    """
    Adapts enrollment.sweep's recorder duck-type (only ``set_phase()`` is
    ever called on it) to a Timeline log, so ``run_static_sweep`` can drive
    an SD-deck recording exactly like it drives a 100 Hz DataRecorder one —
    same excitation protocol, same phase labels, no duplicated logic.
    Shared by main.py's ``--usd`` flag and the dashboard's Offline Analysis
    tab.
    """

    def __init__(self, tl: Timeline) -> None:
        self._tl = tl

    def set_phase(self, phase: str) -> None:
        self._tl.log("phase", phase)
