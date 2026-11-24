"""
UAVibes dashboard — session state and background drone worker.

DroneWorker
-----------
Persistent daemon thread that opens a SyncCrazyflie connection, streams IMU
data into a shared ring buffer, and exposes a MotorController reference so
other threads (e.g. the sweep routine) can command motors directly.

An optional DataRecorder can be attached at any time; when attached, every
incoming sample is also written to CSV.

Session state helpers
---------------------
init_state()      — populate st.session_state keys on first load
buffer_to_df()    — convert ring buffer deque to a pandas DataFrame
preprocess_buffer() — run highpass + harmonic decon on recent buffer data
"""

import io
import re
import csv
import sys
import time
import queue
import logging
import threading
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Redirect cflib stderr to a safe ASCII-only stream so cp1252 terminals
# don't crash on Unicode error messages from the radio driver.
class _SafeStderr(io.TextIOWrapper):
    def write(self, s: str) -> int:
        try:
            return super().write(s)
        except (UnicodeEncodeError, UnicodeDecodeError):
            return super().write(s.encode("ascii", errors="replace").decode("ascii"))

try:
    if sys.stderr and not isinstance(sys.stderr, _SafeStderr):
        sys.stderr = _SafeStderr(
            sys.stderr.buffer, encoding="ascii", errors="replace", line_buffering=True
        )
except Exception:
    pass

from crazyflie.connection import open_sync
from crazyflie.imu import ImuLogger
from crazyflie.motors import MotorController
from enrollment.recorder import DataRecorder
from pipeline.preprocessing import (
    highpass, harmonic_deconvolution, infer_fs,
    MOTOR_COLS, MIN_MOTOR_PWM,
)
from pipeline.motor_model import ensure_bootstrapped as _ensure_motor_models_bootstrapped

BUFFER_MAXLEN  = 2000   # 20 s at 100 Hz


# ── DroneWorker ───────────────────────────────────────────────────────────────

class DroneWorker(threading.Thread):
    """
    Background thread: holds the Crazyflie connection open, feeds the ring
    buffer, and accepts motor commands from other threads.

    Lifecycle
    ---------
    1. ``DroneWorker(uri).start()``  — starts connecting in background
    2. ``worker.connected.wait()``   — block until connected (optional)
    3. ``worker.stop()``             — signal clean shutdown
    """

    def __init__(self, uri: str) -> None:
        super().__init__(daemon=True)
        self.uri              = uri
        self.buffer           = deque(maxlen=BUFFER_MAXLEN)
        self._cmds            = queue.SimpleQueue()
        self.connected        = threading.Event()
        self._stop_evt        = threading.Event()
        self.error: "str | None" = None

        # Exposed after connection — other threads may read/call these
        self.cf                                 = None   # raw cflib Crazyflie
        self.motors:   "MotorController | None" = None
        self.imu:      "ImuLogger | None"       = None
        self.recorder: "DataRecorder | None"    = None

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            with open_sync(self.uri) as scf:
                # cflib does NOT raise into this thread when the radio/USB
                # link dies mid-session (e.g. "too many packets lost") — it
                # just fires connection_lost internally and closes its own
                # link object. Without this hook the while-loop below would
                # keep spinning forever, is_alive() would stay True, and the
                # UI would never notice the drone is gone.
                scf.cf.connection_lost.add_callback(self._on_link_lost)

                self.cf     = scf.cf
                self.motors = MotorController(scf.cf)
                self.imu    = ImuLogger(scf.cf)
                self.imu.start(freq=100, callback=self._on_sample, motors=self.motors)
                self.connected.set()

                while not self._stop_evt.is_set():
                    self._drain_cmds()
                    time.sleep(0.02)

                try:
                    self.imu.stop()
                except Exception:
                    pass
                try:
                    self.motors.stop_all()
                except Exception:
                    pass

        except BaseException as exc:
            # Catch SystemExit/KeyboardInterrupt too — cflib may raise these
            # on unrecoverable radio errors ("too many packets lost").
            self.error = str(exc) or type(exc).__name__
        finally:
            self.connected.clear()
            # Nullify references so dead-object access from sweep threads fails
            # gracefully rather than crashing.
            self.cf     = None
            self.motors = None
            self.imu    = None
            if self.recorder is not None:
                try:
                    self.recorder.stop()
                except Exception:
                    pass
                self.recorder = None

    # ── link-loss callback (cflib internal thread) ────────────────────────────

    def _on_link_lost(self, link_uri: str, msg: str) -> None:
        """
        Fired by cflib's own Crazyflie.connection_lost Caller when the link
        driver reports an unrecoverable error (e.g. "too many packets
        lost"). Setting _stop_evt makes the run() loop above exit within
        one 20ms tick, so is_alive() goes False almost immediately and the
        UI's auto-disconnect cleanup can pick it up on the very next rerun.
        """
        self.error = msg or "Connection lost"
        self._stop_evt.set()

    # ── sample callback (cflib thread) ────────────────────────────────────────

    def _on_sample(self, ts: float, data: dict) -> None:
        try:
            entry = {"ts": ts, **data}
            self.buffer.append(entry)
            if self.recorder is not None and self.recorder.is_open:
                self.recorder.record(ts, data)
        except Exception:
            pass  # never let a callback crash the cflib internal thread

    # ── command processing ────────────────────────────────────────────────────

    def _drain_cmds(self) -> None:
        while True:
            try:
                cmd = self._cmds.get_nowait()
            except queue.Empty:
                return
            t = cmd.get("type")
            if t == "stop_motors" and self.motors:
                self.motors.stop_all()
            elif t == "set_freq" and self.imu:
                self.imu.set_frequency(cmd["hz"])

    def send(self, cmd: dict) -> None:
        self._cmds.put(cmd)

    def stop(self) -> None:
        self._stop_evt.set()


# ── session state ─────────────────────────────────────────────────────────────
#
# Motor sweeps run synchronously in the main Streamlit script thread (see
# app.py's ``_run_sweep_blocking``) rather than in a background thread, so
# there is no cross-thread handoff object needed here: st.session_state is
# only ever touched from the main thread, which is the one Streamlit
# actually supports.

def init_state() -> None:
    """Populate every st.session_state key exactly once (idempotent)."""
    _ensure_motor_models_bootstrapped()

    defaults: dict = {
        # Live drone (just the connection now — the old Live tab, and the
        # Mode/Enrollment/Authentication sidebar controls that only it used,
        # were removed 2026-09-07; Offline Analysis has its own independent
        # record/build-model flow and doesn't touch these)
        "worker":        None,
        "uri_value":     "usb://0",
        # Offline analysis
        "off_model":            None,
        "off_model_path":       "",
        "off_test_results":     None,
        "off_test_emb":         None,
        "off_enroll_last_csv":  "",
        "off_test_last_csv":    "",
        # Spectrogram / filter QA tab
        "spec_last_csv":        "",
        "spec_raw_df":          None,
        "spec_phases":          [],
        "spec_df":              None,
        "spec_fs":              None,
        # Human-in-the-loop motor-model calibration tab
        "calib_last_csv":       "",
        "calib_raw_df":         None,
        "calib_fs":             None,
        "calib_segments":       [],
        "calib_idx":            0,
        "calib_confirmed":      [],
        "calib_out_path":       "",
        # Unexpected disconnect info
        "_disconnect_msg":      "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── buffer helpers ────────────────────────────────────────────────────────────

def buffer_to_df(
    buffer: deque,
    last_n: "int | None" = None,
) -> "pd.DataFrame | None":
    """
    Convert the ring buffer to a pandas DataFrame.

    Parameters
    ----------
    last_n : keep only the last N entries (None → all).

    Returns None when the buffer is empty.
    """
    if not buffer:
        return None
    data = list(buffer)
    if last_n is not None:
        data = data[-last_n:]
    if not data:
        return None
    df = pd.DataFrame(data).rename(columns={"ts": "timestamp_s"})
    if "phase" not in df.columns:
        df["phase"] = "live"
    for col in MOTOR_COLS:
        if col not in df.columns:
            df[col] = 0
    # Reset timestamps relative to window start
    df["timestamp_s"] = df["timestamp_s"] - df["timestamp_s"].iloc[0]
    return df.reset_index(drop=True)


def preprocess_csv(
    path: "str | Path",
    phase_filter: "str | list[str] | None" = None,
) -> "tuple[pd.DataFrame, float] | tuple[None, None]":
    """
    Load and preprocess a UAVibes enrollment CSV.

    Returns (df_preprocessed, fs) on success, or (None, None) on error.
    Applies highpass + harmonic deconvolution (skipped if no motor data).
    """
    from pipeline.preprocessing import (
        load_session, infer_fs, highpass,
        harmonic_deconvolution, MIN_MOTOR_PWM,
    )
    try:
        df = load_session(path, phase_filter=phase_filter)
        if df.empty:
            return None, None
        fs = infer_fs(df)
        df = highpass(df, fs)
        if df[MOTOR_COLS].max().max() > MIN_MOTOR_PWM:
            df = harmonic_deconvolution(df, fs)
        return df, fs
    except Exception:
        return None, None


def preprocess_buffer(
    buffer: deque,
    fs: float = 100.0,
    process_s: float = 5.0,
) -> "pd.DataFrame | None":
    """
    Run highpass + harmonic deconvolution on the last ``process_s`` seconds
    of buffer data.  Returns the preprocessed DataFrame with noise.* columns,
    or None if there is not enough data (or preprocessing fails).

    The filter design uses the sampling rate *measured* from the buffer's
    own timestamps (via infer_fs), not the caller-supplied ``fs``. The UI's
    sampling-rate slider can desync from what the drone is actually
    streaming (e.g. a set_freq command sent while disconnected still
    updates session_state even though it never reached the firmware) —
    designing a Butterworth filter for the wrong Nyquist frequency raises a
    ValueError that would otherwise crash the whole Streamlit app.
    """
    n  = int(process_s * fs)
    df = buffer_to_df(buffer, last_n=n)
    if df is None or len(df) < 16:
        return None

    try:
        actual_fs = infer_fs(df)
    except ValueError:
        actual_fs = fs

    try:
        df = highpass(df, actual_fs, cutoff_hz=5.0)
        if df[MOTOR_COLS].max().max() > MIN_MOTOR_PWM:
            df = harmonic_deconvolution(df, actual_fs)
    except ValueError:
        return None

    return df


# ── model explorer ────────────────────────────────────────────────────────────

def list_saved_models(model_dir: "Path | str") -> "list[dict]":
    """
    List ``*.pkl`` model files in ``model_dir``, newest first.

    Each entry: ``{"path": Path, "name": str, "size_kb": float, "mtime": datetime}``.
    Returns an empty list if the directory doesn't exist yet.
    """
    from datetime import datetime as _dt

    model_dir = Path(model_dir)
    if not model_dir.exists():
        return []

    items = []
    for p in sorted(model_dir.glob("*.pkl")):
        stat = p.stat()
        items.append({
            "path":    p,
            "name":    p.stem,
            "size_kb": stat.st_size / 1024.0,
            "mtime":   _dt.fromtimestamp(stat.st_mtime),
        })
    items.sort(key=lambda d: d["mtime"], reverse=True)
    return items


def import_model_file(data: bytes, filename: str, model_dir: "Path | str") -> Path:
    """Write uploaded model bytes into model_dir, returning the saved path."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if not filename.endswith(".pkl"):
        filename += ".pkl"
    out = model_dir / filename
    out.write_bytes(data)
    return out


# ── recording explorer (for multi-session enrollment pooling) ────────────────

def list_saved_csvs(data_dir: "Path | str") -> "list[dict]":
    """
    List ``*.csv`` recordings in ``data_dir``, newest first.

    Each entry: ``{"path": Path, "name": str, "size_kb": float, "mtime": datetime}``.
    Returns an empty list if the directory doesn't exist yet.
    """
    from datetime import datetime as _dt

    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []

    items = []
    for p in sorted(data_dir.glob("*.csv")):
        stat = p.stat()
        items.append({
            "path":    p,
            "name":    p.name,
            "size_kb": stat.st_size / 1024.0,
            "mtime":   _dt.fromtimestamp(stat.st_mtime),
        })
    items.sort(key=lambda d: d["mtime"], reverse=True)
    return items


_SD_LOG_RE = re.compile(r"^log\d+$")


def list_sd_logs(data_dir: "Path | str") -> "list[dict]":
    """
    List raw SD-deck binary log files (``logNN``, no extension — the
    Bitcraze firmware's own naming, copied straight off the card) in
    ``data_dir``, newest first. Same entry shape as list_saved_csvs, except
    sorted by the numeric log id rather than mtime — the Crazyflie has no
    RTC, so every file it writes carries the same bogus fixed timestamp
    (seen 2026-09-06: all logNN files dated 2015-12-31). The log id itself
    is assigned incrementing by the firmware, so it's the only reliable
    "newest first" ordering available here.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []

    items = []
    for p in sorted(data_dir.iterdir()):
        m = _SD_LOG_RE.match(p.name)
        if p.is_file() and m:
            stat = p.stat()
            items.append({
                "path":    p,
                "name":    p.name,
                "size_kb": stat.st_size / 1024.0,
                "log_id":  int(m.group(0)[3:]),
            })
    items.sort(key=lambda d: d["log_id"], reverse=True)
    return items


# ── human-in-the-loop motor-model calibration ────────────────────────────────

def detect_hold_segments(df: pd.DataFrame) -> "list[dict]":
    """
    Find every single-motor m{N}_hold_{pwm} phase in a loaded SD recording,
    for the human-in-the-loop motor-model calibration tool (see
    usd_motor_analysis.rank_candidates for the peak-scoring it's paired
    with). Returns {"phase", "motor", "pwm"} dicts sorted by (motor, pwm).

    Ramps and all-four phases are deliberately excluded here: holds are
    steady-state (one clean, unambiguous spectral line per motor), which is
    what this tool needs — ramps smear the line across frequency, and
    all-four phases mix four lines together (a harder attribution problem
    to revisit once the single-motor model this builds is solid).
    """
    from usd_motor_analysis import _HOLD_RE
    segs = []
    for phase in df["phase"].dropna().unique():
        m = _HOLD_RE.match(str(phase))
        if m:
            segs.append({"phase": str(phase), "motor": int(m.group(1)), "pwm": int(m.group(2))})
    segs.sort(key=lambda s: (s["motor"], s["pwm"]))
    return segs


def append_calibration_row(path: "Path | str", row: dict) -> None:
    """
    Append one confirmed (motor, pwm, vbat, f0) row to the persistent
    human-in-the-loop calibration CSV, writing the header on first write.

    Appends immediately on every confirmation rather than batching at the
    end of a session — a crashed or refreshed dashboard should never lose
    already-confirmed ground truth.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
