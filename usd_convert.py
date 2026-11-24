#!/usr/bin/env python3
"""
UAVibes — Convert a micro-SD binary log into the standard UAVibes CSV schema.
============================================================================

Decodes a logNN file written by the SD deck (official Bitcraze format, via
tools/cfusdlog.py) and produces a CSV with the project's standard columns:

  timestamp_s, phase, acc.x/y/z, gyro.x/y/z, motor.m1-m4 [, vbat_v]

The SD log contains IMU (and pm.vbatMV) at high rate but no motor PWM —
the firmware's motor log variables stay at 0 under motorPowerSet override,
which is the only drive path UAVibes uses. PWM and phase labels are instead
reconstructed from the PC-side timeline CSV written by test_usd_logging.py:
the wall-clock instant of the `usd_start` event is aligned to t=0 of the
SD log (the param write that starts logging), and every later pwm/phase
event is placed on the SD timebase by wall-clock offset. Radio latency
makes this alignment good to a few tens of ms — more than enough for
phase labels (seconds long) and comb-filter PWM tracking (256-sample
blocks).

Also prints a spectral summary of the hold segment — the first look at the
band above the 50 Hz telemetry Nyquist — and, when PWM+vbat are available,
compares the strongest measured peaks against the motor model's predicted
shaft and blade-pass lines.

Usage:
  python usd_convert.py data/usd/log00 --timeline data/usd/timeline_X.csv
  python usd_convert.py data/usd/log00              # no timeline: pwm=0, phase=usd_log
"""

import re
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import cfusdlog                                    # official Bitcraze decoder
from pipeline.motor_model import pwm_to_hz, blade_pass_hz
from enrollment.sweep import MOTOR_PAIRS

IMU_VARS = ["acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"]

# Two independent phase-naming vocabularies feed this: test_usd_motors.py's
# per-motor calibration script (m{N}_ramp/hold/off_{pwm}, all_ramp/hold_{pwm})
# and enrollment/sweep.py's real enrollment protocol (m{n}_only/hold,
# {pair}_ramp/hold for the 6 MOTOR_PAIRS, all_four_ramp/hold, ramp_down,
# plus run_stepped_hold's ramp_to_{pwm}/hold_{pwm}). Both are matched here
# so the same converter handles calibration sweeps and real enrollments.
_PHASE_SINGLE_RE   = re.compile(r"^m([1-4])_(?:ramp|hold|off|only)(?:_|$)")
_PHASE_ALL_RE      = re.compile(r"^(?:all|all_four)_(?:ramp|hold)(?:_|$)")
_PHASE_ALL_EXACT   = {"ramp_down"}
_PHASE_STEPPED_RE  = re.compile(r"^(?:ramp_to|hold)_\d+$")
_PHASE_PAIR = {
    f"{label}_{stage}": motors
    for label, motors in MOTOR_PAIRS.items()
    for stage in ("ramp", "hold")
}


def _active_motors_for_phase(phase: str) -> "frozenset[int]":
    """
    Which motor id(s) are physically commanded during this phase, from its
    label alone. baseline/sync_pulses/stop/*_pause/settle_pause/start and
    anything unrecognised return empty (no motor attribution — those
    windows aren't used for per-motor analysis).
    """
    m = _PHASE_SINGLE_RE.match(phase)
    if m:
        return frozenset({int(m.group(1))})
    if (_PHASE_ALL_RE.match(phase) or phase in _PHASE_ALL_EXACT
            or _PHASE_STEPPED_RE.match(phase)):
        return frozenset({1, 2, 3, 4})
    if phase in _PHASE_PAIR:
        return frozenset(_PHASE_PAIR[phase])
    return frozenset()


def load_timeline(path: "Path | str") -> pd.DataFrame:
    tl = pd.read_csv(path)
    required = {"wall_time_s", "event", "value"}
    if not required.issubset(tl.columns):
        raise ValueError(f"Timeline CSV must have columns {required}")
    return tl


def convert(log_path: Path, timeline_path: "Path | None") -> "tuple[pd.DataFrame, dict]":
    data = cfusdlog.decode(str(log_path))
    if data is None or "fixedFrequency" not in data:
        keys = list(data.keys()) if data else []
        raise ValueError(
            f"No 'fixedFrequency' block found in {log_path} (found: {keys}). "
            "Check the config.txt on the card has the 'on:fixedFrequency' section."
        )
    ff = data["fixedFrequency"]

    # Timestamps: decoder yields milliseconds (v1 int ms, v2 float ms)
    ts_ms = np.asarray(ff["timestamp"], dtype=float)
    t = (ts_ms - ts_ms[0]) / 1000.0

    df = pd.DataFrame({"timestamp_s": t})
    missing = [v for v in IMU_VARS if v not in ff]
    if missing:
        raise ValueError(f"IMU variables missing from the SD log: {missing}. "
                         "Check the variable list in config.txt.")
    for v in IMU_VARS:
        df[v] = np.asarray(ff[v], dtype=float)

    if "pm.vbatMV" in ff:
        df["vbat_v"] = np.asarray(ff["pm.vbatMV"], dtype=float) / 1000.0

    # Sampling-rate health check
    dt = np.diff(ts_ms) / 1000.0
    dt = dt[dt > 0]
    stats = {
        "n_samples":  len(df),
        "duration_s": float(t[-1]),
        "fs_median":  float(1.0 / np.median(dt)) if len(dt) else float("nan"),
        "dt_jitter_ms": float(np.std(dt) * 1000.0) if len(dt) else float("nan"),
        "n_gaps_over_5ms": int(np.sum(dt > 0.005)),
        "clock_rate": None,   # filled in below once the timeline is read
    }

    # Phase + PWM from the PC timeline, aligned at the usd_start event.
    #
    # Phase is resolved FIRST and PWM attribution depends on it: the
    # timeline's generic "pwm" event only ever carries a bare value (set_pwm
    # in test_usd_motors.py logs it once per call regardless of which
    # motor_ids were actually commanded), so on its own it cannot say WHICH
    # motor(s) that PWM applied to. The phase label can (test_usd_motors.py
    # names phases m{N}_ramp/hold/off_{pwm} vs all_ramp/hold_{pwm}) — so the
    # generic pwm series still supplies the correct time-resolved VALUE
    # (important during ramps, which log many intermediate pwm events), and
    # the phase-derived active-motor set decides which column(s) get it and
    # which get 0. Before this fix every motor.mX column showed the same
    # broadcast value even when 3 of 4 motors were physically stopped —
    # confirmed 2026-09-02 comparing log04 against the dashboard spectrogram.
    df["phase"] = "usd_log"
    for m in (1, 2, 3, 4):
        df[f"motor.m{m}"] = 0
    if timeline_path is not None:
        tl = load_timeline(timeline_path)
        start_rows = tl[tl["event"] == "usd_start"]
        if start_rows.empty:
            raise ValueError("Timeline has no usd_start event — cannot align.")
        t0_wall = float(start_rows["wall_time_s"].iloc[0])

        # Two-point (offset + rate) alignment instead of offset-only.
        # PC wall-clock (time.time(), used for every timeline event) and the
        # SD deck's own onboard clock are two independent oscillators; a
        # single-point offset at usd_start corrects their starting gap but
        # not a RATE mismatch between them, which grows linearly over a
        # long recording. Confirmed 2026-09-02 on log04 (269s): comparing
        # usd_start -> usd_stop wall-clock elapsed against the SD log's own
        # first -> last sample elapsed showed a 1.1% rate mismatch, i.e.
        # ~3s of drift by the end — exactly the growing background/
        # prediction misalignment seen in the dashboard spectrogram.
        # usd_stop is logged in every session (test_usd_motors.py's finally
        # block) right after the firmware is told to stop logging, so it
        # gives a second reference point for free, no protocol change or
        # new recording needed — this applies retroactively to old timelines
        # too, as long as they have both events.
        rate = 1.0
        stop_rows = tl[tl["event"] == "usd_stop"]
        if not stop_rows.empty:
            t1_wall = float(stop_rows["wall_time_s"].iloc[0])
            wall_elapsed = t1_wall - t0_wall
            sd_elapsed = float(df["timestamp_s"].iloc[-1] - df["timestamp_s"].iloc[0])
            if wall_elapsed > 0:
                rate = sd_elapsed / wall_elapsed
        stats["clock_rate"] = rate

        def _to_sd_time(wall_time_s: "pd.Series") -> "pd.Series":
            return (wall_time_s.astype(float) - t0_wall) * rate

        ph_ev = tl[tl["event"] == "phase"].copy()
        ph_ev["t_sd"] = _to_sd_time(ph_ev["wall_time_s"])
        for _, row in ph_ev.iterrows():
            df.loc[df["timestamp_s"] >= row["t_sd"], "phase"] = str(row["value"])

        pwm_ev = tl[tl["event"] == "pwm"].copy()
        pwm_ev["t_sd"] = _to_sd_time(pwm_ev["wall_time_s"])
        pwm_series = np.zeros(len(df))
        for _, row in pwm_ev.iterrows():
            pwm_series[df["timestamp_s"] >= row["t_sd"]] = float(row["value"])

        active_by_phase = {p: _active_motors_for_phase(p) for p in df["phase"].unique()}
        for m in (1, 2, 3, 4):
            is_active = df["phase"].map(lambda p, m=m: m in active_by_phase[p]).values
            df[f"motor.m{m}"] = np.where(is_active, pwm_series, 0.0).astype(int)

    # Standard column order (extra vbat_v goes last)
    cols = (["timestamp_s", "phase"] + IMU_VARS
            + [f"motor.m{m}" for m in (1, 2, 3, 4)]
            + (["vbat_v"] if "vbat_v" in df.columns else []))
    return df[cols], stats


def spectral_summary(df: pd.DataFrame, fs: float) -> None:
    """Top spectral peaks of the hold segment (or strongest-PWM span)."""
    from scipy.signal import welch, find_peaks

    seg = df[df["phase"] == "hold"]
    if len(seg) < 256:
        pwm = df[[f"motor.m{m}" for m in (1, 2, 3, 4)]].max(axis=1)
        seg = df[pwm >= pwm.max() * 0.9] if pwm.max() > 0 else df
    if len(seg) < 256:
        print("  (segment too short for a spectral summary)")
        return

    mean_pwm  = float(seg[[f"motor.m{m}" for m in (1, 2, 3, 4)]].values.max(axis=1).mean())
    mean_vbat = float(seg["vbat_v"].mean()) if "vbat_v" in seg.columns else None

    print(f"\nSpectral summary — segment '{seg['phase'].iloc[0]}', "
          f"{len(seg)} samples, mean PWM {mean_pwm:.0f}"
          + (f", mean vbat {mean_vbat:.2f} V" if mean_vbat else ""))

    if mean_pwm > 500:
        f0  = pwm_to_hz(mean_pwm, mean_vbat)
        fbp = blade_pass_hz(mean_pwm, mean_vbat)
        print(f"  motor model predicts: shaft f0 = {f0:.1f} Hz, "
              f"blade-pass = {fbp:.1f} Hz")

    for ch in ["acc.x", "acc.z", "gyro.x", "gyro.z"]:
        x = seg[ch].values.astype(float)
        x = x - x.mean()
        nper = min(1024, len(x))
        freqs, psd = welch(x, fs=fs, nperseg=nper)
        psd_db = 10 * np.log10(np.maximum(psd, 1e-20))
        band = freqs >= 20
        peaks, props = find_peaks(psd_db[band], prominence=6.0, distance=5)
        order = np.argsort(-props["prominences"])[:6]
        top = sorted(float(freqs[band][pk]) for pk in peaks[order])
        print(f"  {ch:8s} top peaks (Hz): {[round(p, 1) for p in top]}")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert SD binary log to UAVibes CSV")
    p.add_argument("logfile", type=Path, help="binary logNN file copied from the card")
    p.add_argument("--timeline", type=Path, default=None,
                   help="PC-side timeline CSV from test_usd_logging.py "
                        "(reconstructs motor.m1-m4 and phase)")
    p.add_argument("--out", type=Path, default=None,
                   help="output CSV path (default: <logfile>_1khz.csv)")
    args = p.parse_args()

    out = args.out or args.logfile.with_name(args.logfile.name + "_1khz.csv")

    df, stats = convert(args.logfile, args.timeline)
    df.to_csv(out, index=False)

    print(f"Decoded {args.logfile}")
    print(f"  samples      : {stats['n_samples']:,}")
    print(f"  duration     : {stats['duration_s']:.1f} s")
    print(f"  measured fs  : {stats['fs_median']:.1f} Hz")
    print(f"  dt jitter    : {stats['dt_jitter_ms']:.2f} ms (std)")
    print(f"  gaps > 5 ms  : {stats['n_gaps_over_5ms']}")
    if stats["fs_median"] < 900:
        print("  WARNING: measured rate well below 1 kHz — check config.txt "
              "frequency/mode and card write speed.")
    if stats["clock_rate"] is not None:
        rate = stats["clock_rate"]
        drift_pct = (rate - 1.0) * 100
        print(f"  PC/SD clock rate : {rate:.6f}  ({drift_pct:+.3f}%, corrected in "
              f"phase/pwm alignment)")
        if abs(drift_pct) > 0.3:
            print(f"    -> at {stats['duration_s']:.0f}s this would have been "
                  f"{(rate-1)*stats['duration_s']:+.2f}s of drift uncorrected — "
                  "large enough to matter for anything but the shortest holds.")
    print(f"Saved -> {out}")

    spectral_summary(df, stats["fs_median"])


if __name__ == "__main__":
    main()
