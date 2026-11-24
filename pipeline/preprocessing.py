"""
UAVibes Signal Processing Pipeline — Steps 1-3
==============================================

Step 1 — Ingestion
  Load the enrollment CSV, ensure all expected columns exist, infer the
  sampling rate from the timestamp column.

Step 2 — Macro-deconvolution
  Zero-phase Butterworth highpass filter removes slow flight dynamics
  (accelerometer bias drift, manoeuvre inertia).  Cutoff empirically ≥ 5 Hz.

Step 3 — Harmonic deconvolution (OPTIONAL since 2026-09)
  Originally meant to strip the motors as noise and keep only the chassis
  residual. The motor-swap experiment showed the identity is mostly in the
  motors, so this step is now just one alternative representation (studies
  compare with and without it); the key uses the motor lines directly.
  The spinning motors inject deterministic high-energy lines at the blade-pass
  frequency f₀ = RPM/60 and its harmonics.  An adaptive comb filter (series of
  IIR notch filters) removes them block-by-block, tracking each motor's own
  instantaneous speed from its logged motor.m1-m4 PWM value (each of the four
  motors gets its own notch cascade — they run at measurably different RPM
  for the same PWM, see pipeline/motor_model.py MOTOR_GAIN).

Output convention
-----------------
Both filtering steps write results into *new* columns named ``noise.<channel>``
(e.g. ``noise.acc.x``) so the raw columns are preserved for comparison.
These noise-floor columns are the input to Step 4 (features.py).

PWM → frequency conversion
---------------------------
Delegated to ``pipeline.motor_model`` — the canonical, empirically
calibrated map (laser tachometer, 2026-08-26):

  RPM_base = 0.0934 × (PWM × vbat) + 4321      f₀ = RPM / 60  [Hz]

...times a per-motor gain (2026-09-02, MOTOR_GAIN = {1: 0.973, 2: 0.990,
3: 1.041, 4: 1.049}): the four motors run at measurably different RPM for
the same PWM, so harmonic_deconvolution notches each motor.mX column at
its own frequency rather than one frequency averaged across all four.

Battery voltage matters (~3000 RPM/V); when no live vbat is available the
model's NOMINAL_VBAT is used. See motor_model.py for provenance and limits.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, iirnotch, lfilter

from pipeline.motor_model import pwm_to_hz, MIN_SPIN_PWM

# ── column name constants ─────────────────────────────────────────────────────

IMU_COLS   = ["acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"]
MOTOR_COLS = ["motor.m1", "motor.m2", "motor.m3", "motor.m4"]
NOISE_COLS = ["noise." + c for c in IMU_COLS]

# ── tunable thresholds ────────────────────────────────────────────────────────

MIN_MOTOR_PWM = MIN_SPIN_PWM   # PWM below this → motor considered stopped
MIN_MOTOR_HZ  = 10.0    # skip notch if f₀ < this (too close to highpass cutoff)


# ── Step 1: ingestion ─────────────────────────────────────────────────────────

def load_session(
    path: "Path | str",
    phase_filter: "str | list[str] | None" = None,
) -> pd.DataFrame:
    """
    Load a UAVibes enrollment CSV produced by auth_client/main.py.

    Parameters
    ----------
    phase_filter
        Keep only rows whose ``phase`` column matches this value or list.
        ``None`` keeps all rows.

    Returns
    -------
    DataFrame with columns: timestamp_s, phase, acc.*/gyro.*, motor.m1-m4.
    Missing motor columns are added as zeros (older recordings without
    motor telemetry).
    """
    df = pd.read_csv(path, dtype={"phase": str})
    df["timestamp_s"] = df["timestamp_s"].astype(float)

    for col in MOTOR_COLS:
        if col not in df.columns:
            df[col] = 0

    if phase_filter is not None:
        phases = [phase_filter] if isinstance(phase_filter, str) else list(phase_filter)
        df = df[df["phase"].isin(phases)].reset_index(drop=True)

    return df


def infer_fs(df: pd.DataFrame) -> float:
    """
    Estimate the sampling rate from the median inter-sample interval.

    Returns a rounded float (e.g. 100.0, 50.0) so downstream code can treat
    it as a nominal rate rather than a noisy measurement.
    """
    dt = df["timestamp_s"].diff().dropna()
    valid = dt[(dt > 0) & (dt < 1.0)]
    if valid.empty:
        raise ValueError("Cannot infer sampling rate — check timestamp_s column.")
    median_dt = valid.median()
    return round(1.0 / median_dt)


# ── Step 2: macro-deconvolution (highpass) ───────────────────────────────────

def highpass(
    df: pd.DataFrame,
    fs: float,
    cutoff_hz: float = 5.0,
    order: int = 4,
) -> pd.DataFrame:
    """
    Zero-phase Butterworth highpass filter on all 6 IMU channels.

    Writes results into ``noise.<channel>`` columns; original columns are
    unchanged.  ``sosfiltfilt`` is used for zero-phase response.

    Parameters
    ----------
    cutoff_hz
        Highpass cutoff frequency.  5 Hz strips slow manoeuvre dynamics while
        keeping structural resonances (typically 20–500 Hz).
    order
        Filter order.  4th-order gives −80 dB/decade roll-off.
    """
    out = df.copy()
    sos = butter(order, cutoff_hz, btype="high", fs=fs, output="sos")
    for col in IMU_COLS:
        out["noise." + col] = sosfiltfilt(sos, df[col].values.astype(float))
    return out


# ── Step 3: harmonic deconvolution (adaptive comb filter) ────────────────────
# pwm_to_hz is imported from pipeline.motor_model (canonical empirical map)
# and re-exported here for existing callers (e.g. ui/app.py).

def _notch_cascade(
    signal: np.ndarray,
    f0: float,
    fs: float,
    n_harmonics: int,
    Q: float,
) -> np.ndarray:
    """
    Apply a cascade of IIR notch filters at f₀, 2f₀, … n_harmonics×f₀.

    Uses causal ``lfilter`` (one-pass) for block-by-block processing.
    Harmonics above the Nyquist limit are silently skipped.
    """
    out  = signal.copy()
    nyq  = fs / 2.0 - 1.0
    for k in range(1, n_harmonics + 1):
        fk = k * f0
        if fk >= nyq:
            break
        b, a = iirnotch(fk, Q, fs)
        out  = lfilter(b, a, out)
    return out


def harmonic_deconvolution(
    df: pd.DataFrame,
    fs: float,
    n_harmonics: int = 5,
    Q: float = 22.0,
    vbat: "float | None" = None,
    block_n: int = 256,
) -> pd.DataFrame:
    """
    Block-wise adaptive comb filter using motor PWM to track motor frequency.

    For each non-overlapping block of ``block_n`` samples, EACH active
    motor's own PWM is converted to its own fundamental frequency f₀ (via
    the empirical motor model, including its validated per-motor
    MOTOR_GAIN — see ``pipeline.motor_model``), and a separate notch
    cascade is applied for it. Four motors spinning together therefore
    means up to four distinct notch cascades per block, not one averaged
    one: the four motors differ by several percent in real RPM at a given
    PWM (2026-09-02 calibration), which is comparable to or larger than a
    single notch's bandwidth (~4.5 % of its centre frequency at Q=22) — a
    shared averaged f₀ would center the notch on none of the four real
    lines precisely.

    Blocks where a given motor reads below ``MIN_MOTOR_PWM`` are skipped
    for that motor only (motor off — nothing to remove for it that block).

    Operates on ``noise.*`` columns if present (i.e. run after highpass);
    falls back to raw IMU columns otherwise.

    Parameters
    ----------
    n_harmonics
        Number of harmonics to remove per motor (f₀, 2f₀ … n_harmonics×f₀).
        Harmonics at or above the Nyquist limit are silently skipped (see
        ``_notch_cascade``) — relevant at 1 kHz, where f₀ ≈ 250-300 Hz
        means only the first ~1-2 harmonics actually fit under fs/2.
    Q
        Quality factor of each notch.  Higher Q → narrower notch band.
        Q = 22 (2026-09-07, chosen for the first 1 kHz onboard enrollment
        pass) gives ~4.5 % bandwidth relative to the notch frequency —
        wider than the previous Q = 30 default, to stay robust to the f₀
        error introduced by using a single mean PWM/vbat per block instead
        of the true instantaneous value.
    vbat
        Fallback battery voltage when the DataFrame has no ``vbat_v``
        column; ``None`` then uses the motor model's ``NOMINAL_VBAT``.
        When ``vbat_v`` IS present (recordings converted from the SD deck
        via ``usd_convert.py`` carry it from ``pm.vbatMV``), the actual
        per-block mean battery voltage is used instead — the ~3000 RPM/V
        sensitivity means this tightens f₀ by several Hz over a
        recording's battery sag, compared to one fixed nominal value.
    block_n
        Processing block length in samples.  At 100 Hz, 256 samples ≈ 2.56 s;
        at 1 kHz, ≈ 0.26 s — in both cases short enough that the motor
        frequency (and, at 1 kHz, vbat) changes little within a block.
    """
    out = df.copy()

    # Prefer noise.* (post-highpass), fall back to raw
    src_cols = [
        ("noise." + c) if ("noise." + c) in df.columns else c
        for c in IMU_COLS
    ]
    dst_cols = ["noise." + c for c in IMU_COLS]

    motor_pwm = {
        motor_id: df[col].values.astype(float)
        for motor_id, col in enumerate(MOTOR_COLS, start=1)
    }
    vbat_col = df["vbat_v"].values.astype(float) if "vbat_v" in df.columns else None
    n = len(df)

    for src, dst in zip(src_cols, dst_cols):
        raw    = df[src].values.astype(float)
        result = raw.copy()

        for start in range(0, n, block_n):
            end   = min(start + block_n, n)
            block = raw[start:end]

            block_vbat = float(vbat_col[start:end].mean()) if vbat_col is not None else vbat

            for motor_id, pwm_col in motor_pwm.items():
                pwms   = pwm_col[start:end]
                active = pwms[pwms > MIN_MOTOR_PWM]
                if active.size == 0:
                    continue  # this motor off — nothing to remove for it
                mean_pwm = active.mean()
                f0 = pwm_to_hz(mean_pwm, block_vbat, motor_id=motor_id)
                if f0 < MIN_MOTOR_HZ:
                    continue
                block = _notch_cascade(block, f0, fs, n_harmonics, Q)

            result[start:end] = block

        out[dst] = result

    return out


# ── Combined pipeline entry point ─────────────────────────────────────────────

def preprocess(
    path: "Path | str",
    hp_cutoff_hz:  float = 5.0,
    n_harmonics:   int   = 5,
    Q:             float = 22.0,
    vbat:          "float | None" = None,
    phase_filter:  "str | list[str] | None" = None,
) -> "tuple[pd.DataFrame, float]":
    """
    Full preprocessing pipeline: load → highpass → harmonic deconvolution.

    Parameters
    ----------
    path
        Path to a UAVibes enrollment CSV file.
    hp_cutoff_hz
        Highpass cutoff frequency for macro-deconvolution (Step 2).
    n_harmonics
        Harmonics to remove in comb filter (Step 3).
    Q
        Notch quality factor (Step 3, default 22 — see
        ``harmonic_deconvolution``'s docstring).
    vbat
        Fallback battery voltage for the PWM→f₀ motor model when the CSV
        has no ``vbat_v`` column; ``None`` uses the nominal value. Ignored
        (the CSV's own per-block vbat is used instead) for recordings
        converted from the SD deck via ``usd_convert.py`` (Step 3).
    phase_filter
        Optional phase label(s) to keep (e.g. ``"all_four_hold"``).

    Returns
    -------
    (df, fs)
        df  — original columns + noise.acc.x/y/z + noise.gyro.x/y/z
        fs  — inferred sampling rate in Hz
    """
    df = load_session(path, phase_filter=phase_filter)
    fs = infer_fs(df)
    print(f"  [preprocess] Loaded {len(df):,} samples at {fs:.0f} Hz "
          f"from {Path(path).name}")

    df = highpass(df, fs, cutoff_hz=hp_cutoff_hz)
    print(f"  [preprocess] Highpass @ {hp_cutoff_hz} Hz applied.")

    has_motors = (df[MOTOR_COLS].max().max() > MIN_MOTOR_PWM)
    if has_motors:
        df = harmonic_deconvolution(df, fs,
                                    n_harmonics=n_harmonics, Q=Q,
                                    vbat=vbat)
        print(f"  [preprocess] Harmonic deconvolution: "
              f"{n_harmonics} harmonics, Q={Q}.")
    else:
        print("  [preprocess] No motor data — harmonic deconvolution skipped.")

    return df, fs
