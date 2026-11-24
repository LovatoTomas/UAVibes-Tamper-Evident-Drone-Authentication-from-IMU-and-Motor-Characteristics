"""
UAVibes — Empirical PWM -> motor fundamental frequency (RPM) calibration.
==========================================================================

STATUS (2026-08-26): superseded as the source of the PWM->RPM map. The
canonical, laser-tachometer-calibrated model now lives in
``pipeline.motor_model`` (RPM = 0.0934*(PWM*vbat) + 4321). This module is
kept as a *spectral cross-validation* tool: it tries to recover the motor
line from IMU data alone, which is only possible when the fundamental is
below Nyquist — at 100 Hz telemetry it is not (f0 = 250-300 Hz at
operating PWM), which is why its fits on current recordings are poor.
It becomes useful again on high-rate (>=1 kHz) onboard recordings.

Motivation (historical)
------------------------
The original DEFAULT_MAX_RPM = 22 000 linear-through-zero model was an
early-project placeholder, not a measured or datasheet-derived value —
it did not even reproduce from the "820 kV, 3.7 V" figures quoted next
to it (820 * 3.7 = 3034, not 22000). This module estimates the true
value empirically from recorded IMU data, generically for any drone or
motor, using two independent methods:

ramp method (needs no dedicated recording)
  Tracks the dominant vibration-frequency ridge across a normal ramp
  phase (e.g. "all_four_ramp", "m1_only") via short-time FFT, and
  regresses it against the known instantaneous PWM at each time point.
  Works on recordings you already have.

stepped-hold method (needs a run_stepped_hold() recording)
  Cleaner cross-check: one stationary FFT per discrete PWM level
  (phase labels "hold_<pwm>" or "hold_m<id>_<pwm>"), no ramp smearing.

Both fit a through-origin line f0 = k * PWM and report
MAX_RPM = k * 65535 * 60 together with R² and a harmonic-consistency
check: a genuine rotor imbalance line has visible energy at 2x its own
frequency too (when that harmonic is itself below Nyquist). If the
candidate ridge fails this check, or R² is poor, the true fundamental
may lie above what the sampling rate can observe — this module reports
that explicitly rather than returning a falsely confident number.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import stft

from pipeline.preprocessing import load_session, infer_fs, MOTOR_COLS

_ALL_CHANNELS = ["acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"]


# ── shared helpers ────────────────────────────────────────────────────────────

def _fit_through_origin(pwm: np.ndarray, f0: np.ndarray) -> dict:
    """Least-squares fit of f0 = k * pwm forced through the origin."""
    k = float(np.sum(pwm * f0) / np.sum(pwm ** 2))
    f_pred = k * pwm
    ss_res = float(np.sum((f0 - f_pred) ** 2))
    ss_tot = float(np.sum((f0 - f0.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "slope_hz_per_pwm": k,
        "max_rpm_estimate": k * 65535.0 * 60.0,
        "r_squared": r2,
    }


def _harmonic_consistency(freqs: np.ndarray, spec: np.ndarray, f0: float,
                          fs: float, tol_hz: float = 1.5) -> "bool | None":
    """
    Check whether a secondary peak exists near 2*f0 in the same spectrum.
    A genuine rotor-imbalance line shows harmonics; noise usually doesn't.
    Returns None if 2*f0 exceeds Nyquist (can't check), else True/False.
    """
    nyq = fs / 2.0
    if 2 * f0 >= nyq:
        return None
    band = (freqs >= 2 * f0 - tol_hz) & (freqs <= 2 * f0 + tol_hz)
    if not band.any():
        return None
    local_peak = spec[band].max()
    baseline = np.median(spec[(freqs > 3) & (freqs < nyq)])
    return bool(local_peak > 3 * baseline)


# ── method 1: ramp-based (no dedicated recording needed) ─────────────────────

def _calibrate_ramp_single_channel(
    df: pd.DataFrame, fs: float, channel: str,
    search_band_hz: tuple[float, float], min_pwm: float, snr_percentile: float,
) -> dict:
    x = df[channel].values.astype(float)
    x = x - x.mean()

    nperseg = min(256, len(x))
    f, t, Zxx = stft(x, fs=fs, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    power = np.abs(Zxx) ** 2

    band_mask = (f >= search_band_hz[0]) & (f <= search_band_hz[1])
    if not band_mask.any():
        raise ValueError(f"search_band_hz {search_band_hz} outside Nyquist ({fs/2} Hz)")
    f_band = f[band_mask]
    power_band = power[band_mask, :]
    idx = np.argmax(power_band, axis=0)
    f_ridge = f_band[idx]
    p_ridge = power_band[idx, np.arange(power_band.shape[1])]

    ts = df["timestamp_s"].values
    pwm_mean = df[MOTOR_COLS].mean(axis=1).values
    pwm_at_t = np.interp(t, ts - ts[0], pwm_mean)

    keep = (pwm_at_t >= min_pwm) & (p_ridge >= np.percentile(p_ridge, snr_percentile))
    pwm_fit, f_fit = pwm_at_t[keep], f_ridge[keep]
    if len(pwm_fit) < 5:
        raise ValueError("not enough high-confidence ridge points")

    fit = _fit_through_origin(pwm_fit, f_fit)

    # Harmonic-consistency check on the single strongest frame
    best_i = np.argmax(p_ridge[keep])
    frame_idx = np.where(keep)[0][best_i]
    harmonic_ok = _harmonic_consistency(f, power[:, frame_idx], f_ridge[frame_idx], fs)

    fit.update({
        "channel": channel,
        "n_points": len(pwm_fit),
        "pwm_range": (float(pwm_fit.min()), float(pwm_fit.max())),
        "harmonic_consistent": harmonic_ok,
    })
    return fit


def calibrate_from_ramp(
    csv_path: "str | Path",
    phase: str = "all_four_ramp",
    channels: "list[str] | None" = None,
    search_band_hz: tuple[float, float] = (3.0, 48.0),
    min_pwm: float = 2000.0,
    snr_percentile: float = 50.0,
) -> dict:
    """
    Estimate MAX_RPM from a ramp phase already present in a normal UAVibes
    recording — no dedicated calibration sweep required.

    search_band_hz defaults to nearly the full observable band (3 Hz to
    just under Nyquist at 100 Hz) rather than a narrow guess, since the
    whole point is to find out where the fundamental actually is without
    assuming the (unverified) 22 000 RPM constant.

    Tries every IMU channel and returns the best fit (by R²), with all
    per-channel results attached under "all_channel_results" for
    transparency.
    """
    df = load_session(csv_path, phase_filter=phase)
    if df.empty:
        raise ValueError(f"No rows for phase={phase!r} in {csv_path}")
    fs = infer_fs(df)

    chans = channels if channels is not None else _ALL_CHANNELS
    results = []
    for ch in chans:
        try:
            r = _calibrate_ramp_single_channel(
                df, fs, ch, search_band_hz, min_pwm, snr_percentile
            )
            results.append(r)
        except Exception as exc:
            results.append({"channel": ch, "error": str(exc)})

    ok_results = [r for r in results if "error" not in r]
    if not ok_results:
        raise ValueError(f"Calibration failed on every channel: {results}")

    ok_results.sort(key=lambda r: r["r_squared"], reverse=True)
    best = dict(ok_results[0])
    best["method"] = "ramp"
    best["phase"] = phase
    best["fs"] = fs
    best["all_channel_results"] = results
    return best


# ── method 2: stepped-hold (dedicated recording, cleaner) ────────────────────

_HOLD_PATTERN = re.compile(r"^hold(?:_m\d+)?_(\d+)$")


def calibrate_from_stepped_hold(
    csv_path: "str | Path",
    channels: "list[str] | None" = None,
    search_band_hz: tuple[float, float] = (3.0, 48.0),
) -> dict:
    """
    Cross-check calibration using discrete stationary PWM holds from
    run_stepped_hold() (phase labels "hold_<pwm>" or "hold_m<id>_<pwm>").
    Fully stationary FFT per level — no ramp smearing — but requires a
    dedicated recording (CLI: --mode stepped).
    """
    df_full = load_session(csv_path)
    fs = infer_fs(df_full)
    nyq = fs / 2.0

    hold_phases = [p for p in df_full["phase"].unique()
                  if _HOLD_PATTERN.match(str(p))]
    if len(hold_phases) < 3:
        raise ValueError(
            "Fewer than 3 stepped-hold phases found — this needs a "
            "run_stepped_hold() recording (CLI: --mode stepped)."
        )

    chans = channels if channels is not None else _ALL_CHANNELS
    all_results = []
    for ch in chans:
        pwm_f0 = []
        harmonics = []
        for phase in hold_phases:
            pwm = int(_HOLD_PATTERN.match(str(phase)).group(1))
            sub = df_full[df_full["phase"] == phase]
            if len(sub) < 32:
                continue
            x = sub[ch].values.astype(float)
            x = x - x.mean()
            freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
            spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
            band = (freqs >= search_band_hz[0]) & (freqs <= min(search_band_hz[1], nyq - 0.5))
            if not band.any():
                continue
            f0 = float(freqs[band][np.argmax(spec[band])])
            pwm_f0.append((pwm, f0))
            harmonics.append(_harmonic_consistency(freqs, spec, f0, fs))

        if len(pwm_f0) < 3:
            all_results.append({"channel": ch, "error": "not enough usable hold phases"})
            continue

        pwm_arr = np.array([p for p, _ in pwm_f0], dtype=float)
        f0_arr = np.array([f for _, f in pwm_f0], dtype=float)
        fit = _fit_through_origin(pwm_arr, f0_arr)
        fit.update({
            "channel": ch,
            "n_points": len(pwm_f0),
            "pwm_levels": sorted(pwm_f0),
            "harmonic_consistent_fraction": (
                sum(1 for h in harmonics if h) / sum(1 for h in harmonics if h is not None)
                if any(h is not None for h in harmonics) else None
            ),
        })
        all_results.append(fit)

    ok_results = [r for r in all_results if "error" not in r]
    if not ok_results:
        raise ValueError(f"Calibration failed on every channel: {all_results}")

    ok_results.sort(key=lambda r: r["r_squared"], reverse=True)
    best = dict(ok_results[0])
    best["method"] = "stepped_hold"
    best["fs"] = fs
    best["all_channel_results"] = all_results
    return best


# ── report formatting ─────────────────────────────────────────────────────────

def format_report(result: dict) -> str:
    lines = [
        f"Method            : {result['method']}",
        f"Best channel      : {result['channel']}  (R^2 = {result['r_squared']:.3f})",
        f"Sampling rate     : {result['fs']:.0f} Hz  (Nyquist = {result['fs']/2:.0f} Hz)",
        f"n points used     : {result['n_points']}",
        f"Fitted slope      : {result['slope_hz_per_pwm']:.6f} Hz/PWM",
        f"MAX_RPM estimate  : {result['max_rpm_estimate']:.0f}",
        f"Implied f0 @ 65535 PWM : {result['max_rpm_estimate']/60:.1f} Hz",
    ]
    hc = result.get("harmonic_consistent", result.get("harmonic_consistent_fraction"))
    if hc is None:
        lines.append("Harmonic check    : inconclusive (2*f0 exceeds Nyquist -- "
                     "cannot verify the ridge is a true rotor line)")
    elif hc is False or hc == 0.0:
        lines.append("Harmonic check    : FAILED -- candidate ridge shows no energy "
                     "at 2x its frequency. Likely NOT the true motor fundamental; "
                     "treat this estimate with caution.")
    else:
        lines.append(f"Harmonic check    : passed ({hc if isinstance(hc, bool) else f'{hc*100:.0f}%'} "
                     "of levels show a consistent 2nd harmonic)")
    if result["r_squared"] < 0.8:
        lines.append("WARNING: R^2 < 0.8 -- fit quality is weak, treat MAX_RPM "
                     "estimate as unreliable.")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Estimate MAX_RPM (PWM -> motor fundamental frequency) "
                    "from a UAVibes recording, generically for any drone/motor."
    )
    p.add_argument("csv_path")
    p.add_argument("--method", choices=["ramp", "stepped"], default="ramp")
    p.add_argument("--phase", default="all_four_ramp",
                   help="Ramp phase to analyse (ramp method only)")
    args = p.parse_args()

    if args.method == "ramp":
        result = calibrate_from_ramp(args.csv_path, phase=args.phase)
    else:
        result = calibrate_from_stepped_hold(args.csv_path)

    print(format_report(result))
