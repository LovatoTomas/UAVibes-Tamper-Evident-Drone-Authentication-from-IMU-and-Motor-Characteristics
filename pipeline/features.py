"""
UAVibes Signal Processing Pipeline — Step 4: Feature Extraction
===============================================================

Slides a configurable window over the preprocessed noise-floor signal and
extracts a fixed-length feature vector from each window.

Feature set (default: 120 features per window)
-----------------------------------------------
For each of the 6 noise channels  (noise.acc.x/y/z, noise.gyro.x/y/z):

  Statistical (3):
    variance   — total vibrational energy in the window
    skewness   — asymmetry of the amplitude distribution
    kurtosis   — impulsiveness / tail weight (excess kurtosis, Fisher def.)

  Spectral (3):
    centroid   — "centre of gravity" of the power spectrum  [Hz]
    roll-off   — frequency below which 85% of spectral energy lies  [Hz]
    flatness   — geometric/arithmetic mean ratio (0 = tonal, 1 = noise-like)

  Structural (13):
    lfcc00…12  — Linear-Frequency Cepstral Coefficients from a 26-band
                 triangular filterbank covering [0, fs/2].  LFCCs capture
                 the structural transfer function of the chassis independent
                 of absolute energy level.

  Per-channel subtotal: 3 + 3 + 13 = 19 features

Cross-axis coherence (6):
  Mean squared coherence between:
    acc.x–acc.y, acc.x–acc.z, acc.y–acc.z   (structural coupling)
    acc.x–gyro.x, acc.y–gyro.y, acc.z–gyro.z (gyroscopic coupling)

Total: 6 × 19 + 6 = 120 features per window.

Usage
-----
  from pipeline.preprocessing import preprocess
  from pipeline.features import extract_features

  df, fs = preprocess("data/cf_001_static_20260609_120000.csv")
  feat_df = extract_features(df, fs, window_s=1.0, hop_s=0.5)
  X = feat_df.drop(columns=["window_start_s", "phase"]).values
"""

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq, dct
from scipy.signal import coherence
from scipy.stats import skew, kurtosis as scipy_kurtosis

# ── column constants ──────────────────────────────────────────────────────────

NOISE_COLS = [
    "noise.acc.x", "noise.acc.y", "noise.acc.z",
    "noise.gyro.x", "noise.gyro.y", "noise.gyro.z",
]

_COHERENCE_PAIRS = [
    ("noise.acc.x",  "noise.acc.y"),
    ("noise.acc.x",  "noise.acc.z"),
    ("noise.acc.y",  "noise.acc.z"),
    ("noise.acc.x",  "noise.gyro.x"),
    ("noise.acc.y",  "noise.gyro.y"),
    ("noise.acc.z",  "noise.gyro.z"),
]

_META_COLS = {"window_start_s", "phase"}


# ── low-level spectral helpers ────────────────────────────────────────────────

def _power_spectrum(x: np.ndarray, fs: float) -> "tuple[np.ndarray, np.ndarray]":
    """
    Hanning-windowed FFT → (freqs [Hz], one-sided power spectrum).
    Power is normalised so that ``np.sum(power)`` approximates signal variance.
    """
    win   = np.hanning(len(x))
    X     = np.abs(rfft(x * win)) ** 2
    freqs = rfftfreq(len(x), d=1.0 / fs)
    # Guard against all-zero window
    if X.sum() < 1e-20:
        X = np.full_like(X, 1e-20)
    return freqs, X


def _spectral_centroid(freqs: np.ndarray, power: np.ndarray) -> float:
    return float(np.dot(freqs, power) / power.sum())


def _spectral_rolloff(
    freqs: np.ndarray,
    power: np.ndarray,
    roll: float = 0.85,
) -> float:
    cum = np.cumsum(power)
    idx = np.searchsorted(cum, roll * cum[-1])
    return float(freqs[min(idx, len(freqs) - 1)])


def _spectral_flatness(power: np.ndarray) -> float:
    log_mean = np.mean(np.log(np.maximum(power, 1e-20)))
    arith    = float(np.mean(power))
    return float(np.exp(log_mean) / arith) if arith > 1e-20 else 0.0


def _lfcc(
    x: np.ndarray,
    fs: float,
    n_filters: int = 26,
    n_coeffs: int  = 13,
) -> np.ndarray:
    """
    Linear-Frequency Cepstral Coefficients.

    Algorithm
    ---------
    1. Hanning-windowed FFT → power spectrum.
    2. Triangular linear filterbank over [0, fs/2] with n_filters bands.
    3. Log of filter bank energies.
    4. DCT (type-II, ortho norm) → take first n_coeffs coefficients.

    Unlike MFCCs the filterbank is uniform on the linear frequency axis,
    which better resolves the mechanical resonances of a stiff chassis
    (typically 50-500 Hz, evenly distributed in Hz not mel).
    """
    freqs, power = _power_spectrum(x, fs)
    fmax         = freqs[-1]
    centres      = np.linspace(0.0, fmax, n_filters + 2)

    # Build triangular filterbank
    bank = np.zeros((n_filters, len(freqs)))
    for i in range(n_filters):
        lo, mid, hi = centres[i], centres[i + 1], centres[i + 2]
        if mid > lo:
            up = (freqs >= lo) & (freqs <= mid)
            bank[i, up] = (freqs[up] - lo) / (mid - lo)
        if hi > mid:
            down = (freqs > mid) & (freqs <= hi)
            bank[i, down] = (hi - freqs[down]) / (hi - mid)

    energies = np.maximum(bank @ power, 1e-20)
    return dct(np.log(energies), type=2, norm="ortho")[:n_coeffs]


# ── per-channel features ──────────────────────────────────────────────────────

def _channel_features(
    x: np.ndarray,
    fs: float,
    n_lfcc: int,
    prefix: str,
) -> "dict[str, float]":
    """Return a flat dict of all features for one signal channel."""
    freqs, power = _power_spectrum(x, fs)
    lfccs        = _lfcc(x, fs, n_coeffs=n_lfcc)

    out: dict[str, float] = {
        f"{prefix}.var":      float(np.var(x)),
        f"{prefix}.skew":     float(skew(x)),
        f"{prefix}.kurt":     float(scipy_kurtosis(x)),
        f"{prefix}.centroid": _spectral_centroid(freqs, power),
        f"{prefix}.rolloff":  _spectral_rolloff(freqs, power),
        f"{prefix}.flatness": _spectral_flatness(power),
    }
    for i, c in enumerate(lfccs):
        out[f"{prefix}.lfcc{i:02d}"] = float(c)
    return out


# ── public API ────────────────────────────────────────────────────────────────

def extract_window_features(
    window: pd.DataFrame,
    fs: float,
    n_lfcc: int = 13,
) -> "dict[str, float]":
    """
    Extract the complete feature vector from a single time window.

    Parameters
    ----------
    window : DataFrame slice containing ``noise.*`` columns.
    fs     : sampling rate in Hz.
    n_lfcc : LFCC coefficients per channel (default 13).

    Returns
    -------
    Flat dict mapping feature name → float value.
    """
    features: dict[str, float] = {}

    for col in NOISE_COLS:
        if col not in window.columns:
            continue
        x      = window[col].values.astype(float)
        prefix = col.replace("noise.", "")  # e.g. "acc.x"
        features.update(_channel_features(x, fs, n_lfcc, prefix))

    # Cross-axis coherence
    for col_a, col_b in _COHERENCE_PAIRS:
        if col_a not in window.columns or col_b not in window.columns:
            continue
        a = window[col_a].values.astype(float)
        b = window[col_b].values.astype(float)
        nperseg = min(len(a), 64)   # short segment — window may be 100 samples
        _, coh  = coherence(a, b, fs=fs, nperseg=nperseg)
        name_a  = col_a.replace("noise.", "")
        name_b  = col_b.replace("noise.", "")
        features[f"coh.{name_a}__{name_b}"] = float(np.mean(coh))

    return features


def extract_features(
    df: pd.DataFrame,
    fs: float,
    window_s: float = 1.0,
    hop_s:    float = 0.5,
    n_lfcc:   int   = 13,
) -> pd.DataFrame:
    """
    Slide a window over ``df`` and extract features for every hop position.

    Parameters
    ----------
    df       : preprocessed DataFrame containing ``noise.*`` columns.
    fs       : sampling rate in Hz (from ``infer_fs``).
    window_s : window length in seconds (default 1.0 s).
    hop_s    : hop size in seconds (default 0.5 s → 50% overlap).
    n_lfcc   : LFCC coefficients per channel.

    Returns
    -------
    DataFrame with one row per window.  Columns are feature names plus
    ``window_start_s`` and ``phase`` metadata columns.
    """
    window_n = max(8, int(round(window_s * fs)))
    hop_n    = max(1, int(round(hop_s * fs)))
    n        = len(df)

    if n < window_n:
        raise ValueError(
            f"Signal too short: {n} samples, need at least {window_n} "
            f"({window_s:.1f}s at {fs:.0f} Hz)."
        )

    times = df["timestamp_s"].values
    rows: list[dict] = []

    for start in range(0, n - window_n + 1, hop_n):
        end    = start + window_n
        window = df.iloc[start:end]
        feats  = extract_window_features(window, fs, n_lfcc=n_lfcc)

        feats["window_start_s"] = float(times[start])
        feats["phase"] = (
            window["phase"].mode().iloc[0]
            if "phase" in window.columns and not window["phase"].empty
            else ""
        )
        rows.append(feats)

    if not rows:
        raise ValueError("No complete windows extracted.")

    result = pd.DataFrame(rows)

    # Put metadata columns first for readability
    meta  = [c for c in ("window_start_s", "phase") if c in result.columns]
    other = [c for c in result.columns if c not in _META_COLS]
    return result[meta + other]


def feature_matrix(feat_df: pd.DataFrame) -> np.ndarray:
    """
    Extract the numeric feature matrix from the output of ``extract_features``,
    dropping ``window_start_s`` and ``phase`` metadata columns.

    Returns
    -------
    np.ndarray of shape (n_windows, n_features).
    """
    drop = [c for c in feat_df.columns if c in _META_COLS]
    return feat_df.drop(columns=drop).values.astype(float)
