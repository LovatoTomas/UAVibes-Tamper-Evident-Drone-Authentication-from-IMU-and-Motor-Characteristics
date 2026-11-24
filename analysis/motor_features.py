"""
Physical per-motor features (2026-10-02) — the "motor identity" direction
agreed with the professor after the motor-swap experiment.

For every single-motor steady hold (sweep-m{N} at 58k, and the 7 steps of
each stepped-m{N} ladder) the motor's own line is measured directly instead
of being filtered away:

  f0, gain       shaft frequency and measured / base-model speed — how much
                 friction / load the motor sees at that command
  h1_<ch>        level of the f0 line above the channel's noise floor (dB),
                 per IMU channel — how strongly, and in which direction, the
                 rotor's imbalance shakes the airframe
  h2_<ch>, h3_<ch>  2nd / 3rd harmonic relative to the fundamental (dB) —
                 misalignment / non-sinusoidal forcing; only where k*f0 is
                 below 95 % of Nyquist (i.e. the low/mid PWM steps)
  floor_<ch>     broadband level below the line (dB) — brush / bearing noise

One row per hold. Cached in analysis/results_motor/motor_features.csv.
Run:  python -m analysis.motor_features
"""

import re
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import welch

from analysis.dataset import ROOT, iter_recordings
from pipeline.preprocessing import load_session, infer_fs, highpass
from pipeline.motor_model import pwm_to_hz

OUT = ROOT / "analysis" / "results_motor"
CHANNELS = ("acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z")
_HOLD = re.compile(r"^m([1-4])_hold(?:_(\d+))?$")
SEARCH = 0.15       # f0 search window around the base-model prediction
LINE_TOL = 0.015    # +-1.5 % around k*f0 when reading a line level


def _db(x):
    return 10 * np.log10(np.maximum(x, 1e-30))


def hold_features(seg: pd.DataFrame, fs: float, motor: int,
                  search: "tuple[float, float] | None" = None) -> "dict | None":
    """search: (low, high) multiples of the base-model prediction where the
    shaft line is looked for; default (1-SEARCH, 1+SEARCH) as in the studies."""
    pwm = float(seg[f"motor.m{motor}"].mean())
    vbat = float(seg["vbat_v"].mean())
    f_pred = pwm_to_hz(pwm, vbat)
    if f_pred < 20:
        return None
    nper = 4096 if len(seg) >= 8000 else 2048
    psd = {}
    for c in CHANNELS:
        x = seg[f"noise.{c}"].values
        f, p = welch(x - x.mean(), fs=fs, nperseg=min(nper, len(x)))
        psd[c] = p
    nyq = fs / 2
    total = sum(p / np.median(p) for p in psd.values())
    lo, hi = search or (1 - SEARCH, 1 + SEARCH)
    band = (f > f_pred * lo) & (f < min(f_pred * hi, 0.98 * nyq))
    if not band.any():
        return None
    k = int(np.argmax(np.where(band, total, -np.inf)))
    if 0 < k < len(f) - 1:
        a, b, c3 = np.log(total[k - 1:k + 2])
        den = a - 2 * b + c3
        f0 = f[k] + (0.5 * (a - c3) / den if den != 0 else 0.0) * (f[1] - f[0])
    else:
        f0 = f[k]
    out = dict(pwm=pwm, vbat=vbat, f_pred=f_pred, f0=float(f0), gain=float(f0 / f_pred),
               prom_db=float(_db(total[k] / np.median(total[band]))))

    def line(p, fc):
        m = (f > fc * (1 - LINE_TOL)) & (f < fc * (1 + LINE_TOL))
        return _db(p[m].max()) if m.any() else np.nan

    flo = (f > 20) & (f < 0.9 * f0)
    for c, p in psd.items():
        L1 = line(p, f0)
        floor = _db(np.median(p[flo])) if flo.any() else np.nan
        out[f"h1_{c}"] = L1 - floor
        out[f"floor_{c}"] = floor
        for kk in (2, 3):
            out[f"h{kk}_{c}"] = (line(p, kk * f0) - L1) if kk * f0 < 0.95 * nyq else np.nan
    return out


def _one(rec: dict) -> "list[dict]":
    m = re.match(r"^(sweep|stepped)-m([1-4])$", rec["purpose"])
    if not m:
        return []
    motor = int(m.group(2))
    try:
        df = load_session(rec["path"])
    except Exception:
        return []
    fs = infer_fs(df)
    hp = highpass(df, fs)
    rows = []
    for ph in hp["phase"].dropna().unique():
        mm = _HOLD.match(str(ph))
        if not mm or int(mm.group(1)) != motor:
            continue
        seg = hp[hp.phase == ph]
        if len(seg) < 2000:
            continue
        r = hold_features(seg, fs, motor)
        if r:
            meta = {k: rec.get(k) for k in ("drone", "kind", "session", "day", "purpose", "motors", "stage")}
            rows.append(dict(**meta, motor=motor, phase=ph, step=int(round(r["pwm"], -3)), **r))
    return rows


def all_recordings() -> "list[dict]":
    recs = [r for r in iter_recordings() if re.match(r"^(sweep|stepped)-m[1-4]$", r["purpose"])]
    try:
        from analysis.swap import iter_swap_recordings
        recs += [r for r in iter_swap_recordings() if re.match(r"^sweep-m[1-4]$", r["purpose"])]
    except Exception:
        pass
    return recs


def build(processes: int = 16) -> pd.DataFrame:
    recs = all_recordings()
    with Pool(processes) as pool:
        parts = pool.map(_one, recs, chunksize=2)
    df = pd.DataFrame([r for p in parts for r in p])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "motor_features.csv", index=False)
    return df


def load() -> pd.DataFrame:
    f = OUT / "motor_features.csv"
    return pd.read_csv(f) if f.exists() else build()


if __name__ == "__main__":
    d = build()
    print(d.shape)
    print(d.groupby(["kind", "drone"]).size())
