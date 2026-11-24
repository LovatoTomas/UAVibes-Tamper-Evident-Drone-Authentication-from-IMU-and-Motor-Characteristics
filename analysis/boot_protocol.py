"""
Boot-protocol design (2026-10-09): which motor excitation, and how long,
does key generation need?

From the stepped ladders (each motor alone, 7 PWM levels, 15 s holds) the
per-motor features (speed gain + broadband floor, 6 channels) are recomputed
on TRUNCATED holds — the first D seconds after a 0.5 s settle — so every
candidate protocol "motor m at PWM levels S, D seconds each" can be evaluated
without new recordings. Each candidate is then plugged into the combined
motors + board fuzzy-extractor simulation (combined_key.run_config).

Also extracted, exploratory: the speed LAG of each motor while following the
5 s linear ramp between ladder steps (a first-order motor lags a ramp input by
its mechanical time constant, which depends on friction and inertia).

Run:  python -m analysis.boot_protocol
"""

import json
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import stft

from analysis.dataset import ROOT, iter_recordings
from analysis.motor_features import hold_features, CHANNELS
from analysis.keygen import session_starts
from analysis import combined_key as ck
from pipeline.preprocessing import load_session, infer_fs, highpass
from pipeline.motor_model import pwm_to_hz

OUT = ROOT / "analysis" / "results_boot"
DURATIONS = (2.0, 3.0, 5.0, 10.0)
SETTLE_S = 0.5
_HOLD = re.compile(r"^m([1-4])_hold_(\d+)$")
_RAMP = re.compile(r"^m([1-4])_ramp_(\d+)$")


def ramp_lag(seg: pd.DataFrame, fs: float, motor: int) -> "float | None":
    """Lag (s) of the measured shaft frequency behind the commanded ramp."""
    pwm = seg[f"motor.m{motor}"].values.astype(float)
    vb = float(seg["vbat_v"].mean())
    if pwm.max() < 9000:
        return None
    x = sum(seg[f"noise.{c}"].values for c in ("acc.x", "acc.y", "gyro.x", "gyro.y"))
    f, tt, Z = stft(x - x.mean(), fs=fs, nperseg=256, noverlap=256 - 25)
    P = np.abs(Z) ** 2
    t_idx = (tt * fs).astype(int).clip(0, len(seg) - 1)
    pred = np.array([pwm_to_hz(p, vb) for p in pwm[t_idx]])
    meas = []
    for j, fp in enumerate(pred):
        band = (f > fp * 0.8) & (f < fp * 1.2)
        meas.append(f[band][np.argmax(P[band, j])] if band.any() and fp > 30 else np.nan)
    meas = np.array(meas)
    ok = np.isfinite(meas) & (pred > 30)
    if ok.sum() < 20:
        return None
    dt = tt[1] - tt[0]
    best, best_err = None, np.inf
    for lag_steps in range(0, int(0.6 / dt)):
        a, b = meas[lag_steps:], pred[:len(pred) - lag_steps]
        m = np.isfinite(a) & (b > 30)
        if m.sum() < 15:
            break
        g = np.sum(a[m] * b[m]) / np.sum(b[m] ** 2)
        err = np.mean((a[m] - g * b[m]) ** 2)
        if err < best_err:
            best, best_err = lag_steps * dt, err
    return best


def _one(rec):
    m = re.match(r"^stepped-m([1-4])$", rec["purpose"])
    if not m:
        return []
    motor = int(m.group(1))
    df = load_session(rec["path"])
    fs = infer_fs(df)
    hp = highpass(df, fs)
    rows = []
    meta = {k: rec[k] for k in ("drone", "kind", "session", "day")}
    for ph in hp.phase.dropna().unique():
        mh = _HOLD.match(str(ph))
        if mh and int(mh.group(1)) == motor:
            seg = hp[hp.phase == ph]
            s0 = int(SETTLE_S * fs)
            for D in DURATIONS:
                part = seg.iloc[s0:s0 + int(D * fs)]
                if len(part) < int(0.9 * D * fs):
                    continue
                r = hold_features(part, fs, motor)
                if r:
                    rows.append(dict(meta, motor=motor, step=int(mh.group(2)), D=D, what="hold",
                                     gain=r["gain"], **{f"floor_{c}": r[f"floor_{c}"] for c in CHANNELS}))
        mr = _RAMP.match(str(ph))
        if mr and int(mr.group(1)) == motor:
            lag = ramp_lag(hp[hp.phase == ph], fs, motor)
            if lag is not None:
                rows.append(dict(meta, motor=motor, step=int(mr.group(2)), D=np.nan, what="ramp", lag=lag))
    return rows


def extract(processes: int = 16) -> pd.DataFrame:
    f = OUT / "boot_features.csv"
    if f.exists():
        return pd.read_csv(f)
    OUT.mkdir(parents=True, exist_ok=True)
    recs = [r for r in iter_recordings() if r["purpose"].startswith("stepped-m")]
    with Pool(processes) as pool:
        parts = pool.map(_one, recs, chunksize=1)
    df = pd.DataFrame([r for p in parts for r in p])
    df.to_csv(f, index=False)
    return df


def motor_table_for(F: pd.DataFrame, steps, D) -> "tuple[pd.DataFrame, list]":
    h = F[(F.what == "hold") & (F.D == D) & F.step.isin(steps)]
    long = h.melt(id_vars=["drone", "kind", "session", "day", "motor", "step"],
                  value_vars=["gain"] + [f"floor_{c}" for c in CHANNELS], var_name="feat")
    long["name"] = "m" + long.motor.astype(str) + "_" + long.step.astype(str) + "_" + long.feat
    W = long.pivot_table(index=["drone", "kind", "session", "day"], columns="name", values="value").reset_index()
    W["stage"], W["motors"] = "", W.drone
    starts = session_starts()
    W["start"] = [starts.get((r.drone, r.kind, int(r.session)), "") for r in W.itertuples()]
    feats = [c for c in W.columns if c.startswith("m") and c[1:2].isdigit()]
    W = W.dropna(subset=feats, thresh=int(0.9 * len(feats)))
    return W, feats


def lag_separation(F: pd.DataFrame) -> pd.DataFrame:
    r = F[F.what == "ramp"]
    rows = []
    for (motor, step), g in r.groupby(["motor", "step"]):
        w = g.groupby("drone").lag.std().mean()
        b = g.groupby("drone").lag.mean().std()
        rows.append(dict(motor=motor, step=step, n=len(g), within=w, between=b, ratio=b / w if w else np.nan,
                         **{f"mean_{d}": g[g.drone == d].lag.mean() for d in "ABCD"}))
    return pd.DataFrame(rows)


CANDIDATES = {
    "58k": [58000],
    "45k": [45000],
    "35k": [35000],
    "25k": [25000],
    "15k": [15000],
    "15k+35k": [15000, 35000],
    "25k+45k": [25000, 45000],
    "15k+35k+58k": [15000, 35000, 58000],
    "ladder(7)": [10000, 15000, 20000, 25000, 35000, 45000, 58000],
}


def main():
    F = extract()
    B, bf = ck.board_table()
    _, hb_ = ck.params(B, bf)
    wb, hb = ck.params(B, bf)
    out = {"lag": lag_separation(F).round(4).to_dict(orient="records"), "candidates": {}}
    print(lag_separation(F).round(3).to_string(index=False))
    for name, steps in CANDIDATES.items():
        for D in DURATIONS:
            M, mf = motor_table_for(F, steps, D)
            if M.empty:
                continue
            ck.K = 8.0
            wm, hm = ck.params(M[M.kind == "enroll"], mf)
            src = {"motors": ck.Source(M, mf, wm), "board": ck.Source(B, bf, wb)}
            best = None
            for t in range(0, 7):
                r = ck.run_config(src, t, None)
                rej = 1 - np.mean([v["accept"] for v in r["genuine"].values()])
                imp = float(np.mean([v for v in r["impostor"].values()]))
                rm = ck.run_config({"motors": src["motors"]}, t, None)
                imp_m = float(np.mean([v for v in rm["impostor"].values()]))
                rej_m = 1 - np.mean([v["accept"] for v in rm["genuine"].values()])
                H = float(hm.sum() + hb.sum())
                row = dict(t=t, genuine_reject=round(float(rej), 3), impostor=round(imp, 4),
                           motors_only_reject=round(float(rej_m), 3), motors_only_impostor=round(imp_m, 3),
                           motor_bits=round(float(hm.sum()), 1), total_bits=round(H, 1),
                           residual=round(max(H - 2 * t * H / (len(mf) + len(bf)), 0), 1))
                if best is None and row["genuine_reject"] <= 0.05:
                    best = row
            motor_seconds = len(steps) * 4 * (D + SETTLE_S + 1.0)    # + ~1 s ramp per level
            out["candidates"][f"{name}|D={D:g}"] = dict(steps=steps, D=D, motor_seconds=motor_seconds, best=best)
            print(f"{name:14s} D={D:4g}s motors≈{motor_seconds:5.0f}s ->", best, flush=True)
    (OUT / "boot_protocol.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
