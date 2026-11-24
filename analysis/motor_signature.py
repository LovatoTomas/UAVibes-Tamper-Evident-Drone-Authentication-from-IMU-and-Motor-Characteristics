"""
Per-motor speed signature from vibration data alone (no laser) — the
self-calibration idea of the 2026-09-19 pivot, and the "motor fallback"
identity signal.

For every single-motor steady hold (sweep-m{N} at 58 000 PWM, and the 7
steps of each stepped-m{N} ladder) the shaft fundamental is located as the
strongest spectral peak within ±15 % of the BASE model prediction (no
per-motor gain, so the search is not biased towards drone A's calibration),
and converted to a multiplicative gain = f_measured / f_base_predicted.
A drone's motor signature is then the vector of its four gains.
"""

import re
import numpy as np
import pandas as pd
from scipy.signal import welch

from pipeline.preprocessing import load_session, infer_fs, highpass
from pipeline.motor_model import pwm_to_hz

_HOLD_SINGLE = re.compile(r"^m([1-4])_hold(?:_(\d+))?$")
SEARCH = 0.15


def measure_hold(seg: pd.DataFrame, fs: float, motor: int) -> "dict | None":
    pwm = float(seg[f"motor.m{motor}"].mean())
    vbat = float(seg["vbat_v"].mean())
    f_pred = pwm_to_hz(pwm, vbat)                  # base model, motor_id=None
    if f_pred < 20:
        return None
    total = None
    for c in ("noise.acc.x", "noise.acc.y", "noise.acc.z",
              "noise.gyro.x", "noise.gyro.y", "noise.gyro.z"):
        x = seg[c].values
        f, p = welch(x - x.mean(), fs=fs, nperseg=min(2048, len(x)))
        p = p / np.median(p)                       # channel-normalised
        total = p if total is None else total + p
    band = (f > f_pred * (1 - SEARCH)) & (f < f_pred * (1 + SEARCH))
    if not band.any():
        return None
    k = np.argmax(np.where(band, total, -np.inf))
    # parabolic interpolation for sub-bin frequency
    if 0 < k < len(f) - 1:
        a, b, c = np.log(total[k - 1:k + 2])
        off = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        f_meas = f[k] + off * (f[1] - f[0])
    else:
        f_meas = f[k]
    prom_db = 10 * np.log10(total[k] / np.median(total[band]))
    return dict(pwm=pwm, vbat=vbat, f_pred=f_pred, f_meas=float(f_meas),
                gain=float(f_meas / f_pred), prom_db=float(prom_db))


def signatures(recordings) -> pd.DataFrame:
    """One row per single-motor hold: drone, kind, session, motor, pwm, gain..."""
    rows = []
    for rec in recordings:
        if not re.match(r"^(sweep|stepped)-m[1-4]$", rec["purpose"]):
            continue
        df = load_session(rec["path"])
        fs = infer_fs(df)
        hp = highpass(df, fs)
        for ph in hp["phase"].unique():
            m = _HOLD_SINGLE.match(str(ph))
            if not m:
                continue
            motor = int(m.group(1))
            seg = hp[hp["phase"] == ph]
            if len(seg) < 2000:
                continue
            r = measure_hold(seg, fs, motor)
            if r:
                rows.append(dict(drone=rec["drone"], kind=rec["kind"],
                                 session=rec["session"], day=rec["day"],
                                 purpose=rec["purpose"], motor=motor, **r))
    return pd.DataFrame(rows)


def classify_sessions(sig: pd.DataFrame) -> dict:
    """E7: session-level LOSO / held-out / controls on per-motor gain vectors."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    s = sig.copy()
    s["step"] = s.pwm.round(-3).astype(int)
    s["sid"] = s.drone + "-" + s.kind + "-" + s.session.astype(str)
    variants = {
        "gains_58k_only": (s[s.purpose.str.startswith("sweep")], ["motor"]),
        "gains_stepped": (s[s.purpose.str.startswith("stepped")], ["motor", "step"]),
        "gains_stepped_no_M1": (s[s.purpose.str.startswith("stepped") & (s.motor != 1)], ["motor", "step"]),
    }
    res = {}
    for name, (sel, keys) in variants.items():
        M = sel.pivot_table(index=["sid", "drone", "kind", "session", "day"], columns=keys,
                            values="gain", aggfunc="mean").dropna(axis=1)
        idx = M.index.to_frame(index=False)
        X = M.values
        en = (idx.kind == "enroll").values

        def fit_pred(tr, te, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X[tr], y[tr])
            return m.predict(X[te])

        r = {}
        for task, y, mask in [
            ("drone_A_vs_B", (idx.drone == "B").values.astype(int), en),
            ("control_day_A", (idx.day == "2026-09-26").values.astype(int), en & (idx.drone == "A").values),
            ("control_null_B", idx.session.isin([2, 4]).values.astype(int), en & (idx.drone == "B").values),
        ]:
            ok = []
            for k in sorted(idx.session[mask].unique()):
                te = mask & (idx.session == k).values
                tr = mask & ~te
                if len(set(y[tr])) < 2:
                    continue
                ok += list(fit_pred(tr, te, y) == y[te])
            r[task] = f"{sum(ok)}/{len(ok)}"
        te = (idx.kind == "test").values
        yd = (idx.drone == "B").values.astype(int)
        r["heldout_test"] = f"{int((fit_pred(en, te, yd) == yd[te]).sum())}/{int(te.sum())}"
        res[name] = r
    return res


if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.dataset import iter_recordings, ROOT
    out = ROOT / "analysis" / "results"
    out.mkdir(parents=True, exist_ok=True)
    from analysis.experiments import AB_CUTOFF
    sig = signatures(r for r in iter_recordings() if r["drone"] in ("A", "B") and r["day"] <= AB_CUTOFF)
    sig.to_csv(out / "motor_signatures.csv", index=False)
    (out / "E7_motor_signature.json").write_text(json.dumps(classify_sessions(sig), indent=1))
