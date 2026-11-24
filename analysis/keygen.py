"""
Key generation from vibrations: drift handling and motor-transplant attack
(2026-10-03).

Scheme simulated (fuzzy commitment over masked bits, the simplest secure
sketch used for PUFs):
  - one bit per feature: value above / below a fixed POPULATION threshold
    (median of the drones' first-day enrollment, a system parameter);
  - the helper data stored (publicly) on the drone selects the N_BITS
    features farthest from their threshold, in units of the population
    within-drone spread (dark-bit masking) and stores codeword(K) XOR bits;
  - at boot the drone re-measures, recomputes those bits, and the error
    correction recovers K if at most t*N_BITS bits differ.

Drift policies (the key K never changes; only the public helper does):
  fixed     helper computed once at enrollment;
  refresh   after every SUCCESSFUL boot, the helper is recomputed from that
            boot's measurement (Kusters & Willems 2021: sequential helper
            update after successful reconstruction). Because K is recovered
            first, the new helper still encodes the same K;
  refresh3  as refresh, but the reference is the median of the last three
            successful boots (less sensitive to one noisy boot).
Security of the update: a helper can only be rewritten after a successful
reconstruction, so an impostor that is never accepted can never move it
(the template-poisoning attacks on self-updating biometrics, Biggio et al.
2012, need accepted samples). Measured below: impostor acceptance against
the helper state at every moment of the genuine drone's history.

Feature sets: physical (speed gain + broadband floor per motor/step), and
chassis-band spectra of the 58k single-motor holds (the part of the identity
that partly follows the frame, see scambio_motori_A_B.md §4.4).

Run:  python -m analysis.keygen
"""

import json
import glob

import numpy as np
import pandas as pd

from analysis.dataset import ROOT, load_dataset, cols
from analysis.experiments import SINGLE_58K, CHASSIS_MAX_HZ, _shape
from analysis.motor_features import load as load_motor_features
from analysis.motor_identity import session_vectors, _within_scale

OUT = ROOT / "analysis" / "results_keygen"
DRONES = ["A", "B", "C", "D"]
META = ["drone", "kind", "session", "day", "stage", "motors"]


# ── session start times (chronological order of boots) ─────────────────────

def session_starts() -> dict:
    out = {}
    for f in glob.glob(str(ROOT / "data" / "fixed_rig*" / "drone*" / "session_*.json")):
        m = json.load(open(f, encoding="utf-8"))
        ids = [a["attempt_id"] for r in m["recordings"] for a in r["attempts"] if a.get("converted_path")]
        if ids:
            out[(m["drone"], m.get("kind", "enroll"), int(m["session"]))] = min(ids)
    for f in glob.glob(str(ROOT / "data" / "fixed_rig_swap" / "s*_*" / "session.json")):
        m = json.load(open(f, encoding="utf-8"))
        ids = [a["attempt_id"] for r in m["recordings"] for a in r["attempts"] if a.get("converted_path")]
        if ids:
            out[(m["drone"], "swap", int(m["session"]))] = min(ids)
    return out


# ── feature matrices (one row per session) ──────────────────────────────────

def spectral_vectors(kind: str = "psd") -> pd.DataFrame:
    """Session mean of the chassis-band spectrum of each 58k single-motor hold."""
    ds = load_dataset(verbose=False)
    try:
        from analysis.swap import load_swap
        sw = load_swap()
        ds = pd.concat([ds, sw], ignore_index=True)
    except Exception:
        pass
    h = ds[ds.phase.isin(SINGLE_58K)].copy()
    band = [c for c in cols(h, "psd") if int(c.rsplit(":", 1)[1]) <= CHASSIS_MAX_HZ - 10]
    if kind == "shape":
        h, band = _shape(h, band)
    h["stage"] = h.get("stage", pd.Series(index=h.index, dtype=object)).fillna("")
    h["motors"] = h.get("motors", pd.Series(index=h.index, dtype=object)).fillna(h.drone)
    g = h.groupby(META + ["phase"])[band].mean()
    W = g.unstack("phase")
    W.columns = [f"{ph}:{c}" for c, ph in W.columns]
    return W.reset_index()


def feature_sets() -> "dict[str, pd.DataFrame]":
    mf = load_motor_features()
    phys_lad = session_vectors(mf, "gain_floor_ladder")
    phys58 = session_vectors(mf, "s58")
    spec = spectral_vectors("psd")
    shape = spectral_vectors("shape")
    for d in (phys_lad, phys58, spec, shape):
        d["session"] = d.session.astype(int)
        d["stage"] = d.stage.fillna("")
    key = META
    both58 = phys58.merge(spec, on=key, how="inner")
    return {"phys_ladder": phys_lad, "phys_58k": phys58, "spec_58k": spec,
            "shape_58k": shape, "phys+spec_58k": both58}


def fcols(df):
    return [c for c in df.columns if c not in META]


# ── 1. which features follow the frame, which the motors (swap day) ─────────

def attribution(df: pd.DataFrame) -> pd.Series:
    """0 = the feature stays with the FRAME, 1 = it follows the MOTORS.
    Uses the clean comparison of the swap day: A before, B before, frame A +
    B's motors (B's motors kept their behaviour, A's did not)."""
    sw = df[df.kind == "swap"]
    preA = sw[(sw.stage == "pre") & (sw.drone == "A")]
    preB = sw[(sw.stage == "pre") & (sw.drone == "B")]
    hyb = sw[(sw.stage == "swap") & (sw.drone == "A")]
    c = fcols(df)
    a, b, h = preA[c].mean(), preB[c].mean(), hyb[c].mean()
    noise = pd.concat([preA[c] - a, preB[c] - b, hyb[c] - h]).std()
    informative = (b - a).abs() > 3 * noise
    idx = ((h - a) / (b - a)).where(informative)
    return idx.clip(-0.5, 1.5)


# ── key machinery ───────────────────────────────────────────────────────────

class KeyModel:
    def __init__(self, thr: pd.Series, scale: pd.Series, n_bits: int):
        self.thr, self.scale, self.n = thr, scale, n_bits

    def helper(self, x: pd.Series) -> dict:
        margin = ((x - self.thr).abs() / self.scale).dropna()
        sel = margin.sort_values(ascending=False).index[: self.n]
        return dict(sel=list(sel), ref=(x[sel] > self.thr[sel]).values)

    def distance(self, h: dict, x: pd.Series) -> float:
        b = (x[h["sel"]] > self.thr[h["sel"]]).values
        ok = ~x[h["sel"]].isna().values
        return float((b[ok] != h["ref"][ok]).mean()) if ok.any() else np.nan


def simulate(df: pd.DataFrame, n_bits: int, t: float, policy: str, starts: dict) -> dict:
    c = fcols(df)
    df = df.copy()
    df["start"] = [starts.get((r.drone, r.kind, int(r.session)), "") for r in df.itertuples()]
    first = {d: df[(df.drone == d) & (df.kind == "enroll")].day.min() for d in DRONES}
    enr = df[(df.kind == "enroll") & (df.day.values == df.drone.map(first).values)]
    thr = enr.groupby("drone")[c].median().median()
    scale = _within_scale(enr, c)
    km = KeyModel(thr, scale, n_bits)
    res = {}
    for T in DRONES:
        e = enr[enr.drone == T]
        if e.empty:
            continue
        h = km.helper(e[c].median())
        boots = df[(df.drone == T) & (df.motors == T) & ~df.index.isin(e.index)].sort_values("start")
        history, states, recent = [], [h], [e[c].median()]
        for _, r in boots.iterrows():
            x = r[c].astype(float)
            dist = km.distance(h, x)
            ok = bool(dist <= t)
            history.append(dict(session=f"{r.kind[0]}{int(r.session)}", day=r.day, stage=r.stage,
                                dist=round(dist, 3), accepted=ok))
            if ok and policy != "fixed":
                recent.append(x)
                ref = x if policy == "refresh" else pd.concat(recent[-3:], axis=1).median(axis=1)
                h = km.helper(ref)
                states.append(h)
        # impostors: every session of the other drones, against EVERY helper state T went through
        imp = df[(df.drone != T) & (df.kind != "swap")]
        imp_d = np.array([[km.distance(s, r[c].astype(float)) for s in states] for _, r in imp.iterrows()])
        res[T] = dict(
            genuine=history,
            genuine_reject_rate=float(np.mean([not g["accepted"] for g in history])) if history else None,
            impostor_min_distance=float(np.nanmin(imp_d)) if imp_d.size else None,
            impostor_accept_rate=float(np.nanmean(imp_d <= t)) if imp_d.size else None,
            n_helper_states=len(states))
    return res


def transplant(df: pd.DataFrame, n_bits: int, t: float, starts: dict) -> pd.DataFrame:
    """A's and B's keys (enrolled on their first day, then refreshed by every
    genuine boot up to the swap day, including the same-day 'pre' boots)
    against each swap-day configuration."""
    c = fcols(df)
    df = df.copy()
    df["start"] = [starts.get((r.drone, r.kind, int(r.session)), "") for r in df.itertuples()]
    first = {d: df[(df.drone == d) & (df.kind == "enroll")].day.min() for d in DRONES}
    enr = df[(df.kind == "enroll") & (df.day.values == df.drone.map(first).values)]
    thr = enr.groupby("drone")[c].median().median()
    scale = _within_scale(enr, c)
    km = KeyModel(thr, scale, n_bits)
    rows = []
    for T in ("A", "B"):
        e = enr[enr.drone == T]
        h = km.helper(e[c].median())
        genuine = df[(df.drone == T) & (df.motors == T) & ~df.index.isin(e.index)
                     & ((df.kind != "swap") | (df.stage == "pre"))].sort_values("start")
        for _, r in genuine.iterrows():          # refresh chain up to the swap day
            x = r[c].astype(float)
            if km.distance(h, x) <= t:
                h = km.helper(x)
        for _, r in df[df.kind == "swap"].sort_values("start").iterrows():
            rows.append(dict(key_of=T, session=int(r.session), stage=r.stage, frame=r.drone, motors=r.motors,
                             dist=round(km.distance(h, r[c].astype(float)), 3)))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    starts = session_starts()
    sets = feature_sets()
    out = {"attribution": {}, "drift": {}, "transplant": {}}

    for name in ("phys_58k", "spec_58k", "shape_58k"):
        a = attribution(sets[name]).dropna()
        out["attribution"][name] = dict(
            n_informative=int(len(a)), median=float(a.median()),
            frac_follow_frame=float((a < 0.3).mean()), frac_follow_motors=float((a > 0.7).mean()))
        print(f"attribution {name}: {out['attribution'][name]}", flush=True)
    # frame-bound / motor-bound subsets of the spectral features
    a_spec = attribution(sets["spec_58k"])
    frame_cols = list(a_spec[a_spec < 0.3].index)
    motor_cols = list(a_spec[a_spec > 0.7].index)
    sets["spec_frame_bound"] = sets["spec_58k"][META + frame_cols]
    sets["spec_motor_bound"] = sets["spec_58k"][META + motor_cols]
    print(f"frame-bound spectral features: {len(frame_cols)}, motor-bound: {len(motor_cols)}", flush=True)

    for name, n in (("phys_ladder", 64), ("phys_58k", 16), ("spec_58k", 64), ("phys+spec_58k", 64)):
        out["drift"][name] = {}
        for t in (0.10, 0.15, 0.20, 0.25):
            for policy in ("fixed", "refresh", "refresh3"):
                r = simulate(sets[name], n, t, policy, starts)
                out["drift"][name][f"t{t:.2f}_{policy}"] = r
                frr = np.mean([v["genuine_reject_rate"] for v in r.values() if v["genuine_reject_rate"] is not None])
                far = np.mean([v["impostor_accept_rate"] for v in r.values() if v["impostor_accept_rate"] is not None])
                print(f"drift {name:15s} n={n} t={t:.2f} {policy:8s}: genuine rejected {frr:.2f}  "
                      f"impostor accepted {far:.3f}", flush=True)

    for name, n in (("phys_58k", 16), ("spec_58k", 64), ("spec_frame_bound", 64), ("spec_motor_bound", 64),
                    ("shape_58k", 64), ("phys+spec_58k", 64)):
        if len(fcols(sets[name])) < n:
            n = len(fcols(sets[name]))
        tr = transplant(sets[name], n, 0.20, starts)
        tr.to_csv(OUT / f"transplant_{name}.csv", index=False)
        piv = tr.groupby(["key_of", "stage", "frame", "motors"]).dist.agg(["min", "median", "max"]).round(2)
        out["transplant"][name] = piv.reset_index().to_dict(orient="records")
        print(f"\ntransplant {name} (n={n} bits; distance to the key, accepted if <= 0.20):\n{piv}", flush=True)

    (OUT / "keygen.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
