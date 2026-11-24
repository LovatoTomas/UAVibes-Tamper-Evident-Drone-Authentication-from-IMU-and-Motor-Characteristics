"""
Motor-identity study (2026-10-02): physical per-motor features, fusion with
the phase-conditioned spectral verifier, and a first feasibility check of
vibration-derived KEY generation.

Session-level evaluation (one decision per session = one boot), same rules
as the previous studies: splits by session, LOSO on enrollment, held-out test
sessions, cross-day (enrol on each drone's first day only), plus the motor-
swap sessions as tampering cases.

Scores (higher = less like the enrolled drone), all brought to a common scale
by NESTED normalisation: for every enrolled model, the genuine scores of its
own training sessions (inner leave-one-out) give a median / spread, and test
scores are expressed in those units — so one global threshold works for every
drone and scores of different methods can be averaged (fusion).

  phys_<set>   per-motor physical features (analysis/motor_features.py),
               mean clipped squared z-distance to the drone's enrollment
               median, in units of the population within-drone spread
  spec         phase-conditioned Ledoit-Wolf score on the chassis-band 58k
               single-motor spectrum (best one-class method of the 4-drone study)
  fused        mean of the normalised phys and spec scores

Key-generation check: per feature one bit = side of the population median,
"dark-bit" masking keeps only features whose enrolled value is at least
K_MASK within-spreads from the threshold (mask = public helper data).
Reported: reliable bits per drone, genuine bit-error rate on later boots,
and the distance of other drones' bits under the same helper data.

Run:  python -m analysis.motor_identity
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis.dataset import ROOT, load_dataset
from analysis.experiments import _eer, representations
from analysis.four_drones import _lw, _score, DRONES
from analysis.motor_features import load as load_motor_features, CHANNELS

OUT = ROOT / "analysis" / "results_motor"
CLIP = 25.0          # cap on a single feature's squared z (one misdetected line must not decide)
K_MASK = 2.0         # dark-bit masking: keep bits whose enrolled value is >= 2 spreads from threshold


# ── session-level physical feature vectors ──────────────────────────────────

FEATURE_SETS = {
    # name: (source, steps or None for all, feature prefixes)
    "gain_ladder": ("lad", None, ("gain",)),
    "gain_floor_ladder": ("lad", None, ("gain", "floor")),
    "all_ladder": ("lad", None, ("gain", "floor", "h1")),
    "boot_low_pwm": ("lad", (10000, 15000, 20000, 25000), ("gain", "floor")),
    "s58": ("s58", None, ("gain", "floor")),
}


def session_vectors(mf: pd.DataFrame, set_name: str) -> pd.DataFrame:
    src, steps, prefixes = FEATURE_SETS[set_name]
    d = mf.copy()
    d["src"] = np.where(d.purpose.str.startswith("sweep"), "s58", "lad")
    d = d[d.src == src]
    if steps:
        d = d[d.step.isin(steps)]
    cols = (["gain"] if "gain" in prefixes else []) +         [f"{p}_{c}" for p in prefixes if p != "gain" for c in CHANNELS if f"{p}_{c}" in d.columns]
    long = d.melt(id_vars=["drone", "kind", "session", "day", "stage", "motors", "motor", "step"],
                  value_vars=cols, var_name="feat")
    long["name"] = "m" + long.motor.astype(str) + "_" + long.step.astype(str) + "_" + long.feat
    long["stage"] = long.stage.fillna("")
    long["motors"] = long.motors.fillna(long.drone)
    W = long.pivot_table(index=["drone", "kind", "session", "day", "stage", "motors"],
                         columns="name", values="value", aggfunc="mean")
    W = W.loc[:, W.notna().mean() > 0.9]
    W.index = W.index.set_names(["drone", "kind", "session", "day", "stage", "motors"])
    return W.reset_index()


def _within_scale(train: pd.DataFrame, cols) -> pd.Series:
    """Pooled within-drone spread per feature (robust: MAD of session values
    around each drone's median) — a system-level noise model."""
    dev = train.groupby("drone")[cols].transform(lambda x: x - x.median())
    s = 1.4826 * dev.abs().median()
    return s.where(s > 0, dev.abs().mean()).fillna(1.0).clip(lower=1e-6)


def phys_scorer(cols):
    def score(train_T: pd.DataFrame, test: pd.DataFrame, scale: pd.Series) -> np.ndarray:
        mu = train_T[cols].median()
        z2 = ((test[cols] - mu) / scale) ** 2
        return z2.clip(upper=CLIP).mean(axis=1, skipna=True).values
    return score


# ── spectral session scores ─────────────────────────────────────────────────

class SpecScorer:
    """Mean phase-conditioned LW score of a session's recordings."""

    def __init__(self, ds: pd.DataFrame, rep="shape_chassis_single58k"):
        # the swap sessions are scored too (tampering cases): add their windows
        try:
            from analysis.swap import load_swap
            sw = load_swap()
            if not sw.empty:
                ds = pd.concat([ds, sw[ds.columns.intersection(sw.columns)]], ignore_index=True)
        except Exception:
            pass
        base, ext = representations(ds), representations(ds, extended=True)
        sub, self.feats = {**base, **ext}[rep]
        self.sub = sub
        self.key = sub.drone + "|" + sub.kind + "|" + sub.session.astype(str)

    def __call__(self, train_keys, test_keys) -> np.ndarray:
        tr = self.sub[self.key.isin(train_keys)]
        out = []
        for k in test_keys:
            te = self.sub[self.key == k]
            if te.empty or tr.empty:
                out.append(np.nan)
                continue
            g = _score(tr, te, self.feats, _lw, True)
            out.append(float(g.s.mean()) if len(g) else np.nan)
        return np.array(out)


def _key(df: pd.DataFrame) -> pd.Series:
    return df.drone + "|" + df.kind + "|" + df.session.astype(str)


# ── nested-normalised evaluation ────────────────────────────────────────────

def _norm_params(genuine: np.ndarray) -> "tuple[float, float]":
    g = genuine[~np.isnan(genuine)]
    if len(g) < 2:
        return (float(g[0]) if len(g) else 0.0), 1.0
    med = float(np.median(g))
    spread = float(np.median(np.abs(g - med)) * 1.4826) or float(np.std(g)) or 1.0
    return med, spread


def evaluate(V: pd.DataFrame, cols, spec: "SpecScorer | None", scenario: str) -> pd.DataFrame:
    """Rows: target, test session, is_genuine, phys / spec / fused normalised scores."""
    first = {d: V[V.drone == d].day.min() for d in DRONES}
    en = V[V.kind == "enroll"]
    rows = []
    for T in DRONES:
        if scenario == "loso":
            folds = [(en[(en.drone == T) & (en.session != k)],
                      pd.concat([en[(en.drone == T) & (en.session == k)], en[(en.drone != T) & (en.session == k)]]))
                     for k in sorted(en.session.unique())]
        elif scenario == "heldout":
            folds = [(en[en.drone == T], V[V.kind == "test"])]
        elif scenario == "crossday":
            tr = en[(en.drone == T) & (en.day == first[T])]
            te = pd.concat([V[(V.drone == T) & (V.day != first[T]) & (V.kind != "swap")],
                            V[(V.kind == "test") & (V.drone != T)]])
            folds = [(tr, te)]
        elif scenario == "swap":
            if T not in ("A", "B"):
                continue
            folds = [(en[en.drone == T], V[V.kind == "swap"])]
        else:
            raise ValueError(scenario)
        for tr, te in folds:
            if te.empty or len(tr) < 2:
                continue
            # population noise model from every drone's training sessions of this fold
            pool = en[~_key(en).isin(_key(te))] if scenario == "loso" else en
            if scenario == "crossday":
                pool = en[en.day.values == en.drone.map(first).values]
            scale = _within_scale(pool, cols)
            ps = phys_scorer(cols)
            # inner leave-one-out genuine scores -> normalisation
            inner_p, inner_s = [], []
            for j in range(len(tr)):
                rest, one = tr.drop(tr.index[j]), tr.iloc[[j]]
                inner_p.append(ps(rest, one, scale)[0])
                if spec is not None:
                    inner_s.append(spec(set(_key(rest)), [_key(one).iloc[0]])[0])
            mp, sp = _norm_params(np.array(inner_p))
            p = (ps(tr, te, scale) - mp) / sp
            r = te[["drone", "kind", "session", "day", "stage", "motors"]].assign(
                target=T, genuine=(te.drone == T).values & (te.motors == T).values, phys=p)
            if spec is not None:
                ms, ss = _norm_params(np.array(inner_s))
                s = (spec(set(_key(tr)), list(_key(te))) - ms) / ss
                r = r.assign(spec=s, fused=np.nanmean(np.vstack([p, s]), axis=0))
            rows.append(r)
    return pd.concat(rows, ignore_index=True)


def summarise(R: pd.DataFrame) -> dict:
    out = {}
    for m in [c for c in ("phys", "spec", "fused") if c in R]:
        x = R.dropna(subset=[m])
        y = (~x.genuine).astype(int).values
        if len(set(y)) < 2:
            continue
        o = dict(auc=float(roc_auc_score(y, x[m])), eer=float(_eer(y, x[m].values)))
        o["per_target_auc"] = {T: float(roc_auc_score((~g.genuine).astype(int), g[m]))
                               for T, g in x.groupby("target") if g.genuine.nunique() == 2}
        out[m] = o
    return out


# ── key-generation feasibility ──────────────────────────────────────────────

def keygen(V: pd.DataFrame, cols) -> dict:
    """Dark-bit-masked 1-bit-per-feature keys. Enrollment = first-day enroll
    sessions (the realistic case); later boots = every later session."""
    first = {d: V[V.drone == d].day.min() for d in DRONES}
    en = V[V.kind == "enroll"]
    enr = en[en.day.values == en.drone.map(first).values]
    thr = enr.groupby("drone")[cols].median().median()           # population threshold
    scale = _within_scale(enr, cols)
    res = {}
    for T in DRONES:
        mu = enr[enr.drone == T][cols].median()
        mask = ((mu - thr).abs() >= K_MASK * scale).values
        key = (mu.values > thr.values)[mask]
        later = V[(V.drone == T) & (V.motors == T) & (V.day != first[T]) & (V.kind != "swap")]
        same = en[(en.drone == T) & (en.day == first[T])]

        def ber(rows):
            if rows.empty or mask.sum() == 0:
                return []
            b = rows[cols].values[:, mask] > thr.values[mask]
            return list((b != key).mean(axis=1))
        imp = {}
        for O in DRONES:
            if O == T:
                continue
            oth = V[(V.drone == O) & (V.kind != "swap")]
            imp[O] = float(np.mean(ber(oth))) if len(oth) else None
        swap = V[(V.kind == "swap") & ((V.drone == T) | (V.motors == T))]
        res[T] = dict(
            n_features=len(cols), reliable_bits=int(mask.sum()),
            ber_same_day_sessions=[round(x, 3) for x in ber(same)],
            ber_later_days=[round(x, 3) for x in ber(later)],
            impostor_mean_distance=imp,
            swap_sessions={f"w{int(r.session):02d} {r.stage} frame{r.drone}+mot{r.motors}": round(b, 3)
                           for (_, r), b in zip(swap.iterrows(), ber(swap))} if len(swap) else {},
        )
    return res


def main():
    mf = load_motor_features()
    ds = load_dataset(verbose=False)
    spec = SpecScorer(ds)
    results = {"auth": {}, "keygen": {}}
    for set_name in FEATURE_SETS:
        V = session_vectors(mf, set_name)
        cols = [c for c in V.columns if c[:1] == "m" and c[1:2].isdigit() and c[2:3] == "_"]
        V[cols] = V[cols].astype(float)
        use_spec = spec if set_name in ("gain_floor_ladder", "s58") else None
        r = {}
        for scen in ("loso", "heldout", "crossday", "swap"):
            if scen == "swap" and set_name != "s58":
                continue       # swap sessions only contain the 58k single-motor stages
            R = evaluate(V, cols, use_spec, scen)
            R.to_csv(OUT / f"scores_{set_name}_{scen}.csv", index=False)
            r[scen] = summarise(R) if scen != "swap" else \
                R.groupby(["target", "stage", "drone", "motors"])[[c for c in ("phys", "spec", "fused") if c in R]] \
                 .median().round(2).reset_index().to_dict(orient="records")
            print(set_name, scen, {k: (round(v["auc"], 3), round(v["eer"], 3)) for k, v in r[scen].items()}
                  if scen != "swap" else "", flush=True)
        results["auth"][set_name] = r
        results["keygen"][set_name] = keygen(V, cols)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "motor_identity.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    return results


if __name__ == "__main__":
    main()
