"""
Four-drone identification study (2026-09-28) — A, B, C, D.

Extends the A-vs-B study (experiments.py, kept unchanged so its results stay
reproducible) to four nominally identical Crazyflie 2.1. Same evaluation
rules — splits by SESSION, never by window; every result paired with
controls — plus two things only possible with more drones / more days:

  X1  cross-day coherence: is a drone's signature on day 2 as close to its
      own day-1 signature as two same-day sessions are to each other, and
      still further from every other drone? (A 09-23/09-26, C and D
      09-27/09-28, B 09-20 vs its test session of 09-28.)
  CROSS-DAY split: train on each drone's first day only, test on everything
      recorded on a later day — the realistic "enrol once, verify later".
  C vs D: recorded on the SAME two days, so their separation carries no
      drone=day confound at all.

Results go to analysis/results_4drones/; the report
paper/experiments/identificazione_4_droni.md interprets them.
Run:  python -m analysis.four_drones [STEP ...]
"""

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.feature_selection import f_classif
from sklearn.covariance import LedoitWolf

from analysis.dataset import load_dataset, cols, ROOT
from analysis.experiments import (
    representations, classifiers, one_class_scorers, AE, _eer, _shape, loso,
    e1_static, SINGLE_58K, CHASSIS_MAX_HZ, RNG,
)

warnings.filterwarnings("ignore")
OUT = ROOT / "analysis" / "results_4drones"
OUT.mkdir(parents=True, exist_ok=True)
DRONES = ["A", "B", "C", "D"]


def _save(tag, obj):
    (OUT / f"{tag}.json").write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")


def first_day(ds: pd.DataFrame) -> "dict[str, str]":
    return {d: ds[ds.drone == d].day.min() for d in DRONES}


def selected_reps(ds):
    base, ext = representations(ds), representations(ds, extended=True)
    return base, ext


# ── X1: cross-day coherence ──────────────────────────────────────────────────

def x1_crossday_coherence(reps: dict) -> dict:
    """Recording-mean vectors (same stage compared only with same stage),
    standardised over everything. Per drone: mean distance between two of
    its recordings from the SAME day, from DIFFERENT days, and from it to
    other drones. drift = crossday/sameday (1 = no day effect);
    margin = between/crossday (>1 = still closer to itself after a day).
    NN-ID: each later-day recording's nearest neighbour among ALL first-day
    recordings of the same stage (all drones) — fraction that is its own
    drone (chance 0.25)."""
    res = {}
    for name, (sub, feats) in reps.items():
        mv = sub.groupby("rec_id").agg(drone=("drone", "first"), purpose=("purpose", "first"),
                                        day=("day", "first"), kind=("kind", "first"))
        mv = mv[mv.purpose != "unseen-pwm"]
        X = StandardScaler().fit_transform(sub.groupby("rec_id")[feats].mean().loc[mv.index].values)
        P = np.array([(i, j) for idx in mv.groupby("purpose").indices.values()
                      for i in idx for j in idx if i < j])
        d = np.linalg.norm(X[P[:, 0]] - X[P[:, 1]], axis=1)
        dr, dy = mv.drone.values, mv.day.values
        fd = {k: mv[mv.drone == k].day.min() for k in DRONES}
        out = {}
        for k in DRONES:
            kk = (dr[P[:, 0]] == k) & (dr[P[:, 1]] == k)
            k_any = (dr[P[:, 0]] == k) | (dr[P[:, 1]] == k)
            same = dy[P[:, 0]] == dy[P[:, 1]]
            between = k_any & ~kk
            r = dict(sameday=float(d[kk & same].mean()),
                     crossday=float(d[kk & ~same].mean()) if (kk & ~same).any() else None,
                     between=float(d[between].mean()))
            if r["crossday"]:
                r["drift"] = r["crossday"] / r["sameday"]
                r["margin"] = r["between"] / r["crossday"]
            # nearest-neighbour identification of later-day recordings
            later = np.where((dr == k) & (dy != fd[k]))[0]
            ref = np.where(np.array([dy[i] == fd[dr[i]] for i in range(len(dr))]))[0]
            hits = []
            for i in later:
                cand = [j for j in ref if mv.purpose.values[j] == mv.purpose.values[i]]
                if cand:
                    j = cand[int(np.argmin(np.linalg.norm(X[cand] - X[i], axis=1)))]
                    hits.append(dr[j] == k)
            if hits:
                r["nn_id_later_day"] = float(np.mean(hits))
                r["n_later"] = len(hits)
            out[k] = r
        res[name] = out
        print(f"  {name:28s} " + " ".join(
            f"{k}: drift={o.get('drift', float('nan')):.2f} margin={o.get('margin', float('nan')):.2f} "
            f"nn={o.get('nn_id_later_day', float('nan')):.2f}" for k, o in out.items()), flush=True)
    return res


# ── E2: supervised 4-class ───────────────────────────────────────────────────

def _vote(df, key):
    g = df.groupby(key).agg(y=("drone", "first"), p=("pred", lambda s: s.value_counts().idxmax()))
    return float((g.y == g.p).mean()), g


def _summ(te, pred):
    p = te[["rec_id", "session_id", "drone", "purpose", "day"]].assign(pred=pred)
    rec_acc, g = _vote(p, "rec_id")
    ses_acc, _ = _vote(p, "session_id")
    out = dict(window_bal_acc=float(balanced_accuracy_score(p.drone, p.pred)),
               recording_acc=rec_acc, session_acc=ses_acc, n_rec=int(len(g)))
    out["confusion_recordings"] = confusion_matrix(g.y, g.p, labels=DRONES).tolist()
    uns = p[p.purpose == "unseen-pwm"]
    if len(uns):
        out["unseen_pwm_recording_acc"] = _vote(uns, "rec_id")[0]
        out["unseen_pwm_window_acc"] = float((uns.pred == uns.drone).mean())
    return out


def e2_multiclass(ds, reps: dict, models=None) -> dict:
    fd = first_day(ds)
    res = {}
    for name, (sub, feats) in reps.items():
        en, ts = sub[sub.kind == "enroll"], sub[sub.kind == "test"]
        res[name] = {}
        for mname, make in classifiers().items():
            if models and mname not in models:
                continue
            if mname == "svm_rbf" and len(en) > 15000:
                continue   # kernel SVM on ~40k windows: hours per config; linear models cover these sets
            t0 = time.time()
            r = {}
            # LOSO: fold k holds out session k of every drone
            preds = []
            for k in sorted(en.session.unique()):
                tr, te = en[en.session != k], en[en.session == k]
                m = make().fit(tr[feats].values, tr.drone.values)
                preds.append(te.assign(pred=m.predict(te[feats].values)))
            p = pd.concat(preds)
            r["loso"] = _summ(p, p.pred.values)
            # held-out test sessions
            m = make().fit(en[feats].values, en.drone.values)
            r["heldout_test"] = _summ(ts, m.predict(ts[feats].values))
            # cross-day: train on each drone's first day (enrollment only)
            first = sub.day.values == sub.drone.map(fd).values
            tr = sub[first & (sub.kind == "enroll").values]
            te = sub[~first]
            m = make().fit(tr[feats].values, tr.drone.values)
            r["crossday"] = _summ(te, m.predict(te[feats].values))
            # binary controls (session-level LOSO, same code as the A/B study)
            ctl = {}
            for dname in DRONES:
                d = en[en.drone == dname]
                if d.day.nunique() > 1:
                    ctl[f"day_{dname}"] = loso(d.assign(y=(d.day != fd[dname]).astype(int), fold=d.session),
                                               feats, make)["recording_acc"]
            b = en[en.drone == "B"]
            ctl["null_B"] = loso(b.assign(y=b.session.isin([2, 4]).astype(int), fold=b.session),
                                 feats, make)["recording_acc"]
            cd = en[en.drone.isin(["C", "D"])]
            ctl["C_vs_D_sameday"] = loso(cd.assign(y=(cd.drone == "D").astype(int), fold=cd.session),
                                         feats, make)["recording_acc"]
            r["controls"] = ctl
            r["seconds"] = round(time.time() - t0, 1)
            res[name][mname] = r
            print(f"  {name:28s} {mname:8s} loso rec={r['loso']['recording_acc']:.2f} "
                  f"ses={r['loso']['session_acc']:.2f} test={r['heldout_test']['recording_acc']:.2f} "
                  f"xday={r['crossday']['recording_acc']:.2f} CvsD={ctl['C_vs_D_sameday']:.2f} "
                  f"day={[round(v, 2) for k, v in ctl.items() if k.startswith('day')]} "
                  f"null={ctl['null_B']:.2f} ({r['seconds']}s)", flush=True)
    return res


# ── E4 / E4b: one-class authentication, 4 drones ────────────────────────────

def _knn5(Xtr, Xte):
    sc = StandardScaler().fit(Xtr)
    nn = NearestNeighbors(n_neighbors=min(5, len(Xtr))).fit(sc.transform(Xtr))
    return nn.kneighbors(sc.transform(Xte))[0].mean(1)


def _lw(Xtr, Xte):
    sc = StandardScaler().fit(Xtr)
    return np.sqrt(LedoitWolf().fit(sc.transform(Xtr)).mahalanobis(sc.transform(Xte)))


def _score(tr, te, feats, scorer, conditioned):
    """Per-recording mean anomaly score of te against enrollment tr."""
    if not conditioned:
        s = scorer(tr[feats].values, te[feats].values)
        r = te[["rec_id", "drone"]].assign(s=s)
    else:
        parts = []
        for ph, te_ph in te.groupby("phase"):
            tr_ph = tr[tr.phase == ph]
            if len(tr_ph) < 10:
                continue
            parts.append(te_ph[["rec_id", "drone"]].assign(s=scorer(tr_ph[feats].values, te_ph[feats].values)))
        if not parts:
            return pd.DataFrame(columns=["s", "drone"])
        r = pd.concat(parts)
    return r.groupby("rec_id").agg(s=("s", "mean"), drone=("drone", "first"))


def _auc(g, target):
    y = (g.drone != target).astype(int).values
    if len(set(y)) < 2:
        return None
    return dict(auc=float(roc_auc_score(y, g.s.values)), eer=_eer(y, g.s.values))


def oneclass_eval(ds, sub, feats, scorer, conditioned, loso_only=False) -> dict:
    fd = first_day(ds)
    en, ts = sub[sub.kind == "enroll"], sub[sub.kind == "test"]
    out = {}
    pooled = {"loso": [], "heldout": [], "crossday": []}
    for T in DRONES:
        o = {}
        gs = []
        for k in sorted(en.session.unique()):
            tr = en[(en.drone == T) & (en.session != k)]
            te = pd.concat([en[(en.drone == T) & (en.session == k)], en[en.drone != T]])
            gs.append(_score(tr, te, feats, scorer, conditioned))
        g = pd.concat(gs)
        o["loso"] = _auc(g, T)
        o["loso_per_impostor"] = {I: _auc(g[g.drone.isin([T, I])], T)["auc"] for I in DRONES if I != T}
        pooled["loso"].append(g.assign(t=T))
        if not loso_only:
            g = _score(en[en.drone == T], ts, feats, scorer, conditioned)
            o["heldout"] = _auc(g, T)
            pooled["heldout"].append(g.assign(t=T))
            # cross-day: enrol on T's first-day enrollment sessions only;
            # genuine = T's later-day recordings, impostors = other drones' test sessions
            tr = en[(en.drone == T) & (en.day == fd[T])]
            te = pd.concat([sub[(sub.drone == T) & (sub.day != fd[T])], ts[ts.drone != T]])
            if (te.drone == T).any():
                g = _score(tr, te, feats, scorer, conditioned)
                o["crossday"] = _auc(g, T)
                pooled["crossday"].append(g.assign(t=T))
            # day control: first-day session held out vs later-day sessions of T
            d1 = en[(en.drone == T) & (en.day == fd[T])]
            later = sub[(sub.drone == T) & (sub.day != fd[T])]
            if len(later) and d1.session.nunique() > 1:
                gs = []
                for k in sorted(d1.session.unique()):
                    g = _score(d1[d1.session != k], pd.concat([d1[d1.session == k], later]),
                               feats, scorer, conditioned)
                    lab = sub.groupby("rec_id").day.first()
                    gs.append(g.assign(y=(lab.loc[g.index] != fd[T]).astype(int).values))
                g = pd.concat(gs)
                o["day_control"] = float(roc_auc_score(g.y, g.s))
        out[T] = o
    # a single global threshold is what a deployed verifier would use; per-target
    # AUCs above hide score-scale differences between drones -> also pool them
    for key, parts in pooled.items():
        if parts:
            g = pd.concat(parts)
            y = (g.drone != g.t).astype(int).values
            out[f"pooled_{key}"] = dict(auc=float(roc_auc_score(y, g.s)), eer=_eer(y, g.s.values))
    return out


def e4_oneclass(ds, reps: dict, conditioned: bool, scorers: dict) -> dict:
    res = {}
    for name, (sub, feats) in reps.items():
        res[name] = {}
        for sname, sc in scorers.items():
            t0 = time.time()
            r = oneclass_eval(ds, sub, feats, sc, conditioned, loso_only=sname.startswith("ae_"))
            res[name][sname] = r
            aucs = " ".join(f"{T}={r[T]['loso']['auc']:.2f}" for T in DRONES)
            extra = ""
            if "heldout" in r["A"]:
                extra = (f" test={np.mean([r[T]['heldout']['auc'] for T in DRONES]):.2f}"
                         f" xday={np.mean([r[T]['crossday']['auc'] for T in DRONES if 'crossday' in r[T]]):.2f}"
                         f" dayctl={[round(r[T].get('day_control', np.nan), 2) for T in DRONES]}")
            print(f"  {name:28s} {sname:18s} LOSO {aucs} pooledEER={r['pooled_loso']['eer']:.3f}{extra} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return res


# ── E5: AE / PCA latent, 4-class ─────────────────────────────────────────────

def e5_latent(reps: dict) -> dict:
    res = {}
    for name, (sub, feats) in reps.items():
        en = sub[sub.kind == "enroll"]
        res[name] = {}
        for kind in ("ae8", "pca8"):
            preds = []
            for k in sorted(en.session.unique()):
                tr, te = en[en.session != k], en[en.session == k]
                if kind == "ae8":
                    f = AE(k=8).fit(tr[feats].values).latent
                else:
                    f = make_pipeline(StandardScaler(), PCA(8, random_state=RNG)).fit(tr[feats].values).transform
                clf = LogisticRegression(max_iter=3000).fit(f(tr[feats].values), tr.drone.values)
                preds.append(te.assign(pred=clf.predict(f(te[feats].values))))
            p = pd.concat(preds)
            s = _summ(p, p.pred.values)
            res[name][kind] = {k: s[k] for k in ("window_bal_acc", "recording_acc", "session_acc")}
            print(f"  {name:28s} {kind}: rec={s['recording_acc']:.2f} win={s['window_bal_acc']:.2f}", flush=True)
    return res


# ── E6: where in frequency the drones differ ────────────────────────────────

def e6_profile(ds) -> dict:
    en = ds[(ds.kind == "enroll") & ds.phase.isin(SINGLE_58K)]
    f = cols(ds, "psd")
    F, _ = f_classif(en[f].values, en.drone.values)
    return {"psd_single58k_F_by_bin_4class": {k: float(v) for k, v in zip(f, F)}}


# ── E7: motor-speed signature, 4 drones ─────────────────────────────────────

def e7_motor(sig: pd.DataFrame, fd: dict) -> dict:
    s = sig.copy()
    s["step"] = s.pwm.round(-3).astype(int)
    s["sid"] = s.drone + "-" + s.kind + "-" + s.session.astype(str)
    variants = {"gains_58k_only": (s[s.purpose.str.startswith("sweep")], ["motor"]),
                "gains_stepped": (s[s.purpose.str.startswith("stepped")], ["motor", "step"])}
    res = {}
    for name, (sel, keys) in variants.items():
        M = sel.pivot_table(index=["sid", "drone", "kind", "session", "day"], columns=keys,
                            values="gain", aggfunc="mean").dropna(axis=1)
        idx = M.index.to_frame(index=False)
        X, y = M.values, idx.drone.values
        en = (idx.kind == "enroll").values

        def fit_pred(tr, te):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[tr], y[tr])
            return m.predict(X[te])
        ok = []
        for k in sorted(idx.session[en].unique()):
            te = en & (idx.session == k).values
            ok += list(fit_pred(en & ~te, te) == y[te])
        te = (idx.kind == "test").values
        first = (idx.day.values == idx.drone.map(fd).values)
        r = {"loso_sessions": f"{sum(ok)}/{len(ok)}",
             "heldout_test": f"{int((fit_pred(en, te) == y[te]).sum())}/{int(te.sum())}"}
        tr, tx = en & first, ~first
        pr = fit_pred(tr, tx)
        r["crossday"] = f"{int((pr == y[tx]).sum())}/{int(tx.sum())}"
        r["crossday_errors"] = [f"{a}->{b}" for a, b in zip(idx.sid[tx], pr) if a.split("-")[0] != b]
        res[name] = r
        print(f"  {name}: {r}", flush=True)
    return res


# ── E9: which single stage is enough (4 drones) ─────────────────────────────

def e9_per_stage(ds) -> dict:
    holds = ds[ds.phase.str.contains("hold", na=False) & (ds.purpose != "unseen-pwm")]
    psd = cols(ds, "psd")
    band = [c for c in psd if int(c.rsplit(":", 1)[1]) <= CHASSIS_MAX_HZ - 10]
    res = {}
    for pur in sorted(holds.purpose.unique()):
        h = holds[holds.purpose == pur]
        hs, fs = _shape(h, band)
        row = {}
        for name, (frame, f) in {"psd": (h, psd), "shape_chassis": (hs, fs)}.items():
            r = oneclass_eval(ds, frame, f, _knn5, conditioned=True)
            row[name] = dict(loso=float(np.mean([r[T]["loso"]["auc"] for T in DRONES])),
                             crossday=float(np.mean([r[T]["crossday"]["auc"] for T in DRONES if "crossday" in r[T]])),
                             pooled_loso_eer=r["pooled_loso"]["eer"])
        res[pur] = row
        print(f"  {pur:16s} psd loso={row['psd']['loso']:.3f} xday={row['psd']['crossday']:.3f} | "
              f"shape loso={row['shape_chassis']['loso']:.3f} xday={row['shape_chassis']['crossday']:.3f}", flush=True)
    return res


# ── E11: template update (tracking drift) ───────────────────────────────────

def e11_template_update(ds, reps: dict) -> dict:
    """The design principle says a drone stays authenticated along a
    TRAJECTORY, not at a fixed point. With C's and D's motors drifting
    between days (X1, E7), test the simplest trajectory mechanism: after a
    successful later-day verification, add that session to the enrollment.
    For each drone with >= 2 later-day sessions: enrol on day 1 (+ one
    later-day session, rotated), genuine = the OTHER later-day sessions,
    impostors = other drones' test sessions. Compared on the identical
    genuine/impostor recordings with day-1-only enrollment."""
    fd = first_day(ds)
    res = {}
    for name, (sub, feats) in reps.items():
        ts = sub[sub.kind == "test"]
        res[name] = {}
        for T in DRONES:
            later = sub[(sub.drone == T) & (sub.day != fd[T])]
            later_ids = sorted(later.session_id.unique())
            if len(later_ids) < 2:
                continue
            d1 = sub[(sub.drone == T) & (sub.day == fd[T]) & (sub.kind == "enroll")]
            base_g, upd_g = [], []
            for add in later_ids:
                gen = later[later.session_id != add]
                te = pd.concat([gen, ts[ts.drone != T]])
                base_g.append(_score(d1, te, feats, _knn5, True))
                upd_g.append(_score(pd.concat([d1, later[later.session_id == add]]), te, feats, _knn5, True))
            b, u = pd.concat(base_g), pd.concat(upd_g)
            res[name][T] = dict(day1_only=_auc(b, T), updated=_auc(u, T), n_later_sessions=len(later_ids))
        print(f"  {name:28s} " + " ".join(
            f"{T}: {r['day1_only']['auc']:.2f}->{r['updated']['auc']:.2f}" for T, r in res[name].items()), flush=True)
    return res


def _signatures() -> pd.DataFrame:
    """Per-motor gains of every single-motor hold (motor_signature.py), cached
    in results_4drones/motor_signatures.csv."""
    from analysis.dataset import iter_recordings
    from analysis.motor_signature import signatures
    f = OUT / "motor_signatures.csv"
    if not f.exists():
        signatures(iter_recordings()).to_csv(f, index=False)
    return pd.read_csv(f)


def main(which=None):
    ds = load_dataset(verbose=False)
    fd = first_day(ds)
    base, ext = selected_reps(ds)
    allreps = {**base, **ext}
    key = {k: allreps[k] for k in ("comb120_allholds", "psd_allholds", "psd_chassis_single58k",
                                   "shape_chassis_single58k", "shape_allholds")}
    oc = {k: allreps[k] for k in ("comb120_allholds", "psd_chassis_single58k", "shape_chassis_single58k")}
    sc = one_class_scorers()
    steps = {
        "X1_crossday_coherence": lambda: x1_crossday_coherence(allreps),
        "E1": lambda: e1_static(ds, reps_dict=allreps),
        "E2": lambda: e2_multiclass(ds, allreps, models=("logreg", "svm_lin", "svm_rbf", "knn")),
        "E2_rf": lambda: e2_multiclass(ds, key, models=("rf",)),
        "E4b_conditioned": lambda: e4_oneclass(ds, key, True, {"knn5_per_phase": _knn5,
                                                                "ledoitwolf_mahal_per_phase": _lw}),
        "E4_unconditioned": lambda: e4_oneclass(ds, oc, False, sc),
        "E5_latent": lambda: e5_latent({k: allreps[k] for k in ("psd_allholds", "psd_chassis_single58k",
                                                                 "comb120_allholds")}),
        "E6": lambda: e6_profile(ds),
        "E9_per_stage": lambda: e9_per_stage(ds),
        "E7_motor_signature": lambda: e7_motor(_signatures(), fd),
        "E11_template_update": lambda: e11_template_update(ds, key),
    }
    for tag, fn in steps.items():
        if which and tag not in which:
            continue
        print(f"== {tag} ==", flush=True)
        _save(tag, fn())


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:]) or None)
