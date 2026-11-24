"""
Drone A vs drone B identification experiments (2026-09-26).

Evaluation rules applied everywhere (see the report for why):
  1. Splits are by SESSION, never by window. Leave-one-session-out (LOSO):
     fold k trains on enrollment sessions != k of both drones and tests on
     session k of both drones. Held-out: train on all 5 enrollment sessions,
     test on the fixed_rig_test sessions (recorded on other occasions).
  2. Every positive result is paired with confound controls run through the
     identical pipeline:
       - DAY control: drone A's sessions of 2026-09-23 vs 2026-09-26 (all of
         drone B was recorded on 2026-09-20, so "drone" and "day" coincide);
       - NULL control: an arbitrary split of drone B's same-day sessions
         (s1,s3,s5 vs s2,s4) — measures how separable sessions are when
         there is no real systematic difference.
     A drone result only counts as identity evidence if it is clearly above
     both controls.
Results go to analysis/results/*.json; the report interprets them.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score
from sklearn.feature_selection import f_classif

from analysis.dataset import load_dataset, cols, ROOT

warnings.filterwarnings("ignore")
OUT = ROOT / "analysis" / "results"
OUT.mkdir(parents=True, exist_ok=True)
RNG = 0
AB_CUTOFF = "2026-09-26"

# ── representations ──────────────────────────────────────────────────────────

SWEEP_HOLDS = {"m1_hold", "m2_hold", "m3_hold", "m4_hold", "all_four_hold",
               "diag_13_hold", "diag_24_hold", "front_12_hold", "rear_34_hold",
               "left_14_hold", "right_23_hold"}
SINGLE_58K = {"m1_hold", "m2_hold", "m3_hold", "m4_hold"}
CHASSIS_MAX_HZ = 375      # below the slowest shaft fundamental at 58k PWM (~389 Hz,
                          # drone A M1) — no direct motor line or harmonic inside


ENERGY_FEATS = (".var", ".lfcc00")   # the only 120-D features that scale with overall
                                     # amplitude; lfcc01+ are gain-invariant (a constant
                                     # gain only shifts DCT coefficient 0), as are skew,
                                     # kurtosis, centroid, roll-off, flatness, coherence


def _shape(df: pd.DataFrame, psd_cols: list) -> "tuple[pd.DataFrame, list]":
    """Level-free log-PSD: per window and channel, subtract the channel's mean
    log-power over the selected bins. Keeps where energy sits in frequency,
    drops how much energy there is overall (which a faster motor, or a
    tighter/looser mount, would change without the chassis changing)."""
    out = df.copy()
    new = []
    for ch in sorted({c.split(":")[1] for c in psd_cols}):
        cc = [c for c in psd_cols if c.split(":")[1] == ch]
        centred = df[cc].values - df[cc].values.mean(axis=1, keepdims=True)
        names = [c.replace("psd:", "shape:") for c in cc]
        out[names] = centred
        new += names
    return out, new


def representations(ds: pd.DataFrame, extended: bool = False) -> "dict[str, tuple[pd.DataFrame, list]]":
    """name -> (window subset, feature columns). extended=True adds the
    amplitude-free variants used as the level-confound check."""
    holds = ds[ds.phase.str.contains("hold", na=False)]
    sweep = holds[holds.phase.isin(SWEEP_HOLDS)]
    single = holds[holds.phase.isin(SINGLE_58K)]
    psd = cols(ds, "psd")
    psd_sub = [c for c in psd if int(c.rsplit(":", 1)[1]) <= CHASSIS_MAX_HZ - 10]
    if extended:
        comb_ne = [c for c in cols(ds, "comb") if not c.endswith(ENERGY_FEATS)]
        s_single, f_single = _shape(single, psd_sub)
        s_sweep, f_sweep = _shape(sweep, psd_sub)
        s_all, f_all = _shape(holds, psd)
        return {
            "comb120_noenergy_allholds": (holds, comb_ne),
            "comb120_noenergy_58k": (sweep, comb_ne),
            "shape_allholds": (s_all, f_all),
            "shape_chassis_single58k": (s_single, f_single),
            "shape_chassis_58k": (s_sweep, f_sweep),
        }
    return {
        # all steady holds (stepped ladders dominate the window count)
        "comb120_allholds": (holds, cols(ds, "comb")),
        "raw120_allholds":  (holds, cols(ds, "raw")),
        "psd_allholds":     (holds, psd),
        # 58k sweep holds only (1 motor / pairs / all four)
        "comb120_58k":      (sweep, cols(ds, "comb")),
        "psd_58k":          (sweep, psd),
        # CHASSIS-ONLY: single motor, 58k, band below any motor line
        "psd_chassis_single58k": (single, psd_sub),
        # same band but also pairs/all-four (beat frequencies possible)
        "psd_chassis_58k":  (sweep, psd_sub),
    }


def classifiers() -> dict:
    return {
        "logreg":  lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)),
        "svm_lin": lambda: make_pipeline(StandardScaler(), LinearSVC(C=0.01, max_iter=20000)),
        "svm_rbf": lambda: make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale")),
        "rf":      lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                  n_jobs=-1, random_state=RNG),
        "knn":     lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
    }

# ── tasks: (label function, sessions used, fold assignment) ──────────────────

def task_frames(sub: pd.DataFrame) -> "dict[str, pd.DataFrame]":
    en = sub[sub.kind == "enroll"].copy()
    drone = en.assign(y=(en.drone == "B").astype(int), fold=en.session)
    a = en[en.drone == "A"]
    day = a.assign(y=(a.day == "2026-09-26").astype(int), fold=a.session)
    b = en[en.drone == "B"]
    null = b.assign(y=b.session.isin([2, 4]).astype(int), fold=b.session)
    return {"drone_A_vs_B": drone, "control_day_A": day, "control_null_B": null}


def _agg_acc(df_pred: pd.DataFrame, key: str) -> float:
    g = df_pred.groupby(key).agg(y=("y", "first"), p=("pred", "mean"))
    return float(((g.p >= 0.5).astype(int) == g.y).mean())


def loso(frame: pd.DataFrame, feats: list, make) -> dict:
    preds = []
    for k in sorted(frame.fold.unique()):
        tr, te = frame[frame.fold != k], frame[frame.fold == k]
        if tr.y.nunique() < 2:
            continue
        m = make().fit(tr[feats].values, tr.y.values)
        preds.append(te[["y", "rec_id", "session_id"]].assign(pred=m.predict(te[feats].values)))
    p = pd.concat(preds)
    return dict(window_bal_acc=float(balanced_accuracy_score(p.y, p.pred)),
                recording_acc=_agg_acc(p, "rec_id"),
                session_acc=_agg_acc(p, "session_id"),
                n_windows=int(len(p)))


def heldout(sub: pd.DataFrame, feats: list, make) -> dict:
    tr, te = sub[sub.kind == "enroll"], sub[sub.kind == "test"]
    m = make().fit(tr[feats].values, (tr.drone == "B").astype(int).values)
    p = te[["rec_id", "session_id", "purpose"]].assign(
        y=(te.drone == "B").astype(int).values, pred=m.predict(te[feats].values))
    out = dict(window_bal_acc=float(balanced_accuracy_score(p.y, p.pred)),
               recording_acc=_agg_acc(p, "rec_id"),
               session_acc=_agg_acc(p, "session_id"))
    uns = p[p.purpose == "unseen-pwm"]
    if len(uns):
        out["unseen_pwm_window_acc"] = float((uns.pred == uns.y).mean())
        out["unseen_pwm_recording_acc"] = _agg_acc(uns, "rec_id")
    return out

# ── E1: static statistics — within vs between drone distances ────────────────

def e1_static(ds: pd.DataFrame, reps_dict=None) -> dict:
    """Per recording, average its hold windows into one vector; compare the
    distance between two recordings of the SAME stage from the same drone
    (different sessions) vs from different drones. Separation index =
    mean(between) / mean(within); permutation p-value on the drone labels."""
    res = {}
    rng = np.random.default_rng(RNG)
    for name, (sub, feats) in (reps_dict or representations(ds)).items():
        en = sub[sub.kind == "enroll"]
        mv = en.groupby("rec_id").agg(drone=("drone", "first"), purpose=("purpose", "first"),
                                       session=("session", "first"))
        X = StandardScaler().fit_transform(en.groupby("rec_id")[feats].mean().loc[mv.index].values)
        within, between = [], []
        pairs = []
        for pur, idx in mv.groupby("purpose").indices.items():
            for i in idx:
                for j in idx:
                    if i < j:
                        pairs.append((i, j))
        pairs = np.array(pairs)
        d = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)
        lab = mv.drone.values

        def ratio(labels):
            same = labels[pairs[:, 0]] == labels[pairs[:, 1]]
            return d[~same].mean() / d[same].mean()
        obs = ratio(lab)
        # permute drone labels at SESSION level (keeps session structure)
        sess = mv.drone + mv.session.astype(str)
        keys = sess.unique()
        perm = []
        for _ in range(999):
            shuffled = dict(zip(keys, rng.permutation([k[0] for k in keys])))
            perm.append(ratio(sess.map(shuffled).values))
        res[name] = dict(between_over_within=float(obs),
                         p_value=float((np.sum(np.array(perm) >= obs) + 1) / 1000))
    return res

# ── E2/E3: supervised classification + controls ─────────────────────────────

def e2_supervised(ds: pd.DataFrame, reps=None, models=None, reps_dict=None) -> dict:
    res = {}
    for name, (sub, feats) in (reps_dict or representations(ds)).items():
        if reps and name not in reps:
            continue
        res[name] = {}
        for mname, make in classifiers().items():
            if models and mname not in models:
                continue
            t0 = time.time()
            r = {t: loso(fr, feats, make) for t, fr in task_frames(sub).items()}
            r["heldout_test"] = heldout(sub, feats, make)
            r["seconds"] = round(time.time() - t0, 1)
            res[name][mname] = r
            print(f"  {name:24s} {mname:8s} drone rec={r['drone_A_vs_B']['recording_acc']:.2f} "
                  f"day={r['control_day_A']['recording_acc']:.2f} null={r['control_null_B']['recording_acc']:.2f} "
                  f"test={r['heldout_test']['recording_acc']:.2f}", flush=True)
    return res

# ── autoencoder (sklearn MLP, no torch) ─────────────────────────────────────

class AE:
    """Dense autoencoder d -> 64 -> k -> 64 -> d (tanh), trained X->X."""
    def __init__(self, k=8, seed=RNG):
        self.sc = StandardScaler()
        self.m = MLPRegressor(hidden_layer_sizes=(64, k, 64), activation="tanh",
                              alpha=1e-3, max_iter=400, early_stopping=True,
                              random_state=seed)

    MAX_TRAIN = 3000   # random subsample for training speed (~1 min/fit on the
                       # full ~8k windows with sklearn's CPU MLP); the scaler still
                       # sees all training windows

    def fit(self, X):
        Z = self.sc.fit_transform(X)
        if len(Z) > self.MAX_TRAIN:
            Z = Z[np.random.default_rng(RNG).choice(len(Z), self.MAX_TRAIN, replace=False)]
        self.m.fit(Z, Z)
        return self

    def latent(self, X):
        h = self.sc.transform(X)
        for W, b in list(zip(self.m.coefs_, self.m.intercepts_))[:2]:
            h = np.tanh(h @ W + b)
        return h

    def recon_error(self, X):
        Z = self.sc.transform(X)
        return ((self.m.predict(Z) - Z) ** 2).mean(axis=1)

# ── E4: one-class authentication (the real use case) ─────────────────────────

def _mahal_scores(Xtr, Xte, ncomp=20):
    p = make_pipeline(StandardScaler(), PCA(n_components=min(ncomp, Xtr.shape[1]), random_state=RNG)).fit(Xtr)
    Ztr, Zte = p.transform(Xtr), p.transform(Xte)
    mu = Ztr.mean(0); icov = np.linalg.pinv(np.cov(Ztr, rowvar=False))
    d = Zte - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, icov, d))


def one_class_scorers() -> dict:
    """name -> f(Xtrain_genuine, Xtest) -> anomaly score (higher = less like enrolled)."""
    def ocsvm(Xtr, Xte):
        m = make_pipeline(StandardScaler(), OneClassSVM(nu=0.05, gamma="scale")).fit(Xtr)
        return -m.decision_function(Xte)

    def iforest(Xtr, Xte):
        m = make_pipeline(StandardScaler(), IsolationForest(n_estimators=300, random_state=RNG)).fit(Xtr)
        return -m.decision_function(Xte)

    def knn(Xtr, Xte):
        sc = StandardScaler().fit(Xtr)
        nn = NearestNeighbors(n_neighbors=10).fit(sc.transform(Xtr))
        return nn.kneighbors(sc.transform(Xte))[0].mean(1)

    def ae_recon(Xtr, Xte):
        return AE(k=8).fit(Xtr).recon_error(Xte)

    def ae_latent(Xtr, Xte):
        ae = AE(k=8).fit(Xtr)
        Ltr, Lte = ae.latent(Xtr), ae.latent(Xte)
        mu = Ltr.mean(0); icov = np.linalg.pinv(np.cov(Ltr, rowvar=False))
        d = Lte - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, icov, d))

    return {"mahalanobis_pca20": _mahal_scores, "ocsvm": ocsvm, "iforest": iforest,
            "knn10": knn, "ae_recon": ae_recon, "ae_latent_mahal": ae_latent}


def _eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def e4_one_class(ds: pd.DataFrame, reps=None, reps_dict=None) -> dict:
    """Enroll ONE drone only (4 of its sessions); genuine = its held-out 5th
    session, impostor = every enrollment session of the other drone. Scores
    are averaged per recording (one decision per 5-140 s recording) before
    ROC/EER. Also a DAY control: enroll A's 2026-09-23 sessions, 'impostor'
    = A's 2026-09-26 sessions."""
    res = {}
    for name, (sub, feats) in (reps_dict or representations(ds)).items():
        if reps and name not in reps:
            continue
        en = sub[sub.kind == "enroll"]
        res[name] = {}
        for sname, score in one_class_scorers().items():
            out = {}
            for target in ("A", "B"):
                ys, ss = [], []
                for k in sorted(en.session.unique()):
                    tr = en[(en.drone == target) & (en.session != k)]
                    te = pd.concat([en[(en.drone == target) & (en.session == k)],
                                    en[en.drone != target]])
                    s = score(tr[feats].values, te[feats].values)
                    r = te[["rec_id"]].assign(s=s, y=(te.drone != target).astype(int).values)
                    g = r.groupby("rec_id").agg(s=("s", "mean"), y=("y", "first"))
                    ys.append(g.y.values); ss.append(g.s.values)
                y, s = np.concatenate(ys), np.concatenate(ss)
                out[f"enroll_{target}"] = dict(auc=float(roc_auc_score(y, s)), eer=_eer(y, s))
            # day control on drone A
            a = en[en.drone == "A"]
            tr, te = a[a.day == "2026-09-23"], a[a.day == "2026-09-26"]
            # genuine reference: leave one 09-23 session out
            ys, ss = [], []
            for k in sorted(tr.session.unique()):
                trk = tr[tr.session != k]
                tek = pd.concat([tr[tr.session == k], te])
                s = score(trk[feats].values, tek[feats].values)
                r = tek[["rec_id"]].assign(s=s, y=(tek.day == "2026-09-26").astype(int).values)
                g = r.groupby("rec_id").agg(s=("s", "mean"), y=("y", "first"))
                ys.append(g.y.values); ss.append(g.s.values)
            y, s = np.concatenate(ys), np.concatenate(ss)
            out["control_day_A"] = dict(auc=float(roc_auc_score(y, s)), eer=_eer(y, s))
            res[name][sname] = out
            print(f"  {name:24s} {sname:18s} AUC A={out['enroll_A']['auc']:.2f} B={out['enroll_B']['auc']:.2f} "
                  f"dayctl={out['control_day_A']['auc']:.2f}", flush=True)
    return res

# ── E5: unsupervised latent space — AE vs PCA ───────────────────────────────

def e5_latent(ds: pd.DataFrame, reps=("psd_allholds", "psd_chassis_single58k", "comb120_allholds")) -> dict:
    """Fit AE (and PCA of the same size) on the training sessions of BOTH
    drones without labels, then check whether a linear classifier on the
    k-dim code separates the drones on the held-out session (LOSO)."""
    res = {}
    for name, (sub, feats) in representations(ds).items():
        if name not in reps:
            continue
        en = sub[sub.kind == "enroll"]
        res[name] = {}
        for kind in ("ae8", "pca8"):
            preds = []
            for k in sorted(en.session.unique()):
                tr, te = en[en.session != k], en[en.session == k]
                if kind == "ae8":
                    enc = AE(k=8).fit(tr[feats].values); f = enc.latent
                else:
                    p = make_pipeline(StandardScaler(), PCA(8, random_state=RNG)).fit(tr[feats].values); f = p.transform
                clf = LogisticRegression(max_iter=2000).fit(f(tr[feats].values), (tr.drone == "B").astype(int))
                preds.append(te[["rec_id", "session_id"]].assign(
                    y=(te.drone == "B").astype(int).values, pred=clf.predict(f(te[feats].values))))
            p = pd.concat(preds)
            res[name][kind] = dict(window_bal_acc=float(balanced_accuracy_score(p.y, p.pred)),
                                   recording_acc=_agg_acc(p, "rec_id"), session_acc=_agg_acc(p, "session_id"))
            print(f"  {name:24s} {kind}: rec={res[name][kind]['recording_acc']:.2f}", flush=True)
    return res

# ── E6: which features / frequencies separate the drones ────────────────────

def e6_importance(ds: pd.DataFrame) -> dict:
    en = ds[(ds.kind == "enroll") & ds.phase.str.contains("hold", na=False)]
    y = (en.drone == "B").astype(int).values
    out = {}
    for rep in ("comb", "psd"):
        f = cols(ds, rep)
        F, _ = f_classif(en[f].values, y)
        s = pd.Series(F, index=f).sort_values(ascending=False)
        out[rep] = {k: float(v) for k, v in s.head(25).items()}
    # frequency profile of separation for single-motor 58k holds
    single = en[en.phase.isin(SINGLE_58K)]
    f = cols(ds, "psd")
    F, _ = f_classif(single[f].values, (single.drone == "B").astype(int).values)
    prof = pd.Series(F, index=f)
    out["psd_single58k_F_by_bin"] = {k: float(v) for k, v in prof.items()}
    return out


# ── E4b: stage-conditioned one-class authentication ─────────────────────────

def e4b_conditioned(ds: pd.DataFrame, reps_dict: dict) -> dict:
    """Like E4, but each recording is only compared with enrollment windows
    of the SAME excitation phase (e.g. m1_hold, m3_hold_35000, all_four_hold).
    Rationale: E4's single model per drone had to describe 15 stages x 7 PWM
    levels at once — a strongly multimodal cloud a single Gaussian cannot fit
    (Mahalanobis at chance). In a real challenge-response authentication the
    verifier knows which stage it commanded, so conditioning on it is free.
    Scorers: kNN mean distance (k=5) and Ledoit-Wolf Mahalanobis (shrinkage
    needed: only ~36 enrollment windows per phase for the 58k holds)."""
    from sklearn.covariance import LedoitWolf

    def knn(Xtr, Xte):
        sc = StandardScaler().fit(Xtr)
        nn = NearestNeighbors(n_neighbors=min(5, len(Xtr))).fit(sc.transform(Xtr))
        return nn.kneighbors(sc.transform(Xte))[0].mean(1)

    def lw_mahal(Xtr, Xte):
        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
        lw = LedoitWolf().fit(Ztr)
        return np.sqrt(lw.mahalanobis(Zte))

    scorers = {"knn5_per_phase": knn, "ledoitwolf_mahal_per_phase": lw_mahal}
    res = {}
    for name, (sub, feats) in reps_dict.items():
        en = sub[sub.kind == "enroll"]
        res[name] = {}
        for sname, score in scorers.items():
            out = {}
            def run(train_mask_fn, test_df, y_fn):
                rows = []
                for ph, te_ph in test_df.groupby("phase"):
                    tr_ph = train_mask_fn(ph)
                    if len(tr_ph) < 10:
                        continue
                    s = score(tr_ph[feats].values, te_ph[feats].values)
                    rows.append(te_ph[["rec_id"]].assign(s=s, y=y_fn(te_ph)))
                r = pd.concat(rows)
                g = r.groupby("rec_id").agg(s=("s", "mean"), y=("y", "first"))
                return g.y.values, g.s.values
            for target in ("A", "B"):
                ys, ss = [], []
                for k in sorted(en.session.unique()):
                    tr = en[(en.drone == target) & (en.session != k)]
                    te = pd.concat([en[(en.drone == target) & (en.session == k)], en[en.drone != target]])
                    y, s = run(lambda ph: tr[tr.phase == ph], te,
                               lambda d: (d.drone != target).astype(int).values)
                    ys.append(y); ss.append(s)
                y, s = np.concatenate(ys), np.concatenate(ss)
                out[f"enroll_{target}"] = dict(auc=float(roc_auc_score(y, s)), eer=_eer(y, s))
                # held-out test sessions: enroll on all 5, genuine = target's test
                # sessions, impostor = other drone's test sessions
                tst = sub[sub.kind == "test"]
                y, s = run(lambda ph: en[(en.drone == target) & (en.phase == ph)], tst,
                           lambda d: (d.drone != target).astype(int).values)
                out[f"heldout_enroll_{target}"] = dict(auc=float(roc_auc_score(y, s)), eer=_eer(y, s))
            a = en[en.drone == "A"]
            tr_day, te_day = a[a.day == "2026-09-23"], a[a.day == "2026-09-26"]
            ys, ss = [], []
            for k in sorted(tr_day.session.unique()):
                trk = tr_day[tr_day.session != k]
                tek = pd.concat([tr_day[tr_day.session == k], te_day])
                y, s = run(lambda ph: trk[trk.phase == ph], tek,
                           lambda d: (d.day == "2026-09-26").astype(int).values)
                ys.append(y); ss.append(s)
            y, s = np.concatenate(ys), np.concatenate(ss)
            out["control_day_A"] = dict(auc=float(roc_auc_score(y, s)), eer=_eer(y, s))
            res[name][sname] = out
            print(f"  {name:28s} {sname:28s} AUC A={out['enroll_A']['auc']:.2f} B={out['enroll_B']['auc']:.2f} "
                  f"test A={out['heldout_enroll_A']['auc']:.2f} B={out['heldout_enroll_B']['auc']:.2f} "
                  f"dayctl={out['control_day_A']['auc']:.2f}", flush=True)
    return res


# ── E1 day control, E4c, E9 (moved here from one-off scripts for reproducibility)

def e1_day_control(ds: pd.DataFrame, reps_dict: dict) -> dict:
    """Drone effect vs day effect in the same distance metric: inside drone A,
    distance between recordings of different days / of the same day."""
    res = {}
    for name, (sub, feats) in reps_dict.items():
        en = sub[sub.kind == "enroll"]
        mv = en.groupby("rec_id").agg(drone=("drone", "first"), purpose=("purpose", "first"), day=("day", "first"))
        X = StandardScaler().fit_transform(en.groupby("rec_id")[feats].mean().loc[mv.index].values)
        P = np.array([(i, j) for idx in mv.groupby("purpose").indices.values() for i in idx for j in idx if i < j])
        d = np.linalg.norm(X[P[:, 0]] - X[P[:, 1]], axis=1)
        dr, dy = mv.drone.values, mv.day.values
        same = dr[P[:, 0]] == dr[P[:, 1]]
        aa = (dr[P[:, 0]] == "A") & (dr[P[:, 1]] == "A")
        sameday = dy[P[:, 0]] == dy[P[:, 1]]
        res[name] = dict(drone_ratio=float(d[~same].mean() / d[same].mean()),
                         day_ratio_within_A=float(d[aa & ~sameday].mean() / d[aa & sameday].mean()))
    return res


def e4c_ae_latent_conditioned(ds: pd.DataFrame, reps_dict: dict) -> dict:
    """AE trained on the enrolled drone only; kNN in its 8-D latent, per phase."""
    res = {}
    for name, (sub, feats) in reps_dict.items():
        en = sub[sub.kind == "enroll"]
        out = {}
        for target in ("A", "B"):
            ys, ss = [], []
            for k in sorted(en.session.unique()):
                tr = en[(en.drone == target) & (en.session != k)]
                te = pd.concat([en[(en.drone == target) & (en.session == k)], en[en.drone != target]])
                ae = AE(k=8).fit(tr[feats].values)
                Ltr, Lte = ae.latent(tr[feats].values), ae.latent(te[feats].values)
                parts = []
                for ph in te.phase.unique():
                    mt, me = (tr.phase == ph).values, (te.phase == ph).values
                    if mt.sum() < 10:
                        continue
                    nn = NearestNeighbors(n_neighbors=5).fit(Ltr[mt])
                    parts.append(pd.DataFrame({"rec": te.rec_id.values[me],
                                               "s": nn.kneighbors(Lte[me])[0].mean(1),
                                               "y": (te.drone.values[me] != target).astype(int)}))
                g = pd.concat(parts).groupby("rec").agg(s=("s", "mean"), y=("y", "first"))
                ys.append(g.y.values); ss.append(g.s.values)
            y, s = np.concatenate(ys), np.concatenate(ss)
            out[f"enroll_{target}"] = dict(auc=float(roc_auc_score(y, s)), eer=_eer(y, s))
        res[name] = out
    return res


def e9_per_stage(ds: pd.DataFrame) -> dict:
    """One-class per-phase kNN using ONE stage at a time: which single
    excitation is enough? AUC averaged over enrolling A and enrolling B."""
    holds = ds[ds.phase.str.contains("hold", na=False)]
    psd = cols(ds, "psd")
    sub = [c for c in psd if int(c.rsplit(":", 1)[1]) <= CHASSIS_MAX_HZ - 10]

    def knn_score(tr, te, f):
        sc = StandardScaler().fit(tr[f].values)
        nn = NearestNeighbors(n_neighbors=min(5, len(tr))).fit(sc.transform(tr[f].values))
        return nn.kneighbors(sc.transform(te[f].values))[0].mean(1)

    res = {}
    for pur in sorted(holds.purpose.unique()):
        if pur == "unseen-pwm":
            continue
        h = holds[(holds.purpose == pur) & (holds.kind == "enroll")]
        hs, fs = _shape(h, sub)
        row = {}
        for name, (frame, f) in {"psd": (h, psd), "shape_chassis": (hs, fs)}.items():
            aucs = []
            for target in ("A", "B"):
                ys, ss = [], []
                for k in sorted(frame.session.unique()):
                    tr = frame[(frame.drone == target) & (frame.session != k)]
                    te = pd.concat([frame[(frame.drone == target) & (frame.session == k)],
                                    frame[frame.drone != target]])
                    parts = []
                    for ph, tph in te.groupby("phase"):
                        trp = tr[tr.phase == ph]
                        if len(trp) < 10:
                            continue
                        parts.append(tph[["rec_id", "phase"]].assign(
                            s=knn_score(trp, tph, f), y=(tph.drone != target).astype(int).values))
                    g = pd.concat(parts).groupby(["rec_id", "phase"]).agg(s=("s", "mean"), y=("y", "first"))
                    ys.append(g.y.values); ss.append(g.s.values)
                aucs.append(roc_auc_score(np.concatenate(ys), np.concatenate(ss)))
            row[name] = float(np.mean(aucs))
        res[pur] = row
    return res


def main(which=None):
    """Regenerate every result file used by paper/experiments/identificazione_droni_A_B.md."""
    ds = load_dataset(verbose=False)
    # the A/B study as reported on 2026-09-26: drones A and B, data recorded
    # up to that day (C, D and B's second test session came later)
    ds = ds[ds.drone.isin(["A", "B"]) & (ds.day <= AB_CUTOFF)].reset_index(drop=True)
    base, ext = representations(ds), representations(ds, extended=True)
    oc = {k: base[k] for k in ("comb120_allholds", "psd_chassis_single58k")}
    oc["shape_chassis_single58k"] = ext["shape_chassis_single58k"]
    cond = {k: base[k] for k in ("comb120_allholds", "psd_allholds", "psd_chassis_single58k")}
    cond.update({k: ext[k] for k in ("shape_chassis_single58k", "comb120_noenergy_allholds", "shape_allholds")})
    steps = {
        "E1": lambda: e1_static(ds),
        "E1_day_control": lambda: e1_day_control(ds, base),
        "E1_levelfree": lambda: e1_static(ds, reps_dict=ext),
        "E2": lambda: e2_supervised(ds),
        "E2_levelfree": lambda: e2_supervised(ds, models=("logreg", "svm_lin", "svm_rbf", "knn"), reps_dict=ext),
        "E4": lambda: e4_one_class(ds, reps_dict=oc),
        "E4b": lambda: e4b_conditioned(ds, cond),
        "E4c_ae_latent_conditioned": lambda: e4c_ae_latent_conditioned(ds, oc),
        "E5": lambda: e5_latent(ds),
        "E6": lambda: e6_importance(ds),
        "E9_per_stage": lambda: e9_per_stage(ds),
    }
    for tag, fn in steps.items():
        if which and tag not in which:
            continue
        print(f"== {tag} ==", flush=True)
        (OUT / f"{tag}.json").write_text(json.dumps(fn(), indent=1), encoding="utf-8")


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:]) or None)
