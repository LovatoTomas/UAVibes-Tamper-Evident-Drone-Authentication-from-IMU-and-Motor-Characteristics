"""
Motor-swap experiment analysis (A <-> B, 2026-09-30).

Question: after exchanging the four motors between drones A and B (same
positions: A's M1 motor goes into B's M1 slot, ...), does each read-out follow
the FRAME or the MOTORS?

For every recording of a swap session (data/fixed_rig_swap/sNN_.../) two
independent read-outs are computed, each answering "looks like A or B?":

  CHASSIS  — the chassis-band spectrum (5-375 Hz, below every motor line at
             58k PWM) of each single-motor 58k hold, compared phase by phase
             with A's and B's windows of the same phase. Two representations
             (level-included log-PSD, level-free shape) x two scorers
             (Ledoit-Wolf Mahalanobis per drone, kNN in a shared scaling),
             the same ones that worked best in the four-drone study.
  MOTORS   — each motor's speed gain (measured / base-model prediction) at
             58k, compared with A's and B's gain for that motor position
             (z-distance in units of the within-drone session spread).

References: (1) A's and B's five enrollment sessions; (2) if present, the
same-day "pre" swap sessions (original configuration, recorded just before
the swap) — the drift-free reference, since motors slow down over time.

Outputs, per session folder: analysis/verdict.json + analysis/verdict.md.
Summary over all sessions: data/fixed_rig_swap/summary.md and
analysis/results_swap/summary.json.

Run:  python -m analysis.swap            (all sessions)
      python -m analysis.swap 3          (one session number)
"""

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf

from analysis.dataset import ROOT, CACHE, load_dataset, cols, _process
from analysis.experiments import _shape, SINGLE_58K, CHASSIS_MAX_HZ
from analysis.motor_signature import measure_hold
from pipeline.preprocessing import load_session, infer_fs, highpass

SWAP_ROOT = ROOT / "data" / "fixed_rig_swap"
OUT = ROOT / "analysis" / "results_swap"
PAIR = ("A", "B")


# ── data ─────────────────────────────────────────────────────────────────────

def iter_swap_recordings(root: Path = SWAP_ROOT):
    for mf in sorted(root.glob("s*_*/session.json")):
        d = json.load(open(mf, encoding="utf-8"))
        for r in d["recordings"]:
            acc = [a for a in r["attempts"]
                   if a["status"] in ("matched", "completed") and a.get("converted_path")]
            if not acc:
                continue
            a = acc[-1]
            p = Path(a["converted_path"])
            yield dict(drone=d["drone"], kind="swap", session=d["session"], day=d["created_iso"][:10],
                       purpose=r["purpose"], path=str(p if p.is_absolute() else ROOT / p),
                       motors=d["motors"], stage=d["swap_stage"], session_dir=str(mf.parent),
                       vbat_rest=a.get("vbat_rest"))


def load_swap(root: Path = SWAP_ROOT) -> pd.DataFrame:
    """Window-level features of every swap recording (same cache as dataset.py)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    parts = []
    for rec in iter_swap_recordings(root):
        p = Path(rec["path"])
        key = hashlib.md5(f"{p.name}:{p.stat().st_mtime_ns}:v1".encode()).hexdigest()[:12]
        cf = CACHE / f"{p.stem}_{key}.pkl"
        if cf.exists():
            w = pd.read_pickle(cf)
        else:
            w = _process(rec)
            w.to_pickle(cf)
        # session metadata always from the swap manifest, never from the
        # cache (a cached file may have been processed under another label)
        parts.append(w.assign(drone=rec["drone"], kind="swap", session=rec["session"], day=rec["day"],
                              purpose=rec["purpose"],
                              session_id=f"{rec['drone']}-swap-{rec['session']:02d}",
                              rec_id=f"w{rec['session']:02d}_{Path(rec['path']).stem}",
                              motors=rec["motors"], stage=rec["stage"], session_dir=rec["session_dir"],
                              path=rec["path"]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _chassis_reps(df: pd.DataFrame) -> "dict[str, tuple[pd.DataFrame, list]]":
    h = df[df.phase.isin(SINGLE_58K)]
    band = [c for c in cols(df, "psd") if int(c.rsplit(":", 1)[1]) <= CHASSIS_MAX_HZ - 10]
    hs, fs = _shape(h, band)
    return {"psd_chassis_single58k": (h, band), "shape_chassis_single58k": (hs, fs)}


# ── chassis read-out ─────────────────────────────────────────────────────────

def _lw_score(tr, te):
    sc = StandardScaler().fit(tr)
    return np.sqrt(LedoitWolf().fit(sc.transform(tr)).mahalanobis(sc.transform(te)))


def chassis_readout(test: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """Per recording of `test`: mean score vs A and vs B (same phase only).
    log-ratio = log(score_A / score_B): < 0 looks like A, > 0 looks like B."""
    out = {}
    test_reps, ref_reps = _chassis_reps(test), _chassis_reps(ref)
    for rep, (te, f) in test_reps.items():
        rf = ref_reps[rep][0]
        rows = []
        for ph, te_ph in te.groupby("phase"):
            r_ph = rf[rf.phase == ph]
            refs = {d: r_ph[r_ph.drone == d][f].values for d in PAIR}
            if min(len(v) for v in refs.values()) < 5:     # a sweep-m hold is ~5 s = ~9 windows per session
                continue
            both = np.vstack(list(refs.values()))
            sc = StandardScaler().fit(both)
            X = te_ph[f].values
            for scorer in ("ledoitwolf", "knn5"):
                s = {}
                for d in PAIR:
                    if scorer == "ledoitwolf":
                        s[d] = _lw_score(refs[d], X).mean()
                    else:
                        nn = NearestNeighbors(n_neighbors=5).fit(sc.transform(refs[d]))
                        s[d] = nn.kneighbors(sc.transform(X))[0].mean()
                rows.append(dict(purpose=te_ph.purpose.iloc[0], phase=ph, scorer=scorer,
                                 score_A=float(s["A"]), score_B=float(s["B"]),
                                 log_ratio=float(np.log(s["A"] / s["B"]))))
        out[rep] = rows
    return out


# ── motor read-out ───────────────────────────────────────────────────────────

def _gains_of(recs: pd.DataFrame) -> pd.DataFrame:
    """Per-motor 58k gain for each single-motor recording (unique paths)."""
    rows = []
    for path, g in recs.groupby("path"):
        purpose = g.purpose.iloc[0]
        if not purpose.startswith("sweep-m"):
            continue
        m = int(purpose[-1])
        df = load_session(path)
        fs = infer_fs(df)
        hp = highpass(df, fs)
        seg = hp[hp.phase == f"m{m}_hold"]
        if len(seg) < 2000:
            continue
        r = measure_hold(seg, fs, m)
        if r:
            rows.append(dict(session_id=g.session_id.iloc[0], drone=g.drone.iloc[0], motor=m, **r))
    return pd.DataFrame(rows)


def reference_gains() -> pd.DataFrame:
    """A's and B's 58k single-motor gains from the enrollment sessions
    (computed once by the four-drone study)."""
    s = pd.read_csv(ROOT / "analysis" / "results_4drones" / "motor_signatures.csv")
    s = s[s.drone.isin(PAIR) & (s.kind == "enroll") & s.purpose.str.startswith("sweep-m")]
    return s[["drone", "session", "motor", "gain"]]


def motor_readout(gains: pd.DataFrame, ref: pd.DataFrame,
                  spread_ref: "pd.DataFrame | None" = None) -> "list[dict]":
    """z-distance of each measured gain to A's and B's median gain for that
    motor position. Spread = within-drone session-to-session spread of
    `spread_ref` (default: `ref`), robust (median absolute deviation) so the
    known single-session misdetections of B's M3/M4 do not inflate it. With a
    one-session reference (the same-day pre sessions) the centre comes from
    it and the spread from the enrollment sessions."""
    spread_ref = ref if spread_ref is None else spread_ref
    rows = []
    for _, g in gains.iterrows():
        r = ref[ref.motor == g.motor]
        med = r.groupby("drone").gain.median()
        rs = spread_ref[spread_ref.motor == g.motor]
        mad = rs.groupby("drone").gain.apply(lambda x: 1.4826 * np.median(np.abs(x - x.median())))
        sd = max(float(mad.mean()), 0.005)
        zA, zB = abs(g.gain - med["A"]) / sd, abs(g.gain - med["B"]) / sd
        rows.append(dict(motor=int(g.motor), gain=float(g.gain), ref_A=float(med["A"]),
                         ref_B=float(med["B"]), z_A=float(zA), z_B=float(zB),
                         looks_like="A" if zA < zB else "B"))
    return rows


# ── verdicts ─────────────────────────────────────────────────────────────────

def _vote(values, key="log_ratio"):
    lr = np.array([v[key] for v in values])
    return ("A" if lr.mean() < 0 else "B"), float(lr.mean()), int((lr < 0).sum()), int((lr > 0).sum())


def analyse_session(sdir: Path, swap: pd.DataFrame, enroll_ref: pd.DataFrame, gref: pd.DataFrame) -> dict:
    s = swap[swap.session_dir == str(sdir)]
    if s.empty:
        return {}
    meta = dict(session=int(s.session.iloc[0]), stage=s.stage.iloc[0], frame=s.drone.iloc[0],
                motors=s.motors.iloc[0], day=s.day.iloc[0], n_recordings=int(s.rec_id.nunique()))
    res = {"config": meta}
    refs = {"enrollment": enroll_ref}
    pre = swap[(swap.stage == "pre") & (swap.day == meta["day"])]
    if set(pre.drone) >= set(PAIR) and meta["stage"] != "pre":
        refs["pre_same_day"] = pre
    for rname, ref in refs.items():
        ch = chassis_readout(s, ref)
        v = {}
        for rep, rows in ch.items():
            for scorer in ("ledoitwolf", "knn5"):
                rr = [r for r in rows if r["scorer"] == scorer]
                if rr:
                    who, lr, nA, nB = _vote(rr)
                    v[f"{rep}/{scorer}"] = dict(looks_like=who, mean_log_ratio=lr,
                                                phases_A=nA, phases_B=nB)
        res[f"chassis_vs_{rname}"] = dict(summary=v, per_phase=ch)
    g = _gains_of(s)
    if len(g):
        mref = gref
        if "pre_same_day" in refs:
            gp = _gains_of(pre)
            if len(gp):
                mref = gp.assign(session=0)[["drone", "session", "motor", "gain"]]
                res["motors_vs_pre_same_day"] = motor_readout(g, mref, spread_ref=gref)
        res["motors_vs_enrollment"] = motor_readout(g, gref)
    return res


def _headline(res: dict) -> dict:
    """One line per read-out: majority over configurations/motors."""
    h = {}
    for k in ("chassis_vs_enrollment", "chassis_vs_pre_same_day"):
        if k in res:
            votes = [v["looks_like"] for v in res[k]["summary"].values()]
            h[k] = f"{votes.count('A')}×A / {votes.count('B')}×B"
    for k in ("motors_vs_enrollment", "motors_vs_pre_same_day"):
        if k in res:
            votes = [m["looks_like"] for m in res[k]]
            h[k] = f"{votes.count('A')}×A / {votes.count('B')}×B"
    return h


def _md(res: dict) -> str:
    c = res["config"]
    lines = [f"# Swap session {c['session']:02d} — {c['stage']}, frame {c['frame']}, motors {c['motors']}",
             "", f"Day {c['day']}, {c['n_recordings']} recordings.", "",
             "Expected if the identity belongs to the **frame**: frame signature = " + c["frame"] +
             "; if it belongs to the **motors**: frame signature = " + c["motors"] +
             ". The motor signature must follow the motors anyway (= " + c["motors"] + ").", "",
             "| Read-out | Result |", "|---|---|"]
    for k, v in _headline(res).items():
        lines.append(f"| {k} | {v} |")
    for k in ("chassis_vs_enrollment", "chassis_vs_pre_same_day"):
        if k in res:
            lines += ["", f"## {k}", "", "| configuration | looks like | mean log(sA/sB) | stages A / B |", "|---|---|---|---|"]
            for name, v in res[k]["summary"].items():
                lines.append(f"| {name} | **{v['looks_like']}** | {v['mean_log_ratio']:+.3f} | "
                             f"{v['phases_A']} / {v['phases_B']} |")
    for k in ("motors_vs_enrollment", "motors_vs_pre_same_day"):
        if k in res:
            lines += ["", f"## {k}", "", "| motor | gain | ref. A | ref. B | z A | z B | looks like |",
                      "|---|---|---|---|---|---|---|"]
            for m in res[k]:
                lines.append(f"| M{m['motor']} | {m['gain']:.3f} | {m['ref_A']:.3f} | {m['ref_B']:.3f} | "
                             f"{m['z_A']:.1f} | {m['z_B']:.1f} | **{m['looks_like']}** |")
    return "\n".join(lines) + "\n"


def run(session: "int | None" = None, root: Path = SWAP_ROOT, verbose: bool = True) -> dict:
    swap = load_swap(root)
    if swap.empty:
        if verbose:
            print(f"No swap session with converted data in {root}")
        return {}
    ds = load_dataset(verbose=False)
    enroll_ref = ds[ds.drone.isin(PAIR) & (ds.kind == "enroll")]
    gref = reference_gains()
    summary = {}
    for sdir in sorted(swap.session_dir.unique()):
        n = int(swap[swap.session_dir == sdir].session.iloc[0])
        if session is not None and n != session:
            continue
        res = analyse_session(Path(sdir), swap, enroll_ref, gref)
        adir = Path(sdir) / "analysis"
        adir.mkdir(parents=True, exist_ok=True)
        (adir / "verdict.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        (adir / "verdict.md").write_text(_md(res), encoding="utf-8")
        summary[Path(sdir).name] = dict(config=res["config"], **_headline(res))
        if verbose:
            print(Path(sdir).name, _headline(res), flush=True)
    if session is None:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        md = ["| session | stage | frame | motors | frame vs enrollment | frame vs pre | motors vs enrollment | motors vs pre |",
              "|---|---|---|---|---|---|---|---|"]
        for name, s in summary.items():
            c = s["config"]
            md.append(f"| {name} | {c['stage']} | {c['frame']} | {c['motors']} | "
                      f"{s.get('chassis_vs_enrollment', '')} | {s.get('chassis_vs_pre_same_day', '')} | "
                      f"{s.get('motors_vs_enrollment', '')} | {s.get('motors_vs_pre_same_day', '')} |")
        (root / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    import sys
    run(int(sys.argv[1]) if len(sys.argv) > 1 else None)
