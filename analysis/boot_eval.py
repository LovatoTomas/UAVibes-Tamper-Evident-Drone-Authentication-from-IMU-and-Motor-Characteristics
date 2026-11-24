"""
Evaluation of the REAL boot sessions (2026-10-13): the boot protocol recorded
end-to-end (board at rest + each motor at 58k in the same power-up), for the
four drones. Removes the main limitation of the earlier key studies, where
motor and board measurements came from different sessions.

  1. data quality (losses, self-test retries, SD stalls, tilt)
  2. per-feature separation on real boots
  3. key, all enrollment choices: for every drone and every set of 3 of its
     boot sessions used as enrollment, the remaining sessions are genuine
     boots and every session of the other drones is an impostor; motors only
     / board only / combined; t = 0..4 (no helper refresh: each boot is
     compared with the enrollment directly — the conservative case)
  4. one-week stability of the board part: enrollment from the board
     sessions of 2026-10-07/08, verification on the boot sessions of 10-13
  5. long-term change of the motor part: per-motor speed in the September
     studies, the motor swap (09-30) and the boot sessions (10-13)
  6. tilt: drone B remounted ~2.7 deg off between two boots
  7. entropy estimate on real boots
  8. cross-day (2026-10-14 onward): enrolled on the first day's boots, the
     later days' boots verified with the real fuzzy extractor, without and
     with the on-board helper refresh, plus impostors

Sections 1-7 use only the first day (FIRST_DAY), so report 11 reproduces as
written when later boots are added; section 8 runs once there is a later day.

Run:  python -m analysis.boot_eval     -> analysis/results_boot_eval/
"""

import itertools
import json
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.dataset import ROOT
from keyproto import boot as kb
from keyproto.fe import FuzzyExtractor

OUT = ROOT / "analysis" / "results_boot_eval"
DRONES = ["A", "B", "C", "D"]
N_ENROLL = 3
FIRST_DAY = "2026-10-13"     # the boots of report 11


# ── data ────────────────────────────────────────────────────────────────────

def load_boots() -> "tuple[pd.DataFrame, list]":
    names = kb.system_params()["features"]
    rows = []
    for s in sorted(glob.glob(str(ROOT / "data" / "fixed_rig_boot" / "s*_drone*"))):
        m = json.load(open(os.path.join(s, "session.json"), encoding="utf-8"))
        bp = m.get("board_params") or {}
        conv = [a for r in m["recordings"] for a in r["attempts"] if a.get("converted_path")]
        if not conv:
            continue
        d = pd.read_csv(conv[-1]["converted_path"])
        rest = d[d.phase == "rest_hold"]
        dt = np.diff(d.timestamp_s.values)
        x, info = kb.boot_features(s)
        row = dict(session=m["session"], drone=m["drone"], start=m["created_iso"],
                   day=m["created_iso"][:10],
                   temp=bp.get("board_temp_c"), vbat=bp.get("vbat_rest"),
                   st_attempts=bp.get("stDone"), st_ok=bp.get("stResult") == 0,
                   rest_s=len(rest) / 1000.0, g_x=float(rest["acc.x"].mean()), g_y=float(rest["acc.y"].mean()),
                   n_stalls=int((dt > 0.3).sum()), lost_s=float(dt[dt > 0.01].sum()),
                   n_missing=info["n_missing"], **dict(zip(names, x)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("session").reset_index(drop=True), names


# ── key machinery at symbol level (accept iff wrong symbols <= t) ───────────

def symbols(x: np.ndarray, off: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.floor((x + off) / w)


def enroll_ref(X: np.ndarray, w: np.ndarray):
    ref = np.nanmedian(X, axis=0)
    idx = np.floor(ref / w)
    return (idx + 0.5) * w - ref, idx


def n_wrong(x, off, idx, w) -> int:
    q = symbols(x, off, w)
    return int(np.sum((q != idx) | np.isnan(q)))


def key_eval(B: pd.DataFrame, names, cols) -> dict:
    w_all = np.asarray(kb.system_params()["widths"], dtype=float)
    sel = [names.index(c) for c in cols]
    w = w_all[sel]
    X = B[cols].values.astype(float)
    gen, imp = [], []
    per_drone = {}
    for T in DRONES:
        ids = np.where(B.drone.values == T)[0]
        g_d = []
        for combo in itertools.combinations(ids, N_ENROLL):
            off, idx = enroll_ref(X[list(combo)], w)
            for i in ids:
                if i not in combo:
                    g_d.append(n_wrong(X[i], off, idx, w))
            for i in np.where(B.drone.values != T)[0]:
                imp.append(n_wrong(X[i], off, idx, w))
        gen += g_d
        per_drone[T] = dict(genuine_wrong_max=int(max(g_d)) if g_d else None,
                            genuine_wrong_mean=float(np.mean(g_d)) if g_d else None, n=len(g_d))
    gen, imp = np.array(gen), np.array(imp)
    by_t = {t: dict(frr=float(np.mean(gen > t)), far=float(np.mean(imp <= t))) for t in range(0, 6)}
    return dict(n_symbols=len(cols), genuine_hist=np.bincount(gen, minlength=len(cols) + 1).tolist(),
                impostor_hist=np.bincount(imp, minlength=len(cols) + 1).tolist(),
                genuine_max=int(gen.max()), impostor_min=int(imp.min()), margin=int(imp.min() - gen.max()),
                by_t=by_t, per_drone=per_drone, n_genuine=len(gen), n_impostor=len(imp))


# ── board stability over one week ───────────────────────────────────────────

def board_week(B: pd.DataFrame, names, board_cols) -> dict:
    """Enroll the board part from the 10-07/08 board sessions (same feature
    extraction, gravity-compensated), verify on the 10-13 boot sessions."""
    rows = []
    for s in sorted(glob.glob(str(ROOT / "data" / "fixed_rig_board" / "s*_drone*"))):
        m = json.load(open(os.path.join(s, "session.json"), encoding="utf-8"))
        bp = m.get("board_params") or {}
        if not bp.get("firmware"):
            continue
        try:
            x, info = kb.boot_features(s)
        except Exception:
            continue
        rows.append(dict(drone=m["drone"], day=m["created_iso"][:10], **dict(zip(names, x))))
    O = pd.DataFrame(rows)
    w = np.asarray(kb.system_params()["widths"], dtype=float)[[names.index(c) for c in board_cols]]
    out = {}
    for T in DRONES:
        o = O[O.drone == T][board_cols].values.astype(float)
        o = o[~np.isnan(o).any(axis=1)]
        off, idx = enroll_ref(o, w)
        g = [n_wrong(x, off, idx, w) for x in B[B.drone == T][board_cols].values.astype(float)]
        im = [n_wrong(x, off, idx, w) for x in B[B.drone != T][board_cols].values.astype(float)]
        shift = (np.nanmedian(B[B.drone == T][board_cols].values.astype(float), axis=0) - np.nanmedian(o, axis=0)) / w
        out[T] = dict(old_boots=int(len(o)), new_boots=len(g), genuine_wrong=g, impostor_min_wrong=int(min(im)),
                      shift_in_bins={c: round(float(v), 2) for c, v in zip(board_cols, shift)})
    return out


# ── motor speed over time ───────────────────────────────────────────────────

def motor_history(B: pd.DataFrame) -> pd.DataFrame:
    s = pd.read_csv(ROOT / "analysis" / "results_motor" / "motor_features.csv")
    s = s[s.purpose.str.startswith("sweep")]
    s["when"] = np.where(s.kind == "swap", "2026-09-30 swap " + s.stage.fillna(""), s.day)
    hist = s[(s.kind != "swap") | (s.stage == "pre")].groupby(["drone", "day", "motor"]).gain.median().unstack()
    hist.index = [f"{d} {day}" for d, day in hist.index]
    now = B.groupby(["drone", "day"])[[f"m{m}_58000_gain" for m in (1, 2, 3, 4)]].median()
    now.columns = [1, 2, 3, 4]
    now.index = [f"{d} {day} (boot)" for d, day in now.index]
    return pd.concat([hist, now]).sort_index().round(3)


def cross_day(A: pd.DataFrame, names, groups: dict) -> dict:
    """Enroll each drone on its first N_ENROLL boots of FIRST_DAY (as the
    keystore does), then replay every later boot of every drone in time order
    with the real fuzzy extractor:
      - genuine, no refresh: the day-1 helper as enrolled;
      - genuine, refresh: refresh_onboard after each accepted boot, the
        remaining day-1 boots included (what the drone would really do);
      - impostor: later boots of the other drones against both helpers.
    Wrong-symbol counts per group (motors / board / combined) against the
    enrollment reference show which part moved."""
    sysp = kb.system_params()
    fe = FuzzyExtractor(sysp["widths"], sysp["t"])
    w = np.asarray(sysp["widths"], dtype=float)
    A = A.sort_values("start")
    X = lambda df: df[names].values.astype(float)
    out = {}
    for T in DRONES:
        mine = A[A.drone == T]
        day1 = mine[mine.day == FIRST_DAY]
        if len(day1) < N_ENROLL:
            continue
        enr = day1.iloc[:N_ENROLL]
        ref = np.nanmedian(X(enr), axis=0)
        key, h0, _ = fe.gen(ref)
        off, idx = enroll_ref(X(enr), w)
        h = h0
        boots = []
        for _, r in mine.iloc[N_ENROLL:].iterrows():
            x = r[names].values.astype(float)
            k_static = fe.rep(x, h0)[0] == key
            k, W = fe.rep(x, h)
            if k == key:
                h = fe.refresh_onboard(h, W, x)
            if r.day == FIRST_DAY:
                continue
            boots.append(dict(session=int(r.session), day=r.day, accepted_no_refresh=bool(k_static),
                              accepted_refresh=bool(k == key),
                              **{f"wrong_{g}": n_wrong(x[sel], off[sel], idx[sel], w[sel])
                                 for g, sel in groups.items()}))
        later = A[(A.drone != T) & (A.day != FIRST_DAY)]
        imp = [dict(session=int(r.session), drone=r.drone,
                    accepted=bool(fe.rep(r[names].values.astype(float), h0)[0] == key
                                  or fe.rep(r[names].values.astype(float), h)[0] == key),
                    wrong_combined=n_wrong(r[names].values.astype(float), off, idx, w))
               for _, r in later.iterrows()]
        out[T] = dict(enrolled_sessions=[int(v) for v in enr.session], genuine=boots,
                      impostors_accepted=int(sum(i["accepted"] for i in imp)), n_impostors=len(imp),
                      impostor_min_wrong=min((i["wrong_combined"] for i in imp), default=None))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    B_all, names = load_boots()
    B_all.to_csv(OUT / "boot_features_real.csv", index=False)
    B = B_all[B_all.day == FIRST_DAY].reset_index(drop=True)
    motor_cols = [c for c in names if c.startswith("m")]
    board_cols = [c for c in names if not c.startswith("m")]
    res = {"n_boots": B.groupby("drone").size().to_dict()}

    q = B[["session", "drone", "temp", "vbat", "st_attempts", "rest_s", "n_stalls", "lost_s", "n_missing", "g_x", "g_y"]]
    res["quality"] = q.round(3).to_dict(orient="records")
    print(q.round(3).to_string(index=False))

    sep = []
    for c in names:
        wsd = B.groupby("drone")[c].std().mean()
        bsd = B.groupby("drone")[c].mean().std()
        sep.append(dict(feature=c, within=wsd, between=bsd, ratio=bsd / wsd if wsd else np.nan,
                        group="motor" if c.startswith("m") else "board"))
    S = pd.DataFrame(sep)
    res["separation"] = S.round(4).to_dict(orient="records")
    res["separation_median_ratio"] = S.groupby("group").ratio.median().round(1).to_dict()
    print("\nmedian between/within ratio:", res["separation_median_ratio"])

    res["key"] = {}
    for name, cols in (("motors_only", motor_cols), ("board_only", board_cols), ("combined", motor_cols + board_cols)):
        r = key_eval(B, names, cols)
        res["key"][name] = r
        print(f"\n{name}: genuine max wrong {r['genuine_max']}, impostor min wrong {r['impostor_min']}, "
              f"margin {r['margin']}  ({r['n_genuine']} genuine, {r['n_impostor']} impostor trials)")
        for t in range(0, 5):
            print(f"   t={t}: FRR={r['by_t'][t]['frr']:.3f} FAR={r['by_t'][t]['far']:.4f}")

    res["board_one_week"] = board_week(B, names, board_cols)
    print("\nboard, enrolled on 10-07/08 board sessions, verified on 10-13 boots:")
    for T, v in res["board_one_week"].items():
        print(f"   {T}: genuine wrong {v['genuine_wrong']}, impostor min {v['impostor_min_wrong']}")

    H = motor_history(B_all)
    res["motor_history"] = {k: v for k, v in H.to_dict(orient="index").items()}
    print("\nmotor speed gain over time:\n", H.to_string())

    b = B[B.drone == "B"][["session", "g_x", "oX"]]
    res["tilt_B"] = b.round(4).to_dict(orient="records")

    wb = np.asarray(kb.system_params()["widths"], dtype=float)
    sb = B.groupby("drone")[names].mean().std().values
    h = np.clip(np.log2(sb / wb) + 2.05, 0, None)
    res["entropy_bits_upper"] = dict(motors=float(np.nansum(h[:len(motor_cols)])),
                                     board=float(np.nansum(h[len(motor_cols):])),
                                     total=float(np.nansum(h)))
    print("\nentropy upper bound (bits):", {k: round(v, 1) for k, v in res["entropy_bits_upper"].items()})
    later_days = sorted(set(B_all.day) - {FIRST_DAY})
    if later_days:
        groups = {"motors": np.array([c in motor_cols for c in names]),
                  "board": np.array([c in board_cols for c in names]),
                  "combined": np.ones(len(names), dtype=bool)}
        res["cross_day"] = dict(days=later_days, per_drone=cross_day(B_all, names, groups))
        print(f"\ncross-day (enrolled {FIRST_DAY}, verified {', '.join(later_days)}):")
        for T, v in res["cross_day"]["per_drone"].items():
            g = v["genuine"]
            print(f"   {T}: {sum(b['accepted_no_refresh'] for b in g)}/{len(g)} accepted without refresh, "
                  f"{sum(b['accepted_refresh'] for b in g)}/{len(g)} with refresh; wrong motors "
                  f"{[b['wrong_motors'] for b in g]}, board {[b['wrong_board'] for b in g]}; "
                  f"impostors accepted {v['impostors_accepted']}/{v['n_impostors']}")
    (OUT / "boot_eval.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


if __name__ == "__main__":
    main()
