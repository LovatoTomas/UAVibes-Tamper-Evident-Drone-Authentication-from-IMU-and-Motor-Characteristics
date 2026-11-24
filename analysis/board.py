"""
Board (IMU, MEMS-PUF style) fingerprint analysis (2026-10-08).

Input: data/fixed_rig_board/sNN_drone{X}/session.json (+ converted rest
recording). One session = one boot.

Features
  self-test, rewritten per axis as
    d<ax> = (pos - neg) / 2   ~ sensitivity x electrostatic force (mass, springs, electrodes)
    o<ax> = (pos + neg) / 2   ~ offset during self-test (includes gravity: needs the same
                                boot orientation — level on the rig / on the ground)
  gyro bias (3), accelerometer scale (1)       — computed by the firmware at boot
  rest noise: std of each IMU channel over the 30 s motors-off recording (6)

Temperature: a pooled within-drone linear slope per feature (board temp read
at boot) is estimated and removed (values referred to 30 °C).

Key bits, MEMS-PUF style: each feature is quantised into 2^b equiprobable
intervals of an assumed Gaussian population (mean / sd of the drone means —
only 4 drones, a crude population model), Gray-coded. Enrollment value = the
drone's first-day median; every later boot is compared bit by bit.

Run:  python -m analysis.board
"""

import json
import glob

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis.dataset import ROOT

OUT = ROOT / "analysis" / "results_board"
SELFTEST = ["dX", "dY", "dZ", "oX", "oY", "oZ"]
BOOT = ["gBiasX", "gBiasY", "gBiasZ", "accScale"]
NOISE = ["n_acc.x", "n_acc.y", "n_acc.z", "n_gyro.x", "n_gyro.y", "n_gyro.z"]
T_REF = 30.0


def load() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(ROOT / "data" / "fixed_rig_board" / "s*_*" / "session.json")):
        m = json.load(open(f, encoding="utf-8"))
        bp = m.get("board_params") or {}
        if not bp.get("firmware"):
            continue
        r = dict(session=m["session"], drone=m["drone"], day=m["created_iso"][:10],
                 temp=bp.get("board_temp_c"), **{k: bp[k] for k in bp if k.startswith(("st", "gBias", "accScale"))})
        conv = [a for rec in m["recordings"] for a in rec["attempts"] if a.get("converted_path")]
        if conv:
            p = conv[-1]["converted_path"]
            d = pd.read_csv(ROOT / p if not p[1:3] == ":\\" else p)
            rest = d[d.phase == "rest_hold"]
            for c in ("acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"):
                r[f"n_{c}"] = float(rest[c].std())
        rows.append(r)
    D = pd.DataFrame(rows).sort_values("session").reset_index(drop=True)
    D["st_valid"] = ((D.stPosX != D.stNegX) & (D.stPosY != D.stNegY) & (D.stPosZ != D.stNegZ)
                     & ((D.stPosX.abs() + D.stNegX.abs()) > 15000))
    for a in "XYZ":
        D[f"d{a}"] = np.where(D.st_valid, (D[f"stPos{a}"] - D[f"stNeg{a}"]) / 2, np.nan)
        D[f"o{a}"] = np.where(D.st_valid, (D[f"stPos{a}"] + D[f"stNeg{a}"]) / 2, np.nan)
    return D


def temp_slopes(D: pd.DataFrame, feats) -> pd.Series:
    out = {}
    for f in feats:
        x = D.temp - D.groupby("drone").temp.transform("mean")
        y = D[f] - D.groupby("drone")[f].transform("mean")
        ok = x.notna() & y.notna()
        out[f] = float((x[ok] * y[ok]).sum() / (x[ok] ** 2).sum())
    return pd.Series(out)


def compensate(D: pd.DataFrame, feats, slopes: pd.Series) -> pd.DataFrame:
    C = D.copy()
    for f in feats:
        C[f] = D[f] - slopes[f] * (D.temp - T_REF)
    return C


def separation(D: pd.DataFrame, feats) -> pd.DataFrame:
    rows = []
    for f in feats:
        w = D.groupby("drone")[f].std().mean()
        b = D.groupby("drone")[f].mean().std()
        dd = D.groupby(["drone", "day"])[f].mean().unstack()
        shift = float((dd.iloc[:, -1] - dd.iloc[:, 0]).abs().mean()) if dd.shape[1] > 1 else np.nan
        rows.append(dict(feature=f, within_sd=w, between_sd=b, ratio=b / w, day_shift=shift,
                         bits_est=max(np.log2(b / w) - 2, 0)))     # -2: reliability margin, see report
    return pd.DataFrame(rows)


def gray(n: int) -> int:
    return n ^ (n >> 1)


def quantise(x: np.ndarray, mu: float, sd: float, bits: int) -> np.ndarray:
    edges = norm.ppf(np.arange(1, 2 ** bits) / 2 ** bits, loc=mu, scale=sd)
    return np.array([gray(int(np.searchsorted(edges, v))) for v in x])


def bit_errors(D: pd.DataFrame, feats, bits: int) -> dict:
    """Enrollment = first-day median per drone; returns per drone the
    fraction of wrong bits on every later-day boot, and other drones'
    distance under the same quantiser."""
    first = D.groupby("drone").day.min()
    enr = D[D.day.values == D.drone.map(first).values]
    mu, sd = enr.groupby("drone")[feats].median().mean(), enr.groupby("drone")[feats].median().std()

    def code(row):
        out = []
        for f in feats:
            v = row[f]
            if np.isnan(v):
                out += [None] * bits
                continue
            q = quantise(np.array([v]), mu[f], sd[f], bits)[0]
            out += [(q >> k) & 1 for k in range(bits)]
        return out

    def dist(a, b):
        p = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        return sum(x != y for x, y in p) / len(p) if p else np.nan

    res = {}
    ref = {T: code(enr[enr.drone == T][feats].median()) for T in enr.drone.unique()}
    for T in ref:
        later = D[(D.drone == T) & (D.day != first[T])]
        same = D[(D.drone == T) & (D.day == first[T])]
        res[T] = dict(
            n_bits=len(feats) * bits,
            same_day=[round(dist(ref[T], code(r)), 3) for _, r in same.iterrows()],
            later_days=[round(dist(ref[T], code(r)), 3) for _, r in later.iterrows()],
            others={O: round(float(np.nanmean([dist(ref[T], code(r)) for _, r in D[D.drone == O].iterrows()])), 3)
                    for O in ref if O != T})
    return res


def centred_bits(D: pd.DataFrame, feats, k: float = 8.0) -> dict:
    """Quantisation with per-device helper offsets (shielding functions,
    Linnartz & Tuyls 2003): bin width w = k * within-drone sd (incl. the
    day-to-day shift); at enrollment the helper stores the offset that puts
    the enrolled value at the CENTRE of its bin, so a later boot flips only
    if its noise exceeds w/2 = k/2 sd. The helper reveals the position
    inside the bin, not the bin index. Entropy of the index for a Gaussian
    population: H ~ log2(sigma_between / w) + 2.05 bits (differential
    entropy of the population, 4 drones -> crude)."""
    first = D.groupby("drone").day.min()
    enr = D[D.day.values == D.drone.map(first).values]
    sw = D.groupby("drone")[feats].std().mean()
    sb = D.groupby("drone")[feats].mean().std()
    w = k * sw
    out = {"bin_width": w.to_dict(), "entropy_bits": (np.log2(sb / w) + 2.05).clip(lower=0).to_dict(), "drones": {}}
    for T in enr.drone.unique():
        e = enr[enr.drone == T][feats].median()
        off = (np.floor(e / w) + 0.5) * w - e          # public helper
        idx_e = np.floor((e + off) / w)
        def flips(rows):
            """flipped features per boot; boots missing any feature (invalid
            self-test) are skipped — a real boot would retry the self-test"""
            r = []
            for _, row in rows.iterrows():
                idx = np.floor((row[feats].astype(float) + off) / w)
                if idx.isna().any() or idx_e.isna().any():
                    continue
                r.append(int((idx != idx_e).sum()))
            return r
        others = flips(D[D.drone != T])
        out["drones"][T] = dict(
            later_boots_flipped_features=flips(D[(D.drone == T) & (D.day != first[T])]),
            same_day_flipped_features=flips(D[(D.drone == T) & (D.day == first[T])]),
            other_drones_max_matching_features=int(len(feats) - min(others)) if others else None,
            other_drones_min_flipped_features=int(min(others)) if others else None)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = load()
    feats = SELFTEST + BOOT + [c for c in NOISE if c in D]
    slopes = temp_slopes(D, feats)
    C = compensate(D, feats, slopes)
    res = {"n_boots": D.groupby("drone").size().to_dict(),
           "n_valid_selftest": D.groupby("drone").st_valid.sum().astype(int).to_dict(),
           "temp_range": [float(D.temp.min()), float(D.temp.max())],
           "temp_slopes": slopes.to_dict(),
           "separation_raw": separation(D, feats).round(3).to_dict(orient="records"),
           "separation_tcomp": separation(C, feats).round(3).to_dict(orient="records"),
           "drone_means": C.groupby("drone")[SELFTEST].mean().round(1).to_dict(orient="index"),
           "within_corr_selftest": (C[SELFTEST] - C.groupby("drone")[SELFTEST].transform("mean")).corr().round(2).to_dict()}
    res["bit_errors"] = {}
    for name, fs in (("selftest", SELFTEST), ("selftest+boot", SELFTEST + BOOT),
                     ("selftest+boot+noise", feats)):
        for b in (2, 3, 4):
            res["bit_errors"][f"{name}_{b}bit"] = bit_errors(C, fs, b)
    for k in (6.0, 8.0, 10.0):
        res[f"centred_k{k:g}_selftest"] = centred_bits(C, SELFTEST, k)
        res[f"centred_k{k:g}_all"] = centred_bits(C, feats, k)
    (OUT / "board.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(separation(C, feats).round(2).to_string(index=False))
    for k, v in res["bit_errors"].items():
        print(k, {T: (x['n_bits'], np.nanmax(x['same_day'] + x['later_days']), np.nanmin(list(x['others'].values()))) for T, x in v.items()})
    for k in (6.0, 8.0, 10.0):
        for nm in ("selftest", "all"):
            r = res[f"centred_k{k:g}_{nm}"]
            print(f"centred k={k:g} {nm}: entropy~{sum(r['entropy_bits'].values()):.1f} bits; ",
                  {T: (max(v['later_boots_flipped_features'] + v['same_day_flipped_features']),
                       v['other_drones_max_matching_features']) for T, v in r['drones'].items()})
    return res


if __name__ == "__main__":
    main()
