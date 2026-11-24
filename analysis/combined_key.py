"""
Combined key from MOTORS + BOARD (2026-10-08): fuzzy-extractor simulation in
three configurations (motors only / board only / both), on real data.

Scheme (one symbol per feature, same machinery for both sources):
  - system parameters (fixed for the whole fleet, public): per feature the
    within-drone spread sigma_w and the bin width w = K * sigma_w, estimated on
    the ENROLLMENT data of all drones;
  - enrollment of a drone: reference value per feature (median of its first-
    day measurements); the public helper stores, per feature, the offset that
    puts the reference at the CENTRE of its bin (shielding functions,
    Linnartz & Tuyls 2003); the secret is the vector of bin indices;
  - reconstruction at boot: re-measure, apply the offsets, re-quantise; a
    code over symbols (e.g. Reed-Solomon: a syndrome sketch of 2t symbols)
    corrects up to t wrong symbols -> accepted iff #wrong symbols <= t;
  - drift: after every accepted boot the helper is recomputed from the median
    of the last three accepted measurements (refresh3; the secret recovered
    first, so the key itself never changes).

Measurements:
  motors  per session: speed gain + broadband floor (6 channels) of each
          single-motor 58k hold (sweep-m1..m4) -> 28 features; the only stage
          also present in the motor-swap sessions
  board   per boot: self-test sensitivity/offset (6, temperature-compensated),
          gyro bias y/z, accel scale, gyro rest noise x/y/z -> 12 features
Motors and board were recorded in different sessions, so a "boot" pairs a
motor session with a board boot of the same physical part (declared limit).

Scenarios: genuine later boots (chronological, with refresh), other drones,
motor transplant (motor-swap sessions: frame X + motors of Y, with X's board),
board swap (T's motors + another board), motors remounted ("post").
Run:  python -m analysis.combined_key
"""

import json
import itertools

import numpy as np
import pandas as pd

from analysis.dataset import ROOT
from analysis.motor_features import load as load_motor_features
from analysis.motor_identity import session_vectors
from analysis.keygen import session_starts
from analysis import board as boardmod

OUT = ROOT / "analysis" / "results_combined"
DRONES = ["A", "B", "C", "D"]
K = 8.0
BOARD_FEATS = ["dX", "dY", "dZ", "oX", "oY", "oZ", "gBiasY", "gBiasZ", "accScale",
               "n_gyro.x", "n_gyro.y", "n_gyro.z"]


# ── data ────────────────────────────────────────────────────────────────────

def motor_table() -> pd.DataFrame:
    V = session_vectors(load_motor_features(), "s58")
    V["session"] = V.session.astype(int)
    V["stage"] = V.stage.fillna("")
    starts = session_starts()
    V["start"] = [starts.get((r.drone, r.kind, r.session), "") for r in V.itertuples()]
    feats = [c for c in V.columns if c[:1] == "m" and c[1:2].isdigit() and c[2:3] == "_"]
    V[feats] = V[feats].astype(float)
    return V, feats


def board_table() -> pd.DataFrame:
    D = boardmod.load()
    feats = [f for f in BOARD_FEATS if f in D]
    slopes = boardmod.temp_slopes(D, feats)
    C = boardmod.compensate(D, feats, slopes)
    C = C[C.st_valid].copy()
    return C, feats


def params(df: pd.DataFrame, feats) -> "tuple[pd.Series, pd.Series]":
    sw = df.groupby("drone")[feats].std().mean()
    sb = df.groupby("drone")[feats].mean().std()
    return K * sw, (np.log2(sb / (K * sw)) + 2.05).clip(lower=0)


# ── fuzzy-extractor primitives (symbol level) ───────────────────────────────

def helper(ref: pd.Series, w: pd.Series, idx: "pd.Series | None" = None) -> dict:
    """Offsets that put `ref` at the centre of bin `idx`. At enrollment idx is
    the reference's own bin (the secret); at a refresh the SAME idx is kept,
    so tracking drift never changes the secret."""
    if idx is None:
        idx = np.floor(ref / w)
    return dict(off=(idx + 0.5) * w - ref, idx=idx)


def n_wrong(h: dict, x: pd.Series, w: pd.Series) -> int:
    idx = np.floor((x + h["off"]) / w)
    bad = (idx != h["idx"]) | idx.isna()
    return int(bad.sum())


class Source:
    def __init__(self, table, feats, w):
        self.t, self.f, self.w = table, feats, w


def run_config(srcs: "dict[str, Source]", t: int, starts_order) -> dict:
    """srcs: name -> Source; returns accept rates per scenario."""
    names = list(srcs)
    first = {n: {d: s.t[s.t.drone == d].day.min() for d in DRONES} for n, s in srcs.items()}

    def enroll_rows(n, d):
        s = srcs[n]
        e = s.t[(s.t.drone == d) & (s.t.day == first[n][d])]
        if "kind" in e:
            e = e[e.kind == "enroll"]
        return e

    def genuine_rows(n, d):
        s = srcs[n]
        r = s.t[(s.t.drone == d) & (s.t.day != first[n][d])]
        if n == "motors":
            r = r[(r.kind != "swap") | ((r.stage == "pre") & (r.motors == d))]
            r = r.sort_values("start")
        return r

    def total_wrong(hs, xs):
        return sum(n_wrong(hs[n], xs[n][srcs[n].f].astype(float), srcs[n].w) for n in names)

    res = {"genuine": {}, "impostor": {}, "board_swap": {}, "motor_transplant": {}, "motors_remounted": {}}
    helpers_final = {}
    for T in DRONES:
        refs = {n: enroll_rows(n, T)[srcs[n].f].median() for n in names}
        hs = {n: helper(refs[n], srcs[n].w) for n in names}
        gen = {n: genuine_rows(n, T) for n in names}
        L = max(len(g) for g in gen.values())
        states, outcomes, recent = [dict(hs)], [], {n: [refs[n]] for n in names}
        for i in range(L):
            xs = {n: gen[n].iloc[i % len(gen[n])] for n in names}
            wr = total_wrong(hs, xs)
            ok = wr <= t
            outcomes.append(dict(step=i, wrong=wr, accepted=ok))
            if ok:
                for n in names:
                    recent[n].append(xs[n][srcs[n].f].astype(float))
                    hs[n] = helper(pd.concat(recent[n][-3:], axis=1).median(axis=1), srcs[n].w, hs[n]["idx"])
                states.append(dict(hs))
        helpers_final[T] = dict(hs)
        res["genuine"][T] = dict(accept=float(np.mean([o["accepted"] for o in outcomes])),
                                 wrong=[o["wrong"] for o in outcomes])

        # other drones: every combination of their measurements, against every helper state of T
        others = {n: srcs[n].t[(srcs[n].t.drone != T) & ((srcs[n].t.kind != "swap") if "kind" in srcs[n].t else True)]
                  for n in names}
        acc = []
        for O in DRONES:
            if O == T:
                continue
            combos = itertools.product(*[others[n][others[n].drone == O].iterrows() for n in names])
            for combo in combos:
                xs = {n: r for n, (_, r) in zip(names, combo)}
                acc += [total_wrong(st, xs) <= t for st in states]
        res["impostor"][T] = float(np.mean(acc)) if acc else None

        # board swap: T's genuine motors + another drone's board
        if set(names) == {"motors", "board"}:
            acc = []
            for O in DRONES:
                if O == T:
                    continue
                for (_, m), (_, b) in itertools.product(gen["motors"].iterrows(),
                                                        srcs["board"].t[srcs["board"].t.drone == O].iterrows()):
                    acc.append(total_wrong(helpers_final[T], {"motors": m, "board": b}) <= t)
            res["board_swap"][T] = float(np.mean(acc))

    # motor transplant & remount (swap sessions; A and B only)
    if "motors" in names:
        mt = srcs["motors"].t
        sw = mt[mt.kind == "swap"]
        for (frame, mot), g in sw[sw.stage.isin(["swap", "post"])].groupby(["drone", "motors"]):
            stage = g.stage.iloc[0]
            for key_of in (frame, mot):
                acc = []
                boards = (srcs["board"].t[srcs["board"].t.drone == frame].iterrows()
                          if "board" in names else [(None, None)])
                for (_, m), (_, b) in itertools.product(g.iterrows(), list(boards)):
                    xs = {"motors": m}
                    if b is not None:
                        xs["board"] = b
                    acc.append(total_wrong(helpers_final[key_of], xs) <= t)
                tag = "motors_remounted" if stage == "post" else "motor_transplant"
                res[tag][f"frame{frame}+motors{mot} vs key {key_of}"] = float(np.mean(acc))
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    M, mf = motor_table()
    B, bf = board_table()
    wm, hm = params(M[M.kind == "enroll"], mf)
    wb, hb = params(B, bf)
    sources = {"motors": Source(M, mf, wm), "board": Source(B, bf, wb)}
    configs = {"motors_only": ["motors"], "board_only": ["board"], "motors+board": ["motors", "board"]}
    out = {"K": K, "entropy_per_feature": {"motors": hm.round(2).to_dict(), "board": hb.round(2).to_dict()}}
    for cname, names in configs.items():
        H = float(sum((hm if n == "motors" else hb).sum() for n in names))
        n_sym = sum(len(sources[n].f) for n in names)
        hbar = H / n_sym
        out[cname] = {"n_symbols": n_sym, "entropy_upper_bits": round(H, 1), "by_t": {}}
        for t in range(0, 9):
            r = run_config({n: sources[n] for n in names}, t, None)
            frr = 1 - np.mean([v["accept"] for v in r["genuine"].values()])
            far = np.mean([v for v in r["impostor"].values() if v is not None])
            r["summary"] = dict(t=t, genuine_reject=round(float(frr), 3), impostor_accept=round(float(far), 4),
                                residual_entropy_bits=round(max(H - 2 * t * hbar, 0), 1))
            out[cname]["by_t"][t] = r
            print(cname, r["summary"],
                  "| board swap", {k: round(v, 2) for k, v in r["board_swap"].items()},
                  "| transplant", {k: round(v, 2) for k, v in r["motor_transplant"].items()},
                  "| remount", {k: round(v, 2) for k, v in r["motors_remounted"].items()}, flush=True)
    (OUT / "combined_key.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
