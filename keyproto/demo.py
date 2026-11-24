"""
End-to-end demonstration on the real recordings (2026-10-08).

  enrollment  station measures the drone (first-day motor sessions + board
              boots), runs Gen, stores the secret-derived password, writes
              the public helper to the drone
  boot        drone measures (a motor session + a board boot of the same
              physical parts, see combined_key.py for the pairing limit),
              runs Rep -> password, SPAKE2 with the station, then an
              AES-GCM message on the session key; on success the helper is
              refreshed (drift tracking, robust: re-tagged)
  attacks     other drones; foreign board with the right motors; motor
              transplant (motor-swap sessions); motors remounted; tampered
              helper; online password guessing; offline brute force cost

Run:  python -m keyproto.demo        -> analysis/results_proto/demo.json
"""

import itertools
import json
import math
import os
import time

import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from analysis.dataset import ROOT
from analysis import combined_key as ck
from keyproto.fe import FuzzyExtractor, P
from keyproto import spake2

OUT = ROOT / "analysis" / "results_proto"
T_CORR = 2
DRONES = ["A", "B", "C", "D"]


class Station:
    def __init__(self):
        self.db = {}            # drone id -> password secret (from Gen)
        self.failures = {}

    def enroll(self, drone, key):
        self.db[drone] = key
        self.failures[drone] = 0

    def session(self, drone, drone_party_factory, max_failures=5):
        """Runs SPAKE2 + key confirmation + one encrypted message."""
        if self.failures.get(drone, 0) >= max_failures:
            return False, "locked (too many failed attempts)"
        B = spake2.Party("B", self.db[drone], drone.encode(), b"station")
        A = drone_party_factory()
        if A is None:
            self.failures[drone] = self.failures.get(drone, 0) + 1
            return False, "drone could not reconstruct its secret"
        confA = A.finish(B.msg)
        confB = B.finish(A.msg)
        if not (B.verify(confA) and A.verify(confB)):
            self.failures[drone] += 1
            return False, "key confirmation failed"
        nonce = os.urandom(12)
        ct = AESGCM(A.session_key).encrypt(nonce, b"telemetry frame 0001", None)
        pt = AESGCM(B.session_key).decrypt(nonce, ct, None)
        self.failures[drone] = 0
        return pt == b"telemetry frame 0001", "ok"


def vec(row_m, row_b, mf, bf):
    parts = []
    if row_m is not None:
        parts.append(row_m[mf].astype(float).values)
    if row_b is not None:
        parts.append(row_b[bf].astype(float).values)
    return np.concatenate(parts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    M, mf = ck.motor_table()
    B, bf = ck.board_table()
    wm, hm = ck.params(M[M.kind == "enroll"], mf)
    wb, hb = ck.params(B, bf)
    widths = np.concatenate([wm.values, wb.values])
    fe = FuzzyExtractor(widths, T_CORR)
    station = Station()
    firstM = {d: M[M.drone == d].day.min() for d in DRONES}
    firstB = {d: B[B.drone == d].day.min() for d in DRONES}
    log = {"params": dict(n_symbols=fe.n, t=T_CORR, field=P, motors_features=len(mf), board_features=len(bf),
                          entropy_upper_bits=round(float(hm.sum() + hb.sum()), 1)),
           "boots": {}, "attacks": {}}
    helpers, Ws = {}, {}

    for d in DRONES:
        em = M[(M.drone == d) & (M.day == firstM[d]) & (M.kind == "enroll")]
        eb = B[(B.drone == d) & (B.day == firstB[d])]
        ref = np.concatenate([em[mf].median().values, eb[bf].median().values])
        key, helper, _ = fe.gen(ref)
        station.enroll(d, key)
        helpers[d] = helper

        gm = M[(M.drone == d) & (M.day != firstM[d]) & ((M.kind != "swap") | ((M.stage == "pre") & (M.motors == d)))]
        gm = gm.sort_values("start")
        gb = B[(B.drone == d) & (B.day != firstB[d])]
        recent, results = [ref], []
        for i in range(max(len(gm), len(gb))):
            x = vec(gm.iloc[i % len(gm)], gb.iloc[i % len(gb)], mf, bf)
            out = {}

            def party():
                k, W = fe.rep(x, helpers[d])
                out["W"], out["reason"] = (W, None) if k is not None else (None, W)
                return spake2.Party("A", k, d.encode(), b"station") if k is not None else None
            ok, msg = station.session(d, party)
            results.append(dict(boot=i, ok=ok, msg=msg if ok else (out.get("reason") or msg)))
            if ok:
                recent.append(x)
                helpers[d] = fe.refresh(helpers[d], out["W"], recent[-3:])
        log["boots"][d] = dict(accepted=sum(r["ok"] for r in results), total=len(results),
                               failures=[r for r in results if not r["ok"]])
        print(d, log["boots"][d]["accepted"], "/", log["boots"][d]["total"], flush=True)

    def attempt(target, x):
        k, _ = fe.rep(x, helpers[target])
        if k is None:
            return False
        A = spake2.Party("A", k, target.encode(), b"station")
        Bp = spake2.Party("B", station.db[target], target.encode(), b"station")
        ca, cb = A.finish(Bp.msg), Bp.finish(A.msg)
        return Bp.verify(ca)

    # other drones (all measurement combinations) against each target
    acc = {}
    for T in DRONES:
        n_ok = n = 0
        for O in DRONES:
            if O == T:
                continue
            for (_, m), (_, b) in itertools.product(M[(M.drone == O) & (M.kind != "swap")].iterrows(),
                                                    B[B.drone == O].iterrows()):
                n += 1
                n_ok += attempt(T, vec(m, b, mf, bf))
        acc[T] = f"{n_ok}/{n}"
    log["attacks"]["other_drones"] = acc
    # foreign board, genuine motors
    fb = {}
    for T in DRONES:
        gm = M[(M.drone == T) & (M.kind != "swap")]
        n_ok = n = 0
        for O in DRONES:
            if O == T:
                continue
            for (_, m), (_, b) in itertools.product(gm.iterrows(), B[B.drone == O].iterrows()):
                n += 1
                n_ok += attempt(T, vec(m, b, mf, bf))
        fb[T] = f"{n_ok}/{n}"
    log["attacks"]["foreign_board"] = fb
    # motor transplant / remount (swap sessions, with the frame's own board)
    sw = M[(M.kind == "swap") & M.stage.isin(["swap", "post"])]
    tr = {}
    for (frame, mot, stage), g in sw.groupby(["drone", "motors", "stage"]):
        for key_of in sorted({frame, mot}):
            n_ok = n = 0
            for (_, m), (_, b) in itertools.product(g.iterrows(), B[B.drone == frame].iterrows()):
                n += 1
                n_ok += attempt(key_of, vec(m, b, mf, bf))
            tr[f"{stage}: frame {frame} + motors {mot} -> key {key_of}"] = f"{n_ok}/{n}"
    log["attacks"]["motor_swap_sessions"] = tr
    # tampered helper (shift the offsets of 3 board features by one bin)
    d = "A"
    h = dict(helpers[d])
    off = list(h["offsets"])
    for i in range(len(mf), len(mf) + 3):
        off[i] += widths[i]
    h["offsets"] = off
    gb = B[(B.drone == d)].iloc[-1]
    gm = M[(M.drone == d) & (M.kind != "swap")].sort_values("start").iloc[-1]
    xg = vec(gm, gb, mf, bf)
    res_t = {}
    res_t["untouched helper"] = "accepted" if fe.rep(xg, helpers[d])[0] is not None else fe.rep(xg, helpers[d])[1]
    res_t["3 offsets shifted by one bin"] = fe.rep(xg, h)[1] if fe.rep(xg, h)[0] is None else "ACCEPTED"
    h1 = dict(helpers[d]); o1 = list(h1["offsets"]); o1[len(mf)] += widths[len(mf)]; h1["offsets"] = o1
    res_t["1 offset shifted by one bin (within error correction)"] = fe.rep(xg, h1)[1] if fe.rep(xg, h1)[0] is None else "ACCEPTED"
    h2 = dict(helpers[d]); s2 = list(h2["sketch"]); s2[0] = (s2[0] + 1) % P; h2["sketch"] = s2
    res_t["1 sketch symbol changed"] = fe.rep(xg, h2)[1] if fe.rep(xg, h2)[0] is None else "ACCEPTED"
    log["attacks"]["tampered_helper"] = res_t
    # online guessing: station locks after 5 failures
    st2 = Station()
    st2.enroll("A", station.db["A"])
    tries = [st2.session("A", lambda: spake2.Party("A", os.urandom(32), b"A", b"station"))[1] for _ in range(7)]
    log["attacks"]["online_guessing"] = tries
    # offline brute force cost: one guess = quantised vector -> tag check
    W_true = fe.rep(vec(gm, B[B.drone == d].iloc[-1], mf, bf), helpers[d])[1]
    t0, n = time.perf_counter(), 2000
    for _ in range(n):
        fe._tag(W_true, helpers[d])
    per = (time.perf_counter() - t0) / n
    Hu = float(hm.sum() + hb.sum())
    leak_worst = 2 * T_CORR * math.log2(max(1, float(np.max(np.concatenate([
        (6 * M.groupby("drone")[mf].mean().std() / wm).values,
        (6 * B.groupby("drone")[bf].mean().std() / wb).values])))))
    log["attacks"]["offline_bruteforce"] = dict(
        seconds_per_guess_python=per, entropy_upper_bits=round(Hu, 1),
        sketch_leak_avg_bits=round(2 * T_CORR * Hu / fe.n, 1), sketch_leak_worst_bits=round(leak_worst, 1),
        years_single_core_at_upper_bound=per * 2 ** (Hu - 2 * T_CORR * Hu / fe.n) / 3.15e7,
        years_single_core_worst_case=per * 2 ** max(Hu - leak_worst, 0) / 3.15e7)
    (OUT / "demo.json").write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
    print(json.dumps(log["attacks"], indent=1, default=str))


if __name__ == "__main__":
    main()
