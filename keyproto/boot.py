"""
Key generation from BOOT sessions (2026-10-09) — glue between the recorded
boot protocol (micro_sweep "boot-seq": 10 s rest + each motor alone at 58k)
and the fuzzy extractor + SPAKE2 prototype. Used by the dashboard's
"Identity check" tab. Same scheme as the drone (FW2) since 2026-10-29: no
robustness tag in the helper (it was an offline oracle); the station keeps
the helper's fingerprint (FuzzyExtractor.fingerprint), current and previous,
refuses a helper it does not know and binds the fingerprint into SPAKE2.

  data/keystore/system.json        fleet-wide public parameters: feature list,
                                   bin widths, temperature slopes, t
  data/keystore/station.json       ground-station database: drone -> PAKE
                                   password (would live on the station only),
                                   helper fingerprints
  data/keystore/drone{X}_helper.json  what the drone stores in its flash: the
                                   public helper data (no secret)

Features (same definitions as the studies):
  motors  m{i}_58000_gain, m{i}_58000_floor_<ch> from the first 3 s of each
          single-motor hold after a 0.5 s settle
  board   self-test sensitivity/offset (temperature-compensated to 30 °C),
          gyro bias y/z, accel scale, gyro rest noise x/y/z
"""

import json
from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from keyproto.fe import FuzzyExtractor
from keyproto import spake2

ROOT = Path(__file__).resolve().parent.parent
KEYSTORE = ROOT / "data" / "keystore"
BOOT_DIR = ROOT / "data" / "fixed_rig_boot"
K_WIDTH = 8.0
T_CORR = 2
SETTLE_S, HOLD_USE_S, BOOT_PWM = 0.5, 3.0, 58000
CHANNELS = ("acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z")
BOARD_FEATS = ["dX", "dY", "dZ", "oX", "oY", "oZ", "gBiasY", "gBiasZ", "accScale",
               "n_gyro.x", "n_gyro.y", "n_gyro.z"]
MOTOR_FEATS = [f"m{m}_{BOOT_PWM}_{f}" for m in (1, 2, 3, 4) for f in ["gain"] + [f"floor_{c}" for c in CHANNELS]]


def _jload(p, default=None):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def _jsave(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1), encoding="utf-8")


# ── system parameters (from the existing studies' data) ─────────────────────

def build_system_params() -> dict:
    from analysis import board as boardmod
    from analysis.boot_protocol import extract, motor_table_for
    D = boardmod.load()
    slopes = boardmod.temp_slopes(D, BOARD_FEATS)
    C = boardmod.compensate(D, BOARD_FEATS, slopes)
    C = C[C.st_valid]
    wb = K_WIDTH * C.groupby("drone")[BOARD_FEATS].std().mean()
    M, mf = motor_table_for(extract(), [BOOT_PWM], 3.0)
    M = M[M.kind == "enroll"]
    wm = K_WIDTH * M.groupby("drone")[MOTOR_FEATS].std().mean()
    sysp = dict(features=MOTOR_FEATS + BOARD_FEATS,
                widths=[float(v) for v in list(wm[MOTOR_FEATS].values) + list(wb[BOARD_FEATS].values)],
                temp_slopes={k: float(v) for k, v in slopes.items()}, t=T_CORR, K=K_WIDTH,
                note="bin widths = K x within-drone sd on the fixed-rig studies' enrollment data")
    _jsave(KEYSTORE / "system.json", sysp)
    return sysp


def system_params() -> dict:
    return _jload(KEYSTORE / "system.json") or build_system_params()


# ── feature extraction from one boot session folder ─────────────────────────

# Shaft-line search window (x base-model prediction). The studies used
# 0.85-1.15; the motors now run at 1.03-1.12 of the prediction, so the true
# line sat near the upper edge and a stronger unrelated peak could win
# (drone B boot s07, M2 read 0.86 instead of ~1.10). Re-centred on the
# observed fleet range.
F0_SEARCH = (0.92, 1.25)
GRAVITY_LSB = 32768 / 24      # accelerometer LSB per g at the self-test's 24 g range
MIN_HOLD_S = 1.5    # the SD deck occasionally stalls 0.2-0.7 s while writing
                    # (seen in the first boot sessions, 2026-10-13); a spectrum
                    # across a gap is wrong, so only the longest gap-free stretch
                    # of each hold is used, if at least this long


def _longest_contiguous(seg: pd.DataFrame, fs: float) -> pd.DataFrame:
    if len(seg) < 2:
        return seg
    gap = np.diff(seg["timestamp_s"].values) > 3.0 / fs
    starts = np.r_[0, np.where(gap)[0] + 1]
    ends = np.r_[np.where(gap)[0] + 1, len(seg)]
    k = int(np.argmax(ends - starts))
    return seg.iloc[starts[k]:ends[k]]


def boot_features(session_dir) -> "tuple[np.ndarray, dict]":
    from analysis.motor_features import hold_features
    from pipeline.preprocessing import load_session, infer_fs, highpass
    sysp = system_params()
    sd = Path(session_dir)
    m = _jload(sd / "session.json")
    bp = m.get("board_params") or {}
    conv = [a for r in m["recordings"] for a in r["attempts"] if a.get("converted_path")]
    if not conv:
        raise ValueError("session without a converted recording (the SD log was not uploaded)")
    p = Path(conv[-1]["converted_path"])
    df = load_session(p if p.is_absolute() else ROOT / p)
    fs = infer_fs(df)
    hp = highpass(df, fs)
    f = {}
    for mot in (1, 2, 3, 4):
        seg = hp[hp.phase == f"m{mot}_hold_{BOOT_PWM}"]
        part = _longest_contiguous(seg.iloc[int(SETTLE_S * fs):], fs).iloc[: int(HOLD_USE_S * fs)]
        r = hold_features(part, fs, mot, search=F0_SEARCH) if len(part) >= MIN_HOLD_S * fs else None
        for k in ["gain"] + [f"floor_{c}" for c in CHANNELS]:
            f[f"m{mot}_{BOOT_PWM}_{k}"] = (r or {}).get(k, np.nan)
    valid = (bp.get("stPosX") != bp.get("stNegX")) and bp.get("stResult") == 0
    T = bp.get("board_temp_c")
    # The self-test OFFSET o includes gravity, so it moves when the drone is
    # mounted at a slightly different tilt (drone B, 2026-10-13: remounted
    # ~2.7 deg off, oX moved by 80 LSB). Gravity is measured at the same boot
    # during the rest phase and removed: 1 g = 32768/24 LSB at the 24 g
    # self-test range. Removes most of the effect (80 -> ~16 LSB on B).
    rest = df[df.phase == "rest_hold"]
    g_lsb = {a: GRAVITY_LSB * float(rest[f"acc.{a.lower()}"].mean()) if len(rest) > 100 else np.nan
             for a in "XYZ"}
    for a in "XYZ":
        for name, val in ((f"d{a}", (bp.get(f"stPos{a}", np.nan) - bp.get(f"stNeg{a}", np.nan)) / 2),
                          (f"o{a}", (bp.get(f"stPos{a}", np.nan) + bp.get(f"stNeg{a}", np.nan)) / 2 - g_lsb[a])):
            f[name] = val - sysp["temp_slopes"][name] * (T - 30.0) if (valid and T is not None) else np.nan
    for k in ("gBiasY", "gBiasZ", "accScale"):
        v = bp.get(k, np.nan)
        f[k] = v - sysp["temp_slopes"][k] * (T - 30.0) if T is not None else v
    for c in ("gyro.x", "gyro.y", "gyro.z"):
        f[f"n_{c}"] = float(rest[c].std()) if len(rest) > 100 else np.nan
    x = np.array([f.get(k, np.nan) for k in sysp["features"]], dtype=float)
    return x, dict(drone=m["drone"], session=m["session"], board_temp_c=T, selftest_valid=valid,
                   n_missing=int(np.isnan(x).sum()))


def historical_reference(drone: str) -> np.ndarray:
    """Enrollment vector from the existing studies (first-day stepped 58k
    holds, 3 s, + first-day board boots), so the tab works before any boot
    session has been recorded."""
    from analysis import board as boardmod
    from analysis.boot_protocol import extract, motor_table_for
    sysp = system_params()
    M, _ = motor_table_for(extract(), [BOOT_PWM], 3.0)
    M = M[(M.drone == drone) & (M.kind == "enroll")]
    M = M[M.day == M.day.min()]
    D = boardmod.load()
    C = boardmod.compensate(D, BOARD_FEATS, pd.Series(sysp["temp_slopes"]))
    C = C[(C.drone == drone) & C.st_valid]
    C = C[C.day == C.day.min()]
    ref = {**M[MOTOR_FEATS].median().to_dict(), **C[BOARD_FEATS].median().to_dict()}
    return np.array([ref[k] for k in sysp["features"]], dtype=float)


# ── enrollment / boot ───────────────────────────────────────────────────────

def _fe():
    sysp = system_params()
    return FuzzyExtractor(sysp["widths"], sysp["t"], robust=False)


def enroll(drone: str, reference: np.ndarray, source: str) -> dict:
    key, helper, _ = _fe().gen(reference)
    helper["enrolled_from"] = source      # informative; not in the fingerprint
    station = _jload(KEYSTORE / "station.json", {})
    station[drone] = dict(password=key.hex(), failures=0,
                          helper_fp=FuzzyExtractor.fingerprint(helper), helper_fp_prev=None)
    _jsave(KEYSTORE / "station.json", station)
    _jsave(KEYSTORE / f"drone{drone}_helper.json", helper)
    return helper


def boot(drone: str, measurement: np.ndarray, max_failures: int = 5) -> dict:
    """Drone side: Rep -> SPAKE2 with the station -> speed alarm -> AES-GCM
    frame. On success the helper is refreshed on-board (offset history
    only); with a speed alarm access is denied and the helper is kept."""
    station = _jload(KEYSTORE / "station.json", {})
    helper = _jload(KEYSTORE / f"drone{drone}_helper.json")
    if drone not in station or helper is None:
        return dict(ok=False, stage="enrollment", reason="drone not enrolled")
    if station[drone]["failures"] >= max_failures:
        return dict(ok=False, stage="station", reason="station locked: too many failed attempts")
    fe = _fe()
    res = dict(n_symbols=fe.n, t=fe.t)
    entry = station[drone]
    fp = FuzzyExtractor.fingerprint(helper)
    if not entry.get("helper_fp"):
        # enrolled before 2026-10-29 (helper with a tag, no fingerprint in the
        # station): recorded at the first boot. In this PC simulation there is
        # no attacker in between; on the drone this is an explicit operator step.
        entry["helper_fp"], entry["helper_fp_prev"] = fp, None
        res["fingerprint_registered"] = True
    if fp not in (entry["helper_fp"], entry.get("helper_fp_prev")):
        entry["failures"] += 1
        _jsave(KEYSTORE / "station.json", station)
        return dict(res, ok=False, stage="station",
                    reason="helper data not recognised by the station (modified, or an old copy)")
    key, W = fe.rep(measurement, helper)
    if key is None:
        entry["failures"] += 1
        _jsave(KEYSTORE / "station.json", station)
        return dict(res, ok=False, stage="fuzzy extractor", reason=W)
    ident = drone.encode() + b"|" + bytes.fromhex(fp)     # the fingerprint is bound into SPAKE2
    A = spake2.Party("A", key, ident, b"station")
    B = spake2.Party("B", bytes.fromhex(entry["password"]), ident, b"station")
    ca, cb = A.finish(B.msg), B.finish(A.msg)
    if not (B.verify(ca) and A.verify(cb)):
        station[drone]["failures"] += 1
        _jsave(KEYSTORE / "station.json", station)
        return dict(res, ok=False, stage="PAKE", reason="key confirmation failed")
    station[drone]["failures"] = 0
    # diagnostics: how many symbols were wrong before correction
    from keyproto.fe import quantise, P
    q = quantise(measurement, helper["offsets"], helper["widths"])
    wrong = sum(((v if v is not None else 0) % P) != w for v, w in zip(q, W))
    res["wrong_symbols"] = int(wrong)
    # speed alarm: measured against the helper's reference BEFORE refreshing
    from keyproto.fe import gain_index, speed_alarm
    sysp = system_params()
    z = fe.shift_bins(helper, W, measurement) * sysp["K"]
    shift = {m: float(z[i]) for m, i in gain_index(sysp["features"]).items()}
    alarm = speed_alarm(shift)
    res.update(speed_shift_sigma=shift, speed_alarm=alarm, refreshed=not alarm)
    if alarm:
        # right key, altered mechanics: access denied (not a guessing failure),
        # reference kept, so the alarm repeats until re-enrollment
        station[drone]["speed_alarm"] = dict(at=datetime.now().isoformat(timespec="seconds"),
                                             motors=alarm, shift_sigma=shift)
        station[drone]["alarms"] = station[drone].get("alarms", 0) + 1
        _jsave(KEYSTORE / "station.json", station)
        return dict(res, ok=False, stage="speed_alarm",
                    reason="access denied: speed of motor(s) "
                           + ", ".join(f"M{m}" for m in alarm) + " changed (propellers/motors tampered with)")
    # no alarm: the reference follows the drift (offset history); the station
    # records the new fingerprint and keeps the one just used as the previous
    new = fe.refresh_onboard(helper, W, measurement)
    entry["helper_fp_prev"], entry["helper_fp"] = fp, FuzzyExtractor.fingerprint(new)
    _jsave(KEYSTORE / f"drone{drone}_helper.json", new)
    _jsave(KEYSTORE / "station.json", station)
    nonce = os.urandom(12)
    msg = f"encrypted telemetry from drone {drone}".encode()
    ct = AESGCM(A.session_key).encrypt(nonce, msg, None)
    pt = AESGCM(B.session_key).decrypt(nonce, ct, None)
    return dict(res, ok=True, stage="encrypted channel",
                session_key_prefix=A.session_key.hex()[:16], ciphertext_prefix=ct.hex()[:32],
                decrypted=pt.decode())
