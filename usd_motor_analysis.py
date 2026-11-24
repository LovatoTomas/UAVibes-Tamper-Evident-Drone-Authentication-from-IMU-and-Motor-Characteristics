#!/usr/bin/env python3
"""
UAVibes — Analysis of a per-motor SD recording (from test_usd_motors.py).
=========================================================================

Input: the converted 1 kHz CSV (usd_convert.py output, phases included).

Produces three answers:

A. PER-MOTOR SHAFT LINES — for every m<id>_hold_<pwm> segment, finds the
   spectral line nearest the motor model's prediction and reports the
   measured shaft frequency / RPM and the deviation from the model.
   A consistently negative deviation on M1 only would confirm the
   tachometer-foil drag hypothesis; the per-motor spread quantifies how
   wide an adaptive comb filter must reach.

B. ALL-FOUR DECOMPOSITION — splits each all_hold_<pwm> segment's spectrum
   into 4 per-motor lines via attribute_motor_lines(): the validated
   per-motor model predicts a narrow search window per motor, and a small
   greedy assignment resolves any overlap (adjacent motors can differ by
   under 1% in gain). This is NOT a human-in-the-loop step — with 4
   comparably-strong, closely-spaced lines there's no reliable way for a
   human to tell them apart from one spectrum by eye; the disambiguating
   information only exists in the model's prediction and in whether an
   attribution holds up across PWM levels (the monotonicity check here,
   same logic as Part A's).

C. STRUCTURAL RESONANCE CANDIDATES — Campbell logic: motor lines move
   with PWM, chassis modes do not. Peaks that persist at a fixed frequency
   across most segments (different PWMs, single- AND all-drive), after
   excluding every attributed motor line (all four for an all-drive
   segment, via Part B — excluding only one averaged line would leave the
   other three in the pool to be mistaken for resonances) and its
   harmonics, are structural candidates. Known candidates from the first
   1 kHz recording (103.4, 125.9, 162.9, 333.7 Hz) are flagged when
   re-found.
"""

import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks

import sys
sys.path.insert(0, str(Path(__file__).parent))
from pipeline.motor_model import pwm_to_hz

CHANNELS = ["acc.x", "acc.z", "gyro.x", "gyro.z"]
KNOWN_CANDIDATES = [103.4, 125.9, 162.9, 333.7]   # from the 2026-08-27 recording

_HOLD_RE = re.compile(r"^m(\d)_hold_(\d+)$")
_ALL_RE  = re.compile(r"^all_hold_(\d+)$")

# Candidates within this window are scored, not just nearest-picked. This
# matters because a search window wide enough to catch real motor-to-motor
# RPM spread (which can be several %) can also brush against a structural
# resonance line (synthetic-data validation caught exactly this: a ~163 Hz
# resonance got picked instead of a ~193 Hz motor line whose ±15% window
# nearly reached it). Fixed 2026-08-27 (log02): the scorer used to rank
# candidates purely by proximity-to-prediction plus flat bonuses/penalties
# (harmonic support +0.15, known-resonance -0.25) with NO weight on how
# strong a peak actually is. On M1/M3 @24000 this handed the pick to a
# 6-10 dB whisper 20+ Hz off target instead of the genuine motor line
# sitting right at ~38-43 dB prominence, 30+ dB louder and much closer to
# the prediction — the flat harmonic bonus alone was enough to flip the
# decision. Prominence (dB) is now the PRIMARY score; proximity/resonance/
# harmonic are small dB-scale adjustments that only matter as tie-breakers
# between comparably strong candidates (which is the scenario they were
# designed for: weak, ambiguous excitation at low PWM, not a dominant
# clean line). Shared with ui/motor_calibration.py's human-in-the-loop
# candidate proposals — same scoring, same tuning, one place to fix it.
SEARCH_PCT           = 0.15   # candidate window: +/- 15% of predicted f0
RESONANCE_GUARD      = 3.0    # Hz — candidates this close to a KNOWN_CANDIDATE are penalised
HARMONIC_TOL         = 4.0    # Hz — tolerance for the 2nd-harmonic support check
PROXIMITY_DB_PER_PCT = 0.5    # dB penalty per 1% relative deviation from prediction
RESONANCE_PENALTY_DB = 5.0
HARMONIC_BONUS_DB    = 3.0


def score_candidate(pk: float, prom_db: float, f0_pred: float,
                    all_peaks: "list[float]", fs: float) -> float:
    """Prominence-led score for one candidate peak — see constants above."""
    score = prom_db
    score -= PROXIMITY_DB_PER_PCT * abs(pk - f0_pred) / f0_pred * 100
    if any(abs(pk - k) <= RESONANCE_GUARD for k in KNOWN_CANDIDATES):
        score -= RESONANCE_PENALTY_DB
    h2 = 2 * pk
    h2_eff = h2 if h2 < fs / 2 else fs - h2       # allow an aliased 2nd harmonic
    if any(abs(p - h2_eff) <= HARMONIC_TOL for p in all_peaks):
        score += HARMONIC_BONUS_DB
    return score


def rank_candidates(seg: pd.DataFrame, fs: float, f0_pred: float,
                    search_pct: float = SEARCH_PCT) -> "list[tuple[float, float, float]]":
    """
    Spectral peaks near f0_pred, best-first by score_candidate.

    Returns a list of (freq_hz, prominence_db, score), sorted descending by
    score — empty if nothing was found within +/- search_pct of f0_pred.
    """
    cand = segment_peaks(seg, fs, return_prominence=True)
    all_peaks = [pk for pk, _ in cand]
    near = [(pk, prom) for pk, prom in cand
            if (1 - search_pct) * f0_pred <= pk <= (1 + search_pct) * f0_pred]
    scored = [(pk, prom, score_candidate(pk, prom, f0_pred, all_peaks, fs))
              for pk, prom in near]
    scored.sort(key=lambda c: -c[2])
    return scored


ATTRIB_SEARCH_PCT = 0.10   # narrower than SEARCH_PCT (0.15): rank_candidates
                           # is used to DISCOVER where a line is when we
                           # don't fully trust the prediction yet; this is
                           # used to APPLY an already-validated per-motor
                           # model, where a wide window would just readmit
                           # the ambiguity this function exists to remove.
                           # 10%, not tighter, because the model has never
                           # been directly validated in the all-drive regime
                           # (it's calibrated single-drive) — some residual
                           # slack is honest, not sloppy.


def attribute_motor_lines(
    seg: pd.DataFrame, fs: float, pwm: int, vbat: "float | None",
    search_pct: float = ATTRIB_SEARCH_PCT,
) -> "tuple[dict[int, tuple[float, float, float] | None], dict[int, float]]":
    """
    Split an all-drive segment's spectrum into 4 per-motor lines, using the
    per-motor motor model as a trusted prior instead of asking a human to
    tell 4 similarly-strong, closely-spaced peaks apart by eye (which
    doesn't work here — unlike a single-motor segment, there's no way to
    visually confirm "this peak belongs to M3" from one spectrum alone;
    that information only exists in how the model predicts each motor's
    OWN line to sit, and in how consistent an attribution is across PWM
    levels — the same thing a human would have to reconstruct anyway, just
    less reliably. See usd_motor_analysis module docstring, Part C.)

    Each motor gets its own narrow search window around the model's
    prediction; when two windows overlap (adjacent motors can differ by
    under 1% in gain) a candidate peak is claimed by at most one motor via
    a greedy highest-score-first assignment, not by 4 independent
    searches that could double-claim the same peak.

    Returns (assigned, predicted):
      assigned  — {motor_id: (freq, prominence_db, score) or None}
      predicted — {motor_id: predicted_freq_hz}, always populated (used by
                  callers even for motors that came back None, e.g. to
                  report "expected here, nothing found").
    """
    predicted = {m: pwm_to_hz(pwm, vbat, motor_id=m) for m in (1, 2, 3, 4)}
    all_cand = segment_peaks(seg, fs, return_prominence=True)
    all_peaks = [pk for pk, _ in all_cand]

    per_motor_candidates: "dict[int, list[tuple[float, float, float]]]" = {}
    for m, f0 in predicted.items():
        near = [(pk, prom) for pk, prom in all_cand
                if (1 - search_pct) * f0 <= pk <= (1 + search_pct) * f0]
        scored = [(pk, prom, score_candidate(pk, prom, f0, all_peaks, fs))
                  for pk, prom in near]
        scored.sort(key=lambda c: -c[2])
        per_motor_candidates[m] = scored

    assigned: "dict[int, tuple[float, float, float] | None]" = {}
    claimed: "set[float]" = set()
    remaining = set(predicted)
    while remaining:
        best = None   # (motor, freq, prom, score)
        for m in remaining:
            for pk, prom, score in per_motor_candidates[m]:
                if pk in claimed:
                    continue
                if best is None or score > best[3]:
                    best = (m, pk, prom, score)
                break   # candidates are pre-sorted best-first per motor
        if best is None:
            break
        m, pk, prom, score = best
        assigned[m] = (pk, prom, score)
        claimed.add(pk)
        remaining.discard(m)
    for m in predicted:
        assigned.setdefault(m, None)

    return assigned, predicted


def segment_peaks(seg: pd.DataFrame, fs: float,
                  fmin: float = 20.0, prominence_db: float = 6.0,
                  return_prominence: bool = False):
    """
    Union of prominent spectral peaks across CHANNELS, merged within 1.5 Hz.

    nperseg is capped at 2048 (not the full segment) so Welch actually
    averages multiple windows instead of computing one near-single-shot
    periodogram — that averaging is what keeps peak-location noise small
    when several comparable-amplitude tones sit close together (motor
    line + harmonics + structural resonances). 2048 @ 1 kHz still gives
    ~0.5 Hz resolution, far finer than the few-Hz precision we need.

    Returns a plain ``list[float]`` of merged peak frequencies, or, when
    return_prominence=True, a ``list[(freq, max_prominence_db)]`` — the
    prominence lets a caller rank/cap peaks instead of treating every
    detection as equally significant (needed to keep short/noisy segments
    from flooding a cross-segment vote with dozens of weak peaks).
    """
    hits: list[tuple[float, float]] = []   # (freq, prominence_db)
    for ch in CHANNELS:
        if ch not in seg.columns:
            continue
        x = seg[ch].values.astype(float)
        x = x - x.mean()
        nper = min(2048, len(x))
        f, psd = welch(x, fs=fs, nperseg=nper)
        psd_db = 10 * np.log10(np.maximum(psd, 1e-20))
        band = f >= fmin
        pk, props = find_peaks(psd_db[band], prominence=prominence_db, distance=5)
        for idx, prom in zip(pk, props["prominences"]):
            hits.append((float(f[band][idx]), float(prom)))

    hits.sort(key=lambda h: h[0])
    merged: list[list[tuple[float, float]]] = []
    for h in hits:
        if merged and h[0] - merged[-1][-1][0] <= 1.5:
            merged[-1].append(h)
        else:
            merged.append([h])

    if return_prominence:
        return [(float(np.mean([f for f, _ in g])), max(p for _, p in g))
                for g in merged]
    return [float(np.mean([f for f, _ in g])) for g in merged]


# ── Reusable analysis core (shared by this script's CLI and the dashboard's
# Resonances tab — one implementation of the algorithm, two presentations) ──

def collect_segments(df: pd.DataFrame) -> "tuple[list[dict], list[dict]]":
    """
    Split a converted CSV into single-motor (m<id>_hold_<pwm>) and
    all-drive (all_hold_<pwm>) hold segments by phase name. Segments too
    short for a usable Welch estimate (2000 samples single, 1000 all-drive)
    are dropped.
    """
    single_segs: "list[dict]" = []
    all_segs: "list[dict]" = []
    for phase in df["phase"].unique():
        m = _HOLD_RE.match(str(phase))
        if m:
            seg = df[df["phase"] == phase]
            if len(seg) >= 2000:
                single_segs.append({"motor": int(m.group(1)), "pwm": int(m.group(2)), "seg": seg})
            continue
        m = _ALL_RE.match(str(phase))
        if m:
            seg = df[df["phase"] == phase]
            if len(seg) >= 1000:
                all_segs.append({"pwm": int(m.group(1)), "seg": seg})
    single_segs.sort(key=lambda s: (s["motor"], s["pwm"]))
    all_segs.sort(key=lambda s: s["pwm"])
    return single_segs, all_segs


def attribute_single_drive(single_segs: "list[dict]", fs: float) -> "list[dict]":
    """
    Part A's core: for every single-motor hold segment, the model's
    prediction, the best-matching measured line (if any), and the
    deviation between them. One row per segment, in single_segs' order.

    Row keys: motor, pwm, vbat, f0_pred, f0_meas (None if no candidate was
    found), dev_pct (None likewise), n_candidates (how many peaks were in
    the search window — >1 means the pick had competition, worth flagging
    to a reader).
    """
    rows: "list[dict]" = []
    for s in single_segs:
        seg = s["seg"]
        vbat = float(seg["vbat_v"].mean()) if "vbat_v" in seg.columns else None
        f0_pred = pwm_to_hz(s["pwm"], vbat)
        ranked = rank_candidates(seg, fs, f0_pred)
        row = {"motor": s["motor"], "pwm": s["pwm"], "vbat": vbat,
               "f0_pred": f0_pred, "n_candidates": len(ranked)}
        if ranked:
            f0_meas = ranked[0][0]
            row["f0_meas"] = f0_meas
            row["dev_pct"] = (f0_meas - f0_pred) / f0_pred * 100
        else:
            row["f0_meas"] = None
            row["dev_pct"] = None
        rows.append(row)
    return rows


def single_drive_motor_lines(rows: "list[dict]") -> "dict[tuple, float]":
    """{(motor, pwm): f0_meas} from attribute_single_drive()'s rows, matched-only."""
    return {(r["motor"], r["pwm"]): r["f0_meas"] for r in rows if r["f0_meas"] is not None}


def attribute_all_drive(all_segs: "list[dict]", fs: float) -> "dict[int, tuple[dict, dict]]":
    """Part B's core: {pwm: (assigned, predicted)} via attribute_motor_lines()."""
    all_line_attribution: "dict[int, tuple[dict, dict]]" = {}
    for a in all_segs:
        seg = a["seg"]
        if len(seg) < 2000:
            continue
        vbat = float(seg["vbat_v"].mean()) if "vbat_v" in seg.columns else None
        assigned, predicted = attribute_motor_lines(seg, fs, a["pwm"], vbat)
        all_line_attribution[a["pwm"]] = (assigned, predicted)
    return all_line_attribution


def build_vote_segs(
    single_segs: "list[dict]", all_segs: "list[dict]",
    motor_lines: "dict[tuple, float]",
    all_line_attribution: "dict[int, tuple[dict, dict]]",
) -> "list[dict]":
    """
    Package single- and all-drive segments, each with the motor line(s) it
    should have excluded, for find_resonance_candidates(). An all-drive
    segment carries up to 4 exclude_f0s (one per attributed motor) instead
    of a single averaged one — excluding only the average, as an earlier
    version did, left 3 of the 4 real motor lines in the candidate pool to
    be mistaken for resonances.
    """
    return (
        [{"seg": s["seg"],
          "exclude_f0s": ([motor_lines[(s["motor"], s["pwm"])]]
                          if (s["motor"], s["pwm"]) in motor_lines else [])}
         for s in single_segs]
        + [{"seg": s["seg"],
            "exclude_f0s": [hit[0] for hit in all_line_attribution.get(
                s["pwm"], ({}, {}))[0].values() if hit is not None]}
           for s in all_segs if s["pwm"] in all_line_attribution]
    )


def find_resonance_candidates(
    vote_segs: "list[dict]", fs: float,
    known_candidates: "list[float]" = KNOWN_CANDIDATES,
) -> dict:
    """
    Part C's core: spectral peaks that persist across many segments AFTER
    excluding each segment's own motor line(s) and harmonics — Campbell-
    diagram logic (motor lines move with PWM, resonances don't) as a
    cross-segment vote instead of eyeballing one spectrum.

    vote_segs can span MULTIPLE recording sessions concatenated together
    (the caller decides what "a segment" is) — a resonance confirmed
    across independent sessions (different days, different battery levels)
    is much stronger evidence than one confirmed only within a single
    session's own segments.

    Returns {"n_segs", "known": [...], "other": [...],
    "per_seg_candidates": [...]}. "known" checks each of known_candidates
    directly. "other" is best-effort discovery of new candidates via fixed
    3 Hz bins — deliberately non-chaining (plain floor division, not
    proximity clustering): an earlier proximity-clustering version
    collapsed a dense pool of weak peaks into one giant nonsense cluster
    with impossible vote counts (2026-08-27, log01).
    """
    per_seg_candidates: "list[list[float]]" = []
    for s in vote_segs:
        cand = segment_peaks(s["seg"], fs, prominence_db=4.0, return_prominence=True)
        excl: "list[float]" = []
        for f0 in s["exclude_f0s"]:
            for k in (1, 2):
                fk = k * f0
                excl.append(fk if fk < fs / 2 else fs - fk)   # aliased harmonic
        kept = [(pk, prom) for pk, prom in cand if all(abs(pk - e) > 8.0 for e in excl)]
        kept.sort(key=lambda h: -h[1])
        per_seg_candidates.append([pk for pk, _ in kept[:12]])

    n_segs = len(vote_segs)
    known_results = []
    for k in known_candidates:
        hits = sum(1 for seg_pk in per_seg_candidates if any(abs(pk - k) <= 3.0 for pk in seg_pk))
        known_results.append({
            "freq": k, "hits": hits, "n_segs": n_segs,
            "reproduced": hits >= max(2, n_segs // 4),
        })

    votes: "dict[int, list[float]]" = {}
    for seg_pk in per_seg_candidates:
        for pk in seg_pk:
            b = int(pk // 3.0)
            votes.setdefault(b, []).append(pk)
    ranked = sorted(votes.items(), key=lambda kv: -len(kv[1]))
    other_results = []
    for b, hits in ranked:
        if len(hits) < max(3, n_segs // 3):
            continue
        fmean = float(np.mean(hits))
        if any(abs(fmean - k) <= 3.0 for k in known_candidates):
            continue
        other_results.append({"freq": fmean, "hits": len(hits), "n_segs": n_segs})

    return {"n_segs": n_segs, "known": known_results, "other": other_results,
            "per_seg_candidates": per_seg_candidates}


def main() -> None:
    p = argparse.ArgumentParser(description="Per-motor SD recording analysis")
    p.add_argument("csvfile", type=Path, help="converted 1 kHz CSV (usd_convert.py output)")
    args = p.parse_args()

    # Picks up whichever model is active in the dashboard (if any dashboard
    # session has ever run on this machine) instead of always falling back
    # to motor_model.py's hardcoded literal defaults — this analysis is
    # exactly the kind of thing that should reflect a deliberately-switched
    # model, e.g. comparing a vibration-based refit against the laser one.
    # Read-only in the common case (models already exist on disk); only
    # writes anything the very first time ever, to seed the two originals.
    from pipeline.motor_model import ensure_bootstrapped, get_active_model
    ensure_bootstrapped()
    _active = get_active_model()
    print(f"(using motor model: {_active.name} — {_active.description})")

    df = pd.read_csv(args.csvfile, dtype={"phase": str})
    dt = np.diff(df["timestamp_s"].values)
    dt = dt[(dt > 0) & (dt < 0.1)]
    fs = 1.0 / np.median(dt)
    print(f"{args.csvfile.name}: {len(df):,} samples, fs ~ {fs:.0f} Hz\n")

    # ── collect hold segments ────────────────────────────────────────────────
    # all_segs is a LIST: test_usd_motors.py can step the all-four condition
    # through a whole PWM ladder (not just one top value), giving a
    # decomposition and a Campbell resonance check at every level.
    single_segs, all_segs = collect_segments(df)

    if not single_segs:
        print("No m<id>_hold_<pwm> segments found — was this recorded with "
              "test_usd_motors.py and converted with --timeline?")
        return

    # ── A. per-motor shaft lines ─────────────────────────────────────────────
    print("=" * 76)
    print("A. PER-MOTOR SHAFT LINES  (measured vs motor-model prediction)")
    print("=" * 76)
    print(f"{'seg':>12} {'vbat':>6} {'f0 pred':>8} {'f0 meas':>8} "
          f"{'delta':>7} {'RPM meas':>9}")

    a_rows = attribute_single_drive(single_segs, fs)
    motor_lines = single_drive_motor_lines(a_rows)
    per_motor_dev: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: []}

    for r in a_rows:
        label = f"M{r['motor']}@{r['pwm']}"
        vbat_txt = f"{r['vbat']:.2f}" if r["vbat"] is not None else "  nan"
        if r["f0_meas"] is None:
            print(f"{label:>12} {vbat_txt:>6} {r['f0_pred']:>8.1f} "
                  f"{'--':>8} {'--':>7}  (no line within +/-{SEARCH_PCT*100:.0f}% of prediction)")
            continue
        per_motor_dev[r["motor"]].append(r["dev_pct"])
        flag = " *" if r["n_candidates"] > 1 else ""
        print(f"{label:>12} {vbat_txt:>6} {r['f0_pred']:>8.1f} "
              f"{r['f0_meas']:>8.1f} {r['dev_pct']:>+6.1f}% {r['f0_meas']*60:>9.0f}{flag}")
    print("  (* = multiple candidates were in range; picked by prominence-led "
          "score, proximity/harmonic/resonance as tie-breakers)")

    # Monotonicity self-check: for a fixed motor, measured f0 should increase
    # with PWM. A violation is a strong signal that a pick grabbed the wrong
    # peak (most likely a nearby resonance) rather than a genuine RPM dip.
    print("\nMonotonicity check (measured f0 should rise with PWM, per motor):")
    for mid in (1, 2, 3, 4):
        seq = [(pwm, motor_lines[(mid, pwm)]) for pwm in
               sorted(p for (m, p) in motor_lines if m == mid)]
        if len(seq) < 2:
            continue
        bad = [(seq[i][0], seq[i+1][0]) for i in range(len(seq)-1)
               if seq[i+1][1] <= seq[i][1]]
        status = "OK" if not bad else f"VIOLATED between PWM {bad}"
        print(f"  M{mid}: {[round(f,1) for _, f in seq]}  -> {status}")

    print("\nPer-motor mean deviation from the model (calibrated on M1 with foil):")
    devs = {m: np.mean(v) for m, v in per_motor_dev.items() if v}
    for m, d in sorted(devs.items()):
        note = "  <- foil-carrying calibration motor" if m == 1 else ""
        print(f"  M{m}: {d:+.1f}%{note}")
    if len(devs) >= 2:
        spread = max(devs.values()) - min(devs.values())
        print(f"  motor-to-motor spread: {spread:.1f}% "
              f"(an adaptive comb must cover +/-{spread/2:.1f}% around the mean)")

    # ── B. all-four decomposition ─────────────────────────────────────────────
    # Automated 4-way attribution, not a human judging peaks: with 4 motors
    # of comparable strength overlapping in frequency, there's no reliable
    # way to tell "this peak is M3" from a single spectrum by eye — that
    # information only exists in (a) how the validated per-motor model
    # predicts each motor's own line, and (b) whether an attribution stays
    # consistent across PWM levels. attribute_motor_lines() does (a); the
    # monotonicity check below does (b). See module docstring.
    all_line_attribution = attribute_all_drive(all_segs, fs)
    if all_segs:
        print("=" * 76)
        print(f"B. ALL-FOUR DECOMPOSITION  ({len(all_segs)} PWM level(s))")
        print("=" * 76)
        for a in all_segs:
            if a["pwm"] not in all_line_attribution:
                print(f"\n@ {a['pwm']} PWM: segment too short ({len(a['seg'])} samples), skipped")
                continue
            assigned, predicted = all_line_attribution[a["pwm"]]
            vbat = float(a["seg"]["vbat_v"].mean()) if "vbat_v" in a["seg"].columns else None
            print(f"\n@ {a['pwm']} PWM  (vbat {vbat or float('nan'):.2f} V)")
            for m in (1, 2, 3, 4):
                pred = predicted[m]
                hit = assigned[m]
                if hit is None:
                    print(f"  M{m}: predicted {pred:>6.1f} Hz -> no match within "
                          f"+/-{ATTRIB_SEARCH_PCT*100:.0f}%")
                else:
                    f, prom, score = hit
                    dev = (f - pred) / pred * 100
                    flag = "  (weak/ambiguous match)" if score < 5.0 else ""
                    print(f"  M{m}: predicted {pred:>6.1f} Hz -> matched {f:>6.1f} Hz "
                          f"({dev:+5.1f}%, {prom:4.1f}dB prominence){flag}")

        # Same monotonicity logic as Part A, applied to the attributed
        # all-drive lines — a violation here means an attribution grabbed
        # the wrong peak (most likely because two motors' windows
        # overlapped and the greedy assignment picked the weaker option
        # for one of them), not a genuine RPM dip.
        if len(all_line_attribution) >= 2:
            print("\nMonotonicity check (attributed f0 should rise with PWM, per motor):")
            for mid in (1, 2, 3, 4):
                seq = [(pwm, all_line_attribution[pwm][0][mid][0])
                       for pwm in sorted(all_line_attribution)
                       if all_line_attribution[pwm][0][mid] is not None]
                if len(seq) < 2:
                    continue
                bad = [(seq[i][0], seq[i+1][0]) for i in range(len(seq)-1)
                       if seq[i+1][1] <= seq[i][1]]
                status = "OK" if not bad else f"VIOLATED between PWM {bad}"
                print(f"  M{mid}: {[round(f,1) for _, f in seq]}  -> {status}")

    # ── C. fixed-frequency (structural) candidates ───────────────────────────
    print("\n" + "=" * 76)
    print("C. STRUCTURAL RESONANCE CANDIDATES  (fixed across PWMs and motors)")
    print("=" * 76)

    # Include the all-four segments too when available: they are a stronger,
    # differently-shaped excitation than any single motor and are the
    # condition closest to the very first 1 kHz recording that surfaced
    # the KNOWN_CANDIDATES — a resonance confirmed under BOTH excitation
    # types is much better evidence than either alone. test_usd_motors.py
    # can record several all-four PWM levels, each contributing here.
    #
    # Each vote_seg carries its own exclude_f0s: one line for a single-
    # drive segment, up to four for an all-drive one (from Part B's
    # attribution above) — excluding only the averaged f0 for an all-drive
    # segment, as an earlier version of this did, left 3 of the 4 real
    # motor lines in the candidate pool to be mistaken for resonances.
    vote_segs = build_vote_segs(single_segs, all_segs, motor_lines, all_line_attribution)
    result = find_resonance_candidates(vote_segs, fs)
    n_segs = result["n_segs"]

    # (1) Direct, robust check: does each already-known candidate reappear?
    print(f"Known-candidate check ({n_segs} segments incl. all-four hold "
          f"if present, tolerance +/-3 Hz):")
    for k in result["known"]:
        print(f"  {k['freq']:>7.1f} Hz : confirmed in {k['hits']}/{n_segs} segments"
              + ("  <-- reproduced" if k["reproduced"] else ""))

    # (2) Best-effort discovery of NEW candidates.
    print(f"\nOther recurring peaks (fixed 3 Hz bins, top-12-per-segment pool):")
    print(f"{'freq (Hz)':>10} {'segments':>9}   note")
    for o in result["other"][:12]:
        print(f"{o['freq']:>10.1f} {o['hits']:>6}/{n_segs}")
    if not result["other"]:
        print("  (nothing new beyond the known candidates)")


if __name__ == "__main__":
    main()
