#!/usr/bin/env python3
"""
UAVibes — Analysis of a chassis tap-test recording (from tap_test.py).
=========================================================================

No motor forcing during a tap-test recording (motors stay at 0 the whole
time), so unlike usd_motor_analysis.py there is nothing to exclude —
every prominent peak found here is, by construction, NOT a motor line.
Cross-check its frequencies against the candidates found in the motor
recordings (usd_motor_analysis.py's Part C, e.g. 103.4 / 125.9 / 162.9 /
~31 Hz as of 2026-09-07) — a genuine structural resonance should show up
in both, from two independent excitation sources (motors vs a hand tap).

Also reports the undisturbed baseline segment separately: a candidate
that's just as strong at baseline (nothing touching the drone) as during
active tapping is suspect — likely an electrical/sensor artifact rather
than something the taps actually excited.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from usd_motor_analysis import segment_peaks, KNOWN_CANDIDATES


def main() -> None:
    p = argparse.ArgumentParser(description="Tap-test recording analysis")
    p.add_argument("csvfile", type=Path, help="converted 1 kHz CSV (usd_convert.py output)")
    args = p.parse_args()

    df = pd.read_csv(args.csvfile, dtype={"phase": str})
    dt = np.diff(df["timestamp_s"].values)
    dt = dt[(dt > 0) & (dt < 0.1)]
    fs = 1.0 / np.median(dt)
    print(f"{args.csvfile.name}: {len(df):,} samples, fs ~ {fs:.0f} Hz\n")

    tap_phases = [ph for ph in df["phase"].unique()
                  if str(ph).startswith("tap_")]
    if not tap_phases:
        print("No tap_* phases found — was this recorded with tap_test.py "
              "and converted with --timeline?")
        return
    print(f"Tap phases found: {tap_phases}\n")

    print("=" * 76)
    print("TAP RESPONSE — top spectral peaks (no motor forcing, nothing to exclude)")
    print("=" * 76)

    whole = df[df["phase"].isin(tap_phases)]
    cand = segment_peaks(whole, fs, fmin=15.0, prominence_db=4.0, return_prominence=True)
    cand.sort(key=lambda c: -c[1])
    print(f"\nAll tap locations combined ({len(whole):,} samples):")
    for pk, prom in cand[:15]:
        known = next((k for k in KNOWN_CANDIDATES if abs(pk - k) <= 3.0), None)
        tag = f"  <-- matches known candidate {known} Hz" if known else ""
        print(f"  {pk:7.1f} Hz   {prom:5.1f} dB{tag}")
    if not cand:
        print("  (nothing above the prominence threshold)")

    print("\nPer tap-location breakdown (top 6 by prominence, for context — "
          "which point excited what):")
    for ph in tap_phases:
        seg = df[df["phase"] == ph]
        if len(seg) < 500:
            print(f"  {ph:12s}: (too short, skipped)")
            continue
        c = segment_peaks(seg, fs, fmin=15.0, prominence_db=4.0, return_prominence=True)
        c.sort(key=lambda x: -x[1])
        top = [f"{pk:.1f}Hz({prom:.0f}dB)" for pk, prom in c[:6]]
        print(f"  {ph:12s}: {', '.join(top) if top else '(nothing above threshold)'}")

    baseline = df[df["phase"] == "baseline"]
    if len(baseline) > 500:
        print(f"\nBaseline comparison ({len(baseline):,} undisturbed samples, "
              "nothing touching the drone):")
        print("A genuinely tap-excited peak should be much weaker or absent here "
              "— if it's just as strong at baseline, that's a red flag it's an "
              "electrical/sensor artifact, not a structural response to the taps.")
        cb = segment_peaks(baseline, fs, fmin=15.0, prominence_db=3.0, return_prominence=True)
        cb.sort(key=lambda x: -x[1])
        for pk, prom in cb[:8]:
            near_tap = any(abs(pk - tpk) <= 2.0 for tpk, _ in cand)
            flag = "  <-- also strong during tapping, see above" if near_tap else ""
            print(f"  {pk:7.1f} Hz   {prom:5.1f} dB{flag}")
        if not cb:
            print("  (nothing above threshold — clean baseline, good sign)")


if __name__ == "__main__":
    main()
