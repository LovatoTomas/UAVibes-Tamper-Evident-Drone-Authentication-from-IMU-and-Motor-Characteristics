#!/usr/bin/env python3
"""
UAVibes — Chassis tap test (impact excitation, motors OFF).
=============================================================

Classic modal-testing technique: a short, sharp tap excites ALL frequencies
at once (like striking a bell), and the structure "rings" back more
strongly at its own natural (resonant) frequencies than at any other —
regardless of what excited it. Since the motors are off for the whole
recording, ANY prominent peak found afterward is, by construction, NOT a
motor line — there is no motor forcing to produce one. This makes the tap
test a clean, independent cross-check against the candidate resonances
found in the motor recordings (103.4 / 125.9 / 162.9 / ~31 Hz as of
2026-09-07) — evidence not confounded by anything the motors could be
doing.

How to tap (see also the printed prompts during the run):
  - Where: NOT one point — the tip of each of the 4 arms (near the motor
    mount, farthest from the centre, so bending modes are well excited)
    PLUS the centre/top plate. A single, symmetric point can be a "node"
    for some mode shapes (doesn't move for that mode, so a tap there can't
    excite it at all) — spreading taps across 5 different points is what
    covers modes a single point might miss entirely.
  - With what: not a bare fingertip — too soft, acts like a low-pass
    filter on the impulse and starves exactly the 100-350 Hz range this
    is aimed at. A fingernail, a pen's plastic cap, a small screwdriver
    handle — anything hard but non-damaging. Tap and pull back
    immediately; don't let the tool rest against the frame afterward
    (that damps the very ringing you're trying to record).
  - How many: 5 taps per location, ~1.5s apart (enough for each impulse's
    ringing to die down before the next) — averaging several taps is what
    keeps the result from depending on any one imperfect tap.

No motor/timeline reconstruction needed — motors never move, so there is
no PWM to track. This script only needs to start/stop SD logging and pace
the tap prompts; the phase markers it writes are for bookkeeping (so you
can later tell which stretch of the recording corresponds to which tap
location) rather than for pipeline processing like the motor scripts.

Flow: run this -> tap along with the prompts -> power off -> pull card ->
  python usd_convert.py data/usd/logNN --timeline <timeline csv>
  python tap_test_analysis.py data/usd/logNN_1khz.csv
"""

import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

from crazyflie.connection import open_sync, radio_uri
from crazyflie.motors import MotorController
from enrollment.usd_timeline import Timeline

BASELINE_S     = 8.0     # undisturbed silence first — a reference "no excitation"
                         # segment; kept well above ~2s (nperseg=2048 @ 1kHz)
                         # so the Welch estimate is actually averaged, not a
                         # near-single-shot periodogram — a too-short baseline
                         # reads as spuriously huge, meaningless "peaks" purely
                         # from noise, the same estimation-noise lesson learned
                         # for HOLD_S elsewhere in this project.
TAPS_PER_POINT = 5
TAP_GAP_S      = 1.5     # pause between taps at the same point
POINT_GAP_S    = 2.0     # pause when moving to the next point
TAP_POINTS     = [
    ("M1 arm tip",      "tap_m1"),
    ("M2 arm tip",      "tap_m2"),
    ("M3 arm tip",      "tap_m3"),
    ("M4 arm tip",      "tap_m4"),
    ("centre/top plate", "tap_center"),
]

_DEFAULT_OUTPUT = Path("data") / "usd"


def main() -> None:
    p = argparse.ArgumentParser(description="Chassis tap test (motors off)")
    p.add_argument("--uri", help="Explicit URI (default: radio 80/2M)")
    p.add_argument("--taps-per-point", type=int, default=TAPS_PER_POINT)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    args = p.parse_args()

    uri = args.uri or radio_uri(channel=80, datarate="2M", address="E7E7E7E7E7")
    print(f"Connecting to {uri} ...")

    tl_path = args.output_dir / f"timeline_tap_{datetime.now():%Y%m%d_%H%M%S}.csv"
    total_s = (BASELINE_S
               + len(TAP_POINTS) * (args.taps_per_point * TAP_GAP_S + POINT_GAP_S))
    print(f"Protocol: baseline + {len(TAP_POINTS)} points x "
          f"{args.taps_per_point} taps -> ~{total_s:.0f}s (~{total_s/60:.1f} min)")

    with open_sync(uri) as scf:
        cf = scf.cf
        print("Connected.")

        def get_param(name: str) -> "str | None":
            try:
                return cf.param.get_value(name, timeout=3)
            except Exception as exc:
                print(f"  cannot read {name}: {exc}")
                return None

        if str(get_param("deck.bcUSD")) not in ("1", "1.0"):
            print("FAIL: SD deck not detected.")
            return
        if str(get_param("usd.canLog")) not in ("1", "1.0"):
            print("FAIL: deck present but not ready (card FAT32? config.txt? "
                  "card inserted before power-on?).")
            return
        print("Deck ready.")

        # Motors stay at 0 throughout — this recording's whole point is
        # having NO motor forcing to confound the tap response with. Still
        # instantiate the controller and force a stop, in case anything
        # left an override engaged from an earlier session.
        motors = MotorController(cf)
        motors.stop_all()

        print("\nMotors will stay OFF the entire recording — this is a pure")
        print("impact-excitation test. Have your tapping tool ready (see the")
        print("module docstring / the guidance already given for what to use).")
        input("Press Enter to start (Ctrl-C to abort)...")

        tl = Timeline(tl_path)

        try:
            cf.param.set_value("usd.logging", "1")
            tl.log("usd_start")
            print("\nOnboard logging STARTED.")

            tl.log("phase", "baseline")
            print(f"\nHold still — {BASELINE_S:.0f}s of undisturbed baseline...")
            time.sleep(BASELINE_S)

            for label, phase_tag in TAP_POINTS:
                tl.log("phase", phase_tag)
                print(f"\n=== Tap the {label} ===")
                for i in range(1, args.taps_per_point + 1):
                    print(f"  tap {i}/{args.taps_per_point} now — "
                          f"(short, sharp, pull back immediately)")
                    time.sleep(TAP_GAP_S)
                print(f"  done with {label}, moving on...")
                time.sleep(POINT_GAP_S)

            tl.log("phase", "stop")

        finally:
            motors.stop_all()
            try:
                cf.param.set_value("usd.logging", "0")
                tl.log("usd_stop")
                print("\nOnboard logging STOPPED.")
            except Exception as exc:
                print(f"WARNING: could not stop logging cleanly: {exc}")
            tl.close()
            time.sleep(0.5)

    print(f"\nTimeline saved -> {tl_path}")
    print("\nNext steps:")
    print("  1. power off, pull the microSD card, copy the newest logNN into data/usd/")
    print(f"  2. python usd_convert.py data/usd/logNN --timeline {tl_path}")
    print("  3. python tap_test_analysis.py data/usd/logNN_1khz.csv")


if __name__ == "__main__":
    main()
