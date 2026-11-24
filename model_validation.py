#!/usr/bin/env python3
"""
UAVibes — Validation session for the empirical motor model (motor_model.py).
============================================================================

Spins a motor at a ladder of PWM setpoints and, for each one, prints the
model's PREDICTION computed with the live battery voltage:

    shaft RPM        = 0.0934 * (PWM * vbat) + 4321
    tach display     = 2 * shaft RPM   (reflective foil on both blades)

You verify with the laser tachometer. Press Enter to just eyeball-confirm
and move on, or type the displayed value to have the error computed and
logged. A summary compares errors against the calibration RMSE (~324 RPM).

The default ladder is deliberately OFF-GRID from the calibration points
(which were 8k, 12k, 16k, ... 55k): validating at interpolated PWMs tests
the model, not the fit's memory of its own training points.

Random multi-motor mode (--random-n): instead of a fixed ladder for one
motor, each round redraws BOTH a random motor to read AND a random PWM
from a pool, with all four motors always spinning together (drive=all is
forced). This is deliberately unblocked/non-exhaustive — the point is to
sample the (motor, PWM) space cheaply (a handful of rounds instead of a
full 4-motors x N-PWMs grid) while every reading is taken under the same
realistic all-four-spinning current draw. Each round prints TWO
predictions side by side: the plain canonical model (M1-only, no
per-motor correction) and a per-motor MULTIPLICATIVE correction using
MOTOR_GAIN, imported from pipeline.motor_model (formalised there on
2026-09-02 from four sessions run with this exact script: 47 points,
all-four drive, PWM 35k-55k). That work already settled the
multiplicative-vs-additive question — a gain alone, no additive term,
fully explains the per-motor deviation over this range (residual scatter
after correction: 1.0-2.4 %, no leftover correlation with vbat or PWM).
This script's baseline-vs-corrected comparison each round is now mainly a
live sanity check that a new session still agrees with the formalised
gain, not an open research question.

The random-mode PWM pool is weighted toward the actual FLIGHT range
(35k-55k, the drone only leaves the ground above roughly 45k): the model
we ultimately need is the one that holds where the drone actually flies,
not the one that's easiest to hold a battery through. Because a fresh
--random-n session starts spinning all four motors directly at whatever
high PWM gets drawn first (no ramp, no low-PWM warm-up lap), use a fully
charged battery. --min-vbat (default 3.1V) stops the session automatically
if resting vbat sags below it between rounds, rather than silently keep
logging near-brownout points the way the 2026-09-02 all-drive sweep did
past ~40k PWM once its battery was already worn down.

Usage:
  python model_validation.py                    # M1, single drive, fixed ladder
  python model_validation.py --drive all        # 4 motors (restrained drone!)
  python model_validation.py --pwms 15000 33000 48000
  python model_validation.py --random-n 12      # randomized motor+PWM, flight-range pool
  python model_validation.py --random-n 12 --seed 7   # reproducible draw
  python model_validation.py --random-n 10 --pwm-pool 48000 50000 52000 55000  # flight-only
"""

import sys
import csv
import time
import random
import argparse
from datetime import datetime
from pathlib import Path

from crazyflie.connection import open_sync, radio_uri
from crazyflie.motors import MotorController
from pipeline.motor_model import pwm_to_rpm, N_BLADES, MOTOR_GAIN
from rpm_calibration import VbatMonitor

# Off-grid ladder: none of these PWMs was a calibration point.
DEFAULT_PWMS = [10_000, 14_000, 18_000, 22_000, 27_000,
                32_000, 37_000, 42_000, 47_000, 52_000]
SETTLE_S     = 3.0
CAL_RMSE_RPM = 324.0    # pooled-fit RMSE from the 2026-08-26 calibration

# MOTOR_GAIN is imported from pipeline.motor_model — formalised there on
# 2026-09-02 from exactly the four random-mode sessions this script
# produced (47 points, all-four drive, 35k-55k PWM). This script keeps
# reporting a "baseline vs corrected" comparison each round mainly as a
# live sanity check that new sessions still agree with the formalised gain.

# Random-mode PWM pool: weighted toward the actual FLIGHT range. The model
# we care about is the one the drone flies on, and the drone only leaves
# the ground above roughly 45k PWM — so most draws land at or above that,
# with a couple of mid-range points (35k/40k) kept only as a cheap sanity
# cross-check, not as the focus. Off-grid vs the 2026-08-26 calibration
# ladder (...,45k,50k,55k) throughout.
RANDOM_PWM_POOL = [35_000, 40_000, 45_000, 47_000, 49_000, 51_000, 53_000, 55_000]

# Under a sustained 4-motor draw at 45k-55k PWM, vbat WILL sag to roughly
# 3.4-3.7V from a resting 4.1-4.2V purely from normal LiPo internal
# resistance — this happens within the first round even on a full battery
# (observed 2026-09-02: 4.10V -> 3.57V after one 53k-PWM round) and is not
# a sign of a depleted cell. The 2026-09-02 all-drive session's erratic
# residuals happened specifically once vbat fell into the ~3.0-3.3V band —
# and that session also had the (now-fixed, see rpm_calibration.py) vbat-
# staleness bug likely amplifying the noise there. So the floor should sit
# near that lower band, not above the normal high-PWM sag zone: 3.5V would
# trip on almost every flight-range round regardless of battery health.
MIN_VBAT_V = 3.1

_DEFAULT_OUTPUT = Path("data") / "calibration"


def main() -> None:
    p = argparse.ArgumentParser(description="Motor-model validation via laser tachometer")
    p.add_argument("--uri", help="Explicit URI (default: radio 80/2M)")
    p.add_argument("--motor", type=int, default=1, choices=[1, 2, 3, 4],
                   help="Motor to read (default: 1)")
    p.add_argument("--drive", choices=["single", "all"], default="single",
                   help="single: only the read motor spins; all: all 4 spin "
                        "together (REQUIRES the drone to be restrained)")
    p.add_argument("--pwms", type=int, nargs="+", default=DEFAULT_PWMS,
                   help="PWM validation ladder (default: %(default)s — "
                        "off-grid from the calibration points)")
    p.add_argument("--random-n", type=int, default=None,
                   help="Run N randomized rounds instead of the fixed ladder: "
                        "each round redraws both the motor to read and the PWM "
                        "from --motors-pool / --pwm-pool. Forces --drive all "
                        "(all four motors always spinning together).")
    p.add_argument("--motors-pool", type=int, nargs="+", default=[1, 2, 3, 4],
                   choices=[1, 2, 3, 4],
                   help="Random mode: motors to draw from (default: 1 2 3 4)")
    p.add_argument("--pwm-pool", type=int, nargs="+", default=RANDOM_PWM_POOL,
                   help="Random mode: PWM values to draw from (default: %(default)s)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random mode: seed for reproducible draws (default: none)")
    p.add_argument("--min-vbat", type=float, default=MIN_VBAT_V,
                   help="Random mode: stop the session if resting vbat drops below "
                        "this before a round (default: %(default)s V) — avoids "
                        "collecting unreliable near-brownout points at high PWM")
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    args = p.parse_args()
    if args.random_n is not None:
        args.drive = "all"

    uri = args.uri or radio_uri(channel=80, datarate="2M", address="E7E7E7E7E7")
    print(f"Connecting to {uri} ...")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"model_validation_{datetime.now():%Y%m%d_%H%M%S}.csv"

    rows: list[dict] = []
    try:
        with open_sync(uri) as scf:
            print("Connected.")
            motors = MotorController(scf.cf)
            vmon   = VbatMonitor(scf.cf)
            time.sleep(1.5)

            print(f"\nBattery: {vmon.vbat:.2f} V")
            if args.random_n is not None:
                print(f"RANDOM mode: {args.random_n} rounds, drive=all always")
                print(f"  motor pool: {args.motors_pool}   pwm pool: {args.pwm_pool}")
                print(f"  min-vbat guard: {args.min_vbat:.2f} V (session stops early below this)")
                if vmon.vbat < args.min_vbat:
                    print(f"\n  WARNING: starting vbat {vmon.vbat:.2f} V is already below "
                          f"--min-vbat {args.min_vbat:.2f} V — use a freshly charged battery.")
                print("\nSAFETY: ALL FOUR motors will spin at flight-range PWM (up to full")
                print("collective thrust) — restrained drone only.")
            else:
                print(f"Reading M{args.motor}, drive mode: {args.drive}")
                print(f"Validation ladder ({len(args.pwms)} points): {args.pwms}")
                if args.drive == "all":
                    print("\nSAFETY: ALL FOUR motors will spin — restrained drone only.")
                else:
                    print("\nSAFETY: props ON, keep clear of all props.")
            input("Press Enter to start (Ctrl-C to abort)...")

            def _set_pwm(pwm: int) -> None:
                if args.drive == "all":
                    for m in (1, 2, 3, 4):
                        motors.set_motor(m, pwm)
                else:
                    motors.set_motor(args.motor, pwm)

            try:
                if args.random_n is not None:
                    rng = random.Random(args.seed)
                    for i in range(1, args.random_n + 1):
                        if vmon.vbat < args.min_vbat:
                            print(f"\n  vbat {vmon.vbat:.2f} V below --min-vbat "
                                  f"{args.min_vbat:.2f} V — stopping early to avoid "
                                  "unreliable near-brownout points. Swap the battery "
                                  "and rerun for more rounds.")
                            break
                        mid = rng.choice(args.motors_pool)
                        pwm = rng.choice(args.pwm_pool)
                        _set_pwm(pwm)
                        print(f"\n[{i}/{args.random_n}] all motors spinning at {pwm} PWM "
                              f"-> READ M{mid}")
                        time.sleep(SETTLE_S)

                        vbat = vmon.vbat
                        rpm_base = pwm_to_rpm(pwm, vbat)
                        rpm_mult = rpm_base * MOTOR_GAIN[mid]
                        tach_base = N_BLADES * rpm_base
                        tach_mult = N_BLADES * rpm_mult

                        print(f"  PREDICTED @ {vbat:.2f} V:")
                        print(f"    baseline (no per-motor correction): {tach_base:8.0f} tach")
                        print(f"    multiplicative M{mid} (k={MOTOR_GAIN[mid]:.3f}):    "
                              f"{tach_mult:8.0f} tach")

                        raw = input("  measured tach value (Enter=skip, q=quit): ").strip().lower()
                        if raw == "q":
                            raise KeyboardInterrupt
                        row = {
                            "timestamp_iso":     datetime.now().isoformat(timespec="seconds"),
                            "round":             i,
                            "motor":             mid,
                            "drive":             "all",
                            "pwm":               pwm,
                            "vbat_v":            round(vbat, 3),
                            "rpm_pred_base":     round(rpm_base, 1),
                            "rpm_pred_mult":     round(rpm_mult, 1),
                            "tach_measured":     "",
                            "rpm_measured":      "",
                            "error_base_rpm":    "",
                            "error_mult_rpm":    "",
                        }
                        if raw:
                            try:
                                measured = float(raw.replace(",", "."))
                                rpm_meas = measured / N_BLADES
                                err_base = rpm_meas - rpm_base
                                err_mult = rpm_meas - rpm_mult
                                row.update({
                                    "tach_measured":  measured,
                                    "rpm_measured":   round(rpm_meas, 1),
                                    "error_base_rpm": round(err_base, 1),
                                    "error_mult_rpm": round(err_mult, 1),
                                })
                                print(f"    error vs baseline: {err_base:+.0f} RPM "
                                      f"({err_base/rpm_base*100:+.1f}%)")
                                print(f"    error vs mult-corrected: {err_mult:+.0f} RPM "
                                      f"({err_mult/rpm_mult*100:+.1f}%)")
                            except ValueError:
                                print("    not a number — stored as skipped.")
                        rows.append(row)
                else:
                    for i, pwm in enumerate(args.pwms):
                        _set_pwm(pwm)
                        print(f"\n[{i+1}/{len(args.pwms)}] "
                              + ("all motors" if args.drive == "all" else f"M{args.motor}")
                              + f" spinning at {pwm} PWM")
                        time.sleep(SETTLE_S)

                        vbat = vmon.vbat
                        rpm_pred  = pwm_to_rpm(pwm, vbat)
                        tach_pred = N_BLADES * rpm_pred

                        print(f"  PREDICTED @ {vbat:.2f} V:")
                        print(f"    shaft RPM    : {rpm_pred:8.0f}   "
                              f"(+/- {CAL_RMSE_RPM:.0f} = calibration RMSE)")
                        print(f"    tach display : {tach_pred:8.0f}   "
                              f"({N_BLADES} blades -> expect {tach_pred-2*CAL_RMSE_RPM:.0f}"
                              f"-{tach_pred+2*CAL_RMSE_RPM:.0f} on the display)")

                        raw = input("  measured tach value (Enter=skip, q=quit): ").strip().lower()
                        if raw == "q":
                            raise KeyboardInterrupt
                        row = {
                            "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
                            "motor":         args.motor,
                            "drive":         args.drive,
                            "pwm":           pwm,
                            "vbat_v":        round(vbat, 3),
                            "rpm_predicted": round(rpm_pred, 1),
                            "tach_predicted": round(tach_pred, 1),
                            "tach_measured": "",
                            "rpm_measured":  "",
                            "error_rpm":     "",
                        }
                        if raw:
                            try:
                                measured = float(raw.replace(",", "."))
                                rpm_meas = measured / N_BLADES
                                err = rpm_meas - rpm_pred
                                row.update({
                                    "tach_measured": measured,
                                    "rpm_measured":  round(rpm_meas, 1),
                                    "error_rpm":     round(err, 1),
                                })
                                verdict = ("within RMSE" if abs(err) <= CAL_RMSE_RPM else
                                           "within 2xRMSE" if abs(err) <= 2*CAL_RMSE_RPM else
                                           "OUTSIDE 2xRMSE")
                                print(f"    error: {err:+.0f} RPM "
                                      f"({err/rpm_pred*100:+.1f}%)  -> {verdict}")
                            except ValueError:
                                print("    not a number — stored as skipped.")
                        rows.append(row)
            finally:
                motors.stop_all()
                vmon.stop()

    except KeyboardInterrupt:
        print("\nAborted — saving what was collected.")

    if not rows:
        print("Nothing collected.")
        sys.exit(0)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {out_path}")

    if args.random_n is not None:
        measured = [r for r in rows if r["error_mult_rpm"] != ""]
        if measured:
            base_errs = [abs(float(r["error_base_rpm"])) for r in measured]
            mult_errs = [abs(float(r["error_mult_rpm"])) for r in measured]
            within2_base = sum(1 for e in base_errs if e <= 2 * CAL_RMSE_RPM)
            within2_mult = sum(1 for e in mult_errs if e <= 2 * CAL_RMSE_RPM)
            print(f"\nValidation summary ({len(measured)} measured points, random motor+PWM draws):")
            print(f"  mean |error| baseline (no correction) : {sum(base_errs)/len(base_errs):6.0f} RPM")
            print(f"  mean |error| multiplicative-corrected : {sum(mult_errs)/len(mult_errs):6.0f} RPM")
            print(f"  within 2x calibration RMSE ({2*CAL_RMSE_RPM:.0f}): "
                  f"baseline {within2_base}/{len(measured)}   "
                  f"multiplicative {within2_mult}/{len(measured)}")

            print("\n  Per-motor signed residual AFTER the formalised MOTOR_GAIN correction")
            print("  (mean near 0% -> this session still agrees with the formalised gain;")
            print("  a consistent nonzero mean -> something has drifted since 2026-09-02")
            print("  (prop wear, a motor swap, ...) and MOTOR_GAIN may need re-fitting):")
            for mid in sorted({int(r["motor"]) for r in measured}):
                pts = [r for r in measured if int(r["motor"]) == mid]
                pct = [float(r["error_mult_rpm"]) / float(r["rpm_pred_mult"]) * 100 for r in pts]
                mean_pct = sum(pct) / len(pct)
                print(f"    M{mid}: n={len(pts):2d}  mean signed error = {mean_pct:+5.1f}% of predicted")

            if sum(mult_errs)/len(mult_errs) < sum(base_errs)/len(base_errs):
                print("\n  VERDICT: MOTOR_GAIN correction still reduces error vs baseline — "
                      "check the per-motor breakdown above for any new leftover bias.")
            else:
                print("\n  VERDICT: MOTOR_GAIN correction did NOT reduce error vs "
                      "baseline on this sample — something may have changed since the "
                      "2026-09-02 calibration; consider re-fitting MOTOR_GAIN.")
    else:
        measured = [r for r in rows if r["error_rpm"] != ""]
        if measured:
            errs = [abs(float(r["error_rpm"])) for r in measured]
            within1 = sum(1 for e in errs if e <= CAL_RMSE_RPM)
            within2 = sum(1 for e in errs if e <= 2*CAL_RMSE_RPM)
            print(f"\nValidation summary ({len(measured)} measured points):")
            print(f"  mean |error| : {sum(errs)/len(errs):6.0f} RPM")
            print(f"  max  |error| : {max(errs):6.0f} RPM")
            print(f"  within 1x calibration RMSE ({CAL_RMSE_RPM:.0f}): {within1}/{len(measured)}")
            print(f"  within 2x calibration RMSE ({2*CAL_RMSE_RPM:.0f}): {within2}/{len(measured)}")
            if within2 == len(measured):
                print("  VERDICT: model validated — all points within 2x RMSE.")
            else:
                print("  VERDICT: some points OUTSIDE 2x RMSE — inspect those rows; "
                      "if systematic, recalibrate (rpm_calibration.py) and update "
                      "pipeline/motor_model.py.")


if __name__ == "__main__":
    main()
