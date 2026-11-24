#!/usr/bin/env python3
"""
UAVibes — Empirical PWM -> RPM calibration via laser tachometer.
================================================================

Interactive session: the script drives ONE motor at a time through a
ladder of PWM setpoints; at each step you aim the laser tachometer at the
spinning prop and type in the displayed value. Battery voltage (pm.vbat)
is shown and logged with every reading, since motor RPM at a given PWM
depends on it.

Reflective foil is on BOTH blades of the two-blade prop, so the
tachometer displays 2x the true rotation rate: the script divides the
entered value by 2 and stores both raw and corrected values.

The default ladder (8k..55k, 11 points) spans the full usable range with
even spacing — enough to reveal whether the map is linear or curved.
The final extra point repeats 30000 PWM so the same setpoint is measured
at the start-of-session and end-of-session battery level, quantifying
the vbat effect directly.

Output CSV columns:
  timestamp_iso, motor, pwm, tach_displayed, rpm, vbat_v, note

Drive modes (--drive):
  single (default)  only the motor being read spins. No liftoff possible,
                    easiest laser aiming, cleanest per-motor map.
  all               all four motors spin at the same PWM; you read EVERY
                    motor in --motors (default all four) at each step
                    before the PWM changes. Captures the battery-sag
                    effect of 4x current draw — in real flight all motors
                    run together, so RPM at a given PWM is expected to be
                    LOWER than in single mode. CAUTION: with props on, the
                    drone produces full collective thrust above ~40000 PWM;
                    only use this mode with the drone restrained (cage rig).

Motor order (--order, single-drive with 2+ motors only):
  interleaved (default)  at each PWM level, read M1 then M2 then M3 then
                    M4 (whichever are in --motors) before moving to the
                    next PWM level. Any session drift (battery sag,
                    warm-up) affects all motors nearly equally at each
                    comparison point, instead of confounding with test
                    order — this was shown to matter (2026-08-27: an
                    unrelated peak-picking bug in the spectral method had
                    briefly looked like exactly this kind of order-linked
                    drift, until the bug was found and the effect turned
                    out to be a genuine, reproducible per-motor RPM
                    difference; interleaving here rules the ambiguity out
                    by construction rather than by post-hoc analysis).
  block             finish the whole ladder for M1, then M2, then M3, then
                    M4. Faster to run (aim the laser once per motor block
                    instead of once per reading) but confounds per-motor
                    differences with session time.

Usage:
  python rpm_calibration.py                          # M1 only
  python rpm_calibration.py --motors 1 2 3 4          # interleaved by PWM level (recommended)
  python rpm_calibration.py --motors 1 2 3 4 --order block
  python rpm_calibration.py --drive all               # 4 motors spin, read all 4 at each step
  python rpm_calibration.py --pwms 10000 20000 30000
  python rpm_calibration.py --uri usb://0
"""

import sys
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path

from crazyflie.connection import open_sync, radio_uri, get_first_uri
from crazyflie.motors import MotorController
from cflib.crazyflie.log import LogConfig

# Ladder: even coverage 8k-55k. Below ~8000 the coreless motors spin
# unreliably (stiction); 55000 is a single motor, no liftoff possible.
DEFAULT_PWMS = [8_000, 12_000, 16_000, 20_000, 25_000, 30_000,
                35_000, 40_000, 45_000, 50_000, 55_000]
REPEAT_PWM   = 30_000   # re-measured at the end for the vbat-drain check
SETTLE_S     = 3.0      # spin-up settle before asking for a reading

_DEFAULT_OUTPUT = Path("data") / "calibration"


class VbatMonitor:
    """Tiny pm.vbat logger (2 Hz) — keeps the latest value."""

    def __init__(self, cf) -> None:
        self._cfg = LogConfig(name="UAVibeVbat", period_in_ms=500)
        self._cfg.add_variable("pm.vbat", "float")
        self._cfg.data_received_cb.add_callback(self._on_data)
        self.vbat: float = float("nan")
        cf.log.add_config(self._cfg)
        self._cfg.start()

    def _on_data(self, _ts, data, _cfg) -> None:
        self.vbat = float(data.get("pm.vbat", float("nan")))

    def stop(self) -> None:
        try:
            self._cfg.stop()
            self._cfg.delete()
        except Exception:
            pass


def _prompt_reading(pwm: int, vbat: float, label: str = "") -> "tuple[float, str] | None":
    """
    Ask the operator for the tachometer value. Returns (displayed_value, note)
    or None to skip this point. 'q' aborts the whole session (saves progress).
    """
    tag = f"{label} " if label else ""
    while True:
        raw = input(
            f"  PWM {pwm:>6d} | vbat {vbat:.2f} V | {tag}tach reading "
            f"(number, s=skip, q=quit): "
        ).strip().lower()
        if raw == "q":
            raise KeyboardInterrupt
        if raw in ("s", ""):
            return None
        try:
            value = float(raw.replace(",", "."))
        except ValueError:
            print("    Not a number — try again.")
            continue
        if value <= 0:
            print("    Must be positive — try again.")
            continue
        note = input("    optional note (Enter for none): ").strip()
        return value, note


def _prompt_reading_multi(
    pwm: int, vmon: "VbatMonitor", motor_ids: "list[int]",
) -> "dict[int, tuple[float, str, float] | None]":
    """
    Ask for one tachometer reading per motor in motor_ids (all already
    spinning together at pwm). No per-reading note prompt — four readings
    per PWM step is enough typing already. Returns {motor_id: (value, "",
    vbat_at_this_reading) or None}. 'q' at any point aborts the whole session.

    vbat is sampled fresh for EACH motor at the moment its value is typed,
    not once for the whole group: typing four tach readings can take tens
    of seconds under heavy 4-motor current draw, during which vbat keeps
    sagging — a single shared timestamp would silently misattribute a
    stale (too-low, for the earlier-read motors) voltage to every row in
    the group (found 2026-09-02, comparing a --drive all session against
    the canonical model: residuals grew from negative for the first-read
    motor to positive for the last-read one, tracking read order, not
    genuine per-motor RPM differences).
    """
    print(f"  PWM {pwm:>6d} | vbat {vmon.vbat:.2f} V — reading each motor:")
    results: "dict[int, tuple[float, str, float] | None]" = {}
    for mid in motor_ids:
        while True:
            raw = input(f"    M{mid} tach reading (number, s=skip, q=quit): ").strip().lower()
            if raw == "q":
                raise KeyboardInterrupt
            if raw in ("s", ""):
                results[mid] = None
                break
            try:
                value = float(raw.replace(",", "."))
            except ValueError:
                print("      Not a number — try again.")
                continue
            if value <= 0:
                print("      Must be positive — try again.")
                continue
            results[mid] = (value, "", vmon.vbat)
            break
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="PWM -> RPM laser-tachometer calibration")
    p.add_argument("--uri", help="Explicit URI (default: radio 80/2M, USB fallback)")
    p.add_argument("--motors", type=int, nargs="+", default=None,
                   choices=[1, 2, 3, 4],
                   help="Motors to READ (default: 1 for --drive single, "
                        "1 2 3 4 for --drive all)")
    p.add_argument("--drive", choices=["single", "all"], default="single",
                   help="single: only the motor being read spins at a time; "
                        "all: all 4 motors spin together (battery-sag test, "
                        "REQUIRES the drone to be restrained) and you read "
                        "every motor in --motors at each PWM step")
    p.add_argument("--order", choices=["interleaved", "block"], default="interleaved",
                   help="single-drive with 2+ motors only: interleaved (default) "
                        "reads M1,M2,M3,M4 at each PWM level before moving on "
                        "(controls for session drift); block finishes the whole "
                        "ladder for one motor before starting the next")
    p.add_argument("--pwms", type=int, nargs="+", default=DEFAULT_PWMS,
                   help="PWM ladder (default: %(default)s)")
    p.add_argument("--repeat", type=int, default=1,
                   help="Run the full ladder N times back-to-back. The natural "
                        "battery drain between runs spreads vbat across readings, "
                        "which is what the battery-compensated fit RPM ~ PWM*vbat "
                        "needs — no controlled charge levels required (default: 1)")
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    args = p.parse_args()
    if args.motors is None:
        args.motors = [1, 2, 3, 4] if args.drive == "all" else [1]

    uri = args.uri
    if uri is None:
        uri = radio_uri(channel=80, datarate="2M", address="E7E7E7E7E7")
    print(f"Connecting to {uri} ...")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"rpm_map_{datetime.now():%Y%m%d_%H%M%S}.csv"

    rows: list[dict] = []
    try:
        with open_sync(uri) as scf:
            print("Connected.")
            motors = MotorController(scf.cf)
            vmon   = VbatMonitor(scf.cf)
            time.sleep(1.5)   # first vbat sample

            print(f"\nBattery: {vmon.vbat:.2f} V")
            order_note = f"   order: {args.order}" if args.drive == "single" and len(args.motors) > 1 else ""
            print(f"Motors to READ: {args.motors}   drive mode: {args.drive}{order_note}")
            print(f"PWM ladder: {args.pwms}  + repeat {REPEAT_PWM} at the end")
            if args.drive == "all":
                print("\nSAFETY: ALL FOUR motors will spin with props ON — above")
                print("~40000 PWM the drone produces full collective thrust.")
                print("ONLY proceed if the drone is restrained (cage rig).")
            else:
                print("\nSAFETY: props ON. Only one motor spins at a time (no liftoff),")
                print("but keep fingers/hair/laser-cable clear of all props.")
            input("Press Enter to start (Ctrl-C to abort)...")

            def _set_pwm(read_mid: int, pwm: int) -> None:
                if args.drive == "all":
                    for m in (1, 2, 3, 4):
                        motors.set_motor(m, pwm)
                else:
                    motors.set_motor(read_mid, pwm)

            def _record(mid: int, pwm: int, displayed: float, note: str, tag: str, run_idx: int,
                        vbat_override: "float | None" = None) -> float:
                rpm = displayed / 2.0   # foil on both blades -> 2 pulses/rev
                rows.append({
                    "timestamp_iso":  datetime.now().isoformat(timespec="seconds"),
                    "run":            run_idx,
                    "motor":          mid,
                    "drive":          args.drive,
                    "pwm":            pwm,
                    "tach_displayed": displayed,
                    "rpm":            rpm,
                    "vbat_v":         round(vbat_override if vbat_override is not None else vmon.vbat, 3),
                    "note":           note or tag,
                })
                return rpm

            try:
                ladder = list(args.pwms) + [REPEAT_PWM]
                for run_idx in range(1, args.repeat + 1):
                    if args.repeat > 1:
                        print(f"\n##### RUN {run_idx}/{args.repeat} "
                              f"(vbat now {vmon.vbat:.2f} V) #####")

                    if args.drive == "all":
                        # All four motors spin together; read every motor in
                        # --motors at each PWM level before moving on.
                        print(f"\n=== All-four drive — {len(ladder)} points, "
                              f"reading motors {args.motors} ===")
                        for i, pwm in enumerate(ladder):
                            is_repeat = (i == len(ladder) - 1)
                            tag = "repeat_vbat_check" if is_repeat else ""
                            _set_pwm(0, pwm)
                            print(f"\n[{i+1}/{len(ladder)}] all motors spinning at {pwm} PWM"
                                  + ("  (REPEAT point - battery drain check)" if is_repeat else ""))
                            time.sleep(SETTLE_S)

                            multi = _prompt_reading_multi(pwm, vmon, args.motors)
                            for mid, result in multi.items():
                                if result is None:
                                    print(f"    M{mid}: skipped.")
                                    continue
                                displayed, note, vbat_at_read = result
                                rpm = _record(mid, pwm, displayed, note, tag, run_idx,
                                               vbat_override=vbat_at_read)
                                print(f"    M{mid} recorded: {rpm:.0f} RPM "
                                      f"(displayed {displayed:.0f}) @ {vbat_at_read:.2f} V")
                        motors.stop_all()
                        time.sleep(1.0)

                    elif args.order == "interleaved" and len(args.motors) > 1:
                        # At each PWM level, read M1 then M2 then M3 then M4
                        # (whichever are requested) before moving to the next
                        # level — controls for session drift, see module docstring.
                        print(f"\n=== Interleaved single drive — motors {args.motors}, "
                              f"{len(ladder)} PWM levels ===")
                        for i, pwm in enumerate(ladder):
                            is_repeat = (i == len(ladder) - 1)
                            tag = "repeat_vbat_check" if is_repeat else ""
                            print(f"\n[{i+1}/{len(ladder)}] level {pwm} PWM"
                                  + ("  (REPEAT point - battery drain check)" if is_repeat else ""))
                            for mid in args.motors:
                                motors.set_motor(mid, pwm)
                                time.sleep(SETTLE_S)
                                result = _prompt_reading(pwm, vmon.vbat, label=f"M{mid}")
                                motors.stop_motor(mid)
                                if result is None:
                                    print("    skipped.")
                                    continue
                                displayed, note = result
                                rpm = _record(mid, pwm, displayed, note, tag, run_idx)
                                print(f"    recorded: {rpm:.0f} RPM (displayed {displayed:.0f}) "
                                      f"@ {vmon.vbat:.2f} V")
                                time.sleep(0.3)   # brief gap before the next motor spins up

                    else:
                        # Block order: finish the whole ladder for one motor
                        # before starting the next.
                        for mid in args.motors:
                            print(f"\n=== Reading M{mid} (single drive, block order) — "
                                  f"{len(ladder)} points ===")
                            for i, pwm in enumerate(ladder):
                                is_repeat = (i == len(ladder) - 1)
                                tag = "repeat_vbat_check" if is_repeat else ""
                                motors.set_motor(mid, pwm)
                                print(f"\n[{i+1}/{len(ladder)}] M{mid} spinning at {pwm} PWM"
                                      + ("  (REPEAT point - battery drain check)" if is_repeat else ""))
                                time.sleep(SETTLE_S)

                                result = _prompt_reading(pwm, vmon.vbat)
                                if result is None:
                                    print("    skipped.")
                                    continue
                                displayed, note = result
                                rpm = _record(mid, pwm, displayed, note, tag, run_idx)
                                print(f"    recorded: {rpm:.0f} RPM (displayed {displayed:.0f}) "
                                      f"@ {vmon.vbat:.2f} V")

                            motors.stop_motor(mid)
                            time.sleep(1.0)
            finally:
                motors.stop_all()
                vmon.stop()

    except KeyboardInterrupt:
        print("\nAborted by operator — saving what was collected.")

    if not rows:
        print("No readings collected, nothing to save.")
        sys.exit(0)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} readings -> {out_path}")

    # ── quick on-the-spot fit so the shape is visible immediately ─────────────
    try:
        import numpy as np

        def _lin_fit(x: "np.ndarray", y: "np.ndarray"):
            a, b = np.polyfit(x, y, 1)
            pred = a * x + b
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            return a, b, r2, float(np.max(np.abs(y - pred)))

        groups = sorted({(r["motor"], r["drive"]) for r in rows})
        for mid, drive in groups:
            pts = [(r["pwm"], r["rpm"], r["vbat_v"]) for r in rows
                   if r["motor"] == mid and r["drive"] == drive
                   and r["note"] != "repeat_vbat_check"]
            if len(pts) < 3:
                continue
            pwm_a  = np.array([x[0] for x in pts], dtype=float)
            rpm_a  = np.array([x[1] for x in pts], dtype=float)
            vbat_a = np.array([x[2] for x in pts], dtype=float)

            # Model 1: RPM = a*PWM + b  (ignores battery)
            a1, b1, r2_1, res_1 = _lin_fit(pwm_a, rpm_a)
            # Model 2: RPM = a*(PWM*vbat) + b  (battery-compensated —
            # physically motivated: effective voltage ~ duty * vbat.
            # Only distinguishable from model 1 if vbat varied across
            # readings, e.g. multiple --repeat runs.)
            a2, b2, r2_2, res_2 = _lin_fit(pwm_a * vbat_a, rpm_a)

            print(f"\nM{mid} [{drive} drive] — {len(pts)} points, "
                  f"vbat range {vbat_a.min():.2f}-{vbat_a.max():.2f} V")
            print(f"   model 1  RPM = {a1:.4f}*PWM + {b1:.0f}"
                  f"            R^2 = {r2_1:.4f}  max res = {res_1:.0f} RPM")
            print(f"   model 2  RPM = {a2:.4f}*(PWM*vbat) + {b2:.0f}"
                  f"   R^2 = {r2_2:.4f}  max res = {res_2:.0f} RPM")
            print(f"   implied RPM @ 65535 PWM (model 1) = {a1*65535 + b1:.0f}")
            print("   (canonical model lives in pipeline/motor_model.py — "
                  "update its constants if this fit supersedes it)")
            if vbat_a.max() - vbat_a.min() < 0.15:
                print("   NOTE: vbat barely varied in this session — model 1 vs 2 "
                      "not yet distinguishable. Use --repeat 2/3 to spread vbat.")
            if min(r2_1, r2_2) < 0.99:
                print("   NOTE: R^2 < 0.99 — visible curvature; consider a "
                      "piecewise-linear table instead of a single line.")

        # vbat-drain check via the repeat point
        for mid, drive in groups:
            base = [r for r in rows if r["motor"] == mid and r["drive"] == drive
                    and r["pwm"] == REPEAT_PWM and r["note"] != "repeat_vbat_check"]
            rep  = [r for r in rows if r["motor"] == mid and r["drive"] == drive
                    and r["note"] == "repeat_vbat_check"]
            if base and rep:
                d_rpm  = rep[0]["rpm"] - base[0]["rpm"]
                d_vbat = rep[0]["vbat_v"] - base[0]["vbat_v"]
                print(f"\nM{mid} [{drive}] battery-drain effect @ {REPEAT_PWM} PWM: "
                      f"{d_rpm:+.0f} RPM over {d_vbat:+.2f} V "
                      f"({'%.0f RPM/V' % (d_rpm/d_vbat) if abs(d_vbat) > 0.01 else 'vbat unchanged'})")
    except Exception as exc:
        print(f"(fit skipped: {exc})")


if __name__ == "__main__":
    main()
