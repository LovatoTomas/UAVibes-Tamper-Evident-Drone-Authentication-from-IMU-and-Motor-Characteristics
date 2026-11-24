#!/usr/bin/env python3
"""
UAVibes — Enrollment & Data Collection
======================================
Connects to a Crazyflie 2.1 (USB or CrazyRadio) and runs one of five
enrollment routines, saving timestamped IMU + motor telemetry to CSV.

Usage examples
--------------
  # Spectral characterisation — PROPS OFF, no constraint (session 1)
  python main.py --mode stepped --drone-id cf_001

  # Custom steps and hold duration
  python main.py --mode stepped --drone-id cf_001 --step-pwms 10000 20000 35000 55000 --step-hold 45

  # Static sweep on a desk (propellers removed) — USB auto-detect
  python main.py --mode static --drone-id cf_001

  # Static sweep with per-motor sweeps before the combined run
  python main.py --mode static --drone-id cf_001 --individual-motors

  # Automated hover (requires Flow Deck v2) — CrazyRadio default channel
  python main.py --mode hover-auto --radio --drone-id cf_001

  # Manual-record hover (fly yourself, no positioning deck needed)
  python main.py --mode hover-manual --drone-id cf_001

  # Both static + hover in sequence
  python main.py --mode both --drone-id cf_001 --sweep-max-pwm 30000

  # Scan and exit
  python main.py --scan
  python main.py --scan-radio
"""

import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

from crazyflie.connection import (
    get_first_uri, scan, open_sync,
    radio_uri, scan_radio,
    _RADIO_DEFAULT_ADDRESS,
)
from crazyflie.motors import MotorController
from crazyflie.imu import ImuLogger, FREQ_OPTIONS, DEFAULT_FREQ
from enrollment.recorder import DataRecorder
from enrollment.sweep import (
    run_static_sweep,
    run_stepped_hold,
    DEFAULT_MAX_PWM,
    DEFAULT_SWEEP_DURATION,
    DEFAULT_STEPPED_PWMS,
    DEFAULT_STEP_HOLD_S,
    DEFAULT_STEP_RAMP_S,
)
from enrollment.hover import run_hover_auto, run_hover_manual
from enrollment.usd_timeline import Timeline, UsdPhaseRecorder

_DATARATE_CHOICES  = ["250K", "1M", "2M"]
_DEFAULT_OUTPUT    = Path("data")
_DEFAULT_HOVER_S   = 45.0
_DEFAULT_LOCAL_PWM = 40_000   # individual-motor / motor-pair cap — see
                              # run_static_sweep's local_max_pwm docstring
                              # (2026-09-08, log15: asymmetric thrust from
                              # single-motor/adjacent-pair drive tilts the
                              # drone into the rig's support wires above
                              # ~50 000 PWM; the balanced combined stage is
                              # unaffected and keeps using --sweep-max-pwm).


def _run_usd_static_sweep(cf, motors: MotorController, args) -> None:
    """
    Run enrollment.sweep.run_static_sweep() while recording onboard at
    ~1 kHz via the microSD deck instead of 100 Hz CRTP telemetry.

    Only the excitation protocol is shared with the normal --mode static
    path; the recording path is entirely different (SD deck + PC-side
    Timeline, see enrollment/usd_timeline.py), which is why this lives in
    its own function rather than branching inside the DataRecorder-based
    mode block below.
    """
    def get_param(name: str) -> "str | None":
        try:
            return cf.param.get_value(name, timeout=3)
        except Exception as exc:
            print(f"  cannot read {name}: {exc}")
            return None

    if str(get_param("deck.bcUSD")) not in ("1", "1.0"):
        print("ERROR: SD deck not detected.")
        sys.exit(1)
    if str(get_param("usd.canLog")) not in ("1", "1.0"):
        print("ERROR: deck present but not ready (card FAT32? config.txt? "
              "card inserted before power-on?).")
        sys.exit(1)
    print("SD deck ready.")

    usd_dir  = args.output_dir / "usd"
    tl_path  = usd_dir / f"timeline_static_{args.drone_id}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    stages = ["combined (all 4 motors)"]
    if args.individual_motors:
        stages.insert(0, "individual motors")
    if args.motor_pairs:
        stages.insert(-1 if args.individual_motors else 0, "motor pairs")

    print(f"\n{'='*50}")
    print(f" STATIC SWEEP — onboard SD @ ~1 kHz")
    print(f"{'='*50}")
    print(f" Timeline : {tl_path}")
    print(f" Max PWM  : {args.sweep_max_pwm}  (individual/pairs capped at "
          f"{args.local_max_pwm})")
    print(f" Duration : {DEFAULT_SWEEP_DURATION:.0f}s ramp-up (fixed)")
    print(f" Mode     : {' + '.join(stages)}")
    _confirm("SAFETY: drone must be restrained (e.g. the test rig) — this "
             "drives real props at flight-relevant PWM, not a bare desk sweep.")

    tl = Timeline(tl_path)
    phase_recorder = UsdPhaseRecorder(tl)

    def progress_cb(phase, elapsed_s, total_s, remaining_s, motor_pwm) -> None:
        tl.log("pwm", max(motor_pwm.values()))

    try:
        cf.param.set_value("usd.logging", "1")
        tl.log("usd_start")
        print("\nOnboard logging STARTED.")

        run_static_sweep(
            motors        = motors,
            recorder      = phase_recorder,
            max_pwm       = args.sweep_max_pwm,
            duration_s    = DEFAULT_SWEEP_DURATION,
            individual    = args.individual_motors,
            pairs         = args.motor_pairs,
            local_max_pwm = args.local_max_pwm,
            progress_cb   = progress_cb,
        )
    except Exception:
        motors.stop_all()
        raise
    finally:
        motors.stop_all()
        try:
            cf.param.set_value("usd.logging", "0")
            tl.log("usd_stop")
            print("\nOnboard logging STOPPED.")
        except Exception as exc:
            print(f"WARNING: could not stop logging cleanly: {exc}")
        tl.close()
        time.sleep(0.5)   # let the deck flush its buffer to the card

    print(f"\nTimeline saved -> {tl_path}")
    print("\nNext steps:")
    print("  1. power off, pull the microSD card, copy the newest logNN into "
          f"{usd_dir}/")
    print(f"  2. python usd_convert.py {usd_dir}/logNN --timeline {tl_path}")
    print(f"  3. python test_usd_enrollment.py {usd_dir}/logNN_1khz.csv")


def _output_path(drone_id: str, mode: str, output_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{drone_id}_{mode}_{ts}.csv"


def _confirm(prompt: str) -> None:
    """Print a safety prompt and wait for Enter."""
    input(f"\n{prompt}\n  Press Enter when ready (Ctrl-C to abort)...")


def main() -> None:
    p = argparse.ArgumentParser(
        description="UAVibes — Enrollment & Data Collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── connection ────────────────────────────────────────────────────────────
    conn = p.add_argument_group("Connection")
    conn.add_argument("--uri",
                      help="Explicit URI, e.g. usb://0 or radio://0/80/2M")
    conn.add_argument("--radio", action="store_true",
                      help="Connect via CrazyRadio (builds URI from channel/datarate/address)")
    conn.add_argument("--channel",  type=int, default=80,
                      help="CrazyRadio channel 0-125 (default: 80)")
    conn.add_argument("--datarate", default="2M", choices=_DATARATE_CHOICES,
                      help="CrazyRadio data rate (default: 2M)")
    conn.add_argument("--address",  default="E7E7E7E7E7",
                      help="Crazyflie radio address hex (default: E7E7E7E7E7)")
    conn.add_argument("--scan",       action="store_true",
                      help="List USB devices and exit")
    conn.add_argument("--scan-radio", action="store_true",
                      help="Scan all radio channels and exit (~10 s)")

    # ── enrollment ────────────────────────────────────────────────────────────
    enr = p.add_argument_group("Enrollment")
    enr.add_argument("--mode", default="static",
                     choices=["static", "hover-auto", "hover-manual", "both", "stepped"],
                     help="Enrollment mode (default: static)")
    enr.add_argument("--drone-id", default="drone",
                     help="Drone identifier used in output filename (default: drone)")
    enr.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT,
                     help=f"Output directory for CSV files (default: {_DEFAULT_OUTPUT})")
    enr.add_argument("--freq", type=int, default=DEFAULT_FREQ, choices=FREQ_OPTIONS,
                     help=f"IMU sampling rate in Hz (default: {DEFAULT_FREQ})")

    # ── static sweep ──────────────────────────────────────────────────────────
    # Combined-sweep duration is intentionally NOT configurable: a mismatch
    # between the enrollment session's ramp duration and later test sessions'
    # was found to confound comparisons (different PWM ramp rates sweep
    # through excitation frequencies at different speeds). It is fixed at
    # DEFAULT_SWEEP_DURATION for every recording, enrollment or test.
    sw = p.add_argument_group("Static sweep options")
    sw.add_argument("--sweep-max-pwm", type=int, default=DEFAULT_MAX_PWM,
                    help=f"Peak PWM for the COMBINED all-four stage, 0-65535 "
                         f"(default: {DEFAULT_MAX_PWM})")
    sw.add_argument("--local-max-pwm", type=int, default=_DEFAULT_LOCAL_PWM,
                    help="Peak PWM for the individual-motor and motor-pair "
                         "stages specifically (default: %(default)s) — kept "
                         "lower than --sweep-max-pwm because single-motor/"
                         "adjacent-pair drive tilts the drone into the rig's "
                         "support wires above ~50000 PWM (see "
                         "run_static_sweep's docstring); irrelevant without "
                         "--individual-motors/--motor-pairs.")
    sw.add_argument("--individual-motors", action="store_true",
                    help="Sweep each motor individually before the combined run")
    sw.add_argument("--motor-pairs", action="store_true",
                    help="Sweep all 6 motor pairs (2 diagonal + 4 adjacent) "
                         "before the combined run")
    sw.add_argument("--usd", action="store_true",
                    help="Record via the onboard microSD deck at ~1 kHz "
                         "instead of 100 Hz CRTP telemetry (static sweep "
                         "only). Requires the SD deck with config.txt (see "
                         "CLAUDE.md). Produces a PC-side timeline CSV; "
                         "convert the copied logNN file with usd_convert.py "
                         "afterwards — see the printed next-steps.")

    # ── hover ─────────────────────────────────────────────────────────────────
    hv = p.add_argument_group("Hover options")
    hv.add_argument("--hover-duration", type=float, default=_DEFAULT_HOVER_S,
                    help=f"Hover recording duration in seconds (default: {_DEFAULT_HOVER_S})")

    # ── stepped hold (spectral characterisation) ──────────────────────────────
    st = p.add_argument_group("Stepped-hold options  (--mode stepped)")
    st.add_argument("--step-pwms", type=int, nargs="+", default=DEFAULT_STEPPED_PWMS,
                    metavar="PWM",
                    help=f"PWM setpoints to visit in order "
                         f"(default: {DEFAULT_STEPPED_PWMS})")
    st.add_argument("--step-hold", type=float, default=DEFAULT_STEP_HOLD_S,
                    metavar="SEC",
                    help=f"Stationary hold duration per step in seconds "
                         f"(default: {DEFAULT_STEP_HOLD_S})")
    st.add_argument("--step-ramp", type=float, default=DEFAULT_STEP_RAMP_S,
                    metavar="SEC",
                    help=f"Ramp duration between consecutive steps in seconds "
                         f"(default: {DEFAULT_STEP_RAMP_S})")

    args = p.parse_args()

    if args.usd and args.mode != "static":
        print("ERROR: --usd is only supported with --mode static "
              "(PWM reconstruction in usd_convert.py assumes the static-"
              "sweep phase vocabulary).")
        sys.exit(1)

    # ── scan-only modes ───────────────────────────────────────────────────────

    if args.scan:
        devices = scan()
        if devices:
            print(f"Found {len(devices)} USB device(s):")
            for uri, name in devices:
                print(f"  {uri}  —  {name or '(no name)'}")
        else:
            print("No USB Crazyflie found.")
        sys.exit(0)

    if args.scan_radio:
        try:
            addr_int = int(args.address, 16)
        except ValueError:
            print(f"ERROR: Invalid address '{args.address}'")
            sys.exit(1)
        print(f"Scanning radio channels (address 0x{args.address.upper()}) — ~10 s...")
        devices = scan_radio(addr_int)
        if devices:
            print(f"Found {len(devices)} device(s):")
            for uri, name in devices:
                print(f"  {uri}  —  {name or '(no name)'}")
        else:
            print("No Crazyflie found. Try --radio for direct connect.")
        sys.exit(0)

    # ── resolve URI ───────────────────────────────────────────────────────────

    uri = args.uri
    if uri is None and args.radio:
        try:
            int(args.address, 16)
        except ValueError:
            print(f"ERROR: Invalid address '{args.address}'")
            sys.exit(1)
        uri = radio_uri(channel=args.channel, datarate=args.datarate,
                        address=args.address)
        print(f"CrazyRadio URI: {uri}")

    if uri is None:
        print("Scanning for Crazyflie via USB...")
        uri = get_first_uri()
        if uri is None:
            print("ERROR: No Crazyflie found.")
            print("  For CrazyRadio: python main.py --radio")
            sys.exit(1)
        print(f"Found: {uri}")

    # ── connect and run ───────────────────────────────────────────────────────

    print(f"Connecting to {uri} ...")
    try:
        with open_sync(uri) as scf:
            print("Connected.\n")

            recorder = DataRecorder()
            imu      = ImuLogger(scf.cf)
            motors   = MotorController(scf.cf)

            if args.usd:
                # SD deck records onboard at ~1 kHz independently of the
                # CRTP link — the 100 Hz ImuLogger/DataRecorder path below
                # is not needed (and would just add unused radio traffic).
                _run_usd_static_sweep(scf.cf, motors, args)
                motors.stop_all()
                imu = None
            else:
                imu.start(freq=args.freq, callback=recorder.record, motors=motors)
                time.sleep(1.0)   # allow firmware TOC download and first packets

                motor_status = "IMU + motor telemetry" if imu.has_motors else \
                               "IMU only (PWM will be 0)"
                print(f"{motor_status} at {args.freq} Hz.")

            # ── stepped hold (spectral characterisation) ──────────────────────
            if args.mode == "stepped":
                out = _output_path(args.drone_id, "stepped", args.output_dir)
                total_s = len(args.step_pwms) * (args.step_hold + args.step_ramp) + 10
                print(f"\n{'='*50}")
                print(f" STEPPED-HOLD — Spectral Characterisation")
                print(f"{'='*50}")
                print(f" Output     : {out}")
                print(f" Steps      : {args.step_pwms}")
                print(f" Hold/step  : {args.step_hold:.0f}s")
                print(f" Ramp/step  : {args.step_ramp:.0f}s")
                print(f" Total time : ~{total_s:.0f}s ({total_s/60:.1f} min)")
                _confirm("SAFETY: REMOVE PROPELLERS. Drone on flat surface, no constraint needed.")

                recorder.start(out, phase="start")
                try:
                    run_stepped_hold(
                        motors    = motors,
                        recorder  = recorder,
                        pwm_steps = args.step_pwms,
                        hold_s    = args.step_hold,
                        ramp_s    = args.step_ramp,
                    )
                except Exception:
                    motors.stop_all()
                    raise
                finally:
                    path, n = recorder.stop()
                    print(f"\nSaved {n:,} samples -> {path}")

            # ── static sweep (100 Hz CRTP path — skipped when --usd already
            #    ran the SD-based version above) ──────────────────────────────
            if args.mode in ("static", "both") and not args.usd:
                out = _output_path(args.drone_id, "static", args.output_dir)
                print(f"\n{'='*50}")
                print(f" STATIC SWEEP")
                print(f"{'='*50}")
                print(f" Output   : {out}")
                print(f" Max PWM  : {args.sweep_max_pwm}  (individual/pairs capped at "
                      f"{args.local_max_pwm})")
                print(f" Duration : {DEFAULT_SWEEP_DURATION:.0f}s ramp-up (fixed)")
                stages = ["combined (all 4 motors)"]
                if args.individual_motors:
                    stages.insert(0, "individual motors")
                if args.motor_pairs:
                    stages.insert(-1 if args.individual_motors else 0, "motor pairs")
                print(f" Mode     : {' + '.join(stages)}")
                _confirm("SAFETY: remove propellers or secure the drone on a surface.")

                recorder.start(out, phase="start")
                try:
                    run_static_sweep(
                        motors        = motors,
                        recorder      = recorder,
                        max_pwm       = args.sweep_max_pwm,
                        duration_s    = DEFAULT_SWEEP_DURATION,
                        individual    = args.individual_motors,
                        pairs         = args.motor_pairs,
                        local_max_pwm = args.local_max_pwm,
                    )
                except Exception:
                    motors.stop_all()
                    raise
                finally:
                    path, n = recorder.stop()
                    print(f"\nSaved {n:,} samples → {path}")

            # ── hover-auto ────────────────────────────────────────────────────
            if args.mode in ("hover-auto", "both"):
                out = _output_path(args.drone_id, "hover", args.output_dir)
                print(f"\n{'='*50}")
                print(f" AUTOMATED HOVER  (requires Flow Deck or Lighthouse)")
                print(f"{'='*50}")
                print(f" Output   : {out}")
                print(f" Duration : {args.hover_duration:.0f}s hover")
                _confirm("SAFETY: install propellers, clear 2 m radius around drone.")

                recorder.start(out, phase="pre_takeoff")
                try:
                    run_hover_auto(
                        cf         = scf.cf,
                        recorder   = recorder,
                        duration_s = args.hover_duration,
                    )
                except Exception:
                    scf.cf.high_level_commander.land(0.0, 3.0)
                    raise
                finally:
                    path, n = recorder.stop()
                    print(f"\nSaved {n:,} samples → {path}")

            # ── hover-manual ──────────────────────────────────────────────────
            if args.mode == "hover-manual":
                out = _output_path(args.drone_id, "hover_manual", args.output_dir)
                print(f"\n{'='*50}")
                print(f" MANUAL HOVER RECORDING")
                print(f"{'='*50}")
                print(f" Output   : {out}")
                print(f" Duration : up to {args.hover_duration:.0f}s")
                _confirm("Fly the drone manually. Recording starts immediately.")

                recorder.start(out, phase="pre_flight")
                try:
                    run_hover_manual(
                        recorder   = recorder,
                        duration_s = args.hover_duration,
                    )
                finally:
                    path, n = recorder.stop()
                    print(f"\nSaved {n:,} samples → {path}")

            if imu is not None:
                imu.stop()
            motors.stop_all()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise

    print("\nDone. Disconnected.")


if __name__ == "__main__":
    main()
