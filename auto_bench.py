"""
Unattended bench run on one drone (firmware FW2, drone clamped in the rig).

The operator places a drone in the rig, switches it on and starts:

    python auto_bench.py --drone C --enroll        # first time: 3 enrollment boots, then the checks
    python auto_bench.py --drone A                 # 12 on-board checks + 2 SD boot sessions

For every test the script restarts the drone over the radio (Crazyradio ->
nRF51 power switch: the STM32, the sensors and the decks are switched off
for 1 s and on again, like the power button), waits for it to boot,
reconnects and checks that it really rebooted (gyro bias and accelerometer
scale are measured again at every power-up, so they change). Then it runs
one test:

- an on-board check: the same exchange as the dashboard tab "On-board key"
  (Station.run), written to data/keystore/onboard/log.jsonl with the
  condition (default "nominal") and auto=True;
- an SD boot session: the same recording as the "Boot" session type of the
  "New Rig Session" tab (data/fixed_rig_boot/sNN_droneX/), to be uploaded
  later from the dashboard as usual.

A failed test (rejected key, speed alarm, drone not answering, no reboot,
link lost, SD deck not ready, boot calibration not finished) is retried
after a fresh restart, up to --retries times (default 3); every attempt is
in the log, so the false-rejection analysis still counts the failures. It
stops at once, without retries, when retrying cannot help (wrong drone,
old firmware, battery not recovering). On a stop it plays an alarm until
Ctrl+C; at the end the success chime, and it lists every failed attempt.

The dashboard must be closed (or disconnected): the Crazyradio serves one
program at a time.

--sim runs the on-board part against the Python twin of the firmware
(keyproto.onboard_station.SimDrone) with the recorded boots, in a temporary
station: no hardware, no SD sessions. Used to check the script itself.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)                    # data paths (data/...) are relative to auth_client/
try:                              # a Windows console may not print σ or arrows
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import numpy as np

from keyproto import boot as kb
from keyproto import onboard as ob
from keyproto import onboard_station as kos
from ui.sound import play

RADIO_DEFAULT = "radio://0/80/2M/E7E7E7E7E7"
CONDITIONS = ("nominal", "propeller", "tape", "payload", "cloned_helper", "other")
REBOOT_KEYS = ("gBiasX", "gBiasY", "gBiasZ", "accScale")


class BenchError(Exception):
    """A test or its preparation failed: retried (fresh restart) up to
    --retries times, then the bench stops and calls the operator."""


class FatalError(BenchError):
    """Retrying cannot help (wrong drone, old firmware, flat battery...):
    stop at once."""


def say(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ── hardware: connection, restart, checks ───────────────────────────────────

class Drone:
    """One open cflib connection to the drone (closed before every restart)."""

    def __init__(self, uri: str):
        from cflib.crazyflie import Crazyflie
        from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
        import crazyflie.connection  # noqa: F401  (Windows libusb patches)
        import cflib.crtp
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self.scf = SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache"))
        self.lost = False
        self.scf.cf.connection_lost.add_callback(lambda *_: setattr(self, "lost", True))
        self.scf.open_link()
        self.cf = self.scf.cf

    def close(self) -> None:
        try:
            self.scf.close_link()
        except Exception:
            pass


def connect(uri: str, wait_s: float) -> Drone:
    """Connect, retrying for up to wait_s (the drone may still be booting)."""
    end, last = time.monotonic() + wait_s, None
    while time.monotonic() < end:
        try:
            return Drone(uri)
        except Exception as exc:
            last = exc
            time.sleep(2.0)
    raise BenchError(f"cannot connect to the drone ({uri}): {last}")


def restart(radio_uri: str) -> None:
    """Power-cycle the STM32, sensors and decks through the nRF51."""
    from cflib.utils.power_switch import PowerSwitch
    import crazyflie.connection  # noqa: F401
    import cflib.crtp
    cflib.crtp.init_drivers(enable_debug_driver=False)
    ps = PowerSwitch(radio_uri)
    try:
        ps.stm_power_cycle()
    except Exception as exc:
        raise BenchError(f"restart over the radio failed ({radio_uri}): {exc}")
    finally:
        ps.close()


def board_params(d: Drone) -> "dict | None":
    from crazyflie.battery import read_board_params
    return read_board_params(d.cf)


def wait_board_ready(d: Drone, max_s: float = 20.0) -> dict:
    """Board parameters once the firmware has finished its boot calibration
    (gyro bias and accelerometer scale found: the drone must be still)."""
    end, bp = time.monotonic() + max_s, None
    while time.monotonic() < end:
        bp = board_params(d)
        if bp is not None and bp.get("gBiasOk") == 1 and bp.get("accScaleOk") == 1:
            return bp
        time.sleep(1.0)
    if bp is None:
        raise BenchError("the drone has no 'uavibe' parameters (FW2 running?)")
    raise BenchError(f"boot calibration not finished after {max_s:.0f} s "
                     f"(gBiasOk {bp.get('gBiasOk')}, accScaleOk {bp.get('accScaleOk')}): drone moving?")


def battery_gate(d: Drone, max_wait_s: float) -> float:
    """Resting battery voltage; waits (motors off) while it is below the
    dashboard's threshold, and gives up after max_wait_s."""
    from crazyflie.battery import read_vbat
    from enrollment.micro_sweep import VBAT_MIN_REST_V, VBAT_RECHECK_S
    t0 = time.monotonic()
    while True:
        v = read_vbat(d.cf)
        if v is None or v >= VBAT_MIN_REST_V:
            return v
        if time.monotonic() - t0 > max_wait_s:
            raise FatalError(f"battery at rest {v:.2f} V, below {VBAT_MIN_REST_V:.2f} V for "
                             f"{max_wait_s / 60:.0f} min: charge it or change it")
        say(f"battery at rest {v:.2f} V < {VBAT_MIN_REST_V:.2f} V: waiting...")
        time.sleep(VBAT_RECHECK_S)


# ── the two tests ───────────────────────────────────────────────────────────

def progress_line(phase, motor, pct) -> None:
    print(f"\r    {phase}{f' M{motor}' if motor else ''} {pct:3d}%   ", end="", flush=True)


def onboard_test(station: kos.Station, drone: str, mode: str, log_path: Path, extra: dict) -> dict:
    """One on-board boot: enrollment boot (mode "enroll") or authentication."""
    hs = None
    for _ in range(5):                 # the app task may answer a moment after the link is up
        hs = station.hello()
        if hs is not None:
            break
        time.sleep(1.0)
    if hs is None:
        raise BenchError("no answer on the app channel: does the drone run firmware FW2?")
    if not hs["signed_commands"]:
        raise FatalError("the drone sends no challenge: firmware older than 2026-10-29, reflash it")
    if mode == "enroll" and hs["enrolled"]:
        raise FatalError(f"the drone is already enrolled as {hs['drone']}: to enroll it again use "
                         "'Restart enrollment' in the dashboard")
    if mode == "auth" and (not hs["enrolled"] or hs["drone"] != drone):
        raise FatalError(f"the drone is {'enrolled as ' + hs['drone'] if hs['enrolled'] else 'not enrolled'}, "
                         f"expected {drone}: wrong drone in the rig?")
    r = station.run(drone, progress=progress_line)
    print()
    rec = dict(at=datetime.now().isoformat(timespec="seconds"), expected=drone,
               **({} if mode == "enroll" else extra),
               **{k: v for k, v in r.items() if k != "features"})
    rec.update(sim=extra.get("sim", False), auto=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    if mode == "enroll":
        if r.get("mode") != "enroll":
            raise BenchError(f"enrollment boot failed: {r.get('reason', r)}")
        if r.get("n_missing"):
            # take the boot just saved out of the enrollment set (kept aside), then retry
            research = station.store / "research"
            boots = json.loads((research / f"enroll_{drone}.json").read_text(encoding="utf-8"))
            bad = boots.pop()
            (research / f"enroll_{drone}.json").write_text(json.dumps(boots), encoding="utf-8")
            with open(research / f"enroll_{drone}_discarded.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(bad) + "\n")
            raise BenchError(f"enrollment boot with {r['n_missing']} missing quantities (set aside)")
        say(f"enrollment boot saved ({r['seconds']:.0f} s)")
        return r
    if not r.get("ok"):
        extra_txt = (f", speed alarm on M{', M'.join(map(str, r['speed_alarm']))}"
                     if r.get("speed_alarm") else "")
        raise BenchError(f"REJECTED ({r.get('stage')}): {r.get('reason')}"
                         f" - wrong symbols {r.get('wrong')}{extra_txt}")
    shifts = r.get("speed_shift_sigma") or {}
    say(f"accepted in {r['seconds']:.0f} s, wrong symbols {r.get('wrong')}, speed shifts "
        + " ".join(f"M{m} {v:+.1f} sd" for m, v in shifts.items()))
    return r


def sd_boot_session(d: Drone, drone: str, vbat_rest: "float | None") -> str:
    """One boot session recorded on the SD card, as the dashboard records it.
    Returns the session folder."""
    from crazyflie.battery import read_log_float
    from crazyflie.motors import MotorController
    from enrollment.micro_sweep import MAX_PWM, run_micro, COOLDOWN_S
    from enrollment.session_manifest import (SessionManifest, micro_stem, rig_dir_for,
                                             swap_session_dirs, unique_path)
    from enrollment.usd_timeline import Timeline, UsdPhaseRecorder

    cf = d.cf

    def get(name):
        try:
            return str(cf.param.get_value(name, timeout=3))
        except Exception:
            return None

    if get("usd.logging") in ("1", "1.0"):
        cf.param.set_value("usd.logging", "0")
        time.sleep(0.3)
    if get("deck.bcUSD") not in ("1", "1.0"):
        raise BenchError("SD deck not detected")
    if get("usd.canLog") not in ("1", "1.0"):
        raise BenchError("SD deck present but not ready (card inserted? FAT32? config.txt?)")

    rig = rig_dir_for("boot")
    nums = [int(m.group(1)) for f in rig.glob("s*_*") if f.is_dir()
            and (m := re.match(r"s(\d+)_", f.name))] if rig.exists() else []
    n = max(nums, default=0) + 1
    sess_dir = rig / f"s{n:02d}_drone{drone}"
    dirs = swap_session_dirs(sess_dir)
    mpath = sess_dir / "session.json"
    manifest = SessionManifest.load_or_new(mpath, drone, n, "boot")
    bp = board_params(d)
    manifest.board_params = dict(firmware=bp is not None, **(bp or {}),
                                 board_temp_c=read_log_float(cf, "baro.temp"), vbat_rest=vbat_rest,
                                 read_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    manifest.save(mpath)

    dirs["timelines"].mkdir(parents=True, exist_ok=True)
    stem = micro_stem(drone, n, "boot-seq", kind="boot")
    tl_path = unique_path(dirs["timelines"] / f"{stem}.csv")
    attempt = manifest.start_attempt("boot-seq", MAX_PWM, tl_path)
    attempt.vbat_rest = vbat_rest
    manifest.save(mpath)

    tl = Timeline(tl_path)
    if vbat_rest is not None:
        tl.log("vbat_rest", f"{vbat_rest:.3f}")
    motors = MotorController(cf)

    def prog(phase, elapsed, total, remaining, pwms):
        tl.log("pwm", max(pwms.values()))
        print(f"\r    {phase} {remaining:3.0f} s left   ", end="", flush=True)

    try:
        cf.param.set_value("usd.logging", "1")
        tl.log("usd_start")
        run_micro("boot-seq", motors, UsdPhaseRecorder(tl), progress_cb=prog)
        print()
        if d.lost:
            raise BenchError("link lost during the SD recording")
        cf.param.set_value("usd.logging", "0")
        tl.log("usd_stop")
        attempt.status = "completed"
    except BaseException:
        attempt.status = "interrupted"
        for f in (motors.stop_all, lambda: cf.param.set_value("usd.logging", "0"),
                  lambda: tl.log("usd_stop")):
            try:
                f()
            except Exception:
                pass
        raise
    finally:
        tl.close()
        manifest.save(mpath)
    time.sleep(COOLDOWN_S)        # let the deck close the file before the power is cut
    return sess_dir.as_posix()


# ── main loop ───────────────────────────────────────────────────────────────

def alarm_until_ctrl_c(max_s: float = 900.0) -> None:
    say("Ctrl+C to stop the alarm")
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < max_s:
            play(False)
            time.sleep(15.0)
    except KeyboardInterrupt:
        pass


def plan(args, enrolled_in_station: bool) -> "list[str]":
    steps = []
    if args.enroll:
        steps += ["enroll"] * kos.N_ENROLL
    steps += ["sd"] * args.sd + ["auth"] * args.onboard
    return steps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drone", required=True, choices=list("ABCD"))
    ap.add_argument("--onboard", type=int, default=12, help="on-board checks (default 12)")
    ap.add_argument("--sd", type=int, default=2, help="SD boot sessions, recorded first (default 2)")
    ap.add_argument("--enroll", action="store_true",
                    help=f"first {kos.N_ENROLL} enrollment boots and the helper data (drone not enrolled)")
    ap.add_argument("--condition", default="nominal", choices=CONDITIONS)
    ap.add_argument("--note", default=None, help="free note written to the log (which motor, what change...)")
    ap.add_argument("--uri", default=RADIO_DEFAULT, help=f"link to the drone (default {RADIO_DEFAULT})")
    ap.add_argument("--radio-uri", default=None, help="radio address for the restart (default: --uri if radio)")
    ap.add_argument("--boot-wait", type=float, default=6.0, help="seconds between restart and reconnection")
    ap.add_argument("--pause", type=float, default=20.0, help="seconds of rest between tests (motors cool down)")
    ap.add_argument("--battery-wait", type=float, default=600.0, help="max wait for the battery, seconds")
    ap.add_argument("--retries", type=int, default=3,
                    help="retries of a failed test, each after a fresh restart (default 3)")
    ap.add_argument("--beep-on-retry", action="store_true", help="also beep at every retried failure")
    ap.add_argument("--alarm-min", type=float, default=15.0, help="minutes the alarm repeats after a stop")
    ap.add_argument("--sim", action="store_true", help="simulated drone, temporary station, no hardware")
    args = ap.parse_args()

    if args.condition != "nominal" and args.sd:
        ap.error("SD sessions of an altered drone go through the dashboard (props/payload): use --sd 0")
    if args.enroll and args.condition != "nominal":
        ap.error("enroll a drone in its nominal condition")
    radio_uri = args.radio_uri or (args.uri if args.uri.startswith("radio://") else RADIO_DEFAULT)
    sysp = kb.system_params()
    extra = dict(condition=args.condition, condition_note=args.note, sim=args.sim)

    if args.sim:
        store = Path(tempfile.mkdtemp(prefix="uavibes_bench_"))
        recorded = []
        for sd in sorted(kb.BOOT_DIR.glob(f"s*_drone{args.drone}")):
            try:
                x = kob_features(sd, sysp)
                if np.isfinite(x).all():          # boots with a hold lost to the SD stall are skipped
                    recorded.append(x)
            except Exception:
                pass
        if not recorded:
            ap.error(f"no recorded boot of drone {args.drone} for the simulation")
        steps_n = (kos.N_ENROLL * args.enroll + args.onboard) * (args.retries + 1)
        boots = [recorded[i % len(recorded)] for i in range(steps_n)]
        sim = kos.SimDrone(sysp, boots, kos.station_key(store, create=True)[1])
        args.sd = 0
    else:
        store = kos.STORE
    station_db = kos.Station(kos.Link(), sysp, store).db()
    if args.enroll and args.drone in station_db:
        ap.error(f"drone {args.drone} is already in the station: 'Restart enrollment' in the dashboard first")
    if not args.enroll and args.onboard and args.drone not in station_db:
        ap.error(f"drone {args.drone} is not enrolled in the station: add --enroll")
    log_path = store / "log.jsonl"
    steps = plan(args, args.drone in station_db)
    per = {"enroll": 75, "auth": 75, "sd": 70}
    say(f"drone {args.drone}: {len(steps)} tests ({', '.join(f'{steps.count(k)} {k}' for k in ('enroll', 'sd', 'auth') if steps.count(k))}), "
        f"about {sum(per[s] for s in steps) / 60:.0f} min if the battery holds"
        + (f"; station {store}" if args.sim else ""))

    prev = None
    d = None
    done = {"enroll": 0, "sd": 0, "auth": 0}
    failures = []                 # (test, reason) of every failed attempt, retried or not
    sd_dirs = []                  # every SD session started, in recording order

    def close():
        nonlocal d
        if d is not None:
            d.close()
            d = None

    def prepare() -> "float | None":
        """Restart, reconnect, wait for the boot calibration, check the
        reboot and the battery. Returns the resting battery voltage."""
        nonlocal d, prev
        from crazyflie.battery import read_vbat
        from enrollment.micro_sweep import VBAT_MIN_REST_V
        while True:
            close()
            restart(radio_uri)
            time.sleep(args.boot_wait)
            d = connect(args.uri, 30.0)
            bp = wait_board_ready(d)
            if prev is not None and all(bp[k] == prev[k] for k in REBOOT_KEYS):
                raise BenchError("the drone did not reboot (same gyro bias and accelerometer scale)")
            prev = bp
            v = read_vbat(d.cf)
            if v is None or v >= VBAT_MIN_REST_V:
                say(f"rebooted, battery {v:.2f} V" if v is not None else "rebooted")
                return v
            # recover with the motors off, then restart again so the test
            # starts right after a power-up, like the others
            battery_gate(d, args.battery_wait)

    def one_test(step: str) -> None:
        if args.sim:
            onboard_test(kos.Station(sim, sysp, store), args.drone, step, log_path, extra)
            return
        v = prepare()
        if step == "sd":
            folder = None
            try:
                folder = sd_boot_session(d, args.drone, v)
            finally:
                # an interrupted session may still have a file on the card: list it too
                sd_dirs.append(folder or "(interrupted SD session, see the dashboard)")
            say(f"SD boot session recorded: {folder}")
        else:
            onboard_test(kos.Station(kos.CflibLink(d.cf), sysp, store), args.drone, step, log_path, extra)
            if d.lost:
                raise BenchError("link lost")
        close()

    try:
        if not args.sim:
            d = connect(args.uri, 20.0)
            prev = board_params(d)
            if prev is None:
                raise FatalError("the drone has no 'uavibe' parameters: is it running FW2?")
            close()
        for i, step in enumerate(steps, 1):
            for attempt in range(args.retries + 1):
                say(f"-- test {i}/{len(steps)}: {step}" + (f" (retry {attempt})" if attempt else "") + " --")
                try:
                    one_test(step)
                    break
                except FatalError:
                    raise
                except Exception as exc:          # BenchError, or cflib/link trouble
                    print()
                    why = str(exc) if isinstance(exc, BenchError) else repr(exc)
                    failures.append((f"{i} {step}", why))
                    close()
                    if attempt == args.retries:
                        raise BenchError(f"{why} (and {args.retries} retries failed)")
                    say(f"FAILED: {why} -> retrying")
                    if args.beep_on_retry:
                        play(False)
                    if not args.sim:
                        time.sleep(args.pause)
            done[step] += 1
            if step == "enroll" and done["enroll"] == kos.N_ENROLL:
                if args.sim:
                    r = kos.Station(sim, sysp, store).finish_enrollment(args.drone)
                else:
                    d = connect(args.uri, 30.0)     # same boot: no restart needed for the transfer
                    r = kos.Station(kos.CflibLink(d.cf), sysp, store).finish_enrollment(args.drone)
                    close()
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(at=datetime.now().isoformat(timespec="seconds"),
                                            sim=args.sim, expected=args.drone, mode="finish_enrollment",
                                            auto=True, **r), default=str) + "\n")
                if not r.get("ok"):
                    raise FatalError(f"enrollment not completed: {r.get('reason')}")
                say(f"drone {args.drone} enrolled ({r['helper_bytes']} B of helper data)")
            if i < len(steps) and not args.sim:
                time.sleep(args.pause)
    except Exception as exc:          # BenchError, or anything unexpected (files...)
        print()
        say(f"STOPPED: {exc if isinstance(exc, BenchError) else repr(exc)}")
        report(done, failures, sd_dirs)
        close()
        alarm_until_ctrl_c(args.alarm_min * 60)
        return 1
    except KeyboardInterrupt:
        print()
        say("interrupted by the operator (the drone stops its motors by itself within 3 s)")
        report(done, failures, sd_dirs)
        close()
        return 130
    say("all done")
    report(done, failures, sd_dirs)
    play(True)
    return 0


def report(done: dict, failures: list, sd_dirs: list) -> None:
    say(f"completed: {done}")
    if failures:
        say(f"{len(failures)} failed attempt(s), all in the log:")
        for where, why in failures:
            print(f"    test {where}: {why}")
    if sd_dirs:
        say("SD sessions to upload from the dashboard (New Rig Session, type Boot), in this order: "
            + ", ".join(sd_dirs))


def kob_features(session_dir, sysp) -> np.ndarray:
    return ob.onboard_features(ob.boot_inputs(session_dir), sysp)


if __name__ == "__main__":
    sys.exit(main())
