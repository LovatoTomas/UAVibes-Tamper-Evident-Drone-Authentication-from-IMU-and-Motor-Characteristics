"""
Dynamic hover enrollment routines for UAVibes.

Two modes
---------
hover_auto    Uses cflib's HighLevelCommander to command takeoff, hover, and
              land autonomously.  Requires a positioning deck (Flow Deck v2 or
              Lighthouse) so the firmware can maintain a stable position hold.

hover_manual  The PC simply records IMU telemetry while the operator flies the
              drone manually (via cfclient or a gamepad).  No flight commands
              are issued — use this when no positioning deck is available.
"""

import time
import threading
from cflib.crazyflie import Crazyflie
from enrollment.recorder import DataRecorder

TAKEOFF_HEIGHT_M  = 1.0    # target hover altitude in metres
TAKEOFF_DURATION_S = 3.0   # time to reach hover altitude
LAND_DURATION_S   = 3.0    # time to descend to the ground
POST_LAND_WAIT_S  = 2.0    # settling time after landing command


def run_hover_auto(
    cf:         Crazyflie,
    recorder:   DataRecorder | None = None,
    duration_s: float = 45.0,
) -> None:
    """
    Automated hover using the onboard HighLevelCommander.

    Sequence: enable HL commander → takeoff → hover → land → stop.

    Parameters
    ----------
    cf         : Crazyflie instance from the open_sync context
    recorder   : DataRecorder — receives phase labels during flight
    duration_s : hover recording duration (seconds), excluding takeoff/landing
    """
    print("[HOVER] Enabling high-level commander...")
    cf.param.set_value("commander.enHighLevel", "1")
    time.sleep(0.3)

    print(f"[HOVER] Takeoff → {TAKEOFF_HEIGHT_M} m over {TAKEOFF_DURATION_S:.0f}s")
    if recorder is not None:
        recorder.set_phase("takeoff")
    cf.high_level_commander.takeoff(TAKEOFF_HEIGHT_M, TAKEOFF_DURATION_S)
    time.sleep(TAKEOFF_DURATION_S + 1.0)   # extra second for stabilisation

    print(f"[HOVER] Hovering for {duration_s:.0f}s — recording...")
    if recorder is not None:
        recorder.set_phase("hover")
    _countdown(duration_s)

    print(f"[HOVER] Landing over {LAND_DURATION_S:.0f}s")
    if recorder is not None:
        recorder.set_phase("landing")
    cf.high_level_commander.land(0.0, LAND_DURATION_S)
    time.sleep(LAND_DURATION_S + POST_LAND_WAIT_S)
    cf.high_level_commander.stop()

    print("[HOVER] Landed and stopped.")


def run_hover_manual(
    recorder:   DataRecorder | None = None,
    duration_s: float = 45.0,
) -> None:
    """
    Manual recording mode — no flight commands are issued.

    Fly the drone yourself; this function records IMU telemetry for
    duration_s seconds (or until Enter is pressed).

    Parameters
    ----------
    recorder   : DataRecorder — phase set to "hover_manual"
    duration_s : max recording duration (seconds)
    """
    if recorder is not None:
        recorder.set_phase("hover_manual")

    print(f"[HOVER] Manual mode — recording for up to {duration_s:.0f}s.")
    print("  Fly the drone. Press Enter to stop early.\n")

    stop_event = threading.Event()

    def _wait_enter():
        input()
        stop_event.set()

    listener = threading.Thread(target=_wait_enter, daemon=True)
    listener.start()

    _countdown(duration_s, stop_event=stop_event)

    print("\n[HOVER] Recording complete.")


# ── helpers ───────────────────────────────────────────────────────────────────

def _countdown(duration_s: float, stop_event: threading.Event | None = None,
               tick: float = 0.5) -> None:
    """Print a live progress line; stop early if stop_event is set."""
    start = time.time()
    while True:
        elapsed   = time.time() - start
        remaining = duration_s - elapsed
        if remaining <= 0:
            break
        if stop_event is not None and stop_event.is_set():
            break
        print(
            f"\r  {elapsed:5.1f}s / {duration_s:.0f}s  "
            f"[{'█' * int(elapsed / duration_s * 30):30s}]",
            end="", flush=True,
        )
        time.sleep(tick)
    print()
