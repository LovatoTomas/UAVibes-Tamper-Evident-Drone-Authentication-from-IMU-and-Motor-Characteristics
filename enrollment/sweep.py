"""
Static motor sweep enrollment routines for UAVibes.

The drone must be on a flat surface with propellers removed (or securely
restrained) before running these routines.

run_static_sweep — Sweep protocol
----------------------------------
Excitation order: individual motors -> motor pairs -> combined (all four).
Each stage samples a different set of chassis structural modes (this is
standard experimental modal-testing practice: different excitation points
preferentially excite different mode shapes, so multiple excitation
patterns are needed to fully characterise a structure's dynamics):

  Combined (all 4)   -- symmetric/vertical "bounce" mode; closest to real
                         flight loading.
  Individual motors  -- local bending of a single arm; localises
                         asymmetries to a specific arm/mount.
  Diagonal pairs     -- M1+M3 (CCW) and M2+M4 (CW), the same pairing used
                         for yaw control in flight; excites torsional modes.
  Adjacent pairs     -- front/rear (M1+M2 / M3+M4) and left/right
                         (M1+M4 / M2+M3); excites pitch- and roll-coupled
                         bending respectively.

Optional individual phase (--individual-motors):
  For each motor M1..M4 in turn:
    ramp 0 -> max_pwm over INDIVIDUAL_RAMP_S seconds
    hold at peak for HOLD_S seconds
    cut to 0, wait INTER_MOTOR_S before next motor

Optional pairs phase (--motor-pairs):
  For each of the 6 motor pairs in MOTOR_PAIRS:
    ramp 0 -> max_pwm over INDIVIDUAL_RAMP_S seconds (both motors together)
    hold at peak for HOLD_S seconds
    cut to 0, wait INTER_MOTOR_S before next pair

Combined phase (always runs):
    ramp all four motors 0 -> max_pwm over sweep_duration_s seconds
    hold at peak for HOLD_S seconds
    ramp back down to 0 over RAMP_DOWN_S seconds
    cut to 0

Phase labels written to the recorder
-------------------------------------
  m1_only / m2_only / m3_only / m4_only  -- individual ramps
  m{n}_hold                               -- individual hold at peak
  {pair}_ramp / {pair}_hold               -- pair ramp/hold, pair in MOTOR_PAIRS
  all_four_ramp                           -- combined ramp-up
  all_four_hold                           -- combined peak hold
  ramp_down                               -- combined ramp-down

run_stepped_hold — Step protocol
----------------------------------
Ramps to each PWM step in sequence, holds for hold_s seconds at each.
Designed for spectral characterisation: each hold gives a stationary
signal at a known motor speed for FFT analysis.

Phase labels: hold_{pwm}  (e.g. hold_15000, hold_35000)
"""

import time
from crazyflie.motors import MotorController
from enrollment.recorder import DataRecorder

DEFAULT_MAX_PWM        = 35_000   # ~53% — safe below liftoff for desk test
DEFAULT_SWEEP_DURATION = 60.0     # seconds for the combined ramp-up
HOLD_S                 =  5.0     # hold duration at peak power
RAMP_DOWN_S            = 10.0     # duration of the ramp-down to zero
INDIVIDUAL_RAMP_S      = 15.0     # ramp duration per single-motor sweep
INTER_MOTOR_S          =  3.0     # pause between individual motor sweeps
_STEP_HZ               = 10       # motor command updates per second

# Default steps for spectral characterisation (props OFF, no constraint)
DEFAULT_STEPPED_PWMS  = [10_000, 15_000, 20_000, 25_000, 35_000, 45_000, 55_000]
DEFAULT_STEP_HOLD_S   = 30.0     # seconds of stationary recording per step
DEFAULT_STEP_RAMP_S   =  5.0     # ramp duration between consecutive steps

# Motor pairs for modal-excitation diversity (Crazyflie X-configuration:
# M1 front-left CCW, M2 front-right CW, M3 rear-right CCW, M4 rear-left CW).
# Diagonal pairs share rotation direction (the same pairing flight control
# uses for yaw); adjacent pairs probe pitch- and roll-coupled bending.
MOTOR_PAIRS: dict[str, tuple[int, int]] = {
    "diag_13":  (1, 3),   # diagonal, CCW pair -- yaw/torsion-coupled
    "diag_24":  (2, 4),   # diagonal, CW pair -- yaw/torsion-coupled
    "front_12": (1, 2),   # front pair -- pitch-coupled
    "rear_34":  (3, 4),   # rear pair -- pitch-coupled
    "left_14":  (1, 4),   # left pair -- roll-coupled
    "right_23": (2, 3),   # right pair -- roll-coupled
}


def _ramp(
    motors:    MotorController,
    motor_ids: list[int],
    from_pwm:  int,
    to_pwm:    int,
    duration_s: float,
    recorder:  DataRecorder | None,
    phase:     str,
    tick_cb=None,
) -> None:
    """
    Linearly ramp the selected motors from from_pwm to to_pwm over duration_s.

    tick_cb(phase, dt, motor_pwm), if given, is called after every motor
    command with the elapsed slice ``dt`` and a ``{1: pwm, 2: pwm, ...}``
    snapshot for all four motors (0 for motors not in ``motor_ids``).
    """
    if recorder is not None:
        recorder.set_phase(phase)

    steps      = max(1, int(duration_s * _STEP_HZ))
    step_delay = duration_s / steps
    delta      = (to_pwm - from_pwm) / steps
    power      = float(from_pwm)

    for _ in range(steps):
        power += delta
        pwm  = int(power)
        pwms = {mid: 0 for mid in (1, 2, 3, 4)}
        for mid in motor_ids:
            motors.set_motor(mid, pwm)
            pwms[mid] = pwm
        if tick_cb is not None:
            tick_cb(phase, step_delay, pwms)
        time.sleep(step_delay)


def _hold(
    motors:    MotorController,
    motor_ids: list[int],
    pwm:       int,
    duration_s: float,
    recorder:  DataRecorder | None,
    phase:     str,
    tick_cb=None,
    tick_s:    float = 0.2,
) -> None:
    """Hold the selected motors at a constant pwm for duration_s, ticking tick_cb."""
    if recorder is not None:
        recorder.set_phase(phase)

    pwms = {mid: 0 for mid in (1, 2, 3, 4)}
    for mid in motor_ids:
        pwms[mid] = pwm

    elapsed = 0.0
    while elapsed < duration_s:
        step = min(tick_s, duration_s - elapsed)
        time.sleep(step)
        elapsed += step
        if tick_cb is not None:
            tick_cb(phase, step, pwms)


def run_static_sweep(
    motors:      MotorController,
    recorder:    DataRecorder | None = None,
    max_pwm:     int   = DEFAULT_MAX_PWM,
    duration_s:  float = DEFAULT_SWEEP_DURATION,
    individual:  bool  = False,
    pairs:       bool  = False,
    local_max_pwm: "int | None" = None,
    progress_cb=None,
) -> None:
    """
    Execute the full static sweep enrollment routine.

    Parameters
    ----------
    motors      : MotorController connected to the drone
    recorder    : DataRecorder (optional) — receives set_phase() calls
    max_pwm     : peak PWM for the COMBINED all-four stage (0-65535); keep
                  below liftoff threshold. Balanced thrust means no net
                  tilt here, so this can safely go into the flight-relevant
                  range (see local_max_pwm below for why individual/pairs
                  need a lower cap).
    duration_s  : ramp-up duration for the combined all-four sweep (seconds)
    individual  : if True, sweep each motor alone first
    pairs       : if True, sweep all 6 motor pairs (2 diagonal + 4 adjacent)
                  before the combined sweep — see MOTOR_PAIRS and the module
                  docstring for why this improves fingerprint richness.
    local_max_pwm : peak PWM for the individual-motor and motor-pair stages
                  specifically. Defaults to max_pwm (old behaviour) when
                  None. Confirmed 2026-09-08 (log15): single-motor and
                  ADJACENT-pair drive (front_12/rear_34/left_14/right_23 —
                  asymmetric thrust, unlike the balanced diagonal pairs or
                  the combined stage) tilts the drone enough to press the
                  chassis against the rig's support wires above roughly
                  50 000 PWM, injecting rig-contact noise (not a chassis
                  property) into exactly the phases meant to isolate local
                  chassis asymmetries — the opposite of what they're for.
                  Pass something like 40 000 to stay clear of that with
                  margin while still exciting local modes well.
    progress_cb : optional callback invoked frequently as
                  ``progress_cb(phase, elapsed_s, total_s, remaining_s, motor_pwm)``
                  where ``motor_pwm`` is a ``{1: pwm, ..., 4: pwm}`` dict.
                  Exceptions raised inside it are swallowed so a UI bug can
                  never interrupt the physical sweep.
    """
    all_motors = [1, 2, 3, 4]
    local_max_pwm = max_pwm if local_max_pwm is None else local_max_pwm

    total_s = duration_s + HOLD_S + RAMP_DOWN_S
    if individual:
        total_s += 4 * (INDIVIDUAL_RAMP_S + HOLD_S + INTER_MOTOR_S) + 5.0
    if pairs:
        total_s += len(MOTOR_PAIRS) * (INDIVIDUAL_RAMP_S + HOLD_S + INTER_MOTOR_S) + 5.0

    _elapsed = [0.0]

    def tick_cb(phase: str, dt: float, pwms: dict) -> None:
        _elapsed[0] += dt
        if progress_cb is None:
            return
        try:
            progress_cb(
                phase       = phase,
                elapsed_s   = _elapsed[0],
                total_s     = total_s,
                remaining_s = max(0.0, total_s - _elapsed[0]),
                motor_pwm   = pwms,
            )
        except Exception:
            pass

    if individual:
        print(f"[SWEEP] Individual sweep -- 4 motors x {INDIVIDUAL_RAMP_S:.0f}s ramp "
              f"(capped at {local_max_pwm} PWM)")
        for mid in all_motors:
            print(f"  M{mid}: 0 -> {local_max_pwm} PWM over {INDIVIDUAL_RAMP_S:.0f}s", flush=True)
            _ramp(motors, [mid], 0, local_max_pwm, INDIVIDUAL_RAMP_S,
                  recorder, f"m{mid}_only", tick_cb)

            _hold(motors, [mid], local_max_pwm, HOLD_S, recorder, f"m{mid}_hold", tick_cb)

            motors.stop_motor(mid)
            print(f"  M{mid} stopped. Waiting {INTER_MOTOR_S:.0f}s...", flush=True)
            _hold(motors, [], 0, INTER_MOTOR_S, recorder, f"m{mid}_pause", tick_cb)

        print(f"  Individual sweep done. Pausing 5s.")
        _hold(motors, [], 0, 5.0, recorder, "settle_pause", tick_cb)

    if pairs:
        print(f"[SWEEP] Motor-pair sweep -- {len(MOTOR_PAIRS)} pairs x "
              f"{INDIVIDUAL_RAMP_S:.0f}s ramp (capped at {local_max_pwm} PWM)")
        for label, (ma, mb) in MOTOR_PAIRS.items():
            print(f"  {label} (M{ma}+M{mb}): 0 -> {local_max_pwm} PWM over "
                  f"{INDIVIDUAL_RAMP_S:.0f}s", flush=True)
            _ramp(motors, [ma, mb], 0, local_max_pwm, INDIVIDUAL_RAMP_S,
                  recorder, f"{label}_ramp", tick_cb)

            _hold(motors, [ma, mb], local_max_pwm, HOLD_S, recorder, f"{label}_hold", tick_cb)

            motors.stop_motor(ma)
            motors.stop_motor(mb)
            print(f"  {label} stopped. Waiting {INTER_MOTOR_S:.0f}s...", flush=True)
            _hold(motors, [], 0, INTER_MOTOR_S, recorder, f"{label}_pause", tick_cb)

        print(f"  Pair sweep done. Pausing 5s.")
        _hold(motors, [], 0, 5.0, recorder, "settle_pause", tick_cb)

    # Combined all-four sweep
    print(f"[SWEEP] Combined: 4 motors, 0 -> {max_pwm} PWM over {duration_s:.0f}s")
    _ramp(motors, all_motors, 0, max_pwm, duration_s,
          recorder, "all_four_ramp", tick_cb)

    print(f"[SWEEP] Peak hold: {HOLD_S:.0f}s at {max_pwm} PWM", flush=True)
    _hold(motors, all_motors, max_pwm, HOLD_S, recorder, "all_four_hold", tick_cb)

    print(f"[SWEEP] Ramp down over {RAMP_DOWN_S:.0f}s", flush=True)
    _ramp(motors, all_motors, max_pwm, 0, RAMP_DOWN_S,
          recorder, "ramp_down", tick_cb)

    motors.stop_all()
    time.sleep(0.3)   # let the radio/USB link settle before returning control
    print("[SWEEP] Complete. All motors stopped.")


def run_stepped_hold(
    motors:     MotorController,
    recorder:   DataRecorder | None = None,
    pwm_steps:  list[int] = None,
    hold_s:     float = DEFAULT_STEP_HOLD_S,
    ramp_s:     float = DEFAULT_STEP_RAMP_S,
) -> None:
    """
    Spectral characterisation sweep: ramp to each PWM step, hold, repeat.

    At each step the signal is stationary (constant motor speed), giving a
    clean FFT snapshot of the chassis structural response at that excitation
    level.  Phase label for each hold segment: ``hold_{pwm}``.

    Parameters
    ----------
    motors    : MotorController connected to the drone
    recorder  : DataRecorder (optional)
    pwm_steps : list of PWM setpoints to visit in order
    hold_s    : stationary recording duration at each step (seconds)
    ramp_s    : ramp duration between consecutive steps (seconds)
    """
    if pwm_steps is None:
        pwm_steps = DEFAULT_STEPPED_PWMS

    all_motors = [1, 2, 3, 4]
    prev_pwm   = 0

    print(f"[STEPPED] {len(pwm_steps)} steps x {hold_s:.0f}s hold "
          f"({ramp_s:.0f}s ramp between steps)")
    print(f"[STEPPED] Steps: {pwm_steps}")

    for pwm in pwm_steps:
        # Ramp from previous level to this step
        print(f"\n[STEPPED] Ramp {prev_pwm} -> {pwm} over {ramp_s:.0f}s ...",
              flush=True)
        _ramp(motors, all_motors, prev_pwm, pwm, ramp_s,
              recorder, f"ramp_to_{pwm}")

        # Stationary hold — this is the useful enrollment segment
        phase = f"hold_{pwm}"
        print(f"[STEPPED] Hold at {pwm} PWM for {hold_s:.0f}s  [phase={phase}]",
              flush=True)
        if recorder is not None:
            recorder.set_phase(phase)

        elapsed = 0.0
        tick    = 0.5
        while elapsed < hold_s:
            time.sleep(tick)
            elapsed += tick
            remaining = hold_s - elapsed
            print(f"\r  {elapsed:4.0f}s / {hold_s:.0f}s  remaining {remaining:4.0f}s",
                  end="", flush=True)
        print()

        prev_pwm = pwm

    # Ramp back down to zero
    print(f"\n[STEPPED] Ramp down {prev_pwm} -> 0 over {RAMP_DOWN_S:.0f}s",
          flush=True)
    _ramp(motors, all_motors, prev_pwm, 0, RAMP_DOWN_S,
          recorder, "ramp_down")

    motors.stop_all()
    print("[STEPPED] Complete. All motors stopped.")
