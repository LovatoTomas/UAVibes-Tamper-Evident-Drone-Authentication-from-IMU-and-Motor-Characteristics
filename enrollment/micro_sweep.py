"""
UAVibes — Standard micro-recording protocol (fixed-base rig, 2026-09).
=============================================================================

One excitation stage (a single motor, a motor pair, or the combined
sweep) as its OWN short, independent recording, instead of one continuous
multi-phase session — a dropped connection now costs only the stage in
progress, not the whole ~5 minute protocol (see enrollment.session_manifest
for how the resulting attempts get tracked and matched back to uploaded SD
files).

Every timing and PWM value is FIXED — one standard way to record, shared by
both drones, decided 2026-09-19:

  - PWM ceiling 58,000 for every stage (not 55,000, not per-stage tiers).
    Chosen from the Nyquist margin at 1 kHz sampling, not from the old
    rig-contact concern (moot now — the new base is bolted to the table,
    not suspended on wires it could tilt into): at full battery (4.2 V),
    drone2's M1 fit already puts the shaft fundamental at ~456 Hz for
    55,000 PWM, uncomfortably close to the 500 Hz Nyquist limit; 58,000
    keeps a safety margin without extending far enough to alias the
    fundamental on the faster-running motor.
  - Same ramp/hold durations as enrollment.sweep's combined/individual/pair
    stages (INDIVIDUAL_RAMP_S, HOLD_S, DEFAULT_SWEEP_DURATION, RAMP_DOWN_S)
    — unchanged from what's already validated, just run one stage at a time
    instead of back-to-back.
  - Phase labels written to the timeline reuse enrollment.sweep's existing
    vocabulary (m{id}_only/_hold, {pair}_ramp/_hold, all_four_ramp/_hold,
    ramp_down) so usd_convert.py's phase->motor attribution needs no
    changes — a micro-recording's phase label is exactly what that stage
    would have been called inside the old continuous run_static_sweep().
  - Added 2026-09-20: a per-motor **stepped-hold ladder** (`stepped-m1`..
    `stepped-m4`), missing from the first cut of this module despite being
    agreed on 2026-09-19 — single motor through 7 PWM steps up to the new
    58,000 ceiling, for clean multi-point motor-model characterisation
    (the sweep-m{id} stages only ever give ONE point at 58,000). Phase
    labels use the m{id}_ramp_{pwm}/m{id}_hold_{pwm} form (matches
    usd_convert.py's _PHASE_SINGLE_RE, same vocabulary test_usd_motors.py
    already uses) rather than the bare ramp_to_{pwm}/hold_{pwm} form
    (which usd_convert.py attributes to all four motors — wrong here).
"""

from enrollment.sweep import (
    _ramp, _hold, MOTOR_PAIRS,
    HOLD_S, INDIVIDUAL_RAMP_S, DEFAULT_SWEEP_DURATION, RAMP_DOWN_S,
)
from crazyflie.motors import MotorController

MAX_PWM    = 58_000
COOLDOWN_S = 8.0   # pause between micro-recordings: motor cooling + a little
                   # battery recovery — 2026-09-19 discussion (5-10s range).

STEPPED_PWMS   = [10_000, 15_000, 20_000, 25_000, 35_000, 45_000, MAX_PWM]
STEPPED_RAMP_S = 5.0    # transition between steps — not analysed, just needs
                        # to be short (matches enrollment.sweep's stepped-hold
                        # default)
STEPPED_HOLD_S = 15.0   # per step — same value as test_usd_motors.py's
                        # per-motor HOLD_S, chosen there because 6s gave
                        # several-% spectral-estimation noise even with a
                        # correct peak-picker; 15s keeps enough Welch
                        # averages (nperseg=2048 @ ~1kHz) to avoid that.

# purpose -> ("single", motor_id) | ("pair", MOTOR_PAIRS key) | ("combined", None)
#          | ("stepped", motor_id)
STANDARD_PROTOCOL: "list[tuple[str, tuple[str, object]]]" = (
    [(f"sweep-m{m}", ("single", m)) for m in (1, 2, 3, 4)]
    + [(f"sweep-{label}", ("pair", label)) for label in MOTOR_PAIRS]
    + [("sweep-all4", ("combined", None))]
    + [(f"stepped-m{m}", ("stepped", m)) for m in (1, 2, 3, 4)]
)

# ── TEST-ONLY stage (added 2026-09-20) ──────────────────────────────────────
# A 16th recording that exists only in held-out TEST sessions: all four
# motors driven through a fixed, deliberately un-enrollment-like PWM profile,
# to check the identity is a property of the chassis and not of the specific
# PWM levels / monotone ramp shapes seen at enrollment:
#   - levels are all OFF the enrollment ladders (none of 10/15/20/25/35/45/58k),
#   - the profile goes up AND down (enrollment only ever ramps upward),
#   - fast 1.5 s transitions (enrollment ramps take 5-60 s).
# The profile is fixed, not random, so both drones and both test sessions get
# the identical excitation. All four motors share one PWM at any instant on
# purpose: usd_convert.py reconstructs a single PWM series per phase, so
# per-motor-different PWMs would be mis-attributed and the comb filter would
# notch the wrong lines — an asymmetric variant needs the converter extended
# first. Phase labels ramp_to_{pwm}/hold_{pwm}/ramp_down are the all-four
# vocabulary usd_convert.py already understands. Peak 56,000 < MAX_PWM.
UNSEEN_PROFILE_PWMS = [32_000, 13_000, 52_000, 22_000, 41_000, 17_000,
                       56_000, 28_000, 47_000, 12_000]
UNSEEN_RAMP_S = 1.5
UNSEEN_HOLD_S = 6.0

TEST_ONLY_PROTOCOL: "list[tuple[str, tuple[str, object]]]" = [
    ("unseen-pwm", ("profile", None)),
]

# ── BOARD sessions (2026-10-03): the IMU at rest, motors off ────────────────
# Board-bound fingerprint (MEMS-PUF style), independent of the motors: one
# 30 s recording with every motor OFF, for the sensor's own noise. The values
# the firmware computes at boot (accelerometer self-test response, gyro bias,
# accelerometer scale — custom firmware, param group "uavibe") are read by
# the dashboard at the start of each board session. One board session = one
# boot of the drone.
REST_S = 30.0
BOARD_ONLY_PROTOCOL: "list[tuple[str, tuple[str, object]]]" = [
    ("rest-imu", ("rest", None)),
]
BOARD_PURPOSES = [p for p, _ in BOARD_ONLY_PROTOCOL]

# ── BOOT sessions (2026-10-09): the key-generation boot protocol ────────────
# What the drone does at every power-up to regenerate its key, chosen from the
# existing data (analysis/boot_protocol.py, report protocollo_avvio.md): rest
# phase for the board fingerprint, then each motor ALONE at 58k for a 3 s
# usable hold (+0.5 s settle) - the shortest option with no genuine errors
# and no impostors in the combined motors+board simulation. Rig-bound by
# design (58k on a single motor would tip a free drone). ONE SD recording per
# session; one session = one boot. ~32 s in total.
BOOT_REST_S = 16.0         # was 10 s for the first A/B boot sessions (2026-10-13):
                           # the SD deck stalls once per log for ~0.2+0.7 s somewhere
                           # 7.5-15 s after logging starts, which landed in M1's hold
                           # (lost in A-s01, B-s06). 16 s puts that stall in the
                           # motors-off phase. An SD-logging artifact only: on-board
                           # key generation would not log to the card.
BOOT_PWM = MAX_PWM
BOOT_RAMP_S = 1.0
BOOT_HOLD_S = 3.5          # 0.5 s settle + 3 s analysed
BOOT_PAUSE_S = 1.0
BOOT_ONLY_PROTOCOL: "list[tuple[str, tuple[str, object]]]" = [
    ("boot-seq", ("boot", None)),
]
BOOT_PURPOSES = [p for p, _ in BOOT_ONLY_PROTOCOL]

_PROTOCOL_MAP = dict(STANDARD_PROTOCOL + TEST_ONLY_PROTOCOL + BOARD_ONLY_PROTOCOL + BOOT_ONLY_PROTOCOL)
PURPOSES = [p for p, _ in STANDARD_PROTOCOL]
TEST_PURPOSES = PURPOSES + [p for p, _ in TEST_ONLY_PROTOCOL]


# Automatic mode (dashboard "Automatic recording", 2026-09-27): rest
# between consecutive stages so the battery (on USB charge) recovers and the
# motors cool — 90 s normally, 150 s from the all-four stage onwards (the
# all-four run and the 140 s single-motor ladders that follow it drain and
# heat the most).
AUTO_PAUSE_S = 90.0
AUTO_PAUSE_AFTER_ALL4_S = 150.0


def auto_pause_after(purpose: str, purposes: "list[str]") -> float:
    """Rest to observe after `purpose` in automatic mode: 150 s if the stage
    is sweep-all4 or comes after it in the session's order, else 90 s."""
    if "sweep-all4" in purposes and purposes.index(purpose) >= purposes.index("sweep-all4"):
        return AUTO_PAUSE_AFTER_ALL4_S
    return AUTO_PAUSE_S


# ── MOTOR-SWAP sessions (2026-09-30) ────────────────────────────────────────
# Reduced protocol for the A<->B motor-swap experiment: only the four
# single-motor 58k stages. They carry both read-outs the experiment needs —
# the chassis-band spectrum (the best cross-day chassis representation in the
# four-drone study) and each motor's speed at 58k (motor signature; A's and
# B's motors differ by 5-10 %, enough at a single PWM point) — and every
# enrollment session of A and B already contains them, so the identities to
# compare against exist. ~7 min per session with the 90 s pauses.
SWAP_PURPOSES = [f"sweep-m{m}" for m in (1, 2, 3, 4)]

# Battery gate (2026-09-30): a stage only starts if the drone's RESTING
# voltage is at least this. Fully charged at rest every drone reads
# 4.15-4.22 V; the weakest stages of the four-drone study started at
# 3.74-3.93 V and partly coincide with its worst-identified sessions.
VBAT_MIN_REST_V = 4.10
VBAT_WAIT_MAX_S = 600.0     # automatic mode: extra wait for the charge before giving up
VBAT_RECHECK_S = 15.0


def purposes_for(kind: str) -> "list[str]":
    """Stages of a session: 15 standard for enrollment, those + unseen-pwm
    for a held-out test session, the 4 single-motor stages for a swap one."""
    if kind == "swap":
        return list(SWAP_PURPOSES)
    if kind == "board":
        return list(BOARD_PURPOSES)
    if kind in ("boot", "props", "payload"):     # altered drone, same boot protocol
        return list(BOOT_PURPOSES)
    return list(TEST_PURPOSES if kind == "test" else PURPOSES)


def micro_duration_s(purpose: str) -> float:
    """Excitation-only duration (ramp+hold[+rampdown]) for one purpose —
    used for the progress bar; does not include COOLDOWN_S."""
    kind, _ = _PROTOCOL_MAP[purpose]
    if kind == "combined":
        return DEFAULT_SWEEP_DURATION + HOLD_S + RAMP_DOWN_S
    if kind == "stepped":
        return len(STEPPED_PWMS) * (STEPPED_RAMP_S + STEPPED_HOLD_S)
    if kind == "profile":
        return len(UNSEEN_PROFILE_PWMS) * (UNSEEN_RAMP_S + UNSEEN_HOLD_S) + RAMP_DOWN_S
    if kind == "rest":
        return REST_S
    if kind == "boot":
        return BOOT_REST_S + 4 * (BOOT_RAMP_S + BOOT_HOLD_S + BOOT_PAUSE_S)
    return INDIVIDUAL_RAMP_S + HOLD_S


def run_micro(purpose: str, motors: MotorController, recorder, progress_cb=None) -> None:
    """
    Run exactly one micro-recording — one entry of STANDARD_PROTOCOL.

    recorder receives set_phase() calls (an enrollment.usd_timeline.
    UsdPhaseRecorder in the dashboard's SD-deck path). progress_cb, if
    given, is called as progress_cb(phase, elapsed_s, total_s, remaining_s,
    motor_pwm) — same shape as enrollment.sweep.run_static_sweep's, so
    dashboard code already written for that can drive this too. Motors are
    always left stopped on return (including on an exception — the caller
    should still stop_all() defensively if the connection itself died).
    """
    if purpose not in _PROTOCOL_MAP:
        raise ValueError(f"Unknown purpose {purpose!r}; must be one of {TEST_PURPOSES}")
    kind, arg = _PROTOCOL_MAP[purpose]
    total_s = micro_duration_s(purpose)
    _elapsed = [0.0]

    def tick_cb(phase: str, dt: float, pwms: dict) -> None:
        _elapsed[0] += dt
        if progress_cb is None:
            return
        try:
            progress_cb(phase, _elapsed[0], total_s, max(0.0, total_s - _elapsed[0]), pwms)
        except Exception:
            pass

    if kind == "combined":
        ids = [1, 2, 3, 4]
        _ramp(motors, ids, 0, MAX_PWM, DEFAULT_SWEEP_DURATION, recorder, "all_four_ramp", tick_cb)
        _hold(motors, ids, MAX_PWM, HOLD_S, recorder, "all_four_hold", tick_cb)
        _ramp(motors, ids, MAX_PWM, 0, RAMP_DOWN_S, recorder, "ramp_down", tick_cb)
        motors.stop_all()

    elif kind == "single":
        mid = arg
        _ramp(motors, [mid], 0, MAX_PWM, INDIVIDUAL_RAMP_S, recorder, f"m{mid}_only", tick_cb)
        _hold(motors, [mid], MAX_PWM, HOLD_S, recorder, f"m{mid}_hold", tick_cb)
        motors.stop_motor(mid)

    elif kind == "pair":
        label = arg
        ma, mb = MOTOR_PAIRS[label]
        _ramp(motors, [ma, mb], 0, MAX_PWM, INDIVIDUAL_RAMP_S, recorder, f"{label}_ramp", tick_cb)
        _hold(motors, [ma, mb], MAX_PWM, HOLD_S, recorder, f"{label}_hold", tick_cb)
        motors.stop_motor(ma)
        motors.stop_motor(mb)

    elif kind == "boot":
        motors.stop_all()
        _hold(motors, [], 0, BOOT_REST_S, recorder, "rest_hold", tick_cb)
        for mid in (1, 2, 3, 4):
            _ramp(motors, [mid], 0, BOOT_PWM, BOOT_RAMP_S, recorder, f"m{mid}_ramp_{BOOT_PWM}", tick_cb)
            _hold(motors, [mid], BOOT_PWM, BOOT_HOLD_S, recorder, f"m{mid}_hold_{BOOT_PWM}", tick_cb)
            motors.stop_motor(mid)
            _hold(motors, [], 0, BOOT_PAUSE_S, recorder, "pause", tick_cb)

    elif kind == "rest":
        motors.stop_all()      # every motor OFF for the whole recording
        _hold(motors, [], 0, REST_S, recorder, "rest_hold", tick_cb)

    elif kind == "profile":
        ids = [1, 2, 3, 4]
        prev = 0
        for pwm in UNSEEN_PROFILE_PWMS:
            _ramp(motors, ids, prev, pwm, UNSEEN_RAMP_S, recorder, f"ramp_to_{pwm}", tick_cb)
            _hold(motors, ids, pwm, UNSEEN_HOLD_S, recorder, f"hold_{pwm}", tick_cb)
            prev = pwm
        _ramp(motors, ids, prev, 0, RAMP_DOWN_S, recorder, "ramp_down", tick_cb)
        motors.stop_all()

    else:  # "stepped"
        mid = arg
        prev = 0
        for pwm in STEPPED_PWMS:
            _ramp(motors, [mid], prev, pwm, STEPPED_RAMP_S, recorder, f"m{mid}_ramp_{pwm}", tick_cb)
            _hold(motors, [mid], pwm, STEPPED_HOLD_S, recorder, f"m{mid}_hold_{pwm}", tick_cb)
            prev = pwm
        motors.stop_motor(mid)
