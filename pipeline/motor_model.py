"""
UAVibes — Empirical motor model: PWM + battery voltage -> RPM / frequency.
==========================================================================

THE canonical PWM->RPM map for the project. Every conversion from motor
command to rotation rate (comb-filter f0, dashboard RPM display, analysis
scripts) must go through this module — do not hard-code slopes elsewhere.

The model
---------
    RPM_base   = RPM_SLOPE * (PWM * vbat) + RPM_OFFSET
    RPM(motor) = RPM_base * MOTOR_GAIN[motor_id]         (motor_id given)

A brushed coreless DC motor's speed follows the *effective* voltage it
sees, which is duty cycle x battery voltage — hence the PWM*vbat product.
This single formula holds regardless of how many motors are running: the
apparent "4-motors-spin-slower" effect is entirely battery sag (4x current
draw lowers vbat under load), which the vbat term absorbs.

Provenance — base model (laser-tachometer calibration, 2026-08-26)
--------------------------------------------------------------------
Measured on Crazyflie 2.1, motor M1, stock 2-blade 45 mm props, reflective
foil on both blades (tach reading / 2 = true RPM). 67 valid points across
PWM 8 000-55 000 and vbat 2.95-4.06 V, in both single-motor and all-four
drive. Pooled fit quality: R^2 = 0.995, RMSE = 324 RPM (~5.4 Hz shaft
rate). Battery sensitivity ~3 000 RPM/V. The previous assumption
(RPM = PWM/65535 * 22000, linear through zero) was off by 1.3-2.6x and had
the wrong shape (the true map has a large intercept: once static friction
is overcome the motor jumps to thousands of RPM).

Data:   auth_client/data/calibration/rpm_map_20260826_111629.csv (single)
        auth_client/data/calibration/rpm_map_20260826_114209.csv (all)
Script: auth_client/rpm_calibration.py  (re-run to recalibrate — e.g. for
        a different airframe, motor set, or prop set; this map is
        platform-specific even though its FORM generalises)

Provenance — per-motor gain, MOTOR_GAIN (2026-09-02, four sessions)
--------------------------------------------------------------------
The base model was calibrated on M1 alone (single-motor drive). Validating
it against the other three motors — with all four spinning together at
PWM 35 000-55 000, the actual flight-relevant regime, since the drone only
leaves the ground above roughly 45 000 PWM — showed a small but real and
reproducible per-motor deviation, stable across two charge cycles (47
points total, vbat 3.14-3.82 V):

    M1: k=0.973   M2: k=0.990   M3: k=1.041   M4: k=1.049

After applying MOTOR_GAIN the residual has no remaining correlation with
vbat (r=0.05) or PWM (r=-0.04), and a 2-parameter (gain + additive offset)
fit gave no meaningful improvement over the gain alone — so a single
multiplicative factor per motor, with no additive term, fully explains the
deviation over this range. Residual scatter after correction: 1.0-2.4%,
i.e. already down at the laser-tachometer's own measurement noise floor.
M1's own gain (0.973, not 1.0) captures a small systematic effect present
even for the calibration motor itself once all four spin together — likely
the real voltage at each motor's ESC sagging slightly more under 4x
current draw than pm.vbat (sampled once, upstream) reflects.

Data:   auth_client/data/calibration/model_validation_20260902_101021.csv
        auth_client/data/calibration/model_validation_20260902_101334.csv
        auth_client/data/calibration/model_validation_20260902_110924.csv
        auth_client/data/calibration/model_validation_20260902_111206.csv
Script: auth_client/model_validation.py --random-n N  (randomized per-round
        motor+PWM draw, all four always spinning, weighted toward the
        flight-relevant 35k-55k PWM pool)

Validity and limits
-------------------
- Base model calibrated over PWM 8 000-55 000; outside that range values
  are extrapolations (below ~8 000 the intercept makes them physically
  crude).
- MOTOR_GAIN was derived specifically from the 35 000-55 000, all-four-
  drive regime — that is also the regime that matters for enrollment and
  authentication (the drone flies with all four motors on, above ~45 000
  PWM), so no extrapolation is needed for the primary use case. Applying
  it at low PWM or single-motor drive is an unverified extrapolation of
  the gain table, not of the base model.
- PWM below MIN_SPIN_PWM is treated as motor stopped -> 0.
- motor_id is optional and defaults to None (no gain applied — the bare
  base-model prediction, unchanged from every caller written before
  2026-09-02). Pass motor_id in {1,2,3,4} to apply the validated
  correction; this is what harmonic_deconvolution now does per motor
  channel (see preprocessing.py).
- MCU-friendly by design: 2 multiplies + 1 add for the base model (+ 1
  more multiply when motor_id is given), and every input (own PWM
  command, pm.vbat, own motor index) is already known to the drone's
  firmware — the same formula is directly portable on-board.
"""

# Pooled least-squares fit, single + all drive, 2026-08-26 session.
RPM_SLOPE    = 0.09336   # RPM per (PWM count x Volt)
RPM_OFFSET   = 4321.0    # RPM
NOMINAL_VBAT = 3.7       # V — used when no live battery reading is available

MIN_SPIN_PWM = 500       # below this the motor is considered stopped
PWM_CAL_RANGE = (8_000, 55_000)   # range actually covered by calibration

# Per-motor multiplicative correction vs the M1-calibrated base model,
# validated in the all-four-drive, 35k-55k PWM (flight) regime — see
# provenance above. Motor 1 is included (0.973, not 1.0): even the
# calibration motor itself runs a bit slower than its own single-drive
# model once all four motors share the battery.
MOTOR_GAIN = {1: 0.973, 2: 0.990, 3: 1.041, 4: 1.049}
MOTOR_GAIN_PWM_RANGE = (35_000, 55_000)   # regime the gains were validated in

N_BLADES = 2             # stock Crazyflie 2.1 props


def pwm_to_rpm(pwm: float, vbat: "float | None" = None,
                motor_id: "int | None" = None) -> float:
    """
    Motor shaft speed in RPM for a given PWM command (0-65535).

    vbat is the battery voltage under load (pm.vbat); pass None to use
    NOMINAL_VBAT. motor_id (1-4) applies the validated per-motor
    MOTOR_GAIN correction; None (default) returns the bare base-model
    prediction, unchanged from before MOTOR_GAIN existed. Returns 0.0
    when the motor is stopped (pwm < MIN_SPIN_PWM).
    """
    if pwm < MIN_SPIN_PWM:
        return 0.0
    v = NOMINAL_VBAT if vbat is None else float(vbat)
    rpm = RPM_SLOPE * (float(pwm) * v) + RPM_OFFSET
    if motor_id is not None:
        rpm *= MOTOR_GAIN[motor_id]
    return rpm


def pwm_to_hz(pwm: float, vbat: "float | None" = None,
              motor_id: "int | None" = None) -> float:
    """Shaft rotation frequency in Hz (= RPM / 60). 0.0 when stopped."""
    return pwm_to_rpm(pwm, vbat, motor_id) / 60.0


def blade_pass_hz(pwm: float, vbat: "float | None" = None,
                   motor_id: "int | None" = None) -> float:
    """
    Blade-pass frequency in Hz: the dominant aerodynamic forcing line,
    N_BLADES x shaft rate (2-blade props -> 2x).
    """
    return N_BLADES * pwm_to_hz(pwm, vbat, motor_id)


# ── Model registry — swappable named calibrations ───────────────────────────
#
# Everything above this point is "the model" as a set of module-level
# constants — fine as long as there's only ever one. As of 2026-09-07 there
# are several in circulation (the original laser-only base fit, the
# per-motor-gain refinement, and whatever the vibration-based human-in-the-
# loop tool in the dashboard produces next) and the dashboard needs to
# switch between them live to compare, e.g., how the Spectrogram tab's
# predicted lines land under each one. Rather than keep a second parallel
# set of "current" variables, set_active_model() below REBINDS RPM_SLOPE/
# RPM_OFFSET/NOMINAL_VBAT and MUTATES MOTOR_GAIN in place, so pwm_to_rpm()
# above (and anyone who reads these constants live, e.g. via the module
# rather than a stale `from motor_model import RPM_SLOPE`) keeps working
# unchanged — the "active model" IS just these same constants, editable.
#
# Only MOTOR_GAIN is ever imported by name elsewhere (`from
# pipeline.motor_model import MOTOR_GAIN`, in ui/app.py and
# model_validation.py) — mutating it in place (.clear()+.update()) rather
# than rebinding is what keeps those call sites in sync automatically,
# since they hold a reference to the same dict object. RPM_SLOPE/
# RPM_OFFSET/NOMINAL_VBAT are never imported by name anywhere else in the
# codebase (checked 2026-09-07), so rebinding them is safe.

import json
import dataclasses
from datetime import datetime
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "calibration" / "models"
_ACTIVE_MARKER = MODELS_DIR / "_active.txt"


@dataclasses.dataclass
class MotorModel:
    """One named, saveable calibration — everything set_active_model() needs."""
    name: str
    description: str = ""
    rpm_slope: float = RPM_SLOPE
    rpm_offset: float = RPM_OFFSET
    nominal_vbat: float = NOMINAL_VBAT
    motor_gain: "dict[int, float]" = dataclasses.field(
        default_factory=lambda: {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    source: str = ""
    created_iso: str = ""

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["motor_gain"] = {str(k): v for k, v in self.motor_gain.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MotorModel":
        d = dict(d)
        d["motor_gain"] = {int(k): float(v) for k, v in d.get("motor_gain", {}).items()}
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls) if f.name in d})


# The model these module constants describe right now (== current values
# above) until something else is activated or restored from disk.
_ACTIVE_MODEL = MotorModel(
    name="laser_gain_v2",
    description="Base laser model + per-motor gain from all-drive validation.",
    rpm_slope=RPM_SLOPE, rpm_offset=RPM_OFFSET, nominal_vbat=NOMINAL_VBAT,
    motor_gain=dict(MOTOR_GAIN),
    source="rpm_calibration.py + model_validation.py, 47 points, PWM 35-55k all-drive",
    created_iso="2026-09-02",
)


def get_active_model() -> MotorModel:
    return _ACTIVE_MODEL


def set_active_model(model: MotorModel, persist: bool = True) -> None:
    """
    Make `model` the one pwm_to_rpm() etc. use, dashboard-wide (this is a
    plain module global, not per-Streamlit-session state — with one drone
    and one operator that's the right scope: every tab should agree on
    which model is active). persist=False skips the marker-file write
    (used internally when restoring the already-persisted choice at
    startup, to avoid a redundant identical write).
    """
    global RPM_SLOPE, RPM_OFFSET, NOMINAL_VBAT, _ACTIVE_MODEL
    _ACTIVE_MODEL = model
    RPM_SLOPE = model.rpm_slope
    RPM_OFFSET = model.rpm_offset
    NOMINAL_VBAT = model.nominal_vbat
    MOTOR_GAIN.clear()
    MOTOR_GAIN.update(model.motor_gain)
    if persist:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _ACTIVE_MARKER.write_text(model.name, encoding="utf-8")


def list_saved_models() -> "list[MotorModel]":
    """All models saved in MODELS_DIR, in filename order."""
    if not MODELS_DIR.exists():
        return []
    out = []
    for p in sorted(MODELS_DIR.glob("*.json")):
        try:
            out.append(MotorModel.from_dict(json.loads(p.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return out


def save_model(model: MotorModel) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{model.name}.json"
    path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    return path


def delete_model(name: str) -> None:
    p = MODELS_DIR / f"{name}.json"
    if p.exists():
        p.unlink()


_bootstrapped = False


def ensure_bootstrapped() -> None:
    """
    Idempotent (safe to call on every dashboard rerun — only the first call
    per process does anything). Seeds the two known historical models into
    MODELS_DIR the first time it's empty, and restores whichever model was
    last marked active so the choice survives a dashboard restart.

    Deliberately NOT called automatically at import time: only the
    dashboard (ui/state.init_state()) calls this, so a plain CLI script
    that just needs pwm_to_rpm() (usd_convert.py, rpm_calibration.py, ...)
    imports this module with zero disk I/O and sees exactly the hardcoded
    defaults above, same as before this registry existed.
    """
    global _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not any(MODELS_DIR.glob("*.json")):
        save_model(MotorModel(
            name="laser_base_v1",
            description="Original single-motor (M1) laser calibration — no per-motor gain.",
            rpm_slope=0.09336, rpm_offset=4321.0, nominal_vbat=3.7,
            motor_gain={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            source="rpm_calibration.py, 67 points, PWM 8k-55k, single+all drive, M1 only",
            created_iso="2026-08-26",
        ))
        save_model(MotorModel(
            name="laser_gain_v2",
            description="Base laser model + per-motor gain from all-drive validation.",
            rpm_slope=0.09336, rpm_offset=4321.0, nominal_vbat=3.7,
            motor_gain={1: 0.973, 2: 0.990, 3: 1.041, 4: 1.049},
            source="rpm_calibration.py + model_validation.py, 47 points, "
                   "PWM 35-55k all-drive",
            created_iso="2026-09-02",
        ))

    if _ACTIVE_MARKER.exists():
        name = _ACTIVE_MARKER.read_text(encoding="utf-8").strip()
        for m in list_saved_models():
            if m.name == name:
                set_active_model(m, persist=False)
                break
