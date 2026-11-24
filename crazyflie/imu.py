"""
Extended IMU logger for UAVibes — streams acc, gyro, and motor PWM.

Motor PWM is read from a MotorController reference passed to start(),
not from a firmware log variable. Earlier versions logged the firmware
"motor.m1-m4" log group — that variable reflects the flight-mixer's
output and is never updated when motors are driven via the
motorPowerSet.m1-m4 *override* parameters (see crazyflie/motors.py),
which is the only path UAVibes ever uses. The result was silent: every
recorded motor.m1-m4 column was always 0, so harmonic_deconvolution()'s
"has_motors" check always failed and Step 3 of the pipeline never ran on
any real hardware recording. Since MotorController already knows the
exact PWM it just commanded, sourcing it locally is both simpler (one
LogConfig block instead of two) and correct by construction — no
firmware round-trip, no CRTP 26-byte packing concerns for a second block.

Callback signature: Callable[[float, dict], None]
  float — wall-clock timestamp in seconds (time.time())
  dict  — keys: acc.x, acc.y, acc.z, gyro.x, gyro.y, gyro.z,
                motor.m1, motor.m2, motor.m3, motor.m4
"""

import time
import threading
from collections.abc import Callable
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

from crazyflie.motors import MotorController

FREQ_OPTIONS = [2, 5, 10, 25, 50, 100]
DEFAULT_FREQ = 100

_PERIOD_MS: dict[int, int] = {
    2:   500,
    5:   200,
    10:  100,
    25:   40,
    50:   20,
    100:  10,
}

_IMU_VARS = [
    ("acc.x",  "float"),
    ("acc.y",  "float"),
    ("acc.z",  "float"),
    ("gyro.x", "float"),
    ("gyro.y", "float"),
    ("gyro.z", "float"),
]

SampleCallback = Callable[[float, dict], None]


class ImuLogger:
    """Manages the IMU LogConfig block and merges in live motor PWM."""

    def __init__(self, cf: Crazyflie) -> None:
        self._cf         = cf
        self._lock       = threading.Lock()
        self._imu_data:  dict = {}
        self._cfg_imu:   LogConfig | None = None
        self._freq       = DEFAULT_FREQ
        self._cb:        SampleCallback | None = None
        self._motors:    "MotorController | None" = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self, freq: int = DEFAULT_FREQ,
              callback: SampleCallback | None = None,
              motors: "MotorController | None" = None) -> bool:
        self._cb     = callback
        self._motors = motors
        return self._attach(freq)

    def stop(self) -> None:
        self._detach()

    def set_frequency(self, freq: int) -> bool:
        """Hot-swap to a new sampling rate. Returns True on success."""
        if freq not in _PERIOD_MS:
            return False
        if freq == self._freq:
            return True
        old_freq = self._freq
        self._detach()
        if not self._attach(freq):
            self._attach(old_freq)
            return False
        return True

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def latest(self) -> dict:
        with self._lock:
            return {**self._imu_data, **self._motor_dict()}

    @property
    def freq(self) -> int:
        return self._freq

    @property
    def has_motors(self) -> bool:
        return self._motors is not None

    # ── internal ──────────────────────────────────────────────────────────────

    def _motor_dict(self) -> dict:
        if self._motors is None:
            return {"motor.m1": 0, "motor.m2": 0, "motor.m3": 0, "motor.m4": 0}
        powers = self._motors.get_all_powers()
        return {f"motor.m{mid}": powers.get(mid, 0) for mid in (1, 2, 3, 4)}

    def _attach(self, freq: int) -> bool:
        period = _PERIOD_MS[freq]

        cfg_imu = LogConfig(name="UAVibeIMU", period_in_ms=period)
        for name, typ in _IMU_VARS:
            cfg_imu.add_variable(name, typ)
        cfg_imu.data_received_cb.add_callback(self._on_imu)
        try:
            self._cf.log.add_config(cfg_imu)
            cfg_imu.start()
        except Exception:
            return False
        self._cfg_imu = cfg_imu
        self._freq    = freq
        return True

    def _detach(self) -> None:
        if self._cfg_imu is not None:
            try:
                self._cfg_imu.stop()
            except Exception:
                pass
            try:
                self._cfg_imu.delete()
            except Exception:
                pass
        self._cfg_imu = None

    def _on_imu(self, _ts, data: dict, _cfg) -> None:
        now = time.time()
        with self._lock:
            self._imu_data = data.copy()
            merged = {**self._imu_data, **self._motor_dict()}
        if self._cb:
            self._cb(now, merged)
