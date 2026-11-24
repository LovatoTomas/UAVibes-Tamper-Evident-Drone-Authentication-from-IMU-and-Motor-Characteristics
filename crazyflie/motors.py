"""
Individual motor control via the Crazyflie motorPowerSet parameter group.

motorPowerSet.enable = 1 bypasses normal flight logic and lets the host
set each motor's PWM duty directly (0-65535).  Always cleared on stop_all()
so the firmware returns to normal operation.
"""

import time
import threading
from cflib.crazyflie import Crazyflie

MAX_POWER  = 65535
MIN_POWER  = 0
POWER_STEP = 1000

MOTOR_INFO: dict[int, tuple[str, str, str]] = {
    1: ("M1", "Front-Left ", "CCW"),
    2: ("M2", "Front-Right", "CW "),
    3: ("M3", "Rear-Right ", "CCW"),
    4: ("M4", "Rear-Left  ", "CW "),
}


class MotorController:
    """Thread-safe wrapper around the motorPowerSet parameter group."""

    _ENABLE = "motorPowerSet.enable"
    _PARAMS = {
        1: "motorPowerSet.m1",
        2: "motorPowerSet.m2",
        3: "motorPowerSet.m3",
        4: "motorPowerSet.m4",
    }

    def __init__(self, cf: Crazyflie) -> None:
        self._cf      = cf
        self._powers  = {mid: 0 for mid in self._PARAMS}
        self._enabled = False
        self._lock    = threading.Lock()

    def _set_param(self, name: str, value) -> None:
        self._cf.param.set_value(name, str(value))

    def _enable_override(self) -> None:
        if not self._enabled:
            self._set_param(self._ENABLE, 1)
            self._enabled = True

    def _disable_override(self) -> None:
        if self._enabled:
            self._set_param(self._ENABLE, 0)
            self._enabled = False

    def set_motor(self, motor_id: int, power: int) -> None:
        """Set a single motor to the requested power (clamped to 0-65535)."""
        power = max(MIN_POWER, min(MAX_POWER, power))
        with self._lock:
            self._enable_override()
            self._set_param(self._PARAMS[motor_id], power)
            self._powers[motor_id] = power

    def stop_motor(self, motor_id: int) -> None:
        """Set one motor to zero (does not disable override)."""
        with self._lock:
            self._set_param(self._PARAMS[motor_id], 0)
            self._powers[motor_id] = 0

    def stop_all(self) -> None:
        """
        Zero all motors and release the motorPowerSet override.

        Commands are paced with a small delay rather than fired back-to-back:
        an unpaced burst of 5 rapid param writes right as a sweep ends was a
        plausible contributor to "too many packets lost" link errors
        observed immediately after long sweeps.
        """
        with self._lock:
            for mid, param in self._PARAMS.items():
                self._set_param(param, 0)
                self._powers[mid] = 0
                time.sleep(0.02)
            self._disable_override()

    def get_power(self, motor_id: int) -> int:
        with self._lock:
            return self._powers[motor_id]

    def get_all_powers(self) -> dict[int, int]:
        with self._lock:
            return self._powers.copy()
