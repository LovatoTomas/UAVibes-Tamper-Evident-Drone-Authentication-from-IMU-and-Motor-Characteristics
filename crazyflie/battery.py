"""
One-shot reads of slow on-board quantities, done right before a recording
stage — never continuously:

  read_vbat(cf)          resting battery voltage (pm.vbat) for the battery gate
  read_log_float(cf, v)  median of a few samples of any float log variable
                         (e.g. baro.temp, the board temperature)
  read_board_params(cf)  the board fingerprint the custom firmware computes at
                         every boot (param group "uavibe": accelerometer
                         self-test response, gyro bias, accelerometer scale)

pm.vbat / baro.temp are not in the live IMU telemetry: the ImuLogger block
already carries six floats (24 of the 26 bytes a CRTP log packet allows), so
a tiny separate LogConfig is added, a few samples collected, and removed.
"""

import threading

from cflib.crazyflie.log import LogConfig

BOARD_PARAMS = ("stDone", "stResult", "stPosX", "stPosY", "stPosZ", "stNegX", "stNegY", "stNegZ",
                "gBiasX", "gBiasY", "gBiasZ", "gBiasOk", "accScale", "accScaleOk")


def read_log_float(cf, name: str, n_samples: int = 5, timeout: float = 2.5,
                   period_ms: int = 100) -> "float | None":
    """Median of n samples of a float log variable, or None if the read fails."""
    vals: "list[float]" = []
    done = threading.Event()

    def _cb(_ts, data, _cfg):
        vals.append(float(data[name]))
        if len(vals) >= n_samples:
            done.set()

    cfg = LogConfig(name="UAVibeOneShot", period_in_ms=period_ms)
    cfg.add_variable(name, "float")
    try:
        cf.log.add_config(cfg)
        cfg.data_received_cb.add_callback(_cb)
        cfg.start()
        done.wait(timeout)
    except Exception:
        return None
    finally:
        try:
            cfg.stop()
            cfg.delete()
        except Exception:
            pass
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def read_vbat(cf, n_samples: int = 5, timeout: float = 2.5) -> "float | None":
    """Median of n pm.vbat samples at 10 Hz, or None if the read fails."""
    return read_log_float(cf, "pm.vbat", n_samples, timeout)


def read_board_params(cf) -> "dict | None":
    """The 'uavibe' param group of the custom firmware, or None when the drone
    runs a firmware without it."""
    out = {}
    for p in BOARD_PARAMS:
        try:
            v = cf.param.get_value(f"uavibe.{p}", timeout=3)
        except Exception:
            return None
        if v is None:
            return None
        try:
            out[p] = float(v)
        except (TypeError, ValueError):
            return None
    return out
