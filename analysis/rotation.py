"""
Rotation direction of each motor from the IMU alone (2026-10-01).

A rotor's residual imbalance produces an in-plane force that ROTATES with the
shaft: at the shaft frequency f0 the horizontal accelerations acc.x / acc.y
trace an ellipse whose sense of rotation is the rotor's. The sign of the
imaginary part of the cross-spectrum S_xy(f0) = E[X(f0) conj(Y(f0))] therefore
flips when the motor spins the other way (same for gyro.x / gyro.y).

Used to check the motor-swap sessions: on a Crazyflie adjacent motors spin in
opposite directions and the direction is fixed by the motor's wiring, so a
motor mounted in the wrong position (e.g. M3 <-> M4) spins backwards — its
sign flips relative to the correctly assembled sessions.
"""

import numpy as np
import pandas as pd
from scipy.signal import csd, welch

from pipeline.preprocessing import load_session, infer_fs, highpass
from pipeline.motor_model import pwm_to_hz

SEARCH = 0.15


def direction_of_hold(seg: pd.DataFrame, fs: float, motor: int) -> "dict | None":
    pwm = float(seg[f"motor.m{motor}"].mean())
    vbat = float(seg["vbat_v"].mean())
    f_pred = pwm_to_hz(pwm, vbat)
    ax, ay = seg["noise.acc.x"].values, seg["noise.acc.y"].values
    gx, gy = seg["noise.gyro.x"].values, seg["noise.gyro.y"].values
    nper = min(1024, len(ax))
    f, pxx = welch(ax, fs=fs, nperseg=nper)
    _, pyy = welch(ay, fs=fs, nperseg=nper)
    band = (f > f_pred * (1 - SEARCH)) & (f < f_pred * (1 + SEARCH))
    if not band.any():
        return None
    k = int(np.argmax(np.where(band, pxx + pyy, -np.inf)))
    _, sxy = csd(ax, ay, fs=fs, nperseg=nper)
    _, gxy = csd(gx, gy, fs=fs, nperseg=nper)
    # normalised: +1 / -1 = perfectly circular motion one way / the other
    acc_rot = float(sxy[k].imag / np.sqrt(pxx[k] * pyy[k]))
    _, gxx = welch(gx, fs=fs, nperseg=nper)
    _, gyy = welch(gy, fs=fs, nperseg=nper)
    gyr_rot = float(gxy[k].imag / np.sqrt(gxx[k] * gyy[k]))
    return dict(f0=float(f[k]), acc_rot=acc_rot, gyro_rot=gyr_rot)


def directions(path: str, motor: int) -> "dict | None":
    df = load_session(path)
    fs = infer_fs(df)
    hp = highpass(df, fs)
    seg = hp[hp.phase == f"m{motor}_hold"]
    if len(seg) < 2000:
        return None
    return direction_of_hold(seg, fs, motor)
