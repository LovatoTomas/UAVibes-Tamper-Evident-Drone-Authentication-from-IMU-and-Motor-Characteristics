"""
Figures and numbers for the thesis section on the motor model (chapter 4):
how the PWM -> speed model was built with the laser tachometer, how the speed is
read back from the vibration, and how much motors differ within and between drones.

    python -m analysis.figures_motor_model
        -> analysis/figures/fig17_modello_motore.png
           analysis/figures/fig18_spettrogramma_scala.png
           analysis/results_motor_model/summary.json

Data: data/calibration/ (laser sessions of drone A, then called drone 1, and of
drone B's M1, then drone 2) and the stepped-hold ladders of data/fixed_rig/.
The base model is refitted here from its 67 points (the two laser sessions of
26 Aug, single and all-motor drive, repeat checks left out) and must match the
constants of pipeline/motor_model.py.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, spectrogram, welch

import os
ROOT = Path(__file__).resolve().parent.parent
CAL = ROOT / "data" / "calibration"
FIG = Path(__file__).resolve().parent / "figures"
OUT = Path(__file__).resolve().parent / "results_motor_model"

# Paper mode: write the figures into the paper's figures/ directory without the
# per-panel titles (the paper caption identifies the panels). Off by default, so
# `python -m analysis.figures_motor_model` and reproduce.py keep the thesis figures.
PAPER = bool(os.environ.get("UAVIBE_PAPER_FIG"))
if PAPER:
    FIG = ROOT.parent / "deliverables" / "paper" / "figures"

from analysis.palette import DRONE as C
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

BASE_FILES = ("rpm_map_20260826_111629.csv", "rpm_map_20260826_114209.csv")
GAIN_FILES = ("rpm_map_20260902_091410.csv", "model_validation_20260902_101021.csv",
              "model_validation_20260902_101334.csv", "model_validation_20260902_110924.csv",
              "model_validation_20260902_111206.csv")
B_FILE = "rpm_map_20260914_185421.csv"
LADDERS = {"A": "data/fixed_rig/droneA/converted/20260923_092521_droneA_s01_stepped-m1_1khz.csv",
           "B": "data/fixed_rig/droneB/converted/20260920_115938_droneB_s02_stepped-m1_1khz.csv"}


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fit(x, y):
    a, b = np.polyfit(x, y, 1)
    p = a * x + b
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(a), float(b), float(r2), float(np.sqrt(((y - p) ** 2).mean()))


def base_rpm(pwm, vbat, a, b):
    return a * pwm * vbat + b


def laser_data():
    base = pd.concat([pd.read_csv(CAL / f) for f in BASE_FILES])
    base = base[base.note.fillna("") != "repeat_vbat_check"]
    a, b, r2, rmse = fit((base.pwm * base.vbat_v).values, base.rpm.values)

    # per-motor gain of drone A: all-motor drive, 35k-55k, laser reading vs base model
    rows = []
    for f in GAIN_FILES:
        d = pd.read_csv(CAL / f)
        rpm = d["rpm"] if "rpm" in d else d["rpm_measured"]
        rows.append(pd.DataFrame(dict(motor=d.motor, drive=d.drive, pwm=d.pwm, vbat=d.vbat_v, rpm=rpm)))
    g = pd.concat(rows)
    g = g[(g.drive == "all") & (g.pwm >= 35000)].dropna(subset=["rpm"]).copy()  # skipped readings
    g["ratio"] = g.rpm / base_rpm(g.pwm, g.vbat, a, b)

    bm = pd.read_csv(CAL / B_FILE)
    bm = bm[bm.note.fillna("") != "repeat_vbat_check"]
    ab, bb, r2b, rmseb = fit((bm.pwm * bm.vbat_v).values, bm.rpm.values)
    bm["ratio"] = bm.rpm / base_rpm(bm.pwm, bm.vbat_v, a, b)
    return base, g, bm, dict(a=a, b=b, r2=r2, rmse=rmse, n=len(base),
                             pwm=[int(base.pwm.min()), int(base.pwm.max())],
                             vbat=[float(base.vbat_v.min()), float(base.vbat_v.max())],
                             B_m1=dict(a=ab, b=bb, r2=r2b, n=len(bm)))


def vibration_vs_laser(model):
    """Speed read from the IMU spectrum vs the laser, same motors, same days."""
    a, b = model["a"], model["b"]
    out = {}
    # drone A, 6 Sep: per-motor gain from the vibration (single motor, 8k-48k)
    v = pd.read_csv(CAL / "vibration_calibration_20260906_104906.csv")
    v["ratio"] = v.rpm_confirmed / base_rpm(v.pwm, v.vbat_v, a, b)
    out["A_vibration_gain"] = {int(m): round(float(x), 3) for m, x in v.groupby("motor").ratio.median().items()}
    # drone B M1, 14 Sep: vibration line vs B's own laser line at the same command and voltage
    vb = pd.read_csv(CAL / "vibration_calibration_20260914_185043.csv")
    vb = vb[vb.motor == 1]
    bl = model["B_m1"]
    laser = bl["a"] * vb.pwm * vb.vbat_v + bl["b"]
    out["B_m1_vibration_minus_laser_pct"] = [round(float(x), 1) for x in 100 * (vb.rpm_confirmed / laser - 1)]
    return out


def fig_model(base, g, bm, model):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURF,
                                   gridspec_kw=dict(width_ratios=[1.5, 1]))
    for ax in (ax1, ax2):
        _style(ax)
    x = base.pwm * base.vbat_v / 1000
    for drive, mk, fc in (("single", "o", C["A"]), ("all", "o", SURF)):
        s = base.drive == drive
        ax1.scatter(x[s], base.rpm[s] / 1000, s=22, marker=mk, facecolor=fc, edgecolor=C["A"], lw=1.2,
                    label=f"drone A, M1, {'motor alone' if drive == 'single' else 'all four motors'}", zorder=3)
    xs = np.linspace(25, 225, 50)
    ax1.plot(xs, (model["a"] * xs * 1000 + model["b"]) / 1000, color=INK, lw=1.6, label="motor model", zorder=2)
    xb = bm.pwm * bm.vbat_v / 1000
    ax1.scatter(xb, bm.rpm / 1000, s=24, marker="s", facecolor=C["B"], edgecolor=SURF, lw=0.8,
                label="drone B, M1, motor alone", zorder=3)
    ax1.set_xlabel("PWM × battery voltage (×1000 V)", color=INK2, fontsize=9)
    ax1.set_ylabel("speed measured by laser (×1000 RPM)", color=INK2, fontsize=9)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    if not PAPER: ax1.set_title("Motor model from the laser tachometer", color=INK, fontsize=10, loc="left")

    rng = np.random.default_rng(0)
    for m in range(1, 5):
        r = g[g.motor == m].ratio.values
        ax2.scatter(m + rng.uniform(-0.12, 0.12, len(r)), r, s=16, color=C["A"], alpha=0.55, lw=0)
        ax2.plot([m - 0.25, m + 0.25], [np.median(r)] * 2, color=C["A"], lw=2.4, zorder=5,
                 path_effects=[pe.Stroke(linewidth=4.5, foreground=SURF), pe.Normal()])
    rb = bm[bm.pwm >= 35000].ratio.values
    ax2.scatter(5 + rng.uniform(-0.12, 0.12, len(rb)), rb, s=20, marker="s", color=C["B"], lw=0)
    ax2.plot([4.75, 5.25], [np.median(rb)] * 2, color=C["B"], lw=2.4, zorder=5,
             path_effects=[pe.Stroke(linewidth=4.5, foreground=SURF), pe.Normal()])
    ax2.axhline(1, color=INK2, lw=0.8, ls="--")
    ax2.set_xticks([1, 2, 3, 4, 5], ["A M1", "A M2", "A M3", "A M4", "B M1"])
    ax2.set_ylabel("measured / predicted speed", color=INK2, fontsize=9)
    if not PAPER: ax2.set_title("Same model, different motors (35k-55k PWM)", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig17_modello_motore.png", dpi=200, facecolor=SURF)
    plt.close(fig)
    return {f"A_M{m}": round(float(g[g.motor == m].ratio.median()), 3) for m in range(1, 5)} | \
        {"B_M1": round(float(np.median(rb)), 3), "points_per_motor_A": {int(m): int(n) for m, n in g.motor.value_counts().sort_index().items()}}


def fig_ladders(model):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURF, sharey=True)
    peaks = {}
    for ax, (drone, path) in zip(axes, LADDERS.items()):
        d = pd.read_csv(ROOT / path)
        d = d[d.phase != "usd_log"]
        t = d.timestamp_s.values - d.timestamp_s.values[0]
        fs = 1 / np.median(np.diff(t))
        sos = butter(4, 5, "highpass", fs=fs, output="sos")
        P = 0
        for ch in ("acc.x", "acc.y", "acc.z"):
            f, tt, S = spectrogram(sosfiltfilt(sos, d[ch].values), fs=fs, nperseg=1024, noverlap=768)
            P = P + S
        db = 10 * np.log10(P + 1e-12)
        # the firmware low-passes the IMU, so high lines are weak: show each frequency
        # relative to its own median level over the recording
        db = db - np.median(db, axis=1, keepdims=True)
        ax.pcolormesh(tt, f, db, shading="auto", cmap="Greys", vmin=5, vmax=35,
                      rasterized=True)
        pred = (model["a"] * d["motor.m1"].values * d.vbat_v.values + model["b"]) / 60
        pred[d["motor.m1"].values < 500] = np.nan
        shown = np.where(np.char.find(d.phase.values.astype(str), "hold") >= 0, pred, np.nan)
        ax.plot(t, shown, color=C[drone], lw=1.3, ls=(0, (3, 3)), label="motor model prediction")
        ax.set_ylim(0, 500)
        ax.set_xlabel("time (s)", color=INK2, fontsize=9)
        if not PAPER: ax.set_title(f"Drone {drone}, M1 alone, seven commands from 10k to 58k PWM", color=INK, fontsize=10, loc="left")
        if PAPER: ax.text(0.03, 0.95, f"drone {drone}", transform=ax.transAxes, ha="left", va="top", fontsize=9, color=INK)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.legend(frameon=True, facecolor=SURF, edgecolor=GRID, fontsize=8, loc="lower right", labelcolor=INK)
        # shaft line at the 58k hold: strongest bin near the prediction
        h = d.phase.values == "m1_hold_58000"
        fp = np.nanmedian(pred[h])
        th = (tt >= t[h][0] + 1) & (tt <= t[h][-1])
        band = (f > 0.85 * fp) & (f < 1.25 * fp)
        peaks[drone] = dict(predicted_hz=round(float(fp), 1),
                            measured_hz=round(float(f[band][np.argmax(P[band][:, th].mean(axis=1))]), 1))
    axes[0].set_ylabel("frequency (Hz)", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig18_spettrogramma_scala.png", dpi=200, facecolor=SURF)
    plt.close(fig)
    return peaks


MANUAL_CASES = (("log05_1khz.csv", "m1_hold_24000", "a clear line"),
                 ("log08_1khz.csv", "m4_hold_40000", "a second strong peak"),
                 ("log07_1khz.csv", "m3_hold_40000", "two peaks 2 Hz apart"))


def fig_manual(model):
    """The view of the manual calibration (dashboard, Motor Calibration tab): the
    spectrum of one hold, the candidate peaks near the prediction ranked by the same
    scorer the tab used, and the peak the operator confirmed (6 Sep, drone A)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from usd_motor_analysis import rank_candidates
    v = pd.read_csv(CAL / "vibration_calibration_20260906_104906.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), facecolor=SURF)
    for ax, (src, phase, what) in zip(axes, MANUAL_CASES):
        _style(ax)
        d = pd.read_csv(ROOT / "data" / "usd" / src)
        seg = d[d.phase == phase]
        fs = 1 / np.median(np.diff(seg.timestamp_s.values))
        row = v[(v.source_csv == src) & (v.phase == phase)].iloc[0]
        pred = (model["a"] * row.pwm * row.vbat_v + model["b"]) / 60
        x = seg["acc.x"].values.astype(float)
        f, psd = welch(x - x.mean(), fs=fs, nperseg=min(2048, len(x)))
        db = 10 * np.log10(np.maximum(psd, 1e-20))
        w = (f > 0.75 * pred) & (f < 1.25 * pred)
        db = db - db[w].max()
        ax.plot(f[w], db[w], color=INK2, lw=1.0, zorder=2)
        ax.axvline(pred, color=INK, lw=1.2, ls=(0, (3, 3)), zorder=3)
        ranked = rank_candidates(seg, fs, pred)
        groups = []  # candidates closer than 4 Hz share one label
        for i, (pk, _, _) in enumerate(ranked[:4]):
            for gr in groups:
                if abs(gr[0] - pk) < 4:
                    gr[1].append(i + 1)
                    break
            else:
                groups.append([pk, [i + 1]])
        for pk, ids in groups:
            y = db[w][np.abs(f[w] - pk) < 4].max()
            ax.annotate(", ".join(f"#{k}" for k in ids), (pk, y), xytext=(7, 3), textcoords="offset points",
                        ha="left", fontsize=7.5, color=INK, zorder=5,
                        bbox=dict(boxstyle="square,pad=0.1", fc=SURF, ec="none"))
        ax.axvline(row.f0_confirmed_hz, color=C["A"], lw=2.0, alpha=0.85, zorder=1)
        ax.set_ylim(-55, 9)
        if not PAPER: ax.set_title(f"M{row.motor} at {row.pwm // 1000}k PWM: {what}", color=INK, fontsize=9.5, loc="left")
        if PAPER: ax.text(0.03, 0.96, f"M{row.motor} at {row.pwm // 1000}k PWM", transform=ax.transAxes, ha="left", va="top", fontsize=9, color=INK)
        ax.set_xlabel("frequency (Hz)", color=INK2, fontsize=9)
    axes[0].set_ylabel("level (dB, peak = 0)", color=INK2, fontsize=9)
    axes[2].plot([], [], color=C["A"], lw=2, label="line confirmed by the operator")
    axes[2].plot([], [], color=INK, lw=1.2, ls=(0, (3, 3)), label="motor model prediction")
    axes[2].legend(frameon=True, facecolor=SURF, edgecolor=GRID, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig19_spettro_calibrazione.png", dpi=200, facecolor=SURF)
    plt.close(fig)
    return dict(points=int(len(v)), methods={k: int(n) for k, n in v.method.value_counts().items()})


def main():
    OUT.mkdir(exist_ok=True)
    base, g, bm, model = laser_data()
    summary = dict(model=model, laser_gain=fig_model(base, g, bm, model),
                   vibration=vibration_vs_laser(model), ladder_58k=fig_ladders(model),
                   manual_calibration=fig_manual(model))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
