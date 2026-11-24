"""
Figures of the thesis evaluation chapter (8) that no study script draws:
speed shift of every motor in the on-board authentications, with the alarm threshold.

    python -m analysis.figures_evaluation   (after python -m analysis.onboard_log)
        -> analysis/figures/fig21_spostamenti_velocita.png

Data: analysis/results_onboard_log/attempts.csv (station log of the drones running FW2).
Excluded, as in the log analysis: D's day with a damaged motor (excluded_hw_fault) and the
boots of the enrollment discarded for the tick-sampling bug (excluded_fw_bug).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
from analysis.palette import DRONE as C
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
THRESHOLD = 5.0


def fig_speed_shifts():
    d = pd.read_csv(HERE / "results_onboard_log" / "attempts.csv")
    d = d[d.claimed.isin(list("ABCD"))]
    fig, ax = plt.subplots(figsize=(10, 3.8), facecolor=SURF)
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    rng = np.random.default_rng(0)
    ticks, labels = [], []
    for i, dr in enumerate("ABCD"):
        for m in range(1, 5):
            x = i * 5 + (m - 1)
            ticks.append(x)
            labels.append(f"M{m}")
            col = f"shift_m{m}"
            ok = d[(d.claimed == dr) & (d.condition == "nominal") & (d.outcome == "accepted")][col].dropna()
            ax.scatter(x + rng.uniform(-0.28, 0.28, len(ok)), ok, s=9, color=C[dr], alpha=0.55, lw=0, zorder=3)
            al = d[(d.claimed == dr) & (d.condition == "nominal") & (d.outcome == "speed_alarm")][col].dropna()
            al = al[al.abs() > THRESHOLD]
            ax.scatter([x] * len(al), al, s=40, marker="x", color=INK, lw=1.4, zorder=4)
            pr = d[(d.claimed == dr) & (d.condition == "propeller")][col].dropna()
            pr = pr[pr.abs() > THRESHOLD]
            ax.scatter([x] * len(pr), pr, s=46, marker="D", facecolor="none", edgecolor=INK, lw=1.3, zorder=4)
        ax.text(i * 5 + 1.5, 7.2, f"drone {dr}", ha="center", fontsize=9, color=INK)
    for y in (THRESHOLD, -THRESHOLD):
        ax.axhline(y, color=INK2, lw=1, ls="--", zorder=2)
    ax.text(18.7, THRESHOLD + 0.3, "alarm at ±5σ", ha="right", fontsize=8, color=INK2)
    ax.set_xticks(ticks, labels)
    ax.set_xlim(-0.8, 18.8)
    ax.set_ylim(-19.5, 8.2)
    ax.set_ylabel("speed shift from the reference (σ)", color=INK2, fontsize=9)
    ax.scatter([], [], s=12, color=INK2, label="accepted nominal boot")
    ax.scatter([], [], s=40, marker="x", color=INK, label="nominal boot denied by the alarm")
    ax.scatter([], [], s=46, marker="D", facecolor="none", edgecolor=INK, label="different propeller")
    ax.legend(frameon=True, facecolor=SURF, edgecolor=GRID, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig21_spostamenti_velocita.png", dpi=200, facecolor=SURF)
    plt.close(fig)


def main():
    fig_speed_shifts()


if __name__ == "__main__":
    main()
