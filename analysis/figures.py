"""
Thesis figures for the drone A vs B identification study (static PNG, print).

Colour: two categorical identities only — drone A = slot 1 (blue #2a78d6),
drone B = slot 2 (now the thesis colour rule of analysis/palette.py: A #9B0014, B #095579) (these
two pass the all-pairs CVD/normal-vision checks); every figure also labels
the drones directly, so identity is never colour-alone. Text stays in ink
tones, never series colours. One y-axis per panel.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.dataset import ROOT

OUT = ROOT / "analysis" / "figures"
RES = ROOT / "analysis" / "results"
from analysis.palette import DRONE
C = {"A": DRONE["A"], "B": DRONE["B"]}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_motor_signature():
    s = pd.read_csv(RES / "motor_signatures.csv")
    st = s[s.purpose.str.startswith("stepped")].copy()
    st["step"] = (st.pwm.round(-3) / 1000).astype(int)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1), sharey=True, facecolor=SURF)
    for m, ax in zip((1, 2, 3, 4), axes):
        _style(ax)
        for d in ("A", "B"):
            g = st[(st.drone == d) & (st.motor == m)]
            for (_, _), gs in g.groupby(["kind", "session"]):
                gs = gs.sort_values("step")
                ax.plot(gs.step, gs.gain, color=C[d], lw=0.8, alpha=0.35)
            mean = g.groupby("step").gain.mean()
            ax.plot(mean.index, mean.values, color=C[d], lw=2, marker="o", ms=4)
            ax.annotate(f"drone {d}", (mean.index[-1], mean.values[-1]), xytext=(4, 0),
                        textcoords="offset points", color=INK, fontsize=8, va="center")
        ax.set_title(f"M{m}", color=INK, fontsize=10)
        ax.set_xlabel("PWM (×1000)", color=INK2, fontsize=8)
        ax.set_xlim(5, 72)
    axes[0].set_ylabel("measured / predicted speed", color=INK2, fontsize=8)
    fig.suptitle("Motor signature: gain per motor and PWM step (thin lines = single sessions)",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_firma_motore.png", dpi=200)
    plt.close(fig)


def fig_separation_profile():
    r = json.load(open(RES / "E6.json"))["psd_single58k_F_by_bin"]
    df = pd.DataFrame({"ch": [k.split(":")[1] for k in r], "hz": [int(k.split(":")[2]) + 5 for k in r],
                       "F": list(r.values())})
    prof = df.groupby("hz").F.max()
    fig, ax = plt.subplots(figsize=(8, 3), facecolor=SURF)
    _style(ax)
    ax.axvspan(385, 475, color=GRID, zorder=0)
    ax.text(430, prof.max() * 0.95, "motor line\n(58k PWM)", ha="center", va="top", color=INK2, fontsize=8)
    ax.axvline(375, color=INK2, lw=0.8, ls="--")
    ax.text(370, prof.max() * 0.95, "edge of the\n\"chassis-only\" band", ha="right", va="top", color=INK2, fontsize=8)
    ax.plot(prof.index, prof.values, color=INK, lw=2)
    ax.set_xlabel("frequenza (Hz)", color=INK2, fontsize=8)
    ax.set_ylabel("F di ANOVA, A vs B\n(max sui 6 canali)", color=INK2, fontsize=8)
    ax.set_title("Dove si separano i droni — motore singolo a 58k PWM", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_profilo_separazione.png", dpi=200)
    plt.close(fig)


def fig_embedding(ds, representations):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    reps = representations(ds)
    from analysis.experiments import representations as R
    ext = R(ds, extended=True)
    panels = [("psd_chassis_single58k", reps["psd_chassis_single58k"], "chassis-only band, with level"),
              ("shape_chassis_single58k", ext["shape_chassis_single58k"], "chassis-only band, shape only"),
              ("comb120_allholds", reps["comb120_allholds"], "120 features, all holds")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), facecolor=SURF)
    for ax, (name, (sub, feats), title) in zip(axes, panels):
        _style(ax)
        g = sub.groupby("rec_id")
        M = g[feats].mean()
        meta = g.agg(drone=("drone", "first"), kind=("kind", "first"))
        en = meta.kind == "enroll"
        Z = PCA(2, random_state=0).fit(StandardScaler().fit(M[en]).transform(M[en]))
        sc = StandardScaler().fit(M[en])
        P = Z.transform(sc.transform(M))
        for d in ("A", "B"):
            for kind, mk, fc in (("enroll", "o", C[d]), ("test", "^", SURF)):
                sel = (meta.drone == d) & (meta.kind == kind)
                ax.scatter(P[sel, 0], P[sel, 1], s=22, marker=mk, facecolor=fc,
                           edgecolor=C[d] if kind == "test" else SURF, linewidth=1.2 if kind == "test" else 0.8,
                           label=f"drone {d} — {'enrollment' if kind == 'enroll' else 'test'}")
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("PC1", color=INK2, fontsize=8); ax.set_ylabel("PC2", color=INK2, fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("One recording = one point (mean of its windows). PCA fitted on the enrollment only",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "fig3_embedding_pca.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_motor_signature()
    fig_separation_profile()
    from analysis.dataset import load_dataset
    from analysis.experiments import representations
    from analysis.experiments import AB_CUTOFF
    ds = load_dataset(verbose=False)
    fig_embedding(ds[ds.drone.isin(["A", "B"]) & (ds.day <= AB_CUTOFF)].reset_index(drop=True), representations)


if __name__ == "__main__":
    main()
