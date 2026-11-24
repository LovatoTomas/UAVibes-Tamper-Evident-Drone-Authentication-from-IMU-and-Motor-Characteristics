"""
Thesis figures for the four-drone study (static PNG, print).

Colour: four categorical identities in the reference palette's fixed order
(thesis colour rule, analysis/palette.py: A #9B0014, B #095579, C #DDA73B, D #6D9773); every figure also
labels the drones directly, so identity is never colour-alone. Text stays in
ink tones. One y-axis per panel.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.dataset import ROOT
from analysis.figures import _style, INK, INK2, GRID, SURF

OUT = ROOT / "analysis" / "figures"
RES = ROOT / "analysis" / "results_4drones"
from analysis.palette import DRONE as C
DRONES = ["A", "B", "C", "D"]


def _session_start() -> "dict[str, str]":
    """'C-e4' -> first accepted attempt id (YYYYMMDD_HHMMSS_...) of that session."""
    import glob
    out = {}
    for f in glob.glob(str(ROOT / "data" / "fixed_rig*" / "drone*" / "session_*.json")):
        m = json.load(open(f, encoding="utf-8"))
        ids = [a["attempt_id"] for r in m["recordings"] for a in r["attempts"] if a.get("converted_path")]
        out[f"{m['drone']}-{m.get('kind', 'enroll')[0]}{m['session']}"] = min(ids)
    return out


def fig_motor_chronology():
    """Median per-motor gain (stepped ladders, 15k-58k) of every session, in
    the order the sessions were recorded: shows cross-day stability for A/B
    and the progressive slow-down of C-M3 / D-M1 / D-M3."""
    s = pd.read_csv(RES / "motor_signatures.csv")
    st = s[s.purpose.str.startswith("stepped") & (s.pwm > 12000)].copy()
    st["sid"] = st.drone + "-" + st.kind.str[0] + st.session.astype(str)
    g = st.groupby(["sid", "drone", "day", "motor"]).gain.median().reset_index()
    order = sorted(st.sid.unique(), key=_session_start().get)
    xpos = {sid: i for i, sid in enumerate(order)}
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3), sharey=True, facecolor=SURF)
    for m, ax in zip((1, 2, 3, 4), axes):
        _style(ax)
        for d in DRONES:
            gd = g[(g.drone == d) & (g.motor == m)].copy()
            gd["x"] = gd.sid.map(xpos)
            gd = gd.sort_values("x")
            ax.plot(gd.x, gd.gain, color=C[d], lw=2, marker="o", ms=5,
                    markeredgecolor=SURF, markeredgewidth=1)
            ax.annotate(d, (gd.x.iloc[-1], gd.gain.iloc[-1]), xytext=(5, 0),
                        textcoords="offset points", color=INK, fontsize=8, va="center")
        days = pd.Series({sid: st[st.sid == sid].day.iloc[0] for sid in order})
        for i in range(1, len(order)):
            if days.iloc[i] != days.iloc[i - 1]:
                ax.axvline(i - 0.5, color=INK2, lw=0.6, ls=":")
        ax.set_title(f"M{m}", color=INK, fontsize=10)
        ax.set_xticks([])
        ax.set_xlabel("sessions in recording order (: = new day)", color=INK2, fontsize=7)
    axes[0].set_ylabel("measured / predicted speed\n(median of the 15k-58k steps)", color=INK2, fontsize=8)
    fig.suptitle("Motor signature over time: A and B stable; C (M2, M3) and D (M1, M3) slow down session after session",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_firma_motore_cronologia.png", dpi=200)
    plt.close(fig)


def fig_embedding4(ds):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from analysis.experiments import representations
    base, ext = representations(ds), representations(ds, extended=True)
    panels = [(base["psd_chassis_single58k"], "chassis-only band, with level"),
              (ext["shape_chassis_single58k"], "chassis-only band, shape only"),
              (base["comb120_allholds"], "120 features, all holds")]
    fd = {d: ds[ds.drone == d].day.min() for d in DRONES}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), facecolor=SURF)
    for ax, ((sub, feats), title) in zip(axes, panels):
        _style(ax)
        g = sub.groupby("rec_id")
        M = g[feats].mean()
        meta = g.agg(drone=("drone", "first"), day=("day", "first"))
        first = meta.day.values == meta.drone.map(fd).values
        sc = StandardScaler().fit(M[first])
        pca = PCA(2, random_state=0).fit(sc.transform(M[first]))
        P = pca.transform(sc.transform(M))
        for d in DRONES:
            for later, mk in ((False, "o"), (True, "^")):
                sel = ((meta.drone == d).values & (first != later))
                ax.scatter(P[sel, 0], P[sel, 1], s=20, marker=mk,
                           facecolor=SURF if later else C[d], edgecolor=C[d] if later else SURF,
                           linewidth=1.1 if later else 0.7,
                           label=f"{d} — {'later day' if later else 'first day'}")
            c = P[(meta.drone == d).values].mean(0)
            ax.annotate(d, c, color=INK, fontsize=10, fontweight="bold", ha="center", va="center")
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("PC1", color=INK2, fontsize=8)
        ax.set_ylabel("PC2", color=INK2, fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, frameon=False, fontsize=7, labelcolor=INK)
    fig.suptitle("One recording = one point. PCA fitted on the first day of each drone only; "
                 "empty triangles = later days", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "fig5_embedding_4droni.png", dpi=200)
    plt.close(fig)


def fig_crossday_spectra(ds):
    """Mean level-free log-PSD (acc.x, chassis band) of the single-motor 58k
    holds per drone and day — the day-2 curve should sit on its own day-1
    curve, not on another drone's."""
    from analysis.experiments import _shape, SINGLE_58K, CHASSIS_MAX_HZ
    from analysis.dataset import cols
    h = ds[ds.phase.isin(SINGLE_58K)]
    band = [c for c in cols(ds, "psd") if int(c.rsplit(":", 1)[1]) <= CHASSIS_MAX_HZ - 10]
    hs, fs = _shape(h, band)
    fd = {d: ds[ds.drone == d].day.min() for d in DRONES}
    ch = [c for c in fs if c.split(":")[1] == "acc.x"]
    hz = [int(c.rsplit(":", 1)[1]) + 5 for c in ch]
    fig, axes = plt.subplots(4, 4, figsize=(12, 9), sharex=True, sharey=True, facecolor=SURF)
    for i, d in enumerate(DRONES):
        for j, ph in enumerate(sorted(SINGLE_58K)):
            ax = axes[i, j]
            _style(ax)
            for o in DRONES:
                if o != d:
                    so = hs[(hs.drone == o) & (hs.phase == ph) & (hs.day == fd[o])]
                    ax.plot(hz, so[ch].mean().values, color=GRID, lw=1.4, zorder=1)
            sd = hs[(hs.drone == d) & (hs.phase == ph)]
            for later, ls, lab in ((False, "-", "first day"), (True, "--", "later day")):
                sel = sd[(sd.day != fd[d]) if later else (sd.day == fd[d])]
                if len(sel):
                    ax.plot(hz, sel[ch].mean().values, color=C[d], lw=1.8, ls=ls, label=lab, zorder=3)
            if i == 0:
                ax.set_title(ph.replace("_hold", "").upper() + " a 58k PWM", color=INK, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"drone {d}\nlog-PSD acc.x (forma)", color=INK, fontsize=8)
            if i == 3:
                ax.set_xlabel("frequenza (Hz)", color=INK2, fontsize=8)
    h, l = axes[0, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=GRID, lw=1.4)); l.append("other drones (first day)")
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Spectral shape in the chassis-only band, single motor at 58k: each drone against itself "
                 "on another day", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig6_spettri_giorni.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_motor_chronology()
    from analysis.dataset import load_dataset
    ds = load_dataset(verbose=False)
    fig_embedding4(ds)
    fig_crossday_spectra(ds)
    fig_swap_map()          # defined further down (motor-swap study)



def fig_swap_map():
    """Motor-swap experiment: chassis-band 58k single-motor spectrum (the 4
    stages concatenated) of every swap-day session, PCA fitted on those
    sessions; one point per configuration = the mean of its sessions (2-3 each,
    B after only 1). Colour = whose MOTORS are mounted, marker = FRAME."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from analysis.swap import load_swap, _chassis_reps
    sw = load_swap()
    h, f = _chassis_reps(sw)["psd_chassis_single58k"]
    blocks = []
    for ph in sorted(h.phase.unique()):
        X = h[h.phase == ph].groupby(["session", "stage", "drone", "motors"])[f].mean()
        blocks.append(X)
    X = pd.concat(blocks, axis=1).dropna()
    meta = X.index.to_frame(index=False)
    sc = StandardScaler().fit(X.values)
    P = PCA(2, random_state=0).fit_transform(sc.transform(X.values))
    meta["pc1"], meta["pc2"] = P[:, 0], P[:, 1]
    conf = meta.groupby(["stage", "drone", "motors"], as_index=False)[["pc1", "pc2"]].mean()
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=SURF)
    _style(ax)
    for r in conf.itertuples():
        mk = "o" if r.drone == "A" else "s"
        filled = r.stage == "pre"
        ax.scatter(r.pc1, r.pc2, s=90, marker=mk, facecolor=C[r.motors] if filled else SURF,
                   edgecolor=C[r.motors], linewidth=2, zorder=3)
        lab = {"pre": f"{r.drone} before", "swap": f"frame {r.drone} + motors {r.motors}",
               "post": f"{r.drone} after"}[r.stage]
        ax.annotate(lab, (r.pc1, r.pc2), xytext=(8, 5), textcoords="offset points", color=INK, fontsize=8)
    ax.margins(x=0.25, y=0.15)
    ax.set_xlabel("PC1", color=INK2, fontsize=8)
    ax.set_ylabel("PC2", color=INK2, fontsize=8)
    ax.set_title("Motor swap A\u2194B: colour = mounted motors (red A, blue B), "
                 "circle = frame A, square = frame B,\nfilled = before the swap", color=INK,
                 fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig7_scambio_motori.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
