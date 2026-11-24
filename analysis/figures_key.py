"""
Thesis figures for the board fingerprint, the key and the boot protocol
(reports 6-10, 2026-10). Same conventions as figures.py / figures4.py:
drones in fixed categorical order (thesis colour rule, analysis/palette.py),
always also labelled in text; ink-coloured text; one y-axis per panel.

  fig8_autotest_giorni.png      self-test per drone and day (temperature-compensated)
  fig9_temperatura.png          temperature dependence (sensitivity d vs gyro bias)
  fig10_chiave_configurazioni.png  wrong symbols: genuine vs other drones, 3 configurations
  fig11_protocollo_avvio.png    boot protocol timeline + candidate protocols
  fig12_architettura_chiave.png enrollment / boot architecture

Run:  python -m analysis.figures_key
"""

import json
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis.palette import P1, P2
from matplotlib.patches import FancyBboxPatch

from analysis.dataset import ROOT
from analysis.figures import _style, INK, INK2, GRID, SURF
from analysis.figures4 import C, DRONES
from analysis import board as boardmod

OUT = ROOT / "analysis" / "figures"


def _board():
    D = boardmod.load()
    feats = ["dX", "dY", "dZ", "oX", "oY", "oZ", "gBiasY", "gBiasZ"]
    slopes = boardmod.temp_slopes(D, feats)
    return D, boardmod.compensate(D, feats, slopes), slopes


def fig_autotest_days():
    _, Cc, _ = _board()
    Cc = Cc[Cc.st_valid]
    days = sorted(Cc.day.unique())
    feats = [("dX", "sensitivity x"), ("dY", "sensitivity y"), ("dZ", "sensitivity z"),
             ("oX", "offset x"), ("oY", "offset y"), ("oZ", "offset z")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), facecolor=SURF)
    for ax, (f, title) in zip(axes.flat, feats):
        _style(ax)
        for i, d in enumerate(DRONES):
            for j, day in enumerate(days):
                v = Cc[(Cc.drone == d) & (Cc.day == day)][f]
                x = i + (j - 0.5) * 0.28 + np.linspace(-0.05, 0.05, len(v))
                ax.scatter(x, v, s=22, marker="o" if j == 0 else "^",
                           facecolor=C[d] if j == 0 else SURF, edgecolor=C[d], linewidth=1.2, zorder=3)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"drone {d}" for d in DRONES], color=INK2, fontsize=8)
        ax.set_title(f"{title} (LSB, at 30 °C)", color=INK, fontsize=9, loc="left")
    h = [plt.Line2D([], [], marker="o", ls="", markerfacecolor=INK2, markeredgecolor=INK2),
         plt.Line2D([], [], marker="^", ls="", markerfacecolor=SURF, markeredgecolor=INK2)]
    fig.legend(h, [f"day 1 ({days[0]})", f"day 2 ({days[-1]})"], loc="lower center", ncol=2,
               frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Accelerometer self-test: each point is a boot. It varies by a few LSB from one boot "
                 "to the next, by hundreds between drones", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "fig8_autotest_giorni.png", dpi=200)
    plt.close(fig)


def fig_temperature():
    D, _, slopes = _board()
    D = D[D.st_valid]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURF)
    for ax, (f, title) in zip(axes, [("dX", "self-test sensitivity x (LSB)"),
                                    ("gBiasY", "gyroscope bias y (LSB)")]):
        _style(ax)
        for d in DRONES:
            g = D[D.drone == d]
            dev = g[f] - g[f].mean()
            ax.scatter(g.temp, dev, s=20, color=C[d], zorder=3)
            ax.annotate(d, (g.temp.iloc[-1], dev.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                        color=INK, fontsize=8, va="center")
        xs = np.linspace(D.temp.min(), D.temp.max(), 10)
        ax.plot(xs, slopes[f] * (xs - D.temp.mean()), color=INK2, lw=1.2, ls="--")
        ax.set_xlabel("board temperature at boot (°C)", color=INK2, fontsize=8)
        ax.set_ylabel("deviation from the drone's mean", color=INK2, fontsize=8)
        ax.set_title(f"{title}: slope {slopes[f]:+.2f}/°C", color=INK, fontsize=9, loc="left")
    fig.suptitle("Effect of temperature (dashed = common slope, removed by bringing everything to 30 °C)",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig9_temperatura.png", dpi=200)
    plt.close(fig)


def fig_key_configurations():
    """Histogram of wrong symbols against the enrolled helper: genuine later
    boots (paired) vs every combination of another drone's measurements."""
    from analysis import combined_key as ck
    M, mf = ck.motor_table()
    B, bf = ck.board_table()
    wm, _ = ck.params(M[M.kind == "enroll"], mf)
    wb, _ = ck.params(B, bf)
    src = {"motors": ck.Source(M[M.kind != "swap"], mf, wm), "board": ck.Source(B, bf, wb)}
    configs = [("motors only (28 symbols)", ["motors"]), ("board only (12 symbols)", ["board"]),
               ("motors + board (40 symbols)", ["motors", "board"])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), facecolor=SURF)
    for ax, (title, names) in zip(axes, configs):
        _style(ax)
        gen, imp = [], []
        for T in DRONES:
            hs, rows_g, rows_i = {}, {}, {}
            for n in names:
                s = src[n]
                first = s.t[s.t.drone == T].day.min()
                e = s.t[(s.t.drone == T) & (s.t.day == first)]
                if "kind" in e:
                    e = e[e.kind == "enroll"]
                hs[n] = ck.helper(e[s.f].median(), s.w)
                rows_g[n] = list(s.t[(s.t.drone == T) & (s.t.day != first)].iterrows())
                rows_i[n] = list(s.t[s.t.drone != T].iterrows())
            L = max(len(v) for v in rows_g.values())
            for i in range(L):
                gen.append(sum(ck.n_wrong(hs[n], rows_g[n][i % len(rows_g[n])][1][src[n].f].astype(float), src[n].w)
                               for n in names))
            combos = itertools.product(*[rows_i[n] for n in names])
            for k, combo in enumerate(combos):
                if k > 3000:
                    break
                imp.append(sum(ck.n_wrong(hs[n], r[src[n].f].astype(float), src[n].w)
                               for n, (_, r) in zip(names, combo)))
        n_sym = sum(len(src[n].f) for n in names)
        bins = np.arange(-0.5, n_sym + 1.5, 1)
        ax.hist(imp, bins=bins, color=GRID, density=True, label="other drones")
        ax.hist(gen, bins=bins, color=P1, density=True, alpha=0.9, label="right drone (later boots)")
        ax.axvline(2.5, color=INK, lw=1, ls="--")
        ax.text(2.7, ax.get_ylim()[1] * 0.92, "t = 2", color=INK, fontsize=8)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("wrong symbols", color=INK2, fontsize=8)
    axes[0].set_ylabel("fraction of cases", color=INK2, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Key: symbols to correct. Accepted when left of the dashed line "
                 "(no refresh of the helper data in this figure)", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig10_chiave_configurazioni.png", dpi=200)
    plt.close(fig)


def fig_boot_protocol():
    from enrollment.micro_sweep import BOOT_REST_S, BOOT_RAMP_S, BOOT_HOLD_S, BOOT_PAUSE_S, BOOT_PWM
    r = json.load(open(ROOT / "analysis" / "results_boot" / "boot_protocol.json", encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), facecolor=SURF, gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    _style(ax)
    t = BOOT_REST_S
    ax.axvspan(0, BOOT_REST_S, color=GRID, zorder=0)
    ax.text(BOOT_REST_S / 2, 35, "motors off:\nself-test, bias, scale,\nnoise at rest",
            ha="center", va="center", color=INK, fontsize=8)
    for m in (1, 2, 3, 4):
        xs = [t, t + BOOT_RAMP_S, t + BOOT_RAMP_S + BOOT_HOLD_S, t + BOOT_RAMP_S + BOOT_HOLD_S]
        ys = [0, BOOT_PWM / 1000, BOOT_PWM / 1000, 0]
        ax.plot(xs, ys, color=INK, lw=1.8)
        ax.axvspan(t + BOOT_RAMP_S + 0.5, t + BOOT_RAMP_S + BOOT_HOLD_S, color=P1, alpha=0.15, zorder=0)
        ax.text(t + BOOT_RAMP_S + BOOT_HOLD_S / 2, BOOT_PWM / 1000 + 3, f"M{m}", ha="center", color=INK, fontsize=8)
        t += BOOT_RAMP_S + BOOT_HOLD_S + BOOT_PAUSE_S
    ax.set_xlim(0, t)
    ax.set_ylim(0, 70)
    ax.set_xlabel("time since power-up (s)", color=INK2, fontsize=8)
    ax.set_ylabel("PWM of the running motor (×1000)", color=INK2, fontsize=8)
    ax.set_title(f"Boot protocol (~{t:.0f} s): in blue the 3 s analysed per motor",
                 color=INK, fontsize=9, loc="left")
    ax = axes[1]
    _style(ax)
    for name, v in r["candidates"].items():
        b = v["best"]
        if b is None:
            continue
        x, y = v["motor_seconds"], b["motors_only_impostor"]
        chosen = name == "58k|D=3"
        ax.scatter(x, y, s=45 if chosen else 20, color=P1 if chosen else INK2, zorder=3)
        if name in ("58k|D=3", "45k|D=3", "25k|D=3", "15k+35k|D=3", "15k+35k+58k|D=3"):
            ax.annotate(name.replace("|D=", ", ") + " s", (x, y), xytext=(4, 3), textcoords="offset points",
                        color=INK, fontsize=7)
    ax.set_xlabel("seconds with motors running per boot", color=INK2, fontsize=8)
    ax.set_ylabel("other drones accepted, motor part only", color=INK2, fontsize=8)
    ax.set_title("Candidate protocols (motors + board: 0 impostors for all)", color=INK, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig11_protocollo_avvio.png", dpi=200)
    plt.close(fig)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(12, 4.6), facecolor=SURF)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, text, fill=SURF, edge=INK2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                    facecolor=fill, edgecolor=edge, linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=INK, fontsize=8)

    def arrow(x0, y0, x1, y1, label=""):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, label, ha="center", color=INK2, fontsize=7)

    ax.text(0.1, 4.35, "BOOT (every power-up, on board the drone)", color=INK, fontsize=9, fontweight="bold")
    box(0.1, 2.9, 2.0, 1.1, "measurements at boot\nboard: self-test, bias,\nscale, noise\nmotors: 58k, 3 s each", GRID)
    box(2.6, 2.9, 2.2, 1.1, "quantisation\nwith offsets\n(helper data)")
    box(5.3, 2.9, 2.2, 1.1, "error correction\nReed–Solomon (t = 2)\n+ robustness tag")
    box(8.0, 2.9, 1.6, 1.1, "secret W\n→ password\n(HKDF)")
    box(10.1, 2.9, 1.8, 1.1, "SPAKE2 with the\nstation → session\nkey", "#dbe8f8", P1)
    arrow(2.1, 3.45, 2.6, 3.45)
    arrow(4.8, 3.45, 5.3, 3.45)
    arrow(7.5, 3.45, 8.0, 3.45)
    arrow(9.6, 3.45, 10.1, 3.45)
    box(5.3, 1.55, 2.2, 0.85, "helper data in EEPROM\n(~0.8 KB, not secret)")
    arrow(6.4, 2.4, 6.4, 2.9, "")
    arrow(6.0, 2.9, 6.0, 2.4)
    ax.text(7.6, 2.0, "read at boot;\nrefreshed after a\nsuccessful boot", color=INK2, fontsize=7, va="center")
    box(10.1, 1.55, 1.8, 0.85, "encrypted channel\nAES-GCM", "#dbe8f8", P1)
    arrow(11.0, 2.9, 11.0, 2.4)
    ax.text(0.1, 0.95, "ENROLLMENT (once, with the station)", color=INK, fontsize=9, fontweight="bold")
    box(0.1, 0.05, 3.2, 0.7, "reference measurements of the drone (median)", GRID)
    box(3.8, 0.05, 3.3, 0.7, "Gen: secret W, helper data → drone")
    box(7.6, 0.05, 4.3, 0.7, "password derived from W → station database")
    arrow(3.3, 0.4, 3.8, 0.4)
    arrow(7.1, 0.4, 7.6, 0.4)
    ax.text(0.1, 2.35, "with the drone off, memory\nholds only the helper data:\nthe key does not exist", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig12_architettura_chiave.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_autotest_days()
    fig_temperature()
    fig_key_configurations()
    fig_boot_protocol()
    fig_architecture()
    if (ROOT / "analysis" / "results_boot_eval" / "boot_eval.json").exists():
        fig_real_boots()
        fig_stability()



def fig_real_boots():
    """Real boot sessions (10-13): wrong symbols, genuine vs impostor, all
    enrollment choices of 3 sessions; three configurations."""
    r = json.load(open(ROOT / "analysis" / "results_boot_eval" / "boot_eval.json", encoding="utf-8"))["key"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), facecolor=SURF)
    for ax, (k, title) in zip(axes, [("motors_only", "motors only (28 symbols)"),
                                     ("board_only", "board only (12 symbols)"),
                                     ("combined", "motors + board (40 symbols)")]):
        _style(ax)
        g, i = np.array(r[k]["genuine_hist"], float), np.array(r[k]["impostor_hist"], float)
        x = np.arange(len(g))
        ax.bar(x, i / i.sum(), width=0.9, color=P2, alpha=0.55, label=f"other drones ({int(i.sum())} trials)")
        ax.bar(x, g / g.sum(), width=0.6, color=P1, label=f"right drone ({int(g.sum())} trials)")
        ax.axvline(2.5, color=INK, lw=1, ls="--")
        ax.text(2.7, ax.get_ylim()[1] * 0.92, "t = 2", color=INK, fontsize=8)
        ax.set_xlim(-0.8, min(len(g), 30))
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("wrong symbols", color=INK2, fontsize=8)
    axes[0].set_ylabel("fraction of trials", color=INK2, fontsize=8)
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK)
    # no title: the thesis caption describes it (one day of boots; the genuine bars at 7-8 are B-s06)
    fig.tight_layout()
    fig.savefig(OUT / "fig13_avvii_reali.png", dpi=200)
    plt.close(fig)


def fig_stability():
    """Motor speed per motor over time (one panel per drone) vs the board
    part enrolled a week earlier."""
    r = json.load(open(ROOT / "analysis" / "results_boot_eval" / "boot_eval.json", encoding="utf-8"))
    H = pd.DataFrame(r["motor_history"]).T
    fig = plt.figure(figsize=(13, 6.2), facecolor=SURF)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9])
    shades = {"1": 1.0, "2": 0.75, "3": 0.5, "4": 0.3}
    for k, d in enumerate(DRONES):
        ax = fig.add_subplot(gs[0, k])
        _style(ax)
        rows = H[H.index.str.startswith(d + " ")]
        x = np.arange(len(rows))
        for m, ls in zip(("1", "2", "3", "4"), ("-", "--", ":", "-.")):
            y = rows[m].values.astype(float)
            ax.plot(x, y, color=C[d], ls=ls, lw=1.6, marker="o", ms=3, alpha=shades[m])
            ax.annotate(f"M{m}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", color=INK, fontsize=7,
                        va="center")
        ax.set_xticks(x)
        ax.set_xticklabels([i.split(" ", 1)[1].replace("2026-", "").replace(" (boot)", "\nboot").replace(" (avvio)", "\nboot")
                            for i in rows.index],
                           fontsize=7, color=INK2)
        ax.set_ylim(0.85, 1.13)
        ax.set_xlim(-0.3, len(rows) - 0.3)
        ax.set_title(f"drone {d}: motor speed (58k)", color=INK, fontsize=9, loc="left")
        if k == 0:
            ax.set_ylabel("measured / predicted", color=INK2, fontsize=8)
    ax = fig.add_subplot(gs[1, :])
    _style(ax)
    wk = r["board_one_week"]
    for k, d in enumerate(DRONES):
        g = wk[d]["genuine_wrong"]
        ax.scatter(np.full(len(g), k) + np.linspace(-0.12, 0.12, len(g)), g, color=C[d], s=28, zorder=3)
        ax.scatter([k], [wk[d]["impostor_min_wrong"]], marker="v", color=INK2, s=34, zorder=3)
    ax.axhline(2.5, color=INK, lw=1, ls="--")
    ax.text(3.2, 2.7, "t = 2", color=INK, fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"drone {d}" for d in DRONES], fontsize=8, color=INK2)
    ax.set_ylabel("wrong symbols out of 12", color=INK2, fontsize=8)
    ax.set_title("Board part: enrollment with the sessions of 10-07/08, check with the boots of 10-13 "
                 "(dots: boots of the drone; triangle: closest impostor)", color=INK, fontsize=9, loc="left")
    fig.suptitle("Stability over time: the motors change after every remount (A after the swap of 09-30), "
                 "the board stays the same", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig14_stabilita_nel_tempo.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    main()
