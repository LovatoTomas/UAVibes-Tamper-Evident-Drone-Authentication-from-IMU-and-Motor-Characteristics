"""
Report 13 — five-day series (2026-10-31 → 11-04): replay of the boots recorded
on the SD card, with the drone's own code (keyproto.onboard, byte-exact with
the firmware), and the false rejections of the authentications run on board.

Data
  SD boots   data/fixed_rig_boot/sNN_droneX, created 31 Oct - 4 Nov, uploaded.
             Left out, with the reason in EXCLUDED:
               - B before its re-enrollment of 31 Oct 15:04 (s40, a diagnostic
                 boot after M4 changed);
               - D on 31 Oct (M2 damaged, then replaced; same exclusion as the
                 on-board log, analysis/onboard_log_labels.json);
               - attempts without a converted SD file (B s46).
  On-board   data/keystore/onboard/log.jsonl through analysis.onboard_log
             (conditions, exclusions, speed shifts of every authentication).

What it computes (python -m analysis.longitudinal, results in
analysis/results_longitudinal/)
  1. genuine replay: each drone enrolled on the median of its first 3 SD boots
     of the series (keyproto.onboard.gen), then every later boot in time order
     through Rep, with and without the on-board refresh, speed shifts;
  2. impostors / cloning: every boot of drone Y against the helper of drone X
     (= Y carrying a copy of X's memory), all ordered pairs; also the older
     recorded boots of the other drones (13-14 Oct, 18 Oct; C's second day
     stays out, as in the thesis) against the five-day helpers;
  3. transplant: quantities of X's motors with Y's board (and the reverse)
     against X's helper, all pairs of boots;
  4. exact speed symbols: on the on-board authentications of the five days,
     false rejections as a function of the speed window T, and the altered
     boots of 18-19 Oct that would be caught;
  figures fig15_soglia_velocita.png, fig16_simboli_errati.png.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from keyproto import boot as kb
from keyproto import onboard as ob

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results_longitudinal"
FIG = Path(__file__).resolve().parent / "figures"
FIRST, LAST = "2026-10-31", "2026-11-04"
N_ENROLL = 3
C_DAY2 = {27, 28, 29, 30}            # C's second day (14 Oct): out of the thesis
EXCLUDED = {
    ("B", 40): "diagnostic boot before B's re-enrollment of 31 Oct (M4 changed)",
}
EXCLUDED_DAYS = {("D", "2026-10-31"): "M2 damaged, then replaced"}
MOTOR = slice(0, 28)
BOARD = slice(28, 40)


def sessions() -> pd.DataFrame:
    rows = []
    for d in sorted(kb.BOOT_DIR.glob("s*_drone*")):
        try:
            m = json.loads((d / "session.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        conv = [a for r in m["recordings"] for a in r["attempts"] if a.get("converted_path")]
        rows.append(dict(dir=d, drone=m["drone"], session=m["session"], at=m["created_iso"],
                         day=m["created_iso"][:10], converted=bool(conv)))
    return pd.DataFrame(rows).sort_values("at").reset_index(drop=True)


def features(df: pd.DataFrame, sysp: dict) -> pd.DataFrame:
    xs = []
    for r in df.itertuples():
        try:
            x = ob.onboard_features(ob.boot_inputs(r.dir), sysp)
        except Exception as exc:          # a hold lost to the SD stall, an odd log
            print(f"  skipped {r.dir.name}: {exc}")
            x = None
        xs.append(x)
    df = df.assign(x=xs)
    return df[df.x.notna()].reset_index(drop=True)


def wrong_count(x, helper: bytes, ref_W, sysp: dict) -> int:
    """Symbols that differ from the enrolled reading (the distance Rep has to
    correct), also beyond t, where Rep itself only says 'too many'."""
    h = ob.unpack_helper(helper)
    q = ob.quantise(x, h["offsets"], sysp["widths"])
    return int(sum((qi is None) or (qi % ob.P != wi % ob.P) for qi, wi in zip(q, ref_W)))


def enroll(boots: pd.DataFrame, drone: str, sysp: dict):
    ref = np.median(np.vstack(boots.x.values[:N_ENROLL]), axis=0)
    pw, helper = ob.gen(ref, sysp, drone)
    ref_W = ob.quantise(ref, ob.unpack_helper(helper)["offsets"], sysp["widths"])
    return pw, helper, ref_W


def attempt(x, helper, pw, ref_W, sysp) -> dict:
    out = dict(wrong=wrong_count(x, helper, ref_W, sysp))
    key, W, _ = ob.rep(x, helper, sysp)
    out["key"] = key is not None and key == pw
    if out["key"]:
        sh = ob.speed_shift(helper, W, x, sysp)
        out["max_shift"] = float(np.nanmax(np.abs(sh)))
        out["alarm"] = bool(ob.alarm_mask(sh))
        out["W"] = W
    else:
        out["max_shift"], out["alarm"], out["W"] = np.nan, False, None
    out["granted"] = out["key"] and not out["alarm"]
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    sysp = kb.system_params()
    allS = sessions()
    series = allS[(allS.day >= FIRST) & (allS.day <= LAST) & allS.converted].copy()
    series["excluded"] = [EXCLUDED.get((r.drone, r.session)) or EXCLUDED_DAYS.get((r.drone, r.day))
                          for r in series.itertuples()]
    print(f"SD boots in the series: {len(series)}, excluded {series.excluded.notna().sum()}")
    series = features(series[series.excluded.isna()], sysp)
    older = allS[(allS.day < FIRST) & allS.converted
                 & ~((allS.drone == "C") & allS.session.isin(C_DAY2))]
    older = features(older, sysp)
    drones = sorted(series.drone.unique())

    # 1. genuine replay
    enr, gen_rows = {}, []
    for d in drones:
        b = series[series.drone == d].reset_index(drop=True)
        pw, helper, ref_W = enroll(b, d, sysp)
        enr[d] = (pw, helper, ref_W)
        for refresh in (False, True):
            h = helper
            for r in b.iloc[N_ENROLL:].itertuples():
                a = attempt(r.x, h, pw, ref_W, sysp)
                gen_rows.append(dict(drone=d, refresh=refresh, session=r.session, day=r.day,
                                     **{k: v for k, v in a.items() if k != "W"}))
                if refresh and a["granted"]:
                    h = ob.refresh(h, a["W"], r.x, sysp)
    G = pd.DataFrame(gen_rows)

    # 2. impostors / cloning
    imp_rows = []
    for src, tag in ((series, "five-day"), (older, "older")):
        for X in drones:
            pw, helper, ref_W = enr[X]
            for r in src[src.drone != X].itertuples():
                a = attempt(r.x, helper, pw, ref_W, sysp)
                imp_rows.append(dict(claimed=X, actual=r.drone, set=tag, session=r.session,
                                     **{k: v for k, v in a.items() if k != "W"}))
    I = pd.DataFrame(imp_rows)

    # 3. transplant: motors of one drone, board of another, claimed as the first
    #    ("X's motors on Y's board") or as the second ("Y's motors on X's board")
    tr_rows = []
    for X in drones:
        pw, helper, ref_W = enr[X]
        bx = series[series.drone == X]
        for Y in drones:
            if Y == X:
                continue
            by = series[series.drone == Y]
            for rx in bx.itertuples():
                for ry in by.itertuples():
                    a1 = attempt(np.r_[rx.x[MOTOR], ry.x[BOARD]], helper, pw, ref_W, sysp)
                    a2 = attempt(np.r_[ry.x[MOTOR], rx.x[BOARD]], helper, pw, ref_W, sysp)
                    tr_rows.append(dict(claimed=X, other=Y, case="X motors on Y board",
                                        **{k: v for k, v in a1.items() if k != "W"}))
                    tr_rows.append(dict(claimed=X, other=Y, case="Y motors on X board",
                                        **{k: v for k, v in a2.items() if k != "W"}))
    T = pd.DataFrame(tr_rows)

    # 4. exact speed symbols, on the on-board authentications
    from analysis.onboard_log import load
    L = load()
    nom = L[(L.condition == "nominal") & (L.outcome != "error") & (L.day >= FIRST) & (L.day <= LAST)].copy()
    nom["max_shift"] = nom[[f"shift_m{m}" for m in (1, 2, 3, 4)]].abs().max(axis=1)
    alt = L[L.condition.isin(["propeller", "tape"]) & (L.outcome != "error")].copy()
    alt["max_shift"] = alt[[f"shift_m{m}" for m in (1, 2, 3, 4)]].abs().max(axis=1)
    base_rej = int((nom.outcome != "accepted").sum())
    acc = nom[nom.outcome == "accepted"]
    grid = np.round(np.arange(2.0, 8.01, 0.25), 2)
    thr = []
    for t in grid:
        extra = int((acc.max_shift > t).sum())
        caught = int(((alt.outcome != "accepted") | (alt.max_shift > t)).sum())
        thr.append(dict(T=float(t), extra=extra, rejected=base_rej + extra, n=len(nom),
                        frr=(base_rej + extra) / len(nom), altered_caught=caught, altered_n=len(alt)))
    TH = pd.DataFrame(thr)

    ES = exact_speed(series, older, drones, sysp)

    summary = dict(
        generated=datetime.now().isoformat(timespec="seconds"),
        exact_speed=ES,
        sd_boots=dict(used=int(len(series)), per_drone=series.drone.value_counts().sort_index().to_dict(),
                      older_used=int(len(older))),
        genuine={("refresh" if rf else "no_refresh"): dict(
            n=int(len(g)), key=int(g.key.sum()), granted=int(g.granted.sum()),
            wrong_max=int(g.wrong.max()), shift_max=float(np.nanmax(g.max_shift)) if g.key.any() else None,
            per_drone={d: f"{int(gg.granted.sum())}/{len(gg)}" for d, gg in g.groupby("drone")})
            for rf, g in G.groupby("refresh")},
        impostors={tag: dict(n=int(len(g)), key=int(g.key.sum()), granted=int(g.granted.sum()),
                             wrong_min=int(g.wrong.min()),
                             wrong_quantiles=[int(v) for v in np.percentile(g.wrong, [5, 50, 95])])
                   for tag, g in I.groupby("set")},
        transplant={case: dict(n=int(len(g)), key=int(g.key.sum()), granted=int(g.granted.sum()),
                               wrong_min=int(g.wrong.min()),
                               pairs_with_acceptance=sorted({f"{r.claimed}<-{r.other}" for r in g[g.granted].itertuples()}))
                    for case, g in T.groupby("case")},
        speed_window=dict(n=int(len(nom)), base_rejections=base_rej,
                          max_shift_accepted=float(acc.max_shift.max()),
                          table=TH[TH["T"].isin([3.0, 4.0, 5.0, 6.0])].to_dict("records"),
                          altered_n=int(len(alt))),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    G.to_csv(OUT / "genuine.csv", index=False)
    I.to_csv(OUT / "impostors.csv", index=False)
    T.drop(columns=[], errors="ignore").to_csv(OUT / "transplant.csv", index=False)
    TH.to_csv(OUT / "speed_window.csv", index=False)
    figures(G, I, TH)
    print(json.dumps(summary, indent=1, default=str))


def exact_speed(series: pd.DataFrame, older: pd.DataFrame, drones, sysp: dict) -> dict:
    """The firmware variant under discussion: the four speed symbols leave the
    Reed-Solomon correction and must match exactly, with a window of +-T sigma
    around the reference speed; the other 36 symbols keep t = 2. Compared with
    the current firmware (40 symbols, t = 2, speed alarm at 5 sigma).

    Symbols are evaluated against the reference values directly: a symbol is
    wrong when the value leaves the bin of width w centred on the reference
    (what the centring offsets do), a speed symbol when it moves by more than
    T sigma. Refresh = reference moved to the median of the last 3 accepted
    boots, as on board."""
    w = np.asarray(sysp["widths"], float)
    K = float(sysp["K"])
    gi = ob.gain_index(sysp["features"])
    speed = np.array(sorted(gi.values()))
    other = np.array([i for i in range(len(w)) if i not in set(speed)])
    t = int(sysp["t"])

    def judge(x, ref, T):
        d = np.asarray(x, float) - ref
        wrong_all = int(np.sum(~np.isfinite(d) | (np.abs(d) > w / 2)))
        wrong_other = int(np.sum(~np.isfinite(d[other]) | (np.abs(d[other]) > w[other] / 2)))
        sh = K * d[speed] / w[speed]
        maxsh = float(np.nanmax(np.abs(sh))) if np.isfinite(sh).any() else np.inf
        current = wrong_all <= t and maxsh <= 5.0         # 40 symbols t=2 + alarm at 5 sigma
        exact = wrong_other <= t and maxsh <= T           # 36 symbols t=2 + 4 exact speed symbols
        return current, exact

    out = {}
    refs = {d: np.median(np.vstack(series[series.drone == d].x.values[:N_ENROLL]), axis=0) for d in drones}
    for T in (3.0, 4.0, 5.0, 6.0):
        res = dict(genuine=[0, 0, 0], impostor=[0, 0, 0], transplant_other_motors=[0, 0, 0],
                   transplant_other_board=[0, 0, 0])            # [n, accepted now, accepted exact]
        for d in drones:                                        # genuine, with refresh
            b = series[series.drone == d].reset_index(drop=True)
            hist = []
            for r in b.iloc[N_ENROLL:].itertuples():
                ref = np.median(np.vstack(hist[-3:]), axis=0) if hist else refs[d]
                cur, ex = judge(r.x, ref, T)
                res["genuine"][0] += 1
                res["genuine"][1] += cur
                res["genuine"][2] += ex
                if ex:
                    hist.append(np.asarray(r.x, float))
        for X in drones:
            for r in pd.concat([series, older])[lambda s: s.drone != X].itertuples():
                cur, ex = judge(r.x, refs[X], T)
                res["impostor"][0] += 1
                res["impostor"][1] += cur
                res["impostor"][2] += ex
            for Y in drones:
                if Y == X:
                    continue
                for rx in series[series.drone == X].itertuples():
                    for ry in series[series.drone == Y].itertuples():
                        for key, x in (("transplant_other_motors", np.r_[ry.x[MOTOR], rx.x[BOARD]]),
                                       ("transplant_other_board", np.r_[rx.x[MOTOR], ry.x[BOARD]])):
                            cur, ex = judge(x, refs[X], T)
                            res[key][0] += 1
                            res[key][1] += cur
                            res[key][2] += ex
        out[f"T={T:g}"] = {k: dict(n=v[0], accepted_current=int(v[1]), accepted_exact=int(v[2])) for k, v in res.items()}
    return out


def figures(G: pd.DataFrame, I: pd.DataFrame, TH: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.palette import P1 as BLUE, P2 as ORANGE   # thesis colour rule (analysis/palette.py)
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})

    # fig15: false rejections vs speed window (one series, no legend)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.plot(TH["T"], 100 * TH["frr"], color=BLUE, lw=2)
    ax.axvline(5, color=MUTED, lw=1, ls="--")
    r5 = TH[TH["T"] == 5.0].iloc[0]
    ax.annotate(f"5σ: {int(r5.rejected)}/{int(r5.n)} rejected", (5, 100 * r5.frr),
                xytext=(5.3, 100 * r5.frr + 2), color=INK, fontsize=8)
    ax.set_xlabel("speed window T (within-drone σ)")
    ax.set_ylabel("false rejections (%)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig15_soglia_velocita.png", dpi=200)
    plt.close(fig)

    # fig16: wrong symbols, genuine (with refresh) vs impostors (five-day set)
    g = G[G.refresh].wrong.values
    i = I[I.set == "five-day"].wrong.values
    bins = np.arange(-0.5, 41.5, 1)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.hist(g, bins=bins, color=BLUE, label=f"genuine ({len(g)})", rwidth=0.85)
    ax.hist(i, bins=bins, color=ORANGE, label=f"impostor ({len(i)})", rwidth=0.85)
    ax.axvline(2.5, color=MUTED, lw=1, ls="--")
    ax.text(2.8, ax.get_ylim()[1] * 0.9, "t = 2", color=INK, fontsize=8)
    ax.set_xlabel("wrong symbols (of 40)")
    ax.set_ylabel("boots")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig16_simboli_errati.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
