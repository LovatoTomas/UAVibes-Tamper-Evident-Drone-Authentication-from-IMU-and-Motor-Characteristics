"""
On-board key (firmware FW2) on the real drones: what the station logged.

Every attempt made from the dashboard's 📡 tab is a line of
data/keystore/onboard/log.jsonl. This script turns that log into the numbers
the thesis needs and, above all, into the one the professor asked for before
the speed symbols are taken out of the error correction and put into the
password (approved 2026-10-19, "as long as it does not cause too many
failures"):

  1. per drone and per day, nominal boots only: accepted / rejected,
     corrected symbols, largest speed shift of the boot;
  2. the speed shift of each motor on nominal accepted boots (in
     within-drone sigma, against the helper's reference, i.e. the median of
     the last 3 accepted boots): spread per motor and per drone;
  3. exact speed symbols: with a bin of width 2T centred on the reference, a
     speed symbol changes, hence the password, iff |shift| > T. For T = 3, 4,
     5 (5 = the current alarm): how many nominal boots would have been
     rejected, and how many altered boots (propeller, tape) caught;
  4. how far the observed rejections bound the false-rejection rate (95 %
     Clopper-Pearson upper limit), and how many nominal boots with no
     rejection are needed to bound it below 5, 2 and 1 %;
  5. the altered boots (tape, propeller, cloned helper): what stopped them.

The condition of each attempt (nominal or altered) comes from the record's
`condition` field (the dashboard asks for it since 2026-10-29) or, for the
earlier attempts, from analysis/onboard_log_labels.json (reconstructed from
the operator's reports of 2026-10-18/19). Simulated runs are left out.

Run:  python -m analysis.onboard_log     -> analysis/results_onboard_log/
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis.dataset import ROOT

LOG = ROOT / "data" / "keystore" / "onboard" / "log.jsonl"
LABELS = ROOT / "analysis" / "onboard_log_labels.json"
OUT = ROOT / "analysis" / "results_onboard_log"
THRESHOLDS = (3.0, 4.0, 5.0)
FRR_TARGETS = (0.05, 0.02, 0.01)


def load(log: Path = LOG, labels: Path = LABELS) -> pd.DataFrame:
    """One row per real authentication attempt, with its condition."""
    rows = [json.loads(l) for l in Path(log).read_text(encoding="utf-8").splitlines() if l.strip()]
    lab = json.loads(Path(labels).read_text(encoding="utf-8"))["windows"] if Path(labels).exists() else []
    out = []
    for r in rows:
        if r.get("sim") or r.get("mode") != "auth":
            continue
        at = str(r.get("at", ""))
        cond, note = r.get("condition"), r.get("condition_note")
        for w in lab:
            # a window fills in the condition of the attempts logged without one;
            # an "override" window (decided after the fact, e.g. a motor found
            # damaged) replaces even the condition written at recording time
            if (not cond or w.get("override")) and w["from"] <= at <= w["to"]                     and w.get("claimed") in (None, r.get("expected")):
                cond, note = w["condition"], w.get("note")
                break
        sh = r.get("speed_shift_sigma") or {}
        shifts = {int(k): (np.nan if v is None else float(v)) for k, v in sh.items()}
        # Stages and reasons are English since 2026-10-29; the attempts logged
        # before carry the Italian wording of the first version of the station.
        stage, reason = r.get("stage"), str(r.get("reason", ""))
        if r.get("ok"):
            outcome = "accepted"
        elif stage in ("speed_alarm", "allarme velocità"):
            outcome = "speed_alarm"
        elif stage == "drone" and ("too many wrong symbols" in reason or "troppi simboli" in reason):
            outcome = "key_rejected"
        elif stage in ("station", "stazione") and ("not recognised" in reason or "non riconosciuto" in reason):
            outcome = "helper_refused"
        elif stage in ("station", "stazione") and ("confirmation" in reason or "conferma" in reason):
            outcome = "pake_failed"
        else:
            outcome = "error"          # timeout, link, station not ready...: not a decision on the identity
        out.append(dict(at=at, day=at[:10], claimed=r.get("expected"), condition=cond or "unlabelled",
                        note=note, outcome=outcome, wrong=r.get("wrong"), reason=r.get("reason"),
                        **{f"shift_m{m}": shifts.get(m, np.nan) for m in (1, 2, 3, 4)}))
    df = pd.DataFrame(out)
    if len(df):
        sc = [f"shift_m{m}" for m in (1, 2, 3, 4)]
        df["max_abs_shift"] = df[sc].abs().max(axis=1, skipna=True)
    return df


def cp_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper limit of a rate with k events in n."""
    if n == 0:
        return float("nan")
    return 1.0 if k >= n else float(stats.beta.ppf(conf, k + 1, n - k))


def n_needed(p: float, conf: float = 0.95) -> int:
    """Nominal boots with zero rejections needed for an upper limit below p."""
    return int(math.ceil(math.log(1 - conf) / math.log(1 - p)))


def analyse(df: pd.DataFrame) -> dict:
    sc = [f"shift_m{m}" for m in (1, 2, 3, 4)]
    nom = df[df.condition == "nominal"]
    dec = nom[nom.outcome != "error"]                 # attempts with a decision on the identity
    acc = dec[dec.outcome == "accepted"]
    res = dict(n_records=int(len(df)), conditions=df.condition.value_counts().to_dict(),
               outcomes_nominal=nom.outcome.value_counts().to_dict())

    per = []
    for (d, day), g in dec.groupby(["claimed", "day"]):
        a = g[g.outcome == "accepted"]
        per.append(dict(drone=d, day=day, n=int(len(g)), accepted=int(len(a)),
                        rejected=int(len(g) - len(a)),
                        wrong_symbols=a.wrong.dropna().astype(int).value_counts().sort_index().to_dict(),
                        max_abs_shift=float(a.max_abs_shift.max()) if len(a) else None))
    res["per_drone_day"] = per
    res["days"] = sorted(dec.day.unique().tolist())

    # 2. spread of the speed shifts on nominal accepted boots
    v = acc[sc].to_numpy(dtype=float)
    res["shift"] = dict(
        n_boots=int(len(acc)),
        sd_per_motor={m: float(np.nanstd(v[:, i], ddof=1)) if np.isfinite(v[:, i]).sum() > 1 else None
                      for i, m in enumerate((1, 2, 3, 4))},
        sd_pooled=float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else None,
        max_abs=float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else None,
        sd_per_drone={d: float(np.nanstd(g[sc].to_numpy(dtype=float), ddof=1)) for d, g in acc.groupby("claimed")
                      if len(g) > 1})

    # 3. exact speed symbols: extra rejections and altered boots caught
    alt = df[df.condition.isin(["propeller", "tape"]) & (df.outcome != "error")]
    sd = res["shift"]["sd_pooled"]
    exact = []
    for T in THRESHOLDS:
        extra = int((acc.max_abs_shift > T).sum())
        n = int(len(dec))
        k = int((dec.outcome != "accepted").sum()) + extra
        caught = int(((alt.outcome != "accepted") | (alt.max_abs_shift > T)).sum())
        # P(any of the 4 motors beyond T), computed without cancellation for tiny values
        p4 = float(-np.expm1(4 * np.log1p(-2 * stats.norm.sf(T / sd)))) if sd else None
        exact.append(dict(T=T, extra_rejections=extra, rejections_total=k, n=n,
                          frr=k / n if n else None, frr_upper95=cp_upper(k, n),
                          altered_caught=caught, altered_n=int(len(alt)),
                          gaussian_rejection_per_boot=p4))
    res["exact_speed_symbols"] = exact

    # 4. what the current data bound, and what is needed
    k0 = int((dec.outcome != "accepted").sum())
    res["frr_now"] = dict(rejections=k0, n=int(len(dec)), frr=k0 / len(dec) if len(dec) else None,
                          upper95=cp_upper(k0, len(dec)))
    res["boots_needed_zero_failures"] = {f"{int(p * 100)}%": n_needed(p) for p in FRR_TARGETS}

    # 5. altered boots
    res["altered"] = [dict(at=r.at, claimed=r.claimed, condition=r.condition, outcome=r.outcome,
                           wrong=r.wrong, max_abs_shift=None if not np.isfinite(r.max_abs_shift) else float(r.max_abs_shift),
                           note=r.note)
                      for r in df[~df.condition.isin(["nominal", "excluded_fw_bug", "excluded_hw_fault"])].itertuples()]
    return res


def report(res: dict) -> str:
    L = ["# On-board key (FW2): what the station log says", "",
         "Generated by `python -m analysis.onboard_log` (do not edit by hand). Real boots only; condition "
         "from the log or from `analysis/onboard_log_labels.json`. Speed shifts in within-drone σ, against "
         "the reference of the helper data (median of the last 3 accepted boots).", "",
         f"Attempts: {res['n_records']} · conditions: " + ", ".join(f"{k} {v}" for k, v in res["conditions"].items()),
         f"Days with nominal boots: {', '.join(res['days']) or '—'}", "",
         "## 1. Nominal boots per drone and day", "",
         "| Drone | Day | Boots | Accepted | Rejected | Corrected symbols (n: boots) | Max \\|shift\\| |",
         "|---|---|---|---|---|---|---|"]
    for p in res["per_drone_day"]:
        ws = ", ".join(f"{k}: {v}" for k, v in p["wrong_symbols"].items()) or "—"
        ms = f"{p['max_abs_shift']:.2f}σ" if p["max_abs_shift"] is not None else "—"
        L.append(f"| {p['drone']} | {p['day']} | {p['n']} | {p['accepted']} | {p['rejected']} | {ws} | {ms} |")
    s = res["shift"]
    L += ["", "## 2. Speed shift on the accepted nominal boots", "",
          f"{s['n_boots']} boots. Standard deviation per motor: "
          + ", ".join(f"M{m} {v:.2f}σ" for m, v in s["sd_per_motor"].items() if v is not None)
          + (f"; pooled {s['sd_pooled']:.2f}σ; largest |shift| {s['max_abs']:.2f}σ." if s["sd_pooled"] else "."),
          "Per drone: " + ", ".join(f"{d} {v:.2f}σ" for d, v in s["sd_per_drone"].items()) + ".", "",
          "## 3. Exact speed symbols in the password", "",
          "Bin of width 2T centred on the reference: the symbol (hence the password) changes when |shift| > T.", "",
          "| T | Extra rejections | Total rejections / boots | FRR | FRR, 95 % upper limit | Altered boots caught | Rejection per boot (Gaussian) |",
          "|---|---|---|---|---|---|---|"]
    for e in res["exact_speed_symbols"]:
        g = f"{e['gaussian_rejection_per_boot']:.1e}" if e["gaussian_rejection_per_boot"] is not None else "—"
        L.append(f"| {e['T']:.0f}σ | {e['extra_rejections']} | {e['rejections_total']} / {e['n']} | "
                 f"{100 * e['frr']:.1f} % | {100 * e['frr_upper95']:.1f} % | {e['altered_caught']} / {e['altered_n']} | {g} |")
    f = res["frr_now"]
    L += ["", "The Gaussian column assumes normally distributed shifts with the pooled deviation of §2, "
          "independent between the 4 motors. **It is not an estimate until the data cover many days**: "
          + (f"today every drone has boots of a single day ({len(res['days'])} days in all), while offline, "
             "across days, normal boots moved by up to 3.1σ (`paper/experiments/chiave_a_bordo.md` §8). "
             "The column of observed rejections is the one that counts."
             if len(res["days"]) < 5 else "with more days the estimate becomes sensible, but it stays an assumption."), "",
          "## 4. How many boots are needed", "",
          f"Today: {f['rejections']} rejections in {f['n']} nominal boots, FRR {100 * (f['frr'] or 0):.1f} %, "
          f"95 % upper limit {100 * f['upper95']:.1f} %.", "",
          "Consecutive nominal boots with no rejection needed to say, at 95 %, that the FRR is below: "
          + ", ".join(f"{k} → {v}" for k, v in res["boots_needed_zero_failures"].items())
          + ". They must be spread over several days (and battery charges), because the drift shows "
            "from one day to the next.", "",
          "## 5. Altered boots and security tests", "",
          "| When | Claims to be | Condition | Outcome | Corrected symbols | Max \\|shift\\| | Note |",
          "|---|---|---|---|---|---|---|"]
    for a in res["altered"]:
        ms = f"{a['max_abs_shift']:.2f}σ" if a["max_abs_shift"] is not None else "—"
        w = int(a["wrong"]) if a["wrong"] is not None and np.isfinite(a["wrong"]) and a["wrong"] >= 0 else "—"
        L.append(f"| {a['at']} | {a['claimed']} | {a['condition']} | {a['outcome']} | {w} | {ms} | {a['note'] or ''} |")
    return "\n".join(L) + "\n"


def main() -> None:
    if not LOG.exists():
        raise SystemExit(f"no log at {LOG}")
    df = load()
    res = analyse(df)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "attempts.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (OUT / "report.md").write_text(report(res), encoding="utf-8")
    print(report(res))


if __name__ == "__main__":
    main()
