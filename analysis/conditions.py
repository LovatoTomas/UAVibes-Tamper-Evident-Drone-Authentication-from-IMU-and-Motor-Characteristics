"""Altered-drone boots: propeller change and added payload (2026-10-14).

Each session in data/fixed_rig_props/ or data/fixed_rig_payload/ is one boot
(same boot-seq protocol as data/fixed_rig_boot/) of a drone that was altered
after enrollment. It is compared with the drone's own baseline: the median of
its last 3 nominal boot sessions recorded BEFORE it (same mounting, closest in
time). For every altered boot:

- wrong symbols of the key (all / motors / board) against that baseline, the
  same count the fuzzy extractor corrects (accept iff <= t);
- the same count for the drone's other nominal boots, the normal range;
- the shift of every feature in within-drone standard deviations
  (sigma = bin width / K; a symbol flips past +-K/2 sigma);
- the motor speed change per motor (%), and the speed alarm (keyproto.fe:
  any motor's gain more than SPEED_ALARM_SIGMA from the reference — the
  tampering check that sits next to the key, since one motor = one symbol);
- whether the station's current helper (data/keystore/) would regenerate the
  key — computed without touching the keystore.

Expected (thesis hypothesis): a propeller change moves the motor symbols
(load -> speed and vibration) and breaks the key; a payload on this light
frame moves little, and the board part (static) not at all.

    python -m analysis.conditions            # every altered session
    python -m analysis.conditions props 3    # one session

Writes <session>/analysis/verdict.{json,md} and
analysis/results_conditions/{conditions.json, summary.md}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from keyproto import boot as kb
from keyproto.fe import FuzzyExtractor, SPEED_ALARM_SIGMA, gain_index, speed_alarm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "results_conditions"
KIND_DIRS = {"props": ROOT / "data" / "fixed_rig_props", "payload": ROOT / "data" / "fixed_rig_payload"}
N_BASE = 3


def _sessions(root: Path) -> "list[tuple[Path, dict]]":
    out = []
    for f in sorted(root.glob("s*_*/session.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        if any(a.get("converted_path") for r in m["recordings"] for a in r["attempts"]):
            out.append((f.parent, m))
    return out


def _nominal(drone: str) -> "list[tuple[str, np.ndarray]]":
    """(created_iso, features) of the drone's nominal boots, oldest first."""
    rows = [(m["created_iso"], kb.boot_features(d)[0]) for d, m in _sessions(kb.BOOT_DIR) if m["drone"] == drone]
    return sorted(rows, key=lambda r: r[0])


def _wrong(x, ref, w) -> np.ndarray:
    """Per-symbol wrong flag with offsets centring ref (NaN counts as wrong)."""
    idx = np.floor(ref / w)
    off = (idx + 0.5) * w - ref
    q = np.floor((x + off) / w)
    return (q != idx) | np.isnan(q)


def evaluate(session_dir: Path, m: dict) -> dict:
    sysp = kb.system_params()
    names = sysp["features"]
    w = np.asarray(sysp["widths"], dtype=float)
    K, t = sysp["K"], sysp["t"]
    is_mot = np.array([n in kb.MOTOR_FEATS for n in names])
    x, info = kb.boot_features(session_dir)

    nominal = _nominal(m["drone"])
    before = [v for iso, v in nominal if iso < m["created_iso"]]
    base = before[-N_BASE:] if len(before) >= N_BASE else [v for _, v in nominal[:N_BASE]]
    if not base:
        raise ValueError(f"no nominal boot session for drone {m['drone']}")
    ref = np.nanmedian(np.vstack(base), axis=0)
    base_ids = {id(v) for v in base}
    others = [v for _, v in nominal if id(v) not in base_ids]

    wr = _wrong(x, ref, w)
    normal = [_wrong(v, ref, w) for v in others]
    z = (x - ref) / (w / K)
    gains = {f"M{i}": float(100 * (x[names.index(f"m{i}_{kb.BOOT_PWM}_gain")] /
                                   ref[names.index(f"m{i}_{kb.BOOT_PWM}_gain")] - 1)) for i in (1, 2, 3, 4)}

    gi = gain_index(names)
    shift = {mm: float(z[i]) for mm, i in gi.items()}
    normal_shift = [float(np.nanmax(np.abs((v - ref)[list(gi.values())] / (w / K)[list(gi.values())])))
                    for v in others]

    ks = None
    helper = kb._jload(kb.KEYSTORE / f"drone{m['drone']}_helper.json")
    if helper is not None:
        fe = FuzzyExtractor(w, t, robust="tag" in helper)   # keystore helpers have no tag since 2026-10-29
        key, W = fe.rep(x, helper)
        ks = dict(ok=key is not None, reason=None if key is not None else W,
                  enrolled_from=helper.get("enrolled_from"))
        if key is not None:
            zk = fe.shift_bins(helper, W, x) * K
            ks["speed_shift_sigma"] = {mm: float(zk[i]) for mm, i in gi.items()}
            ks["speed_alarm"] = speed_alarm(ks["speed_shift_sigma"])
        ks["granted"] = bool(ks["ok"] and not ks.get("speed_alarm"))   # alarm -> access denied

    order = np.argsort(-np.nan_to_num(np.abs(z), nan=np.inf))
    return dict(
        session=m["session"], kind=m["kind"], drone=m["drone"], created=m["created_iso"],
        condition=m.get("condition"), board_temp_c=info["board_temp_c"], n_missing=info["n_missing"],
        baseline=dict(n=len(base), before_session=len(before) >= N_BASE),
        wrong=dict(all=int(wr.sum()), motors=int(wr[is_mot].sum()), board=int(wr[~is_mot].sum())),
        accepted=bool(wr.sum() <= t), t=int(t), n_symbols=len(names),
        granted=bool(wr.sum() <= t and not speed_alarm(shift)),       # key AND no speed alarm
        normal_wrong=dict(all=[int(v.sum()) for v in normal],
                          motors=[int(v[is_mot].sum()) for v in normal],
                          board=[int(v[~is_mot].sum()) for v in normal]),
        speed_change_pct=gains,
        speed_shift_sigma=shift, speed_alarm=speed_alarm(shift), speed_alarm_sigma=SPEED_ALARM_SIGMA,
        normal_max_speed_shift_sigma=normal_shift,
        top_shifts=[dict(feature=names[i], z=None if np.isnan(z[i]) else round(float(z[i]), 1),
                         wrong=bool(wr[i])) for i in order[:8]],
        median_abs_z=dict(motors=float(np.nanmedian(np.abs(z[is_mot]))),
                          board=float(np.nanmedian(np.abs(z[~is_mot])))),
        keystore=ks, half_bin_sigma=K / 2,
    )


def _cond_text(r: dict) -> str:
    c = r.get("condition") or {}
    if r["kind"] == "props":
        mot = ", ".join(f"M{i}" for i in c.get("motors", [])) or "?"
        return f"propellers: {c.get('change', '?')} (motors {mot})"
    return (f"payload: {c.get('mass_g', '?')} g, {c.get('position', '?')}, "
            f"mounted with {c.get('mount', '?')}")


def verdict_md(r: dict) -> str:
    rng = lambda v: f"{min(v)}–{max(v)}" if v else "—"
    nw = r["normal_wrong"]
    L = [f"### Session {r['session']:02d} — drone {r['drone']} — {_cond_text(r)}", ""]
    if (r.get("condition") or {}).get("notes"):
        L += [f"Notes: {r['condition']['notes']}", ""]
    L += [
        f"Reference: median of the {r['baseline']['n']} "
        + ("previous nominal boot sessions." if r["baseline"]["before_session"]
           else "oldest nominal boot sessions (there were not enough before this one)."),
        "",
        "| | wrong symbols | normal boots of the same drone |",
        "|---|---|---|",
        f"| motors ({len(kb.MOTOR_FEATS)}) | **{r['wrong']['motors']}** | {rng(nw['motors'])} |",
        f"| board ({len(kb.BOARD_FEATS)}) | **{r['wrong']['board']}** | {rng(nw['board'])} |",
        f"| total ({r['n_symbols']}) | **{r['wrong']['all']}** | {rng(nw['all'])} |",
        "",
        f"**{'Key regenerated' if r['accepted'] else 'Key NOT regenerated'}** "
        f"(corrects up to t = {r['t']} symbols).",
    ]
    al = r["speed_alarm"]
    ns = r["normal_max_speed_shift_sigma"]
    L += ["", (f"**Speed alarm: YES** (motors {', '.join(f'M{i}' for i in al)}; "
               if al else "Speed alarm: no (")
          + "speed shift in σ: "
          + ", ".join(f"M{k} {v:+.1f}" for k, v in r["speed_shift_sigma"].items())
          + f"; threshold ±{r['speed_alarm_sigma']:.0f}σ"
          + (f", normal boots up to {max(ns):.1f}σ" if ns else "") + ")."]
    L += ["", f"**Access {'granted' if r['granted'] else 'DENIED'}** "
              f"(it needs the regenerated key and no speed alarm)."]
    if r["keystore"] is not None:
        k = r["keystore"]
        L.append(f"With the station's current helper data: key "
                 f"{'regenerated' if k['ok'] else 'NOT regenerated'}"
                 + ((", speed alarm " + ("YES (" + ", ".join(f"M{i}" for i in k["speed_alarm"]) + ")"
                                         if k["speed_alarm"] else "no")) if k["ok"] else "")
                 + f" → access {'granted' if k['granted'] else 'denied'}.")
    L += ["", "Motor speed against the reference: "
          + ", ".join(f"{k} {v:+.1f} %" for k, v in r["speed_change_pct"].items()) + ".", "",
          f"Most shifted quantities (in within-drone standard deviations; a symbol changes beyond "
          f"±{r['half_bin_sigma']:.0f}):", ""]
    L += [f"- `{s['feature']}`: {s['z'] if s['z'] is not None else 'missing'}"
          + (" — symbol changed" if s["wrong"] else "") for s in r["top_shifts"]]
    return "\n".join(L) + "\n"


def run(kind: "str | None" = None, session: "int | None" = None, verbose: bool = True) -> list:
    results = []
    for k, root in KIND_DIRS.items():
        if kind and k != kind:
            continue
        for d, m in _sessions(root):
            if session is not None and m["session"] != session:
                continue
            r = evaluate(d, m)
            (d / "analysis").mkdir(exist_ok=True)
            (d / "analysis" / "verdict.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
            (d / "analysis" / "verdict.md").write_text(verdict_md(r), encoding="utf-8")
            results.append(r)
            if verbose:
                print(f"{k} s{m['session']:02d} drone {m['drone']}: wrong motors {r['wrong']['motors']}, "
                      f"board {r['wrong']['board']} -> key {'OK' if r['accepted'] else 'REJECTED'}, "
                      f"speed alarm {r['speed_alarm'] or 'no'} -> access "
                      f"{'granted' if r['granted'] else 'DENIED'}")
    if kind is None and session is None:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "conditions.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        md = ["# Boots with changed propellers and with payload", ""]
        for k, title in (("props", "## Propeller change"), ("payload", "## Payload")):
            rs = [r for r in results if r["kind"] == k]
            md += [title, ""] + ([verdict_md(r) for r in rs] or ["No session recorded.", ""])
        (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
        if verbose:
            print(f"-> {OUT}")
    return results


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0] if a else None, int(a[1]) if len(a) > 1 else None)
