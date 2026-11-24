"""
Accelerometer self-test alone, over every recorded boot (board, boot, propeller
and payload sessions, 7 Oct - 4 Nov 2026): is each boot assigned to the right
drone? python -m analysis.selftest_all  ->  analysis/results_selftest_all/summary.json

A run counts only if it passes the validity check of the current firmware
(pos != neg on every axis and |pos.x| + |neg.x| > 15000 LSB): the two runs of D
on the first day that failed it (identical readings, 6 g range) are left out,
as the firmware now repeats them. C's second day of 14 Oct stays out, as
everywhere in the thesis. Rule: sensitivity d = (pos - neg) / 2 per axis,
nearest per-drone median, leave-one-out.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results_selftest_all"
TREES = ("fixed_rig_board", "fixed_rig_boot", "fixed_rig_props", "fixed_rig_payload")
C_DAY2 = {27, 28, 29, 30}


def load() -> pd.DataFrame:
    rows = []
    for tree in TREES:
        for d in sorted((ROOT / "data" / tree).glob("s*_drone*")):
            try:
                m = json.loads((d / "session.json").read_text(encoding="utf-8"))
            except Exception:
                continue
            b = m.get("board_params") or {}
            if not b.get("firmware") or b.get("stResult") != 0:
                continue
            if tree == "fixed_rig_boot" and m["drone"] == "C" and m["session"] in C_DAY2:
                continue
            pos = [b[f"stPos{a}"] for a in "XYZ"]
            neg = [b[f"stNeg{a}"] for a in "XYZ"]
            valid = all(p != n for p, n in zip(pos, neg)) and abs(pos[0]) + abs(neg[0]) > 15000
            rows.append(dict(tree=tree, drone=m["drone"], session=m["session"], day=m["created_iso"][:10],
                             valid=valid, **{f"d{a}": (p - n) / 2 for a, p, n in zip("XYZ", pos, neg)}))
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load()
    v = df[df.valid].reset_index(drop=True)
    X = v[["dX", "dY", "dZ"]].values
    hit = []
    for i in range(len(v)):
        others = np.ones(len(v), bool)
        others[i] = False
        med = v[others].groupby("drone")[["dX", "dY", "dZ"]].median()
        dist = {d: np.linalg.norm(X[i] - med.loc[d].values) for d in med.index}
        hit.append(min(dist, key=dist.get) == v.drone.iloc[i])
    v["identified"] = hit
    summary = dict(runs=int(len(df)), invalid_runs=int((~df.valid).sum()), valid_runs=int(len(v)),
                   identified=int(v.identified.sum()), days=int(v.day.nunique()),
                   per_kind={k: int(n) for k, n in v.tree.value_counts().items()},
                   per_drone={k: int(n) for k, n in v.drone.value_counts().sort_index().items()})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    v.to_csv(OUT / "boots.csv", index=False)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
