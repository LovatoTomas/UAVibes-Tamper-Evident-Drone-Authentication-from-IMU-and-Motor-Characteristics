"""
On-board feature computation (firmware FW2) checked on the real boots.

The firmware cannot run the offline pipeline as is (~37 KB of free RAM, no
float64 arrays of the whole log): it stores one hold as int16 and skips the
5 Hz high-pass (keyproto/onboard.py, firmware/uavibe_key/src/core). This
script recomputes every recorded boot (data/fixed_rig_boot/) both ways and
compares:

  1. feature by feature, in within-drone standard deviations (sigma = bin
     width / K): systematic offset and spread of on-board minus offline;
  2. within-drone stability of each feature group, on-board vs offline;
  3. the key at symbol level, exactly as in report 11 (every choice of 3
     enrollment boots per drone, no refresh): genuine / impostor wrong
     symbols and FRR / FAR for t = 0..4, on-board vs offline;
  4. the int16 storage: saturated samples and quantisation noise against the
     lowest spectral floors.

A note on the sampling rate: the SD logs run at ~1005-1013 Hz by their own
timestamps (the offline pipeline uses the session median, 1017 Hz); the
drone samples at the FreeRTOS tick and assumes 1000 Hz. That is a constant
scale on the speed gain (about -1.5 %), the same at enrollment and at every
boot, hence harmless; it shows as the gain offset in (1).

Run:  python -m analysis.onboard_eval     -> analysis/results_onboard/
"""

import glob
import json

import numpy as np
import pandas as pd

from analysis import boot_eval as be
from analysis.dataset import ROOT
from keyproto import boot as kb
from keyproto import onboard as ob

OUT = ROOT / "analysis" / "results_onboard"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sysp = kb.system_params()
    names = sysp["features"]
    sig = np.asarray(sysp["widths"]) / sysp["K"]
    on, off, sat = [], [], []
    for s in sorted(glob.glob(str(ROOT / "data" / "fixed_rig_boot" / "s*_drone*"))):
        inp = ob.boot_inputs(s)
        x_off, _ = kb.boot_features(s)
        meta = dict(drone=inp["drone"], session=inp["session"])
        on.append(dict(meta, **dict(zip(names, ob.onboard_features(inp, sysp)))))
        off.append(dict(meta, **dict(zip(names, x_off))))
        for h in inp["holds"]:
            if h is not None:
                sat.append(int((np.abs(h["raw"].astype(int)) >= 32767).sum()))
    On, Off = pd.DataFrame(on), pd.DataFrame(off)
    On.to_csv(OUT / "features_onboard.csv", index=False)
    res = dict(n_boots=len(On), missing=dict(onboard=int(On[names].isna().sum().sum()),
                                             offline=int(Off[names].isna().sum().sum())))

    groups = {"gain": [n for n in names if n.endswith("gain")],
              "floor": [n for n in names if "floor" in n],
              "board": [n for n in names if not n.startswith("m")]}
    res["groups"] = {}
    for g, cols in groups.items():
        ix = [names.index(c) for c in cols]
        d = (On[cols].values - Off[cols].values) / sig[ix]
        res["groups"][g] = dict(
            offset_sigma_median=float(np.nanmedian(d)),
            spread_sigma_median=float(np.nanmedian(np.nanstd(d, axis=0))),
            max_abs_dev_from_offset=float(np.nanmax(np.abs(d - np.nanmedian(d, axis=0)))),
            within_drone_sd_sigma=dict(
                onboard=float((On.groupby("drone")[cols].std().mean() / sig[ix]).median()),
                offline=float((Off.groupby("drone")[cols].std().mean() / sig[ix]).median())))
        v = res["groups"][g]
        print(f"{g:6s} offset {v['offset_sigma_median']:+.2f} sigma, spread {v['spread_sigma_median']:.2f}; "
              f"within-drone sd on-board {v['within_drone_sd_sigma']['onboard']:.2f} "
              f"vs offline {v['within_drone_sd_sigma']['offline']:.2f}")

    res["key"] = {}
    for nm, cols in (("combined", names), ("motors_only", groups["gain"] + groups["floor"]),
                     ("board_only", groups["board"])):
        r_on = be.key_eval(On.assign(start=range(len(On))), names, cols)
        r_off = be.key_eval(Off.assign(start=range(len(Off))), names, cols)
        res["key"][nm] = dict(onboard=r_on, offline=r_off)
        print(f"{nm:12s} t=2: on-board FRR {r_on['by_t'][2]['frr']:.2f} FAR {r_on['by_t'][2]['far']:.3f} "
              f"(genuine max {r_on['genuine_max']}, impostor min {r_on['impostor_min']}) | offline FRR "
              f"{r_off['by_t'][2]['frr']:.2f} FAR {r_off['by_t'][2]['far']:.3f}")

    lsb = ob.LSB.astype(float)
    q_db = 10 * np.log10(lsb ** 2 / 12 / (ob.FS / 2))
    floors = Off[groups["floor"]]
    res["int16"] = dict(saturated_samples=int(sum(sat)), holds=len(sat),
                        quant_noise_db=dict(acc=float(q_db[0]), gyro=float(q_db[3])),
                        lowest_floor_db=dict(acc=float(floors[[c for c in floors if "acc" in c]].min().min()),
                                             gyro=float(floors[[c for c in floors if "gyro" in c]].min().min())))
    print("int16:", res["int16"])
    (OUT / "onboard_eval.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"-> {OUT}")
    return res


if __name__ == "__main__":
    main()
