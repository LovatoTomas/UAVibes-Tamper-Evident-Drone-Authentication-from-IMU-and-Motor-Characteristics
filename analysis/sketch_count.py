"""
Exact count of plausible readings that survive the public syndrome (defensive, follows
analysis/sketch_attack.py). For each drone, enumerate the high-entropy symbols over their likely
bins (+/- Z sd around the population mean, the near-deterministic symbols fixed at their mode),
compute the syndrome of every candidate with numpy, and count how many match the target's public
syndrome -- and whether the true reading is among the survivors. This is the reviewer's question:
"how many physically plausible candidates survive the syndrome check".

    python -m analysis.sketch_count   ->  analysis/results_sketch_attack/count.json
"""
import json
import math
from pathlib import Path

import numpy as np

from keyproto import boot as kb
from keyproto import onboard as ob
from keyproto.fe import P
from analysis.sketch_attack import syndrome_matrix, drone_refs, population_model, ENROLL, _DetRNG

OUT = Path(__file__).resolve().parent / "results_sketch_attack"
Z = 3.5                     # likely region: mean +/- Z sd (bins). smaller than sketch_attack's 4
MAX_GRID = 8_000_000       # cap on the enumerated product; trims the widest symbols if exceeded


def count_for(target, M, refs, idx, sysp):
    w = np.asarray(sysp["widths"], float)
    supports, probs, ent = population_model(idx, target, Z=Z)
    Wtrue = (idx[target] % P).astype(np.int64)
    s_star = tuple(int(v) for v in (M @ Wtrue) % P)

    # make sure the true bin is in each symbol's grid, so "does the true reading survive" is
    # answered honestly (otherwise the attacker's grid could miss it for a reason unrelated to the
    # syndrome); note separately whether it was already inside the +-Z sd region.
    true_in_region = True
    grids = []
    for i in range(len(supports)):
        s = list(supports[i])
        if int(idx[target][i]) not in s:
            s.append(int(idx[target][i]))
            true_in_region = False
        grids.append(np.array(sorted(set(v % P for v in s)), dtype=np.int64))

    enum = [i for i in range(len(grids)) if len(grids[i]) > 1]
    fixed = {i: int(grids[i][0]) for i in range(len(grids)) if len(grids[i]) == 1}
    # trim: if the product is too large, fix the widest-support symbols at the true bin
    enum.sort(key=lambda i: len(grids[i]))
    size = 1
    keep = []
    for i in enum:
        if size * len(grids[i]) > MAX_GRID:
            fixed[i] = int(idx[target][i]) % P
        else:
            size *= len(grids[i])
            keep.append(i)
    enum = keep
    enum_box = int(np.prod([len(grids[i]) for i in enum])) if enum else 1

    Mi = (M.astype(np.int64))
    const = np.zeros(4, dtype=np.int64)
    for i, v in fixed.items():
        const = (const + Mi[:, i] * v) % P

    # The attacker fixes the near-deterministic symbols at the population MODE (grids[i][0]), not
    # at the target's true value -- it does not know the target. So the enumerated grid may or may
    # not contain the true reading; we check whether the surviving candidate equals the true W.
    cols = [grids[i] for i in enum]
    Mcols = [Mi[:, i] for i in enum]
    base = (const[None, :]) % P
    batch = base                                           # B x 4, running syndromes
    sizes = []
    for lvl in range(len(enum)):
        vals = cols[lvl]
        contrib = (np.outer(vals, Mcols[lvl]) % P)         # V x 4
        batch = ((batch[:, None, :] + contrib[None, :, :]) % P).reshape(-1, 4)
        sizes.append(len(vals))
    match_idx = np.nonzero(np.all(batch == np.array(s_star), axis=1))[0]
    survivors = int(len(match_idx))

    # reconstruct every surviving reading and test it against the true reading
    survivor_is_true = False
    for flat in match_idx:
        W = Wtrue.copy()                                   # start from true, overwrite non-fixed
        for i, v in fixed.items():
            W[i] = v
        rem = int(flat)
        for lvl in range(len(enum) - 1, -1, -1):
            rem, j = divmod(rem, sizes[lvl])
            W[enum[lvl]] = int(cols[lvl][j])
        if np.array_equal(W % P, Wtrue):
            survivor_is_true = True

    # does the true reading even lie in the attacker's grid? (fixed symbols at mode == true value,
    # and every enumerated symbol's true bin present). If not, the attacker's search cannot find it.
    true_reachable = all(int(Wtrue[i]) == v for i, v in fixed.items()) and \
        all(int(idx[target][i]) % P in set(int(x) for x in grids[i]) for i in enum)

    return dict(
        enum_symbols=len(enum),
        enum_box=int(enum_box),
        survivors_in_grid=survivors,
        survivor_is_true_reading=bool(survivor_is_true),
        true_reading_in_attacker_grid=bool(true_reachable),
        true_bin_within_region=bool(true_in_region),
        reduction_bits=round(math.log2(enum_box / survivors), 1) if survivors else None,
    )


def main():
    sysp = kb.system_params()
    n = len(sysp["widths"])
    M = np.array(syndrome_matrix(n, n - 2 * sysp["t"]), dtype=np.int64)
    refs, idx, _ = drone_refs(sysp)
    res = {}
    for target in ENROLL:
        res[target] = count_for(target, M, refs, idx, sysp)
        print(target, res[target])
    OUT.mkdir(exist_ok=True)
    (OUT / "count.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
