"""
Does the public secure sketch leak the reading? (defensive security analysis of our own
prototype, prompted by a thesis review, 2026-11-12)

    python -m analysis.sketch_attack   ->  analysis/results_sketch_attack/{summary.json, report.md}

The helper data stored on the drone hold the code-offset sketch  S = W - C , with C a random
Reed-Solomon codeword of RS[40, 36] over GF(65537). Because the code is LINEAR, anyone who reads
the memory (attacker capability A2) knows the 4 syndrome equations that any valid reading must
satisfy:  M . W = M . S , where M (4 x 40) is a basis of the left null space of the code's
Vandermonde. This script asks, on the REAL enrolled drones, how much of W an attacker can still
not guess once the public syndrome is known -- i.e. whether "the password is never checkable
offline" holds against a memory reader.

Threat model used here (matches the thesis A2 + A4, attacker owns drones of the same model):
  * the attacker has the target's public helper data (offsets + sketch);
  * the attacker does NOT know the target drone; it models the population of W from the OTHER
    drones (leave-one-out), exactly what owning same-model drones gives.

What it computes, per target drone:
  1. per-quantity entropy of W under the leave-one-out population model, and the total H(W)
     (this is the ~56-bit figure, recomputed here as the attacker would have it, without the
     target);
  2. the residual collision entropy H2(W | syndrome): how much unpredictability of W survives the
     4 public equations, from a Monte-Carlo estimate of the syndrome collision probability;
  3. a concrete structured attack: fix the symbols that the population pins down, enumerate the
     few high-entropy symbols over their plausible support, solve the 4 linear syndrome equations
     for 4 chosen symbols, and count how many candidate readings are BOTH syndrome-consistent and
     population-plausible -- and whether the target's true W is the only survivor.

No project text, figures or reproduce.py steps are touched; this is a standalone diagnostic.
"""
import json
import math
import time
from itertools import product
from pathlib import Path

import numpy as np

from keyproto import boot as kb
from keyproto import onboard as ob
from keyproto.fe import P

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results_sketch_attack"
RESEARCH = ROOT / "data" / "keystore" / "onboard" / "research"
# the five-day enrollment files (one current file per drone, three boots each)
ENROLL = {"A": "enroll_A_old_20261104_113034.json", "B": "enroll_B.json",
          "C": "enroll_C.json", "D": "enroll_D.json"}
N_ENROLL = 3
Z = 4.0          # plausible support of a symbol: reference bin +/- Z * between-drone sd (in bins)
MC = 400_000     # Monte-Carlo samples for the residual-entropy estimate


# ---- Reed-Solomon syndrome: M (n-k rows) with M . codeword = 0 -------------------------------
def syndrome_matrix(n=40, k=36):
    """Left null space of the n x k Vandermonde (rows x^0..x^{k-1}, x = 1..n), over GF(P)."""
    xs = list(range(1, n + 1))
    V = [[pow(x, j, P) for j in range(k)] for x in xs]          # n x k
    # augment with identity to track row operations: find rows r with r . V = 0
    # solve V^T . r = 0  (k equations, n unknowns) -> null space of V^T (k x n)
    A = [[V[i][j] for i in range(n)] for j in range(k)]         # k x n = V^T
    return _null_space(A, n)


def _inv(a):
    return pow(a % P, P - 2, P)


def _null_space(A, ncols):
    """Basis of the null space of A (rows of length ncols) over GF(P)."""
    M = [row[:] for row in A]
    nrows = len(M)
    pivot_col = {}
    r = 0
    for c in range(ncols):
        piv = next((i for i in range(r, nrows) if M[i][c] % P), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        iv = _inv(M[r][c])
        M[r] = [(v * iv) % P for v in M[r]]
        for i in range(nrows):
            if i != r and M[i][c] % P:
                f = M[i][c]
                M[i] = [(a - f * b) % P for a, b in zip(M[i], M[r])]
        pivot_col[c] = r
        r += 1
        if r == nrows:
            break
    free = [c for c in range(ncols) if c not in pivot_col]
    basis = []
    for fc in free:
        vec = [0] * ncols
        vec[fc] = 1
        for c, pr in pivot_col.items():
            vec[c] = (-M[pr][fc]) % P
        basis.append(vec)
    return basis


def syndrome(M, V):
    return tuple(sum(mij * vj for mij, vj in zip(row, V)) % P for row in M)


# ---- population model (leave-one-out) --------------------------------------------------------
def drone_refs(sysp):
    """Reference reading W (bin indices) and the raw reference vector of each drone, from its
    three enrollment boots, exactly as gen() computes them."""
    w = np.asarray(sysp["widths"], float)
    refs, idx = {}, {}
    for d, fn in ENROLL.items():
        boots = json.loads((RESEARCH / fn).read_text(encoding="utf-8"))
        X = np.vstack([np.asarray(b["features"], float) for b in boots[:N_ENROLL]])
        ref = np.median(X, axis=0)
        refs[d] = ref
        idx[d] = np.floor(ref / w).astype(np.int64)      # true bin index per quantity
    return refs, idx, w


def population_model(idx, target, Z=Z):
    """For each quantity, the plausible bin indices and their probabilities, seen by an attacker
    who knows every drone except `target` (leave-one-out). Mean and sd of the bin index over the
    other drones; support = round(mean) +/- Z*sd (at least the mean bin)."""
    others = [d for d in idx if d != target]
    stacked = np.vstack([idx[d] for d in others]).astype(float)
    mean = stacked.mean(axis=0)
    sd = stacked.std(axis=0, ddof=1) if len(others) > 1 else np.ones(stacked.shape[1])
    sd = np.maximum(sd, 1e-6)
    supports, probs, ent = [], [], []
    for mu, s in zip(mean, sd):
        lo, hi = int(math.floor(mu - Z * s)), int(math.ceil(mu + Z * s))
        bins = list(range(lo, hi + 1)) or [int(round(mu))]
        p = np.array([math.exp(-0.5 * ((b - mu) / s) ** 2) for b in bins])
        p = p / p.sum()
        supports.append(bins)
        probs.append(p)
        ent.append(float(-(p * np.log2(np.clip(p, 1e-300, 1))).sum()))
    return supports, probs, np.array(ent)


# ---- the experiment --------------------------------------------------------------------------
def analyse():
    sysp = kb.system_params()
    M = syndrome_matrix(len(sysp["widths"]), len(sysp["widths"]) - 2 * sysp["t"])
    refs, idx, w = drone_refs(sysp)
    rng = np.random.default_rng(0)
    results = {}
    for target in ENROLL:
        # the public helper of the target, as stored on the drone
        _, helper = ob.gen(refs[target], sysp, target, rng=_DetRNG())
        h = ob.unpack_helper(helper)
        Wtrue = [int(v) % P for v in idx[target]]
        synd_pub = syndrome(M, [int(s) for s in h["sketch"]])      # = M . W_true (public)
        assert synd_pub == syndrome(M, Wtrue)

        supports, probs, ent = population_model(idx, target)
        H_total = float(ent.sum())

        # Monte-Carlo residual collision entropy of W given the syndrome:
        # sample plausible W, bucket by syndrome, estimate collision prob of the syndrome.
        hi_ent = np.argsort(ent)[::-1]
        # vectorised Monte-Carlo: sample MC readings, syndrome = (W @ M^T) mod P, count collisions
        Mt = np.array(M, dtype=np.int64).T                       # 40 x 4
        samp = np.empty((MC, len(supports)), dtype=np.int64)
        for i in range(len(supports)):
            samp[:, i] = rng.choice(np.array(supports[i]) % P, size=MC, p=probs[i])
        synd = (samp @ Mt) % P                                   # MC x 4
        # pack the 4 syndrome coords into one key and count collisions
        keys = ((synd[:, 0] * P + synd[:, 1]) * P + synd[:, 2]) * P + synd[:, 3]
        _, counts = np.unique(keys, return_counts=True)
        coll = int((counts * (counts - 1)).sum())
        pairs = MC * (MC - 1)
        p_coll = coll / pairs if pairs else 0.0
        H2_synd = -math.log2(p_coll) if p_coll > 0 else float("inf")
        synd_counts = counts
        # collision entropy of W itself (independent symbols): H2 = -log2 sum p^2
        H2_W = float(sum(-math.log2((p ** 2).sum()) for p in probs))
        H2_resid = max(0.0, H2_W - H2_synd)

        # structured attack: enumerate the high-entropy symbols, solve the syndrome for 4 symbols,
        # count syndrome-consistent AND plausible candidates; check the true W is among them.
        free_cost = float(ent[hi_ent[:12]].sum())     # bits to enumerate if we brute the top ones
        survivors, true_found, enum = _structured_count(M, supports, probs, ent, synd_pub, Wtrue, rng)

        results[target] = dict(
            H_shannon=round(H_total, 1),
            H2_W=round(H2_W, 1),
            H2_syndrome=round(H2_synd, 1),
            H2_residual_given_sketch=round(H2_resid, 1),
            distinct_syndromes=int(len(synd_counts)),
            top_quantity_bits={sysp["features"][i]: round(float(ent[i]), 1) for i in hi_ent[:8]},
            n_quantities_above_1bit=int((ent > 1).sum()),
            structured_free_box=enum,
            structured_survivors=survivors,
            true_reading_survives=true_found,
            est_search_bits=round(free_cost, 1),
        )
    OUT.mkdir(exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    _write_report(results, sysp)
    print(json.dumps(results, indent=1))


def _structured_count(M, supports, probs, ent, synd_pub, Wtrue, rng, max_enum=3_000_000, mc=300_000):
    """The attacker guesses the few entropy-carrying symbols and lets the 4 syndrome equations pin
    four others. Deterministic symbols (support == 1) are fixed. 'Free' = entropy symbols except
    four solved. Returns (plausible_survivors, true_W_found, free_box_size): survivors = plausible
    readings consistent with the public syndrome. If the free box is too large, survivors is a
    Monte-Carlo estimate (sampling free symbols from their population distribution)."""
    n = len(supports)
    order = np.argsort(ent)[::-1]
    solve = _pick_solvable(M, [int(c) for c in order[::-1]], n)       # 4 symbols pinned by syndrome
    # the attacker guesses only the real entropy carriers (> 0.3 bit); the rest, nearly
    # deterministic over the population, are fixed at their most likely bin.
    free = [i for i in range(n) if ent[i] > 0.3 and i not in solve]
    fixed = {i: int(supports[i][int(np.argmax(probs[i]))]) for i in range(n) if i not in solve and i not in free}
    inv = np.array(_mat_inv4([[row[i] for i in solve] for row in M]), dtype=object)
    if inv is None:
        return None, None, 0
    support_sets = [set(int(v) % P for v in supports[i]) for i in range(n)]
    box = 1.0
    for i in free:
        box *= len(supports[i])

    Marr = np.array(M, dtype=object)
    base = np.array([synd_pub[r] - sum(Marr[r][i] * v for i, v in fixed.items()) for r in range(4)], dtype=object)
    # vectorised Monte-Carlo over the free symbols
    mc = min(mc, 120_000)
    cols = {i: (np.array(supports[i]) % P) for i in free}
    hits = 0
    true_found = False
    Wtrue_mod = [int(v) % P for v in Wtrue]
    for _ in range(mc):
        rhs = base.copy()
        chosen = {}
        for i in free:
            v = int(rng.choice(cols[i], p=probs[i]))
            chosen[i] = v
            for r in range(4):
                rhs[r] -= Marr[r][i] * v
        solved = [int(sum(inv[a][b] * (rhs[b] % P) for b in range(4)) % P) for a in range(4)]
        if all(solved[a] in support_sets[solve[a]] for a in range(4)):
            hits += 1
            if not true_found:
                W = [0] * n
                for i, v in chosen.items():
                    W[i] = v
                for i, v in fixed.items():
                    W[i] = v % P
                for a, i in enumerate(solve):
                    W[i] = solved[a]
                if W == Wtrue_mod:
                    true_found = True
    est = hits / mc * box
    return float(est), true_found, float(box)


def _pick_solvable(M, low_first, n):
    chosen = []
    cols = []
    for i in low_first:
        col = [row[i] for row in M]
        test = cols + [col]
        if _rank(test) == len(test):
            chosen.append(i)
            cols = test
            if len(chosen) == 4:
                break
    return chosen


def _rank(rows):
    Mx = [[v % P for v in r] for r in rows]
    r = 0
    ncol = len(Mx[0])
    for c in range(ncol):
        piv = next((i for i in range(r, len(Mx)) if Mx[i][c] % P), None)
        if piv is None:
            continue
        Mx[r], Mx[piv] = Mx[piv], Mx[r]
        iv = _inv(Mx[r][c])
        Mx[r] = [(v * iv) % P for v in Mx[r]]
        for i in range(len(Mx)):
            if i != r and Mx[i][c] % P:
                f = Mx[i][c]
                Mx[i] = [(a - f * b) % P for a, b in zip(Mx[i], Mx[r])]
        r += 1
        if r == len(Mx):
            break
    return r


def _mat_inv4(A):
    n = len(A)
    M = [[A[i][j] % P for j in range(n)] + [1 if i == j else 0 for j in range(n)] for i in range(n)]
    for c in range(n):
        piv = next((i for i in range(c, n) if M[i][c] % P), None)
        if piv is None:
            return None
        M[c], M[piv] = M[piv], M[c]
        iv = _inv(M[c][c])
        M[c] = [(v * iv) % P for v in M[c]]
        for i in range(n):
            if i != c and M[i][c] % P:
                f = M[i][c]
                M[i] = [(a - f * b) % P for a, b in zip(M[i], M[c])]
    return [row[n:] for row in M]


class _DetRNG:
    """Deterministic stand-in for secrets in gen(): the codeword is random but we only need the
    syndrome of W, which does not depend on C, so any codeword works."""
    def randbelow(self, n):
        return 12345 % n


def _write_report(results, sysp):
    lines = ["# Secure-sketch leakage: can a memory reader recover the reading offline?",
             "",
             "Generated by `python -m analysis.sketch_attack` (defensive analysis of our own scheme).",
             "Attacker knows the target's public helper data (offsets + sketch) and models the",
             "population of readings from the other three drones (leave-one-out).", ""]
    lines.append("| Drone | H(W) Shannon | H2(W) | H2(syndrome) | residual H2(W\\|sketch) | >1-bit quantities | survivors of the syndrome | true W survives |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for d, r in results.items():
        surv = r["structured_survivors"]
        surv_s = "n/a (enum too large)" if surv is None else str(surv)
        lines.append(f"| {d} | {r['H_shannon']} | {r['H2_W']} | {r['H2_syndrome']} | "
                     f"{r['H2_residual_given_sketch']} | {r['n_quantities_above_1bit']} | {surv_s} | "
                     f"{r['true_reading_survives']} |")
    lines += ["",
              "`residual H2(W|sketch)` is the collision entropy of the reading that remains after the",
              "four public syndrome equations are known: it is the real offline-guessing strength",
              "against a memory reader. `survivors of the syndrome` counts the readings that are both",
              "syndrome-consistent and population-plausible in the structured enumeration.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    t0 = time.time()
    analyse()
    print(f"\n({time.time() - t0:.0f} s)")
