"""
Robust fuzzy extractor for vector-valued physical measurements.

Pipeline (Gen at enrollment, Rep at every boot):

  1. Quantiser (shielding functions, Linnartz & Tuyls 2003). Fleet-wide
     public bin widths w_i. At Gen the reference value of each feature is
     put at the centre of its bin by a public offset; the bin indices form
     the secret symbol vector W (integers). Drift: after a successful boot the
     offsets are recomputed from new measurements while KEEPING the indices,
     so W never changes (`refresh`).
  2. Secure sketch, code-offset construction (Dodis, Reyzin, Smith 2004)
     with a Reed-Solomon code RS[n, n-2t] over GF(P), P prime: at Gen a
     uniformly random codeword C is drawn and SS = (W - C) mod P is public.
     At Rep, C' = (W' - SS) mod P = C + E; Berlekamp-Welch decoding corrects
     up to t wrong symbols, giving W = C + SS.
  3. Robustness (Boyen et al. 2005, random-oracle construction), only with
     robust=True (the default, used by the analyses of reports 9-11): the
     public helper carries tag = H("tag" | W | helper), checked at Rep; any
     change to the helper data (offsets, sketch, parameters) is rejected
     instead of producing a related key. The tag is, however, an offline
     oracle for whoever reads the helper (guesses of W can be checked against
     it without the station), so the scheme of the thesis (on board since
     2026-10-29, and the identity tab, keyproto/boot.py) uses robust=False and
     lets the station check the helper through `fingerprint`, which does not
     depend on W.
  4. Key: R = HKDF(W, salt = H(helper-id)).

Entropy note: R can never have more entropy than W minus what the sketch
leaks (worst case (n-k)*log2(P) bits; see the report). With the low entropy
of these sources R must be used as a PAKE password, never as a key directly.
"""

import hashlib
import hmac
import json
import secrets

import numpy as np

P = 65537          # prime field for the RS code (symbols = bin indices mod P)

# Speed alarm (tampering with motors / propellers, 2026-10-14). The key
# tolerates t wrong symbols, and one motor's speed is a single symbol, so a
# propeller change (-8...-11 % speed, 9-12 within-drone sigma) still
# regenerates the key. Separate check after a successful Rep: any motor whose
# speed gain moved more than SPEED_ALARM_SIGMA from the helper's reference
# raises an alarm and access is DENIED (threat model agreed with the
# professor, 2026-10-15: altered mechanics = not the enrolled drone). The
# key was regenerated, so the denial is authenticated (the station knows it
# is that drone, tampered) and it is not counted as a guessing failure; the
# reference is NOT refreshed, so the alarm repeats until re-enrollment
# (legitimate maintenance = re-enrollment). Slow ageing is followed by the
# refresh (median of the last 3 accepted boots), so it never accumulates
# into an alarm. Normal boots on the 22 + 4 recorded ones moved at most 3.1
# sigma (on-board features 2.6).
SPEED_ALARM_SIGMA = 5.0


def gain_index(features) -> "dict[int, int]":
    """motor (1-4) -> position of its speed-gain feature."""
    return {int(n[1]): i for i, n in enumerate(features) if n.startswith("m") and n.endswith("_gain")}


def speed_alarm(shift_sigma: "dict[int, float]") -> "list[int]":
    """Motors whose speed moved past the alarm threshold (NaN = not measured)."""
    return [m for m, z in sorted(shift_sigma.items()) if np.isfinite(z) and abs(z) > SPEED_ALARM_SIGMA]


# ── GF(P) Reed-Solomon (evaluation form) + Berlekamp-Welch decoder ──────────

def _inv(a: int) -> int:
    return pow(a % P, P - 2, P)


def _poly_eval(coeffs, x):
    r = 0
    for c in reversed(coeffs):
        r = (r * x + c) % P
    return r


def rs_encode(msg, n):
    """msg: k coefficients -> codeword = evaluations at x = 1..n."""
    return [_poly_eval(msg, x) for x in range(1, n + 1)]


def _solve_mod(A, b):
    """Gaussian elimination mod P; returns one solution or None."""
    m, ncol = len(A), len(A[0])
    M = [row[:] + [bb] for row, bb in zip(A, b)]
    piv_cols, r = [], 0
    for c in range(ncol):
        p = next((i for i in range(r, m) if M[i][c] % P), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        iv = _inv(M[r][c])
        M[r] = [(v * iv) % P for v in M[r]]
        for i in range(m):
            if i != r and M[i][c] % P:
                f = M[i][c]
                M[i] = [(a - f * bq) % P for a, bq in zip(M[i], M[r])]
        piv_cols.append(c)
        r += 1
        if r == m:
            break
    for i in range(r, m):
        if M[i][-1] % P:
            return None
    x = [0] * ncol
    for i, c in enumerate(piv_cols):
        x[c] = M[i][-1]
    return x


def _poly_div(num, den):
    num = num[:]
    out = [0] * (len(num) - len(den) + 1)
    iv = _inv(den[-1])
    for i in range(len(out) - 1, -1, -1):
        q = (num[i + len(den) - 1] * iv) % P
        out[i] = q
        for j, d in enumerate(den):
            num[i + j] = (num[i + j] - q * d) % P
    return out, num[:len(den) - 1]


def rs_decode(received, k, t):
    """Berlekamp-Welch: find E (deg t, monic) and Q (deg < k+t) with
    Q(x_i) = y_i E(x_i). Returns the k message coefficients or None."""
    n = len(received)
    xs = list(range(1, n + 1))
    A, b = [], []
    for x, y in zip(xs, received):
        row = [(pow(x, j, P)) for j in range(k + t)]            # Q coefficients
        row += [(-y * pow(x, j, P)) % P for j in range(t)]      # E coeffs (except monic)
        A.append(row)
        b.append((y * pow(x, t, P)) % P)
    sol = _solve_mod(A, b)
    if sol is None:
        return None
    Q = sol[:k + t]
    E = sol[k + t:] + [1]
    msg, rem = _poly_div(Q, E)
    if any(r % P for r in rem):
        return None
    msg = (msg + [0] * k)[:k]
    cw = rs_encode(msg, n)
    if sum(c != y for c, y in zip(cw, received)) > t:
        return None
    return msg


# ── helper data ─────────────────────────────────────────────────────────────

def _h(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(len(p).to_bytes(4, "big") + p)
    return h.digest()


def _enc(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _hkdf(ikm: bytes, salt: bytes, info: bytes, n: int = 32) -> bytes:
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    out, t, i = b"", b"", 1
    while len(out) < n:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        out += t
        i += 1
    return out[:n]


def quantise(x, offsets, widths):
    x, o, w = (np.asarray(v, dtype=float) for v in (x, offsets, widths))
    q = np.floor((x + o) / w)
    return [int(v) if np.isfinite(v) else None for v in q]


class FuzzyExtractor:
    """widths: fleet-wide public bin widths (one per feature); t: correctable
    wrong symbols; robust: robustness tag in the helper (see the module
    docstring; False = no tag, the station checks `fingerprint` instead)."""

    def __init__(self, widths, t: int, robust: bool = True):
        self.w = np.asarray(widths, dtype=float)
        self.n = len(self.w)
        self.t = t
        self.k = self.n - 2 * t
        self.robust = robust
        assert self.k > 0

    # Gen ------------------------------------------------------------------
    def gen(self, reference):
        ref = np.asarray(reference, dtype=float)
        idx = np.floor(ref / self.w)
        offsets = (idx + 0.5) * self.w - ref
        W = [int(v) % P for v in idx]
        C = rs_encode([secrets.randbelow(P) for _ in range(self.k)], self.n)
        sketch = [(a - c) % P for a, c in zip(W, C)]
        helper = dict(v=1, n=self.n, t=self.t, P=P, widths=[float(v) for v in self.w],
                      offsets=[float(v) for v in offsets], sketch=sketch)
        if self.robust:
            helper["tag"] = self._tag(W, helper)
        return self._key(W, helper), helper, [int(v) for v in idx]

    # Rep ------------------------------------------------------------------
    def rep(self, measurement, helper):
        """-> (key, W) or (None, reason)."""
        if helper.get("n") != self.n or helper.get("t") != self.t or helper.get("P") != P:
            return None, "helper parameters altered"
        q = quantise(measurement, helper["offsets"], helper["widths"])
        Wp = [(v if v is not None else 0) % P for v in q]
        Cp = [(a - s) % P for a, s in zip(Wp, helper["sketch"])]
        msg = rs_decode(Cp, self.k, self.t)
        if msg is None:
            return None, "too many wrong symbols"
        C = rs_encode(msg, self.n)
        W = [(c + s) % P for c, s in zip(C, helper["sketch"])]
        if self.robust and not hmac.compare_digest(self._tag(W, helper), str(helper.get("tag", ""))):
            return None, "robustness tag mismatch (helper data tampered)"
        return self._key(W, helper), W

    # drift tracking --------------------------------------------------------
    def refresh(self, helper, W, recent_measurements):
        """New offsets centring the median of recent accepted measurements on
        the SAME secret bins; sketch unchanged; tag recomputed (requires W,
        i.e. only possible after a successful reconstruction)."""
        ref = np.nanmedian(np.asarray(recent_measurements, dtype=float), axis=0)
        idx = self._unwrap(W, ref)
        new = dict(helper)
        new["offsets"] = [float(v) for v in (np.asarray(idx) + 0.5) * self.w - ref]
        return self._retag(W, new)

    def refresh_onboard(self, helper, W, measurement, keep: int = 3):
        """Drone-side refresh without storing measurements (which would leak
        W): each accepted boot is stored only as its own centring offsets
        (idx + 0.5) * w - x, the same information the helper already
        publishes; the new offsets are the median over the last `keep`.
        Equivalent to centring on the median of the last measurements."""
        x = np.asarray(measurement, dtype=float)
        idx = np.asarray(self._unwrap(W, x), dtype=float)
        own = (idx + 0.5) * self.w - x
        hist = [list(h) for h in helper.get("history", [])] + [[float(v) for v in own]]
        hist = hist[-keep:]
        new = dict(helper)
        new["history"] = hist
        new["offsets"] = [float(v) for v in np.nanmedian(np.asarray(hist, dtype=float), axis=0)]
        return self._retag(W, new)

    def shift_bins(self, helper, W, measurement) -> np.ndarray:
        """Distance of each measured value from the helper's reference (the
        centre of its enrolled bin, i.e. the median of the last refreshes), in
        bin widths, signed (NaN where missing). W is the CORRECTED symbol
        vector from rep, so a value that left its bin is still measured from
        the enrolled centre. Equals (offset - own centring offset) / w."""
        x = np.asarray(measurement, dtype=float)
        idx = np.asarray(self._unwrap(W, np.nan_to_num(x)), dtype=float)
        own = (idx + 0.5) * self.w - x
        return (np.asarray(helper["offsets"], dtype=float) - own) / self.w

    # internals -------------------------------------------------------------
    def _unwrap(self, W, ref):
        """True bin index (not mod P) closest to the reference."""
        approx = np.floor(ref / self.w)
        return [int(a + ((w - a) % P if (w - a) % P < P // 2 else (w - a) % P - P))
                for w, a in zip(W, approx)]

    def _tag(self, W, helper) -> str:
        h = {k: v for k, v in helper.items() if k not in ("tag", "enrolled_from")}
        return _h(b"uavibe-tag", _enc(W), _enc(h)).hex()

    def _retag(self, W, helper) -> dict:
        helper.pop("tag", None)
        if self.robust:
            helper["tag"] = self._tag(W, helper)
        return helper

    @staticmethod
    def fingerprint(helper) -> str:
        """Public fingerprint of a helper (tag and the informative
        `enrolled_from` excluded), kept by the station when robust=False.
        It does not depend on W, so it is no oracle."""
        h = {k: v for k, v in helper.items() if k not in ("tag", "enrolled_from")}
        return _h(b"uavibe-helper", _enc(h)).hex()

    def _key(self, W, helper) -> bytes:
        # salt binds the key to the immutable part of the helper (sketch +
        # parameters), not to the offsets, which change at every refresh
        salt = _h(b"uavibe-salt", _enc(helper["sketch"]), _enc([helper["n"], helper["t"], helper["P"]]))
        return _hkdf(_enc(W), salt, b"uavibe-pake-password")
