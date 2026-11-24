"""Robust fuzzy extractor (keyproto/fe.py): Reed-Solomon, Gen/Rep, robustness, refresh."""

import random

import numpy as np
import pytest

from keyproto.fe import FuzzyExtractor, rs_encode, rs_decode, P, quantise


@pytest.mark.parametrize("n,t", [(12, 0), (12, 2), (40, 2), (40, 4)])
def test_rs_corrects_up_to_t(n, t):
    rng = random.Random(1)
    k = n - 2 * t
    for _ in range(30):
        msg = [rng.randrange(P) for _ in range(k)]
        cw = rs_encode(msg, n)
        r = cw[:]
        for i in rng.sample(range(n), t):
            r[i] = (r[i] + rng.randrange(1, P)) % P
        assert rs_decode(r, k, t) == msg


def test_rs_rejects_more_than_t():
    rng = random.Random(2)
    n, t = 40, 2
    k = n - 2 * t
    fails = 0
    for _ in range(30):
        msg = [rng.randrange(P) for _ in range(k)]
        r = rs_encode(msg, n)
        for i in rng.sample(range(n), t + 1):
            r[i] = (r[i] + rng.randrange(1, P)) % P
        fails += rs_decode(r, k, t) != msg
    assert fails == 30


@pytest.fixture
def fe_ref():
    rng = np.random.default_rng(0)
    w = np.full(40, 10.0)
    ref = rng.normal(0, 100, 40)
    fe = FuzzyExtractor(w, 2)
    key, helper, idx = fe.gen(ref)
    return fe, ref, key, helper, w


def test_rep_same_key_with_noise(fe_ref):
    fe, ref, key, helper, w = fe_ref
    noisy = ref + np.random.default_rng(1).normal(0, 0.5, 40)    # << w/2: no bin change
    k2, _ = fe.rep(noisy, helper)
    assert k2 == key


def test_rep_corrects_two_wrong_symbols(fe_ref):
    fe, ref, key, helper, w = fe_ref
    x = ref.copy()
    x[[3, 17]] += 3 * w[[3, 17]]
    assert fe.rep(x, helper)[0] == key


def test_rep_rejects_three_wrong_symbols(fe_ref):
    fe, ref, key, helper, w = fe_ref
    x = ref.copy()
    x[[1, 2, 3]] += 3 * w[[1, 2, 3]]
    k, reason = fe.rep(x, helper)
    assert k is None and "too many" in reason


def test_missing_values_count_as_wrong(fe_ref):
    fe, ref, key, helper, w = fe_ref
    x = ref.copy()
    x[[5, 6]] = np.nan
    assert fe.rep(x, helper)[0] == key
    x[7] = np.nan
    assert fe.rep(x, helper)[0] is None


def test_tampered_offset_rejected_by_tag(fe_ref):
    """A one-bin change the code would silently correct must still be caught."""
    fe, ref, key, helper, w = fe_ref
    h = dict(helper)
    off = list(h["offsets"])
    off[0] += w[0]
    h["offsets"] = off
    k, reason = fe.rep(ref, h)
    assert k is None and "tag" in reason


def test_non_robust_mode_no_tag_fingerprint_catches_tampering():
    """robust=False (scheme of the thesis since 2026-10-29): no tag in the
    helper, so the helper alone gives no offline check of a guess; a changed
    offset still decodes, but the public fingerprint changes."""
    rng = np.random.default_rng(3)
    w = np.full(40, 10.0)
    ref = rng.normal(0, 100, 40)
    fe = FuzzyExtractor(w, 2, robust=False)
    key, helper, _ = fe.gen(ref)
    assert "tag" not in helper
    fp = FuzzyExtractor.fingerprint(helper)
    h = dict(helper, offsets=[helper["offsets"][0] + 0.1] + helper["offsets"][1:])
    assert fe.rep(ref, h)[0] == key and FuzzyExtractor.fingerprint(h) != fp
    k, W = fe.rep(ref, helper)
    new = fe.refresh_onboard(helper, W, ref + 1.0)
    assert "tag" not in new and FuzzyExtractor.fingerprint(new) != fp
    assert FuzzyExtractor.fingerprint(dict(helper, enrolled_from="x", tag="old")) == fp


def test_tampered_sketch_rejected(fe_ref):
    fe, ref, key, helper, w = fe_ref
    h = dict(helper)
    s = list(h["sketch"])
    s[0] = (s[0] + 1) % P
    h["sketch"] = s
    assert fe.rep(ref, h)[0] is None


def test_informational_field_not_covered(fe_ref):
    fe, ref, key, helper, w = fe_ref
    h = dict(helper, enrolled_from="anything")
    assert fe.rep(ref, h)[0] == key


def test_refresh_tracks_drift_keeps_key(fe_ref):
    fe, ref, key, helper, w = fe_ref
    drift = 0.4 * w                         # would cross bins without refresh after a few steps
    h, x = helper, ref
    for _ in range(4):
        x = x + drift
        k, W = fe.rep(x, h)
        assert k == key
        h = fe.refresh(h, W, [x])
    assert fe.rep(ref + 4 * drift, helper)[0] is None   # the original helper alone fails


def test_refresh_onboard_keeps_key(fe_ref):
    """Median of the last 3 boots lags one boot behind: it follows drift
    slower than w/4 per boot (steady deviation 2*step < w/2)."""
    fe, ref, key, helper, w = fe_ref
    h, x = helper, ref
    for _ in range(8):
        x = x + 0.2 * w
        k, W = fe.rep(x, h)
        assert k == key
        h = fe.refresh_onboard(h, W, x)
        assert len(h["history"]) <= 3
    assert fe.rep(x + 0.2 * w, helper)[0] is None        # the original helper alone fails


def test_refresh_onboard_ignores_one_outlier_boot(fe_ref):
    """A single odd (but accepted) boot must not move the offsets."""
    fe, ref, key, helper, w = fe_ref
    h = helper
    for x in (ref, ref, ref + 0.45 * w, ref):
        k, W = fe.rep(x, h)
        assert k == key
        h = fe.refresh_onboard(h, W, x)
    assert np.allclose(h["offsets"], fe.refresh_onboard(helper, W, ref)["offsets"])


def test_different_reference_gives_no_key(fe_ref):
    fe, ref, key, helper, w = fe_ref
    other = ref + np.random.default_rng(5).normal(0, 300, 40)
    assert fe.rep(other, helper)[0] is None


def test_quantise_handles_nan():
    assert quantise([1.0, np.nan], [0.0, 0.0], [1.0, 1.0]) == [1, None]


# ── speed alarm (tampering with motors / propellers) ───────────────────────

def test_shift_bins_measures_from_enrolled_centre(fe_ref):
    fe, ref, key, helper, w = fe_ref
    k, W = fe.rep(ref, helper)
    assert np.allclose(fe.shift_bins(helper, W, ref), 0, atol=1e-9)
    x = ref.copy()
    x[4] += 1.3 * w[4]                       # out of its bin: symbol corrected, shift still 1.3
    x[9] -= 0.3 * w[9]
    x[11] = np.nan
    k, W = fe.rep(x, helper)
    assert k == key
    s = fe.shift_bins(helper, W, x)
    assert s[4] == pytest.approx(1.3) and s[9] == pytest.approx(-0.3) and np.isnan(s[11])


def test_speed_alarm_threshold():
    from keyproto.fe import speed_alarm, SPEED_ALARM_SIGMA, gain_index
    z = {1: SPEED_ALARM_SIGMA + 0.1, 2: -SPEED_ALARM_SIGMA - 0.1, 3: SPEED_ALARM_SIGMA - 0.1, 4: np.nan}
    assert speed_alarm(z) == [1, 2]
    names = ["m1_58000_gain", "m1_58000_floor_acc.x", "m2_58000_gain", "m3_58000_gain", "m4_58000_gain", "dX"]
    assert gain_index(names) == {1: 0, 2: 2, 3: 3, 4: 4}
