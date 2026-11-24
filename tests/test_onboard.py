"""On-board key protocol: Python side (keyproto/onboard.py) and, through WSL,
byte-exact agreement with the C core of the firmware (firmware/uavibe_key)."""

import glob
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from keyproto import onboard as ob
from keyproto.fe import P
from tests.conftest import ROOT, has_data, first_day_boots

REPO = ROOT.parent
TEST_DIR = REPO / "firmware" / "uavibe_key" / "test"

SYSP = dict(features=[f"m{i // 7 + 1}_58000_gain" if i < 28 and i % 7 == 0 else f"f{i}" for i in range(40)], widths=list(np.linspace(0.05, 40, 40)),
            temp_slopes={}, t=2, K=8.0)


def _ref(seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(SYSP["widths"]) * rng.normal(0, 30, 40)


# ── Python only ─────────────────────────────────────────────────────────────

def test_gen_rep_refresh_python():
    ref = _ref()
    pw, h = ob.gen(ref, SYSP, "A")
    assert len(h) == 8 + 4 * 40 * 2 and h[:4] == b"UVH3"        # v3: no robustness tag
    assert ob.unpack_helper(h)["drone_id"] == ord("A")
    k, W, wrong = ob.rep(ref, h, SYSP)
    assert k == pw and wrong == 0
    w = np.asarray(SYSP["widths"])
    x = ref.copy(); x[[1, 30]] += 3 * w[[1, 30]]
    assert ob.rep(x, h, SYSP)[0] == pw
    x[2] += 3 * w[2]
    assert ob.rep(x, h, SYSP)[0] is None
    hh, xx = h, ref
    for _ in range(6):
        xx = xx + 0.2 * w
        k, W, _ = ob.rep(xx, hh, SYSP)
        assert k == pw
        hh = ob.refresh(hh, W, xx, SYSP)
    assert ob.unpack_helper(hh)["history"].shape == (3, 40)


def test_helper_fingerprint_binds_name_offsets_and_params():
    """Without a tag, Rep does not check the helper: the fingerprint does
    (station side). Any change to the helper, even one that still decodes to
    the same password, changes the fingerprint; so do other system parameters,
    which also change the password."""
    ref = _ref(1)
    pw, h = ob.gen(ref, SYSP, "A")
    fp = ob.helper_fp(h, SYSP)
    forged = bytearray(h); forged[7] = ord("B")               # same helper, other drone name
    assert ob.helper_fp(bytes(forged), SYSP) != fp
    nudged = bytearray(h); nudged[8:12] = np.float32(ob.unpack_helper(h)["offsets"][0] + 1e-3 * SYSP["widths"][0]).tobytes()
    assert ob.rep(ref, bytes(nudged), SYSP)[0] == pw          # still the same password...
    assert ob.helper_fp(bytes(nudged), SYSP) != fp            # ...but not the same helper
    other = dict(SYSP, widths=list(np.asarray(SYSP["widths"]) * 1.0001))
    assert ob.rep(ref, h, other)[0] != pw                     # other system parameters
    assert ob.helper_fp(h, other) != fp


def test_legacy_v2_helper_read_and_migrated():
    """A v2 helper (with its 32-byte tag, drones enrolled before 2026-10-29)
    is still read: same password, tag dropped, refreshed as v3."""
    ref = _ref(2)
    pw, h3 = ob.gen(ref, SYSP, "D")
    h2 = b"UVH2" + h3[4:] + bytes(range(32))                  # what the old firmware stored
    u = ob.unpack_helper(h2)
    assert u["legacy"] and u["body"] == h3
    assert ob.helper_fp(h2, SYSP) == ob.helper_fp(h3, SYSP)
    k, W, _ = ob.rep(ref, h2, SYSP)
    assert k == pw and ob.refresh(h2, W, ref, SYSP)[:4] == b"UVH3"
    with pytest.raises(ValueError):
        ob.unpack_helper(h2[:-1])


def test_speed_shift_and_data_frame_python():
    ref = _ref(5)
    pw, h = ob.gen(ref, SYSP, "C")
    w = np.asarray(SYSP["widths"])
    x = ref.copy()
    x[7] += 7 / SYSP["K"] * w[7]             # motor 2: +7 sigma (beyond the half bin: 1 wrong symbol)
    x[21] = np.nan                           # motor 4 not measured
    k, W, wrong = ob.rep(x, h, SYSP)
    assert k == pw and wrong == 2            # m2 moved bin, m4 missing
    sh = ob.speed_shift(h, W, x, SYSP)
    assert sh[0] == pytest.approx(0, abs=1e-3) and sh[1] == pytest.approx(7, abs=1e-3) and np.isnan(sh[3])
    assert ob.alarm_mask(sh) == 0b0010
    fp = ob.helper_fp(h, SYSP)
    d = ob.unpack_data(ob.pack_data(ord("C"), wrong, sh, 0b0010, False, fp, b"hi"))
    assert d["drone"] == "C" and d["wrong"] == 2 and d["alarm"] == [2] and d["refreshed"] is False
    assert d["shift_sigma"][2] == 7.0 and d["shift_sigma"][4] is None and d["text"] == "hi"
    assert d["helper_fp"] == fp


def test_cpace_python():
    sid, fp = os.urandom(32), os.urandom(32)
    d, s = ob.CPace("drone", b"p" * 32, sid, b"A", fp), ob.CPace("station", b"p" * 32, sid, b"A", fp)
    d.finish(s.Y); s.finish(d.Y)
    assert d.verify(s.confirm()) and s.verify(d.confirm()) and d.data_key() == s.data_key()
    assert d.commit(fp) == s.commit(fp) != s.commit(os.urandom(32))
    e = ob.CPace("station", b"q" * 32, sid, b"A", fp)
    d2 = ob.CPace("drone", b"p" * 32, sid, b"A", fp)
    d2.finish(e.Y); e.finish(d2.Y)
    assert not d2.verify(e.confirm()) and not e.verify(d2.confirm())
    e = ob.CPace("station", b"p" * 32, sid, b"B", fp)          # other drone name
    d3 = ob.CPace("drone", b"p" * 32, sid, b"A", fp)
    d3.finish(e.Y); e.finish(d3.Y)
    assert not e.verify(d3.confirm())
    e = ob.CPace("station", b"p" * 32, sid, b"A", fp)          # same password, other helper
    d4 = ob.CPace("drone", b"p" * 32, sid, b"A", os.urandom(32))
    d4.finish(e.Y); e.finish(d4.Y)
    assert not e.verify(d4.confirm())


def test_framing_python():
    r = ob.Reassembler()
    for n in (0, 1, 28, 29, 100, 420):
        data = os.urandom(n)
        pk = ob.fragment(0x86, data)
        assert all(len(p) <= ob.MTU for p in pk)
        out = [r.feed(p) for p in pk][-1]
        assert out == (0x86, data)


# ── C core, through WSL ─────────────────────────────────────────────────────

def _wsl_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    return f"/mnt/{s[0].lower()}{s[2:]}"


@pytest.fixture(scope="module")
def host():
    if shutil.which("wsl") is None:
        pytest.skip("WSL not available")
    b = subprocess.run(["wsl", "-e", "make", "-s", "-C", _wsl_path(TEST_DIR)], capture_output=True, text=True)
    if b.returncode != 0:
        pytest.skip(f"host build failed: {b.stderr[-300:]}")
    proc = subprocess.Popen(["wsl", "-e", _wsl_path(TEST_DIR / "host_test")], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, bufsize=1)

    def run(*args) -> str:
        proc.stdin.write(" ".join(args) + "\n")
        proc.stdin.flush()
        return proc.stdout.readline().strip()

    yield run
    proc.stdin.close()
    proc.wait(timeout=10)


hx = lambda b: b.hex() if b else "-"
f32hex = lambda x: np.asarray(x, dtype="<f4").tobytes().hex()


def test_c_elligator_matches(host):
    for _ in range(64):
        h = os.urandom(32)
        assert host("elligator", h.hex()) == ob.elligator_map(h).hex()


def test_c_cpace_matches(host):
    for drone_pw_ok in (True, False):
        prs, sid, fp = os.urandom(32), os.urandom(32), os.urandom(32)
        s = ob.CPace("station", prs if drone_pw_ok else os.urandom(32), sid, b"C", fp)
        sc = os.urandom(32)
        d = ob.CPace("drone", prs, sid, b"C", fp, scalar=sc)
        Y, isk, cd, cs, dk, cm = host("cpace", prs.hex(), sid.hex(), b"C".hex(), fp.hex(), sc.hex(),
                                      s.Y.hex()).split()
        d.finish(s.Y); s.finish(d.Y)
        assert Y == d.Y.hex() and isk == d.isk.hex()
        assert cd == d.confirm().hex() and cs == d.confirm("station").hex() and dk == d.data_key().hex()
        assert cm == d.commit(fp).hex()
        assert s.verify(bytes.fromhex(cd)) is drone_pw_ok


def test_c_rejects_low_order_point(host):
    out = host("cpace", "00" * 32, "00" * 32, "41", "22" * 32, "11" * 32, "00" * 32)
    assert out.endswith("lowpoint")


def test_c_aead_matches(host):
    key, nonce = os.urandom(32), os.urandom(24)
    for n in (0, 5, 64):
        pt, ad = os.urandom(n), b"A"
        frame = bytes.fromhex(host("aead", key.hex(), nonce.hex(), ad.hex(), hx(pt)))
        assert frame == ob.aead_lock(key, pt, ad, nonce)
        assert ob.aead_unlock(key, frame, ad) == pt


def test_c_framing_matches(host):
    for n in (0, 28, 70):
        data = os.urandom(n)
        pk = ob.fragment(0x83, data)
        assert host("frag", "83", hx(data)).split() == [p.hex() for p in pk]
        assert host("reasm", *[p.hex() for p in pk]) == f"83 {hx(data)}"


def test_c_rep_matches_python(host):
    # the C core is compiled for the real system parameters: use them
    from keyproto import boot as kb
    if not (ROOT / "data" / "keystore" / "system.json").exists():
        pytest.skip("needs data/keystore/system.json (the C header is generated from it)")
    sysp = kb.system_params()
    rng = np.random.default_rng(3)
    w = np.asarray(sysp["widths"])
    for trial in range(40):
        ref = w * rng.normal(0, 40, 40)
        pw, h = ob.gen(ref, sysp, "D")
        x = ref + rng.normal(0, 0.05, 40) * w
        bad = rng.choice(40, trial % 4, replace=False)          # 0..3 wrong symbols
        x[bad] += 3 * w[bad]
        out = host("rep", h.hex(), f32hex(x)).split()
        k, W, wrong = ob.rep(x, h, sysp)
        if k is None:
            assert out[0] == "err"
        else:
            assert out[0] == "ok" and out[1] == k.hex() and int(out[3]) == wrong
            assert np.frombuffer(bytes.fromhex(out[2]), "<u4").tolist() == [int(v) for v in W]
            nh = host("refresh", h.hex(), np.asarray(W, "<u4").tobytes().hex(), f32hex(x))
            assert nh == ob.refresh(h, W, x, sysp).hex()
    # helper fingerprint: v3, a tampered helper, and a v2 helper with its tag
    tampered = bytearray(h); tampered[10] ^= 1                 # one offset bit
    for blob in (h, bytes(tampered), b"UVH2" + h[4:] + os.urandom(32)):
        fp, v3 = host("fp", blob.hex()).split()
        assert fp == ob.helper_fp(blob, sysp).hex() and v3 == ob.unpack_helper(blob)["body"].hex()
    assert host("fp", bytes(tampered).hex()).split()[0] != host("fp", h.hex()).split()[0]
    assert host("rep", (h + b"\0").hex(), f32hex(ref)).split()[0] == "err"     # wrong length


def test_c_refresh_chain_matches(host):
    from keyproto import boot as kb
    if not (ROOT / "data" / "keystore" / "system.json").exists():
        pytest.skip("needs data/keystore/system.json")
    sysp = kb.system_params()
    w = np.asarray(sysp["widths"])
    ref = w * np.random.default_rng(4).normal(0, 40, 40)
    pw, h = ob.gen(ref, sysp, "B")
    x = ref.copy()
    for i in range(6):
        x = x + 0.2 * w
        if i == 3:
            x[5] = np.nan
        out = host("rep", h.hex(), f32hex(x)).split()
        k, W, _ = ob.rep(x, h, sysp)
        assert out[1] == k.hex() == pw.hex()
        hc = host("refresh", h.hex(), np.asarray(W, "<u4").tobytes().hex(), f32hex(x))
        h = ob.refresh(h, W, x, sysp)
        assert hc == h.hex()


def test_c_speed_shift_and_data_frame_match(host):
    from keyproto import boot as kb
    if not (ROOT / "data" / "keystore" / "system.json").exists():
        pytest.skip("needs data/keystore/system.json")
    sysp = kb.system_params()
    w = np.asarray(sysp["widths"])
    gi = [0, 7, 14, 21]
    rng = np.random.default_rng(6)
    alarms = 0
    for trial in range(40):
        ref = w * rng.normal(0, 40, 40)
        pw, h = ob.gen(ref, sysp, "A")
        x = ref + rng.normal(0, 0.05, 40) * w
        m = gi[trial % 4]
        x[m] += rng.uniform(-12, 12) / sysp["K"] * w[m]       # one motor moved by up to 12 sigma
        if trial % 7 == 0:
            x[gi[(trial + 1) % 4]] = np.nan
        k, W, wrong = ob.rep(x, h, sysp)
        if trial % 5 == 0:                                     # refreshed helper (history in play)
            h = ob.refresh(h, W, x, sysp)
            k, W, wrong = ob.rep(x, h, sysp)
        sh = ob.speed_shift(h, W, x, sysp)
        mask = ob.alarm_mask(sh)
        alarms += bool(mask)
        expect = ob.pack_data(ord("A"), 2, sh, mask, not mask, ob.helper_fp(h, sysp), b"t")
        assert host("shift", h.hex(), np.asarray(W, "<u4").tobytes().hex(), f32hex(x)) == expect.hex()
    assert 0 < alarms < 40


def test_c_station_signature_matches_pynacl(host):
    seed, pk = ob.station_keypair(os.urandom(32))
    for mtype, payload in ((0x02, os.urandom(16)), (0x0A, b""), (0x05, os.urandom(328))):
        ch = os.urandom(16)
        sig = ob.sign_cmd(seed, mtype, ch, payload)
        assert ob.verify_cmd(pk, mtype, ch, payload, sig)
        assert host("sigcheck", pk.hex(), f"{mtype:02x}", ch.hex(), hx(payload), sig.hex()) == "ok"
        bad = bytes([sig[0] ^ 1]) + sig[1:]
        assert host("sigcheck", pk.hex(), f"{mtype:02x}", ch.hex(), hx(payload), bad.hex()) == "bad"
        assert host("sigcheck", pk.hex(), f"{mtype:02x}", os.urandom(16).hex(), hx(payload), sig.hex()) == "bad"
        assert host("sigcheck", pk.hex(), f"{mtype ^ 1:02x}", ch.hex(), hx(payload), sig.hex()) == "bad"


@pytest.mark.skipif(not has_data("fixed_rig_boot"), reason="needs recorded boot sessions")
def test_c_features_match_reference_on_real_boots(host, tmp_path):
    from keyproto import boot as kb
    sysp = kb.system_params()
    names = sysp["features"]
    sig = np.asarray(sysp["widths"]) / sysp["K"]
    for s in first_day_boots()[::3]:
        inp = ob.boot_inputs(s)
        f = tmp_path / f"{Path(s).name}.bin"
        f.write_bytes(ob.pack_inputs(inp))
        xc = np.array([float(v) for v in host("features", _wsl_path(f)).split()])
        xp = ob.onboard_features(inp, sysp)
        assert np.array_equal(np.isnan(xc), np.isnan(xp)), s
        ok = ~np.isnan(xp)
        dev = np.abs(xc[ok] - xp[ok]) / sig[ok]
        assert dev.max() < 0.01, (s, [names[i] for i in np.where(ok)[0][dev > 0.01]])
