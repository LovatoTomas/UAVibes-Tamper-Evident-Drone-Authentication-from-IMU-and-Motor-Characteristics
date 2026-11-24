"""Station protocol (keyproto/onboard_station.py) end to end against the
simulated drone, fed with the on-board features of the recorded boots."""

import glob

import numpy as np
import pytest

from keyproto import onboard as ob
from keyproto.onboard_station import Station, SimDrone, MAX_FAILURES, station_key
from tests.conftest import ROOT, has_data, first_day_boots

pytestmark = pytest.mark.skipif(not has_data("fixed_rig_boot") or not has_data("keystore", "system.json"),
                                reason="needs recorded boot sessions and data/keystore/system.json")


@pytest.fixture(scope="module")
def feats():
    from keyproto import boot as kb
    sysp = kb.system_params()
    out = {}
    for s in first_day_boots():
        inp = ob.boot_inputs(s)
        out.setdefault(inp["drone"], []).append(ob.onboard_features(inp, sysp))
    return sysp, out


def enrolled(tmp_path, sysp, boots, drone):
    sim = SimDrone(sysp, boots, station_key(tmp_path, create=True)[1])
    st = Station(sim, sysp, tmp_path)
    assert st.hello()["enrolled"] is False
    for i in range(3):
        r = st.run(drone)
        assert r["mode"] == "enroll" and r["n_boots"] == i + 1
    assert st.finish_enrollment(drone)["ok"]
    h = st.hello()
    assert h["enrolled"] and h["drone"] == drone
    return sim, st


def test_enroll_then_authenticate(tmp_path, feats):
    sysp, F = feats
    sim, st = enrolled(tmp_path, sysp, F["C"], "C")
    for _ in range(len(F["C"]) - 3):
        r = st.run("C")
        assert r["ok"], r
        assert "drone C" in r["message"]
    assert "password" in st.db()["C"] and "features" not in st.db()["C"]


def test_other_drone_measurement_rejected(tmp_path, feats):
    sysp, F = feats
    sim, st = enrolled(tmp_path, sysp, F["C"][:3] + [F["D"][0], F["A"][0]], "C")
    for _ in range(2):                       # C's helper, another drone's measurement
        r = st.run("C")
        assert not r["ok"]
    assert st.db()["C"]["failures"] == 2


def test_wrong_name_fails_confirmation(tmp_path, feats):
    sysp, F = feats
    sim, st = enrolled(tmp_path, sysp, F["C"] + F["C"], "C")
    st2 = Station(sim, sysp, tmp_path)
    db = st2.db()
    db["A"] = dict(password=db["C"]["password"], failures=0,   # even with the right password
                   helper_fp=db["C"]["helper_fp"])             # and the right helper fingerprint
    import json
    (tmp_path / "station.json").write_text(json.dumps(db))
    r = st2.run("A")                                           # station believes it is A
    assert not r["ok"] and "confirmation" in r["reason"]


def test_helper_only_once_and_lockout(tmp_path, feats):
    sysp, F = feats
    sim, st = enrolled(tmp_path, sysp, F["C"][:3] + [F["D"][0]] * MAX_FAILURES + [F["C"][3]], "C")
    st.enroll_boots("C")
    assert not st.finish_enrollment("C")["ok"]                # drone already holds a helper
    for _ in range(MAX_FAILURES):
        assert not st.run("C")["ok"]
    r = st.run("C")
    assert not r["ok"] and "locked" in r["reason"]
    st.reset_failures("C")


def test_speed_alarm_denies_and_keeps_reference(tmp_path, feats):
    """A propeller-like change on one motor (-7 sigma speed gain): the key is
    regenerated but access is denied, the station records the alarm (not as a
    failure), the drone does not refresh; a normal boot afterwards is accepted
    and refreshed again."""
    sysp, F = feats
    base = F["C"][:3]
    ref = np.nanmedian(np.vstack(base), axis=0)
    w = np.asarray(sysp["widths"])
    moved = ref.copy()
    moved[7] -= 7 / sysp["K"] * w[7]                     # M2 slower
    sim, st = enrolled(tmp_path, sysp, base + [moved, ref], "C")
    h0 = sim.helper
    r = st.run("C")
    assert not r["ok"] and r["stage"] == "speed_alarm" and "session_key_prefix" not in r
    assert r["speed_alarm"] == [2] and r["refreshed"] is False and "access denied" in r["message"]
    assert r["speed_shift_sigma"][2] == pytest.approx(-7, abs=0.05)
    assert sim.helper == h0                              # reference kept
    db = st.db()["C"]
    assert db["speed_alarm"]["motors"] == [2] and db["alarms"] == 1 and db["failures"] == 0
    r = st.run("C")
    assert r["ok"] and r["speed_alarm"] == [] and r["refreshed"] is True
    assert sim.helper != h0


def test_fingerprint_follows_refresh_and_survives_lost_commit(tmp_path, feats):
    """After each accepted boot the station records the fingerprint of the
    refreshed helper; if the COMMIT is lost the drone keeps the old helper,
    which the station still accepts as its previous fingerprint."""
    sysp, F = feats
    sim, st = enrolled(tmp_path, sysp, F["C"][:3] + [F["C"][3]] * 4, "C")
    assert st.db()["C"]["helper_fp"] == ob.helper_fp(sim.helper, sysp).hex()
    assert st.run("C")["ok"]
    assert st.db()["C"]["helper_fp"] == ob.helper_fp(sim.helper, sysp).hex()
    sim.drop_commit = True
    h_before = sim.helper
    assert st.run("C")["ok"]                              # accepted, refresh not acknowledged
    assert sim.helper == h_before                         # old helper kept by the drone...
    assert st.db()["C"]["helper_fp_prev"] == ob.helper_fp(h_before, sysp).hex()
    sim.drop_commit = False
    assert st.run("C")["ok"]                              # ...and still accepted as the previous one
    assert st.db()["C"]["helper_fp"] == ob.helper_fp(sim.helper, sysp).hex()
    assert st.db()["C"]["failures"] == 0


def test_modified_helper_refused_by_station(tmp_path, feats):
    """Offsets nudged by an attacker who can write the memory: Rep still gives
    the right password (no tag on the drone), but the station does not
    recognise the helper and counts a failure; an old copy written back after
    two refreshes is refused too."""
    sysp, F = feats
    sim, st = enrolled(tmp_path, sysp, F["C"][:3] + [F["C"][3]] * 5, "C")
    old = sim.helper
    assert st.run("C")["ok"] and st.run("C")["ok"]        # two refreshes: `old` is neither current nor previous
    good = sim.helper
    u = ob.unpack_helper(good)
    off = u["offsets"].copy()
    off[0] += np.float32(0.01 * sysp["widths"][0])
    sim.helper = good[:8] + off.astype("<f4").tobytes() + good[8 + 4 * len(off):]
    assert ob.rep(F["C"][3], sim.helper, sysp)[0] == ob.rep(F["C"][3], good, sysp)[0]
    r = st.run("C")
    assert not r["ok"] and r["stage"] == "station" and "not recognised" in r["reason"]
    assert st.db()["C"]["failures"] == 1
    sim.helper = old
    r = st.run("C")
    assert not r["ok"] and "not recognised" in r["reason"]
    sim.helper = good
    assert st.run("C")["ok"]


def test_legacy_drone_needs_fingerprint_registration(tmp_path, feats):
    """A drone enrolled before 2026-10-29: v2 helper (with tag) on the drone,
    no fingerprint in the station. Authentication is refused until the
    operator registers the fingerprint; the password is unchanged."""
    sysp, F = feats
    sim, st = enrolled(tmp_path, sysp, F["C"][:3] + [F["C"][3]] * 2, "C")
    sim.helper = b"UVH2" + sim.helper[4:] + bytes(32)     # as the old firmware stored it
    db = st.db()
    for k in ("helper_fp", "helper_fp_prev"):
        db["C"].pop(k)
    import json
    (tmp_path / "station.json").write_text(json.dumps(db))
    r = st.run("C")
    assert not r["ok"] and "no helper fingerprint" in r["reason"] and st.db()["C"]["failures"] == 0
    reg = st.register_helper_fp()
    assert reg["ok"] and reg["drone"] == "C" and reg["legacy"]
    r = st.run("C")
    assert r["ok"], r
    assert sim.helper[:4] == b"UVH3"                      # refreshed as v3


def test_cloned_helper_on_another_drone_rejected(tmp_path, feats):
    """Memory read-out + cloning: C's helper, read from C after a refresh
    (with history, > one message), written into D's memory; D then claims to
    be C and is rejected by the fuzzy extractor. Writing over an enrolled
    drone is refused."""
    sysp, F = feats
    simC, st = enrolled(tmp_path, sysp, F["C"][:3] + [F["C"][3]], "C")
    assert st.run("C")["ok"]                              # refresh: helper now has history
    blob = st.read_helper()
    assert blob == simC.helper and len(blob) > ob.HELPER_CHUNK
    assert not st.put_helper(blob)                        # C is enrolled: refused
    simD = SimDrone(sysp, F["D"][:2], simC.pk)
    stD = Station(simD, sysp, tmp_path)                   # same station database (C's password)
    assert stD.read_helper() is None
    assert stD.put_helper(blob) and simD.helper == blob
    assert stD.hello()["drone"] == "C"
    for _ in range(2):
        r = stD.run("C")
        assert not r["ok"] and r["stage"] == "drone" and "too many wrong symbols" in r["reason"]


def test_commands_need_the_station_signature(tmp_path, feats):
    """START / ENROLL_HELPER / ERASE are obeyed only with the station's
    signature over the drone's fresh challenge: another key, a replayed
    START, or an erase from someone else are ignored."""
    sysp, F = feats
    sim, st = enrolled(tmp_path, sysp, F["C"][:3] + [F["C"][3]] * 3, "C")
    other = tmp_path / "other"
    station_key(other, create=True)
    import shutil
    shutil.copy(tmp_path / "station.json", other / "station.json")
    rogue = Station(sim, sysp, other)                     # same database, different key
    n_before = len(sim.boots)
    r = rogue.run("C")
    assert not r["ok"] and "not signed" in r["reason"] and len(sim.boots) == n_before   # motors never started
    assert not rogue.erase()["ok"] and sim.helper is not None
    # replay: record a signed START, then send it again
    st.hello()
    sig = ob.sign_cmd(st.key[0], ob.MSG["START"], st.challenge, b"s" * 16)
    for p in ob.fragment(ob.MSG["START"], b"s" * 16 + sig):
        sim.send(p)
    assert len(sim.boots) == n_before - 1                 # first time obeyed
    while sim.recv(0.01):
        pass
    for p in ob.fragment(ob.MSG["START"], b"s" * 16 + sig):
        sim.send(p)
    assert len(sim.boots) == n_before - 1                 # replay ignored
    assert st.erase()["ok"] and sim.helper is None
    assert st.hello()["enrolled"] is False


def test_station_without_key_cannot_command(tmp_path, feats):
    sysp, F = feats
    sim = SimDrone(sysp, F["C"][:1], station_key(tmp_path / "k", create=True)[1])
    st = Station(sim, sysp, tmp_path / "nokey")
    r = st.run("C")
    assert not r["ok"] and "keygen" in r["reason"] and len(sim.boots) == 1
