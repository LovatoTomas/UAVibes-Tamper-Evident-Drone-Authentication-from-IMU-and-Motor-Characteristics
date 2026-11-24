"""Key from boot sessions (keyproto/boot.py) on the recorded data; skipped without data."""

import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import ROOT, has_data
from keyproto import boot as kb

pytestmark = pytest.mark.skipif(not has_data("fixed_rig_boot") or not has_data("keystore", "system.json"),
                                reason="needs recorded boot sessions and data/keystore/system.json")


@pytest.fixture
def keystore(tmp_path, monkeypatch):
    shutil.copy(ROOT / "data" / "keystore" / "system.json", tmp_path / "system.json")
    monkeypatch.setattr(kb, "KEYSTORE", tmp_path)
    return tmp_path


def sessions_of(drone):
    """Boots of the first day only (the protocol validated in report 11):
    later days are real data whose outcome is a result, not a test."""
    from analysis.boot_eval import FIRST_DAY
    out = []
    for s in sorted(glob.glob(str(ROOT / "data" / "fixed_rig_boot" / "s*_drone*"))):
        m = json.load(open(Path(s) / "session.json"))
        if m["drone"] == drone and m["created_iso"].startswith(FIRST_DAY):
            out.append(s)
    return out


def test_longest_contiguous_skips_gaps():
    t = np.r_[np.arange(0, 1, 0.001), np.arange(1.5, 3.5, 0.001)]
    seg = pd.DataFrame({"timestamp_s": t})
    assert len(kb._longest_contiguous(seg, 1000.0)) == 2000


def test_features_complete_on_new_protocol():
    for d in "CD":
        for s in sessions_of(d):
            x, info = kb.boot_features(s)
            assert info["n_missing"] == 0, s


def test_genuine_accepted_impostor_rejected(keystore):
    drones = [d for d in "ABCD" if len(sessions_of(d)) >= 4]
    X = {d: [kb.boot_features(s)[0] for s in sessions_of(d)] for d in drones}
    for d in drones:
        kb.enroll(d, np.nanmedian(np.vstack(X[d][:3]), axis=0), "test")
    for d in drones:
        r = kb.boot(d, X[d][-1])
        assert r["ok"], (d, r)
        for o in drones:
            if o != d:
                assert not kb.boot(o, X[d][-1])["ok"]
                st = json.loads((keystore / "station.json").read_text())
                for v in st.values():
                    v["failures"] = 0
                (keystore / "station.json").write_text(json.dumps(st))


def test_station_locks_after_failures(keystore):
    s = sessions_of("A")
    x = kb.boot_features(s[0])[0]
    kb.enroll("A", x, "test")
    junk = x + 1e6
    for _ in range(5):
        assert not kb.boot("A", junk)["ok"]
    r = kb.boot("A", x)
    assert not r["ok"] and "locked" in r["reason"]


def test_identity_tab_fingerprint_and_legacy_helper(keystore):
    """Identity tab, same scheme as the drone: no tag in the helper; a helper
    whose offsets were changed is refused by the station even though it still
    gives the right key; a helper enrolled before 2026-10-29 (with a tag, no
    fingerprint in the station) is accepted and gets its fingerprint."""
    s = sessions_of("A")
    x = kb.boot_features(s[0])[0]
    h = kb.enroll("A", x, "test")
    assert "tag" not in h
    hp = keystore / "droneA_helper.json"
    good = json.loads(hp.read_text())
    bad = dict(good, offsets=[good["offsets"][0] + 1e-6] + good["offsets"][1:])
    hp.write_text(json.dumps(bad))
    r = kb.boot("A", x)
    assert not r["ok"] and r["stage"] == "station" and "not recognised" in r["reason"]
    hp.write_text(json.dumps(dict(good, tag="00" * 32)))      # as an old keystore helper
    st = json.loads((keystore / "station.json").read_text())
    st["A"] = dict(password=st["A"]["password"], failures=0)
    (keystore / "station.json").write_text(json.dumps(st))
    r = kb.boot("A", x)
    assert r["ok"] and r.get("fingerprint_registered")
    assert "tag" not in json.loads(hp.read_text())
    assert kb.boot("A", x)["ok"]


def test_cross_day_replay_on_relabelled_boots():
    """Last two boots of each drone posing as a later day: the machinery must
    accept them (they are same-day boots) and accept no impostor."""
    from analysis import boot_eval as be
    B, names = be.load_boots()
    B = B[B.day == be.FIRST_DAY].copy()
    for d in B.drone.unique():
        last = B[B.drone == d].sort_values("start").index[-2:]
        B.loc[last, "day"] = "2099-01-01"
    mot = np.array([n.startswith("m") for n in names])
    res = be.cross_day(B, names, {"motors": mot, "board": ~mot, "combined": np.ones(len(names), bool)})
    for d, v in res.items():
        assert len(v["genuine"]) == 2
        assert v["impostors_accepted"] == 0
        assert all(g["accepted_refresh"] for g in v["genuine"]), (d, v["genuine"])


def test_speed_alarm_offline_denies_and_keeps_helper(keystore):
    """Offline key (Identity check tab): a motor 7 sigma slower regenerates
    the key but raises a speed alarm -> access denied, not counted as a
    failure, helper NOT refreshed; a normal boot afterwards is accepted."""
    sysp = kb.system_params()
    w = np.asarray(sysp["widths"])
    X = [kb.boot_features(s)[0] for s in sessions_of("C")]
    ref = np.nanmedian(np.vstack(X[:3]), axis=0)
    kb.enroll("C", ref, "test")
    h0 = (keystore / "droneC_helper.json").read_text()
    moved = ref.copy()
    i = sysp["features"].index("m3_58000_gain")
    moved[i] -= 7 / sysp["K"] * w[i]
    r = kb.boot("C", moved)
    assert not r["ok"] and r["stage"] == "speed_alarm"
    assert r["speed_alarm"] == [3] and not r["refreshed"] and "decrypted" not in r
    assert (keystore / "droneC_helper.json").read_text() == h0
    db = json.loads((keystore / "station.json").read_text())["C"]
    assert db["speed_alarm"]["motors"] == [3] and db["alarms"] == 1 and db["failures"] == 0
    r = kb.boot("C", ref)
    assert r["ok"] and r["speed_alarm"] == [] and r["refreshed"]
    assert (keystore / "droneC_helper.json").read_text() != h0
