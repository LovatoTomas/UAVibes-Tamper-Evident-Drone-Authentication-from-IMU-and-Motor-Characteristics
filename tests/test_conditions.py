"""Altered-drone sessions (props/payload kinds and analysis/conditions.py)."""

import json

import pandas as pd
import pytest
from scipy.signal import resample

from enrollment import session_manifest as sm
from enrollment.micro_sweep import purposes_for
from tests.conftest import ROOT, has_data


def test_condition_kinds_use_boot_protocol_and_own_trees():
    for k in sm.CONDITION_KINDS:
        assert purposes_for(k) == ["boot-seq"]
        assert sm.rig_dir_for(k) != sm.rig_dir_for("boot")
    assert sm.micro_stem("A", 2, "boot-seq", kind="props").split("_")[3] == "e02"


def test_condition_saved_only_for_condition_kinds(tmp_cwd):
    m = sm.SessionManifest.load_or_new("p.json", "A", 1, "payload")
    m.condition = dict(mass_g=5.0, position="centro", mount="nastro", notes="")
    m.save("p.json")
    assert sm.SessionManifest.load("p.json").condition["mass_g"] == 5.0
    b = sm.SessionManifest.load_or_new("b.json", "A", 1, "boot")
    assert "condition" not in b.to_dict()


@pytest.mark.skipif(not has_data("fixed_rig_boot"), reason="needs recorded boot sessions")
def test_conditions_analysis_detects_motor_change(tmp_path, monkeypatch):
    from analysis import conditions as cond
    # the latest boot session of C whose SD log is already uploaded and converted
    conv = None
    for src in sorted((ROOT / "data" / "fixed_rig_boot").glob("s*_droneC"), reverse=True):
        m = json.loads((src / "session.json").read_text(encoding="utf-8"))
        conv = next((a["converted_path"] for r in m["recordings"] for a in r["attempts"]
                     if a.get("converted_path")), None)
        if conv:
            break
    if conv is None:
        pytest.skip("no converted boot session of drone C")
    df = pd.read_csv(ROOT / conv)
    root = tmp_path / "fixed_rig_props"

    def make(n, frame):
        d = root / f"s{n:02d}_droneC"
        (d / "converted").mkdir(parents=True)
        frame.to_csv(d / "converted" / "x_1khz.csv", index=False)
        mm = dict(m, kind="props", session=n, created_iso="2099-01-01T00:00:00",
                  condition=dict(change="test", motors=[1], notes=""))
        mm["recordings"] = [dict(r, attempts=[dict(a, converted_path=str(d / "converted" / "x_1khz.csv"))
                                              for a in r["attempts"] if a.get("converted_path")][-1:])
                            for r in m["recordings"]]
        (d / "session.json").write_text(json.dumps(mm), encoding="utf-8")

    make(1, df)                                   # unchanged boot: must be accepted
    # M1 hold played back 10 % slower: every line moves down 10 %. Band-limited
    # (FFT) resampling — the shaft line sits at ~450 Hz with fs = 1 kHz, where
    # linear interpolation would create spurious images instead of a shift.
    slow = df.copy()
    hold = (slow.phase == "m1_hold_58000").values
    n = int(hold.sum())
    for c in ("acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"):
        slow.loc[hold, c] = resample(df.loc[hold, c].values, int(round(n / 0.9)))[:n]
    make(2, slow)
    monkeypatch.setitem(cond.KIND_DIRS, "props", root)
    monkeypatch.setattr(cond, "OUT", tmp_path / "out")
    r = {x["session"]: x for x in cond.run("props", verbose=False)}
    assert r[1]["accepted"] and r[1]["wrong"]["board"] == 0
    assert r[2]["speed_change_pct"]["M1"] == pytest.approx(-10, abs=1)
    assert r[2]["wrong"]["board"] == 0
    assert (root / "s02_droneC" / "analysis" / "verdict.md").exists()
