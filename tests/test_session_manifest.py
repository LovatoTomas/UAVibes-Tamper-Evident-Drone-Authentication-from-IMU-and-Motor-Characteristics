"""Session bookkeeping and the never-overwrite upload path (enrollment/session_manifest.py)."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from enrollment import session_manifest as sm


def test_micro_stem_kinds():
    ts = datetime(2026, 10, 13, 9, 18, 22)
    assert sm.micro_stem("A", 3, "sweep-m1", ts) == "20261013_091822_droneA_s03_sweep-m1"
    assert sm.micro_stem("A", 3, "sweep-m1", ts, kind="test").endswith("_t03_sweep-m1")
    assert sm.micro_stem("B", 5, "boot-seq", ts, kind="boot").endswith("_a05_boot-seq")
    assert sm.micro_stem("A", 7, "sweep-m2", ts, kind="swap", motors="B", stage="swap") == \
        "20261013_091822_frameA_motorsB_swap_w07_sweep-m2"


def test_rig_dirs_are_separate():
    dirs = {k: sm.rig_dir_for(k) for k in sm.KINDS}
    assert len(set(dirs.values())) == len(dirs)
    with pytest.raises(ValueError):
        sm.rig_dir_for("nope")


def test_unique_path_never_overwrites(tmp_path):
    p = tmp_path / "x.raw"
    assert sm.unique_path(p) == p
    p.write_text("1")
    p2 = sm.unique_path(p)
    assert p2.name == "x_2.raw"
    p2.write_text("2")
    assert sm.unique_path(p).name == "x_3.raw"


def test_load_or_new_refuses_kind_mismatch(tmp_cwd):
    mp = Path("m.json")
    sm.SessionManifest.load_or_new(mp, "A", 1, "enroll").save(mp)
    with pytest.raises(ValueError):
        sm.SessionManifest.load_or_new(mp, "A", 1, "test")


def test_swap_config_validation():
    sm.validate_swap_config("swap", "A", "B")
    sm.validate_swap_config("pre", "A", "A")
    with pytest.raises(ValueError):
        sm.validate_swap_config("swap", "A", "A")
    with pytest.raises(ValueError):
        sm.validate_swap_config("pre", "A", "B")


def _attempt(m, mp, purpose, tl_dir, with_usd=True):
    tl = tl_dir / f"{purpose}_{len(m.all_attempts())}.csv"
    tl.write_text("wall_time_s,event,value\n" + ("1,usd_start,\n2,usd_stop,\n" if with_usd else ""))
    a = m.start_attempt(purpose, 58000, tl)
    a.status = "completed"
    m.save(mp)
    return a


def test_upload_matching_with_discard_and_dead_attempt(tmp_cwd):
    """The FIFO must keep a discarded attempt's slot (its log is still on the
    card) and skip an attempt that never started SD logging."""
    dirs = sm.drone_dirs(sm.rig_dir_for("enroll"), "A")
    for d in dirs.values():
        d.mkdir(parents=True)
    mp = sm.manifest_path(sm.rig_dir_for("enroll"), "A", 1)
    m = sm.SessionManifest.load_or_new(mp, "A", 1, "enroll")
    a1 = _attempt(m, mp, "sweep-m1", dirs["timelines"])
    a2 = _attempt(m, mp, "sweep-m2", dirs["timelines"])
    m.discard_attempt(a2.attempt_id)
    _attempt(m, mp, "sweep-m3", dirs["timelines"], with_usd=False)      # died before usd_start
    a4 = _attempt(m, mp, "sweep-m2", dirs["timelines"])                 # redo
    assert m.recordings[1].needs_recording is False                     # redo exists now

    conv = lambda raw, tl: (pd.DataFrame({"x": [1]}), {"n_samples": 1})
    uploads = [("log00", b"f0"), ("log01", b"f1"), ("log02", b"f2")]
    res, left = sm.ingest_upload_batch(m, mp, uploads, dirs, conv)
    assert [r[1] for r in res] == [True, True, True] and left == []
    m = sm.SessionManifest.load(mp)
    by_id = {a.attempt_id: a for a in m.all_attempts()}
    assert Path(by_id[a1.attempt_id].raw_log_path).read_bytes() == b"f0"
    assert "discarded" in by_id[a2.attempt_id].raw_log_path                # kept, not converted
    assert by_id[a2.attempt_id].converted_path is None
    assert Path(by_id[a4.attempt_id].raw_log_path).read_bytes() == b"f2"


def test_same_log_names_never_overwrite(tmp_cwd):
    """The 2026-09-20 bug: log00 of a later session must not replace log00 of an earlier one."""
    dirs = sm.drone_dirs(sm.rig_dir_for("enroll"), "B")
    for d in dirs.values():
        d.mkdir(parents=True)
    conv = lambda raw, tl: (pd.DataFrame({"x": [1]}), {"n_samples": 1})
    paths = []
    for session, payload in ((1, b"session1"), (2, b"session2")):
        mp = sm.manifest_path(sm.rig_dir_for("enroll"), "B", session)
        m = sm.SessionManifest.load_or_new(mp, "B", session, "enroll")
        _attempt(m, mp, "sweep-m1", dirs["timelines"])
        sm.ingest_upload_batch(m, mp, [("log00", payload)], dirs, conv)
        paths.append(Path(sm.SessionManifest.load(mp).all_attempts()[0].raw_log_path))
    assert paths[0] != paths[1]
    assert paths[0].read_bytes() == b"session1" and paths[1].read_bytes() == b"session2"


def test_extra_uploads_go_to_unmatched(tmp_cwd):
    dirs = sm.drone_dirs(sm.rig_dir_for("enroll"), "C")
    for d in dirs.values():
        d.mkdir(parents=True)
    mp = sm.manifest_path(sm.rig_dir_for("enroll"), "C", 1)
    m = sm.SessionManifest.load_or_new(mp, "C", 1, "enroll")
    _attempt(m, mp, "sweep-m1", dirs["timelines"])
    conv = lambda raw, tl: (pd.DataFrame({"x": [1]}), {"n_samples": 1})
    res, left = sm.ingest_upload_batch(m, mp, [("log00", b"a"), ("log01", b"b")], dirs, conv)
    assert len(res) == 1 and len(left) == 1 and left[0].parent.name == "unmatched"
