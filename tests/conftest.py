import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def has_data(*parts) -> bool:
    return (ROOT / "data" / Path(*parts)).exists()


@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch):
    """Run in an empty directory: session_manifest paths are relative to cwd."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def first_day_boots() -> "list[str]":
    """Boot sessions of the first day (report 11), sorted. Later days are new
    real data — their outcome is a result to analyse, not a test fixture."""
    import glob
    import json
    from analysis.boot_eval import FIRST_DAY
    out = []
    for s in sorted(glob.glob(str(ROOT / "data" / "fixed_rig_boot" / "s*_drone*"))):
        if json.load(open(Path(s) / "session.json"))["created_iso"].startswith(FIRST_DAY):
            out.append(s)
    return out
