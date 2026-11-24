"""analysis/onboard_log.py on a synthetic station log (no data needed)."""

import json

import pytest

from analysis import onboard_log as ol


def _rec(at, drone, ok=True, stage=None, reason=None, wrong=0, shift=(0.1, -0.2, 0.3, 0.0), **kw):
    r = dict(at=at, sim=False, expected=drone, mode="auth", ok=ok, wrong=wrong,
             speed_shift_sigma={str(m + 1): v for m, v in enumerate(shift)} if shift else None)
    if stage:
        r["stage"] = stage
    if reason:
        r["reason"] = reason
    r.update(kw)
    return r


@pytest.fixture
def log(tmp_path):
    rows = [
        _rec("2026-11-01T10:00:00", "A", condition="nominal"),
        _rec("2026-11-01T10:02:00", "A", condition="nominal", shift=(0.0, 3.5, 0.0, 0.0)),   # beyond 3 sigma
        _rec("2026-11-01T10:04:00", "A", ok=False, stage="drone", reason="fuzzy extractor: troppi simboli sbagliati",
             wrong=-1, shift=None, condition="nominal"),
        _rec("2026-11-01T10:06:00", "A", ok=False, stage="station", reason="timeout", shift=None,
             condition="nominal"),                                                           # no decision
        _rec("2026-11-01T10:08:00", "A", ok=False, stage="speed_alarm", wrong=1,
             shift=(0.0, 0.0, 0.0, -11.0), condition="propeller"),
        _rec("2026-11-01T10:10:00", "A", condition="tape", shift=(0.0, 0.0, 4.5, 0.0)),        # accepted by the key
        _rec("2026-11-01T10:12:00", "B", condition=None),                                     # from the labels
        dict(at="2026-11-01T10:14:00", sim=True, expected="A", mode="auth", ok=True),          # simulated: ignored
        dict(at="2026-11-01T10:15:00", sim=False, mode="attack_read"),                        # not an attempt
    ]
    p = tmp_path / "log.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    lab = tmp_path / "labels.json"
    lab.write_text(json.dumps(dict(windows=[dict(**{"from": "2026-11-01T10:11:00", "to": "2026-11-01T10:13:00"},
                                                 claimed="B", condition="nominal", note="lab")])))
    return p, lab


def test_conditions_outcomes_and_exact_symbols(log):
    df = ol.load(*log)
    assert len(df) == 7
    assert df.condition.value_counts().to_dict() == {"nominal": 5, "propeller": 1, "tape": 1}
    res = ol.analyse(df)
    assert res["frr_now"]["n"] == 4 and res["frr_now"]["rejections"] == 1        # timeout excluded
    ex = {e["T"]: e for e in res["exact_speed_symbols"]}
    assert ex[3.0]["extra_rejections"] == 1 and ex[4.0]["extra_rejections"] == 0
    assert ex[3.0]["altered_caught"] == 2 and ex[5.0]["altered_caught"] == 1     # tape at 4.5 sigma
    assert res["boots_needed_zero_failures"] == {"5%": 59, "2%": 149, "1%": 299}
    assert 0 < res["frr_now"]["upper95"] < 1
    assert "## 3." in ol.report(res)


def test_clopper_pearson_rule_of_three():
    assert ol.cp_upper(0, 100) == pytest.approx(0.0295, abs=5e-4)
    assert ol.cp_upper(3, 3) == 1.0
