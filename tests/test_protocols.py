"""Recording protocols (enrollment/micro_sweep.py) and the converter's phase attribution."""

import pytest

from enrollment import micro_sweep as ms
from usd_convert import _active_motors_for_phase
from pipeline.motor_model import pwm_to_rpm, MIN_SPIN_PWM


class FakeMotors:
    def __init__(self):
        self.calls = []

    def set_pwm(self, *a, **k):
        self.calls.append(("set", a, k))

    def set_motors(self, *a, **k):
        self.calls.append(("set", a, k))

    def stop_all(self):
        self.calls.append(("stop_all",))

    def stop_motor(self, m):
        self.calls.append(("stop", m))

    def __getattr__(self, name):                 # any other MotorController method
        return lambda *a, **k: self.calls.append((name, a, k))


class Rec:
    def __init__(self):
        self.phases = []

    def set_phase(self, p):
        self.phases.append(p)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    import enrollment.sweep as sw
    monkeypatch.setattr(sw.time, "sleep", lambda s: None)


def test_purposes_per_kind():
    assert len(ms.purposes_for("enroll")) == 15
    assert ms.purposes_for("test")[-1] == "unseen-pwm" and len(ms.purposes_for("test")) == 16
    assert ms.purposes_for("swap") == ["sweep-m1", "sweep-m2", "sweep-m3", "sweep-m4"]
    assert ms.purposes_for("board") == ["rest-imu"]
    assert ms.purposes_for("boot") == ["boot-seq"]


def test_boot_protocol_phases_and_duration():
    r = Rec()
    ms.run_micro("boot-seq", FakeMotors(), r)
    assert r.phases[0] == "rest_hold"
    for m in (1, 2, 3, 4):
        assert f"m{m}_ramp_{ms.BOOT_PWM}" in r.phases and f"m{m}_hold_{ms.BOOT_PWM}" in r.phases
    assert ms.micro_duration_s("boot-seq") == pytest.approx(
        ms.BOOT_REST_S + 4 * (ms.BOOT_RAMP_S + ms.BOOT_HOLD_S + ms.BOOT_PAUSE_S))


def test_rest_protocol_keeps_motors_off():
    r, m = Rec(), FakeMotors()
    ms.run_micro("rest-imu", m, r)
    assert r.phases == ["rest_hold"]
    assert ("stop_all",) in m.calls


def test_auto_pause_rule():
    P = ms.purposes_for("enroll")
    assert ms.auto_pause_after("sweep-m1", P) == ms.AUTO_PAUSE_S
    assert ms.auto_pause_after("sweep-all4", P) == ms.AUTO_PAUSE_AFTER_ALL4_S
    assert ms.auto_pause_after("stepped-m4", P) == ms.AUTO_PAUSE_AFTER_ALL4_S


@pytest.mark.parametrize("phase,motors", [
    ("m2_hold_58000", {2}), ("m3_ramp_58000", {3}), ("m1_hold", {1}), ("m4_only", {4}),
    ("all_four_hold", {1, 2, 3, 4}), ("hold_32000", {1, 2, 3, 4}), ("ramp_to_13000", {1, 2, 3, 4}),
    ("rest_hold", set()), ("pause", set()), ("usd_log", set()),
])
def test_converter_phase_attribution(phase, motors):
    assert set(_active_motors_for_phase(phase)) == motors


def test_motor_model_basics():
    assert pwm_to_rpm(0, 4.0) == 0
    assert pwm_to_rpm(MIN_SPIN_PWM - 1, 4.0) == 0
    assert pwm_to_rpm(40000, 4.0) > pwm_to_rpm(20000, 4.0) > 0
    assert pwm_to_rpm(40000, 4.0) > pwm_to_rpm(40000, 3.5)       # battery sag slows the motor
