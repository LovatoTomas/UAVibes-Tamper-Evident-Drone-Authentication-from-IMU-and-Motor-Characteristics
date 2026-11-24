"""SPAKE2 (keyproto/spake2.py)."""

import pytest

from keyproto import spake2


def run(pw_a, pw_b):
    A = spake2.Party("A", pw_a, b"drone", b"station")
    B = spake2.Party("B", pw_b, b"drone", b"station")
    ca, cb = A.finish(B.msg), B.finish(A.msg)
    return A, B, ca, cb


def test_same_password_agrees():
    A, B, ca, cb = run(b"secret", b"secret")
    assert A.verify(cb) and B.verify(ca)
    assert A.session_key == B.session_key


def test_wrong_password_fails_confirmation():
    A, B, ca, cb = run(b"secret", b"other")
    assert not A.verify(cb) and not B.verify(ca)


def test_fresh_keys_each_run():
    A1, _, _, _ = run(b"secret", b"secret")
    A2, _, _, _ = run(b"secret", b"secret")
    assert A1.session_key != A2.session_key


def test_invalid_peer_point_rejected():
    A = spake2.Party("A", b"secret", b"drone", b"station")
    with pytest.raises(ValueError):
        A.finish(b"\x00" * 31 + b"\x80")
