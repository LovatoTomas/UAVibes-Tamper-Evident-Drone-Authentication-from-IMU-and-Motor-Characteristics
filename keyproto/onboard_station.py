"""
Ground station for the on-board key (firmware variant FW2, firmware/uavibe_key).

Enrollment (trusted environment, drone on the rig, not yet enrolled):
  1. each boot (power-cycle the drone first: the board part is measured at
     power-up) the station sends START; the drone runs the boot protocol and,
     because it has no helper yet, answers with its feature vector;
  2. after N_ENROLL boots the station builds the helper from their median
     (keyproto.onboard.gen), sends it to the drone (stored in EEPROM, accepted
     only while not enrolled) and keeps ONLY the password.
  Shared before any authentication: system parameters (compiled into the
  firmware, uv_sysparams.h, and in data/keystore/system.json on the station);
  drone -> station: nothing secret after enrollment; station -> drone: the
  helper (public); station keeps: drone name, password, failure counter and
  the fingerprint of the drone's current helper (keyproto.onboard.helper_fp)
  plus the previous one.

Authentication (every power-up): START -> drone measures, runs Rep with its
helper -> CPACE_D carries the fingerprint of the helper the drone used: the
station accepts it only if it is the current or the previous one it
recorded (a modified helper is refused and counted as a failure), and the
fingerprint enters the CPace generator -> mutual key confirmation -> one
encrypted frame (keyproto.onboard.pack_data: corrected symbols, speed shift
of each motor, speed alarm, fingerprint of the refreshed helper) -> after a
refresh the station records the new fingerprint (keeping the one just used
as the previous) and sends COMMIT; the drone stores the refreshed helper only
after checking COMMIT, otherwise it keeps the one the station knows as the
previous. The station locks a drone after MAX_FAILURES failures.

No robustness tag (since 2026-10-29, see keyproto/onboard.py): it would be an
offline oracle for whoever reads the helper. Drones enrolled before carry a v2
helper with a tag; the firmware drops it, and register_helper_fp (operator,
protected setting) records the fingerprint once.

Signed commands (since 2026-10-29): START, ENROLL_HELPER and ERASE carry the
station's Ed25519 signature over a fresh challenge the drone sent in STATUS
(keyproto.onboard.sign_cmd), so nobody else in radio range can spin the
motors, enroll or erase a drone, and a recorded command cannot be replayed.
The key pair is data/keystore/onboard/station_key.json (secret; created by
`python -m keyproto.onboard_station keygen`, which also writes the public key
into firmware/uavibe_key/src/core/uv_station_key.h: reflash after it). During
a boot the station sends KEEPALIVE every second; a drone that hears nothing
for keyproto.onboard.LINK_TIMEOUT_S stops the motors and gives up.

A speed alarm (keyproto.fe.SPEED_ALARM_SIGMA) DENIES access: the
key exchange succeeded, so the station knows it is that drone, but its
mechanics changed (propellers/motors). The denial is recorded in the station
database (`speed_alarm`, `alarms`), is not counted as a failure (no
password guessing happened), the session key is discarded, and the drone
keeps its old reference, so it repeats until re-enrollment.

The feature vectors of the enrollment boots are kept in
data/keystore/onboard/research/ for the thesis (comparison with the SD
logs); a real station would delete them after Gen.
"""

from __future__ import annotations

import json
import os
import queue
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from keyproto import onboard as ob

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "keystore" / "onboard"
N_ENROLL = 3
MAX_FAILURES = 5
M = ob.MSG


class Link:
    """Packet transport (30-byte app-channel packets)."""

    def send(self, pkt: bytes) -> None:
        raise NotImplementedError

    def recv(self, timeout: float) -> "bytes | None":
        raise NotImplementedError


class CflibLink(Link):
    def __init__(self, cf):
        self.cf = cf
        self.q: "queue.Queue[bytes]" = queue.Queue()
        cf.appchannel.packet_received.add_callback(self._rx)

    def _rx(self, data):
        self.q.put(bytes(data))

    def close(self):
        try:
            self.cf.appchannel.packet_received.remove_callback(self._rx)
        except Exception:
            pass

    def send(self, pkt: bytes) -> None:
        self.cf.appchannel.send_packet(pkt)

    def recv(self, timeout: float) -> "bytes | None":
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None


def _jload(p: Path, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def _jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1), encoding="utf-8")


KEY_HEADER = ROOT.parent / "firmware" / "uavibe_key" / "src" / "core" / "uv_station_key.h"


def station_key(store: Path = STORE, create: bool = False) -> "tuple[bytes, bytes] | None":
    """(seed, public key) of the station's Ed25519 key, from
    <store>/station_key.json; created only with create=True."""
    p = Path(store) / "station_key.json"
    k = _jload(p)
    if k is None:
        if not create:
            return None
        seed, pk = ob.station_keypair(os.urandom(32))
        _jsave(p, dict(seed=seed.hex(), public=pk.hex(), created=datetime.now().isoformat(timespec="seconds"),
                       note="SECRET: the station signs START / ENROLL_HELPER / ERASE with it"))
        return seed, pk
    return bytes.fromhex(k["seed"]), bytes.fromhex(k["public"])


class Station:
    def __init__(self, link: Link, sysp: dict, store: Path = STORE):
        self.link, self.sysp, self.store = link, sysp, Path(store)
        self.rx = ob.Reassembler()
        self.key = station_key(self.store)
        self.challenge: "bytes | None" = None

    # ── transport ──────────────────────────────────────────────────────────
    def _send(self, mtype: str, payload: bytes = b"") -> None:
        for p in ob.fragment(M[mtype], payload):
            self.link.send(p)

    def _recv(self, timeout: float) -> "tuple[str, bytes] | None":
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            pkt = self.link.recv(max(0.01, end - time.monotonic()))
            if pkt is None:
                continue
            msg = self.rx.feed(pkt)
            if msg:
                return ob.MSG_NAME.get(msg[0], hex(msg[0])), msg[1]
        return None

    # ── database ───────────────────────────────────────────────────────────
    @property
    def db_path(self) -> Path:
        return self.store / "station.json"

    def db(self) -> dict:
        return _jload(self.db_path, {})

    def enroll_boots(self, drone: str) -> list:
        return _jload(self.store / "research" / f"enroll_{drone}.json", [])

    def reset_failures(self, drone: str) -> None:
        d = self.db()
        if drone in d:
            d[drone]["failures"] = 0
            _jsave(self.db_path, d)

    def forget(self, drone: str) -> None:
        """Remove the drone from the station and drop its enrollment boots."""
        d = self.db()
        d.pop(drone, None)
        _jsave(self.db_path, d)
        p = self.store / "research" / f"enroll_{drone}.json"
        if p.exists():
            p.rename(p.with_name(f"{p.stem}_old_{datetime.now():%Y%m%d_%H%M%S}.json"))

    # ── protocol ───────────────────────────────────────────────────────────
    def hello(self, timeout: float = 2.0) -> "dict | None":
        """STATUS of the drone; also takes the fresh challenge that the next
        signed command must cover (None from a firmware without signatures)."""
        self.rx = ob.Reassembler()
        self._send("HELLO")
        end = time.monotonic() + timeout
        r = None
        while time.monotonic() < end:              # stale messages of an earlier exchange are skipped
            r = self._recv(max(0.01, end - time.monotonic()))
            if r is None or r[0] == "STATUS":
                break
        if not r or r[0] != "STATUS" or len(r[1]) < 5:
            return None
        s = r[1]
        self.challenge = s[5:5 + ob.CHALLENGE] if len(s) >= 5 + ob.CHALLENGE else None
        return dict(state=s[0], enrolled=bool(s[1]), n=s[2], t=s[3], drone=chr(s[4]) if s[1] else None,
                    signed_commands=self.challenge is not None)

    def _signed(self, mtype: str, payload: bytes = b"") -> "str | None":
        """Send a signed command over a fresh challenge (HELLO first).
        Returns None, or the reason it could not be sent."""
        if self.key is None:
            return ("the station key is missing (data/keystore/onboard/station_key.json): run "
                    "`python -m keyproto.onboard_station keygen` and reflash the firmware")
        if self.hello() is None:
            return "the drone does not answer"
        if self.challenge is None:
            return "the drone sends no challenge: does it run firmware older than 2026-10-29? Reflash it"
        sig = ob.sign_cmd(self.key[0], M[mtype], self.challenge, payload)
        self.challenge = None                     # one command per challenge
        self.rx = ob.Reassembler()
        self._send(mtype, payload + sig)
        return None

    def erase(self, timeout: float = 3.0) -> dict:
        """The drone forgets its helper (signed ERASE). The station entry is
        not touched (forget() does that)."""
        why = self._signed("ERASE")
        if why:
            return dict(ok=False, reason=why)
        r = self._recv(timeout)
        if r and r[0] == "FAIL":
            return dict(ok=False, reason=ob.FAIL_REASON.get(r[1][0], f"code {r[1][0]}"))
        if not r or r[0] != "HELPER_OK" or not r[1] or r[1][0] != 1:
            return dict(ok=False, reason="the drone did not confirm the erase (is it in a boot?)")
        return dict(ok=True)

    def run(self, expected_drone: str, progress=None, timeout: float = 120.0) -> dict:
        """One boot: START, then enrollment features or the authentication.
        expected_drone: who the station believes it is talking to (the
        password it uses); choose another drone to test an impostor."""
        t0 = time.monotonic()
        sid_s = os.urandom(16)
        why = self._signed("START", sid_s)
        if why:
            return dict(mode="auth", ok=False, stage="station", reason=why)
        db = self.db()
        entry = db.get(expected_drone)
        cp = None
        last_ka = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if time.monotonic() - last_ka >= ob.KEEPALIVE_S:
                self._send("KEEPALIVE")           # without it the drone stops the motors
                last_ka = time.monotonic()
            r = self._recv(min(1.0, ob.KEEPALIVE_S))
            if r is None:
                continue
            kind, data = r
            if kind == "PROGRESS" and progress:
                progress(ob.PHASES.get(data[0], "?"), data[1], data[2])
            elif kind == "FEATURES":
                x = np.frombuffer(data, "<f4").astype(float)
                rec = dict(at=datetime.now().isoformat(timespec="seconds"), features=x.tolist())
                boots = self.enroll_boots(expected_drone) + [rec]
                _jsave(self.store / "research" / f"enroll_{expected_drone}.json", boots)
                return dict(mode="enroll", ok=True, features=x, n_boots=len(boots),
                            n_missing=int(np.isnan(x).sum()), seconds=time.monotonic() - t0)
            elif kind == "FAIL":
                reason = ob.FAIL_REASON.get(data[0], f"code {data[0]}")
                wrong = (data[1] - 256 if data[1] > 127 else data[1]) if len(data) > 1 else None   # int8
                if cp is not None or data[0] in (1, 2):
                    self._failure(expected_drone)
                return dict(mode="auth", ok=False, stage="drone", reason=reason, wrong=wrong,
                            seconds=time.monotonic() - t0)
            elif kind == "CPACE_D":
                if entry is None:
                    self._send("ABORT")
                    return dict(mode="auth", ok=False, stage="station", reason="drone not enrolled in the station")
                if entry["failures"] >= MAX_FAILURES:
                    self._send("ABORT")
                    return dict(mode="auth", ok=False, stage="station",
                                reason="station locked: too many failed attempts")
                if len(data) != 80:
                    self._send("ABORT")
                    return dict(mode="auth", ok=False, stage="station",
                                reason="CPace message without the helper fingerprint: does the drone run "
                                       "firmware older than 2026-10-29? Reflash it")
                hfp = data[48:80]
                if not entry.get("helper_fp"):
                    self._send("ABORT")
                    return dict(mode="auth", ok=False, stage="station",
                                reason="no helper fingerprint registered for this drone (enrolled before "
                                       "2026-10-29): use “Register the helper fingerprint”")
                if hfp.hex() not in (entry["helper_fp"], entry.get("helper_fp_prev")):
                    # the drone's helper is not the one the station knows (nor the
                    # previous one): modified memory, or an old copy written back
                    self._send("ABORT")
                    self._failure(expected_drone)
                    return dict(mode="auth", ok=False, stage="station",
                                reason="helper data not recognised by the station (modified, or an old "
                                       "copy): fingerprint different from the registered one",
                                seconds=time.monotonic() - t0)
                sid = sid_s + data[:16]
                cp = ob.CPace("station", bytes.fromhex(entry["password"]), sid, expected_drone.encode(), hfp)
                try:
                    cp.finish(data[16:48])
                except Exception:
                    self._send("ABORT")
                    self._failure(expected_drone)
                    return dict(mode="auth", ok=False, stage="station", reason="the drone's point is not valid")
                self._send("CPACE_S", cp.Y)
            elif kind == "CONFIRM_D":
                if cp is None or not cp.verify(data):
                    self._send("ABORT")
                    self._failure(expected_drone)
                    return dict(mode="auth", ok=False, stage="station",
                                reason="the drone's key confirmation is not valid (different password)",
                                seconds=time.monotonic() - t0)
                self._send("CONFIRM_S", cp.confirm())
            elif kind == "DATA":
                pt = ob.aead_unlock(cp.data_key(), data, expected_drone.encode()) if cp else None
                if pt is None:
                    self._failure(expected_drone)
                    return dict(mode="auth", ok=False, stage="station", reason="encrypted frame not valid")
                info = ob.unpack_data(pt)
                now = datetime.now().isoformat(timespec="seconds")
                d = self.db()
                d[expected_drone]["failures"] = 0
                out = dict(mode="auth", message=info.get("text", ""),
                           wrong=info.get("wrong"), speed_alarm=info.get("alarm", []),
                           speed_shift_sigma=info.get("shift_sigma"), refreshed=info.get("refreshed"),
                           seconds=time.monotonic() - t0)
                if info.get("alarm"):
                    # right password, but a motor's speed moved far from the
                    # reference: propellers/motors touched -> access denied, the
                    # session key is not used. The drone keeps its old
                    # reference, so this repeats until re-enrollment.
                    d[expected_drone]["speed_alarm"] = dict(at=now, motors=info["alarm"],
                                                            shift_sigma=info["shift_sigma"])
                    d[expected_drone]["alarms"] = d[expected_drone].get("alarms", 0) + 1
                    _jsave(self.db_path, d)
                    return dict(out, ok=False, stage="speed_alarm",
                                reason="access denied: speed of motor(s) "
                                       + ", ".join(f"M{m}" for m in info["alarm"])
                                       + " changed (propellers/motors tampered with)")
                committed = False
                if info.get("refreshed"):
                    # the drone refreshed its helper: record the new fingerprint
                    # (the one just used becomes the previous) BEFORE sending
                    # COMMIT, so whichever of the two the drone ends up storing
                    # (COMMIT received or lost) is accepted at the next boot
                    d[expected_drone]["helper_fp_prev"] = hfp.hex()
                    d[expected_drone]["helper_fp"] = info["helper_fp"].hex()
                    committed = True
                d[expected_drone]["last_ok"] = now
                _jsave(self.db_path, d)
                if committed:
                    self._send("COMMIT", cp.commit(info["helper_fp"]))
                return dict(out, ok=True, session_key_prefix=cp.data_key().hex()[:16])
        self._send("ABORT")
        return dict(mode="auth", ok=False, stage="station", reason="timeout")

    def _failure(self, drone: str) -> None:
        d = self.db()
        if drone in d:
            d[drone]["failures"] += 1
            _jsave(self.db_path, d)

    # ── security test: memory read-out and cloning ──────────────────────────
    def read_helper(self, timeout: float = 2.0) -> "bytes | None":
        """The helper exactly as stored in the drone's EEPROM (public data:
        what an attacker reading the memory gets). None if not enrolled."""
        self.rx = ob.Reassembler()
        out, total, off = b"", None, 0
        while total is None or off < total:
            self._send("GET_HELPER", off.to_bytes(2, "little"))
            r = self._recv(timeout)
            if not r or r[0] != "HELPER_PART" or len(r[1]) < 4:
                return None
            total, got = int.from_bytes(r[1][:2], "little"), int.from_bytes(r[1][2:4], "little")
            if total == 0:
                return None
            if got != off or len(r[1]) == 4:
                return None
            out += r[1][4:]
            off += len(r[1]) - 4
        return out

    def put_helper(self, blob: bytes, timeout: float = 5.0) -> bool:
        """Write a helper (e.g. copied from another drone) into the drone's
        EEPROM. The drone accepts it only while not enrolled (signed ERASE first)."""
        self.rx = ob.Reassembler()
        for off in range(0, len(blob), ob.HELPER_CHUNK):
            part = blob[off:off + ob.HELPER_CHUNK]
            self._send("PUT_HELPER", len(blob).to_bytes(2, "little") + off.to_bytes(2, "little") + part)
            r = self._recv(timeout)
            last = off + len(part) == len(blob)
            if not r or r[0] != "HELPER_OK" or not r[1] or r[1][0] != (1 if last else 2):
                return False
        return True

    def finish_enrollment(self, drone: str, timeout: float = 5.0) -> dict:
        """Gen from the median of the enrollment boots, helper to the drone,
        password to the station."""
        boots = self.enroll_boots(drone)
        if len(boots) < N_ENROLL:
            return dict(ok=False, reason=f"{N_ENROLL} enrollment boots are needed, there are {len(boots)}")
        X = np.array([b["features"] for b in boots[-N_ENROLL:]], dtype=float)
        ref = np.nanmedian(X, axis=0)
        if np.isnan(ref).any():
            return dict(ok=False, reason="quantities missing in every enrollment boot")
        pw, helper = ob.gen(ref, self.sysp, drone)
        why = self._signed("ENROLL_HELPER", helper)
        if why:
            return dict(ok=False, reason=why)
        r = self._recv(timeout)
        if r and r[0] == "FAIL":
            return dict(ok=False, reason=ob.FAIL_REASON.get(r[1][0], f"code {r[1][0]}"))
        if not r or r[0] != "HELPER_OK" or not r[1] or r[1][0] != 1:
            return dict(ok=False, reason="the drone did not accept the helper data (is it already enrolled? "
                                         "erase it first with “Restart enrollment”)")
        d = self.db()
        d[drone] = dict(password=pw.hex(), failures=0, enrolled_at=datetime.now().isoformat(timespec="seconds"),
                        n_boots=N_ENROLL, helper_bytes=len(helper),
                        helper_fp=ob.helper_fp(helper, self.sysp).hex(), helper_fp_prev=None)
        _jsave(self.db_path, d)
        return dict(ok=True, helper_bytes=len(helper))

    def register_helper_fp(self, timeout: float = 2.0) -> dict:
        """Migration of a drone enrolled before 2026-10-29 (v2 helper with a
        tag, no fingerprint in the station): read the helper from the drone
        and record its fingerprint. Trust on first use, so only by the
        operator, in the protected enrollment setting."""
        blob = self.read_helper(timeout)
        if blob is None:
            return dict(ok=False, reason="the drone is not enrolled or does not answer")
        drone = chr(ob.unpack_helper(blob)["drone_id"])
        d = self.db()
        if drone not in d:
            return dict(ok=False, reason=f"the drone presents itself as {drone}, which is not enrolled in the station")
        d[drone]["helper_fp"] = ob.helper_fp(blob, self.sysp).hex()
        d[drone]["helper_fp_prev"] = None
        d[drone]["helper_fp_registered_at"] = datetime.now().isoformat(timespec="seconds")
        _jsave(self.db_path, d)
        return dict(ok=True, drone=drone, legacy=ob.unpack_helper(blob)["legacy"], helper_bytes=len(blob))


# ── simulated drone (tests, and a dry run of the dashboard without hardware) ──

class SimDrone(Link):
    """Python twin of firmware/uavibe_key/src/uv_app.c. Each START consumes
    the next feature vector of `boots` (as if the drone had been power-cycled
    and measured again)."""

    def __init__(self, sysp: dict, boots: "list[np.ndarray]", station_pk: bytes):
        """station_pk: the public key compiled into the firmware."""
        self.sysp, self.boots, self.pk = sysp, list(boots), station_pk
        self.challenge: "bytes | None" = os.urandom(ob.CHALLENGE)
        self.helper: "bytes | None" = None
        self.out: "queue.Queue[bytes]" = queue.Queue()
        self.rx = ob.Reassembler()
        self.cp = self.W = self.x = self.wrong = None
        self.pending: "bytes | None" = None     # refreshed helper waiting for COMMIT
        self.drop_commit = False                # test: the station's COMMIT never arrives

    def _emit(self, mtype: str, payload: bytes = b"") -> None:
        for p in ob.fragment(M[mtype], payload):
            self.out.put(p)

    def recv(self, timeout: float) -> "bytes | None":
        try:
            return self.out.get(timeout=min(timeout, 0.01))
        except queue.Empty:
            return None

    def send(self, pkt: bytes) -> None:
        msg = self.rx.feed(pkt)
        if not msg:
            return
        kind, data = ob.MSG_NAME.get(msg[0]), msg[1]
        if kind in ("START", "ENROLL_HELPER", "ERASE"):
            # signed by the station over the current challenge, used once
            payload, sig = data[:-ob.SIG], data[-ob.SIG:]
            ok = len(data) >= ob.SIG and ob.verify_cmd(self.pk, msg[0], self.challenge, payload, sig)
            self.challenge = os.urandom(ob.CHALLENGE)
            if not ok:
                self._emit("FAIL", bytes([10, 255]))
                return
            data = payload
        if kind == "HELLO":
            did = ob.unpack_helper(self.helper)["drone_id"] if self.helper else ord("?")
            self.challenge = os.urandom(ob.CHALLENGE)
            self._emit("STATUS", bytes([0, int(self.helper is not None), len(self.sysp["widths"]),
                                        self.sysp["t"], did]) + self.challenge)
        elif kind == "ERASE":
            self.helper = self.pending = None
            self._emit("HELPER_OK", bytes([1]))
        elif kind == "START":
            if len(data) != 16:
                return
            self.x = np.asarray(self.boots.pop(0), dtype=float)
            self.sid_s = data
            for ph in (0, 1, 2, 3, 4):
                self._emit("PROGRESS", bytes([ph, 0, 0]))
            if self.helper is None:
                self._emit("FEATURES", ob.f32(self.x).tobytes())
                return
            pw, W, wrong = ob.rep(self.x, self.helper, self.sysp)
            if pw is None:
                self._emit("FAIL", bytes([1, (wrong if wrong >= 0 else 255) & 0xFF]))
                return
            self.W, self.wrong, self.pending = W, wrong, None
            sid_d = os.urandom(16)
            hfp = ob.helper_fp(self.helper, self.sysp)
            self.cp = ob.CPace("drone", pw, self.sid_s + sid_d, bytes([ob.unpack_helper(self.helper)["drone_id"]]),
                               hfp)
            self._emit("CPACE_D", sid_d + self.cp.Y + hfp)
        elif kind == "CPACE_S":
            self.cp.finish(data)
            self._emit("CONFIRM_D", self.cp.confirm())
        elif kind == "CONFIRM_S":
            if not self.cp.verify(data):
                self._emit("FAIL", bytes([3, self.wrong]))
                return
            shift = ob.speed_shift(self.helper, self.W, self.x, self.sysp)
            mask = ob.alarm_mask(shift)
            if not mask:                       # with a speed alarm the reference is kept
                self.pending = ob.refresh(self.helper, self.W, self.x, self.sysp)
            new_fp = ob.helper_fp(self.pending or self.helper, self.sysp)
            did = ob.unpack_helper(self.helper)["drone_id"]
            text = (f"UAVibes drone {chr(did)}: key regenerated on board"
                    + (", SPEED ALARM: access denied" if mask else "")).encode()
            plain = ob.pack_data(did, self.wrong, shift, mask, not mask, new_fp, text)
            self._emit("DATA", ob.aead_lock(self.cp.data_key(), plain, bytes([did])))
        elif kind == "COMMIT":
            # the refreshed helper is stored only once the station has recorded it
            if (self.pending is not None and not self.drop_commit
                    and data == self.cp.commit(ob.helper_fp(self.pending, self.sysp))):
                self.helper = self.pending
            self.pending = None
        elif kind == "GET_HELPER":
            off = int.from_bytes(data[:2], "little")
            h = self.helper or b""
            part = h[off:off + ob.HELPER_CHUNK]
            self._emit("HELPER_PART", len(h).to_bytes(2, "little") + off.to_bytes(2, "little") + part)
        elif kind == "PUT_HELPER":
            total, off, part = int.from_bytes(data[:2], "little"), int.from_bytes(data[2:4], "little"), data[4:]
            ok = 0
            if self.helper is None and off + len(part) <= total:
                self._staging = (getattr(self, "_staging", b"")[:off] if off else b"") + part
                ok = 2
                if off + len(part) == total:
                    ob.unpack_helper(self._staging)
                    self.helper, ok = self._staging, 1
            self._emit("HELPER_OK", bytes([ok]))
        elif kind == "ENROLL_HELPER":
            ok = self.helper is None
            if ok:
                ob.unpack_helper(data)
                self.helper = data
            self._emit("HELPER_OK", bytes([int(ok)]))


# ── command line: key of the station ────────────────────────────────────────

def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m keyproto.onboard_station",
                                 description="Station key (Ed25519) for the signed commands of FW2.")
    ap.add_argument("cmd", choices=["keygen", "show"],
                    help="keygen: create the key if missing and write the firmware header; show: print it")
    a = ap.parse_args(argv)
    k = station_key(STORE, create=(a.cmd == "keygen"))
    if k is None:
        print("no station key yet: run `python -m keyproto.onboard_station keygen`")
        return 1
    print(f"station public key: {k[1].hex()}  ({STORE / 'station_key.json'})")
    if a.cmd == "keygen":
        KEY_HEADER.write_text(ob.station_key_header(k[1]), encoding="utf-8")
        print(f"written {KEY_HEADER} -> rebuild and reflash FW2 (make; make cload) on every drone")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
