"""
On-board key protocol (firmware variant FW2, "uavibe-key"): the station-side
and reference implementation of what firmware/uavibe_key/ runs on the drone.
Byte-exact with the C code (checked by tests/test_onboard_c.py against the
host build of the same sources).

Differences from the PC prototype (keyproto/fe.py + spake2.py), all forced by
a 168 MHz microcontroller with ~37 KB of free RAM and no JSON:

- helper data in a fixed binary layout (EEPROM), not JSON; the fleet-wide
  bin widths are compiled into the firmware and bound to the helper through
  a digest of the system parameters;
- BLAKE2b (Monocypher) instead of SHA-256/HKDF for the key; same
  construction (key = H(W, sketch));
- NO robustness tag in the helper (since 2026-10-29, helper format v3,
  agreed with the professor): the tag H(W, helper) of the PC prototype is an
  offline oracle, because whoever reads the helper can test guesses of W
  against it without ever talking to the station. Instead the station keeps
  a fingerprint of the drone's current helper, H(sysparams, helper), which
  does not depend on W; the drone sends it with its CPace message and it is
  bound into the CPace generator, so a modified helper (e.g. offsets nudged
  to move the speed-alarm reference) is refused by the station or makes the
  exchange fail. After a refresh the drone sends the new fingerprint inside
  the encrypted DATA frame and stores the new helper only once the station
  has acknowledged it (COMMIT); the station keeps the previous fingerprint
  too, so a lost COMMIT does not lock the drone out;
- CPace on X25519 (Elligator 2 generator, Monocypher) instead of SPAKE2 on
  edwards25519 (Monocypher has no generic point addition); CPace is the
  CFRG-recommended balanced PAKE. The transcript follows the structure of
  draft-irtf-cfrg-cpace but is NOT byte-compatible with its test vectors
  (BLAKE2b instead of SHA-512, own domain separators);
- XChaCha20-Poly1305 for the encrypted frame (Monocypher crypto_aead_lock,
  libsodium crypto_aead_xchacha20poly1305_ietf: identical).

The Reed-Solomon code and Berlekamp-Welch decoder are those of fe.py.
"""

from __future__ import annotations

import hashlib
import secrets
import struct

import numpy as np
from nacl import bindings as nb

from keyproto.fe import P, rs_encode, rs_decode, gain_index, SPEED_ALARM_SIGMA

HELPER_MAGIC = b"UVH3"
LEGACY_MAGIC = b"UVH2"     # v2 = v3 + robustness tag[32]; still read (tag dropped) to migrate enrolled drones
MAX_HIST = 3


# ── hashing (length-prefixed parts, BLAKE2b-256) ────────────────────────────

def H(*parts: bytes, key: bytes = b"") -> bytes:
    h = hashlib.blake2b(digest_size=32, key=key)
    for p in parts:
        h.update(struct.pack("<H", len(p)) + p)
    return h.digest()


def f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


# ── system parameters (fleet-wide, public, compiled into the firmware) ──────

def sysparams_bytes(sysp: dict) -> bytes:
    """Canonical encoding of what decides the symbols: feature count, t,
    widths (float32), temperature slopes of the board features (float32, 0
    for the motor features)."""
    names = sysp["features"]
    slopes = [sysp["temp_slopes"].get(n, 0.0) for n in names]
    return (struct.pack("<BB", len(names), sysp["t"]) + f32(sysp["widths"]).tobytes()
            + f32(slopes).tobytes())


def sysparams_digest(sysp: dict) -> bytes:
    return H(b"uavibe-sysparams-v2", sysparams_bytes(sysp))


# ── helper data: binary layout ──────────────────────────────────────────────
#  v3: magic "UVH3" | n u8 | t u8 | n_hist u8 | drone_id u8 | offsets f32[n]
#      | sketch u32[n] | history f32[n_hist][n]                 (little-endian)
#  v2 (until 2026-10-28): magic "UVH2", same fields, then a robustness tag[32].

def pack_helper(n, t, offsets, sketch, history, drone_id: int) -> bytes:
    hist = f32(history).reshape(-1, n) if len(history) else np.zeros((0, n), np.float32)
    return (HELPER_MAGIC + struct.pack("<BBBB", n, t, len(hist), drone_id) + f32(offsets).tobytes()
            + np.asarray(sketch, dtype="<u4").tobytes() + hist.tobytes())


def unpack_helper(b: bytes) -> dict:
    """v3, or a v2 helper of a drone enrolled before 2026-10-29 (its tag is
    dropped: `body` is always the v3 encoding, `legacy` tells which it was)."""
    legacy = b[:4] == LEGACY_MAGIC
    if b[:4] != HELPER_MAGIC and not legacy:
        raise ValueError("not a UAVibes helper (v3 or v2)")
    if len(b) < 8:
        raise ValueError("helper length mismatch")
    n, t, nh, did = struct.unpack_from("<BBBB", b, 4)
    o = 8 + 4 * n * (2 + nh)
    if len(b) != o + (32 if legacy else 0):
        raise ValueError("helper length mismatch")
    offsets = np.frombuffer(b, "<f4", n, 8)
    sketch = np.frombuffer(b, "<u4", n, 8 + 4 * n).astype(np.int64)
    hist = np.frombuffer(b, "<f4", n * nh, 8 + 8 * n).reshape(nh, n)
    return dict(n=n, t=t, drone_id=did, offsets=offsets, sketch=sketch, history=hist,
                body=HELPER_MAGIC + b[4:o], legacy=legacy)


def helper_fp(helper: bytes, sysp: dict) -> bytes:
    """Fingerprint of a helper (always of its v3 encoding), kept by the
    station and bound into CPace. Public: it does not depend on W."""
    return H(b"uavibe-helper-v3", sysparams_digest(sysp), unpack_helper(helper)["body"])


def _w_bytes(W) -> bytes:
    return np.asarray([int(v) % P for v in W], dtype="<u4").tobytes()


def _key(spd: bytes, W, sketch) -> bytes:
    return H(b"uavibe-key-v2", spd, np.asarray(sketch, dtype="<u4").tobytes(), _w_bytes(W))


def quantise(x, offsets, widths) -> "list[int | None]":
    """floor((x + o) / w) in double precision on float32 inputs (as the C)."""
    x, o, w = (f32(v).astype(np.float64) for v in (x, offsets, widths))
    q = np.floor((x + o) / w)
    return [int(v) if np.isfinite(v) else None for v in q]


# ── Gen (station, enrollment) / Rep / refresh (drone) ───────────────────────

def gen(reference, sysp: dict, drone: str, rng=secrets) -> "tuple[bytes, bytes]":
    """-> (password, helper bytes). Run once, in the trusted enrollment step.
    The drone's name (one character) is in the helper, so in its fingerprint
    (helper_fp), and it also enters the CPace generator."""
    w = f32(sysp["widths"]).astype(np.float64)
    ref = f32(reference).astype(np.float64)
    n, t = len(w), int(sysp["t"])
    k = n - 2 * t
    idx = np.floor(ref / w)
    offsets = f32((idx + 0.5) * w - ref)
    W = [int(v) % P for v in idx]
    C = rs_encode([rng.randbelow(P) for _ in range(k)], n)
    sketch = [(a - c) % P for a, c in zip(W, C)]
    return _key(sysparams_digest(sysp), W, sketch), pack_helper(n, t, offsets, sketch, [], ord(drone))


def rep(x, helper: bytes, sysp: dict) -> "tuple[bytes | None, object, int]":
    """-> (password, W, wrong symbols) or (None, reason, wrong or -1).
    No integrity check of the helper here (no tag, see the module docstring):
    a modified helper is caught by the station through helper_fp."""
    h = unpack_helper(helper)
    n, t = h["n"], h["t"]
    if n != len(sysp["widths"]) or t != sysp["t"]:
        return None, "helper parameters altered", -1
    q = quantise(x, h["offsets"], sysp["widths"])
    Wp = [(v if v is not None else 0) % P for v in q]
    Cp = [int(a - s) % P for a, s in zip(Wp, h["sketch"])]
    msg = rs_decode(Cp, n - 2 * t, t)
    if msg is None:
        return None, "too many wrong symbols", -1
    C = rs_encode(msg, n)
    W = [int(c + s) % P for c, s in zip(C, h["sketch"])]
    wrong = sum(int(a != b) for a, b in zip(W, Wp))
    return _key(sysparams_digest(sysp), W, h["sketch"]), W, wrong


def _unwrap(W, x, w) -> "list[float]":
    """True bin index closest to x / w consistent with W mod P (NaN if x is)."""
    out = []
    for Wi, xi, wi in zip(W, f32(x).astype(np.float64), f32(w).astype(np.float64)):
        if not np.isfinite(xi):
            out.append(np.nan)
            continue
        a = np.floor(xi / wi)
        d = (Wi - int(a)) % P
        out.append(a + (d if d < P // 2 else d - P))
    return out


def refresh(helper: bytes, W, x, sysp: dict) -> bytes:
    """On-board refresh after an accepted boot: this boot's own centring
    offsets join the history (last MAX_HIST); the new offsets are their
    per-feature median (mean of the two middle values for an even count),
    NaN ignored, previous offset kept if none. Returns a v3 helper."""
    h = unpack_helper(helper)
    n = h["n"]
    w = f32(sysp["widths"]).astype(np.float64)
    own = f32((np.asarray(_unwrap(W, x, w)) + 0.5) * w - f32(x).astype(np.float64))
    hist = np.vstack([h["history"], own[None, :]])[-MAX_HIST:].astype(np.float32)
    off = h["offsets"].copy()
    for i in range(n):
        v = np.sort(hist[:, i][np.isfinite(hist[:, i])])
        if len(v):
            off[i] = v[len(v) // 2] if len(v) % 2 else np.float32((np.float32(v[len(v) // 2 - 1])
                                                                   + np.float32(v[len(v) // 2])) * np.float32(0.5))
    return pack_helper(n, h["t"], off, h["sketch"], hist, h["drone_id"])


def speed_shift(helper: bytes, W, x, sysp: dict) -> np.ndarray:
    """Speed alarm input (see keyproto.fe.SPEED_ALARM_SIGMA): shift of each
    motor's speed gain from the helper's reference, in within-drone sigma,
    float32[4] (NaN if not measured). Computed after Rep, before refresh:
    K * (offset - own centring offset) / w, own as in refresh."""
    h = unpack_helper(helper)
    w = f32(sysp["widths"]).astype(np.float64)
    gi = gain_index(sysp["features"])
    out = np.full(4, np.nan, np.float32)
    for m in range(1, 5):
        i = gi[m]
        xi = np.float64(f32(x[i]))
        if not np.isfinite(xi):
            continue
        idx = _unwrap([W[i]], [xi], [w[i]])[0]
        own = np.float32((idx + 0.5) * w[i] - xi)
        out[m - 1] = np.float32(float(np.float32(sysp["K"])) * (np.float64(h["offsets"][i]) - np.float64(own)) / w[i])
    return out


def alarm_mask(shift) -> int:
    """bit m-1 set = motor m past the threshold (same test as the C)."""
    return sum(1 << m for m in range(4) if np.isfinite(shift[m]) and abs(float(shift[m])) > np.float32(SPEED_ALARM_SIGMA))


# ── DATA frame plaintext ────────────────────────────────────────────────────
#  "UVD2" | drone_id u8 | wrong u8 | alarm mask u8 | refreshed u8
#  | shift i16[4] (0.01 sigma, -32768 = not measured)
#  | helper fingerprint[32] (of the refreshed helper; the current one if not refreshed) | text

DATA_MAGIC = b"UVD2"
DATA_HDR = 48


def _centi(z) -> int:
    if not np.isfinite(z):
        return -32768
    return int(max(-32767, min(32767, np.rint(np.float32(z) * np.float32(100.0)))))


def pack_data(drone_id: int, wrong: int, shift, mask: int, refreshed: bool, hfp: bytes, text: bytes) -> bytes:
    assert len(hfp) == 32
    return (DATA_MAGIC + struct.pack("<BBBB", drone_id, wrong & 0xFF, mask, int(refreshed))
            + struct.pack("<4h", *[_centi(z) for z in shift]) + hfp + text)


def unpack_data(b: bytes) -> dict:
    if b[:4] != DATA_MAGIC or len(b) < DATA_HDR:
        return dict(text=b.decode(errors="replace"))
    did, wrong, mask, refreshed = struct.unpack_from("<BBBB", b, 4)
    sh = struct.unpack_from("<4h", b, 8)
    return dict(drone=chr(did), wrong=wrong, alarm=[m + 1 for m in range(4) if mask >> m & 1],
                refreshed=bool(refreshed),
                shift_sigma={m + 1: (None if v == -32768 else v / 100) for m, v in enumerate(sh)},
                helper_fp=b[16:48], text=b[DATA_HDR:].decode(errors="replace"))


# ── CPace on X25519 ─────────────────────────────────────────────────────────

_p = 2 ** 255 - 19
_A = 486662


def elligator_map(hidden: bytes) -> bytes:
    """Elligator 2 direct map, as Monocypher's crypto_elligator_map
    (non-square 2; the two most significant bits of the input are ignored)."""
    r = int.from_bytes(hidden, "little") & ((1 << 254) - 1)
    r %= _p
    w = (-_A * pow(1 + 2 * r * r, _p - 2, _p)) % _p
    e = pow((w * w * w + _A * w * w + w) % _p, (_p - 1) // 2, _p)
    u = w if e in (0, 1) else (-w - _A) % _p
    return u.to_bytes(32, "little")


def x25519(scalar: bytes, u: bytes) -> bytes:
    return nb.crypto_scalarmult(scalar, u)


def cpace_generator(prs: bytes, sid: bytes, drone_id: bytes, hfp: bytes) -> bytes:
    # "CPace-UAVibe-*" keeps the project's former name: a protocol constant
    # shared with the firmware. v2 (2026-10-29): the helper fingerprint enters
    # the generator (the "associated data" of the CPace draft), so drone and
    # station must agree on the helper, not only on the password.
    return elligator_map(H(b"CPace-UAVibe-v2-gen", prs, sid, drone_id, hfp))


class CPace:
    """One side of the exchange. role 'drone' or 'station'; hfp = helper
    fingerprint the drone used (helper_fp), sent in clear in CPACE_D."""

    def __init__(self, role: str, prs: bytes, sid: bytes, drone_id: bytes, hfp: bytes,
                 scalar: "bytes | None" = None):
        self.role, self.sid = role, sid
        self.g = cpace_generator(prs, sid, drone_id, hfp)
        self.y = scalar or secrets.token_bytes(32)
        self.Y = x25519(self.y, self.g)
        self.isk = None

    def finish(self, peer_Y: bytes) -> None:
        K = x25519(self.y, peer_Y)              # raises on a low-order peer point
        Yd, Ys = (self.Y, peer_Y) if self.role == "drone" else (peer_Y, self.Y)
        self.isk = H(b"CPace-UAVibe-v2-isk", self.sid, K, Yd, Ys)
        self.Yd, self.Ys = Yd, Ys

    def confirm(self, who: "str | None" = None) -> bytes:
        who = who or self.role
        return H(b"confirm-" + who.encode(), self.Yd, self.Ys, key=self.isk)

    def verify(self, tag: bytes) -> bool:
        other = "station" if self.role == "drone" else "drone"
        return secrets.compare_digest(tag, self.confirm(other))

    def data_key(self) -> bytes:
        return H(b"data", key=self.isk)

    def commit(self, new_hfp: bytes) -> bytes:
        """Station's acknowledgement of the refreshed helper (COMMIT): the
        drone stores the refreshed helper only after checking it."""
        return H(b"commit", new_hfp, key=self.isk)


# ── commands signed by the station (Ed25519, 2026-10-29) ────────────────────
# START (motors spin), ENROLL_HELPER and ERASE are accepted by the drone only
# with the station's Ed25519 signature (SHA-512, as libsodium / PyNaCl;
# Monocypher's optional monocypher-ed25519.c on the drone) over
#   "UVCMD1" | message type u8 | challenge[16] | payload
# The challenge is a fresh random value the drone puts in every STATUS (answer
# to HELLO) and forgets after one signed command, so a recorded command cannot
# be replayed. The station's public key is compiled into the firmware
# (uv_station_key.h, written by `python -m keyproto.onboard_station keygen`).

CMD_DOMAIN = b"UVCMD1"
CHALLENGE = 16
SIG = 64


def cmd_message(mtype: int, challenge: bytes, payload: bytes) -> bytes:
    assert len(challenge) == CHALLENGE
    return CMD_DOMAIN + bytes([mtype]) + challenge + payload


def station_keypair(seed: bytes) -> "tuple[bytes, bytes]":
    """seed (32 B, secret) -> (seed, public key 32 B)."""
    from nacl.signing import SigningKey
    return seed, bytes(SigningKey(seed).verify_key)


def sign_cmd(seed: bytes, mtype: int, challenge: bytes, payload: bytes = b"") -> bytes:
    from nacl.signing import SigningKey
    return SigningKey(seed).sign(cmd_message(mtype, challenge, payload)).signature


def verify_cmd(pk: bytes, mtype: int, challenge: "bytes | None", payload: bytes, sig: bytes) -> bool:
    from nacl.signing import VerifyKey
    from nacl.exceptions import BadSignatureError
    if challenge is None or len(sig) != SIG:
        return False
    try:
        VerifyKey(pk).verify(cmd_message(mtype, challenge, payload), sig)
        return True
    except (BadSignatureError, ValueError):
        return False


def station_key_header(pk: bytes) -> str:
    """firmware/uavibe_key/src/core/uv_station_key.h — generated, do not edit."""
    return ("/* Generated by `python -m keyproto.onboard_station keygen`: public key of the\n"
            " * ground station (Ed25519). Only it can sign START, ENROLL_HELPER and ERASE.\n"
            " * The secret half is in auth_client/data/keystore/onboard/station_key.json. */\n"
            "#pragma once\n\n"
            "static const unsigned char UV_STATION_PK[32] = {"
            + ", ".join(f"0x{b:02x}" for b in pk) + "};\n")


def aead_lock(key: bytes, plain: bytes, ad: bytes = b"", nonce: "bytes | None" = None) -> bytes:
    nonce = nonce or secrets.token_bytes(24)
    return nonce + nb.crypto_aead_xchacha20poly1305_ietf_encrypt(plain, ad, nonce, key)


def aead_unlock(key: bytes, frame: bytes, ad: bytes = b"") -> "bytes | None":
    try:
        return nb.crypto_aead_xchacha20poly1305_ietf_decrypt(frame[24:], ad, frame[:24], key)
    except Exception:
        return None


# ── message framing over the app channel (30-byte packets) ──────────────────
#  byte 0: message type; byte 1: fragment index (high nibble) | count (low);
#  bytes 2..: up to 28 payload bytes. Messages up to 15 * 28 = 420 bytes.

MTU = 30
FRAG = MTU - 2

MSG = dict(
    # station -> drone
    HELLO=0x01,          # ()                             -> drone answers STATUS (with a fresh challenge)
    START=0x02,          # sid_station[16] | sig[64]      -> runs the boot protocol (MOTORS SPIN)
    CPACE_S=0x03,        # Y_station[32]
    CONFIRM_S=0x04,      # tag[32]
    ENROLL_HELPER=0x05,  # helper bytes | sig[64] (enrollment only)
    ABORT=0x06,          # stops the protocol, motors off (not signed: it can only stop)
    GET_HELPER=0x07,     # offset u16                     -> HELPER_PART (security test: memory read-out)
    PUT_HELPER=0x08,     # total u16 | offset u16 | data  (only while not enrolled; cloning test)
    COMMIT=0x09,         # tag[32] = CPace.commit(new helper fingerprint): drone stores the refreshed helper
    ERASE=0x0A,          # sig[64]                        -> drone forgets its helper (re-enrollment)
    KEEPALIVE=0x0B,      # ()  every second during a boot; without it for UV_LINK_TIMEOUT_MS the drone stops
    # drone -> station
    STATUS=0x81,         # state u8 | enrolled u8 | n u8 | t u8 | drone_id[1] | challenge[16]
    PROGRESS=0x82,       # phase u8 | motor u8 | percent u8
    FEATURES=0x83,       # f32[n] (only while not enrolled: enrollment mode)
    CPACE_D=0x84,        # sid_drone[16] | Y_drone[32] | helper fingerprint[32]
    CONFIRM_D=0x85,      # tag[32]
    DATA=0x86,           # nonce[24] | ciphertext | mac[16]
    FAIL=0x87,           # reason u8 | wrong symbols i8
    HELPER_OK=0x88,      # stored u8 (PUT_HELPER: 2 = part stored, 1 = complete and loaded)
    HELPER_PART=0x89,    # total u16 | offset u16 | data (total 0 = not enrolled)
)
MSG_NAME = {v: k for k, v in MSG.items()}
HELPER_CHUNK = 400       # helper bytes per GET/PUT_HELPER part (a message carries at most 15 x 28 = 420 B)
# reason codes of the drone's FAIL message (and of the log variable uvkey.result)
FAIL_REASON = {1: "fuzzy extractor: too many wrong symbols",
               2: "helper robustness tag not valid (firmware older than 2026-10-29)",
               3: "the station's key confirmation is not valid", 4: "drone not enrolled",
               5: "measurement not ready: bias and scale not computed yet (wait a few seconds "
                  "after power-up, drone still) or measurement failed", 6: "the station's point is not valid",
               7: "interrupted",
               8: "speed alarm: access denied (only in the log variable uvkey.result)",
               9: "refreshed helper not acknowledged by the station: previous one kept (only in uvkey.result)",
               10: "command not signed by the station (signature or challenge not valid): ignored",
               11: "link with the station lost during the boot: motors stopped"}
LINK_TIMEOUT_S = 3.0     # UV_LINK_TIMEOUT_MS on the drone
KEEPALIVE_S = 1.0
PHASES = {0: "rest (board)", 1: "motor ramp", 2: "motor hold", 3: "pause", 4: "computing",
          5: "key exchange", 6: "done"}


def fragment(mtype: int, payload: bytes) -> "list[bytes]":
    chunks = [payload[i:i + FRAG] for i in range(0, len(payload), FRAG)] or [b""]
    if len(chunks) > 15:
        raise ValueError("message too long")
    return [bytes([mtype, (i << 4) | len(chunks)]) + c for i, c in enumerate(chunks)]


class Reassembler:
    def __init__(self):
        self.parts, self.mtype = {}, None

    def feed(self, pkt: bytes) -> "tuple[int, bytes] | None":
        mtype, idx, cnt = pkt[0], pkt[1] >> 4, pkt[1] & 0x0F
        if idx == 0 or mtype != self.mtype:
            self.parts, self.mtype = {}, mtype
        self.parts[idx] = pkt[2:]
        if cnt and len(self.parts) == cnt:
            data = b"".join(self.parts[i] for i in range(cnt))
            self.parts, self.mtype = {}, None
            return mtype, data
        return None


# ── C header with the system parameters ─────────────────────────────────────

def sysparams_header(sysp: dict) -> str:
    """firmware/uavibe_key/src/core/uv_sysparams.h — generated, do not edit."""
    from keyproto import boot as kb
    from pipeline import motor_model as mm
    from enrollment import micro_sweep as ms
    names = sysp["features"]
    slopes = [sysp["temp_slopes"].get(n, 0.0) for n in names]
    def fl(x) -> str:                        # C literal of the float32 value (exact round trip)
        t = f"{float(np.float32(x)):.9g}"
        return (t if any(ch in t for ch in ".en") else t + ".0") + "f"
    arr = lambda v: ", ".join(fl(x) for x in v)
    L = ["/* Generated by keyproto/onboard.py (sysparams_header) from data/keystore/system.json.",
         " * Fleet-wide public parameters of the UAVibes key. Do not edit by hand. */",
         "#pragma once", "",
         f"#define UV_N {len(names)}", f"#define UV_T {sysp['t']}",
         f"#define UV_N_MOTOR_FEATS {len(kb.MOTOR_FEATS)}",
         f"#define UV_N_BOARD_FEATS {len(kb.BOARD_FEATS)}", "",
         "/* feature order: " + ", ".join(names) + " */",
         f"static const float UV_WIDTHS[UV_N] = {{{arr(sysp['widths'])}}};",
         f"static const float UV_TEMP_SLOPES[UV_N] = {{{arr(slopes)}}};", "",
         "/* boot protocol (enrollment/micro_sweep.py) */",
         f"#define UV_REST_S {ms.BOOT_REST_S:.3f}f", f"#define UV_PWM {ms.BOOT_PWM}",
         f"#define UV_RAMP_S {ms.BOOT_RAMP_S:.3f}f", f"#define UV_HOLD_S {ms.BOOT_HOLD_S:.3f}f",
         f"#define UV_PAUSE_S {ms.BOOT_PAUSE_S:.3f}f", f"#define UV_SETTLE_S {kb.SETTLE_S:.3f}f",
         "#define UV_NFFT 2048", "#define UV_FS 1000.0f", "",
         "/* motor model (pipeline/motor_model.py): RPM = a*(PWM*vbat) + b */",
         f"#define UV_RPM_A {fl(mm.RPM_SLOPE)}", f"#define UV_RPM_B {fl(mm.RPM_OFFSET)}",
         f"#define UV_F0_LO {fl(kb.F0_SEARCH[0])}", f"#define UV_F0_HI {fl(kb.F0_SEARCH[1])}",
         f"#define UV_GRAVITY_LSB {fl(kb.GRAVITY_LSB)}", "",
         "/* speed alarm (keyproto/fe.py SPEED_ALARM_SIGMA): sigma = width / K */",
         f"#define UV_BIN_SIGMAS {fl(sysp['K'])}", f"#define UV_SPEED_ALARM_SIGMA {fl(SPEED_ALARM_SIGMA)}",
         "static const int UV_GAIN_IDX[4] = {" + ", ".join(str(gain_index(names)[m]) for m in range(1, 5)) + "};", "",
         "static const unsigned char UV_SYSPARAMS_DIGEST[32] = {"
         + ", ".join(f"0x{b:02x}" for b in sysparams_digest(sysp)) + "};", ""]
    return "\n".join(L)


# ── reference of the on-board features (same math as uv_features.c) ────────
# Differences from keyproto/boot.py, forced by the RAM budget: no 5 Hz
# high-pass (the floor starts at 20 Hz; the mean is removed), samples stored
# as int16. Selection is the same: UV_NFFT samples after the settle time
# (boot.py's 3 s also reduce to the first 2048 in its one-segment Welch).

NFFT = 2048
FS = 1000.0
# int16 storage of a hold, centred on the rest-phase mean: the firmware's
# low-pass filters leave the ~450 Hz motor line far below one sensor LSB, so
# the scale must be much finer than the sensor's (holds stay within 0.06 g
# and 12 dps of the rest mean on all 22 recorded boots)
LSB = np.array([2.0 ** -16] * 3 + [2.0 ** -10] * 3, dtype=np.float32)   # UV_ACC_LSB, UV_GYRO_LSB
CH = ("acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z")
MAX_GAP_S = 0.05      # SD-log gaps up to this are stitched over in boot_inputs


def to_raw(v, ch: int, centre: float = 0.0) -> np.ndarray:
    return np.clip(np.rint((f32(v) - np.float32(centre)) / LSB[ch]), -32768, 32767).astype(np.int16)


def hold_features_ref(raw: np.ndarray, pwm: float, vbat: float, sysp_f0=(0.92, 1.25)) -> "tuple[float, list]":
    """raw: (6, NFFT) int16 -> (gain, floor dB per channel)."""
    from pipeline.motor_model import RPM_SLOPE, RPM_OFFSET
    fp = (RPM_SLOPE * (pwm * vbat) + RPM_OFFSET) / 60.0
    n = np.arange(NFFT)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * n / NFFT)
    psd = []
    for c in range(6):
        x = (raw[c].astype(np.float64) - raw[c].mean()) * float(LSB[c])
        p = np.abs(np.fft.rfft(x * w)) ** 2 / (FS * np.sum(w * w))
        p[1:-1] *= 2
        psd.append(p)
    f = np.arange(NFFT // 2 + 1) * FS / NFFT
    total = sum(p / np.median(p) for p in psd)
    band = (f > fp * sysp_f0[0]) & (f < min(fp * sysp_f0[1], 0.98 * FS / 2))
    k = int(np.argmax(np.where(band, total, -np.inf)))
    a, b, c3 = np.log(total[k - 1:k + 2])
    den = a - 2 * b + c3
    f0 = f[k] + (0.5 * (a - c3) / den if den != 0 else 0.0) * (f[1] - f[0])
    flo = (f > 20) & (f < 0.9 * f0)
    return f0 / fp, [10 * np.log10(max(np.median(p[flo]), 1e-30)) for p in psd]


def boot_inputs(session_dir) -> dict:
    """What the drone would have measured at this boot, from its SD log:
    rest samples, the four holds as int16 (longest gap-free stretch after
    the settle time, first NFFT samples), PWM, mean vbat, board params."""
    from pathlib import Path
    from keyproto import boot as kb
    from pipeline.preprocessing import load_session
    m = kb._jload(Path(session_dir) / "session.json")
    conv = [a for r in m["recordings"] for a in r["attempts"] if a.get("converted_path")]
    p = Path(conv[-1]["converted_path"])
    df = load_session(p if p.is_absolute() else kb.ROOT / p)
    rest = df[df.phase == "rest_hold"][list(CH)].values.astype(np.float32)
    holds = []
    for mot in (1, 2, 3, 4):
        seg = df[df.phase == f"m{mot}_hold_{kb.BOOT_PWM}"]
        # The SD log has short gaps (3-30 ms, the logging task delayed by the
        # card) that the drone's own RAM acquisition will not have: stitched
        # over; only the real stalls (0.2-0.7 s) split the hold.
        part = kb._longest_contiguous(seg.iloc[int(kb.SETTLE_S * FS):], 1 / MAX_GAP_S * 3).iloc[:NFFT]
        if len(part) < NFFT:
            holds.append(None)
            continue
        centre = rest.mean(axis=0, dtype=np.float64).astype(np.float32)
        raw = np.vstack([to_raw(part[c].values, i, centre[i]) for i, c in enumerate(CH)])
        holds.append(dict(raw=raw, pwm=float(part[f"motor.m{mot}"].mean()), vbat=float(part["vbat_v"].mean())))
    return dict(drone=m["drone"], session=m["session"], rest=rest, holds=holds,
                bp=m.get("board_params") or {})


def board_features_ref(bp: dict, rest: np.ndarray, sysp: dict) -> np.ndarray:
    from keyproto import boot as kb
    names = kb.BOARD_FEATS
    sl = sysp["temp_slopes"]
    T = bp.get("board_temp_c")
    T = np.nan if T is None else T
    valid = bp.get("stPosX") != bp.get("stNegX") and bp.get("stResult") == 0 and np.isfinite(T)
    big = len(rest) > 100
    f = {}
    for i, a in enumerate("XYZ"):
        g = kb.GRAVITY_LSB * rest[:, i].astype(np.float64).mean() if big else np.nan
        d = (bp.get(f"stPos{a}", np.nan) - bp.get(f"stNeg{a}", np.nan)) / 2
        o = (bp.get(f"stPos{a}", np.nan) + bp.get(f"stNeg{a}", np.nan)) / 2 - g
        f[f"d{a}"] = d - sl[f"d{a}"] * (T - 30) if valid else np.nan
        f[f"o{a}"] = o - sl[f"o{a}"] * (T - 30) if valid else np.nan
    for k in ("gBiasY", "gBiasZ", "accScale"):
        f[k] = bp.get(k, np.nan) - sl[k] * (T - 30)
    for i, c in enumerate(("gyro.x", "gyro.y", "gyro.z")):
        f[f"n_{c}"] = rest[:, 3 + i].astype(np.float64).std(ddof=1) if big else np.nan
    return np.array([f[k] for k in names])


def onboard_features(inp: dict, sysp: dict) -> np.ndarray:
    """Feature vector in sysp order, computed as the drone would."""
    from keyproto import boot as kb
    mot = []
    for h in inp["holds"]:
        if h is None:
            mot += [np.nan] * 7
        else:
            g, fl = hold_features_ref(h["raw"], h["pwm"], h["vbat"], kb.F0_SEARCH)
            mot += [g] + fl
    return np.r_[mot, board_features_ref(inp["bp"], inp["rest"], sysp)]


def pack_inputs(inp: dict) -> bytes:
    """Binary input of the host test program (test/host_test.c, 'features'):
    n_rest u32 | rest f32[n_rest][6] | 4 x (ok u8, pwm f32, vbat f32, raw i16[6][NFFT])
    | stPos i16[3] | stNeg i16[3] | stResult i32 | temp f32 | gBias f32[3] | accScale f32"""
    bp = inp["bp"]
    out = struct.pack("<I", len(inp["rest"])) + f32(inp["rest"]).tobytes()
    for h in inp["holds"]:
        if h is None:
            out += struct.pack("<Bff", 0, 0, 0) + bytes(2 * 6 * NFFT)
        else:
            out += struct.pack("<Bff", 1, h["pwm"], h["vbat"]) + h["raw"].astype("<i2").tobytes()
    g = lambda k, d=0: bp.get(k) if bp.get(k) is not None else d
    out += struct.pack("<3h3hi", *(int(g(f"stPos{a}")) for a in "XYZ"), *(int(g(f"stNeg{a}")) for a in "XYZ"),
                       int(g("stResult", -1)))
    T = bp.get("board_temp_c")
    out += struct.pack("<f3ff", np.nan if T is None else T, g("gBiasX", np.nan), g("gBiasY", np.nan),
                       g("gBiasZ", np.nan), g("accScale", np.nan))
    return out
