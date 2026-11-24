"""
SPAKE2 over edwards25519 (structure of RFC 9382), with explicit key
confirmation, built on libsodium's group operations (PyNaCl bindings).

  A = drone, B = ground station, w = password scalar derived from the
  fuzzy-extractor output (drone) or from the enrolled secret (station).

  A: x random, X = x*G + w*M          -> B
  B: y random, Y = y*G + w*N          -> A
  A: K = x*(Y - w*N)     B: K = y*(X - w*M)
  TT = H(identities, X, Y, K, w)  ->  Ke (session key), KcA, KcB (confirmation)
  confirmations: MAC(KcA, TT) A->B, MAC(KcB, TT) B->A

Security idea (why a low-entropy w is acceptable here): a passive attacker
learns nothing that lets it test passwords offline; an active attacker can
test ONE password guess per protocol run, which the station rate-limits.
M and N are derived by hashing to the curve (no known discrete logs).
"""

import hashlib
import hmac
import secrets

from nacl import bindings as nb

L = 2 ** 252 + 27742317777372353535851937790883648493     # group order


def _point_from_seed(seed: bytes) -> bytes:
    return nb.crypto_core_ed25519_from_uniform(hashlib.sha256(seed).digest())


# The seeds keep the project's former name (UAVibe): they are protocol constants,
# and changing them would change every derived key and the saved results.
M = _point_from_seed(b"UAVibe SPAKE2 M")
N = _point_from_seed(b"UAVibe SPAKE2 N")


def _scalar(x: int) -> bytes:
    return (x % L).to_bytes(32, "little")


def _mul(k: int, P: bytes) -> bytes:
    return nb.crypto_scalarmult_ed25519_noclamp(_scalar(k), P)


def _mul_base(k: int) -> bytes:
    return nb.crypto_scalarmult_ed25519_base_noclamp(_scalar(k))


def password_scalar(secret: bytes) -> int:
    return int.from_bytes(hashlib.sha512(b"uavibe-spake2-w" + secret).digest(), "little") % L


def _transcript(idA, idB, X, Y, K, w) -> bytes:
    h = hashlib.sha256()
    for part in (idA, idB, X, Y, K, _scalar(w)):
        h.update(len(part).to_bytes(8, "little") + part)
    return h.digest()


def _keys(TT: bytes):
    ke = hmac.new(TT, b"session key", hashlib.sha256).digest()
    kc = hmac.new(TT, b"confirmation keys", hashlib.sha512).digest()
    return ke, kc[:32], kc[32:]


class Party:
    def __init__(self, role: str, secret: bytes, idA: bytes, idB: bytes):
        assert role in ("A", "B")
        self.role, self.w, self.idA, self.idB = role, password_scalar(secret), idA, idB
        self.r = secrets.randbelow(L - 1) + 1
        blind = M if role == "A" else N
        self.msg = nb.crypto_core_ed25519_add(_mul_base(self.r), _mul(self.w, blind))

    def finish(self, peer_msg: bytes) -> bytes:
        """Compute keys; returns this party's confirmation MAC."""
        if not nb.crypto_core_ed25519_is_valid_point(peer_msg):
            raise ValueError("invalid peer point")
        unblind = N if self.role == "A" else M
        K = _mul(self.r, nb.crypto_core_ed25519_sub(peer_msg, _mul(self.w, unblind)))
        X, Y = (self.msg, peer_msg) if self.role == "A" else (peer_msg, self.msg)
        self.TT = _transcript(self.idA, self.idB, X, Y, K, self.w)
        self.ke, kcA, kcB = _keys(self.TT)
        self._mine, self._theirs = (kcA, kcB) if self.role == "A" else (kcB, kcA)
        return hmac.new(self._mine, self.TT, hashlib.sha256).digest()

    def verify(self, peer_conf: bytes) -> bool:
        return hmac.compare_digest(hmac.new(self._theirs, self.TT, hashlib.sha256).digest(), peer_conf)

    @property
    def session_key(self) -> bytes:
        return self.ke
