"""
UAVibes key-generation prototype (2026-10): the drone regenerates a secret
from its own measurements at every boot (robust fuzzy extractor), and uses it
as the password of a PAKE (SPAKE2) with the ground station, which yields a
strong session key for an encrypted channel.

  fe.py      quantiser with shielding offsets + Reed-Solomon code-offset
             secure sketch over GF(p) + robustness tag + key derivation
  spake2.py  SPAKE2 over edwards25519 (RFC 9382 style) with key confirmation
  demo.py    end-to-end run on the real recordings (enrollment, boots,
             attacks), writes analysis/results_proto/
"""
