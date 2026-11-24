/*
 * Hashing, CPace and the encrypted frame, on Monocypher 4 (BLAKE2b, X25519,
 * Elligator 2, XChaCha20-Poly1305). Station side: keyproto/onboard.py.
 *
 * CPace (balanced PAKE, CFRG): both sides derive the generator from the
 * password, g = Elligator2(H(prs, sid, id, helper fingerprint)) — the
 * fingerprint (v2, 2026-10-29) binds the helper the drone used, so a modified
 * helper makes the exchange fail even when it still decodes to the same
 * password; each sends Y = X25519(y, g);
 * K = X25519(y, Y_peer); ISK = H(sid, K, Y_drone, Y_station). Explicit key
 * confirmation with keyed hashes; the data key is derived from ISK.
 * Structure of draft-irtf-cfrg-cpace, own encoding (not its test vectors).
 */
#include <string.h>

#include "monocypher.h"
#include "monocypher-ed25519.h"
#include "uv_core.h"

void uv_hash(uint8_t out[32], const uint8_t* key, size_t key_len,
             int n_parts, const uint8_t* const* parts, const size_t* lens) {
  crypto_blake2b_ctx ctx;
  crypto_blake2b_keyed_init(&ctx, 32, key, key ? key_len : 0);
  for (int i = 0; i < n_parts; i++) {
    uint8_t lp[2] = {(uint8_t)(lens[i] & 0xff), (uint8_t)(lens[i] >> 8)};
    crypto_blake2b_update(&ctx, lp, 2);
    crypto_blake2b_update(&ctx, parts[i], lens[i]);
  }
  crypto_blake2b_final(&ctx, out);
}

void uv_cpace_start(uv_cpace_t* c, const uint8_t prs[32], const uint8_t sid[32],
                    uint8_t drone_id, const uint8_t hfp[32], const uint8_t scalar[32]) {
  /* "CPace-UAVibe-*" keeps the project's former name: a protocol constant,
   * byte-exact with keyproto/onboard.py. */
  static const char dom[] = "CPace-UAVibe-v2-gen";
  uint8_t h[32];
  memset(c, 0, sizeof(*c));
  memcpy(c->sid, sid, 32);
  const uint8_t* parts[] = {(const uint8_t*)dom, prs, sid, &drone_id, hfp};
  size_t lens[] = {sizeof(dom) - 1, 32, 32, 1, 32};
  uv_hash(h, NULL, 0, 5, parts, lens);
  crypto_elligator_map(c->g, h);
  memcpy(c->y, scalar, 32);
  crypto_x25519(c->Y, c->y, c->g);
  crypto_wipe(h, sizeof(h));
}

int uv_cpace_finish(uv_cpace_t* c, const uint8_t peer_Y[32]) {
  static const char dom[] = "CPace-UAVibe-v2-isk";
  uint8_t K[32], z = 0;
  crypto_x25519(K, c->y, peer_Y);
  for (int i = 0; i < 32; i++) z |= K[i];
  if (!z) return -1;                              /* low-order peer point */
  memcpy(c->Yd, c->Y, 32);
  memcpy(c->Ys, peer_Y, 32);
  const uint8_t* parts[] = {(const uint8_t*)dom, c->sid, K, c->Yd, c->Ys};
  size_t lens[] = {sizeof(dom) - 1, 32, 32, 32, 32};
  uv_hash(c->isk, NULL, 0, 5, parts, lens);
  crypto_wipe(K, sizeof(K));
  crypto_wipe(c->y, sizeof(c->y));
  return 0;
}

void uv_cpace_confirm(const uv_cpace_t* c, bool drone, uint8_t tag[32]) {
  const char* dom = drone ? "confirm-drone" : "confirm-station";
  const uint8_t* parts[] = {(const uint8_t*)dom, c->Yd, c->Ys};
  size_t lens[] = {strlen(dom), 32, 32};
  uv_hash(tag, c->isk, 32, 3, parts, lens);
}

bool uv_cpace_verify_station(const uv_cpace_t* c, const uint8_t tag[32]) {
  uint8_t t[32];
  uv_cpace_confirm(c, false, t);
  bool ok = crypto_verify32(t, tag) == 0;
  crypto_wipe(t, sizeof(t));
  return ok;
}

void uv_cpace_data_key(const uv_cpace_t* c, uint8_t key[32]) {
  static const char dom[] = "data";
  const uint8_t* parts[] = {(const uint8_t*)dom};
  size_t lens[] = {sizeof(dom) - 1};
  uv_hash(key, c->isk, 32, 1, parts, lens);
}

void uv_cpace_commit(const uv_cpace_t* c, const uint8_t new_hfp[32], uint8_t tag[32]) {
  static const char dom[] = "commit";
  const uint8_t* parts[] = {(const uint8_t*)dom, new_hfp};
  size_t lens[] = {sizeof(dom) - 1, 32};
  uv_hash(tag, c->isk, 32, 2, parts, lens);
}

bool uv_cpace_verify_commit(const uv_cpace_t* c, const uint8_t new_hfp[32], const uint8_t tag[32]) {
  uint8_t t[32];
  uv_cpace_commit(c, new_hfp, t);
  bool ok = crypto_verify32(t, tag) == 0;
  crypto_wipe(t, sizeof(t));
  return ok;
}

bool uv_cmd_verify(const uint8_t pk[32], uint8_t type, const uint8_t challenge[UV_CHALLENGE],
                   const uint8_t* payload, size_t len, const uint8_t sig[UV_SIG]) {
  static const char dom[] = "UVCMD1";
  static uint8_t msg[sizeof(dom) - 1 + 1 + UV_CHALLENGE + UV_CMD_MAX_PAYLOAD];
  const size_t hdr = sizeof(dom) - 1 + 1 + UV_CHALLENGE;
  if (len > UV_CMD_MAX_PAYLOAD) return false;
  memcpy(msg, dom, sizeof(dom) - 1);
  msg[sizeof(dom) - 1] = type;
  memcpy(msg + sizeof(dom), challenge, UV_CHALLENGE);
  if (len) memcpy(msg + hdr, payload, len);
  return crypto_ed25519_check(sig, pk, msg, hdr + len) == 0;
}

size_t uv_aead_lock(uint8_t* frame, const uint8_t key[32], const uint8_t nonce[24],
                    const uint8_t* ad, size_t ad_len, const uint8_t* plain, size_t len) {
  memcpy(frame, nonce, 24);
  crypto_aead_lock(frame + 24, frame + 24 + len, key, nonce, ad, ad_len, plain, len);
  return 24 + len + 16;
}

/* ── framing ──────────────────────────────────────────────────────────── */

void uv_reasm_init(uv_reasm_t* r) { memset(r, 0, offsetof(uv_reasm_t, data)); }

bool uv_reasm_feed(uv_reasm_t* r, const uint8_t* pkt, size_t len) {
  if (len < 2) return false;
  uint8_t type = pkt[0], idx = pkt[1] >> 4, cnt = pkt[1] & 0x0f;
  if (cnt == 0 || idx >= cnt) return false;
  if (idx == 0 || type != r->type || cnt != r->count) {
    uv_reasm_init(r);
    r->type = type;
    r->count = cnt;
  }
  size_t n = len - 2;
  if (idx < cnt - 1 && n != UV_FRAG) return false;  /* only the last fragment may be short */
  memcpy(r->data + idx * UV_FRAG, pkt + 2, n);
  r->got |= 1u << idx;
  if (idx == cnt - 1) r->len = idx * UV_FRAG + n;
  return r->got == (1u << cnt) - 1;
}

int uv_frag_count(size_t len) { return len ? (int)((len + UV_FRAG - 1) / UV_FRAG) : 1; }

size_t uv_frag_packet(uint8_t pkt[UV_MTU], uint8_t type, const uint8_t* data, size_t len, int i) {
  int cnt = uv_frag_count(len);
  size_t off = (size_t)i * UV_FRAG, n = len > off ? len - off : 0;
  if (n > UV_FRAG) n = UV_FRAG;
  pkt[0] = type;
  pkt[1] = (uint8_t)((i << 4) | cnt);
  if (n) memcpy(pkt + 2, data + off, n);
  return n + 2;
}
