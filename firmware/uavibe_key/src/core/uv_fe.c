/*
 * Fuzzy extractor, drone side: Rep and refresh. Gen runs once on the
 * station at enrollment (keyproto/onboard.py gen); the drone only stores the
 * helper it receives.
 *
 * Symbols: q_i = floor((x_i + offset_i) / w_i) mod P. Secure sketch:
 * code-offset with a Reed-Solomon RS[UV_N, UV_K] code over GF(65537),
 * decoded with Berlekamp-Welch (up to UV_T wrong symbols). Password
 * BLAKE2b(W, sketch). Same bytes as Python.
 *
 * No robustness tag since 2026-10-29 (helper v3): the old tag BLAKE2b(W,
 * helper), stored in the helper, let whoever reads the memory test guesses of
 * W offline. The helper's integrity is now checked by the station, through
 * the fingerprint uv_helper_fp bound into CPace (uv_crypto.c).
 */
#include <math.h>
#include <string.h>

#include "uv_core.h"

#define P UV_P

/* ── helper layout ─────────────────────────────────────────────────────── */

static void put32(uint8_t* p, uint32_t v) {
  p[0] = v; p[1] = v >> 8; p[2] = v >> 16; p[3] = v >> 24;
}
static uint32_t get32(const uint8_t* p) {
  return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24);
}
static void putf(uint8_t* p, float f) { uint32_t v; memcpy(&v, &f, 4); put32(p, v); }
static float getf(const uint8_t* p) { uint32_t v = get32(p); float f; memcpy(&f, &v, 4); return f; }

static size_t body_len(const uv_helper_t* h) { return 8 + 4 * h->n * (2 + h->n_hist); }

size_t uv_helper_pack(const uv_helper_t* h, uint8_t* out) {
  uint8_t* p = out;
  memcpy(p, "UVH3", 4);
  p[4] = h->n; p[5] = h->t; p[6] = h->n_hist; p[7] = h->drone_id;
  p += 8;
  for (int i = 0; i < h->n; i++, p += 4) putf(p, h->offsets[i]);
  for (int i = 0; i < h->n; i++, p += 4) put32(p, h->sketch[i]);
  for (int j = 0; j < h->n_hist; j++)
    for (int i = 0; i < h->n; i++, p += 4) putf(p, h->hist[j][i]);
  return p - out;
}

bool uv_helper_is_legacy(const uint8_t* in, size_t len) {
  return len >= 4 && memcmp(in, "UVH2", 4) == 0;
}

int uv_helper_unpack(uv_helper_t* h, const uint8_t* in, size_t len) {
  bool legacy = uv_helper_is_legacy(in, len);
  if (len < 8 || (!legacy && memcmp(in, "UVH3", 4) != 0)) return UV_ERR_FORMAT;
  memset(h, 0, sizeof(*h));
  h->n = in[4]; h->t = in[5]; h->n_hist = in[6]; h->drone_id = in[7];
  if (h->n != UV_N || h->t != UV_T || h->n_hist > UV_MAX_HIST) return UV_ERR_PARAMS;
  if (len != body_len(h) + (legacy ? 32 : 0)) return UV_ERR_FORMAT;   /* v2: tag dropped */
  const uint8_t* p = in + 8;
  for (int i = 0; i < h->n; i++, p += 4) h->offsets[i] = getf(p);
  for (int i = 0; i < h->n; i++, p += 4) h->sketch[i] = get32(p);
  for (int j = 0; j < h->n_hist; j++)
    for (int i = 0; i < h->n; i++, p += 4) h->hist[j][i] = getf(p);
  return UV_OK;
}

void uv_helper_fp(const uv_helper_t* h, uint8_t fp[32]) {
  static uint8_t body[UV_HELPER_MAX_BYTES];
  size_t bl = uv_helper_pack(h, body);
  static const char dom[] = "uavibe-helper-v3";
  const uint8_t* parts[] = {(const uint8_t*)dom, UV_SYSPARAMS_DIGEST, body};
  size_t lens[] = {sizeof(dom) - 1, 32, bl};
  uv_hash(fp, NULL, 0, 3, parts, lens);
}

/* ── password ──────────────────────────────────────────────────────────── */

static void w_bytes(const uint32_t W[UV_N], uint8_t out[4 * UV_N]) {
  for (int i = 0; i < UV_N; i++) put32(out + 4 * i, W[i] % P);
}

static void compute_password(const uv_helper_t* h, const uint32_t W[UV_N], uint8_t pw[32]) {
  uint8_t wb[4 * UV_N], sb[4 * UV_N];
  w_bytes(W, wb);
  for (int i = 0; i < UV_N; i++) put32(sb + 4 * i, h->sketch[i]);
  static const char dom[] = "uavibe-key-v2";
  const uint8_t* parts[] = {(const uint8_t*)dom, UV_SYSPARAMS_DIGEST, sb, wb};
  size_t lens[] = {sizeof(dom) - 1, 32, sizeof(sb), sizeof(wb)};
  uv_hash(pw, NULL, 0, 4, parts, lens);
}

/* ── GF(P) arithmetic and Reed-Solomon ─────────────────────────────────── */

static uint32_t mulm(uint32_t a, uint32_t b) { return (uint32_t)(((uint64_t)a * b) % P); }
static uint32_t subm(uint32_t a, uint32_t b) { return (a + P - b) % P; }

static uint32_t powm(uint32_t a, uint32_t e) {
  uint32_t r = 1;
  a %= P;
  for (; e; e >>= 1, a = mulm(a, a))
    if (e & 1) r = mulm(r, a);
  return r;
}
static uint32_t invm(uint32_t a) { return powm(a, P - 2); }

static uint32_t poly_eval(const uint32_t* c, int n, uint32_t x) {
  uint32_t r = 0;
  for (int i = n - 1; i >= 0; i--) r = (mulm(r, x) + c[i]) % P;
  return r;
}

#define NCOL (UV_K + 2 * UV_T)   /* unknowns: Q (k+t coefficients) + E (t, monic) */

typedef struct {
  uint32_t M[UV_N][NCOL + 1];
  uint32_t sol[NCOL];
  uint32_t msg[UV_K];
} rs_scratch_t;

size_t uv_rep_scratch_bytes(void) { return sizeof(rs_scratch_t); }

/* Gaussian elimination mod P (reduced row echelon); 0 if consistent */
static int solve(rs_scratch_t* s) {
  int r = 0, piv[NCOL];
  int npiv = 0;
  for (int c = 0; c < NCOL && r < UV_N; c++) {
    int p = -1;
    for (int i = r; i < UV_N; i++)
      if (s->M[i][c]) { p = i; break; }
    if (p < 0) continue;
    if (p != r)
      for (int j = 0; j <= NCOL; j++) { uint32_t t = s->M[r][j]; s->M[r][j] = s->M[p][j]; s->M[p][j] = t; }
    uint32_t iv = invm(s->M[r][c]);
    for (int j = 0; j <= NCOL; j++) s->M[r][j] = mulm(s->M[r][j], iv);
    for (int i = 0; i < UV_N; i++) {
      if (i == r || !s->M[i][c]) continue;
      uint32_t f = s->M[i][c];
      for (int j = 0; j <= NCOL; j++) s->M[i][j] = subm(s->M[i][j], mulm(f, s->M[r][j]));
    }
    piv[npiv++] = c;
    r++;
    uv_yield();
  }
  for (int i = r; i < UV_N; i++)
    if (s->M[i][NCOL]) return -1;
  memset(s->sol, 0, sizeof(s->sol));
  for (int i = 0; i < npiv; i++) s->sol[piv[i]] = s->M[i][NCOL];
  return 0;
}

/* Berlekamp-Welch: received y (codeword + errors) -> message, 0 on success */
static int rs_decode(const uint32_t y[UV_N], rs_scratch_t* s) {
  const int k = UV_K, t = UV_T;
  for (int i = 0; i < UV_N; i++) {
    uint32_t x = i + 1, xp = 1;
    for (int j = 0; j < k + t; j++, xp = mulm(xp, x)) s->M[i][j] = xp;
    xp = 1;
    for (int j = 0; j < t; j++, xp = mulm(xp, x)) s->M[i][k + t + j] = (P - mulm(y[i], xp)) % P;
    s->M[i][NCOL] = mulm(y[i], powm(x, t));
  }
  if (solve(s)) return -1;
  /* Q / E, E monic of degree t */
  uint32_t Q[UV_K + UV_T], E[UV_T + 1];
  memcpy(Q, s->sol, sizeof(Q));
  for (int j = 0; j < t; j++) E[j] = s->sol[k + t + j];
  E[t] = 1;
  uint32_t* m = s->msg;
  memset(m, 0, sizeof(s->msg));
  for (int i = k - 1; i >= 0; i--) {             /* long division, E monic */
    uint32_t q = Q[i + t];
    m[i] = q;
    for (int j = 0; j <= t; j++) Q[i + j] = subm(Q[i + j], mulm(q, E[j]));
  }
  for (int j = 0; j < t; j++)
    if (Q[j]) return -1;                         /* remainder must be zero */
  int bad = 0;
  for (int i = 0; i < UV_N; i++) bad += poly_eval(m, k, i + 1) != y[i];
  return bad > t ? -1 : 0;
}

/* ── Rep ───────────────────────────────────────────────────────────────── */

int uv_rep(const float* x, const uv_helper_t* h, void* scratch,
           uint8_t password[32], uint32_t W[UV_N], int* wrong) {
  rs_scratch_t* s = (rs_scratch_t*)scratch;
  if (h->n != UV_N || h->t != UV_T) return UV_ERR_PARAMS;
  uint32_t Wp[UV_N], Cp[UV_N];
  for (int i = 0; i < UV_N; i++) {
    double q = floor(((double)x[i] + (double)h->offsets[i]) / (double)UV_WIDTHS[i]);
    int64_t qi = isfinite(q) ? (int64_t)q : 0;
    Wp[i] = (uint32_t)(((qi % (int64_t)P) + P) % P);
    Cp[i] = subm(Wp[i], h->sketch[i] % P);
  }
  if (rs_decode(Cp, s)) return UV_ERR_DECODE;
  int nw = 0;
  for (int i = 0; i < UV_N; i++) {
    W[i] = (poly_eval(s->msg, UV_K, i + 1) + h->sketch[i]) % P;
    nw += W[i] != Wp[i];
  }
  *wrong = nw;
  compute_password(h, W, password);
  return UV_OK;
}

/* ── refresh ───────────────────────────────────────────────────────────── */

static void sort3(float* v, int n) {
  for (int i = 1; i < n; i++)
    for (int j = i; j > 0 && v[j - 1] > v[j]; j--) { float t = v[j]; v[j] = v[j - 1]; v[j - 1] = t; }
}

/* Offset that would centre x[i] in its bin, the bin being the one given by
 * the (corrected) symbol W[i] nearest to x[i] / w. */
static float own_offset(int i, uint32_t Wi, float x) {
  double w = UV_WIDTHS[i], xi = x;
  double a = floor(xi / w);
  int64_t d = ((((int64_t)Wi - (int64_t)a) % (int64_t)P) + P) % P;
  double idx = a + (d < (int64_t)(P / 2) ? d : d - (int64_t)P);
  return (float)((idx + 0.5) * w - xi);
}

void uv_refresh(uv_helper_t* h, const uint32_t W[UV_N], const float* x) {
  float own[UV_N];
  for (int i = 0; i < UV_N; i++)
    own[i] = isfinite(x[i]) ? own_offset(i, W[i], x[i]) : NAN;
  if (h->n_hist == UV_MAX_HIST) {
    memmove(h->hist[0], h->hist[1], sizeof(h->hist[0]) * (UV_MAX_HIST - 1));
    h->n_hist--;
  }
  memcpy(h->hist[h->n_hist++], own, sizeof(own));
  for (int i = 0; i < UV_N; i++) {
    float v[UV_MAX_HIST];
    int n = 0;
    for (int j = 0; j < h->n_hist; j++)
      if (isfinite(h->hist[j][i])) v[n++] = h->hist[j][i];
    if (!n) continue;
    sort3(v, n);
    h->offsets[i] = (n & 1) ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) * 0.5f;
  }
}

int uv_speed_shift(const uv_helper_t* h, const uint32_t W[UV_N], const float* x, float shift[4]) {
  int mask = 0;
  for (int m = 0; m < 4; m++) {
    int i = UV_GAIN_IDX[m];
    if (!isfinite(x[i])) { shift[m] = NAN; continue; }
    float own = own_offset(i, W[i], x[i]);
    shift[m] = (float)((double)UV_BIN_SIGMAS * ((double)h->offsets[i] - (double)own) / (double)UV_WIDTHS[i]);
    if (fabsf(shift[m]) > UV_SPEED_ALARM_SIGMA) mask |= 1 << m;
  }
  return mask;
}

static int16_t centi(float z) {
  if (!isfinite(z)) return INT16_MIN;
  float c = rintf(z * 100.0f);
  return (int16_t)(c > 32767.0f ? 32767 : c < -32767.0f ? -32767 : c);
}

size_t uv_data_pack(uint8_t* out, uint8_t drone_id, int wrong, int mask, int refreshed,
                    const float shift[4], const uint8_t hfp[32], const char* text, size_t text_len) {
  memcpy(out, "UVD2", 4);
  out[4] = drone_id;
  out[5] = (uint8_t)wrong;
  out[6] = (uint8_t)mask;
  out[7] = (uint8_t)(refreshed != 0);
  for (int m = 0; m < 4; m++) {
    uint16_t v = (uint16_t)centi(shift[m]);
    out[8 + 2 * m] = (uint8_t)(v & 0xFF);
    out[9 + 2 * m] = (uint8_t)(v >> 8);
  }
  memcpy(out + 16, hfp, 32);
  memcpy(out + UV_DATA_HDR, text, text_len);
  return UV_DATA_HDR + text_len;
}
