/*
 * Boot-protocol features, as auth_client/keyproto/boot.py computes them from
 * the SD log, with two differences forced by the RAM budget (both checked on
 * the recorded boots, see firmware/FIRMWARE.md):
 *   - no 5 Hz high-pass: the floor is read above 20 Hz, where a 4th-order
 *     5 Hz Butterworth has unit gain within 1e-4; the mean is removed;
 *   - samples stored as int16 around the rest mean (see uv_core.h).
 * Both checked on the 22 recorded boots: same key performance as the offline
 * features (firmware/FIRMWARE.md).
 * The spectrum is the one-segment Welch estimate the Python code ends up
 * using: periodic Hann window over UV_NFFT = 2048 samples after the settle
 * time, one-sided density.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "uv_core.h"

static const float UV_LSB[UV_NCH] = {UV_ACC_LSB, UV_ACC_LSB, UV_ACC_LSB,
                                     UV_GYRO_LSB, UV_GYRO_LSB, UV_GYRO_LSB};

/* ── rest phase ───────────────────────────────────────────────────────── */

void uv_rest_init(uv_rest_t* r) { memset(r, 0, sizeof(*r)); }

void uv_rest_add(uv_rest_t* r, const float s[UV_NCH]) {
  r->n++;
  for (int c = 0; c < UV_NCH; c++) {
    double d = (double)s[c] - r->mean[c];
    r->mean[c] += d / r->n;
    r->m2[c] += d * ((double)s[c] - r->mean[c]);
  }
}

float uv_rest_std(const uv_rest_t* r, int ch) {
  return r->n > 1 ? (float)sqrt(r->m2[ch] / (r->n - 1)) : NAN;
}

int16_t uv_to_raw(float v, int ch, float centre, uint32_t* sat) {
  float q = rintf((v - centre) / UV_LSB[ch]);
  if (q > 32767.0f || q < -32768.0f) {
    if (sat) (*sat)++;
    q = q > 0 ? 32767.0f : -32768.0f;
  }
  return (int16_t)q;
}

float uv_f_pred(float pwm, float vbat) { return (UV_RPM_A * (pwm * vbat) + UV_RPM_B) / 60.0f; }

/* ── FFT ──────────────────────────────────────────────────────────────── */

#define PI_F 3.14159265358979323846f

/* in-place radix-2 complex FFT, interleaved re/im, m points (power of 2) */
static void cfft(float* z, int m) {
  for (int i = 1, j = 0; i < m; i++) {           /* bit reversal */
    int bit = m >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float tr = z[2 * i], ti = z[2 * i + 1];
      z[2 * i] = z[2 * j]; z[2 * i + 1] = z[2 * j + 1];
      z[2 * j] = tr; z[2 * j + 1] = ti;
    }
  }
  for (int len = 2; len <= m; len <<= 1) {
    int half = len >> 1;
    for (int k = 0; k < half; k++) {
      float wr = cosf(-2.0f * PI_F * k / len), wi = sinf(-2.0f * PI_F * k / len);
      for (int s = k; s < m; s += len) {
        int a = 2 * s, b = 2 * (s + half);
        float xr = z[b] * wr - z[b + 1] * wi, xi = z[b] * wi + z[b + 1] * wr;
        z[b] = z[a] - xr; z[b + 1] = z[a + 1] - xi;
        z[a] += xr; z[a + 1] += xi;
      }
    }
  }
}

void uv_power_spectrum(float* x) {
  const int N = UV_NFFT, M = N / 2;
  cfft(x, M);                                    /* z[n] = x[2n] + i x[2n+1] */
  /* split: X[k] = (Z[k] + conj Z[M-k])/2 - i e^{-2 pi i k/N} (Z[k] - conj Z[M-k])/2 */
  float x0 = x[0] + x[1], xm = x[0] - x[1];      /* X[0], X[M] (real) */
  for (int k = 1; k <= M / 2; k++) {
    int j = M - k;
    float ar = x[2 * k], ai = x[2 * k + 1], br = x[2 * j], bi = x[2 * j + 1];
    float er = 0.5f * (ar + br), ei = 0.5f * (ai - bi);      /* (Z[k] + conj Z[j]) / 2 */
    float or_ = 0.5f * (ai + bi), oi = -0.5f * (ar - br);    /* -i (Z[k] - conj Z[j]) / 2 */
    float c = cosf(-2.0f * PI_F * k / N), s = sinf(-2.0f * PI_F * k / N);
    float tr = or_ * c - oi * s, ti = or_ * s + oi * c;
    x[2 * k] = er + tr; x[2 * k + 1] = ei + ti;              /* X[k] */
    x[2 * j] = er - tr; x[2 * j + 1] = -(ei - ti);           /* X[M-k] = conj(E - T) */
  }
  /* power in place: slot k (indices 2k, 2k+1) -> x[k], increasing k is safe */
  for (int k = 1; k < M; k++) x[k] = x[2 * k] * x[2 * k] + x[2 * k + 1] * x[2 * k + 1];
  x[0] = x0 * x0;
  x[M] = xm * xm;
}

/* ── one hold ─────────────────────────────────────────────────────────── */

/* k-th smallest of a[0..n) (a is reordered). Iterative quickselect: no
 * recursion and no libc qsort, whose recursion overflowed the app task's
 * stack on the drone (2026-10-18). The value is exact, as with a sort. */
static float select_kth(float* a, int n, int k) {
  int lo = 0, hi = n - 1;
  while (lo < hi) {
    float pivot = a[(lo + hi) / 2];
    int i = lo, j = hi;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) { float t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else return a[k];
  }
  return a[k];
}

/* numpy-style median of v[0..n) using scratch (destroys nothing in v) */
static float median_of(const float* v, int n, float* scratch) {
  memcpy(scratch, v, sizeof(float) * n);
  float hi = select_kth(scratch, n, n / 2);
  if (n & 1) return hi;
  float lo = scratch[0];                      /* largest of the lower half */
  for (int i = 1; i < n / 2; i++)
    if (scratch[i] > lo) lo = scratch[i];
  return 0.5f * (lo + hi);
}

/* one-sided PSD density of channel c into work[0..UV_NBIN) */
static void channel_psd(const int16_t* raw, int c, float* work) {
  const int N = UV_NFFT;
  double mean = 0.0;
  for (int i = 0; i < N; i++) mean += raw[i];
  mean /= N;
  double wsum2 = 0.0;
  for (int i = 0; i < N; i++) {
    float w = 0.5f - 0.5f * cosf(2.0f * PI_F * i / N);      /* periodic Hann */
    wsum2 += (double)w * (double)w;
    work[i] = (float)((raw[i] - mean) * (double)UV_LSB[c]) * w;
  }
  uv_power_spectrum(work);
  float scale = (float)(1.0 / ((double)UV_FS * wsum2));
  for (int k = 0; k < UV_NBIN; k++) work[k] *= (k == 0 || k == UV_NBIN - 1) ? scale : 2.0f * scale;
}

int uv_hold_features(const int16_t raw[UV_NCH][UV_NFFT], float pwm, float vbat,
                     float* work, float* band, float* gain, float floor_db[UV_NCH]) {
  const float df = UV_FS / UV_NFFT, nyq = UV_FS / 2.0f;
  *gain = NAN;
  for (int c = 0; c < UV_NCH; c++) floor_db[c] = NAN;
  float fp = uv_f_pred(pwm, vbat);
  if (!(fp >= 20.0f)) return -1;
  float hi = fminf(fp * UV_F0_HI, 0.98f * nyq);
  int k_lo = (int)floorf(fp * UV_F0_LO / df) + 1, k_hi = (int)ceilf(hi / df) - 1;
  while (k_lo * df <= fp * UV_F0_LO) k_lo++;                /* strict f > lo */
  while (k_hi * df >= hi) k_hi--;                           /* strict f < hi */
  if (k_hi < k_lo || k_hi - k_lo + 3 > UV_BAND_MAX || k_lo < 1 || k_hi > UV_NBIN - 2) return -2;
  int nb = k_hi - k_lo + 3;                                 /* band + one neighbour each side */
  for (int i = 0; i < nb; i++) band[i] = 0.0f;

  /* pass 1: total = sum_c psd_c / median(psd_c), on the band only */
  for (int c = 0; c < UV_NCH; c++) {
    channel_psd(raw[c], c, work);
    float med = median_of(work, UV_NBIN, work + UV_NBIN);
    for (int i = 0; i < nb; i++) band[i] += work[k_lo - 1 + i] / med;
    uv_yield();
  }
  int kb = 1;
  for (int i = 2; i < nb - 1; i++)
    if (band[i] > band[kb]) kb = i;                         /* first maximum, as np.argmax */
  float a = logf(band[kb - 1]), b = logf(band[kb]), c3 = logf(band[kb + 1]);
  float den = a - 2.0f * b + c3;
  float f0 = (k_lo - 1 + kb) * df + (den != 0.0f ? 0.5f * (a - c3) / den : 0.0f) * df;
  *gain = f0 / fp;

  /* pass 2: broadband floor = median psd over 20 Hz < f < 0.9 f0 */
  int f_lo = 0, f_hi = -1;
  for (int k = 0; k < UV_NBIN; k++) {
    float f = k * df;
    if (f > 20.0f && f < 0.9f * f0) {
      if (f_hi < 0) f_lo = k;
      f_hi = k;
    }
  }
  if (f_hi < f_lo) return -3;
  for (int c = 0; c < UV_NCH; c++) {
    channel_psd(raw[c], c, work);
    float m = median_of(work + f_lo, f_hi - f_lo + 1, work + UV_NBIN);
    floor_db[c] = 10.0f * log10f(fmaxf(m, 1e-30f));
    uv_yield();
  }
  return 0;
}

/* ── board ────────────────────────────────────────────────────────────── */

void uv_board_features(const uv_boot_params_t* bp, const uv_rest_t* rest, float* xb) {
  /* order: dX dY dZ oX oY oZ gBiasY gBiasZ accScale n_gyro.x n_gyro.y n_gyro.z */
  const float* sl = UV_TEMP_SLOPES + UV_N_MOTOR_FEATS;
  float dT = bp->temp_c - 30.0f;
  bool valid = bp->st_pos[0] != bp->st_neg[0] && bp->st_result == 0 && !isnan(bp->temp_c);
  for (int a = 0; a < 3; a++) {
    float g = rest->n > 100 ? UV_GRAVITY_LSB * (float)rest->mean[a] : NAN;
    float d = (bp->st_pos[a] - bp->st_neg[a]) / 2.0f;
    float o = (bp->st_pos[a] + bp->st_neg[a]) / 2.0f - g;
    xb[a] = valid ? d - sl[a] * dT : NAN;
    xb[3 + a] = valid ? o - sl[3 + a] * dT : NAN;
  }
  xb[6] = bp->gbias[1] - sl[6] * dT;
  xb[7] = bp->gbias[2] - sl[7] * dT;
  xb[8] = bp->acc_scale - sl[8] * dT;
  for (int a = 0; a < 3; a++) xb[9 + a] = rest->n > 100 ? uv_rest_std(rest, 3 + a) : NAN;
}
