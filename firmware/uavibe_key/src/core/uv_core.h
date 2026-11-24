/*
 * UAVibes on-board key — portable core (no FreeRTOS, no Crazyflie headers).
 *
 * Compiled into the firmware app (firmware/uavibe_key) and, unchanged, into
 * the host test program (firmware/uavibe_key/test) that the Python test suite
 * runs against keyproto/onboard.py on the same inputs.
 *
 *   features : boot-protocol measurements -> feature vector x[UV_N]
 *              (same definitions as auth_client/keyproto/boot.py)
 *   fe       : fuzzy extractor, Rep + on-board refresh (same as
 *              keyproto/onboard.py: RS over GF(65537)); no robustness tag since
 *              2026-10-29 (helper v3): the station checks the helper through
 *              its fingerprint, bound into CPace
 *   cpace    : CPace on X25519 with an Elligator 2 generator (Monocypher)
 *   msg      : fragmentation of messages into 30-byte app-channel packets
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "uv_sysparams.h"

#define UV_P        65537u
#define UV_K        (UV_N - 2 * UV_T)
#define UV_MAX_HIST 3
#define UV_NCH      6                    /* acc.x, acc.y, acc.z, gyro.x, gyro.y, gyro.z */
#define UV_NBIN     (UV_NFFT / 2 + 1)
#define UV_BAND_MAX 300                  /* f0 search band, in bins (+2 neighbours) */

/* int16 storage of one hold (24 KB), centred on the rest-phase mean:
 * value = centre + raw * LSB. The firmware's low-pass filters leave the
 * ~450 Hz motor line far below one sensor LSB, so the scale is much finer
 * than the sensor's; the 22 recorded boots stay within 0.06 g and 12 dps of
 * the rest mean (8x and 2.6x margin). Saturated samples are counted. */
#define UV_ACC_LSB  (1.0f / 65536.0f)    /* g,     +-0.5 g  around the rest mean */
#define UV_GYRO_LSB (1.0f / 1024.0f)     /* deg/s, +-32 dps around the rest mean */

/* Called between the slow steps of the computations (per channel of a hold,
 * per column of the RS solver). Provided by the caller: on the drone it
 * sleeps one tick, because the Crazyflie feeds its hardware watchdog
 * (100-350 ms) from the FreeRTOS idle task, which a higher-priority task
 * computing for longer than that starves -> reset. Empty on the PC. */
void uv_yield(void);

/* ── features ─────────────────────────────────────────────────────────── */

/* Running mean / variance of the rest phase (acc for gravity, gyro noise). */
typedef struct {
  uint32_t n;
  double mean[UV_NCH];
  double m2[UV_NCH];
} uv_rest_t;

void uv_rest_init(uv_rest_t* r);
void uv_rest_add(uv_rest_t* r, const float s[UV_NCH]);
float uv_rest_std(const uv_rest_t* r, int ch);       /* ddof = 1, as pandas */

/* *sat is incremented on saturation (may be NULL) */
int16_t uv_to_raw(float v, int ch, float centre, uint32_t* sat);

/* Base motor model prediction of the shaft frequency [Hz]. */
float uv_f_pred(float pwm, float vbat);

/*
 * Features of one single-motor hold: speed gain (measured f0 / prediction)
 * and broadband floor [dB] of each channel. raw[c][i], i < UV_NFFT, is the
 * hold after the settle time. work: UV_NFFT + 2 floats; band: UV_BAND_MAX.
 * Returns 0, or <0 on failure (outputs set to NAN).
 */
int uv_hold_features(const int16_t raw[UV_NCH][UV_NFFT], float pwm, float vbat,
                     float* work, float* band, float* gain, float floor_db[UV_NCH]);

/* What the firmware measured at this boot (param group "uavibe" + baro). */
typedef struct {
  int16_t st_pos[3], st_neg[3];
  int st_result;                         /* 0 = self-test passed */
  float temp_c;
  float gbias[3];
  float acc_scale;
} uv_boot_params_t;

/* Board features (x[UV_N_MOTOR_FEATS ...]) from the boot params + rest stats. */
void uv_board_features(const uv_boot_params_t* bp, const uv_rest_t* rest, float* xb);

/* Real FFT of x[UV_NFFT] in place -> power |X[k]|^2 in x[0..UV_NFFT/2]. */
void uv_power_spectrum(float* x);

/* ── fuzzy extractor ──────────────────────────────────────────────────── */

typedef struct {
  uint8_t n, t, n_hist, drone_id;
  float offsets[UV_N];
  uint32_t sketch[UV_N];
  float hist[UV_MAX_HIST][UV_N];
} uv_helper_t;

/* v3 helper: "UVH3" | n | t | n_hist | drone_id | offsets | sketch | history.
 * A v2 helper ("UVH2", drones enrolled before 2026-10-29) has a 32-byte tag
 * after the same fields: still accepted by uv_helper_unpack (tag dropped), so
 * buffers that read the EEPROM need UV_HELPER_BUF_BYTES. */
#define UV_HELPER_MAX_BYTES (8 + 4 * UV_N * (2 + UV_MAX_HIST))
#define UV_HELPER_BUF_BYTES (UV_HELPER_MAX_BYTES + 32)

size_t uv_helper_pack(const uv_helper_t* h, uint8_t* out);            /* always v3 */
int uv_helper_unpack(uv_helper_t* h, const uint8_t* in, size_t len);  /* v3 or v2 */
bool uv_helper_is_legacy(const uint8_t* in, size_t len);

/* Fingerprint of the helper: BLAKE2b("uavibe-helper-v3", sysparams digest,
 * v3 encoding). Kept by the station, sent in CPACE_D, bound into CPace.
 * Public: it does not depend on W. */
void uv_helper_fp(const uv_helper_t* h, uint8_t fp[32]);

enum { UV_OK = 0, UV_ERR_DECODE = -1, UV_ERR_PARAMS = -3, UV_ERR_FORMAT = -4 };

/*
 * Rep: password (32 B) and the symbol vector W from a measurement.
 * scratch: at least uv_rep_scratch_bytes() bytes (the hold buffer is reused).
 * *wrong = symbols corrected (only meaningful on success).
 */
size_t uv_rep_scratch_bytes(void);
int uv_rep(const float* x, const uv_helper_t* h, void* scratch,
           uint8_t password[32], uint32_t W[UV_N], int* wrong);

/* On-board refresh after an accepted boot (keeps W, moves the offsets). */
void uv_refresh(uv_helper_t* h, const uint32_t W[UV_N], const float* x);

/*
 * Speed alarm (tampering with motors/propellers): shift of each motor's speed
 * gain from the helper's reference, in within-drone sigma (NaN if not
 * measured). Call after a successful uv_rep, BEFORE uv_refresh. Returns the
 * alarm mask: bit m-1 set if |shift| of motor m > UV_SPEED_ALARM_SIGMA.
 */
int uv_speed_shift(const uv_helper_t* h, const uint32_t W[UV_N], const float* x, float shift[4]);

/*
 * Plaintext of the DATA frame: "UVD2" | drone_id | wrong | alarm mask |
 * refreshed | shift int16[4] (0.01 sigma, INT16_MIN = not measured) |
 * fingerprint of the refreshed helper (the current one if not refreshed)[32]
 * | text. Returns the length written (UV_DATA_HDR + text_len).
 */
#define UV_DATA_HDR 48
size_t uv_data_pack(uint8_t* out, uint8_t drone_id, int wrong, int mask, int refreshed,
                    const float shift[4], const uint8_t hfp[32], const char* text, size_t text_len);

/* ── hashing / CPace ──────────────────────────────────────────────────── */

/* BLAKE2b-256 over (u16 LE length | part) for each part; key may be NULL. */
void uv_hash(uint8_t out[32], const uint8_t* key, size_t key_len,
             int n_parts, const uint8_t* const* parts, const size_t* lens);

typedef struct {
  uint8_t sid[32];
  uint8_t g[32], y[32], Y[32];
  uint8_t Yd[32], Ys[32];
  uint8_t isk[32];
} uv_cpace_t;

/* drone side; scalar = 32 random bytes; hfp = fingerprint of the helper used */
void uv_cpace_start(uv_cpace_t* c, const uint8_t prs[32], const uint8_t sid[32],
                    uint8_t drone_id, const uint8_t hfp[32], const uint8_t scalar[32]);
int uv_cpace_finish(uv_cpace_t* c, const uint8_t peer_Y[32]);   /* 0, or -1 if low order */
void uv_cpace_confirm(const uv_cpace_t* c, bool drone, uint8_t tag[32]);
bool uv_cpace_verify_station(const uv_cpace_t* c, const uint8_t tag[32]);
void uv_cpace_data_key(const uv_cpace_t* c, uint8_t key[32]);
/* station's COMMIT of the refreshed helper (fingerprint new_hfp) */
void uv_cpace_commit(const uv_cpace_t* c, const uint8_t new_hfp[32], uint8_t tag[32]);
bool uv_cpace_verify_commit(const uv_cpace_t* c, const uint8_t new_hfp[32], const uint8_t tag[32]);

/* ── commands signed by the station (Ed25519, see keyproto/onboard.py) ── */

#define UV_CHALLENGE 16
#define UV_SIG       64
#define UV_CMD_MAX_PAYLOAD UV_HELPER_MAX_BYTES     /* the largest signed payload: ENROLL_HELPER */

/* true if sig is the station's signature (public key pk) over
 * "UVCMD1" | type | challenge | payload */
bool uv_cmd_verify(const uint8_t pk[32], uint8_t type, const uint8_t challenge[UV_CHALLENGE],
                   const uint8_t* payload, size_t len, const uint8_t sig[UV_SIG]);

/* frame = nonce[24] | ciphertext | mac[16]; returns frame length */
size_t uv_aead_lock(uint8_t* frame, const uint8_t key[32], const uint8_t nonce[24],
                    const uint8_t* ad, size_t ad_len, const uint8_t* plain, size_t len);

/* ── app-channel framing (see keyproto/onboard.py) ────────────────────── */

#define UV_MTU  30
#define UV_FRAG (UV_MTU - 2)
#define UV_MSG_MAX (15 * UV_FRAG)

typedef struct {
  uint8_t type, count;
  uint16_t got;                          /* bit i: fragment i received */
  size_t len;
  uint8_t data[UV_MSG_MAX];
} uv_reasm_t;

void uv_reasm_init(uv_reasm_t* r);
/* returns true when a whole message is ready in r->type / r->data / r->len */
bool uv_reasm_feed(uv_reasm_t* r, const uint8_t* pkt, size_t len);
/* number of packets; packet i is written by uv_frag_packet */
int uv_frag_count(size_t len);
size_t uv_frag_packet(uint8_t pkt[UV_MTU], uint8_t type, const uint8_t* data, size_t len, int i);
