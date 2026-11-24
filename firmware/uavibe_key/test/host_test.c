/*
 * Host build of the on-board core, driven by auth_client/tests/test_onboard_c.py:
 * one command per line on stdin, results on stdout (hex for bytes).
 *
 *   features <file>                  -> 40 floats (file: keyproto.onboard.pack_inputs)
 *   rep <helper> <x f32[]>           -> ok <password> <W u32[]> <wrong> | err <code>
 *   refresh <helper> <W u32[]> <x>   -> <helper>
 *   shift <helper> <W u32[]> <x>     -> <DATA plaintext of an accepted boot (text "t"),
 *                                        with the fingerprint of this helper>
 *   fp <helper>                      -> <fingerprint> <helper re-packed as v3>
 *   elligator <32 B>                 -> <32 B>
 *   cpace <prs> <sid> <id> <hfp> <scalar> <peer Y>
 *                                    -> <Y> <isk> <confirm drone> <confirm station> <data key>
 *                                       <commit of hfp>
 *   aead <key> <nonce> <ad> <plain>  -> <frame>
 *   sigcheck <pk> <type> <challenge> <payload> <sig>  -> ok | bad
 *   frag <type> <data>               -> <packet> ...
 *   reasm <packet> ...               -> <type> <data>
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "monocypher.h"
#include "uv_core.h"

void uv_yield(void) {}

static uint8_t buf_a[UV_HELPER_MAX_BYTES * 2], buf_b[4096], buf_c[4096], buf_d[4096], buf_e[4096];

static size_t unhex(const char* s, uint8_t* out, size_t max) {
  size_t n = 0;
  if (!s || !strcmp(s, "-")) return 0;
  for (; s[0] && s[1] && n < max; s += 2) {
    unsigned v;
    sscanf(s, "%2x", &v);
    out[n++] = (uint8_t)v;
  }
  return n;
}

static void phex(const uint8_t* b, size_t n) {
  for (size_t i = 0; i < n; i++) printf("%02x", b[i]);
  if (!n) printf("-");
}

static float getf(const uint8_t* p) { float f; memcpy(&f, p, 4); return f; }

static uint8_t scratch[16384];

static void cmd_features(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) { printf("err open\n"); return; }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t* d = malloc(len);
  if (fread(d, 1, len, f) != (size_t)len) { printf("err read\n"); fclose(f); free(d); return; }
  fclose(f);
  const uint8_t* p = d;
  uint32_t nrest;
  memcpy(&nrest, p, 4); p += 4;
  uv_rest_t rest;
  uv_rest_init(&rest);
  for (uint32_t i = 0; i < nrest; i++, p += 24) {
    float s[6];
    memcpy(s, p, 24);
    uv_rest_add(&rest, s);
  }
  float x[UV_N];
  static int16_t raw[UV_NCH][UV_NFFT];
  static float work[UV_NFFT + 2], band[UV_BAND_MAX];
  for (int m = 0; m < 4; m++) {
    uint8_t ok = p[0];
    float pwm = getf(p + 1), vbat = getf(p + 5);
    p += 9;
    memcpy(raw, p, sizeof(raw));
    p += sizeof(raw);
    float gain, fl[UV_NCH];
    if (!ok || uv_hold_features((const int16_t(*)[UV_NFFT])raw, pwm, vbat, work, band, &gain, fl)) {
      gain = NAN;
      for (int c = 0; c < UV_NCH; c++) fl[c] = NAN;
    }
    x[7 * m] = gain;
    for (int c = 0; c < UV_NCH; c++) x[7 * m + 1 + c] = fl[c];
  }
  uv_boot_params_t bp;
  memcpy(bp.st_pos, p, 6); p += 6;
  memcpy(bp.st_neg, p, 6); p += 6;
  int32_t sr; memcpy(&sr, p, 4); p += 4;
  bp.st_result = sr;
  bp.temp_c = getf(p); p += 4;
  for (int a = 0; a < 3; a++, p += 4) bp.gbias[a] = getf(p);
  bp.acc_scale = getf(p);
  uv_board_features(&bp, &rest, x + UV_N_MOTOR_FEATS);
  for (int i = 0; i < UV_N; i++) printf("%.9g ", x[i]);
  printf("\n");
  free(d);
}

int main(void) {
  static char line[65536];
  setvbuf(stdout, NULL, _IOLBF, 0);
  while (fgets(line, sizeof(line), stdin)) {
    char* cmd = strtok(line, " \r\n");
    if (!cmd) continue;
    char* a[8] = {0};
    for (int i = 0; i < 8; i++) a[i] = strtok(NULL, " \r\n");

    if (!strcmp(cmd, "features")) {
      cmd_features(a[0]);
    } else if (!strcmp(cmd, "fp")) {
      uv_helper_t h;
      size_t hl = unhex(a[0], buf_a, sizeof(buf_a));
      int e = uv_helper_unpack(&h, buf_a, hl);
      if (e) { printf("err %d\n", e); continue; }
      uint8_t fp[32];
      uv_helper_fp(&h, fp);
      phex(fp, 32); printf(" ");
      size_t n = uv_helper_pack(&h, buf_b);
      phex(buf_b, n); printf("\n");
    } else if (!strcmp(cmd, "rep") || !strcmp(cmd, "refresh") || !strcmp(cmd, "shift")) {
      uv_helper_t h;
      size_t hl = unhex(a[0], buf_a, sizeof(buf_a));
      int e = uv_helper_unpack(&h, buf_a, hl);
      if (e) { printf("err %d\n", e); continue; }
      float x[UV_N];
      uint32_t W[UV_N];
      if (!strcmp(cmd, "rep")) {
        unhex(a[1], (uint8_t*)x, sizeof(x));
        uint8_t pw[32];
        int wrong = 0;
        e = uv_rep(x, &h, scratch, pw, W, &wrong);
        if (e) { printf("err %d\n", e); continue; }
        printf("ok "); phex(pw, 32); printf(" "); phex((uint8_t*)W, sizeof(W)); printf(" %d\n", wrong);
      } else if (!strcmp(cmd, "shift")) {
        unhex(a[1], (uint8_t*)W, sizeof(W));
        unhex(a[2], (uint8_t*)x, sizeof(x));
        float sh[4];
        int mask = uv_speed_shift(&h, W, x, sh);
        uint8_t fp[32];
        uv_helper_fp(&h, fp);
        size_t n = uv_data_pack(buf_b, h.drone_id, 2, mask, !mask, sh, fp, "t", 1);
        phex(buf_b, n); printf("\n");
      } else {
        unhex(a[1], (uint8_t*)W, sizeof(W));
        unhex(a[2], (uint8_t*)x, sizeof(x));
        uv_refresh(&h, W, x);
        size_t n = uv_helper_pack(&h, buf_b);
        phex(buf_b, n); printf("\n");
      }
    } else if (!strcmp(cmd, "elligator")) {
      uint8_t in[32], out[32];
      unhex(a[0], in, 32);
      crypto_elligator_map(out, in);
      phex(out, 32); printf("\n");
    } else if (!strcmp(cmd, "cpace")) {
      uint8_t prs[32], sid[32], id, hfp[32], sc[32], peer[32], t[32];
      unhex(a[0], prs, 32); unhex(a[1], sid, 32); unhex(a[2], &id, 1);
      unhex(a[3], hfp, 32); unhex(a[4], sc, 32); unhex(a[5], peer, 32);
      uv_cpace_t c;
      uv_cpace_start(&c, prs, sid, id, hfp, sc);
      phex(c.Y, 32); printf(" ");
      if (uv_cpace_finish(&c, peer)) { printf("lowpoint\n"); continue; }
      phex(c.isk, 32); printf(" ");
      uv_cpace_confirm(&c, true, t); phex(t, 32); printf(" ");
      uv_cpace_confirm(&c, false, t); phex(t, 32); printf(" ");
      uv_cpace_data_key(&c, t); phex(t, 32); printf(" ");
      uv_cpace_commit(&c, hfp, t); phex(t, 32); printf("\n");
    } else if (!strcmp(cmd, "aead")) {
      uint8_t key[32], nonce[24];
      unhex(a[0], key, 32); unhex(a[1], nonce, 24);
      size_t adl = unhex(a[2], buf_b, sizeof(buf_b)), pl = unhex(a[3], buf_c, sizeof(buf_c));
      size_t n = uv_aead_lock(buf_d, key, nonce, buf_b, adl, buf_c, pl);
      phex(buf_d, n); printf("\n");
    } else if (!strcmp(cmd, "sigcheck")) {
      uint8_t pk[32], type, ch[UV_CHALLENGE], sig[UV_SIG];
      unhex(a[0], pk, 32); unhex(a[1], &type, 1); unhex(a[2], ch, UV_CHALLENGE);
      size_t pl = unhex(a[3], buf_b, sizeof(buf_b));
      unhex(a[4], sig, UV_SIG);
      printf("%s\n", uv_cmd_verify(pk, type, ch, buf_b, pl, sig) ? "ok" : "bad");
    } else if (!strcmp(cmd, "frag")) {
      uint8_t type;
      unhex(a[0], &type, 1);
      size_t n = unhex(a[1], buf_b, sizeof(buf_b));
      for (int i = 0; i < uv_frag_count(n); i++) {
        uint8_t pkt[UV_MTU];
        size_t pl = uv_frag_packet(pkt, type, buf_b, n, i);
        phex(pkt, pl); printf(" ");
      }
      printf("\n");
    } else if (!strcmp(cmd, "reasm")) {
      static uv_reasm_t r;
      uv_reasm_init(&r);
      bool done = false;
      for (int i = 0; i < 8 && a[i]; i++) {
        size_t n = unhex(a[i], buf_e, sizeof(buf_e));
        done = uv_reasm_feed(&r, buf_e, n);
      }
      if (done) { printf("%02x ", r.type); phex(r.data, r.len); printf("\n"); }
      else printf("incomplete\n");
    } else {
      printf("err unknown\n");
    }
  }
  return 0;
}
