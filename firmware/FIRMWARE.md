# Drone firmware: variants and changes

*Updated 2026-10-29. Every variant is described with all its changes from the official Bitcraze firmware, the reason for each change and where its code is. Paths `src/...` refer to the sibling repository `crazyflie-firmware` (base commit in `patches/BASE_COMMIT.txt`, `2026.04 +34`).*

## 1. The variants

| Variant | What it does | Built in | Binary | When to use it |
|---|---|---|---|---|
| **FW0** — official | Bitcraze firmware, unchanged | — | — | Never on the rig: it does not see the home-made SD deck |
| **FW1** — `board-selftest` | SD deck forced + board fingerprint at every boot (accelerometer self-test, bias, scale) exposed as parameters `uavibe.*` | `crazyflie-firmware/` (`make`) | `crazyflie-firmware/build/cf2.bin` | Every recording made with the dashboard (enrollment, test, swap, board, boot, propellers, payload) |
| **FW2** — `uavibe-key` | FW1 + an app that runs the boot protocol on board, rebuilds the password with the fuzzy extractor and runs CPace with the station | `firmware/uavibe_key/` (`make`) | `firmware/uavibe_key/build/cf2.bin` | Dashboard tab **📡 On-board key** (real-time authentication) |

FW2 contains FW1: SD recordings keep working with FW2, so a drone can stay on FW2. FW1 remains the reference variant for the data of the analyses.

### Switching variant

The two variants are built in different folders and do not overwrite each other. To switch, flash the other binary (drone in bootloader mode: hold the power button ~3 s, **with only that drone powered on**):

```bash
# in WSL, from the folder that contains both repositories
cd crazyflie-firmware && make -j8 && make cload              # FW1
cd UAVibe/firmware/uavibe_key && make -j8 && make cload      # FW2
```

The FW2 helper data (key `uvkey/helper` in the EEPROM) survive a change of firmware: back on FW2 the drone is still enrolled. They are erased by the station's signed `ERASE` message (button "Restart enrollment" in the tab, §3.8).

---

## 2. FW1 — `board-selftest`: changes

Full patch: `patches/fw1-board-selftest.patch` (apply with `git apply --ignore-whitespace` on the base commit). Configuration: `patches/fw1.config`.

### 2.1 Configuration (Kconfig)

| Option | Value | Default | Why |
|---|---|---|---|
| `CONFIG_DECK_FORCE` | `"bcUSD"` | `"none"` | The SD deck is home-made and has no OneWire memory with its identifier: without forcing it the firmware does not initialise it |

It is the only option that differs from the default configuration (checked on 2026-10-14 by comparing `build/.config` with `alldefconfig`). Note: `CONFIG_DECK_USD_USE_ALT_PINS_AND_SPI` is **not** enabled in the build in use, although earlier notes said it was needed; the deck works on the standard pins.

### 2.2 Accelerometer self-test at every boot (board fingerprint, MEMS-PUF style)

| File | Change |
|---|---|
| `src/drivers/bosch/src/bmi088_accel.c` | Two global variables `bmi088_accel_selftest_pos/neg` keep the raw readings (LSB, 24 g full scale) of the last self-test, which the Bosch driver only used for a pass/fail result |
| `src/drivers/bosch/interface/bmi088.h` | `extern` declaration of the two variables |
| `src/hal/src/sensors_bmi088_bmp3xx.c` | In `sensorsDeviceInit()`, **before** the normal accelerometer set-up: power-up, `bmi088_accel_init`, self-test with up to 3 attempts, plausibility check; then the original code initialises the accelerometer as always. About +0.35 s at boot |
| same file | Read-only parameter group `uavibe`: `stDone` (attempts used), `stResult` (0 = valid), `stPosX..Z`, `stNegX..Z`, `gBiasX..Z` and `gBiasOk` (gyroscope bias computed at boot), `accScale` and `accScaleOk` (accelerometer scale computed at boot) |

**Why the retries and the plausibility check.** The Bosch pass/fail check is not enough for a fingerprint. On 2026-10-07, one boot out of four on drone A returned identical positive and negative readings (not updated). On drone D, identical readings once counted as "passed", and once the test ran at the 6 g full scale instead of 24 g (values ÷4). An attempt counts as valid only if pos ≠ neg on every axis and |pos.x| + |neg.x| > 15000 LSB (≈ 25–28k on the four drones at 24 g, ≈ 6.5k at 6 g).

**Why bias and scale as parameters.** The official firmware computes them at boot and subtracts/normalises them, so they never reach the logs. Exposing them only costs the parameter declarations (no new logic).

### 2.3 Configuration block in EEPROM ignored (to be checked)

| File | Change |
|---|---|
| `src/utils/src/configblockeeprom.c` | `if (cb_ok == false)` → `if (cb_ok == true)`: the configuration block stored in EEPROM (radio channel and address, trims) is **always** replaced by the defaults |

This change was not introduced by the key work. It probably served to bring every drone back to the default radio channel and address. To be confirmed: if it is no longer needed it can go, because it prevents storing per-drone radio settings.

### 2.4 Changes to the build scripts (not to the firmware)

`scripts/kconfig/lxdialog/check-lxdialog.sh`: `main() {}` → `int main(void) { return 0; }`. The recent Ubuntu gcc (WSL) rejects the implicit declaration and `make menuconfig` would not start. The other files git marks as modified only changed permissions or line endings (working copy shared between Windows and WSL): no content difference.

---

## 3. FW2 — `uavibe-key`: changes

FW2 = FW1 + an out-of-tree app (folder `firmware/uavibe_key/`, built against `crazyflie-firmware`), plus one weak hook in the stabilizer (§3.3, `patches/fw2-stabilizer-hook.patch`).

### 3.1 Configuration (`uavibe_key/app-config`)

| Option | Value | Why |
|---|---|---|
| `CONFIG_APP_ENABLE` | `y` | Application layer of the firmware (task `appMain`) |
| `CONFIG_APP_PRIORITY` | `1` | Same priority as the SD logging task, used only while sampling (§3.3) |
| `CONFIG_APP_STACKSIZE` | `1200` words (4.8 KB) | Computation and cryptography run in the app task (700 words in the first version, §3.3) |
| `CONFIG_DECK_FORCE` | `"bcUSD"` | As in FW1 |
| `CONFIG_DECK_LOCO`, `CONFIG_DECK_LIGHTHOUSE` | disabled | Positioning decks never used on the rig: they free ~14 KB of CCM and some RAM for the app's buffers |

### 3.2 The code

| File | Content |
|---|---|
| `src/uv_app.c` | The only Crazyflie-specific part: task, app channel, motors (`motorPowerSet`), synchronous sampling through `uvStabilizerHook`, reading of the FW1 parameters `uavibe.*`, EEPROM (`storage`), hardware random generator of the STM32F405, verification of the station's signed commands, link timeout. Parameter `uvkey.enrolled` (read-only); log `uvkey.{state, phase, result, wrong, alarm, saturated, runs, stackFree, computeMs, dropped}` |
| `src/core/uv_features.c` | Quantities of the boot protocol (same definitions as `auth_client/keyproto/boot.py`): rest statistics, spectrum of each hold (2048-point real FFT, Hann window, one-sided density), motor speed and broadband floor, board quantities with temperature and gravity compensation |
| `src/core/uv_fe.c` | Fuzzy extractor, drone side: Rep (quantisation, Reed–Solomon RS[40,36] over GF(65537) with Berlekamp–Welch decoding; **no robustness tag since 2026-10-29**, §3.7), helper fingerprint (`uv_helper_fp`) and helper refresh (offsets centred on the median of the last 3 accepted boots); speed alarm (`uv_speed_shift`, §3.6) and plaintext of the `DATA` frame (`uv_data_pack`) |
| `src/core/uv_crypto.c` | BLAKE2b hashing, CPace on X25519, XChaCha20-Poly1305 frame, Ed25519 check of the station's commands (`uv_cmd_verify`), fragmentation of messages into 30-byte packets |
| `src/core/uv_sysparams.h` | System parameters (bin widths, temperature slopes, t, protocol timings, motor model, K and speed-alarm threshold, digest), **generated** by `tools/gen_sysparams.py` from `auth_client/data/keystore/system.json` |
| `src/core/uv_station_key.h` | Public key of the ground station (Ed25519), **generated** by `python -m keyproto.onboard_station keygen` (§3.8) |
| `src/vendor/monocypher/` | Monocypher 4.0.2 (BSD-2 / CC0), unchanged: X25519, Elligator 2, BLAKE2b, XChaCha20-Poly1305, and the optional `monocypher-ed25519.c` (Ed25519 with SHA-512); provenance in `VERSION.txt` |

`src/core/` depends neither on FreeRTOS nor on the Crazyflie: it builds unchanged on the PC (`test/host_test.c`) and is compared with the Python reference byte for byte.

### 3.3 Differences from the offline analysis, and why

| Offline (`keyproto/boot.py`) | On board | Reason | Measured effect (§5) |
|---|---|---|---|
| 5 Hz high-pass (Butterworth, order 4) | No high-pass, the mean is removed | RAM | None on the floor (read above 20 Hz) |
| float64 samples from the SD log | int16 centred on the rest mean, LSB 2⁻¹⁶ g and 2⁻¹⁰ °/s | RAM: one hold = 24 KB | Quantisation noise 19–30 dB below the lowest floor; 0 saturations |
| fs = 1017 Hz (median of the session's timestamps) | One sample per stabilizer cycle (as the SD log), fs assumed 1000 Hz | The drone has no better reference; sampling on the tick instead of the cycle distorted the line (synchronous sampling, below) | Constant scale on the gain (−1.5 %), the same at enrollment and at every boot |
| Longest gap-free stretch of the SD log | 2048 consecutive samples in RAM | On board there are no SD-card stalls | — |
| SHA-256, HKDF, JSON | BLAKE2b, binary format | Monocypher, no JSON on the microcontroller | Same construction |
| SPAKE2 on edwards25519 | CPace on X25519 | Monocypher has no generic point addition; CPace is the balanced PAKE recommended by the CFRG | — |

**Found while porting.** The firmware low-passes the accelerometer and gyroscope (30 Hz and 80 Hz, `sensors_bmi088_bmp3xx.c`). The ~450 Hz motor line therefore arrives strongly attenuated, **below one sensor LSB**, and survives only because the filter output is floating point. A first version with the sensor's LSB (0.49 mg) wiped it out and found random peaks; hence the fine scale centred on the rest mean.

**First test on the drone (2026-10-18): reset after M1.** The drone reset right after the hold of M1, and at the next boot the self-test failed (`watchdogNormalStart [FAIL]`, red LED of M1 blinking in groups). Two suspects were fixed first. (1) Watchdog: the Crazyflie feeds its hardware watchdog (timeout 100–350 ms, `watchdog.c`) only from the FreeRTOS idle task, and the feature computation of a hold (12 FFTs of 2048 points with medians, partly in software double precision) is longer than that; `uv_yield()` (one tick of `vTaskDelay(1)`) now runs after each channel of a hold, after each column of the Reed–Solomon solver and around the X25519 multiplications (empty on the PC, so results do not change). (2) Stack overflow of the app task (700 words): the median used the recursive libc `qsort`; it is now an iterative quickselect (identical values, C–Python tests unchanged), the stack is 1200 words, and the log variable `uvkey.stackFree` reports the least free stack. To read the exact reason of a reset: `python tools/fw_why_reset.py` in `auth_client`, before power-cycling (it prints the assert snapshot: overflowing task, file and line, or the registers of a hard fault). **The real cause**, read with that tool: `Assert failed at src/hal/src/radiolink.c:171`, the queue of incoming radio packets (5 slots) was full. The tasks that drain the link's service packets and the app channel (`CRTP_SRV_TASK_PRI`, `PLATFORM_SRV_TASK_PRI`) and the idle task run at priority **0**; the computation, at priority 1, starved them. Fix: the app stays at priority 1 only while sampling, where it sleeps every millisecond like the SD logger; features, Rep and CPace run at priority 0 (`vTaskPrioritySet`, `sampling_mode`/`compute_mode` in `uv_app.c`), sharing the CPU round-robin with those tasks. Log variable `uvkey.computeMs`: longest computation of one hold, must stay below the 1 s pause.

**Synchronous sampling (2026-10-18, after the first 3 on-board enrollment boots on D).** With the fixes above the protocol completed, but the speeds were wrong and unstable: M1 1.10–1.12 on board against 1.05 in an SD boot recorded right after on the same drone; M2 and M3 jumped by 5–11 % from one boot to the next. Cause: the IMU drives the stabilizer at its own clock, ~1005–1013 Hz, while the app read the latest sample at every FreeRTOS tick (1000 Hz), skipping one sample every ~100; the motor line, close to Nyquist, then moves by a few Hz at random. Reproduced in simulation on the same SD data: speeds change by up to 8–13 %. Fix: a weak, empty function `uvStabilizerHook()`, called at every stabilizer cycle at the point where the SD deck logs synchronously (`src/modules/src/stabilizer.c`, patch `patches/fw2-stabilizer-hook.patch`; it does nothing in FW1). The app overrides it: while armed it copies one sample (acc, gyro, vbat) into a 64-slot ring that the measurement loop drains. The app now measures exactly the samples that end up in the SD logs, and rest and hold durations are counted in samples as in the analysis. Log variable `uvkey.dropped`: samples lost with the ring full, must stay 0. The 3 enrollment boots made before this fix were discarded and the drone enrolled again.

**Helper lost after the first successful boot (2026-10-18).** The helper created by the station has no history (328 B, 360 B at the time); after the first refresh on board it has one (488 B, up to 808 B with 3 boots; then 520 and 840 B). The app packed it into the 420-B message buffer, overflowing it, and at the next power-up read it back truncated, so the drone reported "not enrolled". Fix: a dedicated buffer `helperbuf`, in normal RAM rather than CCM. None of these problems could be seen on the PC, where the C code was checked only for its results, not for timing and stack on a microcontroller.

### 3.4 Memory (2026-10-29)

| | FW1 | FW2 |
|---|---|---|
| Flash | 297 KB | 317 KB (31 % of 1 MB) |
| RAM (128 KB) | 93 KB | 121 KB, **9.9 KB free** |
| CCM (64 KB) | 65 KB (full) | 62 KB, 3.5 KB free |

Split by origin (2026-11-12, lab build, KiB = 1024 B; `python firmware/tools/memory_figure.py` reads `uavibe_key/build/cf2.map` and writes `tools/memory_split.json` and thesis fig. 20): flash stock 269.1 / ours 40.0 / free 698.9 of 1008 KiB; main RAM stock 89.9 / ours 28.4 / free 9.6 of 128; CCM stock 49.0 / ours 11.6 / free 3.4 of 64. FW1 from its map: flash 297.5 KiB, CCM 63.4 KiB. Service build (`UV_RESEARCH=0`): flash 308.5 KiB, RAM unchanged. Thesis fig. 20 is drawn from the service build's map (`--map=`): ours 39.8 / stock 268.7 / free 699.5 KiB of flash; the thesis presents that build as FW2 and does not mention the lab build (user's decision, 2026-11-12). The table above uses decimal KB.

App buffers: 24 KB of samples of one hold in RAM (reused by the Reed–Solomon decoder); FFT, search band, helper data, refreshed helper waiting for `COMMIT` and messages in CCM; 0.85 KB in RAM for the message checked against the station's signature.

### 3.5 Protocol between station and drone (app channel, 30-byte packets)

Every message: byte 0 = type, byte 1 = fragment index (high 4 bits) | number of fragments (low 4 bits), then up to 28 bytes of data. Reference implementation: `auth_client/keyproto/onboard.py`, station `auth_client/keyproto/onboard_station.py`.

| Type | Direction | Content |
|---|---|---|
| `HELLO` 0x01 / `STATUS` 0x81 | S→D / D→S | state, enrolled yes/no, n, t, drone name, **challenge** (16 random bytes, new at every `STATUS`, valid for one signed command) |
| `START` 0x02 | S→D | station sid (16 B) + **station's Ed25519 signature** (64 B, §3.8). **The drone runs the protocol: the motors spin** |
| `KEEPALIVE` 0x0B | S→D | empty, every second during the boot: after 3 s without any packet from the station the drone stops the motors and gives up (`FAIL` 11) |
| `PROGRESS` 0x82 | D→S | phase, motor, percent |
| `FEATURES` 0x83 | D→S | 40 float32 — **only while the drone is not enrolled** (enrollment) |
| `ENROLL_HELPER` 0x05 / `HELPER_OK` 0x88 | S→D / D→S | helper data (328 B, v3) + station's signature (64 B); accepted only while the drone is not enrolled |
| `ERASE` 0x0A / `HELPER_OK` | S→D / D→S | station's signature (64 B): the drone erases its helper data (new enrollment). Replaces the parameter `uvkey.erase`, removed because anyone in radio range could set it |
| `GET_HELPER` 0x07 / `HELPER_PART` 0x89 | S→D / D→S | **Security test only** (memory read-out; built only with `UV_SECURITY_TESTS`=1, the default; `-DUV_SECURITY_TESTS=0` for a drone in service): offset u16 → total u16, offset u16, up to 400 B of the helper (always v3, up to 808 B with the history, more than one 420-B message) |
| `PUT_HELPER` 0x08 / `HELPER_OK` | S→D / D→S | **Security test only** (cloning): total u16, offset u16, part; accepted only while the drone is not enrolled (signed `ERASE` first); `HELPER_OK` 2 = part received, 1 = complete and loaded |
| `CPACE_D` 0x84 | D→S | drone sid (16 B) + drone's Y (32 B) + fingerprint of the helper it used (32 B) |
| `CPACE_S` 0x03 | S→D | station's Y (32 B) |
| `CONFIRM_D` 0x85 / `CONFIRM_S` 0x04 | | key confirmations (32 B) |
| `DATA` 0x86 | D→S | nonce (24 B) + ciphertext + MAC (16 B). Plaintext: "UVD2", name, corrected symbols, speed-alarm mask, refreshed yes/no, speed shift of the 4 motors (int16, hundredths of σ), fingerprint of the refreshed helper (32 B; the current one if not refreshed), text |
| `COMMIT` 0x09 | S→D | BLAKE2b keyed with ISK of ("commit", new fingerprint) (32 B): only after checking it does the drone store the refreshed helper in EEPROM |
| `FAIL` 0x87 | D→S | reason, wrong symbols (reasons in `keyproto/onboard.py` `FAIL_REASON`; 10 = command without a valid signature, 11 = link lost) |
| `ABORT` 0x06 | S→D | stops (motors off); not signed, since it can only stop |

CPace: generator g = Elligator2(BLAKE2b("CPace-UAVibe-v2-gen", password, sid, name, helper fingerprint)) (v1, until 2026-10-28, without the fingerprint); Y = X25519(y, g); K = X25519(y, Y_peer); ISK = BLAKE2b("CPace-UAVibe-v2-isk", sid, K, Y_drone, Y_station); confirmations with BLAKE2b keyed with ISK; data key = BLAKE2b(ISK, "data"). Every field is preceded by its 2-byte length. The structure follows draft-irtf-cfrg-cpace but it is **not** byte-compatible with its test vectors. The constants keep the project's former name "UAVibe".

### 3.6 Speed alarm (added 2026-10-14; denies access since 2026-10-18)

**Why.** Each motor weighs one symbol out of 40 and the code corrects t = 2. Changing the propellers (tests of 2026-10-14 on drone D) moves the motor's speed by 8–11 %, i.e. 9–12 σ, while a normal boot moves at most 3.1 σ (14 boot-vs-reference comparisons on the 26 recorded boots). Yet the key is still regenerated, so the key alone does not reveal tampering with propellers or motors.

**What it does.** After a successful Rep the drone knows the corrected symbols W. For each motor it computes the speed shift from the helper's reference (the centre of the enrolled bin, i.e. the median of the last 3 boots): K·(offset − offset that would centre today's measurement)/w, in σ. It works also when the symbol changed and was corrected. Beyond **±5 σ** (`UV_SPEED_ALARM_SIGMA`, from `keyproto/fe.py` `SPEED_ALARM_SIGMA`) the alarm goes off:
- access is **denied** (threat model agreed with the supervisor on 2026-10-15: altered mechanics = no longer the enrolled drone). The CPace exchange still completes, because the key is right, so the outcome travels authenticated in the encrypted `DATA` frame (text "…SPEED ALARM: access denied"). The station knows it is that drone, tampered with; it discards the session key and records the alarm (`speed_alarm`, counter `alarms` in `station.json`). It does not count it as a failed attempt, since no password was guessed;
- the drone does **not** refresh its helper, so the reference does not follow the tampering and the alarm repeats at every boot until the drone is enrolled again (legitimate maintenance, such as a propeller change, needs a new enrollment).

Without an alarm the refresh works as before, and it is what follows ageing: the reference moves with the median of the last 3 accepted boots, so slow drift never accumulates up to the threshold. No new data are needed: the reference is already in the helper, and the system-parameter digest does not change (K and the threshold are outside `sysparams_bytes`). Log variables `uvkey.alarm` (mask of the last boot with a regenerated key) and `uvkey.result` = 8 after an access denied by the alarm.

**Threshold.** 5 σ lies between the largest normal boot (3.1 σ offline, 2.6 σ with the on-board quantities) and the smallest propeller change (8.8 σ). It is tuned on a single modified drone: the false-alarm rate can only be estimated with more boots (`python -m analysis.onboard_log`).

### 3.7 No robustness tag: helper fingerprint kept by the station (2026-10-29)

**Why.** Until 2026-10-28 the helper contained a tag BLAKE2b(W, helper), with which the drone detected modified helper data (a *robust* fuzzy extractor). But whoever reads the memory gets both the tag and the rest of the helper: they can test values of W at home against the tag, without ever talking to the station, and the limit of 5 attempts is of no use. Against a memory reader the security fell to the uncertainty of W given the helper.

**What changed.**
- The helper (format v3, magic "UVH3") no longer has the tag. Rep no longer checks the helper's integrity; the password is unchanged (BLAKE2b(W, sketch)).
- The **station** keeps the **fingerprint** of the drone's current helper, BLAKE2b("uavibe-helper-v3", parameter digest, v3 helper), and the previous one. The fingerprint does not depend on W, so it is no oracle.
- The drone sends the fingerprint of the helper it used in `CPACE_D` and puts it into the **CPace generator**. The station refuses a fingerprint that is neither the current nor the previous one (a failed attempt); if an attacker in the middle replaced it with the right one, the exchange would fail anyway. A helper whose offsets were nudged, for example to move the reference of the speed alarm, is therefore refused even when it still gives the right password.
- After a refresh the drone keeps the new helper in RAM (`pending`), sends its fingerprint in the encrypted `DATA` frame and stores it in EEPROM only after the station's `COMMIT`. The station records the new fingerprint (keeping the one just used as the previous) before sending `COMMIT`: if `COMMIT` is lost, the drone keeps the old helper, which the station still accepts as the previous one (`uvkey.result` = 9).

**Migration of drones already enrolled.** The firmware still reads a v2 helper (with the tag), drops the tag and stores it again as v3 at power-up; the password does not change, so no new enrollment is needed. The station, however, has no fingerprint yet: it is registered once from tab 📡, section 4, "Register the helper fingerprint" (`Station.register_helper_fp`: reads the helper from the drone and stores its fingerprint). It is trust on first use, so it belongs in the lab, like enrollment.

**Cost.** Coarser diagnostics: modified helper data no longer give "tag not valid" on the drone, but a refusal by the station ("helper data not recognised"). The PC prototype (`keyproto/fe.py`) keeps a `robust=True` mode, used by the analyses of reports 9–11 so that they reproduce as written; the identity tab and the drone use the fingerprint.

### 3.8 Commands signed by the station (2026-10-29)

**Why.** `START` makes the motors spin and arrives before any authentication: anyone in radio range could start the protocol, and with `uvkey.erase` erase the enrollment.

**What it does.** The station has an Ed25519 key pair (`auth_client/data/keystore/onboard/station_key.json`, secret); the public key is compiled into the firmware (`src/core/uv_station_key.h`, written by `python -m keyproto.onboard_station keygen`). At every `STATUS` the drone sends a new 16-byte challenge. `START`, `ENROLL_HELPER` and `ERASE` end with the station's signature over "UVCMD1" | type | challenge | content: the drone checks it (`uv_cmd_verify`, Monocypher `crypto_ed25519_check` from the optional `monocypher-ed25519.c` of the same 4.0.2 release) and uses the challenge only once, so a recorded command sent again is ignored (`FAIL` 10). The station always sends `HELLO` before a signed command, and `hello()` skips messages left over from an earlier exchange.

**Checked by.** `test_onboard.py`: the C check agrees with PyNaCl (right signature accepted; changed signature, challenge or type refused). `test_onboard_station.py`: another key neither starts the motors nor erases, a resent `START` is ignored, a station without a key cannot command.

**After the change.** `keygen` once (done on 2026-10-29), then `make` and `make cload` on every drone. A drone with older firmware sends no challenge: the station says so and sends it no command.

### 3.9 Board values not readable over the radio in a service build (2026-11-11)

**Why.** FW1 exposes the board fingerprint (self-test readings, gyroscope bias, accelerometer scale) as the read-only param group `uavibe`, which the dashboard and `auto_bench.py` read. Params can be read by anyone in radio range, so on a drone in service the 12 board quantities, a large part of the reading $W$ the key comes from, would be public, and the only part of $W$ an attacker on the link could not learn would be the motor quantities.

**What changed.**
- FW1 patch: the sensor driver offers `uvBoardFingerprintGet()` (`sensors_bmi088_bmp3xx.h`), and the `uavibe` param group is compiled only when `UV_HIDE_BOARD_PARAMS` is **not** defined.
- FW2 reads the board values through `uvBoardFingerprintGet()` instead of the params (`measure()` and the check before `START` in `uv_app.c`).
- FW2 build switch: `make` (= `UV_RESEARCH=1`, default) keeps the params and the GET/PUT_HELPER test messages for the lab; `make clean && make UV_RESEARCH=0` builds the service variant without either (`-DUV_HIDE_BOARD_PARAMS -DUV_SECURITY_TESTS=0`).

**Checked.** Service build: no `uavibe` param names in the binary, flash 316 KB, RAM 9.9 KB free. Lab build and FW1: params present as before. Python and C tests unchanged (the change touches only the firmware glue). Not yet flashed on a drone.

**Note.** The bench script and the dashboard need the lab build: the service build gives them no board values (the bench script uses them to check that the drone rebooted).

---

## 4. Enrollment: what station and drone exchange

| What | Who holds it | Secret? | When |
|---|---|---|---|
| System parameters (widths, temperature slopes, t, protocol, motor model) | Firmware (`uv_sysparams.h`) and station (`system.json`) | No, the same for the whole fleet | Build |
| Station's public key | Firmware (`uv_station_key.h`) | No | Build |
| Measurements of the 3 enrollment boots | Drone → station | **Yes** (they give W) | Only during enrollment, in a protected setting |
| Helper data (offsets, sketch, name; 328 B, up to 808 B with the history) | Station → drone EEPROM | No | End of enrollment; then refreshed by the drone after every accepted boot |
| Fingerprint of the current and previous helper | Station | No (it does not depend on W) | End of enrollment; then at every confirmed refresh |
| Password (32 B) | Station only | Yes | End of enrollment |
| Station's signing key | Station only | Yes | `keygen`, once |
| Failed-attempt counter | Station | — | Always |

Procedure: drone on the rig, not enrolled. Three boots (power-cycle each time, because the board part is measured at power-up), then the station takes the median of the three measurements, runs Gen, sends the helper and keeps only the password and the fingerprint. From then on the drone no longer sends its measurements: it answers `START` with the key exchange. To enroll it again its helper must be erased (signed `ERASE`).

In the prototype the station keeps the enrollment measurements in `data/keystore/onboard/research/` for the thesis (comparison with the SD logs). A real station would delete them after Gen.

---

## 5. Verification

| What | How | Result |
|---|---|---|
| C equal to Python | `auth_client/tests/test_onboard.py`: builds `src/core` on the PC (WSL) and compares with `keyproto/onboard.py` | Elligator, CPace (Y, ISK, confirmations, data key, commit), AEAD, framing, Rep (password, W, corrected symbols), helper refresh and fingerprint, Ed25519 check: **identical byte for byte**; quantities on the real boots within 1 % of σ |
| Station protocol | `tests/test_onboard_station.py`: the station against the Python twin of the firmware, with the on-board quantities of the 22 real boots | Enrollment, genuine boots accepted, other drones' measurements rejected, wrong name rejected, lock after 5 attempts, fingerprint following the refreshes, lost `COMMIT`, nudged offsets refused, old copy refused, v2 migration, unsigned/replayed commands ignored |
| On-board vs offline quantities | `python -m analysis.onboard_eval` (`analysis/results_onboard/`) | Within-drone stability: gain 0.42 σ on board vs 0.70 offline, floor 0.67 vs 0.60, board identical. Key (t = 2, every choice of 3 enrollment boots, no refresh): **genuine rejections 12 % vs 10 %, impostors 0 vs 0**; 0 saturated samples in 86 holds |
| Speed alarm | `test_onboard.py` (C vs Python on the `DATA` plaintext), `test_onboard_station.py`, `python -m analysis.conditions` | Plaintext identical byte for byte. Propellers changed on D (simulated chain with the on-board quantities): M1 only → key rejected (too many symbols); all 4 → key regenerated but **alarm on M1 and M4**, access denied, helper not refreshed; 25 g payload → no alarm |
| Firmware build | `make` in `firmware/uavibe_key` | No warnings (`-Werror`) |
| **On the real drones (2026-10-18/19)** | Tab 📡 On-board key; log analysed by `python -m analysis.onboard_log` | A, B, D enrolled on board; genuine authentications accepted (0–2 corrected symbols, speed shifts ≤ 0.9 σ, 38 s each); B carrying A's helper rejected 3/3; tape on the blades rejected in 5 of 6 trials; a different propeller on M4: key regenerated, speed alarm at −11.6 σ, access denied. The changes of 2026-10-29 (§3.7, §3.8, link timeout) are not tried on the drones yet |

## 6. Limits to declare

- The changes of 2026-10-29 (no tag and fingerprint, signed commands, link timeout) are tested on the PC only; on the drones, timing, stack use (`uvkey.stackFree`) and the Ed25519 check must still be verified.
- If the radio drops during the protocol, the drone stops the motors after 3 s without packets from the station (`KEEPALIVE` every second), also in the middle of a ramp or a hold; before 2026-10-29 it completed the ~38 s on its own.
- The drone refuses `START` until the firmware has computed the gyroscope bias and the accelerometer scale (`uavibe.gBiasOk`, `accScaleOk`: a few seconds still after power-up).
- `ABORT` is not authenticated (it can only stop the motors), nor are the test commands `GET_HELPER`/`PUT_HELPER` (the first reads public data, the second writes only on a drone already erased with a signed `ERASE`); they must be compiled out for service (`UV_SECURITY_TESTS`=0).
- The drone tells the station when the fuzzy extractor fails (`FAIL`): useful for diagnosis, it reveals only the outcome.
- The computation of the quantities is not constant-time (not a requirement: the measurements are not secret in themselves, but W derives from them; a side-channel attack needs physical access during the boot).
- Randomness: hardware generator of the STM32F405, passed through BLAKE2b with a timer value; not certified.
- CPace with our own encodings, not those of the CFRG draft; balanced PAKE (a compromised station reveals the passwords; an augmented PAKE such as OPAQUE would not).
