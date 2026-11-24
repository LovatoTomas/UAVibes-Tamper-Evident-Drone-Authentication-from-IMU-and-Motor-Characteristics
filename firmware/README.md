# firmware/

Firmware patches and on-device C core for the UAVibes key application. Full documentation of every variant, modification, and known issue is in `FIRMWARE.md`.

## Variants

| Variant | Based on | Purpose |
|---------|----------|---------|
| **FW1** | Crazyflie 2.1 stock | Exposes IMU self-test results at boot; enables custom SD deck |
| **FW2** | FW1 | Adds the full UAVibes key application: boot protocol, fuzzy extractor, CPace on the drone |

## Applying the patches (FW1 and FW2)

You need the `crazyflie-firmware` repository and an ARM cross-compiler.

```bash
# 1. Clone the official firmware at the tested base commit
git clone https://github.com/bitcraze/crazyflie-firmware
cd crazyflie-firmware
git checkout $(cat /path/to/UAVibes/firmware/patches/BASE_COMMIT.txt)

# 2. Apply FW1 patches (SD deck config + IMU self-test)
cp /path/to/UAVibes/firmware/patches/fw1.config .config
git apply /path/to/UAVibes/firmware/patches/fw1-board-selftest.patch

# 3. Build (in WSL or Linux)
sudo apt install make gcc-arm-none-eabi
make

# 4. Flash (drone in bootloader mode: hold power button ~3 s)
make cload
```

Verify before flashing:
```bash
grep CONFIG_DECK_FORCE build/.config    # must show bcUSD
```

## FW2: on-device key application

FW2 = FW1 + the out-of-tree app in `uavibe_key/`. It runs the full boot protocol on the drone: 10 s rest, four single-motor holds at 58 000 PWM, feature extraction, fuzzy extractor Rep, and CPace with the ground station.

```bash
# After applying FW1 patches:
git apply /path/to/UAVibes/firmware/patches/fw2-stabilizer-hook.patch

# Build the out-of-tree app
cd /path/to/UAVibes/firmware/uavibe_key
make                # builds against the crazyflie-firmware tree at CRAZYFLIE_BASE
make cload          # flash
```

Set `CRAZYFLIE_BASE` if the firmware tree is not at the default sibling path:
```bash
make CRAZYFLIE_BASE=/path/to/crazyflie-firmware
```

## `uavibe_key/` layout

```
uavibe_key/
├── Makefile
├── Kbuild / app-config
├── src/
│   ├── uv_app.c              # Crazyflie task: radio protocol, SD sampling hook
│   └── core/                 # Portable C core (no Crazyflie dependencies)
│       ├── uv_core.h         # Public API
│       ├── uv_features.c     # Feature extraction (gain, broadband floor, board params)
│       ├── uv_fe.c           # Fuzzy extractor: quantiser + Reed-Solomon over GF(65537)
│       ├── uv_crypto.c       # CPace on X25519/Elligator2, XChaCha20-Poly1305 framing
│       ├── uv_sysparams.h    # System parameters (quantiser offsets, RS code)
│       └── uv_station_key.h  # Station Ed25519 public key (compiled in)
└── src/vendor/monocypher/    # Monocypher 4.0.2 (X25519, Ed25519, ChaCha20-Poly1305)
```

The `core/` modules have no dependency on the Crazyflie SDK or FreeRTOS and are tested standalone against the Python reference implementation:

```bash
cd uavibe_key/test
make
./host_test        # or: wsl -e make && wsl -e ./host_test
```

## Generating a new station key

```bash
python -m keyproto.onboard_station keygen
# Writes:
#   data/keystore/onboard/station_key.json   (secret — never commit)
#   firmware/uavibe_key/src/core/uv_station_key.h  (public key compiled into firmware)
```

Reflash the firmware after regenerating the key.
