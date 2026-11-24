# keyproto/

Python implementation of the UAVibes key-generation and authentication protocol. All modules have a matching test in `tests/`; `fe.py` and the core of `onboard.py` are verified byte-for-byte against the firmware C implementation (`firmware/uavibe_key/`).

## End-to-end demo

```bash
python -m keyproto.demo
```

Runs a complete enrollment + authentication cycle with four simulated drones, prints the generated keys and the authentication result for each.

## Modules

### `fe.py` — Fuzzy extractor

Shielding-offset quantiser + Reed–Solomon sketch over GF(65537), with a station-side helper fingerprint.

```python
from keyproto.fe import FuzzyExtractor

fe = FuzzyExtractor(n=40, t=2, K=8)      # 40 features, correct up to 2 wrong symbols, 8-bin quantiser

# Enrollment (run once per drone, store helper)
W_enroll = feature_vector_from_boot_session()
key, helper = fe.gen(W_enroll)

# Authentication (every boot)
W_boot = feature_vector_from_boot_session()
key_rep, tag_ok = fe.rep(W_boot, helper)
if key_rep is None:
    print("too many wrong symbols — rejected")
```

**Parameters** (see `fe.py` for defaults):
- `n` — number of features (40: 28 motor + 12 board)
- `t` — maximum correctable wrong symbols (2)
- `K` — number of quantisation bins per feature (8)

### `onboard.py` — On-device protocol reference

Byte-exact Python twin of the firmware C core (`firmware/uavibe_key/src/core/`). Used by tests and by the simulation mode of the dashboard's **Chiave a bordo** tab.

```python
from keyproto.onboard import OnboardProtocol, gen_sysparams

params = gen_sysparams()                    # or load from data/keystore/system.json
proto  = OnboardProtocol(params)

# Simulate what the drone computes from a boot session
features = proto.extract_features(boot_csv_path)
key, helper = proto.gen(features)
```

### `onboard_station.py` — Ground station (FW2)

Manages the station database for the on-device protocol: enrollment, CPace exchange, helper refresh, speed-alarm check, and attack tests.

```python
from keyproto.onboard_station import Station

station = Station("data/keystore/onboard/station.json", params)
station.enroll("droneA", features_from_three_boots)

result = station.run(drone_connection)      # drives a full CPace exchange over the radio
# result.granted == True if key matched and no speed alarm
```

### `boot.py` — PC-side boot key (FW1 path)

Extracts features from recorded boot sessions (SD-card CSVs) and runs the PC-side key exchange (SPAKE2). Used by the dashboard's **Verifica identità** tab.

```python
from keyproto.boot import BootKey

bk = BootKey("data/keystore/system.json", "data/keystore/station.json")
bk.enroll("droneA", boot_session_paths[:3])         # three boot sessions

result = bk.authenticate("droneA", boot_session_path)
print(result.granted, result.wrong_symbols, result.speed_shifts)
```

### `spake2.py` — SPAKE2 key exchange

SPAKE2 on edwards25519 (via PyNaCl) with key confirmation. Used by the PC-side path (`boot.py`); the on-device path uses CPace (`onboard.py`).

```python
from keyproto.spake2 import SPAKE2Station, SPAKE2Drone

password = b"shared-secret"
drone_msg, drone_state  = SPAKE2Drone.start(password)
station_msg, session_key = SPAKE2Station.respond(password, drone_msg)
session_key2 = SPAKE2Drone.finish(drone_state, station_msg)
assert session_key == session_key2
```

## Security notes

- The helper data stored on the drone is **public by design**: it carries no information about the key beyond what any observer of the drone's vibration could already measure.
- The station-side helper fingerprint binds the helper to the specific drone and system parameters; a modified helper is rejected before any CPace exchange takes place.
- The speed alarm fires after a successful key exchange if any motor's speed shifted by more than 5σ from the enrolled reference — a successful key regeneration does not by itself grant access.
- `data/keystore/onboard/station_key.json` holds the station's Ed25519 **private** signing key and must never be committed or shared. The corresponding public key is compiled into the firmware (`firmware/uavibe_key/src/core/uv_station_key.h`).
