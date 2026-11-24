# UAVibes — Drone Physical Identity from IMU and Motor Characteristics

UAVibes is a proof-of-concept system that derives a cryptographic key from the physical characteristics of a drone's motors and inertial sensor (IMU), without ever storing the key on the device. At every power-up the drone measures its own vibration signature and IMU self-test response, reconstructs the key with a fuzzy extractor, and authenticates to a ground station with a password-authenticated key exchange (CPace). With the drone switched off, its memory holds only public helper data.

This repository accompanies the paper:

> *UAVibes: Tamper-Evident Drone Authentication from IMU and Motor Characteristics*  
> Anonymous Submission — USENIX Security 2027

---

## Repository layout

```
UAVibe-public/
├── README.md
├── requirements.txt          # Python dependencies
├── pytest.ini
├── reproduce.py              # regenerates every result and figure in the paper
├── run_dashboard.py          # launches the Streamlit dashboard
├── auto_bench.py             # unattended bench-test runner (requires a real drone)
├── main.py                   # legacy CLI enrollment runner
├── usd_convert.py            # converts raw SD-card binary logs to CSV
├── usd_motor_analysis.py     # motor speed analysis on SD recordings
├── rpm_calibration.py        # laser-tachometer calibration session
├── model_validation.py       # validates motor-model predictions
├── tap_test.py / tap_test_analysis.py  # tap-test tool (optional)
│
├── crazyflie/                # Crazyflie drone connection and telemetry
├── enrollment/               # data-collection protocol (micro-sweep, session manifest)
├── pipeline/                 # signal processing: preprocessing, features, motor model
├── analysis/                 # all study scripts and figure generators
├── keyproto/                 # fuzzy extractor, CPace key exchange, boot protocol
├── tests/                    # automated test suite
├── ui/                       # Streamlit dashboard (8 tabs)
│
├── firmware/                 # firmware patches and on-device C core
│   ├── FIRMWARE.md           # full description of every firmware variant and patch
│   ├── patches/              # FW1 and FW2 patches against crazyflie-firmware
│   └── uavibe_key/           # out-of-tree Crazyflie app (FW2): portable C core + build
│
└── data/
    └── README.md             # dataset structure and download link
```

---

## Installation

Python 3.10 or later is required.

```bash
pip install -r requirements.txt
```

The dashboard and analysis scripts do not require a physical drone; a real drone is needed only for `auto_bench.py` and for new data collection.

On **Windows**, `cflib` requires the **WinUSB** driver for the Crazyflie USB device and the CrazyRadio PA dongle (install with [Zadig](https://zadig.akeo.ie/)). The connection modules monkey-patch `cflib`'s device-finding routine to use `libusb-1.0`; this patch is applied automatically at import time.

---

## Reproducing all paper results

Download the dataset (see `data/README.md`) and place it under `data/`. Then run:

```bash
python reproduce.py           # all steps, reusing cached intermediate results
python reproduce.py --fresh   # also rebuilds derived caches
python reproduce.py --list    # print all available steps
python reproduce.py board key fig-key   # run specific steps by name
```

A full run takes roughly 4 hours on a modern laptop. Every figure and table in the paper is regenerated; output goes to `analysis/figures/` and `analysis/results_*/`.

---

## Streamlit dashboard

```bash
python run_dashboard.py       # opens http://localhost:8501
```

The dashboard provides eight tabs:
- **New Rig Session** — standard data-collection protocol for a bolted drone
- **Verifica identità** — enroll from recorded boots and test authentication
- **Chiave a bordo** — on-device key exchange (real drone or simulation)
- **Offline Analysis**, **Spectrogram**, **Motor Calibration**, **Resonances** — analysis and visualization of recordings

---

## Automated tests

```bash
cd <repo root>
python -m pytest
```

The test suite (~5 s) covers the fuzzy extractor, the key exchange, the session-manifest logic, the motor model, the boot protocol, and byte-exact agreement of the firmware C core with the Python reference implementation (requires WSL with `make` and `arm-none-eabi-gcc`).

---

## Firmware

`firmware/FIRMWARE.md` documents every variant and modification. Two variants are used:

- **FW1** — stock Crazyflie firmware with `CONFIG_DECK_FORCE=bcUSD` (enables the custom SD deck) and a patch that exposes IMU self-test results at boot.
- **FW2** — FW1 plus the out-of-tree key application (`firmware/uavibe_key/`), which runs the full boot protocol and CPace on the drone itself.

Build in WSL:
```bash
# FW1
cd crazyflie-firmware   # sibling repo
make

# FW2
cd firmware/uavibe_key
make
make cload              # flash (drone in bootloader mode)
```

---

## Modules

### `pipeline/`

Signal-processing building blocks, drone-agnostic (only the standard IMU CSV format is assumed):

| Module | Purpose |
|--------|---------|
| `preprocessing.py` | Butterworth highpass (>5 Hz); adaptive comb filter keyed on PWM→RPM |
| `features.py` | 120-D feature vector per 1 s window (variance, skewness, kurtosis, spectral shape, 13 LFCC, cross-axis coherence) |
| `motor_model.py` | Canonical PWM→RPM map (single source of truth); per-motor gain table |
| `manifold.py` | ManifoldModel: StandardScaler → UMAP → Mahalanobis authentication |
| `calibration.py` | Spectral cross-validation tool for motor-model calibration |

### `enrollment/`

Data-collection infrastructure for the bolted-rig protocol:

| Module | Purpose |
|--------|---------|
| `micro_sweep.py` | Fixed 15-stage protocol (single motors, pairs, combined, stepped ladders) |
| `session_manifest.py` | Per-drone/session attempt tracking; matches uploaded SD files to stages |
| `recorder.py` | Real-time CSV recorder via CRTP telemetry |
| `usd_timeline.py` | Timeline and phase-recorder shared by all SD-deck paths |

### `keyproto/`

| Module | Purpose |
|--------|---------|
| `fe.py` | Robust fuzzy extractor: shielding-offset quantiser + Reed–Solomon RS[40,36] over GF(65537) |
| `onboard.py` | Byte-exact Python reference of the firmware C core |
| `onboard_station.py` | Ground-station side of the on-device protocol |
| `boot.py` | Extracts features from recorded boot sessions; runs the PC-side key exchange |
| `spake2.py` | SPAKE2 on edwards25519 (via PyNaCl) with key confirmation |

### `analysis/`

One script per study; `reproduce.py` runs them all in order. Each script writes its results to a `results_*/` subdirectory and its figures to `analysis/figures/`. See the paper's Evaluation section for the mapping between scripts and figures.

---

## Data format

IMU recordings are CSV files with columns:

```
timestamp_s, phase, acc.x, acc.y, acc.z, gyro.x, gyro.y, gyro.z, motor.m1, motor.m2, motor.m3, motor.m4
```

Sampling rate: ~1 kHz (SD-deck path) or ~100 Hz (CRTP telemetry). The `phase` column labels protocol segments (e.g. `rest_hold`, `m1_hold_58000`). Load a recording with:

```python
from pipeline.preprocessing import load_session
df, fs = load_session("data/fixed_rig/droneA/converted/20261001_120000_droneA_s01_sweep-m1_1khz.csv",
                      phase_filter="m1_hold_58000")
```

---

## Citation

```bibtex
@inproceedings{uavibes2027,
  title     = {{UAVibes}: Tamper-Evident Drone Authentication from {IMU} and Motor Characteristics},
  author    = {Anonymous},
  booktitle = {Proceedings of the 36th USENIX Security Symposium},
  year      = {2027}
}
```

---

## License

The code is released under the MIT License. See `LICENSE`.
